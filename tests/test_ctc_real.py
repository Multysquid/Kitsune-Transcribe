"""The real Parakeet 0.6B through kitsune/ctc_student.py on CPU (marked slow; each test skips without the real files).

Reads (never writes) the converted teacher dir (KITSUNE_PARAKEET_DIR, default <real data root>/cache/parakeet-tdt_ctc-
0.6b-ja-hf), the laptop's data shards, and a parakeet_out root (KITSUNE_PARAKEET_OUT, default <real root>/parakeet_out).
To run it from a worktree: KITSUNE_REAL_DATA_ROOT=<main checkout> python -m pytest -m slow tests/test_ctc_real.py"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixtures import REAL, need_real  # noqa: E402

from kitsune import ctc_kd  # noqa: E402
from kitsune import ctc_student as CS  # noqa: E402
from kitsune import student as S  # noqa: E402
from kitsune.ctc_targets import collate_frame_targets, load_ctc_targets  # noqa: E402

MODEL_DIR = Path(os.environ.get("KITSUNE_PARAKEET_DIR") or REAL / "cache" / "parakeet-tdt_ctc-0.6b-ja-hf")
PARAKEET_OUT = Path(os.environ.get("KITSUNE_PARAKEET_OUT") or REAL / "parakeet_out")
JSUT = REAL / "data" / "shards" / "eval_jsut" / "eval-00000.parquet"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def anchor():
    need_real(MODEL_DIR / "model.safetensors", MODEL_DIR / CS.CTC_HEAD_FILE)
    return CS.load_parakeet_ctc(MODEL_DIR)


@pytest.fixture(scope="module")
def tools():
    from transformers import AutoTokenizer

    return CS.ctc_features(MODEL_DIR), AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)


def read_rows(shard: Path, n: int | None = None):
    import pyarrow.parquet as pq

    from kitsune.audio import decode_audio

    t = pq.read_table(shard, columns=["id", "audio"])
    t = t.slice(0, n) if n else t
    return t.column("id").to_pylist(), [decode_audio(a) for a in t.column("audio").to_pylist()]


def run(model, feats, waves, batch=8):
    out = []
    with torch.inference_mode():
        for i in range(0, len(waves), batch):
            f, fl = feats(waves[i:i + batch])
            lp, n = CS.ctc_log_probs(model, f, CS.lengths_to_mask(fl, f.shape[1]))
            out += [(lp[b, :int(n[b])].clone(), int(n[b])) for b in range(lp.shape[0])]
    return out


def test_anchor_reproduces_the_golden_ctc_transcripts(anchor, tools):
    """The unpruned build of the CTC path gives kitsune/parakeet_golden.json's 32 CTC transcripts (made by
    kitsune.parakeet.ParakeetTeacher on CPU), and build_ctc_student(all, 4096) is the anchor tensor for tensor."""
    need_real(JSUT)
    feats, tok = tools
    golden = json.loads((ROOT / "kitsune" / "parakeet_golden.json").read_text(encoding="utf-8"))
    ids, waves = read_rows(JSUT, len(golden["ids"]))
    assert ids == golden["ids"]
    outs = run(anchor, feats, waves)
    got = [CS.decode_ids(tok, CS.greedy_ctc_ids(lp[None], torch.tensor([n]))[0]) for lp, n in outs]
    assert got == golden["ctc"]
    assert [n for _, n in outs] == [CS.expected_n_frames(len(w)) for w in waves]
    st = CS.build_ctc_student(anchor, list(range(24)), 4096, None)
    assert all(torch.equal(a, b) for a, b in zip(st.state_dict().values(), anchor.state_dict().values()))
    again = run(st, feats, waves[:8])
    assert max(float((a - b).abs().max()) for (a, _), (b, _) in zip(again, outs[:8])) == 0.0


@pytest.mark.parametrize("n, ffn, total", [(16, 2560, 308_524_033), (8, 768, 98_468_865), (4, 768, 52_190_209)])
def test_real_student_shapes(anchor, n, ffn, total):
    layers = CS.resolve_layers(n)
    g = torch.Generator().manual_seed(n)
    imp = {(l, name): torch.rand(4096, generator=g) for l in range(24) for name in S.FFN_NAMES}
    st = CS.build_ctc_student(anchor, layers, ffn, imp)
    c = CS.param_counts(st)
    assert c["total"] == c["closed_form"] == CS.closed_form_ctc_params(n, ffn) == total
    keep = S.select_ffn_neurons(imp, layers, ffn, 4096)
    for i in (0, len(layers) - 1):
        tl = layers[i]
        for name in S.FFN_NAMES:
            s_ff, t_ff = getattr(st.encoder.layers[i], name), getattr(anchor.encoder.layers[tl], name)
            idx = keep[(tl, name)]
            assert torch.equal(s_ff.linear1.weight, t_ff.linear1.weight[idx])
            assert torch.equal(s_ff.linear1.bias, t_ff.linear1.bias[idx])
            assert torch.equal(s_ff.linear2.weight, t_ff.linear2.weight[:, idx])
        s_bn, t_bn = st.encoder.layers[i].conv.norm, anchor.encoder.layers[tl].conv.norm
        assert torch.equal(s_bn.running_mean, t_bn.running_mean) and torch.equal(s_bn.running_var, t_bn.running_var)
    assert layers[-1] == 23 and torch.equal(st.ctc_head.weight, anchor.ctc_head.weight)


def test_anchor_matches_stored_parakeet_targets(anchor, tools):
    """The anchor on the audio of stored parakeet_out rows: the same frame count and greedy CTC path, and a KD loss
    against its own stored targets that is only the fp16 rounding plus the rest bucket. (Targets from the box ran a
    bf16 encoder on CUDA, so a few greedy paths may differ there; on laptop CPU fp32 shards they are identical.)"""
    feats, _ = tools
    npzs = sorted(PARAKEET_OUT.glob("eval_jsut/*.npz")) + sorted(PARAKEET_OUT.glob("reazon_small/*.npz"))
    shards = [REAL / "data" / "shards" / p.parent.name / f"{p.stem}.parquet" for p in npzs]
    need_real(PARAKEET_OUT / "meta.json", *shards)
    if not npzs:
        pytest.skip(f"no eval_jsut / reazon_small shards under {PARAKEET_OUT}")
    same = rows = 0
    kl = frames = 0.0
    for npz in npzs:
        targets = load_ctc_targets(npz)
        ids, waves = read_rows(REAL / "data" / "shards" / npz.parent.name / f"{npz.stem}.parquet", None)
        keep = [i for i, x in enumerate(ids) if x in targets][:64]
        outs = run(anchor, feats, [waves[i] for i in keep])
        for i, (lp, n) in zip(keep, outs):
            t = targets[ids[i]]
            assert n == t.n_frames, ids[i]
            same += CS.greedy_ctc_ids(lp[None], torch.tensor([n]))[0] == t.ctc_ids.tolist()
            rows += 1
            loss = ctc_kd.ctc_kd_losses(lp[None], collate_frame_targets([t]))
            kl += float(loss["kl_dense"] + loss["kl_blank"])
            frames += n
    print(f"\n{rows} stored rows: greedy path identical on {same}; KD KL {kl / frames:.2e} per frame")
    assert same >= 0.98 * rows and kl / frames < 2e-3
    assert np.isfinite(kl)
