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

from fixtures import REAL, need_real, no_real_data  # noqa: E402

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


def read_rows(shard: Path, n: int | None = None, want=None):
    """(ids, waves) of the shard's first n rows, or of its first n rows whose id is in `want`. Only those rows are
    decoded (a box-size shard holds ~2,000 rows)."""
    import pyarrow.parquet as pq

    from kitsune.audio import decode_audio

    ids = pq.read_table(shard, columns=["id"]).column("id").to_pylist()
    idx = [i for i, x in enumerate(ids) if want is None or x in want][:n]
    audio = pq.read_table(shard, columns=["audio"]).column("audio").take(idx).to_pylist()
    return [ids[i] for i in idx], [decode_audio(a) for a in audio]


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


ROWS_PER_SOURCE = 64


def test_anchor_matches_stored_parakeet_targets(anchor, tools):
    """The anchor on the audio of stored parakeet_out rows: the same frame count and greedy CTC path, and a KD loss
    against its own stored targets that is only storage rounding and the encoder's precision. Bounded for a full box
    parakeet_out: ROWS_PER_SOURCE rows of eval_jsut and of reazon_small, from each source's first shards in name order.

    Measured on these 128 rows. Laptop CPU fp32 shards (scripts/02p): 128/128 paths, KL 2.4e-6 per frame. Label-box
    shards (bf16 encoder on CUDA; labels/full rev 211b767): 127/128 paths, KL 4.8e-3 per frame, but the median
    utterance is at 4e-5 per frame; a few utterances sit at 0.05-0.2 per frame with identical paths. That is the bf16
    encoder's own noise: a CPU bf16 encoder is as far from the box targets (eval_jsut 7.6e-3 per frame for either).
    A misaligned or mis-binned target costs O(1) per frame, so the bounds below still catch those."""
    feats, _ = tools
    need_real(PARAKEET_OUT / "meta.json")
    by_source = {s: sorted(PARAKEET_OUT.glob(f"{s}/*.npz")) for s in ("eval_jsut", "reazon_small")}
    if not any(by_source.values()):
        no_real_data(f"no eval_jsut / reazon_small shards under {PARAKEET_OUT}")
    same = rows = 0
    kl = frames = 0.0
    per_utt = []
    for source, npzs in by_source.items():
        left = ROWS_PER_SOURCE
        for npz in npzs:
            if left <= 0:
                break
            shard = REAL / "data" / "shards" / source / f"{npz.stem}.parquet"
            need_real(shard)
            targets = load_ctc_targets(npz)
            ids, waves = read_rows(shard, left, want=targets)
            left -= len(ids)
            for uid, (lp, n) in zip(ids, run(anchor, feats, waves)):
                t = targets[uid]
                assert n == t.n_frames, uid
                same += CS.greedy_ctc_ids(lp[None], torch.tensor([n]))[0] == t.ctc_ids.tolist()
                rows += 1
                loss = ctc_kd.ctc_kd_losses(lp[None], collate_frame_targets([t]))
                kl += float(loss["kl_dense"] + loss["kl_blank"])
                frames += n
                per_utt.append(float(loss["kl_dense"] + loss["kl_blank"]) / max(n, 1))
    assert rows > 0, f"no stored row of {PARAKEET_OUT} is in the data shards under {REAL / 'data' / 'shards'}"
    med = float(np.median(per_utt))
    print(f"\n{rows} stored rows: greedy path identical on {same}; KD KL {kl / frames:.2e} per frame (median "
          f"utterance {med:.2e}, max {max(per_utt):.2e})")
    assert same >= 0.98 * rows and kl / frames < 2e-2 and med < 1e-3
    assert np.isfinite(kl)


def test_lost_student_overfits_real_stored_targets(anchor, tools):
    """A P-0.05B-shaped pruned student (FFN kept by a random ranking: the mechanics, not the init, are under test)
    learns 8 real JSUT rows from their stored parakeet_out targets with the study's CTC-KD objective on CPU: AdamW
    1e-3, BN frozen at the teacher's stats, clip 1. Starts at the "lost" class (all-blank output) and must reach the
    targets' greedy text. Measured on the label box's eval_jsut targets: objective 25.3 -> 0.088, CER vs the targets
    1.0 -> 0.005 (68 s on 4 threads); the saved P-0.05B on 10 rows went 29.2 -> 0.05 and CER 1.0 -> 0 by step 30."""
    from kitsune.patches import freeze_batchnorm
    from kitsune.text import cer

    feats, tok = tools
    need_real(PARAKEET_OUT / "meta.json")
    npzs = sorted(PARAKEET_OUT.glob("eval_jsut/*.npz"))
    if not npzs:
        no_real_data(f"no eval_jsut shard under {PARAKEET_OUT}")
    shard = REAL / "data" / "shards" / "eval_jsut" / f"{npzs[0].stem}.parquet"
    need_real(shard)
    targets = load_ctc_targets(npzs[0])
    ids, waves = read_rows(shard, 8, want=targets)
    assert len(ids) == 8
    tg = [targets[i] for i in ids]
    want = [CS.decode_ids(tok, t.ctc_ids) for t in tg]
    g = torch.Generator().manual_seed(4)
    imp = {(l, name): torch.rand(4096, generator=g) for l in range(24) for name in S.FFN_NAMES}
    torch.manual_seed(0)
    model = CS.build_ctc_student(anchor, CS.resolve_layers(4), 768, imp)
    freeze_batchnorm(model)
    f, fl = feats(waves)
    mask = CS.lengths_to_mask(fl, f.shape[1])
    batch = collate_frame_targets(tg)

    def evaluate():
        model.eval()
        with torch.no_grad():
            lp, n = CS.ctc_log_probs(model, f, mask)
            losses = ctc_kd.ctc_kd_losses(lp, batch)
            obj = float(ctc_kd.ctc_kd_objective(losses, losses["n_tokens"]))
        hyp = [CS.decode_ids(tok, h) for h in CS.greedy_ctc_ids(lp, n)]
        return obj, float(np.mean([cer(h, w) for h, w in zip(hyp, want)]))

    obj0, cer0 = evaluate()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.0)
    for _ in range(30):
        model.train()
        lp, n = CS.ctc_log_probs(model, f, mask)
        assert n.tolist() == [t.n_frames for t in tg]
        losses = ctc_kd.ctc_kd_losses(lp, batch)
        loss = ctc_kd.ctc_kd_objective(losses, losses["n_tokens"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    obj, err = evaluate()
    print(f"\noverfit 8 real rows, 30 steps: objective {obj0:.3f} -> {obj:.4f}, CER vs targets {cer0:.3f} -> {err:.4f}")
    assert cer0 >= 0.9 and obj < 0.05 * obj0 and err <= 0.05
