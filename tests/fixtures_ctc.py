"""Tiny Parakeet CTC models and synthetic Parakeet frame targets in the repo's exact formats, for the CTC tests.

    from fixtures_ctc import tiny_ctc_model, tiny_parakeet_dir, tiny_student_dir, make_fake_parakeet_out

Everything is built locally and deterministically from a seed; nothing needs the real model, the HF cache or network:
  tiny_tokenizer()           a ParakeetTokenizer with Parakeet's real layout: 3073 ids, <unk> 0, "▁" 2, <blank> 3072
                             (= the pad token), single-character BPE pieces, the Metaspace decoder
  write_processor(dir)       processor_config.json (the real 80-mel ParakeetFeatureExtractor settings) + tokenizer
  tiny_ctc_config/_model     a random ParakeetForCTC with the real vocab, blank, mel count and 8x subsampling but a tiny
                             encoder (d 32, 2 heads, FFN 64): non-zero biases, non-trivial BN stats
  tiny_parakeet_dir(dir)     a look-alike of the converted teacher dir (tools/publish_parakeet.py): a ParakeetForTDT
                             whose encoder IS the returned tiny CTC model's, ctc_head.safetensors as a Linear (V, H),
                             kitsune_model.json, the processor files
  tiny_student_dir(dir)      a saved student (kitsune.ctc_student.save_ctc_student) of a tiny model
  synthetic_log_probs(T)     peaky random CTC log-probs (T, 3073): ~2/3 of the frames blank-certain
  make_fake_parakeet_out(fc) parakeet_out/<source>/<stem>.{npz,jsonl} + meta.json for a fixtures.FakeCorpus, exactly
                             the format of scripts/02p_parakeet_pass.py (kitsune.parakeet_targets.pack_shard /
                             write_shard, check_shard-clean), with n_frames = expected_n_frames(the row's audio) so the
                             student's frames align with the targets; returns the written ground truth per id
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kitsune import ctc_student as CS  # noqa: E402
from kitsune import parakeet_targets as pt  # noqa: E402
from kitsune.ctc_targets import FrameTargets, frame_targets  # noqa: E402

V, BLANK = CS.CTC_VOCAB, CS.CTC_BLANK
N_MEL = 80
SETTINGS = dict(k_tdt=8, k_ctc=8, max_symbols=10, ctc_dense_thr=0.95)
_CHARS = [chr(c) for c in range(0x3041, 0x3097)] + [chr(c) for c in range(0x4E00, 0x4E00 + 3000)]


# ------------------------------------------------------------------------------------------------ tokenizer


def tiny_tokenizer():
    """ParakeetTokenizer over 3073 ids laid out like the real one (single characters instead of Parakeet's pieces)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import ParakeetTokenizer

    pieces = ["<unk>", "。", "▁"] + _CHARS[: V - 4] + ["<blank>"]
    assert len(pieces) == V and len(set(pieces)) == V
    tok = Tokenizer(models.BPE(vocab={p: i for i, p in enumerate(pieces)}, merges=[], unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    return ParakeetTokenizer(tokenizer_object=tok, unk_token="<unk>", pad_token="<blank>")


def write_processor(d, decoder_type: str = "tdt"):
    """processor_config.json (the real ParakeetFeatureExtractor settings), tokenizer.json, tokenizer_config.json."""
    from transformers import ParakeetFeatureExtractor, ParakeetProcessor

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ParakeetProcessor(ParakeetFeatureExtractor(), tiny_tokenizer(), blank_token="<blank>",
                      decoder_type=decoder_type).save_pretrained(d)
    return d


# ------------------------------------------------------------------------------------------------ models


def tiny_encoder_dict(n_layers: int = 4, d: int = 32, heads: int = 2, ffn: int = 64, channels: int = 8,
                      kernel: int = 9) -> dict:
    from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig

    return ParakeetEncoderConfig(hidden_size=d, num_hidden_layers=n_layers, num_attention_heads=heads,
                                 num_key_value_heads=heads, intermediate_size=ffn, num_mel_bins=N_MEL,
                                 subsampling_factor=8, subsampling_conv_channels=channels, conv_kernel_size=kernel,
                                 scale_input=True).to_dict()


def tiny_ctc_config(n_layers: int = 4, ffn: int = 64, **kw):
    return CS.ctc_config(tiny_encoder_dict(n_layers=n_layers, ffn=ffn, **kw), n_layers, ffn)


def tiny_ctc_model(seed: int = 0, n_layers: int = 4, ffn: int = 64, **kw):
    """Random tiny ParakeetForCTC in eval mode (untrained: its frames are near-uniform, not CTC-like)."""
    from transformers import ParakeetForCTC

    torch.manual_seed(seed)
    m = ParakeetForCTC(tiny_ctc_config(n_layers=n_layers, ffn=ffn, **kw)).eval()
    with torch.no_grad():
        for bn in (x for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d)):
            bn.running_mean.normal_(0.0, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
            bn.num_batches_tracked.fill_(1000)
        # HF zero-inits biases; non-zero so bias slices are checked by value
        for mod in m.modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)) and mod.bias is not None:
                mod.bias.normal_(0.0, 0.1)
    return m


def tiny_parakeet_dir(d, seed: int = 0, n_layers: int = 4, ffn: int = 64, **kw):
    """A converted-teacher-dir look-alike. Returns (dir, the ParakeetForCTC whose weights it holds)."""
    from safetensors.torch import save_file
    from transformers import ParakeetForTDT, ParakeetTDTConfig

    d = Path(d)
    ctc = tiny_ctc_model(seed=seed, n_layers=n_layers, ffn=ffn, **kw)
    enc = tiny_encoder_dict(n_layers=n_layers, ffn=ffn, **kw)
    cfg = ParakeetTDTConfig(encoder_config=enc, vocab_size=V, blank_token_id=BLANK, pad_token_id=BLANK,
                            decoder_hidden_size=16, num_decoder_layers=1, durations=[0, 1, 2, 3, 4],
                            max_symbols_per_step=10, hidden_act="relu")
    torch.manual_seed(seed + 1)
    tdt = ParakeetForTDT(cfg).eval()
    tdt.encoder.load_state_dict(ctc.encoder.state_dict(), strict=True)
    tdt.save_pretrained(d)
    w = ctc.ctc_head.weight.detach()[:, :, 0].contiguous()
    save_file({"weight": w, "bias": ctc.ctc_head.bias.detach().contiguous()}, str(d / CS.CTC_HEAD_FILE),
              metadata={"format": "pt"})
    (d / "kitsune_model.json").write_text(json.dumps({"source_repo": "tiny", "tiny": True}), encoding="utf-8")
    write_processor(d)
    return d, ctc


def tiny_student_dir(d, seed: int = 0, n_layers: int = 3, ffn: int = 48, name: str = "tiny", **meta):
    """A student dir as scripts/03c writes one, from a tiny random model (not a pruned teacher)."""
    d = Path(d)
    src = write_processor(d.parent / f"{d.name}_src")
    m = tiny_ctc_model(seed=seed, n_layers=n_layers, ffn=ffn)
    counts = CS.param_counts(m)
    base = dict(name=name, family="ctc", init_class="pruned_kept", params_total=counts["total"],
                params_non_embedding=counts["non_embedding"], closed_form_params=counts["closed_form"], seed=seed,
                bn="teacher", teacher=f"{CS.TEACHER_REPO}@{CS.TEACHER_REVISION}", enc_layers=list(range(n_layers)),
                ffn=ffn)
    base.update(meta)
    CS.save_ctc_student(m, d, src, base)
    return d, m


# ------------------------------------------------------------------------------------------------ frame targets


def synthetic_log_probs(n_frames: int, rng: np.random.Generator, blank_frac: float = 0.65,
                        tokens: tuple[int, int] = (3, 400)) -> torch.Tensor:
    """(T, V) float32 CTC log-probs: blank-certain frames (p_blank ~ 0.99), and dense frames whose argmax is a token
    (sometimes the blank at p < 0.95), with a spread top-k and a small tail."""
    z = torch.from_numpy(rng.normal(0.0, 1.0, size=(n_frames, V)).astype(np.float32))
    dense = rng.random(n_frames) >= blank_frac
    for t in range(n_frames):
        if dense[t]:
            cand = rng.choice(np.arange(*tokens), size=5, replace=False)
            z[t, cand] += torch.from_numpy(rng.uniform(4.0, 11.0, size=5).astype(np.float32))
            z[t, BLANK] += float(rng.uniform(5.0, 11.0))
        else:
            z[t, BLANK] += 14.0
    return torch.log_softmax(z.double(), -1).float()


def synthetic_frame_targets(n_frames: int, rng: np.random.Generator, k: int = 8, **kw) -> FrameTargets:
    """FrameTargets exactly as the label pass would store them for synthetic_log_probs (the same code path)."""
    from kitsune.parakeet import ctc_targets

    lp = synthetic_log_probs(n_frames, rng, **kw)[None]
    t = ctc_targets(lp, torch.tensor([n_frames]), k_ctc=k, dense_thr=SETTINGS["ctc_dense_thr"])
    return frame_targets(n_frames, t["ctc_blank_lp"][0], t["ctc_dense_frame"][0], t["ctc_topk_idx"][0],
                         t["ctc_topk_lp"][0])


def _tdt_path(ft: FrameTargets, rng: np.random.Generator, k: int) -> dict:
    """A valid TDT record (check_shard-clean) whose tokens are the CTC greedy path: one duration-1 step per frame,
    emitting at the first frame of each collapsed CTC run."""
    col0 = ft.col0()
    prev = np.concatenate([[-1], col0[:-1]])
    emit = np.where((col0 != prev) & (col0 != BLANK), col0, BLANK)
    S = len(emit)
    idx = np.zeros((S, k), np.int64)
    for s in range(S):
        others = rng.choice(np.setdiff1d(np.arange(3, 400), [emit[s]]), size=k - 1, replace=False)
        idx[s] = np.concatenate([[emit[s]], others])
    lp = np.sort(np.log(rng.dirichlet(np.ones(k), size=S) * 0.99 + 1e-9), axis=1)[:, ::-1]
    return dict(step_frame=np.arange(S), step_dur=np.ones(S, np.int64), step_forced=np.zeros(S, bool),
                tdt_topk_idx=idx, tdt_topk_lp=lp.astype(np.float32),
                tdt_dur_lp=np.log(rng.dirichlet(np.ones(5), size=S)).astype(np.float32))


@dataclass
class FakeParakeetOut:
    root: Path
    targets: dict[str, FrameTargets] = field(default_factory=dict)  # id -> what was written
    rows: dict[str, dict] = field(default_factory=dict)  # id -> the jsonl row
    n_samples: dict[str, int] = field(default_factory=dict)  # id -> audio samples the frames were computed from


def make_fake_parakeet_out(fc, root=None, *, seed: int = 0, sources=None, k: int = 8) -> FakeParakeetOut:
    """parakeet_out for every teacher-labelled row of a fixtures.FakeCorpus (label-time shard = all rows of the chunk,
    so n_scanned / shard_ids_sha256 describe the shard the label box read). Shards come only for the stems teacher_out
    has, as on the label box, which runs both passes over the same extent. meta.json = build_meta(SETTINGS)."""
    import soundfile as sf

    from kitsune.store import ids_sha256
    from kitsune.text import cer as cer_fn

    root = Path(root) if root else fc.root / "parakeet_out"
    rng = np.random.default_rng(seed)
    tok = tiny_tokenizer()
    out = FakeParakeetOut(root)
    by_shard: dict[tuple[str, str], list] = {}
    for u in fc.utts.values():
        if sources is None or u.source in sources:
            by_shard.setdefault((u.source, u.stem), []).append(u)
    for (source, stem), chunk in sorted(by_shard.items()):
        if not (fc.teacher_out / source / f"{stem}.npz").exists():
            continue  # a data shard the label box never read (extra_shard_rows): no label pass covers it
        utts, rows = [], []
        for u in chunk:
            if not u.has_teacher:
                continue
            n = sf.info(io.BytesIO(u.audio)).frames
            T = CS.expected_n_frames(n)
            ft = synthetic_frame_targets(T, rng, k=k)
            rec = dict(id=u.id, duration=u.duration, n_frames=T, truncated=False,
                       ctc_blank_lp=ft.blank_lp.astype(np.float32), ctc_dense_frame=ft.dense_frame,
                       ctc_topk_idx=ft.topk_idx.astype(np.int64), ctc_topk_lp=ft.topk_lp.astype(np.float32),
                       **_tdt_path(ft, rng, k))
            utts.append(rec)
            ctc_hyp = CS.decode_ids(tok, ft.ctc_ids)
            hyp = CS.decode_ids(tok, pt.utt_tokens(rec))
            rows.append(dict(id=u.id, hyp=hyp, ctc_hyp=ctc_hyp, ref=u.text, cer=round(cer_fn(hyp, u.text), 4),
                             ctc_cer=round(cer_fn(ctc_hyp, u.text), 4), duration=round(float(u.duration), 3),
                             n_tok=int(len(pt.utt_tokens(rec))), n_steps=T, n_frames=T, n_forced=0, truncated=False))
            out.targets[u.id], out.rows[u.id], out.n_samples[u.id] = ft, rows[-1], n
        arrays = pt.pack_shard(utts, settings=SETTINGS, n_scanned=len(chunk),
                               shard_ids_sha=ids_sha256([u.id for u in chunk]))
        pt.write_shard(root / source, stem, arrays, rows)
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(pt.build_meta(SETTINGS), indent=1), encoding="utf-8")
    return out
