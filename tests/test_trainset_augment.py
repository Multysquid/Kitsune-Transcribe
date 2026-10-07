"""The CTC family's train-data augmentation (kitsune.trainset's "augmentation" section; scripts/04_distill.py augment.*,
the module docstring's "Train-data augmentation"):

- off, a frame store's micro-batches are what they were before it existed (rebuilt here from the store's primitives),
  with no `aug`; on with every probability 0, the same tensors plus zero counts
- the pieces on hand-made targets: the greedy path without the Python loop, a join that keeps each piece's targets
  (sentence marks included) at its frame offset and merges a token repeated across the join as CTC reads it, a cut
  that is a prefix and loses the final token, the cut's interval, the group size, the frame-aligned audio join and the
  mix at the drawn SNR
- on a real frame store: every row the loss reads has ctc_frames(samples) == n_frames; truncate keeps the frames
  before its cut and removes the final token; concat puts each piece at its offset, its group size divides the rows and
  its padded frames never exceed the micro-batch's; mix keeps targets and lengths and hits the SNR; the same index list
  gives the same batch in any process, another seed another
- the trainer: augment.* validated (an AED student's rules: tests/test_aed_augment.py), changeable on resume; a tiny
  CTC run with all three augmentations
  through its smoke phase logs the aug/* shares and the clamp, and a crash resumed from a mid state ends with the
  uninterrupted run's weights
CPU only, tiny models, synthetic data in the real on-disk formats (tests/fixtures.py, tests/fixtures_ctc.py)."""
import copy
import dataclasses
import json
import math
import os
import pickle
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import load_script, make_fake_corpus, make_fake_selection, make_noise_bank, make_rir_bank  # noqa: E402
from fixtures_ctc import make_fake_parakeet_out, tiny_student_dir  # noqa: E402
from kitsune import trainset as T  # noqa: E402
from kitsune.ctc_targets import FrameTargets, collate_frame_targets  # noqa: E402
from kitsune.parakeet_targets import ctc_greedy  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
BLANK, MARK, K = 3072, 1, 8  # MARK: "。" in the tiny tokenizer (tests/fixtures_ctc.py)
FS = T.FRAME_SAMPLES
FRAME_KEYS = ("frame_mask", "dense_mask", "blank_lp", "topk_idx", "topk_lp", "ctc_targets", "ctc_target_lengths",
              "n_frames", "n_tok")


@pytest.fixture(autouse=True)
def grad_enabled():
    """Autograd on, as in a trainer process (an earlier test file can leave it off for the whole pytest process)."""
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(prev)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A synthetic corpus (two train sources of 0.4-2.5 s rows, three eval sets), its selection and parakeet_out
    (frames aligned to the audio), the train frame store, and a tiny saved CTC student with the trainer's base config
    (family ctc, steps clock, micro-batches of ~3 s, no evals or checkpoints in the loop unless a test asks)."""
    root = tmp_path_factory.mktemp("aug")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (40, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=31)
    sel = make_fake_selection(fc, greedy_n=4, probe_n=5)
    po = make_fake_parakeet_out(fc, seed=9)
    store = T.build_frame_stores(sel, fc.data, po.root, root / "cache" / "ctc_train", ["src_a", "src_b"], ["train"],
                                 log=lambda s: None)
    sdir, _ = tiny_student_dir(root / "student", seed=3, n_layers=2, ffn=48, name="tiny-p")
    base = {
        "family": "ctc", "parakeet_root": str(po.root),
        "student": str(sdir), "data_root": str(fc.data), "teacher_root": str(fc.teacher_out),
        "second_root": str(fc.second_out), "selection": str(sel), "cache_dir": str(root / "cache"),
        "runs_root": str(root / "runs"), "sources": ["src_a", "src_b"], "eval_sets": EVAL,
        "device": "cpu", "autocast": "none",
        "optim": {"lr": 3e-3},
        "loss": {"l2sp_lambda": 0.0},
        "schedule": {"warmup_steps": 3, "cooldown_frac": 0.2, "clock": "steps", "max_steps": 6},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_min": None, "every_steps": None, "greedy_subset": 3, "batch_s": 20, "check_baselines": False},
        "ckpt": {"weights_every_min": None, "full_local_every_min": None, "keep_local": 5, "full_after_smoke": False,
                 "upload_full_at": []},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2, "capture_env": False},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
    }
    return dict(root=root, fc=fc, po=po, sel=sel, store=store, base=base)


def micro_batches(store, micro_s: float = 4.0, seed: int = 0) -> list[list[int]]:
    """Every micro-batch of epoch 0 of the frame planner over the store: the duration-homogeneous shapes training
    meets."""
    p = T.StepPlanner(store.utts, step_audio_s=8, micro_audio_s=micro_s, max_dec_len=None, pool_micro=4, seed=seed)
    return [mb for step in p.epoch_plan(0) for mb in step]


def row_ids(mb: dict, r: int) -> list[int]:
    """Row r's CTC target ids out of the flat ctc_targets."""
    n = mb["ctc_target_lengths"].tolist()
    s = sum(n[:r])
    return mb["ctc_targets"][s:s + n[r]].tolist()


def ft(col0: list[int], seed: int = 0) -> FrameTargets:
    """FrameTargets whose argmax is `col0` (blank-only frames off the dense set), with a plausible top-k."""
    rng = np.random.default_rng(seed)
    col0 = np.asarray(col0, dtype=np.int64)
    dense = np.flatnonzero(col0 != BLANK).astype(np.int32)
    idx = np.zeros((len(dense), K), np.int16)
    for j, d in enumerate(dense):
        idx[j] = np.concatenate([[col0[d]], rng.choice(np.arange(3, 300), size=K - 1, replace=False)])
    lp = np.sort(np.log(rng.dirichlet(np.ones(K), size=len(dense)) * 0.99 + 1e-6), axis=1)[:, ::-1].astype(np.float16)
    blank_lp = np.where(col0 == BLANK, -0.01, -4.0).astype(np.float16)
    return FrameTargets(len(col0), blank_lp, dense, idx, lp, np.asarray(ctc_greedy(col0), dtype=np.int32))


def reference_batch(store, idx: list[int]) -> dict:
    """A frame store's micro-batch as FrameBatchDataset built it before the augmentation existed, from the store's own
    primitives: decoded audio zero-padded, the stored durations / agree, the store indices, ids and sources, and
    collate_frame_targets of the stored targets."""
    waves = [store.wave(i) for i in idx]
    lengths = np.array([len(w) for w in waves], dtype=np.int64)
    wave = np.zeros((len(idx), int(lengths.max())), dtype=np.float32)
    for b, w in enumerate(waves):
        wave[b, :len(w)] = w
    out = dict(wave=torch.from_numpy(wave), lengths=torch.from_numpy(lengths),
               durations=torch.tensor([store.utts[i].duration for i in idx], dtype=torch.float32),
               agree=torch.tensor([store.utts[i].agree for i in idx], dtype=torch.float32),
               index=torch.tensor(idx, dtype=torch.int64), ids=[store.utts[i].id for i in idx],
               sources=[store.utts[i].source for i in idx], dropped=[])
    out.update(collate_frame_targets([store.targets(i) for i in idx]))
    out["n_tok"] = out["ctc_target_lengths"].clone()
    return out


def assert_same(a: dict, b: dict, keys=None):
    for k in keys or b:
        if isinstance(b[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
            assert torch.equal(a[k], b[k]) or (a[k].is_floating_point() and torch.equal(a[k].isnan(), b[k].isnan())
                                               and torch.equal(a[k].nan_to_num(), b[k].nan_to_num())), k
        else:
            assert a[k] == b[k], k


# ---------------------------------------------------------------------------------------------------- off is off


def test_off_is_the_plain_dataset(env):
    """Without an augmentation a frame store's micro-batches are exactly what they were before it existed (every key,
    dtype and value, rebuilt from the store's primitives) and carry no `aug`; with an Augment whose probabilities are
    all 0 they are the same tensors plus zero counts."""
    store = env["store"]
    plain = T.dataset_for(store)
    assert isinstance(plain, T.FrameBatchDataset) and plain.augment is None
    zero = T.FrameBatchDataset(store, augment=T.Augment(seed=5))
    for idx in micro_batches(store)[:12]:
        got, want = plain[idx], reference_batch(store, idx)
        assert set(got) == set(want) and "aug" not in got
        assert_same(got, want)
        z = zero[idx]
        assert set(z) == set(want) | {"aug"}
        assert_same(z, want)
        assert z["aug"] == dict(utts=len(idx), rows=len(idx), concat_groups=0, concat_utts=0, truncated=0, mixed=0,
                                cut_padded=0, end_padded=0, end_trimmed=0, speech_mixed=0, reverbed=0, noised=0,
                                gained=0, clipped=0, coded=0)


def test_augment_spec_and_where_it_applies(env, tmp_path):
    """Augment refuses settings outside its ranges; dataset_for takes a CTC augmentation for a frame store only (a
    token store takes an AED one: tests/test_aed_augment.py) and a frame store no AED one; with_augment shares the plain
    dataset's arrays."""
    for bad in (dict(truncate_p=1.5), dict(concat_p=-0.1), dict(mix_p=2), dict(truncate_min_frac=1.0),
                dict(truncate_min_s=-1), dict(concat_max_s=0), dict(concat_max_n=1), dict(mix_snr_db=(10, 5)),
                dict(mix_snr_db=(-3, 5)), dict(seed=-1), dict(truncate_pause_p=1.5),
                dict(truncate_p=0.5),  # a cut needs the punctuation ids: without them a mark looks like a word
                dict(truncate_p=0.5, punct_ids=(BLANK,)), dict(truncate_pad_p=1.5), dict(end_pad_p=-0.1),
                dict(pad_frames=(0, 5)), dict(pad_frames=(3, 2)), dict(pad_frames=(1, T.PAD_MAX_FRAMES + 1)),
                dict(pad_frames=(1.5, 2)), dict(pad_frames=(1,)), dict(truncate_min_row_s=-1)):
        with pytest.raises(ValueError, match="not an augmentation"):
            T.Augment(**{"seed": 1, **bad})
    with pytest.raises(TypeError):
        T.FrameBatchDataset(env["store"], augment={"truncate_p": 1.0})
    a = T.Augment.from_config(dict(enabled=True, seed=None, truncate_p=0.5, concat_max_s=28.0, mix_snr_db=[5, 20],
                                   concat_max_n=3, truncate_pause_p=0.25), seed=11, concat_max_s=2.5,
                              punct_ids=[8, MARK])
    assert a == T.Augment(seed=11, truncate_p=0.5, concat_max_s=2.5, concat_max_n=3, mix_snr_db=(5.0, 20.0),
                          truncate_pause_p=0.25, punct_ids=(MARK, 8))
    plain = T.dataset_for(env["store"])
    aug = plain.with_augment(a)
    assert aug.augment == a and plain.augment is None and aug.ids is plain.ids and aug.n_frames is plain.n_frames
    fc = env["fc"]
    tok = T.build_stores(env["sel"], fc.data, fc.teacher_out, tmp_path / "tok", ["src_b"], ["train"],
                         log=lambda s: None)
    with pytest.raises(ValueError, match="CTC augmentation"):
        T.dataset_for(tok, augment=a)
    with pytest.raises(ValueError, match="on a frame store"):
        T.dataset_for(env["store"], augment=T.Augment(seed=1, concat_p=0.5, max_tokens=191))


# ---------------------------------------------------------------------------------------------------- the pieces


def test_greedy_ids_is_ctc_greedy():
    rng = np.random.default_rng(0)
    cases = [[], [BLANK] * 5, [7, 7, 7], [7, BLANK, 7], [BLANK, 3, 3, BLANK, 4, 4, 4, 3]]
    cases += [rng.choice([BLANK, BLANK, 5, 6, 7], size=int(rng.integers(1, 60))).tolist() for _ in range(200)]
    for c in cases:
        got = T.greedy_ids(np.asarray(c))
        assert got.dtype == np.int32 and got.tolist() == ctc_greedy(c), c


def test_join_keeps_each_piece_at_its_offset():
    """A joined row's targets: each piece's blank_lp, dense frames (shifted by its offset) and top-k, so a sentence mark
    at the end of one piece now sits inside the row, at that piece's offset; the CTC path is the greedy path of the
    joined argmax - the pieces' paths one after the other, except that a token ending one piece and starting the next
    with no blank between is one run, which CTC reads (and the target says) once."""
    a = ft([BLANK, 40, BLANK, BLANK, 41, MARK, BLANK], seed=1)  # "... 。" then trailing blank
    b = ft([BLANK, 50, 50, BLANK, 51, BLANK, MARK, BLANK, BLANK], seed=2)
    j = T.join_frame_targets([a, b])
    assert j.n_frames == 16 and j.col0().tolist() == a.col0().tolist() + b.col0().tolist()
    assert j.ctc_ids.tolist() == a.ctc_ids.tolist() + b.ctc_ids.tolist() == [40, 41, MARK, 50, 51, MARK]
    assert int(np.flatnonzero(j.col0() == MARK)[0]) == 5 and j.ctc_ids.tolist().index(MARK) == 2  # the inner mark
    assert np.array_equal(j.blank_lp, np.concatenate([a.blank_lp, b.blank_lp]))
    assert np.array_equal(j.dense_frame, np.concatenate([a.dense_frame, b.dense_frame + 7]))
    assert np.array_equal(j.topk_idx, np.concatenate([a.topk_idx, b.topk_idx])) and j.dense_frame.dtype == np.int32
    assert np.array_equal(j.topk_lp, np.concatenate([a.topk_lp, b.topk_lp]))
    # a token at a's last frame and b's first frame: one run in the joined row
    c, d = ft([BLANK, 60, 61], seed=3), ft([61, BLANK, 62], seed=4)
    jj = T.join_frame_targets([c, d, a])
    assert c.ctc_ids.tolist() + d.ctc_ids.tolist() == [60, 61, 61, 62]
    assert jj.ctc_ids.tolist() == ctc_greedy(jj.col0()) == [60, 61, 62, 40, 41, MARK]
    one = T.join_frame_targets([a])
    assert one.n_frames == a.n_frames and np.array_equal(one.ctc_ids, a.ctc_ids)


def test_cut_and_its_interval():
    """cut_frame_targets keeps the first c frames (a prefix of the CTC path). truncate_frames allows exactly the c >=
    max(ceil(min_frac x T), frames of min_s, 1) whose first removed token (the first one STARTING at or after c) is a
    word: never the mark (a cut between a sentence's last word and its mark would keep a complete sentence without its
    mark), never nothing; truncate_cut draws among them, pause_p of the time among those inside a >= 4-frame pause;
    None when there are none."""
    # 30 frames: tokens at 3, 8-9 (one run), 16, and the mark's run at 20-22; pauses (blank runs >= 4) at 4-7, 10-15,
    # 23-29 - the one at 17-19, between the last word and the mark, is 3 frames and is excluded anyway
    col0 = [BLANK] * 3 + [40] + [BLANK] * 4 + [41, 41] + [BLANK] * 6 + [42] + [BLANK] * 3 + [MARK] * 3 + [BLANK] * 7
    t = ft(col0, seed=5)
    assert t.n_frames == 30
    nxt = T.next_token_start(t.col0())
    assert nxt.tolist() == [40] * 4 + [41] * 5 + [42] * 8 + [MARK] * 4 + [BLANK] * 9  # frame 9: 41's run began at 8
    assert np.flatnonzero(T.pause_frames(t.col0())).tolist() == [4, 5, 6, 7, *range(10, 16), *range(23, 30)]
    for c in (1, 4, 9, 17, 20, 23, 30):
        x = T.cut_frame_targets(t, c)
        assert x.n_frames == c and np.array_equal(x.col0(), t.col0()[:c]) and np.array_equal(x.blank_lp, t.blank_lp[:c])
        assert np.array_equal(x.dense_frame, t.dense_frame[t.dense_frame < c])
        assert np.array_equal(x.topk_idx, t.topk_idx[t.dense_frame < c]) and x.ctc_ids.tolist() == ctc_greedy(col0[:c])
        assert x.ctc_ids.tolist() == t.ctc_ids.tolist()[:len(x.ctc_ids)]
    P = (MARK,)
    assert T.truncate_frames(t, 0.3, 0.0, P).tolist() == list(range(9, 17))  # 17-20 would remove only the mark
    assert T.truncate_frames(t, 0.0, 0.0, ()).tolist() == list(range(1, 21))  # without punct_ids the mark is a "word"
    rng = np.random.default_rng(0)
    got = {T.truncate_cut(t, rng, 0.3, 0.0, P) for _ in range(400)}
    assert got == set(range(9, 17))
    for c in got:  # the kept part is never a complete sentence: the last word (42) and the mark both go
        ids = T.cut_frame_targets(t, c).ctc_ids.tolist()
        assert MARK not in ids and 42 not in ids and ids == t.ctc_ids.tolist()[:len(ids)]
    assert {T.truncate_cut(t, rng, 0.3, 0.0, P, pause_p=1.0) for _ in range(300)} == set(range(10, 16))
    mixed = [T.truncate_cut(t, rng, 0.3, 0.0, P, pause_p=0.5) for _ in range(2000)]
    # half from the pause frames, half uniform over all 8 (6 of them in a pause): 0.5 + 0.5 x 6/8 = 0.875
    assert set(mixed) == set(range(9, 17)) and 0.83 < np.mean([c in range(10, 16) for c in mixed]) < 0.92
    assert {T.truncate_cut(t, rng, 0.0, 1.0, P) for _ in range(200)} == set(range(13, 17))  # 1 s = 12.5 -> 13 frames
    assert T.truncate_cut(t, rng, 0.9, 0.0, P) is None  # lo 27: no word starts that late
    assert T.truncate_cut(ft([BLANK] * 30), rng, 0.0, 0.0, P) is None
    assert T.truncate_cut(ft([40] + [BLANK] * 20), rng, 0.0, 0.0, P) is None  # the last word starts at frame 0
    # a joined row of two sentences (the second piece at frame 10): never in the first one's pause before its mark
    # (4-8: the first sentence would stay complete without its mark), but after its mark (9-11, the first sentence
    # complete WITH its mark, the second one cut) and inside the second one, before its last word (12-14)
    a = ft([BLANK, 50, BLANK, 51, BLANK, BLANK, BLANK, BLANK, MARK, BLANK], seed=6)
    b = ft([BLANK, 52, BLANK, BLANK, 53, BLANK, MARK], seed=7)
    j = T.join_frame_targets([a, b])
    assert T.truncate_frames(j, 0.0, 0.0, P).tolist() == [1, 2, 3, 9, 10, 11, 12, 13, 14]
    for c in (9, 10, 11):
        assert T.cut_frame_targets(j, c).ctc_ids.tolist() == [50, 51, MARK]


def test_concat_group_size():
    """k divides the rows and k x the longest row (in whole frames, 80 ms each) stays within concat_max_s; uniform
    among those; 0 when none qualifies."""
    a = T.Augment(seed=0, concat_max_s=28.0, concat_max_n=4)
    rng = np.random.default_rng(0)
    assert {T.concat_k(12, 50, a, rng) for _ in range(200)} == {2, 3, 4}  # 4 x 4.0 s <= 28
    assert {T.concat_k(12, 100, a, rng) for _ in range(100)} == {2, 3}  # 8 s rows: 3 x 8 = 24, 4 x 8 = 32
    assert {T.concat_k(10, 50, a, rng) for _ in range(100)} == {2}  # 3 and 4 do not divide 10
    assert T.concat_k(7, 10, a, rng) == 0 and T.concat_k(2, 200, a, rng) == 0 and T.concat_k(1, 5, a, rng) == 0
    assert T.concat_k(2, 175, a, rng) == 2  # exactly 2 x 14.0 s = 28 s
    assert {T.concat_k(12, 10, T.Augment(seed=0, concat_max_n=6), rng) for _ in range(300)} == {2, 3, 4, 6}


def test_join_waves_aligns_every_piece_to_its_frames():
    """Every piece but the last padded with ~1e-5 noise (never zeros) or trimmed by < 160 samples to exactly 1280 x its
    frames; the last as it is; the joined length gives the summed frames."""
    rng = np.random.default_rng(1)
    # pads of 640, 1080 (the largest: 1120 - a remainder 40), 1 sample; trims of 1 and 159 samples; then the last
    lens = [16000, 13000, 1281, 25599, 25759, 7000]
    frames = [T.ctc_frames(n) for n in lens]
    assert frames == [13, 11, 1, 20, 20, 6]
    waves = [np.full(n, 0.1 * (j + 1), np.float32) for j, n in enumerate(lens)]
    j = T.join_waves(waves, frames, rng)
    assert j.dtype == np.float32 and T.ctc_frames(len(j)) == sum(frames) and len(j) == FS * sum(frames[:-1]) + lens[-1]
    off = 0
    for p, (n, t) in enumerate(zip(lens, frames)):
        span = j[off: off + (FS * t if p < len(lens) - 1 else n)]
        keep = min(n, len(span))
        assert np.all(span[:keep] == np.float32(0.1 * (p + 1))), p
        pad = span[keep:]
        assert len(pad) == {0: 640, 1: 1080, 3: 1}.get(p, 0), p
        if len(pad):
            assert np.abs(pad).max() < 1e-4 and np.count_nonzero(pad) == len(pad), p
        if len(pad) > 100:
            assert 0.5e-5 < pad.std() < 2e-5, p
        off += len(span)
    for n_last in (1, 159, 160, 1279, 1280, 1281, 5000):  # the last piece's own length gives its own frames
        jj = T.join_waves([waves[0], np.zeros(n_last, np.float32)], [13, T.ctc_frames(n_last)], rng)
        assert T.ctc_frames(len(jj)) == 13 + T.ctc_frames(n_last)


def test_mix_into_hits_the_snr_and_never_writes_its_inputs():
    """mix_into adds a segment of src (a share in [0.5, 1] of the shorter row) at an offset in dst, scaled so that dst's
    power over the span over the segment's is the drawn SNR; dst and src are left as they were; a silent span on
    either side is skipped (no level to scale to, and scaling a silent segment up would only make its noise loud)."""
    rng = np.random.default_rng(3)
    t = np.arange(48000) / 16000
    dst = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    src = (0.05 * np.sin(2 * np.pi * 330 * t[:30000]) + 0.01 * rng.standard_normal(30000)).astype(np.float32)
    d0, s0 = dst.copy(), src.copy()
    for snr_db in ((10.0, 10.0), (5.0, 20.0)):
        for _ in range(20):
            out, info = T.mix_into(dst, src, rng, snr_db)
            assert np.array_equal(dst, d0) and np.array_equal(src, s0) and out.shape == dst.shape
            n, off, st = info["n"], info["offset"], info["src_start"]
            assert 15000 <= n <= 30000 and 0 <= off <= 48000 - n and 0 <= st <= 30000 - n
            diff = out.astype(np.float64) - dst
            assert not diff[:off].any() and not diff[off + n:].any()
            np.testing.assert_allclose(diff[off:off + n], info["gain"] * src[st:st + n], rtol=1e-5, atol=1e-7)
            got = 10 * np.log10(np.mean(dst[off:off + n].astype(np.float64) ** 2) / np.mean(diff[off:off + n] ** 2))
            assert snr_db[0] - 1e-3 <= info["snr_db"] <= snr_db[1] + 1e-3 and abs(got - info["snr_db"]) < 1e-3
    assert T.mix_into(np.zeros(16000, np.float32), src, rng, (10.0, 10.0)) is None  # a silent row
    assert T.mix_into(dst, np.full(16000, 1e-6, np.float32), rng, (10.0, 10.0)) is None  # a silent interferer
    assert T.mix_into(dst, src[:1], rng, (10.0, 10.0)) is None


def test_a_quiet_pad_after_a_cut_and_after_a_whole_row():
    """pad_quiet on hand-made rows: after a cut the row keeps its first c frames' audio and targets, then n frames of
    quiet whose targets are pure blank at the row's most confident blank (at least PAD_MIN_BLANK_LP), and its CTC path
    is still the cut's - no mark; after a whole row (audio not a whole number of frames: squared to them first, with
    low noise) the final mark stays, the quiet after it. ctc_frames of the audio is the row's frames either way."""
    rng = np.random.default_rng(0)
    col0 = [BLANK, 40, BLANK, BLANK, 41, BLANK, 42, BLANK, MARK]  # a sentence ending in its mark
    t, T0 = ft(col0, seed=5), len(col0)
    wave = (0.1 * rng.standard_normal(FS * T0 - 37)).astype(np.float32)  # 37 samples short of whole frames
    assert T.ctc_frames(len(wave)) == T0
    r = T._AugRow(wave, t, [0], [0])
    r.cut_at(5)
    r.pad_quiet(3, wave, np.random.default_rng(1))
    assert (r.cut, r.pad, r.ft.n_frames, T.ctc_frames(len(r.wave))) == (5, 3, 8, 8)
    assert FS * 8 - T.END_BELOW <= len(r.wave) <= FS * 8  # the padded end drawn, as a cut's
    assert np.array_equal(r.wave[:FS * 5], wave[:FS * 5])
    assert r.ft.col0().tolist() == col0[:5] + [BLANK] * 3 and r.ft.ctc_ids.tolist() == [40, 41]  # no mark
    b = np.float16(max(float(t.blank_lp[:5].max()), T.PAD_MIN_BLANK_LP))
    assert (r.ft.blank_lp[5:] == b).all() and np.array_equal(r.ft.blank_lp[:5], t.blank_lp[:5])
    assert (r.ft.dense_frame < 5).all() and r.ft.topk_idx.shape[1] == t.k
    w = T._AugRow(wave, t, [0], [0])
    w.pad_quiet(4, wave, np.random.default_rng(2))
    assert (w.cut, w.pad, w.ft.n_frames, T.ctc_frames(len(w.wave))) == (None, 4, T0 + 4, T0 + 4)
    assert FS * (T0 + 4) - T.END_BELOW <= len(w.wave) <= FS * (T0 + 4)
    assert np.array_equal(w.wave[:len(wave)], wave) and np.abs(w.wave[len(wave):FS * T0]).max() < 1e-4
    assert w.ft.col0().tolist() == col0 + [BLANK] * 4 and w.ft.ctc_ids.tolist() == [40, 41, 42, MARK]  # mark kept
    assert np.array_equal(w.ft.blank_lp[:T0], t.blank_lp) and (w.ft.blank_lp[T0:] == b).all()


def test_quiet_pad_is_room_tone_or_near_silence(monkeypatch):
    """quiet_pad: on the room-tone path every 1280-sample block of the pad is one of the source's whole frames among
    its PAD_ROOM_FRAMES quietest below PAD_ROOM_MAX_POWER (a voice-detector chunk's own silence); a source without
    such a frame, and the other path, give AUG_PAD_STD noise (an app's zero pad, without exact zeros)."""
    rng = np.random.default_rng(3)
    loud = (0.1 * rng.standard_normal(FS * 20)).astype(np.float32)  # mean square 1e-2: no room tone in it
    quiet = (0.003 * rng.standard_normal(FS * 3)).astype(np.float32)  # 9e-6, below 1e-4
    src = np.concatenate([loud[:FS * 10], quiet, loud[FS * 10:]])
    monkeypatch.setattr(T, "PAD_ROOM_P", 1.0)
    pad = T.quiet_pad(src, 5, np.random.default_rng(4))
    blocks = [src[FS * j:FS * (j + 1)] for j in (10, 11, 12)]
    assert pad.shape == (FS * 5,) and pad.dtype == np.float32
    assert all(any(np.array_equal(pad[FS * i:FS * (i + 1)], q) for q in blocks) for i in range(5))
    noise = T.quiet_pad(loud, 4, np.random.default_rng(5))
    assert noise.shape == (FS * 4,) and np.std(noise) == pytest.approx(T.AUG_PAD_STD, rel=0.1)
    monkeypatch.setattr(T, "PAD_ROOM_P", 0.0)
    assert np.std(T.quiet_pad(src, 2, np.random.default_rng(6))) == pytest.approx(T.AUG_PAD_STD, rel=0.1)


def test_the_stream_is_a_function_of_seed_and_index_list():
    a, b = T.augment_rng(3, [5, 1, 9]), T.augment_rng(3, np.array([5, 1, 9]))
    assert np.array_equal(a.random(8), b.random(8))
    assert not np.array_equal(T.augment_rng(3, [5, 1, 9]).random(8), T.augment_rng(4, [5, 1, 9]).random(8))
    assert not np.array_equal(T.augment_rng(3, [5, 1, 9]).random(8), T.augment_rng(3, [1, 5, 9]).random(8))


# ------------------------------------------------------------------------------------------- on a real frame store


def test_every_row_keeps_its_frames(env):
    """With all three augmentations on, over every micro-batch of an epoch: each row's audio gives exactly its target
    frames (ctc_frames(samples) == n_frames), its padding is zero, its CTC targets are the greedy path of its argmax,
    each joined row's ids are store ids and the counts add up."""
    store = env["store"]
    pos = {u.id: i for i, u in enumerate(store.utts)}
    ds = T.FrameBatchDataset(store, augment=T.Augment(seed=1, truncate_p=0.5, truncate_min_s=0.3, concat_p=0.7,
                                                       concat_max_s=2.5, mix_p=0.5, punct_ids=(MARK,),
                                                       truncate_pad_p=0.5, end_pad_p=0.5))
    seen = dict(truncated=0, mixed=0, concat_groups=0, rows=0, utts=0, concat_utts=0, cut_padded=0, end_padded=0)
    for idx in micro_batches(store, micro_s=4.0):
        mb = ds[idx]
        a = mb["aug"]
        for k in seen:
            seen[k] += a[k]
        B = len(mb["ids"])
        assert a["rows"] == B == mb["wave"].shape[0] and a["utts"] == len(idx) and not mb["dropped"]
        pieces = [x.split("+") for x in mb["ids"]]
        assert all(p in pos for ps in pieces for p in ps) and sum(len(p) for p in pieces) <= len(idx)
        if a["concat_groups"]:
            assert a["concat_groups"] == B and len(idx) % B == 0 and a["concat_utts"] == len(idx)
        for r in range(B):
            n, t = int(mb["lengths"][r]), int(mb["n_frames"][r])
            assert T.ctc_frames(n) == t == int(mb["frame_mask"][r].sum()), (idx, r)
            assert not mb["wave"][r, n:].any()
            col0 = torch.where(mb["dense_mask"][r, :t], mb["topk_idx"][r, :t, 0], torch.full((t,), BLANK)).tolist()
            assert row_ids(mb, r) == ctc_greedy(col0) and int(mb["n_tok"][r]) == len(row_ids(mb, r))
            assert mb["sources"][r] == store.utts[pos[pieces[r][0]]].source and int(mb["index"][r]) == pos[pieces[r][0]]
    assert seen["truncated"] and seen["mixed"] and seen["concat_groups"]  # each one happened somewhere
    assert seen["cut_padded"] and seen["end_padded"]


def test_truncate_keeps_the_frames_before_its_cut(env):
    """truncate alone (p 1): a cut row's targets are the row's first c frames, bit for bit, and its audio the row's
    first L samples, L drawn where the audio is still c frames (1280 c - 1120 .. + 159: never always on an 80 ms
    boundary, which the first recipe's student learned as its "no mark" cue); c is one of truncate_frames, so its CTC
    path is a strict prefix of the row's whose next token (the first one the cut removed) is a word, never the mark;
    a row with nothing to cut stays whole."""
    store = env["store"]
    plain = T.dataset_for(store)
    a = T.Augment(seed=2, truncate_p=1.0, truncate_min_frac=0.3, truncate_min_s=0.3, punct_ids=(MARK,))
    ds = plain.with_augment(a)
    n_cut, ends = 0, []
    for idx in micro_batches(store)[:20]:
        p, g = plain[idx], ds[idx]
        assert g["ids"] == p["ids"] and g["aug"]["concat_groups"] == g["aug"]["mixed"] == 0
        cut = 0
        for b, i in enumerate(idx):
            T0, c = int(p["n_frames"][b]), int(g["n_frames"][b])
            cand = T.truncate_frames(store.targets(i), 0.3, 0.3, (MARK,))
            if c == T0:  # p 1: only a row with nothing to cut stays whole (no word token starting late enough)
                assert not len(cand)
                n = int(p["lengths"][b])
                assert int(g["lengths"][b]) == n and torch.equal(g["wave"][b, :n], p["wave"][b, :n])
                assert row_ids(g, b) == row_ids(p, b) and float(g["durations"][b]) == float(p["durations"][b])
                continue
            cut += 1
            L = int(g["lengths"][b])
            assert c in set(cand.tolist()) and T.ctc_frames(L) == c
            assert FS * c - T.END_BELOW <= L <= FS * c + T.END_ABOVE
            assert torch.equal(g["wave"][b, :L], p["wave"][b, :L]) and not g["wave"][b, L:].any()
            ends.append(L - FS * c)
            for k in ("blank_lp", "dense_mask", "topk_idx", "topk_lp", "frame_mask"):
                assert torch.equal(g[k][b, :c], p[k][b, :c]), k
            assert not g["frame_mask"][b, c:].any() and not g["dense_mask"][b, c:].any()
            ids, full_ids = row_ids(g, b), row_ids(p, b)
            assert len(ids) < len(full_ids) and ids == full_ids[:len(ids)]  # a prefix: the final token is gone
            assert full_ids[len(ids)] != MARK  # and the first token the cut removed is a word, not only the mark
            assert float(g["durations"][b]) == pytest.approx(L / 16000)
        assert g["aug"]["truncated"] == cut
        n_cut += cut
    assert n_cut > 10
    # the ends spread over the frame: a full last frame (the valid mel frames a multiple of 8) about 1 in 8
    full = sum(((FS * 100 + e) // T.HOP) % 8 == 0 for e in ends)
    assert min(ends) < 0 < max(ends) and full < 0.4 * len(ends)


def test_short_rows_are_never_cut(env):
    """truncate_min_row_s: with truncate at p 1, a row shorter than the bound (its frames x 80 ms) is never cut, and
    truncate_frames has no candidate for it; a longer row with a candidate is cut as before."""
    store = env["store"]
    plain = T.dataset_for(store)
    ds = plain.with_augment(T.Augment(seed=10, truncate_p=1.0, truncate_min_s=0.3, punct_ids=(MARK,),
                                      truncate_min_row_s=1.5))
    short = long_cut = 0
    for idx in micro_batches(store)[:25]:
        p, g = plain[idx], ds[idx]
        for b, i in enumerate(idx):
            T0, t = int(p["n_frames"][b]), int(g["n_frames"][b])
            ft = store.targets(i)
            if T0 * 0.08 < 1.5:
                short += 1
                assert t == T0 and not len(T.truncate_frames(ft, 0.3, 0.3, (MARK,), 1.5))
            elif len(T.truncate_frames(ft, 0.3, 0.3, (MARK,), 1.5)):
                long_cut += 1
                assert t < T0
    assert short > 10 and long_cut > 5


def test_concat_puts_each_piece_at_its_offset(env):
    """concat alone (p 1): every row of a micro-batch joined into groups of k (k divides the rows); each piece's
    audio starts at 1280 x its frame offset, padded with low noise up to there; its targets sit at that offset as the
    store holds them; the row's CTC path is the greedy path of the joined argmax; the joined micro-batch's padded
    frames (rows x longest) never exceed the plain one's, and its padded samples exceed the plain one's by the pads
    alone (< 1280 samples per join)."""
    store = env["store"]
    pos = {u.id: i for i, u in enumerate(store.utts)}
    plain = T.dataset_for(store)
    ds = plain.with_augment(T.Augment(seed=3, concat_p=1.0, concat_max_s=2.6, concat_max_n=4))
    joined = 0
    for idx in micro_batches(store, micro_s=4.0):
        p, g = plain[idx], ds[idx]
        n, B = len(idx), len(g["ids"])
        if not g["aug"]["concat_groups"]:  # no k qualified: the micro-batch as it was
            assert_same({k: v for k, v in g.items() if k != "aug"}, p)
            assert all(n % k or k * int(p["n_frames"].max()) * 0.08 > 2.6 + 1e-9 for k in range(2, 5))
            continue
        k = n // B
        joined += B
        assert n % B == 0 and 2 <= k <= 4 and k * int(p["n_frames"].max()) * 0.08 <= 2.6 + 1e-9
        assert sorted(x for r in g["ids"] for x in r.split("+")) == sorted(p["ids"])
        assert B * int(g["n_frames"].max()) <= n * int(p["n_frames"].max())  # the padded frames
        assert B * g["wave"].shape[1] <= n * p["wave"].shape[1] + B * (k - 1) * 1120  # the audio: the pads (<= 70 ms)
        for r, rid in enumerate(g["ids"]):
            pieces = [pos[x] for x in rid.split("+")]
            assert len(pieces) == k and int(g["index"][r]) == pieces[0]
            assert float(g["durations"][r]) == pytest.approx(sum(store.utts[i].duration for i in pieces), rel=1e-6)
            off = 0
            for j, i in enumerate(pieces):
                t, w = store.targets(i), store.wave(i)
                last = j == len(pieces) - 1
                span = len(w) if last else FS * t.n_frames
                got = g["wave"][r, FS * off:FS * off + span].numpy()
                keep = min(len(w), span)
                assert np.array_equal(got[:keep], w[:keep])
                if span > keep:
                    assert np.abs(got[keep:]).max() < 1e-4 and np.count_nonzero(got[keep:]) == span - keep
                fr = slice(off, off + t.n_frames)
                assert torch.equal(g["blank_lp"][r, fr], torch.from_numpy(t.blank_lp.astype(np.float32)))
                d = torch.from_numpy(t.dense_frame.astype(np.int64)) + off
                assert int(g["dense_mask"][r, fr].sum()) == len(d) and bool(g["dense_mask"][r, d].all())
                assert torch.equal(g["topk_idx"][r, d], torch.from_numpy(t.topk_idx.astype(np.int64)))
                assert torch.equal(g["topk_lp"][r, d], torch.from_numpy(t.topk_lp.astype(np.float32)))
                off += t.n_frames
            assert int(g["n_frames"][r]) == off == T.ctc_frames(int(g["lengths"][r]))
            joined_col0 = np.concatenate([store.targets(i).col0() for i in pieces])
            assert row_ids(g, r) == ctc_greedy(joined_col0)
    assert joined >= 3


def test_mix_keeps_the_targets_and_hits_the_snr(env):
    """mix alone (p 1, SNR fixed at 10 dB): the targets, lengths and metadata are the plain micro-batch's; each mixed
    row differs from its plain audio on one contiguous span only, where the added signal is 10 dB below the row."""
    store = env["store"]
    plain = T.dataset_for(store)
    ds = plain.with_augment(T.Augment(seed=4, mix_p=1.0, mix_snr_db=(10.0, 10.0)))
    mixed = 0
    for idx in micro_batches(store)[:20]:
        p, g = plain[idx], ds[idx]
        assert_same(g, p, FRAME_KEYS + ("lengths", "ids", "sources", "durations", "index"))
        rows = 0
        for b in range(len(idx)):
            diff = (g["wave"][b].double() - p["wave"][b].double()).numpy()
            nz = np.flatnonzero(diff)
            if not len(nz):
                continue
            rows += 1
            s, e = int(nz[0]), int(nz[-1]) + 1
            assert 0 <= s < e <= int(p["lengths"][b])
            row = p["wave"][b].double().numpy()[s:e]
            snr = 10 * np.log10(np.mean(row ** 2) / np.mean(diff[s:e] ** 2))
            assert abs(snr - 10.0) < 0.01, snr
        assert rows == g["aug"]["mixed"] == (len(idx) if len(idx) >= 2 else 0)  # p 1: every row of 2+, none alone
        mixed += rows
    assert mixed > 10


def test_same_index_list_same_batch(env):
    """The augmentation of a micro-batch is a function of (seed, its index list): the same list gives the same batch
    from another dataset object (a worker's unpickled copy), another seed or another order of the list another."""
    store = env["store"]
    a = T.Augment(seed=6, truncate_p=0.5, truncate_min_s=0.3, concat_p=0.5, concat_max_s=2.6, mix_p=0.5,
                  punct_ids=(MARK,), truncate_pad_p=0.5, end_pad_p=0.5)
    ds = T.dataset_for(store, augment=a)
    other = pickle.loads(pickle.dumps(ds))
    assert other.augment == a and other._mm is None
    differs = 0
    for idx in micro_batches(store)[:15]:
        x, y = ds[idx], other[idx]
        assert_same(x, y)
        assert_same(ds[idx], x)  # and again from the same object
        z = T.dataset_for(store, augment=dataclasses.replace(a, seed=7))[idx]
        differs += not (z["wave"].shape == x["wave"].shape and torch.equal(z["wave"], x["wave"]))
    assert differs >= 5


def test_end_pads_never_widen_the_micro_batch(env):
    """end pads alone (p 1): every whole row but the micro-batch's longest gets 1-5 frames of quiet, within the longest
    row's frames - the padded frames never grow -; the audio and targets before the pad are the plain row's (the audio
    squared to its frames), the pad quiet, its targets pure blank, the CTC path unchanged (the final mark kept), and the
    row's duration grows by the pad."""
    store = env["store"]
    plain = T.dataset_for(store)
    ds = plain.with_augment(T.Augment(seed=8, end_pad_p=1.0, pad_frames=(1, 5)))
    padded = 0
    for idx in micro_batches(store)[:25]:
        p, g = plain[idx], ds[idx]
        width = int(p["n_frames"].max())
        assert int(g["n_frames"].max()) == width and g["aug"]["truncated"] == g["aug"]["cut_padded"] == 0
        n_pad = 0
        for b in range(len(idx)):
            T0, t = int(p["n_frames"][b]), int(g["n_frames"][b])
            if T0 == width:  # the longest row: no room
                assert t == T0 and int(g["lengths"][b]) == int(p["lengths"][b])
                continue
            n_pad += 1
            assert 1 <= t - T0 <= min(5, width - T0)
            n = int(p["lengths"][b])
            L = int(g["lengths"][b])
            assert FS * t - T.END_BELOW <= L <= FS * t and T.ctc_frames(L) == t
            keep = min(n, FS * T0)
            assert torch.equal(g["wave"][b, :keep], p["wave"][b, :keep])
            if L > FS * T0:  # the pad's audio (its end drawn): quiet, room tone or near-silent noise
                q = g["wave"][b, FS * T0:L].double()
                assert float((q ** 2).mean()) < T.PAD_ROOM_MAX_POWER
            for k in ("blank_lp", "dense_mask", "topk_idx", "topk_lp"):
                assert torch.equal(g[k][b, :T0], p[k][b, :T0]), k
            assert not g["dense_mask"][b, T0:t].any() and bool(g["frame_mask"][b, :t].all())
            assert row_ids(g, b) == row_ids(p, b)
            assert float(g["durations"][b]) == pytest.approx(L / 16000)
        assert g["aug"]["end_padded"] == n_pad
        padded += n_pad
    assert padded > 10


def test_a_cut_pad_keeps_the_cut_and_adds_quiet(env):
    """truncate and cut pads (p 1 each): a cut row is its cut's first c frames - c one of truncate_frames, the audio,
    the targets and the CTC path (a prefix that lost its final token) the plain row's - then 1-5 frames of quiet with
    pure-blank targets, never past the micro-batch's width; every cut row has its pad (a cut leaves room), a row with
    nothing to cut stays whole (end_pad_p 0)."""
    store = env["store"]
    plain = T.dataset_for(store)
    ds = plain.with_augment(T.Augment(seed=9, truncate_p=1.0, truncate_min_s=0.3, punct_ids=(MARK,),
                                      truncate_pad_p=1.0, pad_frames=(1, 5)))
    cut_rows = 0
    for idx in micro_batches(store)[:20]:
        p, g = plain[idx], ds[idx]
        width = int(p["n_frames"].max())
        for b, i in enumerate(idx):
            T0, t = int(p["n_frames"][b]), int(g["n_frames"][b])
            cand = set(T.truncate_frames(store.targets(i), 0.3, 0.3, (MARK,)).tolist())
            if not cand:
                assert t == T0 and int(g["lengths"][b]) == int(p["lengths"][b])
                continue
            cut_rows += 1
            L = int(g["lengths"][b])
            assert t <= width and FS * t - T.END_BELOW <= L <= FS * t and T.ctc_frames(L) == t
            full = store.targets(i).col0().tolist()
            fits = [c for c in range(max(t - 5, 1), t) if c in cand
                    and row_ids(g, b) == ctc_greedy(full[:c]) and len(ctc_greedy(full[:c])) < len(ctc_greedy(full))
                    and torch.equal(g["blank_lp"][b, :c], p["blank_lp"][b, :c])
                    and torch.equal(g["wave"][b, :FS * c - T.END_BELOW], p["wave"][b, :FS * c - T.END_BELOW])
                    and not g["dense_mask"][b, c:t].any()]
            assert fits, (idx, b)
        assert g["aug"]["cut_padded"] == g["aug"]["truncated"] and g["aug"]["end_padded"] == 0
    assert cut_rows > 10


# ---------------------------------------------------------------------------------------------------- the trainer


def test_mark_tail_and_slice():
    """mark_tail finds the last word's run end and the marks after a blank gap; none without a word, without a final
    mark, or with the mark right behind the word. slice_frame_targets keeps frames [a, b) with their own frame
    numbers."""
    W = 300
    assert T.mark_tail(ft([W, W, BLANK, BLANK, BLANK, MARK, BLANK], 1), [MARK]) == (1, 5, 5)
    assert T.mark_tail(ft([W, BLANK, MARK, BLANK, MARK], 2), [MARK]) == (0, 2, 4)  # marks, a blank between them
    assert T.mark_tail(ft([W, MARK, BLANK], 3), [MARK]) is None  # no gap
    assert T.mark_tail(ft([W, BLANK, BLANK], 4), [MARK]) is None  # no final mark
    assert T.mark_tail(ft([BLANK, MARK], 5), [MARK]) is None  # no word
    f = ft([W, BLANK, 301, 301, BLANK, MARK], 6)
    s = T.slice_frame_targets(f, 2, 6)
    assert s.n_frames == 4 and s.col0().tolist() == [301, 301, BLANK, MARK] and s.ctc_ids.tolist() == [301, MARK]
    assert np.array_equal(s.blank_lp, f.blank_lp[2:6])


def test_end_trim_moves_the_mark_up_behind_the_last_word():
    """A row ending "word, 3 blank frames, mark": the blank frames go, the mark's frame follows the word, the greedy
    path is unchanged, and the audio ends at a drawn sample of the new last frame (still its frames); a row without a
    gap is left as it is."""
    W = 300
    f = ft([W, W, BLANK, 301, BLANK, BLANK, BLANK, MARK], 7)
    wave = np.random.default_rng(1).standard_normal(FS * 8).astype(np.float32)
    r = T._AugRow(wave, f, [0], [0])
    assert r.trim_end([MARK], np.random.default_rng(2))
    assert r.ft.col0().tolist() == [W, W, BLANK, 301, MARK] and r.ft.ctc_ids.tolist() == [W, 301, MARK]
    assert T.ctc_frames(len(r.wave)) == 5 and np.array_equal(r.wave, wave[:len(r.wave)]) and r.trimmed
    lens = []
    for seed in range(40):
        q = T._AugRow(wave, f, [0], [0])
        q.trim_end([MARK], np.random.default_rng(seed))
        lens.append(len(q.wave))
    assert min(lens) >= FS * 5 - T.END_BELOW and max(lens) <= FS * 5 + T.END_ABOVE and len(set(lens)) > 10
    g = T._AugRow(wave[:FS * 3], ft([W, W, MARK], 8), [0], [0])
    assert not g.trim_end([MARK], np.random.default_rng(0)) and not g.trimmed and len(g.wave) == FS * 3


def mark_tailed(monkeypatch):
    """Every stored row's targets end "word, 3 blank frames, mark" (its frame count unchanged): the fake teacher's
    random tokens rarely end in a mark."""
    real = T.FrameBatchDataset.targets

    def targets(self, i):
        c = real(self, i).col0().copy()
        if len(c) >= 6:
            c[-5:] = [300, BLANK, BLANK, BLANK, MARK]
        return ft(c.tolist(), int(i))

    monkeypatch.setattr(T.FrameBatchDataset, "targets", targets)


def test_end_trim_on_a_frame_store(env, monkeypatch):
    """end_trim_p 1: every row that is not cut loses its 3 blank frames before its final mark - n_frames, ctc targets
    and audio in step (ctc_frames(samples) == n_frames) -, durations its audio's; cut rows (truncate_p 0.5) never."""
    mark_tailed(monkeypatch)
    st = env["store"]
    for truncate in (0.0, 0.5):
        ds = T.dataset_for(st, augment=T.Augment(seed=3, end_trim_p=1.0, truncate_p=truncate, truncate_min_s=0.3,
                                                 punct_ids=(MARK,)))
        trimmed = 0
        for idx in micro_batches(st)[:12]:
            got = ds[idx]
            here = 0
            for b, i in enumerate(idx):
                n, T0 = int(got["n_frames"][b]), ds.targets(i).n_frames
                assert T.ctc_frames(int(got["lengths"][b])) == n
                if T0 >= 6 and n == T0 - 3 and row_ids(got, b)[-1:] == [MARK]:  # trimmed: the mark behind the word
                    here += 1
                    assert got["durations"][b] == pytest.approx(int(got["lengths"][b]) / T.TARGET_SR)
                else:  # cut (inside the row, never trimmed after) or too short to carry the tail
                    assert n < T0 - 3 or T0 < 6 or n == T0
            assert got["aug"]["end_trimmed"] == here
            trimmed += here
        assert trimmed > 0


def test_background_keeps_targets_and_lengths(env, tmp_path):
    """noise_p 1: every row's audio has the bank's background added (its length and targets unchanged); without the
    bank the augmentation is refused; the same index list gives the same background."""
    from kitsune.noise_bank import NoiseBank

    bank = NoiseBank.load(make_noise_bank(tmp_path / "bank"))
    st = env["store"]
    with pytest.raises(ValueError, match="needs a noise bank"):
        T.dataset_for(st, augment=T.Augment(seed=1, noise_p=1.0))
    ds = T.dataset_for(st, augment=T.Augment(seed=1, noise_p=1.0, noise_snr_db=(5.0, 5.0)), noise=bank)
    plain = T.dataset_for(st)
    for idx in micro_batches(st)[:6]:
        got, ref = ds[idx], plain[idx]
        assert got["aug"]["noised"] == len(idx)
        assert torch.equal(got["lengths"], ref["lengths"]) and torch.equal(got["ctc_targets"], ref["ctc_targets"])
        assert not torch.equal(got["wave"], ref["wave"])
        again = ds[idx]
        assert torch.equal(again["wave"], got["wave"])


def test_validate_the_augment_block():
    """augment.* off by default and checked whether or not it is on (an AED student's own rules - the cut table, no
    pads -: tests/test_aed_augment.py); a resume may change it - it does not shape the step plan -, unlike the
    RESUME_FIXED keys."""
    m = load_script("04_distill")
    assert m.DEFAULTS["augment"] == {"enabled": False, "seed": None, "truncate_p": 0.0, "truncate_min_frac": 0.3,
                                     "truncate_min_s": 1.0, "truncate_pause_p": 0.5, "concat_p": 0.0,
                                     "concat_max_s": 28.0, "concat_max_n": 4, "mix_p": 0.0, "mix_snr_db": [5.0, 20.0],
                                     "truncate_pad_p": 0.0, "end_pad_p": 0.0, "pad_frames": [1, 5],
                                     "truncate_min_row_s": 0.0, "cuts": None, "cuts_sha256": None,
                                     "end_trim_p": 0.0, "noise_p": 0.0, "noise_snr_db": [0.0, 20.0],
                                     "noise_bank": None, "noise_bank_sha256": None, "speech_p": 0.0,
                                     "speech_snr_db": [10.0, 25.0], "speech_talkers": [1, 4], "speech_batch_p": 0.5,
                                     "reverb_p": 0.0, "rir_bank": None, "rir_bank_sha256": None, "gain_p": 0.0,
                                     "gain_db": [-20.0, 10.0], "codec_p": 0.0, "codecs": ["mp3", "gsm", "ulaw8k"]}
    assert not m.augment_on(m.load_config(None, [])) and not m.augment_on({})
    ctc = ["family=ctc", "parakeet_root=po"]
    assert m.augment_on(m.load_config(None, ctc + ["augment.enabled=true", "augment.mix_p=0.5"]))
    for bad, match in ((["augment.enabled=yes"], "augment.enabled must be true or false"),
                       (["augment.seed=-1"], "augment.seed"), (["augment.seed=1.5"], "augment.seed"),
                       (["augment.truncate_p=1.5"], "augment.truncate_p"),
                       (["augment.concat_p=-0.1"], "augment.concat_p"),
                       (["augment.mix_p=x"], "augment.mix_p"), (["augment.truncate_min_frac=1"], "truncate_min_frac"),
                       (["augment.truncate_pause_p=2"], "augment.truncate_pause_p"),
                       (["augment.truncate_min_s=-1"], "truncate_min_s"), (["augment.concat_max_s=0"], "concat_max_s"),
                       (["augment.concat_max_n=1"], "concat_max_n"), (["augment.concat_max_n=2.0"], "concat_max_n"),
                       (["augment.mix_snr_db=[20, 5]"], "mix_snr_db"), (["augment.mix_snr_db=[-5, 5]"], "mix_snr_db"),
                       (["augment.mix_snr_db=[5]"], "mix_snr_db"),
                       (["augment.truncate_pad_p=1.5"], "augment.truncate_pad_p"),
                       (["augment.end_pad_p=-0.1"], "augment.end_pad_p"), (["augment.pad_frames=[0, 5]"], "pad_frames"),
                       (["augment.pad_frames=[3, 2]"], "pad_frames"), (["augment.pad_frames=[1, 26]"], "pad_frames"),
                       (["augment.pad_frames=[1.0, 2]"], "pad_frames"), (["augment.pad_frames=3"], "pad_frames"),
                       (["augment.truncate_min_row_s=-1"], "truncate_min_row_s"),
                       (["augment.end_trim_p=2"], "augment.end_trim_p"), (["augment.noise_p=-1"], "augment.noise_p"),
                       (["augment.noise_snr_db=[10, 0]"], "noise_snr_db"),
                       (["augment.noise_bank_sha256=xyz"], "noise_bank_sha256"),
                       (["augment.speech_p=2"], "augment.speech_p"), (["augment.reverb_p=-0.1"], "augment.reverb_p"),
                       (["augment.gain_p=1.5"], "augment.gain_p"), (["augment.codec_p=-1"], "augment.codec_p"),
                       (["augment.speech_snr_db=[25, 10]"], "speech_snr_db"), (["augment.gain_db=[5, -5]"], "gain_db"),
                       (["augment.speech_talkers=[0, 3]"], "speech_talkers"),
                       (["augment.speech_talkers=[2, 9]"], "speech_talkers"),
                       (["augment.codecs=[\"mp3\", \"aac\"]"], "augment.codecs"),
                       (["augment.codecs=[\"mp3\", \"mp3\"]"], "augment.codecs"),
                       (["augment.rir_bank_sha256=xyz"], "rir_bank_sha256"),
                       (["augment.enabled=true", "augment.reverb_p=0.3"], "needs augment.rir_bank"),
                       (["augment.enabled=true", "augment.codec_p=0.3", "augment.codecs=[]"], "at least one"),
                       (["augment.enabled=true", "augment.speech_p=0.3"], "needs augment.noise_bank with speech"),
                       (["augment.enabled=true", "augment.noise_p=0.3"], "needs augment.noise_bank")):
        with pytest.raises(SystemExit, match=match):
            m.load_config(None, ctc + bad)
    saved = m.load_config(None, ctc)
    changed, same = m.resume_overrides(saved, [("augment.enabled", True), ("augment.concat_p", 0.5),
                                               ("augment.seed", None)])
    assert changed == {"augment.enabled": True, "augment.concat_p": 0.5} and same == {"augment.seed": None}
    assert not any(f.startswith("augment") for f in m.RESUME_FIXED)


def write_config(env, name: str, over: dict) -> str:
    def merged(base, o):
        out = copy.deepcopy(base)
        for k, v in o.items():
            out[k] = merged(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
        return out

    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


def one_run(env, name: str) -> Path:
    runs = list((env["root"] / "runs").glob(f"{name}-2*"))
    assert len(runs) == 1, runs
    return runs[0]


def events(run: Path, kind: str | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def utts_of(run: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((run / "metrics" / "train_utts").glob("part-*.parquet"))])


# the end-to-end run: all three augmentations, through a smoke phase of 3 steps, full states every 2 steps
AUG_OVER = {"augment": {"enabled": True, "truncate_p": 0.7, "truncate_min_s": 0.3, "concat_p": 1.0, "mix_p": 0.7,
                        "truncate_pad_p": 0.7, "end_pad_p": 0.5, "end_trim_p": 0.5, "noise_p": 0.5, "speech_p": 0.5,
                        "speech_batch_p": 0.5, "reverb_p": 0.5, "gain_p": 0.5, "codec_p": 0.5},
            "smoke": {"enabled": True, "steps": 3, "min_audio_s_per_s": 0, "require_loss_decrease": False,
                      "pad_utts": 4, "decode_per_set": 2},
            "ckpt": {"full_every_steps": 2}}


@pytest.fixture(scope="module")
def aug_run(env):
    m = load_script("04_distill")
    assert m.main(["--config", write_config(env, "ctc-aug", aug_over(env))]) == 0
    return one_run(env, "ctc-aug")


def aug_over(env) -> dict:
    """AUG_OVER with the env's background bank (music, speech, song, noise) and RIR bank (built once per module, under
    its root)."""
    d, r = env["root"] / "noise_bank", env["root"] / "rir_bank"
    if not (d / "index.json").exists():
        make_noise_bank(d, kinds=("music", "speech", "song", "noise"))
    if not (r / "index.json").exists():
        make_rir_bank(r)
    out = copy.deepcopy(AUG_OVER)
    out["augment"]["noise_bank"], out["augment"]["rir_bank"] = str(d), str(r)
    return out


def test_a_ctc_run_with_every_augmentation(env, aug_run):
    """A tiny CTC student trained with all three augmentations: the smoke phase passes (its checks read the plain
    dataset; its dropped-audio share counts utterances), the clamp of concat_max_s to the longest train utterance is
    an `augment` event, every step row has aug/concat_frac, aug/truncated_frac and aug/mixed_frac in [0, 1] - each
    above 0 somewhere -, and the train records carry joined rows' ids."""
    run = aug_run
    evs = events(run)
    kinds = [e["kind"] for e in evs]
    assert kinds[-1] == "logger_close" and "tb_tag_unmapped" not in kinds and "nonfinite_grad_skipped" not in kinds
    aug = next(e for e in evs if e["kind"] == "augment")
    longest = max(u.duration for u in env["store"].utts)
    assert aug["concat_max_s"] == pytest.approx(longest) and aug["concat_max_s_clamped"] is True
    assert aug["concat_max_s_config"] == 28.0 and aug["longest_train_s"] == pytest.approx(longest, abs=1e-3)
    assert aug["seed"] == 1234 and aug["concat_p"] == 1.0 and aug["mix_snr_db"] == [5.0, 20.0]
    assert aug["punct_ids"] == {"1": "。"} and aug["truncate_pause_p"] == 0.5  # the tiny tokenizer's only mark
    smoke = {e["kind"]: e for e in evs if e["kind"].startswith("smoke_")}
    assert smoke["smoke_padded_row"]["ok"] and smoke["smoke_longest_fwd_bwd"]["ok"]
    assert smoke["smoke_steps"]["steps"] == 3 and smoke["smoke_steps"]["dropped"] == 0
    st = pd.read_parquet(run / "metrics" / "steps.parquet")
    assert st["step"].tolist() == [1, 2, 3, 4, 5, 6] and np.isfinite(st["loss/objective"]).all()
    for tag in ("aug/concat_frac", "aug/truncated_frac", "aug/mixed_frac", "aug/cut_padded_frac", "aug/noised_frac",
                "aug/speech_frac", "aug/reverb_frac", "aug/gain_frac", "aug/codec_frac"):
        assert ((st[tag] >= 0) & (st[tag] <= 1)).all() and st[tag].max() > 0, tag
    # the acoustic steps' settings and banks (DECISIONS H12), and the codec probe's verdict
    assert (aug["speech_p"], aug["reverb_p"], aug["gain_p"], aug["codec_p"], aug["rir_clips"]) == (0.5, 0.5, 0.5, 0.5, 3)
    assert aug["codecs"] == ["mp3", "gsm", "ulaw8k"] and set(aug["noise_kind_hours"]) == {"music", "speech", "song",
                                                                                       "noise"}
    probe = next(e for e in evs if e["kind"] == "augment_codecs")
    assert probe["available"] == ["mp3", "gsm", "ulaw8k"] and probe["unavailable"] == {}
    end = st["aug/end_padded_frac"]  # whole rows: in a joined micro-batch most are its longest, so maybe never
    assert ((end >= 0) & (end <= 1)).all()
    assert (aug["truncate_pad_p"], aug["end_pad_p"], aug["pad_frames"]) == (0.7, 0.5, [1, 5])
    tags = set(pd.read_parquet(run / "metrics" / "scalars.parquet")["tag"])
    assert {"aug/concat_frac", "aug/truncated_frac", "aug/mixed_frac", "aug/masked_frac"} <= tags
    utts = utts_of(run)
    joined = utts[utts["id"].str.contains("+", regex=False)]
    # a joined row's per-utterance record (none when every joined row of this tiny run - about one - was cut inside its
    # first piece: a cut row keeps only the pieces that start before its cut; the dataset tests cover joined ids)
    assert (joined["n_tok"] > 0).all()
    ids = {u.id for u in env["store"].utts}
    assert all(x in ids for r in utts["id"] for x in r.split("+"))
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_a_resume_reset_takes_the_recipe_sets_of_the_env_word(env, monkeypatch):
    """DECISIONS H1 end to end on the trainer side - the recipe test box in miniature: a finished CTC run on the epochs
    clock, trained without augmentation, is re-run from its pre_cooldown state to the SAME T with the recipe on. The
    sets come the way the box gets them: launch's env word (KITSUNE_RESUME_SETS, typed loosely here; launch normalises
    it) -> fullrun.parse_resume_sets -> the queue's sets_once (schedule.resume_reset=true first: kitsune.full_queue
    resume_pull / adopt) -> one --set each (FullQueue.argv_for) -> 04_distill --resume <pre_cooldown> takes them: one
    resume_reset at the pre_cooldown step whose re-planned T is the first run's (the paired A/B: the steps before it,
    their data and LR, are the baseline's), the augment event with the set values and the trainer's defaults for the
    rest, aug/* shares on the re-run steps only, and the recipe in every state saved after the reset - so a
    crash-resume without the sets (the queue drops them once the reset is applied) keeps it - to the end at the same T.
    """
    from kitsune import fullrun

    m = load_script("04_distill")
    path = write_config(env, "ctc-recipe", {"schedule": {"clock": "epochs", "epochs": 1, "max_steps": None}})
    assert m.main(["--config", path]) == 0
    run = one_run(env, "ctc-recipe")
    first = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    T = int(first["steps"])
    (pc_event,) = [e for e in events(run, "checkpoint") if e.get("reason") == "pre_cooldown"]
    t_c = int(pc_event["name"].rsplit("_", 1)[1])
    assert first["status"] == "complete" and t_c + 2 <= T and not events(run, "augment")
    pc = run / "checkpoints" / pc_event["name"]
    rid = run.name
    word = ",".join(f"{rid}:{s}" for s in ("schedule.epochs=01", "augment.enabled=True", "augment.truncate_p=.3",
                                           "augment.concat_p=0.50", "augment.mix_p=5e-2"))
    sets = ["schedule.resume_reset=true", *fullrun.parse_resume_sets(word)[rid]]
    assert sets == ["schedule.resume_reset=true", "schedule.epochs=1", "augment.enabled=true", "augment.truncate_p=0.3",
                    "augment.concat_p=0.5", "augment.mix_p=0.05"]
    argv = sum((["--set", s] for s in sets), [])
    # ckpt.full_every_steps is the test's own (a state after the reset for the crash below), not a queue set
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", str(T))
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--resume", str(pc), *argv, "--set", "ckpt.full_every_steps=1"])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    (r,) = events(run, "resume_reset")
    assert (r["at_step"], r["T"], r["total_steps_before"], r["total_steps_after"], r["epochs"], r["resume_resets"]) \
        == (t_c, T, T, T, 1, 1)
    newest = run / "checkpoints" / f"full_step_{T - 1}"
    saved = json.loads((newest / "trainer.json").read_text(encoding="utf-8"))["cfg"]
    want_aug = dict(m.DEFAULTS["augment"], enabled=True, truncate_p=0.3, concat_p=0.5, mix_p=0.05)
    assert saved["augment"] == want_aug and saved["schedule"]["resume_reset"] is False
    assert m.main(["--resume", str(newest)]) == 0  # no sets: the state's config holds the recipe
    assert len(events(run, "resume_reset")) == 1
    augs = events(run, "augment")
    assert len(augs) == 2  # the reset's start and the crash-resume's, both with the recipe
    for a in augs:
        assert (a["truncate_p"], a["concat_p"], a["mix_p"], a["truncate_pause_p"], a["truncate_min_s"], a["seed"]) == (
            0.3, 0.5, 0.05, 0.5, 1.0, 1234)
        assert a["punct_ids"] == {"1": "。"} and a["mix_snr_db"] == [5.0, 20.0]
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert (s["status"], s["steps"], s["resume_resets"]) == ("complete", T, 1)
    end = json.loads((run / "checkpoints" / f"full_step_{T}" / "trainer.json").read_text(encoding="utf-8"))
    assert end["cfg"]["augment"] == want_aug and end["st"]["resume_resets"] == 1
    st = pd.read_parquet(run / "metrics" / "steps.parquet").drop_duplicates("step", keep="last").set_index("step")
    assert st.index.tolist() == list(range(1, T + 1))
    tags = ("aug/concat_frac", "aug/truncated_frac", "aug/mixed_frac")
    after = st.loc[t_c + 1:, list(tags)]
    assert ((after >= 0) & (after <= 1)).all().all() and after.to_numpy().sum() > 0  # the recipe acted
    assert st.loc[:t_c, list(tags)].isna().all().all()  # the baseline's steps: no augmentation


def test_a_crash_resumed_with_augmentation_is_the_uninterrupted_run(env, aug_run, monkeypatch):
    """A crash before step 5 resumed from full_step_4 ends with exactly the uninterrupted run's weights: the resumed
    loader augments steps 5 and 6 as the uninterrupted one did (the same rows, cuts and mixes; the same losses)."""
    m = load_script("04_distill")
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "5")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", write_config(env, "ctc-aug-crash", aug_over(env))])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    crash = one_run(env, "ctc-aug-crash")
    assert m.main(["--resume", str(crash / "checkpoints" / "full_step_4")]) == 0
    a = torch.load(aug_run / "checkpoints" / "full_step_6" / "model.pt", weights_only=True)
    b = torch.load(crash / "checkpoints" / "full_step_6" / "model.pt", weights_only=True)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ua, ub = utts_of(aug_run), utts_of(crash)
    for step in (5, 6):
        x = ua[ua["step"] == step].sort_values("id")
        y = ub[(ub["step"] == step) & (ub["attempt"] == 1)].sort_values("id")
        assert len(x) and x["id"].tolist() == y["id"].tolist() and x["duration"].tolist() == y["duration"].tolist()
        assert x["kl"].tolist() == y["kl"].tolist()
    sa = pd.read_parquet(aug_run / "metrics" / "steps.parquet").set_index("step")
    sb = pd.read_parquet(crash / "metrics" / "steps.parquet").drop_duplicates("step", keep="last").set_index("step")
    for tag in ("loss/objective", "aug/concat_frac", "aug/truncated_frac", "aug/mixed_frac", "aug/cut_padded_frac",
                "aug/end_padded_frac"):
        np.testing.assert_array_equal(sa[tag].to_numpy(), sb[tag].to_numpy())
