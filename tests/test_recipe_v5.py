"""The recipe audit's three fixes (2026-10-09, findings B1-B3; kitsune.trainset's "augmentation" section, scripts/
04_distill.py augment.cut_keep_word / guard_per_piece / end_trim_voiced, make_full_configs RECIPE_V5), each its own key,
false by default:

- off, every function and dataset is as before the key: the cut candidates, the drawn cut and the generator's state
  after it, the trimmed rows, the acoustic chain's spared rows and the aug counters' keys (and the start pieces the
  trainer now always hands the CTC augmentation change nothing by themselves)
- cut_keep_word (B1): no cut keeps only the bare start piece "▁" - not in a ReazonSpeech-like lead-in before the first
  word, not in a joined row's later piece; a cut at a piece boundary after a piece with a word stays allowed; a word
  merged into the start piece is a word
- guard_per_piece (B2): a 2 x 1.6 s joined row under truncate_min_row_s 3 keeps only its boundary candidate (either
  family) and, under background_min_row_s 3, gets no background; the joined row's own rule still holds
- end_trim_voiced (B3): the end trim keeps a row's voiced audio that runs past its last word's spike, leaves a
  '▁ ... 。' row and a row whose voice runs into its mark alone, and no longer shrinks a frame-0-only-word row to 2
  frames
- the trainer: the keys validated (bools; the CTC-only ones refused on an AED student), changeable on a resume, the
  `augment` event and console line as before them when off; fullrun's resume sets take them as bools; RECIPE_V5 is
  recipe v4 with the three on and validates on a CTC student
CPU only, hand-made targets and synthetic audio (tests/fixtures.py, tests/fixtures_ctc.py)."""
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import make_full_configs as M  # noqa: E402
from fixtures import load_script, make_fake_corpus, make_fake_selection, make_noise_bank  # noqa: E402
from fixtures_ctc import make_fake_parakeet_out, tiny_tokenizer  # noqa: E402
from kitsune import fullrun as fr  # noqa: E402
from kitsune import trainset as T  # noqa: E402
from kitsune.ctc_targets import FrameTargets  # noqa: E402
from kitsune.noise_bank import NoiseBank  # noqa: E402
from kitsune.parakeet_targets import ctc_greedy  # noqa: E402

B, MARK, S, K = 3072, 1, 2, 8  # blank; "。" and the bare start piece "▁" in the tiny tokenizer (tests/fixtures_ctc.py)
P, SID = (MARK,), (S,)
FS, SR = T.FRAME_SAMPLES, T.TARGET_SR
FIXES = ("cut_keep_word", "guard_per_piece", "end_trim_voiced")
EOS = 3
RID = "full-p01-20261001T184145Z"


def ft(col0: list[int], seed: int = 0) -> FrameTargets:
    """FrameTargets whose argmax is `col0` (as tests/test_trainset_augment.py's)."""
    rng = np.random.default_rng(seed)
    col0 = np.asarray(col0, dtype=np.int64)
    dense = np.flatnonzero(col0 != B).astype(np.int32)
    idx = np.zeros((len(dense), K), np.int16)
    for j, d in enumerate(dense):
        idx[j] = np.concatenate([[col0[d]], rng.choice(np.arange(3, 300), size=K - 1, replace=False)])
    lp = np.sort(np.log(rng.dirichlet(np.ones(K), size=len(dense)) * 0.99 + 1e-6), axis=1)[:, ::-1].astype(np.float16)
    blank_lp = np.where(col0 == B, -0.01, -4.0).astype(np.float16)
    return FrameTargets(len(col0), blank_lp, dense, idx, lp, np.asarray(ctc_greedy(col0), dtype=np.int32))


def piece20(w: int) -> list[int]:
    """A 1.6 s utterance (20 frames): the start piece on frame 0, four words 4 frames apart, its mark on frame 18."""
    return [S, B, w, B, B, B, w + 1, B, B, B, w + 2, B, B, B, w + 3, B, B, B, MARK, B]


def voiced(n: int, voiced_n: int, seed: int = 0) -> np.ndarray:
    """n samples: a tone for the first voiced_n, then near-silence (-100 dBFS noise): voiced_end == voiced_n."""
    rng = np.random.default_rng(seed)
    w = (1e-5 * rng.standard_normal(n)).astype(np.float32)
    w[:voiced_n] = (0.1 * np.sin(2 * np.pi * 300 * np.arange(voiced_n) / SR)).astype(np.float32)
    return w


def same_rng(a: np.random.Generator, b: np.random.Generator) -> bool:
    return a.bit_generator.state == b.bit_generator.state


# ------------------------------------------------------------------------------------------- B1: cut_keep_word

# ReazonSpeech's habit: "▁" on frame 0, then a 30-frame (2.4 s) untranscribed lead-in, words at 31, 35, 39, the mark
LEADIN = [S] + [B] * 30 + [40] + [B] * 3 + [41] + [B] * 3 + [42] + [B] * 2 + [MARK] + [B] * 2


def test_cut_keep_word_off_is_today_and_on_never_keeps_only_the_start_piece():
    """Off (and with no start_ids passed) the candidates are today's - every frame from lo to the last word's start,
    the 17 lead-in frames 14-30 among them, each of which keeps only '▁' - and truncate_cut draws as before (the same
    cut, the same generator state after it); on, only the cuts that keep a word (32-39), so no draw ever keeps '▁'
    alone, pause cuts included. Without start_ids the start piece would count as a word; a word merged into it
    ("▁お": another id) is one."""
    t = ft(LEADIN, 1)
    assert t.n_frames == 45
    today = list(range(14, 40))  # lo = ceil(0.3 x 45) = 14; the last word (42) starts at 39
    assert T.truncate_frames(t, 0.3, 1.0, P).tolist() == today
    assert T.truncate_frames(t, 0.3, 1.0, P, 0.0, offsets=[0], keep_word=False, start_ids=SID,
                             per_piece=False).tolist() == today
    for pp in (0.0, 0.5, 1.0):
        for seed in range(30):
            r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
            assert T.truncate_cut(t, r1, 0.3, 1.0, P, pp) == T.truncate_cut(t, r2, 0.3, 1.0, P, pp, offsets=[0],
                                                                          start_ids=SID)
            assert same_rng(r1, r2)
    on = T.truncate_frames(t, 0.3, 1.0, P, offsets=[0], keep_word=True, start_ids=SID).tolist()
    assert on == list(range(32, 40))
    assert T.truncate_frames(t, 0.3, 1.0, P, keep_word=True, start_ids=SID).tolist() == on  # offsets None: [0]
    assert T.truncate_frames(t, 0.3, 1.0, P, keep_word=True).tolist() == today  # "▁" a word without start_ids

    def content_free(c):
        ids = T.cut_frame_targets(t, c).ctc_ids.tolist()
        return not [i for i in ids if i not in (S, MARK)]

    rng = np.random.default_rng(0)
    off_cuts = [T.truncate_cut(t, rng, 0.3, 1.0, P, 0.5) for _ in range(2000)]
    # half of the cuts in a pause (all 17 of its candidates in the lead-in), half uniform over 26: ~0.83 content-free
    assert 0.78 < np.mean([content_free(c) for c in off_cuts]) < 0.88
    on_cuts = [T.truncate_cut(t, rng, 0.3, 1.0, P, pp, offsets=[0], keep_word=True, start_ids=SID)
               for pp in (0.0, 0.5, 1.0) for _ in range(300)]
    assert set(on_cuts) == set(on) and not any(content_free(c) for c in on_cuts)
    merged = ft([45] + LEADIN[1:], 2)  # "▁お" on frame 0: a word, so a lead-in cut keeps one
    assert T.truncate_frames(merged, 0.3, 1.0, P, keep_word=True, start_ids=SID).tolist() == today


def test_cut_keep_word_in_a_joined_row():
    """A joined row of '▁ 50 。' (frames 0-6) and '▁ <8-frame lead-in> 51 52 。' (frames 7-21): off, the cuts after
    '▁' alone (1, 2) and inside the second piece's lead-in (8-16) are candidates as today; on, a cut must keep a word of
    the piece it ends in - 6 (after the first piece's mark) and the boundary 7 (the first piece kept whole, its word
    with it) stay, and 17-19 inside the second piece after its first word."""
    a = ft([S, B, 50, B, B, MARK, B], 3)
    b = ft([S] + [B] * 8 + [51, B, B, 52, B, MARK], 4)
    j = T.join_frame_targets([a, b])
    offs = [0, 7]
    assert j.n_frames == 22
    off = [1, 2, 6, 7, *range(8, 20)]
    assert T.truncate_frames(j, 0.0, 0.0, P).tolist() == off
    assert T.truncate_frames(j, 0.0, 0.0, P, offsets=offs, start_ids=SID).tolist() == off
    on = T.truncate_frames(j, 0.0, 0.0, P, offsets=offs, keep_word=True, start_ids=SID).tolist()
    assert on == [6, 7, 17, 18, 19]
    for c in on:  # the kept part's last piece holds a word: split the path at its start pieces
        ids = T.cut_frame_targets(j, c).ctc_ids.tolist()
        last = ids[len(ids) - ids[::-1].index(S):]
        assert [i for i in last if i != MARK], (c, ids)
    assert T.content_kept(j.col0(), offs, P, SID).tolist() == [c in on or c in (3, 4, 5, 20, 21) for c in range(22)]


# ----------------------------------------------------------------------------------------- B2: guard_per_piece


def test_guard_per_piece_cuts():
    """Two 1.6 s utterances joined (3.2 s, so the joined row passes truncate_min_row_s 3): off, cuts inside either
    piece as today; on, only the boundary (frame 20: the first utterance kept whole, the second dropped) - with
    cut_keep_word too. Pure tightening: a 2 x 1.2 s row (under 3 s) still has none, a standalone 1.6 s row none, and a
    long piece keeps its own candidates next to a short one. An AED row of the same pieces (its cut table's frames)
    loses every candidate inside them as well."""
    j = T.join_frame_targets([ft(piece20(60), 5), ft(piece20(70), 6)])
    offs = [0, 20]
    today = [13, 14, *range(19, 35)]  # lo = max(ceil(0.3 x 40), ceil(12.5)) = 13; never before a mark (15-18, 35-38)
    assert T.truncate_frames(j, 0.3, 1.0, P, 3.0).tolist() == today
    assert T.truncate_frames(j, 0.3, 1.0, P, 3.0, offsets=offs, start_ids=SID).tolist() == today
    assert T.truncate_frames(j, 0.3, 1.0, P, 3.0, offsets=offs, per_piece=True).tolist() == [20]
    assert T.truncate_frames(j, 0.3, 1.0, P, 3.0, offsets=offs, per_piece=True, keep_word=True,
                             start_ids=SID).tolist() == [20]
    rng = np.random.default_rng(0)
    assert {T.truncate_cut(j, rng, 0.3, 1.0, P, 0.5, 3.0, offsets=offs, per_piece=True) for _ in range(50)} == {20}
    assert T.inside_short_pieces(40, offs, 3.0).tolist() == [0 < f < 20 or 20 < f < 40 for f in range(40)]
    short = T.join_frame_targets([ft(piece20(60)[:13] + [B, MARK], 7), ft(piece20(70)[:13] + [B, MARK], 8)])
    assert short.n_frames == 30  # 2 x 1.2 s = 2.4 s: the joined row's own rule
    for per in (False, True):
        assert T.truncate_frames(short, 0.0, 0.0, P, 3.0, offsets=[0, 15], per_piece=per).tolist() == []
        assert T.truncate_frames(ft(piece20(60)), 0.0, 0.0, P, 3.0, offsets=[0], per_piece=per).tolist() == []
    long_piece = [S, B] + sum(([80 + k, B, B, B, B] for k in range(8)), []) + [MARK] + [B] * 7  # 50 frames, 4 s
    lj = T.join_frame_targets([ft(piece20(60), 9), ft(long_piece, 10)])
    off_l = T.truncate_frames(lj, 0.0, 0.0, P, 3.0).tolist()
    assert [c for c in off_l if c < 20] and [c for c in off_l if c > 20]
    assert T.truncate_frames(lj, 0.0, 0.0, P, 3.0, offsets=[0, 20], per_piece=True).tolist() == [
        c for c in off_l if c >= 20]

    # the AED family: the same two pieces as Cohere rows, their cut-table entries at the CTC words' frames
    class Cuts:
        def n_frames(self, i):
            return 20

        def entries(self, i):  # cuts before the 2nd, 3rd and 4th word, keeping 1, 2 and 3 Cohere tokens
            return tuple(np.asarray(x, np.int64) for x in ([3, 7, 11], [3, 7, 11], [6, 10, 14], [1, 2, 3]))

    def tok_row(i):
        tok = np.asarray([10, 11, 12, 13, EOS], np.int64)
        return T._TokRow(np.zeros(20 * FS, np.float32), tok, np.zeros((5, 4), np.int64), np.zeros((5, 4), np.float32),
                         i, 20)

    row = T._join_tok_rows([tok_row(0), tok_row(1)], EOS, np.random.default_rng(0))
    assert row.offsets == offs and row.frames == [20, 20]
    for per, want in ((False, [13, 14, *range(23, 35)]), (True, [])):
        a = T.Augment(seed=1, max_tokens=50, truncate_p=1.0, truncate_min_row_s=3.0, guard_per_piece=per)
        frames, kept, pause, bad = T.aed_cut_candidates(row, Cuts(), a, EOS)
        assert frames.tolist() == want and bad == 0
        assert kept.tolist() == [3 if f < 20 else 4 + (f - 23) // 4 + 1 for f in want]


def test_guard_per_piece_background(tmp_path):
    """background_min_row_s 3, background noise (then background speech) on every row: off, the joined 2 x 1.6 s row
    (3.2 s) gets it, as a 3.2 s single row does, and only the 1.6 s single row is spared; on, the joined row is spared
    too (a piece under 3 s), the single rows as off - the rows before it byte for byte (their draws are unchanged). An
    AED row of the same pieces is spared the same way; the counters' keys stay the chain's."""
    bank = NoiseBank.load(make_noise_bank(tmp_path / "bg", kinds=("music", "speech", "song", "noise")))
    pa, pb = (T._AugRow(voiced(25600, 25600, 3 + i), ft(piece20(60 + 10 * i)), [i], [0]) for i in range(2))
    joined = T._join_rows([pa, pb], np.random.default_rng(0))
    assert joined.offsets == [0, 20] and len(joined.wave) == 51200
    assert T.piece_samples(51200, [0, 20]) == [25600, 25600] and T.piece_samples(25600, [0]) == [25600]
    waves = [voiced(51200, 51200, 1), voiced(25600, 25600, 2), joined.wave]

    def rows():
        return [T._AugRow(waves[0].copy(), ft(piece20(60) * 2), [2], [0]),
                T._AugRow(waves[1].copy(), ft(piece20(60)), [3], [0]),
                T._AugRow(waves[2].copy(), joined.ft, joined.pieces, joined.offsets)]

    for step, changed, spared in ((dict(noise_p=1.0), "noised", "noise_spared"),
                                  (dict(speech_p=1.0, speech_batch_p=0.5), "speech_mixed", "speech_spared")):
        got = {}
        for per in (False, True):
            a = T.Augment(seed=1, background_min_row_s=3.0, guard_per_piece=per, **step)
            rs = rows()
            got[per] = (T.acoustic_chain(rs, [w.copy() for w in waves], bank, None, a, np.random.default_rng(5)), rs)
        (c_off, r_off), (c_on, r_on) = got[False], got[True]
        assert list(c_on) == list(c_off) == ["speech_mixed", "reverbed", "noised", "gained", "clipped", "coded",
                                             "short_rows", "speech_spared", "noise_spared"]
        assert (c_off["short_rows"], c_off[changed], c_off[spared]) == (1, 2, 1)
        assert (c_on["short_rows"], c_on[changed], c_on[spared]) == (2, 1, 2)
        assert [np.array_equal(r.wave, w) for r, w in zip(r_off, waves)] == [False, True, False]
        assert [np.array_equal(r.wave, w) for r, w in zip(r_on, waves)] == [False, True, True]
        assert all(np.array_equal(x.wave, y.wave) for x, y in zip(r_off[:2], r_on[:2]))

    def tok_row(w, i):
        return T._TokRow(w, np.asarray([10, EOS]), np.zeros((2, 4), np.int64), np.zeros((2, 4), np.float32), i,
                         T.ctc_frames(len(w)))

    aed = T._join_tok_rows([tok_row(voiced(25600, 25600, 6), 0), tok_row(voiced(25600, 25600, 7), 1)], EOS,
                           np.random.default_rng(0))
    for per, spared in ((False, 0), (True, 1)):
        a = T.Augment(seed=1, max_tokens=50, noise_p=1.0, background_min_row_s=3.0, guard_per_piece=per)
        c = T.acoustic_chain([aed], [aed.wave], bank, None, a, np.random.default_rng(1))
        assert (c["short_rows"], c["noise_spared"], c["noised"]) == (spared, spared, 1 - spared)


# ------------------------------------------------------------------------------------------ B3: end_trim_voiced


def test_end_trim_voiced():
    """A row '▁ 70 <15 blank frames> 。' whose voice runs to frame 8, 6 frames past the word's spike: off, the trim
    keeps 4 frames (0.32 s: the voiced audio lost, today's trim); on, every frame up to the voiced end, then the mark -
    10 frames, all of the voice kept. A '▁ ... 。' row (no word) off becomes '▁ 。' in 2 frames, on is left alone; a
    frame-0-only word ("▁はい") with 1 s of voice off becomes 2 frames, on keeps its 13 voiced frames; a row whose voice
    runs into its mark is not trimmed. A refused trim draws nothing."""
    col0 = [S, B, 70] + [B] * 15 + [MARK, B]
    wave = voiced(20 * FS, 9 * FS, 1)
    assert T.voiced_end(wave) == 9 * FS
    t = ft(col0, 1)
    assert T.mark_tail(t, P) == T.mark_tail(t, P, SID) == (2, 18, 18)
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    off, today = T._AugRow(wave, t, [0], [0]), T._AugRow(wave, t, [0], [0])
    assert off.trim_end(P, r1, voiced=False, start_ids=SID) and today.trim_end(P, r2) and same_rng(r1, r2)
    assert off.ft.col0().tolist() == today.ft.col0().tolist() == [S, B, 70, MARK]
    assert np.array_equal(off.wave, today.wave) and 4 * FS - T.END_BELOW <= len(off.wave) <= 4 * FS + T.END_ABOVE
    lens = []
    for seed in range(20):
        on = T._AugRow(wave, t, [0], [0])
        assert on.trim_end(P, np.random.default_rng(seed), voiced=True, start_ids=SID) and on.trimmed
        assert on.ft.col0().tolist() == [S, B, 70] + [B] * 6 + [MARK] and on.ft.ctc_ids.tolist() == [S, 70, MARK]
        assert T.ctc_frames(len(on.wave)) == on.ft.n_frames == 10 and np.array_equal(on.wave, wave[:len(on.wave)])
        lens.append(len(on.wave))
    assert min(lens) >= T.voiced_end(wave) and len(set(lens)) > 5  # the end still drawn (end_samples)

    no_word = ft([S] + [B] * 10 + [MARK, B], 2)
    w = voiced(13 * FS, int(0.6 * SR), 2)
    assert T.mark_tail(no_word, P) == (0, 11, 11) and T.mark_tail(no_word, P, SID) is None
    off = T._AugRow(w, no_word, [0], [0])
    assert off.trim_end(P, np.random.default_rng(0)) and off.ft.col0().tolist() == [S, MARK]  # today: '▁ 。' 2 frames
    on, rng = T._AugRow(w, no_word, [0], [0]), np.random.default_rng(0)
    assert not on.trim_end(P, rng, voiced=True, start_ids=SID) and not on.trimmed
    assert on.ft is no_word and on.wave is w and same_rng(rng, np.random.default_rng(0))

    frame0 = ft([45] + [B] * 16 + [MARK, B], 3)  # "▁はい" (a merged start piece: a word) on frame 0, the mark at 17
    w = voiced(19 * FS, SR, 3)  # 1 s of voice in 1.52 s
    off = T._AugRow(w, frame0, [0], [0])
    assert off.trim_end(P, np.random.default_rng(1)) and off.ft.n_frames == 2  # today: 0.16 s
    on = T._AugRow(w, frame0, [0], [0])
    assert on.trim_end(P, np.random.default_rng(1), voiced=True, start_ids=SID)
    assert on.ft.n_frames == 14 and on.ft.ctc_ids.tolist() == [45, MARK] and len(on.wave) >= SR

    full = T._AugRow(voiced(20 * FS, 20 * FS, 4), t, [0], [0])  # the voice runs into the mark's frames
    rng = np.random.default_rng(2)
    assert not full.trim_end(P, rng, voiced=True, start_ids=SID) and same_rng(rng, np.random.default_rng(2))


def test_the_augment_spec():
    """Off by default (from_config of an empty block too); bools only; start_ids below the blank, sorted; an AED
    augmentation takes guard_per_piece but no start_ids, cut_keep_word or end_trim_voiced."""
    a = T.Augment(seed=1)
    assert (a.start_ids, a.cut_keep_word, a.guard_per_piece, a.end_trim_voiced) == ((), False, False, False)
    assert T.Augment.from_config({}, seed=1) == a
    b = T.Augment.from_config({"cut_keep_word": True, "end_trim_voiced": True, "start_ids": [9]}, seed=1,
                              punct_ids=[1], start_ids=[5, 2])
    assert (b.start_ids, b.cut_keep_word, b.end_trim_voiced, b.guard_per_piece) == ((2, 5), True, True, False)
    for bad in (dict(cut_keep_word=1), dict(guard_per_piece="true"), dict(end_trim_voiced=None),
                dict(start_ids=(B,)), dict(start_ids=(-1,)), dict(max_tokens=50, start_ids=(2,)),
                dict(max_tokens=50, cut_keep_word=True), dict(max_tokens=50, end_trim_voiced=True)):
        with pytest.raises(ValueError, match="not an augmentation"):
            T.Augment(seed=1, **bad)
    assert T.Augment(seed=1, max_tokens=50, guard_per_piece=True).guard_per_piece


# --------------------------------------------------------------------------------------------- on a frame store


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A small train frame store of 0.4-4.5 s rows (tests/fixtures.py's tones: voiced to their end)."""
    root = tmp_path_factory.mktemp("v5")
    ev = ["eval_jsut", "eval_cv8", "eval_reazon"]
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (30, "train"), "src_b": (18, "train"),
                                                    **{s: (2, "eval") for s in ev}},
                          dur_range=(0.4, 4.5), token_range=(256, 296), no_second=tuple(ev), seed=41)
    sel = make_fake_selection(fc, greedy_n=4, probe_n=5)
    po = make_fake_parakeet_out(fc, seed=11)
    return T.build_frame_stores(sel, fc.data, po.root, root / "cache" / "ctc_train", ["src_a", "src_b"], ["train"],
                                log=lambda s: None)


def lead_in_targets(monkeypatch):
    """Every stored row's targets as ReazonSpeech's teacher writes them: '▁' on frame 0, an untranscribed lead-in over
    the first half, its words, then "word, 3 blank frames, mark" (its frame count unchanged)."""
    real = T.FrameBatchDataset.targets

    def targets(self, i):
        c = real(self, i).col0().copy()
        if len(c) >= 10:
            c[0], c[1:len(c) // 2] = S, B
            c[-5:] = [300, B, B, B, MARK]
        return ft(c.tolist(), int(i))

    monkeypatch.setattr(T.FrameBatchDataset, "targets", targets)


def micro_batches(st) -> list[list[int]]:
    out = []
    for micro, seed in ((4.0, 0), (12.0, 1)):
        p = T.StepPlanner(st.utts, step_audio_s=24, micro_audio_s=micro, max_dec_len=None, pool_micro=4, seed=seed)
        out += [mb for step in p.epoch_plan(0) for mb in step]
    return out


AUG_KEYS = ["utts", "rows", "concat_groups", "concat_utts", "truncated", "mixed", "cut_padded", "end_padded",
            "end_trimmed", "speech_mixed", "reverbed", "noised", "gained", "clipped", "coded"]  # the counters as today


def test_the_keys_on_a_frame_store(store, monkeypatch):
    """A recipe of joins, cuts and end trims on rows with a lead-in: the counters' keys are today's with every key off
    and with each on; the start pieces alone (the trainer passes them always) change no batch; every row keeps
    ctc_frames(samples) == n_frames. Off, some cut rows keep '▁' alone (or a joined row's last piece does); with
    cut_keep_word none does. The fixture rows are voiced to their end, so end_trim_voiced trims none of them."""
    lead_in_targets(monkeypatch)
    kw = dict(seed=3, truncate_p=0.7, truncate_min_s=0.3, punct_ids=P, concat_p=0.5, end_trim_p=0.7)
    mbs = micro_batches(store)
    plain = T.dataset_for(store, augment=T.Augment(**kw))
    sums = {}
    for name, extra in (("off", {}), ("start_ids", dict(start_ids=SID)),
                        *((k, {k: True, "start_ids": SID}) for k in FIXES),
                        ("all", dict(start_ids=SID, **{k: True for k in FIXES}))):
        ds = T.dataset_for(store, augment=T.Augment(**kw, **extra))
        n_bare = trimmed = 0
        for idx in mbs:
            got = ds[idx]
            assert list(got["aug"]) == AUG_KEYS, name
            if name == "start_ids":
                ref = plain[idx]
                assert all(torch.equal(v, ref[k]) if isinstance(v, torch.Tensor) else v == ref[k]
                           for k, v in got.items())
            lens = got["ctc_target_lengths"].tolist()
            for b in range(len(lens)):
                assert T.ctc_frames(int(got["lengths"][b])) == int(got["n_frames"][b])
                ids = got["ctc_targets"][sum(lens[:b]):sum(lens[:b + 1])].tolist()
                last = ids[len(ids) - ids[::-1].index(S):] if S in ids else ids  # the row's last piece
                n_bare += not [i for i in last if i not in (S, MARK)]
            trimmed += got["aug"]["end_trimmed"]
        sums[name] = (n_bare, trimmed)
    assert sums["off"][0] > 0 and sums["off"][1] > 0 and sums["start_ids"] == sums["off"]
    assert sums["cut_keep_word"][0] == 0 and sums["all"][0] == 0
    assert sums["end_trim_voiced"][1] == 0 and sums["all"][1] == 0  # tones to their end: no silent frame to drop
    assert sums["guard_per_piece"] == sums["off"]  # truncate_min_row_s and background_min_row_s 0: nothing short


# ----------------------------------------------------------------------------------------------------- the trainer

CTC = ["family=ctc", "parakeet_root=po"]


def test_the_keys_in_the_trainer_config():
    """DEFAULTS: the three false, at the block's end. Validated as bools (validate's bool rule and validate_augment's
    own, for a block checked alone); on an AED student cut_keep_word and end_trim_voiced are refused when the
    augmentation is on (its cut table and its end trim need neither), guard_per_piece taken. A resume may change them
    (not RESUME_FIXED)."""
    m = load_script("04_distill")
    assert list(m.DEFAULTS["augment"])[-3:] == list(FIXES) and m.AUDIT_FIX_KEYS == FIXES
    assert all(m.DEFAULTS["augment"][k] is False for k in FIXES)
    on = m.load_config(None, CTC + ["augment.enabled=true", *(f"augment.{k}=true" for k in FIXES)])
    assert all(on["augment"][k] is True for k in FIXES)
    for k in FIXES:
        for bad in ("yes", "1", "null", '"true"'):
            with pytest.raises(SystemExit, match=f"augment.{k} must be true or false"):
                m.load_config(None, CTC + [f"augment.{k}={bad}"])
        cfg = m.load_config(None, CTC)
        cfg["augment"][k] = 1
        with pytest.raises(SystemExit, match=f"augment.{k} must be true or false, got 1"):
            m.validate_augment(cfg)
    for k in ("cut_keep_word", "end_trim_voiced"):
        with pytest.raises(SystemExit, match=f"augment.{k} is a CTC student's fix"):
            m.load_config(None, ["augment.enabled=true", "augment.concat_p=0.5", f"augment.{k}=true"])
        assert m.load_config(None, [f"augment.{k}=true"])["augment"][k] is True  # off: checked when it is turned on
    assert m.load_config(None, ["augment.enabled=true", "augment.guard_per_piece=true"])["augment"]["guard_per_piece"]
    saved = m.load_config(None, CTC)
    changed, same = m.resume_overrides(saved, [("augment.cut_keep_word", True), ("augment.guard_per_piece", True),
                                               ("augment.end_trim_voiced", False)])
    assert changed == {"augment.cut_keep_word": True, "augment.guard_per_piece": True}
    assert same == {"augment.end_trim_voiced": False}
    assert not any(f.startswith("augment") for f in m.RESUME_FIXED)


def fake_run(m, sets: list[str]):
    """setup_augment's inputs without a model: the config, a dataset of three durations whose with_augment returns
    the spec, the tiny tokenizer, a logger that keeps its events."""
    events = []

    class DS:
        duration = np.array([1.0, 2.5, 4.0], np.float32)

        def __len__(self):
            return 3

        def with_augment(self, spec, cuts=None, noise=None, rirs=None):
            return spec

    R = SimpleNamespace(cfg=m.load_config(None, sets), ds=DS(), tokenizer=tiny_tokenizer(),
                        train=SimpleNamespace(info={}),
                        log=SimpleNamespace(event=lambda kind, **kw: events.append(dict(kind=kind, **kw))))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        spec = m.setup_augment(R)
    (event,) = [e for e in events if e["kind"] == "augment"]
    return spec, event, out.getvalue()


# the CTC `augment` event's fields before the fixes (scripts/04_distill.py setup_augment and noise_fields at d599b2e)
EVENT_KEYS = {"kind", "seed", "truncate_p", "truncate_min_frac", "truncate_min_s", "truncate_pause_p", "punct_ids",
              "concat_p", "concat_max_n", "concat_max_s", "concat_max_s_config", "longest_train_s",
              "concat_max_s_clamped", "mix_p", "mix_snr_db", "truncate_pad_p", "end_pad_p", "pad_frames",
              "truncate_min_row_s", "end_trim_p", "noise_p", "noise_snr_db", "noise_bank", "noise_bank_sha256",
              "noise_clips", "noise_hours", "noise_kind_hours", "speech_p", "speech_snr_db", "speech_talkers",
              "speech_batch_p", "reverb_p", "rir_bank", "rir_bank_sha256", "rir_clips", "gain_p", "gain_db", "codec_p",
              "codecs"}


def test_the_augment_event_and_line():
    """start_token_ids reads the bare start piece (the tiny tokenizer's id 2, as Parakeet's); setup_augment hands it to
    every CTC augmentation. With the fixes off the `augment` event has exactly its fields before them and the console
    line no word of them; on, the event adds each key that is on (and start_ids when a fix that reads them is), the
    line a clause each. An AED student's event and line take guard_per_piece the same way."""
    m = load_script("04_distill")
    tok = tiny_tokenizer()
    assert m.start_token_ids(tok) == {2: "▁"} and 2 not in m.punct_token_ids(tok)
    base = CTC + ["augment.enabled=true", "augment.truncate_p=0.2", "augment.concat_p=0.5", "augment.end_trim_p=0.3"]
    spec, ev, line = fake_run(m, base)
    assert set(ev) == EVENT_KEYS and spec.start_ids == (2,)
    assert not any(k for k in FIXES if getattr(spec, k)) and "keeps a word" not in line and "per joined" not in line
    spec, ev, line = fake_run(m, base + [f"augment.{k}=true" for k in FIXES])
    assert set(ev) == EVENT_KEYS | {*FIXES, "start_ids"} and all(ev[k] is True for k in FIXES)
    assert ev["start_ids"] == {"2": "▁"} and all(getattr(spec, k) for k in FIXES)
    assert ("every cut keeps a word, row guards per joined utterance, end trim keeps the voiced audio, seed 1234"
            in line)
    spec, ev, line = fake_run(m, base + ["augment.guard_per_piece=true"])
    assert set(ev) == EVENT_KEYS | {"guard_per_piece"} and "row guards per joined utterance, seed" in line
    aed = ["augment.enabled=true", "augment.concat_p=0.5"]
    spec, ev, line = fake_run(m, aed)
    assert spec.max_tokens is not None and not set(FIXES) & set(ev) and spec.start_ids == ()
    spec, ev, line = fake_run(m, aed + ["augment.guard_per_piece=true"])
    assert ev["guard_per_piece"] is True and spec.guard_per_piece and "row guards per joined utterance" in line


def test_the_keys_are_resume_sets():
    """fullrun: the three are bool resume sets (appended, so the earlier keys keep their order), spelled as JSON
    bools for the trainer's --set - in KITSUNE_RESUME_SETS and in a registry continuation's sets alike - and those
    words load on a CTC config."""
    assert fr.RESUME_SET_KEYS[-3:] == tuple(f"augment.{k}" for k in FIXES)
    assert all(fr.RESUME_SET_KINDS[f"augment.{k}"] == "bool" for k in FIXES)
    assert all(fr.resume_set_rule(f"augment.{k}") == "true or false" for k in FIXES)
    got = fr.parse_resume_sets(f"{RID}:augment.cut_keep_word=True,{RID}:augment.guard_per_piece=FALSE,"
                               f"{RID}:augment.end_trim_voiced=true")
    assert got == {RID: ["augment.cut_keep_word=true", "augment.guard_per_piece=false",
                         "augment.end_trim_voiced=true"]}
    for bad in ("yes", "1", ""):
        with pytest.raises(ValueError, match="augment.cut_keep_word must be true or false"):
            fr.parse_resume_sets(f"{RID}:augment.cut_keep_word={bad}")
    assert fr.continue_sets({"sets": {f"augment.{k}": True for k in FIXES}}) == [f"augment.{k}=true" for k in FIXES]
    with pytest.raises(ValueError, match="augment.end_trim_voiced must be true or false"):
        fr.continue_set_text("augment.end_trim_voiced", 1)
    m = load_script("04_distill")
    cfg = m.load_config(None, CTC + got[RID])
    assert (cfg["augment"]["cut_keep_word"], cfg["augment"]["guard_per_piece"], cfg["augment"]["end_trim_voiced"]) == (
        True, False, True)


def test_recipe_v5():
    """RECIPE_V5 = recipe v4 with the three fixes on, proposed for the next run (DECISIONS pending): no full run,
    continuation or generated config uses it; it loads on a CTC student's config and is refused on an AED one (its
    cut table and its end trim need neither CTC fix)."""
    assert M.RECIPE_V5 == dict(M.RECIPE_V4, cut_keep_word=True, guard_per_piece=True, end_trim_voiced=True)
    assert list(M.RECIPE_V5)[:len(M.RECIPE_V4)] == list(M.RECIPE_V4)
    assert all(r["augment"] is not M.RECIPE_V5 for r in M.FULL_RUNS.values())
    assert all(c["augment"] is not M.RECIPE_V5 for c in M.CONTINUATIONS.values())
    m = load_script("04_distill")
    full = ROOT / "configs" / "full"
    sets = [f"augment.{k}={json.dumps(v)}" for k, v in M.RECIPE_V5.items()]
    c = m.load_config(str(full / "full-p01.json"), sets)
    assert c["augment"] == dict(m.DEFAULTS["augment"], **M.RECIPE_V5)
    with pytest.raises(SystemExit, match="is a CTC student's fix"):
        m.load_config(str(full / "full-t06.json"), sets)
    for name in ("full-t06", "full-p03", "full-p01", "full-p005"):
        block = json.loads((full / f"{name}.json").read_text(encoding="utf-8")).get("augment") or {}
        assert not set(FIXES) & set(block), name
