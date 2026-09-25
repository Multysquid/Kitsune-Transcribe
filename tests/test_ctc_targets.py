"""Frame targets of the CTC students (kitsune/ctc_targets.py): loading parakeet_out shards, collation, and the CTC path
against what the label pass wrote. The synthetic shards come from tests/fixtures_ctc.py; the real-data test reads a
parakeet_out root (KITSUNE_PARAKEET_OUT, default <real data root>/parakeet_out) and skips when there is none."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixtures import REAL, make_fake_corpus, need_real  # noqa: E402
from fixtures_ctc import (make_fake_parakeet_out, synthetic_frame_targets, synthetic_log_probs,  # noqa: E402
                          tiny_tokenizer)

from kitsune import ctc_student as CS  # noqa: E402
from kitsune import parakeet_targets as pt  # noqa: E402
from kitsune.ctc_targets import (FrameTargets, collate_frame_targets, frame_targets, infeasible,  # noqa: E402
                                 load_ctc_targets, targets_from_log_probs)
from kitsune.parakeet import ctc_collapse, ctc_targets  # noqa: E402

BLANK = CS.CTC_BLANK


def jsonl_rows(npz: Path) -> dict[str, dict]:
    lines = npz.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in (json.loads(line) for line in lines if line.strip())}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("ctc_targets")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (40, "train"), "eval_x": (9, "eval")},
                          teacher_skipped={"src_a": 2}, missing_audio={"src_a": 1})
    return fc, make_fake_parakeet_out(fc, seed=3)


def test_fixture_shards_are_in_the_label_pass_format(fake):
    fc, po = fake
    meta = json.loads((po.root / "meta.json").read_text(encoding="utf-8"))
    assert meta["blank"] == BLANK and meta["vocab"] == CS.CTC_VOCAB and meta["k_ctc"] == 8
    npzs = sorted(po.root.glob("*/*.npz"))
    assert len(npzs) == 3  # src_a: 2 shards of 32 rows, eval_x: 1
    for npz in npzs:
        stem_rows = [u for u in fc.utts.values() if u.source == npz.parent.name and u.stem == npz.stem]
        assert pt.check_shard(npz, meta, [u.id for u in stem_rows]) == []


def test_load_ctc_targets_reads_what_was_written(fake):
    fc, po = fake
    seen = 0
    for npz in sorted(po.root.glob("*/*.npz")):
        got = load_ctc_targets(npz)
        rows = [json.loads(line) for line in npz.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
        assert list(got) == [r["id"] for r in rows]  # shard order
        for uid, ft in got.items():
            want = po.targets[uid]
            assert ft.n_frames == want.n_frames
            for name, dtype in (("blank_lp", np.float16), ("dense_frame", np.int32), ("topk_idx", np.int16),
                                ("topk_lp", np.float16), ("ctc_ids", np.int32)):
                a = getattr(ft, name)
                assert a.dtype == dtype, name
                assert np.array_equal(a, getattr(want, name)), (uid, name)
            seen += 1
    assert seen == len(po.targets) == len(fc.ids())  # every teacher-labelled row, incl. the one whose audio is gone


def test_ctc_ids_equal_the_jsonl_ctc_hyp_and_the_greedy_path(fake):
    _, po = fake
    tok = tiny_tokenizer()
    for npz in sorted(po.root.glob("*/*.npz")):
        rows = jsonl_rows(npz)
        for uid, ft in load_ctc_targets(npz).items():
            assert CS.decode_ids(tok, ft.ctc_ids) == rows[uid]["ctc_hyp"]
            assert rows[uid]["n_frames"] == ft.n_frames
            assert ft.ctc_ids.tolist() == pt.ctc_greedy(pt.ctc_col0(ft.n_frames, ft.dense_frame, ft.topk_idx))


def test_n_frames_align_with_the_audio(fake):
    """The fixture's frame counts come from the rows' audio, as the real label pass's do (the frame preflight)."""
    _, po = fake
    for uid, ft in po.targets.items():
        assert ft.n_frames == CS.expected_n_frames(po.n_samples[uid]) > 0


def test_targets_from_log_probs_is_the_label_pass_path():
    rng = np.random.default_rng(0)
    lps = [synthetic_log_probs(n, rng) for n in (30, 12)]
    lp = torch.zeros(2, 30, CS.CTC_VOCAB)
    lp[0], lp[1, :12] = lps[0], lps[1]
    got = targets_from_log_probs(lp, torch.tensor([30, 12]))
    ref = ctc_targets(lp, torch.tensor([30, 12]), k_ctc=8, dense_thr=0.95)
    for b, ft in enumerate(got):
        assert ft.n_frames == [30, 12][b]
        assert np.array_equal(ft.blank_lp, ref["ctc_blank_lp"][b].astype(np.float16))
        assert np.array_equal(ft.dense_frame, ref["ctc_dense_frame"][b])
        assert np.array_equal(ft.topk_idx, ref["ctc_topk_idx"][b].astype(np.int16))
        assert ft.ctc_ids.tolist() == ref["ctc_tokens"][b].tolist() == ctc_collapse(lps[b].argmax(-1).tolist(), BLANK)
        assert np.array_equal(ft.col0(), lps[b].argmax(-1).numpy())
        assert 0 < len(ft.dense_frame) < ft.n_frames


def test_collate_pads_and_widens():
    rng = np.random.default_rng(1)
    tg = [synthetic_frame_targets(n, rng) for n in (17, 5, 11)]
    b = collate_frame_targets(tg)
    B, T, k = 3, 17, 8
    assert b["frame_mask"].shape == b["dense_mask"].shape == b["blank_lp"].shape == (B, T)
    assert b["topk_idx"].shape == b["topk_lp"].shape == (B, T, k)
    assert b["frame_mask"].dtype == b["dense_mask"].dtype == torch.bool and b["blank_lp"].dtype == torch.float32
    assert (b["topk_idx"].dtype, b["topk_lp"].dtype) == (torch.long, torch.float32)
    assert b["n_frames"].tolist() == [17, 5, 11] and b["n_frames"].dtype == torch.long
    assert b["frame_mask"].sum(1).tolist() == [17, 5, 11]
    assert not bool((b["dense_mask"] & ~b["frame_mask"]).any())
    for i, t in enumerate(tg):
        d = torch.from_numpy(t.dense_frame.astype(np.int64))
        assert b["dense_mask"][i].nonzero()[:, 0].tolist() == d.tolist()
        assert torch.equal(b["topk_idx"][i, d], torch.from_numpy(t.topk_idx.astype(np.int64)))
        assert torch.equal(b["topk_lp"][i, d], torch.from_numpy(t.topk_lp.astype(np.float32)))
        assert torch.equal(b["blank_lp"][i, :t.n_frames], torch.from_numpy(t.blank_lp.astype(np.float32)))
        off = ~b["dense_mask"][i]
        assert bool((b["topk_idx"][i][off] == 0).all()) and bool(torch.isinf(b["topk_lp"][i][off]).all())
        assert bool((b["blank_lp"][i, t.n_frames:] == 0).all())
    assert b["ctc_target_lengths"].tolist() == [len(t.ctc_ids) for t in tg]
    assert b["ctc_targets"].tolist() == [x for t in tg for x in t.ctc_ids.tolist()]
    wide = collate_frame_targets(tg, t_max=23)
    assert wide["frame_mask"].shape == (3, 23) and torch.equal(wide["frame_mask"][:, :17], b["frame_mask"])
    with pytest.raises(ValueError, match="t_max"):
        collate_frame_targets(tg, t_max=16)
    with pytest.raises(ValueError, match="mixed"):
        collate_frame_targets([tg[0], synthetic_frame_targets(4, rng, k=4)])


def test_frame_targets_validates_lengths():
    with pytest.raises(ValueError, match="blank log-probs"):
        FrameTargets(3, np.zeros(2, np.float16), np.zeros(0, np.int32), np.zeros((0, 8), np.int16),
                     np.zeros((0, 8), np.float16), np.zeros(0, np.int32))
    ft = frame_targets(0, [], [], np.zeros((0, 8)), np.zeros((0, 8)))
    assert ft.n_frames == 0 and ft.ctc_ids.tolist() == [] and ft.k == 8
    assert collate_frame_targets([ft, ft])["frame_mask"].shape == (2, 0)


def test_infeasible():
    assert not infeasible([1, 2, 3], 3) and infeasible([1, 1], 2) and not infeasible([1, 1], 3)
    assert not infeasible([], 0)


# ------------------------------------------------------------------------------------------------ real shards


def _real_parakeet_out() -> Path:
    return Path(os.environ.get("KITSUNE_PARAKEET_OUT") or REAL / "parakeet_out")


def test_real_shards_ctc_path_equals_ctc_hyp_and_frames_equal_the_audio():
    """On stored parakeet_out shards: decode(ctc_ids) == jsonl ctc_hyp on every row (K3), and, where the data shard
    is on this machine, expected_n_frames(audio) == the stored n_frames on every row (K4's arithmetic)."""
    root = _real_parakeet_out()
    model_dir = Path(os.environ.get("KITSUNE_PARAKEET_DIR") or REAL / "cache" / "parakeet-tdt_ctc-0.6b-ja-hf")
    need_real(root / "meta.json", model_dir / "tokenizer.json")
    from transformers import AutoTokenizer

    from kitsune.audio import decode_audio
    from kitsune.store import read_shard

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    n_rows = n_audio = 0
    for npz in sorted(root.glob("*/*.npz")):
        rows = jsonl_rows(npz)
        targets = load_ctc_targets(npz)
        for uid, ft in targets.items():
            assert CS.decode_ids(tok, ft.ctc_ids) == rows[uid]["ctc_hyp"], uid
            assert ft.n_frames == rows[uid]["n_frames"]
            n_rows += 1
        shard = REAL / "data" / "shards" / npz.parent.name / f"{npz.stem}.parquet"
        if shard.exists():
            t = read_shard(shard, columns=["id", "audio"])
            for uid, audio in zip(t.column("id").to_pylist(), t.column("audio").to_pylist()):
                if uid in targets:
                    assert CS.expected_n_frames(len(decode_audio(audio))) == targets[uid].n_frames, uid
                    n_audio += 1
    assert n_rows > 0
    print(f"\n{n_rows} rows: ctc path == ctc_hyp; {n_audio} rows: expected_n_frames == n_frames")
