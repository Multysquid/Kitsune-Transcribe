"""kitsune/parakeet_targets.py: the stored Parakeet format round-trips, and every invariant a trainer relies on is
checked. CPU only, numpy only; utterances are synthetic but obey the decoder's rules (a blank always advances, the guard
forces a step of 1, decoding runs past the last frame)."""
import json
import math
import re
import zipfile

import numpy as np
import pytest

from kitsune import parakeet_targets as pt
from kitsune.store import ids_sha256

SETTINGS = dict(k_tdt=4, k_ctc=3, max_symbols=10, ctc_dense_thr=0.95)
META = dict(format_version=1, **SETTINGS)


def _topk(rng, n, k, first=None):
    """n rows of k distinct classes (column 0 = `first` when given) and descending log-probs."""
    rows = []
    for r in range(n):
        head = [int(first[r])] if first is not None else []
        rest = [int(c) for c in rng.choice(pt.VOCAB, size=k + 1, replace=False) if int(c) not in head]
        rows.append((head + rest)[:k])
    idx = np.array(rows, dtype=np.int16).reshape(n, k)
    lp = np.sort(np.log(rng.dirichlet(np.ones(k + 2), size=n)[:, :k]), axis=1)[:, ::-1].reshape(n, k)
    return idx, lp.astype(np.float32)


def make_utt(rng, uid, T, *, n_dense=None, empty_hyp=False, force_guard=False):
    """A valid decoder trace over T frames."""
    frames, durs, forced, emitted = [], [], [], []
    t, sym = 0, 0
    while t < T:
        blank = empty_hyp or (not force_guard and rng.random() < 0.4)
        tok = pt.BLANK if blank else int(rng.integers(0, pt.BLANK))
        d = int(rng.integers(0, 5))
        if force_guard and not blank:
            d = 0
        if blank and d == 0:
            d = 1
        sym = sym + 1 if d == 0 else 0
        f = sym >= 10
        if f:
            d, sym = 1, 0
        frames.append(t)
        durs.append(d)
        forced.append(f)
        emitted.append(tok)
        t += d
    S = len(frames)
    idx, lp = _topk(rng, S, SETTINGS["k_tdt"], first=emitted)
    dur_lp = np.log(rng.dirichlet(np.ones(5), size=S)).astype(np.float32).reshape(S, 5)
    blank_lp = np.log(rng.uniform(0.5, 1.0, size=T)).astype(np.float32)
    if n_dense == 0:
        blank_lp = np.zeros(T, np.float32)
    # keep clear of the threshold so fp16 rounding cannot move a frame across it
    blank_lp = np.where(np.abs(blank_lp - math.log(0.95)) < 0.01, -0.2, blank_lp).astype(np.float32)
    dense = np.where(blank_lp < math.log(0.95))[0]
    cidx, clp = _topk(rng, len(dense), SETTINGS["k_ctc"])
    return dict(id=uid, duration=T * 0.08, n_frames=T, truncated=False, step_frame=np.array(frames),
                step_dur=np.array(durs), step_forced=np.array(forced), tdt_topk_idx=idx, tdt_topk_lp=lp,
                tdt_dur_lp=dur_lp, ctc_blank_lp=blank_lp, ctc_dense_frame=dense, ctc_topk_idx=cidx, ctc_topk_lp=clp)


def make_shard(tmp_path, utts, shard_ids, name="train-00000"):
    arrays = pt.pack_shard(utts, settings=SETTINGS, n_scanned=len(shard_ids), shard_ids_sha=ids_sha256(shard_ids))
    npz = tmp_path / f"{name}.npz"
    np.savez_compressed(npz, **arrays)
    with open(tmp_path / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for u in utts:
            f.write(json.dumps(dict(id=u["id"])) + "\n")
    return npz, arrays


@pytest.fixture
def shard(tmp_path):
    rng = np.random.default_rng(0)
    utts = [make_utt(rng, "a", 40), make_utt(rng, "c", 1, n_dense=0), make_utt(rng, "d", 12, empty_hyp=True),
            make_utt(rng, "e", 5, force_guard=True)]
    shard_ids = ["a", "b", "c", "d", "e"]  # b was skipped (>30 s or undecodable)
    npz, arrays = make_shard(tmp_path, utts, shard_ids)
    return npz, arrays, utts, shard_ids


def test_round_trip_is_exact_including_edge_cases(shard):
    """A 1-frame utterance with no dense frames, an empty hypothesis and a guard-bounded one pack and load back to the
    same values (fp16 fields compare after the same cast)."""
    npz, arrays, utts, shard_ids = shard
    sh = pt.load_shard(npz)
    assert sh.n == 4 and sh.ids == ["a", "c", "d", "e"] and sh.settings == SETTINGS
    for key, val in arrays.items():
        np.testing.assert_array_equal(sh.z[key], val, err_msg=key)
    for i, u in enumerate(utts):
        got = sh.utt(i)
        np.testing.assert_array_equal(got["frames"], u["step_frame"])
        np.testing.assert_array_equal(got["dur"], u["step_dur"])
        np.testing.assert_array_equal(got["tdt_idx"], u["tdt_topk_idx"])
        np.testing.assert_array_equal(got["tdt_lp"], u["tdt_topk_lp"].astype(np.float16).astype(np.float32))
        np.testing.assert_array_equal(got["tokens"], pt.utt_tokens(u))
    assert len(sh.utt(2)["tokens"]) == 0
    assert sh.utt(1)["frames"].tolist() == [0]
    assert sh.z["dense_offsets"][2] - sh.z["dense_offsets"][1] == 0
    assert sh.utt(3)["forced"].any()


def test_frames_are_the_cumsum_of_durations_and_u_counts_prior_tokens(shard):
    npz, _, utts, _ = shard
    sh = pt.load_shard(npz)
    for i in range(sh.n):
        u = sh.utt(i)
        assert u["frames"].tolist() == [0] + np.cumsum(u["dur"])[:-1].tolist()
        emitted = u["tdt_idx"][:, 0] != pt.BLANK
        assert u["u"].tolist() == [int(emitted[:s].sum()) for s in range(len(emitted))]
        assert int(u["u"][-1] + emitted[-1]) == len(u["tokens"])


def test_ctc_expansion(shard):
    """Blank-only frames expand to [blank, -1...] / [blank_lp, -inf...]; dense frames carry their stored top-k."""
    npz, _, utts, _ = shard
    sh = pt.load_shard(npz)
    for i, u in enumerate(utts):
        idx, lp = sh.ctc(i)
        T = u["n_frames"]
        assert idx.shape == (T, SETTINGS["k_ctc"]) and idx.dtype == np.int64 and lp.dtype == np.float32
        dense = set(u["ctc_dense_frame"].tolist())
        for t in range(T):
            if t in dense:
                j = u["ctc_dense_frame"].tolist().index(t)
                assert idx[t].tolist() == u["ctc_topk_idx"][j].tolist()
            else:
                assert idx[t].tolist() == [pt.BLANK, -1, -1]
                assert lp[t, 0] == np.float32(np.float16(u["ctc_blank_lp"][t])) and np.all(np.isneginf(lp[t, 1:]))


def test_ctc_greedy_collapses_repeats_then_drops_blanks():
    B = pt.BLANK
    assert pt.ctc_greedy([B, 5, 5, B, 5, 7, 7, B]) == [5, 5, 7]
    assert pt.ctc_greedy([]) == []
    assert pt.ctc_greedy(pt.ctc_col0(4, np.array([1, 2]), np.array([[9, 1], [9, 2]]))) == [9]


def test_check_shard_accepts_a_sound_shard(shard):
    npz, _, _, shard_ids = shard
    assert pt.check_shard(npz, META, shard_ids) == []


def _tamper(tmp_path, arrays, **changes):
    a = dict(arrays)
    a.update(changes)
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, **a)
    return p


def test_check_shard_flags_every_broken_invariant(shard, tmp_path):
    npz, arrays, _, shard_ids = shard

    def problems(**changes):
        return pt.check_shard(_tamper(tmp_path, arrays, **changes), META, shard_ids)

    so = arrays["step_offsets"].copy()
    so[1], so[2] = so[2], so[1]
    assert any("step_offsets" in p for p in problems(step_offsets=so))
    sf = arrays["step_frame"].copy()
    sf[1] += 1
    assert any("cumsum" in p or "outside" in p for p in problems(step_frame=sf))
    sd = arrays["step_dur"].copy()
    sd[:] = 0
    assert problems(step_dur=sd)
    tok = arrays["tokens"].copy()
    tok[0] = (tok[0] + 1) % pt.BLANK
    assert any("tokens" in p for p in problems(tokens=tok))
    lp = arrays["tdt_topk_lp"].copy()
    lp[0] = lp[0][::-1]
    assert any("sorted" in p for p in problems(tdt_topk_lp=lp))
    bl = arrays["ctc_blank_lp"].copy()
    bl[:] = 0
    assert any("dense" in p for p in problems(ctc_blank_lp=bl))
    df = arrays["ctc_dense_frame"].copy()
    if len(df):
        df[0] = 999
        assert any("dense" in p for p in problems(ctc_dense_frame=df))
    assert any("settings" in p for p in problems(k_ctc=np.int64(5)))
    assert any("sha256" in p for p in problems(shard_ids_sha256=np.array("0" * 64)))
    assert any("subsequence" in p for p in problems(ids=np.array(["e", "c", "d", "a"])))
    assert any("n_scanned" in p for p in problems(n_scanned=np.int64(4)))
    assert any("format_version" in p for p in problems(format_version=np.int64(2)))
    assert pt.check_shard(npz, dict(META, max_symbols=0), shard_ids)  # meta says guard off


def test_shard_done_is_id_based(shard, tmp_path):
    """Equal row counts are not enough: other ids, a changed setting or a missing jsonl are not done."""
    npz, _, utts, shard_ids = shard
    assert pt.shard_done(npz, shard_ids, SETTINGS)
    assert pt.shard_done(npz, shard_ids, dict(SETTINGS, max_symbols=10.0))
    assert not pt.shard_done(npz, ["a", "b", "c", "d", "x"], SETTINGS)  # same count, other ids
    assert not pt.shard_done(npz, shard_ids, dict(SETTINGS, k_tdt=8))
    assert not pt.shard_done(npz, shard_ids, dict(SETTINGS, ctc_dense_thr=0.9))
    assert not pt.shard_done(tmp_path / "missing.npz", shard_ids, SETTINGS)
    npz.with_suffix(".jsonl").write_text(json.dumps(dict(id="a")) + "\n", encoding="utf-8")
    assert not pt.shard_done(npz, shard_ids, SETTINGS)  # jsonl ends do not match
    npz.with_suffix(".jsonl").unlink()
    assert not pt.shard_done(npz, shard_ids, SETTINGS)
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    assert not pt.shard_done(tmp_path / "junk.npz", shard_ids, SETTINGS)


def test_empty_shard_packs_and_is_done(tmp_path):
    """Every row skipped: fixed-shape empty arrays, and the shard still counts as covering its data shard."""
    npz, arrays = make_shard(tmp_path, [], ["x", "y"])
    assert arrays["tdt_topk_idx"].shape == (0, SETTINGS["k_tdt"]) and arrays["ctc_topk_lp"].shape == (0, SETTINGS["k_ctc"])
    assert pt.load_shard(npz).n == 0
    assert pt.check_shard(npz, META, ["x", "y"]) == []
    assert pt.shard_done(npz, ["x", "y"], SETTINGS)


def test_dtypes_are_the_documented_ones(shard):
    _, arrays, _, _ = shard
    want = dict(step_frame=np.int16, step_dur=np.int8, step_forced=np.bool_, tdt_topk_idx=np.int16,
                tdt_topk_lp=np.float16, tdt_dur_lp=np.float16, tokens=np.int16, ctc_blank_lp=np.float16,
                ctc_dense_frame=np.int16, ctc_topk_idx=np.int16, ctc_topk_lp=np.float16, n_frames=np.int32,
                duration=np.float32, truncated=np.bool_, step_offsets=np.int64, k_tdt=np.int64,
                ctc_dense_thr=np.float64, format_version=np.int64, n_scanned=np.int64)
    for key, dt in want.items():
        assert arrays[key].dtype == dt, key
    assert arrays["shard_ids_sha256"].dtype.kind == "U" and arrays["ids"].dtype.kind == "U"


def test_docstring_names_every_key_the_writer_emits(shard):
    """The module docstring is the format's only definition: it must name every array pack_shard writes."""
    _, arrays, _, _ = shard
    fmt = pt.__doc__.split("FORMAT (npz")[1].split("Tail mass")[0]
    names = set(re.findall(r"[a-z][a-z_0-9]*", fmt))
    assert set(arrays) <= names, set(arrays) - names


def test_write_shard_writes_the_pair_compressed(shard, tmp_path):
    """write_shard goes through labelpass.write_pair (jsonl then npz, fsynced) with compression on."""
    pytest.importorskip("kitsune.labelpass")
    _, arrays, utts, _ = shard
    out = tmp_path / "out" / "src"
    pt.write_shard(out, "train-00000", arrays, [dict(id=u["id"]) for u in utts])
    npz = out / "train-00000.npz"
    assert npz.exists() and (out / "train-00000.jsonl").exists()
    with zipfile.ZipFile(npz) as zf:
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())
    sh = pt.load_shard(npz)
    assert sh.ids == ["a", "c", "d", "e"]


def _load_02p():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "02p_parakeet_pass.py"
    spec = importlib.util.spec_from_file_location("parakeet_pass_02p", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_02p_refuses_an_unpinned_model_dir_and_a_changed_meta(tmp_path, monkeypatch):
    """A model dir that fails the sha256 pins exits 3 before anything loads; an existing meta.json written with other
    settings exits 65 (never mixed settings in one root)."""
    pytest.importorskip("kitsune.parakeet")
    mod = _load_02p()
    out = tmp_path / "parakeet_out"
    monkeypatch.setattr(mod, "verify_model_dir", lambda d: ["model.safetensors: sha256 mismatch"])
    with pytest.raises(SystemExit) as e:
        mod.main(["--data", str(tmp_path / "data"), "--out", str(out), "--model-dir", str(tmp_path / "m")])
    assert e.value.code == 3 and not out.exists()

    monkeypatch.setattr(mod, "verify_model_dir", lambda d: [])
    mod.main(["--data", str(tmp_path / "data"), "--out", str(out), "--model-dir", str(tmp_path / "m")])  # no shards
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["k_tdt"] == 8 and meta["max_symbols"] == 10 and meta["blank"] == pt.BLANK
    with pytest.raises(SystemExit) as e:
        mod.main(["--data", str(tmp_path / "data"), "--out", str(out), "--model-dir", str(tmp_path / "m"), "--k-tdt", "4"])
    assert e.value.code == 65
