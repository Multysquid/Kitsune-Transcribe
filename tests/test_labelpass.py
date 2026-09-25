"""The shared label-pass runner (kitsune/labelpass.py) and the teacher pass built on it (scripts/02_teacher_pass.py).

CPU only, no model: the runner is driven with fake callbacks that record the order of events, and 02's done check,
packing and adoption run on tests/fixtures.py's fake corpus. The runner's promises are what keeps a label box honest:
the GPU sees one stream across shard boundaries, a shard's npz never lands before its jsonl, an OOM loses no row, and
a follow-mode lane neither misses a late shard nor exits while one is still coming.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import _write_teacher, load_script, make_fake_corpus  # noqa: E402

from kitsune import labelpass  # noqa: E402
from kitsune.labelpass import ShardJob, ordered_subsequence, partition_ok, run, write_pair  # noqa: E402
from kitsune.store import ShardInfo, read_manifest, shard_ids  # noqa: E402

tp = load_script("02_teacher_pass")


def _job(tmp_path, stem, rows=6, source="s"):
    return ShardJob(ShardInfo(f"shards/{source}/{stem}.parquet", source, "train", rows, rows / 3600), stem,
                    tmp_path / "out" / source)


def _runner(events, n_rows=6, batch=2, oom_above=None, oom_calls=None):
    """Fake callbacks: a table is its row list, batches are consecutive pairs, gpu returns the row index doubled."""
    def load_table(job):
        events.append(("load", job.stem))
        return dict(stem=job.stem, rows=list(range(job.info.rows)))

    def plan_batches(job, t):
        return [np.arange(i, min(i + batch, len(t["rows"]))) for i in range(0, len(t["rows"]), batch)]

    def prepare(t, idx):
        return dict(stem=t["stem"], idx=list(idx)), np.asarray(idx)

    def gpu(inputs, ok, t):
        if oom_calls is not None:
            oom_calls.append(len(ok))
        if oom_above is not None and len(ok) > oom_above:
            raise torch.OutOfMemoryError("fake OOM")
        events.append(("gpu", t["stem"], tuple(int(i) for i in ok)))
        return {int(i): 2 * int(i) for i in ok}

    written = {}

    def finish(job, t, per, stats):
        events.append(("finish", job.stem))
        written[job.stem] = dict(per)
        return dict(rows=len(per))

    return dict(load_table=load_table, plan_batches=plan_batches, prepare=prepare, gpu=gpu, finish=finish), written


def test_partition_covers_every_shard_exactly_once():
    """Two Cohere lanes split the manifest by crc32(path): every shard must land in exactly one lane, for any mod."""
    paths = [f"shards/src/train-{i:05d}.parquet" for i in range(200)] + ["shards/eval_jsut/eval-00000.parquet"]
    for mod in range(1, 5):
        for p in paths:
            assert sum(partition_ok(p, mod, r) for r in range(mod)) == 1
        if mod > 1:
            assert all(any(partition_ok(p, mod, r) for p in paths) for r in range(mod))


def test_ordered_subsequence():
    full = ["a", "b", "c", "d"]
    assert ordered_subsequence(["a", "c", "d"], full) and ordered_subsequence([], full)
    assert ordered_subsequence(full, full)
    assert not ordered_subsequence(["c", "a"], full)
    assert not ordered_subsequence(["a", "x"], full)
    assert not ordered_subsequence(["a", "a"], full)


def test_window_crosses_shard_boundaries(tmp_path):
    """The next shard's table is read, and its first batch prepared, before the GPU runs the current shard's last
    batch: the GPU never idles while a shard is read or written."""
    events = []
    cbs, written = _runner(events)
    jobs = [_job(tmp_path, "train-00000"), _job(tmp_path, "train-00001"), _job(tmp_path, "train-00002")]
    totals = run(lambda: list(jobs), workers=2, prefetch=2, follow_marker=None, heartbeat=None, progress=None, **cbs)
    last_gpu_a = events.index(("gpu", "train-00000", (4, 5)))
    assert events.index(("load", "train-00001")) < last_gpu_a
    assert [e for e in events if e[0] == "gpu"] == [("gpu", s, (i, i + 1)) for s in ("train-00000", "train-00001",
                                                                                    "train-00002") for i in (0, 2, 4)]
    assert [e[1] for e in events if e[0] == "finish"] == ["train-00000", "train-00001", "train-00002"]
    assert written["train-00001"] == {i: 2 * i for i in range(6)}
    assert totals["shards"] == 3 and totals["rows"] == 18 and totals["batches"] == 9


def test_empty_shard_is_still_finished(tmp_path):
    """A shard with no batch (every row > 30 s) still gets its (empty) outputs, so it counts as done."""
    events = []
    cbs, written = _runner(events)
    jobs = [_job(tmp_path, "train-00000", rows=0), _job(tmp_path, "train-00001", rows=2)]
    run(lambda: jobs, workers=1, prefetch=1, follow_marker=None, heartbeat=None, progress=None, **cbs)
    assert written == {"train-00000": {}, "train-00001": {0: 0, 1: 2}}


def test_write_pair_writes_jsonl_before_npz_both_fsynced(tmp_path, monkeypatch):
    """The npz marks a shard done, so it must never be visible before a durable jsonl (NTFS NUL-byte files after a
    hard kill were observed in practice)."""
    log = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (log.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (log.append(("replace", Path(b).name)), real_replace(a, b))[1])
    out = tmp_path / "o" / "src"
    write_pair(out / "train-00000.jsonl", [dict(id="a", hyp="あ"), dict(id="b", hyp="い")],
               out / "train-00000.npz", dict(ids=np.array(["a", "b"]), x=np.arange(3)), compressed=False)
    assert log == ["fsync", ("replace", "train-00000.jsonl"), "fsync", ("replace", "train-00000.npz")]
    assert (out / "train-00000.jsonl").read_text(encoding="utf-8").count("\n") == 2
    with np.load(out / "train-00000.npz") as z:
        assert z["ids"].tolist() == ["a", "b"]
    assert not list(out.glob("*.tmp"))
    write_pair(out / "c.jsonl", [], out / "c.npz", dict(x=np.zeros(1000)), compressed=True)
    assert (out / "c.npz").stat().st_size < (out / "train-00000.npz").stat().st_size + 8000


def test_follow_picks_up_late_shards_and_exits_after_the_marker(tmp_path, monkeypatch):
    """A lane behind a still-ingesting 01 waits (touching its heartbeat, so it is not killed as hung), picks up the
    shard appended later, and exits only once the marker exists and the work is drained."""
    events = []
    cbs, written = _runner(events)
    marker, hb = tmp_path / "ingest.done", tmp_path / "hb" / "cohere-0"
    a, b = _job(tmp_path, "train-00000"), _job(tmp_path, "train-00001")
    calls = []
    sleeps = []

    def todo():
        calls.append(1)
        n = len(calls)
        if n <= 2:
            return []  # nothing ingested yet
        if n <= 5:
            return [a]
        if n == 6:
            marker.touch()  # ingest finishes while the scan that finds b runs: b must still be labelled
        return [a, b]

    def fake_sleep(s):
        assert hb.exists(), "heartbeat must be touched during polls"
        sleeps.append(s)

    monkeypatch.setattr(labelpass.time, "sleep", fake_sleep)
    run(todo, workers=1, prefetch=2, follow_marker=marker, heartbeat=hb, progress=None, poll_s=7.0, **cbs)
    assert set(written) == {"train-00000", "train-00001"}
    assert sleeps and set(sleeps) == {7.0}
    assert len(calls) >= 7
    assert [e[1] for e in events if e[0] == "load"] == ["train-00000", "train-00001"]  # never submitted twice


def test_oom_halves_the_batch_and_keeps_every_row(tmp_path):
    events, calls = [], []
    cbs, written = _runner(events, n_rows=8, batch=8, oom_above=2, oom_calls=calls)
    emptied = []
    msgs = []
    totals = run(lambda: [_job(tmp_path, "train-00000", rows=8)], workers=1, prefetch=1, follow_marker=None,
                 heartbeat=None, progress=None, on_oom=lambda: emptied.append(1), log=msgs.append, **cbs)
    assert written["train-00000"] == {i: 2 * i for i in range(8)}
    assert calls == [8, 4, 2, 2, 4, 2, 2]
    assert len(emptied) == 3 and totals["oom_splits"] == 3
    assert any("OOM split 8" in m for m in msgs)


def test_oom_split_frees_the_failed_batch_before_the_halves(tmp_path):
    """The failed call's locals (its CUDA tensors on a box) must be dead before on_oom and the halves run."""
    import gc
    import weakref

    class Big:
        pass

    cbs, written = _runner([], n_rows=4, batch=4)
    refs, alive_at = [], []

    def gpu(inputs, ok, t):
        if refs:
            alive_at.append(refs[0]() is not None)
        if len(ok) > 2:
            big = Big()
            refs.append(weakref.ref(big))
            raise torch.OutOfMemoryError("fake OOM")
        return {int(i): 2 * int(i) for i in ok}

    def on_oom():
        gc.collect()
        alive_at.append(refs[0]() is not None)

    cbs["gpu"] = gpu
    totals = run(lambda: [_job(tmp_path, "train-00000", rows=4)], workers=1, prefetch=1, follow_marker=None,
                 heartbeat=None, progress=None, on_oom=on_oom, **cbs)
    assert written["train-00000"] == {i: 2 * i for i in range(4)}
    assert totals["oom_splits"] == 1
    assert alive_at and not any(alive_at)


def test_single_row_oom_reraises(tmp_path):
    cbs, _ = _runner([], batch=1, oom_above=0)
    with pytest.raises(torch.OutOfMemoryError):
        run(lambda: [_job(tmp_path, "train-00000", rows=2)], workers=1, prefetch=1, follow_marker=None,
            heartbeat=None, progress=None, **cbs)
    cbs, _ = _runner([], batch=4, oom_above=0)
    with pytest.raises(torch.OutOfMemoryError):  # oom_split off: the first OOM is fatal
        run(lambda: [_job(tmp_path, "train-00000", rows=4)], workers=1, prefetch=1, follow_marker=None,
            heartbeat=None, progress=None, oom_split=False, **cbs)


def test_progress_lines_and_heartbeat(tmp_path):
    import json
    cbs, _ = _runner([])
    prog, hb = tmp_path / "lanes" / "cohere-0.jsonl", tmp_path / "hb" / "cohere-0"
    jobs = [_job(tmp_path, "train-00000"), _job(tmp_path, "train-00001", rows=3)]
    run(lambda: jobs, workers=2, prefetch=3, follow_marker=None, heartbeat=hb, progress=prog, **cbs)
    lines = [json.loads(ln) for ln in prog.read_text(encoding="utf-8").splitlines()]
    assert [(ln["stem"], ln["rows"], ln["adopted"], ln["source"], ln["split"]) for ln in lines] == [
        ("train-00000", 6, False, "s", "train"), ("train-00001", 3, False, "s", "train")]
    assert all(set(ln) >= {"stem", "source", "split", "hours", "wall_s", "adopted", "rows"} for ln in lines)
    assert hb.exists()


def test_writer_errors_surface(tmp_path):
    cbs, _ = _runner([])

    def bad_finish(job, t, per, stats):
        raise OSError("disk full")

    cbs["finish"] = bad_finish
    with pytest.raises(OSError, match="disk full"):
        run(lambda: [_job(tmp_path, "train-00000")], workers=1, prefetch=1, follow_marker=None, heartbeat=None,
            progress=None, **cbs)


# ---------------------------------------------------------------- 02 on the runner


def _args(**kw):
    return argparse.Namespace(**(dict(k=16, save_encoder=False) | kw))


def _fc(tmp_path):
    return make_fake_corpus(tmp_path, {"src_a": (40, "train"), "eval_x": (10, "eval")}, rows_per_shard=16)


def test_shard_is_done_by_ids(tmp_path):
    """Done means these ids, not this row count: a same-size npz over other rows is refused, a subsequence with
    skipped rows is accepted, and a missing jsonl is not done."""
    fc = _fc(tmp_path)
    info = next(s for s in read_manifest(fc.data) if s.source == "src_a")
    stem = Path(info.path).stem
    npz = fc.teacher_out / "src_a" / f"{stem}.npz"
    ids = shard_ids(fc.data, info)
    assert tp.shard_is_done(npz, ids, _args())
    assert not tp.shard_is_done(npz, ids, _args(k=8))
    other = [i + "x" for i in ids]
    assert not tp.shard_is_done(npz, other, _args())
    # a subsequence: one row skipped by the pass
    rows = [fc.utts[i] for i in ids]
    _write_teacher(fc, "src_a", stem, rows[:3] + rows[4:], n_scanned=len(ids))
    assert tp.shard_is_done(npz, ids, _args())
    npz.with_suffix(".jsonl").unlink()
    assert not tp.shard_is_done(npz, ids, _args())


def test_pack_teacher_shard_matches_the_fixture_packing(tmp_path):
    """The extracted packer is the old process_shard packing: same keys, dtypes, shapes and bytes as tests/fixtures.py,
    which the trainer and make_selection tests read."""
    fc = _fc(tmp_path)
    info = next(s for s in read_manifest(fc.data) if s.source == "src_a")
    stem = Path(info.path).stem
    ids = shard_ids(fc.data, info)
    utts = [fc.utts[i] for i in ids]
    per = {i: dict(tokens=u.tokens, top_idx=u.topk_idx, top_lp=u.topk_lp, lse=u.lse, truncated=u.truncated,
                   hyp=u.hyp, enc=None) for i, u in enumerate(utts)}
    durs = np.array([u.duration for u in utts], dtype=np.float32)
    packed = tp.pack_teacher_shard(ids, [u.text for u in utts], durs, len(ids), per, sorted(per), fc.k, False,
                                   fc.prompt)
    with np.load(fc.teacher_out / "src_a" / f"{stem}.npz") as z:
        assert sorted(z.files) == sorted(packed)
        for key in z.files:
            if key == "cer":
                continue  # the fixture stores a made-up CER; the pass computes it
            a, b = np.asarray(packed[key]), z[key]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
    assert packed["cer"].dtype == np.float32 and packed["cer"].shape == (len(ids),)
    empty = tp.pack_teacher_shard([], [], np.zeros(0, np.float32), 3, {}, [], fc.k, False, None)
    assert empty["topk_idx"].shape == (0, fc.k) and int(empty["n_scanned"]) == 3 and empty["prompt"].size == 0


def test_adoption_copies_byte_identically_only_on_matching_ids_and_prompt(tmp_path):
    fc = _fc(tmp_path)
    info = next(s for s in read_manifest(fc.data) if s.source == "src_a")
    stem = Path(info.path).stem
    ids = shard_ids(fc.data, info)
    out = tmp_path / "labels" / "teacher_out"
    job = ShardJob(info, stem, out / "src_a")
    assert not tp.try_adopt(fc.teacher_out, job, [i + "x" for i in ids], fc.prompt, _args())
    assert not tp.try_adopt(fc.teacher_out, job, ids, fc.prompt + [1], _args())
    assert not tp.try_adopt(fc.teacher_out, job, ids, None, _args())
    assert not (out / "src_a").exists() or not list((out / "src_a").iterdir())
    assert tp.try_adopt(fc.teacher_out, job, ids, fc.prompt, _args())
    for ext in (".npz", ".jsonl"):
        assert (out / "src_a" / f"{stem}{ext}").read_bytes() == (fc.teacher_out / "src_a" / f"{stem}{ext}").read_bytes()
    assert tp.shard_is_done(out / "src_a" / f"{stem}.npz", ids, _args())
    # a jsonl that does not hold exactly the npz ids is not adopted
    other = tmp_path / "other"
    shutil.copytree(fc.teacher_out, other)
    j = other / "src_a" / f"{stem}.jsonl"
    lines = j.read_text(encoding="utf-8").splitlines(keepends=True)
    j.write_text(lines[0] + "".join(lines[2:-1]) + lines[-1], encoding="utf-8")
    job2 = ShardJob(info, stem, tmp_path / "out2" / "src_a")
    assert not tp.try_adopt(other, job2, ids, fc.prompt, _args())


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["02_teacher_pass.py", *map(str, argv)])
    with pytest.raises(SystemExit) as e:
        tp.main()
    return e.value.code


def test_strict_existing_exits_65_on_a_mismatched_output(tmp_path, monkeypatch, capsys):
    """On the label box an existing output that is not done over the shard's ids is evidence, not something to
    recompute over: exit 65 names the stem before any model is loaded."""
    fc = _fc(tmp_path)
    out = tmp_path / "out"
    shutil.copytree(fc.teacher_out / "src_a", out / "src_a")
    victim = sorted((out / "src_a").glob("*.jsonl"))[0]
    victim.unlink()
    monkeypatch.setattr(tp.torch.cuda, "mem_get_info", lambda: pytest.fail("the model must not be loaded"))
    assert _main(monkeypatch, "--data", fc.data, "--out", out, "--sources", "src_a", "--strict-existing") == 65
    assert victim.stem in capsys.readouterr().err


def test_strict_existing_meta_mismatch_exits_65(tmp_path, monkeypatch):
    fc = _fc(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(fc.teacher_out / "meta.json", out / "meta.json")  # fake/teacher: other settings
    assert _main(monkeypatch, "--data", fc.data, "--out", out, "--strict-existing") == 65


def test_adopt_only_exits_65_when_the_gate_shard_cannot_be_adopted(tmp_path, monkeypatch, capsys):
    """Gate sets are never recomputed on the box: a shard that is not adoptable (here the seed was made with other
    settings) stops the lane."""
    fc = _fc(tmp_path)
    monkeypatch.setattr(tp.torch.cuda, "mem_get_info", lambda: pytest.fail("the model must not be loaded"))
    code = _main(monkeypatch, "--data", fc.data, "--out", tmp_path / "out", "--sources", "eval_x",
                 "--adopt-from", fc.teacher_out, "--adopt-only", "eval_x")
    assert code == 65 and "eval_x/" in capsys.readouterr().err


def test_main_adopts_everything_and_loads_no_model(tmp_path, monkeypatch):
    """With a seed made under the pinned settings every shard is adopted, the progress file records it, and the pass
    returns without touching the GPU."""
    import json
    fc = _fc(tmp_path)
    meta = json.loads((fc.teacher_out / "meta.json").read_text(encoding="utf-8"))
    meta.update(model=tp.MODEL_ID, model_revision=tp.MODEL_REVISION)
    (fc.teacher_out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    out, prog = tmp_path / "out", tmp_path / "lanes" / "c.jsonl"
    monkeypatch.setattr(tp.torch.cuda, "mem_get_info", lambda: pytest.fail("the model must not be loaded"))
    monkeypatch.setattr(sys, "argv", ["02", "--data", str(fc.data), "--out", str(out), "--adopt-from",
                                      str(fc.teacher_out), "--adopt-only", "eval_x", "--strict-existing",
                                      "--progress", str(prog)])
    tp.main()
    shards = read_manifest(fc.data)
    lines = [json.loads(ln) for ln in prog.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(shards) and all(ln["adopted"] for ln in lines)
    for s in shards:
        stem = Path(s.path).stem
        assert (out / s.source / f"{stem}.npz").read_bytes() == (fc.teacher_out / s.source / f"{stem}.npz").read_bytes()
    tp.main()  # a re-run finds everything done by ids and adopts nothing again
    assert len(prog.read_text(encoding="utf-8").splitlines()) == len(shards)


def test_partition_flags_select_disjoint_shards(tmp_path, monkeypatch, capsys):
    fc = _fc(tmp_path)
    monkeypatch.setattr(tp.torch.cuda, "mem_get_info", lambda: (_ for _ in ()).throw(RuntimeError("stop here")))
    counts = []
    for rem in (0, 1):
        monkeypatch.setattr(sys, "argv", ["02", "--data", str(fc.data), "--out", str(tmp_path / "out"),
                                          "--shard-mod", "2", "--shard-rem", str(rem)])
        try:
            tp.main()
        except (RuntimeError, SystemExit):
            pass
        first = capsys.readouterr().out.splitlines()[1]
        counts.append(int(first.split("/")[1].split()[0]))
    assert sum(counts) == len(read_manifest(fc.data))
