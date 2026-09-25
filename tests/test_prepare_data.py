"""Tests for the ingest paths of scripts/01_prepare_data.py (emilia_yodas, emilia_nc, galgame, HF parquet) and its
disk guard.

The HF listing and downloads are faked: tars and parquet files are built in tmp_path with FLAC bytes (under .mp3 /
.ogg member names for the tars: sf.info sniffs the container, not the extension), so no network and no MP3 encoder
are needed. CPU only.

The box rebuilds the audio shards from scratch while the teacher outputs were made from the laptop's shards, which
were often built by an interrupted-and-resumed ingest; the two are joined by id. So a resumed ingest must give
exactly the ids and splits of a fresh one, and the galgame hold-out must be the first N KEPT rows. The label box also
decides per shard stem whether a shard is labelled and prunes labelled audio, so a resumed ingest must also give the
same stems (shard files and their rows), every shard needs its id sidecar, and every read of stored ids must work from
the sidecars alone.
"""
import argparse
import functools
import io
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import load_script  # noqa: E402

import kitsune.store as store  # noqa: E402
from kitsune.store import (  # noqa: E402
    ShardWriter, ids_sha256, iter_rows, load_progress, lock_data_root, read_ids, read_manifest, shard_ids,
    sidecar_meta, sidecar_path,
)

prep = load_script("01_prepare_data")


def flac(seconds: float, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(np.arange(int(seconds * sr)) / 7)).astype(np.float32), sr, format="FLAC")
    return buf.getvalue()


def make_tar(path: Path, clips: list[tuple[str, str, float]], gz: bool = False, worker: str = ""):
    """clips: (key, text, seconds) -> <worker/>key.mp3 + key.json members (json first, as in the real tars)."""
    with tarfile.open(path, "w:gz" if gz else "w") as tf:
        for key, text, sec in clips:
            for ext, data in (("json", json.dumps({"text": text, "language": "ja", "duration": sec}).encode()),
                              ("mp3", flac(sec))):
                info = tarfile.TarInfo(f"{worker}{key}.{ext}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    """Serve local tar files as if they were HF repo files; downloads are copies, frees are no-ops on the source."""
    files: dict[str, Path] = {}

    class Api:
        def list_repo_files(self, repo, repo_type=None, revision=None):
            assert revision == prep.REVISIONS[repo], "every listing must use the pinned revision"
            return sorted(files) + ["README.md"]

    def hf_hub_download(repo, filename, repo_type=None, revision=None, cache_dir=None):
        assert filename in files and revision == prep.REVISIONS[repo]
        dst = tmp_path / "dl" / filename.replace("/", "_")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(files[filename], dst)
        return str(dst)

    monkeypatch.setattr(prep, "HfApi", Api)
    monkeypatch.setattr(prep, "hf_hub_download", hf_hub_download)  # the real Ingest.download (and its disk guard) runs
    monkeypatch.setattr(prep.Ingest, "free_download", staticmethod(lambda local: Path(local).unlink(missing_ok=True)))
    return files


def ingest(root: Path, source: str, fn, *args, step: str | None = None, **kwargs):
    ing = prep.Ingest(root, root / "raw", source, None, step=step)
    fn(ing, *args, **kwargs)
    return ing


def ids(root: Path, source: str) -> list[str]:
    return [r["id"] for sh in read_manifest(root) if sh.source == source for r in iter_rows(root / sh.path, ["id"])]


def split_ids(root: Path, source: str) -> list[tuple[str, str]]:
    return sorted((sh.split, r["id"]) for sh in read_manifest(root) if sh.source == source
                  for r in iter_rows(root / sh.path, ["id"]))


@pytest.fixture()
def small_shards(monkeypatch):
    """Toy shard and hold-out sizes: the real 2048 rows per shard and 1000 galgame eval rows need ~3000 clips. The
    rows_per_shard default is bound at def time, so patching store.ROWS_PER_SHARD would do nothing."""
    monkeypatch.setattr(prep, "ShardWriter", functools.partial(ShardWriter, rows_per_shard=4))
    monkeypatch.setattr(prep, "GALGAME_EVAL_ROWS", 3)


def crash_after(monkeypatch, n: int):
    """Make Ingest.add raise after n calls (a crash mid input file); returns a function that undoes it."""
    real, calls = prep.Ingest.add, [0]

    def add(self, *a, **kw):
        calls[0] += 1
        if calls[0] > n:
            raise RuntimeError("simulated crash")
        return real(self, *a, **kw)
    monkeypatch.setattr(prep.Ingest, "add", add)
    return lambda: monkeypatch.setattr(prep.Ingest, "add", real)


def make_galgame_tar(path: Path, clips: list[tuple[str, str, float]]):
    """clips: (key, text, seconds) -> key.ogg + key.txt members, alternating which comes first (webdataset pairs
    arrive in either order)."""
    with tarfile.open(path, "w") as tf:
        for j, (key, text, sec) in enumerate(clips):
            pair = [("txt", text.encode()), ("ogg", flac(sec))]
            for ext, data in (pair if j % 2 else pair[::-1]):
                info = tarfile.TarInfo(f"{key}.{ext}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


def make_reazon_parquet(path: Path, names: list[str], whisper: list | None = None):
    """A ReazonSpeech mirror file; `whisper` adds its whisper_transcript column (token id lists, None = null)."""
    audio_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    fields = [("audio", audio_t), ("transcription", pa.string()), ("name", pa.string())]
    rows = [dict(audio=dict(bytes=flac(1.0), path=None), transcription="テキストです。", name=n) for n in names]
    if whisper is not None:
        fields.append(("whisper_transcript", pa.list_(pa.int64())))
        for r, w in zip(rows, whisper):
            r["whisper_transcript"] = w
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(fields)), path)


def layout(root: Path, source: str) -> tuple[dict, dict, list]:
    """What a resumed ingest must leave exactly as a fresh one does: {manifest path: ids read from the shard file},
    {manifest path: sidecar metadata}, and every parquet file under the source's directory (no orphans)."""
    listed = [sh for sh in read_manifest(root) if sh.source == source]
    rows = {sh.path: [r["id"] for r in iter_rows(root / sh.path, ["id"])] for sh in listed}
    metas = {sh.path: sidecar_meta(root, sh) for sh in listed}
    files = sorted(p.relative_to(root).as_posix() for p in (root / "shards" / source).rglob("*.parquet"))
    return rows, metas, files


def crash_in_flush(monkeypatch, where: str, nth: int):
    """Kill the nth shard flush after the shard's rename: before its sidecar ("sidecar") or before its manifest line
    ("manifest"). Returns a function that undoes it."""
    name = {"sidecar": "_write_sidecar", "manifest": "append_manifest"}[where]
    real, calls = getattr(store, name), [0]

    def step(*a, **kw):
        calls[0] += 1
        if calls[0] == nth:
            raise RuntimeError("simulated crash")
        return real(*a, **kw)
    monkeypatch.setattr(store, name, step)
    return lambda: monkeypatch.setattr(store, name, real)


def prune(root: Path):
    """What the label box's janitor does to labelled audio: delete the shard files, keep the manifest and sidecars."""
    for f in (root / "shards").glob("*/*.parquet"):
        f.unlink()


def test_emilia_nc_ingest_filters_budget_and_resume(tmp_path, fake_hub):
    t0, t1 = tmp_path / "JA-B000000_standard.tar.gz", tmp_path / "JA-B000001_standard.tar.gz"
    make_tar(t0, [("JA_B00000_S00000_W000000", "こんにちは、今日は。", 2.0),
                  ("JA_B00000_S00000_W000001", "hello there, this is english", 2.0),  # no Japanese script
                  ("JA_B00000_S00001_W000000", "", 2.0),  # empty text
                  ("JA_B00000_S00001_W000001", "天気がいいですね。", 40.0)], gz=True, worker="worker_0/")  # > 30 s
    make_tar(t1, [("JA_B00001_S00000_W000000", "ありがとう。", 3.0),
                  ("JA_B00001_S00000_W000001", "さようなら。", 3.0)], gz=True, worker="worker_1/")
    fake_hub.update({t0.name: t0, t1.name: t1})
    root = tmp_path / "data"
    ing = ingest(root, "emilia_nc", prep.ingest_emilia_nc, 1.5 / 3600)  # budget 1.5 s: stop right after the first 2 s clip
    assert ids(root, "emilia_nc") == ["emilia_nc/JA_B00000_S00000_W000000"]
    assert not ing.is_finished(t0.name), "a tar cut short by the budget is not finished"
    ing = ingest(root, "emilia_nc", prep.ingest_emilia_nc, float("inf"))  # raise the budget: resume, no duplicates
    got = ids(root, "emilia_nc")
    assert got == ["emilia_nc/JA_B00000_S00000_W000000", "emilia_nc/JA_B00001_S00000_W000000",
                   "emilia_nc/JA_B00001_S00000_W000001"]
    # the empty transcript has no Japanese script either, so it is counted with the English one
    assert ing.stats["non_ja"] == 2 and ing.stats["bad_duration"] == 1
    assert ing.stats["dup"] == 1 and ing.is_finished(t0.name) and ing.is_finished(t1.name)


def test_emilia_yodas_never_ingests_the_holdout_tar_or_videos(tmp_path, fake_hub):
    train_tar, other_tar, eval_tar = (tmp_path / n for n in ("JA-B000000.tar", "JA-B000001.tar", "JA-B000029.tar"))
    make_tar(train_tar, [("JA_vidA_W000001", "学習用のクリップです。", 2.0)])
    make_tar(eval_tar, [("JA_vidE_W000001", "評価用のクリップです。", 2.0)])
    make_tar(other_tar, [("JA_vidE_W000002", "評価用の動画の続きです。", 2.0),  # same video as the hold-out
                         ("JA_vidB_W000001", "別の動画です。", 2.0)])
    fake_hub.update({"JA/JA-B000000.tar": train_tar, "JA/JA-B000001.tar": other_tar, "JA/JA-B000029.tar": eval_tar})
    root = tmp_path / "data"
    ingest(root, "emilia_yodas", prep.ingest_emilia, 1.5 / 3600)  # a small budget: the first clip only
    ingest(root, "eval_emilia", prep.ingest_emilia_eval, prep.EMILIA_EVAL_TAR, 10)
    assert ids(root, "eval_emilia") == ["eval_emilia/JA_vidE_W000001"]
    ing = ingest(root, "emilia_yodas", prep.ingest_emilia, 1e6)  # the full-run budget
    got = ids(root, "emilia_yodas")
    assert got == ["emilia_yodas/JA_vidA_W000001", "emilia_yodas/JA_vidB_W000001"]
    assert ing.stats["eval_video"] == 1 and not ing.is_finished("JA/JA-B000029.tar")


def test_disk_guard_stops_before_a_download(tmp_path, fake_hub, monkeypatch):
    t0 = tmp_path / "JA-B000000_standard.tar.gz"
    make_tar(t0, [("JA_B00000_S00000_W000000", "こんにちは。", 2.0)], gz=True, worker="worker_0/")
    fake_hub[t0.name] = t0
    monkeypatch.setattr(prep.shutil, "disk_usage", lambda p: SimpleNamespace(free=int((prep.MIN_FREE_GB - 1) * 1e9)))
    with pytest.raises(SystemExit, match="GB free"):
        ingest(tmp_path / "data", "emilia_nc", prep.ingest_emilia_nc, float("inf"))



def test_one_ingest_per_data_root(tmp_path, monkeypatch):
    """Two ingests into one data root (another launcher started while the first still runs) overwrite each other's
    shards and progress: the second exits before it touches anything, and the lock goes with its holder."""
    root = tmp_path / "data"

    def ingested(*a, **kw):
        raise AssertionError("ingested while another run held the data root")

    monkeypatch.setattr(prep, "ingest_hf_parquet", ingested)
    monkeypatch.setattr(sys, "argv", ["01_prepare_data.py", "--data", str(root), "--sources", "eval_jsut"])
    first = lock_data_root(root)
    assert first is not None
    try:
        assert lock_data_root(root) is None  # a second handle is refused, as a second process's would be
        with pytest.raises(SystemExit, match="another 01_prepare_data is ingesting"):
            prep.main()
    finally:
        first.close()
    for _ in range(40):  # Windows may take a moment to drop a closed handle's lock
        again = lock_data_root(root)
        if again is not None:
            break
        time.sleep(0.05)
    assert again is not None
    again.close()


def test_reazon_tiers_dedup_against_every_smaller_tier():
    assert set(prep.DEDUP_AGAINST["reazon_large"]) == {"reazon_small", "reazon_medium"}
    assert prep.DEDUP_AGAINST["reazon_medium"] == ("reazon_small",)
    assert "reazon_large" in prep.HF_PARQUET_SOURCES and "emilia_nc" in prep.ALL_SOURCES


def test_galgame_resume_matches_a_fresh_run(tmp_path, fake_hub, small_shards, monkeypatch):
    t0, t1 = tmp_path / "a-000.tar", tmp_path / "a-001.tar"
    make_galgame_tar(t0, [(f"k{i}", "テキストです。", 40.0 if i == 1 else 1.0) for i in range(10)])  # k1 > 30 s
    make_galgame_tar(t1, [(f"m{i}", "テキストです。", 1.0) for i in range(3)])
    fake_hub.update({t0.name: t0, t1.name: t1})
    fresh = tmp_path / "fresh"
    ingest(fresh, "galgame", prep.ingest_galgame, 2)
    # absolute, not only fresh == resumed: the box's fresh ingest must match the laptop's frozen teacher outputs
    want = sorted([("eval", f"galgame/k{i}") for i in (0, 2, 3)]  # the first 3 KEPT rows: k1 is too long
                  + [("train", f"galgame/{k}") for k in [f"k{i}" for i in range(4, 10)] + ["m0", "m1", "m2"]])
    assert split_ids(fresh, "galgame") == want
    # crash after 8 adds (k1 rejected): train-00000 = k4..k7 is flushed while the eval rows k0, k2, k3 are still
    # buffered, a crash inside the first tar before the eval shard is written. After 10 adds the first tar is finished
    # (eval shard included) and the crash is in the second: the hold-out count must then come from the manifest.
    after_crash = {8: [("train", f"galgame/k{i}") for i in range(4, 8)], 10: [x for x in want if "/m" not in x[1]]}
    for n, dup in ((8, 4), (10, 0)):
        resumed = tmp_path / f"resumed{n}"
        restore = crash_after(monkeypatch, n)
        with pytest.raises(RuntimeError, match="simulated crash"):
            ingest(resumed, "galgame", prep.ingest_galgame, 2)
        assert split_ids(resumed, "galgame") == after_crash[n]
        restore()
        ing = ingest(resumed, "galgame", prep.ingest_galgame, 2)
        assert split_ids(resumed, "galgame") == want, f"crash after {n} adds"
        assert ing.stats["dup"] == dup


def test_parquet_resume_matches_a_fresh_run(tmp_path, fake_hub, small_shards, monkeypatch):
    repo, split = prep.HF_PARQUET_SOURCES["reazon_small"]
    sizes = (8, 3)
    for j, n in enumerate(sizes):
        p = tmp_path / f"train-{j}.parquet"
        make_reazon_parquet(p, [f"f{j}r{i}" for i in range(n)])
        fake_hub[f"data/train-{j}.parquet"] = p
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    ingest(fresh, "reazon_small", prep.ingest_hf_parquet, repo, split)
    want = sorted(("train", f"reazon_small/f{j}r{i}") for j, n in enumerate(sizes) for i in range(n))
    assert split_ids(fresh, "reazon_small") == want
    restore = crash_after(monkeypatch, 6)  # train-00000 (f0r0..f0r3) flushed; f0r4, f0r5 lost with the buffer
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest(resumed, "reazon_small", prep.ingest_hf_parquet, repo, split)
    restore()
    ing = ingest(resumed, "reazon_small", prep.ingest_hf_parquet, repo, split)
    assert split_ids(resumed, "reazon_small") == want and ing.stats["dup"] == 4
    # a larger tier skips every row already in a smaller one (DEDUP_AGAINST), by name
    p = tmp_path / "train-2.parquet"
    make_reazon_parquet(p, ["new0"])
    fake_hub["data/train-2.parquet"] = p
    ing = ingest(resumed, "reazon_medium", prep.ingest_hf_parquet, *prep.HF_PARQUET_SOURCES["reazon_medium"])
    assert ids(resumed, "reazon_medium") == ["reazon_medium/new0"] and ing.stats["dup"] == sum(sizes)


def add_reazon_files(tmp_path: Path, fake_hub: dict, sizes: tuple[int, ...], tag: str = "f"):
    for j, n in enumerate(sizes):
        p = tmp_path / f"{tag}-train-{j}.parquet"
        make_reazon_parquet(p, [f"{tag}{j}r{i}" for i in range(n)])
        fake_hub[f"data/train-{j}.parquet"] = p


def test_read_ids_prefers_the_sidecar_and_names_both_paths(tmp_path):
    """read_ids serves id/duration reads from the sidecar, so they survive the shard's deletion; other columns, and
    shards written before sidecars existed (the laptop's), are read from the shard."""
    root = tmp_path / "data"
    w = ShardWriter(root, "src", "train", rows_per_shard=2)
    for i, dur in enumerate((1.25, 2.5, 0.75)):
        w.add(f"src/u{i}", flac(0.5), f"text{i}", dur, 16000)
    first, second = w.close()
    assert [r["id"] for r in iter_rows(root / first.path, ["id"])] == shard_ids(root, first) == ["src/u0", "src/u1"]
    assert sidecar_meta(root, first) == dict(schema=1, input=None, step=None, rows=2,
                                             ids_sha256=ids_sha256(["src/u0", "src/u1"]))
    (root / first.path).unlink()
    assert read_ids(root, first, columns=("id", "duration")) == [dict(id="src/u0", duration=1.25),
                                                                 dict(id="src/u1", duration=2.5)]
    with pytest.raises(FileNotFoundError) as e:
        read_ids(root, first, columns=("id", "text"))  # not in the sidecar: the shard is needed
    assert str(root / first.path) in str(e.value) and str(sidecar_path(root, first)) in str(e.value)
    sidecar_path(root, second).unlink()  # a laptop shard: no sidecar
    assert sidecar_meta(root, second) is None and read_ids(root, second, ("id", "text")) == [dict(id="src/u2",
                                                                                                  text="text2")]


def test_orphan_shard_is_removed_and_stems_match_a_clean_run(tmp_path, fake_hub, small_shards, capsys):
    """A kill between a shard's rename and its manifest line leaves a shard (and sidecar) that is not listed, numbered
    right after the manifest's last. Numbering after the files on disk then shifted every later stem against a clean
    run, while the box decides per stem whether a shard is labelled. The next writer numbers after the manifest and
    removes the orphan. Unlisted shards above that stem are no kill's doing but a damaged manifest (their inputs may
    be marked finished and never read again): the ingest stops and deletes nothing."""
    repo, split = prep.HF_PARQUET_SOURCES["reazon_small"]
    add_reazon_files(tmp_path, fake_hub, (8, 3))
    clean, dirty = tmp_path / "clean", tmp_path / "dirty"
    ingest(clean, "reazon_small", prep.ingest_hf_parquet, repo, split)
    ingest(dirty, "reazon_small", prep.ingest_hf_parquet, repo, split, max_inputs=1)  # train-00000, train-00001
    shards = dirty / "shards" / "reazon_small"
    shutil.copy(shards / "train-00001.parquet", shards / "train-00002.parquet")
    shutil.copy(shards / "_ids" / "train-00001.parquet", shards / "_ids" / "train-00002.parquet")
    ingest(dirty, "reazon_small", prep.ingest_hf_parquet, repo, split)
    assert "removed orphan shards/reazon_small/train-00002.parquet" in capsys.readouterr().out
    rows, metas, files = layout(dirty, "reazon_small")
    assert (rows, metas, files) == layout(clean, "reazon_small")
    assert list(rows) == [f"shards/reazon_small/train-0000{i}.parquet" for i in range(3)]

    def on_disk(root: Path) -> dict:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "shards").rglob("*.parquet")}

    above = {"two orphans": ["train-00003.parquet"],  # train-00002 alone would pass for a kill's orphan
             "manifest lost": ["train-00001.parquet", "_ids/train-00001.parquet"]}  # likewise train-00000
    for damage, named in above.items():
        root = tmp_path / damage
        ingest(root, "reazon_small", prep.ingest_hf_parquet, repo, split, max_inputs=1)
        shards = root / "shards" / "reazon_small"
        if damage == "two orphans":
            for n in (2, 3):
                shutil.copy(shards / "train-00001.parquet", shards / f"train-0000{n}.parquet")
        else:
            (root / "manifest.jsonl").unlink()
        before = on_disk(root)
        with pytest.raises(SystemExit, match="manifest does not describe the disk") as e:
            ingest(root, "reazon_small", prep.ingest_hf_parquet, repo, split)
        assert ", ".join(f"shards/reazon_small/{f}" for f in named) + " not in manifest.jsonl" in str(e.value)
        assert on_disk(root) == before


@pytest.mark.parametrize("where", ["sidecar", "manifest"])
def test_crash_resume_gives_identical_stems(tmp_path, fake_hub, small_shards, monkeypatch, where):
    """Not only the ids: a run killed inside a shard flush and resumed leaves the same shard files, rows and sidecars as
    an uninterrupted one. The crash hits the second flush after its rename: parquet train-00001 (f0r4..f0r7), galgame
    eval-00000 (the hold-out, flushed at tar 0's end)."""
    repo, split = prep.HF_PARQUET_SOURCES["reazon_small"]
    add_reazon_files(tmp_path, fake_hub, (8, 3))
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    ingest(fresh, "reazon_small", prep.ingest_hf_parquet, repo, split)
    restore = crash_in_flush(monkeypatch, where, 2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest(resumed, "reazon_small", prep.ingest_hf_parquet, repo, split)
    restore()
    assert (resumed / "shards" / "reazon_small" / "train-00001.parquet").is_file()  # the orphan the crash left
    ingest(resumed, "reazon_small", prep.ingest_hf_parquet, repo, split)
    assert layout(resumed, "reazon_small") == layout(fresh, "reazon_small")

    fake_hub.clear()
    t0, t1 = tmp_path / "a-000.tar", tmp_path / "a-001.tar"
    make_galgame_tar(t0, [(f"k{i}", "テキストです。", 40.0 if i == 1 else 1.0) for i in range(10)])  # k1 > 30 s
    make_galgame_tar(t1, [(f"m{i}", "テキストです。", 1.0) for i in range(3)])
    fake_hub.update({t0.name: t0, t1.name: t1})
    ingest(fresh, "galgame", prep.ingest_galgame, 2)
    restore = crash_in_flush(monkeypatch, where, 2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest(resumed, "galgame", prep.ingest_galgame, 2)
    restore()
    assert (resumed / "shards" / "galgame" / "eval-00000.parquet").is_file()
    ingest(resumed, "galgame", prep.ingest_galgame, 2)
    rows, metas, files = layout(resumed, "galgame")
    assert (rows, metas, files) == layout(fresh, "galgame")
    assert rows["shards/galgame/eval-00000.parquet"] == ["galgame/k0", "galgame/k2", "galgame/k3"]


def test_sidecar_written_before_the_manifest_line_and_names_its_input(tmp_path, fake_hub, small_shards, monkeypatch):
    """A listed shard always has its sidecar (the label box reads ids from it once the audio is pruned), and the
    sidecar names the upstream input and ingest step the rows came from: the extent record maps inputs to stems with
    it. The galgame hold-out stem belongs to tar 0; the Emilia tar the 300 h step cuts has stems under both steps."""
    real, checked = store.append_manifest, []

    def append_manifest(root, shards):
        for info in shards:
            ids_ = [r["id"] for r in iter_rows(Path(root) / info.path, ["id"])]
            assert sidecar_path(root, info).is_file() and shard_ids(root, info) == ids_
            meta = sidecar_meta(root, info)
            assert meta["schema"] == 1 and meta["rows"] == info.rows == len(ids_)
            assert meta["ids_sha256"] == ids_sha256(ids_)
            durs = [r["duration"] for r in iter_rows(Path(root) / info.path, ["duration"])]
            assert [r["duration"] for r in read_ids(root, info, ("id", "duration"))] == durs
            checked.append(info.path)
        real(root, shards)
    monkeypatch.setattr(store, "append_manifest", append_manifest)

    root = tmp_path / "data"
    t0, t1 = tmp_path / "a-000.tar", tmp_path / "a-001.tar"
    make_galgame_tar(t0, [(f"k{i}", "テキストです。", 1.0) for i in range(6)])
    make_galgame_tar(t1, [(f"m{i}", "テキストです。", 1.0) for i in range(5)])
    fake_hub.update({t0.name: t0, t1.name: t1})
    ingest(root, "galgame", prep.ingest_galgame, 2)
    assert {sh.path: (sidecar_meta(root, sh)["input"], sidecar_meta(root, sh)["step"])
            for sh in read_manifest(root)} == {
        "shards/galgame/eval-00000.parquet": ("a-000.tar", "galgame"),  # k0..k2
        "shards/galgame/train-00000.parquet": ("a-000.tar", "galgame"),  # k3..k5
        "shards/galgame/train-00001.parquet": ("a-001.tar", "galgame"),  # m0..m3, full
        "shards/galgame/train-00002.parquet": ("a-001.tar", "galgame"),  # m4
    }

    fake_hub.clear()
    e0, e1 = tmp_path / "JA-B000000.tar", tmp_path / "JA-B000001.tar"
    make_tar(e0, [(f"JA_vid0_W00000{i}", "日本語です。", 2.0) for i in range(3)])
    make_tar(e1, [(f"JA_vid1_W00000{i}", "日本語です。", 2.0) for i in range(4)])
    fake_hub.update({"JA/JA-B000000.tar": e0, "JA/JA-B000001.tar": e1})
    ingest(root, "emilia_yodas", prep.ingest_emilia, 9 / 3600, step="emilia_yodas@300h")  # cut after 2 clips of tar 1
    ingest(root, "emilia_yodas", prep.ingest_emilia, float("inf"), step="emilia_yodas")  # the rest continues tar 1
    got = {sh.path: (sidecar_meta(root, sh)["input"], sidecar_meta(root, sh)["step"], sh.rows)
           for sh in read_manifest(root) if sh.source == "emilia_yodas"}
    assert got == {
        "shards/emilia_yodas/train-00000.parquet": ("JA/JA-B000000.tar", "emilia_yodas@300h", 3),
        "shards/emilia_yodas/train-00001.parquet": ("JA/JA-B000001.tar", "emilia_yodas@300h", 2),
        "shards/emilia_yodas/train-00002.parquet": ("JA/JA-B000001.tar", "emilia_yodas", 2),
    }
    assert checked == [sh.path for sh in read_manifest(root)]


def test_cross_source_reads_survive_pruned_shards(tmp_path, fake_hub, small_shards, monkeypatch):
    """The label box deletes a shard's audio once it is labelled, while later ingest steps still read the stored ids:
    reazon_large dedups against reazon_small, eval_emilia leaves out emilia_yodas's videos and emilia_yodas those of
    eval_emilia, the Emilia budget counts stored durations, and a resumed galgame skips what it stored. With every
    shard pruned after every step, the sidecars alone give the same result as the full shards."""
    rs, rl = tmp_path / "rs.parquet", tmp_path / "rl.parquet"
    make_reazon_parquet(rs, [f"a{i}" for i in range(5)])
    make_reazon_parquet(rl, ["a1", "b0", "a3", "b1"])
    e0, e1, ev = (tmp_path / n for n in ("JA-B000000.tar", "JA-B000001.tar", "JA-B000029.tar"))
    make_tar(e0, [("JA_vidA_W000001", "学習用です。", 2.0), ("JA_vidA_W000002", "学習用です。", 2.0),
                  ("JA_vidB_W000001", "学習用です。", 2.0)])
    make_tar(e1, [("JA_vidE_W000002", "評価用の動画です。", 2.0), ("JA_vidC_W000001", "学習用です。", 2.0)])
    make_tar(ev, [("JA_vidE_W000001", "評価用です。", 2.0), ("JA_vidA_W000003", "学習用の動画です。", 2.0),
                  ("JA_vidF_W000001", "評価用です。", 2.0)])
    emilia = {"JA/JA-B000000.tar": e0, "JA/JA-B000001.tar": e1, "JA/JA-B000029.tar": ev}
    g0, g1 = tmp_path / "a-000.tar", tmp_path / "a-001.tar"
    make_galgame_tar(g0, [(f"k{i}", "テキストです。", 40.0 if i == 1 else 1.0) for i in range(10)])
    make_galgame_tar(g1, [(f"m{i}", "テキストです。", 1.0) for i in range(3)])
    galgame = {g0.name: g0, g1.name: g1}

    def run(root: Path, pruned: bool) -> dict:
        def step(files: dict, source: str, fn, *args):
            fake_hub.clear()
            fake_hub.update(files)
            ing = ingest(root, source, fn, *args)
            if pruned:
                prune(root)
            return ing

        step({"data/train-0.parquet": rs}, "reazon_small", prep.ingest_hf_parquet,
             *prep.HF_PARQUET_SOURCES["reazon_small"])
        step(emilia, "emilia_yodas", prep.ingest_emilia, 3 / 3600)  # vidA_W000001, vidA_W000002
        step(emilia, "eval_emilia", prep.ingest_emilia_eval, prep.EMILIA_EVAL_TAR, 10)
        restore = crash_after(monkeypatch, 8)  # galgame train-00000 = k4..k7 stored, then a crash inside tar 0
        with pytest.raises(RuntimeError, match="simulated crash"):
            step(galgame, "galgame", prep.ingest_galgame, 1)
        restore()
        if pruned:
            prune(root)
        large = step({"data/train-0.parquet": rl}, "reazon_large", prep.ingest_hf_parquet,
                     *prep.HF_PARQUET_SOURCES["reazon_large"])
        yodas = step(emilia, "emilia_yodas", prep.ingest_emilia, 7 / 3600)
        gal = step(galgame, "galgame", prep.ingest_galgame, 2)
        sources = ("reazon_small", "reazon_large", "emilia_yodas", "eval_emilia", "galgame")
        stored = {s: sorted((sh.split, i) for sh in read_manifest(root) if sh.source == s for i in shard_ids(root, sh))
                  for s in sources}  # the pruned root has only the sidecars to read
        if not pruned:
            assert stored == {s: split_ids(root, s) for s in sources}  # the sidecars hold the shards' ids
        return dict(ids=stored, stems=[(sh.path, sh.rows) for sh in read_manifest(root)],
                    inputs={sh.path: sidecar_meta(root, sh)["input"] for sh in read_manifest(root)},
                    dup=(large.stats["dup"], yodas.stats["dup"], gal.stats["dup"]),
                    eval_video=yodas.stats["eval_video"], seconds=yodas.stats["seconds"])

    kept, pruned = run(tmp_path / "kept", False), run(tmp_path / "pruned", True)
    assert not list((tmp_path / "pruned" / "shards").glob("*/*.parquet"))
    assert pruned == kept
    got = kept["ids"]
    assert [i for _, i in got["reazon_large"]] == ["reazon_large/b0", "reazon_large/b1"]
    assert [i for _, i in got["eval_emilia"]] == ["eval_emilia/JA_vidE_W000001", "eval_emilia/JA_vidF_W000001"]
    # 4 s stored + vidB 2 s + vidC 2 s >= 7 s; vidE_W000002 (the hold-out's video) is left out on the way
    assert [i for _, i in got["emilia_yodas"]] == ["emilia_yodas/JA_vidA_W000001", "emilia_yodas/JA_vidA_W000002",
                                                   "emilia_yodas/JA_vidB_W000001", "emilia_yodas/JA_vidC_W000001"]
    assert kept["eval_video"] == 1 and kept["seconds"] == 8.0 and kept["dup"] == (2, 2, 4)
    assert len(got["galgame"]) == 12 and ("eval", "galgame/k0") in got["galgame"]
    assert {p: i for p, i in kept["inputs"].items() if not p.startswith("shards/galgame/")} == {
        "shards/reazon_small/train-00000.parquet": "data/train-0.parquet",
        "shards/reazon_small/train-00001.parquet": "data/train-0.parquet",
        "shards/emilia_yodas/train-00000.parquet": "JA/JA-B000000.tar",
        "shards/eval_emilia/eval-00000.parquet": prep.EMILIA_EVAL_TAR,
        "shards/reazon_large/train-00000.parquet": "data/train-0.parquet",
        "shards/emilia_yodas/train-00001.parquet": "JA/JA-B000000.tar",
        "shards/emilia_yodas/train-00002.parquet": "JA/JA-B000001.tar",
    }


def test_resumed_emilia_budget_cuts_where_a_fresh_run_does(tmp_path, fake_hub, small_shards, monkeypatch):
    """A resumed run counts the stored float32 durations against the Emilia budget; a fresh run that summed the float64
    header durations cut at another clip wherever the budget binds between the two sums, so a box restart inside the
    300 h step must cut where a fresh run does. (The laptop's one clean 300 h run cut 0.276 s past the budget, well
    outside the <= 0.114 s float32 drift over its 119,961 clips, so its cut stays.) 16002 frames at 16 kHz are
    1.000125 s, whose float32 is ~5e-8 s larger; the budget sits between the float64 and the float32 sum of six
    clips."""
    n = 16002
    d, d32 = n / 16000, float(np.float32(n / 16000))
    s64 = s32 = 0.0
    for _ in range(6):
        s64, s32 = s64 + d, s32 + d32
    assert s64 < s32
    tar = tmp_path / "JA-B000000.tar"
    make_tar(tar, [(f"JA_vid{i}_W000001", "日本語です。", (n + 0.5) / 16000) for i in range(8)])  # exactly n frames
    fake_hub["JA/JA-B000000.tar"] = tar
    hours = (s64 + s32) / 2 / 3600
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    ingest(fresh, "emilia_yodas", prep.ingest_emilia, hours)
    restore = crash_after(monkeypatch, 5)  # train-00000 (4 clips) stored, the 5th lost with the buffer
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest(resumed, "emilia_yodas", prep.ingest_emilia, hours)
    restore()
    ingest(resumed, "emilia_yodas", prep.ingest_emilia, hours)
    want = [f"emilia_yodas/JA_vid{i}_W000001" for i in range(6)]
    assert ids(fresh, "emilia_yodas") == want and ids(resumed, "emilia_yodas") == want
    assert layout(resumed, "emilia_yodas") == layout(fresh, "emilia_yodas")


def test_max_inputs_takes_a_sorted_prefix_and_galgame_alias_conflicts(tmp_path, fake_hub, monkeypatch):
    """--max-inputs SRC=N takes the first N of a source's sorted upstream files; that is how a subset of the labelled
    extent is rebuilt. progress.json records the listing size and each input's bytes for the extent record."""
    repo, split = prep.HF_PARQUET_SOURCES["reazon_large"]
    add_reazon_files(tmp_path, fake_hub, (1, 1, 1))
    root = tmp_path / "data"
    ingest(root, "reazon_large", prep.ingest_hf_parquet, repo, split, max_inputs=2)
    assert ids(root, "reazon_large") == ["reazon_large/f0r0", "reazon_large/f1r0"]
    prog = load_progress(root, "reazon_large")
    assert prog["n_listed"] == 3 and prog["finished_inputs"] == ["data/train-0.parquet", "data/train-1.parquet"]
    assert prog["input_bytes"] == {f: fake_hub[f].stat().st_size for f in prog["finished_inputs"]}

    fake_hub.clear()
    for j in range(2):
        t = tmp_path / f"JA-B00000{j}_standard.tar.gz"
        make_tar(t, [(f"JA_B0000{j}_S00000_W000000", "こんにちは。", 2.0)], gz=True, worker="worker_0/")
        fake_hub[t.name] = t
    ingest(root, "emilia_nc", prep.ingest_emilia_nc, float("inf"), max_inputs=1)
    assert ids(root, "emilia_nc") == ["emilia_nc/JA_B00000_S00000_W000000"]
    assert load_progress(root, "emilia_nc")["n_listed"] == 2

    fake_hub.clear()
    for j in (0, 1, 29):  # the hold-out's tar is not an input of emilia_yodas
        t = tmp_path / f"JA-B0000{j:02d}.tar"
        make_tar(t, [(f"JA_vid{j}_W000001", "日本語です。", 2.0)])
        fake_hub[f"JA/{t.name}"] = t
    ingest(root, "emilia_yodas", prep.ingest_emilia, float("inf"), max_inputs=1)
    assert ids(root, "emilia_yodas") == ["emilia_yodas/JA_vid0_W000001"]
    assert load_progress(root, "emilia_yodas")["n_listed"] == 2
    # the extent record maps each capped input to its stems through the sidecars
    assert {sh.path: (sidecar_meta(root, sh)["input"], sidecar_meta(root, sh)["step"])
            for sh in read_manifest(root)} == {
        "shards/reazon_large/train-00000.parquet": ("data/train-0.parquet", "reazon_large"),
        "shards/reazon_large/train-00001.parquet": ("data/train-1.parquet", "reazon_large"),
        "shards/emilia_nc/train-00000.parquet": ("JA-B000000_standard.tar.gz", "emilia_nc"),
        "shards/emilia_yodas/train-00000.parquet": ("JA/JA-B000000.tar", "emilia_yodas"),
    }

    assert prep.parse_max_inputs("reazon_large=40,emilia_nc=6") == {"reazon_large": 40, "emilia_nc": 6}
    for bad in ("reazon_large", "reazon_large=0", "reazon_large=4 ", "cv=1", "eval_emilia=1", "galgame=2,galgame=3"):
        with pytest.raises(argparse.ArgumentTypeError):
            prep.parse_max_inputs(bad)

    calls = []
    monkeypatch.setattr(prep, "ingest_galgame", lambda ing, n: calls.append(("galgame", n)))
    monkeypatch.setattr(prep, "ingest_hf_parquet", lambda ing, repo, split, **kw: calls.append((ing.source, kw)))

    def main(i: int, *argv):  # a root per run: Windows may take a moment to drop the previous run's lock
        monkeypatch.setattr(sys, "argv", ["01_prepare_data.py", "--data", str(tmp_path / f"cli{i}"), *argv])
        prep.main()

    main(0, "--sources", "galgame", "reazon_large", "--max-inputs", "galgame=2,reazon_large=3")
    main(1, "--sources", "galgame", "--galgame-shards", "4")
    main(2, "--sources", "galgame")
    main(3, "--sources", "reazon_large", "galgame", "--max-inputs", "reazon_large=5", "--max-inputs", "galgame=1")
    assert calls == [("galgame", 2), ("reazon_large", dict(max_inputs=3, whisper_dir=None)), ("galgame", 4),
                     ("galgame", 6), ("reazon_large", dict(max_inputs=5, whisper_dir=None)), ("galgame", 1)]
    for i, argv in enumerate([("--galgame-shards", "3", "--max-inputs", "galgame=2"),
                              ("--max-inputs", "galgame=2", "--max-inputs", "reazon_large=4,galgame=3")]):
        with pytest.raises(SystemExit) as e:  # a repeated flag must not silently drop a cap either
            main(4 + i, "--sources", "galgame", *argv)
        assert e.value.code == 2 and len(calls) == 6


def test_whisper_capture_keeps_nulls_and_is_written_before_finish_input(tmp_path, fake_hub, monkeypatch):
    """The second opinion for ReazonSpeech is the mirror's whisper_transcript column. 01 keeps it while the file is on
    disk (reading it again from the Hub means the whole mirror again), before the input is marked finished, nulls
    as null (02b falls back for them). A part left by a killed run is rewritten; sources without the column get none."""
    repo, split = prep.HF_PARQUET_SOURCES["reazon_small"]
    p = tmp_path / "train-0.parquet"
    make_reazon_parquet(p, ["n0", "n1", "n2"], whisper=[[50364, 1, 2], None, []])
    fake_hub["data/train-0.parquet"] = p
    wdir = tmp_path / "whisper"
    part = wdir / "reazon_small" / "train-0.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"left by a killed run")
    real, seen = prep.Ingest.finish_input, []

    def finish_input(self, name):
        seen.append((name, part.is_file() and part.read_bytes() != b"left by a killed run"))
        real(self, name)
    monkeypatch.setattr(prep.Ingest, "finish_input", finish_input)
    root = tmp_path / "data"
    ingest(root, "reazon_small", prep.ingest_hf_parquet, repo, split, whisper_dir=wdir)
    assert seen == [("data/train-0.parquet", True)]
    t = pq.read_table(part)
    assert t.column_names == ["name", "whisper_transcript"]
    assert t.to_pylist() == [dict(name="n0", whisper_transcript=[50364, 1, 2]),
                             dict(name="n1", whisper_transcript=None), dict(name="n2", whisper_transcript=[])]
    assert not list(wdir.glob("reazon_small/*.tmp"))

    fake_hub.clear()
    q = tmp_path / "eval-0.parquet"
    make_reazon_parquet(q, ["e0"])  # the ja_asr eval sets have no whisper_transcript column
    fake_hub["data/eval-0.parquet"] = q
    ingest(root, "eval_jsut", prep.ingest_hf_parquet, *prep.HF_PARQUET_SOURCES["eval_jsut"], whisper_dir=wdir)
    assert ids(root, "eval_jsut") == ["eval_jsut/eval-0-0"]
    assert sorted(d.name for d in wdir.iterdir()) == ["reazon_small"]


def test_hold_file_blocks_the_next_download(tmp_path, fake_hub, monkeypatch):
    """The label box's janitor holds the ingest while too much unlabelled audio is on disk: 01 waits before its next
    download and touches its heartbeat at every poll, so the controller does not take the wait for a hang. It is also
    touched at each download start and every HEARTBEAT_ROWS kept rows."""
    t0 = tmp_path / "JA-B000000_standard.tar.gz"
    make_tar(t0, [(f"JA_B00000_S00000_W00000{i}", "こんにちは。", 2.0) for i in range(2)], gz=True, worker="worker_0/")
    fake_hub[t0.name] = t0
    state = tmp_path / "state"
    hold, hb = state / "ingest.hold", state / "hb" / "ingest"
    state.mkdir()
    hold.touch()
    real_download, events = prep.hf_hub_download, []

    def download(*a, **kw):
        events.append(("download", hb.is_file()))
        hb.unlink()  # the rows must touch it again
        return real_download(*a, **kw)

    def sleep(s):
        events.append(("sleep", s, hb.is_file()))
        hb.unlink()  # the next poll must touch it again
        if len(events) == 3:
            hold.unlink()
    monkeypatch.setattr(prep, "hf_hub_download", download)
    monkeypatch.setattr(prep, "time", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(prep, "HEARTBEAT_ROWS", 2)
    root = tmp_path / "data"
    ing = prep.Ingest(root, root / "raw", "emilia_nc", None, heartbeat=hb, hold_file=hold)
    prep.ingest_emilia_nc(ing, float("inf"))
    assert events == [("sleep", prep.HOLD_POLL_S, True)] * 3 + [("download", True)]  # touched at the download start
    assert hb.is_file()  # touched again at the 2nd kept row
    assert ids(root, "emilia_nc") == [f"emilia_nc/JA_B00000_S00000_W00000{i}" for i in range(2)]
