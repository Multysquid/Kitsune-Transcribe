"""Tests for the ingest paths of scripts/01_prepare_data.py (emilia_yodas, emilia_nc, galgame, HF parquet) and its
disk guard.

The HF listing and downloads are faked: tars and parquet files are built in tmp_path with FLAC bytes (under .mp3 /
.ogg member names for the tars: sf.info sniffs the container, not the extension), so no network and no MP3 encoder
are needed. CPU only.

The box rebuilds the audio shards from scratch while the teacher outputs were made from the laptop's shards, which
were often built by an interrupted-and-resumed ingest; the two are joined by id. So a resumed ingest must give
exactly the ids and splits of a fresh one, and the galgame hold-out must be the first N KEPT rows.
"""
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

from kitsune.store import ShardWriter, iter_rows, lock_data_root, read_manifest  # noqa: E402

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


def ingest(root: Path, source: str, fn, *args):
    ing = prep.Ingest(root, root / "raw", source, None)
    fn(ing, *args)
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


def make_reazon_parquet(path: Path, names: list[str]):
    audio_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([("audio", audio_t), ("transcription", pa.string()), ("name", pa.string())])
    rows = [dict(audio=dict(bytes=flac(1.0), path=None), transcription="テキストです。", name=n) for n in names]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


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
