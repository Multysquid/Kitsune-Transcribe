"""Tests for the Emilia ingest paths of scripts/01_prepare_data.py (emilia_yodas, emilia_nc) and its disk guard.

The HF listing and downloads are faked: tars are built in tmp_path with FLAC bytes under .mp3 member names (sf.info
sniffs the container, not the extension), so no network and no MP3 encoder are needed. CPU only.
"""
import io
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import load_script  # noqa: E402

from kitsune.store import iter_rows, lock_data_root, read_manifest  # noqa: E402

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
