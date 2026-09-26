"""kitsune.ctc_preflight: the frame preflight's decode, parallel across processes by shard.

  tasks       every task is a row range of one shard, at most `block` rows; together they cover the file once
  same result the process pool, the threads and a plain loop give the same length (or error text) for every row, in
  row order: 16 kHz FLAC, 24 kHz and 48 kHz WAV (resampled), and a row that does not decode
  real data   (skipped without it) 8 processes on real Emilia MP3 rows are faster than one: the speed-up the box's
              192 cores depend on. Measured on the laptop, it is printed
"""
import io
import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from kitsune import ctc_preflight as P
from kitsune.audio import decode_audio


def _wav(sr: int, seconds: float, fmt: str) -> bytes:
    rng = np.random.default_rng(sr + int(seconds * 1000))
    b = io.BytesIO()
    sf.write(b, (0.1 * rng.standard_normal(int(sr * seconds))).astype(np.float32), sr, format=fmt)
    return b.getvalue()


def _packed(tmp_path: Path, blobs: list[bytes]) -> tuple[Path, np.ndarray]:
    path = tmp_path / "audio.bin"
    path.write_bytes(b"".join(blobs))
    return path, np.concatenate([[0], np.cumsum([len(b) for b in blobs])]).astype(np.int64)


def test_tasks_are_cut_by_shard():
    assert P.shard_tasks([5, 0, 3], block=2) == [(0, 2), (2, 4), (4, 5), (5, 7), (7, 8)]
    assert P.shard_tasks([2048], block=256)[-1] == (1792, 2048) and len(P.shard_tasks([2048])) == 8
    assert P.shard_tasks([]) == []
    assert 1 <= P.default_workers() <= P.MAX_PROCESSES


def _child_env(names):
    return {k: os.environ.get(k) for k in names}


def test_the_pool_is_sized_by_the_cgroup_quota_and_its_children_start_one_thread_pools(tmp_path, monkeypatch):
    """A vast container's os.cpu_count() is the host's (the first A100 run: 128 with cpu.max 15.36 CPUs): the pool
    takes the quota, rounded up; each decoder's BLAS/OpenMP pools start at one thread, and the parent's env is back
    after the pool."""
    cpu_max = tmp_path / "cpu.max"
    monkeypatch.setattr(P, "CPU_MAX", cpu_max)
    base = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    cpu_max.write_text("1536000 100000\n")
    assert P.usable_cpus() == min(16, base) and P.default_workers() <= 16
    cpu_max.write_text("max 100000\n")
    assert P.usable_cpus() == base
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with P.one_thread_children(), ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        child = pool.submit(_child_env, P.ONE_THREAD_ENV).result()
    assert child == {k: "1" for k in P.ONE_THREAD_ENV}
    assert os.environ["OMP_NUM_THREADS"] == "7" and "OPENBLAS_NUM_THREADS" not in os.environ


def test_processes_threads_and_a_loop_agree(tmp_path):
    blobs = [_wav(16000, 0.5 + 0.1 * i, "FLAC") for i in range(4)] + [_wav(24000, 0.73, "WAV"),
                                                                        _wav(48000, 0.61, "WAV"), b"not audio"]
    path, offs = _packed(tmp_path, blobs)
    want = []
    for b in blobs:
        try:
            want.append(len(decode_audio(b)))
        except Exception:  # noqa: BLE001
            want.append(None)
    procs, how = P.decoded_lengths(path, offs, 2, shard_rows=[3, 4], block=2, processes=True)
    assert how == dict(mode="processes", workers=2, tasks=4)
    threads, how_t = P.decoded_lengths(path, offs, 3, shard_rows=[3, 4], block=2, processes=False)
    assert how_t["mode"] == "threads" and procs[:-1] == threads[:-1]
    assert procs[:-1] == want[:-1] and want[-1] is None
    assert procs[-1].startswith("LibsndfileError") and threads[-1].startswith("LibsndfileError")  # the error text
    assert abs(procs[4] - 11680) <= 1 and abs(procs[5] - 9760) <= 1  # 24 and 48 kHz resampled to 16 kHz
    # the default: few rows decode in threads (a pool's start would cost more than it saves)
    assert P.decoded_lengths(path, offs, 8)[1]["mode"] == "threads"
    with pytest.raises(ValueError, match="shard_rows cover 3 rows"):
        P.decoded_lengths(path, offs, 2, shard_rows=[3])


@pytest.mark.slow
def test_eight_processes_beat_one_on_real_emilia_mp3(tmp_path):
    """The speed-up on real data (skipped without the first run's Emilia shards): 8 processes vs a serial loop over
    the same rows of D:/Shizu-ko-distill/data/shards/emilia_nc (read-only), start-up included."""
    import pyarrow.parquet as pq

    root = Path(os.environ.get("KITSUNE_REAL_DATA_ROOT", "D:/Shizu-ko-distill")) / "data" / "shards" / "emilia_nc"
    shards = sorted(root.glob("train-*.parquet"))[:4]
    if len(shards) < 4:
        pytest.skip(f"no Emilia shards under {root}")
    blobs, rows = [], []
    for s in shards:
        a = pq.ParquetFile(s).read_row_group(0, columns=["audio"]).column("audio").to_pylist()
        blobs += a
        rows.append(len(a))
    path, offs = _packed(tmp_path, blobs)
    t = time.time()
    serial = P.decode_range(str(path), offs)
    t_serial = time.time() - t
    t = time.time()
    par, how = P.decoded_lengths(path, offs, 8, shard_rows=rows, processes=True)
    t_par = time.time() - t
    assert par == serial and all(isinstance(x, int) for x in par) and how["mode"] == "processes"
    print(f"\n{len(blobs)} Emilia MP3 rows: serial {len(blobs) / t_serial:.0f} rows/s, 8 processes "
          f"{len(blobs) / t_par:.0f} rows/s incl. start (x{t_serial / t_par:.2f})")
    assert t_par < t_serial
