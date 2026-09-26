"""The frame preflight's decode (kitsune.trainset.build_frame_stores, decision 15), parallel across processes by shard.

The preflight decodes every packed row of a frame store once, to measure its length in samples (the student's frame
count is a function of it). Threads do not speed this up: soundfile reads an in-memory container through libsndfile's
virtual I/O, whose read/seek/tell callbacks are Python (cffi) functions, so every MP3 frame the decoder pulls takes the
GIL. Measured on the laptop (i9-13900HX, other jobs running), 1,024 real Emilia MP3 rows (24 kHz, 8.8 s mean):
serial 140 rows/s, 8 threads 155 rows/s, 8 processes 727 rows/s once started (x5.2). A spawned process costs ~3 s to
start (numpy, soundfile, librosa), which is why a small pool looked no faster on a benchmark of a few hundred rows;
a store build pays it once. The worker imports neither torch nor kitsune.trainset, so the start stays that cheap.

The work is cut by shard: each task is a contiguous row range of ONE data shard in the packed audio file (the store
packs rows shard by shard), split into blocks of at most `block` rows so a 2,048-row shard spreads over several
workers. Results come back in row order. Small builds (fewer than PROCESS_MIN_ROWS rows, or one worker) decode in
threads in this process, where a pool's start would cost more than it saves.

On a vast container os.cpu_count() is the HOST's CPU count (the first A100 run: 128, with a cgroup cpu.max of 15.36
CPUs), so the pool is sized by the CPUs this process may use (default_workers: affinity and the cgroup quota), and
each decoder starts with its BLAS/OpenMP pools at one thread (ONE_THREAD_ENV): numpy's OpenBLAS starts a pool of
nproc threads at import, and 64 decoders x 64 threads on a host with a low pids.max (vast/onstart.sh caps the pools
only below 16 pids per CPU) would fail a thread start in a worker and break the pool - box B's first phase.
"""
from __future__ import annotations

import contextlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

PROCESS_MIN_ROWS = 2000  # below this a build decodes in threads (a pool's ~3 s start would dominate)
BLOCK = 256  # rows per task: a shard of 2,048 rows becomes 8 tasks
MAX_PROCESSES = 64  # the box has 192 cores; 64 decoders read ~10k Emilia rows/s, more only adds start-up and memory
WIN_MAX_PROCESSES = 61  # ProcessPoolExecutor's limit on Windows
# the thread pools a decoder process starts with one thread: it decodes one row at a time, and a pool sized to the
# host's nproc in each of 64 processes would spend the container's pids budget (spawned children inherit the env)
ONE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def usable_cpus() -> int:
    """The CPUs this process may use: its affinity (os.cpu_count() where there is none), and at most the cgroup v2
    cpu.max quota rounded up (a vast container's quota is often far below the host's nproc)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = CPU_MAX.read_text().split()[:2]
        if quota != "max":
            n = min(n, -(-int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n)


def default_workers() -> int:
    """Decode processes of a build: every usable CPU (usable_cpus) up to MAX_PROCESSES (61 on Windows)."""
    cap = WIN_MAX_PROCESSES if sys.platform == "win32" else MAX_PROCESSES
    return max(1, min(cap, usable_cpus()))


@contextlib.contextmanager
def one_thread_children():
    """ONE_THREAD_ENV set to 1 while the pool starts its processes (spawned children copy the env at their start, and
    OpenBLAS reads it when numpy loads, before any initializer could run); the previous values restored after."""
    old = {k: os.environ.get(k) for k in ONE_THREAD_ENV}
    os.environ.update({k: "1" for k in ONE_THREAD_ENV})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def shard_tasks(shard_rows: Sequence[int], block: int = BLOCK) -> list[tuple[int, int]]:
    """[lo, hi) row ranges of the packed file: each within one shard (shard_rows = the rows each shard contributed, in
    packing order), at most `block` rows long."""
    out, lo = [], 0
    for n in shard_rows:
        hi = lo + int(n)
        out += [(a, min(a + block, hi)) for a in range(lo, hi, block)]
        lo = hi
    return out


def decode_range(audio_path: str, offsets: np.ndarray) -> list:
    """The decoded length in samples of the rows whose byte offsets into audio_path are offsets[j]..offsets[j + 1], or
    the error text of a row that does not decode (the preflight counts it as a frame mismatch). Own file handle, no
    memory map (on Windows a map would keep the file from being renamed)."""
    from kitsune.audio import decode_audio

    out = []
    with open(audio_path, "rb") as f:
        for j in range(len(offsets) - 1):
            f.seek(int(offsets[j]))
            try:
                out.append(len(decode_audio(f.read(int(offsets[j + 1] - offsets[j])))))
            except Exception as e:  # noqa: BLE001 - reported per row, the preflight decides
                out.append(f"{type(e).__name__}: {e}"[:200])
    return out


def decoded_lengths(audio_path: Path, offsets: np.ndarray, workers: int | None = None, *,
                    shard_rows: Sequence[int] | None = None, block: int = BLOCK,
                    processes: bool | None = None) -> tuple[list, dict]:
    """(the decoded length in samples, or an error text, of every packed row of audio_path in row order; how it ran:
    mode "processes" | "threads", workers, tasks). offsets: (n+1,) byte offsets. shard_rows: rows per data shard in
    packing order (default: one shard). processes: None = processes when there are at least PROCESS_MIN_ROWS rows and
    more than one worker."""
    offsets = np.asarray(offsets, dtype=np.int64)
    n = len(offsets) - 1
    tasks = shard_tasks([n] if shard_rows is None else shard_rows, block)
    if sum(hi - lo for lo, hi in tasks) != n:
        raise ValueError(f"shard_rows cover {sum(hi - lo for lo, hi in tasks)} rows, the packed file has {n}")
    nw = max(1, min(int(workers or default_workers()), max(len(tasks), 1)))
    use_procs = (n >= PROCESS_MIN_ROWS and nw > 1) if processes is None else bool(processes)
    args = [(str(audio_path), offsets[lo:hi + 1]) for lo, hi in tasks]
    if use_procs:
        import multiprocessing as mp

        # spawn, never fork: the trainer calling this holds torch (and maybe CUDA) and logger threads. The pool starts
        # its processes as map submits the tasks, so the one-thread env covers the whole block
        with one_thread_children(), ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as pool:
            parts = list(pool.map(decode_range, *zip(*args), chunksize=1)) if args else []
    else:
        with ThreadPoolExecutor(max_workers=nw) as pool:
            parts = list(pool.map(lambda a: decode_range(*a), args))
    return [x for p in parts for x in p], dict(mode="processes" if use_procs else "threads", workers=nw,
                                                tasks=len(tasks))


__all__ = ["BLOCK", "MAX_PROCESSES", "ONE_THREAD_ENV", "PROCESS_MIN_ROWS", "decode_range", "decoded_lengths",
           "default_workers", "one_thread_children", "shard_tasks", "usable_cpus"]
