"""Shared shard runner for the label passes (scripts/02_teacher_pass.py now, scripts/02p_parakeet_pass.py next).

A pass is a set of callbacks; `run` owns the pipeline around them:
  load_table(job) -> table                       reads one shard (a single reader thread, one shard ahead)
  plan_batches(job, table) -> [row index arrays]  the shard's batches, in the order they go to the GPU
  prepare(table, idx) -> (inputs, ok_idx)        decode + featurise on a CPU thread pool; ok_idx are the rows that
                                                 decoded (inputs is None when none did)
  gpu(inputs, ok_idx, table) -> {row: result}    runs on the calling (main) thread; a sequence aligned with ok_idx
                                                 is accepted too
  finish(job, table, per_row, stats) -> dict|None  packs and writes the shard on a single writer thread

Work is one flat stream of (shard, batch) items across shard boundaries, with `prefetch` prep items in flight, so the
GPU never waits for a shard to be read, packed or written (02 used to idle 5-8 s per shard there). Threads only, no
process pool: a row is skipped only by the pass's own deterministic rules, identically in every pass.

An OOM inside `gpu` empties the cache and runs the two halves of the batch (recursively); a single row that still
OOMs re-raises. With `follow_marker` set the runner polls `todo()` until the marker exists and nothing is left, so a
lane can run behind a still-ingesting 01.
"""
import json
import os
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from kitsune.store import ShardInfo

MAX_PENDING_WRITES = 2  # finished shards waiting for the writer; beyond this the GPU loop waits (bounded memory)


@dataclass
class ShardJob:
    info: ShardInfo
    stem: str
    out_dir: Path


def partition_ok(path: str, mod: int, rem: int) -> bool:
    """Stateless shard partition between lanes: every manifest path lands in exactly one of `mod` lanes."""
    return zlib.crc32(path.encode()) % mod == rem


def ordered_subsequence(sub: list[str], full: list[str]) -> bool:
    """True when `sub` is `full` with some items left out and the rest in the same order."""
    it = iter(full)
    return all(any(x == y for y in it) for x in sub)


def write_pair(jsonl: Path, rows: list[dict], npz: Path, arrays: dict, compressed: bool):
    """jsonl first, then npz, each tmp + fsync + rename: the npz's presence marks the shard done, so a kill can leave
    a jsonl without its npz (recomputed) but never an npz without its jsonl."""
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, jsonl)
    tmp = npz.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, npz)


def write_json_atomic(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def touch(path: Path | None):
    """Heartbeat: never fatal (a full disk must not kill a pass that is otherwise fine)."""
    if path is None:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.utime(path, None)
    except OSError:
        pass


def append_progress(path: Path | None, line: dict):
    """One JSON line per finished shard (`{stem, source, split, hours, wall_s, adopted, rows}`)."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _oom_types() -> tuple:
    try:
        import torch
    except ImportError:  # pragma: no cover - every pass imports torch; the runner itself does not need it
        return ()
    return (torch.OutOfMemoryError,)


class _Shard:
    def __init__(self, job: ShardJob, table, batches: list):
        self.job, self.table, self.batches = job, table, batches
        self.left = len(batches)
        self.fed = 0
        self.per_row: dict = {}
        self.t0 = time.time()
        self.stats = dict(batches=len(batches), bad_decode=0, oom_splits=0, gpu_s=0.0)


def run(todo: Callable[[], list[ShardJob]], *, load_table, plan_batches, prepare, gpu, finish,
        workers: int, prefetch: int, follow_marker: Path | None, heartbeat: Path | None,
        progress: Path | None, poll_s: float = 30.0, oom_split: bool = True, on_oom=None, log=print) -> dict:
    """Run every job `todo()` returns (and, in follow mode, every job it returns later). Returns totals."""
    oom_types = _oom_types()
    prefetch = max(1, int(prefetch))
    totals = dict(shards=0, rows=0, batches=0, bad_decode=0, oom_splits=0)
    t_start = time.time()
    submitted: set = set()
    jobs: deque = deque()
    first_call = [True]

    def refill():
        if not first_call[0] and follow_marker is None:
            return  # without follow the job list is fixed at the start
        first_call[0] = False
        for j in todo():
            key = (j.info.source, j.stem, str(j.out_dir))
            if key not in submitted:
                submitted.add(key)
                jobs.append(j)

    def run_gpu(sh: _Shard, inputs, ok):
        # The split runs outside the except block: inside it the traceback keeps the failed call's frames (and
        # their CUDA tensors) alive, so on_oom()/empty_cache() could not free them before the halves run.
        t = time.time()
        failed = False
        res = None
        try:
            res = gpu(inputs, ok, sh.table)
        except oom_types:
            if not oom_split or len(ok) <= 1:
                raise
            failed = True
        if not failed:
            sh.stats["gpu_s"] += time.time() - t  # only a successful call counts; split halves add their own
            if isinstance(res, dict):
                return res
            return {int(i): r for i, r in zip(ok, res)}
        inputs = res = None
        if on_oom is not None:
            on_oom()
        sh.stats["oom_splits"] += 1
        log(f"  {sh.job.info.source}/{sh.job.stem}: OOM split {len(ok)}")
        out = {}
        for half in np.array_split(np.asarray(ok), 2):
            inp, ok2 = prepare(sh.table, half)
            if inp is not None and len(ok2):
                out.update(run_gpu(sh, inp, ok2))
        return out

    def write(sh: _Shard):
        stats = dict(sh.stats, wall_s=time.time() - sh.t0)
        res = finish(sh.job, sh.table, sh.per_row, stats) or {}
        line = dict(stem=sh.job.stem, source=sh.job.info.source, split=sh.job.info.split,
                    hours=float(sh.job.info.hours), wall_s=round(stats["wall_s"], 3), adopted=False,
                    rows=len(sh.per_row))
        line.update(res)
        append_progress(progress, line)
        return line

    with ThreadPoolExecutor(max_workers=1) as reader, ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
            ThreadPoolExecutor(max_workers=1) as writer:
        preloaded: list = []  # [(job, table future)] at most one: the shard after the one being fed
        window: deque = deque()  # (shard, prep future) in GPU order
        writes: deque = deque()
        feeding: list = []  # [shard] whose batches are still entering the window

        def submit_write(sh: _Shard):
            while writes and writes[0].done():
                writes.popleft().result()  # surface writer errors early
            while len(writes) >= MAX_PENDING_WRITES:
                writes.popleft().result()
            writes.append(writer.submit(write, sh))
            totals["shards"] += 1
            totals["rows"] += len(sh.per_row)
            totals["batches"] += sh.stats["batches"]
            totals["bad_decode"] += sh.stats["bad_decode"]
            totals["oom_splits"] += sh.stats["oom_splits"]

        def preload_next():
            if not jobs:
                refill()
            if jobs and not preloaded:
                j = jobs.popleft()
                preloaded.append((j, reader.submit(load_table, j)))

        def open_next() -> bool:
            if not preloaded:
                preload_next()
            if not preloaded:
                return False
            job, fut = preloaded.pop()
            table = fut.result()
            sh = _Shard(job, table, list(plan_batches(job, table)))
            preload_next()  # the reader starts on the following shard while this one's batches are prepared
            if sh.left == 0:
                submit_write(sh)  # nothing to decode (e.g. every row > 30 s): still write the (empty) outputs
            else:
                feeding.append(sh)
            return True

        def fill():
            while len(window) < prefetch:
                if not feeding and not open_next():
                    return
                if not feeding:
                    continue
                sh = feeding[0]
                idx = sh.batches[sh.fed]
                sh.fed += 1
                if sh.fed == len(sh.batches):
                    feeding.pop(0)
                window.append((sh, idx, pool.submit(prepare, sh.table, idx)))

        while True:
            fill()
            if not window:
                if follow_marker is None:
                    break
                marker_seen = Path(follow_marker).exists()  # checked before todo(): no shard can slip in between
                refill()
                if jobs:
                    continue
                if marker_seen:
                    break
                touch(heartbeat)
                time.sleep(poll_s)
                continue
            sh, idx, fut = window.popleft()
            inputs, ok = fut.result()
            sh.stats["bad_decode"] += len(idx) - len(ok)
            if inputs is not None and len(ok):
                sh.per_row.update(run_gpu(sh, inputs, ok))
            touch(heartbeat)
            sh.left -= 1
            if sh.left == 0:
                submit_write(sh)
        while writes:
            writes.popleft().result()
    totals["wall_s"] = time.time() - t_start
    return totals
