"""Fetch-ahead for the audio rebuild's upstream downloads (decision F3): several files come down at once while the
ingest still reads them one at a time, in plan order.

Why: box 1 (instance 53711872, 2026-10-01) rebuilt the full extent's 571 GB in 3.1 h, one hf_hub_download after the
other (~50-70 MB/s, CPU load < 1 on 32 threads), while its download gate measured 129 MB/s on single files. Each input
was downloaded, THEN ingested, so the link sat idle during every ingest and the CPU during every download.
scripts/01_prepare_data.py now hands the inputs of an ingest step to a FetchAhead, which keeps up to `ahead` of them
downloading or downloaded ahead of the ingest; the ingest takes them strictly in plan order (take), so its writes, and
with them the shards, sidecars, manifest and progress files, are those of the one-at-a-time path, byte for byte.

Guarantees:
- order: take(key) only ever returns keys[taken], whatever order the downloads finish in; an error surfaces at the
  take of the key that failed, i.e. where the one-at-a-time path would have met it.
- disk: a worker claims the next key only while fewer than `ahead` keys are claimed and not yet taken, so at most
  `ahead` files are downloading or waiting on disk, plus the one the ingest is reading.
- retries: each download is tried TRIES times. A transient error (HF 429 and 5xx, a dead or reset connection, a
  read timeout, a size-consistency failure, hf_xet errors) waits BACKOFF_S (+-20 % jitter; a 429's Retry-After is
  honoured), both capped at BACKOFF_MAX_S. A permanent one (a missing repo, revision or file, 401/403/404/410, a full
  disk, a programming error) is raised at once: no retry can fix it.
- a hung download: the take that waits for it raises DownloadStalled once the file has not grown for stall_s. The
  stuck thread cannot be killed from Python, and a second attempt into the same cache dir would wait on the first
  one's hf file lock, so the process exits instead; bootstrap's `retry 3 timeout` re-runs 01, which resumes (finished
  inputs are skipped, files completed ahead are cache hits).
- shutdown: the workers are daemon threads, so a failed or SIGTERMed ingest never waits for a download it will not
  read (a ThreadPoolExecutor's atexit join would: with a hung download the process never exited). close() wakes
  every worker (a backoff wait included) and joins them; a normal close frees the files nobody took, an abort (the
  ingest raised) keeps them for the resumed run, where they are cache hits.

The summary line close() logs ("downloads: T of P planned files (G GB) in W s, up to K ahead; the ingest waited S s")
measures the effect on a box: W - S is the ingest's own time, S what the link still costs it, so K can be re-tuned
(KITSUNE_DOWNLOAD_AHEAD) without a commit. Stdlib only at import (huggingface_hub's error classes are imported where an
error is classified).
"""
import errno
import random
import threading
import time
from pathlib import Path

DEFAULT_AHEAD = 6  # files downloading or ready ahead of the ingest (F3); 0 = one at a time (the pre-F3 path)
MAX_AHEAD = 16
TRIES = 5  # attempts per file
BACKOFF_S = (10, 30, 90, 270)  # the wait after attempt 1, 2, 3, 4 (+-JITTER)
BACKOFF_MAX_S = 300
JITTER = 0.2
STALL_S = 900  # a file that has not grown for this long fails its take (6 x 2.5 GB at the gate's 31.7 MB/s: ~473 s)
POLL_S = 15  # how often a waiting take checks the file's growth
WAIT_LOG_S = 60  # how often a waiting take logs what it waits for
ABORT_JOIN_S = 10  # how long close(abort=True) waits for the workers (they are daemon threads: they die with the process)

PERMANENT_STATUS = (401, 403, 404, 410)  # the bootstrap helper's refused() rule: retrying cannot fix them


def _print(msg: str):
    print(msg, flush=True)  # 01's output goes to kitsune.log through a pipe: unflushed lines arrive late


class DownloadStalled(RuntimeError):
    """A download made no progress for stall_s."""


class _Cancelled(Exception):
    """close() came during a backoff wait."""


def status_of(e: BaseException) -> int | None:
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else None


def permanent(e: BaseException) -> bool:
    """True for an error no retry can fix. Everything else counts as transient: 429, 5xx, httpx transport errors
    (connect, read timeout, reset), LocalEntryNotFoundError (how hf_hub_download reports a connection that died at its
    HEAD request), the size-consistency OSError of a truncated download, hf_xet's errors. Mind the class tree
    (huggingface_hub 1.23): HfHubHTTPError is an OSError and an httpx.HTTPError, LocalEntryNotFoundError a
    FileNotFoundError, so the OSError rules below must not catch them."""
    if not isinstance(e, Exception):
        return True  # KeyboardInterrupt, SystemExit (a guard's refusal)
    try:
        from huggingface_hub import errors as hf
        not_found = tuple(c for c in (getattr(hf, n, None) for n in (
            "RepositoryNotFoundError", "RevisionNotFoundError", "RemoteEntryNotFoundError", "GatedRepoError",
            "DisabledRepoError")) if c is not None)
        local_miss = getattr(hf, "LocalEntryNotFoundError", None)
    except ImportError:  # pragma: no cover - 01 always has it
        not_found, local_miss = (), None
    if local_miss is not None and isinstance(e, local_miss):
        return False
    if not_found and isinstance(e, not_found):
        return True
    if status_of(e) in PERMANENT_STATUS:
        return True
    if isinstance(e, OSError) and e.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
        return True
    # programming errors (a fake that asserts, a bad argument): the same on every attempt
    return isinstance(e, (TypeError, AttributeError, NameError, ImportError, AssertionError, LookupError))


def retry_after(e: BaseException) -> float | None:
    """A 429's Retry-After header in seconds (the numeric form only), else None."""
    if status_of(e) != 429:
        return None
    try:
        v = float(e.response.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None
    return v if v >= 0 else None


def backoff_s(attempt: int, e: BaseException, rng=random) -> float:
    """The wait after failed attempt `attempt` (1-based)."""
    ra = retry_after(e)
    if ra is not None:
        return min(ra, BACKOFF_MAX_S)
    base = BACKOFF_S[min(attempt, len(BACKOFF_S)) - 1]
    return min(base * (1 + rng.uniform(-JITTER, JITTER)), BACKOFF_MAX_S)


def _short(e: BaseException, n: int = 300) -> str:
    s = " ".join(str(e).split())
    return s if len(s) <= n else s[:n] + "..."


def with_retries(fn, what: str, *, tries: int = TRIES, log=_print, wait=time.sleep, rng=random, on_retry=None):
    """fn() with the retry policy above; `wait(seconds)` sleeps between attempts (a truthy return cancels: close()).
    Re-raises the last error, or a permanent one at once."""
    for k in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  classified below
            if permanent(e) or k == tries:
                raise
            d = backoff_s(k, e, rng)
            log(f"  {what}: attempt {k}/{tries} failed ({type(e).__name__}: {_short(e)}); retrying in {d:.0f} s")
            if on_retry is not None:
                on_retry()
            if wait(d):
                raise _Cancelled(what) from e
    raise AssertionError("unreachable")  # pragma: no cover


def dir_bytes(d: Path) -> int:
    """Bytes under d now, partial (.incomplete) files included: how a waiting take sees a download grow."""
    try:
        return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
    except OSError:
        return 0


class FetchAhead:
    """Download `keys` (in order, unique) up to `ahead` at a time with `fetch(key) -> path`, for a consumer that takes
    them in that order. `free(path)` deletes an untaken file at a normal close; `guard(key, in_flight_others)` may
    return a message that refuses the key (the disk guard: it fails at that key's take as SystemExit(message));
    `progress(key)` is the bytes of the key's download so far (stall detection and the heartbeat)."""

    def __init__(self, keys, fetch, *, ahead: int = DEFAULT_AHEAD, free=None, guard=None, progress=None,
                 name: str = "downloads", log=_print, clock=time.monotonic, stall_s: float = STALL_S,
                 poll_s: float = POLL_S, wait_log_s: float = WAIT_LOG_S, abort_join_s: float = ABORT_JOIN_S,
                 tries: int = TRIES, rng=None):
        self.keys = list(keys)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("FetchAhead keys must be unique")
        if ahead < 1:
            raise ValueError("ahead must be >= 1 (0 is the caller's one-at-a-time path)")
        self.ahead, self.fetch, self.free, self.guard, self.progress = ahead, fetch, free, guard, progress
        self.name, self.log, self.clock = name, log, clock
        self.stall_s, self.poll_s, self.wait_log_s, self.abort_join_s = stall_s, poll_s, wait_log_s, abort_join_s
        self.tries, self.rng = tries, rng or random.Random()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._next = 0  # index of the next key a worker claims
        self._taken = 0  # keys handed to the consumer
        self._state: dict[str, tuple[str, object]] = {}  # key -> (running|ready|failed, path|error)
        self._sizes: dict[str, int] = {}
        self._attempted: dict[str, float] = {}  # key -> clock() at its latest attempt (a retry is no stall)
        self._closed = False
        self.stats = dict(planned=len(self.keys), taken=0, bytes=0, retries=0, discarded=0, waited_s=0.0,
                          alive_after_close=0)
        self._t0 = clock()
        self._threads = [threading.Thread(target=self._work, name=f"kitsune-fetchahead-{i}", daemon=True)
                         for i in range(min(ahead, len(self.keys)))]
        for t in self._threads:
            t.start()

    # ---------------------------------------------------------------------------------------------- the workers
    def _in_flight(self) -> int:
        return sum(1 for s, _ in self._state.values() if s == "running")

    def _work(self):
        while True:
            with self._cv:
                while (not self._stop.is_set() and self._next < len(self.keys)
                       and self._next - self._taken >= self.ahead):
                    self._cv.wait()
                if self._stop.is_set() or self._next >= len(self.keys):
                    return
                i, key = self._next, self.keys[self._next]
                self._next += 1
                self._state[key] = ("running", None)
            try:
                result = ("ready", self._download(i, key))
            except BaseException as e:  # noqa: BLE001  handed to the consumer's take, in plan order
                result = ("failed", e)
            with self._cv:
                if result[0] == "ready":
                    p = Path(result[1])
                    self._sizes[key] = p.stat().st_size if p.is_file() else 0
                self._state[key] = result
                self._cv.notify_all()

    def _download(self, i: int, key):
        def attempt():
            with self._cv:
                self._attempted[key] = self.clock()
            if self.guard is not None:
                problem = self._guard(i, key)
                if problem:
                    raise SystemExit(problem)
            return self.fetch(key)

        def wait(s):
            return self._stop.wait(s)

        def counted():
            with self._cv:
                self.stats["retries"] += 1

        return with_retries(attempt, f"{self.name}: {key}", tries=self.tries, log=self.log, wait=wait, rng=self.rng,
                            on_retry=counted)

    def _guard(self, i: int, key) -> str | None:
        """The guard's verdict for key i. A refusal of a key the ingest has not reached yet waits for the ingest to
        get there (earlier files are freed on the way) and asks again: the one-at-a-time path refuses only the
        next download, on the free space of that moment."""
        while True:
            with self._cv:
                others = self._in_flight() - 1
            problem = self.guard(key, max(others, 0))
            if not problem:
                return None
            with self._cv:
                if self._taken >= i or self._stop.is_set():
                    return problem
                self._cv.wait()

    # ---------------------------------------------------------------------------------------------- the consumer
    def next_key(self):
        """The key the next take must ask for (None once every key was taken)."""
        with self._cv:
            return self.keys[self._taken] if self._taken < len(self.keys) else None

    def take(self, key, beat=None):
        """The downloaded path of `key`, which must be the next key in plan order (else KeyError: the caller then
        downloads it itself). Waits for it; re-raises its download error; raises DownloadStalled when its file has
        not grown for stall_s. `beat()` is called whenever the file grew (the ingest's heartbeat)."""
        with self._cv:
            if self._closed or self._taken >= len(self.keys) or self.keys[self._taken] != key:
                raise KeyError(key)
        t0 = last_growth = last_log = self.clock()
        last_bytes = -1
        try:
            while True:
                with self._cv:
                    state, value = self._state.get(key, ("pending", None))
                    if state in ("ready", "failed"):
                        self._taken += 1
                        self._cv.notify_all()
                        if state == "failed":
                            raise value
                        self.stats["taken"] += 1
                        self.stats["bytes"] += self._sizes.get(key, 0)
                        return value
                    self._cv.wait(self.poll_s)
                    state = self._state.get(key, ("pending", None))[0]
                    if state in ("ready", "failed"):
                        continue
                now = self.clock()
                b = self.progress(key) if self.progress is not None else 0
                if b > last_bytes:
                    if last_bytes >= 0 and beat is not None:
                        beat()
                    last_bytes, last_growth = b, now
                with self._cv:  # a new attempt (after a backoff) restarts the stall clock: its wait was no hang
                    last_growth = max(last_growth, self._attempted.get(key, last_growth))
                if now - last_growth >= self.stall_s:
                    raise DownloadStalled(
                        f"{self.name}: {key} has not grown for {now - last_growth:.0f} s ({b / 1e6:.0f} MB so far, "
                        f"{state}); stopping so that a fresh process retries it")
                if now - last_log >= self.wait_log_s:
                    with self._cv:
                        flying = self._in_flight()
                        ready = sum(1 for s, _ in self._state.values() if s == "ready")
                    self.log(f"  {self.name}: waiting for {key}: {b / 1e6:.0f} MB so far, {flying} in flight, "
                             f"{ready} ready")
                    last_log = now
        finally:
            self.stats["waited_s"] += self.clock() - t0

    # ---------------------------------------------------------------------------------------------- the end
    def close(self, abort: bool = False) -> dict:
        """Stop the workers and join them (abort: at most abort_join_s, else stall_s, for a download in flight that
        nobody will read). A normal close frees the files nobody took; an abort keeps them on disk, where the resumed
        run finds them as cache hits. Logs and returns the stats. Idempotent."""
        with self._cv:
            if self._closed:
                return self.stats
            self._closed = True
            self._stop.set()
            self._cv.notify_all()
        deadline = time.monotonic() + (self.abort_join_s if abort else self.stall_s)
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))
        alive = [t.name for t in self._threads if t.is_alive()]
        self.stats["alive_after_close"] = len(alive)
        with self._cv:
            taken = set(self.keys[:self._taken])
            untaken = [(k, v) for k, (s, v) in self._state.items() if s == "ready" and k not in taken]
        if not abort and self.free is not None:
            for _, path in untaken:
                try:
                    self.free(Path(path))
                except OSError as e:
                    self.log(f"  {self.name}: could not free {path}: {e}")
                self.stats["discarded"] += 1
        s = self.stats
        wall = self.clock() - self._t0
        self.log(f"  {self.name}: downloads: {s['taken']} of {s['planned']} planned files ({s['bytes'] / 1e9:.1f} GB) "
                 f"in {wall:.0f} s, up to {self.ahead} ahead; the ingest waited {s['waited_s']:.0f} s; "
                 f"{s['retries']} retries; {s['discarded']} discarded"
                 + (f"; {len(untaken)} kept for the resumed run" if abort and untaken else "")
                 + (f"; still running: {', '.join(alive)}" if alive else ""))
        return s

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(abort=exc_type is not None)
        return False
