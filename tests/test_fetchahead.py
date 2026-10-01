"""Tests for kitsune/fetchahead.py: the audio rebuild's downloads ahead of the ingest (decision F3).

The rebuild's output must not depend on how many files download at once, so take() hands them out strictly in plan
order whatever order they finish in; at most `ahead` are claimed and not taken (the temp disk bound); transient Hub
errors retry with backoff, permanent ones surface at once and in plan order; a hung download fails its take
(DownloadStalled) instead of hanging the box; and a failed or killed ingest exits promptly, never waiting for a
download it will not read (the daemon-thread choice). The fetches are fakes writing small files. CPU only, no network.
"""
import errno
import os
import random
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest
from huggingface_hub import errors as hf

from kitsune import fetchahead
from kitsune.fetchahead import DownloadStalled, FetchAhead

ROOT = Path(__file__).resolve().parents[1]


def http_error(status: int, retry_after: str | None = None, cls=hf.HfHubHTTPError):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    resp = httpx.Response(status, headers=headers, request=httpx.Request("GET", "https://huggingface.co/x"))
    return cls(f"{status} from the Hub", response=resp)


class Files:
    """A fake fetch that writes <dir>/<key> after `latency(key)` seconds; counts the downloads running at once."""

    def __init__(self, d: Path, latency=lambda k: 0.0):
        self.d, self.latency = d, latency
        d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.running = self.max_running = 0
        self.calls: list[str] = []
        self.max_on_disk = 0

    def __call__(self, key: str) -> Path:
        with self.lock:
            self.calls.append(key)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            time.sleep(self.latency(key))
            p = self.d / key
            p.write_text(f"content of {key}", encoding="utf-8")
            with self.lock:
                self.max_on_disk = max(self.max_on_disk, sum(1 for _ in self.d.iterdir()))
            return p
        finally:
            with self.lock:
                self.running -= 1


def no_fetchahead_threads(timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not any(t.name.startswith("kitsune-fetchahead-") and t.is_alive() for t in threading.enumerate()):
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def fast_backoff(monkeypatch):
    monkeypatch.setattr(fetchahead, "BACKOFF_S", (0.01, 0.01, 0.01, 0.01))


def test_take_returns_in_plan_order_whatever_finishes_first(tmp_path):
    keys = [f"k{i}" for i in range(10)]
    rng = random.Random(7)
    lat = {k: (10 - i) * 0.01 + rng.uniform(0, 0.02) for i, k in enumerate(keys)}  # the later, the faster
    files = Files(tmp_path / "dl", lambda k: lat[k])
    with FetchAhead(keys, files, ahead=4, poll_s=0.01, log=lambda m: None) as fa:
        got = [fa.take(k) for k in keys]
    assert [p.name for p in got] == keys
    assert [p.read_text(encoding="utf-8") for p in got] == [f"content of {k}" for k in keys]
    assert sorted(files.calls) == sorted(keys) and len(files.calls) == len(keys)
    assert no_fetchahead_threads()


def test_at_most_ahead_downloads_and_ahead_plus_one_files_on_disk(tmp_path):
    """The first K downloads meet at a barrier (it times out unless all K run at once); never K+1 run, and with the
    consumer freeing each file after use there are never more than K+1 files on disk."""
    k_ahead, keys = 3, [f"k{i}" for i in range(12)]
    barrier = threading.Barrier(k_ahead, timeout=10)
    inner = Files(tmp_path / "dl", lambda k: 0.01)

    def fetch(key):
        if int(key[1:]) < k_ahead:
            barrier.wait()
        return inner(key)

    freed = []
    fa = FetchAhead(keys, fetch, ahead=k_ahead, free=lambda p: p.unlink(), poll_s=0.01, log=lambda m: None)
    for k in keys:
        p = fa.take(k)
        time.sleep(0.02)  # the ingest: the pool fills up meanwhile
        assert sum(1 for _ in inner.d.iterdir()) <= k_ahead + 1
        p.unlink()
        freed.append(k)
    stats = fa.close()
    assert inner.max_running == k_ahead and inner.max_on_disk <= k_ahead + 1
    assert stats["taken"] == len(keys) and stats["discarded"] == 0 and stats["retries"] == 0
    assert not barrier.broken


def test_transient_errors_retry_with_backoff_and_retry_after(tmp_path):
    """503, 503, 429 with Retry-After 45, then success: the waits are the policy's 10 and 30 s (+-20 %) and the
    header's 45 s. LocalEntryNotFoundError (a connection that died at the HEAD request) and a truncated download's
    consistency OSError are transient too."""
    errs = [http_error(503), http_error(503), http_error(429, "45")]
    waits, logged = [], []

    def fn():
        if errs:
            raise errs.pop(0)
        return "done"

    assert fetchahead.with_retries(fn, "x", wait=waits.append, log=logged.append, rng=random.Random(1)) == "done"
    assert len(waits) == 3 and 8 <= waits[0] <= 12 and 24 <= waits[1] <= 36 and waits[2] == 45
    assert "attempt 1/5 failed (HfHubHTTPError: 503 from the Hub); retrying in" in logged[0]
    assert fetchahead.backoff_s(1, http_error(429, "9999"), random.Random(1)) == fetchahead.BACKOFF_MAX_S
    for e in (http_error(500), http_error(502), http_error(429), hf.LocalEntryNotFoundError("connection error"),
              OSError("Consistency check failed: file should be of size 5 but has size 3"), httpx.ReadTimeout("t"),
              httpx.ConnectError("refused"), RuntimeError("hf_xet: stream reset"), ValueError("bad chunk")):
        assert not fetchahead.permanent(e), e
    for e in (http_error(404, cls=hf.RepositoryNotFoundError), http_error(404, cls=hf.RemoteEntryNotFoundError),
              http_error(404, cls=hf.RevisionNotFoundError), http_error(401), http_error(403), http_error(410),
              OSError(errno.ENOSPC, "No space left on device"), TypeError("x"), AssertionError("fake"),
              KeyError("k"), SystemExit("guard"), KeyboardInterrupt()):
        assert fetchahead.permanent(e), e


def test_pool_retries_a_transient_error_and_counts_it(tmp_path, fast_backoff):
    failed = set()
    files = Files(tmp_path / "dl")

    def fetch(key):
        if key == "k1" and key not in failed:
            failed.add(key)
            raise http_error(503)
        return files(key)

    logged = []
    with FetchAhead(["k0", "k1", "k2"], fetch, ahead=2, poll_s=0.01, log=logged.append) as fa:
        assert [fa.take(k).name for k in ("k0", "k1", "k2")] == ["k0", "k1", "k2"]
    assert fa.stats["retries"] == 1 and any("k1: attempt 1/5 failed" in m for m in logged)
    assert "3 of 3 planned files" in logged[-1] and "1 retries" in logged[-1]


def test_permanent_errors_surface_in_order_and_are_not_retried(tmp_path, fast_backoff):
    keys = [f"k{i}" for i in range(6)]
    files = Files(tmp_path / "dl")
    calls = []
    err = http_error(404, cls=hf.RemoteEntryNotFoundError)

    def fetch(key):
        calls.append(key)
        if key == "k3":
            raise err
        return files(key)

    freed = []
    fa = FetchAhead(keys, fetch, ahead=6, free=lambda p: freed.append(p.name), poll_s=0.01, log=lambda m: None)
    assert [fa.take(k).name for k in keys[:3]] == keys[:3]
    with pytest.raises(hf.RemoteEntryNotFoundError) as e:
        fa.take("k3")
    assert e.value is err and calls.count("k3") == 1
    fa.close()
    assert sorted(freed) == ["k4", "k5"]  # fetched past the error, never taken: discarded
    assert no_fetchahead_threads()


def test_retries_exhausted_raise_the_last_error(tmp_path, fast_backoff):
    n = [0]

    def fetch(key):
        n[0] += 1
        raise http_error(500 + n[0])

    with FetchAhead(["k0"], fetch, ahead=2, tries=3, poll_s=0.01, log=lambda m: None) as fa:
        with pytest.raises(hf.HfHubHTTPError, match="503 from the Hub"):
            fa.take("k0")
    assert n[0] == 3


def test_guard_waits_for_the_ingest_then_fails_the_item_as_SystemExit_in_order(tmp_path):
    """The disk guard refuses k2 while the ingest has not reached it (space frees as earlier files go) and asks again
    once k2 is next; it refuses k3 for good, which surfaces as SystemExit at k3's take, after k0..k2. It sees the
    other downloads in flight."""
    files = Files(tmp_path / "dl", lambda k: 0.01)
    reached, seen = set(), []

    def guard(key, in_flight):
        seen.append((key, in_flight))
        if key == "k2" and "k1" not in reached:
            return "k2: not yet"
        if key == "k3":
            return "only 3 GB free on the data disk; stopping before k3"
        return None

    with FetchAhead([f"k{i}" for i in range(4)], files, ahead=4, guard=guard, poll_s=0.01, log=lambda m: None) as fa:
        assert fa.take("k0").name == "k0"
        assert fa.take("k1").name == "k1"
        reached.add("k1")
        assert fa.take("k2").name == "k2"
        with pytest.raises(SystemExit, match="GB free.*stopping before k3"):
            fa.take("k3")
    assert [k for k, _ in seen].count("k2") >= 2 and "k2" in files.calls and "k3" not in files.calls
    assert all(0 <= n <= 3 for _, n in seen)


def test_take_refuses_a_key_out_of_order(tmp_path):
    with FetchAhead(["a", "b"], Files(tmp_path / "dl"), ahead=1, poll_s=0.01, log=lambda m: None) as fa:
        with pytest.raises(KeyError):
            fa.take("b")
        assert fa.next_key() == "a" and fa.take("a").name == "a" and fa.next_key() == "b"


def test_stall_raises_DownloadStalled_and_abort_returns_promptly(tmp_path):
    """A download that never returns: the take beats while the file grows, then raises once it has not grown for
    stall_s (on an injected clock), and close(abort=True) does not wait for the stuck thread beyond abort_join_s."""
    release = threading.Event()

    def fetch(key):
        release.wait(30)
        raise RuntimeError("released")

    sizes = iter([0, 10, 20, 30] + [30] * 10_000)
    t = [0.0]

    def clock():
        t[0] += 50.0
        return t[0]

    beats, logged = [], []
    fa = FetchAhead(["big"], fetch, ahead=1, progress=lambda k: next(sizes), clock=clock, stall_s=500, poll_s=0.01,
                    wait_log_s=200, abort_join_s=0.5, log=logged.append)
    try:
        with pytest.raises(DownloadStalled, match="big has not grown for"):
            fa.take("big", beat=lambda: beats.append(1))
        assert len(beats) == 3  # 0 -> 10 -> 20 -> 30
        assert any("waiting for big: 0 MB so far, 1 in flight, 0 ready" in m for m in logged)
        t0 = time.monotonic()
        stats = fa.close(abort=True)
        assert time.monotonic() - t0 < 5 and stats["alive_after_close"] == 1
        assert "still running: kitsune-fetchahead-0" in logged[-1]
    finally:
        release.set()
    assert no_fetchahead_threads()


def test_a_retry_restarts_the_stall_clock(tmp_path, monkeypatch):
    """Backoff waits make no bytes: a key whose attempts keep failing is not a stall while attempts still start."""
    monkeypatch.setattr(fetchahead, "BACKOFF_S", (0.1,) * 4)  # 8 failures: >= 0.64 s without a byte, > stall_s
    n = [0]

    def fetch(key):
        n[0] += 1
        if n[0] <= 8:
            raise http_error(503)
        (tmp_path / key).write_text("x", encoding="utf-8")
        return tmp_path / key

    with FetchAhead(["k"], fetch, ahead=1, progress=lambda k: 0, stall_s=0.6, poll_s=0.01, tries=10,
                    log=lambda m: None) as fa:
        assert fa.take("k").name == "k"
    assert n[0] == 9 and fa.stats["retries"] == 8


def test_close_joins_threads_and_discards_unconsumed(tmp_path):
    keys = [f"k{i}" for i in range(6)]
    for abort in (False, True):
        files = Files(tmp_path / f"dl{abort}")
        freed = []
        fa = FetchAhead(keys, files, ahead=6, free=lambda p: (freed.append(p.name), p.unlink()), poll_s=0.01,
                        log=lambda m: None)
        assert [fa.take(k).name for k in keys[:2]] == keys[:2]
        end = time.monotonic() + 10
        while len(files.calls) < 6 and time.monotonic() < end:
            time.sleep(0.01)
        stats = fa.close(abort=abort)
        assert no_fetchahead_threads()
        on_disk = sorted(p.name for p in files.d.iterdir())
        if abort:  # kept for the resumed run (cache hits there)
            assert freed == [] and on_disk == keys and stats["discarded"] == 0
        else:
            assert sorted(freed) == keys[2:] and on_disk == keys[:2] and stats["discarded"] == 4
        assert fa.close() is stats  # idempotent
        with pytest.raises(KeyError):
            fa.take("k2")


def test_backoff_wait_is_interrupted_by_close(tmp_path, monkeypatch):
    monkeypatch.setattr(fetchahead, "BACKOFF_S", (1000,) * 4)
    monkeypatch.setattr(fetchahead, "BACKOFF_MAX_S", 1000)
    started = threading.Event()

    def fetch(key):
        started.set()
        raise http_error(503)

    fa = FetchAhead(["k"], fetch, ahead=1, poll_s=0.01, log=lambda m: None)
    assert started.wait(10)
    t0 = time.monotonic()
    fa.close()
    assert time.monotonic() - t0 < 5 and no_fetchahead_threads(1)


HUNG_SCRIPT = textwrap.dedent("""
    import sys, threading, time
    sys.path.insert(0, {root!r})
    from pathlib import Path
    from kitsune.fetchahead import FetchAhead
    d = Path({d!r})

    def fetch(key):
        if key == "k1":
            threading.Event().wait()  # a download that never returns
        p = d / key
        p.write_text("x")
        return p

    fa = FetchAhead(["k0", "k1", "k2"], fetch, ahead=3, poll_s=0.05, abort_join_s=1.0)
    print("took", fa.take("k0").name, flush=True)
    if {mode!r} == "raise":
        with fa:
            raise RuntimeError("the ingest failed")
    time.sleep(60)
""")


def run_hung(tmp_path: Path, mode: str) -> subprocess.Popen:
    script = tmp_path / f"hung_{mode}.py"
    script.write_text(HUNG_SCRIPT.format(root=str(ROOT), d=str(tmp_path), mode=mode), encoding="utf-8")
    return subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_process_exits_promptly_when_the_ingest_fails_during_a_hung_download(tmp_path):
    """The daemon-thread choice: an ingest error while another file's download hangs ends the process (non-zero)
    in about abort_join_s; a ThreadPoolExecutor's exit join would wait for the hung download forever."""
    p = run_hung(tmp_path, "raise")
    try:
        out, err = p.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        raise AssertionError("the process hung on its download thread")
    assert p.returncode != 0 and "took k0" in out and "the ingest failed" in err, (out, err)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals: the box's `timeout` sends SIGTERM")
def test_sigterm_kills_it_promptly(tmp_path):
    p = run_hung(tmp_path, "sleep")
    try:
        assert p.stdout.readline().strip() == "took k0"
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=5)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == -signal.SIGTERM
