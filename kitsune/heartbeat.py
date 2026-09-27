"""Heartbeat files of the full-data runs: a process shows it is alive by touching a file; a reader looks at its mtime.

Two layers (the full-run build contract, section 1.3; the label box's label_hb is the model):

  per item   the queue gives every child KITSUNE_HEARTBEAT=$KITSUNE_STATE/hb/<item> (kitsune.fullrun.item_hb_path). The
             trainer beats once per logged step and once per eval batch (Beating around its featuriser), 05 once per
             batch, the store build and bounded waits under beating(). The queue's stall check kills and resumes an
             item whose file has gone stale, so a hung trainer costs one item retry, not the whole box.
  per box    only the controllers (bootstrap, the queue's poll, supervise's bounded finish calls) touch
             $KITSUNE_STATE/train_hb; vast/watchdog.sh reads it (KITSUNE_WATCHDOG_HB_FILE) and stops the box, or only
             alerts, when it goes stale.

The file is empty: readers use its mtime (age_s), so a beat is one utime and there is no content to parse or to find
half-written. Every function is a no-op without a path (neither an argument nor $KITSUNE_HEARTBEAT set): code that
beats runs unchanged on the laptop, in the tests and in any run outside a full box.

  beat(path=None, *, force=False)          touch the file, at most once per MIN_INTERVAL_S per path unless force, so
                                           a per-step call costs a dict lookup on most steps
  beating(path=None, every_s, max_s)       a context manager: a daemon thread beats at entry and every every_s while
                                           the block runs. max_s BOUNDS it: after max_s the beats stop while the block
                                           goes on, so a truly hung call (an upload, a store build) still goes stale
                                           and its stall check or watchdog fires. Only an explicit None is unbounded:
                                           a max_s that is not a finite number (NaN, inf, "abc") counts as 0, the entry
                                           beat only, because a bound lost to a bad value would keep a hung box alive.
                                           The thread is joined for at most 1 s at exit
  Beating(fn, path=None)                   wraps a featuriser: every call beats, then calls fn. The evaluators call
                                           the featuriser once per batch, so wrapping it gives per-batch beats with no
                                           edit to kitsune/evaluate.py or kitsune/ctc_eval.py; attributes delegate to
                                           fn
  age_s(path)                              seconds since the last beat, None when there is no file

Nothing here ever raises: a heartbeat that cannot be written must never end the run it reports on (an unwritable path
only makes the file go stale, which is what its reader is for). Stdlib only.
"""
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ENV = "KITSUNE_HEARTBEAT"
MIN_INTERVAL_S = 5.0
# beating()'s interval bounds: a zero, negative or NaN every_s would spin the thread (Event.wait(nan) returns at
# once), an infinite or huge one would kill it (Event.wait raises OverflowError past threading.TIMEOUT_MAX)
_MIN_EVERY_S = 0.01
_DEFAULT_EVERY_S = 30.0
_JOIN_S = 1.0  # beating()'s bound on joining its thread at exit

_last: dict[str, float] = {}  # resolved path -> time.monotonic() of its last touch (the rate limit)
_lock = threading.Lock()


def _resolve(path) -> str | None:
    """The path to beat: the argument, else $KITSUNE_HEARTBEAT; None (no-op) when neither is set or it is empty."""
    try:
        p = os.environ.get(ENV) if path is None else os.fspath(path)
    except TypeError:
        return None
    return p or None


def beat(path: str | os.PathLike | None = None, *, force: bool = False) -> None:
    """Touch the heartbeat file (create it and its parent dirs if missing), at most once per MIN_INTERVAL_S per path
    unless force. No path -> nothing. Never raises."""
    try:
        p = _resolve(path)
        if p is None:
            return
        now = time.monotonic()
        key = os.path.abspath(p)
        with _lock:
            last = _last.get(key)
            if not force and last is not None and now - last < MIN_INTERVAL_S:
                return
            _last[key] = now
        f = Path(p)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch(exist_ok=True)  # creates it, or sets its mtime to now
    except Exception:  # noqa: BLE001  a heartbeat never ends the process it reports on
        pass


def _finite(v, default: float) -> float:
    """v as a finite float; default when it is not a number or not finite (NaN, +-inf)."""
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    return f if math.isfinite(f) else default


@contextmanager
def beating(path=None, every_s: float = 30.0, max_s: float | None = None):
    """Beat at entry and every every_s from a daemon thread until the block exits or max_s has passed; after max_s the
    thread stops beating while the block goes on (a hung call still goes stale). No path -> a plain context, no
    thread. The thread is joined for at most 1 s at exit.

    Bad values never lift the bound: max_s None (explicit) is the only unbounded case; a max_s that is not a finite
    number counts as 0 (the entry beat only). An every_s that is not a finite number is the default 30 s; it is kept
    between 0.01 s and threading.TIMEOUT_MAX."""
    p = _resolve(path)
    if p is None:
        yield
        return
    every = min(max(_finite(every_s, _DEFAULT_EVERY_S), _MIN_EVERY_S), threading.TIMEOUT_MAX)
    limit = None if max_s is None else _finite(max_s, 0.0)
    stop = threading.Event()
    t0 = time.monotonic()
    beat(p, force=True)  # at entry, in the caller's thread: the file is fresh before the block starts

    def run():
        while not stop.wait(every):
            if limit is not None and time.monotonic() - t0 > limit:
                return
            beat(p, force=True)

    th = None
    try:
        th = threading.Thread(target=run, name="kitsune-heartbeat", daemon=True)
        th.start()
    except Exception:  # noqa: BLE001  no thread (e.g. the interpreter is shutting down): the block still runs
        th = None
    try:
        yield
    finally:
        stop.set()
        if th is not None:
            th.join(_JOIN_S)


class Beating:
    """A featuriser wrapper: each call beats (rate-limited, beat()) and then returns fn(*a, **kw); every other
    attribute is fn's. The evaluators call the featuriser once per batch, so this gives per-batch beats."""

    def __init__(self, fn, path=None):
        self._fn = fn
        self._path = path

    def __call__(self, *a, **kw):
        beat(self._path)
        return self._fn(*a, **kw)

    def __getattr__(self, name):
        # only reached for names the wrapper itself lacks; its own two never delegate (copy/pickle build an instance
        # without __init__ and look up attributes before _fn exists)
        if name in ("_fn", "_path"):
            raise AttributeError(name)
        return getattr(self._fn, name)

    def __repr__(self):
        return f"Beating({self._fn!r})"


def age_s(path) -> float | None:
    """Seconds since the file's mtime (now - mtime); None when it is missing or unreadable, or there is no path."""
    try:
        p = _resolve(path)
        if p is None:
            return None
        return time.time() - os.stat(p).st_mtime
    except Exception:  # noqa: BLE001
        return None
