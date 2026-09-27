"""kitsune/heartbeat.py: the full runs' heartbeat files (touch + mtime). No-op without a path, rate-limited, bounded by
max_s, never raises. CPU only, stdlib only; the timings are generous because other test processes share the host."""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT  # noqa: E402

from kitsune import heartbeat as hb  # noqa: E402

OLD = 1_000_000_000.0  # an mtime far in the past: a beat moves it to now


def _age_old(p: Path):
    os.utime(p, (OLD, OLD))


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def _wait_for(cond, timeout: float = 5.0) -> bool:
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_constants():
    assert hb.ENV == "KITSUNE_HEARTBEAT"
    assert hb.MIN_INTERVAL_S == 5.0


def test_beat_is_a_no_op_without_a_path_or_env(tmp_path, monkeypatch):
    monkeypatch.delenv(hb.ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    hb.beat()
    hb.beat(force=True)
    monkeypatch.setenv(hb.ENV, "")  # an empty value is no path either
    hb.beat()
    assert list(tmp_path.iterdir()) == []
    assert hb.age_s(None) is None


def test_beat_uses_the_env_path_and_creates_parents(tmp_path, monkeypatch):
    p = tmp_path / "state" / "hb" / "full-p01"
    monkeypatch.setenv(hb.ENV, str(p))
    hb.beat()
    assert p.is_file() and p.stat().st_size == 0  # empty: readers use the mtime
    assert 0 <= hb.age_s(p) < 60
    assert 0 <= hb.age_s(None) < 60  # age_s(None) reads the env path too


def test_rate_limit_one_mtime_change_within_the_interval_unless_forced(tmp_path):
    p = tmp_path / "hb" / "item"
    hb.beat(p)
    assert p.is_file()
    _age_old(p)
    hb.beat(p)  # within MIN_INTERVAL_S of the first: skipped
    hb.beat(str(p))  # the same path as a str: the same limit
    assert _mtime(p) == OLD
    hb.beat(p, force=True)
    assert _mtime(p) > OLD + 1


def test_rate_limit_is_per_path(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    hb.beat(a)
    hb.beat(b)  # another path is not limited by a's beat
    assert a.is_file() and b.is_file()


def test_rate_limit_passes_after_the_interval(tmp_path, monkeypatch):
    p = tmp_path / "x"
    clock = [100.0]
    monkeypatch.setattr(hb.time, "monotonic", lambda: clock[0])
    hb.beat(p)
    _age_old(p)
    clock[0] += hb.MIN_INTERVAL_S - 0.1
    hb.beat(p)
    assert _mtime(p) == OLD
    clock[0] += 0.2
    hb.beat(p)
    assert _mtime(p) > OLD + 1


def test_beating_without_a_path_starts_no_thread(monkeypatch):
    monkeypatch.delenv(hb.ENV, raising=False)
    before = threading.active_count()
    with hb.beating(every_s=0.01):
        assert threading.active_count() == before
        assert not [t for t in threading.enumerate() if t.name == "kitsune-heartbeat"]


def test_beating_beats_at_entry_and_while_the_block_runs(tmp_path):
    p = tmp_path / "hb" / "stores-ctc"
    with hb.beating(p, every_s=0.05):
        assert p.is_file()  # the entry beat is synchronous
        _age_old(p)
        assert _wait_for(lambda: _mtime(p) > OLD + 1), "no beat from the thread"
    assert not [t for t in threading.enumerate() if t.name == "kitsune-heartbeat" and t.is_alive()]


def test_beating_stops_after_max_s_while_the_block_goes_on(tmp_path):
    p = tmp_path / "hb" / "hung"
    t0 = time.monotonic()
    with hb.beating(p, every_s=0.05, max_s=0.6):
        _age_old(p)
        assert _wait_for(lambda: _mtime(p) > OLD + 1, timeout=0.5), "no beat before max_s"
        time.sleep(max(0.0, t0 + 1.5 - time.monotonic()))  # well past max_s
        _age_old(p)
        time.sleep(0.5)  # 10 more intervals: a live thread would have beaten
        assert _mtime(p) == OLD, "beating went on after max_s: a hung call would never go stale"


def test_beating_exit_is_bounded(tmp_path):
    p = tmp_path / "x"
    t = time.monotonic()
    with hb.beating(p, every_s=30.0):  # the thread sleeps 30 s; the exit must not wait for it
        pass
    assert time.monotonic() - t < 3.0


def test_beating_uses_the_env_path(tmp_path, monkeypatch):
    p = tmp_path / "env-hb"
    monkeypatch.setenv(hb.ENV, str(p))
    with hb.beating(every_s=10):
        assert p.is_file()


def test_beating_passes_exceptions_through(tmp_path):
    with pytest.raises(KeyError):
        with hb.beating(tmp_path / "x", every_s=0.05):
            raise KeyError("boom")


class Feat:
    """A featuriser stand-in: callable, with attributes the evaluators read."""
    sample_rate = 16000

    def __init__(self):
        self.calls = []

    def __call__(self, wave, lengths=None):
        self.calls.append((wave, lengths))
        return ("feats", wave)

    def to(self, device):
        return f"moved to {device}"


def test_Beating_beats_once_per_call_and_delegates(tmp_path, monkeypatch):
    p = tmp_path / "hb" / "m4-full-p01"
    beats = []
    real = hb.beat
    monkeypatch.setattr(hb, "beat", lambda path=None, **kw: (beats.append((path, kw)), real(path, **kw)))
    f = Feat()
    w = hb.Beating(f, p)
    assert w(1, lengths=2) == ("feats", 1) and w(3) == ("feats", 3)
    assert f.calls == [(1, 2), (3, None)]
    assert beats == [(p, {}), (p, {})]  # one (rate-limited) beat per call, never forced
    assert p.is_file()
    assert w.sample_rate == 16000 and w.to("cuda") == "moved to cuda" and w.calls is f.calls
    with pytest.raises(AttributeError):
        w.no_such_attribute


def test_Beating_without_a_path_is_transparent(tmp_path, monkeypatch):
    monkeypatch.delenv(hb.ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    f = Feat()
    assert hb.Beating(f)(7) == ("feats", 7)
    assert list(tmp_path.iterdir()) == []


def test_Beating_reads_the_env_at_call_time(tmp_path, monkeypatch):
    p = tmp_path / "late"
    monkeypatch.delenv(hb.ENV, raising=False)
    w = hb.Beating(Feat())
    monkeypatch.setenv(hb.ENV, str(p))
    w(0)
    assert p.is_file()


def test_Beating_survives_copy(tmp_path):
    import copy
    w = copy.copy(hb.Beating(Feat(), tmp_path / "c"))
    assert w(1) == ("feats", 1)


def test_an_unwritable_path_never_raises(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    bad = blocker / "sub" / "hb"  # a file where a directory must be
    hb.beat(bad)
    hb.beat(bad, force=True)
    with hb.beating(bad, every_s=0.01, max_s=0.1):
        time.sleep(0.05)
    assert hb.Beating(Feat(), bad)(1) == ("feats", 1)
    assert hb.age_s(bad) is None
    hb.beat(123)  # not a path at all
    with hb.beating(object()):
        pass
    monkeypatch.setenv(hb.ENV, str(bad))
    hb.beat()
    assert not bad.exists()


def test_age_s(tmp_path):
    p = tmp_path / "a"
    assert hb.age_s(p) is None
    p.touch()
    os.utime(p, (time.time() - 100, time.time() - 100))
    assert 99 <= hb.age_s(p) < 130


def test_stdlib_only_at_import():
    code = ("import sys; import kitsune.heartbeat; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('numpy', 'torch', 'pandas', 'pyarrow')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
