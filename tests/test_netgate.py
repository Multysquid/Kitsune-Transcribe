"""kitsune.netgate, the full box's download gate (fix 1): time three pinned upstream files before the paid rebuild.

A fake download (it writes the sample's bytes into the cache dir and advances an injected clock at a chosen rate)
stands in for hf_hub_download: pass and slow verdicts and their exit codes 0/3, a download error or a file of the wrong
size exits 1, a slow host is cut at the cutoff without waiting for its samples, every timed file is deleted, the
timeouts never go below launch's 40 MB/s sizing or 30 min, --timeouts prints them, and the samples are the files 01
downloads at 01's pinned revisions. CPU only, no network.
"""
import importlib.util
import json
import math
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kitsune import extent, netgate  # noqa: E402


class Clock:
    """A monotonic clock the fake download advances: seconds = bytes / rate."""

    def __init__(self):
        self.t, self.lock = 1000.0, threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t

    def advance(self, s: float):
        with self.lock:
            self.t += s


def fake_download(clock: Clock, rate: float, *, size_of=None, fail=None, hang_on=None, calls=None):
    """hf_hub_download's stand-in: writes the file into cache_dir/<repo>/<filename> and advances the clock by its
    bytes / rate. size_of(sample) -> the bytes written (default: the pinned size, as a sparse-ish file of that
    length); fail: a filename that raises; hang_on: a filename that never returns (a stalled link)."""
    by_name = {s.filename: s for s in netgate.SAMPLES}

    def download(repo, filename, *, revision, cache_dir):
        if calls is not None:
            calls.append((repo, filename, revision, cache_dir))
        if filename == fail:
            raise ConnectionError("Server disconnected")
        s = by_name[filename]
        path = Path(cache_dir) / repo.replace("/", "--") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        n = size_of(s) if size_of else s.bytes
        with open(path, "wb") as f:
            f.truncate(n)  # the length without writing ~1 GB
        if filename == hang_on:
            clock.advance(1e6)  # whatever the cutoff, it has passed; the thread then never returns
            threading.Event().wait()
        clock.advance(n / rate)
        return str(path)

    return download


GATE_ENV = {"KITSUNE_GATE_BYTES": str(int(571.2e9)), "KITSUNE_GATE_MAX_H": "5",
            "KITSUNE_REBUILD_BYTES": str(int(571.2e9)), "KITSUNE_PULL_BYTES": str(int(25e9)),
            "KITSUNE_MACHINE_ID": "54650", "KITSUNE_CPU_QUOTA": "61"}


@pytest.fixture
def gate_env(monkeypatch):
    for k, v in GATE_ENV.items():
        monkeypatch.setenv(k, v)


def run_gate(tmp_path, rate, **kw):
    clock = Clock()
    out, dest = tmp_path / "state" / "download_gate.json", tmp_path / "netgate"
    rc = netgate.main(["--out", str(out), "--dir", str(dest)], download=fake_download(clock, rate, **kw), clock=clock)
    return rc, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None), dest


def test_the_minimum_rate_is_the_reference_extent_in_five_hours():
    assert netgate.GATE_REF_GB == 571.2 and netgate.DEFAULT_MAX_H == 5.0 and netgate.CUTOFF_FACTOR == 2.0
    assert netgate.min_rate(571.2e9, 5.0) == pytest.approx(31.73e6, rel=1e-3)
    assert sum(s.bytes for s in netgate.SAMPLES) == pytest.approx(2.48e9, rel=0.01)


def test_the_samples_are_the_files_01_downloads_at_its_pinned_revisions():
    spec = importlib.util.spec_from_file_location("prepare_data_01_gate", ROOT / "scripts" / "01_prepare_data.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    assert len(netgate.SAMPLES) == 3 and len({s.repo for s in netgate.SAMPLES}) == 3
    for s in netgate.SAMPLES:
        assert prep.REVISIONS[s.repo] == s.revision, s.repo
    repos = {s.repo for s in netgate.SAMPLES}
    assert {prep.GALGAME_REPO, prep.EMOLIA_REPO, "japanese-asr/whisper_transcriptions.reazonspeech.large"} == repos


def test_a_fast_host_passes_and_the_timeouts_follow_its_rate(tmp_path, gate_env, capsys):
    calls = []
    rc, rec, dest = run_gate(tmp_path, 100e6, calls=calls)
    assert rc == 0 and rec["verdict"] == "pass" and rec["format"] == 1, rec
    assert [c[:3] for c in calls] == [(s.repo, s.filename, s.revision) for s in netgate.SAMPLES]  # one at a time
    assert rec["rate_bytes_s"] == pytest.approx(100e6, rel=1e-3)
    assert rec["min_rate_bytes_s"] == pytest.approx(31.73e6, rel=1e-3)
    assert rec["projected_h"] == pytest.approx(571.2e9 / 100e6 / 3600, rel=1e-3)
    assert [s["filename"] for s in rec["samples"]] == [s.filename for s in netgate.SAMPLES]
    assert all(s["mb_s"] == pytest.approx(100, rel=1e-2) for s in rec["samples"])
    assert rec["machine_id"] == "54650" and rec["cpu_quota"] == "61" and rec["gate_max_h"] == 5.0
    assert set(rec) >= {"verdict", "reason", "samples", "rate_bytes_s", "min_rate_bytes_s", "gate_bytes", "gate_max_h",
                        "projected_h", "rebuild_bytes", "rebuild_timeout_min", "pull_bytes", "pull_timeout_min",
                        "machine_id", "cpu_quota", "nproc", "wall", "time_utc"}
    # at 100 MB/s the rebuild would take less than launch's 40 MB/s sizing: that floor holds (01 is CPU-bound there)
    assert (rec["pull_timeout_min"], rec["rebuild_timeout_min"]) == netgate.timeouts(100e6, 571.2e9, 25e9)
    assert rec["rebuild_timeout_min"] == math.ceil(extent.REBUILD_BASE_MIN + extent.REBUILD_SLACK * 571.2e9
                                                   / extent.REBUILD_BYTES_PER_S / 60)
    assert not dest.exists(), "the gate's cache dir is removed"
    assert "netgate: pass" in capsys.readouterr().out


def test_every_timed_file_is_deleted_before_the_next_download(tmp_path, gate_env):
    seen = []

    def watching(clock):
        inner = fake_download(clock, 100e6)

        def download(repo, filename, *, revision, cache_dir):
            seen.append(sorted(p.name for p in Path(cache_dir).rglob("*") if p.is_file()))
            return inner(repo, filename, revision=revision, cache_dir=cache_dir)

        return download

    clock = Clock()
    out, dest = tmp_path / "g.json", tmp_path / "d"
    assert netgate.main(["--out", str(out), "--dir", str(dest)], download=watching(clock), clock=clock) == 0
    assert seen == [[], [], []]


def test_a_slow_host_is_refused_with_exit_3(tmp_path, gate_env):
    rc, rec, _ = run_gate(tmp_path, 20e6)  # complete, but below 31.7 MB/s: 571 GB would take ~7.9 h
    assert rc == netgate.EXIT_SLOW == 3 and rec["verdict"] == "slow"
    assert rec["projected_h"] == pytest.approx(571.2e9 / 20e6 / 3600, rel=1e-3) and "> 5 h" in rec["reason"]
    assert rec["rebuild_timeout_min"] is not None  # recorded, never used


def test_a_very_slow_host_is_cut_at_the_cutoff(tmp_path, gate_env):
    """Study box A #1's 2.9 MB/s: the samples would take ~14 min; the gate stops at CUTOFF_FACTOR x their time at the
    minimum rate (~157 s) and says slow, with the bytes that had arrived."""
    cutoff = netgate.CUTOFF_FACTOR * sum(s.bytes for s in netgate.SAMPLES) / netgate.min_rate(571.2e9, 5)
    assert 150 < cutoff < 165
    rc, rec, _ = run_gate(tmp_path, 2.9e6)
    assert rc == 3 and rec["verdict"] == "slow" and "cut after" in rec["reason"], rec
    assert len(rec["samples"]) == 0  # the first sample (493 MB) alone needs ~170 s at 2.9 MB/s
    # a link that stalls outright: the download never returns, the gate still ends
    rc, rec, _ = run_gate(tmp_path / "stall", 100e6, hang_on=netgate.SAMPLES[1].filename)
    assert rc == 3 and rec["verdict"] == "slow" and len(rec["samples"]) == 1


def test_a_download_error_or_a_wrong_size_exits_1(tmp_path, gate_env):
    rc, rec, _ = run_gate(tmp_path, 100e6, fail=netgate.SAMPLES[2].filename)
    assert rc == 1 and rec["verdict"] == "error" and "Server disconnected" in rec["reason"]
    assert rec["pull_timeout_min"] is None and rec["rebuild_timeout_min"] is None
    rc, rec, _ = run_gate(tmp_path / "b", 100e6, size_of=lambda s: s.bytes - 1)
    assert rc == 1 and rec["verdict"] == "error" and "bytes on disk" in rec["reason"]


def test_bad_gate_env_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KITSUNE_GATE_MAX_H", "0")
    rc = netgate.main(["--out", str(tmp_path / "g.json"), "--dir", str(tmp_path / "d")],
                      download=fake_download(Clock(), 1e9), clock=Clock())
    assert rc == 1 and "KITSUNE_GATE" in capsys.readouterr().err


def test_timeouts_never_go_below_the_sizing_or_30_min():
    fast_pull, fast_rebuild = netgate.timeouts(1e9, 571.2e9, 25e9)
    assert fast_pull == 31  # 30 + 1.5 x 25 s
    assert fast_rebuild == math.ceil(30 + 1.5 * 571.2e9 / 40e6 / 60)
    slow_pull, slow_rebuild = netgate.timeouts(31.73e6, 571.2e9, 25e9)
    assert slow_rebuild > fast_rebuild and slow_pull > fast_pull
    assert slow_rebuild == math.ceil(30 + 1.5 * 571.2e9 / 31.73e6 / 60)
    assert netgate.timeouts(1e9, 0, 0) == (30, 30)


def test_the_timeouts_cli_prints_a_passed_gates_minutes(tmp_path, gate_env, capsys):
    rc, rec, _ = run_gate(tmp_path, 100e6)
    capsys.readouterr()
    assert netgate.main(["--timeouts", str(tmp_path / "state" / "download_gate.json")]) == 0
    assert capsys.readouterr().out.split() == [str(rec["pull_timeout_min"]), str(rec["rebuild_timeout_min"])]
    slow = tmp_path / "slow.json"
    slow.write_text(json.dumps(dict(rec, verdict="slow")), encoding="utf-8")
    assert netgate.main(["--timeouts", str(slow)]) == 1
    assert netgate.main(["--timeouts", str(tmp_path / "missing.json")]) == 1
