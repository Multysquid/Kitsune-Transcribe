"""vast/supervise.py for a full-data runs box (KITSUNE_JOB=full; build contract section 5): the queue command
(`python -m kitsune.full_queue run --box $KITSUNE_BOX`), the unchanged decide_queue, and the controllers' heartbeat
$KITSUNE_STATE/train_hb beaten (bounded) while the supervisor's own finish.py calls run - and only then, never for the
study box. CPU only; the queue and finish.py are scripted stand-ins."""
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vast"))
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

supervise = importlib.import_module("supervise")


def test_supervise_main_runs_the_full_queue_for_a_full_box(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setenv("KITSUNE_JOB", "full")
    monkeypatch.setenv("KITSUNE_BOX", "p01")
    monkeypatch.setattr(supervise, "supervise_queue",
                        lambda cmd, state, dry, hb=None: seen.update(cmd=cmd, state=state, hb=hb) or 0)
    assert supervise.main(["--state", str(tmp_path / "state" / "supervise.json")]) == 0
    assert seen["cmd"][1:] == ["-m", "kitsune.full_queue", "run", "--box", "p01"] and seen["cmd"][0] == sys.executable
    assert seen["hb"] == tmp_path / "state" / "train_hb"
    # the study box: the study queue, no heartbeat (its call stays the three positional arguments)
    monkeypatch.setenv("KITSUNE_JOB", "study")
    monkeypatch.setattr(supervise, "supervise_queue", lambda cmd, state, dry: seen.update(cmd=cmd, hb="none") or 0)
    assert supervise.main(["--state", str(tmp_path / "s.json")]) == 0
    assert seen["cmd"][1:] == ["-m", "kitsune.study_queue", "run", "--box", "p01"] and seen["hb"] == "none"


@pytest.mark.parametrize("rc, action", [(0, "destroy"), (4, "stop"), (3, "stop"), (1, "restart")])
def test_a_full_queues_exits_take_the_queue_policy(rc, action):
    assert supervise.decide_queue(rc, 1)[0] == action


def run_full_supervisor(tmp_path, monkeypatch, rcs, hb=True, finish_s=0.6):
    """supervise_queue for a full box with the queue's exits scripted and every finish.py call slow; the train_hb
    mtimes are sampled while each finish call runs."""
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    hb_file = state / "train_hb"
    calls, finishes, samples = [], [], []

    def queue(argv, env):
        calls.append(list(argv))
        return rcs.pop(0)

    def slow_finish(args, timeout=None):
        finishes.append(list(args))
        t_end = time.time() + finish_s
        seen = []
        while time.time() < t_end:
            seen.append(hb_file.stat().st_mtime if hb_file.exists() else None)
            time.sleep(0.02)
        samples.append((args[0], seen))
        return 0

    monkeypatch.setattr(supervise, "FINISH_BEAT_EVERY_S", 0.05)
    monkeypatch.setattr(supervise, "run_trainer", queue)
    monkeypatch.setattr(supervise, "call_finish", slow_finish)
    rc = supervise.supervise_queue(["python", "-m", "kitsune.full_queue", "run", "--box", "p01"],
                                   state / "supervise.json", hb=hb_file if hb else None)
    return rc, calls, finishes, samples, hb_file


def test_train_hb_advances_while_a_slow_finish_runs(tmp_path, monkeypatch):
    rc, calls, finishes, samples, hb = run_full_supervisor(tmp_path, monkeypatch, [1, 0])
    assert rc == 0 and [f[0] for f in finishes] == ["--sync-only", "--destroy"]
    for what, seen in samples:  # beaten at the call's start and on through it
        assert seen[0] is not None and len(set(seen)) >= 3, what
    # after the call the beats stop: the supervisor never beats while the queue runs
    m = hb.stat().st_mtime
    time.sleep(0.2)
    assert hb.stat().st_mtime == m


def test_the_study_box_supervisor_writes_no_train_hb(tmp_path, monkeypatch):
    rc, calls, finishes, samples, hb = run_full_supervisor(tmp_path, monkeypatch, [0], hb=False, finish_s=0.1)
    assert rc == 0 and not hb.exists() and all(set(seen) == {None} for _, seen in samples)


def test_the_finish_beat_is_bounded_by_the_calls_timeout(tmp_path, monkeypatch):
    from kitsune import heartbeat

    got = []
    real = heartbeat.beating

    def spy(path=None, every_s=30.0, max_s=None):
        got.append((Path(path).name, every_s, max_s))
        return real(path, every_s=every_s, max_s=max_s)

    monkeypatch.setattr(heartbeat, "beating", spy)
    run_full_supervisor(tmp_path, monkeypatch, [1, 4], finish_s=0.0)
    assert got == [("train_hb", 0.05, supervise.SYNC_TIMEOUT_S + 60),
                   ("train_hb", 0.05, supervise.FINISH_TIMEOUT_S["stop"] + 60)]
    assert json.loads((tmp_path / "state" / "supervise.json").read_text())["final"]["action"] == "stop"
