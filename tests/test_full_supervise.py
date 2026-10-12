"""vast/supervise.py for a full-data runs box (KITSUNE_JOB=full; build contract section 5): the queue command
(`python -m kitsune.full_queue run --box $KITSUNE_BOX`), the queue's policy (decide_queue: destroy, stop or restart,
the restart budget, a halt's reason, a restart during the queue), and the controllers' heartbeat $KITSUNE_STATE/train_hb
beaten (bounded) while the supervisor's own finish.py calls run - and only then. The history file, the lock and the
bounded final finish are tests/test_infra.py's. CPU only; the queue and finish.py are scripted stand-ins."""
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
                        lambda cmd, state, dry, hb=None: seen.update(cmd=cmd, state=state, dry=dry, hb=hb) or 0)
    assert supervise.main(["--state", str(tmp_path / "state" / "supervise.json")]) == 0
    assert seen["cmd"][1:] == ["-m", "kitsune.full_queue", "run", "--box", "p01"] and seen["cmd"][0] == sys.executable
    assert seen["hb"] == tmp_path / "state" / "train_hb" and seen["dry"] is False
    assert supervise.main(["--state", str(tmp_path / "s.json"), "--dry-run", "--queue-cmd", "python -c pass"]) == 0
    assert seen["cmd"] == ["python", "-c", "pass"] and seen["dry"] is True
    # any other job (the retired train and study jobs; the label box runs vast/label.py): no queue
    seen.clear()
    for job in ("study", "train", "label", None):
        if job is None:
            monkeypatch.delenv("KITSUNE_JOB")
        else:
            monkeypatch.setenv("KITSUNE_JOB", job)
        assert supervise.main(["--state", str(tmp_path / "s.json")]) == 2 and seen == {}


@pytest.mark.parametrize("rc, n_fail, action", [(0, 0, "destroy"), (4, 1, "stop"), (3, 1, "stop"), (1, 1, "restart"),
                                               (None, 2, "restart"), (1, 3, "stop")])
def test_decide_queue(rc, n_fail, action):
    assert supervise.decide_queue(rc, n_fail)[0] == action


def run_queue_supervisor(tmp_path, monkeypatch, rcs, summary_reason=None, state=None):
    """supervise_queue with the queue's exits scripted and finish.py recorded."""
    calls, finishes = [], []

    def queue(argv, env):
        calls.append(list(argv))
        return rcs.pop(0)

    monkeypatch.setattr(supervise, "run_queue", queue)
    monkeypatch.setattr(supervise, "call_finish", lambda args, timeout=None: finishes.append(list(args)) or 0)
    state_path = tmp_path / "state" / "supervise.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state is not None:
        state_path.write_text(json.dumps(state), encoding="utf-8")
    if summary_reason:
        (tmp_path / "state" / "queue_summary.json").write_text(json.dumps({"reason": summary_reason}))
    rc = supervise.supervise_queue(["python", "-m", "kitsune.full_queue", "run", "--box", "p01"], state_path)
    return rc, calls, finishes, json.loads(state_path.read_text(encoding="utf-8"))


def test_supervise_queue_destroys_a_finished_box(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(tmp_path, monkeypatch, [0])
    assert rc == 0 and len(calls) == 1 and [f[0] for f in finishes] == ["--destroy"]
    assert state["final"] == {"action": "destroy", "reason": "queue finished (exit 0)", "wall": state["final"]["wall"]}


def test_supervise_queue_restarts_a_crashed_queue_then_stops(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(tmp_path, monkeypatch, [1, 1, 1])
    assert rc == 1 and len(calls) == 1 + supervise.MAX_QUEUE_RESTARTS
    assert [f[0] for f in finishes] == ["--sync-only"] * 3 + ["--stop"]
    rc, calls, finishes, state = run_queue_supervisor(tmp_path / "b", monkeypatch, [1, 0])
    assert rc == 0 and [f[0] for f in finishes] == ["--sync-only", "--destroy"]


def test_supervise_queue_stops_on_a_halt_with_its_reason(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(tmp_path, monkeypatch, [4],
                                                      summary_reason="train item full-p01 failed for good")
    assert rc == 4 and finishes[-1][0] == "--stop" and "full-p01 failed for good" in state["final"]["reason"]


def test_supervise_queue_counts_an_interrupted_queue_and_never_reruns_a_final_box(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(
        tmp_path, monkeypatch, [0], state={"attempts": [{"t0": 1.0, "queue": True}], "final": None})
    assert rc == 0 and state["attempts"][0]["interrupted"] is True and len(calls) == 1
    (tmp_path / "state" / "halt").write_text("{}")
    rc, calls, finishes, _ = run_queue_supervisor(tmp_path, monkeypatch, [], state=state)
    assert calls == [] and finishes == []


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
    monkeypatch.setattr(supervise, "run_queue", queue)
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


def test_without_a_heartbeat_file_nothing_is_beaten(tmp_path, monkeypatch):
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
