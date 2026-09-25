"""The label box's lifecycle scripts: vast/onstart.sh hands KITSUNE_JOB=label to vast/label.py, and vast/watchdog.sh
stops a label box whose controller died (the orphan rule).

CPU only, no network: the scripts (or the functions and the subshell cut out of them) run under Git Bash with
PY/KITSUNE_PY=bash and fake python files that only record that they ran; skipped without bash.
"""
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VAST = ROOT / "vast"


def find_bash() -> str | None:  # as test_infra.find_bash
    import shutil
    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if cand and Path(cand).exists() and "system32" not in cand.lower():  # System32\bash.exe is WSL, not Git Bash
            return cand
    return None


def need_bash() -> str:
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    return bash


def test_lifecycle_scripts_keep_lf_endings_and_ascii():
    for name in ("onstart.sh", "watchdog.sh"):
        raw = (VAST / name).read_bytes()
        assert b"\r\n" not in raw, f"{name}: bash on the box chokes on CRLF"
        raw.decode("ascii")
    assert len((VAST / "onstart.sh").read_bytes()) < 16 * 1024


# ---------------------------------------------------------------------------------------------------------- onstart

def subshell_body() -> str:
    text = (VAST / "onstart.sh").read_text(encoding="utf-8")
    return re.search(r"^\(\n(.*?)^\) < /dev/null &$", text, re.M | re.S).group(1)


def test_label_dispatch_sits_after_the_lock_and_before_the_train_path():
    """label.py must start only under the supervise.lock check (a second onstart, or --rearm's check, sees it), and the
    train body (supervise.json shortcut, then bootstrap.sh) must stay as it was when KITSUNE_JOB is unset."""
    body = subshell_body()
    lock = body.index("supervise.lock")
    job = body.index('"${KITSUNE_JOB:-train}" = label')
    assert lock < job < body.index('-s "$KITSUNE_STATE/supervise.json"') < body.index(
        'bash "$KITSUNE_DIR/vast/bootstrap.sh"')
    dispatch = body[job:body.index("\n    fi\n", job)]
    assert 'exec "$PY" "$KITSUNE_DIR/vast/label.py"' in dispatch and "exec 7>&-" in dispatch
    assert "bootstrap.sh" not in dispatch and "supervise.py" not in dispatch


@pytest.mark.parametrize("job,history,expect", [
    ("label", False, ["label.py"]),
    ("label", True, ["label.py"]),  # a supervise.json (train history) does not divert a label box
    (None, False, ["bootstrap.sh", "supervise.py"]),
    ("train", True, ["supervise.py"]),
], ids=["label", "label-with-train-history", "unset", "train-restart"])
def test_onstart_subshell_dispatches_on_kitsune_job(tmp_path, job, history, expect):
    """With KITSUNE_JOB=label the detached subshell execs vast/label.py and runs neither bootstrap.sh nor supervise.py;
    without it (or =train) the train path is unchanged."""
    bash = need_bash()
    state, repo = tmp_path / "state", tmp_path / "repo"
    (repo / "vast").mkdir(parents=True)
    state.mkdir()
    for name in ("bootstrap.sh", "supervise.py", "label.py"):  # run by bash (PY=bash below): each notes that it ran
        (repo / "vast" / name).write_text(f'echo {name} >> "$KITSUNE_STATE/ran"\n', encoding="utf-8", newline="\n")
    if history:
        (state / "supervise.json").write_text('{"attempts": [{"t0": 1.0, "resume": null}]}', encoding="utf-8")
    script = tmp_path / "subshell.sh"
    script.write_text("\n".join(["set -euo pipefail", "log() { printf '%s\\n' \"$*\"; }", "PY=bash", subshell_body()]),
                      encoding="utf-8", newline="\n")
    env = dict(os.environ, KITSUNE_STATE=state.as_posix(), KITSUNE_DIR=repo.as_posix())
    env.pop("KITSUNE_JOB", None)
    if job is not None:
        env["KITSUNE_JOB"] = job
    r = subprocess.run([bash, str(script)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (state / "ran").read_text(encoding="utf-8").split() == expect
    assert ("handing over to vast/label.py" in r.stdout) == (job == "label"), r.stdout


def test_rearm_moves_the_label_state_aside(tmp_path):
    """A stopped label box continued with `onstart.sh --rearm` must re-plan: label.json (its recorded steps and final)
    and label_hb (else the watchdog would arm on the old heartbeat) go to rearm-<stamp>/ with the rest."""
    text = (VAST / "onstart.sh").read_text(encoding="utf-8")
    body = re.search(r"^rearm\(\) \{[^\n]*\n.*?^\}\n", text, re.M | re.S).group(0)
    assert "label.json" in body and "label_hb" in body
    bash = need_bash()
    state = tmp_path / "state"
    state.mkdir()
    for name in ("halt", "deadline", "label.json", "label_hb", "ledger_keep"):
        (state / name).write_text("x", encoding="utf-8")
    script = tmp_path / "rearm.sh"
    script.write_text("\n".join(["set -euo pipefail", "log() { printf '%s\\n' \"$*\"; }",
                                 f'KITSUNE_STATE="{state.as_posix()}"', body, "rearm", ""]),
                      encoding="utf-8", newline="\n")
    r = subprocess.run([bash, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    moved = [d for d in state.iterdir() if d.name.startswith("rearm-")]
    assert len(moved) == 1
    assert sorted(p.name for p in moved[0].iterdir()) == ["deadline", "halt", "label.json", "label_hb"]
    assert (state / "ledger_keep").exists(), "only the listed lifecycle files move"


def test_boot_log_names_the_job():
    text = (VAST / "onstart.sh").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith('log "boot $boots'))
    assert "job ${KITSUNE_JOB:-train}" in line


# --------------------------------------------------------------------------------------------------------- watchdog

# sleep: the poll interval (1 s here) sleeps for real; the retry sleep after a stop (300 s) ends the watchdog instead,
# and so does a poll count cap (a watchdog that should never fire must not run forever)
FAKE_SLEEP = ('() { _fake_sleeps=$(( ${_fake_sleeps:-0} + 1 )); '
              'if [ "$1" -ge 300 ] || [ "$_fake_sleeps" -gt "${FAKE_SLEEP_MAX:-60}" ]; then exit 0; fi; '
              'command sleep "$1"; }')


def watchdog_env(tmp_path: Path, orphan_s: str | None, **kw) -> tuple[dict, Path, Path]:
    state, repo = tmp_path / "state", tmp_path / "repo"
    (repo / "vast").mkdir(parents=True, exist_ok=True)
    state.mkdir(exist_ok=True)
    record = tmp_path / "finish_calls"
    # run by bash (KITSUNE_PY=bash): records the wall second and the args of every call
    (repo / "vast" / "finish.py").write_text(f'echo "$(date +%s) $*" >> "{record.as_posix()}"\n', encoding="utf-8",
                                             newline="\n")
    env = dict(os.environ, KITSUNE_STATE=state.as_posix(), KITSUNE_DIR=repo.as_posix(), KITSUNE_PY="bash",
               KITSUNE_MAX_HOURS="5.5", KITSUNE_WATCHDOG_POLL_S="1", **kw)
    env.pop("KITSUNE_WATCHDOG_ORPHAN_S", None)
    env.pop("KITSUNE_NO_SELF_STOP", None)
    if orphan_s is not None:
        env["KITSUNE_WATCHDOG_ORPHAN_S"] = orphan_s
    env["BASH_FUNC_sleep%%"] = FAKE_SLEEP
    return env, state, record


def calls(record: Path) -> list[tuple[int, str]]:
    if not record.exists():
        return []
    out = []
    for ln in record.read_text(encoding="utf-8").splitlines():
        t, _, args = ln.partition(" ")
        out.append((int(t), args))
    return out


def test_watchdog_dry_run_prints_the_orphan_rule(tmp_path):
    bash = need_bash()
    for orphan_s, shown in (("900", True), ("0", False), (None, False)):
        env, state, _ = watchdog_env(tmp_path, orphan_s)
        r = subprocess.run([bash, str(VAST / "watchdog.sh"), "--dry-run"], capture_output=True, text=True, env=env,
                           timeout=60)
        assert r.returncode == 0, r.stderr
        assert ("orphan rule" in r.stdout) == shown, r.stdout
        if shown:
            assert "stale by 900 s" in r.stdout and "1800 s" in r.stdout and "finish.py --sync-only" in r.stdout
        assert not (state / "deadline").exists(), "a dry run must not fix the deadline"


def test_watchdog_syncs_then_stops_when_the_armed_heartbeat_goes_stale(tmp_path):
    """label.py touched label_hb during this boot and then died: after ORPHAN_S the watchdog uploads what it can
    (finish.py --sync-only, the job comes from KITSUNE_JOB) and stops the box, not waiting for the 30 h cap."""
    bash = need_bash()
    env, state, record = watchdog_env(tmp_path, "2")
    t0 = int(time.time())
    hb = state / "label_hb"
    hb.write_text("", encoding="utf-8")
    os.utime(hb, (t0 + 1, t0 + 1))  # touched after the watchdog's start: armed at once
    r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    got = calls(record)
    assert [a for _, a in got] == ["--sync-only", "--stop --no-sync --reason watchdog: label controller dead"], got
    assert "heartbeat stale" in r.stdout and "never touched" not in r.stdout, r.stdout
    assert got[0][0] - (t0 + 1) > 2, "not before the heartbeat is ORPHAN_S old"


def test_watchdog_ignores_a_previous_boots_heartbeat_until_twice_the_limit(tmp_path):
    """A label_hb left by the previous boot is older than this watchdog: that alone must not stop a restarted box whose
    controller is still starting up; only 2 x ORPHAN_S without a fresh touch does."""
    bash = need_bash()
    env, state, record = watchdog_env(tmp_path, "2")
    t0 = int(time.time())
    hb = state / "label_hb"
    hb.write_text("", encoding="utf-8")
    os.utime(hb, (t0 - 1000, t0 - 1000))
    r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    got = calls(record)
    assert [a for _, a in got] == ["--sync-only", "--stop --no-sync --reason watchdog: label controller dead"], got
    assert "never touched" in r.stdout, r.stdout
    assert got[0][0] - t0 >= 2 * 2 + 1, "a stale old heartbeat does not fire before 2 x ORPHAN_S"


@pytest.mark.parametrize("orphan_s,halt", [("0", False), (None, False), ("1", True)],
                         ids=["orphan-0", "unset", "halted"])
def test_watchdog_orphan_rule_stays_off(tmp_path, orphan_s, halt):
    """ORPHAN_S=0 (the train job) never looks at label_hb, and a halt marker (finish.py has taken over the end) turns
    the rule off: a stale heartbeat for 5 polls triggers nothing."""
    bash = need_bash()
    env, state, record = watchdog_env(tmp_path, orphan_s, FAKE_SLEEP_MAX="5")
    hb = state / "label_hb"
    hb.write_text("", encoding="utf-8")
    t0 = int(time.time())
    os.utime(hb, (t0 - 1000, t0 - 1000))
    if halt:
        (state / "halt").write_text('{"action": "stop"}', encoding="utf-8")
    r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls(record) == [], r.stdout
    assert "controller" not in r.stdout, r.stdout


def test_watchdog_uses_kitsune_py():
    """onstart exports KITSUNE_PY (the venv python, or python3 on an image without it); the watchdog used a fixed path."""
    text = (VAST / "watchdog.sh").read_text(encoding="utf-8")
    assert 'PY="${KITSUNE_PY:-/venv/main/bin/python}"' in text
    assert 'ORPHAN_S="${KITSUNE_WATCHDOG_ORPHAN_S:-0}"' in text
