"""Run a full-data box's queue on the vast box and decide what happens to the instance when it ends.

vast/onstart.sh execs this on a full box (KITSUNE_JOB=full, vast/launch.py --job full) once bootstrap has passed. It
runs `python -m kitsune.full_queue run --box $KITSUNE_BOX` (kitsune/full_queue.py) and decides with decide_queue:
  exit 0                       -> vast/finish.py --destroy  (it verifies the HF uploads first, and stops if it can't)
  exit 4 (a training or store item failed for good) or 3 (a trainer's throughput floor)
                               -> vast/finish.py --stop     (the disk is kept for a human to inspect)
  any other exit, or a container restart during the queue
                               -> the queue again, up to MAX_QUEUE_RESTARTS times, then --stop
The queue resumes its runs itself (finished items skipped, runs resumed from their full states): the supervisor
restarts only the queue. Any other non-zero exit first forces a log sync (finish.py --sync-only): a hard crash
(segfault, OOM kill, CUDA fault) skips the queue's own syncs. Each attempt records the OOM kills the container's cgroup
counted while it ran (oom_kills): the rc alone does not tell one that took a DataLoader worker from any other error.

The attempt history lives in $KITSUNE_STATE/supervise.json, so if the container restarts in the middle of an attempt
(host reboot) the interrupted attempt counts as a failure and the same policy applies; once a final decision is
recorded the supervisor never starts the queue again. The file is written with fsyncs (save_state), and one that does
not read as a history anyway is moved aside and ends in a recorded stop, as the history it held (a final decision, the
restarts already used) is unknown. finish.py writes its halt marker ($KITSUNE_STATE/halt) only when it acts, after its
sync and --destroy's verification, so a restart in between (a host reboot during the minutes-long upload) leaves a
recorded decision and no marker; onstart.sh then boots the supervisor again, which runs that finish once more (it is
idempotent: the Hub skips what it already has, then verify and act) instead of idling until the watchdog's deadline,
which only stops the box. An exclusive lock ($KITSUNE_STATE/supervise.lock, flock) keeps a second supervisor
(vast/onstart.sh run again by hand during a healthy run) from treating the live attempt as interrupted. onstart's
detached subshell holds the same lock while bootstrap runs and closes it right before it execs the supervisor, so the
supervisor waits for it (wait_for_lock, up to LOCK_WAIT_S) before it steps aside: a process left over from bootstrap
that still held it (a train_hb toucher's `sleep 60`, box p01-chain on 2026-10-01) made the supervisor exit at once, and
the box idled with no supervisor.

The final finish.py call is bounded too (FINISH_TIMEOUT_S): its sync and --destroy's verification talk to the Hub (the
repo listing has no timeout, and a crawling link has none overall), and a hang there would keep the GPU billing until
the watchdog's deadline. When it times out or fails, `finish.py --stop --no-sync` follows (itself bounded: no Hub sync,
the infra upload is capped); a timed-out destroy thus ends as a stop, which keeps the disk. vast/watchdog.sh stays the
backstop for a supervisor that hangs anyway. The box watchdog reads the controllers' heartbeat $KITSUNE_STATE/train_hb:
the queue beats it while it runs (the supervisor does not), and the supervisor beats it, bounded
(kitsune.heartbeat.beating, max_s = the call's timeout + 60 s), around its own finish.py calls - the --sync-only after a
queue failure and the final --stop/--destroy - so a slow but live upload is not taken for a dead controller, while a
hung one still goes stale. vast/label.py, the label box's controller, uses load_state, save_state, acquire_lock and
final_finish of this module.

A chain box (KITSUNE_BOX=p01-chain; contract addendum E) runs its controller through the same command. Its exit 5 with
its chain summary in $KITSUNE_STATE/queue_summary.json (kind chain, status gate_failed or failed: it ended before box 1
trained) destroys - finish.py --destroy verifies the chain's summaries, verdicts and infra on the Hub first - with no
--sync-only before it; any other exit 5 is an ordinary failure. Each attempt records the chain stage its summary names,
and the restart budget (MAX_QUEUE_RESTARTS) counts the failures of the last attempt's stage only, so the restarts of
stage 1 never use up box 1's; every other box's attempts have stage None and share one budget, as before.

Usage (started by vast/onstart.sh; KITSUNE_JOB, KITSUNE_BOX and KITSUNE_OUT_REPO come from the instance env):
  python vast/supervise.py
  KITSUNE_JOB=full KITSUNE_BOX=p01 python vast/supervise.py --dry-run \
      --queue-cmd "python -c 'import sys; sys.exit(4)'"                     # exercise the policy locally
"""
import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # kitsune.heartbeat (a full box's bounded beats around finish.py)
    sys.path.insert(1, str(ROOT))

EXIT_OK, EXIT_THROUGHPUT, EXIT_HALT = 0, 3, 4  # EXIT_HALT: the queue's halt (an item failed for good)
# a chain box's controller (kitsune/full_queue.py ChainController): it ended before its last stage trained - a failed
# gate, a failed stage-2 bootstrap, box 1 no longer fits - and nothing unique is on the disk: destroy (finish.py
# verifies the chain's Hub records first). Only with its chain summary; any other exit 5 is an ordinary failure
EXIT_CHAIN_DESTROY = 5
CHAIN_NAMES = ("p01-chain",)  # kitsune.fullrun.CHAIN_NAMES (this script stays stdlib-only at import)
CHAIN_DESTROY_STATUSES = ("gate_failed", "failed")
MAX_QUEUE_RESTARTS = 2  # the queue: restarted after a crash at most this often, then the box stops
# (a chain box: per stage, so a restart spent in stage 1 never takes one from box 1's training)
FINISH_BEAT_EVERY_S = 30.0  # train_hb beats while a finish.py call runs (kitsune.heartbeat.beating)
SYNC_TIMEOUT_S = 1800
# the final --stop/--destroy; --destroy may still upload the ~9 GB full state and hashes every expected file to verify
FINISH_TIMEOUT_S = {"stop": 1800, "destroy": 3600}
FALLBACK_TIMEOUT_S = 900  # finish.py --stop --no-sync: infra upload capped at 180 s, then vast REST (3 tries) and CLI
# supervise.lock: a lock still held when the supervisor starts is polled every LOCK_POLL_S for up to LOCK_WAIT_S before
# it steps aside (wait_for_lock). Another live supervisor only delays its refusal by that long
LOCK_WAIT_S = 180.0
LOCK_POLL_S = 2.0
CGROUP = Path("/sys/fs/cgroup")


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [supervise] {msg}", flush=True)


def is_chain_summary(summary) -> bool:
    return isinstance(summary, dict) and summary.get("kind") == "chain" and summary.get("box") in CHAIN_NAMES


def decide_queue(rc: int | None, n_failures: int, reason: str | None = None,
                 summary: dict | None = None) -> tuple[str, str]:
    """The queue's policy -> (action, reason), action in {"destroy", "stop", "restart"}. rc None = the queue was
    interrupted by a container restart; reason = its queue_summary.json's (a halt says why). A chain box's exit 5 with
    its chain summary (kind chain, a CHAIN_NAMES box, status gate_failed or failed) destroys: it ended before its last
    stage trained, and finish.py verifies its records on the Hub before the destroy."""
    if rc == EXIT_CHAIN_DESTROY and is_chain_summary(summary) and summary.get("status") in CHAIN_DESTROY_STATUSES:
        return "destroy", f"chain ended before box 1 trained: {reason or summary.get('reason') or 'see its summary'}"
    if rc == EXIT_OK:
        return "destroy", "queue finished (exit 0)"
    if rc == EXIT_HALT:
        return "stop", f"queue halted (exit 4): {reason or 'see queue_summary.json'}"
    if rc == EXIT_THROUGHPUT:
        return "stop", f"queue: throughput below the floor (exit 3): {reason or ''}".rstrip(": ")
    what = "interrupted by a container restart" if rc is None else f"exit {rc}"
    if n_failures > MAX_QUEUE_RESTARTS:
        return "stop", f"queue failed {n_failures} times ({what}): {reason or 'see queue_summary.json'}"
    return "restart", f"queue failed ({what}); restarting it (it resumes from its state)"


def queue_reason(state_dir: Path) -> str | None:
    try:
        return json.loads((state_dir / "queue_summary.json").read_text(encoding="utf-8")).get("reason")
    except (OSError, ValueError, AttributeError):
        return None


def queue_summary(state_dir: Path) -> dict | None:
    """$KITSUNE_STATE/queue_summary.json (the queue's, or a chain's controller summary), or None."""
    try:
        doc = json.loads((state_dir / "queue_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def attempt_stage(state_dir: Path):
    """The chain stage the controller was in (its summary's stage), or None (not a chain: every attempt shares the one
    restart budget, as before)."""
    summ = queue_summary(state_dir)
    return summ.get("stage") if is_chain_summary(summ) else None


def beating(hb: Path | None, timeout: float):
    """The bounded controller beat around one finish.py call (kitsune.heartbeat.beating, max_s = timeout + 60 s); a
    plain context without hb."""
    if hb is None:
        return contextlib.nullcontext()
    from kitsune import heartbeat

    return heartbeat.beating(hb, every_s=FINISH_BEAT_EVERY_S, max_s=float(timeout) + 60)


def supervise_queue(queue_cmd: list[str], state_path: Path, dry_run: bool = False, *, hb: Path | None = None) -> int:
    """Run the queue, restart it after a failure (decide_queue), then the final finish.py call; the history (attempts
    with rc, the final decision) and its crash handling as the module docstring says. hb: the controllers' heartbeat
    file the box's watchdog reads, beaten around the finish.py calls."""
    lock = wait_for_lock(state_path.with_suffix(".lock"))
    if lock is None:
        log(f"another supervisor holds {state_path.with_suffix('.lock')}; not starting a second one")
        return 0
    state = load_state(state_path)
    dry = ["--dry-run"] if dry_run else []
    aside = state.pop("corrupt", None)
    if aside:
        reason = f"supervise.json unreadable (moved to {aside}); attempt history unknown"
        state["final"] = {"action": "stop", "reason": reason, "wall": time.time(), "corrupt_state": aside}
        save_state(state_path, state)
        log(f"decision: stop ({reason})")
        final_finish("stop", reason, dry, hb=hb)
        return 1
    final = state.get("final")
    if final:
        log(f"a final decision is already recorded ({final}); not starting the queue again")
        if not state_path.with_name("halt").exists():
            log("no halt marker: the container restarted before finish acted on that decision; running it again")
            final_finish(final["action"], final.get("reason", ""), dry, hb=hb)
        return 0
    for a in state["attempts"]:
        if "rc" not in a:  # the container died while the queue ran
            a.update(rc=None, interrupted=True, stage=attempt_stage(state_path.parent))
    while True:
        if state["attempts"]:
            last = state["attempts"][-1]
            # a chain's restart budget is per stage (the attempt's stage from the chain summary; None elsewhere)
            failures = sum(1 for a in state["attempts"] if a["rc"] != EXIT_OK and a.get("stage") == last.get("stage"))
            action, reason = decide_queue(last["rc"], failures, queue_reason(state_path.parent),
                                          queue_summary(state_path.parent))
            log(f"decision after queue attempt {len(state['attempts'])}: {action} ({reason})")
            if action != "restart":
                state["final"] = {"action": action, "reason": reason, "wall": time.time()}
                save_state(state_path, state)
                final_finish(action, reason, dry, hb=hb)
                return last["rc"] if last["rc"] is not None else 1
        attempt = {"t0": time.time(), "queue": True}
        state["attempts"].append(attempt)
        save_state(state_path, state)
        env = dict(os.environ, KITSUNE_ATTEMPT=str(len(state["attempts"])))
        oom0 = oom_kills()
        rc = run_queue(queue_cmd, env)
        oom1 = oom_kills()
        attempt.update(rc=rc, t1=time.time(), stage=attempt_stage(state_path.parent),
                       oom_kills=oom1 - oom0 if oom0 is not None and oom1 is not None else None)
        save_state(state_path, state)
        log(f"queue attempt {len(state['attempts'])} exited {rc} after {(attempt['t1'] - attempt['t0']) / 60:.1f} min")
        # the stop and destroy paths sync everything anyway (a chain's exit 5: its destroy)
        if rc not in (EXIT_OK, EXIT_HALT, EXIT_THROUGHPUT, EXIT_CHAIN_DESTROY):
            with beating(hb, SYNC_TIMEOUT_S):
                call_finish(["--sync-only", *dry], timeout=SYNC_TIMEOUT_S)


def oom_kills(cg: Path | None = None) -> int | None:
    """The container's OOM-kill counter (cgroup v2 memory.events, else v1 memory.oom_control); None without one. The
    rc names an OOM kill only when it hit the process itself (-9): one that took a DataLoader worker ends a trainer
    with a plain error, and the host-wide RAM numbers in its logs do not show the container's own limit."""
    cg = cg or CGROUP
    for rel in ("memory.events", "memory/memory.oom_control"):
        try:
            for line in (cg / rel).read_text().splitlines():
                k, _, v = line.partition(" ")
                if k == "oom_kill":
                    return int(v)
        except (OSError, ValueError):
            continue
    return None


def run_queue(argv: list[str], env: dict) -> int:
    log(f"run: {shlex.join(argv)}")
    return subprocess.run(argv, cwd=ROOT, env=env).returncode


def call_finish(args: list[str], timeout: float | None = None) -> int:
    argv = [sys.executable, str(ROOT / "vast" / "finish.py"), *args]
    log(f"finish: {shlex.join(argv)}")
    try:
        return subprocess.run(argv, cwd=ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        log(f"finish timed out after {timeout} s")
        return 124


def final_finish(action: str, reason: str, dry: list[str], hb: Path | None = None):
    """The final finish.py --stop/--destroy, bounded; `finish.py --stop --no-sync` after a timeout or failure. hb: the
    box's controller heartbeat, beaten (bounded) while each call runs (vast/label.py passes none)."""
    with beating(hb, FINISH_TIMEOUT_S[action]):
        rc = call_finish([f"--{action}", "--reason", reason, *dry], timeout=FINISH_TIMEOUT_S[action])
    log(f"finish exited {rc}")
    if rc not in (0, 2):  # 2: --destroy's verification failed and the instance was stopped
        why = "timed out" if rc == 124 else f"exited {rc}"
        with beating(hb, FALLBACK_TIMEOUT_S):
            rc = call_finish(["--stop", "--no-sync", "--reason", f"finish --{action} {why} ({reason})", *dry],
                             timeout=FALLBACK_TIMEOUT_S)
        log(f"fallback stop exited {rc}")


def load_state(path: Path, required: str = "attempts") -> dict:
    """The attempt history (none yet: a fresh one). A file that is not one - empty or NUL-filled after an unclean host
    crash, torn bytes, a disk fault, a bad hand edit - is moved aside to <name>.corrupt-<unix time> and flagged
    ("corrupt": where it went): the history it held is unknown (a final decision? the restarts already used?), so
    supervise_queue() stops the box instead of starting the queue afresh. `required` is the key every valid state has:
    "attempts" (a list) for this supervisor, "phase" for vast/label.py (whose fresh state is {"phase": None,
    "final": None})."""
    fresh = {"attempts": [], "final": None} if required == "attempts" else {required: None, "final": None}
    if not path.exists():
        return fresh
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if required == "attempts":
            ok = isinstance(state, dict) and isinstance(state.get("attempts"), list)
            why = None if ok else "not an attempt history"
        else:
            why = None if isinstance(state, dict) and required in state else f"no {required!r} key"
    except ValueError as e:  # JSONDecodeError, and UnicodeDecodeError for bytes that are not UTF-8
        why = f"{type(e).__name__}: {e}"
    if why is None:
        return state
    aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    path.replace(aside)
    log(f"{path} is unreadable ({why[:200]}); moved to {aside}")
    return {**fresh, "corrupt": str(aside)}


def _sync_dir(p: Path):
    """fsync a directory, so a rename into it is on disk (POSIX; as scripts/04_distill.py _sync_dir). Windows cannot
    open a directory for that (PermissionError); NTFS journals the rename itself."""
    if os.name == "nt":
        return
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_state(path: Path, state: dict):
    """Written to a .tmp file, fsynced, renamed over supervise.json, the directory fsynced: an unclean host crash
    right after a save leaves the old or the new history, never the empty or NUL-filled file a rename-over without
    them can leave (kitsune/store.py fsync_path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(state, indent=1))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _sync_dir(path.parent)


def acquire_lock(path: Path):
    """Exclusive, non-blocking flock held for the supervisor's lifetime (by the returned file object; the queue does
    not inherit it). None if another process holds it. Without fcntl (Windows) there is nothing to guard."""
    try:
        import fcntl
    except ImportError:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def wait_for_lock(path: Path, wait_s: float | None = None, poll_s: float | None = None, *, sleep=time.sleep,
                  clock=time.monotonic):
    """acquire_lock, polled every poll_s (LOCK_POLL_S) for up to wait_s (LOCK_WAIT_S) while another process holds the
    lock; the wait is logged once. A short-lived holder (a process of the bootstrap that handed over, which inherited
    onstart's lock fd) must never stop the box; a second live supervisor still holds it after the wait, and this one
    then steps aside as before. None if it is still held."""
    wait_s = LOCK_WAIT_S if wait_s is None else wait_s
    poll_s = LOCK_POLL_S if poll_s is None else poll_s
    lock = acquire_lock(path)
    if lock is not None:
        return lock
    t0 = clock()
    log(f"{path} is held by another process; waiting up to {wait_s:.0f} s for it (polling every {poll_s:.0f} s)")
    while lock is None and clock() - t0 < wait_s:
        sleep(poll_s)
        lock = acquire_lock(path)
    if lock is not None:
        log(f"got {path} after {clock() - t0:.0f} s")
    return lock


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-repo", default=os.environ.get("KITSUNE_OUT_REPO") or None)
    ap.add_argument("--state", default=str(Path(os.environ.get("KITSUNE_STATE", "/workspace/kitsune_state")) / "supervise.json"))
    ap.add_argument("--queue-cmd", default=None,
                    help="override the queue command (default: python -m kitsune.full_queue run --box $KITSUNE_BOX)")
    ap.add_argument("--dry-run", action="store_true", help="run the queue, but finish.py only prints (no stop/destroy)")
    args = ap.parse_args(argv)

    if os.environ.get("KITSUNE_JOB") != "full":
        log(f"KITSUNE_JOB is {os.environ.get('KITSUNE_JOB')!r}: this supervisor runs a full-data box's queue "
            f"(KITSUNE_JOB=full)")
        return 2
    box = os.environ.get("KITSUNE_BOX") or ""
    queue_cmd = shlex.split(args.queue_cmd) if args.queue_cmd else [sys.executable, "-m", "kitsune.full_queue", "run",
                                                                     "--box", box]
    hb = Path(args.state).parent / "train_hb"  # kitsune.fullrun.TRAIN_HB in $KITSUNE_STATE
    log(f"full queue: box={box} out_repo={args.out_repo} state={args.state} dry_run={args.dry_run}")
    return supervise_queue(queue_cmd, Path(args.state), args.dry_run, hb=hb)


if __name__ == "__main__":
    sys.exit(main())
