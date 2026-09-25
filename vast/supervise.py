"""Run the trainer on the vast box and decide what happens to the instance when it ends (D50a).

Policy (decide()):
  exit 0                                   -> vast/finish.py --destroy  (verifies the HF upload first, stops if it can't)
  exit 3 (ThroughputTooLow)                -> vast/finish.py --stop     (a slow host does not get faster)
  any other failure, the FIRST one, reached step >= 100 and a local full state exists
                                           -> resume once: 04 --resume <newest full state>
  a second failure, a failure before step 100, or no full state to resume from
                                           -> vast/finish.py --stop
Why: after the smoke phase (100 steps) the code path is proven on this host, so a later crash is most likely transient
(a host hiccup, a NaN spike) and one resume costs minutes. A failure before step 100 is a code or hardware problem that
a restart would only repeat on the meter, and a second failure means the same. Stop, not destroy, on every failure path:
the disk (checkpoints, logs) survives for a human to inspect.

Every non-zero exit first forces a log sync (finish.py --sync-only --no-full): a hard crash (segfault, OOM kill, CUDA
fault) skips the trainer's own final sync. The ~9 GB full state is left out there because a resume needs it only
locally and every stop path (finish.py --stop) uploads it anyway; the GPU idles while this sync runs. Each attempt
records the OOM kills the container's cgroup counted while it ran (oom_kills; "(oom_kill +N)" in its exit line): the
rc alone does not tell one that took a DataLoader worker from any other error. The attempt history lives in
$KITSUNE_STATE/supervise.json, so if the container restarts in the middle of an attempt (host reboot) the interrupted
attempt counts as a failure (its run dir is found again from its start time or its --resume path) and the same policy
applies; once a final decision is recorded the supervisor never starts another run. An exclusive lock
($KITSUNE_STATE/supervise.lock, flock) keeps a second supervisor (vast/onstart.sh run again by hand during a healthy
run) from treating the live attempt as interrupted.

The final finish.py call is bounded too (FINISH_TIMEOUT_S): its sync and --destroy's verification talk to the Hub (the
repo listing has no timeout, and a crawling link has none overall), and a hang there would keep the GPU billing until the
watchdog's deadline (up to 5.5 h after boot for an early failure). When it times out or fails, `finish.py --stop
--no-sync` follows (itself bounded: no Hub sync, the infra upload is capped); a timed-out destroy thus ends as a stop,
which keeps the disk. vast/watchdog.sh stays the backstop for a supervisor that hangs anyway.

Contract with scripts/04_distill.py: CLI `--config <json> [--set hf.output_repo=<repo>] [--resume <path>]`; exit codes
0 ok / 3 throughput too low / anything else failure; it writes runs/<run_id>/config.json at start,
runs/<run_id>/metrics/scalars.jsonl rows {"step": ...}, and full states as runs/<run_id>/checkpoints/full_step_<N>[.ext].

Usage (started by vast/onstart.sh; KITSUNE_CONFIG / KITSUNE_OUT_REPO come from the instance env):
  python vast/supervise.py
  python vast/supervise.py --dry-run --train-cmd "python -c 'import sys; sys.exit(3)'"   # exercise the policy locally
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finish import FULL_RE, newest_checkpoint  # noqa: E402

EXIT_OK, EXIT_THROUGHPUT = 0, 3
MIN_RESUME_STEP = 100
MAX_FAILURES = 2
SYNC_TIMEOUT_S = 1800
# the final --stop/--destroy; --destroy may still upload the ~9 GB full state and hashes every expected file to verify
FINISH_TIMEOUT_S = {"stop": 1800, "destroy": 3600}
FALLBACK_TIMEOUT_S = 900  # finish.py --stop --no-sync: infra upload capped at 180 s, then vast REST (3 tries) and CLI
TAIL_BYTES = 256 << 10
CGROUP = Path("/sys/fs/cgroup")


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [supervise] {msg}", flush=True)


def decide(rc: int | None, step: int, n_failures: int, full_state: Path | None) -> tuple[str, str]:
    """-> (action, reason), action in {"destroy", "stop", "resume"}. rc None = attempt interrupted by a restart."""
    if rc == EXIT_OK:
        return "destroy", "trainer finished (exit 0)"
    if rc == EXIT_THROUGHPUT:
        return "stop", "throughput below the floor (exit 3)"
    what = "interrupted by a container restart" if rc is None else f"exit {rc}"
    if n_failures >= MAX_FAILURES:
        return "stop", f"second failure ({what}) at step {step}"
    if step < MIN_RESUME_STEP:
        return "stop", f"failed before step {MIN_RESUME_STEP} ({what} at step {step})"
    if full_state is None:
        return "stop", f"failed at step {step} ({what}) with no local full state to resume from"
    return "resume", f"first failure ({what}) at step {step}; resuming from {full_state.name}"


def find_run_dir(runs_root: Path, since: float) -> Path | None:
    """The run dir the trainer created or touched since `since` (newest config.json wins), else None. Never an older
    one: after --rearm the previous run's dirs (and full states) stay under runs/, and a fresh attempt that died
    before writing its config.json would otherwise be credited with the old run's step and resumed from its state."""
    if not runs_root.is_dir():
        return None
    cands = [d for d in runs_root.iterdir() if d.is_dir() and (d / "config.json").exists()]
    cands = [d for d in cands if max((d / "config.json").stat().st_mtime, d.stat().st_mtime) >= since - 5]
    return max(cands, key=lambda d: (d / "config.json").stat().st_mtime) if cands else None


def tail_lines(path: Path, nbytes: int = TAIL_BYTES) -> list[str]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - nbytes))
        return f.read().decode("utf-8", "replace").splitlines()


def last_step(run_dir: Path | None) -> int:
    """Highest optimizer step logged: tail of metrics/scalars.jsonl, else events.jsonl, else 0."""
    if run_dir is None:
        return 0
    for rel in ("metrics/scalars.jsonl", "events.jsonl"):
        steps = []
        for line in tail_lines(run_dir / rel):
            try:
                s = json.loads(line).get("step")
            except (json.JSONDecodeError, AttributeError):
                continue  # the first line of a tail is usually cut; a torn last line is possible after a crash
            if isinstance(s, (int, float)):
                steps.append(int(s))
        if steps:
            return max(steps)
    return 0


def latest_full_state(run_dir: Path | None) -> Path | None:
    return newest_checkpoint(run_dir / "checkpoints", FULL_RE) if run_dir else None


def train_argv(train_cmd: list[str], config: str, out_repo: str | None, resume: Path | None) -> list[str]:
    argv = [*train_cmd, "--config", config]
    if out_repo:
        argv += ["--set", f"hf.output_repo={out_repo}"]
    if resume is not None:
        argv += ["--resume", str(resume)]
    return argv


def oom_kills(cg: Path | None = None) -> int | None:
    """The container's OOM-kill counter (cgroup v2 memory.events, else v1 memory.oom_control); None without one. The
    rc names an OOM kill only when it hit the trainer itself (-9): one that took a DataLoader worker ends the trainer
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


def run_trainer(argv: list[str], env: dict) -> int:
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


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"{path} is corrupt; starting a fresh history")
    return {"attempts": [], "final": None}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)


def acquire_lock(path: Path):
    """Exclusive, non-blocking flock held for the supervisor's lifetime (by the returned file object; the trainer
    does not inherit it). None if another process holds it. Without fcntl (Windows) there is nothing to guard."""
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


def attempt_run_dir(a: dict, runs_root: Path) -> Path | None:
    """The run dir of a recorded attempt: stored when it ended; for one interrupted by a restart, the --resume path's
    run dir, else the run dir created or touched since the attempt started."""
    if a.get("run_dir"):
        return Path(a["run_dir"])
    if a.get("resume"):
        return Path(a["resume"]).parent.parent  # runs/<id>/checkpoints/full_step_<N>
    return find_run_dir(runs_root, a["t0"])


def supervise(config: str, out_repo: str | None, runs_root: Path, train_cmd: list[str], state_path: Path,
              dry_run: bool = False) -> int:
    lock = acquire_lock(state_path.with_suffix(".lock"))  # held until this function returns
    if lock is None:
        log(f"another supervisor holds {state_path.with_suffix('.lock')}; not starting a second one")
        return 0
    state = load_state(state_path)
    if state.get("final"):
        log(f"a final decision is already recorded ({state['final']}); not starting another run")
        return 0
    for a in state["attempts"]:
        if "rc" not in a:  # the container died while this attempt was running
            rd = attempt_run_dir(a, runs_root)
            a.update(rc=None, interrupted=True, run_dir=str(rd) if rd else None, step=last_step(rd))
    dry = ["--dry-run"] if dry_run else []

    while True:
        resume = None
        if state["attempts"]:
            last = state["attempts"][-1]
            run_dir = Path(last["run_dir"]) if last.get("run_dir") else None
            failures = sum(1 for a in state["attempts"] if a["rc"] != EXIT_OK)
            full = latest_full_state(run_dir) if last["rc"] != EXIT_OK else None
            action, reason = decide(last["rc"], last["step"], failures, full)
            log(f"decision after attempt {len(state['attempts'])}: {action} ({reason})")
            if action != "resume":
                state["final"] = {"action": action, "reason": reason, "wall": time.time()}
                save_state(state_path, state)
                rc = call_finish([f"--{action}", "--reason", reason, *dry], timeout=FINISH_TIMEOUT_S[action])
                log(f"finish exited {rc}")
                if rc not in (0, 2):  # 2: --destroy's verification failed and the instance was stopped
                    why = "timed out" if rc == 124 else f"exited {rc}"
                    rc = call_finish(["--stop", "--no-sync", "--reason", f"finish --{action} {why} ({reason})", *dry],
                                     timeout=FALLBACK_TIMEOUT_S)
                    log(f"fallback stop exited {rc}")
                return last["rc"] if last["rc"] is not None else 1
            resume = full

        attempt = {"t0": time.time(), "resume": str(resume) if resume else None}
        state["attempts"].append(attempt)
        save_state(state_path, state)
        env = dict(os.environ, KITSUNE_ATTEMPT=str(len(state["attempts"])))
        oom0 = oom_kills()
        rc = run_trainer(train_argv(train_cmd, config, out_repo, resume), env)
        oom1 = oom_kills()
        run_dir = attempt_run_dir(attempt, runs_root)
        attempt.update(rc=rc, t1=time.time(), run_dir=str(run_dir) if run_dir else None, step=last_step(run_dir),
                       oom_kills=oom1 - oom0 if oom0 is not None and oom1 is not None else None)
        save_state(state_path, state)
        oom = f" (oom_kill +{attempt['oom_kills']})" if attempt["oom_kills"] else ""
        log(f"attempt {len(state['attempts'])} exited {rc}{oom} at step {attempt['step']} after "
            f"{(attempt['t1'] - attempt['t0']) / 60:.1f} min (run dir {run_dir})")
        if rc != EXIT_OK:
            call_finish(["--sync-only", "--no-full", *dry], timeout=SYNC_TIMEOUT_S)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.environ.get("KITSUNE_CONFIG", "configs/viability.json"))
    ap.add_argument("--out-repo", default=os.environ.get("KITSUNE_OUT_REPO") or None)
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--state", default=str(Path(os.environ.get("KITSUNE_STATE", "/workspace/kitsune_state")) / "supervise.json"))
    ap.add_argument("--train-cmd", default=None, help="override the trainer command (default: python scripts/04_distill.py)")
    ap.add_argument("--dry-run", action="store_true", help="run the trainer, but finish.py only prints (no stop/destroy)")
    args = ap.parse_args(argv)

    train_cmd = shlex.split(args.train_cmd) if args.train_cmd else [sys.executable, str(ROOT / "scripts" / "04_distill.py")]
    log(f"config={args.config} out_repo={args.out_repo} state={args.state} dry_run={args.dry_run}")
    return supervise(args.config, args.out_repo, Path(args.runs_root), train_cmd, Path(args.state), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
