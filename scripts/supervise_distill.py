"""Supervisor for 04_distill.py on the laptop (scripts/run_overfit_tests.cmd): relaunch it with --resume after a crash.

Why: this laptop's GPU intermittently faults under load (`CUDA error: an illegal instruction was encountered`,
`unspecified launch failure`; see scripts/run_teacher_pass.py, which relaunched the teacher pass 11 times in ~104 min).
A CUDA fault poisons the process and 04_distill.py exits 1, but its full states (ckpt.full_after_smoke, then every
ckpt.full_local_every_min / full_every_steps) are exact resume points.

Policy, per attempt:
  exit 0, or 3 (ThroughputTooLow: a slow machine does not get faster)     -> done, return that exit code
  any other exit, the run this supervisor started has a full state          -> 04 --resume <that run dir> (its newest)
  any other exit, no full state yet                                         -> start again (a new run dir; the dead one
                                                                               stays for inspection)
An attempt makes progress when the run's newest full state is newer after it than before. After BLOCKING_AFTER
attempts in a row without progress the next ones run with CUDA_LAUNCH_BLOCKING=1 (serialised launches: slower, but no
diagnostic run of the teacher pass ever faulted that way) until one makes progress. The supervisor gives up after
--max-stalls attempts in a row without progress (a deterministic failure that needs a human) or --max-attempts without
progress in total, and returns the last exit code (1 if it does not fit an exit code). Attempts that made progress are
not counted: each moves the run's newest full state forward, so the run stays bounded, and a fault can cost a fast
attempt or two before the blocking one that progresses (a total cap ended long runs that were still progressing).

Every attempt's start and end go to stdout and, appended, to --log.

Usage: python scripts/supervise_distill.py --config configs/overfit_1s.json [--log overfit_pipeline.log]
       [--set KEY=VALUE ...] [--max-attempts 20] [--max-stalls 5] [--script scripts/04_distill.py]
"""
import argparse
import glob
import importlib.util
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts" / "04_distill.py"
EXIT_OK, EXIT_THROUGHPUT = 0, 3
FULL_RE = re.compile(r"^full_step_(\d+)$")
BLOCKING_AFTER = 2


def say(msg: str, log: str | None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [supervise] {msg}"
    print(line, flush=True)
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_trainer():
    """scripts/04_distill.py as a module, for its config resolution (run_name, runs_root; defaults and --set)."""
    spec = importlib.util.spec_from_file_location("kitsune_distill", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_dirs(runs_root: Path, run_name: str) -> set[Path]:
    return {p for p in runs_root.glob(f"{glob.escape(run_name)}-*") if p.is_dir()} if runs_root.is_dir() else set()


def newest_full(run: Path | None) -> int | None:
    """Step of the run's newest complete full state (checkpoints/full_step_<N>/trainer.pt), None if it has none."""
    ck = run / "checkpoints" if run is not None else None
    if ck is None or not ck.is_dir():
        return None
    steps = [int(m[1]) for p in ck.iterdir() if (m := FULL_RE.match(p.name)) and (p / "trainer.pt").is_file()]
    return max(steps) if steps else None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="04_distill.py --config of the fresh start")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="passed on to every attempt")
    ap.add_argument("--log", default=None, help="append the attempt lines to this file too")
    ap.add_argument("--max-attempts", type=int, default=20, help="attempts without a newer full state, in total")
    ap.add_argument("--max-stalls", type=int, default=5, help="attempts in a row without a newer full state")
    ap.add_argument("--script", default=str(TRAINER), help="the trainer (tests substitute a fake one)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    trainer = load_trainer()
    cfg = trainer.load_config(args.config, args.set)
    runs_root, name = trainer.rpath(cfg["runs_root"]), cfg["run_name"]
    run, rc, stalled, idle = None, None, 0, 0
    for attempt in itertools.count(1):
        before = newest_full(run)
        blocking = stalled >= BLOCKING_AFTER
        start = ["--resume", str(run)] if before is not None else ["--config", args.config]
        cmd = [sys.executable, args.script, *start]
        for s in args.set:
            cmd += ["--set", s]
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1") if blocking else None
        existing = run_dirs(runs_root, name)
        say(f"{name} attempt {attempt}: " + (f"resume {run} from full_step_{before}" if before is not None else "fresh")
            + (" with CUDA_LAUNCH_BLOCKING=1" if blocking else ""), args.log)
        try:
            rc = subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            say(f"{name} attempt {attempt}: interrupted", args.log)
            return 130
        if before is None:  # this attempt's run dir: the new one (names end with a UTC stamp)
            new = sorted(run_dirs(runs_root, name) - existing, key=lambda p: p.name)
            run = new[-1] if new else run
        after = newest_full(run)
        progressed = after is not None and (before is None or after > before)
        stalled = 0 if progressed else stalled + 1
        idle += not progressed
        say(f"{name} attempt {attempt} ended (exit {rc}); run {run}; newest full state {before} -> {after}", args.log)
        if rc in (EXIT_OK, EXIT_THROUGHPUT):
            return rc
        if stalled >= args.max_stalls:
            say(f"{name}: giving up after {stalled} attempts in a row without a newer full state", args.log)
            break
        if idle >= args.max_attempts:
            say(f"{name}: giving up after {idle} attempts without a newer full state", args.log)
            break
    return rc if rc is not None and 0 <= rc < 256 else 1


if __name__ == "__main__":
    sys.exit(main())
