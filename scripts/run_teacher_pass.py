"""Supervisor for 02_teacher_pass.py: relaunch it when the process dies.

Why: on this laptop the GPU intermittently throws `CUDA error: an illegal instruction was encountered` (~once per
few hundred batches, independent of allocator / cuDNN / attention kernel / TF32 / threading - verified). The
faults cluster on some shards' long-utterance batches, but every diagnostic run with CUDA_LAUNCH_BLOCKING=1
survived, so it looks timing/power dependent. A CUDA fault poisons the process, but the teacher pass is atomic per
shard and resumes from the next unfinished shard, so a restart costs ~20 s. This wrapper restarts until the pass
exits 0; after two consecutive attempts without a finished shard it escalates to CUDA_LAUNCH_BLOCKING=1
(serialised launches, ~1.8x slower) until progress resumes, and gives up only after three blocking-mode attempts
in a row make no progress (that is a deterministic failure that needs a human).

Usage: python scripts/run_teacher_pass.py [supervisor options] [any wrapped-script arguments]
Supervisor options: --script <path> (default scripts/02_teacher_pass.py), --watch-dir <dir> and
--done-suffix <ext> override where finished-shard files are counted (progress detection).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sup = argparse.ArgumentParser(add_help=False)
    sup.add_argument("--script", default=str(ROOT / "scripts" / "02_teacher_pass.py"))
    sup.add_argument("--watch-dir", default=None)
    sup.add_argument("--done-suffix", default=".npz")
    opts, args = sup.parse_known_args()
    script = Path(opts.script)

    if opts.watch_dir:
        out_root = Path(opts.watch_dir)
    elif "--out" in args:
        out_root = Path(args[args.index("--out") + 1])
    elif "--limit-rows" in args:
        out_root = ROOT / "teacher_out_smoke"
    else:
        out_root = ROOT / "teacher_out"

    def n_done(root: Path = out_root) -> int:
        return sum(1 for f in root.rglob(f"*{opts.done_suffix}")) if root.exists() else 0

    attempt, stalled, stalled_blocking, t0 = 0, 0, 0, time.time()
    while True:
        attempt += 1
        blocking = stalled >= 2
        before = n_done()
        print(f"=== attempt {attempt}: {before} shards done, {(time.time() - t0) / 60:.0f} min elapsed"
              f"{', CUDA_LAUNCH_BLOCKING=1' if blocking else ''} ===", flush=True)
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1") if blocking else None
        # a blocking attempt only clears the stuck shard (shards complete in manifest order), then we go back to fast mode
        extra = ["--limit-shards", str(before + 1)] if blocking else []
        rc = subprocess.run([sys.executable, str(script), *args, *extra], env=env).returncode
        if rc == 0 and not blocking:
            print(f"=== finished after {attempt} attempt(s), {(time.time() - t0) / 3600:.2f} h ===", flush=True)
            return 0
        after = n_done()
        if after > before:
            stalled, stalled_blocking = 0, 0
        else:
            stalled += 1
            stalled_blocking = stalled_blocking + 1 if blocking else 0
        print(f"=== attempt {attempt} ended (exit {rc}); progress {before} -> {after} shards; "
              f"stalled {stalled}x (blocking {stalled_blocking}x) ===", flush=True)
        if stalled_blocking >= 3:
            print("=== giving up: three blocking-mode attempts without progress ===", flush=True)
            return rc
        time.sleep(10)  # let the driver settle before re-creating the context


if __name__ == "__main__":
    sys.exit(main())
