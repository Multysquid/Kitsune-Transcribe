"""Supervisor for 02_teacher_pass.py: relaunch it when the process dies.

Why: on this laptop the GPU intermittently throws `CUDA error: an illegal instruction was encountered` (~once per
few hundred batches, independent of allocator / cuDNN / attention kernel / TF32 / threading - verified). A CUDA
fault poisons the process, but the teacher pass is atomic per shard and resumes from the next unfinished shard,
so a restart costs ~20 s. This wrapper restarts until the pass exits 0, and gives up if three consecutive attempts
make no progress (a deterministic failure that needs a human).

Usage: python scripts/run_teacher_pass.py [any 02_teacher_pass.py arguments]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "02_teacher_pass.py"


def n_done(out_root: Path) -> int:
    return sum(1 for _ in out_root.rglob("*.npz")) if out_root.exists() else 0


def main():
    args = sys.argv[1:]
    out_root = ROOT / "teacher_out"
    if "--out" in args:
        out_root = Path(args[args.index("--out") + 1])
    elif "--limit-rows" in args:
        out_root = ROOT / "teacher_out_smoke"

    attempt, stalled, t0 = 0, 0, time.time()
    while True:
        attempt += 1
        before = n_done(out_root)
        print(f"=== attempt {attempt}: {before} shards done, {(time.time() - t0) / 60:.0f} min elapsed ===", flush=True)
        rc = subprocess.run([sys.executable, str(SCRIPT), *args]).returncode
        if rc == 0:
            print(f"=== finished after {attempt} attempt(s), {(time.time() - t0) / 3600:.2f} h ===", flush=True)
            return 0
        after = n_done(out_root)
        stalled = stalled + 1 if after == before else 0
        print(f"=== attempt {attempt} died (exit {rc}); progress {before} -> {after} shards; "
              f"{'no progress ' + str(stalled) + 'x' if stalled else 'restarting'} ===", flush=True)
        if stalled >= 3:
            print("=== giving up: three attempts without progress ===", flush=True)
            return rc
        time.sleep(10)  # let the driver settle before re-creating the context


if __name__ == "__main__":
    sys.exit(main())
