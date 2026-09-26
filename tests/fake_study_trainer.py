"""A stand-in for every process the study box's queue starts (kitsune/study_queue.py), for its CPU dry runs.

It writes what the queue reads, in the trainer's formats, fast: runs/<run_name>-<stamp>/ with config.json
({"config": ...}), events.jsonl, metrics/scalars.jsonl (sched/train_s, time/data_wait_s, time/step_s with wall times),
checkpoints (step_<N>/ weights, full_step_<N>/ states) and summary.json (calibrate / lr_probe / branch blocks), and
exits as the trainer would. One JSON file per invocation goes into the dir $FAKE_LOG (t0, t1, argv,
CUDA_VISIBLE_DEVICES, run_name, rc), so a test can check the order, the concurrency and the GPU pins; a process killed
by the queue leaves none.

Modes (by argv[1]):
  train      scripts/04_distill.py: --config C / --resume RUN_DIR, --set k=v (JSON values, as the trainer parses them)
  stores     python -m kitsune.study_queue build-stores --config C
  anchor     python -m kitsune.study_queue anchor ... --out DIR
  speed      tools/speed_probe.py --model M --family F --out FILE
Behaviour knobs (env, JSON): FAKE_SETUP_S {run_name prefix: seconds before the first step}; FAKE_STEP_S (seconds per
step, default 0.002); FAKE_WAIT {run_name prefix: data-wait fraction; ".w12": the fraction with perf.num_workers 12};
FAKE_OBJ {probe run name: objective or "nan" (diverges: FloatingPointError)}; FAKE_CRASH {run_name: fraction} (the
first launch writes its states up to that fraction of max_steps, then exits 1); FAKE_MAIN_S (seconds a main run
takes after writing its fraction states, or {run_name prefix: seconds}); KITSUNE_CRASH_AT_STEP as the trainer reads it
(a crash before that step, full states every ckpt.full_every_steps); FAKE_CALIB_CRASH {run_name prefix: step} (a
calibration run's first try exits 1 at that step).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def env_json(name: str, default):
    v = os.environ.get(name)
    return json.loads(v) if v else default


def by_prefix(table: dict, name: str, default=None):
    best = None
    for k, v in table.items():
        if name.startswith(k) and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best else default


def main_s(name: str) -> float:
    v = env_json("FAKE_MAIN_S", 0.05)
    return float(by_prefix(v, name, 0.05) if isinstance(v, dict) else v)


def apply_set(cfg: dict, s: str):
    key, _, raw = s.partition("=")
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse(argv: list[str]) -> dict:
    out, sets, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--set":
            sets.append(argv[i + 1])
            i += 2
        elif a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[a[2:]] = argv[i + 1]
            i += 2
        else:
            out[a[2:]] = True
            i += 1
    out["sets"] = sets
    return out


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def ckpt(run: Path, name: str, full: bool = False):
    d = run / "checkpoints" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / ("model.pt" if full else "model.safetensors")).write_bytes(name.encode() * 8)
    if full:
        (d / "trainer.pt").write_bytes(b"state")


def event(run: Path, kind: str, **f):
    with open(run / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "wall": time.time(), **f}) + "\n")


def train(args: dict, record: dict) -> int:
    root = Path.cwd()
    if args.get("resume"):
        run = root / args["resume"]
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))["config"]
        for s in args["sets"]:
            apply_set(cfg, s)
        resumed = True
    else:
        cfg = json.loads((root / args["config"]).read_text(encoding="utf-8"))
        for s in args["sets"]:
            apply_set(cfg, s)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = root / "runs" / f"{cfg['run_name']}-{stamp}"
        n = 1
        while run.exists():
            run = root / "runs" / f"{cfg['run_name']}-{stamp}-{n}"
            n += 1
        run.mkdir(parents=True)
        write_json(run / "config.json", {"config": cfg, "argv": sys.argv})
        resumed = False
    name = cfg["run_name"]
    record.update(run_name=name, run_dir=run.name, resumed=resumed)
    event(run, "phase", name="setup")
    br = cfg.get("branch") or {}
    if br.get("parent"):
        parent = root / br["parent"]
        pcfg = json.loads((parent / "config.json").read_text(encoding="utf-8"))["config"]
        M = int(pcfg["schedule"]["max_steps"])
        rs, es = round(br["resume_frac"] * M), round(br["end_frac"] * M)
        if not (parent / "checkpoints" / f"full_step_{rs}").is_dir():
            write_json(run / "summary.json", {"status": "failed", "error": f"SystemExit: no full state {rs}"})
            return 2
        record["parent"] = parent.name
        time.sleep(float(os.environ.get("FAKE_BRANCH_S", "0.05")))
        ckpt(run, f"step_{es}")
        ckpt(run, f"full_step_{es}", full=True)
        write_json(run / "summary.json", {"status": "complete", "steps": es, "branch": {
            "parent_run_id": parent.name, "resume_step": rs, "end_step": es, "t_c": float(rs)}})
        return 0
    time.sleep(float(by_prefix(env_json("FAKE_SETUP_S", {}), name, 0.0)))
    M = int(cfg["schedule"]["max_steps"])
    if (cfg.get("calibrate") or {}).get("enabled"):
        return calibrate(run, cfg, name, M)
    crash_at = int(os.environ["KITSUNE_CRASH_AT_STEP"]) if os.environ.get("KITSUNE_CRASH_AT_STEP") else None
    every = (cfg.get("ckpt") or {}).get("full_every_steps")
    if (cfg.get("lr_probe") or {}).get("enabled"):
        obj = by_prefix(env_json("FAKE_OBJ", {}), name, 1.0)
        start = 0
        if resumed:
            fulls = sorted(int(p.name.rsplit("_", 1)[1]) for p in (run / "checkpoints").glob("full_step_*"))
            start = fulls[-1] if fulls else 0
            record["resumed_from"] = start
        for s in range(start + 1, M + 1):
            if crash_at is not None and s >= crash_at:
                write_json(run / "summary.json", {"status": "failed", "error": f"RuntimeError: crash at {s}"})
                return 1
            if every and s % int(every) == 0:
                ckpt(run, f"full_step_{s}", full=True)
        ckpt(run, f"full_step_{M}", full=True)
        if obj == "nan":
            write_json(run / "summary.json", {"status": "failed",
                                               "error": "FloatingPointError: 4 consecutive steps with a non-finite"})
            return 1
        write_json(run / "summary.json", {"status": "complete", "steps": M, "lr_probe": {
            "objective": float(obj), "lr": cfg["optim"]["lr"], "max_steps": M, "family": cfg.get("family", "aed")}})
        return 0
    # a main run: the fraction states and weights, the end ones; a first launch may crash (FAKE_CRASH)
    ck = cfg.get("ckpt") or {}
    fracs = ck.get("full_at_fracs") or []
    crash = by_prefix(env_json("FAKE_CRASH", {}), name) if not resumed else None
    marks = {round(f * M): f for f in fracs}
    for f in ck.get("weights_at_fracs") or []:
        ckpt(run, f"step_{round(f * M)}")
    for step in sorted(marks):
        if crash is not None and step > crash * M:
            break
        ckpt(run, f"full_step_{step}", full=True)
    if crash is not None:
        time.sleep(main_s(name))
        write_json(run / "summary.json", {"status": "failed", "error": "RuntimeError: fake crash"})
        return 1
    if resumed:
        record["resumed_from"] = max((int(p.name.rsplit("_", 1)[1]) for p in (run / "checkpoints").glob("full_step_*")),
                                     default=0)
    time.sleep(main_s(name))
    ckpt(run, f"step_{M}")
    ckpt(run, f"full_step_{M}", full=True)
    write_json(run / "summary.json", {"status": "complete", "steps": M})
    return 0


def calibrate(run: Path, cfg: dict, name: str, M: int) -> int:
    """Steps until the STOP file or max_steps: per step the loop clock, its data wait, the wall time."""
    step_s = float(os.environ.get("FAKE_STEP_S", "0.002"))
    workers = (cfg.get("perf") or {}).get("num_workers")
    waits = env_json("FAKE_WAIT", {})
    w12 = {k[: -len(".w12")]: v for k, v in waits.items() if k.endswith(".w12")}
    plain = {k: v for k, v in waits.items() if not k.endswith(".w12")}
    frac = by_prefix(w12 if workers == 12 else plain, name, 0.01)
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    t, s = 0.0, 0
    crash = by_prefix(env_json("FAKE_CALIB_CRASH", {}), name) if "-try" not in name else None
    with open(run / "metrics" / "scalars.jsonl", "a", encoding="utf-8") as f:
        for s in range(1, M + 1):
            if (run / "STOP").exists():
                break
            if crash is not None and s >= crash:
                write_json(run / "summary.json", {"status": "failed", "error": "RuntimeError: fake calibration crash"})
                return 1
            time.sleep(step_s)
            dt = 1.0 + 0.001 * (s % 7)
            t += dt
            wall = time.time()
            for tag, v in (("sched/train_s", t), ("time/data_wait_s", frac * dt), ("time/step_s", dt)):
                f.write(json.dumps({"step": s, "wall": wall, "tag": tag, "value": v}) + "\n")
            f.flush()
    micro = float(cfg["batch"]["micro_audio_s"])
    write_json(run / "summary.json", {"status": "complete", "steps": s, "calibrate": {
        "micro_audio_s": micro, "workers": workers if isinstance(workers, int) else 8}})
    return 0


def main() -> int:
    mode, args = sys.argv[1], parse(sys.argv[2:])
    record = dict(mode=mode, t0=time.time(), argv=sys.argv[1:], gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  item=os.environ.get("KITSUNE_QUEUE_ITEM"))
    rc = 0
    try:
        if mode == "train":
            rc = train(args, record)
        elif mode == "stores":
            time.sleep(float(os.environ.get("FAKE_STORES_S", "0.05")))
        elif mode == "anchor":
            out = Path(args["out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "events.jsonl").write_text(json.dumps({"kind": "summary"}) + "\n", encoding="utf-8")
            write_json(out / "summary.json", {"status": "complete", "anchor": True})
        elif mode == "speed":  # tools/speed_probe.py: one --out, merged per system
            out = Path(args["out"])
            got = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"systems": {}}
            got["systems"][args["system"]] = {"kind": args["kind"], "model": args.get("model"), "rtf": 0.01}
            write_json(out, got)
        else:
            rc = 2
    finally:
        record.update(t1=time.time(), rc=rc)
        if os.environ.get("FAKE_LOG"):  # a dir: one file per process (concurrent appends to one file interleave)
            d = Path(os.environ["FAKE_LOG"])
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"{record['t0']:.6f}-{os.getpid()}.tmp"
            tmp.write_text(json.dumps(record), encoding="utf-8")
            tmp.replace(tmp.with_suffix(".json"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
