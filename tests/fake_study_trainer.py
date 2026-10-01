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
calibration run's first try exits 1 at that step); FAKE_FLAT {run_name prefix: true} (the smoke's loss does not fall:
a calibration run logs it, a run with smoke.require_loss_decrease fails with SmokeFailed); FAKE_FAIL {item prefix: rc}
(an anchor or speed item exits rc: a failed readout); FAKE_HUB (a tests/fake_runs_repo.py dir: a main run commits its
log syncs there, FAKE_SYNCS of them, FAKE_BOX its box, retried on a 429 after FAKE_SYNC_WAITS).

The full runs (kitsune/full_queue.py; a config on the epochs clock, as configs/full/*.json): train_full writes what the
queue, its stall check and its smoke verdict read (build contract 4.4) - per step metrics/scalars.jsonl rows
(sched/train_s, time/step_s, mem/step_peak_reserved_gb, time/data_wait_s) and a beat of $KITSUNE_HEARTBEAT; the setup
events (threads, smoke_sdpa, memory_probe, dev_store, data.stores_reused, frame_preflight for family ctc); periodic full
states (full_step_<N>/ with model.pt, optimizer.pt, l2sp.pt, trainer.pt, trainer.json and a `checkpoint` event); the WSD
cooldown at 0.8 of the steps (phase cooldown, the pre_cooldown full state, ckpt_upload_ok); the export step_<S>/ and
summary.json (steps, epochs, stopped_early, end_reason, resume_resets, checkpoints.weights, throughput). Knobs (env,
JSON; {prefix: value} tables keyed by run name unless said otherwise): FAKE_EPOCH_STEPS (steps per epoch, 20; or a
table), FAKE_STEP_S (seconds per step), FAKE_STEP_VALUE (the logged time/step_s, 1.0; or a table), FAKE_MEM_GB (20.0),
FAKE_FULL_EVERY (steps between periodic full states, 5), FAKE_HB ("0": no beats), FAKE_HANG {name: step} (the first
launch stops beating at that step and sleeps until killed), FAKE_RC {name: rc or [rc per launch]} (that launch exits rc
at step FAKE_RC_AT, 2, after a failed summary; 0 in a list runs through), FAKE_ERROR {name: error} (every launch fails
at step 1 with that summary error, e.g. "SmokeFailed: ..."), FAKE_ERROR_RESUME (the same for resumed launches only, e.g.
"ResumeMismatch: ..."), FAKE_EARLY {name: frac} (an early stop at that fraction: early_stop with action cooldown, the
cooldown over 0.2 of the steps so far, summary stopped_early), FAKE_TIMED {name: every N steps} (timed full states into
the scratch DirHub FAKE_SCRATCH: one commit of the state and its runs/<id>/timed_state.json pointer that deletes the
previous state, a squash, a timed_state_upload_ok event), FAKE_DEADLINE_S (a KITSUNE_DEADLINE nearer than this, 3600: a
deadline_cooldown event with action start), FAKE_THREADS (torch_threads, default KITSUNE_THREADS_PER_GPU or 8),
FAKE_WAIT_FRAC (the loader's share: every step's time/data_wait_s = it x the step's time/step_s, 0.01; the summary's
throughput.data_wait_frac is the whole run's, start-ups included), FAKE_STARTUP_WAIT_S (added to the time/data_wait_s of
each launch's first step, as a real loader's spawned workers start: 0; each launch logs phase {name: "train", at_step}
before its first step, as the trainer does). --set schedule.resume_reset=true on a resumed launch: a resume_reset event
and summary resume_resets + 1 (the next full state follows it). One FAKE_LOG record per launch as above, plus launch,
resumed_from, deadline (KITSUNE_DEADLINE) and heartbeat (KITSUNE_HEARTBEAT).
More modes: readout (scripts/05_evaluate.py: --out DIR --ckpt DIR --system S: study.json with metrics m4 FAKE_M4 {system
prefix: value}, 0.12, m4_teacher, m4_ratio, jg, jg_nostyle, gate_pooled and strata.eval_jsut.cer_nostyle; exit 2 when
--ckpt is missing), eval (a registry eval item: --out DIR; FAKE_WRITES {item: {path under --out: JSON}}), check-resume
(python -m kitsune.full_queue check-resume --run-dir D: exits FAKE_CHECK_RESUME {run dir name prefix: rc}, 0). FAKE_FAIL
also fails stores, readout and eval items (rc, or [rc per launch]); FAKE_ITEM_S {item prefix: seconds} is how long a
stores, readout, eval or speed item takes.
"""
import json
import math
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


def hub_syncs(run: Path, seconds: float):
    """A main run's log syncs into the shared fake runs repo (FAKE_HUB: tests/fake_runs_repo.py): FAKE_SYNCS commits of
    runs/<run id>/events.jsonl spread over `seconds`, each retried on a 429 after the waits FAKE_SYNC_WAITS (JSON list),
    as kitsune/runlog.py's upload_retries do; a sync whose retries all fail is a sync_failed event (the trainer's next
    sync uploads the same files again). Without FAKE_HUB: just the time."""
    if not os.environ.get("FAKE_HUB"):
        time.sleep(seconds)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_runs_repo import DirHub, RateLimited

    hub, n = DirHub(os.environ["FAKE_HUB"]), int(os.environ.get("FAKE_SYNCS", "3"))
    waits = [0.0, *env_json("FAKE_SYNC_WAITS", [0.05, 0.2, 0.8])]
    for i in range(n):
        time.sleep(seconds / n)
        event(run, "sync", i=i)
        for attempt, w in enumerate(waits):
            time.sleep(w)
            try:
                hub.commit({f"runs/{run.name}/events.jsonl": (run / "events.jsonl").read_bytes()}, writer=run.name,
                           box=os.environ.get("FAKE_BOX"))
                event(run, "sync_ok", i=i, attempt=attempt)
                break
            except RateLimited:
                event(run, "sync_error", i=i, attempt=attempt)
        else:
            event(run, "sync_failed", i=i)


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
    if (cfg.get("schedule") or {}).get("clock") == "epochs":  # a full run (configs/full/*.json)
        return train_full(run, cfg, name, resumed, record, args["sets"])
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
        return calibrate(run, cfg, name, M, record)
    crash_at = int(os.environ["KITSUNE_CRASH_AT_STEP"]) if os.environ.get("KITSUNE_CRASH_AT_STEP") else None
    sm = cfg.get("smoke") or {}
    if not resumed and not (cfg.get("lr_probe") or {}).get("enabled") and sm.get("enabled") and \
            sm.get("require_loss_decrease") and by_prefix(env_json("FAKE_FLAT", {}), name, False):
        event(run, "smoke_steps", steps=int(sm.get("steps", 100)), loss_first=9.7, loss_last=9.7,
              loss_decreasing=False)
        write_json(run / "summary.json", {"status": "failed",
                                           "error": "SmokeFailed: loss did not fall over the smoke steps (9.7 -> 9.7)"})
        return 1
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
    hub_syncs(run, main_s(name))
    ckpt(run, f"step_{M}")
    ckpt(run, f"full_step_{M}", full=True)
    write_json(run / "summary.json", {"status": "complete", "steps": M})
    return 0


def calibrate(run: Path, cfg: dict, name: str, M: int, record: dict) -> int:
    """Steps until the STOP file or max_steps: per step the loop clock, its data wait, the wall time. With
    calibrate.barrier: READY, then wait for GO (or STOP) before the first step. The smoke checks at smoke.steps: a
    smoke_steps event (FAKE_FLAT {run_name prefix: true}: the loss did not fall), and with smoke.require_loss_decrease a
    flat loss fails the run as the trainer's SmokeFailed."""
    step_s = float(os.environ.get("FAKE_STEP_S", "0.002"))
    workers = (cfg.get("perf") or {}).get("num_workers")
    waits = env_json("FAKE_WAIT", {})
    w12 = {k[: -len(".w12")]: v for k, v in waits.items() if k.endswith(".w12")}
    plain = {k: v for k, v in waits.items() if not k.endswith(".w12")}
    frac = by_prefix(w12 if workers == 12 else plain, name, 0.01)
    sm = cfg.get("smoke") or {}
    smoke_n = int(sm.get("steps", 100)) if sm.get("enabled") else 0
    flat = bool(by_prefix(env_json("FAKE_FLAT", {}), name, False))
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    if (cfg.get("calibrate") or {}).get("barrier"):
        (run / "READY").write_text(f"{time.time()}\n", encoding="utf-8")
        record["ready"] = time.time()
        while not ((run / "GO").exists() or (run / "STOP").exists()):
            if time.time() - record["ready"] > 120:
                write_json(run / "summary.json", {"status": "failed", "error": "RuntimeError: no GO"})
                return 1
            time.sleep(0.002)
        record["released"] = time.time()
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
            record.setdefault("first_step", wall)
            for tag, v in (("sched/train_s", t), ("time/data_wait_s", frac * dt), ("time/step_s", dt)):
                f.write(json.dumps({"step": s, "wall": wall, "tag": tag, "value": v}) + "\n")
            f.flush()
            if s == smoke_n:
                event(run, "smoke_steps", steps=s, loss_first=9.7, loss_last=9.7 if flat else 8.1,
                      loss_decreasing=not flat)
                if flat and sm.get("require_loss_decrease"):
                    write_json(run / "summary.json", {
                        "status": "failed", "error": "SmokeFailed: loss did not fall over the smoke steps (9.7 -> 9.7)"})
                    return 1
    micro = float(cfg["batch"]["micro_audio_s"])
    write_json(run / "summary.json", {"status": "complete", "steps": s, "calibrate": {
        "micro_audio_s": micro, "workers": workers if isinstance(workers, int) else 8}})
    return 0


# ============================================================================================ the full runs' modes


STATE_FILES = ("model.pt", "optimizer.pt", "l2sp.pt", "trainer.pt")  # + trainer.json (fullrun.STATE_FILES_REQUIRED)
SCRATCH_MARK = ".scratch_pending"


def per_name(var: str, name: str, default):
    """An env knob that is either one value or a {prefix: value} table."""
    v = env_json(var, default)
    return by_prefix(v, name, default) if isinstance(v, dict) else v


def launch_count(folder: Path, key: str) -> int:
    """This launch's number for `key` (1 = the first): a counter file in folder (the run dir, or the checkout)."""
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f".fake_launches-{key}"
    n = int(f.read_text(encoding="utf-8")) + 1 if f.is_file() else 1
    f.write_text(str(n), encoding="utf-8")
    return n


def rc_of(v, launch: int) -> int:
    """A FAKE_RC / FAKE_FAIL value for this launch: an rc, or a list of rcs per launch (the last one repeats)."""
    if isinstance(v, list):
        return int(v[min(launch, len(v)) - 1]) if v else 0
    return int(v)


def item_rc(record: dict) -> int:
    """FAKE_FAIL for a stores, readout or eval item: its rc for this launch (0 when the table does not name it)."""
    v = by_prefix(env_json("FAKE_FAIL", {}), record["item"] or "", None)
    return 0 if v is None else rc_of(v, launch_count(Path.cwd() / ".fake_counts", record["item"] or "item"))


def item_sleep(record: dict):
    time.sleep(float(by_prefix(env_json("FAKE_ITEM_S", {}), record["item"] or "", 0.0)))


def full_state(run: Path, step: int, reason: str, st: dict, mark: str | None = None) -> Path:
    """checkpoints/full_step_<step>/ as save_full writes it (a .tmp dir renamed in), trainer.json with reason and
    st."""
    import shutil  # here, not at the top: every launch of this fake (the study box's too) starts as fast as the base's

    d = run / "checkpoints" / f"full_step_{step}"
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in STATE_FILES:
        (tmp / f).write_bytes(f"{run.name}:{step}:{f}".encode() * 4)
    write_json(tmp / "trainer.json", {"format": 1, "step": step, "reason": reason, "run_id": run.name, "st": st})
    if mark:
        (tmp / mark).touch()
    shutil.rmtree(d, ignore_errors=True)
    tmp.replace(d)
    return d


def timed_upload(run: Path, step: int, epoch: float, st: dict):
    """A timed full state into the scratch DirHub FAKE_SCRATCH, as kitsune/scratch.py sends it: one commit of the state
    and its pointer (runs/<id>/timed_state.json, format 1) that deletes the run's previous state, then a squash."""
    import hashlib

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_runs_repo import DirHub

    d = full_state(run, step, "timed", st, mark=SCRATCH_MARK)
    hub = DirHub(os.environ["FAKE_SCRATCH"])
    base = f"runs/{run.name}/checkpoints"
    files, meta = {}, {}
    for f in sorted(p for p in d.iterdir() if p.name != SCRATCH_MARK):
        data = f.read_bytes()
        files[f"{base}/{d.name}/{f.name}"] = data
        meta[f.name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    pointer = {"format": 1, "run_id": run.name, "name": d.name, "step": step, "epoch": epoch, "wall": time.time(),
               "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kitsune_sha": None,
               "planner_fingerprint": "fake", "n_train_utts": 100, "selection_sha256": None, "micro_audio_s": 600.0,
               "files": meta, "host": {"hostname": "fake", "machine_id": None, "container_id": None}}
    files[f"runs/{run.name}/timed_state.json"] = json.dumps(pointer, indent=1).encode()
    old = sorted({"/".join(p.split("/")[:4]) for p in hub.listing(base)} - {f"{base}/{d.name}"})
    hub.commit(files, writer=run.name, delete=old)
    hub.squash()
    (d / SCRATCH_MARK).unlink()
    event(run, "timed_state_upload_ok", name=d.name, step=step, attempt=1, gb=0.0, upload_s=0.0, deleted=len(old),
          squash_ok=True)


def fail(run: Path, error: str, rc: int = 1) -> int:
    write_json(run / "summary.json", {"status": "failed", "run_id": run.name, "error": error})
    return rc


def run_wait_frac(run: Path) -> float:
    """summary.throughput.data_wait_frac as the trainer reports it: the whole run's sum(time/data_wait_s) /
    sum(time/step_s), every launch's start-up included (the merged rows: a resumed launch's step replaces the old)."""
    rows: dict[tuple[str, int], float] = {}
    for line in (run / "metrics" / "scalars.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[(r["tag"], r["step"])] = r["value"]
    wait = sum(v for (t, _), v in rows.items() if t == "time/data_wait_s")
    tot = sum(v for (t, _), v in rows.items() if t == "time/step_s")
    return wait / tot if tot > 0 else 0.0


def train_full(run: Path, cfg: dict, name: str, resumed: bool, record: dict, sets: list[str]) -> int:
    """A full run on the epochs clock (the module docstring): the steps, events, states and summary the full-run queue
    reads, fast."""
    launch = launch_count(run, "run")
    record.update(launch=launch)
    step_s = float(os.environ.get("FAKE_STEP_S", "0.002"))
    eps = int(per_name("FAKE_EPOCH_STEPS", name, 20))
    epochs = int((cfg.get("schedule") or {}).get("epochs") or 1)
    fam = cfg.get("family", "aed")
    hb = os.environ.get("KITSUNE_HEARTBEAT") if os.environ.get("FAKE_HB", "1") != "0" else None
    st = dict(total_steps=epochs * eps, t_c=None, pre_cooldown_done=False, early_stop={"triggered": None},
              resume_resets=0, end_reason="schedule")
    start = 0
    if resumed:
        fulls = sorted(int(p.name.rsplit("_", 1)[1]) for p in (run / "checkpoints").glob("full_step_*")
                       if p.name.rsplit("_", 1)[1].isdigit() and (p / "trainer.json").is_file())
        if fulls:
            start = fulls[-1]
            st.update(json.loads((run / "checkpoints" / f"full_step_{start}" / "trainer.json")
                                 .read_text(encoding="utf-8"))["st"])
        record["resumed_from"] = start
        err = by_prefix(env_json("FAKE_ERROR_RESUME", {}), name)
        if err:
            return fail(run, err)
    event(run, "phase", name="setup")
    if resumed:
        event(run, "resume", step=start, host={"hostname": "fake"})
    if resumed and (cfg.get("schedule") or {}).get("resume_reset"):
        before = dict(total_steps_before=st["total_steps"], early_stop_before=st["early_stop"]["triggered"])
        st.update(total_steps=epochs * eps, t_c=None, pre_cooldown_done=False, early_stop={"triggered": None},
                  resume_resets=st["resume_resets"] + 1, end_reason="schedule")
        if st["total_steps"] <= start:
            return fail(run, f"SystemExit: the reset's new total {st['total_steps']} steps <= step {start}")
        event(run, "resume_reset", at_step=start, total_steps_after=st["total_steps"], epochs=epochs, **before)
    threads = int(os.environ.get("FAKE_THREADS") or os.environ.get("KITSUNE_THREADS_PER_GPU") or 8)
    event(run, "threads", torch_threads=threads, interop_threads=1,
          env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                              "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS",
                                              "KITSUNE_CPU_QUOTA", "KITSUNE_THREADS_PER_GPU")})
    if fam == "ctc":
        for store in ("train", "eval", "dev"):
            event(run, "frame_preflight", store=store, reused=True, rows=100, dropped=0)
    event(run, "dev_store", name="dev_p60_s1234_fake", n=60, per_source=60, seed=1234, ids_sha256="0" * 64,
          hours=0.5, n_in_train=0)
    event(run, "data", train_utts=100, dev_utts=60, stores_reused={"train": True, "eval": True, "dev": True})
    mem = float(os.environ.get("FAKE_MEM_GB", "20.0"))
    event(run, "memory_probe", peak_gb={"longest": mem - 2, "most_rows": mem - 1}, max_reserved_gb=mem)
    if not resumed:
        event(run, "smoke_sdpa", ok=True, backend="fake")
    deadline = os.environ.get("KITSUNE_DEADLINE")
    if deadline and float(deadline) - time.time() < float(os.environ.get("FAKE_DEADLINE_S", "3600")):
        event(run, "deadline_cooldown", action="start", at_step=start, deadline=float(deadline),
              left_s=float(deadline) - time.time(), T=st["total_steps"], T_before=st["total_steps"])
    err = by_prefix(env_json("FAKE_ERROR", {}), name)
    rcv = by_prefix(env_json("FAKE_RC", {}), name)
    rc = rc_of(rcv, launch) if rcv is not None else 0
    rc_at = max(start + 1, int(os.environ.get("FAKE_RC_AT", "2")))
    hang = by_prefix(env_json("FAKE_HANG", {}), name) if launch == 1 else None
    early = by_prefix(env_json("FAKE_EARLY", {}), name) if not st["resume_resets"] else None
    every = int(os.environ.get("FAKE_FULL_EVERY", "5"))
    timed = by_prefix(env_json("FAKE_TIMED", {}), name)
    sv = float(per_name("FAKE_STEP_VALUE", name, 1.0))
    wf, w0 = float(os.environ.get("FAKE_WAIT_FRAC", "0.01")), float(os.environ.get("FAKE_STARTUP_WAIT_S", "0"))
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    event(run, "phase", name="train", at_step=start)
    s = start
    with open(run / "metrics" / "scalars.jsonl", "a", encoding="utf-8") as f:
        while s < st["total_steps"]:
            s += 1
            if err:
                return fail(run, err)
            if rc and s >= rc_at:
                return fail(run, f"RuntimeError: fake exit {rc}", rc)
            if hang is not None and s >= int(hang):
                record["hung_at"] = s
                while True:  # no beat, no step: the queue's stall check kills it
                    time.sleep(0.05)
            time.sleep(step_s)
            wall = time.time()
            for tag, v in (("sched/train_s", s * sv), ("time/step_s", sv), ("mem/step_peak_reserved_gb", mem),
                           ("time/data_wait_s", wf * sv + (w0 if s == start + 1 else 0.0))):
                f.write(json.dumps({"step": s, "wall": wall, "tag": tag, "value": v}) + "\n")
            f.flush()
            if hb:
                Path(hb).parent.mkdir(parents=True, exist_ok=True)
                Path(hb).touch()
            T = st["total_steps"]
            if early is not None and not st["early_stop"]["triggered"] and s >= round(float(early) * T):
                trig = {"trigger": "patience", "action": "cooldown", "at_step": s, "metric": "dev_ce"}
                st["early_stop"]["triggered"] = trig
                st.update(t_c=s, total_steps=s + max(1, math.ceil(0.2 * s)), end_reason="early_stop")
                event(run, "early_stop", step=s, **trig)
            if not st["pre_cooldown_done"] and s >= (st["t_c"] or math.ceil(0.8 * T)):
                st["pre_cooldown_done"] = True
                d = full_state(run, s, "pre_cooldown", st)
                event(run, "checkpoint", ckpt="full", reason="pre_cooldown", name=d.name, step=s)
                event(run, "ckpt_upload_ok", ckpt="full", name=d.name, step=s)
                event(run, "phase", name="cooldown", at_step=s, clock="epochs")
            if s % every == 0:
                d = full_state(run, s, "periodic", st)
                event(run, "checkpoint", ckpt="full", reason="periodic", name=d.name, step=s)
            if timed and s % int(timed) == 0:
                timed_upload(run, s, epochs * s / st["total_steps"], st)
    T = st["total_steps"]
    ckpt(run, f"step_{T}")
    full_state(run, T, "end", st)
    event(run, "checkpoint", ckpt="full", reason="end", name=f"full_step_{T}", step=T)
    trig = st["early_stop"]["triggered"]
    fulls = sorted(int(p.name.rsplit("_", 1)[1]) for p in (run / "checkpoints").glob("full_step_*")
                   if p.name.rsplit("_", 1)[1].isdigit())
    write_json(run / "summary.json", {
        "status": "complete", "run_id": run.name, "family": fam, "steps": T, "epochs": round(T / eps, 4),
        "stopped_early": trig, "early_stop_trigger": trig, "end_reason": st["end_reason"],
        "resume_resets": st["resume_resets"], "checkpoints": {"weights": [T], "full": fulls},
        "throughput": {"data_wait_frac": run_wait_frac(run)}, "dev_history": []})
    return 0


def readout(args: dict, record: dict) -> int:
    """scripts/05_evaluate.py's study mode: study.json (the metrics the queue records) and its tables dir."""
    rc = item_rc(record)
    item_sleep(record)
    if rc:
        return rc
    if not Path(args["ckpt"]).is_dir():
        return 2
    out = Path(args["out"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    m4 = float(by_prefix(env_json("FAKE_M4", {}), args.get("system") or "", 0.12))
    write_json(out / "study.json", {"system": args.get("system"), "metrics": {
        "m4": m4, "m4_teacher": 0.1, "m4_ratio": m4 / 0.1, "jg": 0.09, "jg_nostyle": 0.08, "gate_pooled": 0.07},
        "strata": {"eval_jsut": {"cer": 0.06, "cer_nostyle": 0.05}}})
    event(out, "study_tables", system=args.get("system"), m4=m4)
    return 0


def eval_item(args: dict, record: dict) -> int:
    """A registry eval item (the quant readout, a Whisper eval, a compare): --out gets FAKE_WRITES[item]'s files."""
    rc = item_rc(record)
    item_sleep(record)
    if rc:
        return rc
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    for rel, obj in (by_prefix(env_json("FAKE_WRITES", {}), record["item"] or "", {}) or {}).items():
        write_json(out / rel, obj)
    write_json(out / "fake_eval.json", {"argv": sys.argv[1:]})
    return 0


def main() -> int:
    mode, args = sys.argv[1], parse(sys.argv[2:])
    record = dict(mode=mode, t0=time.time(), argv=sys.argv[1:], gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  item=os.environ.get("KITSUNE_QUEUE_ITEM"), deadline=os.environ.get("KITSUNE_DEADLINE"),
                  heartbeat=os.environ.get("KITSUNE_HEARTBEAT"))
    rc = 0
    try:
        if mode == "train":
            rc = train(args, record)
        elif mode == "stores":
            time.sleep(float(os.environ.get("FAKE_STORES_S", "0.05")))
            item_sleep(record)
            rc = item_rc(record)
        elif mode == "readout":
            rc = readout(args, record)
        elif mode == "eval":
            rc = eval_item(args, record)
        elif mode == "check-resume":
            rc = int(by_prefix(env_json("FAKE_CHECK_RESUME", {}), Path(args["run-dir"]).name, 0))
        elif mode in ("anchor", "speed") and by_prefix(env_json("FAKE_FAIL", {}), record["item"] or "", None):
            rc = int(by_prefix(env_json("FAKE_FAIL", {}), record["item"] or ""))  # a readout that fails
        elif mode == "anchor":
            out = Path(args["out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "events.jsonl").write_text(json.dumps({"kind": "summary"}) + "\n", encoding="utf-8")
            write_json(out / "summary.json", {"status": "complete", "anchor": True})
        elif mode == "speed":  # tools/speed_probe.py: one --out, merged per system
            item_sleep(record)
            out = Path(args["out"])
            got = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"systems": {}}
            got["systems"][args["system"]] = {"kind": args["kind"], "model": args.get("model"), "rtf": 0.01,
                                              "hf_token": bool(os.environ.get("HF_TOKEN"))}
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
