"""The study box's queue: one box plan of the size study, from the store build to the last verified upload.

A box (CONTRACT.md section 6: A = the Cohere box, B = Parakeet + bridge, replicate = T-0.1B's second seed on a 1x box;
plus the shakedown, STUDY.md 6.1) runs several trainers at once, one per GPU (CUDA_VISIBLE_DEVICES pinned), through
these phases. The plan is kitsune.prereg.rules()["boxes"][box] (runs, probe_classes, calibrate, reference,
numbers_file, numbers_from, extras); every rule it applies - the calibration window and its data-wait limit, the probe
grids and the edge rule, max_steps, the branch fractions - comes from kitsune.prereg, never from this file.

  stores       the label stores (the trainer's cache under cache_dir: the train and eval stores) are built ONCE, by one
               process alone (`build-stores`, the trainer's own store code), before any trainer starts; every run then
               finds its fingerprint and opens the shared cache read-only
  calibrate    the box's calibrate list in groups of as many runs as GPUs (a short last group is filled with other runs
               of the list as unmeasured load, so every run is measured under the same concurrency), 250 steps each at
               the planned micro-batch: calib-<run> configs (lr_probe + calibrate: metrics only). A group's windows are
               aligned (group_windows): each run's 200-step window starts at or after step 50 once EVERY run of the group
               has reached step 50, so each is measured while the others train; once all have theirs the queue ends the
               group with the STOP file. t_i = the median step time over the window, logging included (the difference
               of consecutive steps' loop clocks), data_wait_frac = the loader's share of it; the literal steps
               (50, 250] are kept next to it (fixed_window, for the record). A run at or above
               prereg.DATA_WAIT_MAX: perf.num_workers 12 for every run of the box and the whole calibration again; still
               loader-bound: the box halts
  probes       every grid point of the box's probe classes, packed onto the GPUs longest first (steps x the calibrated
               step time of the probed run); a class whose winner sits on a grid edge gets its one extension point
               (prereg.choose_lr: x2 / /2), an edge winner after it halts the box. A probe that diverges (the trainer's
               FloatingPointError) scores as non-finite and loses
  numbers      max_steps (prereg.max_steps against the box's reference run), the probes' objectives and the chosen LRs
               go into the box's PREREG_numbers file (prereg.write_numbers), which is uploaded to the runs repo
               (study/<numbers_file>), verified on the Hub, and its sha256 logged (a `prereg_numbers` event) BEFORE the
               first study step. A box with numbers_from (the replicate) takes its runs' max_steps and LR from that
               box's uploaded file instead (derived_numbers). Once written, a box never writes its numbers again: a
               restart reuses the file, and one already on the Hub that this box did not write halts it
  wave         per GPU a main run, then its T/2 branch (<run>-half with branch.parent = the main's local run dir), each
               filled with --set schedule.max_steps, optim.lr, eval.mini.every_steps (prereg's mini fraction of
               max_steps), the calibrated micro-batch and, after a loader-bound calibration, perf.num_workers; the
               mains start SYNC_OFFSET_S apart, so their log syncs to the shared runs repo do not come together
  extras       "anchor": re-score the first run's 0.6B (ANCHOR_CKPT in the runs repo) on the study's eval rows with
               scripts/05_evaluate.py, in a GPU gap (a free GPU during the probes or the wave's tail); "speed":
               tools/speed_probe.py for every student and both teachers, one model at a time on an idle host, at the
               end (skipped with a logged note while the tool is absent)
Every finished run dir is uploaded and verified at once (lean: vast/finish.py expected_files(lean=True) - the logs,
metrics, evals and summary, the exported weights at 0.4 and at the end, the branch's, and the full states the config
uploads: the 0.8 one of the runs of at most 0.1B). Local full states stay on the box.

State: $KITSUNE_STATE/queue.json (atomic, fsynced) holds every item's status, attempts and run dir, the calibration
table, the probe results, the LR choice and the numbers file's sha256. A restarted queue (vast/supervise.py restarts it
after a crash, onstart.sh after a container restart) kills its previous trainers first, skips every finished item,
resumes a main run or a probe from its local full state (--resume), starts a branch again from its parent's state, and
runs an unfinished calibration again from the start. A run dir given up this way is moved to runs/_abandoned/.
Exit (vast/supervise.py keys on it): 0 everything done and verified, EXIT_HALT a pre-registered halt (calibration
loader-bound, an LR edge after its extension, a refused numbers file), EXIT_THROUGHPUT a trainer's throughput floor,
EXIT_FAIL anything else; $KITSUNE_STATE/queue_summary.json says what happened (finish.py uploads it with the infra
logs; the queue also puts it at study/box-<box>/queue_summary.json in the runs repo).

Usage (the box runs it through vast/supervise.py; KITSUNE_BOX, KITSUNE_OUT_REPO and KITSUNE_STATE from the env):
  python -m kitsune.study_queue run --box A
  python -m kitsune.study_queue plan --box B              # the items it would run, nothing started
  python -m kitsune.study_queue students --box A          # the student dirs the box pulls (vast/bootstrap.sh)
  python -m kitsune.study_queue build-stores --config configs/study/calib-study-t06.json
Pure Python at import (stdlib + kitsune.prereg); the trainer's modules only inside build-stores.
"""
import argparse
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from kitsune import prereg

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = "configs/study"
STATE_FILE, SUMMARY_FILE, EVENTS_FILE = "queue.json", "queue_summary.json", "events.jsonl"
EXIT_OK, EXIT_FAIL, EXIT_THROUGHPUT, EXIT_HALT = 0, 1, 3, 4
PLAN_KEYS = ("runs", "probe_classes", "calibrate", "reference", "numbers_file", "numbers_from", "extras")
BOXES = ("A", "B", "replicate", "shakedown")
RETRY_NUM_WORKERS = 12  # the pre-registered loader retry (prereg rules: calibration.loader_bound)
SYNC_OFFSET_S = 45.0  # between the starts of the wave's main runs: their log syncs (every 20 min) stay apart
NUMBERS_DIR = "study"  # the numbers file's folder in the runs repo, and in the box's checkout
ANCHOR_RUN = "viability-b20x2560-20260925T071746Z"  # decision 29: the first run's 0.6B at its final step
ANCHOR_CKPT = f"runs/{ANCHOR_RUN}/checkpoints/step_9774"
ANCHOR_CONFIG = f"{CONFIG_DIR}/anchor-b20.json"
SPEED_TOOL = "tools/speed_probe.py"
PARAKEET_DIR = "models/parakeet-tdt_ctc-0.6b-ja-hf"  # the converted teacher in the data repo (kitsune.parakeet)
# the teachers' systems for the speed probe (tools/speed_probe.py --kind; the Cohere teacher comes from the HF cache):
# system -> (kind, model dir or None)
TEACHERS = {"cohere": ("cohere", None), "parakeet-ctc": ("parakeet-ctc", PARAKEET_DIR),
            "parakeet-tdt": ("parakeet-tdt", PARAKEET_DIR)}
SPEED_PER_SET = 40  # ids per eval set of the speed probe's fixed list (its default; one list for every system)
CALIB_GROUP_TIMEOUT_S = 5400  # a group whose windows are not all in by then is stopped and counts as failed
STOP_GRACE_S = 1800  # how long a stopped run may take to leave (its end phase)
MAX_ATTEMPTS = 2  # per item: a run resumes (or a branch restarts) once; a second failure is final
STAMP_RE = r"-\d{8}T\d{6}Z(?:-\d+)?"
DIVERGED = ("FloatingPointError",)  # a probe whose trainer stopped on non-finite gradients: its objective is non-finite


class QueueError(RuntimeError):
    """Anything the queue cannot go on from (EXIT_FAIL)."""


class Halt(RuntimeError):
    """A pre-registered halt (EXIT_HALT): the box stops for the owner."""


class ThroughputHalt(Halt):
    """A trainer's throughput floor (its exit 3): a slow host does not get faster (EXIT_THROUGHPUT)."""


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [queue] {msg}", flush=True)


# ================================================================================================ metrics readers


def read_events(run_dir) -> list[dict]:
    """A run dir's events.jsonl (a torn last line is skipped)."""
    p = Path(run_dir) / "events.jsonl"
    out = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


STEP_TAGS = {"sched/train_s": "train_s", "time/data_wait_s": "data_wait_s", "time/step_s": "step_s"}


def _merge_line(rows: dict, line: str):
    try:
        r = json.loads(line)
    except ValueError:
        return
    key = STEP_TAGS.get(r.get("tag"))
    if key is None or not isinstance(r.get("step"), int):
        return
    row = rows.setdefault(r["step"], {})
    row[key] = r.get("value")
    if key == "train_s":
        row["wall"] = r.get("wall")


def read_step_rows(run_dir) -> dict[int, dict]:
    """step -> {train_s (the loop clock when the step was logged), data_wait_s, step_s, wall} from a run's
    metrics/scalars.jsonl (flushed at every step; a later line of the same step, a resumed launch's, wins)."""
    rows: dict[int, dict] = {}
    p = Path(run_dir) / "metrics" / "scalars.jsonl"
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            _merge_line(rows, line)
    return rows


class StepTail:
    """read_step_rows, incrementally: each poll() reads only what was appended since (the calibration monitor polls
    every run of a group every few seconds, and a small run's file grows by ~7 KB a step)."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "metrics" / "scalars.jsonl"
        self.pos, self.rest, self.rows = 0, b"", {}

    def poll(self) -> dict[int, dict]:
        if not self.path.is_file():
            return self.rows
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)
        data = self.rest + data
        lines = data.split(b"\n")
        self.rest = lines.pop()  # a line still being written
        for line in lines:
            _merge_line(self.rows, line.decode("utf-8", "replace"))
        return self.rows


def calibration_stats(rows: dict[int, dict], a: int, b: int) -> dict:
    """The step-time table of steps (a, b] (prereg: the median step time over the window, logging included): each
    step's time is the difference of its loop clock and the previous step's (both logged), so it holds the data wait,
    the forward and backward, the optimizer, the step's logging and anything else between the two; data_wait_frac is
    the loader's share of the summed time. Steps without their predecessor (a skipped step) are left out."""
    dts, waits, done = [], [], []
    for s in range(int(a) + 1, int(b) + 1):
        cur, prev = rows.get(s), rows.get(s - 1)
        if not cur or not prev or cur.get("train_s") is None or prev.get("train_s") is None:
            continue
        dt = float(cur["train_s"]) - float(prev["train_s"])
        if not (math.isfinite(dt) and dt > 0):
            continue
        dts.append(dt)
        waits.append(float(cur.get("data_wait_s") or 0.0))
        done.append(s)
    if not dts:
        return dict(t_step_s=None, data_wait_frac=None, steps_measured=0, window=[int(a), int(b)], mean_step_s=None)
    return dict(t_step_s=round(statistics.median(dts), 6), data_wait_frac=round(sum(waits) / sum(dts), 6),
                steps_measured=len(dts), window=[int(a), max(done)], mean_step_s=round(sum(dts) / len(dts), 6))


def group_windows(rows_by_run: dict[str, dict], a: int, n: int) -> dict[str, tuple[int, int] | None]:
    """The aligned calibration windows of a group running concurrently: t_all = the wall time at which the LAST run
    logged step a; each run's window (s0, e] starts at its first logged step s0 >= a with wall >= t_all and ends once
    it holds n measured steps (a step and its predecessor both logged; calibration_stats), so every step of it falls
    after every run of the group reached step a (the run that got there last measures exactly steps (a, a + n] unless
    it skipped a step). None for a run whose window is not complete yet (or while some run has not reached step a)."""
    walls = {}
    for run, rows in rows_by_run.items():
        w = (rows.get(a) or {}).get("wall")
        if w is None:
            return {r: None for r in rows_by_run}
        walls[run] = float(w)
    t_all = max(walls.values())
    out = {}
    for run, rows in rows_by_run.items():
        s0 = min((s for s, r in rows.items() if s >= a and r.get("wall") is not None and float(r["wall"]) >= t_all),
                 default=None)
        out[run] = None
        if s0 is None:
            continue
        got = 0
        for s in range(s0 + 1, max(rows) + 1):
            if (rows.get(s) or {}).get("train_s") is not None and (rows.get(s - 1) or {}).get("train_s") is not None:
                got += 1
                if got == n:
                    out[run] = (s0, s)
                    break
    return out


# ============================================================================================================ plans


def box_plan(box: str, rules: dict | None = None) -> dict:
    """The box's plan: kitsune.prereg.rules()["boxes"][box] (the pre-registered two-box split, CONTRACT.md 6), or the
    shakedown's (shakedown_plan). Refuses a plan without the keys the queue reads."""
    r = rules if rules is not None else prereg.rules()
    if box == "shakedown":
        return shakedown_plan(r)
    boxes = r.get("boxes")
    if not isinstance(boxes, dict) or box not in boxes:
        raise QueueError(f"kitsune.prereg.rules() has no boxes.{box} (the box plans of CONTRACT.md section 6: "
                         f"{', '.join(k for k in BOXES if k != 'shakedown')})")
    plan = dict(boxes[box])
    plan.setdefault("numbers_from", None)  # only a box that takes its numbers from another names one
    plan.setdefault("reference", None)  # a box without calibration has none
    plan.setdefault("extras", [])
    if missing := [k for k in PLAN_KEYS if k not in plan]:
        raise QueueError(f"boxes.{box} lacks {missing}")
    unknown = [x for x in [*plan["runs"], *plan["calibrate"], *([plan["reference"]] if plan["reference"] else [])]
               if x not in r["runs"]]
    if unknown or any(c not in r["lr_probes"]["classes"] for c in plan["probe_classes"]):
        raise QueueError(f"boxes.{box} names runs or probe classes the rules do not have: {unknown}")
    if plan["calibrate"] and plan["reference"] not in plan["calibrate"]:
        raise QueueError(f"boxes.{box}: the reference run {plan['reference']} is not calibrated on the box")
    return dict(plan, box=box)


def first_box_runs(rules: dict) -> list[str]:
    """The runs of the first study box (box A), which the shakedown exercises: rules()["boxes"]["A"]["runs"]; before
    the rules carry box plans, the Cohere box of CONTRACT.md section 6 (the Transcribe runs but the bridge and the
    replicate)."""
    a = (rules.get("boxes") or {}).get("A")
    if a:
        return list(a["runs"])
    return [run for run in rules["waves"]["1"] + rules["waves"]["2"]
            if rules["runs"][run]["family"] == "aed" and rules["runs"][run]["lr_from"] != "bridge"]


def shakedown_plan(rules: dict) -> dict:
    """STUDY.md 6.1 on a 1x box: the stores of the real extent, then per box-A run a 100-step smoke at the planned
    micro (shake-smoke-<run>), a crash and resume (shake-resume), a toy parent and its T/2 branch (shake-parent,
    shake-parent-half), a complete eval on a subset (shake-eval); every run dir uploaded and verified. No calibration,
    no probes, no numbers: it may run on a PREREG with pending fields."""
    runs = first_box_runs(rules)
    return dict(box="shakedown", runs=runs, probe_classes=[], calibrate=[], reference=None, numbers_file=None,
                numbers_from=None, extras=[], shakedown=True,
                items=[*(f"shake-smoke-{r}" for r in runs), "shake-resume", "shake-parent", "shake-parent-half",
                       "shake-eval"])


def config_path(name: str) -> str:
    return f"{CONFIG_DIR}/{name}.json"


def _box_student_runs(box: str, r: dict) -> list[str]:
    plan = box_plan(box, r)
    runs = list(plan["runs"]) + list(plan["calibrate"])
    runs += [r["lr_probes"]["classes"][c]["probed_on"] for c in plan["probe_classes"]]
    if "speed" in plan["extras"]:
        runs += list(r["runs"])
    return list(dict.fromkeys(runs))


def box_students(box: str, rules: dict | None = None) -> list[str]:
    """The student dirs (repo paths) the box pulls, and only those: the ones of its runs, its calibrated runs and its
    probes' runs; with the speed extra every study student (the probe times them all)."""
    r = rules if rules is not None else prereg.rules()
    return list(dict.fromkeys(r["runs"][run]["student"] for run in _box_student_runs(box, r)))


def box_ctc_students(box: str, rules: dict | None = None) -> list[str]:
    """The Parakeet-derived ones among box_students: they carry the CC-BY-4.0 attribution (MODEL_CARD.md)."""
    r = rules if rules is not None else prereg.rules()
    return list(dict.fromkeys(r["runs"][run]["student"] for run in _box_student_runs(box, r)
                              if r["runs"][run]["family"] == "ctc"))


def box_configs(box: str, rules: dict | None = None) -> list[str]:
    """Every config file (repo path) the box's queue reads: the ones the launch checks exist at the commit it runs."""
    r = rules if rules is not None else prereg.rules()
    plan = box_plan(box, r)
    if plan.get("shakedown"):
        return [config_path(n) for n in [*plan["runs"], *plan["items"]]]
    out = [config_path(f"calib-{run}") for run in plan["calibrate"]]
    out += [config_path(prereg.probe_run_name(c, lr)) for c in plan["probe_classes"]
            for lr in r["lr_probes"]["classes"][c]["grid"]]
    out += [config_path(n) for run in plan["runs"] for n in (run, f"{run}-half")]
    out += [ANCHOR_CONFIG] if "anchor" in plan["extras"] else []
    return list(dict.fromkeys(out))


# the study box's local checkpoints beyond one run's (vast/launch.py adds them to kitsune.extent.sizing): a full state is
# the fp32 weights, AdamW's two moments and the L2-SP anchor (16 bytes a parameter); a run keeps up to STATES_PER_RUN of
# them at once (keep_local 2, the kept 0.4 one, the 0.8 one of a small run, its branch's 2), and exports bf16 weights
# (2 bytes a parameter) WEIGHTS_PER_RUN times; the probes running at once keep 2 each until the queue prunes them
STATE_BYTES_PER_PARAM, STATES_PER_RUN, WEIGHTS_PER_RUN, PROBE_STATES = 16, 6, 4, 2
SHAKEDOWN_EXTRA_GB = 30


def study_extra_gb(box: str, rules: dict | None = None, n_gpus: int | None = None) -> float:
    """The disk a box needs for its checkpoints beyond one run's (kitsune.extent.DISK_BASE_GB holds those)."""
    r = rules if rules is not None else prereg.rules()
    plan = box_plan(box, r)
    if plan.get("shakedown"):
        return float(SHAKEDOWN_EXTRA_GB)
    params = [int(r["runs"][run]["params_total"]) for run in plan["runs"]]
    runs_b = sum(p * (STATE_BYTES_PER_PARAM * STATES_PER_RUN + 2 * WEIGHTS_PER_RUN) for p in params)
    probed = [int(r["runs"][r["lr_probes"]["classes"][c]["probed_on"]]["params_total"]) for c in plan["probe_classes"]]
    g = n_gpus or (4 if len(plan["runs"]) > 1 else 1)
    probes_b = g * max(probed, default=0) * STATE_BYTES_PER_PARAM * PROBE_STATES
    return round((runs_b + probes_b) / 1e9, 1)


def box_extra_dirs(box: str, rules: dict | None = None) -> list[str]:
    """Other repo dirs the box pulls: the Parakeet teacher's converted dir for the speed extra."""
    r = rules if rules is not None else prereg.rules()
    return [PARAKEET_DIR] if "speed" in box_plan(box, r)["extras"] else []


# ============================================================================================================ hub


def _finish():
    """vast/finish.py (stdlib-only at import): sync, verify, expected_files, hub_retry."""
    vast = str(REPO / "vast")
    if vast not in sys.path:
        sys.path.insert(0, vast)
    import finish

    return finish


class HubUploader:
    """The runs repo, through vast/finish.py: a run dir's lean sync and verification, one file up and verified, a file
    or a folder down."""

    def __init__(self, repo: str, repo_type: str = "model"):
        self.repo, self.repo_type, self._api = repo, repo_type, None

    def api(self):
        if self._api is None:
            self._api = _finish().hf_api()
        return self._api

    def sync_run(self, run_dir: Path) -> list[str]:
        """Upload the run dir (lean) and verify it; the problems (empty: every expected file is on the Hub)."""
        fin = _finish()
        try:
            fin.sync(self.api(), self.repo, self.repo_type, run_dir, False, False, lean=True)
        except Exception as e:  # noqa: BLE001  the verification below says what is missing
            log(f"sync of {run_dir.name} failed: {type(e).__name__}: {e}")
        return fin.verify(self.api(), self.repo, self.repo_type, fin.expected_files(run_dir, False, lean=True))

    def put_file(self, local: Path, path_in_repo: str) -> list[str]:
        fin = _finish()
        fin.hub_retry(lambda: self.api().upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                                     repo_id=self.repo, repo_type=self.repo_type,
                                                     commit_message=f"study box: {path_in_repo}"),
                      f"upload {path_in_repo}")
        return fin.verify(self.api(), self.repo, self.repo_type, {path_in_repo: Path(local)},
                          prefix_of=lambda p: p.rsplit("/", 1)[0])

    def exists(self, path_in_repo: str) -> bool:
        return bool(_finish().hub_retry(lambda: self.api().file_exists(self.repo, path_in_repo,
                                                                       repo_type=self.repo_type),
                                        f"exists {path_in_repo}"))

    def download(self, path_in_repo: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(_finish().hub_retry(lambda: hf_hub_download(self.repo, path_in_repo, repo_type=self.repo_type,
                                                                local_dir=str(local_dir)), f"download {path_in_repo}"))

    def download_dir(self, prefix: str, local_dir: Path) -> Path:
        from huggingface_hub import snapshot_download

        _finish().hub_retry(lambda: snapshot_download(self.repo, repo_type=self.repo_type, local_dir=str(local_dir),
                                                      allow_patterns=[f"{prefix}/*"]), f"download {prefix}")
        return Path(local_dir) / prefix


# ============================================================================================================ queue


@dataclass
class Settings:
    """Everything the queue takes from its environment (the tests replace the commands, GPUs, uploader and rules)."""
    root: Path = REPO
    state_dir: Path = field(default_factory=lambda: Path(os.environ.get("KITSUNE_STATE", "/workspace/kitsune_state")))
    out_repo: str | None = field(default_factory=lambda: os.environ.get("KITSUNE_OUT_REPO") or None)
    gpus: list[str] | None = None  # None: KITSUNE_GPUS (comma list), else nvidia-smi's indices
    python: str = sys.executable
    train_cmd: list[str] | None = None  # default: python scripts/04_distill.py
    stores_cmd: list[str] | None = None  # default: python -m kitsune.study_queue build-stores
    eval_cmd: list[str] | None = None  # default: python scripts/05_evaluate.py
    anchor_cmd: list[str] | None = None  # default: python -m kitsune.study_queue anchor (download, then eval_cmd)
    speed_cmd: list[str] | None = None  # default: python tools/speed_probe.py (when it exists)
    uploader: object = None  # default: HubUploader(out_repo); None without a repo (nothing uploaded, nothing verified)
    rules: dict | None = None  # default: kitsune.prereg.rules()
    rules_path: Path | None = None  # default: <root>/study/PREREG.json
    allow_pending: bool = False  # write_numbers on a PREREG with pending fields (dry runs only)
    poll_s: float = 10.0
    sync_offset_s: float = SYNC_OFFSET_S
    calib_timeout_s: float = CALIB_GROUP_TIMEOUT_S
    host: str | None = None
    env: dict = field(default_factory=dict)  # extra env for every child


def detect_gpus() -> list[str]:
    if os.environ.get("KITSUNE_GPUS"):
        return [g.strip() for g in os.environ["KITSUNE_GPUS"].split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60).stdout
        gpus = [x.strip() for x in out.splitlines() if x.strip().isdigit()]
        if gpus:
            return gpus
    except (OSError, subprocess.SubprocessError):
        pass
    return ["0"]


def _atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=str))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _pdeathsig():  # pragma: no cover - Linux only: a trainer dies with the queue that started it
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


class Queue:
    def __init__(self, box: str, settings: Settings | None = None):
        self.box = box
        self.s = settings or Settings()
        self.root = Path(self.s.root)
        self.rules = self.s.rules if self.s.rules is not None else prereg.rules()
        self.plan = box_plan(box, self.rules)
        self.gpus = list(self.s.gpus or detect_gpus())
        self.state_path = Path(self.s.state_dir) / STATE_FILE
        self.state = self._load_state()
        self.uploader = self.s.uploader if self.s.uploader is not None else (
            HubUploader(self.s.out_repo) if self.s.out_repo else None)
        self.procs: dict[str, subprocess.Popen] = {}
        self.rules_path = Path(self.s.rules_path) if self.s.rules_path else self.root / "study" / prereg.RULES_JSON
        self._pending_uploads: list[str] = []

    # -------------------------------------------------------------------------------------------------- state

    def _load_state(self) -> dict:
        fresh = dict(box=self.box, version=1, started=time.time(), items={}, calibration=None, num_workers=None,
                     probes={}, lr_choice=None, numbers=None, stores_done=False, final=None, abandoned=[])
        if not self.state_path.is_file():
            return fresh
        st = json.loads(self.state_path.read_text(encoding="utf-8"))
        if st.get("box") != self.box:
            raise QueueError(f"{self.state_path} is box {st.get('box')!r}'s queue, not {self.box!r}'s")
        return {**fresh, **st}

    def save(self):
        _atomic_json(self.state_path, self.state)

    def event(self, kind: str, **fields):
        """A lifecycle record in $KITSUNE_STATE/events.jsonl (finish.py's file; uploaded with the infra logs)."""
        p = Path(self.s.state_dir) / EVENTS_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall": time.time(), "source": "queue", "box": self.box, "kind": kind, **fields},
                               default=str) + "\n")
        log(f"{kind}: " + json.dumps(fields, default=str)[:600])

    def item(self, name: str) -> dict:
        return self.state["items"][name]

    def add_item(self, name: str, kind: str, config: str | None, sets: list[str] | None = None, *,
                 after: str | None = None, run: str | None = None, affinity: str | None = None,
                 env: dict | None = None, measured: bool = True) -> str:
        """Register an item once (a restart finds it registered, with its status)."""
        if name not in self.state["items"]:
            self.state["items"][name] = dict(kind=kind, config=config, sets=list(sets or []), after=after, run=run,
                                             affinity=affinity, env=dict(env or {}), measured=measured,
                                             status="pending", attempts=[], run_dir=None, verified=None,
                                             result=None)
        return name

    # ---------------------------------------------------------------------------------------------- the run

    def run(self) -> int:
        final = self.state.get("final")
        if final:
            log(f"box {self.box} already ended ({final['status']}: {final.get('reason')}); nothing to do")
            return int(final["rc"])
        self.kill_orphans()
        if self.uploader is not None:  # a restart: finished run dirs whose upload never ran or never verified
            self._pending_uploads = [n for n, it in self.state["items"].items()
                                     if it["status"] == "done" and it["run_dir"] and not it["verified"]
                                     and it["kind"] not in ("stores", "speed")]
        self.event("queue_start", gpus=self.gpus, plan={k: self.plan.get(k) for k in PLAN_KEYS},
                   restart=bool(self.state["items"]))
        rc, status, reason = EXIT_OK, "complete", None
        try:
            self.phase_stores()
            if self.plan.get("shakedown"):
                self.phase_shakedown()
            else:
                self.phase_calibrate()
                self.phase_probes()
                self.phase_numbers()
                self.phase_wave()
                self.phase_extras()
            self.drain_uploads()
            bad = sorted(n for n, it in self.state["items"].items()
                         if (it["status"] == "failed" and it["kind"] not in ("anchor", "speed"))
                         or (it["status"] == "done" and it["verified"] is False))
            readouts = sorted(n for n, it in self.state["items"].items()
                              if it["status"] == "failed" and it["kind"] in ("anchor", "speed"))
            if bad:
                rc, status, reason = EXIT_FAIL, "failed", f"items failed or not verified on the Hub: {bad}"
            elif readouts:  # a readout the owner can redo elsewhere: logged, never a reason to keep a box
                reason = f"complete; readouts failed (redo them from the Hub): {readouts}"
        except ThroughputHalt as e:
            rc, status, reason = EXIT_THROUGHPUT, "halted", str(e)
        except Halt as e:
            rc, status, reason = EXIT_HALT, "halted", str(e)
        except QueueError as e:
            rc, status, reason = EXIT_FAIL, "failed", str(e)
        finally:
            self.stop_all()
        # a failure is left open (the supervisor may restart the queue); a completed or halted box is final
        if status != "failed":
            self.state["final"] = dict(status=status, reason=reason, rc=rc, wall=time.time())
        self.save()
        self.event("queue_end", status=status, reason=reason, rc=rc)
        self.write_summary(status, reason, rc)
        return rc

    # ----------------------------------------------------------------------------------------- processes

    def _cmd(self, kind: str) -> list[str]:
        py = self.s.python
        if kind == "train":
            return list(self.s.train_cmd or [py, str(self.root / "scripts" / "04_distill.py")])
        if kind == "stores":
            return list(self.s.stores_cmd or [py, "-m", "kitsune.study_queue", "build-stores"])
        if kind == "eval":
            return list(self.s.eval_cmd or [py, str(self.root / "scripts" / "05_evaluate.py")])
        if kind == "anchor":
            return list(self.s.anchor_cmd or [py, "-m", "kitsune.study_queue", "anchor"])
        return list(self.s.speed_cmd or [py, str(self.root / SPEED_TOOL)])

    def run_dirs_of(self, run_name: str) -> list[Path]:
        runs = self.root / "runs"
        pat = re.compile(re.escape(run_name) + STAMP_RE + "$")
        return sorted((d for d in runs.iterdir() if d.is_dir() and pat.match(d.name)),
                      key=lambda d: d.stat().st_mtime) if runs.is_dir() else []

    def find_run_dir(self, name: str) -> Path | None:
        """The run dir of the item's last attempt: its resumed dir, else the newest <name>-<stamp> dir created since the
        attempt started."""
        it = self.item(name)
        att = it["attempts"][-1] if it["attempts"] else None
        if att and att.get("resume"):
            return self.root / att["resume"]
        t0 = att["t0"] if att else 0
        cands = [d for d in self.run_dirs_of(name) if d.stat().st_mtime >= t0 - 5]
        return cands[-1] if cands else None

    def _rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.root.resolve()).as_posix()

    def start(self, name: str, gpu: str, resume: Path | None = None):
        it = self.item(name)
        kind = it["kind"]
        env = dict(os.environ, **self.s.env, **it["env"], CUDA_VISIBLE_DEVICES=str(gpu), KITSUNE_QUEUE_ITEM=name)
        if kind in ("stores",):
            argv = self._cmd("stores") + ["--config", it["config"], *sum((["--set", s] for s in it["sets"]), [])]
        elif kind == "anchor":
            argv = self.anchor_argv(name)
        elif kind == "speed":
            argv = self._cmd("speed") + it["sets"]
        else:
            argv = self._cmd("train")
            argv += ["--resume", self._rel(resume)] if resume is not None else [
                "--config", it["config"], "--set", f"run_name={name}"]
            argv += sum((["--set", s] for s in it["sets"]), [])
            if self.s.out_repo:
                argv += ["--set", f"hf.output_repo={self.s.out_repo}"]
        attempt = dict(t0=time.time(), gpu=str(gpu), resume=self._rel(resume) if resume is not None else None,
                       argv=argv)
        if len(it["attempts"]) >= 1 and it["env"].get("KITSUNE_CRASH_AT_STEP"):
            env.pop("KITSUNE_CRASH_AT_STEP", None)  # the shakedown's crash: the first attempt only
            attempt["crash_env_dropped"] = True
        logs = Path(self.s.state_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = open(logs / f"{name}.log", "ab")
        kw = dict(cwd=str(self.root), env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if os.name == "posix":
            kw.update(start_new_session=True, preexec_fn=_pdeathsig)
        proc = subprocess.Popen(argv, **kw)
        out.close()
        attempt["pid"] = proc.pid
        it["attempts"].append(attempt)
        it["status"] = "running"
        self.procs[name] = proc
        self.save()
        self.event("item_start", item=name, item_kind=kind, gpu=str(gpu), attempt=len(it["attempts"]),
                   resume=attempt["resume"], argv=argv)
        return proc

    def _signal(self, proc: subprocess.Popen, sig):
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            else:
                proc.terminate()
        except (OSError, ProcessLookupError):
            pass

    def stop_all(self):
        """Terminate every trainer this queue still runs (a halt or an error): the process group, so its loader
        workers go too."""
        for name, proc in list(self.procs.items()):
            if proc.poll() is None:
                log(f"terminating {name} (pid {proc.pid})")
                self._signal(proc, signal.SIGTERM)
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    self._signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
                it = self.item(name)
                if it["status"] == "running":
                    it["status"] = "interrupted"
        self.procs.clear()
        self.save()

    def kill_orphans(self):
        """A restart: the trainers the previous queue process left running (a crash of the queue itself, not of the
        container) are killed before anything starts, so no GPU runs two jobs; their items resume below."""
        for name, it in self.state["items"].items():
            if it["status"] != "running" or not it["attempts"]:
                continue
            pid = it["attempts"][-1].get("pid")
            it["status"] = "interrupted"
            if os.name != "posix" or not pid:
                continue
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if name in cmdline or "04_distill" in cmdline or "05_evaluate" in cmdline:
                log(f"killing {name}'s orphaned process group {pid}")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass
        self.save()

    def set_aside(self, run_dir: Path | None, why: str):
        """A run dir given up (a calibration run again, a branch started again from its parent): moved to
        runs/_abandoned/, where finish.py's run-dir scan does not look."""
        if run_dir is None or not Path(run_dir).is_dir():
            return
        dest = self.root / "runs" / "_abandoned" / Path(run_dir).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(dest))
        self.state["abandoned"].append(dict(run_dir=Path(run_dir).name, why=why, wall=time.time()))
        self.event("run_dir_abandoned", run_dir=Path(run_dir).name, why=why)

    # -------------------------------------------------------------------------------------------- execution

    def _ready(self, name: str) -> bool:
        it = self.item(name)
        return it["status"] in ("pending", "interrupted", "retry") and (
            it["after"] is None or self.item(it["after"])["status"] == "done")

    def _next_for(self, gpu: str, todo: list[str], running: dict) -> str | None:
        busy = {r["name"] for r in running.values()}
        for name in todo:
            it = self.item(name)
            if name in busy or not self._ready(name):
                continue
            if it["affinity"] is not None and str(it["affinity"]) != str(gpu) and str(it["affinity"]) in self.gpus:
                continue
            return name
        return None

    def execute(self, todo: list[str], *, filler=None, monitor=None, stagger: bool = False, on_done=None):
        """Run the items of `todo` on the free GPUs until each has ended (done, failed or skipped). on_done(name) may
        append items to todo (a branch, a probe's extension, a retry); filler(gpu) gives an item for a GPU that nothing
        in todo is ready for (the anchor); monitor(running) runs at every poll (the calibration windows). A Halt or
        QueueError from a hook stops every running item and propagates."""
        running: dict[str, dict] = {}  # gpu -> {"name", "proc"}
        try:
            while True:
                started = 0
                for gpu in self.gpus:
                    if gpu in running:
                        continue
                    name = self._next_for(gpu, todo, running)
                    if name is None and filler is not None:
                        name = filler(gpu)
                    if name is None:
                        continue
                    if stagger and started:
                        time.sleep(self.s.sync_offset_s)
                    resume = self.resume_point(name)
                    running[gpu] = dict(name=name, proc=self.start(name, gpu, resume))
                    started += 1
                self.drain_uploads(one=True)
                if not running:
                    if not any(self._ready(n) for n in todo):
                        waiting = [n for n in todo if self.item(n)["status"] in ("pending", "interrupted", "retry")]
                        if waiting:  # their dependency failed: they can never run
                            for n in waiting:
                                self.item(n)["status"] = "skipped"
                                self.event("item_skipped", item=n, after=self.item(n)["after"])
                            self.save()
                        return
                    continue
                if monitor is not None:
                    monitor(running)
                time.sleep(self.s.poll_s)
                for gpu, r in list(running.items()):
                    rc = r["proc"].poll()
                    if rc is None:
                        continue
                    del running[gpu]
                    self.procs.pop(r["name"], None)
                    self.finished(r["name"], rc, todo, on_done)
        except BaseException:
            for r in running.values():
                if r["proc"].poll() is None:
                    self._signal(r["proc"], signal.SIGTERM)
            for r in running.values():
                try:
                    r["proc"].wait(60)
                except subprocess.TimeoutExpired:
                    self._signal(r["proc"], getattr(signal, "SIGKILL", signal.SIGTERM))
                if self.item(r["name"])["status"] == "running":
                    self.item(r["name"])["status"] = "interrupted"
                self.procs.pop(r["name"], None)
            self.save()
            raise

    def resume_point(self, name: str) -> Path | None:
        """Where an interrupted or failed item goes on: a main run or a probe from its newest local full state (the
        same run dir), nothing else (a calibration runs again, a branch starts again from its parent)."""
        it = self.item(name)
        if it["status"] not in ("interrupted", "retry") or it["kind"] not in ("main", "probe", "shake"):
            if it["status"] in ("interrupted", "retry") and it["kind"] in ("branch", "calib"):
                self.set_aside(self.root / it["run_dir"] if it["run_dir"] else self.find_run_dir(name),
                               f"{it['kind']} started again")
                it["run_dir"] = None
            return None
        rd = self.root / it["run_dir"] if it["run_dir"] else (self.find_run_dir(name) if it["attempts"] else None)
        if rd is None or not rd.is_dir():
            return None
        fin = _finish()
        full = fin.newest_checkpoint(rd / "checkpoints", fin.FULL_RE)
        if full is None:
            self.set_aside(rd, "no full state to resume from")
            it["run_dir"] = None
            return None
        return rd

    def finished(self, name: str, rc: int, todo: list[str], on_done=None):
        it = self.item(name)
        att = it["attempts"][-1]
        rd = self.find_run_dir(name) if it["kind"] not in ("stores", "speed") else None
        if it["kind"] == "anchor":
            rd = self.root / it["run_dir"] if it["run_dir"] else None
        att.update(rc=rc, t1=time.time(), run_dir=self._rel(rd) if rd else None)
        if rd is not None:
            it["run_dir"] = self._rel(rd)
        summary = self.summary_of(rd)
        self.event("item_end", item=name, rc=rc, run_dir=it["run_dir"], minutes=round((att["t1"] - att["t0"]) / 60, 1),
                   status=(summary or {}).get("status"))
        kind = it["kind"]
        if rc == 0:
            it["status"] = "done"
            it["result"] = self.result_of(it, summary)
        elif rc == EXIT_THROUGHPUT and kind in ("calib", "probe", "main", "branch", "shake"):
            it["status"] = "failed"
            self.save()
            raise ThroughputHalt(f"{name}: throughput below the trainer's floor (exit 3): a slow host")
        elif kind == "probe" and any(str((summary or {}).get("error", "")).startswith(d) for d in DIVERGED):
            it["status"], it["result"] = "done", dict(objective=None, diverged=str(summary.get("error"))[:300])
        elif kind in ("anchor", "speed"):
            it["status"] = "failed"  # a readout, never a reason to stop the box
        elif kind == "calib":
            it["status"] = "failed"  # the calibration group decides
        elif sum(1 for a in it["attempts"] if a.get("rc") not in (None, 0)) < MAX_ATTEMPTS and kind != "stores":
            # failures only: an attempt a queue restart interrupted (no rc) is no failure of the item
            it["status"] = "retry"
            todo.append(name) if name not in todo else None
            self.event("item_retry", item=name, rc=rc)
        else:
            it["status"] = "failed"
            if kind == "stores":
                self.save()
                raise QueueError(f"{name}: the store build failed (exit {rc}); see {Path(self.s.state_dir)}/logs")
        self.save()
        if it["status"] == "done" and rd is not None and kind != "stores":
            self.ensure_attribution(it, rd)
            self._pending_uploads.append(name)
        if on_done is not None and it["status"] in ("done", "failed"):
            on_done(name)

    @staticmethod
    def summary_of(rd: Path | None) -> dict | None:
        try:
            return json.loads((Path(rd) / "summary.json").read_text(encoding="utf-8")) if rd else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def result_of(it: dict, summary: dict | None) -> dict | None:
        s = summary or {}
        if it["kind"] == "probe":
            obj = (s.get("lr_probe") or {}).get("objective")
            return dict(objective=obj if isinstance(obj, (int, float)) and math.isfinite(obj) else None)
        if it["kind"] == "calib":
            return s.get("calibrate")
        return dict(steps=s.get("steps"), status=s.get("status")) if s else None

    def ensure_attribution(self, it: dict, rd: Path):
        """The CC-BY-4.0 attribution travels with a Parakeet-derived student (its dir's MODEL_CARD.md): into the run
        dir and every exported weights dir that lacks it, so the lean upload carries it."""
        run = it.get("run")
        spec = self.rules["runs"].get(run) if run else None
        if not spec or spec["family"] != "ctc":
            return
        card = self.root / spec["student"] / "MODEL_CARD.md"
        if not card.is_file():
            return
        fin = _finish()
        targets = [rd] + [p for p in (rd / "checkpoints").glob("*") if p.is_dir() and fin.WEIGHTS_RE.match(p.name)]
        for t in targets:
            if not (t / "MODEL_CARD.md").is_file():
                shutil.copyfile(card, t / "MODEL_CARD.md")

    def drain_uploads(self, one: bool = False):
        """Upload and verify the finished run dirs (lean), one per call inside the scheduler loop."""
        while self._pending_uploads:
            name = self._pending_uploads.pop(0)
            it = self.item(name)
            if self.uploader is None or not it["run_dir"]:
                it["verified"] = None
            else:
                problems = self.uploader.sync_run(self.root / it["run_dir"])
                if problems:  # once more: a transient Hub error, a commit that raced another writer
                    problems = self.uploader.sync_run(self.root / it["run_dir"])
                it["verified"] = not problems
                self.event("item_uploaded", item=name, run_dir=it["run_dir"], verified=not problems,
                           problems=problems[:20])
            if it["kind"] == "probe" and it["run_dir"]:
                self.prune_states(self.root / it["run_dir"])
            self.save()
            if one:
                return

    def prune_states(self, run_dir: Path):
        """A finished LR probe's local full states (resume points only; a probe uploads none): its result is its
        summary and metrics, and the disk is sized for the probes running at once (study_extra_gb)."""
        gone = []
        for p in sorted((run_dir / "checkpoints").glob("full_step_*")) if (run_dir / "checkpoints").is_dir() else []:
            shutil.rmtree(p, ignore_errors=True)
            gone.append(p.name)
        if gone:
            self.event("probe_states_pruned", run_dir=run_dir.name, states=gone)

    # ---------------------------------------------------------------------------------------------- phases

    def phase_stores(self):
        if self.state["stores_done"]:
            return
        configs = self.store_configs()
        for fam, cfg in configs.items():
            name = self.add_item(f"stores-{fam}", "stores", cfg, ["schedule.max_steps=20000", "optim.lr=0.0001"])
            if self.item(name)["status"] != "done":
                self.item(name)["status"] = "pending"
                self.execute([name])
        self.state["stores_done"] = True
        self.save()
        self.event("stores_built", configs=configs)

    def store_configs(self) -> dict[str, str]:
        """family -> one config of the box of that family: its data keys are every run's, so the stores it builds are
        the ones every run of that family opens."""
        runs = list(self.plan["runs"]) + list(self.plan["calibrate"])
        out = {}
        for run in runs:
            fam = self.rules["runs"][run]["family"]
            out.setdefault(fam, config_path(run))
        return out

    # calibration ------------------------------------------------------------------------------------------

    def calib_groups(self) -> list[list[tuple[str, bool]]]:
        """The calibrate list in groups of len(gpus); a short last group is filled with other runs of the list as
        unmeasured load (measured False)."""
        runs, g = list(self.plan["calibrate"]), len(self.gpus)
        groups = [runs[i:i + g] for i in range(0, len(runs), g)]
        out = []
        for grp in groups:
            members = [(r, True) for r in grp]
            fill = [r for r in runs if r not in grp]
            while len(members) < g and fill:
                members.append((fill.pop(0), False))
            out.append(members)
        return out

    def phase_calibrate(self):
        if not self.plan["calibrate"] or self.state["calibration"]:
            return
        a, b = (int(x) for x in prereg.CALIB_STEPS)
        for rnd in (1, 2):
            workers = RETRY_NUM_WORKERS if rnd == 2 else None
            table = {}
            for gi, group in enumerate(self.calib_groups()):
                table.update(self.calibrate_group(rnd, gi, group, a, b - a, workers))
            bound = {r: c["data_wait_frac"] for r, c in table.items() if c["data_wait_frac"] >= prereg.DATA_WAIT_MAX}
            self.event("calibration", round=rnd, num_workers=workers, table=table, loader_bound=bound)
            if not bound:
                self.state["calibration"] = table
                self.state["num_workers"] = workers
                self.save()
                return
            if rnd == 2:
                raise Halt(f"calibration still loader-bound after perf.num_workers {RETRY_NUM_WORKERS}: {bound}")
            log(f"loader-bound {bound}: perf.num_workers {RETRY_NUM_WORKERS} and the calibration again")

    def calibrate_group(self, rnd: int, gi: int, group: list[tuple[str, bool]], a: int, n: int,
                        workers: int | None) -> dict:
        """Run one group concurrently (two tries); -> {run: calibration entry} for its measured runs."""
        for attempt in (1, 2):
            names = {}
            for run, measured in group:
                name = f"calib-{run}" + ("" if measured else f"-load{gi}") + (f"-w{workers}" if workers else "") + (
                    f"-try{attempt}" if attempt > 1 else "")
                sets = [f"perf.num_workers={workers}"] if workers else []
                self.add_item(name, "calib", config_path(f"calib-{run}"), sets, run=run, measured=measured)
                for d in self.run_dirs_of(name):  # a restart: the group runs again as a whole, every run fresh
                    self.set_aside(d, "calibration group run again")
                self.item(name).update(status="pending", run_dir=None, result=None)
                names[name] = run
            tails, stopped, t0 = {}, [False], time.time()

            def monitor(running):
                for r in running.values():
                    if r["name"] not in tails:
                        rd = self.find_run_dir(r["name"])
                        if rd is not None:
                            tails[r["name"]] = StepTail(rd)
                if stopped[0]:
                    return
                if len(tails) == len(names):
                    wins = group_windows({k: t.poll() for k, t in tails.items()}, a, n)
                    if all(wins.values()):
                        self.stop_group(names, "windows complete")
                        stopped[0] = True
                        return
                if time.time() - t0 > self.s.calib_timeout_s:
                    self.stop_group(names, f"no complete windows after {self.s.calib_timeout_s:.0f} s")
                    stopped[0] = True
                if any(self.item(k)["status"] == "failed" for k in names):
                    self.stop_group(names, "a run of the group failed")
                    stopped[0] = True

            self.execute(list(names), monitor=monitor)
            rows = {k: read_step_rows(self.root / self.item(k)["run_dir"]) if self.item(k)["run_dir"] else {}
                    for k in names}
            wins = group_windows(rows, a, n)
            ok = all(self.item(k)["status"] == "done" for k in names) and all(wins.values())
            if ok:
                out = {}
                for k, run in names.items():
                    if not self.item(k)["measured"]:
                        continue
                    s0, s1 = wins[k]
                    st = calibration_stats(rows[k], s0, s1)
                    fixed = calibration_stats(rows[k], a, a + n)  # the literal steps (50, 250], for the record
                    res = self.item(k)["result"] or {}
                    micro = res.get("micro_audio_s") or self.rules["runs"][run]["micro_audio_s"]
                    out[run] = dict(t_step_s=st["t_step_s"], micro_audio_s=float(micro),
                                    data_wait_frac=st["data_wait_frac"], steps_measured=st["steps_measured"],
                                    window=st["window"], mean_step_s=st["mean_step_s"], workers=res.get("workers"),
                                    run_dir=self.item(k)["run_dir"],
                                    fixed_window={x: fixed[x] for x in ("window", "t_step_s", "data_wait_frac")})
                return out
            self.event("calibration_group_failed", round=rnd, group=gi, attempt=attempt,
                       status={k: self.item(k)["status"] for k in names}, windows=wins)
            for k in names:  # the group runs again as a whole: this attempt's items are not the box's failures
                self.item(k)["status"] = "superseded"
            self.save()
        raise QueueError(f"calibration group {gi} (round {rnd}) failed twice: {[r for r, _ in group]}")

    def stop_group(self, names: dict, why: str):
        for k in names:
            rd = self.find_run_dir(k)
            if rd is not None:
                (rd / "STOP").write_text(why + "\n", encoding="utf-8")
        self.event("calibration_stop", items=list(names), why=why)

    # probes -----------------------------------------------------------------------------------------------

    def probe_est(self, cls: str) -> float:
        p = self.rules["lr_probes"]["classes"][cls]
        t = ((self.state["calibration"] or {}).get(p["probed_on"]) or {}).get("t_step_s") or 1.0
        return float(p["max_steps"]) * float(t)

    def run_sets(self, run: str, calib: dict | None = None) -> list[str]:
        """The calibrated micro-batch and loader workers every trainer of `run` gets: the calibration table's (the
        numbers file's for a study run; the replicate's is the run it replicates)."""
        sets = []
        calib = calib if calib is not None else (self.state["calibration"] or {})
        c = calib.get(run) or calib.get(prereg.REPLICATE_OF if run == prereg.REPLICATE else "")
        if c and c.get("micro_audio_s"):
            sets.append(f"batch.micro_audio_s={float(c['micro_audio_s'])!r}")
        if self.state["num_workers"]:
            sets.append(f"perf.num_workers={int(self.state['num_workers'])}")
        return sets

    def probe_item(self, cls: str, lr: float) -> str:
        p = self.rules["lr_probes"]["classes"][cls]
        grid = [float(x) for x in p["grid"]]
        name = prereg.probe_run_name(cls, lr)
        on_grid = any(prereg._key(x) == prereg._key(lr) for x in grid)
        cfg = config_path(name if on_grid else prereg.probe_run_name(cls, grid[0]))
        sets = self.run_sets(p["probed_on"]) + ([] if on_grid else [f"optim.lr={float(lr)!r}"])
        return self.add_item(name, "probe", cfg, sets, run=p["probed_on"])

    def probe_results(self, cls: str) -> dict:
        """{lr: objective} of the class's finished probes (None: diverged, it loses)."""
        out = {}
        for name, it in self.state["items"].items():
            if it["kind"] == "probe" and it["status"] == "done" and name.startswith(f"probe-{cls}-") and \
                    name[len(f"probe-{cls}-"):] and self._probe_cls(name) == cls:
                lr = float(name[len(f"probe-{cls}-"):])
                out[lr] = (it["result"] or {}).get("objective")
        return out

    def _probe_cls(self, name: str) -> str | None:
        """The class of probe-<class>-<lr> (class names contain dashes: kept-t03)."""
        for cls in sorted(self.rules["lr_probes"]["classes"], key=len, reverse=True):
            if name.startswith(f"probe-{cls}-"):
                return cls
        return None

    def phase_probes(self):
        classes = list(self.plan["probe_classes"])
        if not classes or self.state["lr_choice"]:
            return
        todo = []
        for cls in sorted(classes, key=self.probe_est, reverse=True):  # longest first: LPT packing
            for lr in self.rules["lr_probes"]["classes"][cls]["grid"]:
                name = self.probe_item(cls, float(lr))
                if self.item(name)["status"] != "done":
                    todo.append(name)
        decided = {}

        def decide(cls: str):
            res = self.probe_results(cls)
            grid = [prereg._key(x) for x in self.rules["lr_probes"]["classes"][cls]["grid"]]
            if any(g not in {prereg._key(x) for x in res} for g in grid):
                return  # its grid is not in yet
            try:
                choice = prereg.choose_lr({cls: {lr: (math.inf if v is None else v) for lr, v in res.items()}})[cls]
            except ValueError as e:
                raise QueueError(f"probe class {cls}: {e}") from e
            self.state["probes"][cls] = {prereg.lr_tag(lr): v for lr, v in sorted(res.items())}
            if choice["decision"] == "extend":
                name = self.probe_item(cls, choice["next_lr"])
                if self.item(name)["status"] != "done" and name not in todo:
                    todo.append(name)
                self.event("lr_probe_extend", cls=cls, winner=choice["winner"], next_lr=choice["next_lr"])
            elif choice["decision"] == "halt":
                self.event("lr_probe_halt", cls=cls, results=self.state["probes"][cls])
                raise Halt(f"LR probe class {cls}: the winner {prereg.lr_tag(choice['winner'])} is again at a grid edge "
                           f"after its one extension ({self.state['probes'][cls]})")
            else:
                decided[cls] = choice
                self.event("lr_chosen", cls=cls, lr=choice["lr"], results=self.state["probes"][cls])
            self.save()

        def on_done(name: str):
            it = self.item(name)
            if it["kind"] == "probe":
                if it["status"] == "failed":
                    raise QueueError(f"{name} failed twice: the probe has no objective")
                decide(self._probe_cls(name))

        for cls in classes:  # a restart: classes whose probes are all in already
            if cls not in decided:
                decide(cls)
        self.execute(todo, on_done=on_done, filler=self.anchor_filler)
        for cls in classes:
            if cls not in decided:
                decide(cls)
        if missing := [c for c in classes if c not in decided]:
            raise QueueError(f"LR probe classes without a chosen LR: {missing}")
        self.state["lr_choice"] = {c: dict(decided[c], tested=[prereg.lr_tag(x) for x in decided[c]["tested"]])
                                   for c in classes}
        self.save()

    # numbers ----------------------------------------------------------------------------------------------

    def numbers_path(self) -> Path:
        return self.root / NUMBERS_DIR / self.plan["numbers_file"]

    def phase_numbers(self):
        """The box's PREREG_numbers file: written once, uploaded, verified and its sha256 logged, before the wave."""
        num = self.state["numbers"]
        path, remote = self.numbers_path(), f"{NUMBERS_DIR}/{self.plan['numbers_file']}"
        if not (num and num.get("sha256")):
            if self.uploader is not None and self.uploader.exists(remote):
                raise Halt(f"{remote} is already in the runs repo, but this box did not write it: a box writes its "
                           f"numbers once, before its first study step (relaunch decisions are the owner's)")
            try:
                sha, numbers = self.write_numbers(path)
            except ValueError as e:
                raise Halt(f"the PREREG numbers were refused: {e}") from e
            num = self.state["numbers"] = dict(path=self._rel(path), sha256=sha, remote=remote, uploaded=False,
                                               max_steps=numbers["max_steps"], lr=numbers["lr"],
                                               calibration=numbers.get("calibration"), written=time.time())
            self.save()
            self.event("prereg_numbers_written", path=num["path"], sha256=sha)
        elif prereg.file_sha256(path) != num["sha256"]:
            raise Halt(f"{path} changed since it was written (sha256 {num['sha256'][:12]}...): the numbers are fixed")
        if not num["uploaded"]:
            problems = self.uploader.put_file(path, remote) if self.uploader is not None else []
            if problems:
                problems = self.uploader.put_file(path, remote)
            if problems:
                raise QueueError(f"{remote}: not verified on the Hub: {problems}")
            num.update(uploaded=True, verified=self.uploader is not None, uploaded_wall=time.time())
            self.save()
            # the pre-registration's record: this line comes before the first study step
            self.event("prereg_numbers", sha256=num["sha256"], path=num["path"], remote=remote,
                       verified=num["verified"])

    @staticmethod
    def _per_box(fn) -> bool:
        """The rules' per-box form of a numbers function (kitsune.prereg with the box plans: a `box` parameter)."""
        return "box" in inspect.signature(fn).parameters

    def write_numbers(self, path: Path) -> tuple[str, dict]:
        """The box's numbers by the rules' own functions: max_steps and the LR choice from the calibration table and
        the probes, prereg.write_numbers for the file (per box when the rules have box plans)."""
        if self.plan.get("numbers_from"):
            return self.derived_numbers(path)
        calib = {run: {k: c[k] for k in prereg.CALIB_KEYS} for run, c in self.state["calibration"].items()}
        box_kw = {"box": self.box} if self._per_box(prereg.write_numbers) else {}
        steps = prereg.max_steps(calib, t_ref_run=self.plan["reference"],
                                 **(box_kw if self._per_box(prereg.max_steps) else {}))
        probes = {cls: {float(lr): (math.inf if v is None else v) for lr, v in res.items()}
                  for cls, res in self.state["probes"].items()}
        choices = self.state["lr_choice"] or {}
        lrs = {run: float(choices[self.rules["runs"][run]["lr_from"]]["lr"]) for run in self.plan["runs"]}
        sha = prereg.write_numbers(path, calib, probes, lrs, steps, rules_path=self.rules_path, host=self.s.host,
                                   allow_pending=self.s.allow_pending, **box_kw)
        return sha, json.loads(path.read_text(encoding="utf-8"))

    def derived_numbers(self, path: Path) -> tuple[str, dict]:
        """A box with numbers_from (the replicate): its runs take the max_steps and LR of the run they replicate from
        that box's uploaded numbers file (prereg.REPLICATE_OF for the replicate) - by prereg.replicate_numbers and
        prereg.write_numbers(box=..., numbers_from=...) when the rules have them, else written here with the source's
        name and sha256 in the canonical form of prereg.write_numbers. The source's calibration comes along for the
        fill (the micro-batch the replicated run was measured at)."""
        src = self.rules["boxes"][self.plan["numbers_from"]]["numbers_file"]
        remote = f"{NUMBERS_DIR}/{src}"
        if self.uploader is None:
            raise ValueError(f"{remote}: no runs repo to read it from")
        local = self.uploader.download(remote, Path(self.s.state_dir) / "numbers_from")
        data = local.read_bytes()
        srcn = json.loads(data)
        if "numbers_from" in inspect.signature(prereg.write_numbers).parameters and \
                hasattr(prereg, "replicate_numbers"):
            steps, lrs, _ = prereg.replicate_numbers(local)
            sha = prereg.write_numbers(path, {}, {}, lrs, steps, box=self.box, numbers_from=local,
                                       rules_path=self.rules_path, host=self.s.host, allow_pending=self.s.allow_pending)
            numbers = json.loads(path.read_text(encoding="utf-8"))
            return sha, dict(numbers, calibration=numbers.get("calibration") or srcn.get("calibration") or {})
        steps, lrs = {}, {}
        for run in self.plan["runs"]:
            of = prereg.REPLICATE_OF if run == prereg.REPLICATE else run
            steps[run] = int(srcn["max_steps"].get(run) or srcn["max_steps"][of])
            lrs[run] = float(srcn["lr"].get(run) or srcn["lr"][of])
        calib = {run: srcn["calibration"][prereg.REPLICATE_OF if run == prereg.REPLICATE else run]
                 for run in self.plan["runs"] if (prereg.REPLICATE_OF if run == prereg.REPLICATE else run)
                 in (srcn.get("calibration") or {})}
        if not self.s.allow_pending and (left := prereg.pending(json.loads(self.rules_path.read_text(encoding="utf-8")))):
            raise ValueError(f"{self.rules_path} still has {len(left)} pending field(s), e.g. {left[:3]}")
        import hashlib
        import platform

        numbers = {"numbers_from": {"box": self.plan["numbers_from"], "file": remote,
                                    "sha256": hashlib.sha256(data).hexdigest()},
                   "max_steps": dict(sorted(steps.items())), "lr": dict(sorted(lrs.items())),
                   "calibration": calib, "rules_sha256": prereg.rules_sha256(self.rules_path),
                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "host": self.s.host if self.s.host is not None else platform.node()}
        out = (json.dumps(numbers, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
        prereg._write_bytes(Path(path), out)
        return hashlib.sha256(out).hexdigest(), numbers

    # the wave ---------------------------------------------------------------------------------------------

    def fill_sets(self, run: str) -> list[str]:
        """What the queue sets on a study run's config (and its branch's, identically): max_steps and LR from the
        numbers file, the mini-eval cadence (prereg's fraction of max_steps), the calibrated micro-batch and workers."""
        num = self.state["numbers"]
        n, lr = int(num["max_steps"][run]), float(num["lr"][run])
        mini = max(1, int(round(float(self.rules["evals"]["mini_every_frac"]) * n)))
        sets = [f"schedule.max_steps={n}", f"optim.lr={lr!r}", f"eval.mini.every_steps={mini}"]
        return sets + self.run_sets(run, num.get("calibration") or self.state["calibration"] or {})

    def phase_wave(self):
        todo = []
        for i, run in enumerate(self.plan["runs"]):
            gpu = self.gpus[i % len(self.gpus)]
            name = self.add_item(run, "main", config_path(run), self.fill_sets(run), run=run, affinity=gpu)
            half = f"{run}-half"
            self.add_item(half, "branch", config_path(half), [], run=run, after=run, affinity=gpu)
            todo += [name, half]

        def on_done(name: str):
            it = self.item(name)
            if it["kind"] == "main" and it["status"] == "done":
                half = self.item(f"{name}-half")
                half["sets"] = self.fill_sets(it["run"]) + [f"branch.parent={it['run_dir']}"]
                half["affinity"] = it["attempts"][-1]["gpu"]
                self.save()

        for name in todo:  # a restart: a main that is done already gives its branch its parent
            if self.item(name)["kind"] == "main" and self.item(name)["status"] == "done":
                on_done(name)
        self.execute([n for n in todo if self.item(n)["status"] != "done"], on_done=on_done,
                     filler=self.anchor_filler, stagger=True)

    # extras -----------------------------------------------------------------------------------------------

    def anchor_filler(self, gpu: str) -> str | None:
        """The anchor re-score in a GPU gap: once, on a GPU nothing else wants."""
        if "anchor" not in self.plan["extras"]:
            return None
        name = self.add_item("anchor", "anchor", ANCHOR_CONFIG, [])
        return name if self.item(name)["status"] in ("pending", "interrupted") else None

    def anchor_argv(self, name: str) -> list[str]:
        """`anchor`: fetch the first run's 0.6B from the runs repo, then scripts/05_evaluate.py on the study's eval rows
        into a run dir of its own (runs/anchor-b20-<stamp>/: uploaded and verified like a run), in a process of its own
        so the scheduler never waits for the download."""
        it = self.item(name)
        out = self.root / "runs" / f"anchor-b20-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        it["run_dir"] = self._rel(out)
        return self._cmd("anchor") + ["--dest", str(Path(self.s.state_dir) / "anchor"), "--config", it["config"],
                                      "--out", str(out), "--cache-dir", str(self.root / "cache"),
                                      *(["--out-repo", self.s.out_repo] if self.s.out_repo else []),
                                      "--eval-cmd", json.dumps(self._cmd("eval"))]

    def phase_extras(self):
        if "anchor" in self.plan["extras"] and self.add_item("anchor", "anchor", ANCHOR_CONFIG) and \
                self.item("anchor")["status"] in ("pending", "interrupted"):
            self.execute(["anchor"])
        if "speed" in self.plan["extras"]:
            self.phase_speed()

    def phase_speed(self):
        """The speed probe of every study student and both teachers, one model at a time on an idle host (nothing else
        runs by now)."""
        tool = self.root / SPEED_TOOL
        if not tool.is_file() and self.s.speed_cmd is None:
            self.event("speed_skipped", why=f"{SPEED_TOOL} is not in this checkout (WP5b)")
            return
        out = self.root / "runs" / f"speed-{self.box}"
        out.mkdir(parents=True, exist_ok=True)
        # a student's speed is its shape's: the init dir every box pulls stands for the trained weights
        systems = [(run, spec["family"], spec["student"]) for run, spec in self.rules["runs"].items()]
        systems += [(k, kind, path) for k, (kind, path) in TEACHERS.items()]
        for sys_name, kind, path in systems:
            args = ["--kind", kind, *(["--model", path] if path else []), "--system", sys_name,
                    "--store", str(self.root / "cache" / "eval"), "--per-set", str(SPEED_PER_SET),
                    "--out", str(out / "speed.json"), "--require-idle"]
            name = self.add_item(f"speed-{sys_name}", "speed", None, args)
            if self.item(name)["status"] != "done":
                self.execute([name])
        with open(out / "events.jsonl", "a", encoding="utf-8") as f:  # makes it a run dir for finish.py's uploads
            for sys_name, _, _ in systems:
                it = self.item(f"speed-{sys_name}")
                f.write(json.dumps({"kind": "speed", "system": sys_name, "status": it["status"],
                                    "wall": time.time()}) + "\n")
        self.state["items"].setdefault("speed", dict(kind="speed-dir", config=None, sets=[], after=None, run=None,
                                                     affinity=None, env={}, measured=False, status="done",
                                                     attempts=[], run_dir=self._rel(out), verified=None, result=None))
        self._pending_uploads.append("speed")
        self.save()

    # shakedown --------------------------------------------------------------------------------------------

    def phase_shakedown(self):
        """STUDY.md 6.1, one item after the other on the one GPU; every run dir uploaded and verified."""
        todo = []
        for name in self.plan["items"]:
            kind, after, env, run = "shake", None, {}, None
            if name == "shake-parent-half":
                kind, after = "branch", "shake-parent"
            if name == "shake-resume":
                env = {"KITSUNE_CRASH_AT_STEP": "45"}  # a crash before step 45; the resume starts from step 40
            if name.startswith("shake-smoke-"):
                run = name[len("shake-smoke-"):]
            self.add_item(name, kind, config_path(name), [], after=after, env=env, run=run)
            todo.append(name)

        def on_done(name: str):
            if name == "shake-parent" and self.item(name)["status"] == "done":
                self.item("shake-parent-half")["sets"] = [f"branch.parent={self.item(name)['run_dir']}"]
                self.save()

        on_done("shake-parent")
        self.execute([n for n in todo if self.item(n)["status"] != "done"], on_done=on_done)
        resumed = self.item("shake-resume")
        if resumed["status"] == "done" and not any(a.get("resume") for a in resumed["attempts"]):
            raise QueueError("shake-resume finished without the crash and resume it is there to exercise")

    # summary ----------------------------------------------------------------------------------------------

    def write_summary(self, status: str, reason: str | None, rc: int):
        items = {n: dict(kind=it["kind"], status=it["status"], run_dir=it["run_dir"], verified=it["verified"],
                         attempts=len(it["attempts"]), result=it["result"]) for n, it in self.state["items"].items()}
        summary = dict(box=self.box, status=status, reason=reason, rc=rc, gpus=self.gpus, items=items,
                       calibration=self.state["calibration"], num_workers=self.state["num_workers"],
                       probes=self.state["probes"], lr_choice=self.state["lr_choice"], numbers=self.state["numbers"],
                       abandoned=self.state["abandoned"], started=self.state["started"], ended=time.time())
        path = Path(self.s.state_dir) / SUMMARY_FILE
        _atomic_json(path, summary)
        if self.uploader is not None:
            try:
                self.uploader.put_file(path, f"{NUMBERS_DIR}/box-{self.box}/{SUMMARY_FILE}")
            except Exception as e:  # noqa: BLE001  finish.py uploads it with the infra logs anyway
                log(f"queue summary upload failed: {type(e).__name__}: {e}")


# ====================================================================================================== commands


def plan_items(box: str, rules: dict | None = None) -> list[dict]:
    """What `run` would register, in order (the `plan` command; nothing is started): stores, calibration groups,
    probes (the pre-registered grids; extensions only as needed), numbers, the wave, the extras."""
    r = rules if rules is not None else prereg.rules()
    plan = box_plan(box, r)
    out = [dict(phase="stores", config=config_path(plan["runs"][0] if plan["runs"] else plan["calibrate"][0]))]
    if plan.get("shakedown"):
        return out + [dict(phase="shakedown", item=n, config=config_path(n)) for n in plan["items"]]
    q = Queue.__new__(Queue)
    q.plan, q.gpus = plan, ["0", "1", "2", "3"] if box in ("A", "B") else ["0"]
    for gi, group in enumerate(Queue.calib_groups(q)):
        out.append(dict(phase="calibrate", group=gi, runs=[run for run, m in group if m],
                        load=[run for run, m in group if not m]))
    for cls in plan["probe_classes"]:
        for lr in r["lr_probes"]["classes"][cls]["grid"]:
            out.append(dict(phase="probe", item=prereg.probe_run_name(cls, lr), cls=cls, lr=float(lr)))
    out.append(dict(phase="numbers", file=f"{NUMBERS_DIR}/{plan['numbers_file']}", numbers_from=plan["numbers_from"]))
    for run in plan["runs"]:
        out += [dict(phase="wave", item=run, config=config_path(run)),
                dict(phase="wave", item=f"{run}-half", config=config_path(f"{run}-half"), after=run)]
    out += [dict(phase="extras", item=x) for x in plan["extras"]]
    return out


def _load_trainer():
    spec = importlib.util.spec_from_file_location("kitsune_script_04_distill", REPO / "scripts" / "04_distill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_stores(config: str, sets: list[str]) -> int:
    """The label stores of `config`'s data keys and family, exactly as the trainer's setup_data builds them (its
    build_train_store and build_eval_store: a token store, or with the CTC trainer a frame store and its frame
    preflight; before that trainer, train_store_spec + trainset.build_stores): the box builds them once, alone, and
    every run then reuses them (their fingerprint matches)."""
    D = _load_trainer()
    from kitsune import trainset

    cfg = D.load_config(config, sets)

    class _Log:
        @staticmethod
        def event(kind, **fields):
            print(json.dumps({"event": kind, **fields}, default=str)[:2000], flush=True)

    t0 = time.time()
    builder = getattr(D, "build_train_store", None)  # the trainer's own, of the config's family (the CTC trainer's)
    if builder is not None:
        train = builder(cfg, _Log)
    elif cfg.get("family", "aed") == "ctc":
        print("a CTC store needs the CTC trainer's build_train_store (WP4b): not in this checkout", flush=True)
        return 2
    else:
        name, ids = D.train_store_spec(cfg, _Log)
        train = trainset.build_stores(D.rpath(cfg["selection"]), D.rpath(cfg["data_root"]),
                                      D.rpath(cfg["teacher_root"]), D.rpath(cfg["cache_dir"]) / name, cfg["sources"],
                                      ["train"], ids=ids, log=print)
    ev = D.build_eval_store(cfg, _Log)
    print(f"stores: train {len(train)} utts ({train.hours:.1f} h), eval {len(ev)} utts, {time.time() - t0:.0f} s",
          flush=True)
    return 0


def anchor(args) -> int:
    """The anchor item (decision 29): the first run's 0.6B from the runs repo (ANCHOR_CKPT, ~1.2 GB, once), then the
    evaluator on the study's eval rows (the anchor config's data keys) into --out, the stores in the box's cache."""
    ckpt = Path(args.dest) / ANCHOR_CKPT
    if not (ckpt / "config.json").is_file():
        if not args.out_repo:
            log("anchor: no runs repo to fetch the checkpoint from")
            return 2
        HubUploader(args.out_repo).download_dir(ANCHOR_CKPT, Path(args.dest))
    argv = json.loads(args.eval_cmd) + ["--config", args.config, "--ckpt", str(ckpt), "--out", args.out,
                                        "--cache-dir", args.cache_dir, "--max-temp", "0"]
    log(f"anchor: {' '.join(argv)}")
    return subprocess.run(argv, cwd=str(REPO)).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "plan", "students"):
        p = sub.add_parser(c)
        p.add_argument("--box", default=os.environ.get("KITSUNE_BOX"), choices=BOXES)
    sub.choices["run"].add_argument("--gpus", default=None, help="comma list of CUDA_VISIBLE_DEVICES values")
    b = sub.add_parser("build-stores")
    b.add_argument("--config", required=True)
    b.add_argument("--set", action="append", default=[])
    an = sub.add_parser("anchor")
    for k in ("--dest", "--config", "--out", "--cache-dir", "--eval-cmd"):
        an.add_argument(k, required=True)
    an.add_argument("--out-repo", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build-stores":
        return build_stores(args.config, args.set)
    if args.cmd == "anchor":
        return anchor(args)
    if not args.box:
        ap.error("--box (or KITSUNE_BOX) is required")
    try:
        if args.cmd == "students":
            print("\n".join(box_students(args.box)))
            return 0
        if args.cmd == "plan":
            for row in plan_items(args.box):
                print(json.dumps(row))
            return 0
        gpus = [g for g in args.gpus.split(",") if g] if args.gpus else None
        return Queue(args.box, Settings(gpus=gpus)).run()
    except QueueError as e:
        log(f"refused: {e}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
