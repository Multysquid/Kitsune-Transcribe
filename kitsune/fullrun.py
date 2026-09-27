"""The full-data runs' shared core: names, paths, the box registry and the small helpers every full-run package uses.

The full runs (plan v3) train P-0.1B alone on box `p01` (1x RTX 5090, Parakeet labels only), then T-0.6B, P-0.3B and
P-0.05B on box `full` (2x RTX 5090, one shared GPU queue), after two short smokes (`full-smoke` = smoke A, `smoke-b`).
The selection (scripts/make_selection.py full mode, kitsune/devslice.py), the trainer (scripts/04_distill.py), the
box queue (kitsune/full_queue.py), the vast scripts (vast/launch.py, bootstrap.sh, finish.py) and the evaluators all
import their shared names from here, so a constant cannot drift between them. The binding definitions are the full-run
build contract (sections 1 and 2); this module is its code.

  constants    the box names, the job name, the box state files under $KITSUNE_STATE, the Hub layout of the queue
               summaries and of the timed states in the scratch repo, the frozen eval manifest, the full and smoke
               selections, the dev split, the full_study recipe block and the data blocks of the full and smoke runs,
               the item kinds, statuses, stall limits and fault actions, and one ENV_* name per environment variable
  helpers      shard_split (a dev row's audio and labels live in its train shard), full_recipe_problems (the
               recipe block a full selection must carry), seeded_subset (make_selection's seeded draw), dev_pick (the
               scored dev ids: per source a seeded draw of per_source kept dev rows, in selection order), the resume
               flags' parsers (parse_resume_sets / parse_resume_reset: launch -> env -> full_queue resume-pull),
               run_id_of, the Hub and scratch paths, pointer_problems (the timed-state pointer, format 1)
  registry     configs/full/boxes.json, hand-written next to the generated configs/full/*.json (make_full_configs.py
               --check validates it with registry_problems). One source of truth: launch reads the hours, price, GPU
               count and watchdog from it, bootstrap the students and extra files, the queue its items.
               load_registry validates it and fills the defaults; box_* read one box; the CLI serves bootstrap

Registry (contract 2.3). Top level {"version": 1, "boxes": {<box>: <box spec>}} with box names from BOX_NAMES; any key
starting with "_" is a comment, anywhere. Paths (data_config, item config) are repo-relative POSIX paths resolved
against `root` (the checkout; default the one this file is in). A box spec:

  gpus int >= 1; data_config (under configs/full/); est_hours <= max_hours, max_dph (> 0; launch's defaults);
  deadline_reserve_min >= 0 (the queue's per-item KITSUNE_DEADLINE = box deadline - this); watchdog {orphan_s int
  >= 0, action stop|alert}; extra_gb >= 0 (0); timed_states (false: the scratch repo is required when true); gate
  (true: the download gate); smoke (false: writes smoke_verdict.json, allows faults and verdict specs);
  max_attempts >= 1 (4); extra_files, extra_dirs (data-repo paths the box pulls, []); faults ([]); items (non-empty)

Items run in registry order where the queue allows. Every item: name (ITEM_RE, unique in its box), kind (ITEM_KINDS),
needs (EARLIER items of the box; filled with the implicit ones below), stall_min (absent: STALL_MIN_DEFAULT[kind];
a number > 0: that limit in minutes; an explicit null: no stall check, the queue's overrun kill after OVERRUN_FACTOR x
max_hours instead), max_hours (> 0; required for train and for a null stall_min, else optional, filled None),
droppable (train: false by default; every other kind must be, and is by default, true), verdict (specs; smoke boxes).
Per kind:

  stores   config, eval_only (false), sets ([] of "key=value": the build's --set values)
  train    config, study_run (a kitsune.prereg run: the student's registered build), family (aed|ctc, the config's),
           plan_total_steps / plan_hours (both or neither; the matching full run's plan, for smoke check 3; null)
  readout  of (an earlier train item of the box)
  speed    system, speed_kind (tools/speed_probe.py --kind), at most one model source (none: cohere), args ([]),
           only_if_new_machine (another registry box, or null)
  eval     argv (a template), model sources

Model sources: `of` (an earlier train item of this box: its run dir's config and final checkpoint), `of_box` + `of` (a
train item of another registry box, resolved from that box's Hub queue summary), `weights` (a list of {name, run_id,
step}: fixed runs-repo runs), `model` (a data-repo dir listed in extra_dirs). A same-box `of` is added to needs (the
checkpoint exists only once its training is done), and so is every item an argv names as {out:<item>}. Argv and
verdict json templates may use only PLACEHOLDERS, {out:<earlier item>}, and {config:<name>} / {ckpt:<name>} of the
item's own weights; {config}, {ckpt}, {run_id} and {run_name} need `of`. The queue substitutes them (kitsune/
full_queue.py). Verdict spec {check "1".."16", json, path "a.b.c", min, max, equals}: json and path go together, a
condition needs a path. Fault {id, action (FAULT_ACTIONS), item (a train item of the box), at_step, after_event,
min_attempt (1), seconds}: only on smoke boxes; sigstop/kill need at_step, wipe_run_dir after_event, deadline seconds,
freeze_controller_hb at_step and seconds > watchdog.orphan_s on a box whose watchdog only alerts. Item configs must
exist (check_files) and carry the box data config's DATA_KEYS values (a missing pull_parakeet counts as false); a train
item's family is its config's (default aed). Every name and path the box env or a command line carries is one
env-string word (vast/launch.py env_string).

CLI (vast/bootstrap.sh; exit 0 ok, 2 refused - a bad registry, an unknown box, a student that is not the registered
build):
  python -m kitsune.fullrun students --box p01 [--root R]         # the train items' student dirs, one per line
  python -m kitsune.fullrun extra-files --box p01                 # the box's extra data-repo files, one per line
  python -m kitsune.fullrun extra-dirs --box full-smoke           # its extra data-repo dirs
  python -m kitsune.fullrun check-students --box p01 --root R     # every pulled student is the registered build
  python -m kitsune.fullrun show --box full                       # the box spec with defaults, env and configs (JSON)
The registry is $KITSUNE_FULL_REGISTRY when set (tests), else <root>/configs/full/boxes.json.

Stdlib only at import: numpy (seeded_subset) and kitsune.prereg (the registered study runs: study_run, the student
checks) are imported inside the functions that need them, so the vast scripts can import this with the system Python.
"""
import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------------ names

JOB = "full"  # KITSUNE_JOB=full
BOXES_FILE, ENV_REGISTRY = "configs/full/boxes.json", "KITSUNE_FULL_REGISTRY"
BOX_NAMES = ("full-smoke", "p01", "full", "smoke-b")  # smoke A, box 1, box 2, smoke B
HUB_DIR = "full"  # runs repo: full/box-<box>/{queue_summary.json, smoke_verdict.json, infra/<container id>/}
STATE_DEFAULT = "/workspace/kitsune_state"  # $KITSUNE_STATE on a box

# the box state files under $KITSUNE_STATE (train_hb: the controllers only; hb/<item>: the child, via KITSUNE_HEARTBEAT)
TRAIN_HB, HB_DIR, RESUME_PLAN, VERDICT_FILE = "train_hb", "hb", "resume_plan.json", "smoke_verdict.json"
ALERTS_FILE, GATE_FILE, SUMMARY_FILE, DEADLINE_FILE = ("watchdog_alerts.jsonl", "download_gate.json",
                                                       "queue_summary.json", "deadline")

# timed full states in the scratch repo: runs/<run_id>/checkpoints/full_step_<N>/ (exactly one per run) and the pointer
# runs/<run_id>/timed_state.json. A local timed state carries SCRATCH_MARK while its upload is pending - never the
# runs-repo UPLOAD_MARK (.upload_pending), which finish.py would treat as a state the runs repo must hold
TIMED_POINTER, SCRATCH_MARK, TIMED_REASON, POINTER_FORMAT = "timed_state.json", ".scratch_pending", "timed", 1
STATE_FILES_REQUIRED = ("model.pt", "optimizer.pt", "l2sp.pt", "trainer.pt", "trainer.json")
STATE_FILES_OPTIONAL = ("aux_ctc.pt",)
_UPLOAD_MARK = ".upload_pending"  # scripts/04_distill.py UPLOAD_MARK (it imports torch, so it is not imported here)

# the frozen study eval manifest (data repo @ 4b0d801): every full readout scores it (--manifest), and the full and
# smoke selections' eval rows equal its ids per set, in order
FROZEN_MANIFEST = "labels/full/selections/study_manifest.json"
FROZEN_MANIFEST_SHA256 = "ef56dec2b69bfe54f96a5ad1df36bde38255b68799873ab5f12346da78ac3796"
FULL_DIR = "labels/full/selections/full_study"  # write-once, next to (never over) the frozen study files
FULL_SELECTION, SMOKE_SELECTION = FULL_DIR + "/full.parquet", FULL_DIR + "/smoke.parquet"

# the dev slice: rows of whole train shards (videos for Emilia) marked split "dev", keep True, labels in their train
# stems; the trainer's early stop reads a seeded pick of them (dev_pick), never the test sets
DEV_SPLIT, SPLITS = "dev", ("train", "dev", "eval")
DEV_RULE = 1  # the dev-slice rule version the selection records (selection_recipe.full_study.dev_rule)
# the recipe block of a full selection (config selection_recipe.full_study): the study's F1a threshold, dedup length
# and probe size (kitsune.prereg.STUDY_SELECTION; tests assert they are equal), no draw (every kept hour), dev rule 1
FULL_STUDY = {"f1a_max": 0.5, "dedup_min_chars": 15, "probe_n": 300, "draw_audio_s": None, "dev_rule": DEV_RULE}
SMOKE_DRAW_AUDIO_S = 360000  # the smoke selection's seeded 100 h train draw on the study extent
SMOKE_STUDY = dict(FULL_STUDY, draw_audio_s=SMOKE_DRAW_AUDIO_S)
# the data blocks of the full runs (all the labels) and of the smokes (the study extent): scripts/make_selection.py
# builds the selections from them; configs/full/data-*.json (make_full_configs) carry them (+ family, pull_parakeet)
FULL_DATA = {"data_root": "data", "teacher_root": "labels/full/teacher_out", "second_root": "labels/full/second_out",
             "parakeet_root": "labels/full/parakeet_out", "selection": FULL_SELECTION,
             "extent": {"name": "full", "root": "labels/full", "inputs": {}},
             "sources": ["reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame"],
             "eval_sets": ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"],
             "selection_recipe": {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "emilia_nc=0.2"],
                                  "filter_eval_sets": [], "partial_second_opinion": [], "study": None,
                                  "full_study": dict(FULL_STUDY)}}
SMOKE_DATA = copy.deepcopy(FULL_DATA)
SMOKE_DATA["selection"] = SMOKE_SELECTION
SMOKE_DATA["extent"]["inputs"] = {"reazon_large": 53, "emilia_yodas": "300h", "emilia_nc": 8, "galgame": 3}
SMOKE_DATA["selection_recipe"]["full_study"] = dict(SMOKE_STUDY)
# the keys a box's item configs share with its data config (registry check)
DATA_KEYS = ("data_root", "teacher_root", "second_root", "parakeet_root", "selection", "extent", "sources", "eval_sets",
             "selection_recipe", "pull_parakeet")

RUN_ID_RE = r"^[a-z0-9][a-z0-9.-]*-\d{8}T\d{6}Z(?:-\d+)?$"  # <item>-<YYYYMMDDTHHMMSSZ>[-n], 04 build's run dir name
ITEM_RE = r"^[a-z0-9][a-z0-9.-]*$"
RESUME_SET_KEYS = ("schedule.epochs",)  # the only config key launch --resume-set may change

# ------------------------------------------------------------------------------------------------ environment (1.4)

ENV_JOB, ENV_BOX, ENV_N_GPUS, ENV_CONFIG = "KITSUNE_JOB", "KITSUNE_BOX", "KITSUNE_N_GPUS", "KITSUNE_CONFIG"
ENV_SHA, ENV_DATA_REPO, ENV_DATA_REVISION = "KITSUNE_SHA", "KITSUNE_DATA_REPO", "KITSUNE_DATA_REVISION"
ENV_OUT_REPO, ENV_MAX_HOURS, ENV_DPH, ENV_TZ = "KITSUNE_OUT_REPO", "KITSUNE_MAX_HOURS", "KITSUNE_DPH", "TZ"
ENV_REBUILD_TIMEOUT_MIN, ENV_PULL_TIMEOUT_MIN = "KITSUNE_REBUILD_TIMEOUT_MIN", "KITSUNE_PULL_TIMEOUT_MIN"
ENV_SCRATCH_REPO, ENV_MACHINE_ID = "KITSUNE_SCRATCH_REPO", "KITSUNE_MACHINE_ID"
ENV_GATE_BYTES, ENV_GATE_MAX_H = "KITSUNE_GATE_BYTES", "KITSUNE_GATE_MAX_H"
ENV_REBUILD_BYTES, ENV_PULL_BYTES = "KITSUNE_REBUILD_BYTES", "KITSUNE_PULL_BYTES"
ENV_WATCHDOG_HB_FILE, ENV_WATCHDOG_ORPHAN_S = "KITSUNE_WATCHDOG_HB_FILE", "KITSUNE_WATCHDOG_ORPHAN_S"
ENV_WATCHDOG_ORPHAN_ACTION = "KITSUNE_WATCHDOG_ORPHAN_ACTION"
ENV_RESUME, ENV_RESUME_RESET, ENV_RESUME_SETS = "KITSUNE_RESUME", "KITSUNE_RESUME_RESET", "KITSUNE_RESUME_SETS"
ENV_CPU_QUOTA, ENV_THREADS_PER_GPU = "KITSUNE_CPU_QUOTA", "KITSUNE_THREADS_PER_GPU"
ENV_OMP_NUM_THREADS, ENV_OPENBLAS_NUM_THREADS = "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"
ENV_MKL_NUM_THREADS, ENV_NUMEXPR_NUM_THREADS = "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
ENV_RAYON_NUM_THREADS, ENV_TOKIO_WORKER_THREADS = "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"
# the six thread pools onstart caps per GPU (fix 2), in the order the threads event lists them
ENV_THREAD_POOLS = (ENV_OMP_NUM_THREADS, ENV_OPENBLAS_NUM_THREADS, ENV_MKL_NUM_THREADS, ENV_NUMEXPR_NUM_THREADS,
                    ENV_RAYON_NUM_THREADS, ENV_TOKIO_WORKER_THREADS)
ENV_CGROUP, ENV_HEARTBEAT, ENV_DEADLINE = "KITSUNE_CGROUP", "KITSUNE_HEARTBEAT", "KITSUNE_DEADLINE"
ENV_QUEUE_ITEM, ENV_CUDA_VISIBLE_DEVICES = "KITSUNE_QUEUE_ITEM", "CUDA_VISIBLE_DEVICES"
ENV_STATE = "KITSUNE_STATE"

# ------------------------------------------------------------------------------------------------ items

ITEM_KINDS = ("stores", "train", "readout", "speed", "eval")
ITEM_STATUSES = ("pending", "running", "interrupted", "retry", "done", "failed", "skipped", "not_needed")
# minutes without a beat on $STATE/hb/<item> before the queue kills and retries an item (DECISIONS C4)
STALL_MIN_DEFAULT = {"stores": 360, "train": 45, "readout": 30, "speed": 30, "eval": 60}
OVERRUN_FACTOR = 2.0  # an item with stall_min null is killed after this x its max_hours of wall time
FAULT_ACTIONS = ("sigstop", "kill", "wipe_run_dir", "deadline", "freeze_controller_hb")
# the argv / verdict-json placeholders the queue fills; plus {out:<item>}, {config:<name>}, {ckpt:<name>}
PLACEHOLDERS = ("python", "root", "state", "box", "cache_dir", "hf_cache", "hub_cache", "manifest", "out", "run_id",
                "run_name", "config", "ckpt")
SPEED_KINDS = ("aed", "cohere", "ctc", "parakeet-ctc", "parakeet-tdt", "whisper")  # speed_probe --kind (+ WP6's)
FAMILIES = ("aed", "ctc")  # 04_distill.FAMILIES

_ENV_WORD = r"[A-Za-z0-9_./:@+,=-]+"  # vast/launch.py env_string: one shell word, no quotes


# ================================================================================================ helpers


def state_dir() -> Path:
    """The box state dir: $KITSUNE_STATE, else STATE_DEFAULT."""
    return Path(os.environ.get(ENV_STATE) or STATE_DEFAULT)


def item_hb_path(item: str, state=None) -> Path:
    """<state>/hb/<item>: the item's heartbeat file (the queue passes it to the child as KITSUNE_HEARTBEAT)."""
    if not isinstance(item, str) or not re.fullmatch(ITEM_RE, item):
        raise ValueError(f"item name {item!r} does not match {ITEM_RE}")
    return Path(state if state is not None else state_dir()) / HB_DIR / item


def shard_split(split: str) -> str:
    """The shard split a row's audio and labels live in: a dev row's are in its train shard (<src>/train-NNNNN)."""
    return "train" if split == DEV_SPLIT else split


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def full_recipe_problems(block) -> list[str]:
    """Why a selection_recipe.full_study block is not the registered full recipe: exactly FULL_STUDY's keys, the study's
    f1a_max, dedup_min_chars and probe_n, draw_audio_s null or a finite number > 0, dev_rule DEV_RULE. Empty: it is."""
    if not isinstance(block, dict):
        return [f"selection_recipe.full_study {block!r} is not an object"]
    p = []
    missing, unknown = [k for k in FULL_STUDY if k not in block], sorted(k for k in block if k not in FULL_STUDY)
    if missing or unknown:
        p.append(f"selection_recipe.full_study keys: missing {missing}, unknown {unknown}")
    for k in ("f1a_max", "dedup_min_chars", "probe_n"):
        if k in block and not (_num(block[k]) and block[k] == FULL_STUDY[k]):
            p.append(f"selection_recipe.full_study.{k} {block[k]!r}, registered {FULL_STUDY[k]!r}")
    d = block.get("draw_audio_s")
    if d is not None and not (_num(d) and d > 0):
        p.append(f"selection_recipe.full_study.draw_audio_s {d!r}: null or a number of seconds > 0")
    if "dev_rule" in block and not (_int(block["dev_rule"]) and block["dev_rule"] == DEV_RULE):
        p.append(f"selection_recipe.full_study.dev_rule {block['dev_rule']!r}, registered {DEV_RULE}")
    return p


def seeded_subset(ids, n: int, seed: int, tag: str) -> set[str]:
    """min(n, len) ids drawn without replacement; depends only on (seed, tag, the id set), not on the order given.
    The same draw as scripts/make_selection.py seeded_subset (the probe and greedy subsets), so a full selection's
    subsets and the trainer's dev pick use one rule."""
    import zlib

    import numpy as np
    ids = sorted(ids)
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    return {ids[i] for i in rng.choice(len(ids), size=min(n, len(ids)), replace=False)}


def dev_pick(rows, sources, per_source: int, seed: int) -> list[str]:
    """The scored dev ids: rows = (id, source) of the selection's KEPT split-"dev" rows in selection order; per source
    in `sources`, seeded_subset(its ids, per_source, seed, f"dev_scored:{source}"). Returns the picked ids in selection
    order. ValueError when a source has no row (a selection without a dev slice for it) or per_source < 1."""
    if not _int(per_source) or per_source < 1:
        raise ValueError(f"per_source must be an int >= 1, got {per_source!r}")
    sources = list(sources)
    rows = [(str(i), str(s)) for i, s in rows]
    by_source: dict[str, list[str]] = {}
    for i, s in rows:
        by_source.setdefault(s, []).append(i)
    if missing := [s for s in sources if not by_source.get(s)]:
        raise ValueError(f"no kept dev rows for {missing}: the selection has no dev slice for them")
    picked: set[str] = set()
    for s in sources:
        picked |= seeded_subset(by_source[s], per_source, seed, f"dev_scored:{s}")
    wanted, seen, out = set(sources), set(), []
    for i, s in rows:
        if s in wanted and i in picked and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def parse_resume_sets(s: str | None) -> dict[str, list[str]]:
    """KITSUNE_RESUME_SETS ("<run_id>:schedule.epochs=4,<run_id>:...") -> {run_id: ["schedule.epochs=4", ...]} (values
    normalised). ValueError on a bad run id (RUN_ID_RE), a key outside RESUME_SET_KEYS, epochs that are not an int
    >= 1, or one key given twice for a run. None or "" -> {}."""
    out: dict[str, list[str]] = {}
    if not s:
        return out
    for part in s.split(","):
        run_id, sep, kv = part.strip().partition(":")
        if not sep or not re.fullmatch(RUN_ID_RE, run_id):
            raise ValueError(f"resume set {part!r}: expected <run_id>:<key>=<value> with a run id like "
                             f"full-p03-20260927T120000Z")
        key, eq, val = kv.partition("=")
        if not eq or key not in RESUME_SET_KEYS:
            raise ValueError(f"resume set {part!r}: only {', '.join(RESUME_SET_KEYS)} may change on a resume")
        if key == "schedule.epochs":
            if not re.fullmatch(r"\d+", val) or int(val) < 1:
                raise ValueError(f"resume set {part!r}: schedule.epochs must be an int >= 1")
            val = str(int(val))
        if any(x.partition("=")[0] == key for x in out.get(run_id, [])):
            raise ValueError(f"resume set {part!r}: {key} given twice for {run_id}")
        out.setdefault(run_id, []).append(f"{key}={val}")
    return out


def parse_resume_reset(s: str | None) -> list[str]:
    """KITSUNE_RESUME_RESET ("<run_id>[,<run_id>]") -> the run ids, in order, once each. ValueError on an empty entry
    or a bad run id. None or "" -> []."""
    out: list[str] = []
    if not s:
        return out
    for part in s.split(","):
        rid = part.strip()
        if not re.fullmatch(RUN_ID_RE, rid):
            raise ValueError(f"resume reset {part!r}: not a run id like full-p03-20260927T120000Z")
        if rid not in out:
            out.append(rid)
    return out


def run_id_of(run_dir: str) -> str:
    """The run id of a run dir ("runs/<id>" -> "<id>"): its last POSIX component (a Windows separator counts as one)."""
    return PurePosixPath(str(run_dir).replace("\\", "/")).name


def box_summary_path(box) -> str:
    """The box's queue summary in the runs repo (the source of truth of a resume on a new host)."""
    return f"{HUB_DIR}/box-{box}/{SUMMARY_FILE}"


def box_verdict_path(box) -> str:
    return f"{HUB_DIR}/box-{box}/{VERDICT_FILE}"


def box_infra_dir(box, container_id) -> str:
    return f"{HUB_DIR}/box-{box}/infra/{container_id}"


def scratch_state_dir(run_id, step) -> str:
    """A timed full state in the scratch repo (exactly one per run: each upload deletes the previous one)."""
    return f"runs/{run_id}/checkpoints/full_step_{step}"


def scratch_pointer(run_id) -> str:
    """The pointer to a run's timed state in the scratch repo (format POINTER_FORMAT, pointer_problems)."""
    return f"runs/{run_id}/{TIMED_POINTER}"


# the pointer's keys and their types (contract 2.4); "str?" is a string or null
_POINTER_TYPES = {"format": "int", "run_id": "str", "name": "str", "step": "int", "epoch": "num", "wall": "num",
                  "time_utc": "str", "kitsune_sha": "str?", "planner_fingerprint": "str", "n_train_utts": "int",
                  "selection_sha256": "str?", "micro_audio_s": "num", "files": "dict", "host": "dict"}
_TYPE_OK = {"int": _int, "num": _num, "str": lambda v: isinstance(v, str),
            "str?": lambda v: v is None or isinstance(v, str), "dict": lambda v: isinstance(v, dict)}
_HEX64 = r"[0-9a-f]{64}"


def pointer_problems(ptr: dict) -> list[str]:
    """Why a timed-state pointer (runs/<run_id>/timed_state.json, format 1) is not well formed: every key present with
    its type, name == full_step_<step>, files = every STATE_FILES_REQUIRED entry plus only STATE_FILES_OPTIONAL ones
    (never a marker file), each {size: int >= 0, sha256: 64 lowercase hex}, host {hostname, machine_id, container_id}.
    Empty: resume-pull may trust it (and still checks the files against it)."""
    if not isinstance(ptr, dict):
        return [f"pointer {ptr!r} is not an object"]
    p = []
    for k, t in _POINTER_TYPES.items():
        if k not in ptr:
            p.append(f"pointer: no {k}")
        elif not _TYPE_OK[t](ptr[k]):
            p.append(f"pointer: {k} {ptr[k]!r} is not {t.rstrip('?')}{' or null' if t.endswith('?') else ''}")
    if _int(ptr.get("format")) and ptr["format"] != POINTER_FORMAT:
        p.append(f"pointer: format {ptr['format']}, this code reads {POINTER_FORMAT}")
    step = ptr.get("step")
    if _int(step) and step < 0:
        p.append(f"pointer: step {step} < 0")
    if _int(step) and isinstance(ptr.get("name"), str) and ptr["name"] != f"full_step_{step}":
        p.append(f"pointer: name {ptr['name']!r} is not full_step_{step}")
    if _int(ptr.get("n_train_utts")) and ptr["n_train_utts"] < 0:
        p.append(f"pointer: n_train_utts {ptr['n_train_utts']} < 0")
    files = ptr.get("files")
    if isinstance(files, dict):
        if markers := [f for f in files if f in (SCRATCH_MARK, _UPLOAD_MARK)]:
            p.append(f"pointer: files lists the marker file(s) {markers}")
        if missing := [f for f in STATE_FILES_REQUIRED if f not in files]:
            p.append(f"pointer: files lacks {missing}")
        allowed = (*STATE_FILES_REQUIRED, *STATE_FILES_OPTIONAL, SCRATCH_MARK, _UPLOAD_MARK)
        if extra := [f for f in files if f not in allowed]:
            p.append(f"pointer: files lists {extra}, not state files")
        for f, meta in files.items():
            if not isinstance(meta, dict):
                p.append(f"pointer: files.{f} is not an object")
                continue
            if not (_int(meta.get("size")) and meta["size"] >= 0):
                p.append(f"pointer: files.{f}.size {meta.get('size')!r} is not an int >= 0")
            if not (isinstance(meta.get("sha256"), str) and re.fullmatch(_HEX64, meta["sha256"])):
                p.append(f"pointer: files.{f}.sha256 {meta.get('sha256')!r} is not 64 hex digits")
    host = ptr.get("host")
    if isinstance(host, dict):
        if not isinstance(host.get("hostname"), str):
            p.append(f"pointer: host.hostname {host.get('hostname')!r} is not str")
        for k in ("machine_id", "container_id"):
            if k not in host or not (host[k] is None or isinstance(host[k], str)):
                p.append(f"pointer: host.{k} {host.get(k)!r} is not str or null")
    return p


# ================================================================================================ the registry


class RegistryError(ValueError):
    """configs/full/boxes.json is missing or invalid; .problems lists every reason."""

    def __init__(self, msg: str, problems: list[str] | None = None):
        super().__init__(msg)
        self.problems = list(problems or [msg])


_REQ = object()  # a required field
_BOX_FIELDS = {"gpus": _REQ, "data_config": _REQ, "est_hours": _REQ, "max_hours": _REQ, "max_dph": _REQ,
               "extra_gb": 0, "deadline_reserve_min": _REQ, "watchdog": _REQ, "timed_states": False, "gate": True,
               "smoke": False, "max_attempts": 4, "extra_files": [], "extra_dirs": [], "faults": [], "items": _REQ}
_ITEM_FIELDS = {"name": _REQ, "kind": _REQ, "needs": [], "stall_min": _REQ, "max_hours": None, "droppable": _REQ,
                "verdict": []}  # stall_min and droppable: filled per kind
_SOURCE_FIELDS = {"of": None, "of_box": None, "weights": [], "model": None}
_KIND_FIELDS = {
    "stores": {"config": _REQ, "eval_only": False, "sets": []},
    "train": {"config": _REQ, "study_run": _REQ, "family": _REQ, "plan_total_steps": None, "plan_hours": None},
    "readout": {"of": _REQ},
    "speed": {"system": _REQ, "speed_kind": _REQ, **_SOURCE_FIELDS, "args": [], "only_if_new_machine": None},
    "eval": {"argv": _REQ, **_SOURCE_FIELDS},
}
_FAULT_FIELDS = {"id": _REQ, "action": _REQ, "item": _REQ, "at_step": None, "after_event": None, "min_attempt": 1,
                 "seconds": None}
_VERDICT_KEYS = ("check", "json", "path", "min", "max", "equals")
_WATCHDOG_ACTIONS = ("stop", "alert")


def _fill(where: str, spec: dict, fields: dict, p: list[str]) -> dict:
    """spec with its defaults filled (a copy); unknown keys and missing required ones are problems. Keys starting with
    "_" are comments and kept as they are."""
    out = {k: copy.deepcopy(v) for k, v in spec.items()}
    if unknown := [k for k in spec if k not in fields and not str(k).startswith("_")]:
        p.append(f"{where}: unknown key(s) {unknown}")
    for k, d in fields.items():
        if k not in out:
            if d is _REQ:
                p.append(f"{where}: no {k}")
            else:
                out[k] = copy.deepcopy(d)
    return out


def _word(where: str, v, p: list[str]) -> bool:
    if not (isinstance(v, str) and re.fullmatch(_ENV_WORD, v)):
        p.append(f"{where} {v!r} is not one env-string word ({_ENV_WORD})")
        return False
    return True


def _is_rel_path(v) -> bool:
    """A repo-relative POSIX path: one env-string word, no drive, no leading /, no empty, '.' or '..' part."""
    return (isinstance(v, str) and bool(re.fullmatch(_ENV_WORD, v)) and not v.startswith("/") and ":" not in v
            and not any(x in ("", ".", "..") for x in v.split("/")))


def _rel_path(where: str, v, p: list[str]) -> bool:
    if not _word(where, v, p):
        return False
    if not _is_rel_path(v):
        p.append(f"{where} {v!r} is not a repo-relative POSIX path")
        return False
    return True


def _str_list(where: str, v, p: list[str]) -> list:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        p.append(f"{where} {v!r} is not a list of strings")
        return []
    return v


def _placeholders(where: str, s, item: dict, earlier: list[str], p: list[str]) -> list[str]:
    """Check one template string's placeholders against the item; returns the items it names as {out:<item>}."""
    if not isinstance(s, str):
        p.append(f"{where} {s!r} is not a string")
        return []
    if "{" in re.sub(r"\{[^{}]*\}", "", s) or "}" in re.sub(r"\{[^{}]*\}", "", s):
        p.append(f"{where} {s!r}: an unmatched brace")
    outs = []
    ws = item.get("weights") if isinstance(item.get("weights"), list) else []
    weights = {w.get("name") for w in ws if isinstance(w, dict) and isinstance(w.get("name"), str)}
    for tok in re.findall(r"\{([^{}]*)\}", s):
        key, colon, arg = tok.partition(":")
        if colon:
            if key == "out" and arg in earlier:
                outs.append(arg)
            elif key == "out":
                p.append(f"{where}: {{out:{arg}}} names no earlier item of the box")
            elif key in ("config", "ckpt") and arg in weights:
                pass
            elif key in ("config", "ckpt"):
                p.append(f"{where}: {{{tok}}} names no entry of the item's weights")
            else:
                p.append(f"{where}: unknown placeholder {{{tok}}}")
        elif tok not in PLACEHOLDERS:
            p.append(f"{where}: unknown placeholder {{{tok}}}")
        elif tok in ("config", "ckpt", "run_id", "run_name") and not item.get("of"):
            p.append(f"{where}: {{{tok}}} needs the item's `of`")
    return outs


def _check_verdict(where: str, specs, item: dict, earlier: list[str], p: list[str]) -> list:
    if not isinstance(specs, list):
        p.append(f"{where}.verdict {specs!r} is not a list")
        return []
    for j, v in enumerate(specs):
        w = f"{where}.verdict[{j}]"
        if not isinstance(v, dict):
            p.append(f"{w} {v!r} is not an object")
            continue
        if unknown := [k for k in v if k not in _VERDICT_KEYS and not str(k).startswith("_")]:
            p.append(f"{w}: unknown key(s) {unknown}")
        c = v.get("check")
        if not (isinstance(c, str) and re.fullmatch(r"[1-9][0-9]?", c) and 1 <= int(c) <= 16):
            p.append(f"{w}.check {c!r} is not one of \"1\"..\"16\"")
        if ("json" in v) != ("path" in v):
            p.append(f"{w}: json and path go together")
        if "json" in v:
            _placeholders(f"{w}.json", v["json"], item, earlier, p)
        if "path" in v and not (isinstance(v["path"], str) and all(v["path"].split("."))):
            p.append(f"{w}.path {v['path']!r} is not a dotted key path")
        for k in ("min", "max"):
            if k in v and not _num(v[k]):
                p.append(f"{w}.{k} {v[k]!r} is not a number")
        if any(k in v for k in ("min", "max", "equals")) and "path" not in v:
            p.append(f"{w}: a condition needs json and path")
    return specs


def _check_sources(where: str, it: dict, bname: str, box: dict, train_names: list[str], earlier: list[str],
                   p: list[str]) -> list[str]:
    """The model sources of a speed / eval item; returns the implicit needs (a same-box `of`)."""
    implicit = []
    of, of_box = it.get("of"), it.get("of_box")
    if of_box is not None:
        if of is None:
            p.append(f"{where}: of_box {of_box!r} without of")
        if of_box == bname or not isinstance(of_box, str):
            p.append(f"{where}: of_box {of_box!r} is not another registry box")
    elif of is not None:
        if of in train_names and of in earlier:
            implicit.append(of)
        else:
            p.append(f"{where}: of {of!r} is not an earlier train item of box {bname}")
    if of is not None:
        _word(f"{where}.of", of, p)
    weights = it.get("weights")
    if not isinstance(weights, list):
        p.append(f"{where}.weights {weights!r} is not a list")
    else:
        names = []
        for j, w in enumerate(weights):
            ww = f"{where}.weights[{j}]"
            if not isinstance(w, dict) or set(w) != {"name", "run_id", "step"}:
                p.append(f"{ww} {w!r} is not {{name, run_id, step}}")
                continue
            if _word(f"{ww}.name", w["name"], p) and w["name"] in names:
                p.append(f"{ww}.name {w['name']!r} is given twice")
            names.append(w["name"])
            if not (isinstance(w["run_id"], str) and re.fullmatch(RUN_ID_RE, w["run_id"])):
                p.append(f"{ww}.run_id {w['run_id']!r} is not a run id")
            if not (_int(w["step"]) and w["step"] >= 0):
                p.append(f"{ww}.step {w['step']!r} is not an int >= 0")
    model = it.get("model")
    if model is not None and _rel_path(f"{where}.model", model, p) and model not in (box.get("extra_dirs") or []):
        p.append(f"{where}.model {model!r} is not in the box's extra_dirs")
    return implicit


def _check_item(where: str, it: dict, bname: str, box: dict, earlier: list[str], train_names: list[str],
                reg_boxes: dict, p: list[str]) -> dict:
    kind = it.get("kind")
    if not isinstance(kind, str) or kind not in ITEM_KINDS:
        p.append(f"{where}: kind {kind!r} is not one of {ITEM_KINDS}")
        return copy.deepcopy(it)
    fields = dict(_ITEM_FIELDS, **_KIND_FIELDS[kind], stall_min=STALL_MIN_DEFAULT[kind], droppable=kind != "train")
    out = _fill(where, it, fields, p)
    g = out.get
    needs = _str_list(f"{where}.needs", g("needs"), p)
    for n in needs:
        if n not in earlier:
            p.append(f"{where}: needs {n!r}, which is not an earlier item of box {bname}")
    sm, mh = g("stall_min"), g("max_hours")
    if sm is not None and not (_num(sm) and sm > 0):
        p.append(f"{where}.stall_min {sm!r}: a number of minutes > 0, or null for no stall check")
    if mh is not None and not (_num(mh) and mh > 0):
        p.append(f"{where}.max_hours {mh!r} is not a number > 0")
    if mh is None and kind == "train":
        p.append(f"{where}: a train item needs max_hours")
    elif mh is None and sm is None:
        p.append(f"{where}: stall_min null (no stall check) needs max_hours (the overrun kill)")
    d = g("droppable")
    if not isinstance(d, bool):
        p.append(f"{where}.droppable {d!r} is not a bool")
    elif kind != "train" and not d:
        p.append(f"{where}: droppable false is for train items only ({kind} items are always droppable)")
    if g("verdict") and not box.get("smoke"):
        p.append(f"{where}: verdict specs on box {bname}, which is not a smoke box")
    implicit: list[str] = []
    if kind in ("stores", "train") and "config" in out:
        _rel_path(f"{where}.config", g("config"), p)
    if kind == "stores":
        if not isinstance(g("eval_only"), bool):
            p.append(f"{where}.eval_only {g('eval_only')!r} is not a bool")
        for s in _str_list(f"{where}.sets", g("sets"), p):
            if _word(f"{where}.sets", s, p) and not re.fullmatch(r"[A-Za-z0-9_.]+=.+", s):
                p.append(f"{where}.sets {s!r} is not key=value")
    elif kind == "train":
        from kitsune import prereg
        if "study_run" in out and not (isinstance(g("study_run"), str) and g("study_run") in prereg.RUNS):
            p.append(f"{where}.study_run {g('study_run')!r} is not a kitsune.prereg run")
        if "family" in out and g("family") not in FAMILIES:
            p.append(f"{where}.family {g('family')!r} is not one of {FAMILIES}")
        pts, ph = g("plan_total_steps"), g("plan_hours")
        if pts is not None and not (_int(pts) and pts >= 1):
            p.append(f"{where}.plan_total_steps {pts!r} is not an int >= 1")
        if ph is not None and not (_num(ph) and ph > 0):
            p.append(f"{where}.plan_hours {ph!r} is not a number > 0")
        if (pts is None) != (ph is None):
            p.append(f"{where}: plan_total_steps and plan_hours go together")
    elif kind == "readout":
        if g("of") in train_names and g("of") in earlier:
            implicit.append(g("of"))
        elif "of" in out:
            p.append(f"{where}: of {g('of')!r} is not an earlier train item of box {bname}")
    else:  # speed, eval
        implicit += _check_sources(where, out, bname, box, train_names, earlier, p)
    if kind == "speed":
        if "system" in out:
            _word(f"{where}.system", g("system"), p)
        if "speed_kind" in out and g("speed_kind") not in SPEED_KINDS:
            p.append(f"{where}.speed_kind {g('speed_kind')!r} is not one of {SPEED_KINDS}")
        n_src = (g("of") is not None) + (len(g("weights")) if isinstance(g("weights"), list) else 0) + (
            g("model") is not None)
        if n_src > 1:
            p.append(f"{where}: a speed item times one model (of, one weights entry or model), got {n_src}")
        if g("speed_kind") == "cohere" and n_src:
            p.append(f"{where}: speed_kind cohere times the teacher, no model source")
        for j, a in enumerate(_str_list(f"{where}.args", g("args"), p)):
            implicit += _placeholders(f"{where}.args[{j}]", a, out, earlier, p)
        o = g("only_if_new_machine")
        if o is not None and not (isinstance(o, str) and o != bname and o in reg_boxes):
            p.append(f"{where}.only_if_new_machine {o!r} is not another registry box")
    if kind == "eval" and "argv" in out:
        argv = g("argv")
        if not isinstance(argv, list) or not argv:
            p.append(f"{where}.argv {argv!r} is not a non-empty list")
        else:
            for j, a in enumerate(argv):
                implicit += _placeholders(f"{where}.argv[{j}]", a, out, earlier, p)
    _check_verdict(where, g("verdict"), out, earlier, p)
    out["needs"] = list(dict.fromkeys([*needs, *implicit]))
    return out


def _check_faults(where: str, faults, box: dict, train_names: list[str], p: list[str]) -> list:
    if not isinstance(faults, list):
        p.append(f"{where}.faults {faults!r} is not a list")
        return []
    if faults and box.get("smoke") is not True:
        p.append(f"{where}: faults on a box that is not a smoke box")
    out, ids = [], []
    wd = box.get("watchdog") if isinstance(box.get("watchdog"), dict) else {}
    for j, f in enumerate(faults):
        w = f"{where}.faults[{j}]"
        if not isinstance(f, dict):
            p.append(f"{w} {f!r} is not an object")
            continue
        f = _fill(w, f, _FAULT_FIELDS, p)
        out.append(f)
        fid, action = f.get("id"), f.get("action")
        if isinstance(fid, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", fid):
            if fid in ids:
                p.append(f"{w}: id {fid!r} is given twice")
            ids.append(fid)
        else:
            p.append(f"{w}.id {fid!r} is not a name")
        if action not in FAULT_ACTIONS:
            p.append(f"{w}.action {action!r} is not one of {FAULT_ACTIONS}")
        if f.get("item") not in train_names:
            p.append(f"{w}.item {f.get('item')!r} is not a train item of the box")
        at, ev, sec, ma = f.get("at_step"), f.get("after_event"), f.get("seconds"), f.get("min_attempt")
        if at is not None and not (_int(at) and at >= 0):
            p.append(f"{w}.at_step {at!r} is not an int >= 0")
        if ev is not None and not (isinstance(ev, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", ev)):
            p.append(f"{w}.after_event {ev!r} is not an event kind")
        if sec is not None and not (_num(sec) and sec > 0):
            p.append(f"{w}.seconds {sec!r} is not a number > 0")
        if not (_int(ma) and ma >= 1):
            p.append(f"{w}.min_attempt {ma!r} is not an int >= 1")
        if action in ("sigstop", "kill", "freeze_controller_hb") and at is None:
            p.append(f"{w}: {action} needs at_step")
        if action == "wipe_run_dir" and ev is None:
            p.append(f"{w}: wipe_run_dir needs after_event")
        if action in ("deadline", "freeze_controller_hb") and sec is None:
            p.append(f"{w}: {action} needs seconds")
        if action == "freeze_controller_hb":
            if wd.get("action") != "alert":
                p.append(f"{w}: freeze_controller_hb on a box whose watchdog action is {wd.get('action')!r}: it "
                         f"would stop the box; only an alert box may freeze its controller heartbeat")
            if _num(sec) and _int(wd.get("orphan_s")) and not sec > wd["orphan_s"]:
                p.append(f"{w}: seconds {sec} must exceed watchdog.orphan_s {wd['orphan_s']} (the alert must fire)")
    return out


def _check_box(bname: str, box, reg_boxes: dict, p: list[str]) -> dict | None:
    where = f"boxes.{bname}"
    if not isinstance(box, dict):
        p.append(f"{where} is not an object")
        return None
    out = _fill(where, box, _BOX_FIELDS, p)
    if "gpus" in out and not (_int(out["gpus"]) and out["gpus"] >= 1):
        p.append(f"{where}.gpus {out['gpus']!r} is not an int >= 1")
    dc = out.get("data_config")
    if dc is not None and _rel_path(f"{where}.data_config", dc, p) and not dc.startswith("configs/full/"):
        p.append(f"{where}.data_config {dc!r} is not under configs/full/")
    for k in ("est_hours", "max_hours", "max_dph"):
        if k in out and not (_num(out[k]) and out[k] > 0):
            p.append(f"{where}.{k} {out[k]!r} is not a number > 0")
    if _num(out.get("est_hours")) and _num(out.get("max_hours")) and out["max_hours"] < out["est_hours"]:
        p.append(f"{where}: max_hours {out['max_hours']} < est_hours {out['est_hours']}")
    for k in ("extra_gb", "deadline_reserve_min"):
        if k in out and not (_num(out[k]) and out[k] >= 0):
            p.append(f"{where}.{k} {out[k]!r} is not a number >= 0")
    wd = out.get("watchdog")
    if "watchdog" in out:
        if not isinstance(wd, dict) or set(wd) != {"orphan_s", "action"}:
            p.append(f"{where}.watchdog {wd!r} is not {{orphan_s, action}}")
        else:
            if not (_int(wd["orphan_s"]) and wd["orphan_s"] >= 0):
                p.append(f"{where}.watchdog.orphan_s {wd['orphan_s']!r} is not an int >= 0")
            if wd["action"] not in _WATCHDOG_ACTIONS:
                p.append(f"{where}.watchdog.action {wd['action']!r} is not one of {_WATCHDOG_ACTIONS}")
    for k in ("timed_states", "gate", "smoke"):
        if not isinstance(out.get(k), bool):
            p.append(f"{where}.{k} {out.get(k)!r} is not a bool")
    if not (_int(out.get("max_attempts")) and out["max_attempts"] >= 1):
        p.append(f"{where}.max_attempts {out.get('max_attempts')!r} is not an int >= 1")
    for k in ("extra_files", "extra_dirs"):
        for x in _str_list(f"{where}.{k}", out.get(k), p):
            _rel_path(f"{where}.{k}", x, p)
    items = out.get("items")
    if "items" in out and (not isinstance(items, list) or not items):
        p.append(f"{where}.items {items!r} is not a non-empty list")
        items = []
    items = items or []
    train_names = [it.get("name") for it in items if isinstance(it, dict) and it.get("kind") == "train"]
    filled, earlier = [], []
    for j, it in enumerate(items):
        w = f"{where}.items[{j}]"
        if not isinstance(it, dict):
            p.append(f"{w} {it!r} is not an object")
            continue
        name = it.get("name")
        if isinstance(name, str) and re.fullmatch(ITEM_RE, name):
            w = f"{where}.items.{name}"
            if name in earlier:
                p.append(f"{w}: the name is given twice")
        else:
            p.append(f"{w}.name {name!r} does not match {ITEM_RE}")
        filled.append(_check_item(w, it, bname, out, earlier, train_names, reg_boxes, p))
        earlier.append(name)
    out["items"] = filled
    out["faults"] = _check_faults(where, out.get("faults"), out, train_names, p)
    return out


def _read_json(root: Path, rel: str, cache: dict):
    if rel not in cache:
        try:
            cache[rel] = json.loads((root / rel).read_text(encoding="utf-8"))
        except FileNotFoundError:
            cache[rel] = FileNotFoundError(f"{rel} does not exist under {root}")
        except (OSError, ValueError) as e:
            cache[rel] = ValueError(f"{rel}: not readable JSON ({type(e).__name__}: {e})")
    return cache[rel]


def _data_keys(cfg: dict) -> dict:
    return {k: (cfg.get(k, False) if k == "pull_parakeet" else cfg.get(k)) for k in DATA_KEYS}


def _check_files(boxes: dict, root: Path, p: list[str]):
    """Every box's data config and item configs exist and are objects; the item configs carry the data config's
    DATA_KEYS values; a train item's family is its config's."""
    cache: dict = {}
    for bname, box in boxes.items():
        dc = box.get("data_config")  # a path that is not repo-relative is a problem already; it is never read
        data = _read_json(root, dc, cache) if _is_rel_path(dc) else None
        if isinstance(data, Exception) or (_is_rel_path(dc) and not isinstance(data, dict)):
            p.append(f"boxes.{bname}.data_config: {data if isinstance(data, Exception) else 'not a JSON object'}")
            data = None
        for it in box.get("items") or []:
            cfg_path = it.get("config") if it.get("kind") in ("stores", "train") else None
            if not _is_rel_path(cfg_path):
                continue
            w = f"boxes.{bname}.items.{it.get('name')}"
            cfg = _read_json(root, cfg_path, cache)
            if isinstance(cfg, Exception) or not isinstance(cfg, dict):
                p.append(f"{w}.config: {cfg if isinstance(cfg, Exception) else 'not a JSON object'}")
                continue
            if data is not None:
                got, want = _data_keys(cfg), _data_keys(data)
                if diff := [k for k in DATA_KEYS if got[k] != want[k]]:
                    p.append(f"{w}: {cfg_path} differs from the box data config {dc} in {diff}")
            if it.get("kind") == "train" and (cfg.get("family") or "aed") != it.get("family"):
                p.append(f"{w}: family {it.get('family')!r}, but {cfg_path} trains {cfg.get('family') or 'aed'!r}")


def _check(reg, root, check_files: bool) -> tuple[list[str], dict | None]:
    """(problems, the registry with its defaults filled)."""
    if not isinstance(reg, dict):
        return [f"the registry {type(reg).__name__} is not an object"], None
    p: list[str] = []
    out = {k: copy.deepcopy(v) for k, v in reg.items() if str(k).startswith("_")}
    if unknown := [k for k in reg if k not in ("version", "boxes") and not str(k).startswith("_")]:
        p.append(f"registry: unknown key(s) {unknown}")
    if not (_int(reg.get("version")) and reg["version"] == 1):
        p.append(f"registry: version {reg.get('version')!r}, this code reads 1")
    out["version"] = reg.get("version")
    boxes = reg.get("boxes")
    if not isinstance(boxes, dict) or not boxes:
        p.append(f"registry: boxes {boxes!r} is not a non-empty object")
        boxes = {}
    if unknown := [b for b in boxes if b not in BOX_NAMES]:
        p.append(f"registry: box(es) {unknown} are not in BOX_NAMES {BOX_NAMES}")
    out["boxes"] = {}
    for bname, box in boxes.items():
        filled = _check_box(bname, box, boxes, p)
        if filled is not None:
            out["boxes"][bname] = filled
    # of_box: a train item of another registry box (a box that is not an object is a problem already)
    for bname, box in out["boxes"].items():
        for it in box["items"]:
            ob, of = it.get("of_box"), it.get("of")
            if it.get("kind") not in ("speed", "eval") or not isinstance(ob, str) or ob == bname or of is None:
                continue
            if ob not in boxes:
                p.append(f"boxes.{bname}.items.{it.get('name')}: of_box {ob!r} is not a registry box")
            elif ob in out["boxes"] and of not in [x.get("name") for x in out["boxes"][ob]["items"]
                                                   if x.get("kind") == "train"]:
                p.append(f"boxes.{bname}.items.{it.get('name')}: of {of!r} is not a train item of box {ob}")
    if check_files:
        _check_files(out["boxes"], Path(root) if root is not None else REPO, p)
    return p, out


def registry_problems(reg: dict, *, root=None, check_files=True) -> list[str]:
    """Why reg is not a valid registry (contract 2.3); empty: it is. root: the checkout its paths resolve against
    (default this one); check_files False skips reading the data and item configs (launch reads them at its sha)."""
    return _check(reg, root, check_files)[0]


def _registry_root(path: Path) -> Path:
    """The checkout a registry file belongs to: <X> for <X>/configs/full/boxes.json, else this repo."""
    parts = path.resolve().parts
    return path.resolve().parents[2] if tuple(parts[-3:]) == tuple(BOXES_FILE.split("/")) else REPO


def _load(path_or_dict, root, check_files: bool) -> tuple[dict, Path]:
    if isinstance(path_or_dict, dict):
        reg, where = path_or_dict, "registry"
    else:
        env = os.environ.get(ENV_REGISTRY) if path_or_dict is None else None
        f = Path(path_or_dict if path_or_dict is not None else env or Path(root or REPO) / BOXES_FILE)
        if not f.is_file():
            if path_or_dict is None and not env:
                raise RegistryError(f"{BOXES_FILE} is not in this checkout (WP2c)")
            raise RegistryError(f"{f} does not exist" + (f" ({ENV_REGISTRY})" if env else ""))
        try:
            reg = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RegistryError(f"{f}: not readable JSON ({type(e).__name__}: {e})") from None
        where = str(f)
        if root is None:
            root = _registry_root(f)
    root = Path(root) if root is not None else REPO
    problems, filled = _check(reg, root, check_files)
    if problems:
        raise RegistryError(f"{where} is not a valid box registry:\n  " + "\n  ".join(problems), problems)
    return filled, root


def load_registry(path_or_dict=None, *, root=None, check_files=True) -> dict:
    """The validated registry with its defaults filled (a copy). None reads $KITSUNE_FULL_REGISTRY, else
    <root or this repo>/configs/full/boxes.json; a path reads that file; a dict (launch: the JSON at the launched sha)
    is validated the same way. RegistryError when it is missing or invalid. Paths resolve against root, by default the
    checkout the file is in (<X> for <X>/configs/full/boxes.json), else this repo."""
    return _load(path_or_dict, root, check_files)[0]


def _resolved(registry, root, check_files: bool = False) -> tuple[dict, Path]:
    """A loaded registry and the root its paths resolve against: None loads the file with every check; a given
    registry (loaded or raw) is validated and filled again without reading files (idempotent, cheap)."""
    if registry is None:
        return _load(None, root, True)
    return _load(registry, root, check_files)


def box_spec(box, registry=None) -> dict:
    """The box's spec with its defaults filled. RegistryError for a box the registry does not have."""
    reg, _ = _resolved(registry, None)
    if box not in reg["boxes"]:
        raise RegistryError(f"box {box!r} is not in the registry (it has {', '.join(reg['boxes']) or 'none'})")
    return copy.deepcopy(reg["boxes"][box])


def box_items(box, registry=None) -> list[dict]:
    """The box's items, in registry order, with their defaults (and implicit needs) filled."""
    return box_spec(box, registry)["items"]


def train_items(box, registry=None) -> list[dict]:
    return [it for it in box_items(box, registry) if it["kind"] == "train"]


def box_configs(box, registry=None) -> list[str]:
    """Every config the box reads, repo-relative and once each: its data config, its stores and train items' configs,
    and the registry file itself (launch checks they are committed at the sha it rents)."""
    spec = box_spec(box, registry)
    out = [spec["data_config"], *(it["config"] for it in spec["items"] if it["kind"] in ("stores", "train")),
           BOXES_FILE]
    return list(dict.fromkeys(out))


def _config_student(root: Path, rel: str) -> str:
    try:
        cfg = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RegistryError(f"{rel}: cannot read it under {root} ({type(e).__name__}: {e})") from None
    s = cfg.get("student") if isinstance(cfg, dict) else None
    if not isinstance(s, str) or not s:
        raise RegistryError(f"{rel} names no student")
    return s.rstrip("/")


def box_students(box, registry=None, root=None) -> list[str]:
    """The student dirs (repo paths) of the box's train items' configs, in item order, once each: what bootstrap pulls
    and check-students checks."""
    reg, r = _resolved(registry, root)
    items = box_spec(box, reg)["items"]
    return list(dict.fromkeys(_config_student(r, it["config"]) for it in items if it["kind"] == "train"))


def box_ctc_students(box, registry=None, root=None) -> list[str]:
    """The Parakeet-family ones among box_students (family ctc): they carry the CC-BY-4.0 attribution card."""
    reg, r = _resolved(registry, root)
    items = box_spec(box, reg)["items"]
    return list(dict.fromkeys(_config_student(r, it["config"]) for it in items
                              if it["kind"] == "train" and it["family"] == "ctc"))


def box_extra_files(box, registry=None) -> list[str]:
    """The data-repo files the box pulls and requires besides the labels and students (e.g. the frozen manifest and the
    selection sidecar)."""
    return list(box_spec(box, registry)["extra_files"])


def box_extra_dirs(box, registry=None) -> list[str]:
    """The data-repo dirs the box pulls and requires (e.g. the Parakeet teacher for its speed probe)."""
    return list(box_spec(box, registry)["extra_dirs"])


def student_checks(box, read_meta, registry=None, root=None) -> list[str]:
    """Why the students the box trains are not the registered builds: each train item's config must train its
    study_run's registered student dir, and read_meta(student dir) -> its student_meta.json (a dict) must pass
    kitsune.prereg.student_problems(study_run, meta). A meta it cannot read is a problem too. Empty: every student is
    the registered build (launch checks the data repo's copies, bootstrap the pulled ones)."""
    from kitsune import prereg
    reg, r = _resolved(registry, root)
    problems, metas = [], {}
    for it in box_spec(box, reg)["items"]:
        if it["kind"] != "train":
            continue
        run = it["study_run"]
        s, want = _config_student(r, it["config"]), prereg.RUNS[run]["student"]
        if s != want:
            problems.append(f"{it['name']}: {it['config']} trains {s}, {run} registers {want}")
        if s not in metas:
            try:
                metas[s] = read_meta(s)
            except Exception as e:  # noqa: BLE001  a missing or unreadable meta refuses like a wrong one
                metas[s] = None
                problems.append(f"{s}/student_meta.json: cannot read it ({type(e).__name__}: {e})")
        if metas[s] is not None:
            problems += [f"{s}: {x}" for x in prereg.student_problems(run, metas[s])]
    return list(dict.fromkeys(problems))


def box_env(box, registry=None) -> dict[str, str]:
    """The registry's part of the box env (launch): KITSUNE_N_GPUS and the watchdog's heartbeat file (train_hb), orphan
    limit and action."""
    spec = box_spec(box, registry)
    return {ENV_N_GPUS: str(spec["gpus"]), ENV_WATCHDOG_HB_FILE: TRAIN_HB,
            ENV_WATCHDOG_ORPHAN_S: str(spec["watchdog"]["orphan_s"]),
            ENV_WATCHDOG_ORPHAN_ACTION: spec["watchdog"]["action"]}


# ================================================================================================ CLI

EXIT_OK, EXIT_REFUSED = 0, 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kitsune.fullrun", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("students", "extra-files", "extra-dirs", "check-students", "show"):
        sp = sub.add_parser(c)
        sp.add_argument("--box", default=os.environ.get(ENV_BOX), choices=BOX_NAMES)
        sp.add_argument("--root", default=None,
                        help="the checkout: the registry and configs are read there, and check-students reads the "
                             "pulled student dirs there (default: this repo)")
    args = ap.parse_args(argv)
    if not args.box:
        ap.error(f"--box (or {ENV_BOX}) is required")
    root = Path(args.root) if args.root else None
    try:
        reg, r = _resolved(None, root)
        if args.cmd == "students":
            lines = box_students(args.box, reg, r)
        elif args.cmd == "extra-files":
            lines = box_extra_files(args.box, reg)
        elif args.cmd == "extra-dirs":
            lines = box_extra_dirs(args.box, reg)
        elif args.cmd == "show":
            print(json.dumps({"box": args.box, "env": box_env(args.box, reg), "configs": box_configs(args.box, reg),
                              "students": box_students(args.box, reg, r), "spec": box_spec(args.box, reg)},
                             indent=2, ensure_ascii=False))
            return EXIT_OK
        else:  # check-students: vast/bootstrap.sh after the pull
            problems = student_checks(args.box, lambda s: json.loads(
                (r / s / "student_meta.json").read_text(encoding="utf-8")), reg, r)
            for x in problems:
                print(f"student refused: {x}", file=sys.stderr)
            if not problems:
                print(f"box {args.box}: {len(box_students(args.box, reg, r))} student(s) are the registered builds")
            return EXIT_REFUSED if problems else EXIT_OK
    except RegistryError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if lines:
        print("\n".join(lines))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
