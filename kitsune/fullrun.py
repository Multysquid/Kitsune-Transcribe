"""The full-data runs' shared core: names, paths, the box registry and the small helpers every full-run package uses.

The full runs (plan v3) trained P-0.1B alone on box `p01` (1x RTX 5090, Parakeet labels only; box 1), which now
re-runs that run's cooldown with the CTC train-data augmentation on, the recipe test (DECISIONS H1: --resume-reset with
--resume-set schedule.epochs and augment.*; G3's 8-epoch continuation is postponed, H2), then T-0.6B on box `full-t`
and P-0.3B and P-0.05B on box `full-p` (each 1x RTX 5090, one queue;
DECISIONS G2: box 2's 2x box `full` is retired, its name kept for the tests' fixtures only), after two short smokes
(`full-smoke` = smoke A, `smoke-b`).
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
               flags' parsers (parse_resume_sets with its typed, normalised values - resume_set_value - and
               parse_resume_reset: launch -> env -> full_queue resume-pull),
               run_id_of, the Hub and scratch paths, pointer_problems (the timed-state pointer, format 1)
  registry     configs/full/boxes.json, hand-written next to the generated configs/full/*.json (make_full_configs.py
               --check validates it with registry_problems). One source of truth: launch reads the hours, price, GPU
               count and watchdog from it, bootstrap the students and extra files, the queue its items.
               load_registry validates it and fills the defaults; box_* read one box; the CLI serves bootstrap

Registry (contract 2.3). Top level {"version": 1, "boxes": {<box>: <box spec>}} with box names from BOX_NAMES (a chain
box's from CHAIN_NAMES, below); any key
starting with "_" is a comment, anywhere (in the watchdog block and a weights entry too). Paths (data_config, item
config) are repo-relative POSIX paths resolved against `root` (the checkout; default the one the registry file is in,
else this one), or read with a `read_json(rel)` callable instead (launch: the file at the sha it rents). A registry
load_registry returns remembers its root, so the box readers given it read the configs there. A box spec:

  gpus int >= 1; data_config (under configs/full/); est_hours <= max_hours, max_dph (> 0; launch's defaults);
  deadline_reserve_min >= 0 (the queue's per-item KITSUNE_DEADLINE = box deadline - this); watchdog {orphan_s int
  >= 0, action stop|alert}; extra_gb >= 0 (0); timed_states (false: the scratch repo is required when true); gate
  (true: the download gate); smoke (false: writes smoke_verdict.json, allows faults and verdict specs);
  max_attempts >= 1 (4); extra_files, extra_dirs (data-repo paths the box pulls, []); faults ([]); items (non-empty);
  min_ram_gb (null: launch's 64 GB a GPU; a number > 0: the host RAM launch's offer filter asks for instead, vast/
  launch.py full_job; a chain takes the largest of its parts')

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
  speed    system, speed_kind (tools/speed_probe.py --kind), args ([]), only_if_new_machine (another registry box, or
           null), and the model source speed_probe needs: aed / ctc exactly one (of, one weights entry or model);
           parakeet-ctc / parakeet-tdt `model` (the teacher's data-repo dir); cohere none; whisper `model`, or none
           with the key in args (["--model", "<key>", "--hf-cache", "{hf_cache}"]; refused without either)
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
freeze_controller_hb at_step and seconds >= watchdog.orphan_s + WATCHDOG_POLL_S on a box whose watchdog only alerts
(on a chain: under the last stage's stop orphan_s). Item configs must
exist (check_files) and carry the box data config's DATA_KEYS values (a missing pull_parakeet counts as false); a train
item's family is its config's (default aed). Without pull_parakeet, every stores and train item config has the data
config's family (default aed): bootstrap pulls the labels the data config's pull plan names (kitsune/extent.py
pull_plan), which then hold the train labels of that family only (ctc: parakeet_out, plus teacher_out for the eval
stems; aed: teacher_out). Every name and path the box env or a command line carries is one env-string word
(vast/launch.py env_string).

Chain boxes (contract addendum E; CHAIN_NAMES, p01-chain): an entry with the key `chain` names existing boxes as
parts, in two stages, and has no items of its own: {chain: [stage 1, stage 2], est_hours <= max_hours, max_dph,
extra_gb (>= the last stage's parts'), gate (must be true)}. Stage 1 {parts (parts[0] = gate_box), gate_box (a smoke
box with train items: its verdict checks 1-11 gate stage 2), gate_by_hours (default the gate box's max_hours; <= the
stage's max_hours), max_hours (first boot to stage 1's sub-deadline; < the chain's), rebuild (the extent its bootstrap
rebuilds; default parts[0]'s data config)}; stage 2 {parts, rebuild}. Each box is a part once, every part has the same
gpus, only the gate part may carry faults or an alert watchdog, and the chain's max_hours - stage 1's must hold the
last stage's est_hours. With check_files, each part's extent lies within its stage's rebuild (the same roots; parakeet
pulled when a part needs it) and stage 1's rebuild within stage 2's, whose bootstrap reuses its shards. The filled
registry keeps a chain in this normalised raw form; box_spec derives its spec (the parts' GPU count, the last stage's
rebuild as data config, stage 1's watchdog, the union of the stage views' extra files) and never stores it. The
readers take stage= (a chain: that stage's view, None the union; ignored on a plain box), box_env(stage=) adds
KITSUNE_CHAIN_STAGE and stage 1's KITSUNE_WATCHDOG_HANDOVER_S, and box_items / train_items refuse a chain: the chain
controller (kitsune/full_queue.py) runs its parts' queues one after the other.

CLI (vast/bootstrap.sh; exit 0 ok, 2 refused - a bad registry, an unknown box, a student that is not the registered
build):
  python -m kitsune.fullrun students --box p01 [--root R]         # the train items' student dirs, one per line
  python -m kitsune.fullrun extra-files --box p01                 # the box's extra data-repo files, one per line
  python -m kitsune.fullrun extra-dirs --box full-smoke           # its extra data-repo dirs
  python -m kitsune.fullrun check-students --box p01 --root R     # every pulled student is the registered build
  python -m kitsune.fullrun show --box full-t                     # the box spec with defaults, env and configs (JSON)
  python -m kitsune.fullrun check-students --box p01-chain --stage 2 --root R   # a chain: one stage's view
A chain's --stage defaults to $KITSUNE_CHAIN_STAGE, else 1 (show adds both stage views).
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
# smoke A, box p01 (box 1, then its recipe test: box 1's cooldown again), "full" (the retired 2x box 2: no longer in
# the registry, kept for tests/fixtures_full.py's 2-GPU box only), smoke B, and box 2 as two 1x boxes (DECISIONS G2):
# full-t (T-0.6B) and full-p (P-0.3B, P-0.05B, the Whisper models and their quantised readouts)
BOX_NAMES = ("full-smoke", "p01", "full", "smoke-b", "full-t", "full-p")
# chain boxes (contract addendum E, DECISIONS D): one rental that runs registry boxes one after the other, in two
# stages with an automatic gate between them (kitsune/full_queue.py ChainController). p01-chain = smoke A and smoke B,
# then box 1, on one 1x RTX 5090
CHAIN_NAMES = ("p01-chain",)
ALL_BOX_NAMES = BOX_NAMES + CHAIN_NAMES  # the registry's box-name check; --box of the fullrun, full_queue, launch CLIs
HUB_DIR = "full"  # runs repo: full/box-<box>/{queue_summary.json, smoke_verdict.json, infra/<container id>/}
STATE_DEFAULT = "/workspace/kitsune_state"  # $KITSUNE_STATE on a box
# a chain's controller state under $KITSUNE_STATE: chain/chain.json, chain/<part>/ (each part's queue state dir) and
# chain/stage1/ (stage 1's bootstrap records); the watchdog's mode file $KITSUNE_STATE/watchdog_mode ("<stop|alert>
# <orphan_s>"), which the controller writes when the gate part has ended
CHAIN_DIR, CHAIN_STATE, WATCHDOG_MODE_FILE = "chain", "chain.json", "watchdog_mode"
CHAIN_KIND = "chain"  # the `kind` of a chain's queue summary
GATE_CHECKS = tuple(str(n) for n in range(1, 12))  # smoke A's built-in checks: all must pass for stage 2 to start

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
# the config keys launch --resume-set may change, and the kind of each one's value (RESUME_SET_KINDS; resume_set_value
# parses and normalises it). DECISIONS G3: P-0.1B's continuation sets its epochs and its early-stop patience (ints of at
# least RESUME_SET_INT_MIN[key]). DECISIONS H1: the recipe test box re-runs box 1's cooldown from its pre_cooldown state
# with the CTC train-data augmentation on - augment.enabled (a bool) and the three probabilities the owner's recipe sets
# (truncate 0.3, concat 0.5, mix 0.2: floats in [0, 1]). The augmentation's other keys (its seed, the truncate bounds
# and pause share, concat_max_s / concat_max_n, mix_snr_db) keep the trainer's defaults on purpose: a resume that moved
# them would test another recipe than the one P-0.3B is to train with, so they are not settable here. None of these
# keys shapes the step plan (scripts/04_distill.py RESUME_FIXED holds none of them: augment.* happens inside the train
# loader), so a resume keeps its epoch position; 04_distill validates every value again on the box (validate: epochs,
# bools; validate_augment: the probabilities, and augment.enabled on a CTC student only)
RESUME_SET_KEYS = ("schedule.epochs", "early_stop.patience", "augment.enabled", "augment.truncate_p",
                   "augment.concat_p", "augment.mix_p")
RESUME_SET_INT_MIN = {"schedule.epochs": 1, "early_stop.patience": 1}
RESUME_SET_KINDS = {"schedule.epochs": "int", "early_stop.patience": "int", "augment.enabled": "bool",
                    "augment.truncate_p": "prob", "augment.concat_p": "prob", "augment.mix_p": "prob"}
# a probability's spelling: a plain decimal (digits, one point, an exponent), no sign, no "inf" / "nan" / "1_0" - the
# forms float() would also take but JSON (04_distill's --set) would not, or not as a number
_DECIMAL_RE = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

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
# a chain box (addendum E): the stage bootstrap and the CLIs serve (launch: 1; the controller's stage-2 bootstrap: 2),
# the watchdog's stage-1 hand-over bound (first boot + this many seconds), and the bootstrap phases' toucher bound
ENV_CHAIN_STAGE, ENV_WATCHDOG_HANDOVER_S = "KITSUNE_CHAIN_STAGE", "KITSUNE_WATCHDOG_HANDOVER_S"
ENV_PHASE_HB_MAX_S = "KITSUNE_PHASE_HB_MAX_S"

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


def resume_set_rule(key: str) -> str:
    """The rule a resume set's value must follow (RESUME_SET_KINDS), as launch's --resume-set help and the refusals
    say it."""
    kind = RESUME_SET_KINDS[key]
    if kind == "int":
        return f"an int >= {RESUME_SET_INT_MIN[key]}"
    return "true or false" if kind == "bool" else "a probability in [0, 1]"


def resume_set_value(key: str, val: str) -> str:
    """A resume set's value of key (one of RESUME_SET_KEYS) in its one normalised spelling - the word launch puts in
    KITSUNE_RESUME_SETS, resume-pull records in the plan and the queue passes to the trainer as --set key=<it>:
      int   digits only, at least RESUME_SET_INT_MIN[key], as str(int): "012" -> "12" (no sign, no decimal point)
      bool  true or false in any case, as JSON: "True" -> "true". 04_distill's --set parses JSON, and the string
            "True" it would keep instead fails its bool check - on the box, after the paid boot and the store build
      prob  a plain decimal in [0, 1] (_DECIMAL_RE), as the shortest repr of its float: "0.30", ".3", "3e-1" -> "0.3",
            "1" -> "1.0"; JSON reads that back as the same float
    One spelling per value is what lets the queue compare launches: adopt() records `resume_sets_differ` when the env's
    sets differ from the resume plan's, and a relaunch of the same continuation spelled ".3" must not look like a
    different one. ValueError naming the rule otherwise."""
    kind = RESUME_SET_KINDS[key]
    if kind == "int":
        if re.fullmatch(r"\d+", val) and int(val) >= RESUME_SET_INT_MIN[key]:
            return str(int(val))
    elif kind == "bool":
        if val.lower() in ("true", "false"):
            return val.lower()
    elif re.fullmatch(_DECIMAL_RE, val) and 0.0 <= float(val) <= 1.0:
        return repr(float(val))
    raise ValueError(f"{key} must be {resume_set_rule(key)}, not {val!r}")


def parse_resume_sets(s: str | None) -> dict[str, list[str]]:
    """KITSUNE_RESUME_SETS ("<run_id>:schedule.epochs=4,<run_id>:augment.enabled=true,...") -> {run_id:
    ["schedule.epochs=4", "augment.enabled=true", ...]} (in the order given; each value normalised by
    resume_set_value: "012" -> "12", "True" -> "true", ".30" -> "0.3"). ValueError on a bad run id (RUN_ID_RE), a key
    outside RESUME_SET_KEYS, a value that breaks its key's rule (RESUME_SET_KINDS: an int >= RESUME_SET_INT_MIN, true /
    false, a probability in [0, 1]), or one key given twice for a run. None or "" -> {}."""
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
        try:
            val = resume_set_value(key, val)
        except ValueError as e:
            raise ValueError(f"resume set {part!r}: {e}") from None
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
               "smoke": False, "max_attempts": 4, "extra_files": [], "extra_dirs": [], "faults": [], "items": _REQ,
               "min_ram_gb": None}
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
# vast/watchdog.sh's poll (KITSUNE_WATCHDOG_POLL_S, default 60): a stale heartbeat is seen at the first poll past
# orphan_s, so a freeze_controller_hb window must last orphan_s + this for its alert to fall inside it (box 53693389:
# the queue cut F5's window at 480 s of its 900; the full window, held by the queue's hold_freeze, covers 600 + 60)
WATCHDOG_POLL_S = 60
# a chain box (addendum E.1.2-E.1.3): its own fields, and each stage's (the gated first stage, then the last one); a
# stage's rebuild and the first stage's gate_by_hours are filled (a stored filled chain validates again unchanged)
_CHAIN_FIELDS = {"chain": _REQ, "est_hours": _REQ, "max_hours": _REQ, "max_dph": _REQ, "extra_gb": 0, "gate": True}
_STAGE_FIELDS = ({"parts": _REQ, "gate_box": _REQ, "gate_by_hours": None, "max_hours": _REQ, "rebuild": None},
                 {"parts": _REQ, "rebuild": None})
# a plain box's fields that a chain derives from its parts (E.1.5): never written on a chain
_CHAIN_DERIVED = ("items", "data_config", "watchdog", "faults", "smoke", "timed_states", "extra_files", "extra_dirs",
                  "max_attempts", "gpus", "deadline_reserve_min", "min_ram_gb")
_ROOT_KEYS = ("data_root", "teacher_root", "second_root", "parakeet_root")  # a chain stage shares one data root


def _keys(d: dict) -> set:
    """d's keys without the comments (keys starting with "_")."""
    return {k for k in d if not str(k).startswith("_")}


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


def _gives_option(args, opt: str) -> bool:
    """args (a command-line list) gives the option opt a value the way argparse reads it: `opt value` (a value that
    is not itself an option) or `opt=value`. False for anything that is not a list (a problem already)."""
    if not isinstance(args, list):
        return False
    for j, a in enumerate(args):
        if not isinstance(a, str):
            continue
        if a == opt and j + 1 < len(args) and isinstance(args[j + 1], str) and args[j + 1] \
                and not args[j + 1].startswith("-"):
            return True
        if a.startswith(opt + "=") and len(a) > len(opt) + 1:
            return True
    return False


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
            if not isinstance(w, dict) or _keys(w) != {"name", "run_id", "step"}:
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
        # the model source speed_probe needs per --kind (it refuses every kind but cohere without --model at once): a
        # registry typo here would otherwise surface only on the rented box, as a failed speed item
        sk, n_ckpt = g("speed_kind"), (g("of") is not None) + (
            len(g("weights")) if isinstance(g("weights"), list) else 0)
        n_src = n_ckpt + (g("model") is not None)
        if n_src > 1:
            p.append(f"{where}: a speed item times one model (of, one weights entry or model), got {n_src}")
        elif sk == "cohere" and n_src:
            p.append(f"{where}: speed_kind cohere times the teacher, no model source")
        elif sk in ("aed", "ctc") and n_src != 1:
            p.append(f"{where}: speed_kind {sk} times one student: give of, one weights entry or model")
        elif sk in ("parakeet-ctc", "parakeet-tdt") and g("model") is None:
            p.append(f"{where}: speed_kind {sk} times the Parakeet teacher: give model (its data-repo dir, in "
                     f"extra_dirs), not of or weights")
        elif sk == "whisper" and n_ckpt:
            p.append(f"{where}: speed_kind whisper times a Whisper model: give none (--model <key> in args) or "
                     f"model, not of or weights")
        elif sk == "whisper" and g("model") is None and not _gives_option(g("args"), "--model"):
            # without a model source the queue's argv has no --model, so the Whisper key must come from args
            p.append(f"{where}: speed_kind whisper needs --model <key> in args (or model)")
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
            elif _num(sec) and _int(wd.get("orphan_s")) and not sec >= wd["orphan_s"] + WATCHDOG_POLL_S:
                p.append(f"{w}: seconds {sec} must be >= watchdog.orphan_s {wd['orphan_s']} + the watchdog's poll "
                         f"{WATCHDOG_POLL_S} s (WATCHDOG_POLL_S): the alert comes at its first poll past orphan_s")
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
    if out.get("min_ram_gb") is not None and not (_num(out["min_ram_gb"]) and out["min_ram_gb"] > 0):
        p.append(f"{where}.min_ram_gb {out['min_ram_gb']!r} is not null or a number > 0")
    wd = out.get("watchdog")
    if "watchdog" in out:
        if not isinstance(wd, dict) or _keys(wd) != {"orphan_s", "action"}:
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


def _read_config(root: Path, rel: str, read_json=None):
    """The parsed JSON of the repo-relative path rel: read_json(rel) when given (launch: the file at its sha), else
    root/rel. Raises what the read raises."""
    if read_json is not None:
        return read_json(rel)
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _read_json(root: Path, rel: str, cache: dict, read_json=None):
    """_read_config once per path; a failed read is cached as the exception that says why."""
    if rel not in cache:
        try:
            cache[rel] = _read_config(root, rel, read_json)
        except FileNotFoundError:
            cache[rel] = FileNotFoundError(f"{rel} does not exist " + (
                f"under {root}" if read_json is None else "(read_json)"))
        except Exception as e:  # noqa: BLE001  a git read may raise anything; every failure is a problem, not a crash
            cache[rel] = ValueError(f"{rel}: not readable JSON ({type(e).__name__}: {e})")
    return cache[rel]


def _data_keys(cfg: dict) -> dict:
    return {k: (cfg.get(k, False) if k == "pull_parakeet" else cfg.get(k)) for k in DATA_KEYS}


def _check_files(boxes: dict, root: Path, p: list[str], read_json=None):
    """Every box's data config and item configs exist and are objects; the item configs carry the data config's
    DATA_KEYS values; a train item's family is its config's; without pull_parakeet every stores and train item config
    has the data config's family (the box pulls only that family's train labels)."""
    cache: dict = {}
    for bname, box in boxes.items():
        dc = box.get("data_config")  # a path that is not repo-relative is a problem already; it is never read
        data = _read_json(root, dc, cache, read_json) if _is_rel_path(dc) else None
        if isinstance(data, Exception) or (_is_rel_path(dc) and not isinstance(data, dict)):
            p.append(f"boxes.{bname}.data_config: {data if isinstance(data, Exception) else 'not a JSON object'}")
            data = None
        for it in box.get("items") or []:
            cfg_path = it.get("config") if it.get("kind") in ("stores", "train") else None
            if not _is_rel_path(cfg_path):
                continue
            w = f"boxes.{bname}.items.{it.get('name')}"
            cfg = _read_json(root, cfg_path, cache, read_json)
            if isinstance(cfg, Exception) or not isinstance(cfg, dict):
                p.append(f"{w}.config: {cfg if isinstance(cfg, Exception) else 'not a JSON object'}")
                continue
            family = cfg.get("family") or "aed"  # 04_distill's default
            if data is not None:
                got, want = _data_keys(cfg), _data_keys(data)
                if diff := [k for k in DATA_KEYS if got[k] != want[k]]:
                    p.append(f"{w}: {cfg_path} differs from the box data config {dc} in {diff}")
                # bootstrap pulls what extent.pull_plan(data config) names: both teachers' labels for every stem only
                # with pull_parakeet; else the train labels of the data config's family alone (ctc: parakeet_out,
                # plus teacher_out for the eval stems; aed: teacher_out), so an item of the other family would find
                # its train labels missing on the box, after the paid rebuild
                data_family = data.get("family") or "aed"
                if not data.get("pull_parakeet") and family != data_family:
                    p.append(f"{w}: {cfg_path} trains family {family}, but the box data config {dc} is family "
                             f"{data_family} without pull_parakeet, so the box pulls only {data_family} train labels")
            if it.get("kind") == "train" and family != it.get("family"):
                p.append(f"{w}: family {it.get('family')!r}, but {cfg_path} trains {family!r}")


def _check_chain(cname: str, raw: dict, plain: dict, p: list[str]) -> dict:
    """A chain box (addendum E.1.2-E.1.4 rules 1-3): its normalised raw form (its own keys and the stages' keys, with
    rebuild and gate_by_hours filled) - never the derived spec, which box_spec computes. plain: the filled plain
    boxes."""
    where = f"boxes.{cname}"
    if derived := [k for k in raw if k in _CHAIN_DERIVED]:
        p.append(f"{where}: {derived} are derived from the chain's parts (addendum E.1.5), never written on a chain")
    out = _fill(where, {k: v for k, v in raw.items() if k not in _CHAIN_DERIVED}, _CHAIN_FIELDS, p)
    for k in ("est_hours", "max_hours", "max_dph"):
        if k in out and not (_num(out[k]) and out[k] > 0):
            p.append(f"{where}.{k} {out[k]!r} is not a number > 0")
    if _num(out.get("est_hours")) and _num(out.get("max_hours")) and out["max_hours"] < out["est_hours"]:
        p.append(f"{where}: max_hours {out['max_hours']} < est_hours {out['est_hours']}")
    if not (_num(out.get("extra_gb")) and out["extra_gb"] >= 0):
        p.append(f"{where}.extra_gb {out.get('extra_gb')!r} is not a number >= 0")
    if out.get("gate") is not True:
        p.append(f"{where}.gate {out.get('gate')!r}: a chain's gate must be true (the gate part's check 4 reads the "
                 f"boot's download_gate.json)")
    stages = out.get("chain")
    if not isinstance(stages, list) or len(stages) != 2 or not all(isinstance(s, dict) for s in stages):
        if "chain" in out:
            p.append(f"{where}.chain is not a list of exactly 2 stage objects (the gated stage, then the last one)")
        return out
    filled, seen, gpus = [], [], set()
    for k, st in enumerate(stages):
        w = f"{where}.chain[{k}]"
        if k and (first_only := [x for x in ("gate_box", "gate_by_hours", "max_hours") if x in st]):
            p.append(f"{w}: {first_only} belong to the gated first stage only")
            st = {x: v for x, v in st.items() if x not in first_only}
        fs = _fill(w, st, _STAGE_FIELDS[k], p)
        parts = fs.get("parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(x, str) for x in parts):
            if "parts" in fs:
                p.append(f"{w}.parts {parts!r} is not a non-empty list of registry box names")
            parts = []
        for x in parts:
            if x in seen:
                p.append(f"{w}: box {x!r} is a part twice in the chain (each box runs once)")
            seen.append(x)
            if x in CHAIN_NAMES:
                p.append(f"{w}: part {x!r} is a chain box: a chain's parts are plain registry boxes")
            elif x not in plain:
                p.append(f"{w}: part {x!r} is not a registry box")
            else:
                gpus.add(plain[x].get("gpus"))
        specs = {x: plain[x] for x in parts if x in plain}
        gb = fs.get("gate_box")
        if k == 0:
            mh = fs.get("max_hours")
            if "max_hours" in fs and not (_num(mh) and mh > 0):
                p.append(f"{w}.max_hours {mh!r} is not a number > 0 (hours from first boot to stage 1's end)")
            elif _num(mh) and _num(out.get("max_hours")) and not mh < out["max_hours"]:
                p.append(f"{w}.max_hours {mh} must be < the chain's max_hours {out['max_hours']}")
            if "gate_box" in fs and (not parts or gb != parts[0]):
                p.append(f"{w}.gate_box {gb!r} must be the stage's first part ({parts[0] if parts else None!r})")
            gspec = specs.get(gb)
            if gspec is not None and not (gspec["smoke"] and any(it.get("kind") == "train" for it in gspec["items"])):
                p.append(f"{w}.gate_box {gb!r} must be a smoke box with >= 1 train item: the gate is its built-in "
                         f"checks {GATE_CHECKS[0]}-{GATE_CHECKS[-1]}")
            if fs.get("gate_by_hours") is None and gspec is not None:
                fs["gate_by_hours"] = gspec["max_hours"]
            gbh = fs.get("gate_by_hours")
            if gbh is not None and not (_num(gbh) and gbh > 0 and (not _num(mh) or gbh <= mh)):
                p.append(f"{w}.gate_by_hours {gbh!r}: a number of hours > 0 and <= the stage's max_hours {mh!r}")
        if fs.get("rebuild") is None and parts and parts[0] in specs:
            fs["rebuild"] = specs[parts[0]]["data_config"]
        rb = fs.get("rebuild")
        if specs and rb not in [s["data_config"] for s in specs.values()]:
            p.append(f"{w}.rebuild {rb!r} is not the data config of one of the stage's parts")
        for x, s in specs.items():  # rule 2: only the gate part may inject faults or merely alert
            if k == 0 and x == gb:
                continue
            if s["faults"]:
                p.append(f"{w}: part {x!r} has faults: only the gate part ({gb!r}) may")
            action = (s.get("watchdog") or {}).get("action")
            if action != "stop":
                p.append(f"{w}: part {x!r}'s watchdog action is {action!r}: every part but the gate part runs with "
                         f"the watchdog in stop mode")
        filled.append(fs)
    gb, last_parts = (filled[0].get("gate_box"), filled[1].get("parts") or []) if len(filled) == 2 else (None, [])
    stop_s = [plain[x]["watchdog"].get("orphan_s") for x in last_parts if x in plain
              and isinstance(plain[x].get("watchdog"), dict)]
    if gb in plain and stop_s and all(_int(x) for x in stop_s):  # the mode set_mode switches to once the gate part
        for f in plain[gb].get("faults") or []:  # has returned: never one in which a freeze could stop the box
            if isinstance(f, dict) and f.get("action") == "freeze_controller_hb" and _num(f.get("seconds")) and \
                    not f["seconds"] < max(stop_s):
                p.append(f"{where}: the gate part's freeze {f.get('id')!r} ({f['seconds']} s) must be shorter than the "
                         f"last stage's stop orphan_s {max(stop_s)}: stop mode must never fire on a freeze")
    if len(gpus) > 1:
        p.append(f"{where}: its parts have different gpus {sorted(gpus, key=str)}: one rental has one GPU count")
    last = [plain[x] for x in filled[1].get("parts") or [] if x in plain]
    most = max((s["extra_gb"] for s in last), default=0)
    if last and _num(out.get("extra_gb")) and out["extra_gb"] < most:
        p.append(f"{where}.extra_gb {out['extra_gb']} is below its last stage's parts' ({most})")
    s1 = filled[0].get("max_hours")
    if _num(out.get("max_hours")) and _num(s1) and last:  # rule 3: the last stage fits after stage 1's end
        need = sum(float(s["est_hours"]) for s in last)
        if out["max_hours"] - s1 < need:
            p.append(f"{where}: max_hours {out['max_hours']} - stage 1's {s1} leaves {out['max_hours'] - s1:g} h, "
                     f"below the last stage's est_hours {need:g}")
    out["chain"] = filled
    return out


def _pulls_parakeet(cfg: dict) -> bool:
    """A data config pulls parakeet_out (kitsune.extent pull_plan): family ctc, or pull_parakeet."""
    return (cfg.get("family") or "aed") == "ctc" or bool(cfg.get("pull_parakeet"))


def _check_chain_files(cname: str, spec: dict, plain: dict, root: Path, p: list[str], read_json=None):
    """E.1.4 rule 4: each stage's rebuild config holds every part of the stage (kitsune.extent within, the same data
    and label roots, parakeet_out when a part pulls it), and stage 1's rebuild lies within the last stage's, so the
    last stage's rebuild reuses stage 1's shards."""
    from kitsune import extent  # pyarrow (kitsune.store): only when the files are checked

    cache: dict = {}
    where = f"boxes.{cname}"

    def read(rel):
        cfg = _read_json(root, rel, cache, read_json) if _is_rel_path(rel) else None
        return cfg if isinstance(cfg, dict) else None

    rebuilds = []
    for k, st in enumerate(spec.get("chain") or []):
        rb = st.get("rebuild")
        rcfg = read(rb)
        rebuilds.append(rcfg)
        if rcfg is None:
            p.append(f"{where}.chain[{k}].rebuild {rb!r}: not a readable JSON object")
            continue
        for part in st.get("parts") or []:
            pcfg = read((plain.get(part) or {}).get("data_config"))
            if pcfg is None:  # a missing part data config is the plain box's problem already
                continue
            dc = plain[part]["data_config"]
            try:
                within = extent.within(rcfg, pcfg)
            except Exception as e:  # noqa: BLE001  a malformed extent block is a problem, not a crash
                within = [f"{type(e).__name__}: {e}"]
            if within:
                p.append(f"{where}: part {part}'s extent ({dc}) is not within stage {k + 1}'s rebuild {rb}: {within}")
            if diff := [x for x in _ROOT_KEYS if pcfg.get(x) != rcfg.get(x)]:
                p.append(f"{where}: part {part}'s {dc} differs from stage {k + 1}'s rebuild {rb} in {diff}")
            if _pulls_parakeet(pcfg) and not _pulls_parakeet(rcfg):
                p.append(f"{where}: part {part} needs parakeet_out ({dc}), but stage {k + 1}'s rebuild {rb} does not "
                         f"pull it (family ctc or pull_parakeet)")
    if len(rebuilds) == 2 and all(c is not None for c in rebuilds):
        try:
            within = extent.within(rebuilds[1], rebuilds[0])
        except Exception as e:  # noqa: BLE001
            within = [f"{type(e).__name__}: {e}"]
        if within:
            p.append(f"{where}: stage 1's rebuild is not within the last stage's (whose rebuild reuses its shards): "
                     f"{within}")


def _check(reg, root, check_files: bool, read_json=None) -> tuple[list[str], dict | None]:
    """(problems, the registry with its defaults filled). Chain boxes are split off before the plain boxes are
    checked and are checked against the filled plain boxes; a plain box's of_box / only_if_new_machine never names a
    chain (it has no items and no queue summary of its own items)."""
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
    if unknown := [b for b in boxes if b not in ALL_BOX_NAMES]:
        p.append(f"registry: box(es) {unknown} are not in BOX_NAMES {BOX_NAMES} (or CHAIN_NAMES {CHAIN_NAMES})")
    chains = {b: x for b, x in boxes.items() if isinstance(x, dict) and "chain" in x}
    plain = {b: x for b, x in boxes.items() if b not in chains}
    for b in chains:
        if b in BOX_NAMES:
            p.append(f"boxes.{b}: a plain box (BOX_NAMES) has no `chain`; a chain box's name is one of {CHAIN_NAMES}")
    for b in plain:
        if b in CHAIN_NAMES:
            p.append(f"boxes.{b}: {b} is a chain box (CHAIN_NAMES): its entry needs `chain` (addendum E.1.2)")
    filled: dict = {}
    for bname, box in plain.items():
        f = _check_box(bname, box, plain, p)
        if f is not None:
            filled[bname] = f
    # of_box: a train item of another registry box (a box that is not an object is a problem already)
    for bname, box in filled.items():
        for it in box["items"]:
            ob, of = it.get("of_box"), it.get("of")
            if it.get("kind") not in ("speed", "eval") or not isinstance(ob, str) or ob == bname or of is None:
                continue
            if ob not in plain:
                p.append(f"boxes.{bname}.items.{it.get('name')}: of_box {ob!r} is not a registry box"
                         + (" with items (it is a chain)" if ob in chains else ""))
            elif ob in filled and of not in [x.get("name") for x in filled[ob]["items"] if x.get("kind") == "train"]:
                p.append(f"boxes.{bname}.items.{it.get('name')}: of {of!r} is not a train item of box {ob}")
    plain_filled = dict(filled)
    for cname, raw in chains.items():
        filled[cname] = _check_chain(cname, raw, plain_filled, p)
    out["boxes"] = {b: filled[b] for b in boxes if b in filled}
    if check_files:
        r = Path(root) if root is not None else REPO
        _check_files(plain_filled, r, p, read_json)
        for cname in chains:
            _check_chain_files(cname, filled[cname], plain_filled, r, p, read_json)
    return p, out


class _Registry(dict):
    """A registry as load_registry returns it: the plain registry dict (equal to, and serialised as, the dict it holds)
    that also remembers `root`, the checkout its config paths resolved against. The box readers and registry_problems
    given it without a root read the configs there, so code that loads a registry file once and passes the dict on
    reads that file's checkout, never this repo's by accident. A copy (copy.deepcopy) keeps it; dict(reg) or a JSON
    round trip drops it, and the paths then resolve against this repo again."""
    root: Path | None = None


def _with_root(reg: dict, root) -> dict:
    """reg as a _Registry that remembers root (a shallow copy: its nested objects are reg's)."""
    out = _Registry(reg)
    out.root = Path(root) if root is not None else None
    return out


def _root_of(reg, root):
    """The root reg's paths resolve against: root when given, else the one a loaded registry remembers, else None
    (this repo)."""
    return root if root is not None else getattr(reg, "root", None)


def registry_problems(reg: dict, *, root=None, check_files=True, read_json=None) -> list[str]:
    """Why reg is not a valid registry (contract 2.3); empty: it is. root: the checkout its paths resolve against
    (default the one a loaded registry remembers, else this one); check_files False skips reading the data and item
    configs; read_json(rel) -> the parsed JSON of a repo-relative path reads them instead of root (launch: the files
    at the sha it rents, e.g. lambda rel: json.loads(git_show(sha, rel)))."""
    return _check(reg, _root_of(reg, root), check_files, read_json)[0]


def _registry_root(path: Path) -> Path:
    """The checkout a registry file belongs to: <X> for <X>/configs/full/boxes.json, else this repo."""
    parts = path.resolve().parts
    return path.resolve().parents[2] if tuple(parts[-3:]) == tuple(BOXES_FILE.split("/")) else REPO


def _load(path_or_dict, root, check_files: bool, read_json=None) -> tuple[dict, Path]:
    if isinstance(path_or_dict, dict):
        reg, where = path_or_dict, "registry"
        root = _root_of(path_or_dict, root)
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
    problems, filled = _check(reg, root, check_files, read_json)
    if problems:
        raise RegistryError(f"{where} is not a valid box registry:\n  " + "\n  ".join(problems), problems)
    return _with_root(filled, root), root


def load_registry(path_or_dict=None, *, root=None, check_files=True, read_json=None) -> dict:
    """The validated registry with its defaults filled (a copy). None reads $KITSUNE_FULL_REGISTRY, else
    <root or this repo>/configs/full/boxes.json; a path reads that file; a dict (launch: the JSON at the launched sha)
    is validated the same way. RegistryError when it is missing or invalid. Paths resolve against root, by default the
    checkout the file is in (<X> for <X>/configs/full/boxes.json), for a dict the root a loaded registry remembers,
    else this repo; read_json(rel) reads the configs instead (registry_problems). The dict returned remembers its root
    (_Registry), so box_students(box, reg) and student_checks read the configs where this call did."""
    return _load(path_or_dict, root, check_files, read_json)[0]


def _resolved(registry, root, read_json=None) -> tuple[dict, Path]:
    """A loaded registry and the root its paths resolve against: None loads the file with every check (the configs
    read with read_json when given); a given registry (loaded or raw) is validated and filled again without reading
    files (idempotent, cheap), against root, else the root it remembers, else this repo."""
    if registry is None:
        return _load(None, root, True, read_json)
    return _load(registry, root, False)


def _chain_entry(box, reg: dict) -> dict | None:
    """The chain box's normalised entry in a filled registry, or None (a plain box, or no such box)."""
    b = reg["boxes"].get(box)
    return b if isinstance(b, dict) and "chain" in b else None


def _no_box(box, reg: dict) -> RegistryError:
    return RegistryError(f"box {box!r} is not in the registry (it has {', '.join(reg['boxes']) or 'none'})")


def is_chain(box, registry=None) -> bool:
    """box is a chain box of the registry (its entry has `chain`: addendum E.1.2)."""
    reg, _ = _resolved(registry, None)
    return _chain_entry(box, reg) is not None


def _uniq(xs) -> list:
    return list(dict.fromkeys(xs))


def chain_stages(box, registry=None) -> list[dict]:
    """A chain's filled stages (E.1.5 `chain`): {stage (1, 2), parts, gate_box (stage 1; else None), gate_by_hours,
    max_hours (stage 1's sub-deadline in hours from first boot; else None), rebuild (the extent the stage's bootstrap
    rebuilds), data_configs {part: its data config}, watchdog}. The watchdog of stage 1 is its gate part's (smoke A:
    600 s, alert, so its heartbeat fault is seen); every later part runs in stop mode (E.1.4 rule 2), so stage 2's is
    {orphan_s: the largest of its parts', action: stop} - also the mode the controller switches the watchdog to once the
    gate part has ended. RegistryError for a box that is not a chain."""
    reg, _ = _resolved(registry, None)
    c = _chain_entry(box, reg)
    if c is None:
        if box not in reg["boxes"]:
            raise _no_box(box, reg)
        raise RegistryError(f"box {box!r} is not a chain box")
    out = []
    for k, st in enumerate(c["chain"]):
        parts = list(st["parts"])
        specs = {x: reg["boxes"][x] for x in parts}
        if k == 0:
            wd = dict(specs[st["gate_box"]]["watchdog"])
        else:
            wd = {"orphan_s": max(int(s["watchdog"]["orphan_s"]) for s in specs.values()), "action": "stop"}
        out.append(dict(stage=k + 1, parts=parts, gate_box=st.get("gate_box"), gate_by_hours=st.get("gate_by_hours"),
                        max_hours=st.get("max_hours"), rebuild=st["rebuild"],
                        data_configs={x: specs[x]["data_config"] for x in parts}, watchdog=wd))
    return out


def chain_resume_hint(box: str, last_part: str = "p01") -> str:
    """Why a chain is not resumed as a chain, and what the owner runs instead (addendum E.8): launch --resume and
    full_queue resume-pull print it when they refuse a chain box."""
    return "\n".join([
        f"a chain box ({box}) is not resumed as a chain. What to run, by where it died "
        f"({box_summary_path(box)} in the runs repo says: gate, stage, parts.{last_part}):",
        "  - in stage 1 (gate null or failed): the chain again, fresh (a smoke is cheap; new stamps, the same Hub "
        "paths)",
        f"  - in stage 2 after the {last_part} part started on that rental (parts.{last_part}.status not pending, and "
        f"{box_summary_path(last_part)} has the chain's container_id and started == parts.{last_part}.queue_started): "
        f"vast/launch.py --job full --box {last_part} --resume [--resume-reset ...] --machine <new id> --max-hours "
        f"<left + setup> --scratch-repo ... (the unchanged box-{last_part} resume)",
        f"  - in stage 2 before {last_part} started (during the stage-2 bootstrap): --box {last_part} fresh, or the "
        f"chain fresh (the owner decides: the gate passed on the old machine only)"])


def part_state_dir(part, state=None) -> Path:
    """<state>/chain/<part>: a chain part's queue state dir (its queue.json, summary, verdict, logs, events, hb/)."""
    if not isinstance(part, str) or not re.fullmatch(ITEM_RE, part):
        raise ValueError(f"part name {part!r} does not match {ITEM_RE}")
    return Path(state if state is not None else state_dir()) / CHAIN_DIR / part


def _selection_files(root: Path, rel: str, read_json=None) -> list[str]:
    """The data-repo files a part whose data config is not its stage's rebuild needs for its own selection: the
    selection, its sidecar, and kitsune.devslice.selection_files (the frozen manifest of a full_study one,
    kitsune.prereg.study_files of a study one)."""
    import importlib

    try:
        cfg = _read_config(root, rel, read_json)
    except Exception as e:  # noqa: BLE001  a git read may raise anything
        raise RegistryError(f"{rel}: cannot read it ({type(e).__name__}: {e})") from None
    sel = cfg.get("selection") if isinstance(cfg, dict) else None
    if not isinstance(sel, str) or not sel:
        raise RegistryError(f"{rel} names no selection")
    devslice = importlib.import_module("kitsune.devslice")  # stdlib only; imported once it is needed
    return _uniq([sel, devslice.sidecar_path(sel), *devslice.selection_files(cfg)])


def stage_view(box, stage: int, registry=None, root=None, *, read_json=None) -> dict:
    """What one stage of a chain pulls and runs (E.1.6): {stage, parts, gate_box, rebuild, data_configs, students and
    ctc_students (the union over its parts, in part order), extra_files (the parts' extra files, plus the selection
    files of every part whose data config is not the stage's rebuild: stage 1's smoke-b adds study_1000h.parquet and
    its sidecar), extra_dirs (the union), timed_states (any part's), watchdog}. The item configs are read as
    box_students reads them (root, or read_json)."""
    reg, r = _resolved(registry, root, read_json)
    stages = chain_stages(box, reg)
    if not (_int(stage) and 1 <= stage <= len(stages)):
        raise RegistryError(f"chain {box} has stages 1..{len(stages)}, not {stage!r}")
    st = stages[stage - 1]
    students, ctc, files, dirs, timed = [], [], [], [], False
    for part in st["parts"]:
        spec = reg["boxes"][part]
        students += box_students(part, reg, r, read_json=read_json)
        ctc += box_ctc_students(part, reg, r, read_json=read_json)
        files += spec["extra_files"]
        dirs += spec["extra_dirs"]
        timed = timed or bool(spec["timed_states"])
        if spec["data_config"] != st["rebuild"]:
            files += _selection_files(r, spec["data_config"], read_json)
    return dict(stage=stage, parts=list(st["parts"]), gate_box=st["gate_box"], rebuild=st["rebuild"],
                data_configs=dict(st["data_configs"]), students=_uniq(students), ctc_students=_uniq(ctc),
                extra_files=_uniq(files), extra_dirs=_uniq(dirs), timed_states=timed, watchdog=dict(st["watchdog"]))


def _stages_of(box, reg: dict, stage) -> list[int]:
    """The stages a chain reader serves: stage N alone, or (None) every stage (launch checks the union)."""
    n = len(_chain_entry(box, reg)["chain"])
    if stage is None:
        return list(range(1, n + 1))
    if not (_int(stage) and 1 <= stage <= n):
        raise RegistryError(f"chain {box} has stages 1..{n}, not {stage!r}")
    return [stage]


def _chain_spec(box, reg: dict, r: Path, read_json=None) -> dict:
    """A chain's derived box spec (E.1.5), computed and never stored: the parts' GPU count, the last stage's rebuild
    as the data config (launch sizes the disk and the gate on it), the chain's own hours, price, extra disk and gate,
    stage 1's watchdog (launch's env), the largest deadline reserve of the last stage, timed states if any part keeps
    them, the union of both stage views' extra files and dirs, the largest min_ram_gb of its parts (None when none
    sets one), no items (the controller runs its parts' queues)."""
    c = _chain_entry(box, reg)
    stages = chain_stages(box, reg)
    views = [stage_view(box, s["stage"], reg, r, read_json=read_json) for s in stages]
    last = [reg["boxes"][x] for x in stages[-1]["parts"]]
    return dict(gpus=reg["boxes"][stages[0]["parts"][0]]["gpus"], data_config=stages[-1]["rebuild"],
                est_hours=c["est_hours"], max_hours=c["max_hours"], max_dph=c["max_dph"], extra_gb=c["extra_gb"],
                gate=c["gate"], watchdog=dict(stages[0]["watchdog"]),
                deadline_reserve_min=max(s["deadline_reserve_min"] for s in last),
                min_ram_gb=max((reg["boxes"][x].get("min_ram_gb") for st in stages for x in st["parts"]
                                if reg["boxes"][x].get("min_ram_gb") is not None), default=None),
                timed_states=any(v["timed_states"] for v in views),
                extra_files=_uniq(f for v in views for f in v["extra_files"]),
                extra_dirs=_uniq(d for v in views for d in v["extra_dirs"]), smoke=False, faults=[], items=[],
                max_attempts=4, chain=stages)


def box_spec(box, registry=None, *, root=None, read_json=None) -> dict:
    """The box's spec with its defaults filled; a chain's derived spec (_chain_spec: it reads the parts' configs under
    root, or with read_json). RegistryError for a box the registry does not have."""
    reg, r = _resolved(registry, root, read_json)
    if box not in reg["boxes"]:
        raise _no_box(box, reg)
    if _chain_entry(box, reg) is not None:
        return _chain_spec(box, reg, r, read_json)
    return copy.deepcopy(reg["boxes"][box])


def _plain_spec(box, reg: dict, what: str) -> dict:
    """A plain box's spec; RegistryError for a chain (its parts' queues run its items: the chain controller)."""
    if _chain_entry(box, reg) is not None:
        raise RegistryError(f"box {box} is a chain box: {what} - a chain box runs through the chain controller "
                            f"(kitsune/full_queue.py ChainController)")
    if box not in reg["boxes"]:
        raise _no_box(box, reg)
    return copy.deepcopy(reg["boxes"][box])


def box_items(box, registry=None) -> list[dict]:
    """The box's items, in registry order, with their defaults (and implicit needs) filled. RegistryError for a chain:
    a chain box runs through the chain controller."""
    reg, _ = _resolved(registry, None)
    return _plain_spec(box, reg, "it has no items of its own")["items"]


def train_items(box, registry=None) -> list[dict]:
    return [it for it in box_items(box, registry) if it["kind"] == "train"]


def box_configs(box, registry=None) -> list[str]:
    """Every config the box reads, repo-relative and once each: its data config, its stores and train items' configs,
    and the registry file itself (launch checks they are committed at the sha it rents). A chain: every part's, both
    stages' rebuild configs and the registry."""
    reg, _ = _resolved(registry, None)
    c = _chain_entry(box, reg)
    if c is not None:
        parts = [x for st in c["chain"] for x in st["parts"]]
        return _uniq([*(cfg for x in parts for cfg in box_configs(x, reg)), *(st["rebuild"] for st in c["chain"]),
                      BOXES_FILE])
    spec = _plain_spec(box, reg, "")
    out = [spec["data_config"], *(it["config"] for it in spec["items"] if it["kind"] in ("stores", "train")),
           BOXES_FILE]
    return _uniq(out)


def _config_student(root: Path, rel: str, read_json=None) -> str:
    try:
        cfg = _read_config(root, rel, read_json)
    except Exception as e:  # noqa: BLE001  a git read may raise anything; it refuses like a missing file
        where = "with read_json" if read_json is not None else f"under {root}"
        raise RegistryError(f"{rel}: cannot read it {where} ({type(e).__name__}: {e})") from None
    s = cfg.get("student") if isinstance(cfg, dict) else None
    if not isinstance(s, str) or not s:
        raise RegistryError(f"{rel} names no student")
    return s.rstrip("/")


def _chain_view_union(box, reg: dict, r: Path, read_json, stage, key: str) -> list:
    """The union of a chain's stage views' `key` (stage None: every stage; else that one)."""
    return _uniq(x for s in _stages_of(box, reg, stage) for x in stage_view(box, s, reg, r, read_json=read_json)[key])


def box_students(box, registry=None, root=None, *, read_json=None, stage=None) -> list[str]:
    """The student dirs (repo paths) of the box's train items' configs, in item order, once each: what bootstrap pulls
    and check-students checks. The configs are read under root (default: the root a loaded registry remembers, else
    this repo), or with read_json(rel) when given (launch: at its sha). A chain: stage N's view (bootstrap and
    check-students of that stage), or with stage None the union of its stages (launch); stage is ignored on a plain
    box."""
    reg, r = _resolved(registry, root, read_json)
    if _chain_entry(box, reg) is not None:
        return _chain_view_union(box, reg, r, read_json, stage, "students")
    items = _plain_spec(box, reg, "")["items"]
    return _uniq(_config_student(r, it["config"], read_json) for it in items if it["kind"] == "train")


def box_ctc_students(box, registry=None, root=None, *, read_json=None, stage=None) -> list[str]:
    """The Parakeet-family ones among box_students (family ctc): they carry the CC-BY-4.0 attribution card. A chain's
    stage as box_students'."""
    reg, r = _resolved(registry, root, read_json)
    if _chain_entry(box, reg) is not None:
        return _chain_view_union(box, reg, r, read_json, stage, "ctc_students")
    items = _plain_spec(box, reg, "")["items"]
    return _uniq(_config_student(r, it["config"], read_json) for it in items
                 if it["kind"] == "train" and it["family"] == "ctc")


def box_extra_files(box, registry=None, *, stage=None, root=None, read_json=None) -> list[str]:
    """The data-repo files the box pulls and requires besides the labels and students (e.g. the frozen manifest and the
    selection sidecar). A chain's stage as box_students' (its view adds the selection files of a part whose data
    config is not its stage's rebuild)."""
    reg, r = _resolved(registry, root, read_json)
    if _chain_entry(box, reg) is not None:
        return _chain_view_union(box, reg, r, read_json, stage, "extra_files")
    return list(_plain_spec(box, reg, "")["extra_files"])


def box_extra_dirs(box, registry=None, *, stage=None, root=None, read_json=None) -> list[str]:
    """The data-repo dirs the box pulls and requires (e.g. the Parakeet teacher for its speed probe). A chain's stage
    as box_students'."""
    reg, r = _resolved(registry, root, read_json)
    if _chain_entry(box, reg) is not None:
        return _chain_view_union(box, reg, r, read_json, stage, "extra_dirs")
    return list(_plain_spec(box, reg, "")["extra_dirs"])


def student_checks(box, read_meta, registry=None, root=None, *, read_json=None, stage=None) -> list[str]:
    """Why the students the box trains are not the registered builds: each train item's config must train its
    study_run's registered student dir, and read_meta(student dir) -> its student_meta.json (a dict) must pass
    kitsune.prereg.student_problems(study_run, meta). A meta it cannot read is a problem too. Empty: every student is
    the registered build (launch checks the data repo's copies, bootstrap the pulled ones). The item configs are read
    as box_students reads them (root, or read_json at launch's sha). A chain: its stage's parts (stage None: all)."""
    from kitsune import prereg
    reg, r = _resolved(registry, root, read_json)
    c = _chain_entry(box, reg)
    if c is not None:
        parts = [x for s in _stages_of(box, reg, stage) for x in c["chain"][s - 1]["parts"]]
        return _uniq(x for part in parts for x in student_checks(part, read_meta, reg, r, read_json=read_json))
    problems, metas = [], {}
    for it in _plain_spec(box, reg, "")["items"]:
        if it["kind"] != "train":
            continue
        run = it["study_run"]
        s, want = _config_student(r, it["config"], read_json), prereg.RUNS[run]["student"]
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


def box_env(box, registry=None, stage: int = 1) -> dict[str, str]:
    """The registry's part of the box env (launch): KITSUNE_N_GPUS and the watchdog's heartbeat file (train_hb), orphan
    limit and action. A chain: that stage's watchdog, KITSUNE_CHAIN_STAGE, and in stage 1 the watchdog's hand-over
    bound KITSUNE_WATCHDOG_HANDOVER_S = 3600 x (gate_by_hours + 0.5): with no mode file from the controller by first
    boot + that, the watchdog syncs and stops the box (vast/watchdog.sh). stage is ignored on a plain box."""
    reg, _ = _resolved(registry, None)
    if _chain_entry(box, reg) is None:
        spec = _plain_spec(box, reg, "")
        return {ENV_N_GPUS: str(spec["gpus"]), ENV_WATCHDOG_HB_FILE: TRAIN_HB,
                ENV_WATCHDOG_ORPHAN_S: str(spec["watchdog"]["orphan_s"]),
                ENV_WATCHDOG_ORPHAN_ACTION: spec["watchdog"]["action"]}
    (s,) = _stages_of(box, reg, 1 if stage is None else stage)
    st = chain_stages(box, reg)[s - 1]
    env = {ENV_N_GPUS: str(reg["boxes"][st["parts"][0]]["gpus"]), ENV_WATCHDOG_HB_FILE: TRAIN_HB,
           ENV_WATCHDOG_ORPHAN_S: str(st["watchdog"]["orphan_s"]),
           ENV_WATCHDOG_ORPHAN_ACTION: st["watchdog"]["action"], ENV_CHAIN_STAGE: str(s)}
    if s == 1:
        env[ENV_WATCHDOG_HANDOVER_S] = str(int(3600 * (float(st["gate_by_hours"]) + 0.5)))
    return env


# ================================================================================================ CLI

EXIT_OK, EXIT_REFUSED = 0, 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kitsune.fullrun", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("students", "extra-files", "extra-dirs", "check-students", "show"):
        sp = sub.add_parser(c)
        sp.add_argument("--box", default=os.environ.get(ENV_BOX), choices=ALL_BOX_NAMES)
        sp.add_argument("--stage", type=int, default=None,
                        help=f"a chain box's stage (default ${ENV_CHAIN_STAGE}, else 1): its stage view (ignored on a "
                             f"plain box)")
        sp.add_argument("--root", default=None,
                        help="the checkout: the registry and configs are read there, and check-students reads the "
                             "pulled student dirs there (default: this repo)")
    args = ap.parse_args(argv)
    if not args.box:
        ap.error(f"--box (or {ENV_BOX}) is required")
    root = Path(args.root) if args.root else None
    try:
        reg, r = _resolved(None, root)
        chain = _chain_entry(args.box, reg) is not None
        stage = None
        if chain:
            env_stage = os.environ.get(ENV_CHAIN_STAGE) or "1"
            if args.stage is None and not re.fullmatch(r"[1-9][0-9]*", env_stage):
                raise RegistryError(f"{ENV_CHAIN_STAGE} {env_stage!r} is not a stage number")
            stage = args.stage if args.stage is not None else int(env_stage)
        if args.cmd == "students":
            lines = box_students(args.box, reg, r, stage=stage)
        elif args.cmd == "extra-files":
            lines = box_extra_files(args.box, reg, stage=stage, root=r)
        elif args.cmd == "extra-dirs":
            lines = box_extra_dirs(args.box, reg, stage=stage, root=r)
        elif args.cmd == "show":
            doc = {"box": args.box, "env": box_env(args.box, reg, stage=stage or 1),
                   "configs": box_configs(args.box, reg), "students": box_students(args.box, reg, r, stage=stage),
                   "spec": box_spec(args.box, reg, root=r)}
            if chain:  # the derived spec plus both stage views
                doc["stages"] = {str(s["stage"]): stage_view(args.box, s["stage"], reg, r)
                                 for s in chain_stages(args.box, reg)}
            print(json.dumps(doc, indent=2, ensure_ascii=False))
            return EXIT_OK
        else:  # check-students: vast/bootstrap.sh after the pull (a chain: only this stage's students are on disk)
            problems = student_checks(args.box, lambda s: json.loads(
                (r / s / "student_meta.json").read_text(encoding="utf-8")), reg, r, stage=stage)
            for x in problems:
                print(f"student refused: {x}", file=sys.stderr)
            if not problems:
                what = f" (stage {stage})" if chain else ""
                print(f"box {args.box}{what}: {len(box_students(args.box, reg, r, stage=stage))} student(s) are the "
                      f"registered builds")
            return EXIT_REFUSED if problems else EXIT_OK
    except RegistryError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if lines:
        print("\n".join(lines))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
