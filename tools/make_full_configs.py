"""Generate the full-data runs' configs, configs/full/<name>.json, from the size study's generator (build contract 7).

Trainer configs (the four full runs and their smokes). Every one is the study run's config as
tools/make_study_configs.py makes it (run_config(study_run): the trainer part of configs/next_run_template.json,
study/data.json, the study's settings; the queue fills max_steps, lr and the mini cadence left null), then:
  1. the data keys (kitsune.fullrun DATA_KEYS) replaced whole by the run's data block: kitsune.fullrun FULL_DATA (the
     full selection labels/full/selections/full_study/full.parquet with its dev slice, the full extent, the full_study
     recipe) or SMOKE_DATA (the smoke selection on the study extent: a seeded 100 h train draw with its own dev slice),
     with pull_parakeet true (both label roots) except on full-p01, which box 1 trains on the Parakeet labels alone
     (the key left out: the box pulls parakeet_out for every stem and teacher_out for the eval stems only, fix 9);
  2. the student's row of FULL_RUNS: the epochs clock (schedule.epochs 3/3/4/4, no max_steps), the study's warm-up,
     optim.lr, the micro-batch and the step audio (the study's REALISED audio per step at the new micro-batch,
     measured with tools/full_plan.py: FULL_RUNS' step values realise it within 0.01 % on full.parquet, PLAN_FILE),
     eval.dev.greedy (the P students' argmax is free, T-0.6B's decode is not), schedule.end_reserve_min (the
     default 30; full-t06 55, so its 45 min M4 readout still starts after a run the deadline cooldown shortened:
     READOUT_RESERVE);
  3. COMMON: the deadline cooldown (4d), the memory probe's extended passes and 50 OOM skips, the 5090's bf16 peak for
     the MFU, a complete eval every epoch and a mini eval every 2,000 steps, the dev slice (600 rows per source, seed
     1234) every 0.1 epoch, local full states every 30 min, the pre_cooldown state uploaded and a timed state to the
     scratch repo every 120 min (the queue sets hf.scratch_repo), log syncs every 60 min, full step rows every 10th
     step, metrics/scalars.parquet at the close only, and the early stop on dev_ce (mean of 5, patience 6, at least 5
     values, action cooldown, the test sets refused). Everything else stays the study's (BN, L2-SP 0, aux-CTC 0, w_ctc
     0.8, SpecAugment, the smoke block, seed 1234, verdict v2).
  smoke-<x> is full-<x> with SMOKE (the smoke data, the same epochs: the 100 h draw is the budget; no complete evals
  in the loop and a 100-row greedy subset at the end; minis every 100 steps; 60 dev rows per source; full states every
  5 min and timed states every 10 min (smoke-p01: 3 and 3, so the wipe fault's second attempt uploads one); log syncs
  every 10 min (smoke-p01: 5) with every step row full; a 2 min end reserve and 4d checked every 10 steps over a
  20-step window), memory.probe_shapes = the full data's worst micro-batches of the matching student (PLAN_FILE, so
  the smoke's memory probe sees box 1's and box 2's worst shapes), and on smoke-t06 and smoke-p03 a forced early stop
  (early_stop.min_delta_abs 1e9: no dev eval can improve by that; smoke-t06 with patience 8, so it still ends after
  300 steps or more, smoke check 2's VRAM window).

Data configs (launch's --config and bootstrap's KITSUNE_CONFIG: what a box rebuilds and pulls; no trainer settings):
  data-p01      FULL_DATA + family ctc (box 1: CTC only, no pull_parakeet)
  data-full     FULL_DATA + pull_parakeet true (box 2: both families)
  data-smoke    SMOKE_DATA + pull_parakeet true (smoke A)
  data-smoke-b  study/data.json's data keys verbatim (smoke B: the frozen study selection, so every study-weight eval,
                speed probe and Whisper eval of the box shares one eval store)

Inputs besides the two generators' own: PLAN_FILE (configs/full/plan/full_study.json, in a folder of its own so every
configs/full/*.json is a config), the JSON tools/full_plan.py wrote for full.parquet and smoke.parquet (--import-plan
records a new measurement; the local selection paths become the repo paths). The generator refuses a record measured
with other step values than FULL_RUNS'. It gives the smoke configs' probe shapes, and the numbers the hand-written
registry must carry (registry_numbers), which tests/test_full_configs.py checks: the full runs' plan_total_steps (the
smoke train items', for smoke check 3), their plan_hours = plan v3's hours at the step count measured on full.parquet
(PLAN_V3: hours x measured T / plan T, 2 decimals; also the full train items' max_hours until box 1's speed replaces
them), and F4's seconds, the deadline fault of smoke-p005 (deadline_fault_s). After a rebuilt selection: tools/
full_plan.py --json on full.parquet and on smoke.parquet, --import-plan with both (it prints what boxes.json must be
changed in), then those numbers into boxes.json by hand, then --check and the tests.

Box full's hours (box 2): SPEED_FILE (configs/full/plan/box2_hours.json, --import-speed) records the measured speeds
box2_hours projects them from: smoke A's per-student s/step (smoke verdict check 3) and, after box 1, box 1's s/step,
in-run overhead, CTC store and bootstrap hours (tools/box1_go.py --json). It prints, and --check holds, box full's
train items' max_hours and its est_hours / max_hours (smoke only: provisional; the PR after box 1 adds box 1's part,
contract 7).

configs/full/boxes.json, the box registry (kitsune/fullrun.py), is hand-written; --check validates it with
fullrun.registry_problems (every data and item config present, its data keys equal to its box's data config's; the
chained box p01-chain's rules of contract addendum E.1.4: its parts, hours and watchdogs, each part's extent within its
stage's rebuild config and stage 1's within stage 2's), checks that every readout of a full box fits in its run's end
reserve (readout_reserve_problems; a chain has no items, its parts are checked) and that the numbers bound to the plan
record equal it (registry_drift: smoke A's plan_total_steps / plan_hours, F4's seconds, box 1's train hours; box
full's hours to the speed record; the chain's own hours are addendum E.6's and tests/test_full_configs.py checks them
against its parts).

Usage:
  python tools/make_full_configs.py                  # write configs/full/*.json (and remove stale generated ones)
  python tools/make_full_configs.py --check          # exit 1 if a committed file differs or the registry is invalid
  python tools/make_full_configs.py --import-plan full_plan_full.json full_plan_smoke.json   # a new measurement
  python tools/make_full_configs.py --import-speed --smoke-verdict smoke_verdict.json      # box 2's hours (smoke A)
  python tools/make_full_configs.py --import-speed --box1-go box1_go.json                  # ... and box 1's
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import make_study_configs as study  # noqa: E402  (the size study's generator: run_config, _merge, render)

from kitsune import fullrun  # noqa: E402

OUT_DIR = ROOT / "configs" / "full"
BOXES = "boxes.json"  # hand-written (kitsune.fullrun.BOXES_FILE); never written or removed here
PLAN_FILE = "plan/full_study.json"  # under OUT_DIR: tools/full_plan.py on full.parquet and smoke.parquet
RUNS_REPO = study.RUNS_REPO
STUDY_DATA = ROOT / "study" / "data.json"

# the full students (contract 7): their study run, schedule.epochs, warm-up (the study's, kitsune.prereg), optim.lr,
# batch.micro_audio_s / step_audio_s (DECISIONS C10: the study's realised audio per step, tools/full_plan.py),
# eval.dev.greedy, whether the run pulls both label roots (full-p01: box 1 has the Parakeet labels only) and
# schedule.end_reserve_min (end_reserve: the trainer's default 30, full-t06 55; READOUT_RESERVE below)
FULL_RUNS = {
    "t06": dict(study_run="study-t06", epochs=3, warmup=300, lr=2e-4, micro=450, step=1730, dev_greedy=False,
                pull_parakeet=True, end_reserve=55),
    "p03": dict(study_run="study-p03", epochs=3, warmup=300, lr=2e-4, micro=600, step=1350, dev_greedy=True,
                pull_parakeet=True, end_reserve=30),
    "p01": dict(study_run="study-p01", epochs=4, warmup=1000, lr=1e-3, micro=1600, step=1500, dev_greedy=True,
                pull_parakeet=False, end_reserve=30),
    "p005": dict(study_run="study-p005", epochs=4, warmup=1000, lr=1e-3, micro=1600, step=1500, dev_greedy=True,
                 pull_parakeet=True, end_reserve=30),
}
# READOUT_RESERVE. A readout runs right after its training item on the same GPU (Resolution 25) under the same
# KITSUNE_DEADLINE (kitsune.full_queue item_deadline: the box deadline less deadline_reserve_min). After a run that the
# deadline cooldown (4d) shortened, 04_distill.fit_epochs_deadline has planned the end phase to finish
# schedule.end_reserve_min before that deadline (plus the final eval's and the last dev eval's estimates), so the
# readout gets what the trainer's final saves and uploads leave of the end reserve, and the no-start rule skips a
# droppable readout whose max_hours need more (item_not_started): box 2 would lose T-0.6B's binding M4 and, through
# their needs, its 7 quantised readouts and speed-full-t06. So every full run's end reserve holds its readout's
# max_hours (boxes.json: m4-full-t06 0.75 h, the P readouts 0.3 h) + END_PHASE_SLACK_MIN of its own end phase and the
# queue's hand-off [X]: t06 45 + 10 = 55 (the default 30 would leave about 28 min), the P runs 18 + 10 <= 30.
# --check refuses a registry whose readout no longer fits (readout_reserve_problems), e.g. after the PR that refreshes
# box full's hours from box 1. It costs only when 4d acts: the shortened run then trains 25 min less. Smoke boxes are
# exempt: contract 7's 2 min smoke reserve is what lets F4 make 4d act within the smoke, and a smoke run that 4d
# shortens without a fault is a late smoke.
END_PHASE_SLACK_MIN = 10
DEV_PER_SOURCE, DEV_SEED, SMOKE_DEV_PER_SOURCE = 600, 1234, 60
COMMON = {
    "schedule": {"clock": "epochs", "max_steps": None, "deadline_cooldown": True},
    "memory": {"max_oom_skips": 50, "probe_extended": True},
    "perf": {"peak_tflops": 209.5},  # RTX 5090 dense bf16 (the MFU's denominator)
    "eval": {"every_min": None, "every_steps": None, "full_every_epochs": 1, "full_at_fracs": None,
             "mini": {"every_steps": 2000},
             "dev": {"every_epochs": 0.1, "every_steps": None, "per_source": DEV_PER_SOURCE, "seed": DEV_SEED,
                     "at_start": True, "at_end": True}},
    "ckpt": {"weights_every_min": None, "full_local_every_min": 30, "keep_local": 2, "full_at_fracs": None,
             "weights_at_fracs": None, "upload_full_at": ["pre_cooldown"], "upload_full_every_min": 120},
    "log": {"sync_every_min": 60, "full_scalars_every_steps": 10, "scalars_parquet": "close"},
    "hf": {"output_repo": RUNS_REPO, "scratch_repo": None},
    "early_stop": {"enabled": True, "metric": "dev_ce", "patience": 6, "min_delta_rel": 0.005, "min_delta_abs": 0.0,
                   "min_evals": 5, "floor": None, "action": "cooldown", "smooth": 5, "allow_test_sets": False},
}
SMOKE = {
    "eval": {"full_every_epochs": None, "final_full_greedy": False, "greedy_subset": 100, "mini": {"every_steps": 100},
             "dev": {"per_source": SMOKE_DEV_PER_SOURCE}},
    "ckpt": {"full_local_every_min": 5, "upload_full_every_min": 10},
    "log": {"sync_every_min": 10, "full_scalars_every_steps": 1},
    "schedule": {"end_reserve_min": 2, "deadline_check_steps": 10, "deadline_window_steps": 20},
}
# smoke-p01 carries the wipe fault F3 (after a timed upload of its second attempt): its states and syncs come faster
SMOKE_OVER = {"p01": {"ckpt": {"full_local_every_min": 3, "upload_full_every_min": 3}, "log": {"sync_every_min": 5}}}
# the forced early stop, one per family (smoke check 10); >= kitsune.full_queue FORCED_MIN_DELTA_ABS
FORCED = {"t06": {"early_stop": {"min_delta_abs": 1e9, "patience": 8}},
          "p03": {"early_stop": {"min_delta_abs": 1e9}}}

# plan v3 (docs_plan_v3_live.md 5, calc_v2): each full run's hours at its max epochs (m54650 speeds, dev checks
# included) and the step count they were computed for [X]. The registry's plan_hours rescale them to the step count
# measured on full.parquet (the hours per step are the plan's; smoke check 3 compares the smoke's measured s/step with
# them)
PLAN_V3 = {"t06": (35.19, 73452), "p03": (22.29, 109608), "p01": (13.56, 110520), "p005": (9.92, 110528)}
DEADLINE_FAULT_ITEM, DEADLINE_FAULT_STUDENT = "smoke-p005", "p005"
DEADLINE_FAULT_SHARE, DEADLINE_FAULT_ROUND_S = 0.5, 10  # contract 7: S <= 0.5 x the projected run time

# BOX 2'S HOURS (contract 7 "item hours after box 1", 12.5, Resolution 31). Box full's train items' max_hours and its
# est_hours / max_hours are projected from MEASURED speeds, recorded in SPEED_FILE (written by --import-speed, never by
# hand): smoke A's median steady s/step per student (smoke verdict check 3's sec_per_step, the only measurement of
# T-0.6B, P-0.3B and P-0.05B at the full configs' batches) and, once box 1 has ended, box 1's own (tools/box1_go.py
# --json). Box 1 trains P-0.1B only, so the contract's refresh is a transfer anyway: every student's smoke s/step x r,
# r = box 1's steady s/step / smoke A's of P-0.1B (the host and full-data effect; 1 without box 1), and box 1's in-run
# overhead o replaces the plan's 0.08. Until box 1's part exists the record is smoke-only and boxes.json says
# "provisional" (launch prints a warning). The owner's box-2 preparation (DECISIONS F, 2026-10-01) refreshed them from
# the smoke before box 1 ended, so box 2's cap and cost line no longer carry plan v3's step times, which smoke check 3
# found 1.6-1.9x too slow; the PR after box 1 adds box 1's part (contract 7's own step).
# The run model is plan v3's (calc_v3.run_h): steps x s/step x (1 + o) + the run's fixed time (start-up, memory probe,
# first and final complete evals, end phase) + its dev checks (every COMMON eval.dev.every_epochs epoch).
SPEED_FILE = "plan/box2_hours.json"  # under OUT_DIR, next to PLAN_FILE
BOX2 = "full"
BOX2_TRAIN = ("t06", "p03", "p005")
OVERHEAD_PLAN = 0.08  # calc_v3 o (central): evals, saves, uploads and loader stalls on top of the steady step [X]
FIXED_S = {"t06": 435, "p03": 348, "p005": 820, "p01": 837}  # calc_v3 fixed [M + X]
DEV_CHECK_S = {"t06": 9.6, "p03": 9.2, "p005": 8.4, "p01": 8.5}  # calc_v3 DEV_CHECK_S, A100 = 5090 [X]
# setup: boot, gate, label pull and the audio rebuild. Central [X]: 571 GB at the download gate's 129 MB/s (box 1's
# host; F3's parallel upstream downloads must reach it) = 1.23 h + ~0.35 h boot, gate and labels. Pessimistic [M]: box 1
# measured 3.1 h with one download stream (7715f3f, before F3) + 0.2 h. A box-1 record's bootstrap_h raises the
# pessimistic value when it is larger.
SETUP_H = {"central": 1.6, "pess": 3.3}
STORES_H = {"central": 1.37, "pess": 1.91}  # calc_v3 setup_parts stores (both stores, full extent) [X]
AED_STORE_SHARE = 0.5  # the AED store's time / the CTC store's: smoke A's stores-aed 0.33 min / stores-ctc 0.67 [M]
STORES_PESS = 1.91 / 1.37  # a box-1 record's measured CTC store: central x (1 + AED_STORE_SHARE), pessimistic x this
# after T-0.6B on its GPU: m4-full-t06 (smoke: 1.3 min on the 100 h run's eval sets, which are the full ones), the 7
# T quant readouts (smoke-B #1: 0.7-1.5 min each on P-0.3B, T ~3x) over the two idle GPUs, the re-time pair (~1 min)
# [M + X]; the worst case also holds the last quant item's whole 0.75 h max_hours (the no-start rule's need)
POST_T06_H, POST_T06_MAX_H = 0.3, 1.2
# the other GPU's eval pool: the P M4 readouts, 14 P and 7 P-0.1B quant readouts (smoke-B #1: 0.65-1.0 min each) and
# the four Whisper models on all five sets (whisper-large-v3: 5,000 JSUT rows in 3.2 min; ~25 min for all four) [M + X]
POOL_B_H = 1.2
END_H = 0.35  # calc_v3 end: finish's uploads and the destroy
# max_hours' host margin: smoke A ran on a Ryzen 9950X (calc_v3's "fast" CPU class); calc_v3's pessimistic T-0.6B
# epoch is 13.635 / 10.846 = 1.26 x its fast one (box 2's only 2x offer today, m54650, is a Zen2 EPYC)
HOST_PESS = 1.26
STALL_RECOVERY_H = 1.25  # one stall of T-0.6B: stall_min 45 + up to 30 min of progress since its last full state


class PlanError(ValueError):
    """PLAN_FILE is missing, malformed, or measured with other step values than FULL_RUNS'."""


class SpeedError(ValueError):
    """SPEED_FILE is malformed, or an --import-speed input lacks what box2_hours needs."""


# ------------------------------------------------------------------------------------------------ the plan record


def _plan_want(x: str) -> dict:
    r = FULL_RUNS[x]
    return dict(family="aed" if x.startswith("t") else "ctc", micro_audio_s=float(r["micro"]),
                step_audio_s=float(r["step"]), epochs=int(r["epochs"]))


def plan_problems(plan) -> list[str]:
    """Why a PLAN_FILE record cannot serve the generator: {"full": <tools/full_plan.py JSON of full.parquet>,
    "smoke": <the same of smoke.parquet>}, each at the repo's selection path, measured for every FULL_RUNS student
    with its family, micro-batch, step and epochs, with total_steps (and worst_shapes on full.parquet)."""
    if not isinstance(plan, dict):
        return [f"not an object: {type(plan).__name__}"]
    p = []
    for which, sel in (("full", fullrun.FULL_SELECTION), ("smoke", fullrun.SMOKE_SELECTION)):
        rec = plan.get(which)
        if not isinstance(rec, dict):
            p.append(f"{which}: no tools/full_plan.py record")
            continue
        got_sel = (rec.get("selection") or {}).get("path")
        if got_sel != sel:
            p.append(f"{which}: measured on {got_sel!r}, not {sel}")
        students = rec.get("students") if isinstance(rec.get("students"), dict) else {}
        for x in FULL_RUNS:
            s = students.get(x)
            if not isinstance(s, dict):
                p.append(f"{which}: no student {x}")
                continue
            want = _plan_want(x)
            if diff := {k: s.get(k) for k in want if s.get(k) != want[k]}:
                p.append(f"{which}.{x}: measured with {diff}, FULL_RUNS says {({k: want[k] for k in diff})}")
            if not (isinstance(s.get("total_steps"), int) and s["total_steps"] > 0):
                p.append(f"{which}.{x}: total_steps {s.get('total_steps')!r}")
            if which == "full" and not (isinstance(s.get("worst_shapes"), list) and s["worst_shapes"]):
                p.append(f"{which}.{x}: no worst_shapes")
    return p


def load_plan(out_dir: Path = OUT_DIR) -> dict:
    f = Path(out_dir) / PLAN_FILE
    try:
        plan = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PlanError(f"{f}: not readable ({type(e).__name__}: {e})") from None
    if problems := plan_problems(plan):
        raise PlanError(f"{f}: " + "; ".join(problems))
    return plan


def plan_total_steps(x: str, plan: dict) -> int:
    """The full run's T on full.parquet (plan_epochs pins it at the start): the registry's smoke plan_total_steps."""
    return int(plan["full"]["students"][x]["total_steps"])


def plan_hours(x: str, plan: dict) -> float:
    """Plan v3's hours of the full run rescaled to the measured T (2 decimals): the registry's plan_hours of the smoke
    item and max_hours of the full run's train item."""
    h, t = PLAN_V3[x]
    return round(h * plan_total_steps(x, plan) / t, 2)


def plan_sec_per_step(x: str) -> float:
    """Plan v3's seconds per step of the full run (its hours over its step count; the rescale keeps it)."""
    h, t = PLAN_V3[x]
    return h * 3600 / t


def deadline_fault_bound_s(plan: dict) -> float:
    """Contract 7's bound on F4's seconds: DEADLINE_FAULT_SHARE x smoke-p005's projected run time = its steps on the
    smoke selection (plan["smoke"]) x plan v3's seconds per step of full-p005."""
    x = DEADLINE_FAULT_STUDENT
    return DEADLINE_FAULT_SHARE * int(plan["smoke"]["students"][x]["total_steps"]) * plan_sec_per_step(x)


def deadline_fault_s(plan: dict) -> int:
    """F4's seconds: the bound rounded down to DEADLINE_FAULT_ROUND_S (the item's KITSUNE_DEADLINE = start + this; 4d
    must then schedule, start or compress the cooldown, smoke check 8)."""
    return int(math.floor(deadline_fault_bound_s(plan) / DEADLINE_FAULT_ROUND_S) * DEADLINE_FAULT_ROUND_S)


def registry_numbers(plan: dict) -> dict:
    """The numbers the hand-written boxes.json takes from the plan record: per full student its plan_total_steps and
    plan_hours, and F4's seconds."""
    return dict(students={x: dict(plan_total_steps=plan_total_steps(x, plan), plan_hours=plan_hours(x, plan))
                          for x in FULL_RUNS}, deadline_fault_s=deadline_fault_s(plan),
                deadline_fault_bound_s=round(deadline_fault_bound_s(plan), 2))


# ------------------------------------------------------------------------------------------------ box 2's hours


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _up(x: float, step: float) -> float:
    """x rounded UP to a multiple of step (0.01, 0.1, 1), robust to binary noise (25.94 stays 25.94)."""
    return round(math.ceil(round(x / step, 6)) * step, 2)


def speed_problems(rec) -> list[str]:
    """Why a SPEED_FILE record cannot serve box2_hours: {"smoke": {..., "sec_per_step": {<every FULL_RUNS student>:
    s > 0}}, "box1": null | {..., "sec_per_step": s > 0, "overhead": 0 <= o < 1 | null, "stores_ctc_h": h > 0 | null,
    "bootstrap_h": h > 0 | null}}."""
    if not isinstance(rec, dict):
        return [f"not an object: {type(rec).__name__}"]
    p = []
    smoke = rec.get("smoke")
    if not isinstance(smoke, dict):
        p.append("smoke: no smoke A record (smoke verdict check 3's sec_per_step)")
    else:
        sps = smoke.get("sec_per_step") if isinstance(smoke.get("sec_per_step"), dict) else {}
        p += [f"smoke.sec_per_step.{x}: {sps.get(x)!r}, not a number > 0" for x in FULL_RUNS
              if not (_num(sps.get(x)) and sps[x] > 0)]
    b1 = rec.get("box1")
    if b1 is not None and not isinstance(b1, dict):
        p.append(f"box1: {type(b1).__name__}, not an object or null")
    elif b1 is not None:
        if not (_num(b1.get("sec_per_step")) and b1["sec_per_step"] > 0):
            p.append(f"box1.sec_per_step: {b1.get('sec_per_step')!r}, not a number > 0")
        o = b1.get("overhead")
        if o is not None and not (_num(o) and 0 <= o < 1):
            p.append(f"box1.overhead: {o!r}, not null or a number in [0, 1)")
        for k in ("stores_ctc_h", "bootstrap_h"):
            if b1.get(k) is not None and not (_num(b1[k]) and b1[k] > 0):
                p.append(f"box1.{k}: {b1[k]!r}, not null or a number > 0")
    return p


def load_speed(out_dir: Path = OUT_DIR) -> dict | None:
    """The SPEED_FILE record; None when there is none (box full's hours are then not held); SpeedError when it is
    unreadable or speed_problems finds any."""
    f = Path(out_dir) / SPEED_FILE
    if not f.is_file():
        return None
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SpeedError(f"{f}: not readable ({type(e).__name__}: {e})") from None
    if problems := speed_problems(rec):
        raise SpeedError(f"{f}: " + "; ".join(problems))
    return rec


def box2_hours(plan: dict, rec: dict, reserve_min: float = 60) -> dict:
    """Box full's hours from the plan record's step counts and a SPEED_FILE record (the model above; reserve_min: the
    box's deadline_reserve_min). Returns run_h (every FULL_RUNS student's run, P-0.1B's as box 1's cross-check), items
    (box full's train items' max_hours), the setup and stores hours (central, pessimistic), the two GPU lanes, the
    box's est_hours (the longer lane) and max_hours (the pessimistic T-0.6B path + its worst tail + the deadline
    reserve + one stall), and T-0.6B's pessimistic slack before its deadline-cooldown point."""
    sps, b1 = rec["smoke"]["sec_per_step"], rec.get("box1")
    r = b1["sec_per_step"] / sps["p01"] if b1 else 1.0
    o = b1["overhead"] if b1 and b1.get("overhead") is not None else OVERHEAD_PLAN
    dev_every = COMMON["eval"]["dev"]["every_epochs"]
    runs = {}
    for x in FULL_RUNS:
        steps_h = plan_total_steps(x, plan) * sps[x] * r * (1 + o) / 3600
        runs[x] = _up(steps_h + (FIXED_S[x] + FULL_RUNS[x]["epochs"] / dev_every * DEV_CHECK_S[x]) / 3600, 0.01)
    setup_c, setup_p = SETUP_H["central"], SETUP_H["pess"]
    stores_c, stores_p = STORES_H["central"], STORES_H["pess"]
    if b1 and b1.get("bootstrap_h"):
        setup_p = max(setup_p, round(b1["bootstrap_h"] + 0.2, 2))
    if b1 and b1.get("stores_ctc_h"):
        stores_c = round(b1["stores_ctc_h"] * (1 + AED_STORE_SHARE), 2)
        stores_p = round(stores_c * STORES_PESS, 2)
    ctc_c = stores_c / (1 + AED_STORE_SHARE)
    lane_a = setup_c + stores_c + runs["t06"] + POST_T06_H + END_H  # the stores, then T-0.6B and its tail
    lane_b = setup_c + ctc_c + runs["p03"] + runs["p005"] + POOL_B_H + END_H  # P-0.3B, P-0.05B, the eval pool
    t06_end_p = setup_p + stores_p + runs["t06"] * HOST_PESS
    mx = _up(t06_end_p + POST_T06_MAX_H + reserve_min / 60 + STALL_RECOVERY_H, 1)
    # 4d plans T-0.6B's cooldown to end its end_reserve_min before KITSUNE_DEADLINE (the box's less reserve_min)
    slack = mx - reserve_min / 60 - FULL_RUNS["t06"]["end_reserve"] / 60 - t06_end_p
    return dict(r=round(r, 4), o=o, run_h=runs, items={f"full-{x}": runs[x] for x in BOX2_TRAIN},
                setup_h=(setup_c, setup_p), stores_h=(stores_c, stores_p),
                lanes={"A": round(lane_a, 2), "B": round(lane_b, 2)}, est_hours=_up(max(lane_a, lane_b), 0.1),
                max_hours=int(mx), t06_slack_h=round(slack, 2))


def _smoke_record(verdict: dict, src) -> dict:
    """SPEED_FILE's smoke part from a smoke A verdict (check 3's evidence)."""
    ev = ((verdict.get("checks") or {}).get("3") or {}).get("evidence") or {}
    sps = ev.get("sec_per_step") if isinstance(ev.get("sec_per_step"), dict) else {}
    out = {x: sps.get(f"smoke-{x}") for x in FULL_RUNS}
    if bad := [x for x, v in out.items() if not (_num(v) and v > 0)]:
        raise SpeedError(f"{src}: smoke check 3 has no sec_per_step for {['smoke-' + x for x in bad]} (a smoke A "
                         f"verdict, full/box-full-smoke/smoke_verdict.json)")
    return dict(source=fullrun.box_verdict_path(verdict.get("box") or "full-smoke"), sha=verdict.get("sha"),
                machine_id=verdict.get("machine_id"), time_utc=verdict.get("time_utc"), sec_per_step=out)


def import_speed(smoke_verdict: Path | None = None, box1_json: Path | None = None, out_dir: Path = OUT_DIR) -> dict:
    """Record SPEED_FILE: its smoke part from a smoke A verdict (else the existing record's) and its box-1 part from
    tools/box1_go.py --json (its "box1" object; else the existing record's). Refused (SpeedError, nothing written)
    when the result fails speed_problems."""
    try:
        old = load_speed(out_dir) or {}
    except SpeedError:
        old = {}
    rec = {"_comment": "Box full's measured speeds (tools/make_full_configs.py --import-speed; never edited by hand): "
                       "smoke: smoke A's verdict check 3 (median steady s/step per student at the full configs' "
                       "batches); box1: box 1's measurement from tools/box1_go.py --json (null until box 1 ended: the "
                       "hours are then provisional). make_full_configs.box2_hours projects box full's train items' "
                       "max_hours and its est_hours / max_hours from it, and --check holds boxes.json to them.",
           "smoke": old.get("smoke"), "box1": old.get("box1")}
    if smoke_verdict is not None:
        try:
            v = json.loads(Path(smoke_verdict).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SpeedError(f"{smoke_verdict}: not readable ({type(e).__name__}: {e})") from None
        rec["smoke"] = _smoke_record(v, smoke_verdict)
    if box1_json is not None:
        try:
            b = json.loads(Path(box1_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SpeedError(f"{box1_json}: not readable ({type(e).__name__}: {e})") from None
        rec["box1"] = b.get("box1") if isinstance(b, dict) and "box1" in b else b
        if rec["box1"] is None:
            raise SpeedError(f"{box1_json}: no box1 measurement (box1_go wrote none: box 1 has not ended)")
    if problems := speed_problems(rec):
        raise SpeedError("; ".join(problems))
    f = Path(out_dir) / SPEED_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(render(rec).encode("utf-8"))
    return rec


# the registry's numbers that must always equal the plan record's (registry_drift): smoke A's train items (smoke check
# 3's projection), F4's seconds and box 1's train hours. Box full's train hours and its est / max hours are held to the
# speed record instead (SPEED_FILE, box2_hours), when there is one
PLAN_BOUND_BOXES = {"full-smoke": "smoke", "p01": "full"}


def registry_drift(reg: dict, plan: dict, speed: dict | None = None) -> list[str]:
    """Where a loaded registry's plan-bound numbers (PLAN_BOUND_BOXES) differ from registry_numbers(plan), and, given a
    speed record, where box full's differ from box2_hours: after --import-plan of a rebuilt selection or
    --import-speed of a new measurement, what boxes.json must be changed in by hand."""
    nums, p = registry_numbers(plan), []
    boxes = reg.get("boxes") or {}
    for box, prefix in PLAN_BOUND_BOXES.items():
        if box not in boxes:
            continue
        items = {it["name"]: it for it in boxes[box].get("items") or []}
        for x, want in nums["students"].items():
            it = items.get(f"{prefix}-{x}")
            if it is None:
                continue
            got = ({"plan_total_steps": it.get("plan_total_steps"), "plan_hours": it.get("plan_hours")}
                   if prefix == "smoke" else {"max_hours": it.get("max_hours")})
            exp = dict(want) if prefix == "smoke" else {"max_hours": want["plan_hours"]}
            if got != exp:
                p.append(f"boxes.{box}.items.{it['name']}: {got}, the plan record gives {exp}")
        for f in boxes[box].get("faults") or []:
            if f.get("action") == "deadline" and f.get("item") == DEADLINE_FAULT_ITEM \
                    and f.get("seconds") != nums["deadline_fault_s"]:
                p.append(f"boxes.{box}.faults.{f.get('id')}: seconds {f.get('seconds')}, the plan record gives "
                         f"{nums['deadline_fault_s']} (bound {nums['deadline_fault_bound_s']} s)")
    if speed is not None and BOX2 in boxes:
        box = boxes[BOX2]
        h = box2_hours(plan, speed, reserve_min=box.get("deadline_reserve_min") or 60)
        items = {it["name"]: it for it in box.get("items") or []}
        for name, want in h["items"].items():
            if name in items and items[name].get("max_hours") != want:
                p.append(f"boxes.{BOX2}.items.{name}: {{'max_hours': {items[name].get('max_hours')}}}, the speed "
                         f"record gives {{'max_hours': {want}}}")
        got = {"est_hours": box.get("est_hours"), "max_hours": box.get("max_hours")}
        exp = {"est_hours": h["est_hours"], "max_hours": h["max_hours"]}
        if got != exp:
            p.append(f"boxes.{BOX2}: {got}, the speed record gives {exp}")
    return p


def readout_reserve_problems(reg: dict, root: Path) -> list[str]:
    """Every readout of a plain non-smoke box of a loaded registry whose max_hours + END_PHASE_SLACK_MIN exceed its
    run's schedule.end_reserve_min (the config under root): after a run that 4d shortened, the no-start rule would
    skip it (READOUT_RESERVE). A chained box (contract addendum E) has no items of its own: its parts are checked."""
    p = []
    for bname, box in (reg.get("boxes") or {}).items():
        if box.get("smoke") or "chain" in box:
            continue
        items = {it["name"]: it for it in box.get("items") or []}
        for it in items.values():
            run = items.get(it.get("of"))
            if it["kind"] != "readout" or not it.get("max_hours") or run is None:
                continue
            try:
                reserve = json.loads((Path(root) / run["config"]).read_text(encoding="utf-8"))["schedule"][
                    "end_reserve_min"]
            except (OSError, ValueError, KeyError, TypeError) as e:
                p.append(f"boxes.{bname}.items.{it['name']}: no schedule.end_reserve_min in {run['config']} "
                         f"({type(e).__name__})")
                continue
            need = float(it["max_hours"]) * 60 + END_PHASE_SLACK_MIN
            if float(reserve) < need:
                p.append(f"boxes.{bname}.items.{it['name']}: max_hours {it['max_hours']} "
                         f"({float(it['max_hours']) * 60:g} min) + {END_PHASE_SLACK_MIN} min of the trainer's end "
                         f"phase exceed {run['config']}'s schedule.end_reserve_min {reserve}: after a run the "
                         f"deadline cooldown shortened, the no-start rule would skip the readout (make_full_configs "
                         f"READOUT_RESERVE)")
    return p


# ------------------------------------------------------------------------------------------------ the configs


def _with_data(cfg: dict, data: dict) -> dict:
    """cfg with every DATA_KEYS key replaced whole by data's (in place, so the keys keep their order; a key data lacks
    is removed: full-p01 has no pull_parakeet). Never merged: the study extent's inputs must not survive into the full
    extent's empty inputs."""
    out = {}
    for k, v in cfg.items():
        if k not in fullrun.DATA_KEYS:
            out[k] = v
        elif k in data:
            out[k] = copy.deepcopy(data[k])
    for k in fullrun.DATA_KEYS:
        if k in data and k not in out:
            out[k] = copy.deepcopy(data[k])
    return out


def full_data(x: str) -> dict:
    d = copy.deepcopy(fullrun.FULL_DATA)
    if FULL_RUNS[x]["pull_parakeet"]:
        d["pull_parakeet"] = True
    return d


def smoke_data() -> dict:
    return dict(copy.deepcopy(fullrun.SMOKE_DATA), pull_parakeet=True)


def full_config(x: str, r: dict | None = None) -> dict:
    """full-<x>: the study run's config with the full data, FULL_RUNS[x] and COMMON."""
    run = FULL_RUNS[x]
    cfg = _with_data(study.run_config(run["study_run"], r, RUNS_REPO), full_data(x))
    cfg = study._merge(cfg, COMMON)
    return study._merge(cfg, {
        "run_name": f"full-{x}",
        "schedule": {"epochs": run["epochs"], "warmup_steps": run["warmup"], "end_reserve_min": run["end_reserve"]},
        "optim": {"lr": run["lr"]},
        "batch": {"micro_audio_s": run["micro"], "step_audio_s": run["step"]},
        "eval": {"dev": {"greedy": run["dev_greedy"]}},
    })


def smoke_config(x: str, plan: dict, r: dict | None = None) -> dict:
    """smoke-<x>: full-<x> on the smoke data with SMOKE, the full data's worst shapes and the forced trigger."""
    cfg = _with_data(full_config(x, r), smoke_data())
    cfg = study._merge(cfg, SMOKE)
    cfg = study._merge(cfg, SMOKE_OVER.get(x, {}))
    cfg = study._merge(cfg, FORCED.get(x, {}))
    shapes = [{"name": s["name"], "durations": list(s["durations"])}
              for s in plan["full"]["students"][x]["worst_shapes"]]
    return study._merge(cfg, {"run_name": f"smoke-{x}", "memory": {"probe_shapes": shapes}})


def _study_data_keys() -> dict:
    d = json.loads(STUDY_DATA.read_text(encoding="utf-8"))
    return {k: d[k] for k in fullrun.DATA_KEYS if k in d}


def data_configs() -> dict[str, dict]:
    """The four data configs (the registry's box data_config values)."""
    return {
        "data-p01": {"_comment": "Box p01's data config (box 1, P-0.1B alone; launch --config, bootstrap "
                                 "KITSUNE_CONFIG): the full selection and extent (kitsune.fullrun FULL_DATA), family "
                                 "ctc and no pull_parakeet, so the box pulls parakeet_out for every stem and "
                                 "teacher_out for the eval sets' eval stems only (kitsune.extent.pull_plan, fix 9). "
                                 "Generated by tools/make_full_configs.py.",
                     **copy.deepcopy(fullrun.FULL_DATA), "family": "ctc"},
        "data-full": {"_comment": "Box full's data config (box 2, T-0.6B, P-0.3B, P-0.05B): the full selection and "
                                  "extent (kitsune.fullrun FULL_DATA) with pull_parakeet true (both label roots for "
                                  "every stem). Generated by tools/make_full_configs.py.",
                      **copy.deepcopy(fullrun.FULL_DATA), "pull_parakeet": True},
        "data-smoke": {"_comment": "Box full-smoke's data config (smoke A): the smoke selection (a seeded 100 h train "
                                   "draw with its own dev slice) on the study extent (kitsune.fullrun SMOKE_DATA), "
                                   "pull_parakeet true. Generated by tools/make_full_configs.py.",
                       **smoke_data()},
        "data-smoke-b": {"_comment": "Box smoke-b's data config (smoke B): study/data.json's data keys verbatim, the "
                                     "frozen study selection, so every study-weight eval, speed probe and Whisper "
                                     "eval of the box shares one eval store (its stores-eval item builds it from "
                                     "configs/study/study-t06.json, which carries the same keys). Generated by "
                                     "tools/make_full_configs.py.",
                         **_study_data_keys()},
    }


def all_configs(out_dir: Path = OUT_DIR, r: dict | None = None, plan: dict | None = None) -> dict[str, dict]:
    """name -> config for every generated file under configs/full/ (PlanError without a valid PLAN_FILE; plan: the
    record already loaded)."""
    plan = plan if plan is not None else load_plan(out_dir)
    out = {f"full-{x}": full_config(x, r) for x in FULL_RUNS}
    out.update({f"smoke-{x}": smoke_config(x, plan, r) for x in FULL_RUNS})
    out.update(data_configs())
    return out


render = study.render


def _generated(out_dir: Path) -> list[Path]:
    """The *.json files directly in out_dir that are the generator's to write or remove (everything but boxes.json)."""
    return sorted(p for p in Path(out_dir).glob("*.json") if p.name != BOXES)


def write_all(out_dir: Path = OUT_DIR) -> tuple[list[str], list[str]]:
    """Write every config; remove the generated-looking *.json in out_dir the generator no longer makes (never
    boxes.json nor PLAN_FILE). Returns (written, removed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = all_configs(out_dir)
    for name, cfg in cfgs.items():
        (out_dir / f"{name}.json").write_bytes(render(cfg).encode("utf-8"))
    stale = [p.name for p in _generated(out_dir) if p.stem not in cfgs]
    for name in stale:
        (out_dir / name).unlink()
    return sorted(cfgs), stale


def registry_check(out_dir: Path = OUT_DIR, plan: dict | None = None, speed: dict | None = None) -> list[str]:
    """fullrun.registry_problems of out_dir/boxes.json, its config paths resolved in the checkout out_dir belongs to
    (<X> for <X>/configs/full); for a valid registry also readout_reserve_problems and, given the plan record,
    registry_drift (with the speed record, box full's hours too)."""
    f = Path(out_dir) / BOXES
    if not f.is_file():
        return [f"{BOXES}: missing (the box registry is hand-written next to the generated configs)"]
    try:
        reg = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{BOXES}: not readable JSON ({type(e).__name__}: {e})"]
    root = Path(out_dir).resolve().parents[1]
    if problems := fullrun.registry_problems(reg, root=root):
        return [f"{BOXES}: {p}" for p in problems]
    reg = fullrun.load_registry(reg, root=root, check_files=False)
    problems = readout_reserve_problems(reg, root) + (registry_drift(reg, plan, speed) if plan is not None else [])
    return [f"{BOXES}: {p}" for p in problems]


def check(out_dir: Path = OUT_DIR) -> list[str]:
    """The differences between the files in out_dir and the generator's output (parsed JSON, so a CRLF checkout
    compares equal), and every problem of the hand-written registry (registry_check: fullrun.registry_problems, the
    readouts' end reserve, the numbers bound to the plan record and, with a SPEED_FILE record, box full's hours)."""
    out_dir = Path(out_dir)
    try:
        plan = load_plan(out_dir)
        cfgs = all_configs(out_dir, plan=plan)
    except PlanError as e:
        return [str(e)] + registry_check(out_dir)
    try:
        speed, speed_err = load_speed(out_dir), []
    except SpeedError as e:
        speed, speed_err = None, [str(e)]
    problems = []
    for name, cfg in cfgs.items():
        p = out_dir / f"{name}.json"
        if not p.is_file():
            problems.append(f"{p.name}: missing")
        elif json.loads(p.read_text(encoding="utf-8")) != json.loads(render(cfg)):
            problems.append(f"{p.name}: differs from the generator's")
    problems += [f"{p.name}: not made by the generator" for p in _generated(out_dir) if p.stem not in cfgs]
    return problems + speed_err + registry_check(out_dir, plan, speed)


def import_plan(full_json: Path, smoke_json: Path, out_dir: Path = OUT_DIR) -> dict:
    """Record a tools/full_plan.py measurement of full.parquet and smoke.parquet as PLAN_FILE: each JSON as the tool
    wrote it, its local selection path replaced by the repo path (the file name must be full.parquet / smoke.parquet).
    Refused (PlanError, nothing written) when a record lacks a FULL_RUNS student or was measured with other values."""
    plan = {"_comment": "tools/full_plan.py on the full-data selections (labels/full/selections/full_study/"
                        "full.parquet and smoke.parquet), recorded by tools/make_full_configs.py --import-plan: the "
                        "tool's JSON, the local selection path replaced by the repo path. make_full_configs reads the "
                        "smoke configs' memory.probe_shapes (worst_shapes on full.parquet) from it, and "
                        "tests/test_full_configs.py the registry's plan_total_steps / plan_hours and the deadline "
                        "fault's bound (registry_numbers)."}
    for which, src, sel in (("full", full_json, fullrun.FULL_SELECTION), ("smoke", smoke_json, fullrun.SMOKE_SELECTION)):
        rec = json.loads(Path(src).read_text(encoding="utf-8"))
        path = str((rec.get("selection") or {}).get("path") or "").replace("\\", "/")
        if path.rsplit("/", 1)[-1] != sel.rsplit("/", 1)[-1]:
            raise PlanError(f"{src}: measured on {path!r}, not a {sel.rsplit('/', 1)[-1]}")
        rec["selection"]["path"] = sel
        plan[which] = rec
    if problems := plan_problems(plan):
        raise PlanError("; ".join(problems))
    f = Path(out_dir) / PLAN_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(render(plan).encode("utf-8"))
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compare the committed files with the generator's output and validate boxes.json")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--import-plan", nargs=2, metavar=("FULL_JSON", "SMOKE_JSON"), default=None,
                    help="record tools/full_plan.py's JSON of full.parquet and smoke.parquet as " + PLAN_FILE +
                         ", then write the configs")
    ap.add_argument("--import-speed", action="store_true",
                    help="record box full's measured speeds as " + SPEED_FILE + " (--smoke-verdict and/or --box1-go; "
                         "the part not given is kept) and print the hours boxes.json must carry")
    ap.add_argument("--smoke-verdict", default=None, metavar="JSON",
                    help="--import-speed: smoke A's smoke_verdict.json (full/box-full-smoke/smoke_verdict.json)")
    ap.add_argument("--box1-go", default=None, metavar="JSON", help="--import-speed: tools/box1_go.py --json's file")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if (args.smoke_verdict or args.box1_go) and not args.import_speed:
        ap.error("--smoke-verdict / --box1-go go with --import-speed")
    if args.import_speed:
        if args.check or args.import_plan:
            ap.error("--import-speed is not combined with --check or --import-plan")
        try:
            rec = import_speed(Path(args.smoke_verdict) if args.smoke_verdict else None,
                               Path(args.box1_go) if args.box1_go else None, out_dir)
            plan = load_plan(out_dir)
            reg = json.loads((out_dir / BOXES).read_text(encoding="utf-8"))
            h = box2_hours(plan, rec, reserve_min=reg["boxes"][BOX2].get("deadline_reserve_min") or 60)
        except (SpeedError, PlanError, OSError, ValueError, KeyError) as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"wrote {out_dir / SPEED_FILE} ({'smoke A and box 1' if rec['box1'] else 'smoke A only: provisional'}"
              f"; r {h['r']:g}, o {h['o']:g}); {BOXES} (hand-written) must carry: "
              + ", ".join(f"{n} max_hours {v}" for n, v in h["items"].items())
              + f"; box {BOX2} est_hours {h['est_hours']:g}, max_hours {h['max_hours']} (lanes {h['lanes']['A']:g} / "
              f"{h['lanes']['B']:g} h; T-0.6B's pessimistic slack {h['t06_slack_h']:g} h; P-0.1B projects to "
              f"{h['run_h']['p01']:g} h)")
        todo = registry_check(out_dir, plan, rec)
        print(f"{BOXES}: " + ("carries the speed record's hours" if not todo else f"{len(todo)} change(s) by hand")
              + "".join(f"\n  {p}" for p in todo))
        return 0
    if args.check:
        problems = check(out_dir)
        print(f"{out_dir}: " + ("up to date" if not problems else f"{len(problems)} problem(s)")
              + "".join(f"\n  {p}" for p in problems))
        return 1 if problems else 0
    plan = None
    try:
        if args.import_plan:
            plan = import_plan(Path(args.import_plan[0]), Path(args.import_plan[1]), out_dir)
            nums = registry_numbers(plan)
            print(f"wrote {out_dir / PLAN_FILE}; {BOXES} (hand-written) must carry: "
                  + ", ".join(f"{x} plan_total_steps {v['plan_total_steps']} plan_hours {v['plan_hours']}"
                              for x, v in nums["students"].items())
                  + f"; F4 seconds {nums['deadline_fault_s']} (bound {nums['deadline_fault_bound_s']} s)")
        written, removed = write_all(out_dir)
    except PlanError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(f"wrote {len(written)} configs to {out_dir}" + (f"; removed {removed}" if removed else ""))
    if plan is not None:  # what the hand-written registry must now be changed in (--check fails until it is)
        todo = registry_check(out_dir, plan)
        print(f"{BOXES}: " + ("carries the plan record's numbers" if not todo else f"{len(todo)} change(s) by hand")
              + "".join(f"\n  {p}" for p in todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
