"""The full runs' step plans on a selection, as the trainer will make them: the kept split-"train" rows through
kitsune.trainset.StepPlanner (pool_micro 50, seed 1234; the trainer's make_planner and plan_epochs), per student.

Why: batch.step_audio_s is RESUME_FIXED and the plan (plan v3, DECISIONS C10) keeps the study's REALISED audio per step,
not its nominal value. The planner groups whole micro-batches into steps (a step ends where adding the next micro-batch
would overshoot more than stopping undershoots), so the realised mean depends on the micro-batch size: the study's
T-0.6B at micro 600 / step 1500 realised 1,730.7 s, P-0.3B at micro 1200 / step 1500 1,176.5 s, P-0.1B and P-0.05B at
micro 1600 / step 1500 1,538.5 s (epoch 0's step_real_s_mean of the study runs' plan events). The full configs change the
micro-batch sizes, so their step values are measured here: on the frozen study selection (the same pool) they must
reproduce those numbers, and they are re-checked on full.parquet before launch. The same readout gives the smoke configs'
memory.probe_shapes (worst_shapes on full.parquet) and the step counts behind plan_total_steps.

Per student (--student NAME=FAMILY:MICRO:STEP:EPOCHS, repeatable; default the four full students below):
  steps_per_epoch, total_steps   every epoch's plan length and their sum: the T that plan_epochs pins (less the rows a
                                 frame store's preflight drops, 0 in the study)
  step_real_s {mean, min, max}   epoch 0's real audio seconds per step; study_step_real_s and delta_pct: against the
                                 study value of the student the name starts with (t06, p03, p01, p005; "p03-1300" too)
  micro_per_step, pad_eff (pad_eff_dec for aed), micro_utts_max, micro_targets_max, excluded_dec_len (aed: decoder
                                 inputs over max_dec_len 200)
  steps_per_0p1_epoch            epoch 0's steps / 10: the gap between two dev checks (eval.dev.every_epochs 0.1)
  cooldown_start_step            ceil(0.8 T) + 1: the first WSD cooldown step (schedule.cooldown_frac 0.2)
  worst_shapes                   [{"name": "most_rows" | "longest" | "most_targets", "durations": [...]}] of epoch 0:
                                 the micro-batch with the most rows, the one with the longest utterance (StepPlanner.
                                 worst_micro_batches) and the one with the most target tokens; the value of
                                 memory.probe_shapes (durations longest first, 3 decimals)
The ctc family plans frames (max_dec_len None): only the durations drive its plan. The selection's n_tok is the Cohere
token count, not the CTC target length the frame store holds, so a ctc student's micro_targets_max and most_targets
use it as a proxy (the step counts and audio are exact).

Usage (CPU, a few minutes on the full selection; no labels or audio needed):
  python tools/full_plan.py --selection D:/kitsune-study/selections/study_1000h.parquet --json plan.json
  python tools/full_plan.py --selection full.parquet --student p03=ctc:600:1350:3 --student p03-1300=ctc:600:1300:3
"""
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POOL_MICRO, SEED, MAX_DEC_LEN, COOLDOWN_FRAC = 50, 1234, 200, 0.2  # 04_distill DEFAULTS / the study and full configs
FAMILIES = ("aed", "ctc")
# the full students: family, batch.micro_audio_s, batch.step_audio_s, schedule.epochs (contract 3, DECISIONS C10)
DEFAULT_STUDENTS = {"t06": ("aed", 450, 1730, 3), "p03": ("ctc", 600, 1350, 3), "p01": ("ctc", 1600, 1500, 4),
                    "p005": ("ctc", 1600, 1500, 4)}
# epoch 0's step_real_s_mean of the study runs' plan events (study-t06-20260926T174027Z at micro 600, study-p03-
# 20260926T172336Z at 1200, study-p01-20260926T172421Z and study-p005-20260926T172507Z at 1600; step 1500 each)
STUDY_STEP_REAL_S = {"t06": 1730.7220159744832, "p03": 1176.470242226075, "p01": 1538.4610859879444,
                     "p005": 1538.4610859879444}
SHAPE_DECIMALS = 3


@dataclass(frozen=True)
class Student:
    name: str
    family: str
    micro_audio_s: float
    step_audio_s: float
    epochs: int

    @property
    def study_key(self) -> str | None:
        base = self.name.split("-", 1)[0]
        return base if base in STUDY_STEP_REAL_S else None


def parse_student(spec: str) -> Student:
    """NAME=FAMILY:MICRO:STEP:EPOCHS, e.g. p03=ctc:600:1350:3."""
    name, eq, rest = spec.partition("=")
    parts = rest.split(":")
    try:
        if not (eq and name and len(parts) == 4 and parts[0] in FAMILIES):
            raise ValueError
        s = Student(name, parts[0], float(parts[1]), float(parts[2]), int(parts[3]))
        if s.micro_audio_s <= 0 or s.step_audio_s <= 0 or s.epochs < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"--student {spec!r}: expected NAME=FAMILY:MICRO:STEP:EPOCHS with FAMILY in "
                                         f"{FAMILIES}, MICRO and STEP > 0, EPOCHS >= 1 (e.g. p03=ctc:600:1350:3)")
    return s


def default_students() -> list[Student]:
    return [Student(n, f, float(m), float(s), e) for n, (f, m, s, e) in DEFAULT_STUDENTS.items()]


def train_utts(selection) -> list:
    """The kept split-"train" rows in selection order (the train store's order: sources in config order, stems
    sorted, rows in teacher_out order) as kitsune.trainset.Utt."""
    import pandas as pd

    from kitsune import trainset

    sel = pd.read_parquet(selection, columns=["id", "source", "split", "keep", "duration", "n_tok"])
    sel = sel[sel["keep"] & (sel["split"] == "train")]
    return [trainset.Utt(i, s, float(d), int(n), 0, 0, 0)
            for i, s, d, n in zip(sel["id"], sel["source"], sel["duration"], sel["n_tok"])]


def worst_shapes(planner, plan) -> list[dict]:
    """memory.probe_shapes of epoch 0's plan: the micro-batch with the most rows, the longest (StepPlanner's rule) and
    the one with the most target tokens, each as its durations, longest first."""
    import numpy as np

    mbs = [mb for step in plan for mb in step]
    dur, n_tok = planner.dur, planner.n_tok
    picked = {"most_rows": max(mbs, key=lambda mb: (len(mb), float(dur[mb].max()))),
              "longest": planner.worst_micro_batches(plan)["longest"],
              "most_targets": max(mbs, key=lambda mb: int(n_tok[mb].sum()))}
    return [{"name": k, "durations": [round(float(x), SHAPE_DECIMALS) for x in np.sort(dur[mb])[::-1]]}
            for k, mb in picked.items()]


def plan_student(st: Student, utts: list, cache: dict) -> dict:
    """One student's readout; `cache` keeps a planner and its epoch plans per (family, micro, step), so two students
    with the same batch settings (p01, p005) are planned once."""
    from kitsune import trainset

    key = (st.family, st.micro_audio_s, st.step_audio_s)
    if key not in cache:
        cache[key] = (trainset.StepPlanner(utts, step_audio_s=st.step_audio_s, micro_audio_s=st.micro_audio_s,
                                           max_dec_len=None if st.family == "ctc" else MAX_DEC_LEN,
                                           pool_micro=POOL_MICRO, seed=SEED, prompt_len=len(trainset.PROMPT)), {})
    planner, plans = cache[key]
    for e in range(st.epochs):
        if e not in plans:
            plans[e] = planner.epoch_plan(e)
    steps = [len(plans[e]) for e in range(st.epochs)]
    total = sum(steps)
    s0 = planner.epoch_stats[0]
    study = STUDY_STEP_REAL_S.get(st.study_key) if st.study_key else None
    out = {"family": st.family, "micro_audio_s": st.micro_audio_s, "step_audio_s": st.step_audio_s,
           "epochs": st.epochs, "max_dec_len": None if st.family == "ctc" else MAX_DEC_LEN,
           "steps_per_epoch": steps, "total_steps": total,
           "step_real_s": {"mean": s0["step_real_s_mean"], "min": s0["step_real_s_min"], "max": s0["step_real_s_max"]},
           "study_step_real_s": study,
           "delta_pct": None if study is None else 100.0 * (s0["step_real_s_mean"] - study) / study,
           "micro_per_step": s0["micro_per_step"], "pad_eff": s0["pad_eff_audio"],
           "micro_utts_max": s0["micro_utts_max"], "micro_targets_max": s0["micro_targets_max"],
           "excluded_dec_len": s0["excluded_dec_len"], "steps_per_0p1_epoch": round(steps[0] / 10, 1),
           "cooldown_start_step": math.ceil((1.0 - COOLDOWN_FRAC) * total) + 1,
           "real_h": s0["real_h"], "planner_fingerprint": planner.fingerprint,
           "worst_shapes": worst_shapes(planner, plans[0])}
    if st.family == "aed":
        out["pad_eff_dec"] = s0["pad_eff_dec"]
    return out


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(selection, students: list[Student], log=print) -> dict:
    """The readout of every student on the selection (the --json document)."""
    t0 = time.time()
    h = _sha256(selection)
    utts = train_utts(selection)
    log(f"{selection}: {len(utts)} kept train rows, {sum(u.duration for u in utts) / 3600:.2f} h "
        f"({time.time() - t0:.1f} s)")
    cache: dict = {}
    out = {"selection": {"path": str(selection), "sha256": h}, "n_train_utts": len(utts),
           "train_hours": sum(u.duration for u in utts) / 3600, "pool_micro": POOL_MICRO, "seed": SEED, "students": {}}
    for st in students:
        t = time.time()
        r = out["students"][st.name] = plan_student(st, utts, cache)
        d = "" if r["delta_pct"] is None else f" ({r['delta_pct']:+.2f} % vs the study's {r['study_step_real_s']:.1f})"
        log(f"{st.name:>10} {st.family} micro {st.micro_audio_s:g} step {st.step_audio_s:g}: "
            f"{r['steps_per_epoch']} steps/epoch, T {r['total_steps']}, real s/step {r['step_real_s']['mean']:.1f}{d}, "
            f"micro/step {r['micro_per_step']:.2f}, pad_eff {r['pad_eff']:.4f}, 0.1 epoch {r['steps_per_0p1_epoch']} "
            f"steps, cooldown from {r['cooldown_start_step']}, worst micro rows "
            f"{[len(s['durations']) for s in r['worst_shapes']]} ({time.time() - t:.1f} s)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True, help="a selection parquet (the kept split-train rows are planned)")
    ap.add_argument("--student", action="append", type=parse_student, default=None, metavar="NAME=FAMILY:MICRO:STEP:EP",
                    help="a student to plan (repeatable; default t06=aed:450:1730:3 p03=ctc:600:1350:3 "
                         "p01=ctc:1600:1500:4 p005=ctc:1600:1500:4)")
    ap.add_argument("--json", default=None, metavar="OUT", help="also write the readout as JSON")
    args = ap.parse_args(argv)
    students = args.student or default_students()
    if len({s.name for s in students}) != len(students):
        ap.error("--student names must be unique")
    out = run(args.selection, students)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
