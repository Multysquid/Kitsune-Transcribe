"""The size study's pre-registration: the rules fixed before the study box starts, and the numbers the box writes.

Two layers, so that no number the study reports can be chosen after its results are seen (STUDY.md 4.8):

  rules      everything decided in advance: the runs and their exact parameter counts, the training settings per init
             class, the LR probe grids and the edge rule, the calibration rule for max_steps, the T/2 branch, the
             selection rule and the eval manifest, metrics, the noise model, the limit rule, the practical bars, the
             anchor, the invalidation rules and all 30 owner decisions. `rules()` returns them; `write_rules(dir)`
             writes study/PREREG.json (canonical JSON, the hashed form) and study/PREREG.md (the same content for
             reading). Both are committed to main by PR before the study box launches.
  numbers    what only the study host can measure: the calibration table, max_steps per run, the LR probe objectives
             and the chosen LRs. The box computes them mechanically with max_steps() and choose_lr(), then
             write_numbers() writes PREREG_numbers.json and returns its sha256, which the box logs as an event and
             uploads before the first study step. write_numbers refuses numbers that do not follow from the
             measurements by these rules, and rules that are still pending.

Some rule fields need the sealed labels (labels/full/COMPLETE.json): the manifest and selection hashes, the Galgame
view ids, the teacher baselines on the manifest and the final reazon_large cap. Until then they are the string
"pending" (pending() lists them). scripts/make_selection.py --config study/data.json writes all of them into the
selection's sidecar (labels/full/selections/study_1000h.json), and the final pre-launch commit fills them:

    python -m kitsune.prereg --write study/                                   # the rules, with pending fields
    python -m kitsune.prereg --write study/ --sidecar labels/full/selections/study_1000h.json   # the final commit
    python -m kitsune.prereg --check study/     # the committed files equal the generator's output; lists pending

Pure Python (stdlib only); no torch.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PENDING = "pending"
RULES_VERSION = 1
RULES_JSON, RULES_MD, NUMBERS_JSON = "PREREG.json", "PREREG.md", "PREREG_numbers.json"

# --------------------------------------------------------------------------------------------------- the runs

T_REF_RUN, REF_STEPS = "study-t06", 9366  # T = the T-0.6B's 4 epochs of 1,000 h (STUDY.md 5.1)
REPLICATE, REPLICATE_OF = "study-t01-s1235", "study-t01"  # the replicate trains T-0.1B's shape and steps, seed 1235

# run -> what the study fixes about it (CONTRACT.md section 1, STUDY.md 1.1 and 2.2). params_* are the exact meta-device
# counts every builder asserts; lr_from is the LR probe class whose winner the run takes
RUNS = {
    "study-t06": dict(family="aed", init_class="pruned_kept", student="students/study/t06", seed=1234,
                      shape="encoder 20 of 48 layers [0,2,5,7,10,12,15,17,20,22,25,27,30,32,35,37,40,42,45,47], "
                            "d 1280, FFN 5120 -> 2560; decoder layers {0,2,5,7}, D 1024; vocab 16,384 tied",
                      params_total=616_963_328, params_non_embedding=599_137_536, lr_from="kept-t03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=600, wave=1),
    "study-t03": dict(family="aed", init_class="pruned_kept", student="students/study/t03", seed=1234,
                      shape="encoder 10 layers [0,5,10,16,21,26,31,37,42,47], d 1280, FFN 2560; decoder {0,7}, "
                            "D 1024; vocab 16,384 tied",
                      params_total=320_752_384, params_non_embedding=302_926_592, lr_from="kept-t03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1200, wave=1),
    "study-bridge": dict(family="aed", init_class="scratch", student="students/study/bridge", seed=1234,
                         shape="the T-0.3B shape, from scratch", params_total=320_752_384,
                         params_non_embedding=302_926_592, lr_from="bridge", warmup_steps=2000, weight_decay=1e-3,
                         bn="train mode", aux_ctc=0.3, micro_audio_s=1200, wave=1),
    "study-t01": dict(family="aed", init_class="scratch", student="students/study/t01", seed=1234,
                      shape="encoder 12 layers, d 512, 8 x 64, FFN 2048, kernel 9; decoder 4 layers, D 512, 8 x 64, "
                            "FFN 2048; vocab 16,384 tied",
                      params_total=103_996_416, params_non_embedding=95_083_520, lr_from="scratch",
                      warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1600, wave=2),
    REPLICATE: dict(family="aed", init_class="scratch", student="students/study/t01-s1235", seed=1235,
                    shape="the T-0.1B shape, seed 1235", params_total=103_996_416,
                    params_non_embedding=95_083_520, lr_from="scratch", warmup_steps=2000, weight_decay=1e-3,
                    bn="train mode", aux_ctc=0.3, micro_audio_s=1600, wave="own 1x box after PREREG_numbers.json"),
    "study-t005": dict(family="aed", init_class="scratch", student="students/study/t005", seed=1234,
                       shape="encoder 10 layers, d 384, 6 x 64, FFN 1536, kernel 9; decoder 3 layers, D 384, 6 x 64, "
                             "FFN 1536; vocab 16,384 tied",
                       params_total=51_209_600, params_non_embedding=44_524_928, lr_from="scratch",
                       warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1600,
                       wave=2),
    "study-p03": dict(family="ctc", init_class="pruned_kept", student="students/study/p03", seed=1234,
                      shape="Parakeet encoder 16 of 24 layers [0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23], d 1024, "
                            "FFN 4096 -> 2560; the Parakeet CTC head verbatim (3,073 classes)",
                      params_total=308_524_033, params_non_embedding=305_374_208, lr_from="kept-p03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1200, wave=1),
    "study-p01": dict(family="ctc", init_class="pruned_lost", student="students/study/p01", seed=1234,
                      shape="Parakeet encoder 8 layers [0,3,7,10,13,16,20,23], FFN 768; CTC head",
                      params_total=98_468_865, params_non_embedding=95_319_040, lr_from="lost",
                      warmup_steps=1000, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1600, wave=2),
    "study-p005": dict(family="ctc", init_class="pruned_lost", student="students/study/p005", seed=1234,
                       shape="Parakeet encoder 4 layers [0,8,15,23], FFN 768; CTC head",
                       params_total=52_190_209, params_non_embedding=49_040_384, lr_from="lost",
                       warmup_steps=1000, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                       micro_audio_s=1600, wave=2),
}

# ------------------------------------------------------------------------------------------------ the LR probes

# class -> the run shape it is probed on, its grid and its WSD schedule (warm-up / stable / cooldown steps; the
# warm-up always ends before the cooldown, the probe bug of the draft). STUDY.md 2.3
PROBES = {
    "scratch": dict(probed_on="study-t01", grid=[5e-4, 1e-3, 2e-3], max_steps=5000, warmup=2000, stable=2000,
                    cooldown=1000),
    "bridge": dict(probed_on="study-bridge", grid=[4e-4, 1e-3], max_steps=5000, warmup=2000, stable=2000,
                   cooldown=1000),
    "lost": dict(probed_on="study-p01", grid=[1e-4, 3e-4, 1e-3], max_steps=3000, warmup=1000, stable=1400,
                 cooldown=600),
    "kept-t03": dict(probed_on="study-t03", grid=[1e-4, 2e-4], max_steps=2000, warmup=300, stable=1300, cooldown=400),
    "kept-p03": dict(probed_on="study-p03", grid=[1e-4, 2e-4], max_steps=2000, warmup=300, stable=1300, cooldown=400),
}
EDGE_FACTOR = 2.0  # a winner at a grid edge gets one more point: x2 at the top, /2 at the bottom

# --------------------------------------------------------------------------------------------- calibration

CALIB_KEYS = ("t_step_s", "micro_audio_s", "data_wait_frac", "steps_measured")
CALIB_STEPS = (50, 250)  # t_i = the median step time over these steps, logging included
DATA_WAIT_MAX = 0.05  # at or above: perf.num_workers 12 and calibrate again; still at or above: the box halts

# ------------------------------------------------------------------------------------------------ the selection

# the recipe block every study run config carries as selection_recipe.study (scripts/make_selection.py applies it,
# vast/launch.py checks the selection was built with it). probe_n is per train source, as the existing probe convention
# draws it: 300 x the five sources = 1,500 kept rows
STUDY_SELECTION = {"f1a_max": 0.5, "dedup_min_chars": 15, "draw_audio_s": 3_600_000, "probe_n": 300,
                   "neutral_max_cer": 0.5}
# drop reasons of a study selection, after the existing ones (truncated, not_judged, no_agree, agree>A, no_audio)
STUDY_REASONS = ("not_in_parakeet", "f1a_disagree", "eval_dup", "ctc_infeasible", "not_drawn")
MANIFEST_FILE = "study_manifest.json"  # next to the selection
ONE_ROOT_MAX_FRAC = 0.001  # K5: train rows in teacher_out only (not_in_parakeet) above this share refuse the launch
STUDY_SOURCES = ["reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame"]
STUDY_EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"]
GATE_SETS = ("eval_jsut", "eval_cv8", "eval_reazon")
M4_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", "galgame:neutral")
POOL_MIN_HOURS = 1010  # the reazon_large cap: the smallest N whose pool after all filters holds >= this
COHERE_CER_PREREG = {"eval_jsut": 0.0830, "eval_cv8": 0.0407, "eval_reazon": 0.0628}  # kitsune.evaluate's, 0.05 pp

# ------------------------------------------------------------------------------------------------ the decisions

# decision -> (the accepted option, what it means); the owner accepted every recommendation (study/decisions.json)
DECISIONS = {
    1: ("a", "the limit: tolerance delta 10 % against the family's largest student; delta 5 and 20 % also reported"),
    2: ("S3", "one 4x A100 SXM4 40 GB box, calibration + probes + 2 waves; S3b, then S1p as fallbacks"),
    3: ("a", "control runs: the bridge and a replicate of T-0.1B with seed 1235"),
    4: ("a", "budget cuts in this order: the replicate, T at 3 epochs, the contingency; never stage away 0.05B"),
    5: ("b", "HF storage: no action (HF PRO, 1 TB)"),
    6: ("a", "a T/2 branch in every run"),
    7: ("a", "LR probes for all classes plus the bridge"),
    8: ("a", "PREREG numbers written by the box; it halts only on a check fixed in advance"),
    9: ("D", "mix D, reazon_large about 50-55 inputs"),
    10: ("a", "M4 within a family; JSUT+Galgame across families, descriptive only"),
    11: ("a", "Galgame neutral view (cer(kotoba, ref) <= 0.5) as primary; the other two reported"),
    12: ("a", "P-0.1B and P-0.05B as the d = 1024 ladder, B8x768 and B4x768"),
    13: ("F1a", "shared agreement filter CER(Transcribe hyp, Parakeet TDT hyp) <= 0.5"),
    14: ("t15", "eval-leak dedup: exact normalised text match against eval references of >= 15 characters"),
    15: ("a", "frame mismatches: drop and count; hard fail above 0.1 % of train rows or on any eval row"),
    16: ("teacher", "BN of pruned students: teacher stats, frozen; recal for both pruned Transcribe students if "
                    "T-0.6B's CPU step-0 KL is not lower with the teacher stats"),
    17: ("a", "T-0.3B = B10x2560 + decoder {0,7}"),
    18: ("W", "from-scratch shapes W: 512x12/4 and 384x10/3"),
    19: ("a", "re-initialise the subsampling conv; keep the 16,384 tied vocabulary"),
    20: ("yes", "aux CTC 0.3 for the scratch AED runs, the bridge included"),
    21: ("a", "dropout 0 and the first run's SpecAugment for all runs"),
    22: ("greedy", "Parakeet CTC target: the greedy CTC path"),
    23: ("w08", "w_ctc 0.8, no ablation"),
    24: ("switch", "the CTC trainer is a family switch inside 04_distill"),
    25: ("0", "L2-SP 0 for all runs"),
    26: ("steps", "equal compute on the steps clock with calibrated max_steps"),
    27: ("laptop", "students built on the laptop CPU"),
    28: ("a100", "the speed probe on the A100"),
    29: ("rescore", "re-score the first run's 0.6B (step_9774) on the manifest"),
    30: ("cond", "follow-ups conditional on their triggers only"),
}


# ================================================================================================= the rules


def lr_tag(lr: float) -> str:
    """An LR as the run names spell it: 1e-3, 4e-4, 2.5e-4 (probe-<class>-<tag>)."""
    mant, _, exp = f"{float(lr):.6e}".partition("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


def probe_run_name(cls: str, lr: float) -> str:
    return f"probe-{cls}-{lr_tag(lr)}"


def _pending_selection() -> dict:
    return {"status": PENDING,
            "fill": "python -m kitsune.prereg --write study/ --sidecar labels/full/selections/study_1000h.json, "
                    "from the selection built on the sealed labels (scripts/make_selection.py --config "
                    "study/data.json)",
            "extent_inputs": PENDING, "selection_sha256": PENDING, "manifest_sha256": PENDING,
            "ids_sha256": {"train": PENDING, "probe": PENDING, **{s: PENDING for s in STUDY_EVAL_SETS}},
            "n": {"train": PENDING, "probe": PENDING, **{s: PENDING for s in STUDY_EVAL_SETS}},
            "galgame_views": {v: {"n": PENDING, "ids_sha256": PENDING} for v in ("neutral", "all", "label_box")},
            "train_hours": PENDING, "pool_hours": PENDING}


def _pending_baselines() -> dict:
    per = {s: PENDING for s in (*GATE_SETS, "eval_emilia", "galgame:neutral", "galgame:all", "galgame:label_box")}
    return {"status": PENDING, "fill": "the sidecar's baselines: corpus CER (kitsune.evaluate.corpus_cer) on the "
                                       "manifest rows",
            "cohere": dict(per, m4=PENDING), "parakeet_ctc": dict(per, m4=PENDING),
            "parakeet_tdt": dict(per, m4=PENDING)}


def rules(sidecar: dict | None = None) -> dict:
    """Everything the study fixes in advance (STUDY.md 4.8 point 1). Without `sidecar` the fields that need the sealed
    labels are "pending"; with the study selection's sidecar (make_selection.py's study_1000h.json) they are filled
    from it. Pure data, deterministic: the same sidecar gives the same rules."""
    r = {
        "prereg_version": RULES_VERSION,
        "study": "Kitsune size study: 7 students + bridge + replicate on the same 1,000 h at equal A100 compute",
        "design": "STUDY.md (final); CONTRACT.md for the interfaces",
        "decisions": {str(n): {"option": k, "answer": a} for n, (k, a) in DECISIONS.items()},
        "runs": {run: dict(spec) for run, spec in RUNS.items()},
        "waves": {"1": ["study-t06", "study-t03", "study-p03", "study-bridge"],
                  "2": ["study-t01", "study-t005", "study-p01", "study-p005"],
                  "replicate": {"run": REPLICATE, "when": "on its own 1x box once PREREG_numbers.json exists",
                                "max_steps": f"{REPLICATE_OF}'s", "lr": f"{REPLICATE_OF}'s"}},
        "training": {
            "all_runs": {"optimiser": "AdamW beta 0.9/0.98, eps 1e-8, clip 1.0",
                         "schedule": "WSD on the steps clock: linear warm-up, stable, 1 - sqrt cooldown over the last "
                                     "20 % of max_steps",
                         "step_audio_s": 1500, "specaugment": "the first run's: 2 frequency masks of 27, 2-5 time "
                                                              "masks of 5 %", "dropout": 0.0, "layerdrop": 0.0,
                         "l2sp_lambda": 0.0, "early_stop": False, "subset.train_audio_s": None,
                         "memory": "the memory probe confirms each micro_audio_s; aux-CTC runs fall back to micro "
                                   "1200 if 1600 does not fit, and calibration measures the consequence"},
            "aed_loss": {"w_kl": 1.0, "w_ce": 0.8, "kl": "17 bins: the stored top-16 plus a rest bucket, T = 1",
                         "aux_ctc": {"weight": "0.3 for the scratch runs (bridge included), 0 otherwise",
                                     "blank": 16384, "head": "training-only nn.Linear(d_enc, 16,385), dropped at "
                                                             "export", "targets": "the teacher's stored greedy ids"}},
            "ctc_loss": {"w_kl": 1.0, "w_ctc": 0.8, "normaliser": "N_u, the CTC target tokens of the step",
                         "temperature": 1.0, "dense_frames": "p_blank < 0.95: the stored top-8 union {blank} plus a "
                                                             "rest bucket", "blank_frames": "2-bin KL {blank, "
                                                                                            "not-blank}",
                         "ctc": "F.ctc_loss(log_softmax fp32, reduction='sum', blank=3072, zero_infinity=True)",
                         "target": "Parakeet's greedy CTC path, ctc_greedy(ctc_col0(...)) = the jsonl ctc_hyp",
                         "alignment": "ParakeetFeatureExtractor features (80 mel, preemphasis 0.97, per-feature "
                                      "normalisation, no dither); no speed/tempo change, time warp, crop or concat"},
            "per_class": {
                "pruned_kept": {"runs": ["study-t06", "study-t03", "study-p03"], "warmup_steps": 300,
                                "weight_decay": 0.0, "bn": "teacher stats, eval mode (affine trains)"},
                "pruned_lost": {"runs": ["study-p01", "study-p005"], "warmup_steps": 1000, "weight_decay": 0.0,
                                "bn": "teacher stats, eval mode (affine trains)",
                                "definition": "a pruned student whose step-0 CER against its teacher is >= 90 % on "
                                              "the 60-utterance CPU gate"},
                "scratch": {"runs": ["study-t01", REPLICATE, "study-t005", "study-bridge"], "warmup_steps": 2000,
                            "weight_decay": 1e-3, "weight_decay_on": "parameters with ndim >= 2 only",
                            "bn": "train mode; evals use the running stats; grad_ckpt off"}},
            "bn_fallback": DECISIONS[16][1],
        },
        "lr_probes": {
            "classes": {c: dict(p, runs=sorted(run for run, s in RUNS.items() if s["lr_from"] == c))
                        for c, p in PROBES.items()},
            "objective": "the family's training objective per target token, teacher-forced, at the probe's end, "
                         "pooled over the complete gate sets eval_jsut, eval_cv8, eval_reazon (AED: w_kl KL + w_ce CE "
                         "without aux CTC; CTC: w_kl KL + w_ctc CTC); lowest wins; a non-finite objective loses; a "
                         "tie goes to the lower LR",
            "edge_rule": "a winner at an edge of the tested grid gets one more point (x2 above, /2 below); a winner "
                         "at an edge after that extension halts the box (an invalidation rule)",
            "run_names": "probe-<class>-<lr>, lr like 1e-3 / 4e-4",
        },
        "calibration": {
            "reference_run": T_REF_RUN, "reference_steps": REF_STEPS,
            "max_steps": f"max_steps_i = round({REF_STEPS} x t_{T_REF_RUN} / t_i), half up",
            "t_i": f"the median step time over steps {CALIB_STEPS[0]}-{CALIB_STEPS[1]} at the planned micro_audio_s, "
                   f"logging included, on the study host with the run's wave group running concurrently",
            "min_steps_measured": CALIB_STEPS[1] - CALIB_STEPS[0], "data_wait_max": DATA_WAIT_MAX,
            "loader_bound": f"data_wait_frac >= {DATA_WAIT_MAX}: perf.num_workers 12 and calibrate again; still "
                            f">= {DATA_WAIT_MAX}: the box halts",
            "excluded": "evals, checkpoints and the T/2 branch do not count toward T",
            "replicate": f"{REPLICATE} takes {REPLICATE_OF}'s max_steps",
        },
        "branch": {"resume_frac": 0.4, "end_frac": 0.5, "t_c": "the resume step",
                   "what": "<run>-half resumes the local full state at 0.4 x max_steps and cools down to 0.5 x "
                           "max_steps: the WSD schedule of a run whose budget is T/2 (same warm-up, stable LR and "
                           "data order, 20 % cooldown); one complete eval at its end"},
        "evals": {"complete_at_fracs": [0.2, 0.4, 0.6, 0.8], "final": "after the cooldown, at max_steps",
                  "mini_every_frac": 0.025, "checkpoint": "the final one, never the best eval; plus the branch's",
                  "reported": "per set, group and pooled: corpus CER, ratio to the own teacher, imitation CER, "
                              "no-style CER, S/D/I, runaway rate (CER > 100 % or hyp > 2 x ref), empty-output rate, "
                              "AED truncation; teacher-forced KL/CE (AED) or dense/blank KL, CTC per token, argmax "
                              "agreement and argmax-blank share (CTC); probe vs held-out KL; step-seconds, steps, "
                              "epochs, audio-hours; params total and non-embedding; speed and VRAM"},
        "data": {
            "extent": {"name": "full", "root": "labels/full",
                       "inputs": {"reazon_large": PENDING, "emilia_yodas": "300h", "emilia_nc": 8, "galgame": 3}},
            "reazon_large_cap": f"the smallest reazon_large input count whose pool after all filters holds >= "
                                f"{POOL_MIN_HOURS} h (expected 50-55), read from the sealed labels",
            "sources": list(STUDY_SOURCES), "eval_sets": list(STUDY_EVAL_SETS),
        },
        "selection": {
            "file": "labels/full/selections/study_1000h.parquet (+ study_1000h.json, study_manifest.json)",
            "recipe": {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "emilia_nc=0.2"],
                       "filter_eval_sets": [], "partial_second_opinion": [], "study": dict(STUDY_SELECTION)},
            "seed": 1234,
            "rules": [
                "1 not_in_parakeet: candidates are the extent's rows present in both teacher_out and parakeet_out",
                "2 F0, the label box's judges: truncated (Cohere, no EOS); no_agree; agree>A with A = 0.5 for Reazon "
                "(whisper-large-v3) and Galgame (Parakeet), 0.2 for Emilia-YODAS and NC (Emilia's WhisperX text)",
                "3 f1a_disagree: CER(Transcribe hyp, Parakeet TDT hyp) > 0.5, normalised by Parakeet's length",
                "4 eval_dup: the normalised reference or Cohere hypothesis equals an eval or hold-out reference of "
                ">= 15 characters",
                "5 ctc_infeasible: U + adjacent repeats of the greedy CTC target > n_frames",
                "6 not_drawn: keep = a seeded 3,600,000 s subset pooled over the sources (the frozen train list; "
                "subset.train_audio_s null for all runs); probe = 300 kept rows per train source (1,500)",
                "7 eval sets: every row present in both roots, unfiltered (filter_eval_sets []); views at scoring",
            ],
            "reasons": ["kept", "truncated", "no_agree", "agree>A", "no_audio", *STUDY_REASONS],
            "coverage": f"launch refuses a selection with any eval row not_in_parakeet (K6: both passes label the eval "
                        f"sets whole) or more than {100 * ONE_ROOT_MAX_FRAC:g} % of its train rows not_in_parakeet "
                        f"(K5)",
            "frame_preflight": "at store build, for every row: the ParakeetFeatureExtractor length of the rebuilt "
                               "audio, 8x subsampled, equals the stored n_frames; mismatches are dropped and counted, "
                               "the box fails above 0.1 % of train rows or on any eval row (decision 15)",
        },
        "manifest": _pending_selection(),
        "metrics": {
            "m4": "the macro mean of complete-set corpus CER over eval_jsut, eval_cv8, eval_reazon and Galgame-neutral,"
                  " equal weights (within-family primary)",
            "qualifiers": {"out_of_domain": ["eval_jsut", "eval_cv8"], "in_domain": ["eval_reazon", "galgame:neutral"]},
            "cross_family": "JSUT + Galgame-neutral, raw and no-style, paired CIs; descriptive only, no threshold",
            "galgame_views": {"neutral": "primary: the rows whose kotoba-whisper-v2.0 hypothesis (the laptop's "
                                         "second_out/galgame/eval-00000.jsonl) has cer(hyp2, ref) <= 0.5 and a "
                                         "non-empty reference",
                              "all": "every Galgame hold-out row of the manifest",
                              "label_box": "the label box's filter: not truncated and agree(Cohere, Parakeet) <= 0.5"},
            "cer": "kitsune.evaluate.corpus_cer: sum(S+D+I) / sum(ref chars) on normalize_ja text, empty refs "
                   "skipped",
        },
        "noise": {"sigma_prior": 0.016, "replicate_estimate": "0.886 x |ln(M4_replicate / M4_study-t01)|",
                  "sigma_rule": "sigma_run = max(0.016, the replicate's estimate)",
                  "ci_student_ratio": "ln r +- 1.96 sqrt(v_boot + 2 sigma_run^2)",
                  "ci_student_vs_teacher": "ln r +- 1.96 sqrt(v_boot + sigma_run^2)",
                  "bootstrap": {"B": 10000, "seed": 1234, "kind": "paired utterance bootstrap, stratified: "
                                                                 "resampled within each set, the same indices for "
                                                                 "every system"}},
        "limit_rule": {
            "families": {"aed": {"top": "study-t06", "walk": ["study-t03", "study-t01", "study-t005"]},
                         "ctc": {"top": "study-p03", "walk": ["study-p01", "study-p005"]}},
            "ratio": "r_s = M4_s / M4_top",
            "delta": 0.10, "delta_reported": [0.05, 0.10, 0.20],
            "calls": {"WITHIN": "the CI's upper end <= ln(1 + delta)", "OUTSIDE": "the CI's lower end > ln(1 + delta)",
                      "UNRESOLVED": "otherwise"},
            "walk": "down the sizes: the limit is the smallest size WITHIN with every larger size WITHIN; the first "
                    "OUTSIDE ends it (limit between X and Y); an UNRESOLVED gives 'at or below X, Y unresolved' and "
                    "the smaller sizes are descriptive; one decisive claim per family, no Holm correction",
            "replicate": f"{REPLICATE} feeds sigma_run only; the walk uses {REPLICATE_OF}",
        },
        "readouts": {
            "transcribe_ladder": "T-0.6B -> T-0.3B (pruned) -> T-0.1B -> T-0.05B (scratch)",
            "init_effect": "M4(bridge) / M4(T-0.3B)",
            "scratch_ladder": "bridge -> T-0.1B -> T-0.05B, a second tolerance walk with top = bridge; the "
                              "acceleration g(0.1->0.05) - g(0.3s->0.1) with a CI, reported only here",
            "parakeet_labels": "P 0.3 -> 0.1 is always labelled 'size + loss of function + depth'; P 0.1 -> 0.05 "
                               "is clean",
            "distillation_gaps": "M4(T-0.6B)/M4(Cohere) and M4(P-0.3B)/M4(Parakeet CTC); not steps of the walk",
            "g_per_halving": "every step with a CI; descriptive only",
            "budget": "per non-top size the call at T/2 and at T; Delta = ln r(T) - ln r(T/2) with CI (v_boot + 2 "
                      "sigma_run^2): below 0 -> 'compute-limited: the gap closes with compute', else 'not "
                      "compute-limited at T'",
        },
        "practical_bars": ["the smallest student whose JSUT+Galgame-neutral CI upper end is <= Parakeet 0.6B TDT",
                           "the smallest student within 1.2x and within 1.5x of its own teacher on M4",
                           "a Pareto view: CER against A100 RTF and VRAM"],
        "anchor": {"model": "the first run's 0.6B, step_9774", "flag": "the study's T-0.6B has a gate-pooled CER "
                                                                       "worse than the anchor's by > 5 % relative "
                                                                       "with a paired CI excluding 0",
                   "then": "interpretation halts until the cause (BN, L2-SP, LR, data) is understood"},
        "baselines": _pending_baselines(),
        "cohere_prereg": dict(COHERE_CER_PREREG, tolerance_pp=0.05,
                              note="K11: recomputed from labels/full/teacher_out; a drift means re-registering the "
                                   "baselines or reusing the laptop's eval labels before this commit"),
        "halts": ["calibration still loader-bound after perf.num_workers 12",
                  "an LR probe winner at a grid edge after its one extension",
                  "the frame preflight above 0.1 % of train rows or on any eval row",
                  "a selection, manifest or PREREG hash that differs from this file",
                  "any rule field still pending"],
        "invalid_if": ["a run misses its max_steps, or resumes with a changed config",
                       "a selection or manifest hash differs between runs",
                       "the teacher baselines do not reproduce",
                       "K3 or K4 fails, or the frame preflight exceeds its threshold",
                       "this rules commit comes after any study result, or PREREG_numbers.json after the first study "
                       "step",
                       "the anchor regression flag fires and stays unexplained",
                       "an LR winner sits at a grid edge after its one extension",
                       "calibration stays loader-bound after raising the workers"],
        "budget": {"cut_order": ["the replicate", "T at 3 epochs", "the contingency"],
                   "follow_ups": ["a d-pruned P-0.1B if P-0.1B is OUTSIDE",
                                  "a 2T extension from the pre-cooldown state if T/2 -> T says compute-limited",
                                  "a second replicate if a decisive call falls within about 1 sigma of delta"],
                   "follow_up_rule": "conditional on the trigger, and only if the contingency is unused"},
        "numbers": {"file": NUMBERS_JSON, "keys": ["calibration", "max_steps", "lr_probes", "lr", "rules_sha256",
                                                   "written_utc", "host"],
                    "when": "written and uploaded by the study box, its sha256 logged as an event, before the first "
                            "study step"},
    }
    if sidecar is not None:
        _fill(r, sidecar)
    return r


def _fill(r: dict, sc: dict):
    """The pending fields from the study selection's sidecar (scripts/make_selection.py): the extent it was built on,
    the file and id hashes, the Galgame views and the teacher baselines (corpus CER) on the manifest rows."""
    ids, n = sc["ids_sha256"], sc["n"]
    views = sc.get("galgame_views") or {}
    inputs = _inputs_in_order(dict((sc.get("extent") or {}).get("inputs") or {}))
    r["manifest"] = {
        "status": "filled", "from": sc["selection"]["path"], "extent_inputs": inputs,
        "selection_sha256": sc["selection"]["sha256"], "manifest_sha256": sc["manifest"]["sha256"],
        "ids_sha256": {"train": ids["train"], "probe": ids["probe"], **{s: ids["eval"][s] for s in ids["eval"]}},
        "n": {"train": n["train"], "probe": n["probe"], **{s: n["eval"][s] for s in n["eval"]}},
        "galgame_views": {v: {"n": views[v]["n"], "ids_sha256": views[v]["ids_sha256"]} for v in sorted(views)},
        "train_hours": sc["draw"]["drawn_s"] / 3600, "pool_hours": sc["draw"]["pool_s"] / 3600,
    }
    r["data"]["extent"]["inputs"] = inputs
    b = sc.get("baselines") or {}
    r["baselines"] = {"status": "filled", "from": "the sidecar's baselines on the manifest rows",
                      **{system: {k: v["cer"] if isinstance(v, dict) else v for k, v in b[system].items()}
                         for system in sorted(b)}}


def _inputs_in_order(inputs) -> dict | str:
    """extent.inputs in the rules' order (the mix's sources), whichever order a JSON file or a sidecar has them in."""
    if not isinstance(inputs, dict):
        return inputs
    order = ["reazon_large", "emilia_yodas", "emilia_nc", "galgame"]
    return {k: inputs[k] for k in [*[o for o in order if o in inputs], *sorted(set(inputs) - set(order))]}


def _filled_parts(r: dict) -> dict:
    """The parts of a rules dict that the sidecar fills (pending until the labels are sealed)."""
    return {"manifest": r.get("manifest"), "baselines": r.get("baselines"),
            "extent_inputs": ((r.get("data") or {}).get("extent") or {}).get("inputs")}


def regenerate(committed: dict) -> dict:
    """rules() with the filled parts taken from a committed PREREG.json: everything else must equal the code, so a
    committed file whose rule part was edited by hand (or is stale) differs from this."""
    r = rules()
    parts = _filled_parts(committed)
    r["manifest"], r["baselines"] = parts["manifest"], parts["baselines"]
    r["data"]["extent"]["inputs"] = _inputs_in_order(parts["extent_inputs"])
    return r


def pending(obj, where: str = "") -> list[str]:
    """The dotted paths of every field that is still "pending"."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in pending(v, f"{where}{k}.")]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in pending(v, f"{where}{i}.")]
    return [where.rstrip(".")] if obj == PENDING else []


def rules_json(r: dict) -> bytes:
    """The canonical bytes of the rules: write_rules writes exactly this as PREREG.json."""
    return (json.dumps(r, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def rules_sha256(path) -> str:
    """sha256 of a PREREG.json's canonical bytes (rules_json of its parsed content), not of the file as checked out:
    git on the Windows laptop checks text files out with CRLF (core.autocrlf), the Linux box with LF, and both must
    log the same rules_sha256."""
    return hashlib.sha256(rules_json(json.loads(Path(path).read_text(encoding="utf-8")))).hexdigest()


def _fmt(v) -> str:
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, float):
        return lr_tag(v) if 0 < abs(v) < 0.01 else f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def _flat(v) -> bool:
    """A value that fits on one bullet: a scalar, or a short list of scalars."""
    if isinstance(v, list):
        return all(not isinstance(x, (dict, list)) for x in v) and len(_fmt(v)) <= 120
    return not isinstance(v, dict)


def _md_tree(obj, depth: int = 0) -> list[str]:
    """Nested bullets: a dict's keys in bold, a list's items as plain bullets."""
    pad, out = "  " * depth, []
    for k, v in (obj.items() if isinstance(obj, dict) else enumerate(obj, 1)):
        head = f"{pad}- **{k}**" if isinstance(obj, dict) else f"{pad}-"
        if _flat(v):
            out.append(f"{head}: {_fmt(v)}" if isinstance(obj, dict) else f"{head} {_fmt(v)}")
        else:
            out.append(head)
            out += _md_tree(v, depth + 1)
    return out


def _sorted(obj):
    if isinstance(obj, dict):
        return {k: _sorted(obj[k]) for k in sorted(obj)}
    return [_sorted(x) for x in obj] if isinstance(obj, list) else obj


FILLED_SECTIONS = ("manifest", "baselines")  # rendered with sorted keys: filled from the sidecar or read back from JSON


def rules_md(r: dict) -> str:
    """PREREG.md: the same rules for reading (PREREG.json is the binding form)."""
    left = sorted(pending(r))
    lines = ["# Size study: pre-registration (rules)", "",
             f"Generated by `python -m kitsune.prereg --write study/` from `kitsune/prereg.py`; PREREG.json is the "
             f"binding form (its sha256 goes into PREREG_numbers.json as `rules_sha256`). Rules version "
             f"{r['prereg_version']}.", "",
             (f"**{len(left)} field(s) still pending** (they need the sealed labels; the final pre-launch commit "
              f"fills them with `--sidecar`): " + ", ".join(f"`{p}`" for p in left[:12]) +
              (" ..." if len(left) > 12 else "") if left else "**Complete**: no field is pending."), ""]
    lines += ["## Runs", "", "| run | family | init | params total | non-embedding | LR from | warm-up | wd | aux CTC "
                             "| micro | wave |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, s in r["runs"].items():
        lines.append(f"| {run} | {s['family']} | {s['init_class']} | {s['params_total']:,} | "
                     f"{s['params_non_embedding']:,} | {s['lr_from']} | {s['warmup_steps']} | {_fmt(s['weight_decay'])}"
                     f" | {_fmt(s['aux_ctc'])} | {s['micro_audio_s']} | {s['wave']} |")
    lines += ["", "Shapes:", ""] + [f"- **{run}**: {s['shape']}; seed {s['seed']}; BN {s['bn']}"
                                    for run, s in r["runs"].items()]
    lines += ["", "## LR probes", "", "| class | probed on | grid | steps | warm-up / stable / cooldown | runs |",
              "|---|---|---|---|---|---|"]
    for c, p in r["lr_probes"]["classes"].items():
        lines.append(f"| {c} | {p['probed_on']} | {', '.join(lr_tag(x) for x in p['grid'])} | {p['max_steps']:,} | "
                     f"{p['warmup']} / {p['stable']} / {p['cooldown']} | {', '.join(p['runs'])} |")
    lines += ["", f"- **objective**: {r['lr_probes']['objective']}", f"- **edge rule**: {r['lr_probes']['edge_rule']}",
              ""]
    titles = [("decisions", "Owner decisions (all at the recommendation)"), ("waves", "Waves"),
              ("training", "Training"), ("calibration", "Calibration and max_steps"), ("branch", "The T/2 branch"),
              ("evals", "Evals"), ("data", "Data"), ("selection", "Selection"), ("manifest", "Manifest and hashes"),
              ("metrics", "Metrics"), ("noise", "Noise model"), ("limit_rule", "The limit rule (primary answer)"),
              ("readouts", "Decomposition and readouts"), ("practical_bars", "Practical bars"), ("anchor", "Anchor"),
              ("baselines", "Teacher baselines on the manifest"), ("cohere_prereg", "Cohere baselines registered "
                                                                                    "by the first run"),
              ("halts", "The box halts only when"), ("invalid_if", "The conclusion is invalid if"),
              ("budget", "Budget"), ("numbers", "Numbers written by the box")]
    for key, title in titles:
        lines += [f"## {title}", ""] + _md_tree(_sorted(r[key]) if key in FILLED_SECTIONS else r[key]) + [""]
    return "\n".join(lines).rstrip() + "\n"


def _write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_rules(out_dir, sidecar: dict | None = None) -> str:
    """study/PREREG.json + study/PREREG.md; returns the rules' sha256 (rules_sha256 of the written PREREG.json)."""
    out_dir = Path(out_dir)
    r = rules(sidecar)
    data = rules_json(r)
    _write_bytes(out_dir / RULES_JSON, data)
    _write_bytes(out_dir / RULES_MD, rules_md(r).encode("utf-8"))
    return hashlib.sha256(data).hexdigest()


# ================================================================================================ the numbers


def calibration_problems(calib: dict) -> list[str]:
    """Why a calibration table cannot give max_steps: a run of the waves missing, a key missing, a step time that is
    not a positive number, fewer steps than the window, or a loader-bound run (data_wait_frac >= DATA_WAIT_MAX: raise
    perf.num_workers to 12 and calibrate again; if it stays, the box halts)."""
    problems = []
    want = [run for run in RUNS if run != REPLICATE]
    for run in want:
        if run not in calib:
            problems.append(f"{run}: not calibrated")
    for run, c in calib.items():
        if run not in RUNS:
            problems.append(f"{run}: not a study run")
            continue
        if missing := [k for k in CALIB_KEYS if k not in c]:
            problems.append(f"{run}: missing {missing}")
            continue
        t = c["t_step_s"]
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t <= 0:
            problems.append(f"{run}: t_step_s {t!r} is not a positive number")
        if c["steps_measured"] < CALIB_STEPS[1] - CALIB_STEPS[0]:
            problems.append(f"{run}: {c['steps_measured']} steps measured, the window is "
                            f"{CALIB_STEPS[1] - CALIB_STEPS[0]}")
        if c["data_wait_frac"] >= DATA_WAIT_MAX:
            problems.append(f"{run}: loader-bound (data_wait_frac {c['data_wait_frac']:.3f} >= {DATA_WAIT_MAX})")
    return problems


def max_steps(calib: dict, t_ref_run: str = T_REF_RUN, ref_steps: int = REF_STEPS) -> dict[str, int]:
    """run -> max_steps = round(ref_steps x t_ref / t_run), half up: every run gets the reference run's step-time
    budget T (STUDY.md 5.1). `calib` is {run: {"t_step_s": ...}} as measured on the study host; the replicate, when
    not calibrated itself, takes the run it replicates."""
    if t_ref_run not in calib:
        raise ValueError(f"the reference run {t_ref_run} is not calibrated")
    t_ref = float(calib[t_ref_run]["t_step_s"])
    out = {}
    for run, c in calib.items():
        t = float(c["t_step_s"])
        if not (math.isfinite(t) and t > 0 and math.isfinite(t_ref) and t_ref > 0):
            raise ValueError(f"{run}: step time {t} (reference {t_ref}) is not a positive number")
        out[run] = int(math.floor(ref_steps * t_ref / t + 0.5))
    if REPLICATE not in out and REPLICATE_OF in out:
        out[REPLICATE] = out[REPLICATE_OF]
    return out


_derive_max_steps = max_steps  # write_numbers' parameter of the same name (the contract's) shadows the function


def _key(lr) -> float:
    """An LR key (a float, or a tag like "1e-3" as JSON stores it) as a float rounded to 12 significant digits, so
    that 2 x 1e-3 and "2e-3" are one grid point."""
    return float(f"{float(lr):.12g}")


def _winner(results: dict[float, float]) -> float:
    def score(lr):
        v = results[lr]
        return (v if isinstance(v, (int, float)) and math.isfinite(v) else math.inf, lr)

    return min(results, key=score)


def choose_lr(probe_results: dict) -> dict:
    """class -> the LR decision from its probes {lr: final objective} (STUDY.md 2.3), by the edge rule:
      {"decision": "chosen", "lr": w}         the winner w lies inside the tested grid;
      {"decision": "extend", "next_lr": x}    the winner of the pre-registered grid sits at its edge: probe one more
                                              point, x = w x 2 (top edge) or w / 2 (bottom edge), then call again;
      {"decision": "halt", "lr": None}        after that one extension the winner is again at an edge.
    Every entry also carries the winner and the tested grid. Lowest objective wins; non-finite loses; a tie goes to
    the lower LR. Raises ValueError on results that do not follow the rule (a grid point missing, an extra point
    other than the extension, more than one)."""
    out = {}
    for cls, res in probe_results.items():
        if cls not in PROBES:
            raise ValueError(f"{cls}: not a probe class ({', '.join(PROBES)})")
        res = {_key(lr): v for lr, v in res.items()}
        grid = [_key(x) for x in PROBES[cls]["grid"]]
        if missing := [lr_tag(x) for x in grid if x not in res]:
            raise ValueError(f"{cls}: no result for the grid point(s) {missing}")
        extra = sorted(set(res) - set(grid))
        base_w = _winner({lr: res[lr] for lr in grid})
        at_top, at_bottom = base_w == max(grid), base_w == min(grid)
        ext = _key(base_w * EDGE_FACTOR) if at_top else _key(base_w / EDGE_FACTOR) if at_bottom else None
        tested = sorted(res)
        entry = {"tested": tested, "grid": grid}
        if not extra:
            w = base_w
            entry.update(winner=w, decision="extend" if ext is not None else "chosen",
                         lr=None if ext is not None else w, next_lr=ext)
        else:
            if len(extra) > 1 or extra[0] != ext:
                raise ValueError(f"{cls}: tested {[lr_tag(x) for x in extra]} beyond the grid; the edge rule allows "
                                 f"only {lr_tag(ext) if ext is not None else 'nothing (the grid winner is inside)'}")
            w = _winner(res)
            edge = w in (tested[0], tested[-1])
            entry.update(winner=w, decision="halt" if edge else "chosen", lr=None if edge else w, next_lr=None)
        out[cls] = entry
    return out


def run_lrs(choices: dict) -> dict[str, float]:
    """run -> the peak LR its probe class chose (T-0.6B takes T-0.3B's, T-0.05B and the replicate T-0.1B's, P-0.05B
    P-0.1B's). Raises ValueError unless every class a run needs is "chosen"."""
    out = {}
    for run, spec in RUNS.items():
        c = choices.get(spec["lr_from"])
        if c is None or c.get("decision") != "chosen":
            raise ValueError(f"{run}: its probe class {spec['lr_from']} has no chosen LR "
                             f"({None if c is None else c.get('decision')})")
        out[run] = float(c["lr"])
    return out


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_numbers(path, calibration: dict, probes: dict, lrs: dict, max_steps: dict, *, rules_path=None,
                  host: str | None = None, allow_pending: bool = False) -> str:
    """PREREG_numbers.json (atomic); returns its sha256. Keys: calibration {run: {t_step_s, micro_audio_s,
    data_wait_frac, steps_measured}}, max_steps {run: int}, lr_probes {class: {lr tag: objective}}, lr {run: float},
    rules_sha256 (rules_sha256 of the committed study/PREREG.json, `rules_path`), written_utc, host (default the
    machine's node name).

    Mechanical by construction: it raises ValueError if the calibration has problems, if `max_steps` is not what
    max_steps() derives from the calibration, if `lrs` is not run_lrs(choose_lr(probes)) (so every class must be
    "chosen"), or if the rules still have a pending field (allow_pending=True only for dry runs)."""
    max_steps_given, max_steps = max_steps, _derive_max_steps  # the parameter keeps the contract's name
    rules_path = Path(rules_path) if rules_path is not None else REPO / "study" / RULES_JSON
    if not allow_pending and (left := pending(json.loads(rules_path.read_text(encoding="utf-8")))):
        raise ValueError(f"{rules_path} still has {len(left)} pending field(s), e.g. {left[:3]}: commit the filled "
                         f"rules (python -m kitsune.prereg --write study/ --sidecar ...) before the study box starts")
    if problems := calibration_problems(calibration):
        raise ValueError("calibration: " + "; ".join(problems))
    want_steps = max_steps(calibration)
    if {k: int(v) for k, v in max_steps_given.items()} != want_steps:
        raise ValueError(f"max_steps {max_steps_given} is not the calibrated {want_steps}")
    want_lr = run_lrs(choose_lr(probes))
    if {k: float(v) for k, v in lrs.items()} != want_lr:
        raise ValueError(f"lr {lrs} is not the probes' choice {want_lr}")
    numbers = {
        "calibration": {run: {k: calibration[run][k] for k in CALIB_KEYS} for run in sorted(calibration)},
        "max_steps": dict(sorted(want_steps.items())),
        "lr_probes": {cls: {lr_tag(lr): res[lr] for lr in sorted(res, key=_key)}
                      for cls, res in sorted(probes.items())},
        "lr": dict(sorted(want_lr.items())),
        "rules_sha256": rules_sha256(rules_path),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host if host is not None else platform.node(),
    }
    data = (json.dumps(numbers, sort_keys=True, indent=1) + "\n").encode("utf-8")
    _write_bytes(Path(path), data)
    return hashlib.sha256(data).hexdigest()


# ======================================================================================================== CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="DIR", help="write PREREG.json + PREREG.md into DIR (study/)")
    g.add_argument("--check", metavar="DIR", help="the files in DIR equal what --write would write; list pending")
    ap.add_argument("--sidecar", default=None, help="the study selection's sidecar (study_1000h.json): fills the "
                                                    "pending fields")
    args = ap.parse_args(argv)
    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8")) if args.sidecar else None
    if args.write:
        sha = write_rules(args.write, sidecar)
        left = pending(rules(sidecar))
        print(f"wrote {Path(args.write) / RULES_JSON} (sha256 {sha}) and {RULES_MD}; {len(left)} pending field(s)")
        return 0
    d = Path(args.check)
    ok, r = check_rules(d, sidecar)
    left = pending(r) if r is not None else ["(no PREREG.json)"]
    print(f"{d}: {'up to date' if ok else 'DIFFERS from kitsune/prereg.py'}; {len(left)} pending field(s)"
          + (": " + ", ".join(left) if left else ""))
    return 0 if ok else 1


def check_rules(d, sidecar: dict | None = None) -> tuple[bool, dict | None]:
    """(the committed PREREG.json and PREREG.md in `d` are what the code writes, the rules). Without `sidecar` the
    filled parts are taken from the committed file itself (regenerate)."""
    d = Path(d)
    if not (d / RULES_JSON).is_file():
        return False, None
    committed = json.loads((d / RULES_JSON).read_text(encoding="utf-8"))
    r = rules(sidecar) if sidecar is not None else regenerate(committed)
    # compared as parsed JSON and as text with universal newlines: a CRLF checkout is the same file
    ok = committed == json.loads(rules_json(r)) and (d / RULES_MD).is_file() and \
        (d / RULES_MD).read_text(encoding="utf-8") == rules_md(r)
    return ok, r


if __name__ == "__main__":
    sys.exit(main())
