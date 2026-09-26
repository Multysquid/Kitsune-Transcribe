"""The size study's pre-registration: the rules fixed before the study box starts, and the numbers the box writes.

Two layers, so that no number the study reports can be chosen after its results are seen (STUDY.md 4.8):

  rules      everything decided in advance: the runs and their exact parameter counts, the size ladder, the boxes, the
             training settings per init class and the step-0 gate, the LR probe grids and the edge rule, the
             calibration rule for max_steps, the T/2 branch, the selection rule and the eval manifest, metrics, the
             noise model, the limit rule, the practical bars, the anchor, the invalidation rules and all 30 owner
             decisions with the owner's changes of 2026-09-26. `rules()` returns them; `write_rules(dir)` writes
             study/PREREG.json (canonical JSON, the hashed form) and study/PREREG.md (the same content for reading).
             Both are committed to main by PR before the first study box launches.
  numbers    what only a study host can measure: the calibration table, max_steps per run, the LR probe objectives
             and the chosen LRs. The study runs on two boxes and a 1x box for the replicate (BOXES, the owner's
             decision of 2026-09-26); each computes its numbers mechanically with max_steps(box=...) and choose_lr() on
             its own host, then write_numbers(..., box=...) writes PREREG_numbers_<box>.json and returns its sha256,
             which the box logs as an event and uploads before its first study step. The replicate box writes its
             file from box A's (replicate_numbers). write_numbers refuses numbers that do not follow from the
             measurements by these rules, and rules that are still pending.

Some rule fields need the sealed labels (labels/full/COMPLETE.json): the manifest and selection hashes, the Galgame
view ids, the teacher baselines on the manifest and the final reazon_large cap. Until then they are the string
"pending" (pending() lists them). scripts/make_selection.py --config study/data.json writes all of them into the
selection's sidecar (labels/full/selections/study_1000h.json), and the final pre-launch commit fills them:

    python -m kitsune.prereg --write study/                                   # the rules, with pending fields
    python -m kitsune.prereg --write study/ --sidecar labels/full/selections/study_1000h.json   # the final commit
    python -m kitsune.prereg --check study/     # the committed files equal the generator's output; lists pending

The fill takes only the pre-registered selection (sidecar_problems): the registered sources, eval sets, recipe, seed
and extent, reazon_large at the cap the rule gives (the smallest N whose pool holds >= POOL_MIN_HOURS h, from the
sidecar's pool_hours_if_capped), and every eval set, view, teacher and stratum present. Otherwise a selection built
with another recipe could fill the hashes of a PREREG whose text says something else.

The layout is machine-read by kitsune.study_stats (tools/study_report.py --prereg): "analysis" holds delta, the
sigma_run rule and the bootstrap settings; "manifest" holds sets.<eval set>.ids_sha256, galgame_views.<view>.ids_sha256,
manifest_sha256 and selection_sha256; "baselines" holds <teacher>.<stratum> corpus CERs (teachers cohere,
parakeet-ctc, parakeet-tdt; strata the eval sets with Galgame as galgame_neutral / galgame_all / galgame_label_box;
plus m4), all as fractions.

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
RULES_VERSION = 2  # 2: the owner's changes of 2026-09-26 (T-0.3B shape, init class, 3-point grids, boxes)
# NUMBERS_JSON is the numbers file of the whole study on one host (write_numbers without a box); the study's boxes each
# write their own (BOXES[box]["numbers_file"]). NUMBERS_KEYS: every numbers file's keys; the replicate's adds
# numbers_from (box A's file and its sha256)
RULES_JSON, RULES_MD, NUMBERS_JSON = "PREREG.json", "PREREG.md", "PREREG_numbers.json"
NUMBERS_KEYS = ("box", "calibration", "max_steps", "lr_probes", "lr", "rules_sha256", "written_utc", "host")

# --------------------------------------------------------------------------------------------------- the runs

T_REF_RUN, REF_STEPS = "study-t06", 9366  # T = the T-0.6B's 4 epochs of 1,000 h (STUDY.md 5.1)
REPLICATE, REPLICATE_OF = "study-t01-s1235", "study-t01"  # the replicate trains T-0.1B's shape and steps, seed 1235

# run -> what the study fixes about it (CONTRACT.md section 1, STUDY.md 1.1 and 2.2). params_* are the exact meta-device
# counts every builder asserts; lr_from is the LR probe class whose winner the run takes; box is the host that trains
# it (BOXES). T-0.3B and the bridge are B8x2560 + decoder {0,2,5,7} by the owner's decision of 2026-09-26 (OWNER_CHANGES)
RUNS = {
    "study-t06": dict(family="aed", init_class="pruned_kept", student="students/study/t06", seed=1234,
                      shape="encoder 20 of 48 layers [0,2,5,7,10,12,15,17,20,22,25,27,30,32,35,37,40,42,45,47], "
                            "d 1280, FFN 5120 -> 2560; decoder layers {0,2,5,7}, D 1024; vocab 16,384 tied",
                      params_total=616_963_328, params_non_embedding=599_137_536, lr_from="kept-t03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=600, box="A"),
    "study-t03": dict(family="aed", init_class="pruned_kept", student="students/study/t03", seed=1234,
                      shape="encoder 8 of 48 layers [0,7,13,20,27,34,40,47] (evenly_spaced(8, 48)), d 1280, FFN 5120 "
                            "-> 2560; T-0.6B's decoder layers {0,2,5,7}, D 1024; vocab 16,384 tied",
                      params_total=301_822_208, params_non_embedding=283_996_416, lr_from="kept-t03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1200, box="A"),
    "study-bridge": dict(family="aed", init_class="scratch", student="students/study/bridge", seed=1234,
                         shape="the T-0.3B shape (encoder 8 layers, d 1280, 8 x 160, FFN 2560; decoder 4 layers, "
                               "D 1024, 8 x 128, FFN 4096; vocab 16,384 tied), from scratch",
                         params_total=301_822_208, params_non_embedding=283_996_416, lr_from="bridge",
                         warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1200,
                         box="B"),
    "study-t01": dict(family="aed", init_class="scratch", student="students/study/t01", seed=1234,
                      shape="encoder 12 layers, d 512, 8 x 64, FFN 2048, kernel 9; decoder 4 layers, D 512, 8 x 64, "
                            "FFN 2048; vocab 16,384 tied",
                      params_total=103_996_416, params_non_embedding=95_083_520, lr_from="scratch",
                      warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1600, box="A"),
    REPLICATE: dict(family="aed", init_class="scratch", student="students/study/t01-s1235", seed=1235,
                    shape="the T-0.1B shape, seed 1235", params_total=103_996_416,
                    params_non_embedding=95_083_520, lr_from="scratch", warmup_steps=2000, weight_decay=1e-3,
                    bn="train mode", aux_ctc=0.3, micro_audio_s=1600, box="replicate"),
    "study-t005": dict(family="aed", init_class="scratch", student="students/study/t005", seed=1234,
                       shape="encoder 10 layers, d 384, 6 x 64, FFN 1536, kernel 9; decoder 3 layers, D 384, 6 x 64, "
                             "FFN 1536; vocab 16,384 tied",
                       params_total=51_209_600, params_non_embedding=44_524_928, lr_from="scratch",
                       warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1600,
                       box="A"),
    "study-p03": dict(family="ctc", init_class="pruned_kept", student="students/study/p03", seed=1234,
                      shape="Parakeet encoder 16 of 24 layers [0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23], d 1024, "
                            "FFN 4096 -> 2560; the Parakeet CTC head verbatim (3,073 classes)",
                      params_total=308_524_033, params_non_embedding=305_374_208, lr_from="kept-p03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1200, box="B"),
    "study-p01": dict(family="ctc", init_class="pruned_lost", student="students/study/p01", seed=1234,
                      shape="Parakeet encoder 8 layers [0,3,7,10,13,16,20,23], FFN 768; CTC head",
                      params_total=98_468_865, params_non_embedding=95_319_040, lr_from="lost",
                      warmup_steps=1000, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1600, box="B"),
    "study-p005": dict(family="ctc", init_class="pruned_lost", student="students/study/p005", seed=1234,
                       shape="Parakeet encoder 4 layers [0,8,15,23], FFN 768; CTC head",
                       params_total=52_190_209, params_non_embedding=49_040_384, lr_from="lost",
                       warmup_steps=1000, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                       micro_audio_s=1600, box="B"),
}
# the fixed models the size ladder also measures against (STUDY.md 1.1 and 1.3; kitsune.study_stats.PARAMS_TOTAL):
# Cohere Transcribe itself and Parakeet's unpruned CTC path (24 x 4096), the two distillation gaps' far ends
TEACHER_PARAMS = {"cohere": 2_065_647_872, "parakeet-ctc": 610_898_945}
# STUDY.md 1.3: every step of the walks and the distillation gaps, (big, small, what changes besides size); the
# halvings h = log2 of the total-parameter ratio are computed from the counts above (size_ladder)
LADDER = (
    ("study-t06", "study-t03", "nothing (pruned -> pruned; only the encoder depth, 20 -> 8 of 48 layers)"),
    ("study-t03", "study-t01", "init (pruned -> scratch), LR, weight decay, aux CTC, BN mode, width"),
    ("study-bridge", "study-t01", "width and depth shape only (both scratch)"),
    ("study-t01", "study-t005", "nothing (scratch -> scratch)"),
    ("parakeet-ctc", "study-p03", "teacher -> student: the distillation gap, not in the walk"),
    ("study-p03", "study-p01", "function kept -> lost; depth 16 -> 8; LR and warm-up"),
    ("study-p01", "study-p005", "nothing"),
    ("cohere", "study-t06", "teacher -> student: the distillation gap, not in the walk"),
)

# ------------------------------------------------------------------------------------------------ the LR probes

# class -> the run shape it is probed on, its grid and its WSD schedule (warm-up / stable / cooldown steps; the
# warm-up always ends before the cooldown, the probe bug of the draft). STUDY.md 2.3. Every grid has 3 points up front
# (the owner's decision of 2026-09-26: the pruned classes' 1e-4 / 2e-4 and the bridge's 4e-4 / 1e-3 had only 2, so
# their winner always sat on an edge and always cost an extension)
PROBES = {
    "scratch": dict(probed_on="study-t01", grid=[5e-4, 1e-3, 2e-3], max_steps=5000, warmup=2000, stable=2000,
                    cooldown=1000),
    "bridge": dict(probed_on="study-bridge", grid=[2e-4, 4e-4, 1e-3], max_steps=5000, warmup=2000, stable=2000,
                   cooldown=1000),
    "lost": dict(probed_on="study-p01", grid=[1e-4, 3e-4, 1e-3], max_steps=3000, warmup=1000, stable=1400,
                 cooldown=600),
    "kept-t03": dict(probed_on="study-t03", grid=[1e-4, 2e-4, 4e-4], max_steps=2000, warmup=300, stable=1300,
                     cooldown=400),
    "kept-p03": dict(probed_on="study-p03", grid=[1e-4, 2e-4, 4e-4], max_steps=2000, warmup=300, stable=1300,
                     cooldown=400),
}
EDGE_FACTOR = 2.0  # a winner at a grid edge gets one more point: x2 at the top, /2 at the bottom

# ------------------------------------------------------------------------------------------------ the boxes

# The owner's decision of 2026-09-26 ("Cohere box first"): box A (4x A100) trains the four Transcribe students in one
# wave, each followed by its T/2 branch on the same GPU, probes kept-t03 and scratch, and re-scores the anchor in a gap;
# box B (4x A100, later) trains the three Parakeet students and the bridge the same way, probes lost, kept-p03 and
# bridge, calibrates study-t06 for CALIB_STEPS as its host's reference step time (without training it) and runs the
# speed probes of all 9 students + both teachers at its end, one model at a time; the replicate trains on a 1x box after
# box A with study-t01's max_steps and the scratch LR from box A's numbers. Each box measures t_study-t06 on its own
# host, so every max_steps is equal compute on the host that trains the run. Consumers (vast/*, kitsune.study_queue)
# read this block through rules()["boxes"]; its structure is part of the wave-2 contract (CONTRACT.md section 6).
BOXES = {
    "A": {"runs": ["study-t06", "study-t03", "study-t01", "study-t005"], "probe_classes": ["kept-t03", "scratch"],
          "calibrate": ["study-t06", "study-t03", "study-t01", "study-t005"], "reference": "study-t06",
          "numbers_file": "PREREG_numbers_A.json", "extras": ["anchor"]},
    "B": {"runs": ["study-p03", "study-p01", "study-p005", "study-bridge"],
          "probe_classes": ["lost", "kept-p03", "bridge"],
          "calibrate": ["study-t06", "study-p03", "study-p01", "study-p005", "study-bridge"], "reference": "study-t06",
          "numbers_file": "PREREG_numbers_B.json", "extras": ["speed"]},
    "replicate": {"runs": ["study-t01-s1235"], "probe_classes": [], "calibrate": [], "numbers_from": "A",
                  "numbers_file": "PREREG_numbers_replicate.json", "extras": []},
}

# --------------------------------------------------------------------------------------------- calibration

CALIB_KEYS = ("t_step_s", "micro_audio_s", "data_wait_frac", "steps_measured")
CALIB_STEPS = (50, 250)  # t_i = the median step time over these steps, logging included
DATA_WAIT_MAX = 0.05  # at or above: perf.num_workers 12 and calibrate again; still at or above: the box halts
# max_steps are rounded (half up) to a multiple of this: then the T/2 branch's resume step 0.4 M, its end 0.5 M and the
# cooldown start of a run whose budget is M/2, 0.8 x M/2, are all exact integers, so the branch's WSD schedule is
# exactly a T/2 run's. study-t06 itself therefore trains round_to(9366) = 9,370 steps.
MAX_STEPS_MULTIPLE = 10

# ------------------------------------------------------------------------------------------------ init class, step 0

# STUDY.md 2.2: a pruned student is "function lost" when its step-0 CER against its teacher is >= this on the
# 60-utterance CPU gate. By the owner's decision of 2026-09-26 the rule classifies the Parakeet students only.
FUNCTION_LOST_CER = 0.90
# What the gate measured on the students as built (student_meta.json of each; rebuilt 2026-09-26 on the first run's
# recovered calibration ids). Transcribe: scripts/03_build_student.py --device cpu --step0-eval-utts 20 (20 ids per gate
# set, seed 1234): teacher-forced KL per token and the pooled greedy corpus CER against the teacher's transcripts.
# Parakeet: scripts/03c_build_ctc_student.py's gate (20 utterances <= 15 s per gate set, seed 2) on the saved (bf16)
# student: the mean per-utterance CER of its greedy CTC output against the teacher's (the correct CTC collapse), which
# is what the rule reads, the corpus CER and the frame KL.
STEP0 = {
    "study-t06": dict(kl=4.2068, cer_vs_teacher=1.3226, function="kept (owner decision)"),
    "study-t03": dict(kl=7.949, cer_vs_teacher=1.1554, function="kept (owner decision)"),
    "study-p03": dict(cer_vs_teacher=0.4133, cer_vs_teacher_corpus=0.3691, frame_kl=1.6873, function="kept"),
    "study-p01": dict(cer_vs_teacher=0.9866, cer_vs_teacher_corpus=0.9862, frame_kl=3.5032, function="lost"),
    "study-p005": dict(cer_vs_teacher=1.0, cer_vs_teacher_corpus=1.0, frame_kl=5.0075, function="lost"),
}
# the Parakeet students' gate before that rebuild (the same shapes calibrated on a later re-draw of the ids, b06deb35...,
# 2026-09-25): the same classes
STEP0_BEFORE_REBUILD = {"study-p03": 0.4383, "study-p01": 0.9829, "study-p005": 0.9997}

# ------------------------------------------------------------------------------------------------ owner changes

# The owner accepted all 30 decisions at the recommendation, then changed these on 2026-09-26 (CONTRACT.md section 6)
OWNER_CHANGES = [
    "T-0.3B = B8x2560 + decoder {0,2,5,7} (decision 17; 301,822,208 parameters, encoder layers evenly_spaced(8, 48)): "
    "B10x2560 + decoder {0,7} started at a step-0 KL of 15.7 on the gate, worse than random; B8 with the four-layer "
    "decoder at 8.1. It keeps T-0.6B's decoder, so T 0.6 -> 0.3 changes only the encoder depth. The bridge is this "
    "shape from scratch.",
    "The pruned Transcribe students (T-0.6B, T-0.3B) keep the function-kept settings (warm-up 300, probe class "
    "kept-t03) although their step-0 CER against the teacher is >= 90 %: an AED decoder derails at step 0, and the "
    "first run trained this recipe fine. The 90 % step-0 rule classifies the Parakeet students only.",
    "Every LR-probe grid has 3 points up front: kept-t03 and kept-p03 1e-4 / 2e-4 / 4e-4, bridge 2e-4 / 4e-4 / 1e-3 "
    "(scratch and lost unchanged); the edge rule is unchanged.",
    "The first run's 1,000 calibration ids (importance_ids_sha256 5e31cd68...) were recovered from the HF dataset's "
    "history, and every pruned student (T-0.3B, P-0.3B, P-0.1B, P-0.05B) was rebuilt on them; T-0.6B keeps the first "
    "run's FFN selection.",
    "The study runs on two boxes, Cohere first (BOXES): box A trains the Transcribe students, box B the Parakeet "
    "students and the bridge, the replicate a 1x box after box A; each box measures study-t06's step time on its own "
    "host and writes its own numbers file (decision 2).",
]
# decision -> what it said before the owner's change of 2026-09-26
CHANGED_DECISIONS = {2: "S3: one 4x A100 SXM4 40 GB box, calibration + probes + 2 waves; S3b, then S1p as fallbacks",
                     17: "a: T-0.3B = B10x2560 + decoder {0,7} (320,752,384 parameters)"}
# decision -> what the label checks of wave 1 measured on the real labels (tools/label_checks.py, 2026-09-26)
WAVE1_FACTS = {
    15: "K4: the stored n_frames equals the ParakeetFeatureExtractor length of the decoded audio on 400 / 400 rows, but "
        "an n_frames computed from the stored duration differs on 11.6 % of CV8 rows: the frame preflight decodes the "
        "audio, it never derives the length from a duration",
    22: "K3: the greedy CTC path of the stored columns equals the stored ctc_hyp on 35,327 / 35,327 rows; K7: 0 "
        "CTC-infeasible rows; K8: CER(ctc_hyp, TDT hyp) 3.1 %, and against the reference greedy CTC and TDT score "
        "16.13 vs 16.20 % on the labelled train rows: the greedy CTC target costs nothing in text quality",
}

# ------------------------------------------------------------------------------------------------ the selection

# the recipe block every study run config carries as selection_recipe.study (scripts/make_selection.py applies it,
# vast/launch.py checks the selection was built with it and that it is this one). probe_n is per train source, as the
# existing probe convention draws it: 300 x the five sources = 1,500 kept rows
STUDY_SELECTION = {"f1a_max": 0.5, "dedup_min_chars": 15, "draw_audio_s": 3_600_000, "probe_n": 300,
                   "neutral_max_cer": 0.5}
SELECTION_SEED = 1234  # make_selection --seed of the study selection: the draw, the probe and the greedy subsets
STUDY_SCHEMA = 1  # of the study selection's sidecar and manifest (scripts/make_selection.py writes both)
# drop reasons of a study selection, after the existing ones (truncated, not_judged, no_agree, agree>A, no_audio)
STUDY_REASONS = ("not_in_parakeet", "f1a_disagree", "eval_dup", "ctc_infeasible", "not_drawn")
SELECTION_FILE = "labels/full/selections/study_1000h.parquet"
MANIFEST_FILE = "study_manifest.json"  # next to the selection
ONE_ROOT_MAX_FRAC = 0.001  # K5: train rows in teacher_out only (not_in_parakeet) above this share refuse the launch
STUDY_SOURCES = ["reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame"]
STUDY_EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"]
# the extent inputs study/data.json registers; reazon_large's cap is the cap rule's (POOL_MIN_HOURS), from the labels
STUDY_INPUTS = {"reazon_large": None, "emilia_yodas": "300h", "emilia_nc": 8, "galgame": 3}
GATE_SETS = ("eval_jsut", "eval_cv8", "eval_reazon")
GALGAME_VIEWS = ("neutral", "all", "label_box")
# the scoring strata, named as kitsune.study_stats names them: every eval set, but Galgame as its three views
STRATA = (*[s for s in STUDY_EVAL_SETS if s != "galgame"], *(f"galgame_{v}" for v in GALGAME_VIEWS))
M4_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", "galgame_neutral")
TEACHERS = ("cohere", "parakeet-ctc", "parakeet-tdt")  # the baselines' system names (CONTRACT.md 5)
POOL_MIN_HOURS = 1010  # the reazon_large cap: the smallest N whose pool after all filters holds >= this
COHERE_CER_PREREG = {"eval_jsut": 0.0830, "eval_cv8": 0.0407, "eval_reazon": 0.0628}  # kitsune.evaluate's, 0.05 pp

# ------------------------------------------------------------------------------------------------ the analysis

# STUDY.md 4.3-4.7; the "analysis" block of the rules holds them in the form kitsune.study_stats.settings_from_prereg
# reads (the prose in metrics, noise and limit_rule says the same, from these constants)
DELTA, DELTAS = 0.10, (0.05, 0.10, 0.20)  # decision 1: the primary tolerance; all three always reported
SIGMA_PRIOR = 0.016  # sigma_run's prior: the first run's residual SD of the 4-set macro around its trend
SIGMA_FACTOR = 0.886  # sqrt(pi) / 2: one replicate pair's |ln ratio| x this is unbiased for sigma_run
BOOT_B, BOOT_SEED, Z = 10_000, 1234, 1.96
TEACHER_BARS = (1.2, 1.5)  # the practical bars against the own teacher (4.7)
ANCHOR_FLAG_REL = 0.05  # 4.1: T-0.6B worse than the anchor by more than 5 % relative, paired CI excluding 0

# ------------------------------------------------------------------------------------------------ the decisions

# decision -> (the accepted option, what it means); the owner accepted every recommendation (study/decisions.json),
# then changed decisions 2 and 17 on 2026-09-26 (CHANGED_DECISIONS holds what they said before; OWNER_CHANGES why)
DECISIONS = {
    1: ("a", "the limit: tolerance delta 10 % against the family's largest student; delta 5 and 20 % also reported"),
    2: ("AB", "two 4x A100 boxes, Cohere first: box A the Transcribe students, box B the Parakeet students and the "
              "bridge, each calibrating on its own host; the replicate on a 1x box after box A (BOXES)"),
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
    17: ("c", "T-0.3B = B8x2560 + decoder {0,2,5,7} (T-0.6B's decoder; 301,822,208 parameters)"),
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


def round_to(x: float, multiple: int = MAX_STEPS_MULTIPLE) -> int:
    """x rounded half up to a multiple of `multiple`: round_to(9366) = 9370, round_to(27_385) = 27_390."""
    return int(multiple * math.floor(x / multiple + 0.5))


def box_of(run: str) -> str:
    """The box that trains a study run (RUNS[run]["box"], one of BOXES)."""
    return RUNS[run]["box"]


def probe_box(cls: str) -> str:
    """The box that probes an LR class: the one whose probe_classes lists it."""
    return next(b for b, spec in BOXES.items() if cls in spec["probe_classes"])


def params_of(system: str) -> int:
    """Total parameters of a study run or of a teacher (TEACHER_PARAMS)."""
    return RUNS[system]["params_total"] if system in RUNS else TEACHER_PARAMS[system]


def size_ladder() -> list[dict]:
    """STUDY.md 1.3 from the registered counts: every step with its halvings h = log2(N_big / N_small) (rounded to
    3 decimals; kitsune.study_stats computes g per halving with the exact ratio of the same counts)."""
    return [dict(big=big, small=small, params=[params_of(big), params_of(small)],
                 h=round(math.log2(params_of(big) / params_of(small)), 3), changes=changes)
            for big, small, changes in LADDER]


def study_files(selection: str) -> tuple[str, str]:
    """(sidecar, manifest) of a study selection, as repo paths: <selection without .parquet>.json and
    study_manifest.json in the selection's folder (make_selection writes both; pull_plan and launch need both)."""
    stem = selection[: -len(".parquet")] if selection.endswith(".parquet") else selection
    folder = selection.rsplit("/", 1)[0] + "/" if "/" in selection else ""
    return f"{stem}.json", f"{folder}{MANIFEST_FILE}"


FILL = ("python -m kitsune.prereg --write study/ --sidecar labels/full/selections/study_1000h.json, from the selection "
        "built on the sealed labels (scripts/make_selection.py --config study/data.json)")


def registered_recipe() -> dict:
    """The run configs' selection_recipe (study/data.json carries it verbatim): the label box's judges (F0) and the
    study block."""
    return {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "emilia_nc=0.2"], "filter_eval_sets": [],
            "partial_second_opinion": [], "study": dict(STUDY_SELECTION)}


def cap_rule(pool_hours_if_capped: dict) -> int | None:
    """The reazon_large cap the rule gives: the smallest N whose pool after all filters holds >= POOL_MIN_HOURS, from
    {N (int or its str, as JSON keys are): pool hours}; None when no N listed reaches it (rebuild at a larger cap)."""
    ok = [int(n) for n, h in pool_hours_if_capped.items() if isinstance(h, (int, float)) and h >= POOL_MIN_HOURS]
    return min(ok) if ok else None


def _pending_manifest() -> dict:
    """The manifest block with every field that needs the sealed labels pending. The filled block has exactly these
    keys (check_rules compares the shapes). sets / galgame_views hold what kitsune.study_stats compares with
    study_manifest.json; train and probe are the selection's other two id lists."""
    return {"status": PENDING, "source": FILL,
            "selection_file": SELECTION_FILE, "selection_sha256": PENDING,
            "manifest_file": study_files(SELECTION_FILE)[1], "manifest_sha256": PENDING,
            "sets": {s: {"ids_sha256": PENDING, "n": PENDING} for s in STUDY_EVAL_SETS},
            "galgame_views": {v: {"ids_sha256": PENDING, "n": PENDING} for v in GALGAME_VIEWS},
            "train": {"ids_sha256": PENDING, "n": PENDING, "hours": PENDING},
            "probe": {"ids_sha256": PENDING, "n": PENDING},
            "pool_hours": PENDING}


def _pending_baselines() -> dict:
    """Per teacher (CONTRACT.md 5 names), per scoring stratum (kitsune.study_stats names) and M4: the corpus CER on
    the manifest rows, as fractions."""
    return {"status": PENDING, "source": "the sidecar's baselines: corpus CER (kitsune.evaluate.corpus_cer) on the "
                                         "manifest rows; m4 = the macro mean over " + ", ".join(M4_SETS),
            **{t: {k: PENDING for k in (*STRATA, "m4")} for t in TEACHERS}}


def rules(sidecar: dict | None = None) -> dict:
    """Everything the study fixes in advance (STUDY.md 4.8 point 1). Without `sidecar` the fields that need the sealed
    labels are "pending"; with the study selection's sidecar (make_selection.py's study_1000h.json) they are filled
    from it, and a sidecar that is not the pre-registered selection raises ValueError (sidecar_problems). Pure data,
    deterministic: the same sidecar gives the same rules."""
    r = {
        "prereg_version": RULES_VERSION,
        "study": "Kitsune size study: 7 students + bridge + replicate on the same 1,000 h at equal A100 compute",
        "design": "STUDY.md (final); CONTRACT.md for the interfaces",
        "decisions": {str(n): {"option": k, "answer": a,
                               **({"changed_2026_09_26": {"was": CHANGED_DECISIONS[n]}} if n in CHANGED_DECISIONS
                                  else {}),
                               **({"wave1_facts": WAVE1_FACTS[n]} if n in WAVE1_FACTS else {})}
                      for n, (k, a) in DECISIONS.items()},
        "owner_changes": {"date": "2026-09-26", "changes": list(OWNER_CHANGES)},
        "runs": {run: dict(spec) for run, spec in RUNS.items()},
        "size_ladder": size_ladder(),
        "boxes": json.loads(json.dumps(BOXES)),  # a deep copy: the caller may edit its rules
        "init_class": {
            "rule": f"a pruned Parakeet student is pruned_lost when its step-0 CER against the teacher (mean per "
                    f"utterance, greedy CTC) is >= {100 * FUNCTION_LOST_CER:.0f} % on the 60-utterance CPU gate, else "
                    f"pruned_kept",
            "classifies": sorted(r for r, s in RUNS.items() if s["family"] == "ctc" and s["init_class"] != "scratch"),
            "transcribe": "the pruned Transcribe students (study-t06, study-t03) are pruned_kept by the owner's "
                          "decision of 2026-09-26 (warm-up 300, probe class kept-t03) although their step-0 CER is "
                          ">= 90 %: an AED decoder derails at step 0 (runaway greedy decodes), and the first run "
                          "trained this recipe fine",
            "gate": {"transcribe": "scripts/03_build_student.py --device cpu --step0-eval-utts 20 (seed 1234): 20 ids "
                                   "per gate set eval_jsut, eval_cv8, eval_reazon; teacher-forced KL per token and the "
                                   "pooled greedy corpus CER against the teacher's transcripts",
                     "parakeet": "scripts/03c_build_ctc_student.py: 20 utterances <= 15 s from the first shard of each "
                                 "gate set (seed 2) through the teacher's CTC path and the saved (bf16) student; the "
                                 "mean per-utterance CER of the greedy CTC outputs (the rule's number), the corpus CER "
                                 "and the frame KL"},
            "step0": {run: dict(v) for run, v in STEP0.items()},
            "step0_before_rebuild": dict(STEP0_BEFORE_REBUILD, what="cer_vs_teacher of the Parakeet students built "
                                                                    "on the re-drawn calibration ids b06deb35... "
                                                                    "(2026-09-25), before the rebuild on the first "
                                                                    "run's ids"),
            "threshold": FUNCTION_LOST_CER,
        },
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
                                "weight_decay": 0.0, "bn": "teacher stats, eval mode (affine trains)",
                                "definition": "a pruned Parakeet student whose step-0 CER against its teacher is < 90 % "
                                              "on the 60-utterance CPU gate; the pruned Transcribe students by the "
                                              "owner's decision (init_class)"},
                "pruned_lost": {"runs": ["study-p01", "study-p005"], "warmup_steps": 1000, "weight_decay": 0.0,
                                "bn": "teacher stats, eval mode (affine trains)",
                                "definition": "a pruned Parakeet student whose step-0 CER against its teacher is >= "
                                              "90 % on the 60-utterance CPU gate (init_class)"},
                "scratch": {"runs": ["study-t01", REPLICATE, "study-t005", "study-bridge"], "warmup_steps": 2000,
                            "weight_decay": 1e-3, "weight_decay_on": "parameters with ndim >= 2 only",
                            "bn": "train mode; evals use the running stats; grad_ckpt off"}},
            "bn_fallback": DECISIONS[16][1],
        },
        "lr_probes": {
            "classes": {c: dict(p, runs=sorted(run for run, s in RUNS.items() if s["lr_from"] == c), box=probe_box(c))
                        for c, p in PROBES.items()},
            "grids": "3 points up front for every class (owner, 2026-09-26)",
            "objective": "the family's training objective per target token, teacher-forced, at the probe's end, "
                         "pooled over the complete gate sets eval_jsut, eval_cv8, eval_reazon (AED: w_kl KL + w_ce CE "
                         "without aux CTC; CTC: w_kl KL + w_ctc CTC); lowest wins; a non-finite objective loses; a "
                         "tie goes to the lower LR",
            "edge_rule": "a winner at an edge of the tested grid gets one more point (x2 above, /2 below); a winner "
                         "at an edge after that extension halts the box (an invalidation rule)",
            "run_names": "probe-<class>-<lr>, lr like 1e-3 / 4e-4",
        },
        "calibration": {
            "reference_run": T_REF_RUN, "reference_steps": REF_STEPS, "max_steps_multiple": MAX_STEPS_MULTIPLE,
            "max_steps": f"max_steps_i = round_to_{MAX_STEPS_MULTIPLE}({REF_STEPS} x t_{T_REF_RUN} / t_i) = "
                         f"{MAX_STEPS_MULTIPLE} x floor({REF_STEPS} x t_{T_REF_RUN} / t_i / {MAX_STEPS_MULTIPLE} + "
                         f"1/2), both step times measured on the host that trains run i ({T_REF_RUN} itself: "
                         f"{round_to(REF_STEPS):,})",
            "t_i": f"the median step time over steps {CALIB_STEPS[0]}-{CALIB_STEPS[1]} at the planned micro_audio_s, "
                   f"logging included, on the box's host with the box's calibrated runs running concurrently",
            "per_box": f"each box calibrates its 'calibrate' list (BOXES) and measures {T_REF_RUN} on its own host: "
                       f"box B calibrates {T_REF_RUN} for {CALIB_STEPS[1]} steps without training it",
            "min_steps_measured": CALIB_STEPS[1] - CALIB_STEPS[0], "data_wait_max": DATA_WAIT_MAX,
            "loader_bound": f"data_wait_frac >= {DATA_WAIT_MAX}: perf.num_workers 12 and calibrate again; still "
                            f">= {DATA_WAIT_MAX}: the box halts",
            "excluded": "evals, checkpoints and the T/2 branch do not count toward T",
            "replicate": f"{REPLICATE} is not calibrated: it takes {REPLICATE_OF}'s max_steps and the scratch LR from "
                         f"box A's numbers file",
        },
        "branch": {"resume_frac": 0.4, "end_frac": 0.5, "t_c": "the resume step",
                   "what": "<run>-half resumes the local full state at 0.4 x max_steps and cools down to 0.5 x "
                           "max_steps: the WSD schedule of a run whose budget is T/2 (same warm-up, stable LR and "
                           "data order, 20 % cooldown); one complete eval at its end",
                   "exact": f"max_steps is a multiple of {MAX_STEPS_MULTIPLE}, so 0.4 x max_steps, 0.5 x max_steps and "
                            f"a T/2 run's cooldown start 0.8 x (max_steps / 2) are exact integers"},
        "evals": {"complete_at_fracs": [0.2, 0.4, 0.6, 0.8], "final": "after the cooldown, at max_steps",
                  "mini_every_frac": 0.025, "checkpoint": "the final one, never the best eval; plus the branch's",
                  "reported": "per set, group and pooled: corpus CER, ratio to the own teacher, imitation CER, "
                              "no-style CER, S/D/I, runaway rate (CER > 100 % or hyp > 2 x ref), empty-output rate, "
                              "AED truncation; teacher-forced KL/CE (AED) or dense/blank KL, CTC per token, argmax "
                              "agreement and argmax-blank share (CTC); probe vs held-out KL; step-seconds, steps, "
                              "epochs, audio-hours; params total and non-embedding; speed and VRAM"},
        "data": {
            "extent": {"name": "full", "root": "labels/full",
                       "inputs": {k: PENDING if v is None else v for k, v in STUDY_INPUTS.items()}},
            "reazon_large_cap": f"the smallest reazon_large input count whose pool after all filters holds >= "
                                f"{POOL_MIN_HOURS} h (expected 50-55), read from the sealed labels (the sidecar's "
                                f"details.pool_hours_if_capped; the fill refuses any other count)",
            "sources": list(STUDY_SOURCES), "eval_sets": list(STUDY_EVAL_SETS),
        },
        "selection": {
            "file": f"{SELECTION_FILE} (+ {', '.join(study_files(SELECTION_FILE))})",
            "recipe": registered_recipe(),
            "seed": SELECTION_SEED,
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
            "frame_preflight": "at store build, for every row: the ParakeetFeatureExtractor length of the rebuilt, "
                               "decoded audio, 8x subsampled, equals the stored n_frames (never a length derived from "
                               "the stored duration: K4, decision 15); mismatches are dropped and counted, the box "
                               "fails above 0.1 % of train rows or on any eval row (decision 15)",
        },
        "manifest": _pending_manifest(),
        "metrics": {
            "m4": "the macro mean of complete-set corpus CER over eval_jsut, eval_cv8, eval_reazon and Galgame-neutral,"
                  " equal weights (within-family primary)",
            "strata": f"every eval set is one stratum, Galgame is its three views instead: {', '.join(STRATA)}",
            "qualifiers": {"out_of_domain": ["eval_jsut", "eval_cv8"], "in_domain": ["eval_reazon", "galgame_neutral"]},
            "cross_family": "JSUT + Galgame-neutral, raw and no-style, paired CIs; descriptive only, no threshold",
            "galgame_views": {"neutral": "primary: the rows whose kotoba-whisper-v2.0 hypothesis (the laptop's "
                                         "second_out/galgame/eval-00000.jsonl) has cer(hyp2, ref) <= 0.5 and a "
                                         "non-empty reference",
                              "all": "every Galgame hold-out row of the manifest",
                              "label_box": "the label box's filter: not truncated and agree(Cohere, Parakeet) <= 0.5"},
            "cer": "kitsune.evaluate.corpus_cer: sum(S+D+I) / sum(ref chars) on normalize_ja text, empty refs "
                   "skipped",
        },
        "noise": {"sigma_prior": SIGMA_PRIOR,
                  "replicate_estimate": f"{SIGMA_FACTOR} x |ln(M4_{REPLICATE} / M4_{REPLICATE_OF})|",
                  "sigma_rule": f"sigma_run = max({SIGMA_PRIOR}, the replicate's estimate)",
                  "ci_student_ratio": f"ln r +- {Z} sqrt(v_boot + 2 sigma_run^2)",
                  "ci_student_vs_teacher": f"ln r +- {Z} sqrt(v_boot + sigma_run^2)",
                  "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "kind": "paired utterance bootstrap, stratified: "
                                                                       "resampled within each stratum, the same "
                                                                       "indices for every system"}},
        "analysis": {  # the machine-read form (kitsune.study_stats.settings_from_prereg); the prose blocks say the same
            "delta": DELTA, "deltas": list(DELTAS),
            "sigma_run": {"prior": SIGMA_PRIOR, "factor": SIGMA_FACTOR, "replicate": [REPLICATE, REPLICATE_OF]},
            "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "z": Z},
            "teacher_bars": list(TEACHER_BARS), "anchor_flag_rel": ANCHOR_FLAG_REL,
            "strata": list(STRATA), "m4_strata": list(M4_SETS), "teachers": list(TEACHERS),
        },
        "limit_rule": {
            "families": {"aed": {"top": "study-t06", "walk": ["study-t03", "study-t01", "study-t005"]},
                         "ctc": {"top": "study-p03", "walk": ["study-p01", "study-p005"]}},
            "ratio": "r_s = M4_s / M4_top",
            "delta": DELTA, "delta_reported": list(DELTAS),
            "calls": {"WITHIN": "the CI's upper end <= ln(1 + delta)", "OUTSIDE": "the CI's lower end > ln(1 + delta)",
                      "UNRESOLVED": "otherwise"},
            "walk": "down the sizes: the limit is the smallest size WITHIN with every larger size WITHIN; the first "
                    "OUTSIDE ends it (limit between X and Y); an UNRESOLVED gives 'at or below X, Y unresolved' and "
                    "the smaller sizes are descriptive; one decisive claim per family, no Holm correction",
            "replicate_role": f"{REPLICATE} feeds sigma_run only; the walk uses {REPLICATE_OF}",
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
                           f"the smallest student within {TEACHER_BARS[0]}x and within {TEACHER_BARS[1]}x of its own "
                           f"teacher on M4",
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
                  "any rule field still pending",
                  "the replicate box without box A's numbers file, or with one written under other rules"],
        "invalid_if": ["a run misses its max_steps, or resumes with a changed config",
                       "a selection or manifest hash differs between runs (on one box or across boxes)",
                       "the teacher baselines do not reproduce",
                       "K3 or K4 fails, or the frame preflight exceeds its threshold",
                       "this rules commit comes after any study result, or a box's numbers file after that box's first "
                       "study step",
                       "the anchor regression flag fires and stays unexplained",
                       "an LR winner sits at a grid edge after its one extension",
                       "calibration stays loader-bound after raising the workers"],
        "budget": {"cut_order": ["the replicate", "T at 3 epochs", "the contingency"],
                   "follow_ups": ["a d-pruned P-0.1B if P-0.1B is OUTSIDE",
                                  "a 2T extension from the pre-cooldown state if T/2 -> T says compute-limited",
                                  "a second replicate if a decisive call falls within about 1 sigma of delta"],
                   "follow_up_rule": "conditional on the trigger, and only if the contingency is unused"},
        "numbers": {"files": {b: spec["numbers_file"] for b, spec in BOXES.items()},
                    "keys": list(NUMBERS_KEYS),
                    "when": "written and uploaded by each box, its sha256 logged as an event, before that box's first "
                            "study step (write_numbers(..., box=<A|B|replicate>))",
                    "per_box": "a box's file holds its own calibration table, the max_steps and LRs of the runs it "
                               "trains and the probes it ran; the replicate's holds no calibration and no probes, and "
                               "names box A's file (numbers_from: its sha256) whose study-t01 max_steps and LR it "
                               "takes"},
    }
    if sidecar is not None:
        _fill(r, sidecar)
    return r


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _is_count(v, lo: int = 1) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


def _cer(v):
    """A baseline entry of the sidecar ({"cer": x, "edits": ...}, or a bare number) as its CER."""
    return v.get("cer") if isinstance(v, dict) else v


def sidecar_problems(sc: dict) -> list[str]:
    """Why a sidecar is not the pre-registered study selection, so that its hashes must not fill the PREREG: the
    selection must have been built from the registered sources, eval sets, recipe (F0 thresholds and the study block),
    seed and extent (every input as registered; reazon_large at the cap the rule gives, the smallest N whose pool
    holds >= POOL_MIN_HOURS h), under the registered file names; and it must hold every field the rules fill: a
    non-empty id list with its sha256 for the train set, the probe (probe_n per train source), every eval set and
    every Galgame view, and a finite baseline per teacher, stratum and M4."""
    if not isinstance(sc, dict):
        return [f"a sidecar is a JSON object, not {type(sc).__name__}"]
    p = []
    want = {"schema": STUDY_SCHEMA, "sources": STUDY_SOURCES, "eval_sets": STUDY_EVAL_SETS,
            "recipe": registered_recipe(), "seed": SELECTION_SEED}
    p += [f"{k} {sc.get(k)!r}, registered {v!r}" for k, v in want.items() if sc.get(k) != v]
    sel_file, man_file = (sc.get("selection") or {}), (sc.get("manifest") or {})
    for what, block, path in (("selection", sel_file, SELECTION_FILE),
                              ("manifest", man_file, study_files(SELECTION_FILE)[1])):
        if block.get("path") != path or not _is_sha(block.get("sha256")):
            p.append(f"{what} {block.get('path')!r} (sha256 {str(block.get('sha256'))[:12]!r}): registered {path} "
                     f"with its sha256")
    ext = sc.get("extent") or {}
    inputs = ext.get("inputs") or {}
    fixed = {k: v for k, v in STUDY_INPUTS.items() if v is not None}
    if ext.get("name") != "full" or {k: inputs.get(k) for k in inputs if k != "reazon_large"} != fixed:
        p.append(f"extent {ext.get('name')!r} inputs {inputs!r}: registered 'full' with {fixed} and reazon_large")
    cap = inputs.get("reazon_large")
    caps = ((sc.get("details") or {}).get("pool_hours_if_capped") or {}).get("reazon_large") or {}
    rule = cap_rule(caps)
    if not _is_count(cap):
        p.append(f"extent reazon_large {cap!r} is not an input count")
    elif rule is None:
        best = max((h for h in caps.values() if isinstance(h, (int, float))), default=None)
        p.append(f"no reazon_large cap up to {cap} leaves {POOL_MIN_HOURS} h after all filters (the most: "
                 f"{best if best is None else round(best, 1)} h): rebuild the selection at a larger cap")
    elif cap != rule:
        p.append(f"extent reazon_large {cap}: the cap rule gives {rule} (the smallest N whose pool holds >= "
                 f"{POOL_MIN_HOURS} h); rebuild the selection with reazon_large {rule}")
    draw = sc.get("draw") or {}
    if caps and _is_count(cap) and isinstance(caps.get(str(cap)), (int, float)) and isinstance(
            draw.get("pool_s"), (int, float)) and abs(caps[str(cap)] - draw["pool_s"] / 3600) > 1e-6 * max(
            1.0, caps[str(cap)]):
        p.append(f"pool_hours_if_capped at reazon_large {cap} ({caps[str(cap)]:.3f} h) differs from the draw's pool "
                 f"({draw['pool_s'] / 3600:.3f} h)")
    if draw.get("budget_s") != STUDY_SELECTION["draw_audio_s"] or not (
            isinstance(draw.get("drawn_s"), (int, float)) and 0 < draw["drawn_s"] <= STUDY_SELECTION["draw_audio_s"]
            and isinstance(draw.get("pool_s"), (int, float)) and draw["pool_s"] > STUDY_SELECTION["draw_audio_s"]):
        p.append(f"draw {draw!r}: registered a budget of {STUDY_SELECTION['draw_audio_s']} s from a larger pool")
    ids, n = sc.get("ids_sha256") or {}, sc.get("n") or {}
    probe_n = STUDY_SELECTION["probe_n"] * len(STUDY_SOURCES)
    for what, h, k, lo in (("train", ids.get("train"), n.get("train"), 1),
                           ("probe", ids.get("probe"), n.get("probe"), probe_n),
                           *((s, (ids.get("eval") or {}).get(s), (n.get("eval") or {}).get(s), 1)
                             for s in STUDY_EVAL_SETS)):
        if not _is_sha(h) or not _is_count(k, lo) or (what == "probe" and k != probe_n):
            p.append(f"{what}: {k!r} ids (sha256 {str(h)[:12]!r}); registered "
                     + (f"{probe_n} ids" if what == "probe" else "a non-empty id list") + " with its sha256")
    views = sc.get("galgame_views") or {}
    for v in GALGAME_VIEWS:
        b = views.get(v) or {}
        if not _is_sha(b.get("ids_sha256")) or not _is_count(b.get("n")):
            p.append(f"Galgame view {v}: {b!r}; registered a non-empty id list with its sha256")
    base = sc.get("baselines") or {}
    for t in TEACHERS:
        bad = [k for k in (*STRATA, "m4") if not (isinstance(_cer((base.get(t) or {}).get(k)), (int, float))
                                                  and math.isfinite(_cer(base[t][k])) and _cer(base[t][k]) >= 0)]
        if bad:
            p.append(f"baselines {t}: no finite corpus CER for {bad}")
    return p


def _fill(r: dict, sc: dict):
    """The pending fields from the study selection's sidecar (scripts/make_selection.py): the file and id hashes of
    the selection, its manifest and views, the train and pool hours, the reazon_large cap and the teacher baselines
    (corpus CER) on the manifest rows. Raises ValueError unless the sidecar is the pre-registered selection
    (sidecar_problems): the pending mechanism is what makes the registration complete, so it only takes the one
    selection the rules describe."""
    if problems := sidecar_problems(sc):
        raise ValueError(f"the sidecar is not the pre-registered study selection ({len(problems)} problem(s)): "
                         + "; ".join(problems))
    ids, n, views, draw = sc["ids_sha256"], sc["n"], sc["galgame_views"], sc["draw"]
    r["manifest"] = dict(
        _pending_manifest(), status="filled",
        source="the study selection's sidecar (scripts/make_selection.py --config study/data.json, sealed labels)",
        selection_sha256=sc["selection"]["sha256"], manifest_sha256=sc["manifest"]["sha256"],
        sets={s: {"ids_sha256": ids["eval"][s], "n": n["eval"][s]} for s in STUDY_EVAL_SETS},
        galgame_views={v: {"ids_sha256": views[v]["ids_sha256"], "n": views[v]["n"]} for v in GALGAME_VIEWS},
        train={"ids_sha256": ids["train"], "n": n["train"], "hours": draw["drawn_s"] / 3600},
        probe={"ids_sha256": ids["probe"], "n": n["probe"]},
        pool_hours=draw["pool_s"] / 3600)
    r["data"]["extent"]["inputs"]["reazon_large"] = sc["extent"]["inputs"]["reazon_large"]
    b = sc["baselines"]
    r["baselines"] = dict(_pending_baselines(), status="filled",
                          **{t: {k: float(_cer(b[t][k])) for k in (*STRATA, "m4")} for t in TEACHERS})


def _shape(obj):
    """The key structure of a JSON value (dicts by key, lists by length), without the values."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    return [_shape(v) for v in obj] if isinstance(obj, list) else None


def filled_problems(committed: dict) -> list[str]:
    """Why the filled parts of a committed PREREG.json are not a fill of these rules: the manifest and baselines blocks
    must have exactly the keys of their pending templates (no eval set, view, teacher or stratum lost or added, even
    as a hand edit), and data.extent.inputs must be the registered inputs with reazon_large an input count (or
    pending)."""
    p = []
    for key, template in (("manifest", _pending_manifest()), ("baselines", _pending_baselines())):
        if _shape(committed.get(key)) != _shape(template):
            p.append(f"{key}: its keys differ from the rules' ({key} block of kitsune/prereg.py)")
    inputs = ((committed.get("data") or {}).get("extent") or {}).get("inputs")
    cap = inputs.get("reazon_large") if isinstance(inputs, dict) else None
    if not isinstance(inputs, dict) or set(inputs) != set(STUDY_INPUTS) or \
            any(inputs[k] != v for k, v in STUDY_INPUTS.items() if v is not None) or \
            not (cap == PENDING or _is_count(cap)):
        p.append(f"data.extent.inputs {inputs!r}: registered {STUDY_INPUTS} with reazon_large pending or a count")
    return p


def regenerate(committed: dict) -> dict:
    """rules() with the filled parts taken from a committed PREREG.json (the manifest and baselines blocks and the
    reazon_large cap): everything else must equal the code, so a committed file whose rule part was edited by hand
    (or is stale) differs from this."""
    r = rules()
    r["manifest"], r["baselines"] = committed.get("manifest"), committed.get("baselines")
    inputs = ((committed.get("data") or {}).get("extent") or {}).get("inputs")
    if isinstance(inputs, dict) and "reazon_large" in inputs:
        r["data"]["extent"]["inputs"]["reazon_large"] = inputs["reazon_large"]
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
        return ", ".join(_fmt(x) for x in v) if v else "[]"
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
             f"binding form (its sha256 goes into every box's PREREG_numbers_<box>.json as `rules_sha256`). Rules "
             f"version {r['prereg_version']}.", "",
             (f"**{len(left)} field(s) still pending** (they need the sealed labels; the final pre-launch commit "
              f"fills them with `--sidecar`): " + ", ".join(f"`{p}`" for p in left[:12]) +
              (" ..." if len(left) > 12 else "") if left else "**Complete**: no field is pending."), ""]
    lines += ["## Runs", "", "| run | family | init | params total | non-embedding | LR from | warm-up | wd | aux CTC "
                             "| micro | box |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, s in r["runs"].items():
        lines.append(f"| {run} | {s['family']} | {s['init_class']} | {s['params_total']:,} | "
                     f"{s['params_non_embedding']:,} | {s['lr_from']} | {s['warmup_steps']} | {_fmt(s['weight_decay'])}"
                     f" | {_fmt(s['aux_ctc'])} | {s['micro_audio_s']} | {s['box']} |")
    lines += ["", "Shapes:", ""] + [f"- **{run}**: {s['shape']}; seed {s['seed']}; BN {s['bn']}"
                                    for run, s in r["runs"].items()]
    lines += ["", "## Size ladder (STUDY.md 1.3)", "", "| step | params | halvings h | what changes besides size |",
              "|---|---|---|---|"]
    lines += [f"| {s['big']} -> {s['small']} | {s['params'][0]:,} -> {s['params'][1]:,} | {s['h']:.3f} | "
              f"{s['changes']} |" for s in r["size_ladder"]]
    lines += ["", "## LR probes", "", "| class | box | probed on | grid | steps | warm-up / stable / cooldown | runs |",
              "|---|---|---|---|---|---|---|"]
    for c, p in r["lr_probes"]["classes"].items():
        lines.append(f"| {c} | {p['box']} | {p['probed_on']} | {', '.join(lr_tag(x) for x in p['grid'])} | "
                     f"{p['max_steps']:,} | {p['warmup']} / {p['stable']} / {p['cooldown']} | {', '.join(p['runs'])} |")
    lines += ["", f"- **objective**: {r['lr_probes']['objective']}", f"- **edge rule**: {r['lr_probes']['edge_rule']}",
              f"- **grids**: {r['lr_probes']['grids']}", ""]
    titles = [("owner_changes", "The owner's changes of 2026-09-26"),
              ("decisions", "Owner decisions (all at the recommendation; 2 and 17 changed on 2026-09-26)"),
              ("boxes", "Boxes (who trains, probes and calibrates what)"),
              ("init_class", "Init class and the step-0 gate"),
              ("training", "Training"), ("calibration", "Calibration and max_steps"), ("branch", "The T/2 branch"),
              ("evals", "Evals"), ("data", "Data"), ("selection", "Selection"), ("manifest", "Manifest and hashes"),
              ("metrics", "Metrics"), ("noise", "Noise model"),
              ("analysis", "Analysis settings (the form kitsune.study_stats reads)"),
              ("limit_rule", "The limit rule (primary answer)"),
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


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _box(box: str | None) -> dict | None:
    if box is not None and box not in BOXES:
        raise ValueError(f"box {box!r}: not one of {sorted(BOXES)}")
    return BOXES[box] if box is not None else None


def calibration_problems(calib: dict, box: str | None = None) -> list[str]:
    """Why a calibration table cannot give max_steps: a run of the box's 'calibrate' list missing (box None: the whole
    study on one host, every run but the replicate), a run the box does not calibrate (another box's, or on the
    replicate box any), the replicate present (it takes study-t01's max_steps from box A's numbers: its own step time
    would give it another budget, and then it no longer measures the run-to-run noise of T-0.1B), a key missing, a step
    time or micro_audio_s that is not a positive number, fewer steps than the window, a data_wait_frac that is not a
    fraction, or a loader-bound run (data_wait_frac >= DATA_WAIT_MAX: raise perf.num_workers to 12 and calibrate
    again; if it stays, the box halts)."""
    spec = _box(box)
    want = [run for run in RUNS if run != REPLICATE] if spec is None else list(spec["calibrate"])
    problems = [f"{run}: not calibrated" for run in want if run not in calib]
    for run, c in calib.items():
        if run == REPLICATE:
            problems.append(f"{run}: not calibrated by the rules (it takes {REPLICATE_OF}'s max_steps from box A's "
                            f"numbers)")
            continue
        if run not in RUNS:
            problems.append(f"{run}: not a study run")
            continue
        if run not in want:
            problems.append(f"{run}: not calibrated on box {box} (it trains on box {box_of(run)})")
            continue
        if missing := [k for k in CALIB_KEYS if k not in c]:
            problems.append(f"{run}: missing {missing}")
            continue
        for k in ("t_step_s", "micro_audio_s"):
            if not (_number(c[k]) and c[k] > 0):
                problems.append(f"{run}: {k} {c[k]!r} is not a positive number")
        if not (_is_count(c["steps_measured"], 0) and c["steps_measured"] >= CALIB_STEPS[1] - CALIB_STEPS[0]):
            problems.append(f"{run}: {c['steps_measured']!r} steps measured, the window is "
                            f"{CALIB_STEPS[1] - CALIB_STEPS[0]}")
        w = c["data_wait_frac"]
        if not (_number(w) and 0 <= w <= 1):
            problems.append(f"{run}: data_wait_frac {w!r} is not a fraction")
        elif w >= DATA_WAIT_MAX:
            problems.append(f"{run}: loader-bound (data_wait_frac {w:.3f} >= {DATA_WAIT_MAX})")
    return problems


def max_steps(calib: dict, t_ref_run: str = T_REF_RUN, ref_steps: int = REF_STEPS,
              box: str | None = None) -> dict[str, int]:
    """run -> max_steps = round_to(ref_steps x t_ref / t_run): every run gets the reference run's step-time budget T
    (STUDY.md 5.1), rounded half up to a multiple of MAX_STEPS_MULTIPLE (so the T/2 branch is exactly a T/2 run's
    schedule). `calib` is {run: {"t_step_s": ...}} as measured on ONE host: with `box`, that box's host, and the result
    holds the box's runs (box B's reference study-t06 is calibrated there but not trained, so it is not in the result);
    without, every calibrated run of the whole study on one host. The replicate always takes the max_steps of the run
    it replicates: here when that run is in the table (its own entry, if any, is ignored here and refused by
    calibration_problems), on the replicate box through replicate_numbers (box A's file); a replicate with other steps
    would not measure run-to-run noise."""
    spec = _box(box)
    if box == "replicate":
        raise ValueError("the replicate box is not calibrated: it takes study-t01's max_steps from box A's numbers "
                         "(replicate_numbers)")
    if t_ref_run not in calib:
        raise ValueError(f"the reference run {t_ref_run} is not calibrated")
    t_ref = float(calib[t_ref_run]["t_step_s"])
    runs = [run for run in calib if run != REPLICATE] if spec is None else list(spec["runs"])
    out = {}
    for run in runs:
        if run not in calib:
            raise ValueError(f"{run}: not calibrated on box {box}")
        t = float(calib[run]["t_step_s"])
        if not (math.isfinite(t) and t > 0 and math.isfinite(t_ref) and t_ref > 0):
            raise ValueError(f"{run}: step time {t} (reference {t_ref}) is not a positive number")
        out[run] = round_to(ref_steps * t_ref / t)
    if spec is None and REPLICATE_OF in out:
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


def run_lrs(choices: dict, box: str | None = None) -> dict[str, float]:
    """run -> the peak LR its probe class chose (T-0.6B takes T-0.3B's, T-0.05B and the replicate T-0.1B's, P-0.05B
    P-0.1B's), for the runs `box` trains (every run without a box). Raises ValueError unless every class a run needs
    is "chosen"; the replicate box probes nothing (replicate_numbers takes the scratch LR from box A's file)."""
    spec = _box(box)
    if box == "replicate":
        raise ValueError("the replicate box runs no probes: it takes the scratch LR from box A's numbers "
                         "(replicate_numbers)")
    out = {}
    for run in (RUNS if spec is None else spec["runs"]):
        cls = RUNS[run]["lr_from"]
        c = choices.get(cls)
        if c is None or c.get("decision") != "chosen":
            raise ValueError(f"{run}: its probe class {cls} has no chosen LR "
                             f"({None if c is None else c.get('decision')})")
        out[run] = float(c["lr"])
    return out


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replicate_numbers(numbers_a) -> tuple[dict[str, int], dict[str, float], dict]:
    """(max_steps, lr, numbers_from) of the replicate box from box A's numbers file (a path): study-t01's max_steps and
    LR (the scratch class's winner on box A) for study-t01-s1235, and where they came from ({box, file, sha256}).
    Raises ValueError unless the file is box A's and holds both."""
    path = Path(numbers_a)
    a = json.loads(path.read_text(encoding="utf-8"))
    if a.get("box") != BOXES["replicate"]["numbers_from"]:
        raise ValueError(f"{path}: box {a.get('box')!r}, the replicate takes its numbers from box "
                         f"{BOXES['replicate']['numbers_from']}")
    steps, lr = (a.get("max_steps") or {}).get(REPLICATE_OF), (a.get("lr") or {}).get(REPLICATE_OF)
    if not (_is_count(steps) and _number(lr) and lr > 0):
        raise ValueError(f"{path}: no max_steps / lr for {REPLICATE_OF} ({steps!r}, {lr!r})")
    return ({REPLICATE: int(steps)}, {REPLICATE: float(lr)},
            {"box": a["box"], "file": path.name, "sha256": file_sha256(path)})


def write_numbers(path, calibration: dict, probes: dict, lrs: dict, max_steps: dict, *, box: str | None = None,
                  numbers_from=None, rules_path=None, host: str | None = None, allow_pending: bool = False) -> str:
    """A numbers file (atomic); returns its sha256. With `box` (A, B or replicate; the study's boxes) the file must be
    named BOXES[box]["numbers_file"] and holds that box's numbers; without, PREREG_numbers.json of the whole study on
    one host. Keys (NUMBERS_KEYS): box, calibration {run: {t_step_s, micro_audio_s, data_wait_frac, steps_measured}},
    max_steps {run: int}, lr_probes {class: {lr tag: objective}}, lr {run: float}, rules_sha256 (rules_sha256 of the
    committed study/PREREG.json, `rules_path`), written_utc, host (default the machine's node name); the replicate's
    adds numbers_from {box, file, sha256} of box A's file.

    Mechanical by construction: it raises ValueError if the calibration has problems (calibration_problems(box)), if
    `max_steps` is not what max_steps(box=box) derives from it, if the probes are not the box's probe classes, if `lrs`
    is not run_lrs(choose_lr(probes), box) (so every class must be "chosen"), or if the rules still have a pending
    field (allow_pending=True only for dry runs). The replicate box takes no calibration and no probes: `numbers_from`
    is box A's numbers file, written under the same rules, and max_steps / lrs must be its study-t01 entries for
    study-t01-s1235 (replicate_numbers)."""
    max_steps_given, max_steps = max_steps, _derive_max_steps  # the parameter keeps the contract's name
    spec = _box(box)
    if spec is not None and Path(path).name != spec["numbers_file"]:
        raise ValueError(f"box {box} writes {spec['numbers_file']}, not {Path(path).name}")
    rules_path = Path(rules_path) if rules_path is not None else REPO / "study" / RULES_JSON
    if not allow_pending and (left := pending(json.loads(rules_path.read_text(encoding="utf-8")))):
        raise ValueError(f"{rules_path} still has {len(left)} pending field(s), e.g. {left[:3]}: commit the filled "
                         f"rules (python -m kitsune.prereg --write study/ --sidecar ...) before the study box starts")
    rules_sha = rules_sha256(rules_path)
    extra = {}
    if box == "replicate":
        if calibration or probes:
            raise ValueError("the replicate box takes no calibration and no probes (box A's numbers give them)")
        if numbers_from is None:
            raise ValueError("the replicate box needs numbers_from: box A's numbers file")
        want_steps, want_lr, extra["numbers_from"] = replicate_numbers(numbers_from)
        a_rules = json.loads(Path(numbers_from).read_text(encoding="utf-8")).get("rules_sha256")
        if a_rules != rules_sha:
            raise ValueError(f"box A's numbers were written under rules {str(a_rules)[:12]}, these are {rules_sha[:12]}")
    else:
        if numbers_from is not None:
            raise ValueError(f"numbers_from is the replicate box's input, not box {box}'s")
        if problems := calibration_problems(calibration, box):
            raise ValueError("calibration: " + "; ".join(problems))
        want_steps = max_steps(calibration, box=box)
        if spec is not None and set(probes) != set(spec["probe_classes"]):
            raise ValueError(f"box {box} probes {sorted(spec['probe_classes'])}, not {sorted(probes)}")
        want_lr = run_lrs(choose_lr(probes), box)
    if {k: int(v) for k, v in max_steps_given.items()} != want_steps:
        raise ValueError(f"max_steps {max_steps_given} is not the calibrated {want_steps}")
    if {k: float(v) for k, v in lrs.items()} != want_lr:
        raise ValueError(f"lr {lrs} is not the probes' choice {want_lr}")
    numbers = {
        "box": box,
        "calibration": {run: {k: calibration[run][k] for k in CALIB_KEYS} for run in sorted(calibration)},
        "max_steps": dict(sorted(want_steps.items())),
        # a diverged probe (a non-finite objective, which loses) is null: the file is strict JSON for every reader
        "lr_probes": {cls: {lr_tag(lr): float(res[lr]) if _number(res[lr]) else None for lr in sorted(res, key=_key)}
                      for cls, res in sorted(probes.items())},
        "lr": dict(sorted(want_lr.items())),
        "rules_sha256": rules_sha,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host if host is not None else platform.node(),
        **extra,
    }
    data = (json.dumps(numbers, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
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
    try:
        if args.write:
            sha = write_rules(args.write, sidecar)
            left = pending(rules(sidecar))
            print(f"wrote {Path(args.write) / RULES_JSON} (sha256 {sha}) and {RULES_MD}; {len(left)} pending field(s)")
            return 0
        d = Path(args.check)
        ok, r = check_rules(d, sidecar)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    left = pending(r) if r is not None else ["(no PREREG.json)"]
    problems = filled_problems(json.loads((d / RULES_JSON).read_text(encoding="utf-8"))) if r is not None else []
    print(f"{d}: {'up to date' if ok else 'DIFFERS from kitsune/prereg.py'}; {len(left)} pending field(s)"
          + (": " + ", ".join(left) if left else "") + "".join(f"\n  {p}" for p in problems))
    return 0 if ok else 1


def check_rules(d, sidecar: dict | None = None) -> tuple[bool, dict | None]:
    """(the committed PREREG.json and PREREG.md in `d` are what the code writes, the rules). Without `sidecar` the
    filled parts are taken from the committed file itself (regenerate), and must have the shape of a fill
    (filled_problems); with it they must be the ones this sidecar fills (which refuses a sidecar that is not the
    pre-registered selection)."""
    d = Path(d)
    if not (d / RULES_JSON).is_file():
        return False, None
    committed = json.loads((d / RULES_JSON).read_text(encoding="utf-8"))
    r = rules(sidecar) if sidecar is not None else regenerate(committed)
    # compared as parsed JSON and as text with universal newlines: a CRLF checkout is the same file
    ok = committed == json.loads(rules_json(r)) and not filled_problems(committed) and (d / RULES_MD).is_file() and \
        (d / RULES_MD).read_text(encoding="utf-8") == rules_md(r)
    return ok, r


if __name__ == "__main__":
    sys.exit(main())
