"""Generate the full-data runs' configs, configs/full/<name>.json, from the size study's generator (build contract 7).

Trainer configs (the four full runs and their smokes). Every one is the study run's config as
tools/make_study_configs.py makes it (run_config(study_run): the trainer part of configs/next_run_template.json,
study/data.json, the study's settings; the queue fills max_steps, lr and the mini cadence left null), then:
  1. the data keys (kitsune.fullrun DATA_KEYS) replaced whole by the run's data block: kitsune.fullrun FULL_DATA (the
     full selection labels/full/selections/full_study/full.parquet with its dev slice, the full extent, the full_study
     recipe) or SMOKE_DATA (the smoke selection on the study extent: a seeded 100 h train draw with its own dev slice,
     pull_parakeet true: both label roots). No full run pulls both roots (pull_parakeet left out): each box pulls one
     family's labels (kitsune.extent.pull_plan): box p01 and box full-p (data-p01, data-p: ctc) parakeet_out for every
     stem and teacher_out for the eval stems only (fix 9), box full-t (data-t: aed) teacher_out for every stem;
  2. the student's row of FULL_RUNS: the epochs clock (schedule.epochs: FULL_RUNS, no max_steps), the study's warm-up,
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
     0.8, SpecAugment, the smoke block, seed 1234, verdict v2);
  4. the run's augment block, when its FULL_RUNS row names one: full-p03 trains with RECIPE (DECISIONS H1 step 2, the
     CTC train-data augmentation, v2 since H6; the block names only what it changes, the trainer's defaults the rest).
  smoke-<x> is full-<x> without its augment block (the configs smoke A ran stay byte for byte as they were) and with
  SMOKE (the smoke data, SMOKE_EPOCHS: the epochs smoke A ran, 3/3/4/4, whatever the full runs' are now - the 100 h
  draw is the budget, and its plan_total_steps are the full runs' at those epochs; no complete
  evals in the loop and a 100-row greedy subset at the end; minis every 100 steps; 60 dev rows per source; full states
  every 5 min and timed states every 10 min (smoke-p01: 3 and 3, so the wipe fault's second attempt uploads one); log
  syncs every 10 min (smoke-p01: 5) with every step row full; a 2 min end reserve and 4d checked every 10 steps over a
  20-step window), memory.probe_shapes = the full data's worst micro-batches of the matching student (PLAN_FILE, so the
  smoke's memory probe sees box 1's and box 2's worst shapes), and on smoke-t06 and smoke-p03 a forced early stop
  (early_stop.min_delta_abs 1e9: no dev eval can improve by that; smoke-t06 with patience 8, so it still ends after 300
  steps or more, smoke check 2's VRAM window).

Data configs (launch's --config and bootstrap's KITSUNE_CONFIG: what a box rebuilds and pulls; no trainer settings):
  data-p01      FULL_DATA + family ctc (box p01: box 1 and its recipe test, CTC only, no pull_parakeet)
  data-t        FULL_DATA + family aed (box full-t: T-0.6B; teacher_out for every stem, no Parakeet labels)
  data-p        FULL_DATA + family ctc (box full-p: P-0.3B, the Whisper and quant pool; data-p01's content)
  data-full     FULL_DATA + pull_parakeet true (both families; the retired 2x box 2's, no registry box uses it: kept
                for scripts/make_selection.py's full mode and a box whose labels need both roots)
  data-smoke    SMOKE_DATA + pull_parakeet true (smoke A)
  data-smoke-b  study/data.json's data keys verbatim (smoke B: the frozen study selection, so every study-weight eval,
                speed probe and Whisper eval of the box shares one eval store)

Inputs besides the two generators' own: PLAN_FILE (configs/full/plan/full_study.json, in a folder of its own so every
configs/full/*.json is a config), the JSON tools/full_plan.py wrote for full.parquet and smoke.parquet at SMOKE_EPOCHS
(its "full" and "smoke" parts; --import-plan records a new measurement; the local selection paths become the repo
paths), and its "launch" part: tools/full_plan.py on full.parquet at the epochs the boxes run now (FULL_RUNS; a student
with a continuation at its CONTINUATIONS epochs: since DECISIONS H13 the cooldown re-runs keep their runs' own 6, 4
and, H14, 10), --import-launch-plan:
  python tools/full_plan.py --selection <full.parquet> --student t06=aed:450:1730:<E> --student p03=ctc:600:1350:<E>
      --student p005=ctc:1600:1500:<E> --student p01=ctc:1600:1500:<E> --json plan_launch.json
  python tools/make_full_configs.py --import-launch-plan plan_launch.json
The generator refuses a record measured with other step values or epochs. It gives the smoke configs' probe shapes
(from "full": the smoke ran those epochs), the boxes' step counts (from "launch"), and the numbers the hand-written
registry must carry (registry_numbers), which tests/test_full_configs.py checks: the full runs' plan_total_steps (the
smoke train items', for smoke check 3), their plan_hours = plan v3's hours at the step count measured on full.parquet
(PLAN_V3: hours x measured T / plan T, 2 decimals; also the full train items' max_hours until box 1's speed replaces
them), and F4's seconds, the deadline fault of smoke-p005 (deadline_fault_s). After a rebuilt selection: tools/
full_plan.py --json on full.parquet and on smoke.parquet, --import-plan with both (it prints what boxes.json must be
changed in), then those numbers into boxes.json by hand, then --check and the tests.

The boxes' hours (HOURS_BOXES: box full-t, box p005, and box p-cool, whose three continuations are the cooldown re-runs
of P-0.3B, P-0.1B and P-0.05B, DECISIONS H14): SPEED_FILE
(configs/full/plan/box2_hours.json, --import-speed) records the measured speeds box_hours projects them from: smoke A's
per-student s/step (smoke verdict check 3) and box 1's s/step, in-run overhead, CTC store and bootstrap hours
(tools/box1_go.py --json, at the runs-repo revision of box 1's 4-epoch record), a run with the augmentation recipe at
AUGMENT_STEP_FACTOR x its s/step (concat's longer rows). It prints, and --check holds, each box's train items'
max_hours and its est_hours / max_hours. A continuation's train item carries a `continues` block (kitsune.fullrun,
DECISIONS H14: the source box, run id, from / to step, resets before and the trainer sets), which --check holds equal
to continues_block's of CONTINUATIONS (continues_problems), and CONTINUATIONS to the plan record
(continuation_plan_problems).

configs/full/boxes.json, the box registry (kitsune/fullrun.py), is hand-written; --check validates it with
fullrun.registry_problems (every data and item config present, its data keys equal to its box's data config's; the
chained box p01-chain's rules of contract addendum E.1.4: its parts, hours and watchdogs, each part's extent within its
stage's rebuild config and stage 1's within stage 2's), checks that every readout of a full box fits in its run's end
reserve (readout_reserve_problems; a chain has no items, its parts are checked) and that the numbers bound to the plan
record equal it (registry_drift: smoke A's plan_total_steps / plan_hours and F4's seconds; the HOURS_BOXES' hours to
the speed record; the chain's own hours are addendum E.6's and tests/test_full_configs.py checks them against its
parts), and that every continues block is the generator's (continues_problems).

Usage:
  python tools/make_full_configs.py                  # write configs/full/*.json (and remove stale generated ones)
  python tools/make_full_configs.py --check          # exit 1 if a committed file differs or the registry is invalid
  python tools/make_full_configs.py --import-plan full_plan_full.json full_plan_smoke.json   # a new measurement
  python tools/make_full_configs.py --import-launch-plan plan_launch.json                   # the boxes' epochs
  python tools/make_full_configs.py --import-speed --smoke-verdict smoke_verdict.json      # the hours (smoke A)
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

# THE RECIPE TEST'S RECIPE (DECISIONS H0/H1/H4; box p01, instance 54012106, 2026-10-03): the CTC train-data
# augmentation (scripts/04_distill.py augment.*, kitsune/trainset.py) as box 1's cooldown re-run trained with it:
# truncate 0.3 (a row cut before a word, never between a sentence's last word and its mark, half of the cuts in a
# pause), concat 0.5 (a micro-batch's rows joined k at a time inside its planned padded frames, <= 28 s) and weak
# mixing, mix_p 0.05 (another row 5-20 dB down under the clean row's targets; the owner's launch line). Kept as the
# record of what ran: the recipe test box's --resume-set flags (then CONTINUATIONS p01 and continuation_flags, which
# the registry's continues blocks replaced with DECISIONS H14)
RECIPE_TEST = {"enabled": True, "truncate_p": 0.3, "concat_p": 0.5, "mix_p": 0.05}
# THE RECIPE (v2, DECISIONS H6, owner 2026-10-03 "Fix, then launch P-0.3B anew with the new recipe"), after the external
# review of the test's model: truncate 0.2 (0.3 left the final 。 off 3-8 % of complete sentences, JSUT 92 % against the
# gate's 95 %), truncate_min_row_s 3.0 (rows under 3 s are never cut: short utterances lost more words), concat 0.5 as
# tested (joined broadcast chunks 29.8 -> 17.4 % CER, the collab +5 pp) and mixing OFF (0.05 already made two equally
# loud voices much worse, 56 % against 47 % CER, mostly deleted text). The trainer now ends every cut row's audio at a
# drawn sample (trainset.end_samples): the first recipe cut on 80 ms boundaries only, and the tested model learned that
# alignment, not the words, as its "no mark" cue (the review's own cuts: 2.8 % false marks on a boundary, 100 %
# mid-frame - every cut through a word and most of the app's chunk ends). The quiet pads (truncate_pad_p / end_pad_p)
# stay OFF: in the CPU pilot from A0 (30 steps; streamfix/periods-model/out/pause_probe_v2.log) they only moved the
# overall mark bias (end pads 0.3 cut the right marks after a complete sentence to 0.35-0.60), and they too ended on
# frame boundaries then. Every other augment.* key stays the trainer's default, so the block names only what the recipe
# sets; mix_p stays in it, at 0, so the record says mixing is off. P-0.3B's run carries it as its augment block
# (FULL_RUNS p03)
RECIPE = {"enabled": True, "truncate_p": 0.2, "concat_p": 0.5, "mix_p": 0.0, "truncate_min_row_s": 3.0}
# THE AED RECIPE (DECISIONS H7, owner 2026-10-05: "prepare the T-box run with 8 epochs with the same recipe"): RECIPE on
# the T student's token targets (kitsune.trainset's "AED rows"). An AED row has no frame targets, so its cuts come from
# the cut table (AED_CUTS, tools/aed_cut_table.py on full.parquet and the two label roots: every frame where the
# Parakeet teacher's CTC alignment has a word start whose boundary maps onto the Cohere tokens - 96.5-99.8 % of the rows
# >= 3 s have one, 55-69 % one in a pause), pinned by its sha256 and pulled by box full-t (boxes.json extra_files); a
# cut row keeps the Cohere tokens before that boundary and ends in EOS. The joins and their cap (batch.max_dec_len) as
# the trainer's; no pads (an AED student takes none). end_trim_p 0.3 (DECISIONS H7, fix B; the owner on 2026-10-05,
# after the P-0.3B review: 7 % of JSUT's complete sentences lost their mark): about a third of the rows that are not
# cut lose their trailing silence down to a 10-80 ms tail, their mark kept, so a clean tight end is no longer only a
# cut's. Since DECISIONS H11 also recipe v3's background (below): RECIPE_AED is RECIPE_V3 plus the cut table
AED_CUTS = "labels/full/selections/full_study/aed_cuts.parquet"
AED_CUTS_SHA256 = "cde6504ff3c9a1d539385eec745ca9301570b42fc1c182ffdc16ccbf4a9eb4bb"
# RECIPE V3 (DECISIONS H8, the owner on 2026-10-05: "retrain p0.1 and p0.05 with the changes and see if our changes fixed
# the problem"): recipe v2 plus the two fixes of the P-0.3B review on the CTC students - B, end_trim_p 0.3 (a third of
# the rows that are not cut lose their trailing silence, their mark's frames moved up behind their last word:
# trainset._AugRow.trim_end), and A, background audio under 30 % of the rows (noise_p 0.3 at the trainer's 0-20 dB):
# MUSAN's music without vocals and its noise (NOISE_BANK, tools/build_noise_bank.py, pinned by the sha256 of its
# index.json and pulled by box p01, boxes.json extra_dirs), mixed under the whole row - a joined row's utterances and
# the gaps between them alike -, the targets the teacher's on the clean audio. Why: on 30 s windows of stream audio
# (talk over game sound and music) the Parakeet teacher writes nothing and P-0.3B with it, while both transcribe the
# same talk alone. Box p005 tested it first: P-0.05B from step 0 at 10 epochs (FULL_RUNS p005; DECISIONS H9, H10); the
# cooldown re-runs of P-0.3B, P-0.1B and P-0.05B on box p-cool (CONTINUATIONS, DECISIONS H13, H14) came after
# DECISIONS H12: the bank is v2 since (aug/musan-bg-v1 - music without vocals and noise only - stays on the data repo,
# never trained with): v1's music and noise plus 10 h of songs with lyrics and 10 h of speech (17 languages)
NOISE_BANK = "aug/musan-bg-v2"
NOISE_BANK_SHA256 = "1d56c7bf5cb0f1ad036f551a81c0029360ba821186f3c5abe41d344c381339f0"
RECIPE_V3 = {**RECIPE, "end_trim_p": 0.3, "noise_p": 0.3, "noise_bank": NOISE_BANK,
             "noise_bank_sha256": NOISE_BANK_SHA256}
# RECIPE V4 (DECISIONS H12, the owner on 2026-10-07: "Lets do room echo, volume and codec variations and also
# background speech and songs with lyrics. The actual output should not change, thats still the main model but it
# should become equivariant to these variables"): recipe v3 plus the other acoustic steps of kitsune.trainset, each
# per row, the targets always the teacher's on the clean row - the student is to write the main speaker's words
# whatever the room, level, codec or what plays behind them:
#   speech_p 0.15  1-4 voices behind the row at 10-25 dB of voiced power below it (half of them other rows of the
#                  micro-batch, Japanese; half the bank's speech, 17 languages); two or more are babble
#   noise_p 0.3    (v3's) now draws songs with lyrics too, by duration: music 15 h, noise 6.2 h, songs 10 h
#   reverb_p 0.2   a room impulse response of RIR_BANK (OpenSLR 28: 600 small and 600 medium simulated rooms and 325
#                  real ones; RT60 0.12 / 0.41 / 1.17 s at p10 / 50 / 90), one room for the row and its voices
#   gain_p 0.3     -20..+10 dB, clipped at full scale
#   codec_p 0.1    MP3 (30-100 kbit/s), GSM 6.10 or mu-law at 8 kHz, encoded and decoded in memory (libsndfile)
# The loader pays for them (measured on real rows: +45-65 % of its decode's CPU, the codecs most); FULL_RUNS'
# `workers` gives P-0.05B, whose GPU takes ~7,900 audio-s/s, 12 loader workers instead of the default 8
RIR_BANK = "aug/rirs-v1"
RIR_BANK_SHA256 = "af6d19310210aa2578c902b55d2e8d6f9e67924ede368a86fe5d6f8bce40e588"
RECIPE_V4 = {**RECIPE_V3, "speech_p": 0.15, "reverb_p": 0.2, "rir_bank": RIR_BANK, "rir_bank_sha256": RIR_BANK_SHA256,
             "gain_p": 0.3, "codec_p": 0.1}
# DECISIONS H11 (the owner, 2026-10-07: "make sure we have augmentation for noises, music and so on. The last
# generalisation failures were due to not having those so its extremly important we now add them"): box T's AED
# recipe takes the background as well - the same bank, rate and SNRs as the CTC students (kitsune.trainset mixes it
# under an AED row after its join, cut and trim, the Cohere tokens unchanged); box full-t pulls the bank
RECIPE_AED = {**RECIPE_V4, "cuts": AED_CUTS, "cuts_sha256": AED_CUTS_SHA256}  # recipe v4 since DECISIONS H12
# THE COOLDOWN RE-RUNS' RECIPES (DECISIONS H14, box p-cool; CONTINUATIONS). A continuation resumes a state whose config
# may hold other ranges (a re-save of an earlier test), and a resume keeps every augment key it does not set, so both
# spell the acoustic steps' ranges and the short-row guard explicitly:
#   RECIPE_V4_FULL  recipe v4 with the trainer's defaults written out (04_distill DEFAULTS augment: background noise
#                   0-20 dB, background speech 10-25 dB, volume -20..+10 dB, codecs MP3 / GSM 6.10 / mu-law,
#                   background_min_row_s 0: every row may get a background) - P-0.1B's re-run
#   RECIPE_GENTLE   recipe v4 at milder settings: background noise 5-20 dB, background speech 15-25 dB, volume
#                   -10..+10 dB, codecs MP3 and mu-law (no GSM), and no background speech or noise under rows shorter
#                   than 3 s (background_min_row_s 3.0; reverb, volume and codec still apply) - P-0.3B's and P-0.05B's
RECIPE_V4_FULL = {**RECIPE_V4, "noise_snr_db": [0.0, 20.0], "speech_snr_db": [10.0, 25.0], "gain_db": [-20.0, 10.0],
                  "codecs": ["mp3", "gsm", "ulaw8k"], "background_min_row_s": 0.0}
RECIPE_GENTLE = {**RECIPE_V4, "noise_snr_db": [5.0, 20.0], "speech_snr_db": [15.0, 25.0], "gain_db": [-10.0, 10.0],
                 "codecs": ["mp3", "ulaw8k"], "background_min_row_s": 3.0}
# RECIPE V5 - PROPOSED for the next run, DECISIONS pending; no box uses it: recipe v4 with the three fixes of the
# 2026-10-09 recipe audit (the lone-。 empties; kitsune.trainset's "augmentation" section, B1-B3) turned on -
#   cut_keep_word    every cut keeps a content token of the utterance it ends in (never the bare start piece "▁"
#                    alone: ~3.4 % of ReazonSpeech's cuts kept 1-3 s of an untranscribed lead-in with no word)
#   guard_per_piece  truncate_min_row_s 3.0 holds for each utterance of a joined row (no cut inside one under 3 s), as
#                    background_min_row_s would (0 here, as in v4: every row may get a background)
#   end_trim_voiced  the end trim keeps the audio up to the row's last voiced sound (the last word's token sits at its
#                    onset: the old trim removed a median 0.23 s of its sound) and counts no "▁" as a word
# A CTC recipe: an AED student takes guard_per_piece only (04_distill validate_augment refuses the other two there)
RECIPE_V5 = {**RECIPE_V4, "cut_keep_word": True, "guard_per_piece": True, "end_trim_voiced": True}
# the full students (contract 7): their study run, schedule.epochs, warm-up (the study's, kitsune.prereg), optim.lr,
# batch.micro_audio_s / step_audio_s (DECISIONS C10: the study's realised audio per step, tools/full_plan.py),
# eval.dev.greedy, whether the run pulls both label roots (none does: each box pulls its family's labels, data-t /
# data-p / data-p01) and schedule.end_reserve_min (end_reserve: the trainer's default 30, full-t06 55; READOUT_RESERVE
# below). EPOCHS: DECISIONS G1, confirmed by the owner on 2026-10-02 (~11:30Z, "3 / 3 / 5" after the epoch analysis,
# D:/kitsune-tmp/fullbuild/epochs/RECOMMENDATION.md): T-0.6B 3, P-0.3B 3, P-0.05B 5, each with COMMON's early-stop
# patience; P-0.3B 6 since DECISIONS H5 (the owner, 2026-10-03, after the recipe test: its augmentation varies the
# data every epoch, and the early stop still cools down early if dev stops improving); T-0.6B 8 since DECISIONS H7 (the
# owner, 2026-10-05, after P-0.3B's 6-epoch result, with the same recipe). The owner's earlier request
# of the same day to plan 10 epochs for these runs was WITHDRAWN after that
# analysis (DECISIONS G "Rejected: 10 epochs"). P-0.05B 10 and P-0.1B 8 since DECISIONS H9 (the owner, 2026-10-05, for
# the P test box: "the p-0.05 run needs more epochs as well as the 0.1 run. They are smaller and as p-0.3 has shown more
# epochs improve the performance of these models"; chose "P-0.1B 8, P-0.05B 10" and "Continue box 1's run"). P-0.1B's
# config stays box 1's 4-epoch one byte for byte (its run continues: CONTINUATIONS sets its epochs on the resume - 8
# under H9, 4 again since H13's cooldown re-run), and so do P-0.3B's 6-epoch and P-0.05B's 10-epoch ones (their H14
# re-runs set their recipes on the resume).
# This is the one epoch parameter: a change here, then --import-launch-plan of a full_plan.py record at the new
# epochs, then the printed hours (--import-speed) into boxes.json. AUGMENT: the run's augment block (full_config; the
# smoke configs never take it), None for none. DECISIONS H1 step 2: P-0.3B trains with RECIPE (v2 since H6, after
# the recipe test on box p01 and its external review); T-0.6B (box full-t) with RECIPE_AED (H7); P-0.05B with RECIPE_V3
# on box p01 (H8); P-0.1B's 4-epoch run (box 1, done: its config is the one the continuation resumes, byte for byte) has
# none - its continuation sets RECIPE_V3 (CONTINUATIONS)
FULL_RUNS = {
    "t06": dict(study_run="study-t06", epochs=8, warmup=300, lr=2e-4, micro=450, step=1730, dev_greedy=False,
                pull_parakeet=False, end_reserve=55, augment=RECIPE_AED),
    "p03": dict(study_run="study-p03", epochs=6, warmup=300, lr=2e-4, micro=600, step=1350, dev_greedy=True,
                pull_parakeet=False, end_reserve=30, augment=RECIPE),
    "p01": dict(study_run="study-p01", epochs=4, warmup=1000, lr=1e-3, micro=1600, step=1500, dev_greedy=True,
                pull_parakeet=False, end_reserve=30, augment=None),
    "p005": dict(study_run="study-p005", epochs=10, warmup=1000, lr=1e-3, micro=1600, step=1500, dev_greedy=True,
                 pull_parakeet=False, end_reserve=30, augment=RECIPE_V4, workers=12),
}
# the epochs smoke A ran (2026-10-01): the smoke configs keep them (the 100 h draw is the smoke's budget), and the plan
# record's "full" and "smoke" parts are measured at them (smoke A's check 3 projected the full runs at these T; F4's
# seconds come from smoke-p005's), so neither moves when the full runs' epochs do
SMOKE_EPOCHS = {"t06": 3, "p03": 3, "p01": 4, "p005": 4}
# the smoke configs' data block: before box 2's split every full run but full-p01 pulled both roots, and the smoke
# config is built from the full one, so pull_parakeet sat at the study config's position in smoke-t06/p03/p005 and at
# the end of smoke-p01; building them from this keeps every smoke config byte-identical (smoke A ran them)
SMOKE_BASE_PULL = ("t06", "p03", "p005")
# box p01's continuation, now THE RECIPE TEST (DECISIONS H1; the 8-epoch continuation of G3, patience 12, is postponed:
# H2), launched with --resume-reset run_id --resume-set run_id:schedule.epochs=<epochs> and one --resume-set
# run_id:augment.<key>=<value> per RECIPE_TEST key (fullrun.RESUME_SET_KEYS). It re-runs ONLY the
# cooldown of box 1's run: from its pre_cooldown state (from_step: checkpoints/full_step_86328, on the Hub) to the SAME
# end (epochs 4: T 107,910 on full.parquet, so 21,582 steps, ~2 h) with the recipe on. The WSD cooldown starts at
# t_c = 0.8 T = 86,328 = from_step: the whole re-run is cooldown. Everything before from_step - the data order, the
# batches, the LR - is the 4-epoch baseline's, and so are the cooldown's step plan and LR (the planner resumes at the
# state's position, T and t_c are the baseline's): its readout against the baseline's (runs-repo revision c4604304, M4
# 11.45 %, G4) isolates the recipe, a paired A/B. No patience set (patience None): the early stop's action is
# "cooldown", and a trigger inside a cooldown changes nothing (04_distill early_stop_trigger: a run already in its
# cooldown goes on to its scheduled end; end_reason stays "schedule"), so the state's patience 6 stays, as in the
# baseline, and the only config difference is augment.*. from_step is the pre_cooldown state the reset goes on from;
# the 4-epoch record stays runs-repo revision c4604304 (G4): the re-run overwrites the run's summary, config, final
# evals, export and its pre_cooldown state's trainer.pt/.json (same weights, the recipe in its config) at the head
# DECISIONS H8: box 1's run again from the same state, now with RECIPE_V3 (RECIPE_TEST stays the record of what the
# first test ran); H9: to 8 EPOCHS - G3's continuation, with its patience 12, after all. The reset plans T for 8 epochs
# on full.parquet (~215,800 steps; plan["launch"]), so the stable phase at the peak LR goes on from from_step (the
# state is the end of box 1's stable phase: no LR jump) to t_c = 0.8 T and the cooldown follows: ~129,500 steps, every
# one with the recipe. Its early stop now matters (a long stable phase before the cooldown): patience 12 dev checks
# (1.2 epochs at eval.dev.every_epochs 0.1) instead of COMMON's 6, so the switch to augmented data does not cool it
# down early (the first test's dev CER jumped 0.155 -> 0.170 at its first check with the recipe, then fell). Not a
# paired A/B with box 1 and the first test any more (more epochs and the recipe together):
# the review's fixes are checked by its probes
# DECISIONS H13 (the owner, 2026-10-08, on whether P-0.3B's and P-0.1B's 80 % checkpoints with the new recipe would fix
# them and take on its behaviour: "Build the setup already for both"): THE RECIPE-V4 COOLDOWN RE-RUNS, H1's design
# again. Each continuation re-runs ONLY its run's cooldown: from its pre_cooldown state (the runs repo's
# checkpoints/full_step_<from_step>, trainer.json reason pre_cooldown) to the SAME end (the run's own epochs, so its T,
# and t_c = 0.8 T = from_step: the whole re-run is cooldown) with RECIPE_V4 on - P-0.3B on box full-p (169,376 ->
# 211,720: 42,344 steps) and P-0.1B on box p01 (86,328 -> 107,910: 21,582 steps). Everything before from_step is the
# baseline's (data order, batches, LR), so each readout against its baseline record - `revision`: P-0.3B's box full-p
# final summary (3674d2d7, M4 9.73 %), P-0.1B's box 1 record (c4604304, M4 11.45 %; G4) - isolates what recipe v4's
# cooldown adds and costs: a paired A/B. No patience set: an early-stop trigger inside a cooldown changes nothing
# (04_distill early_stop_trigger), so each state's own patience stays. The readouts write new dirs (P-0.3B's state
# counts no reset yet: -r1; P-0.1B's is A1's re-save, st.resume_resets 1: -r2), the baselines' readouts stay; a re-run
# overwrites its run's summary, config, final evals, export and its pre_cooldown state's trainer.pt/.json (same
# weights) at the head of the runs repo - the baselines stay at their revisions -, as A1 did. H9's 8-epoch
# continuation of P-0.1B is set aside, not lost: it can start later from the same pre_cooldown weights (epochs 8,
# patience 12, as H9 planned)
# DECISIONS H14 (the owner, 2026-10-09): ONE BOX, p-cool (1x RTX 5090, data-p01: one rebuild and one CTC store, which
# the three P runs' configs share - the full data, family ctc), re-runs three done runs' cooldowns, IN THIS ORDER, each
# from its runs-repo pre_cooldown state (checkpoints/full_step_<from_step>, trainer.json reason pre_cooldown) to the
# run's own end (to_step), then that run's M4 readout and 7 quantised readouts (H13's two re-runs moved there, box
# p005's P-0.05B joined them; boxes full-p and p01 hold their runs' records again):
#   p03-cool   P-0.3B, box full-p's 6-epoch run (baseline revision 3674d2d7, M4 9.73 %): 169,376 -> 211,720 (42,344
#              steps, all of them cooldown: t_c = 0.8 T = from_step), RECIPE_GENTLE; its state counts no reset: -r1
#   p01-cool   P-0.1B, box 1's 4-epoch run (c4604304, M4 11.45 %, G4): 86,328 -> 107,910 (21,582 steps), RECIPE_V4_FULL;
#              the runs repo's state is A1's re-save (st.resume_resets 1): -r2
#   p005-cool  P-0.05B, box p005's 10-epoch run (d2caddfc), which its early stop cooled down: from that cooldown's start,
#              67,449, to its T, 80,939 (13,490 steps), RECIPE_GENTLE, keep_cooldown: the reset keeps the state's
#              early-stop cooldown record (04_distill schedule.resume_reset_keep_cooldown), so the re-run replays the
#              same t_c, T and LR instead of planning the 10-epoch schedule again; no reset yet: -r1
# Each sets the run's own epochs (so a re-planned T is the run's), schedule.deadline_cooldown false (4d must never
# compress a paired cooldown: a box out of time is stopped by the watchdog and resumed), p005-cool
# schedule.resume_reset_keep_cooldown true, and every key of its recipe, the ranges included (continuation_set_dict).
# The registry carries them (boxes.json p-cool's train items' continues blocks, continues_block; --check holds them), so
# the box always resumes and launch --box p-cool takes no reset or set flags. box: where it runs; source_box: the box
# whose Hub summary records the run (the same item name and config); resets_before: the state's st.resume_resets, so
# the readout writes runs/m4-<run_id>-r<resets_before + 1>; revision: the baseline record (runs repo) the readout is
# paired with: everything before from_step is the baseline's, so the readout isolates the cooldown's recipe. A re-run
# overwrites its run's summary, config, final evals and export at the runs repo's head; the baselines stay at their
# revisions. Queue order = this dict's order
CONTINUATIONS = {
    "p03-cool": dict(box="p-cool", source_box="full-p", student="p03", run_id="full-p03-20261003T230143Z",
                     from_step=169376, to_step=211720, resets_before=0, epochs=6, keep_cooldown=False,
                     augment=RECIPE_GENTLE, revision="3674d2d7ae7480594b079234b1549014fe96f783"),
    "p01-cool": dict(box="p-cool", source_box="p01", student="p01", run_id="full-p01-20261001T184145Z",
                     from_step=86328, to_step=107910, resets_before=1, epochs=4, keep_cooldown=False,
                     augment=RECIPE_V4_FULL, revision="c4604304db76e068df7bbe39d00d006b74d6c134"),
    "p005-cool": dict(box="p-cool", source_box="p005", student="p005", run_id="full-p005-20261008T160232Z",
                      from_step=67449, to_step=80939, resets_before=0, epochs=10, keep_cooldown=True,
                      augment=RECIPE_GENTLE, revision="d2caddfc308c10495253c0204c4edf8a6222dac0"),
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

# THE BOXES' HOURS (contract 7 "item hours after box 1", 12.5, Resolution 31; DECISIONS G2/G3). The HOURS_BOXES' train
# items' max_hours and their est_hours / max_hours are projected from MEASURED speeds, recorded in SPEED_FILE (written
# by --import-speed, never by hand): smoke A's median steady s/step per student (smoke verdict check 3's sec_per_step,
# the only measurement of T-0.6B, P-0.3B and P-0.05B at the full configs' batches) and box 1's own (tools/box1_go.py
# --json --revision <its 4-epoch record>). Box 1 trained P-0.1B only, so the refresh is a transfer: every student's
# smoke s/step x r, r = box 1's steady s/step / smoke A's of P-0.1B (the host and full-data effect, 1.036; 1 without
# box 1), and box 1's in-run overhead o (0.0465) replaces the plan's 0.08. Without box 1's part the record is
# smoke-only and launch warns that the hours are provisional.
# The run model is plan v3's (calc_v3.run_h): steps x s/step x r x (1 + o) + the run's fixed time (start-up, memory
# probe, first and final complete evals, end phase) + its dev checks (every COMMON eval.dev.every_epochs epoch). It
# reproduces box 1 (9.16 h modelled vs 9.152 h measured) and so already holds box 1's 0.305 / 0.2715 s/step (x 1.124,
# evals included): that factor is r x (1 + o) plus the fixed and dev-check time, and must not be applied again. The
# steps are plan["launch"]'s (full.parquet at the epochs the boxes run now); a continuation runs from its pre_cooldown
# step to its new T, with the dev checks of the epochs left. A run with the augmentation recipe steps at
# augment_step_factor x the s/step (below).
SPEED_FILE = "plan/box2_hours.json"  # under OUT_DIR, next to PLAN_FILE (launch's SPEED_RECORD names it)
# the boxes whose hours are box_hours of the speed record (DECISIONS G2/G3/H1: box 2 as two 1x RTX 5090 boxes, T and P,
# and the continuation boxes): their train runs, in queue order, a FULL_RUNS student's run from step 0 or
# "cont:<name>" for CONTINUATIONS[name]. Box full-t trains T-0.6B; box p005 trained P-0.05B alone (the P test box,
# DECISIONS H10; its fresh run's hours, the record); box p-cool the three cooldown re-runs (DECISIONS H14). Boxes full-p
# and p01 left it with H14: their runs are done (full-p's 6-epoch run; box 1's 4 epochs and A1's re-run) and their
# entries are the records of what they ran or had planned (origin/main's, H9's 8-epoch continuation on p01), so their
# hours are no longer held to the speed record
HOURS_BOXES = {"full-t": ("t06",), "p005": ("p005",), "p-cool": ("cont:p03-cool", "cont:p01-cool", "cont:p005-cool")}
# CONCAT'S EXTRA ATTENTION (DECISIONS H1: "add a few % for concat's extra attention"). A joined row holds k utterances
# (k <= concat_max_n 4, <= 28 s) in the frames its micro-batch had planned, but its self-attention grows with the
# square of its length: on full.parquet at concat_p 0.5 the attention's elements grow 1.27x over the planned
# micro-batches (the training-plan investigator's count, streamfix/training-plan/v2/aug_stats.json, P-0.1B and P-0.3B
# alike), and self-attention is a small share of a P student's step on the 5090 (the projections, FFNs and
# convolutions, the KD loss and the loader take the rest), so a run that joins rows is planned at +5 % on its steady
# s/step [X]. truncate and mix add no GPU time (a cut row is shorter, a mix is one add in the loader). Measured by the
# recipe test itself: its steps.parquet s/step against box 1's own cooldown steps gives the factor P-0.3B's box takes
AUGMENT_STEP_FACTOR = 1.05
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
STORES_PESS = 1.91 / 1.37  # a box-1 record's measured CTC store: central x 1, pessimistic x this
# ONE store per box (each box pulls one family's labels): box 1's measured CTC store (0.548 h), the AED-only store of
# box full-t taken as equal [X] (smoke A's AED store took half the CTC one's time, but on the full extent T-0.6B's
# targets are 10 GiB of tokens); without a box-1 record calc_v3's both-stores time / (1 + AED_STORE_SHARE)
# the tails, (central, worst) hours after each box's last run [M + X]; the worst case also holds the last droppable
# item's whole max_hours (the no-start rule's need):
#   box full-t: m4-full-t06 (smoke: 1.3 min on the 100 h run's eval sets, which are the full ones), 7 T quant readouts
#               (smoke-B #1: 0.7-1.5 min each on P-0.3B, T ~3x) and the decision-22 re-time pair (~1 min), one GPU
#   box full-p: 2 M4 readouts, 14 P quant readouts (smoke-B #1: 0.65-1.0 min each) and the four Whisper models on all
#               five sets (whisper-large-v3: 5,000 JSUT rows in 3.2 min; ~25 min for all four), one GPU. Kept as it
#               is with P-0.05B postponed (H2): its M4 and 7 quant readouts (~0.2 h) left the box and stay in the
#               tail as margin, so the box need not be re-derived when P-0.05B comes back
#   box p01:    m4-full-p01 and 7 P-0.1B quant readouts, plus resume-pull and check-resume before the stores
#               (CONT_PULL_H: any box whose runs hold a continuation)
#   box p005:   m4-full-p005 and 7 P-0.05B quant readouts (box p01's readouts without the pull)
#   box full-p's cooldown re-run (DECISIONS H13): m4-full-p03 and its 7 quant readouts only - the Whisper evals are
#               done and verified, so resume-pull keeps them done -, measured on 2026-10-04 at ~7 min for all eight
#               (21:05-21:12Z: its run dirs' timestamps), so box p01's tail, plus the pull (CONT_PULL_H)
#   box p-cool: three M4 readouts and 21 quant readouts (P-0.3B's eight took ~7 min, P-0.1B's and P-0.05B's are
#               smaller) [X], plus one pull per continuation (CONT_PULL_H: resume-pull downloads each run's
#               pre_cooldown state, check-resume runs before each)
# Boxes full-p and p01 are no HOURS_BOXES since DECISIONS H14 (their entries are records): their tails stay as the
# record of their hours
POST_T_H = (0.6, 1.35)
POOL_P_H = (0.9, 1.65)  # box full-p's fresh run (DECISIONS G2), the record of its hours
P01_TAIL_H = (0.2, 0.5)
P005_TAIL_H = (0.2, 0.5)
P03_CONT_TAIL_H = (0.2, 0.5)
PCOOL_TAIL_H = (0.4, 1.0)
CONT_PULL_H = (0.1, 0.3)  # once per continuation of the box
BOX_TAIL_H = {"full-t": POST_T_H, "p005": P005_TAIL_H, "p-cool": PCOOL_TAIL_H}
END_H = 0.35  # calc_v3 end: finish's uploads and the destroy
# max_hours' host margin: smoke A ran on a Ryzen 9950X (calc_v3's "fast" CPU class); calc_v3's pessimistic T-0.6B
# epoch is 13.635 / 10.846 = 1.26 x its fast one (box 2's only 2x offer on 2026-10-01, m54650, a Zen2 EPYC, is
# blocklisted since: any slow-CPU host is what it stands for)
HOST_PESS = 1.26
STALL_RECOVERY_H = 1.25  # one trainer stall: stall_min 45 + up to 30 min of progress since its last full state


class PlanError(ValueError):
    """PLAN_FILE is missing, malformed, or measured with other step values than FULL_RUNS'."""


class SpeedError(ValueError):
    """SPEED_FILE is malformed, or an --import-speed input lacks what box_hours needs."""


# ------------------------------------------------------------------------------------------------ the plan record


def launch_epochs(x: str) -> int:
    """The epochs student x's run is planned for now: its continuation's (CONTINUATIONS: the run's own since DECISIONS
    H13, P-0.1B 4), else FULL_RUNS'."""
    cont = next((c for c in CONTINUATIONS.values() if c["student"] == x), None)
    return int(cont["epochs"] if cont else FULL_RUNS[x]["epochs"])


def _plan_want(x: str, epochs: int) -> dict:
    r = FULL_RUNS[x]
    return dict(family="aed" if x.startswith("t") else "ctc", micro_audio_s=float(r["micro"]),
                step_audio_s=float(r["step"]), epochs=int(epochs))


def plan_problems(plan) -> list[str]:
    """Why a PLAN_FILE record cannot serve the generator: {"full": <tools/full_plan.py JSON of full.parquet>,
    "smoke": <the same of smoke.parquet>} measured at SMOKE_EPOCHS, and "launch": <the same of full.parquet> at
    launch_epochs (FULL_RUNS', a continuation's), each at the repo's selection path (launch: the same file as full, by
    its sha256), measured for every FULL_RUNS student with its family, micro-batch, step and epochs, with total_steps
    (steps_per_epoch on launch: a continuation's dev checks; worst_shapes on full.parquet)."""
    if not isinstance(plan, dict):
        return [f"not an object: {type(plan).__name__}"]
    p = []
    for which, sel in (("full", fullrun.FULL_SELECTION), ("smoke", fullrun.SMOKE_SELECTION),
                       ("launch", fullrun.FULL_SELECTION)):
        rec = plan.get(which)
        if not isinstance(rec, dict):
            p.append(f"{which}: no tools/full_plan.py record" + (" (make_full_configs --import-launch-plan)"
                                                                  if which == "launch" else ""))
            continue
        got_sel = (rec.get("selection") or {}).get("path")
        if got_sel != sel:
            p.append(f"{which}: measured on {got_sel!r}, not {sel}")
        if which == "launch" and isinstance(plan.get("full"), dict):
            a, b = (rec.get("selection") or {}).get("sha256"), (plan["full"].get("selection") or {}).get("sha256")
            if a != b:
                p.append(f"launch: measured on a full.parquet of sha256 {a!r}, the full part's is {b!r}")
        students = rec.get("students") if isinstance(rec.get("students"), dict) else {}
        for x in FULL_RUNS:
            s = students.get(x)
            if not isinstance(s, dict):
                p.append(f"{which}: no student {x}")
                continue
            ep, src = (launch_epochs(x), "FULL_RUNS / CONTINUATIONS") if which == "launch" else (SMOKE_EPOCHS[x],
                                                                                                 "SMOKE_EPOCHS")
            want = _plan_want(x, ep)
            if diff := {k: s.get(k) for k in want if s.get(k) != want[k]}:
                p.append(f"{which}.{x}: measured with {diff}, {src} says {({k: want[k] for k in diff})}")
            if not (isinstance(s.get("total_steps"), int) and s["total_steps"] > 0):
                p.append(f"{which}.{x}: total_steps {s.get('total_steps')!r}")
            if which == "full" and not (isinstance(s.get("worst_shapes"), list) and s["worst_shapes"]):
                p.append(f"{which}.{x}: no worst_shapes")
            spe = s.get("steps_per_epoch")
            if which == "launch" and not (isinstance(spe, list) and len(spe) == ep and all(
                    isinstance(v, int) and v > 0 for v in spe) and sum(spe) == s.get("total_steps")):
                p.append(f"{which}.{x}: steps_per_epoch {spe!r} is not {ep} counts summing to total_steps")
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
    """The full run's T on full.parquet at SMOKE_EPOCHS (plan_epochs pins it at the start): the registry's smoke
    plan_total_steps (smoke check 3's projection)."""
    return int(plan["full"]["students"][x]["total_steps"])


def launch_total_steps(x: str, plan: dict) -> int:
    """The T box x's run plans now: full.parquet at launch_epochs (plan["launch"])."""
    return int(plan["launch"]["students"][x]["total_steps"])


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
    """Why a SPEED_FILE record cannot serve box_hours: {"smoke": {..., "sec_per_step": {<every FULL_RUNS student>:
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


def _run_h(steps: float, sps: float, r: float, o: float, fixed_s: float, dev_checks: float, dev_s: float) -> float:
    """Plan v3's run model (calc_v3.run_h): steps x s/step x r x (1 + o) + fixed + dev checks, hours rounded up."""
    return _up((steps * sps * r * (1 + o) + fixed_s + dev_checks * dev_s) / 3600, 0.01)


def augment_step_factor(augment: dict | None) -> float:
    """The run model's s/step factor of a run with this augment block (RECIPE, or None): AUGMENT_STEP_FACTOR when it
    joins rows (enabled, concat_p > 0), else 1."""
    a = augment or {}
    return AUGMENT_STEP_FACTOR if a.get("enabled") and float(a.get("concat_p") or 0) > 0 else 1.0


def continuation_run(name: str, plan: dict, rec: dict) -> dict:
    """A continuation's run (CONTINUATIONS[name]): the steps from its pre_cooldown state to its end (to_step -
    from_step), their dev checks (the steps over the run's steps per epoch on full.parquet - plan["launch"] at its epochs - every
    COMMON eval.dev.every_epochs: P-0.05B's 13,490 steps of 26,977.7 per epoch, 5.0), and its hours by the run model (at
    augment_step_factor x the s/step when its recipe joins rows). total_steps is the step it trains to (to_step: the
    run's T, P-0.05B's its early stop's cooldown T), plan_total_steps the launch record's T at its epochs."""
    c = CONTINUATIONS[name]
    x = c["student"]
    sps, b1 = rec["smoke"]["sec_per_step"], rec.get("box1")
    r = b1["sec_per_step"] / sps["p01"] if b1 else 1.0
    o = b1["overhead"] if b1 and b1.get("overhead") is not None else OVERHEAD_PLAN
    st = plan["launch"]["students"][x]
    T = int(st["total_steps"])
    steps = int(c["to_step"]) - int(c["from_step"])
    checks = steps / (T / len(st["steps_per_epoch"])) / COMMON["eval"]["dev"]["every_epochs"]
    f = augment_step_factor(c.get("augment"))
    return dict(name=name, student=x, total_steps=int(c["to_step"]), plan_total_steps=T, steps=steps,
                dev_checks=round(checks, 2), step_factor=f,
                hours=_run_h(steps, sps[x] * f, r, o, FIXED_S[x], checks, DEV_CHECK_S[x]))


def box_hours(plan: dict, rec: dict, box: str, reserve_min: float = 60) -> dict:
    """Box `box`'s hours (HOURS_BOXES) from the plan record's launch step counts and a SPEED_FILE record (the model
    above; reserve_min: the box's deadline_reserve_min). One GPU, one queue: setup, its one store, its runs in order
    (a continuation: continuation_run), its tail (BOX_TAIL_H, plus CONT_PULL_H once per continuation) and the end.
    Returns r, o, run_h (every FULL_RUNS student's run at
    launch_epochs and augment_step_factor of its augment block, P-0.1B's from step 0 as a cross-check), items (the
    box's train items' max_hours), continuations (the box's continuation_run results, in queue order; continuation: the
    first of them, None for a box without), setup_h, store_h and tail_h as (central, pessimistic), est_hours (the
    central path)
    and max_hours (the pessimistic setup and store, the runs x HOST_PESS, the worst tail, the deadline reserve and one
    stall), and slack_h: what the pessimistic path leaves before the last run's deadline-cooldown point."""
    sps, b1 = rec["smoke"]["sec_per_step"], rec.get("box1")
    r = b1["sec_per_step"] / sps["p01"] if b1 else 1.0
    o = b1["overhead"] if b1 and b1.get("overhead") is not None else OVERHEAD_PLAN
    dev_every = COMMON["eval"]["dev"]["every_epochs"]
    run_h = {x: _run_h(launch_total_steps(x, plan), sps[x] * augment_step_factor(FULL_RUNS[x]["augment"]), r, o,
                       FIXED_S[x], launch_epochs(x) / dev_every, DEV_CHECK_S[x]) for x in FULL_RUNS}
    setup_c, setup_p = SETUP_H["central"], SETUP_H["pess"]
    if b1 and b1.get("bootstrap_h"):
        setup_p = max(setup_p, round(b1["bootstrap_h"] + 0.2, 2))
    store_c = b1["stores_ctc_h"] if b1 and b1.get("stores_ctc_h") else STORES_H["central"] / (1 + AED_STORE_SHARE)
    store_p = round(store_c * STORES_PESS, 2)
    runs = HOURS_BOXES[box]
    conts, items, last = [], {}, None
    for x in runs:  # in queue order; "cont:<name>": CONTINUATIONS[name]
        if x.startswith("cont:"):
            cr = continuation_run(x[len("cont:"):], plan, rec)
            conts.append(cr)
            items[f"full-{cr['student']}"], last = cr["hours"], cr["student"]
        else:
            items[f"full-{x}"], last = run_h[x], x
    # each continuation's run is pulled (its pre_cooldown state) and checked before the stores
    tail = tuple(t + len(conts) * c for t, c in zip(BOX_TAIL_H[box], CONT_PULL_H))
    train = sum(items.values())
    est = _up(setup_c + store_c + train + tail[0] + END_H, 0.1)
    pess_end = setup_p + store_p + train * HOST_PESS
    mx = _up(pess_end + tail[1] + reserve_min / 60 + STALL_RECOVERY_H, 1)
    # 4d plans the last run's cooldown to end its end_reserve_min before KITSUNE_DEADLINE (the box's less reserve_min)
    slack = mx - reserve_min / 60 - FULL_RUNS[last]["end_reserve"] / 60 - pess_end
    return dict(box=box, r=round(r, 4), o=o, run_h=run_h, items=items, continuations=conts,
                continuation=conts[0] if conts else None,
                setup_h=(setup_c, setup_p), store_h=(round(store_c, 3), store_p),
                tail_h=tuple(round(t, 2) for t in tail),
                est_hours=est, max_hours=int(mx), slack_h=round(slack, 2))


def continuation_set_dict(name: str) -> dict:
    """A continuation's trainer sets (CONTINUATIONS[name]) as the registry's continues block carries them, native JSON
    values in this order: schedule.epochs (the run's own), schedule.deadline_cooldown false (4d never compresses a paired
    cooldown), schedule.resume_reset_keep_cooldown true when it keeps the state's early-stop cooldown, then one
    augment.<key> per key of its recipe. The queue passes them after schedule.resume_reset=true, each spelled by
    fullrun.continue_set_text."""
    c = CONTINUATIONS[name]
    out = {"schedule.epochs": int(c["epochs"]), "schedule.deadline_cooldown": False}
    if c["keep_cooldown"]:
        out["schedule.resume_reset_keep_cooldown"] = True
    out.update({f"augment.{k}": copy.deepcopy(v) for k, v in (c.get("augment") or {}).items()})
    return out


def continues_block(name: str) -> dict:
    """The continues block of CONTINUATIONS[name]'s train item in boxes.json (kitsune.fullrun "Continuations"): the
    source box, the run id, from / to step, the resets before and the sets (continuation_set_dict)."""
    c = CONTINUATIONS[name]
    return dict(box=c["source_box"], run_id=c["run_id"], from_step=int(c["from_step"]), to_step=int(c["to_step"]),
                resets_before=int(c["resets_before"]), sets=continuation_set_dict(name))


def _uncommented(d: dict) -> dict:
    """A continues block without its comment keys ("_..."), at its top and in its sets."""
    out = {k: v for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(out.get("sets"), dict):
        out["sets"] = {k: v for k, v in out["sets"].items() if not str(k).startswith("_")}
    return out


def continues_problems(reg: dict) -> list[str]:
    """Where a loaded registry's continues blocks differ from the generator's: every CONTINUATIONS entry is the
    continues block (continues_block) of its box's train item full-<student> - its sets in the same order and spelling
    (fullrun.continue_sets; JSON's 1 is not true) -, and no train item continues a run CONTINUATIONS does not name."""
    p, boxes = [], reg.get("boxes") or {}
    want: dict[str, dict] = {}
    for n, c in CONTINUATIONS.items():
        want.setdefault(c["box"], {})[f"full-{c['student']}"] = n
    for bname in boxes:
        got, mine = fullrun.continues_of(bname, reg), want.get(bname) or {}
        for item, n in mine.items():
            if item not in got:
                p.append(f"boxes.{bname}: no train item {item} with a continues block (make_full_configs "
                         f"CONTINUATIONS {n!r})")
                continue
            g, w = _uncommented(got[item]), continues_block(n)
            diff = [k for k in w if k != "sets" and g.get(k, "<absent>") != w[k]]
            try:
                same_sets = fullrun.continue_sets(g) == fullrun.continue_sets(w) and g["sets"] == w["sets"]
            except (ValueError, KeyError):
                same_sets = False
            diff += [] if same_sets else ["sets"]
            if diff:
                p.append(f"boxes.{bname}.items.{item}.continues differs from make_full_configs "
                         f"continues_block({n!r}) in {diff}")
        p += [f"boxes.{bname}.items.{item}.continues: not a make_full_configs CONTINUATIONS entry" for item in got
              if item not in mine]
    p += [f"boxes.{b}: not in the registry (make_full_configs CONTINUATIONS)" for b in want if b not in boxes]
    return p


def continuation_plan_problems(plan: dict) -> list[str]:
    """Where CONTINUATIONS disagree with the plan record's launch part (the student's plan at the run's own epochs): a
    re-run that plans its schedule anew (no keep_cooldown) goes from the step before the planned cooldown
    (cooldown_start_step - 1: its pre_cooldown state) to the plan's T; one that keeps its early stop's cooldown ends at
    or before the plan's T. Every continuation runs at least one step, at FULL_RUNS' epochs (the run's own)."""
    p = []
    for n, c in CONTINUATIONS.items():
        st = plan["launch"]["students"][c["student"]]
        T, cs = int(st["total_steps"]), st.get("cooldown_start_step")
        if not 0 < c["from_step"] < c["to_step"]:
            p.append(f"CONTINUATIONS.{n}: from_step {c['from_step']} .. to_step {c['to_step']} is no run")
        if c["epochs"] != FULL_RUNS[c["student"]]["epochs"]:
            p.append(f"CONTINUATIONS.{n}: epochs {c['epochs']}, the run's own are {FULL_RUNS[c['student']]['epochs']} "
                     f"(FULL_RUNS)")
        if c["keep_cooldown"]:
            if c["to_step"] > T:
                p.append(f"CONTINUATIONS.{n}: to_step {c['to_step']} is past the plan's T {T}: a kept early-stop "
                         f"cooldown ends before the planned end")
        else:
            if c["to_step"] != T:
                p.append(f"CONTINUATIONS.{n}: to_step {c['to_step']}, the plan record's T at {c['epochs']} epochs is "
                         f"{T}")
            if cs is not None and c["from_step"] != cs - 1:
                p.append(f"CONTINUATIONS.{n}: from_step {c['from_step']}, the plan's pre_cooldown step is {cs - 1}")
    return p


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
    rec = {"_comment": "The full boxes' measured speeds (tools/make_full_configs.py --import-speed; never edited by "
                       "hand): smoke: smoke A's verdict check 3 (median steady s/step per student at the full configs' "
                       "batches); box1: box 1's measurement from tools/box1_go.py --json at the runs-repo revision of "
                       "its 4-epoch record (null until box 1 ended: the hours are then provisional). "
                       "make_full_configs.box_hours projects the HOURS_BOXES' train items' max_hours and their "
                       "est_hours / max_hours from it, and --check holds boxes.json to them.",
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
# 3's projection) and F4's seconds. The HOURS_BOXES' train hours and their est / max hours are held to the speed
# record instead (SPEED_FILE, box_hours), when there is one (boxes full-t, p005 and p-cool; boxes full-p and p01, records
# since DECISIONS H14, are held to neither)
PLAN_BOUND_BOXES = {"full-smoke": "smoke"}


def registry_drift(reg: dict, plan: dict, speed: dict | None = None) -> list[str]:
    """Where a loaded registry's plan-bound numbers (PLAN_BOUND_BOXES) differ from registry_numbers(plan), and, given a
    speed record, where the HOURS_BOXES' differ from box_hours: after --import-plan of a rebuilt selection,
    --import-launch-plan or --import-speed of a new measurement, what boxes.json must be changed in by hand."""
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
    for bname in (HOURS_BOXES if speed is not None else ()):
        if bname not in boxes:
            continue
        box = boxes[bname]
        h = box_hours(plan, speed, bname, reserve_min=box.get("deadline_reserve_min") or 60)
        items = {it["name"]: it for it in box.get("items") or []}
        for name, want in h["items"].items():
            if name in items and items[name].get("max_hours") != want:
                p.append(f"boxes.{bname}.items.{name}: {{'max_hours': {items[name].get('max_hours')}}}, the speed "
                         f"record gives {{'max_hours': {want}}}")
        got = {"est_hours": box.get("est_hours"), "max_hours": box.get("max_hours")}
        exp = {"est_hours": h["est_hours"], "max_hours": h["max_hours"]}
        if got != exp:
            p.append(f"boxes.{bname}: {got}, the speed record gives {exp}")
    return p


def box1_wall(rec: dict) -> str:
    """Box 1's measured training wall, for the cross-check line."""
    b1 = rec.get("box1") or {}
    return f"{b1['train_wall_h']:g} h measured" if _num(b1.get("train_wall_h")) else "not measured"


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


def full_config(x: str, r: dict | None = None, data: dict | None = None, augment: bool = True) -> dict:
    """full-<x>: the study run's config with the full data (data: another data block in its place), FULL_RUNS[x],
    COMMON and, unless augment is false, the run's augment block (FULL_RUNS[x]["augment"]: the keys it changes, at the
    config's end; the trainer's DEFAULTS fill the rest)."""
    run = FULL_RUNS[x]
    cfg = _with_data(study.run_config(run["study_run"], r, RUNS_REPO), full_data(x) if data is None else data)
    cfg = study._merge(cfg, COMMON)
    cfg = study._merge(cfg, {
        "run_name": f"full-{x}",
        "schedule": {"epochs": run["epochs"], "warmup_steps": run["warmup"], "end_reserve_min": run["end_reserve"]},
        "optim": {"lr": run["lr"]},
        "batch": {"micro_audio_s": run["micro"], "step_audio_s": run["step"]},
        "eval": {"dev": {"greedy": run["dev_greedy"]}},
    })
    if augment and run["augment"]:
        cfg = study._merge(cfg, {"augment": dict(run["augment"])})
    if augment and run.get("workers"):  # the full run's loader (never the smoke configs': smoke A's, byte for byte)
        cfg = study._merge(cfg, {"perf": {"num_workers": int(run["workers"])}})
    return cfg


def smoke_config(x: str, plan: dict, r: dict | None = None) -> dict:
    """smoke-<x>: full-<x> without its augment block on the smoke data with SMOKE, SMOKE_EPOCHS, the full data's worst
    shapes and the forced trigger. Never the augment block: the smoke configs are the ones smoke A ran, byte for byte
    (tests/test_full_configs.py SMOKE_SHA256), and the recipe's own test is box p01's (DECISIONS H1), not a smoke."""
    base = dict(copy.deepcopy(fullrun.FULL_DATA), pull_parakeet=True) if x in SMOKE_BASE_PULL else None
    cfg = _with_data(full_config(x, r, data=base, augment=False), smoke_data())
    cfg = study._merge(cfg, {"schedule": {"epochs": SMOKE_EPOCHS[x]}})
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
    """The data configs (the registry's box data_config values; data-full: no box's, kept)."""
    return {
        "data-p01": {"_comment": "Box p01's data config (box 1, P-0.1B alone; launch --config, bootstrap "
                                 "KITSUNE_CONFIG): the full selection and extent (kitsune.fullrun FULL_DATA), family "
                                 "ctc and no pull_parakeet, so the box pulls parakeet_out for every stem and "
                                 "teacher_out for the eval sets' eval stems only (kitsune.extent.pull_plan, fix 9). "
                                 "Generated by tools/make_full_configs.py.",
                     **copy.deepcopy(fullrun.FULL_DATA), "family": "ctc"},
        "data-t": {"_comment": "Box full-t's data config (T-0.6B alone): the full selection and extent "
                               "(kitsune.fullrun FULL_DATA), family aed and no pull_parakeet, so the box pulls "
                               "teacher_out (Cohere's labels) for every stem and no Parakeet labels "
                               "(kitsune.extent.pull_plan). Generated by tools/make_full_configs.py.",
                   **copy.deepcopy(fullrun.FULL_DATA), "family": "aed"},
        "data-p": {"_comment": "Box full-p's data config (P-0.3B with the augmentation recipe, the Whisper models and "
                               "the quantised readouts; P-0.05B postponed, DECISIONS H2): data-p01's content, family "
                               "ctc and no pull_parakeet (parakeet_out for every stem, teacher_out for the eval stems: "
                               "enough for the CTC store and the token eval store the readouts read). Generated by "
                               "tools/make_full_configs.py.",
                   **copy.deepcopy(fullrun.FULL_DATA), "family": "ctc"},
        "data-full": {"_comment": "Both label roots (pull_parakeet true) on the full selection and extent "
                                  "(kitsune.fullrun FULL_DATA): the retired 2x box 2's data config; no registry box "
                                  "uses it since box 2 became boxes full-t and full-p (DECISIONS G2), kept for "
                                  "scripts/make_selection.py's full mode and a box whose labels need both roots. "
                                  "Generated by tools/make_full_configs.py.",
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
    (<X> for <X>/configs/full); for a valid registry also readout_reserve_problems, continues_problems and, given the
    plan record, registry_drift (with the speed record, the HOURS_BOXES' hours too)."""
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
    problems = readout_reserve_problems(reg, root) + continues_problems(reg) + (
        registry_drift(reg, plan, speed) if plan is not None else [])
    return [f"{BOXES}: {p}" for p in problems]


def check(out_dir: Path = OUT_DIR) -> list[str]:
    """The differences between the files in out_dir and the generator's output (parsed JSON, so a CRLF checkout
    compares equal), CONTINUATIONS against the plan record (continuation_plan_problems), and every problem of the
    hand-written registry (registry_check: fullrun.registry_problems, the readouts' end reserve, the continues blocks,
    the numbers bound to the plan record and, with a SPEED_FILE record, the HOURS_BOXES' hours)."""
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
    return problems + continuation_plan_problems(plan) + speed_err + registry_check(out_dir, plan, speed)


def _plan_json(src, sel: str) -> dict:
    """A tools/full_plan.py JSON with its local selection path replaced by the repo path sel (PlanError when it was
    measured on another file name)."""
    rec = json.loads(Path(src).read_text(encoding="utf-8"))
    path = str((rec.get("selection") or {}).get("path") or "").replace("\\", "/")
    if path.rsplit("/", 1)[-1] != sel.rsplit("/", 1)[-1]:
        raise PlanError(f"{src}: measured on {path!r}, not a {sel.rsplit('/', 1)[-1]}")
    rec["selection"]["path"] = sel
    return rec


PLAN_COMMENT = ("tools/full_plan.py on the full-data selections (labels/full/selections/full_study/full.parquet and "
                "smoke.parquet) at the smoke's epochs (SMOKE_EPOCHS), recorded by tools/make_full_configs.py "
                "--import-plan, and on full.parquet at the boxes' epochs now (launch: FULL_RUNS, a continuation's), "
                "--import-launch-plan: the tool's JSON, the local selection path replaced by the repo path. "
                "make_full_configs reads the smoke configs' memory.probe_shapes (worst_shapes on full.parquet) and the "
                "registry's plan_total_steps / plan_hours and the deadline fault's bound (registry_numbers) from full "
                "and smoke, and the boxes' step counts (box_hours) from launch.")


def import_launch_plan(launch_json: Path, out_dir: Path = OUT_DIR) -> dict:
    """Record a tools/full_plan.py measurement of full.parquet at launch_epochs as PLAN_FILE's "launch" part (the
    "full" and "smoke" parts stay as recorded: the smoke configs' probe shapes and the registry's smoke numbers do not
    move). Refused (PlanError, nothing written) when the result fails plan_problems."""
    f = Path(out_dir) / PLAN_FILE
    try:
        plan = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PlanError(f"{f}: not readable ({type(e).__name__}: {e}); --import-plan first") from None
    plan["_comment"] = PLAN_COMMENT
    plan["launch"] = _plan_json(launch_json, fullrun.FULL_SELECTION)
    if problems := plan_problems(plan):
        raise PlanError("; ".join(problems))
    f.write_bytes(render(plan).encode("utf-8"))
    return plan


def import_plan(full_json: Path, smoke_json: Path, out_dir: Path = OUT_DIR) -> dict:
    """Record a tools/full_plan.py measurement of full.parquet and smoke.parquet as PLAN_FILE: each JSON as the tool
    wrote it, its local selection path replaced by the repo path (the file name must be full.parquet / smoke.parquet).
    Refused (PlanError, nothing written) when a record lacks a FULL_RUNS student or was measured with other values."""
    plan = {"_comment": PLAN_COMMENT}
    for which, src, sel in (("full", full_json, fullrun.FULL_SELECTION),
                            ("smoke", smoke_json, fullrun.SMOKE_SELECTION)):
        plan[which] = _plan_json(src, sel)
    try:  # the launch part stays (a rebuilt selection needs --import-launch-plan again: its sha256 then differs)
        plan["launch"] = json.loads((Path(out_dir) / PLAN_FILE).read_text(encoding="utf-8")).get("launch")
    except (OSError, ValueError, AttributeError):
        plan["launch"] = None
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
    ap.add_argument("--import-launch-plan", default=None, metavar="LAUNCH_JSON",
                    help="record tools/full_plan.py's JSON of full.parquet at the boxes' epochs (FULL_RUNS; a "
                         "continuation's) as " + PLAN_FILE + "'s launch part, then write the configs")
    ap.add_argument("--import-speed", action="store_true",
                    help="record the full boxes' measured speeds as " + SPEED_FILE + " (--smoke-verdict and/or "
                         "--box1-go; the part not given is kept) and print the hours boxes.json must carry")
    ap.add_argument("--smoke-verdict", default=None, metavar="JSON",
                    help="--import-speed: smoke A's smoke_verdict.json (full/box-full-smoke/smoke_verdict.json)")
    ap.add_argument("--box1-go", default=None, metavar="JSON", help="--import-speed: tools/box1_go.py --json's file")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if (args.smoke_verdict or args.box1_go) and not args.import_speed:
        ap.error("--smoke-verdict / --box1-go go with --import-speed")
    if sum(bool(x) for x in (args.import_speed, args.import_plan, args.import_launch_plan, args.check)) > 1:
        ap.error("--check, --import-plan, --import-launch-plan and --import-speed go one at a time")
    if args.import_speed:
        try:
            rec = import_speed(Path(args.smoke_verdict) if args.smoke_verdict else None,
                               Path(args.box1_go) if args.box1_go else None, out_dir)
            plan = load_plan(out_dir)
            reg = json.loads((out_dir / BOXES).read_text(encoding="utf-8"))
            hs = [box_hours(plan, rec, b, reserve_min=(reg["boxes"].get(b) or {}).get("deadline_reserve_min") or 60)
                  for b in HOURS_BOXES]
        except (SpeedError, PlanError, OSError, ValueError, KeyError) as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"wrote {out_dir / SPEED_FILE} ({'smoke A and box 1' if rec['box1'] else 'smoke A only: provisional'}"
              f"; r {hs[0]['r']:g}, o {hs[0]['o']:g}); {BOXES} (hand-written) must carry:")
        for h in hs:
            print(f"  box {h['box']}: " + ", ".join(f"{n} max_hours {v}" for n, v in h["items"].items())
                  + f"; est_hours {h['est_hours']:g}, max_hours {h['max_hours']} (setup {h['setup_h'][0]:g} / "
                  f"{h['setup_h'][1]:g} h, store {h['store_h'][0]:g} / {h['store_h'][1]:g} h, tail "
                  f"{h['tail_h'][0]:g} / {h['tail_h'][1]:g} h; the pessimistic slack {h['slack_h']:g} h)"
                  + "".join(f"; continuation {c['name']} {c['steps']} steps to T {c['total_steps']}, "
                            f"{c['dev_checks']:g} dev checks"
                            + (f", s/step x {c['step_factor']:g} (the recipe's concat)" if c["step_factor"] != 1
                               else "") for c in h["continuations"]))
        print("  P-0.1B from step 0 at its continuation's epochs projects to "
              f"{hs[0]['run_h']['p01']:g} h (box 1's 4 epochs: {box1_wall(rec)})")
        for name, c in CONTINUATIONS.items():  # the registry's continues blocks, as the queue passes their sets
            print(f"  continuation {name} on box {c['box']}: box {c['source_box']}'s run {c['run_id']} from "
                  f"full_step_{c['from_step']} to step {c['to_step']}"
                  + (" (its early-stop cooldown kept)" if c["keep_cooldown"] else "")
                  + f", readout runs/m4-{c['run_id']}-r{c['resets_before'] + 1}, baseline revision "
                    f"{c['revision'][:8]}; sets " + " ".join(fullrun.continue_sets(continues_block(name))))
        for b in dict.fromkeys(c["box"] for c in CONTINUATIONS.values()):
            print(f"  launch --box {b} (the registry carries its continuations)")
        todo = registry_check(out_dir, plan, rec)
        print(f"{BOXES}: " + ("carries the speed record's hours" if not todo else f"{len(todo)} change(s) by hand")
              + "".join(f"\n  {p}" for p in todo))
        return 0
    if args.import_launch_plan:
        try:
            plan = import_launch_plan(Path(args.import_launch_plan), out_dir)
            written, removed = write_all(out_dir)
        except (PlanError, OSError, ValueError) as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        print(f"wrote {out_dir / PLAN_FILE} (launch: "
              + ", ".join(f"{x} {launch_epochs(x)} epochs T {launch_total_steps(x, plan)}" for x in FULL_RUNS)
              + f") and {len(written)} configs" + (f"; removed {removed}" if removed else ""))
        try:
            speed = load_speed(out_dir)
        except SpeedError as e:
            speed = None
            print(f"no usable speed record ({e}): the boxes' hours are not checked")
        todo = registry_check(out_dir, plan, speed)
        print(f"{BOXES}: " + ("carries the launch record's hours" if not todo else f"{len(todo)} change(s) by hand "
                              f"(make_full_configs --import-speed prints them all)")
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
