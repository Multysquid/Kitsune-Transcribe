"""Distil the Kitsune student from the stored teacher outputs: the viability run (run sheet D19-D52, spec section 8).

One run, in order (every phase is an event in runs/<run_id>/events.jsonl):
  1. setup     config (JSON; every key overridable with --set a.b=value), RunLogger (env capture, TensorBoard,
               open-format metrics, HF sync), the student from 03 in fp32 master weights (bf16 autocast for the body;
               the LM head in fp32 outside autocast, as in the teacher pass), rel-pos patch, BatchNorm frozen (or
               training: bn.mode), the frozen decoder pos_emb, the aux-CTC head (loss.aux_ctc_weight), AdamW, the
               decoupled L2-SP anchor, train/eval stores and the step planner
  2. smoke     (fresh runs) a decode of seeded rows of every train source and eval set first (decode_preflight: a
               whole set that fails stops the run), then the memory probe (CUDA: forward+backward on the three worst
               micro-batches with the optimizer state's bytes reserved and, when steps accumulate several
               micro-batches, the gradients held; on OOM halve micro_audio_s down to memory.min_micro_audio_s, then
               per-layer gradient checkpointing; on Windows under a cap at the free VRAM, see cap_vram), then LogMel
               vs the HF extractor (a mean |diff| above LOGMEL_MEAN_DIFF_MAX, other masks or an error in the check
               stop the run), the training featuriser's dither on the device (train_dither_check),
               forward+backward on the longest padded micro-batch with finite gradients, padded rows == the same
               utterances run alone (mean KL over the shortest smoke.pad_utts; see padded_row_check), SDPA backends, a
               FLOP count for the MFU estimate and an HF upload round trip
  3. step 0    eval of the untrained student: teacher-forced on the eval sets and the train probe, greedy on the fixed
               subsets. It runs BEFORE any weight update (the spec lists the 100 smoke steps first, which would make
               "step 0" a 100-step model)
  4. train     WSD on the loop clock (evals and checkpoints included; see wsd_lr; every switch of LR phase is an
               `lr_phase` event), L2-SP after every optimizer step,
               gradients clipped (the pre-clip norm is logged). The first smoke.steps steps are the smoke run: at most
               smoke.max_dropped_frac of their rows undecodable, finite losses, the loss trend, and throughput -
               below smoke.min_audio_s_per_s audio-s/s the run exits with code 3 (ThroughputTooLow). ~20 of them run
               under torch.profiler (perf.profile_smoke, on under CUDA: smoke_profiler; runs/<run_id>/smoke/profile/,
               left out of the throughput check). A full state is written right after them, and before the cooldown
               starts.
               Early stop (early_stop, below) and the STOP file can end this phase before the budget is used up
  5. end       final weights + full state (uploaded in the background), final eval with greedy decode of the FULL eval
               sets, verdict (kitsune.evaluate.verdict with eval.verdict_version's trend definition, verdict_options,
               and eval.reference's model next to the teacher gate, reported only; "N/A" with its numbers under
               eval.gate false: the sanity/overfit runs), summary.json (uploads "pending") and a log sync, uploads
               awaited (bounded),
               summary.json with their results, final forced sync; exit 0

Evals in the loop (run_eval): every eval.every_min minutes of loop clock (eval.every_steps steps when set), or at the
end of every eval.every_epochs-th / eval.full_every_epochs-th epoch instead. An epoch ends where the planner's plan for
it ends (StepPlanner), on every clock. A full_every_epochs eval decodes the COMPLETE eval sets greedily, like the final
eval (when the loop ends at its step - it ran past T, max_steps or the STOP file at an epoch end, early stop - the end
phase reuses it as the final eval instead of decoding again: an `eval_final_reused` event); the others decode the fixed
greedy subset. Every one of them (step 0 and the final eval too) is a "full" eval: the complete eval sets
teacher-forced, the probe, one record in the history the verdict and the early stop read.
Mini evals (eval.mini, run_mini_eval): every mini.every_steps optimizer steps, except at step 0, at a step with a full
eval in the loop and when the loop is due to end at that step (the final eval runs there; on the wall clock judged from
the last mini's wall time, so a first mini, a checkpoint save at that step or a STOP file can still leave one at the
final eval's step), a small eval of its own on fixed seeded subsets -
mini.val_per_set utterances of each gate set (the whole eval subset of the overfit runs) and mini.train_utts of the
train probe - teacher-forced and, with mini.greedy, greedy; logged under eval/mini/, evals/step_<N>_mini/ and
`eval_mini` events, never in the verdict's history nor the early stop. After every eval, full or mini, the headline
numbers (kitsune.evaluate.headline: val CER vs the reference and vs the teacher pooled over the gate sets, train CER
from the greedy decode of train utterances, val/train teacher-forced KL and top-1) go to summary/full/<name> or
summary/mini/<name> (CERs as fractions, with <name>_pct copies in percent; a full eval also val_cer_utts, the gate
utterances its val CER pools: the step-0 eval decodes the fixed subset, a full_every_epochs eval the complete sets),
and one console line (" on U utts (subset|complete)" after the val CER of a full eval):
  [full eval] step N epoch E | val CER x.x% (vs teacher y.y%) | train CER vs teacher z.z% (vs ref w.w%) | val KL ...

The combined loss, the run's overall loss chart: one quantity, the training objective w_kl * KL + w_ce * CE (loss.*)
per target token - the same 17-bin KL and CE on the teacher tokens, masking and normalisation as the step's loss,
without the decoupled L2-SP term (kitsune.evaluate.combined_loss) - as three series, each with one scope throughout:
  combined_loss/train     every optimizer step: the step's objective (= loss/objective; augmented audio, train mode)
  combined_loss/val       every mini eval: its val subset's gate sets, teacher-forced and pooled token-weighted (the
                          monitor-only hold-outs never), plus a step-0 point on the same utterances (run_eval's
                          mini_val: a teacher-forced pass only, not a mini eval). Needs eval.mini
  combined_loss/val_full  every eval that decodes the complete eval sets (full_every_epochs, the final eval): the
                          complete gate sets. Never step 0, whose eval decodes the greedy subset
They are the first cards of 2_loss_accuracy (00_combined/) and one Custom Scalars chart overlays them
(kitsune.runlog.TB_LAYOUT); each eval's summary.json keeps the sums under combined_loss.

Early stop (early_stop.enabled; off in DEFAULTS, on in configs/viability.json and configs/overfit_*.json): after every
eval inside the loop (never the step-0 eval, never the final one) the metric - "probe_kl" (teacher-forced KL on the
train probe), "heldout_kl" (the gate sets' pooled held-out KL, as the verdict reads it) or "train_loss" (the mean
loss/objective = w_kl * KL + w_ce * CE of the optimizer steps since the previous eval; not loss/total, whose decoupled
L2-SP value is the distance from the initial weights and grows with every step, and not the headline train_loss, the
probe's teacher-forced KL) - improves when best - value > max(min_delta_abs, min_delta_rel * |best|); the first
in-loop eval sets the best. Once early_stop.min_evals in-loop evals are done, early_stop.patience evals in a row
without an improvement (reason "patience") or a value <= early_stop.floor (reason "floor") trigger early_stop.action,
once:
  stop       leave the loop right after that eval: the end phase as usual (final eval with the probe, verdict, summary,
             uploads; exit 0)
  cooldown   start the WSD cooldown now (1 - sqrt over schedule.cooldown_frac x the loop time, or steps, done so far,
             capped by what is left of the budget; on the epoch clock in steps, so the epoch plan ends early), then the
             end phase. Already in the cooldown: the run goes on to its scheduled end
Manual stop: create the file runs/<run_id>/STOP (on the vast box: touch
/workspace/Kitsune-Transcribe/runs/<run_id>/STOP). It is checked before every optimizer step, whatever
early_stop.enabled says: the step under way finishes (its eval and checkpoints included), then the "stop" path with
reason "stop_file". Every trigger is an `early_stop` event (metric, value, best, best_step, evals_since_best, reason,
action, at_step, epoch; `cooldown` with the new schedule) and summary.json's `early_stop_trigger` (null if none);
summary.json's `stopped_early` holds it only when it shortened the run ("stop", or a cooldown begun before the
scheduled one: null for a trigger with cooldown.already, which left the run to its scheduled end); the
scalars early_stop/{value,best,evals_since_best,triggered} follow every checked eval. The early-stop state is part of
the full state: a resumed run continues the patience count, and one that had already triggered "stop" goes straight
to the end phase (a triggered cooldown carries on to its end).

Exit codes (vast/supervise.py and scripts/supervise_distill.py key on them): 0 finished, 3 throughput too low, 1
anything else. Any exception is logged as an event with its traceback, a partial summary.json is written and the logs
are force-synced before the exception propagates.

Checkpoints under runs/<run_id>/checkpoints/:
  step_<N>/        bf16 weights via kitsune.student.save_student (processor + student_meta.json with a `trained`
                   block); loadable with kitsune.student.load_student. Every ckpt.weights_every_min, uploaded to
                   <repo>:runs/<run_id>/checkpoints/step_<N>/ when hf.output_repo is set. All are kept locally.
  full_step_<N>/   model.pt (fp32 state_dict), optimizer.pt (the host AdamW's under optim.offload "cpu", same
                   format), l2sp.pt (theta_0), trainer.pt (config, progress counters and clocks, eval history,
                   planner position, RNG states, logger state). Every ckpt.full_local_every_min, the newest
                   ckpt.keep_local are kept; uploaded at ckpt.upload_full_at ("pre_cooldown", "end"; one meant for
                   the Hub carries UPLOAD_MARK from its creation until its upload succeeds: a resume queues the
                   pre_cooldown one again while it is marked, as a crash may have cut its upload short, and one whose
                   upload failed is kept past keep_local for vast/finish.py). Written as <name>.tmp/, its
                   files fsynced, then renamed, so neither a process crash nor a host crash leaves a torn one.
`--resume <full_step_N dir | run dir>` restores all of it into the same run dir and continues the same schedule and
time budget; the config comes from the checkpoint, with this invocation's --set overrides applied on top (--config is
then ignored). optim.*, loss.* and memory.grad_ckpt among them go on top of the restored optimizer, L2-SP and memory
choice too; a new value for a key that shapes the step plan (RESUME_FIXED: seed, mix, sources, selection, subset.*,
batch.*) stops the resume with the key named (repeating the checkpoint's value is fine). A resume moves every
full_step_<M>/ and step_<M>/ newer than its state (an abandoned attempt's, or weights saved after the newest full state)
to checkpoints/abandoned-<UTC stamp>/ (set_aside_newer), so rotation, a later run-dir resume and finish.py see only the
resumed run's.

Time: in schedule.clock "wall" (the real run) the budget T = train_hours of loop wall-clock from the first training
step, evals and checkpoints included; the step-0 eval and the final eval are outside it. The clock continues across a
resume (time between a crash and the resume is not counted). On the vast box T is also clipped at every loop start so
the end phase finishes before the watchdog's fixed deadline ($KITSUNE_DEADLINE or $KITSUNE_STATE/deadline; see
fit_budget). schedule.clock "steps" makes T = schedule.max_steps optimizer steps, and eval.every_steps /
ckpt.*_every_steps replace the minute cadences when set: deterministic schedules for tests and debugging.
schedule.clock "epochs" makes T = the optimizer steps of schedule.epochs full passes over the train set (plan_epochs),
with the warmup capped at ceil(10 %) of them; eval.every_epochs runs the eval at every N-th epoch end instead of the
minute/step cadence (the last epoch's is the end phase's final eval; on the steps and wall clocks a complete-set
eval.full_every_epochs eval at the loop's last step is reused as the final eval).

Overfit sanity runs (configs/overfit_*.json, scripts/run_overfit_tests.cmd): subset.train_audio_s / eval_audio_s train
and evaluate on seeded subsets of about that many seconds (audio_subset; logged as `subset` events with every id),
eval.probe_is_train makes the whole train subset the teacher-forced probe and eval.probe_greedy_audio_s adds a greedy
decode of part of it, so memorisation (probe KL and CER vs teacher -> 0) shows next to the held-out eval;
specaug.enabled false trains un-augmented. optim.offload "cpu" keeps AdamW and fp32 master weights in host memory
(CpuOffloadAdamW), for a GPU that holds the weights and gradients but not the optimizer state (the 8 GB laptop).

The size study (every default below is the trainer as it was before them; the study runs on the steps clock):
  bn.mode           "frozen" (default; the pruned students keep the teacher's BN statistics) or "train" (the students
                    trained from scratch: BN trains in the training forward and updates its running stats, which the
                    evals and greedy decodes use through eval mode; every train_mode() call and BN check follows the
                    mode (kitsune.patches.train_mode / assert_bn_mode), bn/max_abs_drift is reported instead of
                    asserted 0, gradient checkpointing is never turned on - the memory probe halves the micro-batch
                    down to memory.min_micro_audio_s instead - and a skipped step (non-finite, OOM) puts the running
                    stats back as they were before it). bn.momentum: null = the modules' own
  loss.aux_ctc_weight  > 0: a training-only aux-CTC head (setup_aux_ctc) on the encoder output, d_enc -> V + 1 with the
                    blank V, on the teacher's greedy ids (no prompt, no EOS), its summed CTC loss divided by the step's
                    target tokens and weighted into the gradient; loss/aux_ctc per token, loss/total includes it,
                    loss/objective (and the combined-loss curves, the early stop) does not. In the optimizer and the
                    full state (aux_ctc.pt), never in the exported weights; the memory probe includes its logits
  optim.weight_decay   > 0: AdamW's decoupled decay on the parameters of 2+ dimensions only, never on a bias (the 2-dim
                    attention biases bias_u / bias_v included), a second group without it (decays, setup_optim; a run
                    started at 0 keeps its single group)
  schedule          on the steps clock the warm-up is not capped, so validate() rejects warmup_steps >= (1 -
                    cooldown_frac) x max_steps: a warm-up still running at the cooldown never reaches the peak LR
  eval.full_at_fracs   a complete eval (as full_every_epochs: teacher-forced and greedy on the complete eval sets, the
                    probe) right after step round(f x max_steps); the final eval always runs. Set eval.every_min null
                    for these alone (every_min counts loop minutes on every clock); mini evals keep their cadence
  ckpt.full_at_fracs / weights_at_fracs   a full state (kept locally past keep_local) / exported weights at step
                    round(f x max_steps); ckpt.upload_full_at "frac:<f>" uploads the full state of fraction f
  branch.parent     a local run dir: this launch is its T/2 branch (branch_start_state). It continues the parent's
                    full state at round(resume_frac x M) (M = the parent's max_steps; its ckpt.full_at_fracs keeps
                    it) - weights, optimizer, aux head, counters, RNG and the planner's position, so the data order goes
                    on - under a config equal to the parent's but for BRANCH_FREE, to round(end_frac x M) with the
                    cooldown from the resume step: warm-up and stable LR are the parent's, the cooldown the last 20 %
                    of a T/2 budget, the WSD schedule of a run with max_steps round(end_frac x M) (exactly for M a
                    multiple of 10 at the default fractions; else the rounding moves the cooldown's start by up to 0.8
                    of a step). Then one complete final eval, and no other: the branch ignores eval.full_at_fracs and
                    ckpt.*_at_fracs (a config copied from the parent's carries them; the `branch` event lists them
                    under fracs_ignored). A new run dir <parent run_name>-half-<stamp> (unless run_name differs from
                    the parent's); a `branch` event and summary.json's `branch` carry parent_run_id, resume_step,
                    end_step and t_c; a crash resumes it like any run
  specaug.seed      null = the run's seed: every micro-batch's masks from (seed, step, micro-batch index)
                    (specaug_seed), so a resumed or branched run augments exactly as its parent would have
  lr_probe.enabled  an LR probe: metrics only - no step-0, in-loop or mini evals, no greedy decode, no weights, no
                    checkpoint uploads (local full states for a resume only) - and at the end the training objective
                    teacher-forced on the complete gate sets (lr_probe_eval): an `lr_probe_result` event and
                    summary.json's lr_probe {objective, per_set, lr, max_steps, family}
  family            "aed" (default: the Cohere-family students above) or "ctc": a Parakeet-family student (a
                    transformers ParakeetForCTC dir from scripts/03c_build_ctc_student.py) distilled from the Parakeet
                    teacher's per-frame CTC targets under parakeet_root (STUDY.md 2.1, decisions 15, 22-24). The same run
                    - schedule, clocks, checkpoints, resume, branch, LR probe, evals, early stop, uploads - with:
                      model      kitsune.ctc_student.load_ctc_student (sdpa; no decoder, no pos_emb), the rel-pos patch,
                                 BatchNorm in bn.mode (the pruned P students keep the teacher's stats, frozen)
                      features   kitsune.ctc_student.ctc_features' LogMel: 80 mel, pre-emphasis 0.97, per-feature
                                 normalisation, NO dither (the Parakeet extractor has none); SpecAugment masks only
                      data       frame stores (kitsune.trainset.build_frame_stores; cache dirs ctc_train / ctc_eval /
                                 ctc_*_sub*: store_dir) of the selection's rows with their parakeet_out targets, built with
                                 the FRAME PREFLIGHT: every row's decoded audio must give its stored n_frames (decision 15:
                                 a mismatch - or a row that does not decode - is dropped and counted, more than 0.1 % of
                                 the train rows or any eval row fails the run; a `frame_preflight` event per store, a
                                 reused cache's too; the decode runs in processes by shard, kitsune.ctc_preflight). The
                                 planner packs
                                 by padded audio (no decoder lengths). The run reads no teacher_out; scripts/05_evaluate.py
                                 scores a CTC student on the token eval store (build_eval_store's default builds both)
                      loss       kitsune.ctc_kd: (w_kl x KL on every frame + loss.w_ctc x CTC on the teacher's greedy CTC
                                 path) / N_u, the CTC target tokens of the whole step (T = 1, no T^2). loss/objective,
                                 loss/kl (= kl_dense + kl_blank), loss/kl_dense, loss/kl_blank and loss/ctc are per
                                 target token; ctc/argmax_agree, ctc/argmax_blank (the blank-collapse watch),
                                 ctc/teacher_blank, ctc/frac_dense and ctc/kl_per_frame per valid frame
                      evals      kitsune.ctc_eval: one pass gives the teacher-forced frame metrics per set (the eval
                                 summaries' kl = KL per target token, ce = CTC per target token, top1 = frame argmax
                                 agreement, plus kl_dense, kl_blank, argmax_blank, teacher_blank, ...) and the greedy CTC
                                 decode (kitsune.ctc_student.greedy_ctc_ids), scored vs the reference and vs the teacher's
                                 stored ctc_hyp. The teacher baselines (at start) and the verdict's teacher are Parakeet
                                 CTC's (ctc_hyp) on the same ids, checked against kitsune.evaluate's registered numbers
                                 (study/PREREG.json; K6 while pending); the verdict is kitsune.evaluate.verdict(...,
                                 family="ctc"); the combined loss and the LR probe's objective are
                                 w_kl x KL + w_ctc x CTC per target token
                      smoke      LogMel vs the Parakeet extractor, the longest micro-batch's forward+backward, padded rows
                                 vs alone on the frame log-probs, the FLOP count of the CTC model; the memory probe's
                                 worst case is the micro-batch with the most padded frames (frames x 3073 fp32 logits)
                      export     step_<N>/ is a ParakeetForCTC dir (bf16 weights, fp32 BN stats) with the Parakeet
                                 processor and tokenizer, student_meta.json (its `trained` block) and the CC-BY-4.0
                                 MODEL_CARD.md (kitsune.ctc_student.save_ctc_student); load_ctc_student loads it
                    loss.w_ctc: the CTC term's weight (0.8, decision 23; the CTC family only; loss.w_kl is shared). The
                    aux-CTC head (loss.aux_ctc_weight) is an AED option: a CTC student refuses it
RESUME_FIXED also holds bn.mode, loss.aux_ctc_weight, specaug.seed and family. summary.json carries family,
selection_sha256 (the selection file's sha256, as its store build hashed it) and started_utc (the run's first start;
a T/2 branch's own).

Test hook: the environment variable KITSUNE_CRASH_AT_STEP=<n> raises a RuntimeError just before step n (exercises
the crash path and --resume; it lives in the environment so a resumed run does not inherit it from the config).

Heavy imports (transformers, the eval/student/logging modules) happen inside main(): DataLoader workers are spawned
and re-import this file as __mp_main__, and should only pay for torch.

Usage:
  python scripts/04_distill.py --config configs/viability.json --set hf.output_repo=<user>/<repo>
  python scripts/04_distill.py --config configs/smoke_laptop.json
  python scripts/04_distill.py --config configs/overfit_10s.json
  python scripts/04_distill.py --config configs/viability.json --resume runs/<run_id>/checkpoints/full_step_<N>

Full-data runs: dev slice, early stopping v2, COOLDOWN, resume reset, deadline cooldown, 4c (every default below is
the trainer as it was before them; configs/full/*.json turn them on):
  eval.dev          the dev slice: the selection's kept split "dev" rows (whole train shards held out by
                    make_selection.py full mode; kitsune.fullrun DEV_SPLIT; their audio and labels live in their train
                    shards, kitsune.trainset), a seeded per_source of them per train source (fullrun.dev_pick; seed
                    null: the run's) in a store of their own (build_dev_store: cache dir dev_p<n>_s<seed>_<sha8>, a
                    frame store with its frame preflight for family ctc; the box's store step builds it too). setup_data
                    writes runs/<run_id>/dev_ids.json and a `dev_store` event (n_in_train must be 0; ids_sha256 hashes
                    the sorted ids and names the store, ids_sha256_selection hashes them in selection order and equals
                    the selection sidecar's dev.scored_default.ids_sha256 at its per_source and seed), and a resume on
                    other dev rows stops (st["dev"]; per_source and seed are RESUME_FIXED). run_dev_eval (every
                    every_steps steps, or at each every_epochs fraction of an epoch; a step-0 one under at_start, a
                    final one under at_end): teacher-forced on every dev row, greedy too with greedy (CTC: the argmax
                    of the same pass, free; AED: a decode), pooled over the sources: dev_objective (the family's
                    training objective), dev_kl and dev_ce (CTC: the CTC loss) per target token; one record in
                    st["dev_history"] (summary.json's dev_history), evals/step_<N>_dev/, eval/dev/* scalars and an
                    `eval_dev` event, never an `eval` event nor the verdict's history. With early stopping off a run
                    with dev evals trains exactly as one without
  early stopping v2 metrics dev_ce / dev_kl / dev_objective (checked after every dev eval instead of after the full
                    evals; need eval.dev); smooth k > 1: the rule reads the mean of the last k values and counts
                    nothing before it has k (early_stop/raw next to early_stop/value; the `early_stop` event gains raw
                    and smooth); allow_test_sets false refuses heldout_kl (it pools the gate sets the runs are scored
                    on). stopped_early now also holds a dev trigger and the COOLDOWN file's
  COOLDOWN          runs/<run_id>/COOLDOWN, checked before every step with the STOP file, acts once whatever
                    early_stop.enabled says: the early stop's cooldown action with reason "cooldown_file" (the WSD
                    cooldown starts now over cooldown_frac x the progress so far); inside a cooldown it only logs a
                    `cooldown_file` event (ignored). The pre_cooldown state is then saved and uploaded as for any
                    cooldown
  resume reset      --resume <state> --set schedule.resume_reset=true [--set schedule.epochs=<E>]: continue past an
                    early stop or a finished cooldown, normally from the uploaded pre_cooldown state (a state inside
                    its cooldown puts the LR back at its peak). One-shot: resume_reset_start clears total_steps,
                    pre_cooldown_done / _full, the early-stop state (dev_history stays) and 4d's record, renames
                    COOLDOWN to COOLDOWN.consumed-<stamp> (and deletes the runs repo's copy, which a new-host resume
                    would pull back) and sets the flag false before any save; plan_epochs plans the new
                    schedule.epochs; resume_reset_finish refuses a new T at or before the step and records the reset
                    (st["schedule_resets"], a `resume_reset` event, summary.json's resume_resets). Refused on a fresh
                    start, on a T/2 branch and with a STOP file; a resume that sets another schedule.epochs without
                    the flag is refused too when it would change nothing (plan_epochs keeps the state's total_steps;
                    a switch to the epoch clock from a state without a plan still plans it)
  deadline cooldown schedule.deadline_cooldown (clock epochs or steps; the wall clock has fit_budget): at loop start
  (4d)              and every deadline_check_steps steps after the smoke phase, fit_epochs_deadline projects the end
                    phase against $KITSUNE_DEADLINE from the loop-clock s/step over >= deadline_window_steps steps of
                    this launch (deadline_rate) and schedules, starts or compresses the WSD cooldown in its own record
                    st["deadline_cooldown"] (Run.progress takes the smaller T of it and an early stop's, cooldown_start
                    the earlier t_c), or clears a record whose cooldown has not begun once the run fits again; no step
                    left ends the loop (`no_time_left`). Exit 0, summary.json's end_reason "deadline"
  4c                memory.probe_extended adds the micro-batch with the most rows (and, family ctc, the most CTC
                    targets and the largest CTC lattice) to the memory probe; memory.probe_shapes adds synthetic
                    micro-batches of the train rows nearest given durations, held within the micro-batch as the
                    planner packs one (synth_micro_batch; the full data's worst shapes on a smoke run; a shape larger
                    than the micro-batch is skipped). The memory_probe event gains extended, headroom_gb and shapes
  summary.json      end_reason (schedule | early_stop | cooldown_file | stop_file | deadline) and resume_resets always;
                    dev_history, deadline_cooldown and schedule_resets when there are any
  also              the epoch clock refuses a schedule.max_steps; eval.dev refuses lr_probe.enabled;
                    selection_recipe.full_study is a full selection's recipe (validate_box)

Full-data runs: timed states, heartbeats, 4e, fix 7 (the full-run build contract 4.3; every key is off by default):
  ckpt.upload_full_every_min  (4a) every N minutes of loop clock a full state is saved (reason "timed", TIMED_REASON;
                    after the step's periodic saves, and the periodic full cadence restarts from it) and uploaded to
                    hf.scratch_repo, the private scratch model repo the owner creates (never created here), by
                    ScratchUploader through kitsune.scratch: one commit adds the state under runs/<run_id>/checkpoints/
                    full_step_<N>/ with its pointer runs/<run_id>/timed_state.json and deletes the run's previous one,
                    then the repo's history is squashed. The dir carries SCRATCH_MARK (never UPLOAD_MARK) until its
                    upload ends; rotate_full keeps it meanwhile, and of the marked dirs only the newest. A timed state
                    at the step of a periodic one rewrites its trainer.pt/.json (so it records its own reason and
                    cadence), and so does a pre_cooldown state at the step of a periodic or timed one in a run with
                    timed states (the full-run queue's reset picks the runs repo's state by that reason). One upload at
                    a time: a state due while one runs waits for it (no local save; one timed_state_skipped per due
                    window it misses, with the running upload's running_s, so a hung one shows). The uploads are never
                    awaited - the end phase and the failure path cancel the queued ones - except by a trainer-only
                    rewrite of the same step's dir (end, kept fraction, pre_cooldown), for at most UPLOAD_WAIT_S. A
                    resume whose checkpoints/ still hold SCRATCH_MARK (a crash cut an upload short) sends the state it
                    resumes from at once; one without a scratch uploader drops such marks. After a timed upload went
                    up, and after the pre_cooldown state is queued for the runs repo, the loop forces a log sync
                    (sync_due), so the Hub's logs are as fresh as the newest state. With hf.output_repo set it needs
                    hf.scratch_repo (build refuses: the full-run queue passes it); without an output repo timed states
                    are skipped (one timed_state_skipped). summary.json: timed_states {repo, count, last}
  resume on a new host  (4b) trainer.pt/.json name the saving host; the `resume` event has host, prev_host and
                    new_host. With hf.output_repo set, a resume of a state that records its host refuses a run dir
                    without events.jsonl (a new host must pull the run's logs first: kitsune.full_queue resume-pull; the
                    logger's first sync would otherwise overwrite the Hub's). A planner state that does not fit the
                    rebuilt train store raises ResumeMismatch; resume_check(run_dir) tells the same on the CPU first
  heartbeats        with $KITSUNE_HEARTBEAT set (the full-run queue sets it per item; kitsune.heartbeat): every logged
                    step beats, every eval batch beats (setup_processing wraps R.feat_eval in heartbeat.Beating), and so
                    do the end phase's waits and the failure path, under a bounded heartbeat.beating.
                    setup_processing also logs a `threads` event (torch's pools and the *_NUM_THREADS values)
  log.scalars_parquet  (4e) "close": metrics/scalars.parquet is written at the logger's close only (its final sync, the
                    failure path's too), not at every sync; metrics/scalars.jsonl is unchanged
  log.full_scalars_every_steps  (fix 7) N > 1: after the smoke phase only every N-th step's row holds the full set
                    (system stats, per source, buckets, token diagnostics), the others the CORE_STEP_TAGS keys;
                    steps.parquet keeps a row per step. mem/step_peak_reserved_gb (CUDA) is logged every step, and
                    summary.json's throughput has data_wait_frac (the loader's share of the steps' time)

Train-data augmentation (augment.*, family "ctc"; off by default, and off it trains exactly as before). The owner's
review of P-0.1B inside the app found two faults of the data, not of the model size: every train row is one whole
utterance ending in a sentence mark, so the student puts 。 wherever its input ends (and none at a real sentence end
inside a long chunk), and it never saw two utterances or two voices in one input (ReazonSpeech 11.9 % CER per
utterance, 33.7 % with 4 joined; crosstalk far below Whisper). The train loader's micro-batches (setup_augment: a copy
of R.ds, R.train_ds, that make_loader pickles into its workers) are therefore joined (concat_p: all rows of a
micro-batch into groups of k, the teacher frame targets concatenated, so real marks sit inside long rows), cut inside
a sentence (truncate_p: at a frame whose first removed token is a word - never a mark or comma, the student
tokenizer's punct_token_ids -, half of them in a pause; audio and targets at the same frame - the targets came from the
whole utterance, so the frames before the cut carry no final mark, and the kept part is never a complete sentence
stripped of its mark), padded with quiet (truncate_pad_p per cut row: the sentence still runs, no mark; end_pad_p per
whole row: the mark kept before the quiet; pad_frames 80 ms frames, never past the micro-batch's longest row) and
mixed with another row's speech 5-20 dB down (mix_p; the targets the
clean row's), in that order (kitsune.trainset's "augmentation" section). Memory: a joined micro-batch has rows / k rows
of at most k x its longest row's frames, so its padded frames never exceed the planned micro-batch's, and setup_augment
clamps concat_max_s to the longest train utterance (the `augment` event), so no joined row is longer than the one the
memory probe's longest micro-batch ran; the probe and the smoke checks read the plain R.ds, every eval (the dev slice
and the train probe included) a plain dataset of its own. Deterministic:
each micro-batch's augmentation is a function of (augment.seed or the run's seed, its index list), so a crash-resume -
which reads the run's own config - or a T/2 branch augments every step as the original. A resume may change augment.*
(it does not shape the step plan; its `resume` event lists the change), a branch may not (BRANCH_FREE). Logged per step:
aug/concat_frac (the share of the step's utterances in a joined row), aug/truncated_frac, aug/mixed_frac,
aug/cut_padded_frac and aug/end_padded_frac (shares of its rows); a joined row's train_utts record has the pieces' ids
joined by "+", the first piece's source, their summed duration (a cut row: the seconds kept; plus a pad's). The smoke's
dropped-audio share counts utterances, not joined rows (batch_utts)
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from kitsune import trainset  # noqa: E402
from kitsune.features import LogMel, SpecAugment  # noqa: E402
from kitsune.kd import L2SP, aux_ctc_loss, aux_ctc_targets, kd_losses, kd_objective, module_key  # noqa: E402
from kitsune.patches import (  # noqa: E402  (assert_bn_frozen: scripts/05_evaluate.py's check, through this module)
    BN_MODES,
    assert_bn_frozen,
    assert_bn_mode,
    bn_running_stats,
    patch_relpos_once_per_batch,
    train_mode,
)
from kitsune.runlog import _replace as _replace_file  # noqa: E402  (atomic file replace with the Windows retry)
from kitsune.store import fsync_path  # noqa: E402

EXIT_OK, EXIT_THROUGHPUT, EXIT_FAIL = 0, 3, 1
# the end phase's bound on each wait for checkpoint uploads, so an upload that never returns cannot keep the trainer
# (and the GPU) running. On a hung Hub connection (no HTTP timeout) the waits after the final eval add up to 22 min:
# this, END_SYNC_JOIN_S and RunLogger's close_join_s (600 s); 32 min when save_full's end rewrite first waits this long
# for a running upload of its own dir. schedule.end_reserve_min (30) must also hold finish.py's sync and verification
# and the watchdog's 10 min sync lead (vast/watchdog.sh), so such a run can reach the deadline: the watchdog then stops
# the box (disk kept) instead of finish.py destroying it
UPLOAD_WAIT_S = 600
# the end phase's wait for a running log sync before its own one (the final eval and verdict, ahead of the uploads)
END_SYNC_JOIN_S = 120
# the failure path's wait for them, after the logs are closed: a crashed trainer should get to the supervisor's resume.
# What it cuts off goes up anyway (Uploader.abandon)
FAILED_UPLOAD_WAIT_S = 120
SR = 16000
# shm_cap's warning: below this many loader workers the train mix (mostly MP3/OGG, ~1k audio-s/s per worker on a slow
# core; kitsune.trainset.default_num_workers) cannot feed an A100 (~2.5k audio-s/s)
SHM_FEW_WORKERS, DECODE_AUDIO_S_PER_WORKER = 4, 1000
FULL_RE = re.compile(r"^full_step_(\d+)$")
WEIGHTS_RE = re.compile(r"^step_(\d+)$")
# an empty file in a full state meant for the Hub (ckpt.upload_full_at) until its upload succeeds: rotate_full keeps
# the dir meanwhile and vast/finish.py uploads and verifies it before a destroy (the same name there); never uploaded
UPLOAD_MARK = ".upload_pending"
# the full runs' timed states (ckpt.upload_full_every_min; kitsune.fullrun, which this file does not import at load):
# the reason a timed full state is saved with, and the empty file it carries while its upload to the scratch repo is
# pending (ScratchUploader; never UPLOAD_MARK, which vast/finish.py treats as a state the runs repo must hold)
TIMED_REASON = "timed"
SCRATCH_MARK = ".scratch_pending"
# log.full_scalars_every_steps (fix 7): the step-row keys every step keeps; the rest only every N-th step
CORE_STEP_TAGS = ("loss/", "combined_loss/", "opt/lr", "opt/grad_norm", "opt/clip_coef", "time/", "perf/", "data/",
                  "sched/", "mem/")
TERMS = ("kl", "ce", "top1_match", "student_entropy_coarse", "teacher_entropy_coarse", "student_tail", "teacher_tail",
         "teacher_p1")
TERM_TAGS = dict(kl="loss/kl", ce="loss/ce", top1_match="tok/top1", student_entropy_coarse="tok/entropy_student",
                 teacher_entropy_coarse="tok/entropy_teacher", student_tail="tok/tail_student",
                 teacher_tail="tok/tail_teacher", teacher_p1="tok/teacher_p1")
BUCKETS = (("p1_gt_0.99", lambda p1: p1 > 0.99), ("p1_lt_0.9", lambda p1: p1 < 0.9))

# Defaults = the viability run (configs/viability.json spells out the same values, and turns early_stop on). Paths are
# relative to the repo root.
DEFAULTS = {
    # family: "aed" (a Cohere-family student, CohereAsr AED) or "ctc" (a Parakeet-family student, ParakeetForCTC, on
    # the frame targets under parakeet_root; the module docstring's "family" entry). A CTC run requires parakeet_root
    # (below: the local dir with parakeet_out's <src>/<stem>.{npz,jsonl} and meta.json), the one run that reads it
    "family": "aed",
    "run_name": "viability-b20x2560", "student": "students/b20x2560-d4", "data_root": "data",
    "teacher_root": "teacher_out", "second_root": "second_out", "selection": "selection/viability.parquet",
    "cache_dir": "cache", "runs_root": "runs",
    "sources": ["reazon_small", "emilia_yodas", "galgame"],
    "eval_sets": ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"], "mix": "natural",
    # how the selection is built from sources / eval_sets: make_selection.py's --agree-max, --agree-max-source
    # (SOURCE=A, as that flag takes it: a list, so a config replaces it whole), --filter-eval-sets and
    # --partial-second-opinion (sources that train on their judged shards only, by decision: galgame for the viability
    # run, whose 02b pass the laptop GPU could not finish). Read by scripts/make_selection.py --config and checked by
    # vast/launch.py against the selection's own record; not by the trainer. study: the size study's selection rules
    # (kitsune.prereg.STUDY_SELECTION, which study/data.json carries verbatim; null: not a study selection); a dict of
    # exactly those keys (validate_box). full_study: a full-data selection's recipe (kitsune.fullrun.FULL_STUDY: the
    # study's rules plus the dev slice; make_selection.py full mode); null, or a block full_recipe_problems accepts,
    # never together with study (validate_box)
    "selection_recipe": {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "eval_emilia=0.2"],
                         "filter_eval_sets": ["eval_emilia", "galgame"], "partial_second_opinion": ["galgame"],
                         "study": None, "full_study": None},
    # the label extent (kitsune/extent.py; make_selection.py, vast/launch.py and bootstrap.sh read it), the Parakeet
    # soft-target root and the label box's settings (vast/label.py): configs/full.json sets them, the trainer never
    # reads them. None, so a config's object replaces the default whole (_merge does not recurse into None).
    # pull_parakeet: the box also pulls parakeet_root's labels for this run (kitsune.extent.pull_plan; the size study's
    # box pulls both teachers for every run, study/data.json); needs parakeet_root; the trainer does not read it
    "extent": None, "parakeet_root": None, "label": None, "pull_parakeet": False,
    # smoke runs: seeded id subsets, small caches. *_audio_s: seeded subsets of about that many seconds of audio
    # (audio_subset; the overfit runs), the eval one pooled over eval_sets and decoded greedily in full at every eval
    "subset": {"train_utts": None, "eval_utts_per_set": None, "train_audio_s": None, "eval_audio_s": None},
    "device": "auto", "autocast": "bfloat16",  # autocast: "bfloat16" or "none" (fp32; the CPU tests)
    # aux_ctc_weight: > 0 adds the training-only aux-CTC head (d_enc -> V + 1, blank V) on the encoder output, trained
    # on the teacher's greedy ids with this weight (the from-scratch students; setup_aux_ctc, kitsune.kd.aux_ctc_loss);
    # 0 = no head. w_ctc: family "ctc" only, the CTC term's weight next to w_kl (decision 23; w_ce is the AED one's)
    "loss": {"w_kl": 1.0, "w_ce": 0.8, "l2sp_lambda": 0.05, "aux_ctc_weight": 0.0, "w_ctc": 0.8},
    # seed: the SpecAugment masks' seed (null: the run's seed); each micro-batch's masks come from a generator seeded
    # with (seed, step, micro-batch index), so a resumed or branched run augments exactly as its parent would have
    "specaug": {"enabled": True, "freq_masks": 2, "freq_width": 27, "time_masks_min": 2, "time_masks_max": 5,
                "time_width": 0.05, "seed": None},
    # the CTC family's train-data augmentation (kitsune.trainset's "augmentation" section; the module docstring's
    # "Train-data augmentation"): the TRAIN loader's micro-batches only - never an eval, dev, probe, smoke check or the
    # memory probe. enabled false (the default) trains exactly as before. seed: null = the run's seed; every micro-batch
    # draws from (seed, its index list) alone. truncate_p: per row, cut it at a frame >= truncate_min_frac of its
    # frames and >= truncate_min_s seconds whose first removed token is a word (never a sentence mark or comma: the
    # student tokenizer's, punct_token_ids), so the kept part ends inside a sentence and its targets carry no final mark;
    # truncate_pause_p of the cuts land inside a pause (>= 320 ms of teacher blank), as a voice-detector chunk ends.
    # concat_p: per
    # micro-batch, join all its rows into groups of k (a divisor of the row count, 2..concat_max_n, with k x the longest
    # row <= concat_max_s; the trainer clamps concat_max_s to the longest train utterance, an `augment` event). mix_p:
    # per row, add a segment of another row of the micro-batch at an SNR drawn from mix_snr_db ([low, high] dB below
    # the row, low >= 0), the targets the clean row's. truncate_pad_p / end_pad_p: per cut row (no mark: the sentence
    # still runs) / per whole row (its mark kept, before the quiet), append [low, high] pad_frames 80 ms frames of quiet
    # (the row's room tone or near-silent noise) with blank targets - the app's padded chunk ends -, never past the
    # micro-batch's longest row. A resume may change any of them (the step plan does not depend on them; the change is
    # in its `resume` event's overrides), a T/2 branch none (BRANCH_FREE)
    "augment": {"enabled": False, "seed": None, "truncate_p": 0.0, "truncate_min_frac": 0.3, "truncate_min_s": 1.0,
                "truncate_pause_p": 0.5, "concat_p": 0.0, "concat_max_s": 28.0, "concat_max_n": 4, "mix_p": 0.0,
                "mix_snr_db": [5.0, 20.0], "truncate_pad_p": 0.0, "end_pad_p": 0.0, "pad_frames": [1, 5]},
    # BatchNorm: mode "frozen" (eval mode, running stats fixed, affine trainable: the pruned students, whose BN holds
    # the teacher's statistics) or "train" (a student trained from scratch: BN trains, its running stats update, eval
    # and greedy decoding use them; gradient checkpointing is never turned on). momentum: null = the modules' own
    "bn": {"mode": "frozen", "momentum": None},
    # offload "cpu": AdamW and fp32 master weights in host memory (CpuOffloadAdamW), for a GPU that holds the weights
    # and gradients but not the AdamW state; offload_fused: torch's fused CPU AdamW kernel there instead of foreach.
    # `fused` is the on-device (CUDA) optimizer's flag. weight_decay: AdamW's decoupled decay, on the parameters of 2 or
    # more dimensions only (a second param group without decay for biases - bias_u / bias_v too -, norm and BN affine;
    # decays, setup_optim)
    "optim": {"lr": 1e-4, "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0, "clip": 1.0, "fused": True,
              "max_nonfinite_skips": 3, "offload": "none", "offload_fused": False},
    # clock "steps": warmup_steps must end before the cooldown starts (validate), and is not capped. resume_reset: a
    # resume's one-shot flag (--resume --set schedule.resume_reset=true): the run continues past its early stop or
    # cooldown with a fresh schedule, e.g. more schedule.epochs (resume_reset_start; the module docstring's full-data
    # section). deadline_cooldown (clock epochs or steps): schedule, start or compress the WSD cooldown so the end
    # phase finishes before $KITSUNE_DEADLINE (fit_epochs_deadline), checked at loop start and every
    # deadline_check_steps steps from the loop-clock s/step over at least deadline_window_steps steps of this launch
    "schedule": {"warmup_steps": 300, "cooldown_frac": 0.2, "train_hours": 4.0, "clock": "wall", "max_steps": None,
                 "end_reserve_min": 30, "epochs": None, "resume_reset": False, "deadline_cooldown": False,
                 "deadline_check_steps": 100, "deadline_window_steps": 1000},
    "batch": {"step_audio_s": 1500, "micro_audio_s": 400, "pool_micro": 50, "max_dec_len": 200},
    # probe_extended: the memory probe also runs the micro-batch with the most rows (and, family ctc, the most CTC
    # targets and the largest CTC lattice); probe_shapes: null or [{"name", "durations"}], synthetic micro-batches of
    # the train rows nearest those durations (the full data's worst shapes on a smoke run; probe_passes)
    "memory": {"grad_ckpt": "auto", "probe_longest_bucket": True, "min_micro_audio_s": 100, "max_oom_skips": 3,
               "probe_extended": False, "probe_shapes": None},
    # loader_timeout_s: seconds the loop waits for a worker micro-batch before it raises (a crash the supervisor can
    # resume) instead of hanging until the watchdog; 0 = wait forever (trainset.make_loader). profile_smoke: a
    # torch.profiler record of ~20 smoke steps into runs/<run_id>/smoke/profile/ (kitsune.profiling, smoke_profiler);
    # "auto" = on under CUDA, off on CPU; it does not change what the steps compute, but it costs a few minutes of
    # host time (kitsune.profiling: cost), which on the wall clock comes out of the loop's training time
    "perf": {"relpos_patch": True, "compile": False, "num_workers": "auto", "prefetch": 4, "tf32": True,
             "peak_tflops": 312.0, "train_exact_dither": False, "loader_timeout_s": 600, "profile_smoke": "auto"},
    # every_epochs: eval at the end of every N-th epoch instead of every_min / every_steps. full_every_epochs: the
    # same, but each of those evals decodes the COMPLETE eval sets greedily (as the final eval does) instead of the
    # greedy subset; set one of the two. probe_is_train: the probe is the whole train set (a small subset) rather than
    # its in_probe rows. probe_greedy_audio_s: also greedy-decode a seeded ~N s of the probe (null:
    # subset.eval_audio_s; both null: no probe decode). mini: a small eval of its own every mini.every_steps steps
    # (null: none, and no combined_loss/val curve; run_mini_eval). gate: false = no GO/NO-GO verdict (sanity/overfit
    # runs: "N/A" + the numbers). verdict_version: which pre-registered trend definition the verdict uses
    # (kitsune.evaluate.verdict): 1, the one the first A100 run was judged by (kept here and in every config written
    # before v2, so its INCONCLUSIVE reproduces), or 2 (the CER trend on the pre-cooldown evals, complete-set numbers
    # when every eval in its window has them, de-duplicated end points, the cooldown gain and the pre-cooldown slope
    # reported); verdict_min_epoch_gap: v2's de-duplication, in epochs (verdict_options). reference: null, or
    # {"name": ..., "path": <JSON file, relative to the repo root>} - a reference model's corpus CER on the complete
    # gate sets (kitsune.evaluate.load_reference has the file format), which the verdict reports next to the teacher
    # gate, per set and pooled, without any effect on the tier (reference_model). full_at_fracs (steps clock): null, or
    # a sorted list of fractions in (0, 1): a complete eval (as a full_every_epochs one) right after step
    # round(f * max_steps), next to the other cadences (frac_steps)
    "eval": {"every_min": 20, "every_steps": None, "greedy_subset": 500, "probe": True, "final_full_greedy": True,
             "batch_s": 400, "check_baselines": True, "every_epochs": None, "probe_is_train": False,
             "probe_greedy_audio_s": None, "full_every_epochs": None,
             "mini": {"every_steps": None, "val_per_set": 32, "train_utts": 64, "greedy": True}, "gate": True,
             # dev: the dev-slice eval (run_dev_eval), on when every_epochs (a number > 0: at every such fraction of
             # an epoch) or every_steps is set: teacher-forced (greedy too: CTC free, AED a decode) on a seeded
             # per_source rows per train source of the selection's split "dev" (seed null: the run's), its pooled
             # objective / KL / CE the early stop's dev_* metrics; at_start: a step-0 one (never checked), at_end: a
             # final one. Null the leaves, never the dict (validate's bool check)
             "dev": {"every_epochs": None, "every_steps": None, "per_source": 600, "seed": None, "greedy": False,
                     "at_start": True, "at_end": True},
             "verdict_version": 1, "verdict_min_epoch_gap": 0.25, "reference": None, "full_at_fracs": None},
    # full_at_fracs / weights_at_fracs (steps clock): null, or fractions in (0, 1): a full state / exported weights at
    # step round(f * max_steps); those full states stay local past keep_local's rotation (a T/2 branch resumes the
    # 0.4 one). upload_full_at: "pre_cooldown", "end" and "frac:<f>" (the full state saved at fraction f)
    # upload_full_every_min (the full runs' timed states): null, or every N minutes of loop clock a full state (reason
    # "timed") to hf.scratch_repo, replacing the run's previous one there (ScratchUploader; the module docstring)
    "ckpt": {"weights_every_min": 30, "full_local_every_min": 30, "weights_every_steps": None,
             "full_every_steps": None, "keep_local": 2, "upload_full_at": ["pre_cooldown", "end"],
             "full_after_smoke": True, "full_at_fracs": None, "weights_at_fracs": None, "upload_full_every_min": None},
    # full_scalars_every_steps (fix 7): N > 1 logs the full step row (system stats, per source, buckets, token
    # diagnostics) only every N-th step after the smoke phase, CORE_STEP_TAGS on the others. scalars_parquet (4e):
    # "sync" rewrites metrics/scalars.parquet at every log sync, "close" only at the logger's close
    "log": {"layer_stats_every": 100, "hist_every": 1000, "train_utts_flush": 500, "sync_every_min": 10,
            "samples_per_eval": 8, "capture_env": True, "full_scalars_every_steps": 1, "scalars_parquet": "sync"},
    # scratch_repo: the private scratch model repo of the timed states ("owner/name"; the full-run queue sets it from
    # $KITSUNE_SCRATCH_REPO). The owner creates it: the trainer never does
    "hf": {"output_repo": None, "private": True, "scratch_repo": None},
    # decode_per_set: seeded rows of every train source and eval set decoded before anything else (decode_preflight;
    # 0: skip); max_dropped_frac: the share of undecodable rows over the smoke steps that fails the smoke (smoke_end)
    "smoke": {"enabled": True, "steps": 100, "min_audio_s_per_s": 600, "pad_utts": 32, "pad_max_mean_kl": 0.05,
              "pad_min_argmax_agree": None, "require_loss_decrease": True, "decode_per_set": 8,
              "max_dropped_frac": 0.01},
    # checked after every in-loop eval (the module docstring; early_stop_update, early_stop_trigger). Off here, so a
    # config that does not mention it trains to its budget; viability.json turns it on with these values. A dev_*
    # metric is checked after every dev eval instead (eval.dev). smooth: the rule reads the mean of the last `smooth`
    # values (none counted before there are that many); allow_test_sets false refuses a metric that reads the test
    # sets (TEST_SET_METRICS: heldout_kl pools JSUT / CV8 / Reazon-test)
    "early_stop": {"enabled": False, "metric": "heldout_kl", "patience": 3, "min_delta_rel": 0.005,
                   "min_delta_abs": 0.0, "min_evals": 3, "floor": None, "action": "cooldown", "smooth": 1,
                   "allow_test_sets": True},
    # the T/2 branch (steps clock; the module docstring): parent = a local run dir whose full state at
    # round(resume_frac x its max_steps) this run continues, to round(end_frac x max_steps) with the cooldown from the
    # resume step: the WSD schedule of a run with budget end_frac x T, and one eval, the final one (it ignores
    # eval.full_at_fracs and ckpt.*_at_fracs). null: an ordinary run
    "branch": {"parent": None, "resume_frac": 0.4, "end_frac": 0.5},
    # the LR probe (steps clock; the module docstring): metrics only, and at the end the training objective
    # teacher-forced on the complete gate sets (lr_probe_eval)
    "lr_probe": {"enabled": False},
    # the study box's calibration run (kitsune/study_queue.py; the block "the study box" at the end of this file): an
    # lr_probe run (metrics only) that ends with its step-time table - the median step time, logging included, over
    # the steps (window[0], window[1]] and the data-wait share - instead of the teacher-forced objective. barrier:
    # before its first step the run writes runs/<run_id>/READY and waits for runs/<run_id>/GO (or STOP), which the
    # box writes into every run of a calibration group at once, so the group's windows all fall while all of it trains
    "calibrate": {"enabled": False, "window": [50, 250], "barrier": False},
    "seed": 1234,
}
# the dev-slice metrics (run_dev_eval's pooled numbers, read from st["dev_history"]): the training objective, its KL
# and its CE (CTC: the CTC loss) per target token
DEV_METRICS = ("dev_ce", "dev_kl", "dev_objective")
EARLY_STOP_METRICS = ("probe_kl", "heldout_kl", "train_loss", *DEV_METRICS)
# the metrics that read the test sets (heldout_kl pools the gate sets JSUT / CV8 / Reazon-test): early_stop.
# allow_test_sets false refuses them, so the full runs' stop cannot peek at what they are scored on (probe_kl reads
# train rows, train_loss the objective, dev_* the dev slice)
TEST_SET_METRICS = ("heldout_kl",)
# config keys a resume cannot change (resume_overrides): they shape the step plan or the data it walks, and the
# planner state it restores (the epoch position) is only valid for the plan it was saved with; or they decide what the
# model and its loss are (BN mode, the aux-CTC head in the full state, the augmentation stream, the student family);
# or which dev rows the early stop reads (eval.dev.per_source / seed: a change would compare other rows' numbers).
# augment.* is not among them: the train-data augmentation happens inside the loader's dataset and leaves the step
# plan and its position alone, so a resume may turn it on or off or retune it (a continuation that adds it); a
# crash-resume keeps it anyway, since it reads the run's own config
RESUME_FIXED = ("seed", "mix", "sources", "selection", "subset", "batch.step_audio_s", "batch.micro_audio_s",
                "batch.pool_micro", "batch.max_dec_len", "bn.mode", "loss.aux_ctc_weight", "specaug.seed", "family",
                "eval.dev.per_source", "eval.dev.seed")
STOP_FILE = "STOP"  # runs/<run_id>/STOP: finish the step under way, then the end phase (reason "stop_file")
# runs/<run_id>/COOLDOWN: start the WSD cooldown now, once (reason "cooldown_file"; ignored inside a cooldown), whatever
# early_stop.enabled says (cooldown_file). A resume_reset renames it to COOLDOWN.consumed-<UTC stamp>
COOLDOWN_FILE = "COOLDOWN"
# the LR probe's objective pools these complete sets (the D32a gate sets, kitsune.evaluate.GATE_SETS)
LR_PROBE_SETS = tuple(trainset.EVAL_SETS)
# a branch requires the parent's config to equal its own except these keys (and everything under the ones ending
# in ".")
BRANCH_FREE = ("run_name", "branch.", "hf.", "eval.full_at_fracs", "ckpt.", "log.")
AUX_CTC_PREFIX = "aux_ctc."  # parameter names of the aux-CTC head (R.param_names; never in exported weights)
FAMILIES = ("aed", "ctc")  # the student families (config key family; the module docstring)
# the CTC family's per-utterance and per-step terms (train_step_ctc): kl (= kl_dense + kl_blank) and ctc first, as the
# AED TERMS start with kl and ce (the non-finite-step event reads the first two), then the frame counts and metrics
CTC_TERMS = ("kl", "ctc", "argmax_agree", "n_frames", "n_tokens", "kl_dense", "kl_blank", "n_dense", "argmax_blank",
             "teacher_blank")
CTC_TAGS = dict(kl="loss/kl", ctc="loss/ctc", kl_dense="loss/kl_dense", kl_blank="loss/kl_blank")  # per target token
CTC_FRAME_TAGS = dict(argmax_agree="ctc/argmax_agree", argmax_blank="ctc/argmax_blank",
                      teacher_blank="ctc/teacher_blank", n_dense="ctc/frac_dense")  # per valid frame


class ThroughputTooLow(RuntimeError):
    """Smoke-phase throughput below smoke.min_audio_s_per_s: exit code 3 (the supervisor stops, never resumes)."""


class SmokeFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------- config


def _merge(base: dict, over: dict, where: str = "") -> dict:
    """Deep-merge `over` into a copy of `base`; unknown keys are an error (a typo must not silently do nothing).
    Keys starting with "_" are comments and dropped."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue
        if k not in out:
            raise SystemExit(f"unknown config key {where}{k}")
        out[k] = _merge(out[k], v, f"{where}{k}.") if isinstance(out[k], dict) and isinstance(v, dict) else v
    return out


def apply_set(cfg: dict, assignment: str) -> tuple[str, object]:
    """--set a.b=value: the value is parsed as JSON when it is valid JSON, else kept as a string."""
    key, eq, raw = assignment.partition("=")
    if not eq or not key:
        raise SystemExit(f"--set expects key=value, got {assignment!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    node, parts = cfg, key.split(".")
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            raise SystemExit(f"unknown config key {key}")
        node = node[p]
    if parts[-1] not in node:
        raise SystemExit(f"unknown config key {key}")
    node[parts[-1]] = value
    return key, value


def load_config(path: str | None, sets: list[str]) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path) if Path(path).is_absolute() else (Path.cwd() / path if (Path.cwd() / path).exists() else ROOT / path)
        cfg = _merge(cfg, json.loads(p.read_text(encoding="utf-8")))
    for s in sets:
        apply_set(cfg, s)
    validate(cfg)
    return cfg


def _leaves(d: dict, where: str = ""):
    """("a.b.c", value) for every non-dict value of a nested config."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, f"{where}{k}.")
        else:
            yield f"{where}{k}", v


def validate(cfg: dict):
    # every key whose default is a bool must stay one: --set perf.tf32=False is not JSON, so apply_set keeps the
    # string 'False', which is truthy and would silently do the opposite of what was asked
    flat = dict(_leaves(cfg))
    for key, default in _leaves(DEFAULTS):
        if isinstance(default, bool) and not isinstance(flat.get(key), bool):
            raise SystemExit(f"{key} must be true or false, got {flat.get(key)!r}")
    if cfg["autocast"] not in ("bfloat16", "none", None):
        raise SystemExit(f"autocast must be 'bfloat16' or 'none' (fp16 would need a GradScaler), got {cfg['autocast']}")
    sch, sub, ev_cfg = cfg["schedule"], cfg["subset"], cfg["eval"]
    if sch["clock"] not in ("wall", "steps", "epochs"):
        raise SystemExit(f"schedule.clock must be 'wall', 'steps' or 'epochs', got {sch['clock']}")
    if sch["clock"] == "steps" and not sch["max_steps"]:
        raise SystemExit("schedule.clock 'steps' needs schedule.max_steps")
    if sch["clock"] == "epochs" and not _pos_int(sch["epochs"]):
        raise SystemExit(f"schedule.clock 'epochs' needs schedule.epochs (an int >= 1), got {sch['epochs']}")
    if cfg["memory"]["grad_ckpt"] not in ("auto", True, False):
        raise SystemExit(f"memory.grad_ckpt must be 'auto', true or false, got {cfg['memory']['grad_ckpt']}")
    if cfg["optim"]["offload"] not in ("none", "cpu"):
        raise SystemExit(f"optim.offload must be 'none' or 'cpu', got {cfg['optim']['offload']}")
    if not (_number(cfg["perf"]["loader_timeout_s"]) and cfg["perf"]["loader_timeout_s"] >= 0):
        raise SystemExit(f"perf.loader_timeout_s must be a number of seconds >= 0 (0: no timeout), got "
                         f"{cfg['perf']['loader_timeout_s']}")
    if not (cfg["perf"]["profile_smoke"] == "auto" or isinstance(cfg["perf"]["profile_smoke"], bool)):
        raise SystemExit(f"perf.profile_smoke must be 'auto', true or false, got {cfg['perf']['profile_smoke']!r}")
    sm = cfg["smoke"]
    if not (isinstance(sm["decode_per_set"], int) and not isinstance(sm["decode_per_set"], bool)
            and sm["decode_per_set"] >= 0):
        raise SystemExit(f"smoke.decode_per_set must be an int >= 0, got {sm['decode_per_set']}")
    if not (_number(sm["max_dropped_frac"]) and 0 <= sm["max_dropped_frac"] <= 1):
        raise SystemExit(f"smoke.max_dropped_frac must be a fraction in [0, 1], got {sm['max_dropped_frac']}")
    for a, b in (("train_audio_s", "train_utts"), ("eval_audio_s", "eval_utts_per_set")):
        if sub[a] is not None and sub[b]:
            raise SystemExit(f"subset.{a} and subset.{b} are two ways to pick the same subset: set one")
    for key, v in (("subset.train_audio_s", sub["train_audio_s"]), ("subset.eval_audio_s", sub["eval_audio_s"]),
                   ("eval.probe_greedy_audio_s", ev_cfg["probe_greedy_audio_s"])):
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0):
            raise SystemExit(f"{key} must be null or a number of seconds > 0, got {v}")
    for key in ("every_epochs", "full_every_epochs"):
        if ev_cfg[key] is not None and not _pos_int(ev_cfg[key]):
            raise SystemExit(f"eval.{key} must be null or an int >= 1, got {ev_cfg[key]}")
    if ev_cfg["every_epochs"] and ev_cfg["full_every_epochs"]:
        raise SystemExit("eval.every_epochs and eval.full_every_epochs are two cadences of the same eval: set one")
    mini = ev_cfg["mini"]
    if mini["every_steps"] is not None and not _pos_int(mini["every_steps"]):
        raise SystemExit(f"eval.mini.every_steps must be null or an int >= 1, got {mini['every_steps']}")
    for key in ("val_per_set", "train_utts"):
        if not (isinstance(mini[key], int) and not isinstance(mini[key], bool) and mini[key] >= 0):
            raise SystemExit(f"eval.mini.{key} must be an int >= 0, got {mini[key]}")
    if ev_cfg["probe_is_train"] and not ev_cfg["probe"]:
        raise SystemExit("eval.probe_is_train needs eval.probe")
    if not (_pos_int(ev_cfg["verdict_version"]) and ev_cfg["verdict_version"] in (1, 2)):
        raise SystemExit(f"eval.verdict_version must be 1 or 2, got {ev_cfg['verdict_version']!r}")
    if not (_number(ev_cfg["verdict_min_epoch_gap"]) and ev_cfg["verdict_min_epoch_gap"] >= 0):
        raise SystemExit(f"eval.verdict_min_epoch_gap must be a number of epochs >= 0, got "
                         f"{ev_cfg['verdict_min_epoch_gap']!r}")
    ref = ev_cfg["reference"]
    if ref is not None and not (isinstance(ref, dict) and set(ref) <= {"name", "path"}
                                and isinstance(ref.get("path"), str) and ref["path"]
                                and (ref.get("name") is None or isinstance(ref["name"], str))):
        raise SystemExit(f"eval.reference must be null or {{\"name\": <label>, \"path\": <JSON file>}}, got {ref!r}")
    es = cfg["early_stop"]
    if es["metric"] not in EARLY_STOP_METRICS:
        raise SystemExit(f"early_stop.metric must be one of {', '.join(EARLY_STOP_METRICS)}, got {es['metric']}")
    if es["action"] not in ("stop", "cooldown"):
        raise SystemExit(f"early_stop.action must be 'stop' or 'cooldown', got {es['action']}")
    if not _pos_int(es["patience"]):
        raise SystemExit(f"early_stop.patience must be an int >= 1, got {es['patience']}")
    if not (isinstance(es["min_evals"], int) and not isinstance(es["min_evals"], bool) and es["min_evals"] >= 0):
        raise SystemExit(f"early_stop.min_evals must be an int >= 0, got {es['min_evals']}")
    for key in ("min_delta_rel", "min_delta_abs"):
        if not (_number(es[key]) and es[key] >= 0):
            raise SystemExit(f"early_stop.{key} must be a number >= 0, got {es[key]}")
    if es["floor"] is not None and not _number(es["floor"]):
        raise SystemExit(f"early_stop.floor must be null or a number, got {es['floor']}")
    if es["enabled"] and es["metric"] == "probe_kl" and not ev_cfg["probe"]:
        raise SystemExit("early_stop.metric 'probe_kl' needs eval.probe")
    validate_study(cfg)
    validate_full(cfg)
    validate_state(cfg)
    validate_augment(cfg)


def validate_study(cfg: dict):
    """The size study's keys (bn, loss.aux_ctc_weight, optim.weight_decay, the steps clock's warm-up, the fraction
    evals and checkpoints, branch, specaug.seed, lr_probe, family and loss.w_ctc); their defaults are the trainer as it
    was before them."""
    sch, ck, ev_cfg = cfg["schedule"], cfg["ckpt"], cfg["eval"]
    steps_clock = sch["clock"] == "steps"
    if cfg["family"] not in FAMILIES:
        raise SystemExit(f"family must be one of {', '.join(FAMILIES)}, got {cfg['family']!r}")
    if not (_number(cfg["loss"]["w_ctc"]) and cfg["loss"]["w_ctc"] >= 0):
        raise SystemExit(f"loss.w_ctc must be a number >= 0, got {cfg['loss']['w_ctc']!r}")
    if cfg["family"] == "ctc" and not (isinstance(cfg["parakeet_root"], str) and cfg["parakeet_root"]):
        raise SystemExit("family 'ctc' needs parakeet_root (the local parakeet_out dir with <src>/<stem>.npz/.jsonl and "
                         "meta.json)")
    bn = cfg["bn"]
    if bn["mode"] not in BN_MODES:
        raise SystemExit(f"bn.mode must be one of {', '.join(BN_MODES)}, got {bn['mode']!r}")
    if bn["momentum"] is not None and not (_number(bn["momentum"]) and 0 < bn["momentum"] <= 1):
        raise SystemExit(f"bn.momentum must be null (the modules' own) or a number in (0, 1], got {bn['momentum']!r}")
    if bn["mode"] == "train" and cfg["memory"]["grad_ckpt"] is True:
        raise SystemExit("bn.mode 'train' never uses gradient checkpointing (the recomputed forward would update the "
                         "BatchNorm running stats twice): set memory.grad_ckpt to 'auto' or false")
    for key in ("aux_ctc_weight",):
        if not (_number(cfg["loss"][key]) and cfg["loss"][key] >= 0):
            raise SystemExit(f"loss.{key} must be a number >= 0, got {cfg['loss'][key]!r}")
    if cfg["family"] == "ctc" and float(cfg["loss"]["aux_ctc_weight"]) > 0:
        raise SystemExit("loss.aux_ctc_weight is the AED students' aux-CTC head; a CTC student's CTC loss is its main "
                         "loss (loss.w_ctc): set loss.aux_ctc_weight 0 with family 'ctc'")
    if not (_number(cfg["optim"]["weight_decay"]) and cfg["optim"]["weight_decay"] >= 0):
        raise SystemExit(f"optim.weight_decay must be a number >= 0, got {cfg['optim']['weight_decay']!r}")
    if not (isinstance(sch["warmup_steps"], int) and not isinstance(sch["warmup_steps"], bool)
            and sch["warmup_steps"] >= 0):
        raise SystemExit(f"schedule.warmup_steps must be an int >= 0, got {sch['warmup_steps']!r}")
    if steps_clock and sch["warmup_steps"] >= (1.0 - float(sch["cooldown_frac"])) * float(sch["max_steps"]):
        # the steps clock does not cap the warm-up (warmup_steps), and wsd_lr multiplies the cooldown by the warm-up
        # factor: a warm-up still running where the cooldown starts never reaches the peak LR at all
        raise SystemExit(f"schedule.warmup_steps {sch['warmup_steps']} does not end before the cooldown starts at step "
                         f"{(1.0 - float(sch['cooldown_frac'])) * float(sch['max_steps']):g} = (1 - cooldown_frac "
                         f"{sch['cooldown_frac']}) x max_steps {sch['max_steps']}: the run would never reach its peak LR")
    fracs = {}
    for key, section, want_sorted in (("eval.full_at_fracs", ev_cfg, True), ("ckpt.full_at_fracs", ck, False),
                                      ("ckpt.weights_at_fracs", ck, False)):
        v = section[key.split(".")[1]]
        if v is None:
            continue
        if not (isinstance(v, list) and v and all(_number(f) and 0 < f < 1 for f in v) and len(set(v)) == len(v)
                and (not want_sorted or v == sorted(v))):
            raise SystemExit(f"{key} must be null or a{' sorted' if want_sorted else ''} list of distinct fractions in "
                             f"(0, 1), got {v!r}")
        if not steps_clock:
            raise SystemExit(f"{key} needs schedule.clock 'steps' (fractions of schedule.max_steps)")
        fracs[key] = [float(f) for f in v]
    ups = ck["upload_full_at"]
    if not isinstance(ups, list):
        raise SystemExit(f"ckpt.upload_full_at must be a list, got {ups!r}")
    for u in ups:
        f = upload_frac(u)
        if u not in ("pre_cooldown", "end") and (f is None or f not in fracs.get("ckpt.full_at_fracs", [])):
            raise SystemExit(f"ckpt.upload_full_at entries are 'pre_cooldown', 'end' or 'frac:<f>' with f one of "
                             f"ckpt.full_at_fracs ({ck['full_at_fracs']}), got {u!r}")
    sa = cfg["specaug"]["seed"]
    if sa is not None and not (isinstance(sa, int) and not isinstance(sa, bool) and sa >= 0):
        raise SystemExit(f"specaug.seed must be null (the run's seed) or an int >= 0, got {sa!r}")
    br = cfg["branch"]
    if br["parent"] is not None:
        if not (isinstance(br["parent"], str) and br["parent"]):
            raise SystemExit(f"branch.parent must be null or a run dir, got {br['parent']!r}")
        if not steps_clock:
            raise SystemExit("branch.parent needs schedule.clock 'steps' (the branch's schedule is in steps)")
        if cfg["lr_probe"]["enabled"]:
            raise SystemExit("branch.parent and lr_probe.enabled are two different runs: set one")
    if not (_number(br["resume_frac"]) and _number(br["end_frac"])
            and 0 < br["resume_frac"] < br["end_frac"] <= 1):
        raise SystemExit(f"branch.resume_frac and branch.end_frac must satisfy 0 < resume_frac < end_frac <= 1, got "
                         f"{br['resume_frac']!r}, {br['end_frac']!r}")
    if br["parent"] is not None and br["resume_frac"] > 1.0 - float(sch["cooldown_frac"]):
        raise SystemExit(f"branch.resume_frac {br['resume_frac']} is past the parent's cooldown start "
                         f"(1 - cooldown_frac = {1.0 - float(sch['cooldown_frac']):g}): its state is no longer at the "
                         "stable LR")
    if cfg["lr_probe"]["enabled"]:
        if not steps_clock:
            raise SystemExit("lr_probe.enabled needs schedule.clock 'steps'")
        missing = [s for s in LR_PROBE_SETS if s not in cfg["eval_sets"]]
        if missing or cfg["subset"]["eval_audio_s"] is not None or cfg["subset"]["eval_utts_per_set"]:
            raise SystemExit(f"lr_probe.enabled scores the COMPLETE gate sets {', '.join(LR_PROBE_SETS)}: eval_sets must "
                             f"hold them (missing: {missing}) and subset.eval_audio_s / eval_utts_per_set be null")


def validate_full(cfg: dict):
    """The full-data runs' trainer keys (eval.dev, early_stop.smooth / allow_test_sets and the dev metrics,
    schedule.resume_reset / deadline_*, memory.probe_extended / probe_shapes); their defaults are the trainer as it was
    before them. Also refused: the epoch clock with a schedule.max_steps (Run.max_steps() ends every clock there, so a
    full config copied from a study config would stop long before its cooldown) and the dev eval in an LR probe
    (metrics only). selection_recipe.full_study is validate_box's (the box path's keys)."""
    sch, es, dev = cfg["schedule"], cfg["early_stop"], cfg["eval"]["dev"]
    if dev["every_epochs"] is not None and not (_number(dev["every_epochs"]) and dev["every_epochs"] > 0):
        raise SystemExit(f"eval.dev.every_epochs must be null or a number of epochs > 0, got {dev['every_epochs']!r}")
    if dev["every_steps"] is not None and not _pos_int(dev["every_steps"]):
        raise SystemExit(f"eval.dev.every_steps must be null or an int >= 1, got {dev['every_steps']!r}")
    if dev["every_epochs"] is not None and dev["every_steps"] is not None:
        raise SystemExit("eval.dev.every_epochs and eval.dev.every_steps are two cadences of the same dev eval: set "
                         "one")
    if not _pos_int(dev["per_source"]):
        raise SystemExit(f"eval.dev.per_source must be an int >= 1, got {dev['per_source']!r}")
    if dev["seed"] is not None and not (isinstance(dev["seed"], int) and not isinstance(dev["seed"], bool)
                                        and dev["seed"] >= 0):
        raise SystemExit(f"eval.dev.seed must be null (the run's seed) or an int >= 0, got {dev['seed']!r}")
    if dev_on(cfg) and cfg["lr_probe"]["enabled"]:
        raise SystemExit("eval.dev and lr_probe.enabled: an LR probe is metrics only (no eval in the loop); set the "
                         "eval.dev cadences null")
    if not _pos_int(es["smooth"]):
        raise SystemExit(f"early_stop.smooth must be an int >= 1 (1: no smoothing), got {es['smooth']!r}")
    if es["metric"] in TEST_SET_METRICS and not es["allow_test_sets"]:
        raise SystemExit(f"early_stop.metric {es['metric']!r} reads the test sets (it pools the gate sets eval_jsut, "
                         "eval_cv8 and eval_reazon, which the runs are scored on) and early_stop.allow_test_sets is "
                         "false: use a dev metric (" + ", ".join(DEV_METRICS) + ") with eval.dev")
    if es["enabled"] and es["metric"] in DEV_METRICS and not dev_on(cfg):
        raise SystemExit(f"early_stop.metric {es['metric']!r} reads the dev evals: set eval.dev.every_epochs or "
                         "eval.dev.every_steps")
    if sch["clock"] == "epochs" and sch["max_steps"] is not None:
        raise SystemExit(f"schedule.clock 'epochs' with schedule.max_steps {sch['max_steps']!r}: max_steps ends the "
                         "run on every clock (a study config's value would stop it before its cooldown); set it null")
    if sch["deadline_cooldown"] and sch["clock"] not in ("epochs", "steps"):
        raise SystemExit(f"schedule.deadline_cooldown needs schedule.clock 'epochs' or 'steps' (the wall clock fits "
                         f"its budget to the deadline itself: fit_budget), got {sch['clock']!r}")
    for key in ("deadline_check_steps", "deadline_window_steps"):
        if not _pos_int(sch[key]):
            raise SystemExit(f"schedule.{key} must be an int >= 1, got {sch[key]!r}")
    shapes = cfg["memory"]["probe_shapes"]
    if shapes is not None:
        ok = isinstance(shapes, list) and bool(shapes)
        names = []
        for s in shapes if ok else ():
            ok = ok and (isinstance(s, dict) and set(s) == {"name", "durations"} and isinstance(s["name"], str)
                         and re.fullmatch(r"[A-Za-z0-9_-]+", s["name"]) is not None
                         and isinstance(s["durations"], list) and bool(s["durations"])
                         and all(_number(d) and d > 0 for d in s["durations"]))
            names.append(s.get("name") if isinstance(s, dict) else None)
        if not ok or len(set(names)) != len(names):
            raise SystemExit(f"memory.probe_shapes must be null or a non-empty list of {{\"name\": [A-Za-z0-9_-]+ "
                             f"(unique), \"durations\": [seconds > 0, ...]}}, got {shapes!r}")


HF_REPO_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"  # "owner/name", as the Hub names a repo


def validate_state(cfg: dict):
    """The full runs' state and logging keys (the module docstring's "Full-data runs: timed states, heartbeats, 4e, fix
    7"): ckpt.upload_full_every_min, hf.scratch_repo, log.full_scalars_every_steps and log.scalars_parquet; their
    defaults are the trainer as it was before them. That a run asking for timed states has a scratch repo to send them
    to is build()'s check, not this one: scripts/05_evaluate.py validates the same configs, and the full-run queue sets
    hf.scratch_repo on the trainer's command line only."""
    ck, hf, lg = cfg["ckpt"], cfg["hf"], cfg["log"]
    every = ck["upload_full_every_min"]
    if every is not None and not (_number(every) and every > 0):
        raise SystemExit(f"ckpt.upload_full_every_min must be null or a number of minutes > 0, got {every!r}")
    repo = hf["scratch_repo"]
    if repo is not None and not (isinstance(repo, str) and re.fullmatch(HF_REPO_RE, repo)):
        raise SystemExit(f"hf.scratch_repo must be null or a repo id \"owner/name\", got {repo!r}")
    if repo is not None and repo == hf["output_repo"]:
        raise SystemExit(f"hf.scratch_repo {repo} is hf.output_repo: the timed states go to a repo of their own (each "
                         "upload squashes that repo's history)")
    if not _pos_int(lg["full_scalars_every_steps"]):
        raise SystemExit(f"log.full_scalars_every_steps must be an int >= 1, got {lg['full_scalars_every_steps']!r}")
    if lg["scalars_parquet"] not in ("sync", "close"):
        raise SystemExit(f"log.scalars_parquet must be 'sync' or 'close', got {lg['scalars_parquet']!r}")


def validate_augment(cfg: dict):
    """augment.* (the CTC family's train-data augmentation; DEFAULTS, the module docstring), checked whether or not it
    is enabled, so a typo in a block that a later --set turns on fails now. The AED students refuse it: their token
    targets are per decoder position, and a cut or joined row has no such targets (kitsune.trainset.dataset_for)."""
    a = cfg["augment"]
    seed = a["seed"]
    if seed is not None and not (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0):
        raise SystemExit(f"augment.seed must be null (the run's seed) or an int >= 0, got {seed!r}")
    for key in ("truncate_p", "truncate_pause_p", "concat_p", "mix_p", "truncate_pad_p", "end_pad_p"):
        if not (_number(a[key]) and 0 <= a[key] <= 1):
            raise SystemExit(f"augment.{key} must be a probability in [0, 1], got {a[key]!r}")
    pf = a["pad_frames"]
    if not (isinstance(pf, (list, tuple)) and len(pf) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in pf)
            and 1 <= pf[0] <= pf[1] <= trainset.PAD_MAX_FRAMES):
        raise SystemExit(f"augment.pad_frames must be [low, high] whole 80 ms frames with 1 <= low <= high <= "
                         f"{trainset.PAD_MAX_FRAMES}, got {pf!r}")
    if not (_number(a["truncate_min_frac"]) and 0 <= a["truncate_min_frac"] < 1):
        raise SystemExit(f"augment.truncate_min_frac must be a fraction in [0, 1), got {a['truncate_min_frac']!r}")
    if not (_number(a["truncate_min_s"]) and a["truncate_min_s"] >= 0):
        raise SystemExit(f"augment.truncate_min_s must be a number of seconds >= 0, got {a['truncate_min_s']!r}")
    if not (_number(a["concat_max_s"]) and a["concat_max_s"] > 0):
        raise SystemExit(f"augment.concat_max_s must be a number of seconds > 0, got {a['concat_max_s']!r}")
    if not (isinstance(a["concat_max_n"], int) and not isinstance(a["concat_max_n"], bool) and a["concat_max_n"] >= 2):
        raise SystemExit(f"augment.concat_max_n must be an int >= 2 (utterances per joined row), got "
                         f"{a['concat_max_n']!r}")
    snr = a["mix_snr_db"]
    if not (isinstance(snr, (list, tuple)) and len(snr) == 2 and all(_number(x) for x in snr)
            and 0 <= snr[0] <= snr[1]):
        raise SystemExit(f"augment.mix_snr_db must be [low, high] dB with 0 <= low <= high (the interferer below the "
                         f"row: the targets are the row's, so it must stay the dominant voice), got {snr!r}")
    if a["enabled"] and not is_ctc(cfg):
        raise SystemExit(f"augment.enabled is the CTC family's (family 'ctc'): an AED student's targets are per "
                         f"decoder position and cannot follow a cut or joined row; this config's family is "
                         f"{family(cfg)!r}")


def augment_on(cfg: dict) -> bool:
    """augment.enabled (False for a config without the block: a unit test's partial config, a state from before it)."""
    return bool((cfg.get("augment") or {}).get("enabled"))


def upload_frac(entry) -> float | None:
    """The fraction of a ckpt.upload_full_at entry "frac:<f>", else None."""
    if not (isinstance(entry, str) and entry.startswith("frac:")):
        return None
    try:
        return float(entry[5:])
    except ValueError:
        return None


def _pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def epoch_mode(cfg: dict) -> bool:
    """Evals at epoch ends (the epoch clock, eval.every_epochs or eval.full_every_epochs): eval records, events and
    scalars carry the epoch."""
    return cfg["schedule"]["clock"] == "epochs" or bool(epoch_cadence(cfg))


def epoch_cadence(cfg: dict) -> int | None:
    """In-loop evals at the end of every N-th epoch: eval.full_every_epochs or eval.every_epochs (None: every_min /
    every_steps). The planner's position says where an epoch ends on every clock (StepPlanner.step_done moves it to
    the next epoch after the last step of an epoch's plan)."""
    ev_cfg = cfg["eval"]
    return int(ev_cfg["full_every_epochs"] or ev_cfg["every_epochs"] or 0) or None


def rpath(value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def bn_mode(cfg: dict) -> str:
    """bn.mode ("frozen" for a config without the key: a unit test's partial config)."""
    return (cfg.get("bn") or {}).get("mode", "frozen")


def family(cfg: dict) -> str:
    """The student family ("aed" for a config without the key: a unit test's partial config, a state from before)."""
    return cfg.get("family") or "aed"


def is_ctc(cfg: dict) -> bool:
    return family(cfg) == "ctc"


def aux_ctc_weight(cfg: dict) -> float:
    return float((cfg.get("loss") or {}).get("aux_ctc_weight", 0.0) or 0.0)


def lr_probe_on(cfg: dict) -> bool:
    return bool((cfg.get("lr_probe") or {}).get("enabled"))


def frac_step(frac: float, max_steps: int) -> int:
    """The step of fraction `frac` of a run of max_steps optimizer steps: round(frac x max_steps) (Python's round, the
    same for a parent's ckpt.full_at_fracs and its branch's resume_frac)."""
    return int(round(float(frac) * int(max_steps)))


def frac_steps(fracs, max_steps: int | None) -> dict[int, float]:
    """step -> fraction for eval.full_at_fracs / ckpt.*_at_fracs (None or empty: none; so without a step count)."""
    return {frac_step(f, max_steps): float(f) for f in fracs or ()} if max_steps else {}


def branch_fracs(cfg: dict) -> dict:
    """The fraction schedules a T/2 branch's config sets and the branch ignores (loop(); its `branch` event lists
    them): a branch config copied from its parent's carries the parent's eval.full_at_fracs and ckpt.*_at_fracs."""
    ck = cfg["ckpt"]
    got = {"eval.full_at_fracs": cfg["eval"]["full_at_fracs"], "ckpt.full_at_fracs": ck["full_at_fracs"],
           "ckpt.weights_at_fracs": ck["weights_at_fracs"],
           "ckpt.upload_full_at": [u for u in ck["upload_full_at"] if upload_frac(u) is not None]}
    return {k: v for k, v in got.items() if v}


def specaug_seed(cfg: dict, step: int, micro: int) -> int:
    """The seed of the SpecAugment masks of micro-batch `micro` (0-based) of optimizer step `step`: a 64-bit mix of
    (specaug.seed, or the run's seed when null; step; micro), so the masks are a pure function of them - a resumed run,
    and a branch of the run, augment every step exactly as the original, whatever happened to other micro-batches."""
    base = cfg["specaug"].get("seed")
    base = int(cfg["seed"]) if base is None else int(base)
    return int(np.random.SeedSequence([base, int(step), int(micro)]).generate_state(1, np.uint64)[0])


# -------------------------------------------------------------------------------------------------------- schedule


def wsd_lr(peak: float, step: int, t: float, T: float, warmup_steps: int, cooldown_frac: float,
           t_c: float | None = None) -> tuple[float, int]:
    """Warmup-stable-decay. `step` is the 1-based optimizer step about to be taken, `t` the progress when it starts
    (loop seconds, or steps done) out of the budget `T`. Linear warmup over warmup_steps, constant until
    t >= t_c = (1 - cooldown_frac) T (or the given t_c: an early-stop cooldown, Run.cooldown_start), then
    peak * (1 - sqrt((t - t_c) / (T - t_c))). Returns (lr, phase) with phase 0 warmup, 1 stable, 2 cooldown."""
    warm = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
    t_c = (1.0 - cooldown_frac) * T if t_c is None else t_c
    if t < t_c:
        return peak * warm, 0 if warm < 1.0 else 1
    frac = min(1.0, (t - t_c) / max(T - t_c, 1e-12))
    return peak * warm * (1.0 - math.sqrt(frac)), 2


def warmup_steps(sch: dict, total_steps: int | None = None) -> int:
    """schedule.warmup_steps; on the epoch clock at most ceil(10 %) of the run's total_steps (a 100-step overfit run
    would otherwise never leave the warmup)."""
    w = int(sch["warmup_steps"])
    if sch["clock"] == "epochs" and total_steps:
        w = min(w, math.ceil(0.1 * total_steps))
    return w


def early_stop_state() -> dict:
    """The early-stop part of the full state (Run.st["early_stop"]): the best value and its step, in-loop evals
    checked, evals since the best, the last value, the loss/objective sum and count since the previous eval (metric
    "train_loss"), the trigger (the `early_stop` event's fields, or None), whether the loop must end (`stop`) and an
    early cooldown's schedule (`cooldown`: t_c and T on the run's clock, or None). recent: the last early_stop.smooth
    raw values (None for a missing one), whose mean the rule reads; cooldown_file: the COOLDOWN file's one action
    ({at_step[, ignored]}, or None). A state written before a key existed gets it from here (train)."""
    return dict(best=None, best_step=None, evals=0, evals_since_best=0, value=None, loss_sum=0.0, loss_n=0,
                triggered=None, stop=False, cooldown=None, recent=[], cooldown_file=None)


def st_full_defaults() -> dict:
    """The full-data runs' part of Run.st (fresh objects each call; a state written before them keeps these defaults
    through R.st.update): dev_history (one record per dev eval, run_dev_eval), dev_last_step / dev_last_epoch (the
    last dev eval's position, dev_due), dev (the dev rows' {ids_sha256, n_ids, per_source, n, store_ids_sha256}: a
    resume on other rows stops, setup_dev), deadline_cooldown (4d's record, fit_epochs_deadline; None: none),
    deadline_rate (the last loop-clock s/step 4d used, for a later launch's first checks), schedule_resets (one record
    per resume_reset) and resume_resets (their count, summary.json's)."""
    return dict(dev_history=[], dev_last_step=0, dev_last_epoch=0.0, dev=None, deadline_cooldown=None,
                deadline_rate=None, schedule_resets=[], resume_resets=0)


def early_stop_update(es: dict, ec: dict, value, step: int) -> str | None:
    """One in-loop eval's update of the early-stop state `es` under the rule `ec` (config early_stop). The eval improves
    when best - value > max(min_delta_abs, min_delta_rel * |best|) (strictly: a tie is no improvement; the first
    checked eval always improves; a missing or non-finite value never does); the best moves only on an improvement.
    Returns the trigger, never before ec["min_evals"] checked evals: "floor" (value <= ec["floor"]), "patience"
    (ec["patience"] evals in a row without an improvement), else None."""
    ok = value is not None and math.isfinite(float(value))
    value = float(value) if ok else None
    es["evals"] += 1
    es["value"] = value
    best = es["best"]
    if ok and (best is None or best - value > max(float(ec["min_delta_abs"]), float(ec["min_delta_rel"]) * abs(best))):
        es["best"], es["best_step"], es["evals_since_best"] = value, int(step), 0
    else:
        es["evals_since_best"] += 1
    if es["evals"] < int(ec["min_evals"]):
        return None
    if ec["floor"] is not None and ok and value <= float(ec["floor"]):
        return "floor"
    if es["evals_since_best"] >= int(ec["patience"]):
        return "patience"
    return None


def due(t: float, step: int, last_t: float, last_step: int, every_min, every_steps) -> bool:
    """Cadence check: every `every_steps` optimizer steps if set, else every `every_min` minutes of loop clock."""
    if every_steps:
        return step - last_step >= int(every_steps)
    return every_min is not None and t - last_t >= float(every_min) * 60


def deadline_unix() -> float | None:
    """When the vast instance is stopped whatever the run is doing (vast/watchdog.sh): $KITSUNE_DEADLINE, else
    $KITSUNE_STATE/deadline (unix seconds, written once at first boot by vast/onstart.sh). None off the box."""
    raw = os.environ.get("KITSUNE_DEADLINE")
    state = os.environ.get("KITSUNE_STATE")
    if not raw and state and (Path(state) / "deadline").is_file():
        raw = (Path(state) / "deadline").read_text(encoding="utf-8").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def shm_cap(nw: int, prefetch: int, micro_audio_s: float, shm: str = "/dev/shm") -> tuple[int, int, dict | None]:
    """Fit the DataLoader's in-flight micro-batches into /dev/shm (Linux). Worker tensors reach the trainer through
    shared memory under every sharing strategy (file_system shm_opens too, so KITSUNE_SHARING does not help). An
    overflow (a container left with Docker's 64 MB) does not crash: the worker prints "unable to allocate shared
    memory(shm) ... (28)" and drops that micro-batch, and the loader would wait for it forever; perf.loader_timeout_s
    turns that stall into a crash vast/supervise.py can resume. Up to num_workers * prefetch micro-batches of up to
    micro_audio_s padded float32 audio are in flight: keep them within half the free space (measured once, at loop
    start) by prefetching less, then using fewer workers, then decoding in-process. Returns (workers, prefetch, change
    or None). A cut below SHM_FEW_WORKERS workers adds a rough decode ceiling to the change
    (DECODE_AUDIO_S_PER_WORKER per worker, one for in-process decoding): the MP3/OGG-heavy train mix then feeds ~1k
    audio-s/s per worker, below what an A100 takes, so the run trains fewer epochs than the GPU allows."""
    if nw <= 0 or not os.path.isdir(shm):
        return nw, prefetch, None
    try:
        free = shutil.disk_usage(shm).free
    except OSError:
        return nw, prefetch, None
    per, budget = micro_audio_s * SR * 4 * 1.25, 0.5 * free  # the waves; targets and ids are small
    n, p = nw, prefetch
    while n * p * per > budget and p > 2:
        p -= 1
    while n * p * per > budget and n > 1:
        n -= 1
    while n * p * per > budget and p > 1:
        p -= 1
    if n * p * per > budget:
        n = 0
    if (n, p) == (nw, prefetch):
        return nw, prefetch, None
    change = dict(shm_free_gb=round(free / 2**30, 3), workers=[nw, n], prefetch=[prefetch, p],
                  micro_audio_s=micro_audio_s)
    if n < min(nw, SHM_FEW_WORKERS):  # the cut, not the configured count, left too few workers
        change["decode_ceiling_audio_s_per_s"] = max(n, 1) * DECODE_AUDIO_S_PER_WORKER
    return n, p, change


# ------------------------------------------------------------------------------------------------------- the run


@dataclass
class Run:
    """Everything one training process holds. `st` is the part that goes into the full state verbatim."""

    cfg: dict
    run_dir: Path
    device: torch.device
    amp: bool
    log: object = None
    model: object = None
    processor: object = None
    tokenizer: object = None
    student_meta: dict = field(default_factory=dict)
    feat_train: object = None
    feat_eval: object = None
    specaug: object = None
    opt: object = None
    l2sp: object = None
    params: list = field(default_factory=list)
    param_names: list = field(default_factory=list)
    train: object = None
    evalstore: object = None
    devstore: object = None  # the dev-slice store (build_dev_store; None: eval.dev off)
    deadline_marks: list = field(default_factory=list)  # 4d's (step, loop clock) samples of this launch (deadline_rate)
    ds: object = None  # the train store's plain dataset: the smoke checks, the memory probe, the FLOP count
    train_ds: object = None  # the train loader's (setup_augment): ds, or under augment.enabled its augmenting copy
    planner: object = None
    probe_ids: list = field(default_factory=list)
    probe_greedy_ids: list = field(default_factory=list)
    greedy_ids: list = field(default_factory=list)
    mini_val_ids: list = field(default_factory=list)  # the mini eval's fixed subsets (mini_subsets)
    mini_train_ids: list = field(default_factory=list)
    mini_rows: dict = field(default_factory=dict)  # their reference / teacher text, read once (mini_teacher_rows)
    src_index: dict = field(default_factory=dict)
    uploader: object = None
    scratch: object = None  # the timed states' ScratchUploader (hf.scratch_repo with an output repo; build), else None
    bn0: dict = field(default_factory=dict)
    gen: object = None
    resumed_from: Path | None = None
    budget_s: float | None = None  # wall-clock T clipped to the instance deadline (fit_budget); None = train_hours
    # (step, greedy summary) of the loop's newest complete-set eval (eval.full_every_epochs): the end phase reuses it as
    # the final eval when the loop ends at that step. Not in st: a resumed run decodes again
    last_complete: tuple | None = None
    vram_cap_gb: float | None = None  # the caching allocator's cap on Windows (cap_vram); None = no cap
    reference: dict | None = None  # eval.reference's per-set CER, read at setup (reference_model); None = none
    aux_ctc: object = None  # the training-only aux-CTC head (setup_aux_ctc; loss.aux_ctc_weight > 0), else None
    # family "ctc": the teacher text (reference, ctc_hyp) of the eval store and the probe rows, read once per run
    # (kitsune.ctc_eval.store_text), key -> {id: row}
    texts: dict = field(default_factory=dict)
    branch_start: bool = False  # this launch starts a T/2 branch from its parent's state (build; not a resume)
    loop_t0: float | None = None
    t_start: float = field(default_factory=time.time)
    # branch: the T/2 branch's schedule (branch_state; None for an ordinary run). fulls_kept: the full states of
    # ckpt.full_at_fracs, which rotate_full keeps. started_utc: the run's first start (train(); a resume keeps it, a T/2
    # branch sets its own; None in a state saved before it existed)
    st: dict = field(default_factory=lambda: dict(
        step=0, train_s=0.0, smoke_done=False, pre_cooldown_done=False, pre_cooldown_full=None, step0_done=False,
        last_eval_t=0.0, last_eval_step=0, last_weights_t=0.0, last_weights_step=0, last_full_t=0.0,
        last_full_step=0, memory={}, flops_per_padded_s=None, history=[], nonfinite_skips=0, nonfinite_total=0,
        oom_skips=0, audio_s=0.0, tokens=0, step_time_s=0.0, smoke_losses=[], smoke_audio_s=0.0, smoke_time_s=0.0,
        smoke_dropped=0, smoke_utts=0, epoch=0, epoch_progress=0.0, resumes=0, weights=[], fulls=[],
        last_objective=None, lr_phase=None, last_eval_epoch=0, total_steps=None, early_stop=early_stop_state(),
        mini_history=[], branch=None, fulls_kept=[], started_utc=None, **st_full_defaults(), **st_timed_defaults()))

    def clock(self) -> float:
        """Loop clock in seconds: continues across resumes, frozen outside the training loop."""
        return self.st["train_s"] + (time.monotonic() - self.loop_t0 if self.loop_t0 is not None else 0.0)

    def max_steps(self) -> int | None:
        """The run's last optimizer step on the steps clock: schedule.max_steps, or a T/2 branch's end step; None when
        no step count ends the run (schedule.max_steps unset)."""
        br = self.st.get("branch")
        if br:
            return int(br["end_step"])
        m = self.cfg["schedule"]["max_steps"]
        return int(m) if m else None

    def base_progress(self) -> tuple[float, float]:
        """(t, T) of the schedule alone: max_steps, the epoch plan's total_steps, or train_hours / the deadline-fitted
        budget_s on the wall clock, before any cooldown record shortens it (progress)."""
        s = self.cfg["schedule"]
        if s["clock"] == "steps":
            return float(self.st["step"]), float(self.max_steps())
        if s["clock"] == "epochs":  # total_steps: plan_epochs
            return float(self.st["step"]), float(self.st["total_steps"])
        return self.clock(), self.budget_s if self.budget_s is not None else float(s["train_hours"]) * 3600

    def cooldowns(self, deadline: bool = True) -> list[dict]:
        """The cooldown records that move the schedule: the early stop's (early_stop.cooldown) and 4d's
        (st["deadline_cooldown"], unless deadline is False), each {t_c, T, ...}."""
        recs = [self.st["early_stop"]["cooldown"], self.st.get("deadline_cooldown") if deadline else None]
        return [c for c in recs if c]

    def progress(self, deadline: bool = True) -> tuple[float, float]:
        """(t, T): loop seconds or steps done, out of the budget; T ends early under an early-stop or 4d cooldown
        record, the smallest T of them (deadline False: without 4d's, the plan it would shorten)."""
        t, T = self.base_progress()
        return t, min([T] + [c["T"] for c in self.cooldowns(deadline)])

    def cooldown_start(self, T: float) -> float:
        """t_c of the WSD schedule with budget T: (1 - cooldown_frac) T, or where an early-stop or 4d cooldown begins
        (the earlier of their t_c when both exist), or a T/2 branch's resume step."""
        cds = self.cooldowns()
        if cds:
            return min(c["t_c"] for c in cds)
        br = self.st.get("branch")
        return float(br["t_c"]) if br else (1.0 - float(self.cfg["schedule"]["cooldown_frac"])) * T

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp)

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"


@contextmanager
def frozen_eval(model):
    """Eval mode for the block, then every module's own training flag back (frozen BN stays frozen)."""
    flags = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for m, t in flags:
            m.training = t


def out_repo(cfg: dict) -> str | None:
    return (cfg.get("hf") or {}).get("output_repo") or None


def bound_hub_http():
    """Give huggingface_hub's shared HTTP client a timeout. Its default client has none, and a request the
    server accepted and never answered (create_repo, preupload, the commit POST) would block forever: the trainer
    would never exit and a paid instance would idle until the watchdog. The pinned client (hub ==1.23.0) is kept as it
    is otherwise (its request hook, redirects). Calls that pass timeout=None themselves (list_repo_tree, repo_info)
    and hf_xet's own transfers are not covered; the bounded waits in RunLogger.close and Uploader are."""
    import httpx
    import huggingface_hub
    from huggingface_hub.utils import _http

    def factory():
        c = _http.default_client_factory()
        c.timeout = httpx.Timeout(60, read=300)  # read: well above the Hub's 60 s commit timeout on its side
        return c

    huggingface_hub.set_client_factory(factory)


def hf_api():
    from huggingface_hub import HfApi  # looked up at call time: tests substitute a fake

    bound_hub_http()
    return HfApi()


# ------------------------------------------------------------------------------------------------------- uploads


class Uploader:
    """Checkpoint uploads to <repo>:runs/<run_id>/checkpoints/<name>/ in one background thread, so a 1-9 GB upload
    never blocks training. Every attempt is an event; failures are retried and never raise (vast/finish.py uploads
    again and verifies before an instance is destroyed). A directory with a pending upload is never rotated away; a
    success removes a full state's UPLOAD_MARK (save_full), so one whose upload failed stays too, for finish.py.
    The thread is a daemon fed by a queue (not a ThreadPoolExecutor, whose worker the interpreter joins at exit): an
    upload that never returns must not keep the trainer, and with it a paid instance, alive, so wait() and shutdown()
    take a bound and the thread dies with the process."""

    def __init__(self, api, repo: str | None, run_id: str, private: bool, log, retries=(15, 60, 180)):
        self.api, self.repo, self.run_id, self.private, self.log, self.retries = api, repo, run_id, private, log, retries
        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.pending: dict[Path, Future] = {}
        self._repo_ready = False
        # the marker files a dir may carry that never go up; build() adds SCRATCH_MARK when the run sends timed states
        # (a pre_cooldown or end state can be the same dir as a timed one, so both marks can be in it)
        self.ignore = [UPLOAD_MARK]

    def submit(self, local: Path, name: str) -> Future | None:
        if not self.repo:
            return None
        if self.worker is None:
            self.worker = threading.Thread(target=self._work, name="ckpt-upload", daemon=True)
            self.worker.start()
        fut = Future()
        self.pending[Path(local)] = fut
        self.queue.put((fut, Path(local), name))
        return fut

    def _work(self):
        while (job := self.queue.get()) is not None:
            fut, local, name = job
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(self._upload(local, name))
                except BaseException as e:  # noqa: BLE001  (surfaces in wait(), as the pool's future did)
                    fut.set_exception(e)

    def busy(self) -> set[Path]:
        return {p for p, f in self.pending.items() if not f.done()}

    def _upload(self, local: Path, name: str) -> bool:
        for attempt, wait in enumerate((0, *self.retries)):
            if wait:
                time.sleep(wait)
            t0 = time.time()
            try:
                if not self._repo_ready:
                    self.api.create_repo(self.repo, repo_type="model", private=self.private, exist_ok=True)
                    self._repo_ready = True
                self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=str(local),
                                       path_in_repo=f"runs/{self.run_id}/checkpoints/{name}",
                                       commit_message=f"{self.run_id}: checkpoint {name}",
                                       ignore_patterns=list(self.ignore))
                upload_s = round(time.time() - t0, 1)
                # listed inside the try: a file renamed away meanwhile (the end save's trainer.pt.tmp) is a logged,
                # retried attempt, not an exception that escapes the upload's future
                gb = round(sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1e9, 3)
                try:
                    (local / UPLOAD_MARK).unlink(missing_ok=True)  # on the Hub now: rotation may take it
                except OSError:
                    pass  # kept a little longer, and uploaded once more by finish.py: no harm
                self.log.event("ckpt_upload_ok", name=name, attempt=attempt, gb=gb, upload_s=upload_s)
                return True
            except Exception as e:
                self.log.event("ckpt_upload_error", name=name, attempt=attempt, error=f"{type(e).__name__}: {e}"[:2000])
        self.log.event("ckpt_upload_failed", name=name, attempts=len(self.retries) + 1)
        return False

    def wait(self, timeout: float | None = None) -> dict[str, bool | None]:
        """name -> the upload's result; None for one still running after `timeout` s (finish.py uploads it again)."""
        done = concurrent.futures.wait(list(self.pending.values()), timeout=timeout).done
        return {p.name: f.result() if f in done else None for p, f in list(self.pending.items())}

    def shutdown(self, timeout: float = 0.0):
        """Stop the thread once the queued uploads are done, waiting at most `timeout` s for them."""
        if self.worker is not None:
            self.queue.put(None)
            self.worker.join(timeout)

    def abandon(self, timeout: float) -> list[str]:
        """The failure path: wait at most `timeout` s for the pending uploads, cancel the queued ones, stop the thread;
        returns the names of the ones that did not finish. The one running dies with the process: weights go up with
        the supervisor's post-crash sync (vast/finish.py), a pre_cooldown full state with the resumed run (build)."""
        pending = {p: f for p, f in self.pending.items() if not f.done()}
        done = concurrent.futures.wait(list(pending.values()), timeout=timeout).done
        for f in pending.values():
            f.cancel()  # False for the running one
        self.shutdown()
        return [p.name for p, f in pending.items() if f not in done]


class ScratchUploader(Uploader):
    """The timed full states (ckpt.upload_full_every_min) to the private scratch repo hf.scratch_repo, in one daemon
    thread like Uploader's (bounded wait/abandon, the thread dies with the process): each upload is
    kitsune.scratch.upload_state - one commit that adds the state and its pointer runs/<run_id>/timed_state.json and
    deletes the run's previous state, then a history squash - with the pointer made from the dir and the `meta` its
    submit passed (timed_meta). It never calls create_repo (the owner creates the repo; a token scoped to it cannot
    create one) and never touches UPLOAD_MARK (a pre_cooldown state that is also a timed one still owes the runs repo
    its upload). SCRATCH_MARK goes when the upload ends, whether it went up or finally failed; a success also clears it
    from every older full_step_* dir (one left by a cancelled or crashed upload), so rotate_full never keeps more than
    the newest marked dir. The loop reads what went up from `ok` (the worker appends; timed_state) and `sync_due`: a
    log sync after each success (and after the pre_cooldown submit, save_full), which the loop runs, never this thread.
    Timed uploads are never awaited - the end phase and the failure path cancel the queued ones (abandon(0)) - except
    by save_full's trainer-only rewrite of the same step's dir, for at most UPLOAD_WAIT_S (the pointer must name the
    bytes the upload commits)."""

    def __init__(self, api, repo: str, run_id: str, log, retries=None):
        from kitsune import scratch

        super().__init__(api, repo, run_id, True, log, retries=scratch.RETRY_S if retries is None else retries)
        self.meta: dict[Path, dict] = {}
        self.ok: list[dict] = []  # {name, step, time_utc} of every upload that went up in this launch (worker thread)
        self.seen = 0  # how many of them the loop has put into st["timed"]
        self.sync_due = threading.Event()
        # the busy skip last logged: (st["last_timed_step"], the due window since it); timed_state logs one per window
        self.skipped_at: tuple[int, int] | None = None
        self.running: tuple[str, float] | None = None  # (name, time.monotonic() at its start) of the upload under way

    def running_s(self) -> float | None:
        """How long the upload under way has run (None: none): a hung one (xet transfers have no timeout) keeps
        growing in the loop's timed_state_skipped events."""
        run = self.running  # one read: the worker replaces the tuple
        return None if run is None else round(time.monotonic() - run[1], 1)

    def submit(self, local: Path, name: str, meta: dict | None = None) -> Future | None:
        """Queue the upload of the full state `local` (checkpoints/<name>/); meta: make_pointer's keyword arguments."""
        self.meta[Path(local)] = dict(meta or {})
        return super().submit(local, name)

    @staticmethod
    def _unmark(d: Path):
        try:
            (d / SCRATCH_MARK).unlink(missing_ok=True)
        except OSError:
            pass  # kept: rotation keeps only the newest marked dir, and the next success removes it

    def _upload(self, local: Path, name: str) -> bool:
        from kitsune import scratch

        local, ok, ptr = Path(local), False, None
        self.running = (name, time.monotonic())
        try:
            ptr = scratch.make_pointer(self.run_id, local, **self.meta.pop(local, {}))
            ok = bool(scratch.upload_state(self.api, self.repo, self.run_id, local, ptr, log=self.log,
                                           retries=self.retries)["ok"])
        except Exception as e:  # noqa: BLE001  (a state it cannot even hash: logged, never raised into the future)
            q = FULL_RE.match(local.name)
            self.log.event("timed_state_upload_failed", name=name, **({"step": int(q[1])} if q else {}), attempts=0,
                           error=f"{type(e).__name__}: {e}"[:2000])
        finally:
            self._unmark(local)
            self.running = None
        if ok:
            m = FULL_RE.match(local.name)
            for p in local.parent.iterdir():
                if (q := FULL_RE.match(p.name)) and m and int(q[1]) < int(m[1]):
                    self._unmark(p)
            self.ok.append(dict(name=name, step=int(ptr["step"]),
                                time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")))
            self.sync_due.set()
        return ok


def st_timed_defaults() -> dict:
    """The timed states' part of Run.st (fresh objects; a state saved before them resumes with these): the loop clock and
    step of the last timed save (its cadence), and the last timed state that went up ({name, step, time_utc, count}:
    count = the run's timed uploads that went up, across launches; None before the first). Also data_wait_s, the
    steps' waits for the loader (log_step), for summary.json's throughput.data_wait_frac."""
    return dict(last_timed_t=0.0, last_timed_step=0, timed=None, data_wait_s=0.0)


def hf_roundtrip(R: Run) -> dict:
    """Upload a small file into the output repo and read it back: the credentials, the repo and both directions work
    before the paid hours start (a run whose results cannot leave the box is worthless). A failed attempt is retried
    on the checkpoint uploads' schedule: the create_repo and commit POSTs are sent once by the hub, and one 5xx, 429
    or dropped connection must not stop a paid run after its bootstrap. Bad credentials (401/403) and different bytes
    still fail at once; a retried commit of the same bytes is a no-op."""
    api, repo = R.uploader.api, out_repo(R.cfg)
    payload = json.dumps(dict(run_id=R.run_dir.name, nonce=uuid.uuid4().hex, wall=time.time())).encode()
    rel = f"runs/{R.run_dir.name}/smoke/roundtrip.json"
    (R.run_dir / "smoke").mkdir(exist_ok=True)
    (R.run_dir / "smoke" / "roundtrip.json").write_bytes(payload)
    for attempt, wait in enumerate((0, *R.uploader.retries)):
        if wait:
            time.sleep(wait)
        t0 = time.time()
        try:
            api.create_repo(repo, repo_type="model", private=R.cfg["hf"]["private"], exist_ok=True)
            api.upload_file(path_or_fileobj=payload, path_in_repo=rel, repo_id=repo, repo_type="model",
                            commit_message=f"{R.run_dir.name}: smoke round trip")
            t1 = time.time()
            with tempfile.TemporaryDirectory() as d:
                got = Path(api.hf_hub_download(repo, rel, repo_type="model", local_dir=d)).read_bytes()
            break
        except Exception as e:
            R.log.event("smoke_hf_roundtrip_error", attempt=attempt, error=f"{type(e).__name__}: {e}"[:2000])
            status = getattr(getattr(e, "response", None), "status_code", None)
            if attempt == len(R.uploader.retries) or status in (401, 403):
                raise
    out = dict(ok=got == payload, repo=repo, path=rel, attempt=attempt, upload_s=round(t1 - t0, 2),
               download_s=round(time.time() - t1, 2))
    R.log.event("smoke_hf_roundtrip", **out)
    if not out["ok"]:
        raise SmokeFailed(f"HF round trip to {repo} returned different bytes")
    return out


# ---------------------------------------------------------------------------------------------------------- model


def setup_model(R: Run, grad_ckpt: bool, bn_mode: str = "frozen"):
    """Student -> device in fp32, frozen pos_emb, rel-pos patch, BatchNorm in `bn_mode` (the trainer passes bn.mode;
    scripts/05_evaluate.py, which only evaluates, keeps the default "frozen"), bn.momentum, optional
    checkpointing/compile. Family "ctc": a ParakeetForCTC dir (kitsune.ctc_student.load_ctc_student, sdpa) - no
    decoder, so no pos_emb; the rest the same (a pruned P student's BN holds the teacher's stats, frozen by bn.mode
    "frozen")."""
    from kitsune import student as S

    cfg = R.cfg
    sdir = rpath(cfg["student"])
    ctc = is_ctc(cfg)
    if ctc:
        from kitsune import ctc_student as CS

        model = CS.load_ctc_student(sdir, R.device, dtype=torch.float32)
    else:
        model = S.load_student(sdir, R.device, dtype=torch.float32)
        model.model.decoder.pos_emb.weight.requires_grad_(False)  # fixed sinusoids stored as an Embedding
    if cfg["perf"]["relpos_patch"]:
        patch_relpos_once_per_batch(model)
    train_mode(model, bn_mode)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        train_mode(model, bn_mode)
    momentum = (cfg.get("bn") or {}).get("momentum")
    if momentum is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.momentum = float(momentum)
    if cfg["perf"]["compile"]:  # in place: state_dict keys (L2-SP names, checkpoints) stay unchanged
        if ctc:
            model.encoder.compile(dynamic=True)
        else:
            model.model.encoder.compile(dynamic=True)
            model.model.decoder.compile(dynamic=True)
    R.model = model
    R.student_meta = S.load_meta(sdir)
    R.bn0 = {n: (m.running_mean.detach().clone(), m.running_var.detach().clone())
             for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    seen = set()
    R.params, R.param_names = [], []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            R.params.append(p)
            R.param_names.append(n)


def setup_aux_ctc(R: Run):
    """loss.aux_ctc_weight > 0: the training-only aux-CTC head R.aux_ctc = nn.Linear(d_enc, V + 1) in fp32 on the
    device, on the encoder's final hidden states (before the decoder's encoder->decoder projection), blank = V (the
    extra class; 16,384 for Cohere's vocabulary). Its init is PyTorch's default drawn from its own seed (the run's), so
    it does not depend on what else consumed the global RNG. Its parameters join R.params (the optimizer, gradient
    clipping, per-module stats as "aux_ctc") and the full state (aux_ctc.pt; save_full), never the exported weights
    (step_<N>/, the Hub upload: save_student writes the HF model only). No-op at weight 0."""
    R.aux_ctc = None
    if aux_ctc_weight(R.cfg) <= 0:
        return
    mc = R.model.config
    d_enc, vocab = int(mc.encoder_config.hidden_size), int(mc.vocab_size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(np.random.SeedSequence([int(R.cfg["seed"]), 17]).generate_state(1, np.uint64)[0]))
        head = torch.nn.Linear(d_enc, vocab + 1)
    R.aux_ctc = head.to(R.device, dtype=torch.float32)
    for n, p in R.aux_ctc.named_parameters():
        R.params.append(p)
        R.param_names.append(AUX_CTC_PREFIX + n)
    R.log.event("aux_ctc", weight=aux_ctc_weight(R.cfg), d_enc=d_enc, classes=vocab + 1, blank=vocab,
                params=sum(p.numel() for p in R.aux_ctc.parameters()))


def setup_processing(R: Run):
    """Featurisers, SpecAugment, processor and tokenizer, from the student dir (save_student put the teacher's
    processor there, and every step_<N>/ carries it on). A student dir without a loadable processor is an error: the
    old fallback to the teacher tokenizer read the gated teacher repo, which the box token cannot (a 403 in place of
    the real cause), and where it worked it saved every checkpoint without processor or tokenizer files."""
    from transformers import AutoProcessor

    sdir = rpath(R.cfg["student"])
    try:
        R.processor = AutoProcessor.from_pretrained(str(sdir))
    except Exception as e:
        raise RuntimeError(f"{sdir}: no loadable processor (03_build_student saves processor_config.json, "
                           f"tokenizer.json and tokenizer_config.json next to the weights): {type(e).__name__}: {e}"
                           ) from e
    fe = R.processor.feature_extractor
    if is_ctc(R.cfg):
        # the ParakeetFeatureExtractor path (kitsune.ctc_student.ctc_features: its filterbank, 80 mel, pre-emphasis
        # 0.97, per-feature normalisation, no dither; ValueError for another extractor): the frames the targets were
        # made on. No dither, so the train and eval featurisers are the same computation
        from kitsune.ctc_student import ctc_features

        R.feat_eval = ctc_features(sdir, R.device).logmel
        R.feat_train = ctc_features(sdir, R.device).logmel
    else:
        R.feat_eval = LogMel.from_feature_extractor(fe).to(R.device)
        R.feat_train = LogMel.from_feature_extractor(fe, exact_dither=R.cfg["perf"]["train_exact_dither"]).to(R.device)
    R.tokenizer = R.processor.tokenizer
    R.specaug = SpecAugment(**{k: v for k, v in R.cfg["specaug"].items() if k not in ("enabled", "seed")})
    R.gen = torch.Generator(device=R.device)  # re-seeded for every micro-batch (specaug_seed)
    from kitsune import fullrun, heartbeat

    if os.environ.get(heartbeat.ENV):  # the full-run queue's per-item heartbeat: every eval batch beats (the evaluators
        # call the featuriser once per batch), so a long eval is never silent to the queue's stall check
        R.feat_eval = heartbeat.Beating(R.feat_eval)
    if getattr(R, "log", None) is not None:  # the thread pools this process runs with (the full smoke's check 5)
        R.log.event("threads", torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                    env={k: os.environ.get(k) for k in (*fullrun.ENV_THREAD_POOLS, fullrun.ENV_CPU_QUOTA,
                                                        fullrun.ENV_THREADS_PER_GPU)})


# ------------------------------------------------------------------------------------------------------ optimizer


class CpuOffloadAdamW:
    """torch.optim.AdamW on fp32 master weights in host memory (optim.offload "cpu"), for a GPU that holds the model's
    fp32 weights and gradients but not the AdamW state: the 617M student needs 9.9 GB for weights, gradients, m and v,
    the laptop's RTX 4070 has 8 GB.

    The device keeps its fp32 parameters and gradients for forward/backward (the trainer clips there and logs the
    pre-clip norm as usual). step() copies the gradients into host buffers, frees the device gradients and runs AdamW
    on the masters (foreach, or torch's fused CPU kernel with fused=True); the trainer then applies the decoupled
    L2-SP pull to the masters (its theta_0 lives in host memory too) and push() copies the masters into the device
    parameters. Same hyper-parameters, same order of operations: the maths is the on-device optimizer's
    (tests/test_overfit.py checks the two give the same weights).

    The masters are fp32 copies of fp32 device weights, so after every push() they are bit-identical to the model's
    state_dict. The full state therefore stores only state_dict() (the same format as the on-device optimizer's), and
    load_state_dict() - called after the model's weights are loaded - rebuilds the masters from the device weights.

    Host buffers are views into chunks of CHUNK elements, pinned when the device is CUDA (fast, asynchronous copies):
    CUDA's caching host allocator rounds every pinned block up to a power of two, so one block per tensor (or one
    2.5 GB block) would waste up to half of it. Pinning falls back to pageable memory if the allocation fails."""

    CHUNK = 1 << 26  # fp32 elements (256 MiB)

    def __init__(self, params, *, lr: float, betas: tuple[float, float], eps: float, weight_decay: float,
                 fused: bool = False, decay: list[bool] | None = None):
        self.params = list(params)
        if not self.params or any(p.dtype != torch.float32 for p in self.params):
            raise ValueError("CpuOffloadAdamW needs fp32 device parameters (the masters are rebuilt from them)")
        self.device = self.params[0].device
        self.fused = bool(fused)
        self.pinned, self.pin_error, self.chunks = self.device.type == "cuda", None, []
        self.master = self._host_views()
        self.grads = self._host_views()
        self.pull()
        for m in self.master:
            m.requires_grad_(True)  # leaves the optimizer and L2SP treat as trainable; never part of a graph
        impl = dict(fused=True) if self.fused else dict(foreach=True)
        # decay: one flag per parameter, the on-device optimizer's param groups (decay_groups) over the masters
        groups = self.master if decay is None else decay_groups(self.master, decay, weight_decay)
        self.opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **impl)

    def _host_views(self) -> list[torch.Tensor]:
        out, chunk, used = [], None, 0
        left = sum(p.numel() for p in self.params)
        for p in self.params:
            n = p.numel()
            if chunk is None or used + n > chunk.numel():
                chunk, used = self._alloc(max(n, min(self.CHUNK, left))), 0
            out.append(chunk[used:used + n].view(p.shape))
            used += n
            left -= n
        return out

    def _alloc(self, n: int) -> torch.Tensor:
        if self.pinned:
            try:
                t = torch.empty(n, dtype=torch.float32, pin_memory=True)
                self.chunks.append(t)
                return t
            except RuntimeError as e:  # e.g. the OS refuses to page-lock more memory
                self.pinned, self.pin_error = False, f"{type(e).__name__}: {e}"[:300]
        t = torch.empty(n, dtype=torch.float32)
        self.chunks.append(t)
        return t

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def param_groups(self) -> list[dict]:
        return self.opt.param_groups

    def info(self) -> dict:
        n = sum(p.numel() for p in self.params)
        return dict(params=n, tensors=len(self.params), impl="fused" if self.fused else "foreach",
                    pinned=self.pinned, pin_error=self.pin_error, chunks=len(self.chunks),
                    host_buffers_gb=round(sum(c.numel() for c in self.chunks) * 4 / 2**30, 3),
                    host_adam_state_gb=round(n * 8 / 2**30, 3))

    def zero_grad(self, set_to_none: bool = True):
        """The device gradients (the host buffers are overwritten by every step)."""
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        """Device gradients (already clipped) -> host buffers, device gradients freed, AdamW on the masters. A
        parameter without a gradient is skipped, as torch.optim skips it. The device weights change only at push()."""
        for p, m, g in zip(self.params, self.master, self.grads):
            if p.grad is None:
                m.grad = None
            else:
                g.copy_(p.grad, non_blocking=True)
                m.grad = g
        self._sync()  # the host must not read the buffers before the copies land
        self.zero_grad(set_to_none=True)
        self.opt.step()

    @torch.no_grad()
    def push(self):
        """Masters -> device parameters (after step() and the L2-SP pull)."""
        for p, m in zip(self.params, self.master):
            p.copy_(m, non_blocking=True)
        self._sync()  # the next step writes the masters again

    @torch.no_grad()
    def pull(self):
        """Device parameters -> masters."""
        for p, m in zip(self.params, self.master):
            m.copy_(p.detach(), non_blocking=True)
        self._sync()

    def state_dict(self) -> dict:
        return self.opt.state_dict()

    def load_state_dict(self, sd: dict):
        """Call after the model's weights are loaded: the masters are rebuilt from them. Also takes a state saved by
        the on-device optimizer (its fused/foreach flags are replaced by this one's)."""
        self.pull()
        self.opt.load_state_dict(sd)
        for g in self.opt.param_groups:
            g["fused"], g["foreach"] = (True, None) if self.fused else (None, True)


def offloaded(R: Run) -> bool:
    return isinstance(R.opt, CpuOffloadAdamW)


def _weights(R: Run) -> list[torch.Tensor]:
    """The weights the optimizer and L2-SP update: the host masters under offload, else the model's parameters."""
    return R.opt.master if offloaded(R) else R.params


def decays(name: str, p: torch.Tensor) -> bool:
    """Whether optim.weight_decay decays parameter `name`: 2 or more dimensions (matrices, embeddings, convolution
    kernels) and not a bias - the conformer attention's relative-position biases bias_u / bias_v are (heads, head_dim)
    but biases all the same. The rest (1-dim biases, LayerNorm and BatchNorm affine) never decays."""
    return p.ndim >= 2 and not name.rsplit(".", 1)[-1].startswith("bias")


def decay_groups(params: list[torch.Tensor], decay: list[bool], weight_decay: float) -> list[dict]:
    """AdamW param groups for optim.weight_decay: {"decay": True} with the decay over the parameters `decays` picks,
    {"decay": False} at 0 over the rest, each only if it has parameters. The "decay" key goes into the optimizer's
    state_dict with its group, so a resume's --set optim.weight_decay reaches the right one."""
    out = [dict(params=[p for p, d in zip(params, decay) if d], weight_decay=float(weight_decay), decay=True),
           dict(params=[p for p, d in zip(params, decay) if not d], weight_decay=0.0, decay=False)]
    return [g for g in out if g["params"]]


def setup_optim(R: Run):
    """AdamW (on the device, or CpuOffloadAdamW under optim.offload "cpu") and the decoupled L2-SP anchor (not on the
    aux-CTC head: a fresh head has no init worth pulling back to). The param groups, fixed for the whole run
    (st["optim_groups"]): "ndim" - decay_groups, for a run started with optim.weight_decay > 0 - or "single", one group
    of every parameter with optim.weight_decay, which a run started at 0 keeps (as every run before the grouping did:
    a resume's --set optim.weight_decay then decays that group, as before)."""
    o = R.cfg["optim"]
    layout = R.st.get("optim_groups") or ("ndim" if float(o["weight_decay"]) > 0 else "single")
    R.st["optim_groups"] = layout
    decay = [decays(n, p) for n, p in zip(R.param_names, R.params)] if layout == "ndim" else None
    exclude = (AUX_CTC_PREFIX,)
    if o["offload"] == "cpu":
        R.opt = CpuOffloadAdamW(R.params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                weight_decay=o["weight_decay"], fused=bool(o["offload_fused"]), decay=decay)
        R.l2sp = L2SP(zip(R.param_names, R.opt.master), lam=R.cfg["loss"]["l2sp_lambda"],
                      exclude=exclude)  # theta_0 in host memory
        R.log.event("optim_offload", **R.opt.info())
        return
    fused = bool(o["fused"]) and R.device.type == "cuda"
    groups = R.params if decay is None else decay_groups(R.params, decay, o["weight_decay"])
    R.opt = torch.optim.AdamW(groups, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                              weight_decay=o["weight_decay"], fused=fused)
    R.l2sp = L2SP(zip(R.param_names, R.params), lam=R.cfg["loss"]["l2sp_lambda"], exclude=exclude)


def apply_optim_hparams(R: Run):
    """This config's optim.betas / eps / weight_decay onto the optimizer's groups (after a resume restored the saved
    ones): weight_decay on a decaying group ("decay" True, or the "single" layout's one group), 0 on the other."""
    o = R.cfg["optim"]
    for g in R.opt.param_groups:
        g.update(betas=tuple(o["betas"]), eps=float(o["eps"]),
                 weight_decay=float(o["weight_decay"]) if g.get("decay", True) else 0.0)


# ----------------------------------------------------------------------------------------------------------- data


def _seeded_ids(pool: list[str], n: int, rng: np.random.Generator) -> list[str]:
    return pool if len(pool) <= n else sorted(rng.choice(pool, size=n, replace=False).tolist())


def audio_subset(ids: list[str], durations, budget_s: float, rng: np.random.Generator,
                 min_single_s: float = 0.3) -> tuple[list[str], dict]:
    """About `budget_s` seconds of audio out of `ids`: walk them in a seeded order (sorted first, so the draw depends
    only on the rng and the id set, not on file order) and take every utterance that still fits (total <= budget_s).
    If none fits (budget_s below every duration), the single utterance of >= min_single_s (of any length if there
    is none) whose duration is closest to budget_s. Returns (ids in draw order, a record for the log: budget, count,
    total, whether the fallback was used, the ids and their durations)."""
    if not len(ids):
        raise ValueError("audio_subset: no utterances to draw from")
    order = sorted(range(len(ids)), key=ids.__getitem__)
    pool = [ids[i] for i in order]
    dur = [float(durations[i]) for i in order]
    shortest = min(dur)
    pick, total = [], 0.0
    for j in rng.permutation(len(pool)).tolist():
        if total + dur[j] <= budget_s:
            pick.append(j)
            total += dur[j]
            if budget_s - total < shortest:
                break
    fallback = not pick
    if fallback:
        cand = [j for j in range(len(pool)) if dur[j] >= min_single_s] or list(range(len(pool)))
        pick = [min(cand, key=lambda j: (abs(dur[j] - budget_s), j))]
        total = dur[pick[0]]
    return [pool[j] for j in pick], dict(budget_s=float(budget_s), n=len(pick), total_s=round(total, 3),
                                          fallback=fallback, ids=[pool[j] for j in pick],
                                          durations=[round(dur[j], 3) for j in pick])


def _subset_dir(prefix: str, budget_s: float, seed: int, ids: list[str]) -> str:
    """Cache dir name of a duration subset: the id hash keeps configs that draw different ids apart."""
    return f"{prefix}_{budget_s:g}s_s{seed}_{hashlib.sha256(chr(10).join(sorted(ids)).encode()).hexdigest()[:8]}"


def train_store_spec(cfg: dict, log) -> tuple[str, list[str] | None]:
    """The train store setup_data builds: its name under cache_dir and the ids it holds (None: every kept train row of
    cfg["sources"]). subset.train_audio_s draws a seeded duration subset (a `subset` event with every id; the name
    carries the ids' hash), subset.train_utts a seeded one that starts with probe rows. scripts/05_evaluate.py takes
    its probe rows from the same ids and packs them into TOKEN stores named after this name. Family "ctc": a duration
    subset draws from every row (the frame planner excludes none), and the frame store lives under
    store_dir(cfg, name) = cache_dir/"ctc_"<name>, so a token store built under the plain name (05's) never touches
    it."""
    sel, teach = rpath(cfg["selection"]), rpath(cfg["teacher_root"])
    sub, seed = cfg["subset"], int(cfg["seed"])
    if sub["train_audio_s"] is not None:
        rows = trainset.read_selection(sel, cfg["sources"], ["train"])
        if not is_ctc(cfg):
            prompt_len = len(trainset._teacher_meta(teach)["prompt"])
            rows = rows[rows["n_tok"] + prompt_len - 1 <= int(cfg["batch"]["max_dec_len"])]  # what the planner can use
        ids, rec = audio_subset(rows["id"].tolist(), rows["duration"].to_numpy(), float(sub["train_audio_s"]),
                                np.random.default_rng([seed, 11]))
        log.event("subset", split="train", sources=cfg["sources"], seed=seed, pool=len(rows), **rec)
        return _subset_dir("train", rec["budget_s"], seed, ids), ids
    if sub["train_utts"]:
        n = int(sub["train_utts"])
        rows = trainset.read_selection(sel, cfg["sources"], ["train"])
        rng = np.random.default_rng([seed, 1])
        probe = rows["id"][rows["in_probe"]].tolist()[: max(1, n // 4)]
        rest = _seeded_ids(rows["id"][~rows["id"].isin(probe)].tolist(), n - len(probe), rng)
        return f"train_sub{n}_s{seed}", probe + rest
    return "train", None


def eval_store_spec(cfg: dict, log) -> tuple[str, list[str] | None]:
    """The eval store's name under cache_dir and the ids it holds (None: every kept row of cfg["eval_sets"]): the
    seeded subset subset.eval_audio_s (pooled over the sets; a `subset` event) or subset.eval_utts_per_set asks for.
    Both families' eval stores of a config hold these rows (the frame store under store_dir's "ctc_" name)."""
    sel = rpath(cfg["selection"])
    sub, seed = cfg["subset"], int(cfg["seed"])
    if sub["eval_audio_s"] is not None:  # pooled over the eval sets
        rows = trainset.read_selection(sel, cfg["eval_sets"], ["eval"])
        ids, rec = audio_subset(rows["id"].tolist(), rows["duration"].to_numpy(), float(sub["eval_audio_s"]),
                                np.random.default_rng([seed, 12]))
        log.event("subset", split="eval", sources=cfg["eval_sets"], seed=seed, pool=len(rows), **rec)
        return _subset_dir("eval", rec["budget_s"], seed, ids), ids
    if sub["eval_utts_per_set"]:
        n = int(sub["eval_utts_per_set"])
        rows = trainset.read_selection(sel, cfg["eval_sets"], ["eval"])
        rng = np.random.default_rng([seed, 2])
        ids = []
        for s in cfg["eval_sets"]:
            r = rows[rows["source"] == s]
            greedy = _seeded_ids(r["id"][r["in_greedy_subset"]].tolist(), min(n, int(cfg["eval"]["greedy_subset"])), rng)
            ids += greedy + _seeded_ids(r["id"][~r["id"].isin(greedy)].tolist(), max(0, n - len(greedy)), rng)
        return f"eval_sub{n}_s{seed}", ids
    return "eval", None


FRAME_STORE_PREFIX = "ctc_"  # a frame store's cache dir: this + the token store's name (store_dir)


def store_dir(cfg: dict, name: str, frames: bool | None = None) -> Path:
    """The cache dir of the store `name` (train_store_spec's / eval_store_spec's): cache_dir/<name> for a token store,
    cache_dir/ctc_<name> for a frame store (frames None: the run's family's). The two kinds of one config's rows never
    share a directory: they share cache_dir on the study box, and a build of the other kind would replace the files."""
    frames = is_ctc(cfg) if frames is None else frames
    return rpath(cfg["cache_dir"]) / ((FRAME_STORE_PREFIX if frames else "") + name)


def _store(cfg: dict, name: str, sources: list[str], split: str, ids: list[str] | None = None, log=None, *,
           frames: bool) -> trainset.Stores:
    """One store: a token store (kitsune.trainset.build_stores, teacher_root) or a frame store with its frame preflight
    (build_frame_stores, parakeet_root; family "ctc" only), under store_dir. A preflight that fails the build is logged
    as a `frame_preflight` event (ok false, its counts) before it stops the run."""
    sel, data = rpath(cfg["selection"]), rpath(cfg["data_root"])
    where = store_dir(cfg, name, frames)
    if not frames:
        return trainset.build_stores(sel, data, rpath(cfg["teacher_root"]), where, sources, [split], ids=ids, log=print)
    if not is_ctc(cfg):
        raise ValueError(f"a frame store is family ctc's: this config's family is {family(cfg)!r}")
    try:
        return trainset.build_frame_stores(sel, data, rpath(cfg["parakeet_root"]), where, sources, [split], ids=ids,
                                           log=print)
    except trainset.FramePreflightFailed as e:
        if log is not None:
            log.event("frame_preflight", store=split, **e.report)
        raise


def log_frame_preflight(log, split: str, store: trainset.Stores):
    """The `frame_preflight` event of a frame store (its build's report, stores.json's; `reused` when this run did not
    build it): decision 15's counts - rows checked, mismatches dropped per split, undecodable rows, the train share
    against its limit."""
    rep = dict(store.info.get("frame_preflight") or {})
    rep.pop("mismatches", None)  # the dropped ids are in `dropped` (and every one in stores.json)
    log.event("frame_preflight", store=split, reused=bool(store.info.get("reused")), built=store.info.get("created"),
              **rep)


def build_train_store(cfg: dict, log) -> trainset.Stores:
    """The train store of cfg["sources"] (cached under cache_dir): train_store_spec's rows, of the run's family (family
    "ctc": the frame store, under store_dir). setup_data's; the box's store step builds it ahead of the runs."""
    name, ids = train_store_spec(cfg, log)
    return _store(cfg, name, cfg["sources"], "train", ids=ids, log=log, frames=is_ctc(cfg))


def build_eval_store(cfg: dict, log, frames: bool | None = None) -> trainset.Stores:
    """The eval store of cfg["eval_sets"] (cached under cache_dir, shared with 03's eval cache): eval_store_spec's rows.
      frames None  (the default: scripts/05_evaluate.py's and the box's store step, kitsune/study_queue.py) the TOKEN
                   store (teacher_root), the one the evaluator scores every system on, of either family and in either
                   evaluator path (its CTC eval takes the frame targets from parakeet_out itself), and the speed
                   probe's cache/eval. For a family-ctc config the trainer's frame store of the same rows is built (or
                   reused) first, so this one call pre-builds every eval store the config's runs and evals open: a box
                   must never leave it to runs that start in parallel (a store build takes no lock).
      frames True  the frame store alone, with its frame preflight (family ctc: setup_data's)
      frames False the token store alone"""
    name, ids = eval_store_spec(cfg, log)
    if frames is None and is_ctc(cfg):
        _store(cfg, name, cfg["eval_sets"], "eval", ids=ids, log=log, frames=True)
    return _store(cfg, name, cfg["eval_sets"], "eval", ids=ids, log=log, frames=bool(frames))


def dev_on(cfg: dict) -> bool:
    """The dev-slice eval runs (eval.dev.every_epochs or every_steps set; False for a config without the block: a unit
    test's partial config, a state from before it)."""
    dev = (cfg.get("eval") or {}).get("dev") or {}
    return bool(dev.get("every_epochs") or dev.get("every_steps"))


def dev_metric(cfg: dict) -> bool:
    """The early stop reads the dev evals (a DEV_METRICS metric): checked after each dev eval, not after the full
    evals."""
    return (cfg.get("early_stop") or {}).get("metric") in DEV_METRICS


def _ids_sha256(ids) -> str:
    """sha256 of the sorted ids joined by newlines (the dev pick's identity: dev_store_name's sha8, st["dev"]). Not
    the selection sidecar's digest, which hashes the ids in selection order (dev_ids' ids_sha256_selection)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def dev_ids(cfg: dict, log) -> tuple[list[str], dict]:
    """The dev rows the dev eval scores: kitsune.fullrun.dev_pick over the selection's kept split-"dev" rows of
    cfg["sources"] (a seeded eval.dev.per_source per source, all of a smaller one; seed eval.dev.seed, else the run's),
    in selection order. A `subset` event (split "dev") gives the counts and both digests; the ids go to dev_ids.json
    (setup_data). Returns (ids, record: seed, per_source, n, ids_sha256, ids_sha256_selection, pool and picked per
    source, ids per source). The record holds two digests of the same ids:
      ids_sha256            _ids_sha256, over the SORTED ids: the pick's identity (the store name's sha8, st["dev"]
                            and the resume guard stay on it)
      ids_sha256_selection  kitsune.store.ids_sha256 over the ids in selection order, the digest make_selection.py
                            full mode records for its lists: it equals the selection sidecar's
                            dev.scored_default.ids_sha256 when per_source, seed and cfg["sources"] are the sidecar's
                            (per_source 600, seed 1234, every source of the selection), so a run's dev slice can be
                            checked against the sidecar by hash (the sorted digest would never match it)
    A selection without dev rows for a source stops the run: a full selection (make_selection.py full mode) has
    them."""
    from kitsune import fullrun
    from kitsune.store import ids_sha256

    dev = cfg["eval"]["dev"]
    sel = rpath(cfg["selection"])
    per = int(dev["per_source"])
    seed = int(cfg["seed"] if dev["seed"] is None else dev["seed"])
    rows = trainset.read_selection(sel, cfg["sources"], [fullrun.DEV_SPLIT])
    try:
        ids = fullrun.dev_pick(zip(rows["id"].tolist(), rows["source"].tolist()), cfg["sources"], per, seed)
    except ValueError as e:
        raise SystemExit(f"eval.dev needs kept split '{fullrun.DEV_SPLIT}' rows of every source in {sel}: {e}"
                         ) from None
    src_of = dict(zip(rows["id"].tolist(), rows["source"].tolist()))
    by_src = {s: [i for i in ids if src_of[i] == s] for s in cfg["sources"]}
    rec = dict(seed=seed, per_source=per, n=len(ids), ids_sha256=_ids_sha256(ids),
               ids_sha256_selection=ids_sha256(ids),  # ids in selection order, as dev_pick returns them
               pool={s: int((rows["source"] == s).sum()) for s in cfg["sources"]},
               picked={s: len(v) for s, v in by_src.items()})
    log.event("subset", split=fullrun.DEV_SPLIT, sources=cfg["sources"], **rec)
    return ids, dict(rec, ids=by_src)


def dev_store_name(rec: dict) -> str:
    """The dev store's cache name (store_dir adds "ctc_" for a frame store): per_source, seed and the ids' hash, so
    configs that pick other rows never share one."""
    return f"dev_p{rec['per_source']}_s{rec['seed']}_{rec['ids_sha256'][:8]}"


def build_dev_store(cfg: dict, log) -> trainset.Stores | None:
    """The dev store (None unless dev_on): dev_ids' rows of cfg["sources"], split "dev", of the run's family (a frame
    store with its frame preflight for family ctc; the dev split reads the audio of its rows' train shards,
    kitsune.trainset), built or reused under store_dir(cfg, dev_store_name). It writes no file of its own and needs no
    run dir: the box's store step (kitsune.full_queue build-stores) calls it next to build_train_store and
    build_eval_store with a logger that has .event() only. The pick's record (dev_ids) rides along in the store's
    in-memory info["dev_pick"], for setup_data's dev_ids.json and `dev_store` event."""
    if not dev_on(cfg):
        return None
    ids, rec = dev_ids(cfg, log)
    from kitsune import fullrun

    store = _store(cfg, dev_store_name(rec), cfg["sources"], fullrun.DEV_SPLIT, ids=ids, log=log, frames=is_ctc(cfg))
    store.info["dev_pick"] = rec  # in memory only, as "reused"
    return store


def probe_greedy_subset(cfg: dict, probe_ids: list[str], duration: dict[str, float], log) -> list[str]:
    """The probe rows greedy-decoded at every full eval (eval.probe_greedy_audio_s, else subset.eval_audio_s: a
    seeded ~N s of the probe; none if both are null or the probe is empty), logged as a `subset` event. duration: id ->
    seconds of every probe row."""
    pg_budget = cfg["eval"]["probe_greedy_audio_s"]
    pg_budget = cfg["subset"]["eval_audio_s"] if pg_budget is None else pg_budget
    if pg_budget is None or not probe_ids:
        return []
    seed = int(cfg["seed"])
    ids, rec = audio_subset(probe_ids, [duration[i] for i in probe_ids], float(pg_budget),
                            np.random.default_rng([seed, 13]))
    log.event("subset", split="probe_greedy", seed=seed, pool=len(probe_ids), **rec)
    return ids


def greedy_subset_ids(cfg: dict, evalstore) -> list[str]:
    """The fixed greedy subset every full eval decodes (the complete-set evals take its summary from their rows): per
    eval set its in_greedy_subset rows (all of a set without them), a seeded eval.greedy_subset of them if more; the
    whole eval subset under subset.eval_audio_s."""
    if cfg["subset"]["eval_audio_s"] is not None:
        return [u.id for u in evalstore.utts]  # the whole (small) eval subset, at every eval
    rng = np.random.default_rng([int(cfg["seed"]), 3])
    ids = []
    for s in cfg["eval_sets"]:
        cand = evalstore.indices(source=s, in_greedy_subset=True) or evalstore.indices(source=s)
        pick = cand if len(cand) <= cfg["eval"]["greedy_subset"] else sorted(
            rng.choice(cand, size=int(cfg["eval"]["greedy_subset"]), replace=False).tolist())
        ids += [evalstore.utts[i].id for i in pick]
    return ids


def setup_data(R: Run):
    """Train/eval stores (cached under cache_dir, shared with 03's eval cache), the dev store under eval.dev
    (setup_dev), probe and greedy ids, planner. The store opens and builds run under a bounded heartbeat
    (kitsune.heartbeat.beating: a full-extent build takes hours; a no-op without $KITSUNE_HEARTBEAT); the `data` event
    says which stores this run reused (stores_reused: a frame store records it, a token store never does)."""
    from kitsune import heartbeat

    cfg, log = R.cfg, R.log
    t0 = time.time()
    with heartbeat.beating(max_s=36000):
        R.train = build_train_store(cfg, log)
        if is_ctc(cfg):  # decision 15's counts of each store (a reused cache's too), before the next one is built
            log_frame_preflight(log, "train", R.train)
        R.evalstore = build_eval_store(cfg, log, frames=is_ctc(cfg))  # never the token store a ctc run does not read
        if is_ctc(cfg):
            log_frame_preflight(log, "eval", R.evalstore)
        R.devstore = build_dev_store(cfg, log)
    dev = setup_dev(R) if getattr(R, "devstore", None) is not None else {}

    if cfg["eval"]["probe"] and cfg["eval"]["probe_is_train"]:
        R.probe_ids = [u.id for u in R.train.utts]
    else:
        R.probe_ids = [R.train.utts[i].id for i in R.train.indices(in_probe=True)] if cfg["eval"]["probe"] else []
    es = cfg["early_stop"]
    if es["enabled"] and es["metric"] == "probe_kl" and not R.probe_ids:  # validate() cannot see an empty probe
        raise SystemExit("early_stop.metric probe_kl needs a non-empty probe: this selection/subset has no probe rows "
                         "(set eval.probe_is_train, or use metric heldout_kl / train_loss)")
    # greedy CER on the train data itself, next to the held-out one
    R.probe_greedy_ids = probe_greedy_subset(cfg, R.probe_ids, {u.id: u.duration for u in R.train.utts}, log)
    R.greedy_ids = greedy_subset_ids(cfg, R.evalstore)
    R.mini_val_ids, R.mini_train_ids = mini_subsets(R)
    R.ds = trainset.dataset_for(R.train)  # FrameBatchDataset for a frame store (family "ctc")
    R.train_ds = setup_augment(R)  # the train loader's: R.ds, or its augmenting copy (augment.enabled)
    R.src_index = {s: i for i, s in enumerate(sorted({u.source for u in R.train.utts}))}
    log.event("data", train_utts=len(R.train), train_h=round(R.train.hours, 3),
              per_source=R.train.info.get("per_source"), dropped=R.train.info.get("dropped"),
              eval_utts=len(R.evalstore), eval_h=round(R.evalstore.hours, 3),
              eval_per_set=R.evalstore.info.get("per_source"), eval_dropped=R.evalstore.info.get("dropped"),
              probe=len(R.probe_ids), greedy=len(R.greedy_ids),
              build_s=round(time.time() - t0, 1),
              **(dict(probe_greedy=len(R.probe_greedy_ids)) if R.probe_greedy_ids else {}),
              **(dict(mini_val=len(R.mini_val_ids), mini_train=len(R.mini_train_ids))
                 if cfg["eval"]["mini"]["every_steps"] else {}),
              **dev, stores_reused={k: bool(s.info.get("reused")) for k, s in (
                  ("train", R.train), ("eval", R.evalstore), ("dev", getattr(R, "devstore", None))) if s is not None})


def setup_dev(R: Run) -> dict:
    """setup_data's part for the dev store (eval.dev on): runs/<run_id>/dev_ids.json {seed, per_source, ids_sha256,
    ids_sha256_selection, ids {source: [...]}} (informational: the ids are a pure function of the selection,
    per_source and the seed; ids_sha256_selection is the digest to compare with the selection sidecar's
    dev.scored_default.ids_sha256, dev_ids), the frame preflight's `frame_preflight` event (family ctc), the
    `dev_store` event {name, n, per_source, seed, ids_sha256, ids_sha256_selection, hours, n_in_train} and the guard:
    the dev rows are a train row nowhere (n_in_train must be 0), and a resume scores the rows its state scored
    (st["dev"], on the sorted ids_sha256; eval.dev.per_source / seed are RESUME_FIXED, the selection is too, so only a
    changed file or dev_pick could move them). The store's own rows count too (n, store_ids_sha256): a store rebuilt
    on a new host can hold fewer of the picked rows (a shard missing from the pull drops its rows, the frame preflight
    drops some), and the early stop's smoothing and patience must not mix numbers over two row sets; a state from
    before those keys takes them from this store. Returns the `data` event's dev_utts, dev_h and dev_per_source."""
    cfg, log, store = R.cfg, R.log, getattr(R, "devstore", None)
    rec = store.info["dev_pick"]
    if is_ctc(cfg):
        log_frame_preflight(log, "dev", store)
    ids = [i for v in rec["ids"].values() for i in v]
    n_in_train = len(set(ids) & {u.id for u in R.train.utts})
    info = dict(name=store.cache_dir.name, n=len(store), n_ids=rec["n"], per_source=rec["per_source"],
                seed=rec["seed"], ids_sha256=rec["ids_sha256"], ids_sha256_selection=rec["ids_sha256_selection"],
                hours=round(store.hours, 4), n_in_train=n_in_train)
    log.event("dev_store", **info)
    if n_in_train:
        raise SystemExit(f"{n_in_train} of the dev rows are in the train store too ({cfg['selection']}): the early "
                         "stop would read rows the run trains on")
    held = dict(n=len(store), store_ids_sha256=_ids_sha256(u.id for u in store.utts))
    want = dict(ids_sha256=rec["ids_sha256"], n_ids=rec["n"], per_source=rec["per_source"], **held)
    prev = R.st.get("dev")
    if prev is None:
        R.st["dev"] = want
    elif prev["ids_sha256"] != want["ids_sha256"]:
        raise SystemExit(f"the dev rows changed: this run's state scored {prev} and the selection now gives "
                         f"{want} (the selection file or eval.dev differs from the run's)")
    elif any(k in prev and prev[k] != v for k, v in held.items()):
        raise SystemExit(f"the dev store {store.cache_dir} holds {held['n']} rows (store_ids_sha256 "
                         f"{held['store_ids_sha256'][:12]}), this run's state scored {prev.get('n')} "
                         f"({str(prev.get('store_ids_sha256'))[:12]}) of the same {rec['n']} picked rows: a shard "
                         "missing from this host's data, or other audio, left other rows in it")
    else:
        prev.update(held)  # a state from before these keys: this store's rows from now on
    path = R.run_dir / "dev_ids.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(seed=rec["seed"], per_source=rec["per_source"], ids_sha256=rec["ids_sha256"],
                                   ids_sha256_selection=rec["ids_sha256_selection"], ids=rec["ids"]), indent=1) + "\n",
                   encoding="utf-8")
    _replace_file(tmp, path)
    return dict(dev_utts=len(store), dev_h=round(store.hours, 3), dev_per_source=store.info.get("per_source"))


def setup_augment(R: Run):
    """The train loader's dataset (loop's make_loader): R.ds itself, or under augment.enabled a copy of it sharing its
    arrays (FrameBatchDataset.with_augment) that augments every micro-batch (kitsune.trainset.Augment; the module
    docstring's "Train-data augmentation"). R.ds stays plain for the smoke checks, the memory probe and the FLOP count.
    seed: augment.seed, else the run's. concat_max_s is clamped to the longest train utterance (stored duration): a
    joined row is then never longer than the row the memory probe's longest micro-batch ran, and its micro-batch never
    has more padded frames than the planned one, so the probe's peak still bounds every joined micro-batch. An
    `augment` event (and a console line) records the settings in use at every start, a resume's too."""
    cfg = R.cfg
    if not augment_on(cfg):
        return R.ds
    a = cfg["augment"]
    seed = int(cfg["seed"]) if a["seed"] is None else int(a["seed"])
    longest = float(np.max(R.ds.duration)) if len(R.ds) else 0.0  # the stored durations, as the planner packs them
    used = min(float(a["concat_max_s"]), longest) if longest > 0 else float(a["concat_max_s"])
    punct = punct_token_ids(R.tokenizer)
    if float(a["truncate_p"]) > 0 and not any(t in SENTENCE_MARKS for t in punct.values()):
        raise SystemExit(f"augment.truncate_p {a['truncate_p']} needs the student tokenizer's sentence marks "
                         f"({''.join(SENTENCE_MARKS)}) to keep every cut inside a sentence, and this tokenizer has none "
                         f"(punctuation found: {punct})")
    spec = trainset.Augment.from_config(a, seed=seed, concat_max_s=used, punct_ids=sorted(punct))
    R.log.event("augment", seed=seed, truncate_p=spec.truncate_p, truncate_min_frac=spec.truncate_min_frac,
                truncate_min_s=spec.truncate_min_s, truncate_pause_p=spec.truncate_pause_p,
                punct_ids={str(i): t for i, t in sorted(punct.items())}, concat_p=spec.concat_p,
                concat_max_n=spec.concat_max_n, concat_max_s=used, concat_max_s_config=float(a["concat_max_s"]),
                longest_train_s=round(longest, 3), concat_max_s_clamped=used < float(a["concat_max_s"]),
                mix_p=spec.mix_p, mix_snr_db=list(spec.mix_snr_db), truncate_pad_p=spec.truncate_pad_p,
                end_pad_p=spec.end_pad_p, pad_frames=list(spec.pad_frames))
    print(f"augment: truncate_p {spec.truncate_p:g} (pause cuts {spec.truncate_pause_p:g}, never removing only "
          f"{''.join(punct.values()) or 'punctuation'}), concat_p {spec.concat_p:g} (k <= {spec.concat_max_n}, "
          f"<= {used:g} s{' = the longest train utterance' if used < float(a['concat_max_s']) else ''}), mix_p "
          f"{spec.mix_p:g} at {spec.mix_snr_db[0]:g}-{spec.mix_snr_db[1]:g} dB, quiet pads after cuts "
          f"{spec.truncate_pad_p:g} / after whole rows {spec.end_pad_p:g} ({spec.pad_frames[0]}-{spec.pad_frames[1]} "
          f"frames), seed {seed}", flush=True)
    return R.ds.with_augment(spec)


# the text of the tokens truncate must never remove alone (punct_token_ids): sentence marks and commas, full- and
# half-width (the Parakeet ja vocabulary has 。 ? ! and 、 - ids 1, 25, 27 and 8 in tokenizer.json f604c293...)
SENTENCE_MARKS = ("。", "?", "!", "？", "！", ".", "．")
PUNCT_TEXT = (*SENTENCE_MARKS, "、", ",", "，")


def punct_token_ids(tokenizer) -> dict[int, str]:
    """{token id: its text} of the vocabulary's punctuation tokens (PUNCT_TEXT, with the word-boundary marker "▁"
    stripped): the tokens augment's truncate never removes alone, so that a cut never keeps a complete sentence without
    its mark (kitsune.trainset.truncate_frames). Read from the student's own tokenizer, so a vocabulary with other ids
    (the tests' tiny one has only 。, at id 1) gets its own; the `augment` event records them."""
    vocab = tokenizer.get_vocab()
    return {int(i): t.replace("▁", "") for t, i in vocab.items()
            if t.replace("▁", "") and t.replace("▁", "") in PUNCT_TEXT}


def batch_utts(mb: dict) -> int:
    """The utterances a micro-batch holds: one per row, except under augment.enabled, where a joined row holds several
    and the dataset counts them (its `aug`). The smoke's dropped-audio share is per utterance either way."""
    aug = mb.get("aug")
    return int(aug["utts"]) if aug else len(mb["ids"])


def mini_subsets(R: Run) -> tuple[list[str], list[str]]:
    """The mini eval's fixed subsets (eval.mini; empty when mini.every_steps is null), drawn once from the seed and the
    data, so every mini eval of a run - resumes included - scores the same utterances:
      val    mini.val_per_set seeded utterances of each GATE set the run evaluates (all of a smaller set; every eval set
             if none is a gate set), or the whole eval subset when subset.eval_audio_s makes it a small pooled one
             (the overfit runs)
      train  mini.train_utts seeded utterances of the train probe (the train set if there is no probe)
    Each is logged as a `subset` event (split mini_val / mini_train) with its ids."""
    cfg, seed = R.cfg, int(R.cfg["seed"])
    mc = cfg["eval"]["mini"]
    if not mc["every_steps"]:
        return [], []
    from kitsune.evaluate import GATE_SETS

    if cfg["subset"]["eval_audio_s"] is not None:
        val = [u.id for u in R.evalstore.utts]
    else:
        rng = np.random.default_rng([seed, 14])
        sets = [s for s in cfg["eval_sets"] if s in GATE_SETS] or list(cfg["eval_sets"])
        val = []
        for s in sets:
            val += _seeded_ids([R.evalstore.utts[i].id for i in R.evalstore.indices(source=s)],
                               int(mc["val_per_set"]), rng)
    pool = R.probe_ids or [u.id for u in R.train.utts]
    train = list(_seeded_ids(pool, int(mc["train_utts"]), np.random.default_rng([seed, 15])))
    for split, ids, store in (("mini_val", val, R.evalstore), ("mini_train", train, R.train)):
        chosen = set(ids)
        utts = [u for u in store.utts if u.id in chosen]
        per = {}
        for u in utts:
            per[u.source] = per.get(u.source, 0) + 1
        R.log.event("subset", split=split, seed=seed, n=len(ids), total_s=round(sum(u.duration for u in utts), 3),
                    per_source=per, ids=ids)
    return val, train


def make_planner(R: Run, micro_audio_s: float) -> trainset.StepPlanner:
    """The step planner over the train store; family "ctc" plans frames (max_dec_len None: no decoder to bound,
    micro-batches cut by padded audio alone; batch.max_dec_len is unused)."""
    b, mix = R.cfg["batch"], R.cfg["mix"]
    weights = None if mix in (None, "natural") else mix
    return trainset.StepPlanner(R.train.utts, step_audio_s=b["step_audio_s"], micro_audio_s=micro_audio_s,
                                max_dec_len=None if is_ctc(R.cfg) else b["max_dec_len"], pool_micro=b["pool_micro"],
                                seed=int(R.cfg["seed"]), weights=weights,
                                prompt_len=len(R.train.info.get("prompt", trainset.PROMPT)))


def plan_epochs(R: Run):
    """Clock "epochs": T = the optimizer steps of schedule.epochs full passes, summed over the planner's epoch plans
    (the step count of an epoch can differ by one between epochs). Computed once, kept in the full state."""
    sch = R.cfg["schedule"]
    if sch["clock"] != "epochs" or R.st.get("total_steps"):
        return
    per = [len(R.planner.epoch_plan(e)) for e in range(int(sch["epochs"]))]
    R.st["total_steps"] = sum(per)
    R.log.event("schedule", clock="epochs", epochs=len(per), total_steps=sum(per), steps_per_epoch=per,
                warmup_steps=warmup_steps(sch, sum(per)),
                first_cooldown_step=math.ceil((1.0 - float(sch["cooldown_frac"])) * sum(per)) + 1)


# -------------------------------------------------------------------------------------------------- forward pass


def forward_logits(R: Run, mb: dict, featurizer, gen=None, capture: dict | None = None, with_encoder: bool = False):
    """Student logits (N, V) at the micro-batch's target positions: LogMel (fp32), SpecAugment if `gen`, the body
    under autocast, the head in fp32 outside it. Returns (logits, tgt_row, masked_frac or None), and with_encoder
    (the aux-CTC loss) a 4th item: (the encoder's final hidden states (B, T_enc, d_enc), before the decoder's
    projection; their valid frames per row (B,))."""
    dev = R.device
    feats, fmask = featurizer(mb["wave"].to(dev, non_blocking=True), mb["lengths"].to(dev, non_blocking=True))
    mfrac = None
    if gen is not None:
        feats, mfrac = R.specaug(feats, fmask, gen)
    hooks = []
    if capture is not None:
        for name, mod in capture.pop("_modules", []):
            hooks.append(mod.register_forward_hook(_capture_hook(capture, name)))
    try:
        with R.autocast():
            out = R.model.model(input_features=feats, attention_mask=fmask,
                                decoder_input_ids=mb["decoder_input_ids"].to(dev, non_blocking=True),
                                decoder_attention_mask=mb["dec_mask"].to(dev, non_blocking=True),
                                use_cache=False)
            h = out.last_hidden_state
    finally:
        for hk in hooks:
            hk.remove()
    rows = mb["tgt_row"].to(dev, non_blocking=True)
    head = R.model.proj_out
    with torch.autocast(device_type=dev.type, enabled=False):
        logits = F.linear(h[rows, mb["tgt_pos"].to(dev, non_blocking=True)].float(), head.weight.float(),
                          head.bias.float() if head.bias is not None else None)
    if not with_encoder:
        return logits, rows, mfrac
    enc = out.encoder_last_hidden_state
    enc_len = R.model.model.encoder._get_subsampling_output_length(fmask.sum(-1)).clamp(max=enc.shape[1])
    return logits, rows, mfrac, (enc, enc_len)


def forward_ctc(R: Run, mb: dict, featurizer, gen=None, capture: dict | None = None):
    """Family "ctc": the student's frame log-probs of a FrameBatchDataset micro-batch - LogMel (fp32), SpecAugment if
    `gen` (masks only: nothing may change the frame count), the encoder under autocast, the CTC head in fp32 outside it,
    log-softmax in fp32 (kitsune.ctc_student.ctc_log_probs). Returns (log_probs (B, T, V), the student's valid frames
    per row (B,), masked_frac or None). T may exceed the batch's longest n_frames by the padding's frame; the loss cuts
    it (kitsune.ctc_kd)."""
    from kitsune.ctc_student import ctc_log_probs

    dev = R.device
    feats, fmask = featurizer(mb["wave"].to(dev, non_blocking=True), mb["lengths"].to(dev, non_blocking=True))
    mfrac = None
    if gen is not None:
        feats, mfrac = R.specaug(feats, fmask, gen)
    hooks = []
    if capture is not None:
        for name, mod in capture.pop("_modules", []):
            hooks.append(mod.register_forward_hook(_capture_hook(capture, name)))
    try:
        with R.autocast():
            lp, n_frames = ctc_log_probs(R.model, feats, fmask)
    finally:
        for hk in hooks:
            hk.remove()
    return lp, n_frames, mfrac


def ctc_frame_batch(mb: dict) -> dict:
    """The frame targets of a FrameBatchDataset micro-batch, as kitsune.ctc_kd.ctc_kd_losses reads them (it moves them
    to the log-probs' device)."""
    from kitsune.ctc_eval import frame_batch

    return frame_batch(mb)


def aux_ctc_sum(R: Run, enc: tuple, mb: dict) -> torch.Tensor:
    """The aux-CTC loss of one micro-batch, summed over its rows (kitsune.kd.aux_ctc_loss): R.aux_ctc on the encoder
    output of forward_logits(with_encoder=True), in fp32 outside autocast as the LM head, blank = the head's last class,
    targets = the teacher's greedy ids without EOS (aux_ctc_targets). The caller divides by the step's target tokens."""
    h, enc_len = enc
    dev, head = R.device, R.aux_ctc
    store = getattr(R, "train", None)
    eos = int(store.info.get("eos", trainset.EOS)) if store is not None else trainset.EOS
    targets, target_len = aux_ctc_targets(mb["top_idx"][:, 0].to(dev, non_blocking=True),
                                          mb["tgt_row"].to(dev, non_blocking=True), len(mb["ids"]), eos)
    with torch.autocast(device_type=dev.type, enabled=False):
        logits = F.linear(h.float(), head.weight.float(), head.bias.float())
    return aux_ctc_loss(logits, enc_len, targets, target_len, blank=head.out_features - 1)


def _capture_hook(store: dict, name: str):
    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
        if isinstance(t, torch.Tensor):
            t = t.detach().flatten()
            store[name] = t[:: max(1, math.ceil(t.numel() / 1_000_000))].float().clone()
    return hook


def activation_modules(model) -> list[tuple[str, torch.nn.Module]]:
    if not hasattr(model, "model"):  # a ParakeetForCTC (family "ctc"): the encoder only
        enc = model.encoder.layers
        return [("encoder.subsampling", model.encoder.subsampling)] + [
            (f"encoder.layers.{i}", enc[i]) for i in sorted({0, len(enc) // 2, len(enc) - 1})]
    enc, dec = model.model.encoder.layers, model.model.decoder.layers
    out = [("encoder.subsampling", model.model.encoder.subsampling)]
    out += [(f"encoder.layers.{i}", enc[i]) for i in sorted({0, len(enc) // 2, len(enc) - 1})]
    out += [(f"decoder.layers.{i}", dec[i]) for i in sorted({0, len(dec) - 1})]
    return out


# ---------------------------------------------------------------------------------------------------- train step


def _sq_norms(tensors: list[torch.Tensor]) -> list[float]:
    """Squared L2 norm of each tensor, with one host transfer for all of them."""
    return torch.stack([n.float() for n in torch._foreach_norm(tensors)]).square().tolist()


def _by_module(R: Run, values: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for n, v in zip(R.param_names, values):
        out[module_key(n)] = out.get(module_key(n), 0.0) + v
    return out


def _grad_sq_by_module(R: Run) -> dict[str, float]:
    return _by_module(R, _sq_norms([p.grad if p.grad is not None else torch.zeros((), device=p.device)
                                    for p in R.params]))


@torch.no_grad()
def _put_back(saved: list[tuple[torch.Tensor, torch.Tensor]]):
    """Copy each saved tensor back into its buffer (train_step's BN running stats, when it skips a step)."""
    for buf, keep in saved:
        buf.copy_(keep)


def optimizer_update(R: Run, step: int, lr: float, stats_step: bool, hist_step: bool, capture: dict | None,
                     bn_keep: list, nonfinite_fields) -> dict | None:
    """The part of an optimizer step after its backward passes, the same for both families: per-module gradient norms
    and gradient histograms (stats / hist steps), clipping (the pre-clip norm), AdamW, the decoupled L2-SP pull (on the
    host masters and pushed under offload), update ratios, weight and activation histograms and the BN drift check,
    then the gradients freed. A non-finite gradient norm skips the step instead: gradients dropped, the BN running
    stats put back (bn_keep), a `nonfinite_grad_skipped` event with nonfinite_fields()'s culprit fields, and
    FloatingPointError after optim.max_nonfinite_skips in a row; returns None. Else the start of the step's
    statistics (grad_norm and clip coefficient are the caller's to add)."""
    cfg = R.cfg
    out: dict = {}
    if stats_step:
        out["grad_sq_by_module"] = _grad_sq_by_module(R)
        before = [p.detach().clone() for p in _weights(R)]  # offload: the host masters, not a second model on the GPU
    if hist_step:
        _grad_hists(R, step)
    clip = cfg["optim"]["clip"]
    gnorm = torch.nn.utils.clip_grad_norm_(R.params, clip if clip else float("inf"), foreach=True)
    gnorm = float(gnorm)
    if not math.isfinite(gnorm):
        R.opt.zero_grad(set_to_none=True)
        _put_back(bn_keep)
        R.st["nonfinite_skips"] += 1
        R.st["nonfinite_total"] += 1
        R.log.event("nonfinite_grad_skipped", at_step=step, grad_norm=gnorm, consecutive=R.st["nonfinite_skips"],
                    **nonfinite_fields(), **({"bn_stats_restored": True} if bn_keep else {}))
        if R.st["nonfinite_skips"] > int(cfg["optim"]["max_nonfinite_skips"]):
            raise FloatingPointError(f"{R.st['nonfinite_skips']} consecutive steps with a non-finite gradient norm")
        return None
    R.st["nonfinite_skips"] = 0
    R.opt.step()
    if offloaded(R):  # L2-SP on the host masters, its value from the same pass (a second one costs ~1 s), then push
        out["l2sp"] = float(R.l2sp.apply_(lr, value=True))
        R.opt.push()
    else:
        R.l2sp.apply_(lr)
    if stats_step:  # ||delta theta|| / ||theta|| per module, the L2-SP pull included
        diffs = torch._foreach_sub([p.detach() for p in _weights(R)], before)
        dn = _by_module(R, _sq_norms(diffs))
        del diffs, before
        pn = _by_module(R, _sq_norms([p.detach() for p in _weights(R)]))
        out["update_ratio"] = {k: math.sqrt(dn[k] / pn[k]) if pn[k] > 0 else float("nan") for k in dn}
    if hist_step:
        _weight_hists(R, step)
        for name, t in (capture or {}).items():
            R.log.hist(f"act/{name}", t, step)
        _bn_drift_check(R, step)
    R.opt.zero_grad(set_to_none=True)
    out["grad_norm"], out["clip_coef"] = gnorm, min(1.0, clip / (gnorm + 1e-6)) if clip else 1.0
    return out


def train_step(R: Run, step: int, lr: float, mbs: list[dict], epoch: int) -> dict | None:
    """One optimizer step over its micro-batches. Returns the step's CPU statistics, or None if it was skipped
    (non-finite gradient norm or OOM; the planner position still moves on). Family "ctc": train_step_ctc."""
    if is_ctc(R.cfg):
        return train_step_ctc(R, step, lr, mbs, epoch)
    cfg, dev = R.cfg, R.device
    lg = cfg["log"]
    dropped = [i for mb in mbs for i in mb["dropped"]]  # before the filter: a micro-batch can lose all its rows
    if dropped:
        R.log.event("dropped_audio", at_step=step, n=len(dropped), ids=dropped)
    # each micro-batch keeps its index in the step's plan: its SpecAugment masks are seeded with it (specaug_seed)
    mb_index = [j for j, mb in enumerate(mbs) if len(mb["ids"])]
    mbs = [mb for mb in mbs if len(mb["ids"])]
    n_tok = sum(int(mb["top_idx"].shape[0]) for mb in mbs)
    if not mbs or n_tok == 0:
        R.log.event("empty_step", at_step=step)
        return None
    for g in R.opt.param_groups:
        g["lr"] = lr
    # bn.mode "train": every forward updates the BN running stats, and a skipped step must not keep that (a non-finite
    # forward would leave NaN statistics to every later eval, while training on batch statistics goes on unaware):
    # copied here, put back if the step is skipped
    bn_keep = [(b, b.clone()) for b in bn_running_stats(R.model)] if bn_mode(cfg) == "train" else []
    w_aux = aux_ctc_weight(cfg) if getattr(R, "aux_ctc", None) is not None else 0.0
    aux_sum = torch.zeros((), device=dev)  # the step's aux-CTC loss, summed over its rows
    stats_step = lg["layer_stats_every"] and step % int(lg["layer_stats_every"]) == 0
    hist_step = lg["hist_every"] and step % int(lg["hist_every"]) == 0
    capture = {"_modules": activation_modules(R.model)} if hist_step else None
    S = len(R.src_index)
    nt = len(TERMS)
    tot = torch.zeros(nt, device=dev)
    by_src = torch.zeros(S, nt, device=dev)
    n_src = torch.zeros(S, device=dev)
    by_bucket = torch.zeros(len(BUCKETS), nt, device=dev)
    n_bucket = torch.zeros(len(BUCKETS), device=dev)
    per_utt, meta = [], []
    audio_real = audio_pad = dec_real = dec_pad = 0.0
    logits = losses = oom = res = ctc = None
    gen = R.gen if cfg["specaug"]["enabled"] else None
    j = -1
    try:
        for j, mb in enumerate(mbs):
            if gen is not None:
                gen.manual_seed(specaug_seed(cfg, step, mb_index[j]))
            res = forward_logits(R, mb, R.feat_train, gen=gen, capture=capture if j == 0 else None,
                                 with_encoder=w_aux > 0)
            logits, rows, mfrac = res[:3]
            losses = kd_losses(logits, mb["top_idx"].to(dev, non_blocking=True), mb["top_lp"].to(dev, non_blocking=True))
            if w_aux > 0:  # the aux-CTC term over the same N as the KD objective (kitsune.kd module docstring)
                ctc = aux_ctc_sum(R, res[3], mb)
                (kd_objective(losses, n_tok, cfg["loss"]["w_kl"], cfg["loss"]["w_ce"]) + w_aux * ctc / n_tok).backward()
                aux_sum += ctc.detach()
            else:
                kd_objective(losses, n_tok, cfg["loss"]["w_kl"], cfg["loss"]["w_ce"]).backward()
            res = ctc = None
            with torch.no_grad():
                T = torch.stack([losses[k].detach().float() for k in TERMS], dim=1)  # (N, nt)
                sidx = torch.tensor([R.src_index[s] for s in mb["sources"]], device=dev)[rows]
                tot += T.sum(0)
                by_src.index_add_(0, sidx, T)
                n_src += torch.bincount(sidx, minlength=S).float()
                p1 = losses["teacher_p1"]
                for b, (_, fn) in enumerate(BUCKETS):
                    m = fn(p1).float()
                    by_bucket[b] += (T * m[:, None]).sum(0)
                    n_bucket[b] += m.sum()
                B = len(mb["ids"])
                per_utt.append(torch.zeros(B, nt, device=dev).index_add_(0, rows, T))
                meta.append((mb, mfrac.detach() if mfrac is not None else torch.zeros(B, device=dev)))
            lens = mb["lengths"]
            audio_real += float(lens.sum()) / SR
            audio_pad += float(lens.max()) * len(lens) / SR
            dec_real += float(mb["dec_mask"].sum())
            dec_pad += float(mb["dec_mask"].numel())
            logits = losses = None
    except torch.OutOfMemoryError as e:
        oom = str(e)[:500]
    if oom is not None:  # outside the except: its traceback no longer pins the step's activations
        logits = losses = res = ctc = None
        per_utt.clear()
        meta.clear()
        R.opt.zero_grad(set_to_none=True)
        _put_back(bn_keep)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        R.st["oom_skips"] += 1
        R.log.event("oom_step_skipped", at_step=step, n_micro=len(mbs), at_micro=j, audio_pad_s=round(audio_pad, 1),
                    count=R.st["oom_skips"], error=oom, ids=[mb["ids"] for mb in mbs],
                    padded_s=[round(float(mb["lengths"].max()) * len(mb["ids"]) / SR, 1) for mb in mbs],
                    dec_len=[int(mb["dec_mask"].shape[1]) for mb in mbs],
                    **({"bn_stats_restored": True} if bn_keep else {}))
        if R.st["oom_skips"] > int(cfg["memory"]["max_oom_skips"]):
            raise torch.OutOfMemoryError(f"{R.st['oom_skips']} steps skipped for OOM (limit "
                                         f"{cfg['memory']['max_oom_skips']}); last: {oom}")
        return None

    def nonfinite_fields() -> dict:
        # everything needed to find the culprit afterwards: the step's ids per micro-batch, each micro-batch's loss
        # sums, and the utterances whose own loss is non-finite (empty if only the backward overflowed)
        pus = [pu.cpu() for pu in per_utt]
        bad = [mb["ids"][b] for (mb, _), pu in zip(meta, pus) for b in range(len(mb["ids"]))
               if not bool(torch.isfinite(pu[b, :2]).all())]
        return dict(nonfinite_ids=bad, ids=[mb["ids"] for mb in mbs],
                    mb_n_tok=[int(mb["top_idx"].shape[0]) for mb in mbs],
                    mb_kl_sum=[float(pu[:, 0].sum()) for pu in pus], mb_ce_sum=[float(pu[:, 1].sum()) for pu in pus])

    out = optimizer_update(R, step, lr, stats_step, hist_step, capture, bn_keep, nonfinite_fields)
    if out is None:
        return None

    # one host transfer for the step's statistics
    flat = torch.cat([tot, by_src.flatten(), n_src, by_bucket.flatten(), n_bucket, aux_sum.reshape(1)]).cpu().numpy()
    k = 0

    def take(n):
        nonlocal k
        v = flat[k:k + n]
        k += n
        return v

    out.update(tot=take(nt), by_src=take(S * nt).reshape(S, nt), n_src=take(S), by_bucket=take(len(BUCKETS) * nt)
               .reshape(len(BUCKETS), nt), n_bucket=take(len(BUCKETS)), n_tok=n_tok, audio_real=audio_real,
               audio_pad=audio_pad, dec_real=dec_real, dec_pad=dec_pad, n_micro=len(mbs))
    aux_step = float(take(1)[0])
    if w_aux > 0:  # per target token, as loss/kl and loss/ce
        out["aux_ctc"] = aux_step / n_tok
    utts, masked = [], []
    for (mb, mf), pu in zip(meta, per_utt):
        pu, mf = pu.cpu().numpy(), mf.float().cpu().numpy()
        masked.append(mf)
        n = mb["n_tok"].numpy()
        for b in range(len(mb["ids"])):
            utts.append(dict(step=step, epoch=epoch, id=mb["ids"][b], source=mb["sources"][b],
                             duration=float(mb["durations"][b]), n_tok=int(n[b]), kl=float(pu[b, 0] / n[b]),
                             ce=float(pu[b, 1] / n[b]), top1_acc=float(pu[b, 2] / n[b]), masked_frac=float(mf[b]),
                             agree=float(mb["agree"][b]), attempt=R.st["resumes"]))
    out["utts"] = utts
    out["masked_frac"] = float(np.concatenate(masked).mean()) if masked else float("nan")
    out["dropped"] = len(dropped)
    out["n_utts"] = len(utts)
    return out


def ctc_terms(losses: dict) -> torch.Tensor:
    """kitsune.ctc_kd.ctc_kd_losses(per_utt=True)'s vectors as a (B, len(CTC_TERMS)) float matrix (kl = kl_dense +
    kl_blank), detached."""
    cols = {"kl": losses["kl_dense"] + losses["kl_blank"], "n_tokens": losses["n_tokens"]}
    return torch.stack([(cols[k] if k in cols else losses[k]).detach().float() for k in CTC_TERMS], dim=1)


def ctc_step_loss(R: Run, mb: dict, featurizer, n_tok_step: int, gen=None, capture: dict | None = None):
    """One micro-batch's share of a CTC step's loss: forward_ctc, then kitsune.ctc_kd.ctc_kd_losses per utterance
    against the micro-batch's frame targets, and ctc_kd_objective = (w_kl (KL dense + KL blank) + w_ctc CTC) / the
    STEP's N_u (n_tok_step), so the micro-batches' losses add up to the step's. Returns (loss, the per-utterance
    losses, masked_frac or None). train_step_ctc's and fwd_bwd's; the loss-parity test checks it against the library."""
    from kitsune.ctc_kd import ctc_kd_losses, ctc_kd_objective

    lp, _, mfrac = forward_ctc(R, mb, featurizer, gen=gen, capture=capture)
    losses = ctc_kd_losses(lp, ctc_frame_batch(mb), per_utt=True)
    lp = None
    return ctc_kd_objective(losses, n_tok_step, R.cfg["loss"]["w_kl"], R.cfg["loss"]["w_ctc"]), losses, mfrac


def train_step_ctc(R: Run, step: int, lr: float, mbs: list[dict], epoch: int) -> dict | None:
    """train_step of family "ctc": every micro-batch's forward, CTC-KD loss over the step's N_u (the CTC target tokens
    of all its micro-batches: ctc_step_loss) and backward, then optimizer_update. The step's statistics are the sums of
    CTC_TERMS over the step, per source and per utterance (train_utts: n_tok = U, kl and ce = its KL and CTC per target
    token - per 1 for a silent one -, top1_acc = its frame argmax agreement). Skipped (None) for OOM or a non-finite
    gradient norm, as train_step."""
    cfg, dev = R.cfg, R.device
    lg = cfg["log"]
    dropped = [i for mb in mbs for i in mb["dropped"]]
    if dropped:
        R.log.event("dropped_audio", at_step=step, n=len(dropped), ids=dropped)
    mb_index = [j for j, mb in enumerate(mbs) if len(mb["ids"])]
    mbs = [mb for mb in mbs if len(mb["ids"])]
    if not mbs:
        R.log.event("empty_step", at_step=step)
        return None
    n_tok = sum(int(mb["n_tok"].sum()) for mb in mbs)
    for g in R.opt.param_groups:
        g["lr"] = lr
    bn_keep = [(b, b.clone()) for b in bn_running_stats(R.model)] if bn_mode(cfg) == "train" else []
    stats_step = lg["layer_stats_every"] and step % int(lg["layer_stats_every"]) == 0
    hist_step = lg["hist_every"] and step % int(lg["hist_every"]) == 0
    capture = {"_modules": activation_modules(R.model)} if hist_step else None
    S, nt = len(R.src_index), len(CTC_TERMS)
    tot = torch.zeros(nt, device=dev)
    by_src = torch.zeros(S, nt, device=dev)
    n_src = torch.zeros(S, device=dev)
    per_utt, meta = [], []
    audio_real = audio_pad = 0.0
    loss = losses = oom = None
    gen = R.gen if cfg["specaug"]["enabled"] else None
    j = -1
    try:
        for j, mb in enumerate(mbs):
            if gen is not None:
                gen.manual_seed(specaug_seed(cfg, step, mb_index[j]))
            loss, losses, mfrac = ctc_step_loss(R, mb, R.feat_train, n_tok, gen=gen,
                                                capture=capture if j == 0 else None)
            loss.backward()
            with torch.no_grad():
                T = ctc_terms(losses)  # (B, nt)
                sidx = torch.tensor([R.src_index[s] for s in mb["sources"]], device=dev)
                tot += T.sum(0)
                by_src.index_add_(0, sidx, T)
                n_src += torch.bincount(sidx, minlength=S).float()
                per_utt.append(T)
                B = len(mb["ids"])
                meta.append((mb, mfrac.detach() if mfrac is not None else torch.zeros(B, device=dev)))
            lens = mb["lengths"]
            audio_real += float(lens.sum()) / SR
            audio_pad += float(lens.max()) * len(lens) / SR
            loss = losses = None
    except torch.OutOfMemoryError as e:
        oom = str(e)[:500]
    if oom is not None:  # outside the except: its traceback no longer pins the step's activations
        loss = losses = None
        per_utt.clear()
        meta.clear()
        R.opt.zero_grad(set_to_none=True)
        _put_back(bn_keep)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        R.st["oom_skips"] += 1
        R.log.event("oom_step_skipped", at_step=step, n_micro=len(mbs), at_micro=j, audio_pad_s=round(audio_pad, 1),
                    count=R.st["oom_skips"], error=oom, ids=[mb["ids"] for mb in mbs],
                    padded_s=[round(float(mb["lengths"].max()) * len(mb["ids"]) / SR, 1) for mb in mbs],
                    padded_frames=[int(mb["n_frames"].max()) * len(mb["ids"]) for mb in mbs],
                    **({"bn_stats_restored": True} if bn_keep else {}))
        if R.st["oom_skips"] > int(cfg["memory"]["max_oom_skips"]):
            raise torch.OutOfMemoryError(f"{R.st['oom_skips']} steps skipped for OOM (limit "
                                         f"{cfg['memory']['max_oom_skips']}); last: {oom}")
        return None

    def nonfinite_fields() -> dict:
        pus = [pu.cpu() for pu in per_utt]
        bad = [mb["ids"][b] for (mb, _), pu in zip(meta, pus) for b in range(len(mb["ids"]))
               if not bool(torch.isfinite(pu[b, :2]).all())]
        return dict(nonfinite_ids=bad, ids=[mb["ids"] for mb in mbs], mb_n_tok=[int(mb["n_tok"].sum()) for mb in mbs],
                    mb_kl_sum=[float(pu[:, 0].sum()) for pu in pus], mb_ctc_sum=[float(pu[:, 1].sum()) for pu in pus])

    out = optimizer_update(R, step, lr, stats_step, hist_step, capture, bn_keep, nonfinite_fields)
    if out is None:
        return None
    flat = torch.cat([tot, by_src.flatten(), n_src]).cpu().numpy()  # one host transfer for the step's statistics
    out.update(tot=flat[:nt], by_src=flat[nt:nt + S * nt].reshape(S, nt), n_src=flat[nt + S * nt:], n_tok=n_tok,
               audio_real=audio_real, audio_pad=audio_pad, n_micro=len(mbs))
    col = {k: i for i, k in enumerate(CTC_TERMS)}
    utts, masked = [], []
    for (mb, mf), pu in zip(meta, per_utt):
        pu, mf = pu.cpu().numpy(), mf.float().cpu().numpy()
        masked.append(mf)
        for b in range(len(mb["ids"])):
            u, fr = max(float(pu[b, col["n_tokens"]]), 1.0), max(float(pu[b, col["n_frames"]]), 1.0)
            utts.append(dict(step=step, epoch=epoch, id=mb["ids"][b], source=mb["sources"][b],
                             duration=float(mb["durations"][b]), n_tok=int(pu[b, col["n_tokens"]]),
                             kl=float(pu[b, col["kl"]] / u), ce=float(pu[b, col["ctc"]] / u),
                             top1_acc=float(pu[b, col["argmax_agree"]] / fr), masked_frac=float(mf[b]),
                             agree=float(mb["agree"][b]), attempt=R.st["resumes"]))
    out["utts"] = utts
    out["masked_frac"] = float(np.concatenate(masked).mean()) if masked else float("nan")
    out["dropped"] = len(dropped)
    out["n_utts"] = len(utts)
    augs = [mb["aug"] for mb in mbs if "aug" in mb]
    if augs:  # augment.enabled: the step's sums of its micro-batches' counts (log_step's aug/* shares)
        out["aug"] = {k: sum(int(a[k]) for a in augs) for k in augs[0]}
    return out


def _groups(R: Run, tensors: list[torch.Tensor]):
    """(module, its tensors as one flat tensor), one module at a time: concatenating every module up front would put a
    third full-size fp32 copy on the device next to the weights and gradients (2.3 GiB for the real student, more than
    the 8 GB laptop has left at a histogram step)."""
    by: dict[str, list[torch.Tensor]] = {}
    for n, t in zip(R.param_names, tensors):
        if t is not None:
            by.setdefault(module_key(n), []).append(t)
    for k, ts in by.items():
        yield k, torch.cat([t.detach().flatten() for t in ts])


def _grad_hists(R: Run, step: int):
    for k, t in _groups(R, [p.grad for p in R.params]):
        R.log.hist(f"grad/{k}", t, step)


def _weight_hists(R: Run, step: int):
    for k, t in _groups(R, list(R.params)):
        R.log.hist(f"weight/{k}", t, step)


def _bn_drift_check(R: Run, step: int):
    """bn/max_abs_drift: how far the BatchNorm running stats are from the student init's (R.bn0, taken when
    setup_model loads it; a resumed state's are the same under "frozen"). bn.mode "frozen": BN is frozen for the whole
    run, so they must be exactly those, else AssertionError. "train": BN trains, the drift is reported only (after the
    check that every BN is in train mode)."""
    mode = bn_mode(R.cfg)
    n = assert_bn_mode(R.model, mode)
    drift = 0.0
    for name, m in R.model.named_modules():
        if name in R.bn0:
            m0, v0 = R.bn0[name]
            drift = max(drift, float((m.running_mean - m0).abs().max()), float((m.running_var - v0).abs().max()))
    R.log.scalars({"bn/max_abs_drift": drift, "bn/modules": n}, step)
    if mode == "frozen" and drift != 0.0:
        raise AssertionError(f"BatchNorm running stats moved by {drift} although BN is frozen")


# ------------------------------------------------------------------------------------------------------- logging


def log_step(R: Run, step: int, lr: float, phase: int, out: dict, wait_s: float, step_s: float, e: int,
             s: int) -> float:
    """The step's row (steps.parquet, scalars, TensorBoard) and per-utterance records. Returns the objective
    (w_kl * KL + w_ce * CE, the optimised loss; loss/total adds the decoupled L2-SP value, which is not in the
    gradient). The objective goes out twice: as loss/objective and as combined_loss/train, the train curve of the
    combined-loss chart next to combined_loss/val (kitsune.evaluate.combined_loss, the same quantity on the gate sets).
    It is measured on the step's augmented audio (SpecAugment on its log-mel features, under specaug.enabled) in train
    mode, with the weights before this step's update; the val points are un-augmented, in eval mode. That difference
    is the point of a train curve (what the optimizer sees), not something to correct.
    Family "ctc": the objective is w_kl * KL + w_ctc * CTC per CTC target token (N_u), logged with loss/kl (= kl_dense
    + kl_blank), loss/kl_dense, loss/kl_blank and loss/ctc per target token, and the frame metrics ctc/argmax_agree,
    ctc/argmax_blank, ctc/teacher_blank, ctc/frac_dense, ctc/kl_per_frame per valid frame (ctc_step_rows)."""
    from kitsune.runlog import system_stats

    cfg = R.cfg
    ctc = is_ctc(cfg)
    n_tok = out["n_tok"]
    tot = out["tot"]
    l2sp = out["l2sp"] if "l2sp" in out else float(R.l2sp.value())
    if ctc:
        loss_row, objective, family_row = ctc_step_rows(R, out, l2sp)
    else:
        w_kl, w_ce = cfg["loss"]["w_kl"], cfg["loss"]["w_ce"]
        kl, ce = tot[0] / n_tok, tot[1] / n_tok
        objective = w_kl * kl + w_ce * ce
        # loss/total: what the step minimised (the objective, plus the aux-CTC term when there is one) + the L2-SP value
        aux = aux_ctc_weight(cfg) * out["aux_ctc"] if "aux_ctc" in out else 0.0
        loss_row = {"loss/total": objective + aux + l2sp, "loss/objective": objective, "loss/kl": kl, "loss/ce": ce,
                    "loss/l2sp": l2sp, "combined_loss/train": objective}
        family_row = {}
    t, T = R.progress()
    n_steps = R.planner.epoch_stats[e]["steps"] if e in R.planner.epoch_stats else 1
    R.st["epoch"], R.st["epoch_progress"] = e, e + (s + 1) / n_steps
    R.st["audio_s"] += out["audio_real"]
    R.st["tokens"] += n_tok
    R.st["step_time_s"] += step_s
    R.st["data_wait_s"] = R.st.get("data_wait_s", 0.0) + wait_s  # summary.json's throughput.data_wait_frac
    R.st["last_objective"] = float(objective)
    flops = None
    if R.st["flops_per_padded_s"]:
        mult = 4.0 if R.st["memory"].get("grad_ckpt") else 3.0
        flops = mult * R.st["flops_per_padded_s"] * out["audio_pad"]
    row = {
        **loss_row,
        "opt/lr": lr, "opt/grad_norm": out["grad_norm"], "opt/clip_coef": out["clip_coef"],
        "time/step_s": step_s, "time/data_wait_s": wait_s, "time/compute_s": step_s - wait_s,
        "perf/audio_s_per_s": out["audio_real"] / step_s, "perf/tokens_per_s": n_tok / step_s,
        "perf/pad_eff_audio": out["audio_real"] / max(out["audio_pad"], 1e-9),
        **({} if ctc else {"perf/pad_eff_dec": out["dec_real"] / max(out["dec_pad"], 1e-9)}),  # no decoder in a CTC
        "aug/masked_frac": out["masked_frac"],
        "data/epoch": e, "data/epoch_progress": R.st["epoch_progress"], "data/utts": out["n_utts"],
        "data/audio_s": out["audio_real"], "data/tokens": n_tok, "data/micro_batches": out["n_micro"],
        "data/dropped": out["dropped"], "sched/phase": phase, "sched/progress": t / T if T else float("nan"),
        "sched/train_s": R.clock(),
    }
    if "aux_ctc" in out:  # the aux-CTC loss per target token (unweighted; loss.aux_ctc_weight x it is in loss/total)
        row["loss/aux_ctc"] = out["aux_ctc"]
    if "aug" in out:  # augment.enabled (train_step_ctc): the step's share of utterances in a joined row, of rows cut,
        a = out["aug"]  # mixed and padded; next to aug/masked_frac, and like it only on full rows (lean_step)
        utts, rows = max(int(a["utts"]), 1), max(int(a["rows"]), 1)
        row.update({"aug/concat_frac": a["concat_utts"] / utts, "aug/truncated_frac": a["truncated"] / rows,
                    "aug/mixed_frac": a["mixed"] / rows, "aug/cut_padded_frac": a.get("cut_padded", 0) / rows,
                    "aug/end_padded_frac": a.get("end_padded", 0) / rows})
    if flops is not None:
        row["perf/tflops"] = flops / step_s / 1e12
        row["perf/mfu"] = flops / step_s / (float(cfg["perf"]["peak_tflops"]) * 1e12)
    if ctc:
        row.update(family_row)
        row["perf/frames_per_s"] = family_row["data/frames"] / step_s
    else:
        for i, k in enumerate(TERMS[2:], start=2):
            row[TERM_TAGS[k]] = tot[i] / n_tok
        for src, si in R.src_index.items():
            n = out["n_src"][si]
            if n > 0:
                row[f"src/{src}/kl"] = out["by_src"][si, 0] / n
                row[f"src/{src}/ce"] = out["by_src"][si, 1] / n
                row[f"src/{src}/top1"] = out["by_src"][si, 2] / n
                row[f"src/{src}/tokens"] = n
        for b, (name, _) in enumerate(BUCKETS):
            n = out["n_bucket"][b]
            row[f"bucket/{name}/frac"] = n / n_tok
            if n > 0:
                row[f"bucket/{name}/kl"] = out["by_bucket"][b, 0] / n
                row[f"bucket/{name}/ce"] = out["by_bucket"][b, 1] / n
                row[f"bucket/{name}/top1"] = out["by_bucket"][b, 2] / n
    if R.device.type == "cuda":
        row["mem/step_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        # the step's reserved peak (the loop resets both peaks before every step): what the full smoke checks against
        # the 5090's memory, since the caching allocator's reserve, not the allocation, is what runs out
        row["mem/step_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    if lean_step(R, step):  # fix 7: the core keys only (system_stats' NVML, psutil and cgroup reads skipped too)
        row = {k: v for k, v in row.items() if k.startswith(CORE_STEP_TAGS)}
    else:
        row.update(system_stats())
    R.log.step_row(row, step)
    R.log.train_utts(out["utts"])
    if "grad_sq_by_module" in out:
        R.log.scalars({f"layers/grad_norm/{k}": math.sqrt(v) for k, v in out["grad_sq_by_module"].items()}, step)
        R.log.scalars({f"layers/update_ratio/{k}": v for k, v in out["update_ratio"].items()}, step)
        R.log.scalars({f"l2sp/dist/{k}": v for k, v in R.l2sp.per_module_distance().items()}, step)
        R.log.scalars({f"l2sp/rel/{k}": v for k, v in R.l2sp.per_module_distance(relative=True).items()}, step)
    from kitsune import heartbeat

    heartbeat.beat()  # $KITSUNE_HEARTBEAT (the full-run queue's per-item file; none: a no-op), at most every 5 s
    return objective


def lean_step(R: Run, step: int) -> bool:
    """log.full_scalars_every_steps (fix 7) = N > 1: step `step` logs only the CORE_STEP_TAGS keys of its row - after
    the smoke phase (whose steps, the profiled ones among them, always log everything), on steps not divisible by N.
    steps.parquet still gets a row for it (NaN in the other columns). The full row's system_stats() reads NVML eight
    times per GPU, psutil and the cgroup on every step: ~20-36 ms, which the lean steps save."""
    n = int((R.cfg.get("log") or {}).get("full_scalars_every_steps", 1) or 1)
    if n <= 1 or step % n == 0:
        return False
    sm = R.cfg.get("smoke") or {}
    return bool(R.st.get("smoke_done") or not (sm.get("enabled") and sm.get("steps")))


def ctc_step_rows(R: Run, out: dict, l2sp: float) -> tuple[dict, float, dict]:
    """A CTC step's loss rows, objective and family rows from train_step_ctc's sums (log_step): the objective (w_kl *
    KL + w_ctc * CTC) / N_u - kitsune.ctc_kd.ctc_kd_objective over the step, what its gradient came from (a step of
    silent rows only divides by 1, as the loss did) - with its terms per target token; the frame metrics per valid
    frame (the argmax-blank share is the blank-collapse watch: the teacher's own share is ctc/teacher_blank); per source
    the KL and CTC per target token, the frame agreement, the tokens and the utterances."""
    cfg = R.cfg
    col = {k: i for i, k in enumerate(CTC_TERMS)}
    tot = out["tot"]
    n = max(float(out["n_tok"]), 1.0)
    frames = max(float(tot[col["n_frames"]]), 1.0)
    per_tok = {k: float(tot[col[k]]) / n for k in CTC_TAGS}
    objective = float(cfg["loss"]["w_kl"]) * per_tok["kl"] + float(cfg["loss"]["w_ctc"]) * per_tok["ctc"]
    loss_row = {"loss/total": objective + l2sp, "loss/objective": objective,
                **{CTC_TAGS[k]: v for k, v in per_tok.items()}, "loss/l2sp": l2sp, "combined_loss/train": objective}
    fam = {tag: float(tot[col[k]]) / frames for k, tag in CTC_FRAME_TAGS.items()}
    fam.update({"ctc/kl_per_frame": float(tot[col["kl"]]) / frames, "ctc/frames_per_token": frames / n,
                "data/frames": float(tot[col["n_frames"]])})
    for src, si in R.src_index.items():
        if out["n_src"][si] > 0:
            b = out["by_src"][si]
            u, f = max(float(b[col["n_tokens"]]), 1.0), max(float(b[col["n_frames"]]), 1.0)
            fam.update({f"src/{src}/kl": float(b[col["kl"]]) / u, f"src/{src}/ctc": float(b[col["ctc"]]) / u,
                        f"src/{src}/argmax_agree": float(b[col["argmax_agree"]]) / f,
                        f"src/{src}/tokens": float(b[col["n_tokens"]]), f"src/{src}/utts": float(out["n_src"][si])})
    return loss_row, objective, fam


# ---------------------------------------------------------------------------------------------------------- evals


def combined_val_full(cfg: dict, tf_sum: dict) -> dict | None:
    """combined_loss/val_full's record of a complete-set eval (kitsune.evaluate.combined_loss over the gate sets of its
    teacher-forced pass, with scope "complete"), None without a token. Family "ctc": the CTC objective
    (kitsune.ctc_eval.combined_loss_ctc: w_kl x KL + w_ctc x CTC per target token)."""
    c = combined_loss(cfg, tf_sum)
    return dict(c, scope="complete") if c else None


def combined_loss(cfg: dict, tf_sum: dict | None) -> dict | None:
    """The family's training objective on a teacher-forced eval summary, pooled over its gate sets: AED
    kitsune.evaluate.combined_loss (w_kl x KL + w_ce x CE), CTC kitsune.ctc_eval.combined_loss_ctc (w_kl x KL + w_ctc
    x CTC); None without a token."""
    if is_ctc(cfg):
        from kitsune.ctc_eval import combined_loss_ctc

        return combined_loss_ctc(tf_sum, cfg["loss"]["w_kl"], cfg["loss"]["w_ctc"])
    from kitsune import evaluate as ev

    return ev.combined_loss(tf_sum, cfg["loss"]["w_kl"], cfg["loss"]["w_ce"])


def eval_summary(step: int, train_s: float, final: bool, complete: bool, tf_sum: dict, probe_sum: dict | None,
                 gr_sum: dict, full_sum: dict | None, pg_sum: dict | None, wall_s: float, *, n_probe_greedy: int,
                 n_probe: int, epoch: float | None = None, combined: dict | None = None) -> dict:
    """A full eval's evals/step_<N>/summary.json (run_eval; scripts/05_evaluate.py writes the same for a checkpoint).
    The headline pools the complete sets' greedy numbers when they were decoded (full_sum), else the greedy subset's
    (the scope says which: the step-0 eval decodes the subset, a full_every_epochs run's later ones the complete sets,
    on the same curve). probe_greedy / epoch / combined_loss only when given, so the runs without them keep their
    records unchanged."""
    from kitsune import evaluate as ev
    from kitsune.ctc_eval import frame_headline

    head_gr = full_sum if full_sum is not None else gr_sum
    # a CTC summary's val_top1 pooled per frame (its unit); an AED record is left as headline() gives it
    head = frame_headline(ev.headline(tf=tf_sum, greedy=head_gr, probe=probe_sum, probe_greedy=pg_sum), tf_sum)
    summary = dict(step=step, train_s=train_s, final=final, complete=complete, tf=tf_sum, probe=probe_sum,
                   greedy=gr_sum, greedy_full=full_sum, wall_s=wall_s, headline=head,
                   headline_scope=dict(val_greedy="complete" if full_sum is not None else "subset",
                                       val_greedy_utts=head_gr.get("n_utts", 0),
                                       val_cer_utts=ev.headline_val_utts(head_gr),
                                       train_greedy_utts=n_probe_greedy, train_tf_utts=n_probe))
    if pg_sum is not None:
        summary["probe_greedy"] = pg_sum
    if epoch is not None:
        summary["epoch"] = epoch
    if combined:
        summary["combined_loss"] = combined
    return summary


def eval_history_record(step: int, elapsed_s: float, tf_sum: dict, gr_sum: dict, probe_sum: dict | None,
                        pg_sum: dict | None, head: dict, epoch: float | None = None, full_sum: dict | None = None,
                        lr_phase: str | None = None) -> dict:
    """A full eval's record in the history the verdict and the early stop read (kitsune.evaluate.eval_record: the
    fixed subset's greedy numbers, and next to them the complete sets' at a complete eval (full_sum) and the LR phase
    of the last step, which verdict v2 reads; the probe's greedy CERs, the epoch in epoch mode (and under verdict v2)
    and the headline). The early stop reads heldout_kl / probe_kl only, so the added numbers change nothing there."""
    from kitsune import evaluate as ev

    rec = ev.eval_record(step, elapsed_s, tf=tf_sum, greedy=gr_sum, probe=probe_sum, greedy_full=full_sum,
                         lr_phase=lr_phase)
    if pg_sum and "all" in pg_sum:
        rec["probe_greedy"] = {k: pg_sum["all"][k] for k in ("cer_teacher_corpus", "cer_ref_corpus", "n")}
    if epoch is not None:
        rec["epoch"] = epoch
    rec["headline"] = head
    return rec


def eval_event_sets(tf_sum: dict, gr_sum: dict, full_sum: dict | None) -> dict:
    """The `eval` event's per-set numbers: kl / top1 of the teacher-forced pass (every eval utterance), cer / ratio /
    trunc of the greedy decode the headline pools - the complete sets at a complete eval (full_sum; the event's
    greedy_scope "complete"), else the fixed subset ("subset") - so they match summary.json's greedy_full and the
    verdict. A complete eval keeps the subset's own next to them as cer_subset / ratio_subset / trunc_subset: the
    numbers of summary.json's greedy, the ones the history (the verdict's trend, the early stop) reads."""
    def greedy(d: dict) -> dict:
        return dict(cer=round(d["cer_ref_corpus"], 4), ratio=round(d["ratio_vs_teacher"], 3),
                    trunc=round(d["trunc_rate"], 4))

    brief = {s: dict(kl=round(d["kl"], 4), top1=round(d["top1"], 4)) for s, d in tf_sum.get("sets", {}).items()}
    for s, d in (full_sum if full_sum is not None else gr_sum).get("sets", {}).items():
        brief.setdefault(s, {}).update(greedy(d))
    if full_sum is not None:
        for s, d in gr_sum.get("sets", {}).items():
            brief.setdefault(s, {}).update({f"{k}_subset": v for k, v in greedy(d).items()})
    return brief


def ctc_text(R: Run, key: str, store, ids: list[str] | None = None) -> dict[str, dict]:
    """The teacher text (reference, the teacher's ctc_hyp and its CER, truncated) of a frame store's rows - every row, or
    `ids` - read once per run into R.texts[key] (kitsune.ctc_eval.store_text)."""
    from kitsune.ctc_eval import store_text

    if key not in R.texts:
        R.texts[key] = store_text(store, ids)
    return R.texts[key]


def ctc_eval_parts(R: Run, complete: bool, mini_val: bool) -> tuple:
    """run_eval's numbers for family "ctc", in its variables' order: (tf_sum, tf_df, combined, probe_sum, probe_df,
    pg_sum, pg_df, gr_sum, gr_df, full_sum). One pass over the whole eval store (kitsune.ctc_eval.ctc_eval_records:
    frame metrics and greedy CTC decode of every row); the teacher-forced summaries are kitsune.ctc_eval's
    (summarise_ctc_tf), the greedy ones kitsune.evaluate.summarise_greedy's over the rows of each scope - the fixed
    greedy subset (gr_sum; its rows are the greedy tables unless the eval is complete) and, at a complete eval, every
    row (full_sum). The combined loss: val_full (complete) over the gate sets of the pass, val (mini_val) over the mini
    eval's val subset of the same pass. The probe: one pass over its rows and the probe-greedy ones."""
    from kitsune import ctc_eval as ce
    from kitsune import evaluate as ev

    cfg = R.cfg
    bs = float(cfg["eval"]["batch_s"])
    t0 = time.time()
    raw, hyps, dropped = ce.ctc_eval_records(R.model, R.evalstore, R.feat_eval, R.device, bs, amp=R.amp)
    wall = time.time() - t0
    bad = ce.bad_audio_per_set(R.evalstore, dropped)
    extra = dict(n_bad_audio=len(dropped), bad_audio=dropped[:50], bad_audio_per_set=bad, wall_s=wall)
    tf_sum, tf_df = ce.summarise_ctc_tf(raw, **extra)
    combined = {}  # series -> its combined_loss record, the scope it pooled next to the numbers
    if complete and (c := combined_val_full(cfg, tf_sum)):
        combined["val_full"] = c
    if mini_val and R.mini_val_ids:
        mini_tf, _ = ce.summarise_ctc_tf(raw[raw["id"].isin(set(R.mini_val_ids))].reset_index(drop=True))
        if c := combined_loss(cfg, mini_tf):
            combined["val"] = dict(c, scope="mini")
    gall = ce.greedy_frame(raw, hyps, ctc_text(R, "eval", R.evalstore), R.tokenizer)
    audio_s = float(raw["duration"].sum()) if len(raw) else 0.0
    greedy_meta = dict(extra, rtf=wall / audio_s if audio_s else float("nan"))
    subset = set(R.greedy_ids)
    full_sum = None
    if complete:
        full_sum = ev.summarise_greedy(gall, **greedy_meta)
        gr_df = gall
        gr_df["in_greedy_subset"] = gr_df["id"].isin(subset)
        gr_sum = ev.summarise_greedy(gr_df[gr_df["in_greedy_subset"]], wall_s=wall)
    else:
        gr_df = gall[gall["id"].isin(subset)].reset_index(drop=True)
        sub_dropped = [i for i in dropped if i in subset]
        gr_sum = ev.summarise_greedy(gr_df, n_bad_audio=len(sub_dropped), bad_audio=sub_dropped[:50],
                                     bad_audio_per_set=ce.bad_audio_per_set(R.evalstore, sub_dropped), wall_s=wall,
                                     rtf=greedy_meta["rtf"])
    probe_sum = probe_df = pg_sum = pg_df = None
    if R.probe_ids or R.probe_greedy_ids:
        ids = list(dict.fromkeys(list(R.probe_ids) + list(R.probe_greedy_ids)))
        t1 = time.time()
        praw, phyps, pdropped = ce.ctc_eval_records(R.model, R.train, R.feat_eval, R.device, bs, ids=ids, amp=R.amp,
                                                    decode=bool(R.probe_greedy_ids))
        pwall = time.time() - t1
        if R.probe_ids:
            want = set(R.probe_ids)
            pd_ = [i for i in pdropped if i in want]
            probe_sum, probe_df = ce.summarise_ctc_tf(
                praw[praw["id"].isin(want)].reset_index(drop=True), n_bad_audio=len(pd_), bad_audio=pd_[:50],
                bad_audio_per_set=ce.bad_audio_per_set(R.train, pd_), wall_s=pwall)
        if R.probe_greedy_ids:
            want = set(R.probe_greedy_ids)
            g_raw = praw[praw["id"].isin(want)].reset_index(drop=True)
            pg_df = ce.greedy_frame(g_raw, phyps, ctc_text(R, "probe", R.train, ids), R.tokenizer)
            pd_ = [i for i in pdropped if i in want]
            pg_sum = ev.summarise_greedy(pg_df, n_bad_audio=len(pd_), bad_audio=pd_[:50],
                                         bad_audio_per_set=ce.bad_audio_per_set(R.train, pd_), wall_s=pwall)
    return tf_sum, tf_df, combined, probe_sum, probe_df, pg_sum, pg_df, gr_sum, gr_df, full_sum


def run_eval(R: Run, step: int, final: bool = False, complete: bool | None = None, mini_val: bool = False) -> dict:
    """Teacher-forced on every eval utterance and on the train probe, greedy on the fixed subsets - or on the COMPLETE
    eval sets (`complete`; default: the final eval under eval.final_full_greedy; the loop passes it for every
    eval.full_every_epochs eval), the subset summary then taken from those rows so the history stays comparable.
    Tables, summary, scalars, the headline numbers (summary/full/..., log_headline) and samples go to the logger; one
    record goes to the history the verdict and the early stop read (the subset's greedy numbers, and at a complete eval
    the complete sets' next to them as greedy_full: eval_history_record). The `eval` event's
    per-set CERs are those of the headline's scope, named in its greedy_scope (eval_event_sets). Returns the greedy
    summary the verdict should judge: the complete-set one when there is one (the subset one if
    eval.final_full_greedy is off).

    The combined loss (kitsune.evaluate.combined_loss; summary.json's combined_loss, one record per series): an eval
    of the complete sets logs combined_loss/val_full, the gate sets' teacher-forced pass (every eval runs that pass on
    the complete sets, but only these evals are complete ones: the step-0 eval and the every_min / every_steps ones
    decode the greedy subset). mini_val (the step-0 eval, when mini evals are on) adds a teacher-forced pass over the
    mini eval's val subset for combined_loss/val's first point, the scope of every later one (run_mini_eval): the
    mini evals themselves stay out of step 0, the early stop and the verdict.

    Family "ctc": the same records from one pass over the eval store (ctc_eval_parts): a CTC student's greedy decode
    is the argmax of the log-probs the teacher-forced metrics score, so every eval decodes every eval row and the
    fixed subset's numbers are taken from those rows."""
    from kitsune import evaluate as ev

    cfg, log = R.cfg, R.log
    bs = float(cfg["eval"]["batch_s"])
    complete = bool(final and cfg["eval"]["final_full_greedy"]) if complete is None else bool(complete)
    t0 = time.time()
    log.event("eval_start", at_step=step, final=final, complete=complete)
    if is_ctc(cfg):
        (tf_sum, tf_df, combined, probe_sum, probe_df, pg_sum, pg_df, gr_sum, gr_df,
         full_sum) = ctc_eval_parts(R, complete, mini_val)
    else:
        tf_sum, tf_df = ev.teacher_forced_eval(R.model, R.evalstore, R.feat_eval, R.device, bs, amp=R.amp)
        w_kl, w_ce = cfg["loss"]["w_kl"], cfg["loss"]["w_ce"]
        combined = {}  # series -> its combined_loss record, the scope it pooled next to the numbers
        if complete and (c := combined_val_full(cfg, tf_sum)):
            combined["val_full"] = c
        if mini_val and R.mini_val_ids:
            mini_tf, _ = ev.teacher_forced_eval(R.model, R.evalstore, R.feat_eval, R.device, bs, ids=R.mini_val_ids,
                                                amp=R.amp)
            if c := ev.combined_loss(mini_tf, w_kl, w_ce):
                combined["val"] = dict(c, scope="mini")
        probe_sum = probe_df = None
        if R.probe_ids:
            probe_sum, probe_df = ev.teacher_forced_eval(R.model, R.train, R.feat_eval, R.device, bs, ids=R.probe_ids,
                                                         amp=R.amp)
        pg_sum = pg_df = None
        if R.probe_greedy_ids:  # un-augmented greedy decode of train utterances: memorisation shows as CER vs teacher
            pg_sum, pg_df = ev.greedy_eval(R.model, R.train, R.probe_greedy_ids, R.feat_eval, R.device, bs,
                                           tokenizer=R.tokenizer, amp=R.amp)
        full_sum = None
        if complete:
            full_sum, gr_df = ev.greedy_eval(R.model, R.evalstore, None, R.feat_eval, R.device, bs,
                                             tokenizer=R.tokenizer, amp=R.amp)
            gr_df["in_greedy_subset"] = gr_df["id"].isin(set(R.greedy_ids))
            gr_sum = ev.summarise_greedy(gr_df[gr_df["in_greedy_subset"]], wall_s=full_sum["wall_s"])
        else:
            gr_sum, gr_df = ev.greedy_eval(R.model, R.evalstore, R.greedy_ids, R.feat_eval, R.device, bs,
                                           tokenizer=R.tokenizer, amp=R.amp)
    assert_bn_mode(R.model, bn_mode(cfg))  # an eval gives every module its own flag back
    # rows the dataset could not decode are left out of these numbers (and of the verdict's gate sets, which says
    # so): the eval-side twin of train_step's `dropped_audio` event
    bad = {k: s for k, s in (("tf", tf_sum), ("probe", probe_sum), ("greedy", gr_sum), ("greedy_full", full_sum),
                             ("probe_greedy", pg_sum)) if s and s.get("n_bad_audio")}
    if bad:
        log.event("eval_dropped_audio", at_step=step, final=final, n={k: int(s["n_bad_audio"]) for k, s in bad.items()},
                  per_set={k: s.get("bad_audio_per_set", {}) for k, s in bad.items()},
                  ids={k: s.get("bad_audio", []) for k, s in bad.items()})

    for src, g in tf_df.groupby("source", sort=True):
        log.table(f"tf_{src}", g.reset_index(drop=True), step)
    if probe_df is not None:
        log.table("probe", probe_df, step)
    if pg_df is not None:
        log.table("probe_greedy", pg_df, step)
    for src, g in gr_df.groupby("source", sort=True):
        log.table(f"greedy_{src}", g.reset_index(drop=True), step)
    extra = {}  # only in the runs that use them, so the viability run's records are unchanged
    if pg_sum is not None:
        extra["probe_greedy"] = pg_sum
    if epoch_mode(cfg):
        extra["epoch"] = R.st["epoch_progress"]  # epochs done: 0 at step 0, e + 1 at the end of epoch e
    summary = eval_summary(step, R.clock(), final, complete, tf_sum, probe_sum, gr_sum, full_sum, pg_sum,
                           round(time.time() - t0, 1), n_probe_greedy=len(R.probe_greedy_ids),
                           n_probe=len(R.probe_ids), epoch=extra.get("epoch"), combined=combined)
    head = summary["headline"]
    log.eval_json("summary", summary, step)
    scal = ev.flatten(tf_sum, "eval/tf")
    scal.update(ev.flatten(gr_sum, "eval/greedy"))
    # the over-fitting gaps pool the GATE sets on the held-out side, as the verdict's gap and the early stop do: the
    # monitor-only sets (eval_emilia, galgame) come from training sources, and their all-sets KL is in eval/tf/all/kl
    if probe_sum:
        scal.update(ev.flatten(probe_sum, "eval/probe"))
        if "val_loss" in head and "train_loss" in head:  # heldout_kl - probe_kl
            scal["eval/kl_gap_heldout_minus_probe"] = head["val_loss"] - head["train_loss"]
    if pg_sum:
        scal.update(ev.flatten(pg_sum, "eval/probe_greedy"))
        sub = ev.headline(greedy=gr_sum, probe_greedy=pg_sum)  # on the fixed greedy subset, also at a complete eval
        if "val_cer_vs_teacher" in sub and "train_cer_vs_teacher" in sub:
            scal["eval/cer_teacher_gap_heldout_minus_probe"] = sub["val_cer_vs_teacher"] - sub["train_cer_vs_teacher"]
    if full_sum:
        scal.update(ev.flatten(full_sum, "eval/greedy_full"))
    if "epoch" in extra:
        scal["eval/epoch"] = extra["epoch"]
    scal["eval/wall_s"] = summary["wall_s"]
    scal.update({f"combined_loss/{k}": c["value"] for k, c in combined.items()})
    log.scalars(scal, step)
    log_headline(R, "full", head, step, scope=summary["headline_scope"])
    samples = ev.pick_samples(gr_df, int(cfg["log"]["samples_per_eval"]), seed=int(cfg["seed"]))
    if pg_df is not None:  # train utterances (their source says so) after the held-out ones
        samples += ev.pick_samples(pg_df, int(cfg["log"]["samples_per_eval"]), seed=int(cfg["seed"]))
    log.samples(step, samples)

    if not final and is_ctc(cfg):  # one pass over every eval row whatever the scope, the decode inside it
        R.st["eval_cost"] = dict(tf_s=float(tf_sum.get("wall_s", 0.0)),
                                 probe_s=float(probe_sum.get("wall_s", 0.0)) if probe_sum else 0.0,
                                 greedy_s=0.0, greedy_n=len(R.evalstore))
    elif not final:  # what fit_budget scales the final eval's duration from (a complete decode: nothing to scale)
        R.st["eval_cost"] = dict(tf_s=float(tf_sum.get("wall_s", 0.0)),
                                 probe_s=(float(probe_sum.get("wall_s", 0.0)) if probe_sum else 0.0)
                                 + (float(pg_sum.get("wall_s", 0.0)) if pg_sum else 0.0),
                                 greedy_s=float(gr_sum.get("wall_s", 0.0)),
                                 greedy_n=len(R.evalstore) if full_sum is not None else len(R.greedy_ids))
    # verdict v2 de-duplicates its trend window by epochs: its records carry the epoch outside epoch mode too
    rec_epoch = extra.get("epoch", R.st["epoch_progress"] if verdict_options(cfg) else None)
    rec = eval_history_record(step, R.clock(), tf_sum, gr_sum, probe_sum, pg_sum, head, epoch=rec_epoch,
                              full_sum=full_sum, lr_phase=lr_phase_name(R.st.get("lr_phase")))
    hist = R.st["history"]
    if hist and hist[-1]["step"] == step:
        hist[-1] = rec
    else:
        hist.append(rec)
    if pg_sum and "all" in pg_sum:
        extra["probe_cer_teacher"] = round(pg_sum["all"]["cer_teacher_corpus"], 4)
    if combined:
        extra["combined_loss"] = {k: round(c["value"], 5) for k, c in combined.items()}
    # greedy_scope ahead of the per-set numbers: the console's [event] line is cut at 300 characters
    log.event("eval", at_step=step, final=final, complete=complete,
              greedy_scope=summary["headline_scope"]["val_greedy"], wall_s=summary["wall_s"],
              sets=eval_event_sets(tf_sum, gr_sum, full_sum),
              probe_kl=probe_sum["all"]["kl"] if probe_sum and "all" in probe_sum else None,
              headline={k: round(v, 5) for k, v in head.items()},
              **{k: v for k, v in extra.items() if k != "probe_greedy"})
    return full_sum if full_sum is not None else gr_sum


def log_headline(R: Run, kind: str, head: dict, step: int, scope: dict | None = None):
    """An eval's headline numbers (kitsune.evaluate.headline) in one place: the scalars summary/<kind>/<name> (kind
    "full" or "mini"; CERs as fractions, with a <name>_pct copy in percent for TensorBoard) and one console line. With
    the full eval's headline_scope, also summary/<kind>/val_cer_utts (how many gate utterances val_cer pools) and the
    scope in the line: the step-0 point of the curve decodes the fixed subset, later ones may decode the complete
    sets."""
    from kitsune.evaluate import HEADLINE_CER

    row = {f"summary/{kind}/{k}": v for k, v in head.items()}
    row.update({f"summary/{kind}/{k}_pct": 100.0 * head[k] for k in HEADLINE_CER if k in head})
    if scope and "val_cer" in head:
        row[f"summary/{kind}/val_cer_utts"] = float(scope["val_cer_utts"])
    R.log.scalars(row, step)
    print(headline_line(kind, step, R.st["epoch_progress"], head, scope), flush=True)


def headline_line(kind: str, step: int, epoch: float, head: dict, scope: dict | None = None) -> str:
    """'[full eval] step N epoch E | val CER x.x% (vs teacher y.y%) | train CER vs teacher z.z% (vs ref w.w%) | val KL
    a.aaa train KL b.bbb'; n/a for a number that eval did not measure. With a scope (run_eval's headline_scope) the val
    CER says what it decoded: '... (vs teacher y.y%) on U utts (subset) | ...' or '(complete)'."""
    def pct(k):
        return f"{100.0 * head[k]:.1f}%" if k in head else "n/a"

    def kl(k):
        return f"{head[k]:.3f}" if k in head else "n/a"

    on = f" on {scope['val_cer_utts']} utts ({scope['val_greedy']})" if scope and "val_cer" in head else ""
    return (f"[{kind} eval] step {step} epoch {epoch:.2f} | val CER {pct('val_cer')} (vs teacher "
            f"{pct('val_cer_vs_teacher')}){on} | train CER vs teacher {pct('train_cer_vs_teacher')} (vs ref "
            f"{pct('train_cer')}) | val KL {kl('val_loss')} train KL {kl('train_loss')}")


def mini_teacher_rows(R: Run, kind: str, store, ids: list[str]) -> dict[str, dict]:
    """id -> the reference and teacher text greedy_eval scores a mini subset against (its teacher_rows), read from the
    store's index once per run: without them every greedy_eval call reads the whole index.parquet (233 k rows for the
    real train set)."""
    if kind not in R.mini_rows:
        pos = {u.id: i for i, u in enumerate(store.utts)}
        sub = store.subset([pos[i] for i in ids])
        fr = sub.frame()
        R.mini_rows[kind] = {u.id: dict(ref=ref, hyp=hyp, cer=u.teacher_cer, truncated=u.truncated)
                             for u, ref, hyp in zip(sub.utts, fr["ref"].tolist(), fr["hyp"].tolist())}
    return R.mini_rows[kind]


def mini_due(R: Run, step: int) -> bool:
    """A mini eval is due after optimizer step `step` (the loop skips it when a full eval runs at that step, or when
    the loop is due to end there and the final eval follows; on the wall clock that is judged from the last mini's wall
    time, a prediction rather than a guarantee)."""
    every = R.cfg["eval"]["mini"]["every_steps"]
    return bool(every) and step > 0 and step % int(every) == 0 and bool(R.mini_val_ids or R.mini_train_ids)


def run_mini_eval(R: Run, step: int) -> dict:
    """The mini eval (eval.mini): teacher-forced KL / CE / top-1 and, with mini.greedy, greedy CER vs the reference and
    vs the teacher, on the fixed mini subsets (mini_subsets: val from the gate sets, train from the probe). Its own
    pass - its own batches, eval mode without gradients (kitsune.evaluate), BatchNorm checked frozen afterwards -
    logged apart from the full evals: tables and summary.json under evals/step_<N>_mini/, scalars under eval/mini/
    (its wall time eval/mini/wall_s), the headline under summary/mini/, an `eval_mini` event and a row of
    st["mini_history"] (summary.json's mini_history). It never feeds the verdict's history, the early stop or the
    final-eval estimate. Its teacher-forced val pass also gives combined_loss/val (kitsune.evaluate.combined_loss over
    the gate sets of the val subset, summary.json's combined_loss): the held-out curve of the combined-loss chart, one
    scope from its step-0 point (run_eval's mini_val) to the last mini. Returns the headline. Family "ctc": one pass per
    subset gives both its frame metrics and its greedy CTC decode (kitsune.ctc_eval.ctc_eval), the combined loss is the
    CTC objective."""
    from kitsune import evaluate as ev

    cfg, log = R.cfg, R.log
    bs, greedy = float(cfg["eval"]["batch_s"]), cfg["eval"]["mini"]["greedy"]
    t0 = time.time()
    parts, frames = {}, {}
    for kind, store, ids in (("tf", R.evalstore, R.mini_val_ids), ("probe", R.train, R.mini_train_ids)):
        if not ids:
            continue
        g = "greedy" if kind == "tf" else "probe_greedy"
        if is_ctc(cfg):
            from kitsune.ctc_eval import ctc_eval

            parts[kind], frames[kind], gdf, meta = ctc_eval(
                R.model, store, R.feat_eval, R.device, bs, ids=ids, tokenizer=R.tokenizer,
                text=mini_teacher_rows(R, kind, store, ids) if greedy else None, amp=R.amp, decode=bool(greedy))
            if greedy:
                parts[g], frames[g] = ev.summarise_greedy(gdf, **meta), gdf
            continue
        parts[kind], frames[kind] = ev.teacher_forced_eval(R.model, store, R.feat_eval, R.device, bs, ids=ids,
                                                           amp=R.amp)
        if greedy:
            parts[g], frames[g] = ev.greedy_eval(R.model, store, ids, R.feat_eval, R.device, bs,
                                                 tokenizer=R.tokenizer, amp=R.amp,
                                                 teacher_rows=mini_teacher_rows(R, kind, store, ids))
    assert_bn_mode(R.model, bn_mode(cfg))  # an eval gives every module its own flag back
    wall = round(time.time() - t0, 1)
    comb = combined_loss(cfg, parts.get("tf"))
    combined = {"val": dict(comb, scope="mini")} if comb else {}

    for kind in ("tf", "greedy"):
        if kind in frames:
            for src, g in frames[kind].groupby("source", sort=True):
                log.table(f"{kind}_{src}", g.reset_index(drop=True), step, suffix="mini")
    for kind in ("probe", "probe_greedy"):
        if kind in frames:
            log.table(kind, frames[kind], step, suffix="mini")
    head = ev.headline(tf=parts.get("tf"), greedy=parts.get("greedy"), probe=parts.get("probe"),
                       probe_greedy=parts.get("probe_greedy"))
    if is_ctc(cfg):
        from kitsune.ctc_eval import frame_headline

        frame_headline(head, parts.get("tf"))  # val_top1 per frame, as eval_summary
    epoch = R.st["epoch_progress"]
    log.eval_json("summary", dict(step=step, train_s=R.clock(), epoch=epoch, mini=True, wall_s=wall, headline=head,
                                  n_val=len(R.mini_val_ids), n_train=len(R.mini_train_ids), **parts,
                                  **({"combined_loss": combined} if combined else {})),
                  step, suffix="mini")
    scal = {}
    for kind, s in parts.items():
        scal.update(ev.flatten(s, f"eval/mini/{kind}"))
    scal["eval/mini/wall_s"] = wall
    scal.update({f"combined_loss/{k}": c["value"] for k, c in combined.items()})
    log.scalars(scal, step)
    log_headline(R, "mini", head, step)
    R.st["mini_history"].append(dict(step=int(step), epoch=epoch, elapsed_s=R.clock(), wall_s=wall, **head))
    brief = {s: dict(kl=round(d["kl"], 4), top1=round(d["top1"], 4))
             for s, d in (parts.get("tf") or {}).get("sets", {}).items()}
    for s, d in (parts.get("greedy") or {}).get("sets", {}).items():
        brief.setdefault(s, {}).update(cer=round(d["cer_ref_corpus"], 4), cer_teacher=round(d["cer_teacher_corpus"], 4))
    log.event("eval_mini", at_step=step, epoch=epoch, wall_s=wall, n_val=len(R.mini_val_ids),
              n_train=len(R.mini_train_ids), sets=brief, headline={k: round(v, 5) for k, v in head.items()},
              **({"combined_loss": {k: round(c["value"], 5) for k, c in combined.items()}} if combined else {}))
    return head


def dev_due(R: Run, step: int) -> bool:
    """A dev eval is due after optimizer step `step` (eval.dev on and its store open): every eval.dev.every_steps
    steps since the last one, or at each crossing of a multiple of eval.dev.every_epochs by the epoch progress
    (e + (s + 1) / steps of epoch e, log_step: 0.1, 0.2, ... and exactly at the epoch ends, whatever the full-eval
    cadence; a skipped step never gets here and the next one catches up). The loop skips it at its last step, where
    the end phase's at_end one runs."""
    if getattr(R, "devstore", None) is None or not dev_on(R.cfg):
        return False
    dev = R.cfg["eval"]["dev"]
    if dev["every_steps"]:
        return step - int(R.st["dev_last_step"]) >= int(dev["every_steps"])
    k = float(dev["every_epochs"])
    return math.floor(R.st["epoch_progress"] / k + 1e-9) > math.floor(float(R.st["dev_last_epoch"]) / k + 1e-9)


def run_dev_eval(R: Run, step: int, final: bool = False) -> dict:
    """The dev-slice eval (eval.dev; the dev store of build_dev_store): its own pass in eval mode without gradients,
    teacher-forced on every dev row - AED kitsune.evaluate.teacher_forced_eval, plus a greedy decode with
    eval.dev.greedy (a real decode: costly); CTC kitsune.ctc_eval.ctc_eval, whose greedy decode is the argmax of the
    same pass (free). Pooled over every dev source (combined_loss pools the gate sets, and no train source is one):
    dev_objective = the family's training objective, dev_kl its KL and dev_ce its CE (CTC: the CTC loss), per target
    token. One record in st["dev_history"] (replacing one of the same step): {step, epoch, elapsed_s, wall_s, final,
    dev_ce, dev_kl, dev_objective, n_tok, n_utts, per_source {src: {ce, kl, n_tok}}[, cer, cer_vs_teacher]}, which
    early_stop_value reads for the DEV_METRICS; tables and summary.json under evals/step_<N>_dev/, the scalars
    eval/dev/{objective, ce, kl, wall_s} and eval/dev/tf/..., eval/dev/greedy/..., and an `eval_dev` event. It never
    writes an `eval` event, the verdict's history, the final-eval estimate or the evalstore's text, and draws no random
    number: with the early stop off, a run with dev evals trains exactly as one without. The early stop's check is the
    caller's (loop). Returns the record."""
    from kitsune import evaluate as ev

    cfg, log, store = R.cfg, R.log, getattr(R, "devstore", None)
    dc = cfg["eval"]["dev"]
    bs, greedy = float(cfg["eval"]["batch_s"]), bool(dc["greedy"])
    t0 = time.time()
    gr_sum = gr_df = None
    if is_ctc(cfg):
        from kitsune.ctc_eval import ctc_eval

        tf_sum, tf_df, gdf, meta = ctc_eval(R.model, store, R.feat_eval, R.device, bs, ids=None, tokenizer=R.tokenizer,
                                            text=ctc_text(R, "dev", store) if greedy else None, amp=R.amp,
                                            decode=greedy)
        if greedy:
            gr_sum, gr_df = ev.summarise_greedy(gdf, **meta), gdf
    else:
        tf_sum, tf_df = ev.teacher_forced_eval(R.model, store, R.feat_eval, R.device, bs, amp=R.amp)
        if greedy:
            gr_sum, gr_df = ev.greedy_eval(R.model, store, None, R.feat_eval, R.device, bs, tokenizer=R.tokenizer,
                                           amp=R.amp, teacher_rows=mini_teacher_rows(R, "dev", store,
                                                                                     [u.id for u in store.utts]))
    assert_bn_mode(R.model, bn_mode(cfg))  # an eval gives every module its own flag back
    wall = round(time.time() - t0, 1)
    c = combined_loss(cfg, tf_sum) or {}
    ce_key = "ctc" if is_ctc(cfg) else "ce"
    rec = dict(step=int(step), epoch=R.st["epoch_progress"], elapsed_s=R.clock(), wall_s=wall, final=bool(final),
               dev_ce=c.get(ce_key), dev_kl=c.get("kl"), dev_objective=c.get("value"), n_tok=int(c.get("n_tok", 0)),
               n_utts=int(tf_sum.get("n_utts", 0)),
               per_source={s: dict(ce=d["ce"], kl=d["kl"], n_tok=int(d["n_tok"]))
                           for s, d in tf_sum.get("sets", {}).items()})
    if gr_sum is not None and "all" in gr_sum:
        rec.update(cer=gr_sum["all"]["cer_ref_corpus"], cer_vs_teacher=gr_sum["all"]["cer_teacher_corpus"])
    hist = R.st["dev_history"]
    if hist and hist[-1]["step"] == rec["step"]:
        hist[-1] = rec
    else:
        hist.append(rec)
    R.st["dev_last_step"], R.st["dev_last_epoch"] = int(step), float(R.st["epoch_progress"])

    for src, g in tf_df.groupby("source", sort=True):
        log.table(f"tf_{src}", g.reset_index(drop=True), step, suffix="dev")
    if gr_df is not None:
        for src, g in gr_df.groupby("source", sort=True):
            log.table(f"greedy_{src}", g.reset_index(drop=True), step, suffix="dev")
    log.eval_json("summary", dict(rec, dev=True, tf=tf_sum, **({"greedy": gr_sum} if gr_sum is not None else {}),
                                  **({"combined_loss": c} if c else {})), step, suffix="dev")
    scal = ev.flatten(tf_sum, "eval/dev/tf")
    if gr_sum is not None:
        scal.update(ev.flatten(gr_sum, "eval/dev/greedy"))
    scal.update({f"eval/dev/{k}": rec[f"dev_{k}"] for k in ("objective", "ce", "kl") if rec[f"dev_{k}"] is not None})
    scal["eval/dev/wall_s"] = wall
    log.scalars(scal, step)
    sets = {s: dict(ce=round(d["ce"], 5), kl=round(d["kl"], 5)) for s, d in rec["per_source"].items()}
    for s, d in (gr_sum or {}).get("sets", {}).items():
        sets.setdefault(s, {}).update(cer=round(d["cer_ref_corpus"], 4), cer_teacher=round(d["cer_teacher_corpus"], 4))
    log.event("eval_dev", at_step=int(step), epoch=rec["epoch"], final=bool(final), wall_s=wall,
              **{k: rec[k] for k in DEV_METRICS}, sets=sets)
    return rec


# ----------------------------------------------------------------------------------------------------- early stop


def early_stop_value(R: Run, metric: str) -> float | None:
    """The early-stop metric right after an eval: the newest eval record's probe_kl / heldout_kl (eval_record), or the
    mean loss/objective of the optimizer steps since the previous eval (None if there were none). loss/objective, not
    loss/total: the L2-SP value in loss/total is not in the gradient and grows with the distance from the initial
    weights, so a window mean of loss/total rises once the objective flattens and patience would fire on a timetable.
    A dev metric (DEV_METRICS): the newest dev eval's number when that eval ran at this very step (run_dev_eval),
    else None."""
    es = R.st["early_stop"]
    if metric == "train_loss":
        return es["loss_sum"] / es["loss_n"] if es["loss_n"] else None
    if metric in DEV_METRICS:
        dh = R.st.get("dev_history") or []
        return dh[-1].get(metric) if dh and dh[-1]["step"] == R.st["step"] else None
    return (R.st["history"][-1] if R.st["history"] else {}).get(metric)


def smooth_k(cfg: dict) -> int:
    """early_stop.smooth (1 for a config without the key: a unit test's, a state from before it)."""
    return int((cfg.get("early_stop") or {}).get("smooth") or 1)


def early_stop_check(R: Run, step: int) -> bool:
    """After every in-loop eval (a dev metric: after every dev eval, loop): with early_stop.enabled and no trigger yet,
    update the early-stop state (early_stop_update), log it and act on a trigger (early_stop_trigger). Returns True when
    the loop must end now. early_stop.smooth k > 1: the raw value joins es["recent"] (the last k; None for a missing or
    non-finite one); nothing is counted - no evals, no patience, no min_evals - before there are k of them (only
    early_stop/raw is logged), then the rule reads their mean (None if any is None: no improvement), so min_evals and
    patience count smoothed checks. early_stop/value is what the rule saw, early_stop/raw the newest value."""
    ec, es = R.cfg["early_stop"], R.st["early_stop"]
    value = early_stop_value(R, ec["metric"])
    es["loss_sum"], es["loss_n"] = 0.0, 0  # the next eval's train-loss window starts here
    if not ec["enabled"] or es["triggered"]:
        return False
    nan = float("nan")
    k = smooth_k(R.cfg)
    raw = {}
    if k > 1:
        v = float(value) if value is not None and math.isfinite(float(value)) else None
        es["recent"] = (list(es.get("recent") or []) + [v])[-k:]
        raw = {"early_stop/raw": nan if v is None else v}
        if len(es["recent"]) < k:
            R.log.scalars(raw, step)
            return False
        value = None if any(x is None for x in es["recent"]) else sum(es["recent"]) / k
    reason = early_stop_update(es, ec, value, step)
    row = {"early_stop/value": nan if es["value"] is None else es["value"],
           "early_stop/best": nan if es["best"] is None else es["best"],
           "early_stop/evals_since_best": es["evals_since_best"], **raw}
    if reason is None:
        row["early_stop/triggered"] = 0.0
    R.log.scalars(row, step)
    return reason is not None and early_stop_trigger(R, reason, ec["action"])


def early_stop_trigger(R: Run, reason: str, action: str) -> bool:
    """Act on an early-stop trigger (reason "patience", "floor" or "stop_file") and log it (`early_stop` event,
    early_stop/triggered = 1). "stop": the loop ends now. "cooldown": the WSD cooldown starts now, over
    schedule.cooldown_frac x the loop time (or the steps; whole steps) done so far, capped by what is left of the
    budget, and the run ends when it is over (Run.progress, Run.cooldown_start); a run already in its cooldown goes on
    to the scheduled end. Returns True when the loop must end now."""
    ec, es, sch = R.cfg["early_stop"], R.st["early_stop"], R.cfg["schedule"]
    step = R.st["step"]
    info = dict(metric=ec["metric"] if ec["enabled"] else None, value=es["value"], best=es["best"],
                best_step=es["best_step"], evals_since_best=es["evals_since_best"], reason=reason, action=action,
                at_step=step, epoch=R.st["epoch_progress"])
    if (k := smooth_k(R.cfg)) > 1:  # value is the smoothed one; raw the newest check's (only then: tests pin the rest)
        info.update(raw=(es.get("recent") or [None])[-1], smooth=k)
    if es["triggered"]:  # the STOP file after an early cooldown began
        info["previous"] = es["triggered"]
    if action == "cooldown":
        t, T = R.progress()
        t_c = R.cooldown_start(T)
        if R.st["pre_cooldown_done"] or t >= t_c or es["cooldown"]:
            info["cooldown"] = dict(already=True, t_c=t_c, T=T, clock=sch["clock"])
        else:
            length = float(sch["cooldown_frac"]) * t
            if sch["clock"] != "wall":
                length = float(math.ceil(length))
            es["cooldown"] = dict(t_c=t, T=t + max(0.0, min(length, T - t)), clock=sch["clock"], at_step=step)
            info["cooldown"] = dict(es["cooldown"], already=False)
    else:
        es["stop"] = True
    es["triggered"] = info
    R.log.event("early_stop", **info)
    R.log.scalars({"early_stop/triggered": 1.0}, step)
    return es["stop"]


def stop_requested(R: Run) -> bool:
    """The manual stop: runs/<run_id>/STOP exists (one stat per optimizer step, whatever early_stop.enabled says) ->
    the "stop" path with reason "stop_file". Before it, the manual cooldown: runs/<run_id>/COOLDOWN (one more stat;
    cooldown_file)."""
    if (R.run_dir / COOLDOWN_FILE).exists():
        cooldown_file(R)
    return (R.run_dir / STOP_FILE).exists() and early_stop_trigger(R, "stop_file", "stop")


def cooldown_file(R: Run):
    """runs/<run_id>/COOLDOWN, acted on once (es["cooldown_file"], in the full state: a resume from a state saved after
    it does not act again; the file itself stays): outside a cooldown, the early stop's cooldown action with reason
    "cooldown_file" (early_stop_trigger: the WSD cooldown starts now over cooldown_frac x the progress so far, the run
    ends when it is over; summary.json's stopped_early), whatever early_stop.enabled says. Inside a cooldown (the
    scheduled one, an early stop's or 4d's) it changes nothing: one `cooldown_file` event {at_step, ignored} and no
    early_stop_trigger, whose "already" path would overwrite the trigger an earlier dev early stop left (and with it
    summary.json's stopped_early)."""
    es = R.st["early_stop"]
    if es.get("cooldown_file"):
        return
    step = R.st["step"]
    t, T = R.progress()
    if R.st["pre_cooldown_done"] or t >= R.cooldown_start(T) or es["cooldown"]:
        es["cooldown_file"] = dict(at_step=step, ignored="cooldown under way")
        R.log.event("cooldown_file", **es["cooldown_file"], epoch=R.st["epoch_progress"])
        return
    early_stop_trigger(R, "cooldown_file", "cooldown")
    es["cooldown_file"] = dict(at_step=step)


# ----------------------------------------------------------------------------------------------------- checkpoints


def _rng_state() -> dict:
    st = np.random.get_state()
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else [],
                numpy=[st[0], [int(x) for x in st[1]], int(st[2]), int(st[3]), float(st[4])], python=random.getstate())


def _set_rng_state(s: dict):
    torch.set_rng_state(s["torch"])
    if s.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    n = s["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    random.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in s["python"]))


def _replace_dir(tmp: Path, final: Path, tries: int = 150):
    """tmp -> final. On Windows a directory cannot be renamed while any file in it is open (PermissionError), and a
    virus scanner or the search indexer may still be reading a just-written multi-GB model.pt / optimizer.pt, so retry
    there for up to ~30 s (as kitsune.runlog._replace does for single files)."""
    if final.exists():
        shutil.rmtree(final)
    for i in range(tries):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            if os.name != "nt" or i == tries - 1:
                raise
            time.sleep(0.2)


def _sync_dir(p: Path):
    """fsync a directory, so the entries renamed into it are on disk (POSIX). Windows cannot open a directory for
    that (PermissionError); NTFS journals the rename itself."""
    if os.name == "nt":
        return
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _flush_dir(tmp: Path):
    """Every file of a just-written checkpoint dir to disk before it is renamed into place. Without it an unclean host
    crash (power loss, kernel panic) soon after a save can leave the rename on disk but not the data: model.pt and
    the rest empty or full of NUL bytes under the name --resume and vast/supervise.py take as the newest state. A
    process crash cannot tear it; this costs the write-back of the save (~8.6 GB for a full state)."""
    for f in sorted(tmp.rglob("*")):
        if f.is_file():
            fsync_path(f)
        elif f.is_dir():
            _sync_dir(f)
    _sync_dir(tmp)


def save_weights(R: Run, step: int, reason: str) -> Path:
    """bf16 weights (+ processor, student_meta.json with a `trained` block) -> checkpoints/step_<N>/, then upload.
    Family "ctc": a ParakeetForCTC dir by kitsune.ctc_student.save_ctc_student - the Parakeet processor and tokenizer
    files copied from the student dir, the CC-BY-4.0 attribution card as MODEL_CARD.md and README.md - which
    kitsune.ctc_student.load_ctc_student loads."""
    from kitsune import student as S

    name = f"step_{step}"
    d = R.ckpt_dir / name
    if d.exists() and step in R.st["weights"]:
        return d
    t0 = time.time()
    tmp = R.ckpt_dir / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    meta = dict(R.student_meta)
    # lr_phase: the phase of the last step, which scripts/05_evaluate.py puts in its history record of this checkpoint
    # (verdict v2 reads it)
    meta["trained"] = dict(run_id=R.run_dir.name, step=step, train_s=round(R.clock(), 1), epoch=R.st["epoch_progress"],
                           reason=reason, time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           init=str(R.cfg["student"]), last_objective=R.st["last_objective"],
                           lr_phase=lr_phase_name(R.st.get("lr_phase")))
    if is_ctc(R.cfg):
        from kitsune.ctc_student import save_ctc_student

        save_ctc_student(R.model, tmp, rpath(R.cfg["student"]), meta)
    else:
        S.save_student(R.model, tmp, R.processor, meta)
    _flush_dir(tmp)
    _replace_dir(tmp, d)
    _sync_dir(R.ckpt_dir)
    R.st["weights"].append(step)
    R.log.event("checkpoint", ckpt="weights", name=name, reason=reason, save_s=round(time.time() - t0, 1),
                gb=round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 3),
                disk_free_gb=_disk_free_gb(d))
    R.uploader.submit(d, name)
    return d


def save_full(R: Run, step: int, reason: str, upload: bool = False, keep: bool = False, *,
              scratch: bool = False) -> Path:
    """Everything --resume needs -> checkpoints/full_step_<N>/ (atomic), then keep the newest ckpt.keep_local. keep: a
    ckpt.full_at_fracs state, which rotation never deletes (st["fulls_kept"], in its own trainer.pt too). With the
    aux-CTC head, its weights go in as aux_ctc.pt (and only here). scratch: a timed state (TIMED_REASON) for the
    scratch repo, which the caller submits to R.scratch: the dir carries SCRATCH_MARK from its creation (renamed in
    with it), or gets it when this step's state already exists (the same-step skip: the weights are the same, and only
    trainer.pt/.json are rewritten, so the state records its reason and the timed cadence the caller just set: a
    resume from it is not due at once). In a run with timed states a pre_cooldown state at the step of an existing one
    is rewritten the same way (the full-run queue's reset picks the runs repo's state whose trainer.json says
    "pre_cooldown", and its smoke check reads the checkpoint event). trainer.pt and trainer.json name the saving host
    (kitsune.scratch.host_info), which a resume compares with its own."""
    from kitsune.scratch import host_info

    name = f"full_step_{step}"
    d = R.ckpt_dir / name
    t0 = time.time()
    newly_kept = keep and name not in R.st.setdefault("fulls_kept", [])
    if newly_kept:
        R.st["fulls_kept"].append(name)
    # the same-step rewrites of the full runs (a run with a scratch uploader); off elsewhere, as before them
    restate = scratch or (reason == "pre_cooldown" and getattr(R, "scratch", None) is not None)

    def trainer_state():
        st = copy.deepcopy(R.st)
        st["train_s"] = R.clock()
        st["fulls"] = sorted(set(st["fulls"]) | {step})
        trainer = dict(format=1, step=step, reason=reason, run_id=R.run_dir.name, cfg=R.cfg, st=st,
                       planner=R.planner.state_dict(), logger=R.log.state_dict(), rng=_rng_state(),
                       time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), host=host_info())
        brief = {k: v for k, v in trainer.items() if k not in ("rng", "st")}
        # the growing per-eval lists stay in trainer.pt only (dev_history: the full runs' dev evals, WP4a; one record
        # per dev eval in every state's brief would only grow it)
        brief["st"] = {k: v for k, v in st.items()
                       if k not in ("history", "smoke_losses", "mini_history", "dev_history")}
        return trainer, brief, st

    if not (d.exists() and step in R.st["fulls"]):
        tmp = R.ckpt_dir / f"{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        trainer, brief, st = trainer_state()
        torch.save(R.model.state_dict(), tmp / "model.pt")
        torch.save(R.opt.state_dict(), tmp / "optimizer.pt")
        torch.save(R.l2sp.state_dict(), tmp / "l2sp.pt")
        if getattr(R, "aux_ctc", None) is not None:
            torch.save(R.aux_ctc.state_dict(), tmp / "aux_ctc.pt")
        torch.save(trainer, tmp / "trainer.pt")
        (tmp / "trainer.json").write_text(json.dumps(brief, indent=1, default=str), encoding="utf-8")
        if upload and R.uploader.repo:  # renamed in with the dir: none meant for the Hub is ever there unmarked, so
            (tmp / UPLOAD_MARK).touch()  # a crash before the submit below still leaves build() the mark to go by
        if scratch:  # the same for a timed state: never a moment where rotation could take it before its upload
            (tmp / SCRATCH_MARK).touch()
        _flush_dir(tmp)
        _replace_dir(tmp, d)
        _sync_dir(R.ckpt_dir)
        R.st["fulls"] = st["fulls"]
        R.log.event("checkpoint", ckpt="full", name=name, reason=reason, save_s=round(time.time() - t0, 1),
                    gb=round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 3),
                    disk_free_gb=_disk_free_gb(d))
    elif reason == "end" or newly_kept or restate:
        # a periodic/after-smoke full state already holds this step's weights and optimizer, but its trainer state
        # predates the stop (early_stop.stop / triggered), or its becoming a kept fraction state (st["fulls_kept"]),
        # a timed state (st["last_timed_*"]) or the pre_cooldown one (st["pre_cooldown_done"] / ["pre_cooldown_full"]);
        # rewrite only trainer.pt/.json so a resume ends the run, keeps the state, or knows what the state is.
        # The pre_cooldown upload of this very dir may still be pending (the loop ended at its step: a STOP file and
        # a skipped step): one not started yet is cancelled and queued again after the rewrite, a running one gets up
        # to UPLOAD_WAIT_S to finish first, so it does not commit one trainer.pt's size with the other's hash
        fut = R.uploader.pending.get(d)
        if fut is not None and fut.cancel():
            upload = True
        elif fut is not None:
            concurrent.futures.wait([fut], timeout=UPLOAD_WAIT_S)  # not .result(): its error is the Uploader's to log
        # the same for a timed upload of this dir (a timed state saved at the loop's last step, or at the step the
        # pre_cooldown state falls on): its pointer hashed the old trainer.pt, and must name the bytes it commits. The
        # end phase has cancelled a queued one (abandon_timed); before it a queued one is waited for too (a cancel
        # would drop the state from the scratch repo). The only wait for a timed upload, and a rare one: the step's
        # timed save and the cooldown's start (or the run's end) must coincide
        sfut = getattr(getattr(R, "scratch", None), "pending", {}).get(d)
        if sfut is not None and not (reason == "end" and sfut.cancel()):
            concurrent.futures.wait([sfut], timeout=UPLOAD_WAIT_S)
        trainer, brief, _ = trainer_state()
        for fname, write in (("trainer.pt", lambda f: torch.save(trainer, f)),
                             ("trainer.json", lambda f: f.write_text(json.dumps(brief, indent=1, default=str),
                                                                     encoding="utf-8"))):
            tmp = d / f"{fname}.tmp"
            write(tmp)
            fsync_path(tmp)
            _replace_file(tmp, d / fname)
        _sync_dir(d)
        R.log.event("checkpoint", ckpt="full", name=name, reason=reason, trainer_only=True,
                    save_s=round(time.time() - t0, 1))
    if scratch:  # the same-step skip too: this step's state, whatever saved it first, is the one the scratch repo gets
        (d / SCRATCH_MARK).touch()
    if upload:
        if R.uploader.repo:  # the Uploader removes it once the upload succeeded (no repo: nothing ever would)
            (d / UPLOAD_MARK).touch()
        R.uploader.submit(d, name)
        if reason == "pre_cooldown" and (sc := getattr(R, "scratch", None)) is not None:
            sc.sync_due.set()  # a run with timed states: the loop syncs the logs next, as fresh as this state
    rotate_full(R)
    return d


def _disk_free_gb(p: Path) -> float | None:
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except OSError:
        return None


def rotate_full(R: Run):
    """Keep the newest ckpt.keep_local full states (~8.6 GB each for the real student). The one a resume started
    from is not special: once keep_local newer ones exist it goes too, or a resumed run fills the 80 GB disk. Kept past
    that: one still uploading, and one meant for the Hub whose upload has not succeeded (UPLOAD_MARK: the
    pre_cooldown state whose upload failed, rotated away at the end save by a periodic one in the cooldown, was on no
    disk and no Hub once the box was destroyed; vast/finish.py uploads and verifies it): normally one more dir. Never
    rotated, nor counted among the keep_local newest: the ckpt.full_at_fracs states (st["fulls_kept"]). Timed states
    (R.scratch): one whose scratch upload runs or waits is kept too, and so is the NEWEST dir with SCRATCH_MARK only -
    an older mark is one a cancelled or crashed upload left (the next success clears it), which must not pin its ~9 GB
    for the rest of the run."""
    keep = max(1, int(R.cfg["ckpt"]["keep_local"]))
    kept = set((getattr(R, "st", None) or {}).get("fulls_kept") or ())
    fulls = sorted((p for p in R.ckpt_dir.iterdir() if FULL_RE.match(p.name) and p.name not in kept),
                   key=lambda p: int(FULL_RE.match(p.name)[1]))
    sc = getattr(R, "scratch", None)
    busy = R.uploader.busy() | (sc.busy() if sc is not None else set())
    marked = [p for p in fulls if (p / SCRATCH_MARK).exists()]
    for p in fulls[:-keep]:
        if p in busy or (p / UPLOAD_MARK).exists() or (marked and p == marked[-1]):
            continue
        shutil.rmtree(p, ignore_errors=True)
        R.log.event("checkpoint_deleted", name=p.name, keep_local=keep)


def timed_meta(R: Run, step: int) -> dict:
    """The pointer fields of the timed state saved at `step` (kitsune.scratch.make_pointer's keyword arguments), taken
    when it is saved: its epoch, the planner's fingerprint and train utterances (what resume_check and a resume compare
    on the new host), the selection's sha256, micro_audio_s, the code sha and this host."""
    from kitsune import scratch

    ps = R.planner.state_dict()
    return dict(step=int(step), epoch=float(R.st["epoch_progress"]), planner_fingerprint=str(ps["fingerprint"]),
                n_train_utts=int(ps["n_utts"]), selection_sha256=selection_sha256(R),
                micro_audio_s=float(R.planner.micro_audio_s), kitsune_sha=scratch.code_sha(ROOT),
                host=scratch.host_info())


def timed_state(R: Run, t: float, step: int):
    """The loop's timed-state block, after the step's periodic saves (ckpt.upload_full_every_min; R.scratch, else
    nothing): records the uploads that went up since the last step in st["timed"]; when a state is due (every N minutes
    of loop clock t) saves it (reason TIMED_REASON, SCRATCH_MARK; a timed save is a full state, so the periodic full
    cadence restarts from it too) and queues its upload, unless one is still running: then nothing is saved, one
    `timed_state_skipped` per due window it misses says so (with the running upload's running_s: an upload that hangs
    - xet transfers have no timeout - shows as a growing number every window, not as one early event), and the state
    goes as soon as the upload ends (never two ~9 GB uploads at once); and runs the log sync a success or the
    pre_cooldown submit asked for (sync_due), once no sync is running - so the Hub's logs are as fresh as the newest
    state a new host would resume from. Timed uploads are never awaited (save_full's rare same-step rewrite aside)."""
    sc = getattr(R, "scratch", None)
    if sc is None:
        return
    st, n = R.st, len(sc.ok)  # the worker thread appends to sc.ok: read up to one length
    st["timed"], sc.seen = _timed_last(R, sc, n), n
    every = R.cfg["ckpt"]["upload_full_every_min"]
    if every is not None and due(t, step, st["last_timed_t"], st["last_timed_step"], every, None):
        if sc.busy():
            # the due windows since the last timed save (1: the first missed one); one event per window
            window = (st["last_timed_step"], int((t - st["last_timed_t"]) // (float(every) * 60)))
            if sc.skipped_at != window:
                sc.skipped_at = window
                R.log.event("timed_state_skipped", reason="busy", at_step=step,
                            uploading=sorted(p.name for p in sc.busy()), running_s=sc.running_s())
        else:
            st["last_timed_t"] = st["last_full_t"] = R.clock()  # set first: the state records itself
            st["last_timed_step"] = st["last_full_step"] = step
            d = save_full(R, step, TIMED_REASON, scratch=True)
            sc.submit(d, d.name, meta=timed_meta(R, step))
            R.log.event("timed_state", name=d.name, step=step, epoch=st["epoch_progress"])
    if sc.sync_due.is_set() and R.log.wait_sync(0):  # never behind a running (maybe stalled) sync: next step then
        sc.sync_due.clear()
        R.log.sync(force=True, wait=False)


def abandon_timed(R: Run, why: str = "end"):
    """The end phase and the failure path: cancel the queued timed uploads and stop the scratch thread without waiting
    (a running upload goes on until the process exits); one `timed_state_upload_abandoned` names what was cut off. No
    scratch uploader, or nothing pending: nothing, not even an event."""
    sc = getattr(R, "scratch", None)
    if sc is not None and (left := sc.abandon(0)):
        R.log.event("timed_state_upload_abandoned", names=left, at=why)


def requeue_timed(R: Run, full: Path, state: dict):
    """A resume (train; never a T/2 branch's start) whose checkpoints/ still hold SCRATCH_MARK: a crash cut a timed
    upload short (the full-run queue retries on the same host), so the scratch repo is a window or more behind, and
    nothing else would send a state before the next due window - a host that died meanwhile would cost up to two
    windows of training. The state resumed from (the newest one left: newer ones were set aside) goes up at once,
    marked as a timed save is; its pointer names the host that saved it, and bytes the Hub already holds only dedup.
    No save, and the cadence stays the state's. Not for an LR probe (no timed states) nor a run that goes straight to
    its end phase (early_stop.stop: the end phase would cancel it or wait for it). One `timed_state` event, requeued."""
    sc = getattr(R, "scratch", None)
    if sc is None or lr_probe_on(R.cfg) or R.st["early_stop"]["stop"]:
        return
    if not any((p / SCRATCH_MARK).is_file() for p in R.ckpt_dir.iterdir() if FULL_RE.match(p.name)):
        return
    from kitsune.scratch import host_info

    step = int(state["step"])
    (full / SCRATCH_MARK).touch()  # as a timed save's: rotation keeps it until the upload ends, a crash retries it
    sc.submit(full, full.name, meta=dict(timed_meta(R, step), host=state.get("host") or host_info()))
    R.log.event("timed_state", name=full.name, step=step, epoch=R.st["epoch_progress"], requeued=True)


def drop_scratch_marks(ckpt_dir: Path) -> list[str]:
    """A resume without a scratch uploader (hf.scratch_repo unset, e.g. by hand on another host) of a run that sent
    timed states: nothing will ever send the dirs a crash left SCRATCH_MARK in, so the marks go - rotate_full would
    otherwise pin the newest marked ~9 GB dir for the rest of the run, and the runs repo's uploads (which ignore only
    UPLOAD_MARK then) would carry the empty file. Returns the dirs' names (the `resume` event)."""
    names = []
    for p in sorted(ckpt_dir.iterdir()) if ckpt_dir.is_dir() else ():
        if FULL_RE.match(p.name) and (p / SCRATCH_MARK).is_file():
            try:
                (p / SCRATCH_MARK).unlink()
                names.append(p.name)
            except OSError:
                pass  # kept: rotation then keeps this dir too, as before
    return names


def _timed_last(R: Run, sc, n: int | None = None) -> dict | None:
    """st["timed"] with the uploads that went up since the loop last looked (sc.ok[sc.seen:n]) folded in."""
    last = R.st.get("timed")
    new = sc.ok[sc.seen:len(sc.ok) if n is None else n]
    if new:
        last = dict(new[-1], count=int((last or {}).get("count", 0)) + len(new))
    return last


def summary_timed(R: Run) -> dict:
    """summary.json's timed_states {repo, count, last}: the timed uploads that went up, across launches (st["timed"],
    and any that ended after the loop), or nothing for a run without a scratch uploader."""
    sc = getattr(R, "scratch", None)
    if sc is None:
        return {}
    last = _timed_last(R, sc)
    return dict(timed_states=dict(repo=sc.repo, count=int((last or {}).get("count", 0)), last=last))


def find_full_state(path: Path) -> Path:
    """--resume accepts a full_step_<N> dir or a run dir (its newest full state)."""
    path = Path(path)
    if FULL_RE.match(path.name) and (path / "trainer.pt").exists():
        return path
    ck = path / "checkpoints"
    fulls = sorted((p for p in ck.iterdir() if FULL_RE.match(p.name) and (p / "trainer.pt").exists()),
                   key=lambda p: int(FULL_RE.match(p.name)[1])) if ck.is_dir() else []
    if not fulls:
        raise SystemExit(f"--resume {path}: no checkpoints/full_step_<N>/trainer.pt found")
    return fulls[-1]


class ResumeMismatch(RuntimeError):
    """A resume whose full state's planner does not fit this host's train store (another selection file, a lost shard,
    a store built from other labels): the same (seed, epoch) would give another step order, so the epoch position means
    nothing. Raised by train() with a message starting "ResumeMismatch:" (summary.json's error then starts with it too);
    the full-run queue fails the item without a retry, which cannot change the store. resume_check finds the same
    before a resume, without the GPU."""


def resume_check(run_dir: Path) -> dict:
    """Whether the newest local full state of `run_dir` (find_full_state) fits this host's train store, on the CPU and
    without building anything: the store of the state's config (train_store_spec's rows, store_dir: the frame store
    for family "ctc") opened as it is on disk, and the planner built from it exactly as make_planner does with the
    state's micro_audio_s (st["memory"], the memory probe's choice). The full-run queue's check-resume runs it before
    a resume on a new host, after its stores phase. Returns {ok, reason, state_fingerprint, store_fingerprint,
    n_utts_state, n_utts_store, micro_audio_s}: reason None when ok, "store not built" when the store is not on disk
    yet (retryable), else "fingerprint mismatch" / "n_utts mismatch" (a resume would raise ResumeMismatch).
    FileNotFoundError when the run dir holds no full state."""
    from types import SimpleNamespace

    try:
        full = find_full_state(Path(run_dir))
    except SystemExit as e:  # not SystemExit: the caller is a library (kitsune.full_queue check-resume), not a CLI
        raise FileNotFoundError(str(e)) from None
    state = torch.load(full / "trainer.pt", map_location="cpu", weights_only=True)
    cfg = _merge(DEFAULTS, copy.deepcopy(state["cfg"]))  # keys added since it was written take the defaults
    micro = float((state["st"].get("memory") or {}).get("micro_audio_s", cfg["batch"]["micro_audio_s"]))
    sp = state.get("planner") or {}
    out = dict(ok=False, reason=None, state_fingerprint=sp.get("fingerprint"), store_fingerprint=None,
               n_utts_state=sp.get("n_utts"), n_utts_store=None, micro_audio_s=micro)
    name, _ = train_store_spec(cfg, SimpleNamespace(event=lambda kind, **kw: None))
    where = store_dir(cfg, name)
    complete = trainset._frame_cache_complete if is_ctc(cfg) else trainset._cache_complete
    if not ((where / "stores.json").is_file() and complete(where)):
        out["reason"] = "store not built"
        return out
    store = trainset.load_stores(where)
    planner = make_planner(SimpleNamespace(cfg=cfg, train=store), micro)
    out.update(store_fingerprint=planner.fingerprint, n_utts_store=len(store))
    if planner.fingerprint != out["state_fingerprint"]:
        out["reason"] = "fingerprint mismatch"
    elif out["n_utts_state"] is not None and int(out["n_utts_state"]) != len(store):
        out["reason"] = "n_utts mismatch"
    else:
        out["ok"] = True
    return out


def set_aside_newer(ckpt_dir: Path, step: int) -> list[str]:
    """A resume from an older full state (backing out of a loss spike) abandons the attempt that went past it. Left in
    place, that attempt's newer full_step_<M>/ would outrank the resumed run's own: rotate_full (the newest keep_local)
    deletes the resumed run's states instead, and a later --resume <run dir> (find_full_state; supervise_distill.py)
    silently continues the abandoned attempt; finish.py would upload its step_<M>/ weights next to the resumed run's.
    So every full_step_<M>/ and step_<M>/ with M > step moves to checkpoints/abandoned-<UTC stamp>/ (a rename:
    nothing is deleted). Every scanner of checkpoints/ (these, finish.py, both supervisors) reads only its top level.
    That includes the weights saved after the newest full state, when the resume is from that one: a crash inside
    the full-state save that follows the same step's weights (periodic or end), or the laptop's minute cadences.
    The resumed run replays the steps but not those saves: on the wall and minute clocks its weights fall due at other
    steps, and a re-fit budget may end it before M, so step_<M>/ would stay next to its own, uploaded and verified by
    finish.py (on the steps clock the same step_<M> comes back). The `resume` event lists the names (set_aside); a copy
    the Uploader already pushed stays on the Hub. Returns the names moved."""
    def newer(regex):
        return [p for p in ckpt_dir.iterdir() if (m := regex.match(p.name)) and int(m[1]) > step]

    moved = sorted(newer(FULL_RE) + newer(WEIGHTS_RE), key=lambda p: p.name)
    if not moved:
        return []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest, n = ckpt_dir / f"abandoned-{stamp}", 1
    while dest.exists():
        dest, n = ckpt_dir / f"abandoned-{stamp}-{n}", n + 1
    dest.mkdir()
    for p in moved:
        _replace_dir(p, dest / p.name)
    return [p.name for p in moved]


# ----------------------------------------------------------------------------------------------------------- smoke


def measure_flops(R: Run, mb: dict) -> float:
    """Forward FLOPs of one micro-batch (torch's FlopCounterMode) per padded audio second, for the MFU estimate.
    Encoder cost scales with padded audio; the decoder and head (~6 % for the real student) are folded in."""
    from torch.utils.flop_counter import FlopCounterMode

    aux = getattr(R, "aux_ctc", None) is not None
    with torch.no_grad(), frozen_eval(R.model), FlopCounterMode(display=False) as fc:
        if is_ctc(R.cfg):  # the encoder and the CTC head on every frame (the loss is small next to them)
            forward_ctc(R, mb, R.feat_eval)
        else:
            res = forward_logits(R, mb, R.feat_eval, with_encoder=aux)
            if aux:  # training compute too (its ~16k-class head on every encoder frame)
                aux_ctc_sum(R, res[3], mb)
    padded = float(mb["lengths"].max()) * len(mb["lengths"]) / SR
    return fc.get_total_flops() / max(padded, 1e-9)


def fwd_bwd(R: Run, mb: dict) -> tuple[float, bool]:
    """Forward and backward of one micro-batch on the training objective, the aux-CTC term included (so the memory
    probe's passes hold its (frames x V + 1) logits too); returns (loss, finite loss and gradients). Family "ctc": the
    CTC-KD objective over the micro-batch's own N_u (ctc_step_loss), with its (frames x 3073) fp32 log-probs."""
    if is_ctc(R.cfg):
        loss, _, _ = ctc_step_loss(R, mb, R.feat_train, int(mb["n_tok"].sum()))
    else:
        aux = getattr(R, "aux_ctc", None) is not None
        res = forward_logits(R, mb, R.feat_train, with_encoder=aux)
        losses = kd_losses(res[0], mb["top_idx"].to(R.device), mb["top_lp"].to(R.device))
        n = int(mb["top_idx"].shape[0])
        loss = kd_objective(losses, n, R.cfg["loss"]["w_kl"], R.cfg["loss"]["w_ce"])
        if aux:
            loss = loss + aux_ctc_weight(R.cfg) * aux_ctc_sum(R, res[3], mb) / n
        res = None
    loss.backward()
    norms = torch._foreach_norm([p.grad for p in R.params if p.grad is not None])
    ok = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(n)) for n in norms)
    R.opt.zero_grad(set_to_none=True)
    return float(loss.detach()), ok


def padded_row_check(R: Run) -> dict:
    """The smoke.pad_utts shortest train utterances, each padded next to the longest, vs the same utterances run
    alone. A masking bug (padding leaking into attention, the convolutions or the relative positions) moves every
    target position, so the gate is the mean KL(alone || padded) over all their positions (plus finiteness, and
    smoke.pad_min_argmax_agree over the pooled positions if set). One short utterance's argmax is no gate: under bf16
    autocast the padded and unpadded runs use different kernel shapes, and a 3-5 token utterance of a correct but
    untrained student flips on a single near-tie. With autocast on, the bound is at least 10x the bf16 noise floor
    measured on the same utterances (alone under autocast vs alone without it). Family "ctc": padded_row_check_ctc,
    the same gate on the frame log-probs."""
    if is_ctc(R.cfg):
        return padded_row_check_ctc(R)
    sm = R.cfg["smoke"]
    order = sorted(range(len(R.train)), key=lambda i: R.train.utts[i].duration)
    if len(order) < 2:
        return dict(skipped="fewer than 2 train utterances")
    longest, shorts = order[-1], order[:min(int(sm["pad_utts"]), len(order) - 1)]
    per = max(1, int(float(R.planner.micro_audio_s) // max(R.train.utts[longest].duration, 1e-3)) - 1)
    kl, noise, agree, diff, finite = [], [], [], 0.0, True

    def kl_rows(lp_ref, logits):  # per position KL(ref || logits), fp32 over the full vocabulary
        return (lp_ref.exp() * (lp_ref - F.log_softmax(logits.float(), -1))).sum(-1).cpu()

    with torch.no_grad(), frozen_eval(R.model):
        for c in range(0, len(shorts), per):
            chunk = shorts[c:c + per]
            both = R.ds[[longest, *chunk]]
            lb, rows, _ = forward_logits(R, both, R.feat_eval)
            for i in chunk:
                alone = R.ds[[i]]
                if R.ds.ids[i] not in both["ids"] or not alone["ids"]:
                    continue  # undecodable audio (the dataset logged it)
                la, _, _ = forward_logits(R, alone, R.feat_eval)
                pb = lb[rows == both["ids"].index(R.ds.ids[i])]
                finite = finite and bool(torch.isfinite(pb).all() and torch.isfinite(la).all())
                lpa = F.log_softmax(la.float(), -1)
                kl.append(kl_rows(lpa, pb))
                agree.append((la.argmax(-1) == pb.argmax(-1)).cpu())
                diff = max(diff, float((la.float() - pb.float()).abs().max()))
                if R.amp:
                    R.amp = False
                    try:
                        l32, _, _ = forward_logits(R, alone, R.feat_eval)
                    finally:
                        R.amp = True
                    noise.append(kl_rows(F.log_softmax(l32.float(), -1), la))
    if not kl:
        return dict(skipped="no decodable utterance")
    kl_t = torch.cat(kl)
    out = dict(n_utts=len(kl), n_tok=int(kl_t.numel()), kl_mean=float(kl_t.mean()), kl_max=float(kl_t.max()),
               argmax_agree=float(torch.cat(agree).float().mean()), max_abs_logit_diff=diff, finite=finite,
               pad_to_s=round(R.train.utts[longest].duration, 2), rows_per_forward=per + 1)
    bound = float(sm["pad_max_mean_kl"])
    if noise:
        out["bf16_noise_kl_mean"] = float(torch.cat(noise).mean())
        bound = max(bound, 10 * out["bf16_noise_kl_mean"])
    min_agree = sm.get("pad_min_argmax_agree")
    out.update(kl_bound=bound, ok=finite and out["kl_mean"] <= bound
               and (min_agree is None or out["argmax_agree"] >= float(min_agree)))
    return out


def padded_row_check_ctc(R: Run) -> dict:
    """padded_row_check of family "ctc": the smoke.pad_utts shortest train utterances, each padded next to the longest,
    vs the same utterances run alone, compared on their valid FRAMES' log-probs - a masking bug (padding leaking into
    the attention, the convolution module or the relative positions) moves every frame. The gate: the mean
    KL(alone || padded) per frame over all their valid frames <= smoke.pad_max_mean_kl (at least 10x the bf16 noise
    floor with autocast on), finite, and smoke.pad_min_argmax_agree over the pooled frames if set. n_tok counts
    frames here."""
    sm = R.cfg["smoke"]
    order = sorted(range(len(R.train)), key=lambda i: R.train.utts[i].duration)
    if len(order) < 2:
        return dict(skipped="fewer than 2 train utterances")
    longest, shorts = order[-1], order[:min(int(sm["pad_utts"]), len(order) - 1)]
    per = max(1, int(float(R.planner.micro_audio_s) // max(R.train.utts[longest].duration, 1e-3)) - 1)
    kl, noise, agree, diff, finite = [], [], [], 0.0, True

    def kl_rows(lp_ref, lp):  # per frame KL(ref || lp), fp32 over the 3073 classes
        return (lp_ref.exp() * (lp_ref - lp)).sum(-1).cpu()

    with torch.no_grad(), frozen_eval(R.model):
        for c in range(0, len(shorts), per):
            chunk = shorts[c:c + per]
            both = R.ds[[longest, *chunk]]
            lb, nb, _ = forward_ctc(R, both, R.feat_eval)
            for i in chunk:
                alone = R.ds[[i]]
                if R.ds.ids[i] not in both["ids"] or not alone["ids"]:
                    continue  # undecodable audio (the dataset logged it)
                la, na, _ = forward_ctc(R, alone, R.feat_eval)
                r, n = both["ids"].index(R.ds.ids[i]), int(na[0])
                pb, pa = lb[r, :n].float(), la[0, :n].float()
                finite = finite and int(nb[r]) == n and bool(torch.isfinite(pb).all() and torch.isfinite(pa).all())
                kl.append(kl_rows(pa, pb))
                agree.append((pa.argmax(-1) == pb.argmax(-1)).cpu())
                diff = max(diff, float((pa - pb).abs().max()) if n else 0.0)
                if R.amp:
                    R.amp = False
                    try:
                        l32, _, _ = forward_ctc(R, alone, R.feat_eval)
                    finally:
                        R.amp = True
                    noise.append(kl_rows(l32[0, :n].float(), pa))
    if not kl:
        return dict(skipped="no decodable utterance")
    kl_t = torch.cat(kl)
    out = dict(n_utts=len(kl), n_tok=int(kl_t.numel()), kl_mean=float(kl_t.mean()), kl_max=float(kl_t.max()),
               argmax_agree=float(torch.cat(agree).float().mean()), max_abs_logit_diff=diff, finite=finite,
               pad_to_s=round(R.train.utts[longest].duration, 2), rows_per_forward=per + 1, unit="frames")
    bound = float(sm["pad_max_mean_kl"])
    if noise:
        out["bf16_noise_kl_mean"] = float(torch.cat(noise).mean())
        bound = max(bound, 10 * out["bf16_noise_kl_mean"])
    min_agree = sm.get("pad_min_argmax_agree")
    out.update(kl_bound=bound, ok=finite and out["kl_mean"] <= bound
               and (min_agree is None or out["argmax_agree"] >= float(min_agree)))
    return out


def decode_preflight(R: Run):
    """Decode smoke.decode_per_set seeded rows of every train source and every eval set in this process, through the
    decode path the loader's workers use (Stores.wave -> kitsune.audio.decode_audio), first thing in the smoke phase.
    The dataset drops an undecodable row and goes on, so one bad upstream file cannot end a paid run; but a failure
    that hits a whole source - a codec, or the resampler that every 24 and 48 kHz file goes through (librosa / soxr /
    numba), broken on a new box - would only thin the data: the run would train without that source and judge the
    gate on the sets that are left. A set that fails on more than half its sample raises SmokeFailed with its first
    error; every set's count is a `smoke_decode` event. The dev store's sources too (eval.dev: the early stop's rows),
    after the others, so their draws are the ones before it."""
    n = int(R.cfg["smoke"]["decode_per_set"])
    if n <= 0:
        return
    rng = np.random.default_rng([int(R.cfg["seed"]), 16])
    sets, bad = {}, []
    dev = getattr(R, "devstore", None)
    for kind, store in (("train", R.train), ("eval", R.evalstore), *((("dev", dev),) if dev is not None else ())):
        by_src = {}
        for i, u in enumerate(store.utts):
            by_src.setdefault(u.source, []).append(i)
        for src, idx in sorted(by_src.items()):
            pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
            failed, first = 0, None
            for i in pick:
                try:
                    store.wave(int(i))
                except Exception as e:
                    failed += 1
                    first = first or f"{type(e).__name__}: {e}"[:300]
            sets[f"{kind}/{src}"] = dict(n=len(pick), failed=failed, first_error=first)
            if 2 * failed > len(pick):
                bad.append(f"{kind} {src} ({failed}/{len(pick)} rows: {first})")
    R.log.event("smoke_decode", sets=sets)
    if bad:
        raise SmokeFailed(f"audio decode fails for whole sets: {'; '.join(bad)}")


# the smoke's bound on mean |LogMel - HF| over the normalised features: a real featuriser bug (a lost pre-emphasis or
# dither scale, misaligned frames) gives 0.1 and more, FFT/TF32 rounding at most ~2.4e-4 (all-silence clips included)
LOGMEL_MEAN_DIFF_MAX = 1e-2


def train_dither_check(R: Run, lengths: list[int]) -> dict:
    """The training featuriser's dither drawn on the device itself: with perf.train_exact_dither false it comes from
    the device's RNG, a branch no CPU run reaches, and its numbers differ from HF's by design, so a comparison of
    features cannot see a wrong scale or placement. On a zero wave the noise is all there is: exactly 0 in the
    padding, std within 10 % of LogMel.dither on the valid samples, and each utterance drawn alone gets the noise it
    got in the batch."""
    feat, dev = R.feat_train, R.device
    S = max(lengths)
    valid = torch.arange(S, device=dev)[None, :] < torch.tensor(lengths, device=dev)[:, None]
    noise = feat._dither(torch.zeros(len(lengths), S, device=dev), lengths, valid)
    std = float(noise[valid].std())
    padding_zero = bool((noise[~valid] == 0).all())
    alone = all(torch.equal(feat._dither(torch.zeros(1, n, device=dev), [n], valid[i:i + 1, :n]), noise[i:i + 1, :n])
                for i, n in enumerate(lengths))
    ok = padding_zero and alone and abs(std / feat.dither - 1.0) <= 0.1
    return dict(ok=ok, std=std, dither=feat.dither, padding_zero=padding_zero, batch_invariant=alone,
                exact=feat.exact_dither, n=len(lengths), device=str(dev))


def hf_features(R: Run, waves: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """The HF extractor's features of `waves`, as the teacher saw them (smoke_checks' reference): the Cohere processor
    (kitsune.features.hf_reference, the teacher pass's call) or, family "ctc", the student dir's
    ParakeetFeatureExtractor, which the Parakeet label pass ran (80 mel, no dither)."""
    if is_ctc(R.cfg):
        out = R.processor.feature_extractor([np.asarray(w, dtype=np.float32) for w in waves], sampling_rate=SR,
                                            return_tensors="pt")
        return out["input_features"], out["attention_mask"]
    from kitsune.features import hf_reference

    return hf_reference(waves, path_or_repo=str(rpath(R.cfg["student"])))


def smoke_checks(R: Run):
    """Checks that cost seconds and catch the failures that would otherwise burn the paid hours silently."""
    from kitsune.patches import sdpa_backend_report

    log = R.log
    t0 = time.time()
    # LogMel vs the HF extractor on a few real utterances (bitwise on CPU; FFT/matmul rounding on CUDA). The max
    # |diff| is informational (TF32 takes it to ~0.1 on an all-silence clip); the mean and the masks fail the smoke,
    # and so does an error that keeps the check from running
    idx = list(range(min(4, len(R.train))))
    waves = [R.train.wave(i) for i in idx]
    bad = None
    try:
        ref, ref_mask = hf_features(R, waves)
        wave = torch.zeros(len(waves), max(len(w) for w in waves))
        for i, w in enumerate(waves):
            wave[i, :len(w)] = torch.from_numpy(w)
        got, mask = R.feat_eval(wave.to(R.device), torch.tensor([len(w) for w in waves], device=R.device))
        d = (got.cpu() - ref).abs() if got.shape == ref.shape else None
        diff, mean = (float(d.max()), float(d.mean())) if d is not None else (float("inf"), float("inf"))
        masks_equal = bool(torch.equal(mask.cpu(), ref_mask.bool()))
        log.event("smoke_logmel_vs_hf", max_abs_diff=diff, mean_abs_diff=mean, masks_equal=masks_equal,
                  n=len(waves), device=str(R.device))
        if not (masks_equal and math.isfinite(mean) and mean <= LOGMEL_MEAN_DIFF_MAX):
            bad = (f"LogMel differs from the HF extractor the teacher saw: mean |diff| {mean:.3g} (bound "
                   f"{LOGMEL_MEAN_DIFF_MAX}), max {diff:.3g}, masks equal {masks_equal}")
    except Exception as e:  # setup_processing already loaded this processor: the check could not run, not a skip
        err = f"{type(e).__name__}: {e}"[:300]
        log.event("smoke_logmel_vs_hf", error=err)
        bad = f"LogMel vs the HF extractor the teacher saw could not run: {err}"
    if bad:
        raise SmokeFailed(bad)
    if waves and R.feat_train.dither > 0:
        dith = train_dither_check(R, [len(w) for w in waves])
        log.event("smoke_train_dither", **dith)
        if not dith["ok"]:
            raise SmokeFailed(f"the training featuriser's dither is off: {dith}")

    worst = R.planner.worst_micro_batches()
    mb = R.ds[worst["longest"]]
    loss, ok = fwd_bwd(R, mb)
    targets = dict(targets=int(mb["n_tok"].sum()), frames=int(mb["n_frames"].sum())) if is_ctc(R.cfg) else dict(
        targets=int(mb["top_idx"].shape[0]))
    log.event("smoke_longest_fwd_bwd", ok=ok, loss=loss, utts=len(mb["ids"]),
              padded_s=round(float(mb["lengths"].max()) * len(mb["ids"]) / SR, 1), **targets)
    if not ok:
        raise SmokeFailed("non-finite loss or gradient on the longest padded micro-batch")

    pad = padded_row_check(R)
    log.event("smoke_padded_row", **pad)
    if pad.get("ok") is False:
        raise SmokeFailed(f"padded rows differ from the utterances alone: mean KL {pad['kl_mean']:.3g} (bound "
                          f"{pad['kl_bound']:.3g}), argmax agree {pad['argmax_agree']:.3f}, finite {pad['finite']}")

    log.event("smoke_sdpa", **sdpa_backend_report(R.device))
    R.st["flops_per_padded_s"] = measure_flops(R, mb)
    log.event("smoke_flops", gflops_fwd_per_padded_audio_s=round(R.st["flops_per_padded_s"] / 1e9, 3))
    if out_repo(R.cfg):
        hf_roundtrip(R)
    log.event("smoke_checks_done", wall_s=round(time.time() - t0, 1))


VRAM_MARGIN_GIB = 0.5  # left outside the Windows VRAM cap: cuBLAS/cuDNN kernels and handles load outside the allocator


def cap_vram(R: Run, max_frac: float | None = None):
    """Windows + CUDA: cap PyTorch's caching allocator, for the whole process, at the VRAM free now (plus what it
    already holds) minus VRAM_MARGIN_GIB. The Windows driver's sysmem fallback (on by default) serves an allocation that
    no longer fits the card from shared system memory instead of failing it, so an oversized micro-batch would run
    several times slower instead of raising OutOfMemoryError, and the memory probe's fallbacks (halve micro_audio_s,
    gradient checkpointing) and train_step's OOM skip would never fire. Under the cap they do. Linux has no fallback.
    max_frac (scripts/05_evaluate.py --vram-frac; the trainer passes none): also at most that fraction of the card, on
    any OS."""
    if R.device.type != "cuda" or (os.name != "nt" and max_frac is None):
        return
    idx = R.device.index if R.device.index is not None else torch.cuda.current_device()  # the API needs an index
    free, total = torch.cuda.mem_get_info(idx)
    held = torch.cuda.memory_reserved(idx)
    frac = min(1.0, max(0.0, (free + held - VRAM_MARGIN_GIB * 2**30) / total))
    if max_frac is not None:
        frac = min(frac, float(max_frac))
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    R.vram_cap_gb = round(frac * total / 2**30, 2)
    R.log.event("vram_cap", cap_gb=R.vram_cap_gb, free_gb=round(free / 2**30, 2), held_gb=round(held / 2**30, 2),
                total_gb=round(total / 2**30, 2), margin_gb=VRAM_MARGIN_GIB, fraction=round(frac, 4),
                **({"max_frac": float(max_frac)} if max_frac is not None else {}))


def probe_passes(R: Run, planner: trainset.StepPlanner, rec: dict):
    """The memory probe's measured passes: forward+backward on each of the plan's worst micro-batches, peak GiB
    allocated / reserved into rec (CUDA). train_step accumulates a step's micro-batches into the same gradients, so
    from the second micro-batch on, forward and backward run with the full fp32 gradients (4 B/param, 2.3 GiB for the
    real student) already allocated. When the plan has steps of more than one micro-batch every pass therefore starts
    from zero-filled gradients (backward adds into them in place, as in training) instead of none.
    memory.probe_extended (4c) adds the micro-batch with the most rows (both families: per-row buffers, the decoder's
    padded positions) and, for the frame planner, the one with the most CTC targets and the one with the largest CTC
    lattice (rows x frames x (2 max targets + 1): the CTC loss's alpha/beta). memory.probe_shapes adds a synthetic
    micro-batch per shape (synth_micro_batch, held within the micro-batch the way the planner packs one), probed as
    "shape:<name>" at the configured micro_audio_s; a shape of two or more rows whose target padded seconds (longest
    x rows) exceed the micro-batch in use - after an OOM fallback, the smaller one - is one training never meets there
    and is only listed, skipped, in rec["shapes"] (the memory_probe event's: n, padded_s, target_padded_s)."""
    cuda = R.device.type == "cuda"
    plan = planner.epoch_plan(0)
    rec["grads_held"] = held = max(len(step) for step in plan) > 1
    worst = planner.worst_micro_batches(plan)
    if getattr(R, "aux_ctc", None) is not None:  # the aux-CTC logits: (padded encoder frames) x (V + 1), fp32, with
        # their log_softmax and gradient - the micro-batch with the most padded audio, not necessarily the longest
        worst["most_padded_frames"] = max((mb for step in plan for mb in step),
                                          key=lambda mb: float(planner.dur[mb].max()) * len(mb))
    mem = (getattr(R, "cfg", None) or {}).get("memory") or {}  # the tests' stub R has no config: nothing extended
    if mem.get("probe_extended"):
        mbs = [mb for step in plan for mb in step]
        worst["most_rows"] = max(mbs, key=lambda mb: (len(mb), float(planner.dur[mb].max())))
        if planner.frames:
            worst["most_targets"] = max(mbs, key=lambda mb: int(planner.n_tok[mb].sum()))
            worst["largest_ctc_lattice"] = max(mbs, key=lambda mb: len(mb) * float(planner.dur[mb].max())
                                               * (2 * int(planner.n_tok[mb].max()) + 1))
    if mem.get("probe_shapes"):
        micro0 = float(R.cfg["batch"]["micro_audio_s"])
        rec["shapes"] = {}
        for shape in mem["probe_shapes"]:
            durs = [float(x) for x in shape["durations"]]
            info = dict(target_padded_s=round(max(durs) * len(durs), 3))
            if len(durs) > 1 and max(durs) * len(durs) > planner.micro_audio_s:  # never packed at this micro-batch
                info.update(n=0, padded_s=0.0, skipped=f"larger than micro_audio_s {planner.micro_audio_s:g}"
                            + (" after an OOM fallback" if planner.micro_audio_s < micro0 else ""))
            else:
                idx = synth_micro_batch(planner, durs, fit=True)  # within the micro-batch, as training packs it
                info.update(n=len(idx), padded_s=round(float(planner.dur[idx].max()) * len(idx), 3) if idx else 0.0)
                if idx:
                    worst[f"shape:{shape['name']}"] = idx
                else:
                    info["skipped"] = "no train row to stand in"
            rec["shapes"][shape["name"]] = info
    for name, idx in worst.items():
        if held:
            for p in R.params:
                p.grad = torch.zeros_like(p)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        fwd_bwd(R, R.ds[idx])  # frees the gradients at the end
        if cuda:
            rec["peak_gb"][name] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            rec["max_reserved_gb"] = max(rec.get("max_reserved_gb", 0.0),
                                         round(torch.cuda.max_memory_reserved() / 2**30, 2))


def synth_micro_batch(planner: trainset.StepPlanner, durations, fit: bool = False) -> list[int]:
    """A synthetic micro-batch for memory.probe_shapes: for each target duration, longest first, the unused train row
    whose duration is nearest (the longer one on a tie), among the rows the planner can use (a token planner excludes
    those past batch.max_dec_len). Store indices, in that order; fewer when the store has fewer rows. The shapes come
    from the full data's worst micro-batches (tools/full_plan.py worst_shapes), so a smoke run on a smaller selection
    probes the full run's memory peak with its own rows. fit: hold a shape of two or more rows to the planner's packing
    rule (trainset.pack_micro_batches: longest row x rows <= micro_audio_s, decoder positions within max_micro_tokens
    when set) by standing in only rows of at most micro_audio_s / rows seconds - where the nearest row would overshoot,
    the nearest shorter one - so the probe never measures a micro-batch that training cannot meet (one row stays
    exempt: the planner gives a row longer than the micro-batch one of its own)."""
    elig = np.ones(len(planner.dur), dtype=bool) if planner.frames else planner.dec_len <= planner.max_dec_len
    k = len(durations)
    if fit and k > 1:
        elig = elig & (planner.dur.astype(np.float64) * k <= float(planner.micro_audio_s))
        if not planner.frames and planner.max_micro_tokens is not None:
            elig = elig & (planner.dec_len * k <= planner.max_micro_tokens)
    order = np.flatnonzero(elig)
    order = order[np.argsort(planner.dur[order], kind="stable")]
    d, n = planner.dur[order], len(order)
    used, out = set(), []
    for target in sorted((float(x) for x in durations), reverse=True):
        j = int(np.searchsorted(d, target))
        lo, hi = j - 1, j
        while lo >= 0 and lo in used:
            lo -= 1
        while hi < n and hi in used:
            hi += 1
        if lo < 0 and hi >= n:
            break
        p = hi if hi < n and (lo < 0 or d[hi] - target <= target - d[lo]) else lo
        used.add(p)
        out.append(int(order[p]))
    return out


def memory_probe(R: Run) -> dict:
    """CUDA only. Run forward+backward on the micro-batches that stress memory most (longest audio, most decoder
    positions, most targets) with the optimizer state's bytes held in reserve, and the step's gradients too when a step
    has several micro-batches (probe_passes). On OOM halve micro_audio_s (down to memory.min_micro_audio_s); if that
    still fails, enable per-layer gradient checkpointing and start again from the configured size - never under
    bn.mode "train" (the recomputed forward would update the running stats a second time), where the probe fails below
    memory.min_micro_audio_s instead; BatchNorm stays in the run's mode (train_mode). With the aux-CTC head its logits
    are part of every pass (fwd_bwd), and the micro-batch with the most padded frames is probed too. Family "ctc": the
    frame planner's worst cases, the longest micro-batch and the one with the most padded frames, whose (frames x 3073)
    fp32 log-probs with their gradient are the CTC student's peak (kitsune.trainset.StepPlanner.worst_micro_batches).
    The choice is kept in the full state, so a resume uses the same batches. On Windows it runs under cap_vram's cap,
    so OOM means OOM."""
    cfg, log = R.cfg, R.log
    bn = bn_mode(cfg)
    micro0 = float(cfg["batch"]["micro_audio_s"])
    ckpt = cfg["memory"]["grad_ckpt"] is True
    if R.device.type != "cuda" or not cfg["memory"]["probe_longest_bucket"]:
        log.event("memory_probe", skipped=f"device {R.device.type}" if R.device.type != "cuda" else "disabled",
                  micro_audio_s=micro0, grad_ckpt=ckpt)
        return dict(micro_audio_s=micro0, grad_ckpt=ckpt)
    if ckpt:
        R.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        train_mode(R.model, bn)
    micro = micro0
    # exp_avg + exp_avg_sq, allocated lazily at the first step (in host memory under optim.offload "cpu")
    adam_bytes = 0 if offloaded(R) else 8 * sum(p.numel() for p in R.params)
    while True:
        planner = make_planner(R, micro) if micro != float(R.planner.micro_audio_s) else R.planner
        rec, reserve, err = dict(peak_gb={}), None, None
        try:
            reserve = torch.empty(adam_bytes, dtype=torch.uint8, device=R.device)
            probe_passes(R, planner, rec)
        except torch.OutOfMemoryError as e:
            err = str(e)[:300]
        reserve = None
        if err is None:
            R.planner = planner
            total = torch.cuda.get_device_properties(R.device).total_memory / 2**30
            # 4c: extended (memory.probe_extended), the synthetic shapes' {n, padded_s[, skipped]} and the headroom
            # the worst pass left on the card (the full runs' budget: <= 28 GiB of the 5090's 32)
            mr = rec.get("max_reserved_gb")
            log.event("memory_probe", ok=True, micro_audio_s=micro, grad_ckpt=ckpt, peak_gb=rec["peak_gb"],
                      grads_held=rec["grads_held"], max_reserved_gb=mr,
                      vram_cap_gb=R.vram_cap_gb, device_gb=round(total, 1),
                      reserved_adam_gb=round(adam_bytes / 2**30, 2), extended=bool(cfg["memory"]["probe_extended"]),
                      headroom_gb=round(total - mr, 2) if mr is not None else None, shapes=rec.get("shapes"))
            return dict(micro_audio_s=micro, grad_ckpt=ckpt)
        # outside the except: its traceback no longer pins the failed pass's activations
        R.opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        log.event("oom_fallback", micro_audio_s=micro, grad_ckpt=ckpt, peaks_so_far=rec["peak_gb"], error=err)
        nxt = probe_fallback(cfg, micro, micro0, ckpt)
        if nxt is None:
            raise torch.OutOfMemoryError(f"OOM even at micro_audio_s={micro} with grad_ckpt={ckpt}"
                                         + (" (bn.mode 'train': gradient checkpointing is never used)"
                                            if bn == "train" else "") + f": {err}")
        if nxt[1] and not ckpt:
            R.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            train_mode(R.model, bn)  # the run's BN mode: frozen stays frozen (a train-mode BN never gets here)
        micro, ckpt = nxt


def probe_fallback(cfg: dict, micro: float, micro0: float, ckpt: bool) -> tuple[float, bool] | None:
    """The memory probe's next (micro_audio_s, grad_ckpt) after an OOM at (micro, ckpt): half the micro-batch while
    that stays >= memory.min_micro_audio_s; then, once, gradient checkpointing from the configured micro0 (under
    memory.grad_ckpt "auto", never under bn.mode "train"); None: nothing left to try."""
    if micro / 2 >= float(cfg["memory"]["min_micro_audio_s"]):
        return micro / 2, ckpt
    if not ckpt and cfg["memory"]["grad_ckpt"] == "auto" and bn_mode(cfg) != "train":
        return micro0, True
    return None


def smoke_profiler(R: Run, smoke_n: int):
    """perf.profile_smoke ("auto": on under CUDA only): a kitsune.profiling.SmokeProfiler over ~20 real training steps
    of the smoke phase (profiling.profile_window: after its warm-up steps, inside smoke.steps), writing under
    runs/<run_id>/smoke/profile/ and logging `smoke_profile` events - the evidence for where the first A100 run's fixed
    cost per micro-batch goes, before any change for speed. It costs a few minutes of host time on the A100
    (extrapolated; profiling's docstring, "cost"), paid in the loop: summary.json's wall_s and cycles[].process_s say
    how many. CPU runs record CPU activities only. None when it is off, the smoke phase is off or too short to record
    a step (a `smoke_profile` event says so), or this launch starts past the window (a resume)."""
    ps = R.cfg["perf"]["profile_smoke"]
    if not (R.device.type == "cuda" if ps == "auto" else ps) or not smoke_n or R.st["smoke_done"]:
        return None
    from kitsune import profiling

    win = profiling.profile_window(smoke_n)
    if win is None:
        R.log.event("smoke_profile", skipped=f"smoke.steps {smoke_n} is too few to record a step after the warm-up")
        return None
    if R.st["step"] >= win[0]:
        return None
    return profiling.SmokeProfiler(R.run_dir / "smoke" / "profile", R.device, *win, emit=R.log.event)


def smoke_end(R: Run):
    """After the first smoke.steps steps: at most smoke.max_dropped_frac of the rows undecodable (a failure only the
    loader's worker processes hit; decode_preflight covers the main process), finite, falling loss and enough
    throughput (else exit 3; it counts decoded audio only, so it cannot see rows that were dropped)."""
    cfg = R.cfg["smoke"]
    losses = R.st["smoke_losses"]
    q = max(1, len(losses) // 4)
    first, last = float(np.mean(losses[:q])), float(np.mean(losses[-q:]))
    rate = R.st["smoke_audio_s"] / max(R.st["smoke_time_s"], 1e-9)
    ok_loss = all(math.isfinite(x) for x in losses) and last < first
    dropped = int(R.st["smoke_dropped"])
    frac = dropped / max(dropped + int(R.st["smoke_utts"]), 1)
    R.log.event("smoke_steps", steps=len(losses), loss_first=first, loss_last=last, loss_decreasing=ok_loss,
                audio_s_per_s=round(rate, 1), floor=cfg["min_audio_s_per_s"], dropped=dropped,
                dropped_frac=round(frac, 4), max_dropped_frac=cfg["max_dropped_frac"])
    R.st["smoke_done"] = True
    if frac > float(cfg["max_dropped_frac"]):
        raise SmokeFailed(f"{dropped} of {dropped + int(R.st['smoke_utts'])} rows had undecodable audio over the smoke "
                          f"steps ({frac:.1%} > smoke.max_dropped_frac {cfg['max_dropped_frac']}; `dropped_audio` "
                          "events name them)")
    if rate < float(cfg["min_audio_s_per_s"]):
        raise ThroughputTooLow(f"{rate:.0f} audio-s/s < {cfg['min_audio_s_per_s']} after {len(losses)} steps")
    if not ok_loss and cfg["require_loss_decrease"]:
        raise SmokeFailed(f"loss did not fall over the smoke steps ({first:.4f} -> {last:.4f})")
    if R.cfg["ckpt"]["full_after_smoke"]:
        save_full(R, R.st["step"], "after_smoke")


# ------------------------------------------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="JSON config (keys: DEFAULTS in this file); default = viability")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key, e.g. --set hf.output_repo=user/repo (repeatable; JSON values)")
    ap.add_argument("--resume", default=None, help="checkpoints/full_step_<N> dir (or a run dir: its newest)")
    return ap.parse_args(argv)


def resume_overrides(saved: dict, overrides: list[tuple[str, object]]) -> tuple[dict, dict]:
    """A resume's --set overrides split into the ones that change the checkpoint's config and the ones that repeat it
    (scripts/supervise_distill.py passes the fresh start's --set on every attempt). Compared with the checkpoint's
    cfg, never with st["memory"]: the memory probe halving micro_audio_s is not a change. A change to a RESUME_FIXED
    key stops here, naming it, rather than as a planner-fingerprint error later or not at all (micro_audio_s: the
    planner is built from the probe's saved choice, so a new value was ignored while the logs said it was applied)."""
    changed, same = {}, {}
    for key, value in overrides:
        old = saved
        for part in key.split("."):
            old = old[part]
        if old == value:
            same[key] = value
            continue
        if any(key == f or f.startswith(key + ".") or key.startswith(f + ".") for f in RESUME_FIXED):
            raise SystemExit(f"--set {key}={json.dumps(value, default=str)} differs from the checkpoint's "
                             f"{json.dumps(old, default=str)}: it changes the step plan, so a resume cannot keep its "
                             "epoch position; start a new run (optim.*, loss.*, augment.* and memory.grad_ckpt can be "
                             "changed on resume)")
        changed[key] = value
    return changed, same


def branch_differences(parent_cfg: dict, cfg: dict) -> list[str]:
    """The config keys whose values differ between a T/2 branch (cfg) and its parent's state (parent_cfg, with the
    keys added since it was written at their defaults), leaving out BRANCH_FREE: the run name, the branch block, the Hub
    repo, the fraction evals, checkpoints and logging."""
    a, b = dict(_leaves(_merge(DEFAULTS, copy.deepcopy(parent_cfg)))), dict(_leaves(cfg))
    missing = object()

    def free(k):
        return any(k == f or (f.endswith(".") and k.startswith(f)) for f in BRANCH_FREE)

    return sorted(k for k in set(a) | set(b) if not free(k) and a.get(k, missing) != b.get(k, missing))


def branch_start_state(cfg: dict) -> tuple[Path, dict]:
    """A T/2 branch's start (branch.parent set, a fresh launch): the parent run dir's local full state at step
    round(resume_frac x M) (M: the parent's schedule.max_steps, from its newest full state; a missing state is an
    error - the parent keeps it by ckpt.full_at_fracs), checked to be the parent at that step under the same config
    as this one but for BRANCH_FREE (branch_differences), before its early stop fired and past its warm-up. Returns
    (that dir, its trainer.pt with st prepared for the branch): the parent's counters, clocks, history and planner
    position go on (the branch IS that run with a T/2 budget); its checkpoint lists start empty; st["branch"] holds
    the schedule - end_step round(end_frac x M), the cooldown from t_c = the resume step (Run.max_steps,
    Run.cooldown_start)."""
    br = cfg["branch"]
    parent = rpath(br["parent"])
    try:
        newest = find_full_state(parent)
    except SystemExit:
        raise SystemExit(f"branch.parent {parent}: not a run dir with a full state (checkpoints/full_step_<N>/)"
                         ) from None
    pcfg = torch.load(newest / "trainer.pt", map_location="cpu", weights_only=True)["cfg"]
    M = pcfg["schedule"].get("max_steps")
    if pcfg["schedule"].get("clock") != "steps" or not M:
        raise SystemExit(f"branch.parent {parent}: not a steps-clock run (schedule.clock "
                         f"{pcfg['schedule'].get('clock')!r}, max_steps {M!r})")
    rs, es = frac_step(br["resume_frac"], M), frac_step(br["end_frac"], M)
    full = parent / "checkpoints" / f"full_step_{rs}"
    if not (full / "trainer.pt").is_file():
        raise SystemExit(f"branch.parent {parent}: no local full state at step {rs} = round(resume_frac "
                         f"{br['resume_frac']} x max_steps {M}) ({full}; the parent keeps it with ckpt.full_at_fracs "
                         f"[{br['resume_frac']}])")
    state = torch.load(full / "trainer.pt", map_location="cpu", weights_only=True)
    if (diff := branch_differences(state["cfg"], cfg)):
        raise SystemExit(f"branch.parent {parent}: its config differs from this one in {diff} (a branch may change "
                         f"only {', '.join(k.rstrip('.') for k in BRANCH_FREE)})")
    if int(state["step"]) != rs or es <= rs:
        raise SystemExit(f"branch.parent {parent}: state at step {state['step']}, resume step {rs}, end step {es}")
    if state["st"]["early_stop"].get("triggered"):
        raise SystemExit(f"branch.parent {parent}: its early stop fired before step {rs}: no T/2 schedule to branch")
    if int(cfg["schedule"]["warmup_steps"]) >= rs:
        raise SystemExit(f"schedule.warmup_steps {cfg['schedule']['warmup_steps']} does not end before the branch's "
                         f"cooldown starts at step {rs}")
    st = copy.deepcopy(state["st"])
    st.update(weights=[], fulls=[], fulls_kept=[], pre_cooldown_done=False, pre_cooldown_full=None, resumes=0,
              started_utc=None)  # the branch's own start (train())
    st["branch"] = dict(parent_run_id=parent.name, resume_step=rs, end_step=es, t_c=float(rs), parent_dir=str(parent),
                        parent_state=str(full), parent_max_steps=int(M), resume_frac=float(br["resume_frac"]),
                        end_frac=float(br["end_frac"]))
    return full, dict(state, st=st, logger=None)


def build(args) -> tuple[Run, dict | None]:
    """Config, run dir and logger; the rest of the setup happens in train() so every failure is logged. A resume
    restores the run dir's state; a T/2 branch (branch.parent) starts a new run dir from its parent's
    (branch_start_state), named <parent run_name>-half unless this config's run_name differs from the parent's."""
    from kitsune.runlog import RunLogger

    state = branch_full = None
    if args.resume:
        full = find_full_state(Path(args.resume))
        state = torch.load(full / "trainer.pt", map_location="cpu", weights_only=True)
        cfg = _merge(DEFAULTS, copy.deepcopy(state["cfg"]))  # keys added since it was written take the defaults
        saved = copy.deepcopy(cfg)
        overrides, repeated = resume_overrides(saved, [apply_set(cfg, s) for s in args.set])
        validate(cfg)
        new_epochs = cfg["schedule"]["epochs"]
        planned = cfg["schedule"]["clock"] == "epochs" and not state["st"].get("total_steps")
        if (new_epochs != saved["schedule"]["epochs"] and new_epochs is not None and not planned
                and not cfg["schedule"]["resume_reset"]):
            # a new epoch count that would change nothing: plan_epochs keeps the plan in the state's total_steps (or
            # the clock is not "epochs"). Planned is the one case it takes: the clock switched to "epochs" on this
            # resume, from a state without a plan (plan_epochs in train)
            raise SystemExit(f"--set schedule.epochs {new_epochs!r} on a resume (the state's "
                             f"{saved['schedule']['epochs']!r}, its plan {state['st'].get('total_steps')!r} steps, "
                             f"schedule.clock {cfg['schedule']['clock']!r}; overrides {sorted(overrides)}) would "
                             "change nothing: it needs --set schedule.resume_reset=true, which plans the run again "
                             "(the module docstring's resume reset)")
        run_dir = full.parent.parent
        if out_repo(cfg) and state.get("host") is not None and not (run_dir / "events.jsonl").is_file():
            # a state that records its host (every state this trainer saves) in a run dir without the run's logs: a
            # new host whose logs were not pulled. Before the logger exists: its first sync would replace the Hub's
            # events.jsonl, scalars.jsonl and steps.parquet with this launch's rows only
            raise SystemExit(f"--resume {full}: {run_dir} has no events.jsonl. On a new host pull the run's logs from "
                             f"{out_repo(cfg)} (runs/{run_dir.name}/, without checkpoints/) into it first (the full "
                             "runs: python -m kitsune.full_queue resume-pull); resuming without them would overwrite "
                             "the run's logs on the Hub with this launch's")
    else:
        cfg = load_config(args.config, args.set)
        if cfg["schedule"]["resume_reset"]:
            raise SystemExit("schedule.resume_reset is a resume's one-shot flag: pass it with --resume <state> (--set "
                             "schedule.resume_reset=true), never in a config or a fresh start")
        overrides = repeated = {}
        if cfg["branch"]["parent"]:
            branch_full, state = branch_start_state(cfg)
            if cfg["run_name"] == state["cfg"]["run_name"]:
                cfg["run_name"] = f"{state['cfg']['run_name']}-half"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = rpath(cfg["runs_root"]) / f"{cfg['run_name']}-{stamp}"
        n = 1
        while run_dir.exists():
            run_dir = rpath(cfg["runs_root"]) / f"{cfg['run_name']}-{stamp}-{n}"
            n += 1
        run_dir.mkdir(parents=True)
    if os.environ.get("KITSUNE_SHARING"):
        torch.multiprocessing.set_sharing_strategy(os.environ["KITSUNE_SHARING"])
    dev = cfg["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if dev in (None, "auto") else dev)
    R = Run(cfg=cfg, run_dir=run_dir, device=device, amp=cfg["autocast"] == "bfloat16")
    every, scratch_repo = cfg["ckpt"]["upload_full_every_min"], cfg["hf"]["scratch_repo"]
    if every is not None and out_repo(cfg) and not scratch_repo:  # before the logger: nothing to close on the way out
        raise SystemExit("ckpt.upload_full_every_min is set but hf.scratch_repo is not: a run that asks for timed states "
                         "must be able to send them (the full-run queue passes --set hf.scratch_repo=$KITSUNE_SCRATCH_REPO)")
    api = hf_api() if out_repo(cfg) or cfg["hf"]["scratch_repo"] else None
    lg = cfg["log"]
    from kitsune import student as S

    R.log = RunLogger(run_dir, cfg, out_repo(cfg), lg["sync_every_min"], resume=state["logger"] if state else None,
                      student_meta=S.load_meta(rpath(cfg["student"])), capture=lg["capture_env"],
                      train_utts_flush_steps=lg["train_utts_flush"], api=api, scalars_parquet=lg["scalars_parquet"])
    R.uploader = Uploader(api, out_repo(cfg), run_dir.name, bool(cfg["hf"]["private"]), R.log)
    if scratch_repo and out_repo(cfg):
        R.scratch = ScratchUploader(api, scratch_repo, run_dir.name, R.log)
        R.uploader.ignore.append(SCRATCH_MARK)  # a pre_cooldown / end state may be a timed one's dir
    elif every is not None:  # no output repo (the laptop, the tests): nowhere a resume could pull a state from
        R.log.event("timed_state_skipped", reason="no output repo")
    if branch_full is not None:
        R.resumed_from, R.branch_start = branch_full, True
    elif state is not None:
        R.resumed_from = full
        moved = set_aside_newer(R.ckpt_dir, int(state["step"]))  # before anything is saved or rotated
        # the pre_cooldown full state goes up only from the process that saved it (finish.py's syncs take the newest
        # full state, the post-crash one none): one whose upload has not succeeded (UPLOAD_MARK: a crash cut it short,
        # or every retry failed) goes again, in the background (busy() keeps it from rotation meanwhile). One already
        # on the Hub does not: the hub dedups only the wire, hf_xet still reads and hashes all ~8.6 GB on the box. The
        # same for the ckpt.full_at_fracs states meant for the Hub ("frac:<f>"; only those carry the mark)
        pc = state["st"].get("pre_cooldown_full")
        again = pc if (pc and out_repo(cfg) and "pre_cooldown" in cfg["ckpt"]["upload_full_at"]
                       and (R.ckpt_dir / pc / UPLOAD_MARK).is_file()) else None
        again_fracs = [n for n in state["st"].get("fulls_kept") or () if out_repo(cfg) and n != pc
                       and (R.ckpt_dir / n / UPLOAD_MARK).is_file()]
        from kitsune.scratch import host_info

        # host: this one; prev_host: the one that saved the state (None: a state from before it was recorded);
        # new_host: they differ (hostname, vast machine or container), i.e. a resume on another box
        host, prev = host_info(), state.get("host")
        new_host = None if prev is None else any(prev.get(k) != host[k] for k in ("hostname", "machine_id",
                                                                                   "container_id"))
        # no scratch uploader: marks a timed upload cut short left behind go (with one, train() sends the state)
        dropped = drop_scratch_marks(R.ckpt_dir) if getattr(R, "scratch", None) is None else []
        R.log.event("resume", from_state=str(full), at_step=state["step"], overrides=overrides, unchanged=repeated,
                    config_arg_ignored=args.config, set_aside=moved, upload_again=again,
                    **({"upload_again_fracs": again_fracs} if again_fracs else {}), host=host, prev_host=prev,
                    new_host=new_host, **({"scratch_marks_dropped": dropped} if dropped else {}))
        for name in ([again] if again else []) + again_fracs:
            R.uploader.submit(R.ckpt_dir / name, name)
    return R, state


def resume_reset_start(R: Run) -> dict:
    """schedule.resume_reset (a resume's one-shot flag; build refuses it on a fresh start), right after train()
    restored the state: the run continues past its early stop or its cooldown with a fresh schedule. Refused on a T/2
    branch (its schedule is its parent's) and while runs/<run_id>/STOP exists (the stop would end it at once). Clears
    st total_steps (plan_epochs plans schedule.epochs again, a --set value included), pre_cooldown_done and
    pre_cooldown_full (the new cooldown saves its own pre_cooldown state), the early-stop state (fresh: patience,
    smoothing, trigger, cooldown; dev_history is kept, and early_stop.enabled decides whether it can fire again) and
    4d's deadline_cooldown; renames runs/<run_id>/COOLDOWN to COOLDOWN.consumed-<UTC stamp> (else it would fire again at
    once) and deletes the runs repo's copy (drop_hub_cooldown: the log syncs never delete, so a later resume on a new
    host would pull it back and cut the continuation short; when that delete fails on a run that has had a COOLDOWN,
    the fresh early-stop state counts the file as acted on, so a copy pulled back is ignored - and so is a new COOLDOWN
    in this continuation: the event warns); counts st["resume_resets"]; and sets the flag false in R.cfg before
    anything is saved, so every later save and a crash-resume from one do not reset again, while a repeat of the same
    argv before the first save resets the same state the same way. Returns what resume_reset_finish records."""
    st, cfg = R.st, R.cfg
    if R.branch_start or st.get("branch"):
        raise SystemExit("schedule.resume_reset: a T/2 branch keeps its parent's schedule; it cannot be reset")
    if (R.run_dir / STOP_FILE).exists():
        raise SystemExit(f"schedule.resume_reset with {R.run_dir / STOP_FILE} present: the STOP file would end the "
                         "continued run at once; remove it first")
    before = dict(at_step=int(st["step"]), time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  total_steps_before=st.get("total_steps"), early_stop_before=copy.deepcopy(st["early_stop"]),
                  pre_cooldown_full_before=st.get("pre_cooldown_full"),
                  lr_phase_before=lr_phase_name(st.get("lr_phase")),
                  deadline_cooldown_before=copy.deepcopy(st.get("deadline_cooldown")))
    st.update(total_steps=None, pre_cooldown_done=False, pre_cooldown_full=None, early_stop=early_stop_state(),
              deadline_cooldown=None)
    cool = R.run_dir / COOLDOWN_FILE
    if cool.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest, n = R.run_dir / f"{COOLDOWN_FILE}.consumed-{stamp}", 1
        while dest.exists():
            dest, n = R.run_dir / f"{COOLDOWN_FILE}.consumed-{stamp}-{n}", n + 1
        cool.replace(dest)
        before["cooldown_file_consumed"] = dest.name
    hub = drop_hub_cooldown(R)  # after the rename: a log sync from now on no longer carries the file
    if hub is not None:
        before["cooldown_hub"] = hub
        # the run has had a COOLDOWN (renamed now, by an earlier reset attempt, or acted on by the state): the Hub may
        # hold its copy
        had = (before.get("cooldown_file_consumed") or before["early_stop_before"].get("cooldown_file")
               or next(R.run_dir.glob(f"{COOLDOWN_FILE}.consumed-*"), None))
        if hub.startswith("failed") and had:
            st["early_stop"]["cooldown_file"] = dict(at_step=int(st["step"]), ignored="consumed by resume_reset")
            before["cooldown_warning"] = (f"the runs repo may still hold runs/{R.run_dir.name}/{COOLDOWN_FILE}: this "
                                          "continuation ignores a COOLDOWN file (use STOP, or delete the Hub copy and "
                                          "reset again)")
    st["resume_resets"] = int(st.get("resume_resets") or 0) + 1
    cfg["schedule"]["resume_reset"] = False
    return before


def drop_hub_cooldown(R: Run) -> str | None:
    """Delete runs/<run_id>/COOLDOWN from the runs repo (hf.output_repo, through the checkpoint Uploader's api) for
    resume_reset_start. The log syncs upload the run dir without deleting anything, so the copy of a COOLDOWN the reset
    consumed stays there, and a later resume on a new host (kitsune.full_queue resume-pull brings back runs/<id>/*)
    would put it back in the run dir, where the continuation's fresh early-stop state would act on it at once. Waits
    (bounded) for a log sync under way first, which may still carry the file. Returns None without an output repo,
    "absent", "deleted" or "failed: <error>" (never raises: the reset goes on, resume_reset_start falls back)."""
    up = getattr(R, "uploader", None)
    api, repo = getattr(up, "api", None), getattr(up, "repo", None)
    if api is None or not repo:
        return None
    wait_sync = getattr(R.log, "wait_sync", None)
    if wait_sync is not None and not wait_sync(120):  # none runs before the loop today; bounded all the same
        return "failed: a log sync that may carry the file is still running"
    path = f"runs/{R.run_dir.name}/{COOLDOWN_FILE}"
    try:
        if not api.file_exists(repo, path, repo_type="model"):
            return "absent"
        api.delete_file(path, repo_id=repo, repo_type="model",
                        commit_message=f"{R.run_dir.name}: {COOLDOWN_FILE} consumed by a resume reset")
        return "deleted"
    except Exception as e:  # noqa: BLE001  (network, auth: recorded in the resume_reset event)
        return f"failed: {type(e).__name__}: {e}"[:500]


def resume_reset_finish(R: Run, before: dict):
    """resume_reset_start's second half, after plan_epochs planned the schedule again: the new T must lie past the
    state's step (else the continuation would end at once); a record {at_step, time_utc, total_steps_before,
    total_steps_after, epochs, early_stop_before, pre_cooldown_full_before, lr_phase_before, ...} joins
    st["schedule_resets"] (summary.json's) and goes out as the `resume_reset` event. A state saved inside its cooldown
    (lr_phase_before "cooldown") puts the LR back at its peak: the event warns."""
    t, T = R.progress()
    if T <= t:
        raise SystemExit(f"schedule.resume_reset: the new schedule ends at {T:g} (schedule.clock "
                         f"{R.cfg['schedule']['clock']!r}), not after the state's {t:g}: raise schedule.epochs (or "
                         "max_steps / train_hours) with --set")
    rec = dict(before, total_steps_after=R.st.get("total_steps"), T=T, epochs=R.cfg["schedule"]["epochs"],
               clock=R.cfg["schedule"]["clock"], resume_resets=R.st["resume_resets"])
    if before["lr_phase_before"] == "cooldown":
        rec["warning"] = "the state was saved inside its cooldown: the LR goes back to its peak"
    R.st["schedule_resets"].append(rec)
    R.log.event("resume_reset", **rec)


def train(R: Run, state: dict | None) -> int:
    from kitsune import evaluate as ev

    cfg, log = R.cfg, R.log
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]) % 2**32)
    random.seed(int(cfg["seed"]))
    if R.device.type == "cuda":
        torch.set_float32_matmul_precision("high" if cfg["perf"]["tf32"] else "highest")
    log.event("phase", name="setup", device=str(R.device), autocast=cfg["autocast"])
    cap_vram(R)  # before anything is allocated on the device, and on every resume too

    if state is not None:
        R.st.update(copy.deepcopy(state["st"]))
        R.st.setdefault("optim_groups", "single")  # a state from before the param groups: its one group (setup_optim)
        R.st.setdefault("started_utc", None)  # a state from before it: unknown (tools/study_report falls back)
        for k, v in early_stop_state().items():  # the state's early_stop replaced the default whole: keys added since
            R.st["early_stop"].setdefault(k, v)
        if not R.branch_start:  # a branch is a new run dir: its first launch is no resume
            R.st["resumes"] += 1
    reset = resume_reset_start(R) if cfg["schedule"]["resume_reset"] else None  # before anything is saved
    if state is None or R.branch_start:  # the run's first start: summary.json's started_utc, kept across resumes
        R.st["started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # true/false: this config's (a resume's --set too); "auto": the memory probe's choice, saved in the full state
    gc = cfg["memory"]["grad_ckpt"]
    grad_ckpt = gc if isinstance(gc, bool) else bool(R.st["memory"].get("grad_ckpt"))
    if state is not None:
        R.st["memory"]["grad_ckpt"] = grad_ckpt  # the flops multiplier and the phase event read it there
    setup_model(R, grad_ckpt, bn_mode(cfg))
    setup_aux_ctc(R)  # loss.aux_ctc_weight > 0: its parameters join R.params before the optimizer is built
    setup_processing(R)
    setup_optim(R)  # L2-SP theta_0 = the student init (the full state's copy replaces it on resume)
    setup_data(R)
    R.planner = make_planner(R, float(R.st["memory"].get("micro_audio_s", cfg["batch"]["micro_audio_s"])))
    try:
        if is_ctc(cfg):  # the CTC students' teacher: Parakeet's CTC path (ctc_hyp), checked against its PREREG numbers
            from kitsune.ctc_eval import TEACHER_SYSTEM, ctc_teacher_baselines

            base = ctc_teacher_baselines(rpath(cfg["parakeet_root"]), cfg["eval_sets"],
                                         check=cfg["eval"]["check_baselines"])
            log.event("teacher_baselines", teacher=TEACHER_SYSTEM, sets=base)
        else:
            base = ev.teacher_baselines(rpath(cfg["teacher_root"]), cfg["eval_sets"],
                                        check=cfg["eval"]["check_baselines"])
            log.event("teacher_baselines", sets=base)
    except FileNotFoundError as e:
        log.event("teacher_baselines", skipped=str(e))
    R.reference = reference_model(cfg)  # a missing or malformed file stops the run here, not at its verdict
    if R.reference:
        log.event("reference", **R.reference)
    from kitsune import student as S

    if is_ctc(cfg):
        from kitsune.ctc_student import param_counts

        report = param_counts(R.model)
    else:
        report = S.param_report(R.model)
    log.event("model", family=family(cfg), params=report, trainable=sum(p.numel() for p in R.params),
              grad_ckpt=grad_ckpt, relpos_patch=cfg["perf"]["relpos_patch"], compile=cfg["perf"]["compile"],
              bn_mode=bn_mode(cfg), optim_groups=R.st["optim_groups"],
              **({"aux_ctc_params": sum(p.numel() for p in R.aux_ctc.parameters())} if R.aux_ctc is not None else {}))

    if state is not None:  # a resume, or a T/2 branch's start from its parent's state (the same restore)
        full = R.resumed_from
        R.model.load_state_dict(torch.load(full / "model.pt", map_location=R.device, weights_only=True))
        if R.aux_ctc is not None:  # loss.aux_ctc_weight is RESUME_FIXED: a state of such a run always has the head
            if not (full / "aux_ctc.pt").is_file():
                raise SystemExit(f"{full}: no aux_ctc.pt, although loss.aux_ctc_weight is {aux_ctc_weight(cfg)}")
            R.aux_ctc.load_state_dict(torch.load(full / "aux_ctc.pt", map_location=R.device, weights_only=True))
        # offload: the AdamW state stays in host memory, and the masters are rebuilt from the weights just loaded -
        # the aux head's too, so it loads first (after it, the masters would keep the fresh head and the first push()
        # would write that back over the loaded one)
        R.opt.load_state_dict(torch.load(full / "optimizer.pt", map_location="cpu" if offloaded(R) else R.device,
                                         weights_only=True))
        R.l2sp.load_state_dict(torch.load(full / "l2sp.pt", map_location="cpu", weights_only=True))
        # both loads restore the saved hyper-parameters (the optimizer's param_groups, the L2-SP lambda): this config's
        # go back on top, so a resume's --set optim.* / loss.l2sp_lambda takes effect (a no-op without one). lr is set
        # every step anyway; fused/foreach only change speed
        apply_optim_hparams(R)
        R.l2sp.lam = float(cfg["loss"]["l2sp_lambda"])
        try:
            R.planner.load_state_dict(state["planner"])
        except ValueError as e:  # another store than the state's: no retry can fix that (resume_check tells first)
            raise ResumeMismatch(f"ResumeMismatch: {full}: {e} (the train store under {R.train.cache_dir} is not the "
                                 "one this state was trained on: the same selection file, labels and audio are "
                                 "needed, and the state's micro_audio_s)") from e
        _set_rng_state(state["rng"])
        plan_epochs(R)  # a no-op unless the clock was switched to "epochs" on this resume
        if reset is not None:  # the reset cleared total_steps: plan_epochs planned schedule.epochs again
            resume_reset_finish(R, reset)
        if R.branch_start:
            br = R.st["branch"]
            ignored = branch_fracs(cfg)
            log.event("branch", **{k: br[k] for k in ("parent_run_id", "resume_step", "end_step", "t_c")},
                      parent_state=br["parent_state"], parent_max_steps=br["parent_max_steps"],
                      train_s=round(R.st["train_s"], 1), epoch=R.st["epoch_progress"], planner=state["planner"],
                      **({"fracs_ignored": ignored} if ignored else {}))
        else:
            log.event("resumed", at_step=R.st["step"], train_s=round(R.st["train_s"], 1),
                      epoch=R.st["epoch_progress"], planner=state["planner"])
            # a timed upload a crash cut short: the state resumed from goes to the scratch repo now (after every
            # refusal of the resume, so a refused one sends nothing)
            requeue_timed(R, full, state)
    else:
        if cfg["smoke"]["enabled"]:
            log.event("phase", name="smoke")
            decode_preflight(R)  # before the probe decodes the worst micro-batches: a clear error, not an empty batch
            R.st["memory"] = memory_probe(R)  # next: the checks below then run at the micro-batch size it chose
            smoke_checks(R)
        else:
            R.st["memory"] = dict(micro_audio_s=float(cfg["batch"]["micro_audio_s"]),
                                  grad_ckpt=cfg["memory"]["grad_ckpt"] is True)
            if R.st["memory"]["grad_ckpt"]:
                R.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                train_mode(R.model, bn_mode(cfg))
            R.st["flops_per_padded_s"] = measure_flops(R, R.ds[R.planner.worst_micro_batches()["longest"]])
        plan_epochs(R)  # after the memory probe: the micro-batch size it chose shapes the plans
        R.planner.epoch_plan(0)
        log.event("plan", micro_audio_s=R.planner.micro_audio_s, **R.planner.stats)
    if lr_probe_on(cfg):
        log.event("lr_probe", lr=float(cfg["optim"]["lr"]), max_steps=R.max_steps(),
                  warmup_steps=int(cfg["schedule"]["warmup_steps"]), cooldown_frac=float(cfg["schedule"]["cooldown_frac"]),
                  off=["step-0 eval", "in-loop evals", "mini evals", "weights", "checkpoint uploads"])
    if not R.st["step0_done"]:
        if lr_probe_on(cfg):  # metrics only: the step-0 eval decodes greedily
            log.event("phase", name="step0_eval", skipped="lr_probe")
        else:
            log.event("phase", name="step0_eval")
            run_eval(R, 0, mini_val=True)  # with combined_loss/val's step-0 point on the mini val subset (minis on)
            if getattr(R, "devstore", None) is not None and cfg["eval"]["dev"]["at_start"]:
                run_dev_eval(R, 0)  # the dev curve's step-0 point, never checked by the early stop
        R.st["step0_done"] = True

    if R.st["early_stop"]["stop"]:  # resumed after an early stop had triggered: straight to the end phase
        log.event("phase", name="train", at_step=R.st["step"], skipped="early_stop",
                  reason=R.st["early_stop"]["triggered"]["reason"])
    else:
        loop(R)

    step = R.st["step"]
    log.event("phase", name="end", at_step=step, train_s=round(R.clock(), 1))
    # timed states are not awaited: the queued ones are cancelled, a running one goes on meanwhile (the end save waits,
    # UPLOAD_WAIT_S at most, only for one of its own step's dir, whose trainer.pt it rewrites)
    abandon_timed(R)
    if lr_probe_on(cfg):
        return end_lr_probe(R, step)
    save_weights(R, step, "end")
    save_full(R, step, "end", upload="end" in cfg["ckpt"]["upload_full_at"])
    # a T/2 branch's one eval is its final one, always on the complete sets
    final_complete = True if R.st.get("branch") else None
    if R.last_complete and R.last_complete[0] == step and (cfg["eval"]["final_full_greedy"] or final_complete):
        # the loop's last eval decoded the complete eval sets at this very step, with these weights (no optimizer step
        # since; a skipped step leaves the step count too): it is the final eval, and its record is already the
        # history's last. A second decode (an epoch-end eval running past T, max_steps or the STOP file at an epoch
        # end, early stop) would give the same numbers and cost a whole complete eval out of the end reserve
        full_sum = R.last_complete[1]
        log.event("eval_final_reused", at_step=step)
    else:
        full_sum = run_eval(R, step, final=True, complete=final_complete)
    dh = R.st["dev_history"]
    if (getattr(R, "devstore", None) is not None and cfg["eval"]["dev"]["at_end"]
            and not (dh and dh[-1]["step"] == step)):
        run_dev_eval(R, step, final=True)  # the dev curve's last point (the loop skips one at its last step)
    verdict = gate_verdict(cfg, family_verdict(cfg, full_sum, R.st["history"], R.reference))
    log.eval_json("verdict", verdict, step)
    log.event("verdict", **verdict)
    if line := reference_line((verdict.get("numbers") or verdict).get("reference")):
        print(line, flush=True)
    headline = (R.st["history"][-1] if R.st["history"] else {}).get("headline")
    # the final eval, the verdict, summary.json (uploads "pending" until close() rewrites it) and the end events go up
    # now, while the ~9 GB end state drains (up to UPLOAD_WAIT_S), not only with close()'s sync after it: on a slow Hub
    # the watchdog stops the box meanwhile. A loop sync still running gets END_SYNC_JOIN_S to end first; a stalled one
    # is close()'s to bound
    log.write_summary(make_summary(R, "complete", verdict=verdict, final=full_sum, uploads="pending", headline=headline))
    from kitsune import heartbeat

    # the waits beat $KITSUNE_HEARTBEAT (none: a no-op) so the queue's stall check sees a live trainer, for at most
    # UPLOAD_WAIT_S + 300 s: a wait that hangs past its own bounds still goes stale
    with heartbeat.beating(max_s=UPLOAD_WAIT_S + 300):
        if log.wait_sync(END_SYNC_JOIN_S):
            log.sync(force=True, wait=False)
        uploads = R.uploader.wait(UPLOAD_WAIT_S)
        summary = make_summary(R, "complete", verdict=verdict, final=full_sum, uploads=uploads, headline=headline)
        R.uploader.shutdown()
        log.close(summary=summary)
    return EXIT_OK


def lr_probe_eval(R: Run, step: int) -> dict:
    """The LR probe's result (lr_probe.enabled): the family's training objective per target token, teacher-forced on
    the COMPLETE gate sets (LR_PROBE_SETS) in eval mode - AED: w_kl x KL + w_ce x CE (kitsune.kd.kd_objective's terms;
    no aux CTC) - pooled token-weighted over the three sets (`objective`) and per set (`per_set`). Tables and the
    summary under evals/step_<N>/, the scalars eval/tf/... and combined_loss/val_full (the same number). Family "ctc":
    w_kl x KL + w_ctc x CTC per CTC target token (kitsune.ctc_kd.ctc_kd_objective's terms; the frame metrics of
    kitsune.ctc_eval, no decode), the result naming them ctc / w_ctc."""
    from kitsune import evaluate as ev

    cfg, log = R.cfg, R.log
    fam = family(cfg)
    w_kl = float(cfg["loss"]["w_kl"])
    second, w2 = ("ctc", float(cfg["loss"]["w_ctc"])) if fam == "ctc" else ("ce", float(cfg["loss"]["w_ce"]))
    ids = [u.id for u in R.evalstore.utts if u.source in LR_PROBE_SETS]
    t0 = time.time()
    log.event("eval_start", at_step=step, final=True, complete=True, lr_probe=True)
    if fam == "ctc":
        from kitsune.ctc_eval import ctc_eval

        tf_sum, tf_df, _, _ = ctc_eval(R.model, R.evalstore, R.feat_eval, R.device, float(cfg["eval"]["batch_s"]),
                                       ids=ids, amp=R.amp, decode=False)
    else:
        tf_sum, tf_df = ev.teacher_forced_eval(R.model, R.evalstore, R.feat_eval, R.device,
                                               float(cfg["eval"]["batch_s"]), ids=ids, amp=R.amp)
    assert_bn_mode(R.model, bn_mode(cfg))
    sets = tf_sum.get("sets", {})
    missing = [s for s in LR_PROBE_SETS if not sets.get(s, {}).get("n_tok")]
    if missing:
        raise RuntimeError(f"lr probe: no scored target token in {missing} (the objective pools all of "
                           f"{', '.join(LR_PROBE_SETS)})")
    n = {s: int(sets[s]["n_tok"]) for s in LR_PROBE_SETS}
    kl = sum(float(sets[s]["kl"]) * n[s] for s in LR_PROBE_SETS) / sum(n.values())
    t2 = sum(float(sets[s][second]) * n[s] for s in LR_PROBE_SETS) / sum(n.values())
    result = dict(objective=w_kl * kl + w2 * t2,
                  per_set={s: w_kl * float(sets[s]["kl"]) + w2 * float(sets[s][second]) for s in LR_PROBE_SETS},
                  lr=float(cfg["optim"]["lr"]), max_steps=R.max_steps(), family=fam, step=int(step), kl=kl,
                  **{second: t2}, w_kl=w_kl, **{f"w_{second}": w2}, n_tok=n,
                  n_utts={s: int(sets[s]["n_utts"]) for s in LR_PROBE_SETS},
                  n_bad_audio=int(tf_sum.get("n_bad_audio", 0)), wall_s=round(time.time() - t0, 1))
    for src, g in tf_df.groupby("source", sort=True):
        log.table(f"tf_{src}", g.reset_index(drop=True), step)
    log.eval_json("lr_probe", dict(result, tf=tf_sum), step)
    scal = ev.flatten(tf_sum, "eval/tf")
    scal["combined_loss/val_full"] = result["objective"]
    log.scalars(scal, step)
    return result


def end_lr_probe(R: Run, step: int) -> int:
    """The LR probe's end phase: its full state (local, never uploaded; no weights), lr_probe_eval, the
    `lr_probe_result` event and summary.json's lr_probe (verdict "N/A": no greedy decode, no gate)."""
    cfg, log = R.cfg, R.log
    save_full(R, step, "end")
    result = lr_probe_eval(R, step)
    log.event("lr_probe_result", **{k: result[k] for k in ("objective", "per_set", "lr", "max_steps", "family")},
              at_step=step, n_tok=result["n_tok"])
    print(f"[lr probe] lr {result['lr']:g} | {result['max_steps']} steps | objective {result['objective']:.4f} | "
          + " | ".join(f"{s} {v:.4f}" for s, v in result["per_set"].items()), flush=True)
    verdict = dict(verdict="N/A", reason="lr probe: the teacher-forced objective on the complete gate sets only")
    log.write_summary(make_summary(R, "complete", verdict=verdict, final=None, uploads="pending", lr_probe=result))
    if log.wait_sync(END_SYNC_JOIN_S):
        log.sync(force=True, wait=False)
    uploads = R.uploader.wait(UPLOAD_WAIT_S)
    summary = make_summary(R, "complete", verdict=verdict, final=None, uploads=uploads, lr_probe=result)
    R.uploader.shutdown()
    log.close(summary=summary)
    return EXIT_OK


def lr_phase_name(phase: int | None) -> str | None:
    """st["lr_phase"] (wsd_lr's phase of the last optimizer step: 0, 1, 2; None before the first) as the name the
    `lr_phase` events and the history records use (kitsune.evaluate.LR_PHASES)."""
    from kitsune.evaluate import LR_PHASES

    return None if phase is None else LR_PHASES[int(phase)]


def verdict_options(cfg: dict) -> dict:
    """kitsune.evaluate.verdict's keyword options for this config: none under eval.verdict_version 1 - the very call
    the first A100 run was judged with, so its verdict reproduces - and version / min_epoch_gap under 2 (verdict()'s
    docstring defines what v2 computes)."""
    ev_cfg = cfg["eval"]
    if int(ev_cfg["verdict_version"]) == 1:
        return {}
    return dict(version=int(ev_cfg["verdict_version"]), min_epoch_gap=float(ev_cfg["verdict_min_epoch_gap"]))


def reference_model(cfg: dict) -> dict | None:
    """eval.reference's per-set corpus CER (kitsune.evaluate.load_reference; its path relative to the repo root, the
    code checkout, as the student path is), or None without one. The trainer reads it at setup and
    scripts/05_evaluate.py before its eval, so a missing or malformed file stops them before any paid work."""
    ref = cfg["eval"]["reference"]
    if not ref:
        return None
    from kitsune import evaluate as ev

    try:
        return ev.load_reference(rpath(ref["path"]), ref.get("name"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"eval.reference: {e}") from e


def verdict_results(final: dict, history: list[dict], reference: dict | None = None,
                    teacher: dict | None = None) -> dict:
    """kitsune.evaluate.verdict's results: the final eval's greedy summary, the history and, when the config has one,
    the reference model's CER (the verdict's "reference", reported only); teacher: the per-set teacher CER to judge
    against (family "ctc": family_verdict's), else the verdict's own (the same ids, or the pre-registered Cohere
    numbers)."""
    return dict(final=final, history=history, **({"reference": reference} if reference else {}),
                **({"teacher": teacher} if teacher else {}))


def family_verdict(cfg: dict, final: dict, history: list[dict], reference: dict | None = None) -> dict:
    """kitsune.evaluate.verdict of the run's family: AED as before; "ctc" is verdict(..., family="ctc"), against its own
    teacher, Parakeet's CTC path - its corpus CER on the same ids as the student's final numbers, never Cohere's
    pre-registered numbers; the registered Parakeet numbers (teacher_prereg / baseline_drift) are kitsune.evaluate's,
    from study/PREREG.json. A set without a same-ids teacher CER gets kitsune.ctc_eval.verdict_teacher's (the PREREG
    number, K6 while pending). Each set's record is named for its teacher (relabel_verdict: teacher_system)."""
    from kitsune import evaluate as ev

    if not is_ctc(cfg):
        return ev.verdict(verdict_results(final, history, reference), **verdict_options(cfg))
    from kitsune.ctc_eval import relabel_verdict, verdict_teacher

    out = ev.verdict(verdict_results(final, history, reference, teacher=verdict_teacher(final)), family="ctc",
                     **verdict_options(cfg))
    return relabel_verdict(out)


def reference_line(bar: dict | None) -> str | None:
    """The console line of the verdict's reference bar (kitsune.evaluate.reference_bar): '[reference] <name> |
    eval_jsut 12.65% vs 7.30% (1.73x) | ... | pooled 12.15% vs 7.44% (1.63x)' - student CER vs the reference's,
    student / reference; None without a reference."""
    if not bar:
        return None

    def cell(label, d):
        return f"{label} {100 * d['student']:.2f}% vs {100 * d['reference']:.2f}% ({d['ratio']:.2f}x)"

    parts = [cell(s, d) for s, d in bar["sets"].items()]
    if bar.get("pooled"):
        parts.append(cell("pooled", bar["pooled"]))
    return f"[reference] {bar['name']} (not a gate) | " + " | ".join(parts)


def gate_verdict(cfg: dict, computed: dict) -> dict:
    """The verdict summary.json reports: kitsune.evaluate.verdict's (GO / PROMISING / NO-GO / INCONCLUSIVE), or with
    eval.gate false (the sanity/overfit runs, whose tiny eval subsets say nothing about the gate) "N/A" with the
    computation's numbers - everything but its verdict and reasons."""
    if cfg["eval"]["gate"]:
        return computed
    return dict(verdict="N/A", reason="gate disabled (sanity/overfit run)",
                numbers={k: v for k, v in computed.items() if k not in ("verdict", "reasons")})


def final_eval_estimate(R: Run) -> float:
    """Seconds the final eval will take: the last periodic eval's parts, its greedy part scaled from the subset to
    the full eval sets (eval.final_full_greedy). 0 before any eval."""
    c = R.st.get("eval_cost")
    if not c:
        return 0.0
    greedy = c["greedy_s"]
    if R.cfg["eval"]["final_full_greedy"] and c["greedy_n"]:
        greedy *= max(1.0, len(R.evalstore) / c["greedy_n"])
    return c["tf_s"] + c["probe_s"] + greedy


def fit_budget(R: Run):
    """Clock "wall": clip T so the end phase ends before the instance's deadline (deadline_unix). train_hours counts
    loop time only, while vast/watchdog.sh counts 5.5 h from first boot: bootstrap, setup, the smoke phase, the step-0
    eval and, after a crash, the restart and the steps replayed since the last full state all come out of the same
    budget. Reserved at the end: the final-eval estimate + schedule.end_reserve_min (final saves, uploads, finish.py's
    verification and the watchdog's own sync lead). Runs at every loop start, so a resume re-fits T to the time
    actually left, and after each eval before the cooldown (the step-0 eval of an untrained student decodes slowly and
    overstates the final eval); the WSD cooldown moves with T, so the run still ends annealed (or goes straight to the
    end phase if no time is left)."""
    sch = R.cfg["schedule"]
    R.budget_s = None
    deadline = deadline_unix()
    if sch["clock"] != "wall" or deadline is None:
        return
    t0_budget = float(sch["train_hours"]) * 3600
    est = final_eval_estimate(R)
    reserve = float(sch["end_reserve_min"]) * 60 + est
    left = deadline - time.time()
    T = R.clock() + left - reserve
    if T < t0_budget:
        R.budget_s = T
    R.log.event("budget", deadline=deadline, left_s=round(left), reserve_s=round(reserve), final_eval_est_s=round(est),
                train_s=round(R.clock()), T=round(min(T, t0_budget)), T_train_hours=t0_budget, clipped=T < t0_budget)


def deadline_on(cfg: dict) -> bool:
    """4d applies: schedule.deadline_cooldown on the epochs or steps clock (validate refuses it on the wall clock)."""
    sch = cfg.get("schedule") or {}
    return bool(sch.get("deadline_cooldown")) and sch.get("clock") in ("epochs", "steps")


def deadline_rate(R: Run) -> float | None:
    """4d's loop-clock seconds per optimizer step, now: over the newest window of at least
    schedule.deadline_window_steps steps of THIS launch (R.deadline_marks, the (step, loop clock) sample of every 4d
    check after the smoke phase, fit_epochs_deadline's; the evals, minis, dev evals and saves in the window are in it),
    else the last rate st["deadline_rate"] kept (an earlier launch's), else None."""
    step, now = int(R.st["step"]), R.clock()
    marks = R.deadline_marks
    W = int(R.cfg["schedule"]["deadline_window_steps"])
    while len(marks) > 1 and step - marks[1][0] >= W:  # the newest sample W steps back is all a rate needs
        marks.pop(0)
    if marks and step - marks[0][0] >= W and step > marks[0][0]:
        return (now - marks[0][1]) / (step - marks[0][0])
    return R.st.get("deadline_rate")


def _evals_ahead(R: Run, t: float, t_end: float) -> int:
    """The in-loop complete evals at epoch ends (eval.full_every_epochs / every_epochs) strictly between steps t and
    t_end (the one at t_end would be the end phase's final eval), an epoch taken as the plan's steps per epoch
    (total_steps / epochs on the epoch clock, else the first planned epoch's)."""
    per = epoch_cadence(R.cfg)
    if not per or t_end <= t:
        return 0
    sch = R.cfg["schedule"]
    if sch["clock"] == "epochs" and R.st.get("total_steps"):
        spe = float(R.st["total_steps"]) / int(sch["epochs"])
    else:
        stats = getattr(R.planner, "epoch_stats", None) or {}
        spe = float(next(iter(stats.values()))["steps"]) if stats else 0.0
    if spe <= 0:
        return 0
    every = spe * per
    return max(0, math.ceil(t_end / every - 1e-9) - 1 - math.floor(t / every + 1e-9))


def fit_epochs_deadline(R: Run, at_start: bool = False) -> dict | None:
    """4d (schedule.deadline_cooldown; the epochs and steps clocks, where fit_budget does nothing): fit the run's end
    phase before $KITSUNE_DEADLINE (deadline_unix; the queue's per-item deadline) by scheduling, starting or compressing
    the WSD cooldown - its own record st["deadline_cooldown"] {t_c, T, ...} next to the early stop's cooldown, never
    through early_stop.triggered (either can act after the other; Run.progress takes the smaller T, Run.cooldown_start
    the earlier t_c). Active once the smoke phase is done and a rate is known (deadline_rate). The time left minus the
    reserve (schedule.end_reserve_min, the final eval's estimate final_eval_estimate, and the last dev eval's wall time
    under eval.dev.at_end) gives fit = the most steps n with n x rate + (the in-loop epoch-end evals among them) x the
    eval estimate <= it; T_new = t + fit, against T_plan = the schedule's T without 4d's record:
      fits (t + fit >= T_plan)   a record whose cooldown has not begun is cleared (action "clear": an early
                                 pessimistic rate never cuts the run for good), else nothing
      in a cooldown already      "compress": T_new with the cooldown's t_c, only when T_new is below the current T
      else                       t_c = max(t, T_new - ceil(cooldown_frac x T_new)): "schedule" (t_c later: the run
                                 uses its time and still anneals over its full share) or "start" (now)
    T_new <= t (no step fits): the record's T = t ends the loop at once and a `no_time_left` event says so. Every action
    is a `deadline_cooldown` event {t_c, T, action, clock, at_step, deadline, left_s, sec_per_step, reserve_s,
    evals_ahead, T_before}, kept as st["deadline_cooldown"] (summary.json's); the scalar sched/deadline_T (t + fit)
    follows every check with a rate; the launch's first active check (the loop start, at_start, or on a fresh run the
    first after the smoke phase) also logs `deadline_check` {deadline, left_s, sec_per_step, T, projected_end_utc,
    fits}. The loop saves the pre_cooldown state at the record's t_c, as for any cooldown. Returns
    the record of an action, else None."""
    cfg, st = R.cfg, R.st
    sch = cfg["schedule"]
    if not deadline_on(cfg):
        return None
    deadline = deadline_unix()
    if deadline is None or (cfg["smoke"]["enabled"] and int(cfg["smoke"]["steps"]) > 0 and not st["smoke_done"]):
        return None
    # the launch's first active check logs deadline_check (a fresh run's loop start falls in the smoke phase: its first
    # check after the smoke phase does); every check leaves a (step, loop clock) sample for deadline_rate
    first = not R.deadline_marks
    if R.deadline_marks and R.deadline_marks[-1][0] == st["step"]:
        R.deadline_marks.pop()  # the same step checked again: its newest sample
    R.deadline_marks.append((int(st["step"]), R.clock()))
    rate = deadline_rate(R)
    t, T = R.progress()
    per_eval = final_eval_estimate(R)
    dh = st.get("dev_history") or []
    dev_end = float(dh[-1].get("wall_s") or 0.0) if dh and dev_on(cfg) and cfg["eval"]["dev"]["at_end"] else 0.0
    reserve = float(sch["end_reserve_min"]) * 60 + per_eval + dev_end
    now = time.time()
    left = deadline - now
    if at_start or first:
        proj = None
        if rate is not None:
            proj = now + (T - t) * rate + _evals_ahead(R, t, T) * per_eval + reserve
        R.log.event("deadline_check", deadline=deadline, left_s=round(left, 1), reserve_s=round(reserve, 1),
                    sec_per_step=rate, T=T, fits=None if proj is None else proj <= deadline,
                    projected_end_utc=(datetime.fromtimestamp(proj, timezone.utc).isoformat(timespec="seconds")
                                       if proj is not None else None))
    if rate is None or rate <= 0:
        return None
    st["deadline_rate"] = rate
    budget = left - reserve

    def cost(n: int) -> float:
        return n * rate + _evals_ahead(R, t, t + n) * per_eval

    # the largest n with cost(n) <= budget (cost grows with n); -1: not even no step
    lo, hi = (0, max(0, int(budget // rate))) if budget >= 0 else (-1, -1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if cost(mid) <= budget else (lo, mid - 1)
    fit = max(lo, 0)
    R.log.scalars({"sched/deadline_T": float(t + fit)}, st["step"])
    T_plan = R.progress(deadline=False)[1]
    dl = st.get("deadline_cooldown")
    base = dict(clock=sch["clock"], at_step=int(st["step"]), deadline=deadline, left_s=round(left, 1),
                sec_per_step=rate, reserve_s=round(reserve, 1), T_before=T)
    if t + fit >= T_plan:
        if dl and dl["t_c"] > t:  # scheduled, not begun: the run fits again
            st["deadline_cooldown"] = None
            rec = dict(base, t_c=None, T=T_plan, action="clear", evals_ahead=_evals_ahead(R, t, T_plan))
            R.log.event("deadline_cooldown", **rec)
            return rec
        return None
    T_new = float(t + fit)
    if st["pre_cooldown_done"] or t >= R.cooldown_start(T):  # in a cooldown (scheduled, early or 4d's) already
        if T_new >= T:
            return None
        rec = dict(base, t_c=R.cooldown_start(T), T=T_new, action="compress")
    else:
        t_c = max(float(t), T_new - math.ceil(float(sch["cooldown_frac"]) * T_new))
        if dl and (dl["t_c"], dl["T"]) == (t_c, T_new):  # the record it has already: no new event
            return None
        rec = dict(base, t_c=t_c, T=T_new, action="schedule" if t_c > t else "start")
    rec["evals_ahead"] = _evals_ahead(R, t, T_new)
    st["deadline_cooldown"] = dict(rec)
    R.log.event("deadline_cooldown", **rec)
    if T_new <= t:
        R.log.event("no_time_left", at_step=int(st["step"]), t=t, T=T_new, deadline=deadline, left_s=round(left, 1),
                    reserve_s=round(reserve, 1), sec_per_step=rate)
    return rec


def loop(R: Run):
    cfg, log = R.cfg, R.log
    sch, ck, ev_cfg = cfg["schedule"], cfg["ckpt"], cfg["eval"]
    nw = cfg["perf"]["num_workers"]
    nw = trainset.default_num_workers() if nw == "auto" else int(nw)
    nw, prefetch, cap = shm_cap(nw, int(cfg["perf"]["prefetch"]), float(R.planner.micro_audio_s))
    if cap:
        log.event("shm_cap", **cap)
        if "decode_ceiling_audio_s_per_s" in cap:  # a later ThroughputTooLow or a slow run traces back to here
            print(f"WARNING: /dev/shm ({cap['shm_free_gb']} GiB free) cut the loader to {nw} worker(s) at prefetch "
                  f"{prefetch}: decoding tops out near {cap['decode_ceiling_audio_s_per_s']} audio-s/s, below what the "
                  f"GPU takes; give the container a larger /dev/shm", flush=True)
    crash_at = int(os.environ["KITSUNE_CRASH_AT_STEP"]) if os.environ.get("KITSUNE_CRASH_AT_STEP") else None
    smoke_n = int(cfg["smoke"]["steps"]) if cfg["smoke"]["enabled"] else 0
    epochs = int(sch["epochs"]) if sch["clock"] == "epochs" else None
    max_steps = R.max_steps()  # schedule.max_steps, or a T/2 branch's end step
    probe = lr_probe_on(cfg)  # metrics only: no evals, minis or weights in the loop, no checkpoint uploads
    # the fraction evals and checkpoints (steps clock): step -> fraction of max_steps. An LR probe keeps only the local
    # full states (for a resume). A T/2 branch has none (branch_fracs lists what it ignores): its one eval is the
    # final one, and its fractions of its end step round(end_frac x M) fall before its resume step round(resume_frac
    # x M) - but for the rounding, which puts round(0.8 x round(0.5 M)) at round(0.4 M) + 1, its first step, for
    # about 1 M in 10
    no_fracs = bool(R.st.get("branch"))
    frac_eval = {} if probe or no_fracs else frac_steps(ev_cfg["full_at_fracs"], max_steps)
    frac_weights = {} if probe or no_fracs else frac_steps(ck["weights_at_fracs"], max_steps)
    frac_full = {} if no_fracs else frac_steps(ck["full_at_fracs"], max_steps)
    upload_at = set() if probe else set(ck["upload_full_at"])

    def passes_done() -> bool:  # clock "epochs": the planner's position is past the last epoch (skipped steps too)
        return epochs is not None and R.planner.epoch >= epochs

    def ending(ahead: float = 0.0) -> bool:  # the loop stops before another step (the final eval then runs at this
        # step); ahead: seconds of work still to come at this step, on the wall clock (the other clocks count steps)
        t_now, T_now = R.progress()
        if sch["clock"] == "wall":
            t_now += ahead
        return (t_now >= T_now or bool(max_steps and R.st["step"] >= max_steps) or passes_done()
                or (R.run_dir / STOP_FILE).exists())

    per_epochs = epoch_cadence(cfg)  # evals at epoch ends (every_epochs / full_every_epochs), else every_min / _steps
    prof = smoke_profiler(R, smoke_n)  # perf.profile_smoke: ~20 steps of the smoke phase under torch.profiler

    fit_budget(R)
    deadline_every = int(sch["deadline_check_steps"]) if deadline_on(cfg) else None  # 4d: here, then every N steps
    fit_epochs_deadline(R, at_start=True)
    log.event("phase", name="train", at_step=R.st["step"], workers=nw, micro_audio_s=R.planner.micro_audio_s,
              grad_ckpt=R.st["memory"].get("grad_ckpt"))
    train_ds = R.ds if getattr(R, "train_ds", None) is None else R.train_ds  # augment.enabled: the augmenting copy
    loader = trainset.make_loader(train_ds, R.planner, nw, prefetch, timeout_s=float(cfg["perf"]["loader_timeout_s"]))
    R.loop_t0 = time.monotonic()
    epoch_seen = R.st["epoch"] if R.st["step"] else -1
    try:
        while True:
            t, T = R.progress()
            if t >= T or (max_steps and R.st["step"] >= max_steps) or passes_done():
                break
            if stop_requested(R):  # runs/<run_id>/STOP
                break
            t_c = R.cooldown_start(T)
            if not R.st["pre_cooldown_done"] and t >= t_c and float(sch["cooldown_frac"]) > 0:
                R.st["pre_cooldown_done"] = True
                br = R.st.get("branch")
                if br and R.st["step"] == br["resume_step"]:
                    # a T/2 branch starts its cooldown at its parent's state, which is the parent's local
                    # full_step_<resume_step>: not saved (nor uploaded) a second time
                    log.event("phase", name="cooldown", at_step=R.st["step"], t=t, T=T, state=br["parent_state"])
                else:
                    R.st["pre_cooldown_full"] = f"full_step_{R.st['step']}"  # in its own trainer.pt too: build()
                    log.event("phase", name="cooldown", at_step=R.st["step"], t=t, T=T)
                    save_full(R, R.st["step"], "pre_cooldown", upload="pre_cooldown" in upload_at)
            step = R.st["step"] + 1
            if crash_at is not None and step >= crash_at:
                raise RuntimeError(f"KITSUNE_CRASH_AT_STEP={crash_at}: simulated crash before step {step}")
            lr, phase = wsd_lr(float(cfg["optim"]["lr"]), step, t, T, warmup_steps(sch, R.st["total_steps"]),
                               float(sch["cooldown_frac"]), t_c=t_c)
            if phase != R.st.get("lr_phase"):
                log.event("lr_phase", at_step=step, phase=("warmup", "stable", "cooldown")[phase], lr=lr, t=t, T=T)
                R.st["lr_phase"] = phase
            profiled = prof is not None and prof.begin(step)  # this step runs under the smoke profiler
            t0 = time.perf_counter()
            if R.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            (e, s), mbs = next(loader)
            wait = time.perf_counter() - t0
            if e != epoch_seen:
                epoch_seen = e
                log.event("epoch", step_in_epoch=s, **(R.planner.epoch_stats.get(e) or dict(epoch=e)))
            if not R.st["smoke_done"] and smoke_n:  # rows the workers could not decode (smoke_end); before train_step,
                # so a step that lost every row counts too. Utterances, not rows: a joined row holds several (augment)
                R.st["smoke_dropped"] += sum(len(mb["dropped"]) for mb in mbs)
                R.st["smoke_utts"] += sum(batch_utts(mb) for mb in mbs)
            out = train_step(R, step, lr, mbs, e)
            R.planner.step_done(e, s)
            if out is None:
                if profiled:  # the profiler saw the skipped step too: it counts as one of its steps
                    prof.end(step, wait, None, len(mbs))
                continue
            if R.device.type == "cuda":
                torch.cuda.synchronize()
            step_s = time.perf_counter() - t0
            R.st["step"] = step
            objective = log_step(R, step, lr, phase, out, wait, step_s, e, s)
            if profiled:  # after the step's logging, part of its per-step cost; the summary is written at the last
                prof.end(step, wait, step_s, out["n_micro"])
            es = R.st["early_stop"]
            es["loss_sum"], es["loss_n"] = es["loss_sum"] + float(objective), es["loss_n"] + 1  # metric "train_loss"
            if not R.st["smoke_done"] and smoke_n:
                R.st["smoke_losses"].append(float(objective))
                # the first steps pay for worker start-up and kernel autotuning, the profiled ones for the profiler
                if step > max(1, smoke_n // 10) and not profiled:
                    R.st["smoke_audio_s"] += out["audio_real"]
                    R.st["smoke_time_s"] += step_s
                if step >= smoke_n:
                    smoke_end(R)
            t = R.clock()
            if probe:
                cadence = False
            elif per_epochs:  # epochs completed since the last eval (the planner's position, on every clock); on the
                # epoch clock the end phase's eval covers the last one, on the others a complete eval at the loop's
                # last step is reused as the final eval (train)
                cadence = R.planner.epoch - R.st["last_eval_epoch"] >= per_epochs and not passes_done()
            else:
                cadence = due(t, step, R.st["last_eval_t"], R.st["last_eval_step"], ev_cfg["every_min"],
                              ev_cfg["every_steps"])
            at_frac = step in frac_eval  # eval.full_at_fracs: a complete eval next to the cadence, never moving it
            eval_now = cadence or at_frac
            if eval_now and (R.st["early_stop"]["cooldown"] or R.st.get("deadline_cooldown")):
                # an early or 4d cooldown's last step: the final eval covers it
                t_now, T_now = R.progress()
                eval_now = t_now < T_now
            if eval_now:
                complete = bool(ev_cfg["full_every_epochs"]) or at_frac
                gsum = run_eval(R, step, complete=complete)
                if complete:
                    R.last_complete = (step, gsum)
                if cadence:
                    R.st["last_eval_t"], R.st["last_eval_step"] = R.clock(), step
                    R.st["last_eval_epoch"] = R.planner.epoch
                if not R.st["pre_cooldown_done"]:  # a fresher final-eval estimate (never moves T in the cooldown)
                    fit_budget(R)
                # action "stop": the end phase right after this eval. A dev metric is checked after the dev evals
                # instead (below): early_stop_check also restarts train_loss's window at every call
                if not dev_metric(cfg) and early_stop_check(R, step):
                    break
            elif (not probe and mini_due(R, step)
                  and not ending(R.st["mini_history"][-1]["wall_s"] if R.st["mini_history"] else 0.0)):
                # never at a step with a full eval in the loop, nor when the loop ends at this step (the final eval
                # follows): on the wall clock also when the mini itself, as long as the last one took, would carry the
                # clock past T. A first mini, the checkpoints saved after it or a STOP file can still end the loop here
                run_mini_eval(R, step)
            # the dev eval (eval.dev) on its own cadence, next to the full and mini evals, and 4d's check: both before
            # the saves, so a state saved at this step holds the dev record and any trigger or deadline record. Never
            # at the loop's last step: the end phase's at_end dev eval covers it
            if not probe and dev_due(R, step) and not ending():
                run_dev_eval(R, step)
                if dev_metric(cfg) and early_stop_check(R, step):  # action "stop": the end phase follows
                    break
            if deadline_every and step % deadline_every == 0:
                fit_epochs_deadline(R)
            if not probe and due(t, step, R.st["last_weights_t"], R.st["last_weights_step"], ck["weights_every_min"],
                                 ck["weights_every_steps"]):
                save_weights(R, step, "periodic")
                R.st["last_weights_t"], R.st["last_weights_step"] = R.clock(), step
            if step in frac_weights:
                save_weights(R, step, f"frac:{frac_weights[step]:g}")
            if step in frac_full:  # kept locally past rotation; uploaded when upload_full_at names its fraction.
                # Before the periodic save of the same step, so the state written records itself as kept
                f = frac_full[step]
                save_full(R, step, f"frac:{f:g}", keep=True, upload=any(upload_frac(u) == f for u in upload_at))
            if due(t, step, R.st["last_full_t"], R.st["last_full_step"], ck["full_local_every_min"],
                   ck["full_every_steps"]):
                R.st["last_full_t"], R.st["last_full_step"] = R.clock(), step
                save_full(R, step, "periodic")
            if not probe:  # 4a: a timed full state to the scratch repo when due, the log sync after one went up
                timed_state(R, t, step)
            log.sync()
    finally:
        if prof is not None:  # the loop ended (or failed) inside the profiled window: what it recorded, never raises
            prof.close()
        R.st["train_s"] = R.clock()
        R.loop_t0 = None
        loader.close()


def selection_sha256(R: Run) -> str | None:
    """The sha256 of the selection file the run trains on (the bytes, as make_selection's sidecar and PREREG.json hash
    it): the train store's record of it (its build fingerprinted that very file), else the file itself (a run that
    failed before its stores); None if neither is there."""
    info = getattr(getattr(R, "train", None), "info", None) or {}
    if info.get("selection_sha256"):
        return info["selection_sha256"]
    try:
        p = rpath(R.cfg["selection"])
        return trainset._sha256(p) if p.is_file() else None
    except (KeyError, TypeError, OSError):
        return None


def end_reason(R: Run) -> str:
    """What set the run's last step (summary.json's end_reason): a "stop" trigger's reason - "stop_file", else
    "early_stop" (patience / floor) -, else the record with the smallest T among the schedule's own ("schedule"), the
    early stop's cooldown ("early_stop", or "cooldown_file" for the COOLDOWN file's) and 4d's ("deadline"), the first
    on a tie; on the wall clock a budget clipped to the deadline (fit_budget) is "deadline" too."""
    st, sch = R.st, R.cfg["schedule"]
    es = st["early_stop"]
    trig = es.get("triggered") or {}
    if es.get("stop"):
        return "stop_file" if trig.get("reason") == "stop_file" else "early_stop"
    try:
        _, T = R.base_progress()
    except TypeError:  # a run that failed before its plan (epoch clock: no total_steps yet); the summary must go out
        T = float("inf")
    budget = getattr(R, "budget_s", None)
    clipped = sch["clock"] == "wall" and budget is not None and budget < float(sch["train_hours"]) * 3600
    cands = [("deadline" if clipped else "schedule", T)]
    if es.get("cooldown"):
        cands.append(("cooldown_file" if trig.get("reason") == "cooldown_file" else "early_stop", es["cooldown"]["T"]))
    if st.get("deadline_cooldown"):
        cands.append(("deadline", st["deadline_cooldown"]["T"]))
    return min(cands, key=lambda c: c[1])[0]


def summary_full(R: Run) -> dict:
    """summary.json's full-data fields: end_reason and resume_resets always; dev_history (the dev evals' records),
    deadline_cooldown (4d's record) and schedule_resets (the resume resets') when there are any. stopped_early keeps
    its meaning (a trigger that shortened the run), now also for a dev metric's trigger and the COOLDOWN file's."""
    st = R.st
    out = dict(end_reason=end_reason(R), resume_resets=int(st.get("resume_resets") or 0))
    for key in ("dev_history", "deadline_cooldown", "schedule_resets"):
        if st.get(key):
            out[key] = st[key]
    return out


def make_summary(R: Run, status: str, **extra) -> dict:
    st, hist = R.st, R.st["history"]
    elapsed = R.log.elapsed() if R.log else time.time() - R.t_start
    dph = float(os.environ["KITSUNE_DPH"]) if os.environ.get("KITSUNE_DPH") else None

    def best(fn):
        vals = [(fn(r), r["step"]) for r in hist]
        vals = [v for v in vals if v[0] is not None and not math.isnan(v[0])]
        return dict(value=min(vals)[0], step=min(vals)[1]) if vals else None

    from kitsune.evaluate import GATE_SETS

    def cer_ratio(r, gate=True):
        """Mean student / teacher corpus-CER ratio of a history record's greedy sets. gate: the gate sets only, as the
        verdict's trend and heldout_kl read them (every set if none is one), so a monitor-only hold-out (eval_emilia,
        galgame; teacher CER several times the gate sets') cannot move the best step; else every evaluated set."""
        g = r.get("greedy") or {}
        if gate:
            g = {s: d for s, d in g.items() if s in GATE_SETS} or g
        v = [d["cer_ref_corpus"] / tc for d in g.values()
             if (tc := d.get("teacher_cer_ref_corpus")) and not math.isnan(tc)]  # NaN: no teacher rows in that set
        return float(np.mean(v)) if v else None

    trig = st["early_stop"]["triggered"]
    br = st.get("branch")
    if br:  # a T/2 branch: its parent and schedule (the `branch` event's fields)
        extra.setdefault("branch", {k: br[k] for k in ("parent_run_id", "resume_step", "end_step", "t_c")})
    return dict(
        status=status, run_id=R.run_dir.name, family=family(R.cfg),
        # the study's checks (kitsune.study_stats): one selection for every run, PREREG_numbers before the first start
        selection_sha256=selection_sha256(R), started_utc=st.get("started_utc"),
        steps=st["step"], epochs=st["epoch_progress"],
        train_s=round(st["train_s"], 1), elapsed_s_total=round(elapsed, 1), resumes=st["resumes"],
        budget_s=R.budget_s,  # None = the full train_hours; else T clipped to the instance deadline
        throughput=dict(audio_s=st["audio_s"], tokens=st["tokens"], step_time_s=round(st["step_time_s"], 1),
                        audio_s_per_s=st["audio_s"] / st["step_time_s"] if st["step_time_s"] else None,
                        tokens_per_s=st["tokens"] / st["step_time_s"] if st["step_time_s"] else None,
                        # the share of the steps' time spent waiting for the loader (the full smoke's check 5)
                        data_wait_frac=st.get("data_wait_s", 0.0) / st["step_time_s"] if st["step_time_s"] else None),
        memory=st["memory"], skipped=dict(nonfinite=st["nonfinite_total"], oom=st["oom_skips"]),
        cost=dict(dph=dph, usd=round(dph * elapsed / 3600, 2) if dph else None, note="trainer process time only"),
        best=dict(greedy_cer_ratio_mean=best(cer_ratio),  # the gate sets, as heldout_kl (eval_record)
                  greedy_cer_ratio_mean_all_sets=best(lambda r: cer_ratio(r, gate=False)),
                  heldout_kl=best(lambda r: r.get("heldout_kl"))),
        # the `early_stop` event's fields of a trigger that shortened the run ("stop", or a cooldown begun early); a
        # trigger inside the scheduled cooldown (cooldown.already) changed nothing: only early_stop_trigger has it
        stopped_early=trig if trig and not (trig.get("cooldown") or {}).get("already") else None,
        early_stop_trigger=trig,  # every trigger; None: none
        history=hist, mini_history=st["mini_history"], checkpoints=dict(weights=st["weights"], full=st["fulls"]),
        **summary_full(R), **summary_timed(R), config=R.cfg, **extra)


# set by main() when it returns or fails with an upload still running: run_script (the __main__ entry) then skips
# interpreter finalization
_HARD_EXIT = False


def uploads_left_running(R: Run) -> bool:
    """A checkpoint upload or log sync still running after the bounded end-phase waits (or an xet commit thread one
    left behind). hf_xet 1.5.1's wait_to_finish re-takes the GIL every 100 ms for its check_signals poll (PyO3 0.26
    detach, then PyEval_RestoreThread); CPython 3.12 calls pthread_exit on a thread that does that while the
    interpreter finalizes, and the unwind through its Rust frames aborts the process (rc -6), which vast/supervise.py
    read as a crash of a finished run. A timed state's upload to the scratch repo (R.scratch) counts too."""
    sc = getattr(R, "scratch", None)
    return bool(R.uploader.busy()) or (sc is not None and bool(sc.busy())) or not R.log.wait_sync(0) or any(
        t.name == "hf-upload-committer" and t.is_alive() for t in threading.enumerate())


def main(argv=None) -> int:
    global _HARD_EXIT
    args = parse_args(argv)
    R, state = build(args)
    try:
        rc = train(R, state)
    except ThroughputTooLow as e:
        R.log.event("throughput_too_low", error=str(e))
        _close_failed(R, "throughput_too_low", e)
        rc = EXIT_THROUGHPUT
    except BaseException as e:
        R.log.exception(e)
        _close_failed(R, "failed", e)
        with suppress(Exception):  # never masks the original exception
            _HARD_EXIT = uploads_left_running(R)  # an upload still running after FAILED_UPLOAD_WAIT_S
        raise
    _HARD_EXIT = uploads_left_running(R)
    return rc


def _close_failed(R: Run, status: str, exc: BaseException):
    """Partial summary + forced sync (RunLogger.close) first, without waiting for the checkpoint uploads (an 8.6 GB
    pre_cooldown full state held them, and the resume, for many minutes); then at most FAILED_UPLOAD_WAIT_S for those,
    and an event naming the ones cut off (an event after close still reaches events.jsonl, which the post-crash sync
    uploads). The queued timed uploads are cancelled without a wait (abandon_timed). All of it beats $KITSUNE_HEARTBEAT
    (bounded), so the queue's stall check does not kill the trainer while it closes. Never masks the original
    exception."""
    from kitsune import heartbeat

    with heartbeat.beating(max_s=FAILED_UPLOAD_WAIT_S + 300):
        try:
            R.log.close(summary=make_summary(R, status, error=f"{type(exc).__name__}: {exc}"[:2000]))
            abandon_timed(R, "failed")
            if left := R.uploader.abandon(FAILED_UPLOAD_WAIT_S):
                R.log.event("ckpt_upload_abandoned", names=left, waited_s=FAILED_UPLOAD_WAIT_S)
        except Exception as e:  # noqa: BLE001
            print(f"closing the logger after a failure failed too: {e!r}", file=sys.stderr)


def exit_process(rc: int):
    """sys.exit(rc); os._exit(rc) when main() returned or failed with an upload still running (uploads_left_running):
    the logs and TensorBoard are closed by then, and finish.py uploads and verifies what the upload did not finish.
    Here, not in main(): the tests call main() directly."""
    if _HARD_EXIT:
        print(f"an upload is still running: exiting {rc} without interpreter finalization", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    sys.exit(rc)


def run_script(argv=None):
    """The __main__ entry: exit_process(main()). main() re-raises a failure, which skipped exit_process: one it
    re-raised with an upload still running (_HARD_EXIT) is reported as the interpreter would (the traceback, or a
    SystemExit's message) and leaves by exit_process too, with EXIT_FAIL (a SystemExit's int code); finalizing with
    the upload's thread in hf_xet aborted the process (rc -6). Any other failure propagates as before."""
    try:
        rc = main(argv)
    except BaseException as e:
        if not _HARD_EXIT:
            raise
        rc = EXIT_FAIL
        if not isinstance(e, SystemExit):
            sys.excepthook(type(e), e, e.__traceback__)
        elif e.code is None or isinstance(e.code, int):
            rc = e.code or 0
        else:
            print(e.code, file=sys.stderr)
    exit_process(rc)


# ------------------------------------------------------------------------------------------------ the study box
# One contiguous block (CONTRACT.md section 7, WP6A): the validation of the box path's keys (pull_parakeet,
# selection_recipe.study, calibrate, and optim.lr as a number, which the generated study configs leave null until the
# box fills it from its PREREG_numbers file), the calibration run's start barrier and its end phase. It hooks into the
# trainer by wrapping validate(), stop_requested() (checked before every step: the barrier waits in the first check)
# and end_lr_probe() at its end, not by editing them, so it merges cleanly with the CTC trainer's changes to the rest of
# this file; a calibration run is an lr_probe run in every other respect (metrics only: no step-0, in-loop or mini
# evals, no weights, no checkpoint uploads).


def calibrate_on(cfg: dict) -> bool:
    return bool((cfg.get("calibrate") or {}).get("enabled"))


def validate_box(cfg: dict):
    """The box path's keys (DEFAULTS: pull_parakeet, selection_recipe.study / full_study, calibrate) and optim.lr.
    full_study: kitsune.fullrun.full_recipe_problems' block, never with study. calibrate needs
    the steps clock and lr_probe.enabled (its loop is the LR probe's: metrics only), no branch, and a window (a, b] of
    whole steps with 0 <= a < b that the run reaches past a (max_steps > a; the study box ends its calibration runs
    with the STOP file once every run of the group has its window, kitsune.study_queue); barrier (a bool) only on a
    calibration run."""
    from kitsune import prereg

    lr = cfg["optim"]["lr"]
    if not (_number(lr) and lr >= 0):  # 0: weights that never move (tests/test_evaluate_script.py)
        raise SystemExit(f"optim.lr must be a number >= 0, got {lr!r} (a generated study config leaves it null until "
                         "the box fills it from its PREREG_numbers file: kitsune/study_queue.py)")
    if cfg["pull_parakeet"] and not cfg.get("parakeet_root"):
        raise SystemExit("pull_parakeet needs parakeet_root (the label root the box pulls the Parakeet targets from)")
    study = cfg["selection_recipe"].get("study")
    if study is not None and not (isinstance(study, dict) and set(study) == set(prereg.STUDY_SELECTION)
                                  and all(_number(v) for v in study.values())):
        raise SystemExit(f"selection_recipe.study must be null or the study's selection block with the keys "
                         f"{sorted(prereg.STUDY_SELECTION)} (numbers; study/data.json), got {study!r}")
    full_study = cfg["selection_recipe"].get("full_study")
    if full_study is not None:  # a full-data selection's recipe (kitsune.fullrun: FULL_STUDY, make_selection full mode)
        from kitsune import fullrun

        if study is not None:
            raise SystemExit("selection_recipe.study and selection_recipe.full_study are two selections' recipes: set "
                             "one (a full-data run's selection is make_selection.py's full mode)")
        if problems := fullrun.full_recipe_problems(full_study):
            raise SystemExit(f"selection_recipe.full_study is not the registered full recipe: {'; '.join(problems)}")
    cal = cfg["calibrate"]
    w = cal["window"]
    if not (isinstance(w, list) and len(w) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in w)
            and 0 <= w[0] < w[1]):
        raise SystemExit(f"calibrate.window must be [a, b], whole steps with 0 <= a < b, got {w!r}")
    if not isinstance(cal["barrier"], bool) or (cal["barrier"] and not cal["enabled"]):
        raise SystemExit(f"calibrate.barrier must be true or false, and true only with calibrate.enabled, got "
                         f"{cal['barrier']!r}")
    if cal["enabled"]:
        sch = cfg["schedule"]
        if sch["clock"] != "steps" or not cfg["lr_probe"]["enabled"] or cfg["branch"]["parent"] is not None:
            raise SystemExit("calibrate.enabled needs schedule.clock 'steps' and lr_probe.enabled (a calibration run "
                             "is an LR-probe-shaped run: metrics only), and no branch.parent")
        if int(sch["max_steps"]) <= w[0]:
            raise SystemExit(f"calibrate.enabled: max_steps {sch['max_steps']} ends before the window "
                             f"({w[0]}, {w[1]}] starts")


def end_calibrate(R: Run, step: int) -> int:
    """A calibration run's end phase (calibrate.enabled; the loop ended at max_steps or on the STOP file): its step-time
    table from its own metrics (kitsune.study_queue.calibration_stats over calibrate.window: the median step time,
    logging included, and the data-wait share), the micro-batch size the memory probe chose and the loader's
    workers; a `calibrate_result` event and summary.json's calibrate. No full state (a calibration is never resumed,
    the box runs it again), no eval."""
    from kitsune import study_queue as Q

    cfg, log = R.cfg, R.log
    a, b = (int(x) for x in cfg["calibrate"]["window"])
    rows = Q.read_step_rows(R.run_dir)
    result = dict(Q.calibration_stats(rows, a, b), micro_audio_s=float(R.planner.micro_audio_s), steps=int(step),
                  workers=next((e.get("workers") for e in reversed(Q.read_events(R.run_dir))
                                if e.get("kind") == "phase" and e.get("name") == "train"), None),
                  memory=R.st["memory"])
    log.event("calibrate_result", **result)
    print(f"[calibrate] {step} steps | window {result['window']} | t_step {result['t_step_s']} s | data wait "
          f"{result['data_wait_frac']} | micro {result['micro_audio_s']:g} s", flush=True)
    verdict = dict(verdict="N/A", reason="calibration: the step-time table only")
    log.write_summary(make_summary(R, "complete", verdict=verdict, final=None, uploads="pending", calibrate=result))
    if log.wait_sync(END_SYNC_JOIN_S):
        log.sync(force=True, wait=False)
    uploads = R.uploader.wait(UPLOAD_WAIT_S)
    summary = make_summary(R, "complete", verdict=verdict, final=None, uploads=uploads, calibrate=result)
    R.uploader.shutdown()
    log.close(summary=summary)
    return EXIT_OK


CALIBRATE_POLL_S = 0.05  # the barrier's check for GO: the group's trainers start their first steps within this


def calibrate_barrier(R: Run):
    """calibrate.barrier: the run is set up (model, memory probe, loader) and about to take its first step. It writes
    runs/<run_id>/READY and waits for GO (or STOP) there: the study box writes GO into every run of a calibration group
    in one pass once all of them are READY, so the group starts its steps together and every run's window falls while
    the whole group trains (kitsune.study_queue.Queue.calibrate_group). The wait is outside every step's time (the
    step clock starts after this check) and before the window."""
    from kitsune import study_queue as Q

    t0 = time.time()
    (R.run_dir / Q.READY_FILE).write_text(json.dumps({"wall": t0, "step": R.st["step"]}) + "\n", encoding="utf-8")
    R.log.event("calibrate_barrier", state="ready", at_step=R.st["step"])
    print(f"[calibrate] ready at step {R.st['step']}: waiting for {R.run_dir / Q.GO_FILE}", flush=True)
    while not ((R.run_dir / Q.GO_FILE).exists() or (R.run_dir / STOP_FILE).exists()):
        time.sleep(CALIBRATE_POLL_S)
    by = Q.GO_FILE if (R.run_dir / Q.GO_FILE).exists() else STOP_FILE
    R.log.event("calibrate_barrier", state="released", by=by, waited_s=round(time.time() - t0, 3))


_validate_trainer, _end_lr_probe_trainer, _stop_requested_trainer = validate, end_lr_probe, stop_requested
_BARRIER_PASSED: set[str] = set()  # the run dirs whose first stop check has waited at the barrier


def validate(cfg: dict):  # noqa: F811  the trainer's checks, then the box keys' (the block comment above)
    _validate_trainer(cfg)
    validate_box(cfg)


def stop_requested(R: Run) -> bool:  # noqa: F811  a calibration group's first step waits at the barrier
    if calibrate_on(R.cfg) and R.cfg["calibrate"]["barrier"] and str(R.run_dir) not in _BARRIER_PASSED:
        _BARRIER_PASSED.add(str(R.run_dir))
        calibrate_barrier(R)
    return _stop_requested_trainer(R)


def end_lr_probe(R: Run, step: int) -> int:  # noqa: F811  a calibration run ends with its step-time table instead
    return end_calibrate(R, step) if calibrate_on(R.cfg) else _end_lr_probe_trainer(R, step)


if __name__ == "__main__":
    run_script()
