# Running the viability run on vast.ai

One A100 for ~4.5 h: the box boots our image, clones this repo at a pinned commit, pulls the small derived data from a
private HF dataset, rebuilds the audio from the public upstream datasets, trains for 4 h, uploads everything to a
private HF model repo, verifies the upload and destroys itself. A watchdog stops it at 5.5 h whatever happens.
Nothing is rented until you run `launch.py --yes`.

The same scripts also rent the **label box**: one RTX 5090 that labels the full download with both teachers and uploads
the labels to the data repo (`--job label`, see "The label run" below), the **size study's boxes**: 4x or 1x A100
that run a queue of calibration, LR probes and study runs (`--job study --box ...`, see "Size study"), and the
**full-data boxes**: 1x or 2x RTX 5090 that train the full students from the box registry `configs/full/boxes.json`
(`--job full --box ...`, see "Full-data runs" at the end).

| file | runs where | what it does |
|---|---|---|
| `launch.py` | laptop | search offers with the strict filter, print the exact `vastai create instance` command, create only with `--yes` |
| `onstart_stub.sh` | box, every container start | what `--onstart` sends (kept under 4 KB): clone at `KITSUNE_SHA`, then run `onstart.sh`; stops the box if the clone fails |
| `onstart.sh` | box, from the stub | env sync, limits, TensorBoard + vast portal, start watchdog, then `bootstrap.sh` + `supervise.py` |
| `bootstrap.sh` | box | pull derived data from `KITSUNE_DATA_REPO`, rebuild audio with `scripts/01_prepare_data.py` (pinned upstream commits, up to `KITSUNE_DOWNLOAD_AHEAD` = 6 files downloading ahead of the ingest), check the id join |
| `supervise.py` | box | run `scripts/04_distill.py`; resume once after a late crash; then `finish.py --destroy` or `--stop` |
| `finish.py` | box | upload, verify every file in the HF output repo (path, size, hash), destroy; stop instead if anything is off |
| `watchdog.sh` | box | stop the instance 5.5 h after first boot (log sync 10 min before); stop (or only alert) when the controller's heartbeat goes stale |
| `blocklist.json` | laptop | machines no job rents (launch.py adds them to every avoid set) |
| `../kitsune/netgate.py` | box (full job) | the download gate: times three pinned upstream files before the paid rebuild, refuses a slow host |

Everything on the box logs to `/workspace/kitsune.log`; lifecycle records and timings are in `/workspace/kitsune_state/`
and get uploaded to `runs/<run_id>/infra/` in the output repo.

## One-time setup

### 1. The image (GitHub Actions, no home upload)

Push a commit that touches `docker/**`, `requirements-train.txt` or `.github/workflows/image.yml`. The `image` workflow
builds `ghcr.io/multysquid/kitsune-train`, smoke-tests it and tags it `sha-<short>` and `<branch>` (~15-25 min). The run
summary shows the digest.

After the **first** build, make the package public once so vast can pull it without credentials (it contains only
code dependencies): github.com -> your profile -> Packages -> `kitsune-train` -> Package settings -> Change visibility
-> Public.

### 2. Hugging Face repos and token

1. Create the two private repos (names are examples; use yours everywhere below):
   ```bash
   hf repos create Multy123/kitsune-data --repo-type dataset --private
   hf repos create Multy123/kitsune-runs --private
   hf upload Multy123/kitsune-runs MODEL_CARD.md README.md
   ```
   The last line (from the repo root, once) gives the run repo its model card: HF renders only the repo-root
   README.md, and its terms (GPL-3.0, non-commercial, trained on Galgame; the Apache-2.0 modified-from notice) are
   the ones every checkpoint also carries as its own README.md.
2. Upload the derived data (~2.3 GB, most of it the student init; the audio is rebuilt on the box). Finish the
   second-opinion pass for every train source and rebuild the selection first, with the run config's recipe (its
   sources, eval sets and `selection_recipe`: the agreement thresholds and the label-filtered hold-outs):
   ```bash
   python scripts/make_selection.py --config configs/viability.json
   ```
   `launch.py` refuses a data repo where a train source's teacher shard has no second opinion, or where the selection
   still has `no_agree` rows, was built with other sources / eval sets / thresholds than the config's, keeps no rows
   of a configured source or eval set, or keeps rows whose teacher output is not uploaded. A config without an
   `extent` (viability) rebuilds with `01_prepare_data.py`'s defaults (the first 6 Galgame tars, 300 h of
   Emilia-YODAS), so its uploaded `teacher_out` must not go beyond them. A config with an `extent`
   (`configs/full.json`, `configs/full_sub3k.json`) reads the label box's `labels/full/` instead: the box rebuilds
   exactly that extent through the canonical ingest order and pulls only its label files (see "Subsets" below), and
   launch refuses it until `labels/full/COMPLETE.json` exists. From the repo root:
   ```bash
   hf upload Multy123/kitsune-data . . --repo-type dataset \
     --include "teacher_out/meta.json" --include "second_out/meta.json" \
     --include "teacher_out/reazon_small/*" --include "teacher_out/emilia_yodas/*" --include "teacher_out/galgame/*" \
     --include "teacher_out/eval_*/*" \
     --include "second_out/reazon_small/*" --include "second_out/emilia_yodas/*" --include "second_out/galgame/*" \
     --include "second_out/eval_*/*" \
     --include "selection/viability.parquet" --include "students/b20x2560-d4/*"
   ```
   Re-running it after an interruption only sends what is missing. Optional, only if your uplink is fast enough: also
   `--include "data/shards/<source>/*.parquet"` for a source to ship its audio instead of rebuilding it on the box
   (bootstrap.sh uses parked shards when they are there).
3. Create a **fine-grained** token at https://huggingface.co/settings/tokens: `kitsune-runs` needs read AND write
   access to its contents (the trainer reads its uploads back), `kitsune-data` needs read. If the page applies one
   permission set to every selected repo, read + write on both is fine. No create permission is needed (both repos
   exist), and no gated-repo access: the parked student dir carries the processor and tokenizer files, and the
   upstream datasets are public. The size study's box B is the exception: its speed probe times the Cohere teacher
   itself, so for that box the token also needs read access to the gated
   `CohereLabs/cohere-transcribe-03-2026` (without it that one readout fails, recorded, and the box still
   ends complete). Bootstrap checks this token against both repos in its first minute on the box.

### 3. vast.ai account

1. Put credit on the account.
2. Account -> Settings -> **Environment Variables**: add `HF_TOKEN` = the token from step 2.3. vast injects it into
   every instance you start; it is never on a command line and never in the image. (It is readable by anyone who can
   SSH into your instances.)
3. Install the CLI on the laptop and store your API key (you type it; it goes to `~/.config/vastai/vast_api_key`):
   ```bash
   pip install vastai==1.8.0
   vastai set api-key <key from https://cloud.vast.ai/manage-keys/>
   ```
4. Add your SSH public key under Account -> Keys **before** renting (keys added later do not reach running instances).

## Each run

1. Commit and push the code you want to run (the box clones exactly that commit), and wait for the image build if
   you changed `docker/` or `requirements-train.txt`. `launch.py` reads the commit the image was built from and
   refuses it while `requirements-train.txt` or `docker/Dockerfile` differ at the commit you run. CI builds an image
   only for pushes that touch those files, so a branch with code changes only has no image tag of its own: add
   `--image-tag main`.
2. Look first (read-only, spends nothing):
   ```bash
   python vast/launch.py --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs
   ```
   It checks the commit is pushed, resolves the image tag to a digest and checks the image was built with this
   commit's dependencies, checks both HF repos with your local login
   (including that both are private and the derived data is complete, see above), then searches offers (verified A100 SXM4 40 GB,
   reliability >= 0.98, driver CUDA >= 13.0, >= 12 CPU cores, >= 64 GB RAM, disk and network (down and up) >= 500 MB/s
   / Mbit/s, room for the 150 GB disk; falls back to SXM4 80 GB), prints a table with each host's bandwidth $/GB, the
   exact create command and the cost cap including ~25 GB down / ~30 GB up. The checks before the search also run
   without the vastai CLI. Offers come and go within minutes, so look again right before renting.
3. Rent:
   ```bash
   python vast/launch.py --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --yes
   ```
   Options: `--offer-id N` to pick another row, `--max-dph 1.0` to cap the price, `--image ...@sha256:...` to pin an
   image by hand, `--config configs/<other>.json`, `--no-self-stop` to debug a box (see "How it ends").

## The next runs: `configs/next_run_template.json`

`configs/viability.json` (the default `--config`) is the first A100 run's config and stays as it was run; the rest of
this README describes it. The scaling study's configs combine two files: the DATA block (`teacher_root`,
`second_root`, `parakeet_root`, `extent`, `student`, `sources`, `eval_sets`, `selection`, `selection_recipe`) from the
label box's extent config (`configs/full_sub3k.json` or a subset derived from it, see "The label box" below) and the
TRAINER settings from `configs/next_run_template.json`; then set `run_name` and `schedule.epochs` and rent with
`--config configs/<study>.json`. The template is viability.json with the first run's report recommendations (its
`_comment` has the details and the arithmetic):

| key | viability.json (first run) | template | why |
|---|---|---|---|
| `batch.micro_audio_s` | 400 | 600, the memory probe's first choice (it halves on OOM) | ~0.14 s fixed cost per micro-batch: ~+18 % audio-s/s at ~29-30 GiB (estimated) |
| `schedule.clock` / `epochs` | wall, 4 h | epochs, 8 | the cooldown counts steps, so evals cost it no LR; the last epoch's eval is the final eval |
| `eval.full_every_epochs` | 1 | 2 (mini evals every 500 steps in between) | complete evals took 13 % of the loop |
| `eval.verdict_version` | 1 | 2 | the CER trend on the pre-cooldown evals, complete sets, de-duplicated end points; cooldown gain and pre-cooldown slope reported |
| `perf.profile_smoke` | "auto" (the default: on under CUDA; the first run had no profiler) | true | the profiler record of the smoke steps (below) |
| `eval.reference` | none | null until Parakeet 0.6B ja's complete-set CER is in a file | the reference model next to the gate (below) |

On the epochs clock the trainer does not shorten the run to the watchdog's deadline (it does so on the wall clock
only), so size `schedule.epochs` to fit well inside `--max-hours`: steps per epoch (the `plan` event; 1,216 for the
viability data) x ~1.05-1.25 s, plus ~4 min per complete eval, ~5 min for the smoke profile (estimated) and ~15 min
of box overhead. Verdict v2 needs at least 3 complete evals before the cooldown: with a complete eval every 2 epochs
and a 20 % cooldown that means 8 epochs or more (epochs 2, 4 and 6 before the cooldown at 6.4); with fewer it reports
"trend: insufficient pre-cooldown evals". So runs under 8 epochs set `eval.full_every_epochs` 1 (the owner's decision
for the scaling study: 4 epochs then give epochs 1-3 before the cooldown, for ~6 % more loop time).

## Watching it

```bash
vastai show instance <id>          # wait for "running"
vastai ssh-url <id>                # -> ssh://root@<ip>:<port>
ssh -p <port> root@<ip> -L 6006:localhost:6006
```
Then open http://localhost:6006 for TensorBoard, and on the box `tail -f /workspace/kitsune.log`. onstart.sh waits
up to ~20 s for its TensorBoard to answer and logs `TensorBoard up on 127.0.0.1:<port>` (if port 6006 was taken on the
box, forward that port instead), `TensorBoard not answering ... (still starting?)` (it is alive but slow: try again in
a minute) or `TensorBoard did not start: <last line of /workspace/tensorboard.log>` (it exited); either way the boot
goes on, and a run without a viewer still logs everything (`tb/` and the open files reach the output repo). It keeps
up to 30,000 points per scalar tag (`--samples_per_plugin scalars=30000`), so the per-step curves are drawn at every
step of the run; TensorBoard's default keeps a random 1,000.

Timeline: ~5 min boot and image pull, ~10-20 min data (derived pull + audio rebuild, see
`/workspace/kitsune_state/bootstrap_timings.jsonl`), 10-15 min smoke phase, 4 h training with evals, ~15 min final
eval, upload and verification. The trainer reads the watchdog's deadline and shortens the 4 h (earlier cooldown)
when the final eval, the uploads and a 30 min reserve would not fit before it, e.g. after a crash and resume; the
`budget` event in `events.jsonl` shows the result. It also ends early when the held-out KL goes flat (`early_stop` in
`configs/viability.json`): after 3 evals in a row (3 epochs) without a 0.5 % improvement it starts the cooldown at once,
over 20 % of the time trained so far, and then goes to the final eval; the `early_stop` event and `stopped_early` in
`summary.json` say when and why. A trigger that comes when the scheduled cooldown has already begun changes nothing:
the run goes to its scheduled end, `stopped_early` stays null and `early_stop_trigger` in `summary.json` (and the event,
with `cooldown.already: true`) records it.

Evals (`eval` in `configs/viability.json`): a full eval at the end of every epoch - teacher-forced and greedy on the
complete eval sets, plus the probe - and a mini eval every 500 optimizer steps (32 utterances per gate set, 64 of the
probe). The overall loss chart is TensorBoard's Custom Scalars tab, `combined_loss: train vs val`: the training
objective 1.0 * KL + 0.8 * CE per target token (no L2-SP term) at every optimizer step (`train`, on augmented audio),
on the mini evals' gate subsets every 500 steps from step 0 on (`val`) and on the complete gate sets at every epoch
end and the final eval (`val_full`); the same three curves are the first cards of `2_loss_accuracy/00_combined/`.
The numbers to read first are in TensorBoard under `2_loss_accuracy/00_summary/full/` and `.../mini/`
(`val_cer_pct`: CER vs the reference pooled over JSUT / CV8 / ReazonSpeech-test, `val_cer_vs_teacher_pct`,
`train_cer_vs_teacher_pct`, `val_loss`, `train_loss`, ...), and each eval prints one line to `kitsune.log`:
`[full eval] step N epoch E | val CER x.x% (vs teacher y.y%) on U utts (complete) | train CER vs teacher z.z% (vs ref
w.w%) | val KL ...`. The first point of `00_summary/full/` (step 0) decodes only the fixed 500-per-set subset (1,500
gate utterances, "subset" in its line), every later point the complete gate sets (14,746); `val_cer_utts` next to
`val_cer` shows which.
The cost of a full eval on the A100 is an estimate until the first run measures it (`eval/wall_s`,
`eval/greedy_full/rtf`): ~5-10 min (the laptop measured ~10 min for this student in batched decoding; the `_comment`
in `configs/viability.json` has the arithmetic), against ~12-47 min of training per epoch.

A reference model next to the gate (optional, off by default): `"reference": {"name": "<label>", "path":
"configs/reference/<model>.json"}` under `eval` in the run config makes the verdict report, per gate set and pooled,
the student's final CER next to that model's and the ratio student / reference (`verdict.reference` in `summary.json`
and `verdict.json`, and one `[reference] ...` line in `kitsune.log`). It never changes the tier. The file is read
from the repo clone at setup, so commit it with the code; a missing or malformed one stops the run before it trains.
Its format (the numbers only show the format; each model's come from scoring its transcripts of the complete gate
sets with the gate's own corpus CER, `kitsune.evaluate.corpus_cer`):
```json
{"name": "parakeet-tdt-0.6b-ja", "scope": "complete gate sets, corpus CER under kitsune.text.normalize_ja",
 "cer": {"eval_jsut": 0.0731, "eval_cv8": 0.0795, "eval_reazon": 0.0718}}
```

To stop the training by hand (it looks flat on TensorBoard, or the results are already what you need):
```bash
ls /workspace/Kitsune-Transcribe/runs/                       # the run id: <run_name>-<UTC stamp>
touch /workspace/Kitsune-Transcribe/runs/<run_id>/STOP
```
The trainer checks for the file before every optimizer step: the step under way finishes (with its eval and
checkpoints, if due), then the normal end phase runs - final weights and full state, final eval (an epoch-end full
eval of that same step is reused, not decoded again), verdict, summary, uploads - and it exits 0, so the box verifies the upload and destroys itself as after a full run (`early_stop` event
with reason `stop_file`). Killing the trainer instead, before its `summary.json` is complete, counts as a crash
(resumed once, or the box is stopped; see below) and skips the final eval. A trainer stuck waiting for its data
loader (`kitsune.log` shows `unable to allocate shared memory` and no new steps) never reaches the STOP check, but it
gives up after `perf.loader_timeout_s` (10 min) and exits as a crash.

## Where the results land

In the output repo under `runs/<run_id>/`: `summary.json` (verdict and final metrics), `metrics/`, `evals/`, `tb/`
(TensorBoard events), `events.jsonl`, `checkpoints/step_<N>/` (bf16 weights every 30 min), the full resume state from
before the cooldown and at the end, and `infra/` (box logs; exported too). `smoke/profile/` holds a torch.profiler
record of ~20 of the smoke steps (steps 22-41 of the 100; `perf.profile_smoke`, on under CUDA; it does not change what
the steps compute, but it costs a few minutes - ~4-5 min extrapolated from the laptop CPU, not measured on a GPU -
which on the wall clock come out of the training time; `summary.json`'s `wall_s` and `cycles[].process_s` have the
measured cost): `summary.json` - per step the data-wait and GPU-kernel-time shares, kernel launches and aten ops per
micro-batch, device syncs and the host scalar reads behind them, the top ops by self CUDA and self CPU time - and the
first two recorded steps' raw trace, `trace_steps_<a>-<b>.json.gz` (gzip, dropped
above 32 MB; open it in https://ui.perfetto.dev or chrome://tracing). The `smoke_profile` event has the headline. Convert to flat files with
`python tools/export_run.py hf://Multy123/kitsune-runs/runs/<run_id> --out <dir>`, or run TensorBoard locally on
a downloaded `tb/` (`python -m tensorboard.main --logdir <dir> --samples_per_plugin scalars=30000`, to see every step
as on the box; the export's `combined_loss` table and `metrics/steps.parquet` hold every step too).

## How it ends

| outcome | instance |
|---|---|
| training exits 0 and every expected file is on the Hub (path, size, sha256) | **destroyed** (billing stops) |
| training exits 0 but verification finds a problem | **stopped** (disk kept, storage still billed) |
| any other exit but 3, or a container restart, after `summary.json` says `complete` (a crash in teardown, a kill while the end state uploads), even a second crash | as for exit 0: **destroyed** once verified, otherwise **stopped**; no resume and no post-crash sync (`--destroy` uploads everything) |
| exit 3 (throughput too low), a crash before step 100, or a second crash | **stopped** |
| first crash after step 100 with a local full state (and no complete `summary.json`) | resumed once, then as above |
| bootstrap or on-start failure | **stopped** (`finish.py --abort`; a full box without a run dir yet is destroyed, see "Full-data runs") |
| 5.5 h after first boot, whatever the state | **stopped** by the watchdog |

A stopped instance keeps its disk. To inspect it: `vastai start instance <id>` (may wait in "scheduling" if the GPU was
re-rented), SSH in, read `/workspace/kitsune.log` and `/workspace/kitsune_state/`. A restarted container never starts a
second run on its own (the `halt` marker). To start a fresh run on it:
```bash
bash /workspace/Kitsune-Transcribe/vast/onstart.sh --rearm; tail -n 20 /workspace/kitsune.log
```
It moves the halt marker, the old deadline and the supervisor/finish history to `/workspace/kitsune_state/rearm-<time>/`
and boots as on a new instance: a new 5.5 h cap from now, a new supervisor history (a new run dir; the old run dirs stay
under `runs/` and are verified again before a destroy). Deleting only the `halt` file does not work: the old deadline
has usually passed (the watchdog would stop the box at once) and the old history already holds a final decision, which
the supervisor then carries out again (a decision with no marker reads as a finish cut short by a restart).
`--rearm` refuses (exit 1, reason in the log: the script writes only to `/workspace/kitsune.log`) while a watchdog or
supervisor of the current container is still running; stop and start the instance first. Running the on-start script
again during a healthy run starts nothing (the supervisor holds `kitsune_state/supervise.lock`). When done:
`vastai destroy instance <id>`. For debugging a fresh box without it stopping itself on a bootstrap error, pass
`--no-self-stop` to launch.py (it adds `-e KITSUNE_NO_SELF_STOP=1` inside the `--env '...'` value). By hand, put it
inside that quoted value: vastai has no `-e` option of its own, and a second `--env` replaces the first, dropping
KITSUNE_SHA and the rest. The watchdog still stops the box at the cap once `onstart.sh` has started it; with the flag,
a box whose stub fails before that (e.g. the clone) keeps running until you destroy it by hand.

## onstart.sh: what runs at every container start

vast/launch.py passes the small `vast/onstart_stub.sh` with `vastai create instance --onstart` (the API may cap that
field near 4 KB); the stub clones the repo at `$KITSUNE_SHA` and execs `vast/onstart.sh` from the clone. onstart.sh
still clones by itself if run on a box without the stub (its clone is a no-op when the checkout is already at
`$KITSUNE_SHA`). It runs as root at EVERY container start (SSH launch mode), and stays under vast's 16 KB on-start
limit (a test holds it there), which is why its story lives here.

Steps: sync the env to `/etc/environment` (SSH/tmux sessions do not inherit the container env), raise the nofile
limit, check `/dev/shm`, size the CPU thread pools (below), start TensorBoard and the vast portal, clone the repo at
`$KITSUNE_SHA`, start `vast/watchdog.sh` (the hard cost cap), then detached: `vast/bootstrap.sh` (data) and
`vast/supervise.py` (training + stop/destroy), all logging to `/workspace/kitsune.log`.
- `KITSUNE_JOB=label` (the label box, `launch.py --job label`): the detached part is `vast/label.py` alone. It holds
  `supervise.lock`, runs its own idempotent steps and resumes from `$KITSUNE_STATE/label.json`, so bootstrap.sh and
  supervise.py are not run.
- `KITSUNE_JOB=study` (a size-study box, `launch.py --job study --box A|B|replicate|shakedown`): the train job's path:
  bootstrap.sh (the extent, both label roots, the box's students), then supervise.py, which runs the box's queue
  (`kitsune/study_queue.py`) instead of one trainer; `KITSUNE_BOX` names the box. `--rearm` leaves
  `$KITSUNE_STATE/queue.json` in place: the queue resumes where it stopped (its PREREG numbers are written once;
  deleting the file by hand starts the box's study over).
- `KITSUNE_JOB=full` (a full-data box, `launch.py --job full --box ...`): the same path with the full queue
  (`kitsune/full_queue.py`); it needs `KITSUNE_BOX` and `KITSUNE_OUT_REPO`, and `$KITSUNE_STATE/train_hb` (the box
  controllers' heartbeat, which its watchdog reads) is touched as bootstrap starts.

Thread pools (fix 2): a vast container's `nproc` is the host's (the first A100 box: 128 CPUs, a cgroup quota of 15.36),
so for every job but label the six pools (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `RAYON_NUM_THREADS`, `TOKIO_WORKER_THREADS`) get the quota over the box's GPUs:
`floor(cpu.max quota / period) / KITSUNE_N_GPUS` (box A's 61.44 CPUs on 4 GPUs: 15), nproc without a quota, at most
16 on a pids budget below 16 per visible CPU (the base image's `12-cpu-thread-limits.sh` trigger), a value already set
winning; `KITSUNE_CPU_QUOTA` and `KITSUNE_THREADS_PER_GPU` go to every child with them. The label job keeps its rule
(16 on a low pids budget only). The loader workers follow the same share (`kitsune.ctc_preflight.per_gpu_cpus`).

Restarts: the watchdog deadline is fixed at first boot; a `halt` marker (written by finish.py or by a failure here)
means the run is over, so a restarted container only brings up the env and the portal for inspection. An interrupted
run is handled by supervise.py's own history: a restart that finds `$KITSUNE_STATE/supervise.json` skips bootstrap
(the data passed its coverage check before the supervisor first ran) and hands straight over to it. Any failure before
the supervisor takes over runs `finish.py --abort` (a stop without a sync; a full box without a run dir yet is
destroyed instead), unless `KITSUNE_NO_SELF_STOP=1`. The detached subshell holds `$KITSUNE_STATE/supervise.lock` while
bootstrap runs (bootstrap and its children never get the lock's fd), then the supervisor holds it, so running the
script again by hand during a run starts nothing new; the supervisor waits up to 3 minutes for a lock still held when
it starts before it steps aside.

Re-arm a halted box for a fresh run: `bash vast/onstart.sh --rearm` moves the halt marker, the deadline, the
supervisor/finish history and the heartbeats (`label_hb`, `train_hb`, `hb/`), a full box's `resume_plan.json`,
`download_gate.json`, `watchdog_alerts.jsonl` and `smoke_verdict.json` to `$KITSUNE_STATE/rearm-<stamp>/`, then boots
as if for the first time (a new cap: export `KITSUNE_MAX_HOURS` first). It refuses while a watchdog or supervisor of
this container still runs (stop/start the instance first).

Instance env (set by launch.py): `KITSUNE_SHA KITSUNE_CONFIG KITSUNE_DATA_REPO KITSUNE_OUT_REPO TZ`, optional
`KITSUNE_DATA_REVISION KITSUNE_MAX_HOURS`; the label job has `KITSUNE_JOB=label` and no `KITSUNE_OUT_REPO`; the study
and full jobs add `KITSUNE_BOX` and `KITSUNE_N_GPUS` (the full job's table is under "Full-data runs"). HF_TOKEN comes
from the vast account env and is never printed.

## The label run (one RTX 5090 labels the full download)

`python vast/launch.py --job label` rents one on-demand RTX 5090 (32 GB) that labels the full downloaded extent
(~13.0k h, `configs/full.json`) with both teachers, and destroys itself once everything is verified on the Hub. On the
box, `onstart.sh` hands over to `vast/label.py` (the controller; `bootstrap.sh` and `supervise.py` are not used). It
rebuilds the audio from the pinned upstreams with one `01_prepare_data.py --extent-config` process and runs, on the one
GPU at the same time, two Cohere processes (`02_teacher_pass.py`, the shards split between them) and one Parakeet
process (`02p_parakeet_pass.py`). Each train shard's audio is deleted as soon as both its label files are checked on
disk, which keeps the disk at 250 GB. `02b_second_opinion.py` runs on the CPU every 20 min; Galgame is judged by the
Parakeet hypothesis. At the end the box writes the extent record, the selections, the reports and `COMPLETE.json`.
Laptop Cohere labels whose ids match the box's shards (reazon_small, galgame tars 0-5, Emilia 300 h, the eval sets)
are copied byte for byte (`provenance/adopted.json`), not recomputed; the three gate sets are always the laptop's.
No audio is ever uploaded, and the laptop's roots (`teacher_out/`, `second_out/`, `selection/`, `students/`) are never
written.

| file | runs where | what it does |
|---|---|---|
| `label.py` | box | the controller: steps (plan, pull, models, self-test, roots, golden check), lanes with heartbeats, pruning, syncs, ETA, finalize, the ending |
| `label_sync.py` | box | write-once uploads under `labels/` and `label_runs/` only, with the lease; per-directory verify; seal |
| `../tools/publish_parakeet.py` | laptop | the one-time Parakeet model upload (below) |

### What lands in the data repo

```
Multy123/kitsune-data
  models/parakeet-tdt_ctc-0.6b-ja-hf/      the Parakeet model, uploaded once from the laptop (8 files)
  labels/full/
    LEASE.json                             which box owns the root (heartbeat; released at the end)
    teacher_out/<source>/<split>-NNNNN.{npz,jsonl}    Cohere, the 02_teacher_pass.py format, unchanged
    parakeet_out/<source>/<split>-NNNNN.{npz,jsonl}   Parakeet TDT + CTC soft targets (kitsune/parakeet_targets.py)
    second_out/<source>/<split>-NNNNN.jsonl           {id, hyp2, model2, agree, cer2}
    extent.json                            every upstream input -> its shard stems and their id hash
    selections/full.parquet, selections/full_sub3k.parquet
    reports/  galgame_judge, parakeet_baselines, labels, throughput, consumer_check (.json)
    provenance/<run_id>.json, provenance/adopted.json
    COMPLETE.json                          written last, in its own commit: the root is sealed after it
  label_runs/<run_id>/                     the box's logs and state files (scrubbed)
```

Every file under `labels/full/` is written once: a relaunch adds missing files and never changes an uploaded one (a
different remote file is an integrity error that stops the box). The only exceptions are `LEASE.json`, and the
finalize outputs (`extent.json`, `selections/`, `reports/`, `provenance/`) until `COMPLETE.json` exists. Nothing is
ever deleted. The Parakeet format (array names, shapes, dtypes, `meta.json`) is documented in the docstring of
`kitsune/parakeet_targets.py`, which also has the loader and the checker; the top-level README.md summarises it.

### One-time setup for the label run

1. **The token.** The label box writes to the data repo and reads the gated Cohere model, so the one fine-grained
   `HF_TOKEN` in vast's Account -> Environment Variables needs:
   - read AND write on `Multy123/kitsune-data`;
   - read AND write on `Multy123/kitsune-runs` (for the A100 runs, as above);
   - "Read access to contents of all public gated repos you can access" (the Cohere Transcribe gate is already
     accepted on your account).

   No new repo and no create permission. The A100 box then also carries data-repo write it never uses. The
   alternative, swapping the account variable between launches, must never happen while a label box is alive: a
   container restart would pick up the new token. The box checks its token in its first minutes (write on the data
   repo, the repo still private, the gated read) and stops with a refusal if any is missing.
2. **The HF storage quota.** Check it at huggingface.co -> Settings -> Billing / Storage. The run adds ~36-49 GB (see
   "HF storage" below) to the repo's ~2.3 GB; the free private quota is believed to be ~100 GB (unverified).
3. **Upload the Parakeet model once** (2.5 GB, ~11 min at 30 Mbit/s; the model, not audio). From the repo root, with
   the converted model in `cache/parakeet-tdt_ctc-0.6b-ja-hf` (the bake-off's; `--hf-dir` for another path):
   ```bash
   python tools/publish_parakeet.py --upload
   ```
   It checks the converted config, extracts the CTC head from the cached NeMo checkpoint
   (`nvidia/parakeet-tdt_ctc-0.6b-ja@44edb27`), runs a CPU sanity check on 8 eval_jsut clips and uploads the folder to
   `models/parakeet-tdt_ctc-0.6b-ja-hf/` in the data repo (`--repo` and `--path-in-repo` change them). The files must
   match the sha256 pins in `kitsune/parakeet.py`: the box and `launch.py` refuse the run otherwise. Without
   `--upload` it only builds and checks the files.

### Launch

The code must be on `main` and pushed (the box clones exactly that commit). Look first (read-only, spends nothing):
```bash
python vast/launch.py --job label --data-repo Multy123/kitsune-data --image-tag main
```
It checks the commit and the image as for a training run, and with your local login: the data repo is private, the
laptop seed files and the Parakeet files (with their pins) are there, the Cohere gate, no `COMPLETE.json` and no live
lease under `labels/full/`, the extent of every label config, and the repo size with the projection (a warning above
80 GB). If labels are already there (a relaunch), it prints what is done per source. It then searches on-demand RTX
5090 offers (verified, CUDA >= 13.0, >= 16 cores, >= 64 GB RAM, >= 500 Mbit/s down, a direct port, 250 GB of disk;
strict tier first: reliability >= 0.97, disk >= 300 MB/s, >= 100 Mbit/s up) and ranks them by the total estimated
cost with download and upload included (`est$`), not by $/h. The cost line reads like
`~$0.72/h (GPU + 250 GB) x ~16 h (12-24; cap 30 h) + ~590 GB down x $.../GB + ~45 GB up x $.../GB ~= $13 (cap ~= $31)`.

Rent (spending money is always this explicit step):
```bash
python vast/launch.py --job label --data-repo Multy123/kitsune-data --image-tag main --yes
```
Options: `--offer-id N`, `--max-dph 1.0` (the default), `--label-configs configs/full.json,configs/full_sub3k.json`
(the default: the selections built at the end), `--cohere-procs N` (default 2), `--avoid-machine ID` (repeatable;
machines that ended a label run with a host failure are skipped anyway), `--disk-gb N`, `--steal-lease` (only when
you know the other box is gone), `--no-self-stop`. The data revision the box reads its seeds from is pinned to the
data repo's head at launch (`KITSUNE_DATA_REVISION`).

### Timeline and cost

| phase | wall | notes |
|---|---|---|
| boot, image pull, plan, pull, models, self-test | 0.3-0.4 h | |
| ingest up to eval_jsut + golden check | 0.1-0.2 h | the rest of the ~570 GB download overlaps the GPU work |
| Cohere, ~12.3k h not adopted | 9.9-18.5 h (central 12.5) | ~78 % of the job: galgame ~5.1k h, reazon_large ~4.9k h, emilia_nc ~1.6k h, emilia_yodas ~0.8k h |
| Parakeet, ~13.05k h, on the same GPU | +1.5-3 h | |
| 02b, pruning, syncs | ~0 | overlapped |
| finalize (selections over ~8 M rows, reports, consumer checks, final verify, seal) | 0.6-0.9 h | |
| **total** | **~12-24 h, central ~16 h** | cap 30 h; the box ends itself at 29 h |

| cost | low | central | high |
|---|---|---|---|
| box, hours x $/h (GPU + 250 GB) | 12 x 0.66 = $7.9 | 16 x 0.72 = $11.5 | 24 x 0.82 = $19.7 |
| download ~585 GB | $0.6 | $1.5 | $5.9 |
| upload ~45 GB | $0 | $0.1 | $0.45 |
| **label run** | **~$9** | **~$13** | **~$26** |
| at the 30 h cap | | | ~$31 |
| each relaunch after a lost box | | +$1-4 | |

The 5090 rates are extrapolated from the laptop; the first hour's rates line tells. Of EUR 50 (~$58) this leaves
~$32-49 for the A100.

### Watching it

```bash
ssh -p <port> root@<ip> tail -f /workspace/kitsune.log
```
It shows the steps, the golden checks (Cohere CER <= 1 % against the laptop hypotheses on 64 eval_jsut rows, Parakeet
<= 2 % against the committed golden on 32), a rates/ETA line every 30 min (x realtime per lane, hours left, ETA against
the deadline) and a line per sync. Each lane logs to `/workspace/lane_<name>.log` (`ingest`, `cohere-0`, `cohere-1`,
`parakeet`); the controller's state is `/workspace/kitsune_state/label.json`. On the Hub, a commit under `labels/full/`
arrives about every 20 min, so a box that dies loses at most ~20 min of labels plus the shards in flight.

### How it ends

| outcome | instance |
|---|---|
| finalize passed and `COMPLETE.json` is verified on the Hub | **destroyed** |
| budget: the deadline minus 60 min with work left, or the work done too late to finalize | final sync + verify, then **destroyed**; relaunch to continue |
| host failure: self-test or golden check fails, a lane fails or hangs 4 times without progress, disk full while ingest is held, Cohere under 400x realtime after an hour of steady work (`slow_host`) | final sync + verify, then **destroyed**; the machine is avoided on the next launch |
| refusal: the token cannot write the data repo or read the gated model, the repo is public, a live lease, `COMPLETE.json` present, seeds or Parakeet files missing or not matching their pins, pulled label settings differ | **stopped** (minutes in; nothing unique yet) |
| integrity: an existing label that does not match its shard, a gate-set adoption that fails, a write-once conflict, a missing reazon capture, a Parakeet row missing for a teacher row, the consumer check fails | final sync of the non-conflicting files, then **stopped**; the disk keeps the evidence |
| a bug in label.py, or the final verify fails (Hub outage at the end) | best-effort sync, then **stopped** (the labels on disk are unique) |
| the controller died or hung | the watchdog syncs after 15 min without its heartbeat, then **stops** |
| anything else | the watchdog syncs 30 min before the 30 h cap and **stops** at it |

A destroy happens only after the Hub provably holds every finished label; any doubt ends in a stop.
`KITSUNE_LABEL_END=stop` turns every destroy into a stop (debugging). launch.py builds the one `--env` value itself
and has no flag for it, so it goes in the vast account environment, which applies to every box (the A100 too):
remove it afterwards. `--no-self-stop` works as for a training run. The reason is in `label_runs/<run_id>/` (`label_end.json`) and in `kitsune.log`.

### Relaunch and continue

- **Destroyed without `COMPLETE.json`** (budget, host failure): run the same launch command again. The new box sees
  the labels already under `labels/full/`, pulls them, re-downloads the audio (shards already labelled pass their
  checks at once and are pruned) and resumes at the first unlabelled shard: +0.5-1.5 h and +$1-4. The launch table
  prints what is already done per source.
- **Stopped** (refusal, integrity, a bug): read `label_runs/<run_id>/` and `/workspace/kitsune.log`, fix the cause,
  then either destroy it by hand and relaunch, or continue in place:
  ```bash
  vastai start instance <id>
  bash /workspace/Kitsune-Transcribe/vast/onstart.sh --rearm; tail -n 20 /workspace/kitsune.log
  ```
  `--rearm` also moves `label.json` and `label_hb` aside; the controller re-plans against its own lease, pulls only
  the files missing locally and resumes.
- **A container restart** on the same disk resumes by itself from `label.json` (no Hub calls; at most one shard per
  lane is lost).
- A second box can never write the same root: the lease refuses a launch or a plan while another container's lease
  heartbeat is under 45 min old. `--steal-lease` overrides it, only for a box you know is gone.

### Subsets

A config chooses how much of the labelled extent a training run uses, with no relabelling. Its `extent.inputs` takes
a prefix of each capped source's sorted upstream inputs: `{"reazon_large": N}` (of 705 files), `{"galgame": N}` (of
115 tars, N >= 1), `{"emilia_nc": N}` (of 66), and `emilia_yodas` as `"300h"` or an int >= 9 (the 300 h cut, then the
rest over the first K tars). `configs/full_sub3k.json` is the example (~3.4k h: reazon_large 216 files, galgame 16
tars, all of emilia_yodas and reazon_small, no emilia_nc). A new subset or new thresholds is a new config plus a
selection derived on the laptop in seconds from the full one:
```bash
python scripts/make_selection.py --config configs/<new>.json \
  --from-selection labels/full/selections/full.parquet --out labels/full/selections/<new>.parquet
```
Upload that one parquet (a new file under `labels/full/selections/`) and launch the A100 run with that config after
the upload, so it pins the new head. The A100 box rebuilds only that subset's audio through the same canonical ingest
order (`01_prepare_data.py --extent-config`), pulls only its label files (`parakeet_out` only for a CTC student,
`"family": "ctc"`, or with `"pull_parakeet": true`: `kitsune.extent.pull_plan`), and requires the
rebuilt shards' ids to match `extent.json` exactly. `launch.py` sizes the disk, the rebuild timeout and the cap from
the record:

| config | download | disk | rebuild cap |
|---|---|---|---|
| `configs/full.json` | ~570 GB | ~1,350 GB | ~6.5 h |
| `configs/full_sub3k.json` | ~163 GB | ~500 GB | ~2.2 h |

The subset is the practical path for the remaining budget.

### The size study's selection and pre-registration

The size study trains every student of both families on one frozen 1,000 h list, built once on the laptop CPU from
the sealed root (`study/data.json` is its data block; every study run config carries those keys verbatim):
```bash
python scripts/make_selection.py --config study/data.json --skip-audio-check   # labels/full pulled to the repo root
python -m kitsune.prereg --write study/ --sidecar labels/full/selections/study_1000h.json   # the pre-launch commit
```
The selection's `selection_recipe.study` block adds, after the label box's judges (F0), the rules `not_in_parakeet`
(rows in both label roots only), `f1a_disagree` (CER of the Cohere vs Parakeet TDT hypotheses > 0.5), `eval_dup` (a
reference or Cohere hypothesis equal to an eval reference of >= 15 characters), `ctc_infeasible` and `not_drawn` (a
seeded 3,600,000 s draw pooled over the sources; `keep` = drawn). Eval sets keep every row present in both roots.
Next to `study_1000h.parquet` it writes, with no timestamp and no machine path (a rebuild from any checkout or output
directory gives the same bytes with the same pyarrow; the parquet records the config's own roots and the content
hashes of the extent record and the kotoba file), `study_1000h.json` (hours per source after each rule, the draw,
`ids_sha256` of the train list, the probe and each eval set, the Galgame views, the teachers' baselines `cohere`,
`parakeet-ctc` and `parakeet-tdt` per scoring stratum and M4, and the pool had reazon_large been capped at N, with the
cap rule's readout: the smallest N whose pool holds >= 1,010 h) and `study_manifest.json` (per eval set the ordered
ids and their sha256; the Galgame views `neutral` from the laptop's `second_out/galgame/eval-00000.jsonl`, `all` and
`label_box`). If the build reports that the configured reazon_large cap is not the rule's, set
`extent.inputs.reazon_large` in `study/data.json` to the rule's N and build again. Upload all three; the study box
pulls all three (`pull_plan`). `launch.py` refuses a study selection built with another recipe or seed than the
pre-registered ones, any eval row without Parakeet labels (K6), more than 0.1 % of its train rows in `teacher_out`
only (K5), or a missing sidecar or manifest.
`kitsune/prereg.py` holds the rules (`study/PREREG.{json,md}`, committed before the study box starts; the fields that
need the sealed labels stay `pending` until the command above fills them, and the fill refuses a sidecar whose
sources, eval sets, recipe, seed, extent or reazon_large cap are not the registered ones) and the functions the box
derives its numbers with: `max_steps` (calibration; the replicate takes study-t01's), `choose_lr` (the edge rule) and
`write_numbers` (`PREREG_numbers.json`, strict JSON, its sha256 logged before the first study step). The rules'
`analysis`, `manifest` and `baselines` blocks are the form `kitsune.study_stats` reads
(`tools/study_report.py --prereg study/PREREG.json`).

### HF storage

| item | size |
|---|---|
| existing (laptop labels, selection, student) | 2.3 GB |
| Parakeet model | 2.5 GB |
| `labels/full/teacher_out` | 16-20 GB (~1 GB of it adopted copies, likely deduplicated) |
| `labels/full/parakeet_out` | 13-22 GB (~1.34 MB per audio hour) |
| `labels/full/second_out` | ~1.5 GB |
| selections | ~0.5 GB |
| **total** | **~36-49 GB**, ~27k files |

If the quota binds: a started root keeps its Parakeet settings (the box refuses mixed settings), but on a fresh root
`--k-tdt 4 --k-ctc 4` for `02p_parakeet_pass.py` saves ~3.5 GB.

### After `COMPLETE.json`

Read `labels/full/reports/galgame_judge.json` (kotoba vs Parakeet on the 58 kotoba-judged galgame shards) and
`reports/parakeet_baselines.json` (TDT and CTC CER per eval set next to Cohere's), then plan the A100 run with
`configs/full_sub3k.json` (or `configs/full.json`, or a derived subset).

## Size study (the study boxes)

The size study (study/STUDY.md; CONTRACT.md sections 6 and 8) runs on up to four boxes: the shakedown first, then
boxes A and B **at the same time**, and the replicate only if the pre-registered rule asks for it. Each is one
`launch.py --job study --box <box>` call; on the box `vast/supervise.py` runs `kitsune/study_queue.py` for that box
instead of one trainer. The box's plan (runs, probe classes, calibrated runs, reference run, numbers file) is
`kitsune.prereg.rules()["boxes"][<box>]`; the rules, the probe grids and every number rule come from `kitsune/prereg.py`.

| box | GPUs | what it runs | central hours (cap) | cost at the 2026-09-25 snapshot |
|---|---|---|---|---|
| `shakedown` | 1 | both families (every run of boxes A and B): the stores of the real extent (the AED store, the CTC store with its frame preflight); per run a 100-step smoke at its planned micro-batch with the run's own start (its warm-up, seed and data order, its smoke gate: finite losses, throughput; at its class's lowest grid LR) and a step time, where a flat loss start fails a pruned run (t06, t03, p03, p01, p005) and is a note (`shake_notes` in the queue summary) for a scratch one (t01, t005, bridge); per family (AED; CTC as `*-ctc`) a crash at step 45 and the resume from step 40, a 50-step toy run and its T/2 branch, a complete eval on a subset (T-0.6B, P-0.3B: in-loop, mini and final); every run dir uploaded and verified | ~2 h (3.5 h) | ~$1.5-2 at ~$0.7-0.9/h |
| `A` (Cohere) | 4 | calibration of study-t06/t03/t01/t005 together, LR probes kept-t03 and scratch (one edge extension each at most), `PREREG_numbers_A.json`, then study-t06, -t03, -t01, -t005 in one wave, each followed by its T/2 branch on the same GPU; the anchor re-score in a GPU gap | ~5.8 h (8 h) | ~$12.5 at ~$2.14/h (4x SXM4 40 GB) |
| `B` (Parakeet + bridge) | 4 | calibration of study-p03/p01/p005/bridge together, then of study-t06 as this host's reference (`calib-study-t06-boxB`) under the load of three of them, probes lost, kept-p03 and bridge, `PREREG_numbers_B.json`, the wave of p03/p01/p005/bridge + branches, then the speed probe, one model at a time: its own students (their trained final weights), both teachers, then box A's students from the runs repo (after a wait of at most 45 min for box A) | ~6.3-7 h (9 h) | ~$13.5-15 |
| `replicate` (conditional) | 1 | study-t01-s1235 and its branch, with study-t01's max_steps and LR from box A's numbers; only if the rule below asks for it | ~4.6 h (6 h) | ~$3-4 |

Hours are the central estimate of STUDY.md 5.2-5.3 plus boot, bootstrap (two stores) and the end; the cap is the
watchdog's (`KITSUNE_MAX_HOURS`, plus the extent's rebuild timeout). Traffic adds ~60 GB down and ~3-12 GB up (lean
uploads) at the host's $/GB. launch.py prints the offer, the create command and the cost line; nothing is rented
without `--yes`. Boxes A and B together: ~$26-28 central, ~$40 at both caps (box B's bridge probes are 5,000 steps each; a review of the code's numbers puts box B at ~7 h central, ~9 h high).

### Before a study box (the owner's steps)

1. The label root is sealed (`labels/full/COMPLETE.json`), the study selection is built on the laptop and uploaded,
   and the filled PREREG is committed (see "The size study's selection and pre-registration" above). launch refuses
   a study box (not the shakedown) while `study/PREREG.json` at the commit has a pending field, and whenever the
   selection's sha256 in the data repo is not PREREG's `manifest.selection_sha256`.
2. The students are built on the laptop (`D:/kitsune-students/study/<x>`) and uploaded, only what the boxes read:
   ```bash
   hf upload Multy123/kitsune-data D:/kitsune-students/study students/study --repo-type dataset \
     --exclude "*.pt" --exclude "*/step0/*"
   ```
   launch checks, for every student of the box (and only those), `config.json`, `model.safetensors`, the processor and
   tokenizer files, `student_meta.json`, `README.md`, and for a Parakeet-derived student its CC-BY-4.0
   `MODEL_CARD.md`; and that each is the registered build: `kitsune.prereg.student_problems` on the data repo's
   `student_meta.json` (complete, family, init class, seed, the exact parameter counts, and for a pruned student the
   first run's calibration ids). A stale upload (the B10 T-0.3B, a P student ranked on re-drawn ids) is refused before
   renting; bootstrap.sh checks the pulled copies again (`python -m kitsune.study_queue check-students`) before the
   audio rebuild. (The first run's `students/b20x2560-d4` has no README.md in the data repo: upload
   kitsune.student's model card as its README.md before relaunching a config that trains it.)
3. The configs are generated and committed: `python tools/make_study_configs.py --check` says "up to date" (run it
   without `--check` after a rule changes, e.g. a probe grid, and commit `configs/study/`). Push, and wait for the
   image if `docker/` or `requirements-train.txt` changed.

### Commands (PowerShell on the laptop)

Every launch runs from a checkout at the commit the box will run (`--sha`, the full 40-hex id; the box clones exactly
that commit, and launch reads the configs and `study/PREREG.json` at it). Code-only commits have no image of their own:
`--image-tag main` (launch checks the image was built with this commit's dependencies). Each line first without
`--yes` (read-only: the preflight, the offer table, the create command and the cost line), then the same line with
`--yes` to rent. From `D:\Shizu-ko-distill` (or any checkout of the repo):
```powershell
git fetch origin
git checkout <sha>
```

**1. The shakedown (done: rented from `0a0abae`).** For the record, the line it was launched with:
```powershell
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast/launch.py --job study --box shakedown --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main --sha 0a0abaeb7a11b26e718130b2664562aab975999a --yes
```

**2. Boxes A and B together.** Once the shakedown's `study/box-shakedown/queue_summary.json` says `complete` (and
its `shake_notes` are read), from the commit that carries this README (the study's code), one line after the other,
each with `--yes` once its look-only run shows no problems:
```powershell
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast/launch.py --job study --box A --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main --sha <sha> --yes
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast/launch.py --job study --box B --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main --sha <sha> --yes
```
Both boxes run the same `<sha>` (the same `study/PREREG.json`: each numbers file records its `rules_sha256`, and the
report checks they agree). They do not clash: launch never looks for other live boxes for a study job (each gets its
own instance label `kitsune-study-<box>-...`); on the Hub each box has its own run ids (box B calibrates its reference
as `calib-study-t06-boxB`, box A as `calib-study-t06`), its own `study/box-<box>/` (queue summary, infra logs) and its
own `study/PREREG_numbers_<box>.json`; within a box the mains start 45 s apart (their 20-minute log syncs stay apart),
and every upload backs off on a 429, a 5xx or a commit that raced another writer (409/412) (the trainers' log syncs
15, 60, 180 s, then the next sync; the queue's uploads 5 s doubling to 10 min with up to 25 % jitter, then verified). `tests/test_study_box.py` runs both queues at once against one
rate-limited runs repo. Box B's speed phase times its own students and the teachers first and box A's four students
last, once box A's `study/box-A/queue_summary.json` lists their finished runs; box A is expected to end first (~5.8 h
against ~6.3-7 h central, and box B has three more probes and the CTC frame store). If box A is still training then,
box B polls its summary every 2 min for at most 45 min (~$1.6), never past its watchdog deadline less 1 h
(`speed_wait` event). After that those four speed items fail as readouts (`readouts_failed`) and box B still ends
complete. A redo re-times EVERY system, both teachers and all students, on one host into a fresh speed.json with
`tools/speed_probe.py`: RTF and VRAM compare only within one host, and `tools/study_report.py` flags a speed.json
that mixes hosts.

**3. The replicate, only if it is needed** (CONTRACT.md 8: after boxes A and B, every limit call is computed at
sigma_run 1.6 % and at 3.2 %; the replicate runs only if a family's LIMIT call - the primary delta 10 % walk - differs
between the two, the study report's `replicate_needed`). Box A's numbers must be on the Hub, written under the same
rules; from the same `<sha>`:
```powershell
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast/launch.py --job study --box replicate --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main --sha <sha> --yes
```
Box B's speed probe skips the replicate with a note (`speed_skipped`, `conditional: true`; it has study-t01's shape)
unless its final weights are already in the runs repo; time it afterwards from the Hub with `tools/speed_probe.py` if
it ran.

The search is relaxed per launch: `num_gpus` 4 for A and B, 1 for the replicate and the shakedown; any A100, SXM4 or
PCIe, 40 GB first, then 80 GB (equal compute is measured on each box's own host, so the variant does not bias
max_steps); verified, reliability >= 0.93 (`STUDY_RELIABILITY`: the strict 0.98 left no 4x offer), driver CUDA >= 13.0,
>= 12 cores and >= 64 GB RAM per GPU, disk and network >= 500 MB/s / Mbit/s down, >= 100 Mbit/s up per GPU. The disk
comes from the extent's sizing with both label stores (the Cohere and the Parakeet store each copy the selected audio)
and the box's checkpoints (`kitsune.study_queue.study_extra_gb`: up to 6 full states per run at 16 bytes a parameter,
the bf16 exports, the probes running at once), about 450 GB for box A. `--offer-id`, `--max-dph` (default 6 $/h for 4
GPUs, 2 for 1), `--disk-gb` and `--max-hours` work as for a training run. launch refuses the replicate while box A's
numbers are not there or were written under other rules than the commit's `study/PREREG.json`. It passes the rented
GPU count (`KITSUNE_N_GPUS`): the queue refuses to run on another count, and a box that trains a wave needs a GPU per
run. The Cohere teacher's speed item (box B) reads the gated model with the box's own `HF_TOKEN` (the vast account
env, One-time setup 3.2: the token needs read access to `CohereLabs/cohere-transcribe-03-2026`); without it that one
readout fails and is recorded, the box still ends complete.

### Numbers files are write-once; a relaunch reuses them

A box writes `study/PREREG_numbers_<box>.json` once, uploads it, verifies it and logs its sha256 (`prereg_numbers`)
before its first study step, and puts its queue summary up right after it. A box that dies after that is relaunched
with the same launch line (a new instance, a fresh `queue.json`): launch sees the numbers file and says `relaunch: ...
box <box> REUSES it`; the queue downloads it, checks its `rules_sha256` against the checkout's `study/PREREG.json`
and the box, and then skips its calibration and probes entirely (never a second measurement, never an overwrite; the
`prereg_numbers_reused` and `prereg_numbers` events). The loader retry's `perf.num_workers` and the smoke gate guard's
decisions come from the earlier box's `study/box-<box>/queue_summary.json`. A numbers file written under other rules
(another commit's PREREG) refuses the launch, and halts a box that gets that far: relaunch from the commit the numbers
were written at. The wave itself starts again from step 0 (new run ids; the earlier box's finished run dirs stay in
the runs repo).

### Watching it (per box)

| what | where |
|---|---|
| the box's state and outcome | runs repo `study/box-<box>/queue_summary.json` (after the numbers, then at the end: `status`, `reason`, every item's status and run dir, `calibration`, `lr_choice`, `numbers`, `smoke_gate_off`, `readouts_failed`); on the box `/workspace/kitsune_state/queue.json` and `queue_summary.json` |
| the numbers | runs repo `study/PREREG_numbers_<box>.json`; its sha256 is the `prereg_numbers` event in `kitsune_state/events.jsonl`, before the first study step's `item_start` |
| logs | on the box `tail -f /workspace/kitsune.log` (bootstrap, then `[queue] item_start`, `item_end`, `calibration_release`, `calibration`, `lr_chosen` / `lr_probe_extend`, `prereg_numbers`, `smoke_gate_off`, `item_uploaded`, `queue_end`) and `/workspace/kitsune_state/logs/<item>.log` per trainer; at the end in the runs repo `study/box-<box>/infra/<container>/` (the infra logs, `logs/`, any `rearm-<stamp>/`) |
| the runs | runs repo `runs/<run>-<stamp>/` (each synced every 20 min while it trains, then uploaded lean and verified as it finishes) |
| TensorBoard | `vastai ssh-url <id>`, then `ssh -p <port> root@<ip> -L 6006:localhost:6006` and http://localhost:6006: every run of the box (with A and B live, forward box B's to another local port, e.g. `-L 6007:localhost:6006`) |

The calibration of a group starts together: each trainer sets up, writes `READY` in its run dir and waits; once all are
there the queue writes `GO` into every one (`calibration_release`), so each run's steps 50-250 - the pre-registered
window, literally - fall while the whole group trains, and the queue stops the group once all have logged step 250.
Every trainer of a run (its calibration, probes, main and branch) gets the same loader: the queue gives each trainer an
equal share of half the host's `/dev/shm` (the `shm` event) and cuts `perf.prefetch`, then `perf.num_workers`, to fit
it. A main whose calibration run (its own start at its class's lowest grid LR) showed no falling loss over the smoke
steps runs without the loss-trend check (`smoke_gate_off`, main and branch alike; a scratch run is ~5 % into its
warm-up at step 100); the replicate follows box A's decision for study-t01.

### How it ends, how to stop it, and what to do on a halt

| outcome | instance | what to do |
|---|---|---|
| the queue exits 0: every item done and verified on the Hub | `finish.py --destroy` (lean): **destroyed** once verified | nothing; check the vast console after the destroy |
| complete with `readouts_failed` (a failed anchor re-score or speed probe, e.g. no `HF_TOKEN` for the Cohere teacher) | **destroyed** like a complete box: a readout is never a reason to keep a paid box | redo that readout from the Hub (the run dirs and weights are there) |
| a pre-registered halt (exit 4): calibration still loader-bound after `perf.num_workers` 12; an LR winner at a grid edge after its one extension; numbers on the Hub written under other rules | **stopped**; `reason` in `queue_summary.json` and the supervisor's decision | an owner decision: read `calibration` / `lr_choice` in the summary. The box has written no numbers (the numbers halt aside), so `vastai destroy instance <id>` and relaunch after the decision (a rule change is a new PREREG commit); a numbers-rules halt: relaunch from the commit the numbers were written at |
| a trainer's throughput floor (exit 3) | **stopped** | a slow host: destroy and relaunch the same line (a new host; numbers already up are reused) |
| the queue crashes (or the container restarts) | the queue again, up to 2 times: it resumes from `queue.json` (finished items skipped, a main run or probe from its local full state, a branch again from its parent, a calibration group again as a whole); then **stopped** | look at `kitsune_state/logs/`; continue in place (below) or destroy and relaunch |
| the cap | **stopped** by the watchdog | continue in place (below) with a longer cap, or destroy and relaunch |

Inside the queue a run that crashes resumes once from its own full state; a second failure marks it failed, the
others go on, and the box ends stopped. A run whose smoke checks fail (`SmokeFailed`) fails at once: the same start
fails the same way. The shakedown's smokes carry the study runs' own gate, so such a failure shows there first.

To stop a box by hand: `vastai stop instance <id>` (disk kept, billed) or `vastai destroy instance <id>` (gone; what
was uploaded stays in the runs repo). With A and B live, check the id against the label (`kitsune-study-A-...` /
`kitsune-study-B-...`) before either. A stopped box continues in place with `vastai start instance <id>` and
`bash /workspace/Kitsune-Transcribe/vast/onstart.sh --rearm`: the rearm keeps `queue.json`, so the queue resumes and
never writes its numbers twice (delete `queue.json` only to start that box's study over, and never after its numbers
are on the Hub: a relaunch reuses them anyway). Do not `touch runs/<run_id>/STOP` in a study run: it ends the run
before its max_steps, which the pre-registration counts as an invalid run.

### The trainer keys the study configs use (scripts/04_distill.py)

| key | default (today's trainer) | study |
|---|---|---|
| `bn.mode` | `frozen` | `train` for the students built from scratch (bridge included): BN trains, evals use its running stats |
| `loss.aux_ctc_weight` | 0 | 0.3 for the scratch runs: a training-only CTC head on the encoder, never exported |
| `optim.weight_decay` | 0 | 1e-3 for the scratch runs, on the parameters of 2+ dimensions only |
| `schedule.clock` / `max_steps` | wall / null | `steps`; max_steps from the box's numbers (the configs hold null, refused until filled) |
| `eval.full_at_fracs` | null | [0.2, 0.4, 0.6, 0.8]: complete evals at those fractions of max_steps (eval.every_min null) |
| `ckpt.full_at_fracs` / `weights_at_fracs` / `upload_full_at "frac:<f>"` | null / null / pre_cooldown, end | [0.4] (+0.8 for the runs of at most 0.1B, uploaded) / [0.4] / [] or ["frac:0.8"] |
| `branch` | parent null | `<run>-half`: parent = the main run's local run dir (set by the queue), resume 0.4, end 0.5 |
| `lr_probe.enabled` | false | the probes: metrics only, the teacher-forced objective on the complete gate sets at the end |
| `specaug.seed` | null (the run's seed) | null: masks from (seed, step, micro-batch), identical across a resume or a branch |
| `calibrate` | enabled false, window [50, 250], barrier false | the calibration runs: an lr_probe run that ends (at max_steps or on the STOP file the queue writes once every run of the group has its window) with its step-time table: a `calibrate_result` event and summary.json's `calibrate`; `barrier` (the queue sets it): `READY`, then wait for `GO` or `STOP` before the first step |
| `pull_parakeet` | false | true (study/data.json): the box pulls both label roots |
| `selection_recipe.study` | null | the pre-registered selection block (study/data.json) |

The CTC trainer's keys (`family`, `parakeet_root`, `loss.w_ctc`) come with the CTC trainer (WP4b); box B needs it, and
its store builder, before it can run.

## Full-data runs (the full boxes)

Plan v3 trains the full students on the whole label set: P-0.1B alone on **box 1** (`p01`, 1x RTX 5090, Parakeet labels
only; done 2026-10-02, 4 epochs, M4 11.45 %), after two short smokes (`full-smoke` = smoke A, `smoke-b`). Box 2 is now
two 1x RTX 5090 boxes (DECISIONS G2): **box T** (`full-t`: T-0.6B, the Cohere labels only, `data-t`) and **box P**
(`full-p`: P-0.3B then P-0.05B, the Parakeet labels, `data-p`, with the Whisper models and their quantised readouts);
T-0.6B and P-0.3B train 3 epochs, P-0.05B 5 (DECISIONS G1, confirmed by the owner on 2026-10-02; the earlier 10-epoch
request was withdrawn), each with the common early-stop patience 6. Box `p01` runs again as **P-0.1B's
continuation** to 8 epochs (DECISIONS G3, below) with P-0.1B's 7 quantised readouts. The 2x box `full` is retired: its
name stays in `fullrun.BOX_NAMES` for the tests' fixtures only, it is not in the registry, and launch refuses it. Each is one `launch.py --job full --box <box>` call; on the box
`vast/supervise.py` runs `kitsune/full_queue.py` for that box. The box registry `configs/full/boxes.json`
(`kitsune/fullrun.py`) is the one source of each box's GPU count, data config, hours (`est_hours` planned, `max_hours`
the watchdog cap), price cap (`max_dph`), extra disk, watchdog and items; launch reads it at the commit the box runs
and keeps no table of its own (`python -m kitsune.fullrun show --box <box>` prints a box's spec and env).

### Before a full box (the owner's steps)

1. **The scratch repo** (the trainers' timed full states; each run keeps exactly one there and replaces it every
   cycle): create it once as a PRIVATE model repo, `hf repos create Multy123/kitsune-scratch --private`, and give the
   vast `HF_TOKEN` (fine-grained) read and write access to it, as to `Multy123/kitsune-runs`. The code never creates
   it: launch refuses a missing or public one (with the laptop login), and the box's plan refuses a token that cannot
   read and write it (exit 3, before anything is pulled). Never pass the runs or the data repo as `--scratch-repo`
   (the trainers delete and squash the scratch repo's history); launch refuses that too.
2. The full selection and its sidecar are uploaded (`labels/full/selections/full_study/`), and the configs and the
   registry are committed (`python tools/make_full_configs.py --check` says "up to date").
3. The image is built for the commit (`--image-tag main` after a merge; launch checks the build commit's pins).

### Commands (PowerShell, from the launch clone at origin/main)

```powershell
Set-Location D:\kitsune-launch; git fetch origin; git checkout --detach origin/main
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box full-smoke --machine <id> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box p01 --machine <smoke A's id> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box smoke-b --machine <id> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box p01 --resume-reset full-p01-20261001T184145Z --resume-set full-p01-20261001T184145Z:schedule.epochs=8 --resume-set full-p01-20261001T184145Z:early_stop.patience=12 --machine <id> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box full-t --machine <id> --avoid-machine <p01's id> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box full-p --machine <id> --avoid-machine <p01's id> --avoid-machine <full-t's id> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
```
Each line first as it is (look-only: the preflight, the offer table, the create command and the cost line), then the
same line with `--yes` to rent. Without `--machine` the offers are ranked by the estimated total (the planned hours at
the offer's $/h with the disk, plus the traffic at its $/GB); `--offer-id` picks another row.

**The order after box 1** (DECISIONS F and G, 2026-10-02): the box-2 preparation merged with a green image build, then
a standalone **smoke-B** (`--box smoke-b`, ~1.5 h on a 1x 5090, report only: it destroys itself on exit 0 whatever its
checks say), its verdict `full/box-smoke-b/smoke_verdict.json` passing checks 12-16 (`python tools/box1_go.py
--revision c4604304db76e068df7bbe39d00d006b74d6c134`: G5), this layout merged, then **box p01's continuation first**
and **boxes full-t and full-p** right after it (they no longer depend on its timing: box p01 scores its own quantised
readouts). The hours of boxes p01, full-t and full-p (`est_hours`, `max_hours` and the train items' no-start needs)
come from the speed record `configs/full/plan/box2_hours.json` (smoke A's measured s/step and box 1's, `box1_go.py
--revision <its 4-epoch record> --json` then `make_full_configs.py --import-speed --box1-go`) at the step counts of
the plan record's launch part (`--import-launch-plan`), so their lines take no `--max-hours`; launch warns while the
record has no box-1 part. launch refuses a box with quantised items (full-p 14, full-t 7, p01 7) without **the quant go
signal**: smoke-b's verdict on the Hub passed overall and every one of checks 12-16, at a commit that is an ancestor of
the one the box runs, with `kitsune/quant.py`, `tools/speed_probe.py`, `scripts/05_evaluate.py`, `kitsune/whisper.py`,
`tools/whisper_eval.py`, `requirements-train.txt` and `docker/Dockerfile` unchanged since (`QUANT_CODE`): a later
change to any of them needs another smoke-B, or the owner's `--allow-unverified-quant` (the refusal becomes a
warning). Offers listed in mainland China are dropped (the Hub is not reachable from there; `FULL_AVOID_COUNTRIES`).

What launch checks and decides, before anything is rented:
- **Offers:** `--tier 5090` (default) or `a100` (option C, default cap $2.60/h). Per GPU: `cpu_cores_effective >= 16`,
  `cpu_ram >= 60` in the query and 64,000 MB on the client, `inet_up >= 100`; reliability >= 0.98, driver CUDA >= 13.0,
  disk_bw and inet_down >= 500, a direct port, the disk from the extent's sizing plus the registry's `extra_gb`. A box
  with `min_ram_gb` (boxes p01, full-t, full-p: 96) asks the query for `cpu_ram >= floor(0.94 x it)` (90) and the
  client for `round(0.97 x it x 1000)` MB (93,120: vast lists 96 GB machines at 95,758-96,709 MB). Box 1's real peaks
  were 33 GiB (trainer) and 37 GiB (store build); its queue summary's 164 GB `peak_rss_gb` is VmRSS summed over the
  trainer and its 8 DataLoader workers, the stores' page cache counted once per process, and reads ~150+ on any large
  host: judge memory by the trainer's `sys/cgroup/anon_gb` (<= ~40) and `sys/cgroup/oom_kill` (0).
  `verified=any` in the query, then the client keeps verified and deverified hosts only (never unverified), and only a
  host whose max rental is at least max(`MIN_RENTAL_DAYS` 4, the cap / 24 + 0.5) days (the caps now, 21-37 h, keep the 4 d; a 104 h cap would need 4.83 d). The registry's `max_dph` drops dearer offers before
  the ranking. `--gpus` may only repeat the registry's count; `--config` only its data config.
- **Avoided machines:** `vast/blocklist.json` (every job, for good: 151760, study box A #1's 2.9 MB/s host;
  54650, which never started p01-chain's instance on 2026-10-01), the
  label runs' failed hosts, and a machine whose full box's download gate said slow in the last 30 days
  (`full/box-*/infra/*/download_gate.json` in the runs repo). A `--machine` on that list is refused.
- **full_preflight:** every config the box reads committed at the commit; every tool its items run there (the queue,
  the trainer, 05, speed_probe with the items' `--kind`s and every `--flag` in their `args`, each eval item's `-m`
  module or script: a box whose items need a CLI or a flag not merged yet is refused here, not by argparse on the
  rented box); its students present and the registered builds
  (`kitsune.prereg.student_problems`); its extra files and dirs; the scratch repo private; the full selection's sidecar
  (`kitsune.devslice.sidecar_problems`) against the Hub's selection and frozen manifest; plus the selection and extent
  checks every extent config gets. `--no-hf-check` is refused.
- **Env** (all from the registry and the checks): `KITSUNE_JOB=full`, `KITSUNE_BOX`, `KITSUNE_CONFIG` (the data
  config), `KITSUNE_N_GPUS`, `KITSUNE_WATCHDOG_HB_FILE=train_hb`, `KITSUNE_WATCHDOG_ORPHAN_S` and `_ORPHAN_ACTION`,
  `KITSUNE_MAX_HOURS` (the registry's cap, no rebuild add-on), `KITSUNE_SCRATCH_REPO` (boxes with timed states),
  `KITSUNE_MACHINE_ID`, the gate's `KITSUNE_GATE_BYTES` (max(the extent's upstream bytes, 571.2 GB)),
  `KITSUNE_GATE_MAX_H` (`--gate-hours`, 5; 0 turns the gate off, and a box whose registry says `gate: false`, smoke-b,
  has none), `KITSUNE_REBUILD_BYTES`, `KITSUNE_PULL_BYTES`, and with `--resume` the `KITSUNE_RESUME*` values below.

### On the box

bootstrap.sh (job full) runs, in order: **download_gate** (`python -m kitsune.netgate`: three pinned upstream files,
~2.5 GB, downloaded with 01's own call but one at a time, the conservative floor; a host that could not pull 571.2 GB in `KITSUNE_GATE_MAX_H` hours, 31.7 MB/s at
5 h, is refused with exit 3 after at most ~3 minutes, and a pass sets the label pull's and the rebuild's per-attempt
timeouts from the measured rate, the rebuild's never below launch's 40 MB/s sizing; hf_xet's chunk cache is off and
its cache dir the gate's own, so a retried gate times the link again, never the disk), **plan** (also the scratch repo's
token check), **pull_derived** (everything but the labels: the meta files, the selection, the registry's students,
extra files and dirs), **resume_pull** (`--resume` only), **check_students** (`python -m kitsune.fullrun
check-students`), **pull_labels** in the background while **rebuild_audio** (01 `--extent-config`) runs, then
**pull_labels_wait** (a failed label pull fails here) and **coverage**. The rebuild downloads up to
`KITSUNE_DOWNLOAD_AHEAD` (default 6, 0..16; optional, from the box's env) upstream files while its single-threaded
ingest reads them in order (decision F3, `kitsune/fetchahead.py`): the output is byte-identical to one at a time,
transient Hub errors retry in-process, and each step logs `downloads: T of P planned files (G GB) in W s, up to K
ahead; the ingest waited S s`, so W - S is the ingest's own time. Box 1 (one at a time) took 3.1 h for 571 GB with a
128.9 MB/s gate; expect ~1-1.3 h on a similar host. On box 1 (CTC, no `pull_parakeet`) the train
stems have Parakeet labels only: the extent and coverage checks read their ids from `parakeet_out`
(`kitsune.extent.label_root_for`, fix 9). Every phase keeps `$KITSUNE_STATE/train_hb` fresh while it runs, for at
most its own worst case: the label pull, pull_labels_wait and the rebuild for their three attempts' timeouts (with the
gate's floor rate the full extent's rebuild may take ~24 h), every other phase for `KITSUNE_PHASE_HB_MAX_S` (12 h),
and no toucher outlives bootstrap: a bootstrap that fails during the rebuild also stops the background label pull
(its timeout and python) and its toucher, and a killed toucher takes its `sleep 60` with it. Bootstrap closes fd 7
(onstart's `supervise.lock`) before anything else and its EXIT trap acts in bootstrap's own process only: on
2026-10-01 a toucher's leftover sleep kept the lock past bootstrap (supervise.py refused to start and the box idled),
and a toucher killed right after its fork ran the inherited EXIT trap and deleted the helper before the coverage check
(Linux bash runs a parent's EXIT trap in a child that SIGTERM reaches before it has reset its traps). supervise.py
also waits up to 3 minutes for a held lock before it steps aside.

The watchdog reads `train_hb` (bootstrap's phases, the queue's poll and the supervisor's bounded finish calls touch
it; each trainer, 05 and the store builds beat their own `hb/<item>`, which the queue's stall check reads). On boxes p01,
full-t and full-p (`action: stop`) a heartbeat stale for `orphan_s` means a dead controller: the watchdog syncs and stops the box
(`watchdog: box controller heartbeat stale`). The smoke box (`action: alert`) freezes it on purpose in its fault test:
there the watchdog only appends `{"wall", "kind": "orphan_alert", "hb", "age_s", "limit_s"}` to
`$KITSUNE_STATE/watchdog_alerts.jsonl` and re-arms once the file is fresh again.

### Watching it

| what | where |
|---|---|
| the box's state | runs repo `full/box-<box>/queue_summary.json` (put as each item starts and ends, and as a run dir appears: the source of truth of a resume); the smokes' `full/box-<box>/smoke_verdict.json` |
| the download gate | `/workspace/kitsune_state/download_gate.json` (verdict, the three samples' MB/s, the timeouts), later in the infra folder |
| logs | on the box `tail -f /workspace/kitsune.log` and `/workspace/kitsune_state/logs/<item>.log`; at the end in the runs repo `full/box-<box>/infra/<container>/` (queue.json, events.jsonl, the per-item logs, the gate, the resume plan, the watchdog's alerts) |
| the runs | runs repo `runs/<run_id>/` (lean: logs, weights, the uploaded pre_cooldown state); the timed states in the scratch repo `runs/<run_id>/checkpoints/full_step_<N>/` with `runs/<run_id>/timed_state.json` |

### How it ends

| outcome | instance |
|---|---|
| the queue exits 0 (every non-droppable training item done, everything verified) | **destroyed** after finish.py's lean verification |
| a training item failed for good (queue exit 4) | **stopped**, disk kept: the owner decides |
| the download gate refuses the host, or any other bootstrap failure before a run dir exists | `finish.py --abort`: the infra goes up, then the box is **destroyed** (nothing on its disk is unique); a slow gate names the reason, and launch avoids the machine for 30 days |
| a bootstrap failure after resume_pull pulled run dirs | **stopped** (`--abort` with a run dir) |
| `train_hb` stale (boxes p01, full-t, full-p) | synced and **stopped** by the watchdog |
| the cap (`max_hours`) | **stopped** by the watchdog |

finish.py for a full box is lean (as the study box's): the logs, every weights dir and the full states the config
uploads, never the timed states (their `.scratch_pending` marker is never uploaded nor expected).

### Resume on a new host

A dead or stopped box continues elsewhere from what the Hub has:
```powershell
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box <box> --resume [--resume-reset <run_id>] [--resume-set <run_id>:<key>=<int>] [--allow-done-trains] --machine <new id> --max-hours <left + setup> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
```
- `--resume` (`KITSUNE_RESUME=1`): bootstrap's resume_pull (`python -m kitsune.full_queue resume-pull`) reads the
  box's Hub queue summary, pulls every started run with its newest full state (the scratch pointer's or the runs
  repo's) and marks the finished ones done; the queue adopts that plan.
- `--resume-reset <run_id>` (repeatable; `KITSUNE_RESUME_RESET`): continue a run from its pre_cooldown state (an
  early-stopped one, or one whose schedule ended: a continuation). `--resume-set <run_id>:<key>=<int>` (repeatable;
  `KITSUNE_RESUME_SETS`): resume with another `schedule.epochs` or `early_stop.patience` (`fullrun.RESUME_SET_KEYS`,
  each an int >= 1, once per run; nothing else may change). Both imply `--resume`.
- **P-0.1B's continuation** (DECISIONS G3): `--box p01 --resume-reset full-p01-20261001T184145Z --resume-set
  full-p01-20261001T184145Z:schedule.epochs=8 --resume-set full-p01-20261001T184145Z:early_stop.patience=12` (the
  command above; `make_full_configs.py --import-speed` prints it). resume_pull takes the pre_cooldown state
  `checkpoints/full_step_86328`, the queue passes the three `--set`s until the trainer has logged its resume_reset and
  a full state after it (then every state carries epochs 8 and patience 12), the trainer re-plans T = 215,815, the
  readout writes `runs/m4-full-p01-20261001T184145Z-r1` and the 7 quantised readouts read the new final step. It
  overwrites box p01's records at the runs repo's head: box 1's 4-epoch record stays citable as revision
  c4604304db76e068df7bbe39d00d006b74d6c134 (`box1_go.py --revision`, `full_report.py --earlier
  full-p01-e4=<pull>/runs/m4-full-p01-20261001T184145Z@c4604304db76e068df7bbe39d00d006b74d6c134`).
- launch refuses a **plain `--resume` of a box whose Hub summary has every train item done** (box p01 before its
  continuation started: it would only adopt box 1's run and score the new quantised items on the 4-epoch weights);
  `--allow-done-trains` lets it through for a box lost in its eval pool (box P after its trainings). A continuation
  lost on its way resumes with a plain `--resume` (its summary shows the run running with its continuation record);
  one lost before its first summary put: launch it again with the same reset/set flags. A `--resume-set` of a done run
  without `--resume-reset` is refused too (resume-pull would refuse it after the paid boot).
- launch refuses a box whose Hub summary is missing and a reset/set id that no train item of the box ran, prints what
  will resume with the newest state step it can see (scratch and runs repo), and warns about a live instance with
  the box's own label prefix `kitsune-full-<box>-<data config stem>-` (not `kitsune-full-<box>*`, which for box `full`
  would also match the smoke box's `kitsune-full-full-smoke-...`): destroy the old box first, two boxes must never
  write the same run dirs.

### The chained box: smoke A, smoke B and box 1 on one rental (`p01-chain`)

Owner decision D (2026-09-30; contract addendum E): smoke A, smoke B and box 1 run on ONE 1x RTX 5090, with an
automatic gate between them. `p01-chain` is a registry entry that names the boxes `full-smoke`, `smoke-b` and `p01` as
its **parts**, in two stages; it has no items of its own. `python -m kitsune.full_queue run --box p01-chain` (what
supervise.py runs) is its controller (`ChainController`), which runs the parts one at a time, each an unchanged box
queue with its own state dir `$KITSUNE_STATE/chain/<part>/`, writing its summary and verdict at the standalone box's
Hub paths (`full/box-<part>/...`): the report and a box-1 resume read them as they are. The standalone boxes stay
defined as fallbacks. The chain ran once (2026-10-01, its gate failed; box 1 then ran as box p01 alone) and stays as
history: its part p01 now carries quantised readouts, so a relaunch would also need smoke-B #2's quant go signal.

### Speed comparability (the report's chart)

Every comparative speed number comes from one host: smoke-B #2's (`tools/full_report.py` takes the machine group with
the most systems as its chart group). smoke-B times the 4 study students in 6 formats, the compile probes, the 4
Whisper models and, when it lands on another machine than smoke A, the bf16 study students and the 3 teachers again.
The full-data P rows use their study weights' record (S1: same shape and code; CTC greedy decoding does not depend on
the weights), so the 4- and 8-epoch P-0.1B share study-p01's. Boxes full-p and p01 run no speed item. Box full-t keeps
decision 22's bf16 re-time pair (speed-full-t06 and speed-study-t06 in one speed.json): when full-t06's tokens on the
200 speed ids drift more than 5 % from study-t06's, S2 scales the chart group's study-t06 record (and its variants) by
that pair's ratio; the pair's own group (2 systems) never becomes the chart group.

```powershell
& C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe vast\launch.py --job full --box p01-chain --machine <id> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
```
Look-only first, then the same line with `--yes`. Launch re-checks the machine's free disk every time it runs: its
offer search keeps only offers with `disk_space` >= the chain's disk (box 1's, ~1,450 GB), so a pinned `--machine`
that no longer has it is refused. The disk and the download gate are sized on box 1's extent (with the chain's
`extra_gb` 120: p01's own 25 GB and ~95 GB of stage-1 leftovers), the boot's rebuild on the smoke (study) extent. Launch
refuses `--config`, `--gate-hours 0`, `--max-hours` below 30 (stage 1's 10.5 h + box 1's 19.5 h; a warning below the
chain's 35 h) and every resume flag, and prints the chain's deadlines: the gate part by first boot + 9 h, stage 1 by
+ 10.5 h, the cap.

What happens on the box:

| step | what | watchdog |
|---|---|---|
| boot | bootstrap on the study extent (`KITSUNE_CHAIN_STAGE=1`: the stage-1 view pulls both parts' students, extra files and both selections) | alert 600 (stage 1's env) |
| smoke A | the `full-smoke` part until `gate_by` (first boot + 9 h: its deadline and stop time; its items' `KITSUNE_DEADLINE` 20 min before) | alert 600: F5's frozen heartbeat must be alerted, never stopped |
| mode switch | the controller writes `$KITSUNE_STATE/watchdog_mode` = `stop 3600` as soon as smoke A returns, before the gate | stop 3600 from here on |
| the gate | automatic: smoke A's verdict checks 1-11 all true, the part's rc 0, the verdict its own and of this commit; recorded once (event `chain_gate`, `chain/chain.json`, the chain summary) | |
| smoke B | the `smoke-b` part until the stage-1 sub-deadline (first boot + 10.5 h), pass or fail: report only (checks 12-16 never gate); one retry, then recorded as failed | |
| gate failed | exit 5: finish verifies the verdict, summaries, logs and events on the Hub, then **destroys** (a failed verification stops) | |
| stage 2 | stage 1's bootstrap records to `chain/stage1/`; box 1 must still fit (else exit 5, destroy); the stage-2 bootstrap (`KITSUNE_CHAIN_STAGE=2`, the full CTC-only extent on the same data root: 01 reuses the study extent's shards, coverage checks every stem) runs as the controller's child, bounded by its own budget and by box 1's fit; rc 2/3 are not retried, anything else once; a failure or its bound exits 5 (destroy: nothing unique is on the disk) | |
| box 1 | the `p01` part (stores-ctc, full-p01, m4-full-p01): its rc is the chain's (0 destroy, 4 stop with the disk kept, 1 a restart that resumes it) | |

- **Heartbeat:** the controller beats `train_hb` only between steps: never while a part runs (its queue beats, and
  F5's freeze must hold) nor while the stage-2 bootstrap runs (its phases' bounded touchers keep it fresh, 3 h at most
  for a phase without a timeout, so a hung phase goes stale and the watchdog stops the box).
- **A stage 1 that never hands over:** with no mode file by first boot + 9.5 h (`KITSUNE_WATCHDOG_HANDOVER_S`), the
  watchdog syncs and **stops** the box, whatever the heartbeat says (a hung controller may hold files the owner reads).
- **A halt that did not take:** a halt marker written during the container's life, 20 min old with the instance still
  up (a failed stop or destroy REST call): the watchdog requests the stop again every 5 min.
- **Restarts:** the supervisor restarts the controller at most twice per stage; `chain/chain.json` records every step,
  so ended parts are not run again and the gate is never evaluated twice; a stage-2 bootstrap left running is killed
  when `/proc` shows it is that process (its start time and command line), else ignored, and started again. A kill of
  the stage-2 bootstrap (at its bound, on a restart, or when the controller fails) takes its whole session: GNU
  `timeout` moves each wrapped phase (01's rebuild among them) into a process group of its own, but never out of the
  session.
- **Watching it:** the chain summary `full/box-p01-chain/queue_summary.json` (stage, step, gate, each part's status,
  both verdicts' checks); each part's own summary and verdict at `full/box-<part>/`; on the box
  `$KITSUNE_STATE/chain/<part>/logs/` and `$KITSUNE_STATE/logs/bootstrap-s2.log`; at the end the infra folder
  `full/box-p01-chain/infra/<container>/` with `chain/` (chain.json, stage1/, every part's records).
- **A chain is not resumed as a chain** (`launch --box p01-chain --resume` exits 1, `resume-pull` exits 3, both say
  what to run): dead in stage 1 (the gate null or failed) -> the chain again, fresh; dead in stage 2 after p01 started
  on that rental -> `--box p01 --resume ...` (stage 2's part is box p01: its Hub summary, timed states and run ids are
  box 1's, and launch checks that the Hub's p01 summary is that rental's); dead during the stage-2 bootstrap -> `--box
  p01` fresh or the chain fresh (the owner's call: the gate passed on the old machine only).
- **Cost** (addendum E.9.5, m54650's rates): ~25.2 h central, ~$21; the 35 h cap ~$29.3; a failed gate ~6.6 h, ~$5.5.
