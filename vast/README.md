# Running the viability run on vast.ai

One A100 for ~4.5 h: the box boots our image, clones this repo at a pinned commit, pulls the small derived data from a
private HF dataset, rebuilds the audio from the public upstream datasets, trains for 4 h, uploads everything to a
private HF model repo, verifies the upload and destroys itself. A watchdog stops it at 5.5 h whatever happens.
Nothing is rented until you run `launch.py --yes`.

The same scripts also rent the **label box**: one RTX 5090 that labels the full download with both teachers and uploads
the labels to the data repo (`--job label`, see "The label run" below), and the **size study's boxes**: 4x or 1x A100
that run a queue of calibration, LR probes and study runs (`--job study --box ...`, see "Size study" at the end).

| file | runs where | what it does |
|---|---|---|
| `launch.py` | laptop | search offers with the strict filter, print the exact `vastai create instance` command, create only with `--yes` |
| `onstart_stub.sh` | box, every container start | what `--onstart` sends (kept under 4 KB): clone at `KITSUNE_SHA`, then run `onstart.sh`; stops the box if the clone fails |
| `onstart.sh` | box, from the stub | env sync, limits, TensorBoard + vast portal, start watchdog, then `bootstrap.sh` + `supervise.py` |
| `bootstrap.sh` | box | pull derived data from `KITSUNE_DATA_REPO`, rebuild audio with `scripts/01_prepare_data.py` (pinned upstream commits), check the id join |
| `supervise.py` | box | run `scripts/04_distill.py`; resume once after a late crash; then `finish.py --destroy` or `--stop` |
| `finish.py` | box | upload, verify every file in the HF output repo (path, size, hash), destroy; stop instead if anything is off |
| `watchdog.sh` | box | stop the instance 5.5 h after first boot (log sync 10 min before) |

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
   upstream datasets are public. Bootstrap checks this token against both repos in its first minute on the box.

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
| bootstrap or on-start failure | **stopped** |
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

The size study (study/STUDY.md; CONTRACT.md section 6) runs on four boxes, in this order. Each is one
`launch.py --job study --box <box>` call; on the box `vast/supervise.py` runs `kitsune/study_queue.py` for that box
instead of one trainer. The box's plan (runs, probe classes, calibrated runs, reference run, numbers file) is
`kitsune.prereg.rules()["boxes"][<box>]`; the rules, the probe grids and every number rule come from `kitsune/prereg.py`.

| box | GPUs | what it runs | central hours (cap) | cost at the 2026-09-25 snapshot |
|---|---|---|---|---|
| `shakedown` | 1 | the stores of the real extent; per box-A run a 100-step smoke at its planned micro-batch with the run's own start (its warm-up, seed and data order, its smoke gate: finite and falling losses, throughput; at its class's lowest grid LR) and a step time; a crash at step 45 and the resume from step 40; a 50-step toy run and its T/2 branch; a complete eval on a subset with the 0.6B (in-loop, mini and final); every run dir uploaded and verified | ~1.6 h (3 h) | ~$1-1.5 at ~$0.6-0.8/h |
| `A` (Cohere) | 4 | calibration of study-t06/t03/t01/t005 together, LR probes kept-t03 and scratch (one edge extension each at most), `PREREG_numbers_A.json`, then study-t06, -t03, -t01, -t005 in one wave, each followed by its T/2 branch on the same GPU; the anchor re-score in a GPU gap | ~5.8 h (8 h) | ~$12.5 at ~$2.14/h (4x SXM4 40 GB) |
| `replicate` | 1 | study-t01-s1235 and its branch, with study-t01's max_steps and LR from box A's numbers (after A) | ~4.6 h (6 h) | ~$3-4 |
| `B` (Parakeet + bridge) | 4 | calibration of study-p03/p01/p005/bridge together, then of study-t06 as this host's reference under the load of three of them, probes lost, kept-p03 and bridge, `PREREG_numbers_B.json`, the wave of p03/p01/p005/bridge + branches, then the speed probe of all 9 students (their trained final weights: box A's and the replicate's from the runs repo) and both teachers, one model at a time | ~6.3 h (8.5 h) | ~$13.5 |

Hours are the central estimate of STUDY.md 5.2-5.3 plus boot, bootstrap (two stores) and the end; the cap is the
watchdog's (`KITSUNE_MAX_HOURS`, plus the extent's rebuild timeout). Traffic adds ~60 GB down and ~3-12 GB up (lean
uploads) at the host's $/GB. launch.py prints the offer, the create command and the cost line; nothing is rented
without `--yes`.

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
   `MODEL_CARD.md`. (The first run's `students/b20x2560-d4` has no README.md in the data repo: upload
   kitsune.student's model card as its README.md before relaunching a config that trains it.)
3. The configs are generated and committed: `python tools/make_study_configs.py --check` says "up to date" (run it
   without `--check` after a rule changes, e.g. a probe grid, and commit `configs/study/`). Push, and wait for the
   image if `docker/` or `requirements-train.txt` changed.

### Commands

Look first (read-only, spends nothing), then rent with `--yes`:
```bash
python vast/launch.py --job study --box shakedown --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main
python vast/launch.py --job study --box A         --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main
python vast/launch.py --job study --box replicate --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main
python vast/launch.py --job study --box B         --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --image-tag main
```
The search is relaxed per launch: `num_gpus` 4 for A and B, 1 for the replicate and the shakedown; any A100, SXM4 or
PCIe, 40 GB first, then 80 GB (equal compute is measured on each box's own host, so the variant does not bias
max_steps); verified, reliability >= 0.93 (`STUDY_RELIABILITY`: the strict 0.98 left no 4x offer), driver CUDA >= 13.0,
>= 12 cores and >= 64 GB RAM per GPU, disk and network >= 500 MB/s / Mbit/s down, >= 100 Mbit/s up per GPU. The disk
comes from the extent's sizing with both label stores (the Cohere and the Parakeet store each copy the selected audio)
and the box's checkpoints (`kitsune.study_queue.study_extra_gb`: up to 6 full states per run at 16 bytes a parameter,
the bf16 exports, the probes running at once), about 450 GB for box A. `--offer-id`, `--max-dph` (default 6 $/h for 4
GPUs, 2 for 1), `--disk-gb` and `--max-hours` work as for a training run. launch also refuses a box whose numbers
file is already in the runs repo (a box writes its numbers once), and the replicate while box A's is not there or was
written under other rules than the commit's `study/PREREG.json`. It passes the rented GPU count (`KITSUNE_N_GPUS`):
the queue refuses to run on another count, and a box that trains a wave needs a GPU per run.

Order: the shakedown, box A, then the replicate before box B if you can: box B's speed probe times each student's
trained final weights, and takes box A's and the replicate's from the runs repo (the queue summaries under
`study/box-<box>/` name their run dirs). A student whose weights are not there yet is skipped with a logged note
(`speed_skipped`; the replicate's shape is study-t01's).

### Watching it

`tail -f /workspace/kitsune.log` shows the bootstrap, then the queue's lines (`[queue] item_start`, `item_end`,
`calibration_release`, `calibration`, `lr_chosen` / `lr_probe_extend`, `prereg_numbers`, `smoke_gate_off`,
`item_uploaded`, `queue_end`). Each trainer logs to `/workspace/kitsune_state/logs/<item>.log` and to its own run dir;
TensorBoard (runs/) shows every run of the box. The queue's state is `/workspace/kitsune_state/queue.json`; its summary
`queue_summary.json` (also in the runs repo at `study/box-<box>/queue_summary.json`). The numbers file goes to
`study/PREREG_numbers_<box>.json` in the runs repo, and its sha256 is the `prereg_numbers` event in
`kitsune_state/events.jsonl`, before the first study step's `item_start`. Every run dir is uploaded and verified as it
finishes (lean: logs, metrics, evals, summary, the weights at 0.4 and at the end, the branch's weights, and the 0.8 full
state of the runs of at most 0.1B; resume states stay on the box, a finished probe's are deleted). finish.py puts the
infra logs, `kitsune_state/logs/` (the store build, the anchor, the speed probes have no run dir of their own) and any
`rearm-<stamp>/` under `study/box-<box>/infra/<container>/`.

The calibration of a group starts together: each trainer sets up, writes `READY` in its run dir and waits; once all are
there the queue writes `GO` into every one (`calibration_release`), so each run's steps 50-250 - the pre-registered
window, literally - fall while the whole group trains, and the queue stops the group once all have logged step 250.
Every trainer of a run (its calibration, probes, main and branch) gets the same loader: the queue gives each trainer an
equal share of half the host's `/dev/shm` (the `shm` event) and cuts `perf.prefetch`, then `perf.num_workers`, to fit
it. A main whose calibration run (its own start at its class's lowest grid LR) showed no falling loss over the smoke
steps runs without the loss-trend check (`smoke_gate_off`, main and branch alike; a scratch run is ~5 % into its
warm-up at step 100); the replicate follows box A's decision for study-t01.

### How it ends, and how to stop it

| outcome | instance |
|---|---|
| the queue exits 0: every item done and verified on the Hub | `finish.py --destroy` (lean): **destroyed** once verified |
| a pre-registered halt (exit 4): calibration still loader-bound after `perf.num_workers` 12, an LR winner at a grid edge after its one extension, the numbers refused or already on the Hub | **stopped**; the reason is in `queue_summary.json` and the supervisor's decision |
| a trainer's throughput floor (exit 3) | **stopped** |
| the queue crashes (or the container restarts) | the queue again, up to 2 times: it resumes from `queue.json` (finished items skipped, a main run or probe from its local full state, a branch again from its parent, a calibration group again as a whole); then **stopped** |
| the cap | **stopped** by the watchdog |

Inside the queue a run that crashes resumes once from its own full state; a second failure marks it failed, the
others go on, and the box ends stopped. A run whose smoke checks fail (`SmokeFailed`) fails at once: the same start
fails the same way. The shakedown's smokes carry the study runs' own gate, so such a failure shows there first. To stop the whole box by hand: `vastai stop instance <id>` (disk kept) or
`vastai destroy instance <id>`. A stopped box continues in place with `vastai start instance <id>` and
`bash /workspace/Kitsune-Transcribe/vast/onstart.sh --rearm`: the rearm keeps `queue.json`, so the queue resumes and
never writes its numbers twice (delete `queue.json` only to start that box's study over, and never after its numbers
are on the Hub). Do not `touch runs/<run_id>/STOP` in a study run: it ends the run before its max_steps, which the
pre-registration counts as an invalid run.

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
