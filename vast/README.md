# Running the viability run on vast.ai

One A100 for ~4.5 h: the box boots our image, clones this repo at a pinned commit, pulls the small derived data from a
private HF dataset, rebuilds the audio from the public upstream datasets, trains for 4 h, uploads everything to a
private HF model repo, verifies the upload and destroys itself. A watchdog stops it at 5.5 h whatever happens.
Nothing is rented until you run `launch.py --yes`.

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
   of a configured source or eval set, or keeps rows whose teacher output is not uploaded. The uploaded `teacher_out`
   must not go beyond what the box rebuilds with `01_prepare_data.py`'s defaults (the first 6 Galgame tars, 300 h of
   Emilia-YODAS): a teacher pass on more of the full download needs a matching rebuild extent, or the box's coverage
   check stops the run after the paid audio rebuild. From the repo root:
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
   image by hand, `--config configs/<other>.json`.

## Watching it

```bash
vastai show instance <id>          # wait for "running"
vastai ssh-url <id>                # -> ssh://root@<ip>:<port>
ssh -p <port> root@<ip> -L 6006:localhost:6006
```
Then open http://localhost:6006 for TensorBoard, and on the box `tail -f /workspace/kitsune.log`. If port 6006 was
taken on the box, the log line `TensorBoard on 127.0.0.1:<port>` says which one to forward instead.

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
complete eval sets, plus the probe - and a mini eval every 200 optimizer steps (32 utterances per gate set, 64 of the
probe). The numbers to read first are in TensorBoard under `2_loss_accuracy/00_summary/full/` and `.../mini/`
(`val_cer_pct`: CER vs the reference pooled over JSUT / CV8 / ReazonSpeech-test, `val_cer_vs_teacher_pct`,
`train_cer_vs_teacher_pct`, `val_loss`, `train_loss`, ...), and each eval prints one line to `kitsune.log`:
`[full eval] step N epoch E | val CER x.x% (vs teacher y.y%) on U utts (complete) | train CER vs teacher z.z% (vs ref
w.w%) | val KL ...`. The first point of `00_summary/full/` (step 0) decodes only the fixed 500-per-set subset (1,500
gate utterances, "subset" in its line), every later point the complete gate sets (14,746); `val_cer_utts` next to
`val_cer` shows which.
The cost of a full eval on the A100 is an estimate until the first run measures it (`eval/wall_s`,
`eval/greedy_full/rtf`): ~5-10 min (the laptop measured ~10 min for this student in batched decoding; the `_comment`
in `configs/viability.json` has the arithmetic), against ~12-47 min of training per epoch.

To stop the training by hand (it looks flat on TensorBoard, or the results are already what you need):
```bash
ls /workspace/Kitsune-Transcribe/runs/                       # the run id: <run_name>-<UTC stamp>
touch /workspace/Kitsune-Transcribe/runs/<run_id>/STOP
```
The trainer checks for the file before every optimizer step: the step under way finishes (with its eval and
checkpoints, if due), then the normal end phase runs - final weights and full state, final eval (an epoch-end full
eval of that same step is reused, not decoded again), verdict, summary, uploads - and it exits 0, so the box verifies the upload and destroys itself as after a full run (`early_stop` event
with reason `stop_file`). Killing the trainer instead counts as a crash (resumed once, or the box is stopped; see below)
and skips the final eval. A trainer stuck waiting for its data loader (`kitsune.log` shows `unable to allocate shared
memory` and no new steps) never reaches the STOP check, but it gives up after `perf.loader_timeout_s` (10 min) and
exits as a crash.

## Where the results land

In the output repo under `runs/<run_id>/`: `summary.json` (verdict and final metrics), `metrics/`, `evals/`, `tb/`
(TensorBoard events), `events.jsonl`, `checkpoints/step_<N>/` (bf16 weights every 30 min), the full resume state from
before the cooldown and at the end, and `infra/` (box logs; exported too). Convert to flat files with
`python tools/export_run.py hf://Multy123/kitsune-runs/runs/<run_id> --out <dir>`, or run TensorBoard locally on
a downloaded `tb/`.

## How it ends

| outcome | instance |
|---|---|
| training exits 0 and every expected file is on the Hub (path, size, sha256) | **destroyed** (billing stops) |
| training exits 0 but verification finds a problem | **stopped** (disk kept, storage still billed) |
| exit 3 (throughput too low), a crash before step 100, or a second crash | **stopped** |
| first crash after step 100 with a local full state | resumed once, then as above |
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
has usually passed (the watchdog would stop the box at once) and the old history already holds a final decision.
`--rearm` refuses (exit 1, reason in the log: the script writes only to `/workspace/kitsune.log`) while a watchdog or
supervisor of the current container is still running; stop and start the instance first. Running the on-start script
again during a healthy run starts nothing (the supervisor holds `kitsune_state/supervise.lock`). When done:
`vastai destroy instance <id>`. For debugging a fresh box without it stopping itself on a bootstrap error, add
`-e KITSUNE_NO_SELF_STOP=1` to the create command (the watchdog still applies).
