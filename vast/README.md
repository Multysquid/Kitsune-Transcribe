# vast.ai operator guide

This folder rents and runs the project's GPU boxes on [vast.ai](https://vast.ai). Two jobs remain:

- **full-data boxes** (`vast/launch.py --job full --box <box>`): one entry of the box registry
  `configs/full/boxes.json` each, a 1x RTX 5090 that trains one student on the whole label set and scores it. These
  boxes made the four released models.
- **the label box** (`vast/launch.py --job label`): one RTX 5090 that labelled the whole download with both teachers
  and sealed the label root every full box trains on.

A box boots our image, clones this repo at a pinned commit, pulls its derived data from a private Hugging Face
dataset, rebuilds the audio from the public upstream datasets, does its work, uploads and verifies everything on the
Hub, then destroys itself. A watchdog stops it at a hard cap whatever happens.

**Who does what.** Nothing is rented until `launch.py --yes` runs. The owner runs every paid step (rent, stop,
destroy, credit top-ups) and every Hugging Face upload, repo creation, deletion and token change. An agent prepares
exact one-line commands, runs `launch.py` only without `--yes` (look-only, it spends nothing) and only reads over ssh.

**Status (2026-10-12).** Every box has ended; no instance runs. Earlier jobs (the viability run, the size study, the
chained smoke box, box p-cool) are retired; "History" points at their code.

| file | runs where | what it does |
|---|---|---|
| `launch.py` | laptop | preflight (commit, image, Hub), offer search, the exact `vastai create instance` command; creates only with `--yes` |
| `onstart_stub.sh` | box | what `--onstart` sends (under 4 KB): clone at `KITSUNE_SHA`, run `onstart.sh`; stops the box if the clone fails |
| `onstart.sh` | box, every container start | env, limits, thread pools, TensorBoard, the watchdog, then the job (see "onstart.sh") |
| `bootstrap.sh` | full box | download gate, pulls, audio rebuild, coverage check |
| `supervise.py` | full box | runs `python -m kitsune.full_queue run --box <box>`, restarts it after a crash, then calls `finish.py` |
| `finish.py` | box | upload, verify every expected file on the Hub (path, size, hash), destroy; stop instead if anything is off |
| `watchdog.sh` | box | stops the box `KITSUNE_MAX_HOURS` after first boot, or when the controller's heartbeat goes stale |
| `label.py`, `label_sync.py` | label box | the label controller and its write-once uploads |
| `blocklist.json` | laptop | machines no job rents |

On the box everything logs to `/workspace/kitsune.log`; lifecycle records live in `/workspace/kitsune_state/` and go
to the Hub at the end, secrets redacted: a full box's to the runs repo under `full/box-<box>/infra/<container>/`, the
label box's to the data repo under `label_runs/<run_id>/`.

## One-time setup

### 1. The image (GitHub Actions, no home upload)

A push that touches `docker/**`, `requirements-train.txt`, `.github/workflows/image.yml`, `kitsune/quant.py` or
`tests/test_quant*.py` runs the `image` workflow: it builds `ghcr.io/multysquid/kitsune-train`, smoke-tests it, runs
the quant tests inside it and only then tags it `sha-<short commit>` and `<branch>` (~15-25 min). A code-only commit
has no image of its own: launch it with `--image-tag main`. launch reads the image's build commit and refuses it while
`requirements-train.txt` or `docker/Dockerfile` differ at the commit to run. The package is public (set once after
the first build under GitHub -> Packages -> Package settings), so vast pulls it without credentials; keep it public
and keep its `sha-*` tags, which run records name.

### 2. Hugging Face repos and token

1. **Three private repos** (the names used throughout; use yours):
   ```powershell
   hf repos create Multy123/kitsune-data --repo-type dataset --private
   hf repos create Multy123/kitsune-runs --private
   hf repos create Multy123/kitsune-scratch --private
   hf upload Multy123/kitsune-runs MODEL_CARD.md README.md
   ```
   `kitsune-data` holds the derived data (see "What lands in the data repo"), `kitsune-runs` every run, readout and
   box record, `kitsune-scratch` the trainers' timed full states (the trainers delete and squash its history, so it
   is never the runs or data repo). All three carry dataset reference transcripts and must stay private: launch
   refuses a public one. The last line gives the runs repo a root card (run it once, from the repo root).
2. **The data repo's contents** are owner uploads from earlier steps: the laptop's labels, the converted Parakeet
   teacher (`python tools/publish_parakeet.py --upload`), the label box's sealed root, the selections built on the
   laptop, the student inits and the augmentation banks. launch checks that a box finds everything it reads.
3. **The token.** A **fine-grained** token (https://huggingface.co/settings/tokens) with read AND write on
   `kitsune-runs` and `kitsune-scratch` (the trainers read their uploads back), read on `kitsune-data` (write too for
   the label box), and "Read access to contents of all public gated repos you can access" for the gated
   `CohereLabs/cohere-transcribe-03-2026` (the label box and the smoke boxes' Cohere speed probe; the account must
   have accepted the gate). No create permission. It goes into the vast account (step 3.2), never on a command line.
   A box checks it in its first minutes and stops with a refusal when anything is missing.
4. **The laptop's own login** (the owner's): launch's Hub checks read with it, read-only. `hf auth login` with a token
   that can read the three private repos (and, for the label box, the gated Cohere repo, its terms accepted);
   `hf auth whoami` shows the account. Without it launch cannot read the private repos and refuses.

### 3. vast.ai account

1. Credit. launch does not check it: keep it above the box's worst case (the `cap` in launch's cost line). A credit
   that runs out stops the instance mid-run.
2. Account -> Settings -> **Environment Variables**: `HF_TOKEN` = the token from step 2.3. vast injects it into every
   instance; anyone who can ssh in can read it. Never change it while a box is alive: a restart picks up the new value.
3. The CLI on the laptop, on PATH for launch.py, with your API key (you type it):
   ```powershell
   pip install vastai==1.8.0
   vastai set api-key <key from https://cloud.vast.ai/manage-keys/>
   ```
   vastai 1.8.0 exits 0 even when the API refuses: read its output line (`stopping instance <id>.`).
4. Your ssh public key under Account -> Keys **before** renting (keys added later miss running instances).

## Full-data runs (the full boxes)

A full box is one entry of `configs/full/boxes.json` (validated by `kitsune/fullrun.py`, documented in its
docstring): GPU count, data config, planned and capped hours (`est_hours`, `max_hours`), price cap (`max_dph`), extra
disk, RAM floor, watchdog rule, the extra files and dirs it pulls, and its items (store builds, trainings, M4
readouts, quantised readouts, speed probes, Whisper evals). launch reads the registry at the commit the box runs and
keeps no table of its own; on the box `kitsune/full_queue.py` runs the items.
`python -m kitsune.fullrun show --box <box>` prints a box's spec and env,
`python -m kitsune.full_queue plan --box <box>` its items in order. The configs in `configs/full/` are generated by
`tools/make_full_configs.py` and byte-pinned by the tests: they are what ran.

| box | what it runs | data config | est / max h, $/h cap | record |
|---|---|---|---|---|
| `full-smoke` (smoke A) | both stores, four short trainers, faults F1-F5, two readouts, bf16 speed of the study students and teachers; checks 1-11 | `data-smoke.json` | 5.2 / 9, $1.00 | ran once, in the chained box (History) |
| `smoke-b` (smoke B) | quant and compile self-tests, fp16, export vs memory, 4 Whisper models, speed probes; checks 12-16, report only | `data-smoke-b.json` | 1.75 / 3, $1.00 | run #2 passed on 2026-10-02 at 078c5e0: the quant go signal |
| `p01` (box 1) | P-0.1B, 4 epochs, Parakeet labels only | `data-p01.json` | 14.3 / 22, $1.10 | 2026-10-01/02 at 7715f3f (with that commit's entry): M4 11.45 %, revision c4604304, released. The entry now records an 8-epoch continuation that never ran |
| `full-p` | P-0.3B, 6 epochs, recipe v2; 7 quantised readouts; 4 Whisper models | `data-p.json` | 30.0 / 42, $1.10 | 2026-10-03/04 at 30b13f3: M4 9.73 %, revision 61afd417, released; ~$26.5 |
| `p005` | P-0.05B, 10 epochs planned (the early stop at step 67,449 started the decay, which ended at step 80,939), recipe v4; 7 quantised readouts | `data-p01.json` | 18.5 / 27, $1.10 | 2026-10-08 at ad0861f: M4 16.35 %, revision d2caddfc, released; ~$10 |
| `full-t` | T-0.6B, 6 epochs, the AED recipe; 7 quantised readouts; the bf16 re-time pair | `data-t.json` | 50.9 / 68, $1.40 | 2026-10-10/12, launched at 23081bb, resumed on a second host at bde0f36: M4 9.42 %, revision d1c468e0, released; ~$63 |

Revisions are of the runs repo, whose run dirs are `runs/<item>-<UTC stamp>/` (e.g.
`runs/full-t06-20261010T055818Z`). The heads of the three P run dirs hold box p-cool's later re-runs, not the
released models: cite the revisions above.

### Before a full box (the owner's steps)

1. The scratch repo exists, is private and the vast token can write it; every box but smoke-b needs `--scratch-repo`.
2. The data repo holds the sealed label root, the full selection and its sidecar (`labels/full/selections/full_study/`,
   built with `scripts/make_selection.py` in full mode, see its Usage block), the student inits and the box's extra
   files and dirs (the cut table from `tools/aed_cut_table.py`, the banks from `tools/build_noise_bank.py` and
   `tools/build_rir_bank.py`).
3. `python tools/make_full_configs.py --check` exits 0 with `<checkout>/configs/full: up to date` (backslashes on
   Windows), and the image is built.
4. The vast credit covers the worst case: rent for `max_hours` at `max_dph` plus traffic (box full-t: ~$95 of rent
   for 68 h at $1.40/h and up to ~$8 of traffic, ~$103 in all).

### Commands (PowerShell, from the launch clone at origin/main)

Launch from a clean clone at the commit the box will run: the box clones exactly that commit (`KITSUNE_SHA`: the
clone's HEAD, or `--sha`), and launch reads the registry and configs there. Any clean clone works; the owner's is
`D:\kitsune-launch`, made once with `git clone https://github.com/Multysquid/Kitsune-Transcribe D:\kitsune-launch`.
The registry's full-t comment names the box branch t-box: PR #54 merged it into main (76b28a1), so origin/main
carries the same box T.

Two refusals apply to these launch lines today:

- **The quant go signal.** A box with quantised items (p01, full-p, p005, full-t) is refused unless smoke-b's verdict
  on the Hub (`full/box-smoke-b/smoke_verdict.json`) passed checks 12-16 at an ancestor commit with the quant code
  (`QUANT_CODE` in `vast/launch.py`: `kitsune/quant.py`, `tools/speed_probe.py`, `scripts/05_evaluate.py`,
  `kitsune/whisper.py`, `tools/whisper_eval.py`, `requirements-train.txt`, `docker/Dockerfile`) unchanged since. That
  code changed after smoke-B #2 (PR #52's fp16 fix, the 2026-10-12 cleanup), so these boxes need a new smoke-B at the
  commit (~1.75 h; it destroys itself on exit 0 whatever its checks say) or the owner's `--allow-unverified-quant`,
  which turns the refusal into a warning. Box full-t ran with the flag.
- **A box that already ran.** Every box but smoke-b has a Hub queue summary with its training done, and a launch of it
  without `--resume` is refused: a fresh queue would overwrite that summary, the record `--resume-reset` reads.
  `--fresh-over-done` lets a deliberate fresh start through; a new run is cleaner as a new registry entry (a new name
  in `kitsune.fullrun.BOX_NAMES` and the registry).

Each box's launch line at main (every box but smoke-b is refused today without `--resume` or `--fresh-over-done`):

```powershell
Set-Location D:\kitsune-launch; git fetch origin; git checkout --detach origin/main
python vast\launch.py --job full --box full-smoke --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
python vast\launch.py --job full --box smoke-b --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs
python vast\launch.py --job full --box p01 --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch --allow-unverified-quant
python vast\launch.py --job full --box full-p --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch --allow-unverified-quant
python vast\launch.py --job full --box p005 --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch --allow-unverified-quant
python vast\launch.py --job full --box full-t --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch --allow-unverified-quant
```

Each line first as it is (look-only: preflight, offer table, create command, cost line), then the same line with
`--yes` to rent. Offers are ranked by the estimated total (planned hours at the offer's $/h with the disk, plus
traffic); `--offer-id N` picks another row, `--machine <id>` rents one machine only, `--avoid-machine <id>`
(repeatable) skips one. Offers churn within minutes: look again right before renting.

### What launch checks, before anything is rented

- **Commit and image:** the commit is on GitHub with the configs committed (`--skip-git-checks` overrides); the image
  tag resolves to a digest built with this commit's dependencies. `--gpus` and `--config` may only repeat the
  registry's values; `--max-hours` and `--max-dph` override its cap and price.
- **The Hub,** read-only with the laptop's login: the data and runs repos are private; the selection matches the data
  config's sources, eval sets and recipe; the extent's root is sealed and `extent.json` lists every file (it also
  sizes the disk, the download and the rebuild timeout). `full_preflight` adds: every config and tool the items run is
  committed (a CLI or flag the commit lacks is refused here, not on the rented box); the students are the registered
  builds (`kitsune.prereg.student_problems`); the extra files and dirs exist, banks and cut table at their pinned
  sha256; the scratch repo is private; the selection sidecar matches; the resume checks (below); the quant go signal.
- **Offers:** 1x RTX 5090, reliability >= 0.98, CUDA >= 13.0, >= 16 cores, disk >= 500 MB/s (`--min-disk-bw`),
  >= 500 Mbit/s down and >= 100 up, a direct port, the extent's disk plus `extra_gb`; verified or deverified hosts, a
  maximum rental of at least max(4 d, the cap + 0.5 d), RAM >= 93,120 MB for `min_ram_gb` 96 (vast lists 96 GB
  machines a little under 96,000 MB), traffic <= $0.01/GB (`--max-gb-cost`), an expected total <= 1.25 x the $/h cap
  x the planned hours plus traffic (`--max-total`), and nothing listed in mainland China (no Hub access there).
- **Avoided machines:** `vast/blocklist.json` (151760, a ~2.9 MB/s Hub link; 54650, which never started its
  instance), the label runs' failed hosts, and machines whose download gate said slow in the last 30 days
  (`full/box-*/infra/*/download_gate.json`). A `--machine` on that list is refused.
- **Output:** the offer table, the `vastai create instance` command (image by digest, `--ssh --direct`, one `--env`
  value without `HF_TOKEN`, `--onstart vast/onstart_stub.sh`, label `kitsune-full-<box>-<data config stem>-<sha7>`)
  and the cost line (expected total and total at the cap).

### On the box

`bootstrap.sh` runs its phases in order (timings in `$KITSUNE_STATE/bootstrap_timings.jsonl`):

1. **download_gate** (`python -m kitsune.netgate`): three pinned upstream files (~2.5 GB), one at a time. A host that
   could not pull 571.2 GB within `--gate-hours` (default 5, i.e. 31.7 MB/s; 0 turns it off; smoke-b has no gate) is
   refused with exit 3 within ~3 minutes. A pass sizes the label pull's and the rebuild's timeouts.
2. **plan** (with the token checks of step 2.3), **pull_derived** (all but the labels), **resume_pull** (`--resume`
   only), **check_students** (`python -m kitsune.fullrun check-students`).
3. **pull_labels** in the background while **rebuild_audio** runs `01_prepare_data.py --extent-config` (the label
   box's canonical ingest order, up to `KITSUNE_DOWNLOAD_AHEAD` = 6 files downloading ahead), then
   **pull_labels_wait** and **coverage** (every stem's rebuilt ids hash to the extent record's; every split joins at
   1.0). The full extent is ~571 GB: box 1 took 3.1 h one file at a time, later boxes 0.9-1.7 h.

Every phase keeps `$KITSUNE_STATE/train_hb` fresh for at most its own worst case. Then the queue runs: the store
build, the training, its M4 readout right after it, then the eval pool (quantised readouts, Whisper, speed probes; a
speed probe has the host to itself). Each child beats its own `hb/<item>` (the queue's stall check), the queue beats
`train_hb` (the watchdog's: stale for `orphan_s`, 3600 s, means a dead controller; smoke A's is 600 s and only
alerts, because its fault test freezes the beat). A trainer keeps a local full state every 30 min and uploads one to
the scratch repo every 120 min (T-0.6B's is ~8.6 GB); its pre-cooldown state and its weights go to the runs repo. Two
files steer it by hand:

```bash
touch /workspace/Kitsune-Transcribe/runs/<run_id>/COOLDOWN   # start the WSD cooldown now (once)
touch /workspace/Kitsune-Transcribe/runs/<run_id>/STOP       # finish the step, then the normal end phase, exit 0
```

### Watching it

```powershell
vastai show instance <id>          # wait for "running" (the image pull takes ~3-5 min)
vastai ssh-url <id>                # -> ssh://root@<host>:<port>
ssh -p <port> root@<host> -L 6006:localhost:6006
```

| what | where |
|---|---|
| logs | on the box `tail -f /workspace/kitsune.log`, per item `/workspace/kitsune_state/logs/<item>.log` |
| TensorBoard | http://localhost:6006 through the tunnel (if 6006 was taken on the box, onstart logs the port it used) |
| the box's state | runs repo `full/box-<box>/queue_summary.json`, put as each item starts and ends: the source of truth of a resume; the smokes' `smoke_verdict.json` beside it |
| the runs | runs repo `runs/<run_id>/`, readouts `runs/m4-<run_id>[-r<N>]/`, other items `runs/<item>-<stamp>/`, speed `runs/speed-<box>-<stamp>/` |
| timed states | scratch repo `runs/<run_id>/checkpoints/full_step_<N>/` and its pointer `runs/<run_id>/timed_state.json` |
| at the end | runs repo `full/box-<box>/infra/<container>/`: queue.json, events, per-item logs, the gate, the resume plan |

`python tools/export_run.py hf://Multy123/kitsune-runs/runs/<run_id> --out <dir>` turns a run into flat parquet and
CSV files; `tools/full_report.py` (its Usage block) builds the report from a lean pull of the runs repo.

### How it ends

| outcome | instance |
|---|---|
| the queue exits 0: every non-droppable training done and every output verified | `finish.py --destroy`: **destroyed** once verified, else **stopped** |
| a training or store item failed for good (queue exit 4) | **stopped**, the disk kept for the owner |
| the queue crashes or the container restarts | the queue again (it resumes from `queue.json`), up to 2 restarts, then **stopped** |
| a readout, speed probe or eval fails | recorded (`readouts_failed`), never a reason to keep a paid box |
| the download gate refuses, or bootstrap fails before a run dir exists | `finish.py --abort`: infra up, then **destroyed**; a slow gate's machine is avoided for 30 days |
| bootstrap fails after resume_pull pulled run dirs | **stopped** |
| `train_hb` stale for `orphan_s` | logs synced, **stopped** by the watchdog |
| the cap (`max_hours` after first boot) | logs synced 10 min before, **stopped** by the watchdog |

finish.py is lean for a full box: logs, weights dirs and the pre-cooldown state, never the timed states. The
supervisor bounds its final call (30 min to stop, 60 min to destroy); one that hangs or fails ends as
`finish.py --stop --no-sync`. A halt marker 20 min old with the instance still up makes the watchdog request the stop
again every 5 min.

A stopped instance keeps its disk and bills storage (~$0.3-0.4/h for ~1.5 TB). To look inside:
`vastai start instance <id>`, ssh in, read `/workspace/kitsune.log` and `/workspace/kitsune_state/`; the `halt` marker
keeps a run from starting. Its local full states can still go to the scratch repo (`kitsune.scratch.make_pointer`
and `upload_state`). Continue in place with `onstart.sh --rearm` (see "onstart.sh"), or destroy it and resume on a
new host. By hand: `vastai stop instance <id>` or `vastai destroy instance <id>` (it asks to confirm; `-y` skips
that); check the id against the instance label first.

### Resume on a new host

```powershell
python vast\launch.py --job full --box <box> --resume [--resume-reset <run_id>] [--resume-set <run_id>:<key>=<value>] --max-hours <hours left + setup> --image-tag main --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch
```

- **`--resume`** (`KITSUNE_RESUME=1`): bootstrap's resume_pull (`python -m kitsune.full_queue resume-pull`) reads the
  box's Hub queue summary, pulls every started run with its newest full state (the scratch repo's or the runs repo's,
  checked against its sha256) and marks the finished items done; the queue adopts that plan.
- **`--resume-reset <run_id>`** (repeatable): continue a run past an early stop or its cooldown, from its pre-cooldown
  state (`schedule.resume_reset` in `scripts/04_distill.py`). Its readout writes `runs/m4-<run_id>-r<N>`, so the
  first readout stays.
- **`--resume-set <run_id>:<key>=<value>`** (repeatable, one key once per run): resume with another value of a key of
  `kitsune.fullrun.RESUME_SET_KEYS` (`launch.py --help` lists them: `schedule.epochs`, `early_stop.patience`,
  `augment.*`, `schedule.deadline_cooldown`; `augment.*` only on a box with a CTC training). With `--resume-reset` of
  the same run it is a continuation, e.g.
  `--resume-reset <rid> --resume-set <rid>:schedule.epochs=8 --resume-set <rid>:early_stop.patience=12`. Both flags
  imply `--resume`.
- **Refused:** a box without a Hub summary; a reset or set id no train item of the box ran; a set-only id of a done
  run; a plain `--resume` of a box whose trainings are all done (`--allow-done-trains` for a box lost in its eval
  pool); a resume of full-p, p01 or p005 without `--allow-continued-runs` (box p-cool's re-runs wrote into those run
  dirs, `RETIRED_CONTINUATIONS` in launch.py). launch prints what will resume and warns about a live instance of the
  box: destroy the old one first, two boxes must never write the same run dirs.
- **`--max-hours`:** the default is the whole box's cap; give a resume what the rest needs plus ~2-4 h of setup.

Box T is the worked example: on 2026-10-10 its trainer segfaulted four times on deverified host 11890, the queue
halted and the box stopped; its newest local state went to the scratch repo, the box was destroyed, and `--resume` on
a verified host finished the run on 2026-10-12. A registry train item may also carry a `continues` block (re-run a
done run of another box; `kitsune/fullrun.py`, "Continuations"); no box has one since p-cool was retired.

### Speed comparability (the report's chart)

RTF and VRAM compare only within one host, so `tools/full_report.py` charts the machine group with the most timed
systems: smoke-B #2's host, which timed the study students (bf16 and six formats), the compile probes, the Whisper
models and the teachers. The full-data P rows use their study weights' record (same shape and code; CTC greedy
decoding does not depend on the weights), so boxes full-p, p01 and p005 time nothing. Box full-t times a bf16 re-time
pair (`speed-full-t06`, `speed-study-t06`): when full-t06 emits more than 5 % more or fewer tokens than study-t06 on
the 200 speed ids, the report scales study-t06's record by the pair's ratio.

## The label box (`--job label`)

One on-demand RTX 5090 (32 GB) labels the whole extent of `configs/full.json` (~13.0k h) and destroys itself once
everything is verified. `onstart.sh` hands over to `vast/label.py`, which rebuilds the audio with one
`01_prepare_data.py --extent-config` process and runs, on the one GPU at once, two Cohere processes
(`02_teacher_pass.py`) and one Parakeet process (`02p_parakeet_pass.py`); `02b_second_opinion.py` runs on the CPU every
20 min. A shard's audio is deleted once both its label files check out, which keeps the disk at 250 GB. Laptop labels
with matching ids are adopted byte for byte; the three gate sets are always the laptop's. No audio is uploaded.

The run (2026-09-25/26, instance 52611168, $0.626/h) took ~11.15 h and ~$9: Cohere at ~412x realtime per process,
Parakeet at ~540x. Its finalize step F6 could not pass (a bug fixed in PR #10), so the box stopped with every label
uploaded and the seal was written from the laptop: `labels/full/` is **sealed** at data-repo commit 34aa704.

### What lands in the data repo

```
Multy123/kitsune-data
  teacher_out/, second_out/              the laptop's labels (the label box's seeds)
  models/parakeet-tdt_ctc-0.6b-ja-hf/    the converted Parakeet teacher (tools/publish_parakeet.py)
  labels/full/                           the label box's root
    LEASE.json                           which box owns the root (a heartbeat; released at the end)
    teacher_out/<source>/<split>-NNNNN.{npz,jsonl}     Cohere (the 02_teacher_pass.py format)
    parakeet_out/<source>/<split>-NNNNN.{npz,jsonl}    Parakeet TDT + CTC (kitsune/parakeet_targets.py)
    second_out/<source>/<split>-NNNNN.jsonl            {id, hyp2, model2, agree, cer2}
    extent.json                          every upstream input -> its shard stems and their id hash
    selections/full.parquet              the label box's selection
    selections/study_1000h.*, study_manifest.json   the size study's selection; the frozen eval manifest
    selections/full_study/               the full runs' selection + sidecar, smoke A's, box full-t's cut table
    reports/, provenance/                baselines, label counts, throughput, consumer check; adopted labels
    COMPLETE.json                        written last, in its own commit: the root is sealed after it
  label_runs/<run_id>/                   the label box's logs and state (scrubbed)
  students/study/<name>/                 the student inits (kitsune.prereg.RUNS)
  aug/musan-bg-v2/, aug/rirs-v1/         the augmentation banks
  selection/viability.parquet, students/b20x2560-d4/   the viability run's (History)
```

The 2026-09-26 run also wrote `selections/full_sub3k.parquet` and `reports/galgame_judge.json`, from a subset config
and a report step since removed. Every file under `labels/full/` is written once: a relaunch adds missing files and
never changes one (a different remote file is an integrity error), except `LEASE.json` and, before the seal, the
finalize outputs. Nothing is deleted; later files go to new paths (`selections/full_study/`).

### Launch

```powershell
python vast\launch.py --job label --data-repo Multy123/kitsune-data --image-tag main
```

Look-only first, then with `--yes`. launch checks the data repo is private, the seed and pinned Parakeet files, the
Cohere gate, no `COMPLETE.json` and no live lease under the root, every label config's extent, and the repo size.
Offers: on-demand verified RTX 5090, >= 16 cores, >= 64 GB RAM, 250 GB of disk, a strict tier first (reliability >=
0.97), ranked by the estimated total with traffic (~16 h planned, cap 30 h). Options: `--offer-id`, `--max-dph`
(default 1), `--max-hours` (default 30), `--config` and `--label-configs` (default `configs/full.json`),
`--cohere-procs` (default the config's 2), `--avoid-machine`, `--disk-gb`, `--steal-lease` (only for a box you know
is gone), `--no-self-stop`.

**`labels/full/` is sealed, so this line is refused today**: a new label run needs a label config with a new
`extent.root`. The full boxes keep reading `labels/full/`.

### How it ends

| outcome | instance |
|---|---|
| finalize passed, `COMPLETE.json` verified on the Hub | **destroyed** |
| budget (deadline minus 60 min with work left) or a host failure (self-test, golden check, a lane failing 4 times, a full disk, Cohere under 400x realtime) | final sync and verify, then **destroyed**; relaunch to continue (a failed host is avoided) |
| refusal (token, public repo, live lease, `COMPLETE.json`, seeds or pins) | **stopped**, minutes in |
| integrity (a label that does not match its shard, a write-once conflict, the consumer check), a bug, a failed final verify | sync of what is safe, then **stopped**: the disk keeps the evidence |
| the controller died or hung | the watchdog syncs after 15 min without its heartbeat, then **stops** |
| anything else | the watchdog syncs 30 min before the 30 h cap and **stops** at it |

A destroy happens only after the Hub provably holds every finished label. The reason is in
`label_runs/<run_id>/label_end.json`. `KITSUNE_LABEL_END=stop` turns every destroy into a stop for debugging; it has
no launch flag, so it goes into the vast account environment (every box sees it): remove it afterwards.

### Relaunch and continue

- **Destroyed without `COMPLETE.json`:** run the same launch line again. The new box pulls the labels already there,
  re-downloads the audio (labelled shards pass at once) and resumes at the first unlabelled shard: +0.5-1.5 h, +$1-4.
- **Stopped:** read `label_runs/<run_id>/` and `/workspace/kitsune.log`, fix the cause, then destroy and relaunch, or
  `vastai start instance <id>` and `bash /workspace/Kitsune-Transcribe/vast/onstart.sh --rearm`.
- **A container restart** resumes by itself from `$KITSUNE_STATE/label.json`.
- The lease refuses a second box while another container's heartbeat is under 45 min old.

### Subsets

A training config can use part of the labelled extent with no relabelling. Its `extent.inputs` takes a prefix of each
capped source's sorted upstream inputs: `{"reazon_large": N}` (of 705 files), `{"galgame": N}` (of 115 tars),
`{"emilia_nc": N}` (of 66), and `emilia_yodas` as `"300h"` or an int 9-29. `configs/full.json` is the whole extent
(`"inputs": {}`); `configs/full/data-smoke.json` is a capped one. A new subset is a new config plus a selection derived
from the label box's in seconds:

```bash
python scripts/make_selection.py --config configs/<new>.json \
  --from-selection labels/full/selections/full.parquet --out labels/full/selections/<new>.parquet
```

Upload it as a new file under `labels/full/selections/` before launching. A full box rebuilds only that subset's audio,
pulls only its label files (`kitsune.extent.pull_plan`) and requires the rebuilt ids to match `extent.json` exactly;
launch sizes its disk and rebuild timeout from the record (the whole extent: ~571 GB down, ~1.4-1.5 TB of disk).

## onstart.sh: what runs at every container start

launch.py sends the small `vast/onstart_stub.sh` as `--onstart` (the API may cap that field near 4 KB); the stub
clones the repo at `$KITSUNE_SHA` and execs `vast/onstart.sh`, which runs as root at EVERY container start and stays
under vast's 16 KB on-start limit (a test holds it there), which is why its story lives here.

Steps: sync the env to `/etc/environment` (ssh sessions do not inherit the container env), raise the nofile limit,
check `/dev/shm`, size the CPU thread pools, start TensorBoard (ports 6006-6008) and the vast portal, clone the repo,
start `vast/watchdog.sh`, then, detached: `bootstrap.sh` and `supervise.py` for `KITSUNE_JOB=full`, or `label.py`
alone for `KITSUNE_JOB=label`. Any other job, or a missing `KITSUNE_SHA`, `KITSUNE_DATA_REPO`, `KITSUNE_CONFIG`,
`KITSUNE_MAX_HOURS` (a full box: also `KITSUNE_BOX`, `KITSUNE_OUT_REPO`), is refused: halt marker, then stop.

- **Thread pools.** A container's `nproc` is the host's, so on a full box `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `RAYON_NUM_THREADS` and `TOKIO_WORKER_THREADS` get the cgroup's `cpu.max`
  quota over the GPUs, at most 16 on a low pids budget; a value already set wins. The label box uses 16 on a low pids
  budget only.
- **Restarts.** The deadline is fixed at first boot. A `halt` marker means the run is over: a restart then only brings
  up the env, TensorBoard and the portal. A restart that finds `$KITSUNE_STATE/supervise.json` skips bootstrap and
  hands over to the supervisor's history. A failure before the supervisor takes over runs `finish.py --abort`
  (destroy a full box without a run dir, else stop; the label box stops) unless `KITSUNE_NO_SELF_STOP=1`.
  `supervise.lock` keeps a second run from starting when the script is run again by hand.
- **Re-arm.** `bash /workspace/Kitsune-Transcribe/vast/onstart.sh --rearm` continues a halted box in place: it moves
  the halt marker, the deadline, the supervisor history, the heartbeats, `label.json`, the resume plan, the gate
  record, the watchdog's alerts and the smoke verdict to `$KITSUNE_STATE/rearm-<stamp>/` and boots as new, with a new
  cap from now (export `KITSUNE_MAX_HOURS` first for another). A full box's `queue.json` stays, so its queue resumes.
  It refuses while a watchdog or supervisor of this container runs: stop and start the instance first. Deleting only
  `halt` does not work (the old deadline and the recorded decision stay).
- **`--no-self-stop`** (launch) puts `KITSUNE_NO_SELF_STOP=1` into the one `--env` value: a failed on-start or bootstrap
  then leaves a fresh box running for debugging; the watchdog still stops it at the cap. By hand, put it inside that
  quoted value: a second `--env` replaces the first.

**Instance env** (launch.py sets it; `HF_TOKEN` comes from the vast account and is never printed): `KITSUNE_JOB`,
`_SHA`, `_CONFIG`, `_DATA_REPO`, `_DATA_REVISION` (the data repo's head at launch), `_MAX_HOURS`, `_DPH`, `_MACHINE_ID`,
`TZ=UTC`. A full box adds `KITSUNE_BOX`, `_OUT_REPO`, `_SCRATCH_REPO`, `_N_GPUS`, the watchdog's
`_WATCHDOG_HB_FILE=train_hb`, `_WATCHDOG_ORPHAN_S` and `_WATCHDOG_ORPHAN_ACTION`, the gate's `_GATE_BYTES` and
`_GATE_MAX_H`, `_REBUILD_BYTES`, `_PULL_BYTES`, `_REBUILD_TIMEOUT_MIN` and on a resume `_RESUME`, `_RESUME_RESET`,
`_RESUME_SETS`. The label box adds `KITSUNE_LABEL_CONFIGS`, `_IMAGE`, `_WATCHDOG_SYNC_LEAD_S=1800`,
`_WATCHDOG_ORPHAN_S=900` and optionally `_STEAL_LEASE`, `_COHERE_PROCS`.

## HF storage

The data repo held ~2.3 GB before the label box (laptop labels, the viability run's selection and student), plus
2.5 GB of Parakeet teacher; a label run adds ~36-49 GB in ~27k files (`parakeet_out` ~1.34 MB per audio hour), and
launch warns when the projection passes 80 GB. Each full run adds its weights, its pre-cooldown state (~8.6 GB for
T-0.6B) and its readouts to the runs repo; the scratch repo keeps one timed state per run. The account used 130.7 GB
on 2026-09-27; HF PRO gives 1 TB of private storage. Deletions are the owner's.

## History

The retired jobs' code and operator text are at commit 76b28a1 (the PR #54 merge the release cards pin):
`git show 76b28a1:vast/README.md` is this guide as it was, and `git show 76b28a1:<path>` gives any removed file.

**The viability run (2026-09-25).** The first box (`--job train`): one A100 SXM4 40 GB at $1.031/h, code 0a53bda,
`configs/viability.json`, a 4 h training loop of the T-0.6B shape (`students/b20x2560-d4`) on 519 h of laptop labels
(`selection/viability.parquet`), $4.38. Pooled CER 12.15 %, verdict INCONCLUSIVE: the limit was generalisation, so
the owner labelled the whole download next. See `git show 76b28a1:configs/viability.json` and `JOBS["train"]` in
`git show 76b28a1:vast/launch.py`. The student and selection stay in the data repo: T-0.6B's init reuses that
student's FFN selection.

**The size study (2026-09-26).** `--job study --box shakedown|A|B|replicate`: seven students and a bridge (a
from-scratch control), each trained once at equal compute on one frozen 1,000 h selection, on 4x A100 boxes A and B
(plus a test box and a failed first box A), ~$63.7. Its lasting outputs are the student inits (`students/study/<name>`),
the frozen eval manifest (sha256 ef56dec2) every M4 number is scored on, and `study/PREREG.json`, kept verbatim and
pinned by a test. The queue phases, `study/PREREG.md`, `tools/study_report.py` and 57 of the 58 `configs/study/*.json`
are removed (`git show 76b28a1:kitsune/study_queue.py`, `git ls-tree --name-only 76b28a1 configs/study/`);
`study-t06.json` stays for smoke-b's eval store.

**The chained smoke box `p01-chain` (2026-10-01).** Smoke A, smoke B and box 1 on one RTX 5090 rental, with an
automatic gate between them (`ChainController` in `kitsune/full_queue.py`). It ran once (instance 53693389, commit
14bfcad, ~2.1 h, ~$2.3): the gate failed on smoke A's checks 5 and 7, smoke B #1 failed checks 12, 14 and 16, and box
1 then ran as box `p01` alone. The entry and the chain code are at `git show 76b28a1:configs/full/boxes.json`,
`git show 76b28a1:kitsune/full_queue.py` and `git show 76b28a1:tests/test_full_chain.py`.

**Box `p-cool` (2026-10-09).** One RTX 5090 rental that re-ran the cooldowns of the P-0.3B, P-0.1B and P-0.05B runs
from their pre-cooldown states with other augmentation recipes (registry `continues` blocks from
`tools/make_full_configs.py`'s `CONTINUATIONS`), then their readouts: ~10.7 h, ~$11.3, runs-repo revision e3a69f2f.
All three tied their baselines; none was released. The re-runs wrote into the same run dirs, so the heads of
`runs/full-p03-20261003T230143Z`, `runs/full-p01-20261001T184145Z` and `runs/full-p005-20261008T160232Z` hold
them; the released models are revisions 61afd417, c4604304 and d2caddfc. See
`git show 76b28a1:configs/full/boxes.json` and `git show 76b28a1:tools/make_full_configs.py`; `--resume-reset`,
`--resume-set` and `continues` blocks stay.
