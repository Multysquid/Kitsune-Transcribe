# Notes for coding agents

Read [README.md](README.md) first: what the models are, how they were made, the repository map. These notes are the
rules for changing the code. The owner (Multysquid on GitHub, Multy123 on Hugging Face) decides scope and runs every
step that costs money or publishes anything.

Code comments cite "DECISIONS Xn", "CONTRACT" / "contract N", "STUDY.md N", "plan v3", "decision N" and run-sheet ids
such as "D38a". They are the owner's private planning notes, not files in this repository: do not look for them, and
do not add new ones to code without saying what they mean.

## Map

| Path | Holds |
|---|---|
| `kitsune/` | The library: the release contract (`quant.py`, `generation.py`), the students, data and labels, the trainer's parts, evaluation, and the box machinery (`fullrun.py` registry and shared names, `full_queue.py` queue) |
| `scripts/` | The numbered pipeline `01` to `05` (`02b`, `02p`, `03c` beside them) and `make_selection.py` |
| `tools/` | Config generation (`make_full_configs.py`), plans, augmentation inputs, speed and Whisper readouts, the report |
| `vast/` | `launch.py` (laptop) and the box lifecycle (`onstart.sh`, `bootstrap.sh`, `watchdog.sh`, `supervise.py`, `finish.py`, `label.py`, `label_sync.py`); `vast/README.md` is the operator guide |
| `configs/`, `study/` | The label config, the generator template, the box registry and the generated configs that ran; the frozen study files |
| `docker/`, `.github/workflows/image.yml` | The training image and its CI, the repository's only workflow |
| `tests/` | The pytest suite, CPU only |

## The release contract: never break it

Users outside this repository depend on these. The cards pin `76b28a1` for the loader, but `RepetitionStop`, the
overview card, `quantization.json`'s `load_with` and the integration follow `main`:

| What | Who uses it |
|---|---|
| `kitsune/quant.py`: `load_quantized(path, device, *, impl="auto", dtype=torch.float32) -> (model, recipe)`, `SCHEMA = 1`, `RECIPE_VERSION = 2`, `FP16_FP32_MODULES`, `QUANT_FILE`, `WEIGHTS_FILE`, the pack-part names (`qweight`, `qscale`, `qblock_scale`, `qtensor_scale`), `read_recipe`, `verify_export`, `recipe_version_problem`, `python -m kitsune.quant inspect`. Importing it loads torch and the stdlib only | The four model cards (clone, `git checkout 76b28a1`, `from kitsune.quant import load_quantized`); every published `quantization.json` (`load_with`); the overview card. `recipe_version_problem` refuses every released folder if `RECIPE_VERSION` or `FP16_FP32_MODULES` changes |
| `kitsune/generation.py`: `RepetitionStop(prompt_len, window=24, max_period=12)` | The T-0.6B card, linked to the repository root ("after 24 repeating tokens") |
| `MODEL_CARD.md` at the repository root | `license_link` (`blob/main/MODEL_CARD.md`) in the README of every uploaded T checkpoint; the template `kitsune/student.py` stamps into new ones |
| The names in `INTEGRATION_NAMES` of `tests/test_release_contract.py` (`kitsune.ctc_student`, `patches`, `audio`, `features`, `trainset`, `text`, `acoustics`, `noise_bank`, and more of `quant`) | The owner's Shisu-ko integration: its evaluation scripts import them from a checkout. Its engine reads the variant-folder format directly, so the format must stay as it is too |
| `kitsune/__init__.py` and the package at the repository root | `sys.path.insert(0, "Kitsune-Transcribe")` in the cards |
| Commits `76b28a1`, `bde0f36`, `32ef44f`, `ff69e32`, `ad0861f`, `30b13f3` | Pinned by the cards and the released `quantization.json` / `student_meta.json`: never rewrite `main`'s history |
| The image `ghcr.io/multysquid/kitsune-train` | Stays public: `vast/launch.py` pulls it anonymously, and run records name its `sha-*` tags |

`tests/test_release_contract.py` checks the names, signatures and constants. Its name stays outside the image CI's
`tests/test_quant*.py` filter on purpose.

## Hard rules

- **The owner runs money and publishing steps**: renting (`vast/launch.py ... --yes`), stopping and destroying boxes,
  vast.ai credit, Hugging Face uploads, deletions, repository creation and visibility, tokens and logins. Prepare the
  exact one-line command and hand it over. You may run `vast/launch.py` without `--yes` (it only searches and prints)
  or with `--dry-run`.
- **Secrets**: `HF_TOKEN` lives only in the vast.ai account environment. Never put a token on a command line, in a file
  or in chat.
- **Boxes over ssh: read only.**
- **The owner's laptop** (Windows 11, RTX 4070 Laptop 8 GB) hard-crashes under heavy CPU load:
  - no GPU for agents: run everything on the CPU (`CUDA_VISIBLE_DEVICES=-1`; `tests/conftest.py` does it for tests);
  - one heavy job at a time: one pytest process, at most `-n 3`, `OMP_NUM_THREADS=2`; never training;
  - temp files, caches and `--basetemp` on `D:` (for example under `D:/kitsune-tmp`), not on the system disk;
  - leave other checkouts alone: `D:/Shizu-ko-distill` holds local data other tools read (never `git clean -x` it),
    `D:/kitsune-launch` is the owner's launch clone and the integration's import path.
- **Never regenerate the release cards.** They are maintained by hand on the Hub; the release scripts live outside
  this repository.
- **Frozen records stay as they are; new work goes to new paths**: the sealed `labels/full` in `Multy123/kitsune-data`
  (commit `34aa704`), the study selection and frozen manifest (`study_manifest.json`, `ef56dec2…`), the training
  selection `full.parquet` (`e9a0695a…`), `study/PREREG.json`, `study/data.json`, `configs/study/study-t06.json`,
  `configs/full/full-*.json`, `smoke-*.json`, `data-*.json` and `configs/full/plan/*`. The run configs are the configs
  that trained the released models.
- **Never commit data** (`.gitignore` covers `data/`, `labels/`, `students/`, `runs/`, `*.parquet`, `*.safetensors`).
- **Never weaken a pin or a digest to make a test pass.** A failing pin means the change touched a frozen record.

## Edits with side effects

| Edit | Effect |
|---|---|
| `docker/**`, `requirements-train.txt`, `.github/workflows/image.yml`, `kitsune/quant.py`, `tests/test_quant*.py` | The image CI's path filter: a push rebuilds the training image, runs `docker/smoke_import.py`, `tests/test_quant.py` and `tests/test_quant_torchao.py` in it and tags it. `tests/test_infra.py` asserts the exact path set |
| `requirements-train.txt`, `docker/Dockerfile` (`IMAGE_INPUTS` in `vast/launch.py`) | Launch refuses an image built from other versions of them. Every line of `requirements-train.txt` is pinned exactly; keep `setuptools` below 81 (TensorBoard's `pkg_resources`) |
| `QUANT_CODE` in `vast/launch.py`: `kitsune/quant.py`, `tools/speed_probe.py`, `scripts/05_evaluate.py`, `kitsune/whisper.py`, `tools/whisper_eval.py`, `requirements-train.txt`, `docker/Dockerfile` | Launch refuses a box with quantised readouts unless box smoke-b passed checks 12-16 at an ancestor commit where all of these were byte-identical (`--allow-unverified-quant` turns that into a warning, the owner's call). They have changed since the last smoke-b, so the next such box needs a new smoke-b at its commit or the owner's `--allow-unverified-quant`; say so in the PR body when you touch them. Keep `QUANT_CODE` in step with renames |
| `configs/full/*.json` (generated by `tools/make_full_configs.py`) | Pinned by sha256 in `tests/test_full_configs.py` (`FULL_P01_SHA256`, `FULL_P03_SHA256`, `FULL_P005_SHA256`, `SMOKE_SHA256`, `H15_OTHER_CONFIGS_SHA256`) and held by `make_full_configs.py --check`. Never hand-edit or regenerate them; never edit the generated `_comment` strings in the generator |
| `configs/full/boxes.json` | The hand-written box registry: `make_full_configs.py --check` validates it (`fullrun.registry_problems`) and `tests/test_full_configs.py` checks its numbers. Launch reads it at the commit it rents |
| Augmentation code (`kitsune/trainset.py`, `acoustics.py`, `noise_bank.py`, `aed_cuts.py`) | `GOLDEN` in `tests/test_recipe_v5.py` pins augmented micro-batches draw for draw (and numpy 2.5.3's random streams): the released recipes depend on them. A change that moves a random draw must keep them, or re-record them with the reason in the commit |
| `kitsune/fullrun.py` `FROZEN_MANIFEST_SHA256`, `kitsune/prereg.py` (`RUNS`, `CALIB_IDS_SHA256`, the selection constants) | Must equal what `study/PREREG.json` froze (`tests/test_prereg.py`); `prereg.py` stays stdlib-only (`kitsune/extent.py` imports it) |
| `study/PREREG.json`, `study/data.json`, `configs/study/study-t06.json` | sha256-pinned in `tests/test_prereg.py` (`FROZEN_SHA256`). `kitsune/evaluate.py` reads `PREREG.json` at import. Never edit |
| `kitsune/parakeet.py` `PARAKEET_FILES`, `kitsune/parakeet_golden.json` | sha256 pins of the converted Parakeet teacher; only `tools/publish_parakeet.py` generates them |
| `scripts/01_prepare_data.py` `REVISIONS`, `scripts/02_teacher_pass.py` `MODEL_REVISION`, `scripts/03_build_student.py` `TEACHER_REVISION` | The upstream pins; 02's and 03's must be equal, and every 40-hex commit in `MODEL_CARD.md` must be one of them (`tests/test_infra.py`) |
| `tools/make_full_configs.py` `AED_CUTS_SHA256`, `NOISE_BANK_SHA256`, `RIR_BANK_SHA256` | Pins of data-repo files the configs name |
| `MODEL_CARD.md` | `tests/test_student.py` pins its wording (`kitsune/student.py` rewrites parts of it by regex), and on the owner's laptop compares `model_card()` with the uploaded study inits' READMEs (`KITSUNE_STUDY_STUDENTS`). Keep its path |
| Trainer config keys (`DEFAULTS` in `scripts/04_distill.py`) | Every saved run config (`runs/<run_id>/config.json`, `trainer.pt`) and every `configs/full/*.json` spells the old `DEFAULTS`, and `_merge` refuses unknown keys. To retire a key, move it to `RETIRED_KEYS` with the value that left its mode off (a block: a dict of its keys) and update `tests/test_retired_keys.py`; never just delete it |
| `vast/README.md` | `tests/test_full_configs.py` parses its box full-t launch line (the `git checkout --detach` line before it, the flags launch defines). Code cites its sections "step 2.3", "Full-data runs" and "onstart.sh": keep those names or change the citing strings and tests with them |
| `vast/onstart.sh`, `vast/watchdog.sh`, `vast/onstart_stub.sh` | ASCII with LF line ends (`.gitattributes` sets `*.sh eol=lf`); `onstart.sh` under 16 KB; `onstart_stub.sh` within `ONSTART_MAX_BYTES` (4,000) |
| `STUDENT_FILES` in `vast/launch.py` | The same list is `STUDENT_FILES` in `vast/bootstrap.sh`'s helper: change both |
| `docker/smoke_import.py` `TORCHAO_NAMES` | Must cover every torchao name `kitsune/quant.py` looks up (`tests/test_infra.py`) |
| A script's file name or path | Tests, `kitsune/quant.py`, `kitsune/study_queue.py`, `scripts/03c_build_ctc_student.py`, `scripts/05_evaluate.py` and `tools/whisper_eval.py` load scripts by path (`importlib.util.spec_from_file_location`; `tests/fixtures.py` `load_script`). Grep for the old name in code, `*.sh`, `*.json` (the registry's argv templates), tests and docs |

`scripts/05_evaluate.py` and `tools/whisper_eval.py` default to `--max-temp 80`: on a CUDA host without a readable GPU
temperature they refuse before any work. The box items pass `--max-temp 0`.

## Testing

On the owner's laptop, in Git Bash, one pytest process at a time, with a `--basetemp` of your own:

```bash
KITSUNE_REAL_DATA_ROOT=D:/kitsune-tmp/no-real-data KITSUNE_TEST_THREADS=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  python -m pytest tests/test_x.py tests/test_y.py -n 3 --dist loadfile -m "not slow" -q -p no:cacheprovider \
  --basetemp D:/kitsune-tmp/pytest-<name>
python tools/make_full_configs.py --check   # exit 0 and "<checkout>/configs/full: up to date" (backslashes on Windows)
```

- Run the test files your change touches, then the whole suite (`tests`) once before the merge, split into chunks
  that each finish in under about ten minutes. There is no pytest CI: the only workflow is the image build.
- `KITSUNE_REAL_DATA_ROOT` pointing at an empty directory makes the real-data tests skip; without it they read the
  checkout itself (one reads `D:/Shizu-ko-distill`). `KITSUNE_REQUIRE_REAL_DATA=1` turns those skips into failures
  when you do want them.
- `KITSUNE_TEST_THREADS` sets each xdist worker's torch and BLAS threads (`tests/conftest.py`). `--dist loadfile` keeps
  a file's module fixtures on one worker.
- The only marker is `slow` (tests that load the real 2B teacher); keep `-m "not slow"`.
- Shell-script tests need Git Bash (`tests/test_infra.py` `find_bash` skips WSL's `bash.exe`); the Linux-only ones take
  `KITSUNE_LINUX_BASH` or skip.
- `make_study_configs --check` and `python -m kitsune.prereg --check` no longer exist: `tests/test_prereg.py` pins the
  frozen study files instead.
- Keep test coverage of live code: when you remove a mode, port a test that also covers live code to the kept path.

## Code style

- Lines up to 120 columns.
- Every module opens with a prose docstring: what it is, why, its formats and exit codes, and a `Usage:` block of
  exact commands. Comments explain why, not what. Update docstrings, `--help` texts and usage blocks with the code.
- British spelling: quantised, licence (noun), initialised, normalised, behaviour.
- `encoding="utf-8"` on every text read and write. Shell scripts with LF line ends.
- Deterministic outputs: generated configs, exports and records carry no timestamps or machine paths and are
  byte-identical on a re-run (`quant.py` writes its JSON with LF on every OS).
- Document a CLI's exit codes in its docstring and keep them: 0 ok, 1 failure, 2 refused or bad arguments, plus the
  documented job codes (for example 3 for the trainer's throughput refusal, 65 for an integrity refusal in 02, 02b and
  the label box).
- Lazy heavy imports where a module must import cheaply: `kitsune/quant.py` (torch and the stdlib),
  `kitsune/fullrun.py`, `kitsune/heartbeat.py`, `kitsune/devslice.py` and `vast/finish.py` (stdlib only, each tested),
  `kitsune/prereg.py` (stdlib only), `scripts/04_distill.py` (heavy imports inside `main()`: DataLoader workers
  re-import the file).
- Naming: pipeline scripts `scripts/NN[x]_name.py`, the library in `kitsune/` (snake_case modules and functions,
  UPPER_CASE constants), laptop and box tools in `tools/`, the vast.ai lifecycle in `vast/`.
- When you remove a symbol, option, file or box name, grep the whole repository (code, `*.sh`, `*.json`, `*.md`,
  tests) for it, including string uses, and remove the dangling ones with no dead helpers left behind.

## Git workflow

- Work on a kebab-case branch, push it, open a pull request (`gh pr create`), wait for any checks
  (`gh pr checks --watch`), then merge with a merge commit: `gh pr merge --merge --delete-branch`. Never merge locally,
  fast-forward, squash or rebase-merge into `main`, never force-push `main`. If the owner says "do not PR yet", stop
  after the push.
- Tags go on the merge commit on `main`, after the merge.
- Commit as the repository's noreply identity (`Multysquid <263708878+Multysquid@users.noreply.github.com>`, already
  configured; commits are signed). Never pass `-c user.email`: GitHub rejects the private address (GH007).
- No `Co-Authored-By` trailer and no "Generated with Claude Code" (or similar) line on commits, merge commits, tags or
  pull request descriptions.
- Commit subjects are imperative and capitalised, without a final period; the body says why. Stage files by name, never
  `git add -A`.
