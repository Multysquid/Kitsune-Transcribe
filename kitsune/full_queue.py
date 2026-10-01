"""The full-data runs' box queue: one registry box (kitsune/fullrun.py, configs/full/boxes.json) from its store builds
to its last verified upload, on a shared GPU queue; plus the resume of a box on a new host.

A box (fullrun.BOX_NAMES: full-smoke = smoke A, p01 = box 1, full = box 2, smoke-b) is the list of registry items of
its spec, in registry order. FullQueue(study_queue.Queue) runs them with the study queue's process, state and upload
machinery (hooks H1-H5 of the build contract, 0.3), but never its plans: it never reads prereg.rules()["boxes"].

  items      stores (python -m kitsune.full_queue build-stores: the trainer's own train, eval and dev store builds,
             once per family, alone), train (scripts/04_distill.py, the run named after the item), readout (the M4
             readout of a train item by scripts/05_evaluate.py on the frozen study manifest), speed
             (tools/speed_probe.py), eval (a registry argv template: the quant readouts, Whisper, the compares). An item
             starts when every item it `needs` is done or not_needed; one that needs a failed or skipped item is skipped
  choice     per free GPU: (1) a ready readout whose training item last ran on this GPU (it follows its run, ahead of
             the next training item), (2) the first ready stores or train item in registry order, (3) the eval pool
             (readout, speed and eval items in registry order), only once no train item is waiting to start. Stores
             items take a GPU slot: no two store builds run at once (a store build takes no lock). A speed item has the
             host to itself (the study's "one model at a time on an idle host": speed_probe's --require-idle sees only
             its own GPU, and two probes would share one speed.json): it starts only when nothing runs and no upload is
             pending, nothing starts beside it and no upload runs under it, and the pool items after it wait for it. An
             item held by a transient refusal (check-resume exit 1, a failed Hub pull) is tried again a poll later,
             once per poll; the GPU takes the next ready item meanwhile
  per child  KITSUNE_HEARTBEAT=$KITSUNE_STATE/hb/<item> (touched by the queue at start; the child beats it) and
             KITSUNE_DEADLINE = the box deadline ($KITSUNE_STATE/deadline) - deadline_reserve_min (no deadline file: no
             variable). A fresh start of an item with max_hours that cannot end before that deadline is skipped
             (item_not_started) when droppable, else started anyway (no_start_overridden; the trainer's 4d cooldown
             shortens it). A readout is tested against the box deadline less readout_margin_s() instead (10 min before
             the watchdog's log sync): a run the 4d cooldown shortened ends inside deadline_reserve_min, and its M4
             readout (box 1's go/no-go number) must still run
  stalls     every poll: an item whose heartbeat is older than its stall_min (the registry's, default
             fullrun.STALL_MIN_DEFAULT) is killed (item_stalled: SIGCONT, SIGTERM, SIGKILL after kill_grace_s) and a
             train item resumes from its newest local full state; an item without a stall check (stall_min null) is
             killed after OVERRUN_FACTOR x max_hours (item_overrun). Both count toward the box's max_attempts
  failures   a trainer's exit 3 (its throughput floor) fails that item only; a summary error SmokeFailed or
             ResumeMismatch fails it at once; any other train or stores failure is retried up to max_attempts (a
             stores item that fails for good skips its dependants); a readout, speed or eval failure is recorded
             (readouts_failed) and never retried or stops the box
  heartbeat  the queue beats $KITSUNE_STATE/train_hb (the box watchdog's file) on every poll, and under beating() while
             it uploads, puts its summary or verdict, downloads weights or pulls a run (ctl_beat_path: never while a
             smoke's freeze_controller_hb fault holds it)
  uploads    every finished run dir and readout or eval out dir, and the speed dir after every speed item (its
             events.jsonl gets one line per speed item that ran), lean, then verified; each verified upload puts the
             summary (coalesced), so an item is `verified` there only once its output is on the Hub
  summary    $KITSUNE_STATE/queue_summary.json, put at full/box-<box>/queue_summary.json in the runs repo on every item
             start (coalesced: at most one put per summary_min_s) and end, when a training run's dir appears, and at the
             end: a new host's resume reads it (the source of truth of the box's run ids)
  faults     smoke boxes only (registry faults): sigstop / kill at a step once the attempt has a full state,
             wipe_run_dir after an event (the run dir moved to runs/_wiped/, its logs synced, the retry pulled back
             from the Hub as a new host would, then check-resume), deadline (the item's KITSUNE_DEADLINE = start +
             seconds), freeze_controller_hb (no train_hb beats for `seconds`). A fault-ended attempt never counts as a
             failure. A part never ends while a fired freeze window is open (hold_freeze, box 53693389): it holds,
             running no items and not beating, until the watchdog's alert is recorded or the window ends, bounded by
             its stop_at and the box deadline less deadline_reserve_min; a queue restart inside the window keeps it
             (register). Then smoke_verdict.json (build contract 5: the built-in checks 1-11 on a box with train items,
             plus the registry's verdict specs), put at full/box-<box>/smoke_verdict.json
Exit of `run` (vast/supervise.py decide_queue): 0 when no train or stores item failed and every non-droppable train
item is done and verified (-> destroy); EXIT_STOP 4 when a train or stores item failed for good or a non-droppable
train item did not finish (-> stop, the disk kept for the owner); 1 on a queue error or an upload that did not verify
(-> the supervisor restarts the queue, which resumes from queue.json). Never 3. A chain box also exits
EXIT_CHAIN_DESTROY 5: it ended before its last stage trained (a failed gate, a failed stage-2 bootstrap, box 1 no
longer fits), nothing unique is on its disk -> finish verifies what it put on the Hub, then destroys.

Chain boxes (contract addendum E; `run --box p01-chain`): ChainController runs the chain's parts one at a time, each an
unchanged FullQueue whose state lives in $KITSUNE_STATE/chain/<part>/ while the box-wide files (train_hb, deadline,
the watchdog's alerts, the download gate) stay in $KITSUNE_STATE (FullSettings box_state_dir; a part's deadline and
stop_at are its sub-deadline, past which it halts). Stage 1: the gate part (full-smoke, until gate_by), the watchdog's
mode file set to stop 3600, the automatic gate on its verdict checks 1-11 (chain_gate), then smoke-b (report only,
until the stage-1 sub-deadline). A failed gate exits 5. Else the stage-2 bootstrap (vast/bootstrap.sh with
KITSUNE_CHAIN_STAGE=2 on the full extent, stage 1's shards reused; timeouts from stage2_timeouts) runs as a
synchronous child, bounded by its own budget and by box 1's fit, and box p01 runs; its rc is the chain's. chain.json
records every step, so a restart goes on where it stopped; the chain summary is at full/box-<chain>/queue_summary.json.
`plan` prints the stages and their parts' items; `resume-pull` refuses a chain (exit 3; fullrun.chain_resume_hint).

Resume on a new host (launch --resume / --resume-reset <run_id> / --resume-set <run_id>:schedule.epochs=<E>): bootstrap
runs `resume-pull` before its paid rebuild. It reads the box's queue summary from the runs repo, pulls every started
run's logs and its newest full state (the scratch repo's timed state or a runs-repo full state, whichever is newer,
checked against the pointer's or the LFS sha256), and writes $KITSUNE_STATE/resume_plan.json. The queue adopts that
plan only when it has no queue.json yet: done items stay done, started runs resume, the rest start fresh; a reset or
set run gets `sets_once` (schedule.resume_reset=true and its sets), passed on every attempt until the run has logged
its resume_reset and a full state after it. Its readout then writes runs/m4-<run_id>-r<N>, so the first one on the Hub
stays. Exit 0 plan written, 3 refused (no summary, an unknown run id, a state that does not match its checksum, a
set-only run that is past its cooldown: use --resume-reset), 1 anything transient (bootstrap retries it).
  done       a train item is done when the box summary says so and its export checkpoints/step_<steps>/ is on the
             Hub; also when the run's own summary.json is complete with its export there (the box died between the
             trainer's end and the queue's item-end put), but only when that summary cannot be an earlier run's: no
             Hub state lies past its steps and, for a continuation, it counts more resume_resets than the run had
             before it. A readout, eval or speed item is done only when the summary also says `verified` (its out dir,
             or the speed dir, reached the Hub); the others start fresh
  continued  a reset or set run is a continuation, recorded in the box summary (items.<n>.continuation: run_id, reset,
             sets, resume_resets_before). A later plain --resume goes on from the newest Hub state that holds its
             reset (st.resume_resets past resume_resets_before); when none does yet, the continuation is applied again
             as it was launched (a reset from the pre_cooldown state, a set from the newest state, with its sets), so
             a host lost during a continuation never falls back to the first run's end
  outputs    resume-pull pulls the training runs only; a fresh item that reads an adopted done item's out dir
             ({out:<item>}), or the first speed item after an adoption (the speed dir's speed.json, which its upload
             would otherwise replace), pulls that dir from the runs repo first (a quant variant's weights excepted)

Usage (vast/supervise.py runs `run` for KITSUNE_JOB=full; KITSUNE_BOX, KITSUNE_OUT_REPO, KITSUNE_SCRATCH_REPO,
KITSUNE_STATE from the env):
  python -m kitsune.full_queue run --box p01 [--gpus 0,1]
  python -m kitsune.full_queue run --box p01-chain               # a chain box: its ChainController
  python -m kitsune.full_queue plan --box full                   # the registry items, in order, nothing started
  python -m kitsune.full_queue build-stores --config configs/full/full-p03.json [--eval-only] [--set k=v]
  python -m kitsune.full_queue resume-pull --box full --root /workspace/Kitsune-Transcribe
  python -m kitsune.full_queue check-resume --run-dir runs/full-p03-20260927T120000Z
Pure Python at import (stdlib, kitsune.fullrun, kitsune.heartbeat, kitsune.study_queue); the trainer (torch) only in the
build-stores and check-resume children, huggingface_hub only when the Hub is called.
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from kitsune import fullrun, heartbeat
from kitsune import study_queue as Q

REPO = Q.REPO
EXIT_OK, EXIT_FAIL, EXIT_STOP = 0, 1, 4  # run (vast/supervise.py decide_queue: 0 destroy, 4 stop, else restart)
EXIT_REFUSED = 3  # resume-pull and check-resume: a refusal no retry can fix
EXIT_THROUGHPUT = Q.EXIT_THROUGHPUT  # a trainer's throughput floor: that item fails, the box goes on
SUMMARY_MIN_S = 60.0  # queue summary puts at most this often, except the ones never delayed
KILL_GRACE_S = 60.0  # SIGTERM -> SIGKILL of a stalled or overrun item
CTL_BEAT_MAX_S = 2700  # the controller heartbeat's bound around one blocking Hub call (< the watchdog's orphan_s)
STORES_BEAT_EVERY_S, STORES_BEAT_MAX_S = 60, 36000  # build-stores: a slow full-extent build is never killed
CHECK_RESUME_TIMEOUT_S = 3600
SPEED_PER_SET, SPEED_SEED = 40, 1234  # the study's 200-id speed list on the manifest's eval rows
RUNNABLE = ("pending", "interrupted", "retry")
ENDED = ("done", "failed", "skipped", "not_needed")
POOL_KINDS = ("readout", "speed", "eval")
DETERMINISTIC = ("SmokeFailed", "ResumeMismatch")  # the same start fails the same way again: no retry
READOUT_METRICS = ("m4", "m4_teacher", "m4_ratio", "jg", "jg_nostyle", "gate_pooled")
SPEED_DIR_KEY = "_speed_dir"  # the speed items' shared out dir in the upload list
FULL_STATE_RE = re.compile(r"^full_step_(\d+)$")
STAMP = "%Y%m%dT%H%M%SZ"

# smoke A's built-in verdict checks (build contract 5)
SMOKE_NOSTART = "smoke-nostart"  # the item the no-start rule must skip
SMOKE_VRAM_GB = 28.0  # check 2: every memory number of every smoke run
SMOKE_MIN_STEPS = 300  # check 2: a VRAM window long enough (the deadline-fault item is exempt)
BOX1_H_MAX, BOX2_H_MAX, FACTOR_MIN = 17.0, 45.0, 0.6  # check 3: the projected box hours and plan factors
# check 5's loader rule: the steady-state share of the steps' time spent waiting on the loader, below DATA_WAIT_MAX
# (the build contract's 0.02, kept: the owner may move it to prereg.DATA_WAIT_MAX, the study's 0.05). Box 53693389
# (2026-10-01) failed check 5 on the whole-run summary.throughput.data_wait_frac (smoke-p005 0.234): every launch builds
# a new loader whose 8 spawned workers take ~6-7 s to start, its first step waits for them, and a short smoke that the
# faults restarted (smoke-p01 3 launches, smoke-p03 2) is dominated by those start-ups. Its steady state was 0.0007 to
# 0.0023 (the smoke profiler's own data_wait_share agrees). So the first LAUNCH_SKIP_STEPS steps after every launch's
# phase{name: "train"} and the smoke profiler's steps are left out (steady_wait); at least WAIT_MIN_STEPS steps must
# remain, and no launch's first step may wait STARTUP_WAIT_MAX_S or more (a loader that only starts slowly, as on a
# disk that cannot open its stores, still fails)
DATA_WAIT_MAX = 0.02
LAUNCH_SKIP_STEPS = 10
WAIT_MIN_STEPS = 50
STARTUP_WAIT_MAX_S = 120.0
FORCED_MIN_DELTA_ABS = 1e8  # check 10: a config whose early_stop.min_delta_abs is this large forces the trigger
SCRATCH_COMMITS_MAX = 2  # check 7: the scratch repo's history after its squashes
ALERT_SLACK_S = 300  # check 7: an alert counts for the freeze when it falls in its window + this
FREEZE = "freeze_controller_hb"
# the no-start rule of a readout (start_deadline): it must end this long before the watchdog's log sync, which runs
# KITSUNE_WATCHDOG_SYNC_LEAD_S (vast/watchdog.sh, default 600) before the box deadline
READOUT_SYNC_MARGIN_S = 600
WATCHDOG_SYNC_LEAD_S = 600


def readout_margin_s() -> float:
    """How long before the box deadline a readout must end: the watchdog's log-sync lead + READOUT_SYNC_MARGIN_S."""
    try:
        lead = float(os.environ.get("KITSUNE_WATCHDOG_SYNC_LEAD_S") or WATCHDOG_SYNC_LEAD_S)
    except ValueError:
        lead = WATCHDOG_SYNC_LEAD_S
    return lead + READOUT_SYNC_MARGIN_S


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [full-queue] {msg}", flush=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dig(obj, path: str):
    """obj["a"]["b"]["c"] for "a.b.c"; None when a key is missing."""
    for k in path.split("."):
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


class ResumeRefused(RuntimeError):
    """A resume no retry can fix (resume-pull exit 3; a wipe fault's pull: the item fails)."""


# ============================================================================================================ hub


class Hub:
    """One Hub repo (the runs repo or the scratch repo) as the resume reads it: a recursive listing with sizes and
    checksums, one JSON file, one file, a folder minus some patterns. Every call backs off like the queue's uploads
    (vast/finish.py hub_retry); a missing file or folder (404) is None / {} / an empty pull, not an error."""

    def __init__(self, repo: str, repo_type: str = "model", api=None):
        self.repo, self.repo_type, self._api = repo, repo_type, api

    def api(self):
        if self._api is None:
            self._api = Q._finish().hf_api()
        return self._api

    @staticmethod
    def _missing(e: BaseException) -> bool:
        return getattr(getattr(e, "response", None), "status_code", None) == 404 or isinstance(e, FileNotFoundError)

    def _call(self, fn, what: str):
        return Q._finish().hub_retry(fn, what)

    def listing(self, prefix: str) -> dict[str, dict]:
        """path -> {size, sha256 (LFS; None for a plain git file), blob_id} for every file under prefix."""
        try:
            items = self._call(lambda: list(self.api().list_repo_tree(self.repo, path_in_repo=prefix, recursive=True,
                                                                        repo_type=self.repo_type)),
                               f"list {self.repo}:{prefix}")
        except Exception as e:  # noqa: BLE001  a missing folder is an empty one
            if self._missing(e):
                return {}
            raise
        out = {}
        for x in items:
            if getattr(x, "size", None) is None:
                continue  # a folder
            lfs = getattr(x, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if sha is None and isinstance(lfs, dict):
                sha = lfs.get("sha256")
            out[x.path] = dict(size=int(x.size), sha256=sha, blob_id=getattr(x, "blob_id", None))
        return out

    def download(self, path: str, local_dir: Path) -> Path | None:
        """path into local_dir/path; None when the repo has no such file."""
        try:
            return Path(self._call(lambda: self.api().hf_hub_download(repo_id=self.repo, filename=path,
                                                                      repo_type=self.repo_type,
                                                                      local_dir=str(local_dir)),
                                   f"download {self.repo}:{path}"))
        except Exception as e:  # noqa: BLE001
            if self._missing(e):
                return None
            raise

    def read_json(self, path: str, scratch_dir: Path) -> dict | None:
        p = self.download(path, scratch_dir)
        if p is None:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            raise ResumeRefused(f"{self.repo}:{path} is not JSON ({e})") from None

    def pull_tree(self, prefix: str, local_dir: Path, ignore: list[str] | None = None) -> Path:
        """Every file under prefix/ but the ignored patterns into local_dir/<prefix>/."""
        self._call(lambda: self.api().snapshot_download(repo_id=self.repo, repo_type=self.repo_type,
                                                        local_dir=str(local_dir), allow_patterns=[f"{prefix}/*"],
                                                        ignore_patterns=list(ignore or [])),
                   f"pull {self.repo}:{prefix}")
        return Path(local_dir) / prefix

    def n_commits(self) -> int:
        return len(self._call(lambda: list(self.api().list_repo_commits(self.repo, repo_type=self.repo_type)),
                              f"commits {self.repo}"))


def _verify(p: Path, meta: dict) -> str | None:
    """Why a pulled file is not the one the listing or pointer describes (size, sha256, else its git blob id)."""
    size = p.stat().st_size
    if meta.get("size") is not None and int(meta["size"]) != size:
        return f"{p.name}: size {size}, expected {meta['size']}"
    if meta.get("sha256"):
        if _sha256_file(p) != meta["sha256"]:
            return f"{p.name}: sha256 differs"
    elif meta.get("blob_id") and Q._finish().git_blob_id(p) != meta["blob_id"]:
        return f"{p.name}: git blob id differs"
    return None


def _states_of(listing: dict[str, dict], run_id: str) -> dict[int, dict[str, dict]]:
    """step -> {file: meta} of the runs repo's runs/<run_id>/checkpoints/full_step_<N>/ files."""
    base = f"runs/{run_id}/checkpoints/"
    out: dict[int, dict[str, dict]] = {}
    for path, meta in listing.items():
        if not path.startswith(base):
            continue
        parts = path[len(base):].split("/")
        m = FULL_STATE_RE.match(parts[0]) if len(parts) == 2 else None
        if m:
            out.setdefault(int(m.group(1)), {})[parts[1]] = meta
    return out


def pull_run(run_id: str, root: Path, runs: Hub, scratch: Hub | None, *, pick: str = "newest",
             refuse_past_cooldown: bool = False, with_state: bool = True, weights_step: int | None = None,
             min_resets: int | None = None) -> dict:
    """A run as a new host needs it (resume-pull, and the smoke's wipe_run_dir fault): runs/<run_id>/* but its
    checkpoints from the runs repo, then (with_state) one full state into runs/<run_id>/checkpoints/full_step_<N>/
    (downloaded into full_step_<N>.tmp/, every file checked, then renamed):
      pick "newest"        the highest step of the scratch repo's timed state (its pointer, fullrun.pointer_problems)
                           and the runs repo's full_step_<N>/ holding trainer.pt
      pick "pre_cooldown"  the highest runs-repo full_step_<N> whose trainer.json says reason pre_cooldown (a reset)
    min_resets: only states whose trainer.json counts st.resume_resets >= min_resets (a continuation goes on only from
    a state that holds its reset; none: no state is pulled, the caller applies the continuation again).
    weights_step: also the export checkpoints/step_<N>/ (a done run whose readout or eval is still to run).
    refuse_past_cooldown: ResumeRefused when the chosen state's trainer.json has st.pre_cooldown_done or
    st.early_stop.triggered (a --resume-set run: epochs cannot be extended past a cooldown without the reset flag).
    Returns {state, step, source ("scratch" | "runs" | None), kitsune_sha, resume_resets (the chosen state's
    st.resume_resets)}. ResumeRefused on a malformed pointer, a file that does not match, or pick pre_cooldown without
    such a state."""
    root = Path(root)
    prefix = f"runs/{run_id}"
    stage = root / "cache" / "hub_pull" / f"{run_id}-{time.time_ns()}"
    try:
        runs.pull_tree(prefix, stage, ignore=[f"{prefix}/checkpoints/*"])
        src = stage / prefix
        if src.is_dir():
            shutil.copytree(src, root / prefix, dirs_exist_ok=True)
        (root / prefix).mkdir(parents=True, exist_ok=True)
        out = dict(state=None, step=None, source=None, kitsune_sha=None, resume_resets=None)
        listing = runs.listing(f"{prefix}/checkpoints") if (with_state or weights_step is not None) else {}
        if weights_step is not None:
            _fetch_files(runs, {p: m for p, m in listing.items()
                                if p.startswith(f"{prefix}/checkpoints/step_{weights_step}/")},
                         root / prefix / "checkpoints" / f"step_{weights_step}", stage, f"{prefix}/checkpoints/"
                         f"step_{weights_step}/")
        if not with_state:
            return out
        cands = []
        ptr = scratch.read_json(fullrun.scratch_pointer(run_id), stage) if (scratch is not None and pick == "newest") \
            else None
        if ptr is not None:
            probs = fullrun.pointer_problems(ptr)
            if not probs and ptr["run_id"] != run_id:
                probs = [f"pointer run_id {ptr['run_id']!r}, not {run_id!r}"]
            if probs:
                raise ResumeRefused(f"{scratch.repo}:{fullrun.scratch_pointer(run_id)} is malformed: {probs}")
            base = fullrun.scratch_state_dir(run_id, ptr["step"])
            cands.append(dict(step=int(ptr["step"]), source="scratch", hub=scratch, base=base,
                              files={f"{base}/{f}": m for f, m in ptr["files"].items()},
                              kitsune_sha=ptr.get("kitsune_sha")))
        states = _states_of(listing, run_id)

        def resets(c) -> int:  # the candidate's st.resume_resets, from its small trainer.json
            if "tj" not in c:
                c["tj"] = c["hub"].read_json(f"{c['base']}/trainer.json", stage) or {}
            return int((c["tj"].get("st") or {}).get("resume_resets") or 0)

        for step in sorted(states, reverse=True):
            files = states[step]
            if "trainer.pt" not in files:
                continue
            base = f"{prefix}/checkpoints/full_step_{step}"
            c = dict(step=step, source="runs", hub=runs, base=base,
                     files={f"{base}/{f}": m for f, m in files.items()}, kitsune_sha=None)
            if pick == "pre_cooldown":
                c["tj"] = runs.read_json(f"{base}/trainer.json", stage) or {}
                if c["tj"].get("reason") != "pre_cooldown":
                    continue
            if min_resets is not None and resets(c) < min_resets:
                continue
            cands.append(c)
            if pick == "pre_cooldown":
                break  # the highest one
        if pick == "pre_cooldown" and not cands:
            raise ResumeRefused(f"{runs.repo}:{prefix}/checkpoints has no pre_cooldown full state to reset from")
        if min_resets is not None:
            cands = [c for c in cands if resets(c) >= min_resets]
        if not cands:
            return out
        best = max(cands, key=lambda c: (c["step"], c["source"] == "runs"))
        dest = root / prefix / "checkpoints" / f"full_step_{best['step']}"
        _fetch_files(best["hub"], best["files"], dest, stage, best["base"] + "/")
        st = (_read_json(dest / "trainer.json") or {}).get("st") or {}
        if refuse_past_cooldown and (st.get("pre_cooldown_done") or (st.get("early_stop") or {}).get("triggered")):
            raise ResumeRefused(f"{run_id}: its newest state ({dest.name}) is past its cooldown or early stop; "
                                f"a --resume-set alone cannot extend it: use --resume-reset {run_id}")
        out.update(state=dest.name, step=best["step"], source=best["source"], kitsune_sha=best["kitsune_sha"],
                   resume_resets=int(st.get("resume_resets") or 0))
        return out
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _fetch_files(hub: Hub, files: dict[str, dict], dest: Path, stage: Path, base: str):
    """files (repo path -> meta) under `base` into dest/<rest>, through dest.tmp: every file checked, then renamed in
    (an existing dest is replaced). ResumeRefused on a missing or mismatching file."""
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for path, meta in sorted(files.items()):
            got = hub.download(path, stage)
            if got is None:
                raise ResumeRefused(f"{hub.repo}:{path} is listed but cannot be downloaded")
            why = _verify(got, meta)
            if why:
                raise ResumeRefused(f"{hub.repo}:{path}: {why}")
            target = tmp / path[len(base):]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), str(target))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # never a half-pulled state next to the run
        raise
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)


# ============================================================================================================ queue


@dataclass
class FullSettings(Q.Settings):
    """The study queue's Settings plus the full box's: the scratch repo, the registry, the host's ids, the Hub readers
    and the timing knobs the tests shrink."""
    scratch_repo: str | None = field(default_factory=lambda: os.environ.get(fullrun.ENV_SCRATCH_REPO) or None)
    registry: dict | None = None  # None: configs/full/boxes.json of root ($KITSUNE_FULL_REGISTRY in the tests)
    machine_id: str | None = field(default_factory=lambda: os.environ.get(fullrun.ENV_MACHINE_ID) or None)
    container_id: str | None = field(default_factory=lambda: os.environ.get("CONTAINER_ID") or None)
    sha: str | None = field(default_factory=lambda: os.environ.get(fullrun.ENV_SHA) or None)
    check_resume_cmd: list[str] | None = None  # default: python -m kitsune.full_queue check-resume
    runs_hub: object = None  # a Hub over the runs repo (default Hub(out_repo)): the wipe fault's pull, the verdict
    scratch_hub: object = None  # a Hub over the scratch repo (default Hub(scratch_repo))
    kill_grace_s: float = KILL_GRACE_S
    summary_min_s: float = SUMMARY_MIN_S
    proc_root: Path = field(default_factory=lambda: Path("/proc"))
    cgroup: Path = field(default_factory=lambda: Path(os.environ.get(fullrun.ENV_CGROUP) or "/sys/fs/cgroup"))
    # a chain part (ChainController): the box's own state dir, where the box-wide files live (train_hb, deadline,
    # watchdog_alerts.jsonl, download_gate.json) while the part's queue state stays in state_dir (<box state>/chain/
    # <part>); None: state_dir. deadline replaces the deadline file's (the part's sub-deadline); past stop_at the
    # queue halts (rc 4, "halted": the running items are stopped, the verdict written)
    box_state_dir: Path | None = None
    deadline: float | None = None
    stop_at: float | None = None


class FullQueue(Q.Queue):
    """One registry box (the module docstring). It overrides the study queue's run, _ready, _next_for, resume_point,
    finished, write_summary and drain_uploads and never calls their base versions; start, execute, stop_all,
    kill_orphans, set_aside, find_run_dir and ensure_attribution are the study queue's."""

    def __init__(self, box: str, settings: FullSettings | None = None, registry: dict | None = None):
        s = settings or FullSettings()
        reg = registry if registry is not None else s.registry
        try:
            reg = fullrun.load_registry(reg, root=s.root, check_files=False) if reg is not None else \
                fullrun.load_registry(None, root=s.root)
            if fullrun.is_chain(box, reg):
                raise fullrun.RegistryError(f"box {box} is a chain box: a chain box runs through the chain controller "
                                            f"(ChainController; `python -m kitsune.full_queue run --box {box}`)")
            self.spec = fullrun.box_spec(box, reg)
        except fullrun.RegistryError as e:
            raise Q.QueueError(str(e)) from None
        self.registry = reg
        self.box_state = Path(s.box_state_dir or s.state_dir)  # train_hb, deadline, alerts, download gate
        self.registry_sha256 = hashlib.sha256(json.dumps(reg, sort_keys=True, separators=(",", ":"),
                                                         ensure_ascii=False).encode()).hexdigest()
        self.items_spec = {it["name"]: it for it in self.spec["items"]}
        self.order = [it["name"] for it in self.spec["items"]]
        self._fresh = True
        trains = [n for n in self.order if self.items_spec[n]["kind"] == "train"]
        plan = dict(box=box, kind=fullrun.JOB, shared_queue=True, runs=trains, calibrate=[], probe_classes=[],
                    reference=None, numbers_file=None, numbers_from=None, extras=[])
        super().__init__(box, s, plan=plan)
        if len(self.gpus) != int(self.spec["gpus"]):
            raise Q.QueueError(f"box {box} is registered with {self.spec['gpus']} GPU(s); the queue found "
                               f"{len(self.gpus)}: {self.gpus}")
        if self.spec["timed_states"] and self.s.out_repo and not self.s.scratch_repo:
            raise Q.QueueError(f"box {box} keeps timed full states: {fullrun.ENV_SCRATCH_REPO} (launch --scratch-repo) "
                               f"is required")
        self.max_attempts = int(self.spec["max_attempts"])
        self._tails: dict[str, Q.StepTail] = {}
        self._last_put, self._put_pending = 0.0, False
        self._freeze_until = 0.0
        self._runs_hub, self._scratch_hub = self.s.runs_hub, self.s.scratch_hub

    # -------------------------------------------------------------------------------------------------- state

    def _load_state(self) -> dict:
        fresh = dict(box=self.box, kind=fullrun.JOB, version=1, registry_sha256=self.registry_sha256,
                     started=time.time(), items={}, final=None, abandoned=[], readouts_failed={}, no_start={},
                     not_needed={}, faults={}, resumed=None, host_mem_peak_gb=None, speed_dir=None)
        if not self.state_path.is_file():
            self._fresh = True
            return fresh
        st = json.loads(self.state_path.read_text(encoding="utf-8"))
        if st.get("kind") != fullrun.JOB:
            raise Q.QueueError(f"{self.state_path} is a {st.get('kind') or 'study'} queue's state, not a full box's")
        if st.get("box") != self.box:
            raise Q.QueueError(f"{self.state_path} is box {st.get('box')!r}'s queue, not {self.box!r}'s")
        self._fresh = False
        return {**fresh, **st}

    def _cmd(self, kind: str) -> list[str]:
        py = self.s.python
        if kind == "stores":
            return list(self.s.stores_cmd or [py, "-m", "kitsune.full_queue", "build-stores"])
        if kind == "check-resume":
            return list(self.s.check_resume_cmd or [py, "-m", "kitsune.full_queue", "check-resume"])
        return super()._cmd(kind)

    def spec_of(self, name: str) -> dict:
        return self.items_spec[name]

    def kind_names(self, *kinds: str) -> list[str]:
        return [n for n in self.order if self.items_spec[n]["kind"] in kinds]

    def runs_hub(self) -> Hub | None:
        if self._runs_hub is None and self.s.out_repo:
            self._runs_hub = Hub(self.s.out_repo)
        return self._runs_hub

    def scratch_hub(self) -> Hub | None:
        if self._scratch_hub is None and self.s.scratch_repo:
            self._scratch_hub = Hub(self.s.scratch_repo)
        return self._scratch_hub

    def ctl_beat_path(self) -> Path | None:
        """The controller heartbeat ($KITSUNE_STATE/train_hb), or None while a freeze_controller_hb fault holds it:
        every controller beat goes through here, so beat() and beating() are no-ops for that window."""
        if time.time() < self._freeze_until:
            return None
        return self.box_state / fullrun.TRAIN_HB

    def ctl_beating(self, max_s: float = CTL_BEAT_MAX_S):
        """heartbeat.beating on the controller heartbeat around one blocking call (bounded: a hung call still goes
        stale for the watchdog); a plain context while a freeze fault holds it."""
        p = self.ctl_beat_path()
        return heartbeat.beating(p, max_s=max_s) if p is not None else contextlib.nullcontext()

    def box_deadline(self) -> float | None:
        """The watchdog's stop time ($KITSUNE_STATE/deadline, vast/onstart.sh; a chain part: its settings' deadline);
        None without one (a local run)."""
        if self.s.deadline is not None:
            return float(self.s.deadline)
        try:
            return float((self.box_state / fullrun.DEADLINE_FILE).read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def item_deadline(self, name: str) -> float | None:
        """The item's KITSUNE_DEADLINE: a fired deadline fault's (its start + seconds), else the box deadline less the
        box's deadline_reserve_min (the drain and finish.py's upload fit in that reserve); None without one. The
        no-start rule tests every item but the readouts against it (build contract 5, Resolution 17; start_deadline)."""
        for f in self.spec["faults"]:
            fs = self.state["faults"].get(f["id"]) or {}
            if f["action"] == "deadline" and f["item"] == name and fs.get("deadline") is not None:
                return float(fs["deadline"])
        dl = self.box_deadline()
        return None if dl is None else dl - float(self.spec["deadline_reserve_min"]) * 60

    def start_deadline(self, name: str) -> float | None:
        """The no-start rule's deadline. A readout (the M4 readout of a training item: box 1's go/no-go number) must end
        readout_margin_s() before the box deadline - 10 min before the watchdog's log sync - not inside the item
        deadline's deadline_reserve_min: a run that the trainer's deadline cooldown (4d) shortened ends inside that
        reserve, and a readout tested against it would be skipped, leaving the box without its number. Every other item
        (eval, speed, stores, train): item_deadline."""
        if self.spec_of(name)["kind"] == "readout":
            dl = self.box_deadline()
            return None if dl is None else dl - readout_margin_s()
        return self.item_deadline(name)

    # ------------------------------------------------------------------------------------------ registration

    def register(self):
        """Every registry item, once (a restart finds them registered with their status); then, on a new host (no
        queue.json yet, a resume_plan.json from resume-pull), the plan adopted; then only_if_new_machine."""
        for name in self.order:
            spec = self.spec_of(name)
            new = name not in self.state["items"]
            self.add_item(name, spec["kind"], spec.get("config"), list(spec.get("sets") or [])
                          if spec["kind"] == "stores" else [], run=spec.get("study_run"))
            if new:
                self.item(name).update(needs=list(spec["needs"]), of=spec.get("of"), out=None, hb_max_gap_s=None,
                                       stalls=0, peak_rss_gb=None, hub_resume=False, sets_once=[], check_resume=None,
                                       check_resume_tries=0, held_until=None, continuation=None, why=None)
        plan_path = Path(self.s.state_dir) / fullrun.RESUME_PLAN
        if self._fresh and plan_path.is_file() and self.state.get("resumed") is None:
            self.adopt(json.loads(plan_path.read_text(encoding="utf-8")))
        for f in self.spec["faults"]:
            self.state["faults"].setdefault(f["id"], dict(id=f["id"], action=f["action"], item=f["item"],
                                                          fired_at=None, outcome=None))
        now = time.time()
        for f, fs in self._open_freezes():  # a restart inside a fired freeze window keeps it (_freeze_until is not
            if float(fs["window_end"]) > now:  # in queue.json): the rest of the window, then hold_freeze, as before
                self._freeze_until = max(self._freeze_until, float(fs["window_end"]))
                self.event("freeze_restored", id=f["id"], window_end=fs["window_end"],
                           left_s=round(float(fs["window_end"]) - now, 1))
        self.machine_checks()
        self.save()

    def adopt(self, plan: dict):
        """Seed the items from $KITSUNE_STATE/resume_plan.json (resume-pull on this new host): done items done and
        verified (their run dirs are on the Hub), resumed runs interrupted in their pulled run dir (check-resume
        before their first start), the rest fresh with a new attempt budget. Every reset / set run id (the plan's,
        and KITSUNE_RESUME_RESET / KITSUNE_RESUME_SETS) gets its sets_once and a `continuation` record (the box
        summary's: a later resume-pull must not take the first run's complete summary.json for this run's end); a
        resumed run that is a continuation already keeps the plan's record. A done speed item keeps its speed dir,
        pulled from the Hub before the next speed item writes into it (_prepare_eval)."""
        if plan.get("box") != self.box:
            raise Q.QueueError(f"{fullrun.RESUME_PLAN} is box {plan.get('box')!r}'s, not {self.box!r}'s")
        env_reset = fullrun.parse_resume_reset(os.environ.get(fullrun.ENV_RESUME_RESET))
        want = {rid: ["schedule.resume_reset=true"] for rid in env_reset}
        for rid, sets in fullrun.parse_resume_sets(os.environ.get(fullrun.ENV_RESUME_SETS)).items():
            want[rid] = ["schedule.resume_reset=true", *sets]
        seeded = {}
        for name, e in (plan.get("items") or {}).items():
            if name not in self.state["items"]:
                self.event("resume_plan_item_unknown", item=name)
                continue
            it = self.item(name)
            if e.get("status") == "done":
                it.update(status="done", verified=True, run_dir=e.get("run_dir"), out=e.get("out"),
                          result=e.get("result"))
            elif e.get("status") == "resume" and it["kind"] == "train":
                it.update(status="interrupted", run_dir=e.get("run_dir"), check_resume="pending")
            else:
                it.update(status="pending", run_dir=None, out=None, result=None)
            if it["kind"] == "train" and it["run_dir"]:
                rid = fullrun.run_id_of(it["run_dir"])
                sets = list(e.get("sets") or []) or want.get(rid, [])
                if rid in want and sets != want[rid]:
                    self.event("resume_sets_differ", item=name, plan=sets, env=want[rid])
                if sets and it["status"] != "done":
                    it["sets_once"] = sets
                    before = e.get("resume_resets_before")
                    it["continuation"] = dict(
                        run_id=rid, reset=bool(e.get("reset")) or rid in env_reset, sets=sets,
                        resume_resets_before=int(before if before is not None else self._local_resets(it["run_dir"])),
                        adopted_utc=_now_utc())
                elif (e.get("continuation") or {}).get("run_id") == rid and it["status"] != "done":
                    it["continuation"] = dict(e["continuation"])
            if it["kind"] == "speed" and it["status"] == "done" and it["out"] and not self.state["speed_dir"]:
                self.state["speed_dir"] = dict(run_dir=it["out"], verified=True, adopted=True, pulled=False)
            seeded[name] = it["status"]
        self.state["resumed"] = dict(plan_created_utc=plan.get("created_utc"),
                                     summary_sha256=plan.get("summary_sha256"), items=seeded, wall=time.time())
        self.event("queue_resume_adopted", items=seeded)

    def _local_resets(self, run_dir: str) -> int:
        """st.resume_resets of the run dir's newest local full state (trainer.json); 0 without one."""
        ck = self.root / run_dir / "checkpoints"
        steps = [int(m.group(1)) for p in (ck.iterdir() if ck.is_dir() else ())
                 if (m := FULL_STATE_RE.match(p.name)) and (p / "trainer.json").is_file()]
        if not steps:
            return 0
        tj = _read_json(ck / f"full_step_{max(steps)}" / "trainer.json") or {}
        return int((tj.get("st") or {}).get("resume_resets") or 0)

    def machine_checks(self):
        """only_if_new_machine: a speed item that re-times what another box timed runs only on another machine
        (that box's queue summary's machine_id; unreadable -> it runs)."""
        for name in self.kind_names("speed"):
            spec, it = self.spec_of(name), self.item(name)
            other = spec.get("only_if_new_machine")
            if not other or it["status"] != "pending" or it["attempts"] or not self.s.machine_id:
                continue
            summ = self.read_runs_json(fullrun.box_summary_path(other))
            mid = (summ or {}).get("machine_id")
            if mid is not None and mid == self.s.machine_id:
                it["status"] = "not_needed"
                self.state["not_needed"][name] = dict(reason="same_machine", machine_id=mid, box=other)
                self.event("item_not_needed", item=name, reason="same_machine", machine_id=mid)

    def read_runs_json(self, path: str) -> dict | None:
        """A JSON file of the runs repo through the uploader (another box's queue summary); None when unreadable."""
        if self.uploader is None:
            return None
        dest = Path(self.s.state_dir) / "hub_reads"
        try:
            with self.ctl_beating():
                return json.loads(Path(self.uploader.download(path, dest)).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001  unreadable: the caller's default
            log(f"{path}: not readable from the runs repo ({type(e).__name__}: {e})")
            return None

    # -------------------------------------------------------------------------------------------- the run

    def run(self) -> int:
        final = self.state.get("final")
        if final:
            log(f"box {self.box} already ended ({final['status']}: {final.get('reason')}); nothing to do")
            return int(final["rc"])
        self.kill_orphans()
        self.register()
        if self.state.get("registry_sha256") != self.registry_sha256:
            self.event("registry_changed", was=self.state.get("registry_sha256"), now=self.registry_sha256)
            self.state["registry_sha256"] = self.registry_sha256
        self._pending_uploads = [n for n in self.order if self.item(n)["status"] == "done"
                                 and self.item(n)["verified"] is not True and self._upload_dir(n)]
        sd = self.state.get("speed_dir")
        if sd and ((sd.get("written") and sd.get("verified") is not True) or any(
                self.item(n)["status"] == "done" and self.item(n)["verified"] is not True
                for n in self.kind_names("speed"))):
            self._pending_uploads.append(SPEED_DIR_KEY)
        self.event("queue_start", gpus=self.gpus, items=self.order, restart=not self._fresh,
                   adopted=self.state.get("resumed") is not None, registry_sha256=self.registry_sha256)
        rc, status, reason = EXIT_OK, "complete", None
        try:
            self.put_summary(force=True)
            self.execute(list(self.order), monitor=self.monitor)
            self.drain_uploads()
            self.hold_freeze()
            self.end_faults()
            rc, status, reason = self.outcome()
        except Q.Halt as e:
            rc, status, reason = EXIT_STOP, "halted", str(e)
        except Q.QueueError as e:
            rc, status, reason = EXIT_FAIL, "failed", str(e)
        finally:
            self.stop_all()
        if status != "failed":  # a failure is left open: the supervisor restarts the queue, which goes on from here
            self.state["final"] = dict(status=status, reason=reason, rc=rc, wall=time.time())
        self.save()
        self.event("queue_end", status=status, reason=reason, rc=rc)
        self.write_summary(status, reason, rc)
        if self.spec["smoke"]:
            self.write_verdict()
        return rc

    def outcome(self) -> tuple[int, str, str | None]:
        """(rc, status, reason) once every runnable item has ended (the module docstring's exit rule)."""
        items = self.state["items"]
        failed = [n for n in self.kind_names("train", "stores") if items[n]["status"] == "failed"]
        unfinished = [n for n in self.kind_names("train") if not self.spec_of(n)["droppable"]
                      and items[n]["status"] != "done"]
        unverified = [n for n in self.order if items[n]["status"] == "done" and items[n]["verified"] is False]
        if self.state.get("speed_dir") and self.state["speed_dir"].get("verified") is False:
            unverified.append("speed")
        readouts = self.state["readouts_failed"]
        if failed:
            return EXIT_STOP, "halted", f"items failed for good (the disk is kept): {failed}"
        if unfinished:
            return EXIT_STOP, "halted", f"training items that did not finish: {unfinished}"
        if unverified:
            return EXIT_FAIL, "failed", f"not verified on the Hub: {unverified}"
        return EXIT_OK, "complete", (f"complete; readouts failed (redo them from the Hub): {sorted(readouts)}"
                                     if readouts else None)

    # ---------------------------------------------------------------------------------------------- choice

    def _ready(self, name: str) -> bool:
        """Runnable and every `needs` done or not_needed; a need that failed or was skipped skips this item."""
        it = self.item(name)
        if it["status"] not in RUNNABLE:
            return False
        for n in it.get("needs") or []:
            st = self.item(n)["status"]
            if st in ("failed", "skipped"):
                self._end(name, "skipped", "item_skipped", reason=f"needs {n}, which is {st}")
                return False
            if st not in ("done", "not_needed"):
                return False
        return True

    def _next_for(self, gpu: str, todo: list[str], running: dict) -> str | None:
        """The module docstring's choice for this GPU: its training item's readout, then the first ready stores or
        train item, then the eval pool once no training item waits to start. A running speed item holds the whole
        host; a speed item starts only on an idle host (nothing running, no upload pending), and the pool items after
        it wait for it (so it is never starved). An item held by a transient refusal is passed over until its hold
        ends: this GPU takes the next ready item."""
        if running or any(self.item(n)["status"] in RUNNABLE for n in self.order):  # a part that ended is not halted
            self.stage_deadline_check()
        busy = {r["name"] for r in running.values()}
        if any(self.item(n)["kind"] == "speed" for n in busy):
            return None

        def candidate(name: str) -> bool:
            return name not in busy and not self._held(name) and self._ready(name)

        for name in self.kind_names("readout"):
            of = self.item(self.item(name)["of"])
            last = (of["attempts"] or [{}])[-1].get("gpu")
            if last == str(gpu) and candidate(name) and self._prepare(name):
                return name
        for name in self.kind_names("stores", "train"):
            if candidate(name) and self._prepare(name):
                return name
        if any(self.item(n)["status"] in RUNNABLE for n in self.kind_names("train")):
            return self._none(running)
        for name in self.kind_names(*POOL_KINDS):
            if not candidate(name):
                continue
            if self.item(name)["kind"] == "speed" and (running or self._pending_uploads):
                return self._none(running)  # it waits for an idle host; nothing after it starts first
            if self._prepare(name):
                return name
        return self._none(running)

    def stage_deadline_check(self):
        """A chain part past its stop_at (FullSettings): event stage_deadline, then Halt. execute() stops the running
        items (SIGTERM, SIGKILL after 60 s: interrupted) and run() records rc 4, "halted", and writes the verdict."""
        if self.s.stop_at is not None and time.time() >= float(self.s.stop_at):
            self.event("stage_deadline", stop_at=float(self.s.stop_at))
            raise Q.Halt(f"stage deadline {datetime.fromtimestamp(float(self.s.stop_at), timezone.utc).isoformat()} "
                         f"passed")

    def _held(self, name: str) -> bool:
        return time.time() < float(self.item(name).get("held_until") or 0)

    def _none(self, running: dict) -> None:
        """_next_for's None. With nothing running, execute() loops without its poll sleep: while an item is held,
        wait here until its hold ends (at most a poll), so a held item is tried once per poll and the loop never
        spins."""
        if not running:
            ends = [float(it.get("held_until") or 0) for it in self.state["items"].values()
                    if it["status"] in RUNNABLE and float(it.get("held_until") or 0) > time.time()]
            if ends:
                time.sleep(max(0.0, min(min(ends) - time.time(), self.s.poll_s)))
        return None

    def _end(self, name: str, status: str, kind: str, **fields):
        """An item that ends without running (skipped, failed before its start, not_needed)."""
        it = self.item(name)
        it["status"] = status
        it["why"] = fields.get("reason") or fields.get("why")
        if status == "failed" and it["kind"] in POOL_KINDS:
            self.state["readouts_failed"][name] = it["why"]
        self.event(kind, item=name, **fields)
        self._fault_outcomes(name)
        self.save()
        self.put_summary(force=True)

    def _prepare(self, name: str) -> bool | None:
        """Everything an item needs before its start. True: start it; False: it ended (skipped or failed); None: not
        now (a transient check: this GPU waits a poll)."""
        it, spec = self.item(name), self.spec_of(name)
        if not it["attempts"] and it["status"] == "pending" and spec.get("max_hours"):
            dl = self.start_deadline(name)
            if dl is not None:
                need, left = float(spec["max_hours"]) * 3600, dl - time.time()
                if need > left:
                    rec = dict(need_s=round(need), left_s=round(left))
                    if spec["droppable"]:
                        self.state["no_start"][name] = dict(rec, skipped=True)
                        self._end(name, "skipped", "item_not_started", **rec)
                        return False
                    if name not in self.state["no_start"]:
                        self.state["no_start"][name] = dict(rec, overridden=True)
                        self.event("no_start_overridden", item=name, **rec)
        if spec["kind"] == "train":
            return self._prepare_train(name)
        if spec["kind"] == "readout":
            return self._prepare_readout(name)
        if spec["kind"] in ("speed", "eval"):
            return self._prepare_eval(name)
        return True

    def _hold(self, name: str, why: str) -> bool | None:
        """A transient refusal before an item's start: held for a poll (_next_for passes it over and takes the next
        ready item meanwhile), tried again then, max_attempts times in all, then failed."""
        it = self.item(name)
        it["check_resume_tries"] = int(it.get("check_resume_tries") or 0) + 1
        self.event("item_held", item=name, why=why, tries=it["check_resume_tries"])
        if it["check_resume_tries"] >= self.max_attempts:
            self._end(name, "failed", "item_failed", reason=f"{why} ({it['check_resume_tries']} tries)")
            return False
        it["held_until"] = time.time() + self.s.poll_s
        self.save()
        return None

    def _prepare_train(self, name: str) -> bool | None:
        it = self.item(name)
        if it.get("hub_resume"):  # the wipe fault: pull the run back as a new host would
            rid = fullrun.run_id_of(it["run_dir"])
            runs, scratch = self.runs_hub(), self.scratch_hub()
            if runs is None:
                self._fault_failed(name)
                self._end(name, "failed", "item_failed", reason="a Hub resume needs the runs repo (KITSUNE_OUT_REPO)")
                return False
            try:
                with self.ctl_beating():
                    got = pull_run(rid, self.root, runs, scratch)
            except ResumeRefused as e:
                self._fault_failed(name)
                self._end(name, "failed", "item_failed", reason=f"hub resume refused: {e}")
                return False
            except Exception as e:  # noqa: BLE001  transient: a poll later
                return self._hold(name, f"hub resume pull failed: {type(e).__name__}: {e}")
            it.update(hub_resume=False, check_resume="pending")
            self.event("hub_resume", item=name, run_id=rid, **got)
            if got["state"] is None:
                self._fault_failed(name)
                self._end(name, "failed", "item_failed", reason=f"hub resume: no full state of {rid} on the Hub")
                return False
        if it.get("check_resume") == "pending" and it["run_dir"]:
            rc = self.check_resume(name)
            if rc == 0:
                it["check_resume"] = "ok"
                it["check_resume_tries"], it["held_until"] = 0, None
            elif rc == EXIT_REFUSED:
                it["check_resume"] = "refused"
                self._fault_failed(name)
                self._end(name, "failed", "item_failed", reason="check-resume refused the pulled state (the store's "
                                                                "planner fingerprint or size differs)")
                return False
            else:
                return self._hold(name, f"check-resume exit {rc}")
        return True

    def check_resume(self, name: str) -> int:
        """python -m kitsune.full_queue check-resume --run-dir <run dir> (the trainer's resume_check, CPU), under the
        controller heartbeat; its output goes to the item's log."""
        it = self.item(name)
        argv = self._cmd("check-resume") + ["--run-dir", it["run_dir"]]
        logs = Path(self.s.state_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, **self.s.env, CUDA_VISIBLE_DEVICES="-1", KITSUNE_QUEUE_ITEM=name)
        with open(logs / f"{name}.log", "ab") as out, self.ctl_beating():
            try:
                rc = subprocess.run(argv, cwd=str(self.root), env=env, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, timeout=CHECK_RESUME_TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                rc = 1
        it.setdefault("check_resume_runs", []).append(dict(rc=rc, wall=time.time()))
        self.event("check_resume", item=name, run_dir=it["run_dir"], rc=rc)
        return rc

    def _prepare_readout(self, name: str) -> bool:
        it = self.item(name)
        of = self.item(it["of"])
        res = of.get("result") or {}
        rd, steps = of.get("run_dir"), res.get("steps")
        ckpt = f"{rd}/checkpoints/step_{steps}" if rd and steps is not None else None
        if ckpt is None or not (self.root / ckpt).is_dir():
            self._end(name, "failed", "item_failed", reason=f"no {ckpt or 'final checkpoint'} of {it['of']}: the "
                                                           f"readout has no fallback checkpoint")
            return False
        if not it["attempts"]:  # the out dir: the run id, and -r<N> after a continuation (the first stays on the Hub)
            n = int(res.get("resume_resets") or 0)
            it["out"] = f"runs/m4-{fullrun.run_id_of(rd)}" + (f"-r{n}" if n >= 1 else "")
        return True

    def _prepare_eval(self, name: str) -> bool:
        it, spec = self.item(name), self.spec_of(name)
        if spec.get("of_box"):
            summ = self.read_runs_json(fullrun.box_summary_path(spec["of_box"]))
            res = (((summ or {}).get("items") or {}).get(spec["of"]) or {}).get("result") or {}
            if not res.get("run_id") or res.get("steps") is None:
                self._end(name, "skipped", "item_skipped", reason="of_box unresolved", of_box=spec["of_box"],
                          of=spec["of"])
                return False
            it["source"] = dict(box=spec["of_box"], run_id=res["run_id"], steps=int(res["steps"]))
        pairs = [(w["run_id"], int(w["step"])) for w in spec.get("weights") or []]
        if it.get("source"):
            pairs.append((it["source"]["run_id"], it["source"]["steps"]))
        if pairs:
            why = self.fetch_weights(pairs)
            if why:
                self._end(name, "failed", "item_failed", reason=why)
                return False
        if spec["kind"] == "eval" and not it["out"]:
            it["out"] = f"runs/{name}-{time.strftime(STAMP, time.gmtime())}"
        why = None
        if spec["kind"] == "speed":
            it["out"] = self.speed_dir()
            why = self.pull_speed_dir()
        why = why or self.pull_inputs(name)
        if why:  # a fresh item never runs on a missing input, nor writes a speed.json that would replace the Hub's
            self._end(name, "failed", "item_failed", reason=why)
            return False
        try:  # every placeholder fillable now: a registry template that is not fails this item, not the queue
            self.argv_for(name, it, None)
        except Q.QueueError as e:
            self._end(name, "failed", "item_failed", reason=str(e))
            return False
        return True

    def pull_out_dir(self, rel: str, ignore: tuple[str, ...] = ()) -> str | None:
        """An out dir that is on the Hub but not on this host (an item done on an earlier host: resume-pull pulls only
        the training runs) from the runs repo into <root>/<rel> (through a stage dir), under the controller heartbeat;
        ignore: patterns under it left on the Hub. Why not (None: pulled, or the Hub has no such dir)."""
        hub = self.runs_hub()
        if hub is None:
            return f"{rel} is not on this host and there is no runs repo ({fullrun.ENV_OUT_REPO}) to pull it from"
        stage = self.root / "cache" / "hub_pull" / f"{Path(rel).name}-{time.time_ns()}"
        try:
            with self.ctl_beating():
                hub.pull_tree(rel, stage, ignore=[f"{rel}/{p}" for p in ignore])
            found = (stage / rel).is_dir()
            if found:
                shutil.copytree(stage / rel, self.root / rel, dirs_exist_ok=True)
        except Exception as e:  # noqa: BLE001  the item that needs it fails (recorded), never the queue
            return f"{rel}: not pulled from the runs repo ({type(e).__name__}: {e})"
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        self.event("out_dir_pulled", run_dir=rel, found=found)
        return None

    def fetch_hub_file(self, rel: str) -> bool:
        """One runs-repo file into <root>/<rel> (through a stage dir); False when there is no runs repo, no such file,
        or the download failed (logged)."""
        hub = self.runs_hub()
        if hub is None:
            return False
        stage = self.root / "cache" / "hub_pull" / f"file-{time.time_ns()}"
        try:
            got = hub.download(rel, stage)
            if got is None:
                return False
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), str(self.root / rel))
            return True
        except Exception as e:  # noqa: BLE001  the verdict check then fails with the file missing as its evidence
            log(f"{rel}: not fetched from the runs repo ({type(e).__name__}: {e})")
            return False
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def pull_speed_dir(self) -> str | None:
        """An adopted speed dir (its done items' records are in the Hub's speed.json) is pulled before the first speed
        item on this host writes into it: speed_probe merges into an existing --out, and the dir's upload replaces the
        Hub's speed.json, which would lose those records."""
        sd = self.state["speed_dir"]
        if not sd.get("adopted") or sd.get("pulled"):
            return None
        why = self.pull_out_dir(sd["run_dir"])
        if why is None:
            sd["pulled"] = True
            self.save()
        return why

    def pull_inputs(self, name: str) -> str | None:
        """Every {out:<item>} of the item's argv or args whose item is done but whose out dir is not on this host (done
        before an adoption), pulled from the runs repo (a quant readout's exported variant/ weights excepted: its
        readers read the eval outputs). Why not (None: all there)."""
        spec = self.spec_of(name)
        for a in [*(spec.get("argv") or []), *(spec.get("args") or [])]:
            for x in re.findall(r"\{out:([^{}]+)\}", a):
                src = self.state["items"].get(x) or {}
                if src.get("status") == "done" and src.get("out") and not (self.root / src["out"]).is_dir():
                    why = self.pull_out_dir(src["out"], ignore=("variant/*",))
                    if why:
                        return why
        return None

    def fetch_weights(self, pairs: list[tuple[str, int]]) -> str | None:
        """Download runs-repo weights (runs/<run_id>/config.json and checkpoints/step_<step>/) into <root>/cache/hub,
        once each; why not (None: every one is there)."""
        hub_cache = self.root / "cache" / "hub"
        for rid, step in pairs:
            done = hub_cache / "runs" / rid / f".pulled-step_{step}"
            if done.is_file():
                continue
            if self.uploader is None:
                return f"no runs repo to fetch runs/{rid}/checkpoints/step_{step} from"
            try:
                with self.ctl_beating():
                    self.uploader.download(f"runs/{rid}/config.json", hub_cache)
                    self.uploader.download_dir(f"runs/{rid}/checkpoints/step_{step}", hub_cache)
            except Exception as e:  # noqa: BLE001  a failed readout, recorded
                return f"weights runs/{rid}/checkpoints/step_{step} not fetched ({type(e).__name__}: {e})"
            ck = hub_cache / "runs" / rid / "checkpoints" / f"step_{step}"
            if not ck.is_dir() or not any(ck.iterdir()):
                return f"runs/{rid}/checkpoints/step_{step} is not in the runs repo"
            done.write_text(_now_utc() + "\n", encoding="utf-8")
            self.event("weights_fetched", run_id=rid, step=step)
        return None

    def speed_dir(self) -> str:
        """runs/speed-<box>-<stamp>: every speed item of this box launch writes one speed.json there (the stamp is fixed
        at the first speed attempt and kept across restarts and adoption)."""
        if not self.state.get("speed_dir"):
            self.state["speed_dir"] = dict(run_dir=f"runs/speed-{self.box}-{time.strftime(STAMP, time.gmtime())}",
                                           verified=None)
        return self.state["speed_dir"]["run_dir"]

    # --------------------------------------------------------------------------------------------- the child

    def model_source(self, name: str) -> dict:
        """The {config}, {ckpt}, {run_id}, {run_name} of an item's `of` (this box's run dir, or another box's in the
        hub cache), and {config:<n>} / {ckpt:<n>} of its weights."""
        it, spec = self.item(name), self.spec_of(name)
        hub = (self.root / "cache" / "hub").as_posix()
        out = {}
        if it.get("source"):  # of_box
            rid, steps = it["source"]["run_id"], it["source"]["steps"]
            out.update(config=f"{hub}/runs/{rid}/config.json", ckpt=f"{hub}/runs/{rid}/checkpoints/step_{steps}",
                       run_id=rid, run_name=spec["of"])
        elif spec.get("of"):
            of = self.item(spec["of"])
            rd, steps = of.get("run_dir"), (of.get("result") or {}).get("steps")
            if rd:
                out.update(config=f"{rd}/config.json", ckpt=f"{rd}/checkpoints/step_{steps}",
                           run_id=fullrun.run_id_of(rd), run_name=spec["of"])
        for w in spec.get("weights") or []:
            out[f"config:{w['name']}"] = f"{hub}/runs/{w['run_id']}/config.json"
            out[f"ckpt:{w['name']}"] = f"{hub}/runs/{w['run_id']}/checkpoints/step_{w['step']}"
        return out

    def fill(self, template: str, name: str) -> str:
        """An argv or verdict template with its placeholders (fullrun.PLACEHOLDERS, {out:<item>}, {config:<name>},
        {ckpt:<name>}) filled for the item."""
        it = self.item(name)
        cache = (self.root / "cache").as_posix()
        ph = dict(python=self.s.python, root=self.root.as_posix(), state=Path(self.s.state_dir).as_posix(),
                  box=self.box, cache_dir=cache, hf_cache=f"{cache}/hf", hub_cache=f"{cache}/hub",
                  manifest=fullrun.FROZEN_MANIFEST, out=it.get("out"), **self.model_source(name))

        def repl(m):
            key = m.group(1)
            if key.startswith("out:"):
                v = self.item(key[4:]).get("out")
            else:
                v = ph.get(key)
            if v is None:
                raise Q.QueueError(f"{name}: placeholder {{{key}}} has no value yet")
            return str(v)

        return re.sub(r"\{([^{}]*)\}", repl, template)

    def argv_for(self, name: str, it: dict, resume: Path | None) -> list[str]:
        """The item's command line (H3): a store build, a trainer (fresh, or --resume with the same hf sets and the
        sets_once of a reset), the binding M4 readout, a speed probe, or the eval item's template."""
        spec = self.spec_of(name)
        kind = spec["kind"]
        if kind == "stores":
            return self._cmd("stores") + ["--config", spec["config"], *(["--eval-only"] if spec["eval_only"] else []),
                                          *sum((["--set", s] for s in spec["sets"]), [])]
        if kind == "train":
            argv = self._cmd("train")
            argv += ["--resume", self._rel(resume)] if resume is not None else [
                "--config", spec["config"], "--set", f"run_name={name}"]
            if self.s.out_repo:
                argv += ["--set", f"hf.output_repo={self.s.out_repo}"]
            if self.spec["timed_states"] and self.s.scratch_repo:
                argv += ["--set", f"hf.scratch_repo={self.s.scratch_repo}"]
            if resume is not None:  # a reset or set run: until the trainer has applied them (sets_applied)
                argv += sum((["--set", s] for s in it.get("sets_once") or []), [])
            return argv
        if kind == "readout":
            of = self.item(it["of"])
            rd, steps, out = of["run_dir"], of["result"]["steps"], it["out"]
            return self._cmd("eval") + [
                "--config", f"{rd}/config.json", "--ckpt", f"{rd}/checkpoints/step_{steps}", "--out", out,
                "--manifest", fullrun.FROZEN_MANIFEST, "--tables", f"{out}/tables", "--system", it["of"],
                "--cache-dir", (self.root / "cache").as_posix(), "--max-temp", "0"]
        if kind == "speed":
            src = self.model_source(name)
            model = src.get("ckpt") or next((v for k, v in src.items() if k.startswith("ckpt:")), None)
            if spec.get("model"):
                model = (self.root / spec["model"]).as_posix()
            return self._cmd("speed") + [
                "--kind", spec["speed_kind"], *(["--model", model] if model else []), "--system", spec["system"],
                "--store", (self.root / "cache" / "eval").as_posix(), "--per-set", str(SPEED_PER_SET),
                "--seed", str(SPEED_SEED), "--out", f"{it['out']}/speed.json", "--require-idle",
                *[self.fill(a, name) for a in spec["args"]]]
        return [self.fill(a, name) for a in spec["argv"]]

    def child_env(self, name: str, it: dict, gpu: str) -> dict:
        """H4: the item's heartbeat file and deadline (none without a box deadline)."""
        env = {fullrun.ENV_HEARTBEAT: str(fullrun.item_hb_path(name, self.s.state_dir))}
        dl = self.item_deadline(name)
        if dl is not None:
            env[fullrun.ENV_DEADLINE] = str(int(dl))
        return env

    def start(self, name: str, gpu: str, resume: Path | None = None):
        it = self.item(name)
        self._start_faults(name)
        heartbeat.beat(fullrun.item_hb_path(name, self.s.state_dir), force=True)
        if it["kind"] == "train" and it.get("sets_once"):
            self.sets_applied(name)
        proc = super().start(name, gpu, resume)
        if resume is not None:
            it["run_dir"] = it["attempts"][-1]["resume"]
        self._tails.pop(name, None)
        self.put_summary()
        return proc

    def sets_applied(self, name: str) -> bool:
        """A reset or set run's sets_once stay on every attempt until its events.jsonl holds a resume_reset event
        followed by a full checkpoint (the trainer then saved the reset state: a later resume must not reset again);
        then resume_reset_applied is recorded and the sets are dropped."""
        it = self.item(name)
        if not it.get("sets_once") or not it["run_dir"]:
            return False
        # this box's attempts only: a pulled events.jsonl may hold an older continuation's reset
        since = min((a["t0"] for a in it["attempts"]), default=time.time())
        seen = None
        for e in Q.read_events(self.root / it["run_dir"]):
            if float(e.get("wall") or 0) < since - 1:
                continue
            if e.get("kind") == "resume_reset":
                seen = e
            elif seen is not None and e.get("kind") == "checkpoint" and e.get("ckpt") == "full":
                it["resume_reset_applied"] = dict(at_step=seen.get("at_step"), checkpoint=e.get("name"),
                                                  sets=it["sets_once"], wall=time.time())
                self.event("resume_reset_applied", item=name, **it["resume_reset_applied"])
                it["sets_once"] = []
                self.save()
                return True
        return False

    def resume_point(self, name: str) -> Path | None:
        """A train item that was interrupted or failed goes on from its run dir when that holds a full state
        (full_step_<N>/trainer.pt); a run dir without one is set aside and the item starts fresh."""
        it = self.item(name)
        if it["kind"] != "train" or it["status"] not in ("interrupted", "retry"):
            return None
        rd = self.root / it["run_dir"] if it["run_dir"] else (self.find_run_dir(name) if it["attempts"] else None)
        if rd is None or not rd.is_dir():
            it["run_dir"], it["continuation"] = None, None
            return None
        ck = rd / "checkpoints"
        if ck.is_dir() and any(FULL_STATE_RE.match(p.name) and (p / "trainer.pt").is_file() for p in ck.iterdir()):
            return rd
        self.set_aside(rd, "no full state to resume from")
        it["run_dir"] = None
        it["sets_once"], it["continuation"] = [], None  # a fresh start cannot take a reset: a new run, no continuation
        return None

    # --------------------------------------------------------------------------------------------- the end

    def finished(self, name: str, rc: int, todo: list[str], on_done=None):
        it, spec = self.item(name), self.spec_of(name)
        kind = spec["kind"]
        att = it["attempts"][-1]
        self._tails.pop(name, None)
        att.update(rc=rc, t1=time.time())
        rd = None
        if kind == "train":
            rd = self.find_run_dir(name)
            if rd is None and it["run_dir"] and (self.root / it["run_dir"]).is_dir():
                rd = self.root / it["run_dir"]
            if rd is not None:
                it["run_dir"] = att["run_dir"] = self._rel(rd)
        elif kind in ("readout", "eval") and it.get("out"):
            rd = self.root / it["out"]
        summary = self.summary_of(rd) if kind == "train" else None
        fault = att.get("fault")
        self.event("item_end", item=name, rc=rc, run_dir=it["run_dir"], out=it.get("out"), fault=fault,
                   stalled=bool(att.get("stalled")), minutes=round((att["t1"] - att["t0"]) / 60, 1),
                   status=(summary or {}).get("status"))
        if att.get("wipe") and rd is not None and rc != 0:
            self._wipe(name, rd)
        error = str((summary or {}).get("error", ""))
        if rc == 0:
            it["status"] = "done"
            it["result"] = self.result_for(name, rd, summary)
        elif kind == "train" and not fault and rc == EXIT_THROUGHPUT:
            it["status"] = "failed"  # this item only: the box's other items go on (DECISIONS C6)
            it["why"] = "throughput below the trainer's floor (exit 3)"
            self.event("item_failed", item=name, rc=rc, reason="throughput")
        elif kind == "train" and not fault and error.startswith(DETERMINISTIC):
            it["status"], it["why"] = "failed", error[:300]
            self.event("item_failed", item=name, rc=rc, error=error[:600], retried=False)
        elif kind in POOL_KINDS:
            why = ("stalled" if att.get("stalled") else "overrun" if att.get("overrun") else f"exit {rc}")
            it["status"], it["why"] = "failed", why
            self.state["readouts_failed"][name] = why
            self.event("item_failed", item=name, rc=rc, reason=why, retried=False)
        elif fault or self.failures(name) < self.max_attempts:
            it["status"] = "retry"  # a train item resumes from its newest local full state (resume_point)
            self.event("item_retry", item=name, rc=rc, fault=fault, failures=self.failures(name))
        else:
            it["status"] = "failed"
            it["why"] = f"failed {self.failures(name)} times (last exit {rc})"
            self.event("item_failed", item=name, rc=rc, reason="attempts", failures=self.failures(name))
        if kind in ("readout", "eval") and rd is not None:
            rd.mkdir(parents=True, exist_ok=True)  # finish.py's run-dir scan needs its events.jsonl
            with open(rd / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": "item_done", "item": name, "status": it["status"], "rc": rc,
                                    "wall": time.time(), "box": self.box}) + "\n")
        if kind == "speed":
            self.speed_item_ended(name, rc)
        if it["status"] == "done" and kind == "train" and rd is not None:
            self.ensure_attribution(it, rd)
            if it.get("sets_once"):
                self.sets_applied(name)
        if it["status"] == "done" and self._upload_dir(name):
            self._pending_uploads.append(name)
        self._fault_outcomes(name)
        self.save()
        self.put_summary(force=True)

    def failures(self, name: str) -> int:
        """The item's real failures: attempts that ended non-zero, but neither an interrupted one (a queue restart)
        nor one a smoke fault ended."""
        return sum(1 for a in self.item(name)["attempts"] if a.get("rc") not in (None, 0) and not a.get("fault"))

    def _upload_dir(self, name: str) -> str | None:
        it = self.item(name)
        if it["kind"] == "train":
            return it["run_dir"]
        return it.get("out") if it["kind"] in ("readout", "eval") else None

    def result_for(self, name: str, rd: Path | None, summary: dict | None) -> dict | None:
        """The queue summary's result of a done item (build contract 5)."""
        it, kind = self.item(name), self.item(name)["kind"]
        if kind == "train":
            s = summary or {}
            return dict(run_id=rd.name if rd else None, steps=s.get("steps"), status=s.get("status"),
                        epochs=s.get("epochs"), stopped_early=bool(s.get("stopped_early")),
                        end_reason=s.get("end_reason"), resume_resets=int(s.get("resume_resets") or 0))
        if kind == "readout":
            study = _read_json(rd / "study.json") if rd else None
            m = (study or {}).get("metrics") or {}
            return dict(system=it["of"], out=it["out"], **{k: m.get(k) for k in READOUT_METRICS},
                        jsut_cer_nostyle=_dig(study or {}, "strata.eval_jsut.cer_nostyle"))
        if kind == "eval":
            return dict(out=it["out"], rc=0)
        if kind == "speed":
            return dict(system=self.spec_of(name)["system"], out=it["out"])
        return None

    def drain_uploads(self, one: bool = False):
        """Upload and verify the finished run dirs, readout and eval out dirs and the speed dir (lean), one per call
        inside the scheduler loop (none while a speed probe runs: it times on an idle host); under the controller
        heartbeat. The speed dir's upload verifies every done speed item it carries. Each upload asks for a summary put
        (coalesced), so the Hub's summary says `verified` once an output is there (resume-pull's rule for done)."""
        if not self._pending_uploads:
            return
        if one and any(self.item(n)["kind"] == "speed" for n in self.procs):
            return
        with self.ctl_beating():
            while self._pending_uploads:
                name = self._pending_uploads.pop(0)
                entry = self.state["speed_dir"] if name == SPEED_DIR_KEY else self.item(name)
                rel = entry["run_dir"] if name == SPEED_DIR_KEY else self._upload_dir(name)
                covered = [n for n in self.kind_names("speed") if self.item(n)["status"] == "done"
                           and self.item(n)["verified"] is not True] if name == SPEED_DIR_KEY else []
                if self.uploader is None or not rel or not (self.root / rel).is_dir():
                    entry["verified"] = None
                else:
                    problems = self.uploader.sync_run(self.root / rel)
                    if problems:  # once more: a transient Hub error, a commit that raced another writer
                        problems = self.uploader.sync_run(self.root / rel)
                    entry["verified"] = not problems
                    for n in covered:  # speed.json holds every speed item done so far
                        self.item(n)["verified"] = not problems
                    self.event("item_uploaded", item=name, run_dir=rel, verified=not problems, problems=problems[:20],
                               **({"speed_items": covered} if name == SPEED_DIR_KEY else {}))
                self.save()
                self.put_summary()
                if one:
                    return

    def speed_item_ended(self, name: str, rc: int):
        """A speed item that ran: its line in <speed dir>/events.jsonl (which makes the dir a run dir for finish.py),
        and the dir's upload queued at once (after the next idle poll), so a record never waits for the end of the
        pool to reach the Hub. A done speed item is `verified` only once that upload is."""
        it = self.item(name)
        d = self.root / self.speed_dir()
        sd = self.state["speed_dir"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "speed", "item": name, "system": self.spec_of(name)["system"],
                                "status": it["status"], "why": it.get("why"), "rc": rc, "wall": time.time(),
                                "box": self.box}) + "\n")
        sd.update(written=True, verified=None)
        if it["status"] == "done":
            it["verified"] = None
        if SPEED_DIR_KEY not in self._pending_uploads:
            self._pending_uploads.append(SPEED_DIR_KEY)

    # --------------------------------------------------------------------------------------------- monitor

    def monitor(self, running: dict):
        """Every poll: the controller beat, run-dir discovery (a summary put at once), the stall and overrun kills, the
        smoke faults, the resource peaks, a coalesced summary put that is due; a chain part's stop_at first."""
        self.stage_deadline_check()
        now = time.time()
        hb = self.ctl_beat_path()
        if hb is not None:  # never beat(None): that would beat this process's own $KITSUNE_HEARTBEAT, if any
            heartbeat.beat(hb, force=True)
        self._freeze_windows(now)
        for r in running.values():
            name, proc = r["name"], r["proc"]
            it, spec = self.item(name), self.spec_of(name)
            att = it["attempts"][-1]
            if spec["kind"] == "train" and not it["run_dir"]:
                rd = self.find_run_dir(name)
                if rd is not None:
                    it["run_dir"] = self._rel(rd)
                    self.event("run_dir_found", item=name, run_dir=it["run_dir"])
                    self.save()
                    self.put_summary(force=True)
            if att.get("kill"):
                self._reap(name, proc, now)
                continue
            try:
                beat_m = os.stat(fullrun.item_hb_path(name, self.s.state_dir)).st_mtime
            except OSError:
                beat_m = 0.0
            age = now - max(beat_m, float(att["t0"]))
            it["hb_max_gap_s"] = round(max(float(it.get("hb_max_gap_s") or 0.0), age), 1)
            sm, mh = spec["stall_min"], spec.get("max_hours")
            if sm is not None and age > float(sm) * 60:
                att["stalled"] = dict(age_s=round(age, 1), limit_s=float(sm) * 60)
                it["stalls"] = int(it.get("stalls") or 0) + 1
                self.event("item_stalled", item=name, age_s=round(age, 1), limit_s=float(sm) * 60,
                           attempt=len(it["attempts"]))
                self._kill(name, proc, "stall")
                self.save()
                continue
            if sm is None and mh and now - float(att["t0"]) > fullrun.OVERRUN_FACTOR * float(mh) * 3600:
                att["overrun"] = dict(wall_s=round(now - att["t0"], 1), limit_s=fullrun.OVERRUN_FACTOR * mh * 3600)
                self.event("item_overrun", item=name, **att["overrun"])
                self._kill(name, proc, "overrun")
                self.save()
                continue
            if self.spec["smoke"] and spec["kind"] == "train":
                self._faults(name, proc, now)
        self._resources(running)
        if self._put_pending and now - self._last_put >= self.s.summary_min_s:
            self.put_summary(force=True)

    def _killpg(self, proc, sig) -> bool:
        """Send sig to the child's process group (posix); False where that does not exist (Windows)."""
        if os.name != "posix" or sig is None:
            return False
        try:
            os.killpg(proc.pid, sig)
        except (OSError, ProcessLookupError):
            pass
        return True

    def _kill(self, name: str, proc, why: str):
        """SIGCONT (a stopped group ignores SIGTERM until it continues), SIGTERM, SIGKILL after kill_grace_s (_reap).
        On Windows terminate(), then kill()."""
        att = self.item(name)["attempts"][-1]
        att["kill"] = dict(why=why, t=time.time())
        if os.name == "posix":
            self._killpg(proc, getattr(signal, "SIGCONT", None))
            self._killpg(proc, signal.SIGTERM)
        else:
            try:
                proc.terminate()
            except OSError:
                pass

    def _reap(self, name: str, proc, now: float):
        att = self.item(name)["attempts"][-1]
        if proc.poll() is None and now - att["kill"]["t"] > self.s.kill_grace_s and not att["kill"].get("killed"):
            att["kill"]["killed"] = now
            if os.name == "posix":
                self._killpg(proc, signal.SIGKILL)
            else:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _resources(self, running: dict):
        """Peak RSS per item (VmRSS summed over its process group, from /proc) and the container's memory.current."""
        proc_root = Path(self.s.proc_root)
        groups = {int(r["proc"].pid): r["name"] for r in running.values()}
        if groups and proc_root.is_dir():
            kb = dict.fromkeys(groups.values(), 0)
            for d in proc_root.iterdir():
                if not d.name.isdigit():
                    continue
                try:
                    pgrp = int((d / "stat").read_text().rsplit(")", 1)[1].split()[2])
                    if pgrp not in groups:
                        continue
                    for line in (d / "status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            kb[groups[pgrp]] += int(line.split()[1])
                except (OSError, ValueError, IndexError):
                    continue
            for name, v in kb.items():
                if v:
                    it = self.item(name)
                    it["peak_rss_gb"] = round(max(float(it.get("peak_rss_gb") or 0.0), v / 2 ** 20), 3)
        try:
            cur = int((Path(self.s.cgroup) / "memory.current").read_text().split()[0]) / 2 ** 30
            self.state["host_mem_peak_gb"] = round(max(float(self.state.get("host_mem_peak_gb") or 0.0), cur), 3)
        except (OSError, ValueError, IndexError):
            pass

    # ---------------------------------------------------------------------------------------------- faults

    def _faults_of(self, name: str) -> list[dict]:
        return [f for f in self.spec["faults"] if f["item"] == name]

    def _fire(self, f: dict, name: str, **fields):
        fs = self.state["faults"][f["id"]]
        fs["fired_at"] = dict(wall=time.time(), attempt=len(self.item(name)["attempts"]), **fields)
        self.event("fault_fired", id=f["id"], action=f["action"], item=name, step=fields.get("step"),
                   wall=fs["fired_at"]["wall"])
        self.save()

    def _start_faults(self, name: str):
        """A deadline fault fires at the start of its first eligible attempt: KITSUNE_DEADLINE = now + seconds."""
        for f in self._faults_of(name):
            fs = self.state["faults"][f["id"]]
            if f["action"] == "deadline" and fs["fired_at"] is None and \
                    len(self.item(name)["attempts"]) + 1 >= f["min_attempt"]:
                fs["deadline"] = time.time() + float(f["seconds"])
                self._fire(f, name, deadline=fs["deadline"])

    def _step(self, name: str) -> int | None:
        it = self.item(name)
        if not it["run_dir"]:
            return None
        tail = self._tails.get(name)
        if tail is None or tail.run_dir != self.root / it["run_dir"]:
            tail = self._tails[name] = Q.StepTail(self.root / it["run_dir"])
        rows = tail.poll()
        return max(rows) if rows else None

    def _attempt_events(self, name: str) -> list[dict]:
        it = self.item(name)
        t0 = float(it["attempts"][-1]["t0"])
        return [e for e in Q.read_events(self.root / it["run_dir"]) if float(e.get("wall") or 0) >= t0 - 1] \
            if it["run_dir"] else []

    def _faults(self, name: str, proc, now: float):
        it = self.item(name)
        att = it["attempts"][-1]
        for f in self._faults_of(name):
            fs = self.state["faults"][f["id"]]
            if fs["fired_at"] is not None or len(it["attempts"]) < f["min_attempt"] or f["action"] == "deadline":
                continue
            if f["action"] == "wipe_run_dir":
                if not any(e.get("kind") == f["after_event"] for e in self._attempt_events(name)):
                    continue
                self._fire(f, name, event=f["after_event"])
                att["fault"], att["wipe"] = f["id"], True
                self._kill(name, proc, f"fault {f['id']}")
                continue
            step = self._step(name)
            if step is None or step < int(f["at_step"]):
                continue
            if f["action"] in ("sigstop", "kill") and not any(
                    e.get("kind") == "checkpoint" and e.get("ckpt") == "full" for e in self._attempt_events(name)):
                continue  # only once this attempt has a full state to resume from
            self._fire(f, name, step=step)
            if f["action"] == FREEZE:
                fs["window_end"] = now + float(f["seconds"])
                self._freeze_until = fs["window_end"]
                continue
            att["fault"] = f["id"]
            if f["action"] == "sigstop":
                if not self._killpg(proc, getattr(signal, "SIGSTOP", None)):
                    self.event("fault_unsupported", id=f["id"], why="no SIGSTOP on this platform")
            else:  # kill
                att["kill"] = dict(why=f"fault {f['id']}", t=now, killed=now)
                if not self._killpg(proc, getattr(signal, "SIGKILL", None)):
                    proc.kill()

    def _open_freezes(self) -> list[tuple[dict, dict]]:
        """(fault, its state) of every freeze_controller_hb fault that fired and has no outcome yet."""
        out = []
        for f in self.spec["faults"]:
            fs = self.state["faults"].get(f["id"]) or {}
            if f["action"] == FREEZE and fs.get("fired_at") is not None and fs.get("window_end") and \
                    fs.get("outcome") is None:
                out.append((f, fs))
        return out

    def _freeze_windows(self, now: float):
        """Every poll (and in hold_freeze): a freeze window that has run its `seconds` closes (released at its
        window_end, release "window")."""
        for f, fs in self._open_freezes():
            if now >= float(fs["window_end"]):
                self._release(f, float(fs["window_end"]), "window")

    def _release(self, f: dict, at: float, why: str):
        """A freeze ends: released (wall), release (alert: the watchdog's alert was recorded during the hold; window:
        its seconds ran out; deadline: the hold reached the box deadline less deadline_reserve_min; cut: end_faults found
        it open, which hold_freeze should make unreachable), outcome recovered; the controller beats again unless
        another freeze still holds."""
        fs = self.state["faults"][f["id"]]
        fs.update(released=at, release=why)
        self._outcome(f, "recovered")
        self._freeze_until = max([float(x["window_end"]) for _, x in self._open_freezes()], default=0.0)

    def watchdog_alerts(self) -> list[dict]:
        """The box watchdog's alert records ($KITSUNE_STATE/watchdog_alerts.jsonl; a chain part: the box's state dir)."""
        p = self.box_state / fullrun.ALERTS_FILE
        out = []
        if p.is_file():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def freeze_alerts(self, fs: dict) -> list[dict]:
        """The watchdog alerts that count for a fired freeze (check 7, and hold_freeze's early release): wall in
        [its fire, its end + ALERT_SLACK_S], its end being its release, or while it is open its planned window_end."""
        t0 = (fs.get("fired_at") or {}).get("wall")
        if t0 is None:
            return []
        end = float(fs.get("released") or fs.get("window_end") or t0)
        return [a for a in self.watchdog_alerts() if float(t0) <= float(a.get("wall") or 0) <= end + ALERT_SLACK_S]

    def hold_freeze(self):
        """Hold the part while a fired freeze window is open (run(): after the last item and the upload drain, before
        end_faults). Box 53693389 (2026-10-01): F5 fired on smoke-p005, the item ended ~50 s later, the rest of the part
        took ~7 min, and end_faults cut the window after 480 s - below the watchdog's orphan_s 600 plus its 60 s poll, so
        no alert could come and check 7 failed (alerts_during 0). Here the part runs no item and never beats (the window
        holds ctl_beat_path at None) until the watchdog's alert is recorded (freeze_alerts: release "alert", at once), the
        window ends (release "window"), or the box deadline less deadline_reserve_min comes (release "deadline"); a
        chain part's stop_at halts it (stage_deadline_check: run()'s Halt, the verdict written, the window left open as
        any Halt leaves it). Expected idle ~orphan_s + the watchdog's poll - what was left of the window at the part's
        end; with no watchdog, the rest of `seconds`. The chain's set_mode comes after the part returns, so the watchdog
        stays in its alert mode for the whole hold."""
        for f, fs in self._open_freezes():
            t_in = time.time()
            dl = self.box_deadline()
            cap = None if dl is None else dl - float(self.spec["deadline_reserve_min"]) * 60
            end = float(fs["window_end"])
            self.event("freeze_hold", id=f["id"], window_end=end, left_s=round(end - t_in, 1),
                       alerts=len(self.freeze_alerts(fs)))
            log(f"{f['id']}: the freeze window is open for {max(0.0, end - t_in):.0f} s more: holding the part (no "
                f"items, no beats) until the watchdog's alert or the window's end")
            self.save()
            try:
                while fs.get("outcome") is None:
                    now = time.time()
                    if self.freeze_alerts(fs):
                        self._release(f, now, "alert")
                    elif now >= end:
                        self._freeze_windows(now)
                    elif cap is not None and now >= cap:
                        self._release(f, now, "deadline")
                    else:
                        self.stage_deadline_check()
                        time.sleep(max(0.0, min(self.s.poll_s, end - now, (cap - now) if cap is not None else end)))
            finally:
                fs["held_s"] = round(time.time() - t_in, 2)
                self.save()

    def _outcome(self, f: dict, outcome: str):
        fs = self.state["faults"][f["id"]]
        fs["outcome"] = outcome
        self.event("fault_outcome", id=f["id"], outcome=outcome)

    def _fault_failed(self, name: str):
        for f in self._faults_of(name):
            fs = self.state["faults"][f["id"]]
            if f["action"] == "wipe_run_dir" and fs["fired_at"] is not None and fs["outcome"] is None:
                self._outcome(f, "failed")

    def _fault_outcomes(self, name: str):
        """A fired fault's outcome once its item has ended for good: done -> recovered, else failed; a fault whose
        item ended before it fired is missed."""
        if not self.spec["faults"] or self.item(name)["status"] not in ENDED:
            return
        for f in self._faults_of(name):
            fs = self.state["faults"][f["id"]]
            if fs["outcome"] is not None:
                continue
            if fs["fired_at"] is None:
                fs["outcome"] = "missed"
                self.event("fault_missed", id=f["id"], action=f["action"], item=name)
            elif f["action"] != FREEZE:
                ok = self.item(name)["status"] == "done" and (f["action"] != "wipe_run_dir"
                                                               or self.item(name).get("check_resume") == "ok")
                self._outcome(f, "recovered" if ok else "failed")

    def end_faults(self):
        for f in self.spec["faults"]:
            fs = self.state["faults"][f["id"]]
            if fs["outcome"] is not None:
                continue
            if fs["fired_at"] is None:
                fs["outcome"] = "missed"
                self.event("fault_missed", id=f["id"], action=f["action"], item=f["item"])
            elif f["action"] == FREEZE:  # hold_freeze closed it unless the part ended in a way that skipped the hold
                self._release(f, time.time(), "cut")
            else:
                self._fault_outcomes(f["item"])
        self._freeze_until = 0.0
        self.save()

    def _wipe(self, name: str, rd: Path):
        """The wipe_run_dir fault, once its process has ended: the run's logs synced to the Hub, the run dir moved to
        runs/_wiped/ (never deleted), the item resumed from the Hub as a new host would (_prepare_train)."""
        it = self.item(name)
        if self.uploader is not None:
            try:
                with self.ctl_beating():
                    self.uploader.sync_run(rd)
            except Exception as e:  # noqa: BLE001  the pull then finds what the trainer's own syncs sent
                log(f"{name}: log sync before the wipe failed: {type(e).__name__}: {e}")
        dest = self.root / "runs" / "_wiped" / rd.name
        if dest.exists():
            dest = dest.with_name(f"{rd.name}-{time.time_ns()}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rd), str(dest))
        it["hub_resume"] = True
        self.event("run_dir_wiped", item=name, run_dir=it["run_dir"], moved_to=self._rel(dest))

    # --------------------------------------------------------------------------------------------- summary

    def put_summary(self, force: bool = False):
        """The running summary: put now (force) or coalesced (at most one put per summary_min_s; a pending one is put
        by the first poll past that mark)."""
        now = time.time()
        if not force and now - self._last_put < self.s.summary_min_s:
            self._put_pending = True
            return
        self._put_pending = False
        self._last_put = now
        self.write_summary("running", None, None)

    def summary(self, status: str, reason: str | None, rc: int | None) -> dict:
        items = {}
        for n in self.order:
            it = self.item(n) if n in self.state["items"] else None
            if it is None:
                continue
            items[n] = dict(kind=it["kind"], status=it["status"], run_dir=it["run_dir"], out=it.get("out"),
                            of=it.get("of"), gpu=(it["attempts"] or [{}])[-1].get("gpu"), verified=it["verified"],
                            attempts=len(it["attempts"]), hb_max_gap_s=it.get("hb_max_gap_s"),
                            stalls=it.get("stalls"), peak_rss_gb=it.get("peak_rss_gb"), result=it["result"],
                            why=it.get("why"), continuation=it.get("continuation"))
        return dict(format=1, kind=fullrun.JOB, box=self.box, status=status, reason=reason, rc=rc, sha=self.s.sha,
                    machine_id=self.s.machine_id, container_id=self.s.container_id, gpus=self.gpus,
                    registry_sha256=self.registry_sha256, deadline=self.box_deadline(),
                    started=self.state["started"], ended=None if status == "running" else time.time(),
                    host_mem_peak_gb=self.state.get("host_mem_peak_gb"), items=items,
                    readouts_failed=self.state["readouts_failed"], no_start=self.state["no_start"],
                    not_needed=self.state["not_needed"], faults=self.faults_list(), resumed=self.state.get("resumed"),
                    speed_dir=(self.state.get("speed_dir") or {}).get("run_dir"),
                    speed_dir_verified=(self.state.get("speed_dir") or {}).get("verified"))

    def faults_list(self) -> list[dict]:
        return [dict(id=fs["id"], action=fs["action"], item=fs["item"], fired_at=fs.get("fired_at"),
                     outcome=fs.get("outcome"), **{k: fs[k] for k in ("window_end", "released", "release", "held_s")
                                                   if k in fs}) for fs in self.state["faults"].values()]

    def write_summary(self, status: str, reason: str | None, rc: int | None):
        path = Path(self.s.state_dir) / fullrun.SUMMARY_FILE
        Q._atomic_json(path, self.summary(status, reason, rc))
        self._put(path, fullrun.box_summary_path(self.box))

    def _put(self, path: Path, path_in_repo: str):
        if self.uploader is None:
            return
        try:
            with self.ctl_beating():
                self.uploader.put_file(path, path_in_repo)
        except Exception as e:  # noqa: BLE001  finish.py uploads the state dir with the infra logs anyway
            log(f"{path_in_repo} upload failed: {type(e).__name__}: {e}")

    # --------------------------------------------------------------------------------------------- verdict

    def write_verdict(self) -> dict:
        """$KITSUNE_STATE/smoke_verdict.json, put at full/box-<box>/smoke_verdict.json (smoke boxes). Its check 7 lists
        the scratch repo on the Hub: under the controller heartbeat too."""
        with self.ctl_beating():
            v = SmokeVerdict(self).build()
        path = Path(self.s.state_dir) / fullrun.VERDICT_FILE
        Q._atomic_json(path, v)
        self.event("smoke_verdict", overall=v["overall"],
                   checks={k: c["pass"] for k, c in v["checks"].items()})
        self._put(path, fullrun.box_verdict_path(self.box))
        return v


# ========================================================================================================= verdict


def _int_in(v, lo: int, hi: int) -> bool:
    """v (an env value: a string) is an integer in [lo, hi]."""
    try:
        return lo <= int(str(v).strip()) <= hi
    except (TypeError, ValueError):
        return False


def steady_wait(wait: dict[int, float], step_s: dict[int, float], launches: list[int], exclude=()) -> dict:
    """Check 5's loader share of one run: sum(time/data_wait_s) / sum(time/step_s) over the steps logged with both,
    leaving out the first LAUNCH_SKIP_STEPS steps after every launch (at_step a of a phase{name: "train"} event: steps
    a+1 ...; its spawned workers start inside them) and the steps in `exclude` (the smoke profiler's). No launch known
    (an event lost): the first logged step's launch. data_wait_frac None when no step remains; startup_wait_s is each
    launch's first-step wait (None when that step was not logged)."""
    launches = sorted(set(launches)) or ([min(step_s) - 1] if step_s else [])
    drop = {s for a in launches for s in range(a + 1, a + 1 + LAUNCH_SKIP_STEPS)} | set(exclude)
    steps = [s for s in sorted(step_s) if s in wait and s not in drop]
    tot = sum(step_s[s] for s in steps)
    return dict(data_wait_frac=sum(wait[s] for s in steps) / tot if tot > 0 else None, steps_measured=len(steps),
                startup_wait_s={a + 1: wait.get(a + 1) for a in launches})


class SmokeVerdict:
    """smoke_verdict.json (build contract 5): the built-in checks 1-11 on a smoke box with train items (smoke A), then
    the registry's verdict specs (checks 12-16; smoke B has only these). A check is {pass: true | false | null,
    evidence}; overall fail when any check is false."""

    def __init__(self, q: FullQueue):
        self.q = q
        self.items = q.state["items"]
        self._events: dict[str, list[dict]] = {}

    # readers ------------------------------------------------------------------------------------------------

    def rd(self, name: str) -> Path | None:
        it = self.items.get(name) or {}
        return self.q.root / it["run_dir"] if it.get("run_dir") else None

    def events(self, name: str) -> list[dict]:
        if name not in self._events:
            rd = self.rd(name)
            self._events[name] = Q.read_events(rd) if rd is not None else []
        return self._events[name]

    def kinds(self, name: str, kind: str) -> list[dict]:
        return [e for e in self.events(name) if e.get("kind") == kind]

    def summary(self, name: str) -> dict:
        rd = self.rd(name)
        return (Q.Queue.summary_of(rd) if rd is not None else None) or {}

    def config(self, name: str) -> dict:
        rd = self.rd(name)
        return ((_read_json(rd / "config.json") or {}).get("config") or {}) if rd is not None else {}

    def scalars(self, name: str, tag: str) -> dict[int, float]:
        rd = self.rd(name)
        out: dict[int, float] = {}
        p = rd / "metrics" / "scalars.jsonl" if rd is not None else None
        if p is None or not p.is_file():
            return out
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("tag") == tag and isinstance(r.get("step"), int) and isinstance(r.get("value"), (int, float)):
                out[r["step"]] = float(r["value"])
        return out

    def trains(self) -> list[str]:
        return self.q.kind_names("train")

    def ran(self) -> list[str]:
        """The train items that started (the no-start item excluded)."""
        return [n for n in self.trains() if self.items[n]["attempts"]]

    def fault(self, action: str) -> tuple[dict | None, dict]:
        f = next((f for f in self.q.spec["faults"] if f["action"] == action), None)
        return f, (self.q.state["faults"].get(f["id"]) if f else None) or {}

    # the checks -------------------------------------------------------------------------------------------

    def build(self) -> dict:
        checks: dict[str, dict] = {}
        if self.trains():
            for n in range(1, 12):
                try:
                    ok, ev = getattr(self, f"check{n}")()
                except Exception as e:  # noqa: BLE001  a check that cannot be computed fails with the reason
                    ok, ev = False, {"error": f"{type(e).__name__}: {e}"}
                checks[str(n)] = {"pass": ok, "evidence": ev}
        for num, c in self.specs().items():
            checks[num] = c
        checks = dict(sorted(checks.items(), key=lambda kv: int(kv[0])))
        hb = {n: it.get("hb_max_gap_s") for n, it in self.items.items() if it.get("hb_max_gap_s") is not None}
        return dict(format=1, box=self.q.box, sha=self.q.s.sha, machine_id=self.q.s.machine_id, time_utc=_now_utc(),
                    overall="fail" if any(c["pass"] is False for c in checks.values()) else "pass", checks=checks,
                    faults=self.q.faults_list(), hb_max_gap_s=hb, alerts=self.alerts())

    def alerts(self) -> list[dict]:
        return self.q.watchdog_alerts()

    def check1(self):
        names = [n for n in self.trains() if n != SMOKE_NOSTART]
        bad = {n: self.items[n]["status"] for n in names
               if self.items[n]["status"] != "done" or not self.kinds(n, "smoke_sdpa")}
        first = next((n for n in names if self.kinds(n, "smoke_sdpa")), None)
        return not bad, dict(not_ok=bad, smoke_sdpa=(self.kinds(first, "smoke_sdpa") or [None])[0] if first else None)

    def check2(self):
        deadline_items = {f["item"] for f in self.q.spec["faults"] if f["action"] == "deadline"}
        rows, probe_peak, probe_reserved, steps = {}, {}, {}, {}
        for n in self.ran():
            v = self.scalars(n, "mem/step_peak_reserved_gb")
            rows[n] = max(v.values()) if v else None
            mp = self.kinds(n, "memory_probe")
            probe_peak[n] = max((max((e.get("peak_gb") or {}).values(), default=0.0) for e in mp), default=None)
            probe_reserved[n] = max((e.get("max_reserved_gb") or 0.0 for e in mp), default=None)
            steps[n] = self.summary(n).get("steps")
        nums = [x for d in (rows, probe_peak, probe_reserved) for x in d.values() if x is not None]
        short = [n for n in self.ran() if n not in deadline_items and n != SMOKE_NOSTART
                 and (steps[n] or 0) < SMOKE_MIN_STEPS]
        ok = bool(nums) and max(nums) <= SMOKE_VRAM_GB and not short
        return ok, dict(step_peak_reserved_gb=rows, probe_peak_gb=probe_peak, probe_max_reserved_gb=probe_reserved,
                        limit_gb=SMOKE_VRAM_GB, steps=steps, short=short, min_steps=SMOKE_MIN_STEPS)

    def sec_per_step(self, name: str) -> float | None:
        smoke_n = int(((self.config(name).get("smoke") or {}).get("steps")) or 100)
        evals = {int(e.get("step", e.get("at_step"))) for e in self.events(name)
                 if str(e.get("kind", "")).startswith("eval") and
                 isinstance(e.get("step", e.get("at_step")), (int, float))}
        v = [x for s, x in self.scalars(name, "time/step_s").items() if s > smoke_n and s not in evals]
        return statistics.median(v) if v else None

    def check3(self):
        proj, factor, sps = {}, {}, {}
        for n in self.ran():
            spec = self.q.spec_of(n)
            if not spec.get("plan_total_steps"):
                continue
            sps[n] = self.sec_per_step(n)
            if sps[n]:
                proj[n] = sps[n] * spec["plan_total_steps"] / 3600
                factor[n] = spec["plan_hours"] / proj[n]
        box1 = proj.get("smoke-p01")
        box2 = (max(proj["smoke-t06"], proj["smoke-p03"] + proj["smoke-p005"])
                if all(k in proj for k in ("smoke-t06", "smoke-p03", "smoke-p005")) else None)
        ok = box1 is not None and box2 is not None and box1 <= BOX1_H_MAX and box2 <= BOX2_H_MAX and \
            bool(factor) and all(f >= FACTOR_MIN for f in factor.values())
        return ok, dict(sec_per_step=sps, projected_h=proj, factor=factor, box1_h=box1, box2_h=box2,
                        box1_h_max=BOX1_H_MAX, box2_h_max=BOX2_H_MAX, factor_min=FACTOR_MIN)

    def check4(self):
        if not self.q.spec["gate"]:
            return None, dict(note="this box has no download gate")
        g = _read_json(self.q.box_state / fullrun.GATE_FILE)  # the boot's gate record (a chain: the box's)
        return (g or {}).get("verdict") == "pass", dict(gate=g)

    def check5(self):
        """The thread pools and the loader's wait, per train item that ran (build contract 5, check 5, as fixed after
        box 53693389).

        Threads: every launch's `threads` event (not only the last) has KITSUNE_THREADS_PER_GPU == t (the queue's), each
        of the six pools (fullrun.ENV_THREAD_POOLS, onstart's fix 2) an integer in [1, t], and torch_threads in
        [max(1, t // 2), t]. Not == t: torch starts its intra-op pool at mkl_get_max_threads(), which MKL_DYNAMIC (MKL's
        default, TRUE) caps at the physical cores, so a whole-machine quota on an SMT host gives min(t, cores): 16 of
        t = 32 on the 9950X (16 cores / 32 threads) of box 53693389, reproduced with the same torch 2.14 on the laptop
        (env 32 -> 24 of its 24 cores, env 20 -> 20, MKL_DYNAMIC=FALSE -> 32). The upper bound is the oversubscription
        fix 2 prevents; the lower one holds on hosts with at most 2 hardware threads a core (cores >= logical / 2 >=
        q / 2 >= t / 2), so a stray set_num_threads(1) still fails. interop_threads is evidence only (no env sets it).

        Loader: steady_wait over the merged time/data_wait_s and time/step_s rows (the trainer logs both every step,
        CORE_STEP_TAGS), the launches from the phase{name: "train", at_step} events and the smoke profiler's steps
        (smoke_profile_start: its warm-up step and the recorded_steps after it) left out; pass iff at least
        WAIT_MIN_STEPS steps remain, their fraction is < DATA_WAIT_MAX and no launch's first step waited
        STARTUP_WAIT_MAX_S or more. summary.throughput.data_wait_frac (the whole run, start-ups included) stays in the
        evidence as data_wait_frac_run."""
        want = os.environ.get(fullrun.ENV_THREADS_PER_GPU)
        try:
            t = int(want) if want is not None else None
        except ValueError:
            t = None
        rng = [max(1, t // 2), t] if t is not None and t >= 1 else None
        threads, wait, ok = {}, {}, rng is not None and bool(self.ran())
        for n in self.ran():
            rows = []
            for e in self.kinds(n, "threads"):
                env = e.get("env") or {}
                pools = {k: env.get(k) for k in fullrun.ENV_THREAD_POOLS}
                tt = e.get("torch_threads")
                good = rng is not None and env.get(fullrun.ENV_THREADS_PER_GPU) == str(t) and \
                    all(_int_in(v, 1, t) for v in pools.values()) and \
                    isinstance(tt, int) and not isinstance(tt, bool) and rng[0] <= tt <= rng[1]
                rows.append(dict(torch_threads=tt, interop_threads=e.get("interop_threads"), pools_env=pools, ok=good))
            threads[n] = rows
            launches = [int(e["at_step"]) for e in self.kinds(n, "phase")
                        if e.get("name") == "train" and isinstance(e.get("at_step"), int)]
            exclude = {s for e in self.kinds(n, "smoke_profile_start") if isinstance(e.get("at_step"), int)
                       for s in range(e["at_step"], e["at_step"] + int(e.get("recorded_steps") or 0) + 1)}
            w = steady_wait(self.scalars(n, "time/data_wait_s"), self.scalars(n, "time/step_s"), launches, exclude)
            w["data_wait_frac_run"] = (self.summary(n).get("throughput") or {}).get("data_wait_frac")
            starts = [v for v in w["startup_wait_s"].values() if v is not None]
            w["ok"] = w["data_wait_frac"] is not None and w["steps_measured"] >= WAIT_MIN_STEPS and \
                w["data_wait_frac"] < DATA_WAIT_MAX and all(v < STARTUP_WAIT_MAX_S for v in starts)
            wait[n] = w
            ok = ok and bool(rows) and all(r["ok"] for r in rows) and w["ok"]
        return ok, dict(threads_per_gpu=want, cpu_quota=os.environ.get(fullrun.ENV_CPU_QUOTA),
                        torch_threads_range=rng, threads=threads,
                        torch_threads={n: [r["torch_threads"] for r in rows] for n, rows in threads.items()},
                        data_wait=wait, data_wait_frac={n: w["data_wait_frac"] for n, w in wait.items()},
                        data_wait_max=DATA_WAIT_MAX, launch_skip_steps=LAUNCH_SKIP_STEPS,
                        wait_min_steps=WAIT_MIN_STEPS, startup_wait_max_s=STARTUP_WAIT_MAX_S,
                        peak_rss_gb={n: self.items[n].get("peak_rss_gb") for n in self.ran()},
                        host_mem_peak_gb=self.q.state.get("host_mem_peak_gb"))

    def check6(self):
        started = sorted(self.ran(), key=lambda n: self.items[n]["attempts"][0]["t0"])
        order_ok = started == [n for n in self.trains() if n in started]
        ev: dict = dict(start_order=started)
        stores_ok = True
        if "stores-ctc" in self.items and "stores-aed" in self.items:
            a, b = self.items["stores-ctc"]["attempts"], self.items["stores-aed"]["attempts"]
            stores_ok = bool(a) and bool(b) and a[-1].get("t1") is not None and a[-1]["t1"] <= b[0]["t0"]
        ctc_bad = []
        for n in self.ran():
            if self.q.spec_of(n)["family"] != "ctc":
                continue
            fp = self.kinds(n, "frame_preflight")
            reused = [((e.get("stores_reused") or {}).values()) for e in self.kinds(n, "data")]
            if not fp or not all(e.get("reused") is True for e in fp) or not reused or \
                    not all(all(v is True for v in r) for r in reused):
                ctc_bad.append(n)
        nostart = self.items.get(SMOKE_NOSTART, {}).get("status")
        nostart_ok = SMOKE_NOSTART not in self.items or nostart == "skipped"
        f, fs = self.fault("kill")
        kill_ok = True
        if f is not None:
            it = self.items[f["item"]]
            fired = (fs.get("fired_at") or {}).get("attempt") or 0
            later = it["attempts"][fired:] if fired else []
            kill_ok = bool(later) and bool(later[0].get("resume")) and later[0]["resume"] == it["run_dir"]
            trains = self.trains()
            nxt = trains[trains.index(f["item"]) + 1] if trains.index(f["item"]) + 1 < len(trains) else None
            kill_ok = kill_ok and (nxt is None or bool(self.items[nxt]["attempts"]))
            ev.update(killed_item=f["item"], resumed_in=later[0].get("resume") if later else None, next_item=nxt)
        ev.update(stores_ctc_before_aed=stores_ok, ctc_not_reused=ctc_bad, nostart_status=nostart)
        return order_ok and stores_ok and not ctc_bad and nostart_ok and kill_ok, ev

    def check7(self):
        ev: dict = {}
        ok = True
        for action in ("sigstop", "wipe_run_dir"):
            f, fs = self.fault(action)
            if f is None:
                continue
            good = fs.get("outcome") == "recovered"
            if action == "wipe_run_dir":
                good = good and self.items[f["item"]].get("check_resume") == "ok"
            ev[f["id"]] = dict(outcome=fs.get("outcome"), ok=good)
            ok = ok and good
        f, fs = self.fault(FREEZE)
        if f is not None:  # the window as it was held: from the fire to its release (hold_freeze), not the plan
            t0 = (fs.get("fired_at") or {}).get("wall")
            end = fs.get("released") or fs.get("window_end") or (t0 or 0) + float(f["seconds"])
            during = self.q.freeze_alerts(fs)
            ev[f["id"]] = dict(alerts_during=len(during), window=[t0, end], held_s=fs.get("held_s"),
                               release=fs.get("release"), orphan_s=(self.q.spec.get("watchdog") or {}).get("orphan_s"))
            ok = ok and bool(during)
        scratch = self.q.scratch_hub()
        timed = [n for n in self.ran() if self.kinds(n, "timed_state_upload_ok")]
        if timed:
            if scratch is None:
                return False, dict(ev, scratch="no scratch repo to list")
            stage = Path(self.q.s.state_dir) / "hub_reads" / "verdict"
            per_run = {}
            for n in timed:
                rid = self.rd(n).name
                listing = scratch.listing(f"runs/{rid}/checkpoints")
                dirs = sorted({p.split("/")[3] for p in listing})
                ptr = scratch.read_json(fullrun.scratch_pointer(rid), stage)
                newest = max(int(e.get("step") or 0) for e in self.kinds(n, "timed_state_upload_ok"))
                good = len(dirs) == 1 and ptr is not None and ptr.get("step") == newest
                per_run[n] = dict(states=dirs, pointer_step=(ptr or {}).get("step"), newest_upload=newest, ok=good)
                ok = ok and good
            commits = scratch.n_commits()
            ev.update(scratch=per_run, scratch_commits=commits, commits_max=SCRATCH_COMMITS_MAX)
            ok = ok and commits <= SCRATCH_COMMITS_MAX
        return ok, ev

    def forced(self) -> list[str]:
        """The smoke train items whose config forces the early-stop trigger (early_stop.min_delta_abs >=
        FORCED_MIN_DELTA_ABS: no dev eval can improve by that much)."""
        return [n for n in self.ran() if float((self.config(n).get("early_stop") or {}).get("min_delta_abs") or 0)
                >= FORCED_MIN_DELTA_ABS]

    def check8(self):
        forced = set(self.forced())
        natural = []
        for n in self.ran():
            if n in forced or (self.config(n).get("schedule") or {}).get("clock") != "epochs":
                continue
            pre = [e for e in self.kinds(n, "checkpoint") if e.get("reason") == "pre_cooldown"]
            names = {e.get("name") for e in pre}
            up = [e for e in self.kinds(n, "ckpt_upload_ok") if e.get("name") in names or not e.get("name")]
            if self.kinds(n, "phase") and any(e.get("name") == "cooldown" for e in self.kinds(n, "phase")) and pre \
                    and up:
                natural.append(n)
        f, _ = self.fault("deadline")
        dl_ok = True
        if f is not None:
            dl_ok = any(e.get("action") in ("schedule", "start", "compress")
                        for e in self.kinds(f["item"], "deadline_cooldown"))
        readouts = {n: (self.items[n]["status"], (self.items[n].get("result") or {}).get("jsut_cer_nostyle"))
                    for n in self.q.kind_names("readout")}
        ro_ok = all(st == "done" and v is not None for st, v in readouts.values())
        return bool(natural) and dl_ok and ro_ok, dict(scheduled_cooldown=natural, forced=sorted(forced),
                                                       deadline_item_ok=dl_ok, readouts=readouts)

    def check9(self):
        bad, dev_fp = [], {}
        for n in self.ran():
            ds = self.kinds(n, "dev_store")
            if not ds or any(e.get("n_in_train") != 0 for e in ds):
                bad.append(n)
            if self.q.spec_of(n)["family"] == "ctc":
                dev_fp[n] = [e for e in self.kinds(n, "frame_preflight") if e.get("store") == "dev"][:1]
                if not dev_fp[n]:
                    bad.append(n)
        return not bad and bool(self.ran()), dict(not_ok=sorted(set(bad)), frame_preflight_dev=dev_fp)

    def check10(self):
        forced = self.forced()
        trains, ev, ok = self.trains(), {}, bool(forced)
        for n in forced:
            es = [e for e in self.kinds(n, "early_stop") if e.get("action") == "cooldown"]
            it = self.items[n]
            nxt = trains[trains.index(n) + 1] if trains.index(n) + 1 < len(trains) else None
            good = bool(es) and bool((it.get("result") or {}).get("stopped_early")) and \
                it["attempts"][-1].get("rc") == 0 and (nxt is None or bool(self.items[nxt]["attempts"]))
            ev[n] = dict(early_stop=bool(es), stopped_early=(it.get("result") or {}).get("stopped_early"),
                         rc=it["attempts"][-1].get("rc"), next_item=nxt, ok=good)
            ok = ok and good
        return ok, dict(forced=forced, items=ev)

    def check11(self):
        st = {n: self.items[n]["status"] for n in self.q.kind_names("speed")}
        return all(v in ("done", "not_needed") for v in st.values()), dict(speed=st)

    # registry specs ---------------------------------------------------------------------------------------

    def specs(self) -> dict[str, dict]:
        """The registry's verdict specs by check number (AND over the specs of one number; a spec of a not_needed
        item is null, never false)."""
        by: dict[str, list[tuple[bool | None, dict]]] = {}
        for name in self.q.order:
            for spec in self.q.spec_of(name).get("verdict") or []:
                by.setdefault(spec["check"], []).append(self.spec_result(name, spec))
        out = {}
        for num, res in by.items():
            vals = [r for r, _ in res]
            ok = False if any(v is False for v in vals) else (True if any(v is True for v in vals) else None)
            out[num] = {"pass": ok, "evidence": [e for _, e in res]}
        return out

    def spec_result(self, name: str, spec: dict) -> tuple[bool | None, dict]:
        it = self.items[name]
        ev = dict(item=name, status=it["status"])
        if it["status"] == "not_needed":
            return None, dict(ev, note=(self.q.state["not_needed"].get(name) or {}))
        if it["status"] != "done":
            return False, ev
        if "json" not in spec:
            return True, ev
        try:
            rel = self.q.fill(spec["json"], name)
        except Q.QueueError as e:
            return False, dict(ev, error=str(e))
        path = self.q.root / rel
        adopted = ((self.q.state.get("resumed") or {}).get("items") or {}).get(name) == "done"
        if adopted and not path.is_file() and not Path(rel).is_absolute():
            ev["fetched"] = self.q.fetch_hub_file(rel)  # done on an earlier host: its out dir is on the Hub only
        val = _dig(_read_json(path), spec["path"])
        ev.update(json=self.q._rel(path) if path.exists() else str(path), path=spec["path"], value=val)
        ok = val is not None
        if "equals" in spec:
            ok = ok and val == spec["equals"]
        if "min" in spec:
            ok = ok and isinstance(val, (int, float)) and val >= spec["min"]
        if "max" in spec:
            ok = ok and isinstance(val, (int, float)) and val <= spec["max"]
        return bool(ok), ev


# ========================================================================================================== chain

EXIT_CHAIN_DESTROY = 5  # a chain that ended before its last stage trained: nothing unique on the disk -> destroy
CHAIN_FORMAT = 1
CHAIN_STEPS = ("s1_gate_part", "s1_gate", "s1_rest", "s2_boot", "s2_p01", "done")
BOOT2_MAX_ATTEMPTS = 2  # stage-2 bootstrap attempts that exited (an interrupted one does not count)
BOOT2_NO_RETRY = (2, 3)  # check_students refused, a plan or 01 refusal: the same attempt fails the same way
BOOT2_POLL_S = 30.0
BOOT2_KILL_GRACE_S = 60.0  # SIGTERM -> SIGKILL of the stage-2 bootstrap's session (every process group in it)
BOOT2_REAP_S = 30.0  # after the SIGKILL: how long the controller waits for the session to be gone
BOOT2_MIN_LEFT_S = 1800  # H2: the stage-2 bootstrap needs at least this long before its bound
BOOT2_PHASE_HB_MAX_S = 10800  # KITSUNE_PHASE_HB_MAX_S of the stage-2 bootstrap: phases without a timeout, 3 h each
BOOT2_MARK = "vast/bootstrap.sh"  # a live stage-2 bootstrap's command line (its identity after a controller restart)
STORES_ALLOWANCE_H, FINISH_ALLOWANCE_H = 0.5, 0.35  # the fit rule: the full-extent stores-ctc [X], and finish
REPORT_PART_TRIES = 2  # a report-only part (smoke-b): one retry in the same process, then recorded as failed
STAGE1_FILES = ("bootstrap_plan.json", "bootstrap_coverage.json", "bootstrap_timings.jsonl", fullrun.GATE_FILE)
PULL_FALLBACK_BYTES = 50e9  # stage2_timeouts without the stage-1 plan's record: the full extent's labels, generously
# the launch env the stage-2 bootstrap must not see: the gate ran at boot (its record stays in place), and a chain is
# never resumed as a chain
BOOT2_ENV_DROP = (fullrun.ENV_GATE_BYTES, fullrun.ENV_GATE_MAX_H, fullrun.ENV_RESUME, fullrun.ENV_RESUME_RESET,
                  fullrun.ENV_RESUME_SETS)


def phase_budget_s(tries: int, minutes: float) -> int:
    """vast/bootstrap.sh phase_budget_s: the longest `retry <tries> timeout -k <=60 <minutes>m ...` runs (every attempt
    with its kill grace, retry's pauses of 60, 120, ... s), plus 10 min."""
    return int(tries * (minutes * 60 + 60) + 30 * tries * (tries - 1) + 600)


def boot2_budget_s(pull_min: float, rebuild_min: float, phase_hb_max_s: float = BOOT2_PHASE_HB_MAX_S) -> int:
    """The stage-2 bootstrap's own bound, its phases' worst cases added up: plan (retry 3 x 10 min) and pull_derived
    (3 x 30 min), check_students and coverage (no timeout: phase_hb_max_s each), and the label pull alongside the
    rebuild (the longer of their three attempts; the rebuild's never below 60 min, as bootstrap's toucher)."""
    return (phase_budget_s(3, 10) + phase_budget_s(3, 30) + 2 * int(phase_hb_max_s)
            + max(phase_budget_s(3, pull_min), phase_budget_s(3, max(rebuild_min, 60))))


def chain_gate(verdict: dict | None, part: str, part_rc: int, sha: str | None, *, local: Path | None = None) -> dict:
    """The automatic gate between a chain's stages (addendum E.3.1): pass only when the gate part ended with rc 0,
    its verdict is there, is its own (box) and of this checkout (sha, when KITSUNE_SHA is set), and every one of
    checks 1-11 is true (false, null or missing all fail; checks 12-16 never enter it). rc 4 always fails: a train or
    stores item that failed for good, or the part's stop_at halt. local: the verdict file, for its sha256."""
    checks = (verdict or {}).get("checks") or {}
    got = {n: (checks.get(n) or {}).get("pass") for n in fullrun.GATE_CHECKS}
    failed = [n for n, v in got.items() if v is not True]
    problems = ([] if verdict else ["verdict missing"]) \
        + ([] if part_rc == 0 else [f"part {part} ended with rc {part_rc}"]) \
        + ([] if not verdict or verdict.get("box") == part else [f"verdict of box {verdict.get('box')!r}"]) \
        + ([] if not verdict or not sha or verdict.get("sha") == sha else [f"verdict sha {verdict.get('sha')}"])
    ev3 = (checks.get("3") or {}).get("evidence") or {}
    return dict(part=part, part_rc=part_rc, result="pass" if not failed and not problems else "fail", checks=got,
                failed=failed, problems=problems, verdict=fullrun.box_verdict_path(part),
                verdict_sha256=_sha256_file(local) if local is not None and Path(local).is_file() else None,
                projected_box1_h=ev3.get("box1_h") if isinstance(ev3, dict) else None, time_utc=_now_utc())


def stage2_timeouts(state_dir: Path, root: Path, rebuild1: str, rebuild2: str) -> dict:
    """The stage-2 bootstrap's per-attempt timeouts (addendum E.4.2), computed on the box: what is left to download
    (the last stage's upstream bytes less stage 1's: tar 8 is counted twice in both, as 01 reads it), its labels + 2 GB
    to pull, at the boot's download gate rate (else kitsune.extent's 40 MB/s sizing), through kitsune.netgate.timeouts.
    The record is the one stage 1's plan read (chain/stage1/bootstrap_plan.json). Without it (a record that cannot be
    read), the gate's full download and PULL_FALLBACK_BYTES: longer timeouts, never shorter."""
    from kitsune import extent, netgate

    state_dir, root = Path(state_dir), Path(root)
    gate = _read_json(state_dir / fullrun.GATE_FILE) or {}
    rate = float(gate.get("rate_bytes_s") or extent.REBUILD_BYTES_PER_S)
    try:
        plan = json.loads((state_dir / fullrun.CHAIN_DIR / "stage1" / "bootstrap_plan.json").read_text(
            encoding="utf-8"))
        rec = extent.load_record(Path(plan["record"]))
        cfg1 = json.loads((root / rebuild1).read_text(encoding="utf-8"))
        cfg2 = json.loads((root / rebuild2).read_text(encoding="utf-8"))
        s2, s1 = extent.sizing(rec, cfg2), extent.sizing(rec, cfg1)
        rest, pullb, source = max(0.0, s2["down_gb"] - s1["down_gb"]) * 1e9, (s2["labels_gb"] + 2) * 1e9, "record"
    except Exception as e:  # noqa: BLE001  never a crash of the controller: conservative bytes instead
        log(f"stage-2 timeouts without the extent record ({type(e).__name__}: {e}): the gate's full download")
        rest = float(os.environ.get(fullrun.ENV_GATE_BYTES) or netgate.GATE_REF_GB * 1e9)
        pullb, source = PULL_FALLBACK_BYTES, f"fallback: {type(e).__name__}"
    pull_min, rebuild_min = netgate.timeouts(rate, rest, pullb)
    return dict(rest_bytes=int(rest), pull_bytes=int(pullb), rate_bytes_s=rate, pull_min=int(pull_min),
                rebuild_min=int(rebuild_min), source=source)


class ChainController:
    """A chain box (addendum E; `python -m kitsune.full_queue run --box p01-chain`, which vast/supervise.py runs): its
    parts, one at a time, each an ordinary FullQueue with its own state dir ($KITSUNE_STATE/chain/<part>/) that writes
    its summary and verdict at the standalone box's Hub paths (full/box-<part>/...).

      stage 1   the gate part (full-smoke) with deadline = stop_at = gate_by; the watchdog's mode file set to stage 2's
                (stop 3600); the gate (chain_gate: decided, recorded in chain.json once, put on the Hub); then every
                other stage-1 part (smoke-b: report only, deadline = stop_at = the stage-1 sub-deadline; one retry,
                then recorded as failed), whatever the gate said
      gate      failed -> exit 5: finish verifies the verdict, logs and events on the Hub, then destroys
      handover  stage 1's bootstrap records copied to chain/stage1/, the stage-2 timeouts and bound computed; box 1
                must still fit (else exit 5); the stage-2 bootstrap (vast/bootstrap.sh, KITSUNE_CHAIN_STAGE=2, the
                last stage's rebuild config) runs as a synchronous child in its own session, killed at its bound
                (boot2_until); rc 0 done, rc 2/3 not retried, anything else retried once; a failure exits 5
      stage 2   box 1 must still fit after the bootstrap (else exit 5), then its parts (p01), whose rc is the chain's
                (0 destroy, 4 stop, 1 restart: p01 resumes from its queue.json)

    Every step is recorded in $KITSUNE_STATE/chain/chain.json (a restart goes on from it: ended parts are not run
    again, the gate is never evaluated twice, a stage-2 bootstrap left running is killed when its identity is
    verified, else ignored, and started again); the chain summary ($KITSUNE_STATE/queue_summary.json, put at
    full/box-<chain>/queue_summary.json) on every step change and at the end. The controller beats train_hb only between
    steps: never while a part runs (the part's queue beats, and a smoke's freeze fault must hold) nor while the stage-2
    bootstrap runs (its phases' bounded touchers keep it fresh, so a hung phase still goes stale). A part never returns
    while its fired freeze window is open (FullQueue.hold_freeze), so neither the controller's beat at the part's end
    nor set_mode's stop mode can cut a freeze short of the alert check 7 needs."""

    def __init__(self, box: str, settings: FullSettings | None = None, registry: dict | None = None, *,
                 bootstrap_cmd: list[str] | None = None, boot_poll_s: float = BOOT2_POLL_S,
                 boot_kill_grace_s: float = BOOT2_KILL_GRACE_S, boot_reap_s: float = BOOT2_REAP_S):
        s = settings or FullSettings()
        reg = registry if registry is not None else s.registry
        try:
            reg = fullrun.load_registry(reg, root=s.root, check_files=False) if reg is not None else \
                fullrun.load_registry(None, root=s.root)
            if not fullrun.is_chain(box, reg):
                raise fullrun.RegistryError(f"box {box} is not a chain box")
            self.stages = fullrun.chain_stages(box, reg)
        except fullrun.RegistryError as e:
            raise Q.QueueError(str(e)) from None
        self.box, self.s, self.registry = box, s, reg
        self.cspec = reg["boxes"][box]
        self.root, self.state_dir = Path(s.root), Path(s.state_dir)
        self.path = self.state_dir / fullrun.CHAIN_DIR / fullrun.CHAIN_STATE
        self.hb = self.state_dir / fullrun.TRAIN_HB
        self.registry_sha256 = hashlib.sha256(json.dumps(reg, sort_keys=True, separators=(",", ":"),
                                                         ensure_ascii=False).encode()).hexdigest()
        self.uploader = s.uploader if s.uploader is not None else (Q.HubUploader(s.out_repo) if s.out_repo else None)
        self.bootstrap_cmd = list(bootstrap_cmd or ["bash", (self.root / "vast" / "bootstrap.sh").as_posix()])
        self.boot_poll_s, self.boot_kill_grace_s = float(boot_poll_s), float(boot_kill_grace_s)
        self.boot_reap_s = float(boot_reap_s)
        self._boot_proc = None
        self.st = self._load()

    # ------------------------------------------------------------------------------------------------ state

    def _box_deadline(self) -> float | None:
        try:
            return float((self.state_dir / fullrun.DEADLINE_FILE).read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def _first_boot(self, deadline: float | None) -> float:
        """$STATE/first_boot (onstart); else the box deadline less KITSUNE_MAX_HOURS (the chain's max_hours), with a
        warning; never the controller's own start, which would move on every restart (a local run without either
        takes chain.json's creation time, which chain.json keeps)."""
        try:
            return float((self.state_dir / "first_boot").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            pass
        if deadline is not None:
            mh = float(os.environ.get(fullrun.ENV_MAX_HOURS) or self.cspec["max_hours"])
            log(f"warning: no {self.state_dir / 'first_boot'}: first boot = the deadline - {mh:g} h")
            return deadline - mh * 3600
        log("warning: neither first_boot nor deadline in the state dir (a local run): first boot = now")
        return time.time()

    def _load(self) -> dict:
        if self.path.is_file():
            st = json.loads(self.path.read_text(encoding="utf-8"))
            if st.get("box") != self.box:
                raise Q.QueueError(f"{self.path} is chain {st.get('box')!r}'s state, not {self.box!r}'s")
            return st
        deadline = self._box_deadline()
        first = self._first_boot(deadline)
        s1 = self.stages[0]
        gate_by = first + float(s1["gate_by_hours"]) * 3600
        s1_deadline = first + float(s1["max_hours"]) * 3600
        if deadline is not None:
            s1_deadline = min(deadline, s1_deadline)
        parts = {p: dict(stage=st["stage"], status="pending", rc=None, tries=0, fails=0, started=None, ended=None,
                         queue_started=None) for st in self.stages for p in st["parts"]}
        return dict(format=CHAIN_FORMAT, box=self.box, sha=self.s.sha, created_utc=_now_utc(), started=time.time(),
                    gate_box=s1["gate_box"], first_boot=first, deadline=deadline, gate_by=gate_by,
                    stage1_deadline=s1_deadline, step=CHAIN_STEPS[0], stage=1, parts=parts, gate=None, boot2=None,
                    watchdog_mode=None, final=None)

    def save(self):
        Q._atomic_json(self.path, self.st)

    def event(self, kind: str, **fields):
        """A chain record in $KITSUNE_STATE/events.jsonl (the parts' queues write theirs in chain/<part>/)."""
        p = self.state_dir / Q.EVENTS_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall": time.time(), "source": "chain", "box": self.box, "kind": kind, **fields},
                               default=str) + "\n")
        log(f"{kind}: " + json.dumps(fields, default=str)[:600])

    def beat(self):
        heartbeat.beat(self.hb, force=True)

    def _step(self, step: str, stage: int | None = None):
        self.st["step"] = step
        if stage is not None:
            self.st["stage"] = stage
        self.save()
        self.put_summary()

    def summary(self, status: str, reason: str | None, rc: int | None) -> dict:
        st = self.st
        parts, verdicts = {}, {}
        for p, ps in st["parts"].items():
            smoke = bool(self.registry["boxes"][p]["smoke"])
            parts[p] = dict(stage=ps["stage"], status=ps["status"], rc=ps["rc"], queue_started=ps["queue_started"],
                            summary=fullrun.box_summary_path(p), verdict=fullrun.box_verdict_path(p) if smoke else None)
            v = _read_json(fullrun.part_state_dir(p, self.state_dir) / fullrun.VERDICT_FILE) if smoke else None
            if isinstance(v, dict):
                verdicts[p] = dict(overall=v.get("overall"),
                                   checks={n: (c or {}).get("pass") for n, c in (v.get("checks") or {}).items()})
        ended = None if status == "running" else ((st.get("final") or {}).get("wall") or time.time())
        return dict(format=1, kind=fullrun.CHAIN_KIND, box=self.box, status=status, reason=reason, rc=rc,
                    sha=self.s.sha, machine_id=self.s.machine_id, container_id=self.s.container_id,
                    registry_sha256=self.registry_sha256, first_boot=st["first_boot"], deadline=st["deadline"],
                    gate_by=st["gate_by"], stage1_deadline=st["stage1_deadline"], stage=st["stage"], step=st["step"],
                    started=st["started"], ended=ended, gate=st["gate"], boot2=st["boot2"], parts=parts,
                    verdicts=verdicts)

    def put_summary(self, status: str = "running", reason: str | None = None, rc: int | None = None):
        """$KITSUNE_STATE/queue_summary.json (vast/supervise.py reads its reason, kind and stage), put at
        full/box-<chain>/queue_summary.json under the controller heartbeat (bounded)."""
        path = self.state_dir / fullrun.SUMMARY_FILE
        Q._atomic_json(path, self.summary(status, reason, rc))
        if self.uploader is None:
            return
        try:
            with heartbeat.beating(self.hb, max_s=CTL_BEAT_MAX_S):
                self.uploader.put_file(path, fullrun.box_summary_path(self.box))
        except Exception as e:  # noqa: BLE001  finish.py verifies and re-puts it at the end
            log(f"{fullrun.box_summary_path(self.box)} upload failed: {type(e).__name__}: {e}")

    def finish(self, status: str, reason: str | None, rc: int) -> int:
        """The chain's end: recorded (a restart returns rc), the chain summary put."""
        self.st["final"] = dict(rc=rc, status=status, reason=reason, wall=time.time())
        self.st["step"] = "done"
        self.save()
        self.event("chain_end", status=status, reason=reason, rc=rc)
        self.beat()
        self.put_summary(status, reason, rc)
        return rc

    # ------------------------------------------------------------------------------------------------ parts

    def part_settings(self, part: str, deadline: float | None, stop_at: float | None) -> FullSettings:
        return replace(self.s, state_dir=fullrun.part_state_dir(part, self.state_dir), box_state_dir=self.state_dir,
                       deadline=deadline, stop_at=stop_at, registry=self.registry, uploader=self.uploader)

    def queue(self, part: str, deadline: float | None, stop_at: float | None) -> FullQueue:
        return FullQueue(part, self.part_settings(part, deadline, stop_at), registry=self.registry)

    def part_window(self, part: str) -> tuple[float | None, float | None]:
        """(deadline, stop_at) of a part (E.4.5): the gate part gate_by, stage 1's other parts the stage-1
        sub-deadline, the last stage's parts the box deadline file's (None, None)."""
        ps = self.st["parts"][part]
        if part == self.st["gate_box"]:
            return self.st["gate_by"], self.st["gate_by"]
        if ps["stage"] == 1:
            return self.st["stage1_deadline"], self.st["stage1_deadline"]
        return None, None

    def _part_start(self, part: str):
        ps = self.st["parts"][part]
        ps.update(status="running", tries=int(ps["tries"]) + 1, started=ps["started"] or time.time())
        self.save()
        self.event("chain_part_start", part=part, attempt=ps["tries"], stage=ps["stage"])
        self.beat()
        self.put_summary()

    def _part_end(self, part: str, rc, why: str | None = None):
        ps = self.st["parts"][part]
        ps["rc"] = rc
        if rc in (EXIT_OK, EXIT_STOP):
            ps.update(status="ended", ended=time.time())
        self.save()
        self.event("chain_part_end", part=part, rc=rc, status=ps["status"], why=why)
        self.beat()
        self.put_summary()

    def run_part(self, part: str) -> int:
        """A part whose failure is the chain's (the gate part, the last stage's): its rc (1 exits the controller, which
        the supervisor restarts: the part resumes from its queue.json). An exception propagates (the same)."""
        ps = self.st["parts"][part]
        if ps["status"] == "ended":
            return int(ps["rc"])
        self._part_start(part)
        q = self.queue(part, *self.part_window(part))
        self._queue_started(part, q)
        rc = q.run()
        self._part_end(part, rc)
        return rc

    def _queue_started(self, part: str, q):
        """parts.<part>.queue_started = the part queue's started, saved and put on the Hub before the part runs: a
        chain that dies in stage 2 leaves this summary, and launch --box p01 --resume matches the Hub's p01 summary to
        this rental by it (addendum E.8), so it must be there for the whole of box 1's first run."""
        self.st["parts"][part]["queue_started"] = q.state.get("started")
        self.save()
        self.put_summary()

    def run_report_part(self, part: str):
        """A report-only part (smoke-b): rc 0 or 4 ends it; rc 1 or an exception while it is built or run is retried
        once in this process (a new queue resumes from its queue.json), then recorded as failed (event
        chain_part_failed) and the chain goes on. Finish's lean sync still uploads its run dirs."""
        ps = self.st["parts"][part]
        while ps["status"] not in ("ended", "failed"):
            self._part_start(part)
            why = None
            try:
                q = self.queue(part, *self.part_window(part))
                self._queue_started(part, q)
                rc = q.run()
            except Exception as e:  # noqa: BLE001  a report-only part never ends the chain
                rc, why = None, f"{type(e).__name__}: {e}"
                log(f"part {part} raised: {why}")
            if rc not in (EXIT_OK, EXIT_STOP):
                ps["fails"] = int(ps["fails"]) + 1
                why = why or f"exit {rc}"
                if ps["fails"] >= REPORT_PART_TRIES:
                    ps.update(status="failed", ended=time.time())
                    self.event("chain_part_failed", part=part, rc=rc, why=why, fails=ps["fails"])
            self._part_end(part, rc, why)

    # ------------------------------------------------------------------------------------------------ the gate

    def mode(self) -> tuple[str, int]:
        """The watchdog mode from the gate part's end on: the last stage's watchdog (stop 3600 for p01-chain)."""
        wd = self.stages[-1]["watchdog"]
        return wd["action"], int(wd["orphan_s"])

    def set_mode(self):
        """$KITSUNE_STATE/watchdog_mode "<action> <orphan_s>" (tmp + rename): vast/watchdog.sh reads it every poll and
        takes it over its env (stage 1's alert 600). Written once the gate part has returned, before the gate, whatever
        it says: a failed gate never leaves the box in alert mode while finish runs, and smoke B runs in stop mode as it
        does standalone. Idempotent (a restart writes it again when it is missing or different)."""
        action, orphan_s = self.mode()
        f = self.state_dir / fullrun.WATCHDOG_MODE_FILE
        want = f"{action} {orphan_s}\n"
        try:
            if f.read_bytes() == want.encode() and self.st.get("watchdog_mode"):
                return
        except OSError:
            pass
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_bytes(want.encode())  # LF on every platform: bash's `read` would keep a CR in the limit
        tmp.replace(f)
        self.st["watchdog_mode"] = dict(action=action, orphan_s=orphan_s, written_utc=_now_utc())
        self.save()
        self.event("watchdog_mode", action=action, orphan_s=orphan_s)

    def _verdict_current(self, v, final: dict) -> bool:
        """A verdict that run() wrote after its final: parseable, and not older than queue.json's final.wall (time_utc
        has whole seconds)."""
        if not isinstance(v, dict) or not isinstance(v.get("checks"), dict):
            return False
        try:
            t = datetime.fromisoformat(str(v.get("time_utc"))).timestamp()
        except ValueError:
            return False
        return not final.get("wall") or t + 1.0 >= float(final["wall"])

    def evaluate_gate(self) -> dict:
        """E.3.1: the gate part's verdict made sure of (a restart between run()'s final and its verdict leaves none:
        the verdict and the summary are built again from its queue.json), then chain_gate. Recorded once."""
        part = self.st["gate_box"]
        pdir = fullrun.part_state_dir(part, self.state_dir)
        vpath = pdir / fullrun.VERDICT_FILE
        final = (_read_json(pdir / Q.STATE_FILE) or {}).get("final") or {}
        v = _read_json(vpath)
        if not self._verdict_current(v, final):
            try:
                q = self.queue(part, *self.part_window(part))
                q.write_verdict()
                if final:
                    q.write_summary(final.get("status"), final.get("reason"), final.get("rc"))
                self.event("chain_verdict_rebuilt", part=part, had=v is not None)
            except Exception as e:  # noqa: BLE001  the gate then fails: verdict missing
                self.event("chain_verdict_rebuild_failed", part=part, error=f"{type(e).__name__}: {e}")
                vpath.unlink(missing_ok=True)
            v = _read_json(vpath)
        rc = self.st["parts"][part]["rc"]
        gate = chain_gate(v if isinstance(v, dict) else None, part, int(rc) if rc is not None else -1, self.s.sha,
                          local=vpath)
        self.event("chain_gate", part=gate["part"], part_rc=gate["part_rc"], result=gate["result"],
                   failed=gate["failed"], problems=gate["problems"], checks=gate["checks"],
                   verdict_sha256=gate["verdict_sha256"], projected_box1_h=gate["projected_box1_h"])
        return gate

    # ------------------------------------------------------------------------------------------------ handover

    def need_s(self) -> int:
        """The fit rule (E.6): the last stage's training (the gate's projected box-1 hours, else its train items'
        max_hours) + its readouts' max_hours + the stores allowance + finish, and its deadline reserve."""
        items = [it for p in self.stages[-1]["parts"] for it in self.registry["boxes"][p]["items"]]
        proj = (self.st.get("gate") or {}).get("projected_box1_h")
        train_h = float(proj) if isinstance(proj, (int, float)) else sum(
            float(it["max_hours"] or 0) for it in items if it["kind"] == "train")
        readout_h = sum(float(it.get("max_hours") or 0) for it in items if it["kind"] == "readout")
        reserve = max(float(self.registry["boxes"][p]["deadline_reserve_min"]) for p in self.stages[-1]["parts"])
        return int(3600 * (train_h + readout_h + STORES_ALLOWANCE_H + FINISH_ALLOWANCE_H) + 60 * reserve)

    def handover(self) -> int | None:
        """H1-H3 (E.3.3): stage 1's bootstrap records to chain/stage1/ (the stage-2 bootstrap overwrites the plan and
        the coverage), the stage-2 timeouts, its bound boot2_until = min(now + its own budget, deadline - need_s), the
        refusal when less than BOOT2_MIN_LEFT_S remain (box 1 no longer fits: exit 5), event chain_handover. None: go
        on to the stage-2 bootstrap."""
        s1dir = self.state_dir / fullrun.CHAIN_DIR / "stage1"
        s1dir.mkdir(parents=True, exist_ok=True)
        for name in STAGE1_FILES:
            if (self.state_dir / name).is_file():
                shutil.copy2(self.state_dir / name, s1dir / name)
        t = stage2_timeouts(self.state_dir, self.root, self.stages[0]["rebuild"], self.stages[-1]["rebuild"])
        now, need, deadline = time.time(), self.need_s(), self.st["deadline"]
        budget = boot2_budget_s(t["pull_min"], t["rebuild_min"])
        until = now + budget if deadline is None else min(now + budget, float(deadline) - need)
        self.st["boot2"] = dict(status="running", until=until, t0=now, need_s=need, budget_s=budget,
                                pull_min=t["pull_min"], rebuild_min=t["rebuild_min"], rest_bytes=t["rest_bytes"],
                                pull_bytes=t["pull_bytes"], rate_bytes_s=t["rate_bytes_s"], timeouts=t["source"],
                                attempts=[])
        self.save()
        left = None if deadline is None else round(float(deadline) - now)
        self.event("chain_handover", left_s=left, need_s=need, boot2_until=until, rest_bytes=t["rest_bytes"],
                   pull_min=t["pull_min"], rebuild_min=t["rebuild_min"], budget_s=budget)
        if until - now < BOOT2_MIN_LEFT_S:
            self.st["boot2"]["status"] = "failed"
            return self.finish("failed", f"box 1 no longer fits: need {need} s, left {left} s (the stage-2 bootstrap "
                                         f"would get {round(until - now)} s)", EXIT_CHAIN_DESTROY)
        self._step("s2_boot", stage=2)
        return None

    # ------------------------------------------------------------------------------------------------ stage-2 bootstrap

    def boot_env(self) -> dict:
        b = self.st["boot2"]
        env = dict(os.environ, **self.s.env)
        for k in BOOT2_ENV_DROP:
            env.pop(k, None)
        env.update({fullrun.ENV_CHAIN_STAGE: "2", fullrun.ENV_CONFIG: self.stages[-1]["rebuild"],
                    fullrun.ENV_PULL_TIMEOUT_MIN: str(b["pull_min"]),
                    fullrun.ENV_REBUILD_TIMEOUT_MIN: str(b["rebuild_min"]),
                    fullrun.ENV_PHASE_HB_MAX_S: str(BOOT2_PHASE_HB_MAX_S), fullrun.ENV_STATE: str(self.state_dir),
                    "KITSUNE_DIR": str(self.root), fullrun.ENV_JOB: fullrun.JOB, fullrun.ENV_BOX: self.box})
        return env

    def _proc_file(self, pid, name: str) -> str | None:
        try:
            return (Path(self.s.proc_root) / str(int(pid)) / name).read_bytes().decode("utf-8", "replace")
        except (OSError, ValueError, TypeError):
            return None

    def _stat(self, pid) -> list[str] | None:
        """/proc/<pid>/stat's fields from field 3 on (after the command name): [0] state, [2] pgrp, [3] session,
        [19] starttime."""
        stat = self._proc_file(pid, "stat")
        try:
            return stat.rsplit(")", 1)[1].split() if stat else None
        except IndexError:
            return None

    def starttime(self, pid) -> str | None:
        """Field 22 of /proc/<pid>/stat (the process's start in clock ticks since boot): with the pid, a process's
        identity across a controller restart (a pid alone may be recycled after a container restart)."""
        f = self._stat(pid)
        return f[19] if f and len(f) > 19 else None

    def pidns(self) -> str | None:
        """The controller's pid namespace (the /proc/self/ns/pid link, "pid:[<inode>]"), recorded with each attempt: a
        container restart makes a new one, whose pids name other processes."""
        try:
            return os.readlink(Path(self.s.proc_root) / "self" / "ns" / "pid")
        except (OSError, NotImplementedError, AttributeError):
            return None

    def session(self, sid) -> dict[int, int]:
        """{pid: pgid} of every live (not zombie) process in session `sid` (field 6 of /proc/<pid>/stat), never the
        controller itself. The stage-2 bootstrap leads its own session (start_new_session), and everything under it stays
        in that session - unlike its process group: GNU timeout, which bootstrap wraps its phases in (plan,
        pull_derived, the label pull, 01's rebuild), calls setpgid(0, 0), so the timeout and its command sit in a group
        of their own that a kill of the bootstrap's group never reaches. Empty off posix or without /proc."""
        out, me = {}, os.getpid()
        try:
            names = [d.name for d in Path(self.s.proc_root).iterdir() if d.name.isdigit()]
        except OSError:
            return out
        for name in names:
            f = self._stat(name)
            try:
                if f and f[0] != "Z" and int(f[3]) == int(sid) and int(name) != me:
                    out[int(name)] = int(f[2])
            except (ValueError, IndexError):
                continue
        return out

    def _leader_alive(self, pid, proc=None) -> bool:
        if proc is not None:
            return proc.poll() is None  # also reaps it: a zombie leader is gone
        f = self._stat(pid)
        return f is not None and f[0] != "Z"

    def _boot_left(self, pid, proc=None) -> bool:
        """The bootstrap, or any process of its session, still running."""
        return self._leader_alive(pid, proc) or (os.name == "posix" and bool(self.session(pid)))

    def signal_boot(self, pid, sig, proc=None):
        """sig to the bootstrap's whole session on posix: its own process group and every other group in the session
        (each timeout-wrapped phase's), never the controller's own group; on Windows (no sessions) terminate / kill the
        process itself."""
        if os.name != "posix":
            try:
                if proc is not None:
                    proc.terminate() if sig == signal.SIGTERM else proc.kill()
                else:
                    os.kill(int(pid), signal.SIGTERM)  # TerminateProcess: never signal 0, which is CTRL_C_EVENT there
            except (OSError, ValueError):
                pass
            return
        for g in sorted({int(pid), *self.session(pid).values()} - {os.getpgrp()}):
            try:
                os.killpg(g, sig)
            except OSError:  # the group is gone (ProcessLookupError) or not ours
                pass

    def kill_boot(self, pid, proc=None):
        """SIGTERM to the bootstrap's session, SIGKILL after boot_kill_grace_s to whatever is left of it, then up to
        boot_reap_s for it to go: the bootstrap and every phase under it, a timeout-wrapped 01 rebuild included, so no
        later attempt and no p01 store ever meets an old 01 still writing the same data root."""
        self.signal_boot(pid, signal.SIGTERM, proc)
        end = time.time() + self.boot_kill_grace_s
        while time.time() < end and self._boot_left(pid, proc):
            time.sleep(min(0.5, max(0.0, end - time.time())))
        if self._boot_left(pid, proc):
            self.signal_boot(pid, getattr(signal, "SIGKILL", signal.SIGTERM), proc)
            end = time.time() + self.boot_reap_s
            while time.time() < end and self._boot_left(pid, proc):
                time.sleep(0.2)
            if self._boot_left(pid, proc):
                log(f"stage-2 bootstrap session {pid} still has processes after SIGKILL: "
                    f"{sorted(self.session(pid))}")
        if proc is not None:
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                pass

    def _interrupted(self, a: dict):
        """An attempt a dead controller left without an end (E.4.1): its session is killed only when it is verified to
        be that attempt's - the bootstrap itself by its stored starttime and a vast/bootstrap.sh command line, or, with
        the bootstrap gone, the processes it left in its session while the pid namespace is still the attempt's (a
        session id's pid is not reused while a process is in that session) - never a recycled pid's; recorded as
        interrupted, which does not count toward BOOT2_MAX_ATTEMPTS."""
        pid, want = a.get("pid"), a.get("starttime")
        cmd = (self._proc_file(pid, "cmdline") or "").replace("\0", " ")
        leader = pid is not None and want is not None and self.starttime(pid) == want and BOOT2_MARK in cmd
        orphans = (not leader and pid is not None and os.name == "posix" and a.get("pidns") is not None
                   and a.get("pidns") == self.pidns() and bool(self.session(pid)))
        if leader or orphans:
            self.kill_boot(pid)
        a.update(t1=time.time(), end="interrupted", killed=leader or orphans)
        self.save()
        self.event("chain_bootstrap_end", attempt=self.st["boot2"]["attempts"].index(a) + 1, rc=a.get("rc"),
                   end="interrupted", seconds=round(a["t1"] - a["t0"], 1), killed=leader or orphans,
                   orphans=orphans)

    def boot_attempt(self) -> dict:
        """One stage-2 bootstrap run: a child in its own session, its output appended to $STATE/logs/bootstrap-s2.log,
        waited for (the controller does not beat meanwhile) and killed at boot2_until."""
        b = self.st["boot2"]
        logs = self.state_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.beat()  # the last controller beat before the bootstrap's own phase touchers take over
        with open(logs / "bootstrap-s2.log", "ab") as out:
            kw = dict(cwd=str(self.root), env=self.boot_env(), stdout=out, stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL)
            if os.name == "posix":
                kw["start_new_session"] = True
            proc = subprocess.Popen(self.bootstrap_cmd, **kw)
        self._boot_proc = proc
        a = dict(t0=time.time(), t1=None, pid=proc.pid, starttime=self.starttime(proc.pid), pidns=self.pidns(),
                 rc=None, end=None)
        b["attempts"].append(a)
        self.save()
        n = len(b["attempts"])
        self.event("chain_bootstrap_start", attempt=n, pid=proc.pid, until=b["until"], argv=self.bootstrap_cmd)
        until = float(b["until"])
        while True:
            try:
                rc = proc.wait(timeout=max(0.05, min(self.boot_poll_s, until - time.time())))
                end = "exit"
                break
            except subprocess.TimeoutExpired:
                if time.time() >= until:
                    log(f"stage-2 bootstrap past its bound ({until}): killing its session {proc.pid}")
                    self.kill_boot(proc.pid, proc)
                    rc, end = proc.poll(), "timeout"
                    break
        if end == "exit" and os.name == "posix" and self.session(proc.pid):
            # an exited bootstrap left processes in its session (its EXIT trap stops the label pull; anything else
            # would still write the data root under the next attempt or p01's stores): they go with it
            left = sorted(self.session(proc.pid))
            log(f"stage-2 bootstrap exited {rc} and left {left} in its session: killing them")
            self.kill_boot(proc.pid, proc)
            a["left"] = left
        self._boot_proc = None
        a.update(t1=time.time(), rc=rc, end=end)
        self.save()
        self.event("chain_bootstrap_end", attempt=n, rc=rc, end=end, seconds=round(a["t1"] - a["t0"], 1))
        self.beat()
        self.put_summary()
        return a

    def boot2(self) -> int | None:
        """The stage-2 bootstrap (E.4.1): attempts until one exits 0 (None: go on), rc 2/3 (not retried), the second
        exited failure, or the bound -> exit 5, the chain failed (nothing unique on the disk: destroy)."""
        b = self.st["boot2"]
        for a in b["attempts"]:
            if a.get("t1") is None:
                self._interrupted(a)

        def failed(why: str) -> int:
            b["status"] = "failed"
            return self.finish("failed", why, EXIT_CHAIN_DESTROY)

        while b["status"] != "done":
            if time.time() >= float(b["until"]):
                last = b["attempts"][-1] if b["attempts"] else {}
                if last.get("end") == "exit":  # a failed attempt, and no time left for its retry: name its exit
                    return failed(f"stage 2 bootstrap failed (exit {last.get('rc')}) and its bound "
                                  f"{round(float(b['until']))} passed before a retry")
                return failed(f"stage 2 bootstrap timed out (its bound {round(float(b['until']))} passed)")
            a = self.boot_attempt()
            if a["end"] == "timeout":
                return failed(f"stage 2 bootstrap timed out after {round(a['t1'] - a['t0'])} s (attempt "
                              f"{len(b['attempts'])})")
            if a["rc"] == 0:
                b["status"] = "done"
                self.save()
                break
            exited = sum(1 for x in b["attempts"] if x.get("end") == "exit")
            if a["rc"] in BOOT2_NO_RETRY:
                return failed(f"stage 2 bootstrap failed (exit {a['rc']}: a refusal, not retried)")
            if exited >= BOOT2_MAX_ATTEMPTS:
                return failed(f"stage 2 bootstrap failed (exit {a['rc']})")
        return None

    # ------------------------------------------------------------------------------------------------ the run

    def run(self) -> int:
        fin = self.st.get("final")
        if fin:
            # the chain summary written again (idempotent): a controller that died between finish()'s chain.json and
            # its summary left the summary "running", and vast/supervise.py destroys on exit 5 only with the chain
            # summary's gate_failed / failed status
            log(f"chain {self.box} already ended ({fin['status']}: {fin.get('reason')}); nothing to do")
            self.put_summary(fin["status"], fin.get("reason"), int(fin["rc"]))
            return int(fin["rc"])
        try:
            return self._run()
        finally:
            proc = self._boot_proc
            if proc is not None and self._boot_left(proc.pid, proc):  # the controller raised while its bootstrap ran
                log(f"controller ended while the stage-2 bootstrap runs: killing its session {proc.pid}")
                self.kill_boot(proc.pid, proc)

    def _run(self) -> int:
        st = self.st
        self.event("chain_start", step=st["step"], stage=st["stage"],
                   restart=bool(st["parts"][st["gate_box"]]["tries"]), first_boot=st["first_boot"],
                   deadline=st["deadline"], gate_by=st["gate_by"], stage1_deadline=st["stage1_deadline"])
        self.beat()
        self.put_summary()
        s1 = self.stages[0]
        if st["step"] == "s1_gate_part":
            rc = self.run_part(s1["gate_box"])
            if rc not in (EXIT_OK, EXIT_STOP):
                self.put_summary("failed", f"gate part {s1['gate_box']} exited {rc}; the supervisor restarts the chain",
                                 EXIT_FAIL)
                return EXIT_FAIL
            self.set_mode()
            self._step("s1_gate")
        if st["step"] == "s1_gate":
            self.set_mode()
            if st["gate"] is None:
                st["gate"] = self.evaluate_gate()
                self.save()
            self._step("s1_rest")
        if st["step"] == "s1_rest":
            self.set_mode()
            for part in s1["parts"][1:]:
                self.run_report_part(part)
            gate = st["gate"]
            if gate["result"] != "pass":
                return self.finish("gate_failed", f"chain gate failed: checks {gate['failed']} problems "
                                                  f"{gate['problems']}", EXIT_CHAIN_DESTROY)
            rc = self.handover()
            if rc is not None:
                return rc
        if st["step"] == "s2_boot":
            rc = self.boot2()
            if rc is not None:
                return rc
            need, deadline, now = self.need_s(), st["deadline"], time.time()
            if deadline is not None and now + need > float(deadline):
                return self.finish("failed", f"box 1 no longer fits: need {need} s, left {round(float(deadline) - now)}"
                                             f" s", EXIT_CHAIN_DESTROY)
            self._step("s2_p01")
        if st["step"] == "s2_p01":
            for part in self.stages[-1]["parts"]:
                rc = self.run_part(part)
                if rc == EXIT_FAIL or rc not in (EXIT_OK, EXIT_STOP):
                    self.put_summary("failed", f"part {part} exited {rc}; the supervisor restarts the chain", rc)
                    return rc
                if rc == EXIT_STOP:
                    return self.finish("halted", f"part {part} halted (rc 4: the disk is kept)", EXIT_STOP)
            return self.finish("complete", None, EXIT_OK)
        return int((st.get("final") or {}).get("rc", EXIT_FAIL))


def chain_plan(box: str, registry: dict | None = None, root: Path | None = None) -> list[dict]:
    """The `plan` command of a chain: one row per stage, then each part's items (with its stage and part)."""
    reg = registry if registry is not None else fullrun.load_registry(None, root=root)
    rows = []
    for st in fullrun.chain_stages(box, reg):
        rows.append(dict(stage=st["stage"], parts=st["parts"], gate_box=st["gate_box"], rebuild=st["rebuild"],
                         gate_by_hours=st["gate_by_hours"], max_hours=st["max_hours"], watchdog=st["watchdog"]))
        for part in st["parts"]:
            rows += [dict(row, stage=st["stage"], part=part) for row in plan_items(part, reg)]
    return rows


# ======================================================================================================= commands


def plan_items(box: str, registry: dict | None = None, root: Path | None = None) -> list[dict]:
    """The `plan` command: the box's registry items in order (nothing started)."""
    reg = registry if registry is not None else fullrun.load_registry(None, root=root)
    return [dict(item=it["name"], kind=it["kind"], needs=it["needs"], config=it.get("config"), of=it.get("of"),
                 max_hours=it.get("max_hours"), droppable=it["droppable"], stall_min=it["stall_min"])
            for it in fullrun.box_items(box, reg)]


def build_stores(config: str, sets: list[str], eval_only: bool = False) -> int:
    """The label stores of `config` exactly as the trainer's setup_data builds them (build_train_store, build_eval_store
    with frames None: every eval store the config's runs and readouts open, and build_dev_store when the trainer has it
    and the config's dev slice is on), or with eval_only the eval store alone. Under a bounded heartbeat
    (STORES_BEAT_MAX_S: a slow full-extent build is never killed). Its log ends with `stores_done {peak_rss_gb,
    wall_s}`."""
    t0 = time.time()
    with heartbeat.beating(every_s=STORES_BEAT_EVERY_S, max_s=STORES_BEAT_MAX_S):
        D = Q._load_trainer()
        cfg = D.load_config(config, sets)

        class _Log:
            @staticmethod
            def event(kind, **fields):
                print(json.dumps({"event": kind, **fields}, default=str)[:2000], flush=True)

        built = {}
        if not eval_only:
            train = D.build_train_store(cfg, _Log)
            built["train"] = len(train)
        ev = D.build_eval_store(cfg, _Log)
        built["eval"] = len(ev)
        dev_builder = getattr(D, "build_dev_store", None)
        if not eval_only and dev_builder is not None:
            dev = dev_builder(cfg, _Log)  # None unless the config's dev slice is on
            built["dev"] = len(dev) if dev is not None else None
    peak = None
    try:
        import resource

        peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)  # KiB on Linux
    except (ImportError, AttributeError):
        pass
    print(f"stores: {json.dumps(built)}", flush=True)
    print("stores_done " + json.dumps({"peak_rss_gb": peak, "wall_s": round(time.time() - t0, 1)}), flush=True)
    return 0


def check_resume(run_dir: str) -> int:
    """The trainer's resume_check on a pulled run dir (CPU): 0 the store matches its newest state; 3 the planner
    fingerprint or the number of utterances differs (the item fails, no retry); 1 the store is not built yet or
    anything else (retried)."""
    try:
        D = Q._load_trainer()
        res = D.resume_check(Path(run_dir))
    except Exception as e:  # noqa: BLE001  retryable
        print(f"check-resume: {type(e).__name__}: {e}", flush=True)
        return EXIT_FAIL
    print("check-resume " + json.dumps(res, default=str), flush=True)
    if res.get("ok"):
        return EXIT_OK
    return EXIT_FAIL if res.get("reason") == "store not built" else EXIT_REFUSED


def done_on_hub(rid: str, s_it: dict, runs: Hub, scratch: Hub | None, stage: Path) -> tuple[int, dict] | None:
    """(steps, result) when a train item's run is done on the Hub, else None:
      - the box summary (the source of truth) says done, and its export checkpoints/step_<steps>/ is on the Hub;
      - or the box died between the trainer's end and the queue's item-end put: the run's own summary.json is complete
        with its export there, but only when that summary cannot be an earlier run's. A continuation (--resume-reset /
        --resume-set) runs in the same run dir, which keeps the first run's complete summary.json (on the Hub too)
        until the continuation's own end: so no Hub state may lie past its steps (the scratch pointer, a runs-repo
        full_step_<N>), and when the box summary records a continuation of this run, the summary must count more
        resume_resets than the run had before it."""
    res = s_it.get("result") or {}
    listing = runs.listing(f"runs/{rid}/checkpoints")

    def exported(steps: int) -> bool:
        return any(p.startswith(f"runs/{rid}/checkpoints/step_{steps}/") for p in listing)

    if s_it.get("status") == "done" and res.get("steps") is not None and exported(int(res["steps"])):
        return int(res["steps"]), res
    hub = runs.read_json(f"runs/{rid}/summary.json", stage) or {}
    if hub.get("status") != "complete" or hub.get("steps") is None:
        return None
    steps = int(hub["steps"])
    cont = s_it.get("continuation") or {}
    if cont.get("run_id") == rid and int(hub.get("resume_resets") or 0) <= int(cont.get("resume_resets_before") or 0):
        return None  # the summary of the run the continuation went on from
    newest = max(_states_of(listing, rid), default=0)
    try:
        ptr = scratch.read_json(fullrun.scratch_pointer(rid), stage) if scratch is not None else None
    except ResumeRefused:
        ptr = None  # not JSON: pull_run refuses it, if it comes to that
    if isinstance((ptr or {}).get("step"), int):
        newest = max(newest, ptr["step"])
    if newest > steps or not exported(steps):
        return None  # a state past the summary's end: a later run (a continuation) went on in this run dir
    return steps, dict(run_id=rid, steps=steps, status="complete", epochs=hub.get("epochs"),
                       stopped_early=bool(hub.get("stopped_early")), end_reason=hub.get("end_reason"),
                       resume_resets=int(hub.get("resume_resets") or 0))


def resume_pull(box: str, root: Path, *, runs: Hub, scratch: Hub | None, registry: dict, state_dir: Path,
                reset: list[str] | None = None, sets: dict[str, list[str]] | None = None,
                sha: str | None = None) -> dict:
    """The new-host pull (the module docstring): reads full/box-<box>/queue_summary.json, pulls every train item's run
    (pull_run) and writes $KITSUNE_STATE/resume_plan.json. ResumeRefused for what no retry fixes."""
    reset, sets = list(reset or []), dict(sets or {})
    root, state_dir = Path(root), Path(state_dir)
    stage = state_dir / "hub_reads" / f"resume-{time.time_ns()}"
    raw = runs.download(fullrun.box_summary_path(box), stage)
    if raw is None:
        raise ResumeRefused(f"{runs.repo} has no {fullrun.box_summary_path(box)}: nothing to resume")
    summary_bytes = raw.read_bytes()
    summ = json.loads(summary_bytes)
    shutil.rmtree(stage, ignore_errors=True)
    s_items = summ.get("items") or {}
    spec = fullrun.box_spec(box, registry)
    trains = [it for it in spec["items"] if it["kind"] == "train"]
    rid_of = {it["name"]: fullrun.run_id_of(s_items[it["name"]]["run_dir"]) for it in trains
              if (s_items.get(it["name"]) or {}).get("run_dir")}
    if unknown := [x for x in [*reset, *sets] if x not in rid_of.values()]:
        raise ResumeRefused(f"run id(s) {unknown} are not a training run of box {box}'s queue summary "
                            f"(it has {sorted(rid_of.values())})")
    special = set(reset) | set(sets)
    entries: dict[str, dict] = {}

    def entry(it, status, **kw):
        base = dict(kind=it["kind"], status=status, run_dir=None, out=None, result=None, state=None, step=None,
                    source=None, reset=False, sets=[], steps=None, kitsune_sha=summ.get("sha"),
                    resume_resets_before=None, continuation=None)
        base.update(kw)
        entries[it["name"]] = base

    def resume(it, got, **kw):
        entry(it, "resume", run_dir=f"runs/{rid_of[it['name']]}", state=got["state"], step=got["step"],
              source=got["source"], kitsune_sha=got["kitsune_sha"] or summ.get("sha"), **kw)

    for it in trains:
        name = it["name"]
        s_it = s_items.get(name) or {}
        rid = rid_of.get(name)
        if rid is None:
            entry(it, "fresh")
            continue
        steps = (s_it.get("result") or {}).get("steps")
        cont = s_it.get("continuation") if (s_it.get("continuation") or {}).get("run_id") == rid else None
        if rid in reset:
            got = pull_run(rid, root, runs, scratch, pick="pre_cooldown")
            resume(it, got, reset=True, sets=["schedule.resume_reset=true", *sets.get(rid, [])], steps=steps,
                   resume_resets_before=got["resume_resets"])
            continue
        if rid not in special:
            done = done_on_hub(rid, s_it, runs, scratch, state_dir / "hub_reads")
            if done is not None:
                steps, result = done
                dependants = [x for x in spec["items"] if x.get("of") == name and not x.get("of_box") and not (
                    (s_items.get(x["name"]) or {}).get("status") == "done"
                    and (s_items.get(x["name"]) or {}).get("verified") is True)]
                pull_run(rid, root, runs, scratch, with_state=False, weights_step=steps if dependants else None)
                entry(it, "done", run_dir=f"runs/{rid}", result=result, steps=steps)
                continue
        if cont is not None and rid not in special:
            # a continuation the host was lost during: on from the newest Hub state that holds its reset
            before = int(cont.get("resume_resets_before") or 0)
            got = pull_run(rid, root, runs, scratch, min_resets=before + 1)
            if got["state"] is not None:
                resume(it, got, steps=steps, continuation=cont)
                continue
            # no Hub state holds the reset yet: the continuation is applied again, as it was launched
            log(f"{rid}: no Hub state holds its continuation's reset yet; the continuation starts again "
                f"({'reset' if cont.get('reset') else 'set'} {cont.get('sets')})")
            if cont.get("reset"):
                got = pull_run(rid, root, runs, scratch, pick="pre_cooldown")
            else:
                got = pull_run(rid, root, runs, scratch, refuse_past_cooldown=True)
            if got["state"] is None:
                raise ResumeRefused(f"{rid}: its continuation has no full state on the Hub to start from")
            resume(it, got, reset=bool(cont.get("reset")), sets=list(cont.get("sets") or []), steps=steps,
                   resume_resets_before=got["resume_resets"])
            continue
        got = pull_run(rid, root, runs, scratch, refuse_past_cooldown=rid in sets)
        if got["state"] is None:
            log(f"warning: {rid} has no full state on the Hub; {name} starts fresh")
            entry(it, "fresh", run_dir=None)
            continue
        once = ["schedule.resume_reset=true", *sets[rid]] if rid in sets else []
        resume(it, got, sets=once, steps=steps, resume_resets_before=got["resume_resets"] if once else None)
    # a readout or eval of a run that goes on (a reset, a set, a lost continuation, a started run) runs again
    reopened = {it["name"] for it in trains if entries[it["name"]]["status"] != "done"}
    for it in spec["items"]:
        if it["kind"] == "train":
            continue
        s_it = s_items.get(it["name"]) or {}
        # done only with its output verified on the Hub (a readout's or eval's out dir, the speed dir's speed.json);
        # an item done but not yet uploaded starts fresh
        if it["kind"] != "stores" and s_it.get("status") == "done" and s_it.get("verified") is True and \
                it.get("of") not in reopened:
            entry(it, "done", out=s_it.get("out"), run_dir=s_it.get("run_dir"), result=s_it.get("result"))
        else:
            entry(it, "fresh")
    for name, e in entries.items():
        if sha and e.get("kitsune_sha") and e["kitsune_sha"] != sha:
            log(f"warning: {name} was made at kitsune {e['kitsune_sha']}, this checkout is {sha}")
    plan = dict(format=1, box=box, created_utc=_now_utc(), summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
                items=entries)
    Q._atomic_json(state_dir / fullrun.RESUME_PLAN, plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kitsune.full_queue", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "plan", "resume-pull"):
        p = sub.add_parser(c)
        p.add_argument("--box", default=os.environ.get(fullrun.ENV_BOX), choices=fullrun.ALL_BOX_NAMES)
    sub.choices["run"].add_argument("--gpus", default=None, help="comma list of CUDA_VISIBLE_DEVICES values")
    sub.choices["resume-pull"].add_argument("--root", default=str(REPO), help="the box's checkout")
    b = sub.add_parser("build-stores")
    b.add_argument("--config", required=True)
    b.add_argument("--eval-only", action="store_true")
    b.add_argument("--set", action="append", default=[])
    c = sub.add_parser("check-resume")
    c.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build-stores":
        try:
            return build_stores(args.config, args.set, args.eval_only)
        except Exception as e:  # noqa: BLE001  the queue retries a failed build
            log(f"build-stores failed: {type(e).__name__}: {e}")
            return EXIT_FAIL
    if args.cmd == "check-resume":
        return check_resume(args.run_dir)
    if not args.box:
        ap.error(f"--box (or {fullrun.ENV_BOX}) is required")
    chain = args.box in fullrun.CHAIN_NAMES  # a chain name is always a chain entry (the registry refuses one without)
    if args.cmd == "plan":
        try:
            for row in (chain_plan(args.box) if chain else plan_items(args.box)):
                print(json.dumps(row))
        except fullrun.RegistryError as e:
            log(f"refused: {e}")
            return EXIT_FAIL
        return EXIT_OK
    if args.cmd == "resume-pull" and chain:  # addendum E.8: a chain is not resumed as a chain
        log(f"resume refused: {fullrun.chain_resume_hint(args.box)}")
        return EXIT_REFUSED
    if args.cmd == "resume-pull":
        root = Path(args.root)
        try:
            out_repo = os.environ.get(fullrun.ENV_OUT_REPO)
            if not out_repo:
                raise ResumeRefused(f"{fullrun.ENV_OUT_REPO} is not set: no runs repo to resume from")
            scratch_repo = os.environ.get(fullrun.ENV_SCRATCH_REPO)
            plan = resume_pull(args.box, root, runs=Hub(out_repo), scratch=Hub(scratch_repo) if scratch_repo else None,
                               registry=fullrun.load_registry(None, root=root), state_dir=fullrun.state_dir(),
                               reset=fullrun.parse_resume_reset(os.environ.get(fullrun.ENV_RESUME_RESET)),
                               sets=fullrun.parse_resume_sets(os.environ.get(fullrun.ENV_RESUME_SETS)),
                               sha=os.environ.get(fullrun.ENV_SHA))
        except (ResumeRefused, fullrun.RegistryError, ValueError) as e:
            log(f"resume refused: {e}")
            return EXIT_REFUSED
        except Exception as e:  # noqa: BLE001  transient: bootstrap retries
            log(f"resume-pull failed: {type(e).__name__}: {e}")
            return EXIT_FAIL
        for name, e in plan["items"].items():
            log(f"{name}: {e['status']}" + (f" from {e['state']} ({e['source']})" if e.get("state") else "")
                + (f" sets {e['sets']}" if e.get("sets") else ""))
        return EXIT_OK
    gpus = [g for g in args.gpus.split(",") if g] if args.gpus else None
    try:
        if chain:
            return ChainController(args.box, FullSettings(gpus=gpus)).run()
        return FullQueue(args.box, FullSettings(gpus=gpus)).run()
    except Q.QueueError as e:
        log(f"refused: {e}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
