"""Distil the Kitsune student from the stored teacher outputs: the viability run (run sheet D19-D52, spec section 8).

One run, in order (every phase is an event in runs/<run_id>/events.jsonl):
  1. setup     config (JSON; every key overridable with --set a.b=value), RunLogger (env capture, TensorBoard,
               open-format metrics, HF sync), the student from 03 in fp32 master weights (bf16 autocast for the body;
               the LM head in fp32 outside autocast, as in the teacher pass), rel-pos patch, BatchNorm frozen, the
               frozen decoder pos_emb, AdamW, the decoupled L2-SP anchor, train/eval stores and the step planner
  2. smoke     (fresh runs) the memory probe first (CUDA: forward+backward on the three worst micro-batches with the
               optimizer state's bytes reserved and, when steps accumulate several micro-batches, the gradients held;
               on OOM halve micro_audio_s down to memory.min_micro_audio_s, then per-layer gradient checkpointing; on
               Windows under a cap at the free VRAM, see cap_vram), then LogMel vs the HF extractor,
               forward+backward on the longest padded micro-batch with finite gradients, padded rows == the same
               utterances run alone (mean KL over the shortest smoke.pad_utts; see padded_row_check), SDPA backends, a
               FLOP count for the MFU estimate and an HF upload round trip
  3. step 0    eval of the untrained student: teacher-forced on the eval sets and the train probe, greedy on the fixed
               subsets. It runs BEFORE any weight update (the spec lists the 100 smoke steps first, which would make
               "step 0" a 100-step model)
  4. train     WSD on the loop clock (evals and checkpoints included; see wsd_lr; every switch of LR phase is an
               `lr_phase` event), L2-SP after every optimizer step,
               gradients clipped (the pre-clip norm is logged). The first smoke.steps steps are the smoke run: finite
               losses, the loss trend, and throughput - below smoke.min_audio_s_per_s audio-s/s the run exits with
               code 3 (ThroughputTooLow). A full state is written right after them, and before the cooldown starts
  5. end       final weights + full state (uploaded in the background), final eval with greedy decode of the FULL eval
               sets, verdict (kitsune.evaluate.verdict), summary.json, uploads awaited, final forced sync; exit 0

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
                   ckpt.keep_local are kept; uploaded at ckpt.upload_full_at ("pre_cooldown", "end"). Written as
                   <name>.tmp/ and renamed, so a crash never leaves a torn one.
`--resume <full_step_N dir | run dir>` restores all of it into the same run dir and continues the same schedule and
time budget; the config comes from the checkpoint, with this invocation's --set overrides applied on top (--config is
then ignored).

Time: in schedule.clock "wall" (the real run) the budget T = train_hours of loop wall-clock from the first training
step, evals and checkpoints included; the step-0 eval and the final eval are outside it. The clock continues across a
resume (time between a crash and the resume is not counted). On the vast box T is also clipped at every loop start so
the end phase finishes before the watchdog's fixed deadline ($KITSUNE_DEADLINE or $KITSUNE_STATE/deadline; see
fit_budget). schedule.clock "steps" makes T = schedule.max_steps optimizer steps, and eval.every_steps /
ckpt.*_every_steps replace the minute cadences when set: deterministic schedules for tests and debugging.
schedule.clock "epochs" makes T = the optimizer steps of schedule.epochs full passes over the train set (plan_epochs),
with the warmup capped at ceil(10 %) of them; eval.every_epochs runs the eval at every N-th epoch end instead of the
minute/step cadence (the last epoch's is the end phase's final eval).

Overfit sanity runs (configs/overfit_*.json, scripts/run_overfit_tests.cmd): subset.train_audio_s / eval_audio_s train
and evaluate on seeded subsets of about that many seconds (audio_subset; logged as `subset` events with every id),
eval.probe_is_train makes the whole train subset the teacher-forced probe and eval.probe_greedy_audio_s adds a greedy
decode of part of it, so memorisation (probe KL and CER vs teacher -> 0) shows next to the held-out eval;
specaug.enabled false trains un-augmented. optim.offload "cpu" keeps AdamW and fp32 master weights in host memory
(CpuOffloadAdamW), for a GPU that holds the weights and gradients but not the optimizer state (the 8 GB laptop).

Test hook: the environment variable KITSUNE_CRASH_AT_STEP=<n> raises a RuntimeError just before step n (exercises
the crash path and --resume; it lives in the environment so a resumed run does not inherit it from the config).

Heavy imports (transformers, the eval/student/logging modules) happen inside main(): DataLoader workers are spawned
and re-import this file as __mp_main__, and should only pay for torch.

Usage:
  python scripts/04_distill.py --config configs/viability.json --set hf.output_repo=<user>/<repo>
  python scripts/04_distill.py --config configs/smoke_laptop.json
  python scripts/04_distill.py --config configs/overfit_10s.json
  python scripts/04_distill.py --config configs/viability.json --resume runs/<run_id>/checkpoints/full_step_<N>
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
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
from kitsune.kd import L2SP, kd_losses, kd_objective, module_key  # noqa: E402
from kitsune.patches import assert_bn_frozen, patch_relpos_once_per_batch, train_mode  # noqa: E402

EXIT_OK, EXIT_THROUGHPUT, EXIT_FAIL = 0, 3, 1
SR = 16000
FULL_RE = re.compile(r"^full_step_(\d+)$")
WEIGHTS_RE = re.compile(r"^step_(\d+)$")
TERMS = ("kl", "ce", "top1_match", "student_entropy_coarse", "teacher_entropy_coarse", "student_tail", "teacher_tail",
         "teacher_p1")
TERM_TAGS = dict(kl="loss/kl", ce="loss/ce", top1_match="tok/top1", student_entropy_coarse="tok/entropy_student",
                 teacher_entropy_coarse="tok/entropy_teacher", student_tail="tok/tail_student",
                 teacher_tail="tok/tail_teacher", teacher_p1="tok/teacher_p1")
BUCKETS = (("p1_gt_0.99", lambda p1: p1 > 0.99), ("p1_lt_0.9", lambda p1: p1 < 0.9))

# Defaults = the viability run (configs/viability.json spells out the same values). Paths are relative to the repo root.
DEFAULTS = {
    "run_name": "viability-b20x2560", "student": "students/b20x2560-d4", "data_root": "data",
    "teacher_root": "teacher_out", "second_root": "second_out", "selection": "selection/viability.parquet",
    "cache_dir": "cache", "runs_root": "runs",
    "sources": ["reazon_small", "emilia_yodas", "galgame"],
    "eval_sets": ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"], "mix": "natural",
    # smoke runs: seeded id subsets, small caches. *_audio_s: seeded subsets of about that many seconds of audio
    # (audio_subset; the overfit runs), the eval one pooled over eval_sets and decoded greedily in full at every eval
    "subset": {"train_utts": None, "eval_utts_per_set": None, "train_audio_s": None, "eval_audio_s": None},
    "device": "auto", "autocast": "bfloat16",  # autocast: "bfloat16" or "none" (fp32; the CPU tests)
    "loss": {"w_kl": 1.0, "w_ce": 0.8, "l2sp_lambda": 0.05},
    "specaug": {"enabled": True, "freq_masks": 2, "freq_width": 27, "time_masks_min": 2, "time_masks_max": 5,
                "time_width": 0.05},
    # offload "cpu": AdamW and fp32 master weights in host memory (CpuOffloadAdamW), for a GPU that holds the weights
    # and gradients but not the AdamW state; offload_fused: torch's fused CPU AdamW kernel there instead of foreach.
    # `fused` is the on-device (CUDA) optimizer's flag
    "optim": {"lr": 1e-4, "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0, "clip": 1.0, "fused": True,
              "max_nonfinite_skips": 3, "offload": "none", "offload_fused": False},
    "schedule": {"warmup_steps": 300, "cooldown_frac": 0.2, "train_hours": 4.0, "clock": "wall", "max_steps": None,
                 "end_reserve_min": 30, "epochs": None},
    "batch": {"step_audio_s": 1500, "micro_audio_s": 400, "pool_micro": 50, "max_dec_len": 200},
    "memory": {"grad_ckpt": "auto", "probe_longest_bucket": True, "min_micro_audio_s": 100, "max_oom_skips": 3},
    "perf": {"relpos_patch": True, "compile": False, "num_workers": "auto", "prefetch": 4, "tf32": True,
             "peak_tflops": 312.0, "train_exact_dither": False},
    # every_epochs: eval at the end of every N-th epoch instead of every_min / every_steps. probe_is_train: the probe
    # is the whole train set (a small subset) rather than its in_probe rows. probe_greedy_audio_s: also greedy-decode
    # a seeded ~N s of the probe (null: subset.eval_audio_s; both null: no probe decode)
    "eval": {"every_min": 20, "every_steps": None, "greedy_subset": 500, "probe": True, "final_full_greedy": True,
             "batch_s": 400, "check_baselines": True, "every_epochs": None, "probe_is_train": False,
             "probe_greedy_audio_s": None},
    "ckpt": {"weights_every_min": 30, "full_local_every_min": 30, "weights_every_steps": None,
             "full_every_steps": None, "keep_local": 2, "upload_full_at": ["pre_cooldown", "end"],
             "full_after_smoke": True},
    "log": {"layer_stats_every": 100, "hist_every": 1000, "train_utts_flush": 500, "sync_every_min": 10,
            "samples_per_eval": 8, "capture_env": True},
    "hf": {"output_repo": None, "private": True},
    "smoke": {"enabled": True, "steps": 100, "min_audio_s_per_s": 600, "pad_utts": 32, "pad_max_mean_kl": 0.05,
              "pad_min_argmax_agree": None, "require_loss_decrease": True},
    "seed": 1234,
}


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


def validate(cfg: dict):
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
    if not isinstance(cfg["specaug"]["enabled"], bool):
        raise SystemExit(f"specaug.enabled must be true or false, got {cfg['specaug']['enabled']}")
    for a, b in (("train_audio_s", "train_utts"), ("eval_audio_s", "eval_utts_per_set")):
        if sub[a] is not None and sub[b]:
            raise SystemExit(f"subset.{a} and subset.{b} are two ways to pick the same subset: set one")
    for key, v in (("subset.train_audio_s", sub["train_audio_s"]), ("subset.eval_audio_s", sub["eval_audio_s"]),
                   ("eval.probe_greedy_audio_s", ev_cfg["probe_greedy_audio_s"])):
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0):
            raise SystemExit(f"{key} must be null or a number of seconds > 0, got {v}")
    if ev_cfg["every_epochs"] is not None and not _pos_int(ev_cfg["every_epochs"]):
        raise SystemExit(f"eval.every_epochs must be null or an int >= 1, got {ev_cfg['every_epochs']}")
    if ev_cfg["probe_is_train"] and not ev_cfg["probe"]:
        raise SystemExit("eval.probe_is_train needs eval.probe")


def _pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def epoch_mode(cfg: dict) -> bool:
    """Evals at epoch ends (the epoch clock or eval.every_epochs): eval records, events and scalars carry the epoch."""
    return cfg["schedule"]["clock"] == "epochs" or bool(cfg["eval"]["every_epochs"])


def rpath(value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


# -------------------------------------------------------------------------------------------------------- schedule


def wsd_lr(peak: float, step: int, t: float, T: float, warmup_steps: int, cooldown_frac: float) -> tuple[float, int]:
    """Warmup-stable-decay. `step` is the 1-based optimizer step about to be taken, `t` the progress when it starts
    (loop seconds, or steps done) out of the budget `T`. Linear warmup over warmup_steps, constant until
    t >= (1 - cooldown_frac) T, then peak * (1 - sqrt((t - t_c) / (T - t_c))). Returns (lr, phase) with phase
    0 warmup, 1 stable, 2 cooldown."""
    warm = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
    t_c = (1.0 - cooldown_frac) * T
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
    shared memory under every sharing strategy (file_system shm_opens too, so KITSUNE_SHARING does not help), and a
    container left with Docker's 64 MB dies with a bus error at the first step. Up to num_workers * prefetch
    micro-batches of up to micro_audio_s padded float32 audio are in flight: keep them within half the free space by
    prefetching less, then using fewer workers, then decoding in-process. Returns (workers, prefetch, change or None)."""
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
    return n, p, dict(shm_free_gb=round(free / 2**30, 3), workers=[nw, n], prefetch=[prefetch, p],
                      micro_audio_s=micro_audio_s)


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
    ds: object = None
    planner: object = None
    probe_ids: list = field(default_factory=list)
    probe_greedy_ids: list = field(default_factory=list)
    greedy_ids: list = field(default_factory=list)
    src_index: dict = field(default_factory=dict)
    uploader: object = None
    bn0: dict = field(default_factory=dict)
    gen: object = None
    resumed_from: Path | None = None
    budget_s: float | None = None  # wall-clock T clipped to the instance deadline (fit_budget); None = train_hours
    vram_cap_gb: float | None = None  # the caching allocator's cap on Windows (cap_vram); None = no cap
    loop_t0: float | None = None
    t_start: float = field(default_factory=time.time)
    st: dict = field(default_factory=lambda: dict(
        step=0, train_s=0.0, smoke_done=False, pre_cooldown_done=False, step0_done=False,
        last_eval_t=0.0, last_eval_step=0, last_weights_t=0.0, last_weights_step=0, last_full_t=0.0,
        last_full_step=0, memory={}, flops_per_padded_s=None, history=[], nonfinite_skips=0, nonfinite_total=0,
        oom_skips=0, audio_s=0.0, tokens=0, step_time_s=0.0, smoke_losses=[], smoke_audio_s=0.0, smoke_time_s=0.0,
        epoch=0, epoch_progress=0.0, resumes=0, weights=[], fulls=[], last_objective=None, lr_phase=None,
        last_eval_epoch=0, total_steps=None))

    def clock(self) -> float:
        """Loop clock in seconds: continues across resumes, frozen outside the training loop."""
        return self.st["train_s"] + (time.monotonic() - self.loop_t0 if self.loop_t0 is not None else 0.0)

    def progress(self) -> tuple[float, float]:
        s = self.cfg["schedule"]
        if s["clock"] == "steps":
            return float(self.st["step"]), float(s["max_steps"])
        if s["clock"] == "epochs":  # total_steps: plan_epochs
            return float(self.st["step"]), float(self.st["total_steps"])
        return self.clock(), self.budget_s if self.budget_s is not None else float(s["train_hours"]) * 3600

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


def hf_api():
    from huggingface_hub import HfApi  # looked up at call time: tests substitute a fake

    return HfApi()


# ------------------------------------------------------------------------------------------------------- uploads


class Uploader:
    """Checkpoint uploads to <repo>:runs/<run_id>/checkpoints/<name>/ in one background thread, so a 1-9 GB upload
    never blocks training. Every attempt is an event; failures are retried and never raise (vast/finish.py uploads
    again and verifies before an instance is destroyed). A directory with a pending upload is never rotated away."""

    def __init__(self, api, repo: str | None, run_id: str, private: bool, log, retries=(15, 60, 180)):
        self.api, self.repo, self.run_id, self.private, self.log, self.retries = api, repo, run_id, private, log, retries
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-upload")
        self.pending: dict[Path, Future] = {}
        self._repo_ready = False

    def submit(self, local: Path, name: str) -> Future | None:
        if not self.repo:
            return None
        fut = self.pool.submit(self._upload, Path(local), name)
        self.pending[Path(local)] = fut
        return fut

    def busy(self) -> set[Path]:
        return {p for p, f in self.pending.items() if not f.done()}

    def _upload(self, local: Path, name: str) -> bool:
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
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
                                       commit_message=f"{self.run_id}: checkpoint {name}")
                self.log.event("ckpt_upload_ok", name=name, attempt=attempt, gb=round(size / 1e9, 3),
                               upload_s=round(time.time() - t0, 1))
                return True
            except Exception as e:
                self.log.event("ckpt_upload_error", name=name, attempt=attempt, error=f"{type(e).__name__}: {e}"[:2000])
        self.log.event("ckpt_upload_failed", name=name, attempts=len(self.retries) + 1)
        return False

    def wait(self) -> dict[str, bool]:
        return {p.name: f.result() for p, f in list(self.pending.items())}

    def shutdown(self):
        self.pool.shutdown(wait=True)


def hf_roundtrip(R: Run) -> dict:
    """Upload a small file into the output repo and read it back: the credentials, the repo and both directions work
    before the paid hours start (a run whose results cannot leave the box is worthless)."""
    api, repo = R.uploader.api, out_repo(R.cfg)
    payload = json.dumps(dict(run_id=R.run_dir.name, nonce=uuid.uuid4().hex, wall=time.time())).encode()
    rel = f"runs/{R.run_dir.name}/smoke/roundtrip.json"
    (R.run_dir / "smoke").mkdir(exist_ok=True)
    (R.run_dir / "smoke" / "roundtrip.json").write_bytes(payload)
    t0 = time.time()
    api.create_repo(repo, repo_type="model", private=R.cfg["hf"]["private"], exist_ok=True)
    api.upload_file(path_or_fileobj=payload, path_in_repo=rel, repo_id=repo, repo_type="model",
                    commit_message=f"{R.run_dir.name}: smoke round trip")
    t1 = time.time()
    with tempfile.TemporaryDirectory() as d:
        got = Path(api.hf_hub_download(repo, rel, repo_type="model", local_dir=d)).read_bytes()
    out = dict(ok=got == payload, repo=repo, path=rel, upload_s=round(t1 - t0, 2), download_s=round(time.time() - t1, 2))
    R.log.event("smoke_hf_roundtrip", **out)
    if not out["ok"]:
        raise SmokeFailed(f"HF round trip to {repo} returned different bytes")
    return out


# ---------------------------------------------------------------------------------------------------------- model


def setup_model(R: Run, grad_ckpt: bool):
    """Student -> device in fp32, frozen pos_emb, rel-pos patch, BN frozen, optional checkpointing/compile."""
    from kitsune import student as S

    cfg = R.cfg
    sdir = rpath(cfg["student"])
    model = S.load_student(sdir, R.device, dtype=torch.float32)
    model.model.decoder.pos_emb.weight.requires_grad_(False)  # fixed sinusoids stored as an Embedding
    if cfg["perf"]["relpos_patch"]:
        patch_relpos_once_per_batch(model)
    train_mode(model)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        train_mode(model)
    if cfg["perf"]["compile"]:  # in place: state_dict keys (L2-SP names, checkpoints) stay unchanged
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


def setup_processing(R: Run):
    """Featurisers, SpecAugment, processor and tokenizer, from the student dir (save_student put the teacher's
    processor there); the defaults of the HF extractor / the teacher tokenizer if it has none."""
    from transformers import AutoProcessor

    sdir = rpath(R.cfg["student"])
    try:
        R.processor = AutoProcessor.from_pretrained(str(sdir))
        fe = R.processor.feature_extractor
        R.feat_eval = LogMel.from_feature_extractor(fe).to(R.device)
        R.feat_train = LogMel.from_feature_extractor(fe, exact_dither=R.cfg["perf"]["train_exact_dither"]).to(R.device)
        R.tokenizer = R.processor.tokenizer
    except Exception as e:
        from kitsune import evaluate as ev

        R.log.event("processor_fallback", error=f"{type(e).__name__}: {e}"[:500])
        R.processor = None
        R.feat_eval = LogMel().to(R.device)
        R.feat_train = LogMel(exact_dither=R.cfg["perf"]["train_exact_dither"]).to(R.device)
        R.tokenizer = ev.teacher_tokenizer()
    R.specaug = SpecAugment(**{k: v for k, v in R.cfg["specaug"].items() if k != "enabled"})
    R.gen = torch.Generator(device=R.device)


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
                 fused: bool = False):
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
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **impl)

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


def setup_optim(R: Run):
    o = R.cfg["optim"]
    if o["offload"] == "cpu":
        R.opt = CpuOffloadAdamW(R.params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                weight_decay=o["weight_decay"], fused=bool(o["offload_fused"]))
        R.l2sp = L2SP(zip(R.param_names, R.opt.master), lam=R.cfg["loss"]["l2sp_lambda"])  # theta_0 in host memory
        R.log.event("optim_offload", **R.opt.info())
        return
    fused = bool(o["fused"]) and R.device.type == "cuda"
    R.opt = torch.optim.AdamW(R.params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                              weight_decay=o["weight_decay"], fused=fused)
    R.l2sp = L2SP(zip(R.param_names, R.params), lam=R.cfg["loss"]["l2sp_lambda"])


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


def setup_data(R: Run):
    """Train/eval stores (cached under cache_dir, shared with 03's eval cache), probe and greedy ids, planner."""
    cfg, log = R.cfg, R.log
    sel, data, teach, cache = rpath(cfg["selection"]), rpath(cfg["data_root"]), rpath(cfg["teacher_root"]), rpath(cfg["cache_dir"])
    sub, seed = cfg["subset"], int(cfg["seed"])
    t0 = time.time()
    if sub["train_audio_s"] is not None:
        rows = trainset.read_selection(sel, cfg["sources"], ["train"])
        prompt_len = len(trainset._teacher_meta(teach)["prompt"])
        rows = rows[rows["n_tok"] + prompt_len - 1 <= int(cfg["batch"]["max_dec_len"])]  # what the planner can use
        ids, rec = audio_subset(rows["id"].tolist(), rows["duration"].to_numpy(), float(sub["train_audio_s"]),
                                np.random.default_rng([seed, 11]))
        log.event("subset", split="train", sources=cfg["sources"], seed=seed, pool=len(rows), **rec)
        R.train = trainset.build_stores(sel, data, teach, cache / _subset_dir("train", rec["budget_s"], seed, ids),
                                        cfg["sources"], ["train"], ids=ids, log=print)
    elif sub["train_utts"]:
        n = int(sub["train_utts"])
        rows = trainset.read_selection(sel, cfg["sources"], ["train"])
        rng = np.random.default_rng([seed, 1])
        probe = rows["id"][rows["in_probe"]].tolist()[: max(1, n // 4)]
        rest = _seeded_ids(rows["id"][~rows["id"].isin(probe)].tolist(), n - len(probe), rng)
        R.train = trainset.build_stores(sel, data, teach, cache / f"train_sub{n}_s{seed}", cfg["sources"], ["train"],
                                        ids=probe + rest, log=print)
    else:
        R.train = trainset.build_stores(sel, data, teach, cache / "train", cfg["sources"], ["train"], log=print)
    if sub["eval_audio_s"] is not None:  # pooled over the eval sets
        rows = trainset.read_selection(sel, cfg["eval_sets"], ["eval"])
        ids, rec = audio_subset(rows["id"].tolist(), rows["duration"].to_numpy(), float(sub["eval_audio_s"]),
                                np.random.default_rng([seed, 12]))
        log.event("subset", split="eval", sources=cfg["eval_sets"], seed=seed, pool=len(rows), **rec)
        R.evalstore = trainset.build_stores(sel, data, teach, cache / _subset_dir("eval", rec["budget_s"], seed, ids),
                                            cfg["eval_sets"], ["eval"], ids=ids, log=print)
    elif sub["eval_utts_per_set"]:
        n = int(sub["eval_utts_per_set"])
        rows = trainset.read_selection(sel, cfg["eval_sets"], ["eval"])
        rng = np.random.default_rng([seed, 2])
        ids = []
        for s in cfg["eval_sets"]:
            r = rows[rows["source"] == s]
            greedy = _seeded_ids(r["id"][r["in_greedy_subset"]].tolist(), min(n, int(cfg["eval"]["greedy_subset"])), rng)
            ids += greedy + _seeded_ids(r["id"][~r["id"].isin(greedy)].tolist(), max(0, n - len(greedy)), rng)
        R.evalstore = trainset.build_stores(sel, data, teach, cache / f"eval_sub{n}_s{seed}", cfg["eval_sets"],
                                            ["eval"], ids=ids, log=print)
    else:
        R.evalstore = trainset.eval_store(sel, data, teach, cache / "eval", cfg["eval_sets"], log=print)

    if cfg["eval"]["probe"] and cfg["eval"]["probe_is_train"]:
        R.probe_ids = [u.id for u in R.train.utts]
    else:
        R.probe_ids = [R.train.utts[i].id for i in R.train.indices(in_probe=True)] if cfg["eval"]["probe"] else []
    pg_budget = cfg["eval"]["probe_greedy_audio_s"]
    pg_budget = sub["eval_audio_s"] if pg_budget is None else pg_budget
    R.probe_greedy_ids = []
    if pg_budget is not None and R.probe_ids:  # greedy CER on the train data itself, next to the held-out one
        dur = {u.id: u.duration for u in R.train.utts}
        R.probe_greedy_ids, rec = audio_subset(R.probe_ids, [dur[i] for i in R.probe_ids], float(pg_budget),
                                               np.random.default_rng([seed, 13]))
        log.event("subset", split="probe_greedy", seed=seed, pool=len(R.probe_ids), **rec)
    if sub["eval_audio_s"] is not None:
        R.greedy_ids = [u.id for u in R.evalstore.utts]  # the whole (small) eval subset, at every eval
    else:
        rng = np.random.default_rng([seed, 3])
        R.greedy_ids = []
        for s in cfg["eval_sets"]:
            cand = R.evalstore.indices(source=s, in_greedy_subset=True) or R.evalstore.indices(source=s)
            pick = cand if len(cand) <= cfg["eval"]["greedy_subset"] else sorted(
                rng.choice(cand, size=int(cfg["eval"]["greedy_subset"]), replace=False).tolist())
            R.greedy_ids += [R.evalstore.utts[i].id for i in pick]
    R.ds = trainset.AudioBatchDataset(R.train)
    R.src_index = {s: i for i, s in enumerate(sorted({u.source for u in R.train.utts}))}
    log.event("data", train_utts=len(R.train), train_h=round(R.train.hours, 3),
              per_source=R.train.info.get("per_source"), dropped=R.train.info.get("dropped"),
              eval_utts=len(R.evalstore), eval_h=round(R.evalstore.hours, 3),
              eval_per_set=R.evalstore.info.get("per_source"), probe=len(R.probe_ids), greedy=len(R.greedy_ids),
              build_s=round(time.time() - t0, 1),
              **(dict(probe_greedy=len(R.probe_greedy_ids)) if R.probe_greedy_ids else {}))


def make_planner(R: Run, micro_audio_s: float) -> trainset.StepPlanner:
    b, mix = R.cfg["batch"], R.cfg["mix"]
    weights = None if mix in (None, "natural") else mix
    return trainset.StepPlanner(R.train.utts, step_audio_s=b["step_audio_s"], micro_audio_s=micro_audio_s,
                                max_dec_len=b["max_dec_len"], pool_micro=b["pool_micro"], seed=int(R.cfg["seed"]),
                                weights=weights, prompt_len=len(R.train.info.get("prompt", trainset.PROMPT)))


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


def forward_logits(R: Run, mb: dict, featurizer, gen=None, capture: dict | None = None):
    """Student logits (N, V) at the micro-batch's target positions: LogMel (fp32), SpecAugment if `gen`, the body
    under autocast, the head in fp32 outside it. Returns (logits, tgt_row, masked_frac or None)."""
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
            h = R.model.model(input_features=feats, attention_mask=fmask,
                              decoder_input_ids=mb["decoder_input_ids"].to(dev, non_blocking=True),
                              decoder_attention_mask=mb["dec_mask"].to(dev, non_blocking=True),
                              use_cache=False).last_hidden_state
    finally:
        for hk in hooks:
            hk.remove()
    rows = mb["tgt_row"].to(dev, non_blocking=True)
    head = R.model.proj_out
    with torch.autocast(device_type=dev.type, enabled=False):
        logits = F.linear(h[rows, mb["tgt_pos"].to(dev, non_blocking=True)].float(), head.weight.float(),
                          head.bias.float() if head.bias is not None else None)
    return logits, rows, mfrac


def _capture_hook(store: dict, name: str):
    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
        if isinstance(t, torch.Tensor):
            t = t.detach().flatten()
            store[name] = t[:: max(1, math.ceil(t.numel() / 1_000_000))].float().clone()
    return hook


def activation_modules(model) -> list[tuple[str, torch.nn.Module]]:
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


def train_step(R: Run, step: int, lr: float, mbs: list[dict], epoch: int) -> dict | None:
    """One optimizer step over its micro-batches. Returns the step's CPU statistics, or None if it was skipped
    (non-finite gradient norm or OOM; the planner position still moves on)."""
    cfg, dev = R.cfg, R.device
    lg = cfg["log"]
    dropped = [i for mb in mbs for i in mb["dropped"]]  # before the filter: a micro-batch can lose all its rows
    if dropped:
        R.log.event("dropped_audio", at_step=step, n=len(dropped), ids=dropped)
    mbs = [mb for mb in mbs if len(mb["ids"])]
    n_tok = sum(int(mb["top_idx"].shape[0]) for mb in mbs)
    if not mbs or n_tok == 0:
        R.log.event("empty_step", at_step=step)
        return None
    for g in R.opt.param_groups:
        g["lr"] = lr
    R.gen.manual_seed(int(cfg["seed"]) * 1_000_003 + step)
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
    logits = losses = oom = None
    gen = R.gen if cfg["specaug"]["enabled"] else None
    j = -1
    try:
        for j, mb in enumerate(mbs):
            logits, rows, mfrac = forward_logits(R, mb, R.feat_train, gen=gen, capture=capture if j == 0 else None)
            losses = kd_losses(logits, mb["top_idx"].to(dev, non_blocking=True), mb["top_lp"].to(dev, non_blocking=True))
            kd_objective(losses, n_tok, cfg["loss"]["w_kl"], cfg["loss"]["w_ce"]).backward()
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
        logits = losses = None
        per_utt.clear()
        meta.clear()
        R.opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        R.st["oom_skips"] += 1
        R.log.event("oom_step_skipped", at_step=step, n_micro=len(mbs), at_micro=j, audio_pad_s=round(audio_pad, 1),
                    count=R.st["oom_skips"], error=oom, ids=[mb["ids"] for mb in mbs],
                    padded_s=[round(float(mb["lengths"].max()) * len(mb["ids"]) / SR, 1) for mb in mbs],
                    dec_len=[int(mb["dec_mask"].shape[1]) for mb in mbs])
        if R.st["oom_skips"] > int(cfg["memory"]["max_oom_skips"]):
            raise torch.OutOfMemoryError(f"{R.st['oom_skips']} steps skipped for OOM (limit "
                                         f"{cfg['memory']['max_oom_skips']}); last: {oom}")
        return None

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
        R.st["nonfinite_skips"] += 1
        R.st["nonfinite_total"] += 1
        # everything needed to find the culprit afterwards: the step's ids per micro-batch, each micro-batch's loss
        # sums, and the utterances whose own loss is non-finite (empty if only the backward overflowed)
        pus = [pu.cpu() for pu in per_utt]
        bad = [mb["ids"][b] for (mb, _), pu in zip(meta, pus) for b in range(len(mb["ids"]))
               if not bool(torch.isfinite(pu[b, :2]).all())]
        R.log.event("nonfinite_grad_skipped", at_step=step, grad_norm=gnorm, consecutive=R.st["nonfinite_skips"],
                    nonfinite_ids=bad, ids=[mb["ids"] for mb in mbs],
                    mb_n_tok=[int(mb["top_idx"].shape[0]) for mb in mbs],
                    mb_kl_sum=[float(pu[:, 0].sum()) for pu in pus], mb_ce_sum=[float(pu[:, 1].sum()) for pu in pus])
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

    # one host transfer for the step's statistics
    flat = torch.cat([tot, by_src.flatten(), n_src, by_bucket.flatten(), n_bucket]).cpu().numpy()
    k = 0

    def take(n):
        nonlocal k
        v = flat[k:k + n]
        k += n
        return v

    out.update(tot=take(nt), by_src=take(S * nt).reshape(S, nt), n_src=take(S), by_bucket=take(len(BUCKETS) * nt)
               .reshape(len(BUCKETS), nt), n_bucket=take(len(BUCKETS)), n_tok=n_tok, grad_norm=gnorm,
               audio_real=audio_real, audio_pad=audio_pad, dec_real=dec_real, dec_pad=dec_pad, n_micro=len(mbs),
               clip_coef=min(1.0, clip / (gnorm + 1e-6)) if clip else 1.0)
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
    """BatchNorm running stats must be exactly the init's: BN is frozen for the whole run."""
    n = assert_bn_frozen(R.model)
    drift = 0.0
    for name, m in R.model.named_modules():
        if name in R.bn0:
            m0, v0 = R.bn0[name]
            drift = max(drift, float((m.running_mean - m0).abs().max()), float((m.running_var - v0).abs().max()))
    R.log.scalars({"bn/max_abs_drift": drift, "bn/modules": n}, step)
    if drift != 0.0:
        raise AssertionError(f"BatchNorm running stats moved by {drift} although BN is frozen")


# ------------------------------------------------------------------------------------------------------- logging


def log_step(R: Run, step: int, lr: float, phase: int, out: dict, wait_s: float, step_s: float, e: int, s: int):
    from kitsune.runlog import system_stats

    cfg = R.cfg
    n_tok = out["n_tok"]
    tot = out["tot"]
    w_kl, w_ce = cfg["loss"]["w_kl"], cfg["loss"]["w_ce"]
    kl, ce = tot[0] / n_tok, tot[1] / n_tok
    l2sp = out["l2sp"] if "l2sp" in out else float(R.l2sp.value())
    objective = w_kl * kl + w_ce * ce
    t, T = R.progress()
    n_steps = R.planner.epoch_stats[e]["steps"] if e in R.planner.epoch_stats else 1
    R.st["epoch"], R.st["epoch_progress"] = e, e + (s + 1) / n_steps
    R.st["audio_s"] += out["audio_real"]
    R.st["tokens"] += n_tok
    R.st["step_time_s"] += step_s
    R.st["last_objective"] = float(objective)
    flops = None
    if R.st["flops_per_padded_s"]:
        mult = 4.0 if R.st["memory"].get("grad_ckpt") else 3.0
        flops = mult * R.st["flops_per_padded_s"] * out["audio_pad"]
    row = {
        "loss/total": objective + l2sp, "loss/objective": objective, "loss/kl": kl, "loss/ce": ce, "loss/l2sp": l2sp,
        "opt/lr": lr, "opt/grad_norm": out["grad_norm"], "opt/clip_coef": out["clip_coef"],
        "time/step_s": step_s, "time/data_wait_s": wait_s, "time/compute_s": step_s - wait_s,
        "perf/audio_s_per_s": out["audio_real"] / step_s, "perf/tokens_per_s": n_tok / step_s,
        "perf/pad_eff_audio": out["audio_real"] / max(out["audio_pad"], 1e-9),
        "perf/pad_eff_dec": out["dec_real"] / max(out["dec_pad"], 1e-9),
        "aug/masked_frac": out["masked_frac"],
        "data/epoch": e, "data/epoch_progress": R.st["epoch_progress"], "data/utts": out["n_utts"],
        "data/audio_s": out["audio_real"], "data/tokens": n_tok, "data/micro_batches": out["n_micro"],
        "data/dropped": out["dropped"], "sched/phase": phase, "sched/progress": t / T if T else float("nan"),
        "sched/train_s": R.clock(),
    }
    if flops is not None:
        row["perf/tflops"] = flops / step_s / 1e12
        row["perf/mfu"] = flops / step_s / (float(cfg["perf"]["peak_tflops"]) * 1e12)
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
    row.update(system_stats())
    R.log.step_row(row, step)
    R.log.train_utts(out["utts"])
    if "grad_sq_by_module" in out:
        R.log.scalars({f"layers/grad_norm/{k}": math.sqrt(v) for k, v in out["grad_sq_by_module"].items()}, step)
        R.log.scalars({f"layers/update_ratio/{k}": v for k, v in out["update_ratio"].items()}, step)
        R.log.scalars({f"l2sp/dist/{k}": v for k, v in R.l2sp.per_module_distance().items()}, step)
        R.log.scalars({f"l2sp/rel/{k}": v for k, v in R.l2sp.per_module_distance(relative=True).items()}, step)
    return objective


# ---------------------------------------------------------------------------------------------------------- evals


def run_eval(R: Run, step: int, final: bool = False) -> dict:
    """Teacher-forced on every eval utterance and on the train probe, greedy on the fixed subsets (final: on the FULL
    eval sets, the subset summary taken from those rows). Tables, summary, scalars and samples go to the logger; one
    record goes to the history the verdict reads. Returns the greedy summary the verdict should judge: the full-set
    one on the final eval (the subset one if eval.final_full_greedy is off)."""
    from kitsune import evaluate as ev

    cfg, log = R.cfg, R.log
    bs = float(cfg["eval"]["batch_s"])
    t0 = time.time()
    log.event("eval_start", at_step=step, final=final)
    tf_sum, tf_df = ev.teacher_forced_eval(R.model, R.evalstore, R.feat_eval, R.device, bs, amp=R.amp)
    probe_sum = probe_df = None
    if R.probe_ids:
        probe_sum, probe_df = ev.teacher_forced_eval(R.model, R.train, R.feat_eval, R.device, bs, ids=R.probe_ids,
                                                     amp=R.amp)
    pg_sum = pg_df = None
    if R.probe_greedy_ids:  # un-augmented greedy decode of train utterances: memorisation shows as CER vs teacher -> 0
        pg_sum, pg_df = ev.greedy_eval(R.model, R.train, R.probe_greedy_ids, R.feat_eval, R.device, bs,
                                       tokenizer=R.tokenizer, amp=R.amp)
    full_sum = None
    if final and cfg["eval"]["final_full_greedy"]:
        full_sum, gr_df = ev.greedy_eval(R.model, R.evalstore, None, R.feat_eval, R.device, bs, tokenizer=R.tokenizer,
                                         amp=R.amp)
        gr_df["in_greedy_subset"] = gr_df["id"].isin(set(R.greedy_ids))
        gr_sum = ev.summarise_greedy(gr_df[gr_df["in_greedy_subset"]], wall_s=full_sum["wall_s"])
    else:
        gr_sum, gr_df = ev.greedy_eval(R.model, R.evalstore, R.greedy_ids, R.feat_eval, R.device, bs,
                                       tokenizer=R.tokenizer, amp=R.amp)
    assert_bn_frozen(R.model)

    for src, g in tf_df.groupby("source", sort=True):
        log.table(f"tf_{src}", g.reset_index(drop=True), step)
    if probe_df is not None:
        log.table("probe", probe_df, step)
    if pg_df is not None:
        log.table("probe_greedy", pg_df, step)
    for src, g in gr_df.groupby("source", sort=True):
        log.table(f"greedy_{src}", g.reset_index(drop=True), step)
    summary = dict(step=step, train_s=R.clock(), final=final, tf=tf_sum, probe=probe_sum, greedy=gr_sum,
                   greedy_full=full_sum, wall_s=round(time.time() - t0, 1))
    extra = {}  # only in the runs that use them, so the viability run's records are unchanged
    if pg_sum is not None:
        extra["probe_greedy"] = pg_sum
    if epoch_mode(cfg):
        extra["epoch"] = R.st["epoch_progress"]  # epochs done: 0 at step 0, e + 1 at the end of epoch e
    summary.update(extra)
    log.eval_json("summary", summary, step)
    scal = ev.flatten(tf_sum, "eval/tf")
    scal.update(ev.flatten(gr_sum, "eval/greedy"))
    if probe_sum:
        scal.update(ev.flatten(probe_sum, "eval/probe"))
        if "all" in probe_sum and "all" in tf_sum:
            scal["eval/kl_gap_heldout_minus_probe"] = tf_sum["all"]["kl"] - probe_sum["all"]["kl"]
    if pg_sum:
        scal.update(ev.flatten(pg_sum, "eval/probe_greedy"))
        if "all" in pg_sum and "all" in gr_sum:
            scal["eval/cer_teacher_gap_heldout_minus_probe"] = (gr_sum["all"]["cer_teacher_corpus"]
                                                                 - pg_sum["all"]["cer_teacher_corpus"])
    if full_sum:
        scal.update(ev.flatten(full_sum, "eval/greedy_full"))
    if "epoch" in extra:
        scal["eval/epoch"] = extra["epoch"]
    scal["eval/wall_s"] = summary["wall_s"]
    log.scalars(scal, step)
    samples = ev.pick_samples(gr_df, int(cfg["log"]["samples_per_eval"]), seed=int(cfg["seed"]))
    if pg_df is not None:  # train utterances (their source says so) after the held-out ones
        samples += ev.pick_samples(pg_df, int(cfg["log"]["samples_per_eval"]), seed=int(cfg["seed"]))
    log.samples(step, samples)

    if not final:  # what fit_budget scales the final eval's duration from
        R.st["eval_cost"] = dict(tf_s=float(tf_sum.get("wall_s", 0.0)),
                                 probe_s=(float(probe_sum.get("wall_s", 0.0)) if probe_sum else 0.0)
                                 + (float(pg_sum.get("wall_s", 0.0)) if pg_sum else 0.0),
                                 greedy_s=float(gr_sum.get("wall_s", 0.0)), greedy_n=len(R.greedy_ids))
    rec = ev.eval_record(step, R.clock(), tf=tf_sum, greedy=gr_sum, probe=probe_sum)
    if pg_sum and "all" in pg_sum:
        rec["probe_greedy"] = {k: pg_sum["all"][k] for k in ("cer_teacher_corpus", "cer_ref_corpus", "n")}
    if "epoch" in extra:
        rec["epoch"] = extra["epoch"]
    hist = R.st["history"]
    if hist and hist[-1]["step"] == step:
        hist[-1] = rec
    else:
        hist.append(rec)
    brief = {s: dict(kl=round(d["kl"], 4), top1=round(d["top1"], 4)) for s, d in tf_sum.get("sets", {}).items()}
    for s, d in gr_sum.get("sets", {}).items():
        brief.setdefault(s, {}).update(cer=round(d["cer_ref_corpus"], 4), ratio=round(d["ratio_vs_teacher"], 3),
                                       trunc=round(d["trunc_rate"], 4))
    if pg_sum and "all" in pg_sum:
        extra["probe_cer_teacher"] = round(pg_sum["all"]["cer_teacher_corpus"], 4)
    log.event("eval", at_step=step, final=final, wall_s=summary["wall_s"], sets=brief,
              probe_kl=probe_sum["all"]["kl"] if probe_sum and "all" in probe_sum else None,
              **{k: v for k, v in extra.items() if k != "probe_greedy"})
    return full_sum if full_sum is not None else gr_sum


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


def save_weights(R: Run, step: int, reason: str) -> Path:
    """bf16 weights (+ processor, student_meta.json with a `trained` block) -> checkpoints/step_<N>/, then upload."""
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
    meta["trained"] = dict(run_id=R.run_dir.name, step=step, train_s=round(R.clock(), 1), epoch=R.st["epoch_progress"],
                           reason=reason, time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           init=str(R.cfg["student"]), last_objective=R.st["last_objective"])
    S.save_student(R.model, tmp, R.processor, meta)
    _replace_dir(tmp, d)
    R.st["weights"].append(step)
    R.log.event("checkpoint", ckpt="weights", name=name, reason=reason, save_s=round(time.time() - t0, 1),
                gb=round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 3),
                disk_free_gb=_disk_free_gb(d))
    R.uploader.submit(d, name)
    return d


def save_full(R: Run, step: int, reason: str, upload: bool = False) -> Path:
    """Everything --resume needs -> checkpoints/full_step_<N>/ (atomic), then keep the newest ckpt.keep_local."""
    name = f"full_step_{step}"
    d = R.ckpt_dir / name
    t0 = time.time()
    if not (d.exists() and step in R.st["fulls"]):
        tmp = R.ckpt_dir / f"{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        st = copy.deepcopy(R.st)
        st["train_s"] = R.clock()
        st["fulls"] = sorted(set(st["fulls"]) | {step})
        trainer = dict(format=1, step=step, reason=reason, run_id=R.run_dir.name, cfg=R.cfg, st=st,
                       planner=R.planner.state_dict(), logger=R.log.state_dict(), rng=_rng_state(),
                       time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        torch.save(R.model.state_dict(), tmp / "model.pt")
        torch.save(R.opt.state_dict(), tmp / "optimizer.pt")
        torch.save(R.l2sp.state_dict(), tmp / "l2sp.pt")
        torch.save(trainer, tmp / "trainer.pt")
        brief = {k: v for k, v in trainer.items() if k not in ("rng", "st")}
        brief["st"] = {k: v for k, v in st.items() if k not in ("history", "smoke_losses")}
        (tmp / "trainer.json").write_text(json.dumps(brief, indent=1, default=str), encoding="utf-8")
        _replace_dir(tmp, d)
        R.st["fulls"] = st["fulls"]
        R.log.event("checkpoint", ckpt="full", name=name, reason=reason, save_s=round(time.time() - t0, 1),
                    gb=round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 3),
                    disk_free_gb=_disk_free_gb(d))
    if upload:
        R.uploader.submit(d, name)
    rotate_full(R)
    return d


def _disk_free_gb(p: Path) -> float | None:
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except OSError:
        return None


def rotate_full(R: Run):
    """Keep the newest ckpt.keep_local full states (~8.6 GB each for the real student). The one a resume started
    from is not special: once keep_local newer ones exist it goes too, or a resumed run fills the 80 GB disk."""
    keep = max(1, int(R.cfg["ckpt"]["keep_local"]))
    fulls = sorted((p for p in R.ckpt_dir.iterdir() if FULL_RE.match(p.name)), key=lambda p: int(FULL_RE.match(p.name)[1]))
    busy = R.uploader.busy()
    for p in fulls[:-keep]:
        if p in busy:
            continue
        shutil.rmtree(p, ignore_errors=True)
        R.log.event("checkpoint_deleted", name=p.name, keep_local=keep)


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


# ----------------------------------------------------------------------------------------------------------- smoke


def measure_flops(R: Run, mb: dict) -> float:
    """Forward FLOPs of one micro-batch (torch's FlopCounterMode) per padded audio second, for the MFU estimate.
    Encoder cost scales with padded audio; the decoder and head (~6 % for the real student) are folded in."""
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), frozen_eval(R.model), FlopCounterMode(display=False) as fc:
        forward_logits(R, mb, R.feat_eval)
    padded = float(mb["lengths"].max()) * len(mb["lengths"]) / SR
    return fc.get_total_flops() / max(padded, 1e-9)


def fwd_bwd(R: Run, mb: dict) -> tuple[float, bool]:
    logits, _, _ = forward_logits(R, mb, R.feat_train)
    losses = kd_losses(logits, mb["top_idx"].to(R.device), mb["top_lp"].to(R.device))
    loss = kd_objective(losses, int(mb["top_idx"].shape[0]), R.cfg["loss"]["w_kl"], R.cfg["loss"]["w_ce"])
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
    measured on the same utterances (alone under autocast vs alone without it)."""
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


def smoke_checks(R: Run):
    """Checks that cost seconds and catch the failures that would otherwise burn the paid hours silently."""
    from kitsune.features import hf_reference
    from kitsune.patches import sdpa_backend_report

    log = R.log
    t0 = time.time()
    # LogMel vs the HF extractor on a few real utterances (bitwise on CPU; FFT/matmul rounding on CUDA)
    idx = list(range(min(4, len(R.train))))
    waves = [R.train.wave(i) for i in idx]
    try:
        ref, ref_mask = hf_reference(waves, path_or_repo=str(rpath(R.cfg["student"])))
        wave = torch.zeros(len(waves), max(len(w) for w in waves))
        for i, w in enumerate(waves):
            wave[i, :len(w)] = torch.from_numpy(w)
        got, mask = R.feat_eval(wave.to(R.device), torch.tensor([len(w) for w in waves], device=R.device))
        diff = float((got.cpu() - ref).abs().max()) if got.shape == ref.shape else float("inf")
        log.event("smoke_logmel_vs_hf", max_abs_diff=diff, masks_equal=bool(torch.equal(mask.cpu(), ref_mask.bool())),
                  n=len(waves), device=str(R.device))
    except Exception as e:  # no processor files in the student dir: informational only
        log.event("smoke_logmel_vs_hf", skipped=f"{type(e).__name__}: {e}"[:300])

    worst = R.planner.worst_micro_batches()
    mb = R.ds[worst["longest"]]
    loss, ok = fwd_bwd(R, mb)
    log.event("smoke_longest_fwd_bwd", ok=ok, loss=loss, utts=len(mb["ids"]),
              padded_s=round(float(mb["lengths"].max()) * len(mb["ids"]) / SR, 1), targets=int(mb["top_idx"].shape[0]))
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


def cap_vram(R: Run):
    """Windows + CUDA: cap PyTorch's caching allocator, for the whole process, at the VRAM free now (plus what it
    already holds) minus VRAM_MARGIN_GIB. The Windows driver's sysmem fallback (on by default) serves an allocation that
    no longer fits the card from shared system memory instead of failing it, so an oversized micro-batch would run
    several times slower instead of raising OutOfMemoryError, and the memory probe's fallbacks (halve micro_audio_s,
    gradient checkpointing) and train_step's OOM skip would never fire. Under the cap they do. Linux has no fallback."""
    if R.device.type != "cuda" or os.name != "nt":
        return
    idx = R.device.index if R.device.index is not None else torch.cuda.current_device()  # the API needs an index
    free, total = torch.cuda.mem_get_info(idx)
    held = torch.cuda.memory_reserved(idx)
    frac = min(1.0, max(0.0, (free + held - VRAM_MARGIN_GIB * 2**30) / total))
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    R.vram_cap_gb = round(frac * total / 2**30, 2)
    R.log.event("vram_cap", cap_gb=R.vram_cap_gb, free_gb=round(free / 2**30, 2), held_gb=round(held / 2**30, 2),
                total_gb=round(total / 2**30, 2), margin_gb=VRAM_MARGIN_GIB, fraction=round(frac, 4))


def probe_passes(R: Run, planner: trainset.StepPlanner, rec: dict):
    """The memory probe's measured passes: forward+backward on each of the plan's worst micro-batches, peak GiB
    allocated / reserved into rec (CUDA). train_step accumulates a step's micro-batches into the same gradients, so
    from the second micro-batch on, forward and backward run with the full fp32 gradients (4 B/param, 2.3 GiB for the
    real student) already allocated. When the plan has steps of more than one micro-batch every pass therefore starts
    from zero-filled gradients (backward adds into them in place, as in training) instead of none."""
    cuda = R.device.type == "cuda"
    plan = planner.epoch_plan(0)
    rec["grads_held"] = held = max(len(step) for step in plan) > 1
    for name, idx in planner.worst_micro_batches(plan).items():
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


def memory_probe(R: Run) -> dict:
    """CUDA only. Run forward+backward on the micro-batches that stress memory most (longest audio, most decoder
    positions, most targets) with the optimizer state's bytes held in reserve, and the step's gradients too when a step
    has several micro-batches (probe_passes). On OOM halve micro_audio_s (down to memory.min_micro_audio_s); if that
    still fails, enable per-layer gradient checkpointing and start again from the configured size. The choice is kept in
    the full state, so a resume uses the same batches. On Windows it runs under cap_vram's cap, so OOM means OOM."""
    cfg, log = R.cfg, R.log
    micro0 = float(cfg["batch"]["micro_audio_s"])
    ckpt = cfg["memory"]["grad_ckpt"] is True
    if R.device.type != "cuda" or not cfg["memory"]["probe_longest_bucket"]:
        log.event("memory_probe", skipped=f"device {R.device.type}" if R.device.type != "cuda" else "disabled",
                  micro_audio_s=micro0, grad_ckpt=ckpt)
        return dict(micro_audio_s=micro0, grad_ckpt=ckpt)
    if ckpt:
        R.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        train_mode(R.model)
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
            log.event("memory_probe", ok=True, micro_audio_s=micro, grad_ckpt=ckpt, peak_gb=rec["peak_gb"],
                      grads_held=rec["grads_held"], max_reserved_gb=rec.get("max_reserved_gb"),
                      vram_cap_gb=R.vram_cap_gb, device_gb=round(total, 1),
                      reserved_adam_gb=round(adam_bytes / 2**30, 2))
            return dict(micro_audio_s=micro, grad_ckpt=ckpt)
        # outside the except: its traceback no longer pins the failed pass's activations
        R.opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        log.event("oom_fallback", micro_audio_s=micro, grad_ckpt=ckpt, peaks_so_far=rec["peak_gb"], error=err)
        if micro / 2 >= float(cfg["memory"]["min_micro_audio_s"]):
            micro /= 2
        elif not ckpt and cfg["memory"]["grad_ckpt"] == "auto":
            ckpt, micro = True, micro0
            R.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            train_mode(R.model)
        else:
            raise torch.OutOfMemoryError(f"OOM even at micro_audio_s={micro} with grad_ckpt={ckpt}: {err}")


def smoke_end(R: Run):
    """After the first smoke.steps steps: finite, falling loss and enough throughput (else exit 3)."""
    cfg = R.cfg["smoke"]
    losses = R.st["smoke_losses"]
    q = max(1, len(losses) // 4)
    first, last = float(np.mean(losses[:q])), float(np.mean(losses[-q:]))
    rate = R.st["smoke_audio_s"] / max(R.st["smoke_time_s"], 1e-9)
    ok_loss = all(math.isfinite(x) for x in losses) and last < first
    R.log.event("smoke_steps", steps=len(losses), loss_first=first, loss_last=last, loss_decreasing=ok_loss,
                audio_s_per_s=round(rate, 1), floor=cfg["min_audio_s_per_s"])
    R.st["smoke_done"] = True
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


def build(args) -> tuple[Run, dict | None]:
    """Config, run dir and logger; the rest of the setup happens in train() so every failure is logged."""
    from kitsune.runlog import RunLogger

    state = None
    if args.resume:
        full = find_full_state(Path(args.resume))
        state = torch.load(full / "trainer.pt", map_location="cpu", weights_only=True)
        cfg = copy.deepcopy(state["cfg"])
        overrides = [apply_set(cfg, s) for s in args.set]
        validate(cfg)
        run_dir = full.parent.parent
    else:
        cfg = load_config(args.config, args.set)
        overrides = []
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
    api = hf_api() if out_repo(cfg) else None
    lg = cfg["log"]
    from kitsune import student as S

    R.log = RunLogger(run_dir, cfg, out_repo(cfg), lg["sync_every_min"], resume=state["logger"] if state else None,
                      student_meta=S.load_meta(rpath(cfg["student"])), capture=lg["capture_env"],
                      train_utts_flush_steps=lg["train_utts_flush"], api=api)
    R.uploader = Uploader(api, out_repo(cfg), run_dir.name, bool(cfg["hf"]["private"]), R.log)
    if state is not None:
        R.resumed_from = full
        R.log.event("resume", from_state=str(full), at_step=state["step"], overrides=dict(overrides),
                    config_arg_ignored=args.config)
    return R, state


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
        R.st["resumes"] += 1
    grad_ckpt = bool(R.st["memory"].get("grad_ckpt", cfg["memory"]["grad_ckpt"] is True))
    setup_model(R, grad_ckpt)
    setup_processing(R)
    setup_optim(R)  # L2-SP theta_0 = the student init (the full state's copy replaces it on resume)
    setup_data(R)
    R.planner = make_planner(R, float(R.st["memory"].get("micro_audio_s", cfg["batch"]["micro_audio_s"])))
    try:
        base = ev.teacher_baselines(rpath(cfg["teacher_root"]), cfg["eval_sets"], check=cfg["eval"]["check_baselines"])
        log.event("teacher_baselines", sets=base)
    except FileNotFoundError as e:
        log.event("teacher_baselines", skipped=str(e))
    from kitsune import student as S

    log.event("model", params=S.param_report(R.model), trainable=sum(p.numel() for p in R.params),
              grad_ckpt=grad_ckpt, relpos_patch=cfg["perf"]["relpos_patch"], compile=cfg["perf"]["compile"])

    if state is not None:
        full = R.resumed_from
        R.model.load_state_dict(torch.load(full / "model.pt", map_location=R.device, weights_only=True))
        # offload: the AdamW state stays in host memory, and the masters are rebuilt from the weights just loaded
        R.opt.load_state_dict(torch.load(full / "optimizer.pt", map_location="cpu" if offloaded(R) else R.device,
                                         weights_only=True))
        R.l2sp.load_state_dict(torch.load(full / "l2sp.pt", map_location="cpu", weights_only=True))
        R.planner.load_state_dict(state["planner"])
        _set_rng_state(state["rng"])
        plan_epochs(R)  # a no-op unless the clock was switched to "epochs" on this resume
        log.event("resumed", at_step=R.st["step"], train_s=round(R.st["train_s"], 1), epoch=R.st["epoch_progress"],
                  planner=state["planner"])
    else:
        if cfg["smoke"]["enabled"]:
            log.event("phase", name="smoke")
            R.st["memory"] = memory_probe(R)  # first: the checks below then run at the micro-batch size it chose
            smoke_checks(R)
        else:
            R.st["memory"] = dict(micro_audio_s=float(cfg["batch"]["micro_audio_s"]),
                                  grad_ckpt=cfg["memory"]["grad_ckpt"] is True)
            if R.st["memory"]["grad_ckpt"]:
                R.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                train_mode(R.model)
            R.st["flops_per_padded_s"] = measure_flops(R, R.ds[R.planner.worst_micro_batches()["longest"]])
        plan_epochs(R)  # after the memory probe: the micro-batch size it chose shapes the plans
        R.planner.epoch_plan(0)
        log.event("plan", micro_audio_s=R.planner.micro_audio_s, **R.planner.stats)
    if not R.st["step0_done"]:
        log.event("phase", name="step0_eval")
        run_eval(R, 0)
        R.st["step0_done"] = True

    loop(R)

    step = R.st["step"]
    log.event("phase", name="end", at_step=step, train_s=round(R.clock(), 1))
    save_weights(R, step, "end")
    save_full(R, step, "end", upload="end" in cfg["ckpt"]["upload_full_at"])
    full_sum = run_eval(R, step, final=True)
    verdict = ev.verdict(dict(final=full_sum, history=R.st["history"]))
    log.eval_json("verdict", verdict, step)
    log.event("verdict", **verdict)
    uploads = R.uploader.wait()
    summary = make_summary(R, "complete", verdict=verdict, final=full_sum, uploads=uploads)
    R.uploader.shutdown()
    log.close(summary=summary)
    return EXIT_OK


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


def loop(R: Run):
    cfg, log = R.cfg, R.log
    sch, ck, ev_cfg = cfg["schedule"], cfg["ckpt"], cfg["eval"]
    nw = cfg["perf"]["num_workers"]
    nw = trainset.default_num_workers() if nw == "auto" else int(nw)
    nw, prefetch, cap = shm_cap(nw, int(cfg["perf"]["prefetch"]), float(R.planner.micro_audio_s))
    if cap:
        log.event("shm_cap", **cap)
    crash_at = int(os.environ["KITSUNE_CRASH_AT_STEP"]) if os.environ.get("KITSUNE_CRASH_AT_STEP") else None
    smoke_n = int(cfg["smoke"]["steps"]) if cfg["smoke"]["enabled"] else 0
    epochs = int(sch["epochs"]) if sch["clock"] == "epochs" else None

    def passes_done() -> bool:  # clock "epochs": the planner's position is past the last epoch (skipped steps too)
        return epochs is not None and R.planner.epoch >= epochs

    fit_budget(R)
    log.event("phase", name="train", at_step=R.st["step"], workers=nw, micro_audio_s=R.planner.micro_audio_s,
              grad_ckpt=R.st["memory"].get("grad_ckpt"))
    loader = trainset.make_loader(R.ds, R.planner, nw, prefetch)
    R.loop_t0 = time.monotonic()
    epoch_seen = R.st["epoch"] if R.st["step"] else -1
    try:
        while True:
            t, T = R.progress()
            if t >= T or (sch["max_steps"] and R.st["step"] >= int(sch["max_steps"])) or passes_done():
                break
            t_c = (1.0 - float(sch["cooldown_frac"])) * T
            if not R.st["pre_cooldown_done"] and t >= t_c and float(sch["cooldown_frac"]) > 0:
                R.st["pre_cooldown_done"] = True
                log.event("phase", name="cooldown", at_step=R.st["step"], t=t, T=T)
                save_full(R, R.st["step"], "pre_cooldown", upload="pre_cooldown" in ck["upload_full_at"])
            step = R.st["step"] + 1
            if crash_at is not None and step >= crash_at:
                raise RuntimeError(f"KITSUNE_CRASH_AT_STEP={crash_at}: simulated crash before step {step}")
            lr, phase = wsd_lr(float(cfg["optim"]["lr"]), step, t, T, warmup_steps(sch, R.st["total_steps"]),
                               float(sch["cooldown_frac"]))
            if phase != R.st.get("lr_phase"):
                log.event("lr_phase", at_step=step, phase=("warmup", "stable", "cooldown")[phase], lr=lr, t=t, T=T)
                R.st["lr_phase"] = phase
            t0 = time.perf_counter()
            if R.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            (e, s), mbs = next(loader)
            wait = time.perf_counter() - t0
            if e != epoch_seen:
                epoch_seen = e
                log.event("epoch", step_in_epoch=s, **(R.planner.epoch_stats.get(e) or dict(epoch=e)))
            out = train_step(R, step, lr, mbs, e)
            R.planner.step_done(e, s)
            if out is None:
                continue
            if R.device.type == "cuda":
                torch.cuda.synchronize()
            step_s = time.perf_counter() - t0
            R.st["step"] = step
            objective = log_step(R, step, lr, phase, out, wait, step_s, e, s)
            if not R.st["smoke_done"] and smoke_n:
                R.st["smoke_losses"].append(float(objective))
                if step > max(1, smoke_n // 10):  # the first steps pay for worker start-up and kernel autotuning
                    R.st["smoke_audio_s"] += out["audio_real"]
                    R.st["smoke_time_s"] += step_s
                if step >= smoke_n:
                    smoke_end(R)
            t = R.clock()
            if ev_cfg["every_epochs"]:  # epochs completed since the last eval; the end phase's eval covers the last
                eval_now = (R.planner.epoch - R.st["last_eval_epoch"] >= int(ev_cfg["every_epochs"])
                            and not passes_done())
            else:
                eval_now = due(t, step, R.st["last_eval_t"], R.st["last_eval_step"], ev_cfg["every_min"],
                               ev_cfg["every_steps"])
            if eval_now:
                run_eval(R, step)
                R.st["last_eval_t"], R.st["last_eval_step"] = R.clock(), step
                R.st["last_eval_epoch"] = R.planner.epoch
                if not R.st["pre_cooldown_done"]:  # a fresher final-eval estimate (never moves T in the cooldown)
                    fit_budget(R)
            if due(t, step, R.st["last_weights_t"], R.st["last_weights_step"], ck["weights_every_min"],
                   ck["weights_every_steps"]):
                save_weights(R, step, "periodic")
                R.st["last_weights_t"], R.st["last_weights_step"] = R.clock(), step
            if due(t, step, R.st["last_full_t"], R.st["last_full_step"], ck["full_local_every_min"],
                   ck["full_every_steps"]):
                R.st["last_full_t"], R.st["last_full_step"] = R.clock(), step
                save_full(R, step, "periodic")
            log.sync()
    finally:
        R.st["train_s"] = R.clock()
        R.loop_t0 = None
        loader.close()


def make_summary(R: Run, status: str, **extra) -> dict:
    st, hist = R.st, R.st["history"]
    elapsed = R.log.elapsed() if R.log else time.time() - R.t_start
    dph = float(os.environ["KITSUNE_DPH"]) if os.environ.get("KITSUNE_DPH") else None

    def best(fn):
        vals = [(fn(r), r["step"]) for r in hist]
        vals = [v for v in vals if v[0] is not None and not math.isnan(v[0])]
        return dict(value=min(vals)[0], step=min(vals)[1]) if vals else None

    def cer_ratio(r):
        g = r.get("greedy") or {}
        v = [d["cer_ref_corpus"] / d["teacher_cer_ref_corpus"] for d in g.values() if d.get("teacher_cer_ref_corpus")]
        return float(np.mean(v)) if v else None

    return dict(
        status=status, run_id=R.run_dir.name, steps=st["step"], epochs=st["epoch_progress"],
        train_s=round(st["train_s"], 1), elapsed_s_total=round(elapsed, 1), resumes=st["resumes"],
        budget_s=R.budget_s,  # None = the full train_hours; else T clipped to the instance deadline
        throughput=dict(audio_s=st["audio_s"], tokens=st["tokens"], step_time_s=round(st["step_time_s"], 1),
                        audio_s_per_s=st["audio_s"] / st["step_time_s"] if st["step_time_s"] else None,
                        tokens_per_s=st["tokens"] / st["step_time_s"] if st["step_time_s"] else None),
        memory=st["memory"], skipped=dict(nonfinite=st["nonfinite_total"], oom=st["oom_skips"]),
        cost=dict(dph=dph, usd=round(dph * elapsed / 3600, 2) if dph else None, note="trainer process time only"),
        best=dict(greedy_cer_ratio_mean=best(cer_ratio), heldout_kl=best(lambda r: r.get("heldout_kl"))),
        history=hist, checkpoints=dict(weights=st["weights"], full=st["fulls"]),
        config=R.cfg, **extra)


def main(argv=None) -> int:
    args = parse_args(argv)
    R, state = build(args)
    try:
        return train(R, state)
    except ThroughputTooLow as e:
        R.log.event("throughput_too_low", error=str(e))
        _close_failed(R, "throughput_too_low", e)
        return EXIT_THROUGHPUT
    except BaseException as e:
        R.log.exception(e)
        _close_failed(R, "failed", e)
        raise


def _close_failed(R: Run, status: str, exc: BaseException):
    """Partial summary + forced sync (RunLogger.close). Never masks the original exception."""
    try:
        R.uploader.shutdown()
        R.log.close(summary=make_summary(R, status, error=f"{type(exc).__name__}: {exc}"[:2000]))
    except Exception as e:  # noqa: BLE001
        print(f"closing the logger after a failure failed too: {e!r}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
