"""Evaluate a checkpoint on a run config's eval sets exactly like the trainer's complete (final) eval.

scripts/04_distill.py ends a run with a full eval of its final weights: teacher-forced KL / CE / top-1 against the stored
teacher top-16 on every utterance of the COMPLETE eval sets, a greedy decode of all of them, the summary with its
headline numbers and combined loss, and the verdict. This script runs that eval on a weights dir the trainer saved
(runs/<run_id>/checkpoints/step_<N>/, or a student init) through the trainer's own code:
  data     04_distill.build_eval_store (the kept selection rows of the config's eval_sets, audio joined by id from the
           data shards, teacher top-k and text from teacher_out), greedy_subset_ids (the fixed greedy subset whose
           summary sits next to the complete one) and, with --probe, the train probe: the in_probe rows of the train
           store the trainer builds (train_store_spec), packed into a store of their own, and probe_greedy_subset
  model    04_distill.setup_model / setup_processing with the checkpoint as the student: fp32 weights, SDPA attention,
           the rel-pos-once-per-batch patch, frozen BatchNorm, the eval featuriser (LogMel of the checkpoint's
           processor), the config's autocast (bf16 on CUDA for the body; the LM head in fp32 outside it, and greedy_eval
           wraps it the same way) and its TF32 setting
  eval     kitsune.evaluate.teacher_forced_records + summarise_tf (= teacher_forced_eval), greedy_eval (the teacher
           pass's decode settings) and summarise_greedy, on the trainer's batches: the sets still to do are evaluated
           in ONE pooled pass of duration-sorted batches of eval.batch_s padded seconds (trainset.eval_batches), the
           very batches of the trainer's final eval when they are the config's eval_sets
  summary  04_distill.eval_summary (headline, headline_scope, combined_loss val_full), eval_history_record and
           gate_verdict over kitsune.evaluate.verdict
On the same hardware, weights and settings the per-utterance results and summary.json are the trainer's to the bit
(tests/test_evaluate_script.py, on CPU). A real run's numbers are close to its evals/step_<N>, not equal: its
checkpoints hold bf16 weights where the trainer evaluated its fp32 masters, and another GPU runs other kernels.

Output (--out; the layout of runs/<run_id>/evals/step_<N>/):
  tf_<set>.parquet, greedy_<set>.parquet  per utterance, as the trainer writes them (greedy with in_greedy_subset)
  probe.parquet, probe_greedy.parquet     with --probe
  summary.json    the trainer's keys: step, train_s, final, complete, tf, probe, greedy (the fixed subset's numbers,
                  from the complete decode), greedy_full, wall_s, headline, headline_scope, probe_greedy, epoch (epoch
                  mode), combined_loss; step / train_s / epoch from the checkpoint's student_meta.json "trained" block,
                  final = the checkpoint is the run's end one. It covers every set in --out
  verdict.json    when the three gate sets are in --out and so is the train probe, if the config has one (eval.probe:
                  pass --probe): gate_verdict of kitsune.evaluate.verdict, the trends read from the run's history
                  before this step (--history: default the summary.json of the run the checkpoint sits in) and this
                  eval. The trainer's record of this eval holds the probe's KL, which the probe-KL and gap trends read,
                  so without the probe the verdict would not be the trainer's: none is written (a verdict_skipped
                  event says why) and an older one goes
  evaluator.json  every invocation: arguments, checkpoint, config, versions, sets, chunks, thermal pauses, status
  events.jsonl    one JSON line per event (store builds, chunks, thermal pauses, the summary)
  .parts/, .work/ the resume state (per-set raw teacher-forced rows, chunk results of an unfinished pass, the --force
                  record)
Resumable: a set whose outputs exist is skipped. The pass over the other sets is cut into chunks of whole batches
(--chunk-s seconds of audio), each saved when done, so a crash loses at most the chunk under way and the same command
continues where it stopped. The pass's per-set outputs are written when it ends; a stop while they are written is
finished from its chunks by the next run, whatever sets it names, so every set keeps the pass's batches. The resume
state is fsynced before it is renamed into place, and a file of it that cannot be read (a host crash can tear one) is
evaluated again, never trusted. --force evaluates the named sets (and, with --probe, the probe) again: its first run
deletes their outputs and the unfinished passes that hold them, and records what it forces (.parts/force.json); the
same forced command after a stop finds the record and continues where it stopped, like any other. The record goes
once everything it names is done (delete it to start a stopped forced run over).
A set evaluated in a later pass than the others was batched with its own pass's sets only (comparable, not bitwise
the trainer's). --out refuses results of other weights or other eval settings (batch_s, autocast, device, data): use a
new --out for those.

Paths: the config's relative paths (data_root, teacher_root, second_root, selection, ...) resolve against --root (the
data checkout; default this repo), so the code can run from a clean worktree against another checkout's data, which
it only reads. The stores are built under --cache-dir (default <out>/cache), never the config's cache_dir. --config
takes a trainer config (configs/*.json) or a run's config.json (the resolved config under its "config" key).

GPU safety (the laptop GPU is unstable under long loads): before every batch the GPU temperature is read (nvidia-ml-py
if importable, else nvidia-smi at most every NVSMI_EVERY_S); at --max-temp or above the eval waits, polling every
--poll-s, until it is down to --resume-temp (a thermal_pause and a thermal_resume event each; --max-temp 0: no guard).
No readable temperature on CUDA stops the eval before it starts, and BLIND_READS failed reads in a row stop it later,
paused or not (the finished chunks are kept). The CUDA caching allocator is capped as the trainer caps it
(04_distill.cap_vram): on Windows at the free VRAM less a margin, since the driver would otherwise serve a batch that
does not fit from shared system memory, several times slower, instead of failing it; --vram-frac also caps it at that
fraction of the card, on any OS. An out-of-memory error stops with a pointer to --batch-s (a smaller one needs a new
--out; its batches are no longer the trainer's).

Usage:
  python scripts/05_evaluate.py --root D:/Shizu-ko-distill --config configs/viability.json \
      --ckpt runs/<run_id>/checkpoints/step_<N> --out <dir> [--probe] [--sets eval_jsut eval_cv8] [--vram-frac 0.95]
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as the trainer, before any CUDA use

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from kitsune import trainset  # noqa: E402
from kitsune.runlog import _finite, _replace  # noqa: E402
from kitsune.store import fsync_path  # noqa: E402

DEFAULT_CONFIG = "configs/viability.json"
# config keys holding paths, resolved against --root when relative (cache_dir is replaced by --cache-dir)
PATH_KEYS = ("student", "data_root", "teacher_root", "second_root", "selection", "runs_root")
NVSMI_EVERY_S = 5.0  # nvidia-smi is a process start per read: at most one read per this many seconds
BLIND_READS = 30  # this many failed temperature reads in a row stop the eval, paused or not
PARTS, WORK = ".parts", ".work"
MANIFEST = "manifest.json"  # in a pass's work dir: its sets and number of chunks
FORCE = "force.json"  # in .parts: what an unfinished --force run evaluates again


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_trainer():
    """scripts/04_distill.py as a module (its name starts with a digit): the trainer's setup, data and summary code."""
    spec = importlib.util.spec_from_file_location("kitsune_distill_for_eval", ROOT / "scripts" / "04_distill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _durable(tmp: Path, path: Path):
    """tmp -> path with its data on disk first (fsync, then rename, as the trainer saves its checkpoints): after an
    unflushed rename a host crash can leave the new name empty or zero-filled on NTFS, and this laptop's GPU faults can
    take the machine down."""
    fsync_path(tmp)
    _replace(tmp, path)


def _write_json(path: Path, obj):
    """Durable JSON that keeps NaN / Infinity (the resume state: summaries must come back exactly as computed)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    _durable(tmp, path)


def _write_output_json(path: Path, obj):
    """Durable strict JSON (non-finite floats as null), as RunLogger.eval_json writes summary.json and verdict.json."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json.dumps(_finite(obj), indent=2, ensure_ascii=False, default=str).encode("utf-8"))
    _durable(tmp, path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path):
    """path's JSON, or None when it is missing or cannot be read (torn: evaluated again, never trusted)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rows(path: Path) -> int | None:
    """A parquet file's row count from its footer, or None when it is missing or cannot be read (torn)."""
    try:
        return pq.read_metadata(path).num_rows
    except Exception:  # noqa: BLE001 - OSError, ArrowInvalid (no parquet footer)
        return None


def _table(df: pd.DataFrame, path: Path):
    """A per-utterance table as RunLogger.table writes it (zstd parquet, no index), durable."""
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    _durable(tmp, path)


def _rmdir_if_empty(p: Path):
    try:
        p.rmdir()
    except OSError:  # not empty, or not there
        pass


# ------------------------------------------------------------------------------------------------------ logging


class EvalLog:
    """events.jsonl in --out (one JSON line per event, no NaN) and a short console line. The trainer's helpers call
    .event(kind, **fields) on it as on their RunLogger."""

    def __init__(self, path: Path):
        self.path = path

    def event(self, kind: str, **kw):
        row = dict(kind=kind, time_utc=_now(), **kw)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_finite(row), ensure_ascii=False, default=str) + "\n")
        short = {k: v for k, v in kw.items() if not (isinstance(v, (list, dict, tuple)) and len(v) > 8)}
        print(f"[{kind}] " + " ".join(f"{k}={v}" for k, v in short.items()), flush=True)


# ------------------------------------------------------------------------------------------------- GPU thermals


class ThermalGuard:
    """check() before every batch: read the GPU temperature (at most every min_interval_s) and, at max_temp or above,
    wait (polling every poll_s) until it is down to resume_temp. Each pause is a thermal_pause and a thermal_resume
    event and a row of `pauses`. A failed read is an event (the first and every 100th) and lets the batch run (or the
    pause go on), but BLIND_READS failed reads in a row raise, paused or not: an unreadable GPU is an unguarded one
    (and in a pause it was last seen hot), as at the start, where no reading means no eval."""

    def __init__(self, read, max_temp: float, resume_temp: float, poll_s: float, log, *, min_interval_s: float = 0.0,
                 source: str = "?", sleep=time.sleep, clock=time.monotonic):
        if not resume_temp < max_temp:
            raise ValueError(f"--resume-temp ({resume_temp}) must be below --max-temp ({max_temp})")
        self.read, self.max_temp, self.resume_temp, self.poll_s = read, float(max_temp), float(resume_temp), poll_s
        self.log, self.min_interval_s, self.source, self.sleep, self.clock = log, min_interval_s, source, sleep, clock
        self.last_read = -math.inf
        self.max_seen = None
        self.checks = self.reads = self.read_errors = 0
        self.blind = 0  # failed reads in a row
        self.pauses: list[dict] = []

    def _read(self) -> float | None:
        self.reads += 1
        try:
            t = float(self.read())
        except Exception as e:  # noqa: BLE001 - one flaky read must not end a long eval by itself
            self.read_errors += 1
            self.blind += 1
            if self.read_errors == 1 or self.read_errors % 100 == 0:
                self.log.event("thermal_read_error", error=f"{type(e).__name__}: {e}"[:300], n=self.read_errors,
                               in_a_row=self.blind)
            return None
        self.blind = 0
        self.max_seen = t if self.max_seen is None else max(self.max_seen, t)
        return t

    def _stop_if_blind(self, paused_at: float | None):
        if self.blind < BLIND_READS:
            return
        where = f"while paused at {paused_at} C" if paused_at is not None else "between batches"
        raise RuntimeError(f"thermal guard: no GPU temperature for {self.blind} reads in a row {where}; stopping (the "
                           "finished chunks are kept: run the same command again)")

    def check(self):
        self.checks += 1
        now = self.clock()
        if now - self.last_read < self.min_interval_s:
            return
        self.last_read = now
        t = self._read()
        if t is None:
            self._stop_if_blind(None)
            return
        if t < self.max_temp:
            return
        self.log.event("thermal_pause", temp_c=t, max_temp=self.max_temp, resume_temp=self.resume_temp,
                       poll_s=self.poll_s)
        cur = None
        while True:
            self.sleep(self.poll_s)
            cur = self._read()
            if cur is None:
                self._stop_if_blind(t)
                continue
            if cur <= self.resume_temp:
                break
        waited = round(self.clock() - now, 1)
        self.last_read = self.clock()
        self.pauses.append(dict(temp_c=t, resumed_c=cur, waited_s=waited, time_utc=_now()))
        self.log.event("thermal_resume", temp_c=cur, waited_s=waited, pauses=len(self.pauses))

    def record(self) -> dict:
        return dict(source=self.source, max_temp=self.max_temp, resume_temp=self.resume_temp, poll_s=self.poll_s,
                    checks=self.checks, reads=self.reads, read_errors=self.read_errors, max_seen_c=self.max_seen,
                    pauses=self.pauses, paused_s=round(sum(p["waited_s"] for p in self.pauses), 1))


class Guarded:
    """The eval featuriser behind the thermal guard. kitsune.evaluate calls the featuriser once per batch, right before
    that batch's forward pass (the previous batch's results are already on the host), so the guard runs between
    batches of every teacher-forced and greedy pass without a change to the eval code. The features are the
    featuriser's own."""

    def __init__(self, featurizer, guard: ThermalGuard):
        self.featurizer, self.guard = featurizer, guard

    def __call__(self, wave, lengths):
        self.guard.check()
        return self.featurizer(wave, lengths)


def _bus_tuple(s) -> tuple[int, ...]:
    """'00000000:01:00.0' / '0000:01:00.0' / '01:00.0' (str or bytes) -> (domain, bus, device, function)."""
    s = s.decode() if isinstance(s, bytes) else str(s)
    head, _, fn = s.strip().partition(".")
    parts = [int(x, 16) for x in head.split(":")]
    return tuple(([0] * (3 - len(parts)) + parts) + [int(fn or "0", 16)])


def _pci_bus_id(idx: int) -> str | None:
    """The PCI bus id of torch's CUDA device idx (NVML / nvidia-smi number GPUs their own way), None if unknown."""
    try:
        p = torch.cuda.get_device_properties(idx)
        return f"{int(p.pci_domain_id):08X}:{int(p.pci_bus_id):02X}:{int(p.pci_device_id):02X}.0"
    except Exception:  # noqa: BLE001
        return None


def _nvml_reader(idx: int, bus: str | None):
    import pynvml  # nvidia-ml-py

    pynvml.nvmlInit()
    handle = None
    if bus:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            if _bus_tuple(pynvml.nvmlDeviceGetPciInfo(h).busId) == _bus_tuple(bus):
                handle = h
                break
    if handle is None:
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def read() -> float:
        return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

    read()
    return read


def _smi_reader(idx: int, bus: str | None):
    exe = shutil.which("nvidia-smi")
    if not exe:
        raise FileNotFoundError("nvidia-smi is not on PATH")
    errors = []
    for target in [t for t in (bus, str(idx)) if t]:
        def read(target=target) -> float:
            r = subprocess.run([exe, f"--id={target}", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30, check=True)
            return float(r.stdout.strip().splitlines()[0])

        try:
            read()
            return read
        except Exception as e:  # noqa: BLE001
            errors.append(f"--id={target}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors))


def make_guard(device: torch.device, args, log) -> ThermalGuard | None:
    """The thermal guard for a CUDA device (None on CPU or with --max-temp 0): nvidia-ml-py, else nvidia-smi. No
    readable temperature stops here, before any work."""
    if device.type != "cuda" or not args.max_temp:
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    bus = _pci_bus_id(idx)
    errors = []
    for source, factory, every in (("nvml", _nvml_reader, 0.0), ("nvidia-smi", _smi_reader, NVSMI_EVERY_S)):
        try:
            read = factory(idx, bus)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{source}: {type(e).__name__}: {e}"[:300])
            continue
        guard = ThermalGuard(read, args.max_temp, args.resume_temp, args.poll_s, log, min_interval_s=every,
                             source=source)
        log.event("thermal_guard", source=source, gpu=idx, pci_bus_id=bus, max_temp=args.max_temp,
                  resume_temp=args.resume_temp, poll_s=args.poll_s, min_interval_s=every, temp_c=read())
        return guard
    raise SystemExit("cannot read the GPU temperature (" + " | ".join(errors) + "): pip install nvidia-ml-py, or pass "
                     "--max-temp 0 to evaluate without the thermal guard")


# ------------------------------------------------------------------------------------------------------ config


def resolve_config(D, args, out: Path, ckpt: Path) -> tuple[dict, Path, str]:
    """The trainer's config (DEFAULTS merged with the file, --set on top, validated as the trainer does), relative
    paths resolved against --root, the stores under --cache-dir and the checkpoint as the student. Returns (cfg, the
    config file, the config's own student path)."""
    p = Path(args.config)
    if not p.is_absolute():  # as 04_distill.load_config: the working directory first, then this repo
        p = Path.cwd() / p if (Path.cwd() / p).exists() else ROOT / p
    raw = _read_json(p)
    if isinstance(raw.get("config"), dict) and "eval_sets" in raw["config"]:
        raw = raw["config"]  # a run's config.json: {run_id, argv, config, student_meta, ...}
    cfg = D._merge(D.DEFAULTS, raw)
    for s in args.set:
        D.apply_set(cfg, s)
    if args.batch_s is not None:
        cfg["eval"]["batch_s"] = float(args.batch_s)
    D.validate(cfg)
    root = Path(args.root).resolve()
    for k in PATH_KEYS:
        if not Path(cfg[k]).is_absolute():
            cfg[k] = str(root / cfg[k])
    student = cfg["student"]
    cfg["cache_dir"] = str(Path(args.cache_dir).resolve() if args.cache_dir else out / "cache")
    cfg["student"] = str(ckpt)  # setup_model / setup_processing load cfg["student"]
    return cfg, p, student


def weights_hash(ckpt: Path) -> str:
    """A cheap identity of the checkpoint's weights: every *.safetensors file's name, size, first and last MiB."""
    h = hashlib.sha256()
    for f in sorted(ckpt.glob("*.safetensors")):
        size = f.stat().st_size
        h.update(f"{f.name}:{size}".encode())
        with open(f, "rb") as fh:
            h.update(fh.read(1 << 20))
            fh.seek(max(0, size - (1 << 20)))
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


def check_identity(out: Path, identity: dict):
    """--out holds the results of one set of weights and eval settings: the first run records them, a later run with
    others stops here (a summary must not pool them)."""
    (out / PARTS).mkdir(parents=True, exist_ok=True)
    p = out / PARTS / "identity.json"
    if p.exists():
        old = _load_json(p)
        if not isinstance(old, dict):
            raise SystemExit(f"{p} cannot be read, so whose results {out} holds is unknown: use a new --out (or delete "
                             "that file if they are this checkpoint's with these settings)")
        diff = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
        if diff:
            raise SystemExit(f"{out} holds results of other weights or eval settings ({', '.join(diff)}: "
                             f"{ {k: old.get(k) for k in diff} } there, { {k: identity.get(k) for k in diff} } now): "
                             "use a new --out")
    else:
        _write_json(p, identity)


def load_history(spec: str, ckpt: Path, trained: dict) -> tuple[list[dict], str | None]:
    """The eval history the verdict's trends read: "none", a run dir or its summary.json, or "auto": the summary.json
    of the run the checkpoint sits in (runs/<run_id>/checkpoints/step_<N>/) when its run_id is the checkpoint's."""
    if spec == "none":
        return [], None
    if spec == "auto":
        if ckpt.parent.name != "checkpoints":
            return [], None
        p = ckpt.parent.parent / "summary.json"
        if not p.is_file():
            return [], None
        s = _read_json(p)
        if trained.get("run_id") and s.get("run_id") != trained["run_id"]:
            return [], None
        return list(s.get("history") or []), str(p)
    p = Path(spec)
    p = p / "summary.json" if p.is_dir() else p
    return list(_read_json(p).get("history") or []), str(p)


# ------------------------------------------------------------------------------------------------ the eval pass


@dataclass
class Ctx:
    D: object
    ev: object
    cfg: dict
    args: argparse.Namespace
    out: Path
    log: EvalLog
    store: object
    greedy_ids: set
    identity_key: str
    R: object = None
    feat: object = None
    guard: ThermalGuard | None = None
    passes: list = field(default_factory=list)
    probe_empty: bool = False  # --probe found no probe rows: the trainer has no probe numbers either

    @property
    def bs(self) -> float:
        return float(self.cfg["eval"]["batch_s"])


def set_files(out: Path, s: str) -> list[Path]:
    """A set's outputs, the part record last (it marks the set done)."""
    return [out / PARTS / f"{s}.tf_raw.parquet", out / f"tf_{s}.parquet", out / f"greedy_{s}.parquet",
            out / PARTS / f"{s}.json"]


def set_done(out: Path, s: str) -> bool:
    """The set's outputs are there and readable, with the rows its part record counts."""
    rec = _load_json(out / PARTS / f"{s}.json")
    return (isinstance(rec, dict) and _rows(out / PARTS / f"{s}.tf_raw.parquet") == rec.get("n_tf")
            and _rows(out / f"greedy_{s}.parquet") == rec.get("n_greedy")
            and _rows(out / f"tf_{s}.parquet") is not None)


def probe_files(out: Path) -> list[Path]:
    """The probe's outputs, the part record last (probe_greedy.parquet only when the probe has a greedy part)."""
    return [out / "probe.parquet", out / "probe_greedy.parquet", out / PARTS / "probe.json"]


def probe_done(out: Path) -> bool:
    rec = _load_json(out / PARTS / "probe.json")
    return (isinstance(rec, dict) and _rows(out / "probe.parquet") is not None
            and (rec.get("probe_greedy") is None or _rows(out / "probe_greedy.parquet") is not None))


def chunk_meta(work: Path, k: int) -> dict | None:
    """Chunk k's record when the chunk is done (the record and its tables readable, with the rows it counts), else
    None: the chunk is evaluated again."""
    meta = _load_json(work / f"chunk_{k:05d}.json")
    if not isinstance(meta, dict) or "n_tf" not in meta or "n_greedy" not in meta:
        return None
    for kind in ("tf", "greedy"):
        n = meta[f"n_{kind}"]
        if n and _rows(work / f"chunk_{k:05d}.{kind}.parquet") != n:
            return None
    return meta


def _work_dirs(out: Path) -> list[Path]:
    root = out / WORK
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def start_force(ctx: Ctx, sets: list[str], probe: bool):
    """--force. Its first run deletes the named sets' outputs (and the probe's), and the unfinished passes that hold
    any of those sets, and records what it forces (.parts/force.json). The same forced command after a stop finds that
    record and deletes nothing, so it continues the chunks done since, as any command does (end_force drops it)."""
    out = ctx.out
    want = dict(sets=list(sets), probe=bool(probe))
    old = _load_json(out / PARTS / FORCE)
    if isinstance(old, dict) and {k: old.get(k) for k in want} == want:
        ctx.log.event("force_continue", since=old.get("time_utc"), **want)
        return
    for s in sets:
        for p in set_files(out, s):
            p.unlink(missing_ok=True)
    if probe:
        for p in probe_files(out):
            p.unlink(missing_ok=True)
    for work in _work_dirs(out):
        man = _load_json(work / MANIFEST)
        if not isinstance(man, dict) or set(man.get("sets") or ()) & set(sets):
            shutil.rmtree(work, ignore_errors=True)
    _write_json(out / PARTS / FORCE, dict(want, time_utc=_now()))
    ctx.log.event("force", **want)


def end_force(ctx: Ctx):
    """The --force record goes once everything it names is done, by the forced command or any other."""
    p = ctx.out / PARTS / FORCE
    if not p.exists():
        return
    rec = _load_json(p)
    if isinstance(rec, dict):  # an unreadable record is of no use: it goes
        if not all(set_done(ctx.out, s) or not ctx.store.indices(source=s) for s in rec.get("sets") or ()):
            return
        if rec.get("probe") and not (probe_done(ctx.out) or ctx.probe_empty):
            return
    p.unlink(missing_ok=True)
    ctx.log.event("force_done", sets=(rec or {}).get("sets"), probe=(rec or {}).get("probe"))


def sweep_work(ctx: Ctx):
    """The passes earlier runs left in .work. One whose chunks are all done stopped while its per-set outputs were
    being written: they are written now, from its chunks (whatever sets this run names), so its sets keep the pass's
    batches. One that holds a set done since can never continue (no later pass batches a done set again) and goes, as
    does one of other weights or settings. The rest stay for the command that continues them."""
    for work in _work_dirs(ctx.out):
        man = _load_json(work / MANIFEST)
        if not isinstance(man, dict) or "sets" not in man or "chunks" not in man:
            continue  # the pass of that fingerprint writes it again
        sets, n = list(man["sets"]), int(man["chunks"])
        if man.get("identity") != ctx.identity_key:
            shutil.rmtree(work, ignore_errors=True)
        elif all(chunk_meta(work, k) is not None for k in range(n)):
            ctx.log.event("pass_finish", sets=sets, chunks=n, work=str(work),
                          note="its chunks were all done: the sets' outputs are written from them")
            finish_pass(ctx, sets, n, work)
            ctx.passes.append(dict(sets=sets, chunks=n, chunks_resumed=n, finished_from_chunks=True))
        elif any(set_done(ctx.out, s) for s in sets):
            ctx.log.event("pass_dropped", sets=sets, work=str(work), note="it holds a set done since")
            shutil.rmtree(work, ignore_errors=True)
    _rmdir_if_empty(ctx.out / WORK)


def chunk_plan(plan: list[list[int]], dur: np.ndarray, chunk_s: float) -> list[list[list[int]]]:
    """Consecutive whole batches of the plan, each chunk >= chunk_s seconds of audio (the last one shorter). A chunk's
    ids, in plan order, are duration-sorted already, so eval_batches over them cuts exactly these batches again."""
    chunks, cur, a = [], [], 0.0
    for b in plan:
        cur.append(b)
        a += float(dur[b].sum())
        if a >= chunk_s:
            chunks.append(cur)
            cur, a = [], 0.0
    if cur:
        chunks.append(cur)
    return chunks


def _oom(ctx: Ctx, e: BaseException):
    ctx.log.event("oom", batch_s=ctx.bs, error=f"{type(e).__name__}: {e}"[:500])
    raise SystemExit(f"CUDA out of memory at eval.batch_s {ctx.bs:g}: the finished chunks are kept, but a smaller "
                     "--batch-s (e.g. 240) changes the batches, so run it into a new --out") from e


def run_pass(ctx: Ctx, sets: list[str]):
    """One pooled pass over `sets` (the trainer's batches over their utterances), in resumable chunks; at its end the
    per-set outputs (finish_pass)."""
    store, ev, R = ctx.store, ctx.ev, ctx.R
    chosen = set(sets)
    idx = [i for i, u in enumerate(store.utts) if u.source in chosen]
    dur = np.array([u.duration for u in store.utts], dtype=np.float64)
    plan = trainset.eval_batches(store.utts, ctx.bs, idx)
    chunks = chunk_plan(plan, dur, float(ctx.args.chunk_s))
    ids_of = [[store.utts[i].id for b in ch for i in b] for ch in chunks]
    fp = hashlib.sha256(json.dumps(dict(identity=ctx.identity_key, sets=sets, chunks=ids_of)).encode()).hexdigest()
    work = ctx.out / WORK / f"pass-{fp[:16]}"
    work.mkdir(parents=True, exist_ok=True)
    manifest = dict(sets=sets, chunks=len(chunks), identity=ctx.identity_key)
    if _load_json(work / MANIFEST) != manifest:
        _write_json(work / MANIFEST, manifest)
    total_s = float(dur[idx].sum())
    done = {k for k in range(len(chunks)) if chunk_meta(work, k) is not None}
    ctx.log.event("pass", sets=sets, utts=len(idx), audio_h=round(total_s / 3600, 3), batches=len(plan),
                  chunks=len(chunks), chunks_done=len(done), batch_s=ctx.bs, work=str(work))
    t_start, audio_now = time.time(), 0.0
    left_s = total_s - sum(float(dur[[i for b in chunks[k] for i in b]].sum()) for k in done)
    for k, ch in enumerate(chunks):
        if k in done:
            continue
        meta_p = work / f"chunk_{k:05d}.json"
        ids = ids_of[k]
        members = [i for b in ch for i in b]
        set_audio: dict[str, float] = {}
        for i in members:
            set_audio[store.utts[i].source] = set_audio.get(store.utts[i].source, 0.0) + float(dur[i])
        try:
            t0 = time.time()
            raw, dropped_tf = ev.teacher_forced_records(R.model, store, ctx.feat, R.device, ctx.bs, ids=ids,
                                                        amp=R.amp)
            t1 = time.time()
            _, gdf = ev.greedy_eval(R.model, store, ids, ctx.feat, R.device, ctx.bs, tokenizer=R.tokenizer, amp=R.amp)
            t2 = time.time()
        except torch.OutOfMemoryError as e:
            _oom(ctx, e)
        ctx.D.assert_bn_frozen(R.model)
        got = set(gdf["id"])
        dropped_g = [i for i in ids if i not in got]
        if len(raw):
            _table(raw, work / f"chunk_{k:05d}.tf.parquet")
        if len(gdf):
            _table(gdf, work / f"chunk_{k:05d}.greedy.parquet")
        audio = float(sum(set_audio.values()))
        _write_json(meta_p, dict(k=k, n=len(ids), n_tf=len(raw), n_greedy=len(gdf), audio_s=audio,
                                 set_audio=set_audio, dropped_tf=dropped_tf,
                                 dropped_greedy=dropped_g, tf_wall_s=t1 - t0, greedy_wall_s=t2 - t1,
                                 temp_max_c=ctx.guard.max_seen if ctx.guard else None, time_utc=_now()))
        audio_now += audio
        el = time.time() - t_start
        eta = (left_s - audio_now) * el / audio_now if audio_now else None
        ctx.log.event("chunk", k=k + 1, of=len(chunks), utts=len(ids), audio_s=round(audio, 1),
                      tf_s=round(t1 - t0, 1), greedy_s=round(t2 - t1, 1), dropped=len(dropped_g),
                      eta_min=round(eta / 60, 1) if eta is not None else None,
                      **({"temp_max_c": ctx.guard.max_seen} if ctx.guard else {}))
    finish_pass(ctx, sets, len(chunks), work)
    ctx.passes.append(dict(sets=sets, utts=len(idx), batches=len(plan), chunks=len(chunks), chunks_resumed=len(done),
                           wall_s=round(time.time() - t_start, 1)))


def finish_pass(ctx: Ctx, sets: list[str], n_chunks: int, work: Path):
    """The pass's per-set outputs from its chunks, in plan order: the raw teacher-forced rows (.parts, for the
    summary), tf_<set> and greedy_<set> tables as the trainer writes them, and the set's record (dropped rows, its
    share of the wall time) last. Then the pass's chunks go (the work dir is kept until every set is written, so a stop
    in here is finished by the next run: sweep_work)."""
    ev, out = ctx.ev, ctx.out
    metas = [chunk_meta(work, k) for k in range(n_chunks)]
    if any(m is None for m in metas):
        raise RuntimeError(f"{work}: chunks {[k for k, m in enumerate(metas) if m is None]} are not done")
    raws = [pd.read_parquet(work / f"chunk_{k:05d}.tf.parquet") for k, m in enumerate(metas) if m["n_tf"]]
    gdfs = [pd.read_parquet(work / f"chunk_{k:05d}.greedy.parquet") for k, m in enumerate(metas) if m["n_greedy"]]
    raw = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame(columns=["id", "source"])
    gdf = pd.concat(gdfs, ignore_index=True) if gdfs else pd.DataFrame(columns=["id", "source"])
    src = {u.id: u.source for u in ctx.store.utts}
    for s in sets:
        for p in set_files(out, s):
            p.unlink(missing_ok=True)
        raw_s = raw[raw["source"] == s].reset_index(drop=True)
        g_s = gdf[gdf["source"] == s].reset_index(drop=True)
        g_s["in_greedy_subset"] = g_s["id"].isin(ctx.greedy_ids)

        def share(key: str) -> float:
            return float(sum(m[key] * m["set_audio"].get(s, 0.0) / m["audio_s"] for m in metas if m["audio_s"]))

        d_tf = [i for m in metas for i in m["dropped_tf"] if src.get(i) == s]
        d_gr = [i for m in metas for i in m["dropped_greedy"] if src.get(i) == s]
        _table(raw_s, out / PARTS / f"{s}.tf_raw.parquet")
        _table(ev.summarise_tf(raw_s)[1], out / f"tf_{s}.parquet")
        _table(g_s, out / f"greedy_{s}.parquet")
        _write_json(out / PARTS / f"{s}.json", dict(
            set=s, n_tf=len(raw_s), n_greedy=len(g_s), dropped_tf=d_tf, dropped_greedy=d_gr,
            tf_wall_s=share("tf_wall_s"), greedy_wall_s=share("greedy_wall_s"), pass_sets=sets,
            batch_s=ctx.bs, chunks=n_chunks, time_utc=_now()))
        ctx.log.event("set_done", set=s, utts=len(g_s), undecodable=len(d_gr))
    shutil.rmtree(work, ignore_errors=True)
    _rmdir_if_empty(out / WORK)


def probe_store(ctx: Ctx):
    """The trainer's probe (run_eval's R.probe_ids) without its whole train store: the in_probe rows of the train
    store setup_data builds (train_store_spec), packed into a store of their own, whose order is the train store's
    (the shards are read in the same order); eval.probe_is_train: that train store itself (a small subset in the
    configs that use it). Returns (store, probe ids) or (None, [])."""
    D, cfg = ctx.D, ctx.cfg
    sel, data, teach, cache = (Path(cfg["selection"]), Path(cfg["data_root"]), Path(cfg["teacher_root"]),
                               Path(cfg["cache_dir"]))
    name, ids = D.train_store_spec(cfg, ctx.log)
    if cfg["eval"]["probe_is_train"]:
        st = trainset.build_stores(sel, data, teach, cache / name, cfg["sources"], ["train"], ids=ids, log=print)
        return st, [u.id for u in st.utts]
    rows = trainset.read_selection(sel, cfg["sources"], ["train"])
    probe = rows["id"][rows["in_probe"]]
    if ids is not None:
        probe = probe[probe.isin(set(ids))]
    probe = probe.tolist()
    if not probe:
        return None, []
    tag = hashlib.sha256("\n".join(sorted(probe)).encode()).hexdigest()[:8]
    st = trainset.build_stores(sel, data, teach, cache / f"probe_{name}_{tag}", cfg["sources"], ["train"], ids=probe,
                               log=print)
    return st, [u.id for u in st.utts]


def run_probe(ctx: Ctx):
    """run_eval's probe part: teacher-forced on the probe, greedy on its probe_greedy_subset; probe.parquet,
    probe_greedy.parquet and their summaries (.parts/probe.json)."""
    ev, R, out = ctx.ev, ctx.R, ctx.out
    t0 = time.time()
    st, probe_ids = probe_store(ctx)
    if not probe_ids:
        ctx.probe_empty = True
        ctx.log.event("probe_empty", note="the selection has no probe rows for these sources")
        return
    pg_ids = ctx.D.probe_greedy_subset(ctx.cfg, probe_ids, {u.id: u.duration for u in st.utts}, ctx.log)
    ctx.log.event("probe", utts=len(probe_ids), audio_h=round(st.hours, 3), greedy=len(pg_ids),
                  build_s=round(time.time() - t0, 1))
    try:
        probe_sum, probe_df = ev.teacher_forced_eval(R.model, st, ctx.feat, R.device, ctx.bs, ids=probe_ids, amp=R.amp)
        pg_sum = pg_df = None
        if pg_ids:
            pg_sum, pg_df = ev.greedy_eval(R.model, st, pg_ids, ctx.feat, R.device, ctx.bs, tokenizer=R.tokenizer,
                                           amp=R.amp)
    except torch.OutOfMemoryError as e:
        _oom(ctx, e)
    ctx.D.assert_bn_frozen(R.model)
    for p in probe_files(out):
        p.unlink(missing_ok=True)
    _table(probe_df, out / "probe.parquet")
    if pg_df is not None:
        _table(pg_df, out / "probe_greedy.parquet")
    _write_json(out / PARTS / "probe.json", dict(probe=probe_sum, probe_greedy=pg_sum, n_probe=len(probe_ids),
                                                 n_probe_greedy=len(pg_ids), time_utc=_now()))
    ctx.log.event("probe_done", utts=len(probe_df), greedy=len(pg_df) if pg_df is not None else 0,
                  wall_s=round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------------------ summary


def write_summary(ctx: Ctx, step: int, trained: dict, ckpt: Path) -> dict | None:
    """summary.json over every set in --out (and the probe, if it was evaluated), as run_eval writes it, and
    verdict.json when the three gate sets are among them and so is the probe, if the config has one. The rows are put
    in the order of the trainer's batches over those sets, so a set of sets evaluated in one pass gives the trainer's
    numbers to the bit."""
    D, ev, cfg, out, store = ctx.D, ctx.ev, ctx.cfg, ctx.out, ctx.store
    present = [s for s in cfg["eval_sets"] if set_done(out, s)]
    if not present:
        return None
    parts = {s: _read_json(out / PARTS / f"{s}.json") for s in present}
    union = set(present)
    plan = trainset.eval_batches(store.utts, ctx.bs, [i for i, u in enumerate(store.utts) if u.source in union])
    rank = {store.utts[i].id: r for r, i in enumerate(i for b in plan for i in b)}

    def ordered(frames: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(frames, ignore_index=True)
        return df.iloc[np.argsort(np.array([rank[i] for i in df["id"]], dtype=np.int64), kind="stable")].reset_index(
            drop=True)

    raw = ordered([pd.read_parquet(out / PARTS / f"{s}.tf_raw.parquet") for s in present])
    gdf = ordered([pd.read_parquet(out / f"greedy_{s}.parquet") for s in present])
    d_tf = sorted((i for s in present for i in parts[s]["dropped_tf"]), key=rank.__getitem__)
    d_gr = sorted((i for s in present for i in parts[s]["dropped_greedy"]), key=rank.__getitem__)
    tf_wall = sum(parts[s]["tf_wall_s"] for s in present)
    gr_wall = sum(parts[s]["greedy_wall_s"] for s in present)
    tf_sum, _ = ev.summarise_tf(raw, n_bad_audio=len(d_tf), bad_audio=d_tf[:50],
                                bad_audio_per_set=ev._bad_audio_per_set(store, d_tf), wall_s=tf_wall)
    rows = gdf.drop(columns="in_greedy_subset")
    audio_s = float(rows["duration"].sum()) if len(rows) else 0.0
    full_sum = ev.summarise_greedy(rows, n_bad_audio=len(d_gr), bad_audio=d_gr[:50],
                                   bad_audio_per_set=ev._bad_audio_per_set(store, d_gr), wall_s=gr_wall,
                                   rtf=gr_wall / audio_s if audio_s else float("nan"))
    gr_sum = ev.summarise_greedy(rows[gdf["in_greedy_subset"].astype(bool)], wall_s=full_sum["wall_s"])

    probe_sum = pg_sum = None
    n_probe = n_pg = 0
    probe_wall = 0.0
    if probe_done(out):
        pr = _read_json(out / PARTS / "probe.json")
        probe_sum, pg_sum, n_probe, n_pg = pr["probe"], pr["probe_greedy"], pr["n_probe"], pr["n_probe_greedy"]
        probe_wall = float(probe_sum.get("wall_s") or 0.0) + (float(pg_sum.get("wall_s") or 0.0) if pg_sum else 0.0)

    train_s = float(trained.get("train_s") or 0.0)
    epoch = float(trained.get("epoch") or 0.0) if D.epoch_mode(cfg) else None
    combined = {}
    if c := D.combined_val_full(cfg, tf_sum):
        combined["val_full"] = c
    summary = D.eval_summary(step, train_s, trained.get("reason") == "end", True, tf_sum, probe_sum, gr_sum, full_sum,
                             pg_sum, round(tf_wall + gr_wall + probe_wall, 1), n_probe_greedy=n_pg, n_probe=n_probe,
                             epoch=epoch, combined=combined)
    _write_output_json(out / "summary.json", summary)
    print(D.headline_line("full", step, epoch or 0.0, summary["headline"], summary["headline_scope"]), flush=True)
    ctx.log.event("summary", sets=present, probe=probe_sum is not None, headline=summary["headline"],
                  combined_loss={k: v["value"] for k, v in combined.items()})

    missing = [s for s in ev.GATE_SETS if s not in present]
    # the trainer's record of this eval holds the probe's KL (unless the probe has no rows), which the verdict's
    # probe-KL and gap trends read: a verdict without it would not be the trainer's
    no_probe = bool(cfg["eval"]["probe"]) and probe_sum is None and not ctx.probe_empty
    if missing or no_probe:
        (out / "verdict.json").unlink(missing_ok=True)  # an earlier verdict does not go with this summary
        ctx.log.event("verdict_skipped", **({"missing_gate_sets": missing} if missing else {}),
                      **({"probe_missing": True, "note": "the config has the train probe (eval.probe): pass --probe"}
                         if no_probe else {}))
        return summary
    hist, src = load_history(ctx.args.history, ckpt, trained)
    rec = D.eval_history_record(step, train_s, tf_sum, gr_sum, probe_sum, pg_sum, summary["headline"], epoch=epoch)
    history = sorted((r for r in hist if int(r["step"]) < step), key=lambda r: int(r["step"])) + [rec]
    verdict = D.gate_verdict(cfg, ev.verdict(dict(final=full_sum, history=history)))
    _write_output_json(out / "verdict.json", verdict)
    ctx.log.event("verdict", verdict=verdict.get("verdict"), reasons=verdict.get("reasons"), history=src,
                  history_steps=[int(r["step"]) for r in history])
    return summary


# --------------------------------------------------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the weights dir to evaluate (runs/<run_id>/checkpoints/step_<N>)")
    ap.add_argument("--out", required=True, help="output dir (the layout of evals/step_<N>/); resumable")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="trainer config or a run's config.json (default %(default)s)")
    ap.add_argument("--root", default=str(ROOT),
                    help="data checkout the config's relative paths resolve against (default: this repo)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key as 04_distill.py --set does (repeatable; JSON values)")
    ap.add_argument("--sets", nargs="+", default=None,
                    help="eval sets to evaluate (default: the config's eval_sets)")
    ap.add_argument("--probe", action="store_true",
                    help="also the train probe (teacher-forced) and its greedy part, as the trainer's evals")
    ap.add_argument("--batch-s", type=float, default=None,
                    help="padded audio seconds per eval batch (default: the config's eval.batch_s, the trainer's "
                         "batches)")
    ap.add_argument("--chunk-s", type=float, default=1800.0,
                    help="audio seconds per resumable chunk of the pass (default %(default)s)")
    ap.add_argument("--cache-dir", default=None, help="where the stores are built (default <out>/cache)")
    ap.add_argument("--history", default="auto",
                    help="the eval history for the verdict's trends: auto (the checkpoint's run), none, or a run dir / "
                         "summary.json")
    ap.add_argument("--step", type=int, default=None, help="the step to report (default: the checkpoint's)")
    ap.add_argument("--force", action="store_true",
                    help="evaluate the named sets (and --probe) again; after a stop, the same command continues")
    ap.add_argument("--max-temp", type=float, default=80.0,
                    help="pause before a batch at this GPU temperature (C) or above; 0: no thermal guard "
                         "(default %(default)s)")
    ap.add_argument("--resume-temp", type=float, default=70.0,
                    help="resume once the GPU is down to this temperature (C) (default %(default)s)")
    ap.add_argument("--poll-s", type=float, default=10.0,
                    help="seconds between temperature reads while paused (default %(default)s)")
    ap.add_argument("--vram-frac", type=float, default=None,
                    help="also cap the CUDA allocator at this fraction of the card (on Windows it is capped at the "
                         "free VRAM less a margin in any case, as the trainer's)")
    args = ap.parse_args(argv)
    if args.max_temp and not args.resume_temp < args.max_temp:
        ap.error("--resume-temp must be below --max-temp")
    if args.vram_frac is not None and not 0 < args.vram_frac <= 1:
        ap.error("--vram-frac must be in (0, 1]")
    if args.chunk_s <= 0:
        ap.error("--chunk-s must be > 0")
    return args


def _versions() -> dict:
    import transformers

    code = None
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        code = dict(sha=sha or None, dirty=bool(dirty))
    except Exception:  # noqa: BLE001
        pass
    return dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, code=code, code_root=str(ROOT))


def main(argv=None) -> int:
    args = parse_args(argv)
    D = load_trainer()
    from kitsune import evaluate as ev
    from kitsune import student as S

    ckpt = Path(args.ckpt).resolve()
    if not (ckpt / "config.json").is_file():
        raise SystemExit(f"{ckpt}: not a weights dir (no config.json)")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = EvalLog(out / "events.jsonl")
    t_start = time.time()
    cfg, cfg_path, cfg_student = resolve_config(D, args, out, ckpt)
    trained = S.load_meta(ckpt).get("trained") or {}
    step = int(args.step if args.step is not None else trained.get("step", 0))
    sets = list(dict.fromkeys(args.sets or cfg["eval_sets"]))
    unknown = [s for s in sets if s not in cfg["eval_sets"]]
    if unknown:
        raise SystemExit(f"--sets {unknown} not among the config's eval_sets {cfg['eval_sets']}")
    dev = cfg["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if dev in (None, "auto") else dev)
    inv = dict(time_utc=_now(), argv=list(argv) if argv is not None else sys.argv[1:], ckpt=str(ckpt), step=step,
               trained=trained, config=str(cfg_path), config_student=cfg_student, root=str(Path(args.root).resolve()),
               cache_dir=cfg["cache_dir"], sets=sets, probe=args.probe, force=args.force, device=str(device),
               autocast=cfg["autocast"], batch_s=float(cfg["eval"]["batch_s"]), chunk_s=args.chunk_s,
               versions=_versions(), status="running")
    log.event("evaluate_start", ckpt=str(ckpt), step=step, sets=sets, probe=args.probe, device=str(device),
              autocast=cfg["autocast"], batch_s=float(cfg["eval"]["batch_s"]), config=str(cfg_path),
              root=inv["root"], out=str(out))
    ctx = None
    try:
        seed = int(cfg["seed"])  # as the trainer (nothing in the eval draws from them)
        torch.manual_seed(seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high" if cfg["perf"]["tf32"] else "highest")
        t0 = time.time()
        store = D.build_eval_store(cfg, log)
        greedy_ids = D.greedy_subset_ids(cfg, store)
        log.event("data", eval_utts=len(store), eval_h=round(store.hours, 3), per_set=store.info.get("per_source"),
                  dropped=store.info.get("dropped"), greedy=len(greedy_ids), build_s=round(time.time() - t0, 1))
        try:
            base = ev.teacher_baselines(cfg["teacher_root"], cfg["eval_sets"], check=cfg["eval"]["check_baselines"])
            log.event("teacher_baselines", sets=base)
        except FileNotFoundError as e:
            log.event("teacher_baselines", skipped=str(e))
        ec = cfg["eval"]
        identity = dict(weights=weights_hash(ckpt), step=step, device=device.type, autocast=cfg["autocast"],
                        batch_s=float(ec["batch_s"]), tf32=bool(cfg["perf"]["tf32"]),
                        relpos_patch=bool(cfg["perf"]["relpos_patch"]), seed=seed, eval_sets=list(cfg["eval_sets"]),
                        sources=list(cfg["sources"]), subset=cfg["subset"], store=store.info.get("fingerprint"),
                        greedy_subset=ec["greedy_subset"], probe=[ec["probe"], ec["probe_is_train"],
                                                                  ec["probe_greedy_audio_s"]])
        check_identity(out, identity)
        ctx = Ctx(D=D, ev=ev, cfg=cfg, args=args, out=out, log=log, store=store, greedy_ids=set(greedy_ids),
                  identity_key=hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest())
        want_probe = args.probe and bool(cfg["eval"]["probe"])
        if args.probe and not cfg["eval"]["probe"]:
            log.event("probe_off", note="the config has eval.probe false: no probe to evaluate")
        sweep_work(ctx)  # a pass stopped while its outputs were written: finished from its chunks
        if args.force:
            start_force(ctx, sets, want_probe)
        todo = [s for s in sets if not set_done(out, s)]
        empty = [s for s in todo if not store.indices(source=s)]
        if empty:
            log.event("sets_empty", sets=empty, note="no kept rows with audio in the eval store")
            todo = [s for s in todo if s not in empty]
        skipped = [s for s in sets if s not in todo and s not in empty]
        do_probe = want_probe and not probe_done(out)
        log.event("todo", sets=todo, skipped_done=skipped, probe=do_probe)
        inv.update(evaluated=todo, skipped=skipped)
        if todo or do_probe:
            R = D.Run(cfg=cfg, run_dir=out, device=device, amp=cfg["autocast"] == "bfloat16")
            R.log = log
            ctx.guard = make_guard(device, args, log)
            # as the trainer: on Windows CUDA always (the driver's sysmem fallback would serve an oversized batch from
            # shared memory instead of raising OOM), --vram-frac on any OS; a no-op otherwise
            D.cap_vram(R, max_frac=args.vram_frac)
            D.setup_model(R, grad_ckpt=False)
            D.setup_processing(R)
            ctx.R = R
            ctx.feat = Guarded(R.feat_eval, ctx.guard) if ctx.guard else R.feat_eval
            if todo:
                run_pass(ctx, todo)
            if do_probe:
                run_probe(ctx)
        end_force(ctx)
        sweep_work(ctx)  # passes that hold a set done now can never continue
        summary = write_summary(ctx, step, trained, ckpt)
        inv.update(status="complete", headline=(summary or {}).get("headline"))
        return 0
    except BaseException as e:
        inv.update(status="failed", error=f"{type(e).__name__}: {e}"[:2000])
        raise
    finally:
        inv["wall_s"] = round(time.time() - t_start, 1)
        if ctx is not None:
            inv["passes"] = ctx.passes
            if ctx.guard is not None:
                inv["thermal"] = ctx.guard.record()
            if ctx.R is not None and ctx.R.vram_cap_gb is not None:
                inv["vram_cap_gb"] = ctx.R.vram_cap_gb
        try:
            p = out / "evaluator.json"
            rec = _load_json(p)
            if not (isinstance(rec, dict) and isinstance(rec.get("invocations"), list)):
                if p.exists():  # unreadable (torn): kept aside, a new record starts
                    _replace(p, p.with_name(f"evaluator.unreadable-{int(time.time())}.json"))
                rec = dict(invocations=[])
            rec["invocations"].append(inv)
            _write_output_json(p, rec)
        except Exception as e2:  # noqa: BLE001 - never masks the eval's own outcome
            print(f"could not update evaluator.json: {e2!r}", file=sys.stderr)
        log.event("evaluate_end", status=inv["status"], wall_s=inv["wall_s"])


if __name__ == "__main__":
    sys.exit(main())
