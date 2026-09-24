"""Run logger: everything a training run measures goes to TensorBoard AND to open formats that survive the instance.

This is a proof of concept, so the logs are the deliverable as much as the weights: every number must be extractable
later without TensorBoard (pandas/pyarrow on the parquet files), and the run dir is mirrored to the HF output repo
every ~10 min because the vast instance is destroyed at the end of the run.

Layout (relative to runs/<run_id>/):
  config.json                 resolved config + argv + student_meta.json (config.<stamp>.json on a restart)
  env/                        git_sha.txt, git_status.txt, git_diff.patch, pip_freeze.txt, nvidia_smi.txt,
                              system.json, sdpa_backends.json      (env/restart-<stamp>/ on a restart)
  tb/                         TensorBoard event files (mirror of everything below)
  metrics/scalars.jsonl       EVERY scalar: {"step","wall","elapsed_s","tag","value"}; a non-finite value is written
                              as null with "nf": "nan"|"inf"|"-inf" (JSON has no NaN)
  metrics/scalars.parquet     rewritten from the jsonl at each sync (tag, step, wall, elapsed_s, value)
  metrics/steps.parquet       one wide row per optimizer step (step, wall, elapsed_s, <every step_row key>)
  metrics/train_utts/part-*.parquet   one row per utterance per time it is trained on (flushed every ~500 steps)
  metrics/hist/part-*.parquet histogram summaries: quantiles, moments, 64-bin counts/edges (json)
  metrics/text.jsonl          every text() call (TensorBoard text is awkward to extract)
  evals/step_<N>/             summary.json, tf_<set>.parquet, greedy_<set>.parquet, probe.parquet (table/eval_json)
  samples/step_<N>.jsonl      text samples (ref / teacher / student), also TensorBoard text
  events.jsonl                lifecycle events, fsynced one by one (phase changes, OOM fallbacks, checkpoints,
                              exceptions with tracebacks, sync errors, verdict)
  logs/stdout.log             tee of sys.stdout / sys.stderr
  summary.json                final: verdict, best/final metrics, throughput, cost, wall times

Sync never blocks training: the caller only flushes file handles, notes the flushed length of every append-only file
and snapshots the in-memory tables; parquet rewrites and HfApi.upload_folder (to runs/<run_id>/, checkpoints/
excluded - the trainer uploads those explicitly) run in a daemon thread. The upload is of a copy of the run dir, the
append-only files cut at those lengths: the Hub client sizes a file when it lists the folder but hashes and reads it
later, and the live files keep growing (and parquet mirrors get replaced) meanwhile, which would commit a pointer
whose size, hash and content disagree. Upload errors are retried with back-off and become events, never exceptions.

Resume: pass the dict from state_dict() (kept in the trainer's full state) as `resume=`. elapsed_s continues, steps
after the restored step are dropped from steps.parquet and purged from TensorBoard, and are then logged again. The
append-only jsonl files keep both copies; the later wall time is the one that counts.
"""
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from array import array
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pajson
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

SCALAR_SCHEMA = pa.schema([("step", pa.int64()), ("wall", pa.float64()), ("elapsed_s", pa.float64()),
                           ("tag", pa.string()), ("value", pa.float64()), ("nf", pa.string())])
TRAIN_UTT_SCHEMA = pa.schema([("step", pa.int64()), ("epoch", pa.int64()), ("id", pa.string()), ("source", pa.string()),
                              ("duration", pa.float32()), ("n_tok", pa.int32()), ("kl", pa.float32()),
                              ("ce", pa.float32()), ("top1_acc", pa.float32()), ("masked_frac", pa.float32()),
                              ("agree", pa.float32())])
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
HIST_BINS = 64
# env vars worth keeping (vast host facts, our own launch settings); anything secret-looking is redacted by name
ENV_PREFIXES = ("VAST", "CONTAINER", "KITSUNE", "PUBLIC_IPADDR", "GPU_", "CUDA", "NVIDIA", "HF_", "PYTORCH", "OMP_",
                "TZ", "HOSTNAME", "SSH_CONNECTION")
SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASS", "AUTH", "CRED", "COOKIE")
SYNC_IGNORE = ["checkpoints/*", "*.tmp"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _num(v) -> float:
    return float(v.item() if hasattr(v, "item") else v)


def _replace(tmp: Path, path: Path, tries: int = 100):
    """tmp -> path. On Windows a file another thread has open (the sync's snapshot copy, a virus scanner) cannot be
    replaced for those few milliseconds (PermissionError), so retry there for up to ~5 s."""
    for i in range(tries):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if os.name != "nt" or i == tries - 1:
                raise
            time.sleep(0.05)


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    _replace(tmp, path)


def _atomic_json(path: Path, obj):
    _atomic_write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8"))


def _atomic_parquet(table: pa.Table, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    _replace(tmp, path)


def _repair_tail(path: Path):
    """Cut a torn last line (process killed mid-append), so line-based readers never see half a record."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        f.seek(data.rfind(b"\n") + 1)
        f.truncate()


def read_scalars_jsonl(path, limit: int | None = None) -> pa.Table:
    """metrics/scalars.jsonl -> table (tag, step, wall, elapsed_s, value) with NaN/inf restored.

    Only complete lines are parsed (up to `limit` bytes if given): the file may be growing while it is read, and a
    copy uploaded mid-append can end in half a line."""
    path = Path(path)
    data = b""
    if path.exists():
        with open(path, "rb") as f:
            data = f.read(limit) if limit is not None else f.read()
        data = data[: data.rfind(b"\n") + 1]
    if data:
        t = pajson.read_json(io.BytesIO(data), parse_options=pajson.ParseOptions(explicit_schema=SCALAR_SCHEMA,
                                                                                 unexpected_field_behavior="ignore"))
    else:
        t = SCALAR_SCHEMA.empty_table()
    value = t.column("value").to_numpy(zero_copy_only=False).astype(np.float64)
    nf = t.column("nf")
    if nf.null_count < len(nf):
        idx = np.flatnonzero(pc.is_valid(nf).to_numpy(zero_copy_only=False))
        value[idx] = [float(x) for x in nf.take(pa.array(idx)).to_pylist()]
    return pa.table({"tag": t.column("tag"), "step": t.column("step"), "wall": t.column("wall"),
                     "elapsed_s": t.column("elapsed_s"), "value": pa.array(value, pa.float64())})


def _copy_prefix(src: Path, dst: Path, limit: int | None = None):
    """Copy the first `limit` bytes of src (None: its size when opened), so the copy is one consistent state of a
    file that keeps growing; a file replaced by os.replace meanwhile is read from the version that was opened."""
    try:
        with open(src, "rb") as f:
            left = os.fstat(f.fileno()).st_size if limit is None else limit
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as g:
                while left > 0 and (chunk := f.read(min(left, 8 << 20))):
                    g.write(chunk)
                    left -= len(chunk)
    except (FileNotFoundError, PermissionError):
        pass  # renamed away, or (Windows) caught mid-replace: the next sync has it


def _next_part(d: Path) -> int:
    parts = sorted(d.glob("part-*.parquet"))
    return int(parts[-1].stem.split("-")[1]) + 1 if parts else 0


# -------------------------------------------------------------------------------------------- env / system


def _run(cmd: list[str], timeout: float = 30) -> str:
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else "") + (
            f"\n[exit {r.returncode}]" if r.returncode else "")
    except Exception as e:
        return f"[failed: {type(e).__name__}: {e}]"


def _versions() -> dict:
    from importlib import metadata

    out = {}
    for name in ("torch", "transformers", "huggingface_hub", "hf_xet", "tokenizers", "safetensors", "numpy",
                 "pyarrow", "pandas", "soundfile", "librosa", "soxr", "jiwer", "tensorboard", "nvidia-ml-py", "psutil"):
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def _ram_total_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return None


def safe_env() -> dict[str, str]:
    """Launch-relevant env vars with anything secret-looking redacted (the name stays, so its presence is visible)."""
    out = {}
    for k, v in sorted(os.environ.items()):
        if not k.upper().startswith(ENV_PREFIXES):
            continue
        out[k] = "<redacted>" if any(m in k.upper() for m in SECRET_MARKERS) else v
    return out


def _try(fn):
    try:
        return fn()
    except Exception as e:  # e.g. cudnn.version() raises when no GPU is visible
        return f"[failed: {type(e).__name__}: {e}]"


def system_info() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    info = dict(host=platform.node(), platform=platform.platform(), python=sys.version, executable=sys.executable,
                cpu_count=os.cpu_count(),
                cpu_affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                ram_gb=_ram_total_gb(), versions=_versions(), torch_cuda=torch.version.cuda,
                cudnn=_try(torch.backends.cudnn.version) if cuda else None, cuda_available=cuda,
                time_utc=_now_iso(), env=safe_env())
    info["libsndfile"] = _try(lambda: __import__("soundfile").__libsndfile_version__)
    if cuda:
        info["gpus"] = _try(lambda: [dict(name=torch.cuda.get_device_name(i),
                                          capability=list(torch.cuda.get_device_capability(i)),
                                          mem_gb=torch.cuda.get_device_properties(i).total_memory / 2**30)
                                     for i in range(torch.cuda.device_count())])
    nv = _nvml()
    if nv is not None:
        info["driver"] = _try(nv.nvmlSystemGetDriverVersion)
        info["nvml_cuda_driver"] = _try(nv.nvmlSystemGetCudaDriverVersion_v2)
    info["instance_id"] = os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL")
    info["offer_dph"] = _try(lambda: float(os.environ["KITSUNE_DPH"])) if os.environ.get("KITSUNE_DPH") else None
    return info


def _sdpa_report() -> dict:
    try:
        from kitsune.patches import sdpa_backend_report

        return sdpa_backend_report()
    except Exception as e:  # patches not importable (or no CUDA): record the global flags at least
        import torch

        b = torch.backends.cuda
        return dict(fallback=f"{type(e).__name__}: {e}", flash=b.flash_sdp_enabled(), mem_efficient=b.mem_efficient_sdp_enabled(),
                    math=b.math_sdp_enabled(), cudnn=getattr(b, "cudnn_sdp_enabled", lambda: None)())


def capture_env(env_dir: Path, extra: dict | None = None):
    """Write the env/ folder. Every item is best-effort: a missing git or nvidia-smi is written down, not raised."""
    env_dir.mkdir(parents=True, exist_ok=True)
    sha = _run(["git", "rev-parse", "HEAD"]).strip()
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    (env_dir / "git_sha.txt").write_text(f"{sha}\nbranch: {branch}\nKITSUNE_SHA: {os.environ.get('KITSUNE_SHA', '')}\n",
                                         encoding="utf-8")
    (env_dir / "git_status.txt").write_text(_run(["git", "status", "--porcelain"]), encoding="utf-8")
    (env_dir / "git_diff.patch").write_text(_run(["git", "diff", "HEAD"]), encoding="utf-8")
    try:
        from importlib import metadata

        freeze = sorted({f"{d.metadata.get('Name')}=={d.version}" for d in metadata.distributions()
                         if d.metadata.get("Name")})
        (env_dir / "pip_freeze.txt").write_text("\n".join(freeze) + "\n", encoding="utf-8")
    except Exception as e:
        (env_dir / "pip_freeze.txt").write_text(f"[failed: {e}]\n", encoding="utf-8")
    smi = shutil.which("nvidia-smi")
    (env_dir / "nvidia_smi.txt").write_text(_run([smi]) + "\n\n" + _run([smi, "-q"]) if smi else "nvidia-smi not found\n",
                                            encoding="utf-8")
    try:
        info = system_info()
    except Exception as e:
        info = dict(error=f"{type(e).__name__}: {e}")
    info.update(extra or {})
    _atomic_json(env_dir / "system.json", info)
    _atomic_json(env_dir / "sdpa_backends.json", _sdpa_report())


_NVML = None
_NVML_TRIED = False
_PROC = None


def _nvml():
    global _NVML, _NVML_TRIED
    if not _NVML_TRIED:
        _NVML_TRIED = True
        try:
            import pynvml  # package nvidia-ml-py

            pynvml.nvmlInit()
            _NVML = pynvml
        except Exception:
            _NVML = None
    return _NVML


def system_stats() -> dict[str, float]:
    """GPU util/mem/power/temp/clocks via NVML, torch CUDA memory, process RSS and CPU %. Missing pieces are
    simply absent (no pynvml, no CUDA, no psutil)."""
    global _PROC
    out = {}
    nv = _nvml()
    if nv is not None:
        try:
            for i in range(nv.nvmlDeviceGetCount()):
                h, p = nv.nvmlDeviceGetHandleByIndex(i), f"sys/gpu{i}"
                probes = {
                    "util": lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu,
                    "mem_util": lambda: nv.nvmlDeviceGetUtilizationRates(h).memory,
                    "mem_used_gb": lambda: nv.nvmlDeviceGetMemoryInfo(h).used / 2**30,
                    "mem_total_gb": lambda: nv.nvmlDeviceGetMemoryInfo(h).total / 2**30,
                    "power_w": lambda: nv.nvmlDeviceGetPowerUsage(h) / 1000,
                    "temp_c": lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    "sm_clock_mhz": lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    "mem_clock_mhz": lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                }
                for k, fn in probes.items():
                    try:
                        out[f"{p}/{k}"] = float(fn())
                    except Exception:
                        pass
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            out["sys/cuda/allocated_gb"] = torch.cuda.memory_allocated() / 2**30
            out["sys/cuda/reserved_gb"] = torch.cuda.memory_reserved() / 2**30
            out["sys/cuda/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        pass
    try:
        import psutil

        if _PROC is None:
            _PROC = psutil.Process()
            _PROC.cpu_percent(None)  # first call only primes the counter
        out["sys/proc/rss_gb"] = _PROC.memory_info().rss / 2**30
        out["sys/proc/cpu_pct"] = _PROC.cpu_percent(None)
        out["sys/proc/threads"] = float(_PROC.num_threads())
        out["sys/ram_used_pct"] = psutil.virtual_memory().percent
    except Exception:
        try:  # Linux without psutil
            with open("/proc/self/statm") as f:
                out["sys/proc/rss_gb"] = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        except Exception:
            pass
    if hasattr(os, "getloadavg"):
        out["sys/load1"] = os.getloadavg()[0]
    return out


# ------------------------------------------------------------------------------------------------------ tee


class _Tee(io.TextIOBase):
    def __init__(self, stream, fh, lock):
        self.stream, self.fh, self.lock = stream, fh, lock

    def write(self, s):
        n = self.stream.write(s)
        with self.lock:
            if not self.fh.closed:
                self.fh.write(s)
                if "\n" in s:
                    self.fh.flush()
        return n

    def flush(self):
        self.stream.flush()
        with self.lock:
            if not self.fh.closed:
                self.fh.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ------------------------------------------------------------------------------------------------ RunLogger


class RunLogger:
    def __init__(self, run_dir, cfg: dict, hf_repo: str | None = None, sync_every_min: float = 10, *,
                 resume: dict | None = None, student_meta: dict | None = None, tee: bool = True,
                 capture: bool = True, train_utts_flush_steps: int = 500, api=None,
                 upload_retries: tuple[float, ...] = (15, 60, 180)):
        import torch  # noqa: F401  (SummaryWriter needs it anyway; import errors surface here, not mid-run)
        from torch.utils.tensorboard import SummaryWriter

        self.dir = Path(run_dir)
        self.run_id = self.dir.name
        self.cfg, self.hf_repo = cfg, hf_repo
        self.sync_every_s = sync_every_min * 60
        self.train_utts_flush_steps = train_utts_flush_steps
        self.upload_retries = upload_retries
        self._api = api
        self._repo_ready = False
        self._lock = threading.RLock()
        self._part_lock = threading.Lock()
        self._sync_thread: threading.Thread | None = None
        self._closed = False
        restart = (self.dir / "events.jsonl").exists()  # a previous logger ran here (resume or re-launch)
        for d in ("env", "tb", "metrics/train_utts", "metrics/hist", "evals", "samples", "logs"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        m = self.dir / "metrics"
        self.p_scalars, self.p_text, self.p_events = m / "scalars.jsonl", m / "text.jsonl", self.dir / "events.jsonl"
        for p in (self.p_scalars, self.p_text, self.p_events):
            _repair_tail(p)

        self._t0 = time.monotonic()
        self._elapsed0 = float(resume["elapsed_s"]) if resume else 0.0
        self.step = int(resume["step"]) if resume else 0
        self._last_sync = time.monotonic()
        self._last_utt_flush = self.step

        # wide step table: numeric columns as array('d') (NaN = not logged that step), ~8 B per cell
        self._steps: dict[str, array] = {}
        self._n_steps = 0
        self._step_col, self._wall_col, self._el_col = array("q"), array("d"), array("d")
        if (m / "steps.parquet").exists():
            self._load_steps(m / "steps.parquet", max_step=self.step if resume else None)
        self._steps_written = 0
        self._hist_rows: list[dict] = []
        self._utt_rows: list[dict] = []

        self._f_scalars = open(self.p_scalars, "a", encoding="utf-8")
        self._f_text = open(self.p_text, "a", encoding="utf-8")
        self._f_log = open(self.dir / "logs" / "stdout.log", "a", encoding="utf-8", errors="replace")
        self._tees = []
        if tee:
            for name in ("stdout", "stderr"):
                orig = getattr(sys, name)
                t = _Tee(orig, self._f_log, self._lock)
                setattr(sys, name, t)
                self._tees.append((name, orig, t))
        self.tb = SummaryWriter(log_dir=str(self.dir / "tb"), purge_step=self.step + 1 if resume else None,
                                max_queue=1000, flush_secs=60)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if student_meta is None:
            student_meta = self._find_student_meta(cfg)
        conf = dict(run_id=self.run_id, created_utc=_now_iso(), argv=sys.argv, cwd=os.getcwd(), resume=resume,
                    config=cfg, student_meta=student_meta)
        _atomic_json(self.dir / ("config.json" if not (self.dir / "config.json").exists() else f"config.{stamp}.json"), conf)
        if capture:
            try:
                capture_env(self.dir / "env" if not restart else self.dir / "env" / f"restart-{stamp}")
            except Exception as e:
                self.event("env_capture_error", error=f"{type(e).__name__}: {e}")
        self.text("config", "```json\n" + json.dumps(conf, indent=2, ensure_ascii=False, default=str) + "\n```", self.step)
        self.event("logger_start", run_id=self.run_id, restart=restart, resume=resume, hf_repo=hf_repo)

    # ------------------------------------------------------------------------------------------ helpers

    @staticmethod
    def _find_student_meta(cfg: dict) -> dict | None:
        try:
            p = Path(cfg.get("student", "")) / "student_meta.json"
            if not p.is_absolute():
                p = ROOT / p
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
        except Exception:
            return None

    def _load_steps(self, path: Path, max_step: int | None):
        t = pq.read_table(path)
        if max_step is not None:
            t = t.filter(pc.less_equal(t.column("step"), max_step))
        self._step_col = array("q", t.column("step").to_pylist())
        self._wall_col = array("d", t.column("wall").to_numpy(zero_copy_only=False).astype(np.float64))
        self._el_col = array("d", t.column("elapsed_s").to_numpy(zero_copy_only=False).astype(np.float64))
        for name in t.column_names:
            if name not in ("step", "wall", "elapsed_s"):
                self._steps[name] = array("d", t.column(name).to_numpy(zero_copy_only=False).astype(np.float64))
        self._n_steps = t.num_rows

    def elapsed(self) -> float:
        return self._elapsed0 + time.monotonic() - self._t0

    def state_dict(self) -> dict:
        """What the trainer keeps in its full state so a resumed run continues the same clock and step axis.

        Also writes out everything buffered up to now (steps.parquet, train_utts and hist parts), so a resume from
        this state loses no rows. Cheap enough (~0.1-0.3 s) for the 30-minute full-state saves."""
        self.flush_train_utts()
        with self._lock:
            hist, self._hist_rows = self._hist_rows, []
        if hist:
            self._write_part("hist", hist)
        self._write_steps(self._steps_table())
        return dict(elapsed_s=self.elapsed(), step=self.step, run_id=self.run_id)

    # --------------------------------------------------------------------------------------------- scalars

    def scalar(self, tag: str, value, step: int):
        v = _num(value)
        wall = time.time()
        row = dict(step=int(step), wall=round(wall, 3), elapsed_s=round(self.elapsed(), 3), tag=tag,
                   value=v if math.isfinite(v) else None)
        if not math.isfinite(v):
            row["nf"] = "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
        with self._lock:
            self._f_scalars.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.tb.add_scalar(tag, v, int(step), walltime=wall)

    def scalars(self, values: dict, step: int):
        for k, v in values.items():
            self.scalar(k, v, step)

    def step_row(self, values: dict, step: int, also_scalars: bool = True):
        """The wide per-optimizer-step row (numeric values only). By default every value is also logged as a
        scalar, so one call per step covers steps.parquet, scalars.jsonl and TensorBoard."""
        vals = {k: _num(v) for k, v in values.items()}
        with self._lock:
            n = self._n_steps
            for k in vals:
                if k not in self._steps:
                    self._steps[k] = array("d", [math.nan]) * n
            for k, col in self._steps.items():
                col.append(vals.get(k, math.nan))
            self._step_col.append(int(step))
            self._wall_col.append(time.time())
            self._el_col.append(self.elapsed())
            self._n_steps += 1
        if also_scalars:
            self.scalars(vals, step)
        self.step = int(step)
        if self._utt_rows and step - self._last_utt_flush >= self.train_utts_flush_steps:
            self.flush_train_utts()
        with self._lock:
            self._f_scalars.flush()  # one flush per step: a crash loses at most the current step's lines

    def _write_steps(self, table: pa.Table):
        # rows only ever grow within a logger's life: a slower background sync must not overwrite a newer write
        with self._part_lock:
            if table.num_rows >= self._steps_written:
                _atomic_parquet(table, self.dir / "metrics" / "steps.parquet")
                self._steps_written = table.num_rows

    def _steps_table(self) -> pa.Table:
        # np.array copies: arrow would otherwise wrap the array('d') buffers zero-copy, and a buffer that is
        # exported cannot grow (the next step_row would raise BufferError)
        with self._lock:
            cols = {"step": np.array(self._step_col, dtype=np.int64), "wall": np.array(self._wall_col, dtype=np.float64),
                    "elapsed_s": np.array(self._el_col, dtype=np.float64)}
            for k, col in self._steps.items():
                cols[k] = np.array(col, dtype=np.float64)
        return pa.table(cols)

    # ------------------------------------------------------------------------------------------ histograms

    def hist(self, tag: str, values, step: int, bins: int = HIST_BINS):
        """Distribution summary of a tensor/array: quantiles (on <= 1M strided samples), moments and bin counts over
        all finite values, the non-finite count; to hist parquet and TensorBoard (as a raw histogram)."""
        import torch

        t = values.detach() if isinstance(values, torch.Tensor) else torch.as_tensor(np.asarray(values))
        t = t.flatten().float()
        finite = torch.isfinite(t)
        n_bad = int((~finite).sum())
        if n_bad:
            t = t[finite]
        n = t.numel()
        row = dict(step=int(step), tag=tag, wall=time.time(), n=n, n_nonfinite=n_bad)
        if n == 0:
            row.update({k: math.nan for k in ("min", "max", "mean", "std", *[f"p{round(q * 100)}" for q in QUANTILES])})
            row.update(counts="[]", edges="[]")
        else:
            mn, mx = float(t.min()), float(t.max())
            td = t.double()
            s, s2 = float(td.sum()), float((td * td).sum())
            sample = t[:: max(1, math.ceil(n / 1_000_000))]
            qs = torch.quantile(sample, torch.tensor(QUANTILES, device=t.device, dtype=sample.dtype)).tolist()
            if mx > mn:
                counts = torch.histc(t, bins=bins, min=mn, max=mx).long().tolist()
                edges = np.linspace(mn, mx, bins + 1).tolist()
            else:
                counts, edges = [n], [mn, mx]
            row.update(min=mn, max=mx, mean=s / n, std=float(td.std(correction=0)) if n > 1 else 0.0,
                       counts=json.dumps(counts), edges=json.dumps(edges))
            row.update({f"p{round(q * 100)}": v for q, v in zip(QUANTILES, qs)})
            with self._lock:
                self.tb.add_histogram_raw(tag, min=mn, max=mx, num=n, sum=s, sum_squares=s2,
                                          bucket_limits=edges[1:], bucket_counts=counts, global_step=int(step))
        with self._lock:
            self._hist_rows.append(row)

    # ------------------------------------------------------------------------------------- text / tables

    def text(self, tag: str, s: str, step: int):
        row = dict(step=int(step), wall=round(time.time(), 3), elapsed_s=round(self.elapsed(), 3), tag=tag, text=s)
        with self._lock:
            self._f_text.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._f_text.flush()
            self.tb.add_text(tag, s, int(step))

    def eval_dir(self, step: int) -> Path:
        d = self.dir / "evals" / f"step_{int(step)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def table(self, name: str, df, step: int):
        """evals/step_<N>/<name>.parquet (e.g. tf_eval_jsut, greedy_eval_cv8, probe) from a DataFrame or arrow table."""
        t = df if isinstance(df, pa.Table) else pa.Table.from_pandas(df, preserve_index=False)
        _atomic_parquet(t, self.eval_dir(step) / f"{name}.parquet")

    def eval_json(self, name: str, obj: dict, step: int):
        """evals/step_<N>/<name>.json (summary.json, verdict.json ...)."""
        _atomic_json(self.eval_dir(step) / f"{name}.json", obj)

    def samples(self, step: int, rows: list[dict], tag: str = "samples"):
        """samples/step_<N>.jsonl plus a TensorBoard markdown table (ref / teacher / student)."""
        with open(self.dir / "samples" / f"step_{int(step)}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        def cell(x):
            return str(x).replace("|", "\\|").replace("\n", " ")

        md = ["| id | cer_ref | cer_teacher | ref | teacher | student |", "|---|---|---|---|---|---|"]
        for r in rows:
            md.append("| " + " | ".join(cell(r.get(k, "")) for k in ("id", "cer_ref", "cer_teacher", "ref",
                                                                    "teacher_hyp", "hyp")) + " |")
        self.text(tag, "\n".join(md), step)

    def write_summary(self, summary: dict):
        _atomic_json(self.dir / "summary.json", summary)

    # ------------------------------------------------------------------------------------------------ events

    def event(self, kind: str, **fields):
        """events.jsonl line, fsynced (these are the lines you want after a crash); also TensorBoard text."""
        row = dict(wall=round(time.time(), 3), time=_now_iso(), elapsed_s=round(self.elapsed(), 3), step=self.step,
                   kind=kind, **fields)
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.p_events, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            if not self._closed:
                self.tb.add_text(f"events/{kind}", "```\n" + line + "\n```", self.step)
        brief = {k: v for k, v in fields.items() if k != "traceback"}
        print(f"[event] {kind} {json.dumps(brief, ensure_ascii=False, default=str)[:300]}", flush=True)

    def exception(self, exc: BaseException | None = None, **fields):
        """Event with type, message and full traceback of `exc` (default: the exception being handled)."""
        exc = exc or sys.exc_info()[1]
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else traceback.format_exc()
        self.event("exception", type=type(exc).__name__ if exc else None, message=str(exc), traceback=tb, **fields)

    # --------------------------------------------------------------------------------------- train utts

    def train_utts(self, records: list[dict]):
        """Per-utterance training records (step, epoch, id, source, duration, n_tok, kl, ce, top1_acc,
        masked_frac, agree, + any extra keys); written as a parquet part every train_utts_flush_steps steps."""
        with self._lock:
            self._utt_rows.extend(records)

    def flush_train_utts(self):
        with self._lock:
            rows, self._utt_rows = self._utt_rows, []
            self._last_utt_flush = self.step
        if rows:
            self._write_part("train_utts", rows)

    def _write_part(self, kind: str, rows: list[dict]):
        if kind == "train_utts":
            extra = sorted({k for r in rows for k in r} - set(TRAIN_UTT_SCHEMA.names))
            t = pa.Table.from_pylist(rows, schema=TRAIN_UTT_SCHEMA)
            for k in extra:
                t = t.append_column(k, pa.array([r.get(k) for r in rows]))
        else:
            t = pa.Table.from_pylist(rows)
        d = self.dir / "metrics" / kind
        with self._part_lock:
            _atomic_parquet(t, d / f"part-{_next_part(d):05d}.parquet")

    # --------------------------------------------------------------------------------------------- sync

    def sync(self, force: bool = False, wait: bool | None = None) -> bool:
        """Rewrite the parquet mirrors and upload the run dir, in a background thread. Non-forced calls are no-ops
        until sync_every_min has passed (or while a sync is still running). force=True always runs and by default
        waits for completion (end of run / exception path)."""
        wait = force if wait is None else wait
        if not force and time.monotonic() - self._last_sync < self.sync_every_s:
            return False
        if self._sync_thread is not None and self._sync_thread.is_alive():
            if not force:
                return False
            self._sync_thread.join()
        self._last_sync = time.monotonic()
        with self._lock:
            self._f_scalars.flush()
            self._f_text.flush()
            self._f_log.flush()
            self.tb.flush()
            hist, self._hist_rows = self._hist_rows, []
            utts, self._utt_rows = self._utt_rows, []
            self._last_utt_flush = self.step
            scalars_bytes = self.p_scalars.stat().st_size  # flushed above: whole lines only
            # every append-only file, flushed above and written only under this lock: whole lines / TB records
            appended = [self.p_scalars, self.p_text, self.p_events, self.dir / "logs" / "stdout.log",
                        *(p for p in (self.dir / "tb").iterdir() if p.is_file())]
            sizes = {p.relative_to(self.dir).as_posix(): p.stat().st_size for p in appended if p.exists()}
        snap = dict(steps=self._steps_table(), hist=hist, utts=utts, step=self.step, scalars_bytes=scalars_bytes,
                    sizes=sizes)
        th = threading.Thread(target=self._sync_worker, args=(snap,), name="runlog-sync", daemon=True)
        self._sync_thread = th
        th.start()
        if wait:
            th.join()
        return True

    def _sync_worker(self, snap: dict):
        try:
            if snap["hist"]:
                self._write_part("hist", snap["hist"])
            if snap["utts"]:
                self._write_part("train_utts", snap["utts"])
            _atomic_parquet(read_scalars_jsonl(self.p_scalars, limit=snap["scalars_bytes"]),
                            self.dir / "metrics" / "scalars.parquet")
            self._write_steps(snap["steps"])
        except Exception as e:
            self.event("sync_error", stage="local", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        if self.hf_repo:
            with tempfile.TemporaryDirectory(prefix=f"kitsune-sync-{self.run_id}-") as stage:
                try:
                    self._stage(Path(stage), snap["sizes"])
                except Exception as e:
                    self.event("sync_error", stage="snapshot", error=f"{type(e).__name__}: {e}",
                               traceback=traceback.format_exc())
                    return
                self._upload(snap["step"], Path(stage))

    def _stage(self, dest: Path, sizes: dict[str, int]):
        """Copy of the run dir to upload (checkpoints/ and *.tmp left out): the append-only files cut at the lengths
        sync() noted under the lock, every other file as it is when copied (the parquet mirrors this worker just
        wrote; files written atomically or once)."""
        for src in list(self.dir.rglob("*")):
            rel = src.relative_to(self.dir).as_posix()
            if rel.split("/", 1)[0] == "checkpoints" or rel.endswith(".tmp") or not src.is_file():
                continue
            _copy_prefix(src, dest / rel, sizes.get(rel))

    def _upload(self, step: int, folder: Path):
        waits = (0.0, *self.upload_retries)
        for attempt, w in enumerate(waits):
            if w:
                time.sleep(w)
            t0 = time.time()
            try:
                api = self._api
                if api is None:
                    from huggingface_hub import HfApi

                    api = self._api = HfApi()
                if not self._repo_ready:
                    api.create_repo(self.hf_repo, repo_type="model", private=((self.cfg or {}).get("hf") or {}).get("private", True),
                                    exist_ok=True)
                    self._repo_ready = True
                api.upload_folder(repo_id=self.hf_repo, repo_type="model", folder_path=str(folder),
                                  path_in_repo=f"runs/{self.run_id}", ignore_patterns=SYNC_IGNORE,
                                  commit_message=f"sync {self.run_id} step {step}")
                self.event("sync_ok", repo=self.hf_repo, attempt=attempt, upload_s=round(time.time() - t0, 1))
                return True
            except Exception as e:
                self.event("sync_error", stage="upload", attempt=attempt, repo=self.hf_repo,
                           error=f"{type(e).__name__}: {e}"[:2000])
        self.event("sync_failed", repo=self.hf_repo, attempts=len(waits))
        return False

    def wait_sync(self):
        if self._sync_thread is not None:
            self._sync_thread.join()

    # ------------------------------------------------------------------------------------------- close

    def close(self, summary: dict | None = None):
        """Write summary.json (if given), flush everything, run a final forced sync, restore stdout/stderr."""
        if self._closed:
            return
        if summary is not None:
            self.write_summary(summary)
        self.flush_train_utts()
        self.event("logger_close", elapsed_s_total=round(self.elapsed(), 1))
        self.sync(force=True, wait=True)
        with self._lock:
            self._closed = True
            self.tb.close()
            self._f_scalars.close()
            self._f_text.close()
            for name, orig, t in reversed(self._tees):
                if getattr(sys, name) is t:
                    setattr(sys, name, orig)
            self._f_log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.exception(exc)
        self.close()
        return False
