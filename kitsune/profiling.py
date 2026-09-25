"""A torch.profiler record of ~20 real optimizer steps of the smoke phase, and its compact summary (perf.profile_smoke).

Why: the first A100 run spent ~45 % of each 1.22 s step on a fixed cost per micro-batch (a regression over 9,636
steps; MFU 18 % where the plan assumed 30-45 %). The suspected cause - eager-mode host dispatch: ~13,200 small ops per
micro-batch, device syncs such as a GPU tensor's .tolist() and a blocking torch.tensor(..., device=cuda), a
per-utterance dither loop - was inferred, never profiled. Before any code change for speed (removing syncs,
torch.compile), the run records the evidence itself, in the smoke phase, where it costs about a minute:
  window   after the smoke's own warm-up (the steps its throughput check leaves out: kernel autotuning, loader
           start-up), one profiler warm-up step, then up to PROFILE_STEPS recorded steps, all inside smoke.steps
           (profile_window). They are the run's own training steps, so the profile shows the real step (data wait,
           forward/backward per micro-batch, clip, AdamW, L2-SP, the logging's host transfers); the profiler only
           observes, so the weights, losses and RNG streams are those of a run without it (tests/test_smoke_profile.py
           on CPU). On the steps and epochs clocks the run is bitwise the same; on the wall clock the minute it costs
           (the steps' overhead and the trace processing) comes out of the loop time, as an eval's or a checkpoint's
           does. The profiled steps are left out of the smoke throughput check
  cycles   the recorded steps in profiler cycles (TRACE_STEPS first, then up to CYCLE_STEPS each): each cycle's events
           are aggregated when it ends and then dropped, which bounds host memory and CUPTI's buffers
  outputs  under runs/<run_id>/smoke/profile/ (uploaded with the logs, verified by vast/finish.py):
             summary.json          what to read (summarise): per step - iteration wall time and its data-wait share,
                                   GPU kernel time and its share, kernel launches (the CUDA runtime's launch calls)
                                   and kernels run, per micro-batch too, aten ops, device syncs (cudaStreamSynchronize /
                                   cudaDeviceSynchronize / cudaEventSynchronize: count and host time; the host reads of
                                   a device scalar, aten::_local_scalar_dense / aten::item, that cause them), the host
                                   time outside any recorded op (Python, the loader); the top ops by self device time
                                   (CPU-side ops, their kernels attributed) and by self CPU time, the top kernels
             trace_steps_<a>-<b>.json.gz   the first cycle's raw Chrome trace (TensorBoard's profile plugin, Perfetto,
                                   chrome://tracing), gzip, kept only up to TRACE_MAX_MB compressed (else summary.json's
                                   trace says why it was dropped)
           and a `smoke_profile` event with the headline numbers. On CPU only CPU activities are recorded (the tests); a
           CPU run's config leaves it off ("auto": on under CUDA only)."""
import gzip
import json
import os
import shutil
import time
import warnings
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Callable

import torch

PROFILE_STEPS = 20  # recorded optimizer steps (the report's "~20"), after one profiler warm-up step
TRACE_STEPS = 2  # the first cycle, whose raw trace is kept: ~150k events per step of the real student on the A100
CYCLE_STEPS = 6  # later cycles: aggregated at their end, then dropped
TRACE_MAX_MB = 32.0  # compressed; a bigger raw trace is dropped (summary.json says so), the summary is kept
TOP_N = 25
NAME_MAX = 160  # kernel names carry whole template signatures
LAUNCH_APIS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx",
               "cudaLaunchCooperativeKernel")
SYNC_APIS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
SCALAR_READS = ("aten::_local_scalar_dense", "aten::item")  # a device scalar read on the host: a sync each
MEMCPY_APIS = ("cudaMemcpyAsync", "cudaMemcpy")


@contextmanager
def _per_cycle():
    """torch warns at every new cycle that it clears the last one's events: aggregating each cycle and then dropping
    its events is the point here (acc_events would keep every event of the ~20 steps in host memory)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Profiler clears events at the end of each cycle")
        yield


def profile_window(smoke_steps: int) -> tuple[int, int] | None:
    """(first, n): the profiler's warm-up step is optimizer step `first` and it records the n steps after it - after
    the first smoke_steps // 10, which the throughput check leaves out (kernel autotuning, loader start-up), and
    leaving the last quarter of the smoke steps (at least one) unprofiled for that check, which skips the profiled
    ones. smoke.steps 100: first 21, steps 22-41 recorded. None when the smoke phase is too short to record a step."""
    first = max(2, int(smoke_steps) // 5 + 1)
    n = min(PROFILE_STEPS, int(smoke_steps) - first - max(1, int(smoke_steps) // 4))
    return (first, n) if n >= 1 else None


def cycle_ends(n: int) -> list[int]:
    """The profiler step numbers (1..n) that end a cycle: TRACE_STEPS first, then CYCLE_STEPS each."""
    ends, k = [], 0
    while k < n:
        k = min(n, k + (TRACE_STEPS if not ends else CYCLE_STEPS))
        ends.append(k)
    return ends


def _rows(key_averages) -> list[dict]:
    """torch.profiler key averages -> plain rows: name, count, self CPU and self device microseconds, on_device (the
    event ran on the device: a kernel, memcpy or memset, not a CPU-side op or runtime call) and annotation (a
    record_function range such as ProfilerStep#N, which Kineto also mirrors on the device timeline: never a kernel)."""
    from torch.autograd import DeviceType

    out = []
    for e in key_averages:
        out.append(dict(name=str(e.key), count=int(e.count), self_cpu_us=float(e.self_cpu_time_total),
                        self_device_us=float(getattr(e, "self_device_time_total", 0.0) or 0.0),
                        on_device=e.device_type != DeviceType.CPU,
                        annotation=bool(getattr(e, "is_user_annotation", False))))
    return out


def _add(totals: dict, rows: list[dict]):
    for r in rows:
        key = (r["name"], r["on_device"], r.get("annotation", False))
        t = totals.setdefault(key, dict(name=r["name"], count=0, self_cpu_us=0.0, self_device_us=0.0,
                                        on_device=r["on_device"], annotation=r.get("annotation", False)))
        t["count"] += r["count"]
        t["self_cpu_us"] += r["self_cpu_us"]
        t["self_device_us"] += r["self_device_us"]


def summarise(rows: list[dict], steps: list[dict], device: str, top_n: int = TOP_N) -> dict:
    """The profile's summary from its aggregated rows (_rows, summed over the cycles) and the loop's timings of the
    recorded steps (steps: dict(step, iter_s, wait_s, step_s, n_micro) each; iter_s from the step's start, data wait
    included, to the end of its logging). Per-step numbers divide by the recorded steps; shares divide by their summed
    iteration time. A pure function of its inputs (the tests feed it CUDA-like rows)."""
    n = len(steps)
    per = float(max(n, 1))
    iter_s = sum(float(s["iter_s"]) for s in steps)
    wait_s = sum(float(s.get("wait_s") or 0.0) for s in steps)
    step_s = sum(float(s["step_s"]) for s in steps if s.get("step_s") is not None)
    micro = sum(int(s.get("n_micro") or 0) for s in steps)
    def is_range(r):  # a record_function range (the profiler's step marks), never an op or a kernel
        return bool(r.get("annotation")) or r["name"].startswith("ProfilerStep")

    marks = [r for r in rows if not r["on_device"] and r["name"].startswith("ProfilerStep")]
    ops = [r for r in rows if not r["on_device"] and not is_range(r)]
    dev = [r for r in rows if r["on_device"] and not is_range(r)]

    def share(us):
        return us / 1e6 / iter_s if iter_s else None

    def entry(r, key):
        return dict(name=r["name"][:NAME_MAX], count=r["count"], count_per_step=r["count"] / per,
                    self_cpu_ms_per_step=r["self_cpu_us"] / 1e3 / per,
                    self_device_ms_per_step=r["self_device_us"] / 1e3 / per, share_of_step=share(r[key]))

    def top(pool, key):
        return [entry(r, key) for r in sorted(pool, key=lambda r: -r[key])[:top_n] if r[key] > 0]

    def named(names):
        return {r["name"]: dict(count=r["count"], count_per_step=r["count"] / per,
                                self_cpu_ms_per_step=r["self_cpu_us"] / 1e3 / per,
                                share_of_step=share(r["self_cpu_us"]))
                for r in ops if r["name"] in names}

    launches = sum(r["count"] for r in ops if r["name"] in LAUNCH_APIS)
    kernels = sum(r["count"] for r in dev)
    syncs = named(SYNC_APIS)
    sync_us = sum(r["self_cpu_us"] for r in ops if r["name"] in SYNC_APIS)
    device_us = sum(r["self_device_us"] for r in dev)
    return dict(
        device=device, steps=[int(s["step"]) for s in steps], n_steps=n,
        micro_batches_per_step=micro / per if n else None,
        time=dict(iter_s_per_step=iter_s / per if n else None, step_s_per_step=step_s / per if n else None,
                  data_wait_s_per_step=wait_s / per if n else None,
                  data_wait_share=wait_s / iter_s if iter_s else None,
                  # the loop's own step timer ends at its device sync; the rest of the iteration is its logging
                  logging_share=(iter_s - step_s) / iter_s if iter_s and step_s else None,
                  outside_ops_cpu_s_per_step=sum(r["self_cpu_us"] for r in marks) / 1e6 / per if marks else None),
        gpu=dict(kernel_ms_per_step=device_us / 1e3 / per, kernel_time_share=share(device_us) if dev else None,
                 kernels_per_step=kernels / per, kernels_per_micro_batch=kernels / micro if micro else None,
                 launches_per_step=launches / per, launches_per_micro_batch=launches / micro if micro else None),
        host=dict(aten_ops_per_step=sum(r["count"] for r in ops if r["name"].startswith("aten::")) / per,
                  aten_ops_per_micro_batch=(sum(r["count"] for r in ops if r["name"].startswith("aten::")) / micro
                                            if micro else None),
                  cpu_op_ms_per_step=sum(r["self_cpu_us"] for r in ops) / 1e3 / per),
        syncs=dict(apis=syncs, sync_ms_per_step=sync_us / 1e3 / per, sync_share=share(sync_us),
                   syncs_per_step=sum(d["count"] for d in syncs.values()) / per,
                   scalar_reads=named(SCALAR_READS), memcpy=named(MEMCPY_APIS),
                   note="the loop's step timer syncs the device once per step itself (torch.cuda.synchronize)"),
        top_ops_by_self_device_time=top(ops, "self_device_us"),
        top_ops_by_self_cpu_time=top(ops, "self_cpu_us"),
        top_kernels_by_device_time=top(dev, "self_device_us"))


class SmokeProfiler:
    """Drives torch.profiler over the recorded window of the training loop (scripts/04_distill.loop): begin(step)
    before a step's data fetch, end(...) after its logging - for a skipped step too, which the profiler also saw. The
    window counts profiler steps (begin/end pairs), not optimizer step numbers, which a skipped step repeats. When the
    last cycle ends, the outputs are written and `emit("smoke_profile", ...)` logs the headline; close() ends a window
    the loop left early (its end or an error), with what was recorded."""

    def __init__(self, out_dir: Path, device: torch.device, first: int, n: int,
                 emit: Callable[..., None] | None = None, trace_max_mb: float = TRACE_MAX_MB):
        from torch.profiler import ProfilerActivity

        self.out_dir, self.device, self.first, self.n = Path(out_dir), torch.device(device), int(first), int(n)
        self.emit = emit or (lambda kind, **kw: None)
        self.trace_max_mb = float(trace_max_mb)
        self.activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.device.type == "cuda" else [])
        self.ends = cycle_ends(self.n)
        self.prof = None
        self.k = 0  # profiler steps done: 0 = the warm-up step under way
        self.done = False
        self.totals: dict = {}
        self.steps: list[dict] = []
        self.cycles: list[dict] = []
        self.trace: dict | None = None
        self.t_step = 0.0
        self.t_start = 0.0

    def _schedule(self, i: int):
        from torch.profiler import ProfilerAction

        if i == 0:
            return ProfilerAction.WARMUP
        if i > self.n:
            return ProfilerAction.NONE
        return ProfilerAction.RECORD_AND_SAVE if i in self.ends else ProfilerAction.RECORD

    @property
    def running(self) -> bool:
        return self.prof is not None and not self.done

    def begin(self, step: int) -> bool:
        """Before optimizer step `step` (its data fetch included): starts the profiler at the window's first step.
        Returns whether this step runs under the profiler (the loop leaves it out of the smoke throughput). Never
        raises: a profiler that cannot start (no CUPTI on the box) costs the profile, not the run."""
        if self.prof is None and not self.done and step >= self.first:
            try:
                self.prof = torch.profiler.profile(activities=self.activities, schedule=self._schedule,
                                                   on_trace_ready=self._cycle_ready)
                self.prof.start()
            except Exception as e:  # noqa: BLE001
                self._fail("start", step, e)
                return False
            self.t_start = time.perf_counter()
            self.emit("smoke_profile_start", at_step=int(step), recorded_steps=self.n, cycles=self.ends,
                      activities=[a.name for a in self.activities])
        if self.running:
            self.t_step = time.perf_counter()
        return self.running

    def end(self, step: int, wait_s: float, step_s: float | None, n_micro: int) -> dict | None:
        """After a step that began under the profiler: its timings (step_s None: skipped), then the profiler's step.
        Returns the summary once the window is over, else None. Never raises (as begin)."""
        if not self.running:
            return None
        iter_s = time.perf_counter() - self.t_step
        if self.k >= 1:  # the warm-up step is not recorded
            self.steps.append(dict(step=int(step), iter_s=iter_s, wait_s=float(wait_s), step_s=step_s,
                                   n_micro=int(n_micro)))
        self.k += 1
        try:
            with _per_cycle():  # a cycle's end: stop, _cycle_ready (aggregate, the first cycle's trace), start the next
                self.prof.step()
            if self.k > self.n:
                return self._finish(complete=True)
        except Exception as e:  # noqa: BLE001
            self._fail("step", step, e)
        return None

    def close(self) -> dict | None:
        """The loop ended or failed inside the window: stop the profiler and write what the finished steps recorded
        (a partial cycle included). Never raises: a profiler failure must not mask the loop's own."""
        if not self.running:
            return None
        try:
            return self._finish(complete=False)
        except Exception as e:  # noqa: BLE001
            self._fail("close", None, e)
            return None

    def _fail(self, where: str, step: int | None, e: Exception):
        """Give up on the profile (never on the run): stop the profiler if it runs, say why in a `smoke_profile`
        event."""
        running, self.done = self.prof is not None and not self.done, True
        if running:
            with suppress(Exception):
                self.prof.stop()
        self.emit("smoke_profile", complete=False, failed_in=where, at_step=step,
                  error=f"{type(e).__name__}: {e}"[:500])

    def _cycle_ready(self, prof):
        t0 = time.perf_counter()
        _add(self.totals, _rows(prof.key_averages()))
        cyc = dict(steps=len(self.steps) - sum(c["steps"] for c in self.cycles), process_s=None)
        if self.trace is None:
            self.trace = self._export(prof, [s["step"] for s in self.steps])
        cyc["process_s"] = round(time.perf_counter() - t0, 2)
        self.cycles.append(cyc)

    def _export(self, prof, steps: list[int]) -> dict:
        """The cycle's Chrome trace, gzip-compressed, kept only up to trace_max_mb."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = f"trace_steps_{steps[0]}-{steps[-1]}.json.gz" if steps else "trace.json.gz"
        raw, gz = self.out_dir / f"{name}.raw.tmp", self.out_dir / f"{name}.tmp"
        try:
            prof.export_chrome_trace(str(raw))
            with open(raw, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            raw_mb, mb = raw.stat().st_size / 2**20, gz.stat().st_size / 2**20
            if mb > self.trace_max_mb:
                gz.unlink()
                return dict(file=None, steps=steps, mb=round(mb, 1), raw_mb=round(raw_mb, 1),
                            dropped=f"{mb:.1f} MB compressed > the {self.trace_max_mb:g} MB cap")
            os.replace(gz, self.out_dir / name)
            return dict(file=name, steps=steps, mb=round(mb, 2), raw_mb=round(raw_mb, 1))
        except Exception as e:  # noqa: BLE001 - a trace export failure costs the trace, not the summary
            return dict(file=None, steps=steps, error=f"{type(e).__name__}: {e}"[:300])
        finally:
            for p in (raw, gz):
                p.unlink(missing_ok=True)

    def _finish(self, complete: bool) -> dict:
        self.prof.stop()  # a partial cycle (close) ends here too: its rows and, if it is the first, its trace
        self.done = True
        summary = summarise(list(self.totals.values()), self.steps, self.device.type)
        summary.update(complete=complete, activities=[a.name for a in self.activities], cycles=self.cycles,
                       trace=self.trace, wall_s=round(time.perf_counter() - self.t_start, 1),
                       torch=torch.__version__)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.out_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.out_dir / "summary.json")
        t, g, s = summary["time"], summary["gpu"], summary["syncs"]
        self.emit("smoke_profile", complete=complete, steps=summary["steps"], wall_s=summary["wall_s"],
                  iter_s_per_step=t["iter_s_per_step"], data_wait_share=t["data_wait_share"],
                  logging_share=t["logging_share"], gpu_kernel_time_share=g["kernel_time_share"],
                  launches_per_step=g["launches_per_step"], launches_per_micro_batch=g["launches_per_micro_batch"],
                  aten_ops_per_micro_batch=summary["host"]["aten_ops_per_micro_batch"],
                  syncs_per_step=s["syncs_per_step"], sync_share=s["sync_share"],
                  top_device=[e["name"][:60] for e in summary["top_ops_by_self_device_time"][:5]],
                  top_cpu=[e["name"][:60] for e in summary["top_ops_by_self_cpu_time"][:5]],
                  trace=(self.trace or {}).get("file"))
        return summary
