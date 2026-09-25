"""The smoke-phase profiler (perf.profile_smoke, kitsune.profiling): ~20 real training steps of the smoke phase under
torch.profiler, summarised into runs/<run_id>/smoke/profile/summary.json with the first cycle's raw trace next to it.

- the window: after the smoke's warm-up steps, inside smoke.steps; cycles of TRACE_STEPS then CYCLE_STEPS
- summarise: per-step and per-micro-batch numbers from the profiler's rows and the loop's timings - kernel launches,
  kernels, device syncs and the host reads that cause them, data-wait and kernel-time shares, the top ops by self
  device and self CPU time (fed CUDA-like rows: the tests never touch the GPU)
- SmokeProfiler on CPU: the outputs, the trace's size cap, a skipped step, a window the loop leaves early
- a whole trainer run with it on vs off: the same steps, losses, gradients and weights (the profiler only observes),
  the outputs in the run dir and the `smoke_profile` events; "auto" leaves it off on CPU

CPU only (the laptop GPU runs other jobs), a tiny random student and a synthetic corpus in the real on-disk formats."""
import gc
import gzip
import json
import os
import sys
import weakref
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import load_script, make_fake_corpus, make_fake_selection  # noqa: E402
from kitsune import profiling as P  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]


def events(run: Path, kind: str | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


# ------------------------------------------------------------------------------------------------- unit level


def test_the_window_and_its_cycles():
    """smoke.steps 100 (the real runs): the profiler's warm-up at step 21, after the 10 steps the throughput check
    leaves out, then steps 22-41 recorded, and the last quarter unprofiled for that check; a short smoke phase records
    what fits, none when nothing does."""
    assert P.profile_window(100) == (21, 20) and P.profile_window(20) == (5, 10)
    assert P.profile_window(6) == (2, 3) and P.profile_window(4) == (2, 1)
    assert P.profile_window(3) is None and P.profile_window(0) is None
    assert P.cycle_ends(20) == [2, 8, 14, 20] and P.cycle_ends(4) == [2, 4] and P.cycle_ends(1) == [1]


def row(name, count, cpu_us=0.0, dev_us=0.0, on_device=False, annotation=False):
    return dict(name=name, count=count, self_cpu_us=cpu_us, self_device_us=dev_us, on_device=on_device,
                annotation=annotation)


def test_summarise_counts_launches_syncs_and_shares():
    """Two recorded steps of 1.0 s each (0.1 s data wait, 0.8 s in the step timer), 4 micro-batches each, on CUDA-like
    rows: kernel launches and kernels per step and per micro-batch, the syncs and the host scalar reads, kernel time
    and data-wait shares; the profiler's own step marks (and their device-side mirror) are neither ops nor kernels."""
    rows = [row("aten::mm", 400, cpu_us=40_000, dev_us=600_000), row("aten::add", 600, cpu_us=30_000, dev_us=100_000),
            row("cudaLaunchKernel", 2000, cpu_us=200_000), row("cudaStreamSynchronize", 8, cpu_us=150_000),
            row("cudaDeviceSynchronize", 2, cpu_us=50_000), row("aten::_local_scalar_dense", 6, cpu_us=1_000),
            row("cudaMemcpyAsync", 10, cpu_us=5_000),
            row("ampere_sgemm_128x64_tn", 400, dev_us=600_000, on_device=True),
            row("vectorized_elementwise_kernel", 1600, dev_us=100_000, on_device=True),
            row("ProfilerStep#3", 1, cpu_us=100_000, annotation=True),
            row("ProfilerStep#4", 1, cpu_us=120_000, annotation=True),
            row("ProfilerStep#3", 1, dev_us=900_000, on_device=True, annotation=True)]
    steps = [dict(step=3, iter_s=1.0, wait_s=0.1, step_s=0.8, n_micro=4), dict(step=4, iter_s=1.0, wait_s=0.1,
                                                                              step_s=0.8, n_micro=4)]
    s = P.summarise(rows, steps, "cuda", top_n=3)
    assert s["steps"] == [3, 4] and s["n_steps"] == 2 and s["micro_batches_per_step"] == 4
    t, g, h, y = s["time"], s["gpu"], s["host"], s["syncs"]
    assert t["iter_s_per_step"] == 1.0 and t["data_wait_share"] == pytest.approx(0.1)
    assert t["logging_share"] == pytest.approx(0.2) and t["outside_ops_cpu_s_per_step"] == pytest.approx(0.11)
    assert g["kernels_per_step"] == 1000 and g["kernels_per_micro_batch"] == 250  # the step marks' mirror left out
    assert g["kernel_ms_per_step"] == pytest.approx(350) and g["kernel_time_share"] == pytest.approx(0.35)
    assert g["launches_per_step"] == 1000 and g["launches_per_micro_batch"] == 250
    assert h["aten_ops_per_step"] == 503 and h["aten_ops_per_micro_batch"] == pytest.approx(503 / 4)
    assert set(y["apis"]) == {"cudaStreamSynchronize", "cudaDeviceSynchronize"} and y["syncs_per_step"] == 5
    assert y["sync_ms_per_step"] == pytest.approx(100) and y["sync_share"] == pytest.approx(0.1)
    assert y["scalar_reads"]["aten::_local_scalar_dense"]["count_per_step"] == 3
    assert y["memcpy"]["cudaMemcpyAsync"]["count"] == 10
    assert [e["name"] for e in s["top_ops_by_self_device_time"]] == ["aten::mm", "aten::add"]
    assert [e["name"] for e in s["top_ops_by_self_cpu_time"]] == ["cudaLaunchKernel", "cudaStreamSynchronize",
                                                                  "cudaDeviceSynchronize"]
    top = s["top_kernels_by_device_time"][0]
    assert top["name"] == "ampere_sgemm_128x64_tn" and top["self_device_ms_per_step"] == pytest.approx(300)
    assert top["share_of_step"] == pytest.approx(0.3) and top["count_per_step"] == 200
    # CPU rows only (the tests' profile): no device numbers to speak of, nothing divides by zero
    cpu = P.summarise([row("aten::mm", 4, cpu_us=10.0)], steps[:1], "cpu")
    assert cpu["gpu"]["kernel_time_share"] is None and cpu["gpu"]["kernels_per_step"] == 0
    assert cpu["top_ops_by_self_device_time"] == [] and cpu["top_kernels_by_device_time"] == []
    assert P.summarise([], [], "cpu")["time"]["iter_s_per_step"] is None


def _steps(prof, first, last, skip=()):
    """A toy training loop on CPU driving the profiler as the trainer does."""
    w = torch.randn(64, 64, requires_grad=True)
    out = None
    for step in range(first, last + 1):
        profiled = prof.begin(step)
        x = torch.randn(32, 64)
        (x @ w).square().mean().backward()
        w.grad = None
        if profiled:
            out = prof.end(step, 0.001, None if step in skip else 0.002, 2)
    return out


def test_the_profiler_on_cpu_writes_its_summary_and_trace(tmp_path):
    got = []
    prof = P.SmokeProfiler(tmp_path / "profile", torch.device("cpu"), first=3, n=7,
                           emit=lambda kind, **kw: got.append(dict(kw, kind=kind)))
    assert _steps(prof, 1, 2) is None and not prof.running  # before the window
    assert _steps(prof, 3, 3) is None and prof.running  # the warm-up step
    torch_prof = weakref.ref(prof.prof)
    s = _steps(prof, 4, 12, skip=(6,))
    assert s is not None and s["complete"] and prof.done and not prof.running
    # the torch profiler, which holds its last cycle's events (GBs on the A100), is freed when the window ends, not
    # when the loop that holds this object does
    gc.collect()
    assert prof.prof is None and torch_prof() is None
    assert s["steps"] == [4, 5, 6, 7, 8, 9, 10] and s["n_steps"] == 7  # after the warm-up step 3; 6 was skipped
    assert s["activities"] == ["CPU"] and [c["steps"] for c in s["cycles"]] == [2, 5]
    assert s["micro_batches_per_step"] == 2 and s["host"]["aten_ops_per_step"] > 0
    assert any(e["name"] == "aten::mm" for e in s["top_ops_by_self_cpu_time"])
    on_disk = json.loads((tmp_path / "profile" / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["steps"] == s["steps"] and on_disk["trace"] == s["trace"]
    assert s["trace"]["file"] == "trace_steps_4-5.json.gz" and s["trace"]["steps"] == [4, 5]
    with gzip.open(tmp_path / "profile" / s["trace"]["file"], "rt", encoding="utf-8") as fh:
        trace = json.load(fh)
    names = {e.get("name") for e in trace["traceEvents"]}
    assert "ProfilerStep#1" in names and "ProfilerStep#2" in names and "ProfilerStep#3" not in names
    assert sorted(p.name for p in (tmp_path / "profile").iterdir()) == ["summary.json", "trace_steps_4-5.json.gz"]
    assert [e["kind"] for e in got] == ["smoke_profile_start", "smoke_profile"]
    assert got[1]["steps"] == s["steps"] and got[1]["trace"] == "trace_steps_4-5.json.gz" and got[1]["complete"]
    assert prof.begin(13) is False and prof.end(13, 0.0, 0.0, 1) is None and prof.close() is None  # done


def test_the_trace_cap_and_a_window_left_early(tmp_path):
    """A raw trace over the cap is dropped (the summary says so); a loop that ends inside the window writes what the
    finished steps recorded, marked incomplete."""
    prof = P.SmokeProfiler(tmp_path / "a", torch.device("cpu"), first=2, n=4, trace_max_mb=1e-6)
    s = _steps(prof, 2, 6)
    assert s["complete"] and s["trace"]["file"] is None and "cap" in s["trace"]["dropped"]
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["summary.json"]
    got = []
    prof = P.SmokeProfiler(tmp_path / "b", torch.device("cpu"), first=2, n=10,
                           emit=lambda kind, **kw: got.append(dict(kw, kind=kind)))
    assert _steps(prof, 2, 5) is None and prof.running
    s = prof.close()
    assert s is not None and not s["complete"] and s["steps"] == [3, 4, 5] and not prof.running
    assert prof.prof is None  # dropped with its events (_release)
    assert [c["steps"] for c in s["cycles"]] == [2, 1] and got[-1]["complete"] is False
    assert json.loads((tmp_path / "b" / "summary.json").read_text(encoding="utf-8"))["complete"] is False


def test_a_profiler_failure_costs_the_profile_not_the_run(tmp_path, monkeypatch):
    """A profiler that cannot start (no CUPTI on a box) or fails at a cycle's end never raises into the loop: the
    step runs unprofiled from then on, and a `smoke_profile` event says where it failed."""
    got = []

    def emit(kind, **kw):
        got.append(dict(kw, kind=kind))

    class Broken:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("CUPTI_ERROR_NOT_INITIALIZED")

    with monkeypatch.context() as mp:
        mp.setattr(torch.profiler, "profile", Broken)
        prof = P.SmokeProfiler(tmp_path / "a", torch.device("cpu"), first=2, n=3, emit=emit)
        assert _steps(prof, 1, 6) is None and prof.done and not prof.running and prof.prof is None
    assert got == [dict(kind="smoke_profile", complete=False, failed_in="start", at_step=2,
                        error="RuntimeError: CUPTI_ERROR_NOT_INITIALIZED")]
    got.clear()
    prof = P.SmokeProfiler(tmp_path / "b", torch.device("cpu"), first=2, n=4, emit=emit)
    monkeypatch.setattr(prof, "_cycle_ready", lambda p: (_ for _ in ()).throw(OSError("disk full")))
    assert _steps(prof, 1, 8) is None and prof.done and prof.close() is None and prof.prof is None
    assert [e["kind"] for e in got] == ["smoke_profile_start", "smoke_profile"]
    assert got[1]["failed_in"] == "step" and got[1]["at_step"] == 4 and "disk full" in got[1]["error"]


# ------------------------------------------------------------------------------------------------ whole runs


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + a tiny student saved exactly as 03 saves one, and a config for it with the
    smoke phase on (6 steps on the step clock)."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("smokeprofile")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (36, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=5)
    sel = make_fake_selection(fc, greedy_n=3, probe_n=4)
    torch.manual_seed(0)
    cfg = CohereAsrConfig().to_dict()
    enc = dict(cfg["encoder_config"], num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=2,
               num_key_value_heads=2, subsampling_conv_channels=8)
    cfg.update(encoder_config=enc, num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=2,
               num_key_value_heads=2, head_dim=32)
    tcfg = CohereAsrConfig.from_dict(cfg)
    tcfg._attn_implementation = tcfg.encoder_config._attn_implementation = "sdpa"
    teacher = CohereAsrForConditionalGeneration(tcfg).eval()
    student = S.build_student(teacher, S.StudentSpec(enc_layers=[0, 1], ffn_dim=128, dec_layers=[0]), None)
    sdir = root / "student"
    S.save_student(student, sdir, proc, dict(format=1, stage="complete", spec=dict(enc_layers=[0, 1], ffn_dim=128,
                                                                                    dec_layers=[0], tie_head=True)))
    base = {
        "student": str(sdir), "data_root": str(fc.data), "teacher_root": str(fc.teacher_out),
        "second_root": str(fc.second_out), "selection": str(sel), "cache_dir": str(root / "cache"),
        "runs_root": str(root / "runs"), "sources": ["src_a", "src_b"], "eval_sets": EVAL,
        "device": "cpu", "autocast": "none", "optim": {"lr": 3e-3},
        "schedule": {"warmup_steps": 2, "cooldown_frac": 0.3, "clock": "steps", "max_steps": 8},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_steps": 1000, "greedy_subset": 2, "batch_s": 20, "check_baselines": False},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 1000, "keep_local": 2, "full_after_smoke": False},
        "log": {"layer_stats_every": 4, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2, "capture_env": False},
        "hf": {"output_repo": None},
        "smoke": {"steps": 6, "min_audio_s_per_s": 0, "require_loss_decrease": False, "decode_per_set": 2},
    }
    return dict(root=root, base=base)


def run_with(env, name: str, profile) -> Path:
    m = load_script("04_distill")
    cfg = json.loads(json.dumps(env["base"]))
    cfg["run_name"] = name
    cfg["perf"]["profile_smoke"] = profile
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    assert m.main(["--config", str(path)]) == 0
    (run,) = list((env["root"] / "runs").glob(f"{name}-2*"))
    return run


def test_the_smoke_profile_changes_nothing_the_run_trains(env):
    """The same run with the profiler on (CPU activities: perf.profile_smoke true) and with "auto" (off on CPU): the
    same steps, losses, gradient norms, LR and final weights, bit for bit; the profile's summary and trace in
    smoke/profile/ and its events, only in the profiled run; the profiled steps are the window's."""
    from safetensors.torch import load_file

    on, off = run_with(env, "prof-on", True), run_with(env, "prof-auto", "auto")
    a, b = pd.read_parquet(on / "metrics" / "steps.parquet"), pd.read_parquet(off / "metrics" / "steps.parquet")
    assert a["step"].tolist() == b["step"].tolist() == list(range(1, 9))
    for col in ("loss/total", "loss/kl", "loss/ce", "opt/grad_norm", "opt/lr", "data/audio_s"):
        np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy(), err_msg=col)
    wa, wb = (load_file(next((r / "checkpoints" / "step_8").glob("*.safetensors"))) for r in (on, off))
    assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
    sa, sb = (json.loads((r / "summary.json").read_text(encoding="utf-8")) for r in (on, off))
    assert [r["heldout_kl"] for r in sa["history"]] == [r["heldout_kl"] for r in sb["history"]]

    first, n = P.profile_window(6)
    (start,) = events(on, "smoke_profile_start")
    assert start["at_step"] == first and start["recorded_steps"] == n and start["activities"] == ["CPU"]
    (done,) = events(on, "smoke_profile")
    assert done["complete"] and done["steps"] == list(range(first + 1, first + n + 1)) == [3, 4, 5]
    s = json.loads((on / "smoke" / "profile" / "summary.json").read_text(encoding="utf-8"))
    assert s["steps"] == done["steps"] and s["device"] == "cpu" and s["host"]["aten_ops_per_step"] > 0
    assert s["micro_batches_per_step"] >= 1 and 0 <= s["time"]["data_wait_share"] < 1
    assert (on / "smoke" / "profile" / s["trace"]["file"]).is_file()
    # the smoke check still ran, after the window (its throughput leaves the profiled steps out)
    assert [e["steps"] for e in events(on, "smoke_steps")] == [6]
    assert not (off / "smoke" / "profile").exists() and not events(off, "smoke_profile")


def test_profile_smoke_config():
    m = load_script("04_distill")
    assert m.DEFAULTS["perf"]["profile_smoke"] == "auto"
    for ok in ("auto", True, False):
        assert m.load_config(None, [f"perf.profile_smoke={json.dumps(ok)}"])["perf"]["profile_smoke"] == ok
    for bad in ("perf.profile_smoke=1", "perf.profile_smoke=yes", "perf.profile_smoke=null"):
        with pytest.raises(SystemExit, match="profile_smoke"):
            m.load_config(None, [bad])
