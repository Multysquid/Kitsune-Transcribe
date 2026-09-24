"""The overfit-run features of scripts/04_distill.py (configs/overfit_*.json, scripts/run_overfit_tests.cmd).

- audio_subset: seeded subsets of about N seconds (budget respected, greedy fill, closest-single fallback,
  deterministic per seed and independent of input order) and the `subset` events that list them
- schedule.clock "epochs": T = the steps of E full passes, warmup capped at 10 %, an eval at every epoch end (the last
  one is the final eval), the probe = the whole train subset (teacher-forced + greedy), SpecAugment off
- optim.offload "cpu" (CpuOffloadAdamW): bit-for-bit the on-device AdamW (unit level), the same weights and loss
  curves as the on-device optimizer in a full run, and an identical continuation after a crash + --resume
- the three overfit configs resolve and validate
- the 8 GB laptop: the memory probe holds a step's gradients when steps accumulate micro-batches, histograms
  concatenate one module at a time, the Windows VRAM cap, checkpoint renames that wait for a scanner, and
  scripts/supervise_distill.py (resume after a crash, give up on stalls)

CPU only (the laptop GPU runs other jobs), a tiny random student and a synthetic corpus in the real on-disk formats."""
import copy
import json
import math
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

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

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
PEAK = 3e-3


def events(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def merged(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merged(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def one_run(root: Path, name: str) -> Path:
    runs = list((root / "runs").glob(f"{name}-2*"))
    assert len(runs) == 1, runs
    return runs[0]


def parts(run: Path, kind: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((run / "metrics" / kind).glob("part-*.parquet"))])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + a tiny student saved exactly as 03 saves one, and a base config for it."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("overfit")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (36, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=5)
    sel = make_fake_selection(fc, greedy_n=4, probe_n=5)

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
        "device": "cpu", "autocast": "none",
        "optim": {"lr": PEAK},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"greedy_subset": 3, "batch_s": 20, "check_baselines": False},
        "log": {"layer_stats_every": 3, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 4},
        "hf": {"output_repo": None},
    }
    return dict(root=root, base=base, fc=fc)


def write_config(env, name: str, over: dict) -> str:
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------------------------------------- unit level


def test_audio_subset_budget_fallback_and_determinism():
    m = load_script("04_distill")
    rng = np.random.default_rng(0)
    ids = [f"src/{i:03d}.flac" for i in range(300)]
    dur = rng.uniform(0.5, 8.0, len(ids))
    d = dict(zip(ids, dur.tolist()))
    for budget in (1.0, 10.0, 60.0, 3600.0):
        got, rec = m.audio_subset(ids, dur, budget, np.random.default_rng([1234, 11]))
        total = sum(d[i] for i in got)
        assert not rec["fallback"] and 0 < total <= budget and len(set(got)) == len(got)
        # greedy fill: every utterance left out would overflow the final total
        assert all(total + d[i] > budget for i in set(ids) - set(got))
        assert rec["ids"] == got and rec["n"] == len(got) and rec["total_s"] == pytest.approx(total, abs=1e-3)
        assert rec["durations"] == [round(d[i], 3) for i in got] and rec["budget_s"] == budget
        # the draw depends on the seed and the id set only, not on the input order
        rev, _ = m.audio_subset(ids[::-1], dur[::-1], budget, np.random.default_rng([1234, 11]))
        assert rev == got
    other, _ = m.audio_subset(ids, dur, 60.0, np.random.default_rng([99, 11]))
    assert set(other) != set(m.audio_subset(ids, dur, 60.0, np.random.default_rng([1234, 11]))[0])
    everything, rec = m.audio_subset(ids, dur, 1e9, np.random.default_rng(0))
    assert sorted(everything) == sorted(ids) and rec["total_s"] == pytest.approx(dur.sum(), abs=1e-2)

    # nothing fits: the single utterance of >= 0.3 s closest to the budget (not the 0.25 s one below the floor)
    small = ["a", "b", "c", "d"]
    got, rec = m.audio_subset(small, [0.25, 0.9, 0.45, 3.0], 0.2, np.random.default_rng(0))
    assert got == ["c"] and rec["fallback"] and rec["total_s"] == 0.45 and rec["n"] == 1
    got, rec = m.audio_subset(small, [1.2, 1.1, 1.6, 3.0], 1.0, np.random.default_rng(0))
    assert got == ["b"] and rec["fallback"]
    got, _ = m.audio_subset(["x", "y"], [0.1, 0.2], 0.05, np.random.default_rng(0))
    assert got == ["x"]  # nothing reaches 0.3 s: the closest of all
    with pytest.raises(ValueError):
        m.audio_subset([], [], 1.0, np.random.default_rng(0))


def test_l2sp_value_from_the_apply_pass():
    """apply_(value=True) moves the weights exactly as apply_() and returns value() of the result (up to rounding)."""
    from kitsune.kd import L2SP

    torch.manual_seed(1)
    a = [torch.randn(4, 5, requires_grad=True), torch.randn(7, requires_grad=True)]
    b = [x.detach().clone().requires_grad_(True) for x in a]
    la, lb = L2SP(zip("xy", a), lam=0.05), L2SP(zip("xy", b), lam=0.05)
    with torch.no_grad():
        for x, y in zip(a, b):
            n = torch.randn_like(x)
            x.add_(n)
            y.add_(n)
    assert la.apply_(0.5) is None
    v = lb.apply_(0.5, value=True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert float(v) == pytest.approx(float(la.value()), rel=1e-6) and float(v) > 0
    assert float(lb.apply_(0.0, value=True)) == pytest.approx(float(lb.value()), rel=1e-6)  # lr 0: nothing moves
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("fused", [False, True])
def test_cpu_offload_adamw_is_adamw(monkeypatch, fused):
    """Masters in (small, here) chunks, gradients copied over and freed, AdamW on the masters, push: the same weights
    as torch.optim.AdamW on the parameters themselves (bitwise with foreach, within rounding with the fused kernel),
    a parameter without a gradient skipped, and a state_dict round trip that rebuilds the masters from reloaded
    weights continues identically."""
    m = load_script("04_distill")
    monkeypatch.setattr(m.CpuOffloadAdamW, "CHUNK", 40)  # 35 + (13 + 24): two chunks, one tensor starts a new one
    torch.manual_seed(0)
    shapes = [(5, 7), (13,), (3, 4, 2)]
    hp = dict(lr=1e-2, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    ref = [torch.randn(s, requires_grad=True) for s in shapes]
    dev = [p.detach().clone().requires_grad_(True) for p in ref]
    opt_ref = torch.optim.AdamW(ref, **hp)
    off = m.CpuOffloadAdamW(dev, fused=fused, **hp)
    assert len(off.chunks) == 4 and off.info()["impl"] == ("fused" if fused else "foreach") and not off.pinned
    tol = dict(atol=1e-6, rtol=0) if fused else dict(atol=0, rtol=0)

    def step(pairs, opts, k):
        grads = [torch.randn(s) for s in shapes]
        if k == 2:
            grads[1] = None  # no gradient this step: skipped, its moments untouched
        for params in pairs:
            for p, g in zip(params, grads):
                p.grad = None if g is None else g.clone()
        for o in opts:
            o.step()
            if isinstance(o, m.CpuOffloadAdamW):
                assert all(p.grad is None for p in o.params)  # the device gradients are freed by step()
                o.push()

    for k in range(5):
        step([ref, dev], [opt_ref, off], k)
        for p, q in zip(ref, dev):
            torch.testing.assert_close(q, p, **tol)
    assert int(off.state_dict()["state"][1]["step"]) == 4 and int(opt_ref.state_dict()["state"][1]["step"]) == 4

    # resume: fresh weights, then the checkpoint's weights loaded into them, then the optimizer state
    sd = copy.deepcopy(off.state_dict())
    dev2 = [torch.randn(s, requires_grad=True) for s in shapes]
    off2 = m.CpuOffloadAdamW(dev2, fused=fused, **hp)
    with torch.no_grad():
        for p, q in zip(dev2, dev):
            p.copy_(q)
    off2.load_state_dict(sd)
    assert all(g["fused"] == (True if fused else None) and g["foreach"] == (None if fused else True)
               for g in off2.param_groups)
    torch.manual_seed(7)
    step([dev], [off], 5)
    torch.manual_seed(7)
    step([dev2], [off2], 5)
    for p, q in zip(dev, dev2):
        assert torch.equal(p, q)
    with pytest.raises(ValueError):
        m.CpuOffloadAdamW([torch.zeros(3, dtype=torch.bfloat16, requires_grad=True)], **hp)


def test_overfit_configs_resolve():
    m = load_script("04_distill")
    want = {"overfit_1s": (1, 1, 1000, 60, 60), "overfit_10s": (10, 10, 100, 60, 60),
            "overfit_1h": (3600, 10, 30, 120, 60)}
    for name, (tr, ev_s, epochs, step_s, micro_s) in want.items():
        c = m.load_config(str(ROOT / "configs" / f"{name}.json"), [])
        assert c["run_name"] == name.replace("_", "-") and c["student"] == "students/b20x2560-d4"
        assert c["sources"] == ["reazon_small", "emilia_yodas", "galgame"] and c["eval_sets"] == EVAL
        assert c["subset"] == dict(train_utts=None, eval_utts_per_set=None, train_audio_s=tr, eval_audio_s=ev_s)
        assert c["schedule"]["clock"] == "epochs" and c["schedule"]["epochs"] == epochs
        assert c["batch"]["step_audio_s"] == step_s and c["batch"]["micro_audio_s"] == micro_s
        if tr <= 60:
            assert c["batch"]["step_audio_s"] >= tr  # one step per epoch
        assert c["specaug"]["enabled"] is False and c["optim"]["offload"] == "cpu"
        assert c["optim"]["offload_fused"] is True  # the fused CPU AdamW kernel: ~1 s per step less than foreach
        # stop once the probe KL (teacher-forced on the whole train subset) is flat: memorised, nothing left to learn
        patience, min_evals = (3, 3) if name == "overfit_1h" else (20, 10)
        assert c["early_stop"] == dict(enabled=True, metric="probe_kl", patience=patience, min_delta_rel=0.02,
                                       min_delta_abs=0.001, min_evals=min_evals, floor=None, action="stop")
        assert c["eval"]["every_epochs"] == 1 and c["eval"]["probe"] and c["eval"]["probe_is_train"]
        # no GO/NO-GO for a sanity run ("N/A" with the numbers); a mini eval every 200 steps (never at an epoch end)
        assert c["eval"]["gate"] is False and c["eval"]["full_every_epochs"] is None
        assert c["eval"]["mini"] == dict(every_steps=200, val_per_set=32, train_utts=64, greedy=True)
        assert c["loss"] == m.DEFAULTS["loss"] == dict(w_kl=1.0, w_ce=0.8, l2sp_lambda=0.05)
        assert {k: c["optim"][k] for k in ("lr", "betas", "clip")} == dict(lr=1e-4, betas=[0.9, 0.98], clip=1.0)
        assert c["autocast"] == "bfloat16" and c["perf"]["relpos_patch"] and c["memory"]["grad_ckpt"] == "auto"
        # a resume point right after the smoke steps and every 10 min: scripts/supervise_distill.py relaunches from
        # the newest after one of this GPU's intermittent CUDA faults
        assert c["ckpt"]["weights_every_min"] == 60 and c["ckpt"]["full_local_every_min"] == 10
        assert c["ckpt"]["keep_local"] == 1 and c["ckpt"]["upload_full_at"] == [] and c["hf"]["output_repo"] is None
        assert c["perf"]["num_workers"] == 2 and c["smoke"]["min_audio_s_per_s"] == 0
        assert not c["smoke"]["require_loss_decrease"] and c["ckpt"]["full_after_smoke"]
    for bad in (["schedule.clock=epochs"], ["schedule.clock=epochs", "schedule.epochs=0"], ["optim.offload=gpu"],
                ["subset.train_audio_s=5", "subset.train_utts=10"], ["subset.eval_audio_s=-1"],
                ["eval.every_epochs=0"], ["eval.probe=false", "eval.probe_is_train=true"], ["specaug.enabled=0"],
                ["perf.loader_timeout_s=-1"], ["perf.loader_timeout_s=10m"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
    via = m.load_config(str(ROOT / "configs" / "viability.json"), [])
    assert via["optim"]["offload"] == "none" and via["specaug"]["enabled"] and via["schedule"]["clock"] == "wall"
    assert via["subset"]["train_audio_s"] is None and via["eval"]["every_epochs"] is None
    assert via["early_stop"] == dict(enabled=True, metric="heldout_kl", patience=3, min_delta_rel=0.005,
                                     min_delta_abs=0.0, min_evals=3, floor=None, action="cooldown")


# --------------------------------------------------------------------------------------------- the 8 GB laptop


class _Rows:
    def __getitem__(self, idx):
        return list(idx)


def test_memory_probe_holds_the_steps_gradients(monkeypatch):
    """train_step accumulates a step's micro-batches into the same gradients, so micro-batch 2+ runs forward and
    backward with the full fp32 gradients already allocated (2.3 GiB for the real student, which the offloaded run no
    longer covers with an AdamW reserve). When steps have several micro-batches every probe pass must start from
    allocated (zero) gradients, else the probe accepts a micro size that OOMs (or spills) from the second micro-batch
    on; with one micro-batch per step it starts from none, as a step's only micro-batch does. Freed afterwards."""
    from kitsune import trainset

    m = load_script("04_distill")
    utts = [trainset.Utt(id=f"u{i}", source="s", duration=2.0 + 0.1 * i, n_tok=5, audio_off=0, audio_len=0, tok_off=0)
            for i in range(12)]
    seen = []

    def fake_fwd_bwd(R, mb):  # records the gradients a pass starts from, then accumulates and frees like fwd_bwd
        seen.append([None if p.grad is None else float(p.grad.abs().sum()) for p in R.params])
        for p in R.params:
            p.grad = torch.ones_like(p) if p.grad is None else p.grad.add_(1.0)
        for p in R.params:
            p.grad = None
        return 0.0, True

    monkeypatch.setattr(m, "fwd_bwd", fake_fwd_bwd)
    R = SimpleNamespace(device=torch.device("cpu"), ds=_Rows(),
                        params=[torch.nn.Parameter(torch.randn(4, 3)), torch.nn.Parameter(torch.randn(5))])
    for step_s, micro_s, held in ((12.0, 5.0, True), (1000.0, 1000.0, False)):
        planner = trainset.StepPlanner(utts, step_audio_s=step_s, micro_audio_s=micro_s, pool_micro=4, seed=0)
        assert (max(len(s) for s in planner.epoch_plan(0)) > 1) == held
        seen.clear()
        rec = dict(peak_gb={})
        m.probe_passes(R, planner, rec)
        assert rec["grads_held"] == held and len(seen) == 3  # longest, most decoder positions, most targets
        assert seen == ([[0.0, 0.0]] * 3 if held else [[None, None]] * 3)
        assert all(p.grad is None for p in R.params)


def test_histograms_concatenate_one_module_at_a_time():
    """A histogram step used to concatenate every module's gradients (or weights) at once: a third full-size fp32 copy
    next to the weights and gradients, 2.3 GiB the 8 GB laptop does not have, outside train_step's OOM handling.
    _groups now yields one module's concatenation at a time and keeps none of the earlier ones alive."""
    m = load_script("04_distill")
    names = ["model.encoder.layers.0.a.weight", "model.encoder.layers.0.b.bias", "model.encoder.layers.1.a.weight",
             "model.decoder.embed_tokens.weight", "model.decoder.pos_emb.weight"]
    ts = [torch.randn(3, 4), torch.randn(5), torch.randn(6), torch.randn(2, 2), None]  # no gradient: left out
    R = SimpleNamespace(param_names=names)
    it = m._groups(R, ts)
    k, t = next(it)
    assert k == "encoder.layers.0" and torch.equal(t, torch.cat([ts[0].flatten(), ts[1]]))
    ref = weakref.ref(t)
    del t
    k, t = next(it)
    assert ref() is None  # the previous module's copy is gone once the caller drops it
    assert k == "encoder.layers.1" and torch.equal(t, ts[2])
    assert [(k, v.tolist()) for k, v in it] == [("decoder.embed_tokens", ts[3].flatten().tolist())]


def test_cap_vram_caps_the_allocator_on_windows_cuda_only(monkeypatch):
    """The Windows driver's sysmem fallback turns a VRAM overflow into a silent spill to shared memory, so neither the
    memory probe's fallbacks nor train_step's OOM skip would ever fire. On Windows + CUDA the caching allocator is
    capped at the free VRAM (plus what it holds) minus a margin; nowhere else."""
    m = load_script("04_distill")
    gib, calls, evs = 2**30, [], []
    monkeypatch.setattr(m.torch.cuda, "mem_get_info", lambda device=None: (int(6.9 * gib), 8 * gib))
    monkeypatch.setattr(m.torch.cuda, "memory_reserved", lambda device=None: gib // 4)
    devices = []

    def set_fraction(f, device=None):  # the real API rejects torch.device("cuda") without an index
        devices.append(device)
        calls.append(f)

    monkeypatch.setattr(m.torch.cuda, "set_per_process_memory_fraction", set_fraction)
    monkeypatch.setattr(m.torch.cuda, "current_device", lambda: 0)
    log = SimpleNamespace(event=lambda kind, **kw: evs.append((kind, kw)))

    def run(dev: str, osname: str):
        R = SimpleNamespace(device=torch.device(dev), log=log, vram_cap_gb=None)
        with monkeypatch.context() as mp:
            mp.setattr(m.os, "name", osname)
            m.cap_vram(R)
        return R

    assert run("cpu", "nt").vram_cap_gb is None and run("cuda", "posix").vram_cap_gb is None and not calls
    R = run("cuda", "nt")
    assert len(calls) == 1 and calls[0] * 8 == pytest.approx(6.9 + 0.25 - m.VRAM_MARGIN_GIB)
    assert devices == [0] and all(isinstance(d, int) for d in devices)
    assert R.vram_cap_gb == pytest.approx(6.9 + 0.25 - m.VRAM_MARGIN_GIB, abs=0.01)
    assert evs == [("vram_cap", dict(cap_gb=R.vram_cap_gb, free_gb=6.9, held_gb=0.25, total_gb=8.0,
                                     margin_gb=m.VRAM_MARGIN_GIB, fraction=round(calls[0], 4)))]


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to rename a directory with a file open in it")
def test_replace_dir_waits_for_a_scanner(tmp_path):
    """A virus scanner or the indexer still reading a just-written checkpoint file made os.replace of its directory
    fail (PermissionError) and the run exit; _replace_dir retries until the file is closed."""
    m = load_script("04_distill")
    tmp, final = tmp_path / "full_step_5.tmp", tmp_path / "full_step_5"
    tmp.mkdir()
    (tmp / "model.pt").write_bytes(b"x" * 1000)
    f = open(tmp / "model.pt", "rb")
    try:
        with pytest.raises(PermissionError):
            os.replace(tmp, final)
        timer = threading.Timer(0.5, f.close)
        timer.start()
        t0 = time.monotonic()
        m._replace_dir(tmp, final)
        timer.join()
    finally:
        f.close()
    assert time.monotonic() - t0 >= 0.4 and not tmp.exists() and (final / "model.pt").read_bytes() == b"x" * 1000


def test_resume_from_an_older_full_state_sets_the_abandoned_attempt_aside(tmp_path):
    """--resume full_step_4 while the attempt it backs out of left full_step_8: rotation kept the highest-numbered
    states and --resume <run dir> took the newest, so the resumed run's own states were deleted and the next run-dir
    resume (supervise_distill.py) silently continued the abandoned attempt. Its newer dirs now move under
    abandoned-*/, where no scanner looks; a resume from the newest state moves nothing."""
    m = load_script("04_distill")
    run = tmp_path / "run"
    ck = run / "checkpoints"
    for name in ("full_step_4", "full_step_8", "step_4", "step_6", "step_9"):
        (ck / name).mkdir(parents=True)
        if name.startswith("full"):
            (ck / name / "trainer.pt").write_bytes(b"x")
    assert m.set_aside_newer(ck, 8) == [] and m.set_aside_newer(ck, 4) == ["full_step_8", "step_6", "step_9"]
    (aside,) = [p for p in ck.iterdir() if p.name.startswith("abandoned-")]
    assert sorted(p.name for p in aside.iterdir()) == ["full_step_8", "step_6", "step_9"]
    assert (aside / "full_step_8" / "trainer.pt").exists() and m.find_full_state(run) == ck / "full_step_4"

    evs = []
    R = SimpleNamespace(cfg={"ckpt": {"keep_local": 1}}, ckpt_dir=ck, uploader=SimpleNamespace(busy=set),
                        log=SimpleNamespace(event=lambda kind, **kw: evs.append((kind, kw["name"]))))
    (ck / "full_step_6").mkdir()
    (ck / "full_step_6" / "trainer.pt").write_bytes(b"x")
    m.rotate_full(R)  # the resumed run's first save
    assert evs == [("checkpoint_deleted", "full_step_4")] and m.find_full_state(run) == ck / "full_step_6"
    assert sorted(p.name for p in ck.iterdir()) == [aside.name, "full_step_6", "step_4"]


FAKE_TRAINER = r'''
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--resume")
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
calls_path = Path(os.environ["FAKE_CALLS"])
calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
act = json.loads(os.environ["FAKE_PLAN"])[len(calls)]
calls.append(dict(config=a.config, resume=a.resume, sets=a.set, blocking=os.environ.get("CUDA_LAUNCH_BLOCKING")))
calls_path.write_text(json.dumps(calls))
if a.resume:
    run = Path(a.resume)
else:
    cfg = json.loads(Path(a.config).read_text())
    run = Path(cfg["runs_root"]) / f"{cfg['run_name']}-20260924T0000{len(calls):02d}Z"
    run.mkdir(parents=True)
if act.get("full") is not None:
    d = run / "checkpoints" / f"full_step_{act['full']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trainer.pt").write_bytes(b"x")
sys.exit(act["rc"])
'''


def test_supervisor_resumes_after_crashes_and_gives_up_on_stalls(tmp_path, monkeypatch):
    """scripts/supervise_distill.py (run_overfit_tests.cmd runs every overfit config under it): this GPU faults every
    ~10-40 min under load and a fault kills the trainer, so a crash is followed by --resume of the run's newest full
    state; attempts without a newer full state escalate to CUDA_LAUNCH_BLOCKING=1 after two and give up after
    --max-stalls (--max-attempts of them in total; attempts that progressed never count); a crash before the first
    full state starts over; exits 0 and 3 are final."""
    sup = load_script("supervise_distill")
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(FAKE_TRAINER, encoding="utf-8")
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

    def supervise(name: str, plan: list[dict], *extra) -> tuple[int, list[dict], list[str], Path]:
        runs = tmp_path / name / "runs"
        cfg = tmp_path / name / "config.json"
        cfg.parent.mkdir()
        cfg.write_text(json.dumps(dict(run_name=name, runs_root=str(runs))), encoding="utf-8")
        calls, log = tmp_path / name / "calls.json", tmp_path / name / "pipeline.log"
        monkeypatch.setenv("FAKE_CALLS", str(calls))
        monkeypatch.setenv("FAKE_PLAN", json.dumps(plan))
        rc = sup.main(["--config", str(cfg), "--log", str(log), "--script", str(fake), "--set", "seed=7", *extra])
        return rc, json.loads(calls.read_text()), log.read_text(encoding="utf-8").splitlines(), runs

    # crash after the first full state -> resume; two resumes without progress -> blocking mode; progress -> back
    rc, calls, lines, runs = supervise("sup-a", [{"full": 2, "rc": 1}, {"rc": 1}, {"rc": -1073741819},
                                                 {"full": 40, "rc": 1}, {"rc": 0}])
    (run,) = runs.iterdir()
    assert rc == 0 and [(c["resume"], c["blocking"]) for c in calls] == [
        (None, None), (str(run), None), (str(run), None), (str(run), "1"), (str(run), None)]
    assert calls[0]["config"] and all(c["sets"] == ["seed=7"] for c in calls)
    assert len(lines) == 10 and "resume" in lines[2] and "full_step_2" in lines[2] and "2 -> 40" in lines[7]

    # no full state ever: every attempt starts over in a new run dir, the third in blocking mode, then it gives up
    rc, calls, lines, runs = supervise("sup-b", [{"rc": 1}] * 3, "--max-stalls", "3")
    assert rc == 1 and [(c["resume"], c["blocking"]) for c in calls] == [(None, None), (None, None), (None, "1")]
    assert len(list(runs.iterdir())) == 3 and "giving up" in lines[-1]

    # ThroughputTooLow is final
    rc, calls, _, _ = supervise("sup-c", [{"rc": 3}, {"rc": 0}])
    assert rc == 3 and len(calls) == 1

    # attempts that made progress do not count toward --max-attempts: a long run that keeps moving is never given up
    rc, calls, lines, _ = supervise("sup-d", [{"full": 10 * (i + 1), "rc": 1} for i in range(21)] + [{"rc": 0}])
    assert rc == 0 and len(calls) == 22 and not any("giving up" in x for x in lines)

    # ... the ones without progress do, in total, even when they never come --max-stalls in a row
    rc, calls, lines, _ = supervise("sup-e", [{"full": 2, "rc": 1}, {"rc": 1}, {"full": 4, "rc": 1}, {"rc": 1},
                                              {"rc": 0}], "--max-attempts", "2")
    assert rc == 1 and len(calls) == 4 and "giving up after 2 attempts without a newer full state" in lines[-1]


# ------------------------------------------------------------------------------------------------ whole runs


def test_epoch_clock_subsets_probe_and_no_specaug(env):
    """~8 s of train audio for 3 epochs of >= 2 steps, ~4 s of pooled eval audio, an eval after every epoch with the
    whole train subset as the probe (teacher-forced) and ~3 s of it decoded greedily, SpecAugment off, offloaded
    AdamW. Everything the TensorBoard view of the overfit runs relies on is checked here."""
    m = load_script("04_distill")
    path = write_config(env, "ov-epochs", {
        "subset": {"train_audio_s": 8, "eval_audio_s": 4},
        "specaug": {"enabled": False},
        "optim": {"offload": "cpu"},
        "schedule": {"clock": "epochs", "epochs": 3, "warmup_steps": 300, "cooldown_frac": 0.3},
        "batch": {"step_audio_s": 4, "micro_audio_s": 3, "pool_micro": 4},
        "eval": {"every_epochs": 1, "probe_is_train": True, "probe_greedy_audio_s": 3},
        "ckpt": {"weights_every_min": 60, "full_local_every_min": 60, "keep_local": 1, "full_after_smoke": False},
        "smoke": {"steps": 2, "min_audio_s_per_s": 0, "require_loss_decrease": False},
    })
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "ov-epochs")
    ev = events(run)

    # the subsets: within budget, logged with every id, and exactly what was trained / evaluated / probed
    sub = {e["split"]: e for e in ev if e["kind"] == "subset"}
    tr, es, pg = sub["train"], sub["eval"], sub["probe_greedy"]
    assert 5 < tr["total_s"] <= 8 and 2 < es["total_s"] <= 4 and 0 < pg["total_s"] <= 3
    for s in (tr, es, pg):
        assert not s["fallback"] and s["n"] == len(s["ids"]) == len(s["durations"]) == len(set(s["ids"]))
        assert s["total_s"] == pytest.approx(sum(s["durations"]), abs=2e-3)
    assert all(i.startswith(("src_a/", "src_b/")) for i in tr["ids"]) and set(pg["ids"]) <= set(tr["ids"])
    assert all(i.startswith(tuple(f"{s}/" for s in EVAL)) for i in es["ids"])
    ev0 = run / "evals" / "step_0"
    assert sorted(pd.read_parquet(ev0 / "probe.parquet")["id"]) == sorted(tr["ids"])  # probe = the train subset
    assert sorted(pd.read_parquet(ev0 / "probe_greedy.parquet")["id"]) == sorted(pg["ids"])
    tf_ids = [i for p in ev0.glob("tf_*.parquet") for i in pd.read_parquet(p)["id"]]
    gr_ids = [i for p in ev0.glob("greedy_*.parquet") for i in pd.read_parquet(p)["id"]]
    assert sorted(tf_ids) == sorted(gr_ids) == sorted(es["ids"])  # teacher-forced AND greedy on the whole eval subset
    data = next(e for e in ev if e["kind"] == "data")
    assert data["train_utts"] == tr["n"] and data["eval_utts"] == es["n"] and data["probe_greedy"] == pg["n"]

    # the epoch clock: T = the steps of 3 passes, warmup min(300, ceil(10 %)), WSD over steps
    sched = next(e for e in ev if e["kind"] == "schedule")
    per, total = sched["steps_per_epoch"], sched["total_steps"]
    assert len(per) == 3 and sum(per) == total and min(per) >= 2
    assert sched["warmup_steps"] == min(300, math.ceil(0.1 * total)) < 300
    steps = pd.read_parquet(run / "metrics" / "steps.parquet")
    assert steps["step"].tolist() == list(range(1, total + 1))
    assert steps.groupby("data/epoch").size().tolist() == per
    for step, lr, phase in zip(steps["step"], steps["opt/lr"], steps["sched/phase"]):
        want = m.wsd_lr(PEAK, int(step), float(step - 1), float(total), sched["warmup_steps"], 0.3)
        assert lr == pytest.approx(want[0]) and phase == want[1], step
    assert steps["sched/phase"].iloc[-1] == 2 and steps["sched/progress"].iloc[-1] == pytest.approx(1.0)
    utts = parts(run, "train_utts")
    for e, g in utts.groupby("epoch"):  # every pass trains on every utterance of the subset exactly once
        assert sorted(g["id"]) == sorted(tr["ids"]), e

    # an eval after every epoch: step 0, the ends of epochs 1 and 2 in the loop, the end of epoch 3 = the final eval
    ends = np.cumsum(per).tolist()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["steps"] == total and summary["epochs"] == 3.0
    hist = summary["history"]
    assert [r["step"] for r in hist] == [0, *ends] and [r["epoch"] for r in hist] == [0.0, 1.0, 2.0, 3.0]
    assert all("probe_kl" in r and "cer_teacher_corpus" in r["probe_greedy"] for r in hist)
    evals = [e for e in ev if e["kind"] == "eval"]
    assert [(e["at_step"], e["final"], e["epoch"]) for e in evals] == [
        (0, False, 0.0), (ends[0], False, 1.0), (ends[1], False, 2.0), (total, True, 3.0)]
    sc = pd.read_parquet(run / "metrics" / "scalars.parquet")
    ep = sc[sc["tag"] == "eval/epoch"].sort_values("step")
    assert ep["step"].tolist() == [0, *ends] and ep["value"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert {"eval/probe/all/kl", "eval/probe_greedy/all/cer_teacher_corpus", "eval/kl_gap_heldout_minus_probe",
            "eval/cer_teacher_gap_heldout_minus_probe", "eval/tf/all/kl", "eval/greedy/all/cer_teacher_corpus",
            "eval/greedy_full/all/cer_ref_corpus"} <= set(sc["tag"])
    for step in (0, *ends):
        assert (run / "evals" / f"step_{step}" / "probe_greedy.parquet").is_file()

    # SpecAugment off: nothing masked, in the step rows and per utterance
    assert (steps["aug/masked_frac"] == 0).all() and (utts["masked_frac"] == 0).all()
    assert next(e for e in ev if e["kind"] == "optim_offload")["impl"] == "foreach"
    assert {"smoke_steps", "smoke_padded_row", "verdict", "logger_close"} <= {e["kind"] for e in ev}


def _offload_config(env, name: str, offload: str) -> str:
    return write_config(env, name, {
        "optim": {"offload": offload},
        "schedule": {"warmup_steps": 3, "cooldown_frac": 0.3, "clock": "steps", "max_steps": 8},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "eval": {"every_steps": 1000, "final_full_greedy": False},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 4, "keep_local": 5},
        "smoke": {"enabled": False},
    })


def _state(run: Path, step: int) -> tuple[dict, dict]:
    d = run / "checkpoints" / f"full_step_{step}"
    return (torch.load(d / "model.pt", map_location="cpu", weights_only=True),
            torch.load(d / "optimizer.pt", map_location="cpu", weights_only=True))


def _assert_same_run(a: Path, b: Path) -> pd.DataFrame:
    (wa, oa), (wb, ob) = _state(a, 8), _state(b, 8)
    assert wa.keys() == wb.keys()
    for k in wa:
        torch.testing.assert_close(wb[k], wa[k], atol=1e-6, rtol=0, msg=k)
    assert oa["state"].keys() == ob["state"].keys()
    for i in oa["state"]:
        for k in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(ob["state"][i][k], oa["state"][i][k], atol=1e-9, rtol=1e-6)
        assert float(ob["state"][i]["step"]) == float(oa["state"][i]["step"])
    sa = pd.read_parquet(a / "metrics" / "steps.parquet")
    sb = pd.read_parquet(b / "metrics" / "steps.parquet")
    assert sa["step"].tolist() == sb["step"].tolist() == list(range(1, 9))
    for col in ("loss/objective", "loss/kl", "loss/ce", "tok/top1", "opt/grad_norm", "opt/lr", "aug/masked_frac"):
        np.testing.assert_allclose(sb[col].to_numpy(), sa[col].to_numpy(), rtol=1e-6, atol=1e-9, err_msg=col)
    np.testing.assert_allclose(sb["loss/l2sp"].to_numpy(), sa["loss/l2sp"].to_numpy(), rtol=1e-5, err_msg="l2sp")
    return sa


def test_offload_matches_the_on_device_optimizer_and_resumes(env, monkeypatch):
    """8 steps with optim.offload none vs cpu (fp32, CPU): the same weights (atol 1e-6), AdamW moments and loss curves;
    the update ratios logged from the host masters match. Then the offloaded run crashes before step 6 and resumes from
    its step-4 full state: the continuation ends at the same weights and replays the same losses."""
    m = load_script("04_distill")
    assert m.main(["--config", _offload_config(env, "ov-none", "none")]) == 0
    assert m.main(["--config", _offload_config(env, "ov-cpu", "cpu")]) == 0
    none, cpu = one_run(env["root"], "ov-none"), one_run(env["root"], "ov-cpu")
    assert not any(e["kind"] == "optim_offload" for e in events(none))
    info = next(e for e in events(cpu) if e["kind"] == "optim_offload")
    assert info["impl"] == "foreach" and info["pinned"] is False and info["params"] > 0
    steps = _assert_same_run(none, cpu)
    assert (steps["aug/masked_frac"] > 0).all()  # SpecAugment on by default
    sc_n, sc_c = (pd.read_parquet(r / "metrics" / "scalars.parquet") for r in (none, cpu))
    ratio = [t for t in sc_n["tag"].unique() if t.startswith("layers/update_ratio/")]
    assert ratio
    for sc in (sc_n, sc_c):
        assert set(sc[sc["tag"].isin(ratio)]["step"]) == {3, 6}
    a = sc_n[sc_n["tag"].isin(ratio)].sort_values(["step", "tag"])["value"].to_numpy()
    b = sc_c[sc_c["tag"].isin(ratio)].sort_values(["step", "tag"])["value"].to_numpy()
    np.testing.assert_allclose(b, a, rtol=1e-5)

    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "6")
    cfg = _offload_config(env, "ov-crash", "cpu")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", cfg])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    crash = one_run(env["root"], "ov-crash")
    assert m.main(["--config", cfg, "--resume", str(crash / "checkpoints" / "full_step_4")]) == 0
    assert next(e for e in events(crash) if e["kind"] == "resumed")["at_step"] == 4
    _assert_same_run(cpu, crash)  # steps.parquet of the resumed run holds 1-4 from the first launch, 5-8 replayed
    utts = parts(crash, "train_utts")
    ref = parts(cpu, "train_utts")
    for step in (5, 6):
        a = ref[ref["step"] == step].sort_values("id")
        b = utts[(utts["step"] == step) & (utts["attempt"] == 1)].sort_values("id")
        assert a["id"].tolist() == b["id"].tolist() and len(a)
        np.testing.assert_allclose(b["kl"].to_numpy(), a["kl"].to_numpy(), rtol=1e-6, atol=1e-9)


def test_resume_applies_optim_l2sp_and_grad_ckpt_overrides(env, monkeypatch):
    """A resume's --set optim.betas/eps/weight_decay, loss.l2sp_lambda and memory.grad_ckpt were logged as applied
    while the restored optimizer, L2-SP and memory choice kept the saved values: now they take effect. A new
    batch.micro_audio_s (it shapes the step plan the resume continues) stops the resume naming the key; repeating the
    checkpoint's value resumes as usual."""
    m = load_script("04_distill")
    cfg = write_config(env, "ov-sets", {
        "optim": {"betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0},
        "loss": {"l2sp_lambda": 0.05},
        "memory": {"grad_ckpt": False},
        "schedule": {"warmup_steps": 3, "cooldown_frac": 0.3, "clock": "steps", "max_steps": 8},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "eval": {"every_steps": 1000, "final_full_greedy": False},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 4, "keep_local": 5},
        "smoke": {"enabled": False},
    })
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "6")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", cfg])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    run = one_run(env["root"], "ov-sets")
    full = str(run / "checkpoints" / "full_step_4")
    with pytest.raises(SystemExit, match=r"batch\.micro_audio_s=2 differs from the checkpoint's 3"):
        m.main(["--resume", full, "--set", "batch.micro_audio_s=2"])
    sets = ["optim.betas=[0.5,0.6]", "optim.eps=0.001", "optim.weight_decay=0.1", "loss.l2sp_lambda=0.5",
            "memory.grad_ckpt=true", "batch.micro_audio_s=3"]
    assert m.main(["--resume", full, *[a for s in sets for a in ("--set", s)]]) == 0
    ev = events(run)
    res = next(e for e in ev if e["kind"] == "resume")
    assert res["unchanged"] == {"batch.micro_audio_s": 3} and res["overrides"] == {
        "optim.betas": [0.5, 0.6], "optim.eps": 0.001, "optim.weight_decay": 0.1, "loss.l2sp_lambda": 0.5,
        "memory.grad_ckpt": True}
    assert [e["grad_ckpt"] for e in ev if e["kind"] == "model"] == [False, True]
    d = run / "checkpoints" / "full_step_8"
    groups = torch.load(d / "optimizer.pt", map_location="cpu", weights_only=True)["param_groups"]
    assert groups and all((tuple(g["betas"]), g["eps"], g["weight_decay"]) == ((0.5, 0.6), 0.001, 0.1) for g in groups)
    assert torch.load(d / "l2sp.pt", map_location="cpu", weights_only=True)["lam"] == 0.5
    assert torch.load(d / "trainer.pt", map_location="cpu", weights_only=True)["st"]["memory"]["grad_ckpt"] is True
