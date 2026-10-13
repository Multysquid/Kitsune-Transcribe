"""The overfit-run features of scripts/04_distill.py (the laptop's overfit sanity runs of 2026-09-24).

- audio_subset: seeded subsets of about N seconds (budget respected, greedy fill, closest-single fallback,
  deterministic per seed and independent of input order) and the `subset` events that list them
- schedule.clock "epochs": T = the steps of E full passes, warmup capped at 10 %, an eval at every epoch end (the last
  one is the final eval), the probe = the whole train subset (teacher-forced + greedy), SpecAugment off
- the overfit options are validated
- the 8 GB laptop: the memory probe holds a step's gradients when steps accumulate micro-batches, histograms
  concatenate one module at a time, the Windows VRAM cap (scripts/05_evaluate.py's since the trainer's own went with
  the laptop), and checkpoint renames that wait for a scanner and come after the files reach the disk
- a resume sets an abandoned attempt aside and applies its optim / L2-SP / grad_ckpt overrides

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


def test_overfit_options_are_validated():
    """The overfit runs' options refuse values they cannot run with; DEFAULTS (and so the viability run, which only
    turned early stopping on) set none of them."""
    m = load_script("04_distill")
    for bad in (["schedule.clock=epochs"], ["schedule.clock=epochs", "schedule.epochs=0"],
                ["subset.train_audio_s=5", "subset.train_utts=10"], ["subset.eval_audio_s=-1"],
                ["eval.every_epochs=0"], ["eval.probe=false", "eval.probe_is_train=true"], ["specaug.enabled=0"],
                ["perf.loader_timeout_s=-1"], ["perf.loader_timeout_s=10m"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
    d = m.load_config(None, [])
    assert d["specaug"]["enabled"] and d["schedule"]["clock"] == "wall"
    assert d["subset"]["train_audio_s"] is None and d["eval"]["every_epochs"] is None
    assert d["early_stop"] == dict(enabled=False, metric="heldout_kl", patience=3, min_delta_rel=0.005,
                                   min_delta_abs=0.0, min_evals=3, floor=None, action="cooldown", smooth=1,
                                   allow_test_sets=True)


# --------------------------------------------------------------------------------------------- the 8 GB laptop


class _Rows:
    def __getitem__(self, idx):
        return list(idx)


def test_memory_probe_holds_the_steps_gradients(monkeypatch):
    """train_step accumulates a step's micro-batches into the same gradients, so micro-batch 2+ runs forward and
    backward with the full fp32 gradients already allocated (2.3 GiB for the real student). When steps have several
    micro-batches every probe pass must start from
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


def test_checkpoints_reach_the_disk_before_their_rename(tmp_path, monkeypatch):
    """save_full and save_weights fsync every file of <name>.tmp/ before renaming it into place and the checkpoints
    dir after it; the end save's trainer-only rewrite fsyncs its .tmp files before each replace. Without that, an
    unclean host crash soon after a save left the rename on disk but not the data: full_step_<N>/ with empty or
    NUL-filled files, the state --resume and vast/supervise.py take as the newest."""
    from kitsune import student as S

    m = load_script("04_distill")
    calls = []
    real_fsync, real_sync, real_dir, real_file = m.fsync_path, m._sync_dir, m._replace_dir, m._replace_file

    def rec(kind, real):
        def f(*a):
            calls.append((kind, *(Path(p).name for p in a)))
            return real(*a)
        return f

    monkeypatch.setattr(m, "fsync_path", rec("fsync", real_fsync))
    monkeypatch.setattr(m, "_sync_dir", rec("sync_dir", real_sync))
    monkeypatch.setattr(m, "_replace_dir", rec("replace", real_dir))
    monkeypatch.setattr(m, "_replace_file", rec("replace", real_file))

    def fake_save_student(model, out, processor, meta):
        out.mkdir(parents=True)
        for f in ("model.safetensors", "config.json", "student_meta.json"):
            (out / f).write_bytes(b"w")

    monkeypatch.setattr(S, "save_student", fake_save_student)
    run = tmp_path / "run"
    R = SimpleNamespace(cfg={"ckpt": {"keep_local": 2}, "student": "s"}, run_dir=run, ckpt_dir=run / "checkpoints",
                        st=dict(fulls=[], weights=[], epoch_progress=0.0, last_objective=None), clock=lambda: 0.0,
                        planner=SimpleNamespace(state_dict=dict), log=SimpleNamespace(event=lambda *a, **k: None,
                                                                                     state_dict=dict),
                        model=torch.nn.Linear(1, 1), opt=SimpleNamespace(state_dict=dict),
                        l2sp=SimpleNamespace(state_dict=dict), student_meta={}, processor=None,
                        uploader=SimpleNamespace(repo=None, submit=lambda d, n: None, busy=set, pending={}))
    R.ckpt_dir.mkdir(parents=True)

    for save, name in ((m.save_full, "full_step_2"), (m.save_weights, "step_2")):
        calls.clear()
        d = save(R, 2, "periodic")
        files = sorted(p.name for p in d.iterdir())
        assert calls == [*[("fsync", f) for f in files], ("sync_dir", f"{name}.tmp"),
                         ("replace", f"{name}.tmp", name), ("sync_dir", "checkpoints")], name
    assert files == ["config.json", "model.safetensors", "student_meta.json"]

    calls.clear()  # the end save at a step that already has a full state: only trainer.pt/.json are rewritten
    m.save_full(R, 2, "end")
    assert calls == [("fsync", "trainer.pt.tmp"), ("replace", "trainer.pt.tmp", "trainer.pt"),
                     ("fsync", "trainer.json.tmp"), ("replace", "trainer.json.tmp", "trainer.json"),
                     ("sync_dir", "full_step_2")]


def test_resume_from_an_older_full_state_sets_the_abandoned_attempt_aside(tmp_path):
    """--resume full_step_4 while the attempt it backs out of left full_step_8: rotation kept the highest-numbered
    states and --resume <run dir> took the newest, so the resumed run's own states were deleted and the next run-dir
    resume silently continued the abandoned attempt. Its newer dirs now move under abandoned-*/, where no scanner
    looks. A resume from the newest state moves only the weights saved after it: the resumed run replays the steps,
    not those saves, so they would stay next to its own and go up with them."""
    m = load_script("04_distill")

    def tree(root, names):
        ck = root / "checkpoints"
        for name in names:
            (ck / name).mkdir(parents=True)
            if name.startswith("full") and not name.endswith(".tmp"):
                (ck / name / "trainer.pt").write_bytes(b"x")
        return ck

    names = ("full_step_4", "full_step_8", "step_4", "step_6", "step_9")
    ck8 = tree(tmp_path / "newest", names)
    assert m.set_aside_newer(ck8, 8) == ["step_9"] and m.set_aside_newer(ck8, 8) == []
    (aside8,) = [p for p in ck8.iterdir() if p.name.startswith("abandoned-")]
    assert [p.name for p in aside8.iterdir()] == ["step_9"]
    assert m.find_full_state(ck8.parent) == ck8 / "full_step_8"
    # a crash inside the full-state save right after the same step's weights: the resume is from the full state
    # before, and the orphaned weights move (the torn .tmp is no checkpoint any scanner takes)
    ckt = tree(tmp_path / "torn", ("full_step_4", "step_4", "step_8", "full_step_8.tmp"))
    assert m.find_full_state(ckt.parent) == ckt / "full_step_4" and m.set_aside_newer(ckt, 4) == ["step_8"]
    assert (ckt / "full_step_8.tmp").is_dir()

    run = tmp_path / "run"
    ck = tree(run, names)
    assert m.set_aside_newer(ck, 4) == ["full_step_8", "step_6", "step_9"]
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


# ------------------------------------------------------------------------------------------------ whole runs


def test_epoch_clock_subsets_probe_and_no_specaug(env):
    """~8 s of train audio for 3 epochs of >= 2 steps, ~4 s of pooled eval audio, an eval after every epoch with the
    whole train subset as the probe (teacher-forced) and ~3 s of it decoded greedily, SpecAugment off. Everything the
    TensorBoard view of the overfit runs relies on is checked here."""
    m = load_script("04_distill")
    path = write_config(env, "ov-epochs", {
        "subset": {"train_audio_s": 8, "eval_audio_s": 4},
        "specaug": {"enabled": False},
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
    assert {"smoke_steps", "smoke_padded_row", "verdict", "logger_close"} <= {e["kind"] for e in ev}


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
