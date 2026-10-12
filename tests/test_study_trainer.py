"""The size study's trainer keys in scripts/04_distill.py (STUDY.md 2.1-2.5, CONTRACT.md section 2) that stay:

- bn.mode "frozen", the only mode: train_mode pins BN, the drift check asserts it, the memory probe falls back to
  gradient checkpointing; the from-scratch students' "train" mode, bn.momentum and loss.aux_ctc_weight > 0 are refused
- optim.weight_decay: one param group, every parameter decayed
- the steps clock's warm-up check, and its LR curves at three lengths
- eval.full_at_fracs, ckpt.full_at_fracs / weights_at_fracs / upload_full_at "frac:<f>" (kept through rotation)
- specaug.seed: the masks are a pure function of (seed, step, micro-batch index), identical across a resume

CPU only (the laptop GPU runs other jobs), a tiny random student and a synthetic corpus in the real on-disk formats."""
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from fixtures import load_script, make_fake_corpus, make_fake_selection  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
PEAK, WARMUP, COOLDOWN, MAX_STEPS = 3e-3, 3, 0.2, 20


def events(run: Path, kind: str | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def merged(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merged(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def one_run(root: Path, name: str) -> Path:
    runs = list((root / "runs").glob(f"{name}-2*"))
    assert len(runs) == 1, runs
    return runs[0]


def lr_curve(m, cfg: dict, steps=None) -> list[tuple[float, int]]:
    """(lr, phase) of every optimizer step of a steps-clock config as loop() computes them: step s starts at t = s - 1
    steps done, out of T = max_steps, with the warm-up the schedule gives (not capped on this clock)."""
    sch = cfg["schedule"]
    M = int(sch["max_steps"])
    return [m.wsd_lr(float(cfg["optim"]["lr"]), s, float(s - 1), float(M), m.warmup_steps(sch),
                     float(sch["cooldown_frac"])) for s in (steps or range(1, M + 1))]


class FakeHub:
    """Stands in for huggingface_hub.HfApi: records every upload, serves upload_file bytes back."""

    def __init__(self):
        self.files, self.folders = {}, []

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False, **kw):
        pass

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None, **kw):
        self.files[path_in_repo] = bytes(path_or_fileobj)

    def upload_folder(self, *, repo_id, folder_path, path_in_repo=None, commit_message=None, repo_type=None, **kw):
        files = sorted(p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*") if p.is_file())
        self.folders.append((path_in_repo, files))

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.files[filename])
        return str(p)


@pytest.fixture(autouse=True)
def grad_enabled():
    """Autograd on at the start of every test here, as in a trainer process. An earlier test file can leave it off for
    the whole pytest process: scripts/03_build_student.py's feature_batches yields inside `with torch.no_grad()`, and
    kitsune.student.recalibrate_batchnorm (itself under no_grad) stops iterating it early, so the suspended generator's
    __exit__ restores grad mode to *off* whenever it is finalized (tests/test_student.py runs just before this file).
    gc.collect() first, so such a generator is finalized now and not in the middle of a test."""
    import gc

    gc.collect()
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(prev)


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    h = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: h)
    return h


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + a tiny student saved exactly as 03 saves one, and a base config: the steps clock
    (20 steps of ~6 audio-s in micro-batches of ~3 s, so a step has 1-3 of them), no smoke phase, no evals or
    checkpoints in the loop unless a test asks for them."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("study")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (36, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=21)
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
        "schedule": {"warmup_steps": WARMUP, "cooldown_frac": COOLDOWN, "clock": "steps", "max_steps": MAX_STEPS},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_min": None, "every_steps": None, "greedy_subset": 3, "batch_s": 20, "check_baselines": False},
        "ckpt": {"weights_every_min": None, "full_local_every_min": None, "keep_local": 2, "full_after_smoke": False,
                 "upload_full_at": []},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
    }
    return dict(root=root, base=base, fc=fc, student=sdir)


def write_config(env, name: str, over: dict) -> str:
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


def tiny_bn_model():
    """A tiny CohereAsr model (2 conformer layers, so 2 BatchNorm1d) with non-trivial BN statistics."""
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    enc = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
               subsampling_conv_channels=8)
    cfg = CohereAsrConfig(encoder_config=enc, vocab_size=512, hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=64)
    torch.manual_seed(0)
    m = CohereAsrForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.running_mean.normal_(0, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
    return m


# ---------------------------------------------------------------------------------------------------- BatchNorm


def test_bn_frozen_is_the_only_mode():
    """BN_MODES is ("frozen",) since the size study's "train" was retired: train_mode pins BN in eval mode across a
    later .train() (its affine still trainable), assert_bn_mode is assert_bn_frozen, and both refuse another mode; the
    config refuses bn.mode "train" and any bn.momentum but null."""
    from kitsune.patches import BN_MODES, assert_bn_frozen, assert_bn_mode, train_mode

    m = load_script("04_distill")
    assert BN_MODES == m.BN_MODES == ("frozen",)
    model = tiny_bn_model()
    bns = [x for x in model.modules() if isinstance(x, nn.BatchNorm1d)]
    assert len(bns) == 2
    train_mode(model, "frozen")
    model.train()
    assert not any(b.training for b in bns) and all(b.weight.requires_grad for b in bns)
    assert assert_bn_mode(model, "frozen") == assert_bn_frozen(model) == 2
    for bad in ("train", "recal"):
        with pytest.raises(ValueError):
            train_mode(model, bad)
        with pytest.raises(ValueError):
            assert_bn_mode(model, bad)
    with pytest.raises(SystemExit, match="bn.mode must be one of frozen"):
        m.load_config(None, ["bn.mode=train"])
    for bad in (["bn.momentum=0.3"], ["bn.momentum=0"], ["bn.momentum=x"]):
        with pytest.raises(SystemExit, match="bn.momentum must be null"):
            m.load_config(None, bad)
    assert m.load_config(None, [])["bn"] == {"mode": "frozen", "momentum": None}


def test_bn_drift_check_asserts_the_frozen_stats():
    """_bn_drift_check: bn/max_abs_drift of the running stats from the init, logged; BatchNorm is frozen for the whole
    run, so any drift is an AssertionError."""
    from kitsune.patches import train_mode

    m = load_script("04_distill")
    model = tiny_bn_model()
    train_mode(model, "frozen")
    bn0 = {n: (x.running_mean.detach().clone(), x.running_var.detach().clone())
           for n, x in model.named_modules() if isinstance(x, nn.BatchNorm1d)}
    logged = []
    R = SimpleNamespace(cfg=m.load_config(None, []), model=model, bn0=bn0,
                        log=SimpleNamespace(scalars=lambda row, step: logged.append((step, dict(row)))))
    m._bn_drift_check(R, 1)
    assert logged[-1] == (1, {"bn/max_abs_drift": 0.0, "bn/modules": 2})
    with torch.no_grad():
        next(x for x in model.modules() if isinstance(x, nn.BatchNorm1d)).running_mean.add_(0.25)
    with pytest.raises(AssertionError, match="although BN is frozen"):
        m._bn_drift_check(R, 2)


def test_the_memory_probe_halves_then_checkpoints():
    """The memory probe's fallbacks after an OOM: halve micro_audio_s down to memory.min_micro_audio_s, then gradient
    checkpointing from the configured size (memory.grad_ckpt "auto") and halve again; memory.grad_ckpt false, or a
    probe that started with gradient checkpointing on, halves only."""
    m = load_script("04_distill")

    def walk(cfg, ckpt=False):
        seq, micro = [], 1600.0
        while (nxt := m.probe_fallback(cfg, micro, 1600.0, ckpt)) is not None:
            micro, ckpt = nxt
            seq.append(nxt)
        return seq

    auto = m.load_config(None, ["memory.min_micro_audio_s=200"])
    assert walk(auto) == [(800.0, False), (400.0, False), (200.0, False), (1600.0, True), (800.0, True),
                          (400.0, True), (200.0, True)]
    assert walk(auto, ckpt=True) == [(800.0, True), (400.0, True), (200.0, True)]
    assert walk(m.load_config(None, ["memory.min_micro_audio_s=200", "memory.grad_ckpt=false"])) == [
        (800.0, False), (400.0, False), (200.0, False)]


def test_setup_model_freezes_the_bn(env):
    """setup_model puts BN in the mode it is given - "frozen", the trainer's bn.mode and scripts/05_evaluate.py's
    default -, with or without gradient checkpointing, and leaves every BN module's momentum as it is."""
    from kitsune.patches import assert_bn_mode

    m = load_script("04_distill")
    cfg = m.load_config(None, [f"student={env['student'].as_posix()}"])
    for ckpt in (True, False):
        R = m.Run(cfg=cfg, run_dir=env["root"], device=torch.device("cpu"), amp=False)
        m.setup_model(R, ckpt, m.bn_mode(cfg))
        bns = [x for x in R.model.modules() if isinstance(x, nn.BatchNorm1d)]
        assert bns and assert_bn_mode(R.model, "frozen") == len(bns) and not any(b.training for b in bns)
        assert all(b.momentum == 0.1 for b in bns)
    R = m.Run(cfg=cfg, run_dir=env["root"], device=torch.device("cpu"), amp=False)
    m.setup_model(R, grad_ckpt=False)  # 05_evaluate's call
    assert assert_bn_mode(R.model, "frozen")


# ------------------------------------------------------------------------------------ aux CTC and weight decay


def test_aux_ctc_weight_stays_zero():
    """loss.aux_ctc_weight: 0 (0.0, as every saved config spells it), the only value since the size study's aux-CTC
    head was retired; anything else is refused."""
    m = load_script("04_distill")
    assert m.load_config(None, ["loss.aux_ctc_weight=0"])["loss"]["aux_ctc_weight"] == 0
    for bad in ("0.3", "-1", "x", "null"):
        with pytest.raises(SystemExit, match="loss.aux_ctc_weight must be 0"):
            m.load_config(None, [f"loss.aux_ctc_weight={bad}"])


def test_resume_cannot_change_the_study_keys():
    """specaug.seed and family shape what the model and its loss are: a resume that changes one stops, naming it;
    repeating the checkpoint's value resumes."""
    m = load_script("04_distill")
    assert {"specaug.seed", "family"} <= set(m.RESUME_FIXED)
    saved = m.load_config(None, [])
    for key, value in (("specaug.seed", 7), ("family", "ctc")):
        with pytest.raises(SystemExit, match=key.replace(".", r"\.")):
            m.resume_overrides(saved, [(key, value)])
    assert m.resume_overrides(saved, [("specaug.seed", None), ("family", "aed")])[0] == {}


def test_weight_decay_is_one_group():
    """optim.weight_decay: AdamW's decoupled decay on every parameter, in one param group (the size study's groups of
    the 2+-dimension parameters are retired); a resume's --set optim.weight_decay moves it."""
    m = load_script("04_distill")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    R = SimpleNamespace(cfg=m.load_config(None, ["optim.weight_decay=0.1", "loss.l2sp_lambda=0"]), params=params,
                        param_names=names, device=torch.device("cpu"))
    m.setup_optim(R)
    assert [(g["weight_decay"], len(g["params"])) for g in R.opt.param_groups] == [(0.1, 4)]
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.zeros_like(p)
    R.opt.param_groups[0]["lr"] = 0.5
    R.opt.step()
    for n, p, b in zip(names, params, before):  # zero gradients: AdamW's step is the decay alone
        torch.testing.assert_close(p.detach(), b * (1 - 0.5 * 0.1), msg=n)
    R.cfg["optim"]["weight_decay"] = 0.2  # a resume's --set
    m.apply_optim_hparams(R)
    assert [g["weight_decay"] for g in R.opt.param_groups] == [0.2]
    for bad in (["optim.weight_decay=-1"], ["optim.weight_decay=x"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)


# ---------------------------------------------------------------------------------------------------- schedule


@pytest.mark.parametrize("M,warmup,stable,cooldown", [(5000, 2000, 2000, 1000), (3000, 1000, 1400, 600),
                                                      (2000, 300, 1300, 400)])
def test_the_steps_clock_reaches_the_peak_and_cools_down(M, warmup, stable, cooldown):
    """Three steps-clock schedules (the size study's LR-probe lengths, STUDY.md 2.3; cooldown_frac 0.2 of the steps
    clock, whose warm-up is not capped): the LR rises linearly to the peak at the warm-up's last step, stays there for
    the stable steps and cools over the last 20 % (1 - sqrt) to ~0."""
    m = load_script("04_distill")
    cfg = m.load_config(None, ["schedule.clock=steps", f"schedule.max_steps={M}", f"schedule.warmup_steps={warmup}",
                               "schedule.cooldown_frac=0.2", "optim.lr=1e-3"])
    assert m.warmup_steps(cfg["schedule"]) == warmup  # not capped on this clock
    curve = lr_curve(m, cfg)
    lr = np.array([x[0] for x in curve])
    phase = np.array([x[1] for x in curve])
    assert (phase == 0).sum() == warmup - 1 and (phase == 1).sum() == stable + 1 and (phase == 2).sum() == cooldown
    assert lr[warmup - 1] == 1e-3 and lr[0] == pytest.approx(1e-3 / warmup)
    assert np.all(np.diff(lr[:warmup]) > 0) and np.all(lr[warmup - 1:warmup + stable] == 1e-3)
    cool = lr[warmup + stable:]
    assert cool[0] == 1e-3 and np.all(np.diff(cool) < 0) and cool[-1] == pytest.approx(1e-3 * (1 - math.sqrt(
        (cooldown - 1) / cooldown)))
    assert cool[-1] < 0.05 * 1e-3 and phase[warmup + stable] == 2


def test_the_warmup_must_end_before_the_cooldown():
    """Steps clock: warmup_steps >= (1 - cooldown_frac) x max_steps is refused (wsd_lr multiplies the cooldown by the
    warm-up factor: such a run never reaches its peak LR); the wall and epoch clocks are unchanged."""
    m = load_script("04_distill")
    for M, w in ((2000, 2000), (2000, 1600), (20, 300)):
        with pytest.raises(SystemExit, match="never reach its peak LR"):
            m.load_config(None, ["schedule.clock=steps", f"schedule.max_steps={M}", f"schedule.warmup_steps={w}"])
    m.load_config(None, ["schedule.clock=steps", "schedule.max_steps=2000", "schedule.warmup_steps=1599"])
    m.load_config(None, ["schedule.clock=epochs", "schedule.epochs=1", "schedule.warmup_steps=300"])  # capped there
    m.load_config(None, [])  # the wall clock
    for bad in (["schedule.warmup_steps=-1"], ["schedule.warmup_steps=2.5"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)


# ---------------------------------------------------------------------------------------------- fraction evals


def test_fraction_keys_are_validated():
    m = load_script("04_distill")
    steps = ["schedule.clock=steps", "schedule.max_steps=100", "schedule.warmup_steps=10"]
    ok = m.load_config(None, steps + ["eval.full_at_fracs=[0.2,0.4,0.6,0.8]", "ckpt.full_at_fracs=[0.8,0.4]",
                                      "ckpt.weights_at_fracs=[0.4]",
                                      'ckpt.upload_full_at=["pre_cooldown","end","frac:0.8"]'])
    assert m.frac_steps(ok["eval"]["full_at_fracs"], 100) == {20: 0.2, 40: 0.4, 60: 0.6, 80: 0.8}
    assert m.frac_steps(None, 100) == {} and m.frac_steps([0.5], None) == {}
    assert m.frac_step(0.4, 9366) == 3746 and m.frac_step(0.5, 9366) == 4683
    for bad in (["eval.full_at_fracs=[0.4,0.2]"], ["eval.full_at_fracs=[0]"], ["eval.full_at_fracs=[1.0]"],
                ["eval.full_at_fracs=[]"], ["eval.full_at_fracs=0.4"], ["ckpt.full_at_fracs=[0.4,0.4]"],
                ['ckpt.upload_full_at=["frac:0.5"]'], ['ckpt.upload_full_at=["later"]'],
                ["ckpt.weights_at_fracs=[1.5]"]):
        with pytest.raises(SystemExit):
            m.load_config(None, steps + bad)
    with pytest.raises(SystemExit, match="needs schedule.clock 'steps'"):
        m.load_config(None, ["eval.full_at_fracs=[0.5]"])  # the wall clock has no max_steps to take fractions of


def test_fraction_evals_and_checkpoints(env, hub):
    """eval.full_at_fracs: a complete eval right after step round(f x max_steps) next to the final one;
    ckpt.full_at_fracs: a full state there that keep_local's rotation never deletes (and "frac:<f>" uploads it);
    ckpt.weights_at_fracs: exported weights there, uploaded like step_*."""
    m = load_script("04_distill")
    over = {"eval": {"full_at_fracs": [0.25, 0.5]},
            "ckpt": {"full_at_fracs": [0.3], "weights_at_fracs": [0.5], "full_every_steps": 2, "keep_local": 1,
                     "upload_full_at": ["frac:0.3"]},
            "hf": {"output_repo": "fake-user/kitsune-runs"}}
    assert m.main(["--config", write_config(env, "fracs", over)]) == 0
    run = one_run(env["root"], "fracs")
    ev = events(run, "eval")
    assert [(e["at_step"], e["complete"], e["final"]) for e in ev] == [(0, False, False), (5, True, False),
                                                                       (10, True, False), (20, True, True)]
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert [r["step"] for r in s["history"]] == [0, 5, 10, 20] and "greedy_full" in s["history"][1]
    ck = run / "checkpoints"
    assert sorted(p.name for p in ck.glob("full_step_*")) == ["full_step_20", "full_step_6"]  # 6 = round(0.3 x 20)
    assert sorted(p.name for p in ck.glob("step_*")) == ["step_10", "step_20"]
    assert json.loads((ck / "full_step_6" / "trainer.json").read_text(encoding="utf-8"))["reason"] == "frac:0.3"
    uploaded = [p for p, _ in hub.folders if "/checkpoints/" in p]
    assert f"runs/{run.name}/checkpoints/full_step_6" in uploaded and f"runs/{run.name}/checkpoints/step_10" in uploaded
    assert f"runs/{run.name}/checkpoints/full_step_4" not in uploaded
    assert not (ck / "full_step_6" / m.UPLOAD_MARK).exists()  # on the Hub now
    deleted = {e["name"] for e in events(run, "checkpoint_deleted")}
    assert "full_step_6" not in deleted and {"full_step_2", "full_step_4", "full_step_8"} <= deleted


def test_rotation_keeps_the_fraction_states(tmp_path):
    m = load_script("04_distill")
    ck = tmp_path / "checkpoints"
    for n in (2, 4, 6, 8):
        (ck / f"full_step_{n}").mkdir(parents=True)
    evs = []
    R = SimpleNamespace(cfg={"ckpt": {"keep_local": 1}}, ckpt_dir=ck, uploader=SimpleNamespace(busy=set),
                        st={"fulls_kept": ["full_step_4"]},
                        log=SimpleNamespace(event=lambda kind, **kw: evs.append(kw["name"])))
    m.rotate_full(R)
    assert sorted(p.name for p in ck.iterdir()) == ["full_step_4", "full_step_8"] and evs == ["full_step_2",
                                                                                              "full_step_6"]


# ---------------------------------------------------------------------------------------------------- SpecAugment


def test_specaug_seed_is_a_pure_function_of_seed_step_and_micro_batch():
    """specaug_seed mixes (specaug.seed or the run's seed, step, micro-batch index): the same masks for the same three
    whatever else happened, different ones for another step, micro-batch or seed; specaug.seed overrides the run's."""
    from kitsune.features import SpecAugment

    m = load_script("04_distill")
    cfg = m.load_config(None, [])
    s = m.specaug_seed(cfg, 7, 1)
    assert s == m.specaug_seed(copy.deepcopy(cfg), 7, 1) and 0 <= s < 2**64
    assert len({s, m.specaug_seed(cfg, 8, 1), m.specaug_seed(cfg, 7, 0), m.specaug_seed(cfg, 7, 2)}) == 4
    assert m.specaug_seed(m.load_config(None, ["specaug.seed=1234"]), 7, 1) == s  # null = the run's seed (1234)
    assert m.specaug_seed(m.load_config(None, ["specaug.seed=5"]), 7, 1) != s
    assert m.specaug_seed(m.load_config(None, ["seed=5"]), 7, 1) == m.specaug_seed(
        m.load_config(None, ["specaug.seed=5"]), 7, 1)
    for bad in (["specaug.seed=-1"], ["specaug.seed=1.5"], ["specaug.seed=x"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
    aug = SpecAugment()
    feats, mask = torch.randn(3, 200, 128), torch.ones(3, 200, dtype=torch.bool)
    g = torch.Generator()
    outs = []
    for _ in range(2):
        g.manual_seed(s)
        outs.append(aug(feats, mask, g)[0])
        torch.rand(5, generator=g)  # whatever the generator did afterwards
    assert torch.equal(outs[0], outs[1])
