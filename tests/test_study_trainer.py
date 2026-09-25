"""The size study's trainer keys in scripts/04_distill.py (STUDY.md 2.1-2.5, CONTRACT.md section 2):

- bn.mode "train": BatchNorm trains (running stats move in the training forward, evals use them), every train_mode()
  call and BN check follows the mode, bn/max_abs_drift is reported instead of asserted, and the memory probe never
  falls back to gradient checkpointing; "frozen" (the default) still asserts
- loss.aux_ctc_weight: the training-only aux-CTC head (blank = V) on the encoder output, finite, in the optimizer and
  the full state (aux_ctc.pt), never in the exported weights, and a crash + resume reproduces the uninterrupted run
- optim.weight_decay: decay on the parameters of 2+ dimensions only (two param groups)
- the steps clock's warm-up check, and the LR curves of the three LR-probe lengths
- eval.full_at_fracs, ckpt.full_at_fracs / weights_at_fracs / upload_full_at "frac:<f>" (kept through rotation)
- branch: the T/2 branch reproduces the WSD schedule, the weights and the data order of a run with budget T/2
- specaug.seed: the masks are a pure function of (seed, step, micro-batch index), identical across a resume
- lr_probe: metrics only, and the lr_probe_result on the complete gate sets

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
import pandas as pd  # noqa: E402
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


def steps_of(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "metrics" / "steps.parquet")


def utts_of(run: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((run / "metrics" / "train_utts").glob("part-*.parquet"))])


def scalar(run: Path, tag: str) -> dict[int, float]:
    sc = pd.read_parquet(run / "metrics" / "scalars.parquet")
    sc = sc[sc["tag"] == tag].sort_values("wall", kind="stable")
    return dict(zip(sc["step"].astype(int), sc["value"]))


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


def tiny_batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([160, 97])
    feats = torch.randn(2, 163, 128, generator=g) * (torch.arange(163)[None, :] < lengths[:, None])[..., None]
    amask = torch.arange(163)[None, :] < lengths[:, None]
    dec = torch.tensor([[13764, 7, 4, 16, 20, 21], [13764, 7, 4, 16, 22, 2]]) % 512
    return dict(input_features=feats, attention_mask=amask, decoder_input_ids=dec,
                decoder_attention_mask=torch.tensor([[1] * 6, [1] * 5 + [0]]))


# ---------------------------------------------------------------------------------------------------- BatchNorm


def test_bn_train_mode_moves_running_stats_and_evals_use_them():
    """bn "train": the training forward normalises with batch statistics and updates the running stats; eval mode (the
    trainer's frozen_eval, as every eval and greedy decode) uses the running stats and gives every module its flag
    back. "frozen" (the default) pins BN in eval mode, and each mode's check fails on the other."""
    from kitsune.patches import assert_bn_frozen, assert_bn_mode, freeze_batchnorm, train_mode

    m = load_script("04_distill")
    model = tiny_bn_model()
    bns = [x for x in model.modules() if isinstance(x, nn.BatchNorm1d)]
    assert len(bns) == 2
    batch = tiny_batch()

    freeze_batchnorm(model)  # a frozen BN (a pruned student's) is unfrozen by the train mode, never the reverse
    train_mode(model, "train")
    assert all(b.training for b in bns) and assert_bn_mode(model, "train") == 2
    with pytest.raises(AssertionError, match="train mode"):
        assert_bn_frozen(model)
    with pytest.raises(AssertionError, match="train mode"):
        assert_bn_mode(model, "frozen")
    stats0 = [(b.running_mean.clone(), b.running_var.clone(), int(b.num_batches_tracked)) for b in bns]
    model(**batch, use_cache=False).logits.sum().backward()  # the training forward
    for b, (mu, var, n) in zip(bns, stats0):
        assert not torch.equal(b.running_mean, mu) and not torch.equal(b.running_var, var)
        assert int(b.num_batches_tracked) == n + 1 and b.weight.grad is not None
    moved = [(b.running_mean.clone(), b.running_var.clone()) for b in bns]

    # eval mode normalises with the running stats: a BN hooked on its input gives exactly (x - mean) / sqrt(var + eps)
    seen = {}
    bn0 = bns[0]
    hook = bn0.register_forward_hook(lambda mod, inp, out: seen.update(x=inp[0].detach(), y=out.detach()))
    with torch.no_grad(), m.frozen_eval(model):
        model(**batch, use_cache=False)
    hook.remove()
    x, y = seen["x"], seen["y"]
    want = ((x - moved[0][0][None, :, None]) / torch.sqrt(moved[0][1][None, :, None] + bn0.eps)
            * bn0.weight[None, :, None] + bn0.bias[None, :, None])
    torch.testing.assert_close(y, want)
    assert all(torch.equal(b.running_mean, mm) for b, (mm, _) in zip(bns, moved))  # no update in eval mode
    assert all(b.training for b in bns) and assert_bn_mode(model, "train") == 2  # the flags came back

    train_mode(model, "frozen")  # the default: frozen again, and its check holds
    assert assert_bn_mode(model, "frozen") == 2 and not any(b.training for b in bns)
    model.train()
    assert_bn_frozen(model)
    with pytest.raises(AssertionError, match="not training"):
        assert_bn_mode(model, "train")
    with pytest.raises(ValueError):
        train_mode(model, "recal")


def test_bn_drift_check_reports_in_train_mode_and_asserts_when_frozen():
    """_bn_drift_check: bn/max_abs_drift of the running stats from the init. "frozen": any drift is an AssertionError
    (the default, as before); "train": the drift is logged and the run goes on."""
    from kitsune.patches import train_mode

    m = load_script("04_distill")
    for mode in ("train", "frozen"):
        model = tiny_bn_model()
        train_mode(model, mode)
        bn0 = {n: (x.running_mean.detach().clone(), x.running_var.detach().clone())
               for n, x in model.named_modules() if isinstance(x, nn.BatchNorm1d)}
        logged = []
        R = SimpleNamespace(cfg=m.load_config(None, [f"bn.mode={mode}"]), model=model, bn0=bn0,
                            log=SimpleNamespace(scalars=lambda row, step: logged.append((step, dict(row)))))
        m._bn_drift_check(R, 1)
        assert logged[-1] == (1, {"bn/max_abs_drift": 0.0, "bn/modules": 2})
        with torch.no_grad():
            next(x for x in model.modules() if isinstance(x, nn.BatchNorm1d)).running_mean.add_(0.25)
        if mode == "train":
            m._bn_drift_check(R, 2)
            assert logged[-1][1]["bn/max_abs_drift"] == pytest.approx(0.25) and logged[-1][0] == 2
        else:
            with pytest.raises(AssertionError, match="although BN is frozen"):
                m._bn_drift_check(R, 2)


def test_bn_train_never_uses_gradient_checkpointing():
    """The memory probe's fallbacks after an OOM: halve micro_audio_s down to memory.min_micro_audio_s, then gradient
    checkpointing from the configured size - never under bn.mode "train" (the recomputed forward would update the
    running stats twice), which fails below the minimum instead; and grad_ckpt true is refused with it."""
    m = load_script("04_distill")

    def walk(cfg):
        seq, micro, ckpt = [], 1600.0, False
        while (nxt := m.probe_fallback(cfg, micro, 1600.0, ckpt)) is not None:
            micro, ckpt = nxt
            seq.append(nxt)
        return seq

    frozen = m.load_config(None, ["memory.min_micro_audio_s=200"])
    assert walk(frozen) == [(800.0, False), (400.0, False), (200.0, False), (1600.0, True), (800.0, True),
                            (400.0, True), (200.0, True)]
    train = m.load_config(None, ["memory.min_micro_audio_s=200", "bn.mode=train"])
    assert walk(train) == [(800.0, False), (400.0, False), (200.0, False)]
    assert m.load_config(None, ["bn.mode=train", "memory.grad_ckpt=false"])["memory"]["grad_ckpt"] is False
    with pytest.raises(SystemExit, match="never uses gradient checkpointing"):
        m.load_config(None, ["bn.mode=train", "memory.grad_ckpt=true"])
    for bad in (["bn.mode=recal"], ["bn.momentum=0"], ["bn.momentum=1.5"], ["bn.momentum=x"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
    assert m.load_config(None, ["bn.momentum=0.3"])["bn"]["momentum"] == 0.3


def test_setup_model_honours_the_bn_mode_and_momentum(env):
    """setup_model puts BN in the mode it is given (the trainer passes bn.mode, with or without gradient
    checkpointing; scripts/05_evaluate.py keeps the default "frozen"), and bn.momentum reaches every BN module."""
    from kitsune.patches import assert_bn_mode

    m = load_script("04_distill")
    cfg = m.load_config(None, [f"student={env['student'].as_posix()}", "bn.momentum=0.3", "bn.mode=train"])
    for mode, ckpt in (("train", False), ("frozen", True), ("frozen", False)):
        R = m.Run(cfg=cfg, run_dir=env["root"], device=torch.device("cpu"), amp=False)
        m.setup_model(R, ckpt, mode)
        bns = [x for x in R.model.modules() if isinstance(x, nn.BatchNorm1d)]
        assert bns and assert_bn_mode(R.model, mode) == len(bns)
        assert all(b.training == (mode == "train") for b in bns) and all(b.momentum == 0.3 for b in bns)
    R = m.Run(cfg=m.load_config(None, [f"student={env['student'].as_posix()}"]), run_dir=env["root"],
              device=torch.device("cpu"), amp=False)
    m.setup_model(R, grad_ckpt=False)  # 05_evaluate's call
    assert assert_bn_mode(R.model, "frozen") and all(b.momentum == 0.1 for b in R.model.modules()
                                                     if isinstance(b, nn.BatchNorm1d))


# ---------------------------------------------------------------------------------------------------- aux CTC


def test_aux_ctc_targets_and_loss():
    """The targets are the teacher's greedy ids per row without EOS (row order kept); the loss sums F.ctc_loss over the
    batch in fp32 with the blank the head's last class, is finite, has a gradient, and gives 0 (zero_infinity) for an
    utterance with more targets than frames."""
    from kitsune.kd import aux_ctc_loss, aux_ctc_targets

    greedy = torch.tensor([5, 6, 6, 3, 7, 3, 9, 9, 9])  # row 0: 5 6 6 EOS, row 1: 7 EOS, row 2 (truncated): 9 9 9
    rows = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])
    t, tl = aux_ctc_targets(greedy, rows, 3, eos=3)
    assert t.tolist() == [5, 6, 6, 7, 9, 9, 9] and tl.tolist() == [3, 1, 3] and t.dtype == torch.long
    V = 12
    torch.manual_seed(0)
    logits = torch.randn(3, 8, V + 1, requires_grad=True)
    frames = torch.tensor([8, 5, 8])
    loss = aux_ctc_loss(logits, frames, t, tl, blank=V)
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    want = sum(torch.nn.functional.ctc_loss(lp[:, b:b + 1], t[int(tl[:b].sum()):int(tl[:b + 1].sum())][None],
                                            frames[b:b + 1], tl[b:b + 1], blank=V, reduction="sum")
               for b in range(3))
    value = float(loss.detach())
    assert math.isfinite(value) and value > 0 and value == pytest.approx(float(want.detach()), rel=1e-5)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    # row 2 cannot fit 9 9 9 (5 frames needed) in 3 frames: it adds 0, the others count as before
    short = aux_ctc_loss(logits.detach(), torch.tensor([8, 5, 3]), t, tl, blank=V)
    only01 = aux_ctc_loss(logits.detach()[:2], frames[:2], t[:4], tl[:2], blank=V)
    assert float(short) == pytest.approx(float(only01), rel=1e-5)


def test_memory_probe_includes_the_aux_ctc_logits(monkeypatch):
    """With the aux-CTC head, the memory probe also runs the micro-batch with the most padded encoder frames (its
    (frames x V + 1) logits); without it, the three probes as before."""
    from kitsune import trainset

    m = load_script("04_distill")
    utts = [trainset.Utt(id=f"u{i}", source="s", duration=d, n_tok=5, audio_off=0, audio_len=0, tok_off=0)
            for i, d in enumerate([4.9, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 2.4, 2.4, 0.5, 0.5, 0.5])]
    seen = []
    monkeypatch.setattr(m, "fwd_bwd", lambda R, mb: seen.append(list(mb)) or (0.0, True))

    class Rows:
        def __getitem__(self, idx):
            return list(idx)

    planner = trainset.StepPlanner(utts, step_audio_s=1000.0, micro_audio_s=5.0, pool_micro=4, seed=0)
    for aux in (None, nn.Linear(2, 3)):
        seen.clear()
        R = SimpleNamespace(device=torch.device("cpu"), ds=Rows(), params=[], aux_ctc=aux)
        rec = dict(peak_gb={})
        m.probe_passes(R, planner, rec)
        mbs = [mb for step in planner.epoch_plan(0) for mb in step]
        most = max(float(planner.dur[mb].max()) * len(mb) for mb in mbs)
        assert len(seen) == (3 if aux is None else 4)
        if aux is not None:
            assert float(planner.dur[seen[-1]].max()) * len(seen[-1]) == most


def test_aux_ctc_run_trains_saves_and_resumes_exactly(env, monkeypatch):
    """A from-scratch student's settings on a tiny CPU run - bn.mode train, aux CTC 0.3, weight decay 1e-3: finite
    loss/aux_ctc every step (in loss/total, not in the objective), the BN running stats move (bn/max_abs_drift > 0, no
    assertion), two param groups; the head is in the optimizer and the full state (aux_ctc.pt) but not in the exported
    weights, which load as a student. A crash at step 10 resumed from full_step_8 ends with exactly the uninterrupted
    run's weights, head and SpecAugment masks."""
    from safetensors import safe_open

    from kitsune import student as S

    m = load_script("04_distill")
    over = {"bn": {"mode": "train"}, "loss": {"aux_ctc_weight": 0.3, "l2sp_lambda": 0.0},
            "optim": {"weight_decay": 1e-3}, "schedule": {"max_steps": 12},
            "ckpt": {"full_every_steps": 4, "keep_local": 5}, "log": {"hist_every": 4, "layer_stats_every": 4}}
    assert m.main(["--config", write_config(env, "aux-ref", over)]) == 0
    ref = one_run(env["root"], "aux-ref")
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "10")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", write_config(env, "aux-crash", over)])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    crash = one_run(env["root"], "aux-crash")
    assert m.main(["--resume", str(crash / "checkpoints" / "full_step_8")]) == 0

    st = steps_of(ref)
    aux = st["loss/aux_ctc"].to_numpy()
    assert len(aux) == 12 and np.isfinite(aux).all() and (aux > 0).all()
    np.testing.assert_allclose(st["loss/total"], st["loss/objective"] + 0.3 * aux + st["loss/l2sp"], rtol=1e-6)
    ev = events(ref)
    head = next(e for e in ev if e["kind"] == "aux_ctc")
    assert head["classes"] == 16385 and head["blank"] == 16384 and head["d_enc"] == 64 and head["weight"] == 0.3
    assert next(e for e in ev if e["kind"] == "model")["bn_mode"] == "train"
    drift = scalar(ref, "bn/max_abs_drift")
    assert set(drift) == {4, 8, 12} and all(v > 0 for v in drift.values())
    assert "layers/grad_norm/aux_ctc" in set(pd.read_parquet(ref / "metrics" / "scalars.parquet")["tag"])

    full = ref / "checkpoints" / "full_step_12"
    head_sd = torch.load(full / "aux_ctc.pt", weights_only=True)
    assert head_sd["weight"].shape == (16385, 64) and head_sd["bias"].shape == (16385,)
    groups = torch.load(full / "optimizer.pt", weights_only=True)["param_groups"]
    assert [(g["decay"], g["weight_decay"]) for g in groups] == [(True, 1e-3), (False, 0.0)]
    n_opt = sum(len(g["params"]) for g in groups)
    model_sd = torch.load(full / "model.pt", weights_only=True)
    assert not any(k.startswith("aux_ctc") for k in model_sd)
    w = ref / "checkpoints" / "step_12"
    keys = set()
    for f in w.glob("*.safetensors"):
        with safe_open(str(f), "pt") as fh:
            keys |= set(fh.keys())
    assert keys and not any("aux" in k for k in keys)
    student = S.load_student(w, "cpu")
    trainable = [n for n, _ in student.named_parameters() if "pos_emb" not in n]  # the frozen sinusoids aside
    assert n_opt == len(trainable) + 2  # + the head's weight and bias
    init = S.load_student(env["student"], "cpu")
    bn_key = next(k for k in student.state_dict() if k.endswith("running_mean"))
    assert not torch.equal(student.state_dict()[bn_key], init.state_dict()[bn_key])  # BN trained

    # the crash + resume replays steps 9-12 exactly: weights, the aux head, the masks
    assert next(e for e in events(crash) if e["kind"] == "resumed")["at_step"] == 8
    got = crash / "checkpoints" / "full_step_12"
    for f in ("model.pt", "aux_ctc.pt"):
        a, b = torch.load(full / f, weights_only=True), torch.load(got / f, weights_only=True)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), f
    ua, ub = utts_of(ref), utts_of(crash)
    for step in (9, 10, 11, 12):
        a = ua[ua["step"] == step].sort_values("id")
        b = ub[(ub["step"] == step) & (ub["attempt"] == 1)].sort_values("id")
        assert len(a) and a["id"].tolist() == b["id"].tolist()
        assert a["masked_frac"].tolist() == b["masked_frac"].tolist()  # SpecAugment: (seed, step, micro-batch)
    sa, sb = steps_of(ref).set_index("step"), steps_of(crash).drop_duplicates("step", keep="last").set_index("step")
    np.testing.assert_array_equal(sa["loss/aux_ctc"].to_numpy(), sb["loss/aux_ctc"].to_numpy())


def test_resume_cannot_change_the_study_keys(env):
    """bn.mode, loss.aux_ctc_weight and specaug.seed shape what the model and its loss are: a resume that changes one
    stops, naming it; repeating the checkpoint's value resumes."""
    m = load_script("04_distill")
    assert {"bn.mode", "loss.aux_ctc_weight", "specaug.seed", "family"} <= set(m.RESUME_FIXED)
    saved = m.load_config(None, [])
    for key, value in (("bn.mode", "train"), ("loss.aux_ctc_weight", 0.3), ("specaug.seed", 7)):
        with pytest.raises(SystemExit, match=key.replace(".", r"\.")):
            m.resume_overrides(saved, [(key, value)])
    assert m.resume_overrides(saved, [("bn.mode", "frozen"), ("loss.aux_ctc_weight", 0.0)])[0] == {}


# ---------------------------------------------------------------------------------------------------- weight decay


@pytest.mark.parametrize("offload", ["none", "cpu"])
def test_weight_decay_only_on_matrices(offload):
    """optim.weight_decay > 0: two param groups, the decay on the parameters of 2+ dimensions only (matrices,
    embeddings, conv kernels); biases and norm affine never decay. 0 keeps the one group of before. A resume's
    --set optim.weight_decay moves the decaying group's value only."""
    m = load_script("04_distill")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3), nn.Conv1d(3, 2, 3), nn.BatchNorm1d(2))
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]

    def make(wd, st=None):
        R = SimpleNamespace(cfg=m.load_config(None, [f"optim.weight_decay={wd}", f"optim.offload={offload}",
                                                     "loss.l2sp_lambda=0"]),
                            params=params, param_names=names, device=torch.device("cpu"), st=st or {},
                            log=SimpleNamespace(event=lambda *a, **k: None))
        m.setup_optim(R)
        return R

    R = make(0.1)
    assert R.st["optim_groups"] == "ndim"
    groups = R.opt.param_groups
    assert [(g["decay"], g["weight_decay"], len(g["params"])) for g in groups] == [(True, 0.1, 2), (False, 0.0, 6)]
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.zeros_like(p)
    for g in R.opt.param_groups:
        g["lr"] = 0.5
    R.opt.step()
    if offload == "cpu":
        R.opt.push()
    for n, p, b in zip(names, params, before):  # zero gradients: AdamW's step is the decay alone
        want = b * (1 - 0.5 * 0.1) if p.ndim >= 2 else b
        torch.testing.assert_close(p.detach(), want, msg=n)
    R.cfg["optim"]["weight_decay"] = 0.2  # a resume's --set
    m.apply_optim_hparams(R)
    assert [g["weight_decay"] for g in R.opt.param_groups] == [0.2, 0.0]

    one = make(0.0)
    assert one.st["optim_groups"] == "single" and len(one.opt.param_groups) == 1
    legacy = make(0.1, st={"optim_groups": "single"})  # a run started at 0 (or before the groups): one group stays
    assert len(legacy.opt.param_groups) == 1 and legacy.opt.param_groups[0]["weight_decay"] == 0.1
    for bad in (["optim.weight_decay=-1"], ["optim.weight_decay=x"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)


# ---------------------------------------------------------------------------------------------------- schedule


@pytest.mark.parametrize("M,warmup,stable,cooldown", [(5000, 2000, 2000, 1000), (3000, 1000, 1400, 600),
                                                      (2000, 300, 1300, 400)])
def test_lr_probe_lengths_reach_the_peak_and_cool_down(M, warmup, stable, cooldown):
    """The three LR-probe schedules (STUDY.md 2.3; cooldown_frac 0.2 of the steps clock, whose warm-up is not capped):
    the LR rises linearly to the peak at the warm-up's last step, stays there for the stable steps and cools over the
    last 20 % (1 - sqrt) to ~0; the draft's 2,000-step probe with a 2,000-step warm-up is refused."""
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


# ---------------------------------------------------------------------------------------------------- branch


def test_branch_is_the_t_half_run(env):
    """The T/2 branch of a 20-step parent (resume_frac 0.4, end_frac 0.5) against a run with max_steps 10: its LR
    curve is that run's (warm-up and stable LR the parent's, the cooldown over steps 9-10), it continues the parent's
    weights, optimizer and data order from full_step_8 (the same utterances per step, the same SpecAugment masks) and
    ends with that run's weights; one complete final eval; the `branch` event and summary.json name the parent. A
    config that differs from the parent's outside the free keys, or a parent without the 0.4 state, is refused."""
    m = load_script("04_distill")
    parent_over = {"ckpt": {"full_at_fracs": [0.4], "full_every_steps": 5, "keep_local": 1}}
    assert m.main(["--config", write_config(env, "par", parent_over)]) == 0
    parent = one_run(env["root"], "par")
    assert (parent / "checkpoints" / "full_step_8").is_dir()  # kept past keep_local 1 (full_step_10, 15, 20 later)
    assert m.main(["--config", write_config(env, "half-ref", {"schedule": {"max_steps": 10}})]) == 0
    half = one_run(env["root"], "half-ref")

    # the branch's config: the parent's, with its own run name kept, other checkpoints and a fraction eval
    branch_over = dict(parent_over, run_name="par", branch={"parent": str(parent)},
                       ckpt={"full_every_steps": 1000, "keep_local": 2}, eval={"full_at_fracs": None})
    path = env["root"] / "par-branch.json"
    path.write_text(json.dumps(merged(env["base"], branch_over), indent=1), encoding="utf-8")
    assert m.main(["--config", str(path)]) == 0
    branch = one_run(env["root"], "par-half")

    ev = events(branch)
    b = next(e for e in ev if e["kind"] == "branch")
    assert {k: b[k] for k in ("parent_run_id", "resume_step", "end_step", "t_c")} == dict(
        parent_run_id=parent.name, resume_step=8, end_step=10, t_c=8.0)
    s = json.loads((branch / "summary.json").read_text(encoding="utf-8"))
    assert s["branch"] == dict(parent_run_id=parent.name, resume_step=8, end_step=10, t_c=8.0)
    assert s["steps"] == 10 and s["resumes"] == 0 and s["config"]["run_name"] == "par-half"
    assert [(e["at_step"], e["complete"], e["final"]) for e in ev if e["kind"] == "eval"] == [(10, True, True)]
    assert not (branch / "checkpoints" / "full_step_8").exists()  # the parent's state is not saved again
    assert [(e["phase"], e["at_step"]) for e in ev if e["kind"] == "lr_phase"] == [("cooldown", 9)]

    # the schedule: the T/2 run's curve, step for step
    half_cfg = merged(env["base"], {"schedule": {"max_steps": 10}})
    want = lr_curve(m, half_cfg)
    got = pd.concat([steps_of(parent).query("step <= 8"), steps_of(branch)])
    assert got["step"].tolist() == list(range(1, 11))
    assert got["opt/lr"].tolist() == pytest.approx([x[0] for x in want], rel=1e-12, abs=0)
    assert got["sched/phase"].tolist() == [x[1] for x in want]
    np.testing.assert_array_equal(steps_of(half)["opt/lr"].to_numpy(), got["opt/lr"].to_numpy())

    # the data order and the masks go on from the parent's position; the weights end where the T/2 run's do
    ub, uh = utts_of(branch), utts_of(half)
    for step in (9, 10):
        a, c = ub[ub["step"] == step].sort_values("id"), uh[uh["step"] == step].sort_values("id")
        assert len(a) and a["id"].tolist() == c["id"].tolist()
        assert a["masked_frac"].tolist() == c["masked_frac"].tolist()
        np.testing.assert_allclose(a["kl"].to_numpy(), c["kl"].to_numpy(), rtol=1e-5, atol=1e-7)
    wa = torch.load(branch / "checkpoints" / "full_step_10" / "model.pt", weights_only=True)
    wh = torch.load(half / "checkpoints" / "full_step_10" / "model.pt", weights_only=True)
    for k in wa:
        torch.testing.assert_close(wa[k], wh[k], rtol=1e-5, atol=1e-6, msg=k)

    # refused: another LR (not a free key), and a parent without the state at round(resume_frac x M)
    bad = merged(env["base"], dict(branch_over, optim={"lr": 1e-3}))
    path.write_text(json.dumps(bad, indent=1), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"differs from this one in \['optim\.lr'\]"):
        m.main(["--config", str(path)])
    bad = merged(env["base"], dict(branch_over, branch={"parent": str(parent), "resume_frac": 0.3}))
    path.write_text(json.dumps(bad, indent=1), encoding="utf-8")
    with pytest.raises(SystemExit, match="no local full state at step 6"):
        m.main(["--config", str(path)])
    for bad in (["branch.resume_frac=0.5", "branch.end_frac=0.5"], ["branch.end_frac=1.2"],
                ["branch.parent=runs/x"]):  # the wall clock
        with pytest.raises(SystemExit):
            m.load_config(None, bad)


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


# ---------------------------------------------------------------------------------------------------- LR probe


def test_lr_probe_run(env, hub):
    """lr_probe.enabled on a tiny CPU run: metrics only (no step-0, in-loop or mini eval, no greedy decode, no weights,
    no checkpoint upload, though the config asks for evals, minis and uploads), the LR curve of its schedule, and at
    the end the teacher-forced objective w_kl x KL + w_ce x CE on the complete gate sets, pooled and per set: the
    lr_probe_result event and summary.json's lr_probe."""
    from kitsune import evaluate as ev

    m = load_script("04_distill")
    over = {"lr_probe": {"enabled": True}, "schedule": {"max_steps": 10, "warmup_steps": 4},
            "eval": {"every_steps": 2, "full_at_fracs": [0.5], "mini": {"every_steps": 3}},
            "ckpt": {"weights_every_steps": 2, "full_every_steps": 5, "upload_full_at": ["pre_cooldown", "end"]},
            "hf": {"output_repo": "fake-user/kitsune-runs"}}
    assert m.main(["--config", write_config(env, "probe", over)]) == 0
    run = one_run(env["root"], "probe")
    kinds = [e["kind"] for e in events(run)]
    assert "eval_mini" not in kinds and kinds.count("eval_start") == 1 and "eval" not in kinds and "verdict" not in kinds
    assert not list((run / "checkpoints").glob("step_*"))
    assert not [p for p, _ in hub.folders if "/checkpoints/" in p]  # metrics only
    assert any(p == f"runs/{run.name}" for p, _ in hub.folders)  # the logs still sync
    st = steps_of(run)
    cfg = merged(env["base"], {"schedule": {"max_steps": 10, "warmup_steps": 4}})
    assert st["opt/lr"].tolist() == pytest.approx([x[0] for x in lr_curve(m, cfg)], rel=1e-12, abs=0)

    res = next(e for e in events(run) if e["kind"] == "lr_probe_result")
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "complete" and s["verdict"]["verdict"] == "N/A"
    assert {k: s["lr_probe"][k] for k in ("objective", "per_set", "lr", "max_steps", "family")} == \
        {k: res[k] for k in ("objective", "per_set", "lr", "max_steps", "family")}
    assert res["lr"] == PEAK and res["max_steps"] == 10 and res["family"] == "aed" and set(res["per_set"]) == set(EVAL)
    saved = json.loads((run / "evals" / "step_10" / "lr_probe.json").read_text(encoding="utf-8"))
    tf = saved["tf"]
    assert tf["n_utts"] == 18  # the complete gate sets, 6 utterances each
    c = ev.combined_loss(tf, 1.0, 0.8)
    assert res["objective"] == pytest.approx(c["value"], rel=1e-12) and c["sets"] == sorted(EVAL)
    for x in EVAL:
        assert res["per_set"][x] == pytest.approx(tf["sets"][x]["kl"] + 0.8 * tf["sets"][x]["ce"], rel=1e-12)
    assert scalar(run, "combined_loss/val_full") == {10: pytest.approx(res["objective"])}
    for bad in (["lr_probe.enabled=true"], ["lr_probe.enabled=true", "schedule.clock=steps", "schedule.max_steps=10",
                                            "schedule.warmup_steps=1", 'eval_sets=["eval_jsut"]'],
                ["lr_probe.enabled=true", "schedule.clock=steps", "schedule.max_steps=10", "schedule.warmup_steps=1",
                 "subset.eval_utts_per_set=2"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
