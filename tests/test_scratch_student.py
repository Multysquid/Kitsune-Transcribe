"""From-scratch students (kitsune.student.build_scratch_student, scripts/03_build_student.py --scratch).

The size study's T-0.1B, T-0.05B, the replicate and the bridge are the teacher's architecture at other widths, built
from a random init. Every trap STUDY.md 1.2 lists is pinned here: the counts of the three named shapes (meta device),
student_config before the widths (tied head), explicit head_dim and kv heads, pos_emb in the teacher's sinusoid form,
the subsampling convs re-initialised with the PyTorch default, fresh BatchNorm, the teacher's generation ids, the
seed, the count assert. Tiny shapes on CPU otherwise; the real teacher config/weights are only read when cached."""
import glob
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoConfig, CohereAsrConfig, CohereAsrForConditionalGeneration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitsune import student as S  # noqa: E402

PROMPT = [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13]
TINY = S.ScratchShape(enc_hidden=64, enc_layers=2, enc_heads=2, enc_ffn=128, dec_hidden=48, dec_layers=2, dec_heads=3,
                      dec_ffn=96)


def teacher_config() -> CohereAsrConfig:
    """The real teacher config (NeMo extras, head_dim 128, 8 kv heads, untied) if cached; else CohereAsrConfig(),
    whose defaults are the teacher's HF fields."""
    try:
        return AutoConfig.from_pretrained(S.TEACHER_ID, local_files_only=True)
    except Exception:
        return CohereAsrConfig()


def meta_model(cfg: CohereAsrConfig) -> CohereAsrForConditionalGeneration:
    with torch.device("meta"):
        return CohereAsrForConditionalGeneration(cfg)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def feats(lengths: list[int], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised-log-mel-like features (B, T, 128), zero past each length, and the int64 mask."""
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    mask = (torch.arange(T)[None, :] < torch.tensor(lengths)[:, None]).long()
    return torch.randn(len(lengths), T, 128, generator=g) * mask[..., None], mask


# ------------------------------------------------------------------------------------------------ counts and traps


@pytest.mark.parametrize("name,total,non_emb", [("t01", 103_996_416, 95_083_520), ("t005", 51_209_600, 44_524_928),
                                                ("bridge", 320_752_384, 302_926_592)])
def test_named_shapes_have_the_studys_counts(name, total, non_emb):
    """STUDY.md 1.1, exactly: total (tied head counted once) and non-embedding (minus embed and pos_emb)."""
    assert S.SCRATCH_EXPECTED_PARAMS[name] == total
    cfg = S.scratch_config(teacher_config(), S.SCRATCH_SHAPES[name])
    m = meta_model(cfg)
    assert m.proj_out.weight is m.model.decoder.embed_tokens.weight
    assert n_params(m) == S.closed_form_params(cfg) == total
    assert S.non_embedding_params(m) == non_emb
    assert S.count_fields(m) == dict(params_total=total, params_non_embedding=non_emb, closed_form_params=total)
    sub = n_params(m.model.encoder.subsampling)
    if name in ("t01", "t005"):  # the fixed parts STUDY.md quotes
        assert sub == {"t01": 2_236_928, "t005": 1_712_512}[name]
        assert m.model.decoder.embed_tokens.weight.numel() == {"t01": 8_388_608, "t005": 6_291_456}[name]
        assert total - m.model.decoder.pos_emb.weight.numel() == {"t01": 103_472_128, "t005": 50_816_384}[name]


def test_the_bridge_is_the_t03_shape():
    """The bridge config is exactly T-0.3B's (B10x2560 + decoder {0,7}): only the init differs."""
    tc = teacher_config()
    t03 = S.student_config(tc, S.StudentSpec(S.evenly_spaced(10, 48), 2560, [0, 7]))
    assert S.scratch_config(tc, S.SCRATCH_SHAPES["bridge"]).to_dict() == t03.to_dict()
    assert S.PRUNED_EXPECTED_PARAMS[(10, 2560, (0, 7))] == S.SCRATCH_EXPECTED_PARAMS["bridge"]
    assert n_params(meta_model(t03)) == 320_752_384


def _naive(tc: CohereAsrConfig, shape: S.ScratchShape) -> CohereAsrConfig:
    """student_config, then only hidden/heads/FFN/depth replaced: head_dim and kv heads left as serialised."""
    c = S.student_config(tc, S.StudentSpec([0], tc.encoder_config.intermediate_size, [0]))
    e = c.encoder_config
    e.num_hidden_layers, e.hidden_size, e.intermediate_size = shape.enc_layers, shape.enc_hidden, shape.enc_ffn
    e.num_attention_heads = shape.enc_heads
    c.num_hidden_layers, c.hidden_size, c.intermediate_size = shape.dec_layers, shape.dec_hidden, shape.dec_ffn
    c.num_attention_heads = shape.dec_heads
    return c


def test_head_dim_and_kv_traps():
    """The serialised head_dim 128 silently widens T-0.1B's attention (112,397,312 params); 6 heads with the
    serialised 8 kv heads give num_key_value_groups 0 (T-0.05B). scratch_config sets both explicitly."""
    tc = teacher_config()
    naive01 = meta_model(_naive(tc, S.SCRATCH_SHAPES["t01"]))
    assert n_params(naive01) == 112_397_312 and naive01.model.decoder.layers[0].self_attn.head_dim == 128
    naive005 = meta_model(_naive(tc, S.SCRATCH_SHAPES["t005"]))
    assert naive005.model.decoder.layers[0].self_attn.num_key_value_groups == 0
    assert naive005.model.encoder.layers[0].self_attn.num_key_value_groups == 0
    for name, shape in S.SCRATCH_SHAPES.items():
        cfg = S.scratch_config(tc, shape)
        assert cfg.head_dim == shape.dec_hidden // shape.dec_heads
        assert cfg.num_key_value_heads == cfg.num_attention_heads == shape.dec_heads
        assert cfg.encoder_config.num_key_value_heads == cfg.encoder_config.num_attention_heads == shape.enc_heads
        m = meta_model(cfg)
        for a in (m.model.encoder.layers[0].self_attn, m.model.decoder.layers[0].self_attn,
                  m.model.decoder.layers[0].encoder_attn):
            assert a.num_key_value_groups == 1, name
        assert m.model.encoder.layers[0].self_attn.head_dim == shape.enc_hidden // shape.enc_heads


def test_skipping_student_config_unties_the_head():
    """The raw teacher config has tie_word_embeddings False: a second V x D matrix (+8 % / +12 %). The build goes
    through student_config, so the head is tied and the count is the study's."""
    tc = teacher_config()
    for name, untied in (("t01", 112_385_024), ("t005", 57_501_056)):
        cfg = S.scratch_config(tc, S.SCRATCH_SHAPES[name])
        assert cfg.tie_word_embeddings and cfg.decoder_start_token_id == 13764
        cfg.tie_word_embeddings = False  # what skipping student_config leaves
        assert n_params(meta_model(cfg)) == S.closed_form_params(cfg) == untied != S.SCRATCH_EXPECTED_PARAMS[name]


def test_scratch_config_pins_the_study_settings():
    tc = teacher_config()
    cfg = S.scratch_config(tc, S.SCRATCH_SHAPES["t005"])
    e = cfg.encoder_config
    assert (e.hidden_size, e.num_hidden_layers, e.intermediate_size, e.conv_kernel_size) == (384, 10, 1536, 9)
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size) == (384, 3, 1536)
    assert e.scale_input is False and e.subsampling_conv_channels == 256 and e.num_mel_bins == 128
    assert all(getattr(e, k) == 0.0 for k in ("dropout", "dropout_positions", "layerdrop", "activation_dropout",
                                              "attention_dropout")) and cfg.attention_dropout == 0.0
    assert cfg._attn_implementation == "sdpa" and e._attn_implementation == "sdpa"
    assert cfg.vocab_size == 16384 and cfg.max_position_embeddings == 1024
    assert not hasattr(cfg, "transf_decoder") and not hasattr(cfg, "encoder")  # NeMo shape keys dropped
    assert tc.head_dim == 128 or tc.to_dict().get("head_dim") == 128  # the trap is real in the source config
    with pytest.raises(ValueError):
        S.ScratchShape(100, 2, 3, 64, 64, 1, 2, 64)  # 100 % 3
    with pytest.raises(ValueError):
        S.ScratchShape(64, 2, 2, 64, 64, 1, 2, 64, conv_kernel=8)


# ------------------------------------------------------------------------------------------------ the build


def test_build_scratch_student_tiny():
    tc = teacher_config()
    m = S.build_scratch_student(tc, TINY, seed=1234)
    cfg = m.config
    assert not m.training and all(p.dtype == torch.float32 and p.device.type == "cpu" for p in m.parameters())
    assert n_params(m) == S.closed_form_params(cfg)
    # tied head, the teacher's ids
    assert m.proj_out.weight is m.model.decoder.embed_tokens.weight
    gc = m.generation_config
    assert (gc.decoder_start_token_id, gc.eos_token_id, gc.pad_token_id, gc.bos_token_id) == (13764, 3, 2, 4)
    assert cfg.decoder_start_token_id == 13764 and (cfg.eos_token_id, cfg.pad_token_id, cfg.bos_token_id) == (3, 2, 4)
    # pos_emb: the teacher's sinusoid / sqrt(D) form, exactly
    assert torch.equal(m.model.decoder.pos_emb.weight, S.sinusoid_pos_emb(1024, 48))
    # fresh BatchNorm
    bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm1d)]
    assert len(bns) == TINY.enc_layers
    for bn in bns:
        assert torch.equal(bn.running_mean, torch.zeros(64)) and torch.equal(bn.running_var, torch.ones(64))
        assert int(bn.num_batches_tracked) == 0
        assert torch.equal(bn.weight, torch.ones(64)) and torch.equal(bn.bias, torch.zeros(64))
    # only the subsampling convs left HF's N(0, 0.02): the linear after them and the attention keep it
    assert abs(float(m.model.encoder.subsampling.linear.weight.std()) - 0.02) < 0.002
    assert abs(float(m.model.encoder.layers[0].self_attn.q_proj.weight.std()) - 0.02) < 0.004
    assert m.model.encoder.input_scale == 1.0
    # generate through the 2-layer cache
    x, mask = feats([160, 97])
    out = m.generate(input_features=x, attention_mask=mask, decoder_input_ids=torch.tensor([PROMPT] * 2),
                     max_new_tokens=3, min_new_tokens=3, do_sample=False, num_beams=1)
    assert out.shape == (2, len(PROMPT) + 3)


def test_subsampling_conv_reinit_rms():
    """Each subsampling Conv2d has the PyTorch default init (weights U(+-1/sqrt(fan_in)), std 1/sqrt(3 fan_in));
    the subsampling output at init is ~1e-1, not HF's ~1e-5, and the gradient at init is not dominated by it."""
    tc = teacher_config()
    m = S.build_scratch_student(tc, TINY, seed=3)
    convs = S._subsampling_convs(m)
    assert len(convs) == 5  # 3x3 stride 2, then 2 x (depthwise 3x3 stride 2, pointwise 1x1)
    for c in convs:
        fan_in = c.weight[0].numel()
        bound = 1 / math.sqrt(fan_in)
        assert float(c.weight.abs().max()) <= bound and float(c.bias.abs().max()) <= bound
        assert abs(float(c.weight.std()) / math.sqrt(1 / (3 * fan_in)) - 1) < 0.1, (tuple(c.weight.shape), fan_in)
    torch.manual_seed(3)
    hf = CohereAsrForConditionalGeneration(S.scratch_config(tc, TINY)).eval()  # what HF's init alone gives
    x, mask = feats([300, 250, 200, 150], seed=1)
    with torch.no_grad():
        rms = float(m.model.encoder.subsampling(x, mask).pow(2).mean().sqrt())
        rms_hf = float(hf.model.encoder.subsampling(x, mask).pow(2).mean().sqrt())
    assert rms > 0.03 and rms_hf < 1e-4, (rms, rms_hf)

    def grad_norm(model):
        model.train()
        g = torch.Generator().manual_seed(2)
        lab = torch.randint(5, 16384, (4, 12), generator=g)
        dec = torch.cat([torch.full((4, 1), 13764), lab[:, :-1]], 1)
        logits = model(input_features=x, attention_mask=mask, decoder_input_ids=dec).logits
        nn.functional.cross_entropy(logits.float().reshape(-1, 16384), lab.reshape(-1)).backward()
        return math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None))

    assert grad_norm(m) * 5 < grad_norm(hf)


def test_scratch_build_is_seeded_and_leaves_the_callers_rng_alone():
    tc = teacher_config()
    torch.manual_seed(99)
    before = torch.rand(3)
    torch.manual_seed(99)
    a = S.build_scratch_student(tc, TINY, seed=1234)
    after = torch.rand(3)
    assert torch.equal(before, after)  # the build drew from its own fork
    b = S.build_scratch_student(tc, TINY, seed=1234)
    c = S.build_scratch_student(tc, TINY, seed=1235)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    differ = [k for k in sa if not torch.equal(sa[k], sc[k])]
    assert "model.encoder.subsampling.layers.0.weight" in differ and "model.decoder.embed_tokens.weight" in differ
    assert "model.decoder.pos_emb.weight" not in differ  # fixed, not random


def test_the_count_assert_fires(monkeypatch):
    """A named shape whose count is not the pre-registered one does not build."""
    monkeypatch.setitem(S.SCRATCH_EXPECTED_PARAMS, "tiny", 1)
    with pytest.raises(AssertionError):
        S.build_scratch_student(teacher_config(), TINY, seed=0, name="tiny")
    monkeypatch.setattr(S, "closed_form_params", lambda cfg: -1)
    with pytest.raises(AssertionError):
        S.build_scratch_student(teacher_config(), TINY, seed=0)


def test_sinusoid_pos_emb_form():
    D = 8
    pe = S.sinusoid_pos_emb(5, D)
    assert pe.shape == (5, D) and pe.dtype == torch.float32
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0] * 4) / math.sqrt(D))
    for p in range(5):
        for i in range(D // 2):
            ang = p / 10000 ** (2 * i / D)
            assert abs(float(pe[p, 2 * i]) - math.sin(ang) / math.sqrt(D)) < 1e-6
            assert abs(float(pe[p, 2 * i + 1]) - math.cos(ang) / math.sqrt(D)) < 1e-6


def _teacher_file() -> str | None:
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots/*/model.safetensors"))
    return hits[0] if hits else None


@pytest.mark.skipif(_teacher_file() is None, reason="teacher weights not in the local HF cache")
def test_sinusoid_pos_emb_is_the_teachers_table():
    """The teacher checkpoint's pos_enc (bf16) is this table: max |diff| 6.3e-5 = its bf16 rounding, and all but
    ~0.02 % of the entries round to the very same bf16 value."""
    from safetensors import safe_open

    with safe_open(_teacher_file(), "pt") as f:
        t = f.get_tensor("transf_decoder._embedding.position_embedding.pos_enc").float()
    mine = S.sinusoid_pos_emb(*t.shape)
    assert float((mine - t).abs().max()) < 1e-4
    assert float((mine.bfloat16().float() != t).float().mean()) < 1e-3


# ------------------------------------------------------------------------------------------------ save / load / train


def test_scratch_save_load_roundtrip(tmp_path):
    tc = teacher_config()
    m = S.build_scratch_student(tc, TINY, seed=5)
    S.save_student(m, tmp_path, processor=None, meta=dict(init_class="scratch"))
    loaded, info = CohereAsrForConditionalGeneration.from_pretrained(tmp_path, dtype=torch.float32,
                                                                     output_loading_info=True)
    assert not any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), info
    loaded = S.load_student(tmp_path, "cpu")
    assert loaded.proj_out.weight is loaded.model.decoder.embed_tokens.weight
    c = loaded.config
    assert c.head_dim == 16 and c.num_key_value_heads == 3 and c.encoder_config.num_key_value_heads == 2
    assert c.encoder_config.scale_input is False and c.tie_word_embeddings
    sd = m.state_dict()
    for k, v in loaded.state_dict().items():
        want = sd[k] if ".running_" in k or not v.is_floating_point() else sd[k].to(torch.bfloat16).float()
        assert torch.equal(v, want), k
    gen = json.loads((tmp_path / "generation_config.json").read_text(encoding="utf-8"))
    assert (gen["decoder_start_token_id"], gen["eos_token_id"], gen["pad_token_id"], gen["bos_token_id"]) == \
        (13764, 3, 2, 4)
    x, mask = feats([120, 80], seed=6)
    out = loaded.generate(input_features=x, attention_mask=mask, decoder_input_ids=torch.tensor([PROMPT] * 2),
                          max_new_tokens=3, min_new_tokens=3, do_sample=False)
    assert out.shape == (2, len(PROMPT) + 3)


def test_scratch_student_overfits_40_steps():
    """40 AdamW steps (BN training, the frozen pos_emb, clip 1.0) on 4 fixed utterances: the loss falls steadily."""
    m = S.build_scratch_student(teacher_config(), TINY, seed=0).train()
    m.model.decoder.pos_emb.weight.requires_grad_(False)
    x, mask = feats([150, 120, 100, 80], seed=4)
    g = torch.Generator().manual_seed(4)
    lab = torch.randint(5, 16384, (4, 8), generator=g)
    dec = torch.cat([torch.full((4, 1), 13764), lab[:, :-1]], 1)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
    losses = []
    for step in range(40):
        for group in opt.param_groups:
            group["lr"] = 3e-3 * min(1.0, (step + 1) / 10)
        logits = m(input_features=x, attention_mask=mask, decoder_input_ids=dec).logits
        loss = nn.functional.cross_entropy(logits.float().reshape(-1, 16384), lab.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss))
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < 0.5 * losses[0], losses[::5]
    assert max(losses[30:]) < min(losses[:10]), losses[::5]
    assert all(int(bn.num_batches_tracked) == 40 for bn in m.modules() if isinstance(bn, nn.BatchNorm1d))


# ------------------------------------------------------------------------------------------------ 03 --scratch


def test_build_script_scratch_argument_rules():
    from fixtures import load_script

    mod = load_script("03_build_student")
    a = mod.parse_args(["--scratch", "t01"])
    assert a.bn is None and a.seed == 1234 and a.out == ROOT / "students" / "scratch-t01-s1234"
    assert mod.build_identity(a) == dict(init="scratch", shape="t01", seed=1234)
    b = mod.parse_args([])  # today's defaults: pruned, recalibrated, importance in the kept layers, bf16 teacher
    assert (b.bn, b.importance_layers, b.importance_cache, b.ffn_from, b.teacher_dtype, b.scratch) == \
        ("recal", "spec", None, None, "bf16", None)
    for bad in (["--scratch", "t01", "--bn", "keep"], ["--scratch", "t01", "--ffn-from", "x"],
                ["--scratch", "t01", "--importance-layers", "all"],
                ["--ffn-from", "x", "--importance-layers", "all"], ["--ffn-from", "x", "--importance-cache", "y"]):
        with pytest.raises(SystemExit):
            mod.parse_args(bad)


def test_build_script_scratch_end_to_end_tiny(tmp_path, monkeypatch):
    """03 --scratch on a synthetic corpus with a tiny teacher dir (config + processor): build, save, step-0 eval, the
    study's meta fields, the no-op re-run, a different seed refused on the same --out, an unknown shape refused."""
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not cached: {e}")
    from fixtures import load_script, make_fake_corpus, make_fake_selection

    fc = make_fake_corpus(tmp_path / "corpus", sources={"src_a": (8, "train"), "eval_x": (8, "eval")})
    sel = make_fake_selection(fc, greedy_n=4, probe_n=2)
    teacher_dir = tmp_path / "teacher"
    teacher_config().save_pretrained(teacher_dir)  # no weights: a scratch build only reads the config
    proc.save_pretrained(teacher_dir)
    monkeypatch.setitem(S.SCRATCH_SHAPES, "tiny", TINY)
    out = tmp_path / "student"
    argv = ["--scratch", "tiny", "--seed", "7", "--teacher", str(teacher_dir), "--device", "cpu",
            "--step0-eval-utts", "3", "--eval-sets", "eval_x", "--selection", str(sel), "--data", str(fc.data),
            "--teacher-root", str(fc.teacher_out), "--eval-cache", str(tmp_path / "cache_eval"), "--out", str(out),
            "--eval-batch-s", "20"]
    mod = load_script("03_build_student")
    assert mod.main(argv) == 0

    meta = S.load_meta(out)
    assert meta["stage"] == "complete" and meta["format"] == 2
    assert (meta["family"], meta["init_class"], meta["bn"], meta["seed"]) == ("aed", "scratch", "fresh", 7)
    assert meta["params_total"] == meta["closed_form_params"] == meta["params"]["total"]
    assert meta["params_non_embedding"] == meta["params_total"] - 16384 * 48 - 1024 * 48
    assert meta["scratch"] == dict(name="tiny", enc_hidden=64, enc_layers=2, enc_heads=2, enc_ffn=128, dec_hidden=48,
                                   dec_layers=2, dec_heads=3, dec_ffn=96, conv_kernel=9)
    assert meta["build"] == dict(init="scratch", shape="tiny", seed=7) and meta["expected_params"] is None
    assert meta["teacher"] == str(teacher_dir) and "function_check" not in meta  # not a pruned student
    assert meta["step0"]["n_per_set"] == {"eval_x": 3}
    assert not (out / "importance.pt").exists() and (out / "tokenizer.json").exists()
    card = (out / "README.md").read_text(encoding="utf-8")  # the notice says what this student is: not pruned
    assert "encoder: 2 layers, width 64, FFN 128; decoder: 2 layers, width 48, FFN 96" in card
    assert "randomly initialised" in card and "It was pruned" not in card
    loaded = S.load_student(out, "cpu")
    ref = S.build_scratch_student(teacher_config(), TINY, seed=7).state_dict()
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, ref[k] if ".running_" in k or not v.is_floating_point()
                           else ref[k].to(torch.bfloat16).float()), k

    assert mod.main(argv) == 0  # complete -> nothing to do
    other_seed = [a if a != "7" else "8" for a in argv]
    with pytest.raises(SystemExit, match="different student"):  # not "already built": it is another student
        mod.main(other_seed)
    m2 = S.load_meta(out)
    m2["stage"] = "saved"
    S.write_meta(out, m2)
    with pytest.raises(SystemExit, match="different student"):  # a saved dir is only resumed by the same build
        mod.main(other_seed)
    with pytest.raises(SystemExit, match="not one of"):
        mod.main(["--scratch", "nope", "--out", str(tmp_path / "x"), "--teacher", str(teacher_dir)])
