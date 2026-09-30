"""kitsune.quant on CPU (emulate and fp16; torchao is not on the laptop, tests/test_quant_torchao.py covers it inside
the training image): the exact grids (E2M1 round-half-even, sign, saturation and packing; the E4M3 / E8M0 / int8 /
fp8 scales), the error bound of every pack, the layer filter on tiny CTC and AED students (the tied head, the CTC
head, the pointwise adapter, the alignment skips), apply for every format on both families, the rel-pos patch
routing through the quantised forward, the activation semantics (int8 per token, the W4A4 batch-mate trap, the
autocast rounding), the row padding, the variant dir (a reloaded variant gives the in-memory outputs to the bit, its
bytes are deterministic, w8a16 and w8a8 are one file, corruption is caught, fp16 loads with from_pretrained), the
non-finite monitor, the byte formulas, the kernel census, compare, the selftest's refusal off CUDA, the CLI's exit
codes and the heartbeat of an export.

CPU only, tiny random models built in the test; nothing needs the network or torchao."""
import json
import os
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from kitsune import quant as Q  # noqa: E402

EXPORTABLE = [f for f in Q.QUANT_FORMATS if Q.FORMAT_INFO[f]["export"]]
PACKED = [f for f in Q.QUANT_FORMATS if f != "fp16"]


def tiny_ctc(seed: int = 0, ffn: int = 64):
    from fixtures_ctc import tiny_ctc_model

    return tiny_ctc_model(seed=seed, n_layers=2, ffn=ffn)


def tiny_aed(seed: int = 0):
    """A tiny Cohere ASR student (tests/test_evaluate.py's shape): encoder d 32, one decoder layer, the tied head."""
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    enc = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
               subsampling_conv_channels=8)
    cfg = CohereAsrConfig(encoder_config=enc, vocab_size=16384, hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=64, max_position_embeddings=128,
                          decoder_start_token_id=13764, tie_word_embeddings=True)  # the students' head is tied
    torch.manual_seed(seed)
    m = CohereAsrForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.running_mean.normal_(0, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
            if isinstance(mod, (nn.Linear, nn.Conv1d)) and mod.bias is not None and mod is not m.proj_out:
                mod.bias.normal_(0.0, 0.1)
    return m.eval()


def ctc_batch(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(3, 160, 80, generator=g)
    mask = torch.ones(3, 160, dtype=torch.bool)
    mask[1, 120:] = False
    mask[2, 70:] = False
    return feats, mask


def ctc_out(model, amp=None):
    from kitsune import ctc_student as CS

    feats, mask = ctc_batch()
    with torch.no_grad(), torch.autocast("cpu", dtype=amp or torch.bfloat16, enabled=amp is not None):
        lp, _ = CS.ctc_log_probs(model, feats, mask)
    return lp


def aed_out(model, amp=None):
    g = torch.Generator().manual_seed(2)
    feats = torch.randn(2, 120, 128, generator=g)
    fmask = torch.ones(2, 120, dtype=torch.long)
    fmask[1, 90:] = 0
    dec = torch.tensor([[13764, 7, 4, 16, 98, 98, 5, 9, 11, 13, 300, 301], [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13, 302,
                                                                            2]])
    with torch.no_grad(), torch.autocast("cpu", dtype=amp or torch.bfloat16, enabled=amp is not None):
        h = model.model(input_features=feats, attention_mask=fmask, decoder_input_ids=dec,
                        decoder_attention_mask=(dec != 2).long(), use_cache=False).last_hidden_state
    return torch.nn.functional.linear(h.float(), model.proj_out.weight.float(), model.proj_out.bias.float())


def patched(model):
    from kitsune.patches import patch_relpos_once_per_batch

    patch_relpos_once_per_batch(model)
    return model


# ---------------------------------------------------------------------------------------------- grids


def test_e2m1_rne_midpoints_saturation_sign():
    """Midpoints go to the even code, values above 6 saturate, the sign is torch.signbit's (-0.2 is code 8, like
    torchao), decoding is the table, and packing puts element 2i in the low nibble."""
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert Q.e2m1_decode(Q.e2m1_encode(mids)).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    assert Q.e2m1_encode(mids).tolist() == [0, 2, 2, 4, 4, 6, 6]
    assert Q.e2m1_encode(torch.tensor([7.0, 1e9, -9.0])).tolist() == [7, 7, 15]
    assert Q.e2m1_encode(torch.tensor([-0.2, -0.0, 0.0, 0.3, 0.26])).tolist() == [8, 8, 0, 1, 1]
    vals = torch.tensor(Q.E2M1_VALUES)
    assert torch.equal(Q.e2m1_decode(Q.e2m1_encode(vals)), vals)
    assert torch.equal(Q.e2m1_decode(Q.e2m1_encode(-vals[1:])), -vals[1:])
    codes = torch.arange(16, dtype=torch.uint8).repeat(3, 2)
    packed = Q.pack_nibbles(codes)
    assert packed.shape == (3, 16) and packed[0, 0] == 0 | (1 << 4) and packed[0, 7] == 14 | (15 << 4)
    assert torch.equal(Q.unpack_nibbles(packed), codes)
    with pytest.raises(Q.QuantError):
        Q.pack_nibbles(torch.zeros(2, 3, dtype=torch.uint8))


def test_scales():
    """E4M3 block scales clamp to [e4m3 tiny, 448] before the cast and a zero block gives zero codes; the NVFP4 tensor
    scale is amax / (448 x 6), 1 for an all-zero tensor; the MXFP4 exponents (RCEIL through frexp, FLOOR) and their
    uint8 E8M0 storage; int8 and fp8 per-row scales."""
    assert Q.mx_exponent(torch.tensor([6.0, 7.0, 12.0, 0.75, 1e-40, 0.0, 2.0 ** 127])).tolist() == \
        [0, 1, 1, -3, -127, -127, 125]
    assert Q.mx_exponent(torch.tensor([6.0, 7.0, 8.0, 0.75, 0.0]), "floor").tolist() == [0, 0, 1, -3, -127]
    with pytest.raises(Q.QuantRefused):
        Q.mx_exponent(torch.ones(1), "up")
    w = torch.zeros(4, 32)
    w[0, :16] = torch.linspace(-3, 3, 16)
    w[1, 16:] = 1e-12  # a block far below the others: its scale clamps up to e4m3's smallest normal
    p = Q.pack_weight(w, "nvfp4")
    assert p.qweight.dtype == torch.uint8 and p.qweight.shape == (4, 16) and p.block_scale.dtype == torch.float8_e4m3fn
    assert p.tensor_scale.shape == () and p.tensor_scale.item() == pytest.approx(3.0 / (448 * 6))
    assert p.block_scale.float()[0, 0].item() == pytest.approx((3.0 / 6) / p.tensor_scale.item(), rel=0.07)
    assert p.block_scale.float().min().item() == pytest.approx(Q.NVFP4_SCALE_MIN)
    assert (Q.unpack_nibbles(p.qweight)[2:] == 0).all() and p.block_scale.float().max() <= 448
    assert Q.pack_weight(torch.zeros(2, 16), "nvfp4").tensor_scale.item() == 1.0
    big = torch.full((2, 16), 3e4)  # a constant tensor is exact: 6 x 448 x t is its bf16 value
    assert torch.equal(Q.unpack_weight(Q.pack_weight(big, "nvfp4")), big.to(torch.bfloat16).float())
    x = torch.tensor([[0.5, -6.0] + [0.0] * 30, [7.0] + [0.0] * 31])
    pm = Q.pack_weight(x, "mxfp4")
    assert pm.block_scale.dtype == torch.uint8 and pm.block_scale.tolist() == [[127], [128]]
    assert Q.unpack_weight(pm)[0, :2].tolist() == [0.5, -6.0] and Q.unpack_weight(pm)[1, 0].item() == 8.0
    wi = torch.tensor([[1.0, -2.0, 0.5] + [0.0] * 5, [0.0] * 8])
    pi = Q.pack_weight(wi, "int8")
    assert pi.qweight.dtype == torch.int8 and pi.scale.dtype == torch.float32
    # s = bf16(2 / 127.5) (torchao's scale dtype) = 0.01575 rounds up: -2 / s = -127.007 -> -127; 1 / s = 63.5 -> 64
    s0 = torch.tensor(2.0 / 127.5).to(torch.bfloat16).float().item()
    assert pi.scale[0].item() == s0 == 0.0157470703125 and pi.qweight[0, :3].tolist() == [64, -127, 32]
    assert pi.scale[1].item() == Q.INT8_EPS and (pi.qweight[1] == 0).all()  # an all-zero row: the eps clamp
    pf = Q.pack_weight(wi, "fp8")
    assert pf.qweight.dtype == torch.float8_e4m3fn and pf.scale[0].item() == pytest.approx(2.0 / 448)
    assert pf.scale[1].item() == 1.0 and torch.equal(Q.unpack_weight(pf)[0, :3], wi[0, :3])


@pytest.mark.parametrize("wfmt", Q.WFMTS)
def test_pack_dequant_error_bounds(wfmt):
    """Each element's dequantisation is within half a grid step of its bf16 value, and the pack is bitwise the same
    for an fp32 and a bf16 copy of the same bf16 values (the bf16 cast before quantising)."""
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(48, 64, generator=g) * torch.logspace(-3, 0, 64)[None, :]).to(torch.bfloat16)
    p = Q.pack_weight(w.float(), wfmt)
    p2 = Q.pack_weight(w, wfmt)
    for a, b in zip(p.parts().values(), p2.parts().values()):
        assert a.dtype == b.dtype and torch.equal(a.view(torch.uint8) if a.element_size() == 1 else a,
                                                  b.view(torch.uint8) if b.element_size() == 1 else b)
    d = Q.unpack_weight(p)
    err = (d - w.float()).abs()
    if wfmt == "int8":  # half a step, or what the clamp at 127 cuts off when the bf16 scale rounded down
        bound = torch.maximum(p.scale / 2, w.float().abs().amax(1) - 127 * p.scale)[:, None]
    elif wfmt == "fp8":  # e4m3 has 3 mantissa bits: half an ulp is 1/16 of the value's power of two
        bound = w.float().abs() / 16 + p.scale[:, None] * 2.0 ** -9
    elif wfmt == "nvfp4":  # the E2M1 grid's widest step (4 -> 6) is 2 block units, half of it is 1 unit
        unit = (p.block_scale.float() * p.tensor_scale).repeat_interleave(16, dim=1)
        bound = unit * 1.0 + 1e-12
    else:
        unit = torch.ldexp(torch.ones_like(p.block_scale, dtype=torch.float32),
                           p.block_scale.to(torch.int32) - 127).repeat_interleave(32, dim=1)
        bound = unit * 1.0
    assert (err <= bound * (1 + 1e-4)).all()  # the scales' own fp32 rounding
    with pytest.raises(Q.QuantError, match="non-finite"):
        Q.pack_weight(torch.full((16, 32), float("inf")), wfmt)


# ---------------------------------------------------------------------------------------------- layer filter


def test_select_layers_tiny_ctc():
    """19 Linear (9 per layer + the subsampling's) and 4 pointwise convs (scope linear: the 19); the CTC head (a
    Conv1d(k=1) too), the subsampling Conv2d stack (with its k=1 convs), the depthwise convs and the BatchNorm stay;
    MXFP4 skips the K=48 FFN outputs and the K=80 subsampling Linear with their reasons."""
    m = tiny_ctc()
    sel, kept, skipped = Q.select_layers(m, "int8-w8a8")
    assert len(sel) == 23 and sum(s.kind == "linear" for s in sel) == 19 and not skipped
    pw = [s for s in sel if s.kind == "pointwise_conv1d"]
    assert {s.name for s in pw} == {f"encoder.layers.{i}.conv.pointwise_conv{j}.linear" for i in (0, 1) for j in (1, 2)}
    assert all(s.orig_shape[-1] == 1 and s.source == s.name[:-len(".linear")] for s in pw)
    assert "encoder.layers.0.self_attn.relative_k_proj" in {s.name for s in sel}
    assert kept["ctc_head"].startswith("CTC head") and kept["encoder.layers.0.conv.depthwise_conv"] == "depthwise conv"
    assert kept["encoder.layers.1.conv.norm"].startswith("batchnorm")
    assert {k for k, v in kept.items() if v == "subsampling conv"} >= {"encoder.subsampling.layers.0"}
    assert any(isinstance(m.get_submodule(k), nn.Conv2d) and m.get_submodule(k).kernel_size == (1, 1)
               for k, v in kept.items() if v == "subsampling conv")
    lin, kept_l, _ = Q.select_layers(m, "int8-w8a8", scope="linear")
    assert len(lin) == 19 and kept_l["encoder.layers.0.conv.pointwise_conv1"] == "pointwise conv (scope linear)"
    _, _, skipped = Q.select_layers(tiny_ctc(ffn=48), "mxfp4-w4a4")
    assert skipped["encoder.subsampling.linear"].startswith("K=80") and skipped[
        "encoder.layers.0.feed_forward1.linear2"].startswith("K=48")
    assert len(skipped) == 5
    with pytest.raises(Q.QuantError):
        Q.select_layers(m, "fp16")


def test_select_layers_tiny_aed():
    """The tied proj_out is left out (shared-weight detection and its name), embed_tokens and pos_emb are kept, every
    decoder q/k/v/o/fc1/fc2 and decoder.proj (encoder -> decoder) are quantised; after apply the tie holds."""
    m = tiny_aed()
    assert m.proj_out.weight is m.model.decoder.embed_tokens.weight
    sel, kept, skipped = Q.select_layers(m, "nvfp4-w4a4")
    names = {s.name for s in sel}
    assert "proj_out" not in names and kept["proj_out"].startswith("LM head")
    assert kept["model.decoder.embed_tokens"] == "embedding" and kept["model.decoder.pos_emb"] == "position embedding"
    dec = [n for n in names if n.startswith("model.decoder.")]
    assert "model.decoder.proj" in dec and len(dec) == 11  # 4 self-attn + 4 cross-attn + fc1 + fc2 + proj
    assert sum(s.kind == "pointwise_conv1d" for s in sel) == 4 and not skipped
    rec = Q.apply(m, "nvfp4-w4a4")
    assert m.proj_out.weight is m.model.decoder.embed_tokens.weight and type(m.proj_out) is nn.Linear
    assert rec["tied"] == {"proj_out.weight": "model.decoder.embed_tokens.weight"}
    assert rec["counts"]["layers"] == len(sel) and rec["family"] == "aed"


def test_pointwise_adapter_exact():
    m = tiny_ctc()
    conv = m.encoder.layers[0].conv.pointwise_conv1
    ad = Q.PointwiseConv1dAsLinear(conv)
    assert torch.equal(ad.linear.weight, conv.weight[:, :, 0]) and torch.equal(ad.linear.bias, conv.bias)
    x = torch.randn(2, conv.in_channels, 17)
    x[1, :, 11:] = 0  # a masked tail
    with torch.no_grad():
        assert torch.allclose(ad(x), conv(x), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------------------------- apply


@pytest.mark.parametrize("fmt", Q.QUANT_FORMATS)
def test_apply_emulate_every_format_ctc_and_aed(fmt):
    """Every format on both families: finite outputs close to fp32 (loose, per format), BatchNorm statistics fp32 and
    unchanged, assert_quantized passes, and one batch calls every quantised layer (every relative_k_proj too)."""
    tol = dict(fp16=2e-2, **{"int8-w8a16": 0.05, "int8-w8a8": 0.1, "nvfp4-w4a16": 0.5, "nvfp4-w4a4": 0.8,
                             "mxfp4-w4a4": 1.2, "fp8-w8a8": 0.3})[fmt]
    for make, out in ((tiny_ctc, ctc_out), (tiny_aed, aed_out)):
        ref = out(patched(make()))
        m = patched(make())
        bn = {n: (b.running_mean.clone(), b.running_var.clone()) for n, b in m.named_modules()
              if isinstance(b, nn.BatchNorm1d)}
        rec = Q.apply(m, fmt)
        assert rec["impl"] == ("native" if fmt == "fp16" else "emulate") and rec["format"] == fmt
        y = out(m, amp=torch.float16 if fmt == "fp16" else None)
        assert torch.isfinite(y).all()
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        assert rel < tol, (make.__name__, rel)
        for n, b in m.named_modules():
            if isinstance(b, nn.BatchNorm1d):
                assert b.running_mean.dtype == torch.float32 and torch.equal(b.running_mean, bn[n][0])
                assert torch.equal(b.running_var, bn[n][1])
        Q.assert_quantized(m, rec)
        if fmt != "fp16":
            assert Q.uncalled(m) == [] and Q.counters(m)["calls"] >= rec["counts"]["layers"]
            assert all(Q.quant_layers(m)[n].kq.calls >= 1 for n in rec["layers"] if n.endswith("relative_k_proj"))
            assert rec["bytes"]["quantized"] == sum(Q.packed_bytes(rec["weights"], *v["shape"])
                                                    for v in rec["layers"].values())
        with pytest.raises(Q.QuantError, match="once"):
            Q.apply(m, fmt)


def bf16_out(model) -> torch.Tensor:
    """A forward of a model cast to bf16 as a whole (bf16 inputs, no autocast): the CTC log-probs, or the AED's
    decoder states."""
    from kitsune import ctc_student as CS

    if hasattr(model, "ctc_head"):
        feats, mask = ctc_batch()
        with torch.no_grad():
            return CS.ctc_log_probs(model, feats.to(torch.bfloat16), mask)[0]
    g = torch.Generator().manual_seed(2)
    feats = torch.randn(2, 120, 128, generator=g).to(torch.bfloat16)
    dec = torch.tensor([[13764, 7, 4, 16, 98, 98, 5, 9], [13764, 7, 4, 16, 98, 98, 5, 2]])
    with torch.no_grad():
        return model.model(input_features=feats, attention_mask=torch.ones(2, 120, dtype=torch.long),
                           decoder_input_ids=dec, decoder_attention_mask=(dec != 2).long(),
                           use_cache=False).last_hidden_state


@pytest.mark.parametrize("fmt", PACKED)
def test_apply_on_a_bf16_cast_model(fmt):
    """speed_probe's runners cast the whole model to bf16 before quantising (--dtype auto on CUDA), BatchNorm
    statistics included: apply takes the model as it is, leaves those bf16 statistics bitwise unchanged,
    assert_quantized passes, and a bf16 forward is finite and calls every quantised layer (CTC and AED)."""
    for make in (tiny_ctc, tiny_aed):
        m = patched(make().to(torch.bfloat16))
        bn = {n: (b.running_mean.clone(), b.running_var.clone()) for n, b in m.named_modules()
              if isinstance(b, nn.BatchNorm1d)}
        assert bn and all(v[0].dtype == torch.bfloat16 for v in bn.values())
        rec = Q.apply(m, fmt, impl="emulate")
        assert torch.isfinite(bf16_out(m)).all() and Q.uncalled(m) == []
        for n, b in m.named_modules():
            if isinstance(b, nn.BatchNorm1d):
                assert b.running_mean.dtype == torch.bfloat16 and torch.equal(b.running_mean, bn[n][0])
                assert torch.equal(b.running_var, bn[n][1])
        Q.assert_quantized(m, rec)


def test_assert_quantized_catches_a_changed_bn_statistic():
    """A BatchNorm statistic that changes after apply, in value or in dtype, fails assert_quantized."""
    m = tiny_ctc()
    rec = Q.apply(m, "int8-w8a16")
    bn = m.encoder.layers[0].conv.norm
    kept = bn.running_mean.clone()
    with torch.no_grad():
        bn.running_mean[0] += 1.0
    with pytest.raises(Q.QuantError, match=r"conv\.norm\.running_mean: the BatchNorm statistic changed"):
        Q.assert_quantized(m, rec)
    with torch.no_grad():
        bn.running_mean.copy_(kept)
    Q.assert_quantized(m, rec)
    bn.running_var = bn.running_var.to(torch.bfloat16)
    with pytest.raises(Q.QuantError, match=r"running_var: .*bfloat16, was torch\.float32"):
        Q.assert_quantized(m, rec)


def test_relpos_patch_routes_through_quantlinear():
    """With the rel-pos patch, relative_k_proj's quantised forward runs once per call (not the base Linear's), the
    outputs equal the unpatched quantised model's, and a stride-0 input is projected once."""
    from kitsune.patches import patch_relpos_once_per_batch

    a, b = tiny_ctc(), tiny_ctc()
    Q.apply(a, "int8-w8a8")
    Q.apply(b, "int8-w8a8")
    un = patch_relpos_once_per_batch(b)
    ya, yb = ctc_out(a), ctc_out(b)
    assert torch.allclose(ya, yb, atol=1e-4, rtol=0)
    rk = b.encoder.layers[0].self_attn.relative_k_proj
    rk_a = a.encoder.layers[0].self_attn.relative_k_proj
    assert rk.kq.calls == 1 and rk_a.kq.calls == 1
    assert rk.kq.rows < rk_a.kq.rows  # the patched one projected row 0 only
    rows = rk.kq.rows
    x = torch.randn(1, 5, rk.in_features).expand(3, 5, rk.in_features)
    with torch.no_grad():
        y = rk(x)
    assert y.shape == (3, 5, rk.out_features) and y.stride(0) == 0 and rk.kq.calls == 2 and rk.kq.rows == rows + 5
    un()


def test_act_fake_quant_semantics():
    """int8 per token: a row's grid does not depend on its batch-mates; nvfp4 (W4A4): a batch-mate with a larger amax
    changes the row (the trap hyp_diff_1 measures); under CPU bf16 autocast the input is rounded to bf16 before the
    fake quant; the output is the autocast dtype."""
    g = torch.Generator().manual_seed(5)
    row = torch.randn(1, 64, generator=g)
    mate = torch.randn(1, 64, generator=g) * 50
    both = torch.cat([row, mate])
    assert torch.equal(Q.fake_quant_act(both, "int8")[0], Q.fake_quant_act(row, "int8")[0])
    assert torch.equal(Q.fake_quant_act(both, "fp8")[0], Q.fake_quant_act(row, "fp8")[0])
    assert not torch.equal(Q.fake_quant_act(both, "nvfp4")[0], Q.fake_quant_act(row, "nvfp4")[0])
    assert torch.equal(Q.fake_quant_act(both, "mxfp4")[0], Q.fake_quant_act(row, "mxfp4")[0])
    lin = nn.Linear(64, 32)
    Q.quantize_linear(lin, "int8-w8a8", "emulate")
    x = torch.randn(4, 64, generator=g)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        y = lin(x)
    xe = Q.fake_quant_act(x.to(torch.bfloat16).float(), "int8")
    want = torch.nn.functional.linear(xe, lin.weight).to(torch.bfloat16) + lin.bias.to(torch.bfloat16)
    assert y.dtype == torch.bfloat16 and torch.equal(y, want)
    with torch.no_grad():
        y32 = lin(x)
    assert y32.dtype == torch.float32 and not torch.equal(y32.to(torch.bfloat16), y)


@pytest.mark.parametrize("fmt", PACKED)
def test_row_padding_logic(fmt):
    """min_rows forced on CPU: padded rows change nothing for any format (zero rows add no amax), the padded counter
    counts, an empty input gives an empty output; the bias outside the GEMM equals F.linear(x, W, b)."""
    g = torch.Generator().manual_seed(6)
    a, b = nn.Linear(64, 32), nn.Linear(64, 32)
    b.load_state_dict(a.state_dict())
    Q.quantize_linear(a, fmt, "emulate")
    Q.quantize_linear(b, fmt, "emulate", min_rows=17)
    x = torch.randn(2, 3, 64, generator=g)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    # equal up to the matmul's accumulation order (a BLAS may pick another kernel for 17 rows than for 6)
    assert torch.allclose(ya, yb, rtol=1e-5, atol=1e-6) and b.kq.padded == 1 and a.kq.padded == 0 and b.kq.rows == 6
    with torch.no_grad():
        assert b(torch.zeros(0, 64)).shape == (0, 32) and b(torch.randn(20, 64, generator=g)).shape == (20, 32)
    assert b.kq.padded == 1 and b.kq.calls == 3
    xe = Q.fake_quant_act(x, a.kq.act) if a.kq.act else x
    assert torch.allclose(ya, torch.nn.functional.linear(xe, a.weight, a.bias), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------------------------- variant dir


def save_ctc_dir(d: Path, seed: int = 0) -> Path:
    """A trained CTC checkpoint's dir as save_ctc_student writes it (bf16 weights, fp32 BN stats, the processor
    files, the CC-BY card as README.md and MODEL_CARD.md, student_meta.json)."""
    from fixtures_ctc import write_processor

    from kitsune import ctc_student as CS

    src = write_processor(d.parent / f"{d.name}_src")
    m = tiny_ctc(seed)
    CS.save_ctc_student(m, d, src, dict(name="tiny", family="ctc", params_total=CS.param_counts(m)["total"],
                                         enc_layers=[0, 1], ffn=64, trained=dict(step=10, run_id="x")))
    return d


def save_aed_dir(d: Path, seed: int = 0) -> Path:
    from kitsune import student as S

    S.save_student(tiny_aed(seed), d, None, dict(format=1, stage="complete", trained=dict(step=10)))
    return d


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("quant_dirs")
    return dict(ctc=save_ctc_dir(root / "ctc"), aed=save_aed_dir(root / "aed"), root=root)


def _load_base(fam: str, d: Path):
    from kitsune import ctc_student as CS
    from kitsune import student as S

    return CS.load_ctc_student(d, "cpu") if fam == "ctc" else S.load_student(d, "cpu")


@pytest.mark.parametrize("fam", ["ctc", "aed"])
@pytest.mark.parametrize("fmt", EXPORTABLE)
def test_export_load_roundtrip_bitwise(dirs, tmp_path, fam, fmt):
    """export -> load_quantized: every packed and kept tensor bitwise the in-memory one's, the outputs of the reloaded
    model equal the in-memory model's to the bit (emulate on CPU), quantization.json holds the schema's fields, the
    copied files are the source's bytes, the file has no pickle (safe_open), file_bytes is its size, bytes.deployable
    the sum of its tensors, and a second export writes the same bytes."""
    from safetensors import safe_open

    src = dirs[fam]
    out = tmp_path / "v"
    rec = Q.export(src, out, fmt)
    assert Q.is_quantized_dir(out) and not Q.verify_export(out)
    disk = json.loads((out / Q.QUANT_FILE).read_text(encoding="utf-8"))
    for k in ("schema", "format", "weights", "activations", "scope", "recipe", "source", "layers", "kept", "skipped",
              "tied", "fp16_tensors", "counts", "bytes", "file_bytes", "versions", "load_with", "time_utc", "copied"):
        assert k in disk, k
    assert disk == json.loads(json.dumps(rec)) and "impl" not in disk
    assert disk["source"]["family"] == fam and len(disk["source"]["weights_sha256"]) == 64
    assert disk["file_bytes"][Q.WEIGHTS_FILE] == (out / Q.WEIGHTS_FILE).stat().st_size
    for name in disk["copied"]:
        assert (out / name).read_bytes() == (src / name).read_bytes()
    assert {"config.json", "student_meta.json", "README.md"} <= set(disk["copied"])
    assert not any(n.endswith(".safetensors") for n in disk["copied"])
    assert sorted(Q.expected_variant_files(out)) == sorted(p.name for p in out.iterdir())
    with safe_open(str(out / Q.WEIGHTS_FILE), framework="pt") as fh:
        tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        meta = fh.metadata()
    assert meta == {"format": "pt", "kitsune_quant": disk["weights"], "kitsune_quant_schema": "1"}
    assert disk["bytes"]["deployable"] == sum(t.numel() * t.element_size() for t in tensors.values())
    # the in-memory model of the same checkpoint
    mem = _load_base(fam, src)
    mrec = Q.apply(mem, fmt, keep_packs=True)
    assert mrec["layers"] == disk["layers"] and mrec["skipped"] == disk["skipped"]
    for name in disk["layers"]:
        for part, t in mem.get_submodule(name).kpack.parts().items():
            f = tensors[f"{name}.{part}"]
            assert f.dtype == t.dtype and torch.equal(f.view(torch.uint8) if f.element_size() == 1 else f,
                                                      t.view(torch.uint8) if t.element_size() == 1 else t)
    back, brec = Q.load_quantized(out, "cpu")
    assert brec["format"] == fmt
    msd, bsd = mem.state_dict(), back.state_dict()
    assert set(msd) == set(bsd)
    for k, v in msd.items():
        assert bsd[k].dtype == v.dtype and torch.equal(bsd[k], v), k
    out_fn, amp = (ctc_out, None) if fam == "ctc" else (aed_out, None)
    amp = torch.float16 if fmt == "fp16" else amp
    for m in (mem, back):
        patched(m)
    assert torch.equal(out_fn(mem, amp), out_fn(back, amp))
    # deterministic bytes; the weight-only and W+A variants of a format are one file
    Q.export(src, tmp_path / "v2", fmt)
    assert (tmp_path / "v2" / Q.WEIGHTS_FILE).read_bytes() == (out / Q.WEIGHTS_FILE).read_bytes()
    twin = {"int8-w8a16": "int8-w8a8", "nvfp4-w4a16": "nvfp4-w4a4"}.get(fmt)
    if twin:
        Q.export(src, tmp_path / "twin", twin)
        assert (tmp_path / "twin" / Q.WEIGHTS_FILE).read_bytes() == (out / Q.WEIGHTS_FILE).read_bytes()


def test_verify_export_catches_corruption(dirs, tmp_path):
    """A qweight of the wrong shape, a scale of the wrong dtype, a missing copied tokenizer file, an edited card or a
    recipe that does not match its tensors each give a problem; load_quantized refuses a missing or extra tensor."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    good = tmp_path / "good"
    Q.export(dirs["ctc"], good, "int8-w8a8")
    assert Q.verify_export(good) == []

    def variant(tag: str, edit=None, tensors=None):
        d = tmp_path / tag
        shutil.copytree(good, d)
        if tensors:
            with safe_open(str(d / Q.WEIGHTS_FILE), framework="pt") as fh:
                ts = {k: fh.get_tensor(k) for k in fh.keys()}
                meta = fh.metadata()
            tensors(ts)
            save_file(ts, str(d / Q.WEIGHTS_FILE), metadata=meta)
        if edit:
            edit(d)
        return d

    name = next(iter(json.loads((good / Q.QUANT_FILE).read_text(encoding="utf-8"))["layers"]))
    bad = variant("shape", tensors=lambda ts: ts.__setitem__(f"{name}.qweight", ts[f"{name}.qweight"][:-1].clone()))
    assert any("qweight" in p for p in Q.verify_export(bad))
    bad = variant("dtype", tensors=lambda ts: ts.__setitem__(f"{name}.qscale", ts[f"{name}.qscale"].half()))
    assert any("qscale" in p for p in Q.verify_export(bad))
    bad = variant("tok", edit=lambda d: (d / "tokenizer.json").unlink())
    assert any("tokenizer.json" in p for p in Q.verify_export(bad))
    bad = variant("card", edit=lambda d: (d / "README.md").write_text("license: mit\n", encoding="utf-8"))
    assert any("README.md" in p for p in Q.verify_export(bad))

    def recipe_edit(d):
        r = json.loads((d / Q.QUANT_FILE).read_text(encoding="utf-8"))
        r["layers"][name]["shape"] = [r["layers"][name]["shape"][0] * 2, r["layers"][name]["shape"][1]]
        (d / Q.QUANT_FILE).write_text(json.dumps(r), encoding="utf-8")

    assert any(name in p for p in Q.verify_export(variant("recipe", edit=recipe_edit)))
    extra = variant("extra", tensors=lambda ts: ts.__setitem__("nope.weight", torch.zeros(2)))
    assert Q.verify_export(extra)  # the tensor index no longer matches
    with pytest.raises(Q.QuantError):
        Q.load_quantized(extra, "cpu")


def test_fp16_cpu_hf_loadable_and_the_nonfinite_monitor(dirs, tmp_path):
    """apply('fp16'): Linear / Conv / Embedding fp16, norms and BN fp32, a CPU fp16-autocast forward is finite; the
    exported fp16 dir loads with ParakeetForCTC.from_pretrained to the same tensors; NonFiniteMonitor counts an
    injected overflow (a weight x 1e6) without raising, with the module it showed in first."""
    from transformers import ParakeetForCTC

    m = patched(tiny_ctc())
    rec = Q.apply(m, "fp16")
    assert m.encoder.layers[0].feed_forward1.linear1.weight.dtype == torch.float16
    assert m.ctc_head.weight.dtype == torch.float16 and m.encoder.layers[0].norm_out.weight.dtype == torch.float32
    assert m.encoder.layers[0].conv.norm.weight.dtype == torch.float32 and "ctc_head.weight" in rec["fp16_tensors"]
    mon = Q.NonFiniteMonitor(m).attach()
    assert torch.isfinite(ctc_out(m, amp=torch.float16)).all()
    r = mon.record()
    assert r == dict(batches=0, rows=0, by_module={}, first=None, forwards=1)
    with torch.no_grad():
        m.encoder.layers[1].feed_forward1.linear1.weight.mul_(1e6)  # its fp16 output overflows
    ctc_out(m, amp=torch.float16)
    r = mon.record()
    mon.detach()
    assert r["batches"] == 1 and r["rows"] == 3 and r["forwards"] == 2 and r["first"] == "encoder.layers.1"
    assert r["by_module"]["encoder.layers.1"] == 1 and "encoder.layers.0" not in r["by_module"]
    ctc_out(m, amp=torch.float16)
    assert mon.record()["forwards"] == 2  # detached
    out = tmp_path / "fp16"
    Q.export(dirs["ctc"], out, "fp16")
    hf = ParakeetForCTC.from_pretrained(str(out), dtype=torch.float16)
    back, _ = Q.load_quantized(out, "cpu")
    for k, v in back.state_dict().items():
        if v.dtype == torch.float16:
            assert torch.equal(hf.state_dict()[k], v), k
    assert Q.merge_nonfinite([r, dict(batches=1, rows=2, by_module={"x": 1}, first="y", forwards=4)]) == dict(
        batches=2, rows=5, by_module={**r["by_module"], "x": 1}, first="encoder.layers.1", forwards=6)


def test_the_monitor_flushes_its_record(tmp_path):
    m = patched(tiny_ctc())
    Q.apply(m, "int8-w8a8")
    p = tmp_path / "nf" / "inv.json"
    mon = Q.NonFiniteMonitor(m, flush_path=p, flush_s=0.0).attach()
    ctc_out(m)
    mon.record()
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["nonfinite"]["forwards"] == 1 and rec["counters"]["calls"] == Q.counters(m)["calls"] > 0
    mon.detach()


# ---------------------------------------------------------------------------------------------- small parts


def test_system_names_formats_and_impls(monkeypatch):
    """system_name / split_system round trip (a +compile speed record too); unknown formats and int8-cpu are refused;
    resolve_impl: CPU auto emulates, mxfp4 always emulates, fp16 is native, CUDA auto without torchao raises."""
    assert Q.QUANT_FORMATS == ("fp16", "int8-w8a16", "int8-w8a8", "nvfp4-w4a16", "nvfp4-w4a4", "mxfp4-w4a4",
                               "fp8-w8a8") and "int8-cpu" not in Q.FORMAT_INFO
    assert set(Q.FORMAT_INFO) == set(Q.QUANT_FORMATS) and not Q.FORMAT_INFO["mxfp4-w4a4"]["timed"]
    assert all({"weights", "acts", "block", "bits", "real_impl", "timed", "export", "label"} <= set(v)
               for v in Q.FORMAT_INFO.values())
    for f in Q.QUANT_FORMATS:
        assert Q.split_system(Q.system_name("full-p03", f)) == ("full-p03", f)
        assert Q.split_system(Q.system_name("study-t06", f) + "+compile") == ("study-t06", f)
    assert Q.split_system("full-p03") == ("full-p03", None) and Q.split_system("study-p03+compile") == ("study-p03",
                                                                                                        None)
    assert Q.split_system("a@b") == ("a@b", None)
    for bad in ("int8-cpu", "int4"):
        with pytest.raises(Q.QuantRefused):
            Q.system_name("x", bad)
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    assert Q.resolve_impl("int8-w8a8", "auto", cpu) == "emulate" and Q.resolve_impl("fp16", "auto", cuda) == "native"
    assert Q.resolve_impl("mxfp4-w4a4", "auto", cuda) == "emulate"
    assert Q.resolve_impl("nvfp4-w4a4", "emulate", cuda) == "emulate"
    monkeypatch.setattr(Q, "torchao_available", lambda: False)
    with pytest.raises(Q.QuantError, match="torchao"):
        Q.resolve_impl("nvfp4-w4a4", "auto", cuda)
    with pytest.raises(Q.QuantError, match="torchao"):
        Q.resolve_impl("int8-w8a8", "torchao", cpu)
    for fmt, impl in (("mxfp4-w4a4", "torchao"), ("fp16", "emulate"), ("int8-w8a8", "fast")):
        with pytest.raises(Q.QuantRefused):
            Q.resolve_impl(fmt, impl, cpu)
    monkeypatch.setattr(Q, "torchao_available", lambda: True)
    assert Q.resolve_impl("fp8-w8a8", "auto", cuda) == "torchao"
    assert Q.identity("fp8-w8a8", "emulate", "linear+pw", "rceil") == dict(fmt="fp8-w8a8", impl="emulate",
                                                                            scope="linear+pw", mx_rounding="rceil",
                                                                            schema=1)
    assert "torchao" in Q.identity("fp8-w8a8", "torchao", "linear+pw", "rceil")


def test_weight_bytes_formula():
    """int8 N*K + 4N; nvfp4 N*K/2 + N*K/16 + 4; mxfp4 N*K/2 + N*K/32; fp8 N*K + 4N - the bytes the exporter writes per
    packed weight, equal to the pack's own nbytes."""
    n, k = 64, 128
    want = dict(int8=n * k + 4 * n, nvfp4=n * k // 2 + n * k // 16 + 4, mxfp4=n * k // 2 + n * k // 32,
                fp8=n * k + 4 * n)
    for wfmt, b in want.items():
        assert Q.packed_bytes(wfmt, n, k) == b == Q.pack_weight(torch.randn(n, k), wfmt).nbytes()
    m = tiny_ctc()
    base = Q.weight_bytes(m)
    assert base["quantized"] == 0 and base["deployable"] == base["kept"] and base["resident"] > base["deployable"]
    rec = Q.apply(m, "nvfp4-w4a16")
    wb = Q.weight_bytes(m)
    assert wb["deployable"] < base["deployable"] and wb["quantized"] == sum(
        Q.packed_bytes("nvfp4", *v["shape"]) for v in rec["layers"].values())


def test_kernel_census_on_cpu():
    """The op counts and kernel classes; an emulated int8 layer's calls are counted, but it runs no int8 GEMM, so
    fallback_mm is None (not applicable) and torch._int_mm is not wrapped."""
    lin = nn.Linear(64, 32)
    Q.quantize_linear(lin, "int8-w8a8", "emulate")
    x = torch.randn(4, 64)
    with torch.no_grad():
        c = Q.kernel_census(lambda: lin(x), "cpu", model=Q._One(lin))
    assert c["ops"].get("aten::linear", 0) >= 1 and c["int8_act_calls"] == 1 and c["fallback_mm"] is None
    assert c["int8_mm"] is None
    assert Q._kernel_class("sm120_xmma_gemm_e2m1_e2m1") == "fp4" and Q._kernel_class("cutlass_i8i8_gemm") == "int8"
    assert Q._kernel_class("sm90_gemm_e4m3") == "fp8" and Q._kernel_class("ampere_bf16_s16816gemm") == "bf16"
    assert "fallback_mm" not in Q.kernel_census(lambda: lin(x), "cpu")  # no model: ops and kernels only


def test_kernel_census_counts_int8_gemms_that_returned():
    """fallback_mm counts successes, not attempts: torchao's safe_int_mm runs torch._int_mm in try/except and silently
    runs an fp32 matmul when it raises, and the profiler records aten::_int_mm for the raising call too. A layer
    standing in for a torchao int8-activation one (impl torchao; its call counted) with: a raising torch._int_mm and
    the fp32 fallback -> fallback_mm 1; a torch._int_mm that returns -> 0; an int8 GEMM run past the wrapper
    (torch.ops.aten._int_mm) -> None (cannot be told); the counters off (speed_probe --compile) -> None. The wrapper is
    gone afterwards."""
    lin = nn.Linear(64, 32)
    Q.quantize_linear(lin, "int8-w8a8", "emulate")
    lin.kq.impl = "torchao"  # the census only reads the impl; this forward stays the plain one on CPU
    x = torch.randn(4, 64)
    a, b = torch.randint(-5, 5, (17, 64), dtype=torch.int8), torch.randint(-5, 5, (64, 32), dtype=torch.int8)
    a4, bad = torch.randint(-5, 5, (4, 16), dtype=torch.int8), torch.randint(-5, 5, (8, 16), dtype=torch.int8)

    def fallback():
        lin(x)
        try:
            torch._int_mm(a4, bad)  # the inner dims differ: it raises, as _int_mm does at M <= 16 on CUDA
        except RuntimeError:
            torch.matmul(a4.float(), bad.float().t())

    def ran():
        lin(x)
        torch._int_mm(a, b)

    def past():
        lin(x)
        torch.ops.aten._int_mm(a, b)

    orig = torch._int_mm
    with torch.no_grad():
        c = Q.kernel_census(fallback, "cpu", model=Q._One(lin))
        assert c["ops"]["aten::_int_mm"] == 1 and c["int8_mm"] == dict(calls=1, ok=0, raised=1)
        assert c["int8_act_calls"] == 1 and c["fallback_mm"] == 1
        c = Q.kernel_census(ran, "cpu", model=Q._One(lin))
        assert c["int8_mm"] == dict(calls=1, ok=1, raised=0) and c["fallback_mm"] == 0
        c = Q.kernel_census(past, "cpu", model=Q._One(lin))
        assert c["int8_mm"]["unseen"] == 1 and c["fallback_mm"] is None and c["int8_act_calls"] == 1
        Q.set_counting(lin, False)
        c = Q.kernel_census(ran, "cpu", model=Q._One(lin))
        assert c["int8_act_calls"] is None and c["int8_mm"] is None and c["fallback_mm"] is None
    assert torch._int_mm is orig


def test_low_gemm_evidence():
    """The selftest's autocast check reads ops, not kernel names: cuBLASLt's sm_120 GEMMs (nvjet_*) carry no dtype
    in their names. fp4 / fp8: a scaled-matmul op (or a named kernel of that class) passes, an unrecognised kernel
    name alongside is a warning, neither fails; int8: every int8 call's GEMM returned (fallback_mm 0); weight-only
    formats need no low-precision GEMM."""
    nvjet = dict(ops={"aten::_scaled_mm": 1, "aten::linear": 1}, gemm_classes={},
                 kernels={"nvjet_sm120_tst_128x256_64x4_1x2_h_bz_coopA_TNT": 1})
    assert Q._kernel_class("nvjet_sm120_tst_128x256_64x4_1x2_h_bz_coopA_TNT") == "other"
    for fmt in ("nvfp4-w4a4", "fp8-w8a8"):
        r = Q.low_gemm_evidence(fmt, nvjet)
        assert r["ok"] and r["warning"] and "1 scaled-matmul op" in r["evidence"]
        assert not Q.low_gemm_evidence(fmt, dict(ops={"aten::mm": 1}, gemm_classes={"bf16": 1}))["ok"]
    named = Q.low_gemm_evidence("nvfp4-w4a4", dict(ops={}, gemm_classes={"fp4": 2}))
    assert named["ok"] and named["warning"] is None
    assert Q.low_gemm_evidence("fp8-w8a8", dict(ops={"mslk::f8f8bf16_rowwise": 1}, gemm_classes={}))["ok"]
    assert Q.low_gemm_evidence("int8-w8a8", dict(ops={"aten::_int_mm": 1}, gemm_classes={}, int8_act_calls=1,
                                                 fallback_mm=0))["ok"]
    for fb, calls in ((1, 1), (None, 1), (0, 0)):
        assert not Q.low_gemm_evidence("int8-w8a8", dict(ops={"aten::_int_mm": 1}, gemm_classes={"int8": 1},
                                                         int8_act_calls=calls, fallback_mm=fb))["ok"]
    for fmt in ("int8-w8a16", "nvfp4-w4a16"):
        assert Q.low_gemm_evidence(fmt, dict(ops={}, gemm_classes={}))["ok"]


def test_the_mxfp4_canary_tells_a_config_error_from_a_refusal(monkeypatch):
    """Decision 20's canary builds torchao's MXFP4 AUTO config on its own first: a config that does not build is a
    warning with its error (config_error), never counted as "AUTO refused MXFP4" (raised stays None)."""
    def no_config():
        raise TypeError("__init__() got an unexpected keyword argument 'kernel_preference'")

    monkeypatch.setattr(Q, "mxfp4_auto_config", no_config)
    rec = dict(warnings=[])
    Q._selftest_mxfp4(rec, torch.device("cpu"), (64, 64))
    assert rec["mxfp4_auto"]["raised"] is None and "kernel_preference" in rec["mxfp4_auto"]["config_error"]
    assert "not tried" in rec["warnings"][0]


def _eval_dir(d: Path, hyps: dict, refs: dict, tf=None):
    import pandas as pd

    d.mkdir(parents=True, exist_ok=True)
    for s, h in hyps.items():
        pd.DataFrame(dict(id=list(h), hyp=list(h.values()), ref=[refs[i] for i in h])).to_parquet(
            d / f"greedy_{s}.parquet")
        if tf is not None:
            pd.DataFrame(dict(id=list(h), kl=[tf] * len(h))).to_parquet(d / f"tf_{s}.parquet")
    return d


def test_compare_eval_dirs_and_the_cli(tmp_path, capsys):
    refs = {"a": "あいうえお", "b": "かきくけこ", "c": "さしすせそ"}
    one = _eval_dir(tmp_path / "one", {"eval_jsut": {"a": "あいうえお", "b": "かきくけ"}, "eval_cv8": {"c": "さし"}},
                    refs, tf=0.5)
    same = _eval_dir(tmp_path / "same", {"eval_jsut": {"a": "あいうえお", "b": "かきくけ"}}, refs, tf=0.5)
    other = _eval_dir(tmp_path / "other", {"eval_jsut": {"a": "あいうえ", "b": "かきくけ"}}, refs, tf=0.5)
    r = Q.compare_eval_dirs(one, same)
    assert r["same"] and set(r["sets"]) == {"eval_jsut"} and r["only_a"] == ["eval_cv8"]
    assert not Q.compare_eval_dirs(one, _eval_dir(tmp_path / "tf", {"eval_jsut": {"a": "あいうえお", "b": "かきくけ"}},
                                                  refs, tf=0.6))["same"]
    r = Q.compare_eval_dirs(one, other)
    assert not r["same"] and r["sets"]["eval_jsut"]["n_diff"] == 1
    assert Q.compare_eval_dirs(one, other, exact=False, tol_cer=0.11)["same"]
    assert not Q.compare_eval_dirs(one, other, exact=False, tol_cer=0.05)["same"]
    assert not Q.compare_eval_dirs(one, tmp_path / "empty")["same"]
    js = tmp_path / "cmp" / "compare.json"
    assert Q.main(["compare", str(one), str(same), "--exact", "--json-out", str(js)]) == 0
    assert json.loads(js.read_text(encoding="utf-8"))["same"] is True
    assert Q.main(["compare", str(one), str(other), "--json-out", str(js)]) == 1
    assert Q.main(["compare", str(one), str(other), "--tol-cer", "0.2"]) == 0
    capsys.readouterr()


def test_selftest_off_cuda_is_not_ok(tmp_path, capsys):
    """The selftest times real kernels: on CPU it records why it cannot run and is not ok (exit 1), with its JSON."""
    rec = Q.selftest("cpu")
    assert rec["ok"] is False and rec["checks"][0]["name"] == "environment" and not rec["checks"][0]["ok"]
    out = tmp_path / "st.json"
    assert Q.main(["selftest", "--device", "cpu", "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False
    capsys.readouterr()


def test_cli_exit_codes_and_the_export_heartbeat(dirs, tmp_path, monkeypatch, capsys):
    """export: 0, then 2 on an existing --out (no --force) and on an unknown format or int8-cpu; inspect 0 on a good
    dir, 1 on a broken one; a readout with an unknown format 2. An export beats $KITSUNE_HEARTBEAT."""
    hb = tmp_path / "hb" / "quant-item"
    monkeypatch.setenv("KITSUNE_HEARTBEAT", str(hb))
    out = tmp_path / "v"
    assert Q.main(["export", "--ckpt", str(dirs["ctc"]), "--fmt", "nvfp4-w4a4", "--out", str(out)]) == 0
    assert hb.is_file()
    assert Q.main(["export", "--ckpt", str(dirs["ctc"]), "--fmt", "nvfp4-w4a4", "--out", str(out)]) == 2
    assert Q.main(["export", "--ckpt", str(dirs["ctc"]), "--fmt", "nvfp4-w4a4", "--out", str(out), "--force"]) == 0
    for fmt in ("int8-cpu", "w2"):
        assert Q.main(["export", "--ckpt", str(dirs["ctc"]), "--fmt", fmt, "--out", str(tmp_path / fmt)]) == 2
    assert Q.main(["export", "--ckpt", str(out), "--fmt", "int8-w8a8", "--out", str(tmp_path / "vv")]) == 2
    assert Q.main(["inspect", str(out)]) == 0
    (out / "config.json").write_text("{}", encoding="utf-8")
    assert Q.main(["inspect", str(out)]) == 1
    assert Q.main(["readout", "--config", "c.json", "--ckpt", str(dirs["ctc"]), "--fmt", "int8-cpu", "--out",
                   str(tmp_path / "r"), "--cache-dir", str(tmp_path / "c"), "--manifest", "m.json"]) == 2
    capsys.readouterr()


class FakeAOTensor(torch.Tensor):
    """A traceable wrapper subclass shaped like torchao's int8 weight (qdata int8 (N, K), scale (N, 1)): the torchao
    adapter's generic part (__tensor_flatten__ / __tensor_unflatten__, the roles, the exact scale conversion) on the
    laptop, where torchao itself is absent."""

    @staticmethod
    def __new__(cls, qdata, scale, shape, extra=None):
        return torch.Tensor._make_wrapper_subclass(cls, shape, dtype=torch.bfloat16, device=qdata.device)

    def __init__(self, qdata, scale, shape, extra=None):
        self.qdata, self.scale, self.extra = qdata, scale, extra

    def __tensor_flatten__(self):
        return ["qdata", "scale"] + (["extra"] if self.extra is not None else []), None

    @classmethod
    def __tensor_unflatten__(cls, inner, ctx, outer_size, outer_stride):
        return cls(inner["qdata"], inner["scale"], outer_size, inner.get("extra"))

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(func)

    def dequantize(self):
        return self.qdata.float() * self.scale.float()


def test_the_torchao_adapter_swaps_a_subclass_inner_tensors(monkeypatch):
    """to_torchao places the pack's codes and scales into the template's inner tensors (by role: dtype and shape),
    from_torchao reads them back; a scale dtype that cannot hold the pack's values, or an inner tensor of no known
    role, raises instead of converting silently."""
    n, k = 16, 32
    w = torch.randn(n, k)
    pack = Q.pack_weight(w, "int8")
    tpl = FakeAOTensor(torch.zeros(n, k, dtype=torch.int8), torch.ones(n, 1), (n, k))
    monkeypatch.setattr(Q, "_template", lambda n_, k_, fmt, device: tpl)
    t = Q.to_torchao(pack, "int8-w8a8", "cpu")
    assert type(t) is FakeAOTensor and t.shape == (n, k) and torch.equal(t.qdata, pack.qweight)
    assert torch.equal(t.scale.reshape(-1), pack.scale) and torch.equal(t.dequantize(), Q.unpack_weight(pack))
    back = Q.from_torchao(t, "int8-w8a8")
    assert torch.equal(back.qweight, pack.qweight) and torch.equal(back.scale, pack.scale)
    e4m3 = FakeAOTensor(torch.zeros(n, k, dtype=torch.int8), torch.ones(n, 1).to(torch.float8_e4m3fn), (n, k))
    monkeypatch.setattr(Q, "_template", lambda n_, k_, fmt, device: e4m3)
    with pytest.raises(Q.QuantError, match="does not hold the pack"):
        Q.to_torchao(pack, "int8-w8a8", "cpu")
    odd = FakeAOTensor(torch.zeros(n, k, dtype=torch.int8), torch.ones(n, 1), (n, k), extra=torch.zeros(3, 3))
    monkeypatch.setattr(Q, "_template", lambda n_, k_, fmt, device: odd)
    with pytest.raises(Q.QuantError, match="cannot place"):
        Q.to_torchao(pack, "int8-w8a8", "cpu")
