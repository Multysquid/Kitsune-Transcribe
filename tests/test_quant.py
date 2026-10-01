"""kitsune.quant on CPU (emulate and fp16; torchao is not on the laptop, tests/test_quant_torchao.py covers it inside
the training image): the exact grids (E2M1 round-half-even, sign, saturation and packing; the E4M3 / E8M0 / int8 /
fp8 scales; NVFP4's data scaled by torchao's reciprocal), the error bound of every pack, the layer filter on tiny CTC
and AED students (the tied head, the CTC head, the pointwise adapter, the alignment skips), apply for every format on
both families (and on a model cast to bf16 as a whole, as speed_probe's runners do; a changed BatchNorm statistic
fails assert_quantized), the rel-pos patch routing through the quantised forward, the activation semantics (int8 per
token, the W4A4 batch-mate trap, the autocast rounding), the row padding, the variant dir (a reloaded variant gives
the in-memory outputs to the bit, its bytes are deterministic, w8a16 and w8a8 are one file, corruption is caught, fp16
loads with from_pretrained), the non-finite monitor, the byte formulas, the kernel census (int8 GEMMs counted when they
return, not when attempted; nothing counted with the counters off), the selftest's autocast evidence (ops, not kernel
names) and MXFP4 canary, compare, the selftest's refusal off CUDA, the CLI's exit codes and the heartbeat of an export.
F1 / F4 (DECISIONS F): the fp8 recipe is torchao's arithmetic (bf16 row scales, activation_value_lb), the bf16 scales
are device-independent (exhaustively), a variant of older quant code is refused, and the selftest's zero-row, padded
checkpoint and compiled-block sub-checks pass on CPU (emulate) and catch the pre-F1 fp8 NaN.

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
    # F1: s = bf16(2 / 448) = 0.00445556640625 (torchao's scale dtype), a little under 2 / 448: 1 / s = 224.4 -> 224,
    # -2 / s = -448.9 clamps to -448, 0.5 / s = 112.2 -> 112; the dequantisation is no longer exact
    s8 = (torch.tensor(2.0, dtype=torch.bfloat16) / 448).float().item()
    assert pf.qweight.dtype == torch.float8_e4m3fn and pf.scale[0].item() == s8 == 0.00445556640625
    assert pf.scale[1].item() == 1.0 and (pf.qweight[1].float() == 0).all()  # an all-zero weight row: scale 1
    assert torch.equal(Q.unpack_weight(pf)[0, :3], torch.tensor([224.0, -448.0, 112.0]) * s8)


def _torchao_nvfp4_codes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torchao 0.18's nvfp4_quantize (two-level, per_tensor_amax_to_scale), its arithmetic verbatim: fp32 from the
    start, block scales (amax / 6) / t clamped to [e4m3 tiny, 448] in e4m3, and the data MULTIPLIED by (1 / t) / bs
    "to match the MSLK triton kernel numerics". Returns (E2M1 codes (N, K), block scales)."""
    t = x.float().abs().max().to(torch.float32) / (448.0 * 6.0)
    d = x.float().reshape(x.shape[0], -1, 16)
    bs = torch.clamp((torch.amax(torch.abs(d), dim=-1) / 6.0).to(torch.float32) / t,
                     min=torch.finfo(torch.float8_e4m3fn).tiny, max=448.0).to(torch.float8_e4m3fn)
    recip = (1.0 / t) / bs.to(torch.float32)
    return Q.e2m1_encode(torch.clamp(d * recip.unsqueeze(-1), -6.0, 6.0).reshape(x.shape)), bs


def test_nvfp4_codes_use_torchaos_reciprocal_scaling():
    """The NVFP4 pack and the W4A4 activation grid scale the data as torchao 0.18 does, x * ((1 / t) / bs): on a
    real-sized weight and on activations the codes equal that form's, and they differ from the division x / (bs * t)
    on some (the form is what decides a value near an E2M1 midpoint). tests/test_quant_torchao.py checks the same
    against torchao itself in the image."""
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(2560, 1024, generator=g) * 0.02).to(torch.bfloat16)
    codes, bs = _torchao_nvfp4_codes(w)
    ours = Q.pack_weight(w, "nvfp4")
    assert torch.equal(ours.block_scale.view(torch.uint8), bs.view(torch.uint8))
    assert torch.equal(Q.unpack_nibbles(ours.qweight), codes)
    t = ours.tensor_scale
    divided = Q.e2m1_encode((w.float().reshape(2560, 64, 16) / (bs.float() * t)[..., None]).clamp(-6, 6)).reshape(
        2560, 1024)
    assert int((divided != codes).sum()) > 0  # the test tells the two forms apart
    x = (torch.randn(400, 1024, generator=g) * torch.logspace(-1, 1, 400)[:, None]).to(torch.bfloat16)
    xc, xbs = _torchao_nvfp4_codes(x)
    kc, kbs, _ = Q._nvfp4(x.float())
    assert torch.equal(kc, xc) and torch.equal(kbs.view(torch.uint8), xbs.view(torch.uint8))


def _torchao_fp8_rows(x: torch.Tensor, lb: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """torchao 0.18's float8 per row on a bf16 tensor, its arithmetic verbatim: _choose_scale_float8 (block [1, K]) -
    the row amax in the tensor's dtype, clamped up to hp_value_lb when set (Float8DynamicActivationFloat8WeightConfig's
    activation_value_lb), divided by 448 in that dtype, then fp32 - and _quantize_affine_float8: fp32(x) / s clamped to
    +-448, cast to e4m3. No guard for a zero row. Returns (codes (N, K), scales (N,))."""
    xb = x.to(torch.bfloat16)
    amax = xb.abs().amax(dim=1, keepdim=True)
    if lb is not None:
        amax = torch.clamp(amax, min=lb)
    s = (amax / 448.0).to(torch.float32)
    return (xb.to(torch.float32) / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), s.reshape(-1)


def test_fp8_recipe_is_torchaos_arithmetic():
    """F1 (DECISIONS F; box 53693389's smoke B): the fp8 activation grid and the fp8 weight pack are torchao 0.18's
    recipe bit for bit, on rows of every magnitude: the row scale bf16(amax / 448) (FP8_SCALE_DTYPE), the activation's
    amax clamped up to FP8_ACT_LB first. An all-zero activation row (a padded frame) gets the scale bf16(2^-40) / 448
    and exact zeros, where torchao's default (no lb) divides 0 by 0; an all-zero weight row keeps scale 1 and zero codes
    (the documented deviation: torchao has no weight lb). The pre-F1 fp32 scale differs from it in most rows, so this
    test tells the two forms apart."""
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(400, 1024, generator=g) * torch.logspace(-6, 3, 400)[:, None]).to(torch.bfloat16)
    zero = [7, 123]
    x[zero] = 0.0
    nz = [i for i in range(400) if i not in zero]
    tq, ts = _torchao_fp8_rows(x, lb=Q.FP8_ACT_LB)
    q, s = Q._fp8_rows(x.float(), lb=Q.FP8_ACT_LB)
    assert torch.equal(s, ts) and torch.equal(q.view(torch.uint8), tq.view(torch.uint8))
    y = Q.fake_quant_act(x.float(), "fp8")
    assert torch.equal(y, tq.float() * ts[:, None]) and torch.isfinite(y).all()
    assert Q.FP8_ACT_LB == 2.0 ** -40 and float(torch.tensor(Q.FP8_ACT_LB, dtype=torch.bfloat16)) == Q.FP8_ACT_LB
    lb_scale = (torch.tensor(Q.FP8_ACT_LB, dtype=torch.bfloat16) / 448).float().item()
    assert all(s[i].item() == lb_scale for i in zero) and (q[zero].float() == 0).all() and (y[zero] == 0).all()
    dq, ds = _torchao_fp8_rows(x)  # torchao's default config: no lb, a zero row is 0 / 0
    assert (ds[zero] == 0).all() and torch.isnan(dq[zero].float()).all()
    p = Q.pack_weight(x, "fp8")
    assert torch.equal(p.scale[nz], ts[nz]) and torch.equal(p.qweight[nz].view(torch.uint8),
                                                            tq[nz].view(torch.uint8))
    assert (p.scale[zero] == 1.0).all() and (p.qweight[zero].float() == 0).all()
    old = x.float().abs().amax(dim=1) / 448  # the pre-F1 fp32 scale (14bfcad)
    assert float((old[nz] != ts[nz]).float().mean()) > 0.5


def _torchao_int8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torchao 0.18's symmetric int8 per row on a bf16 tensor (Int8Tensor.from_hp, the weight's and the activation's
    call alike), its arithmetic verbatim: choose_qparams_affine - max(-min(amin, 0), max(amax, 0)) / ((127 - -128) /
    2) in the tensor's dtype, clamped to float32 eps, then fp32 - and quantize_affine: clamp(round(x * (1.0 / s)) + 0,
    -128, 127). Returns (codes (N, K) fp32, scales (N,))."""
    xb = x.to(torch.bfloat16)
    mn = torch.min(xb.amin(dim=1, keepdim=True), torch.zeros(1, dtype=xb.dtype))
    mx = torch.max(xb.amax(dim=1, keepdim=True), torch.zeros(1, dtype=xb.dtype))
    s = torch.clamp(torch.max(-mn, mx) / (float(127 - -128) / 2), min=torch.finfo(torch.float32).eps).to(torch.float32)
    return torch.clamp(torch.round(xb * (1.0 / s)) + 0, -128, 127), s.reshape(-1)


def test_int8_recipe_is_torchaos_arithmetic():
    """The int8 weight pack and the W8A8 activation grid are torchao 0.18's recipe bit for bit on real-sized tensors:
    the codes round(x * (1 / s)), torchao's reciprocal form. On a 2560 x 1024 N(0, 0.02) weight - the selftest's torchao
    parity weight (seed 2) among them - the division x / s gives another code on a tie (x / s = k + 0.5 exactly), so
    this test tells the two forms apart (the 96 x 128 weights of the parity tests had no such tie). The activations:
    an N(0, 1) 4096 x 1024 input (with an all-zero row and a tiny one) has rows with a -128 code and the eps clamp
    applies, where the pre-fix recipe ([-127, 127], eps 1e-5) differed. tests/test_quant_torchao.py checks the same
    against torchao itself in the image."""
    for seed in (0, 2):
        g = torch.Generator().manual_seed(seed)
        w = (torch.randn(2560, 1024, generator=g) * 0.02).to(torch.bfloat16)
        tq, ts = _torchao_int8_rows(w)
        p = Q.pack_weight(w, "int8")
        assert torch.equal(p.scale, ts) and torch.equal(p.qweight.float(), tq), seed
        divided = torch.clamp(torch.round(w.float() / ts[:, None]), -128, 127)
        assert int((divided != tq).sum()) > 0, seed  # the ties: 12 codes at seed 0, 13 at seed 2
    g = torch.Generator().manual_seed(6)
    x = torch.randn(4096, 1024, generator=g).to(torch.bfloat16)
    x[5] = 0.0
    x[9] *= 1e-6
    tq, ts = _torchao_int8_rows(x)
    q, s = Q._int8_rows(x.float(), Q.INT8_ACT_DIV, Q.INT8_ACT_EPS, Q.INT8_ACT_QMIN, scale_dtype=Q.INT8_ACT_SCALE_DTYPE)
    assert torch.equal(s, ts) and torch.equal(q, tq)
    assert torch.equal(Q.fake_quant_act(x.float(), "int8"), tq * ts[:, None])
    assert (tq == -128).any(dim=1).sum() > 0 and (tq[5] == 0).all() and ts[5].item() == Q.INT8_EPS
    old_s = torch.clamp((x.float().abs().amax(1) / 127.5).to(torch.bfloat16), min=1e-5).float()
    old_q = torch.clamp(torch.round(x.float() / old_s[:, None]), -127, 127)
    assert not torch.equal(old_s, ts) and not torch.equal(old_q, tq)


def test_fp8_and_int8_bf16_scales_are_the_same_from_cpu_division_and_cudas_reciprocal():
    """Why the bf16-rounded fp8 (F1) and int8 scales are device-independent: for every positive finite bf16 amax,
    bf16(a / d) - the CPU's true division - equals bf16(fp32(a) * fp32(1 / d)) - CUDA's kernel for a tensor divided by
    a Python scalar - for d = 448 (fp8) and 127.5 (int8). The pack is computed on the CPU anyway (pack_weight); this
    is what lets torchao's own quantize_ on the GPU give the same scales (the selftest's torchao parity)."""
    a = torch.arange(2 ** 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    a = a[torch.isfinite(a) & (a > 0)]
    assert a.numel() == 32639  # 0x7F7F: every positive finite bf16, its 127 subnormals included
    for d in (Q.F8_MAX, Q.INT8_DIV):
        cpu = a / d
        recip = (a.float() * torch.tensor(1.0 / d, dtype=torch.float32)).to(torch.bfloat16)
        assert int((cpu.view(torch.int16) != recip.view(torch.int16)).sum()) == 0, d


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


class _AssertsRowMajor(nn.Linear):
    """torchao 0.18's NVFP4 activation quantiser in miniature: `assert data_hp.is_contiguous()` on the 2-D input
    QuantLinear hands its GEMM (x.reshape(-1, K))."""

    def forward(self, x):
        assert x.reshape(-1, self.in_features).is_contiguous(), "Only support contiguous data for now"
        return super().forward(x)


def _pointwise_with_assert(seed: int = 0) -> tuple[nn.Conv1d, Q.PointwiseConv1dAsLinear]:
    torch.manual_seed(seed)
    conv = nn.Conv1d(64, 64, 1)
    ad = Q.PointwiseConv1dAsLinear(conv)
    ad.linear.__class__ = _AssertsRowMajor
    return conv, ad


def test_pointwise_conv_hands_its_linear_a_row_major_input():
    """Box 53693389's smoke B: every nvfp4-w4a4 probe failed at its batch-1 warm-up, eager and compiled. At B = 1 the
    (B, C, T) depthwise output's transpose reshaped to a (T, C) view with strides (1, T), which torchao's NVFP4
    quantiser refuses; B > 1 had copied already. The adapter now hands a row-major input in both modes, at both
    batch sizes, with the conv's numbers."""
    conv, ad = _pointwise_with_assert()
    dw = nn.Conv1d(64, 64, 3, padding=1, groups=64)  # the depthwise conv before it: T fastest
    for b in (1, 3):
        x = dw(torch.randn(b, 64, 37))
        with torch.no_grad():
            assert torch.allclose(ad(x), conv(x), atol=1e-5, rtol=0), b
    compiled = torch.compile(ad, backend="aot_eager", dynamic=True)
    for b in (3, 1):  # batched first, then the B = 1 recompile (where dynamo traced a non-contiguous fake tensor)
        x = dw(torch.randn(b, 64, 41))
        with torch.no_grad():
            assert torch.allclose(compiled(x), conv(x), atol=1e-5, rtol=0), b


def test_the_row_major_op():
    """kitsune::row_major: a registered custom op whose output is a row-major copy (its fake says so, which is what
    torch.compile keeps), passing torch.library.opcheck; registering it again (a reload of the module) is the same op,
    and torch.compile still traces it afterwards: PR #35's version called torch.library.custom_op a second time here,
    whose half-built Library deregistered the op's schema, and every later compile through it failed."""
    x = torch.randn(2, 8, 5).transpose(1, 2)
    y = Q.row_major(x)
    assert y.is_contiguous() and torch.equal(y, x) and y.data_ptr() != x.data_ptr()
    res = torch.library.opcheck(Q.row_major, (x,))
    assert all(v == "SUCCESS" for v in res.values()), res
    again = Q._register_row_major()
    assert again is Q.row_major and again(x).is_contiguous() and torch.equal(again(x), x)
    import gc

    gc.collect()  # what deregistered the schema was the collection of the failed second registration
    conv, ad = _pointwise_with_assert(1)
    torch.compiler.reset()
    x = torch.randn(2, 64, 9)
    with torch.no_grad():
        assert torch.allclose(torch.compile(ad, backend="aot_eager", dynamic=True)(x), conv(x), atol=1e-5, rtol=0)


def test_the_selftest_names_the_int8_kernel_and_tells_mxfp4s_refusal_from_another_error(monkeypatch):
    """sm_120's int8 GEMM (cutlass_80_wmma_tensorop_i161616gemm_s8_*) is int8, not other_gemm (box 53693389's
    cosmetic int8 warning). Decision 20's canary counts only a refusal that names the hardware as the expected one;
    any other error from torchao's AUTO kernel is a warning that decision 20 went unchecked."""
    assert Q._kernel_class("cutlass_80_wmma_tensorop_i161616gemm_s8_128x128_128x2_tn_align16") == "int8"

    class Lin(nn.Linear):
        def forward(self, x):
            raise self.err

    for err, hw in ((NotImplementedError("MXFP4 scaling only supported in CUDA for B200/B300"), True),
                    (RuntimeError("shape mismatch"), False), (AssertionError("B200"), False)):
        monkeypatch.setattr(Q, "mxfp4_auto_config", lambda: object())
        monkeypatch.setattr(Q, "_ao", lambda name: (lambda lin, cfg: setattr(lin, "__class__", Lin)
                                                     or setattr(lin, "err", err)))
        rec = dict(warnings=[])
        Q._selftest_mxfp4(rec, torch.device("cpu"), (64, 64))
        out = rec["mxfp4_auto"]
        assert out["raised"] is True and out["refused_for_hardware"] is hw, err
        assert bool(rec["warnings"]) is (not hw) and all("decision 20 unchecked" in w for w in rec["warnings"])


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
        assert set(Q.RECIPE_KEYS) <= set(rec["recipe"])  # scout s7's recipe keys, null where they do not apply
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
              "tied", "fp16_tensors", "counts", "bytes", "file_bytes", "versions", "load_with", "time_utc", "copied",
              "recipe_version", "recipe_sha256"):
        assert k in disk, k
    assert disk == json.loads(json.dumps(rec)) and "impl" not in disk
    assert disk["recipe_version"] == Q.RECIPE_VERSION == 2 and disk["recipe_sha256"] == Q.recipe_sha256(fmt)
    if fmt == "fp8-w8a8":  # F1's constants in the record
        assert disk["recipe"]["act_value_lb"] == 2.0 ** -40 and disk["recipe"]["act_scale_dtype"] == "bfloat16"
        assert disk["recipe"]["fp8_scale_dtype"] == "bfloat16"
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
                                                                            schema=1, recipe_version=Q.RECIPE_VERSION)
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


def _parts(d: Path, identity: dict | None = None, **sets):
    """A 05 --out dir's .parts records: identity.json and one <set>.json per keyword (its pass_sets, batch_s,
    chunks, batches_sha256)."""
    (d / ".parts").mkdir(parents=True, exist_ok=True)
    ident = dict(weights="w1", step=10, store="fp-store", batch_s=400.0, device="cuda", autocast="bf16", tf32=False,
                 relpos_patch=True, family="ctc", eval_sets=["eval_jsut"], sources=["eval_jsut"], subset=None,
                 quant=dict(fmt="int8-w8a8", impl="torchao", scope="linear+pw", source="file"))
    ident.update(identity or {})
    (d / ".parts" / "identity.json").write_text(json.dumps(ident), encoding="utf-8")
    for s, rec in sets.items():
        (d / ".parts" / f"{s}.json").write_text(json.dumps(dict(dict(pass_sets=[s], batch_s=400.0, chunks=3,
                                                                     batches_sha256="b" * 64), **rec)),
                                                encoding="utf-8")


def test_compare_names_the_tf_columns_that_differ_and_guards_the_batches(tmp_path):
    """Box 53693389's check 14: every hypothesis equal, the tf tables 3e-7 apart in their KL sums. Exact stays
    bitwise; tf_diff names the column with its n_diff and max_rel (ulp noise at a glance); equal tables on equal
    batches are the same; the same tables on other batches are not (batches_equal, a reason), a variant file's own
    weights and its quant source excepted. The number of chunks is not a batching key (the PR #35 review: 05's chunks
    are runs of whole batches, so they follow --chunk-s): the pass plan's batches_sha256 is, compared only when both
    set records have it (older 05 code wrote none)."""
    import numpy as np
    import pandas as pd

    refs = {"a": "あいうえお", "b": "かきくけこ"}
    hyps = {"eval_jsut": {"a": "あいうえお", "b": "かきくけ"}}
    one, two = _eval_dir(tmp_path / "one", hyps, refs), _eval_dir(tmp_path / "two", hyps, refs)
    tf = pd.DataFrame(dict(id=["a", "b"], kl=[0.25, 0.5], n_tok=[5, 4], top1=[float("nan"), 1.0]))
    tf.to_parquet(one / "tf_eval_jsut.parquet")
    tf2 = tf.copy()
    tf2.loc[1, "kl"] = float(np.nextafter(np.float64(0.5), 1.0))  # one ulp in one row of one column
    tf2.iloc[::-1].to_parquet(two / "tf_eval_jsut.parquet")  # another row order: compared by id
    r = Q.compare_eval_dirs(one, two)["sets"]["eval_jsut"]
    assert not r["same"] and r["hyps_equal"] and r["tf_equal"] is False and r["batches_equal"] is None
    assert set(r["tf_diff"]) == {"kl"} and r["tf_diff"]["kl"]["n_diff"] == 1 and r["tf_diff"]["kl"]["max_rel"] < 1e-15
    tf.iloc[::-1].to_parquet(two / "tf_eval_jsut.parquet")  # NaN equals NaN
    _parts(one, eval_jsut={})
    _parts(two, dict(weights="w2", step=0, quant=dict(fmt="int8-w8a8", impl="torchao", scope="linear+pw",
                                                       source="memory")), eval_jsut={})
    r = Q.compare_eval_dirs(one, two)["sets"]["eval_jsut"]
    assert r["same"] and r["tf_equal"] and r["tf_diff"] == {} and r["batches_equal"] is True
    _parts(two, eval_jsut=dict(chunks=11))  # the same batches cut into other chunks (another --chunk-s)
    r = Q.compare_eval_dirs(one, two)["sets"]["eval_jsut"]
    assert r["same"] and r["batches_equal"] is True, r
    for old_side in (one, two):  # a set record of older 05 code (no batches_sha256): compared without it
        _parts(old_side, eval_jsut=dict(batches_sha256=None))
        assert Q.compare_eval_dirs(one, two)["sets"]["eval_jsut"]["batches_equal"] is True
        _parts(old_side, eval_jsut={})
    for ident, rec, key in ((dict(batch_s=300.0), {}, "batch_s"), (dict(store="other"), {}, "store"),
                            ({}, dict(batches_sha256="c" * 64), "set.batches_sha256"),
                            ({}, dict(pass_sets=["eval_jsut", "x"]), "set.pass_sets")):
        _parts(two, ident, eval_jsut=rec)
        r = Q.compare_eval_dirs(one, two)
        s = r["sets"]["eval_jsut"]
        assert not r["same"] and s["batches_equal"] is False and key in s["reason"], (key, s)
        assert "set.chunks" not in s["reason"]
        assert Q.compare_eval_dirs(one, two, exact=False, tol_cer=0.0)["same"]  # the CER mode does not ask
    js = tmp_path / "cmp.json"
    assert Q.main(["compare", str(one), str(two), "--exact", "--json-out", str(js)]) == 1
    got = json.loads(js.read_text(encoding="utf-8"))["sets"]["eval_jsut"]
    assert got["batches_equal"] is False and got["tf_diff"] == {}


def test_the_pack_is_computed_on_the_cpu_and_installed_on_the_modules_device(monkeypatch):
    """pack_weight returns CPU tensors (CUDA's division by a Python scalar would round some nvfp4 / fp8 scales
    otherwise than the export does), and _install moves the pack to its module's device."""
    w = torch.randn(32, 64)
    for wf in Q.WFMTS:
        p = Q.pack_weight(w, wf)
        assert all(t.device.type == "cpu" for t in p.parts().values()), wf
        assert Q._packs_equal(p, Q.pack_weight(w.to(torch.bfloat16), wf)), wf
    seen = []
    real_to = Q.QuantPack.to
    monkeypatch.setattr(Q.QuantPack, "to", lambda self, dev: seen.append(torch.device(dev)) or real_to(self, dev))
    Q.quantize_linear(nn.Linear(64, 32), "nvfp4-w4a4", "emulate")
    assert seen == [torch.device("cpu")]


def test_the_selftests_pack_device_check_is_skipped_off_cuda():
    """The pack's device parity is a hard check of the GPU selftest (smoke B); off CUDA the selftest stops at its
    environment check, and the parity helper itself holds on the CPU."""
    rec = Q.selftest("cpu")
    assert [c["name"] for c in rec["checks"]] == ["environment"] and "device_arith" not in rec
    checks, rec = [], dict(warnings=[])
    Q._selftest_pack_device(rec, lambda name, ok, detail=None: checks.append((name, ok)), torch.device("cpu"), (16, 64))
    assert checks == [("pack_device_parity", True)]
    assert rec["device_arith"] == dict(rows=16, nvfp4_tensor_scale_rows_differ=0, fp8_fp32_scale_rows_differ=0,
                                       fp8_bf16_scale_rows_differ=0)


class _TorchaoDefaultFp8(nn.Module):
    """A stand-in for a real fp8-w8a8 Linear under torchao's DEFAULT config (no activation_value_lb): the activation's
    row scale bf16(amax / 448) with no clamp, so a zero row is 0 / 0 (_torchao_fp8_rows), the bias added outside."""

    def __init__(self, emu: nn.Linear):
        super().__init__()
        self.w, self.b = emu.weight.detach().float(), emu.bias.detach()
        self.weight = emu.weight

    def forward(self, x):
        q, s = _torchao_fp8_rows(x.reshape(-1, x.shape[-1]))
        y = torch.nn.functional.linear(q.float() * s[:, None], self.w).to(torch.bfloat16)
        return y + self.b.to(torch.bfloat16)


def test_selftest_zero_rows_catch_a_nan_real_path():
    """F4's zero-row sub-check (_selftest_zero_rows): an M = 17 input with two all-zero rows. Emulate against emulate
    passes for every timed format, its zero rows exactly the bias; a real fp8 layer with torchao's default recipe (no
    activation_value_lb: 0 / 0 on a zero row, box 53693389's NaN) fails "fp8-w8a8 zero rows"."""
    g = torch.Generator().manual_seed(9)
    k, n = 64, 48
    w, b = torch.randn(n, k, generator=g) * 0.02, torch.randn(n, generator=g) * 0.01

    def make(fmt):
        lin = nn.Linear(k, n)
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        return Q.quantize_linear(lin, fmt, "emulate")

    def run(fmt, real):
        rec, checks = dict(formats={}, warnings=[]), []
        Q._selftest_zero_rows(rec, lambda name, ok, detail=None: checks.append((name, ok)), torch.device("cpu"), fmt,
                              make(fmt), real, b, k, g)
        return rec, checks

    for fmt in Q.TIMED_FORMATS:
        rec, checks = run(fmt, make(fmt))
        zr = rec["formats"][fmt]["zero_rows"]
        assert checks == [(f"{fmt} zero rows", True)] and zr["zero_rows_equal_bias"] and zr["rel_err"] == 0.0
        assert zr["all_zero_finite"] and not rec["warnings"]
    rec, checks = run("fp8-w8a8", _TorchaoDefaultFp8(make("fp8-w8a8")))
    zr = rec["formats"]["fp8-w8a8"]["zero_rows"]
    assert checks == [("fp8-w8a8 zero rows", False)] and zr["finite"] is False and not zr["all_zero_finite"]
    assert "act_quant_kwargs" in zr and rec["warnings"]


_FP8_ROWS = Q._fp8_rows


def _pre_f1_act_fp8_rows(x, *, lb=None):
    """Q._fp8_rows with the activations (the calls with an lb) quantised by torchao's default config: no lb."""
    return _FP8_ROWS(x) if lb is None else _torchao_fp8_rows(x)


def test_selftest_padded_ckpt_on_cpu(dirs, monkeypatch):
    """F4's padded checkpoint sub-check (_selftest_ckpt, emulate in place of torchao on CPU), on a tiny CTC and AED
    student: every format's unpadded and padded checks pass, with both models' NonFiniteMonitor records. With the
    pre-F1 fp8 activation recipe (torchao's default: no lb, so a padded frame's zero row is 0 / 0) the padded fp8
    check fails as box 53693389's fp8 records did: every row but the batch's longest is non-finite."""
    for fam, rows in (("ctc", len(Q.CKPT_CTC_LENGTHS)), ("aed", len(Q.CKPT_AED_LENGTHS))):
        rec, checks = dict(ckpt={}, warnings=[]), []
        Q._selftest_ckpt(rec, lambda name, ok, detail=None: checks.append((name, ok)), torch.device("cpu"), dirs[fam],
                         impl="emulate")
        first = "finite" if fam == "ctc" else "decodes"
        assert checks == [(f"ckpt {fam} {f} {c}", True) for f in Q.TIMED_FORMATS for c in (first, "padded finite")]
        for f in Q.TIMED_FORMATS:
            p = rec["ckpt"][str(dirs[fam])][f]["padded"]
            assert p["nonfinite"]["rows"] == p["nonfinite_emulate"]["rows"] == 0 and p["nonfinite"]["forwards"] >= 1
        monkeypatch.setattr(Q, "TIMED_FORMATS", ("fp8-w8a8",))
        monkeypatch.setattr(Q, "_fp8_rows", _pre_f1_act_fp8_rows)
        rec, checks = dict(ckpt={}, warnings=[]), []
        Q._selftest_ckpt(rec, lambda name, ok, detail=None: checks.append((name, ok)), torch.device("cpu"), dirs[fam],
                         impl="emulate")
        assert checks[-1] == (f"ckpt {fam} fp8-w8a8 padded finite", False)
        assert rec["ckpt"][str(dirs[fam])]["fp8-w8a8"]["padded"]["nonfinite"]["rows"] == rows - 1
        monkeypatch.undo()


def test_selftest_compile_block_on_cpu():
    """F4's compile sub-check (_selftest_compile) on CPU: the conformer-convolution-like block of quantised Linears
    (emulate), torch.compile'd with aot_eager (dynamic) at B = 3 then B = 1, equals the eager block for every
    COMPILE_FORMATS entry, and dynamo compiled a graph for each (the B = 1 call recompiles: a new graph). This drives
    kitsune::row_major and the quantised Linears under tracing; inductor and torchao's kernels are smoke B's."""
    rec, checks = dict(warnings=[]), []
    Q._selftest_compile(rec, lambda name, ok, detail=None: checks.append((name, ok)), torch.device("cpu"),
                        impl="emulate", backend="aot_eager", block=dict(channels=64, frames=29))
    assert checks == [(f"compile {f} B={b}", True) for f in Q.COMPILE_FORMATS for b in (3, 1)]
    assert Q.COMPILE_FORMATS == ("int8-w8a8", "nvfp4-w4a4", "fp8-w8a8") and Q.COMPILE_BLOCK["batches"] == (3, 1)
    assert 3 * Q.COMPILE_BLOCK["frames"] % 8 == 7  # M = 999: 7 mod 8, box 53693389's int8 + compile case
    for f in Q.COMPILE_FORMATS:
        calls = rec["compile"]["formats"][f]["calls"]
        assert calls["B=3"]["graphs_new"] >= 1 and calls["B=1"]["graphs_new"] >= 1 and calls["B=1"]["rel_err"] == 0.0
        assert calls["B=3"]["rows"] == 87 and rec["compile"]["block"]["kernel"] == 9


def test_old_recipe_variants_are_refused(dirs, tmp_path):
    """F4: a variant exported by quant code older than RECIPE_VERSION (no recipe_version in its quantization.json:
    version 1, every export up to 8ff3bd5, smoke B #1's included) is never loaded: load_quantized raises with "export
    it again"; the current one loads."""
    out = tmp_path / "v"
    Q.export(dirs["ctc"], out, "fp8-w8a8")
    Q.load_quantized(out, "cpu")
    rec = json.loads((out / Q.QUANT_FILE).read_text(encoding="utf-8"))
    for old in (None, 1):
        r = dict(rec)
        if old is None:
            del r["recipe_version"]
        else:
            r["recipe_version"] = old
        (out / Q.QUANT_FILE).write_text(json.dumps(r), encoding="utf-8")
        assert Q.verify_export(out) == []  # the file is intact: only its recipe is old
        with pytest.raises(Q.QuantError, match="recipe version 1, not 2.*export it again"):
            Q.load_quantized(out, "cpu")


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
