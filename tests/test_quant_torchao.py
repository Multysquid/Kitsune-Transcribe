"""kitsune.quant against torchao 0.18 itself: torchao's recipe is the reference for every format it implements
(kitsune.quant, "torchao decides"), so the canonical pack must equal what torchao makes of the same bf16 weight, bit
for bit, and the torchao tensors kitsune builds from a pack must dequantise to exactly the pack. torchao is not on
the laptop: these run inside the training image (.github/workflows/image.yml runs this file and tests/test_quant.py
after the smoke; a failure blocks the image tag) and on smoke B. A mismatch is fixed in kitsune/quant.py's recipe
constants, never by widening a tolerance here.

  CPU (the image CI)   every torchao name kitsune uses resolves and every format's config builds (the MXFP4 AUTO
                       config of the selftest's decision-20 canary too); the int8 packs (weight-only and W8A8) equal
                       torchao's quantize_ of the same weight; the NVFP4 pack (on a real-sized weight, where the
                       reciprocal scaling decides codes) and the W4A4 activation grid, and the MXFP4 (RCEIL) pack,
                       equal torchao's own quantisers (they run on CPU in the image: an API error there fails, it is
                       drift; only a CPU-kernel refusal skips); the FP8 per-row pack where torchao runs it on CPU (it
                       refuses there today: smoke B's selftest covers it); an int8 to_torchao dequantises to the pack;
                       an int8 W8A8 Linear through torchao on CPU is reproduced bit for bit from kitsune's activation
                       grid (INT8_ACT_DIV, INT8_ACT_QMIN, INT8_ACT_SCALE_DTYPE), and the emulation is within the
                       selftest's tolerance of it; F1: the fp8 config carries activation_value_lb = FP8_ACT_LB, and
                       torchao's own float8 primitives (_choose_scale_float8 / _quantize_affine_float8, plain torch
                       ops) give kitsune's fp8 activation grid and weight pack bit for bit
  CUDA (smoke B, a     every timed format's to_torchao dequantises to its pack; torchao's fp8 PerRow quantize_ of a
  5090)                weight is kitsune's pack (F1); every W*A* format gives exactly the bias on an all-zero row (F4);
                       the compiled conformer-convolution block matches eager (F4); kitsune.quant.selftest passes with
                       every sub-check (real kernels within SELFTEST_TOL of the emulation at every M, no int8 fp32
                       fallback after padding, weights quantised under autocast, torchao parity, zero rows, compile, and
                       a tiny CTC and AED student unpadded and padded); a pack computed on CUDA equals the CPU one bit
                       for bit
"""
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

torchao = pytest.importorskip("torchao")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from kitsune import quant as Q  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="torchao's GPU kernels: smoke B (kitsune.quant selftest)")


def weight(n: int = 96, k: int = 128, seed: int = 0) -> torch.Tensor:
    """A bf16 weight with a spread of row and block magnitudes (and one all-zero row)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g) * torch.logspace(-3, 0, k)[None, :] * torch.logspace(-1, 1, n)[:, None]
    w[3] = 0.0
    return w.to(torch.bfloat16)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.element_size() != b.element_size():
        return False
    if a.element_size() == 1:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a.view(torch.int16 if a.element_size() == 2 else torch.int32),
                       b.view(torch.int16 if b.element_size() == 2 else torch.int32))


def torchao_linear(w: torch.Tensor, fmt: str, device="cpu") -> nn.Linear:
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(w.to(device))
    Q._ao("quantize_")(lin, Q.torchao_config(fmt))
    return lin


def test_every_name_and_config_resolves():
    assert Q.torchao_version() == torchao.__version__
    for fmt in Q.TORCHAO_CONFIGS:
        assert type(Q.torchao_config(fmt)).__name__ == Q.TORCHAO_CONFIGS[fmt]
    for name in ("quantize_", "PerRow", "MXDynamicActivationMXWeightConfig", "KernelPreference"):
        assert Q._ao(name) is not None
    assert Q.resolve_impl("int8-w8a8", "torchao", torch.device("cpu")) == "torchao"
    # the selftest's decision-20 canary: the MXFP4 W4A4 config with the AUTO kernel preference, as it builds it
    cfg = Q.mxfp4_auto_config()
    assert type(cfg).__name__ == "MXDynamicActivationMXWeightConfig"
    assert cfg.kernel_preference == Q._ao("KernelPreference").AUTO
    assert cfg.activation_dtype == cfg.weight_dtype == torch.float4_e2m1fn_x2
    # F1: a padded frame's all-zero activation row quantises to zeros, not 0 / 0 (torchao's hp_value_lb)
    assert Q.torchao_config("fp8-w8a8").activation_value_lb == Q.FP8_ACT_LB == 2.0 ** -40


def test_fp8_scale_equals_torchaos_primitives_on_cpu():
    """F1 against torchao 0.18's own float8 primitives (torchao.quantization.quant_primitives: plain torch ops, so they
    run on CPU in the image, where quantize_ refuses fp8): _choose_scale_float8 per row (block [1, K]) with
    hp_value_lb = FP8_ACT_LB, then _quantize_affine_float8, on bf16 activations with all-zero rows, give kitsune's
    scales and codes bit for bit (zero rows: zeros); without the lb, on a weight, the same on every non-zero row, and
    torchao's zero-row scale is 0 where kitsune keeps 1 (the documented deviation). Called by keyword, strict: a
    renamed argument is API drift and fails."""
    from torchao.quantization.quant_primitives import _choose_scale_float8, _quantize_affine_float8

    g = torch.Generator().manual_seed(5)
    x = (torch.randn(64, 256, generator=g) * torch.logspace(-5, 2, 64)[:, None]).to(torch.bfloat16)
    x[[3, 40]] = 0.0

    def theirs(t, lb):
        s = _choose_scale_float8(t, block_size=[1, t.shape[1]], float8_dtype=torch.float8_e4m3fn, hp_value_lb=lb)
        return _quantize_affine_float8(t, s, float8_dtype=torch.float8_e4m3fn), s.float().reshape(-1)

    tq, ts = _cpu_or_skip(lambda: theirs(x, Q.FP8_ACT_LB), "_choose_scale_float8 (activation)", strict=True)
    q, s = Q._fp8_rows(x.float(), lb=Q.FP8_ACT_LB)
    assert torch.equal(s, ts), "fp8 activation scales differ from torchao's"
    assert same_bits(q, tq.reshape(q.shape)), "fp8 activation codes differ from torchao's"
    assert (q[[3, 40]].float() == 0).all() and torch.isfinite(Q.fake_quant_act(x.float(), "fp8")).all()
    w = weight(96, 128)  # row 3 all zero
    wq, ws = _cpu_or_skip(lambda: theirs(w, None), "_choose_scale_float8 (weight)", strict=True)
    p = Q.pack_weight(w, "fp8")
    nz = [i for i in range(96) if i != 3]
    assert torch.equal(p.scale[nz], ws[nz]) and same_bits(p.qweight[nz], wq.reshape(p.qweight.shape)[nz])
    assert ws[3].item() == 0.0 and p.scale[3].item() == 1.0 and (p.qweight[3].float() == 0).all()


@pytest.mark.parametrize("fmt", ["int8-w8a16", "int8-w8a8"])
def test_int8_pack_equals_torchaos(fmt):
    """torchao's quantize_ of a bf16 Linear, read back (from_torchao), is kitsune's pack of the same weight, bit for
    bit: the int8 codes and the per-row scales (INT8_DIV, INT8_EPS)."""
    w = weight()
    theirs = Q.from_torchao(torchao_linear(w, fmt).weight, fmt)
    ours = Q.pack_weight(w, "int8")
    assert same_bits(theirs.qweight, ours.qweight), "int8 codes differ from torchao's"
    assert torch.equal(theirs.scale.float(), ours.scale), "int8 scales differ from torchao's"


def _cpu_or_skip(fn, what: str, *, strict: bool = False):
    """fn(), or a skip when torchao does not run it on CPU. strict (the quantisers known to run in the image: NVFP4,
    MXFP4): only a CPU-kernel refusal (NotImplementedError, RuntimeError) skips; a TypeError, AttributeError,
    AssertionError or QuantError is API drift, and fails."""
    refusals = (NotImplementedError, RuntimeError) + (() if strict else (AssertionError, AttributeError, TypeError,
                                                                           Q.QuantError))
    try:
        return fn()
    except refusals as e:
        pytest.skip(f"torchao {torchao.__version__} does not run {what} on CPU here ({type(e).__name__}: "
                    f"{str(e)[:160]}): smoke B's selftest covers it")


def real_weight(seed: int = 0) -> torch.Tensor:
    """A real-sized bf16 weight, N(0, 0.02) (2560 x 1024): on it the NVFP4 data scaling's form decides about a
    thousand codes (on weight() the division and torchao's reciprocal happen to agree everywhere)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(2560, 1024, generator=g) * 0.02).to(torch.bfloat16)


@pytest.mark.parametrize("which", ["spread", "real"])
def test_nvfp4_pack_equals_torchaos(which):
    """torchao's NVFP4 quantiser (two-level: the per-tensor amax / (448 x 6) scale, e4m3 block scales per 16, the data
    multiplied by (1 / t) / bs, packed E2M1) on the same bf16 weight gives kitsune's pack bit for bit."""
    w = weight() if which == "spread" else real_weight()
    t = _cpu_or_skip(lambda: Q._ao("per_tensor_amax_to_scale")(w.float().abs().max()), "per_tensor_amax_to_scale",
                     strict=True)
    scales, data = _cpu_or_skip(lambda: Q._ao("nvfp4_quantize")(w, block_size=16, per_tensor_scale=t),
                                "nvfp4_quantize", strict=True)
    ours = Q.pack_weight(w, "nvfp4")
    assert torch.equal(t.float().reshape(()), ours.tensor_scale), "the NVFP4 tensor scale differs"
    assert same_bits(scales.reshape(ours.block_scale.shape), ours.block_scale), "NVFP4 block scales differ"
    theirs = data.reshape(ours.qweight.shape).view(torch.uint8)
    n_diff = int((Q.unpack_nibbles(theirs) != Q.unpack_nibbles(ours.qweight)).sum())
    assert same_bits(theirs, ours.qweight), f"NVFP4 codes differ on {n_diff} of {w.numel()}"


def test_nvfp4_activation_grid_equals_torchaos():
    """The W4A4 activation: torchao's dynamic path quantises the call input with the tensor scale from its whole amax
    (per_tensor_amax_to_scale(max |x|)) through the same nvfp4_quantize; kitsune's fake-quant (_nvfp4 on the input
    rounded to bf16) gives the same tensor scale, block scales and codes."""
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(400, 1024, generator=g) * torch.logspace(-1, 1, 400)[:, None]).to(torch.bfloat16)
    t = _cpu_or_skip(lambda: Q._ao("per_tensor_amax_to_scale")(x.abs().max()), "per_tensor_amax_to_scale",
                     strict=True)
    scales, data = _cpu_or_skip(lambda: Q._ao("nvfp4_quantize")(x, 16, t), "nvfp4_quantize", strict=True)
    codes, bs, tt = Q._nvfp4(x.float())
    assert torch.equal(t.float().reshape(()), tt), "the activation's tensor scale differs"
    assert same_bits(scales.reshape(bs.shape), bs), "the activation's block scales differ"
    theirs = Q.unpack_nibbles(data.reshape(400, 512).view(torch.uint8))
    assert torch.equal(theirs, codes), f"the activation's codes differ on {int((theirs != codes).sum())}"


def test_mxfp4_pack_equals_torchaos_rceil():
    """torchao's MX quantiser in RCEIL mode (decision 20), E2M1 elements, blocks of 32, gives kitsune's pack."""
    w = weight()
    elem = getattr(torch, "float4_e2m1fn_x2", None)
    scale, data = _cpu_or_skip(lambda: Q._ao("to_mx")(w, elem, 32, scaling_mode=Q._ao("ScaleCalculationMode").RCEIL),
                               "to_mx (MXFP4, RCEIL)", strict=True)
    ours = Q.pack_weight(w, "mxfp4", mx_rounding="rceil")
    assert same_bits(scale.reshape(ours.block_scale.shape).view(torch.uint8), ours.block_scale), "E8M0 differs"
    assert same_bits(data.reshape(ours.qweight.shape).view(torch.uint8), ours.qweight), "MXFP4 codes differ"


@needs_cuda
def test_fp8_pack_equals_torchaos():
    """torchao's float8 per-row weight (Float8DynamicActivationFloat8WeightConfig(PerRow), quantize_ on the GPU: it
    refuses fp8 on CPU, so before F1 this test ran nowhere) is kitsune's pack on every non-zero row (F1: the
    bf16-rounded row scale); on the all-zero row kitsune keeps scale 1 and zero codes (torchao: scale 0)."""
    w = weight()
    lin = torchao_linear(w, "fp8-w8a8", device="cuda")
    assert lin.weight.act_quant_kwargs.hp_value_lb == Q.FP8_ACT_LB
    theirs = Q.from_torchao(lin.weight, "fp8-w8a8")
    ours = Q.pack_weight(w, "fp8")
    nz = [i for i in range(w.shape[0]) if i != 3]
    assert same_bits(theirs.qweight[nz], ours.qweight[nz]), "fp8 codes differ from torchao's"
    assert torch.equal(theirs.scale.float()[nz], ours.scale[nz]), "fp8 row scales differ from torchao's"
    assert ours.scale[3].item() == 1.0 and (ours.qweight[3].float() == 0).all()


def _dequant_equal(t: torch.Tensor, pack: Q.QuantPack):
    d = t.dequantize() if hasattr(t, "dequantize") else t
    want = Q.unpack_weight(pack.to("cpu"))
    got = d.detach().float().cpu()
    # torchao dequantises to its tensor's dtype: equal once rounded to it
    assert torch.equal(got, want.to(d.dtype).float()), f"{type(t).__name__} does not dequantise to the pack"


@pytest.mark.parametrize("fmt", ["int8-w8a16", "int8-w8a8"])
def test_to_torchao_int8_on_cpu(fmt):
    pack = Q.pack_weight(weight(seed=1), "int8")
    _dequant_equal(Q.to_torchao(pack, fmt, "cpu"), pack)


def _int8_act_candidates(x: torch.Tensor, pw: Q.QuantPack, yr: torch.Tensor) -> list:
    """The share of torchao's W8A8 outputs that each variant of the activation recipe (divisor, lowest code, scale
    dtype) and of the output's rescaling (fp32 dequantised GEMM; exact integer GEMM rescaled in fp32; in bf16 steps;
    by the bf16 product of the scales) reproduces bit for bit, best first."""
    import itertools

    qw = pw.qweight.double()
    out = []
    for div, qmin, sdt, form in itertools.product((127.0, 127.5), (-127, -128), (torch.bfloat16, torch.float32),
                                                  ("dequant_fp32", "int_fp32", "int_bf16_chain", "int_scale_bf16",
                                                   "int_to_bf16_first")):
        xf = x.float()
        sx = torch.clamp((xf.abs().amax(1) / div).to(sdt), min=Q.INT8_ACT_EPS).float()
        qx = torch.clamp(torch.round(xf / sx[:, None]), qmin, 127)
        yint = (qx.double() @ qw.T).float()
        sw = pw.scale[None, :]
        if form == "dequant_fp32":
            y = torch.nn.functional.linear(qx * sx[:, None], Q.unpack_weight(pw)).to(torch.bfloat16)
        elif form == "int_fp32":
            y = (yint * sx[:, None] * sw).to(torch.bfloat16)
        elif form == "int_bf16_chain":
            y = (yint * sx[:, None]).to(torch.bfloat16) * sw.to(torch.bfloat16)
        elif form == "int_scale_bf16":
            y = (yint * (sx[:, None] * sw).to(torch.bfloat16).float()).to(torch.bfloat16)
        else:
            y = yint.to(torch.bfloat16) * sx[:, None].to(torch.bfloat16) * sw.to(torch.bfloat16)
        out.append((round(float((y == yr).float().mean()), 4), div, qmin, str(sdt).replace("torch.", ""), form))
    return sorted(out, reverse=True)


def test_int8_w8a8_activation_recipe_is_torchaos():
    """A W8A8 Linear through torchao on CPU against kitsune's activation grid: with kitsune's recipe (per token, the
    scale INT8_ACT_SCALE_DTYPE(amax / INT8_ACT_DIV), codes clamped to [INT8_ACT_QMIN, 127]) some rescaling of the exact
    integer GEMM gives torchao's bf16 outputs bit for bit (the image CI of 772e3b7 found torchao rescaling in bf16); the
    emulation itself rescales in fp32 (one rounding, scout s7's emulate rule), so it is compared within the selftest's
    tolerance. (No bias: both paths add it the same way, outside the GEMM.)"""
    import json

    g = torch.Generator().manual_seed(2)
    w = weight(64, 128)
    real, emu = nn.Linear(128, 64, bias=False), nn.Linear(128, 64, bias=False)
    with torch.no_grad():
        real.weight.copy_(w.float())
        emu.weight.copy_(w.float())
    Q.quantize_linear(real, "int8-w8a8", "torchao")
    Q.quantize_linear(emu, "int8-w8a8", "emulate")
    x = (torch.randn(32, 128, generator=g) * torch.logspace(-2, 1, 32)[:, None]).to(torch.bfloat16)
    with torch.no_grad():
        yr, ye = real(x), emu(x)
    assert yr.dtype == ye.dtype == torch.bfloat16
    rel = float((yr.float() - ye.float()).norm() / ye.float().norm())
    cands = _int8_act_candidates(x, Q.pack_weight(w, "int8"), yr)
    mine = (Q.INT8_ACT_DIV, Q.INT8_ACT_QMIN, str(Q.INT8_ACT_SCALE_DTYPE).replace("torch.", ""))
    exact = [c for c in cands if c[1:4] == mine and c[0] == 1.0]
    if not exact or not rel < Q.SELFTEST_TOL["int8-w8a8"]:
        pytest.fail(json.dumps(dict(rel=rel, weight=type(real.weight).__name__, mine=mine, best=cands[:8])))


@needs_cuda
@pytest.mark.parametrize("fmt", [f for f in Q.TIMED_FORMATS])
def test_to_torchao_dequantizes_to_the_pack_on_cuda(fmt):
    wfmt = Q._SPLIT[fmt][0]
    pack = Q.pack_weight(weight(256, 512, seed=3).cuda(), wfmt)
    _dequant_equal(Q.to_torchao(pack, fmt, "cuda"), pack)


@needs_cuda
def test_the_pack_is_the_same_on_cuda_and_cpu():
    w = weight(256, 512, seed=4)
    for wfmt in Q.WFMTS:
        a, b = Q.pack_weight(w, wfmt), Q.pack_weight(w.cuda(), wfmt).to("cpu")
        for (ka, ta), (kb, tb) in zip(a.parts().items(), b.parts().items()):
            assert ka == kb and same_bits(ta, tb), (wfmt, ka)


@needs_cuda
@pytest.mark.parametrize("fmt", ["int8-w8a8", "nvfp4-w4a4", "fp8-w8a8"])
def test_w_a_formats_zero_rows_finite_on_cuda(fmt):
    """F4: a real W*A* Linear on an input with all-zero rows (a padded frame) gives exactly the bias there, finite,
    within SELFTEST_TOL of the emulation (_selftest_zero_rows; pre-F1 fp8 gave NaN)."""
    g = torch.Generator().manual_seed(6)
    w, b = torch.randn(256, 512, generator=g) * 0.02, torch.randn(256, generator=g) * 0.01

    def make(impl):
        lin = nn.Linear(512, 256, device="cuda")
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        return Q.quantize_linear(lin, fmt, impl)

    rec, checks = dict(formats={}, warnings=[]), []
    Q._selftest_zero_rows(rec, lambda name, ok, detail=None: checks.append((name, ok, detail)),
                          torch.device("cuda"), fmt, make("emulate"), make("torchao"), b.cuda(), 512, g)
    assert checks[0][1], checks


@needs_cuda
def test_compiled_conv_block_on_cuda():
    """F4: the conformer-convolution-like block of torchao Linears, torch.compile'd with inductor at B = 3 then B = 1
    (box 53693389's nvfp4 contiguity assert and int8 CUBLAS_STATUS_NOT_SUPPORTED), matches eager per format."""
    rec, checks = dict(warnings=[]), []
    Q._selftest_compile(rec, lambda name, ok, detail=None: checks.append((name, ok, detail)), torch.device("cuda"))
    assert [c[0] for c in checks] == [f"compile {f} B={b}" for f in Q.COMPILE_FORMATS for b in (3, 1)]
    assert all(c[1] for c in checks), [c for c in checks if not c[1]]


@needs_cuda
def test_the_selftest_passes_on_this_gpu(tmp_path):
    """Every sub-check of check 12, on tiny CTC and AED students for the checkpoint part (unpadded and padded)."""
    from test_quant import save_aed_dir, save_ctc_dir

    ckpt = [save_ctc_dir(tmp_path / "ctc"), save_aed_dir(tmp_path / "aed")]
    rec = Q.selftest("cuda:0", rows=(1, 7, 17, 128), ckpt=ckpt)
    assert rec["ok"], [c for c in rec["checks"] if not c["ok"]]
    names = {c["name"] for c in rec["checks"]}
    assert {"pack_device_parity", "torchao parity fp8-w8a8", "torchao parity int8-w8a8"} <= names
    assert {f"{f} zero rows" for f in Q.TIMED_FORMATS} <= names
    assert {f"compile {f} B={b}" for f in Q.COMPILE_FORMATS for b in (3, 1)} <= names
    assert {f"ckpt {d.name} {f} padded finite" for d in ckpt for f in Q.TIMED_FORMATS} <= names
    assert rec["recipe_version"] == Q.RECIPE_VERSION and rec["device_arith"]["fp8_bf16_scale_rows_differ"] == 0

