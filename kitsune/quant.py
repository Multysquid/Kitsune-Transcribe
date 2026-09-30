"""Quantised variants of the students: one packed form per layer, an emulated and a torchao runtime built from it, a
safetensors variant dir that loads back to the same numbers, and the checks the full runs score them with.

The full runs score every trained student in seven formats (the full-run build contract section 8; scout s7): what a
format costs in CER (scripts/05_evaluate.py --quant, the queue's `quant-<fmt>-<run>` readouts), what it saves in bytes
(the variant dir the readout exports and uploads) and what it buys in speed on the RTX 5090 (tools/speed_probe.py
--quant, smoke B). The report (tools/full_report.py) reads QUANT_FORMATS, FORMAT_INFO, split_system and the records
written here.

Formats (QUANT_FORMATS, in this order; FORMAT_INFO holds weights, acts, block, bits, real_impl, timed, export, label):

  fp16          Linear / Conv / Embedding weights and biases in IEEE fp16, norms and BatchNorm fp32, fp16 autocast. A
                portability check of the bf16-trained students (overflow shows in NonFiniteMonitor), not a speed option
  int8-w8a16    int8 weights per output channel (symmetric), bf16 activations            torchao Int8WeightOnlyConfig
  int8-w8a8     the same weights, int8 activations per token (dynamic, symmetric)       Int8DynamicActivationInt8Weight
  nvfp4-w4a16   E2M1 weights in blocks of 16 with an E4M3 scale each and one fp32 scale per tensor   NVFP4WeightOnly
  nvfp4-w4a4    the same weights, NVFP4 activations (the tensor scale from the whole call input)    NVFP4DynamicAct...
  mxfp4-w4a4    E2M1 weights and activations in blocks of 32 with a power-of-two (E8M0) scale: no kernel runs it on
                sm_120, so it is always emulated (decision 20) and never timed as a speed
  fp8-w8a8      E4M3 weights per output channel, E4M3 activations per token       Float8DynamicActivationFloat8Weight
The weight-only and weight+activation variant of a format share one packed weight: their files are byte-identical.
int8-cpu (torch.ao dynamic quantisation) is not built (owner answer A1).

Principles:
  one packed form   every quantised Linear has ONE canonical QuantPack, computed here by a pure-torch quantiser from its
                    weight rounded to bf16 (pack_weight). Every runtime is built from it: the emulated one (the exact
                    dequantised weight and a fake-quantised activation, fp32 matmul), the torchao one (the pack placed
                    into torchao's own tensor subclass, to_torchao) and the safetensors file (export). In memory is
                    bf16 W -> P -> module, from the file it is file -> P -> module: a reloaded variant gives the
                    in-memory outputs to the bit, by construction (tests/test_quant.py)
  torchao decides   for every format torchao implements, its recipe is the reference: the pack must equal what torchao
                    0.18's own quantize_ makes of the same bf16 weight, bit for bit (tests/test_quant_torchao.py, run
                    inside the training image by its CI and on smoke B, since the laptop has no torchao). A mismatch is
                    fixed here, in the recipe constants below, never by widening a tolerance
  no silent paths   --quant-impl auto never becomes emulate for a timed format on CUDA (resolve_impl raises without
                    torchao); a torchao error raises; the int8 small-M trap (torch._int_mm refuses M <= 16 on CUDA and
                    torchao silently falls back to an fp32 matmul) is padded away to 17 rows and counted
  module identity   a selected nn.Linear is class-swapped in place to QuantLinear (names, state_dict keys and hooks
                    stay); a pointwise Conv1d(k=1) of a conformer convolution module is replaced by
                    PointwiseConv1dAsLinear, whose inner `.linear` is then treated like every other Linear
  imports           torch and the stdlib only at import: torchao, transformers and safetensors are imported where
                    they are used, so the report can import this module on the laptop

Numerics (the exact recipes; the constants are the torchao parity test's to pin):
  every scale is computed in fp32 from the bf16-rounded weight, so the pack is the same bits on CPU and CUDA (the
  export runs on CPU); torch.round is round half to even; values are clamped to +-448 before any cast to e4m3
  E2M1          codes 0-7 = {0, .5, 1, 1.5, 2, 3, 4, 6}, bit 3 the sign (torch.signbit: -0.2 is code 8, as torchao);
                round to nearest, ties to the even code; two codes per byte, element 2i in the low nibble
  int8 weights  s = max(bf16(amax_row / INT8_DIV), INT8_EPS) (torchao keeps the weight's dtype for the scale:
                INT8_SCALE_DTYPE), q = clamp(round(w / s), -128, 127); dequant q * s
  int8 acts     per token: s = max(bf16(amax_row / INT8_ACT_DIV), INT8_ACT_EPS), q = clamp(round(x / s),
                INT8_ACT_QMIN, 127)
  nvfp4         t = amax(|w|) / (448 * 6) (an all-zero tensor: 1); per 16 along K bs = e4m3(clamp((b / 6) / t,
                NVFP4_SCALE_MIN, 448)); codes of clamp(w / (bs * t), +-6); dequant E2M1[c] * (bs * t). Activations
                (w4a4): the same with t from the amax of the whole call input (padding included, as torchao's dynamic
                per-tensor scale)
  mxfp4         per 32 along K, b the block amax: rceil e = ceil(log2(b / 6)) exactly through frexp; floor e =
                floor(log2 b) - 2; b == 0: e = -127; e clamped to [-127, 127] and stored as uint8 e + 127 (E8M0); codes
                of clamp(w * 2^-e, +-6) (e == -127 scales by 1, as torchao); dequant E2M1[c] * 2^e
  fp8           per row s = amax / 448 (0 -> 1), q = e4m3(clamp(w / s, +-448)); activations per token the same way
  bias          added outside the quantised GEMM, in every format and impl (cuBLASLt's FP4 GEMM failed at M=1 with a
                bias, pytorch #157054; and emulate and torchao then agree)
  emulate GEMM  the input is rounded to the autocast dtype when autocast is on (the real path sees autocast's bf16),
                then fake-quantised (W*A* formats), then F.linear in fp32 with autocast off on the exact dequantised
                weight (weight-only formats: rounded to bf16, which is what the real path dequantises to); the output
                is cast to the autocast dtype, and the bias added in it (as the real path adds it to its GEMM's output).
                torchao's int8 W8A8 rescales its exact integer GEMM in bf16 steps: the emulation (one fp32 rounding)
                differs from it by a bf16 ulp on some outputs, on the same activation codes (the parity test) The process's matmul precision applies: emulate and real are compared
                with tolerances only

Layer filter (select_layers): every nn.Linear except one whose weight is shared with another module (T-0.6B's
proj_out, tied to the token embedding) and names ending in proj_out or lm_head; scope "linear+pw" (the default,
decision 19) adds the pointwise_conv1/2 of every ParakeetEncoderConvolutionModule (never the CTC head, also a
Conv1d(k=1), nor the subsampling's Conv2d(k=1)). A layer whose shape the format's kernels cannot take (K % 16 for
nvfp4, K % 32 for mxfp4, K or N % 8 for int8/fp8) is `skipped` with the reason and stays 16-bit. Kept as they are: the
heads (ctc_head is read in fp32 at eval), the embeddings, the subsampling and depthwise convolutions, the norms, the
BatchNorm (its running statistics never change: fp32 from a master, bf16 after speed_probe's whole-model cast)
and the attention's rel-pos biases.

Variant dir (export; load_quantized reads it back, verify_export checks it):
  model.safetensors   plain tensors (no pickle). Every tensor that is not a quantised layer's weight under its HF key
                      in its dtype in the source file (a tied tensor once, under the embedding's key: `tied`); per
                      quantised layer <name>.qweight (int8 / packed fp4 as uint8 / e4m3), <name>.qscale (fp32, int8 and
                      fp8), <name>.qblock_scale (e4m3 for nvfp4; uint8 E8M0 for mxfp4: safetensors 0.8 stores neither
                      float8_e8m0fnu nor float4_e2m1fn_x2), <name>.qtensor_scale (fp32, nvfp4) and <name>.bias. fp16:
                      the HF keys and shapes, loadable with from_pretrained. The header's metadata holds no time, so
                      the bytes are deterministic and the two variants of one format are the same file
  quantization.json   the recipe (schema 1): format, weights, activations, scope, recipe constants, source (dir,
                      weights_sha256, family, architecture), layers {name: kind, shape, orig_shape, bias}, kept,
                      skipped, tied, fp16_tensors, counts, bytes, file_bytes, copied files with their sha256, the tensor
                      index digest, versions, load_with
  every other flat file of the source dir, byte-identical (config.json, the processor and tokenizer files,
  generation_config.json, README.md / MODEL_CARD.md with the source's card and licence, student_meta.json)

CLI (python -m kitsune.quant; exit codes of contract 1.5):
  export  --ckpt DIR --fmt F --out DIR [--scope] [--mx-rounding] [--device cpu] [--force]      0 / 1 / 2 refused
  readout --config C --ckpt BF16_DIR --fmt F --out OUT --cache-dir CACHE --manifest M [--max-temp 0] [--system S]
          [--impl auto]: exports to OUT/variant (unless verify_export passes there already), then runs
          scripts/05_evaluate.py in-process on it (--ckpt OUT/variant --out OUT --tables OUT/tables): what ships is
          what is scored. The system is <run_name>@<fmt>. 0 / 1 / 2 refused
  inspect DIR                                                                                          0 ok / 1
  compare A B [--exact | --tol-cer x] [--json-out FILE]   two 05 --out dirs, over the sets both hold: {same, sets}
                                                                                                0 same / 1 differ
  selftest --device cuda:0 --out FILE [--ckpt DIR ...] [--rows 1,7,17,128,1000]   smoke B check 12: real torchao
          kernels against emulate per format and M, the kernel census (every int8 GEMM returned after padding: no
          fp32 fallback), weights quantised and the low-precision GEMM run under autocast, MXFP4 refused by
          torchao's AUTO kernel. {ok, ...}                                                                  0 / 1
export, compare and selftest beat the item's heartbeat (kitsune.heartbeat.beating, max 1800 s; a no-op without
KITSUNE_HEARTBEAT); readout's 05 part beats through 05's featuriser.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

QUANT_FORMATS = ("fp16", "int8-w8a16", "int8-w8a8", "nvfp4-w4a16", "nvfp4-w4a4", "mxfp4-w4a4", "fp8-w8a8")
SCOPES = ("linear+pw", "linear")
IMPLS = ("auto", "torchao", "emulate")
MX_ROUNDINGS = ("rceil", "floor")
SCHEMA = 1
QUANT_FILE = "quantization.json"
WEIGHTS_FILE = "model.safetensors"
COMPILE_SUFFIX = "+compile"  # speed records of a torch.compile'd model: <base>[@<fmt>]+compile

FORMAT_INFO = {
    "fp16": dict(weights="fp16", acts="fp16", block=None, bits=16, real_impl="native fp16", timed=True, export=True,
                 label="FP16"),
    "int8-w8a16": dict(weights="int8", acts="bf16", block="row", bits=8, real_impl="torchao Int8WeightOnlyConfig",
                       timed=True, export=True, label="INT8 W8A16"),
    "int8-w8a8": dict(weights="int8", acts="int8", block="row", bits=8,
                      real_impl="torchao Int8DynamicActivationInt8WeightConfig", timed=True, export=True,
                      label="INT8 W8A8"),
    "nvfp4-w4a16": dict(weights="nvfp4", acts="bf16", block=16, bits=4, real_impl="torchao NVFP4WeightOnlyConfig",
                        timed=True, export=True, label="NVFP4 W4A16"),
    "nvfp4-w4a4": dict(weights="nvfp4", acts="nvfp4", block=16, bits=4,
                       real_impl="torchao NVFP4DynamicActivationNVFP4WeightConfig", timed=True, export=True,
                       label="NVFP4 W4A4"),
    "mxfp4-w4a4": dict(weights="mxfp4", acts="mxfp4", block=32, bits=4, real_impl=None, timed=False, export=True,
                       label="MXFP4 W4A4 (simulated)"),
    "fp8-w8a8": dict(weights="fp8", acts="fp8", block="row", bits=8,
                     real_impl="torchao Float8DynamicActivationFloat8WeightConfig(PerRow)", timed=True, export=True,
                     label="FP8 W8A8"),
}
# fmt -> (the packed weight's format, the activation's fake-quant or None)
_SPLIT = {"int8-w8a16": ("int8", None), "int8-w8a8": ("int8", "int8"), "nvfp4-w4a16": ("nvfp4", None),
          "nvfp4-w4a4": ("nvfp4", "nvfp4"), "mxfp4-w4a4": ("mxfp4", "mxfp4"), "fp8-w8a8": ("fp8", "fp8")}
WFMTS = ("int8", "nvfp4", "mxfp4", "fp8")
ACTIVATIONS = {"fp16": "fp16", "int8-w8a16": "bf16", "int8-w8a8": "int8-per-token-dynamic", "nvfp4-w4a16": "bf16",
               "nvfp4-w4a4": "nvfp4-dynamic", "mxfp4-w4a4": "mxfp4-dynamic", "fp8-w8a8": "fp8-per-token-dynamic"}
# torchao 0.18's config class per format it implements (the only torchao names outside the adapter's lookups)
TORCHAO_CONFIGS = {"int8-w8a16": "Int8WeightOnlyConfig", "int8-w8a8": "Int8DynamicActivationInt8WeightConfig",
                   "nvfp4-w4a16": "NVFP4WeightOnlyConfig", "nvfp4-w4a4": "NVFP4DynamicActivationNVFP4WeightConfig",
                   "fp8-w8a8": "Float8DynamicActivationFloat8WeightConfig"}
# where torchao keeps them (its prototype namespaces move between releases; the first module that has a name wins)
_AO_MODULES = ("torchao.quantization", "torchao.prototype.mx_formats",
               "torchao.prototype.mx_formats.inference_workflow", "torchao.prototype.mx_formats.nvfp4_tensor", "torchao.prototype.mx_formats.mx_tensor",
               "torchao.prototype.mx_formats.utils", "torchao.prototype.mx_formats.config",
               "torchao.quantization.granularity", "torchao.quantization.quantize_.common")

# ------------------------------------------------------------------------------------------ recipe constants
# (pinned by tests/test_quant_torchao.py against torchao 0.18: a mismatch is fixed here)
F4_MAX = 6.0  # the largest E2M1 magnitude
F8_MAX = 448.0  # the largest float8_e4m3fn magnitude
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
INT8_DIV = 127.5  # torchao choose_qparams_affine SYMMETRIC over [-128, 127]: amax / ((127 - -128) / 2)
INT8_EPS = float(torch.finfo(torch.float32).eps)
# torchao computes an int8 scale in the tensor's own dtype (bf16), then clamps it to eps: the scale is rounded to it
# before the codes are computed, and the file's F32 holds it exactly. Measured by the parity test in the image (CI of
# 4278cea: torchao 0.18's int8 weight scales were bf16(amax / 127.5) on every row, fp32 ones differed)
INT8_SCALE_DTYPE = torch.bfloat16
# torchao 0.18's int8 activation (Int8Tensor, per token): the scale bf16(amax / 127.5) like the weight's, the codes
# clamped to [-127, 127]. Measured in the image (CI of 772e3b7): this emulation gives torchao's CPU W8A8 outputs bit
# for bit, where / 127 matched none and [-128, 127] three in four
INT8_ACT_DIV = 127.5
INT8_ACT_QMIN = -127
INT8_ACT_EPS = 1e-5
INT8_ACT_SCALE_DTYPE = torch.bfloat16
NVFP4_SCALE_MIN = float(torch.finfo(torch.float8_e4m3fn).tiny)  # torchao nvfp4_quantize clamps the block scale here
NVFP4_BLOCK, MXFP4_BLOCK = 16, 32
E8M0_BIAS = 127
PAD_ROWS = 17  # torch._int_mm refuses M <= 16 on CUDA: an int8-activation torchao call is padded to this many rows
# smoke B's selftest: the relative Frobenius error of a real torchao Linear against its emulation
SELFTEST_TOL = {"int8-w8a16": 5e-3, "int8-w8a8": 5e-3, "nvfp4-w4a16": 1e-2, "nvfp4-w4a4": 2e-2, "fp8-w8a8": 1e-2}
TIMED_FORMATS = tuple(f for f in QUANT_FORMATS if f != "fp16" and FORMAT_INFO[f]["timed"])
BEAT_MAX_S = 1800  # the heartbeat bound of export / compare / selftest (contract 8)
LOAD_WITH = ("kitsune.quant.load_quantized (transformers from_pretrained would re-initialise the missing .weight "
             "tensors)")
# source files a variant dir does not copy: the weights (re-written) and anything pickled
_NOT_COPIED = (".safetensors", ".safetensors.index.json", ".bin", ".pt", ".pth", ".tmp")
_PACK_PARTS = {"int8": ("qweight", "qscale"), "fp8": ("qweight", "qscale"),
               "nvfp4": ("qweight", "qblock_scale", "qtensor_scale"), "mxfp4": ("qweight", "qblock_scale")}


class QuantError(ValueError):
    """A quantisation that cannot be done as asked (the CLI: exit 1)."""


class QuantRefused(QuantError):
    """A request refused before any work (an existing --out, an unknown format; the CLI: exit 2)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------- names


def check_format(fmt: str, *, allow_none: bool = False) -> str:
    if fmt in QUANT_FORMATS or (allow_none and fmt == "none"):
        return fmt
    raise QuantRefused(f"unknown quant format {fmt!r}: one of {QUANT_FORMATS}"
                       + (" (int8-cpu is not built: owner answer A1)" if fmt == "int8-cpu" else ""))


def system_name(base: str, fmt: str) -> str:
    """The system a variant is scored and timed under: <base>@<fmt> (a speed record of a compiled model adds
    +compile). The suffix is required: 05's publish_tables replaces a system's tables, so a variant under the base
    name would wipe the bf16 tables."""
    return f"{base}@{check_format(fmt)}"


def split_system(name: str) -> tuple[str, str | None]:
    """<base>@<fmt>[+compile] -> (base, fmt); a name without a known format suffix -> (name less +compile, None)."""
    s = name[:-len(COMPILE_SUFFIX)] if name.endswith(COMPILE_SUFFIX) else name
    base, sep, fmt = s.rpartition("@")
    if sep and base and fmt in QUANT_FORMATS:
        return base, fmt
    return s, None


def torchao_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("torchao")
    except PackageNotFoundError:
        return None


def torchao_available() -> bool:
    try:
        import torchao  # noqa: F401
    except Exception:  # noqa: BLE001 - absent, or a broken extension: not usable either way
        return False
    return True


def resolve_impl(fmt: str, impl: str, device) -> str:
    """The runtime a format gets: fp16 "native"; mxfp4 always "emulate" (no kernel on sm_120, decision 20); the
    others "emulate" when asked, else "torchao" (auto: on CUDA; on CPU auto emulates). A CUDA auto without torchao,
    or an explicit torchao that cannot be imported, raises: a timed format never falls back to emulation silently."""
    check_format(fmt)
    if impl not in IMPLS:
        raise QuantRefused(f"--quant-impl must be one of {IMPLS}, got {impl!r}")
    dev = torch.device(device)
    if fmt == "fp16":
        if impl != "auto":
            raise QuantRefused(f"fp16 has one implementation (plain fp16 weights under fp16 autocast): --quant-impl "
                               f"{impl} does not apply")
        return "native"
    if fmt == "mxfp4-w4a4":
        if impl == "torchao":
            raise QuantRefused("mxfp4-w4a4 has no torchao kernel on this hardware (decision 20): it is emulated")
        return "emulate"
    if impl == "emulate":
        return "emulate"
    if impl == "torchao" or dev.type == "cuda":
        if not torchao_available():
            raise QuantError(f"{fmt} on {dev} needs torchao (requirements-train.txt pins torchao==0.18.0), which is "
                             "not importable here; --quant-impl emulate simulates the format instead (never timed)")
        return "torchao"
    return "emulate"


def identity(fmt: str, impl: str, scope: str, mx_rounding: str) -> dict:
    """The block a 05 --out identity gains for a quantised eval (only then: existing --out dirs keep theirs)."""
    rec = dict(fmt=fmt, impl=impl, scope=scope, mx_rounding=mx_rounding, schema=SCHEMA)
    if impl == "torchao":
        rec["torchao"] = torchao_version()
    return rec


# ---------------------------------------------------------------------------------------------- E2M1 / E8M0


def e2m1_encode(v: torch.Tensor) -> torch.Tensor:
    """fp values -> uint8 E2M1 codes (0-15): |v| clamped to 6, rounded to the nearest of E2M1_VALUES with ties to the
    even code, bit 3 = torch.signbit (so -0.0 and a negative value rounding to 0 are code 8, as torchao)."""
    v = v.float()
    a = v.abs().clamp(max=F4_MAX).contiguous()
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    lo = torch.bucketize(a, mids, right=False)
    hi = torch.bucketize(a, mids, right=True)
    idx = torch.where(lo == hi, lo, torch.where(lo % 2 == 0, lo, hi))  # at a midpoint: the even of the two codes
    return idx.to(torch.uint8) | (torch.signbit(v).to(torch.uint8) << 3)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """uint8 E2M1 codes -> fp32 by table lookup (float4_e2m1fn_x2 has no CPU conversion)."""
    table = torch.tensor(E2M1_VALUES + tuple(-x for x in E2M1_VALUES), dtype=torch.float32, device=codes.device)
    return table[codes.long()]


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """(..., K) uint8 codes (K even) -> (..., K/2) uint8: element 2i in the low nibble, 2i+1 in the high one (the
    float4_e2m1fn_x2 / torchao pack_uint4 order)."""
    if codes.shape[-1] % 2:
        raise QuantError(f"cannot pack an odd number of E2M1 codes ({codes.shape[-1]})")
    c = codes.to(torch.uint8)
    return (c[..., 0::2] & 0xF) | ((c[..., 1::2] & 0xF) << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """(..., K/2) uint8 -> (..., K) uint8 codes (pack_nibbles' inverse)."""
    p = packed.to(torch.uint8)
    return torch.stack([p & 0xF, p >> 4], dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)


def mx_exponent(b: torch.Tensor, rounding: str = "rceil") -> torch.Tensor:
    """The unbiased E8M0 exponent of MXFP4 blocks from their amax b (>= 0), int32 in [-127, 127]. rceil (decision 20;
    torchao's): ceil(log2(b / 6)), exact through frexp (b / 6 = m 2^E with m in [0.5, 1): E - 1 when m == 0.5, else
    E); floor (the MX paper): floor(log2 b) - 2. b == 0 gives -127. Never .to(float8_e8m0fnu), which rounds."""
    if rounding not in MX_ROUNDINGS:
        raise QuantRefused(f"--mx-rounding must be one of {MX_ROUNDINGS}, got {rounding!r}")
    b = b.float()
    if rounding == "rceil":
        m, e = torch.frexp(b / F4_MAX)
        e = torch.where(m == 0.5, e - 1, e)
    else:
        m, e = torch.frexp(b)
        e = e - 3  # floor(log2 b) = E - 1, minus 2 (6 = 1.5 * 2^2, the largest E2M1 power)
    e = torch.where(b > 0, e, torch.full_like(e, -E8M0_BIAS))
    return e.clamp(-E8M0_BIAS, E8M0_BIAS).to(torch.int32)


# ---------------------------------------------------------------------------------------------- packing


@dataclass
class QuantPack:
    """One quantised weight, the canonical form every runtime and the file are built from. qweight: int8 (N, K) for
    int8, uint8 (N, K/2) packed E2M1 for nvfp4 / mxfp4, float8_e4m3fn (N, K) for fp8; scale: fp32 (N,) per row (int8,
    fp8); block_scale: float8_e4m3fn (N, K/16) (nvfp4) or uint8 E8M0 (N, K/32) (mxfp4); tensor_scale: fp32 0-dim
    (nvfp4); mx_rounding: the E8M0 rounding (mxfp4)."""

    wfmt: str
    shape: tuple
    qweight: torch.Tensor
    scale: torch.Tensor | None = None
    block_scale: torch.Tensor | None = None
    tensor_scale: torch.Tensor | None = None
    mx_rounding: str | None = None

    def parts(self) -> dict[str, torch.Tensor]:
        """The file's tensors of this layer, by suffix (_PACK_PARTS order)."""
        have = dict(qweight=self.qweight, qscale=self.scale, qblock_scale=self.block_scale,
                    qtensor_scale=self.tensor_scale)
        return {k: have[k] for k in _PACK_PARTS[self.wfmt]}

    def to(self, device) -> "QuantPack":
        mv = (lambda t: None if t is None else t.to(device))  # noqa: E731
        return QuantPack(self.wfmt, tuple(self.shape), mv(self.qweight), mv(self.scale), mv(self.block_scale),
                         mv(self.tensor_scale), self.mx_rounding)

    def nbytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.parts().values()))


def _int8_rows(x: torch.Tensor, div: float, eps: float, qmin: int, scale_dtype=torch.float32
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8 (torchao's choose_qparams_affine order): the scale amax / div rounded to scale_dtype,
    then clamped to eps (an all-zero row gets eps), the codes round(x / s) in fp32, clamped to [qmin, 127]."""
    amax = x.abs().amax(dim=1)
    s = torch.clamp((amax / div).to(scale_dtype), min=eps).float()
    q = torch.clamp(torch.round(x / s[:, None]), qmin, 127)
    return q, s


def _fp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax(dim=1)
    s = torch.where(amax > 0, amax / F8_MAX, torch.ones_like(amax))
    q = (x / s[:, None]).clamp(-F8_MAX, F8_MAX).to(torch.float8_e4m3fn)
    return q, s


def _nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(codes (N, K) uint8, block scales (N, K/16) e4m3, tensor scale fp32 0-dim) of an fp32 (N, K)."""
    n, k = x.shape
    amax = x.abs().amax()
    t = torch.where(amax > 0, amax / (F8_MAX * F4_MAX), torch.ones_like(amax))
    blocks = x.reshape(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    b = blocks.abs().amax(dim=-1)
    bs = ((b / F4_MAX) / t).clamp(NVFP4_SCALE_MIN, F8_MAX).to(torch.float8_e4m3fn)
    d = (blocks / (bs.float() * t)[..., None]).clamp(-F4_MAX, F4_MAX)
    return e2m1_encode(d).reshape(n, k), bs, t.float()


def _nvfp4_dequant(codes: torch.Tensor, bs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    n, k = codes.shape
    vals = e2m1_decode(codes).reshape(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    return (vals * (bs.float() * t.float())[..., None]).reshape(n, k)


def _mxfp4(x: torch.Tensor, rounding: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes (N, K) uint8, unbiased exponents (N, K/32) int32) of an fp32 (N, K)."""
    n, k = x.shape
    blocks = x.reshape(n, k // MXFP4_BLOCK, MXFP4_BLOCK)
    e = mx_exponent(blocks.abs().amax(dim=-1), rounding)
    # 2^-e, exact; an E8M0 of 0 (e = -127) scales by 1 as torchao does (only an all-zero block gets there in practice)
    f = torch.where(e == -E8M0_BIAS, torch.ones_like(e, dtype=torch.float32),
                    torch.ldexp(torch.ones_like(e, dtype=torch.float32), -e))
    d = (blocks * f[..., None]).clamp(-F4_MAX, F4_MAX)
    return e2m1_encode(d).reshape(n, k), e


def _mxfp4_dequant(codes: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    n, k = codes.shape
    vals = e2m1_decode(codes).reshape(n, k // MXFP4_BLOCK, MXFP4_BLOCK)
    return torch.ldexp(vals, e.to(torch.int32)[..., None].expand_as(vals)).reshape(n, k)


def align_problem(wfmt: str, n: int, k: int) -> str | None:
    """Why the format's kernels cannot take an (N, K) weight, or None."""
    if wfmt == "nvfp4" and k % NVFP4_BLOCK:
        return f"K={k} is not a multiple of {NVFP4_BLOCK} (nvfp4 blocks)"
    if wfmt == "mxfp4" and k % MXFP4_BLOCK:
        return f"K={k} is not a multiple of {MXFP4_BLOCK} (mxfp4 blocks)"
    if wfmt in ("int8", "fp8") and (k % 8 or n % 8):
        return f"(N, K)=({n}, {k}): the {wfmt} kernels need N and K multiples of 8"
    return None


def pack_weight(w: torch.Tensor, wfmt: str, *, mx_rounding: str = "rceil") -> QuantPack:
    """The canonical pack of a 2-D weight, from the weight rounded to bf16 (a bf16 and an fp32 copy of the same bf16
    values give the same bits). Non-finite weights raise QuantError."""
    if wfmt not in WFMTS:
        raise QuantRefused(f"unknown weight format {wfmt!r}: one of {WFMTS}")
    if w.dim() != 2:
        raise QuantError(f"pack_weight takes a 2-D weight, got {tuple(w.shape)}")
    n, k = (int(x) for x in w.shape)
    if wfmt in ("nvfp4", "mxfp4") and (why := align_problem(wfmt, n, k)):
        raise QuantError(why)  # the blocks need it; the int8 / fp8 kernels' N, K % 8 is select_layers' to apply
    wb = w.detach().to(torch.bfloat16).float()
    if not bool(torch.isfinite(wb).all()):
        raise QuantError(f"a weight of shape {(n, k)} holds non-finite values: it cannot be quantised")
    if wfmt == "int8":
        q, s = _int8_rows(wb, INT8_DIV, INT8_EPS, -128, scale_dtype=INT8_SCALE_DTYPE)
        return QuantPack("int8", (n, k), q.to(torch.int8), scale=s.float())
    if wfmt == "fp8":
        q, s = _fp8_rows(wb)
        return QuantPack("fp8", (n, k), q, scale=s.float())
    if wfmt == "nvfp4":
        codes, bs, t = _nvfp4(wb)
        return QuantPack("nvfp4", (n, k), pack_nibbles(codes), block_scale=bs, tensor_scale=t)
    codes, e = _mxfp4(wb, mx_rounding)
    return QuantPack("mxfp4", (n, k), pack_nibbles(codes), block_scale=(e + E8M0_BIAS).to(torch.uint8),
                     mx_rounding=mx_rounding)


def unpack_weight(p: QuantPack) -> torch.Tensor:
    """The exact fp32 dequantisation of a pack."""
    if p.wfmt in ("int8", "fp8"):
        return p.qweight.float() * p.scale.float()[:, None]
    codes = unpack_nibbles(p.qweight)
    if p.wfmt == "nvfp4":
        return _nvfp4_dequant(codes, p.block_scale, p.tensor_scale)
    if p.wfmt == "mxfp4":
        return _mxfp4_dequant(codes, p.block_scale.to(torch.int32) - E8M0_BIAS)
    raise QuantError(f"unknown weight format {p.wfmt!r}")


def fake_quant_act(x: torch.Tensor, act: str, *, mx_rounding: str = "rceil") -> torch.Tensor:
    """An activation quantised and dequantised on the format's exact grid, fp32, x's shape: int8 and fp8 per token
    (a row of the flattened (M, K) input), nvfp4 with the whole call's amax as its tensor scale (a batch-mate with a
    larger amax changes a row's grid: the W4A4 trap hyp_diff_1 measures), mxfp4 per 32 along K."""
    k = x.shape[-1]
    x2 = x.float().reshape(-1, k)
    if x2.shape[0] == 0:
        return x2.reshape(x.shape)
    if act == "int8":
        q, s = _int8_rows(x2, INT8_ACT_DIV, INT8_ACT_EPS, INT8_ACT_QMIN, scale_dtype=INT8_ACT_SCALE_DTYPE)
        y = q * s[:, None]
    elif act == "fp8":
        q, s = _fp8_rows(x2)
        y = q.float() * s[:, None]
    elif act == "nvfp4":
        codes, bs, t = _nvfp4(x2)
        y = _nvfp4_dequant(codes, bs, t)
    elif act == "mxfp4":
        codes, e = _mxfp4(x2, mx_rounding)
        y = _mxfp4_dequant(codes, e)
    else:
        raise QuantError(f"unknown activation format {act!r}")
    return y.reshape(x.shape)


# ---------------------------------------------------------------------------------------------- modules


@dataclass
class QuantState:
    """A QuantLinear's format, runtime and counters (calls, rows seen, calls padded to min_rows, int8-activation
    torchao calls that reached the GEMM with M <= 16: torchao's silent fp32 fallback, structurally 0)."""

    fmt: str
    impl: str
    wfmt: str
    act: str | None
    block: object
    kind: str
    name: str
    min_rows: int = 0
    count: bool = True
    calls: int = 0
    rows: int = 0
    padded: int = 0
    fallback_risk: int = 0
    mx_rounding: str = "rceil"


def _weight_only_dequant(p: QuantPack) -> torch.Tensor:
    """The emulated weight of a weight-only format: the dequantisation rounded to bf16, which is what the real
    kernels dequantise to (int8 / nvfp4 weight-only compute in bf16)."""
    return unpack_weight(p).to(torch.bfloat16).float()


class QuantLinear(nn.Linear):
    """An nn.Linear whose class was swapped in place (apply / load_quantized): the same module, name, state_dict
    keys and hooks, with .kq (QuantState) and a weight that is either the exact fp32 dequantisation (emulate: the
    activation is fake-quantised per call) or torchao's tensor subclass built from the pack (torchao). The bias is
    added outside the quantised GEMM. Never constructed directly."""

    kq: QuantState

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kq = self.kq
        dev = x.device.type
        ac = torch.is_autocast_enabled(dev)
        adt = torch.get_autocast_dtype(dev) if ac else None
        out_dtype = adt if ac else x.dtype
        k, n = self.in_features, self.out_features
        lead = x.shape[:-1]
        x2 = x.reshape(-1, k)
        m = int(x2.shape[0])
        if kq.count:
            kq.calls += 1
            kq.rows += m
        if m == 0:
            return x.new_zeros(*lead, n, dtype=out_dtype)
        if m < kq.min_rows:
            x2 = torch.cat([x2, x2.new_zeros(kq.min_rows - m, k)])
            if kq.count:
                kq.padded += 1
        if kq.impl == "emulate":
            xe = x2.to(adt).float() if ac else x2.float()
            if kq.act:
                xe = fake_quant_act(xe, kq.act, mx_rounding=kq.mx_rounding)
            with torch.autocast(device_type=dev, enabled=False):
                y = F.linear(xe, self.weight.float())
        else:
            if kq.count and kq.act == "int8" and dev == "cuda" and x2.shape[0] <= 16:
                kq.fallback_risk += 1
            y = F.linear(x2.to(self.weight.dtype), self.weight)
        # the GEMM's output in the output dtype first, then the bias in it: what the real path does (its GEMM returns
        # the autocast dtype), so emulate rounds where it rounds
        y = y[:m].to(out_dtype)
        if self.bias is not None:
            y = y + self.bias.to(out_dtype)
        return y.reshape(*lead, n)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, fmt={self.kq.fmt}, impl={self.kq.impl}"


class PointwiseConv1dAsLinear(nn.Module):
    """A pointwise Conv1d(k=1) of a conformer convolution module as a Linear over the channels: .linear holds the
    conv's weight[:, :, 0] (bitwise) and bias; forward (B, C, T) -> linear(x.transpose(1, 2)).transpose(1, 2), exact up
    to the accumulation order. The layer's name becomes <conv>.linear."""

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        n, k, _ = conv.weight.shape
        lin = nn.Linear(k, n, bias=conv.bias is not None, device="meta")
        lin.weight = nn.Parameter(conv.weight.detach()[:, :, 0].clone(), requires_grad=conv.weight.requires_grad)
        if conv.bias is not None:
            lin.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=conv.bias.requires_grad)
        self.linear = lin
        self.in_channels, self.out_channels = k, n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


def _is_pointwise(m: nn.Module, parent: nn.Module | None) -> bool:
    return (isinstance(m, nn.Conv1d) and tuple(m.kernel_size) == (1,) and m.groups == 1 and tuple(m.stride) == (1,)
            and tuple(m.padding) == (0,) and tuple(m.dilation) == (1,)
            and type(parent).__name__ == "ParakeetEncoderConvolutionModule")


@dataclass
class LayerSel:
    """A layer select_layers picked: its name after quantisation (a pointwise conv's is <conv>.linear), its module's
    name in the HF model, kind "linear" or "pointwise_conv1d", the (N, K) of its Linear, the HF weight's shape and
    whether it has a bias."""

    name: str
    source: str
    kind: str
    shape: tuple
    orig_shape: list
    bias: bool


def _parents(model: nn.Module) -> dict[str, nn.Module]:
    out = {}
    for pname, p in model.named_modules():
        for cname, _ in p.named_children():
            out[f"{pname}.{cname}" if pname else cname] = p
    return out


def _kept_reason(name: str, m: nn.Module) -> str:
    if name.endswith("ctc_head"):
        return "CTC head (fp32 at eval)"
    if name.endswith(("proj_out", "lm_head")):
        return "LM head (tied to the token embedding; fp32 at eval)"
    if isinstance(m, nn.Embedding):
        return "position embedding" if name.endswith("pos_emb") else "embedding"
    if isinstance(m, nn.Conv2d):
        return "subsampling conv"
    if isinstance(m, nn.Conv1d):
        return "depthwise conv" if m.groups > 1 else "pointwise conv (scope linear)"
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        return "batchnorm (running statistics never changed)"
    if isinstance(m, nn.LayerNorm):
        return "norm"
    if isinstance(m, nn.Linear):
        return "shares its weight with another module"
    return f"{type(m).__name__} parameters"


def select_layers(model: nn.Module, fmt: str, scope: str = "linear+pw"
                  ) -> tuple[list[LayerSel], dict[str, str], dict[str, str]]:
    """(selected, kept, skipped) of a model for a format (module docstring, "Layer filter"): kept maps every other
    module holding parameters to why it stays; skipped maps an in-scope layer the format's kernels cannot take to
    the reason."""
    if check_format(fmt) == "fp16":
        raise QuantError("fp16 casts every Linear / Conv / Embedding: it selects no layers to pack")
    if scope not in SCOPES:
        raise QuantRefused(f"--quant-scope must be one of {SCOPES}, got {scope!r}")
    wfmt = _SPLIT[fmt][0]
    names: dict[int, list[str]] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        names.setdefault(id(p), []).append(n)
    shared = {pid for pid, ns in names.items() if len(ns) > 1}
    parents = _parents(model)
    selected, kept, skipped = [], {}, {}
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            raise QuantError(f"{name} is quantised already: apply a format to a model once")
        if not name:
            continue
        if isinstance(m, nn.Linear):
            if name.endswith(("proj_out", "lm_head")) or id(m.weight) in shared:
                kept[name] = _kept_reason(name, m)
                continue
            n, k = (int(x) for x in m.weight.shape)
            sel = LayerSel(name, name, "linear", (n, k), [n, k], m.bias is not None)
        elif scope == "linear+pw" and _is_pointwise(m, parents.get(name)):
            n, k = (int(x) for x in m.weight.shape[:2])
            sel = LayerSel(f"{name}.linear", name, "pointwise_conv1d", (n, k), [n, k, 1], m.bias is not None)
        else:
            if any(True for _ in m.parameters(recurse=False)):
                kept[name] = _kept_reason(name, m)
            continue
        if why := align_problem(wfmt, *sel.shape):
            skipped[sel.name] = why
            continue
        selected.append(sel)
    return selected, kept, skipped


# ---------------------------------------------------------------------------------------------- apply


def _model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def model_family(model: nn.Module) -> str:
    """"ctc" (ParakeetForCTC) or "aed" (a Cohere ASR model); anything else (Whisper included: never quantised)
    raises."""
    name = type(model).__name__
    if name == "ParakeetForCTC":
        return "ctc"
    if name.startswith("CohereAsr"):
        return "aed"
    raise QuantError(f"{name}: only the students (ParakeetForCTC, CohereAsrForConditionalGeneration) are quantised")


def _bn_snapshot(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Every BatchNorm's running statistics as they are (their dtype included): apply and load_quantized take it
    before they touch the model, assert_quantized compares against it."""
    return {n: {k: t.detach().clone() for k in ("running_mean", "running_var")
                if (t := getattr(m, k, None)) is not None}
            for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}


def _unique_params(model: nn.Module) -> int:
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            n += p.numel()
    return n


def _swap_in(model: nn.Module, name: str, new: nn.Module):
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, attr, new)


def _cast_fp16(model: nn.Module) -> list[str]:
    """Linear / Conv1d / Conv2d / Embedding weights and biases to fp16 in place (p.data: the Parameters and their
    ties stay); norms and BatchNorm stay fp32 (fp16 LayerNorm or BN affine params fail on CPU). Returns the state_dict
    keys now fp16, a tied tensor under every name."""
    kinds = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)
    done = set()
    for m in model.modules():
        if isinstance(m, kinds):
            for p in (m.weight, getattr(m, "bias", None)):
                if p is not None and id(p) not in done and p.is_floating_point():
                    p.data = p.data.half()
                    done.add(id(p))
    keys = []
    for n, p in model.named_parameters(remove_duplicate=False):
        if id(p) in done:
            keys.append(n)
    return sorted(keys)


def _install(module: nn.Linear, pack: QuantPack, fmt: str, impl: str, sel_name: str, kind: str, *,
             keep_pack: bool = False, cache: dict | None = None):
    """Turn a Linear into the QuantLinear of `pack` (in place)."""
    wfmt, act = _SPLIT[fmt]
    dev = module.weight.device
    pack = pack.to(dev)
    if impl == "emulate":
        w = unpack_weight(pack) if act else _weight_only_dequant(pack)
    elif impl == "torchao":
        w = to_torchao(pack, fmt, dev, cache=cache)
    else:
        raise QuantError(f"{fmt}: impl {impl!r} cannot install a packed weight")
    module.__class__ = QuantLinear
    module.weight = nn.Parameter(w, requires_grad=False)
    min_rows = PAD_ROWS if (impl == "torchao" and act == "int8" and dev.type == "cuda") else 0
    module.kq = QuantState(fmt=fmt, impl=impl, wfmt=wfmt, act=act, block=FORMAT_INFO[fmt]["block"], kind=kind,
                           name=sel_name, min_rows=min_rows, mx_rounding=pack.mx_rounding or "rceil")
    if keep_pack:
        module.kpack = pack


def _recipe_consts(fmt: str, mx_rounding: str) -> dict:
    wfmt, act = _SPLIT.get(fmt, ("fp16", None))
    base = dict(bf16_cast_before_quant=fmt != "fp16", bias="outside_gemm" if fmt != "fp16" else "fp16",
                e2m1_rounding="rne" if wfmt in ("nvfp4", "mxfp4") else None)
    if fmt == "fp16":
        return dict(base, cast="Linear/Conv/Embedding weights and biases to fp16; norms and BatchNorm fp32",
                    autocast="float16")
    rec = dict(base, block=FORMAT_INFO[fmt]["block"])
    if wfmt == "int8":
        rec.update(int8_div=INT8_DIV, int8_eps=INT8_EPS, int8_range=[-128, 127],
                   int8_scale_dtype=str(INT8_SCALE_DTYPE).replace("torch.", ""))
    if wfmt == "fp8":
        rec.update(fp8_scale="amax/448 per row (0 -> 1)")
    if wfmt == "nvfp4":
        rec.update(tensor_scale="amax/(448*6)", block_scale_min=NVFP4_SCALE_MIN)
    if wfmt == "mxfp4":
        rec.update(mx_rounding=mx_rounding, e8m0="uint8 = e + 127")
    if act == "int8":
        rec.update(act_int8_div=INT8_ACT_DIV, act_qmin=INT8_ACT_QMIN, act_eps=INT8_ACT_EPS, pad_rows_min=PAD_ROWS,
                   act_scale_dtype=str(INT8_ACT_SCALE_DTYPE).replace("torch.", ""))
    if act == "nvfp4":
        rec.update(act_tensor_scale="whole call input")
    if act in ("fp8", "mxfp4"):
        rec.update(act_block="per token" if act == "fp8" else MXFP4_BLOCK)
    return rec


def _versions() -> dict:
    out = dict(torch=torch.__version__, torchao=torchao_version(), transformers=None, safetensors=None, code_sha=None)
    for k in ("transformers", "safetensors"):
        try:
            out[k] = __import__(k).__version__
        except Exception:  # noqa: BLE001
            pass
    try:
        import subprocess

        out["code_sha"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                         timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return out


def apply(model: nn.Module, fmt: str, *, impl: str = "auto", scope: str = "linear+pw", mx_rounding: str = "rceil",
          keep_packs: bool = False) -> dict:
    """Quantise a loaded student in place (on its device, after any dtype cast; never model.to(dtype=...) it
    afterwards): fp16 casts; every other format packs each selected layer (pack_weight) and installs it for the
    resolved impl (emulate, or torchao's subclass built from the pack). Returns the recipe record (the variant file's
    schema less its file fields, plus "impl"); ends with assert_quantized."""
    check_format(fmt)
    if scope not in SCOPES:
        raise QuantRefused(f"--quant-scope must be one of {SCOPES}, got {scope!r}")
    if mx_rounding not in MX_ROUNDINGS:
        raise QuantRefused(f"--mx-rounding must be one of {MX_ROUNDINGS}, got {mx_rounding!r}")
    if getattr(model, "_kitsune_quant", None) is not None:
        raise QuantError("this model is quantised already: apply a format to a model once")
    family = model_family(model)
    dev = _model_device(model)
    impl_r = resolve_impl(fmt, impl, dev)
    t0 = time.time()
    bn = _bn_snapshot(model)
    params_total = _unique_params(model)
    layers, kept, skipped, fp16 = {}, {}, {}, []
    if fmt == "fp16":
        fp16 = _cast_fp16(model)
        params_q = sum(p.numel() for n, p in model.named_parameters() if n in set(fp16))
    else:
        wfmt, _ = _SPLIT[fmt]
        sel, kept, skipped = select_layers(model, fmt, scope)
        cache: dict = {}
        params_q = 0
        for s in sel:
            if s.kind == "pointwise_conv1d":
                _swap_in(model, s.source, PointwiseConv1dAsLinear(model.get_submodule(s.source)))
            mod = model.get_submodule(s.name)
            pack = pack_weight(mod.weight, wfmt, mx_rounding=mx_rounding)
            _install(mod, pack, fmt, impl_r, s.name, s.kind, keep_pack=keep_packs, cache=cache)
            layers[s.name] = dict(kind=s.kind, shape=list(s.shape), orig_shape=list(s.orig_shape), bias=s.bias)
            params_q += s.shape[0] * s.shape[1]
    tied = {}
    if family == "aed" and model.config.tie_word_embeddings:
        tied["proj_out.weight"] = "model.decoder.embed_tokens.weight"
    wfmt = _SPLIT[fmt][0] if fmt != "fp16" else "fp16"
    recipe = dict(
        schema=SCHEMA, format=fmt, weights=wfmt, activations=ACTIVATIONS[fmt], scope=scope if fmt != "fp16" else None,
        recipe=_recipe_consts(fmt, mx_rounding), family=family, architecture=type(model).__name__,
        layers=layers, kept=kept, skipped=skipped, tied=tied, fp16_tensors=fp16,
        counts=dict(params_total=params_total, params_quantized=params_q,
                    share_quantized=params_q / params_total if params_total else 0.0, layers=len(layers),
                    linear=sum(v["kind"] == "linear" for v in layers.values()),
                    pointwise=sum(v["kind"] == "pointwise_conv1d" for v in layers.values())),
        impl=impl_r, versions=_versions(), load_with=LOAD_WITH, wall_s=None, time_utc=_now())
    model._kitsune_quant = dict(recipe=recipe, bn=bn)
    recipe["bytes"] = {k: v for k, v in weight_bytes(model).items() if k != "resident"}
    assert_quantized(model, recipe)
    recipe["wall_s"] = round(time.time() - t0, 3)
    return recipe


def _probe_relpos(m: QuantLinear) -> str | None:
    """Run a relative_k_proj on a stride-0 (2, 3, K) input: its quantised forward must be reached exactly once (the
    rel-pos patch routes through type(self).forward). The counters are restored."""
    kq = m.kq
    saved = (kq.calls, kq.rows, kq.padded, kq.fallback_risk)
    dtype = m.weight.dtype if kq.impl == "torchao" else torch.float32
    x = torch.zeros(1, 3, m.in_features, device=m.weight.device, dtype=dtype).expand(2, 3, m.in_features)
    try:
        with torch.no_grad():
            m(x)
        got = kq.calls - saved[0]
    finally:
        kq.calls, kq.rows, kq.padded, kq.fallback_risk = saved
    return None if (got == 1 or not kq.count) else f"its quantised forward ran {got} times on one call"


def assert_quantized(model: nn.Module, recipe: dict) -> None:
    """Raise QuantError unless the model is what the recipe says: every recipe layer a QuantLinear of the format (a
    torchao weight a tensor subclass, an emulated one a plain fp32 Parameter), every relative_k_proj quantised or
    skipped and reaching its quantised forward, the heads plain tensors and the tie intact, the BatchNorm statistics
    bitwise unchanged in the dtype they had when apply / load_quantized snapshotted them (fp32 from a master or a
    variant dir; bf16 when speed_probe's runner cast the whole model before quantising, the study's timing
    convention: apply never changes them either way), no in-scope Linear left behind (fp16: the fp16 tensors fp16,
    norms and BN fp32)."""
    fmt = recipe["format"]
    problems = []
    state = getattr(model, "_kitsune_quant", None) or {}
    for name, snap in (state.get("bn") or {}).items():
        m = model.get_submodule(name)
        for k, want in snap.items():
            got = getattr(m, k, None)
            if got is None or got.dtype != want.dtype or not torch.equal(got, want):
                problems.append(f"{name}.{k}: the BatchNorm statistic changed (" + (
                    "gone" if got is None else f"{got.dtype}, was {want.dtype}" if got.dtype != want.dtype
                    else "other values") + ")")
    if (recipe.get("tied") or {}).get("proj_out.weight") and model.proj_out.weight is not \
            model.model.decoder.embed_tokens.weight:
        problems.append("proj_out.weight is no longer tied to model.decoder.embed_tokens.weight")
    for name in ("ctc_head", "proj_out"):
        m = getattr(model, name, None)
        if m is not None and (isinstance(m, QuantLinear) or type(m.weight) not in (torch.Tensor, nn.Parameter)):
            problems.append(f"{name}: the head must stay a plain tensor")
    if fmt == "fp16":
        params = dict(model.named_parameters(remove_duplicate=False))
        for k in recipe.get("fp16_tensors") or []:
            if k in params and params[k].dtype != torch.float16:
                problems.append(f"{k} is {params[k].dtype}, not fp16")
        for n, m in model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
                if any(p.dtype != torch.float32 for p in m.parameters(recurse=False)):
                    problems.append(f"{n}: norms stay fp32 in the fp16 format")
    else:
        layers = recipe.get("layers") or {}
        skipped = recipe.get("skipped") or {}
        for name in layers:
            try:
                m = model.get_submodule(name)
            except AttributeError:
                problems.append(f"{name}: not in the model")
                continue
            if not isinstance(m, QuantLinear) or m.kq.fmt != fmt:
                problems.append(f"{name}: not a QuantLinear of {fmt}")
                continue
            plain = type(m.weight) in (torch.Tensor, nn.Parameter)
            if m.kq.impl == "torchao" and plain:
                problems.append(f"{name}: impl torchao but its weight is a plain tensor")
            if m.kq.impl == "emulate" and not (plain and m.weight.dtype == torch.float32):
                problems.append(f"{name}: an emulated weight must be a plain fp32 tensor")
        for n, m in model.named_modules():
            if type(m).__name__ == "ParakeetEncoderAttention":
                rk = f"{n}.relative_k_proj"
                if rk in skipped:
                    continue
                if rk not in layers:
                    problems.append(f"{rk}: not quantised")
                elif isinstance(m.relative_k_proj, QuantLinear) and (why := _probe_relpos(m.relative_k_proj)):
                    problems.append(f"{rk}: {why}")
        kept = recipe.get("kept") or {}
        scope = recipe.get("scope") or "linear+pw"
        parents = _parents(model)
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and not isinstance(m, QuantLinear) and n not in kept and n not in skipped:
                problems.append(f"{n}: an in-scope Linear left unquantised")
            if scope == "linear+pw" and _is_pointwise(m, parents.get(n)) and f"{n}.linear" not in skipped:
                problems.append(f"{n}: a pointwise conv left unquantised")
    if problems:
        raise QuantError(f"{fmt}: the model is not quantised as recorded: " + "; ".join(problems[:20])
                         + (f" (+{len(problems) - 20} more)" if len(problems) > 20 else ""))


def quant_layers(model: nn.Module) -> dict[str, QuantLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, QuantLinear)}


def counters(model: nn.Module) -> dict:
    """The QuantLinear counters summed: calls, rows, padded (calls padded to min_rows), fallback_risk."""
    out = dict(calls=0, rows=0, padded=0, fallback_risk=0)
    for m in quant_layers(model).values():
        for k in out:
            out[k] += int(getattr(m.kq, k))
    return out


def uncalled(model: nn.Module) -> list[str]:
    """The quantised layers never called since apply (an eval that ran must have called every one)."""
    return sorted(n for n, m in quant_layers(model).items() if m.kq.calls == 0)


def set_counting(model: nn.Module, on: bool) -> None:
    """Counters on or off (off before torch.compile: a counter increment is a Python side effect in the graph)."""
    for m in quant_layers(model).values():
        m.kq.count = bool(on)


def _inner_nbytes(t: torch.Tensor) -> int:
    if type(t) not in (torch.Tensor, nn.Parameter) and hasattr(t, "__tensor_flatten__"):
        names, _ = t.__tensor_flatten__()
        return sum(_inner_nbytes(getattr(t, n)) for n in names if getattr(t, n, None) is not None)
    return int(t.numel() * t.element_size())


def packed_bytes(wfmt: str, n: int, k: int) -> int:
    """The bytes the exporter writes for one quantised (N, K) weight (its bias not included)."""
    if wfmt in ("int8", "fp8"):
        return n * k + 4 * n
    if wfmt == "nvfp4":
        return n * k // 2 + n * k // NVFP4_BLOCK + 4
    if wfmt == "mxfp4":
        return n * k // 2 + n * k // MXFP4_BLOCK
    raise QuantError(f"unknown weight format {wfmt!r}")


def weight_bytes(model: nn.Module, *, source_dtypes: dict | None = None) -> dict:
    """{deployable, quantized, kept, resident} in bytes. deployable = quantized + kept = what the exporter writes:
    the packed weights (packed_bytes) plus every other tensor once (a tie once) in its source dtype - the
    source_dtypes given (state_dict key -> torch.dtype), else kitsune's export convention: bf16, BatchNorm statistics
    fp32, fp16 tensors fp16, integer buffers as they are. resident = what the model holds in memory now (a torchao
    weight's inner tensors; an emulated weight's fp32 dequantisation)."""
    qmods = quant_layers(model)
    qweights = {f"{n}.weight" for n in qmods}
    bn_stats = {f"{n}.{b}" for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
                for b in ("running_mean", "running_var")}
    quantized = sum(packed_bytes(m.kq.wfmt, m.out_features, m.in_features) for m in qmods.values())
    kept = resident = 0
    seen = set()
    for key, t in model.state_dict(keep_vars=True).items():
        if id(t) in seen:
            continue
        seen.add(id(t))
        resident += _inner_nbytes(t)
        if key in qweights:
            continue
        if source_dtypes and key in source_dtypes:
            size = torch.empty((), dtype=source_dtypes[key]).element_size()
        elif not t.is_floating_point():
            size = t.element_size()
        elif key in bn_stats:
            size = 4
        elif t.dtype == torch.float16:
            size = 2
        else:
            size = 2
        kept += t.numel() * size
    return dict(deployable=int(quantized + kept), quantized=int(quantized), kept=int(kept), resident=int(resident))


# ---------------------------------------------------------------------------------------------- torchao adapter


def _ao(name: str):
    """A torchao attribute from the first of _AO_MODULES that has it (QuantError when none does)."""
    import importlib

    errs = []
    for mod in _AO_MODULES:
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}: {type(e).__name__}")
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise QuantError(f"torchao {torchao_version()} has no {name} in {_AO_MODULES} ({'; '.join(errs[:3])}): "
                     "kitsune/quant.py's torchao adapter needs updating")


def torchao_config(fmt: str):
    """torchao 0.18's config object of a format (TORCHAO_CONFIGS; section 2 of scout s7)."""
    if fmt == "int8-w8a16":
        return _ao("Int8WeightOnlyConfig")()
    if fmt == "int8-w8a8":
        return _ao("Int8DynamicActivationInt8WeightConfig")()
    if fmt == "nvfp4-w4a16":
        return _ao("NVFP4WeightOnlyConfig")()
    if fmt == "nvfp4-w4a4":
        return _ao("NVFP4DynamicActivationNVFP4WeightConfig")(use_dynamic_per_tensor_scale=True,
                                                               use_triton_kernel=False)
    if fmt == "fp8-w8a8":
        return _ao("Float8DynamicActivationFloat8WeightConfig")(granularity=_ao("PerRow")())
    raise QuantError(f"{fmt}: no torchao config (mxfp4 is emulated, fp16 is plain fp16)")


def _is_subclass(t) -> bool:
    return isinstance(t, torch.Tensor) and type(t) not in (torch.Tensor, nn.Parameter) and \
        hasattr(type(t), "__tensor_flatten__")


def _leaves(t: torch.Tensor, prefix: str = "") -> dict[str, torch.Tensor]:
    """A tensor subclass's plain inner tensors by path (the subclass protocol's __tensor_flatten__, recursively)."""
    if not _is_subclass(t):
        return {prefix: t}
    names, _ = t.__tensor_flatten__()
    out = {}
    for n in names:
        inner = getattr(t, n, None)
        if inner is not None:
            out.update(_leaves(inner, f"{prefix}.{n}" if prefix else n))
    return out


def _rebuild(t: torch.Tensor, new: dict[str, torch.Tensor], prefix: str = "") -> torch.Tensor:
    if not _is_subclass(t):
        return new.get(prefix, t)
    names, ctx = t.__tensor_flatten__()
    inner = {n: _rebuild(getattr(t, n), new, f"{prefix}.{n}" if prefix else n) for n in names
             if getattr(t, n, None) is not None}
    return type(t).__tensor_unflatten__(inner, ctx, t.size(), t.stride())


def _swizzled(t) -> bool:
    return bool(getattr(t, "is_swizzled_scales", False) or getattr(t, "_is_swizzled_scales", False))


def _roles(template: torch.Tensor, n: int, k: int, wfmt: str) -> dict[str, str]:
    """Inner tensor path -> role (qweight, scale, block_scale, tensor_scale, zero_point) of a torchao weight, by
    dtype and shape. An inner tensor this cannot place raises: the adapter must be updated for that layout."""
    roles = {}
    for path, t in _leaves(template).items():
        dt, numel = t.dtype, t.numel()
        if wfmt == "int8" and dt == torch.int8 and numel == n * k:
            roles[path] = "qweight"
        elif wfmt == "fp8" and dt == torch.float8_e4m3fn and numel == n * k:
            roles[path] = "qweight"
        elif wfmt == "nvfp4" and numel == n * k // 2 and t.element_size() == 1 and dt != torch.float8_e4m3fn:
            roles[path] = "qweight"
        elif wfmt in ("int8", "fp8") and t.is_floating_point() and numel == n:
            roles[path] = "scale"
        elif wfmt == "int8" and not t.is_floating_point() and numel == n:
            roles[path] = "zero_point"
        elif wfmt == "nvfp4" and dt == torch.float8_e4m3fn:
            roles[path] = "block_scale"
        elif wfmt == "nvfp4" and t.is_floating_point() and numel == 1:
            roles[path] = "tensor_scale"
        else:
            raise QuantError(f"torchao {torchao_version()} {type(template).__name__}: cannot place its inner tensor "
                             f"{path} ({dt}, {tuple(t.shape)}) of a {wfmt} ({n}, {k}) weight: the adapter needs "
                             "updating")
    need = {"qweight"} | ({"scale"} if wfmt in ("int8", "fp8") else {"block_scale"})
    if missing := need - set(roles.values()):
        raise QuantError(f"torchao {type(template).__name__}: no inner tensor for {sorted(missing)}")
    return roles


def _exact(src: torch.Tensor, dtype: torch.dtype, what: str) -> torch.Tensor:
    """src in dtype, refusing a lossy conversion (a torchao scale dtype the pack does not hold exactly)."""
    out = src.to(dtype)
    if src.is_floating_point() and not torch.equal(out.to(src.dtype), src):
        raise QuantError(f"{what}: torchao keeps it as {dtype}, which does not hold the pack's {src.dtype} values "
                         f"exactly: the recipe must produce {dtype} scales (tests/test_quant_torchao.py)")
    return out


def _template(n: int, k: int, fmt: str, device) -> torch.Tensor:
    """torchao's quantised weight of an (N, K) bf16 Linear under the format's config (its layout and attributes;
    its values are replaced from the pack)."""
    quantize_ = _ao("quantize_")
    lin = nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.normal_(0.0, 0.02)
    quantize_(lin, torchao_config(fmt))
    if not _is_subclass(lin.weight):
        raise QuantError(f"torchao quantize_ left {fmt}'s weight a plain tensor on {device}")
    return lin.weight


def to_torchao(pack: QuantPack, fmt: str, device, *, cache: dict | None = None) -> torch.Tensor:
    """torchao's tensor subclass of a pack: a template of the format's layout (quantize_ of a bf16 Linear of the
    same shape, cached per shape) with its inner tensors replaced by the pack's (the subclass protocol's
    __tensor_flatten__ / __tensor_unflatten__), so the torchao kernels run on exactly the pack's values. Scales are
    converted to torchao's dtype only when that is exact; swizzled NVFP4 block scales are swizzled with torchao's
    to_blocked. Raises QuantError on any layout it cannot place."""
    n, k = pack.shape
    dev = torch.device(device)
    key = (fmt, n, k, str(dev))
    cache = {} if cache is None else cache
    if key not in cache:
        cache[key] = _template(n, k, fmt, dev)
    tpl = cache[key]
    roles = _roles(tpl, n, k, pack.wfmt)
    leaves = _leaves(tpl)
    new = {}
    for path, role in roles.items():
        t = leaves[path]
        if role == "qweight":
            src = pack.qweight.to(dev)
            new[path] = (src.view(t.dtype) if src.element_size() == t.element_size() and t.dtype != src.dtype
                         else src.to(t.dtype)).reshape(t.shape).contiguous()
        elif role == "scale":
            new[path] = _exact(pack.scale.to(dev), t.dtype, "the per-row scale").reshape(t.shape).contiguous()
        elif role == "zero_point":
            if bool(t.any()):
                raise QuantError("torchao's int8 template has a non-zero zero point: not the symmetric recipe")
            new[path] = t
        elif role == "block_scale":
            bs = pack.block_scale.to(dev)
            if tuple(t.shape) != tuple(bs.shape) or _swizzled(tpl):
                bs = _ao("to_blocked")(bs)
                if bs.numel() != t.numel():
                    raise QuantError(f"swizzled NVFP4 block scales {tuple(bs.shape)} do not fit torchao's "
                                     f"{tuple(t.shape)}")
            new[path] = bs.to(t.dtype).reshape(t.shape).contiguous()
        elif role == "tensor_scale":
            new[path] = _exact(pack.tensor_scale.to(dev), t.dtype, "the NVFP4 tensor scale").reshape(t.shape)
    return _rebuild(tpl, new)


def from_torchao(t: torch.Tensor, fmt: str) -> QuantPack:
    """A torchao quantised weight read back as a QuantPack (the parity tests: torchao's own recipe against
    pack_weight). Swizzled NVFP4 block scales are refused (read them before swizzling)."""
    wfmt, _ = _SPLIT[fmt]
    n, k = (int(x) for x in t.shape)
    roles = _roles(t, n, k, wfmt)
    leaves = _leaves(t)
    by = {role: leaves[p] for p, role in roles.items()}
    q = by["qweight"]
    if wfmt == "nvfp4":
        if _swizzled(t):
            raise QuantError("from_torchao cannot read swizzled NVFP4 block scales")
        qw = (q.view(torch.uint8) if q.element_size() == 1 and q.dtype != torch.uint8 else q).reshape(n, k // 2)
        return QuantPack("nvfp4", (n, k), qw.cpu(), block_scale=by["block_scale"].reshape(n, k // 16).cpu(),
                         tensor_scale=by["tensor_scale"].float().reshape(()).cpu())
    return QuantPack(wfmt, (n, k), q.reshape(n, k).cpu(), scale=by["scale"].float().reshape(n).cpu())


# ---------------------------------------------------------------------------------------------- export / load


def is_quantized_dir(path) -> bool:
    return (Path(path) / QUANT_FILE).is_file()


def read_recipe(path) -> dict:
    p = Path(path) / QUANT_FILE
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QuantError(f"{p}: not a readable quantisation recipe ({e})") from e
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA or rec.get("format") not in QUANT_FORMATS:
        raise QuantError(f"{p}: not a schema-{SCHEMA} recipe of a known format")
    return rec


def _family_of_dir(path: Path) -> tuple[str, str]:
    """(family, architecture) of a weights dir from its config.json."""
    try:
        c = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QuantRefused(f"{path}: no readable config.json ({e})") from e
    arch = " ".join(c.get("architectures") or []) + " " + str(c.get("model_type") or "")
    if "ParakeetForCTC" in arch or "parakeet_ctc" in arch:
        return "ctc", "ParakeetForCTC"
    if "Cohere" in arch or "cohere" in arch:
        return "aed", "CohereAsrForConditionalGeneration"
    raise QuantRefused(f"{path}: {arch.strip()!r} is not a student (ParakeetForCTC / CohereAsr): not quantised")


def _weight_files(src: Path) -> list[Path]:
    idx = src / "model.safetensors.index.json"
    if idx.is_file():
        wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map") or {}
        return [src / f for f in sorted(set(wm.values()))]
    if (src / WEIGHTS_FILE).is_file():
        return [src / WEIGHTS_FILE]
    raise QuantRefused(f"{src}: no {WEIGHTS_FILE} (a bf16 export of the trainer: checkpoints/step_<N>)")


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
              "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
              "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn}
_TORCH_ST = {v: k for k, v in _ST_DTYPES.items()}


def _source_dtypes(files: list[Path]) -> dict[str, torch.dtype]:
    from safetensors import safe_open

    out = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as fh:
            for k in fh.keys():
                out[k] = _ST_DTYPES[fh.get_slice(k).get_dtype()]
    return out


def save_safetensors(tensors: dict[str, torch.Tensor], path, metadata: dict[str, str]) -> None:
    """A safetensors file with deterministic bytes: safetensors' own writer orders the header's metadata through a
    hash map, so two exports of one weight differed in their first bytes. The format as safe_open reads it: the
    header size (8 bytes, little-endian), the JSON header (metadata and tensors in sorted key order) padded with
    spaces to a multiple of 8, then every tensor's little-endian bytes in that order, contiguous."""
    if sys.byteorder != "little":
        raise QuantError("save_safetensors writes the host's byte order, which must be little-endian")
    header, off, order = {}, 0, sorted(tensors)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in sorted(metadata.items())}
    for k in order:
        t = tensors[k]
        if t.dtype not in _TORCH_ST:
            raise QuantError(f"{k}: {t.dtype} has no safetensors dtype here")
        n = t.numel() * t.element_size()
        header[k] = {"dtype": _TORCH_ST[t.dtype], "shape": [int(x) for x in t.shape], "data_offsets": [off, off + n]}
        off += n
    head = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as fh:
        fh.write(len(head).to_bytes(8, "little"))
        fh.write(head)
        for k in order:
            t = tensors[k].detach().to("cpu").contiguous()
            fh.write(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        fh.flush()
        os.fsync(fh.fileno())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def weights_sha256(files: list[Path]) -> str:
    """The source weights' identity: the sha256 of the one weights file, or of the shards' sha256 hex digests."""
    if len(files) == 1:
        return _sha256(files[0])
    return hashlib.sha256("".join(_sha256(f) for f in files).encode()).hexdigest()


def _tensor_index(header: dict[str, tuple[str, list]]) -> str:
    """A digest of a safetensors file's (key, dtype, shape) list, sorted by key."""
    s = "\n".join(f"{k}:{d}:{list(sh)}" for k, (d, sh) in sorted(header.items()))
    return hashlib.sha256(s.encode()).hexdigest()


def _header(path: Path) -> tuple[dict[str, tuple[str, list]], dict]:
    """(key -> (dtype, shape), metadata) of a safetensors file, from its JSON header."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        head = json.loads(fh.read(n))
    meta = head.pop("__metadata__", {}) or {}
    return {k: (v["dtype"], list(v["shape"])) for k, v in head.items()}, meta


def _load_student(src: Path, family: str, device):
    if family == "ctc":
        from kitsune import ctc_student as CS

        return CS.load_ctc_student(src, device, dtype=torch.float32)
    from kitsune import student as S

    return S.load_student(src, device, dtype=torch.float32)


def _write_bytes_json(path: Path, obj: dict):
    path.write_bytes((json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))  # LF on every OS


def _source_key(key: str, layers: dict) -> str:
    """The source file's key of a tensor of the adapted model (<conv>.linear.bias was <conv>.bias)."""
    for name, info in layers.items():
        if info["kind"] == "pointwise_conv1d" and key.startswith(name + "."):
            return name[:-len(".linear")] + key[len(name):]
    return key


def export(src_dir, out_dir, fmt: str, *, scope: str = "linear+pw", mx_rounding: str = "rceil", device="cpu",
           force: bool = False) -> dict:
    """A variant dir (module docstring) of a bf16 student export: written to <out>.tmp, renamed into place, then
    verify_export must pass. Refuses (QuantRefused) an existing --out without force and an unknown format. Beats the
    item's heartbeat while it runs."""
    from kitsune import heartbeat

    check_format(fmt)
    src, out = Path(src_dir).resolve(), Path(out_dir).resolve()
    if out.exists() and not force:
        raise QuantRefused(f"{out} exists: use another --out, or --force to replace it")
    if is_quantized_dir(src):
        raise QuantRefused(f"{src} is a quantised variant already: export from the bf16 checkpoint")
    family, arch = _family_of_dir(src)
    files = _weight_files(src)
    with heartbeat.beating(max_s=BEAT_MAX_S):
        return _export(src, out, fmt, family, arch, files, scope, mx_rounding, device, force)


def _export(src: Path, out: Path, fmt, family, arch, files, scope, mx_rounding, device, force) -> dict:
    t0 = time.time()
    src_dtypes = _source_dtypes(files)
    model = _load_student(src, family, torch.device(device))
    recipe = apply(model, fmt, impl="emulate" if fmt != "fp16" else "auto", scope=scope, mx_rounding=mx_rounding,
                   keep_packs=True)
    layers, tied = recipe["layers"], dict(recipe["tied"])
    qweights = {f"{n}.weight" for n in layers}
    tensors: dict[str, torch.Tensor] = {}
    seen: dict[int, str] = {}
    fp16_set = set(recipe["fp16_tensors"])
    for key, t in model.state_dict(keep_vars=True).items():
        if key in qweights:
            continue
        ptr = id(t)
        if ptr in seen or key in tied:
            tied.setdefault(key, seen.get(ptr) or tied[key])
            continue
        seen[ptr] = key
        skey = _source_key(key, layers)
        if skey not in src_dtypes:
            raise QuantError(f"{key}: no source tensor {skey} in {src}")
        dt = torch.float16 if key in fp16_set else src_dtypes[skey]
        tensors[key] = t.detach().to("cpu", dt).contiguous()
    for key in list(tied):
        if tied[key] not in tensors:
            raise QuantError(f"tied {key} -> {tied[key]}: the target is not written")
    for name in layers:
        pack = model.get_submodule(name).kpack
        for part, t in pack.parts().items():
            tensors[f"{name}.{part}"] = t.detach().cpu().contiguous()
    wfmt = recipe["weights"]
    meta = {"format": "pt", "kitsune_quant": wfmt, "kitsune_quant_schema": str(SCHEMA)}
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_safetensors(tensors, tmp / WEIGHTS_FILE, meta)
    copied = {}
    for f in sorted(src.iterdir()):
        if f.is_file() and not f.name.endswith(_NOT_COPIED):
            shutil.copyfile(f, tmp / f.name)
            copied[f.name] = dict(size=f.stat().st_size, sha256=_sha256(f))
    header, _ = _header(tmp / WEIGHTS_FILE)
    qbytes = sum(t.numel() * t.element_size() for k, t in tensors.items()
                 if k.rpartition(".")[2] in ("qweight", "qscale", "qblock_scale", "qtensor_scale")
                 and k.rpartition(".")[0] in layers)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    recipe.pop("impl", None)
    recipe.pop("wall_s", None)
    wbytes = (tmp / WEIGHTS_FILE).stat().st_size
    recipe.update(
        tied=tied,
        source=dict(dir=str(src), weights_sha256=weights_sha256(files), weights_files=[f.name for f in files],
                    family=family, architecture=arch),
        bytes=dict(deployable=int(total), quantized=int(qbytes), kept=int(total - qbytes)),
        # dir_total: the weights and the copied files (the recipe file itself is not counted)
        file_bytes={WEIGHTS_FILE: wbytes, "dir_total": int(wbytes + sum(c["size"] for c in copied.values()))},
        tensors=dict(n=len(header), index_sha256=_tensor_index(header)),
        copied=copied, versions=_versions(), load_with=LOAD_WITH, export_wall_s=round(time.time() - t0, 3),
        time_utc=_now())
    _write_bytes_json(tmp / QUANT_FILE, recipe)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    if problems := verify_export(out):
        raise QuantError(f"{out}: the export does not verify: " + "; ".join(problems[:10]))
    return recipe


def expected_variant_files(path) -> list[str]:
    """The files a variant dir must hold: the weights, the recipe and every copied source file."""
    rec = read_recipe(path)
    return [WEIGHTS_FILE, QUANT_FILE, *sorted(rec.get("copied") or {})]


def _expected_parts(wfmt: str, n: int, k: int) -> dict[str, tuple[str, list]]:
    if wfmt == "int8":
        return dict(qweight=("I8", [n, k]), qscale=("F32", [n]))
    if wfmt == "fp8":
        return dict(qweight=("F8_E4M3", [n, k]), qscale=("F32", [n]))
    if wfmt == "nvfp4":
        return dict(qweight=("U8", [n, k // 2]), qblock_scale=("F8_E4M3", [n, k // NVFP4_BLOCK]),
                    qtensor_scale=("F32", []))
    return dict(qweight=("U8", [n, k // 2]), qblock_scale=("U8", [n, k // MXFP4_BLOCK]))


def verify_export(path) -> list[str]:
    """What is wrong with a variant dir ([] = nothing): the recipe, the weights file's size, header metadata, tensor
    index and every quantised layer's tensors (names, dtypes, shapes), the fp16 tensors, the copied files (size and
    sha256)."""
    path = Path(path)
    try:
        rec = read_recipe(path)
    except QuantError as e:
        return [str(e)]
    problems = []
    w = path / WEIGHTS_FILE
    if not w.is_file():
        return problems + [f"{WEIGHTS_FILE} is missing"]
    fb = rec.get("file_bytes") or {}
    if fb.get(WEIGHTS_FILE) != w.stat().st_size:
        problems.append(f"{WEIGHTS_FILE} is {w.stat().st_size} bytes, the recipe says {fb.get(WEIGHTS_FILE)}")
    try:
        header, meta = _header(w)
    except Exception as e:  # noqa: BLE001 - a torn or foreign file
        return problems + [f"{WEIGHTS_FILE}: unreadable header ({type(e).__name__}: {e})"]
    fmt = rec["format"]
    wfmt = "fp16" if fmt == "fp16" else _SPLIT[fmt][0]
    if meta.get("kitsune_quant") != wfmt or meta.get("kitsune_quant_schema") != str(SCHEMA):
        problems.append(f"{WEIGHTS_FILE} metadata {meta} is not a schema-{SCHEMA} {wfmt} variant's")
    tix = rec.get("tensors") or {}
    if tix.get("n") != len(header) or tix.get("index_sha256") != _tensor_index(header):
        problems.append(f"{WEIGHTS_FILE}: its tensors ({len(header)}) are not the ones the recipe indexed "
                        f"({tix.get('n')})")
    for name, info in (rec.get("layers") or {}).items():
        n, k = info["shape"]
        for part, (dt, shape) in _expected_parts(wfmt, n, k).items():
            got = header.get(f"{name}.{part}")
            if got != (dt, shape):
                problems.append(f"{name}.{part}: {got} in the file, ({dt}, {shape}) expected")
        if f"{name}.weight" in header:
            problems.append(f"{name}.weight: a quantised layer's 16-bit weight is in the file")
        if bool(info.get("bias")) != (f"{name}.bias" in header):
            problems.append(f"{name}.bias: {'missing' if info.get('bias') else 'unexpected'}")
    tied = rec.get("tied") or {}
    for alias, target in tied.items():
        if alias in header or target not in header:
            problems.append(f"tied {alias} -> {target}: the file must hold the target only")
    for key in rec.get("fp16_tensors") or []:
        if key not in tied and (header.get(key) or ("?",))[0] != "F16":
            problems.append(f"{key}: not F16 in the fp16 variant")
    for name, c in (rec.get("copied") or {}).items():
        f = path / name
        if not f.is_file():
            problems.append(f"{name}: a copied source file is missing")
        elif f.stat().st_size != c.get("size") or _sha256(f) != c.get("sha256"):
            problems.append(f"{name}: not the source's bytes")
    return problems


def _skeleton(path: Path, family: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(path), local_files_only=True)
    if family == "ctc":
        from transformers import ParakeetForCTC as cls
    else:
        from transformers import CohereAsrForConditionalGeneration as cls
    model = cls._from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
    subs = [model.config] + [getattr(model.config, s) for s in getattr(model.config, "sub_configs", {}) or {}
                             if getattr(model.config, s, None) is not None]
    for c in subs:
        if getattr(c, "_attn_implementation", None) != "sdpa":
            c._attn_implementation = "sdpa"
    return model


def _set_tensor(model: nn.Module, key: str, t: torch.Tensor):
    mod_name, _, attr = key.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    if attr in mod._parameters and mod._parameters[attr] is not None:
        mod._parameters[attr].data = t
    elif attr in mod._buffers:
        mod._buffers[attr] = t
    else:
        raise QuantError(f"{key}: no such parameter or buffer in the model")


def load_quantized(path, device, *, impl: str = "auto", dtype=torch.float32) -> tuple[nn.Module, dict]:
    """A variant dir as a quantised model: the recipe and config.json; the HF skeleton on CPU (sdpa); the pointwise
    adapters; the file's kept tensors (in `dtype`, BatchNorm statistics fp32, the fp16 variant's fp16 tensors fp16;
    a missing or extra key refuses); the tie; the model to the device; each layer's pack installed for `impl`
    (emulate, or torchao on its device); eval mode; assert_quantized. The caller applies the rel-pos patch and the
    BatchNorm freeze as for any student. Returns (model, recipe)."""
    from safetensors import safe_open

    path = Path(path)
    if problems := verify_export(path):
        raise QuantError(f"{path}: not a valid variant dir: " + "; ".join(problems[:10]))
    rec = read_recipe(path)
    fmt = rec["format"]
    dev = torch.device(device)
    impl_r = resolve_impl(fmt, impl, dev)
    family = (rec.get("source") or {}).get("family") or rec.get("family")
    model = _skeleton(path, family)
    layers = rec.get("layers") or {}
    for name, info in layers.items():
        if info["kind"] == "pointwise_conv1d":
            src = name[:-len(".linear")]
            _swap_in(model, src, PointwiseConv1dAsLinear(model.get_submodule(src)))
    tied = rec.get("tied") or {}
    sd_keys = set(model.state_dict().keys())
    wfmt = "fp16" if fmt == "fp16" else _SPLIT[fmt][0]
    expected = (sd_keys - {f"{n}.weight" for n in layers} - set(tied)) | {
        f"{n}.{part}" for n in layers for part in _PACK_PARTS.get(wfmt, ())}
    fp16_set = set(rec.get("fp16_tensors") or [])
    bn_stats = {f"{n}.{b}" for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
                for b in ("running_mean", "running_var")}
    packs: dict[str, dict] = {n: {} for n in layers}
    with safe_open(str(path / WEIGHTS_FILE), framework="pt", device="cpu") as fh:
        keys = set(fh.keys())
        if keys != expected:
            miss, extra = sorted(expected - keys), sorted(keys - expected)
            raise QuantError(f"{path / WEIGHTS_FILE}: {len(miss)} tensors missing (e.g. {miss[:3]}), {len(extra)} "
                             f"not in the model (e.g. {extra[:3]})")
        for key in sorted(keys):
            t = fh.get_tensor(key)
            layer, _, part = key.rpartition(".")
            if layer in packs and part in _PACK_PARTS.get(wfmt, ()):
                packs[layer][part] = t
                continue
            if t.is_floating_point():
                t = t.to(torch.float32 if key in bn_stats else torch.float16 if key in fp16_set else dtype)
            _set_tensor(model, key, t)
    for alias, target in tied.items():
        if model.get_parameter(alias) is not model.get_parameter(target):
            raise QuantError(f"{alias} is not tied to {target} after loading")
    model = model.to(dev)
    cache: dict = {}
    for name, info in layers.items():
        p = packs[name]
        n, k = info["shape"]
        pack = QuantPack(wfmt, (n, k), p["qweight"], scale=p.get("qscale"), block_scale=p.get("qblock_scale"),
                         tensor_scale=p.get("qtensor_scale"),
                         mx_rounding=(rec.get("recipe") or {}).get("mx_rounding") if wfmt == "mxfp4" else None)
        _install(model.get_submodule(name), pack, fmt, impl_r, name, info["kind"], cache=cache)
    model.eval()
    model._kitsune_quant = dict(recipe=rec, bn=_bn_snapshot(model))
    assert_quantized(model, rec)
    return model, rec


# ---------------------------------------------------------------------------------------------- monitors


def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


class NonFiniteMonitor:
    """Forward hooks on every ParakeetEncoderBlock and CohereAsrDecoderLayer and on the encoder's output: counts the
    model forwards (an encoder pass starts one) with a non-finite value, the rows they touch and the modules they
    showed in, without a host sync per hook (one per forward). record() -> {batches (forwards with a non-finite
    value), rows, by_module {name: forwards}, first (the first module that showed one), forwards}. With flush_path it
    writes its record (and the model's QuantLinear counters) there every flush_s seconds and at detach, so a killed
    eval leaves what it saw."""

    BLOCKS = ("ParakeetEncoderBlock", "CohereAsrDecoderLayer")
    ENCODERS = ("ParakeetEncoder",)

    def __init__(self, model: nn.Module, *, flush_path=None, flush_s: float = 30.0):
        self.model = model
        mods = [(n, m) for n, m in model.named_modules() if type(m).__name__ in self.BLOCKS + self.ENCODERS]
        enc = [(n, m) for n, m in mods if type(m).__name__ in self.ENCODERS]
        blocks = [(n, m) for n, m in mods if type(m).__name__ == self.BLOCKS[0]]
        dec = [(n, m) for n, m in mods if type(m).__name__ == self.BLOCKS[1]]
        self._order = [n for n, _ in blocks] + [f"{n}:output" for n, _ in enc] + [n for n, _ in dec]
        self._mods = dict(blocks=blocks, enc=enc, dec=dec)
        self.forwards = self.batches = self.rows = 0
        self.by_module: Counter = Counter()
        self.first = None
        self._cur = None
        self._handles = []
        self.flush_path = Path(flush_path) if flush_path else None
        self.flush_s = float(flush_s)
        self._last_flush = time.monotonic()

    def attach(self) -> "NonFiniteMonitor":
        if not self._mods["enc"]:
            raise QuantError("NonFiniteMonitor: the model has no ParakeetEncoder to count forwards by")
        for n, m in self._mods["enc"]:
            self._handles.append(m.register_forward_pre_hook(lambda mod, args: self._start()))
            self._handles.append(m.register_forward_hook(self._hook(f"{n}:output")))
        for n, m in self._mods["blocks"] + self._mods["dec"]:
            self._handles.append(m.register_forward_hook(self._hook(n)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._finish()
        self.flush(force=True)

    def _start(self):
        self._finish()
        self._cur = dict(B=None, rows=None, mods={})

    def _hook(self, name: str):
        def hook(module, args, output):
            t = _first_tensor(output)
            if t is None or not t.is_floating_point():
                return
            if self._cur is None:
                self._start()
            bad = ~torch.isfinite(t.detach())
            row = bad.reshape(1, -1).any(1) if t.dim() == 0 else bad.reshape(t.shape[0], -1).any(1)
            cur = self._cur
            if cur["B"] is None:
                cur["B"], cur["rows"] = row.shape[0], row.clone()
            elif row.shape[0] == cur["B"]:
                cur["rows"] = cur["rows"] | row
            else:
                cur["rows"] = cur["rows"] | row.any()
            prev = cur["mods"].get(name)
            cur["mods"][name] = row.any() if prev is None else (prev | row.any())
        return hook

    def _finish(self):
        cur, self._cur = self._cur, None
        if cur is None or cur["B"] is None:
            return
        self.forwards += 1
        rows = int(cur["rows"].sum())
        if rows:
            self.batches += 1
            self.rows += rows
            flags = {n: bool(v) for n, v in cur["mods"].items()}
            for n in self._order:
                if flags.get(n):
                    self.by_module[n] += 1
                    if self.first is None:
                        self.first = n
        self.flush()

    def record(self) -> dict:
        self._finish()
        return dict(batches=self.batches, rows=self.rows, by_module=dict(self.by_module), first=self.first,
                    forwards=self.forwards)

    def flush(self, force: bool = False):
        if self.flush_path is None or (not force and time.monotonic() - self._last_flush < self.flush_s):
            return
        self._last_flush = time.monotonic()
        rec = dict(nonfinite=dict(batches=self.batches, rows=self.rows, by_module=dict(self.by_module),
                                  first=self.first, forwards=self.forwards),
                   counters=counters(self.model), time_utc=_now())
        try:
            self.flush_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.flush_path.with_name(self.flush_path.name + ".tmp")
            _write_bytes_json(tmp, rec)
            os.replace(tmp, self.flush_path)
        except OSError:  # a record that cannot be written must not end the eval it reports on
            pass


def merge_nonfinite(records: list[dict]) -> dict:
    """NonFiniteMonitor records summed (several invocations of one eval)."""
    out = dict(batches=0, rows=0, by_module=Counter(), first=None, forwards=0)
    for r in records:
        for k in ("batches", "rows", "forwards"):
            out[k] += int(r.get(k) or 0)
        out["by_module"].update(r.get("by_module") or {})
        out["first"] = out["first"] or r.get("first")
    out["by_module"] = dict(out["by_module"])
    return out


_OPS = ("aten::_int_mm", "aten::_scaled_mm", "aten::_scaled_mm_v2", "aten::mm", "aten::addmm", "aten::linear",
        "aten::bmm", "aten::matmul")


def _low_op(name: str) -> bool:
    """A profiler op that is itself a low-precision (fp8 / fp4) GEMM: torch's scaled matmuls (torchao 0.18's NVFP4 and
    FP8 linears call torch._scaled_mm) and the MSLK / FBGEMM kernels a torchao kernel preference may pick instead."""
    return "scaled_mm" in name or name.startswith(("mslk::", "fbgemm::"))


def _kernel_class(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("e2m1", "fp4", "nvf4", "f4f4", "mxf4", "_f4")):
        return "fp4"
    if any(s in n for s in ("e4m3", "fp8", "f8f8", "_f8")):
        return "fp8"
    if any(s in n for s in ("i8i8", "int8", "imma", "s8s8", "_i8", "igemm")):
        return "int8"
    if any(s in n for s in ("bf16", "bfloat16")):
        return "bf16"
    if any(s in n for s in ("gemm", "sgemm", "hgemm", "cutlass", "matmul", "_mm")):
        return "other_gemm"
    return "other"


@contextmanager
def count_int_mm():
    """torch._int_mm wrapped for the with-block, yielding {calls, ok, raised}: ok counts the int8 GEMMs that RETURNED.
    That is the evidence the int8 path ran: torchao's safe_int_mm calls torch._int_mm (through torch's out_dtype op,
    which looks torch._int_mm up at call time) inside try/except and silently runs an fp32 matmul when it raises
    (M <= 16 on CUDA, a layout it refuses), and the profiler records an aten::_int_mm for the call that raised as
    well. Only while nothing is compiled (a compiled graph does not look the name up)."""
    orig = torch._int_mm
    n = dict(calls=0, ok=0, raised=0)

    def counted(*args, **kwargs):
        n["calls"] += 1
        try:
            out = orig(*args, **kwargs)
        except BaseException:
            n["raised"] += 1
            raise
        n["ok"] += 1
        return out

    torch._int_mm = counted
    try:
        yield n
    finally:
        torch._int_mm = orig


def kernel_census(fn, device, *, model: nn.Module | None = None) -> dict:
    """What fn() ran, under torch.profiler: the counts of the matmul ops (aten::_int_mm, _scaled_mm, mm, addmm,
    linear, ...; any other scaled-matmul op too) and, on CUDA, the GEMM kernels by class (int8, fp4, fp8, bf16,
    other_gemm). With the quantised model (any device):
      int8_act_calls  its int8-activation QuantLinear calls during fn
      int8_mm         {calls, ok, raised} of torch._int_mm under count_int_mm (ok: the int8 GEMMs that returned),
                      when the model has a torchao int8-activation layer
      fallback_mm     the int8-activation calls of its torchao layers less the int8 GEMMs that returned: torchao's
                      silent fp32 fallback, 0 when every such call ran the int8 GEMM. None when it cannot be told:
                      a model without a torchao int8-activation layer (emulate runs no int8 GEMM), counters off, or
                      an int8 GEMM that ran past the wrapper (the profiler saw aten::_int_mm, the wrapper no call)
    The profiler's aten::_int_mm count is information only: it counts a call that raised. With the model's counters
    off (speed_probe --compile: the calls were not counted), int8_act_calls, int8_mm and fallback_mm are None."""
    from torch.profiler import ProfilerActivity, profile

    dev = torch.device(device)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    qls = list(quant_layers(model).values()) if model is not None else []
    counted = model is not None and all(m.kq.count for m in qls)
    int8 = [m for m in qls if m.kq.act == "int8"]
    real = [m for m in int8 if m.kq.impl == "torchao"]
    before = (sum(m.kq.calls for m in int8), sum(m.kq.calls for m in real))
    t0 = time.perf_counter()
    with ExitStack() as stack:
        n = stack.enter_context(count_int_mm()) if counted and real else None
        with profile(activities=acts) as prof:
            fn()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0
    ops, kernels, classes = Counter(), Counter(), Counter()
    for e in prof.events():
        name = e.name
        if name in _OPS or _low_op(name):
            ops[name] += 1
        dt = getattr(e, "device_type", None)
        if dt is not None and "cuda" in str(dt).lower():
            kernels[name] += 1
            cls = _kernel_class(name)
            if cls != "other":
                classes[cls] += 1
    rec = dict(ops=dict(ops), gemm_classes=dict(classes), kernels=dict(kernels.most_common(40)), wall_s=wall)
    if model is not None:
        rec.update(int8_act_calls=None, int8_mm=None, fallback_mm=None)
        if counted:
            rec["int8_act_calls"] = sum(m.kq.calls for m in int8) - before[0]
        if n is not None:
            rec["int8_mm"] = dict(n)
            if n["calls"] == 0 and ops.get("aten::_int_mm", 0) > 0:  # ran past the wrapper: success unknown
                rec["int8_mm"]["unseen"] = ops["aten::_int_mm"]
            else:
                rec["fallback_mm"] = max(sum(m.kq.calls for m in real) - before[1] - n["ok"], 0)
    return rec


def low_gemm_evidence(fmt: str, census: dict) -> dict:
    """{ok, evidence, warning}: whether a kernel census (with the model) proves the format's low-precision GEMM ran -
    the selftest's autocast check. A weight-only format needs none (torchao dequantises its weight to bf16). int8
    activations: every int8-activation call of the torchao layer returned from torch._int_mm (fallback_mm 0 over at
    least one call). fp4 / fp8 activations: a scaled-matmul op ran (_low_op), or a GEMM kernel of that class by name.
    The kernel name is not required: cuBLASLt's sm_120 GEMMs (nvjet_*) carry no dtype in their names; op evidence
    without a recognised name is a warning, never a failure."""
    act = _SPLIT[fmt][1] if fmt in _SPLIT else None
    classes = census.get("gemm_classes") or {}
    named = sum(v for c, v in classes.items() if c in ("int8", "fp4", "fp8"))
    if not act:
        return dict(ok=True, evidence="weight-only: its weight is dequantised to bf16, no low-precision GEMM",
                    warning=None)
    if act == "int8":
        calls, fb = census.get("int8_act_calls"), census.get("fallback_mm")
        ok = bool(calls) and fb == 0
        evidence = f"{calls} int8-activation call(s), fallback_mm {fb}, torch._int_mm {census.get('int8_mm')}"
    else:
        n = sum(v for k, v in (census.get("ops") or {}).items() if _low_op(k))
        ok = n > 0 or named > 0
        evidence = f"{n} scaled-matmul op(s), {named} low-precision GEMM kernel(s) by name"
    warning = None if not ok or named else ("no GEMM kernel name recognised as int8 / fp4 / fp8 (op evidence only): "
                                            "the census's kernels list what ran")
    return dict(ok=bool(ok), evidence=evidence, warning=warning)


# ---------------------------------------------------------------------------------------------- compare


def _greedy_sets(d: Path) -> dict[str, Path]:
    return {p.name[len("greedy_"):-len(".parquet")]: p for p in sorted(d.glob("greedy_*.parquet"))}


def compare_eval_dirs(a, b, *, exact: bool = True, tol_cer: float = 0.0) -> dict:
    """Two 05 --out dirs over the sets both hold: exact (every greedy hypothesis equal by id, and the teacher-forced
    per-utterance tables tf_<set> equal) or each set's corpus CER within tol_cer. {same, mode, tol_cer, sets: {set:
    {n_a, n_b, n_diff, hyps_equal, tf_equal, cer_a, cer_b, delta}}, only_a, only_b}; same is false when no set is
    shared."""
    import pandas as pd

    from kitsune.evaluate import corpus_cer

    a, b = Path(a), Path(b)
    sa, sb = _greedy_sets(a), _greedy_sets(b)
    both = sorted(set(sa) & set(sb))
    sets = {}
    for s in both:
        ga, gb = pd.read_parquet(sa[s]), pd.read_parquet(sb[s])
        ha = dict(zip(ga["id"].astype(str), ga["hyp"].astype(str)))
        hb = dict(zip(gb["id"].astype(str), gb["hyp"].astype(str)))
        ids = sorted(set(ha) | set(hb))
        n_diff = sum(ha.get(i) != hb.get(i) for i in ids)
        cer_a = corpus_cer(ga["hyp"].astype(str).tolist(), ga["ref"].astype(str).tolist())["cer"]
        cer_b = corpus_cer(gb["hyp"].astype(str).tolist(), gb["ref"].astype(str).tolist())["cer"]
        tf_equal = None
        ta, tb = a / f"tf_{s}.parquet", b / f"tf_{s}.parquet"
        if ta.is_file() and tb.is_file():
            fa = pd.read_parquet(ta).sort_values("id").reset_index(drop=True)
            fb = pd.read_parquet(tb).sort_values("id").reset_index(drop=True)
            tf_equal = bool(list(fa.columns) == list(fb.columns) and fa.equals(fb))
        delta = None if cer_a is None or cer_b is None else abs(float(cer_a) - float(cer_b))
        ok = (n_diff == 0 and tf_equal is not False) if exact else (delta is not None and delta <= tol_cer)
        sets[s] = dict(n_a=len(ha), n_b=len(hb), n_diff=int(n_diff), hyps_equal=n_diff == 0, tf_equal=tf_equal,
                       cer_a=cer_a, cer_b=cer_b, delta=delta, same=bool(ok))
    return dict(same=bool(both) and all(v["same"] for v in sets.values()), mode="exact" if exact else "tol_cer",
                tol_cer=None if exact else float(tol_cer), a=str(a), b=str(b), sets=sets,
                only_a=sorted(set(sa) - set(sb)), only_b=sorted(set(sb) - set(sa)), time_utc=_now())


# ---------------------------------------------------------------------------------------------- selftest


def _rel_frob(a: torch.Tensor, b: torch.Tensor) -> float:
    d = float(torch.linalg.vector_norm((a.float() - b.float()).reshape(-1)))
    n = float(torch.linalg.vector_norm(b.float().reshape(-1)))
    return d / n if n else (0.0 if d == 0 else math.inf)


class _One(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return self.lin(x)


def quantize_linear(lin: nn.Linear, fmt: str, impl: str, *, mx_rounding: str = "rceil",
                    min_rows: int | None = None) -> nn.Linear:
    """One Linear quantised in place (the selftest's layers; apply does this per selected layer)."""
    wfmt, _ = _SPLIT[check_format(fmt)]
    impl_r = resolve_impl(fmt, impl, lin.weight.device)
    _install(lin, pack_weight(lin.weight, wfmt, mx_rounding=mx_rounding), fmt, impl_r, "linear", "linear")
    if min_rows is not None:
        lin.kq.min_rows = int(min_rows)
    return lin


def selftest(device, *, ckpt=None, rows=(1, 7, 17, 128, 1000), shape=(2560, 1024)) -> dict:
    """Smoke B check 12 (module docstring). Per timed format on an (N, K) Linear: torchao against emulate at every M
    (relative Frobenius error within SELFTEST_TOL), the kernel census and the wall time; int8 W8A8: no fp32 fallback
    after padding (every call's torch._int_mm returned: fallback_mm 0, kernel_census) and, unpadded at M=1, whether
    the trap shows (torch._int_mm raising; a warning when it returned); under torch.autocast(bf16) with an fp32 input
    the weight stays torchao's and the low-precision GEMM runs (W*A* formats, by op evidence: low_gemm_evidence);
    MXFP4 through torchao's AUTO kernel must raise (a warning, "re-evaluate decision 20", when it does not). With ckpt
    dirs: the same comparison on the student (a CTC encoder batch; an AED greedy_generate(pin_new=8) at batch 1 and
    8). ok = every hard check passed."""
    dev = torch.device(device)
    rec = dict(schema=SCHEMA, device=str(dev), versions=_versions(), rows=[int(r) for r in rows], shape=list(shape),
               formats={}, checks=[], warnings=[], ckpt={}, ok=False, time_utc=_now())

    def check(name, ok, detail=None):
        rec["checks"].append(dict(name=name, ok=bool(ok), detail=detail))

    if dev.type != "cuda" or not torch.cuda.is_available():
        check("environment", False, f"the selftest times real kernels: it needs a CUDA device, got {dev}")
    elif not torchao_available():
        check("environment", False, "torchao is not importable")
    else:
        rec["gpu"] = torch.cuda.get_device_name(dev)
        rec["capability"] = list(torch.cuda.get_device_capability(dev))
        check("environment", True, rec["gpu"])
        _selftest_layers(rec, check, dev, rows, shape)
        _selftest_mxfp4(rec, dev, shape)
        for d in ckpt or []:
            _selftest_ckpt(rec, check, dev, Path(d))
    rec["ok"] = all(c["ok"] for c in rec["checks"])
    return rec


def _selftest_layers(rec, check, dev, rows, shape):
    n, k = shape
    g = torch.Generator(device="cpu").manual_seed(0)
    w = (torch.randn(n, k, generator=g) * 0.02).to(dev)
    bias = (torch.randn(n, generator=g) * 0.01).to(dev)

    def make(fmt, impl, min_rows=None):
        lin = nn.Linear(k, n, device=dev)
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(bias)
        return quantize_linear(lin, fmt, impl, min_rows=min_rows)

    for fmt in TIMED_FORMATS:
        frec = dict(per_m={})
        rec["formats"][fmt] = frec
        try:
            emu, real = make(fmt, "emulate"), make(fmt, "torchao")
            frec["weight_type"] = type(real.weight).__name__
            for m in rows:
                x = torch.randn(int(m), k, generator=g).to(dev, torch.bfloat16)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ye = emu(x)
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                    yr = real(x)
                    torch.cuda.synchronize(dev)
                    wall = time.perf_counter() - t0
                    census = kernel_census(lambda: real(x), dev, model=_One(real))
                err = _rel_frob(yr, ye)
                frec["per_m"][str(m)] = dict(rel_err=err, wall_s=wall, finite=bool(torch.isfinite(yr).all()),
                                             census=census)
                check(f"{fmt} M={m} real vs emulate", err <= SELFTEST_TOL[fmt] and bool(torch.isfinite(yr).all()),
                      f"rel_err {err:.3g} (tol {SELFTEST_TOL[fmt]})")
                if _SPLIT[fmt][1] == "int8":  # every int8 call's torch._int_mm returned (kernel_census)
                    check(f"{fmt} M={m} no fp32 fallback",
                          census.get("fallback_mm") == 0 and bool(census.get("int8_act_calls")),
                          f"fallback_mm {census.get('fallback_mm')}, int8_act_calls {census.get('int8_act_calls')}, "
                          f"torch._int_mm {census.get('int8_mm')}")
            # fp32 input under bf16 autocast: the weight stays torchao's, the low-precision GEMM runs (W*A*: by op
            # evidence, low_gemm_evidence; a kernel name is recorded, not required)
            before = type(real.weight)
            x = torch.randn(64, k, generator=g).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                census = kernel_census(lambda: real(x), dev, model=_One(real))
            low = low_gemm_evidence(fmt, census)
            frec["autocast"] = dict(census=census, weight_type=type(real.weight).__name__, low_gemm=low)
            check(f"{fmt} autocast keeps the weight quantised", type(real.weight) is before and low["ok"],
                  dict(weight_type=type(real.weight).__name__, evidence=low["evidence"],
                       gemm_classes=census["gemm_classes"]))
            if low["warning"]:
                rec["warnings"].append(f"{fmt} under autocast: {low['warning']}")
            if fmt == "int8-w8a8":  # the trap the padding avoids: unpadded at M=1, torch._int_mm must raise
                trap = make(fmt, "torchao", min_rows=0)
                x1 = torch.randn(1, k, generator=g).to(dev, torch.bfloat16)
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        c1 = kernel_census(lambda: trap(x1), dev, model=_One(trap))
                except Exception as e:  # noqa: BLE001 - torchao let the refusal through: no silent fallback either
                    c1 = dict(raised=f"{type(e).__name__}: {e}"[:400], fallback_mm=1)
                frec["unpadded_m1"] = c1
                if c1.get("fallback_mm") is None:
                    rec["warnings"].append(f"int8-w8a8 unpadded at M=1: whether the int8 GEMM returned could not be "
                                           f"counted (torch._int_mm {c1.get('int8_mm')})")
                elif c1["fallback_mm"] == 0:
                    rec["warnings"].append("int8-w8a8 unpadded at M=1: torch._int_mm returned, so it takes M <= 16 "
                                           "here now and the padding to 17 rows may no longer be needed")
        except Exception as e:  # noqa: BLE001 - recorded: the selftest reports every format
            frec["error"] = f"{type(e).__name__}: {e}"[:600]
            check(f"{fmt} builds and runs", False, frec["error"])


def _selftest_mxfp4(rec, dev, shape):
    """Decision 20: MXFP4 has no kernel on sm_120, so torchao's AUTO kernel preference must raise here."""
    n, k = shape
    out = dict(raised=None, error=None, config_error=None)
    rec["mxfp4_auto"] = out
    try:  # the config first, on its own: a torchao whose config API moved must not read as "AUTO refused MXFP4"
        cfg = mxfp4_auto_config()
    except Exception as e:  # noqa: BLE001
        out["config_error"] = f"{type(e).__name__}: {e}"[:400]
        rec["warnings"].append("MXFP4 through torchao's KernelPreference.AUTO was not tried: its config does not "
                               f"build ({out['config_error']}); decision 20 is unchecked here")
        return
    try:
        lin = nn.Linear(k, n, bias=False, device=dev, dtype=torch.bfloat16)
        _ao("quantize_")(lin, cfg)
        with torch.no_grad():
            lin(torch.randn(32, k, device=dev, dtype=torch.bfloat16))
        torch.cuda.synchronize(dev)
        out["raised"] = False
    except Exception as e:  # noqa: BLE001
        out.update(raised=True, error=f"{type(e).__name__}: {e}"[:400])
    if out["raised"] is False:
        rec["warnings"].append("MXFP4 through torchao's KernelPreference.AUTO ran on this GPU: re-evaluate decision 20")


def mxfp4_auto_config():
    """torchao 0.18's MXFP4 W4A4 config with its AUTO kernel preference (the selftest's decision-20 canary; the image
    CI builds it on CPU, tests/test_quant_torchao.py)."""
    fp4 = torch.float4_e2m1fn_x2
    return _ao("MXDynamicActivationMXWeightConfig")(activation_dtype=fp4, weight_dtype=fp4,
                                                     kernel_preference=_ao("KernelPreference").AUTO)


def _selftest_ckpt(rec, check, dev, d: Path):
    import copy

    out = {}
    rec["ckpt"][str(d)] = out
    try:
        family, _ = _family_of_dir(d)
        base = _load_student(d, family, dev)
        from kitsune.patches import patch_relpos_once_per_batch

        g = torch.Generator(device="cpu").manual_seed(1)
        for fmt in TIMED_FORMATS:
            r = dict()
            out[fmt] = r
            try:
                emu, real = copy.deepcopy(base), copy.deepcopy(base)
                apply(emu, fmt, impl="emulate")
                apply(real, fmt, impl="torchao")
                for m in (emu, real):
                    patch_relpos_once_per_batch(m)
                if family == "ctc":
                    from kitsune import ctc_student as CS

                    feats = torch.randn(4, 500, 80, generator=g).to(dev)
                    mask = torch.ones(4, 500, dtype=torch.bool, device=dev)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        le, _ = CS.ctc_log_probs(emu, feats, mask)
                        t0 = time.perf_counter()
                        lr, _ = CS.ctc_log_probs(real, feats, mask)
                        torch.cuda.synchronize(dev)
                        r["wall_s"] = time.perf_counter() - t0
                    r["rel_err"] = _rel_frob(lr, le)
                    r["argmax_agree"] = float((lr.argmax(-1) == le.argmax(-1)).float().mean())
                    check(f"ckpt {d.name} {fmt} finite", bool(torch.isfinite(lr).all()), r)
                else:
                    from kitsune import evaluate as ev
                    from kitsune import trainset

                    for bsz in (1, 8):
                        feats = torch.randn(bsz, 300, 128, generator=g).to(dev)
                        fmask = torch.ones(bsz, 300, dtype=torch.long, device=dev)
                        res = {}
                        for tag, mdl in (("emulate", emu), ("torchao", real)):
                            with ev._eval_mode(mdl), ev._fp32_head(mdl):
                                t0 = time.perf_counter()
                                res[tag] = ev.greedy_generate(mdl, feats, fmask, 3.0, prompt_ids=trainset.PROMPT,
                                                              eos=trainset.EOS, pad=trainset.PAD, amp=True, pin_new=8)
                                torch.cuda.synchronize(dev)
                                r[f"wall_s_b{bsz}_{tag}"] = time.perf_counter() - t0
                        same = sum(a[0] == b[0] for a, b in zip(res["emulate"], res["torchao"]))
                        r[f"token_rows_equal_b{bsz}"] = f"{same}/{bsz}"
                    check(f"ckpt {d.name} {fmt} decodes", True, r)
                del emu, real
            except Exception as e:  # noqa: BLE001
                r["error"] = f"{type(e).__name__}: {e}"[:600]
                check(f"ckpt {d.name} {fmt}", False, r["error"])
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:600]
        check(f"ckpt {d.name} loads", False, out["error"])


# ---------------------------------------------------------------------------------------------- CLI


def _load_05():
    import importlib.util

    spec = importlib.util.spec_from_file_location("kitsune_evaluate_for_quant", ROOT / "scripts" / "05_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exit_of(e: SystemExit) -> int:
    code = e.code
    if code is None or code == 0:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 2 if str(code).startswith("REFUSED") else 1


def readout(args) -> int:
    """quant readout: export to <out>/variant unless verify_export passes there (a retry reuses it), then 05 on it."""
    out = Path(args.out).resolve()
    variant = out / "variant"
    reuse = False
    if variant.exists() and not verify_export(variant):
        rec = read_recipe(variant)
        reuse = rec["format"] == args.fmt and (rec.get("source") or {}).get("weights_sha256") == weights_sha256(
            _weight_files(Path(args.ckpt)))
    if reuse:
        print(f"{variant}: the verified {args.fmt} variant of this checkpoint is reused", flush=True)
    else:
        export(args.ckpt, variant, args.fmt, device=args.device, force=True)
    argv = ["--config", args.config, "--ckpt", str(variant), "--out", str(out), "--tables", str(out / "tables"),
            "--cache-dir", args.cache_dir, "--manifest", args.manifest, "--max-temp", str(args.max_temp),
            "--quant-impl", args.impl]
    if args.system:
        argv += ["--system", args.system]
    if args.root:
        argv += ["--root", args.root]
    for s in args.set:
        argv += ["--set", s]
    try:
        return int(_load_05().main(argv) or 0)
    except SystemExit as e:
        return _exit_of(e)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m kitsune.quant", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write a variant dir of a bf16 student export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--fmt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--scope", default="linear+pw", choices=SCOPES)
    e.add_argument("--mx-rounding", default="rceil", choices=MX_ROUNDINGS)
    e.add_argument("--device", default="cpu")
    e.add_argument("--force", action="store_true")
    r = sub.add_parser("readout", help="export, then score the variant with scripts/05_evaluate.py")
    r.add_argument("--config", required=True)
    r.add_argument("--ckpt", required=True, help="the bf16 checkpoint (runs/<run_id>/checkpoints/step_<N>)")
    r.add_argument("--fmt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cache-dir", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--max-temp", type=float, default=0.0)
    r.add_argument("--system", default=None, help="default <run_name>@<fmt>")
    r.add_argument("--impl", default="auto", choices=IMPLS)
    r.add_argument("--device", default="cpu", help="the export's device (default cpu)")
    r.add_argument("--root", default=None, help="05's --root (default: this repo)")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="05's --set (repeatable)")
    i = sub.add_parser("inspect", help="print a variant dir's recipe summary and problems")
    i.add_argument("dir")
    c = sub.add_parser("compare", help="compare two 05 --out dirs")
    c.add_argument("a")
    c.add_argument("b")
    g = c.add_mutually_exclusive_group()
    g.add_argument("--exact", action="store_true", help="hyps and teacher-forced tables equal (the default)")
    g.add_argument("--tol-cer", type=float, default=None, help="each shared set's corpus CER within this")
    c.add_argument("--json-out", default=None)
    s = sub.add_parser("selftest", help="smoke B check 12 on a CUDA device")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--out", required=True)
    s.add_argument("--ckpt", action="append", default=[])
    s.add_argument("--rows", default="1,7,17,128,1000")
    return ap.parse_args(argv)


def _write_json_out(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_json(p, obj)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kitsune import heartbeat

    try:
        if args.cmd == "export":
            rec = export(args.ckpt, args.out, args.fmt, scope=args.scope, mx_rounding=args.mx_rounding,
                         device=args.device, force=args.force)
            print(f"{args.out}: {rec['format']} {rec['counts']['layers']} layers, "
                  f"{rec['counts']['share_quantized']:.3f} of the parameters, {rec['file_bytes'][WEIGHTS_FILE]} "
                  "bytes", flush=True)
            return 0
        if args.cmd == "readout":
            check_format(args.fmt)
            return readout(args)
        if args.cmd == "inspect":
            rec = read_recipe(args.dir)
            problems = verify_export(args.dir)
            print(json.dumps(dict(format=rec["format"], scope=rec.get("scope"), counts=rec.get("counts"),
                                  bytes=rec.get("bytes"), file_bytes=rec.get("file_bytes"),
                                  skipped=rec.get("skipped"), source=rec.get("source"), problems=problems),
                             indent=1, ensure_ascii=False))
            return 1 if problems else 0
        if args.cmd == "compare":
            with heartbeat.beating(max_s=BEAT_MAX_S):
                exact = args.tol_cer is None
                res = compare_eval_dirs(args.a, args.b, exact=exact, tol_cer=args.tol_cer or 0.0)
            print(json.dumps(res, indent=1, ensure_ascii=False))
            if args.json_out:
                _write_json_out(args.json_out, res)
            return 0 if res["same"] else 1
        if args.cmd == "selftest":
            rows = tuple(int(x) for x in str(args.rows).split(",") if x.strip())
            with heartbeat.beating(max_s=BEAT_MAX_S):
                res = selftest(args.device, ckpt=args.ckpt, rows=rows)
            _write_json_out(args.out, res)
            print(json.dumps(dict(ok=res["ok"], failed=[c for c in res["checks"] if not c["ok"]],
                                  warnings=res["warnings"]), indent=1, ensure_ascii=False))
            return 0 if res["ok"] else 1
    except QuantRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except QuantError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
