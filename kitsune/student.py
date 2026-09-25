"""Student construction: a shallower, narrower copy of the Cohere Transcribe teacher, initialised from its weights.

Shape (run sheet D5-D12): keep 20 of the 48 conformer layers, evenly spaced; prune each kept layer's two FFNs
5120 -> 2560 by activation importance; keep decoder layers {0,2,5,7}; full vocab; head tied to the token embedding.
Everything except the pruned FFN neurons is a verbatim teacher tensor, so the untrained student is already a
(damaged) copy of the teacher rather than a random network.

Why each step is done the way it is:
- The student is a FRESH `CohereAsrForConditionalGeneration(cfg)` loaded with a remapped state_dict under
  strict=True. Copying teacher module objects instead would keep their `layer_idx` (the decoder KV-cache slot), and
  `generate()` then indexes past the end of a 4-layer cache. A fresh model numbers its layers 0..n-1, and strict
  loading proves that every tensor came from the teacher.
- FFN importance is mean |act(linear1 x)| over valid (unpadded) encoder frames, measured inside the full teacher.
  The kept neurons are sorted by original index, so each slice keeps the teacher's order.
- The head is tied (proj_out.weight IS embed_tokens.weight; it keeps its own bias). In the teacher checkpoint the two
  are bitwise equal, so this costs nothing and saves V x 1024 parameters. The tied value is taken from embed_tokens.
- BatchNorm (one per conv module) is either recalibrated after pruning (the first run: `recalibrate_batchnorm`) or
  keeps the teacher's running stats (the size study, decision 16: `scripts/03_build_student.py --bn keep`). The
  recalibration's argument: the teacher's stats describe inputs that 28 dropped layers and half of every FFN used to
  shape. Against it: the same recalibration moves even the UNPRUNED teacher away from itself (greedy CER vs its own
  transcript 119 % on a CPU check), because per-utterance batch-size-1 statistics are not the pooled statistics the
  teacher was trained with. Recalibration uses batch size 1, so padded frames never enter the statistics, and
  momentum=None, which gives a cumulative average rather than an EMA biased to the last batches. Either way BN then
  stays frozen in eval mode for all of training (kitsune.patches.freeze_batchnorm).
- The checkpoint is saved with bf16 weights but fp32 BN running stats. Those stats are never trained again, and bf16
  would quantise the variances that every normalised frame is divided by.

The decoder prompt, EOS, PAD and vocab are the teacher's (decoder_start_token_id 13764, see teacher_out/meta.json).

From-scratch students (the size study's T-0.1B, T-0.05B, the replicate and the bridge; `build_scratch_student`) are
the teacher's architecture at other widths and depths, randomly initialised. Each trap below was measured once:
- The config starts from `student_config(teacher_config, ...)`, never from the raw teacher config: that one has
  `tie_word_embeddings: False` and would add a second V x D matrix (+8 % at T-0.1B, +12 % at T-0.05B).
- `head_dim = D / heads` and `num_key_value_heads = heads` are set explicitly, in the encoder and the decoder. The
  teacher config serialises `head_dim: 128` (T-0.1B would silently get 1024-wide attention projections) and
  `num_key_value_heads: 8` (6 heads with 8 kv heads gives `num_key_value_groups = 0`).
- The decoder's pos_emb is an nn.Embedding that HF initialises N(0, 0.02); the teacher's is a fixed sinusoid table
  divided by sqrt(D), which the trainer freezes. It is written in exactly the teacher's form (`sinusoid_pos_emb`).
- The subsampling Conv2d layers are re-initialised with the PyTorch default (kaiming-uniform). HF's N(0, 0.02) makes
  five stacked convs shrink the signal to an output RMS of ~8e-6, with a gradient norm of 361-1157 at init; the
  default gives ~0.05 and ~3.5.
- BatchNorm is fresh (running mean 0, var 1, no batches tracked) and trains; generation ids, the processor and
  `scale_input: False` are the teacher's; dropout and layerdrop are 0.
- The build is seeded (its own RNG fork) and asserts its parameter count equals `closed_form_params` and, for the
  named shapes, the count the study pre-registered (SCRATCH_EXPECTED_PARAMS).
"""
import copy
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

from kitsune.patches import unfreeze_batchnorm

TEACHER_ID = "CohereLabs/cohere-transcribe-03-2026"
TEACHER_ENC_LAYERS = 48
TEACHER_DEC_LAYERS = 8
TEACHER_FFN = 5120  # the teacher's encoder FFN width (the model card's "FFN 5120 -> ...")
DECODER_START, EOS, PAD, BOS = 13764, 3, 2, 4
FFN_NAMES = ("feed_forward1", "feed_forward2")
EXPECTED_DEFAULT_PARAMS = 616_963_328  # closed form for B20x2560 / dec4 / V16384 tied (report_model.md section 2)
# The size study's pruned shapes, (kept encoder layers, FFN width, kept decoder layers) -> total parameters (STUDY.md
# 1.1). 03_build_student.py asserts a build of one of these shapes from the 48-layer teacher lands on its count.
PRUNED_EXPECTED_PARAMS = {
    (20, 2560, (0, 2, 5, 7)): EXPECTED_DEFAULT_PARAMS,  # T-0.6B, the first run's student
    (10, 2560, (0, 7)): 320_752_384,  # T-0.3B
}
# save_student writes it next to the weights as README.md, its modification notice fitted to the student (model_card)
MODEL_CARD = Path(__file__).resolve().parents[1] / "MODEL_CARD.md"

_BN = nn.modules.batchnorm._BatchNorm


@dataclass
class StudentSpec:
    enc_layers: list[int]  # teacher encoder layer indices to keep, ascending
    ffn_dim: int  # encoder FFN width; == the teacher's 5120 means no FFN pruning
    dec_layers: list[int]  # teacher decoder layer indices to keep, ascending
    tie_head: bool = True

    def __post_init__(self):
        for name in ("enc_layers", "dec_layers"):
            v = [int(x) for x in getattr(self, name)]
            if not v or v != sorted(set(v)) or v[0] < 0:
                raise ValueError(f"{name} must be non-empty, ascending and unique: {v}")
            setattr(self, name, v)
        self.ffn_dim = int(self.ffn_dim)


def evenly_spaced(n_keep: int, n_total: int) -> list[int]:
    """round(linspace(0, n_total-1, n_keep)) with halves rounded up (numpy's round-half-even could repeat an index).
    Integer arithmetic, so no float edge cases: evenly_spaced(20, 48) = [0, 2, 5, 7, 10, ..., 45, 47]."""
    if not 1 <= n_keep <= n_total:
        raise ValueError(f"need 1 <= n_keep <= n_total, got {n_keep}, {n_total}")
    if n_keep == 1:
        return [0]
    den = 2 * (n_keep - 1)
    return [(2 * i * (n_total - 1) + n_keep - 1) // den for i in range(n_keep)]


def default_spec(n_enc: int = 20, ffn: int = 2560, dec=(0, 2, 5, 7)) -> StudentSpec:
    return StudentSpec(enc_layers=evenly_spaced(n_enc, TEACHER_ENC_LAYERS), ffn_dim=ffn, dec_layers=list(dec))


# ------------------------------------------------------------------------------------------------ config + counts


def student_config(teacher_config: CohereAsrConfig, spec: StudentSpec) -> CohereAsrConfig:
    """The teacher config with the student's depth/width, tied head, explicit decoder start and sdpa everywhere.

    The teacher's config.json is NeMo-flavoured: next to the HF fields it carries `encoder`, `transf_decoder`, `head`,
    etc. describing a 48-layer model. Those keys are dropped here so that nothing in the student's config.json
    contradicts its real shape."""
    t_enc = teacher_config.encoder_config
    if spec.enc_layers[-1] >= t_enc.num_hidden_layers or spec.dec_layers[-1] >= teacher_config.num_hidden_layers:
        raise ValueError(f"spec {spec} exceeds the teacher ({t_enc.num_hidden_layers} enc / "
                         f"{teacher_config.num_hidden_layers} dec layers)")
    if not 1 <= spec.ffn_dim <= t_enc.intermediate_size:
        raise ValueError(f"ffn_dim {spec.ffn_dim} not in [1, {t_enc.intermediate_size}]")
    cfg = copy.deepcopy(teacher_config)
    for key in set(cfg.to_dict()) - set(type(cfg)().to_dict()):
        if hasattr(cfg, key):
            delattr(cfg, key)
    cfg.encoder_config.num_hidden_layers = len(spec.enc_layers)
    cfg.encoder_config.intermediate_size = spec.ffn_dim
    cfg.num_hidden_layers = len(spec.dec_layers)
    cfg.tie_word_embeddings = spec.tie_head
    cfg.decoder_start_token_id = DECODER_START
    # a bf16-loaded teacher leaves dtype=bf16 on BOTH configs, and AutoModel.from_config builds the encoder in it
    cfg.dtype = torch.float32
    cfg.encoder_config.dtype = torch.float32
    # eager -> NaN on padded batches; flex silently collapses the per-head rel-pos bias (report_model.md section 0)
    cfg._attn_implementation = "sdpa"
    cfg.encoder_config._attn_implementation = "sdpa"
    return cfg


def closed_form_params(cfg: CohereAsrConfig) -> int:
    """Parameter count from config fields alone. For the real dims it reduces to report_model.md's
    P = 5,383,424 + L_enc(13,149,440 + 5,122 F) + L_dec 16,796,672 + 2,364,416 + 1024 V + (V tied | 1025 V untied)."""
    e = cfg.encoder_config
    d, f = e.hidden_size, e.intermediate_size
    ab, cb = int(e.attention_bias), int(e.convolution_bias)
    c, k2, n_sub = e.subsampling_conv_channels, e.subsampling_conv_kernel_size ** 2, int(math.log2(e.subsampling_factor))
    f_out = e.num_mel_bins // (e.subsampling_conv_stride ** n_sub)
    sub = (k2 * c + c) + (n_sub - 1) * ((k2 * c + c) + (c * c + c)) + (c * f_out * d + d)
    ffn = d * f + ab * f + f * d + ab * d
    h = e.num_attention_heads * (d // e.num_attention_heads)
    self_attn = (d * h + ab * h) * 3 + (h * d + ab * d) + d * h + 2 * h  # q,k,v + o + relative_k_proj + bias_u/v
    conv = (2 * d * d + cb * 2 * d) + (e.conv_kernel_size * d + cb * d) + 2 * d + (d * d + cb * d)  # pw1, dw, BN, pw2
    enc_layer = 2 * ffn + self_attn + conv + 5 * 2 * d

    D, V, b = cfg.hidden_size, cfg.vocab_size, int(cfg.attention_bias)
    hq, hkv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    attn = (D * hq + b * hq) + 2 * (D * hkv + b * hkv) + (hq * D + b * D)
    dec_layer = 2 * attn + (D * cfg.intermediate_size + cfg.intermediate_size + cfg.intermediate_size * D + D) + 3 * 2 * D
    dec_fixed = V * D + cfg.max_position_embeddings * D + (d * D + D) + 2 * 2 * D  # embed, pos_emb, proj, 2 LN
    head = V if cfg.tie_word_embeddings else V * D + V
    return sub + e.num_hidden_layers * enc_layer + cfg.num_hidden_layers * dec_layer + dec_fixed + head


def _n(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def param_report(model: CohereAsrForConditionalGeneration) -> dict:
    """Unique parameter counts per component, laid out like report_model.md section 2 (all layers of a stack have the
    same shape, so one layer is shown). `total` counts a tied tensor once; `closed_form` is computed from the config."""
    enc, dec = model.model.encoder, model.model.decoder
    tied = model.proj_out.weight is dec.embed_tokens.weight

    def enc_layer(layer) -> dict:
        norms = ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out")
        return dict(ffn=_n(layer.feed_forward1) + _n(layer.feed_forward2), self_attn=_n(layer.self_attn),
                    conv=_n(layer.conv), norms=sum(_n(getattr(layer, k)) for k in norms), total=_n(layer))

    def dec_layer(layer) -> dict:
        norms = ("input_layernorm", "post_attention_layernorm", "final_layernorm")
        return dict(self_attn=_n(layer.self_attn), cross_attn=_n(layer.encoder_attn), mlp=_n(layer.mlp),
                    norms=sum(_n(getattr(layer, k)) for k in norms), total=_n(layer))

    dec_fixed = dict(embed_tokens=_n(dec.embed_tokens), pos_emb=_n(dec.pos_emb), proj=_n(dec.proj),
                     norms=_n(dec.norm) + _n(dec.embedding_layernorm))
    dec_fixed["total"] = sum(dec_fixed.values())
    head = sum(p.numel() for name, p in model.proj_out.named_parameters() if not (tied and name == "weight"))
    total = sum(p.numel() for p in model.parameters())
    rep = dict(
        total=total, closed_form=closed_form_params(model.config), tied_head=tied,
        subsampling=_n(enc.subsampling),
        encoder_layers=len(enc.layers), encoder_layer=enc_layer(enc.layers[0]), encoder_total=_n(enc),
        decoder_layers=len(dec.layers), decoder_layer=dec_layer(dec.layers[0]), decoder_fixed=dec_fixed,
        decoder_total=_n(dec), proj_out=head,
        batchnorm_buffers=sum(b.numel() for name, b in model.named_buffers() if ".running_" in name),
    )
    rep["encoder_share"] = round(rep["encoder_total"] / total, 4)
    assert rep["encoder_total"] + rep["decoder_total"] + head == total, rep
    return rep


def non_embedding_params(model: CohereAsrForConditionalGeneration) -> int:
    """STUDY.md 1.1's size measure: the total minus the vocabulary-sized matrices (the token embedding, and the head
    weight if it is not tied to it) and the frozen pos_emb table. The head's V-sized bias stays in."""
    dec = model.model.decoder
    n = sum(p.numel() for p in model.parameters()) - dec.embed_tokens.weight.numel() - dec.pos_emb.weight.numel()
    if model.proj_out.weight is not dec.embed_tokens.weight:
        n -= model.proj_out.weight.numel()
    return n


def count_fields(model: CohereAsrForConditionalGeneration) -> dict:
    """The count fields every study student's student_meta.json carries (the build asserts total == closed form)."""
    return dict(params_total=sum(p.numel() for p in model.parameters()),
                params_non_embedding=non_embedding_params(model), closed_form_params=closed_form_params(model.config))


# ------------------------------------------------------------------------------------------------ FFN importance


def _as_batch(item, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """(feats (B,T,128), mask (B,T) bool/int or None) -> on `device`; features in the model dtype, mask int64."""
    feats, mask = item
    feats = feats.to(device=device, dtype=dtype, non_blocking=True)
    if mask is None:
        mask = torch.ones(feats.shape[:2], dtype=torch.int64, device=device)
    return feats, mask.to(device=device, dtype=torch.int64, non_blocking=True)


@torch.no_grad()
def ffn_importance(teacher: CohereAsrForConditionalGeneration, feats_iter: Iterable, layers: list[int],
                   device) -> dict[tuple[int, str], torch.Tensor]:
    """Mean |act(linear1 x)| per FFN neuron over the valid encoder frames of every batch in `feats_iter`.

    feats_iter yields (feats (B,T,128) float, mask (B,T) bool) as kitsune.features.LogMel produces them. Only the
    encoder runs. Returns {(teacher_layer, 'feed_forward1'|'feed_forward2'): (intermediate_size,) float32 on CPU}."""
    enc = teacher.model.encoder
    dtype = next(teacher.parameters()).dtype
    width = enc.config.intermediate_size
    sums = {(l, n): torch.zeros(width, dtype=torch.float64, device=device) for l in layers for n in FFN_NAMES}
    state = dict(in_mask=None, frame_mask=None, frames=0)

    def frame_mask(T: int) -> torch.Tensor:  # valid encoder frames of the current batch, computed once per batch
        m = state["frame_mask"]
        if m is None or m.shape[1] != T:
            m = enc._get_output_attention_mask(state["in_mask"], target_length=T).to(torch.float32)
            state["frame_mask"] = m
        return m

    def hook_for(key, act):
        def hook(_mod, _inp, out):
            a = act(out.float()).abs()  # fp32 even for a bf16 teacher
            sums[key] += torch.einsum("btf,bt->f", a, frame_mask(out.shape[1])).double()
        return hook

    handles = []
    for l in layers:
        for n in FFN_NAMES:
            ff = getattr(enc.layers[l], n)
            handles.append(ff.linear1.register_forward_hook(hook_for((l, n), ff.activation)))
    was_training = teacher.training
    teacher.eval()
    try:
        for item in feats_iter:
            feats, mask = _as_batch(item, device, dtype)
            state["in_mask"], state["frame_mask"] = mask, None
            out = enc(input_features=feats, attention_mask=mask)  # a mask is always passed -> out.attention_mask set
            state["frames"] += int(out.attention_mask.sum())
    finally:
        for h in handles:
            h.remove()
        teacher.train(was_training)
    if state["frames"] == 0:
        raise ValueError("ffn_importance: feats_iter was empty")
    return {key: (s / state["frames"]).float().cpu() for key, s in sums.items()}


def select_ffn_neurons(importance: dict | None, enc_layers: list[int], ffn_dim: int,
                       teacher_ffn: int) -> dict[tuple[int, str], torch.Tensor]:
    """{(teacher_layer, ffn_name): kept neuron indices, sorted ascending}; {} when nothing is pruned.
    Ranking is per FFN; ties (e.g. dead neurons) break towards the lower index, so the choice is deterministic."""
    if ffn_dim == teacher_ffn:
        return {}
    if importance is None:
        raise ValueError(f"pruning FFNs {teacher_ffn} -> {ffn_dim} needs an importance dict (see ffn_importance)")
    keep = {}
    for l in enc_layers:
        for n in FFN_NAMES:
            imp = importance.get((l, n))
            if imp is None or tuple(imp.shape) != (teacher_ffn,):
                raise ValueError(f"importance[{(l, n)}] missing or not shaped ({teacher_ffn},)")
            order = torch.sort(imp.detach().float().cpu(), descending=True, stable=True).indices
            keep[(l, n)] = order[:ffn_dim].sort().values
    return keep


def check_keep(keep: dict, spec: StudentSpec, teacher_ffn: int) -> dict[tuple[int, str], torch.Tensor]:
    """A given FFN selection (e.g. another student's, see keep_from_meta) restricted to spec's layers, validated: every
    kept FFN of spec has exactly spec.ffn_dim strictly ascending neuron indices inside the teacher's width."""
    if spec.ffn_dim == teacher_ffn:
        return {}
    out = {}
    for l in spec.enc_layers:
        for n in FFN_NAMES:
            if (l, n) not in keep:
                raise ValueError(f"the given FFN selection has no entry for teacher layer {l} {n}")
            idx = torch.as_tensor(keep[(l, n)], dtype=torch.int64).flatten()
            if len(idx) != spec.ffn_dim or not bool((idx[1:] > idx[:-1]).all()) or int(idx[0]) < 0 or \
                    int(idx[-1]) >= teacher_ffn:
                raise ValueError(f"FFN selection {l}.{n}: need {spec.ffn_dim} ascending indices in [0, {teacher_ffn})")
            out[(l, n)] = idx
    return out


def keep_from_meta(meta: dict) -> dict[tuple[int, str], torch.Tensor]:
    """The FFN selection a built student recorded in its student_meta.json (`kept.ffn`, keys 'layer.ffn_name')."""
    ffn = (meta.get("kept") or {}).get("ffn")
    if not ffn:
        raise ValueError("student_meta.json has no kept.ffn (was that student FFN-pruned?)")
    return {(int(k.split(".", 1)[0]), k.split(".", 1)[1]): torch.as_tensor(v, dtype=torch.int64)
            for k, v in ffn.items()}


def importance_summary(importance: dict, keep: dict) -> dict:
    """How much activation mass each pruned FFN keeps, for student_meta.json."""
    per = {}
    for (l, n), idx in sorted(keep.items()):
        imp = importance[(l, n)].double()
        mask = torch.zeros_like(imp, dtype=torch.bool)
        mask[idx] = True
        per[f"{l}.{n}"] = dict(kept_mass=float(imp[mask].sum() / imp.sum()), min_kept=float(imp[mask].min()),
                               max_dropped=float(imp[~mask].max()) if (~mask).any() else None,
                               mean=float(imp.mean()), near_dead=int((imp < 1e-3 * imp.mean()).sum()))
    masses = [v["kept_mass"] for v in per.values()]
    return dict(per_ffn=per, kept_mass_mean=float(np.mean(masses)) if masses else 1.0,
                kept_mass_min=float(np.min(masses)) if masses else 1.0)


# ------------------------------------------------------------------------------------------------ build


_ENC = re.compile(r"model\.encoder\.layers\.(\d+)\.(.+)$")
_DEC = re.compile(r"model\.decoder\.layers\.(\d+)\.(.+)$")
_FFN = re.compile(r"(feed_forward[12])\.(linear[12])\.(weight|bias)$")


def _remap(teacher_sd: dict, student_keys: list[str], spec: StudentSpec, keep: dict) -> dict:
    """Student key -> the teacher tensor (or FFN slice) that initialises it. Unknown keys raise KeyError."""
    out = {}
    for k in student_keys:
        if m := _ENC.match(k):
            tl, rest = spec.enc_layers[int(m[1])], m[2]
            src = teacher_sd[f"model.encoder.layers.{tl}.{rest}"]
            if (ff := _FFN.match(rest)) and (tl, ff[1]) in keep:
                idx = keep[(tl, ff[1])].to(src.device)
                # linear1: weight rows + bias entries of the kept neurons; linear2: weight columns (its bias is d-sized)
                if ff[2] == "linear1":
                    src = src.index_select(0, idx)
                elif ff[3] == "weight":
                    src = src.index_select(1, idx)
            out[k] = src
        elif m := _DEC.match(k):
            out[k] = teacher_sd[f"model.decoder.layers.{spec.dec_layers[int(m[1])]}.{m[2]}"]
        elif k == "proj_out.weight" and spec.tie_head:
            out[k] = teacher_sd["model.decoder.embed_tokens.weight"]
        else:
            out[k] = teacher_sd[k]
    return out


@torch.no_grad()
def build_student(teacher: CohereAsrForConditionalGeneration, spec: StudentSpec, importance: dict | None,
                  keep: dict | None = None) -> CohereAsrForConditionalGeneration:
    """Fresh fp32 student on CPU, every tensor loaded (strict=True) from the teacher: layers renumbered, FFNs sliced
    to the top-`spec.ffn_dim` neurons by `importance` (sorted by original index), head tied to embed_tokens.
    `keep` = a ready FFN selection instead ({(teacher_layer, ffn_name): indices}, e.g. keep_from_meta of another
    student); `importance` is then ignored. BN running stats are the teacher's (recalibrate_batchnorm replaces them).
    The teacher may be on any device and in any dtype; it is not modified. Returned in eval mode."""
    cfg = student_config(teacher.config, spec)
    teacher_ffn = teacher.config.encoder_config.intermediate_size
    if keep is None:
        keep = select_ffn_neurons(importance, spec.enc_layers, spec.ffn_dim, teacher_ffn)
    else:
        keep = check_keep(keep, spec, teacher_ffn)
    student = CohereAsrForConditionalGeneration(cfg)
    if spec.tie_head and student.proj_out.weight is not student.model.decoder.embed_tokens.weight:
        student.tie_weights()
    sd = _remap(teacher.state_dict(), list(student.state_dict().keys()), spec, keep)
    student.load_state_dict(sd, strict=True)  # copies (and upcasts) into the student's own CPU fp32 storage
    if spec.tie_head:
        assert student.proj_out.weight is student.model.decoder.embed_tokens.weight, "head not tied"
    for j, layer in enumerate(student.model.decoder.layers):  # KV-cache slots; a stale index breaks generate()
        assert layer.self_attn.layer_idx == j and layer.encoder_attn.layer_idx == j
    gc = student.generation_config
    gc.decoder_start_token_id, gc.eos_token_id, gc.pad_token_id = DECODER_START, EOS, PAD
    gc.bos_token_id = teacher.generation_config.bos_token_id
    return student.eval()


# ------------------------------------------------------------------------------------------------ from scratch


@dataclass(frozen=True)
class ScratchShape:
    """A from-scratch student: the teacher's architecture (conformer encoder, transformer decoder, tied V=16384 head,
    128 mels, 8x subsampling with 256 channels) at these widths and depths. Head width = hidden / heads."""
    enc_hidden: int
    enc_layers: int
    enc_heads: int
    enc_ffn: int
    dec_hidden: int
    dec_layers: int
    dec_heads: int
    dec_ffn: int
    conv_kernel: int = 9

    def __post_init__(self):
        for name in ("enc_hidden", "enc_layers", "enc_heads", "enc_ffn", "dec_hidden", "dec_layers", "dec_heads",
                     "dec_ffn", "conv_kernel"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1: {self}")
        if self.enc_hidden % self.enc_heads or self.dec_hidden % self.dec_heads:
            raise ValueError(f"hidden sizes must be divisible by the head counts: {self}")
        if self.conv_kernel % 2 == 0:
            raise ValueError(f"the depthwise conv kernel must be odd ('same' padding): {self}")


# STUDY.md 1.1 (decision 18, shapes W). The bridge is the T-0.3B shape (B10x2560 + decoder {0,7}) from scratch: the
# teacher's widths, 10 encoder and 2 decoder layers.
SCRATCH_SHAPES = {
    "t01": ScratchShape(512, 12, 8, 2048, 512, 4, 8, 2048),
    "t005": ScratchShape(384, 10, 6, 1536, 384, 3, 6, 1536),
    "bridge": ScratchShape(1280, 10, 8, 2560, 1024, 2, 8, 4096),
}
SCRATCH_EXPECTED_PARAMS = {"t01": 103_996_416, "t005": 51_209_600, "bridge": 320_752_384}


def scratch_config(teacher_config: CohereAsrConfig, shape: ScratchShape) -> CohereAsrConfig:
    """student_config (NeMo extras dropped, tied head, decoder start, fp32, sdpa) with the shape's widths and depths.

    head_dim and num_key_value_heads are set explicitly in both stacks: the teacher config serialises head_dim 128 and
    8 kv heads, which a width change alone would silently keep. Dropout, layerdrop and scale_input are pinned to the
    teacher's 0 / False as well, so the config says what the study trains with."""
    t_enc = teacher_config.encoder_config
    # the layer choice is irrelevant here (no teacher tensor is copied); this spec only has to be valid for the teacher
    cfg = student_config(teacher_config, StudentSpec(enc_layers=[0], ffn_dim=t_enc.intermediate_size, dec_layers=[0]))
    e = cfg.encoder_config
    e.num_hidden_layers, e.hidden_size, e.intermediate_size = shape.enc_layers, shape.enc_hidden, shape.enc_ffn
    e.num_attention_heads = e.num_key_value_heads = shape.enc_heads
    e.conv_kernel_size = shape.conv_kernel
    e.scale_input = False
    for k in ("dropout", "dropout_positions", "layerdrop", "activation_dropout", "attention_dropout"):
        setattr(e, k, 0.0)
    if hasattr(e, "head_dim"):  # the encoder derives it from hidden/heads; a stale explicit value would win
        delattr(e, "head_dim")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = shape.dec_layers, shape.dec_hidden, shape.dec_ffn
    cfg.num_attention_heads = cfg.num_key_value_heads = shape.dec_heads
    cfg.head_dim = shape.dec_hidden // shape.dec_heads
    cfg.attention_dropout = 0.0
    return cfg


def sinusoid_pos_emb(n_pos: int, dim: int) -> torch.Tensor:
    """The teacher decoder's fixed position table (NeMo FixedPositionalEncoding): sin on even and cos on odd features
    of pos / 10000^(2i/dim), divided by sqrt(dim), computed in fp32 in NeMo's order. Equal to the checkpoint's
    pos_enc up to its bf16 rounding."""
    pos = torch.arange(0.0, n_pos, dtype=torch.float32)[:, None]
    div = torch.exp((-math.log(10000.0) / dim) * torch.arange(0.0, dim, 2, dtype=torch.float32))
    pe = torch.zeros(n_pos, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return pe.div_(math.sqrt(dim))


def _subsampling_convs(model: CohereAsrForConditionalGeneration) -> list[nn.Conv2d]:
    return [m for m in model.model.encoder.subsampling.modules() if isinstance(m, nn.Conv2d)]


@torch.no_grad()
def build_scratch_student(teacher_config: CohereAsrConfig, shape: ScratchShape, seed: int,
                          name: str | None = None) -> CohereAsrForConditionalGeneration:
    """A randomly initialised fp32 student on CPU in eval mode (STUDY.md 1.2, the from-scratch steps 1-9):

    1. config = scratch_config(teacher_config, shape): student_config first, then the widths; 2. head_dim = D/heads and
    kv heads = heads, explicitly; 3. a fresh model with HF's default init; 4. pos_emb = sinusoid/sqrt(D), the
    teacher's form; 5. the subsampling Conv2d layers re-initialised with the PyTorch default; 6. fresh BatchNorm;
    7. the teacher's generation ids (start 13764, EOS 3, PAD 2, BOS 4) and scale_input False; 8. every random draw
    from a torch RNG fork seeded with `seed` (the caller's RNG state is untouched, the build is reproducible); 9. the
    count is asserted: total == closed_form_params, and == SCRATCH_EXPECTED_PARAMS[name] for a named study shape."""
    cfg = scratch_config(teacher_config, shape)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        student = CohereAsrForConditionalGeneration(cfg)
        for conv in _subsampling_convs(student):
            conv.reset_parameters()  # kaiming-uniform(a=sqrt(5)) weight, U(+-1/sqrt(fan_in)) bias
    if student.proj_out.weight is not student.model.decoder.embed_tokens.weight:
        student.tie_weights()
    assert student.proj_out.weight is student.model.decoder.embed_tokens.weight, "head not tied"
    dec = student.model.decoder
    dec.pos_emb.weight.copy_(sinusoid_pos_emb(cfg.max_position_embeddings, cfg.hidden_size))
    for _, bn in _batchnorms(student):
        bn.reset_parameters()  # running mean 0, var 1, num_batches_tracked 0, weight 1, bias 0
    gc = student.generation_config
    gc.decoder_start_token_id, gc.eos_token_id, gc.pad_token_id, gc.bos_token_id = DECODER_START, EOS, PAD, BOS

    # the traps, checked on the built modules rather than on the config
    enc = student.model.encoder
    assert enc.input_scale == 1.0, "scale_input must be False (the teacher's)"
    for layer in enc.layers:
        a = layer.self_attn
        assert (a.head_dim, a.num_key_value_groups) == (shape.enc_hidden // shape.enc_heads, 1), (a.head_dim, shape)
    for layer in dec.layers:
        for a in (layer.self_attn, layer.encoder_attn):
            assert (a.head_dim, a.num_key_value_groups) == (shape.dec_hidden // shape.dec_heads, 1), (a.head_dim, shape)
    total = sum(p.numel() for p in student.parameters())
    assert total == closed_form_params(cfg), (total, closed_form_params(cfg))
    if name in SCRATCH_EXPECTED_PARAMS:
        assert total == SCRATCH_EXPECTED_PARAMS[name], (name, total, SCRATCH_EXPECTED_PARAMS[name])
    return student.eval()


# ------------------------------------------------------------------------------------------------ BatchNorm


def _batchnorms(model: nn.Module) -> list[tuple[str, _BN]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _BN)]


@torch.no_grad()
def recalibrate_batchnorm(student: CohereAsrForConditionalGeneration, feats_iter: Iterable, device,
                          max_utts: int = 1000) -> dict:
    """Re-estimate every BN's running mean/var from scratch on up to `max_utts` utterances, one at a time.

    feats_iter yields (feats (B,T,128), mask (B,T)) batches; each row is cut to its valid frames and run alone
    (batch size 1: no padded frame enters the statistics). Only the BN modules are in train mode, with momentum=None,
    so after n utterances the running stats are the plain average of n per-utterance statistics. Only the encoder
    runs. The model is left in eval mode with the BN momenta restored. Returns drift stats, measured against the
    stats the student had before (i.e. the teacher's)."""
    bns = _batchnorms(student)
    if not bns:
        raise ValueError("model has no BatchNorm modules")
    before = [(bn.running_mean.detach().float().clone(), bn.running_var.detach().float().clone()) for _, bn in bns]
    momenta = [bn.momentum for _, bn in bns]
    dtype = next(student.parameters()).dtype
    enc = student.model.encoder
    student.eval()
    unfreeze_batchnorm(student)  # a BN frozen by kitsune.patches ignores .train(): recalibration would do nothing
    for _, bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    n_utts, n_frames = 0, 0
    try:
        for item in feats_iter:
            feats, mask = item
            lengths = (mask.sum(dim=1) if mask is not None else torch.full((feats.shape[0],), feats.shape[1])).tolist()
            for i, L in enumerate(lengths):
                if n_utts >= max_utts:
                    break
                L = int(L)
                if L < 1:
                    continue
                x = feats[i:i + 1, :L].to(device=device, dtype=dtype)
                enc(input_features=x, attention_mask=torch.ones(1, L, dtype=torch.int64, device=device))
                n_utts += 1
                n_frames += L
            if n_utts >= max_utts:
                break
    finally:
        for (_, bn), mom in zip(bns, momenta):
            bn.momentum = mom
            bn.eval()
        student.eval()
    if n_utts == 0:
        raise ValueError("recalibrate_batchnorm: feats_iter gave no utterances; the running stats are now reset")

    layers = []
    for (name, bn), (m0, v0) in zip(bns, before):
        m1, v1 = bn.running_mean.float(), bn.running_var.float()
        shift = (m1 - m0).abs() / torch.sqrt(v0 + bn.eps)  # mean shift in units of the old std
        lvr = torch.log((v1 + bn.eps) / (v0 + bn.eps)).abs()
        layers.append(dict(module=name, mean_shift_std=float(shift.mean()), max_shift_std=float(shift.max()),
                           abs_log_var_ratio=float(lvr.mean()), max_abs_log_var_ratio=float(lvr.max())))
    return dict(n_utts=n_utts, n_mel_frames=n_frames, layers=layers,
                mean_shift_std=float(np.mean([l["mean_shift_std"] for l in layers])),
                abs_log_var_ratio=float(np.mean([l["abs_log_var_ratio"] for l in layers])))


# ------------------------------------------------------------------------------------------------ save / load


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, torch.dtype, torch.device)):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dump_json(obj) -> str:
    """Indented JSON, but lists of plain numbers stay on one line (kept FFN indices would be 100k lines otherwise)."""
    s = json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default)
    # only multi-line matches are rewritten: a raw newline never occurs inside a JSON string, so text is untouched
    num = r"-?[\d.eE+-]+"
    return re.sub(rf"\[\n\s*({num}(?:,\n\s*{num})*)\n\s*\]", lambda m: "[" + re.sub(r",\n\s*", ",", m[1]) + "]", s)


def _save_state_dict(model: nn.Module) -> dict:
    """bf16 copy of the state_dict on CPU, except BN running stats (fp32) and integer buffers (as is). Tensors that
    share storage (the tied head) map to one converted tensor, so save_pretrained still sees them as tied."""
    bn_stats = {f"{n}.{b}" for n, _ in _batchnorms(model) for b in ("running_mean", "running_var")}
    memo, out = {}, {}
    for k, v in model.state_dict().items():
        key = (v.data_ptr(), v.dtype, tuple(v.shape), str(v.device))
        if key not in memo:
            if v.is_floating_point():
                memo[key] = v.detach().to("cpu", torch.float32 if k in bn_stats else torch.bfloat16)
            else:
                memo[key] = v.detach().to("cpu")
        out[k] = memo[key]
    return out


def write_meta(out_dir, meta: dict):
    p = Path(out_dir) / "student_meta.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(dump_json(meta), encoding="utf-8")
    tmp.replace(p)


# The part of MODEL_CARD.md's Apache-2.0 modification notice that says what was done to the teacher. It describes the
# first run's student; model_card rewrites it for any other student from its student_meta.json.
_CARD_CHANGES = re.compile(r"It was pruned \(.*?`student_meta\.json` records the exact layers and FFN neurons kept\.",
                           re.S)


def model_card(meta: dict) -> str:
    """MODEL_CARD.md with a modification notice that describes this student (meta = its student_meta.json).

    The card's notice describes the first run's student (20 of 48 layers, FFN 2560, 4 decoder layers, BN
    recalibrated), and a verbatim copy would misstate every other build. A meta with the size study's fields
    (init_class) gets its own sentence: the pruned shape and its BN mode, or, for a from-scratch student, that it has
    the teacher's architecture at another size with randomly initialised weights. A meta without them (the first
    run's format 1, tests) keeps the card as it is. A card whose notice no longer has the expected wording is kept as
    it is, with a warning (save_student must not fail a training run's checkpoint over the card)."""
    card = MODEL_CARD.read_bytes().decode("utf-8")
    init = meta.get("init_class") if meta else None
    if not init:
        return card
    head = (meta.get("build") or {}).get("tie_head", True)
    if init == "scratch":
        s = meta.get("scratch") or {}
        text = (f"It has that model's architecture at another size (encoder: {s.get('enc_layers')} layers, width "
                f"{s.get('enc_hidden')}, FFN {s.get('enc_ffn')}; decoder: {s.get('dec_layers')} layers, width "
                f"{s.get('dec_hidden')}, FFN {s.get('dec_ffn')}; output head tied to the token embedding), but its "
                f"weights were randomly initialised, not copied from that model; it was then trained by distillation "
                f"from the teacher's outputs. The configuration is Cohere's with these sizes; the tokenizer and "
                f"processor files are Cohere's, unmodified. `student_meta.json` records the exact shape and the seed.")
    else:
        ffn = meta.get("ffn")
        ffn_text = f"FFN {TEACHER_FFN} -> {ffn}" if ffn is not None and ffn < TEACHER_FFN else f"FFN {TEACHER_FFN}"
        changes = (["its output head tied to the token embedding"] if head else []) + (
            ["its BatchNorm statistics recalibrated"] if meta.get("bn") == "recal" else [])
        text = (f"It was pruned (encoder: {len(meta.get('enc_layers') or [])} of {TEACHER_ENC_LAYERS} layers, "
                f"{ffn_text}; decoder: {len(meta.get('dec_layers') or [])} of {TEACHER_DEC_LAYERS} layers)"
                + (", " + " and ".join(changes) if changes else "")
                + ", then trained by distillation from the teacher's outputs."
                + (" Its BatchNorm statistics are the teacher's." if meta.get("bn") == "teacher" else "")
                + " The tokenizer and processor files are Cohere's, unmodified. `student_meta.json` records the exact"
                  " layers and FFN neurons kept.")
    new, n = _CARD_CHANGES.subn(lambda _: text, card, count=1)
    if n != 1:
        warnings.warn(f"{MODEL_CARD}: the modification notice's wording changed; README.md is copied unchanged and may "
                      f"not describe this student")
        return card
    return new


def save_student(student: CohereAsrForConditionalGeneration, out_dir, processor, meta: dict):
    """save_pretrained (safetensors, HF key names) with bf16 weights and fp32 BN stats, the processor (feature
    extractor + tokenizer files), a generation_config with the teacher's special ids, the model card as README.md
    (its modification notice describing this student, model_card), and student_meta.json. `processor` may be None
    (tests). student_meta.json is written last."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gc = student.generation_config
    gc.decoder_start_token_id, gc.eos_token_id, gc.pad_token_id = DECODER_START, EOS, PAD
    old_dtype = student.config.dtype
    try:
        # save_original_format=False: keep HF names; the default would re-apply the NeMo names of the teacher repo
        student.save_pretrained(out, state_dict=_save_state_dict(student), save_original_format=False)
    finally:
        student.config.dtype = old_dtype  # save_pretrained overwrites it with the in-memory dtype
    # config.json should describe the file (bf16), which is what from_pretrained(dtype="auto") then gives
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["dtype"] = "bfloat16"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if processor is not None:
        processor.save_pretrained(out)
    # the model card (licence terms, the Apache-2.0 modified-from notice, data credits) travels with every copy of the
    # weights: the student init, each step_<N>/ and any checkpoint later promoted or copied out of the run repo
    if MODEL_CARD.exists():
        (out / "README.md").write_bytes(model_card(meta).encode("utf-8"))
    write_meta(out, meta)


def load_student(path_or_repo, device, dtype=torch.float32) -> CohereAsrForConditionalGeneration:
    """Load a saved student with sdpa attention in eval mode. It does not apply any kitsune.patches: the trainer calls
    patch_relpos_once_per_batch and train_mode (which freezes BN) itself."""
    model = CohereAsrForConditionalGeneration.from_pretrained(str(path_or_repo), dtype=dtype, attn_implementation="sdpa")
    for cfg in (model.config, model.model.encoder.config, model.model.decoder.config):
        if cfg._attn_implementation != "sdpa":
            cfg._attn_implementation = "sdpa"
    if model.config.tie_word_embeddings:
        assert model.proj_out.weight is model.model.decoder.embed_tokens.weight, "tied head came back untied"
    return model.to(device).eval()


def load_meta(path) -> dict:
    p = Path(path) / "student_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def importance_to_state(importance: dict) -> dict:
    """{(layer, name): t} -> {'layer.name': t}: plain string keys for torch.save(weights_only) round trips."""
    return {f"{l}.{n}": t.detach().float().cpu() for (l, n), t in importance.items()}


def importance_from_state(state: dict) -> dict:
    return {(int(k.split(".", 1)[0]), k.split(".", 1)[1]): t for k, t in state.items()}
