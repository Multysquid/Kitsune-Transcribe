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
- BatchNorm (one per conv module) is recalibrated after pruning. The teacher's running stats describe inputs that
  28 dropped layers and half of every FFN used to shape. Recalibration uses batch size 1, so padded frames never
  enter the statistics, and momentum=None, which gives a cumulative average rather than an EMA biased to the last
  batches. BN then stays frozen in eval mode for all of training (kitsune.patches.freeze_batchnorm).
- The checkpoint is saved with bf16 weights but fp32 BN running stats. Those stats are never trained again, and bf16
  would quantise the variances that every normalised frame is divided by.

The decoder prompt, EOS, PAD and vocab are the teacher's (decoder_start_token_id 13764, see teacher_out/meta.json).
"""
import copy
import json
import math
import re
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
DECODER_START, EOS, PAD = 13764, 3, 2
FFN_NAMES = ("feed_forward1", "feed_forward2")
EXPECTED_DEFAULT_PARAMS = 616_963_328  # closed form for B20x2560 / dec4 / V16384 tied (report_model.md section 2)

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
def build_student(teacher: CohereAsrForConditionalGeneration, spec: StudentSpec,
                  importance: dict | None) -> CohereAsrForConditionalGeneration:
    """Fresh fp32 student on CPU, every tensor loaded (strict=True) from the teacher: layers renumbered, FFNs sliced
    to the top-`spec.ffn_dim` neurons by `importance` (sorted by original index), head tied to embed_tokens.
    The teacher may be on any device and in any dtype; it is not modified. Returned in eval mode."""
    cfg = student_config(teacher.config, spec)
    keep = select_ffn_neurons(importance, spec.enc_layers, spec.ffn_dim, teacher.config.encoder_config.intermediate_size)
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


def save_student(student: CohereAsrForConditionalGeneration, out_dir, processor, meta: dict):
    """save_pretrained (safetensors, HF key names) with bf16 weights and fp32 BN stats, the processor (feature
    extractor + tokenizer files), a generation_config with the teacher's special ids, and student_meta.json.
    `processor` may be None (tests). student_meta.json is written last."""
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
