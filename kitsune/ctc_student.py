"""Parakeet-family students of the size study: a pruned copy of Parakeet's CTC path, initialised from its weights.

The teacher is nvidia/parakeet-tdt_ctc-0.6b-ja as converted for the label box (kitsune/parakeet.py pins the files):
a 24-layer FastConformer encoder (d 1024, 8 heads, FFN 4096, 8x subsampling of 80 log-mels) with two heads. Only the
CTC head is kept: the students are transformers `ParakeetForCTC` models (vocab 3073, blank = pad = 3072), so the TDT
prediction net and joint are never loaded. The unpruned build of this path is "the anchor"; it reproduces the label
box's CTC targets.

Why each step is done the way it is:
- The converted dir holds a ParakeetForTDT (encoder + TDT decoder/joint) plus `ctc_head.safetensors`, the NeMo CTC
  head (a k=1 Conv1d) stored as a Linear (V, H). HF's ParakeetForCTC keeps it as a Conv1d, so the weight gets
  `.unsqueeze(-1)`. The encoder keys are the same in both classes (`encoder.*`), and the student is a FRESH model
  loaded with strict=True: every tensor provably comes from the teacher, and the layers are renumbered 0..L-1.
- Encoder layers are kept evenly spaced (kitsune.student.evenly_spaced). Layer 23, the last, is always kept: the CTC
  head was trained on its output, and without it the head reads a representation it has never seen.
- FFN neurons are ranked by kitsune.student.ffn_importance, unchanged: mean |silu(linear1 x)| over the valid encoder
  frames, measured inside the full teacher. A 3-line view exposes the encoder as `.model.encoder`, which is where that
  function looks. Kept neurons are sorted by original index (select_ffn_neurons).
- BatchNorm keeps the teacher's running stats (decision 16: best step-0 KL at P-0.3B; recalibration was worse).
- Dropout, attention/activation dropout and layerdrop are set to 0 explicitly: the converted config carries HF's
  default 0.1, but NeMo trained this model with stochastic depth 0, and the study runs dropout 0 everywhere.
- Attention is sdpa (like the Cohere students): the padded query rows are fully masked, and sdpa returns 0 there
  (checked: no NaN, and a padded row equals the same utterance alone to ~1e-6).
- Saved like the Transcribe students: bf16 weights, fp32 BN running stats. The label box ran this encoder in bf16, so
  the bf16 rounding moves the student towards the model that made its targets, not away from it.

Features (`ctc_features`): the ParakeetFeatureExtractor path, computed by kitsune.features.LogMel with Parakeet's own
filterbank - 80 mel, pre-emphasis 0.97, per-utterance per-bin normalisation and NO dither (the extractor has none).
On CPU it equals the HF extractor bitwise. The student's frame count must equal the stored `n_frames` of its targets,
so nothing that changes the length (speed/tempo change, time warping, cropping) may touch the audio.

Decoding: `greedy_ctc_ids` is the explicit greedy CTC path (argmax per frame, collapse repeats, drop blanks), equal to
kitsune.parakeet.ctc_collapse. `decode_ids` then detokenises ALREADY collapsed ids with `group_tokens=False`: the
ParakeetTokenizer's own default (group_tokens=True) would merge two legitimately repeated tokens, which is not what
the label pass did when it wrote `ctc_hyp`.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from kitsune.features import LogMel, feature_lengths, pad_waves
from kitsune.student import evenly_spaced, ffn_importance, select_ffn_neurons, write_meta

CTC_VOCAB = 3073
CTC_BLANK = 3072
FRAME_S = 0.08
SUBSAMPLING = 8
TEACHER_LAYERS = 24
TEACHER_FFN = 4096
TEACHER_REPO = "nvidia/parakeet-tdt_ctc-0.6b-ja"
TEACHER_REVISION = "44edb27eea9317daf89333e75eb830db4b1cc298"  # == kitsune.parakeet.NEMO_REVISION (asserted in tests)
CTC_HEAD_FILE = "ctc_head.safetensors"
PROCESSOR_FILES = ("processor_config.json", "tokenizer.json", "tokenizer_config.json")
CARD_FILE = "MODEL_CARD.md"
DROPOUT_KEYS = ("dropout", "dropout_positions", "attention_dropout", "activation_dropout", "layerdrop")

_BN = nn.modules.batchnorm._BatchNorm
_FFN = re.compile(r"(feed_forward[12])\.(linear[12])\.(weight|bias)$")
_LAYER = re.compile(r"encoder\.layers\.(\d+)\.(.+)$")


# ------------------------------------------------------------------------------------------------ counts


def closed_form_ctc_params(n_layers: int, ffn: int, *, d: int = 1024, vocab: int = CTC_VOCAB, n_mels: int = 80,
                           channels: int = 256, kernel: int = 9, sub_kernel: int = 3, sub_factor: int = 8) -> int:
    """Parameters of a ParakeetForCTC student from its shape alone. For the real dims (the keyword defaults) it
    reduces to 5,911,553 + L * (8,422,400 + 4,098 * F):
      per layer  2 FFNs 4dF + 2F + 2d; attention (q,k,v,o with bias, relative_k_proj, bias_u/v) 5d^2 + 6d;
                 conv module (pointwise 2d + d, depthwise k, BN affine) 3d^2 + (k + 6)d; 5 LayerNorms 10d
      fixed      subsampling 2,761,728 (conv 1->C, (n_sub - 1) x (depthwise + pointwise), linear C*mels/8 -> d)
                 + the CTC head V*d + V = 3,149,825
    The relative positional encoding is a non-persistent buffer, not a parameter."""
    layer = 4 * d * ffn + 2 * ffn + 8 * d * d + (kernel + 24) * d
    k2, n_sub = sub_kernel * sub_kernel, int(math.log2(sub_factor))
    sub = (k2 * channels + channels) + (n_sub - 1) * ((k2 * channels + channels) + (channels * channels + channels))
    sub += channels * (n_mels // 2 ** n_sub) * d + d
    return sub + vocab * d + vocab + n_layers * layer


def config_closed_form(cfg) -> int:
    """closed_form_ctc_params from a ParakeetCTCConfig's own fields (the tiny test models have other dims)."""
    e = cfg.encoder_config
    assert e.attention_bias and e.convolution_bias and e.subsampling_conv_stride == 2, "closed form assumes these"
    assert e.num_attention_heads * (e.hidden_size // e.num_attention_heads) == e.hidden_size, e
    return closed_form_ctc_params(e.num_hidden_layers, e.intermediate_size, d=e.hidden_size, vocab=cfg.vocab_size,
                                  n_mels=e.num_mel_bins, channels=e.subsampling_conv_channels,
                                  kernel=e.conv_kernel_size, sub_kernel=e.subsampling_conv_kernel_size,
                                  sub_factor=e.subsampling_factor)


def param_counts(model) -> dict:
    """total, non-embedding (total minus the CTC head, STUDY 1.1), the head, subsampling, one layer, closed form."""
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.ctc_head.parameters())
    return dict(total=total, non_embedding=total - head, ctc_head=head,
                subsampling=sum(p.numel() for p in model.encoder.subsampling.parameters()),
                encoder_layers=len(model.encoder.layers),
                encoder_layer=sum(p.numel() for p in model.encoder.layers[0].parameters()),
                closed_form=config_closed_form(model.config))


# ------------------------------------------------------------------------------------------------ config + loading


def ctc_config(encoder_config, n_layers: int, ffn: int, *, vocab: int = CTC_VOCAB, blank: int = CTC_BLANK):
    """ParakeetCTCConfig for a student: the teacher's encoder config with the student's depth/width, every dropout 0,
    fp32, sdpa. vocab/blank/pad are Parakeet's (the blank is the tokenizer's pad token)."""
    from transformers import ParakeetCTCConfig

    enc = encoder_config.to_dict() if hasattr(encoder_config, "to_dict") else dict(encoder_config)
    for key in ("_name_or_path", "architectures", "transformers_version", "dtype", "torch_dtype"):
        enc.pop(key, None)
    enc.update(num_hidden_layers=int(n_layers), intermediate_size=int(ffn), **{k: 0.0 for k in DROPOUT_KEYS})
    # the encoder config's __post_init__ derives kv heads from heads only when the key is absent
    enc["num_key_value_heads"] = enc["num_attention_heads"]
    cfg = ParakeetCTCConfig(encoder_config=enc, vocab_size=vocab, pad_token_id=blank,
                            ctc_loss_reduction="sum", ctc_zero_infinity=True)
    cfg.dtype = torch.float32
    cfg.encoder_config.dtype = torch.float32
    cfg._attn_implementation = "sdpa"
    cfg.encoder_config._attn_implementation = "sdpa"
    return cfg


def _check_vocab(cfg) -> None:
    if cfg.vocab_size != CTC_VOCAB or cfg.pad_token_id != CTC_BLANK:
        raise ValueError(f"not Parakeet's CTC vocab: vocab {cfg.vocab_size}, pad/blank {cfg.pad_token_id} "
                         f"(expected {CTC_VOCAB}, {CTC_BLANK})")


def load_parakeet_ctc(model_dir, device="cpu", dtype=torch.float32):
    """The unpruned anchor (24 x 4096): ParakeetForCTC from the converted dir's encoder weights and
    ctc_head.safetensors, strict, in eval mode. Local files only; the TDT decoder/joint tensors are never read."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, ParakeetForCTC

    d = Path(model_dir)
    tcfg = AutoConfig.from_pretrained(d, local_files_only=True)
    blank = getattr(tcfg, "blank_token_id", getattr(tcfg, "pad_token_id", None))
    if tcfg.vocab_size != CTC_VOCAB or blank != CTC_BLANK:
        raise ValueError(f"{d}: vocab {tcfg.vocab_size} / blank {blank}, expected {CTC_VOCAB} / {CTC_BLANK}")
    e = tcfg.encoder_config
    cfg = ctc_config(e, e.num_hidden_layers, e.intermediate_size)
    model = ParakeetForCTC(cfg)
    sd = {}
    with safe_open(str(d / "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("encoder."):
                sd[k] = f.get_tensor(k)
    head = load_file(str(d / CTC_HEAD_FILE))
    sd["ctc_head.weight"] = head["weight"].reshape(head["weight"].shape[0], -1).unsqueeze(-1)  # Linear -> Conv1d k=1
    sd["ctc_head.bias"] = head["bias"]
    model.load_state_dict(sd, strict=True)
    return model.to(device=device, dtype=dtype).eval()


# ------------------------------------------------------------------------------------------------ FFN importance


class _EncoderView(nn.Module):
    """kitsune.student.ffn_importance reads `<model>.model.encoder`; ParakeetForCTC keeps its encoder at `.encoder`."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.model = nn.Module()
        self.model.encoder = encoder
        self.train(encoder.training)  # ffn_importance restores THIS flag on the encoder when it is done


def ffn_importance_ctc(teacher, feats_iter: Iterable, layers: Sequence[int]) -> dict:
    """{(teacher_layer, 'feed_forward1'|'feed_forward2'): (4096,) float32} - kitsune.student.ffn_importance on the
    Parakeet encoder, on the teacher's own device. feats_iter yields (feats (B,T,80), mask (B,T)) as ctc_features /
    LogMel produce them."""
    device = next(teacher.parameters()).device
    return ffn_importance(_EncoderView(teacher.encoder), feats_iter, list(layers), device)


# ------------------------------------------------------------------------------------------------ build


def resolve_layers(spec, n_teacher: int = TEACHER_LAYERS) -> list[int]:
    """'all' | an int N (evenly spaced over the teacher's layers) | explicit ascending indices -> kept layer list."""
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, int):
        spec = [spec]
    spec = [str(x) for x in spec]
    if spec == ["all"]:
        return list(range(n_teacher))
    if len(spec) == 1:
        return evenly_spaced(int(spec[0]), n_teacher)
    return [int(x) for x in spec]


def _remap(teacher_sd: dict, student_keys: list[str], enc_layers: list[int], keep: dict) -> dict:
    """Student key -> the teacher tensor (or FFN slice) that initialises it. Unknown keys raise KeyError."""
    out = {}
    for k in student_keys:
        if m := _LAYER.match(k):
            tl, rest = enc_layers[int(m[1])], m[2]
            src = teacher_sd[f"encoder.layers.{tl}.{rest}"]
            if (ff := _FFN.match(rest)) and (tl, ff[1]) in keep:
                idx = keep[(tl, ff[1])].to(src.device)
                # linear1: weight rows + bias entries of the kept neurons; linear2: weight columns (its bias is d-sized)
                if ff[2] == "linear1":
                    src = src.index_select(0, idx)
                elif ff[3] == "weight":
                    src = src.index_select(1, idx)
            out[k] = src
        else:
            out[k] = teacher_sd[k]
    return out


@torch.no_grad()
def build_ctc_student(teacher, enc_layers: Sequence[int], ffn: int, importance: dict | None):
    """Fresh fp32 ParakeetForCTC on CPU, every tensor loaded (strict=True) from the teacher ParakeetForCTC: layers
    `enc_layers` (teacher indices, ascending; the last teacher layer must be among them), FFNs cut to the `ffn` most
    important neurons (sorted by original index), the CTC head and the teacher's BN stats verbatim. The teacher may
    be on any device and in any dtype; it is not modified. Returned in eval mode."""
    from transformers import ParakeetForCTC

    _check_vocab(teacher.config)
    t_enc = teacher.config.encoder_config
    n_t, t_ffn = t_enc.num_hidden_layers, t_enc.intermediate_size
    layers = [int(x) for x in enc_layers]
    if not layers or layers != sorted(set(layers)) or layers[0] < 0 or layers[-1] >= n_t:
        raise ValueError(f"enc_layers must be ascending, unique and inside [0, {n_t}): {layers}")
    if layers[-1] != n_t - 1:
        raise ValueError(f"enc_layers must keep the last teacher layer {n_t - 1}: the CTC head reads its output")
    if not 1 <= int(ffn) <= t_ffn:
        raise ValueError(f"ffn {ffn} not in [1, {t_ffn}]")
    keep = select_ffn_neurons(importance, layers, int(ffn), t_ffn)
    cfg = ctc_config(t_enc, len(layers), int(ffn), vocab=teacher.config.vocab_size, blank=teacher.config.pad_token_id)
    student = ParakeetForCTC(cfg)
    sd = _remap(teacher.state_dict(), list(student.state_dict().keys()), layers, keep)
    student.load_state_dict(sd, strict=True)  # copies (and upcasts) into the student's own CPU fp32 storage
    return student.eval()


def kept_neurons(importance: dict | None, enc_layers: Sequence[int], ffn: int, teacher_ffn: int = TEACHER_FFN) -> dict:
    """{'<teacher layer>.<ffn name>': kept indices} for student_meta.json ({} when nothing is pruned)."""
    keep = select_ffn_neurons(importance, list(enc_layers), int(ffn), teacher_ffn)
    return {f"{l}.{n}": idx for (l, n), idx in sorted(keep.items())}


# ------------------------------------------------------------------------------------------------ save / load


def _save_state_dict(model: nn.Module) -> dict:
    """bf16 copy of the state_dict on CPU, except BN running stats (fp32) and integer buffers (as is)."""
    bn_stats = {f"{n}.{b}" for n, m in model.named_modules() if isinstance(m, _BN)
                for b in ("running_mean", "running_var")}
    out = {}
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            out[k] = v.detach().to("cpu", torch.float32 if k in bn_stats else torch.bfloat16).contiguous()
        else:
            out[k] = v.detach().to("cpu")
    return out


def model_card(meta: dict) -> str:
    """The CC-BY-4.0 attribution card of a Parakeet-derived student (MODEL_CARD.md, also written as README.md)."""
    layers, ffn = meta.get("enc_layers"), meta.get("ffn")
    teacher = meta.get("teacher", f"{TEACHER_REPO}@{TEACHER_REVISION}")
    repo, _, rev = teacher.partition("@")
    n = meta.get("params_total")
    shape = f"{len(layers)} of 24 encoder layers {layers}, FFN 4096 -> {ffn}" if layers else "the unpruned encoder"
    return f"""---
license: cc-by-4.0
base_model: {repo}
language:
- ja
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- ctc
- fastconformer
- parakeet
---

# Kitsune size study: Parakeet CTC student ({meta.get("name", "init")})

A pruned copy of the CTC path of NVIDIA's Parakeet TDT-CTC 0.6B Japanese model, made by
[Kitsune-Transcribe](https://github.com/Multysquid/Kitsune-Transcribe) as the initialisation of a size-study student
(transformers `ParakeetForCTC`, {n:,} parameters). It has not been trained by Kitsune yet unless a training run's files
say so.

## Attribution (CC BY 4.0)

This model is derived from **parakeet-tdt_ctc-0.6b-ja** by **NVIDIA**
([{repo}](https://huggingface.co/{repo}), revision `{rev}`), licensed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
NVIDIA does not endorse this derivative.

### Changes made

- converted to transformers `ParakeetForCTC`: the TDT prediction network and joint are dropped; the CTC head is
  NVIDIA's, unchanged (a k=1 convolution);
- pruned: {shape} (FFN neurons ranked by mean activation on 1,000 training utterances; the exact layers and neurons are
  in `student_meta.json`); the BatchNorm running statistics are the teacher's;
- dropout, attention/activation dropout and layer drop set to 0;
- weights stored in bfloat16 (BatchNorm statistics in float32);
- the tokenizer files are NVIDIA's, unchanged; in `processor_config.json` only `decoder_type` is set to `ctc`, so
  `processor.batch_decode(model.generate(...))` merges repeated frames as CTC decoding requires.

## Use

Input: 16 kHz mono audio through the saved `ParakeetFeatureExtractor` (80 log-mels, no dither). Greedy CTC decoding:
argmax per 80 ms frame, collapse repeats, drop the blank (id {CTC_BLANK}).

## Training data

Checkpoints trained from this initialisation by Kitsune additionally carry the terms of their training data, which
include non-commercial and share-alike conditions (Galgame_Speech_ASR, Emilia's NC part); see the Kitsune-Transcribe
`MODEL_CARD.md` and the training run's `config.json`.
"""


def save_ctc_student(model, out_dir, src_model_dir, meta: dict) -> None:
    """save_pretrained (safetensors, HF key names) with bf16 weights and fp32 BN stats; config.json dtype bfloat16;
    the Parakeet processor/tokenizer files copied from `src_model_dir` (processor_config.json with decoder_type
    "ctc"); the CC-BY-4.0 card as MODEL_CARD.md and README.md; student_meta.json last."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _check_vocab(model.config)
    old_dtype = model.config.dtype
    try:
        # no conversion mapping is registered for parakeet; False keeps the HF names whatever a later version does
        model.save_pretrained(out, state_dict=_save_state_dict(model), save_original_format=False)
    finally:
        model.config.dtype = old_dtype  # save_pretrained overwrites it with the in-memory dtype
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["dtype"] = "bfloat16"  # describes the file, which is what from_pretrained(dtype="auto") gives
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    src = Path(src_model_dir)
    for name in PROCESSOR_FILES:
        shutil.copyfile(src / name, out / name)
    proc_path = out / "processor_config.json"
    proc = json.loads(proc_path.read_text(encoding="utf-8"))
    proc["decoder_type"] = "ctc"
    proc_path.write_text(json.dumps(proc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card = model_card(meta)
    (out / CARD_FILE).write_text(card, encoding="utf-8")
    (out / "README.md").write_text(card, encoding="utf-8")
    write_meta(out, meta)


def load_ctc_student(path, device, dtype=torch.float32):
    """A saved student (or any ParakeetForCTC dir) with sdpa attention, in eval mode."""
    from transformers import ParakeetForCTC

    model = ParakeetForCTC.from_pretrained(str(path), dtype=dtype, attn_implementation="sdpa")
    for cfg in (model.config, model.encoder.config):
        if cfg._attn_implementation != "sdpa":
            cfg._attn_implementation = "sdpa"
    _check_vocab(model.config)
    return model.to(device).eval()


def load_meta(path) -> dict:
    p = Path(path) / "student_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------------------------------------ frames + features


def frame_lengths(feat_lens):
    """Encoder frames from valid mel frames, exactly as the encoder's _get_subsampling_output_length: three stride-2
    convolutions (kernel 3, padding 1), each L -> floor((L - 1) / 2) + 1. Works on ints and integer tensors."""
    n = feat_lens
    for _ in range(int(math.log2(SUBSAMPLING))):
        n = (n - 1) // 2 + 1
    return n


def expected_n_frames(n_samples: int) -> int:
    """Encoder frames of a 16 kHz waveform of `n_samples` samples: the ParakeetFeatureExtractor's valid mel frames
    (samples // 160), 8x subsampled as the encoder does (== ceil(frames / 8)). The frame preflight compares it with
    the stored n_frames."""
    mel = int(feature_lengths(torch.tensor([int(n_samples)]))[0])
    return int(frame_lengths(max(mel, 0)))


def lengths_to_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


class CtcFeatures:
    """callable(list of 1-D 16 kHz float32 arrays) -> (feats (B,T,80) float32, feat_lens (B,) int64) on `device`.
    `.logmel` is the underlying kitsune.features.LogMel (for callers that already hold padded device tensors)."""

    def __init__(self, logmel: LogMel, device="cpu"):
        self.device = torch.device(device)
        self.logmel = logmel.to(self.device)

    def to(self, device) -> "CtcFeatures":
        self.device = torch.device(device)
        self.logmel = self.logmel.to(self.device)
        return self

    def __call__(self, waves: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        wave, lengths = pad_waves(waves)
        feats, mask = self.logmel(wave.to(self.device), lengths.to(self.device))
        return feats, mask.sum(dim=1)


def ctc_features(model_dir, device="cpu") -> CtcFeatures:
    """The student's feature path from a Parakeet dir (the teacher's converted dir or a saved student): LogMel with
    the saved ParakeetFeatureExtractor's filterbank and settings, dither 0 (the extractor has no dither)."""
    from transformers import AutoFeatureExtractor

    fe = AutoFeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    if fe.feature_size != 80 or getattr(fe, "dither", 0.0):
        raise ValueError(f"{model_dir}: not Parakeet's features ({type(fe).__name__}, {fe.feature_size} mel)")
    return CtcFeatures(LogMel.from_feature_extractor(fe), device)


# ------------------------------------------------------------------------------------------------ forward + decode


def ctc_log_probs(model, feats: torch.Tensor, feat_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log_probs (B,T,V) float32, n_frames (B,) int64): the encoder, then the CTC head in fp32 outside any autocast
    (the label box's CTC head ran in fp32 too), log-softmax as z - logsumexp(z) (the CPU log_softmax kernel was
    measured 1e-4 off on a 16k vocab, see kitsune/kd.py)."""
    enc = model.encoder(input_features=feats, attention_mask=feat_mask.long(), output_attention_mask=True)
    h = enc.last_hidden_state
    with torch.autocast(device_type=h.device.type, enabled=False):
        w = model.ctc_head.weight.float()
        b = model.ctc_head.bias.float() if model.ctc_head.bias is not None else None
        z = torch.nn.functional.conv1d(h.float().transpose(1, 2), w, b).transpose(1, 2)
        lp = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return lp, enc.attention_mask.sum(-1).long()


def greedy_ctc_ids(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int = CTC_BLANK) -> list[list[int]]:
    """Greedy CTC per row: argmax per valid frame, collapse repeats, drop blanks (== kitsune.parakeet.ctc_collapse).
    One device->host copy of the (B,T) argmax."""
    arg = log_probs.argmax(dim=-1).cpu()
    prev = torch.cat([torch.full_like(arg[:, :1], -1), arg[:, :-1]], dim=1)
    keep = (arg != prev) & (arg != blank)
    out = []
    for b, n in enumerate(lengths.tolist()):
        n = int(n)
        out.append(arg[b, :n][keep[b, :n]].tolist())
    return out


def decode_ids(tokenizer, ids) -> str:
    """Text of one ALREADY collapsed id sequence, as the label pass wrote ctc_hyp: skip special tokens, no grouping
    (the ParakeetTokenizer groups repeats by default; a processor may too). Accepts a tokenizer or a processor."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return tok.decode([int(x) for x in ids], skip_special_tokens=True, group_tokens=False)


def decode_batch(tokenizer, batch) -> list[str]:
    return [decode_ids(tokenizer, ids) for ids in batch]
