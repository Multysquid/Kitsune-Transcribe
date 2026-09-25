"""Parakeet TDT+CTC teacher: pinned model files, loader and a guarded batched greedy TDT decoder.

The model is the laptop's converted HF dir of nvidia/parakeet-tdt_ctc-0.6b-ja (tools/publish_parakeet.py), plus
`ctc_head.safetensors` (the NeMo CTC head, a k=1 Conv1d stored as a Linear) and `kitsune_model.json`. The sha256 of
every file is pinned in PARAKEET_FILES; `verify_model_dir` checks them.

The TDT loop follows transformers' ParakeetForTDT `generate` (generation_parakeet.py, 5.13.1):
- the joint input is `encoder_projector(last_hidden_state)` (HF's pooler_output) at the current frame plus the
  prediction net output, through `config.hidden_act` (relu), then `joint.head` -> 3073 token + 5 duration logits;
- token = argmax of the token logits, duration = DURATIONS[argmax of the duration logits]; a blank with duration 0
  advances 1 frame;
- the prediction net starts from the blank token (decoder_start_token_id) with a zero LSTM state, and is only
  advanced for rows that emit a non-blank token (HF's masked cache update);
- a row stops once its frame pointer reaches its valid encoder length.
On top of that, NeMo's max-symbols guard: after `max_symbols` consecutive duration-0 emissions at one frame, the
step's duration is forced to 1 (`step_forced`). HF's TDT update skips that guard, so one looping row keeps its whole
batch running to max_symbols*T steps. With `max_symbols=None` the loop equals `generate` on active rows.

Per-utterance output of `ParakeetTeacher.run_batch` (the input of kitsune.parakeet_targets.pack_shard; array names
are the npz field names of one utterance, see that module's docstring):
  n_frames int, truncated bool, n_forced int, n_steps int, n_tok int, hyp str, ctc_hyp str,
  step_frame (S,) int, step_dur (S,) int, step_forced (S,) bool,
  tdt_topk_idx (S,k) int, tdt_topk_lp (S,k) float32, tdt_dur_lp (S,5) float32, tokens (N,) int,
  ctc_blank_lp (T,) float32, ctc_dense_frame (D,) int, ctc_topk_idx (D,k) int, ctc_topk_lp (D,k) float32,
  ctc_tokens (M,) int (greedy CTC: argmax per frame, repeats collapsed, blanks dropped)
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

PARAKEET_PATH = "models/parakeet-tdt_ctc-0.6b-ja-hf"        # in the data repo, and locally under $KITSUNE_DIR
NEMO_REPO, NEMO_REVISION = "nvidia/parakeet-tdt_ctc-0.6b-ja", "44edb27eea9317daf89333e75eb830db4b1cc298"
PARAKEET_FILES = {
    "config.json": "84af087beec652e7bd7778a085536e9bd58476c948a8c5d7c3c89d193efdcbc3",
    "ctc_head.safetensors": "d868b3a919a7e29d1471c878304e62990b1eacfbdb9b269b5e11571909b83b23",
    "generation_config.json": "821ee06c5b5d8264acf6f7341611ad190bd9ccb008e324e632f7cb8384cccf18",
    "kitsune_model.json": "03b1486c9e9a8c615e9c8cef5ef1d85ddca7ed264435abf12775b9b329ecae5a",
    "model.safetensors": "31e0e9429aea32c3c856a543dfc58ab82ac30dad552f0a0e339eb5465019e9e6",
    "processor_config.json": "b9b3737cc3e5d20d01613d91a3da2ae93dd609de5d0a43aafe303eed01e5823d",
    "tokenizer.json": "f604c2935adb455b6b5b66bb17ea9527f0b6e880cf26e3158a0c10a1709ac8e6",
    "tokenizer_config.json": "6890aaeb5b2f02f47c3cccfa2b172789a0bb5ad306265f70cfdacbd2424ef16f",
}
BLANK, VOCAB, DURATIONS, FRAME_S = 3072, 3073, (0, 1, 2, 3, 4), 0.08
CTC_HEAD_FILE = "ctc_head.safetensors"
MODEL_JSON = "kitsune_model.json"
SAMPLING_RATE = 16000


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_dir(d: Path) -> list[str]:
    """Every PARAKEET_FILES entry is present with its pinned sha256; returns the problems (empty = ok)."""
    d = Path(d)
    problems = []
    for name, sha in sorted(PARAKEET_FILES.items()):
        p = d / name
        if not p.is_file():
            problems.append(f"{name}: missing in {d}")
            continue
        got = file_sha256(p)
        if got != sha:
            problems.append(f"{name}: sha256 {got} != pinned {sha or '(unset)'}")
    return problems


def lstm_decoder(dec):
    """Step function over HF's ParakeetRNNTDecoder: (tokens (B,), state | None) -> (g (B,H), (h, c)).

    A None state is the zero state, as HF's lazily initialised decoder cache."""
    def step(tok, state):
        out, (h, c) = dec.lstm(dec.embedding(tok[:, None]), state)
        return dec.decoder_projector(out)[:, 0], (h, c)
    return step


def greedy_tdt(f_proj, valid, decoder, joint_head, act, *, blank, durations, k_tdt, max_symbols, hard_cap) -> dict:
    """Batched greedy TDT over projected encoder frames f_proj (B,T,H) with valid lengths `valid` (B,).

    `decoder(tok, state) -> (g, state)` with state a tuple of (L,B,H) tensors (or None at the start); `joint_head`
    maps act(f + g) to vocab+len(durations) logits. Returns per-row lists (numpy): step_frame, step_dur,
    step_forced, tdt_topk_idx, tdt_topk_lp, tdt_dur_lp, tokens, and truncated (bool per row; hit hard_cap)."""
    B, T, _ = f_proj.shape
    dev = f_proj.device
    n_dur = len(durations)
    dur_t = torch.tensor(durations, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)
    valid = valid.to(dev).long()
    tok = torch.full((B,), blank, dtype=torch.long, device=dev)
    g, state = decoder(tok, None)
    t = torch.zeros(B, dtype=torch.long, device=dev)
    sym = torch.zeros(B, dtype=torch.long, device=dev)
    active = t < valid
    steps = 0
    rec = {k: [] for k in ("act", "frame", "dur", "forced", "idx", "lp", "dlp")}
    while bool(active.any()) and steps < hard_cap:
        z = joint_head(act(f_proj[rows, t.clamp(max=T - 1)] + g)).float()
        V = z.shape[-1] - n_dur
        k = z[:, :V].argmax(-1)
        d = dur_t[z[:, V:].argmax(-1)]
        tlp = torch.log_softmax(z[:, :V], -1)
        dlp = torch.log_softmax(z[:, V:], -1)
        d = torch.where((k == blank) & (d == 0), torch.ones_like(d), d)
        sym = torch.where(d == 0, sym + 1, torch.zeros_like(sym))
        if max_symbols:
            forced = sym >= max_symbols
            d = torch.where(forced, torch.ones_like(d), d)
            sym = torch.where(forced, torch.zeros_like(sym), sym)
        else:
            forced = torch.zeros_like(active)
        boosted = tlp.clone()
        boosted[rows, k] = math.inf                      # column 0 is always the emitted token (ties included)
        top_idx = boosted.topk(k_tdt, -1).indices
        rec["act"].append(active)
        rec["frame"].append(t)
        rec["dur"].append(d)
        rec["forced"].append(forced)
        rec["idx"].append(top_idx)
        rec["lp"].append(tlp.gather(-1, top_idx))
        rec["dlp"].append(dlp)
        emit = active & (k != blank)
        if bool(emit.any()):
            g2, state2 = decoder(k, state)
            g = torch.where(emit[:, None], g2, g)
            state = tuple(torch.where(emit[None, :, None], s2, s) for s2, s in zip(state2, state))
        t = t + d * active
        active = t < valid
        steps += 1
    truncated = active.cpu().numpy()
    if steps:
        R = {k: torch.stack(v, 0).cpu().numpy() for k, v in rec.items()}   # (S, B, ...), one copy per batch
    out = {k: [] for k in ("step_frame", "step_dur", "step_forced", "tdt_topk_idx", "tdt_topk_lp", "tdt_dur_lp",
                           "tokens")}
    for b in range(B):
        if not steps:
            sel = np.zeros(0, dtype=bool)
            R = {"frame": np.zeros((0, B), np.int64), "dur": np.zeros((0, B), np.int64),
                 "forced": np.zeros((0, B), bool), "idx": np.zeros((0, B, k_tdt), np.int64),
                 "lp": np.zeros((0, B, k_tdt), np.float32), "dlp": np.zeros((0, B, n_dur), np.float32)}
        else:
            sel = R["act"][:, b]
        out["step_frame"].append(R["frame"][sel, b])
        out["step_dur"].append(R["dur"][sel, b])
        out["step_forced"].append(R["forced"][sel, b].astype(bool))
        out["tdt_topk_idx"].append(R["idx"][sel, b])
        out["tdt_topk_lp"].append(R["lp"][sel, b].astype(np.float32))
        out["tdt_dur_lp"].append(R["dlp"][sel, b].astype(np.float32))
        col0 = R["idx"][sel, b, 0]
        out["tokens"].append(col0[col0 != blank])
    out["truncated"] = [bool(x) for x in truncated]
    return out


def ctc_targets(ctc_lp, valid, *, k_ctc, dense_thr) -> dict:
    """CTC soft targets from log-probs (B,T,V), last class = blank: per-row lists (numpy) of ctc_blank_lp (T_i,),
    ctc_dense_frame (D_i,) (frames with p(blank) < dense_thr), ctc_topk_idx/ctc_topk_lp (D_i,k), ctc_tokens."""
    B, T, V = ctc_lp.shape
    blank = V - 1
    thr = math.log(dense_thr)
    top_lp, top_idx = ctc_lp.topk(k_ctc, -1)
    arg = ctc_lp.argmax(-1)
    blank_lp = ctc_lp[..., blank].float().cpu().numpy()
    top_lp, top_idx, arg = top_lp.float().cpu().numpy(), top_idx.cpu().numpy(), arg.cpu().numpy()
    out = {k: [] for k in ("ctc_blank_lp", "ctc_dense_frame", "ctc_topk_idx", "ctc_topk_lp", "ctc_tokens")}
    for b in range(B):
        n = int(valid[b])
        bl = blank_lp[b, :n]
        dense = np.nonzero(bl < thr)[0]
        out["ctc_blank_lp"].append(bl.astype(np.float32))
        out["ctc_dense_frame"].append(dense.astype(np.int64))
        out["ctc_topk_idx"].append(top_idx[b, dense])
        out["ctc_topk_lp"].append(top_lp[b, dense].astype(np.float32))
        out["ctc_tokens"].append(np.asarray(ctc_collapse(arg[b, :n].tolist(), blank), dtype=np.int64))
    return out


def ctc_collapse(frames: list[int], blank: int = BLANK) -> list[int]:
    """Greedy CTC: collapse repeats, drop blanks."""
    out, prev = [], None
    for x in frames:
        if x != prev and x != blank:
            out.append(x)
        prev = x
    return out


def ctc_linear(weight: torch.Tensor, bias: torch.Tensor) -> torch.nn.Linear:
    """The NeMo CTC head (Conv1d k=1, weight (V,H,1) or (V,H)) as a Linear."""
    w = weight.reshape(weight.shape[0], -1)
    lin = torch.nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(bias)
    return lin


class ParakeetTeacher:
    def __init__(self, d: Path, device: str = "cuda", encoder_dtype: torch.dtype | None = None):
        """AutoProcessor + ParakeetForTDT from the local dir only. The encoder runs in bf16 (fp32 on CPU unless
        encoder_dtype says otherwise); the projector, prediction net, joint and CTC head stay fp32."""
        from safetensors.torch import load_file
        from transformers import AutoProcessor, ParakeetForTDT

        d = Path(d)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(d, local_files_only=True)
        self.model = ParakeetForTDT.from_pretrained(d, local_files_only=True, dtype=torch.float32).eval()
        cfg = self.model.config
        assert cfg.blank_token_id == BLANK and cfg.vocab_size == VOCAB and tuple(cfg.durations) == DURATIONS, cfg
        self.encoder_dtype = encoder_dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
        self.model.to(device)
        self.model.encoder.to(self.encoder_dtype)
        head = load_file(str(d / CTC_HEAD_FILE))
        self.ctc = ctc_linear(head["weight"], head["bias"]).to(device).eval()
        from transformers.activations import ACT2FN
        self.act = ACT2FN[cfg.hidden_act]
        self.decoder = lstm_decoder(self.model.decoder)

    def features(self, waves: list[np.ndarray]) -> dict:
        """Log-mel features on the CPU (the extractor has no dither: deterministic)."""
        return self.processor.feature_extractor(waves, sampling_rate=SAMPLING_RATE, return_tensors="pt")

    def run_batch(self, feats, attention_mask, *, k_tdt=8, k_ctc=8, ctc_dense_thr=0.95, max_symbols=10) -> list[dict]:
        with torch.inference_mode():
            x = feats.to(self.device, self.encoder_dtype)
            am = attention_mask.to(self.device) if attention_mask is not None else None
            enc = self.model.encoder(input_features=x, attention_mask=am, output_attention_mask=True)
            h = enc.last_hidden_state.float()
            B, T = h.shape[:2]
            emask = getattr(enc, "attention_mask", None)
            valid = emask.sum(-1).long() if emask is not None else torch.full((B,), T, device=h.device)
            ctc = ctc_targets(torch.log_softmax(self.ctc(h), -1), valid.cpu(), k_ctc=k_ctc, dense_thr=ctc_dense_thr)
            f = self.model.encoder_projector(h)
            hard_cap = ((max_symbols or self.model.config.max_symbols_per_step) + 1) * T + 1
            tdt = greedy_tdt(f, valid, self.decoder, self.model.joint.head, self.act, blank=BLANK,
                             durations=DURATIONS, k_tdt=k_tdt, max_symbols=max_symbols, hard_cap=hard_cap)
        hyps = self.processor.batch_decode([r.tolist() for r in tdt["tokens"]], skip_special_tokens=True)
        ctc_hyps = self.processor.batch_decode([r.tolist() for r in ctc["ctc_tokens"]], skip_special_tokens=True)
        out = []
        for b in range(B):
            u = {k: v[b] for k, v in tdt.items()}
            u.update({k: v[b] for k, v in ctc.items()})
            u.update(n_frames=int(valid[b]), hyp=hyps[b], ctc_hyp=ctc_hyps[b], n_steps=len(u["step_frame"]),
                     n_tok=len(u["tokens"]), n_forced=int(u["step_forced"].sum()))
            out.append(u)
        return out
