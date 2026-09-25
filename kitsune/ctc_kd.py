"""CTC distillation loss of the Parakeet-family students (STUDY.md 2.1; decisions 22, 23).

Per optimiser step, with N_u = the CTC target tokens of the whole step (all micro-batches):

    L = (w_kl * sum_utt sum_{t < T_i} KL_t + w_ctc * sum_utt CTC_utt) / N_u,     w_kl 1.0, w_ctc 0.8

KL_t is a forward KL between coarse distributions at T = 1 (no T^2 factor; at T = 2, 9-21 % of the mass would fall
outside the stored top-8). The frames align 1:1 with the teacher's (same features, same 8x subsampling), so every
valid frame gets a KL term - there is no blank elimination:
- dense frames (teacher p(blank) < 0.95): bins = the stored top-8 U {blank} plus one rest bucket. p(blank) is stored
  exactly for every frame, so the blank bin is exact even when the blank is not in the top-8. p_rest = 1 - sum(p) with
  the fp16 overshoot rescaled and the rest clamped at 0 exactly as kitsune/kd.py does; the student's rest is the
  logsumexp of its log-probs outside the bins. The gradient w.r.t. a student logit is then q - p on the bins and
  q (1 - p_rest / q_rest) elsewhere: the student matches the teacher's bins and total tail, and is free in the tail's
  shape.
- blank-only frames: a 2-bin KL over {blank, not blank}. The not-blank mass of the student is the logsumexp over the
  non-blank classes (exact even when p(blank) rounds to 1 in fp32, where log1p(-exp(.)) would give -inf).
CTC_utt is `F.ctc_loss(log_probs, the teacher's greedy CTC path, reduction="sum", blank=3072, zero_infinity=True)`
(HF's "mean" divides by target length per utterance; NeMo's default mean_volume is yet another normalisation).

`ctc_kd_losses` returns SUMS (or per-utterance vectors with per_utt=True) so the trainer aggregates micro-batches and
logs per source; `ctc_kd_objective` divides by the step's N_u. The frame metrics (argmax agreement with the teacher,
student / teacher argmax = blank) are counts over valid frames: the argmax-blank share is the blank-collapse watch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

CTC_BLANK = 3072  # == kitsune.ctc_student.CTC_BLANK (not imported: this module needs no transformers)
W_KL, W_CTC = 1.0, 0.8  # decision 23


def _fit(batch: dict, T: int, device) -> dict:
    """The batch on `device` with its frame axis matched to the student's T. Frames past the longest n_frames are
    padding on both sides, so the longer side is cut; a target frame the student does not have is an error."""
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    Tb = out["frame_mask"].shape[1]
    if Tb > T:
        if bool(out["frame_mask"][:, T:].any()):
            raise ValueError(f"targets have valid frames past the student's {T} frames (n_frames {out['n_frames']})")
        for k in ("frame_mask", "dense_mask", "blank_lp", "topk_idx", "topk_lp"):
            out[k] = out[k][:, :T]
    return out


def _per_row(values: Tensor, rows: Tensor, B: int) -> Tensor:
    return torch.zeros(B, dtype=values.dtype, device=values.device).index_add_(0, rows, values)


def ctc_kd_losses(log_probs: Tensor, batch: dict, blank: int = CTC_BLANK, per_utt: bool = False) -> dict[str, Tensor]:
    """log_probs (B,T,V): the student's fp32 log_softmax (fp64 is kept for gradcheck; half types are upcast).
    batch: kitsune.ctc_targets.collate_frame_targets (any frame width >= the longest n_frames). Returns SUMS over the
    batch as 0-dim tensors (per_utt=True: (B,) vectors):
      kl_dense, kl_blank, ctc               differentiable
      n_dense, n_blank_frames, n_frames, n_tokens              counts (n_tokens = sum U, the CTC target tokens)
      argmax_agree, argmax_blank, teacher_blank                counts over valid frames"""
    lp = log_probs if log_probs.dtype == torch.float64 else log_probs.float()
    B, T, V = lp.shape
    b = _fit(batch, T, lp.device)
    Tb = b["frame_mask"].shape[1]
    if Tb < T:
        lp = lp[:, :Tb]  # student frames past every target are padding
    fm = b["frame_mask"]
    dm = b["dense_mask"] & fm
    bm = fm & ~dm
    if int(b["n_frames"].max()) > lp.shape[1]:
        raise ValueError(f"n_frames {b['n_frames'].tolist()} > the student's {lp.shape[1]} frames")

    with torch.autocast(device_type=lp.device.type, enabled=False):
        # dense frames: stored top-k U {blank} + rest (boolean indexing: padded frames never enter any sum)
        rows_d = dm.nonzero()[:, 0]
        lpd = lp[dm]  # (Nd, V)
        idx = b["topk_idx"][dm]  # (Nd, k)
        blank_in = (idx == blank).any(dim=-1)
        idx_all = torch.cat([idx, torch.full_like(idx[:, :1], blank)], dim=-1)  # (Nd, k+1); a duplicate blank bin
        p_blank = torch.where(blank_in, 0.0, b["blank_lp"][dm].to(lp.dtype).exp())  # gets p = 0 when blank is in top-k
        p = torch.cat([b["topk_lp"][dm].to(lp.dtype).exp(), p_blank[:, None]], dim=-1)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0)
        p_r = (1.0 - p.sum(dim=-1)).clamp_min(0.0)
        logq = lpd.gather(-1, idx_all)
        logq_r = lpd.scatter(-1, idx_all, float("-inf")).logsumexp(dim=-1)
        negent = torch.xlogy(p, p).sum(dim=-1) + torch.xlogy(p_r, p_r)
        kl_d = negent - (p * logq).sum(dim=-1) - p_r * logq_r

        # blank-only frames: 2-bin KL
        rows_b = bm.nonzero()[:, 0]
        lpb = lp[bm]  # (Nb, V)
        p_b = b["blank_lp"][bm].to(lp.dtype).exp().clamp(max=1.0)
        p_nb = (1.0 - p_b).clamp_min(0.0)
        lq_b = lpb[:, blank]
        if blank == V - 1:
            lq_nb = lpb[:, :blank].logsumexp(dim=-1)
        else:
            lq_nb = lpb.index_fill(-1, torch.tensor([blank], device=lp.device), float("-inf")).logsumexp(dim=-1)
        kl_b = torch.xlogy(p_b, p_b) + torch.xlogy(p_nb, p_nb) - p_b * lq_b - p_nb * lq_nb

        # CTC on the teacher's greedy path; ctc_loss reads only frames < n_frames of each row
        ctc = F.ctc_loss(lp.transpose(0, 1), b["ctc_targets"], b["n_frames"], b["ctc_target_lengths"], blank=blank,
                         reduction="none", zero_infinity=True)

    with torch.no_grad():
        s_arg = lp.argmax(dim=-1)
        t_col0 = torch.where(dm, b["topk_idx"][..., 0], torch.full_like(s_arg, blank))
        agree = (s_arg == t_col0) & fm
        s_blank = (s_arg == blank) & fm
        t_blank = (t_col0 == blank) & fm

    if per_utt:
        return dict(kl_dense=_per_row(kl_d, rows_d, B), kl_blank=_per_row(kl_b, rows_b, B), ctc=ctc,
                    n_dense=dm.sum(dim=1), n_blank_frames=bm.sum(dim=1), n_frames=fm.sum(dim=1),
                    n_tokens=b["ctc_target_lengths"], argmax_agree=agree.sum(dim=1), argmax_blank=s_blank.sum(dim=1),
                    teacher_blank=t_blank.sum(dim=1))
    return dict(kl_dense=kl_d.sum(), kl_blank=kl_b.sum(), ctc=ctc.sum(), n_dense=dm.sum(), n_blank_frames=bm.sum(),
                n_frames=fm.sum(), n_tokens=b["ctc_target_lengths"].sum(), argmax_agree=agree.sum(),
                argmax_blank=s_blank.sum(), teacher_blank=t_blank.sum())


def ctc_kd_objective(losses: dict[str, Tensor], n_tokens, w_kl: float = W_KL, w_ctc: float = W_CTC) -> Tensor:
    """The trainer's loss for one micro-batch: (w_kl (kl_dense + kl_blank) + w_ctc ctc) / N_u, where N_u is the CTC
    target tokens of the WHOLE optimiser step (all micro-batches), so micro-batch losses add up to the step's loss.
    A step without any target token (only silent utterances) divides by 1."""
    if isinstance(n_tokens, Tensor):
        n = n_tokens.clamp_min(1)
    else:
        n = max(int(n_tokens), 1)
    return (w_kl * (losses["kl_dense"].sum() + losses["kl_blank"].sum()) + w_ctc * losses["ctc"].sum()) / n
