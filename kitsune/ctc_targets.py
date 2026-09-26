"""Per-frame CTC soft targets of the Parakeet teacher, as the CTC students train on them.

Source: parakeet_out/<source>/<stem>.npz (FORMAT in kitsune/parakeet_targets.py). Per utterance the label pass stored
  ctc_blank_lp    log p(blank) at every valid frame (T_i of them, fp16)
  ctc_dense_frame the frames with p(blank) < 0.95 ("dense"), utterance-local
  ctc_topk_idx/lp the teacher's top-8 over all 3073 classes (blank may be among them) on the dense frames (fp16 lp)
Everything else about a frame follows: a blank-only frame's teacher distribution is {blank: p, not-blank: 1 - p}, and
the teacher's argmax (col0) is the blank there and the stored top-1 on a dense frame. The CTC target (decision 22) is
the teacher's greedy CTC path, ctc_greedy(ctc_col0(...)), which is by construction the jsonl `ctc_hyp` tokens (K3).

`FrameTargets` keeps the stored dtypes (a ~1,000 h store holds ~45M frames: 2 bytes per frame plus ~35 bytes per
dense frame). `collate_frame_targets` pads a batch to the student's frame count T and widens to what the loss reads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kitsune.parakeet_targets import BLANK, ctc_col0, ctc_greedy, load_shard


@dataclass
class FrameTargets:
    n_frames: int
    blank_lp: np.ndarray  # float16 (T,)       log p(blank) at every frame
    dense_frame: np.ndarray  # int32 (D,)      frames with p(blank) < ctc_dense_thr, increasing
    topk_idx: np.ndarray  # int16 (D, k)       teacher top-k classes there (blank may appear)
    topk_lp: np.ndarray  # float16 (D, k)      their log-probs, descending
    ctc_ids: np.ndarray  # int32 (U,)          greedy CTC path: collapse(col0) without blanks

    def __post_init__(self):
        self.n_frames = int(self.n_frames)
        if len(self.blank_lp) != self.n_frames:
            raise ValueError(f"{len(self.blank_lp)} blank log-probs for {self.n_frames} frames")
        if len(self.dense_frame) != len(self.topk_idx) or self.topk_idx.shape != self.topk_lp.shape:
            raise ValueError("dense_frame / topk_idx / topk_lp disagree in length or shape")

    @property
    def k(self) -> int:
        return int(self.topk_idx.shape[1])

    def col0(self) -> np.ndarray:
        """The teacher's argmax per frame (int64 (T,))."""
        return ctc_col0(self.n_frames, self.dense_frame, self.topk_idx)


def frame_targets(n_frames, blank_lp, dense_frame, topk_idx, topk_lp) -> FrameTargets:
    """FrameTargets from one utterance's arrays (top-k as (D, k), also for D = 0), cast to the stored dtypes; ctc_ids
    derived from them."""
    idx, lp = np.asarray(topk_idx, dtype=np.int16), np.asarray(topk_lp, dtype=np.float16)
    if idx.ndim != 2 or lp.ndim != 2:
        raise ValueError(f"top-k arrays must be (D, k), got {idx.shape} / {lp.shape}")
    ft = FrameTargets(int(n_frames), np.asarray(blank_lp, dtype=np.float16), np.asarray(dense_frame, dtype=np.int32),
                      idx, lp, np.zeros(0, np.int32))
    ft.ctc_ids = np.asarray(ctc_greedy(ft.col0()), dtype=np.int32)
    return ft


def load_ctc_targets(npz_path) -> dict[str, FrameTargets]:
    """id -> FrameTargets for every utterance of one parakeet_out npz, in shard order."""
    sh = load_shard(npz_path)
    z, out = sh.z, {}
    for i, uid in enumerate(sh.ids):
        fr, de = sh._span("frame_offsets", i), sh._span("dense_offsets", i)
        if uid in out:
            raise ValueError(f"{npz_path}: duplicate id {uid}")
        out[uid] = frame_targets(z["n_frames"][i], z["ctc_blank_lp"][fr], z["ctc_dense_frame"][de],
                                 z["ctc_topk_idx"][de], z["ctc_topk_lp"][de])
    return out


def targets_from_log_probs(log_probs: torch.Tensor, lengths: torch.Tensor, *, k: int = 8,
                           dense_thr: float = 0.95) -> list[FrameTargets]:
    """What the label pass would store for these CTC log-probs (B,T,V) (kitsune.parakeet.ctc_targets, then the npz
    dtypes): for tests, fixtures and the build script's step-0 KD check."""
    from kitsune.parakeet import ctc_targets

    t = ctc_targets(log_probs.detach().float(), lengths.cpu(), k_ctc=k, dense_thr=dense_thr)
    return [frame_targets(len(t["ctc_blank_lp"][b]), t["ctc_blank_lp"][b], t["ctc_dense_frame"][b],
                          t["ctc_topk_idx"][b], t["ctc_topk_lp"][b]) for b in range(len(t["ctc_blank_lp"]))]


def collate_frame_targets(targets: list[FrameTargets], t_max: int | None = None) -> dict[str, torch.Tensor]:
    """Pad a batch of FrameTargets to T = t_max (the student's frame count) or the longest n_frames:
      frame_mask [B,T] bool, dense_mask [B,T] bool, blank_lp [B,T] float32 (0 past n_frames),
      topk_idx [B,T,k] long (0 where not dense), topk_lp [B,T,k] float32 (-inf where not dense),
      ctc_targets [sum U] long, ctc_target_lengths [B] long, n_frames [B] long."""
    if not targets:
        raise ValueError("empty batch")
    ks = {t.k for t in targets}
    if len(ks) != 1:
        raise ValueError(f"mixed top-k widths {sorted(ks)}")
    k = ks.pop()
    B = len(targets)
    longest = max(t.n_frames for t in targets)
    T = longest if t_max is None else int(t_max)
    if T < longest:
        raise ValueError(f"t_max {T} < n_frames {longest}: the student has fewer frames than its targets")
    blank_lp = np.zeros((B, T), np.float32)
    frame_mask = np.zeros((B, T), bool)
    dense_mask = np.zeros((B, T), bool)
    topk_idx = np.zeros((B, T, k), np.int64)
    topk_lp = np.full((B, T, k), -np.inf, np.float32)
    for b, t in enumerate(targets):
        n = t.n_frames
        blank_lp[b, :n] = t.blank_lp
        frame_mask[b, :n] = True
        d = t.dense_frame.astype(np.int64)
        dense_mask[b, d] = True
        topk_idx[b, d] = t.topk_idx
        topk_lp[b, d] = t.topk_lp
    ids = [t.ctc_ids.astype(np.int64) for t in targets]
    return dict(
        frame_mask=torch.from_numpy(frame_mask), dense_mask=torch.from_numpy(dense_mask),
        blank_lp=torch.from_numpy(blank_lp), topk_idx=torch.from_numpy(topk_idx), topk_lp=torch.from_numpy(topk_lp),
        ctc_targets=torch.from_numpy(np.concatenate(ids) if ids else np.zeros(0, np.int64)),
        ctc_target_lengths=torch.tensor([len(x) for x in ids], dtype=torch.long),
        n_frames=torch.tensor([t.n_frames for t in targets], dtype=torch.long),
    )


def infeasible(ids, n_frames: int) -> bool:
    """CTC cannot align U tokens with r adjacent repeats in fewer than U + r frames. The greedy path of the stored
    frames is feasible by construction (it came from those frames); another target sequence (a TDT hyp, a reference)
    may not be."""
    ids = np.asarray(ids)
    repeats = int(np.sum(ids[1:] == ids[:-1])) if len(ids) > 1 else 0
    return len(ids) + repeats > int(n_frames)


__all__ = ["BLANK", "FrameTargets", "collate_frame_targets", "frame_targets", "infeasible", "load_ctc_targets",
           "targets_from_log_probs"]
