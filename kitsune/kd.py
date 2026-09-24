"""Distillation losses on the stored teacher top-16, and the decoupled L2-SP anchor.

What is stored per target token (scripts/02_teacher_pass.py): the teacher's 16 most likely ids and their exact
log-probs (fp32 head, normalised over the full vocab, saved as fp16). The mass outside them, p_r = 1 - sum(p_k), is
therefore known exactly, but not how it spreads over the other 16368 ids. The loss is the exact forward KL between
the two 17-bin "coarse" distributions (16 teacher ids + one rest bucket), at T=1 (report_kd.md section 2):

    KL = sum_k p_k log(p_k / q_k) + p_r log(p_r / q_r),   q_r = student mass outside the teacher's top-16

Its gradient w.r.t. a student logit is q_m - p_m on the top-16 and q_m (1 - p_r/q_r) elsewhere: the student matches
the teacher's top-16 and total tail mass and is free in the tail's shape. fp16 rounding makes sum(p_k) exceed 1 by up
to ~4e-4 on 14-41% of tokens, so p_k is scaled down when that happens and p_r is clamped at 0.

CE is -log q of the teacher's greedy token (top_idx[:, 0]). Everything is fp32 and per position, so the trainer can
aggregate per step, source, utterance and confidence bucket; its loss is (w_kl*sum kl + w_ce*sum ce) / n_tokens.

L2-SP (Li et al. 2018) pulls the weights back towards the init: a loss-form L2 would be rescaled per coordinate by
Adam, so it is applied decoupled, like AdamW's decay, right after optimizer.step(): p -= lr*lam*(p - p0).
"""
from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

W_KL, W_CE = 1.0, 0.8  # D15c


def kd_losses(logits: Tensor, top_idx: Tensor, top_lp: Tensor) -> dict[str, Tensor]:
    """logits (N, V) student logits at the target positions; top_idx (N, K) teacher ids, column 0 = greedy target;
    top_lp (N, K) teacher log-probs (fp16 from disk is fine). Returns per-position (N,) fp32 tensors:

      kl, ce                       differentiable
      top1_match                   1.0 where the student's argmax is the teacher's greedy token
      student_entropy_coarse, teacher_entropy_coarse    entropy of the 17-bin distributions (nats)
      student_tail, teacher_tail   mass outside the teacher's top-K (q_r, p_r)
      teacher_p1                   teacher probability of the greedy token (for confidence buckets)
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        top_idx = top_idx.long()
        z = logits.float()
        # not log_softmax: its CPU kernel sums 16k tiny tail terms into the big one lane by lane and was measured
        # 1e-4 off in log q(top-1) (== the KL/CE bias); logsumexp's cascade sum is exact to ~1e-7
        logq = z - torch.logsumexp(z, dim=-1, keepdim=True)
        logq_k = logq.gather(-1, top_idx)
        logq_r = logq.scatter(-1, top_idx, float("-inf")).logsumexp(dim=-1)

        p_k = top_lp.detach().float().exp()
        p_k = p_k / p_k.sum(dim=-1, keepdim=True).clamp_min(1.0)
        p_r = (1.0 - p_k.sum(dim=-1)).clamp_min(0.0)
        teacher_negent = torch.xlogy(p_k, p_k).sum(dim=-1) + torch.xlogy(p_r, p_r)

        kl = teacher_negent - (p_k * logq_k).sum(dim=-1) - p_r * logq_r
        ce = -logq_k[:, 0]

        with torch.no_grad():
            q_k, q_r = logq_k.exp(), logq_r.exp()
            out_extra = dict(
                top1_match=(z.argmax(dim=-1) == top_idx[:, 0]).float(),
                student_entropy_coarse=-(q_k * logq_k).sum(dim=-1) - q_r * logq_r,
                teacher_entropy_coarse=-teacher_negent,
                student_tail=q_r,
                teacher_tail=p_r,
                teacher_p1=p_k[:, 0],
            )
    return dict(kl=kl, ce=ce, **out_extra)


def kd_objective(losses: dict[str, Tensor], n_tokens: int, w_kl: float = W_KL, w_ce: float = W_CE) -> Tensor:
    """The trainer's loss for one micro-batch: (w_kl*sum kl + w_ce*sum ce) / tokens in the whole optimizer step."""
    return (w_kl * losses["kl"].sum() + w_ce * losses["ce"].sum()) / n_tokens


def module_key(name: str) -> str:
    """Parameter name -> top-level module for logging: encoder.layers.7, decoder.layers.2, encoder.subsampling,
    decoder.embed_tokens, decoder.proj, proj_out, ..."""
    parts = name.split(".")
    if parts[0] == "model":
        parts = parts[1:]
    if len(parts) >= 3 and parts[1] == "layers":
        return ".".join(parts[:3])
    return ".".join(parts[:2]) if len(parts) > 2 else parts[0]


class L2SP:
    """Decoupled L2-SP anchor to the initial weights. Frozen params (requires_grad=False, e.g. the decoder pos_emb)
    are skipped, as are names containing any `exclude` substring. A tied parameter (proj_out.weight is
    embed_tokens.weight) is counted once. theta_0 is kept in bf16 on the params' device: the student init is saved in
    bf16, so this copy is exact and costs 2 bytes/param."""

    def __init__(self, named_params: Iterable[tuple[str, Tensor]], lam: float = 0.05, exclude: Iterable[str] = ()):
        self.lam = float(lam)
        exclude = tuple(exclude)
        self.names: list[str] = []
        self.params: list[Tensor] = []
        seen: set[int] = set()
        for name, p in named_params:
            if not p.requires_grad or id(p) in seen or any(e in name for e in exclude):
                continue
            seen.add(id(p))
            self.names.append(name)
            self.params.append(p)
        self.ref: list[Tensor] = [p.detach().to(torch.bfloat16).clone() for p in self.params]
        self._ref_sq: list[float] | None = None  # ||p0||^2 per param, computed on first relative query

    def _chunks(self, budget: int = 1 << 26):
        """(params, refs) groups of <= `budget` elements: foreach ops without a full-model fp32 temporary."""
        start, n = 0, 0
        for i, p in enumerate(self.params):
            if n and n + p.numel() > budget:
                yield self.params[start:i], self.ref[start:i]
                start, n = i, 0
            n += p.numel()
        if start < len(self.params):
            yield self.params[start:], self.ref[start:]

    @torch.no_grad()
    def apply_(self, lr: float, value: bool = False) -> Tensor | None:
        """p -= lr*lam*(p - p0). Call right after optimizer.step() with that step's lr. value=True also returns
        value() of the updated weights, from the same pass over them: p' - p0 = (1 - lr*lam)(p - p0), so it equals
        value() up to fp32 rounding (a second pass costs ~1 s for 617M params in host memory)."""
        a = lr * self.lam
        if a == 0.0 and not value:
            return None
        sq = []
        for ps, rs in self._chunks():
            diff = torch._foreach_sub(ps, rs)  # promotes the bf16 reference to the params' fp32
            if value:
                sq.extend(n.float().square() for n in torch._foreach_norm(diff))
            if a != 0.0:
                torch._foreach_add_(ps, diff, alpha=-a)
        if not value:
            return None
        if not sq:
            return torch.zeros((), dtype=torch.float32)
        return 0.5 * self.lam * (1.0 - a) ** 2 * torch.stack(sq).sum()

    @torch.no_grad()
    def _sq_dists(self) -> list[Tensor]:
        out = []
        for ps, rs in self._chunks():
            diff = torch._foreach_sub(ps, rs)
            out.extend(n.float().square() for n in torch._foreach_norm(diff))
        return out

    @torch.no_grad()
    def value(self) -> Tensor:
        """lam/2 * sum ||p - p0||^2 as an fp32 0-dim tensor on the params' device (no host sync)."""
        if not self.params:
            return torch.zeros((), dtype=torch.float32)
        return 0.5 * self.lam * torch.stack(self._sq_dists()).sum()

    @torch.no_grad()
    def per_module_distance(self, relative: bool = False) -> dict[str, float]:
        """||p - p0|| per top-level module (see `module_key`); with relative=True divided by ||p0||."""
        sq = defaultdict(float)
        ref_sq = defaultdict(float)
        dists = torch.stack(self._sq_dists()).tolist() if self.params else []
        if relative and self._ref_sq is None:
            self._ref_sq = [float(torch.linalg.vector_norm(r.float()).square()) for r in self.ref]
        for i, (name, d) in enumerate(zip(self.names, dists)):
            key = module_key(name)
            sq[key] += d
            if relative:
                ref_sq[key] += self._ref_sq[i]
        if relative:
            return {k: (v / ref_sq[k]) ** 0.5 if ref_sq[k] > 0 else float("nan") for k, v in sq.items()}
        return {k: v**0.5 for k, v in sq.items()}

    def state_dict(self) -> dict:
        return dict(lam=self.lam, names=list(self.names), ref=[r.cpu() for r in self.ref])

    def load_state_dict(self, sd: dict):
        if list(sd["names"]) != self.names:
            raise ValueError(f"L2SP parameter names differ from the checkpoint ({len(sd['names'])} vs {len(self.names)})")
        self.lam = float(sd["lam"])
        self.ref = [r.to(device=p.device, dtype=torch.bfloat16) for r, p in zip(sd["ref"], self.params)]
        self._ref_sq = None
