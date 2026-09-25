"""Runtime patches for training the Parakeet/CohereAsr encoder, plus BatchNorm freezing and an SDPA report.

Rel-pos once per batch. `ParakeetEncoderRelPositionalEncoding` builds the sinusoids for relative offsets
T-1..-(T-1) expanded to the batch size, so every layer's `relative_k_proj` multiplies B identical (2T-1, d) copies:
about 2d^2 MAC per frame per layer, ~11% of a 2560-FFN layer. The patch computes the sinusoids for one row and
returns a stride-0 `expand` view; `relative_k_proj` then sees a batch whose rows are provably identical (stride 0
on dim 0), projects row 0 once and expands the result. The attention forward itself is untouched: its
`.view(batch, -1, heads, head_dim)` and the matmul with the query accept the expanded tensor, and autograd sums the
broadcast gradient back into the single projection. Any stride-0 batch means identical rows, so the shortcut is
exact whatever produced the tensor; if position dropout is ever > 0 in training, dropout materialises independent
rows and the projection falls back to the full computation by itself.

BatchNorm. Each conformer conv module has a BatchNorm1d. In train mode it would take batch statistics that include
padded frames, and under gradient checkpointing it would update the running stats twice per step. The teacher's
stats come from 276k batches, so they stay frozen (eval mode) for the whole run. `freeze_batchnorm` overrides each BN
module's own `train()` so that a later `model.train()` - from our code, HF's, or a library's - cannot unfreeze it.
"""
import types
import warnings
from typing import Callable

import torch
from torch import nn

_RELPOS_ATTR = "_kitsune_relpos_unpatch"
_BN_ATTR = "_kitsune_bn_frozen"


@torch.no_grad()
def _relpos_one_row(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """ParakeetEncoderRelPositionalEncoding.forward for batch row 0 only, expanded (stride 0) to the batch.

    Same ops as HF on a (1, ...) slice: the matmul is an outer product (K=1), so the values are bitwise equal."""
    B, T = hidden_states.shape[:2]
    dev = hidden_states.device
    position_ids = torch.arange(T - 1, -T, -1, device=dev)
    inv_freq = self.inv_freq[None, :, None].float().to(dev)  # (1, d/2, 1)
    device_type = dev.type if dev.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq @ position_ids[None, None, :].float()).transpose(1, 2)
        pos = torch.stack([freqs.sin(), freqs.cos()], dim=-1)
        pos = pos.reshape(*pos.shape[:-2], -1)
    # cast BEFORE expanding: .to() on an expanded tensor would materialise the B copies again
    return pos.to(dtype=hidden_states.dtype).expand(B, -1, -1)


def _linear_once_over_batch(self, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.shape[0] > 1 and x.stride(0) == 0:
        return nn.Linear.forward(self, x[:1]).expand(x.shape[0], -1, -1)
    return nn.Linear.forward(self, x)


def patch_relpos_once_per_batch(model: nn.Module) -> Callable[[], None]:
    """Make every Parakeet encoder in `model` project relative positions once per batch instead of B times.

    Idempotent: a second call returns the first call's handle. The handle (also idempotent) restores the original
    forwards. Patch before torch.compile. Parity with the unpatched model is covered by tests/test_patches.py."""
    from transformers.models.parakeet.modeling_parakeet import (
        ParakeetEncoderAttention,
        ParakeetEncoderRelPositionalEncoding,
    )

    existing = getattr(model, _RELPOS_ATTR, None)
    if existing is not None:
        return existing

    patched: list[nn.Module] = []
    for m in model.modules():
        if isinstance(m, ParakeetEncoderRelPositionalEncoding):
            m.forward = types.MethodType(_relpos_one_row, m)
            patched.append(m)
        elif isinstance(m, ParakeetEncoderAttention):
            m.relative_k_proj.forward = types.MethodType(_linear_once_over_batch, m.relative_k_proj)
            patched.append(m.relative_k_proj)
    if not any(isinstance(m, ParakeetEncoderRelPositionalEncoding) for m in patched):
        raise ValueError("no ParakeetEncoderRelPositionalEncoding found in model; nothing to patch")

    def unpatch():
        for mod in patched:
            mod.__dict__.pop("forward", None)  # the class method is visible again
        patched.clear()
        model.__dict__.pop(_RELPOS_ATTR, None)

    setattr(model, _RELPOS_ATTR, unpatch)
    return unpatch


def is_relpos_patched(model: nn.Module) -> bool:
    return getattr(model, _RELPOS_ATTR, None) is not None


def _batchnorms(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def _frozen_bn_train(self, mode: bool = True):
    return nn.Module.train(self, False)


def freeze_batchnorm(model: nn.Module) -> int:
    """Put every BatchNorm in eval mode and keep it there across later `.train()` calls. Its affine weight and bias
    stay trainable; only the running statistics are frozen. Returns the number of BN modules."""
    bns = _batchnorms(model)
    for m in bns:
        m.train = types.MethodType(_frozen_bn_train, m)
        setattr(m, _BN_ATTR, True)
        nn.Module.train(m, False)
    return len(bns)


def unfreeze_batchnorm(model: nn.Module) -> int:
    """Undo `freeze_batchnorm` (e.g. for BN recalibration). Modes are left as they are; call `.train()` after."""
    bns = _batchnorms(model)
    for m in bns:
        m.__dict__.pop("train", None)
        m.__dict__.pop(_BN_ATTR, None)
    return len(bns)


def train_mode(model: nn.Module) -> nn.Module:
    model.train()
    freeze_batchnorm(model)
    return model


def assert_bn_frozen(model: nn.Module) -> int:
    """Raise if any BatchNorm is in train mode. Returns the number of BN modules checked."""
    bns = _batchnorms(model)
    bad = [name for name, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    if bad:
        raise AssertionError(f"{len(bad)}/{len(bns)} BatchNorm modules in train mode, e.g. {bad[:3]}")
    return len(bns)


def sdpa_backend_report(device: str | torch.device | None = None, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Which SDPA kernels are enabled, and which ones can actually run the two attention shapes this model uses.

    `encoder`: head_dim 160 with an additive float bias that requires grad (the rel-pos term), so flash attention
    is expected to be unusable; its padded row masks keys AND queries (valid_q & valid_k, as the model does), so it
    has fully masked query rows, where kernels differ. `decoder`: head_dim 128, boolean causal+padding mask. Each
    backend is tried in isolation (forward + backward, tiny shapes) and counts as usable only if the output and the
    q/k/v/bias gradients are finite; `selected` is what torch's dispatcher picks with all enabled."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = torch.backends.cuda
    rep: dict = dict(
        torch=torch.__version__, cuda_available=torch.cuda.is_available(), device=str(dev), dtype=str(dtype),
        enabled=dict(flash=b.flash_sdp_enabled(), mem_efficient=b.mem_efficient_sdp_enabled(),
                     math=b.math_sdp_enabled(), cudnn=b.cudnn_sdp_enabled()),
    )
    if dev.type == "cuda":
        rep["device_name"] = torch.cuda.get_device_name(dev)
        rep["capability"] = list(torch.cuda.get_device_capability(dev))
        rep["flash_attention_available"] = b.is_flash_attention_available()

    def make(case: str):
        B, H, T = 2, 8, 64
        hd = 160 if case == "encoder" else 128
        q, k, v = (torch.randn(B, H, T, hd, device=dev, dtype=dtype, requires_grad=True) for _ in range(3))
        if case == "encoder":
            mask = torch.randn(B, H, T, T, device=dev, dtype=dtype)
            mask[1, :, :, T // 2:] = float("-inf")  # padded keys, as in a real batch
            mask[1, :, T // 2:, :] = float("-inf")  # padded queries: fully masked rows, as the encoder's mask gives
            mask.requires_grad_(True)
        else:
            mask = torch.ones(T, T, device=dev, dtype=torch.bool).tril()[None, None].expand(B, 1, T, T)
        return q, k, v, mask

    backends = {"flash": SDPBackend.FLASH_ATTENTION, "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}
    for case in ("encoder", "decoder"):
        usable = {}
        for name, backend in backends.items():
            q, k, v, mask = make(case)
            try:
                with warnings.catch_warnings(), sdpa_kernel(backend):
                    warnings.simplefilter("ignore")
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                    out.float().sum().backward()
                grads = [q.grad, k.grad, v.grad]
                if mask.grad is not None:  # only where the bias is finite: the model's masked_fill zeroes the rest
                    grads.append(mask.grad[mask.isfinite()])
                finite = bool(torch.isfinite(out).all()) and all(g is not None and bool(torch.isfinite(g).all())
                                                                 for g in grads)
                usable[name] = True if finite else "runs, but gives a non-finite output or gradient"
            except RuntimeError as e:
                usable[name] = str(e).splitlines()[0][:160] or False
        rep[f"{case}_usable"] = usable
        try:  # private but stable since 2.0; purely informational
            q, k, v, mask = make(case)
            choice = torch._fused_sdp_choice(q, k, v, mask, 0.0, False)
            rep[f"{case}_selected"] = SDPBackend(choice).name
        except Exception as e:  # noqa: BLE001
            rep[f"{case}_selected"] = f"unknown ({type(e).__name__})"
    return rep
