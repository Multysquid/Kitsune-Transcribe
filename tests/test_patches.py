"""The rel-pos patch must not change the model's function (padded batch, fp32, CPU), must actually remove the B-fold
projection, and must be idempotent and removable. BatchNorm must stay frozen across model.train(). CPU only."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from kitsune.patches import (  # noqa: E402
    assert_bn_frozen,
    freeze_batchnorm,
    is_relpos_patched,
    patch_relpos_once_per_batch,
    sdpa_backend_report,
    train_mode,
    unfreeze_batchnorm,
)

PROMPT = [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13]
PAD = 2
D, H, LAYERS = 64, 4, 2


def tiny_model():
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    enc = dict(hidden_size=D, num_hidden_layers=LAYERS, num_attention_heads=H, intermediate_size=128,
               subsampling_conv_channels=16)
    cfg = CohereAsrConfig(encoder_config=enc, vocab_size=16384, hidden_size=D, num_hidden_layers=1,
                          num_attention_heads=H, intermediate_size=128)
    torch.manual_seed(0)
    m = CohereAsrForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
    assert m.model.encoder.config._attn_implementation == "sdpa"
    with torch.no_grad():  # non-trivial BN stats, so a BN left in train mode would show
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.running_mean.normal_(0, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
    return m


def padded_batch(mel_lengths=(300, 217, 90), dec_lengths=(7, 3, 5), seed=1):
    g = torch.Generator().manual_seed(seed)
    B, T = len(mel_lengths), max(mel_lengths) + 3
    amask = torch.arange(T)[None, :] < torch.tensor(mel_lengths)[:, None]
    feats = torch.randn(B, T, 128, generator=g) * amask[..., None]
    L = len(PROMPT) + max(dec_lengths)
    dec = torch.full((B, L), PAD, dtype=torch.long)
    dmask = torch.zeros(B, L, dtype=torch.long)
    for i, n in enumerate(dec_lengths):
        dec[i, : len(PROMPT)] = torch.tensor(PROMPT)
        dec[i, len(PROMPT): len(PROMPT) + n] = torch.randint(20, 16384, (n,), generator=g)
        dmask[i, : len(PROMPT) + n] = 1
    return dict(input_features=feats, attention_mask=amask, decoder_input_ids=dec, decoder_attention_mask=dmask)


def logits_of(model, batch):
    return model(**batch, use_cache=False).logits


def test_relpos_patch_parity_padded_batch():
    model = tiny_model()
    batch = padded_batch()
    valid = batch["decoder_attention_mask"].bool()
    for mode in ("eval", "train"):
        model.eval() if mode == "eval" else train_mode(model)
        w = torch.randn(3, batch["decoder_input_ids"].shape[1], 16384, generator=torch.Generator().manual_seed(9))

        model.zero_grad(set_to_none=True)
        ref = logits_of(model, batch)
        (ref * w * valid[..., None]).sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        unpatch = patch_relpos_once_per_batch(model)
        model.zero_grad(set_to_none=True)
        out = logits_of(model, batch)
        (out * w * valid[..., None]).sum().backward()
        unpatch()

        diff = (out - ref)[valid].abs().max().item()
        print(f"  {mode}: max |logit diff| {diff:.3g}")
        assert diff <= 1e-5
        assert torch.isfinite(out[valid]).all()
        grad_diff = max(((model.get_parameter(n).grad - g).abs().max() / g.abs().max().clamp_min(1e-12)).item()
                        for n, g in ref_grads.items())
        print(f"  {mode}: max relative grad diff {grad_diff:.3g}")
        assert grad_diff <= 1e-5
        assert model.model.encoder.layers[0].self_attn.relative_k_proj.weight.grad.abs().sum() > 0


def test_relpos_patch_bf16_autocast_and_grad_checkpointing():
    # the training configuration: bf16 autocast body, per-layer checkpointing that recomputes with the same
    # (stride-0) position tensor
    model = tiny_model()
    batch = padded_batch()
    valid = batch["decoder_attention_mask"].bool()
    train_mode(model)

    def run():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = logits_of(model, batch).float()
        out[valid].square().mean().backward()
        return out, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ref, ref_g = run()
    unpatch = patch_relpos_once_per_batch(model)
    out, g = run()
    assert torch.equal(out[valid], ref[valid])  # same bf16 ops element for element
    model.gradient_checkpointing_enable()
    out_ckpt, g_ckpt = run()
    unpatch()
    model.gradient_checkpointing_disable()
    assert torch.equal(out_ckpt[valid], ref[valid])
    for n, r in ref_g.items():
        scale = r.abs().max().clamp_min(1e-12)
        assert ((g[n] - r).abs().max() / scale).item() < 2e-2, n  # bf16 grads, different summation order
        assert ((g_ckpt[n] - g[n]).abs().max() / scale).item() < 1e-6, n


def test_relpos_patch_removes_batch_copies():
    from torch.utils.flop_counter import FlopCounterMode

    model = tiny_model().eval()
    batch = padded_batch()
    B = batch["input_features"].shape[0]

    def flops():
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            logits_of(model, batch)
        return fc.get_total_flops()

    before = flops()
    unpatch = patch_relpos_once_per_batch(model)
    after = flops()
    unpatch()
    T = int(model.model.encoder._get_subsampling_output_length(torch.tensor(batch["input_features"].shape[1])))
    P = 2 * T - 1
    saved_proj = LAYERS * (B - 1) * 2 * P * D * D  # relative_k_proj on (B-1) fewer copies
    saved_sin = (B - 1) * 2 * (D // 2) * P  # the (d/2 x 1) @ (1 x 2T-1) outer product for the sinusoids
    assert before - after == saved_proj + saved_sin
    assert flops() == before


def test_relpos_patch_idempotent_and_unpatch():
    model = tiny_model().eval()
    batch = padded_batch()
    with torch.no_grad():
        ref = logits_of(model, batch)
    h1 = patch_relpos_once_per_batch(model)
    h2 = patch_relpos_once_per_batch(model)
    assert h1 is h2 and is_relpos_patched(model)
    attn = model.model.encoder.layers[1].self_attn
    assert "forward" in attn.relative_k_proj.__dict__ and "forward" in model.model.encoder.encode_positions.__dict__
    with torch.no_grad():
        pos = model.model.encoder.encode_positions(torch.zeros(3, 5, D))
    assert pos.shape == (3, 9, D) and pos.stride(0) == 0
    with torch.no_grad():
        assert torch.equal(logits_of(model, batch), logits_of(model, batch))

    h1()
    h1()  # a second call is a no-op
    assert not is_relpos_patched(model)
    assert "forward" not in attn.relative_k_proj.__dict__
    assert "forward" not in model.model.encoder.encode_positions.__dict__
    with torch.no_grad():
        assert torch.equal(logits_of(model, batch), ref)
    h3 = patch_relpos_once_per_batch(model)  # patchable again after removal
    assert h3 is not h1
    h3()

    with pytest.raises(ValueError):
        patch_relpos_once_per_batch(nn.Linear(2, 2))


def test_relpos_patch_single_utterance_unchanged():
    model = tiny_model().eval()
    batch = {k: v[:1] for k, v in padded_batch().items()}
    with torch.no_grad():
        ref = logits_of(model, batch)
        unpatch = patch_relpos_once_per_batch(model)
        out = logits_of(model, batch)
        unpatch()
    assert torch.equal(out, ref)


def test_freeze_batchnorm_survives_train():
    model = tiny_model()
    n = freeze_batchnorm(model)
    assert n == LAYERS
    model.train()
    assert model.training and model.model.encoder.layers[0].feed_forward1.training
    assert assert_bn_frozen(model) == LAYERS
    model.eval()
    model.train(True)
    assert_bn_frozen(model)

    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    stats = [(m.running_mean.clone(), m.running_var.clone(), int(m.num_batches_tracked)) for m in bns]
    batch = padded_batch()
    logits_of(model, batch).sum().backward()
    for m, (mu, var, nbt) in zip(bns, stats):
        assert torch.equal(m.running_mean, mu) and torch.equal(m.running_var, var) and int(m.num_batches_tracked) == nbt
        assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0  # affine params still train

    unfreeze_batchnorm(model)
    model.train()
    assert all(m.training for m in bns)
    with pytest.raises(AssertionError):
        assert_bn_frozen(model)
    train_mode(model)
    assert_bn_frozen(model)
    model.train()
    assert_bn_frozen(model)


def test_sdpa_backend_report_cpu(monkeypatch):
    real = torch.nn.functional.scaled_dot_product_attention
    masks = []

    def spy(q, k, v, attn_mask=None, **kw):
        masks.append(attn_mask.detach().clone())
        return real(q, k, v, attn_mask=attn_mask, **kw)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    rep = sdpa_backend_report(device="cpu", dtype=torch.float32)
    assert rep["device"] == "cpu" and set(rep["enabled"]) == {"flash", "mem_efficient", "math", "cudnn"}
    assert rep["encoder_usable"]["math"] is True and rep["decoder_usable"]["math"] is True  # finite on masked rows
    assert "encoder_selected" in rep and "decoder_selected" in rep
    # the encoder case has fully masked query rows, as the model's valid_q & valid_k mask does on a padded row
    enc = [m for m in masks if m.is_floating_point()]
    assert enc and all(bool(torch.isneginf(m).all(dim=-1).any()) for m in enc)
    print(f"  {rep}")

    # a kernel that runs but gives NaN (as eager softmax does on a fully masked row) is not usable
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention",
                        lambda q, k, v, attn_mask=None, **kw: real(q, k, v, attn_mask=attn_mask, **kw) * float("nan"))
    rep = sdpa_backend_report(device="cpu", dtype=torch.float32)
    assert rep["encoder_usable"]["math"] is not True and rep["decoder_usable"]["math"] is not True
