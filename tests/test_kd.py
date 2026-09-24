"""kd_losses against a brute-force 17-bin reference built from full-vocab logits, and the L2-SP anchor. CPU only.

Teacher targets are built exactly as scripts/02_teacher_pass.py stores them: fp32 logits -> logsumexp -> topk ->
log-probs saved as fp16 (ids as int16)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from kitsune.kd import L2SP, W_CE, W_KL, kd_losses, kd_objective, module_key  # noqa: E402

V, K = 16384, 16
REAL_NPZ = ROOT / "teacher_out" / "reazon_small" / "train-00000.npz"


def teacher_logits(n: int, seed: int) -> torch.Tensor:
    """Realistic mixture: mostly peaked rows (p1 from ~0.5 to ~1, small tails) plus a few flat, high-tail rows."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, V, generator=g) * 2.0
    peak = torch.randint(0, V, (n,), generator=g)
    boost = 8.0 + 14.0 * torch.rand(n, generator=g)
    boost[: n // 8] = 0.0  # flat rows: the rest bucket holds most of the mass
    z[torch.arange(n), peak] += boost
    return z


def store_topk(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """What 02_teacher_pass writes, read back the way the trainer will (int64 ids, fp32 log-probs)."""
    sl = z.float()
    lse = torch.logsumexp(sl, dim=-1)
    v, i = torch.topk(sl, K, dim=-1)
    top_lp = (v - lse[:, None]).to(torch.float16)
    top_idx = i.to(torch.int16)
    return top_idx.long(), top_lp.float()


def reference(logits: torch.Tensor, top_idx: torch.Tensor, top_lp: torch.Tensor) -> dict[str, torch.Tensor]:
    """Brute force in float64: explicit 17-bin distributions from full-vocab softmax and an explicit tail mask."""
    q = torch.softmax(logits.double(), dim=-1)
    in_k = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, top_idx, True)
    q_k = q.gather(-1, top_idx)
    q_r = (q * ~in_k).sum(-1)
    p_k = top_lp.double().exp()
    s = p_k.sum(-1, keepdim=True)
    p_k = torch.where(s > 1, p_k / s, p_k)
    p_r = (1 - p_k.sum(-1)).clamp_min(0)
    P = torch.cat([p_k, p_r[:, None]], -1)
    Q = torch.cat([q_k, q_r[:, None]], -1)
    kl = (torch.xlogy(P, P) - P * torch.log(Q)).sum(-1)
    return dict(
        kl=kl,
        ce=-torch.log(q_k[:, 0]),
        top1_match=(q.argmax(-1) == top_idx[:, 0]).double(),
        student_entropy_coarse=-torch.xlogy(Q, Q).sum(-1),
        teacher_entropy_coarse=-torch.xlogy(P, P).sum(-1),
        student_tail=q_r,
        teacher_tail=p_r,
        teacher_p1=p_k[:, 0],
    )


def test_matches_bruteforce_reference():
    zt = teacher_logits(96, seed=0)
    top_idx, top_lp = store_topk(zt)
    g = torch.Generator().manual_seed(1)
    zs = zt + 1.5 * torch.randn(zt.shape, generator=g)  # a student near the teacher
    zs[-16:] = torch.randn(16, V, generator=g) * 3.0  # and some rows where it is far off
    got = kd_losses(zs, top_idx, top_lp)
    ref = reference(zs, top_idx, top_lp)
    assert set(got) == set(ref)
    for key, r in ref.items():
        g_ = got[key]
        assert g_.shape == (96,) and g_.dtype == torch.float32, key
        err = (g_.double() - r).abs().max().item()
        assert err < 1e-5, f"{key}: max abs err {err:.3g}"
    assert (got["kl"] >= -1e-6).all()
    assert (top_idx[:, 0] == zt.argmax(-1)).all()  # col 0 is the greedy token, as stored


def test_gradient_matches_autograd_of_reference():
    zt = teacher_logits(48, seed=2)
    top_idx, top_lp = store_topk(zt)
    zs = zt + 2.0 * torch.randn(zt.shape, generator=torch.Generator().manual_seed(3))

    x = zs.clone().requires_grad_(True)
    out = kd_losses(x, top_idx, top_lp)
    (W_KL * out["kl"].sum() + W_CE * out["ce"].sum()).backward()

    x64 = zs.double().requires_grad_(True)
    ref = reference(x64, top_idx, top_lp)
    (W_KL * ref["kl"].sum() + W_CE * ref["ce"].sum()).backward()
    assert (x.grad.double() - x64.grad).abs().max().item() < 1e-5  # fp32 vs fp64; gradient entries are O(1)

    # and the closed form of the KL gradient: q - p on the top-K, q * (1 - p_r/q_r) elsewhere
    x2 = zs.clone().requires_grad_(True)
    kd_losses(x2, top_idx, top_lp)["kl"].sum().backward()
    q = torch.softmax(zs.double(), -1)
    r = reference(zs, top_idx, top_lp)
    p_k = top_lp.double().exp()
    p_k = p_k / p_k.sum(-1, keepdim=True).clamp_min(1)
    analytic = q * (1 - (r["teacher_tail"] / r["student_tail"]))[:, None]
    analytic.scatter_(-1, top_idx, q.gather(-1, top_idx) - p_k)
    assert (x2.grad.double() - analytic).abs().max().item() < 1e-5


def test_kl_zero_when_student_equals_teacher():
    zt = teacher_logits(128, seed=4)
    top_idx, top_lp = store_topk(zt)

    # (a) a student whose 17-bin distribution IS the stored one: KL is 0 in exact arithmetic. In fp32 every log q
    # is z - lse(z) and inherits lse's rounding (|lse| up to ~25 here, half an ulp ~1e-6), which shifts KL by the
    # same amount because the target weights sum to 1; hence 5e-6, not 0.
    p_k = top_lp.exp()
    p_k = p_k / p_k.sum(-1, keepdim=True).clamp_min(1)
    p_r = (1 - p_k.sum(-1)).clamp_min(1e-30)
    zs = (p_r / (V - K)).log()[:, None].expand(-1, V).clone()  # the rest mass spread evenly over the tail
    zs.scatter_(-1, top_idx, p_k.log())
    kl_a = kd_losses(zs, top_idx, top_lp)["kl"]
    print(f"  stored-coarse student: max KL {kl_a.abs().max().item():.3g}")
    assert kl_a.abs().max().item() < 5e-6

    # (b) student logits == the teacher's own fp32 logits. The stored target differs from the student's exact
    # coarse distribution only by the fp16 rounding of each log-prob (<= half an ulp, |lp|*2^-12 relative) and,
    # where those rounded probabilities sum above 1, by clamping p_r to 0. KL is 0 at p == q with zero gradient,
    # so a perturbation d costs only O(d^2): KL(p~||q) <= chi^2(p~||q) = sum (p~ - q)^2 / q, which is ~1e-8 for
    # the rounding and at most q_r (the true tail, itself <= ~4e-4 when clamping triggers) for the tail bucket.
    out = kd_losses(zt, top_idx, top_lp)
    ref = reference(zt, top_idx, top_lp)
    q = torch.softmax(zt.double(), -1)
    Q = torch.cat([q.gather(-1, top_idx), ref["student_tail"][:, None]], -1)
    pk = top_lp.double().exp()
    pk = torch.where(pk.sum(-1, keepdim=True) > 1, pk / pk.sum(-1, keepdim=True), pk)
    P = torch.cat([pk, (1 - pk.sum(-1)).clamp_min(0)[:, None]], -1)
    chi2 = ((P - Q) ** 2 / Q).sum(-1)
    kl_b = out["kl"].double()
    clamped = int((out["teacher_tail"] == 0).sum())
    print(f"  student == teacher logits: max KL {kl_b.max().item():.3g}, mean {kl_b.mean().item():.3g}, "
          f"max chi2 bound {chi2.max().item():.3g}, rows with p_r clamped {clamped}/{len(kl_b)}")
    assert (kl_b <= chi2 + 5e-6).all()  # + the fp32 floor of (a)
    assert kl_b.max().item() < 5e-4 and kl_b.mean().item() < 5e-5
    assert (out["top1_match"] == 1).all()


def test_fp16_mass_above_one_is_safe():
    # a confident teacher whose rounded top-16 sums above 1 (p1 rounded up to exactly 1): p_r must clamp to 0,
    # p_k must be scaled down, and value and gradient must stay finite
    top_idx = torch.randperm(V, generator=torch.Generator().manual_seed(6))[:K].repeat(4, 1)
    top_lp = torch.full((4, K), -30.0)
    top_lp[:, 0] = 0.0
    top_lp[:, 1] = -9.0
    top_lp = top_lp.half().float()
    x = torch.randn(4, V).requires_grad_(True)
    out = kd_losses(x, top_idx, top_lp)
    assert (out["teacher_tail"] == 0).all()
    kd_objective(out, n_tokens=4).backward()
    assert torch.isfinite(out["kl"]).all() and torch.isfinite(x.grad).all()
    ref = reference(x.detach(), top_idx, top_lp)
    assert (out["kl"].double() - ref["kl"]).abs().max().item() < 1e-5


def test_objective_weights():
    out = dict(kl=torch.tensor([1.0, 2.0]), ce=torch.tensor([0.5, 0.5]))
    assert kd_objective(out, n_tokens=4).item() == pytest.approx((1.0 * 3.0 + 0.8 * 1.0) / 4)
    assert (W_KL, W_CE) == (1.0, 0.8)


@pytest.mark.skipif(not REAL_NPZ.exists(), reason="teacher_out shard not present")
def test_real_teacher_rows():
    z = np.load(REAL_NPZ)  # read-only
    n = 512
    top_idx = torch.from_numpy(z["topk_idx"][:n].astype(np.int64))
    top_lp = torch.from_numpy(z["topk_logprob"][:n].astype(np.float32))
    assert (top_idx[:, 0] == torch.from_numpy(z["tokens"][:n].astype(np.int64))).all()
    p_k = top_lp.exp()
    p_k = p_k / p_k.sum(-1, keepdim=True).clamp_min(1)
    p_r = (1 - p_k.sum(-1)).clamp_min(1e-30)
    zs = (p_r / (V - K)).log()[:, None].expand(-1, V).clone()
    zs.scatter_(-1, top_idx, p_k.log())
    out = kd_losses(zs, top_idx, top_lp)
    assert out["kl"].abs().max().item() < 5e-6  # fp32 floor, see test_kl_zero_when_student_equals_teacher
    assert torch.isfinite(out["teacher_entropy_coarse"]).all()


# --- L2-SP ---------------------------------------------------------------------------------------------------

def tiny_model(tie: bool = True):
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    enc = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
               subsampling_conv_channels=8)
    cfg = CohereAsrConfig(encoder_config=enc, vocab_size=512, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=4, intermediate_size=64, tie_word_embeddings=tie)
    torch.manual_seed(0)
    m = CohereAsrForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
    with torch.no_grad():  # the student init is saved in bf16, so theta_0 is exactly representable
        for p in m.parameters():
            p.copy_(p.bfloat16().float())
    m.model.decoder.pos_emb.weight.requires_grad_(False)  # frozen, as in training
    return m


def test_l2sp_value_apply_and_distances():
    m = tiny_model()
    assert m.proj_out.weight is m.model.decoder.embed_tokens.weight
    reg = L2SP(m.named_parameters(), lam=0.05)
    names = set(reg.names)
    assert "model.decoder.pos_emb.weight" not in names  # frozen -> excluded
    assert "proj_out.weight" not in names and "model.decoder.embed_tokens.weight" in names  # tied: counted once
    assert "proj_out.bias" in names
    assert all(r.dtype == torch.bfloat16 for r in reg.ref)
    assert reg.value().item() == 0.0

    theta0 = {n: p.detach().clone() for n, p in zip(reg.names, reg.params)}
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in reg.params:
            p.add_(0.01 * torch.randn(p.shape, generator=g))
    sq = sum(float(((p.detach().double() - theta0[n].double()) ** 2).sum()) for n, p in zip(reg.names, reg.params))
    assert reg.value().item() == pytest.approx(0.5 * 0.05 * sq, rel=1e-5)

    dist = reg.per_module_distance()
    for key in ("encoder.layers.0", "encoder.layers.1", "encoder.subsampling", "decoder.layers.0",
                "decoder.layers.1", "decoder.embed_tokens", "decoder.proj", "proj_out"):
        assert key in dist, key
    assert not any(k.startswith("decoder.pos_emb") for k in dist)
    assert sum(d**2 for d in dist.values()) == pytest.approx(sq, rel=1e-5)
    rel = reg.per_module_distance(relative=True)
    assert set(rel) == set(dist)
    assert np.isnan(rel["proj_out"])  # only proj_out.bias here, zero-initialised: relative distance undefined
    assert all(v > 0 for k, v in rel.items() if k != "proj_out")

    before = [p.detach().clone() for p in reg.params]
    lr = 1e-4
    reg.apply_(lr)
    for p, b, n in zip(reg.params, before, reg.names):
        expect = b - lr * 0.05 * (b - theta0[n])
        assert torch.allclose(p, expect, rtol=0, atol=1e-7), n

    reg.apply_(1.0 / 0.05 * 0.5)  # lr*lam = 0.5 halves every deviation
    sq_half = sum(float(((p.detach().double() - theta0[n].double()) ** 2).sum()) for n, p in zip(reg.names, reg.params))
    assert sq_half == pytest.approx(sq * (1 - 1e-4 * 0.05) ** 2 * 0.25, rel=1e-4)


def test_l2sp_exclude_and_state_roundtrip():
    m = tiny_model(tie=False)
    reg = L2SP(m.named_parameters(), lam=0.1, exclude=("bias_u", "bias_v"))
    assert not any("bias_u" in n or "bias_v" in n for n in reg.names)
    assert "proj_out.weight" in reg.names  # untied: its own parameter
    sd = reg.state_dict()
    with torch.no_grad():
        for p in reg.params:
            p.add_(1.0)
    reg2 = L2SP(m.named_parameters(), lam=0.1, exclude=("bias_u", "bias_v"))  # would anchor to the moved weights
    reg2.load_state_dict(sd)
    assert reg2.value().item() == pytest.approx(reg.value().item(), rel=1e-6)
    with pytest.raises(ValueError):
        L2SP(m.named_parameters(), lam=0.1).load_state_dict(sd)

    # the foreach chunks cover every parameter once, in order, within the element budget
    chunks = list(reg._chunks(budget=5000))
    assert len(chunks) > 3
    assert [id(p) for ps, _ in chunks for p in ps] == [id(p) for p in reg.params]
    assert all(sum(p.numel() for p in ps) <= 5000 or len(ps) == 1 for ps, _ in chunks)


def test_module_key():
    assert module_key("model.encoder.layers.12.self_attn.q_proj.weight") == "encoder.layers.12"
    assert module_key("model.decoder.layers.3.mlp.fc1.bias") == "decoder.layers.3"
    assert module_key("model.encoder.subsampling.layers.0.weight") == "encoder.subsampling"
    assert module_key("model.decoder.embed_tokens.weight") == "decoder.embed_tokens"
    assert module_key("proj_out.bias") == "proj_out"
