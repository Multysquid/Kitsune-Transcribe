"""CTC distillation loss (kitsune/ctc_kd.py) against independent computations, on CPU."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitsune import ctc_kd as K  # noqa: E402
from kitsune.ctc_targets import FrameTargets, collate_frame_targets, frame_targets, targets_from_log_probs  # noqa: E402
from kitsune.parakeet import ctc_collapse  # noqa: E402

BLANK = K.CTC_BLANK


def real_vocab_batch(seed=0, lengths=(40, 25, 7), k=8):
    """Synthetic teacher log-probs over the real vocab, the targets the label pass would store for them, and the
    teacher log-probs themselves (B, T, V) padded with 0."""
    from fixtures_ctc import synthetic_log_probs

    rng = np.random.default_rng(seed)
    lps = [synthetic_log_probs(n, rng) for n in lengths]
    T = max(lengths)
    teacher = torch.zeros(len(lengths), T, lps[0].shape[1])
    for b, lp in enumerate(lps):
        teacher[b, :len(lp)] = lp
    tg = targets_from_log_probs(teacher, torch.tensor(lengths), k=k)
    return teacher, tg


def test_constants():
    from kitsune import ctc_student as CS

    assert K.CTC_BLANK == CS.CTC_BLANK == 3072 and (K.W_KL, K.W_CTC) == (1.0, 0.8)


def test_student_equal_to_exact_targets_gives_zero_kl():
    """Targets stored without rounding (fp32 here) and a student equal to the teacher: every KL bin term is 0."""
    teacher, tg = real_vocab_batch()
    exact = []
    for b, t in enumerate(tg):  # same frames/indices, but the log-probs read back from the fp32 teacher
        lp = teacher[b, :t.n_frames]
        d = t.dense_frame.astype(np.int64)
        topk_lp = lp[d].gather(-1, torch.from_numpy(t.topk_idx.astype(np.int64))).numpy()
        exact.append(FrameTargets(t.n_frames, lp[:, BLANK].numpy(), t.dense_frame, t.topk_idx, topk_lp, t.ctc_ids))
    out = K.ctc_kd_losses(teacher, collate_frame_targets(exact))
    assert int(out["n_dense"]) > 10 and int(out["n_blank_frames"]) > 10
    assert abs(float(out["kl_dense"])) < 1e-4 and abs(float(out["kl_blank"])) < 1e-4


def test_kl_is_the_rest_bucket_term_when_the_bins_match():
    """Student bins = the teacher's bins scaled so that its rest mass is q_r instead of p_r: the 1+1-bin KL between
    (1 - p_r, p_r) and (1 - q_r, q_r) is all that is left."""
    V, k = 40, 3
    blank = V - 1
    g = torch.Generator().manual_seed(0)
    p_bins = torch.tensor([0.5, 0.2, 0.1])  # top-3 (blank not among them) ...
    p_blank = 0.15  # ... + blank: p_r = 0.05
    p_r = 1 - p_bins.sum() - p_blank
    for q_r in (0.05, 0.2, 0.001, 0.6):
        tail = torch.rand(V - 4, generator=g)
        q = torch.empty(V)
        idx = torch.tensor([5, 9, 2])
        q[idx] = p_bins * (1 - q_r) / (1 - p_r)
        q[blank] = p_blank * (1 - q_r) / (1 - p_r)
        rest = torch.tensor([i for i in range(V) if i not in (5, 9, 2, blank)])
        q[rest] = tail / tail.sum() * q_r
        t = FrameTargets(1, np.array([math.log(p_blank)], np.float32), np.array([0], np.int32),
                         idx.numpy()[None].astype(np.int16), p_bins.log().numpy()[None].astype(np.float32),
                         np.array([5], np.int32))
        out = K.ctc_kd_losses(q.log().double()[None, None], collate_frame_targets([t]), blank=blank)
        pr, qr = float(p_r), q_r
        want = (1 - pr) * math.log((1 - pr) / (1 - qr)) + pr * math.log(pr / qr)
        assert float(out["kl_dense"]) == pytest.approx(want, rel=1e-5, abs=1e-7), q_r
        assert float(out["kl_blank"]) == 0.0


def test_student_equal_to_the_stored_distribution_leaves_only_rounding_and_rest():
    """The real case: student = the teacher's fp32 distribution, targets as stored (fp16). Per frame the KL is the
    fp16 rounding of the bins plus the rest bucket: small and never negative beyond rounding."""
    teacher, tg = real_vocab_batch(seed=3, lengths=(60, 60))
    out = K.ctc_kd_losses(teacher, collate_frame_targets(tg), per_utt=True)
    kl = (out["kl_dense"] + out["kl_blank"]) / out["n_frames"]
    assert torch.all(kl < 2e-3) and torch.all(kl > -1e-5), kl


def test_dense_frame_bins_by_hand():
    """One dense frame with the blank outside the top-k, one with it inside: KL over top-k U {blank} + rest."""
    V, blank = 12, 11
    lp_t = torch.log_softmax(torch.tensor([[3.0, 2.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],
                                           [0.5, 0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0, 2.8]]), -1)
    top = lp_t.topk(3, -1)
    t = FrameTargets(2, lp_t[:, blank].numpy(), np.array([0, 1], np.int32), top.indices.numpy().astype(np.int16),
                     top.values.numpy(), np.array([0, 3], np.int32))
    assert blank not in top.indices[0].tolist() and blank in top.indices[1].tolist()
    lq = torch.log_softmax(torch.randn(2, V, generator=torch.Generator().manual_seed(1)), -1)
    got = K.ctc_kd_losses(lq[None], collate_frame_targets([t]), blank=blank, per_utt=True)
    want = 0.0
    for f in range(2):
        bins = sorted(set(top.indices[f].tolist()) | {blank})
        p = lp_t[f, bins].exp()
        q = lq[f, bins].exp()
        p_r, q_r = 1 - p.sum(), 1 - q.sum()
        want += float((p * (p / q).log()).sum() + p_r * (p_r / q_r).log())
    assert float(got["kl_dense"][0]) == pytest.approx(want, rel=1e-5)


def test_blank_frames_two_bin_kl_by_hand():
    V, blank = 10, 9
    p_b = torch.tensor([0.99, 0.96, 1.0])
    t = FrameTargets(3, p_b.log().numpy(), np.zeros(0, np.int32), np.zeros((0, 4), np.int16),
                     np.zeros((0, 4), np.float32), np.zeros(0, np.int32))
    lq = torch.log_softmax(torch.randn(3, V, generator=torch.Generator().manual_seed(2)) * 2, -1)
    got = K.ctc_kd_losses(lq[None], collate_frame_targets([t]), blank=blank)
    q_b = lq[:, blank].exp()
    want = sum(float(torch.xlogy(p, p) - p * torch.log(q) + torch.xlogy(1 - p, 1 - p) - (1 - p) * torch.log(1 - q))
               for p, q in zip(p_b, q_b))
    assert float(got["kl_blank"]) == pytest.approx(want, rel=1e-5)
    assert int(got["n_blank_frames"]) == 3 and int(got["n_dense"]) == 0
    # the not-blank mass stays exact where p(blank) rounds to 1 in fp32
    z = torch.zeros(1, 1, V)
    z[..., blank] = 40.0
    lq = z - z.logsumexp(-1, keepdim=True)
    t1 = FrameTargets(1, np.array([-1e-6], np.float32), np.zeros(0, np.int32), np.zeros((0, 4), np.int16),
                      np.zeros((0, 4), np.float32), np.zeros(0, np.int32))
    out = K.ctc_kd_losses(lq, collate_frame_targets([t1]), blank=blank)
    assert math.isfinite(float(out["kl_blank"])) and float(out["kl_blank"]) > 0


def test_gradcheck_tiny():
    V, blank, k = 7, 6, 3
    g = torch.Generator().manual_seed(0)
    teacher = torch.log_softmax(torch.randn(2, 5, V, generator=g, dtype=torch.float64) * 3, -1)
    teacher[0, 1, blank] = 10.0  # a blank-only frame
    teacher = torch.log_softmax(teacher, -1)
    tg = targets_from_log_probs(teacher, torch.tensor([5, 3]), k=k)
    batch = collate_frame_targets(tg)
    assert int(batch["dense_mask"].sum()) > 0 and int((batch["frame_mask"] & ~batch["dense_mask"]).sum()) > 0
    z = torch.randn(2, 5, V, generator=g, dtype=torch.float64, requires_grad=True)

    def f(z):
        losses = K.ctc_kd_losses(torch.log_softmax(z, -1), batch, blank=blank)
        return K.ctc_kd_objective(losses, losses["n_tokens"])

    assert torch.autograd.gradcheck(f, (z,), eps=1e-6, atol=1e-5)


def brute_force_ctc(lp: torch.Tensor, target: list[int], blank: int) -> float:
    """-log sum over every frame path that collapses to `target` of prod p (exponential; tiny T and V only)."""
    T, V = lp.shape
    tot = 0.0
    for path in itertools.product(range(V), repeat=T):
        if ctc_collapse(list(path), blank) == target:
            tot += math.exp(sum(float(lp[t, c]) for t, c in enumerate(path)))
    return -math.log(tot)


def test_ctc_sum_reduction_equals_brute_force():
    V, blank = 4, 3
    g = torch.Generator().manual_seed(5)
    lp = torch.log_softmax(torch.randn(3, 5, V, generator=g, dtype=torch.float64), -1)
    lengths, tgts = [5, 4, 5], [[0, 1], [2, 2], []]
    rows = []
    for n, tg in zip(lengths, tgts):
        rows.append(FrameTargets(n, np.full(n, -0.01, np.float32), np.zeros(0, np.int32), np.zeros((0, 2), np.int16),
                                 np.zeros((0, 2), np.float32), np.array(tg, np.int32)))
    out = K.ctc_kd_losses(lp, collate_frame_targets(rows), blank=blank)
    want = sum(brute_force_ctc(lp[b, :n], tg, blank) for b, (n, tg) in enumerate(zip(lengths, tgts)))
    assert float(out["ctc"]) == pytest.approx(want, rel=1e-9)
    assert int(out["n_tokens"]) == 4
    per = K.ctc_kd_losses(lp, collate_frame_targets(rows), blank=blank, per_utt=True)
    assert float(per["ctc"][1]) == pytest.approx(brute_force_ctc(lp[1, :4], [2, 2], blank), rel=1e-9)


def test_ctc_zero_infinity_on_an_infeasible_target():
    V, blank = 5, 4
    lp = torch.log_softmax(torch.randn(1, 3, V), -1)
    t = FrameTargets(3, np.full(3, -0.01, np.float32), np.zeros(0, np.int32), np.zeros((0, 2), np.int16),
                     np.zeros((0, 2), np.float32), np.array([1, 1, 1], np.int32))  # needs 5 frames
    out = K.ctc_kd_losses(lp, collate_frame_targets([t]), blank=blank)
    assert float(out["ctc"]) == 0.0


def test_objective_normalises_by_the_steps_tokens():
    """Micro-batch objectives divided by the step's N_u add up to the whole step's objective."""
    teacher, tg = real_vocab_batch(seed=1, lengths=(30, 18, 44, 9))
    noise = torch.randn(teacher.shape, generator=torch.Generator().manual_seed(4))
    student = torch.log_softmax(teacher + 0.7 * noise, -1)
    whole = K.ctc_kd_losses(student, collate_frame_targets(tg))
    n_step = int(whole["n_tokens"])
    assert n_step == sum(len(t.ctc_ids) for t in tg) > 0
    parts = [K.ctc_kd_losses(student[i:j], collate_frame_targets(tg[i:j])) for i, j in ((0, 1), (1, 4))]
    got = sum(K.ctc_kd_objective(p, n_step) for p in parts)
    want = (whole["kl_dense"] + whole["kl_blank"] + 0.8 * whole["ctc"]) / n_step
    assert float(got) == pytest.approx(float(K.ctc_kd_objective(whole, n_step)), rel=1e-6)
    assert float(got) == pytest.approx(float(want), rel=1e-6)
    w = K.ctc_kd_objective(whole, torch.tensor(n_step), w_kl=0.5, w_ctc=2.0)
    kl = whole["kl_dense"] + whole["kl_blank"]
    assert float(w) == pytest.approx(float((0.5 * kl + 2.0 * whole["ctc"]) / n_step))
    assert float(K.ctc_kd_objective(whole, 0)) == pytest.approx(float(K.ctc_kd_objective(whole, 1)))


def test_per_utterance_values_sum_to_the_totals():
    teacher, tg = real_vocab_batch(seed=2)
    student = torch.log_softmax(teacher + torch.randn(teacher.shape, generator=torch.Generator().manual_seed(9)), -1)
    batch = collate_frame_targets(tg)
    tot, per = K.ctc_kd_losses(student, batch), K.ctc_kd_losses(student, batch, per_utt=True)
    for key in tot:
        assert per[key].shape == (3,), key
        assert float(per[key].sum()) == pytest.approx(float(tot[key]), rel=1e-5), key
    assert per["n_frames"].tolist() == [40, 25, 7]


def test_frame_metrics_count_valid_frames_only():
    teacher, tg = real_vocab_batch(seed=4)
    batch = collate_frame_targets(tg, t_max=50)
    out = K.ctc_kd_losses(torch.nn.functional.pad(teacher, (0, 0, 0, 10)), batch)
    n = int(out["n_frames"])
    assert n == 72 == int(out["n_dense"] + out["n_blank_frames"])
    assert int(out["argmax_agree"]) == n  # the student IS the teacher
    col0 = np.concatenate([t.col0() for t in tg])
    assert int(out["teacher_blank"]) == int(out["argmax_blank"]) == int((col0 == BLANK).sum())
    assert 0 < int(out["teacher_blank"]) < n
    blank_everywhere = torch.full_like(teacher, -20.0)
    blank_everywhere[..., BLANK] = 0.0
    out = K.ctc_kd_losses(torch.log_softmax(blank_everywhere, -1), collate_frame_targets(tg))
    assert int(out["argmax_blank"]) == n and int(out["argmax_agree"]) == int(out["teacher_blank"])


def test_padded_frames_never_enter_the_losses_or_the_gradients():
    teacher, tg = real_vocab_batch(seed=5)
    z = (teacher + torch.randn(teacher.shape, generator=torch.Generator().manual_seed(3))).requires_grad_()
    lp = torch.log_softmax(z, -1)
    fm = collate_frame_targets(tg)["frame_mask"]
    poisoned = torch.where(fm[..., None], lp, torch.full_like(lp, float("nan")))
    a = K.ctc_kd_losses(lp, collate_frame_targets(tg))
    b = K.ctc_kd_losses(poisoned, collate_frame_targets(tg))
    for key in ("kl_dense", "kl_blank", "ctc", "argmax_agree", "argmax_blank"):
        assert torch.equal(a[key], b[key]), key
    for losses in (a, b):  # the clean and the poisoned graph: no gradient reaches a padded frame
        z.grad = None
        K.ctc_kd_objective(losses, losses["n_tokens"]).backward(retain_graph=True)
        assert torch.isfinite(z.grad).all()
        assert float(z.grad[~fm].abs().max()) == 0.0 and float(z.grad[fm].abs().max()) > 0


def test_frame_axis_is_matched_to_the_student():
    teacher, tg = real_vocab_batch(seed=6)
    ref = K.ctc_kd_losses(teacher, collate_frame_targets(tg))
    wide = K.ctc_kd_losses(teacher, collate_frame_targets(tg, t_max=64))  # targets padded past the student
    longer = K.ctc_kd_losses(torch.nn.functional.pad(teacher, (0, 0, 0, 5)), collate_frame_targets(tg))
    for key in ("kl_dense", "kl_blank", "ctc", "n_frames"):
        assert torch.allclose(ref[key], wide[key]) and torch.allclose(ref[key], longer[key]), key
    with pytest.raises(ValueError, match="past the student"):
        K.ctc_kd_losses(teacher[:, :30], collate_frame_targets(tg))


def test_frame_targets_from_arrays_derive_the_ctc_path():
    teacher, _ = real_vocab_batch(seed=7, lengths=(33,))
    lp = teacher[0]
    top = lp.topk(8, -1)
    dense = np.nonzero((lp[:, BLANK] < math.log(0.95)).numpy())[0]
    ft = frame_targets(33, lp[:, BLANK].numpy(), dense, top.indices[dense].numpy(), top.values[dense].numpy())
    assert ft.blank_lp.dtype == np.float16 and ft.dense_frame.dtype == np.int32 and ft.topk_idx.dtype == np.int16
    assert ft.topk_lp.dtype == np.float16 and ft.ctc_ids.dtype == np.int32
    assert ft.ctc_ids.tolist() == ctc_collapse(lp.argmax(-1).tolist(), BLANK)


def test_overfit_ten_utterances_on_cpu():
    """A tiny random ParakeetForCTC trained on 10 real-length utterances of synthetic audio against synthetic frame
    targets aligned to them: the objective falls and the greedy output reaches the targets."""
    from fixtures_ctc import synthetic_frame_targets, tiny_ctc_model

    from kitsune import ctc_student as CS
    from kitsune.features import LogMel

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    from transformers import ParakeetFeatureExtractor

    feats = CS.CtcFeatures(LogMel.from_feature_extractor(ParakeetFeatureExtractor()))
    n_samples = rng.integers(8000, 40000, size=10)
    waves = [(0.1 * np.sin(np.arange(n) / 16000 * 2 * np.pi * rng.uniform(100, 2000)) +
              0.05 * rng.standard_normal(n)).astype(np.float32) for n in n_samples]
    tg = [synthetic_frame_targets(CS.expected_n_frames(n), rng, tokens=(3, 13)) for n in n_samples]
    f, fl = feats(waves)
    mask = CS.lengths_to_mask(fl, f.shape[1])
    model = tiny_ctc_model(seed=0, n_layers=2, ffn=128, d=64, heads=4)
    model.train()
    batch = collate_frame_targets(tg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def token_error() -> tuple[float, int]:
        model.eval()
        with torch.no_grad():
            lp, n = CS.ctc_log_probs(model, f, mask)
        model.train()
        hyp = CS.greedy_ctc_ids(lp, n)
        edits = sum(_edits(h, t.ctc_ids.tolist()) for h, t in zip(hyp, tg))
        return edits / sum(len(t.ctc_ids) for t in tg), sum(h == t.ctc_ids.tolist() for h, t in zip(hyp, tg))

    err0, _ = token_error()
    history = []
    for step in range(250):  # measured: token error 1.0 -> 0.28 at step 150, 0.03 at 200, 0 at 250
        lp, n = CS.ctc_log_probs(model, f, mask)
        assert n.tolist() == [t.n_frames for t in tg]
        losses = K.ctc_kd_losses(lp, batch)
        loss = K.ctc_kd_objective(losses, losses["n_tokens"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    err, exact = token_error()
    assert history[-1] < 0.05 * history[0], (history[0], history[-1])
    assert err0 > 0.9 and err <= 0.05 and exact >= 8, (err0, err, exact)


def _edits(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[:], i
        for j, y in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return d[-1]
