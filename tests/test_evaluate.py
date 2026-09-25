"""kitsune.evaluate: CER conventions, the pre-registered verdict, the teacher baselines, and both evals on a tiny random
CohereAsr model over a real (synthetic) trainset store. CPU only; the teacher tokenizer is used from the HF cache if
present (HF_HUB_OFFLINE), otherwise a stand-in."""
import copy
import json
import os
import re
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from kitsune import evaluate as ev  # noqa: E402
from kitsune.audio import decode_audio  # noqa: E402
from kitsune.text import cer as utt_cer  # noqa: E402
from fixtures import REAL, need_real  # noqa: E402

TEACHER_OUT = REAL / "teacher_out"
EVAL = ["eval_jsut", "eval_cv8"]  # fake corpus sources named like the real sets


# ------------------------------------------------------------------------------------------------------------ CER


def test_corpus_cer_vs_per_utt_mean():
    refs = ["あいうえお", "かき", "さしすせそたちつてと"]
    hyps = ["あいうえお", "かく", "さしすせそたちつてと"]
    per = [utt_cer(h, r) for h, r in zip(hyps, refs)]
    assert per == [0.0, 0.5, 0.0]
    c = ev.corpus_cer(hyps, refs)
    assert c["edits"] == 1 and c["ref_chars"] == 17 and c["n"] == 3
    assert c["cer"] == pytest.approx(1 / 17)  # vs the per-utterance mean 1/6: short utterances dominate the mean
    assert np.mean(per) == pytest.approx(1 / 6)
    # insertions count against the reference length; deletion of a whole utterance counts its chars
    c = ev.corpus_cer(["あいうえおか", ""], ["あいうえお", "かき"])
    assert (c["edits"], c["ref_chars"]) == (3, 7)


def test_corpus_cer_normalisation_and_empty_refs():
    # NFKC + lowercase + punctuation/space stripping, exactly kitsune.text.normalize_ja
    c = ev.corpus_cer(["ABC、です。", "えー"], ["ａｂｃ です！", "…。"])
    assert c["cer"] == 0.0 and c["ref_chars"] == 5
    assert c["n"] == 1 and c["n_empty_ref"] == 1  # the punctuation-only reference is skipped, and counted
    assert np.isnan(ev.corpus_cer(["x"], ["、"])["cer"])


# -------------------------------------------------------------------------------------------------------- verdict


def final(ratios, trunc=(0, 0, 0), n=1000):
    """Final full-set greedy summary with the teacher at its pre-registered CER on the same ids."""
    sets = {}
    for s, r, k in zip(ev.GATE_SETS, ratios, trunc):
        t = ev.TEACHER_CER_PREREG[s]
        sets[s] = dict(cer_ref_corpus=r * t, teacher_cer_ref_corpus=t, n=n, n_truncated=k, trunc_rate=k / n)
    return dict(sets=sets)


def history(cer_ratio, heldout, probe, steps=None):
    steps = steps or [100 * i for i in range(len(cer_ratio))]
    out = []
    for st, c, h, p in zip(steps, cer_ratio, heldout, probe):
        g = {s: dict(cer_ref_corpus=c * t, teacher_cer_ref_corpus=t, trunc_rate=0.0, n=500)
             for s, t in ev.TEACHER_CER_PREREG.items()}
        out.append(dict(step=st, elapsed_s=st * 1.0, heldout_kl=h, probe_kl=p, greedy=g))
    return out


N = 12
FALLING = list(np.linspace(2.0, 1.1, N))  # CER ratio still improving at the end
FLAT = [1.3] * N
KL_DOWN = list(np.linspace(0.8, 0.30, N))
PROBE_DOWN = list(np.linspace(0.7, 0.25, N))  # gap 0.1 -> 0.05: not widening


def test_verdict_go():
    v = ev.verdict(dict(final=final((1.1, 1.15, 1.6), trunc=(1, 2, 0)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "GO", v["reasons"]
    assert v["n_sets_go"] == 2 and v["trunc_ok"] and v["trend"]["gap_widening"] is False
    assert v["sets"]["eval_jsut"]["threshold_go"] == pytest.approx(1.2 * 0.0830)
    assert v["sets"]["eval_jsut"]["teacher_source"] == "same ids"
    assert len(v["trend"]["window_steps"]) == 3  # max(3, ceil(0.2 * 12))


def test_verdict_truncation_blocks_go():
    v = ev.verdict(dict(final=final((1.1, 1.15, 1.6), trunc=(10, 5, 5)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "PROMISING" and not v["trunc_ok"]  # 20/3000 = 0.67 % > 0.5 %, still improving
    assert any("truncation" in r for r in v["reasons"])


def test_verdict_widening_gap_blocks_go():
    probe = list(np.linspace(0.7, 0.10, N))  # probe falls much faster than held-out: gap widens (no overfit yet)
    v = ev.verdict(dict(final=final((1.1, 1.1, 1.1)), history=history(FLAT, KL_DOWN, probe)))
    assert v["trend"]["gap_widening"] is True and not v["trend"]["overfit"]
    assert v["verdict"] == "INCONCLUSIVE"  # within 1.5x but CER flat: not a pre-registered tier


def test_verdict_promising():
    v = ev.verdict(dict(final=final((1.3, 1.4, 1.45)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "PROMISING" and v["trend"]["improving"] and v["n_sets_go"] == 0


def test_verdict_nogo_flat():
    v = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=history([1.9] * N, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "NO-GO" and v["n_sets_promising"] == 1 and not v["trend"]["improving"]


def test_verdict_overfit_is_nogo_even_when_cer_passes():
    heldout = list(np.linspace(0.30, 0.45, N))  # held-out KL rising ...
    probe = list(np.linspace(0.30, 0.10, N))  # ... while the probe keeps falling
    v = ev.verdict(dict(final=final((1.0, 1.0, 1.0)), history=history(FALLING, heldout, probe)))
    assert v["verdict"] == "NO-GO" and v["trend"]["overfit"]
    assert any("over-fitting" in r for r in v["reasons"])


def test_verdict_inconclusive_and_teacher_override():
    v = ev.verdict(dict(final=final((1.8, 1.9, 2.0)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "INCONCLUSIVE"  # beyond 1.5x but improving: pre-registration is silent
    # results["teacher"] overrides the same-ids teacher CER; the drift vs the pre-registered number is reported
    fin = final((1.1, 1.1, 1.1))
    v = ev.verdict(dict(final=fin, history=history(FALLING, KL_DOWN, PROBE_DOWN),
                        teacher={"eval_jsut": 0.0831}))
    assert v["sets"]["eval_jsut"]["teacher_source"] == "results.teacher"
    assert v["sets"]["eval_jsut"]["baseline_drift"] == pytest.approx(0.0001)
    # no history at all: trends unknown, so GO cannot be certified
    v = ev.verdict(dict(final=final((1.0, 1.0, 1.0)), history=[]))
    assert v["verdict"] == "INCONCLUSIVE" and v["trend"]["gap_widening"] is None


def test_verdict_trends_leave_out_the_untrained_step_0_eval():
    """Per-epoch evals leave short histories: step 0, an epoch end or two, the final eval. The untrained student's
    step-0 record (CER ratio ~30, held-out KL ~5) must not enter the trends, or it makes every run 'improving' and hides
    over-fitting; the verdict is the one the trained records give."""
    steps = [0, 1126, 2252]
    # over-fitting on the trained records: held-out KL +10 % while the probe falls, CER within 1.2x
    for probe0 in (4.5, 5.75):  # the step-0 probe KL below or above the step-0 held-out KL
        v = ev.verdict(dict(final=final((1.1, 1.1, 1.15)),
                            history=history([30, 1.12, 1.10], [4.87, 0.60, 0.66], [probe0, 0.62, 0.45], steps)))
        assert v["verdict"] == "NO-GO" and v["trend"]["overfit"], v["reasons"]
        assert v["trend"]["window_steps"] == [1126, 2252]
    # a flat tail with 1/3 sets within 1.5x is the pre-registered NO-GO, not 'still improving'
    v = ev.verdict(dict(final=final((1.8, 2.0, 1.4)),
                        history=history([30, 1.9, 1.9], [4.87, 0.50, 0.45], [4.5, 0.45, 0.40], steps)))
    assert v["verdict"] == "NO-GO" and not v["trend"]["improving"], v["reasons"]
    # ... and one with 2/3 sets within 1.5x is outside the tiers ('CER not improving'), not PROMISING
    v = ev.verdict(dict(final=final((1.3, 1.4, 1.6)),
                        history=history([30, 1.3, 1.3], [4.87, 0.50, 0.45], [4.5, 0.45, 0.40], steps)))
    assert v["verdict"] == "INCONCLUSIVE" and not v["trend"]["improving"], v["reasons"]
    assert v["trend"]["window_steps"] == [1126, 2252] and any("CER not improving" in r for r in v["reasons"])
    # a GO on the trained records stays GO whatever the step-0 numbers were
    for probe0 in (4.5, 5.75):
        v = ev.verdict(dict(final=final((1.1, 1.1, 1.15)),
                            history=history([30, 1.25, 1.10], [4.87, 0.70, 0.60], [probe0, 0.62, 0.53], steps)))
        assert v["verdict"] == "GO", v["reasons"]
    # one trained record: the trend is unknown, not 'improving' (from step 0) and not 'flat'
    for ratios in ((1.3, 1.4, 1.45), (1.8, 2.0, 1.4)):
        v = ev.verdict(dict(final=final(ratios), history=history([30, 1.3], [4.87, 0.6], [4.5, 0.55], [0, 1126])))
        assert v["verdict"] == "INCONCLUSIVE" and v["trend"]["cer_ratio_rel_change"] is None, v["reasons"]
        assert v["trend"]["window_steps"] == [1126] and any("CER trend unknown" in r for r in v["reasons"])
        assert not any("still improving" in r or "CER flat" in r for r in v["reasons"])


def test_verdict_reports_undecodable_gate_rows():
    """A gate set some of whose audio did not decode is judged on the rows that did (the teacher on the same ids): the
    tier stays the same, but per_set carries n and n_bad_audio and a reason says it is not the pre-registered full set
    (a monitor-only set's drops are not a gate matter)."""
    fin = final((1.1, 1.15, 1.6), n=3810)
    base = ev.verdict(dict(final=fin, history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    v = ev.verdict(dict(final=dict(fin, n_bad_audio=675, bad_audio_per_set={"eval_cv8": 673, "eval_emilia": 2}),
                        history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == base["verdict"] == "GO" and v["n_sets_go"] == base["n_sets_go"]
    assert v["sets"]["eval_cv8"]["n"] == 3810 and v["sets"]["eval_cv8"]["n_bad_audio"] == 673
    assert v["sets"]["eval_jsut"]["n"] == 3810 and v["sets"]["eval_jsut"]["n_bad_audio"] == 0
    assert [r for r in v["reasons"] if "undecodable" in r] == [
        "eval_cv8: judged on 3810 decoded rows, 673 undecodable (not the pre-registered full set)"]
    assert not any("undecodable" in r for r in base["reasons"])
    # a gate set none of whose rows decoded has no final result at all; the reason says why
    sets = {s: d for s, d in fin["sets"].items() if s != "eval_reazon"}
    v = ev.verdict(dict(final=dict(sets=sets, bad_audio_per_set={"eval_reazon": 5}), history=[]))
    assert "eval_reazon: no final greedy result (5 undecodable rows)" in v["reasons"]


# ----------------------------------------------------------------------------------------------- verdict v2


def history2(points):
    """Records as the trainer now writes them. points: (step, epoch, lr_phase, subset ratio, complete ratio or None,
    held-out KL, probe KL); the ratio is the same on every gate set, the teacher at its pre-registered CER."""
    out = []
    for st, ep, ph, sub, full, h, p in points:
        def sets(r):
            return {s: dict(cer_ref_corpus=r * t, teacher_cer_ref_corpus=t, trunc_rate=0.0, n=500)
                    for s, t in ev.TEACHER_CER_PREREG.items()}
        rec = dict(step=st, elapsed_s=st * 1.0, heldout_kl=h, probe_kl=p, greedy=sets(sub))
        if ph is not None:
            rec["lr_phase"] = ph
        if ep is not None:
            rec["epoch"] = ep
        if full is not None:
            rec["greedy_full"] = sets(full)
        out.append(rec)
    return out


def wsd_history(ratios, cooldown_from, full=None, kl=None, probe=None):
    """One eval per epoch (step = 100 x epoch), epoch 1..n; the evals from epoch `cooldown_from` on follow a cooldown
    step. full: the complete sets' ratios (None: a subset-only history)."""
    n = len(ratios)
    kl = kl or list(np.linspace(0.8, 0.3, n))
    probe = probe or list(np.linspace(0.7, 0.25, n))
    return history2([(100 * e, float(e), "cooldown" if e >= cooldown_from else "stable", ratios[e - 1],
                      None if full is None else full[e - 1], kl[e - 1], probe[e - 1]) for e in range(1, n + 1)])


def test_verdict_v2_measures_improving_before_the_cooldown():
    """Flat at the peak LR, then the cooldown's one-off drop: v1 reads the drop ('still improving', INCONCLUSIVE, as the
    first A100 run), v2 the flat pre-cooldown evals (the pre-registered NO-GO). The tiers and per-set numbers are the
    same; v2 reports the cooldown gain and the pre-cooldown slope next to them."""
    ratios = [2.6, 2.3, 2.1, 2.0, 1.95, 1.95, 1.95, 1.95, 1.95, 1.8, 1.7, 1.65]
    hist, fin = wsd_history(ratios, cooldown_from=10), final((1.8, 2.0, 1.4))
    v1 = ev.verdict(dict(final=fin, history=hist))
    assert v1["verdict"] == "INCONCLUSIVE" and v1["trend"]["improving"] and v1["trend"]["window_steps"] == [1000, 1100,
                                                                                                             1200]
    v2 = ev.verdict(dict(final=fin, history=hist), version=2)
    assert v2["verdict"] == "NO-GO" and v2["version"] == 2 and not v2["trend"]["improving"], v2["reasons"]
    assert v2["sets"] == v1["sets"] and v2["n_sets_promising"] == v1["n_sets_promising"] == 1
    t = v2["trend"]
    assert t["cer_window_steps"] == [700, 800, 900] and t["n_pre_cooldown"] == 9  # max(3, ceil(0.2 * 9))
    assert t["cer_ratio_rel_change"] == pytest.approx(0.0, abs=1e-12) and t["cer_scope"] == "subset"
    assert t["window_steps"] == [1000, 1100, 1200]  # over-fitting and the KL gap: the end of the run, as in v1
    assert any(re.fullmatch(r"CER trend [+-]0\.0% over the pre-cooldown evals at steps \[700, 800, 900\] \(the fixed "
                            r"greedy subset\)", r) for r in v2["reasons"]), v2["reasons"]
    g = v2["cooldown_gain"]
    assert (g["from_step"], g["to_step"], g["from_epoch"], g["to_epoch"]) == (900, 1200, 9.0, 12.0)
    assert g["scope"] == "subset"
    assert (g["cer_ratio_before"], g["cer_ratio_after"]) == pytest.approx((1.95, 1.65))
    assert g["cer_ratio_rel_change"] == pytest.approx(1.65 / 1.95 - 1)
    assert g["sets"]["eval_jsut"] == pytest.approx(dict(cer_before=1.95 * 0.0830, cer_after=1.65 * 0.0830))
    s = v2["pre_cooldown_slope"]
    assert s["steps"] == [700, 800, 900] and s["epochs"] == [7.0, 8.0, 9.0] and s["per_epoch"] == pytest.approx(0.0)
    assert v2["thresholds"] == dict(v1["thresholds"], min_epoch_gap=0.25)
    # a pre-cooldown slope that still falls: PROMISING under both when 2 sets are within 1.5x
    ratios = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2]
    v2 = ev.verdict(dict(final=final((1.3, 1.4, 1.6)), history=wsd_history(ratios, cooldown_from=10)), version=2)
    assert v2["verdict"] == "PROMISING" and v2["trend"]["improving"]
    s = v2["pre_cooldown_slope"]
    assert s["per_epoch"] == pytest.approx(-0.05) and s["rel_per_epoch"] == pytest.approx(-0.05 / 1.4)
    assert s["rel_change"] == v2["trend"]["cer_ratio_rel_change"] == pytest.approx(-0.1 / 1.4)


def test_verdict_v2_needs_three_pre_cooldown_evals():
    """Fewer than min_points (3) evals at the peak LR: the trend is unknown - neither PROMISING nor the flat NO-GO -
    and the reason says so; over-fitting and the KL gap still read the end of the run."""
    ratios = [1.6, 1.5, 1.4, 1.35, 1.3]
    hist = wsd_history(ratios, cooldown_from=3)
    v1 = ev.verdict(dict(final=final((1.3, 1.4, 1.6)), history=hist))
    assert v1["verdict"] == "PROMISING"  # v1: the cooldown's drop counts as improving
    v2 = ev.verdict(dict(final=final((1.3, 1.4, 1.6)), history=hist), version=2)
    assert v2["verdict"] == "INCONCLUSIVE" and v2["trend"]["cer_ratio_rel_change"] is None
    assert "trend: insufficient pre-cooldown evals (2 after de-duplication, 3 needed)" in v2["reasons"]
    assert v2["trend"]["cer_window_steps"] == [] and v2["trend"]["cer_scope"] is None
    assert v2["pre_cooldown_slope"] is None and v2["cooldown_gain"]["from_step"] == 200
    assert v2["trend"]["window_steps"] == [300, 400, 500] and v2["trend"]["gap_widening"] is False
    v2 = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=hist), version=2)
    assert v2["verdict"] == "INCONCLUSIVE"  # not the flat NO-GO either: the trend is not measured
    # a GO needs no CER trend, and over-fitting is NO-GO whatever the pre-cooldown evals
    v2 = ev.verdict(dict(final=final((1.1, 1.1, 1.3)), history=hist), version=2)
    assert v2["verdict"] == "GO"
    rising = wsd_history(ratios, 3, kl=[0.30, 0.31, 0.33, 0.36, 0.40], probe=[0.30, 0.25, 0.20, 0.15, 0.10])
    v2 = ev.verdict(dict(final=final((1.1, 1.1, 1.3)), history=rising), version=2)
    assert v2["verdict"] == "NO-GO" and v2["trend"]["overfit"]
    # warm-up evals and evals without an LR phase are not pre-cooldown evals
    hist = wsd_history([1.5] * 6, cooldown_from=7)
    hist[0]["lr_phase"] = "warmup"
    del hist[1]["lr_phase"]
    v2 = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=hist), version=2)
    assert v2["trend"]["n_pre_cooldown"] == 4 and v2["trend"]["cer_window_steps"] == [400, 500, 600]
    assert "1 evals without an LR phase are left out of the CER trend" in v2["reasons"]
    assert v2["cooldown_gain"] is None  # the run never reached its cooldown
    with pytest.raises(ValueError, match="version"):
        ev.verdict(dict(final=final((1.0, 1.0, 1.0)), history=[]), version=3)


def test_verdict_v2_uses_the_complete_sets_when_every_eval_in_its_window_has_them():
    """The complete sets (greedy_full) when every record in the pre-cooldown window carries them - here falling while
    the 500-per-set subset looks flat - else the subset, for the whole window; the reason and trend.cer_scope say
    which. The cooldown gain picks its scope the same way for its two evals."""
    sub = [2.0] * 9
    full = [2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6]
    hist = wsd_history(sub, cooldown_from=8, full=full)
    v2 = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=hist), version=2)
    assert v2["trend"]["cer_scope"] == "complete" and v2["trend"]["cer_window_steps"] == [500, 600, 700]
    assert v2["trend"]["improving"] and v2["verdict"] == "INCONCLUSIVE"
    assert v2["trend"]["cer_ratio_rel_change"] == pytest.approx(-0.2 / 1.9)
    assert any(r.endswith("at steps [500, 600, 700] (the complete sets)") for r in v2["reasons"])
    assert v2["cooldown_gain"]["scope"] == "complete" and v2["cooldown_gain"]["cer_ratio_after"] == pytest.approx(1.6)
    # one eval of the window without complete-set numbers (a subset eval): the whole window reads the subset
    del hist[5]["greedy_full"]
    v2 = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=hist), version=2)
    assert v2["trend"]["cer_scope"] == "subset" and not v2["trend"]["improving"] and v2["verdict"] == "NO-GO"
    assert any(r.endswith("(the fixed greedy subset)") for r in v2["reasons"])
    assert v2["cooldown_gain"]["scope"] == "complete"  # its two evals (700 and 900) both have them
    # a complete-set record missing a gate set its subset has is not complete
    hist = wsd_history(sub, cooldown_from=8, full=full)
    del hist[6]["greedy_full"]["eval_cv8"]
    assert ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=hist), version=2)["trend"]["cer_scope"] == "subset"


def test_verdict_v2_drops_near_duplicate_end_points():
    """A record less than min_epoch_gap (0.25) epochs after the previous kept one leaves the trends, the later of the
    two (the first A100 run's final eval, 46 steps after its epoch-8 eval); one without an epoch is kept; the gap is
    configurable (0 keeps everything). The cooldown gain still ends at the final eval."""
    hist = wsd_history([2.0, 1.9, 1.8, 1.7, 1.6, 1.5], cooldown_from=5)
    hist.append(dict(copy.deepcopy(hist[-1]), step=604, epoch=6.04))
    hist[-1]["greedy"] = {s: dict(d, cer_ref_corpus=d["cer_ref_corpus"] * 0.9) for s, d in hist[-1]["greedy"].items()}
    kept, dropped = ev.dedup_history(hist, 0.25)
    assert dropped == [604] and [r["step"] for r in kept] == [100, 200, 300, 400, 500, 600]
    v2 = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=hist), version=2)
    assert v2["trend"]["deduplicated_steps"] == [604] and v2["trend"]["window_steps"] == [400, 500, 600]
    assert any(r.startswith("de-duplicated: the evals at steps [604]") for r in v2["reasons"])
    assert v2["cooldown_gain"]["to_step"] == 604 and v2["cooldown_gain"]["cer_ratio_after"] == pytest.approx(1.35)
    v0 = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=hist), version=2, min_epoch_gap=0.0)
    assert v0["trend"]["deduplicated_steps"] == [] and v0["trend"]["window_steps"] == [500, 600, 604]
    # the gap is measured from the previous KEPT record: evals every 0.1 epoch keep one per 0.25 epoch or more
    dense = history2([(10 * i, 0.1 * i, "stable", 2.0, None, 0.5, 0.4) for i in range(1, 11)])
    assert [r["step"] for r in ev.dedup_history(dense, 0.25)[0]] == [10, 40, 70, 100]
    no_epoch = [{k: v for k, v in r.items() if k != "epoch"} for r in dense]
    assert ev.dedup_history(no_epoch, 0.25) == (no_epoch, [])


def test_verdict_v1_is_unchanged_by_the_v2_fields():
    """The records now carry lr_phase, greedy_full and (under v2) the epoch: v1 reads none of them, so the verdict of
    a history written before them and of the same history with them is identical, key for key (the first A100 run's
    INCONCLUSIVE reproduces: test_the_first_a100_runs_verdict_v1_and_v2)."""
    plain = history(FALLING, KL_DOWN, PROBE_DOWN)
    rich = copy.deepcopy(plain)
    for i, r in enumerate(rich):
        r.update(lr_phase="stable" if i < 8 else "cooldown", epoch=float(i),
                 greedy_full={s: dict(d, cer_ref_corpus=d["cer_ref_corpus"] * 3) for s, d in r["greedy"].items()})
    for ratios in ((1.1, 1.15, 1.6), (1.8, 1.9, 2.0), (1.3, 1.4, 1.45)):
        a = ev.verdict(dict(final=final(ratios), history=plain))
        b = ev.verdict(dict(final=final(ratios), history=rich))
        assert a == b and "version" not in a and "cooldown_gain" not in a


def write_ref(tmp_path, obj, name="ref.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_load_reference_checks_the_file(tmp_path):
    """The reference file: a name and the corpus CER of each gate set it covers, as fractions. The config's name wins
    over the file's; anything malformed raises at load time (the trainer reads it at setup), with the reason."""
    good = dict(name="parakeet-tdt-0.6b-ja", scope="complete gate sets", cer=dict(eval_jsut=0.073, eval_cv8=0.079,
                                                                                  eval_reazon=0.072))
    r = ev.load_reference(write_ref(tmp_path, good))
    assert r == dict(name="parakeet-tdt-0.6b-ja", path=str(tmp_path / "ref.json"), scope="complete gate sets",
                     cer=dict(eval_jsut=0.073, eval_cv8=0.079, eval_reazon=0.072))
    assert ev.load_reference(write_ref(tmp_path, good), name="parakeet (bake-off)")["name"] == "parakeet (bake-off)"
    assert ev.load_reference(write_ref(tmp_path, dict(cer=dict(eval_jsut=0.07))), name="x")["cer"] == dict(
        eval_jsut=0.07)  # a gate set left out is simply not compared
    for bad, match in ((dict(name="x"), "non-empty"), (dict(name="x", cer={}), "non-empty"),
                       (dict(name="x", cer=dict(jsut=0.07)), "not gate sets"),
                       (dict(name="x", cer=dict(galgame=0.2)), "not gate sets"),
                       (dict(name="x", cer=dict(eval_jsut=7.3)), r"fraction in \[0, 1\]"),
                       (dict(name="x", cer=dict(eval_jsut=None)), "fraction"),
                       (dict(name="x", cer=dict(eval_jsut=True)), "fraction"),
                       (dict(cer=dict(eval_jsut=0.07)), "needs a name"), ([0.07], "non-empty")):
        with pytest.raises(ValueError, match=match):
            ev.load_reference(write_ref(tmp_path, bad))


def test_reference_bar_next_to_the_teacher_gate(tmp_path):
    """Per gate set the student's final corpus CER next to the reference's and their ratio; pooled, the student's sum
    of edits / sum of chars and the reference's CERs weighted by the same chars. It never moves the tier: the verdict
    with and without it differs only by the "reference" entry, under v1 and v2."""
    fin = final((1.8, 2.0, 1.4))
    chars = dict(eval_jsut=1000, eval_cv8=3000, eval_reazon=6000)
    for s, d in fin["sets"].items():
        d.update(ref_chars=chars[s], ref_edits=d["cer_ref_corpus"] * chars[s])
    ref = ev.load_reference(write_ref(tmp_path, dict(name="ref", cer=dict(eval_jsut=0.10, eval_cv8=0.05,
                                                                           eval_reazon=0.12))))
    bar = ev.reference_bar(fin, ref)
    assert bar["name"] == "ref" and bar["gating"] is False and bar["not_compared"] == []
    for s, r in (("eval_jsut", 0.10), ("eval_cv8", 0.05), ("eval_reazon", 0.12)):
        st = fin["sets"][s]["cer_ref_corpus"]
        assert bar["sets"][s] == pytest.approx(dict(student=st, reference=r, ratio=st / r))
    st = sum(fin["sets"][s]["ref_edits"] for s in chars) / 10_000
    rf = (0.10 * 1000 + 0.05 * 3000 + 0.12 * 6000) / 10_000
    assert bar["pooled"] == dict(student=pytest.approx(st), reference=pytest.approx(rf), ratio=pytest.approx(st / rf),
                                 sets=sorted(chars))
    # a set the file does not cover is not compared, and the pool covers the others only
    part = ev.reference_bar(fin, dict(ref, cer=dict(eval_jsut=0.10)))
    assert list(part["sets"]) == ["eval_jsut"] and part["not_compared"] == ["eval_cv8", "eval_reazon"]
    assert part["pooled"]["reference"] == pytest.approx(0.10) and part["pooled"]["sets"] == ["eval_jsut"]
    # without edit / char counts (an older summary) there is no pool, the per-set rows stay
    assert ev.reference_bar(final((1.8, 2.0, 1.4)), ref)["pooled"] is None
    hist = history(FALLING, KL_DOWN, PROBE_DOWN)
    for opts in ({}, dict(version=2)):
        base = ev.verdict(dict(final=fin, history=hist), **opts)
        with_ref = ev.verdict(dict(final=fin, history=hist, reference=ref), **opts)
        assert with_ref.pop("reference") == bar and with_ref == base and "reference" not in base


FIRST_RUN = REAL / "cache" / "hf_runs" / "runs" / "viability-b20x2560-20260925T071746Z"


def test_the_first_a100_runs_verdict_v1_and_v2():
    """The first A100 run (its downloaded run dir; real data, read only): v1 on its stored history and final eval gives
    its verdict.json key for key (INCONCLUSIVE, CER trend -2.46 % over steps 8512 / 9728 / 9774, all in the cooldown).
    v2 on the same history completed from the run's files (scripts/05_evaluate.backfill_history: the LR phase from its
    lr_phase events, the complete sets' numbers from each eval's summary.json) - what the trainer now records: still
    INCONCLUSIVE, the CER trend -4.28 % over the pre-cooldown epochs 4-6 on the complete sets, the final eval 46 steps
    after the epoch-8 one de-duplicated, and a cooldown gain of -10.7 % on the ratio (epoch 6 -> the final eval)."""
    need_real(FIRST_RUN / "summary.json", FIRST_RUN / "events.jsonl", FIRST_RUN / "evals" / "step_9774")

    def load(p):
        return json.loads(p.read_text(encoding="utf-8"))

    def strict(o):  # RunLogger writes a non-finite float as null
        if isinstance(o, dict):
            return {k: strict(v) for k, v in o.items()}
        if isinstance(o, list):
            return [strict(v) for v in o]
        return None if isinstance(o, float) and not np.isfinite(o) else o

    from fixtures import load_script

    s = load(FIRST_RUN / "summary.json")
    fin = load(FIRST_RUN / "evals" / "step_9774" / "summary.json")["greedy_full"]
    v1 = ev.verdict(dict(final=fin, history=s["history"]))
    assert strict(v1) == load(FIRST_RUN / "evals" / "step_9774" / "verdict.json") == s["verdict"]
    assert v1["verdict"] == "INCONCLUSIVE" and v1["trend"]["window_steps"] == [8512, 9728, 9774]
    hist = load_script("05_evaluate").backfill_history(s["history"], str(FIRST_RUN / "summary.json"))
    assert [r.get("lr_phase") for r in hist] == [None] + ["stable"] * 6 + ["cooldown"] * 3
    assert [bool(r.get("greedy_full")) for r in hist] == [False] + [True] * 9
    v2 = ev.verdict(dict(final=fin, history=hist), version=2)
    t = v2["trend"]
    assert v2["verdict"] == "INCONCLUSIVE" and v2["sets"] == v1["sets"] and t["improving"]
    assert t["cer_window_steps"] == [4864, 6080, 7296] and t["cer_scope"] == "complete"
    assert t["cer_ratio_rel_change"] == pytest.approx(-0.04283, abs=5e-5)
    assert t["deduplicated_steps"] == [9774] and t["window_steps"] == [7296, 8512, 9728] and not t["overfit"]
    g = v2["cooldown_gain"]
    assert (g["from_step"], g["to_step"], g["scope"]) == (7296, 9774, "complete")
    assert g["cer_ratio_rel_change"] == pytest.approx(-0.1074, abs=5e-4)
    assert g["sets"]["eval_jsut"]["cer_after"] == pytest.approx(v1["sets"]["eval_jsut"]["student"])


def test_lr_phase_at_reads_the_lr_phase_events():
    events = [dict(at_step=1, phase="warmup"), dict(at_step=300, phase="stable"), dict(at_step=7898, phase="cooldown"),
              dict(at_step=7898, phase="cooldown")]  # a resume from before the cooldown logs its start again
    assert [ev.lr_phase_at(s, events) for s in (0, 1, 299, 300, 7897, 7898, 9774)] == [
        None, "warmup", "warmup", "stable", "stable", "cooldown", "cooldown"]
    replay = [dict(at_step=1, phase="warmup"), dict(at_step=300, phase="stable"), dict(at_step=500, phase="cooldown"),
              dict(at_step=300, phase="stable")]  # the largest at_step <= step holds, not the last line
    assert ev.lr_phase_at(600, replay) == "cooldown" and ev.lr_phase_at(400, replay) == "stable"
    assert ev.lr_phase_at(5, []) is None


def test_eval_record_and_flatten():
    tf = dict(sets={"eval_jsut": dict(kl=0.2, ce=0.3, top1=0.9, n_tok=10)}, all=dict(kl=0.2, ce=0.3, top1=0.9),
              wall_s=1.5, bad_audio=["x"])
    probe = dict(sets={}, all=dict(kl=0.1))
    rec = ev.eval_record(300, 1200.0, tf=tf, probe=probe)
    assert rec["heldout_kl"] == 0.2 and rec["probe_kl"] == 0.1 and rec["tf"]["eval_jsut"]["top1"] == 0.9
    assert "lr_phase" not in rec and "greedy" not in rec and "greedy_full" not in rec
    # a complete eval's record: the subset's numbers under greedy (unchanged: what v1, the best step and the curve
    # read) and the complete sets' next to them; the LR phase of the last step (none at step 0)
    def g(cer, n):
        return dict(sets={"eval_jsut": dict(cer_ref_corpus=cer, teacher_cer_ref_corpus=0.08, trunc_rate=0.0, n=n,
                                            ref_edits=1)})
    rec = ev.eval_record(300, 1200.0, tf=tf, greedy=g(0.12, 500), greedy_full=g(0.11, 5000), lr_phase="stable")
    assert rec["greedy"]["eval_jsut"] == dict(cer_ref_corpus=0.12, teacher_cer_ref_corpus=0.08, trunc_rate=0.0, n=500)
    assert rec["greedy_full"]["eval_jsut"] == dict(cer_ref_corpus=0.11, teacher_cer_ref_corpus=0.08, trunc_rate=0.0,
                                                   n=5000)
    assert rec["lr_phase"] == "stable" and "lr_phase" not in ev.eval_record(0, 0.0, tf=tf, lr_phase=None)
    flat = ev.flatten(tf, "eval_tf")
    assert flat["eval_tf/eval_jsut/kl"] == 0.2 and flat["eval_tf/all/top1"] == 0.9 and flat["eval_tf/wall_s"] == 1.5
    assert not any("bad_audio" in k for k in flat)
    # the per-set undecodable counts sit next to each set's other counts (only for a set that lost a row)
    flat = ev.flatten(dict(tf, n_bad_audio=3, bad_audio_per_set={"eval_cv8": 3}), "eval/tf")
    assert flat["eval/tf/n_bad_audio"] == 3.0 and flat["eval/tf/eval_cv8/n_bad_audio"] == 3.0
    assert not any("per_set" in k for k in flat) and "eval/tf/eval_jsut/n_bad_audio" not in flat


def test_combined_loss_pools_the_gate_tokens():
    """combined_loss = (w_kl * sum KL + w_ce * sum CE) / tokens over the gate sets: every gate token weighs the same
    (not the mean of the per-set means), the monitor-only hold-outs are left out, the sums and counts come with it,
    and a summary without gate tokens gives None."""
    tf = dict(sets=dict(eval_jsut=dict(kl=1.0, ce=2.0, top1=0.5, n_tok=10),
                        eval_cv8=dict(kl=0.5, ce=1.0, top1=0.5, n_tok=30),
                        eval_reazon=dict(kl=0.25, ce=4.0, top1=0.5, n_tok=60),
                        galgame=dict(kl=9.0, ce=9.0, top1=0.0, n_tok=1000)),
              all=dict(kl=8.0, ce=8.0, top1=0.1, n_tok=1100))
    s = ev.gate_kd_sums(tf)
    assert s == dict(kl_sum=10 + 15 + 15, ce_sum=20 + 30 + 240, n_tok=100,
                     sets=["eval_cv8", "eval_jsut", "eval_reazon"])
    c = ev.combined_loss(tf, 1.0, 0.8)
    assert c["value"] == pytest.approx((40 + 0.8 * 290) / 100) and (c["kl"], c["ce"]) == (0.4, 2.9)
    assert c["value"] != pytest.approx(np.mean([1.0 + 0.8 * 2.0, 0.5 + 0.8 * 1.0, 0.25 + 0.8 * 4.0]))
    assert (c["w_kl"], c["w_ce"], c["n_tok"]) == (1.0, 0.8, 100)
    assert c["value"] == pytest.approx(ev.headline(tf=tf)["val_loss"] + 0.8 * c["ce"])  # val_loss is its KL part
    assert ev.combined_loss(tf, 2.0, 0.0)["value"] == pytest.approx(0.8)
    assert ev.combined_loss(None, 1.0, 0.8) is None and ev.combined_loss(dict(sets={}), 1.0, 0.8) is None
    assert ev.combined_loss(dict(sets=dict(eval_jsut=dict(kl=1.0, ce=1.0, n_tok=0))), 1.0, 0.8) is None


# ------------------------------------------------------------------------------------------------ teacher baselines


def test_teacher_baselines_match_preregistered():
    need_real(*(TEACHER_OUT / s for s in ev.GATE_SETS))
    b = ev.teacher_baselines(TEACHER_OUT)  # raises if any set drifts by > 0.05 pp
    for s, want in {"eval_jsut": 0.0830, "eval_cv8": 0.0407, "eval_reazon": 0.0628}.items():
        assert abs(b[s]["cer_corpus"] - want) <= 0.0005, (s, b[s])
        assert b[s]["n_empty_ref"] == 0
    assert b["eval_jsut"]["n"] == 5000 and b["eval_cv8"]["n"] == 4483 and b["eval_reazon"]["n"] == 5263
    assert b["eval_reazon"]["cer_mean"] > b["eval_reazon"]["cer_corpus"]  # the mean over-weights short utterances


def test_teacher_baselines_refuse_drift(tmp_path):
    d = tmp_path / "eval_jsut"
    d.mkdir()
    rows = [dict(id=f"eval_jsut/{i}", hyp="あいうえおかきくけこ", ref="あいうえおかきくけこ", cer=0.0, duration=1.0,
                 n_tok=5, truncated=False) for i in range(9)]
    rows.append(dict(id="eval_jsut/9", hyp="", ref="あいうえおかきくけこ", cer=1.0, duration=1.0, n_tok=1, truncated=False))
    (d / "eval-00000.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered"):
        ev.teacher_baselines(tmp_path, ["eval_jsut"])
    b = ev.teacher_baselines(tmp_path, ["eval_jsut"], check=False)
    assert b["eval_jsut"]["cer_corpus"] == pytest.approx(0.1) and b["eval_jsut"]["cer_mean"] == pytest.approx(0.1)


# --------------------------------------------------------------------------------------- tiny model integration


def tiny_model(seed=0):
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    enc = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
               subsampling_conv_channels=8)
    cfg = CohereAsrConfig(encoder_config=enc, vocab_size=16384, hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=64, max_position_embeddings=128,
                          decoder_start_token_id=13764)
    torch.manual_seed(seed)
    m = CohereAsrForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
    with torch.no_grad():
        m.proj_out.weight.mul_(40.0)  # peaky next-token distributions: no near-ties for argmax to flip on
    return m.eval()


class StandInTokenizer:
    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["".join(chr(0x4E00 + t % 2000) for t in s if not (skip_special_tokens and t < 16)) for s in seqs]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return ev.teacher_tokenizer()
    except Exception:
        return StandInTokenizer()


@pytest.fixture(scope="module")
def featurizer():
    from kitsune.features import LogMel

    return LogMel()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from fixtures import make_fake_corpus

    return make_fake_corpus(tmp_path_factory.mktemp("corpus"),
                            {"src_a": (4, "train"), "eval_jsut": (7, "eval"), "eval_cv8": (5, "eval")},
                            rows_per_shard=8, seed=3, dur_range=(0.4, 2.2), no_second=("eval_jsut", "eval_cv8"))


@pytest.fixture(scope="module")
def selection(corpus):
    from fixtures import make_fake_selection

    return make_fake_selection(corpus, greedy_n=4, probe_n=2)


def self_targets(model, featurizer, wave: np.ndarray, T: int, k: int = 16):
    """The model's own greedy continuation of PROMPT for T steps and its top-k log-probs, by an independent path:
    one utterance, full forward per step (no batching, no padding, no KV cache)."""
    feats, mask = featurizer(torch.from_numpy(wave)[None], torch.tensor([len(wave)]))
    dec = list(ev.PROMPT)
    toks, idx, lps = [], [], []
    with torch.no_grad():
        for _ in range(T):
            logits = model(input_features=feats, attention_mask=mask, decoder_input_ids=torch.tensor([dec]),
                           use_cache=False).logits[0, -1].float()
            v, i = torch.topk(logits.log_softmax(-1), k)
            toks.append(int(i[0]))
            idx.append(i.numpy())
            lps.append(v.numpy())
            dec.append(int(i[0]))
    return np.array(toks, np.int16), np.stack(idx).astype(np.int16), np.stack(lps).astype(np.float16)


@pytest.fixture(scope="module")
def self_store(corpus, selection, featurizer, tmp_path_factory):
    """Eval store whose teacher targets are replaced by the tiny model's own outputs (student == teacher)."""
    from kitsune.trainset import eval_store, load_stores

    cache = tmp_path_factory.mktemp("cache") / "eval"
    st = eval_store(selection, corpus.data, corpus.teacher_out, cache, EVAL, log=lambda s: None)
    model = tiny_model(0)
    tok = np.load(cache / "targets_tokens.npy")
    idx = np.load(cache / "targets_topk_idx.npy")
    lp = np.load(cache / "targets_topk_lp.npy")
    for u in st.utts:  # audio from the corpus ground truth, so no memmap is open while the files are rewritten
        t, i, l = self_targets(model, featurizer, decode_audio(corpus.utts[u.id].audio), u.n_tok)
        tok[u.tok_off:u.tok_off + u.n_tok], idx[u.tok_off:u.tok_off + u.n_tok], lp[u.tok_off:u.tok_off + u.n_tok] = t, i, l
    for name, arr in (("tokens", tok), ("topk_idx", idx), ("topk_lp", lp)):
        np.save(cache / f"targets_{name}.npy", arr)
    return load_stores(cache)


def test_teacher_forced_self_consistency(self_store, featurizer):
    """Student == teacher: batched, padded, teacher-forced eval through the training collate must give top-1 = 1 and
    KL ~ 0 at every position - an off-by-one in the target positions would not."""
    from kitsune.patches import freeze_batchnorm

    model = tiny_model(0)
    model.train()
    freeze_batchnorm(model)
    flags = {n: m.training for n, m in model.named_modules()}
    summary, per_utt = ev.teacher_forced_eval(model, self_store, featurizer, "cpu", batch_s=3.0)
    assert {n: m.training for n, m in model.named_modules()} == flags  # train mode and frozen BN restored

    assert len(per_utt) == len(self_store.utts) and set(per_utt["source"]) == set(EVAL)
    assert list(per_utt.columns[:7]) == ["id", "source", "n_tok", "kl", "ce", "top1", "duration"]
    assert (per_utt["top1"] == 1.0).all()
    assert per_utt["kl"].abs().max() < 2e-3  # fp16 storage of the log-probs is the only difference
    ntok = sum(u.n_tok for u in self_store.utts)
    assert summary["all"]["n_tok"] == ntok and summary["n_utts"] == len(self_store.utts)
    assert summary["all"]["top1"] == 1.0 and summary["all"]["kl"] < 2e-3
    # CE of the teacher's own token == the teacher's NLL of it (stored top-1 log-prob)
    nll = -np.concatenate([self_store.targets(i)[2][:, 0].astype(np.float64) for i in range(len(self_store.utts))])
    assert summary["all"]["ce"] == pytest.approx(nll.mean(), abs=2e-3)
    for s in EVAL:
        d = summary["sets"][s]
        assert d["n_utts"] == sum(u.source == s for u in self_store.utts)
        assert d["frac_p1_gt_0.99"] + d["frac_p1_lt_0.9"] <= 1.0
    assert "student_tail" in per_utt.columns and "teacher_p1" in per_utt.columns

    # a different model disagrees: KL clearly positive
    other, _ = ev.teacher_forced_eval(tiny_model(1), self_store, featurizer, "cpu", batch_s=3.0)
    assert other["all"]["kl"] > 0.1 and other["all"]["top1"] < 0.5

    # a subset by id (the probe path)
    ids = [u.id for u in self_store.utts][:3]
    sub, sub_utt = ev.teacher_forced_eval(model, self_store, featurizer, "cpu", batch_s=3.0, ids=ids)
    assert sorted(sub_utt["id"]) == sorted(ids)


def test_evals_count_undecodable_rows_per_set(self_store, featurizer, tokenizer, monkeypatch):
    """An eval row whose audio does not decode is dropped by the dataset; both evals count it per set
    (bad_audio_per_set, which the verdict reads for the gate sets) and score the rest."""
    from kitsune.trainset import AudioBatchDataset

    bad = next(u.id for u in self_store.utts if u.source == "eval_cv8")
    ids = [bad, *[u.id for u in self_store.utts if u.id != bad][:3]]
    real = AudioBatchDataset.audio_bytes
    monkeypatch.setattr(AudioBatchDataset, "audio_bytes",
                        lambda ds, i: b"not audio" if ds.ids[i] == bad else real(ds, i))
    model = tiny_model(0)
    tf, tf_utt = ev.teacher_forced_eval(model, self_store, featurizer, "cpu", batch_s=3.0, ids=ids)
    gr, gr_utt = ev.greedy_eval(model, self_store, ids, featurizer, "cpu", batch_s=3.0, tokenizer=tokenizer)
    for summary, per_utt in ((tf, tf_utt), (gr, gr_utt)):
        assert summary["n_bad_audio"] == 1 and summary["bad_audio"] == [bad]
        assert summary["bad_audio_per_set"] == {"eval_cv8": 1}
        assert sorted(per_utt["id"]) == sorted(ids[1:])
    assert ev.flatten(tf, "eval/tf")["eval/tf/eval_cv8/n_bad_audio"] == 1.0
    monkeypatch.setattr(AudioBatchDataset, "audio_bytes", real)
    clean, _ = ev.teacher_forced_eval(model, self_store, featurizer, "cpu", batch_s=3.0, ids=ids)
    assert clean["n_bad_audio"] == 0 and clean["bad_audio_per_set"] == {}


def manual_greedy(model, featurizer, wave, max_new, eos=3):
    feats, mask = featurizer(torch.from_numpy(wave)[None], torch.tensor([len(wave)]))
    dec = list(ev.PROMPT)
    with torch.no_grad():
        for _ in range(max_new):
            t = int(model(input_features=feats, attention_mask=mask, decoder_input_ids=torch.tensor([dec]),
                          use_cache=False).logits[0, -1].argmax())
            dec.append(t)
            if t == eos:
                break
    return dec[len(ev.PROMPT):]


def test_greedy_eval(corpus, self_store, featurizer, tokenizer):
    model = tiny_model(0)
    ids = [u.id for u in self_store.utts if u.in_greedy_subset]
    assert ids
    summary, per_utt = ev.greedy_eval(model, self_store, ids, featurizer, "cpu", batch_s=3.0, tokenizer=tokenizer)
    assert list(per_utt.columns[:10]) == ["id", "source", "duration", "ref", "teacher_hyp", "hyp", "cer_ref",
                                          "cer_teacher", "truncated", "n_tok"]
    assert sorted(per_utt["id"]) == sorted(ids)
    max_new = min(int(16 + 10 * per_utt["duration"].max()), 128 - 10 - 1)  # per batch: from its longest utterance
    for r in per_utt.itertuples():
        u = corpus.utts[r.id]
        assert r.ref == u.text and r.teacher_hyp == u.hyp  # from the store index == teacher_out jsonl
        assert r.cer_ref == pytest.approx(utt_cer(r.hyp, r.ref)) and r.cer_teacher == pytest.approx(utt_cer(r.hyp, r.teacher_hyp))
        assert r.n_tok == len(r.hyp_ids) and r.truncated == (3 not in r.hyp_ids)
        assert r.n_tok <= max_new
        # batched generate with KV cache == the cacheless single-utterance greedy loop (RepetitionStop only cuts)
        ref_seq = manual_greedy(model, featurizer, decode_audio(u.audio), max_new)
        assert list(r.hyp_ids) == ref_seq[: r.n_tok]
    # teacher baseline on the same ids, from the fake teacher_out
    for s in EVAL:
        g = per_utt[per_utt["source"] == s]
        if not len(g):
            continue
        want = ev.corpus_cer([corpus.utts[i].hyp for i in g["id"]], [corpus.utts[i].text for i in g["id"]])["cer"]
        assert summary["sets"][s]["teacher_cer_ref_corpus"] == pytest.approx(want)
        assert summary["sets"][s]["cer_ref_corpus"] == pytest.approx(ev.corpus_cer(g["hyp"], g["ref"])["cer"])
    assert summary["all"]["n"] == len(ids) and summary["n_bad_audio"] == 0

    # teacher_rows (teacher_out jsonl) override the store text: with the student's own output as the teacher hyp,
    # the imitation CER is 0
    rows = {r.id: dict(ref=r.ref, hyp=r.hyp, cer=0.0, truncated=False) for r in per_utt.itertuples()}
    s2, p2 = ev.greedy_eval(model, self_store, ids, featurizer, "cpu", batch_s=3.0, tokenizer=tokenizer,
                            teacher_rows=rows)
    assert (p2["cer_teacher"] == 0.0).all()
    c = s2["all"]["cer_teacher_corpus"]
    assert c == 0.0 or np.isnan(c)  # NaN only if every hypothesis normalises to "" (nothing to divide by)


def test_greedy_eval_stops_at_eos(self_store, featurizer, tokenizer):
    model = tiny_model(0)
    with torch.no_grad():
        model.proj_out.bias[3] = 1e4  # EOS always wins
    summary, per_utt = ev.greedy_eval(model, self_store, None, featurizer, "cpu", batch_s=3.0, tokenizer=tokenizer)
    assert len(per_utt) == len(self_store.utts)
    assert (per_utt["n_tok"] == 1).all() and not per_utt["truncated"].any()
    assert (per_utt["hyp"] == "").all() and (per_utt["cer_ref"] == 1.0).all()
    assert summary["all"]["trunc_rate"] == 0.0 and summary["all"]["cer_ref_corpus"] == 1.0
    assert summary["all"]["n_empty_hyp"] == len(per_utt)
    samples = ev.pick_samples(per_utt, n=4)
    assert len(samples) == 4 and set(samples[0]) >= {"id", "ref", "teacher_hyp", "hyp", "cer_ref", "cer_teacher"}


# ------------------------------------------------------------------------------ combined loss: eval code == train code


@pytest.fixture(scope="module")
def mixed_store(tmp_path_factory):
    """Gate sets and a monitor-only hold-out (galgame) in one eval store, as the viability run evaluates them."""
    from fixtures import make_fake_corpus, make_fake_selection
    from kitsune.trainset import eval_store

    root = tmp_path_factory.mktemp("mixed")
    sets = ("eval_jsut", "eval_cv8", "galgame")
    fc = make_fake_corpus(root / "corpus", {"src_a": (3, "train"), **{s: (4, "eval") for s in sets}}, rows_per_shard=8,
                          seed=5, dur_range=(0.4, 2.2), no_second=sets)
    sel = make_fake_selection(fc, greedy_n=2, probe_n=1)
    return eval_store(sel, fc.data, fc.teacher_out, root / "cache", list(sets), log=lambda s: None)


def test_combined_val_is_the_training_objective(mixed_store, featurizer):
    """combined_loss/val as the trainer logs it (run_mini_eval: teacher_forced_eval -> kitsune.evaluate.combined_loss)
    equals w_kl * KL + w_ce * CE as the trainer computes it for an optimizer step (train_step: forward_logits,
    kd_losses, kd_objective; log_step's loss/objective and combined_loss/train) on the same batch without
    augmentation (SpecAugment off, the eval featuriser, eval mode; lr 0), to float tolerance: the same tokens, KL, CE
    and normalisation. Pooled over the gate sets only: galgame, in the store and in the ids, is left out; the step
    split over two micro-batches gives the same value (kd_objective divides by the step's tokens)."""
    from types import SimpleNamespace

    from fixtures import load_script
    from kitsune.kd import L2SP
    from kitsune.trainset import AudioBatchDataset, eval_batches

    m = load_script("04_distill")
    store = mixed_store
    w_kl, w_ce = 1.0, 0.8
    gate = [i for i, u in enumerate(store.utts) if u.source in ev.GATE_SETS]
    assert gate and len(gate) < len(store.utts) and {u.source for u in store.utts} >= {"galgame"}
    (batch,) = eval_batches(store.utts, 1e9, gate)  # the one batch the eval packs these utterances into
    ds = AudioBatchDataset(store)

    def train_objective(mbs: list[list[int]]) -> dict:
        model = tiny_model(0)  # eval mode: no dropout, as the eval sees it
        params = [p for p in model.parameters() if p.requires_grad]
        cfg = m.load_config(None, ["specaug.enabled=false", "log.layer_stats_every=0", "log.hist_every=0",
                                   f"loss.w_kl={w_kl}", f"loss.w_ce={w_ce}"])
        rows = []
        R = SimpleNamespace(cfg=cfg, device=torch.device("cpu"), model=model, feat_train=featurizer, specaug=None,
                            gen=torch.Generator(), opt=torch.optim.SGD(params, lr=0.0),
                            l2sp=L2SP(model.named_parameters(), lam=0.0), params=params,
                            param_names=[n for n, p in model.named_parameters() if p.requires_grad],
                            src_index={s: i for i, s in enumerate(sorted({store.utts[i].source for i in gate}))},
                            st=dict(nonfinite_skips=0, nonfinite_total=0, oom_skips=0, resumes=0, epoch=0,
                                    epoch_progress=0.0, audio_s=0.0, tokens=0, step_time_s=0.0, last_objective=None,
                                    flops_per_padded_s=None, memory={}),
                            log=SimpleNamespace(event=lambda *a, **k: None, train_utts=lambda r: None,
                                                step_row=lambda row, step: rows.append(row)),
                            planner=SimpleNamespace(epoch_stats={}), progress=lambda: (0.0, 1.0), clock=lambda: 0.0,
                            autocast=lambda: torch.autocast("cpu", enabled=False))
        out = m.train_step(R, 1, 0.0, [ds[b] for b in mbs], 0)
        objective = m.log_step(R, 1, 0.0, 0, out, 0.0, 1.0, 0, 0)
        return dict(objective=objective, row=rows[0], n_tok=out["n_tok"])

    one = train_objective([batch])
    assert one["row"]["combined_loss/train"] == one["row"]["loss/objective"] == one["objective"]
    assert one["row"]["loss/kl"] > 0.1 and one["row"]["loss/ce"] > 0.1  # a random student: both terms count
    assert one["n_tok"] == sum(store.utts[i].n_tok for i in gate)
    two = train_objective([batch[::2], batch[1::2]])
    assert two["objective"] == pytest.approx(one["objective"], rel=1e-6)

    scal = {}
    cfg = m.load_config(None, ["eval.mini.greedy=false", "eval.batch_s=1e9", f"loss.w_kl={w_kl}", f"loss.w_ce={w_ce}"])
    R = SimpleNamespace(cfg=cfg, model=tiny_model(0), evalstore=store, train=None, feat_eval=featurizer,
                        device="cpu", amp=False, mini_val_ids=[store.utts[i].id for i in batch], mini_train_ids=[],
                        st=dict(epoch_progress=0.0, mini_history=[]), clock=lambda: 0.0,
                        log=SimpleNamespace(event=lambda *a, **k: None, table=lambda *a, **k: None,
                                            eval_json=lambda *a, **k: None, scalars=lambda row, step: scal.update(row)))
    m.run_mini_eval(R, 1)
    assert scal["combined_loss/val"] == pytest.approx(one["objective"], rel=1e-6, abs=1e-7)
    tf, _ = ev.teacher_forced_eval(R.model, store, featurizer, "cpu", 1e9, ids=R.mini_val_ids)
    c = ev.combined_loss(tf, w_kl, w_ce)
    assert c["n_tok"] == one["n_tok"] and c["value"] == scal["combined_loss/val"]
    assert c["kl"] == pytest.approx(one["row"]["loss/kl"], rel=1e-6) and c["ce"] == pytest.approx(one["row"]["loss/ce"],
                                                                                                    rel=1e-6)

    # the monitor-only hold-out in the val ids changes nothing: it is not pooled
    scal.clear()
    R.mini_val_ids = [u.id for u in store.utts]
    m.run_mini_eval(R, 2)
    assert scal["combined_loss/val"] == pytest.approx(one["objective"], rel=1e-5)
    assert scal["eval/mini/tf/all/n_tok"] > one["n_tok"]  # galgame was evaluated, only not pooled
