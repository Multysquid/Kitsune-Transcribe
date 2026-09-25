"""kitsune/study_stats.py and tools/study_report.py: the size study's statistics and report (study/STUDY.md 4.3-4.7,
7), on synthetic systems whose ratios are planted.

Planted systems: every system's per-utterance edits are the SAME base edits times a multiplier m, so every metric's
ratio between two systems is exactly the ratio of their multipliers, in the full data and in every bootstrap
replicate (v_boot = 0). The CI is then ln r +- 1.96 sqrt(k) sigma_run, and each call is a known function of m: at
sigma_run 1.6 % and delta 10 %, a student-student r is WITHIN up to exp(ln 1.1 - 0.0443) = 1.052 and OUTSIDE from
exp(ln 1.1 + 0.0443) = 1.150. The noise model itself is checked by Monte Carlo (the STUDY.md 4.5 table; coverage of
the bootstrap CI on sampled utterances)."""
import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest

from fixtures import ROOT
from kitsune import study_stats as ss
from kitsune.evaluate import corpus_cer
from kitsune.store import ids_sha256

SIZES = {"eval_jsut": 120, "eval_cv8": 100, "eval_reazon": 110, "eval_emilia": 40, "galgame": 90}
N_NEUTRAL = 70

# the planted multipliers (see the module docstring for the call boundaries)
PLANT = {"study-t06": 1.0, "study-t03": 1.02, "study-t01": 1.10, "study-t005": 1.30, "study-bridge": 1.08,
         "study-p03": 1.0, "study-p01": 1.30, "study-p005": 1.50, "cohere": 0.90, "parakeet-ctc": 0.85,
         "parakeet-tdt": 1.35, "anchor-b20": 0.99,
         # the T/2 branches: T-0.3B keeps its gap to T-0.6B, T-0.1B's gap is 20 % wider at T/2
         "study-t06-half": 1.05, "study-t03-half": 1.02 * 1.05, "study-t01-half": 1.10 * 1.2 * 1.05}


def study_tool():
    spec = importlib.util.spec_from_file_location("study_report", ROOT / "tools" / "study_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def planted(mult: dict, seed: int = 0, sizes: dict = SIZES):
    """(tables, manifest): per set, random reference lengths and base edits; system s's edits = mult[s] x base. The
    last Galgame row has an empty reference (it must be left out of every sum); the neutral view is the first
    N_NEUTRAL Galgame rows."""
    rng = np.random.default_rng(seed)
    ids = {s: [f"{s}/u{i:04d}" for i in range(n)] for s, n in sizes.items()}
    ref_len = {s: rng.integers(4, 30, n) for s, n in sizes.items()}
    base = {s: rng.poisson(0.12 * ref_len[s]) + 1 for s in sizes}
    ref_len["galgame"][-1] = 0
    tables = {}
    for name, m in mult.items():
        tables[name] = pd.concat([pd.DataFrame(dict(id=ids[s], set=s, edits=m * base[s], ref_len=ref_len[s],
                                                    edits_nostyle=0.8 * m * base[s])) for s in sizes],
                                 ignore_index=True)
    manifest = {"sets": {s: {"ids": v, "ids_sha256": ids_sha256(v)} for s, v in ids.items()},
                "galgame_views": {"neutral": ids["galgame"][:N_NEUTRAL]}}
    return tables, manifest


@pytest.fixture(scope="module")
def report():
    tables, manifest = planted(PLANT)
    return ss.analyse(ss.build_corpus(tables, manifest), dict(boot_b=500))


# ------------------------------------------------------------------------------------------------ scoring


def test_score_utterances_sums_to_corpus_cer():
    """Per-utterance edits of score_utterances sum to kitsune.evaluate.corpus_cer's corpus edits (the same
    normalisation and alignment), with S + D + I = edits and empty references scored as insertions."""
    rng = np.random.default_rng(3)
    alphabet = list("あいうえおかきくけこアイウ漢字abc、。 ")
    refs = ["".join(rng.choice(alphabet, rng.integers(0, 12))) for _ in range(200)] + ["", "、。"]
    hyps = ["".join(rng.choice(alphabet, rng.integers(0, 12))) for _ in range(len(refs))]
    sc = ss.score_utterances(refs, hyps)
    c = corpus_cer(hyps, refs)
    kept = sc["ref_len"] > 0
    assert int(sc.loc[kept, "edits"].sum()) == c["edits"] and int(sc.loc[kept, "ref_len"].sum()) == c["ref_chars"]
    assert (sc["sub"] + sc["del"] + sc["ins"] == sc["edits"]).all()
    assert int((~kept).sum()) == c["n_empty_ref"]
    assert (sc.loc[~kept, "ins"] == sc.loc[~kept, "hyp_len"]).all()
    # a missing hypothesis (None, or NaN as parquet gives it back) is an empty output, never the text "nan"
    miss = ss.score_utterances(["あいう", "えお"], [None, float("nan")])
    assert miss["edits"].tolist() == [3, 2] and miss["del"].tolist() == [3, 2] and miss["hyp_len"].tolist() == [0, 0]


# ------------------------------------------------------------------------------------------------ planted calls


def test_planted_ratios_give_the_expected_calls_and_walks(report):
    fam = report["families"]
    calls = {(f, e["system"]): e["m4"]["calls"] for f in fam for e in fam[f]["entries"]}
    assert calls[("transcribe", "study-t03")] == {"0.05": "UNRESOLVED", "0.1": "WITHIN", "0.2": "WITHIN"}
    assert calls[("transcribe", "study-t01")] == {"0.05": "OUTSIDE", "0.1": "UNRESOLVED", "0.2": "WITHIN"}
    assert calls[("transcribe", "study-t005")] == {"0.05": "OUTSIDE", "0.1": "OUTSIDE", "0.2": "OUTSIDE"}
    assert calls[("parakeet", "study-p01")]["0.1"] == "OUTSIDE"
    # the scratch ladder's top is the bridge: T-0.1B / bridge = 1.10 / 1.08, T-0.05B / bridge = 1.30 / 1.08
    assert calls[("scratch", "study-t01")]["0.1"] == "WITHIN" and calls[("scratch", "study-t005")]["0.1"] == "OUTSIDE"
    e = fam["transcribe"]["entries"][1]["m4"]
    assert e["ratio"] == pytest.approx(1.10, rel=1e-12) and e["v_boot"] == pytest.approx(0.0, abs=1e-20)
    assert e["n_noisy"] == 2 and e["se"] == pytest.approx(math.sqrt(2) * 0.016)
    assert fam["transcribe"]["primary"]["sentence"] == "the limit is at or below T-0.3B; T-0.1B unresolved"
    assert fam["transcribe"]["primary"]["descriptive"] == ["study-t005"]
    assert fam["transcribe"]["walks"]["0.2"]["sentence"] == "limit reached between T-0.1B and T-0.05B"
    assert fam["transcribe"]["walks"]["0.05"]["sentence"] == "the limit is at or below T-0.6B; T-0.3B unresolved"
    assert fam["parakeet"]["primary"]["sentence"] == "limit reached between P-0.3B and P-0.1B"
    assert fam["parakeet"]["primary"]["descriptive"] == ["study-p005"]
    assert fam["scratch"]["primary"]["sentence"] == "limit reached between T-0.1B and T-0.05B"
    # the qualifier pairs ride along with every call, and the P 0.3 -> 0.1 label is fixed
    q = fam["transcribe"]["entries"][0]["qualifiers"]
    assert q["ood"]["ratio"] == pytest.approx(1.02) and q["ind"]["ratio"] == pytest.approx(1.02)
    assert fam["parakeet"]["entries"][0]["confound"] == "size + loss of function + depth"


def test_metrics_leave_empty_references_out_and_use_the_neutral_view(report):
    b = report["bootstrap"]
    assert b["strata"]["galgame_neutral"] == N_NEUTRAL and b["strata"]["galgame_all"] == SIZES["galgame"] - 1
    assert b["n_empty_ref"]["galgame_all"] == 1 and b["n_empty_ref"]["galgame_neutral"] == 0
    s = report["systems"]["study-t06"]
    m4 = np.mean([s["sets"][k]["cer"] for k in ss.M4_SETS])
    assert s["metrics"]["m4"] == pytest.approx(m4)
    jg = (s["sets"]["eval_jsut"]["cer"] + s["sets"]["galgame_neutral"]["cer"]) / 2
    assert s["metrics"]["jg"] == pytest.approx(jg)
    assert s["metrics"]["jg_nostyle"] == pytest.approx(0.8 * s["metrics"]["jg"])
    assert report["systems"]["study-t03"]["ratio_vs_teacher"]["m4"] == pytest.approx(1.02 / 0.90)


def test_g_per_halving_delta_g_init_effect_and_gaps(report):
    steps = {(s["big"], s["small"]): s for s in report["steps"]}
    st = steps[("study-t03", "study-t01")]
    h = math.log2(320_752_384 / 103_996_416)
    assert st["h"] == pytest.approx(1.625, abs=1e-3) and st["h"] == pytest.approx(h)
    assert st["g"] == pytest.approx((1.10 / 1.02) ** (1 / h) - 1)
    lo, hi = st["ci_g"]
    half = 1.96 * math.sqrt(2) * 0.016
    assert lo == pytest.approx(math.expm1((math.log(1.10 / 1.02) - half) / h))
    assert hi == pytest.approx(math.expm1((math.log(1.10 / 1.02) + half) / h))
    dg = report["delta_g"]
    h1, h2 = math.log2(320_752_384 / 103_996_416), math.log2(103_996_416 / 51_209_600)
    assert dg["delta_g"] == pytest.approx((1.30 / 1.10) ** (1 / h2) - (1.10 / 1.08) ** (1 / h1))
    assert dg["v_boot"] == pytest.approx(0.0, abs=1e-18) and dg["ci"][0] < dg["delta_g"] < dg["ci"][1]
    assert report["init_effect"]["comparison"]["ratio"] == pytest.approx(1.08 / 1.02)
    gap = report["distillation_gaps"]["transcribe"]
    assert gap["ratio"] == pytest.approx(1.0 / 0.90) and gap["n_noisy"] == 1  # the teacher has no run noise
    assert gap["se"] == pytest.approx(0.016)
    # g over the teacher -> student halvings (STUDY.md 1.3 and 4.3: 1.743 Transcribe, 0.986 Parakeet)
    assert gap["h"] == pytest.approx(1.743, abs=1e-3) and gap["g"] == pytest.approx((1 / 0.90) ** (1 / gap["h"]) - 1)
    assert gap["ci_g"][0] < gap["g"] < gap["ci_g"][1] and "not a step of the walk" in gap["label"]
    pg = report["distillation_gaps"]["parakeet"]
    assert pg["h"] == pytest.approx(0.986, abs=1e-3) and pg["ratio"] == pytest.approx(1.0 / 0.85)


def test_budget_readout_calls_compute_limited_only_when_the_gap_closes(report):
    rows = {r["system"]: r for r in report["budget"]["transcribe"]["rows"]}
    t03, t01, t005 = rows["study-t03"], rows["study-t01"], rows["study-t005"]
    assert t03["delta"] == pytest.approx(0.0, abs=1e-12) and t03["label"] == "not compute-limited at T"
    assert t01["delta"] == pytest.approx(math.log(1.10) - math.log(1.10 * 1.2))
    assert t01["compute_limited"] and t01["label"] == "compute-limited: the gap closes with compute"
    assert t01["at_T_half"]["calls"]["0.1"] == "OUTSIDE" and t01["at_T"]["calls"]["0.1"] == "UNRESOLVED"
    assert not t005["available"] and "study-t005-half" in t005["reason"]


def test_practical_bars(report):
    bars = report["bars"]
    # TDT bar on JSUT + Galgame: CI upper end <= TDT; T-0.05B (1.30 / 1.35) passes and is the smallest student
    assert bars["tdt"]["smallest"] == "study-t005"
    assert bars["tdt"]["per_student"]["study-p005"]["meets"] is False
    sm = bars["teacher"]["smallest"]
    # 1.2x: P-0.3B (1 / 0.85 = 1.176) by the point ratio; its CI reaches past 1.2, so T-0.3B (1.133) with the CI
    assert sm["1.2x"] == {"point": "study-p03", "ci": "study-t03"}
    assert sm["1.5x"] == {"point": "study-t005", "ci": "study-t005"}


def test_anchor_regression_flag():
    worse = ss.analyse(ss.build_corpus(*planted({**PLANT, "study-t06": 1.10, "anchor-b20": 1.0})), dict(boot_b=200))
    assert worse["anchor"]["flag"] is True
    assert next(c for c in worse["checks"] if c["rule"] == "anchor_regression")["status"] == "fail"
    assert worse["invalid"] is True
    close = ss.analyse(ss.build_corpus(*planted({**PLANT, "study-t06": 1.03, "anchor-b20": 1.0})), dict(boot_b=200))
    assert close["anchor"]["flag"] is False


def test_sigma_run_uses_the_replicate_when_it_is_larger():
    assert ss.sigma_run(None, 0.1)["sigma_run"] == 0.016
    small = ss.sigma_run(0.1 * 1.01, 0.1)
    assert small["sigma_run"] == 0.016 and small["sigma_hat"] == pytest.approx(0.886 * math.log(1.01))
    rep = ss.analyse(ss.build_corpus(*planted({**PLANT, "study-t01-s1235": 1.10 * 1.05})), dict(boot_b=200))
    s = 0.886 * math.log(1.05)
    assert rep["sigma_run"]["sigma_run"] == pytest.approx(s) and rep["sigma_run"]["source"].startswith("the replicate")
    # the wider noise turns T-0.3B's WITHIN into UNRESOLVED
    assert rep["families"]["transcribe"]["entries"][0]["m4"]["calls"]["0.1"] == "UNRESOLVED"
    assert rep["families"]["transcribe"]["entries"][0]["m4"]["se"] == pytest.approx(math.sqrt(2) * s)


# ------------------------------------------------------------------------------------------------ the walk


@pytest.mark.parametrize("calls, status, limit, stop, rest", [
    (["WITHIN", "WITHIN", "OUTSIDE"], "reached", "b", "c", []),
    (["WITHIN", "UNRESOLVED", "OUTSIDE"], "unresolved", "a", "b", ["c"]),
    (["OUTSIDE", "WITHIN", "WITHIN"], "reached", "top", "a", ["b", "c"]),
    (["UNRESOLVED", "WITHIN", "WITHIN"], "unresolved", "top", "a", ["b", "c"]),
    (["WITHIN", "WITHIN", "WITHIN"], "not_reached", "c", None, []),
    (["WITHIN", None, "WITHIN"], "incomplete", "a", "b", ["c"]),
])
def test_walk_stop_rules(calls, status, limit, stop, rest):
    w = ss.walk("top", list(zip("abc", calls)))
    assert (w["status"], w["limit"], w["stop"], w["descriptive"]) == (status, limit, stop, rest)
    if status == "reached":
        assert w["sentence"] == f"limit reached between {limit} and {stop}"
    if status == "unresolved":
        assert w["sentence"] == f"the limit is at or below {limit}; {stop} unresolved"


def test_tolerance_call_boundaries():
    t = math.log1p(0.10)
    assert ss.tolerance_call(-0.1, t, 0.10) == "WITHIN"  # the upper end AT ln(1 + delta) is within
    assert ss.tolerance_call(t, t + 0.1, 0.10) == "UNRESOLVED"  # the lower end must be strictly above
    assert ss.tolerance_call(t + 1e-9, t + 0.1, 0.10) == "OUTSIDE"
    assert list(ss.tolerance_call(np.array([-1.0, 0.0, 0.2]), np.array([0.0, 0.2, 0.3]), 0.10)) == [
        "WITHIN", "UNRESOLVED", "OUTSIDE"]


def test_pareto_front():
    pts = {"a": (0.10, 0.01, 2.0), "b": (0.12, 0.005, 1.0), "c": (0.13, 0.02, 3.0), "d": (0.10, 0.01, 2.5),
           "e": (0.09, None, 1.0)}
    assert ss.pareto(pts) == ["a", "b"]  # c and d are dominated; e has no speed


# ------------------------------------------------------------------------------------------------ the bootstrap


def test_bootstrap_is_reproducible_and_paired():
    tables, manifest = planted(PLANT)
    c = ss.build_corpus(tables, manifest)
    a, b = ss.bootstrap_sums(c, 300, 7), ss.bootstrap_sums(c, 300, 7)
    for k in c.strata:
        assert np.array_equal(a.ref[k], b.ref[k]) and np.array_equal(a.edits[k], b.edits[k], equal_nan=True)
    other = ss.bootstrap_sums(c, 300, 8)
    assert not np.array_equal(a.ref["eval_jsut"], other.ref["eval_jsut"])
    # chunking the replicates differently draws the same utterances (one stream per stratum): the integer sums are
    # bit-identical; the planted non-integer edits only up to float summation order
    chunked = ss.bootstrap_sums(c, 300, 7, chunk_elems=250)
    assert all(np.array_equal(a.ref[k], chunked.ref[k]) for k in c.strata)
    assert all(np.allclose(a.edits[k], chunked.edits[k], rtol=1e-12, equal_nan=True) for k in c.strata)
    itables, imanifest = planted({"study-t06": 1, "study-t03": 2, "cohere": 3})  # integer edits, as real ones
    ci = ss.build_corpus(itables, imanifest)
    x, y = ss.bootstrap_sums(ci, 300, 7), ss.bootstrap_sums(ci, 300, 7, chunk_elems=250)
    assert all(np.array_equal(x.edits[k], y.edits[k]) for k in ci.strata)
    # the same indices for every system: a corpus of two of the systems resamples exactly the same utterances
    sub = ss.build_corpus({k: itables[k] for k in ("study-t03", "cohere")}, imanifest)
    sb = ss.bootstrap_sums(sub, 300, 7)
    for k in ci.strata:
        assert np.array_equal(sb.ref[k], x.ref[k])
        for s in ("study-t03", "cohere"):
            assert np.array_equal(sb.edits[k][:, sub.index(s)], x.edits[k][:, ci.index(s)])
    # a whole report is reproducible to the bit
    r1, r2 = (json.dumps(ss.analyse(c, dict(boot_b=300, boot_seed=5))) for _ in range(2))
    assert r1 == r2


def test_multiplicities_are_resampling_with_replacement():
    m = ss.multiplicities(np.random.default_rng(0), 50, 400)
    assert m.shape == (400, 50) and (m.sum(axis=1) == 50).all()
    assert abs(m.mean() - 1.0) < 1e-12 and abs(m.var() - (1 - 1 / 50)) < 0.05  # Binomial(50, 1/50) counts


def _sampled_corpus(rng, p_a: float, p_b: float, n: int = 300, run_sd: float = 0.0) -> ss.Corpus:
    """Two trained systems on 4 M4 strata of n fresh utterances from a population whose per-set CER is exactly p_a
    and p_b (shared difficulty u, mean 1, independent of length): true r = p_a / p_b. run_sd scales each system's
    edits by exp(N(0, run_sd)) - planted run-to-run noise."""
    strata = {}
    scale = np.exp(rng.normal(0.0, run_sd, 2)) if run_sd else np.ones(2)
    for k in ss.M4_SETS:
        ln = rng.integers(4, 30, n)
        u = rng.gamma(2.0, 0.5, n)
        e = np.stack([rng.poisson(p * u * ln) * s for p, s in zip((p_a, p_b), scale)], axis=1).astype(float)
        strata[k] = ss.Stratum(name=k, ids=[f"{k}/{i}" for i in range(n)], ref_len=ln, edits=e,
                               nostyle=np.full_like(e, np.nan), n_empty_ref=0)
    return ss.Corpus(systems=["study-t03", "study-t06"], strata=strata, desc={}, hyps={},
                     manifest=ss.Manifest(sets={}, sha256={}))


@pytest.mark.parametrize("sigma", [0.0, 0.016])
def test_ci_covers_the_true_ratio_95_percent(sigma):
    """Monte Carlo through the real bootstrap and CI: 150 studies of freshly sampled utterances (and, with sigma,
    planted run noise of that SD per system and the same sigma_run in the CI) cover the true ln r about 95 % of the
    time."""
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(150):
        c = _sampled_corpus(rng, 0.13, 0.12, run_sd=sigma)
        st = ss.Study(c, dict(boot_b=300, sigma_run_prior=sigma, replicate=["x", "y"]))
        lo, hi = st.compare("study-t03", "study-t06")["ci_ln"]
        hits += lo <= math.log(0.13 / 0.12) <= hi
    assert 0.90 <= hits / 150 <= 0.99


@pytest.mark.parametrize("r, delta, call, expected", [
    (1.00, 0.05, "WITHIN", 0.53), (1.05, 0.10, "WITHIN", 0.49), (1.10, 0.05, "OUTSIDE", 0.49),
    (1.15, 0.20, "WITHIN", 0.43), (1.20, 0.10, "OUTSIDE", 0.95), (1.10, 0.10, "OUTSIDE", 0.02),
])
def test_call_frequencies_reproduce_the_study_table(r, delta, call, expected):
    """STUDY.md 4.5's table (sigma_total 2.4 % on ln r = sqrt(v_boot + 2 x 1.6 %^2) with v_boot = 0.8 %^2): the
    Monte Carlo through ci() / tolerance_call() and the normal-theory probabilities both land within a few points."""
    v_boot = 0.024 ** 2 - 2 * 0.016 ** 2
    mc = ss.simulate_calls(r, delta, v_boot, 0.016, n_noisy=2, n=20_000, seed=1)
    assert mc[call] == pytest.approx(expected, abs=0.03)
    assert ss.call_probabilities(r, delta, 0.024)[call] == pytest.approx(expected, abs=0.015)


# ------------------------------------------------------------------------------------------------ manifest refusals


def _refuse(tables, manifest, match):
    with pytest.raises(ss.ManifestError, match=match):
        ss.build_corpus(tables, manifest)


def test_manifest_refusals():
    tables, manifest = planted({"study-t06": 1.0, "study-t03": 1.1})
    t = tables["study-t03"]
    _refuse({**tables, "study-t03": t.iloc[1:]}, manifest, "1 missing")
    _refuse({**tables, "study-t03": pd.concat([t, t.iloc[:1]])}, manifest, "duplicated")
    extra = t.iloc[:1].assign(id="eval_jsut/stranger")
    _refuse({**tables, "study-t03": pd.concat([t, extra])}, manifest, "1 not in it")
    _refuse({**tables, "study-t03": pd.concat([t, t.iloc[:1].assign(set="eval_x")])}, manifest, "not in the manifest")
    other = t.copy()
    other.loc[other["set"] == "eval_cv8", "ref_len"] += 1
    _refuse({**tables, "study-t03": other}, manifest, "reference lengths")
    bad = json.loads(json.dumps(manifest))
    bad["sets"]["eval_cv8"]["ids"] = bad["sets"]["eval_cv8"]["ids"][::-1]  # same ids, other order: another hash
    _refuse(tables, bad, "hash")
    bad = json.loads(json.dumps(manifest))
    del bad["sets"]["eval_cv8"]["ids_sha256"]
    _refuse(tables, bad, "no ids_sha256")
    bad = json.loads(json.dumps(manifest))
    bad["galgame_views"]["neutral"] = bad["galgame_views"]["neutral"] + ["galgame/stranger"]
    _refuse(tables, bad, "not in the galgame set")
    # a system may lack a whole set: its numbers there are missing, the rest is analysed
    c = ss.build_corpus({**tables, "study-t03": t[t["set"] != "eval_cv8"]}, manifest)
    rep = ss.analyse(c, dict(boot_b=100))
    assert rep["systems"]["study-t03"]["metrics"]["m4"] is None
    assert rep["families"]["transcribe"]["primary"]["status"] == "incomplete"


def test_manifest_shapes_and_the_prereg_hashes():
    """The manifest layouts parse_manifest accepts, and PREREG.json's frozen hashes against the manifest's."""
    ids, g = ["eval_jsut/1", "eval_jsut/2"], ["galgame/1", "galgame/2", "galgame/3"]
    m = ss.parse_manifest({"eval_sets": {"eval_jsut": {"ids": ids, "sha256": ids_sha256(ids)},
                                         "galgame": {"ids": g, "ids_sha256": ids_sha256(g)}},
                           "views": {"galgame": {"neutral": g[:2]}}})
    assert m.views == {"neutral": g[:2]} and m.sha256["eval_jsut"] == ids_sha256(ids)
    m2 = ss.parse_manifest({"eval_jsut": {"ids": ids, "ids_sha256": ids_sha256(ids)}, "written_utc": "x",
                            "galgame": {"ids": g, "ids_sha256": ids_sha256(g),
                                        "views": {"neutral": {"ids": g[:1], "ids_sha256": ids_sha256(g[:1])}}}})
    assert sorted(m2.sets) == ["eval_jsut", "galgame"] and m2.views == {"neutral": g[:1]}
    with pytest.raises(ss.ManifestError, match="view neutral"):
        ss.parse_manifest({"sets": {"galgame": {"ids": g, "ids_sha256": ids_sha256(g)}},
                           "galgame_views": {"neutral": {"ids": g[:1], "ids_sha256": "0" * 64}}})
    hashes = ss.prereg_manifest_hashes({"manifest": {"eval_jsut": ids_sha256(ids), "path": "x", "file": "0" * 64,
                                                     "galgame": {"ids_sha256": ids_sha256(g)}}})
    assert hashes == {"eval_jsut": ids_sha256(ids), "galgame": ids_sha256(g)}
    assert ss.manifest_check(m, hashes)["status"] == "pass"
    assert ss.manifest_check(m, {"eval_jsut": "f" * 64})["status"] == "fail"
    assert ss.manifest_check(m, {})["status"] == "pass"
    assert ss.prereg_baselines({"baselines": {"parakeet-ctc": {"eval_jsut": 0.065, "note": "card", "pct": 6.5}}}) == {
        "parakeet-ctc": {"eval_jsut": 0.065}}


def test_the_tool_refuses_a_table_that_does_not_match_the_manifest(tmp_path, capsys):
    tool = study_tool()
    tables, manifest = planted({"study-t06": 1.0, "study-t03": 1.1})
    (tmp_path / "tables").mkdir()
    for k, df in tables.items():
        (df.iloc[1:] if k == "study-t03" else df).to_parquet(tmp_path / "tables" / f"{k}.parquet")
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    rc = tool.main(["--tables", str(tmp_path / "tables"), "--manifest", str(tmp_path / "m.json"),
                    "--out", str(tmp_path / "out")])
    assert rc == 2 and "REFUSED" in capsys.readouterr().err and not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------------ PREREG and checks


def test_settings_from_prereg():
    st, src = ss.settings_from_prereg(None)
    assert st["delta_primary"] == 0.10 and st["boot_b"] == 10_000 and set(src.values()) == {"default (STUDY.md)"}
    st, src = ss.settings_from_prereg({"analysis": {"delta": 0.2, "sigma_run": {"prior": 0.02},
                                                    "bootstrap": {"B": 2000, "seed": 9}},
                                       "runs": {"replicate": "study-t01-s1235", "delta": 5}})
    assert (st["delta_primary"], st["sigma_run_prior"], st["boot_b"], st["boot_seed"]) == (0.2, 0.02, 2000, 9)
    assert src["boot_b"] == "PREREG.json:analysis.bootstrap.B" and st["replicate"] == ["study-t01-s1235", "study-t01"]
    st, src = ss.settings_from_prereg({"bootstrap": {"B": 500, "seed": 3}, "metrics": {"deltas": [0.05, 0.1]}})
    assert st["boot_b"] == 500 and st["deltas"] == [0.05, 0.1] and src["deltas"] == "PREREG.json:metrics.deltas"
    with pytest.raises(ValueError, match="fraction"):
        ss.settings_from_prereg({"analysis": {"delta": 10}})
    with pytest.raises(ValueError, match="integer >= 100"):
        ss.settings_from_prereg({"analysis": {"bootstrap": {"B": 10}}})


def test_invalidation_checks_from_the_numbers():
    ok = {"lr_probes": {"scratch": {"5e-4": 1.2, "1e-3": 1.1, "2e-3": 1.3}},
          "calibration": {"study-t06": {"data_wait_frac": 0.01}}, "max_steps": {"study-t06": 9366},
          "written_utc": "2026-10-01T00:00:00+00:00"}
    assert ss.lr_edge_check(ok)["status"] == "pass"
    edge = {"lr_probes": {"scratch": {"5e-4": 1.2, "1e-3": 1.1, "2e-3": 1.0, "4e-3": 0.9}}}
    assert ss.lr_edge_check(edge)["status"] == "fail"
    assert ss.loader_check(ok)["status"] == "pass"
    assert ss.loader_check({"calibration": {"study-t06": {"data_wait_frac": 0.07}}})["status"] == "fail"
    summ = {"study-t06": {"steps": 9366, "started_utc": "2026-10-01T01:00:00+00:00"}, "study-t06-half": {"steps": 4683}}
    assert ss.max_steps_check(ok, summ)["status"] == "pass"
    assert ss.max_steps_check(ok, {"study-t06": {"steps": 9000}})["status"] == "fail"
    assert ss.max_steps_check(None, summ)["status"] == "not_checked"
    assert ss._timing_check(ok, summ)["status"] == "pass"
    assert ss._timing_check({**ok, "written_utc": "2026-10-02T00:00:00+00:00"}, summ)["status"] == "fail"
    assert ss.selection_check({"a": {"selection_sha256": "x"}, "b": {"selection_sha256": "y"}})["status"] == "fail"


def test_teacher_baseline_check():
    """Cohere's pre-registered gate baselines (8.30 / 4.07 / 6.28 %) must reproduce within 0.05 pp."""
    tables, manifest = planted({"cohere": 1.0, "study-t06": 1.2})
    st = ss.Study(ss.build_corpus(tables, manifest), dict(boot_b=100))
    assert ss.teacher_baseline_check(st, None)["status"] == "fail"  # the planted CERs are not the real ones
    got = {s: float(ss.stratum_cer(st.point, s)[st.corpus.index("cohere")]) for s in ("eval_jsut", "eval_cv8")}
    chk = ss.teacher_baseline_check(st, {"cohere": got}, defaults={})
    assert chk["status"] == "pass" and len(chk["rows"]) == 2
    near = {"eval_jsut": got["eval_jsut"] + 0.0004}  # inside the 0.05 pp tolerance
    assert ss.teacher_baseline_check(st, {"cohere": near}, defaults={})["status"] == "pass"
    far = {"eval_jsut": got["eval_jsut"] + 0.001}
    assert ss.teacher_baseline_check(st, {"cohere": far}, defaults={})["status"] == "fail"
    # PREREG.json's numbers override the evaluator's set by set (JSUT and CV8 here; Reazon keeps 6.28 %, which the
    # planted teacher misses); a teacher without a table is listed, not failed
    chk = ss.teacher_baseline_check(st, {"cohere": got, "parakeet-ctc": {"eval_jsut": 0.065}})
    assert chk["status"] == "fail"
    assert [r["set"] for r in chk["rows"] if not r["ok"]] == ["eval_reazon"]
    assert "parakeet-ctc/eval_jsut" not in chk["detail"]


# ------------------------------------------------------------------------------------------------ the tool


def test_study_report_end_to_end(tmp_path):
    """The three table layouts (one file per system, one per set, a 05_evaluate dir with text only), PREREG.json,
    PREREG_numbers.json, summaries and a speed JSON -> report.json and report.md with the planted calls."""
    tool = study_tool()
    tables, manifest = planted(PLANT)
    d = tmp_path / "tables"
    d.mkdir()
    for k, df in tables.items():
        if k == "study-t03":  # one file per set
            (d / k).mkdir()
            for s, g in df.groupby("set"):
                g.drop(columns="set").to_parquet(d / k / f"{s}.parquet")
        else:
            df.to_parquet(d / f"{k}.parquet")
    # parakeet-tdt as a 05_evaluate dir: ref / hyp text only, plus files the report ignores
    txt = d / "extra-model"
    txt.mkdir()
    for s, ids in manifest["sets"].items():
        refs = ["あいうえお" if i % 7 else "かきく" for i in range(len(ids["ids"]))]
        hyps = ["あいうえ" if i % 3 else "かきくけ" for i in range(len(ids["ids"]))]
        pd.DataFrame(dict(id=ids["ids"], source=s, ref=refs, hyp=hyps, truncated=False)).to_parquet(
            txt / f"greedy_{s}.parquet")
    pd.DataFrame(dict(id=["x"])).to_parquet(txt / "tf_eval_jsut.parquet")
    # the text system's reference lengths differ from the planted ones, which the report must refuse...
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = ["--tables", str(d), "--manifest", str(tmp_path / "m.json")]
    assert tool.main(args + ["--out", str(tmp_path / "bad")]) == 2
    # ...so it is analysed on its own in a second run below
    (tmp_path / "txt").mkdir()
    txt.rename(tmp_path / "txt" / "extra-model")
    prereg = {"analysis": {"delta": 0.10, "bootstrap": {"B": 400, "seed": 3}}}
    numbers = {"max_steps": {"study-t06": 100}, "calibration": {"study-t06": {"data_wait_frac": 0.01}},
               "lr_probes": {"kept-t03": {"1e-4": 1.0, "2e-4": 0.9, "4e-4": 1.1}}}
    (tmp_path / "prereg.json").write_text(json.dumps(prereg), encoding="utf-8")
    (tmp_path / "numbers.json").write_text(json.dumps(numbers), encoding="utf-8")
    runs = tmp_path / "runs"
    (runs / "study-t06").mkdir(parents=True)
    (runs / "study-t06" / "summary.json").write_text(json.dumps({"steps": 100}), encoding="utf-8")
    speed = {"systems": {"study-t06": {"rtf": 0.02, "vram_gb": 3.0}, "study-t005": {"rtf": 0.005, "vram_gb": 1.0},
                         "cohere": {"rtf": 0.05, "vram_peak_reserved_bytes": 9e9}}}
    (tmp_path / "speed.json").write_text(json.dumps(speed), encoding="utf-8")
    out = tmp_path / "out"
    rc = tool.main(args + ["--prereg", str(tmp_path / "prereg.json"), "--numbers", str(tmp_path / "numbers.json"),
                           "--run-summaries", str(runs), "--speed", str(tmp_path / "speed.json"), "--out", str(out)])
    assert rc == 0
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["bootstrap"]["B"] == 400 and rep["settings_sources"]["boot_b"] == "PREREG.json:analysis.bootstrap.B"
    assert rep["families"]["transcribe"]["primary"]["sentence"] == "the limit is at or below T-0.3B; T-0.1B unresolved"
    assert rep["systems"]["study-t03"]["sets"]["eval_cv8"]["cer"] == pytest.approx(
        1.02 * rep["systems"]["study-t06"]["sets"]["eval_cv8"]["cer"])
    checks = {c["rule"]: c["status"] for c in rep["checks"]}
    assert checks["max_steps"] == "pass" and checks["loader_bound"] == "pass" and checks["lr_edge"] == "pass"
    assert rep["pareto"]["rows"]["cohere"]["vram_gb"] == pytest.approx(9.0)
    assert set(rep["pareto"]["front"]["jg"]) <= {"study-t06", "study-t005", "cohere"}
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "the limit is at or below T-0.3B; T-0.1B unresolved" in md and "## Invalidation checks" in md
    # the 05_evaluate layout on its own: scored from the text, the ignored tf_ file listed
    txt_man = {"sets": manifest["sets"]}
    (tmp_path / "m2.json").write_text(json.dumps(txt_man), encoding="utf-8")
    assert tool.main(["--tables", str(tmp_path / "txt"), "--manifest", str(tmp_path / "m2.json"),
                      "--out", str(tmp_path / "out2"), "--boot-b", "200"]) == 0
    rep2 = json.loads((tmp_path / "out2" / "report.json").read_text(encoding="utf-8"))
    j = rep2["systems"]["extra-model"]["sets"]["eval_jsut"]
    ids = manifest["sets"]["eval_jsut"]["ids"]
    want = corpus_cer(["あいうえ" if i % 3 else "かきくけ" for i in range(len(ids))],
                      ["あいうえお" if i % 7 else "かきく" for i in range(len(ids))])["cer"]
    assert j["cer"] == pytest.approx(want) and j["sub"] + j["del"] + j["ins"] > 0
    assert any(f.endswith("tf_eval_jsut.parquet") for f in rep2["inputs"]["ignored_files"])
    assert rep2["settings_sources"]["boot_b"].startswith("command line")
