"""kitsune/study_stats.py: the scoring, the corpus on the frozen manifest, the paired bootstrap, sigma_run and the CIs
that tools/full_report.py and scripts/05_evaluate.py use (study/STUDY.md 4.3-4.4), on synthetic systems whose ratios
are planted (tests/fixtures_planted.py: every system's edits are the same base edits times a multiplier, so every
ratio is exact). The noise model itself is checked by Monte Carlo (the coverage of the bootstrap CI on sampled
utterances). The PREREG readers are tested on kitsune.prereg's layout (an inline copy of its shape) and on the committed
study/PREREG.json."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from fixtures import ROOT
from fixtures_planted import N_NEUTRAL, SIZES, planted
from kitsune import study_stats as ss
from kitsune.evaluate import corpus_cer
from kitsune.store import ids_sha256

PLANT = {"study-t06": 1.0, "study-t03": 1.02, "study-t01": 1.10, "cohere": 0.90, "parakeet-ctc": 0.85}


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


def test_the_corpus_leaves_empty_references_out_and_uses_the_neutral_view():
    tables, manifest = planted(PLANT)
    c = ss.build_corpus(tables, manifest)
    assert len(c.strata["galgame_neutral"].ids) == N_NEUTRAL
    assert len(c.strata["galgame_all"].ids) == SIZES["galgame"] - 1
    assert c.strata["galgame_all"].n_empty_ref == 1 and c.strata["galgame_neutral"].n_empty_ref == 0
    p = ss.point_sums(c)
    t06, t03 = c.index("study-t06"), c.index("study-t03")
    for k in ss.M4_SETS:
        cer = ss.stratum_cer(p, k)
        assert cer[t03] == pytest.approx(1.02 * cer[t06])
        assert ss.stratum_cer(p, k, nostyle=True)[t06] == pytest.approx(0.8 * cer[t06])


# ------------------------------------------------------------------------------------------------ noise


def test_sigma_run_uses_the_replicate_when_it_is_larger():
    assert ss.sigma_run(None, 0.1)["sigma_run"] == 0.016
    small = ss.sigma_run(0.1 * 1.01, 0.1)
    assert small["sigma_run"] == 0.016 and small["sigma_hat"] == pytest.approx(0.886 * math.log(1.01))
    big = ss.sigma_run(0.1 * 1.05, 0.1)
    assert big["sigma_run"] == pytest.approx(0.886 * math.log(1.05)) and big["source"].startswith("the replicate")
    assert ss.REPLICATE == ("study-t01-s1235", "study-t01")


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


def _m4(sums: ss.Sums) -> np.ndarray:
    return np.mean([ss.stratum_cer(sums, k) for k in ss.M4_SETS], axis=0)


@pytest.mark.parametrize("sigma", [0.0, 0.016])
def test_ci_covers_the_true_ratio_95_percent(sigma):
    """Monte Carlo through the real bootstrap and CI, as tools/full_report.py compares two trained systems (ln r of
    M4 +- 1.96 sqrt(v_boot + 2 sigma_run^2), v_boot over the bootstrap's ln r): 150 studies of freshly sampled
    utterances (and, with sigma, planted run noise of that SD per system and the same sigma_run in the CI) cover the
    true ln r about 95 % of the time."""
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(150):
        c = _sampled_corpus(rng, 0.13, 0.12, run_sd=sigma)
        point, boot = _m4(ss.point_sums(c)), _m4(ss.bootstrap_sums(c, 300, ss.BOOT_SEED))
        v_boot = float(np.var(np.log(boot[:, 0]) - np.log(boot[:, 1]), ddof=1))
        lo, hi, _ = ss.ci(math.log(point[0] / point[1]), v_boot, sigma, 2)
        hits += lo <= math.log(0.13 / 0.12) <= hi
    assert 0.90 <= hits / 150 <= 0.99


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
    # a system may lack a whole set: its numbers there are missing, the rest is there
    c = ss.build_corpus({**tables, "study-t03": t[t["set"] != "eval_cv8"]}, manifest)
    t03, t06 = c.index("study-t03"), c.index("study-t06")
    assert np.isnan(c.strata["eval_cv8"].edits[:, t03]).all() and not np.isnan(c.strata["eval_cv8"].edits[:, t06]).any()
    assert not np.isnan(c.strata["eval_jsut"].edits[:, t03]).any()


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
    pm = ss.prereg_manifest({"manifest": {"eval_jsut": ids_sha256(ids), "path": "x", "file": "0" * 64,
                                          "galgame": {"ids_sha256": ids_sha256(g)}}})
    assert pm["sets"] == {"eval_jsut": ids_sha256(ids), "galgame": ids_sha256(g)} and pm["status"] == "filled"
    chk = ss.manifest_check(m, pm)
    assert chk["status"] == "pass" and "views not compared" in chk["detail"]
    assert ss.manifest_check(m, ss.prereg_manifest({"manifest": {"eval_jsut": "f" * 64}}))["status"] == "fail"
    # nothing compared is never a pass: no PREREG, a pending block, a filled block without a readable hash
    assert ss.manifest_check(m, ss.prereg_manifest(None))["status"] == "not_checked"
    pending = {"manifest": {"status": "pending", "ids_sha256": {"eval_jsut": "pending"}}}
    assert ss.manifest_check(m, ss.prereg_manifest(pending))["status"] == "not_checked"
    unreadable = {"manifest": {"status": "filled", "ids_sha256": {"train": "c" * 64}}}
    assert ss.manifest_check(m, ss.prereg_manifest(unreadable))["status"] == "fail"
    assert ss.prereg_baselines({"baselines": {"parakeet-ctc": {"eval_jsut": 0.065, "note": "card", "pct": 6.5}}}) == {
        "parakeet-ctc": {"eval_jsut": 0.065}}


# ------------------------------------------------------------------------------------------------ PREREG readers


def wp1_prereg(man: ss.Manifest | None = None, baselines: dict | None = None, manifest_sha: str = "b" * 64) -> dict:
    """The blocks of study/PREREG.json the readers read, in kitsune.prereg's layout (WP1): the manifest and baselines
    "pending" until the selection's sidecar filled them, the teachers parakeet_ctc / parakeet_tdt and the Galgame views
    galgame:<view>. With man: the manifest block filled from it; with baselines: that block filled."""
    p = {"manifest": {"status": "pending", "manifest_sha256": "pending", "selection_sha256": "pending",
                      "ids_sha256": {"train": "pending", "eval_jsut": "pending"},
                      "galgame_views": {"neutral": {"n": "pending", "ids_sha256": "pending"}}},
         "baselines": {"status": "pending", "fill": "the sidecar's baselines",
                       "cohere": {"eval_jsut": "pending", "galgame:neutral": "pending", "m4": "pending"},
                       "parakeet_ctc": {"eval_jsut": "pending"}}}
    if man is not None:
        p["manifest"] = {"status": "filled", "from": "labels/full/selections/study_1000h.parquet",
                         "selection_sha256": "a" * 64, "manifest_sha256": manifest_sha,
                         "ids_sha256": {"train": "c" * 64, "probe": "d" * 64, **man.sha256},
                         "n": {s: len(v) for s, v in man.sets.items()},
                         "galgame_views": {v: {"n": len(ids), "ids_sha256": ids_sha256(ids)}
                                           for v, ids in man.views.items()}}
    if baselines is not None:
        p["baselines"] = {"status": "filled", "from": "the sidecar's baselines on the manifest rows", **baselines}
    return p


def test_the_prereg_readers_on_kitsune_prereg_layout():
    """kitsune.prereg's PREREG.json: a pending manifest block compares nothing (not_checked), a filled one compares
    every eval set's, every Galgame view's and the manifest file's hash; the baselines are read under this module's
    names (parakeet_ctc -> parakeet-ctc, galgame:<view> -> galgame_<view>)."""
    tables, manifest = planted(PLANT)
    man = ss.parse_manifest(manifest)
    p = wp1_prereg()
    assert ss.prereg_manifest(p)["status"] == "pending" and ss.prereg_baselines(p) == {}
    assert ss.manifest_check(man, ss.prereg_manifest(p))["status"] == "not_checked"
    pm = ss.prereg_manifest(wp1_prereg(man))
    assert pm["sets"] == man.sha256 and pm["views"] == man.view_sha256 and set(pm["sets"]) == set(SIZES)
    assert (pm["manifest_sha256"], pm["selection_sha256"]) == ("b" * 64, "a" * 64)
    assert ss.manifest_check(man, pm, "b" * 64)["status"] == "pass"
    chk = ss.manifest_check(man, pm, "e" * 64)
    assert chk["status"] == "fail" and "manifest file" in chk["detail"]
    # a different neutral view (the one M4 uses) is a different manifest, though every eval set's ids are the same
    other = json.loads(json.dumps(manifest))
    other["galgame_views"]["neutral"] = other["galgame_views"]["neutral"][1:]
    chk = ss.manifest_check(ss.parse_manifest(other), pm, "b" * 64)
    assert chk["status"] == "fail" and "Galgame view neutral" in chk["detail"]
    # and a set the pre-registration never listed
    keep = [k for k in man.sets if k != "eval_emilia"]
    fewer = ss.prereg_manifest(wp1_prereg(ss.Manifest(sets={k: man.sets[k] for k in keep},
                                                      sha256={k: man.sha256[k] for k in keep},
                                                      views=man.views, view_sha256=man.view_sha256)))
    chk = ss.manifest_check(man, fewer, "b" * 64)
    assert chk["status"] == "fail" and "eval_emilia: in the manifest, not pre-registered" in chk["detail"]
    b = {"cohere": {"eval_jsut": 0.08, "galgame:neutral": 0.1, "m4": 0.07},
         "parakeet_ctc": {"eval_jsut": 0.07, "galgame:all": 0.2, "galgame": 0.21}}
    assert ss.prereg_baselines(wp1_prereg(baselines=b)) == {
        "cohere": {"eval_jsut": 0.08, "galgame_neutral": 0.1, "m4": 0.07},
        "parakeet-ctc": {"eval_jsut": 0.07, "galgame_all": 0.21}}


def test_the_committed_prereg_is_read_whole():
    """The committed study/PREREG.json (filled): its manifest block names every eval set and Galgame view with its
    hash and the frozen manifest's file sha256, and its baselines every teacher on every stratum and M4."""
    from kitsune import fullrun

    pr = json.loads((ROOT / "study" / "PREREG.json").read_text(encoding="utf-8"))
    pm = ss.prereg_manifest(pr)
    assert pm["status"] == "filled" and sorted(pm["sets"]) == sorted(ss.EVAL_SETS)
    assert sorted(pm["views"]) == sorted(ss.GALGAME_VIEWS) and pm["manifest_sha256"] == fullrun.FROZEN_MANIFEST_SHA256
    strata = {*(s for s in ss.EVAL_SETS if s != ss.GALGAME), *(f"galgame_{v}" for v in ss.GALGAME_VIEWS), "m4"}
    assert {t: set(v) for t, v in ss.prereg_baselines(pr).items()} == {t: strata for t in ss.TEACHERS}
