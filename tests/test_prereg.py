"""kitsune/prereg.py: the study's pre-registered rules (study/PREREG.json + .md) and the numbers the study boxes write
(max_steps from calibration, the LR edge rule, PREREG_numbers_<box>.json), with the owner's changes of 2026-09-26.
CPU only."""
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT  # noqa: E402

from kitsune import prereg  # noqa: E402

CONTRACT_RUNS = {"study-t06": ("aed", "pruned_kept"), "study-t03": ("aed", "pruned_kept"),
                 "study-bridge": ("aed", "scratch"), "study-t01": ("aed", "scratch"),
                 "study-t01-s1235": ("aed", "scratch"), "study-t005": ("aed", "scratch"),
                 "study-p03": ("ctc", "pruned_kept"), "study-p01": ("ctc", "pruned_lost"),
                 "study-p005": ("ctc", "pruned_lost")}


# the owner's two-box split of 2026-09-26 as the wave-2 contract gives it (CONTRACT.md 6), with box B's calibrate list
# in its measuring order: the box's own runs first, its reference study-t06 last (prereg.calibration_groups)
OWNER_BOXES = {
    "A": {"runs": ["study-t06", "study-t03", "study-t01", "study-t005"], "probe_classes": ["kept-t03", "scratch"],
          "calibrate": ["study-t06", "study-t03", "study-t01", "study-t005"], "reference": "study-t06",
          "numbers_file": "PREREG_numbers_A.json", "extras": ["anchor"]},
    "B": {"runs": ["study-p03", "study-p01", "study-p005", "study-bridge"],
          "probe_classes": ["lost", "kept-p03", "bridge"],
          "calibrate": ["study-p03", "study-p01", "study-p005", "study-bridge", "study-t06"], "reference": "study-t06",
          "numbers_file": "PREREG_numbers_B.json", "extras": ["speed"]},
    "replicate": {"runs": ["study-t01-s1235"], "probe_classes": [], "calibrate": [], "numbers_from": "A",
                  "numbers_file": "PREREG_numbers_replicate.json", "extras": []}}
T_MODEL = {"study-t06": 1.079, "study-t03": 0.50, "study-bridge": 0.52, "study-t01": 0.369, "study-t005": 0.296,
           "study-p03": 0.62, "study-p01": 0.318, "study-p005": 0.238}  # the modelled step times (STUDY.md 2.4)


def calib(box: str | None = None, t_ref: float = 1.079, **over) -> dict:
    """A plausible calibration table at the modelled step times: every run of the study on one host (box None), or
    the runs a box calibrates (BOXES[box]["calibrate"]; study-t06 measured at t_ref on that host)."""
    runs = [r for r in T_MODEL] if box is None else prereg.BOXES[box]["calibrate"]
    t = dict(T_MODEL, **{"study-t06": t_ref})
    out = {run: {"t_step_s": t[run], "micro_audio_s": prereg.RUNS[run]["micro_audio_s"], "data_wait_frac": 0.01,
                 "steps_measured": 200} for run in runs}
    for run, c in over.items():
        out[run] = dict(out[run], **c)
    return out


def chosen_probes(box: str | None = None) -> dict:
    """Probe results whose winners all lie inside their grids after the edge rule (two of them extended once); only
    the box's probe classes with `box`."""
    p = {"scratch": {5e-4: 3.2, 1e-3: 3.0, 2e-3: 3.1},
         "bridge": {2e-4: 3.5, 4e-4: 3.3, 1e-3: 3.2, 2e-3: 3.4},  # 1e-3 won the grid's top edge, 2e-3 its extension
         "lost": {1e-4: 2.9, 3e-4: 2.7, 1e-3: 2.8},
         "kept-t03": {5e-5: 1.3, 1e-4: 1.2, 2e-4: 1.25, 4e-4: 1.3},  # 1e-4 won the bottom edge, 5e-5 its extension
         "kept-p03": {1e-4: 1.1, 2e-4: 1.0, 4e-4: 1.05}}
    return p if box is None else {c: p[c] for c in prereg.BOXES[box]["probe_classes"]}


def h(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def pool_by_cap(rule: int = 53, top: int = 53) -> dict:
    """pool_hours_if_capped.reazon_large as the sidecar holds it: N -> the pool, reaching 1,010 h first at `rule`."""
    return {str(n): prereg.POOL_MIN_HOURS - rule + n for n in range(1, top + 1)}


def fake_sidecar() -> dict:
    """A sidecar as scripts/make_selection.py writes it for the pre-registered selection: every check of
    prereg.sidecar_problems passes (reazon_large at 53, the smallest N whose pool holds >= 1,010 h)."""
    sets = prereg.STUDY_EVAL_SETS
    views = {v: {"n": n, "ids_sha256": h(v), "n_with_ref": n} for v, n in (("neutral", 810), ("all", 1000),
                                                                         ("label_box", 900))}
    cer = {"cer": 0.05, "edits": 5, "ref_chars": 100, "n": 10, "n_empty_ref": 0}
    return {"schema": prereg.STUDY_SCHEMA, "seed": 1234, "sources": list(prereg.STUDY_SOURCES),
            "eval_sets": list(sets), "recipe": prereg.registered_recipe(),
            "selection": {"path": "labels/full/selections/study_1000h.parquet", "sha256": h("selection")},
            "manifest": {"path": "labels/full/selections/study_manifest.json", "sha256": h("manifest")},
            "extent": {"name": "full", "inputs": {"galgame": 3, "reazon_large": 53, "emilia_yodas": "300h",
                                                  "emilia_nc": 8}},
            "ids_sha256": {"train": h("train"), "probe": h("probe"), "eval": {s: h(s) for s in sets}},
            "n": {"train": 700000, "probe": 1500, "eval": {s: 1000 for s in sets}},
            "galgame_views": views,
            "draw": {"budget_s": 3_600_000.0, "drawn_s": 3_599_990.0, "pool_s": 1010.0 * 3600, "pool_utts": 710000,
                     "drawn_utts": 700000},
            "baselines": {t: {**{k: dict(cer) for k in prereg.STRATA}, "m4": 0.05} for t in prereg.TEACHERS},
            "details": {"pool_hours_if_capped": {"reazon_large": pool_by_cap()}}}


def test_the_committed_rules_are_the_generators():
    """study/PREREG.json and .md are what `python -m kitsune.prereg --write study/` writes (the filled parts, if any,
    as committed); until the labels are sealed exactly the sidecar's fields are pending."""
    ok, r = prereg.check_rules(ROOT / "study")
    assert ok, "study/PREREG.* is stale: python -m kitsune.prereg --write study/"
    left = prereg.pending(r)
    assert left and all(p.split(".")[0] in ("manifest", "baselines") or p == "data.extent.inputs.reazon_large"
                        for p in left)
    assert prereg.main(["--check", str(ROOT / "study")]) == 0


def test_the_rules_carry_the_contract_and_the_design():
    r = prereg.rules()
    assert {run: (s["family"], s["init_class"]) for run, s in r["runs"].items()} == CONTRACT_RUNS
    assert r["runs"]["study-t06"]["params_total"] == 616_963_328 and r["runs"]["study-p005"]["params_total"] == \
        52_190_209
    # T-0.3B and the bridge: B8x2560 + decoder {0,2,5,7} (the owner's decision of 2026-09-26), about 301.82M
    for run in ("study-t03", "study-bridge"):
        assert (r["runs"][run]["params_total"], r["runs"][run]["params_non_embedding"]) == (301_822_208, 283_996_416)
    shape = r["runs"]["study-t03"]["shape"]
    assert "[0,7,13,20,27,34,40,47]" in shape and "{0,2,5,7}" in shape
    assert set(r["lr_probes"]["classes"]) == {"scratch", "bridge", "lost", "kept-t03", "kept-p03"}
    for p in r["lr_probes"]["classes"].values():
        assert p["warmup"] + p["stable"] + p["cooldown"] == p["max_steps"]
        assert p["cooldown"] == round(0.2 * p["max_steps"]) and p["warmup"] < p["max_steps"] - p["cooldown"]
    assert r["lr_probes"]["classes"]["kept-t03"]["runs"] == ["study-t03", "study-t06"]
    assert r["calibration"]["reference_run"] == "study-t06" and r["calibration"]["reference_steps"] == 9366
    assert len(r["decisions"]) == 30 and r["decisions"]["1"]["option"] == "a" and r["limit_rule"]["delta"] == 0.10
    assert r["decisions"]["17"]["option"] == "c" and "B8x2560 + decoder {0,2,5,7}" in r["decisions"]["17"]["answer"]
    assert "B10x2560" in r["decisions"]["17"]["changed_2026_09_26"]["was"] and "was" in r["decisions"]["2"][
        "changed_2026_09_26"]
    assert r["prereg_version"] == 2 and len(r["owner_changes"]["changes"]) == 5
    assert r["noise"]["sigma_prior"] == 0.016 and r["branch"] == dict(r["branch"], resume_frac=0.4, end_frac=0.5)
    data = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
    assert data["selection_recipe"] == r["selection"]["recipe"], "study/data.json carries the registered recipe"
    assert data["selection_recipe"]["study"] == prereg.STUDY_SELECTION
    assert (data["sources"], data["eval_sets"]) == (r["data"]["sources"], r["data"]["eval_sets"])
    assert {k: v for k, v in data["extent"]["inputs"].items() if k != "reazon_large"} == {
        k: v for k, v in r["data"]["extent"]["inputs"].items() if k != "reazon_large"}
    assert r["data"]["extent"]["inputs"]["reazon_large"] == prereg.PENDING
    assert prereg.rules() == r, "deterministic"


def test_a_sidecar_fills_every_pending_field(tmp_path):
    sc = fake_sidecar()
    assert prereg.sidecar_problems(sc) == []
    r = prereg.rules(sc)
    assert prereg.pending(r) == []
    m = r["manifest"]
    assert m["status"] == "filled" and m["sets"]["galgame"] == {"ids_sha256": h("galgame"), "n": 1000}
    assert list(m["sets"]) == prereg.STUDY_EVAL_SETS, "the manifest's sets: the eval sets only (study_stats reads it)"
    assert m["probe"] == {"ids_sha256": h("probe"), "n": 1500}
    assert m["train"]["hours"] == pytest.approx(3_599_990 / 3600)
    assert m["galgame_views"]["neutral"] == {"ids_sha256": h("neutral"), "n": 810}
    assert (m["selection_sha256"], m["manifest_sha256"], m["pool_hours"]) == (h("selection"), h("manifest"), 1010.0)
    assert r["baselines"]["cohere"]["eval_jsut"] == 0.05 and r["baselines"]["parakeet-ctc"]["m4"] == 0.05
    assert set(r["baselines"]) == {"status", "source", *prereg.TEACHERS}
    assert set(r["baselines"]["parakeet-tdt"]) == {*prereg.STRATA, "m4"}
    assert list(r["data"]["extent"]["inputs"]) == ["reazon_large", "emilia_yodas", "emilia_nc", "galgame"]
    assert r["data"]["extent"]["inputs"]["reazon_large"] == 53
    # the filled commit checks against itself without the sidecar (regenerate), and a hand edit of a rule is caught
    sha = prereg.write_rules(tmp_path, sc)
    assert prereg.check_rules(tmp_path)[0] and prereg.check_rules(tmp_path, sc)[0]
    assert sha == prereg.rules_sha256(tmp_path / prereg.RULES_JSON)
    good = json.loads((tmp_path / prereg.RULES_JSON).read_text(encoding="utf-8"))
    for edit in (lambda r: r["limit_rule"].update(delta=0.2),  # a rule
                 lambda r: r["manifest"]["sets"].pop("eval_emilia"),  # a filled part losing a set
                 lambda r: r["baselines"]["cohere"].pop("galgame_all"),
                 lambda r: r["data"]["extent"]["inputs"].update(emilia_nc=9),  # a registered input
                 lambda r: r["data"]["extent"]["inputs"].update(reazon_large="lots")):
        edited = copy.deepcopy(good)
        edit(edited)
        (tmp_path / prereg.RULES_JSON).write_bytes(prereg.rules_json(edited))
        assert not prereg.check_rules(tmp_path)[0]
        assert prereg.main(["--check", str(tmp_path)]) == 1


def _cap(n, rule=53):
    def f(sc):
        sc["extent"]["inputs"]["reazon_large"] = n
        sc["details"]["pool_hours_if_capped"]["reazon_large"] = pool_by_cap(rule, n)
        sc["draw"]["pool_s"] = pool_by_cap(rule, n)[str(n)] * 3600
    return f


MUTATIONS = [  # (what, an edit of the registered sidecar, the refusal it must give)
    ("the smoke recipe", lambda sc: sc["recipe"]["study"].update(draw_audio_s=3600), "recipe"),
    ("another F0", lambda sc: sc["recipe"].update(agree_max=0.6), "recipe"),
    ("another seed", lambda sc: sc.update(seed=99), "seed 99"),
    ("three sources", lambda sc: sc.update(sources=["reazon_small", "emilia_yodas", "galgame"]), "sources"),
    ("eval_emilia gone", lambda sc: (sc["ids_sha256"]["eval"].pop("eval_emilia"), sc["n"]["eval"].pop("eval_emilia"),
                                     sc.update(eval_sets=[s for s in sc["eval_sets"] if s != "eval_emilia"])),
     "eval_emilia"),
    ("an empty eval set", lambda sc: sc["n"]["eval"].update(eval_cv8=0), "eval_cv8"),
    ("another extent", lambda sc: sc["extent"].update(name="sub3k"), "extent 'sub3k'"),
    ("another input", lambda sc: sc["extent"]["inputs"].update(emilia_yodas="100h"), "extent 'full'"),
    ("an input gone", lambda sc: sc["extent"]["inputs"].pop("galgame"), "extent 'full'"),
    ("a cap over the rule", _cap(54), "the cap rule gives 53"),
    ("a cap under the rule", _cap(52), "no reazon_large cap up to 52"),
    ("a pool of 1,000 h at the cap", _cap(53, rule=60), "no reazon_large cap up to 53"),
    ("no cap readout", lambda sc: sc.pop("details"), "no reazon_large cap up to 53"),
    ("a pool that is not the cap's", lambda sc: sc["draw"].update(pool_s=1100.0 * 3600), "differs from the draw"),
    ("a draw over its budget", lambda sc: sc["draw"].update(drawn_s=3_600_001.0), "draw"),
    ("a view gone", lambda sc: sc["galgame_views"].pop("label_box"), "Galgame view label_box"),
    ("an empty view", lambda sc: sc["galgame_views"]["neutral"].update(n=0), "Galgame view neutral"),
    ("a baseline gone", lambda sc: sc["baselines"]["parakeet-tdt"].pop("galgame_all"), "parakeet-tdt"),
    ("a NaN baseline", lambda sc: sc["baselines"]["cohere"]["eval_jsut"].update(cer=float("nan")), "cohere"),
    ("the old system names", lambda sc: sc["baselines"].update(parakeet_ctc=sc["baselines"].pop("parakeet-ctc")),
     "parakeet-ctc"),
    ("a short probe", lambda sc: sc["n"].update(probe=1499), "probe"),
    ("not a sha256", lambda sc: sc["ids_sha256"].update(train="c" * 63), "train"),
    ("another file", lambda sc: sc["selection"].update(path="labels/full/selections/other.parquet"), "selection"),
    ("another schema", lambda sc: sc.update(schema=2), "schema"),
]


@pytest.mark.parametrize("what,edit,match", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_the_fill_takes_only_the_registered_selection(what, edit, match):
    """A sidecar whose selection is not the one the rules describe (another recipe, seed, source list, extent or cap;
    an eval set, view, teacher or stratum missing) must not fill the PREREG: its hashes would stand under rules that
    say something else."""
    sc = fake_sidecar()
    edit(sc)
    assert any(match in p for p in prereg.sidecar_problems(sc)), prereg.sidecar_problems(sc)
    with pytest.raises(ValueError, match="not the pre-registered study selection"):
        prereg.rules(sc)


def test_cap_rule():
    """The smallest N whose pool after all filters holds >= 1,010 h; string keys (JSON) or ints; None if none does."""
    assert prereg.cap_rule(pool_by_cap(53)) == 53 and prereg.cap_rule({1: 1010.0, 2: 1011.0}) == 1
    assert prereg.cap_rule({"9": 1009.99}) is None and prereg.cap_rule({}) is None
    assert prereg.cap_rule({"10": 1012.0, "9": 1010.0, "8": 1009.0}) == 9


def test_the_cli_refuses_a_foreign_sidecar(tmp_path, capsys):
    sc = fake_sidecar()
    sc["seed"] = 99
    side = tmp_path / "side.json"
    side.write_text(json.dumps(sc), encoding="utf-8")
    out = tmp_path / "study"
    assert prereg.main(["--write", str(out), "--sidecar", str(side)]) == 2
    assert "refused" in capsys.readouterr().err and not (out / prereg.RULES_JSON).exists()


def test_the_analysis_block_is_what_study_stats_reads():
    """The machine-read settings (STUDY.md 4.3-4.7) in the form kitsune.study_stats.settings_from_prereg reads, from
    the same constants as the prose; limit_rule's replicate text sits under another key, so it cannot be read as the
    [replicate run, original run] pair."""
    r = prereg.rules()
    a = r["analysis"]
    assert (a["delta"], a["deltas"]) == (0.10, [0.05, 0.10, 0.20])
    assert a["sigma_run"] == {"prior": 0.016, "factor": 0.886, "replicate": ["study-t01-s1235", "study-t01"]}
    assert a["bootstrap"] == {"B": 10000, "seed": 1234, "z": 1.96}
    assert (a["teacher_bars"], a["anchor_flag_rel"]) == ([1.2, 1.5], 0.05)
    assert a["m4_strata"] == ["eval_jsut", "eval_cv8", "eval_reazon", "galgame_neutral"]
    assert "replicate" not in r["limit_rule"] and "replicate" not in r and "delta" not in r
    assert r["limit_rule"]["delta"] == a["delta"] and r["noise"]["sigma_prior"] == a["sigma_run"]["prior"]


@pytest.mark.skipif(importlib.util.find_spec("kitsune.study_stats") is None,
                    reason="kitsune.study_stats (WP5a) is not on this branch yet")
def test_study_stats_reads_the_prereg():
    """Once WP5a is merged: its readers take the analysis settings, the baselines (teacher x stratum, CONTRACT.md 5
    names) and the manifest hashes from a filled PREREG, and nothing from a pending one."""
    from kitsune import study_stats as ss

    r = json.loads(prereg.rules_json(prereg.rules(fake_sidecar())))
    settings, sources = ss.settings_from_prereg(r)
    assert settings["delta_primary"] == 0.10 and settings["replicate"] == ["study-t01-s1235", "study-t01"]
    assert settings["boot_b"] == 10000 and settings["boot_seed"] == 1234 and settings["sigma_run_prior"] == 0.016
    analysis = ("delta_primary", "deltas", "sigma_run_prior", "sigma_hat_factor", "replicate", "boot_b", "boot_seed",
                "z", "teacher_bars", "anchor_flag_rel")
    assert all(sources[k].startswith("PREREG.json:analysis.") for k in analysis if k in sources)
    base = ss.prereg_baselines(r)
    assert set(base) == set(prereg.TEACHERS) and set(base["cohere"]) >= set(prereg.M4_SETS)
    reader = getattr(ss, "prereg_manifest", None) or ss.prereg_manifest_hashes
    got = reader(r)
    sets = got.get("sets", got)
    assert {s: (v if isinstance(v, str) else v.get("ids_sha256")) for s, v in sets.items()} == {
        s: h(s) for s in prereg.STUDY_EVAL_SETS}
    pend = json.loads(prereg.rules_json(prereg.rules()))
    assert ss.prereg_baselines(pend) == {} and ss.settings_from_prereg(pend)[0] == settings


def test_rules_sha256_ignores_the_checkout_line_endings(tmp_path):
    """Git checks the committed JSON out with CRLF on the Windows laptop and LF on the box: one rules_sha256."""
    sha = prereg.write_rules(tmp_path)
    data = (tmp_path / prereg.RULES_JSON).read_bytes()
    (tmp_path / "crlf.json").write_bytes(data.replace(b"\n", b"\r\n"))
    assert prereg.rules_sha256(tmp_path / "crlf.json") == sha == prereg.rules_sha256(tmp_path / prereg.RULES_JSON)
    (tmp_path / prereg.RULES_MD).write_bytes((tmp_path / prereg.RULES_MD).read_bytes().replace(b"\n", b"\r\n"))
    (tmp_path / prereg.RULES_JSON).write_bytes(data.replace(b"\n", b"\r\n"))
    assert prereg.check_rules(tmp_path)[0]


def test_max_steps():
    """max_steps_i = round_to_10(9366 x t_ref / t_i), half up to a multiple of 10 (the T/2 branch's steps are then
    exact); the reference run gets 9,370; the replicate takes study-t01's; a missing reference or a non-positive step
    time refuses."""
    c = calib()
    m = prereg.max_steps(c)
    assert m["study-t06"] == 9370 and set(m) == set(prereg.RUNS)
    assert m["study-t01"] == 10 * math.floor(9366 * 1.079 / 0.369 / 10 + 0.5) == 27390 and m["study-t01-s1235"] == 27390
    assert all(v % 10 == 0 for v in m.values())
    # the replicate always takes study-t01's: its own step time (were it measured) is ignored, and refused as input
    with_rep = dict(c, **{"study-t01-s1235": dict(c["study-t01"], t_step_s=0.2)})
    assert prereg.max_steps(with_rep) == m
    assert prereg.calibration_problems(with_rep) == [
        "study-t01-s1235: not calibrated by the rules (it takes study-t01's max_steps from box A's numbers)"]
    assert prereg.max_steps({"study-t06": {"t_step_s": 1.0}, "study-t03": {"t_step_s": 0.5}}) == {
        "study-t06": 9370, "study-t03": 18730}
    assert prereg.max_steps({"a": {"t_step_s": 2.0}, "b": {"t_step_s": 4.0}}, t_ref_run="a", ref_steps=90) == {
        "a": 90, "b": 50}  # 45 rounds half up to 50
    assert [prereg.round_to(x) for x in (9366, 9365, 9364.99, 27_385, 5, 4.99)] == [9370, 9370, 9360, 27_390, 10, 0]
    with pytest.raises(ValueError, match="reference run"):
        prereg.max_steps({"study-t03": {"t_step_s": 1.0}})
    with pytest.raises(ValueError, match="positive"):
        prereg.max_steps(calib(**{"study-t03": {"t_step_s": 0.0}}))


def test_the_branch_of_a_multiple_of_10_is_a_t_half_run():
    """With max_steps M a multiple of 10 the T/2 branch resumes at 0.4 M, ends at 0.5 M, and a run whose budget is
    M/2 starts its 20 % cooldown at 0.8 x M/2 = 0.4 M: every step is an exact integer, so the branch's WSD schedule is
    exactly that run's (4.2 / 2.5)."""
    r = prereg.rules()
    assert (r["branch"]["resume_frac"], r["branch"]["end_frac"]) == (0.4, 0.5)
    assert r["calibration"]["max_steps_multiple"] == 10
    for m in [*prereg.max_steps(calib()).values(), *prereg.max_steps(calib("B", t_ref=1.1), box="B").values()]:
        resume, end = round(0.4 * m), round(0.5 * m)  # what the trainer and kitsune.study_stats compute
        assert 10 * resume == 4 * m and 10 * end == 5 * m  # exact, no rounding
        assert 10 * round(0.8 * end) == 8 * end and round(0.8 * end) == resume  # the T/2 run's cooldown start
    assert round(0.4 * 9366) * 10 != 4 * 9366  # unrounded, the reference's own branch would not be exact


def test_calibration_problems():
    assert prereg.calibration_problems(calib()) == []
    assert prereg.calibration_problems(calib(**{"study-p01": {"data_wait_frac": 0.05}})) == [
        "study-p01: loader-bound (data_wait_frac 0.050 >= 0.05)"]
    assert prereg.calibration_problems(calib(**{"study-t03": {"steps_measured": 150}})) == [
        "study-t03: 150 steps measured, the window is 200"]
    c = calib()
    del c["study-bridge"]
    assert prereg.calibration_problems(c) == ["study-bridge: not calibrated"]
    assert prereg.calibration_problems(dict(calib(), other={"t_step_s": 1})) == ["other: not a study run"]
    assert prereg.calibration_problems(calib(**{"study-p01": {"data_wait_frac": float("nan")}})) == [
        "study-p01: data_wait_frac nan is not a fraction"]
    assert prereg.calibration_problems(calib(**{"study-t03": {"steps_measured": None, "micro_audio_s": 0}})) == [
        "study-t03: micro_audio_s 0 is not a positive number", "study-t03: None steps measured, the window is 200"]


def test_the_grids_have_three_points():
    """The owner's decision of 2026-09-26: kept-t03 and kept-p03 1e-4 / 2e-4 / 4e-4, bridge 2e-4 / 4e-4 / 1e-3;
    scratch and lost unchanged. With 3 points a winner can lie inside the registered grid for every class."""
    grids = {c: p["grid"] for c, p in prereg.rules()["lr_probes"]["classes"].items()}
    assert grids == {"scratch": [5e-4, 1e-3, 2e-3], "bridge": [2e-4, 4e-4, 1e-3], "lost": [1e-4, 3e-4, 1e-3],
                     "kept-t03": [1e-4, 2e-4, 4e-4], "kept-p03": [1e-4, 2e-4, 4e-4]}
    for cls, g in grids.items():
        mid = {lr: (1.0 if lr == g[1] else 2.0) for lr in g}
        d = prereg.choose_lr({cls: mid})[cls]
        assert (d["decision"], d["lr"]) == ("chosen", g[1])
        for w, ext in ((g[0], g[0] / 2), (g[-1], g[-1] * 2)):  # an edge winner: one more point, x2 or /2
            d = prereg.choose_lr({cls: {lr: (1.0 if lr == w else 2.0) for lr in g}})[cls]
            assert d["decision"] == "extend" and d["next_lr"] == pytest.approx(ext)
    assert prereg.probe_run_name("bridge", 2e-4) == "probe-bridge-2e-4"


def test_choose_lr_edge_rule():
    """Inside the grid: chosen. On an edge of the registered grid: extend by one point (x2 at the top, /2 at the
    bottom). After that extension: chosen if the winner is inside, halt if it is at an edge again."""
    scratch = {5e-4: 3.2, 1e-3: 3.0, 2e-3: 3.1}
    assert prereg.choose_lr({"scratch": scratch})["scratch"]["decision"] == "chosen"
    assert prereg.choose_lr({"scratch": scratch})["scratch"]["lr"] == 1e-3
    top = prereg.choose_lr({"scratch": {5e-4: 3.2, 1e-3: 3.1, 2e-3: 3.0}})["scratch"]
    assert (top["decision"], top["next_lr"], top["lr"]) == ("extend", 4e-3, None)
    bottom = prereg.choose_lr({"lost": {1e-4: 2.6, 3e-4: 2.7, 1e-3: 2.8}})["lost"]
    assert (bottom["decision"], bottom["next_lr"]) == ("extend", 5e-5)
    # after the extension
    ext = {5e-4: 3.2, 1e-3: 3.1, 2e-3: 3.0, 4e-3: 3.05}
    assert prereg.choose_lr({"scratch": ext})["scratch"] == dict(
        prereg.choose_lr({"scratch": ext})["scratch"], decision="chosen", lr=2e-3)
    halt = prereg.choose_lr({"scratch": {**ext, 4e-3: 2.9}})["scratch"]
    assert (halt["decision"], halt["lr"], halt["winner"]) == ("halt", None, 4e-3)
    down = prereg.choose_lr({"lost": {5e-5: 2.65, 1e-4: 2.6, 3e-4: 2.7, 1e-3: 2.8}})["lost"]
    assert (down["decision"], down["lr"]) == ("chosen", 1e-4)
    # the bridge's grid ends at 1e-3: an edge winner there extends to 2e-3, and 2e-3 winning again halts
    assert prereg.choose_lr({"bridge": {2e-4: 3.5, 4e-4: 3.3, 1e-3: 3.2, 2e-3: 3.4}})["bridge"]["lr"] == 1e-3
    assert prereg.choose_lr({"bridge": {2e-4: 3.5, 4e-4: 3.3, 1e-3: 3.2, 2e-3: 3.1}})["bridge"]["decision"] == "halt"
    with pytest.raises(ValueError, match="no result for the grid point"):  # the 2-point grid of wave 1 is not enough
        prereg.choose_lr({"bridge": {4e-4: 3.3, 1e-3: 3.2, 2e-3: 3.4}})
    # JSON keys, NaN and ties
    assert prereg.choose_lr({"scratch": {"5e-4": 3.2, "1e-3": 3.0, "2e-3": 3.1}})["scratch"]["lr"] == 1e-3
    assert prereg.choose_lr({"scratch": {5e-4: 3.2, 1e-3: float("nan"), 2e-3: 3.1}})["scratch"]["decision"] == "extend"
    tie = prereg.choose_lr({"lost": {1e-4: 2.9, 3e-4: 2.7, 1e-3: 2.7}})["lost"]
    assert (tie["decision"], tie["lr"]) == ("chosen", 3e-4)
    # probes that do not follow the rule
    with pytest.raises(ValueError, match="no result for the grid point"):
        prereg.choose_lr({"scratch": {5e-4: 3.2, 1e-3: 3.0}})
    with pytest.raises(ValueError, match="edge rule allows only 4e-3"):
        prereg.choose_lr({"scratch": {5e-4: 3.2, 1e-3: 3.1, 2e-3: 3.0, 3e-3: 2.0}})
    with pytest.raises(ValueError, match="nothing"):
        prereg.choose_lr({"scratch": {**scratch, 4e-3: 2.0}})
    with pytest.raises(ValueError, match="not a probe class"):
        prereg.choose_lr({"medium": {1e-3: 1.0}})


def test_the_probe_extension_halts_at_most_once():
    """The box's loop: probe the grid, extend while the rule says so, then chosen or halt - never a second extension."""
    for truth in (lambda lr: (math.log10(lr) + 5) ** 2, lambda lr: -math.log10(lr), lambda lr: math.log10(lr)):
        res = {lr: truth(lr) for lr in prereg.PROBES["scratch"]["grid"]}
        d = prereg.choose_lr({"scratch": res})["scratch"]
        if d["decision"] == "extend":
            res[d["next_lr"]] = truth(d["next_lr"])
            d = prereg.choose_lr({"scratch": res})["scratch"]
        assert d["decision"] in ("chosen", "halt")


def test_run_lrs_maps_the_classes_to_the_runs():
    lrs = prereg.run_lrs(prereg.choose_lr(chosen_probes()))
    assert lrs["study-t06"] == lrs["study-t03"] == 1e-4 and lrs["study-p03"] == 2e-4 and lrs["study-bridge"] == 1e-3
    assert lrs["study-t01"] == lrs["study-t005"] == lrs["study-t01-s1235"] == 1e-3
    assert lrs["study-p01"] == lrs["study-p005"] == 3e-4
    probes = dict(chosen_probes(), scratch={5e-4: 3.2, 1e-3: 3.1, 2e-3: 3.0})
    with pytest.raises(ValueError, match="scratch has no chosen LR"):
        prereg.run_lrs(prereg.choose_lr(probes))


def test_lr_tags_and_probe_names():
    assert [prereg.lr_tag(x) for x in (1e-3, 4e-4, 2.5e-4, 5e-5, 2e-3, 1e-4)] == [
        "1e-3", "4e-4", "2.5e-4", "5e-5", "2e-3", "1e-4"]
    assert prereg.probe_run_name("kept-t03", 2e-4) == "probe-kept-t03-2e-4"


def test_write_numbers(tmp_path):
    """PREREG_numbers.json: the contract's keys, the numbers only as the rules derive them, rules_sha256 of the
    committed rules; refused against pending rules (unless a dry run) and for numbers the rules do not give."""
    rules_path = tmp_path / "study" / prereg.RULES_JSON
    prereg.write_rules(rules_path.parent, fake_sidecar())
    c, probes = calib(), chosen_probes()
    steps, lrs = prereg.max_steps(c), prereg.run_lrs(prereg.choose_lr(probes))
    out = tmp_path / "PREREG_numbers.json"
    sha = prereg.write_numbers(out, c, probes, lrs, steps, rules_path=rules_path, host="box-1")
    assert sha == prereg.file_sha256(out)
    n = json.loads(out.read_text(encoding="utf-8"))
    assert set(n) == {"box", "calibration", "max_steps", "lr_probes", "lr", "rules_sha256", "written_utc", "host"}
    assert set(n) == set(prereg.NUMBERS_KEYS) and n["box"] is None  # the whole study on one host
    assert n["max_steps"] == steps and n["lr"] == lrs and n["host"] == "box-1"
    assert n["rules_sha256"] == prereg.rules_sha256(rules_path)
    assert n["lr_probes"]["kept-t03"] == {"5e-5": 1.3, "1e-4": 1.2, "2e-4": 1.25, "4e-4": 1.3}
    assert n["calibration"]["study-t06"] == c["study-t06"]
    # the contract's parameter names work as keywords
    assert prereg.write_numbers(path=out, calibration=c, probes=probes, lrs=lrs, max_steps=steps,
                                rules_path=rules_path, host="box-1") == prereg.file_sha256(out)
    with pytest.raises(ValueError, match="is not the calibrated"):
        prereg.write_numbers(out, c, probes, lrs, dict(steps, **{"study-t03": 1}), rules_path=rules_path)
    with pytest.raises(ValueError, match="is not the probes' choice"):
        prereg.write_numbers(out, c, probes, dict(lrs, **{"study-p01": 1e-3}), steps, rules_path=rules_path)
    with pytest.raises(ValueError, match="loader-bound"):
        bad = calib(**{"study-t005": {"data_wait_frac": 0.2}})
        prereg.write_numbers(out, bad, probes, lrs, prereg.max_steps(bad), rules_path=rules_path)
    extend = dict(probes, scratch={5e-4: 3.2, 1e-3: 3.1, 2e-3: 3.0})
    with pytest.raises(ValueError, match="no chosen LR"):
        prereg.write_numbers(out, c, extend, lrs, steps, rules_path=rules_path)
    # a diverged probe loses and is written as null: the file stays strict JSON
    diverged = dict(probes, scratch={5e-4: float("nan"), 1e-3: 3.0, 2e-3: float("inf")})
    assert prereg.write_numbers(out, c, diverged, lrs, steps, rules_path=rules_path, host="box-1")

    def strict(name):
        raise ValueError(f"not JSON: {name}")

    n = json.loads(out.read_text(encoding="utf-8"), parse_constant=strict)
    assert n["lr_probes"]["scratch"] == {"5e-4": None, "1e-3": 3.0, "2e-3": None}
    # the replicate's max_steps is study-t01's, and nothing else passes
    with pytest.raises(ValueError, match="is not the calibrated"):
        prereg.write_numbers(out, c, probes, lrs, dict(steps, **{"study-t01-s1235": steps["study-t01"] + 1}),
                             rules_path=rules_path)
    pending = tmp_path / "pending"
    prereg.write_rules(pending)
    with pytest.raises(ValueError, match="pending field"):
        prereg.write_numbers(out, c, probes, lrs, steps, rules_path=pending / prereg.RULES_JSON)
    assert prereg.write_numbers(out, c, probes, lrs, steps, rules_path=pending / prereg.RULES_JSON,
                                allow_pending=True)


def test_the_cli_writes_and_checks(tmp_path, capsys):
    assert prereg.main(["--write", str(tmp_path)]) == 0
    assert "pending field" in capsys.readouterr().out
    assert prereg.main(["--check", str(tmp_path)]) == 0
    (tmp_path / prereg.RULES_MD).write_text("stale\n", encoding="utf-8")
    assert prereg.main(["--check", str(tmp_path)]) == 1
    side = tmp_path / "side.json"
    side.write_text(json.dumps(fake_sidecar()), encoding="utf-8")
    assert prereg.main(["--write", str(tmp_path), "--sidecar", str(side)]) == 0
    assert "; 0 pending field(s)" in capsys.readouterr().out
    assert prereg.main(["--check", str(tmp_path)]) == 0


# ------------------------------------------------------------------------------------------------ the owner's changes


def test_the_boxes_are_the_owners_split():
    """rules()["boxes"] is exactly the owner's two-box split (other packages read it); every run trains on exactly one
    box, its RUNS box; every probe class is probed on the box that trains the runs taking its LR, on a run of that
    box; both boxes measure study-t06 as their reference; the replicate probes and calibrates nothing and takes box A's
    numbers."""
    r = prereg.rules()
    assert r["boxes"] == OWNER_BOXES and prereg.BOXES == OWNER_BOXES
    assert json.loads(prereg.rules_json(r))["boxes"] == OWNER_BOXES  # as PREREG.json stores it
    trained = [run for spec in OWNER_BOXES.values() for run in spec["runs"]]
    assert sorted(trained) == sorted(prereg.RUNS) and len(trained) == len(set(trained))
    for box, spec in OWNER_BOXES.items():
        assert all(prereg.RUNS[run]["box"] == box == prereg.box_of(run) for run in spec["runs"])
        for cls in spec["probe_classes"]:
            assert prereg.PROBES[cls]["probed_on"] in spec["runs"] and prereg.probe_box(cls) == box
            assert r["lr_probes"]["classes"][cls]["box"] == box
        assert set(spec["runs"]) <= set(spec["calibrate"]) or box == "replicate"
    assert sorted(c for spec in OWNER_BOXES.values() for c in spec["probe_classes"]) == sorted(prereg.PROBES)
    for run, spec in prereg.RUNS.items():  # the LR a run takes comes from its own box, or (the replicate) box A
        assert prereg.probe_box(spec["lr_from"]) == (spec["box"] if spec["box"] != "replicate" else "A")
    assert set(OWNER_BOXES["B"]["calibrate"]) - set(OWNER_BOXES["B"]["runs"]) == {"study-t06"}  # the reference only
    for spec in OWNER_BOXES.values():  # the box's own runs first: measured together, as their wave trains
        assert spec["calibrate"][:len(spec["runs"])] == spec["runs"] or not spec["calibrate"]
    assert r["numbers"]["files"] == {"A": "PREREG_numbers_A.json", "B": "PREREG_numbers_B.json",
                                     "replicate": "PREREG_numbers_replicate.json"}


def test_the_calibration_groups_are_registered():
    """How each box times its runs is fixed before box A writes its numbers (the replicate refuses A's file under other
    rules): box A measures its wave in one group; box B measures its wave exactly as it trains, then study-t06 beside
    three of its runs as unmeasured load, so every t_i is taken with four runs training. The rule is the list order in
    groups of the GPU count, as kitsune.study_queue runs it."""
    r = prereg.rules()
    groups = r["calibration"]["groups"]
    assert groups == {
        "A": [{"measured": ["study-t06", "study-t03", "study-t01", "study-t005"], "load": []}],
        "B": [{"measured": ["study-p03", "study-p01", "study-p005", "study-bridge"], "load": []},
              {"measured": ["study-t06"], "load": ["study-p03", "study-p01", "study-p005"]}],
        "replicate": []}
    assert r["calibration"]["gpus"] == {"A": 4, "B": 4, "replicate": 1}
    for box, grps in groups.items():
        assert [x for g in grps for x in g["measured"]] == OWNER_BOXES[box]["calibrate"]  # each run measured once
        assert all(len(g["measured"]) + len(g["load"]) == r["calibration"]["gpus"][box] for g in grps)
        if grps:  # the wave's runs are measured together, with nothing else training
            assert sorted(grps[0]["measured"]) == sorted(OWNER_BOXES[box]["runs"]) and grps[0]["load"] == []
    # the same rule on another GPU count: a short last group is filled from the list's head
    assert prereg.calibration_groups("B", 3) == [
        {"measured": ["study-p03", "study-p01", "study-p005"], "load": []},
        {"measured": ["study-bridge", "study-t06"], "load": ["study-p03"]}]
    with pytest.raises(ValueError, match="GPUs"):
        prereg.calibration_groups("A", 0)


def test_student_problems_checks_a_built_student_against_the_rules():
    """A box can refuse a stale student before its first step: every registered field of the student's meta (stage,
    family, init class, seed, the three counts) and, for a pruned student, the calibration ids its FFNs were ranked on
    (the Transcribe builder's calibration.importance_ids_sha256, the Parakeet builder's calibration.ids_sha256)."""
    def meta(run, **over):
        s = prereg.RUNS[run]
        m = dict(stage="complete", family=s["family"], init_class=s["init_class"], seed=s["seed"],
                 params_total=s["params_total"], params_non_embedding=s["params_non_embedding"],
                 closed_form_params=s["params_total"])
        if s["init_class"] != "scratch":
            key = "importance_ids_sha256" if s["family"] == "aed" else "ids_sha256"
            m["calibration"] = {key: prereg.CALIB_IDS_SHA256}
        return dict(m, **over)

    for run in prereg.RUNS:
        assert prereg.student_problems(run, meta(run)) == [], run
    assert {r for r, s in prereg.RUNS.items() if s.get("calib_ids_sha256")} == {
        "study-t06", "study-t03", "study-p03", "study-p01", "study-p005"}
    b10 = meta("study-t03", params_total=320_752_384, closed_form_params=320_752_384, params_non_embedding=302_926_592)
    assert [p.split(" ")[1] for p in prereg.student_problems("study-t03", b10)] == [
        "params_total", "params_non_embedding", "closed_form_params"]
    redrawn = meta("study-p01", calibration={"ids_sha256": "b06deb35" + "0" * 56})
    assert prereg.student_problems("study-p01", redrawn) == [
        "study-p01: FFNs ranked on calibration ids b06deb350000..., registered 5e31cd68bebc..."]
    assert prereg.student_problems("study-t01-s1235", meta("study-t01-s1235", seed=1234)) == [
        "study-t01-s1235: seed 1234, registered 1235"]
    assert prereg.student_problems("study-p03", meta("study-p03", stage="saved", init_class="pruned_lost")) == [
        "study-p03: stage 'saved', not complete", "study-p03: init_class 'pruned_lost', registered 'pruned_kept'"]


def test_box_aware_calibration_and_max_steps():
    """Each box measures study-t06 on its own host: box B's max_steps use box B's t_study-t06, so the same run on a
    slower host still gets the equal-compute budget of that host. A box refuses another box's runs and a missing one;
    box B's reference is calibrated but not in its max_steps; the replicate box is not calibrated at all."""
    a, b = calib("A"), calib("B", t_ref=1.2)
    assert prereg.calibration_problems(a, "A") == [] and prereg.calibration_problems(b, "B") == []
    ma, mb = prereg.max_steps(a, box="A"), prereg.max_steps(b, box="B")
    assert list(ma) == OWNER_BOXES["A"]["runs"] and list(mb) == OWNER_BOXES["B"]["runs"]
    assert ma["study-t06"] == 9370 and ma["study-t03"] == prereg.round_to(9366 * 1.079 / 0.50) == 20210
    assert mb["study-p01"] == prereg.round_to(9366 * 1.2 / 0.318) == 35340  # box B's own reference time
    assert mb["study-bridge"] == prereg.round_to(9366 * 1.2 / 0.52) and "study-t06" not in mb
    assert all(v % 10 == 0 for v in (*ma.values(), *mb.values()))
    assert prereg.calibration_problems(dict(a, **{"study-p03": b["study-p03"]}), "A") == [
        "study-p03: not calibrated on box A (it trains on box B)"]
    missing = {k: v for k, v in b.items() if k != "study-t06"}
    assert prereg.calibration_problems(missing, "B") == ["study-t06: not calibrated"]
    with pytest.raises(ValueError, match="reference run"):
        prereg.max_steps(missing, box="B")
    with pytest.raises(ValueError, match="study-bridge: not calibrated on box B"):
        prereg.max_steps({k: v for k, v in b.items() if k != "study-bridge"}, box="B")
    assert prereg.calibration_problems({}, "replicate") == []
    assert prereg.calibration_problems({"study-t01": a["study-t01"]}, "replicate") == [
        "study-t01: not calibrated on box replicate (it trains on box A)"]
    with pytest.raises(ValueError, match="replicate box is not calibrated"):
        prereg.max_steps(a, box="replicate")
    with pytest.raises(ValueError, match="not one of"):
        prereg.calibration_problems(a, "C")
    assert prereg.run_lrs(prereg.choose_lr(chosen_probes("A")), "A") == {
        "study-t06": 1e-4, "study-t03": 1e-4, "study-t01": 1e-3, "study-t005": 1e-3}
    assert prereg.run_lrs(prereg.choose_lr(chosen_probes("B")), "B") == {
        "study-p03": 2e-4, "study-p01": 3e-4, "study-p005": 3e-4, "study-bridge": 1e-3}
    with pytest.raises(ValueError, match="kept-t03 has no chosen LR"):  # box B's probes cannot give box A's LRs
        prereg.run_lrs(prereg.choose_lr(chosen_probes("B")), "A")
    with pytest.raises(ValueError, match="runs no probes"):
        prereg.run_lrs(prereg.choose_lr(chosen_probes("A")), "replicate")


def test_write_numbers_per_box(tmp_path):
    """Box A, box B and the replicate each write their own file (named by the box), with the box's calibration,
    max_steps, probes and LRs only; the replicate's takes study-t01's max_steps and the scratch LR from box A's file,
    records that file's sha256, and refuses a file of another box, other rules, or numbers that are not A's."""
    rules_path = tmp_path / "study" / prereg.RULES_JSON
    prereg.write_rules(rules_path.parent, fake_sidecar())
    shas = {}
    for box, t_ref in (("A", 1.079), ("B", 1.2)):
        c, probes = calib(box, t_ref=t_ref), chosen_probes(box)
        steps, lrs = prereg.max_steps(c, box=box), prereg.run_lrs(prereg.choose_lr(probes), box)
        path = tmp_path / OWNER_BOXES[box]["numbers_file"]
        shas[box] = prereg.write_numbers(path, c, probes, lrs, steps, box=box, rules_path=rules_path, host=f"h{box}")
        n = json.loads(path.read_text(encoding="utf-8"))
        assert set(n) == set(prereg.NUMBERS_KEYS) and n["box"] == box and shas[box] == prereg.file_sha256(path)
        assert list(n["max_steps"]) == sorted(OWNER_BOXES[box]["runs"]) and set(n["lr"]) == set(OWNER_BOXES[box]["runs"])
        assert set(n["calibration"]) == set(OWNER_BOXES[box]["calibrate"]) and set(n["lr_probes"]) == set(probes)
        with pytest.raises(ValueError, match="writes"):  # the file name is the box's
            prereg.write_numbers(tmp_path / "PREREG_numbers.json", c, probes, lrs, steps, box=box, rules_path=rules_path)
        with pytest.raises(ValueError, match="probes"):  # another box's classes
            other = chosen_probes("B" if box == "A" else "A")
            prereg.write_numbers(path, c, other, lrs, steps, box=box, rules_path=rules_path)
    with pytest.raises(ValueError, match="calibration"):  # box A cannot write box B's runs
        prereg.write_numbers(tmp_path / "PREREG_numbers_A.json", calib("B"), chosen_probes("A"), {}, {}, box="A",
                             rules_path=rules_path)
    a_path = tmp_path / "PREREG_numbers_A.json"
    a = json.loads(a_path.read_text(encoding="utf-8"))
    rep = tmp_path / "PREREG_numbers_replicate.json"
    steps, lrs, source = prereg.replicate_numbers(a_path)
    assert steps == {"study-t01-s1235": a["max_steps"]["study-t01"]} and lrs == {"study-t01-s1235": 1e-3}
    assert source == {"box": "A", "file": "PREREG_numbers_A.json", "sha256": shas["A"]}
    sha = prereg.write_numbers(rep, {}, {}, lrs, steps, box="replicate", numbers_from=a_path, rules_path=rules_path,
                               host="h1")
    n = json.loads(rep.read_text(encoding="utf-8"))
    assert sha == prereg.file_sha256(rep) and n["numbers_from"] == source and n["box"] == "replicate"
    assert n["calibration"] == {} and n["lr_probes"] == {} and n["max_steps"] == steps and n["lr"] == lrs
    assert set(n) == {*prereg.NUMBERS_KEYS, "numbers_from"}
    for bad, match in (
            (dict(steps={"study-t01-s1235": steps["study-t01-s1235"] + 10}), "is not the calibrated"),
            (dict(lrs={"study-t01-s1235": 2e-3}), "is not the probes' choice"),
            (dict(calibration=calib("A")), "no calibration"),
            (dict(numbers_from=None), "needs numbers_from"),
            (dict(numbers_from=tmp_path / "PREREG_numbers_B.json"), "box 'B'")):
        kw = dict(calibration={}, probes={}, lrs=lrs, max_steps=steps, numbers_from=a_path)
        kw.update({("max_steps" if k == "steps" else k): v for k, v in bad.items()})
        with pytest.raises(ValueError, match=match):
            prereg.write_numbers(rep, box="replicate", rules_path=rules_path, **kw)
    other_rules = tmp_path / "other" / prereg.RULES_JSON  # box A's file under other rules is not the replicate's input
    sc = fake_sidecar()
    sc["ids_sha256"]["train"] = h("another train set")
    prereg.write_rules(other_rules.parent, sc)
    with pytest.raises(ValueError, match="written under rules"):
        prereg.write_numbers(rep, {}, {}, lrs, steps, box="replicate", numbers_from=a_path, rules_path=other_rules)
    with pytest.raises(ValueError, match="replicate box's input"):
        prereg.write_numbers(a_path, calib("A"), chosen_probes("A"), {}, {}, box="A", numbers_from=a_path,
                             rules_path=rules_path)


def test_the_size_ladder_and_the_init_class_rule():
    """STUDY.md 1.3 from the registered counts (h = log2 of the total-parameter ratio: T 0.6 -> 0.3 is 1.031 halvings
    with the B8 shape); the 90 % step-0 rule classifies the Parakeet students only, and the recorded gate numbers agree
    with every run's init class (the pruned Transcribe students are over 90 % but function kept by the owner)."""
    r = prereg.rules()
    h = {(s["big"], s["small"]): s["h"] for s in r["size_ladder"]}
    assert h[("study-t06", "study-t03")] == round(math.log2(616_963_328 / 301_822_208), 3) == 1.031
    assert h[("study-t03", "study-t01")] == h[("study-bridge", "study-t01")] == 1.537
    assert (h[("study-t01", "study-t005")], h[("parakeet-ctc", "study-p03")], h[("study-p03", "study-p01")],
            h[("study-p01", "study-p005")], h[("cohere", "study-t06")]) == (1.022, 0.986, 1.648, 0.916, 1.743)
    ic = r["init_class"]
    assert ic["classifies"] == ["study-p005", "study-p01", "study-p03"] and ic["threshold"] == 0.9
    step0 = ic["step0"]
    assert set(step0) == {"study-t06", "study-t03", "study-p03", "study-p01", "study-p005"}
    assert all(v is not None for s in step0.values() for v in s.values()), "the gate's step-0 numbers are recorded"
    assert (step0["study-t06"]["kl"], step0["study-t06"]["cer_vs_teacher"]) == (4.2068, 1.3226)
    for run in ("study-t06", "study-t03"):  # over the line, kept by the owner's decision
        assert step0[run]["cer_vs_teacher"] >= 0.9 and r["runs"][run]["init_class"] == "pruned_kept"
    for run in ic["classifies"]:  # the rule decides the Parakeet students' class
        lost = step0[run]["cer_vs_teacher"] >= 0.9
        assert r["runs"][run]["init_class"] == ("pruned_lost" if lost else "pruned_kept")
        assert step0[run]["function"] == ("lost" if lost else "kept")
        assert (ic["step0_before_rebuild"][run] >= 0.9) == lost  # the rebuild on the first run's ids kept the classes
    per = r["training"]["per_class"]
    assert "Parakeet" in per["pruned_lost"]["definition"] and "owner" in per["pruned_kept"]["definition"]


def test_study_stats_counts_are_the_registered_ones():
    """kitsune.study_stats computes g per halving from its own count table: it must hold the registered counts (the
    T-0.3B / bridge change of 2026-09-26 included), and the builders' tables must too."""
    from kitsune import student as S
    from kitsune import study_stats as ss

    for run, spec in prereg.RUNS.items():
        assert (ss.PARAMS_TOTAL[run], ss.PARAMS_NON_EMBEDDING[run]) == (spec["params_total"],
                                                                       spec["params_non_embedding"]), run
    assert {k: ss.PARAMS_TOTAL[k] for k in prereg.TEACHER_PARAMS} == prereg.TEACHER_PARAMS
    assert S.PRUNED_EXPECTED_PARAMS[(8, 2560, (0, 2, 5, 7))] == prereg.RUNS["study-t03"]["params_total"]
    assert S.SCRATCH_EXPECTED_PARAMS == {"t01": prereg.RUNS["study-t01"]["params_total"],
                                         "t005": prereg.RUNS["study-t005"]["params_total"],
                                         "bridge": prereg.RUNS["study-bridge"]["params_total"]}


def test_the_wave1_facts_stand_next_to_decisions_15_and_22():
    d = prereg.rules()["decisions"]
    assert "11.6 %" in d["15"]["wave1_facts"] and "400 / 400" in d["15"]["wave1_facts"]
    assert all(x in d["22"]["wave1_facts"] for x in ("35,327 / 35,327", "0 CTC-infeasible", "3.1 %", "16.13",
                                                      "16.20"))
    assert {k for k, v in d.items() if "wave1_facts" in v} == {"15", "22"}
    assert "decoded audio" in prereg.rules()["selection"]["frame_preflight"]
