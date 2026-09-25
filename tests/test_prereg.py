"""kitsune/prereg.py: the study's pre-registered rules (study/PREREG.json + .md) and the numbers the study box writes
(max_steps from calibration, the LR edge rule, PREREG_numbers.json). Pure Python, CPU only."""
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


def calib(**over) -> dict:
    """A plausible calibration table: every run of the waves at the modelled step times (STUDY.md 2.4)."""
    t = {"study-t06": 1.079, "study-t03": 0.54, "study-bridge": 0.55, "study-t01": 0.369, "study-t005": 0.296,
         "study-p03": 0.62, "study-p01": 0.318, "study-p005": 0.238}
    out = {run: {"t_step_s": v, "micro_audio_s": prereg.RUNS[run]["micro_audio_s"], "data_wait_frac": 0.01,
                 "steps_measured": 200} for run, v in t.items()}
    for run, c in over.items():
        out[run] = dict(out[run], **c)
    return out


def chosen_probes() -> dict:
    """Probe results whose winners all lie inside their grids after the edge rule (the 2-point grids extended once)."""
    return {"scratch": {5e-4: 3.2, 1e-3: 3.0, 2e-3: 3.1},
            "bridge": {4e-4: 3.3, 1e-3: 3.2, 2e-3: 3.4},  # 1e-3 won the grid's top edge, 2e-3 was its extension
            "lost": {1e-4: 2.9, 3e-4: 2.7, 1e-3: 2.8},
            "kept-t03": {5e-5: 1.3, 1e-4: 1.2, 2e-4: 1.25},  # 1e-4 won the bottom edge, 5e-5 was its extension
            "kept-p03": {1e-4: 1.1, 2e-4: 1.0, 4e-4: 1.05}}


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
    assert set(r["lr_probes"]["classes"]) == {"scratch", "bridge", "lost", "kept-t03", "kept-p03"}
    for p in r["lr_probes"]["classes"].values():
        assert p["warmup"] + p["stable"] + p["cooldown"] == p["max_steps"]
        assert p["cooldown"] == round(0.2 * p["max_steps"]) and p["warmup"] < p["max_steps"] - p["cooldown"]
    assert r["lr_probes"]["classes"]["kept-t03"]["runs"] == ["study-t03", "study-t06"]
    assert r["calibration"]["reference_run"] == "study-t06" and r["calibration"]["reference_steps"] == 9366
    assert len(r["decisions"]) == 30 and r["decisions"]["1"]["option"] == "a" and r["limit_rule"]["delta"] == 0.10
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
    """max_steps_i = round(9366 x t_ref / t_i), half up; the reference run gets 9,366; the replicate takes
    study-t01's; a missing reference or a non-positive step time refuses."""
    c = calib()
    m = prereg.max_steps(c)
    assert m["study-t06"] == 9366 and set(m) == set(prereg.RUNS)
    assert m["study-t01"] == math.floor(9366 * 1.079 / 0.369 + 0.5) == 27387 and m["study-t01-s1235"] == 27387
    # the replicate always takes study-t01's: its own step time (were it measured) is ignored, and refused as input
    with_rep = dict(c, **{"study-t01-s1235": dict(c["study-t01"], t_step_s=0.2)})
    assert prereg.max_steps(with_rep) == m
    assert prereg.calibration_problems(with_rep) == [
        "study-t01-s1235: not calibrated by the rules (it takes study-t01's max_steps)"]
    assert prereg.max_steps({"study-t06": {"t_step_s": 1.0}, "study-t03": {"t_step_s": 0.5}}) == {
        "study-t06": 9366, "study-t03": 18732}
    assert prereg.max_steps({"a": {"t_step_s": 2.0}, "b": {"t_step_s": 4.0}}, t_ref_run="a", ref_steps=5) == {
        "a": 5, "b": 3}  # 2.5 rounds half up
    with pytest.raises(ValueError, match="reference run"):
        prereg.max_steps({"study-t03": {"t_step_s": 1.0}})
    with pytest.raises(ValueError, match="positive"):
        prereg.max_steps(calib(**{"study-t03": {"t_step_s": 0.0}}))


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


def test_choose_lr_edge_rule():
    """Inside the grid: chosen. On an edge of the registered grid: extend by one point (x2 at the top, /2 at the
    bottom). After that extension: chosen if the winner is inside, halt if it is at an edge again. A 2-point grid's
    winner is always on an edge, so it always extends first."""
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
    # 2-point grids: always an edge first
    for cls in ("bridge", "kept-t03", "kept-p03"):
        g = prereg.PROBES[cls]["grid"]
        for w in g:
            res = {lr: (1.0 if lr == w else 2.0) for lr in g}
            d = prereg.choose_lr({cls: res})[cls]
            assert d["decision"] == "extend" and d["next_lr"] == pytest.approx(w * 2 if w == max(g) else w / 2)
    assert prereg.choose_lr({"bridge": {4e-4: 3.3, 1e-3: 3.2, 2e-3: 3.4}})["bridge"]["lr"] == 1e-3
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
    assert set(n) == {"calibration", "max_steps", "lr_probes", "lr", "rules_sha256", "written_utc", "host"}
    assert n["max_steps"] == steps and n["lr"] == lrs and n["host"] == "box-1"
    assert n["rules_sha256"] == prereg.rules_sha256(rules_path)
    assert n["lr_probes"]["kept-t03"] == {"5e-5": 1.3, "1e-4": 1.2, "2e-4": 1.25}
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
