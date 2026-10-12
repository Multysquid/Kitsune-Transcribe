"""What is left of the size study's pre-registration, CPU only.

The frozen files: study/PREREG.json (the pre-registration, committed before the first study box: its manifest block and
the teachers' baselines are read by scripts/05_evaluate.py, tools/full_report.py and kitsune.evaluate), study/data.json
(the base layer of every full config, tools/make_study_configs.py) and configs/study/study-t06.json (box smoke-b's
stores-eval config) are pinned by sha256: they are never edited or regenerated (their generators were removed after
the study). kitsune/prereg.py keeps the student registry and the study-selection constants the full pipeline reads;
they must stay the ones PREREG.json froze. student_problems (fullrun.student_checks), cap_rule
(scripts/make_selection.py) and the parameter counts the builders and kitsune.study_stats hold are tested here too."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT  # noqa: E402

from kitsune import fullrun, prereg  # noqa: E402

# sha256 of each frozen file's bytes with LF line endings (git's autocrlf gives a Windows checkout CRLF). PREREG.json's
# is also its rules_sha256, the canonical form the study boxes logged and wrote into their numbers files
FROZEN_SHA256 = {
    "study/PREREG.json": "5d2f84dcf526f470065bd91779cdb967874ce9860b0ab960d5dea478f55dea7e",
    "study/data.json": "5191621933486032cb22950674c7ab3be1e681988f7f153edcf9d2f8bf300102",
    "configs/study/study-t06.json": "4600d04615e6279c2b7ad57e37222ac6a2ee2595880cf0331d6e43386c6fb587",
}
CONTRACT_RUNS = {"study-t06": ("aed", "pruned_kept"), "study-t03": ("aed", "pruned_kept"),
                 "study-bridge": ("aed", "scratch"), "study-t01": ("aed", "scratch"),
                 "study-t01-s1235": ("aed", "scratch"), "study-t005": ("aed", "scratch"),
                 "study-p03": ("ctc", "pruned_kept"), "study-p01": ("ctc", "pruned_lost"),
                 "study-p005": ("ctc", "pruned_lost")}


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_the_frozen_files_are_pinned():
    """study/PREREG.json, study/data.json and configs/study/study-t06.json are the committed bytes (up to the line
    endings): PREREG.json filled from the uploaded selection (data repo commit 4b0d801: its parquet's and the frozen
    manifest's sha256, kitsune.fullrun.FROZEN_MANIFEST_SHA256, and the cap 53 that study/data.json registers)."""
    for rel, want in FROZEN_SHA256.items():
        got = hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert got == want, f"{rel} is frozen (sha256 {got[:12]}..., pinned {want[:12]}...): never edit it"
    pr, data = load("study/PREREG.json"), load("study/data.json")
    assert pr["prereg_version"] == 2 and pr["manifest"]["status"] == pr["baselines"]["status"] == "filled"
    m = pr["manifest"]
    assert m["selection_sha256"].startswith("5bbabcef") and m["manifest_sha256"] == fullrun.FROZEN_MANIFEST_SHA256
    assert pr["data"]["extent"]["inputs"]["reazon_large"] == data["extent"]["inputs"]["reazon_large"] == 53
    assert data["selection"] == prereg.SELECTION_FILE and data["selection_recipe"]["study"] == prereg.STUDY_SELECTION
    assert data["selection_recipe"] == pr["selection"]["recipe"]  # the registered recipe, verbatim


def test_the_kept_registry_and_constants_are_the_frozen_ones():
    """kitsune/prereg.py's RUNS and selection constants equal what PREREG.json froze, and every run has the family and
    init class of the wave-2 contract."""
    pr = load("study/PREREG.json")
    assert pr["runs"] == json.loads(json.dumps(prereg.RUNS))
    assert {r: (s["family"], s["init_class"]) for r, s in prereg.RUNS.items()} == CONTRACT_RUNS
    assert {s.get("calib_ids_sha256") for s in prereg.RUNS.values()} == {prereg.CALIB_IDS_SHA256, None}
    sel = pr["selection"]
    assert sel["recipe"]["study"] == prereg.STUDY_SELECTION and sel["seed"] == prereg.SELECTION_SEED
    assert sel["reasons"] == ["kept", "truncated", "no_agree", "agree>A", "no_audio", *prereg.STUDY_REASONS]
    assert f"more than {100 * prereg.ONE_ROOT_MAX_FRAC:g} % of its train rows not_in_parakeet" in sel["coverage"]
    assert (pr["manifest"]["selection_file"], pr["manifest"]["manifest_file"]) == (
        prereg.SELECTION_FILE, prereg.study_files(prereg.SELECTION_FILE)[1])
    assert pr["analysis"]["m4_strata"] == list(prereg.M4_SETS)
    assert f">= {prereg.POOL_MIN_HOURS} h" in pr["data"]["reazon_large_cap"]
    assert pr["training"]["ctc_loss"]["w_ctc"] == prereg.W_CTC


def test_study_files_names_the_sidecar_and_the_manifest():
    assert prereg.study_files(prereg.SELECTION_FILE) == ("labels/full/selections/study_1000h.json",
                                                         "labels/full/selections/study_manifest.json")
    assert prereg.study_files("sel.parquet") == ("sel.json", "study_manifest.json")


def pool_by_cap(rule: int = 53, top: int = 53) -> dict:
    """pool_hours_if_capped.reazon_large as the sidecar holds it: N -> the pool, reaching 1,010 h first at `rule`."""
    return {str(n): prereg.POOL_MIN_HOURS - rule + n for n in range(1, top + 1)}


def test_cap_rule():
    """The smallest N whose pool after all filters holds >= 1,010 h; string keys (JSON) or ints; None if none does."""
    assert prereg.cap_rule(pool_by_cap(53)) == 53 and prereg.cap_rule({1: 1010.0, 2: 1011.0}) == 1
    assert prereg.cap_rule({"9": 1009.99}) is None and prereg.cap_rule({}) is None
    assert prereg.cap_rule({"10": 1012.0, "9": 1010.0, "8": 1009.0}) == 9


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


def test_the_parameter_counts_are_the_registered_ones():
    """kitsune.study_stats.PARAMS_TOTAL (the full report's study rows) holds the registered counts, and so does the
    Transcribe builder's table of pruned shapes (the T-0.3B / bridge change of 2026-09-26 included); the teachers'
    are the ones PREREG.json's size ladder froze."""
    from kitsune import student as S
    from kitsune import study_stats as ss

    for run, spec in prereg.RUNS.items():
        assert ss.PARAMS_TOTAL[run] == spec["params_total"], run
    assert S.PRUNED_EXPECTED_PARAMS[(20, 2560, (0, 2, 5, 7))] == prereg.RUNS["study-t06"]["params_total"]
    assert S.PRUNED_EXPECTED_PARAMS[(8, 2560, (0, 2, 5, 7))] == prereg.RUNS["study-t03"]["params_total"]
    ladder = {x: p for step in load("study/PREREG.json")["size_ladder"]
              for x, p in zip((step["big"], step["small"]), step["params"])}
    assert {k: ss.PARAMS_TOTAL[k] for k in ("cohere", "parakeet-ctc")} == {k: ladder[k] for k in ("cohere",
                                                                                                    "parakeet-ctc")}
    assert ss.PARAMS_TOTAL[ss.ANCHOR] == prereg.RUNS["study-t06"]["params_total"]  # the first run's 0.6B: its shape
