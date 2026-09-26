"""The size study's selection (scripts/make_selection.py with a selection_recipe.study block) on a synthetic label root
with both teachers (tests/fixtures_study.py), and the checks that read it before money is spent:
kitsune.extent.pull_plan (parakeet_out per family / pull_parakeet) and vast/launch.py selection_problems and
data_problems.

The corpus plants one kind of bad row per rule and a toy extent record spreads galgame over 3 inputs, so every drop
reason, the extent filter and the reazon_large cap readout are exercised; the expectations come from the ground truth
the fixture WROTE, not from the code under test. CPU only, no network.
"""
import copy
import importlib
import json
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, load_script  # noqa: E402
from fixtures_study import LONG_REF, make_study_corpus  # noqa: E402

import kitsune.extent as kextent  # noqa: E402
from kitsune import parakeet_targets as pt  # noqa: E402
from kitsune import prereg  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402

ms = load_script("make_selection")
sys.path.insert(0, str(ROOT / "vast"))
launch = importlib.import_module("launch")

BUDGET_S = 120  # the toy draw: about half of the fixture's ~216 s pool
STUDY = {"f1a_max": 0.5, "dedup_min_chars": 15, "draw_audio_s": BUDGET_S, "probe_n": 4, "neutral_max_cer": 0.5}
SOURCES = ["reazon_small", "emilia_yodas", "galgame"]
EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "galgame"]
F0_REASONS = {"truncated", "no_agree", "agree>0.5", "agree>0.2"}


def study_cfg(inputs=None, **study) -> dict:
    return {"sources": list(SOURCES), "eval_sets": list(EVAL_SETS),
            "teacher_root": "labels/t/teacher_out", "second_root": "labels/t/second_out",
            "parakeet_root": "labels/t/parakeet_out", "selection": "labels/t/selections/study.parquet",
            "extent": {"name": "t", "root": "labels/t", "inputs": dict(inputs or {"emilia_yodas": "300h", "galgame": 3})},
            "selection_recipe": {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2"], "filter_eval_sets": [],
                                 "partial_second_opinion": [], "study": dict(STUDY, **study)}}


def _stem(stem: str, step: str) -> dict:
    return dict(stem=stem, split=stem.split("-")[0], step=step, rows=8, hours=0.01, ids_sha256="0" * 64,
                shard_bytes=1)


def toy_record(st) -> dict:
    """The fixture's stems as a label box extent record: galgame over 3 inputs (its hold-out in input 0), emilia_yodas
    all in the 300 h step, everything else one input."""
    def stems(source: str) -> list[str]:
        return sorted(p.stem for p in (st.teacher_out / source).glob("*.npz"))

    gal = stems("galgame")
    gal_inputs = [[s for s in gal if s.startswith("eval-")] + ["train-00000", "train-00001"],
                  ["train-00002", "train-00003"], ["train-00004"]]
    one = {s: {"inputs": [{"input": f"{s}0", "ordinal": 0, "stems": [_stem(x, s) for x in stems(s)]}]}
           for s in ("reazon_small", "eval_jsut", "eval_cv8", "eval_reazon")}
    return {"schema": kextent.RECORD_SCHEMA, "name": "t", "root": "labels/t",
            "canonical_version": kextent.CANONICAL_VERSION,
            "names": ["reazon_small", "emilia_yodas", "galgame", "eval_jsut", "eval_cv8", "eval_reazon"],
            "inputs": {}, "sources": {
                **one,
                "emilia_yodas": {"inputs": [{"input": "e0", "ordinal": 0,
                                             "stems": [_stem(x, "emilia_yodas@300h") for x in stems("emilia_yodas")]}]},
                "galgame": {"inputs": [{"input": f"g{o}", "ordinal": o, "stems": [_stem(x, "galgame") for x in s]}
                                       for o, s in enumerate(gal_inputs)]}}}


def roots(st) -> list[str]:
    return ["--teacher-out", str(st.teacher_out), "--second-out", str(st.second_out), "--data", str(st.data),
            "--parakeet-out", str(st.parakeet_out), "--kotoba-galgame", str(st.kotoba)]


class SimpleOut:
    """A study selection as written: the parquet, its sidecar and manifest, and its recorded arguments."""

    def __init__(self, out: Path):
        self.parquet, self.sidecar_path = out, out.with_suffix(".json")
        self.manifest_path = out.parent / prereg.MANIFEST_FILE
        self.sel = pd.read_parquet(out)
        self.sidecar = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.args = json.loads(pq.read_schema(out).metadata[b"kitsune_selection"])["args"]

    def reason(self, ids) -> set[str]:
        return set(self.sel.set_index("id").loc[sorted(ids), "reason"])


def build(st, d: Path, cfg: dict | None = None, *extra) -> SimpleOut:
    """make_selection.py's real CLI with a study config, the toy extent record and the fixture's roots."""
    d.mkdir(parents=True, exist_ok=True)
    rec = d / "extent.json"
    kextent.write_record(rec, toy_record(st))
    cfg_p = d / "study_cfg.json"
    cfg_p.write_text(json.dumps(cfg or study_cfg()), encoding="utf-8")
    out = d / "sel" / "study.parquet"
    ms.main(["--config", str(cfg_p), "--out", str(out), "--extent-record", str(rec), "--greedy-n", "3", *roots(st),
             *extra])
    return SimpleOut(out)


@pytest.fixture(scope="module")
def st(tmp_path_factory):
    return make_study_corpus(tmp_path_factory.mktemp("study_corpus"))


@pytest.fixture(scope="module")
def built(st, tmp_path_factory):
    return build(st, tmp_path_factory.mktemp("study_sel"))


# ------------------------------------------------------------------------------------------------ the rules


def test_every_rule_drops_the_rows_planted_for_it(st, built):
    """First match wins, in the pre-registered order; F0 is the existing recipe's (truncated, no_agree, agree>A with
    Emilia's 0.2); the study rules then take the rows the fixture made to fail them, and nothing else."""
    sel = built.sel
    assert set(sel["reason"]) == F0_REASONS | set(prereg.STUDY_REASONS) | {"kept"}, "every reason is exercised"
    assert built.reason(st.missing) == {"not_in_parakeet"}
    assert built.reason(st.disagree) == {"f1a_disagree"}
    assert built.reason(st.dup_ref | st.dup_hyp) == {"eval_dup"}  # the normalised reference, and the Cohere hyp
    assert built.reason(st.infeasible) == {"ctc_infeasible"}
    assert built.reason(st.short_dup) <= {"kept", "not_drawn"}  # a 10-character eval reference is under the bar
    study_rows = sel["reason"].isin(prereg.STUDY_REASONS[:-1])  # every study drop but the draw
    planted = st.missing | st.disagree | st.dup_ref | st.dup_hyp | st.infeasible
    assert set(sel["id"][study_rows]) == planted
    # F0 is exactly the recipe's existing rules: the rows in both roots get build_selection's reasons
    f0 = ms.build_selection(st.teacher_out, st.second_out, st.data, SOURCES, EVAL_SETS, 0.5, 1234, 3, 4, True,
                            {"emilia_yodas": 0.2}, (), (), None)
    both = ~sel["id"].isin(st.missing)
    base = f0.set_index("id").loc[sel["id"][both], "reason"].to_numpy()
    mine = sel["reason"][both].to_numpy()
    assert (mine[base != "kept"] == base[base != "kept"]).all(), "an F0 drop keeps its reason"
    assert set(mine[base == "kept"]) <= set(prereg.STUDY_REASONS) | {"kept"}
    # eval rows are never filtered; the drawn rows are the kept train rows
    ev = sel[sel["split"] == "eval"]
    assert ev["keep"].all() and set(ev["reason"]) == {"kept"}
    assert (sel["keep"] == (sel["reason"] == "kept")).all()
    assert list(sel.columns) == ms.COLUMNS  # the schema of every selection (kitsune.trainset.read_selection)


def test_the_parakeet_only_row_is_counted_not_selected(st, built):
    assert len(st.parakeet_only) == 1 and not built.sel["id"].isin(st.parakeet_only).any()
    assert built.sidecar["details"]["parakeet_only_rows"] == 1


def test_ctc_frames_needed_equals_the_greedy_path(st):
    """U + adjacent repeats, vectorised over a shard, equals ctc_greedy(ctc_col0(...)) per utterance and the ground
    truth the fixture wrote; a hand-made path: [a a _ a b b _ b] -> a a b b: U 4 + 2 repeats."""
    for p in sorted(st.parakeet_out.glob("*/*.npz")):
        sh = pt.load_shard(p)
        z = sh.z
        need = ms.ctc_frames_needed(z["frame_offsets"], z["dense_offsets"], z["ctc_dense_frame"],
                                    z["ctc_topk_idx"][:, 0])
        for i, uid in enumerate(sh.ids):
            fo, do = sh._span("frame_offsets", i), sh._span("dense_offsets", i)
            toks = pt.ctc_greedy(pt.ctc_col0(fo.stop - fo.start, z["ctc_dense_frame"][do], z["ctc_topk_idx"][do]))
            assert need[i] == len(toks) + sum(a == b for a, b in zip(toks, toks[1:])) == st.ctc_need[uid]
    b = pt.BLANK
    path = np.array([5, 5, b, 5, 7, 7, b, 7, 9])  # two utterances: 8 frames, then 1
    dense = np.flatnonzero(path != b)
    utt_local = np.where(dense < 8, dense, dense - 8)
    need = ms.ctc_frames_needed([0, 8, 9], [0, int((dense < 8).sum()), len(dense)], utt_local, path[dense])
    assert need.tolist() == [6, 1]  # the second utterance's 9 is not a repeat of the first's last 7
    with pytest.raises(ValueError, match="outside"):
        ms.ctc_frames_needed([0, 2], [0, 1], [2], [5])


def test_hour_totals_follow_the_rules(built):
    """Per source and in total: teacher_out >= candidates >= after_f0 >= after_f1a >= after_dedup >= after_ctc >=
    drawn, each equal to the parquet's rows that survived that far."""
    sel, hours = built.sel, built.sidecar["hours"]
    order = ["teacher_out", "candidates", "after_f0", "after_f1a", "after_dedup", "after_ctc", "drawn"]
    gone = {"candidates": {"not_in_parakeet"}, "after_f0": F0_REASONS, "after_f1a": {"f1a_disagree"},
            "after_dedup": {"eval_dup"}, "after_ctc": {"ctc_infeasible"}, "drawn": {"not_drawn"}}
    train = sel[sel["split"] == "train"]
    for src in [*SOURCES, "total"]:
        rows = train if src == "total" else train[train["source"] == src]
        assert list(hours[src]) == order
        dropped: set = set()
        for k in order:
            dropped |= gone.get(k, set())
            left = rows[~rows["reason"].isin(dropped)]
            assert hours[src][k]["utts"] == len(left), (src, k)
            assert hours[src][k]["hours"] == pytest.approx(left["duration"].astype(float).sum() / 3600)
        assert [hours[src][k]["utts"] for k in order] == sorted((hours[src][k]["utts"] for k in order), reverse=True)
    assert hours["total"]["drawn"]["hours"] * 3600 == pytest.approx(built.sidecar["draw"]["drawn_s"])


def test_the_draw_meets_its_budget(st, built):
    """The drawn rows sum to at most the budget and fall short of it by less than one pool row; the pool is every
    train row that passed rules 1-5; train_audio_s-free (keep = drawn)."""
    d, sel = built.sidecar["draw"], built.sel
    pool = sel[sel["reason"].isin(["kept", "not_drawn"]) & (sel["split"] == "train")]
    assert d["pool_s"] == pytest.approx(pool["duration"].astype(float).sum()) and d["pool_s"] > BUDGET_S
    assert d["drawn_s"] <= BUDGET_S and BUDGET_S - d["drawn_s"] < pool["duration"].max()
    assert d["drawn_utts"] == int((sel["keep"] & (sel["split"] == "train")).sum()) == built.sidecar["n"]["train"]
    # draw_budget: a pure function of (seed, the id set, durations); pooled over the sources
    ids = pool["id"].tolist()
    dur = pool["duration"].to_numpy()
    a = ms.draw_budget(ids, dur, BUDGET_S, 1234)
    rev = ms.draw_budget(ids[::-1], dur[::-1], BUDGET_S, 1234)
    assert a == rev == set(sel["id"][sel["keep"] & (sel["split"] == "train")])
    assert ms.draw_budget(ids, dur, BUDGET_S, 1235) != a
    assert len({sel.set_index("id").loc[i, "source"] for i in a}) == len(SOURCES)
    with pytest.raises(ValueError, match="raise the extent's cap"):  # a pool of exactly the budget is too small too
        ms.draw_budget(ids, dur, float(dur.astype(np.float64).sum()), 1234)


def test_probe_and_greedy_subsets(built):
    """probe_n kept (= drawn) rows per train source, the existing convention; greedy subsets per eval set; both
    hashed in the sidecar."""
    sel = built.sel
    probe = sel[sel["in_probe"]]
    assert probe["keep"].all() and (probe["split"] == "train").all()
    assert probe.groupby("source").size().to_dict() == {s: STUDY["probe_n"] for s in SOURCES}
    assert built.sidecar["n"]["probe"] == len(probe) and built.sidecar["ids_sha256"]["probe"] == ids_sha256(
        probe["id"].tolist())
    greedy = sel[sel["in_greedy_subset"]]
    assert greedy["keep"].all() and greedy.groupby("source").size().to_dict() == {s: 3 for s in EVAL_SETS}
    train_ids = sel["id"][sel["keep"] & (sel["split"] == "train")].tolist()
    assert built.sidecar["ids_sha256"]["train"] == ids_sha256(train_ids)


def test_manifest_views_and_baselines(st, built):
    """The manifest: per eval set its ordered ids (selection order) and their sha256; the Galgame views (neutral from
    the laptop's kotoba file, all, label_box = the label box's filter). The sidecar repeats the hashes and holds the
    teachers' baselines on those rows."""
    sel, man, side = built.sel, built.manifest, built.sidecar
    for s in EVAL_SETS:
        ids = sel["id"][(sel["source"] == s) & (sel["split"] == "eval") & sel["keep"]].tolist()
        assert man["sets"][s]["ids"] == ids and man["sets"][s]["n"] == len(ids) > 0
        assert man["sets"][s]["ids_sha256"] == ids_sha256(ids) == side["ids_sha256"]["eval"][s]
    gal = sel[(sel["source"] == "galgame") & (sel["split"] == "eval")]
    views = man["galgame_views"]
    assert views["all"]["ids"] == gal["id"].tolist()
    assert views["neutral"]["ids"] == [i for i in gal["id"] if i in st.neutral] and len(st.neutral) > 0
    assert not set(views["neutral"]["ids"]) & st.empty_ref  # a cer-0 row without a reference is not neutral
    box = gal[~gal["truncated"] & gal["agree"].notna() & (gal["agree"] <= 0.5)]
    assert views["label_box"]["ids"] == box["id"].tolist() and 0 < len(box) < len(gal)
    for v, b in views.items():
        assert b["ids_sha256"] == ids_sha256(b["ids"]) == side["galgame_views"][v]["ids_sha256"]
    assert side["galgame_views"]["all"]["n_with_ref"] == len(gal) - 1
    assert man["selection"] == "labels/t/selections/study.parquet"
    base = side["baselines"]
    # the strata as kitsune.study_stats names them (Galgame as its views), the systems as CONTRACT.md 5 names them
    keys = {"eval_jsut", "eval_cv8", "eval_reazon", "galgame_neutral", "galgame_all", "galgame_label_box", "m4"}
    assert set(base) == {"cohere", "parakeet-ctc", "parakeet-tdt"} and all(set(b) == keys for b in base.values())
    m4 = np.mean([base["cohere"][k]["cer"] for k in prereg.M4_SETS])
    assert base["cohere"]["m4"] == pytest.approx(m4)
    # the fixture's Parakeet hyp is Cohere's on every eval row: identical corpus CER
    assert base["parakeet-tdt"]["eval_jsut"] == base["cohere"]["eval_jsut"]
    assert side["selection"] == {"path": "labels/t/selections/study.parquet", "sha256": ms.file_sha256(built.parquet)}
    assert side["manifest"] == {"path": "labels/t/selections/study_manifest.json",
                                "sha256": ms.file_sha256(built.manifest_path)}


def registered(side: dict) -> dict:
    """The toy sidecar dressed as the pre-registered selection (its recipe, seed, sources, eval sets, file names and
    extent; eval_emilia and the two unlabelled sources borrowed from existing entries): what the fill reads is the
    toy build's own ids, views and baselines."""
    sc = copy.deepcopy(side)
    sc.update(sources=list(prereg.STUDY_SOURCES), eval_sets=list(prereg.STUDY_EVAL_SETS),
              recipe=prereg.registered_recipe(), seed=prereg.SELECTION_SEED)
    sc["selection"]["path"] = prereg.SELECTION_FILE
    sc["manifest"]["path"] = prereg.study_files(prereg.SELECTION_FILE)[1]
    for k in ("ids_sha256", "n"):
        sc[k]["eval"]["eval_emilia"] = sc[k]["eval"]["eval_cv8"]
    for b in sc["baselines"].values():
        b["eval_emilia"] = b["eval_cv8"]
    sc["n"]["probe"] = prereg.STUDY_SELECTION["probe_n"] * len(prereg.STUDY_SOURCES)
    sc["extent"] = {"name": "full", "inputs": {**{k: v for k, v in prereg.STUDY_INPUTS.items() if v is not None},
                                               "reazon_large": 2}}
    sc["details"]["pool_hours_if_capped"] = {"reazon_large": {"1": 1000.0, "2": 1010.0}}  # the rule: 2
    sc["draw"].update(budget_s=float(prereg.STUDY_SELECTION["draw_audio_s"]), drawn_s=3_599_000.0,
                      pool_s=1010.0 * 3600)
    return sc


def test_the_sidecar_fills_the_prereg_only_when_it_is_the_registered_selection(built):
    """python -m kitsune.prereg --write study/ --sidecar ...: the toy build (its own recipe, sources and extent) is
    refused, and the same sidecar with the registered recipe, seed, sources and extent fills every pending field from
    the build's own hashes, views and baselines."""
    side = built.sidecar
    with pytest.raises(ValueError, match="not the pre-registered study selection"):
        prereg.rules(side)
    problems = prereg.sidecar_problems(side)
    assert any(p.startswith("sources") for p in problems) and any(p.startswith("recipe") for p in problems)
    sc = registered(side)
    assert prereg.sidecar_problems(sc) == []
    r = prereg.rules(sc)
    assert prereg.pending(r) == []
    m = r["manifest"]
    assert m["train"] == {"ids_sha256": side["ids_sha256"]["train"], "n": side["n"]["train"],
                          "hours": pytest.approx(3_599_000 / 3600)}
    assert m["sets"]["galgame"]["ids_sha256"] == built.manifest["sets"]["galgame"]["ids_sha256"]
    assert m["galgame_views"]["neutral"] == {"ids_sha256": built.manifest["galgame_views"]["neutral"]["ids_sha256"],
                                             "n": len(built.manifest["galgame_views"]["neutral"]["ids"])}
    assert m["manifest_sha256"] == ms.file_sha256(built.manifest_path)
    assert r["baselines"]["cohere"]["m4"] == pytest.approx(side["baselines"]["cohere"]["m4"])
    assert r["baselines"]["parakeet-ctc"]["galgame_neutral"] == side["baselines"]["parakeet-ctc"]["galgame_neutral"][
        "cer"]
    assert r["data"]["extent"]["inputs"]["reazon_large"] == 2


def test_corpus_cer_is_the_evaluators():
    """make_selection's torch-free corpus_cer equals kitsune.evaluate.corpus_cer (the scorer's)."""
    from kitsune.evaluate import corpus_cer

    hyps = ["今日は、いい天気", "", "abc", "テスト", None, "x"]
    refs = ["今日はいい天気です", "空", "ABC!", "", "何か", None]
    assert ms.corpus_cer(hyps, refs) == corpus_cer(hyps, refs)


def test_a_rebuild_gives_the_same_bytes(st, built, tmp_path):
    """No timestamp and no machine path in any of the three files: the same labels and config give the same sha256
    (the hashes pre-register), rebuilt in place or into another directory from another config path (another checkout
    or worktree)."""
    files = ("parquet", "sidecar_path", "manifest_path")
    before = [ms.file_sha256(getattr(built, f)) for f in files]
    again = build(st, built.parquet.parent.parent)
    assert [ms.file_sha256(getattr(again, f)) for f in files] == before
    other = build(st, tmp_path / "elsewhere" / "deeper")
    assert [ms.file_sha256(getattr(other, f)) for f in files] == before
    pd.testing.assert_frame_equal(other.sel, built.sel)
    # what the parquet records instead of paths: the config's own roots and the inputs' content hashes
    def strings(obj):
        if isinstance(obj, dict):
            return [x for v in obj.values() for x in strings(v)]
        return [x for v in obj for x in strings(v)] if isinstance(obj, list) else [obj] if isinstance(obj, str) else []

    assert not [v for v in strings(built.args) if Path(v).is_absolute()]
    assert built.args["config_roots"]["teacher_root"] == "labels/t/teacher_out"
    assert built.args["kotoba_galgame_sha256"] == ms.file_sha256(st.kotoba) == built.sidecar["kotoba"]["sha256"]
    record = kextent.load_record(built.parquet.parent.parent / "extent.json")
    assert built.args["extent_record_sha256"] == built.sidecar["extent_record_sha256"] == ms._json_sha256(record)
    assert set(built.args) == {*ms.STUDY_ARGS, "config_roots", "extent_record_sha256", "kotoba_galgame_sha256"}


def test_the_recipe_is_recorded_and_the_extent_is_followed(st, built, tmp_path):
    """The selection records the study block (launch compares it with the run config's), the extent and the reazon
    cap readout; a smaller galgame cap drops the rows of the inputs it leaves out."""
    assert built.args["study"] == STUDY and built.args["extent"] == {"name": "t", "inputs": {"emilia_yodas": "300h",
                                                                                              "galgame": 3}}
    assert built.args["probe_n"] == STUDY["probe_n"]
    assert built.sidecar["recipe"]["study"] == STUDY and built.sidecar["kotoba"]["sha256"] == ms.file_sha256(st.kotoba)
    caps = built.sidecar["details"]["pool_hours_if_capped"]
    assert set(caps) == {"galgame"} and list(caps["galgame"]) == ["1", "2", "3"]
    assert "reazon_large_cap" not in built.sidecar["details"]  # the rule's readout is reazon_large's only
    assert caps["galgame"]["3"] == pytest.approx(built.sidecar["draw"]["pool_s"] / 3600)
    assert caps["galgame"]["1"] <= caps["galgame"]["2"] <= caps["galgame"]["3"]
    small = build(st, tmp_path, study_cfg({"emilia_yodas": "300h", "galgame": 2}, draw_audio_s=60))
    assert not small.sel["teacher_file"].eq("galgame/train-00004").any()
    assert built.sel["teacher_file"].eq("galgame/train-00004").any()
    assert small.sidecar["draw"]["pool_s"] / 3600 == pytest.approx(caps["galgame"]["2"])


def test_the_cap_readout():
    """make_selection's readout of the reazon_large cap rule (kitsune.prereg.cap_rule) and its report line."""
    by_cap = {"reazon_large": {"1": 1000.0, "2": 1009.0, "3": 1010.5, "4": 1020.0}, "galgame": {"1": 1.0}}
    assert ms.cap_readout({"galgame": {"1": 1.0}}, {"galgame": 1}) is None
    held = ms.cap_readout(by_cap, {"reazon_large": 3, "galgame": 1})
    assert held == {"min_pool_hours": 1010, "configured": 3, "rule": 3, "holds": True}
    assert ms.cap_line(held) == "reazon_large cap: configured 3, the rule (pool >= 1010 h) gives 3"
    over = ms.cap_readout(by_cap, {"reazon_large": 4})
    assert over["rule"] == 3 and not over["holds"] and "set extent.inputs.reazon_large to it" in ms.cap_line(over)
    short = ms.cap_readout({"reazon_large": {"1": 900.0, "2": 950.0}}, {"reazon_large": 2})
    assert short["rule"] is None and "rebuild at a larger cap" in ms.cap_line(short)


def test_refusals(st, tmp_path):
    """What the study path refuses: a label-filtered eval set, a --probe-n next to the recipe's, a derive, a bad
    study block, a teacher shard without its Parakeet files, a pool smaller than the budget, a kotoba file without
    the hold-out's rows."""
    def run(cfg, *extra, d=None):
        return build(st, d or tmp_path / "x", cfg, *extra)

    bad = study_cfg()
    bad["selection_recipe"]["filter_eval_sets"] = ["galgame"]
    with pytest.raises(SystemExit):
        run(bad)
    with pytest.raises(SystemExit):
        run(study_cfg(), "--probe-n", "5")
    with pytest.raises(SystemExit):
        run(study_cfg(), "--from-selection", str(tmp_path / "base.parquet"))
    for study in ({"f1a_max": -1}, {"probe_n": 1.5}, {"dedup_min_chars": 0}):
        assert ms.study_problems(dict(STUDY, **study))
        with pytest.raises(SystemExit):
            run(study_cfg(**study))
    assert ms.study_problems(dict(STUDY, extra=1)) and ms.study_problems({"f1a_max": 0.5})
    assert ms.study_problems(prereg.STUDY_SELECTION) == []
    with pytest.raises(SystemExit, match="raise the extent's cap"):
        run(study_cfg(draw_audio_s=10 ** 7))
    pk = tmp_path / "parakeet_out"
    shutil.copytree(st.parakeet_out, pk)
    (pk / "galgame" / "train-00001.npz").unlink()
    with pytest.raises(SystemExit, match="galgame/train-00001"):
        run(study_cfg(), "--parakeet-out", str(pk))
    kot = tmp_path / "kotoba.jsonl"
    kot.write_text("".join(st.kotoba.read_text(encoding="utf-8").splitlines(keepends=True)[1:]), encoding="utf-8")
    with pytest.raises(SystemExit, match="K10"):
        run(study_cfg(), "--kotoba-galgame", str(kot))


def test_no_audio_is_an_f0_reason_and_an_eval_row_without_parakeet_is_not_in_the_manifest(tmp_path):
    """The audio check applies as it always did (no_audio, after the other F0 reasons); an eval row the Parakeet pass
    did not label is not_in_parakeet and so outside the manifest (launch refuses such a selection)."""
    st = make_study_corpus(tmp_path / "c", seed=5, missing_eval=1)
    victim = next(u for u in st.fc.utts.values() if u.source == "reazon_small" and u.split == "train"
                  and u.has_teacher and not u.truncated and u.agree == 0.0 and u.id not in st.missing)
    shard = st.data / "shards" / "reazon_small" / f"{victim.stem}.parquet"
    t = pq.read_table(shard)
    pq.write_table(t.filter(pa_ne(t, victim.id)), shard)
    out = build(st, tmp_path / "s")
    assert out.reason({victim.id}) == {"no_audio"}
    assert out.reason(st.missing_eval) == {"not_in_parakeet"}
    (miss,) = st.missing_eval
    src = st.fc.utts[miss].source
    assert miss not in out.manifest["sets"][src]["ids"] and len(out.manifest["sets"][src]["ids"]) > 0


def pa_ne(table, uid):
    import pyarrow.compute as pc

    return pc.not_equal(table.column("id"), uid)


# ------------------------------------------------------------------------------------ pull_plan and launch


def listing(st, record: dict, root: str = "labels/t") -> list[str]:
    files = [f"{root}/{r}/meta.json" for r in ("teacher_out", "second_out", "parakeet_out")]
    files += [f"{root}/selections/study.parquet", f"{root}/selections/study.json",
              f"{root}/selections/{prereg.MANIFEST_FILE}", f"{root}/extent.json", f"{root}/COMPLETE.json",
              "students/s/config.json"]
    for src, s in record["sources"].items():
        for inp in s["inputs"]:
            for stem in inp["stems"]:
                files += [f"{root}/{r}/{src}/{stem['stem']}.{e}" for r in ("teacher_out", "parakeet_out")
                          for e in ("npz", "jsonl")]
                files += [f"{root}/second_out/{src}/{stem['stem']}.jsonl"] if src in SOURCES else []
    return files


def plan_cfg(**kw) -> dict:
    cfg = study_cfg({"galgame": 2, "emilia_yodas": "300h"})
    cfg["student"] = "students/s"
    return dict(cfg, **kw)


def pulled(plan, files) -> set[str]:
    return {f for f in files if any(fnmatch(f, pat) for pat in plan["dir_patterns"])} | set(plan["explicit"])


def test_pull_plan_parakeet_per_family(st):
    """aed without pull_parakeet: today's plan, no parakeet_out at all. pull_parakeet (the study box): both teachers
    for every stem. family ctc: parakeet_out for every stem, teacher_out only for the eval sets' eval stems (the Cohere
    baselines), never a train stem's."""
    rec = toy_record(st)
    files = listing(st, rec)
    aed = kextent.pull_plan(plan_cfg(), rec, files)
    assert aed["problems"] == []
    assert not any("parakeet_out" in f for f in aed["dir_patterns"] + aed["explicit"] + aed["required"])
    old = {k: v for k, v in plan_cfg().items() if k != "parakeet_root"}  # a config with no parakeet_root at all
    assert kextent.pull_plan(old, rec, files) == aed
    # a study selection brings its sidecar and manifest (the box checks the manifest, every scorer reads it)
    study_files = ["labels/t/selections/study.json", f"labels/t/selections/{prereg.MANIFEST_FILE}"]
    assert set(study_files) <= pulled(aed, files) and set(study_files) <= set(aed["required"])
    plain = plan_cfg(selection_recipe={k: v for k, v in plan_cfg()["selection_recipe"].items() if k != "study"})
    plan = kextent.pull_plan(plain, rec, files)
    assert plan["problems"] == [] and not set(study_files) & (pulled(plan, files) | set(plan["required"]))
    assert set(aed["required"]) - set(plan["required"]) == set(study_files)
    gone = kextent.pull_plan(plan_cfg(), rec, [f for f in files if f != study_files[1]])["problems"]
    assert len(gone) == 1 and study_files[1] in gone[0]

    both = kextent.pull_plan(plan_cfg(pull_parakeet=True), rec, files)
    assert both["problems"] == []
    got = pulled(both, files)
    assert set(both["required"]) - set(aed["required"]) == {f for f in both["required"] if "/parakeet_out/" in f}
    assert "labels/t/parakeet_out/meta.json" in both["required"]
    for f in aed["required"]:
        assert f in got
    assert "labels/t/parakeet_out/galgame/train-00003.npz" in both["explicit"]  # capped: explicit files
    assert "labels/t/parakeet_out/galgame/train-00004.npz" not in got  # outside the cap
    assert "labels/t/parakeet_out/eval_jsut/*" in both["dir_patterns"]  # uncapped: a directory

    ctc = kextent.pull_plan(plan_cfg(family="ctc"), rec, files)
    assert ctc["problems"] == []
    got = pulled(ctc, files)
    teacher = {f for f in got if "/teacher_out/" in f and not f.endswith("meta.json")}
    assert teacher and all(f.split("/")[-1].startswith("eval-") for f in teacher)
    assert "labels/t/teacher_out/galgame/eval-00000.npz" in teacher and "labels/t/teacher_out/eval_jsut/*" in \
        ctc["dir_patterns"]
    assert not any(p.startswith("labels/t/teacher_out/reazon_small") for p in ctc["dir_patterns"] + ctc["explicit"])
    assert {f for f in got if "/parakeet_out/" in f} == {f for f in pulled(both, files) if "/parakeet_out/" in f}
    assert pulled(kextent.pull_plan(plan_cfg(family="ctc", pull_parakeet=True), rec, files), files) == \
        pulled(both, files)
    # what refuses
    gone = "labels/t/parakeet_out/galgame/train-00001.jsonl"
    p = kextent.pull_plan(plan_cfg(family="ctc"), rec, [f for f in files if f != gone])
    assert len(p["problems"]) == 1 and gone in p["problems"][0]
    assert kextent.pull_plan(plan_cfg(pull_parakeet=True), rec, [f for f in files if f != gone])["problems"]
    assert kextent.pull_plan(plan_cfg(), rec, [f for f in files if f != gone])["problems"] == []
    no_root = {k: v for k, v in plan_cfg(family="ctc").items() if k != "parakeet_root"}
    assert "the config trains a CTC student but has no parakeet_root" in kextent.pull_plan(no_root, rec, files)[
        "problems"]


def test_pull_plan_of_the_full_configs_is_unchanged():
    """configs/full.json and full_sub3k.json have neither family nor pull_parakeet: never parakeet_out."""
    for name in ("full", "full_sub3k"):
        cfg = json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
        assert "family" not in cfg and "pull_parakeet" not in cfg
    cfg = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
    assert cfg["pull_parakeet"] is True and kextent.validate(cfg) == []


def write_sel(path: Path, rows, args=None) -> Path:
    """A selection parquet as make_selection.py writes it (tests/test_infra.py's shape)."""
    import pyarrow as pa

    df = pd.DataFrame(rows, columns=["source", "split", "keep", "reason"])
    df["teacher_file"] = df["source"] + "/" + df["split"] + "-00000"
    t = pa.Table.from_pandas(df, preserve_index=False)
    if args is not None:
        t = t.replace_schema_metadata({b"kitsune_selection": json.dumps(dict(args=args)).encode()})
    pq.write_table(t, path)
    return path


LCFG = {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"], "student": "students/s",
        "selection": "labels/t/selections/study.parquet", "teacher_root": "labels/t/teacher_out",
        "parakeet_root": "labels/t/parakeet_out",
        "selection_recipe": {"agree_max": 0.5, "agree_max_source": [], "filter_eval_sets": [],
                             "partial_second_opinion": [], "study": dict(prereg.STUDY_SELECTION)}}
LARGS = {"sources": LCFG["sources"], "eval_sets": LCFG["eval_sets"], "agree_max": 0.5, "agree_max_source": [],
         "filter_eval_sets": [], "partial_second_opinion": [], "study": dict(prereg.STUDY_SELECTION), "seed": 1234}
LROWS = [("reazon_small", "train", True, "kept"), ("galgame", "train", True, "kept"),
         ("galgame", "train", False, "not_drawn"), ("galgame", "train", False, "f1a_disagree"),
         ("reazon_small", "train", False, "eval_dup"), ("reazon_small", "train", False, "ctc_infeasible"),
         ("galgame", "train", False, "agree>0.5"), ("eval_jsut", "eval", True, "kept"), ("galgame", "eval", True, "kept")]


def test_selection_problems_know_the_study(tmp_path):
    """The study reasons pass with the study recipe and only with it; the recorded study block must equal the
    config's; eval rows without Parakeet labels and a train coverage gap over 0.1 % refuse; with the repo listing the
    sidecar, the manifest and (for a CTC run or pull_parakeet) the kept rows' Parakeet files must be uploaded."""
    name, path = LCFG["selection"], tmp_path / "sel.parquet"
    assert launch.selection_problems(write_sel(path, LROWS, LARGS), name, LCFG) == []
    # the study block: recorded vs configured, both directions
    for args, cfg in ((dict(LARGS, study=dict(STUDY, draw_audio_s=99)), LCFG), (dict(LARGS, study=None), LCFG)):
        problems = launch.selection_problems(write_sel(path, LROWS, args), name, cfg)
        assert any("was built with study" in p for p in problems), problems
    # ... and both must be the pre-registered block, built with the pre-registered seed
    toy_cfg = dict(LCFG, selection_recipe=dict(LCFG["selection_recipe"], study=dict(STUDY)))
    assert launch.selection_problems(write_sel(path, LROWS, dict(LARGS, study=dict(STUDY))), name, toy_cfg) == [
        f"the run config's selection_recipe.study {STUDY} is not the pre-registered {prereg.STUDY_SELECTION} "
        f"(kitsune.prereg; study/data.json carries it)"]
    assert launch.selection_problems(write_sel(path, LROWS, dict(LARGS, seed=7)), name, LCFG) == [
        f"{name} was built with seed 7, the study selection's is pre-registered as 1234: rebuild it with "
        f"scripts/make_selection.py --config <the run config> and upload it"]
    plain_cfg = dict(LCFG, selection_recipe={k: v for k, v in LCFG["selection_recipe"].items() if k != "study"})
    plain_args = {k: v for k, v in LARGS.items() if k != "study"}
    problems = launch.selection_problems(write_sel(path, LROWS, plain_args), name, plain_cfg)
    assert problems == [f"{name} drops rows as ['ctc_infeasible', 'eval_dup', 'f1a_disagree', 'not_drawn'], which this "
                        f"recipe never does: rebuild it with scripts/make_selection.py --config <the run config> and "
                        f"upload it"]
    odd = LROWS + [("galgame", "train", False, "weird")]
    assert any("['weird']" in p for p in launch.selection_problems(write_sel(path, odd, LARGS), name, LCFG))
    # coverage: any eval row, or more than 0.1 % of the train rows, not in parakeet_out
    rows = LROWS + [("eval_jsut", "eval", False, "not_in_parakeet")]
    assert any("eval rows have no Parakeet labels" in p for p in
               launch.selection_problems(write_sel(path, rows, LARGS), name, LCFG))
    rows = LROWS + [("galgame", "train", False, "not_in_parakeet")]
    assert any("train rows are in teacher_out only" in p for p in
               launch.selection_problems(write_sel(path, rows, LARGS), name, LCFG))
    rows = LROWS + [("galgame", "train", False, "not_drawn")] * 1100 + [("galgame", "train", False, "not_in_parakeet")]
    assert launch.selection_problems(write_sel(path, rows, LARGS), name, LCFG) == []  # 1 of 1,108: under 0.1 %
    # the repo listing: sidecar, manifest, teacher files, Parakeet files per family
    have = {"labels/t/selections/study.json", f"labels/t/selections/{prereg.MANIFEST_FILE}"}
    have |= {f"labels/t/teacher_out/{s}/{sp}-00000.{e}" for s, sp in (("reazon_small", "train"), ("galgame", "train"),
                                                                          ("eval_jsut", "eval"), ("galgame", "eval"))
             for e in ("npz", "jsonl")}
    pk = {f.replace("teacher_out", "parakeet_out") for f in have if "teacher_out" in f}
    path = write_sel(path, LROWS, LARGS)
    assert launch.selection_problems(path, name, LCFG, have) == []
    assert launch.selection_problems(path, name, dict(LCFG, pull_parakeet=True), have | pk) == []
    problems = launch.selection_problems(path, name, dict(LCFG, pull_parakeet=True), have)
    assert len(problems) == 1 and "Parakeet targets are not in the data repo: 8 of 8" in problems[0]
    for gone in ("labels/t/selections/study.json", f"labels/t/selections/{prereg.MANIFEST_FILE}"):
        assert launch.selection_problems(path, name, LCFG, have - {gone}) == [
            f"no {gone}: upload the study selection's sidecar and manifest with it"]
    # a CTC run without pull_parakeet needs only the eval rows' teacher files
    ctc_have = {f for f in have | pk if "/train-" not in f or "parakeet_out" in f}
    assert launch.selection_problems(path, name, dict(LCFG, family="ctc"), ctc_have) == []
    assert launch.selection_problems(path, name, LCFG, ctc_have)  # an AED run needs the train rows' too


def test_data_problems_know_the_parakeet_requirement():
    """A non-extent config: a CTC student or pull_parakeet needs parakeet_out's meta and npz for every source and eval
    set; a CTC student needs teacher npz only for the eval sets; configs without either are unchanged."""
    cfg = {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"], "student": "students/s",
           "selection": "selection/s.parquet", "parakeet_root": "parakeet_out"}
    base = ["teacher_out/meta.json", "second_out/meta.json", "selection/s.parquet",  # a CTC student: + its CC-BY card
            *(f"students/s/{n}" for n in launch.STUDENT_FILES + (launch.CTC_CARD,))]
    teacher = [f"teacher_out/{s}/{st}.npz" for s, st in (("reazon_small", "train-00000"), ("galgame", "train-00000"),
                                                          ("galgame", "eval-00000"), ("eval_jsut", "eval-00000"))]
    second = ["second_out/reazon_small/train-00000.jsonl", "second_out/galgame/train-00000.jsonl",
              "second_out/galgame/eval-00000.jsonl"]
    pk = [f.replace("teacher_out", "parakeet_out") for f in teacher] + ["parakeet_out/meta.json"]
    assert launch.data_problems(base + teacher + second, cfg) == []
    assert launch.data_problems(base + teacher + second, dict(cfg, pull_parakeet=True)) == [
        "no parakeet_out/meta.json", "no parakeet_out/reazon_small/*.npz", "no parakeet_out/galgame/*.npz",
        "no parakeet_out/eval_jsut/*.npz"]
    assert launch.data_problems(base + teacher + second + pk, dict(cfg, pull_parakeet=True)) == []
    # a CTC student: no train-only teacher npz needed; the eval sets' still are
    no_train = [f for f in teacher if "reazon_small" not in f]
    assert launch.data_problems(base + no_train + second + pk, dict(cfg, family="ctc")) == []
    assert launch.data_problems(base + no_train + second + pk, cfg) == ["no teacher_out/reazon_small/*.npz"]
    no_jsut = [f for f in teacher if "eval_jsut" not in f]
    assert launch.data_problems(base + no_jsut + second + pk, dict(cfg, family="ctc")) == [
        "no teacher_out/eval_jsut/*.npz"]
    no_root = {k: v for k, v in cfg.items() if k != "parakeet_root"}
    assert launch.data_problems(base + teacher + second, dict(no_root, family="ctc")) == [
        "the config needs Parakeet targets (family ctc or pull_parakeet) but has no parakeet_root"]


def test_a_real_study_selection_passes_launch(st, built):
    """The selection the study path wrote satisfies selection_problems with its config (the recorded arguments and seed
    included), except the toy recipe, which is not the registered one, and the K5 coverage gap the fixture plants on
    purpose (2 train rows missing from parakeet_out in ~112)."""
    cfg = study_cfg()
    problems = launch.selection_problems(built.parquet, cfg["selection"], cfg)
    assert [p for p in problems if "no_agree" not in p] == [
        f"the run config's selection_recipe.study {STUDY} is not the pre-registered {prereg.STUDY_SELECTION} "
        f"(kitsune.prereg; study/data.json carries it)",  # the toy recipe (a 120 s draw) is not the study's
        f"{cfg['selection']}: 2 of {int((built.sel['split'] == 'train').sum())} train rows are in teacher_out only, "
        f"more than 0.1 % (K5): the Parakeet pass is incomplete"]
    assert LONG_REF  # the dedup reference the fixture planted is >= 15 characters
    assert len(LONG_REF) >= STUDY["dedup_min_chars"]
