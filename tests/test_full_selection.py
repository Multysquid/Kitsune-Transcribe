"""The full-data runs' selections (scripts/make_selection.py with a selection_recipe.full_study block) on a synthetic
label root with both teachers (tests/fixtures_study.py), and vast/launch.py selection_problems on them.

The corpus is the study fixture with Emilia-style ids (JA_<video>_W<n>, three rows a video, so videos span the 8-row
shards) and, on top of its planted rows (one kind per study rule), texts planted for dev_dup: two rows of every
reazon_small train shard share a long reference and one more shares a long Parakeet ctc_hyp, so whichever shards the
dev draw picks, the copies inside the pair stay dev, the ones in its buffers are dev_buffer and the rest are dev_dup.
The "frozen" manifest is a study build of the same corpus (its eval rows are what the full build must reproduce),
passed with its own sha (--manifest-sha256). Two limits are patched for the toy sizes: the Emilia dev target
(devslice.DEV_VIDEO_ROWS, 4096 rows on the real data) and K5 (prereg.ONE_ROOT_MAX_FRAC: the fixture's 2 rows missing
from parakeet_out are more than 0.1 % of its ~160 train rows; the K5 test runs with the real limit). CPU only.
"""
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, load_script  # noqa: E402
from fixtures_study import _rewrite_jsonl, make_study_corpus  # noqa: E402

import kitsune.extent as kextent  # noqa: E402
from kitsune import devslice, fullrun, prereg  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402

ms = load_script("make_selection")
sys.path.insert(0, str(ROOT / "vast"))
launch = importlib.import_module("launch")

SOURCES = ["reazon_small", "emilia_yodas", "galgame"]
EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "galgame"]
CORPUS = {"reazon_small": (64, "train"), "emilia_yodas": (48, "train"), "galgame": [(48, "train"), (16, "eval")],
          "eval_jsut": (12, "eval"), "eval_cv8": (8, "eval"), "eval_reazon": (8, "eval")}
VIDEO_ROWS = 7  # the patched Emilia dev target: 3 videos of 3 rows
MAX_FRAC = 0.05  # the patched K5 limit
SMOKE_BUDGET_S = 60.0  # about half the toy pool left after the dev rules
DEV_TEXT = "まみむめもやゆよらりるれろわをん"  # 16 characters: over the 15-character bar
CTC_TEXT = "がぎぐげござじずぜぞだぢづでどば"
F0 = {"truncated", "no_agree", "agree>0.5", "agree>0.2", "no_audio"}
STUDY_DROPS = {"not_in_parakeet", "f1a_disagree", "eval_dup", "ctc_infeasible"}


def emilia_ids(source: str, split: str, i: int):
    return f"{source}/JA_v{i // 3:02d}_x_W{i % 3:06d}" if source == "emilia_yodas" else None


def full_cfg(*, draw=None, selection=None, **recipe) -> dict:
    return {"data_root": "data", "teacher_root": "labels/full/teacher_out", "second_root": "labels/full/second_out",
            "parakeet_root": "labels/full/parakeet_out",
            "selection": selection or (fullrun.FULL_SELECTION if draw is None else fullrun.SMOKE_SELECTION),
            "extent": {"name": "full", "root": "labels/full", "inputs": {"emilia_yodas": "300h"}},
            "sources": list(SOURCES), "eval_sets": list(EVAL_SETS),
            "selection_recipe": dict({"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2"],
                                      "filter_eval_sets": [], "partial_second_opinion": [], "study": None,
                                      "full_study": dict(fullrun.FULL_STUDY, draw_audio_s=draw)}, **recipe)}


def _stem(stem: str, step: str) -> dict:
    return dict(stem=stem, split=stem.split("-")[0], step=step, rows=8, hours=0.01, ids_sha256="0" * 64,
                shard_bytes=1)


def toy_record(root: Path) -> dict:
    """The corpus's stems as the label box's extent record of labels/full: emilia_yodas in the 300 h step, every
    other name one input."""
    def stems(source: str) -> list[str]:
        return sorted(p.stem for p in (root / "teacher_out" / source).glob("*.npz"))

    sources = {s: {"inputs": [{"input": f"{s}0", "ordinal": 0, "stems": [_stem(x, s) for x in stems(s)]}]}
               for s in ("reazon_small", "galgame", "eval_jsut", "eval_cv8", "eval_reazon")}
    sources["emilia_yodas"] = {"inputs": [{"input": "e0", "ordinal": 0,
                                           "stems": [_stem(x, "emilia_yodas@300h") for x in stems("emilia_yodas")]}]}
    return {"schema": kextent.RECORD_SCHEMA, "name": "full", "root": "labels/full",
            "canonical_version": kextent.CANONICAL_VERSION, "names": [*SOURCES, "eval_jsut", "eval_cv8", "eval_reazon"],
            "inputs": {"emilia_yodas": "300h"}, "sources": sources}


def seal(root: Path):
    """The label root's extent.json and COMPLETE.json, as the label box leaves them."""
    kextent.write_record(root / kextent.RECORD_FILE, toy_record(root))
    (root / "COMPLETE.json").write_text(json.dumps({"schema": 1, "name": "full", "files_digest": "d" * 64}),
                                        encoding="utf-8")


class Planted:
    def __init__(self, st):
        """Two rows of every reazon_small train shard get DEV_TEXT as their reference, one more CTC_TEXT as its
        Parakeet ctc_hyp; all are clean rows (labelled, not truncated, agreeing), so they are kept up to the dev
        rules."""
        special = st.missing | st.disagree | st.dup_ref | st.dup_hyp | st.short_dup | st.infeasible
        by_stem: dict[str, list[str]] = {}
        for u in st.fc.utts.values():
            if (u.source == "reazon_small" and u.split == "train" and u.has_teacher and not u.truncated
                    and u.agree is not None and u.agree <= 0.5 and u.id not in special):
                by_stem.setdefault(u.stem, []).append(u.id)
        self.ref, self.ctc = set(), set()
        for stem, rows in sorted(by_stem.items()):
            if len(rows) >= 3:
                self.ref |= set(rows[:2])
                self.ctc.add(rows[2])
                _rewrite_jsonl(st.teacher_out / "reazon_small" / f"{stem}.jsonl", {i: {"ref": DEV_TEXT} for i in rows[:2]})
                _rewrite_jsonl(st.parakeet_out / "reazon_small" / f"{stem}.jsonl", {rows[2]: {"ctc_hyp": CTC_TEXT}})


class Frozen:
    """The toy 'frozen' study build: its manifest and selection."""

    def __init__(self, d: Path):
        self.parquet = d / "labels/full/selections/study_1000h.parquet"
        self.manifest_path = self.parquet.parent / prereg.MANIFEST_FILE
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.sha = ms.file_sha256(self.manifest_path)
        self.sel = pd.read_parquet(self.parquet)
        self.sidecar = json.loads(self.parquet.with_suffix(".json").read_text(encoding="utf-8"))


class Built:
    """A full selection as written: the parquet, its sidecar and its recorded arguments."""

    def __init__(self, out: Path):
        self.parquet, self.sidecar_path = out, out.with_suffix(".json")
        self.sel = pd.read_parquet(out)
        self.sidecar = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        self.meta = json.loads(pq.read_schema(out).metadata[b"kitsune_selection"])
        self.args = self.meta["args"]

    def reason(self, ids) -> set[str]:
        return set(self.sel.set_index("id").loc[sorted(ids), "reason"])


def run(st, frozen: Frozen, d: Path, cfg: dict | None = None, *extra, video_rows=VIDEO_ROWS, max_frac=MAX_FRAC,
        out: Path | None = None, labels_root: Path | None = None, manifest: Path | None = None,
        manifest_sha: str | None = None):
    """make_selection.py's real CLI in full mode: the config file in d, the label root, the toy manifest."""
    cfg = cfg or full_cfg()
    d.mkdir(parents=True, exist_ok=True)
    cfg_p = d / "data-full.json"
    cfg_p.write_text(json.dumps(cfg), encoding="utf-8")
    out = out or d / cfg["selection"]
    argv = ["--config", str(cfg_p), "--labels-root", str(labels_root or st.fc.root), "--out", str(out),
            "--skip-audio-check", "--greedy-n", "3"]
    if manifest is not False:
        argv += ["--manifest", str(manifest or frozen.manifest_path), "--manifest-sha256", manifest_sha or frozen.sha]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(devslice, "DEV_VIDEO_ROWS", video_rows)
        mp.setattr(prereg, "ONE_ROOT_MAX_FRAC", max_frac)
        return ms.main([*argv, *extra]), out


def build(st, frozen, d: Path, cfg: dict | None = None, *extra, **kw) -> Built:
    rc, out = run(st, frozen, d, cfg, *extra, **kw)
    assert rc == 0
    return Built(out)


@pytest.fixture(scope="module")
def st(tmp_path_factory):
    st = make_study_corpus(tmp_path_factory.mktemp("full_corpus"), sources=CORPUS, id_fn=emilia_ids)
    st.planted = Planted(st)
    seal(st.fc.root)
    return st


@pytest.fixture(scope="module")
def frozen(st, tmp_path_factory) -> Frozen:
    d = tmp_path_factory.mktemp("frozen")
    cfg = full_cfg(selection=prereg.SELECTION_FILE, full_study=None,
                   study={"f1a_max": 0.5, "dedup_min_chars": 15, "draw_audio_s": 120, "probe_n": 4,
                          "neutral_max_cer": 0.5})
    (d / "study.json").write_text(json.dumps(cfg), encoding="utf-8")
    ms.main(["--config", str(d / "study.json"), "--out", str(d / prereg.SELECTION_FILE), "--extent-record",
             str(st.fc.root / kextent.RECORD_FILE), "--greedy-n", "3", "--teacher-out", str(st.teacher_out),
             "--second-out", str(st.second_out), "--data", str(st.data), "--parakeet-out", str(st.parakeet_out),
             "--kotoba-galgame", str(st.kotoba), "--skip-audio-check"])
    return Frozen(d)


@pytest.fixture(scope="module")
def built(st, frozen, tmp_path_factory) -> Built:
    return build(st, frozen, tmp_path_factory.mktemp("full"), None, "--study-selection", str(frozen.parquet))


@pytest.fixture(scope="module")
def smoke(st, frozen, tmp_path_factory) -> Built:
    return build(st, frozen, tmp_path_factory.mktemp("smoke"), full_cfg(draw=SMOKE_BUDGET_S))


def stem_of(sel: pd.DataFrame) -> pd.Series:
    return sel["teacher_file"].str.rsplit("/", n=1).str[1]


# ------------------------------------------------------------------------------------------------ the rules


def test_every_rule_drops_its_planted_rows(st, built):
    """The study rules take the rows the fixture made to fail them (a dev row keeps its reason); F0 is the recipe's;
    the full selection has no draw; eval rows are never filtered."""
    sel = built.sel
    assert built.reason(st.missing) == {"not_in_parakeet"}
    assert built.reason(st.disagree) == {"f1a_disagree"}
    assert built.reason(st.dup_ref | st.dup_hyp) == {"eval_dup"}
    assert built.reason(st.infeasible) == {"ctc_infeasible"}
    assert built.reason(st.short_dup) <= {"kept", "dev_buffer"}  # a 10-character eval reference is under the bar
    assert set(sel["id"][sel["reason"].isin(STUDY_DROPS)]) == (st.missing | st.disagree | st.dup_ref | st.dup_hyp
                                                               | st.infeasible)
    assert set(sel["reason"]) <= F0 | set(devslice.FULL_REASONS) - {"not_drawn"} | {"kept"}
    assert {"dev_buffer", "dev_dup"} <= set(sel["reason"])
    assert set(sel["split"]) == {"train", "dev", "eval"}
    ev = sel[sel["split"] == "eval"]
    assert ev["keep"].all() and set(ev["reason"]) == {"kept"}
    assert (sel["keep"] == (sel["reason"] == "kept")).all()
    assert list(sel.columns) == ms.COLUMNS


def test_rows_and_order_are_the_study_builds(built, frozen):
    """Same candidates, same order (selection order: sources in config order, stems sorted, teacher_out row order);
    a row the study kept is kept here unless the dev rules or the draw took it."""
    a, b = built.sel, frozen.sel
    assert a["id"].tolist() == b["id"].tolist()
    moved = (a["split"] != b["split"])
    assert set(a["split"][moved]) == {"dev"} and set(b["split"][moved]) == {"train"}
    was_kept = b["reason"].isin(["kept", "not_drawn"]).to_numpy()
    assert set(a["reason"][was_kept & (a["split"] == "train").to_numpy()]) <= {"kept", "dev_buffer", "dev_dup"}
    same = ~b["reason"].isin(["kept", "not_drawn"])
    assert (a["reason"][same] == b["reason"][same]).all()  # every earlier drop keeps its reason


def test_the_dev_slice(st, built, frozen):
    """Shards sources: every row of the seeded pair is dev, the buffers' live rows dev_buffer, never train-00000 in
    the pair; Emilia: whole videos up to the target; kept dev rows for every source, all train stems, none in the
    manifest; the sidecar's counts and hashes are the parquet's."""
    sel, side = built.sel, built.sidecar
    stem = stem_of(sel)
    dev = sel["split"] == "dev"
    for s in ("reazon_small", "galgame"):
        b = side["dev"]["by_source"][s]
        train_stems = sorted(p.stem for p in (st.teacher_out / s).glob("train-*.npz"))
        pair, bufs = devslice.pick_shard_pair(train_stems, 1234, s)
        assert (b["method"], b["dev_stems"], b["buffer_stems"]) == ("shards", pair, bufs)
        assert "train-00000" not in pair
        mine = sel["source"] == s
        assert set(sel["id"][mine & dev]) == set(sel["id"][mine & stem.isin(pair) & (sel["split"] != "eval")])
        buf = sel[mine & stem.isin(bufs)]
        assert (buf["split"] == "train").all() and not buf["keep"].any() and "dev_buffer" in set(buf["reason"])
        assert set(buf["reason"]) <= {"dev_buffer"} | F0 | STUDY_DROPS
    e = side["dev"]["by_source"]["emilia_yodas"]
    em = sel[sel["source"] == "emilia_yodas"]
    video = em["id"].map(devslice.emilia_video)
    dev_videos = sorted(set(video[em["split"] == "dev"]))
    assert e["method"] == "videos" and e["videos_n"] == len(dev_videos) == 3 and e["video_rows"] == VIDEO_ROWS
    assert e["videos_sha256"] == ids_sha256(dev_videos)
    assert (em["split"][video.isin(dev_videos)] == "dev").all() and (em["split"][~video.isin(dev_videos)] == "train").all()
    for s in SOURCES:
        b, m = side["dev"]["by_source"][s], dev & (sel["source"] == s)
        km = m & sel["keep"]
        assert b["kept_rows"] == int(km.sum()) > 0 and b["rows"] == int(m.sum())
        assert b["ids_sha256"] == ids_sha256(sel["id"][km].tolist())
        assert b["hours"] == pytest.approx(sel["duration"][m].astype(float).sum() / 3600)
    assert sel["teacher_file"][dev].str.fullmatch(r"[a-z_]+/train-\d{5}").all()
    man = {i for b in frozen.manifest["sets"].values() for i in b["ids"]}
    man |= {i for b in frozen.manifest["galgame_views"].values() for i in b["ids"]}
    assert not set(sel["id"][dev]) & man
    # the scored default: fullrun.dev_pick of the kept dev rows (600 per source: all of them here)
    kd = sel[dev & sel["keep"]]
    pick = fullrun.dev_pick(zip(kd["id"], kd["source"]), SOURCES, 600, 1234)
    sc = side["dev"]["scored_default"]
    assert sc == {"per_source": 600, "seed": 1234, "n": len(pick), "hours": pytest.approx(
        kd["duration"][kd["id"].isin(pick)].astype(float).sum() / 3600), "ids_sha256": ids_sha256(pick)}
    assert side["dev"]["rule"] == fullrun.DEV_RULE and side["dev"]["method"] == {s: devslice.DEV_METHOD[s]
                                                                                 for s in SOURCES}
    assert built.meta["kept"]["reazon_small/dev"]["utts"] == side["dev"]["by_source"]["reazon_small"]["kept_rows"]


def test_dev_dup_drops_train_copies_of_dev_texts(st, built):
    """A planted text (reference, or Parakeet ctc_hyp alone) inside the dev pair stays dev and kept; in a buffer it is
    dev_buffer (the buffer comes first); anywhere else in training it is dev_dup."""
    sel = built.sel.set_index("id")
    pair = set(built.sidecar["dev"]["by_source"]["reazon_small"]["dev_stems"])
    bufs = set(built.sidecar["dev"]["by_source"]["reazon_small"]["buffer_stems"])
    seen = {"dev": 0, "dev_buffer": 0, "dev_dup": 0}
    for kind in (st.planted.ref, st.planted.ctc):
        for i in kind:
            stem = sel.at[i, "teacher_file"].rsplit("/", 1)[1]
            if stem in pair:
                assert (sel.at[i, "split"], sel.at[i, "reason"]) == ("dev", "kept")
                seen["dev"] += 1
            elif stem in bufs:
                assert (sel.at[i, "split"], sel.at[i, "reason"]) == ("train", "dev_buffer")
                seen["dev_buffer"] += 1
            else:
                assert (sel.at[i, "split"], sel.at[i, "reason"]) == ("train", "dev_dup")
                seen["dev_dup"] += 1
    assert seen["dev"] >= 2 and seen["dev_dup"] >= 2, seen
    dups = built.sel[built.sel["reason"] == "dev_dup"]
    assert set(dups["id"]) == {i for i in st.planted.ref | st.planted.ctc
                               if sel.at[i, "reason"] == "dev_dup"}  # nothing else collides
    assert built.sidecar["dev"]["dev_dup"]["rows"] == len(dups)
    assert built.sidecar["dev"]["dev_dup"]["by_source"]["reazon_small"]["utts"] == len(dups)
    assert {i for i in st.planted.ctc if sel.at[i, "reason"] == "dev_dup"}, "the ctc_hyp copy alone drops a row"


def test_hours_n_and_hashes_follow_the_parquet(built):
    """hours[stage][source | total] equals the parquet's rows that survived that far; n and ids_sha256 are the
    kept train / dev / probe / eval ids in selection order; the leaving share is after_ctc - after_dev_dup."""
    sel, side = built.sel, built.sidecar
    train0 = sel["split"].isin(["train", "dev"])
    tr = sel["split"] == "train"
    r = sel["reason"]
    stages = {"teacher_out": train0, "candidates": train0 & (r != "not_in_parakeet"),
              "after_f0": train0 & ~r.isin(F0 | {"not_in_parakeet"}),
              "after_f1a": train0 & ~r.isin(F0 | {"not_in_parakeet", "f1a_disagree"}),
              "after_dedup": train0 & ~r.isin(F0 | {"not_in_parakeet", "f1a_disagree", "eval_dup"}),
              "after_ctc": train0 & r.isin(["kept", "dev_buffer", "dev_dup", "not_drawn"]),
              "after_dev": tr & r.isin(["kept", "dev_dup", "not_drawn"]),
              "after_dev_dup": tr & r.isin(["kept", "not_drawn"]), "drawn": tr & (r == "kept")}
    assert list(side["hours"]) == list(stages) == list(ms.FULL_STAGES)
    for k, m in stages.items():
        for s in [*SOURCES, "total"]:
            mm = m if s == "total" else m & (sel["source"] == s)
            assert side["hours"][k][s]["utts"] == int(mm.sum()), (k, s)
            assert side["hours"][k][s]["hours"] == pytest.approx(sel["duration"][mm].astype(float).sum() / 3600)
    tot = [side["hours"][k]["total"]["utts"] for k in stages]
    assert tot == sorted(tot, reverse=True)
    leave = side["hours"]["after_ctc"]["total"]["hours"] - side["hours"]["after_dev_dup"]["total"]["hours"]
    assert side["dev"]["leave_training_hours"] == pytest.approx(leave)
    assert side["dev"]["share_of_pool"] == pytest.approx(leave / side["hours"]["after_ctc"]["total"]["hours"])
    assert side["draw"] is None
    kept = sel["keep"]
    ids = {"train": sel["id"][tr & kept].tolist(), "dev": sel["id"][(sel["split"] == "dev") & kept].tolist(),
           "probe": sel["id"][sel["in_probe"]].tolist()}
    for k, v in ids.items():
        assert side["n"][k] == len(v) and side["ids_sha256"][k] == ids_sha256(v)
    probe = sel[sel["in_probe"]]
    assert probe["keep"].all() and (probe["split"] == "train").all()  # probe_n 300: every kept train row here
    assert len(probe) == int((tr & kept).sum())
    buf = r == "dev_buffer"
    assert side["dev"]["buffer"] == {"rows": int(buf.sum()),
                                     "hours": pytest.approx(sel["duration"][buf].astype(float).sum() / 3600)}


def test_eval_rows_greedy_subsets_and_baselines_are_the_studys(built, frozen, capsys):
    """The eval rows are the manifest's, per set and in order; with the same seed and greedy_n the greedy subsets are
    the study build's; the baselines on the manifest's rows and Galgame views equal the study sidecar's."""
    sel, side = built.sel, built.sidecar
    for s in EVAL_SETS:
        ids = sel["id"][(sel["source"] == s) & (sel["split"] == "eval") & sel["keep"]].tolist()
        assert ids == frozen.manifest["sets"][s]["ids"] and side["n"]["eval"][s] == len(ids)
        assert side["ids_sha256"]["eval"][s] == frozen.manifest["sets"][s]["ids_sha256"]
    ours = sel[(sel["split"] == "eval")].set_index("id")["in_greedy_subset"]
    theirs = frozen.sel[frozen.sel["split"] == "eval"].set_index("id")["in_greedy_subset"]
    pd.testing.assert_series_equal(ours, theirs)
    assert side["baselines"] == frozen.sidecar["baselines"]
    assert side["manifest"] == {"path": fullrun.FROZEN_MANIFEST, "sha256": frozen.sha, "eval_ids_equal": True}


def test_the_sidecar_passes_its_check(built, frozen):
    side = built.sidecar
    assert devslice.sidecar_problems(side, selection_sha256=ms.file_sha256(built.parquet),
                                     manifest_sha256=frozen.sha) == []
    assert side["kind"] == "full_study" and side["schema"] == 1
    assert side["selection"] == {"path": fullrun.FULL_SELECTION, "sha256": ms.file_sha256(built.parquet)}
    assert side["recipe"]["full_study"] == fullrun.FULL_STUDY and side["recipe"]["study"] is None
    assert side["extent"] == {"name": "full", "inputs": {"emilia_yodas": "300h"}}
    assert side["labels_complete_digest"] == "d" * 64 and side["seed"] == 1234
    assert side["k5"] == {"not_in_parakeet": 2, "candidates": int(built.sel["split"].isin(["train", "dev"]).sum()),
                          "frac": pytest.approx(2 / built.sel["split"].isin(["train", "dev"]).sum()),
                          "max_frac": MAX_FRAC, "ok": True, "by_source": side["k5"]["by_source"]}
    assert sum(b["not_in_parakeet"] for b in side["k5"]["by_source"].values()) == 2
    assert set(side["details"]) == {"reasons", "f1a_quantiles", "parakeet_only_rows", "parakeet_truncated"}
    assert side["details"]["parakeet_only_rows"] == 1


def test_a_rebuild_gives_the_same_bytes_and_records_no_path(st, frozen, built, tmp_path):
    """No timestamp and no machine path: rebuilt from another directory and config path, the parquet and the sidecar
    are byte-identical. The recorded arguments are FULL_ARGS plus the config's roots and the inputs' hashes."""
    again = build(st, frozen, tmp_path / "elsewhere" / "deeper")
    assert ms.file_sha256(again.parquet) == ms.file_sha256(built.parquet)
    assert ms.file_sha256(again.sidecar_path) == ms.file_sha256(built.sidecar_path)

    def strings(obj):
        if isinstance(obj, dict):
            return [x for v in obj.values() for x in strings(v)]
        return [x for v in obj for x in strings(v)] if isinstance(obj, list) else [obj] if isinstance(obj, str) else []

    args = built.args
    assert set(args) == {*ms.FULL_ARGS, "config_roots", "extent_record_sha256", "manifest_sha256",
                         "labels_complete_digest"}
    assert not [v for v in strings(args) if Path(v).is_absolute()]
    assert args["full_study"] == fullrun.FULL_STUDY and args["study"] is None and args["probe_n"] == 300
    assert args["manifest_sha256"] == frozen.sha and args["labels_complete_digest"] == "d" * 64
    assert args["extent_record_sha256"] == ms._json_sha256(toy_record(st.fc.root)) == \
        built.sidecar["extent_record_sha256"]
    assert args["config_roots"]["teacher_root"] == "labels/full/teacher_out"
    assert not [v for v in strings(built.sidecar) if Path(v).is_absolute()]


def test_never_writes_a_manifest(built):
    assert sorted(p.name for p in built.parquet.parent.iterdir()) == ["full.json", "full.parquet"]


# ------------------------------------------------------------------------------------------------ K5, the smoke


def test_k5_over_the_limit_exits_3_and_writes_nothing(st, frozen, tmp_path, capsys):
    rc, out = run(st, frozen, tmp_path, max_frac=prereg.ONE_ROOT_MAX_FRAC)
    assert rc == 3 and not out.parent.exists()
    text = capsys.readouterr().out
    assert "K5 FAIL: 2 of" in text and "nothing written" in text
    for i in st.missing:
        assert st.fc.utts[i].source + "/" + st.fc.utts[i].stem in text  # the per-stem breakdown


def test_the_script_exit_codes(st, frozen, tmp_path):
    """As a script: 3 on K5 (the real limit), 1 on a refusal, 2 on bad arguments."""
    cfg_p = tmp_path / "cfg.json"
    cfg_p.write_text(json.dumps(full_cfg()), encoding="utf-8")
    out = tmp_path / fullrun.FULL_SELECTION
    base = [sys.executable, str(ROOT / "scripts" / "make_selection.py"), "--config", str(cfg_p), "--labels-root",
            str(st.fc.root), "--out", str(out), "--skip-audio-check"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    man = ["--manifest", str(frozen.manifest_path), "--manifest-sha256", frozen.sha]
    r = subprocess.run(base + man, capture_output=True, text=True, env=env, encoding="utf-8")
    assert r.returncode == 3, r.stderr[-2000:]
    r = subprocess.run(base, capture_output=True, text=True, env=env, encoding="utf-8")
    assert r.returncode == 1 and "--manifest is required" in r.stderr
    r = subprocess.run(base + man + ["--bogus"], capture_output=True, text=True, env=env, encoding="utf-8")
    assert r.returncode == 2


def test_the_smoke_draw(st, frozen, built, smoke):
    """draw_audio_s: a seeded train draw (tag full:draw) of the live train rows after the dev rules; the dev slice
    is the full selection's (the draw comes after it); the probe comes from the drawn rows."""
    sel, side = smoke.sel, smoke.sidecar
    assert smoke.parquet.name == "smoke.parquet" and side["selection"]["path"] == fullrun.SMOKE_SELECTION
    d = side["draw"]
    pool = sel[(sel["split"] == "train") & sel["reason"].isin(["kept", "not_drawn"])]
    assert d["pool_s"] == pytest.approx(pool["duration"].astype(float).sum()) and d["pool_s"] > SMOKE_BUDGET_S
    assert d["drawn_s"] <= SMOKE_BUDGET_S and SMOKE_BUDGET_S - d["drawn_s"] < pool["duration"].max()
    drawn = set(sel["id"][(sel["split"] == "train") & sel["keep"]])
    assert d["drawn_utts"] == len(drawn) == side["n"]["train"] and d["pool_utts"] == len(pool)
    ids, dur = pool["id"].tolist(), pool["duration"].to_numpy()
    assert ms.draw_budget(ids, dur, SMOKE_BUDGET_S, 1234, tag="full:draw") == drawn
    assert ms.draw_budget(ids, dur, SMOKE_BUDGET_S, 1234, tag="study:draw") != drawn
    assert "not_drawn" in set(sel["reason"])
    assert side["dev"]["by_source"] == built.sidecar["dev"]["by_source"]
    assert (sel["split"] == built.sel["split"]).all()
    probe = sel[sel["in_probe"]]
    assert probe["keep"].all() and set(probe["id"]) <= drawn
    assert side["hours"]["drawn"]["total"]["hours"] * 3600 == pytest.approx(d["drawn_s"])
    assert side["hours"]["after_dev_dup"] == built.sidecar["hours"]["after_dev_dup"]
    assert smoke.args["full_study"] == dict(fullrun.FULL_STUDY, draw_audio_s=SMOKE_BUDGET_S)


# ------------------------------------------------------------------------------------------------ refusals


def refused(st, frozen, d: Path, cfg=None, *extra, **kw) -> str:
    with pytest.raises(SystemExit) as e:
        run(st, frozen, d, cfg, *extra, **kw)
    assert isinstance(e.value.code, str), e.value.code  # a refusal, exit 1 (not argparse's 2)
    return e.value.code


def test_refusals_before_any_work(st, frozen, tmp_path):
    t = iter(range(100))

    def d():
        return tmp_path / f"r{next(t)}"

    assert "--manifest is required" in refused(st, frozen, d(), manifest=False)
    assert "not the frozen manifest's" in refused(st, frozen, d(), manifest_sha="0" * 64)
    cfg = full_cfg()
    cfg["eval_sets"] = ["eval_cv8", "eval_jsut", "eval_reazon", "galgame"]
    assert "in that order" in refused(st, frozen, d(), cfg)
    assert "exclusive" in refused(st, frozen, d(), full_cfg(study=dict(prereg.STUDY_SELECTION)))
    assert "dev_rule" in refused(st, frozen, d(), full_cfg(full_study=dict(fullrun.FULL_STUDY, dev_rule=2)))
    assert "--from-selection" in refused(st, frozen, d(), None, "--from-selection", str(tmp_path / "x.parquet"))
    assert "drop --probe-n" in refused(st, frozen, d(), None, "--probe-n", "5")
    assert "filter_eval_sets" in refused(st, frozen, d(), full_cfg(filter_eval_sets=["galgame"]))
    other = full_cfg(selection="labels/full/selections/other.parquet")
    assert "not under labels/full/selections/full_study/" in refused(st, frozen, d(), other)
    named = full_cfg(selection=f"{fullrun.FULL_DIR}/study_1000h.parquet")
    assert "named like the frozen study selection" in refused(st, frozen, d(), named)
    assert "does not end in the config's selection" in refused(st, frozen, d(), None,
                                                                out=tmp_path / "loose" / "full.parquet")
    near = d()
    (near / fullrun.FULL_DIR).mkdir(parents=True)
    (near / fullrun.FULL_DIR / prereg.MANIFEST_FILE).write_text("{}", encoding="utf-8")
    assert "never writes next to the frozen study files" in refused(st, frozen, near)
    empty = d()
    empty.mkdir()
    assert "COMPLETE.json" in refused(st, frozen, d(), labels_root=empty)
    cfg = full_cfg()
    cfg["sources"] = ["reazon_small", "eval_cv8"]
    cfg["selection_recipe"]["agree_max_source"] = []
    cfg["extent"]["inputs"] = {}
    assert "no dev-slice method" in refused(st, frozen, d(), cfg)
    no_ext = {k: v for k, v in full_cfg().items() if k != "extent"}
    assert "no extent block" in refused(st, frozen, d(), no_ext)
    # the full-mode flags without a full config: argparse's refusal
    plain = full_cfg(full_study=None)
    with pytest.raises(SystemExit) as e:
        run(st, frozen, d(), plain)
    assert e.value.code == 2


def test_eval_rows_that_are_not_the_manifests_refuse(st, frozen, tmp_path):
    man = copy.deepcopy(frozen.manifest)
    man["sets"]["eval_cv8"]["ids"] = man["sets"]["eval_cv8"]["ids"][1:]
    p = tmp_path / "man.json"
    p.write_text(json.dumps(man), encoding="utf-8")
    msg = refused(st, frozen, tmp_path / "b", manifest=p, manifest_sha=ms.file_sha256(p))
    assert "not the frozen manifest's" in msg and "eval_cv8" in msg and "eval_jsut" not in msg


def test_a_partial_pull_fails_fast(st, frozen, tmp_path):
    """A missing second_out jsonl of a train stem or a Parakeet npz refuses before any row is read."""
    for rel in ("second_out/galgame/train-00002.jsonl", "parakeet_out/emilia_yodas/train-00001.npz",
                "teacher_out/eval_cv8/eval-00000.jsonl"):
        root = tmp_path / rel.replace("/", "_")
        for sub in ("teacher_out", "second_out", "parakeet_out"):
            shutil.copytree(st.fc.root / sub, root / sub)
        seal(root)
        (root / rel).unlink()
        msg = refused(st, frozen, tmp_path / ("o_" + root.name), labels_root=root)
        assert "partial pull" in msg and rel in msg, msg


def test_labels_root_fills_the_roots_and_explicit_flags_win(st, frozen, tmp_path, monkeypatch):
    seen = {}

    def capture(teacher_root, second_root, parakeet_root, data_root, *a, **kw):
        seen.update(teacher=Path(teacher_root), second=Path(second_root), parakeet=Path(parakeet_root),
                    stems=kw["stems"])
        raise SystemExit("captured")

    monkeypatch.setattr(ms, "build_full_selection", capture)
    with pytest.raises(SystemExit, match="captured"):
        run(st, frozen, tmp_path / "a")
    root = st.fc.root
    assert seen["teacher"] == root / "teacher_out" and seen["second"] == root / "second_out"
    assert seen["parakeet"] == root / "parakeet_out"
    assert seen["stems"] == kextent.subset_stems(toy_record(root), full_cfg())  # the record under the labels root
    copy_t = tmp_path / "t2"
    shutil.copytree(root / "teacher_out", copy_t)
    with pytest.raises(SystemExit, match="captured"):
        run(st, frozen, tmp_path / "b", None, "--teacher-out", str(copy_t))
    assert seen["teacher"] == copy_t and seen["parakeet"] == root / "parakeet_out"


def test_an_eval_row_without_parakeet_labels_refuses(frozen, tmp_path):
    """K6: every eval row must be in parakeet_out (here with no train row missing, so K5 passes)."""
    small = {"reazon_small": (32, "train"), "emilia_yodas": (24, "train"), "galgame": [(32, "train"), (16, "eval")],
             "eval_jsut": (12, "eval"), "eval_cv8": (8, "eval"), "eval_reazon": (8, "eval")}
    st2 = make_study_corpus(tmp_path / "c", seed=3, sources=small, id_fn=emilia_ids, n_missing=0, missing_eval=1)
    seal(st2.fc.root)
    msg = refused(st2, frozen, tmp_path / "o")
    assert "eval rows have no Parakeet labels" in msg and "(K6)" in msg


# ------------------------------------------------------------------------------------------------ launch


def have_of(b: Built, cfg: dict) -> set[str]:
    kept = b.sel[b.sel["keep"]]["teacher_file"].unique()
    files = {f"{cfg[r]}/{tf}{e}" for r in ("teacher_root", "parakeet_root") for tf in kept for e in (".npz", ".jsonl")}
    return files | set(devslice.selection_files(cfg))


def problems_of(*a) -> list[str]:
    return [p for p in launch.selection_problems(*a) if "no_agree" not in p]


def test_launch_accepts_the_built_selections(built, smoke, frozen, monkeypatch):
    """Each selection passes selection_problems with its own config (its sidecar, the frozen manifest and its files
    in the listing); the full one fails the smoke config and the other way round (full_study differs); a missing
    sidecar or manifest, or a recorded manifest sha that is not the frozen one, is a problem. (The fixture's one
    null-agree Emilia row is a no_agree drop, which launch flags on every selection: left out here.)"""
    monkeypatch.setattr(prereg, "ONE_ROOT_MAX_FRAC", MAX_FRAC)
    full, sm = dict(full_cfg(), pull_parakeet=True), dict(full_cfg(draw=SMOKE_BUDGET_S), pull_parakeet=True)
    assert problems_of(built.parquet, full["selection"], full)[0].startswith(
        f"{fullrun.FULL_SELECTION} was built against the manifest sha256 '{frozen.sha}'")  # not the real frozen one
    monkeypatch.setattr(fullrun, "FROZEN_MANIFEST_SHA256", frozen.sha)
    assert problems_of(built.parquet, full["selection"], full) == []
    assert problems_of(built.parquet, full["selection"], full, have_of(built, full)) == []
    assert problems_of(smoke.parquet, sm["selection"], sm, have_of(smoke, sm)) == []
    for gone in devslice.selection_files(full):
        assert problems_of(built.parquet, full["selection"], full, have_of(built, full) - {gone}) == [
            f"no {gone}: a full selection needs its sidecar and the frozen study manifest"]
    problems = problems_of(built.parquet, sm["selection"], sm)
    assert len(problems) == 1 and "was built with full_study" in problems[0]
    problems = problems_of(smoke.parquet, full["selection"], full)
    assert any("was built with full_study" in p for p in problems)
    assert any("['not_drawn'], which the full recipe never does" in p for p in problems)


FCFG = {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"], "student": "students/s",
        "selection": fullrun.FULL_SELECTION, "teacher_root": "labels/full/teacher_out",
        "parakeet_root": "labels/full/parakeet_out",
        "selection_recipe": {"agree_max": 0.5, "agree_max_source": [], "filter_eval_sets": [],
                             "partial_second_opinion": [], "study": None, "full_study": dict(fullrun.FULL_STUDY)}}
FARGS = {"sources": FCFG["sources"], "eval_sets": FCFG["eval_sets"], "agree_max": 0.5, "agree_max_source": [],
         "filter_eval_sets": [], "partial_second_opinion": [], "study": None, "full_study": dict(fullrun.FULL_STUDY),
         "seed": 1234, "manifest_sha256": fullrun.FROZEN_MANIFEST_SHA256}
FROWS = [("reazon_small", "train", True, "kept"), ("galgame", "train", True, "kept"),
         ("reazon_small", "dev", True, "kept"), ("galgame", "dev", True, "kept"), ("galgame", "dev", False, "truncated"),
         ("reazon_small", "train", False, "dev_buffer"), ("reazon_small", "train", False, "dev_dup"),
         ("galgame", "train", False, "eval_dup"), ("galgame", "train", False, "ctc_infeasible"),
         ("galgame", "train", False, "f1a_disagree"), ("eval_jsut", "eval", True, "kept"),
         ("galgame", "eval", True, "kept")]


def write_sel(path: Path, rows, args=None) -> Path:
    import pyarrow as pa

    df = pd.DataFrame(rows, columns=["source", "split", "keep", "reason"])
    df["teacher_file"] = df["source"] + "/" + df["split"].replace({"dev": "train"}) + "-00000"
    t = pa.Table.from_pandas(df, preserve_index=False)
    if args is not None:
        t = t.replace_schema_metadata({b"kitsune_selection": json.dumps(dict(args=args)).encode()})
    pq.write_table(t, path)
    return path


def test_selection_problems_know_the_full_recipe(tmp_path):
    name, path = FCFG["selection"], tmp_path / "sel.parquet"
    assert launch.selection_problems(write_sel(path, FROWS, FARGS), name, FCFG) == []
    # the recorded block against the configured one, and the configured one against the registered recipe
    smoke_args = dict(FARGS, full_study=dict(fullrun.SMOKE_STUDY))
    assert any("was built with full_study" in p for p in
               launch.selection_problems(write_sel(path, FROWS, smoke_args), name, FCFG))
    assert any("was built with full_study None" in p for p in
               launch.selection_problems(write_sel(path, FROWS, dict(FARGS, full_study=None)), name, FCFG))
    bad = copy.deepcopy(FCFG)
    bad["selection_recipe"]["full_study"]["dev_rule"] = 2
    problems = launch.selection_problems(write_sel(path, FROWS, dict(FARGS, full_study=dict(
        fullrun.FULL_STUDY, dev_rule=2))), name, bad)
    assert problems == ["the run config's selection_recipe.full_study.dev_rule 2, registered 1"]
    assert "was built against the manifest sha256 '00'" in launch.selection_problems(
        write_sel(path, FROWS, dict(FARGS, manifest_sha256="00")), name, FCFG)[0]
    # not_drawn only with a draw
    rows = FROWS + [("galgame", "train", False, "not_drawn")]
    assert any("['not_drawn'], which the full recipe never does" in p for p in
               launch.selection_problems(write_sel(path, rows, FARGS), name, FCFG))
    smoke_cfg = copy.deepcopy(FCFG)
    smoke_cfg["selection_recipe"]["full_study"] = dict(fullrun.SMOKE_STUDY)
    assert launch.selection_problems(write_sel(path, rows, smoke_args), name, smoke_cfg) == []
    # every train source keeps dev rows
    rows = [r for r in FROWS if not (r[0] == "galgame" and r[1] == "dev" and r[2])]
    assert any("keeps no dev rows of ['galgame']" in p for p in
               launch.selection_problems(write_sel(path, rows, FARGS), name, FCFG))
    # an unknown split
    rows = FROWS + [("galgame", "val", True, "kept")]
    assert any("rows of split ['val']" in p for p in launch.selection_problems(write_sel(path, rows, FARGS), name,
                                                                               FCFG))
    # K6, and K5 over the train AND dev rows: 1 of 1,101 passes (1 of the 600 train rows alone would not)
    rows = FROWS + [("eval_jsut", "eval", False, "not_in_parakeet")]
    assert any("(K6)" in p for p in launch.selection_problems(write_sel(path, rows, FARGS), name, FCFG))
    rows = (FROWS + [("galgame", "train", True, "kept")] * 590 + [("galgame", "dev", True, "kept")] * 500
            + [("galgame", "train", False, "not_in_parakeet")])
    assert launch.selection_problems(write_sel(path, rows, FARGS), name, FCFG) == []
    rows = FROWS + [("galgame", "train", True, "kept")] * 590 + [("galgame", "train", False, "not_in_parakeet")] * 2
    assert any("(K5)" in p for p in launch.selection_problems(write_sel(path, rows, FARGS), name, FCFG))


def test_selection_problems_refuse_dev_rows_without_the_full_recipe(tmp_path):
    """A config without full_study (the study's, the viability run's) never takes dev rows or dev reasons."""
    plain = copy.deepcopy(FCFG)
    del plain["selection_recipe"]["full_study"]
    args = {k: v for k, v in FARGS.items() if k not in ("full_study", "manifest_sha256")}
    path, name = tmp_path / "sel.parquet", FCFG["selection"]
    rows = [r for r in FROWS if r[1] != "dev" and r[3] not in {"dev_buffer", "dev_dup"} | STUDY_DROPS]
    assert launch.selection_problems(write_sel(path, rows, args), name, plain) == []
    problems = launch.selection_problems(write_sel(path, rows + [("galgame", "dev", True, "kept")], args), name, plain)
    assert problems == [f"{name} has rows of split ['dev'], which this recipe never writes (dev rows belong to a "
                        f"selection_recipe.full_study selection): rebuild it with scripts/make_selection.py --config "
                        f"<the run config> and upload it"]
    problems = launch.selection_problems(write_sel(path, rows + [("galgame", "train", False, "dev_buffer"),
                                                                 ("galgame", "train", False, "dev_dup")], args),
                                         name, plain)
    assert len(problems) == 1 and "['dev_buffer', 'dev_dup'], which this recipe never does" in problems[0]
    # the study recipe does not take them either
    study_cfg = copy.deepcopy(plain)
    study_cfg["selection_recipe"]["study"] = dict(prereg.STUDY_SELECTION)
    study_args = dict(args, study=dict(prereg.STUDY_SELECTION))
    problems = launch.selection_problems(write_sel(path, rows + [("galgame", "train", False, "dev_dup")], study_args),
                                         name, study_cfg)
    assert problems == [f"{name} drops rows as ['dev_dup'], which the study recipe never does: rebuild it with "
                        f"scripts/make_selection.py --config <the run config> and upload it"]
