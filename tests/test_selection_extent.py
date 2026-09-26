"""scripts/make_selection.py with run-config roots, id sidecars, extents and --from-selection; the full-extent configs.

The fake corpus uses canonical source names so that kitsune.extent plans and subsets it; its extent record is a toy
one (galgame's 6 train stems spread over 3 upstream inputs, its eval hold-out in input 0).
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, load_script, make_fake_corpus  # noqa: E402

import kitsune.extent as kextent  # noqa: E402
from kitsune.store import SIDECAR_DIR  # noqa: E402

ms = load_script("make_selection")
N = ["--greedy-n", "3", "--probe-n", "4"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return make_fake_corpus(
        tmp_path_factory.mktemp("corpus"),
        {"reazon_small": (40, "train"), "galgame": [(48, "train"), (16, "eval")], "eval_jsut": (16, "eval")},
        rows_per_shard=8, seed=3, truncated={"galgame": 3, "reazon_small": 2}, null_agree={"galgame": 2},
        no_second=("eval_jsut",),
    )


def _stem(stem: str, step: str) -> dict:
    return dict(stem=stem, split=stem.split("-")[0], step=step, rows=8, hours=0.01, ids_sha256="0" * 64,
                shard_bytes=1)


def toy_record() -> dict:
    gal = [[_stem("train-00000", "galgame"), _stem("train-00001", "galgame"), _stem("eval-00000", "galgame"),
            _stem("eval-00001", "galgame")],
           [_stem("train-00002", "galgame"), _stem("train-00003", "galgame")],
           [_stem("train-00004", "galgame"), _stem("train-00005", "galgame")]]
    return {"schema": kextent.RECORD_SCHEMA, "name": "t", "root": "labels/t",
            "canonical_version": kextent.CANONICAL_VERSION, "names": ["reazon_small", "galgame", "eval_jsut"],
            "inputs": {}, "sources": {
                "reazon_small": {"inputs": [{"input": "rs0", "ordinal": 0,
                                             "stems": [_stem(f"train-{i:05d}", "reazon_small") for i in range(5)]}]},
                "eval_jsut": {"inputs": [{"input": "jsut", "ordinal": 0,
                                          "stems": [_stem("eval-00000", "eval_jsut"), _stem("eval-00001", "eval_jsut")]}]},
                "galgame": {"inputs": [{"input": f"gal{o}", "ordinal": o, "stems": s} for o, s in enumerate(gal)]}}}


def extent_cfg(inputs=None, **recipe) -> dict:
    return {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"],
            "teacher_root": "labels/t/teacher_out", "second_root": "labels/t/second_out",
            "selection": "labels/t/selections/sel.parquet",
            "extent": {"name": "t", "root": "labels/t", "inputs": inputs or {}},
            "selection_recipe": dict({"agree_max": 0.5, "agree_max_source": [], "filter_eval_sets": ["galgame"],
                                      "partial_second_opinion": []}, **recipe)}


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def roots(fc) -> list[str]:
    return ["--teacher-out", str(fc.teacher_out), "--second-out", str(fc.second_out), "--data", str(fc.data)]


def run(cfg_path: Path, out: Path, *extra) -> pd.DataFrame:
    ms.main(["--config", str(cfg_path), "--out", str(out), *N, *extra])
    return pd.read_parquet(out)


def args_of(path: Path) -> dict:
    return json.loads(pq.read_schema(path).metadata[b"kitsune_selection"])["args"]


def test_config_roots_are_followed(corpus, tmp_path):
    cfg = {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"],
           "teacher_root": str(corpus.teacher_out), "second_root": str(corpus.second_out), "data_root": str(corpus.data),
           "selection_recipe": {"agree_max": 0.5, "agree_max_source": [], "filter_eval_sets": ["galgame"]}}
    p = write_json(tmp_path / "run.json", cfg)
    sel = run(p, tmp_path / "a.parquet")  # no root flags: the config's
    flags = run(p, tmp_path / "b.parquet", *roots(corpus))
    pd.testing.assert_frame_equal(sel, flags)
    assert set(sel["source"]) == {"reazon_small", "galgame", "eval_jsut"} and args_of(tmp_path / "a.parquet")["extent"] is None
    with pytest.raises(SystemExit, match="no teacher output"):  # an explicit flag wins over the config
        run(p, tmp_path / "c.parquet", "--teacher-out", str(tmp_path / "nowhere"))


def test_audio_check_reads_id_sidecars(corpus, tmp_path):
    """A pruned shard (audio deleted, id sidecar kept) counts as present, exactly as the shard did."""
    data = tmp_path / "data"
    shutil.copytree(corpus.data, data)
    before = {sp: ms.audio_ids(data, "galgame", sp) for sp in ("train", "eval")}
    for shard in sorted((data / "shards" / "galgame").glob("*.parquet")):
        t = pq.read_table(shard, columns=["id", "duration"])
        (shard.parent / SIDECAR_DIR).mkdir(exist_ok=True)
        pq.write_table(t, shard.parent / SIDECAR_DIR / shard.name)
        shard.unlink()
    assert {sp: ms.audio_ids(data, "galgame", sp) for sp in ("train", "eval")} == before and before["train"]
    a = ms.build_selection(corpus.teacher_out, corpus.second_out, corpus.data, ["galgame"], ["galgame"])
    b = ms.build_selection(corpus.teacher_out, corpus.second_out, data, ["galgame"], ["galgame"])
    pd.testing.assert_frame_equal(a, b)
    assert "no_audio" not in set(b["reason"])


def test_extent_filter_drops_out_of_extent_stems(corpus, tmp_path):
    rec = tmp_path / "extent.json"
    kextent.write_record(rec, toy_record())
    full = run(write_json(tmp_path / "full.json", extent_cfg()), tmp_path / "full.parquet", *roots(corpus),
               "--extent-record", str(rec))
    sub_p = write_json(tmp_path / "sub.json", extent_cfg({"galgame": 2}))
    sub = run(sub_p, tmp_path / "sub.parquet", *roots(corpus), "--extent-record", str(rec))
    gal = sub[sub["source"] == "galgame"]
    assert set(gal.loc[gal["split"] == "train", "teacher_file"]) == {f"galgame/train-0000{i}" for i in range(4)}
    assert set(gal.loc[gal["split"] == "eval", "teacher_file"]) == {"galgame/eval-00000", "galgame/eval-00001"}
    assert set(full.loc[(full["source"] == "galgame") & (full["split"] == "train"), "teacher_file"]) == {
        f"galgame/train-0000{i}" for i in range(6)}
    for src in ("reazon_small", "eval_jsut"):  # uncapped names keep every stem
        assert (sub["source"] == src).sum() == (full["source"] == src).sum() > 0
    assert args_of(tmp_path / "sub.parquet")["extent"] == {"name": "t", "inputs": {"galgame": 2}}
    assert args_of(tmp_path / "full.parquet")["extent"] == {"name": "t", "inputs": {}}
    # a record for another root, and an invalid extent config (a partial recipe), are refused
    bad = dict(toy_record(), root="labels/other")
    kextent.write_record(tmp_path / "bad.json", bad)
    with pytest.raises(SystemExit):
        run(sub_p, tmp_path / "x.parquet", *roots(corpus), "--extent-record", str(tmp_path / "bad.json"))
    partial = write_json(tmp_path / "partial.json", extent_cfg(partial_second_opinion=["galgame"]))
    with pytest.raises(SystemExit):
        run(partial, tmp_path / "y.parquet", *roots(corpus), "--extent-record", str(rec))
    no_ext = write_json(tmp_path / "noext.json", {k: v for k, v in extent_cfg().items() if k != "extent"})
    with pytest.raises(SystemExit):  # --extent-record without an extent
        run(no_ext, tmp_path / "z.parquet", *roots(corpus), "--extent-record", str(rec))


@pytest.mark.parametrize("change", ["subset", "threshold", "both"])
def test_from_selection_equals_a_from_scratch_build(corpus, tmp_path, change):
    rec = tmp_path / "extent.json"
    kextent.write_record(rec, toy_record())
    base = tmp_path / "base.parquet"
    run(write_json(tmp_path / "full.json", extent_cfg()), base, *roots(corpus), "--extent-record", str(rec))
    cfg = extent_cfg({"galgame": 2} if change != "threshold" else None,
                     **({} if change == "subset" else {"agree_max": 0.3, "agree_max_source": ["galgame=0.2"]}))
    p = write_json(tmp_path / "derived.json", cfg)
    scratch = run(p, tmp_path / "scratch.parquet", *roots(corpus), "--extent-record", str(rec))
    # derived: no teacher_out, second_out or audio (the roots point nowhere)
    nowhere = ["--teacher-out", str(tmp_path / "t"), "--second-out", str(tmp_path / "s"), "--data", str(tmp_path / "d")]
    derived = run(p, tmp_path / "derived.parquet", *nowhere, "--extent-record", str(rec), "--from-selection", str(base))
    pd.testing.assert_frame_equal(derived, scratch)
    assert derived["in_probe"].any() and derived["in_greedy_subset"].any() and not derived["keep"].all()
    a = args_of(tmp_path / "derived.parquet")
    assert a["from_selection"] == {"path": str(base), "sha256": ms.file_sha256(base)}
    assert a["extent"] == args_of(tmp_path / "scratch.parquet")["extent"]


def test_from_selection_refusals(corpus, tmp_path):
    rec = tmp_path / "extent.json"
    kextent.write_record(rec, toy_record())
    sub_p = write_json(tmp_path / "sub.json", extent_cfg({"galgame": 2}))
    base = tmp_path / "base.parquet"
    run(sub_p, base, *roots(corpus), "--extent-record", str(rec))
    derive = ["--extent-record", str(rec), "--from-selection"]
    # a base narrower than the config's extent
    with pytest.raises(SystemExit):
        run(write_json(tmp_path / "full.json", extent_cfg()), tmp_path / "x.parquet", *derive, str(base))
    # a base with no_audio rows
    t = pq.read_table(base)
    reason = t.column("reason").to_pylist()
    reason[0] = "no_audio"
    t = t.set_column(t.schema.get_field_index("reason"), "reason", pa.array(reason, pa.string()))
    pq.write_table(t, tmp_path / "noaudio.parquet")
    with pytest.raises(SystemExit):
        run(sub_p, tmp_path / "x.parquet", *derive, str(tmp_path / "noaudio.parquet"))
    with pytest.raises(ValueError, match="no_audio"):
        ms.derive_selection(pd.read_parquet(tmp_path / "noaudio.parquet"), ["galgame"], ["galgame"])
    # an extent config from a base built without one, and a partial recipe (no extent, so validate does not see it)
    plain = tmp_path / "plain.parquet"
    ms.main(["--sources", "reazon_small", "galgame", "--eval-sets", "eval_jsut", "galgame", "--out", str(plain),
             *roots(corpus), *N])
    with pytest.raises(SystemExit):
        run(sub_p, tmp_path / "x.parquet", *derive, str(plain))
    no_ext = {k: v for k, v in extent_cfg().items() if k != "extent"}
    partial = write_json(tmp_path / "partial.json", dict(no_ext, selection_recipe=dict(
        no_ext["selection_recipe"], partial_second_opinion=["galgame"])))
    with pytest.raises(SystemExit):
        run(partial, tmp_path / "x.parquet", "--from-selection", str(plain))
    # without an extent, a rethreshold from a plain base works and equals the from-scratch build
    thr = write_json(tmp_path / "thr.json", dict(no_ext, selection_recipe=dict(no_ext["selection_recipe"], agree_max=0.2)))
    scratch = run(thr, tmp_path / "s.parquet", *roots(corpus))
    pd.testing.assert_frame_equal(run(thr, tmp_path / "d.parquet", "--from-selection", str(plain)), scratch)


def test_full_configs_resolve():
    """Both configs load through the trainer's load_config, pass kitsune.extent.validate and make_selection's recipe
    checks, and the subset differs from full only where it must (and lies inside it)."""
    m = load_script("04_distill")
    assert all(m.DEFAULTS[k] is None for k in ("extent", "parakeet_root", "label"))
    raw = {n: json.loads((ROOT / "configs" / f"{n}.json").read_text(encoding="utf-8")) for n in ("full", "full_sub3k")}
    for name, cfg in raw.items():
        resolved = m.load_config(str(ROOT / "configs" / f"{name}.json"), [])
        for k, v in cfg.items():
            if not k.startswith("_"):  # a section resolves to itself over the DEFAULTS keys it leaves out
                d = m.DEFAULTS.get(k)
                assert resolved[k] == (dict(d, **v) if isinstance(d, dict) and isinstance(v, dict) else v), k
        assert resolved["loss"] == m.DEFAULTS["loss"]  # the trainer sections inherit DEFAULTS
        assert kextent.validate(cfg) == [], name
        recipe = cfg["selection_recipe"]
        by_source = {s.partition("=")[0] for s in recipe["agree_max_source"]}
        assert by_source <= set(cfg["sources"]) | set(recipe["filter_eval_sets"])
        assert not set(recipe["filter_eval_sets"]) & set(ms.EVAL_SETS) and recipe["partial_second_opinion"] == []
    full, sub = raw["full"], raw["full_sub3k"]
    differ = {k for k in set(full) | set(sub) if k != "_comment" and full.get(k) != sub.get(k)}
    assert differ == {"run_name", "selection", "sources", "extent", "selection_recipe", "label"}
    assert dict(full["extent"], inputs=None) == dict(sub["extent"], inputs=None)
    assert {k for k in full["selection_recipe"] if full["selection_recipe"][k] != sub["selection_recipe"][k]} == {
        "agree_max_source"}
    assert kextent.within(full, sub) == []
    assert sub["extent"]["inputs"] == {"reazon_large": 216, "galgame": 16}
    viability = json.loads((ROOT / "configs" / "viability.json").read_text(encoding="utf-8"))
    assert kextent.extent_block(viability) is None
    assert "labels/" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()


def test_viability_roots_equal_the_old_defaults():
    """--config configs/viability.json resolves the roots it always used (teacher_out, second_out, data)."""
    cfg = json.loads((ROOT / "configs" / "viability.json").read_text(encoding="utf-8"))
    assert (cfg["teacher_root"], cfg["second_root"], cfg["data_root"]) == ("teacher_out", "second_out", "data")
    assert ms._repo_path(cfg["teacher_root"]) == ROOT / "teacher_out"


def test_a_missing_subset_npz_or_base_stem_is_refused(corpus, tmp_path):
    """A partly pulled labels root must not quietly give a smaller selection, built or derived."""
    rec = tmp_path / "extent.json"
    kextent.write_record(rec, toy_record())
    teacher = tmp_path / "teacher_out"
    shutil.copytree(corpus.teacher_out, teacher)
    (teacher / "galgame" / "train-00001.npz").unlink()
    cfg_p = write_json(tmp_path / "sub.json", extent_cfg({"galgame": 2}))
    with pytest.raises(ValueError, match="galgame/train-00001"):
        run(cfg_p, tmp_path / "x.parquet", "--teacher-out", str(teacher), "--second-out", str(corpus.second_out),
            "--data", str(corpus.data), "--extent-record", str(rec))
    base = tmp_path / "base.parquet"
    run(cfg_p, base, *roots(corpus), "--extent-record", str(rec))
    b = pd.read_parquet(base)
    with pytest.raises(ValueError, match="train-00001"):
        ms.derive_selection(b[b["teacher_file"] != "galgame/train-00001"], ["reazon_small", "galgame"],
                            ["eval_jsut", "galgame"], stems=kextent.subset_stems(toy_record(), extent_cfg({"galgame": 2})))


def test_a_row_without_audio_hidden_under_another_reason_blocks_a_derive(corpus, tmp_path):
    """no_audio has the lowest precedence: a base row without audio can be stored as agree>0.5, and a looser derived
    threshold would then keep it. The base's count of rows without audio (any reason) refuses the derive."""
    data = tmp_path / "data"
    shutil.copytree(corpus.data, data)
    for shard in sorted((data / "shards" / "galgame").glob("train-*.parquet")):
        shard.unlink()  # no sidecar either: these rows have no audio
    plain = tmp_path / "plain.parquet"
    ms.main(["--sources", "reazon_small", "galgame", "--eval-sets", "eval_jsut", "galgame", "--out", str(plain),
             "--teacher-out", str(corpus.teacher_out), "--second-out", str(corpus.second_out), "--data", str(data), *N])
    meta = json.loads(pq.read_schema(plain).metadata[b"kitsune_selection"])
    assert meta["n_no_audio_any"] > 0
    no_ext = {k: v for k, v in extent_cfg().items() if k != "extent"}
    loose = write_json(tmp_path / "loose.json", dict(no_ext, selection_recipe=dict(no_ext["selection_recipe"],
                                                                                   agree_max=0.9)))
    with pytest.raises(SystemExit):
        run(loose, tmp_path / "d.parquet", "--from-selection", str(plain))
