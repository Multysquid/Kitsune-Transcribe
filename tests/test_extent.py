"""Tests for kitsune/extent.py and scripts/01_prepare_data.py --extent-config: the canonical ingest sequence and the
extent record.

Utterance ids depend on the ingest order, and the label box decides per shard stem what is labelled. So the canonical
sequence must give the laptop's ids AND stems; a capped extent must give a prefix of the full run's stems, because the
A100 rebuilds a subset and pulls the label box's files for exactly those stems; and the record must say which stems a
subset takes, which files to pull and how big the rebuild is.

The upstreams are faked with tests/test_prepare_data.py's tar and parquet builders, but per repo: one 01 process now
reads every source. Toy sizes: 4 rows per shard, a 3-row galgame hold-out, and a 300 h step whose budget (5 s) cuts
inside the second Emilia tar, so EMILIA_300H_INPUTS is 2 here (9 on the real data). CPU only, no network.
"""
import functools
import gc
import json
import shutil
import sys
import time
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import load_script  # noqa: E402
from test_prepare_data import make_galgame_tar, make_reazon_parquet, make_tar  # noqa: E402

from kitsune import extent  # noqa: E402
from kitsune.extent import Step  # noqa: E402
from kitsune.store import ShardWriter, iter_rows, load_progress, read_manifest, sidecar_meta  # noqa: E402

prep = load_script("01_prepare_data")

R_SMALL, R_LARGE = (prep.HF_PARQUET_SOURCES[s][0] for s in ("reazon_small", "reazon_large"))
BUDGET_S = 5.0  # the toy "300 h": 2 + 2 s in tar 0, then the first 2 s clip of tar 1 crosses it
FULL_SOURCES = ["reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame"]
EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"]
JA = "日本語です。"


@pytest.fixture(scope="module")
def upstream(tmp_path_factory) -> dict[str, dict[str, Path]]:
    """repo -> {upstream file name: local file}. Built to hit every awkward case: reazon_large rows already in
    reazon_small and an input that keeps no row, a galgame hold-out in tar 0, an Emilia video (vidE) in a training tar
    past the 300 h cut and in the eval tar, and an Emolia tar with only English text."""
    d = tmp_path_factory.mktemp("upstream")
    up: dict[str, dict[str, Path]] = {}

    def put(repo: str, name: str, build, *args, **kw):
        path = d / repo.replace("/", "--") / name.replace("/", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        build(path, *args, **kw)
        up.setdefault(repo, {})[name] = path

    put(R_SMALL, "small/train-00000-of-00002.parquet", make_reazon_parquet, [f"s0r{i}" for i in range(5)])
    put(R_SMALL, "small/train-00001-of-00002.parquet", make_reazon_parquet, [f"s1r{i}" for i in range(3)])
    put(R_LARGE, "large/train-00000-of-00003.parquet", make_reazon_parquet, ["s0r1", "l0r0", "l0r1", "s1r2", "l0r2"])
    put(R_LARGE, "large/train-00001-of-00003.parquet", make_reazon_parquet, ["s0r0", "s0r2"])  # all in reazon_small
    put(R_LARGE, "large/train-00002-of-00003.parquet", make_reazon_parquet, [f"l2r{i}" for i in range(5)])
    for s in extent.GATE_SETS:
        put(prep.HF_PARQUET_SOURCES[s][0], "data/test-00000-of-00001.parquet", make_reazon_parquet, ["x0", "x1"])
    for j, keys in enumerate(([f"k{i}" for i in range(6)], [f"m{i}" for i in range(5)], [f"n{i}" for i in range(3)])):
        put(prep.GALGAME_REPO, f"data/galgame-00000{j}.tar", make_galgame_tar, [(k, "テキストです。", 1.0) for k in keys])
    for tar, vids in (("JA-B000000", ["vidA_W000001", "vidA_W000002"]),
                      ("JA-B000001", ["vidB_W000001", "vidE_W000002", "vidC_W000001", "vidB_W000002"]),
                      ("JA-B000002", ["vidD_W000001", "vidE_W000003", "vidD_W000002"]),
                      ("JA-B000029", ["vidE_W000001", "vidA_W000003", "vidF_W000001"])):  # the eval_emilia tar
        put(prep.EMILIA_REPO, f"JA/{tar}.tar", make_tar, [(f"JA_{v}", JA, 2.0) for v in vids])
    for j, clips in enumerate(([("W000000", JA), ("W000001", "english only")], [("W000000", "english words")],
                               [("W000000", JA), ("W000001", JA)])):
        put(prep.EMOLIA_REPO, f"JA-B00000{j}_standard.tar.gz", make_tar,
            [(f"JA_B0000{j}_S00000_{w}", text, 2.0) for w, text in clips], gz=True, worker=f"worker_{j}/")
    return up


def patch_world(mp, upstream: dict, dl: Path) -> SimpleNamespace:
    """Serve `upstream` per repo at the pinned revisions and shrink the sizes; returns the recorded listings and
    downloads (repo, file name), in order."""
    hub = SimpleNamespace(listings=[], downloads=[], upstream=upstream,
                          size=lambda repo, name: upstream[repo][name].stat().st_size)

    class Api:
        def list_repo_files(self, repo, repo_type=None, revision=None):
            assert revision == prep.REVISIONS[repo], "every listing must use the pinned revision"
            hub.listings.append(repo)
            return sorted(upstream.get(repo, {})) + ["README.md"]

    def hf_hub_download(repo, filename, repo_type=None, revision=None, cache_dir=None):
        assert revision == prep.REVISIONS[repo] and filename in upstream[repo]
        hub.downloads.append((repo, filename))
        dst = dl / repo.replace("/", "--") / filename.replace("/", "_")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(upstream[repo][filename], dst)
        return str(dst)

    mp.setattr(prep, "HfApi", Api)
    mp.setattr(prep, "hf_hub_download", hf_hub_download)
    mp.setattr(prep.Ingest, "free_download", staticmethod(lambda local: Path(local).unlink(missing_ok=True)))
    mp.setattr(prep, "ShardWriter", functools.partial(ShardWriter, rows_per_shard=4))
    mp.setattr(prep, "GALGAME_EVAL_ROWS", 3)
    mp.setattr(extent, "CANONICAL", tuple(Step(s.key, s.source, BUDGET_S / 3600) if s.key == "emilia_yodas@300h" else s
                                          for s in extent.CANONICAL))
    mp.setattr(extent, "EMILIA_300H_INPUTS", 2)
    return hub


@pytest.fixture()
def world(upstream, tmp_path, monkeypatch) -> SimpleNamespace:
    return patch_world(monkeypatch, upstream, tmp_path / "dl")


def make_cfg(name: str = "full", inputs: dict | None = None, sources=FULL_SOURCES, eval_sets=EVAL_SETS,
             root: str = "labels/full") -> dict:
    """A run config shaped like configs/full.json."""
    return {"run_name": f"{name}-b20x2560", "student": "students/b20x2560-d4", "data_root": "data",
            "teacher_root": f"{root}/teacher_out", "second_root": f"{root}/second_out",
            "parakeet_root": f"{root}/parakeet_out", "selection": f"{root}/selections/{name}.parquet",
            "extent": {"name": name, "root": root, "inputs": dict(inputs or {})},
            "sources": list(sources), "eval_sets": list(eval_sets),
            "selection_recipe": {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "eval_emilia=0.2"],
                                 "filter_eval_sets": [s for s in ("eval_emilia", "galgame") if s in eval_sets],
                                 "partial_second_opinion": []}}


def write_cfg(path: Path, cfg: dict) -> Path:
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def run01(mp, root: Path, *argv):
    """01's real CLI into `root`. A run into a root the previous run just released may find the lock still held for a
    moment on Windows: retry."""
    mp.setattr(sys, "argv", ["01_prepare_data.py", "--data", str(root), *argv])
    for _ in range(40):
        try:
            return prep.main()
        except SystemExit as e:
            if "another 01_prepare_data" not in str(e.code):
                raise
            gc.collect()
            time.sleep(0.05)
    raise AssertionError(f"{root} stayed locked")


def layout(root: Path) -> dict[str, list[str]]:
    """{manifest path: ids read from the shard file}: the stems and their rows."""
    return {sh.path: [r["id"] for r in iter_rows(root / sh.path, ["id"])] for sh in read_manifest(root)}


def stems_of(paths, source: str) -> set[str]:
    return {Path(p).stem for p in paths if p.split("/")[1] == source}


# the full toy extent, spelled out: what the laptop order gives, and so what the canonical sequence must give
FULL_LAYOUT = {
    **{f"shards/reazon_small/train-0000{i}.parquet": [f"reazon_small/{n}" for n in names] for i, names in enumerate(
        (["s0r0", "s0r1", "s0r2", "s0r3"], ["s0r4"], ["s1r0", "s1r1", "s1r2"]))},
    **{f"shards/{s}/eval-00000.parquet": [f"{s}/test-00000-of-00001-0", f"{s}/test-00000-of-00001-1"]
       for s in extent.GATE_SETS},
    "shards/emilia_yodas/train-00000.parquet": ["emilia_yodas/JA_vidA_W000001", "emilia_yodas/JA_vidA_W000002"],
    "shards/emilia_yodas/train-00001.parquet": ["emilia_yodas/JA_vidB_W000001"],  # the 300 h cut inside tar 1
    "shards/eval_emilia/eval-00000.parquet": ["eval_emilia/JA_vidE_W000001", "eval_emilia/JA_vidF_W000001"],
    **{f"shards/galgame/{stem}.parquet": [f"galgame/{k}" for k in keys] for stem, keys in (
        ("eval-00000", ["k0", "k1", "k2"]), ("train-00000", ["k3", "k4", "k5"]),
        ("train-00001", ["m0", "m1", "m2", "m3"]), ("train-00002", ["m4"]), ("train-00003", ["n0", "n1", "n2"]))},
    # the rest continues tar 1 without vidE (eval_emilia's video)
    "shards/emilia_yodas/train-00002.parquet": ["emilia_yodas/JA_vidC_W000001", "emilia_yodas/JA_vidB_W000002"],
    "shards/emilia_yodas/train-00003.parquet": ["emilia_yodas/JA_vidD_W000001", "emilia_yodas/JA_vidD_W000002"],
    "shards/emilia_nc/train-00000.parquet": ["emilia_nc/JA_B00000_S00000_W000000"],  # tar 1 keeps nothing
    "shards/emilia_nc/train-00001.parquet": [f"emilia_nc/JA_B00002_S00000_W00000{i}" for i in range(2)],
    "shards/reazon_large/train-00000.parquet": ["reazon_large/l0r0", "reazon_large/l0r1", "reazon_large/l0r2"],
    "shards/reazon_large/train-00001.parquet": [f"reazon_large/l2r{i}" for i in range(4)],  # file 1 keeps nothing
    "shards/reazon_large/train-00002.parquet": ["reazon_large/l2r4"],
}


@pytest.fixture(scope="module")
def full(upstream, tmp_path_factory) -> SimpleNamespace:
    """The full toy extent, ingested once by 01 --extent-config, and its record. Read-only for the tests."""
    d = tmp_path_factory.mktemp("full")
    with pytest.MonkeyPatch.context() as mp:
        hub = patch_world(mp, upstream, d / "dl")
        cfg = make_cfg()
        config = write_cfg(d / "full.json", cfg)
        run01(mp, d / "data", "--extent-config", str(config))
        record = extent.build_record(d / "data", cfg, run_ids=["20260925T000000Z-1"], kitsune_sha="0" * 40)
    return SimpleNamespace(root=d / "data", cfg=cfg, config=config, record=record, hub=hub, layout=layout(d / "data"))


def test_canonical_run_reproduces_the_laptop_order(full, world, tmp_path):
    """The laptop ingested reazon_small, galgame's first tars, the gate sets, the rest of galgame, Emilia at 300 h,
    eval_emilia, the rest of Emilia, emilia_nc and reazon_large, each a separate run. The one canonical process gives
    the same ids in the same stems, which is what lets the label box adopt the laptop's labels and pull its files."""
    root = tmp_path / "laptop"

    def step(source: str, fn, *args):
        fn(prep.Ingest(root, root / "raw", source, None), *args)

    step("reazon_small", prep.ingest_hf_parquet, *prep.HF_PARQUET_SOURCES["reazon_small"])
    step("galgame", prep.ingest_galgame, 1)  # the laptop's first --galgame-shards 6
    for s in extent.GATE_SETS:
        step(s, prep.ingest_hf_parquet, *prep.HF_PARQUET_SOURCES[s])
    step("galgame", prep.ingest_galgame, 3)
    step("emilia_yodas", prep.ingest_emilia, BUDGET_S / 3600)  # --emilia-hours 300
    step("eval_emilia", prep.ingest_emilia_eval, prep.EMILIA_EVAL_TAR, prep.EMILIA_EVAL_ROWS)
    step("emilia_yodas", prep.ingest_emilia, float("inf"))
    step("emilia_nc", prep.ingest_emilia_nc, float("inf"))
    step("reazon_large", prep.ingest_hf_parquet, *prep.HF_PARQUET_SOURCES["reazon_large"])
    assert layout(root) == FULL_LAYOUT
    assert full.layout == FULL_LAYOUT
    # and the canonical run went through the steps in the canonical order, each source listed once per step
    assert full.hub.listings == [prep.source_repo(s.source) for s in extent.CANONICAL if s.source != "eval_emilia"]


def test_a_single_uncapped_emilia_call_differs(full, world, tmp_path):
    """Why the sequence exists: `--sources emilia_yodas eval_emilia --emilia-hours inf` ingests all of Emilia with no
    hold-out videos to leave out, so it keeps vidE's training clips, and eval_emilia then loses vidE's clip."""
    root = tmp_path / "single"
    prep.ingest_emilia(prep.Ingest(root, root / "raw", "emilia_yodas", None), float("inf"))
    prep.ingest_emilia_eval(prep.Ingest(root, root / "raw", "eval_emilia", None), prep.EMILIA_EVAL_TAR, 1000)
    got = layout(root)
    assert got["shards/eval_emilia/eval-00000.parquet"] == ["eval_emilia/JA_vidF_W000001"]
    assert got["shards/emilia_yodas/train-00001.parquet"] == [f"emilia_yodas/JA_{v}" for v in (
        "vidB_W000001", "vidE_W000002", "vidC_W000001", "vidB_W000002")]
    assert full.layout["shards/eval_emilia/eval-00000.parquet"] == ["eval_emilia/JA_vidE_W000001",
                                                                    "eval_emilia/JA_vidF_W000001"]


@pytest.mark.parametrize("sources, eval_sets, inputs", [
    (FULL_SOURCES, EVAL_SETS, {"reazon_large": 1, "galgame": 2, "emilia_nc": 1, "emilia_yodas": 2}),  # K = 9 here
    (FULL_SOURCES, EVAL_SETS, {"reazon_large": 2, "galgame": 1, "emilia_nc": 2, "emilia_yodas": "300h"}),
    (["reazon_large", "galgame"], ["galgame"], {"reazon_large": 1}),  # reazon_small is rebuilt, not taken
], ids=["caps", "300h", "few-names"])
def test_capped_extent_is_a_prefix_of_the_full_run(full, world, tmp_path, monkeypatch, sources, eval_sets, inputs):
    """A subset config rebuilds only its closure's inputs, and every stem it rebuilds is the full run's stem with the
    same rows, so the A100 joins the label box's files by stem. The stems it takes are what subset_stems says, their
    ids_sha256 the record's, and sizing's download and shard bytes are what the rebuild really moved."""
    cfg = make_cfg("sub", inputs, sources, eval_sets)
    assert extent.validate(cfg) == [] and extent.within(full.cfg, cfg) == []
    root = tmp_path / "data"
    run01(monkeypatch, root, "--extent-config", str(write_cfg(tmp_path / "sub.json", cfg)))
    got = layout(root)
    assert {p: full.layout[p] for p in got} == got
    closure = {s.source for s, _ in extent.plan_steps(cfg)}
    assert {p.split("/")[1] for p in got} == closure
    taken = extent.subset_stems(full.record, cfg)
    assert set(taken) == set(extent.names(cfg))
    for source in closure:
        want = taken[source] if source in taken else stems_of(full.layout, source)
        assert stems_of(got, source) == want, source
    recorded = {(s, st["stem"]): st["ids_sha256"] for s, src in full.record["sources"].items()
                for inp in src["inputs"] for st in inp["stems"]}
    for sh in read_manifest(root):
        assert sidecar_meta(root, sh)["ids_sha256"] == recorded[(sh.source, Path(sh.path).stem)]
    size = extent.sizing(full.record, cfg)
    assert size["down_gb"] * 1e9 == pytest.approx(sum(world.size(r, f) for r, f in world.downloads))
    assert size["shard_gb"] * 1e9 == pytest.approx(sum((root / sh.path).stat().st_size for sh in read_manifest(root)))
    if inputs.get("emilia_yodas") == 2:  # the rest step reads the cut tar again
        assert world.downloads.count((prep.EMILIA_REPO, "JA/JA-B000001.tar")) == 2
    assert (R_LARGE, "large/train-00002-of-00003.parquet") not in world.downloads


def test_validate(monkeypatch):
    """What an extent config may say. The caps are prefixes of sorted upstream listings, so only sources nothing else
    reads can be capped, emilia_yodas only at or past the 300 h step's tars, and the label roots share one root."""
    for inputs in ({}, {"reazon_large": 216, "galgame": 16}, {"reazon_large": 705, "galgame": 1, "emilia_nc": 66},
                   {"emilia_yodas": "300h"}, {"emilia_yodas": 9}, {"emilia_yodas": 29}):
        assert extent.validate(make_cfg(inputs=inputs)) == [], inputs
    assert extent.validate({"sources": ["galgame"]}) == ["the config has no extent block ({\"name\", \"root\", "
                                                         "\"inputs\"})"]

    def problems(inputs=None, sources=FULL_SOURCES, eval_sets=EVAL_SETS, **over) -> str:
        cfg = make_cfg(inputs=inputs, sources=sources, eval_sets=eval_sets)
        for k, v in over.items():
            if k.startswith("extent_"):
                cfg["extent"][k[len("extent_"):]] = v
            else:
                cfg[k] = v
        got = extent.validate(cfg)
        assert len(got) == 1, got
        return got[0]

    assert problems(sources=FULL_SOURCES + ["reazon_medium"]).startswith("reazon_medium: refused")
    assert problems(sources=FULL_SOURCES + ["cv"]).startswith("cv: refused")
    assert problems(sources=FULL_SOURCES + ["eval"]).startswith("eval: not a canonical source")
    assert "emilia_yodas is \"300h\" or an int 9..29, not 8" in problems({"emilia_yodas": 8})
    assert "not 30" in problems({"emilia_yodas": 30})
    assert "galgame is an int 1..115 (its upstream files) (tar 0 holds the galgame eval hold-out), not 0" in problems(
        {"galgame": 0})
    assert "reazon_large is an int 1..705 (its upstream files), not 706" in problems({"reazon_large": 706})
    assert "not True" in problems({"galgame": True}) and "not '300h'" in problems({"galgame": "300h"})
    assert "not '16'" in problems({"galgame": "16"})
    assert problems({"reazon_small": 3}).startswith("extent.inputs.reazon_small: not cappable")
    assert problems({"eval_emilia": 1}).startswith("extent.inputs.eval_emilia: not cappable")
    assert problems({"emilia_nc": 2}, sources=["reazon_small", "galgame"]) == (
        "extent.inputs.emilia_nc: the config names no emilia_nc (sources / eval_sets)")
    assert problems(extent_caps={}) == "extent.caps: unknown key (the keys are name, root, inputs)"
    assert problems(extent_name="full sub") == "extent.name 'full sub' is not a plain name ([A-Za-z0-9_.-])"
    assert problems(extent_inputs=[216]) == "extent.inputs is a {source: N} object, not [216]"
    # the root holds every label root, and is their common parent
    assert problems(teacher_root="teacher_out") == "teacher_root 'teacher_out' is not under extent.root labels/full"
    assert problems(parakeet_root="labels/fuller/parakeet_out").startswith("parakeet_root")
    assert problems(selection="selection/viability.parquet").startswith("selection")
    nested = {k: make_cfg()[k].replace("labels/full/", "labels/full/a/") for k in (
        "teacher_root", "second_root", "parakeet_root", "selection")}
    assert problems(**nested) == ("extent.root labels/full is not the common parent of teacher_root, second_root, "
                                  "selection, parakeet_root (labels/full/a is)")
    # every path in normal form: consumers join them into Hub paths ('labels/full/' would give 'labels/full//...')
    for root in ("/labels/full", "labels/../full", "labels/full/", "./labels/full", "labels//full", "labels/./full"):
        assert problems(extent_root=root) == (f"extent.root {root!r} is not a normalized relative repo path like "
                                              f"'labels/full'"), root
    assert problems(teacher_root="labels/full/./teacher_out") == (
        "teacher_root 'labels/full/./teacher_out' is not a normalized relative repo path")
    # the label box uploads write-once under labels/ only; label_runs/ is overwritable infra
    for root in ("labels", "label_runs/full", "extents/full", "labels/full/sub"):
        assert problems(extent_root=root) == f"extent.root {root} is not labels/<name>", root
    # without a parakeet_root only the three others count
    cfg = make_cfg()
    del cfg["parakeet_root"]
    assert extent.validate(cfg) == []

    # the names and the recipe are spelled out and well-formed: the trainer's defaults (the viability run's) are not
    # what the extent rebuilds and pulls, and a malformed config is a refusal, never a crash
    for k, v, want in (("eval_sets", 5, "eval_sets is a list of names, not 5"),
                       ("sources", [["galgame"]], "sources is a list of names, not [['galgame']]"),
                       ("sources", "galgame", "sources is a list of names, not 'galgame'"),
                       ("eval_sets", {"galgame": 1}, "eval_sets is a list of names, not {'galgame': 1}")):
        assert extent.validate(dict(make_cfg(), **{k: v})) == [want]
    for k in ("sources", "eval_sets"):
        cfg = make_cfg()
        del cfg[k]
        assert extent.validate(cfg) == [f"{k}: missing; an extent config spells it out (the trainer's default is "
                                        f"the viability run's)"]
    recipe = make_cfg()["selection_recipe"]
    assert problems(selection_recipe=None).startswith("selection_recipe is an object")
    assert problems(selection_recipe=dict(recipe, filter_eval_sets=None)) == (
        "selection_recipe.filter_eval_sets is a list of names, not None")
    # a partial second opinion (the viability run's galgame) cannot be pulled by stem: the extent judges everything
    assert problems(selection_recipe=dict(recipe, partial_second_opinion=["galgame"])).startswith(
        "selection_recipe.partial_second_opinion is ['galgame'], not []")
    del recipe["partial_second_opinion"]
    assert extent.validate(dict(make_cfg(), selection_recipe=recipe)) == []  # make_selection and launch read []


def test_within():
    """A label box labels one extent; every other config it serves must lie inside it (launch checks this before
    renting, and pull_plan against the record)."""
    full, sub = make_cfg(), make_cfg("sub3k", {"reazon_large": 216, "galgame": 16}, sources=[
        "reazon_small", "reazon_large", "emilia_yodas", "galgame"])
    assert extent.within(full, sub) == [] and extent.within(sub, sub) == []
    assert extent.within(sub, full) == [
        "galgame: needs every input, extent 'sub3k' has the first 16 inputs",
        "emilia_nc is not in extent 'sub3k'",
        "reazon_large: needs every input, extent 'sub3k' has the first 216 inputs"]
    h300, k9 = make_cfg(inputs={"emilia_yodas": "300h"}), make_cfg(inputs={"emilia_yodas": 9})
    assert extent.within(k9, h300) == [] and extent.within(full, k9) == []
    assert extent.within(h300, k9) == ["emilia_yodas: needs the first 9 inputs, extent 'full' has the 300 h step"]
    # eval_emilia alone needs only the 300 h step of emilia_yodas
    assert extent.within(h300, make_cfg(sources=["galgame"], eval_sets=["eval_emilia"])) == []
    assert extent.within(full, make_cfg(root="labels/other")) == ["extent.root 'labels/other' is not 'labels/full'"]
    # a source the outer extent rebuilds only as a dependency (reazon_small for reazon_large, the 300 h step for
    # eval_emilia) reaches far enough but was never labelled; the record says the same
    deps = make_cfg(sources=["reazon_large"], eval_sets=["eval_emilia"])
    uses = make_cfg(inputs={"emilia_yodas": "300h"}, sources=["reazon_small", "emilia_yodas"], eval_sets=[])
    assert extent.validate(deps) == extent.validate(uses) == []
    unlabelled = [f"{s}: extent 'full' rebuilds it only as a dependency and has no labels for it"
                  for s in ("reazon_small", "emilia_yodas")]
    assert extent.within(deps, uses) == unlabelled
    record = {"schema": 1, "canonical_version": 1, "name": "full", "root": "labels/full", "names": extent.names(deps),
              "inputs": {}, "sources": {s: {} for s in ("reazon_small", "reazon_large", "emilia_yodas", "eval_emilia")}}
    assert extent.record_problems(record, uses) == [p.replace("extent 'full'", "extent record 'full'")
                                                    for p in unlabelled]
    assert extent.record_problems(record, deps) == []


def test_closure_and_plan_steps():
    assert extent.closure(["reazon_large"]) == {"reazon_large", "reazon_small"}
    assert extent.closure(["eval_emilia"]) == {"eval_emilia", "emilia_yodas@300h"}
    assert extent.closure(["emilia_yodas"]) == {"emilia_yodas", "emilia_yodas@300h", "eval_emilia"}
    assert extent.closure(["galgame", "eval_jsut", "emilia_yodas@300h"]) == {"galgame", "eval_jsut",
                                                                              "emilia_yodas@300h"}
    keys = [s.key for s in extent.CANONICAL]
    assert len(set(keys)) == len(keys) and {s.source for s in extent.CANONICAL} == set(prep.ALL_SOURCES) - {
        "reazon_medium", "cv", "eval"}
    # the orders the ids depend on
    assert keys.index("reazon_small") < keys.index("reazon_large")
    assert keys.index("emilia_yodas@300h") < keys.index("eval_emilia") < keys.index("emilia_yodas")

    def plan(**kw) -> list[tuple[str, int | None]]:
        return [(s.key, cap) for s, cap in extent.plan_steps(make_cfg(**kw))]

    assert plan() == [(k, None) for k in keys]
    assert extent.plan_steps(make_cfg())[4][0].emilia_hours == 300.0
    assert plan(inputs={"reazon_large": 216, "galgame": 16}, sources=[
        "reazon_small", "reazon_large", "emilia_yodas", "galgame"]) == [
        ("reazon_small", None), ("eval_jsut", None), ("eval_cv8", None), ("eval_reazon", None),
        ("emilia_yodas@300h", None), ("eval_emilia", None), ("galgame", 16), ("emilia_yodas", None),
        ("reazon_large", 216)]
    assert plan(inputs={"emilia_yodas": 9}, sources=["emilia_yodas"], eval_sets=[]) == [
        ("emilia_yodas@300h", None), ("eval_emilia", None), ("emilia_yodas", 9)]
    assert plan(inputs={"emilia_yodas": "300h"}, sources=["emilia_yodas"], eval_sets=[]) == [
        ("emilia_yodas@300h", None)]
    assert plan(inputs={"emilia_yodas": "300h"}, sources=["emilia_yodas"], eval_sets=["eval_emilia"]) == [
        ("emilia_yodas@300h", None), ("eval_emilia", None)]
    assert plan(inputs={"reazon_large": 3}, sources=["reazon_large"], eval_sets=[]) == [
        ("reazon_small", None), ("reazon_large", 3)]


def test_build_record_round_trip(full, world, tmp_path, monkeypatch):
    """The record maps every upstream input to the stems it produced, zero-row inputs and the tar the 300 h step cut
    included, and it survives the label box's pruning: sizes then come from pruned.jsonl, the rest from the sidecars."""
    rec = full.record
    extent.write_record(tmp_path / "extent.json", rec)
    data = (tmp_path / "extent.json").read_bytes()
    assert extent.load_record(tmp_path / "extent.json") == rec
    extent.write_record(tmp_path / "again.json", extent.build_record(
        full.root, full.cfg, run_ids=["20260925T000000Z-1"], kitsune_sha="0" * 40))
    assert (tmp_path / "again.json").read_bytes() == data  # a re-run finalize writes the same file
    assert {k: rec[k] for k in ("schema", "name", "root", "canonical_version", "kitsune_sha", "run_ids", "inputs",
                                "min_inputs")} == dict(
        schema=1, name="full", root="labels/full", canonical_version=1, kitsune_sha="0" * 40,
        run_ids=["20260925T000000Z-1"], inputs={}, min_inputs={"emilia_yodas": 2})
    assert rec["steps"] == [s.key for s in extent.CANONICAL] and rec["names"] == extent.names(full.cfg)
    assert rec["tools"] == extent.tools() and set(rec["tools"]) == {"soundfile", "libsndfile", "pyarrow"}
    repos = {prep.source_repo(s) for s in rec["sources"]}
    assert rec["revisions"] == {r: prep.REVISIONS[r] for r in repos} and len(repos) == 8
    assert list(rec["sources"]) == list(dict.fromkeys(s.source for s in extent.CANONICAL))

    def by_input(source: str) -> list:
        return [(i["ordinal"], i["input"], [(st["stem"], st["step"]) for st in i["stems"]])
                for i in rec["sources"][source]["inputs"]]

    assert by_input("emilia_yodas") == [
        (0, "JA/JA-B000000.tar", [("train-00000", "emilia_yodas@300h")]),
        (1, "JA/JA-B000001.tar", [("train-00001", "emilia_yodas@300h"), ("train-00002", "emilia_yodas")]),  # the cut
        (2, "JA/JA-B000002.tar", [("train-00003", "emilia_yodas")])]
    assert by_input("reazon_large")[1] == (1, "large/train-00001-of-00003.parquet", [])  # every row a duplicate
    assert by_input("emilia_nc")[1] == (1, "JA-B000001_standard.tar.gz", [])  # English only
    assert by_input("galgame")[0] == (0, "data/galgame-000000.tar", [("eval-00000", "galgame"),
                                                                      ("train-00000", "galgame")])
    assert by_input("eval_emilia") == [(0, prep.EMILIA_EVAL_TAR, [("eval-00000", "eval_emilia")])]
    manifest = {sh.path: sh for sh in read_manifest(full.root)}
    for source, src in rec["sources"].items():
        repo = prep.source_repo(source)
        listed = [f for f in world.upstream[repo] if f != prep.EMILIA_EVAL_TAR]  # eval_emilia lists nothing
        assert src["repo"] == repo and src["n_listed"] == (None if source == "eval_emilia" else len(listed))
        for inp in src["inputs"]:
            assert inp["bytes"] == world.size(repo, inp["input"])
            for st in inp["stems"]:
                path = f"shards/{source}/{st['stem']}.parquet"
                ids = full.layout[path]
                assert st["ids_sha256"] == extent.ids_sha256(ids) and st["rows"] == len(ids)
                assert st["shard_bytes"] == (full.root / path).stat().st_size
                assert (st["split"], st["hours"]) == (manifest[path].split, manifest[path].hours)
        stems = [st for i in src["inputs"] for st in i["stems"]]
        assert (src["rows"], src["hours"], src["bytes"]) == (sum(st["rows"] for st in stems), sum(
            st["hours"] for st in stems), sum(i["bytes"] for i in src["inputs"]))
    assert extent.ids_sha256(["reazon_small/a", "reazon_small/b"]) == (
        "6b1db3453bb1b6c8d648972b295904107e0d593a887860b3b993ea72c7bb4f22")  # the join key is fixed across boxes

    # pruned: the audio is gone, the sidecars and pruned.jsonl remain; one size was lost (a torn line)
    root = tmp_path / "pruned"
    shutil.copytree(full.root, root)
    lines = []
    for sh in read_manifest(root):
        if sh.split == "train":
            lines.append(json.dumps({"path": sh.path, "bytes": (root / sh.path).stat().st_size}))
            (root / sh.path).unlink()
    lost = json.loads(lines[0])["path"]
    (root / extent.PRUNED_FILE).write_text("\n".join(['{"path": ', *lines[1:]]) + "\n", encoding="utf-8")
    pruned = extent.build_record(root, full.cfg, run_ids=rec["run_ids"], kitsune_sha=rec["kitsune_sha"])
    source, stem = lost.split("/")[1], Path(lost).stem
    for inp in pruned["sources"][source]["inputs"]:
        for st in inp["stems"]:
            if st["stem"] == stem:
                assert st["shard_bytes"] is None
                st["shard_bytes"] = (full.root / lost).stat().st_size
    assert pruned == rec

    # only a finished ingest of this very plan has a record
    prog = extent.read_progress(root)
    extent.write_progress(root, dict(prog, completed=[k for k in prog["completed"] if k != "galgame"]))
    with pytest.raises(ValueError, match=r"has not completed \['galgame'\]"):
        extent.build_record(root, full.cfg, run_ids=[], kitsune_sha="")
    with pytest.raises(ValueError, match="not an ingest of this extent's plan"):
        extent.build_record(full.root, make_cfg(inputs={"galgame": 2}), run_ids=[], kitsune_sha="")
    with pytest.raises(ValueError, match="not a valid extent config"):
        extent.build_record(full.root, make_cfg(inputs={"galgame": 0}), run_ids=[], kitsune_sha="")
    # a source re-ingested outside --extent-config after its step completed: the progress still says complete, but
    # the record must not claim inputs the data root no longer holds (subset_stems and pull_plan would shrink)
    forced = tmp_path / "forced"
    shutil.copytree(full.root, forced)
    run01(monkeypatch, forced, "--sources", "galgame", "--force", "--galgame-shards", "1")
    assert extent.read_progress(forced) == extent.read_progress(full.root)
    with pytest.raises(ValueError, match=r"galgame: the ingest recorded 1 of the 3 inputs this extent reads"):
        extent.build_record(forced, full.cfg, run_ids=[], kitsune_sha="")


def test_subset_stems(full):
    """Which recorded stems a config takes: a capped source those of its first N inputs, emilia_yodas "300h" the 300 h
    step's, K those plus the rest step's of the first K tars, anything uncapped all of them."""
    rec = full.record
    everything = {s: stems_of(full.layout, s) for s in extent.names(full.cfg)}
    assert extent.subset_stems(rec, full.cfg) == everything

    def take(inputs, sources=FULL_SOURCES, eval_sets=EVAL_SETS) -> dict:
        return extent.subset_stems(rec, make_cfg("sub", inputs, sources, eval_sets))

    got = take({"reazon_large": 1, "galgame": 2, "emilia_nc": 1})
    assert got == dict(everything, reazon_large={"train-00000"}, emilia_nc={"train-00000"},
                       galgame={"eval-00000", "train-00000", "train-00001", "train-00002"})
    assert take({"reazon_large": 2})["reazon_large"] == {"train-00000"}  # input 1 kept no row
    assert take({"emilia_yodas": "300h"})["emilia_yodas"] == {"train-00000", "train-00001"}
    assert take({"emilia_yodas": 2})["emilia_yodas"] == {"train-00000", "train-00001", "train-00002"}
    assert take({"emilia_yodas": 3})["emilia_yodas"] == everything["emilia_yodas"]
    assert take({"galgame": 1}, sources=["reazon_large"], eval_sets=["galgame"]) == {
        "reazon_large": everything["reazon_large"], "galgame": {"eval-00000", "train-00000"}}


def test_pull_plan(full):
    """The A100 pulls a capped source's label files one by one and an uncapped one by directory, second opinions only
    where the selection reads them, and never the Parakeet targets; together that is exactly the files it needs."""
    rec = full.record
    judged = {"reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame", "eval_emilia"}
    listing = ["labels/full/teacher_out/meta.json", "labels/full/second_out/meta.json",
               "labels/full/parakeet_out/meta.json", "labels/full/selections/full.parquet",
               "labels/full/selections/sub.parquet", "labels/full/extent.json", "labels/full/COMPLETE.json",
               "students/b20x2560-d4/config.json", "students/b20x2560-d4/model.safetensors",
               "teacher_out/galgame/train-00000.npz"]  # a laptop root: never pulled
    for p in full.layout:
        _, source, name = p.split("/")
        stem = name[:-len(".parquet")]
        listing += [f"labels/full/{r}/{source}/{stem}.{e}" for r in ("teacher_out", "parakeet_out")
                    for e in ("npz", "jsonl")]
        listing += [f"labels/full/second_out/{source}/{stem}.jsonl"] if source in judged else []
    cfg = make_cfg("sub", {"reazon_large": 1, "galgame": 2})
    p = extent.pull_plan(cfg, rec, listing)
    assert p["problems"] == []
    files = (("teacher_out", "npz"), ("teacher_out", "jsonl"), ("second_out", "jsonl"))
    assert sorted(p["explicit"]) == sorted(
        [f"labels/full/{r}/galgame/{s}.{e}" for s in ("eval-00000", "train-00000", "train-00001", "train-00002")
         for r, e in files] + [f"labels/full/{r}/reazon_large/train-00000.{e}" for r, e in files])
    assert "labels/full/teacher_out/eval_jsut/*" in p["dir_patterns"]
    assert "labels/full/second_out/eval_emilia/*" in p["dir_patterns"]  # a label-filtered hold-out
    assert "labels/full/second_out/eval_jsut/*" not in p["dir_patterns"]  # a gate set has no second opinion
    assert not any("reazon_large" in d or "galgame" in d for d in p["dir_patterns"])
    assert not any("parakeet_out" in f for f in p["dir_patterns"] + p["explicit"] + p["required"])
    pulled = {f for f in listing if any(fnmatch(f, pat) for pat in p["dir_patterns"])} | set(p["explicit"])
    assert pulled == set(p["required"]) | {"students/b20x2560-d4/config.json", "students/b20x2560-d4/model.safetensors"}
    assert "labels/full/teacher_out/reazon_large/train-00001.npz" not in pulled
    assert extent.pull_plan(full.cfg, rec, listing)["problems"] == []

    # what refuses
    gone = "labels/full/teacher_out/galgame/train-00001.npz"
    assert extent.pull_plan(cfg, rec, [f for f in listing if f != gone])["problems"] == [
        f"1 of the {len(p['required'])} files the extent needs are not in the data repo, e.g. ['{gone}']"]
    smaller = dict(rec, inputs={"galgame": 1})
    assert extent.pull_plan(cfg, smaller, listing)["problems"] == [
        "galgame: needs the first 2 inputs, extent record 'full' has the first 1 inputs"]
    no_nc = dict(rec, sources={s: v for s, v in rec["sources"].items() if s != "emilia_nc"})
    assert "emilia_nc is not in extent record 'full'" in extent.pull_plan(full.cfg, no_nc, listing)["problems"]
    other = make_cfg("sub", {"galgame": 2}, root="labels/other")
    assert extent.pull_plan(other, rec, listing)["problems"][0] == (
        "extent record is for root 'labels/full', the config's is 'labels/other'")
    assert extent.pull_plan(cfg, dict(rec, canonical_version=2), listing)["problems"][0].startswith(
        "extent record from canonical sequence v2")
    bad = extent.pull_plan(make_cfg(inputs={"galgame": 0}), rec, listing)
    assert bad["explicit"] == bad["required"] == [] and bad["problems"] == extent.validate(make_cfg(inputs={
        "galgame": 0}))


def test_sizing_from_a_toy_record():
    """The A100's disk and rebuild timeout from the record: the upstream bytes the plan downloads (the cut Emilia tar
    twice), the rebuilt shards (dependencies too; an unknown size at the source's upstream bytes per hour), the audio
    the trainer copies, the label files, then the margin and the 50 GB steps."""
    def stem(name, step, hours, size):
        return {"stem": name, "split": "train", "step": step, "rows": 1, "hours": hours, "ids_sha256": "",
                "shard_bytes": size}

    def source(inputs):
        stems = [st for i in inputs for st in i["stems"]]
        return {"repo": "r", "n_listed": len(inputs), "rows": len(stems), "hours": sum(st["hours"] for st in stems),
                "bytes": sum(i["bytes"] for i in inputs),
                "inputs": [dict(i, input=f"f{n}", ordinal=n) for n, i in enumerate(inputs)]}

    rec = {"schema": 1, "sources": {
        "reazon_small": source([{"bytes": 2e9, "stems": [stem("train-00000", "reazon_small", 20.0, 1.9e9)]},
                                {"bytes": 1e9, "stems": [stem("train-00001", "reazon_small", 10.0, None)]}]),
        "reazon_large": source([{"bytes": 20e9, "stems": [stem("train-00000", "reazon_large", 200.0, 19e9)]},
                                {"bytes": 20e9, "stems": []},
                                {"bytes": 20e9, "stems": [stem("train-00001", "reazon_large", 400.0, 38e9)]}]),
        "emilia_yodas": source([{"bytes": 1e9, "stems": [stem(f"train-{n:05d}", "emilia_yodas@300h" if n < 9 else
                                                              "emilia_yodas", 30.0, 1e9)]} for n in range(12)]),
        "eval_emilia": source([{"bytes": 0.4e9, "stems": [stem("eval-00000", "eval_emilia", 2.0, 0.03e9)]}])}}
    cfg = make_cfg("sub", {"reazon_large": 2}, sources=["reazon_large"], eval_sets=[])
    size = extent.sizing(rec, cfg)
    # reazon_small (all, a dependency) + reazon_large's first 2 inputs; small's unknown shard: 10 h x 3 GB / 30 h
    assert size["down_gb"] == pytest.approx(3 + 40) and size["shard_gb"] == pytest.approx(1.9 + 1 + 19)
    assert size["sel_gb"] == pytest.approx(19) and size["hours"] == 200 and size["labels_gb"] == pytest.approx(0.32)
    assert size["disk_gb"] == 200  # 1.1 x (127 + 21.9 + 19 + 0.32) = 185.0
    assert size["rebuild_timeout_min"] == 57  # 30 + 1.5 x 43e9 B / 40e6 B/s / 60 = 56.9
    # the selection's kept hours at the source's shard GB per hour (57 GB / 600 h); make_selection's keys work too
    kept = extent.sizing(rec, cfg, {"reazon_large/train": 500.0, "reazon_large/eval": 100.0})
    assert kept["sel_gb"] == pytest.approx(57) and kept["disk_gb"] == 250  # 1.1 x (127 + 21.9 + 57 + 0.32) = 226.8
    # ... also as the selection metadata stores them
    assert extent.sizing(rec, cfg, {"reazon_large/train": {"utts": 9, "hours": 500.0},
                                    "reazon_large/eval": {"utts": 2, "hours": 100.0}}) == kept
    # Emilia: the 300 h step reads tars 0-8, eval_emilia its tar, the rest tars 8..K-1 (tar 8 again)
    emilia = make_cfg("e", {"emilia_yodas": 11}, sources=["emilia_yodas"], eval_sets=[])
    assert extent.sizing(rec, emilia)["down_gb"] == pytest.approx(9 + 0.4 + 3)
    assert extent.sizing(rec, emilia)["shard_gb"] == pytest.approx(11 + 0.03)
    h300 = make_cfg("e", {"emilia_yodas": "300h"}, sources=["emilia_yodas"], eval_sets=[])
    assert extent.sizing(rec, h300)["down_gb"] == pytest.approx(9) and extent.sizing(rec, h300)["hours"] == 270
    assert extent.sizing(rec, make_cfg("e", sources=["emilia_yodas"], eval_sets=[]))["down_gb"] == pytest.approx(
        9 + 0.4 + 4)
    # the full extent's ~570 GB upstream: a ~6.4 h rebuild cap on the A100
    assert extent.REBUILD_BASE_MIN + extent.REBUILD_SLACK * 570e9 / extent.REBUILD_BYTES_PER_S / 60 == pytest.approx(
        386.25)


def test_extent_config_run_writes_progress_and_a_rerun_is_a_noop(full, world, tmp_path, monkeypatch, capsys):
    """01 records every finished step, so a restarted ingest lane or a retried A100 rebuild skips them: no listing, no
    download, nothing on disk changes. Progress recorded for another plan is dropped, and running every step again is
    still a no-op on what is already on disk."""
    prog = extent.read_progress(full.root)
    keys = [s.key for s in extent.CANONICAL]
    repos = {s.source: prep.source_repo(s.source) for s in extent.CANONICAL}
    assert prog == dict(canonical_version=1, config=str(full.config), extent=full.cfg["extent"],
                        steps=[[k, None] for k in keys], completed=keys, repos=repos,
                        revisions={r: prep.REVISIONS[r] for r in repos.values()}, tools=extent.tools())
    root = tmp_path / "data"
    shutil.copytree(full.root, root)

    def disk() -> dict:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
                if p.is_file() and p.name not in (".ingest.lock", extent.PROGRESS_FILE)}

    before = disk()
    run01(monkeypatch, root, "--extent-config", str(full.config))
    assert world.listings == world.downloads == [] and disk() == before
    assert extent.read_progress(root) == prog
    extent.write_progress(root, dict(prog, steps=prog["steps"][:-1] + [["reazon_large", 1]]))
    run01(monkeypatch, root, "--extent-config", str(full.config))
    assert "records another plan" in capsys.readouterr().out
    assert world.downloads == [] and len(world.listings) == len(keys) - 1  # eval_emilia reads one fixed tar
    assert disk() == before and extent.read_progress(root) == prog


def test_extent_config_refuses_other_flags_and_invalid_configs(full, world, tmp_path, monkeypatch, capsys):
    """The config alone defines what is ingested: a per-source flag next to it would give other ids or stems (argparse
    error, 2), and a config that is no valid extent is a refusal (3), which bootstrap's retry() does not repeat."""
    for i, flags in enumerate((["--sources", "galgame"], ["--max-inputs", "galgame=1"], ["--galgame-shards", "2"],
                               ["--emilia-hours", "300"], ["--emilia-nc-hours", "5"], ["--limit-rows", "3"],
                               ["--force"])):
        with pytest.raises(SystemExit) as e:
            run01(monkeypatch, tmp_path / f"flags{i}", "--extent-config", str(full.config), *flags)
        assert e.value.code == 2 and f"drop {flags[0]}" in capsys.readouterr().err
    bad = write_cfg(tmp_path / "bad.json", make_cfg(sources=FULL_SOURCES + ["reazon_medium"]))
    malformed = write_cfg(tmp_path / "malformed.json", dict(make_cfg(), eval_sets=5))  # a refusal, not a traceback
    for i, config in enumerate((bad, malformed, tmp_path / "missing.json")):
        with pytest.raises(SystemExit) as e:
            run01(monkeypatch, tmp_path / f"bad{i}", "--extent-config", str(config))
        assert e.value.code == prep.REFUSED_EXIT == 3
        assert f"{config}: not an extent config: " in capsys.readouterr().err
        assert not (tmp_path / f"bad{i}" / "manifest.jsonl").exists()
    assert "reazon_medium: refused" in extent.validate(json.loads(bad.read_text(encoding="utf-8")))[0]
    assert world.listings == []
    # a relative CFG not found in the working directory is the checkout's, as for make_selection.py --config
    checkout = tmp_path / "checkout"
    (checkout / "configs").mkdir(parents=True)
    write_cfg(checkout / "configs" / "x.json", make_cfg())
    monkeypatch.setattr(prep, "ROOT", checkout)
    monkeypatch.chdir(tmp_path)
    assert prep.load_extent_config(Path("configs/x.json")) == make_cfg()


def test_extent_config_gives_every_step_the_whisper_dir_hold_file_and_heartbeat(tmp_path, monkeypatch):
    """The label box's ingest lane is `01 --extent-config CFG --whisper-dir DIR --hold-file F --heartbeat H`, and every
    step must get all three: without a reazon file's whisper part 02b stops the box (65) once the step is complete,
    without the hold the unlabelled backlog fills the disk, and without the heartbeat the lane is killed as hung. The
    janitor holds the ingest before the first download and again before reazon_large's."""
    up: dict[str, dict[str, Path]] = {}
    for repo, name, rids in ((R_SMALL, "small/train-00000-of-00001.parquet", ["s0", "s1"]),
                             (R_LARGE, "large/train-00000-of-00002.parquet", ["s1", "l0"]),
                             (R_LARGE, "large/train-00001-of-00002.parquet", ["l1"])):
        path = tmp_path / "up" / name.replace("/", "_")
        path.parent.mkdir(exist_ok=True)
        make_reazon_parquet(path, rids, whisper=[[50364, i] for i in range(len(rids))])
        up.setdefault(repo, {})[name] = path
    patch_world(monkeypatch, up, tmp_path / "dl")
    state, wdir = tmp_path / "state", tmp_path / "whisper"
    hold, hb = state / "ingest.hold", state / "hb" / "ingest"
    state.mkdir()
    hold.touch()
    real_download, events = prep.hf_hub_download, []

    def download(repo, filename, **kw):
        events.append(("download", filename, hold.exists(), hb.is_file()))
        hb.unlink(missing_ok=True)  # the next event must see it touched again
        if repo == R_SMALL:
            hold.touch()  # the backlog grew: reazon_large's first download must wait
        return real_download(repo, filename, **kw)

    def sleep(s):
        events.append(("sleep", s, hb.is_file()))
        hb.unlink(missing_ok=True)
        if sum(e[0] == "sleep" for e in events) % 2 == 0:
            hold.unlink()  # released after two polls
    monkeypatch.setattr(prep, "hf_hub_download", download)
    monkeypatch.setattr(prep, "time", SimpleNamespace(sleep=sleep))
    cfg = write_cfg(tmp_path / "reazon.json", make_cfg(sources=["reazon_small", "reazon_large"], eval_sets=[]))
    root = tmp_path / "data"
    run01(monkeypatch, root, "--extent-config", str(cfg), "--whisper-dir", str(wdir), "--hold-file", str(hold),
          "--heartbeat", str(hb))
    held = [("sleep", prep.HOLD_POLL_S, True)] * 2
    assert events == [*held, ("download", "small/train-00000-of-00001.parquet", False, True),
                      *held, ("download", "large/train-00000-of-00002.parquet", False, True),
                      ("download", "large/train-00001-of-00002.parquet", False, True)]
    for repo, source in ((R_SMALL, "reazon_small"), (R_LARGE, "reazon_large")):
        for name in up[repo]:  # a part for every input of both steps, the whole file's names (dups included)
            part = wdir / source / Path(name).name
            assert [r["name"] for r in iter_rows(part, ["name"])] == [
                r["name"] for r in iter_rows(up[repo][name], ["name"])], part
    assert layout(root) == {"shards/reazon_small/train-00000.parquet": ["reazon_small/s0", "reazon_small/s1"],
                            "shards/reazon_large/train-00000.parquet": ["reazon_large/l0"],
                            "shards/reazon_large/train-00001.parquet": ["reazon_large/l1"]}
    assert extent.read_progress(root)["completed"] == ["reazon_small", "reazon_large"]


def test_a_killed_extent_run_resumes_to_the_full_runs_stems(full, world, tmp_path, monkeypatch):
    """The ingest lane is killed inside a step (here galgame, mid tar 1): the steps before it stay completed, and the
    restarted run gives the full run's stems and the same record."""
    real, calls = prep.Ingest.add, [0]

    def add(self, *a, **kw):
        if self.source == "galgame":
            calls[0] += 1
            if calls[0] > 8:
                raise RuntimeError("simulated kill")
        return real(self, *a, **kw)

    monkeypatch.setattr(prep.Ingest, "add", add)
    root = tmp_path / "data"
    with pytest.raises(RuntimeError, match="simulated kill"):
        run01(monkeypatch, root, "--extent-config", str(full.config))
    keys = [s.key for s in extent.CANONICAL]
    assert extent.read_progress(root)["completed"] == keys[:keys.index("galgame")]
    # the size of the tar in flight is saved at its download, not only when it is finished: a resume that never
    # downloads it again (eval_emilia's early return) would otherwise leave the record without it
    assert load_progress(root, "galgame")["input_bytes"] == {
        f: world.size(prep.GALGAME_REPO, f) for f in ("data/galgame-000000.tar", "data/galgame-000001.tar")}
    monkeypatch.setattr(prep.Ingest, "add", real)
    run01(monkeypatch, root, "--extent-config", str(full.config))
    assert layout(root) == full.layout
    assert extent.build_record(root, full.cfg, run_ids=full.record["run_ids"], kitsune_sha="0" * 40) == full.record
