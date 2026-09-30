"""kitsune/devslice.py: the full selections' dev draw (shard pairs with buffers, whole Emilia videos), the files a
selection needs next to it, and the sidecar check launch runs. Pure functions on synthetic inputs; CPU only.
"""
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, load_script  # noqa: E402

from kitsune import devslice, fullrun, prereg  # noqa: E402
from kitsune.store import ROWS_PER_SHARD  # noqa: E402


def stems(n: int, start: int = 0) -> list[str]:
    return [f"train-{i:05d}" for i in range(start, start + n)]


def test_constants_and_reexports():
    assert devslice.DEV_VIDEO_ROWS == 2 * ROWS_PER_SHARD == 4096  # "about two shards' worth"
    assert (devslice.DEV_PAIR_SHARDS, devslice.DEV_BUFFER_SHARDS) == (2, 1)
    assert set(devslice.DEV_METHOD) == set(fullrun.FULL_DATA["sources"])
    assert {s for s, m in devslice.DEV_METHOD.items() if m == "videos"} == {"emilia_yodas", "emilia_nc"}
    assert set(prereg.STUDY_REASONS) <= set(devslice.FULL_REASONS)
    assert set(devslice.FULL_REASONS) - set(prereg.STUDY_REASONS) == {"dev_buffer", "dev_dup"}
    # the shared names are fullrun's objects, not copies
    for name in ("DEV_SPLIT", "SPLITS", "DEV_RULE", "FULL_STUDY", "FULL_DIR", "FULL_SELECTION", "SMOKE_SELECTION",
                 "FROZEN_MANIFEST", "FROZEN_MANIFEST_SHA256", "seeded_subset", "dev_pick", "shard_split"):
        assert getattr(devslice, name) is getattr(fullrun, name), name
    assert (devslice.SCORED_PER_SOURCE, devslice.SCORED_SEED) == (600, 1234)
    assert devslice.GREEDY_N == 500 and prereg.SELECTION_SEED == 1234  # contract 3 rule 10: the study's subsets


def test_import_is_stdlib_only():
    """launch-side code imports devslice with the laptop's or the box's Python: no numpy/pandas at import."""
    code = ("import sys; sys.path.insert(0, %r); import kitsune.devslice; "
            "print(sorted(m for m in ('numpy', 'pandas', 'pyarrow', 'torch') if m in sys.modules))" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_pick_shard_pair_is_seeded_bounded_and_never_first():
    """k = 1 + rng.integers(n - 3): the pair is two adjacent stems, never train-00000, with a buffer stem on each
    side; deterministic in (seed, source, the stem set), whatever order the stems come in."""
    L = stems(12)
    pair, buffers = devslice.pick_shard_pair(L, 1234, "reazon_small")
    assert devslice.pick_shard_pair(list(reversed(L)), 1234, "reazon_small") == (pair, buffers)
    k = L.index(pair[0])
    assert pair == L[k:k + 2] and buffers == [L[k - 1], L[k + 2]] and k >= 1
    rng = np.random.default_rng([1234, __import__("zlib").crc32(b"full:dev:reazon_small")])
    assert k == 1 + int(rng.integers(len(L) - 3))  # the rule, spelled out
    ks = Counter()
    for seed in range(400):
        p, b = devslice.pick_shard_pair(L, seed, "galgame")
        k = L.index(p[0])
        ks[k] += 1
        assert "train-00000" not in p and len(b) == 2 and b[0] == L[k - 1] and b[1] == L[k + 2]
    assert set(ks) == set(range(1, len(L) - 2))  # every allowed start is reachable, and only those
    # the source is part of the rng tag
    assert any(devslice.pick_shard_pair(L, s, "galgame") != devslice.pick_shard_pair(L, s, "reazon_large")
               for s in range(5))
    # four stems: the only choice
    assert devslice.pick_shard_pair(stems(4), 7, "galgame") == (["train-00001", "train-00002"],
                                                                ["train-00000", "train-00003"])
    # a gap in the numbering is fine (stems are sorted names); train-00000 absent: still never the first stem
    p, b = devslice.pick_shard_pair(stems(5, start=3), 1, "reazon_large")
    assert "train-00003" not in p
    for bad, match in ((stems(3), "at least 4"), (stems(4) + ["train-00001"], "twice"),
                       (stems(4) + ["eval-00000"], "not train stems")):
        with pytest.raises(ValueError, match=match):
            devslice.pick_shard_pair(bad, 1, "galgame")


def test_pick_videos_takes_whole_videos_to_the_row_target():
    rows = {f"v{i:03d}": (i % 7) + 1 for i in range(60)}
    got = devslice.pick_videos(rows, 1234, "emilia_yodas", 20)
    assert got == sorted(got) and sum(rows[v] for v in got) >= 20
    # it stops at the first video that reaches the target: without the last one taken it is short
    order = [sorted(rows)[i] for i in np.random.default_rng(
        [1234, __import__("zlib").crc32(b"full:dev:emilia_yodas")]).permutation(len(rows))]
    taken = order[:len(got)]
    assert sorted(taken) == got and sum(rows[v] for v in taken[:-1]) < 20
    # depends only on the map: not on its order
    assert devslice.pick_videos(dict(reversed(list(rows.items()))), 1234, "emilia_yodas", 20) == got
    assert devslice.pick_videos(rows, 1235, "emilia_yodas", 20) != got
    assert devslice.pick_videos(rows, 1234, "emilia_nc", 20) != got
    with pytest.raises(ValueError, match="every one"):
        devslice.pick_videos(rows, 1, "emilia_nc", sum(rows.values()))
    with pytest.raises(ValueError, match="no train rows"):
        devslice.pick_videos({}, 1, "emilia_nc", 5)


def test_pick_videos_default_target_is_read_at_call_time(monkeypatch):
    rows = {f"v{i}": 3 for i in range(10)}
    monkeypatch.setattr(devslice, "DEV_VIDEO_ROWS", 6)
    assert len(devslice.pick_videos(rows, 1, "emilia_yodas")) == 2


def test_emilia_video_equals_01s():
    """The canonical ingest's _emilia_video (scripts/01_prepare_data.py), which devslice duplicates."""
    prep = load_script("01_prepare_data")
    keys = ["emilia_yodas/JA_Y0mvoyahGYA_W000002", "JA_a_b_c-d_W000123", "emilia_nc/JA_B00000_S00000_W000000",
            "eval_emilia/JA_8OD-KK6anX8_W000010", "reazon_small/000/000734dcb35d6.flac", "JA_noW", "W_only"]
    for k in keys:
        assert devslice.emilia_video(k) == prep._emilia_video(k), k
    assert devslice.emilia_video("emilia_nc/JA_B00000_S00000_W000000") == "B00000_S00000"
    assert devslice.emilia_video("emilia_yodas/JA_a_b_c-d_W000123") == "a_b_c-d"


def table():
    """A toy selection: reazon_small over 8 stems (3 rows each), galgame over 6 train stems + an eval stem, and
    emilia_yodas with 10 videos of 3 rows running across 4-row shards (so a video spans two shards)."""
    rows = [(f"reazon_small/{i:03d}.flac", "reazon_small", f"train-{i // 3:05d}", True) for i in range(24)]
    rows += [(f"galgame/g{i:03d}", "galgame", f"train-{i // 3:05d}", True) for i in range(18)]
    rows += [(f"galgame/e{i:03d}", "galgame", "eval-00000", False) for i in range(4)]
    rows += [(f"emilia_yodas/JA_v{i // 3:02d}_x_W{i % 3:06d}", "emilia_yodas", f"train-{i // 4:05d}", True)
             for i in range(30)]
    ids, src, stem, train = zip(*rows)
    return np.array(ids, dtype=object), np.array(src, dtype=object), np.array(stem, dtype=object), np.array(train)


def test_draw_marks_whole_shards_and_whole_videos():
    ids, src, stem, train = table()
    sources = ["reazon_small", "galgame", "emilia_yodas"]
    extent = {"reazon_small": set(stem[src == "reazon_small"]), "galgame": set(stem[src == "galgame"]),
              "emilia_yodas": set(stem[src == "emilia_yodas"])}
    d = devslice.draw(ids, src, stem, train, sources, 1234, stems=extent, video_rows=5)
    assert devslice.draw(ids, src, stem, train, sources, 1234, stems=None, video_rows=5)["by_source"] == d["by_source"]
    for s in ("reazon_small", "galgame"):
        b = d["by_source"][s]
        pair, bufs = devslice.pick_shard_pair(sorted(x for x in extent[s] if x.startswith("train-")), 1234, s)
        assert b == {"method": "shards", "dev_stems": pair, "buffer_stems": bufs}
        mine = src == s
        assert set(np.flatnonzero(d["dev"] & mine)) == set(np.flatnonzero(mine & train & np.isin(stem, pair)))
        assert set(np.flatnonzero(d["buffer"] & mine)) == set(np.flatnonzero(mine & train & np.isin(stem, bufs)))
    assert not (d["dev"] & ~train).any() and not (d["dev"] & d["buffer"]).any()  # the eval stem never
    e = d["by_source"]["emilia_yodas"]
    assert e["method"] == "videos" and e["video_rows"] == 5 and len(e["videos"]) == 2  # 3 + 3 rows >= 5
    video = np.array([devslice.emilia_video(i) for i in ids])
    dev_videos = set(video[d["dev"] & (src == "emilia_yodas")])
    assert dev_videos == set(e["videos"])
    for v in dev_videos:  # a video is dev whole, across its shards
        assert d["dev"][video == v].all()
    assert not d["buffer"][src == "emilia_yodas"].any()
    with pytest.raises(ValueError, match="no dev-slice method"):
        devslice.draw(ids, src, stem, train, ["eval_jsut"], 1)
    few = {"reazon_small": {"train-00000", "train-00001", "train-00002"}}
    with pytest.raises(ValueError, match="at least 4"):
        devslice.draw(ids, src, stem, train, ["reazon_small"], 1, stems=few)


def test_draw_problems_come_from_the_stems_alone(monkeypatch):
    """What make_selection refuses before it reads a row: a shards source with fewer train stems than the pair and its
    buffers (eval stems do not count; the bound is read at call time), a source without a method. Videos sources are
    judged on their rows by the build."""
    ok = {"reazon_small": set(stems(4)), "galgame": set(stems(4)) | {"eval-00000"}, "emilia_yodas": {"train-00000"}}
    assert devslice.draw_problems(ok, ["reazon_small", "galgame", "emilia_yodas"]) == []
    few = dict(ok, galgame=set(stems(3)) | {"eval-00000"})
    assert devslice.draw_problems(few, ["reazon_small", "galgame"]) == [
        "galgame: 3 train stems in the extent, the dev pair and its buffers need at least 4"]
    assert devslice.draw_problems({}, ["reazon_large"]) == [
        "reazon_large: 0 train stems in the extent, the dev pair and its buffers need at least 4"]
    assert "no dev-slice method" in devslice.draw_problems(ok, ["eval_cv8"])[0]
    monkeypatch.setattr(devslice, "DEV_PAIR_SHARDS", 3)
    assert "at least 5" in devslice.draw_problems(ok, ["reazon_small"])[0]
    # the same bound as pick_shard_pair's
    with pytest.raises(ValueError, match="at least 5"):
        devslice.pick_shard_pair(stems(4), 1, "reazon_small")


def test_selection_files():
    full = fullrun.FULL_DATA
    assert devslice.selection_files(full) == ["labels/full/selections/full_study/full.json", fullrun.FROZEN_MANIFEST]
    assert devslice.selection_files(fullrun.SMOKE_DATA) == ["labels/full/selections/full_study/smoke.json",
                                                            fullrun.FROZEN_MANIFEST]
    study = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
    assert devslice.selection_files(study) == list(prereg.study_files(study["selection"]))
    plain = {"selection": "selection/viability.parquet", "selection_recipe": {"agree_max": 0.5}}
    assert devslice.selection_files(plain) == [] == devslice.selection_files({})
    assert devslice.sidecar_path("labels/full/selections/full_study/full.parquet") == \
        "labels/full/selections/full_study/full.json"


def good_sidecar() -> dict:
    return {"kind": "full_study", "schema": 1,
            "selection": {"path": fullrun.FULL_SELECTION, "sha256": "a" * 64},
            "manifest": {"path": fullrun.FROZEN_MANIFEST, "sha256": fullrun.FROZEN_MANIFEST_SHA256,
                         "eval_ids_equal": True},
            "k5": {"not_in_parakeet": 3, "candidates": 10000, "frac": 0.0003, "max_frac": 0.001, "ok": True},
            "sources": ["reazon_small", "galgame"],
            "dev": {"by_source": {"reazon_small": {"kept_rows": 5}, "galgame": {"kept_rows": 2}}}}


def test_sidecar_problems(monkeypatch):
    sc = good_sidecar()
    assert devslice.sidecar_problems(sc) == []
    assert devslice.sidecar_problems(sc, selection_sha256="a" * 64) == []
    assert "belongs to another build" in devslice.sidecar_problems(sc, selection_sha256="b" * 64)[0]
    # the manifest: the sidecar's sha is always the frozen one; manifest_sha256 (the data repo file's) must be it too,
    # so a caller's sha never stands in for the frozen one
    assert devslice.sidecar_problems(sc, manifest_sha256=fullrun.FROZEN_MANIFEST_SHA256) == []
    other = dict(sc, manifest=dict(sc["manifest"], sha256="c" * 64))
    problems = devslice.sidecar_problems(other, manifest_sha256="c" * 64)
    assert len(problems) == 2 and "is not the frozen manifest's" in problems[0] and "changed" in problems[1]
    assert devslice.sidecar_problems(sc, manifest_sha256="c" * 64) == [
        f"the manifest file has sha256 {'c' * 64}, not the frozen manifest's {fullrun.FROZEN_MANIFEST_SHA256}: the "
        f"data repo's {fullrun.FROZEN_MANIFEST} changed"]
    monkeypatch.setattr(fullrun, "FROZEN_MANIFEST_SHA256", "c" * 64)  # read at call time
    assert devslice.sidecar_problems(other, manifest_sha256="c" * 64) == []
    assert "is not the frozen manifest's" in devslice.sidecar_problems(sc)[0]
    monkeypatch.undo()

    def broken(**change) -> list[str]:
        s = copy.deepcopy(sc)
        for path, v in change.items():
            *head, last = path.split("__")
            d = s
            for k in head:
                d = d[k]
            d[last] = v
        return devslice.sidecar_problems(s)

    cases = {"kind": ("study", "kind"), "schema": (2, "kind"),
             "selection__path": ("labels/full/selections/study_1000h.parquet", "not under"),
             "manifest__path": ("labels/full/selections/full_study/study_manifest.json", "not the frozen"),
             "manifest__sha256": ("0" * 64, "not the frozen manifest's"),
             "manifest__eval_ids_equal": (False, "eval_ids_equal"), "k5__ok": (False, "k5.ok"),
             "sources": ([], "non-empty")}
    for path, (value, match) in cases.items():
        problems = broken(**{path: value})
        assert len(problems) == 1 and match in problems[0], (path, problems)
    s = copy.deepcopy(sc)
    s["dev"]["by_source"]["galgame"]["kept_rows"] = 0
    assert devslice.sidecar_problems(s) == ["sidecar: no kept dev rows for ['galgame']: the trainer's dev pick needs "
                                            "every source"]
    del s["dev"]
    assert "no kept dev rows for ['reazon_small', 'galgame']" in devslice.sidecar_problems(s)[0]
    assert devslice.sidecar_problems([]) == ["the sidecar list is not an object"]


def test_sidecar_problems_never_raise_on_a_malformed_sidecar():
    """launch prints the problems of a downloaded sidecar: a part of the wrong type is a problem, not a traceback."""
    sc = good_sidecar()
    for by in ({"reazon_small": 3, "galgame": {"kept_rows": 2}}, {"reazon_small": {"kept_rows": "5"}, "galgame": None},
               {"reazon_small": {"kept_rows": True}, "galgame": {"kept_rows": 2.5}}, ["reazon_small"], "x"):
        problems = devslice.sidecar_problems(dict(sc, dev={"by_source": by}))
        assert problems and "no kept dev rows for ['reazon_small'" in problems[-1], (by, problems)
    s = dict(sc, sources=["reazon_small", 7, None], dev="x", k5=[], manifest="m", selection=5)
    problems = devslice.sidecar_problems(s)
    assert any("no kept dev rows for ['reazon_small', 7, None]" in p for p in problems)
    assert any("manifest.path None" in p for p in problems) and any("k5.ok" in p for p in problems)
    assert any("selection.path None" in p for p in problems)
    assert devslice.sidecar_problems({"kind": "full_study", "schema": 1, "sources": ["a"],
                                      "dev": {"by_source": {"a": 3}}})
