"""The stem-prefix property the chained box's stage-2 bootstrap relies on (contract addendum E.4.3): stage 1 rebuilds
a capped extent (the smoke's study extent) with `01 --extent-config`, and stage 2 rebuilds the uncapped full extent on
the SAME data root. The second run's plan differs (extent_progress.json records another plan, so every canonical step
runs again), but each source's progress.json skips the inputs stage 1 finished, so only the rest is downloaded, and the
stems are numbered from the manifest: the data root must end up exactly as a single uncapped run leaves it - the same
manifest stems with the same rows, every sidecar's ids_sha256 the record's - with the same downloads in all. The box's
stage-2 coverage check (every stem's ids_sha256 against the record) is the guard on the box; this is the proof before
the launch. MUST PASS before `vast/launch.py --job full --box p01-chain` rents anything.

CPU only, no network: tests/test_extent.py's toy upstreams and patched world (4 rows per shard, a 3-row galgame
hold-out, a toy "300 h" that cuts inside the second Emilia tar, so EMILIA_300H_INPUTS is 2 here)."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
# the toy world and the full toy extent (module-scoped fixtures, imported by name so this module uses them)
from test_extent import FULL_LAYOUT, full, layout, make_cfg, run01, upstream, world, write_cfg  # noqa: E402,F401

from kitsune import extent  # noqa: E402
from kitsune.store import read_manifest, sidecar_meta  # noqa: E402


def recorded_ids(record: dict) -> dict:
    return {(s, st["stem"]): st["ids_sha256"] for s, src in record["sources"].items()
            for inp in src["inputs"] for st in inp["stems"]}


@pytest.mark.parametrize("inputs", [
    # the smoke's shape (fullrun.SMOKE_DATA: reazon_large 53, emilia_yodas "300h", emilia_nc 8, galgame 3): every
    # cappable source capped, Emilia at its 300 h cut
    {"reazon_large": 1, "emilia_yodas": "300h", "emilia_nc": 1, "galgame": 1},
    # Emilia past the cut (the rest step starts in the cut tar), the other caps further in
    {"reazon_large": 2, "emilia_yodas": 2, "emilia_nc": 2, "galgame": 2},
], ids=["smoke-shaped", "past-the-cut"])
def test_a_capped_then_an_uncapped_rebuild_on_one_root_is_a_single_uncapped_run(full, world, tmp_path, monkeypatch,
                                                                                  inputs):
    capped = make_cfg("smoke", inputs)
    assert extent.validate(capped) == [] and extent.within(full.cfg, capped) == []
    root = tmp_path / "data"
    run01(monkeypatch, root, "--extent-config", str(write_cfg(tmp_path / "smoke.json", capped)))
    stage1 = layout(root)
    assert stage1 and {p: full.layout[p] for p in stage1} == stage1, "stage 1's stems are the full run's"
    stage1_downloads = list(world.downloads)
    run01(monkeypatch, root, "--extent-config", str(full.config))  # stage 2: the full extent, the same data root
    assert layout(root) == full.layout == FULL_LAYOUT
    want = recorded_ids(full.record)
    for sh in read_manifest(root):
        assert sidecar_meta(root, sh)["ids_sha256"] == want[(sh.source, Path(sh.path).stem)], sh.path
    assert len(read_manifest(root)) == len(want), "no stem twice, none missing"
    # only the rest came down in stage 2: in all, exactly what one uncapped run downloads (its cut Emilia tar twice)
    assert Counter(world.downloads) == Counter(full.hub.downloads)
    assert not set(stage1_downloads) & set(world.downloads[len(stage1_downloads):]) - {
        (r, f) for (r, f), n in Counter(full.hub.downloads).items() if n > 1}
    # and the record built from this root is the label box's: the on-box coverage check reads it
    assert extent.build_record(root, full.cfg, run_ids=full.record["run_ids"], kitsune_sha="0" * 40) == full.record
