"""The dev slice of the full-data runs' selections: which train rows the trainer's early stop scores instead of training
on (plan v3 decision 17, DECISIONS C8), and the checks of the selection sidecar that launch reads before renting.

A full selection (scripts/make_selection.py with a selection_recipe.full_study block: fullrun.FULL_DATA builds
labels/full/selections/full_study/full.parquet, fullrun.SMOKE_DATA smoke.parquet) keeps the dev rows in the same parquet
as train and eval, with split "dev" (fullrun.DEV_SPLIT). A dev row's audio and labels stay in its train shard, so its
teacher_file is <source>/train-NNNNN (fullrun.shard_split). The draw works on whole shards or whole videos, never on
rows, so a programme, game or video is not split between dev and training:

  shards   reazon_small, reazon_large, galgame: one seeded pair of adjacent train shards (DEV_PAIR_SHARDS) of the
           source's sorted train stems in the extent, plus one buffer shard on each side (DEV_BUFFER_SHARDS). Every
           row of the pair becomes split "dev"; the buffers' live train rows are dropped as `dev_buffer`, so the
           neighbouring shards (which often continue the same broadcast or game) never train. The pair starts at
           k = 1 + rng.integers(n - 3), so train-00000 is never dev (galgame's train-00000 comes from the tar that
           also holds the eval hold-out) and both buffers always exist
  videos   emilia_yodas, emilia_nc: seeded whole videos (emilia_video: the YODAS video id, NC's B<batch>_S<source>)
           until their train rows reach DEV_VIDEO_ROWS (about two shards' worth). A video can span two shards, so the
           videos are counted over the source's train rows of the whole extent, not per shard. No buffer: a video is
           never split
The rng is numpy default_rng([seed, crc32("full:dev:<source>")]), so the draw depends only on the seed, the source and
its stems (or its {video: rows} map). Every row of a picked shard or video keeps its reason; only the `kept` ones have
keep True, so the dev metrics score quality labels only. The rows of the draw leave training: the pair or the videos,
the buffers, and the `dev_dup` rows (a live train row whose normalised reference, Cohere hyp or Parakeet ctc_hyp equals
one of a kept dev row's three texts; make_selection applies it after the draw).

sidecar_problems checks the selection's sidecar (<selection stem>.json, kind "full_study") for vast/launch.py's full
preflight; selection_files names the files a selection needs next to it (the sidecar and the frozen study manifest the
eval rows equal, which is referenced, never copied).

Stdlib only at import (numpy is imported inside the draw functions), like kitsune.fullrun, so launch-side code can
import it cheaply. The shared names (the dev split, the recipe block, the selection paths, seeded_subset, dev_pick) are
kitsune.fullrun's and are re-exported here.
"""
import re
import zlib
from pathlib import PurePosixPath

from kitsune.fullrun import (DEV_RULE, DEV_SPLIT, FROZEN_MANIFEST, FROZEN_MANIFEST_SHA256, FULL_DIR,  # noqa: F401
                             FULL_SELECTION, FULL_STUDY, SMOKE_DRAW_AUDIO_S, SMOKE_SELECTION, SPLITS, dev_pick,
                             full_recipe_problems, seeded_subset, shard_split)

# how each train source's dev rows are drawn (a train source outside this map cannot have a full selection)
DEV_METHOD = {"reazon_small": "shards", "reazon_large": "shards", "galgame": "shards",
              "emilia_yodas": "videos", "emilia_nc": "videos"}
DEV_PAIR_SHARDS, DEV_BUFFER_SHARDS = 2, 1  # adjacent dev shards per source; buffer shards on each side of the pair
DEV_VIDEO_ROWS = 4096  # an Emilia source's dev videos hold at least this many train rows: 2 x store.ROWS_PER_SHARD
DEV_TAG = "full:dev:{source}"  # the draw's rng tag
DRAW_TAG = "full:draw"  # the smoke selection's train draw (make_selection.draw_budget tag; the study's is study:draw)
# the drop reasons a full selection adds to the existing ones (truncated, no_agree, agree>A, no_audio)
FULL_REASONS = ("not_in_parakeet", "f1a_disagree", "eval_dup", "ctc_infeasible", "dev_buffer", "dev_dup", "not_drawn")
SIDECAR_KIND, SIDECAR_SCHEMA = "full_study", 1
SCORED_PER_SOURCE, SCORED_SEED = 600, 1234  # the trainer's default dev pick (eval.dev.per_source / seed): the sidecar's
# scored_default block records its size and hash


def emilia_video(id_or_key: str) -> str:
    """'emilia_yodas/JA_<video>_W000123' or 'JA_<video>_W000123' -> '<video>' (YODAS video ids may contain '_'); an
    emilia_nc id JA_B00000_S00000_W000000 gives B00000_S00000. The same body as scripts/01_prepare_data.py
    _emilia_video (the canonical ingest, which is not imported here; a test pins the two equal)."""
    key = id_or_key.rsplit("/", 1)[-1]
    return key[len("JA_"):key.rfind("_W")] if key.startswith("JA_") and "_W" in key else key


def _rng(seed: int, source: str):
    import numpy as np

    return np.random.default_rng([int(seed), zlib.crc32(DEV_TAG.format(source=source).encode())])


def pick_shard_pair(stems, seed: int, source: str) -> tuple[list[str], list[str]]:
    """(pair, buffers) of a shards-method source: `stems` are its train stems in the extent ("train-NNNNN", any order).
    Sorted into L (n of them), the pair is L[k], L[k+1] and the buffers L[k-1], L[k+2], with k = 1 +
    rng.integers(n - 3): never train-00000 in the pair, both buffers always there. ValueError when n < 4, a stem is
    given twice or a stem is not a train stem."""
    L = sorted(stems)
    if len(set(L)) != len(L):
        raise ValueError(f"{source}: a stem is given twice in {L}")
    if bad := [s for s in L if not re.fullmatch(r"train-\d{5}", s)]:
        raise ValueError(f"{source}: {bad[:3]} are not train stems (train-NNNNN)")
    lo, span = DEV_BUFFER_SHARDS, len(L) - DEV_PAIR_SHARDS - 2 * DEV_BUFFER_SHARDS + 1
    if span < 1:
        raise ValueError(f"{source}: {len(L)} train stems in the extent, the dev pair and its buffers need at least "
                         f"{DEV_PAIR_SHARDS + 2 * DEV_BUFFER_SHARDS}")
    k = lo + int(_rng(seed, source).integers(span))
    pair = L[k:k + DEV_PAIR_SHARDS]
    buffers = L[k - DEV_BUFFER_SHARDS:k] + L[k + DEV_PAIR_SHARDS:k + DEV_PAIR_SHARDS + DEV_BUFFER_SHARDS]
    return pair, buffers


def pick_videos(video_rows: dict, seed: int, source: str, min_rows: int | None = None) -> list[str]:
    """The dev videos of a videos-method source, sorted: the video ids sorted, permuted by the rng, and taken whole
    until their rows reach min_rows (default DEV_VIDEO_ROWS, read at call time). Depends only on (seed, source, the
    {video: rows} map). ValueError when it would take every video (no train row of the source would be left)."""
    min_rows = DEV_VIDEO_ROWS if min_rows is None else int(min_rows)
    vids = sorted(v for v, n in video_rows.items() if n > 0)
    if not vids:
        raise ValueError(f"{source}: no train rows to draw dev videos from")
    out, total = [], 0
    for i in _rng(seed, source).permutation(len(vids)):
        out.append(vids[i])
        total += int(video_rows[vids[i]])
        if total >= min_rows:
            break
    if total < min_rows or len(out) == len(vids):
        raise ValueError(f"{source}: {len(vids)} videos with {sum(int(video_rows[v]) for v in vids)} train rows; the "
                         f"dev target of {min_rows} rows would take every one of them")
    return sorted(out)


def draw(ids, sources_of, stems_of, train, sources, seed: int, *, stems: dict | None = None,
         video_rows: int | None = None) -> dict:
    """The dev draw over a selection's rows (parallel sequences: id, source, teacher stem, and `train`: the row is a
    train row, split "train" before any dev assignment). For every train source in `sources` (each must be in
    DEV_METHOD): shards -> pick_shard_pair over its train stems in the extent (`stems`: source -> extent stems, else
    the stems of its train rows); videos -> pick_videos over the rows per video of its train rows (video_rows: the
    target, default DEV_VIDEO_ROWS at call time). Returns {"dev": bool array (the rows that become split dev),
    "buffer": bool array (the buffer shards' train rows), "by_source": {source: {method, dev_stems, buffer_stems} or
    {method, videos, video_rows}}}. ValueError on a source outside DEV_METHOD or one the draw cannot serve."""
    import numpy as np

    from collections import Counter

    ids = np.asarray(ids, dtype=object)
    src = np.asarray(sources_of, dtype=object)
    stem = np.asarray(stems_of, dtype=object)
    train = np.asarray(train, dtype=bool)
    target = DEV_VIDEO_ROWS if video_rows is None else int(video_rows)
    dev, buf = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
    by_source = {}
    for s in sources:
        method = DEV_METHOD.get(s)
        if method is None:
            raise ValueError(f"{s}: no dev-slice method (the full selection draws dev rows of {sorted(DEV_METHOD)})")
        mine = np.flatnonzero(train & (src == s))
        if method == "shards":
            pool = stems.get(s, set()) if stems is not None else set(stem[mine])
            pair, buffers = pick_shard_pair([x for x in pool if str(x).startswith("train-")], seed, s)
            dev[mine[np.fromiter((x in pair for x in stem[mine]), bool, len(mine))]] = True
            buf[mine[np.fromiter((x in buffers for x in stem[mine]), bool, len(mine))]] = True
            by_source[s] = {"method": method, "dev_stems": pair, "buffer_stems": buffers}
        else:
            video = [emilia_video(str(i)) for i in ids[mine]]
            picked = pick_videos(dict(Counter(video)), seed, s, target)
            chosen = set(picked)
            dev[mine[np.fromiter((v in chosen for v in video), bool, len(mine))]] = True
            by_source[s] = {"method": method, "videos": picked, "video_rows": target}
    return {"dev": dev, "buffer": buf, "by_source": by_source}


def sidecar_path(selection: str) -> str:
    """The sidecar of a selection (repo path): <selection without .parquet>.json (the study's convention)."""
    return str(PurePosixPath(selection[: -len(".parquet")] if selection.endswith(".parquet") else selection)) + ".json"


def selection_files(cfg: dict) -> list[str]:
    """The data-repo files a config's selection needs next to it: a full_study selection its sidecar and the frozen
    study manifest (FROZEN_MANIFEST, which its eval rows equal); a study selection kitsune.prereg.study_files (its
    sidecar and the study_manifest.json in its folder); any other config none."""
    recipe = cfg.get("selection_recipe") or {}
    sel = cfg.get("selection") or ""
    if recipe.get("full_study") is not None:
        return [sidecar_path(sel), FROZEN_MANIFEST]
    if recipe.get("study") is not None:
        from kitsune import prereg

        return list(prereg.study_files(sel))
    return []


def sidecar_problems(sidecar, *, selection_sha256: str | None = None, manifest_sha256: str | None = None) -> list[str]:
    """Why a full selection's sidecar must not be trained on (empty: it may): kind "full_study" and schema 1; its
    selection is under FULL_DIR and, when selection_sha256 is given (launch: the Hub's LFS sha of the parquet),
    has that sha; its manifest is FROZEN_MANIFEST with the sha manifest_sha256 (default FROZEN_MANIFEST_SHA256) and
    eval_ids_equal; the K5 check passed; every train source has kept dev rows."""
    if not isinstance(sidecar, dict):
        return [f"the sidecar {type(sidecar).__name__} is not an object"]
    p = []
    if sidecar.get("kind") != SIDECAR_KIND or sidecar.get("schema") != SIDECAR_SCHEMA:
        p.append(f"sidecar kind {sidecar.get('kind')!r} schema {sidecar.get('schema')!r}, expected {SIDECAR_KIND!r} "
                 f"schema {SIDECAR_SCHEMA} (a full selection's, scripts/make_selection.py full mode)")
    sel = sidecar.get("selection") if isinstance(sidecar.get("selection"), dict) else {}
    if not str(sel.get("path") or "").startswith(FULL_DIR + "/"):
        p.append(f"sidecar selection.path {sel.get('path')!r} is not under {FULL_DIR}/")
    if selection_sha256 is not None and sel.get("sha256") != selection_sha256:
        p.append(f"sidecar selection.sha256 {sel.get('sha256')!r} is not the selection's {selection_sha256}: the "
                 f"sidecar belongs to another build")
    man = sidecar.get("manifest") if isinstance(sidecar.get("manifest"), dict) else {}
    want = manifest_sha256 or FROZEN_MANIFEST_SHA256
    if man.get("path") != FROZEN_MANIFEST:
        p.append(f"sidecar manifest.path {man.get('path')!r} is not the frozen {FROZEN_MANIFEST}")
    if man.get("sha256") != want:
        p.append(f"sidecar manifest.sha256 {man.get('sha256')!r} is not the frozen manifest's {want}")
    if man.get("eval_ids_equal") is not True:
        p.append("sidecar manifest.eval_ids_equal is not true: the eval rows are not the frozen manifest's")
    k5 = sidecar.get("k5") if isinstance(sidecar.get("k5"), dict) else {}
    if k5.get("ok") is not True:
        p.append(f"sidecar k5.ok is not true ({k5.get('not_in_parakeet')!r} of {k5.get('candidates')!r} train rows "
                 f"not in parakeet_out, limit {k5.get('max_frac')!r})")
    by = ((sidecar.get("dev") or {}).get("by_source") or {}) if isinstance(sidecar.get("dev"), dict) else {}
    sources = sidecar.get("sources") if isinstance(sidecar.get("sources"), list) else []
    if not sources:
        p.append("sidecar sources is not a non-empty list")
    if missing := [s for s in sources if not ((by.get(s) or {}).get("kept_rows") or 0) > 0]:
        p.append(f"sidecar: no kept dev rows for {missing}: the trainer's dev pick needs every source")
    return p
