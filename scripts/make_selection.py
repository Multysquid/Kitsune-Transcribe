"""Decide which teacher-labelled utterances a distillation run trains and evaluates on, and record why per row.

Why a separate, persisted selection: the label filter is a pre-registered decision of the run, the training box
never needs second_out's raw rows or the local shards to apply it, and every later number (hours trained, probe,
greedy subsets) must be traceable to one file. The selection is small (~2 MB) and is parked on HF with teacher_out.

Candidates come from teacher_out/<source>/<split>-*.npz ONLY (never data/manifest.jsonl, which also lists shards the
teacher has not labelled). Train sources use split `train`; eval sets use split `eval`.

Rules for TRAIN rows, first match wins (keep = reason == "kept"):
  truncated    the teacher never emitted EOS (repetition or length cut): its labels are garbage (median agree 0.9-3)
  not_judged   (--partial-second-opinion sources only) the row's teacher shard has no second-opinion file at all:
               02b was deliberately not run on it, so the source trains on its judged shards only
  no_agree     no second opinion for the row (02b not run on its shard, or hyp2 null): the gate cannot be evaluated
  agree>A      CER(teacher hyp, second opinion) > --agree-max: the two models disagree, so the label is suspect
  no_audio     id not in data/shards/<source>/*.parquet (skip with --skip-audio-check)
  kept
EVAL rows are never filtered by label quality - the gate compares against the teacher's FULL-set CER - so only
no_audio applies; `truncated` / `agree` are still recorded. Exception: monitor-only hold-outs named in
--filter-eval-sets (eval_emilia, galgame) get the train rules, so they measure clean speech (eval_emilia's last tar
holds English talk that Whisper-medium wrote as katakana, which the teacher-vs-Whisper agreement exposes).

Also marks two fixed seeded subsets (per source, drawn from kept rows sorted by id):
  in_greedy_subset   --greedy-n ids per eval set: greedy decode during training (the full sets only at the end)
  in_probe           --probe-n ids per train source: un-augmented teacher-forced probe of the fit on training data

Output columns: id, source, split, teacher_file (<source>/<stem>, the npz/jsonl holding the row), duration, n_tok,
truncated, agree (null if unknown), teacher_cer, keep, reason, in_greedy_subset, in_probe. The parquet's schema
metadata key b"kitsune_selection" holds the arguments and the summary as JSON.

The recipe of a run (its sources and eval sets, and selection_recipe: --agree-max, --agree-max-source,
--filter-eval-sets, --partial-second-opinion) is written down once, in the run config: --config takes all of it from there, and vast/launch.py
refuses a selection whose recorded arguments differ from the config (a flag forgotten on a rebuild would otherwise
silently change the training data or drop a hold-out). With --config the roots (--teacher-out, --second-out, --data)
default to the config's teacher_root, second_root and data_root.

Extents (kitsune/extent.py): a config with an `extent` block takes only the teacher stems of its subset
(extent.subset_stems of the extent record <extent.root>/extent.json, or --extent-record), and the selection records
`extent: {name, inputs}` among its arguments. The audio check also reads the id sidecars (shards/<s>/_ids/), so it
works after the label box pruned a shard's audio. --from-selection BASE derives the selection of a subset config or of
another threshold from an existing selection alone (no npz, second_out or audio): the rules above are row-local, so
they are recomputed from the stored `truncated` / `agree` and the seeded subsets redrawn; the result equals a
from-scratch build. It refuses a base with no_audio rows and a recipe with partial_second_opinion (a selection does
not record which shards were judged).

Usage:
  python scripts/make_selection.py --config configs/viability.json       # the viability run's selection
  python scripts/make_selection.py --sources reazon_small --agree-max 0.5 --out selection/reazon_only.parquet
  python scripts/make_selection.py --config configs/full_sub3k.json --from-selection labels/full/selections/full.parquet
"""
import argparse
from collections.abc import Sequence
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import kitsune.extent as kextent  # noqa: E402
from kitsune.store import SIDECAR_DIR, fsync_path  # noqa: E402

EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon"]  # the pre-registered gate sets; monitor hold-outs are extra
COLUMNS = ["id", "source", "split", "teacher_file", "duration", "n_tok", "truncated", "agree", "teacher_cer", "keep",
           "reason", "in_greedy_subset", "in_probe"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_rows(teacher_root: Path, source: str, split: str, eos: int, stems: set[str] | None = None) -> pd.DataFrame:
    """One row per utterance with teacher output, in teacher_out order (stems sorted, row order within); with
    `stems` only the npz of those stems (an extent's subset)."""
    parts = []
    for npz in sorted((teacher_root / source).glob(f"{split}-*.npz")):
        if stems is not None and npz.stem not in stems:
            continue
        z = np.load(npz)
        ids = z["ids"].tolist()
        off = z["tok_offsets"]
        n_tok = np.diff(off)
        jrows = read_jsonl(npz.with_suffix(".jsonl"))
        if [r["id"] for r in jrows] != ids:
            raise ValueError(f"{npz}: jsonl ids differ from npz ids")
        truncated = np.array([bool(r["truncated"]) for r in jrows], dtype=bool)
        tok = z["tokens"]
        ends_eos = np.zeros(len(ids), dtype=bool)
        nz = n_tok > 0
        ends_eos[nz] = tok[off[1:][nz] - 1] == eos
        if (truncated == ends_eos).any():  # truncated <=> no final EOS, by construction of 02_teacher_pass
            raise ValueError(f"{npz}: jsonl `truncated` disagrees with the stored tokens")
        parts.append(pd.DataFrame(dict(
            id=ids, source=source, split=split, teacher_file=f"{source}/{npz.stem}",
            duration=z["duration"].astype(np.float32), n_tok=n_tok.astype(np.int32), truncated=truncated,
            teacher_cer=z["cer"].astype(np.float32),
        )))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)


def judged_stems(second_root: Path, source: str) -> set[str]:
    """Teacher shard stems that have a second-opinion file (02b writes second_out/<source>/<stem>.jsonl per shard)."""
    return {p.stem for p in (second_root / source).glob("*.jsonl")}


def agree_map(second_root: Path, source: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for p in sorted((second_root / source).glob("*.jsonl")):
        for r in read_jsonl(p):
            out[r["id"]] = r.get("agree")
    return out


def audio_ids(data_root: Path, source: str, split: str) -> set[str]:
    """Ids present in the local shards; reads only the id column (the audio column is never touched). The id sidecars
    (shards/<source>/_ids/) count too: a pruned shard's audio is rebuilt from its pinned upstream, not lost."""
    out: set[str] = set()
    d = data_root / "shards" / source
    for p in sorted(d.glob(f"{split}-*.parquet")) + sorted((d / SIDECAR_DIR).glob(f"{split}-*.parquet")):
        out.update(pq.read_table(p, columns=["id"]).column("id").to_pylist())
    return out


def seeded_subset(ids: list[str], n: int, seed: int, tag: str) -> set[str]:
    """min(n, len) ids drawn without replacement; depends only on (seed, tag, the id set) - not on file order."""
    ids = sorted(ids)
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    return {ids[i] for i in rng.choice(len(ids), size=min(n, len(ids)), replace=False)}


def build_selection(teacher_root, second_root, data_root, sources: list[str], eval_sets: list[str] = EVAL_SETS,
                    agree_max: float = 0.5, seed: int = 1234, greedy_n: int = 500, probe_n: int = 500,
                    audio_check: bool = True, agree_max_by_source: dict[str, float] | None = None,
                    filter_eval_sets: Sequence[str] = (), partial_second_opinion: Sequence[str] = (),
                    stems: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """`agree_max_by_source` overrides `agree_max` per train source: each source's `agree` is measured against a
    different second model (whisper-large-v3 for reazon, Emilia's own Whisper-medium text for emilia_yodas), so one
    global threshold is not calibrated across sources. For the sources in `partial_second_opinion` a row whose whole
    teacher shard has no second-opinion file is `not_judged` instead of `no_agree`: the partial coverage is the run's
    decision (vast/launch.py accepts it only for these sources), not a pass that has yet to finish. `stems` (source ->
    teacher stems; extent.subset_stems) keeps only those npz: a source it does not list has none."""
    teacher_root, second_root, data_root = Path(teacher_root), Path(second_root), Path(data_root)
    meta_p = teacher_root / "meta.json"
    eos = int(json.loads(meta_p.read_text(encoding="utf-8")).get("eos_token_id", 3)) if meta_p.exists() else 3
    agree_max_by_source = agree_max_by_source or {}
    frames = []
    missing: list[str] = []
    n_no_audio_any = 0
    for source, split in [(s, "train") for s in sources] + [(s, "eval") for s in eval_sets]:
        if stems is not None:  # an extent names its stems: a partly pulled root must not quietly select fewer rows
            missing += [f"{source}/{st}" for st in sorted(stems.get(source, set()))
                        if st.startswith(f"{split}-") and not (teacher_root / source / f"{st}.npz").is_file()]
        df = teacher_rows(teacher_root, source, split, eos, None if stems is None else stems.get(source, set()))
        if df.empty:
            print(f"WARNING: no teacher output for {source}/{split}-*.npz")
            continue
        amap = agree_map(second_root, source)
        agree = df["id"].map(lambda x: amap.get(x)).astype("float64")  # None/missing -> NaN
        df["agree"] = agree.astype(np.float32)
        have_audio = df["id"].isin(audio_ids(data_root, source, split)) if audio_check else pd.Series(True, index=df.index)
        reason = np.full(len(df), "kept", dtype=object)
        reason[~have_audio.to_numpy()] = "no_audio"  # assigned lowest-precedence first, then overwritten
        n_no_audio_any += int((~have_audio).sum())  # before the overwrite: derive_selection refuses a base with any
        if split == "train" or source in filter_eval_sets:
            thr = agree_max_by_source.get(source, agree_max)
            reason[(agree > thr).to_numpy()] = f"agree>{thr:g}"
            reason[agree.isna().to_numpy()] = "no_agree"
            if source in partial_second_opinion:
                shard_stems = df["teacher_file"].str.rsplit("/", n=1).str[1]
                reason[(agree.isna() & ~shard_stems.isin(judged_stems(second_root, source))).to_numpy()] = "not_judged"
            reason[df["truncated"].to_numpy()] = "truncated"
        df["reason"] = reason
        df["keep"] = df["reason"] == "kept"
        kept = df["id"][df["keep"]].tolist()
        chosen = seeded_subset(kept, greedy_n if split == "eval" else probe_n, seed, f"{split}:{source}")
        df["in_greedy_subset"] = df["id"].isin(chosen) if split == "eval" else False
        df["in_probe"] = df["id"].isin(chosen) if split == "train" else False
        frames.append(df[COLUMNS])
    if missing:
        raise ValueError(f"{len(missing)} stem(s) of the extent have no teacher npz under {teacher_root}: "
                         f"{missing[:10]}")
    if not frames:
        raise SystemExit("no teacher output for any requested source")
    sel = pd.concat(frames, ignore_index=True)
    sel.attrs["n_no_audio_any"] = n_no_audio_any
    if sel["id"].duplicated().any():
        raise ValueError(f"duplicate ids across teacher_out, e.g. {sel['id'][sel['id'].duplicated()].iloc[0]}")
    return sel


def derive_selection(base: pd.DataFrame, sources: list[str], eval_sets: list[str] = EVAL_SETS,
                     agree_max: float = 0.5, seed: int = 1234, greedy_n: int = 500, probe_n: int = 500,
                     agree_max_by_source: dict[str, float] | None = None, filter_eval_sets: Sequence[str] = (),
                     stems: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """build_selection's result from an existing selection `base` instead of teacher_out / second_out / the audio:
    the rows of `sources` (train) and `eval_sets` (eval), of `stems` only if given, with reason / keep recomputed
    from the stored truncated / agree (same rules and precedence) and the seeded subsets redrawn. `agree` is stored
    as float32, so it is compared against the float32 threshold (second_out rounds it to 4 decimals, so the order
    against a threshold is the float64 one). Refuses a base with rows whose final reason is no_audio. no_audio has
    the lowest precedence in build_selection (truncated / agree / no_agree overwrite it), so a row without audio can
    also hide under another reason: main refuses a base whose metadata counts any row without audio
    (n_no_audio_any), which this function cannot see. Refuses stems of `stems` absent from the base."""
    if (base["reason"] == "no_audio").any():
        raise ValueError("the base selection has no_audio rows; derive from a selection built with all its audio")
    agree_max_by_source = agree_max_by_source or {}
    frames = []
    for source, split in [(s, "train") for s in sources] + [(s, "eval") for s in eval_sets]:
        df = base[(base["source"] == source) & (base["split"] == split)]
        if stems is not None:
            have = set(df["teacher_file"].str.rsplit("/", n=1).str[1])
            if absent := sorted(st for st in stems.get(source, set()) if st.startswith(f"{split}-") and st not in have):
                raise ValueError(f"{len(absent)} stem(s) of {source}/{split} in the extent are not in the base "
                                 f"selection: {absent[:10]}")
            df = df[df["teacher_file"].str.rsplit("/", n=1).str[1].isin(stems.get(source, set()))]
        df = df.reset_index(drop=True).copy()
        if df.empty:
            print(f"WARNING: no rows of {source}/{split} in the base selection")
            continue
        agree = df["agree"].astype(np.float32)
        reason = np.full(len(df), "kept", dtype=object)
        if split == "train" or source in filter_eval_sets:
            thr = agree_max_by_source.get(source, agree_max)
            reason[(agree > np.float32(thr)).to_numpy()] = f"agree>{thr:g}"
            reason[agree.isna().to_numpy()] = "no_agree"
            reason[df["truncated"].astype(bool).to_numpy()] = "truncated"
        df["reason"] = reason
        df["keep"] = df["reason"] == "kept"
        kept = df["id"][df["keep"]].tolist()
        chosen = seeded_subset(kept, greedy_n if split == "eval" else probe_n, seed, f"{split}:{source}")
        df["in_greedy_subset"] = df["id"].isin(chosen) if split == "eval" else False
        df["in_probe"] = df["id"].isin(chosen) if split == "train" else False
        frames.append(df[COLUMNS])
    if not frames:
        raise SystemExit("no rows of any requested source in the base selection")
    return pd.concat(frames, ignore_index=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def summarize(sel: pd.DataFrame) -> pd.DataFrame:
    """Utterances and hours per source, split and reason."""
    g = sel.groupby(["source", "split", "reason"], sort=True)
    return pd.DataFrame(dict(utts=g.size(), hours=g["duration"].sum() / 3600)).reset_index()


def write_selection(sel: pd.DataFrame, out: Path, meta: dict):
    table = pa.table(dict(
        id=pa.array(sel["id"], pa.string()), source=pa.array(sel["source"], pa.string()),
        split=pa.array(sel["split"], pa.string()), teacher_file=pa.array(sel["teacher_file"], pa.string()),
        duration=pa.array(sel["duration"], pa.float32()), n_tok=pa.array(sel["n_tok"], pa.int32()),
        truncated=pa.array(sel["truncated"], pa.bool_()), agree=pa.array(sel["agree"], pa.float32(), from_pandas=True),
        teacher_cer=pa.array(sel["teacher_cer"], pa.float32()), keep=pa.array(sel["keep"], pa.bool_()),
        reason=pa.array(sel["reason"], pa.string()), in_greedy_subset=pa.array(sel["in_greedy_subset"], pa.bool_()),
        in_probe=pa.array(sel["in_probe"], pa.bool_()),
    ))
    table = table.replace_schema_metadata({b"kitsune_selection": json.dumps(meta, ensure_ascii=False).encode()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    pq.write_table(table, tmp)
    fsync_path(tmp)
    tmp.replace(out)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="run config (e.g. configs/viability.json): --sources, --eval-sets and its selection_recipe "
                         "(--agree-max, --agree-max-source, --filter-eval-sets) come from it, --out defaults to its "
                         "selection; none of those flags may be given too")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="train sources (their teacher_out split `train`); required without --config")
    ap.add_argument("--eval-sets", nargs="*", default=None,
                    help=f"eval sources (split `eval`; default {' '.join(EVAL_SETS)}); none: --eval-sets")
    ap.add_argument("--agree-max", type=float, default=None, help="keep train rows with agree <= this (default 0.5)")
    ap.add_argument("--agree-max-source", action="append", default=None, metavar="SOURCE=A",
                    help="per-source override of --agree-max (repeatable), e.g. emilia_yodas=0.2")
    ap.add_argument("--filter-eval-sets", nargs="*", default=None,
                    help="monitor-only eval sets that get the train label rules (never the pre-registered gate sets)")
    ap.add_argument("--partial-second-opinion", nargs="*", default=None, metavar="SOURCE",
                    help="sources that deliberately train on their judged shards only: rows of a shard without a "
                         "second-opinion file are not_judged, not no_agree")
    ap.add_argument("--out", default=None, help="default: the --config's selection, else selection/viability.parquet")
    ap.add_argument("--teacher-out", default=None, help="default: the --config's teacher_root, else teacher_out")
    ap.add_argument("--second-out", default=None, help="default: the --config's second_root, else second_out")
    ap.add_argument("--data", default=None, help="default: the --config's data_root, else data")
    ap.add_argument("--extent-record", default=None,
                    help="extent record of the --config's extent (default <extent.root>/extent.json)")
    ap.add_argument("--from-selection", default=None, metavar="PATH",
                    help="derive from this selection (no teacher_out, second_out or audio needed): the rows of the "
                         "recipe's sources and extent, reasons recomputed, seeded subsets redrawn")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--greedy-n", type=int, default=500, help="greedy-decode subset size per eval set")
    ap.add_argument("--probe-n", type=int, default=500, help="train probe size per train source")
    ap.add_argument("--skip-audio-check", action="store_true", help="do not look for the audio (no `no_audio` rows)")
    args = ap.parse_args(argv)
    recipe_flags = ("sources", "eval_sets", "agree_max", "agree_max_source", "filter_eval_sets",
                    "partial_second_opinion")
    cfg: dict = {}
    if args.config:
        if given := [f"--{k.replace('_', '-')}" for k in recipe_flags if getattr(args, k) is not None]:
            ap.error(f"--config sets the recipe; drop {' '.join(given)}")
        p = Path(args.config)
        cfg = json.loads((p if p.is_absolute() or p.exists() else ROOT / p).read_text(encoding="utf-8"))
        recipe = cfg.get("selection_recipe")
        if not isinstance(recipe, dict) or not cfg.get("sources") or "eval_sets" not in cfg:
            ap.error(f"{args.config} needs sources, eval_sets and selection_recipe")
        args.sources, args.eval_sets = list(cfg["sources"]), list(cfg["eval_sets"])
        args.agree_max, args.agree_max_source = float(recipe["agree_max"]), list(recipe["agree_max_source"])
        args.filter_eval_sets = list(recipe["filter_eval_sets"])
        args.partial_second_opinion = list(recipe.get("partial_second_opinion", []))
        args.out = args.out or str(_repo_path(cfg.get("selection", "selection/viability.parquet")))
    elif not args.sources:
        ap.error("--sources is required (or --config)")
    else:
        args.eval_sets = EVAL_SETS if args.eval_sets is None else args.eval_sets
        args.agree_max = 0.5 if args.agree_max is None else args.agree_max
        args.agree_max_source, args.filter_eval_sets = args.agree_max_source or [], args.filter_eval_sets or []
        args.partial_second_opinion = args.partial_second_opinion or []
        args.out = args.out or str(ROOT / "selection" / "viability.parquet")
    # the roots: explicit flags win, then the config's (the viability config's are these defaults)
    for flag, key, default in (("teacher_out", "teacher_root", "teacher_out"),
                               ("second_out", "second_root", "second_out"), ("data", "data_root", "data")):
        if getattr(args, flag) is None:
            setattr(args, flag, str(_repo_path(cfg.get(key) or default)))

    by_source = {}
    for item in args.agree_max_source:
        src, _, val = item.partition("=")
        if not val:
            ap.error(f"--agree-max-source expects SOURCE=A, got {item!r}")
        by_source[src] = float(val)
    if set(args.filter_eval_sets) & set(EVAL_SETS):
        ap.error(f"the gate sets {EVAL_SETS} are never label-filtered")
    unknown = set(by_source) - set(args.sources) - set(args.filter_eval_sets)
    if unknown:
        ap.error(f"--agree-max-source for sources not in --sources: {sorted(unknown)}")
    if stray := set(args.partial_second_opinion) - set(args.sources) - set(args.filter_eval_sets):
        ap.error(f"--partial-second-opinion for sources not in --sources: {sorted(stray)}")

    # the extent: only its subset's stems, from the label run's record
    ext, stems = kextent.extent_block(cfg), None
    args.extent = None if ext is None else dict(name=ext.get("name"), inputs=ext.get("inputs") or {})
    if ext is None and args.extent_record:
        ap.error("--extent-record needs a --config with an extent block")
    if ext is not None:
        if problems := kextent.validate(cfg):
            ap.error(f"{args.config}: " + "; ".join(problems))
        rec_path = Path(args.extent_record) if args.extent_record else _repo_path(ext["root"]) / kextent.RECORD_FILE
        record = kextent.load_record(rec_path)
        if problems := kextent.record_problems(record, cfg):
            ap.error(f"{rec_path}: " + "; ".join(problems))
        stems = kextent.subset_stems(record, cfg)

    t0 = time.time()
    if args.from_selection:
        if args.partial_second_opinion:
            ap.error("--from-selection cannot apply partial_second_opinion (a selection does not record which shards "
                     "second_out judged): build it from scratch")
        base_p = Path(args.from_selection)
        raw = (pq.ParquetFile(base_p).schema_arrow.metadata or {}).get(b"kitsune_selection")
        base_args = json.loads(raw).get("args", {}) if raw else {}
        base_ext = base_args.get("extent")
        if (base_ext is None) != (ext is None):
            ap.error(f"{base_p} was built {'without' if base_ext is None else 'with'} an extent, the config has "
                     f"{'none' if ext is None else 'one'}")
        if ext is not None:
            outer = dict(sources=base_args.get("sources") or [], eval_sets=base_args.get("eval_sets") or [],
                         extent=dict(name=base_ext.get("name"), root=ext.get("root"), inputs=base_ext.get("inputs")))
            if problems := kextent.within(outer, cfg):
                ap.error(f"the config's extent is not inside {base_p}'s: " + "; ".join(problems))
        elif missing := (set(args.sources) - set(base_args.get("sources") or [])) | (
                set(args.eval_sets) - set(base_args.get("eval_sets") or [])):
            ap.error(f"{base_p} was not built for {sorted(missing)}")
        args.skip_audio_check = args.skip_audio_check or bool(base_args.get("skip_audio_check"))
        n_hidden = (json.loads(raw) if raw else {}).get("n_no_audio_any")
        if n_hidden or (n_hidden is None and not base_args.get("skip_audio_check")):
            ap.error(f"{base_p} has {n_hidden if n_hidden is not None else 'an unknown number of'} row(s) without "
                     f"audio (possibly under another reason); a looser recipe could keep them: build from scratch")
        try:
            sel = derive_selection(pd.read_parquet(base_p), args.sources, args.eval_sets, args.agree_max, args.seed,
                                   args.greedy_n, args.probe_n, by_source, args.filter_eval_sets, stems)
        except ValueError as e:
            ap.error(f"{base_p}: {e}")
        args.from_selection = dict(path=str(base_p), sha256=file_sha256(base_p))
        sel.attrs["n_no_audio_any"] = 0
    else:
        sel = build_selection(args.teacher_out, args.second_out, args.data, args.sources, args.eval_sets,
                              args.agree_max, args.seed, args.greedy_n, args.probe_n, not args.skip_audio_check,
                              by_source, args.filter_eval_sets, args.partial_second_opinion, stems)
    summary = summarize(sel)
    kept = sel[sel["keep"]].groupby(["source", "split"])["duration"].agg(["size", "sum"])
    meta = dict(args=vars(args), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                n_no_audio_any=int(sel.attrs.get("n_no_audio_any", 0)),
                summary=summary.to_dict(orient="records"),
                kept={f"{s}/{sp}": dict(utts=int(r["size"]), hours=float(r["sum"] / 3600)) for (s, sp), r in kept.iterrows()})
    out = Path(args.out)
    write_selection(sel, out, meta)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False, formatters=dict(hours="{:.3f}".format)))
    print()
    for (s, sp), r in kept.iterrows():
        tot = sel[(sel["source"] == s) & (sel["split"] == sp)]["duration"].sum()
        sub = sel[(sel["source"] == s) & (sel["split"] == sp) & (sel["in_greedy_subset"] | sel["in_probe"])]
        print(f"{s}/{sp}: kept {int(r['size'])} utts, {r['sum'] / 3600:.2f} h of {tot / 3600:.2f} h "
              f"({r['sum'] / max(tot, 1e-9):.1%}); {'greedy subset' if sp == 'eval' else 'probe'} {len(sub)} utts, "
              f"{sub['duration'].sum() / 3600:.3f} h")
    print(f"\nwrote {out} ({len(sel)} rows) in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
