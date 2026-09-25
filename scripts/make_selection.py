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

The size study (a `selection_recipe.study` block, e.g. study/data.json; kitsune/prereg.py STUDY_SELECTION holds the
pre-registered values) builds one frozen train list that every study run of both families reads, on the laptop CPU from
the sealed label root. Rules for TRAIN rows, first match wins, then the existing ones in their order:
  not_in_parakeet  candidates are the rows present in both teacher_out and parakeet_out (a teacher shard without a
                   Parakeet npz/jsonl at all is an error: a partly pulled root)
  (F0)             truncated / no_agree / agree>A / no_audio exactly as above: the label box's judges (second_out:
                   whisper-large-v3 for Reazon, Emilia's WhisperX text for Emilia-YODAS and NC, Parakeet for Galgame)
  f1a_disagree     CER(Cohere hyp, Parakeet TDT hyp) > study.f1a_max, normalised by Parakeet's length
  eval_dup         the normalised reference or Cohere hypothesis equals the normalised reference of ANY eval-split row
                   of the eval sets (the hold-outs included) of >= study.dedup_min_chars characters
  ctc_infeasible   U + adjacent repeats of the greedy CTC target (ctc_greedy(ctc_col0(...)) over the npz's stored
                   frames) > the stored n_frames: F.ctc_loss cannot align it. The greedy path is itself an alignment
                   of its own collapse, so this fires only on an npz whose frames disagree with its n_frames
  not_drawn        keep = a seeded study.draw_audio_s-second subset pooled over the train sources (draw_budget)
  kept
EVAL rows: every row present in both roots (not_in_parakeet otherwise, and no_audio), never label-filtered (the
recipe's filter_eval_sets must be []). The probe is study.probe_n kept rows per train source (the convention above),
the greedy subsets as above. Next to the parquet it writes, deterministically (no timestamps: a rebuild gives the same
bytes, so the hashes pre-register):
  <selection>.json      the sidecar: the recipe and seed, hours per source after each rule, the draw, ids_sha256 of
                        the train list, the probe and each eval set (kitsune.store.ids_sha256, selection order), the
                        Galgame views, the teachers' baselines on the manifest rows and the file hashes
  study_manifest.json   per eval set its ordered ids and their sha256, and the Galgame view id lists: neutral
                        (cer(kotoba-whisper hyp2, ref) <= study.neutral_max_cer and a non-empty reference, from the
                        laptop's second_out/galgame/eval-00000.jsonl: --kotoba-galgame), all, and label_box (the label
                        box's own filter: not truncated, agree <= the galgame threshold)
python -m kitsune.prereg --write study/ --sidecar <selection>.json then fills the PREREG's pending fields.

Usage:
  python scripts/make_selection.py --config configs/viability.json       # the viability run's selection
  python scripts/make_selection.py --sources reazon_small --agree-max 0.5 --out selection/reazon_only.parquet
  python scripts/make_selection.py --config configs/full_sub3k.json --from-selection labels/full/selections/full.parquet
  python scripts/make_selection.py --config study/data.json --skip-audio-check     # the study (labels/full pulled)
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
from kitsune import prereg as kprereg  # noqa: E402
from kitsune.parakeet_targets import BLANK as CTC_BLANK  # noqa: E402
from kitsune.store import SIDECAR_DIR, fsync_path, ids_sha256  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402
from kitsune.text import normalize_ja  # noqa: E402

EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon"]  # the pre-registered gate sets; monitor hold-outs are extra
COLUMNS = ["id", "source", "split", "teacher_file", "duration", "n_tok", "truncated", "agree", "teacher_cer", "keep",
           "reason", "in_greedy_subset", "in_probe"]
# the laptop's kotoba-whisper-v2.0 hypotheses of the galgame hold-out (02b before the label box): the neutral view
KOTOBA_GALGAME = "second_out/galgame/eval-00000.jsonl"
STUDY_SCHEMA = 1


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


# ------------------------------------------------------------------------------------------------- the size study


def study_problems(study) -> list[str]:
    """Problems with a selection_recipe.study block: exactly the keys of kitsune.prereg.STUDY_SELECTION, thresholds
    and the draw budget positive numbers, dedup_min_chars an int >= 1, probe_n an int >= 0."""
    want = kprereg.STUDY_SELECTION
    if not isinstance(study, dict):
        return [f"selection_recipe.study is an object like {want}, not {study!r}"]
    problems = [f"selection_recipe.study.{k}: unknown key (the keys are {', '.join(want)})" for k in study
                if k not in want]
    problems += [f"selection_recipe.study.{k}: missing" for k in want if k not in study]

    def is_int(v):
        return isinstance(v, int) and not isinstance(v, bool)

    for k in ("f1a_max", "neutral_max_cer", "draw_audio_s"):
        v = study.get(k)
        if k in study and not ((is_int(v) or isinstance(v, float)) and np.isfinite(v) and v > 0):
            problems.append(f"selection_recipe.study.{k} is a positive number, not {v!r}")
    for k, lo in (("dedup_min_chars", 1), ("probe_n", 0)):
        if k in study and not (is_int(study[k]) and study[k] >= lo):
            problems.append(f"selection_recipe.study.{k} is an int >= {lo}, not {study[k]!r}")
    return problems


def text_rows(root: Path, files: list[str], fields: tuple[str, ...]) -> pd.DataFrame:
    """id + `fields` (None -> "") of <root>/<file>.jsonl for every <source>/<stem> in `files`, in file order."""
    parts = []
    for tf in files:
        rows = read_jsonl(root / f"{tf}.jsonl")
        parts.append(pd.DataFrame({"id": [r["id"] for r in rows],
                                   **{f: [r.get(f) or "" for r in rows] for f in fields}}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", *fields])


def ctc_frames_needed(frame_offsets, dense_offsets, dense_frame, col0_dense) -> np.ndarray:
    """Per utterance of a parakeet_out npz: U + the adjacent repeats of its greedy CTC target, the frames a CTC
    alignment of it needs (a repeated token needs a blank between its copies). The target is
    kitsune.parakeet_targets.ctc_greedy(ctc_col0(...)) over the utterance's STORED frames (frame_offsets), vectorised
    over the shard: col0 = blank off the dense frames, the stored top-1 on them; a token is emitted where col0 is not
    blank and differs from the previous frame of the same utterance."""
    frame_offsets = np.asarray(frame_offsets, dtype=np.int64)
    dense_offsets = np.asarray(dense_offsets, dtype=np.int64)
    n = len(frame_offsets) - 1
    span = np.diff(frame_offsets)
    utt_d = np.repeat(np.arange(n), np.diff(dense_offsets))
    dense_frame = np.asarray(dense_frame, dtype=np.int64)
    if len(dense_frame) and (np.any(dense_frame < 0) or np.any(dense_frame >= span[utt_d])):
        raise ValueError("a dense CTC frame lies outside its utterance's stored frames")
    col0 = np.full(int(frame_offsets[-1]), CTC_BLANK, dtype=np.int64)
    col0[frame_offsets[:-1][utt_d] + dense_frame] = np.asarray(col0_dense, dtype=np.int64)
    utt = np.repeat(np.arange(n), span)
    first = np.zeros(len(col0), dtype=bool)
    first[frame_offsets[:-1][span > 0]] = True
    same = np.zeros(len(col0), dtype=bool)
    same[1:] = col0[1:] == col0[:-1]
    emit = (col0 != CTC_BLANK) & (first | ~same)
    tok, tu = col0[emit], utt[emit]
    rep = np.zeros(len(tok), dtype=bool)
    rep[1:] = (tok[1:] == tok[:-1]) & (tu[1:] == tu[:-1])
    return np.bincount(tu, minlength=n) + np.bincount(tu[rep], minlength=n)


def parakeet_rows(parakeet_root: Path, files: list[str]) -> pd.DataFrame:
    """One row per utterance of parakeet_out/<file>.{npz,jsonl} for every teacher file <source>/<stem> (the Parakeet
    pass mirrors the data shards, so the stems are the teacher's): id, p_hyp (TDT), p_ctc_hyp, n_frames, p_truncated,
    ctc_need (ctc_frames_needed). Only the CTC arrays of the npz are read. A teacher file with no Parakeet npz or
    jsonl raises: the label box runs both passes on every shard, so a missing one is a partly pulled root."""
    parts, missing = [], []
    for tf in files:
        npz, jl = parakeet_root / f"{tf}.npz", parakeet_root / f"{tf}.jsonl"
        if not (npz.is_file() and jl.is_file()):
            missing.append(tf)
            continue
        with np.load(npz) as zf:  # lazy per key: the TDT arrays are never decompressed
            z = {k: zf[k] for k in ("ids", "n_frames", "truncated", "frame_offsets", "dense_offsets",
                                    "ctc_dense_frame", "ctc_topk_idx")}
        ids = [str(x) for x in z["ids"]]
        jrows = read_jsonl(jl)
        if [r["id"] for r in jrows] != ids:
            raise ValueError(f"{npz}: jsonl ids differ from npz ids")
        col0 = z["ctc_topk_idx"][:, 0] if len(z["ctc_topk_idx"]) else np.zeros(0, np.int64)
        parts.append(pd.DataFrame(dict(
            id=ids, p_hyp=[r.get("hyp") or "" for r in jrows], p_ctc_hyp=[r.get("ctc_hyp") or "" for r in jrows],
            n_frames=z["n_frames"].astype(np.int64), p_truncated=z["truncated"].astype(bool),
            ctc_need=ctc_frames_needed(z["frame_offsets"], z["dense_offsets"], z["ctc_dense_frame"], col0))))
    if missing:
        raise ValueError(f"{len(missing)} teacher shard(s) have no parakeet_out npz + jsonl under {parakeet_root}: "
                         f"{missing[:10]}")
    cols = ["id", "p_hyp", "p_ctc_hyp", "n_frames", "p_truncated", "ctc_need"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def draw_budget(ids: list[str], durations, budget_s: float, seed: int, tag: str = "study:draw") -> set[str]:
    """A seeded subset of at most budget_s seconds, pooled over whatever `ids` holds: the ids sorted, permuted by
    (seed, tag), and the longest prefix of the permutation whose durations sum to <= budget_s (so it falls short of
    the budget by less than one utterance). Depends only on (seed, tag, the id set, their durations), never on file
    order. Raises ValueError when the pool itself is not larger than the budget: the extent is too small (a larger
    reazon_large cap), and taking everything would silently train on less."""
    order = np.argsort(np.asarray(ids, dtype=object), kind="stable")
    ids_sorted = [ids[i] for i in order]
    dur = np.asarray(durations, dtype=np.float64)[order]
    if dur.sum() <= budget_s:
        raise ValueError(f"the pool after all filters holds {dur.sum():.0f} s ({dur.sum() / 3600:.1f} h), not more "
                         f"than the draw budget {budget_s:.0f} s: raise the extent's cap (extent.inputs.reazon_large)")
    perm = np.random.default_rng([seed, zlib.crc32(tag.encode())]).permutation(len(ids_sorted))
    k = int(np.searchsorted(np.cumsum(dur[perm]), budget_s, side="right"))
    return {ids_sorted[i] for i in perm[:k]}


def corpus_cer(hyps, refs) -> dict:
    """kitsune.evaluate.corpus_cer, the scorer's formula, without that module's torch/transformers import (~45 s on the
    laptop): sum(S+D+I) / sum(ref chars) on normalize_ja text, empty references skipped and counted. A test pins the
    two equal."""
    import jiwer

    pairs = [(normalize_ja(h or ""), normalize_ja(r or "")) for h, r in zip(hyps, refs)]
    kept = [(h, r) for h, r in pairs if r]
    out = dict(cer=float("nan"), edits=0, ref_chars=0, n=len(kept), n_empty_ref=len(pairs) - len(kept))
    if kept:
        o = jiwer.process_characters([r for _, r in kept], [h for h, _ in kept])
        out["edits"] = int(o.substitutions + o.deletions + o.insertions)
        out["ref_chars"] = int(o.substitutions + o.deletions + o.hits)
        out["cer"] = out["edits"] / max(out["ref_chars"], 1)
    return out


def _utts_hours(df: pd.DataFrame) -> dict:
    return {"utts": int(len(df)), "hours": float(df["duration"].astype(np.float64).sum() / 3600)}


def _id_block(ids: list[str]) -> dict:
    return {"n": len(ids), "ids_sha256": ids_sha256(ids)}


def build_study_selection(teacher_root, second_root, parakeet_root, data_root, sources: list[str],
                          eval_sets: list[str], study: dict, *, agree_max: float = 0.5,
                          agree_max_by_source: dict[str, float] | None = None, seed: int = 1234, greedy_n: int = 500,
                          audio_check: bool = True, stems: dict[str, set[str]] | None = None, kotoba=None,
                          record: dict | None = None, capped: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """The study selection (module docstring): -> (selection, manifest, sidecar body). F0 is build_selection itself
    (the same second_out joins and thresholds, filter_eval_sets and partial_second_opinion empty); the study rules then
    overwrite `kept` rows only, in their order, so a row carries the first rule it fails. `kotoba`: the laptop's
    kotoba jsonl of the galgame hold-out (required when galgame is an eval set). `record` and `capped` (the extent
    record and the config's extent.inputs) add, per capped source, the pool hours had the cap been N (the rule for the
    final reazon_large cap)."""
    teacher_root, parakeet_root = Path(teacher_root), Path(parakeet_root)
    agree_max_by_source = agree_max_by_source or {}
    sel = build_selection(teacher_root, second_root, data_root, sources, eval_sets, agree_max, seed, greedy_n, 0,
                          audio_check, agree_max_by_source, (), (), stems)
    files = sorted(sel["teacher_file"].unique())
    tx = text_rows(teacher_root, files, ("hyp", "ref")).set_index("id")
    pk = parakeet_rows(parakeet_root, files)
    if pk["id"].duplicated().any():
        raise ValueError(f"duplicate ids across parakeet_out, e.g. {pk['id'][pk['id'].duplicated()].iloc[0]}")
    in_pk = sel["id"].isin(pk["id"]).to_numpy()
    pk = pk.set_index("id")
    hyp, ref = sel["id"].map(tx["hyp"]).to_numpy(object), sel["id"].map(tx["ref"]).to_numpy(object)
    p_hyp = sel["id"].map(pk["p_hyp"]).fillna("").to_numpy(object)
    p_ctc = sel["id"].map(pk["p_ctc_hyp"]).fillna("").to_numpy(object)
    n_frames = sel["id"].map(pk["n_frames"]).fillna(0).to_numpy(np.int64)
    ctc_need = sel["id"].map(pk["ctc_need"]).fillna(0).to_numpy(np.int64)
    train = (sel["split"] == "train").to_numpy()
    reason = sel["reason"].to_numpy(object).copy()

    def live():
        return train & (reason == "kept")

    reason[~in_pk] = "not_in_parakeet"  # rule 1, train and eval: the candidates are the rows in both roots
    stage = {"candidates": train & in_pk, "after_f0": live()}
    idx = np.flatnonzero(live())  # rule 3, F1a: normalised by Parakeet's length (Parakeet's hyp is the reference)
    f1a = np.full(len(sel), np.nan)
    f1a[idx] = [cer_fn(h, p) for h, p in zip(hyp[idx], p_hyp[idx])]
    reason[live() & (f1a > study["f1a_max"])] = "f1a_disagree"
    stage["after_f1a"] = live()
    held = {r for r in (normalize_ja(x) for x in ref[~train]) if len(r) >= study["dedup_min_chars"]}  # rule 4
    idx = np.flatnonzero(live())
    dup = np.zeros(len(sel), dtype=bool)
    dup[idx] = [normalize_ja(r) in held or normalize_ja(h) in held for r, h in zip(ref[idx], hyp[idx])]
    reason[dup] = "eval_dup"
    stage["after_dedup"] = live()
    reason[live() & (ctc_need > n_frames)] = "ctc_infeasible"  # rule 5
    stage["after_ctc"] = pool = live()
    drawn = draw_budget(sel["id"][pool].tolist(), sel["duration"][pool].to_numpy(), float(study["draw_audio_s"]),
                        seed)  # rule 6
    reason[pool & ~sel["id"].isin(drawn).to_numpy()] = "not_drawn"
    sel["reason"] = reason
    sel["keep"] = sel["reason"] == "kept"
    for (source, split), g in sel.groupby(["source", "split"], sort=False):
        chosen = seeded_subset(g["id"][g["keep"]].tolist(), study["probe_n"] if split == "train" else greedy_n, seed,
                               f"{split}:{source}")
        sel.loc[g.index, "in_probe"] = g["id"].isin(chosen) if split == "train" else False
        sel.loc[g.index, "in_greedy_subset"] = g["id"].isin(chosen) if split == "eval" else False
    sel["in_probe"], sel["in_greedy_subset"] = sel["in_probe"].astype(bool), sel["in_greedy_subset"].astype(bool)
    keep = sel["keep"].to_numpy()
    stage["drawn"] = train & keep

    # hours per source after each rule
    hours = {}
    for source in sources:
        src = (sel["source"] == source).to_numpy() & train
        hours[source] = {"teacher_out": _utts_hours(sel[src]),
                         **{k: _utts_hours(sel[src & m]) for k, m in stage.items()}}
    hours["total"] = {k: _utts_hours(sel[train & m]) for k, m in [("teacher_out", train), *stage.items()]}

    # the eval sets, the Galgame views and the teachers' baselines on them
    sets, texts = {}, {}
    for s in eval_sets:
        m = (sel["source"] == s).to_numpy() & ~train & keep
        sets[s] = dict(sel=m, ids=sel["id"][m].tolist())
    views = {}
    if "galgame" in eval_sets:
        m = sets["galgame"]["sel"]
        agree = sel["agree"].astype(np.float32).to_numpy()
        thr = np.float32(agree_max_by_source.get("galgame", agree_max))
        box = m & ~sel["truncated"].to_numpy(bool) & ~np.isnan(agree) & (agree <= thr)
        if kotoba is None:
            raise ValueError("galgame is an eval set: the neutral view needs the laptop's kotoba hypotheses "
                             f"({KOTOBA_GALGAME}, --kotoba-galgame)")
        k2 = {r["id"]: r.get("hyp2") for r in read_jsonl(Path(kotoba))}
        if absent := [i for i in sel["id"][m] if i not in k2]:
            raise ValueError(f"{len(absent)} galgame hold-out row(s) have no kotoba hypothesis in {kotoba} (e.g. "
                             f"{absent[0]}): the label box's galgame eval ids differ from the laptop's (check K10); "
                             f"the neutral view needs a kotoba decode of them")
        neutral = np.zeros(len(sel), dtype=bool)
        for i in np.flatnonzero(m):
            neutral[i] = bool(normalize_ja(ref[i])) and cer_fn(k2[sel["id"].iat[i]] or "", ref[i]) <= \
                study["neutral_max_cer"]
        views = {"neutral": neutral, "all": m, "label_box": box}
    keys = [(s, sets[s]["sel"]) for s in eval_sets if s != "galgame"] + [(f"galgame:{v}", m) for v, m in views.items()]
    baselines = {}
    for system, hyps in (("cohere", hyp), ("parakeet_tdt", p_hyp), ("parakeet_ctc", p_ctc)):
        b = {}
        for k, m in keys:
            c = corpus_cer(list(hyps[m]), list(ref[m]))
            b[k] = dict(c, cer=c["cer"] if np.isfinite(c["cer"]) else None)  # an empty set has no CER (not NaN)
        if all(b.get(k, {}).get("cer") is not None for k in kprereg.M4_SETS):
            b["m4"] = float(np.mean([b[k]["cer"] for k in kprereg.M4_SETS]))
        baselines[system] = b

    manifest = {"schema": STUDY_SCHEMA, "order": "selection order: eval_sets in config order, stems sorted, rows in "
                                                 "teacher_out order; ids_sha256 = kitsune.store.ids_sha256",
                "sets": {s: dict(_id_block(v["ids"]), hours=_utts_hours(sel[v["sel"]])["hours"], ids=v["ids"])
                         for s, v in sets.items()},
                "galgame_views": {v: dict(_id_block(sel["id"][m].tolist()), ids=sel["id"][m].tolist())
                                  for v, m in views.items()}}
    train_ids = sel["id"][stage["drawn"]].tolist()
    probe_ids = sel["id"][sel["in_probe"]].tolist()
    pool_s = float(sel["duration"][pool].astype(np.float64).sum())
    extra = {}
    if record is not None and capped:
        extra["pool_hours_if_capped"] = _pool_by_cap(record, capped, sel, pool, sources)
    q = f1a[~np.isnan(f1a)]
    sidecar = {
        "schema": STUDY_SCHEMA, "seed": seed, "sources": list(sources), "eval_sets": list(eval_sets),
        "hours": hours,
        "draw": {"budget_s": float(study["draw_audio_s"]), "pool_s": pool_s,
                 "drawn_s": float(sel["duration"][stage["drawn"]].astype(np.float64).sum()),
                 "pool_utts": int(pool.sum()), "drawn_utts": len(train_ids)},
        "n": {"train": len(train_ids), "probe": len(probe_ids), "eval": {s: len(v["ids"]) for s, v in sets.items()}},
        "ids_sha256": {"train": ids_sha256(train_ids), "probe": ids_sha256(probe_ids),
                       "eval": {s: ids_sha256(v["ids"]) for s, v in sets.items()}},
        "galgame_views": {v: dict(_id_block(sel["id"][m].tolist()),
                                  n_with_ref=int(sum(bool(normalize_ja(r)) for r in ref[m])))
                          for v, m in views.items()},
        "baselines": baselines,
        "details": {
            "reasons": {f"{s}/{sp}": {k: int(v) for k, v in g["reason"].value_counts().sort_index().items()}
                        for (s, sp), g in sel.groupby(["source", "split"], sort=False)},
            "f1a_quantiles": {str(p): float(np.quantile(q, p)) for p in (0.5, 0.9, 0.99)} if len(q) else {},
            "parakeet_only_rows": int((~pk.index.isin(sel["id"])).sum()),
            "parakeet_truncated": {"train_drawn": int(pk.loc[pk.index.isin(train_ids), "p_truncated"].sum()),
                                   "eval": int(pk.loc[pk.index.isin([i for v in sets.values() for i in v["ids"]]),
                                                      "p_truncated"].sum())},
            **extra},
    }
    return sel, manifest, sidecar


def _pool_by_cap(record: dict, capped: dict, sel: pd.DataFrame, pool: np.ndarray, sources: list[str]) -> dict:
    """source -> {N: the pool hours (every source, after all filters) had this source been capped at its first N
    inputs}, for each capped train source whose cap is an int (the reazon_large cap rule: the smallest N whose pool
    holds >= kitsune.prereg.POOL_MIN_HOURS)."""
    out = {}
    p = sel[pool]
    for src, cap in capped.items():
        if src not in sources or not isinstance(cap, int):
            continue
        ordinal = {st["stem"]: inp["ordinal"] for inp in record["sources"].get(src, {}).get("inputs", [])
                   for st in inp["stems"]}
        mine = p["source"] == src
        rest_h = float(p.loc[~mine, "duration"].astype(np.float64).sum() / 3600)
        per = p.loc[mine].assign(o=p.loc[mine, "teacher_file"].str.rsplit("/", n=1).str[1].map(ordinal))
        by = per.groupby("o")["duration"].sum().astype(np.float64) / 3600
        out[src] = {str(n): rest_h + float(by[by.index < n].sum()) for n in range(1, cap + 1)}
    return out


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    fsync_path(tmp)
    tmp.replace(path)


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
    ap.add_argument("--parakeet-out", default=None,
                    help="study only: default the --config's parakeet_root, else parakeet_out")
    ap.add_argument("--kotoba-galgame", default=None,
                    help=f"study only: the laptop's kotoba-whisper jsonl of the galgame hold-out, for the neutral view "
                         f"(default {KOTOBA_GALGAME} under the repo root)")
    ap.add_argument("--extent-record", default=None,
                    help="extent record of the --config's extent (default <extent.root>/extent.json)")
    ap.add_argument("--from-selection", default=None, metavar="PATH",
                    help="derive from this selection (no teacher_out, second_out or audio needed): the rows of the "
                         "recipe's sources and extent, reasons recomputed, seeded subsets redrawn")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--greedy-n", type=int, default=500, help="greedy-decode subset size per eval set")
    ap.add_argument("--probe-n", type=int, default=None,
                    help="train probe size per train source (default 500; a study recipe sets its own)")
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
        args.study = recipe.get("study")
        args.out = args.out or str(_repo_path(cfg.get("selection", "selection/viability.parquet")))
    elif not args.sources:
        ap.error("--sources is required (or --config)")
    else:
        args.eval_sets = EVAL_SETS if args.eval_sets is None else args.eval_sets
        args.agree_max = 0.5 if args.agree_max is None else args.agree_max
        args.agree_max_source, args.filter_eval_sets = args.agree_max_source or [], args.filter_eval_sets or []
        args.partial_second_opinion = args.partial_second_opinion or []
        args.study = None  # the study recipe lives in a run config only (vast/launch.py checks it there)
        args.out = args.out or str(ROOT / "selection" / "viability.parquet")
    # the roots: explicit flags win, then the config's (the viability config's are these defaults)
    for flag, key, default in (("teacher_out", "teacher_root", "teacher_out"),
                               ("second_out", "second_root", "second_out"), ("data", "data_root", "data"),
                               ("parakeet_out", "parakeet_root", "parakeet_out")):
        if getattr(args, flag) is None:
            setattr(args, flag, str(_repo_path(cfg.get(key) or default)))
    if args.study is not None:
        if problems := study_problems(args.study):
            ap.error(f"{args.config}: " + "; ".join(problems))
        if args.filter_eval_sets or args.partial_second_opinion:
            ap.error("a study recipe keeps every eval row in both roots (filter_eval_sets []) and judges every shard "
                     "(partial_second_opinion [])")
        if args.from_selection:
            ap.error("a study selection reads parakeet_out, which no selection records: build it from scratch")
        if args.probe_n is not None:
            ap.error(f"the study recipe sets probe_n ({args.study['probe_n']}); drop --probe-n")
        args.probe_n = int(args.study["probe_n"])
        args.kotoba_galgame = args.kotoba_galgame or str(ROOT / KOTOBA_GALGAME)
    else:
        args.probe_n = 500 if args.probe_n is None else args.probe_n

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
    ext, stems, record = kextent.extent_block(cfg), None, None
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
    elif args.study is not None:
        return study_main(args, cfg, by_source, stems, record, t0)
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


def study_main(args, cfg: dict, by_source: dict, stems, record, t0: float) -> int:
    """Build and write the study selection, its manifest and its sidecar (the sidecar last: it hashes the other two).
    No timestamp anywhere, so the same labels and arguments give the same three files byte for byte."""
    try:
        sel, manifest, body = build_study_selection(
            args.teacher_out, args.second_out, args.parakeet_out, args.data, args.sources, args.eval_sets, args.study,
            agree_max=args.agree_max, agree_max_by_source=by_source, seed=args.seed, greedy_n=args.greedy_n,
            audio_check=not args.skip_audio_check, stems=stems,
            kotoba=args.kotoba_galgame if "galgame" in args.eval_sets else None, record=record,
            capped=(args.extent or {}).get("inputs"))
    except ValueError as e:
        raise SystemExit(f"study selection: {e}")
    out = Path(args.out)
    sel_rel = cfg.get("selection") or out.name  # the repo-relative name the runs and the PREREG know it by
    man_out = out.parent / kprereg.MANIFEST_FILE
    summary = summarize(sel)
    kept = sel[sel["keep"]].groupby(["source", "split"])["duration"].agg(["size", "sum"])
    meta = dict(args=vars(args), n_no_audio_any=int(sel.attrs.get("n_no_audio_any", 0)),
                summary=summary.to_dict(orient="records"),
                kept={f"{s}/{sp}": dict(utts=int(r["size"]), hours=float(r["sum"] / 3600))
                      for (s, sp), r in kept.iterrows()})
    write_selection(sel, out, meta)
    _write_json(man_out, dict(manifest, selection=sel_rel))
    recipe = {k: cfg["selection_recipe"].get(k) for k in ("agree_max", "agree_max_source", "filter_eval_sets",
                                                          "partial_second_opinion", "study")}
    kotoba = Path(args.kotoba_galgame) if "galgame" in args.eval_sets else None
    sidecar = {"schema": body["schema"],
               "selection": {"path": sel_rel, "sha256": file_sha256(out)},
               "manifest": {"path": str(Path(sel_rel).parent / kprereg.MANIFEST_FILE).replace("\\", "/"),
                            "sha256": file_sha256(man_out)},
               "recipe": recipe, "extent": args.extent,
               "kotoba": None if kotoba is None else {"file": KOTOBA_GALGAME, "sha256": file_sha256(kotoba)},
               **{k: v for k, v in body.items() if k != "schema"}}
    side_out = out.with_suffix(".json")
    _write_json(side_out, sidecar)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.to_string(index=False, formatters=dict(hours="{:.3f}".format)))
    stages = list(body["hours"]["total"])
    print("\nhours per source after each rule: " + " / ".join(stages))
    for src, row in body["hours"].items():
        print(f"  {src:>13}: " + " / ".join(f"{row[k]['hours']:.2f}" for k in stages))
    d = body["draw"]
    views = ", ".join(f"{v} {b['n']}" for v, b in body["galgame_views"].items())
    print(f"draw: {d['drawn_s'] / 3600:.2f} h of a {d['pool_s'] / 3600:.2f} h pool (budget "
          f"{d['budget_s'] / 3600:.2f} h); train {body['n']['train']} utts, probe {body['n']['probe']}; eval "
          + ", ".join(f"{s} {n}" for s, n in body["n"]["eval"].items()) + (f"; galgame views {views}" if views else ""))
    print(f"\nwrote {out}, {man_out} and {side_out} ({len(sel)} rows) in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    main()
