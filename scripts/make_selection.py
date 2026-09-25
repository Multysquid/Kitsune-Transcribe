"""Decide which teacher-labelled utterances a distillation run trains and evaluates on, and record why per row.

Why a separate, persisted selection: the label filter is a pre-registered decision of the run, the training box
never needs second_out's raw rows or the local shards to apply it, and every later number (hours trained, probe,
greedy subsets) must be traceable to one file. The selection is small (~2 MB) and is parked on HF with teacher_out.

Candidates come from teacher_out/<source>/<split>-*.npz ONLY (never data/manifest.jsonl, which also lists shards the
teacher has not labelled). Train sources use split `train`; eval sets use split `eval`.

Rules for TRAIN rows, first match wins (keep = reason == "kept"):
  truncated    the teacher never emitted EOS (repetition or length cut): its labels are garbage (median agree 0.9-3)
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

Usage:
  python scripts/make_selection.py --sources reazon_small --agree-max 0.5 --out selection/viability.parquet
"""
import argparse
from collections.abc import Sequence
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

from kitsune.store import fsync_path  # noqa: E402

EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon"]  # the pre-registered gate sets; monitor hold-outs are extra
COLUMNS = ["id", "source", "split", "teacher_file", "duration", "n_tok", "truncated", "agree", "teacher_cer", "keep",
           "reason", "in_greedy_subset", "in_probe"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_rows(teacher_root: Path, source: str, split: str, eos: int) -> pd.DataFrame:
    """One row per utterance with teacher output, in teacher_out order (stems sorted, row order within)."""
    parts = []
    for npz in sorted((teacher_root / source).glob(f"{split}-*.npz")):
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


def agree_map(second_root: Path, source: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for p in sorted((second_root / source).glob("*.jsonl")):
        for r in read_jsonl(p):
            out[r["id"]] = r.get("agree")
    return out


def audio_ids(data_root: Path, source: str, split: str) -> set[str]:
    """Ids present in the local shards; reads only the id column (the audio column is never touched)."""
    out: set[str] = set()
    for p in sorted((data_root / "shards" / source).glob(f"{split}-*.parquet")):
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
                    filter_eval_sets: Sequence[str] = ()) -> pd.DataFrame:
    """`agree_max_by_source` overrides `agree_max` per train source: each source's `agree` is measured against a
    different second model (whisper-large-v3 for reazon, Emilia's own Whisper-medium text for emilia_yodas), so one
    global threshold is not calibrated across sources."""
    teacher_root, second_root, data_root = Path(teacher_root), Path(second_root), Path(data_root)
    meta_p = teacher_root / "meta.json"
    eos = int(json.loads(meta_p.read_text(encoding="utf-8")).get("eos_token_id", 3)) if meta_p.exists() else 3
    agree_max_by_source = agree_max_by_source or {}
    frames = []
    for source, split in [(s, "train") for s in sources] + [(s, "eval") for s in eval_sets]:
        df = teacher_rows(teacher_root, source, split, eos)
        if df.empty:
            print(f"WARNING: no teacher output for {source}/{split}-*.npz")
            continue
        amap = agree_map(second_root, source)
        agree = df["id"].map(lambda x: amap.get(x)).astype("float64")  # None/missing -> NaN
        df["agree"] = agree.astype(np.float32)
        have_audio = df["id"].isin(audio_ids(data_root, source, split)) if audio_check else pd.Series(True, index=df.index)
        reason = np.full(len(df), "kept", dtype=object)
        reason[~have_audio.to_numpy()] = "no_audio"  # assigned lowest-precedence first, then overwritten
        if split == "train" or source in filter_eval_sets:
            thr = agree_max_by_source.get(source, agree_max)
            reason[(agree > thr).to_numpy()] = f"agree>{thr:g}"
            reason[agree.isna().to_numpy()] = "no_agree"
            reason[df["truncated"].to_numpy()] = "truncated"
        df["reason"] = reason
        df["keep"] = df["reason"] == "kept"
        kept = df["id"][df["keep"]].tolist()
        chosen = seeded_subset(kept, greedy_n if split == "eval" else probe_n, seed, f"{split}:{source}")
        df["in_greedy_subset"] = df["id"].isin(chosen) if split == "eval" else False
        df["in_probe"] = df["id"].isin(chosen) if split == "train" else False
        frames.append(df[COLUMNS])
    if not frames:
        raise SystemExit("no teacher output for any requested source")
    sel = pd.concat(frames, ignore_index=True)
    if sel["id"].duplicated().any():
        raise ValueError(f"duplicate ids across teacher_out, e.g. {sel['id'][sel['id'].duplicated()].iloc[0]}")
    return sel


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
    ap.add_argument("--sources", nargs="+", required=True, help="train sources (their teacher_out split `train`)")
    ap.add_argument("--eval-sets", nargs="*", default=EVAL_SETS, help="eval sources (split `eval`); none: --eval-sets")
    ap.add_argument("--agree-max", type=float, default=0.5, help="keep train rows with agree <= this")
    ap.add_argument("--agree-max-source", action="append", default=[], metavar="SOURCE=A",
                    help="per-source override of --agree-max (repeatable), e.g. emilia_yodas=0.3")
    ap.add_argument("--filter-eval-sets", nargs="*", default=[],
                    help="monitor-only eval sets that get the train label rules (never the pre-registered gate sets)")
    ap.add_argument("--out", default=str(ROOT / "selection" / "viability.parquet"))
    ap.add_argument("--teacher-out", default=str(ROOT / "teacher_out"))
    ap.add_argument("--second-out", default=str(ROOT / "second_out"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--greedy-n", type=int, default=500, help="greedy-decode subset size per eval set")
    ap.add_argument("--probe-n", type=int, default=500, help="train probe size per train source")
    ap.add_argument("--skip-audio-check", action="store_true", help="do not look for the audio (no `no_audio` rows)")
    args = ap.parse_args(argv)

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

    t0 = time.time()
    sel = build_selection(args.teacher_out, args.second_out, args.data, args.sources, args.eval_sets, args.agree_max,
                          args.seed, args.greedy_n, args.probe_n, not args.skip_audio_check, by_source,
                          args.filter_eval_sets)
    summary = summarize(sel)
    kept = sel[sel["keep"]].groupby(["source", "split"])["duration"].agg(["size", "sum"])
    meta = dict(args=vars(args), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
