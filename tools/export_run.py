"""Export one training run to a flat folder of parquet + CSV files with a README describing every file and column.

Why: the run dir is written for crash-safety (append-only jsonl, parquet parts, per-eval folders, TensorBoard event
files); analysis wants one table per kind. This folds everything into export/:
  - TensorBoard events (EventAccumulator: scalars, histograms, text) -> tb_*.parquet/csv. A resumed run's purged steps
    are dropped by the accumulator, exactly as TensorBoard shows them.
  - metrics/*: scalars (rebuilt from scalars.jsonl, the source of truth; the parquet mirror may lag one sync),
    steps, train_utts parts, hist parts, text
  - evals/step_<N>/*.parquet concatenated per kind with a step column (eval_tf, eval_greedy, eval_probe, ...),
    summary.json files flattened to eval_summaries, samples/*.jsonl -> samples, events.jsonl -> events
  - infra/ (box logs and state that vast/finish.py uploads: kitsune.log, supervise.json, bootstrap timings, the
    lifecycle events of the stop/destroy scripts) copied, its events and bootstrap timings also as tables
  - config.json (config.<stamp>.json of each restart), summary.json and env/ copied as they are
CSV is written next to each parquet unless the table exceeds --max-csv-rows (list columns become JSON strings).

Usage:
  python tools/export_run.py runs/viability-b20x2560 --out exports/viability-b20x2560
  python tools/export_run.py hf://<user>/<repo>/runs/viability-b20x2560 --out exports/viability-b20x2560
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from kitsune.runlog import read_scalars_jsonl  # noqa: E402

# column descriptions for the README; anything not listed is described generically
COLUMNS = {
    "step": "optimizer step (0 = before training)", "wall": "unix time (s) when logged",
    "wall_time": "unix time (s) of the TensorBoard event", "elapsed_s": "run clock (s), continues across resumes",
    "tag": "metric name", "value": "metric value (NaN/inf preserved)", "text": "text payload (markdown)",
    "min": "minimum", "max": "maximum", "num": "number of values", "sum": "sum of values",
    "sum_squares": "sum of squared values", "bucket_limits": "JSON list: right edge of each histogram bucket",
    "bucket_counts": "JSON list: count per bucket", "n": "number of finite values", "n_nonfinite": "NaN/inf count",
    "mean": "mean of finite values", "std": "population std of finite values",
    "p1": "1st percentile", "p5": "5th percentile", "p25": "25th percentile", "p50": "median", "p75": "75th percentile",
    "p95": "95th percentile", "p99": "99th percentile", "counts": "JSON list: 64-bin counts over [min, max]",
    "edges": "JSON list: 65 bin edges", "epoch": "epoch index", "id": "utterance id (globally unique)",
    "source": "dataset source / eval set", "set": "eval set (from the file name)", "duration": "audio seconds",
    "n_tok": "target tokens incl. EOS (teacher-forced) / generated tokens incl. EOS (greedy)",
    "kl": "17-bin forward KL(teacher||student), nats per token", "ce": "CE of the teacher greedy token, nats per token",
    "top1": "fraction of tokens where student argmax == teacher greedy token", "top1_acc": "same as top1",
    "masked_frac": "SpecAugment masked-frame fraction", "agree": "02b agreement CER (teacher vs second opinion)",
    "teacher_p1": "mean teacher top-1 probability", "ref": "dataset reference transcript", "teacher_hyp": "teacher hypothesis",
    "hyp": "student greedy hypothesis", "cer_ref": "per-utterance CER vs reference (normalize_ja)",
    "cer_teacher": "per-utterance CER vs teacher hypothesis", "truncated": "no EOS generated (max length or repetition)",
    "teacher_cer": "teacher CER vs reference (teacher_out jsonl)", "teacher_truncated": "teacher row had no EOS",
    "hyp_ids": "generated token ids incl. EOS", "has_teacher": "teacher row found for this id",
    "kind": "event kind", "time": "ISO-8601 UTC time", "fields_json": "all other event fields as JSON",
    "key": "flattened summary key (set/metric)", "file": "source file inside the run dir",
    "student_entropy_coarse": "student entropy over the 17 coarse bins", "teacher_entropy_coarse": "teacher entropy over the 17 coarse bins",
    "student_tail": "student mass outside the teacher top-16", "teacher_tail": "teacher mass outside its top-16",
    "attempt": "trainer launch that logged the row (0 = first, +1 per resume); rows of an earlier attempt with a step "
               "after the next resume's at_step (events) come from weights a crash discarded",
    "emitter": "box script that wrote the record (finish, ...)", "phase": "bootstrap phase",
    "seconds": "wall time of the phase (s)", "end": "unix time the phase ended",
}
FILES = {
    "tb_scalars": "every TensorBoard scalar (mirror of scalars; purged steps dropped)",
    "tb_histograms": "every TensorBoard histogram (weights, grads, activations)",
    "tb_text": "every TensorBoard text entry (config, samples, events)",
    "scalars": "EVERY scalar ever logged, long format (from metrics/scalars.jsonl). After a resume, steps after the "
               "restored step occur twice: keep the row with the larger wall",
    "steps": "one wide row per optimizer step; NaN = not logged at that step",
    "train_utts": "one row per utterance per time it was trained on. After a resume, the steps replayed since the "
                  "restored full state occur twice: keep the rows of the larger attempt",
    "hist": "histogram summaries (quantiles, moments, 64-bin counts)",
    "text": "every text() call",
    "eval_summaries": "every eval summary.json flattened: one row per (step, key)",
    "samples": "text samples written at each eval",
    "events": "lifecycle events (phases, smoke results, OOM fallbacks, checkpoints, exceptions, syncs, verdict)",
    "infra_events": "box lifecycle records from $KITSUNE_STATE/events.jsonl (sync failures, verification, stop/destroy)",
    "infra_bootstrap_timings": "vast/bootstrap.sh phase timings (plan, derived-data pull, audio rebuild, coverage)",
}


def resolve(src: str, out: Path) -> Path:
    """Local run dir as is; hf://<user>/<repo>/runs/<id> is downloaded (checkpoints excluded) under out/_download."""
    if not src.startswith("hf://"):
        p = Path(src)
        if not p.is_dir():
            sys.exit(f"{p} is not a directory")
        return p
    parts = src[len("hf://"):].strip("/").split("/")
    if len(parts) < 4 or parts[2] != "runs":
        sys.exit("expected hf://<user>/<repo>/runs/<run_id>")
    from huggingface_hub import snapshot_download

    repo, prefix = "/".join(parts[:2]), "/".join(parts[2:])
    local = Path(snapshot_download(repo, repo_type="model", local_dir=out / "_download",
                                   allow_patterns=[f"{prefix}/*"], ignore_patterns=[f"{prefix}/checkpoints/*"]))
    return local / prefix


def tb_tables(tb_dir: Path) -> dict[str, pd.DataFrame]:
    from tensorboard.backend.event_processing.event_accumulator import STORE_EVERYTHING_SIZE_GUIDANCE, EventAccumulator
    from tensorboard.util import tensor_util

    acc = EventAccumulator(str(tb_dir), size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    acc.Reload()
    tags = acc.Tags()
    sc = [dict(tag=t, step=e.step, wall_time=e.wall_time, value=e.value) for t in tags.get("scalars", [])
          for e in acc.Scalars(t)]
    hi = [dict(tag=t, step=e.step, wall_time=e.wall_time, min=h.min, max=h.max, num=h.num, sum=h.sum,
               sum_squares=h.sum_squares, bucket_limits=json.dumps(list(h.bucket_limit)), bucket_counts=json.dumps(list(h.bucket)))
          for t in tags.get("histograms", []) for e in acc.Histograms(t) for h in [e.histogram_value]]
    tx = []
    for t in tags.get("tensors", []):
        if acc.SummaryMetadata(t).plugin_data.plugin_name != "text":
            continue
        for e in acc.Tensors(t):
            arr = tensor_util.make_ndarray(e.tensor_proto)
            vals = [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in np.asarray(arr).ravel()]
            # torch's SummaryWriter.add_text stores "<tag>/text_summary"; give back the tag the run logged
            tx.append(dict(tag=t.removesuffix("/text_summary"), step=e.step, wall_time=e.wall_time, text="\n".join(vals)))
    return {"tb_scalars": pd.DataFrame(sc, columns=["tag", "step", "wall_time", "value"]),
            "tb_histograms": pd.DataFrame(hi, columns=["tag", "step", "wall_time", "min", "max", "num", "sum",
                                                       "sum_squares", "bucket_limits", "bucket_counts"]),
            "tb_text": pd.DataFrame(tx, columns=["tag", "step", "wall_time", "text"])}


def _read_parts(d: Path) -> pd.DataFrame | None:
    parts = sorted(d.glob("part-*.parquet"))
    if not parts:
        return None
    return pa.concat_tables([pq.read_table(p) for p in parts], promote_options="permissive").to_pandas()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a torn tail from a killed process
    return rows


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def run_tables(run: Path) -> dict[str, pd.DataFrame]:
    t = {}
    m = run / "metrics"
    if (m / "scalars.jsonl").exists():
        t["scalars"] = read_scalars_jsonl(m / "scalars.jsonl").to_pandas()
    elif (m / "scalars.parquet").exists():
        t["scalars"] = pd.read_parquet(m / "scalars.parquet")
    if (m / "steps.parquet").exists():
        t["steps"] = pd.read_parquet(m / "steps.parquet")
    for kind in ("train_utts", "hist"):
        df = _read_parts(m / kind)
        if df is not None:
            t[kind] = df
    text = _jsonl(m / "text.jsonl")
    if text:
        t["text"] = pd.DataFrame(text)

    per_kind: dict[str, list[pd.DataFrame]] = {}
    summaries = []
    for d in sorted((run / "evals").glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
        step = int(d.name.split("_")[1])
        for f in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(f)
            kind, _, eset = f.stem.partition("_")
            if kind not in ("tf", "greedy"):
                kind, eset = f.stem, ""
            df.insert(0, "step", step)
            if eset:
                df.insert(1, "set", eset)
            per_kind.setdefault(f"eval_{kind}", []).append(df)
        for f in sorted(d.glob("*.json")):
            flat = _flatten(json.loads(f.read_text(encoding="utf-8")))
            summaries += [dict(step=step, file=f.name, key=k, value=v) for k, v in flat.items()]
    for k, dfs in per_kind.items():
        t[k] = pd.concat(dfs, ignore_index=True)
    if summaries:
        t["eval_summaries"] = pd.DataFrame(summaries)

    samples = []
    for f in sorted((run / "samples").glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
        samples += [dict(step=int(f.stem.split("_")[1]), **r) for r in _jsonl(f)]
    if samples:
        t["samples"] = pd.DataFrame(samples)

    infra_ev = _jsonl(run / "infra" / "events.jsonl")
    if infra_ev:
        t["infra_events"] = pd.DataFrame([{"wall": r.get("wall"), "emitter": r.get("source"), "kind": r.get("kind"),
                                           "fields_json": json.dumps({k: v for k, v in r.items()
                                                                      if k not in ("wall", "source", "kind")},
                                                                     ensure_ascii=False, default=str)}
                                          for r in infra_ev])
    timings = _jsonl(run / "infra" / "bootstrap_timings.jsonl")
    if timings:
        t["infra_bootstrap_timings"] = pd.DataFrame(timings)

    ev = _jsonl(run / "events.jsonl")
    if ev:
        base = ["wall", "time", "elapsed_s", "step", "kind"]
        t["events"] = pd.DataFrame([{**{k: r.get(k) for k in base},
                                     "fields_json": json.dumps({k: v for k, v in r.items() if k not in base},
                                                               ensure_ascii=False, default=str)} for r in ev])
    return t


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
            out[c] = out[c].map(lambda v: json.dumps(v.tolist() if isinstance(v, np.ndarray) else v, ensure_ascii=False))
    return out


def write_readme(out: Path, src: str, tables: dict[str, pd.DataFrame], copied: list[str]):
    lines = [f"# Run export: {Path(src).name if not src.startswith('hf://') else src}", "",
             f"Source: `{src}`  ", f"Exported: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "Every table is a parquet file (read with `pandas.read_parquet`); a `.csv` twin exists unless noted. "
             "Losses are in nats per target token; CER values are fractions (0.083 = 8.3 %).", ""]
    for name, df in tables.items():
        desc = FILES.get(name) or (f"per-utterance eval table `{name[5:]}` from every evals/step_<N>/ (step, set added)"
                                   if name.startswith("eval_") else "")
        csv = (out / f"{name}.csv").exists()
        lines += [f"## `{name}.parquet`{' / `' + name + '.csv`' if csv else ' (no CSV: too large)'}", "",
                  f"{desc}. {len(df):,} rows.", "", "| column | type | description |", "|---|---|---|"]
        for c in df.columns:
            d = COLUMNS.get(c) or ("per-utterance mean of that loss term" if name.startswith("eval_") else
                                   "logged metric (see the tag list)" if name == "steps" else "")
            lines.append(f"| `{c}` | {df[c].dtype} | {d} |")
        if name in ("scalars", "tb_scalars") and len(df):
            g = df.groupby("tag")["step"].agg(["count", "min", "max"])
            lines += ["", f"Tags ({len(g)}):", "", "| tag | rows | first step | last step |", "|---|---|---|---|"]
            lines += [f"| `{tag}` | {r['count']} | {r['min']} | {r['max']} |" for tag, r in g.iterrows()]
        lines.append("")
    if copied:
        lines += ["## Copied as is", ""]
        descs = {"config.json": "resolved config, argv and student_meta.json", "summary.json": "final run summary and verdict",
                 "env": "git SHA/diff, pip freeze, nvidia-smi, system.json (host, versions, vast env minus secrets), SDPA backends",
                 "events.jsonl": "raw lifecycle events (one JSON object per line)",
                 "infra": "box logs and state from the vast instance: kitsune.log (bootstrap, supervisor and trainer "
                          "output), watchdog.log, supervise.json (attempts and the final decision), "
                          "bootstrap_timings.jsonl, bootstrap_coverage.json, events.jsonl (finish.py), halt, deadline"}
        restart = "config written by a restart of the trainer (resume); config.json is the first launch's"
        lines += [f"- `{c}`: {descs.get(c) or (restart if c.startswith('config.') else '')}" for c in copied]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(src: str, out: Path, max_csv_rows: int = 2_000_000) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    run = resolve(src, out)
    tables = {}
    if (run / "tb").is_dir():
        tables.update(tb_tables(run / "tb"))
    tables.update(run_tables(run))
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        if len(df) <= max_csv_rows:
            _csv_safe(df).to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
    copied = []
    restarts = sorted(p.name for p in run.glob("config.*.json"))
    for f in ("config.json", *restarts, "summary.json", "events.jsonl"):
        if (run / f).exists():
            shutil.copy2(run / f, out / f)
            copied.append(f)
    for d in ("env", "infra"):
        if (run / d).is_dir():
            shutil.copytree(run / d, out / d, dirs_exist_ok=True)
            copied.append(d)
    write_readme(out, src, tables, copied)
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run dir, or hf://<user>/<repo>/runs/<run_id>")
    ap.add_argument("--out", required=True, help="export folder (created)")
    ap.add_argument("--max-csv-rows", type=int, default=2_000_000, help="skip the CSV twin of larger tables")
    args = ap.parse_args(argv)
    tables = export(args.run, Path(args.out), args.max_csv_rows)
    for name, df in tables.items():
        print(f"  {name:16s} {len(df):>10,} rows")
    print(f"wrote {args.out}/README.md")


if __name__ == "__main__":
    main()
