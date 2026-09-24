"""Export one training run to a flat folder of parquet + CSV files with a README describing every file and column.

Why: the run dir is written for crash-safety (append-only jsonl, parquet parts, per-eval folders, TensorBoard event
files); analysis wants one table per kind. This folds everything into export/:
  - TensorBoard events (EventAccumulator: scalars, histograms, text) -> tb_*.parquet/csv. A resumed run's purged steps
    are dropped by the accumulator, exactly as TensorBoard shows them. TensorBoard files the tags under three buckets
    (1_operational/, 2_loss_accuracy/, 3_misc/); the tables give back the logged tag (`tag`) with `bucket` and
    `tb_tag` next to it, from metrics/tag_map.json (copied as tag_map.json, and as the tag_map table); a tag the map
    lacks (no tag_map.json: a run logged before the buckets, or a copy without the file) is mapped by the same rules
    from the tags of the open files
  - metrics/*: scalars (rebuilt from scalars.jsonl, the source of truth; the parquet mirror may lag one sync),
    steps, train_utts parts, hist parts, text
  - evals/step_<N>/*.parquet concatenated per kind with a step column (eval_tf, eval_greedy, eval_probe, ...), the
    mini evals' evals/step_<N>_mini/ likewise (eval_mini_tf, eval_mini_greedy, ...), summary.json files flattened to
    eval_summaries (a `mini` column; the eval's final and complete flags and its val-CER scope as columns),
    samples/*.jsonl -> samples, events.jsonl -> events
  - infra/ (box logs and state that vast/finish.py uploads: kitsune.log, supervise.json, bootstrap timings, the
    lifecycle events of the stop/destroy scripts) copied, its events and bootstrap timings also as tables
  - config.json (config.<stamp>.json of each restart), summary.json and env/ copied as they are
The open files are append-only, so after a crash and resume they keep the rows of the steps the resumed launch
replaced, and of those it never reached (a clipped time budget): a `discarded` column marks them (the resumes are read
from events.jsonl), so the open tables agree with TensorBoard and steps.
CSV is written next to each parquet unless the table exceeds --max-csv-rows (list columns become JSON strings).

Usage:
  python tools/export_run.py runs/viability-b20x2560 --out exports/viability-b20x2560
  python tools/export_run.py hf://<user>/<repo>/runs/viability-b20x2560 --out exports/viability-b20x2560
"""
import argparse
import json
import math
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

from kitsune.runlog import (TB_BUCKETS, TagMapper, _finite, load_tag_map, read_scalars_jsonl,  # noqa: E402
                            tag_map_entries, tb_inverse, tb_tag)

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
    "attempt": "trainer launch that logged the row: the restored full state's attempt + 1 (0 = first launch); rows of "
               "an earlier attempt with a step after a later resume's restored step come from weights a crash "
               "discarded (`discarded`)",
    "discarded": "the row comes from weights a crash discarded: a step after the restored step of a later resume, "
                 "logged before that resume's logger_start (events). By wall time (scalars, hist, text), by attempt "
                 "(train_utts: a resumed launch that died before its own full state shares its attempt with the next "
                 "launch, and their rows stay unmarked), by the eval_start / eval_mini events (eval tables, "
                 "eval_summaries, samples: no eval at that step after the resume). TensorBoard and steps drop these "
                 "rows",
    "emitter": "box script that wrote the record (finish, ...)", "phase": "bootstrap phase",
    "seconds": "wall time of the phase (s)", "end": "unix time the phase ended",
    "bucket": "TensorBoard bucket of the tag: 1_operational, 2_loss_accuracy or 3_misc",
    "tb_tag": "the tag as TensorBoard shows it (bucketed; metrics/tag_map.json)",
    "plugin": "TensorBoard plugin the tag is logged to: scalars, histograms or text",
    "unmapped": "no bucket rule matched the tag (it went to 3_misc)",
    "mini": "the row comes from a mini eval (evals/step_<N>_mini/), not a full one",
    "final": "the step's summary.json `final`: the end phase decoded this eval (false when the end phase reused an "
             "in-loop complete eval at the last step; verdict.json rows mark the final step either way); empty for a "
             "mini eval",
    "complete": "the step's summary.json `complete`: the eval decoded the complete eval sets (greedy_full/*), not "
                "only the fixed greedy subset; empty for a mini eval",
    "scope": "what the eval's headline/val_cer* pools (summary.json headline_scope.val_greedy): complete = the "
             "complete eval sets, subset = the fixed greedy subset (e.g. the step-0 eval of a full_every_epochs run); "
             "empty for a mini eval",
}
FILES = {
    "tb_scalars": "every TensorBoard scalar (mirror of scalars; purged steps dropped)",
    "tb_histograms": "every TensorBoard histogram (weights, grads, activations)",
    "tb_text": "every TensorBoard text entry (config, samples, events)",
    "scalars": "EVERY scalar ever logged, long format (from metrics/scalars.jsonl). Rows with discarded = true come "
               "from weights a crash discarded (TensorBoard hides them); the rest is what TensorBoard shows",
    "steps": "one wide row per optimizer step; NaN = not logged at that step",
    "train_utts": "one row per utterance per time it was trained on. Rows with discarded = true were trained on by "
                  "weights a crash discarded; the rest has one row per (step, utterance), unless a resumed launch "
                  "died before its own full state (see `discarded`)",
    "hist": "histogram summaries (quantiles, moments, 64-bin counts)",
    "text": "every text() call",
    "eval_summaries": "every eval summary.json flattened (mini evals too, mini = true): one row per (step, mini, "
                      "key) of each number in the file; its flags and scope (booleans and strings) are the final, "
                      "complete and scope columns on every row of that step instead; headline/<name> are the eval's "
                      "headline numbers (CER as fractions), headline/val_cer* over the eval sets `scope` names; NaN: "
                      "null in the file, a NaN or infinity (JSON has none) or a field with no value",
    "samples": "text samples written at each eval",
    "events": "lifecycle events (phases, smoke results, OOM fallbacks, checkpoints, exceptions, syncs, verdict)",
    "infra_events": "box lifecycle records from $KITSUNE_STATE/events.jsonl (sync failures, verification, stop/destroy)",
    "infra_bootstrap_timings": "vast/bootstrap.sh phase timings (plan, derived-data pull, audio rebuild, coverage)",
    "tag_map": "where TensorBoard shows each logged tag, one row per (tag, plugin) (metrics/tag_map.json; a tag it "
               "lacks, all of them for a run logged before the buckets or without the file, computed from "
               "kitsune.runlog.TB_BUCKET_RULES)",
}


def resolve(src: str, out: Path) -> tuple[Path, str | None]:
    """(run dir, Hub commit): a local run dir as is (commit None); hf://<user>/<repo>/runs/<id> is downloaded
    (checkpoints excluded) at the repo's current commit under out/_download/<commit[:12]>.

    The commit is looked up first and any Hub or auth error raises: when snapshot_download's own repo_info request
    fails (offline, a 429/5xx, an expired token) it returns a non-empty local_dir as it is, with only a logged warning
    (HF-X2), so a re-export into the same --out would rebuild every table from an earlier download (a mid-run look).
    One folder per commit: a new commit's folder is empty, so there is nothing stale to fall back to."""
    if not src.startswith("hf://"):
        p = Path(src)
        if not p.is_dir():
            sys.exit(f"{p} is not a directory")
        return p, None
    parts = src[len("hf://"):].strip("/").split("/")
    if len(parts) < 4 or parts[2] != "runs":
        sys.exit("expected hf://<user>/<repo>/runs/<run_id>")
    from huggingface_hub import HfApi, snapshot_download

    repo, prefix = "/".join(parts[:2]), "/".join(parts[2:])
    sha = HfApi().repo_info(repo, repo_type="model").sha
    local = Path(snapshot_download(repo, repo_type="model", revision=sha, local_dir=out / "_download" / sha[:12],
                                   allow_patterns=[f"{prefix}/*"], ignore_patterns=[f"{prefix}/checkpoints/*"]))
    return local / prefix, sha


def _from_tb(file_tag: str, plugin: str, inv: dict[str, dict[str, str]],
             fwd: dict[str, dict[str, str]] | None = None) -> tuple[str, str, str]:
    """Tag in the event files -> (logged tag, tb_tag, bucket), by the inverse (inv) and forward (fwd) tag map. Event
    files written before the buckets hold the logged tags themselves; the bucket columns then say where the bucketed
    layout (tools/regroup_tb.py) puts them."""
    tag = inv.get(plugin, {}).get(file_tag)
    if tag is not None:
        return tag, file_tag, file_tag.split("/", 1)[0]
    head = file_tag.split("/", 1)[0]
    if head in TB_BUCKETS:  # bucketed but in no map, nor made by the rules from a logged tag: the tag cannot be told
        return file_tag, file_tag, head
    tb = (fwd or {}).get(plugin, {}).get(file_tag) or tb_tag(file_tag, plugin)[0]
    return file_tag, tb, tb.split("/", 1)[0]


def tb_tables(tb_dir: Path, tag_map: dict | None = None) -> dict[str, pd.DataFrame]:
    """The event files as tables, each TensorBoard tag given back as the logged tag by tag_map (tag_map.json, or the
    fuller map of tag_mapper() that also covers a run without one)."""
    from tensorboard.backend.event_processing.event_accumulator import STORE_EVERYTHING_SIZE_GUIDANCE, EventAccumulator
    from tensorboard.util import tensor_util

    acc = EventAccumulator(str(tb_dir), size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    acc.Reload()
    tags = acc.Tags()
    inv, fwd = tb_inverse(tag_map or {}), {}
    for tag, plugin, e in tag_map_entries(tag_map or {}):
        fwd.setdefault(plugin, {})[tag] = e["tb_tag"]
    sc = [dict(zip(("tag", "tb_tag", "bucket"), _from_tb(t, "scalars", inv, fwd)), step=e.step,
               wall_time=e.wall_time, value=e.value) for t in tags.get("scalars", []) for e in acc.Scalars(t)]
    hi = [dict(zip(("tag", "tb_tag", "bucket"), _from_tb(t, "histograms", inv, fwd)), step=e.step,
               wall_time=e.wall_time, min=h.min, max=h.max, num=h.num, sum=h.sum, sum_squares=h.sum_squares,
               bucket_limits=json.dumps(list(h.bucket_limit)), bucket_counts=json.dumps(list(h.bucket)))
          for t in tags.get("histograms", []) for e in acc.Histograms(t) for h in [e.histogram_value]]
    tx = []
    for t in tags.get("tensors", []):
        if acc.SummaryMetadata(t).plugin_data.plugin_name != "text":
            continue
        # torch's SummaryWriter.add_text stores "<tag>/text_summary"; give back the tag the run logged
        names = dict(zip(("tag", "tb_tag", "bucket"), _from_tb(t.removesuffix("/text_summary"), "text", inv, fwd)))
        for e in acc.Tensors(t):
            arr = tensor_util.make_ndarray(e.tensor_proto)
            vals = [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in np.asarray(arr).ravel()]
            tx.append(dict(names, step=e.step, wall_time=e.wall_time, text="\n".join(vals)))
    return {"tb_scalars": pd.DataFrame(sc, columns=["tag", "bucket", "tb_tag", "step", "wall_time", "value"]),
            "tb_histograms": pd.DataFrame(hi, columns=["tag", "bucket", "tb_tag", "step", "wall_time", "min", "max",
                                                       "num", "sum", "sum_squares", "bucket_limits", "bucket_counts"]),
            "tb_text": pd.DataFrame(tx, columns=["tag", "bucket", "tb_tag", "step", "wall_time", "text"])}


OPEN_TAGS = (("scalars", "scalars"), ("hist", "histograms"), ("text", "text"))  # open-format table, plugin


def tag_mapper(tables: dict[str, pd.DataFrame], tag_map: dict) -> TagMapper:
    """The run's tag map (metrics/tag_map.json), plus where TB_BUCKET_RULES put every logged tag of the open files
    (scalars, hist, text, events/<kind>) that it lacks, in the order the tags were first logged: the fallback for a
    run logged before the buckets, or a copy without metrics/tag_map.json (its event files bucketed all the same)."""
    tm = TagMapper(tag_map)

    def add(tags, plugin: str):
        for t in tags:
            if tm.lookup(t, plugin) is None:
                tm.resolve(t, plugin)

    for name, plugin in OPEN_TAGS:
        df = tables.get(name)
        if df is not None and "tag" in df.columns:
            add(df["tag"].dropna().unique(), plugin)
    if "events" in tables:
        add((f"events/{k}" for k in tables["events"]["kind"].dropna().unique()), "text")
    return tm


def tag_tables(tables: dict[str, pd.DataFrame], tm: TagMapper) -> None:
    """bucket + tb_tag next to the tag of the open-format tables (scalars, hist, text), and the tag_map table: every
    entry of tm (tag_mapper(): the run's map, plus where the rules put a logged tag the map lacks)."""
    def entry(tag: str, plugin: str) -> dict:
        if tm.lookup(tag, plugin) is None:
            tm.resolve(tag, plugin)
        return tm.lookup(tag, plugin)

    for name, plugin in OPEN_TAGS:
        df = tables.get(name)
        if df is None or "tag" not in df.columns:
            continue
        look = {t: entry(t, plugin) for t in df["tag"].unique()}
        at = df.columns.get_loc("tag") + 1
        df.insert(at, "bucket", df["tag"].map({t: e["bucket"] for t, e in look.items()}))
        df.insert(at + 1, "tb_tag", df["tag"].map({t: e["tb_tag"] for t, e in look.items()}))
    rows = [dict(tag=tag, plugin=plugin, bucket=e["bucket"], tb_tag=e["tb_tag"], unmapped=bool(e.get("unmapped")))
            for tag, plugin, e in tag_map_entries(tm.to_json())]
    if rows:
        tables["tag_map"] = pd.DataFrame(rows).sort_values(["plugin", "bucket", "tb_tag"], ignore_index=True)


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
        elif v is None:  # a non-finite number (the logger writes it as null: JSON has no NaN) or no value: NaN
            out[key] = math.nan
    return out


def launches(ev: list[dict]) -> list[tuple[float, int | None, int]]:
    """(wall, restored step, attempt) of every trainer launch, from its logger_start event in events.jsonl (restored
    step None: a fresh start, attempt 0). attempt is the number the launch's train_utts rows carry: the restored full
    state's + 1, that state written by the launch running at the last `checkpoint` event of its full_step_<S> before
    this launch (a resumed launch that dies before its own full save leaves the next one restoring the same state, so
    both log the same attempt); a state with no such event is taken to be the previous launch's."""
    saves = [(r["wall"], r.get("name")) for r in ev if r.get("kind") == "checkpoint" and r.get("ckpt") == "full"]
    out = []
    for r in ev:
        if r.get("kind") != "logger_start":
            continue
        res = r.get("resume")
        if not (isinstance(res, dict) and "step" in res):
            out.append((r["wall"], None, 0))
            continue
        s = int(res["step"])
        at = [w for w, name in saves if name == f"full_step_{s}" and w < r["wall"]]
        writer = [a for w, _, a in out if at and w <= at[-1]]
        out.append((r["wall"], s, (writer[-1] if writer else out[-1][2] if out else 0) + 1))
    return out


def discarded(df: pd.DataFrame, runs: list[tuple[float, int | None, int]]) -> np.ndarray:
    """Rows logged by weights a crash discarded: a step after a later resume's restored step S, logged before that
    resume (wall < its logger_start), the rows TensorBoard's purge (purge_step S + 1) drops. The append-only files keep
    them, also for steps the resumed launch never reaches again (the budget re-fitted to the time left), so a dedup by
    (tag, step) cannot remove them. train_utts has no wall: its attempt tells the launch, an attempt that two launches
    share (see launches()) is left unmarked."""
    out = np.zeros(len(df), dtype=bool)
    if "step" not in df.columns:
        return out
    step = df["step"].to_numpy(dtype=float)
    if "wall" in df.columns:
        wall = df["wall"].to_numpy(dtype=float)
        for w, s, _ in runs:
            if s is not None:
                out |= (wall < w) & (step > s)
    elif "attempt" in df.columns:
        last = {a: i for i, (_, _, a) in enumerate(runs)}  # attempt -> the last launch that logs it
        by = df["attempt"].map(last).to_numpy(dtype=float)  # NaN (never < k): an attempt no launch accounts for
        for k, (_, s, _) in enumerate(runs):
            if s is not None:
                out |= (by < k) & (step > s)
    return out


def run_tables(run: Path) -> dict[str, pd.DataFrame]:
    t = {}
    ev = _jsonl(run / "events.jsonl")
    runs = launches(ev)
    evals_at: dict[tuple[bool, int], list[float]] = {}  # (mini, step) -> walls of the eval_start / eval_mini events
    for r in ev:
        if r.get("kind") in ("eval_start", "eval_mini") and r.get("at_step") is not None:
            evals_at.setdefault((r["kind"] == "eval_mini", int(r["at_step"])), []).append(r["wall"])

    def stale(step: int, mini: bool) -> bool:
        """evals/step_<N>[_mini]/ (and samples/step_<N>.jsonl) of weights a crash discarded: N after a resume's
        restored step and no eval at N started after that resume (a re-run overwrites the files by name)."""
        walls = evals_at.get((mini, step), [])
        return any(s is not None and step > s and not any(x > w for x in walls) for w, s, _ in runs)

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
    for kind in ("scalars", "train_utts", "hist", "text"):
        if kind in t:
            t[kind]["discarded"] = discarded(t[kind], runs)

    per_kind: dict[str, list[pd.DataFrame]] = {}
    summaries = []
    for d in sorted((run / "evals").glob("step_*"), key=lambda p: (int(p.name.split("_")[1]), p.name)):
        step = int(d.name.split("_")[1])
        mini = d.name.endswith("_mini")  # evals/step_<N>_mini/: a mini eval (eval_mini_<kind> tables)
        gone = stale(step, mini)
        for f in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(f)
            kind, _, eset = f.stem.partition("_")
            if kind not in ("tf", "greedy"):
                kind, eset = f.stem, ""
            df.insert(0, "step", step)
            if eset:
                df.insert(1, "set", eset)
            df["discarded"] = gone
            per_kind.setdefault(f"eval_{'mini_' if mini else ''}{kind}", []).append(df)
        objs = {f.name: json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.json"))}
        # a full eval's flags and val-CER scope, a bool and a string that _flatten leaves out, on every row of its dir
        # (verdict.json's too): step 0 decodes the greedy subset, a full_every_epochs run's later evals the complete
        # sets, and headline/val_cer* holds both on one curve. None for a mini eval (its `mini` column says so)
        s = objs.get("summary.json", {})
        marks = dict(final=s.get("final"), complete=s.get("complete"),
                     scope=(s.get("headline_scope") or {}).get("val_greedy"))
        for name, obj in objs.items():
            summaries += [dict(step=step, mini=mini, **marks, file=name, key=k, value=v, discarded=gone)
                          for k, v in _flatten(obj).items()]
    for k, dfs in per_kind.items():
        t[k] = pd.concat(dfs, ignore_index=True)
    if summaries:
        t["eval_summaries"] = pd.DataFrame(summaries)

    samples = []
    for f in sorted((run / "samples").glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
        n = int(f.stem.split("_")[1])
        samples += [dict(step=n, **r, discarded=stale(n, False)) for r in _jsonl(f)]
    if samples:
        t["samples"] = pd.DataFrame(samples)

    infra_ev = _jsonl(run / "infra" / "events.jsonl")
    if infra_ev:
        t["infra_events"] = pd.DataFrame([{"wall": r.get("wall"), "emitter": r.get("source"), "kind": r.get("kind"),
                                           "fields_json": json.dumps(_finite({k: v for k, v in r.items()
                                                                              if k not in ("wall", "source", "kind")}),
                                                                     ensure_ascii=False, default=str)}
                                          for r in infra_ev])
    timings = _jsonl(run / "infra" / "bootstrap_timings.jsonl")
    if timings:
        t["infra_bootstrap_timings"] = pd.DataFrame(timings)

    if ev:  # left unmarked: a crashed launch's lifecycle events are facts, not metrics
        base = ["wall", "time", "elapsed_s", "step", "kind"]
        t["events"] = pd.DataFrame([{**{k: r.get(k) for k in base},
                                     "fields_json": json.dumps(_finite({k: v for k, v in r.items() if k not in base}),
                                                               ensure_ascii=False, default=str)} for r in ev])
    return t


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
            out[c] = out[c].map(lambda v: json.dumps(_finite(v.tolist() if isinstance(v, np.ndarray) else v),
                                                     ensure_ascii=False))
    return out


def write_readme(out: Path, src: str, tables: dict[str, pd.DataFrame], copied: list[str], commit: str | None = None):
    lines = [f"# Run export: {Path(src).name if not src.startswith('hf://') else src}", "",
             f"Source: `{src}`  ", *([f"Hub commit: `{commit}`  "] if commit else []),
             f"Exported: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "Every table is a parquet file (read with `pandas.read_parquet`); a `.csv` twin exists unless noted. "
             "Losses are in nats per target token; CER values are fractions (0.083 = 8.3 %).", "",
             "## TensorBoard layout", "",
             "TensorBoard groups cards by the first component of a tag, so the run's event files put every tag under "
             "one of three buckets: `1_operational/` (time, throughput, memory, system, data progress with the tokens "
             "per source, schedule, early-stop bookkeeping, eval cost and counts with the CER denominators "
             "`ref_chars`; lifecycle events and the config as text), `2_loss_accuracy/` (first `00_summary/full/` and "
             "`00_summary/mini/`: every eval's headline numbers, the pooled val / train CER vs the reference and vs "
             "the teacher with `_pct` copies, val / train KL and top-1; then train and held-out loss (the train "
             "total and the L2-SP value under `train_loss_incl_l2sp/`: the value climbs all run as the weights "
             "leave the init, so judge training by `train_loss/objective`, `kl` and `ce`), "
             "accuracy as top-1 agreement with the teacher and CER, both per source and per teacher-confidence "
             "bucket, the train probe, the overfit gap, the early-stop metric and its best, the mini evals in "
             "`<section>_mini/`; the eval sample tables) "
             "and `3_misc/` (per-layer stats, L2-SP distances, optimizer internals, augmentation, token diagnostics "
             "with each confidence bucket's share of the tokens, every histogram, and any tag no rule matched). The "
             "tables here keep the tag as logged "
             "(`tag`); `tb_tag` is where TensorBoard shows it and `bucket` its first component. The run's "
             "`metrics/tag_map.json` (copied as `tag_map.json`: `{tag: {tb_tag, bucket, plugin}}`) holds the mapping, "
             "the `tag_map` table lists it; the rules are `TB_BUCKET_RULES` in kitsune/runlog.py. A run logged before "
             "the buckets has the logged tags in its event files; `tools/regroup_tb.py` rebuilds them in this layout.",
             ""]
    tm = tables.get("tag_map")
    if tm is not None and len(tm):
        counts = tm.groupby(["bucket", "plugin"]).size().unstack(fill_value=0)
        plugins = [p for p in ("scalars", "histograms", "text") if p in counts.columns]
        lines += ["Tags per bucket:", "", "| bucket | " + " | ".join(plugins) + " |", "|---|" + "---|" * len(plugins)]
        lines += [f"| `{b}` | " + " | ".join(str(int(counts.loc[b, p])) for p in plugins) + " |" for b in counts.index]
        unmapped = sorted(tm.loc[tm["unmapped"], "tag"])
        lines += ["", f"No rule matched (so in 3_misc): {', '.join(f'`{t}`' for t in unmapped) or 'none'}.", ""]
    for name, df in tables.items():
        desc = FILES.get(name) or (
            f"per-utterance mini-eval table `{name[10:]}` from every evals/step_<N>_mini/ (step, set added)"
            if name.startswith("eval_mini_") else
            f"per-utterance eval table `{name[5:]}` from every evals/step_<N>/ (step, set added)"
            if name.startswith("eval_") else "")
        csv = (out / f"{name}.csv").exists()
        lines += [f"## `{name}.parquet`{' / `' + name + '.csv`' if csv else ' (no CSV: too large)'}", "",
                  f"{desc}. {len(df):,} rows.", "", "| column | type | description |", "|---|---|---|"]
        for c in df.columns:
            d = COLUMNS.get(c) or ("per-utterance mean of that loss term" if name.startswith("eval_") else
                                   "logged metric (see the tag list)" if name == "steps" else "")
            lines.append(f"| `{c}` | {df[c].dtype} | {d} |")
        if name in ("scalars", "tb_scalars") and len(df):
            g = df.groupby("tag").agg(count=("step", "count"), min=("step", "min"), max=("step", "max"),
                                      bucket=("bucket", "first"), tb_tag=("tb_tag", "first"))
            lines += ["", f"Tags ({len(g)}):", "", "| tag | bucket | TensorBoard tag | rows | first step | last step |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| `{tag}` | {r['bucket']} | `{r['tb_tag']}` | {r['count']} | {r['min']} | {r['max']} |"
                      for tag, r in g.iterrows()]
        lines.append("")
    if copied:
        lines += ["## Copied as is", ""]
        descs = {"config.json": "resolved config, argv and student_meta.json", "summary.json": "final run summary and verdict",
                 "tag_map.json": "the run's metrics/tag_map.json: {logged tag: {tb_tag, bucket, plugin}}, where "
                                 "TensorBoard shows each tag (see TensorBoard layout above)",
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
    run, commit = resolve(src, out)
    opened = run_tables(run)
    # tag_map.json, completed from the open files' tags by the rules: a run without one still gets its logged tags back
    tm = tag_mapper(opened, load_tag_map(run / "metrics" / "tag_map.json"))
    tables = tb_tables(run / "tb", tm.to_json()) if (run / "tb").is_dir() else {}
    tables.update(opened)
    tag_tables(tables, tm)
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
    if (run / "metrics" / "tag_map.json").exists():
        shutil.copy2(run / "metrics" / "tag_map.json", out / "tag_map.json")
        copied.append("tag_map.json")
    for d in ("env", "infra"):
        if (run / d).is_dir():
            shutil.copytree(run / d, out / d, dirs_exist_ok=True)
            copied.append(d)
    write_readme(out, src, tables, copied, commit)
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
