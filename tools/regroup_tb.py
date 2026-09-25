"""Rebuild a run's TensorBoard event files in the three-bucket layout (1_operational / 2_loss_accuracy / 3_misc).

TensorBoard groups cards by the first component of a tag, and kitsune.runlog files every tag under one of three
buckets (TB_BUCKET_RULES). A run logged before that has its flat tags in tb/. This rewrites tb/ from what the run dir
keeps in open formats, each logged tag mapped by the same rules:
  metrics/scalars.jsonl     every scalar with its wall time (NaN/inf restored)
  metrics/hist/*.parquet    histograms (sum and sum of squares rebuilt from n, mean and std)
  metrics/text.jsonl        the config and the eval sample tables
  events.jsonl              lifecycle events (TensorBoard text events/<kind>)
Records go in the order the logger wrote them, and a resume is replayed as it happened: the restarted logger's purge
of the steps after the restored one (SessionLog.START) goes into the rebuilt file at that point, so TensorBoard shows
what it showed before. What never reached the open files is not rebuilt (histograms still buffered in a process that
was killed without close()). The rebuilt file starts with the Custom Scalars layout (kitsune.runlog.TB_LAYOUT: the
combined_loss train vs val chart), as every event file of the logger does; a run logged before it gets it too.

The original event files are kept under a name TensorBoard skips. TensorBoard reads every file whose name contains
"tfevents" (a plain .bak suffix would not hide one), so events.out.tfevents.X becomes events.out.tf-events.X.bak; to
undo, delete the *.regrouped file and rename the .bak files back. A second run deletes the *.regrouped file it wrote
before and rebuilds from the open files again: never two copies. metrics/tag_map.json is rewritten to the mapping used.

A run that looks live is refused unless --force, when any of
  - its last logger_start in events.jsonl has no later logger_close (or events.jsonl has none) and events.jsonl,
    logs/stdout.log, metrics/scalars.jsonl, metrics/text.jsonl or a tb/ event file changed in the last 5 minutes (a
    launch that has not logged a scalar yet, with an empty scalars.jsonl, included), or
  - its newest scalar is younger than 5 minutes and summary.json is absent, has no final status, or is older than that
    scalar (a resumed run keeps the failed launch's summary until it ends), or
  - summary.json says "failed", was written in the last 5 minutes, and the run has a full state to resume from
    (checkpoints/full_step_<N>/trainer.pt). Between a crashed launch's logger_close and the relaunch's logger_start
    the run is idle, not over: scripts/supervise_distill.py relaunches it with --resume about 2-3 minutes after the
    crash (without a full state it starts a new run dir instead), and the relaunch opens its tb/ event file seconds
    before its logger_start. The newest scalar's age does not tell: a crash in a long eval comes minutes after it.
--force does not skip the checks made around the rename. An event file that appeared in tb/ while the rebuild ran (a
launch starting; the rebuild takes seconds, more on a long run) makes the tool exit non-zero before it renames
anything. After the rename the originals are watched for 2 s, and if one grows, a writer still holds it open (on
Linux, and on Windows when TensorFlow is installed, renaming an open event file succeeds and the writer goes on writing
into the renamed .bak). Everything is then renamed back and the tool exits non-zero: the run is being written. A
writer that does not flush within those 2 s is not caught by it; the checks above are the guard. On Windows without
TensorFlow the writer's handle blocks the rename itself, with the same outcome.

A run logged before the buckets and resumed (or re-launched) with the bucketed logger has both layouts in tb/: the
earlier launches' flat tags and the new launch's buckets, side by side in TensorBoard. The logger leaves one
tb_layout_mixed event when that happens; run this tool on the run after it ends to rebuild tb/ in one layout.

TODO: every record of the run is read into memory before the rebuild; a run with millions of scalar rows needs a
streaming reader of scalars.jsonl and the hist parts.

Usage:
  python tools/regroup_tb.py runs/overfit-1s-20260924T074759Z
  python tools/regroup_tb.py --all [--runs-root runs]
"""
import argparse
import heapq
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from kitsune.runlog import TB_BUCKETS, TB_LAYOUT, TB_PLUGINS, TagMapper, _atomic_json, read_scalars_jsonl  # noqa: E402

FINAL_STATUS = ("complete", "failed", "throughput_too_low")  # 04_distill's summary.json statuses
LIVE_S = 300.0  # a write this recent makes a run live (see the module docstring)
SETTLE_S = 2.0  # how long the renamed originals are watched for a writer that still holds one open
ACTIVITY = ("events.jsonl", "logs/stdout.log", "metrics/scalars.jsonl", "metrics/text.jsonl")  # + tb/ event files
SUFFIX = ".regrouped"


class BeingWritten(SystemExit):
    """A writer holds an event file in tb/ open (an original, or one a launch opened during the rebuild); nothing was
    changed (or everything was renamed back)."""


def backup_name(name: str) -> str:
    return name.replace("tfevents", "tf-events") + ".bak"


def _jsonl(path: Path):
    """Complete lines only (a live file can end in half a line); a torn or bad line is skipped."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        data = f.read()
    for line in data[: data.rfind(b"\n") + 1].decode("utf-8", "replace").splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def last_scalar_wall(path: Path) -> float | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - (1 << 16)))
        data = f.read()
    for line in reversed(data[: data.rfind(b"\n") + 1].splitlines()):
        try:
            return float(json.loads(line)["wall"])
        except (ValueError, KeyError, TypeError):
            continue
    return None


def logger_open(run: Path) -> bool | None:
    """Whether the last logger_start in events.jsonl has no later logger_close; None when there is no logger_start."""
    last = None
    for r in _jsonl(run / "events.jsonl"):
        if r.get("kind") in ("logger_start", "logger_close"):
            last = r["kind"]
    return None if last is None else last == "logger_start"


def event_files(tb_dir: Path) -> list[Path]:
    """The event files a logger wrote in tb/: every *tfevents* file (TensorBoard reads those) but *.regrouped."""
    if not tb_dir.is_dir():
        return []
    return sorted(p for p in tb_dir.iterdir() if p.is_file() and "tfevents" in p.name and not p.name.endswith(SUFFIX))


def last_write(run: Path) -> tuple[float, str] | None:
    """(mtime, path in the run dir) of the newest of the files a running logger keeps writing."""
    times = []
    for p in [run / f for f in ACTIVITY] + event_files(run / "tb"):
        try:
            times.append((p.stat().st_mtime, p.relative_to(run).as_posix()))
        except OSError:
            continue
    return max(times) if times else None


def live_reason(run: Path, now: float | None = None) -> str | None:
    """Why the run looks live, None if it does not (the rules are in the module docstring)."""
    now = time.time() if now is None else now
    is_open, w = logger_open(run), last_write(run)
    if is_open is not False and w is not None and now - w[0] < LIVE_S:
        what = "events.jsonl has no logger_start" if is_open is None else "its last logger_start has no logger_close"
        return f"{what} and {w[1]} changed {now - w[0]:.0f} s ago"
    p, status, written = run / "summary.json", None, None
    try:
        written = p.stat().st_mtime
        status = json.loads(p.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError, AttributeError):
        pass
    # before the scalar-age rule: a launch can crash minutes after its last scalar (in a long eval)
    if (status == "failed" and now - written < LIVE_S
            and any((d / "trainer.pt").is_file() for d in (run / "checkpoints").glob("full_step_*"))):
        return (f"its launch failed {now - written:.0f} s ago and left a full state to resume from "
                f"(scripts/supervise_distill.py relaunches it)")
    last = last_scalar_wall(run / "metrics" / "scalars.jsonl")
    if last is None or now - last >= LIVE_S:
        return None
    age = f"its newest scalar is {now - last:.0f} s old"
    if written is None:
        return f"no summary.json and {age}"
    if status not in FINAL_STATUS:
        return f"summary.json status {status!r} is not final and {age}"
    if last > written + 1:
        return f"scalars were logged after summary.json (a resumed run) and {age}"
    return None


# ------------------------------------------------------------------------------------------------ records
# (wall, source, seq, what, payload): each source in the order it was written; heapq.merge interleaves them by wall
# and keeps every source's own order


def _scalars(run: Path):
    t = read_scalars_jsonl(run / "metrics" / "scalars.jsonl")
    cols = zip(t.column("wall").to_pylist(), t.column("tag").to_pylist(), t.column("step").to_pylist(),
               t.column("value").to_pylist())
    for i, (wall, tag, step, value) in enumerate(cols):
        yield wall, 3, i, "scalars", (tag, step, value)


def _hists(run: Path):
    parts = sorted((run / "metrics" / "hist").glob("part-*.parquet"))
    if not parts:
        return
    rows = pa.concat_tables([pq.read_table(p) for p in parts], promote_options="permissive").to_pylist()
    rows.sort(key=lambda r: r["wall"])  # a state_dict flush and a sync can write their parts out of order
    for i, r in enumerate(rows):
        counts = json.loads(r.get("counts") or "[]")
        if r["n"] and counts:  # the logger sends a histogram of no finite values to the parquet only
            yield r["wall"], 4, i, "histograms", r


def _texts(run: Path):
    for i, r in enumerate(_jsonl(run / "metrics" / "text.jsonl")):
        yield r["wall"], 1, i, "text", (r["tag"], r["step"], r["text"])


def _events(run: Path):
    """Every event as TensorBoard text events/<kind>, and before a resumed logger's logger_start its purge of the
    steps after the restored one."""
    i = 0
    for r in _jsonl(run / "events.jsonl"):
        resume = r.get("resume") if r.get("kind") == "logger_start" else None
        if isinstance(resume, dict) and "step" in resume:
            yield r["wall"], 2, i, "purge", int(resume["step"]) + 1
            i += 1
        line = json.dumps(r, ensure_ascii=False, default=str)
        yield r["wall"], 2, i, "text", (f"events/{r['kind']}", r["step"], "```\n" + line + "\n```")
        i += 1


def _write(run: Path, out_dir: Path) -> tuple[TagMapper, dict, list[str]]:
    from tensorboard.compat.proto.event_pb2 import Event, SessionLog
    from torch.utils.tensorboard import SummaryWriter

    tm, n, unmapped = TagMapper(), dict.fromkeys(("scalars", "histograms", "text", "purge"), 0), []
    w = SummaryWriter(log_dir=str(out_dir), filename_suffix=SUFFIX, max_queue=10_000, flush_secs=3600)
    try:
        # the Custom Scalars chart the logger writes into every event file (the originals' copies go with them into
        # the .bak files): at step 0, ahead of the records, so no replayed purge drops it
        w.add_custom_scalars(TB_LAYOUT)
        for wall, _, _, what, p in heapq.merge(_scalars(run), _hists(run), _texts(run), _events(run)):
            n[what] += 1
            if what == "purge":
                w._get_file_writer().add_event(Event(step=p, session_log=SessionLog(status=SessionLog.START)),
                                               walltime=wall)
                continue
            tag = p["tag"] if what == "histograms" else p[0]
            tb, warn = tm.resolve(tag, what)
            if warn:
                unmapped.append(f"{what}:{tag}")
            if what == "scalars":
                w.add_scalar(tb, p[2], int(p[1]), walltime=wall)
            elif what == "text":
                w.add_text(tb, p[2], int(p[1]), walltime=wall)
            else:
                num, mean, std, edges = int(p["n"]), float(p["mean"]), float(p["std"]), json.loads(p["edges"])
                w.add_histogram_raw(tb, min=p["min"], max=p["max"], num=num, sum=mean * num,
                                    sum_squares=(std * std + mean * mean) * num, bucket_limits=edges[1:],
                                    bucket_counts=json.loads(p["counts"]), global_step=int(p["step"]), walltime=wall)
    finally:
        w.close()
    return tm, n, unmapped


def _put_back(renamed: list[tuple[Path, Path]]) -> list[str]:
    """Rename the .bak files back to their original names; returns the ones that could not be."""
    stuck = []
    for p, bak in reversed(renamed):
        try:
            if p.exists():
                raise FileExistsError(f"{p.name} exists again")
            bak.rename(p)
        except OSError as e:
            stuck.append(f"{bak.name} ({e})")
    return stuck


def _grown(renamed: list[tuple[Path, Path]], sizes: dict[Path, int], settle_s: float) -> list[str]:
    """The originals whose size changed after the rename, watched for settle_s: a writer still holds them open."""
    end = time.monotonic() + settle_s
    while renamed:
        grown = []
        for p, bak in renamed:
            try:
                size = bak.stat().st_size
            except OSError:
                size = -1  # gone: whatever took it is not this tool
            if size != sizes[p]:
                grown.append(f"{p.name} ({sizes[p]:,} -> {size:,} bytes)")
        if grown or time.monotonic() >= end:
            return grown
        time.sleep(max(0.0, min(0.2, end - time.monotonic())))
    return []


def regroup(run, force: bool = False, settle_s: float | None = None) -> dict:
    """Rebuild <run>/tb in the bucketed layout; returns counts, the files written and kept, the unmapped tags.
    force skips the live check before the rebuild, never the one for a writer after the rename (BeingWritten)."""
    run = Path(run)
    settle_s = SETTLE_S if settle_s is None else settle_s
    if not (run / "metrics" / "scalars.jsonl").exists():
        raise SystemExit(f"{run}: no metrics/scalars.jsonl to rebuild from")
    reason = live_reason(run)
    if reason and not force:
        raise SystemExit(f"{run} looks live ({reason}); wait for it to end or pass --force")
    tb_dir = run / "tb"
    tb_dir.mkdir(exist_ok=True)
    checked = {p.name for p in event_files(tb_dir)}  # the event files there at the live check
    stage = Path(tempfile.mkdtemp(prefix=".regroup-", dir=run))  # same drive: the move into tb/ is a rename
    try:
        tm, n, unmapped = _write(run, stage)
        (new,) = [p for p in stage.iterdir() if "tfevents" in p.name]
        originals = event_files(tb_dir)
        opened = [p.name for p in originals if p.name not in checked]
        if opened:  # a launch started during the rebuild: renaming its open file would hide all it writes from now on
            raise BeingWritten(f"{run}: run is being written: {', '.join(opened)} appeared in tb/ during the rebuild, "
                               f"so a launch started; nothing was changed (--force does not skip this check)")
        renamed, sizes = [], {}
        try:
            for p in originals:
                bak = tb_dir / backup_name(p.name)
                while bak.exists():
                    bak = bak.with_name(bak.name + ".bak")
                sizes[p] = p.stat().st_size
                p.rename(bak)
                renamed.append((p, bak))
        except OSError as e:
            # Windows refuses the rename only while another process has the file open without FILE_SHARE_DELETE: the
            # pure-Python writer of tensorboard / torch when TensorFlow is absent, or a reader such as TensorBoard.
            # TensorFlow's writer (installed on the laptop) allows it, as does any writer on Linux, and then goes on
            # writing into the renamed file: _grown() below is what catches those.
            stuck = _put_back(renamed)
            raise BeingWritten(f"{run}: cannot rename {e.filename} ({e}): another process holds it open (a trainer "
                               f"writing it, or a TensorBoard reading it); nothing was changed"
                               + (f" but {', '.join(stuck)} could not be renamed back" if stuck else ""))
        grown = _grown(renamed, sizes, settle_s)
        if grown:
            stuck = _put_back(renamed)
            raise BeingWritten(f"{run}: run is being written: {', '.join(grown)} grew after the rename, so a writer "
                               f"still holds it open. The originals were renamed back"
                               + (f" except {', '.join(stuck)}, which could not be" if stuck else "")
                               + "; run this after the run ends (--force does not skip this check)")
        stale = [p for p in tb_dir.iterdir() if p.is_file() and p.name.endswith(SUFFIX) and "tfevents" in p.name]
        dest = tb_dir / new.name
        new.replace(dest)
        for p in stale:
            if p != dest:
                p.unlink()  # this tool's own earlier output, rebuilt just now from the same files
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    _atomic_json(run / "metrics" / "tag_map.json", tm.to_json())
    counts = {b: dict.fromkeys(TB_PLUGINS, 0) for b in TB_BUCKETS}
    for entry in tm.map.values():
        for plugin, e in [(entry["plugin"], entry), *(entry.get("other_plugins") or {}).items()]:
            counts[e["bucket"]][plugin] += 1
    return dict(run=run, file=dest, counts=counts, records=n, unmapped=unmapped, live=reason,
                backups=sorted(p.name for p in tb_dir.iterdir() if p.name.endswith(".bak")),
                charts=[chart for charts in TB_LAYOUT.values() for chart in charts])


def report(res: dict) -> str:
    n = res["records"]
    lines = [f"{res['run']}: rebuilt tb/{res['file'].name} from {n['scalars']:,} scalars, {n['histograms']:,} "
             f"histograms, {n['text']:,} text entries ({n['purge']} resume purge{'s' * (n['purge'] != 1)})"
             + (f" [forced: {res['live']}]" if res["live"] else ""),
             f"  {'bucket':18s}" + "".join(f"{p:>12s}" for p in TB_PLUGINS)]
    for b in TB_BUCKETS:
        lines.append(f"  {b:18s}" + "".join(f"{res['counts'][b][p]:>12,}" for p in TB_PLUGINS))
    lines.append(f"  {'tags':18s}" + "".join(f"{sum(c[p] for c in res['counts'].values()):>12,}" for p in TB_PLUGINS))
    lines.append(f"  no rule matched (-> 3_misc): {', '.join(res['unmapped']) or 'none'}")
    lines.append(f"  Custom Scalars charts: {', '.join(res['charts'])}")
    lines.append(f"  originals kept as: {', '.join(res['backups']) or 'none'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run dir (runs/<run_id>)")
    ap.add_argument("--all", action="store_true", help="every run under --runs-root (live ones are skipped)")
    ap.add_argument("--runs-root", default="runs", help="for --all (default: runs)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild a run that looks live (a file still being written is renamed back all the same)")
    args = ap.parse_args(argv)
    if bool(args.run) == args.all:
        ap.error("give a run dir or --all")
    if not args.all:
        print(report(regroup(args.run, force=args.force)))  # a refusal exits 1 with its message
        return 0
    runs = sorted(p for p in Path(args.runs_root).iterdir() if (p / "metrics" / "scalars.jsonl").exists())
    written = 0
    for run in runs:
        try:
            print(report(regroup(run, force=args.force)))
        except SystemExit as e:
            print(f"skipped: {e}")
            written += isinstance(e, BeingWritten)
    print(f"{len(runs)} runs under {args.runs_root}" + (f", {written} being written" if written else ""))
    return 1 if written else 0


if __name__ == "__main__":
    sys.exit(main())
