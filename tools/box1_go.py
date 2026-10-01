"""Box 1's go / no-go for box 2, and the box-1 measurement box 2's hours take (plan v3 section 3; build contract 5:
items["m4-full-p01"].result.m4 < 0.1382; contract 7: box full's hours are refreshed from box 1's measured speed).

Plan v3's list, as this tool reads it from the runs repo (read-only; --hub-dir reads a local folder with the runs
repo's layout instead, for the tests or an offline copy):
  G1 ended      box p01 finished and self-destroyed with verified uploads: full/box-p01/queue_summary.json status
                complete with rc 0, full-p01 and m4-full-p01 done and verified, and the infra halt marker
                (full/box-p01/infra/<container>/halt, finish.py's last upload) says destroy. The Hub cannot prove the
                instance is gone: `vastai show instances` confirms it.
  G2 M4         a sane M4 readout: m4-full-p01's result.m4 below P-0.1B's study value, GO_M4 = 0.1382 (13.82 %; equal is
                no-go); m4_ratio, m4_teacher and the no-style JSUT CER printed with it.
  G3 dev curve  the dev curve and the early-stop decisions were logged: at least MIN_DEV_EVALS eval_dev events
                (early_stop.min_evals 5 smoothed over 5: 9 checks before the rule can decide), one at the start and a
                final one, every dev_ce finite, and early_stop/value scalars (the rule's view after each check);
                the early_stop event (reason, step, epoch), the end reason and dev_ce first / best / last printed.
  G4 measured   the real 5090 speed, setup times and peak RAM: the steady s/step (smoke verdict check 3's rule:
                time/step_s after the smoke phase's config smoke.steps, eval steps excluded, median), r = that / smoke
                A's P-0.1B s/step, the in-run overhead o (the item's wall less plan v3's fixed and dev-check seconds,
                over steps x s/step, minus 1), the CTC store's and the bootstrap's hours (the infra queue.json and
                bootstrap_timings.jsonl), and the peak RAM (the summary's host peak; on vast hosts the cgroup cannot
                be read there, so steps.parquet's sys/cgroup/anon_gb). Fails only when the speed cannot be measured.
  G5 smoke-B    full/box-smoke-b/smoke_verdict.json passed overall and every one of checks 12-16 (DECISIONS F2; launch
                checks that its commit ran the quant code box full will run).
GO when G1, G2, G3, G5 pass and G4 measured; exit 0 GO, 1 NO-GO, 2 not decidable yet (box 1 still running, a file not
on the Hub yet, the Hub unreadable). While box 1 trains, its scalars so far give a partial projection of its run.

It then projects box 2's hours with box 1's measurement (tools/make_full_configs.py box2_hours on this checkout's plan
and speed record) and lists what boxes.json would change; --json OUT writes {go, exit, lines, box1}, whose box1 object
`python tools/make_full_configs.py --import-speed --box1-go OUT` records (only for an ended box 1).

Usage:
  python tools/box1_go.py --cache-dir D:/kitsune-tmp/fullbuild/hubcache --json box1_go.json
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from kitsune import fullrun  # noqa: E402

RUNS_REPO = "Multy123/kitsune-runs"
BOX, TRAIN, READOUT = "p01", "full-p01", "m4-full-p01"
STORES = "stores-ctc"
GO_M4 = 0.1382  # P-0.1B's study M4 (contract 5); box 1 must beat it
MIN_DEV_EVALS = 9  # early_stop.min_evals 5 smoothed values over smooth 5 checks
SMOKE_STEPS_DEFAULT = 100  # the trainer's smoke.steps when the run config does not say
STUDENT = "p01"
EXIT_GO, EXIT_NOGO, EXIT_WAIT = 0, 1, 2
PASS, FAIL, WAIT, INFO = "PASS", "FAIL", "WAIT", "INFO"


class HubUnreadable(RuntimeError):
    """The runs repo could not be read (not a missing file: those are None)."""


class DirReader:
    """A local folder with the runs repo's layout."""

    def __init__(self, root):
        self.root = Path(root)

    def get(self, path: str) -> Path | None:
        p = self.root / path
        return p if p.is_file() else None


class HubReader:
    """The runs repo through huggingface_hub (read-only downloads into cache_dir); a missing file is None."""

    def __init__(self, repo: str = RUNS_REPO, cache_dir=None):
        self.repo, self.cache_dir = repo, cache_dir

    def get(self, path: str) -> Path | None:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError
        except ImportError as e:
            raise HubUnreadable(f"huggingface_hub is not installed ({e})") from None
        try:
            return Path(hf_hub_download(self.repo, path, cache_dir=self.cache_dir))
        except EntryNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001  network, auth, a missing repo: not decidable
            raise HubUnreadable(f"{self.repo}/{path}: {type(e).__name__}: {e}") from None


def read_json(reader, path: str):
    p = reader.get(path)
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def read_jsonl(reader, path: str) -> list[dict]:
    """A JSONL file's objects (a torn line is skipped); [] when it is missing."""
    p = reader.get(path)
    out = []
    for line in (p.read_text(encoding="utf-8", errors="replace").splitlines() if p is not None else []):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def scalar_series(rows: list[dict], tag: str) -> dict[int, float]:
    """scalars.jsonl's values of one tag by step (the newest row of a step wins, as after a resume)."""
    return {r["step"]: float(r["value"]) for r in rows
            if r.get("tag") == tag and isinstance(r.get("step"), int) and _finite(r.get("value"))}


def eval_steps(events: list[dict]) -> set[int]:
    """The steps an eval of any kind ran at (their step times include it): smoke check 3's exclusion."""
    out = set()
    for e in events:
        s = e.get("step", e.get("at_step"))
        if str(e.get("kind", "")).startswith("eval") and isinstance(s, (int, float)):
            out.add(int(s))
    return out


def steady_sec_per_step(rows: list[dict], events: list[dict], smoke_steps: int) -> tuple[float | None, int]:
    """kitsune.full_queue.SmokeVerdict.sec_per_step's rule: the median time/step_s of the steps after smoke.steps
    that ran no eval. -> (median or None, the number of steps it is over)."""
    ev = eval_steps(events)
    v = [x for s, x in scalar_series(rows, "time/step_s").items() if s > smoke_steps and s not in ev]
    return (statistics.median(v) if v else None), len(v)


def attempts_h(item: dict | None) -> float | None:
    """The wall hours of an infra queue.json item's attempts (t0..t1); None without a finished attempt."""
    spans = [a["t1"] - a["t0"] for a in (item or {}).get("attempts") or []
             if _finite(a.get("t0")) and _finite(a.get("t1"))]
    return sum(spans) / 3600 if spans else None


def peak_ram_gb(reader, run: str) -> float | None:
    """steps.parquet's peak sys/cgroup/anon_gb (the container's anonymous memory); None without pyarrow or the file."""
    p = reader.get(f"{run}/metrics/steps.parquet")
    if p is None:
        return None
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(p, columns=["sys/cgroup/anon_gb"])
    except Exception:  # noqa: BLE001  no pyarrow, no such column: the summary's numbers stand
        return None
    vals = [v for v in t.column(0).to_pylist() if _finite(v)]
    return max(vals) if vals else None


class Report:
    def __init__(self):
        self.lines: list[tuple[str, str, str]] = []
        self.box1: dict | None = None
        self.info: dict = {}

    def add(self, gid: str, status: str, text: str):
        self.lines.append((gid, status, text))

    def status(self, gid: str) -> str | None:
        st = [s for g, s, _ in self.lines if g == gid and s != INFO]
        return FAIL if FAIL in st else WAIT if WAIT in st else PASS if st else None

    @property
    def rc(self) -> int:
        st = [self.status(g) for g in ("G1", "G2", "G3", "G4", "G5")]
        return EXIT_WAIT if WAIT in st or None in st else EXIT_NOGO if FAIL in st else EXIT_GO


def _smoke_p01_sps(reader) -> float | None:
    """Smoke A's P-0.1B s/step: this checkout's speed record (make_full_configs SPEED_FILE), else the Hub's verdict."""
    try:
        import make_full_configs as M
        rec = M.load_speed()
        if rec is not None:
            return float(rec["smoke"]["sec_per_step"][STUDENT])
    except Exception:  # noqa: BLE001  fall back to the verdict
        pass
    v = read_json(reader, fullrun.box_verdict_path("full-smoke")) or {}
    ev = ((v.get("checks") or {}).get("3") or {}).get("evidence") or {}
    s = (ev.get("sec_per_step") or {}).get(f"smoke-{STUDENT}")
    return float(s) if _finite(s) else None


def analyse(reader) -> Report:
    """Every go / no-go line of box 1 (HubUnreadable propagates: not decidable)."""
    import make_full_configs as M

    rep = Report()
    summ = read_json(reader, fullrun.box_summary_path(BOX))
    if not isinstance(summ, dict):
        rep.add("G1", WAIT, f"no {fullrun.box_summary_path(BOX)} on the Hub: box 1 has not started its queue")
        _smoke_b(reader, rep)
        return rep
    items = summ.get("items") or {}
    tr, ro = items.get(TRAIN) or {}, items.get(READOUT) or {}
    run = tr.get("run_dir")  # "runs/<run_id>" once the trainer made it
    status, rc = summ.get("status"), summ.get("rc")
    cid = summ.get("container_id")
    infra = fullrun.box_infra_dir(BOX, cid) if cid else None
    rep.info.update(sha=summ.get("sha"), machine_id=summ.get("machine_id"), container_id=cid, run=run)

    # G1 ended. "failed" (queue exit 1) is not an end: the supervisor restarts the queue, which goes on
    ended = status in ("complete", "halted") and summ.get("ended") is not None
    if not ended:
        rep.add("G1", WAIT, f"box 1 is {status or 'not ended'} (stores-ctc {(items.get(STORES) or {}).get('status')}, "
                            f"{TRAIN} {tr.get('status')}, {READOUT} {ro.get('status')})")
    else:
        bad = []
        if status != "complete" or rc != 0:
            bad.append(f"status {status} rc {rc} ({summ.get('reason')})")
        for n, it in ((TRAIN, tr), (READOUT, ro)):
            if it.get("status") != "done" or it.get("verified") is not True:
                bad.append(f"{n} {it.get('status')} verified {it.get('verified')}")
        halt = read_json(reader, f"{infra}/halt") if infra else None
        if bad:
            rep.add("G1", FAIL, "; ".join(bad))
        elif not isinstance(halt, dict):
            rep.add("G1", WAIT, f"ended complete, rc 0, both items verified, but no {infra}/halt yet (finish's last "
                                f"upload); confirm with `vastai show instances`")
        elif halt.get("action") != "destroy":
            rep.add("G1", FAIL, f"finish ended the box with {halt.get('action')!r} ({halt.get('reason')}), not "
                                f"destroy: its disk is still billed (vastai show instances)")
        else:
            rep.add("G1", PASS, f"complete, rc 0, {TRAIN} and {READOUT} verified, halt destroy; confirm with `vastai "
                                f"show instances`")

    # G2 M4
    res = ro.get("result") if isinstance(ro.get("result"), dict) else {}
    m4 = res.get("m4")
    if ro.get("status") != "done":
        rep.add("G2", WAIT if not ended else FAIL, f"{READOUT} is {ro.get('status')}")
    elif not _finite(m4):
        rep.add("G2", FAIL, f"{READOUT} done without a finite m4 ({m4!r})")
    else:
        rep.add("G2", PASS if m4 < GO_M4 else FAIL,
                f"M4 {m4:.4f} {'<' if m4 < GO_M4 else '>='} {GO_M4} (P-0.1B's study value); m4_ratio "
                f"{res.get('m4_ratio')}, m4_teacher {res.get('m4_teacher')}, jsut_cer_nostyle "
                f"{res.get('jsut_cer_nostyle')}")

    # the run: dev curve, early stop, speed
    events = read_jsonl(reader, f"{run}/events.jsonl") if run else []
    rows = read_jsonl(reader, f"{run}/metrics/scalars.jsonl") if run else []
    rsum = read_json(reader, f"{run}/summary.json") if run else None
    rsum = rsum if isinstance(rsum, dict) else {}
    cfg = ((read_json(reader, f"{run}/config.json") or {}).get("config") or {}) if run else {}
    smoke_n = int(((cfg.get("smoke") or {}).get("steps")) or SMOKE_STEPS_DEFAULT)

    dev = [e for e in events if e.get("kind") == "eval_dev"]
    ce = [e.get("dev_ce") for e in dev]
    es_vals = scalar_series(rows, "early_stop/value")
    stops = [e for e in events if e.get("kind") == "early_stop"]
    for e in stops:
        rep.add("G3", INFO, f"early_stop at step {e.get('at_step', e.get('step'))}, epoch {e.get('epoch')}: "
                            f"{e.get('reason')} ({e.get('metric')} {e.get('value')}, best {e.get('best')} at step "
                            f"{e.get('best_step')}), action {e.get('action')}")
    if not run or not dev:
        rep.add("G3", WAIT if not ended else FAIL, f"no dev evals of {TRAIN} on the Hub yet" if not ended
                else f"{TRAIN} logged no eval_dev event")
    else:
        bad = []
        if len(dev) < MIN_DEV_EVALS:
            bad.append(f"{len(dev)} dev evals < {MIN_DEV_EVALS}")
        if not any(e.get("step") == 0 for e in dev):
            bad.append("no dev eval at the start")
        if ended and not any(e.get("final") for e in dev):
            bad.append("no final dev eval")
        if not all(_finite(v) for v in ce):
            bad.append(f"non-finite dev_ce at steps {[e.get('step') for e in dev if not _finite(e.get('dev_ce'))]}")
        if not es_vals:
            bad.append("no early_stop/value scalars (the early-stop rule's decisions)")
        fin = [v for v in ce if _finite(v)]
        text = (f"{len(dev)} dev evals, dev_ce first {fin[0]:.4f} best {min(fin):.4f} last {fin[-1]:.4f}" if fin
                else f"{len(dev)} dev evals") + (f"; {len(es_vals)} early-stop checks; end_reason "
                                                  f"{(tr.get('result') or {}).get('end_reason')}, stopped_early "
                                                  f"{bool(rsum.get('stopped_early'))}")
        rep.add("G3", FAIL if bad and ended else WAIT if bad else PASS, text + ("; " + "; ".join(bad) if bad else ""))

    # G4 measured
    sps, n_steps = steady_sec_per_step(rows, events, smoke_n)
    smoke_sps = _smoke_p01_sps(reader)
    plan = M.load_plan()
    total = M.plan_total_steps(STUDENT, plan)
    if sps is None:
        rep.add("G4", WAIT if not ended else FAIL, f"no steady time/step_s of {TRAIN} after step {smoke_n}"
                if run else f"{TRAIN} has no run dir yet")
    else:
        r = sps / smoke_sps if smoke_sps else None
        rep.add("G4", PASS if ended else WAIT,
                f"steady {sps:.4f} s/step over {n_steps} steps (smoke A {smoke_sps}: r {r:.3f})" if r is not None
                else f"steady {sps:.4f} s/step over {n_steps} steps (smoke A's P-0.1B s/step unknown)")
        if not ended:  # plan v3's run model (make_full_configs.box2_hours) at the speed so far
            dev_every = M.COMMON["eval"]["dev"]["every_epochs"]
            proj = (total * sps * (1 + M.OVERHEAD_PLAN) + M.FIXED_S[STUDENT]
                    + M.FULL_RUNS[STUDENT]["epochs"] / dev_every * M.DEV_CHECK_S[STUDENT]) / 3600
            last = max(scalar_series(rows, "time/step_s"), default=0)
            need = next((it.get("max_hours") for it in (fullrun.load_registry(root=ROOT)["boxes"][BOX].get("items")
                                                          or []) if it["name"] == TRAIN), None)
            rep.add("G4", INFO, f"partial: step {last} of {total}; at this speed {TRAIN} runs ~{proj:.2f} h at "
                                f"o {M.OVERHEAD_PLAN} (its registry need {need} h)")
    qj = read_json(reader, f"{infra}/queue.json") if infra else None
    qitems = (qj or {}).get("items") or {}
    stores_h, train_h = attempts_h(qitems.get(STORES)), attempts_h(qitems.get(TRAIN))
    timings = read_jsonl(reader, f"{infra}/bootstrap_timings.jsonl") if infra else []
    fb = reader.get(f"{infra}/first_boot") if infra else None
    ends = [t["end"] for t in timings if _finite(t.get("end"))]
    boot_h = rebuild_h = None
    if ends and fb is not None:
        try:
            boot_h = (max(ends) - float(fb.read_text(encoding="utf-8").strip())) / 3600
        except ValueError:
            boot_h = None
    if boot_h is None and timings:
        boot_h = sum(t.get("seconds") or 0 for t in timings if t.get("phase") != "pull_labels") / 3600
    rb = [t["seconds"] for t in timings if t.get("phase") == "rebuild_audio" and _finite(t.get("seconds"))]
    rebuild_h = sum(rb) / 3600 if rb else None
    wall_s = train_h * 3600 if train_h is not None else rsum.get("elapsed_s_total")
    steps = (tr.get("result") or {}).get("steps") or rsum.get("steps")
    epochs = (tr.get("result") or {}).get("epochs") or rsum.get("epochs")
    overhead = None
    if sps and _finite(wall_s) and isinstance(steps, int) and steps > 0 and _finite(epochs):
        dev_s = epochs / 0.1 * M.DEV_CHECK_S[STUDENT]
        o_raw = (wall_s - M.FIXED_S[STUDENT] - dev_s) / (steps * sps) - 1
        overhead = round(min(max(o_raw, 0.0), 0.99), 4)
        clamped = " (clamped to [0, 0.99])" if overhead != round(o_raw, 4) else ""
        rep.add("G4", INFO, f"{TRAIN}: {steps} steps, {epochs:.2f} epochs, wall {wall_s / 3600:.2f} h -> in-run "
                            f"overhead o {o_raw:.3f}{clamped}")
    rss = {n: it.get("peak_rss_gb") for n, it in items.items() if _finite(it.get("peak_rss_gb"))}
    host = summ.get("host_mem_peak_gb")
    anon = peak_ram_gb(reader, run) if run and not _finite(host) else None
    rep.add("G4", INFO, f"setup: bootstrap {boot_h if boot_h is None else round(boot_h, 2)} h (rebuild "
                        f"{rebuild_h if rebuild_h is None else round(rebuild_h, 2)} h), {STORES} "
                        f"{stores_h if stores_h is None else round(stores_h, 2)} h; RAM: host peak "
                        f"{host if _finite(host) else anon} GB"
                        f"{'' if _finite(host) else ' (steps.parquet sys/cgroup/anon_gb; no host peak on vast)'}"
                        f", peak RSS per item {rss}")
    if ended and sps is not None and rep.status("G1") == PASS:
        rep.box1 = dict(source=fullrun.box_summary_path(BOX), sha=summ.get("sha"), machine_id=summ.get("machine_id"),
                        container_id=cid, run_id=fullrun.run_id_of(run) if run else None, steps=steps, epochs=epochs,
                        sec_per_step=sps, smoke_sec_per_step=smoke_sps, overhead=overhead,
                        train_wall_h=None if not _finite(wall_s) else round(wall_s / 3600, 3),
                        stores_ctc_h=None if stores_h is None else round(stores_h, 3),
                        bootstrap_h=None if boot_h is None else round(boot_h, 3),
                        rebuild_h=None if rebuild_h is None else round(rebuild_h, 3),
                        host_mem_peak_gb=host if _finite(host) else anon, peak_rss_gb=rss, m4=m4)
    _smoke_b(reader, rep)
    return rep


def _smoke_b(reader, rep: Report):
    """G5: smoke-B's verdict passed overall and checks 12-16."""
    v = read_json(reader, fullrun.box_verdict_path("smoke-b"))
    if not isinstance(v, dict):
        rep.add("G5", FAIL, f"no {fullrun.box_verdict_path('smoke-b')}: run the standalone smoke-B")
        return
    checks = v.get("checks") if isinstance(v.get("checks"), dict) else {}
    vals = {n: (checks[n].get("pass") if isinstance(checks.get(n), dict) else "absent") for n in
            ("12", "13", "14", "15", "16")}
    ok = v.get("overall") == "pass" and all(x is True for x in vals.values())
    rep.add("G5", PASS if ok else FAIL, f"smoke-b at {str(v.get('sha'))[:12]} (machine {v.get('machine_id')}, "
                                        f"{v.get('time_utc')}): overall {v.get('overall')}, "
                                        + ", ".join(f"check {n} {x}" for n, x in vals.items()))


def box2_projection(box1: dict) -> list[str]:
    """Box full's hours with this box-1 measurement (make_full_configs.box2_hours on this checkout's plan and speed
    record) and what boxes.json would change."""
    import make_full_configs as M

    plan, rec = M.load_plan(), M.load_speed()
    if rec is None:
        return ["no speed record in this checkout: make_full_configs.py --import-speed --smoke-verdict ..."]
    rec = dict(rec, box1=box1)
    if problems := M.speed_problems(rec):
        return [f"box 1's measurement cannot serve box2_hours: {'; '.join(problems)}"]
    reg = json.loads((M.OUT_DIR / M.BOXES).read_text(encoding="utf-8"))
    box = reg["boxes"][M.BOX2]
    h = M.box2_hours(plan, rec, reserve_min=box.get("deadline_reserve_min") or 60)
    out = [f"box 2 with box 1's speed (r {h['r']:g}, o {h['o']:g}): "
           + ", ".join(f"{n} {v}" for n, v in h["items"].items())
           + f"; est {h['est_hours']:g} h, max {h['max_hours']} h (lanes {h['lanes']['A']:g} / {h['lanes']['B']:g}; "
           f"T-0.6B's pessimistic slack {h['t06_slack_h']:g} h); P-0.1B by the same model {h['run_h']['p01']:g} h vs "
           f"measured {box1.get('train_wall_h')} h"]
    todo = M.registry_drift(fullrun.load_registry(reg, root=ROOT, check_files=False), plan, rec)
    out += [f"boxes.json would change: {p}" for p in todo] or ["boxes.json already carries these hours"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-repo", default=RUNS_REPO)
    ap.add_argument("--cache-dir", default=None, help="huggingface_hub's cache for the downloads")
    ap.add_argument("--hub-dir", default=None, help="read a local folder with the runs repo's layout instead")
    ap.add_argument("--json", default=None, metavar="OUT",
                    help="write {go, exit, lines, box1} (box1: make_full_configs --import-speed --box1-go's input)")
    args = ap.parse_args(argv)
    reader = DirReader(args.hub_dir) if args.hub_dir else HubReader(args.runs_repo, args.cache_dir)
    try:
        rep = analyse(reader)
    except HubUnreadable as e:
        print(f"not decidable: the runs repo is not readable ({e})")
        rep, rc = None, EXIT_WAIT
    else:
        rc = rep.rc
        for gid, st, text in rep.lines:
            print(f"{gid} {st:4} {text}")
        verdict = {EXIT_GO: "GO: box 2 may launch (vast/README.md, box 2's order)",
                   EXIT_NOGO: "NO-GO: no box 2 until the failed lines are resolved",
                   EXIT_WAIT: "NOT DECIDABLE YET: run again when box 1 has ended and its records are on the Hub"}[rc]
        print(f"\n{verdict}")
        if rep.box1 is not None:
            for line in box2_projection(rep.box1):
                print(line)
    if args.json:
        out = {"go": None if rc == EXIT_WAIT else rc == EXIT_GO, "exit": rc,
               "lines": [dict(id=g, status=s, text=t) for g, s, t in (rep.lines if rep else [])],
               "box1": rep.box1 if rep else None}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
