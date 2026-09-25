"""Compare the two galgame judges on the stems both judged: the laptop's kotoba-whisper second opinions and the label
box's Parakeet ones, against the same (adopted) Cohere hypotheses.

Why: the box judges all of galgame with the Parakeet TDT hypothesis (02b --judge parakeet) instead of kotoba, and the
run configs keep agree_max 0.5 for it. Parakeet has the better galgame CER of the two (20.5 % vs 22.5 % in the
bake-off), so its agree runs tighter and 0.5 may keep more hours than kotoba@0.5 did. This report says how much, and
which Parakeet threshold drops the same hours as kotoba@0.5, so a different threshold is a config decision
(make_selection --from-selection on the laptop), not a relabel.

Inputs (jsonl per stem, as 02 and 02b write them):
  --kotoba DIR    seed/second_out/galgame             {id, hyp2, model2, agree, cer2}, the laptop's 58 judged stems
  --parakeet DIR  labels/full/second_out/galgame      the same, judged by Parakeet
  --teacher DIR   labels/full/teacher_out/galgame     {id, hyp, ref, cer, duration, ...}: hours and teacher-vs-ref CER
Only the stems in both --kotoba and --parakeet count, and only rows both judges gave an agree value.

Output (--out, default labels/full/reports/galgame_judge.json, written atomically):
  stems, rows, hours, null_agree {kotoba, parakeet}
  quantiles {kotoba, parakeet}: agree at q = 0.1 0.25 0.5 0.75 0.9 0.95
  dropped_hours {kotoba, parakeet}: {"0.2": h, ...}, dropped meaning agree > threshold (agree_max is inclusive)
  reference {judge: "kotoba", threshold, dropped_hours}; matched {threshold, dropped_hours}: the smallest Parakeet
    threshold whose dropped hours are closest to kotoba@threshold
  table: rows and hours kept/dropped by kotoba@threshold x Parakeet@threshold (the config's shared agree_max)
  spearman: the rank correlation of the two agree columns
  teacher_cer {kotoba, parakeet}: {kept, dropped} mean teacher-vs-ref CER (the sanity check: the dropped rows should
    be the worse labels whichever judge drops them)

Usage (the label box's finalize runs it from the repo root with the defaults):
  python tools/judge_report.py
  python tools/judge_report.py --kotoba seed/second_out/galgame --parakeet labels/full/second_out/galgame \
      --teacher labels/full/teacher_out/galgame --out labels/full/reports/galgame_judge.json
CPU only, no network.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.7)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
JUDGES = ("kotoba", "parakeet")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stems(d: Path) -> set[str]:
    return {p.stem for p in Path(d).glob("*.jsonl")}


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their positions), as Spearman's rho needs."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def dropped_hours(agree: np.ndarray, hours: np.ndarray, t: float) -> float:
    return float(hours[agree > t].sum())


def matched_threshold(agree: np.ndarray, hours: np.ndarray, target: float) -> float:
    """The smallest threshold on `agree` whose dropped hours come closest to `target`. Dropped hours only change at
    the observed values, so those (and 0) are the candidates."""
    best_t, best_d = 0.0, None
    for t in np.unique(np.concatenate([[0.0], agree])):
        d = abs(dropped_hours(agree, hours, float(t)) - target)
        if best_d is None or d < best_d - 1e-12:
            best_t, best_d = float(t), d
    return best_t


def mean_or_none(x: np.ndarray) -> float | None:
    return round(float(x.mean()), 4) if len(x) else None


def build_report(kotoba_dir: Path, parakeet_dir: Path, teacher_dir: Path, threshold: float = 0.5) -> dict:
    common = sorted(stems(kotoba_dir) & stems(parakeet_dir))
    if not common:
        raise SystemExit(f"no stem judged by both {kotoba_dir} and {parakeet_dir}")
    rows, null = [], dict.fromkeys(JUDGES, 0)
    for stem in common:
        k = {r["id"]: r for r in read_jsonl(Path(kotoba_dir) / f"{stem}.jsonl")}
        p = {r["id"]: r for r in read_jsonl(Path(parakeet_dir) / f"{stem}.jsonl")}
        tpath = Path(teacher_dir) / f"{stem}.jsonl"
        if not tpath.is_file():
            raise SystemExit(f"{tpath} is missing: no hours for stem {stem}")
        for t in read_jsonl(tpath):
            kr, pr = k.get(t["id"]), p.get(t["id"])
            null["kotoba"] += kr is None or kr.get("agree") is None
            null["parakeet"] += pr is None or pr.get("agree") is None
            if kr is None or pr is None or kr.get("agree") is None or pr.get("agree") is None:
                continue
            rows.append((kr["agree"], pr["agree"], float(t.get("duration") or 0.0) / 3600, t.get("cer")))
    if not rows:
        raise SystemExit("no row has an agree value from both judges")
    agree = {"kotoba": np.array([r[0] for r in rows]), "parakeet": np.array([r[1] for r in rows])}
    hours = np.array([r[2] for r in rows])
    tcer = np.array([np.nan if r[3] is None else float(r[3]) for r in rows])

    ref_dropped = dropped_hours(agree["kotoba"], hours, threshold)
    match = matched_threshold(agree["parakeet"], hours, ref_dropped)
    kept = {j: agree[j] <= threshold for j in JUDGES}
    table = {}
    for kk, kname in ((True, "kotoba_kept"), (False, "kotoba_dropped")):
        for pk, pname in ((True, "parakeet_kept"), (False, "parakeet_dropped")):
            m = (kept["kotoba"] == kk) & (kept["parakeet"] == pk)
            table[f"{kname}/{pname}"] = dict(rows=int(m.sum()), hours=round(float(hours[m].sum()), 4))
    has_cer = ~np.isnan(tcer)
    return dict(
        stems=len(common), rows=len(rows), hours=round(float(hours.sum()), 4), null_agree=null,
        threshold=threshold,
        quantiles={j: {str(q): round(float(np.quantile(agree[j], q)), 4) for q in QUANTILES} for j in JUDGES},
        dropped_hours={j: {str(t): round(dropped_hours(agree[j], hours, t), 4) for t in THRESHOLDS} for j in JUDGES},
        reference=dict(judge="kotoba", threshold=threshold, dropped_hours=round(ref_dropped, 4)),
        matched=dict(judge="parakeet", threshold=round(match, 4),
                     dropped_hours=round(dropped_hours(agree["parakeet"], hours, match), 4)),
        table=table,
        spearman=None if (s := spearman(agree["kotoba"], agree["parakeet"])) is None else round(s, 4),
        teacher_cer={j: dict(kept=mean_or_none(tcer[kept[j] & has_cer]), dropped=mean_or_none(tcer[~kept[j] & has_cer]))
                     for j in JUDGES},
    )


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kotoba", default="seed/second_out/galgame")
    ap.add_argument("--parakeet", default="labels/full/second_out/galgame")
    ap.add_argument("--teacher", default="labels/full/teacher_out/galgame")
    ap.add_argument("--out", default="labels/full/reports/galgame_judge.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="the config's galgame agree_max")
    args = ap.parse_args(argv)
    report = build_report(Path(args.kotoba), Path(args.parakeet), Path(args.teacher), args.threshold)
    write_json(Path(args.out), report)
    ref, m = report["reference"], report["matched"]
    print(f"{report['stems']} stems, {report['rows']} rows, {report['hours']:.2f} h; kotoba@{ref['threshold']} drops "
          f"{ref['dropped_hours']:.2f} h, parakeet@{m['threshold']} drops {m['dropped_hours']:.2f} h; "
          f"parakeet@{args.threshold} drops {report['dropped_hours']['parakeet'].get(str(args.threshold), 'n/a')} h; "
          f"spearman {report['spearman']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
