"""The size study's report: the pre-registered limit per family and every readout reported with it (study/STUDY.md
4.3-4.7, 7), from the eval results of every system on the frozen manifest. The statistics are kitsune.study_stats;
this tool reads the inputs, refuses a mismatch with the manifest, and writes report.json and a plain-English report.md.

Inputs
  --tables DIR     the per-utterance results (CONTRACT.md 5), one system per entry, named by run name, cohere,
                   parakeet-ctc, parakeet-tdt, anchor-b20 or <run>-half. Read in any of these layouts:
                     DIR/<system>.parquet                 every set, with a set column
                     DIR/<system>/<set>.parquet           one file per eval set
                     DIR/<system>/greedy_<set>.parquet    a 05_evaluate --out / trainer evals/step_<N> dir as it is
                   Columns id, set, edits, ref_len (+ edits_nostyle, sub, del, ins, hyp, truncated); a table with only
                   ref / hyp text is scored here with kitsune.evaluate's normalisation (study_stats.score_utterances).
                   Other files (tf_*.parquet, probe*.parquet, ...) are ignored.
  --manifest FILE  selections/study_manifest.json: every set's ordered ids and their sha256, and the Galgame views.
                   The manifest must hash to its own sha256s, and every table must hold exactly its ids (none missing,
                   none extra, none twice), with the same reference lengths in every system: anything else stops the
                   report (exit 2) before a number is computed.
  --prereg FILE    study/PREREG.json as kitsune.prereg writes it (or an "analysis" block of numbers): delta, the
                   sigma_run rule, the bootstrap settings and the branch's end_frac (study_stats.settings_from_prereg;
                   absent keys or no file: STUDY.md's defaults, and the report says which), the teacher baselines
                   ("baselines", parakeet_ctc and galgame:<view> keys included) and the manifest block (the eval sets'
                   and Galgame views' id hashes, the manifest file's and the selection's sha256), all compared by the
                   invalidation checks. Out-of-range settings or unreadable JSON: exit 2.
  --numbers FILE|DIR ...  the boxes' numbers files (max_steps, lr_probes, calibration, written_utc, rules_sha256) for
                   the invalidation checks: PREREG_numbers_A.json, _B.json and _replicate.json (a DIR: every
                   PREREG_numbers*.json in it), each keyed by its "box"; every run is checked against the file of the
                   box that trains it, and every file's rules_sha256 against --prereg's (study_stats.numbers_check)
  --run-summaries DIR  the trainer's summaries: its runs root as it is (DIR/<run_name>-<stamp>/summary.json, keyed
                   by config.run_name; the newest start of a run name counts), or DIR/<run>/summary.json, DIR/<run>.json
                   (steps, resumes, config.selection, started_utc or the run dir's config.json created_utc). LR probes
                   and other runs in the same dir are skipped by the checks
  --speed FILE     the A100 speed probe: {system: {"rtf": ..., "vram_gb": ..., "p50_s": ..., "p95_s": ...}} (or
                   under "systems"; study_stats.SPEED_RTF / SPEED_VRAM list the key names accepted)
  --params FILE    {system: params_total} over study_stats.PARAMS_TOTAL (e.g. from the student_meta.json files)
  --boot-b N, --seed N  override the bootstrap (development only: the report marks them as not pre-registered)

Output (--out, written atomically): report.json (everything study_stats.analyse returns, plus the inputs' paths, the
manifest file's sha256, the summaries read and the settings' sources) and report.md. report.md LEADS with the owner's
question (how far the models compress and what to offer): the "what to offer" sentences generated from the calls and
one offer table per family (study_stats.offers: rows = the own teacher and the sizes; params, M4 and its sets, the
ratio to the teacher, Parakeet TDT better / within / worse, speed, VRAM, the delta 10 % call, the T/2 readout), then
the sigma_run sensitivity (every call at 1.6 % and 3.2 %, replicate_needed), then the statistics.

Usage:
  python tools/study_report.py --tables evals/study --manifest labels/full/selections/study_manifest.json \
      --prereg study/PREREG.json --numbers study/PREREG_numbers_A.json study/PREREG_numbers_B.json \
      study/PREREG_numbers_replicate.json --speed speed.json --out reports/study
CPU only. The statistics of 22 systems at B = 10,000 take about 5 s, the imitation CER about a second per system
(--no-imitation skips it); the torch import behind kitsune.evaluate is the slowest part.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from kitsune import study_stats as ss  # noqa: E402

COLUMNS = ("id", "set", "source", "ref", "hyp", "edits", "ref_len", "edits_nostyle", "hyp_nostyle", "sub", "del",
           "ins", "hyp_len", "truncated")


def read_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_parquet(path: Path) -> pd.DataFrame:
    names = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in COLUMNS if c in names])


def _set_of_file(stem: str, sets) -> str | None:
    name = stem[len("greedy_"):] if stem.startswith("greedy_") else stem
    return name if name in sets else None


def load_tables(d: Path, sets) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """system -> one DataFrame of all its sets, in the layouts the module docstring lists; plus the files ignored."""
    tables, ignored = {}, []
    for entry in sorted(Path(d).iterdir()):
        if entry.is_file() and entry.suffix == ".parquet":
            df = _read_parquet(entry)
            if "set" not in df.columns:
                raise SystemExit(f"{entry}: a one-file system table needs a set column")
            tables[entry.stem] = df
        elif entry.is_dir():
            parts = []
            for f in sorted(entry.glob("*.parquet")):
                s = _set_of_file(f.stem, sets)
                if s is None:
                    ignored.append(str(f))
                    continue
                df = _read_parquet(f)
                if "set" in df.columns and set(df["set"].astype(str)) != {s}:
                    said = sorted(set(df["set"].astype(str)))
                    raise SystemExit(f"{f}: its set column says {said}, the file name {s}")
                parts.append(df.drop(columns=["source"], errors="ignore").assign(set=s))
            if parts:
                tables[entry.name] = pd.concat(parts, ignore_index=True)
    if not tables:
        raise SystemExit(f"no system table under {d}")
    return tables, ignored


# the trainer's run dir: <run_name>-<UTC stamp>, plus -<n> when two start in the same second (04_distill build())
RUN_DIR_STAMP = re.compile(r"-(\d{8}T\d{6}Z)(?:-(\d+))?$")


def load_summaries(d: Path | None) -> tuple[dict, dict]:
    """(run name -> summary, notes) from DIR, read in any of these layouts:
         DIR/<run_name>-<stamp>[-n]/summary.json   the trainer's runs root as it is
         DIR/<run>/summary.json, DIR/<run>.json
    A summary is keyed by its config.run_name (the trainer writes config=R.cfg; a T/2 branch's is <run>-half), else by
    the entry's name without the stamp. Several entries of one run name (a run started again from scratch): the newest
    by stamp counts, the others are listed in notes["superseded"]. started_utc, when the summary has none: the run
    dir's config.json created_utc (kitsune.runlog), else its stamp; both precede the run's first step. LR probes and
    other runs are kept here and skipped by the checks (study_stats.is_study_run)."""
    notes = dict(read={}, superseded=[])
    if d is None:
        return {}, notes
    found: dict[str, list] = {}
    for entry in sorted(Path(d).iterdir()):
        f = entry / "summary.json" if entry.is_dir() else entry
        if not (f.is_file() and f.suffix == ".json"):
            continue
        summ = read_json(f)
        if not isinstance(summ, dict):
            continue
        stem = entry.name if entry.is_dir() else entry.stem
        m = RUN_DIR_STAMP.search(stem)
        name = (summ.get("config") or {}).get("run_name") or (stem[:m.start()] if m else stem)
        if not summ.get("started_utc"):
            created = None
            if entry.is_dir() and (entry / "config.json").is_file():
                created = read_json(entry / "config.json").get("created_utc")
            if created or m:
                summ = dict(summ, started_utc=created or m.group(1),
                            started_utc_from="config.json created_utc" if created else "the run dir's stamp")
        order = (m.group(1), int(m.group(2) or 0)) if m else ("", 0)
        found.setdefault(name, []).append((order, str(f), summ))
    out = {}
    for name, cands in found.items():
        cands.sort(key=lambda c: c[0])
        out[name] = cands[-1][2]
        notes["read"][name] = cands[-1][1]
        notes["superseded"] += [c[1] for c in cands[:-1]]
    return out, notes


def load_speed(path) -> dict | None:
    if path is None:
        return None
    raw = read_json(path)
    return raw.get("systems", raw)


def load_params(path) -> dict | None:
    if path is None:
        return None
    raw = read_json(path)
    return {k: int(v["params_total"] if isinstance(v, dict) else v) for k, v in raw.items()}


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ report.md


def pct(x, nd: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f} %"


def rci(c: dict | None) -> str:
    """ratio [CI] of a comparison."""
    if not c:
        return "n/a"
    lo, hi = c["ci_ratio"]
    return f"{c['ratio']:.3f} [{lo:.3f}, {hi:.3f}]"


def gci(g, ci) -> str:
    return "n/a" if g is None else f"{100 * g:+.1f} % [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def disp(rep: dict, s: str | None) -> str:
    if s is None:
        return "none"
    return rep["systems"].get(s, {}).get("display") or ss.display(s)


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return out


def _mci(c: dict | None, nd: int = 2) -> str:
    """ratio [CI] of an offer-table cell ({ratio, ci})."""
    if not c:
        return "n/a"
    return f"{c['ratio']:.{nd}f} [{c['ci'][0]:.{nd}f}, {c['ci'][1]:.{nd}f}]"


def _num(x, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def render_offers(rep: dict) -> list[str]:
    """The report's lead: what to offer (study_stats.offer_text), then one offer table per family (study_stats.offers):
    rows = the own teacher (reference) and the sizes, largest first."""
    off = rep.get("offers")
    if not off:
        return []
    dp = next(iter(off.values()))["delta"]
    L = ["## What to offer", ""] + [f"- {t}" for t in rep.get("offer_text", [])] + [""]
    head = ["model", "params total / non-emb.", "x smaller than teacher", "M4 CER", "JSUT", "CV8", "Reazon",
            "Galgame-neutral", "M4 / own teacher [CI]", "vs Parakeet TDT (JSUT+Galgame) [CI]", "batched RTF",
            "batch-1 p50 / p95 s", "peak VRAM GB", f"call at delta {100 * dp:g} % (M4 / top [CI])", "T/2 readout"]
    for fam, F in off.items():
        rows = []
        for r in F["rows"]:
            sp = r.get("speed") or {}
            sets = r.get("sets") or {}
            name = r["display"] + (" (teacher)" if r["role"] == "teacher" else " (top)" if r["role"] == "top" else "")
            if not r["available"]:
                rows.append([name, f"{_num(r['params_total'], ',')} / {_num(r['params_non_embedding'], ',')}"]
                            + ["no results"] + [""] * (len(head) - 3))
                continue
            tdt = r.get("vs_tdt")
            vt = r.get("vs_top")
            th = r.get("t_half")
            rows.append([
                name, f"{_num(r['params_total'], ',')} / {_num(r['params_non_embedding'], ',')}",
                _num(r.get("smaller_than_teacher"), ".1f"), pct(r["m4"]),
                *(pct(sets.get(k)) for k in ss.M4_SETS),
                _mci(r.get("vs_teacher")), f"{tdt['verdict']} {_mci(tdt)}" if tdt else "n/a",
                _num(sp.get("rtf"), ".4f"), f"{_num(sp.get('p50_s'), '.3f')} / {_num(sp.get('p95_s'), '.3f')}",
                _num(sp.get("vram_gb"), ".2f"),
                "top" if r["role"] == "top" else "" if r["role"] == "teacher" else
                (f"{vt['call']} {_mci(vt, 3)}" if vt else "n/a"),
                "" if r["role"] != "size" else
                (f"{th['label']} (delta ln r {th['delta']:+.3f} [{th['ci'][0]:+.3f}, {th['ci'][1]:+.3f}]; "
                 f"{th['call_at_T_half']} at T/2)" if th else "n/a")])
        L += [f"### Offer table: {F['text']}", ""] + table(head, rows) + [""]
    L += ["vs Parakeet TDT: better / worse = the CI of the JSUT + Galgame-neutral ratio lies below / above 1; within "
          "= it holds 1. Speed from the A100 speed probe (n/a without --speed).", ""]
    return L


def render_sensitivity(rep: dict) -> list[str]:
    """The conditional replicate: every call at each sigma_run of the grid, and whether the replicate is needed."""
    sens = rep.get("sensitivity")
    if not sens:
        return []
    grid = sens["grid"]
    L = ["## Sensitivity to sigma_run (the conditional replicate)", "", sens["text"], ""]
    rows = []
    for f in ("transcribe", "parakeet", "scratch"):
        rows.append([f + (" (triggers)" if f in sens["trigger_families"] else "")]
                    + [sens["at"][ss._skey(g)]["families"][f]["sentence"] for g in grid])
    L += table(["family: delta-primary walk"] + [f"sigma_run {100 * g:g} %" for g in grid], rows) + [""]
    L += [f"replicate_needed: **{'yes' if sens['replicate_needed'] else 'no'}** (state {sens['state']})."]
    moved = [d for v in sens["differences"].values() for d in v]
    if moved:
        L += ["", "Calls that move across the grid (any delta, the bars, the T/2 readout):", ""]
        L += [f"- {d}" for d in moved[:40]] + (["- ..."] if len(moved) > 40 else [])
    return L + [""]


def render_md(rep: dict) -> str:
    st = rep["settings"]
    dp = ss._dkey(float(st["delta_primary"]))
    L = ["# Size study: analysis report", ""]
    meta = rep.get("inputs", {})
    L.append(f"Written {meta.get('written_utc', '?')} from `{meta.get('tables', '?')}` ({len(rep['systems'])} systems) "
             f"and the manifest `{meta.get('manifest', '?')}`. The statistics are `kitsune/study_stats.py`; the "
             f"design is study/STUDY.md sections 4 and 7.")
    L.append("")
    fails = [c for c in rep["checks"] if c["status"] == "fail"]
    if fails:
        L += ["**The conclusion is INVALID** (STUDY.md 7): " + "; ".join(f"{c['rule']}: {c['detail']}" for c in fails),
              ""]
    if rep.get("missing_systems"):
        L += [f"Systems without results: {', '.join(rep['missing_systems'])}.", ""]

    L += render_offers(rep)
    L += render_sensitivity(rep)

    L += [f"## The answer (delta = {100 * float(st['delta_primary']):g} %, pre-registered)", ""]
    for f in ("transcribe", "parakeet", "scratch"):
        F = rep["families"][f]
        tag = " (second walk)" if f == "scratch" else ""
        L.append(f"- **{F['text']}**{tag}: {F['primary']['sentence'] if F['primary'] else 'n/a'}.")
    L += ["", "The limit is the smallest size whose M4 stays within 1 + delta of the family's largest student, with "
              "a CI that includes run-to-run noise. The other tolerances, always reported:", ""]
    rows = [[f] + [rep["families"][f]["walks"][ss._dkey(d)]["sentence"] for d in st["deltas"]]
            for f in ("transcribe", "parakeet", "scratch")]
    L += table(["family"] + [f"delta {100 * d:g} %" for d in st["deltas"]], rows) + [""]

    sg = rep["sigma_run"]
    L += ["## Noise", "",
          f"- sigma_run = {pct(sg['sigma_run'])} per system, from {sg['source']}. Replicate estimate "
          f"{pct(sg['sigma_hat']) if sg['sigma_hat'] is not None else 'n/a'}, prior {pct(sg['prior'])}.",
          f"- Paired, stratified bootstrap: B = {rep['bootstrap']['B']:,}, seed {rep['bootstrap']['seed']}, strata "
          + ", ".join(f"{k} {v:,}" for k, v in rep["bootstrap"]["strata"].items()) + " utterances "
          "(empty references left out: " + ", ".join(f"{k} {v}" for k, v in rep["bootstrap"]["n_empty_ref"].items()
                                                    if v) + ").",
          f"- CI of a ratio: ln r +- {st['z']} sqrt(v_boot + k sigma_run^2), k = 2 student vs student, 1 student vs "
          f"teacher."]
    oc = rep.get("operating_characteristics")
    if oc:
        L.append(f"- The median total SD of ln r in the walks is {pct(oc['sigma_total'])} (the design assumed 2.4 %). "
                 "At that noise the calls come out this often (normal theory, WITHIN / OUTSIDE):")
        L.append("")
        rows = [[f"r = {r}"] + [f"{100 * v['WITHIN']:.0f} % / {100 * v['OUTSIDE']:.0f} %" for v in cells.values()]
                for r, cells in oc["table"].items()]
        L += table(["true r"] + [f"delta {100 * d:g} %" for d in st["deltas"]], rows)
    srcs = rep.get("settings_sources", {})
    nondefault = {k: v for k, v in srcs.items() if not v.startswith("default")}
    L += ["", "Settings: " + (", ".join(f"{k} from {v}" for k, v in nondefault.items()) if nondefault
                              else "every setting is STUDY.md's default (no PREREG.json value)") + ".", ""]

    L += ["## The ladders", ""]
    for f in ("transcribe", "parakeet", "scratch"):
        F = rep["families"][f]
        L += [f"### {F['text']}", "",
              f"Top: {F['top_display']}, M4 {pct(F['top_m4'])}. r = M4(size) / M4(top); the qualifier pairs are "
              "reported with every call.", ""]
        rows = []
        for e in F["entries"]:
            c = e["m4"]
            q = e.get("qualifiers") or {}
            rows.append([e["display"], f"{e['params_total'] / 1e6:.1f}M" if e["params_total"] else "n/a",
                         pct(c["a_value"]) if c else "n/a", rci(c)]
                        + [c["calls"][ss._dkey(d)] if c else "n/a" for d in st["deltas"]]
                        + [rci(q.get("ood")), rci(q.get("ind")), e["confound"]])
        L += table(["size", "params", "M4", "r [95 % CI]"] + [f"delta {100 * d:g} %" for d in st["deltas"]]
                   + ["out-of-domain r", "in-domain r", "besides size"], rows) + [""]

    L += ["## g per halving (descriptive)", "", "g = r^(1/h) - 1, h = log2 of the total-parameter ratio.", ""]
    rows = [[s["step"], f"{s['h']:.3f}" if s["h"] else "n/a", rci(s) if s["available"] else "n/a",
             gci(s.get("g"), s.get("ci_g")), s["label"]] for s in rep["steps"]]
    L += table(["step", "h", "r [CI]", "g per halving [CI]", "besides size"], rows) + [""]
    dg = rep["delta_g"]
    if dg["available"]:
        L.append(f"- Delta-g on the scratch ladder (the only two consecutive steps in one init regime): "
                 f"g(T-0.1B -> T-0.05B) - g(bridge -> T-0.1B) = {100 * dg['delta_g']:+.1f} points "
                 f"[{100 * dg['ci'][0]:+.1f}, {100 * dg['ci'][1]:+.1f}]. Caveat: the bridge's d-1280 shape is unusual "
                 f"for a scratch model, and it is the least-trained scratch run.")
    else:
        L.append(f"- Delta-g: not available ({dg['reason']}).")
    ie = rep["init_effect"]
    L.append(f"- Init effect at 0.3B, M4(bridge) / M4(T-0.3B): {rci(ie['comparison'])}"
             + (" (above 1: pruning from Cohere beats scratch at this size)" if ie["available"] else "") + ".")
    L += ["", "## Distillation gaps (not steps of the walk)", "",
          "Student vs its own teacher (a fixed model: v_boot + sigma_run^2); g per halving over the teacher -> student "
          "parameter ratio, descriptive only.", ""]
    rows = [[f, disp(rep, c["a"]) + " / " + disp(rep, c["b"]), f"{c['h']:.3f}" if c.get("h") else "n/a", rci(c),
             gci(c.get("g"), c.get("ci_g"))] if c else [f, "n/a", "n/a", "n/a", "n/a"]
            for f, c in rep["distillation_gaps"].items()]
    L += table(["family", "student / teacher", "h", "M4 ratio [CI]", "g per halving [CI]"], rows) + [""]

    L += ["## Budget readout: T/2 vs T", "",
          "Each size's ratio to the top at T/2 (the branches) and at T; delta = ln r(T) - ln r(T/2). A CI below 0 "
          "means the gap closes with compute.", ""]
    rows = []
    for f, b in rep["budget"].items():
        for r in b["rows"]:
            if not r["available"]:
                rows.append([f, r["display"], "n/a", "n/a", "n/a", r.get("reason")])
                continue
            h, t = r["at_T_half"], r["at_T"]
            rows.append([f, r["display"], f"{rci(h)} {h['calls'][dp]}", f"{rci(t)} {t['calls'][dp]}",
                         f"{r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]", r["label"]])
    L += table(["family", "size", f"r at T/2, call delta {100 * float(st['delta_primary']):g} %", "r at T, call",
                "delta ln r [CI]", "readout"], rows) + [""]

    bars = rep["bars"]
    L += ["## Practical bars", ""]
    t = bars["tdt"]
    if t["available"]:
        L.append(f"- Smallest student whose JSUT + Galgame-neutral CI upper end is at or below Parakeet TDT's: "
                 f"**{disp(rep, t['smallest'])}**.")
    else:
        L.append("- Parakeet TDT bar: no parakeet-tdt table.")
    for k, v in bars["teacher"]["smallest"].items():
        L.append(f"- Smallest student within {k} of its own teacher on M4: {disp(rep, v['point'])} by the point "
                 f"ratio, {disp(rep, v['ci'])} with the CI's upper end within it.")
    rows = []
    for s, c in bars["teacher"]["per_student"].items():
        tc = bars["tdt"]["per_student"].get(s)
        rows.append([disp(rep, s), rci(c), rci(tc), ("yes" if tc["meets"] else "no") if tc else "n/a"])
    L += [""] + table(["student", "M4 / own teacher [CI]", "JSUT+Galgame / Parakeet TDT [CI]", "meets the TDT bar"],
                      rows) + [""]

    L += ["## Cross-family: JSUT + Galgame-neutral (descriptive only, no threshold)", ""]
    rows = [[disp(rep, s), pct(v["jg"]), pct(v["jg_nostyle"])] for s, v in rep["cross_family"]["per_system"].items()]
    L += table(["system", "raw", "no-style"], rows) + [""]
    rows = [[f"{disp(rep, p['a'])} / {disp(rep, p['b'])}", rci(p["raw"]), rci(p["nostyle"])]
            for p in rep["cross_family"]["equal_size"]]
    L += table(["equal size", "raw r [CI]", "no-style r [CI]"], rows) + [""]

    L += ["## Pareto: CER vs A100 RTF and VRAM", ""]
    pa = rep["pareto"]
    if pa["available"]:
        rows = [[disp(rep, s), pct(r["jg"]), pct(r["m4"]), "n/a" if r["rtf"] is None else f"{r['rtf']:.4g}",
                 "n/a" if r["vram_gb"] is None else f"{r['vram_gb']:.1f}", "yes" if s in pa["front"]["jg"] else ""]
                for s, r in pa["rows"].items()]
        L += table(["system", "JSUT+Galgame", "M4", "RTF", "VRAM GB", "on the front (JSUT+Galgame)"], rows)
        L += ["", f"Front on M4 instead: {', '.join(disp(rep, s) for s in pa['front']['m4'])}. {pa['note']}.", ""]
    else:
        L += [f"Not available: {pa['reason']}.", ""]

    L += ["## Anchor regression flag", ""]
    an = rep["anchor"]
    if an["available"]:
        c = an["comparison"]
        L.append(f"T-0.6B / anchor on the gate-pooled CER: {rci(c)} ({pct(c['a_value'])} vs {pct(c['b_value'])}). "
                 + ("**The flag fires**: interpretation halts until the cause (BN, L2-SP, LR, data) is understood."
                    if an["flag"] else "The flag does not fire."))
    else:
        L.append("Not available: needs study-t06 and anchor-b20.")
    L += ["", "## Invalidation checks (STUDY.md 7)", ""]
    L += table(["rule", "status", "detail"], [[c["rule"], c["status"], c["detail"]] for c in rep["checks"]]) + [""]
    if meta.get("summaries_superseded"):
        L += ["Superseded run summaries (an earlier start of the same run name, not used): "
              + ", ".join(f"`{s}`" for s in meta["summaries_superseded"]) + ".", ""]

    L += ["## Per-system results", "", "Corpus CER per set; group metrics are macro means of the sets' CERs "
          "(gate-pooled: sum / sum over JSUT, CV8, Reazon).", ""]
    cols = [("eval_jsut", "JSUT"), ("eval_cv8", "CV8"), ("eval_reazon", "Reazon"), ("galgame_neutral", "Gal-neutral"),
            ("galgame_all", "Gal-all"), ("eval_emilia", "Emilia")]
    rows = []
    for s, v in rep["systems"].items():
        rows.append([v["display"], v["role"]] + [pct((v["sets"].get(k) or {}).get("cer")) for k, _ in cols]
                    + [pct(v["metrics"]["m4"]), pct(v["metrics"]["jg"]), pct(v["metrics"]["gate_pooled"]),
                       f"{v['ratio_vs_teacher']['m4']:.3f}" if v["ratio_vs_teacher"]["m4"] else ""])
    L += table(["system", "role"] + [c for _, c in cols] + ["M4", "JSUT+Gal", "gate-pooled", "M4 / teacher"], rows)
    L += ["", "Error profile over the M4 sets (S / D / I per reference char; rates per utterance):", ""]
    rows = []
    for s, v in rep["systems"].items():
        d = v.get("m4_sets") or {}
        rows.append([v["display"]] + [pct(d.get(k)) for k in ("sub_rate", "del_rate", "ins_rate", "runaway_rate",
                                                                "empty_hyp_rate", "truncated_rate", "imitation_cer")])
    L += table(["system", "sub", "del", "ins", "runaway", "empty", "truncated", "imitation CER"], rows) + [""]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main


class InputError(ValueError):
    """An input the report cannot use as given (PREREG.json's settings out of range, an unreadable table): the report
    refuses (exit 2) instead of computing numbers from it."""


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_numbers(paths) -> tuple[dict | None, dict, list[str]]:
    """({box: numbers file}, {box: the file's sha256}, the files read) from --numbers: files, or dirs whose
    PREREG_numbers*.json are read. Each file is keyed by its "box" ("all" when it has none, the single-host layout);
    two files of one box, or a file that is not a JSON object, is an InputError."""
    if not paths:
        return None, {}, []
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("PREREG_numbers*.json")) if p.is_dir() else [p]
    if not files:
        raise InputError(f"--numbers {[str(p) for p in paths]}: no numbers file")
    out, shas = {}, {}
    for f in files:
        try:
            data = read_json(f)
        except (OSError, ValueError) as e:
            raise InputError(f"numbers file {f}: {e}") from e
        if not isinstance(data, dict):
            raise InputError(f"numbers file {f}: not a JSON object")
        box = str(data.get("box") or "all")
        if box in out:
            raise InputError(f"numbers files: two files of box {box} ({f})")
        out[box], shas[box] = data, file_sha256(f)
    return out, shas, [str(f) for f in files]


def prereg_rules_sha256(path) -> str | None:
    """kitsune.prereg.rules_sha256 of --prereg (the canonical bytes every numbers file carries as rules_sha256)."""
    if not path:
        return None
    from kitsune import prereg as kp

    return kp.rules_sha256(path)


def build_report(args) -> dict:
    manifest = ss.parse_manifest(read_json(args.manifest))
    tables, ignored = load_tables(args.tables, manifest.sets)
    try:
        corpus = ss.build_corpus(tables, manifest)
    except ss.ManifestError:
        raise
    except ValueError as e:  # a table without edits / ref_len and without text to score
        raise InputError(f"tables: {e}") from e
    try:
        prereg = read_json(args.prereg) if args.prereg else None
        settings, sources = ss.settings_from_prereg(prereg)
    except ValueError as e:  # json.JSONDecodeError is one
        raise InputError(f"PREREG.json {args.prereg}: {e}") from e
    for key, val in (("boot_b", args.boot_b), ("boot_seed", args.seed)):
        if val is not None:
            settings[key], sources[key] = int(val), "command line (NOT the pre-registered value)"
    summaries, summ_notes = load_summaries(args.run_summaries)
    numbers, numbers_sha, numbers_files = load_numbers(args.numbers)
    rep = ss.analyse(corpus, settings, params=load_params(args.params), speed=load_speed(args.speed),
                     numbers=numbers, summaries=summaries,
                     prereg_baselines=ss.prereg_baselines(prereg), prereg_manifest=ss.prereg_manifest(prereg),
                     manifest_file_sha256=file_sha256(args.manifest), imitation=not args.no_imitation,
                     rules_sha256=prereg_rules_sha256(args.prereg), numbers_file_sha256=numbers_sha)
    rep["settings_sources"] = sources
    rep["inputs"] = dict(tables=str(args.tables), manifest=str(args.manifest),
                         manifest_sha256=file_sha256(args.manifest), prereg=_s(args.prereg),
                         numbers=numbers_files, numbers_sha256=numbers_sha, speed=_s(args.speed),
                         params=_s(args.params),
                         run_summaries=_s(args.run_summaries), summaries_read=summ_notes["read"],
                         summaries_superseded=summ_notes["superseded"], ignored_files=ignored,
                         written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return rep


def _s(x):
    return None if x is None else str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--prereg", type=Path)
    ap.add_argument("--numbers", type=Path, nargs="+", help="the boxes' PREREG_numbers_<box>.json, or a dir of them")
    ap.add_argument("--run-summaries", type=Path)
    ap.add_argument("--speed", type=Path)
    ap.add_argument("--params", type=Path)
    ap.add_argument("--boot-b", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-imitation", action="store_true", help="skip the imitation CER (a jiwer pass per system)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        rep = build_report(args)
    except (ss.ManifestError, InputError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out / "report.json", json.dumps(rep, indent=1, ensure_ascii=False))
    write_atomic(args.out / "report.md", render_md(rep))
    for f in ("transcribe", "parakeet", "scratch"):
        p = rep["families"][f]["primary"]
        print(f"{f}: {p['sentence'] if p else 'n/a'}")
    if rep["invalid"]:
        print("INVALID: " + "; ".join(c["rule"] for c in rep["checks"] if c["status"] == "fail"))
    print(f"wrote {args.out / 'report.json'} and report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
