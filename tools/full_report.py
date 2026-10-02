"""The full-data runs' report: what to offer, with every number the owner asked for next to it (plan v3 sections 6-7,
CONTRACT.md 10). It is DESCRIPTIVE: the full runs, their quantised variants and the Whisper yardsticks are outside the
pre-registration (study/PREREG.json), so no call of the size study is made or changed here (flag D). It stacks on the
study: every system is scored on the frozen study manifest, in one corpus with the study's own tables, so a full-data
number and a study number are always over exactly the same utterances.

Inputs (lean pulls of the runs repo; no weights are read)
  --manifest FILE        the frozen study_manifest.json (kitsune.study_stats.parse_manifest verifies it). It is
                         compared with --prereg's manifest block (default study/PREREG.json; "none" skips that); a
                         manifest PREREG did not freeze refuses (exit 2), as 05_evaluate refuses it. "Stacks on the
                         study" = its file sha256 is kitsune.fullrun.FROZEN_MANIFEST_SHA256 and PREREG matches.
  --tables DIR ...       the study's per-utterance tables, DIR/<system>/<set>.parquet (tools/study_report.load_tables:
                         the study students, the teachers, the anchor, the T/2 branches)
  --readouts DIR ...     every study.json under DIR (05_evaluate --out dirs: the M4 readouts runs/m4-<run_id>[-r<N>],
                         the quant readouts, whisper_eval dirs; a quant readout whose quant block's recipe_version is
                         below QUANT_RECIPE_MIN, or absent, is ignored: quant code older than F1). Per hit: study.json
                         (system, family, the quant block of WP5, weights.file_bytes), its tables
                         <hit>/tables/<system>/<set>.parquet or else its greedy_<set>.parquet (scored here as 05 scores
                         them; never the box paths study.json.tables names), summary.json (tf.sets.<set>.kl and n_tok:
                         KL to the own teacher) and whisper.json (params, file bytes, decode counts). A greedy set whose
                         rows are not exactly its manifest ids gets no table, as 05 refuses it: that system lacks the
                         set (inputs.readouts_refused_sets). greedy_<set>.parquet also give the clip durations (the
                         short-clip counts) and the AED token counts (flag S2). One system found twice: the one scored
                         on every manifest set wins, then a real (not simulated) one, then one scored from its exported
                         file over one scored from memory (smoke B scores study-p03@<fmt> both ways; only the file has
                         its true bytes), then the newest time_utc; the others are listed. A --tables system always wins
                         over a readout of that name.
  --speed FILE ...       tools/speed_probe.py outputs (every runs/speed-*/speed.json given). Every file must hold
                         the same id list (ids_sha256: the study's 200 ids); another list refuses (exit 2). Records
                         are grouped by machine + GPU: a group is one machine, and only numbers of one group are
                         comparable. A record's machine: its versions.machine_id (WP6's speed_probe records
                         KITSUNE_MACHINE_ID); else --machine-of; else the machine_id of the queue summary of the box
                         that wrote the file (a summary that names that runs/speed-<box>-<stamp> dir or is that box's,
                         of a launch that started before the record was timed); else versions.host. The wave-1
                         speed_probe that times smoke A records only the host, the container's name, which differs
                         per rental: without the summary, smoke A and smoke B on one machine would be two groups. One
                         name timed twice in one group: the newest time_utc counts, the others are listed
                         (speed_groups.superseded). The CHART GROUP is the group holding the most systems; the speed
                         columns and the charts come from it. F4 (DECISIONS F): a quantised record timed by quant
                         code older than F1 (quant_recipe.version below QUANT_RECIPE_MIN = 2, or absent: box
                         53693389's smoke B #1, whose fp8 records decoded garbage) and a quantised or compiled record
                         without a passing CER sanity (speed_probe's sanity) are never used; each file's failed block
                         and its dir's events.jsonl (the queue's line per speed item) give the probes that failed.
                         report.json's speed_probes and report.md's "Speed probes not measured or left out" list them
                         all, with the offered rows without any speed record (S5) and the queue summaries' speed
                         items not done; check speed_probes fails on a failed or absent one.
  --queue-summaries PATH ...  the full boxes' full/box-<box>/queue_summary.json (CONTRACT.md 5: box, machine_id,
                         started, the speed items' out), as files or dirs searched for them. The summary at the runs
                         repo's layout beside a speed file (<root>/full/box-<box>/ for <root>/runs/speed-<box>-*/) is
                         read without the flag.
  --machine-of NAME=ID ...  the machine of the records in the speed dir NAME (speed-<box>-<stamp>) or of the container
                         host NAME (versions.host), for a file no summary covers (a box launched again elsewhere
                         replaces its summary)
  --run-summaries DIR ... the trainer's runs roots (tools/study_report.load_summaries): steps, epochs, stopped_early,
                         early_stop_trigger, end_reason, resume_resets of the full and study runs
  --params FILE          {system: params_total} (the study's params.json); kitsune.study_stats.PARAMS_TOTAL, a
                         whisper.json or a speed record fill the rest
  --kotoba-jsonl FILE    optional (decision 29; its download was not approved): Kotoba-Whisper v2.0's stored
                         Galgame judge file second_out/galgame/eval-00000.jsonl, sha256 KOTOBA_JSONL_SHA256 (another
                         file refuses): Kotoba's CER on the 810 Galgame-neutral rows vs all 969, the size of the bias
                         its own row selection gives it
  --include-half         also the study's T/2 branches (left out by default: they are budget readouts, not offers)
  --boot-b N, --seed N   the paired bootstrap (default 10,000 replicates, seed 1234: the study's)

Systems and roles (never kitsune.study_stats.is_trained/family_of, which only know "study-" names): teacher (cohere,
parakeet-ctc, parakeet-tdt: fixed models), full (FULL_RUNS), study (the study's students and controls), quant (a
study.json with a quant block; its base is quant.base_system and its format quant.format, never parsed from the
name), whisper (family "whisper", or a WHISPER_SYSTEMS key), anchor / half / other (listed, never offered).

Numbers. One corpus (kitsune.study_stats.build_corpus: every table must hold exactly the manifest's ids, with equal
reference lengths in every system, or the report refuses) and one paired, stratified bootstrap over every system.
Local metrics (FULL_METRICS; never added to study_stats.METRICS, which would change the study report): m4, m4_nostyle,
m4_all (Galgame-all in place of Galgame-neutral: the Whisper headline, caveat 1), m4_all_nostyle, m3 (JSUT, CV8,
Reazon: the "3-set mean"), m3_nostyle, jg, jg_nostyle, gate_pooled, and gap_neutral_all = CER(Galgame-all) -
CER(Galgame-neutral). A ratio's CI is ln r +- 1.96 sqrt(v_boot + k sigma_run^2) with k named per comparison: 1 for a
trained system against its teacher, 2 for full vs study (two trainings), 0 for a quantised variant against its own
bf16 weights (the same run: only the bootstrap). sigma_run is the study's (the prior 1.6 % without the replicate).
Comparisons: vs_teacher (M4 and M4-all, the own teacher: Cohere for T students, Parakeet CTC for P students; none for
Whisper), vs_study (full-x / study-x), vs_bf16 (a variant / its base). KL to the own teacher, token-weighted over the
three gate sets, from each readout's summary.json; within a family only (AED per token, CTC per target token).
Hallucination counts per system over the M4 strata and over M4-all (rows with a reference): empty outputs, runaways
(edits > reference length, or output > 2x it), truncated decodes, SILENCE_PHRASES (Whisper's subtitle phrases where
the reference does not hold them; a heuristic), bracketed spans; the same on clips under SHORT_CLIP_S; and non-empty
outputs on Galgame's empty references. Kana mode: JSUT rows whose reference has >= 3 kanji and whose non-empty
output has none (flag K1 over 1 %).

Speed, per system: its own record in the chart group; else a full-data row takes its study weights' record (flag S1:
same shape, same code); else a record from another group (S3). A full-data AED row whose readout emits more than 5 %
more or fewer tokens on the 200 speed ids than the timed study weights did gets S2, and then its speed is the chart
group's study-t06 record times the full-t06 / study-t06 ratio of box full's re-time pair (the two timed in one speed
file: one machine, one time; box full's group, or the chart group itself when box full rented the chart's machine and
full-t06 has its own record there), labelled so. Its quantised variants, which decode the same tokens, take the same
bf16 ratio (the pair is bf16 only; labelled so), so a variant and its bf16 row stay on one basis: the chart's joins and
the quantisation table's "x bf16 speed" compare like with like (speed.unscaled keeps the record's own numbers). MXFP4
and emulated variants have no speed ("n/a (simulated)", S4). File size, in order: study.json quant.file_bytes,
quant.weights_bytes (a variant scored from memory: never its bf16 checkpoint's weights.file_bytes; WP5 writes it as
{deployable, quantized, kept}, and deployable is what the exporter would write), weights.file_bytes, whisper.json
model.weights_file_bytes, else the speed record's in-memory weights_bytes (both in-memory sources are marked so).

Outputs (--out, written atomically): report.json; report.md (the offer table first, then study -> full, quantisation,
the speed probes not measured or left out, Whisper with its seven caveats verbatim from plan v3 section 7, the charts,
hallucinations, per-set CERs, the flags, inputs and checks); offer.csv; chart_error_vs_speed.svg (x batched RTFx, log; y
M4-all CER %) and chart_error_vs_latency.svg (x batch-1 p50 ms, log), hand-written SVG with deterministic bytes
(matplotlib is not in the box image), drawn from the chart group only; chart_points.json / .csv (the same points, for a
live chart).

Exit codes: 0 written; 2 refused (manifest, tables, speed id lists, an unreadable or malformed input file: JSON,
parquet, the params, a run summary); 1 anything else.

Usage (on the laptop, after a lean pull of the runs repo into D:/kitsune-pull):
  python tools/full_report.py --manifest D:/kitsune-study/selections/study_manifest.json \
      --tables D:/kitsune-study/results/tables --readouts D:/kitsune-pull/runs \
      --speed D:/kitsune-pull/runs/speed-full-smoke-*/speed.json D:/kitsune-pull/runs/speed-smoke-b-*/speed.json \
      --queue-summaries D:/kitsune-pull/full --run-summaries D:/kitsune-pull/runs D:/kitsune-study/results/runs/runs \
      --params D:/kitsune-study/results/params.json --out D:/kitsune-study/full-report
CPU only; about a minute (the bootstrap of every system at B = 10,000 and the torch import behind kitsune.evaluate).
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from kitsune import fullrun  # noqa: E402
from kitsune import study_stats as ss  # noqa: E402
from kitsune.text import normalize_ja  # noqa: E402


def _load_tool(name: str):
    """tools/<name>.py as a module (tools/ is not a package): study_report's readers are reused read-only."""
    spec = importlib.util.spec_from_file_location(f"kitsune_tool_{name}", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = _load_tool("study_report")

# ------------------------------------------------------------------------------------------------ names

# full-data run name -> (its study counterpart: same shape and recipe, the speed fallback; family; display)
FULL_RUNS = {"full-t06": ("study-t06", "aed", "T-0.6B"), "full-p03": ("study-p03", "ctc", "P-0.3B"),
             "full-p01": ("study-p01", "ctc", "P-0.1B"), "full-p005": ("study-p005", "ctc", "P-0.05B")}
STUDY_TO_FULL = {v[0]: k for k, v in FULL_RUNS.items()}
# kitsune.quant.QUANT_FORMATS (CONTRACT.md 1.1; a test compares them when kitsune.quant is importable)
QUANT_FORMATS = ("fp16", "int8-w8a16", "int8-w8a8", "nvfp4-w4a16", "nvfp4-w4a4", "mxfp4-w4a4", "fp8-w8a8")
# kitsune.quant.RECIPE_VERSION (a test compares them): a quant readout or a quantised speed record below it was made by
# quant code older than F1 (DECISIONS F, 2026-10-01; box 53693389's smoke B #1 ran 14bfcad: fp32 fp8 scales, NaN on
# every padded row, its fp8 records decoding garbage) and is never used (F4): such a readout goes to
# inputs.readouts_ignored, such a speed record to speed_probes.excluded (a record or readout without the field is
# version 1)
QUANT_RECIPE_MIN = 2
FORMAT_LABEL = {"fp16": "FP16", "int8-w8a16": "INT8 W8A16", "int8-w8a8": "INT8 W8A8", "nvfp4-w4a16": "NVFP4 W4A16",
                "nvfp4-w4a4": "NVFP4 W4A4", "mxfp4-w4a4": "MXFP4 W4A4", "fp8-w8a8": "FP8 W8A8"}
UNTIMED_FORMATS = ("mxfp4-w4a4",)  # no real kernel on the 5090 (decision 20): accuracy only, never a speed
# kitsune.whisper.WHISPER_MODELS keys (CONTRACT.md 9) -> display
WHISPER_SYSTEMS = {"whisper-large-v3": "Whisper large-v3", "whisper-large-v3-turbo": "Whisper large-v3 turbo",
                   "kotoba-whisper-v2.0": "Kotoba-Whisper v2.0", "whisper-small": "Whisper small"}
KOTOBA = "kotoba-whisper-v2.0"
# the Galgame judge file behind Galgame-neutral (study_1000h.json: second_out/galgame/eval-00000.jsonl)
KOTOBA_JSONL_SHA256 = "6ba037dce6ac1dec2d1c96ba4a7d74c9aed1f2337deb39b214c82f69ade6f9c6"
STUDY_SPEED_IDS_SHA256 = "f27eb7ec8565af86cd8cfd4b067959feded654ced7b59128c657c67940d02802"  # --per-set 40 --seed 1234
TEACHER_OF = {"aed": "cohere", "ctc": "parakeet-ctc"}  # kitsune.evaluate.FAMILY_TEACHER
TEACHER_FAMILY = {"cohere": "aed", "parakeet-ctc": "ctc", "parakeet-tdt": "ctc"}

M4_ALL_STRATA = ("eval_jsut", "eval_cv8", "eval_reazon", "galgame_all")
M3_STRATA = ("eval_jsut", "eval_cv8", "eval_reazon")  # kitsune.evaluate.GATE_SETS
# name -> (aggregation, strata, on no-style edits), study_stats.METRICS' form
FULL_METRICS = {
    "m4": ("macro", ss.M4_SETS, False),
    "m4_nostyle": ("macro", ss.M4_SETS, True),
    "m4_all": ("macro", M4_ALL_STRATA, False),
    "m4_all_nostyle": ("macro", M4_ALL_STRATA, True),
    "m3": ("macro", M3_STRATA, False),
    "m3_nostyle": ("macro", M3_STRATA, True),
    "jg": ("macro", ss.CROSS_SETS, False),
    "jg_nostyle": ("macro", ss.CROSS_SETS, True),
    "gate_pooled": ("pooled", M3_STRATA, False),
}
GAP = "gap_neutral_all"  # CER(galgame_all) - CER(galgame_neutral): what the Kotoba-chosen view takes off
COMPARE_METRICS = ("m4", "m4_all")

# runs/speed-<box>-<stamp>: the speed items of one box launch share this dir (CONTRACT.md 1.1)
SPEED_DIR_RE = re.compile(r"^speed-(?P<box>[a-z0-9][a-z0-9.-]*)-(?P<stamp>\d{8}T\d{6}Z)(?:-\d+)?$")

SHORT_CLIP_S = 2.0
# Whisper's well-known outputs on silence and music (YouTube subtitle credits), compared after normalize_ja; a
# heuristic count, listed in report.json
SILENCE_PHRASES = ("ご視聴ありがとうございました", "ご覧いただきありがとうございます", "チャンネル登録", "字幕")
BRACKETS = re.compile(r"（[^）]*）|\([^)]*\)|\[[^\]]*\]|【[^】]*】")  # sound-effect / speaker tags, not quotes
KANJI = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf々〆]")
KANA_MIN_KANJI = 3
K1_RATE = 0.01
S2_DRIFT = 0.05
Q1_FORMATS = ("int8-w8a16", "int8-w8a8")

FLAGS = {
    "W1": "Galgame-neutral was chosen with Kotoba-Whisper v2.0 (rows with its CER <= 0.5 kept): strongly in "
          "Kotoba's favour, mildly in large-v3's and turbo's; the Whisper headline is M4-all.",
    "W2": "Reazon is in-domain for Kotoba (its test set is held out of Kotoba's training corpus).",
    "W3": "Whisper's exposure to JSUT and CV8 is undisclosed; treat CV8 as possibly seen.",
    "W4": "The students' training rows were filtered by agreement with Whisper-family transcripts: the comparison is "
          "not fully independent.",
    "W5": "Output style: Whisper writes Arabic numerals and its own kana/kanji choices; see the no-style M4.",
    "W6": "Emilia references are Whisper-medium output: Emilia is shown, never compared.",
    "R1": "Reazon is in-domain (Parakeet and the students were trained on ReazonSpeech).",
    "S1": "Speed of the study weights of the same shape and code (the full-data weights were not timed).",
    "S2": "Token drift over 5 % between the full-data weights and the timed study weights on the speed ids: re-timed "
          "(the study row x the full/study ratio of box full's bf16 re-time pair; a variant x the same bf16 ratio) "
          "where that pair exists.",
    "S3": "Speed from another machine or GPU than the chart group's: not comparable with the other rows.",
    "S4": "Simulated (MXFP4, or an emulated run): accuracy only, no speed.",
    "S5": "No speed record.",
    "Q1": "INT8 with fp32 fallbacks (quant.fp32_fallbacks > 0).",
    "K1": "JSUT kana mode over 1 % (a reference with >= 3 kanji, an output with none).",
    "E1": "Early stop fired (the trainer's summary: stopped_early).",
    "D": "Descriptive: the full runs, the variants and Whisper are outside PREREG; no pre-registered call applies.",
}
# plan v3 section 7, "Caveats printed next to every Whisper number", verbatim
WHISPER_CAVEATS = [
    "**Galgame-neutral was chosen with Kotoba-Whisper v2.0 itself:** a row was kept only if Kotoba's CER was 0.5 or "
    "less (810 of 969) [M]. Kotoba's score there is biased low, and large-v3 and turbo may be mildly favoured. So "
    "every table also shows **M4-all** (Galgame-all instead of neutral), and the Whisper-vs-students chart uses "
    "M4-all.",
    "**Reazon is in-domain for Kotoba** (its test set is a held-out part of Kotoba's training corpus), as it is for "
    "Parakeet [S].",
    "**Whisper's exposure to JSUT and CV8 is undisclosed;** treat CV8 as possibly seen [S].",
    "**The students' training rows were filtered by agreement with Whisper-family transcripts** [M], so the "
    "comparison is not fully independent.",
    "**Output style:** Whisper writes Arabic numerals and its own kana/kanji choices, so no-style M4 is printed next "
    "to M4.",
    "**Hallucination counts** (empty, runaway, truncated outputs) are printed per system; 879 M4 clips are under 2 s.",
    "**Descriptive only:** no pre-registered calls apply to the Whisper rows.",
]
BLOCKS = (("parakeet", "Parakeet: the teachers, the full-data P students and their quantised variants"),
          ("cohere", "Cohere Transcribe: the teacher, the full-data T-0.6B and its quantised variants"),
          ("whisper", "Whisper yardsticks (descriptive; flags W1-W6)"),
          ("study", "The study's students (reference: the same manifest, the same columns)"))
OFFER_COLUMNS = ["model", "format", "params", "file MB", "x smaller than teacher (file)", "M4", "M4-all",
                 "3-set mean", "M4 no-style", "JSUT", "CV8", "Reazon", "Gal-neutral", "Gal-all",
                 "M4 / own teacher [CI]", "vs study [CI]", "RTFx", "p50 ms", "VRAM batched / batch-1 GB",
                 "epochs (early stop)", "flags"]


class InputError(ValueError):
    """An input the report cannot use as given: it refuses (exit 2) instead of computing numbers from it."""


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise InputError(f"{path}: {e}") from e


# this module's private copy of study_report (_load_tool: not the one in sys.modules) reads its JSON through
# read_json, so a broken run summary, config.json or params file refuses naming its path (exit 2)
sr.read_json = read_json


def _refusing(what: str, fn, *args):
    """fn(*args), where an unreadable or malformed input (a missing file, broken JSON or parquet, a field of the
    wrong type) refuses with `what` named (InputError, exit 2) instead of a traceback (exit 1)."""
    try:
        return fn(*args)
    except InputError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        raise InputError(f"{what}: {type(e).__name__}: {e}") from e


def _read_parquet(path: Path, columns=None) -> pd.DataFrame:
    """A parquet file (only `columns` that it has, when given); a broken one refuses (InputError)."""
    def read():
        cols = columns if columns is None else [c for c in columns if c in set(pq.read_schema(path).names)]
        return pd.read_parquet(path, columns=cols)
    return _refusing(str(path), read)


def _id_problems(have: list[str], want: list[str]) -> str | None:
    """None when `have` is exactly the ids `want` (any order, each once), else what differs: 05_evaluate's rule for a
    set it scores (_id_problems there)."""
    seen, dup = set(), set()
    for i in have:
        (dup if i in seen else seen).add(i)
    miss, extra = sorted(set(want) - seen), sorted(seen - set(want))
    if not (dup or miss or extra):
        return None
    return (f"{len(miss)} manifest ids missing (e.g. {miss[:3]}), {len(extra)} not in the manifest (e.g. "
            f"{extra[:3]}), {len(dup)} twice (e.g. {sorted(dup)[:3]})")


def _unix(t) -> float | None:
    """Unix seconds from a number (a queue summary's started) or an ISO string (a speed record's time_utc; naive =
    UTC); None when neither."""
    if t is None or isinstance(t, bool):
        return None
    if isinstance(t, (int, float)):
        return float(t) if math.isfinite(t) else None
    try:
        d = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def _num(x):
    """A finite number or None (JSON fields that may be missing, null or a string)."""
    try:
        return ss._f(float(x)) if x is not None and not isinstance(x, bool) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ readouts


def _greedy_frames(d: Path, sets) -> dict[str, pd.DataFrame]:
    """set -> the greedy_<set>.parquet frame of a 05 / whisper_eval --out dir (only the columns read here)."""
    out = {}
    for f in sorted(d.glob("greedy_*.parquet")):
        s = sr._set_of_file(f.stem, sets)
        if s is None:
            continue
        df = _read_parquet(f, ("id", "ref", "hyp", "truncated", "duration", "n_tok"))
        if "id" in df.columns:
            out[s] = df
    return out


def _table_from_greedy(frames: dict[str, pd.DataFrame], man: ss.Manifest) -> tuple[pd.DataFrame | None, dict]:
    """(a system's per-utterance table, {set: why refused}) from its greedy frames, scored exactly as
    05_evaluate.tables_from_frames scores them (kitsune.evaluate.utterance_table in manifest order: edits and the
    no-style edits on normalize_ja strings). Like 05, a set whose rows are not exactly its manifest ids gets no table
    (the system lacks that set) instead of refusing the whole report in build_corpus."""
    from kitsune import evaluate as ev

    parts, refused = [], {}
    for s, df in frames.items():
        if not {"ref", "hyp"} <= set(df.columns):
            refused[s] = "no ref / hyp columns"
            continue
        want = man.sets[s]
        if p := _id_problems(df["id"].astype(str).tolist(), want):
            refused[s] = p
            continue
        d = df.assign(id=df["id"].astype(str)).set_index("id").loc[want]
        trunc = d["truncated"].tolist() if "truncated" in d.columns else None
        parts.append(ev.utterance_table(want, s, d["ref"].tolist(), d["hyp"].tolist(), trunc))
    return (pd.concat(parts, ignore_index=True) if parts else None), refused


def _table_from_dir(d: Path, sets) -> pd.DataFrame | None:
    """<hit>/tables/<system>/<set>.parquet, as tools/study_report.load_tables reads one system's dir (05 publishes
    only sets it did not refuse, so a mismatch here refuses the report in build_corpus)."""
    parts = []
    for f in sorted(d.glob("*.parquet")):
        s = sr._set_of_file(f.stem, sets)
        if s is not None:
            parts.append(_read_parquet(f, sr.COLUMNS).drop(columns=["source"], errors="ignore").assign(set=s))
    return pd.concat(parts, ignore_index=True) if parts else None


def _file_scored(study: dict) -> bool:
    """A variant scored from its exported file (quant.source "file"), not from memory: only it has its true bytes."""
    return (study.get("quant") or {}).get("source") == "file"


def _simulated(study: dict) -> bool:
    q = study.get("quant") or {}
    return bool(q) and (bool(q.get("simulated")) or q.get("impl") == "emulate" or q.get("format") in UNTIMED_FORMATS)


def _recipe_version(v) -> int:
    """A record's quant recipe version: the number, or 1 (the field absent or not a number: quant code before F4)."""
    return int(v) if _num(v) is not None else 1


def _old_quant_recipe(q) -> str | None:
    """Why a 05 study.json quant block may not be used (QUANT_RECIPE_MIN), or None (no quant block: not a variant)."""
    if not isinstance(q, dict) or not q:
        return None
    v = _recipe_version(q.get("recipe_version"))
    if v < QUANT_RECIPE_MIN:
        return (f"quant recipe version {v}: scored by quant code older than F1 (box 53693389's smoke B #1, 14bfcad); "
                "never used (F4)")
    return None


def load_readouts(dirs, man: ss.Manifest) -> tuple[dict, dict]:
    """(system -> readout dict, notes) from every study.json under the --readouts dirs (module docstring)."""
    found: dict[str, list[dict]] = {}
    notes = dict(read={}, superseded=[], ignored=[], refused_sets=[])
    for d in dirs or []:
        d = Path(d)
        if not d.is_dir():
            raise InputError(f"--readouts {d}: not a directory")
        for f in sorted(d.rglob("study.json")):
            rec = read_json(f)
            system = rec.get("system") if isinstance(rec, dict) else None
            if not isinstance(system, str) or not system:
                notes["ignored"].append(dict(path=str(f), why="no system in study.json"))
                continue
            if (why := _old_quant_recipe(rec.get("quant"))) is not None:  # F4: never a pre-F1 quant readout
                notes["ignored"].append(dict(path=str(f), system=system, why=why))
                continue
            hit = f.parent
            frames = _greedy_frames(hit, man.sets)
            table, source, refused = None, None, {}
            if (hit / "tables" / system).is_dir():
                table, source = _table_from_dir(hit / "tables" / system, man.sets), "tables"
            if table is None and frames:
                (table, refused), source = _table_from_greedy(frames, man), "greedy"
                notes["refused_sets"] += [dict(system=system, dir=str(hit), set=s, why=why)
                                          for s, why in sorted(refused.items())]
            if table is None:
                notes["ignored"].append(dict(path=str(f), why="every greedy set refused" if refused else
                                             "no tables/<system>/<set>.parquet and no greedy_<set>.parquet next "
                                             "to it"))
                continue
            durations, ntok = {}, {}
            for df in frames.values():
                # a null or non-numeric cell is unknown for that row (never a crash: the counts that need it skip it)
                ids = df["id"].astype(str).tolist()
                if "duration" in df.columns:
                    durations.update((i, float(v)) for i, v in zip(ids, pd.to_numeric(df["duration"], errors="coerce"))
                                     if math.isfinite(v))
                if "n_tok" in df.columns:
                    ntok.update((i, int(v)) for i, v in zip(ids, pd.to_numeric(df["n_tok"], errors="coerce"))
                                if math.isfinite(v))
            summ = hit / "summary.json"
            wj = hit / "whisper.json"
            sets = sorted(set(table["set"].astype(str)))
            found.setdefault(system, []).append(dict(
                system=system, dir=str(hit), study=rec, table=table, table_from=source, sets=sets,
                complete=set(man.sets) <= set(sets), simulated=_simulated(rec),
                summary=read_json(summ) if summ.is_file() else None, whisper=read_json(wj) if wj.is_file() else None,
                durations=durations, ntok=ntok, time_utc=str(rec.get("time_utc") or "")))
    out = {}
    for system, cands in found.items():
        # smoke B scores study-p03@<fmt> from its exported file and then from memory (mem-*, later): the accuracy is
        # the same (its check 14), but only the file readout knows the variant's true bytes
        cands.sort(key=lambda c: (c["complete"], not c["simulated"], _file_scored(c["study"]), c["time_utc"]))
        out[system] = cands[-1]
        notes["read"][system] = cands[-1]["dir"]
        notes["superseded"] += [dict(system=system, dir=c["dir"], time_utc=c["time_utc"], complete=c["complete"],
                                     simulated=c["simulated"], file_scored=_file_scored(c["study"]))
                                for c in cands[:-1]]
    return out, notes


# ------------------------------------------------------------------------------------------------ roles


def classify(system: str, ro: dict | None) -> dict:
    """role, family, own teacher, base (a variant's bf16 system), format, display and trained-ness of a system."""
    study = (ro or {}).get("study") or {}
    q = study.get("quant") or {}
    if q:
        base, fmt = q.get("base_system"), q.get("format")
        b = classify(base, None) if isinstance(base, str) and base else dict(family=None, display=system)
        return dict(role="quant", family=b["family"], base=base, format=fmt, trained=True,
                    teacher=TEACHER_OF.get(b["family"]), simulated=_simulated(study),
                    display=f"{b['display']} {FORMAT_LABEL.get(fmt, fmt)}")
    if system in ss.TEACHERS:
        return dict(role="teacher", family=TEACHER_FAMILY[system], base=None, format="bf16", trained=False,
                    teacher=None, simulated=False, display=ss.display(system))
    if study.get("family") == "whisper" or (ro or {}).get("whisper") or system in WHISPER_SYSTEMS:
        dtype = (((ro or {}).get("whisper") or {}).get("model") or {}).get("dtype") or "bf16"
        return dict(role="whisper", family="whisper", base=None, format=str(dtype), trained=False, teacher=None,
                    simulated=False, display=WHISPER_SYSTEMS.get(system, system))
    if system in FULL_RUNS:
        _, fam, label = FULL_RUNS[system]
        return dict(role="full", family=fam, base=None, format="bf16", trained=True, teacher=TEACHER_OF[fam],
                    simulated=False, display=f"{label} full")
    b = ss.base_run(system)
    role = ("half" if system.endswith(ss.HALF) else "study" if b in ss.RUNS else "anchor" if b == ss.ANCHOR
            else "other")
    fam = ss.family_of(system) if role != "other" else None
    return dict(role=role, family=fam, base=None, format="bf16", trained=role != "other",
                teacher=TEACHER_OF.get(fam or ""), simulated=False,
                display=f"{ss.display(system)} study" if role in ("study", "half") else ss.display(system))


def study_of(system: str, inf: dict) -> str | None:
    """The study system whose timed weights stand in for a full-data one (same shape, format): full-x -> study-x,
    full-x@fmt -> study-x@fmt; None for everything else."""
    if inf["role"] == "full":
        return FULL_RUNS[system][0]
    if inf["role"] == "quant" and inf["base"] in FULL_RUNS and inf["format"]:
        return f"{FULL_RUNS[inf['base']][0]}@{inf['format']}"
    return None


# ------------------------------------------------------------------------------------------------ statistics


class Stats:
    """The corpus' point sums and paired bootstrap, and the metrics and comparisons of this report on them."""

    def __init__(self, corpus: ss.Corpus, B: int, seed: int, z: float = ss.Z):
        self.corpus, self.z = corpus, float(z)
        self.point = ss.point_sums(corpus)
        self.boot = ss.bootstrap_sums(corpus, int(B), int(seed))
        self._cache = {}
        rep, orig = ss.DEFAULT_SETTINGS["replicate"]
        self.sigma = ss.sigma_run(self.value(rep, "m4"), self.value(orig, "m4"))
        self.s = self.sigma["sigma_run"]

    def values(self, name: str, boot: bool = False):
        """(..., S) values of a FULL_METRICS entry (or GAP); None when one of its strata is not in the corpus."""
        key = (name, boot)
        if key not in self._cache:
            sums = self.boot if boot else self.point
            if name == GAP:
                v = None
                if all(s in sums.ref for s in ("galgame_all", "galgame_neutral")):
                    v = ss.stratum_cer(sums, "galgame_all") - ss.stratum_cer(sums, "galgame_neutral")
            else:
                agg, strata, nostyle = FULL_METRICS[name]
                v = None
                if all(s in sums.ref for s in strata):
                    E = sums.nostyle if nostyle else sums.edits
                    with np.errstate(invalid="ignore", divide="ignore"):
                        if agg == "macro":
                            v = np.mean([E[s] / sums.ref[s][..., None] for s in strata], axis=0)
                        else:
                            v = sum(E[s] for s in strata) / sum(sums.ref[s] for s in strata)[..., None]
            self._cache[key] = v
        return self._cache[key]

    def value(self, system: str, name: str):
        v = self.values(name)
        if v is None or system not in self.corpus.systems:
            return None
        return ss._f(v[self.corpus.index(system)])

    def stratum(self, system: str, name: str, nostyle: bool = False):
        if name not in self.point.ref or system not in self.corpus.systems:
            return None
        return ss._f(ss.stratum_cer(self.point, name, nostyle)[self.corpus.index(system)])

    def compare(self, a: str, b: str, metric: str, n_noisy: int) -> dict | None:
        """r = metric(a) / metric(b) with its CI, n_noisy trained systems' sigma_run^2 in the variance (module
        docstring); verdict better / within / worse by the CI of ln r against 0."""
        pa, pb = self.value(a, metric), self.value(b, metric)
        if pa is None or pb is None or pa <= 0 or pb <= 0:
            return None
        bv = self.values(metric, True)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.log(bv[:, self.corpus.index(a)]) - np.log(bv[:, self.corpus.index(b)])
        fin = np.isfinite(lb)
        v_boot = float(np.var(lb[fin], ddof=1)) if fin.sum() > 1 else float("nan")
        ln_r = math.log(pa / pb)
        lo, hi, se = ss.ci(ln_r, v_boot, self.s, int(n_noisy), self.z)
        return dict(a=a, b=b, metric=metric, a_value=pa, b_value=pb, ratio=pa / pb, delta=pa - pb, ln_ratio=ln_r,
                    ci_ln=[lo, hi], ci_ratio=[math.exp(lo), math.exp(hi)], v_boot=v_boot, se=se, n_noisy=int(n_noisy),
                    verdict="better" if hi < 0 else "worse" if lo > 0 else "within")


def kl_of(summary: dict | None) -> dict | None:
    """KL to the own teacher from a 05 summary.json: per gate set and token-weighted over those present."""
    sets = (((summary or {}).get("tf") or {}).get("sets") or {})
    per = {s: dict(kl=_num((sets.get(s) or {}).get("kl")), n_tok=_num((sets.get(s) or {}).get("n_tok")))
           for s in M3_STRATA if s in sets}
    per = {s: v for s, v in per.items() if v["kl"] is not None and v["n_tok"]}
    if not per:
        return None
    n = sum(v["n_tok"] for v in per.values())
    return dict(gate=sum(v["kl"] * v["n_tok"] for v in per.values()) / n, sets=per,
                complete=set(per) == set(M3_STRATA))


# ------------------------------------------------------------------------------------------------ utterance counts


def _nk(text: str) -> int:
    return len(KANJI.findall(text))


def halluc_counts(df: pd.DataFrame, man: ss.Manifest, durations: dict) -> dict:
    """The hallucination counts of one system (module docstring) over the M4 strata, M4-all, the short clips of
    each, and Galgame's empty references; plus JSUT's kana mode."""
    strata = ss._strata_ids(man)
    df = df.assign(id=df["id"].astype(str), set=df["set"].astype(str))
    by = {s: g.set_index("id") for s, g in df.groupby("set", sort=False)}
    phrases = [normalize_ja(p) for p in SILENCE_PHRASES]

    def rows(names):
        parts = []
        for name in names:
            s, ids = strata.get(name, (None, None))
            if s is None or s not in by:
                return None
            parts.append(by[s].loc[ids])
        return pd.concat(parts)

    def count(t: pd.DataFrame | None) -> dict | None:
        if t is None:
            return None
        rl, ed = t["ref_len"].to_numpy(np.int64), t["edits"].to_numpy(np.int64)
        # a table of counts only (no text): the output length is unknown, so is every count that needs it
        hl = t["hyp_len"].to_numpy(np.int64) if "hyp_len" in t.columns else None
        keep = rl > 0
        hyp = [h if isinstance(h, str) else "" for h in (t["hyp"] if "hyp" in t.columns else [""] * len(t))]
        ref = [normalize_ja(r) if isinstance(r, str) else "" for r in (t["ref"] if "ref" in t.columns
                                                                       else [""] * len(t))]
        tr = t["truncated"].to_numpy().astype(bool) if "truncated" in t.columns else np.zeros(len(t), bool)
        # a phrase the reference itself holds is speech, not a hallucination
        sil = np.array([any(p in normalize_ja(h) and p not in r for p in phrases) for h, r in zip(hyp, ref)], bool)
        br = np.array([bool(BRACKETS.search(h)) for h in hyp], bool)
        k = keep
        text = hl is not None
        return dict(n=int(k.sum()), n_empty=int((k & (hl == 0)).sum()) if text else None,
                    n_runaway=int((k & ((ed > rl) | (hl > 2 * rl))).sum()) if text else None,
                    n_truncated=int((k & tr).sum()), n_silence=int((k & sil).sum()) if text else None,
                    n_bracketed=int((k & br).sum()) if text else None)

    def short(t: pd.DataFrame | None) -> dict | None:
        if t is None or not durations:
            return None
        dur = np.array([durations.get(i, np.nan) for i in t.index], np.float64)
        known = np.isfinite(dur)
        if not known.any():
            return None
        c = count(t[known & (dur < SHORT_CLIP_S)])
        return dict(c, n_duration_known=int(known.sum()))

    m4, m4_all = rows(ss.M4_SETS), rows(M4_ALL_STRATA)
    out = dict(m4=count(m4), m4_all=count(m4_all), m4_short=short(m4), m4_all_short=short(m4_all))
    if ss.GALGAME in by and "hyp_len" in by[ss.GALGAME].columns:
        g = by[ss.GALGAME]
        empty = g["ref_len"].to_numpy(np.int64) == 0
        out["empty_ref"] = dict(n=int(empty.sum()),
                                nonempty_on_empty_ref=int((empty & (g["hyp_len"].to_numpy(np.int64) > 0)).sum()))
    if "eval_jsut" in by and {"ref", "hyp"} <= set(by["eval_jsut"].columns):
        j = by["eval_jsut"]
        refs = [normalize_ja(r if isinstance(r, str) else "") for r in j["ref"]]
        hyps = [normalize_ja(h if isinstance(h, str) else "") for h in j["hyp"]]
        elig = np.array([_nk(r) >= KANA_MIN_KANJI for r in refs], bool)
        km = elig & np.array([bool(h) and _nk(h) == 0 for h in hyps], bool)
        out["kana_mode"] = dict(eligible=int(elig.sum()), n=int(km.sum()),
                                rate=float(km.sum() / elig.sum()) if elig.sum() else None)
    return out


def kotoba_bias(path: Path, tables: dict[str, pd.DataFrame], man: ss.Manifest) -> dict:
    """Decision 29: Kotoba-Whisper v2.0's STORED Galgame hypotheses (the judge file that chose Galgame-neutral)
    scored on the 810 neutral rows and on every Galgame row with a reference, on the tables' references."""
    raw = _refusing(f"--kotoba-jsonl {path}", Path(path).read_bytes)
    sha = hashlib.sha256(raw).hexdigest()
    if sha != KOTOBA_JSONL_SHA256:
        raise InputError(f"--kotoba-jsonl {path}: sha256 {sha[:12]}, not the judge file's {KOTOBA_JSONL_SHA256[:12]}")
    rows = _refusing(f"--kotoba-jsonl {path}",
                     lambda: [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()])
    hyp2 = {str(r["id"]): r.get("hyp2") for r in rows if isinstance(r, dict) and "id" in r}
    model2 = {}
    for r in rows:
        model2[str(r.get("model2"))] = model2.get(str(r.get("model2")), 0) + 1
    ref = None
    for t in tables.values():
        g = t[t["set"].astype(str) == ss.GALGAME]
        if "ref" in g.columns and len(g):
            ref = dict(zip(g["id"].astype(str), g["ref"]))
            break
    out = dict(sha256=sha, n_rows=len(rows), model2=model2)
    if ref is None or ss.GALGAME not in man.sets:
        return dict(out, available=False, reason="no Galgame table with reference text")
    ids = [i for i in man.sets[ss.GALGAME] if i in hyp2]
    sc = ss.score_utterances([ref.get(i, "") for i in ids], [hyp2[i] or "" for i in ids]).assign(id=ids)
    sc = sc[sc["ref_len"] > 0].set_index("id")

    def cer(view_ids):
        t = sc.loc[[i for i in view_ids if i in sc.index]]
        return (float(t["edits"].sum() / t["ref_len"].sum()) if len(t) else None), int(len(t))

    neutral = man.views.get("neutral") or []
    c_n, n_n = cer(neutral)
    c_a, n_a = cer(man.sets[ss.GALGAME])
    return dict(out, available=True, cer_neutral=c_n, n_neutral=n_n, cer_all=c_a, n_all=n_a,
                gap_pp=None if c_n is None or c_a is None else 100 * (c_a - c_n),
                missing=len(man.sets[ss.GALGAME]) - len(ids))


# ------------------------------------------------------------------------------------------------ speed


def load_queue_summaries(paths, speed_paths) -> list[dict]:
    """[{path, box, machine_id, started, claims, speed_items}] of the full boxes' queue summaries (CONTRACT.md 5): the
    --queue-summaries files and the queue_summary.json under the given dirs, plus the one at the runs repo's layout
    beside each speed file (<root>/runs/speed-<box>-<stamp>/speed.json -> <root>/full/box-<box>/queue_summary.json)
    when it exists. claims = the speed dirs the summary names (its speed_dir, its speed items' out); speed_items =
    its speed items' {item, status, why, system} (speed_probes lists the ones not done)."""
    files = []
    for p in paths or []:
        p = Path(p)
        if p.is_dir():
            files += sorted(p.rglob("queue_summary.json"))
        elif p.is_file():
            files.append(p)
        else:
            raise InputError(f"--queue-summaries {p}: no such file or directory")
    for p in speed_paths or []:
        m = SPEED_DIR_RE.match(Path(p).parent.name)
        f = Path(p).parent.parent.parent / "full" / f"box-{m['box']}" / "queue_summary.json" if m else None
        if f is not None and f.is_file():
            files.append(f)
    out, seen = [], set()
    for f in files:
        if str(f.resolve()) in seen:
            continue
        seen.add(str(f.resolve()))
        doc = read_json(f)
        items = (doc.get("items") or {}) if isinstance(doc, dict) else None
        if not isinstance(items, dict) or not isinstance(doc.get("box"), str):
            raise InputError(f"{f}: not a full box's queue summary (no box, or items not a mapping)")
        claims = {Path(doc["speed_dir"]).name} if isinstance(doc.get("speed_dir"), str) else set()
        speed_items = []
        for name, it in sorted(items.items()):
            if isinstance(it, dict) and it.get("kind") == "speed":
                res = it.get("result") if isinstance(it.get("result"), dict) else {}
                claims |= {Path(o).name for o in (it.get("out"), res.get("out")) if isinstance(o, str) and o}
                speed_items.append(dict(item=name, status=it.get("status"), why=it.get("why"),
                                        system=res.get("system")))
        mid = doc.get("machine_id")
        out.append(dict(path=str(f), box=doc["box"], machine_id=None if mid in (None, "") else str(mid),
                        started=_unix(doc.get("started")), claims=sorted(claims), speed_items=speed_items))
    return out


def parse_machine_of(pairs) -> dict[str, str]:
    """--machine-of NAME=ID pairs -> {NAME: ID}."""
    out = {}
    for p in pairs or []:
        name, sep, mid = str(p).partition("=")
        if not sep or not name.strip() or not mid.strip():
            raise InputError(f"--machine-of {p!r}: expected NAME=MACHINE_ID (NAME a speed-<box>-<stamp> dir or a "
                             "container host)")
        out[name.strip()] = mid.strip()
    return out


def machine_of(rec: dict, file: str, qs: list[dict], override: dict[str, str]) -> tuple[str | None, str]:
    """(machine, where from) of one speed record (module docstring, --speed): versions.machine_id; else --machine-of
    by speed dir or host; else the one machine_id of the queue summaries that cover the file (they name its dir or are
    its box's, and their launch started before the record was timed: a box launched again on another host replaces
    its summary, and the earlier launch's records predate the new one); else versions.host."""
    v = rec.get("versions") if isinstance(rec.get("versions"), dict) else {}
    if v.get("machine_id"):
        return str(v["machine_id"]), "versions.machine_id"
    d = Path(file).parent.name
    host = v.get("host") or None
    for k in (d, host):
        if k and k in override:
            return override[k], f"--machine-of {k}"
    m = SPEED_DIR_RE.match(d)
    t = _unix(rec.get("time_utc"))
    if t is None and m:  # the dir's stamp: the box's first speed attempt, at or before every record in it
        try:
            t = datetime.strptime(m["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            t = None
    cover = []
    for q in qs:
        named, same_box = d in q["claims"], m is not None and m["box"] == q["box"]
        if not q["machine_id"] or not (named or same_box):
            continue
        if q["started"] is not None and t is not None:
            if t < q["started"]:
                continue  # timed before that launch started: another launch's (and maybe another host's) record
        elif not named:
            continue  # no times to compare: only a summary that names this very dir
        cover.append(q)
    mids = sorted({q["machine_id"] for q in cover})
    if len(mids) == 1:
        return mids[0], "queue summary " + ", ".join(sorted({q["path"] for q in cover}))
    if host:
        return host, "versions.host" + (f" (the queue summaries disagree: {mids})" if mids else "")
    return None, "unknown"


def _speed_variant(name: str, rec: dict) -> tuple[str | None, bool]:
    """(the quantised format or None, compiled) of a speed record: by its name (<base>@<fmt>[+compile], as
    kitsune.quant.split_system) or else its quant / compile fields."""
    s = name[:-len("+compile")] if name.endswith("+compile") else name
    base, sep, fmt = s.rpartition("@")
    if not (sep and base and fmt in QUANT_FORMATS):
        fmt = rec.get("quant") if rec.get("quant") in QUANT_FORMATS else None
    return fmt, name.endswith("+compile") or bool(rec.get("compile"))


def speed_excluded(name: str, rec: dict) -> str | None:
    """Why a speed record is left out (F4), or None: a quantised record timed by quant code older than F1
    (quant_recipe.version below QUANT_RECIPE_MIN; absent = 1: box 53693389's smoke B #1, whose fp8 records decoded
    garbage), or a quantised or compiled record without a passing CER sanity (speed_probe's sanity against the same
    weights' bf16 decode: absent before F4, ok false when the model decoded garbage)."""
    fmt, compiled = _speed_variant(name, rec)
    if fmt:
        qr = rec.get("quant_recipe") if isinstance(rec.get("quant_recipe"), dict) else {}
        v = _recipe_version(qr.get("version"))
        if v < QUANT_RECIPE_MIN:
            return (f"quant recipe version {v}: timed by quant code older than F1 (box 53693389's smoke B #1, 14bfcad, "
                    "whose fp8 records decoded garbage); never used (F4)")
    if fmt or compiled:
        sanity = rec.get("sanity")
        if not isinstance(sanity, dict):
            return "no CER sanity record (speed_probe before F4): a quantised or compiled speed needs a passing one"
        if sanity.get("ok") is not True:
            return f"CER sanity failed: {sanity.get('reason') or 'not ok'}"
    return None


def _speed_failures(p, doc: dict) -> list[dict]:
    """The probes a speed file and its dir record as failed: the file's failed block (speed_probe: a load, probe or
    reference that raised) and the dir's events.jsonl (the queue's line per speed item that ran; the last line of a
    system counts, failed with its why: "exit 4" is an insane CER, "exit 1" a crash), one entry per system and dir
    ({system, file, why, stage, rc, item, time})."""
    out: dict[str, dict] = {}
    fails = doc.get("failed")
    for name, info in sorted((fails if isinstance(fails, dict) else {}).items()):
        info = info if isinstance(info, dict) else {}
        out[name] = dict(system=name, file=str(p), why=str(info.get("error") or "failed")[:300],
                         stage=info.get("stage"), rc=None, item=None, time=_unix(info.get("time_utc")))
    ev = Path(p).parent / "events.jsonl"
    if ev.is_file():
        last: dict[str, dict] = {}
        for line in ev.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("kind") == "speed" and isinstance(r.get("system"), str):
                last[r["system"]] = r
        for name, r in sorted(last.items()):
            if r.get("status") != "failed":
                continue
            e = out.setdefault(name, dict(system=name, file=str(ev), why=str(r.get("why") or f"exit {r.get('rc')}"),
                                          stage=None, rc=None, item=None, time=None))
            times = [t for t in (e["time"], _unix(r.get("wall"))) if t is not None]
            e.update(rc=r.get("rc"), item=r.get("item"), time=max(times) if times else None)
    return list(out.values())


def load_speed(paths, qs: list[dict] | None = None, override: dict[str, str] | None = None) -> dict:
    """{records: {(group, name): record}, ids, ids_sha256, files, superseded, groups, primary, excluded, failed,
    failures_resolved} from the speed files. One id list for all (else InputError). A group is machine (machine_of) +
    GPU; one record per name and group (the newest time_utc; the others listed). A record speed_excluded names is
    never used (excluded: group, system, file, time_utc, why). failed: the probes the files and their dirs record as
    failed (_speed_failures), unless a used record of that system is newer (failures_resolved)."""
    recs, sup, files, excluded, fails = {}, [], [], [], []
    ids = sha = first = None
    for p in paths or []:
        doc = read_json(p)
        systems = doc.get("systems") if isinstance(doc, dict) else None
        if not isinstance(systems, dict):
            raise InputError(f"--speed {p}: no systems block (not a tools/speed_probe.py output)")
        h = doc.get("ids_sha256")
        if first is None:
            sha, ids, first = h, list(doc.get("ids") or []), p
        elif h != sha:
            raise InputError(f"--speed: {p} was timed on another id list ({str(h)[:12]}) than {first} "
                             f"({str(sha)[:12]}): every system of one report is timed on the same audio")
        files.append(str(p))
        fails += _speed_failures(p, doc)
        for name, rec in sorted(systems.items()):
            if not isinstance(rec, dict) or _num(rec.get("rtf")) is None:
                continue
            machine, where = machine_of(rec, str(p), qs or [], override or {})
            gpu = rec.get("gpu") or (str(rec.get("device")) if rec.get("device") else None)
            key = (f"{machine or '?'} / {gpu or '?'}", name)
            if (why := speed_excluded(name, rec)) is not None:
                excluded.append(dict(group=key[0], system=name, file=str(p), time_utc=rec.get("time_utc"), why=why))
                continue
            rec = dict(rec, _file=str(p), _machine_from=where)
            old = recs.get(key)
            if old is None or str(rec.get("time_utc") or "") > str(old.get("time_utc") or ""):
                if old is not None:
                    sup.append(dict(group=key[0], system=name, file=old["_file"], time_utc=old.get("time_utc")))
                recs[key] = rec
            else:
                sup.append(dict(group=key[0], system=name, file=str(p), time_utc=rec.get("time_utc")))
    groups: dict[str, dict] = {}
    for (g, name), rec in sorted(recs.items()):
        e = groups.setdefault(g, dict(systems=[], compiled=[], files=set(), gpu=rec.get("gpu"),
                                      machine=g.rsplit(" / ", 1)[0], machine_from=set()))
        (e["compiled"] if name.endswith("+compile") else e["systems"]).append(name)
        e["files"].add(rec["_file"])
        e["machine_from"].add(rec["_machine_from"])
    for e in groups.values():
        e["files"], e["machine_from"] = sorted(e["files"]), sorted(e["machine_from"])
    primary = max(sorted(groups), key=lambda g: len(groups[g]["systems"])) if groups else None
    newest: dict[str, float] = {}
    for (_, name), rec in recs.items():
        t = _unix(rec.get("time_utc"))
        if t is not None:
            newest[name] = max(newest.get(name, t), t)
    failed, resolved = [], []
    for f in fails:  # a failure a later good probe of the system superseded is listed apart, not as a failure
        (resolved if f["time"] is not None and newest.get(f["system"], -math.inf) > f["time"] else failed).append(f)
    return dict(records=recs, ids=ids or [], ids_sha256=sha, files=files, superseded=sup, groups=groups,
                primary=primary, excluded=excluded, failed=failed, failures_resolved=resolved)


def speed_probes(rep: dict, sp: dict, qs: list[dict]) -> dict:
    """The probes the report could not use (F4: no speed row is silently empty): failed (load_speed: crashed, or
    exit 4 for an insane CER), excluded (speed_excluded: an old quant recipe, a missing or failed CER sanity), absent
    (offered rows without any speed record, flag S5) and queue_not_done (speed items of the queue summaries whose
    status is neither done nor not_needed: failed, skipped by the no-start rule, never run), plus
    failures_resolved."""
    absent = [r["system"] for r in rep["offer"]["rows"] if "S5" in (r.get("flags") or [])]
    not_done = [dict(it, box=q["box"], path=q["path"]) for q in qs for it in q.get("speed_items") or []
                if it.get("status") not in ("done", "not_needed")]
    return dict(failed=sp["failed"], excluded=sp["excluded"], absent=absent, queue_not_done=not_done,
                failures_resolved=sp["failures_resolved"])


def speed_fields(rec: dict) -> dict:
    rtf = _num(rec.get("rtf"))
    p50, p95 = _num(rec.get("p50_s")), _num(rec.get("p95_s"))
    vb = _num(rec.get("vram_peak_reserved_bytes"))
    v1 = _num(rec.get("vram_peak_reserved_bytes_1"))
    return dict(rtf=rtf, rtfx=1 / rtf if rtf else None, p50_ms=None if p50 is None else 1000 * p50,
                p95_ms=None if p95 is None else 1000 * p95,
                vram_gb=vb / 1e9 if vb is not None else _num(rec.get("vram_gb")),
                vram_1_gb=v1 / 1e9 if v1 is not None else None, weights_bytes=_num(rec.get("weights_bytes")),
                params_total=_num(rec.get("params_total")), tokens_per_utt=_num(rec.get("tokens_per_utt")),
                emulated=bool(rec.get("emulated")), time_utc=rec.get("time_utc"), file=rec.get("_file"),
                machine_from=rec.get("_machine_from"))


def retime_pair(sp: dict, full: str, study: str) -> tuple[str, dict, dict] | None:
    """Box full's re-time pair (decision 22): the full-data and the study weights timed by one box launch, so in one
    speed file (runs/speed-<box>-<stamp>: one machine, one time) and one group; the newest such pair, None without one.
    Any group counts, the chart group too: when box full rented the chart's machine, full-t06 has its own record there
    and only its variants, which fall back to their study variants, need the ratio."""
    best = None
    for g in sorted(sp["groups"]):
        a, b = sp["records"].get((g, full)), sp["records"].get((g, study))
        if a and b and a["_file"] == b["_file"] and (
                best is None or str(a.get("time_utc") or "") > str(best[1].get("time_utc") or "")):
            best = (g, a, b)
    return best


def resolve_speed(system: str, inf: dict, sp: dict, drift: dict | None) -> dict:
    """The speed a system's row shows (module docstring): own record in the chart group, the study weights' (S1),
    another group (S3), the S2 re-time, none (S5), simulated (S4)."""
    if inf["role"] == "quant" and (inf["simulated"] or inf["format"] in UNTIMED_FORMATS):
        return dict(available=False, simulated=True, flags=["S4"], text="n/a (simulated)")
    names = [system] + ([study_of(system, inf)] if study_of(system, inf) else [])
    prim = sp["primary"]
    hit = None
    for n in names:
        if prim and (prim, n) in sp["records"]:
            hit = (prim, n)
            break
    if hit is None:
        for n in names:
            for g in sorted(sp["groups"]):
                if (g, n) in sp["records"]:
                    hit = (g, n)
                    break
            if hit:
                break
    if hit is None:
        return dict(available=False, simulated=False, flags=["S5"], text="n/a")
    g, n = hit
    rec = sp["records"][hit]
    out = dict(speed_fields(rec), available=True, simulated=False, group=g, source=n, flags=[],
               in_chart_group=g == prim)
    if n != system:
        out["flags"].append("S1")
    if g != prim:
        out["flags"].append("S3")
    if out["emulated"]:
        return dict(available=False, simulated=True, flags=["S4"], text="n/a (simulated)", source=n, group=g)
    if drift and drift.get("fires") and n != system:
        out["flags"].append("S2")
        out["drift"] = drift
        # the re-time pair is bf16 only (decision 22): a variant of full-t06 decodes the full-data weights' tokens
        # too, so it takes the same bf16 ratio, and a variant and its bf16 row stay on one basis (x bf16 speed, joins)
        full = system if inf["role"] == "full" else inf["base"]
        stu = FULL_RUNS[full][0] if full in FULL_RUNS else None
        pair = retime_pair(sp, full, stu) if stu else None
        if pair and g == prim:
            pg, a, b = pair
            fa, fb = speed_fields(a), speed_fields(b)
            scale = {k: (fa[k] / fb[k] if fa.get(k) and fb.get(k) else None) for k in ("rtf", "p50_ms", "p95_ms")}
            out["unscaled"] = {k: out[k] for k in ("rtf", "rtfx", "p50_ms", "p95_ms")}
            if scale["rtf"]:
                out["rtf"] = out["rtf"] * scale["rtf"]
                out["rtfx"] = 1 / out["rtf"]
            for k in ("p50_ms", "p95_ms"):
                if scale[k] and out[k] is not None:
                    out[k] = out[k] * scale[k]
            out["retime"] = dict(group=pg, ratio=scale, full=full, study=stu,
                                 note=f"{n}'s chart-group record x the {full}/{stu} ratio timed in {pg}"
                                      + (" (the bf16 pair's ratio, applied to the variant)" if full != system else ""))
        else:
            out["retime"] = dict(group=None, note="no re-time pair (box full's speed-full-t06 and speed-study-t06 "
                                                  "in one speed file): the study weights' speed is shown")
    return out


def token_drift(ntok: dict[str, int], ids: list[str], rec: dict | None) -> dict | None:
    """Mean greedy tokens (EOS included, as speed_probe counts them) over the speed ids in a readout, against the
    timed record's tokens_per_utt; fires over S2_DRIFT."""
    if not ntok or not rec or not ids:
        return None
    have = [ntok[i] for i in ids if i in ntok]
    tpu = _num(rec.get("tokens_per_utt"))
    if not have or not tpu:
        return None
    mean = float(np.mean(have))
    d = mean / tpu - 1
    return dict(readout_tokens_per_utt=mean, timed_tokens_per_utt=tpu, n_ids=len(have), drift=d,
                fires=abs(d) > S2_DRIFT)


# ------------------------------------------------------------------------------------------------ the report


def _file_size(ro: dict | None, speed: dict | None) -> tuple[float | None, str | None]:
    """(bytes, where from). A variant scored from memory (05 --quant on the bf16 --ckpt) has no file of its own: its
    weights.file_bytes are the bf16 checkpoint's, so it takes quant.weights_bytes, what the exporter would write.
    CONTRACT.md 8 leaves that field's shape open; WP5 writes the recipe's bytes {deployable, quantized, kept}, and
    deployable (= quantized + kept) is the file's weight bytes. A plain number is read as it is."""
    st = (ro or {}).get("study") or {}
    q = st.get("quant") or {}
    wb = q.get("weights_bytes")
    wb = wb.get("deployable") if isinstance(wb, dict) else wb
    for v, src in ((q.get("file_bytes"), "quant.file_bytes"),
                   (wb, "in-memory (quant.weights_bytes)"),
                   (None if q else (st.get("weights") or {}).get("file_bytes"), "weights.file_bytes"),
                   ((((ro or {}).get("whisper") or {}).get("model") or {}).get("weights_file_bytes"),
                    "whisper.json")):
        if _num(v):
            return _num(v), src
    if speed and speed.get("weights_bytes"):
        return speed["weights_bytes"], "in-memory (speed record)"
    return None, None


def _params(system: str, inf: dict, params: dict, ro: dict | None, speed: dict | None, info: dict):
    p = {**ss.PARAMS_TOTAL, **(params or {})}
    if system in p:
        return int(p[system])
    if inf["role"] == "full" and FULL_RUNS[system][0] in p:
        return int(p[FULL_RUNS[system][0]])
    if inf["role"] == "quant" and isinstance(inf.get("base"), str):
        b = inf["base"]
        return _params(b, info.get(b) or classify(b, None), params, None, None, info)
    w = _num((((ro or {}).get("whisper") or {}).get("model") or {}).get("params_total"))
    if w:
        return int(w)
    if speed and speed.get("params_total"):
        return int(speed["params_total"])
    return None


def load_all_summaries(dirs) -> tuple[dict, dict]:
    """{run name: the trainer's summary.json} over the --run-summaries dirs (the last dir wins a name)."""
    out, notes = {}, dict(read={}, superseded=[])
    for d in dirs or []:
        if not Path(d).is_dir():
            raise InputError(f"--run-summaries {d}: not a directory")
        summ, n = _refusing(f"--run-summaries {d}", sr.load_summaries, Path(d))
        for name, s in summ.items():
            if not isinstance(s, dict) or not isinstance(s.get("config"), dict):
                continue  # a 05 summary.json (readouts, evals): not a trainer's
            if name in out:
                notes["superseded"].append(notes["read"][name])
            out[name], notes["read"][name] = s, n["read"][name]
        notes["superseded"] += n["superseded"]
    return out, notes


def build_report(args) -> dict:
    man_obj = read_json(args.manifest)
    man = ss.parse_manifest(man_obj)
    man_sha = sr.file_sha256(args.manifest)
    prereg_path = None if args.prereg is None or str(args.prereg).lower() == "none" else Path(args.prereg)
    prereg = read_json(prereg_path) if prereg_path else None
    pchk = ss.manifest_check(man, _refusing(f"--prereg {prereg_path}", ss.prereg_manifest, prereg), man_sha)
    if pchk["status"] == "fail":
        raise InputError(f"the manifest is not the one PREREG.json froze ({prereg_path}): {pchk['detail']}")
    # True: the frozen file and PREREG's hashes; None: the frozen file, PREREG not compared (--prereg none)
    stacks = False if man_sha != fullrun.FROZEN_MANIFEST_SHA256 else True if pchk["status"] == "pass" else None

    tables: dict[str, pd.DataFrame] = {}
    tables_from: dict[str, str] = {}
    for d in args.tables or []:
        try:
            t, _ = _refusing(f"--tables {d}", sr.load_tables, Path(d), man.sets)
        except SystemExit as e:
            raise InputError(f"--tables {d}: {e.code}") from e
        for name, df in t.items():
            if name in tables:
                raise InputError(f"--tables: system {name} is in {tables_from[name]} and in {d}")
            tables[name], tables_from[name] = df, str(d)
    readouts, ro_notes = load_readouts(args.readouts, man)
    for name in sorted(readouts):
        if name in tables:
            ro_notes["superseded"].append(dict(system=name, dir=readouts[name]["dir"],
                                               why=f"a --tables entry of this name ({tables_from[name]}) wins"))
            del readouts[name], ro_notes["read"][name]
            continue
        tables[name], tables_from[name] = readouts[name]["table"], readouts[name]["dir"]
    info = {s: classify(s, readouts.get(s)) for s in tables}
    left_out = sorted(s for s, i in info.items() if i["role"] == "half" and not args.include_half)
    for s in left_out:
        del tables[s], info[s]
    if not tables:
        raise InputError("no system table: give --tables and / or --readouts")
    scored = {s: _refusing(f"the table of {s} ({tables_from[s]})", ss.ensure_scored, df) for s, df in tables.items()}
    corpus = ss.build_corpus(scored, man)
    st = Stats(corpus, args.boot_b, args.seed)

    summaries, summ_notes = load_all_summaries(args.run_summaries)
    params = _refusing(f"--params {args.params}", sr.load_params, args.params) if args.params else {}
    qs = load_queue_summaries(args.queue_summaries, args.speed)
    sp = load_speed(args.speed, qs, parse_machine_of(args.machine_of))
    durations = {}
    for ro in readouts.values():
        durations.update(ro["durations"])

    # per system
    systems = {}
    order = sorted(corpus.systems, key=_sort_key(info))
    for s in order:
        inf, ro = info[s], readouts.get(s)
        drift = None
        stu = study_of(s, inf)
        if stu and inf["family"] == "aed":
            base = s if inf["role"] == "full" else inf["base"]
            base_ro = readouts.get(base)
            prim_rec = sp["records"].get((sp["primary"], FULL_RUNS[base][0])) if sp["primary"] else None
            drift = token_drift((base_ro or {}).get("ntok") or {}, sp["ids"], prim_rec)
        speed = resolve_speed(s, inf, sp, drift)
        size, size_src = _file_size(ro, speed if speed.get("available") else None)
        run = inf["base"] if inf["role"] == "quant" else s
        summ = summaries.get(run) if isinstance(run, str) else None
        metrics = {m: st.value(s, m) for m in FULL_METRICS}
        metrics[GAP] = st.value(s, GAP)
        sets = {name: dict(cer=st.stratum(s, name), cer_nostyle=st.stratum(s, name, True),
                           n=corpus.desc[s][name].get("n")) for name in corpus.strata if name in corpus.desc[s]}
        hal = halluc_counts(scored[s], man, durations)
        q = ((ro or {}).get("study") or {}).get("quant") or None
        flags = []
        if inf["role"] == "whisper":
            flags += ["W1", "W2", "W3", "W4", "W5", "W6"] if s == KOTOBA else ["W1", "W3", "W4", "W5", "W6"]
        if inf["role"] in ("full", "study", "quant", "half", "anchor") or s.startswith("parakeet"):
            flags.append("R1")
        flags += [f for f in speed["flags"] if f not in flags]
        fb = (q or {}).get("fp32_fallbacks")
        if inf["format"] in Q1_FORMATS and (_num(fb) or (isinstance(fb, (list, dict)) and fb)):
            flags.append("Q1")
        km = hal.get("kana_mode") or {}
        if km.get("rate") is not None and km["rate"] > K1_RATE:
            flags.append("K1")
        if summ and summ.get("stopped_early"):
            flags.append("E1")
        systems[s] = dict(
            display=inf["display"], role=inf["role"], family=inf["family"], teacher=inf["teacher"],
            trained=inf["trained"], base=inf["base"], format=inf["format"], simulated=inf["simulated"],
            source=tables_from[s], table_from=(ro or {}).get("table_from", "tables"),
            params_total=_params(s, inf, params, ro, speed if speed.get("available") else None, info),
            file_bytes=size, file_bytes_from=size_src, metrics=metrics, sets=sets, kl=kl_of((ro or {}).get("summary")),
            halluc=hal, speed=speed, quant=q, weights=((ro or {}).get("study") or {}).get("weights"),
            whisper=_whisper_brief((ro or {}).get("whisper")),
            run=None if not summ else {k: summ.get(k) for k in ("run_id", "status", "steps", "epochs", "stopped_early",
                                                               "early_stop_trigger", "end_reason", "resume_resets")},
            flags=flags)

    comparisons = dict(vs_teacher={}, vs_study={}, vs_bf16={})
    for s, v in systems.items():
        t = v["teacher"]
        if v["trained"] and t and t in systems:
            comparisons["vs_teacher"][s] = {m: st.compare(s, t, m, 1) for m in COMPARE_METRICS}
        if v["role"] == "full" and FULL_RUNS[s][0] in systems:
            comparisons["vs_study"][s] = {m: st.compare(s, FULL_RUNS[s][0], m, 2) for m in COMPARE_METRICS}
        if v["role"] == "quant" and v["base"] in systems:
            c = {m: st.compare(s, v["base"], m, 0) for m in COMPARE_METRICS}
            kb, kv = systems[v["base"]]["kl"], v["kl"]
            c["kl_delta"] = (kv["gate"] - kb["gate"]) if kb and kv else None
            comparisons["vs_bf16"][s] = c

    rep = dict(
        descriptive=FLAGS["D"],
        manifest=dict(path=str(args.manifest), sha256=man_sha, stacks_on_study=stacks, prereg=_s(prereg_path),
                      prereg_check=pchk),
        settings=dict(boot_b=int(args.boot_b), seed=int(args.seed), sigma_run=st.sigma, z=st.z,
                      short_clip_s=SHORT_CLIP_S, silence_phrases=list(SILENCE_PHRASES),
                      bracket_pattern=BRACKETS.pattern, kana_min_kanji=KANA_MIN_KANJI, k1_rate=K1_RATE,
                      s2_drift=S2_DRIFT, metrics={k: dict(agg=a, strata=list(x), nostyle=n)
                                                  for k, (a, x, n) in FULL_METRICS.items()},
                      bootstrap_strata={k: len(v.ref_len) for k, v in corpus.strata.items()},
                      n_empty_ref={k: v.n_empty_ref for k, v in corpus.strata.items()}),
        systems=systems, comparisons=comparisons)
    rep["offer"] = offer_table(rep)
    rep["whisper"] = whisper_block(rep, scored, man, args.kotoba_jsonl)
    rep["speed_groups"] = dict(primary=sp["primary"], groups=sp["groups"], ids_sha256=sp["ids_sha256"],
                               n_ids=len(sp["ids"]), superseded=sp["superseded"], queue_summaries=qs,
                               machine_of=parse_machine_of(args.machine_of))
    rep["chart"] = dict(points=chart_points(rep), files=["chart_error_vs_speed.svg", "chart_error_vs_latency.svg",
                                                         "chart_points.json", "chart_points.csv"])
    rep["speed_probes"] = speed_probes(rep, sp, qs)
    rep["flags_legend"] = FLAGS
    rep["checks"] = checks(rep, readouts, sp, man_sha)
    expected = [*FULL_RUNS, *ss.TEACHERS]
    rep["missing_systems"] = [s for s in expected if s not in systems]
    rep["left_out"] = dict(half=left_out)
    rep["inputs"] = dict(manifest=str(args.manifest), tables=[str(d) for d in args.tables or []],
                         readouts=[str(d) for d in args.readouts or []], readouts_read=ro_notes["read"],
                         readouts_superseded=ro_notes["superseded"], readouts_ignored=ro_notes["ignored"],
                         readouts_refused_sets=ro_notes["refused_sets"], speed=sp["files"],
                         queue_summaries=[q["path"] for q in qs],
                         run_summaries=[str(d) for d in args.run_summaries or []],
                         summaries_read=summ_notes["read"], summaries_superseded=summ_notes["superseded"],
                         params=_s(args.params), kotoba_jsonl=_s(args.kotoba_jsonl),
                         written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return ss._clean(rep)


def _s(x):
    return None if x is None else str(x)


def _whisper_brief(w: dict | None) -> dict | None:
    if not w:
        return None
    m = w.get("model") or {}
    sets = w.get("sets") or {}
    tot = {k: sum(int(v.get(k) or 0) for v in sets.values() if isinstance(v, dict))
           for k in ("n_truncated", "n_repetition", "n_length", "n_empty_hyp", "n_timestamp_rows")}
    return dict(repo=m.get("repo"), revision=m.get("revision"), dtype=m.get("dtype"), fp32_head=m.get("fp32_head"),
                params_total=m.get("params_total"), weights_file_bytes=m.get("weights_file_bytes"),
                batching=w.get("batching"), totals=tot,
                sets={s: {k: v.get(k) for k in ("n", "cer_corpus", "n_truncated", "n_repetition", "n_length",
                                                "n_empty_hyp", "n_timestamp_rows", "rtf")}
                      for s, v in sets.items() if isinstance(v, dict)})


def _block_of(system: str, v: dict) -> str | None:
    fam = v["family"]
    if v["role"] == "whisper":
        return "whisper"
    if v["role"] == "teacher":
        return "parakeet" if fam == "ctc" else "cohere"
    if v["role"] == "full" or (v["role"] == "quant" and v["base"] in FULL_RUNS):
        return "parakeet" if fam == "ctc" else "cohere"
    base = v["base"] if v["role"] == "quant" else system
    if base in ss.STUDENTS and v["role"] in ("study", "quant"):
        return "study"
    return None


def _sort_key(info: dict):
    """Report order: the offer blocks (teachers, full runs by size, each followed by its variants in format order,
    Whisper, the study's students with theirs), then everything else by name."""
    runs = [*ss.TEACHERS, *FULL_RUNS, *WHISPER_SYSTEMS, *ss.STUDENTS, *ss.CONTROLS, ss.ANCHOR]

    def key(s):
        inf = info[s]
        base = inf["base"] if inf["role"] == "quant" and isinstance(inf["base"], str) else s
        b = ss.base_run(base)
        r = runs.index(b) if b in runs else len(runs)
        f = QUANT_FORMATS.index(inf["format"]) + 1 if inf["role"] == "quant" and inf["format"] in QUANT_FORMATS \
            else (len(QUANT_FORMATS) + 1 if inf["role"] == "quant" else 0)
        return (r, base.endswith(ss.HALF), b if r == len(runs) else "", f, s)

    return key


def _mci(c: dict | None, nd: int = 2) -> str:
    if not c:
        return "n/a"
    lo, hi = c["ci_ratio"]
    return f"{c['ratio']:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def pct(x, nd: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def offer_table(rep: dict) -> dict:
    """The offer table: one row per offered system, in BLOCKS, with its numbers raw (report.json, offer.csv;
    offer_cells renders a row for report.md)."""
    sysd, comp = rep["systems"], rep["comparisons"]
    rows = []
    for block, _ in BLOCKS:
        for s, v in sysd.items():
            if _block_of(s, v) != block:
                continue
            t = v["teacher"] or (TEACHER_OF.get(v["family"]) if v["role"] == "teacher" else None)
            t_size = (sysd.get(t) or {}).get("file_bytes") if t else None
            sp = v["speed"]
            sets = v["sets"]
            m = v["metrics"]
            vt = (comp["vs_teacher"].get(s) or {}).get("m4")
            vs = (comp["vs_study"].get(s) or {}).get("m4")
            run = v["run"] or {}
            ep = run.get("epochs")
            row = dict(system=s, block=block, display=v["display"], role=v["role"], format=v["format"],
                       params=v["params_total"], file_bytes=v["file_bytes"], file_bytes_from=v["file_bytes_from"],
                       smaller_than_teacher=(t_size / v["file_bytes"] if t_size and v["file_bytes"] else None),
                       m4=m["m4"], m4_all=m["m4_all"], m3=m["m3"], m4_nostyle=m["m4_nostyle"],
                       jsut=(sets.get("eval_jsut") or {}).get("cer"), cv8=(sets.get("eval_cv8") or {}).get("cer"),
                       reazon=(sets.get("eval_reazon") or {}).get("cer"),
                       gal_neutral=(sets.get("galgame_neutral") or {}).get("cer"),
                       gal_all=(sets.get("galgame_all") or {}).get("cer"),
                       vs_teacher=None if not vt else dict(ratio=vt["ratio"], ci=vt["ci_ratio"]),
                       vs_study=None if not vs else dict(ratio=vs["ratio"], ci=vs["ci_ratio"]),
                       rtfx=sp.get("rtfx"), p50_ms=sp.get("p50_ms"), vram_gb=sp.get("vram_gb"),
                       vram_1_gb=sp.get("vram_1_gb"), speed_text=sp.get("text"), epochs=ep,
                       stopped_early=run.get("stopped_early"), flags=v["flags"])
            rows.append(row)
    return dict(columns=OFFER_COLUMNS, rows=rows)


def offer_cells(r: dict) -> list[str]:
    """One offer row as the report.md cells (OFFER_COLUMNS)."""
    def mb(x):
        return "n/a" if x is None else f"{x / 1e6:,.0f}"

    speed_na = r.get("speed_text") or "n/a"
    size = mb(r["file_bytes"]) + (" (in-memory)" if (r["file_bytes_from"] or "").startswith("in-memory") else "")
    ep = "" if r["epochs"] is None else f"{r['epochs']:.2f}" + (" (early stop)" if r["stopped_early"] else "")
    vram = ("n/a" if r["vram_gb"] is None else f"{r['vram_gb']:.2f}") + " / " + \
        ("n/a" if r["vram_1_gb"] is None else f"{r['vram_1_gb']:.2f}")
    return [r["display"], r["format"], "n/a" if r["params"] is None else f"{r['params'] / 1e6:,.0f}M", size,
            "n/a" if r["smaller_than_teacher"] is None else f"{r['smaller_than_teacher']:.1f}",
            pct(r["m4"]), pct(r["m4_all"]), pct(r["m3"]), pct(r["m4_nostyle"]), pct(r["jsut"]), pct(r["cv8"]),
            pct(r["reazon"]), pct(r["gal_neutral"]), pct(r["gal_all"]),
            _mci(dict(ratio=r["vs_teacher"]["ratio"], ci_ratio=r["vs_teacher"]["ci"]) if r["vs_teacher"] else None),
            _mci(dict(ratio=r["vs_study"]["ratio"], ci_ratio=r["vs_study"]["ci"]) if r["vs_study"] else None),
            speed_na if r["rtfx"] is None else f"{r['rtfx']:,.0f}",
            speed_na if r["p50_ms"] is None else f"{r['p50_ms']:.1f}",
            speed_na if r["vram_gb"] is None and r["vram_1_gb"] is None else vram, ep, " ".join(r["flags"])]


def whisper_block(rep: dict, tables: dict, man: ss.Manifest, kotoba_jsonl) -> dict:
    """The Whisper section's data: the caveats, Kotoba's bias from its stored judge file (optional input) and its
    re-decode's rows over CER 0.5 on the neutral view."""
    out = dict(caveats=WHISPER_CAVEATS, systems=[s for s, v in rep["systems"].items() if v["role"] == "whisper"],
               kotoba_bias=None, kotoba_redecode_over_half=None)
    if kotoba_jsonl:
        out["kotoba_bias"] = kotoba_bias(Path(kotoba_jsonl), tables, man)
    if KOTOBA in tables and man.views.get("neutral"):
        t = tables[KOTOBA]
        g = t[t["set"].astype(str) == ss.GALGAME].assign(id=lambda d: d["id"].astype(str)).set_index("id")
        g = g.loc[[i for i in man.views["neutral"] if i in g.index]]
        g = g[g["ref_len"] > 0]
        over = (g["edits"] / g["ref_len"]) > 0.5
        out["kotoba_redecode_over_half"] = dict(n=int(over.sum()), of=int(len(g)))
    return out


def checks(rep: dict, readouts: dict, sp: dict, man_sha: str) -> list[dict]:
    """What the inputs show about how comparable the numbers are: pass / fail / not_checked, like the study's."""
    out = []
    m = rep["manifest"]
    st = m["stacks_on_study"]
    out.append(dict(rule="stacks_on_study", status="pass" if st else "not_checked" if st is None else "fail",
                    detail=f"manifest sha256 {man_sha[:12]} (frozen {fullrun.FROZEN_MANIFEST_SHA256[:12]}); PREREG "
                           f"check {m['prereg_check']['status']}: {m['prereg_check']['detail']}"))
    bad = sorted(s for s, ro in readouts.items()
                 if ((ro["study"].get("manifest") or {}).get("sha256") or man_sha) != man_sha)
    out.append(dict(rule="readout_manifests", status="fail" if bad else ("pass" if readouts else "not_checked"),
                    detail=(f"scored against another manifest file: {bad}" if bad else
                            f"all {len(readouts)} readouts name this manifest's sha256" if readouts else
                            "no readouts")))
    if not sp["files"]:
        out.append(dict(rule="speed_ids", status="not_checked", detail="no speed file"))
    else:
        same = sp["ids_sha256"] == STUDY_SPEED_IDS_SHA256
        out.append(dict(rule="speed_ids", status="pass" if same else "fail",
                        detail=f"{len(sp['ids'])} ids, sha256 {str(sp['ids_sha256'])[:12]}"
                               + (": the study's 200-id list" if same else
                                  f", not the study's ({STUDY_SPEED_IDS_SHA256[:12]})")))
        # another group is expected (box full's re-time pair feeds the S2 ratio); a ROW timed there is not comparable
        s3 = sorted(s for s, v in rep["systems"].items() if "S3" in v["flags"])
        keyed = "; ".join(f"{g}: machine from {', '.join(e['machine_from'])}" for g, e in sp["groups"].items())
        by_host = len(sp["groups"]) > 1 and any("versions.host" in w for e in sp["groups"].values()
                                                for w in e["machine_from"])
        out.append(dict(rule="speed_groups", status="fail" if s3 else "pass",
                        detail=f"{len(sp['groups'])} machine/GPU group(s); the chart group is {sp['primary']}"
                               + (f"; rows timed only elsewhere (S3, not drawn): {s3}" if s3 else
                                  "; every row with a speed has it from the chart group") + f" ({keyed})"
                               + ("; a group is keyed by a container host (a record without machine_id that no "
                                  "queue summary covers): give --queue-summaries or --machine-of if it is another "
                                  "group's machine" if by_host else "")))
    pr = rep.get("speed_probes") or {}
    failed, absent = pr.get("failed") or [], pr.get("absent") or []
    out.append(dict(rule="speed_probes", status="fail" if failed or absent else
                    ("pass" if sp["files"] else "not_checked"),
                    detail=f"{len(failed)} failed probe(s)" + (f" ({', '.join(f['system'] for f in failed)})"
                                                               if failed else "")
                           + f", {len(absent)} offered row(s) without a speed record (S5)"
                           + (f" ({', '.join(absent)})" if absent else "")
                           + f"; {len(pr.get('excluded') or [])} record(s) left out (an old quant recipe, or no / a "
                             f"failed CER sanity), {len(pr.get('queue_not_done') or [])} speed item(s) not done in the "
                             "queue summaries"))
    return out


# ------------------------------------------------------------------------------------------------ chart

CHART_FAMILIES = (("parakeet", "Parakeet family (CTC)"), ("cohere", "Cohere family (AED)"),
                  ("whisper", "Whisper"))


def _chart_family(v: dict) -> str:
    return "whisper" if v["family"] == "whisper" else "parakeet" if v["family"] == "ctc" else "cohere"


def chart_points(rep: dict) -> list[dict]:
    """Every offered row as a chart point: drawn only when it has a speed in the chart group (or the S2 re-time
    from it) and an M4-all. A study row is left to its full-data twin (full-x, full-x@fmt: the same x) only when that
    twin is drawn; otherwise it is drawn itself or listed with its own reason."""
    sysd = rep["systems"]
    pts = []
    for r in rep["offer"]["rows"]:
        s, v = r["system"], sysd[r["system"]]
        sp = v["speed"]
        kind = ("teacher" if v["role"] == "teacher" else "whisper" if v["role"] == "whisper" else
                "variant" if v["role"] == "quant" else "full" if v["role"] == "full" else "study")
        base = v["base"] if kind == "variant" else None
        why = None
        if sp.get("simulated"):
            why = "simulated format: accuracy only, no speed (S4)"
        elif not sp.get("available"):
            why = "no speed record (S5)"
        elif not sp.get("in_chart_group"):
            why = f"timed on another machine/GPU ({sp.get('group')}, S3)"
        elif v["metrics"]["m4_all"] is None:
            why = "no M4-all (a set is missing)"
        twin = None
        if r["block"] == "study":
            full = STUDY_TO_FULL.get(base or s)
            twin = full if kind != "variant" else (f"{full}@{v['format']}" if full else None)
        pts.append(dict(system=s, display=v["display"], family=_chart_family(v), kind=kind, base=base,
                        format=v["format"], rtfx=sp.get("rtfx"), p50_ms=sp.get("p50_ms"),
                        m4_all_pct=None if v["metrics"]["m4_all"] is None else 100 * v["metrics"]["m4_all"],
                        m4_pct=None if v["metrics"]["m4"] is None else 100 * v["metrics"]["m4"],
                        speed_source=sp.get("source"), flags=v["flags"], drawn=why is None, not_drawn=why,
                        _twin=twin))
    # second pass: twins are full-data rows (never in the study block), so hiding a study row never hides another
    drawn = {p["system"] for p in pts if p["drawn"]}
    for p in pts:
        twin = p.pop("_twin")
        if twin in drawn:
            p.update(drawn=False, not_drawn=f"shown as its full-data row {twin}")
    return pts


def _esc(t) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _fmt_tick(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


def log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def lin_ticks(lo: float, hi: float) -> tuple[float, float, list[float]]:
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    raw = (hi - lo) / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    a, b = math.floor(lo / step) * step, math.ceil(hi / step) * step
    n = int(round((b - a) / step))
    return a, b, [round(a + i * step, 10) for i in range(n + 1)]


def _star(cx: float, cy: float, r: float = 7.0, ri: float = 3.0) -> str:
    return " ".join(f"{cx + (r if k % 2 == 0 else ri) * math.cos(-math.pi / 2 + k * math.pi / 5):.1f},"
                    f"{cy + (r if k % 2 == 0 else ri) * math.sin(-math.pi / 2 + k * math.pi / 5):.1f}"
                    for k in range(10))


def _diamond(cx: float, cy: float, r: float = 6.0) -> str:
    return f"{cx:.1f},{cy - r:.1f} {cx + r:.1f},{cy:.1f} {cx:.1f},{cy + r:.1f} {cx - r:.1f},{cy:.1f}"


SVG_STYLE = """<style>
svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--parakeet:#2a78d6;--cohere:#eb6834;--whisper:#1baf7a}
@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--parakeet:#3987e5;--cohere:#d95926;--whisper:#199e70}}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink2);font-size:11px}
.title{fill:var(--ink);font-size:15px;font-weight:600}.sub{fill:var(--muted);font-size:11px}
.tick{font-variant-numeric:tabular-nums}.axl{fill:var(--ink2);font-size:12px}
.lab{fill:var(--ink2);font-size:11px}.lab2{fill:var(--muted);font-size:9.5px}.note{fill:var(--ink2);font-size:11px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.m{stroke:var(--surface);stroke-width:2}.h{fill:var(--surface);stroke-width:2}.j{stroke-width:1;opacity:.55}
.f-parakeet{fill:var(--parakeet)}.f-cohere{fill:var(--cohere)}.f-whisper{fill:var(--whisper)}
.s-parakeet{stroke:var(--parakeet)}.s-cohere{stroke:var(--cohere)}.s-whisper{stroke:var(--whisper)}
.bg{fill:var(--surface)}.key{fill:var(--muted)}.keyh{fill:var(--surface);stroke:var(--muted);stroke-width:2}
</style>"""


def chart_svg(points: list[dict], *, x_key: str, title: str, x_label: str, better: str, gpu: str | None) -> str:
    """One paper-style scatter as a standalone SVG (deterministic bytes): y = M4-all CER %, x = x_key on a log axis;
    families by colour, full-data students as dots, study students without a drawn full-data twin as diamonds,
    quantised variants hollow and joined to their bf16 point, teachers as stars; direct labels where they fit, a
    <title> tooltip on every mark, and the rows not drawn listed under the plot."""
    W, left, right, top, plot_h = 980, 72, 250, 70, 420
    plot_w = W - left - right
    drawn = [p for p in points if p["drawn"] and p.get(x_key) and p["m4_all_pct"] is not None]
    skipped = [p for p in points if p not in drawn]
    notes = [f"{p['display']}: {p['not_drawn'] or 'no ' + x_key}" for p in skipped
             if not (p["not_drawn"] or "").startswith("shown as")]
    H = top + plot_h + 64 + (22 + 16 * len(notes) if notes else 0)
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="{_esc(title)}">', SVG_STYLE, f'<rect class="bg" width="{W}" height="{H}"/>',
         f'<text class="title" x="{left}" y="28">{_esc(title)}</text>',
         f'<text class="sub" x="{left}" y="46">Descriptive (outside PREREG). y: M4-all CER (JSUT, CV8, Reazon, '
         f'Galgame-all). {_esc(better)}. {_esc(gpu or "")}</text>']
    if not drawn:
        L.append(f'<text class="note" x="{left}" y="{top + 20}">No point has a speed in the chart group.</text>')
    else:
        xs = [p[x_key] for p in drawn]
        ys = [p["m4_all_pct"] for p in drawn]
        lx0, lx1 = math.log10(min(xs) / 1.4), math.log10(max(xs) * 1.4)
        xt = log_ticks(10 ** lx0, 10 ** lx1)
        if len(xt) < 2:
            lx0, lx1 = math.floor(lx0), math.ceil(lx1)
            xt = log_ticks(10 ** lx0, 10 ** lx1)
        span = max(ys) - min(ys)
        y0, y1, yt = lin_ticks(max(0.0, min(ys) - 0.08 * span - 0.5), max(ys) + 0.08 * span + 0.5)

        def X(v):
            return left + (math.log10(v) - lx0) / (lx1 - lx0) * plot_w

        def Y(v):
            return top + (y1 - v) / (y1 - y0) * plot_h

        for v in yt:
            L.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
            L.append(f'<text class="tick" x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
        for v in xt:
            L.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top}" y2="{top + plot_h}"/>')
            L.append(f'<text class="tick" x="{X(v):.1f}" y="{top + plot_h + 18}" text-anchor="middle">'
                     f'{_fmt_tick(v)}</text>')
        L.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
        L.append(f'<text class="axl" x="{left + plot_w / 2:.1f}" y="{top + plot_h + 42}" text-anchor="middle">'
                 f'{_esc(x_label)}</text>')
        L.append(f'<text class="axl" transform="translate(18 {top + plot_h / 2:.1f}) rotate(-90)" '
                 f'text-anchor="middle">M4-all CER (%)</text>')
        pos = {p["system"]: (X(p[x_key]), Y(p["m4_all_pct"])) for p in drawn}
        for p in drawn:  # joins first, under every mark
            if p["kind"] == "variant" and p["base"] in pos:
                (x0, y0_), (x1_, y1_) = pos[p["base"]], pos[p["system"]]
                L.append(f'<line class="j s-{p["family"]}" x1="{x0:.1f}" y1="{y0_:.1f}" x2="{x1_:.1f}" '
                         f'y2="{y1_:.1f}"/>')
        # labels avoid every mark, every label placed before them and the legend (drawn below at lx, top + 6: its
        # headers and 7 rows end by top + 174); a label may run past the plot's right edge only beside the legend
        lx = W - right + 24
        boxes: list[tuple[float, float, float, float]] = [(x - 7, y - 7, 14, 14) for x, y in pos.values()]
        boxes.append((lx - 4, top - 8, W - lx + 4, 182))
        order = sorted(drawn, key=lambda p: (p["kind"] == "variant", -p["m4_all_pct"], p["system"]))
        for p in sorted(drawn, key=lambda p: (p["kind"] != "variant", p["system"])):  # variants under the rest
            x, y = pos[p["system"]]
            fam = p["family"]
            tip = (f"{p['display']}: M4-all {p['m4_all_pct']:.2f} %, "
                   + (f"{p[x_key]:,.0f}x real time" if x_key == "rtfx" else f"p50 {p[x_key]:.1f} ms")
                   + (f" (speed of {p['speed_source']})" if p["speed_source"] and p["speed_source"] != p["system"]
                      else "") + (f"; flags {' '.join(p['flags'])}" if p["flags"] else ""))
            if p["kind"] == "teacher":
                mark = f'<polygon class="m f-{fam}" points="{_star(x, y)}"/>'
            elif p["kind"] == "variant":
                mark = f'<circle class="h s-{fam}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>'
            elif p["kind"] == "study":
                mark = f'<polygon class="m f-{fam}" points="{_diamond(x, y)}"/>'
            else:
                mark = f'<circle class="m f-{fam}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'
            L.append(f'<g><title>{_esc(tip)}</title>{mark}</g>')
        for p in order:  # direct labels: bf16 rows first, a variant's format only where it fits
            x, y = pos[p["system"]]
            # a variant joined to its bf16 point is named by its format; one without it (a study variant whose bf16
            # row is shown as its full-data twin) by its whole name, or it would read as the twin's variant
            small = p["kind"] == "variant" and p["base"] in pos
            text = FORMAT_LABEL.get(p["format"], p["format"]) if small else p["display"]
            size = 9.5 if small else 11.0
            w, h = 0.58 * size * len(text), size + 2
            for dx, dy in ((9, 4), (9, -7), (9, 15), (-9 - w, 4), (-9 - w, -7), (-9 - w, 15), (9, -18), (9, 26)):
                bx, by = x + dx, y + dy - size
                if bx < left or bx + w > W - 8 or by < top - 4 or by + h > top + plot_h + 4:
                    continue
                if any(bx < a + c and a < bx + w and by < b + d and b < by + h for a, b, c, d in boxes):
                    continue
                boxes.append((bx, by, w, h))
                L.append(f'<text class="{"lab2" if small else "lab"}" x="{bx:.1f}" y="{y + dy:.1f}">{_esc(text)}'
                         f'</text>')
                break
        ly = top + 6  # the legend (its box is in boxes above)
        L.append(f'<text class="axl" x="{lx}" y="{ly}">Family</text>')
        for i, (fam, name) in enumerate(CHART_FAMILIES):
            yy = ly + 20 + 18 * i
            L.append(f'<circle class="m f-{fam}" cx="{lx + 6}" cy="{yy - 4}" r="5"/>')
            L.append(f'<text class="lab" x="{lx + 18}" y="{yy}">{_esc(name)}</text>')
        ly += 20 + 18 * len(CHART_FAMILIES) + 14
        L.append(f'<text class="axl" x="{lx}" y="{ly}">Mark</text>')
        keys = (("dot", "full-data student, bf16"), ("diamond", "study student (no full-data point)"),
                ("hollow", "quantised variant, joined to bf16"), ("star", "teacher"))
        for i, (k, name) in enumerate(keys):
            yy = ly + 20 + 18 * i
            cx, cy = lx + 6, yy - 4
            if k == "dot":
                L.append(f'<circle class="key" cx="{cx}" cy="{cy}" r="5"/>')
            elif k == "diamond":
                L.append(f'<polygon class="key" points="{_diamond(cx, cy)}"/>')
            elif k == "hollow":
                L.append(f'<circle class="keyh" cx="{cx}" cy="{cy}" r="4.5"/>')
            else:
                L.append(f'<polygon class="key" points="{_star(cx, cy)}"/>')
            L.append(f'<text class="lab" x="{lx + 18}" y="{yy}">{_esc(name)}</text>')
    if notes:
        y = top + plot_h + 72
        L.append(f'<text class="axl" x="{left}" y="{y}">Not drawn</text>')
        for i, n in enumerate(notes):
            L.append(f'<text class="note" x="{left}" y="{y + 18 + 16 * i}">{_esc(n)}</text>')
    L.append("</svg>")
    return "\n".join(L) + "\n"


def points_csv(points: list[dict]) -> str:
    cols = ["system", "display", "family", "kind", "base", "format", "rtfx", "p50_ms", "m4_all_pct", "m4_pct",
            "speed_source", "drawn", "not_drawn", "flags"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for p in points:
        w.writerow(["" if p.get(c) is None else " ".join(p[c]) if c == "flags" else p[c] for c in cols])
    return buf.getvalue()


def offer_csv(offer: dict) -> str:
    cols = ["system", "block", "display", "role", "format", "params", "file_bytes", "file_bytes_from",
            "smaller_than_teacher", "m4", "m4_all", "m3", "m4_nostyle", "jsut", "cv8", "reazon", "gal_neutral",
            "gal_all", "vs_teacher_ratio", "vs_teacher_lo", "vs_teacher_hi", "vs_study_ratio", "vs_study_lo",
            "vs_study_hi", "rtfx", "p50_ms", "vram_gb", "vram_1_gb", "speed_text", "epochs", "stopped_early", "flags"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in offer["rows"]:
        flat = dict(r, flags=" ".join(r["flags"]))
        for k in ("vs_teacher", "vs_study"):
            c = r.get(k)
            flat[f"{k}_ratio"], flat[f"{k}_lo"], flat[f"{k}_hi"] = (c["ratio"], *c["ci"]) if c else (None,) * 3
        w.writerow(["" if flat.get(c) is None else flat[c] for c in cols])
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ report.md


def _stacks_text(stacks) -> str:
    return "yes" if stacks else "NO" if stacks is False else "not checked (PREREG not compared)"


def _stop_text(run: dict) -> str:
    """The trainer summary's early stop in a cell: stopped_early is the trigger's early_stop event fields (a dict) when
    it shortened the run, else null."""
    if not run:
        return "n/a"
    t = run.get("stopped_early")
    if not t:
        return "no"
    if isinstance(t, dict):
        what = t.get("reason") or t.get("metric")
        return "yes" + (f" ({what}" + (f", step {t['at_step']}" if t.get("at_step") is not None else "") + ")"
                        if what else "")
    return "yes"


def _pp(x) -> str:
    return "n/a" if x is None else f"{100 * x:+.2f}"


def speed_probes_md(pr: dict) -> list[str]:
    """report.md's "Speed probes not measured or left out" (F4): every failed, left-out or absent probe and every
    speed item the queue did not finish, one table row each; one line when there is none."""
    def why_failed(f):
        rc = f.get("rc")
        return f["why"] + (f" (exit {rc})" if rc not in (None, "") and f"exit {rc}" not in f["why"] else "")

    rows = [[f["system"], "failed" + (f" ({f['stage']})" if f.get("stage") else ""), why_failed(f), f["file"]]
            for f in pr.get("failed") or []]
    rows += [[x["system"], "left out", x["why"], x["file"]] for x in pr.get("excluded") or []]
    rows += [[s, "no speed record (S5)", "no usable speed record of it or of its study weights", "n/a"]
             for s in pr.get("absent") or []]
    rows += [[x.get("system") or x["item"], f"queue: {x.get('status')}", x.get("why") or "n/a", x["path"]]
             for x in pr.get("queue_not_done") or []]
    L = ["## Speed probes not measured or left out", ""]
    if not rows:
        return L + ["Every speed probe given was measured and used.", ""]
    L += ["A probe that failed (a crash, or exit 4: its CER was not sane against the same weights' bf16 decode), a "
          "record left out (timed by quant code older than F1, or without a passing CER sanity), an offered row with "
          "no speed record at all, and a speed item the queue did not finish.", ""]
    L += sr.table(["system", "what", "why", "file"], rows)
    if pr.get("failures_resolved"):
        L += ["", "Failures a later good probe of the same system superseded: "
              + "; ".join(f"`{f['system']}` ({f['why']})" for f in pr["failures_resolved"]) + "."]
    return L + [""]


def render_md(rep: dict) -> str:
    sysd, comp = rep["systems"], rep["comparisons"]
    m = rep["manifest"]
    sg = rep["speed_groups"]
    L = ["# Full-data runs: results report", "",
         f"**{rep['descriptive']}** (flag D)", "",
         f"Written {rep['inputs']['written_utc']}: {len(sysd)} systems on the manifest `{m['path']}` (sha256 "
         f"{m['sha256'][:12]}; stacks on the study: **{_stacks_text(m['stacks_on_study'])}**; PREREG check: "
         f"{m['prereg_check']['status']}). CER in %; ratios with 95 % CIs (ln r +- 1.96 sqrt(v_boot + k sigma_run^2), "
         f"sigma_run {100 * rep['settings']['sigma_run']['sigma_run']:.1f} %, B = {rep['settings']['boot_b']:,}).",
         ""]
    if sg["primary"]:
        others = [g for g in sg["groups"] if g != sg["primary"]]
        L += [f"Speed columns and charts: the group `{sg['primary']}` ({len(sg['groups'][sg['primary']]['systems'])} "
              f"systems)." + (f" Other groups (their rows carry S3): {', '.join(f'`{g}`' for g in others)}."
                              if others else ""), ""]
    else:
        L += ["No speed file: the speed columns are n/a.", ""]
    if rep["missing_systems"]:
        L += [f"Systems without results: {', '.join(rep['missing_systems'])}.", ""]

    L += ["## Offer table", "",
          "M4 = JSUT, CV8, Reazon, Galgame-neutral (the study's metric); M4-all = Galgame-all in place of "
          "Galgame-neutral; 3-set mean = JSUT, CV8, Reazon. x smaller = the own teacher's file bytes / the row's. "
          "RTFx = audio seconds per wall second, batched; p50 = batch-1 median latency. Flags: see the legend.", ""]
    for block, text in BLOCKS:
        rows = [r for r in rep["offer"]["rows"] if r["block"] == block]
        if rows:
            L += [f"### {text}", ""] + sr.table(OFFER_COLUMNS, [offer_cells(r) for r in rows]) + [""]

    vs = comp["vs_study"]
    if vs:
        L += ["## Study -> full", "",
              "The same shape and recipe on the full data (full / study; k = 2: two trainings).", ""]
        rows = []
        for s, c in vs.items():
            run = sysd[s]["run"] or {}
            stu = FULL_RUNS[s][0]
            rows.append([sysd[s]["display"], pct(sysd[stu]["metrics"]["m4"]), pct(sysd[s]["metrics"]["m4"]),
                         _mci(c["m4"], 3), (c["m4"] or {}).get("verdict", "n/a"), _mci(c["m4_all"], 3),
                         run.get("steps", "n/a"),
                         "n/a" if run.get("epochs") is None else f"{run['epochs']:.2f}",
                         _stop_text(run), 
                         run.get("end_reason") or "n/a", run.get("resume_resets") or 0])
        L += sr.table(["model", "study M4", "full M4", "full / study M4 [CI]", "verdict", "M4-all ratio [CI]",
                       "steps", "epochs", "early stop (trigger)", "end reason", "resets"], rows) + [""]

    vb = comp["vs_bf16"]
    if vb:
        L += ["## Quantisation", "",
              "Each variant against its own bf16 weights (k = 0: the same run, only the bootstrap). KL to the own "
              "teacher, token-weighted over JSUT, CV8, Reazon (AED per token, CTC per target token).", ""]
        rows = []
        for s, c in vb.items():
            v, b = sysd[s], sysd[sysd[s]["base"]]
            kl = v["kl"]
            q = v["quant"] or {}
            sp, bsp = v["speed"], b["speed"]
            rows.append([v["display"], b["display"], pct(b["metrics"]["m4"]), pct(v["metrics"]["m4"]),
                         _pp((c["m4"] or {}).get("delta")), _mci(c["m4"], 3), _pp((c["m4_all"] or {}).get("delta")),
                         "n/a" if not kl else f"{kl['gate']:.4f}",
                         "n/a" if c.get("kl_delta") is None else f"{c['kl_delta']:+.4f}",
                         "n/a" if v["file_bytes"] is None else f"{v['file_bytes'] / 1e6:,.0f}",
                         "n/a" if not (v["file_bytes"] and b["file_bytes"]) else
                         f"{b['file_bytes'] / v['file_bytes']:.2f}",
                         sp.get("text") or ("n/a" if sp.get("rtfx") is None else f"{sp['rtfx']:,.0f}"),
                         # one machine only; an S2 variant and its base share the re-time ratio (resolve_speed)
                         "n/a" if not (sp.get("rtfx") and bsp.get("rtfx") and sp.get("in_chart_group")
                                       and bsp.get("in_chart_group")) else f"{sp['rtfx'] / bsp['rtfx']:.2f}",
                         ((q.get("nonfinite") or {}).get("rows", "n/a")), " ".join(v["flags"])])
        L += sr.table(["variant", "bf16", "bf16 M4", "M4", "delta M4 pp", "ratio vs bf16 [CI]", "delta M4-all pp",
                       "KL", "delta KL", "file MB", "x smaller than bf16", "RTFx", "x bf16 speed",
                       "non-finite rows", "flags"], rows) + [""]
        comp_rows = [(g, n) for g, e in sg["groups"].items() for n in e["compiled"]]
        pr = rep.get("speed_probes") or {}
        comp_bad = [(x["system"], x["why"]) for x in (pr.get("failed") or []) + (pr.get("excluded") or [])
                    if x["system"].endswith("+compile")]
        if comp_rows or comp_bad:
            L += ["torch.compile records (speed only): " + (", ".join(f"`{n}` ({g})" for g, n in comp_rows) or "none")
                  + "." + (" Not measured or left out: " + "; ".join(f"`{n}` ({w})" for n, w in comp_bad) + "."
                           if comp_bad else ""), ""]

    L += speed_probes_md(rep.get("speed_probes") or {})

    wh = rep["whisper"]
    if wh["systems"]:
        L += ["## Whisper", "",
              "Descriptive yardsticks. The headline is **M4-all**; M4 uses the Kotoba-chosen Galgame-neutral view "
              "(W1). The gap is Galgame-all minus Galgame-neutral in pp (the students and teachers lose about 6.3-6.7 "
              "pp there).", ""]
        rows = []
        for s in wh["systems"]:
            v = sysd[s]
            m_ = v["metrics"]
            st_ = v["sets"]
            w = v["whisper"] or {}
            tot = w.get("totals") or {}
            rows.append([v["display"], pct(m_["m4_all"]), pct(m_["m3"]), pct(m_["m4"]) + " (W1)",
                         pct(m_["m4_all_nostyle"]), pct(m_["m4_nostyle"])]
                        + [pct((st_.get(k) or {}).get("cer")) for k in ("eval_jsut", "eval_cv8", "eval_reazon",
                                                                         "galgame_neutral", "galgame_all",
                                                                         "eval_emilia")]
                        + [_pp(m_[GAP]).lstrip("+") if m_[GAP] is not None else "n/a",
                           tot.get("n_truncated", "n/a"), tot.get("n_repetition", "n/a"), " ".join(v["flags"])])
        L += sr.table(["model", "M4-all", "3-set mean", "M4", "M4-all no-style", "M4 no-style", "JSUT", "CV8",
                       "Reazon", "Gal-neutral", "Gal-all", "Emilia (W6)", "gap pp", "truncated", "repetition stops",
                       "flags"], rows) + [""]
        L += ["Caveats printed next to every Whisper number (plan v3 section 7):", ""]
        L += [f"{i}. {c}" for i, c in enumerate(wh["caveats"], 1)] + [""]
        kb = wh["kotoba_bias"]
        if kb and kb.get("available"):
            L.append(f"- Kotoba's stored judge output (decision 29): CER {pct(kb['cer_neutral'])} % on the "
                     f"{kb['n_neutral']} Galgame-neutral rows vs {pct(kb['cer_all'])} % on all {kb['n_all']} "
                     f"with a reference: the neutral view takes {kb['gap_pp']:.2f} pp off Kotoba's Galgame CER.")
        elif kb:
            L.append(f"- Kotoba's stored judge output: not scored ({kb.get('reason')}).")
        else:
            L.append("- Kotoba's stored judge output (decision 29): not given (--kotoba-jsonl; the download was not "
                     "approved), so the bias is not measured.")
        ko = wh["kotoba_redecode_over_half"]
        if ko:
            L.append(f"- Kotoba's re-decode here: {ko['n']} of the {ko['of']} Galgame-neutral rows score above 0.5 "
                     "CER (the judge kept only rows at or below 0.5; a re-decode in other batches moves a few).")
        L.append("")

    L += ["## Charts", "", "![Error vs speed](chart_error_vs_speed.svg)", "",
          "![Error vs latency](chart_error_vs_latency.svg)", ""]
    nd = [p for p in rep["chart"]["points"] if not p["drawn"] and not (p["not_drawn"] or "").startswith("shown as")]
    if nd:
        L += ["Not drawn: " + "; ".join(f"{p['display']} ({p['not_drawn']})" for p in nd) + ".", ""]

    L += ["## Hallucinations", "",
          f"Rows with a reference; clips under {SHORT_CLIP_S:g} s from the greedy parquets' durations. Silence = "
          f"Whisper's subtitle phrases ({', '.join(SILENCE_PHRASES)}); bracketed = a （…）, (…), […] or 【…】 span "
          "in the raw output. Kana mode = JSUT rows with >= 3 kanji in the reference and none in a non-empty "
          "output.", ""]
    rows = []
    for s, v in sysd.items():
        h = v["halluc"]
        a, sh = h.get("m4_all") or {}, h.get("m4_all_short") or {}
        er, km = h.get("empty_ref") or {}, h.get("kana_mode") or {}
        m4 = h.get("m4") or {}
        rows.append([v["display"], a.get("n", "n/a"), a.get("n_empty", "n/a"), a.get("n_runaway", "n/a"),
                     a.get("n_truncated", "n/a"), a.get("n_silence", "n/a"), a.get("n_bracketed", "n/a"),
                     f"{m4.get('n_empty', 'n/a')} / {m4.get('n_runaway', 'n/a')}",
                     "n/a" if not sh else f"{sh['n']}: {sh['n_empty']} / {sh['n_runaway']} / {sh['n_silence']}",
                     "n/a" if not er else f"{er['nonempty_on_empty_ref']} of {er['n']}",
                     "n/a" if km.get("rate") is None else f"{km['n']} of {km['eligible']} ({100 * km['rate']:.2f} %)"])
    L += sr.table(["system", "M4-all rows", "empty", "runaway", "truncated", "silence", "bracketed",
                   "M4 empty / runaway", "short clips: empty / runaway / silence", "output on empty reference",
                   "JSUT kana mode"], rows) + [""]

    L += ["## Per-system CER per set (raw / no-style, %)", ""]
    cols = [("eval_jsut", "JSUT"), ("eval_cv8", "CV8"), ("eval_reazon", "Reazon"), ("galgame_neutral", "Gal-neutral"),
            ("galgame_all", "Gal-all"), ("eval_emilia", "Emilia")]
    rows = []
    for s, v in sysd.items():
        rows.append([v["display"], v["role"]]
                    + [f"{pct((v['sets'].get(k) or {}).get('cer'))} / "
                       f"{pct((v['sets'].get(k) or {}).get('cer_nostyle'))}" for k, _ in cols]
                    + [pct(v["metrics"]["m4"]), pct(v["metrics"]["m4_all"]), pct(v["metrics"]["gate_pooled"])])
    L += sr.table(["system", "role"] + [c for _, c in cols] + ["M4", "M4-all", "gate-pooled"], rows) + [""]

    L += ["## Flags", ""] + [f"- **{k}**: {t}" for k, t in rep["flags_legend"].items()] + [""]
    L += ["## Checks", ""]
    L += sr.table(["check", "status", "detail"], [[c["rule"], c["status"], c["detail"]] for c in rep["checks"]])
    inp = rep["inputs"]
    L += ["", "## Inputs", "",
          f"- tables: {', '.join(inp['tables']) or 'none'}; readouts: {', '.join(inp['readouts']) or 'none'} "
          f"({len(inp['readouts_read'])} read, {len(inp['readouts_superseded'])} superseded, "
          f"{len(inp['readouts_ignored'])} ignored)",
          f"- speed: {', '.join(inp['speed']) or 'none'}; queue summaries (the machine of records without "
          f"machine_id): {', '.join(inp['queue_summaries']) or 'none'}",
          f"- run summaries: {', '.join(inp['run_summaries']) or 'none'} ({len(inp['summaries_read'])} read); params: "
          f"{inp['params'] or 'none'}; Kotoba judge file: {inp['kotoba_jsonl'] or 'none'}"]
    if inp["readouts_refused_sets"]:
        L.append("- greedy sets refused (their ids are not the manifest's; that system lacks the set, as 05 refuses "
                 "it): " + "; ".join(f"{r['system']}/{r['set']} ({r['why']})" for r in inp["readouts_refused_sets"]))
    if rep["left_out"]["half"]:
        L.append(f"- left out (the T/2 branches; --include-half keeps them): {', '.join(rep['left_out']['half'])}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main


def write_outputs(rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pts = rep["chart"]["points"]
    prim = rep["speed_groups"]["primary"]
    gpu = (rep["speed_groups"]["groups"].get(prim) or {}).get("gpu") if prim else None
    sr.write_atomic(out / "report.json", json.dumps(rep, indent=1, ensure_ascii=False))
    sr.write_atomic(out / "report.md", render_md(rep))
    sr.write_atomic(out / "offer.csv", offer_csv(rep["offer"]))
    sr.write_atomic(out / "chart_points.json", json.dumps(dict(group=prim, gpu=gpu, points=pts), indent=1,
                                                          ensure_ascii=False))
    sr.write_atomic(out / "chart_points.csv", points_csv(pts))
    where = f"Timed on {gpu}." if gpu else ""
    sr.write_atomic(out / "chart_error_vs_speed.svg", chart_svg(
        pts, x_key="rtfx", title="Error vs speed: M4-all CER against batched speed", better="Better: down and right",
        x_label=f"Batched speed, x real time (RTFx, log scale){', ' + gpu if gpu else ''}", gpu=where))
    sr.write_atomic(out / "chart_error_vs_latency.svg", chart_svg(
        pts, x_key="p50_ms", title="Error vs latency: M4-all CER against batch-1 latency",
        better="Better: down and left", x_label=f"Batch-1 median latency, ms (log scale){', ' + gpu if gpu else ''}",
        gpu=where))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True, help="the frozen study_manifest.json")
    ap.add_argument("--prereg", default=str(ROOT / "study" / "PREREG.json"),
                    help="the PREREG.json whose manifest block the manifest must match (default: the repo's; none: "
                         "not compared)")
    ap.add_argument("--tables", type=Path, nargs="+", help="the study's tables dir(s): <dir>/<system>/<set>.parquet")
    ap.add_argument("--readouts", type=Path, nargs="+", help="dirs searched for study.json (05 / whisper_eval --out)")
    ap.add_argument("--speed", type=Path, nargs="+", help="speed_probe outputs (runs/speed-*/speed.json)")
    ap.add_argument("--queue-summaries", type=Path, nargs="+",
                    help="full/box-<box>/queue_summary.json files or dirs holding them: the machine of speed records "
                         "without versions.machine_id (the one beside each speed file's runs/ is read anyway)")
    ap.add_argument("--machine-of", nargs="+", metavar="NAME=ID",
                    help="the machine of the records in speed dir NAME (speed-<box>-<stamp>) or of container host "
                         "NAME, where no queue summary says it")
    ap.add_argument("--run-summaries", type=Path, nargs="+", help="the trainer's runs root(s)")
    ap.add_argument("--params", type=Path, help="{system: params_total} (the study's params.json)")
    ap.add_argument("--kotoba-jsonl", type=Path, help="Kotoba's stored Galgame judge file (decision 29; optional)")
    ap.add_argument("--include-half", action="store_true", help="keep the study's T/2 branches")
    ap.add_argument("--boot-b", type=int, default=ss.BOOT_B)
    ap.add_argument("--seed", type=int, default=ss.BOOT_SEED)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if not (args.tables or args.readouts):
        ap.error("give --tables and / or --readouts")
    if args.boot_b < 100:
        ap.error("--boot-b must be >= 100")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        rep = build_report(args)
    except (ss.ManifestError, InputError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    write_outputs(rep, args.out)
    n = sum(p["drawn"] for p in rep["chart"]["points"])
    print(f"{len(rep['systems'])} systems, {len(rep['offer']['rows'])} offer rows, {n} chart points; "
          f"wrote {args.out / 'report.md'} and report.json, offer.csv, the charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
