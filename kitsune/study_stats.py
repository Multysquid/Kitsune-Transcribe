"""Statistics of the size study (study/STUDY.md 4.3-4.7): the pre-registered limit per family and every readout
reported with it, from the per-utterance results of every system on the frozen eval manifest.

Input (CONTRACT.md 5): per system, a per-utterance table with columns id, set, edits, ref_len (character edit distance
and reference length on kitsune.text.normalize_ja strings, the kitsune.evaluate normalisation) and optionally
edits_nostyle, sub, del, ins, hyp, ref, truncated. score_utterances() makes those columns from ref / hyp text when a
table has only the text (a 05_evaluate greedy_<set>.parquet). build_corpus() aligns every table to the manifest's
ordered ids and REFUSES (ManifestError) anything that does not match it: a missing, extra or duplicated id, an unknown
set, a manifest whose ids do not hash to its own sha256, a reference length that differs between two systems. Every
number below is then over exactly the same utterances for every system.

Strata. Every eval set is one stratum; the Galgame set is split into its manifest views instead (galgame_neutral, the
primary view; galgame_all; galgame_label_box). Rows with an empty reference are left out of every sum and counted, as
kitsune.evaluate.corpus_cer does (Galgame has 31 of them).

Metrics (4.3). Corpus CER of a stratum = sum edits / sum ref_len. A group metric is the MACRO mean of its strata's
corpus CERs, equal weights:
  m4           JSUT, CV8, ReazonSpeech-test, Galgame-neutral: the within-family primary
  ood / ind    the qualifier pairs (JSUT, CV8) and (Reazon, Galgame-neutral), reported with every call
  jg           JSUT + Galgame-neutral, the cross-family metric (descriptive only: no threshold is ever applied);
               jg_nostyle the same on edits_nostyle
  gate_pooled  sum edits / sum ref_len over JSUT + CV8 + Reazon pooled: the first run's val_cer (12.15 %) and the
               anchor regression flag's measure (4.1)

Noise (4.4). v_boot comes from a paired, stratified utterance bootstrap: B replicates (10,000), a fixed seed, each
stratum resampled with replacement within itself. A replicate is a vector of multiplicities per utterance, drawn once
per stratum and applied to every system's column, so every system (and every metric) sees the SAME resampled
utterances; the draws depend only on (seed, stratum name, stratum size), not on which systems are in the tables.
Multiplicities and edit counts are integers, so the resampled sums are exact in float64 whatever the BLAS summation
order: the same seed gives bit-identical numbers on any machine. Run-to-run noise sigma_run is the SD of ln M4 of one
system: max(prior 1.6 %, 0.886 |ln(M4_replicate / M4_T-0.1B)|) (0.886 = sqrt(pi) / 2 makes one pair's |difference|
unbiased for sigma). The CI of a ratio r = a / b is ln r +- 1.96 sqrt(v_boot + k sigma_run^2), with k the number of
TRAINED systems among a and b (2 student-student, 1 student-teacher; the teachers are fixed models).

The limit (4.5). Per family, every smaller student s gets r_s = M4_s / M4_top (top: T-0.6B, P-0.3B; the scratch ladder
bridge -> T-0.1B -> T-0.05B is a second walk with top = bridge). Tolerance delta: WITHIN if the CI's upper end
<= ln(1 + delta), OUTSIDE if its lower end > ln(1 + delta), else UNRESOLVED. The walk goes down the sizes: the first
OUTSIDE ends it ("limit reached between X and Y"), an UNRESOLVED ends it ("the limit is at or below X; Y unresolved")
and the sizes below are reported descriptively; X is the smallest size that is WITHIN with every larger size WITHIN
(the top when there is none). delta 10 % is the primary (decision 1); 5 and 20 % are always reported.

Readouts (4.6, 4.7): g per halving for every ladder step, g = r^(1/h) - 1 with h = log2 of the total-parameter ratio
(STUDY.md 1.3), CI mapped from ln r's; delta-g on the scratch ladder only; the init effect bridge / T-0.3B; the
distillation gaps; the T/2 vs T budget readout; the practical bars; a Pareto set over CER, A100 RTF and VRAM; the
anchor regression flag; and the section-7 invalidation rules as far as the inputs show them (checks()).

Everything here is numpy on CPU: the analysis of 22 systems on the full manifest at B = 10,000 takes about 5 s (plus
a second per system for the imitation CER); importing kitsune.evaluate (torch) is the slowest part of a report.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from kitsune.evaluate import BASELINE_TOL, GATE_SETS, TEACHER_CER_PREREG
from kitsune.store import ids_sha256

# ------------------------------------------------------------------------------------------------ the pre-registration

DELTA_PRIMARY = 0.10  # decision 1
DELTAS = (0.05, 0.10, 0.20)  # all three always reported
SIGMA_RUN_PRIOR = 0.016  # the first run's residual SD of the 4-set macro around its trend
SIGMA_HAT_FACTOR = 0.886  # sqrt(pi) / 2: E|X - Y| = 2 sigma / sqrt(pi) for two independent runs
BOOT_B = 10_000
BOOT_SEED = 1234
Z = 1.96
TEACHER_BARS = (1.2, 1.5)  # the practical bars against the own teacher (4.7)
ANCHOR_FLAG_REL = 0.05  # 4.1: T-0.6B worse than the anchor by more than 5 % relative, paired CI excluding 0
DATA_WAIT_MAX = 0.05  # 6.2: calibration must not be loader-bound

DEFAULT_SETTINGS = dict(delta_primary=DELTA_PRIMARY, deltas=list(DELTAS), sigma_run_prior=SIGMA_RUN_PRIOR,
                        sigma_hat_factor=SIGMA_HAT_FACTOR, boot_b=BOOT_B, boot_seed=BOOT_SEED, z=Z,
                        teacher_bars=list(TEACHER_BARS), anchor_flag_rel=ANCHOR_FLAG_REL,
                        replicate=["study-t01-s1235", "study-t01"])

# ------------------------------------------------------------------------------------------------ sets and metrics

EVAL_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame")
GALGAME = "galgame"
GALGAME_VIEWS = ("neutral", "all", "label_box")
NEUTRAL = "galgame_neutral"
M4_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", NEUTRAL)
PAIRS = {"ood": ("eval_jsut", "eval_cv8"), "ind": ("eval_reazon", NEUTRAL)}
CROSS_SETS = ("eval_jsut", NEUTRAL)

# name -> (aggregation, strata, on no-style edits)
METRICS = {
    "m4": ("macro", M4_SETS, False),
    "ood": ("macro", PAIRS["ood"], False),
    "ind": ("macro", PAIRS["ind"], False),
    "jg": ("macro", CROSS_SETS, False),
    "jg_nostyle": ("macro", CROSS_SETS, True),
    "m4_nostyle": ("macro", M4_SETS, True),
    "gate_pooled": ("pooled", tuple(GATE_SETS), False),
}
METRIC_TEXT = {"m4": "M4 (macro JSUT, CV8, Reazon, Galgame-neutral)", "ood": "out-of-domain pair (JSUT, CV8)",
               "ind": "in-domain pair (Reazon, Galgame-neutral)", "jg": "JSUT + Galgame-neutral (raw)",
               "jg_nostyle": "JSUT + Galgame-neutral (no-style)", "m4_nostyle": "M4 on no-style edits",
               "gate_pooled": "gate-pooled CER (JSUT + CV8 + Reazon, sum/sum)"}

# ------------------------------------------------------------------------------------------------ the systems

STUDENTS = ("study-t06", "study-t03", "study-t01", "study-t005", "study-p03", "study-p01", "study-p005")
CONTROLS = ("study-bridge", "study-t01-s1235")
RUNS = STUDENTS + CONTROLS
TEACHERS = ("cohere", "parakeet-ctc", "parakeet-tdt")  # fixed models: no run-to-run noise
ANCHOR = "anchor-b20"
HALF = "-half"

DISPLAY = {"study-t06": "T-0.6B", "study-t03": "T-0.3B", "study-bridge": "bridge", "study-t01": "T-0.1B",
           "study-t01-s1235": "T-0.1B replicate", "study-t005": "T-0.05B", "study-p03": "P-0.3B",
           "study-p01": "P-0.1B", "study-p005": "P-0.05B", "cohere": "Cohere Transcribe",
           "parakeet-ctc": "Parakeet CTC", "parakeet-tdt": "Parakeet TDT", ANCHOR: "anchor (first run's 0.6B)"}

# exact total / non-embedding counts, STUDY.md 1.1 (meta-device builds; every builder asserts total == closed form).
# The anchor is the first run's B20x2560-d4 = the T-0.6B shape; parakeet-ctc is the unpruned CTC path (24 x 4096);
# cohere is the teacher itself (tests/test_student.py pins it), the far end of the Transcribe distillation gap.
PARAMS_TOTAL = {"study-t06": 616_963_328, "study-t03": 320_752_384, "study-bridge": 320_752_384,
                "study-t01": 103_996_416, "study-t01-s1235": 103_996_416, "study-t005": 51_209_600,
                "study-p03": 308_524_033, "study-p01": 98_468_865, "study-p005": 52_190_209,
                "parakeet-ctc": 610_898_945, ANCHOR: 616_963_328, "cohere": 2_065_647_872}
PARAMS_NON_EMBEDDING = {"study-t06": 599_137_536, "study-t03": 302_926_592, "study-bridge": 302_926_592,
                        "study-t01": 95_083_520, "study-t01-s1235": 95_083_520, "study-t005": 44_524_928,
                        "study-p03": 305_374_208, "study-p01": 95_319_040, "study-p005": 49_040_384,
                        "parakeet-ctc": 607_749_120, ANCHOR: 599_137_536}

# the walks (4.5, 4.6): top, then the smaller sizes in order; the own teacher of the family
LADDERS = {
    "transcribe": dict(top="study-t06", sizes=("study-t03", "study-t01", "study-t005"), teacher="cohere",
                       text="Transcribe headline ladder (the owner's recipe at each size)"),
    "parakeet": dict(top="study-p03", sizes=("study-p01", "study-p005"), teacher="parakeet-ctc",
                     text="Parakeet ladder"),
    "scratch": dict(top="study-bridge", sizes=("study-t01", "study-t005"), teacher="cohere",
                    text="scratch ladder (one init regime; top = bridge)"),
}
PRIMARY_FAMILIES = ("transcribe", "parakeet")  # each makes at most one decisive claim; the scratch walk is second
# what changes besides size between the top and each size (STUDY.md 1.3): the label every call carries
CONFOUNDS = {
    ("transcribe", "study-t03"): "nothing (pruned -> pruned)",
    ("transcribe", "study-t01"): "size + init (pruned -> scratch), LR, weight decay, aux CTC, BN mode, width",
    ("transcribe", "study-t005"): "size + init (pruned -> scratch), LR, weight decay, aux CTC, BN mode, width",
    ("parakeet", "study-p01"): "size + loss of function + depth",
    ("parakeet", "study-p005"): "size + loss of function + depth",
    ("scratch", "study-t01"): "width and depth shape only (both scratch)",
    ("scratch", "study-t005"): "width and depth shape only (both scratch)",
}
# every consecutive size step, for g per halving (STUDY.md 1.3); the distillation gaps are not steps
STEPS = (
    ("transcribe", "study-t06", "study-t03", "nothing (pruned -> pruned)"),
    ("transcribe", "study-t03", "study-t01", "init (pruned -> scratch), LR, weight decay, aux CTC, BN mode, width"),
    ("transcribe", "study-t01", "study-t005", "nothing (scratch -> scratch)"),
    ("scratch", "study-bridge", "study-t01", "width and depth shape only (both scratch)"),
    ("parakeet", "study-p03", "study-p01", "size + loss of function + depth"),
    ("parakeet", "study-p01", "study-p005", "nothing"),
)
# the distillation gaps (4.6): (family, student, its own teacher); g over h = 1.743 and 0.986 halvings (STUDY.md 1.3,
# 4.3), reported beside the walk, never inside it
GAPS = (("transcribe", "study-t06", "cohere"), ("parakeet", "study-p03", "parakeet-ctc"))


def base_run(system: str) -> str:
    return system[: -len(HALF)] if system.endswith(HALF) else system


def is_trained(system: str) -> bool:
    """A trained model carries run-to-run noise (sigma_run); a teacher (or any other fixed model) does not."""
    b = base_run(system)
    return b in RUNS or b == ANCHOR or b.startswith("study-")


def family_of(system: str) -> str | None:
    b = base_run(system)
    if b.startswith("study-p"):
        return "ctc"
    if b.startswith("study-") or b == ANCHOR:
        return "aed"
    return {"cohere": "aed", "parakeet-ctc": "ctc", "parakeet-tdt": "ctc"}.get(b)


def teacher_of(system: str) -> str | None:
    """The own teacher of a trained system (Cohere for Transcribe students, Parakeet CTC for P students)."""
    if not is_trained(system):
        return None
    return {"aed": "cohere", "ctc": "parakeet-ctc"}.get(family_of(system) or "")


def display(system: str) -> str:
    b = base_run(system)
    name = DISPLAY.get(b, b)
    return f"{name} T/2" if system.endswith(HALF) else name


def role_of(system: str) -> str:
    b = base_run(system)
    if system.endswith(HALF):
        return "half"
    if b in STUDENTS:
        return "student"
    if b in CONTROLS:
        return "control"
    if b in TEACHERS:
        return "teacher"
    if b == ANCHOR:
        return "anchor"
    return "other"


def system_order(names: Iterable[str]) -> list[str]:
    """Report order: teachers, the anchor, the students down each ladder, the controls, then each run's T/2 branch
    in the same order, then anything else by name."""
    known = [*TEACHERS, ANCHOR, *STUDENTS, *CONTROLS]
    rank = {s: i for i, s in enumerate(known)}
    rank.update({s + HALF: len(known) + i for i, s in enumerate(known)})
    return sorted(names, key=lambda s: (rank.get(s, 2 * len(known)), s))


def params_of(system: str, params: Mapping[str, int] | None = None) -> int | None:
    p = {**PARAMS_TOTAL, **(params or {})}
    return p.get(system, p.get(base_run(system)))


# ------------------------------------------------------------------------------------------------ utterance scoring


def _text(x) -> str:
    """A table's text cell as a string: None or a NaN (a missing value read back from parquet) is the empty string,
    never "nan"."""
    return x if isinstance(x, str) else ""


def score_utterances(refs: Sequence[str], hyps: Sequence[str]) -> pd.DataFrame:
    """Per-utterance edits (S + D + I), ref_len, sub, del, ins and hyp_len on normalize_ja strings, from ONE jiwer
    alignment of all rows, exactly as kitsune.evaluate.corpus_cer counts them (its corpus edits = the sum of these).
    An empty reference scores ref_len 0 and the hypothesis' characters as insertions; corpus sums leave such rows
    out."""
    from kitsune.text import normalize_ja
    import jiwer

    R = [normalize_ja(_text(r)) for r in refs]
    H = [normalize_ja(_text(h)) for h in hyps]
    n = len(R)
    sub, dele, ins = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    keep = [i for i in range(n) if R[i]]
    if keep:
        out = jiwer.process_characters([R[i] for i in keep], [H[i] for i in keep])
        for i, chunks in zip(keep, out.alignments):
            for c in chunks:
                if c.type == "substitute":
                    sub[i] += c.ref_end_idx - c.ref_start_idx
                elif c.type == "delete":
                    dele[i] += c.ref_end_idx - c.ref_start_idx
                elif c.type == "insert":
                    ins[i] += c.hyp_end_idx - c.hyp_start_idx
    for i in range(n):
        if not R[i]:
            ins[i] = len(H[i])
    return pd.DataFrame({"edits": sub + dele + ins, "ref_len": np.array([len(r) for r in R], np.int64),
                         "sub": sub, "del": dele, "ins": ins, "hyp_len": np.array([len(h) for h in H], np.int64)})


def ensure_scored(df: pd.DataFrame) -> pd.DataFrame:
    """A table with edits / ref_len as given; one with only ref / hyp text gets them from score_utterances (and
    edits_nostyle from hyp_nostyle when that column is there); hyp_len is added when hyp is."""
    df = df.reset_index(drop=True)
    have = {"edits", "ref_len"} <= set(df.columns)
    if not have:
        if not {"ref", "hyp"} <= set(df.columns):
            raise ValueError("a per-utterance table needs edits and ref_len, or ref and hyp to score")
        sc = score_utterances(df["ref"].tolist(), df["hyp"].tolist())
        df = df.assign(**{c: sc[c].to_numpy() for c in ("edits", "ref_len", "sub", "del", "ins", "hyp_len")})
    elif "hyp" in df.columns and "hyp_len" not in df.columns:
        from kitsune.text import normalize_ja

        df = df.assign(hyp_len=[len(normalize_ja(_text(h))) for h in df["hyp"].tolist()])
    if "edits_nostyle" not in df.columns and "hyp_nostyle" in df.columns and "ref" in df.columns:
        df = df.assign(edits_nostyle=score_utterances(df["ref"].tolist(), df["hyp_nostyle"].tolist())["edits"])
    return df


# ------------------------------------------------------------------------------------------------ manifest + corpus


class ManifestError(ValueError):
    """A table or the manifest itself does not match the frozen eval manifest: the report refuses to run."""


@dataclass
class Manifest:
    sets: dict[str, list[str]]  # eval set -> ordered ids
    sha256: dict[str, str]
    views: dict[str, list[str]] = field(default_factory=dict)  # Galgame view -> ids


def _ids_and_sha(v) -> tuple[list[str], str | None]:
    if isinstance(v, dict):
        return list(v.get("ids") or []), v.get("ids_sha256") or v.get("sha256")
    return list(v), None


def parse_manifest(obj: Mapping) -> Manifest:
    """study_manifest.json (CONTRACT.md 4) -> Manifest, verified. Read shape (the contract fixes the content, not the
    layout; these are the forms accepted):
        {"sets": {"eval_jsut": {"ids": [...], "ids_sha256": "..."}, ...},
         "galgame_views": {"neutral": [...] | {"ids": [...], "ids_sha256": ...}, "all": ..., "label_box": ...}}
    The sets may also sit under "eval_sets" or at the top level; "sha256" is accepted for "ids_sha256"; the views
    may also be under "views" (or "views": {"galgame": {...}}) or inside the galgame set's entry. Every set needs its
    sha256 (kitsune.store.ids_sha256 of the ordered ids) and must hash to it; a view with a sha256 must hash to it; a
    view must be a subset of the galgame ids."""
    body = obj.get("sets") or obj.get("eval_sets") or obj
    views_obj = obj.get("galgame_views") or obj.get("views") or {}
    if isinstance(views_obj.get(GALGAME), Mapping) and "ids" not in views_obj[GALGAME]:
        views_obj = views_obj[GALGAME]
    sets, sha = {}, {}
    for name, v in body.items():
        if name in ("galgame_views", "views", "eval_sets") or not isinstance(v, (dict, list)):
            continue
        if isinstance(v, dict) and "ids" not in v:
            continue  # metadata block, not a set
        ids, h = _ids_and_sha(v)
        if isinstance(v, dict) and not views_obj and isinstance(v.get("views"), dict):
            views_obj = v["views"]
        if h is None:
            raise ManifestError(f"manifest set {name}: no ids_sha256 to verify its ids against")
        if ids_sha256(ids) != h:
            raise ManifestError(f"manifest set {name}: its {len(ids)} ids hash to {ids_sha256(ids)[:12]}, "
                                f"not to the recorded {str(h)[:12]}")
        if len(set(ids)) != len(ids):
            raise ManifestError(f"manifest set {name}: duplicated ids")
        sets[name], sha[name] = ids, h
    if not sets:
        raise ManifestError("the manifest holds no eval set")
    views = {}
    for vname, v in views_obj.items():
        ids, h = _ids_and_sha(v)
        if h is not None and ids_sha256(ids) != h:
            raise ManifestError(f"manifest Galgame view {vname}: its ids do not hash to the recorded sha256")
        if GALGAME not in sets:
            raise ManifestError(f"manifest has a Galgame view {vname} but no {GALGAME} set")
        extra = set(ids) - set(sets[GALGAME])
        if extra:
            raise ManifestError(f"manifest Galgame view {vname}: {len(extra)} ids are not in the {GALGAME} set, "
                                f"e.g. {sorted(extra)[:3]}")
        views[vname] = ids
    return Manifest(sets=sets, sha256=sha, views=views)


@dataclass
class Stratum:
    name: str
    ids: list[str]  # the rows that count (non-empty reference), manifest order
    ref_len: np.ndarray  # (n,)
    edits: np.ndarray  # (n, S); a NaN column: that system has no table for this set
    nostyle: np.ndarray  # (n, S); NaN where a system has no no-style edits
    n_empty_ref: int


@dataclass
class Corpus:
    systems: list[str]
    strata: dict[str, Stratum]
    desc: dict[str, dict[str, dict]]  # system -> stratum -> descriptive counts (4.2)
    hyps: dict[str, dict[str, list[str]]]  # system -> stratum -> hyp text (for the imitation CER), when given
    manifest: Manifest

    def index(self, system: str) -> int:
        return self.systems.index(system)


def _strata_ids(man: Manifest) -> dict[str, tuple[str, list[str]]]:
    """stratum -> (eval set, ids): every set is one stratum, except Galgame, whose views are (galgame_all = every
    Galgame row when the manifest has no "all" view)."""
    out = {}
    for s, ids in man.sets.items():
        if s != GALGAME:
            out[s] = (s, ids)
    if GALGAME in man.sets:
        views = dict(man.views)
        views.setdefault("all", man.sets[GALGAME])
        for v in (*GALGAME_VIEWS, *sorted(set(views) - set(GALGAME_VIEWS))):
            if v in views:
                out[f"galgame_{v}"] = (GALGAME, views[v])
    return out


def build_corpus(tables: Mapping[str, pd.DataFrame], manifest: Manifest | Mapping) -> Corpus:
    """Align every system's per-utterance table to the manifest (the order of its ids), refusing any mismatch.

    tables: system -> DataFrame with id, set, edits, ref_len (+ optional edits_nostyle, sub, del, ins, hyp,
    hyp_len, truncated; ensure_scored() adds the counts from text). A system may lack a whole set (its numbers there
    are missing); a set it has must hold exactly the manifest's ids of that set, once each."""
    man = manifest if isinstance(manifest, Manifest) else parse_manifest(manifest)
    systems = sorted(tables)
    if not systems:
        raise ManifestError("no system table to analyse")
    per = {}
    for sysname in systems:
        df = ensure_scored(tables[sysname])
        if "set" not in df.columns:
            raise ManifestError(f"{sysname}: the table has no set column")
        per[sysname] = {}
        for s, g in df.groupby("set", sort=True):
            if s not in man.sets:
                raise ManifestError(f"{sysname}: set {s!r} is not in the manifest ({sorted(man.sets)})")
            ids = g["id"].astype(str)
            dup = ids[ids.duplicated()]
            if len(dup):
                raise ManifestError(f"{sysname}/{s}: {len(dup)} duplicated ids, e.g. {dup.tolist()[:3]}")
            want = set(man.sets[s])
            have = set(ids)
            if have != want:
                miss, extra = sorted(want - have), sorted(have - want)
                raise ManifestError(f"{sysname}/{s}: ids differ from the manifest ({len(miss)} missing, e.g. "
                                    f"{miss[:3]}; {len(extra)} not in it, e.g. {extra[:3]})")
            per[sysname][s] = g.set_index(ids)
    strata, desc, hyps = {}, {x: {} for x in systems}, {x: {} for x in systems}
    S = len(systems)
    for name, (s, ids) in _strata_ids(man).items():
        have = [x for x in systems if s in per[x]]
        if not have:
            continue
        ref = None
        E = np.full((len(ids), S), np.nan)
        N = np.full((len(ids), S), np.nan)
        for x in have:
            t = per[x][s].loc[ids]
            rl = t["ref_len"].to_numpy(np.int64)
            if ref is None:
                ref, ref_from = rl, x
            elif not np.array_equal(ref, rl):
                bad = [ids[i] for i in np.flatnonzero(ref != rl)[:3]]
                raise ManifestError(f"{name}: reference lengths of {x} differ from {ref_from}'s on "
                                    f"{int((ref != rl).sum())} ids, e.g. {bad} (another reference or normalisation)")
            j = systems.index(x)
            E[:, j] = t["edits"].to_numpy(np.float64)
            if "edits_nostyle" in t.columns:
                N[:, j] = t["edits_nostyle"].to_numpy(np.float64)
            keep = rl > 0
            d = dict(n=int(keep.sum()))
            for c in ("sub", "del", "ins"):
                if c in t.columns:
                    d[c] = int(t[c].to_numpy()[keep].sum())
            if "hyp_len" in t.columns:
                hl = t["hyp_len"].to_numpy(np.int64)[keep]
                ed, r = t["edits"].to_numpy(np.float64)[keep], rl[keep]
                d["n_runaway"] = int(((ed > r) | (hl > 2 * r)).sum())  # CER > 100 % or hyp > 2x the reference
                d["n_empty_hyp"] = int((hl == 0).sum())
            if "truncated" in t.columns:
                d["n_truncated"] = int(t["truncated"].to_numpy()[keep].astype(bool).sum())
            desc[x][name] = d
            if "hyp" in t.columns:
                hyps[x][name] = [_text(h) for h in t["hyp"].to_numpy()[keep]]
        keep = ref > 0
        strata[name] = Stratum(name=name, ids=[i for i, k in zip(ids, keep) if k], ref_len=ref[keep],
                               edits=E[keep], nostyle=N[keep], n_empty_ref=int((~keep).sum()))
    return Corpus(systems=systems, strata=strata, desc=desc, hyps=hyps, manifest=man)


# ------------------------------------------------------------------------------------------------ sums and bootstrap


@dataclass
class Sums:
    """Per stratum: summed reference chars (shape (...)) and summed edits per system (shape (..., S)). The point
    estimate has shape () / (S,), the bootstrap (B,) / (B, S): every metric is the same expression on either."""
    ref: dict[str, np.ndarray]
    edits: dict[str, np.ndarray]
    nostyle: dict[str, np.ndarray]


def point_sums(corpus: Corpus) -> Sums:
    st = corpus.strata.values()
    return Sums(ref={s.name: np.asarray(float(s.ref_len.sum())) for s in st},
                edits={s.name: s.edits.sum(axis=0) for s in st},
                nostyle={s.name: s.nostyle.sum(axis=0) for s in st})


def stratum_rng(seed: int, name: str) -> np.random.Generator:
    """The stratum's own stream: its draws depend on (seed, stratum name) only, so adding a set or a system changes
    no other stratum's replicates."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(name.encode())]))


def multiplicities(rng: np.random.Generator, n: int, c: int) -> np.ndarray:
    """c bootstrap replicates of n utterances as multiplicity rows (c, n): row b counts how often each utterance was
    drawn in replicate b (n draws with replacement)."""
    idx = rng.integers(0, n, size=(c, n))
    return np.bincount((idx + (np.arange(c, dtype=np.int64) * n)[:, None]).ravel(),
                       minlength=c * n).reshape(c, n)


def bootstrap_sums(corpus: Corpus, B: int = BOOT_B, seed: int = BOOT_SEED, chunk_elems: int = 4_000_000) -> Sums:
    """The paired, stratified utterance bootstrap: per stratum, B multiplicity rows (drawn in chunks of about
    chunk_elems entries) times the stacked [ref_len | edits | no-style edits] columns of every system."""
    ref, edits, nost = {}, {}, {}
    S = len(corpus.systems)
    for name in sorted(corpus.strata):
        st = corpus.strata[name]
        n = len(st.ref_len)
        X = np.concatenate([st.ref_len[:, None].astype(np.float64), st.edits, st.nostyle], axis=1)
        rng = stratum_rng(seed, name)
        out = np.empty((B, X.shape[1]))
        c = max(1, min(B, chunk_elems // max(n, 1)))
        for b0 in range(0, B, c):
            k = min(c, B - b0)
            out[b0:b0 + k] = multiplicities(rng, n, k).astype(np.float64) @ X
        ref[name], edits[name], nost[name] = out[:, 0], out[:, 1:1 + S], out[:, 1 + S:]
    return Sums(ref=ref, edits=edits, nostyle=nost)


def metric_values(sums: Sums, name: str) -> np.ndarray | None:
    """(..., S) values of a METRICS entry; None when one of its strata is not in the corpus."""
    agg, sets, nostyle = METRICS[name]
    if any(s not in sums.ref for s in sets):
        return None
    E = sums.nostyle if nostyle else sums.edits
    with np.errstate(invalid="ignore", divide="ignore"):
        if agg == "macro":
            return np.mean([E[s] / sums.ref[s][..., None] for s in sets], axis=0)
        return sum(E[s] for s in sets) / sum(sums.ref[s] for s in sets)[..., None]


def stratum_cer(sums: Sums, name: str, nostyle: bool = False) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums.nostyle if nostyle else sums.edits)[name] / sums.ref[name][..., None]


# ------------------------------------------------------------------------------------------------ CIs and calls


def sigma_run(m4_replicate: float | None, m4_original: float | None, prior: float = SIGMA_RUN_PRIOR,
              factor: float = SIGMA_HAT_FACTOR) -> dict:
    """4.4: sigma_run = max(prior, factor |ln(M4_rep / M4_orig)|); the prior alone without the replicate pair."""
    hat = None
    if m4_replicate and m4_original and m4_replicate > 0 and m4_original > 0:
        hat = factor * abs(math.log(m4_replicate / m4_original))
    value = prior if hat is None else max(prior, hat)
    source = ("the prior: no replicate pair" if hat is None else
              "the replicate: its estimate exceeds the prior" if hat > prior
              else "the prior: the replicate's estimate is smaller")
    return dict(sigma_run=value, prior=prior, sigma_hat=hat, factor=factor, source=source)


def ci(ln_r: float, v_boot: float, sigma: float, n_noisy: int, z: float = Z) -> tuple[float, float, float]:
    """(lo, hi, se) of ln r: ln r +- z sqrt(v_boot + n_noisy sigma^2)."""
    se = math.sqrt(max(v_boot, 0.0) + n_noisy * sigma ** 2)
    return ln_r - z * se, ln_r + z * se, se


def tolerance_call(lo, hi, delta: float):
    """WITHIN if hi <= ln(1 + delta), OUTSIDE if lo > ln(1 + delta), else UNRESOLVED (vectorised over arrays)."""
    t = math.log1p(delta)
    lo, hi = np.asarray(lo), np.asarray(hi)
    out = np.where(hi <= t, "WITHIN", np.where(lo > t, "OUTSIDE", "UNRESOLVED"))
    return str(out) if out.ndim == 0 else out


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_probabilities(true_ratio: float, delta: float, sigma_total: float, z: float = Z) -> dict:
    """Normal theory of the calls (STUDY.md 4.5's table): ln r_hat ~ N(ln r, sigma_total^2), CI +- z sigma_total."""
    mu, t = math.log(true_ratio), math.log1p(delta)
    w = _phi((t - z * sigma_total - mu) / sigma_total)
    o = 1.0 - _phi((t + z * sigma_total - mu) / sigma_total)
    return dict(WITHIN=w, OUTSIDE=o, UNRESOLVED=1.0 - w - o)


def simulate_calls(true_ratio: float, delta: float, v_boot: float, sigma: float, n_noisy: int = 2,
                   n: int = 20_000, seed: int = 0, z: float = Z) -> dict:
    """Monte Carlo of the calls through ci() and tolerance_call(): n studies whose ln r_hat is drawn around ln
    true_ratio with the CI's own total variance (v_boot + n_noisy sigma^2). Returns the call frequencies."""
    rng = np.random.default_rng(seed)
    se = math.sqrt(v_boot + n_noisy * sigma ** 2)
    lr = math.log(true_ratio) + rng.normal(0.0, se, n)
    lo, hi, _ = ci(0.0, v_boot, sigma, n_noisy, z)  # the half-width does not depend on ln r
    calls = tolerance_call(lr + lo, lr + hi, delta)
    return {c: float((calls == c).mean()) for c in ("WITHIN", "OUTSIDE", "UNRESOLVED")}


def g_of(ln_r: float, h: float) -> float:
    """Growth per halving: r^(1/h) - 1."""
    return math.expm1(ln_r / h)


# ------------------------------------------------------------------------------------------------ the analysis


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class Study:
    """The analysis of one corpus: point sums, the bootstrap and sigma_run, and every comparison built from them."""

    def __init__(self, corpus: Corpus, settings: Mapping | None = None, params: Mapping[str, int] | None = None):
        self.corpus = corpus
        self.st = {**DEFAULT_SETTINGS, **(settings or {})}
        self.params = {**PARAMS_TOTAL, **(params or {})}
        self.z = float(self.st["z"])
        self.deltas = [float(d) for d in self.st["deltas"]]
        self.point = point_sums(corpus)
        self.boot = bootstrap_sums(corpus, int(self.st["boot_b"]), int(self.st["boot_seed"]))
        self._pm, self._bm = {}, {}
        rep, orig = self.st["replicate"]
        self.sigma = sigma_run(self.value(rep, "m4"), self.value(orig, "m4"), float(self.st["sigma_run_prior"]),
                               float(self.st["sigma_hat_factor"]))
        self.s = self.sigma["sigma_run"]

    # -- values
    def has(self, system: str) -> bool:
        return system in self.corpus.systems

    def _metric(self, name: str, boot: bool):
        cache = self._bm if boot else self._pm
        if name not in cache:
            cache[name] = metric_values(self.boot if boot else self.point, name)
        return cache[name]

    def value(self, system: str, metric: str = "m4") -> float | None:
        v = self._metric(metric, False)
        if v is None or not self.has(system):
            return None
        return _f(v[self.corpus.index(system)])

    # -- comparisons
    def compare(self, a: str, b: str, metric: str = "m4", deltas: Iterable[float] | None = None) -> dict | None:
        """r = metric(a) / metric(b), its CI (v_boot + k sigma_run^2, k = trained systems among a, b) and the
        tolerance calls; None when either system or the metric is unavailable."""
        pa, pb = self.value(a, metric), self.value(b, metric)
        if pa is None or pb is None or pa <= 0 or pb <= 0:
            return None
        bv = self._metric(metric, True)
        ia, ib = self.corpus.index(a), self.corpus.index(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.log(bv[:, ia]) - np.log(bv[:, ib])
        fin = np.isfinite(lb)
        v_boot = float(np.var(lb[fin], ddof=1)) if fin.sum() > 1 else float("nan")
        ln_r = math.log(pa / pb)
        k = int(is_trained(a)) + int(is_trained(b))
        lo, hi, se = ci(ln_r, v_boot, self.s, k, self.z)
        out = dict(a=a, b=b, metric=metric, a_value=pa, b_value=pb, ratio=pa / pb, ln_ratio=ln_r,
                   ci_ln=[lo, hi], ci_ratio=[math.exp(lo), math.exp(hi)], v_boot=v_boot, se=se, n_noisy=k,
                   n_boot_nonfinite=int((~fin).sum()))
        ds = self.deltas if deltas is None else list(deltas)
        out["calls"] = {_dkey(d): tolerance_call(lo, hi, d) for d in ds}
        return out

    def boot_ln_ratio(self, a: str, b: str, metric: str = "m4") -> np.ndarray:
        bv = self._metric(metric, True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(bv[:, self.corpus.index(a)]) - np.log(bv[:, self.corpus.index(b)])

    # -- 4.5 the walk
    def family(self, fam: str) -> dict:
        spec = LADDERS[fam]
        top = spec["top"]
        entries = []
        for s in spec["sizes"]:
            c = self.compare(s, top, "m4")
            e = dict(system=s, display=display(s), params_total=self.params.get(s), confound=CONFOUNDS.get((fam, s)),
                     m4=c, available=c is not None)
            if c is not None:
                e["qualifiers"] = {p: self.compare(s, top, p) for p in PAIRS}
            entries.append(e)
        walks = {_dkey(d): walk(top, [(e["system"], e["m4"]["calls"][_dkey(d)] if e["m4"] else None)
                                      for e in entries]) for d in self.deltas}
        return dict(family=fam, text=spec["text"], top=top, top_display=display(top), top_m4=self.value(top),
                    teacher=spec["teacher"], entries=entries, walks=walks,
                    primary=walks.get(_dkey(float(self.st["delta_primary"]))))

    # -- 4.6 g per halving, delta-g, init effect, gaps, budget
    def h(self, big: str, small: str) -> float | None:
        pb, ps = self.params.get(base_run(big)), self.params.get(base_run(small))
        return math.log2(pb / ps) if pb and ps else None

    def g_step(self, fam: str, big: str, small: str, label: str) -> dict:
        c = self.compare(small, big, "m4")
        h = self.h(big, small)
        out = dict(family=fam, big=big, small=small, step=f"{display(big)} -> {display(small)}", label=label, h=h,
                   available=c is not None and h is not None)
        if out["available"]:
            lo, hi = c["ci_ln"]
            out.update(ratio=c["ratio"], ci_ratio=c["ci_ratio"], g=g_of(c["ln_ratio"], h),
                       ci_g=[g_of(lo, h), g_of(hi, h)])
        return out

    def gap(self, fam: str, student: str, teacher: str) -> dict | None:
        """A distillation gap (4.6): M4(student) / M4(own teacher) with its CI (the teacher is a fixed model, so
        v_boot + sigma_run^2), and g per halving over h = log2(N_teacher / N_student) (1.743 Transcribe, 0.986
        Parakeet: STUDY.md 1.3, 4.3), labelled as a gap: never a step of the walk. None without both tables."""
        c = self.compare(student, teacher, "m4")
        if c is None:
            return None
        h = self.h(teacher, student)
        lo, hi = c["ci_ln"]
        return dict(c, family=fam, label="distillation gap (teacher -> student), not a step of the walk", h=h,
                    g=g_of(c["ln_ratio"], h) if h else None, ci_g=[g_of(lo, h), g_of(hi, h)] if h else None)

    def delta_g(self) -> dict:
        """Delta-g = g(T-0.1B -> T-0.05B) - g(bridge -> T-0.1B), the scratch ladder only (4.6). v_boot from the
        bootstrap of delta-g itself; run noise by the delta method over the three independent runs:
        var = sigma^2 (a1^2 + (a1 + a2)^2 + a2^2), a_i = d g_i / d ln r_i = exp(ln r_i / h_i) / h_i."""
        b, m, s = "study-bridge", "study-t01", "study-t005"
        if not all(self.has(x) and self.value(x) for x in (b, m, s)):
            return dict(available=False, reason="needs bridge, T-0.1B and T-0.05B")
        h1, h2 = self.h(b, m), self.h(m, s)
        l1, l2 = math.log(self.value(m) / self.value(b)), math.log(self.value(s) / self.value(m))
        d = g_of(l2, h2) - g_of(l1, h1)
        db = np.expm1(self.boot_ln_ratio(s, m) / h2) - np.expm1(self.boot_ln_ratio(m, b) / h1)
        db = db[np.isfinite(db)]
        v_boot = float(np.var(db, ddof=1))
        a1, a2 = math.exp(l1 / h1) / h1, math.exp(l2 / h2) / h2
        v_run = self.s ** 2 * (a1 ** 2 + (a1 + a2) ** 2 + a2 ** 2)
        se = math.sqrt(v_boot + v_run)
        return dict(available=True, g_upper=g_of(l1, h1), g_lower=g_of(l2, h2), delta_g=d,
                    ci=[d - self.z * se, d + self.z * se], v_boot=v_boot, v_run=v_run, h=[h1, h2],
                    steps=["bridge -> T-0.1B", "T-0.1B -> T-0.05B"])

    def budget(self, fam: str) -> dict:
        """4.6 budget readout per non-top size: the calls at T/2 (s-half vs top-half) and at T, and
        delta = ln r(T) - ln r(T/2) with CI (v_boot + 2 sigma^2): below 0 -> compute-limited."""
        spec = LADDERS[fam]
        top, rows = spec["top"], []
        for s in spec["sizes"]:
            full, half = self.compare(s, top), self.compare(s + HALF, top + HALF)
            row = dict(system=s, display=display(s), at_T=full, at_T_half=half,
                       available=full is not None and half is not None)
            if row["available"]:
                d = full["ln_ratio"] - half["ln_ratio"]
                db = self.boot_ln_ratio(s, top) - self.boot_ln_ratio(s + HALF, top + HALF)
                db = db[np.isfinite(db)]
                v_boot = float(np.var(db, ddof=1))
                lo, hi, se = ci(d, v_boot, self.s, 2, self.z)
                limited = hi < 0
                row.update(delta=d, ci=[lo, hi], v_boot=v_boot, se=se, compute_limited=bool(limited),
                           label="compute-limited: the gap closes with compute" if limited
                           else "not compute-limited at T")
            else:
                miss = [x for x in (s, top, s + HALF, top + HALF) if not self.has(x)]
                row["reason"] = f"missing {', '.join(miss)}" if miss else "metric unavailable"
            rows.append(row)
        return dict(family=fam, top=top, rows=rows)

    # -- 4.7 practical bars, cross-family, Pareto
    def bars(self) -> dict:
        """4.7's practical bars over the seven students (the controls are not candidates), smallest by total
        parameters: the Parakeet TDT bar - JSUT + Galgame-neutral CI upper end at or below TDT's (student vs a fixed
        model: v_boot + sigma_run^2) - and the own-teacher bars 1.2x / 1.5x on M4, both by the point ratio and by the
        CI's upper end (the call WITHIN at delta = bar - 1)."""
        students = [s for s in STUDENTS if self.has(s)]
        by_size = sorted(students, key=lambda s: self.params.get(s) or math.inf)
        tdt = {}
        for s in by_size:
            c = self.compare(s, "parakeet-tdt", "jg")
            tdt[s] = None if c is None else dict(c, meets=c["ci_ln"][1] <= 0.0)
        passing = [s for s in by_size if tdt.get(s) and tdt[s]["meets"]]
        teacher = {}
        for s in by_size:
            t = teacher_of(s)
            c = self.compare(s, t, "m4", deltas=[b - 1.0 for b in self.st["teacher_bars"]]) if t else None
            if c is not None:
                c["point_within"] = {_bkey(b): c["ratio"] <= b for b in self.st["teacher_bars"]}
                c["ci_within"] = {_bkey(b): c["calls"][_dkey(b - 1.0)] == "WITHIN" for b in self.st["teacher_bars"]}
            teacher[s] = c
        smallest = {}
        for b in self.st["teacher_bars"]:
            k = _bkey(b)
            pt = [s for s in by_size if teacher.get(s) and teacher[s]["point_within"][k]]
            cw = [s for s in by_size if teacher.get(s) and teacher[s]["ci_within"][k]]
            smallest[k] = dict(point=pt[0] if pt else None, ci=cw[0] if cw else None)
        return dict(tdt=dict(metric="jg", per_student=tdt, smallest=passing[0] if passing else None,
                             available="parakeet-tdt" in self.corpus.systems),
                    teacher=dict(metric="m4", per_student=teacher, smallest=smallest))

    def cross_family(self) -> dict:
        """JSUT + Galgame-neutral, raw and no-style, per system, and the paired comparisons T vs P at equal size
        (0.3B, 0.1B, 0.05B): descriptive only, no threshold is ever applied (4.3). The Parakeet TDT bar is bars()'."""
        rows = {s: dict(jg=self.value(s, "jg"), jg_nostyle=self.value(s, "jg_nostyle"))
                for s in system_order(self.corpus.systems)}
        pairs = []
        for a, b in (("study-t03", "study-p03"), ("study-t01", "study-p01"), ("study-t005", "study-p005")):
            pairs.append(dict(a=a, b=b, raw=self.compare(a, b, "jg", deltas=[]),
                              nostyle=self.compare(a, b, "jg_nostyle", deltas=[])))
        return dict(per_system=rows, equal_size=pairs)

    def anchor(self) -> dict:
        """4.1's regression flag: the study's T-0.6B worse than the first run's 0.6B (re-scored on the manifest) on
        the gate-pooled CER by more than 5 % relative, with the paired CI of ln r above 0. Both are trained runs, so
        the CI is 4.4's student-student one (v_boot + 2 sigma_run^2)."""
        c = self.compare("study-t06", ANCHOR, "gate_pooled", deltas=[])
        if c is None:
            return dict(available=False, flag=None)
        flag = c["ratio"] > 1.0 + float(self.st["anchor_flag_rel"]) and c["ci_ln"][0] > 0.0
        return dict(available=True, comparison=c, flag=bool(flag), threshold_rel=float(self.st["anchor_flag_rel"]))


def _dkey(d: float) -> str:
    return f"{d:g}"


def _bkey(b: float) -> str:
    return f"{b:g}x"


def walk(top: str, entries: Sequence[tuple[str, str | None]]) -> dict:
    """STUDY.md 4.5's walk over (size, call) pairs, largest first. X = the smallest size WITHIN with all larger
    sizes WITHIN (the top when the first size is not WITHIN):
      the first OUTSIDE at Y -> "limit reached between X and Y"
      the first UNRESOLVED at Y -> "the limit is at or below X; Y unresolved"
      a size without results -> the walk stops there ("incomplete")
      every size WITHIN -> the limit is the smallest size (not reached on the ladder)
    Sizes after the stop are listed as descriptive: they are reported, never claimed."""
    last = top
    for i, (s, call) in enumerate(entries):
        rest = [e[0] for e in entries[i + 1:]]
        if call == "WITHIN":
            last = s
            continue
        if call == "OUTSIDE":
            return dict(status="reached", limit=last, stop=s, descriptive=rest,
                        sentence=f"limit reached between {display(last)} and {display(s)}")
        if call == "UNRESOLVED":
            return dict(status="unresolved", limit=last, stop=s, descriptive=rest,
                        sentence=f"the limit is at or below {display(last)}; {display(s)} unresolved")
        return dict(status="incomplete", limit=last, stop=s, descriptive=rest,
                    sentence=f"the walk stops at {display(s)}: no results for it (limit at or below {display(last)} "
                             f"so far)")
    return dict(status="not_reached", limit=last, stop=None, descriptive=[],
                sentence=f"every size down to {display(last)} is WITHIN: the limit is not reached on this ladder")


def pareto(points: Mapping[str, Sequence[float | None]]) -> list[str]:
    """The systems no other system beats: another dominates p if it is <= p on every objective and < on one.
    Points with a missing objective are left out."""
    ok = {k: np.asarray(v, float) for k, v in points.items() if all(x is not None and math.isfinite(x) for x in v)}
    front = []
    for k, p in ok.items():
        if not any(np.all(q <= p) and np.any(q < p) for j, q in ok.items() if j != k):
            front.append(k)
    return sorted(front, key=lambda k: tuple(ok[k]))


# ------------------------------------------------------------------------------------------------ section 7 checks


def check(name: str, status: str, detail: str, **extra) -> dict:
    """One invalidation rule: status "pass" | "fail" | "not_checked" (the inputs do not show it)."""
    return dict(rule=name, status=status, detail=detail, **extra)


def teacher_baseline_check(study: Study, prereg_baselines: Mapping[str, Mapping[str, float]] | None,
                           defaults: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """The teacher baselines reproduce: Cohere's corpus CER on each gate set within BASELINE_TOL (0.05 pp) of the
    pre-registered number (defaults: kitsune.evaluate.TEACHER_CER_PREREG; PREREG.json's "baselines" override and
    extend them), and every other teacher's pre-registered number (PREREG.json baselines) likewise."""
    if defaults is None:
        import kitsune.evaluate as ev

        # PARAKEET_CTC_CER_PREREG arrives with the evaluator's family switch (WP5b); read it when it is there
        defaults = {"cohere": TEACHER_CER_PREREG, "parakeet-ctc": getattr(ev, "PARAKEET_CTC_CER_PREREG", None) or {}}
    want = {k: dict(v) for k, v in defaults.items() if v}
    for sysname, sets in (prereg_baselines or {}).items():
        want[sysname] = {**want.get(sysname, {}), **{k: float(v) for k, v in sets.items()}}
    rows, fails, missing = [], [], []
    for sysname, sets in want.items():
        for s, v in sets.items():
            if not study.has(sysname) or s not in study.point.ref:
                missing.append(f"{sysname}/{s}")
                continue
            got = _f(stratum_cer(study.point, s)[study.corpus.index(sysname)])
            ok = got is not None and abs(got - v) <= BASELINE_TOL
            rows.append(dict(system=sysname, set=s, prereg=v, measured=got, ok=ok))
            if not ok:
                fails.append(f"{sysname}/{s}: {_pct(got)} vs {_pct(v)}")
    if fails:
        return check("teacher_baselines", "fail", "baselines do not reproduce: " + "; ".join(fails), rows=rows)
    if not rows:
        return check("teacher_baselines", "not_checked", "no teacher table for a pre-registered baseline", rows=rows)
    tail = f" ({', '.join(missing)} not in the tables)" if missing else ""
    return check("teacher_baselines", "pass", f"{len(rows)} baselines within {BASELINE_TOL * 100:.2f} pp" + tail,
                 rows=rows)


def lr_edge_check(numbers: Mapping | None) -> dict:
    """An LR winner at a grid edge after its one extension invalidates (7): the winner of every class's probes
    (lowest objective) must not be the smallest or largest LR probed. PREREG_numbers.json lr_probes {class: {lr:
    objective}} holds the extension point too, so an edge winner there is an edge winner after the extension."""
    probes = (numbers or {}).get("lr_probes")
    if not probes:
        return check("lr_edge", "not_checked", "no lr_probes in PREREG_numbers.json")
    bad, rows = [], {}
    for cls, grid in probes.items():
        pts = sorted((float(lr), float(obj)) for lr, obj in grid.items() if obj is not None)
        if len(pts) < 2:
            bad.append(f"{cls}: fewer than 2 probed LRs")
            continue
        win = min(pts, key=lambda p: p[1])[0]
        edge = win in (pts[0][0], pts[-1][0])
        rows[cls] = dict(winner=win, grid=[p[0] for p in pts], at_edge=edge)
        if edge:
            bad.append(f"{cls}: winner {win:g} at the edge of {[p[0] for p in pts]}")
    return check("lr_edge", "fail" if bad else "pass",
                 "; ".join(bad) if bad else f"{len(rows)} classes, every winner inside its grid", classes=rows)


def loader_check(numbers: Mapping | None, limit: float = DATA_WAIT_MAX) -> dict:
    cal = (numbers or {}).get("calibration")
    if not cal:
        return check("loader_bound", "not_checked", "no calibration in PREREG_numbers.json")
    bad = {r: v.get("data_wait_frac") for r, v in cal.items()
           if v.get("data_wait_frac") is None or float(v["data_wait_frac"]) >= limit}
    return check("loader_bound", "fail" if bad else "pass",
                 f"data_wait >= {limit:.0%} (or missing) for {sorted(bad)}" if bad
                 else f"data_wait < {limit:.0%} for all {len(cal)} calibrated runs", runs=bad)


def max_steps_check(numbers: Mapping | None, summaries: Mapping[str, Mapping] | None) -> dict:
    """Every run reached its pre-registered max_steps (PREREG_numbers.json max_steps against the trainer's summary
    "steps"); a run with a summary but no pre-registered number fails too."""
    ms = (numbers or {}).get("max_steps")
    if not ms or not summaries:
        return check("max_steps", "not_checked", "needs PREREG_numbers.json max_steps and the runs' summaries")
    bad, rows = [], {}
    for run, summ in summaries.items():
        target = ms.get(base_run(run))
        steps = summ.get("steps")
        if run.endswith(HALF):
            target = None if target is None else round(0.5 * int(target))  # the branch ends at round(0.5 M)
        rows[run] = dict(steps=steps, max_steps=target)
        if target is None or steps is None or int(steps) != int(target):
            bad.append(f"{run}: {steps} of {target}")
    return check("max_steps", "fail" if bad else "pass",
                 "; ".join(bad) if bad else f"{len(rows)} runs at their max_steps", runs=rows)


def manifest_check(man: Manifest, prereg_hashes: Mapping[str, str] | None) -> dict:
    """The tables all matched the manifest (build_corpus refuses otherwise); the manifest's hashes must also equal
    the ones PREREG.json froze, when it lists them."""
    if not prereg_hashes:
        return check("manifest", "pass", f"every table matches the manifest ({len(man.sets)} sets); PREREG.json "
                                         f"lists no manifest hashes to compare")
    bad = [s for s, h in prereg_hashes.items() if man.sha256.get(s) not in (None, h)]
    miss = [s for s in prereg_hashes if s not in man.sha256]
    if bad or miss:
        return check("manifest", "fail", f"manifest hashes differ from PREREG.json for {bad}"
                     + (f"; not in the manifest: {miss}" if miss else ""))
    return check("manifest", "pass", f"every table matches the manifest, whose {len(prereg_hashes)} hashes equal "
                                     f"PREREG.json's")


def selection_check(summaries: Mapping[str, Mapping] | None) -> dict:
    """One selection for every run: the summaries' selection_sha256 (when they carry one) must all be equal."""
    shas = {r: s.get("selection_sha256") for r, s in (summaries or {}).items() if s.get("selection_sha256")}
    if not shas:
        return check("selection", "not_checked", "no selection_sha256 in the runs' summaries")
    distinct = sorted(set(shas.values()))
    return check("selection", "pass" if len(distinct) == 1 else "fail",
                 f"{len(shas)} runs on one selection" if len(distinct) == 1
                 else f"{len(distinct)} different selections: " + ", ".join(f"{r}={h[:12]}" for r, h in shas.items()))


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.2f} %"


# ------------------------------------------------------------------------------------------------ the whole report


def analyse(corpus: Corpus, settings: Mapping | None = None, *, params: Mapping[str, int] | None = None,
            speed: Mapping[str, Mapping] | None = None, numbers: Mapping | None = None,
            summaries: Mapping[str, Mapping] | None = None, prereg_baselines: Mapping | None = None,
            prereg_hashes: Mapping[str, str] | None = None, imitation: bool = True) -> dict:
    """Everything the study reports from its eval results, as one JSON-able dict (tools/study_report.py renders it).

    speed: {system: {"rtf": float, "vram_gb": float, ...}} (the A100 speed probe); numbers: PREREG_numbers.json;
    summaries: {run: the trainer's summary.json}; prereg_baselines: {teacher: {set: corpus CER}} from PREREG.json;
    prereg_hashes: {set: ids sha256} from PREREG.json."""
    st = Study(corpus, settings, params)
    out = dict(settings=dict(st.st), sigma_run=st.sigma,
               bootstrap=dict(B=int(st.st["boot_b"]), seed=int(st.st["boot_seed"]), z=st.z,
                              strata={k: len(v.ref_len) for k, v in corpus.strata.items()},
                              n_empty_ref={k: v.n_empty_ref for k, v in corpus.strata.items()}))
    out["systems"] = _systems(st, imitation)
    out["families"] = {f: st.family(f) for f in LADDERS}
    out["steps"] = [st.g_step(f, b, s, lab) for f, b, s, lab in STEPS]
    out["delta_g"] = st.delta_g()
    init = st.compare("study-bridge", "study-t03", "m4")
    out["init_effect"] = dict(available=init is not None, comparison=init)
    out["distillation_gaps"] = {f: st.gap(f, s, t) for f, s, t in GAPS}
    out["budget"] = {f: st.budget(f) for f in LADDERS}
    out["bars"] = st.bars()
    out["cross_family"] = st.cross_family()
    out["pareto"] = _pareto(st, speed)
    out["anchor"] = st.anchor()
    # the sampling noise of the primary comparisons and what it means for the calls (the 4.5 table at our noise)
    ses = [e["m4"]["se"] for f in PRIMARY_FAMILIES for e in out["families"][f]["entries"] if e["m4"]]
    if ses:
        s_tot = float(np.median(ses))
        out["operating_characteristics"] = dict(
            sigma_total=s_tot, table={_dkey(r): {_dkey(d): call_probabilities(r, d, s_tot, st.z) for d in st.deltas}
                                      for r in (1.0, 1.05, 1.10, 1.15, 1.20)})
    expected = [*RUNS, *TEACHERS, ANCHOR, *(r + HALF for r in STUDENTS + ("study-bridge",))]
    out["missing_systems"] = [s for s in expected if not st.has(s)]
    out["extra_systems"] = [s for s in corpus.systems if s not in expected]
    anchor_flag = out["anchor"].get("flag")
    out["checks"] = [
        max_steps_check(numbers, summaries),
        check("resume_config", "not_checked", "the trainer refuses a resume with a changed config (RESUME_FIXED); "
                                              "not visible in the eval outputs"),
        manifest_check(corpus.manifest, prereg_hashes),
        selection_check(summaries),
        teacher_baseline_check(st, prereg_baselines),
        check("labels_frames", "not_checked", "K3 / K4 and the frame preflight are the label checks' and the box's "
                                              "record (tools/label_checks.py, the store build)"),
        _timing_check(numbers, summaries),
        check("anchor_regression", "not_checked" if anchor_flag is None else ("fail" if anchor_flag else "pass"),
              "no anchor or T-0.6B table" if anchor_flag is None else
              ("the flag fires: interpretation halts until the cause is understood" if anchor_flag
               else "T-0.6B is not worse than the anchor by > 5 % with a CI excluding 0")),
        lr_edge_check(numbers),
        loader_check(numbers),
    ]
    out["invalid"] = any(c["status"] == "fail" for c in out["checks"])
    return _clean(out)


def _timing_check(numbers: Mapping | None, summaries: Mapping[str, Mapping] | None) -> dict:
    """PREREG_numbers.json before the first study step: its written_utc against each run's started_utc, when the
    summaries carry one (the rules commit's own order is git's record, not visible here)."""
    w = (numbers or {}).get("written_utc")
    starts = {r: s.get("started_utc") for r, s in (summaries or {}).items() if s.get("started_utc")}
    if not w or not starts:
        return check("prereg_timing", "not_checked", "needs PREREG_numbers.json written_utc and the runs' started_utc")
    early = sorted(r for r, t in starts.items() if str(t) < str(w))
    return check("prereg_timing", "fail" if early else "pass",
                 f"runs started before PREREG_numbers.json was written: {early}" if early
                 else f"PREREG_numbers.json ({w}) precedes all {len(starts)} run starts")


def _systems(st: Study, imitation: bool) -> dict:
    from kitsune.evaluate import corpus_cer

    c = st.corpus
    out = {}
    for s in system_order(c.systems):
        j = c.index(s)
        t = teacher_of(s)
        sets = {}
        for name in c.strata:
            if name not in c.desc[s]:
                continue
            d = dict(c.desc[s][name])
            n_ref = float(st.point.ref[name])
            d["cer"] = _f(stratum_cer(st.point, name)[j])
            d["cer_nostyle"] = _f(stratum_cer(st.point, name, True)[j])
            for k in ("sub", "del", "ins"):
                if k in d:
                    d[f"{k}_rate"] = d[k] / n_ref if n_ref else None
            for k in ("runaway", "empty_hyp", "truncated"):
                if f"n_{k}" in d:
                    d[f"{k}_rate"] = d[f"n_{k}"] / d["n"] if d["n"] else None
            if t and st.has(t) and name in c.desc[t]:
                tc = _f(stratum_cer(st.point, name)[c.index(t)])
                d["ratio_vs_teacher"] = d["cer"] / tc if d["cer"] is not None and tc else None
                if imitation and name in c.hyps[s] and name in c.hyps[t]:
                    d["imitation_cer"] = _f(corpus_cer(c.hyps[s][name], c.hyps[t][name])["cer"])
            sets[name] = d
        metrics = {m: st.value(s, m) for m in METRICS}
        vs_t = {m: (metrics[m] / st.value(t, m) if t and metrics[m] and st.value(t, m) else None)
                for m in ("m4", "jg", "gate_pooled")}
        out[s] = dict(display=display(s), role=role_of(s), family=family_of(s), teacher=t, trained=is_trained(s),
                      params_total=params_of(s, st.params), params_non_embedding=PARAMS_NON_EMBEDDING.get(base_run(s)),
                      sets=sets, metrics=metrics, ratio_vs_teacher=vs_t, m4_sets=_pooled_desc(sets, st.point))
    return out


def _pooled_desc(sets: dict, point: Sums) -> dict | None:
    """The 4.2 error profile over the M4 strata: S / D / I per reference char (sum / sum), runaway, empty-output and
    truncation rates per utterance, and the imitation CER (macro, like M4) where every M4 stratum has one."""
    if not all(s in sets for s in M4_SETS):
        return None
    rows = [sets[s] for s in M4_SETS]
    n_ref = sum(float(point.ref[s]) for s in M4_SETS)
    n = sum(r["n"] for r in rows)
    out = {}
    for k in ("sub", "del", "ins"):
        if all(k in r for r in rows):
            out[f"{k}_rate"] = sum(r[k] for r in rows) / n_ref if n_ref else None
    for k in ("runaway", "empty_hyp", "truncated"):
        if all(f"n_{k}" in r for r in rows):
            out[f"{k}_rate"] = sum(r[f"n_{k}"] for r in rows) / n if n else None
    im = [r.get("imitation_cer") for r in rows]
    out["imitation_cer"] = float(np.mean(im)) if all(x is not None for x in im) else None
    return out


SPEED_RTF = ("rtf", "rtf_batched", "batched_rtf")
SPEED_VRAM = ("vram_gb", "vram_peak_reserved_gb", "peak_reserved_gb", "vram_reserved_gb", "vram_peak_allocated_gb",
              "peak_allocated_gb")


def speed_entry(v: Mapping) -> dict:
    """{rtf, vram_gb} of one speed-probe record; the keys accepted are SPEED_RTF / SPEED_VRAM (GB), or a
    *_bytes form of the VRAM keys (divided by 1e9)."""
    rtf = next((float(v[k]) for k in SPEED_RTF if v.get(k) is not None), None)
    vram = next((float(v[k]) for k in SPEED_VRAM if v.get(k) is not None), None)
    if vram is None:
        vram = next((float(v[k[:-3] + "_bytes"]) / 1e9 for k in SPEED_VRAM if v.get(k[:-3] + "_bytes") is not None),
                    None)
    return dict(rtf=rtf, vram_gb=vram, **{k: v[k] for k in ("p50_s", "p95_s") if k in v})


def _pareto(st: Study, speed: Mapping[str, Mapping] | None) -> dict:
    if not speed:
        return dict(available=False, reason="no speed JSON")
    sp = {s: speed_entry(v) for s, v in speed.items()}
    rows = {s: dict(sp[s], jg=st.value(s, "jg"), m4=st.value(s, "m4")) for s in system_order(sp) if st.has(s)}
    fronts = {m: pareto({s: (r[m], r["rtf"], r["vram_gb"]) for s, r in rows.items()}) for m in ("jg", "m4")}
    return dict(available=True, rows=rows, front=fronts,
                note="jg is the cross-family CER (Reazon left out, Parakeet was trained on it); the m4 front "
                     "mixes families on a metric that favours Parakeet on Reazon")


def _clean(x):
    """JSON-able: numpy scalars to Python, NaN / inf to None, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _f(x)
    if isinstance(x, np.str_):
        return str(x)
    return x


# ------------------------------------------------------------------------------------------------ PREREG settings

PREREG_BLOCKS = ("metrics", "limit", "limit_rule", "noise", "statistics", "stats", "tolerance")


def settings_from_prereg(prereg: Mapping | None) -> tuple[dict, dict]:
    """(settings, sources): the analysis settings PREREG.json fixes, else STUDY.md's defaults (DEFAULT_SETTINGS).

    Read from the "analysis" block, else the top level, else one of the blocks PREREG_BLOCKS names (the first found
    wins; other blocks are never searched, so an unrelated "delta" or "replicate" elsewhere cannot leak in):
      delta (or delta_primary)           the primary tolerance, a fraction (0.10)
      deltas                             every tolerance reported ([0.05, 0.1, 0.2])
      sigma_run: {prior, factor, replicate: [rep run, original run]}  (or sigma_run_prior / sigma_hat_factor)
      bootstrap: {B, seed, z}            (or boot_b / boot_seed)
      teacher_bars ([1.2, 1.5]), anchor_flag_rel (0.05)
    sources says, per setting, "PREREG.json:<path>" or "default (STUDY.md)". Values out of range raise ValueError
    (a percent typed as 10 for 0.10 must not silently widen the tolerance)."""
    st, src = dict(DEFAULT_SETTINGS), {k: "default (STUDY.md)" for k in DEFAULT_SETTINGS}
    if not prereg:
        return st, src
    blocks = []
    if isinstance(prereg.get("analysis"), Mapping):
        blocks.append(("analysis", prereg["analysis"]))
    blocks.append(("", prereg))
    blocks += [(k, prereg[k]) for k in PREREG_BLOCKS if isinstance(prereg.get(k), Mapping)]

    def find(*paths):
        for bname, b in blocks:
            for path in paths:
                cur, ok = b, True
                for p in path.split("."):
                    if isinstance(cur, Mapping) and p in cur:
                        cur = cur[p]
                    else:
                        ok = False
                        break
                if ok and not isinstance(cur, Mapping):
                    return cur, f"PREREG.json:{bname + '.' if bname else ''}{path}"
        return None, None

    spec = {"delta_primary": ("delta_primary", "delta", "tolerance.delta"),
            "deltas": ("deltas", "tolerance.deltas"),
            "sigma_run_prior": ("sigma_run.prior", "sigma_run_prior"),
            "sigma_hat_factor": ("sigma_run.factor", "sigma_hat_factor"),
            "replicate": ("sigma_run.replicate", "replicate"),
            "boot_b": ("bootstrap.B", "bootstrap.b", "boot_b"),
            "boot_seed": ("bootstrap.seed", "boot_seed"),
            "z": ("bootstrap.z", "z"),
            "teacher_bars": ("teacher_bars",), "anchor_flag_rel": ("anchor_flag_rel",)}
    for key, paths in spec.items():
        v, where = find(*paths)
        if where is not None:
            st[key], src[key] = v, where
    for key in ("delta_primary", "sigma_run_prior", "anchor_flag_rel"):
        if not (isinstance(st[key], (int, float)) and 0 < float(st[key]) < 1):
            raise ValueError(f"{key} = {st[key]!r} ({src[key]}): must be a fraction in (0, 1)")
    if not st["deltas"] or not all(isinstance(d, (int, float)) and 0 < d < 1 for d in st["deltas"]):
        raise ValueError(f"deltas = {st['deltas']!r} ({src['deltas']}): fractions in (0, 1)")
    if float(st["delta_primary"]) not in [float(d) for d in st["deltas"]]:
        st["deltas"] = sorted([*st["deltas"], st["delta_primary"]])
    if not (isinstance(st["boot_b"], int) and st["boot_b"] >= 100):
        raise ValueError(f"bootstrap B = {st['boot_b']!r} ({src['boot_b']}): an integer >= 100")
    if not isinstance(st["boot_seed"], int):
        raise ValueError(f"bootstrap seed = {st['boot_seed']!r} ({src['boot_seed']}): an integer")
    if not (isinstance(st["replicate"], (list, tuple)) and len(st["replicate"]) == 2):
        raise ValueError(f"replicate = {st['replicate']!r}: [replicate run, original run]")
    st["replicate"] = list(st["replicate"])
    return st, src


def prereg_baselines(prereg: Mapping | None) -> dict:
    """{teacher: {set: corpus CER}} from PREREG.json "baselines" (fractions), for the baseline check."""
    b = (prereg or {}).get("baselines") or {}
    out = {}
    for sysname, sets in b.items():
        if isinstance(sets, Mapping):
            vals = {k: float(v) for k, v in sets.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1}
            if vals:
                out[sysname] = vals
    return out


def prereg_manifest_hashes(prereg: Mapping | None) -> dict:
    """{set: ids sha256} that PREREG.json froze, from "manifest": {set: sha | {"ids_sha256": sha}} when present."""
    m = (prereg or {}).get("manifest") or {}
    m = m.get("sets", m) if isinstance(m, Mapping) else {}
    out = {}
    for s, v in m.items():
        h = v if isinstance(v, str) else (v.get("ids_sha256") or v.get("sha256")) if isinstance(v, Mapping) else None
        # only the eval sets' digests: a "path" or the manifest file's own sha256 next to them is not a set
        if s in EVAL_SETS and isinstance(h, str) and len(h) == 64 and all(ch in "0123456789abcdef" for ch in h):
            out[s] = h
    return out
