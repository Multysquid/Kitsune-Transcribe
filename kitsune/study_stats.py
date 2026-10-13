"""The size study's scoring and statistics on the frozen eval manifest, which the full-data report reuses.

The size study (study/STUDY.md 4.3-4.7; reported 2026-09-27) compared every system on the per-utterance results of
one frozen eval manifest (labels/full/selections/study_manifest.json). Its analysis (the limit walks, the offers, the
sensitivity and the invalidation checks) was removed after the study; the last commit with it is 76b28a1. What is
left is what tools/full_report.py and scripts/05_evaluate.py read:

Input (CONTRACT.md 5): per system, a per-utterance table with columns id, set, edits, ref_len (character edit distance
and reference length on kitsune.text.normalize_ja strings, the kitsune.evaluate normalisation) and optionally
edits_nostyle, sub, del, ins, hyp, ref, truncated. score_utterances() makes those columns from ref / hyp text when a
table has only the text (a 05_evaluate greedy_<set>.parquet). build_corpus() aligns every table to the manifest's
ordered ids and REFUSES (ManifestError) anything that does not match it: a missing, extra or duplicated id, an unknown
set, a manifest whose ids do not hash to its own sha256, a reference length that differs between two systems. Every
number is then over exactly the same utterances for every system.

Strata. Every eval set is one stratum; the Galgame set is split into its manifest views instead (galgame_neutral, the
primary view; galgame_all; galgame_label_box). Rows with an empty reference are left out of every sum and counted, as
kitsune.evaluate.corpus_cer does (Galgame has 31 of them). Corpus CER of a stratum = sum edits / sum ref_len; M4 is
the macro mean of JSUT, CV8, ReazonSpeech-test and Galgame-neutral (M4_SETS), equal weights.

Noise (4.4). v_boot comes from a paired, stratified utterance bootstrap: B replicates (10,000), a fixed seed, each
stratum resampled with replacement within itself. A replicate is a vector of multiplicities per utterance, drawn once
per stratum and applied to every system's column, so every system (and every metric) sees the SAME resampled
utterances; the draws depend only on (seed, stratum name, stratum size), not on which systems are in the tables.
Multiplicities and edit counts are integers, so the resampled sums are exact in float64 whatever the BLAS summation
order: the same seed gives bit-identical numbers on any machine. Run-to-run noise sigma_run is the SD of ln M4 of one
system: max(prior 1.6 %, 0.886 |ln(M4_replicate / M4_T-0.1B)|) (0.886 = sqrt(pi) / 2 makes one pair's |difference|
unbiased for sigma; REPLICATE is the pair). The CI of a ratio r = a / b is ln r +- 1.96 sqrt(v_boot + k sigma_run^2),
with k the number of TRAINED systems among a and b (2 student-student, 1 student-teacher; the teachers are fixed
models).

The PREREG readers (prereg_baselines, prereg_manifest) read study/PREREG.json as it is written: numbers among prose,
parakeet_ctc / galgame:<view> keys, a manifest block "pending" until the labels were sealed (it is filled).

Everything here is numpy and pandas on CPU (no torch).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from kitsune.store import ids_sha256

# ------------------------------------------------------------------------------------------------ the pre-registration

SIGMA_RUN_PRIOR = 0.016  # the first run's residual SD of the 4-set macro around its trend
SIGMA_HAT_FACTOR = 0.886  # sqrt(pi) / 2: E|X - Y| = 2 sigma / sqrt(pi) for two independent runs
BOOT_B = 10_000
BOOT_SEED = 1234
Z = 1.96
REPLICATE = ("study-t01-s1235", "study-t01")  # sigma_run's pair: the seed replicate and the run it repeats

# ------------------------------------------------------------------------------------------------ sets

EVAL_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame")
GALGAME = "galgame"
GALGAME_VIEWS = ("neutral", "all", "label_box")
NEUTRAL = "galgame_neutral"
M4_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", NEUTRAL)
PAIRS = {"ood": ("eval_jsut", "eval_cv8"), "ind": ("eval_reazon", NEUTRAL)}
CROSS_SETS = ("eval_jsut", NEUTRAL)


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

# exact total counts, STUDY.md 1.1 (meta-device builds; every builder asserts total == closed form).
# The anchor is the first run's B20x2560-d4 = the T-0.6B shape; parakeet-ctc is the unpruned CTC path (24 x 4096);
# cohere is the teacher itself (tests/test_student.py pins it), the far end of the Transcribe distillation gap.
# T-0.3B and the bridge are B8x2560 + decoder {0,2,5,7} (the owner's decision of 2026-09-26; kitsune.prereg.RUNS).
PARAMS_TOTAL = {"study-t06": 616_963_328, "study-t03": 301_822_208, "study-bridge": 301_822_208,
                "study-t01": 103_996_416, "study-t01-s1235": 103_996_416, "study-t005": 51_209_600,
                "study-p03": 308_524_033, "study-p01": 98_468_865, "study-p005": 52_190_209,
                "parakeet-ctc": 610_898_945, ANCHOR: 616_963_328, "cohere": 2_065_647_872}


def base_run(system: str) -> str:
    return system[: -len(HALF)] if system.endswith(HALF) else system


def family_of(system: str) -> str | None:
    b = base_run(system)
    if b.startswith("study-p"):
        return "ctc"
    if b.startswith("study-") or b == ANCHOR:
        return "aed"
    return {"cohere": "aed", "parakeet-ctc": "ctc", "parakeet-tdt": "ctc"}.get(b)


def display(system: str) -> str:
    b = base_run(system)
    name = DISPLAY.get(b, b)
    return f"{name} T/2" if system.endswith(HALF) else name


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
    view_sha256: dict[str, str] = field(default_factory=dict)  # Galgame view -> ids_sha256 of its ids (PREREG compares)


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
    views, view_sha = {}, {}
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
        views[vname], view_sha[vname] = ids, ids_sha256(ids)
    return Manifest(sets=sets, sha256=sha, views=views, view_sha256=view_sha)


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


def stratum_cer(sums: Sums, name: str, nostyle: bool = False) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums.nostyle if nostyle else sums.edits)[name] / sums.ref[name][..., None]


# ------------------------------------------------------------------------------------------------ CIs


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


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------------------------------------ the manifest check


def check(name: str, status: str, detail: str, **extra) -> dict:
    """One check: status "pass" | "fail" | "not_checked" (the inputs do not show it)."""
    return dict(rule=name, status=status, detail=detail, **extra)


def manifest_check(man: Manifest, prereg: Mapping | None, manifest_file_sha256: str | None = None) -> dict:
    """The manifest the tables matched (build_corpus refuses any other) is the one PREREG.json froze (7: "a selection
    or manifest hash differs"). prereg is prereg_manifest(): every eval set's ids sha256 and every Galgame view's
    (M4 depends on the neutral one) must equal PREREG's, the manifest may hold no set PREREG does not list and lack
    none it does, and the manifest file's own sha256 must equal PREREG's manifest_sha256 when both are known. A
    PREREG manifest block that is absent or still pending leaves the check not_checked; a filled one from which no
    hash can be read fails. Never a pass that compared nothing."""
    tables = f"every table matches the manifest ({len(man.sets)} sets)"
    p = prereg or {}
    status = p.get("status")
    if status in (None, "absent"):
        return check("manifest", "not_checked", f"{tables}; PREREG.json has no manifest hashes to compare it with")
    if status == "pending":
        return check("manifest", "not_checked", f"{tables}; PREREG.json's manifest block is still pending")
    sets, views, file_sha = p.get("sets") or {}, p.get("views") or {}, p.get("manifest_sha256")
    if not (sets or views or file_sha):
        return check("manifest", "fail", f"PREREG.json's manifest block ({status}) holds no hash this report can read")
    bad = []
    for s in sorted(set(sets) | set(man.sha256)):
        if s not in man.sha256:
            bad.append(f"{s}: pre-registered, not in the manifest")
        elif s not in sets:
            bad.append(f"{s}: in the manifest, not pre-registered")
        elif sets[s] != man.sha256[s]:
            bad.append(f"{s}: ids hash {man.sha256[s][:12]}, pre-registered {sets[s][:12]}")
    mv = dict(man.view_sha256)
    if GALGAME in man.sha256 and "all" in views:
        mv.setdefault("all", man.sha256[GALGAME])  # no "all" view: it is the whole Galgame set (_strata_ids)
    for v in sorted(set(views) | (set(mv) if views else set())):
        if v not in mv:
            bad.append(f"Galgame view {v}: pre-registered, not in the manifest")
        elif v not in views:
            bad.append(f"Galgame view {v}: in the manifest, not pre-registered")
        elif views[v] != mv[v]:
            bad.append(f"Galgame view {v}: ids hash {mv[v][:12]}, pre-registered {views[v][:12]}")
    notes = [] if views or not man.view_sha256 else ["PREREG.json lists no Galgame view hashes: views not compared"]
    if file_sha and manifest_file_sha256:
        if file_sha != manifest_file_sha256:
            bad.append(f"the manifest file hashes to {manifest_file_sha256[:12]}, pre-registered {file_sha[:12]}")
    elif file_sha:
        notes.append("the manifest file's own sha256 was not compared (no file given)")
    if bad:
        return check("manifest", "fail", "the manifest differs from PREREG.json: " + "; ".join(bad))
    what = [f"{len(sets)} set hashes"] + ([f"{len(views)} Galgame view hashes"] if views else []) \
        + (["the file's sha256"] if file_sha and manifest_file_sha256 else [])
    return check("manifest", "pass", f"{tables}; {', '.join(what)} equal PREREG.json's"
                 + "".join(f"; {n}" for n in notes))


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


# ------------------------------------------------------------------------------------------------ PREREG readers


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _baseline_system(name: str) -> str:
    """kitsune.prereg / make_selection write the teachers as parakeet_ctc / parakeet_tdt; the tables use CONTRACT.md
    5's names (parakeet-ctc, parakeet-tdt). No study system name has an underscore."""
    return name.replace("_", "-")


def _baseline_key(key: str) -> str:
    """A PREREG baseline key -> this module's stratum name: "galgame:<view>" -> "galgame_<view>", a bare "galgame" ->
    "galgame_all"; eval sets and "m4" (checked against the M4 metric) as they are."""
    if key == GALGAME:
        return f"{GALGAME}_all"
    if key.startswith(GALGAME + ":"):
        return f"{GALGAME}_{key.split(':', 1)[1]}"
    return key


def prereg_baselines(prereg: Mapping | None) -> dict:
    """{teacher: {stratum or "m4": corpus CER}} from PREREG.json "baselines" (fractions), names normalised to this
    module's (_baseline_system, _baseline_key), for 05_evaluate's --teachers check. Entries that are not numbers in
    [0, 1] ("pending", a None for an empty set, a note) are left out; a pending block gives {} (the evaluator's
    defaults are then all that is checked)."""
    b = (prereg or {}).get("baselines") or {}
    out = {}
    for sysname, sets in b.items():
        if isinstance(sets, Mapping):
            vals = {_baseline_key(k): float(v) for k, v in sets.items() if _is_num(v) and 0 <= v <= 1}
            if vals:
                out[_baseline_system(sysname)] = vals
    return out


def _sha(v) -> str | None:
    """v if it is a hex sha256 (not "pending", a path or a note), else None; a {"ids_sha256" | "sha256": ...} entry's
    digest."""
    if isinstance(v, Mapping):
        v = v.get("ids_sha256") or v.get("sha256")
    return v if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) else None


def prereg_manifest(prereg: Mapping | None) -> dict:
    """What PREREG.json froze about the eval manifest, for manifest_check:
    {"status", "sets": {eval set: ids sha256}, "views": {Galgame view: ids sha256}, "manifest_sha256",
    "selection_sha256"}. Layouts read: kitsune.prereg's (manifest.status, manifest.ids_sha256.<set>,
    manifest.galgame_views.<view>.ids_sha256, manifest.manifest_sha256, manifest.selection_sha256) and the flat ones
    ({set: sha | {"ids_sha256": sha}} directly under "manifest" or under manifest.sets). Only the eval sets count (train
    and probe are not scored here). status: the block's own, else "filled" when a hash was read, "unreadable" when
    none was, "absent" without a manifest block."""
    m = (prereg or {}).get("manifest")
    out = dict(status="absent", sets={}, views={}, manifest_sha256=None, selection_sha256=None)
    if not isinstance(m, Mapping) or not m:
        return out
    for src in (m.get("ids_sha256"), m.get("sets"), m):
        if isinstance(src, Mapping):
            for s, v in src.items():
                if s in EVAL_SETS and s not in out["sets"] and _sha(v):
                    out["sets"][s] = _sha(v)
    gv = m.get("galgame_views") or m.get("views") or {}
    if isinstance(gv, Mapping):
        out["views"] = {v: _sha(x) for v, x in gv.items() if _sha(x)}
    out["manifest_sha256"], out["selection_sha256"] = _sha(m.get("manifest_sha256")), _sha(m.get("selection_sha256"))
    readable = out["sets"] or out["views"] or out["manifest_sha256"]
    out["status"] = m["status"] if isinstance(m.get("status"), str) else "filled" if readable else "unreadable"
    return out
