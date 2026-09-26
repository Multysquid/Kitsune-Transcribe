"""Evaluation of a student checkpoint: teacher-forced KD metrics, greedy-decode CER and the pre-registered verdict.

Two views of the same model, because they answer different questions:
  teacher_forced_eval   KL / CE / top-1 agreement against the stored teacher top-16, on the teacher's greedy prefix -
                        the quantity training minimises, cheap, smooth (no decode), and comparable between the
                        held-out sets and the train probe (so over-fitting shows up as a widening probe/held-out gap)
  greedy_eval           the real output: greedy decode with the SAME settings as scripts/02_teacher_pass.py
                        (num_beams=1, RepetitionStop, max_new = 16 + 10 s), scored against the dataset reference AND
                        against the teacher's own hypothesis (imitation error, free of reference noise)

CER is reported two ways: the per-utterance mean (kitsune.text.cer, what teacher_out/*.jsonl stores) and the CORPUS
CER = sum of char edits / sum of reference chars over normalize_ja strings, which is what the gate uses - the mean is
dominated by short utterances (one wrong char in a 3-char interjection is 33 %). Empty references are left out of the
corpus sums (they have no chars to divide by) and counted instead; that is also how the pre-registered teacher
baselines were computed (galgame-eval 21.42 % only reproduces that way; the three gate sets have no empty refs).

`store` is a kitsune.trainset.Stores (eval_store(...), or a train store / .subset() for the probe). Batches come from
the trainer's own AudioBatchDataset, so eval sees exactly the training collate: decoder input prompt + tokens[:-1],
target t at position len(prompt)-1+t. The reference and teacher hypothesis per id come from the store's index
(copied from teacher_out/*.jsonl by build_stores); `teacher_rows=load_teacher_rows(...)` overrides them.

The model is put in eval mode for the duration and every module's training flag is restored afterwards (so frozen
BatchNorm stays frozen). Under CUDA the body runs in bf16 autocast; the LM head always runs in fp32 outside autocast,
as in training and in the teacher pass (a bf16 head moves argmax on near-ties).

Two student families (the size study, CONTRACT.md 1): "aed" (Cohere Transcribe students, everything above) and "ctc"
(Parakeet CTC students, kitsune.ctc_student). ctc_eval is the CTC family's one pass over a store: the encoder once per
batch, greedy CTC (argmax, collapse, drop blanks) for the text, and - given the stored Parakeet frame targets - the
teacher-forced frame metrics of kitsune.ctc_kd on the same log-probs. Its greedy per-utterance frame has greedy_eval's
columns (the teacher is Parakeet's stored CTC path: ctc_teacher_rows), so summarise_greedy, headline and the verdict
read both families alike. The own teacher of a family: FAMILY_TEACHER; its pre-registered gate CER:
family_teacher_prereg (Cohere: TEACHER_CER_PREREG; Parakeet CTC: PARAKEET_CTC_CER_PREREG from study/PREREG.json once
the PREREG fill wrote it, None while pending).

The study's per-utterance tables (CONTRACT.md 5; utterance_table): id, set, ref, hyp and, from ONE jiwer alignment of
the normalize_ja strings (the alignment corpus_cer counts), edits, ref_len, hyp_len, sub, del, ins, and the no-style
count edits_nostyle = edits minus the edits of the error regions that only change the script or the numeral style
(STYLE_KINDS; region_kind, the first run's error analysis). Summed over the rows with a reference they are corpus_cer's
edits and reference chars exactly, so tools/study_report.py (kitsune.study_stats) reproduces every corpus CER here.
"""
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import jiwer
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers.generation import StoppingCriteriaList

from kitsune.generation import RepetitionStop
from kitsune.text import cer as utt_cer
from kitsune.text import normalize_ja
from kitsune.trainset import EOS, EVAL_SETS, PAD, PROMPT, AudioBatchDataset, eval_batches

TEACHER_ID = "CohereLabs/cohere-transcribe-03-2026"
GATE_SETS = tuple(EVAL_SETS)  # the D32a gate: JSUT / CV8 / Reazon-test; eval_emilia and galgame are monitor-only
# D32a, pre-registered full-set teacher corpus CER (fractions). teacher_baselines() recomputes them from teacher_out
# and refuses to proceed if they drift by more than 0.05 pp - a changed eval set would silently move every threshold.
TEACHER_CER_PREREG = {"eval_jsut": 0.0830, "eval_cv8": 0.0407, "eval_reazon": 0.0628}
BASELINE_TOL = 0.0005
FAMILIES = ("aed", "ctc")
FAMILY_TEACHER = {"aed": "cohere", "ctc": "parakeet-ctc"}  # a family's own teacher, by its CONTRACT.md 5 system name
PREREG_JSON = Path(__file__).resolve().parents[1] / "study" / "PREREG.json"


# ----------------------------------------------------------------------------------------------------------- CER


def corpus_cer(hyps: Sequence[str], refs: Sequence[str]) -> dict:
    """Corpus CER on normalize_ja strings: sum(S+D+I) / sum(ref chars), empty references skipped and counted."""
    pairs = [(normalize_ja(h or ""), normalize_ja(r or "")) for h, r in zip(hyps, refs)]
    kept = [(h, r) for h, r in pairs if r]
    out = dict(cer=float("nan"), edits=0, ref_chars=0, n=len(kept), n_empty_ref=len(pairs) - len(kept))
    if kept:
        o = jiwer.process_characters([r for _, r in kept], [h for h, _ in kept])
        out["edits"] = int(o.substitutions + o.deletions + o.insertions)
        out["ref_chars"] = int(o.substitutions + o.deletions + o.hits)
        out["cer"] = out["edits"] / max(out["ref_chars"], 1)
    return out


# ------------------------------------------------------------------------------------------------ teacher data


def load_teacher_rows(teacher_root, sources: Iterable[str], split: str | None = None) -> dict[str, dict]:
    """id -> teacher_out jsonl row (hyp, ref, cer, duration, n_tok, truncated) plus its source."""
    rows = {}
    for src in sources:
        for f in sorted((Path(teacher_root) / src).glob(f"{split}-*.jsonl" if split else "*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        r["source"] = src
                        rows[r["id"]] = r
    return rows


def teacher_baselines(teacher_root, sets: Iterable[str] = GATE_SETS, check: bool = True) -> dict[str, dict]:
    """Full-set teacher corpus/mean CER and truncation per eval set, recomputed from teacher_out/<set>/eval-*.jsonl.

    With check=True a pre-registered set whose recomputed corpus CER is off by more than 0.05 pp raises ValueError:
    call this at trainer start-up, not at the end of a 4 h run."""
    out = {}
    for s in sets:
        rows = list(load_teacher_rows(teacher_root, [s], split="eval").values())
        if not rows:
            raise FileNotFoundError(f"no teacher_out/{s}/eval-*.jsonl under {teacher_root}")
        c = corpus_cer([r["hyp"] for r in rows], [r["ref"] for r in rows])
        out[s] = dict(cer_corpus=c["cer"], cer_mean=float(np.mean([r["cer"] for r in rows])), n=len(rows),
                      n_empty_ref=c["n_empty_ref"], trunc_rate=float(np.mean([bool(r["truncated"]) for r in rows])),
                      hours=float(sum(r["duration"] for r in rows)) / 3600, prereg=TEACHER_CER_PREREG.get(s))
        if check and s in TEACHER_CER_PREREG and abs(c["cer"] - TEACHER_CER_PREREG[s]) > BASELINE_TOL:
            raise ValueError(f"{s}: teacher corpus CER {100 * c['cer']:.3f} % differs from the pre-registered "
                             f"{100 * TEACHER_CER_PREREG[s]:.2f} % by more than {100 * BASELINE_TOL:.2f} pp")
    return out


def _cer_value(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1


def prereg_teacher_cer(system: str, prereg=None) -> dict[str, float] | None:
    """A teacher's pre-registered corpus CER per GATE set from the size study's PREREG.json "baselines" block, which
    the PREREG fill (python -m kitsune.prereg --write study/ --sidecar ...) copies from the study selection's sidecar:
    the teacher's stored hypotheses scored on the manifest rows, as fractions. None while the block is pending, and
    when the file is missing or unreadable or lacks a number for a gate set: nothing is registered then, and nothing
    can be checked against it. `prereg`: the parsed PREREG.json or a path to one (default: the committed
    study/PREREG.json). `system` by its CONTRACT.md 5 name (parakeet-ctc; the parakeet_ctc spelling is read too)."""
    if prereg is None or isinstance(prereg, (str, Path)):
        try:
            prereg = json.loads(Path(prereg or PREREG_JSON).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    b = (prereg or {}).get("baselines") if isinstance(prereg, dict) else None
    if not isinstance(b, dict) or b.get("status") == "pending":
        return None
    row = b.get(system) or b.get(system.replace("-", "_"))
    if not isinstance(row, dict) or not all(_cer_value(row.get(s)) for s in GATE_SETS):
        return None
    return {s: float(row[s]) for s in GATE_SETS}


# Parakeet's CTC path, the teacher and ratio baseline of the P students (STUDY.md 4.1): its corpus CER on the manifest's
# gate sets as study/PREREG.json registers it. None until the PREREG fill writes the baselines ("pending").
PARAKEET_CTC_CER_PREREG = prereg_teacher_cer("parakeet-ctc")


def family_teacher_prereg(family: str = "aed") -> dict[str, float] | None:
    """The pre-registered gate CER of a family's own teacher: Cohere's D32a numbers for "aed" (TEACHER_CER_PREREG),
    PARAKEET_CTC_CER_PREREG for "ctc" (None while the study's PREREG baselines are pending)."""
    if family == "aed":
        return dict(TEACHER_CER_PREREG)
    if family == "ctc":
        return dict(PARAKEET_CTC_CER_PREREG) if PARAKEET_CTC_CER_PREREG else None
    raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")


def ctc_teacher_rows(parakeet_rows: dict[str, dict], refs: dict[str, str] | None = None) -> dict[str, dict]:
    """Parakeet's stored CTC path as the teacher of a CTC student's greedy eval (greedy_eval / ctc_eval teacher_rows):
    hyp = the jsonl's ctc_hyp, cer = its ctc_cer, never truncated (CTC has no length cut). `refs` (id -> reference)
    replaces the jsonl's reference, so a CTC student is scored on the very text every other system is (the eval store's
    teacher_out reference; the caller checks the two agree)."""
    out = {}
    for i, r in parakeet_rows.items():
        out[i] = dict(ref=refs[i] if refs is not None and i in refs else r["ref"], hyp=r["ctc_hyp"],
                      cer=float(r["ctc_cer"]), truncated=False)
    return out


def parakeet_baselines(parakeet_root, sets: Iterable[str] = GATE_SETS, *, ids: Iterable[str] | None = None,
                       prereg: dict | None = None, check: bool = True) -> dict[str, dict]:
    """Parakeet's stored eval hypotheses per set from parakeet_out/<set>/eval-*.jsonl: the corpus CER of its CTC path
    (ctc_hyp: the CTC students' own teacher, `cer_corpus`) and of its TDT path (hyp: the off-the-shelf bar,
    `tdt_cer_corpus`), on every row or on `ids` (the study manifest's rows, on which PREREG registered them). prereg:
    {set: CER} of the CTC path (default PARAKEET_CTC_CER_PREREG; None or missing = pending: nothing checked). With
    check=True a registered set whose recomputed CTC corpus CER is off by more than 0.05 pp raises ValueError, as
    teacher_baselines does for Cohere: a changed label set would silently move every ratio."""
    prereg = PARAKEET_CTC_CER_PREREG if prereg is None else prereg
    keep = None if ids is None else set(ids)
    out = {}
    for s in sets:
        rows = [r for r in load_teacher_rows(parakeet_root, [s], split="eval").values()
                if keep is None or r["id"] in keep]
        if not rows:
            raise FileNotFoundError(f"no parakeet_out/{s}/eval-*.jsonl rows under {parakeet_root}")
        ctc = corpus_cer([r["ctc_hyp"] for r in rows], [r["ref"] for r in rows])
        tdt = corpus_cer([r["hyp"] for r in rows], [r["ref"] for r in rows])
        p = (prereg or {}).get(s)
        out[s] = dict(cer_corpus=ctc["cer"], tdt_cer_corpus=tdt["cer"],
                      cer_mean=float(np.mean([r["ctc_cer"] for r in rows])), n=len(rows),
                      n_empty_ref=ctc["n_empty_ref"], prereg=p,
                      tdt_trunc_rate=float(np.mean([bool(r.get("truncated")) for r in rows])))
        if check and p is not None and abs(ctc["cer"] - p) > BASELINE_TOL:
            raise ValueError(f"{s}: Parakeet CTC corpus CER {100 * ctc['cer']:.3f} % differs from the pre-registered "
                             f"{100 * p:.2f} % by more than {100 * BASELINE_TOL:.2f} pp")
    return out


@lru_cache(maxsize=2)
def teacher_tokenizer(name: str = TEACHER_ID):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


# --------------------------------------------------------------------------------------------------- batching


def _indices(store, ids: Iterable[str] | None) -> list[int] | None:
    if ids is None:
        return None
    pos = {u.id: i for i, u in enumerate(store.utts)}
    ids = list(ids)
    missing = [x for x in ids if x not in pos]
    if missing:
        raise KeyError(f"{len(missing)} ids not in the store, e.g. {missing[:3]}")
    return [pos[x] for x in ids]


def _prefetched(ds: AudioBatchDataset, batches: list[list[int]], threads: int = 4, ahead: int = 3):
    """Yield collated micro-batches; decode/resample of the next ones overlaps the GPU work (soundfile and soxr drop
    the GIL). An undecodable utterance is dropped by the dataset and listed in item["dropped"]."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(ds.__getitem__, b) for b in batches[:ahead]]
        for j in range(len(batches)):
            item = futs[j].result()
            futs[j] = None
            if j + ahead < len(batches):
                futs.append(pool.submit(ds.__getitem__, batches[j + ahead]))
            yield item


def _bad_audio_per_set(store, dropped: list[str]) -> dict[str, int]:
    """The undecodable utterances of an eval per source (set): the summaries' bad_audio_per_set, which the verdict
    reports for a gate set judged on fewer rows than its full size. Only sets that lost a row appear."""
    if not dropped:
        return {}
    src = {u.id: u.source for u in store.utts}
    return dict(sorted(Counter(src.get(i, "?") for i in dropped).items()))


def _features(featurizer, item: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    wave = item["wave"].to(device, non_blocking=True)
    lengths = item["lengths"].to(device)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        return featurizer(wave, lengths)


# ------------------------------------------------------------------------------------------- model mode helpers


@contextmanager
def _eval_mode(model):
    """Eval mode for the block, then every module's own training flag back (keeps frozen BN frozen)."""
    flags = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for m, t in flags:
            m.training = t


class _FP32Head(torch.nn.Module):
    """Wraps proj_out so generate() evaluates it in fp32 outside autocast; the wrapped Linear (and its tie to the
    token embedding) is untouched."""

    def __init__(self, inner: torch.nn.Linear):
        super().__init__()
        self.inner = inner

    def forward(self, h):
        with torch.autocast(device_type=h.device.type, enabled=False):
            b = self.inner.bias.float() if self.inner.bias is not None else None
            return F.linear(h.float(), self.inner.weight.float(), b)


@contextmanager
def _fp32_head(model):
    inner = model.proj_out
    model.proj_out = _FP32Head(inner)
    try:
        yield
    finally:
        model.proj_out = inner


def _amp(device, amp: bool | None) -> bool:
    return torch.device(device).type == "cuda" if amp is None else amp


# ------------------------------------------------------------------------------------------ teacher-forced eval


def _default_losses():
    from kitsune.kd import kd_losses

    return kd_losses


@torch.no_grad()
def teacher_forced_eval(model, store, featurizer, device, batch_s: float = 400.0, *, ids: Iterable[str] | None = None,
                        losses_fn: Callable | None = None, amp: bool | None = None) -> tuple[dict, pd.DataFrame]:
    """KL / CE / top-1 of the student against the stored teacher top-k, teacher-forced on the teacher's tokens.

    ids restricts to a subset (e.g. the train probe); default is every utterance in the store. losses_fn defaults to
    kitsune.kd.kd_losses, so eval and training use the same 17-bin KL. Returns (summary, per_utt): per_utt columns
    id, source, n_tok, kl, ce, top1, duration and the per-utterance mean of every other kd_losses term;
    summary["sets"][source] and summary["all"] hold token-normalised (corpus) and utterance-mean values, plus the
    same for the teacher-confidence buckets p1 > 0.99 and p1 < 0.9."""
    t0 = time.time()
    raw, dropped = teacher_forced_records(model, store, featurizer, device, batch_s, ids=ids, losses_fn=losses_fn,
                                          amp=amp)
    return summarise_tf(raw, n_bad_audio=len(dropped), bad_audio=dropped[:50],
                        bad_audio_per_set=_bad_audio_per_set(store, dropped), wall_s=time.time() - t0)


@torch.no_grad()
def teacher_forced_records(model, store, featurizer, device, batch_s: float = 400.0, *,
                           ids: Iterable[str] | None = None, losses_fn: Callable | None = None,
                           amp: bool | None = None) -> tuple[pd.DataFrame, list[str]]:
    """teacher_forced_eval's pass without the summary: (raw, dropped). raw has one row per utterance in batch order
    (id, source, n_tok, duration, sum_<term> of every kd_losses term over its tokens, and the teacher-confidence
    buckets' token counts and sums _n_hi / _<term>_hi, _n_lo / _<term>_lo); dropped lists the ids whose audio could
    not be decoded, in batch order. summarise_tf turns raw into teacher_forced_eval's (summary, per_utt), for any rows
    of it: scripts/05_evaluate.py evaluates a checkpoint in resumable chunks of the same batches and summarises them
    together."""
    losses_fn = losses_fn or _default_losses()
    device = torch.device(device)
    ds = AudioBatchDataset(store)
    batches = eval_batches(store.utts, batch_s, _indices(store, ids))
    W, B = model.proj_out.weight, model.proj_out.bias
    recs, dropped = [], []
    with _eval_mode(model):
        for item in _prefetched(ds, batches):
            dropped += item["dropped"]
            n = len(item["ids"])
            if not n:
                continue
            feats, fmask = _features(featurizer, item, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=_amp(device, amp)):
                h = model.model(input_features=feats, attention_mask=fmask,
                                decoder_input_ids=item["decoder_input_ids"].to(device),
                                decoder_attention_mask=item["dec_mask"].to(device), use_cache=False).last_hidden_state
            rows = item["tgt_row"].to(device)
            logits = F.linear(h[rows, item["tgt_pos"].to(device)].float(), W.float(), B.float() if B is not None else None)
            top_lp = item["top_lp"].to(device)
            terms = {k: v.detach().float() for k, v in losses_fn(logits, item["top_idx"].to(device), top_lp).items()}
            terms.setdefault("teacher_p1", top_lp[:, 0].exp())  # teacher confidence for the p1 buckets

            def per_row(v):
                return torch.zeros(n, device=device).index_add_(0, rows, v).cpu().numpy()

            sums = {k: per_row(v) for k, v in terms.items()}
            buckets = {}
            for name, m in (("hi", terms["teacher_p1"] > 0.99), ("lo", terms["teacher_p1"] < 0.9)):
                buckets[f"_n_{name}"] = per_row(m.float())
                for k in ("kl", "ce", "top1_match"):
                    buckets[f"_{k}_{name}"] = per_row(terms[k] * m)
            for r in range(n):
                rec = dict(id=item["ids"][r], source=item["sources"][r], n_tok=int(item["n_tok"][r]),
                           duration=float(item["durations"][r]))
                rec.update({f"sum_{k}": float(v[r]) for k, v in sums.items()})
                rec.update({k: float(v[r]) for k, v in buckets.items()})
                recs.append(rec)
    return pd.DataFrame(recs), dropped


def summarise_tf(raw: pd.DataFrame, **extra) -> tuple[dict, pd.DataFrame]:
    """teacher_forced_eval's (summary, per_utt) for any rows of teacher_forced_records' raw frame (a default
    RangeIndex; the rows in batch order give the trainer's numbers to the bit). `extra` goes into the summary after
    n_utts (teacher_forced_eval: n_bad_audio, bad_audio, bad_audio_per_set, wall_s)."""
    per_utt = _tf_per_utt(raw)
    summary = dict(sets={}, n_utts=len(raw), **extra)
    if len(raw):
        for src, g in raw.groupby("source", sort=True):
            summary["sets"][src] = _tf_summarise(g, per_utt.loc[g.index])
        summary["all"] = _tf_summarise(raw, per_utt)
    return summary, per_utt


def _tf_per_utt(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return pd.DataFrame(columns=["id", "source", "n_tok", "kl", "ce", "top1", "duration"])
    out = df[["id", "source", "n_tok"]].copy()
    n = df["n_tok"].to_numpy(dtype=np.float64)
    out["kl"] = df["sum_kl"] / n
    out["ce"] = df["sum_ce"] / n
    out["top1"] = df["sum_top1_match"] / n
    out["duration"] = df["duration"]
    for c in df.columns:
        if c.startswith("sum_") and c[4:] not in ("kl", "ce", "top1_match"):
            out[c[4:]] = df[c] / n
    return out


def _tf_summarise(df: pd.DataFrame, per_utt: pd.DataFrame) -> dict:
    ntok = float(df["n_tok"].sum())
    s = dict(n_utts=int(len(df)), n_tok=int(ntok), audio_s=float(df["duration"].sum()))
    for c in df.columns:
        if c.startswith("sum_"):
            s["top1" if c == "sum_top1_match" else c[4:]] = float(df[c].sum()) / ntok  # every token weighs the same
    for k in ("kl", "ce", "top1"):
        s[f"{k}_utt_mean"] = float(per_utt[k].mean())
    for name, label in (("hi", "p1_gt_0.99"), ("lo", "p1_lt_0.9")):
        nb = float(df[f"_n_{name}"].sum())
        s[f"frac_{label}"] = nb / ntok
        for k, out in (("kl", "kl"), ("ce", "ce"), ("top1_match", "top1")):
            s[f"{out}_{label}"] = float(df[f"_{k}_{name}"].sum()) / nb if nb else float("nan")
    return s


# ------------------------------------------------------------------------------------------------ greedy eval


def _teacher_text(store, teacher_rows: dict | None) -> dict[str, tuple]:
    """id -> (ref, teacher hyp, teacher cer, teacher truncated)."""
    if teacher_rows is not None:
        return {i: (r["ref"], r["hyp"], float(r["cer"]), bool(r["truncated"])) for i, r in teacher_rows.items()}
    fr = store.frame()
    return {u.id: (ref, hyp, float(u.teacher_cer), bool(u.truncated))
            for u, ref, hyp in zip(store.utts, fr["ref"].tolist(), fr["hyp"].tolist())}


@torch.no_grad()
def greedy_generate(model, feats: torch.Tensor, fmask: torch.Tensor, max_duration_s: float, *,
                    prompt_ids: Sequence[int], eos: int, pad: int, amp: bool) -> list[tuple[list[int], bool]]:
    """One batch of an AED model's greedy decode with the teacher pass's settings (scripts/02_teacher_pass.py:
    num_beams 1, RepetitionStop, max_new = 16 + 10 x the longest row's seconds): per row the generated ids after the
    prompt up to and including EOS, and whether the row ended with EOS (not truncated). The caller holds eval mode and
    the fp32 head (greedy_eval; tools/speed_probe.py times this very call)."""
    P = len(prompt_ids)
    n = feats.shape[0]
    # identical to 02_teacher_pass: ~3.6 tok/s observed (max ~6.5), 16 + 10 s is a 1.5x margin on the max
    max_new = min(int(16 + 10 * float(max_duration_s)), model.config.max_position_embeddings - P - 1)
    prompt = torch.tensor([list(prompt_ids)] * n, dtype=torch.long, device=feats.device)
    with torch.autocast(device_type=feats.device.type, dtype=torch.bfloat16, enabled=amp):
        seq = model.generate(input_features=feats, attention_mask=fmask, decoder_input_ids=prompt,
                             max_new_tokens=max_new, do_sample=False, num_beams=1, eos_token_id=eos,
                             pad_token_id=pad, stopping_criteria=StoppingCriteriaList([RepetitionStop(P)]))
    gen = (seq.sequences if hasattr(seq, "sequences") else seq)[:, P:].cpu().numpy()
    out = []
    for r in range(n):
        row = gen[r]
        stop = np.flatnonzero((row == eos) | (row == pad))  # generate pads a row after EOS / RepetitionStop
        ended = bool(len(stop) and row[stop[0]] == eos)
        k = int(stop[0]) + int(ended) if len(stop) else len(row)  # a max-length row keeps all its tokens
        out.append((row[:k].tolist(), ended))
    return out


@torch.no_grad()
def greedy_eval(model, store, ids: Iterable[str] | None, featurizer, device, batch_s: float = 400.0, *,
                tokenizer=None, teacher_rows: dict[str, dict] | None = None,
                amp: bool | None = None) -> tuple[dict, pd.DataFrame]:
    """Greedy decode of `ids` (None = the whole store) with the teacher-pass settings, scored vs the reference and vs
    the teacher hypothesis; the teacher baseline is computed on exactly the same ids.

    per_utt columns: id, source, duration, ref, teacher_hyp, hyp, cer_ref, cer_teacher, truncated, n_tok, plus
    teacher_cer, teacher_truncated, hyp_ids (generated ids incl. EOS) and has_teacher."""
    tokenizer = tokenizer or teacher_tokenizer()
    device = torch.device(device)
    t0 = time.time()
    prompt_ids = [int(x) for x in store.info.get("prompt", PROMPT)]
    eos, pad = int(store.info.get("eos", EOS)), int(store.info.get("pad", PAD))
    ds = AudioBatchDataset(store)
    batches = eval_batches(store.utts, batch_s, _indices(store, ids))
    recs, dropped = [], []
    with _eval_mode(model), _fp32_head(model):
        for item in _prefetched(ds, batches):
            dropped += item["dropped"]
            n = len(item["ids"])
            if not n:
                continue
            feats, fmask = _features(featurizer, item, device)
            rows = greedy_generate(model, feats, fmask, float(item["durations"].max()), prompt_ids=prompt_ids,
                                   eos=eos, pad=pad, amp=_amp(device, amp))
            for r, (hyp_ids, ended) in enumerate(rows):
                recs.append(dict(id=item["ids"][r], source=item["sources"][r], duration=float(item["durations"][r]),
                                 hyp_ids=hyp_ids, truncated=not ended, n_tok=len(hyp_ids)))
    per_utt = _greedy_frame(recs, store, teacher_rows, lambda b: tokenizer.batch_decode(b, skip_special_tokens=True))
    wall = time.time() - t0
    audio_s = float(per_utt["duration"].sum()) if len(per_utt) else 0.0
    return summarise_greedy(per_utt, n_bad_audio=len(dropped), bad_audio=dropped[:50],
                            bad_audio_per_set=_bad_audio_per_set(store, dropped), wall_s=wall,
                            rtf=wall / audio_s if audio_s else float("nan")), per_utt


def summarise_greedy(per_utt: pd.DataFrame, **extra) -> dict:
    """greedy_eval's summary for any set of its per_utt rows. The trainer's final eval decodes the full eval sets once
    and takes the fixed greedy subset's summary from those rows, so the last history point stays comparable with the
    earlier subset evals without a second decode."""
    summary = dict(sets={}, n_utts=len(per_utt), **extra)
    if len(per_utt):
        for src, g in per_utt.groupby("source", sort=True):
            summary["sets"][src] = _greedy_summarise(g)
        summary["all"] = _greedy_summarise(per_utt)
    return summary


def _greedy_summarise(g: pd.DataFrame) -> dict:
    ref = corpus_cer(g["hyp"].tolist(), g["ref"].tolist())
    imit = corpus_cer(g["hyp"].tolist(), g["teacher_hyp"].tolist())
    t = g[g["has_teacher"].astype(bool)]
    teach = corpus_cer(t["teacher_hyp"].tolist(), t["ref"].tolist())
    s = dict(n=int(len(g)), audio_s=float(g["duration"].sum()),
             cer_ref_corpus=ref["cer"], cer_ref_mean=float(g["cer_ref"].mean()),
             cer_teacher_corpus=imit["cer"], cer_teacher_mean=float(g["cer_teacher"].mean()),
             n_truncated=int(g["truncated"].sum()), trunc_rate=float(g["truncated"].mean()),
             n_empty_hyp=int((g["hyp"].map(normalize_ja) == "").sum()),
             tok_per_s=float(g["n_tok"].sum()) / max(float(g["duration"].sum()), 1e-9),
             ref_edits=ref["edits"], ref_chars=ref["ref_chars"], n_empty_ref=ref["n_empty_ref"],
             cer_teacher_edits=imit["edits"], cer_teacher_chars=imit["ref_chars"],  # cer_teacher_corpus = their ratio
             teacher_cer_ref_corpus=teach["cer"], teacher_cer_ref_mean=float(t["teacher_cer"].mean()),
             teacher_trunc_rate=float(t["teacher_truncated"].astype(bool).mean()) if len(t) else float("nan"),
             n_missing_teacher=int(len(g) - len(t)))
    tc = s["teacher_cer_ref_corpus"]
    s["ratio_vs_teacher"] = s["cer_ref_corpus"] / tc if tc and not math.isnan(tc) else float("nan")
    return s


def pick_samples(per_utt: pd.DataFrame, n: int = 8, seed: int = 0) -> list[dict]:
    """Text samples for TensorBoard / samples/*.jsonl: the worst half by CER vs teacher, the rest random (fixed seed,
    so the same ids recur across evals while they stay bad)."""
    if not len(per_utt):
        return []
    worst = per_utt.sort_values("cer_teacher", ascending=False, kind="stable").head(n // 2)
    rest = per_utt.drop(worst.index)
    pick = pd.concat([worst, rest.sample(min(n - len(worst), len(rest)), random_state=seed)])
    keep = ["id", "source", "duration", "ref", "teacher_hyp", "hyp", "cer_ref", "cer_teacher", "truncated"]
    return pick[keep].to_dict("records")


# ------------------------------------------------------------------------------------------------- CTC family

GREEDY_COLUMNS = ("id", "source", "duration", "ref", "teacher_hyp", "hyp", "cer_ref", "cer_teacher", "truncated",
                  "n_tok", "teacher_cer", "teacher_truncated", "hyp_ids", "has_teacher")
# the per-utterance sums of ctc_eval's teacher-forced part (kitsune.ctc_kd.ctc_kd_losses per_utt), summarise_ctc_tf's
# input; n_tok = the utterance's CTC target tokens (the objective's N_u)
CTC_TF_SUMS = ("kl_dense", "kl_blank", "ctc", "n_dense", "n_blank_frames", "n_frames", "argmax_agree",
               "argmax_blank", "teacher_blank")
# the frame-target fields of a frame store's micro-batch (kitsune.trainset.FrameBatchDataset, the CTC trainer's
# collate: kitsune.ctc_targets.collate_frame_targets of its rows), what kitsune.ctc_kd.ctc_kd_losses reads
FRAME_BATCH_KEYS = ("frame_mask", "dense_mask", "blank_lp", "topk_idx", "topk_lp", "ctc_targets", "ctc_target_lengths",
                    "n_frames")


def batch_dataset(store):
    """The micro-batch dataset of a store as the trainer reads it: kitsune.trainset.dataset_for (the CTC trainer's: a
    FrameBatchDataset for a frame store, its rows' stored Parakeet frame targets collated into every batch, else an
    AudioBatchDataset) where kitsune.trainset has it, else an AudioBatchDataset (a token store: the only kind before
    the CTC trainer's frame stores). ctc_eval reads either."""
    from kitsune import trainset

    make = getattr(trainset, "dataset_for", None)
    return make(store) if make is not None else AudioBatchDataset(store)


def _greedy_frame(recs: list[dict], store, teacher_rows: dict | None, decode) -> pd.DataFrame:
    """greedy_eval's per-utterance frame from records with hyp_ids (decode: ids list -> texts)."""
    text = _teacher_text(store, teacher_rows)
    hyps = decode([r["hyp_ids"] for r in recs]) if recs else []
    for r, h in zip(recs, hyps):
        t = text.get(r["id"])
        r.update(hyp=h, ref=t[0] if t else "", teacher_hyp=t[1] if t else "", teacher_cer=t[2] if t else float("nan"),
                 teacher_truncated=t[3] if t else None, has_teacher=t is not None)
        r["cer_ref"] = utt_cer(h, r["ref"] or "")
        r["cer_teacher"] = utt_cer(h, r["teacher_hyp"] or "")
    return pd.DataFrame(recs, columns=list(GREEDY_COLUMNS))


@torch.no_grad()
def ctc_eval(model, store, ids: Iterable[str] | None, featurizer, device, batch_s: float = 400.0, *, tokenizer,
             teacher_rows: dict[str, dict] | None = None, targets: dict | None = None,
             amp: bool | None = None) -> dict:
    """A CTC student (transformers ParakeetForCTC) over `ids` of a store (None = all), in the trainer's eval batches:
    per batch the encoder once (bf16 autocast under CUDA; the CTC head in fp32, kitsune.ctc_student.ctc_log_probs),
    then
      greedy    argmax per frame, collapse repeats, drop the blank (greedy_ctc_ids), detokenised as the label pass
                wrote ctc_hyp (decode_ids): scored against the reference and the teacher (teacher_rows: Parakeet's
                stored CTC path, ctc_teacher_rows) exactly as greedy_eval scores an AED student; never truncated
      tf        the training objective's terms on the same log-probs against the stored Parakeet frame targets
                (kitsune.ctc_kd.ctc_kd_losses per utterance): dense and blank-frame KL, CTC on the teacher's greedy
                path, argmax agreement and blank shares. The targets come from
                  a frame store (the CTC trainer's: batch_dataset gives its FrameBatchDataset, whose batches carry their
                    rows' targets as the trainer's do): every row of a batch; the store's frame preflight dropped each
                    row whose frames do not align at its build (a hard fail for an eval row, decision 15), and the
                    dataset drops one whose audio changed since (item["dropped"], like an undecodable one)
                  a token store with `targets` (id -> kitsune.ctc_targets.FrameTargets, collated here): a row whose
                    student frame count differs from its stored n_frames cannot be aligned: it is left out of tf and
                    listed in frame_mismatch (the preflight's condition; the caller applies decision 15); a row without
                    targets in no_targets
                  a token store without `targets`: no tf
    featurizer: (wave, lengths) -> (feats, mask) on the device, the student's LogMel (ctc_features(...).logmel).
    Returns {"greedy": greedy_eval's per_utt columns, "tf": one row per utterance with id, source, duration, n_tok
    (its CTC target tokens) and sum_<term> of CTC_TF_SUMS (summarise_ctc_tf's input), "dropped": undecodable ids,
    "frame_mismatch": [{id, student, stored}], "no_targets": ids, "wall_s"}."""
    from kitsune import ctc_student as CS
    from kitsune.ctc_kd import ctc_kd_losses
    from kitsune.ctc_targets import collate_frame_targets

    device = torch.device(device)
    t0 = time.time()
    ds = batch_dataset(store)
    batches = eval_batches(store.utts, batch_s, _indices(store, ids))
    g_recs, tf_recs, dropped, mismatch, no_targets = [], [], [], [], []
    with _eval_mode(model):
        for item in _prefetched(ds, batches):
            dropped += item["dropped"]
            n = len(item["ids"])
            if not n:
                continue
            feats, fmask = _features(featurizer, item, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=_amp(device, amp)):
                lp, n_frames = CS.ctc_log_probs(model, feats, fmask)
            hyp_ids = CS.greedy_ctc_ids(lp, n_frames)
            frames = n_frames.cpu().tolist()
            for r in range(n):
                g_recs.append(dict(id=item["ids"][r], source=item["sources"][r], duration=float(item["durations"][r]),
                                   hyp_ids=hyp_ids[r], truncated=False, n_tok=len(hyp_ids[r])))
            if "frame_mask" in item:  # a frame store's batch: its targets as the trainer's step and eval read them
                ok = list(range(n))
                L = ctc_kd_losses(lp, {k: item[k] for k in FRAME_BATCH_KEYS}, per_utt=True)
            else:
                if targets is None:
                    continue
                ok = []
                for r, uid in enumerate(item["ids"]):
                    t = targets.get(uid)
                    if t is None:
                        no_targets.append(uid)
                    elif int(t.n_frames) != int(frames[r]):
                        mismatch.append(dict(id=uid, student=int(frames[r]), stored=int(t.n_frames)))
                    else:
                        ok.append(r)
                if not ok:
                    continue
                batch = collate_frame_targets([targets[item["ids"][r]] for r in ok], t_max=int(lp.shape[1]))
                L = ctc_kd_losses(lp[torch.tensor(ok, device=lp.device)], batch, per_utt=True)
            L = {k: v.detach().double().cpu().numpy() for k, v in L.items()}
            for j, r in enumerate(ok):
                rec = dict(id=item["ids"][r], source=item["sources"][r], n_tok=int(L["n_tokens"][j]),
                           duration=float(item["durations"][r]))
                rec.update({f"sum_{k}": float(L[k][j]) for k in CTC_TF_SUMS})
                tf_recs.append(rec)
    greedy = _greedy_frame(g_recs, store, teacher_rows, lambda b: [CS.decode_ids(tokenizer, x) for x in b])
    tf_cols = ["id", "source", "n_tok", "duration", *(f"sum_{k}" for k in CTC_TF_SUMS)]
    return dict(greedy=greedy, tf=pd.DataFrame(tf_recs, columns=tf_cols), dropped=dropped, frame_mismatch=mismatch,
                no_targets=no_targets, wall_s=time.time() - t0)


def summarise_ctc_tf(raw: pd.DataFrame, **extra) -> tuple[dict, pd.DataFrame]:
    """ctc_eval's teacher-forced summary for any rows of its tf frame: per set ("sets") and pooled ("all"), sums over
    the utterances divided by what each measure counts (STUDY.md 4.2's CTC teacher-forced metrics):
      kl            frame KL (dense + blank frames) per CTC target token, and ce = the CTC loss per target token:
                    the training objective's two terms with its normalisation (kitsune.ctc_kd.ctc_kd_objective divides
                    by the step's target tokens), so combined_loss(summary, w_kl, w_ctc) is the CTC objective and the
                    headline's val_loss / eval_record's heldout_kl are the frame KL per target token (ctc: the same
                    number as ce, by its own name)
      top1          frame argmax agreement with the teacher's argmax (col0), per valid frame (also argmax_agree)
      kl_per_frame, kl_dense (per dense frame), kl_blank (per blank-only frame)
      argmax_blank  the student's argmax-blank share of the frames (the blank-collapse watch), teacher_blank the
                    teacher's (59-72 % on the real sets), frac_dense the teacher's dense frames, frames_per_token
      n_utts, n_tok (target tokens), n_frames, audio_s, and the utterance means kl_utt_mean, ce_utt_mean, top1_utt_mean
    per_utt: id, source, n_tok, n_frames, kl, ce, top1, duration, kl_dense, kl_blank, kl_per_frame, argmax_blank,
    teacher_blank (per utterance, with the same normalisations; NaN where a count is 0). The keys, the columns and the
    order of every sum are the CTC trainer's (kitsune.ctc_eval.summarise_ctc_tf, which the trainer's CTC evals write):
    the same rows give the same numbers to the bit. `extra` goes into the summary after n_utts."""
    per_utt = _ctc_tf_per_utt(raw)
    summary = dict(sets={}, n_utts=len(raw), **extra)
    if len(raw):
        for src, g in raw.groupby("source", sort=True):
            summary["sets"][src] = _ctc_tf_summarise(g, per_utt.loc[g.index])
        summary["all"] = _ctc_tf_summarise(raw, per_utt)
    return summary, per_utt


def _ratio(num, den):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


CTC_TF_COLUMNS = ("id", "source", "n_tok", "n_frames", "kl", "ce", "top1", "duration", "kl_dense", "kl_blank",
                  "kl_per_frame", "argmax_blank", "teacher_blank")  # tf_<set>.parquet of a CTC eval


def _ctc_tf_per_utt(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(CTC_TF_COLUMNS)
    if not len(df):
        return pd.DataFrame(columns=cols)
    out = df[["id", "source", "n_tok"]].copy()
    out["n_frames"] = df["sum_n_frames"].astype(np.int64)
    kl = df["sum_kl_dense"] + df["sum_kl_blank"]  # each row's frame KL, then any sum over rows (the trainer's sum_kl)
    out["kl"] = _ratio(kl, df["n_tok"])
    out["ce"] = _ratio(df["sum_ctc"], df["n_tok"])
    out["top1"] = _ratio(df["sum_argmax_agree"], df["sum_n_frames"])
    out["duration"] = df["duration"]
    out["kl_dense"] = _ratio(df["sum_kl_dense"], df["sum_n_dense"])
    out["kl_blank"] = _ratio(df["sum_kl_blank"], df["sum_n_blank_frames"])
    out["kl_per_frame"] = _ratio(kl, df["sum_n_frames"])
    out["argmax_blank"] = _ratio(df["sum_argmax_blank"], df["sum_n_frames"])
    out["teacher_blank"] = _ratio(df["sum_teacher_blank"], df["sum_n_frames"])
    return out[cols]


def _ctc_tf_summarise(df: pd.DataFrame, per_utt: pd.DataFrame) -> dict:
    tot = {k: float(df[f"sum_{k}"].sum()) for k in CTC_TF_SUMS}
    ntok, nfr = float(df["n_tok"].sum()), tot["n_frames"]
    kl = float((df["sum_kl_dense"] + df["sum_kl_blank"]).sum())  # per row first, as the trainer's sum_kl

    def div(a, b):
        return a / b if b else float("nan")

    s = dict(n_utts=int(len(df)), n_tok=int(ntok), n_frames=int(nfr), audio_s=float(df["duration"].sum()),
             kl=div(kl, ntok), ce=div(tot["ctc"], ntok), ctc=div(tot["ctc"], ntok), top1=div(tot["argmax_agree"], nfr),
             argmax_agree=div(tot["argmax_agree"], nfr), kl_per_frame=div(kl, nfr),
             kl_dense=div(tot["kl_dense"], tot["n_dense"]), kl_blank=div(tot["kl_blank"], tot["n_blank_frames"]),
             argmax_blank=div(tot["argmax_blank"], nfr), teacher_blank=div(tot["teacher_blank"], nfr),
             frac_dense=div(tot["n_dense"], nfr), frames_per_token=div(nfr, ntok))
    for k in ("kl", "ce", "top1"):
        s[f"{k}_utt_mean"] = float(np.nanmean(per_utt[k].to_numpy(np.float64))) if len(per_utt) else float("nan")
    return s


# --------------------------------------------------------------------------------- study per-utterance tables

# the error regions that only change how a word is written, not what was said (the first run's error analysis,
# STUDY.md 4.2 / 4.3 "no-style"): numerals written as kanji or digits, the same kana in hiragana or katakana, a word
# in kanji or in kana. An UPPER bound on style: a kanji <-> kana region is not reading-checked
STYLE_KINDS = ("numeral", "hira<->kata", "kanji->kana", "kana->kanji", "kana-other")
TABLE_COLUMNS = ("id", "set", "ref", "hyp", "edits", "ref_len", "hyp_len", "sub", "del", "ins", "edits_style",
                 "edits_nostyle", "truncated")
_KANJI_NUM = set("〇零一二三四五六七八九十百千万億兆")


def _char_class(c: str) -> str:
    o = ord(c)
    if 0x3041 <= o <= 0x309F:
        return "H"  # hiragana
    if 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
        return "K"  # katakana (with the long-vowel mark)
    if c.isdigit() and o < 128:
        return "D"
    if "a" <= c <= "z":
        return "L"  # normalize_ja lowercases latin
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or c in "々〆〇" or 0xF900 <= o <= 0xFAFF or 0x20000 <= o:
        return "J"  # kanji
    return "O"


def _kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def region_kind(ref: str, hyp: str) -> str:
    """The class of one error region (a maximal run of non-matching alignment chunks: its reference and hypothesis
    substrings, normalize_ja text), as the first run's error analysis classified them: insertion, deletion, numeral,
    latin, hira<->kata, kana-other, kanji->kana, kana->kanji, kanji->kanji(+len), hira->hira, kata->kata, mixed-kanji,
    other. STYLE_KINDS are the no-style CER's."""
    if not ref:
        return "insertion"
    if not hyp:
        return "deletion"
    cr, ch = {_char_class(c) for c in ref}, {_char_class(c) for c in hyp}
    if (cr | ch) & {"D"} or (set(ref) & _KANJI_NUM and ch & {"D"}) or (set(hyp) & _KANJI_NUM and cr & {"D"}):
        return "numeral"
    if (set(ref) | set(hyp)) & _KANJI_NUM and (cr == {"J"} or ch == {"J"}) and (
            all(c in _KANJI_NUM for c in ref) or all(c in _KANJI_NUM for c in hyp)):
        return "numeral"
    if "L" in cr or "L" in ch:
        return "latin"
    if _kata2hira(ref) == _kata2hira(hyp):
        return "hira<->kata"
    if cr <= {"H", "K"} and ch <= {"H", "K"} and ("K" in cr) != ("K" in ch) and cr != ch:
        return "kana-other"
    if "J" in cr and ch <= {"H", "K"}:
        return "kanji->kana"
    if cr <= {"H", "K"} and "J" in ch:
        return "kana->kanji"
    if cr == {"J"} and ch == {"J"}:
        return "kanji->kanji" if len(ref) == len(hyp) else "kanji->kanji(len)"
    if cr <= {"H"} and ch <= {"H"}:
        return "hira->hira"
    if cr <= {"K"} and ch <= {"K"}:
        return "kata->kata"
    if "J" in cr and "J" in ch:
        return "mixed-kanji"
    return "other"


def _text(x) -> str:
    """A text cell as a string: None or NaN (a missing value read back from parquet) is the empty string."""
    return x if isinstance(x, str) else ""


def utterance_scores(refs: Sequence[str], hyps: Sequence[str]) -> pd.DataFrame:
    """Per utterance on normalize_ja strings, from ONE jiwer character alignment of every row with a reference (the
    alignment corpus_cer counts, so the sums over those rows are its edits and ref_chars): edits (S + D + I), ref_len,
    hyp_len, sub, del, ins, edits_style (the edits of the error regions whose region_kind is in STYLE_KINDS) and
    edits_nostyle = edits - edits_style. An empty reference scores ref_len 0 and its hypothesis' characters as
    insertions (not style); corpus sums leave such rows out."""
    R = [normalize_ja(_text(r)) for r in refs]
    H = [normalize_ja(_text(h)) for h in hyps]
    n = len(R)
    sub, dele, ins, style = (np.zeros(n, np.int64) for _ in range(4))
    keep = [i for i in range(n) if R[i]]
    if keep:
        out = jiwer.process_characters([R[i] for i in keep], [H[i] for i in keep])
        for i, chunks in zip(keep, out.alignments):
            regions = []  # [ref start, ref end, hyp start, hyp end, edits] per error region
            open_region = False
            for c in chunks:
                if c.type == "equal":
                    open_region = False
                    continue
                if c.type == "substitute":
                    e = c.ref_end_idx - c.ref_start_idx
                    sub[i] += e
                elif c.type == "delete":
                    e = c.ref_end_idx - c.ref_start_idx
                    dele[i] += e
                else:  # insert
                    e = c.hyp_end_idx - c.hyp_start_idx
                    ins[i] += e
                if open_region:  # adjacent non-matching chunks form one region
                    g = regions[-1]
                    g[1], g[3], g[4] = c.ref_end_idx, c.hyp_end_idx, g[4] + e
                else:
                    regions.append([c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx, e])
                    open_region = True
            style[i] = sum(g[4] for g in regions if region_kind(R[i][g[0]:g[1]], H[i][g[2]:g[3]]) in STYLE_KINDS)
    for i in range(n):
        if not R[i]:
            ins[i] = len(H[i])
    edits = sub + dele + ins
    return pd.DataFrame({"edits": edits, "ref_len": np.array([len(r) for r in R], np.int64),
                         "hyp_len": np.array([len(h) for h in H], np.int64), "sub": sub, "del": dele, "ins": ins,
                         "edits_style": style, "edits_nostyle": edits - style})


def utterance_table(ids: Sequence[str], sets, refs: Sequence[str], hyps: Sequence[str],
                    truncated: Sequence[bool] | None = None) -> pd.DataFrame:
    """A system's per-utterance table in CONTRACT.md 5's format, the input of tools/study_report.py: id, set (the eval
    set; one string for all rows or one per row), ref and hyp as given (raw text), the counts of utterance_scores and
    truncated (False when not given: CTC decoding has no length cut). Row order as given."""
    ids = [str(x) for x in ids]
    sets = [str(sets)] * len(ids) if isinstance(sets, str) else [str(x) for x in sets]
    if not (len(sets) == len(refs) == len(hyps) == len(ids)):
        raise ValueError("ids, sets, refs and hyps must have the same length")
    sc = utterance_scores(refs, hyps)
    trunc = np.zeros(len(ids), bool) if truncated is None else np.asarray(
        [bool(x) if x is not None and x == x else False for x in truncated], bool)
    df = pd.DataFrame({"id": ids, "set": sets, "ref": [_text(r) for r in refs], "hyp": [_text(h) for h in hyps]})
    for c in sc.columns:
        df[c] = sc[c].to_numpy()
    df["truncated"] = trunc
    return df[list(TABLE_COLUMNS)]


def table_cer(df: pd.DataFrame, ids: Iterable[str] | None = None, nostyle: bool = False) -> dict:
    """Corpus CER of a per-utterance table (or of its rows `ids`): sum edits / sum ref_len over the rows with a
    reference, as corpus_cer; edits_nostyle with nostyle=True. Returns cer, edits, ref_chars, n, n_empty_ref."""
    if ids is not None:
        want = set(ids)
        df = df[df["id"].isin(want)]
    has_ref = df["ref_len"].to_numpy(np.int64) > 0
    e = int(df[("edits_nostyle" if nostyle else "edits")].to_numpy(np.int64)[has_ref].sum())
    n_ref = int(df["ref_len"].to_numpy(np.int64)[has_ref].sum())
    return dict(cer=e / n_ref if n_ref else float("nan"), edits=e, ref_chars=n_ref, n=int(has_ref.sum()),
                n_empty_ref=int((~has_ref).sum()))


# ------------------------------------------------------------------------------------------ logging helpers


def flatten(summary: dict, prefix: str) -> dict[str, float]:
    """Eval summary -> {"prefix/<set>/key": number, "prefix/all/key": ..., "prefix/wall_s": ...} for
    RunLogger.scalars (non-numeric leaves such as bad_audio are dropped). bad_audio_per_set becomes
    "prefix/<set>/n_bad_audio", next to the set's other counts (only for a set that lost a row)."""
    out = {}

    def walk(d, p):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{p}/{k}")
            elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                out[f"{p}/{k}"] = float(v)

    walk({k: v for k, v in summary.items() if k not in ("sets", "bad_audio_per_set")}, prefix)
    for s, d in summary.get("sets", {}).items():
        walk(d, f"{prefix}/{s}")
    for s, k in (summary.get("bad_audio_per_set") or {}).items():
        out[f"{prefix}/{s}/n_bad_audio"] = float(k)
    return out


HEADLINE_CER = ("val_cer", "val_cer_vs_teacher", "train_cer", "train_cer_vs_teacher")


def _pooled(sets: dict, num: str, den: str) -> float | None:
    """sum(num) / sum(den) over the per-set summaries in `sets` (None without a denominator)."""
    d = sum(float(s.get(den) or 0) for s in sets.values())
    return sum(float(s.get(num) or 0) for s in sets.values()) / d if d else None


def _gate_sets(summary: dict | None) -> dict:
    """The GATE sets of an eval summary's per-set results (every set if none of them is a gate set)."""
    sets = (summary or {}).get("sets") or {}
    gate = {s: d for s, d in sets.items() if s in GATE_SETS}
    return gate or dict(sets)


def headline(tf: dict | None = None, greedy: dict | None = None, probe: dict | None = None,
             probe_greedy: dict | None = None) -> dict:
    """The numbers to read first after an eval, in one place (the trainer logs them as summary/<full|mini>/<name>).
    val_* pool the GATE sets that were evaluated (every evaluated set if none is a gate set), train_* every train
    source of the train-side eval:
      val_cer               corpus CER vs the dataset reference: sum ref_edits / sum ref_chars of the per-set greedy
                            summaries (the gate's measure, pooled)
      val_cer_vs_teacher    corpus CER vs the teacher's hypothesis: sum cer_teacher_edits / sum cer_teacher_chars
      val_loss, val_top1    teacher-forced KL (nats/token) and top-1 agreement with the teacher, token-weighted over the
                            sets (val_loss = eval_record's heldout_kl)
      train_cer, train_cer_vs_teacher  the same CERs on the greedy decode of train utterances
      train_loss, train_top1           teacher-forced on train utterances (the probe)
    CERs are fractions (0.083 = 8.3 %). A number whose inputs were not evaluated is left out."""
    out = {}
    vg, tg = _gate_sets(greedy), dict((probe_greedy or {}).get("sets") or {})
    for side, sets in (("val", vg), ("train", tg)):
        for name, num, den in (("cer", "ref_edits", "ref_chars"), ("cer_vs_teacher", "cer_teacher_edits",
                                                                  "cer_teacher_chars")):
            v = _pooled(sets, num, den)
            if v is not None:
                out[f"{side}_{name}"] = v
    vt = {s: d for s, d in _gate_sets(tf).items() if d.get("n_tok")}
    ntok = sum(d["n_tok"] for d in vt.values())
    if ntok:
        out["val_loss"] = sum(d["kl"] * d["n_tok"] for d in vt.values()) / ntok
        out["val_top1"] = sum(d["top1"] * d["n_tok"] for d in vt.values()) / ntok
    if probe and "all" in probe:
        out["train_loss"], out["train_top1"] = probe["all"]["kl"], probe["all"]["top1"]
    return out


def headline_val_utts(greedy: dict | None) -> int:
    """How many utterances headline()'s val CERs pool: the greedy summary's gate sets (every set if none is one). The
    trainer logs it next to val_cer, since its full evals decode the fixed subset or the complete sets."""
    return sum(int(d.get("n", 0)) for d in _gate_sets(greedy).values())


def gate_kd_sums(tf: dict | None) -> dict | None:
    """A teacher_forced_eval summary's KL and CE pooled token-weighted over its GATE sets (JSUT / CV8 / Reazon: the
    monitor-only hold-outs eval_emilia and galgame are left out; every evaluated set only when none is a gate set, as
    headline() pools val_loss): kl_sum and ce_sum, the sums of the per-token terms, n_tok the target tokens they run
    over, sets the sets pooled. A set's kl / ce is its sum over tokens / n_tok (_tf_summarise), so kl * n_tok gives
    that sum back to float64 rounding: the pooled mean is the sum over every gate token / their count, not a mean of
    per-set means. None without a token."""
    vt = {s: d for s, d in _gate_sets(tf).items() if d.get("n_tok")}
    n = sum(int(d["n_tok"]) for d in vt.values())
    if not n:
        return None
    return dict(kl_sum=sum(float(d["kl"]) * int(d["n_tok"]) for d in vt.values()),
                ce_sum=sum(float(d["ce"]) * int(d["n_tok"]) for d in vt.values()), n_tok=n, sets=sorted(vt))


def combined_loss(tf: dict | None, w_kl: float, w_ce: float) -> dict | None:
    """The training objective measured on an eval: (w_kl * sum KL + w_ce * sum CE) / target tokens over the gate sets
    (gate_kd_sums). The trainer's loss for a step is the same expression over the step's tokens
    (kitsune.kd.kd_objective, logged as loss/objective): per target token of the teacher's sequence (the collate's
    tgt_row / tgt_pos, pads never counted), the 17-bin KL and the CE on the teacher's greedy token of kd_losses, which
    teacher_forced_eval scores with. The decoupled L2-SP value is not part of it, as it is not part of the gradient.
    The two differ only in their input, deliberately: a training step's value is on augmented audio (SpecAugment on
    the log-mel features) in train mode with the weights before its update, an eval's on the un-augmented audio in
    eval mode. Returns gate_kd_sums' record with value (the combined loss), kl and ce per token and the weights; None
    without a token."""
    s = gate_kd_sums(tf)
    if s is None:
        return None
    n = s["n_tok"]
    return dict(value=(float(w_kl) * s["kl_sum"] + float(w_ce) * s["ce_sum"]) / n, kl=s["kl_sum"] / n,
                ce=s["ce_sum"] / n, w_kl=float(w_kl), w_ce=float(w_ce), **s)


def eval_record(step: int, elapsed_s: float, tf: dict | None = None, greedy: dict | None = None,
                probe: dict | None = None, greedy_full: dict | None = None, lr_phase: str | None = None) -> dict:
    """Compact per-eval record; the trainer appends one per eval to the history that verdict() reads.

    greedy holds the fixed greedy subset's per-set numbers at every eval (what the early stop's neighbours, the
    summary's best step and verdict v1 read, so the curve keeps one population from step 0 on); greedy_full, the same
    numbers of the COMPLETE sets, only for an eval that decoded them (verdict v2's CER trend reads them when every eval
    in its window has them). lr_phase: the LR phase of the last optimizer step before the eval ("warmup", "stable" or
    "cooldown", LR_PHASES; none at step 0), which verdict v2 splits the history by."""
    rec = dict(step=int(step), elapsed_s=float(elapsed_s))
    if lr_phase is not None:
        rec["lr_phase"] = str(lr_phase)
    if tf and "all" in tf:
        # the pre-registered held-out KL pools the GATE sets only, so adding a monitor-only eval set (eval_emilia)
        # cannot move the over-fitting test; the all-sets value is kept alongside
        gate = [d for s, d in tf.get("sets", {}).items() if s in GATE_SETS and d.get("n_tok")]
        ntok = sum(d["n_tok"] for d in gate)
        rec["heldout_kl"] = sum(d["kl"] * d["n_tok"] for d in gate) / ntok if ntok else tf["all"]["kl"]
        rec["heldout_kl_all_sets"] = tf["all"]["kl"]
        rec["tf"] = {s: {k: d[k] for k in ("kl", "ce", "top1")} for s, d in tf["sets"].items()}
    if probe and "all" in probe:
        rec["probe_kl"] = probe["all"]["kl"]
    for key, summary in (("greedy", greedy), ("greedy_full", greedy_full)):
        if summary and summary.get("sets"):
            rec[key] = {s: {k: d[k] for k in ("cer_ref_corpus", "teacher_cer_ref_corpus", "trunc_rate", "n")}
                        for s, d in summary["sets"].items()}
    return rec


# the trainer's WSD phases 0 / 1 / 2 (04_distill.wsd_lr) by the names its `lr_phase` events and the history use
LR_PHASES = ("warmup", "stable", "cooldown")


def lr_phase_at(step: int, lr_phase_events: Iterable[dict]) -> str | None:
    """The LR phase of optimizer step `step`, from the trainer's `lr_phase` events (each names the phase that begins at
    its at_step): the phase of the event with the largest at_step <= step. A resume that replays steps logs the same
    boundaries again, so the largest at_step, not the last event in the file, is the one that holds. None for step 0
    (no step taken) and before the first event. It gives the lr_phase of history records written before the trainer
    recorded one (the first A100 run's), exactly as the trainer now records it."""
    best = None
    for e in lr_phase_events:
        at = int(e["at_step"])
        if at <= step and (best is None or at >= best[0]):
            best = (at, str(e["phase"]))
    return best[1] if best is not None and step > 0 else None


# ---------------------------------------------------------------------------------------------------- verdict


def _rel_change(xs: list[float], ys: list[float], scale: float | None = None) -> float | None:
    """Least-squares change of y across the window, relative to `scale` (default the window's mean |y|);
    None with fewer than 2 distinct x."""
    if len(set(xs)) < 2:
        return None
    x, y = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    slope = np.polyfit(x, y, 1)[0]
    scale = scale if scale is not None else float(np.mean(np.abs(y)))
    return float(slope * (x.max() - x.min()) / (scale or 1e-12))


def _tail(records: list[dict], tail_frac: float, min_points: int) -> list[dict]:
    """The last max(min_points, ceil(tail_frac n)) of n records (all of them when there are fewer)."""
    return records[len(records) - min(len(records), max(min_points, math.ceil(tail_frac * len(records)))):]


def _gate_ratio_mean(g: dict | None) -> float | None:
    """A history record's CER measure for the trend: the mean over the gate sets of student / teacher corpus CER, from
    its greedy (subset) or greedy_full (complete sets) numbers; None without a gate set that has a teacher CER."""
    g = g or {}
    v = [g[s]["cer_ref_corpus"] / g[s]["teacher_cer_ref_corpus"] for s in GATE_SETS
         if s in g and g[s].get("teacher_cer_ref_corpus")]
    return float(np.mean(v)) if v else None


def _gates(g: dict | None) -> set[str]:
    return {s for s in GATE_SETS if s in (g or {})}


def _complete(records: list[dict]) -> bool:
    """Every record carries complete-set numbers (greedy_full) for the gate sets its subset numbers cover."""
    return bool(records) and all(_gates(r.get("greedy_full")) >= (_gates(r.get("greedy")) or set(GATE_SETS))
                                 for r in records)


def dedup_history(records: list[dict], min_epoch_gap: float) -> tuple[list[dict], list[int]]:
    """verdict v2's de-duplication of the trained records (in step order): a record that comes less than min_epoch_gap
    epochs after the previous KEPT record is dropped, the later one of the two. The first A100 run's final eval came 46
    steps (0.04 epoch) after its epoch-8 eval: two near-identical models counted twice at the end of the window. A
    record without an epoch (a run outside epoch mode written before v2) is kept. Returns (kept, dropped steps)."""
    kept, dropped = [], []
    for r in records:
        p = kept[-1] if kept else None
        if (p is not None and r.get("epoch") is not None and p.get("epoch") is not None
                and float(r["epoch"]) - float(p["epoch"]) < min_epoch_gap):
            dropped.append(int(r["step"]))
        else:
            kept.append(r)
    return kept, dropped


def _cooldown_gain(records: list[dict]) -> dict | None:
    """verdict v2's cooldown gain: the final eval (the last record) against the last pre-cooldown eval, both on the
    complete sets when both have them, else on the subset. `records` is the whole trained history, NOT the
    de-duplicated one: the "before" eval is its last record whose lr_phase is "stable" - the eval nearest the
    cooldown's start - even when dedup_history left it out of the trend (it came < min_epoch_gap after the one before).
    De-duplication keeps two near-identical models from both weighing on a slope; a two-point gain wants the model
    closest to where the cooldown began. None when the final eval does not follow a cooldown step or no eval came
    before the cooldown."""
    if not records or records[-1].get("lr_phase") != "cooldown":
        return None
    pre = [r for r in records if r.get("lr_phase") == "stable"]
    if not pre:
        return None
    a, b = pre[-1], records[-1]
    key = "greedy_full" if _complete([a, b]) else "greedy"
    before, after = _gate_ratio_mean(a.get(key)), _gate_ratio_mean(b.get(key))
    if before is None or after is None:
        return None
    ga, gb = a.get(key) or {}, b.get(key) or {}
    return dict(from_step=int(a["step"]), to_step=int(b["step"]), from_epoch=a.get("epoch"), to_epoch=b.get("epoch"),
                scope="complete" if key == "greedy_full" else "subset", cer_ratio_before=before, cer_ratio_after=after,
                cer_ratio_rel_change=after / before - 1.0 if before else None,
                sets={s: dict(cer_before=float(ga[s]["cer_ref_corpus"]), cer_after=float(gb[s]["cer_ref_corpus"]))
                      for s in GATE_SETS if s in ga and s in gb})


def verdict(results: dict, *, go_ratio: float = 1.2, promising_ratio: float = 1.5, max_trunc: float = 0.005,
            tail_frac: float = 0.2, min_points: int = 3, improving_rel: float = 0.01, widening_rel: float = 0.02,
            overfit_rel: float = 0.01, version: int = 1, min_epoch_gap: float = 0.25, family: str = "aed",
            teacher_prereg: dict | None = None) -> dict:
    """Gate D32a, pre-registered.

    results = {"final":   greedy_eval summary of the FULL eval sets at the end of the run (its bad_audio_per_set:
                          the rows that could not be decoded, per set),
               "history": [eval_record(...), ...],
               "teacher": optional {set: teacher corpus CER} (default: teacher CER on the same ids from "final",
                          else the pre-registered numbers),
               "reference": optional load_reference(...): a reference model's per-set CER, reported as the
                          verdict's "reference" next to the teacher gate (reference_bar), under either version; no
                          tier depends on it}

    GO         student corpus CER <= 1.2x teacher on >= 2 of the 3 sets, AND truncation <= 0.5 % of the final outputs,
               AND the (held-out KL - probe KL) gap is not widening over the last 20 % of evals
    PROMISING  <= 1.5x on >= 2 sets and the greedy-subset CER still improving over the last 20 % of evals
    NO-GO      fewer than 2 sets within 1.5x and CER flat, or over-fitting (probe KL falling while held-out KL rises
               over the last 20 % of evals)
    INCONCLUSIVE  what the pre-registration does not cover (within 1.5x but flat, or beyond 1.5x but still
               improving, or the CER trend unknown: fewer than 2 evals after step 0); `reasons` says which. It is not
               forced into a tier after the fact.
    "Last 20 %" = the last max(min_points, ceil(0.2 n)) of the n eval records after step 0, by step. The step-0 eval
    of the untrained student stays in the history (the curves) but never enters a trend: its numbers (CER tens of
    times the teacher's, held-out KL ~5) would dominate any window that reaches it - every history of <= 3 records
    with per-epoch evals - forcing "improving" and hiding over-fitting. Trends are least-squares changes across that
    window relative to the window mean (the gap: relative to the mean held-out KL); the thresholds are returned with
    the verdict. A gate set some of whose audio could not be decoded is judged on the rows that decoded (the teacher
    CER on the same ids by default, so the ratio stays paired); per_set carries n and n_bad_audio, and a reason says
    it was not the pre-registered full set. The tiers do not change.

    version 1 is the above, exactly as the first A100 run was judged (INCONCLUSIVE: "still improving" measured inside
    the WSD cooldown, on the 500-per-set subset, with two near-identical end points; configs keep it unless they set
    eval.verdict_version 2). version 2 (pre-registered for the runs after it) keeps the tiers, the per-set gate, the
    thresholds and the measure (the mean over the gate sets of student / teacher corpus CER) and changes which evals
    the trends read and on what:
      1. the trained records (step > 0, by step) are de-duplicated (dedup_history): one that comes less than
         min_epoch_gap (0.25) epochs after the previous kept one is dropped, the later of the two
      2. over-fitting and the KL gap: as in v1 over the last max(min_points, ceil(0.2 n)) of the n de-duplicated
         records, cooldown included (over-fitting in the cooldown still counts)
      3. "CER still improving": over the PRE-COOLDOWN evals only - the de-duplicated records whose lr_phase is
         "stable" (the eval followed a step at the peak LR, after the warm-up and before the cooldown; an early-stop
         cooldown counts as the cooldown) - their last max(min_points, ceil(0.2 m)) of m. At least min_points (3) of
         them are needed, else the trend is unknown ("trend: insufficient pre-cooldown evals") and the run cannot be
         PROMISING or the flat NO-GO. The question is whether more training at the peak LR still helps; the
         cooldown's one-off annealing gain says nothing about that, and a window at the end of a WSD run always
         falls inside it (both are the last 20 %). The numbers are the complete sets' (greedy_full) when every record
         in that window has them, else the fixed subset's (greedy); trend.cer_scope and a reason say which
      4. reported next to the tiers, never gating: pre_cooldown_slope - that window's CER measure, its relative
         change across the window (the one "improving" tests; x = step) and the least-squares slope per epoch
         (per_epoch, and rel_per_epoch = per_epoch / the window's mean) - and cooldown_gain - the final eval (the
         history's last record, whatever the de-duplication dropped) against the last eval before the cooldown (the
         history's last record whose lr_phase is "stable", also taken BEFORE de-duplication: the eval nearest the
         cooldown's start, which with evals closer than min_epoch_gap can be one the trend in 3 left out), on the
         complete sets when both have them: the measure before and after, its relative change, and each gate set's
         CER before and after (null without an eval on each side)
    A record's lr_phase is the trainer's (eval_record); lr_phase_at gives it for a history written before the trainer
    recorded one. Records without it never count as pre-cooldown (a reason counts them).

    family: whose teacher the ratios are to (the size study). "aed" (default) is everything above, with Cohere: its
    output is unchanged by this option. "ctc" judges a Parakeet CTC student against ITS teacher, Parakeet's stored CTC
    path: the final summary's teacher_cer_ref_corpus is that teacher's CER on the same ids when ctc_eval scored it
    (ctc_teacher_rows), and the registered numbers are PARAKEET_CTC_CER_PREREG (teacher_prereg overrides them); while
    they are pending, per_set's teacher_prereg and baseline_drift are None and a set without a same-ids teacher CER
    cannot be judged (a reason says so). The tiers, thresholds and trends are the same; the output also carries
    family, teacher and teacher_prereg_status ("registered" / "pending")."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    prereg = (family_teacher_prereg(family) if teacher_prereg is None else dict(teacher_prereg)) or {}
    fin = results.get("final") or {}
    fsets = fin.get("sets", {})
    bad = fin.get("bad_audio_per_set") or {}
    tover = results.get("teacher") or {}
    reasons, per_set = [], {}
    n_go = n_prom = n_out = n_trunc = 0
    for s in GATE_SETS:
        d = fsets.get(s)
        k = int(bad.get(s, 0))
        if d is None:
            reasons.append(f"{s}: no final greedy result" + (f" ({k} undecodable rows)" if k else ""))
            continue
        tc = d.get("teacher_cer_ref_corpus")
        if s in tover:
            teacher, src = float(tover[s]), "results.teacher"
        elif tc is not None and not math.isnan(tc):
            teacher, src = float(tc), "same ids"
        elif s in prereg:
            teacher, src = prereg[s], "pre-registered"
        else:
            reasons.append(f"{s}: no {FAMILY_TEACHER[family]} CER on these ids and none pre-registered yet (pending): "
                           f"not judged")
            continue
        student = float(d["cer_ref_corpus"])
        # a perfect teacher on these ids (tiny sets only): the student passes only by being perfect too
        ratio = student / teacher if teacher > 0 else (0.0 if student == 0 else math.inf)
        n_go += ratio <= go_ratio
        n_prom += ratio <= promising_ratio
        n_out += int(d["n"])
        n_trunc += int(d["n_truncated"])
        p = prereg.get(s)
        per_set[s] = dict(student=student, teacher=teacher, teacher_source=src, ratio=ratio,
                          teacher_prereg=p, baseline_drift=teacher - p if p is not None else None,
                          threshold_go=go_ratio * teacher, threshold_promising=promising_ratio * teacher,
                          n=int(d["n"]), n_bad_audio=k)
        if k:
            reasons.append(f"{s}: judged on {int(d['n'])} decoded rows, {k} undecodable (not the pre-registered "
                           f"full set)")
    trunc_rate = n_trunc / n_out if n_out else float("nan")
    trunc_ok = n_out > 0 and trunc_rate <= max_trunc

    if version not in (1, 2):
        raise ValueError(f"verdict version must be 1 or 2, got {version!r}")
    hist = sorted((r for r in results.get("history") or [] if r["step"] > 0), key=lambda r: r["step"])  # trained only
    extra = {}
    if version == 1:
        window = _tail(hist, tail_frac, min_points)
        cer_window, cer_key = window, "greedy"
    else:
        kept, dropped = dedup_history(hist, float(min_epoch_gap))
        window = _tail(kept, tail_frac, min_points)
        pre = [r for r in kept if r.get("lr_phase") == "stable"]
        cer_window = _tail(pre, tail_frac, min_points) if len(pre) >= min_points else []
        cer_key = "greedy_full" if _complete(cer_window) else "greedy"
        if dropped:
            reasons.append(f"de-duplicated: the evals at steps {dropped} come less than {min_epoch_gap} epoch after "
                           f"the one before and are left out of the trends")
        if n_unphased := sum(r.get("lr_phase") is None for r in hist):
            reasons.append(f"{n_unphased} evals without an LR phase are left out of the CER trend")

    def series(fn, recs=None):
        pts = [(r["step"], fn(r)) for r in (window if recs is None else recs)]
        pts = [(x, y) for x, y in pts if y is not None and not math.isnan(y)]
        return [x for x, _ in pts], [y for _, y in pts]

    cer_x, cer_y = series(lambda r: _gate_ratio_mean(r.get(cer_key)), cer_window)
    cer_change = _rel_change(cer_x, cer_y) if version == 1 or len(cer_x) >= min_points else None
    held, probe = series(lambda r: r.get("heldout_kl")), series(lambda r: r.get("probe_kl"))
    held_change, probe_change = _rel_change(*held), _rel_change(*probe)
    gx, gy = series(lambda r: r["heldout_kl"] - r["probe_kl"] if "heldout_kl" in r and "probe_kl" in r else None)
    gap_change = _rel_change(gx, gy, scale=float(np.mean(np.abs(held[1]))) if held[1] else None)

    improving = cer_change is not None and cer_change <= -improving_rel
    gap_ok = gap_change is not None and gap_change <= widening_rel
    overfit = (probe_change is not None and held_change is not None
               and probe_change < -overfit_rel and held_change > overfit_rel)
    if version == 1 and cer_change is None:
        reasons.append("CER trend unknown: fewer than 2 greedy evals after step 0 in the window")
    if version == 2:
        scope = ("complete" if cer_key == "greedy_full" else "subset") if cer_window else None
        if cer_change is None:
            reasons.append(f"trend: insufficient pre-cooldown evals ({len(cer_x) if cer_window else len(pre)} after "
                           f"de-duplication, {min_points} needed)")
        else:
            reasons.append(f"CER trend {cer_change:+.1%} over the pre-cooldown evals at steps {cer_x} ("
                           + ("the complete sets)" if scope == "complete" else "the fixed greedy subset)"))
        slope = None
        if cer_change is not None:
            ep = {int(r["step"]): r.get("epoch") for r in cer_window}
            per_epoch = None
            if all(ep.get(x) is not None for x in cer_x) and len({float(ep[x]) for x in cer_x}) >= 2:
                per_epoch = float(np.polyfit(np.asarray([float(ep[x]) for x in cer_x], np.float64),
                                             np.asarray(cer_y, np.float64), 1)[0])
            mean = float(np.mean(np.abs(cer_y)))
            slope = dict(steps=cer_x, epochs=[ep.get(x) for x in cer_x], scope=scope, cer_ratio=cer_y,
                         rel_change=cer_change, per_epoch=per_epoch,
                         rel_per_epoch=per_epoch / mean if per_epoch is not None and mean else None)
        extra = dict(pre_cooldown_slope=slope, cooldown_gain=_cooldown_gain(hist))
    if gap_change is None:
        reasons.append("KL-gap trend unknown: fewer than 2 evals after step 0 with both probe and held-out KL")

    if overfit:
        v = "NO-GO"
        reasons.append(f"over-fitting: probe KL {probe_change:+.1%} while held-out KL {held_change:+.1%} over the window")
    elif n_go >= 2 and trunc_ok and gap_ok:
        v = "GO"
        reasons.append(f"{n_go}/3 sets within {go_ratio}x teacher, truncation {trunc_rate:.2%}, KL gap not widening")
    elif n_prom >= 2 and improving:
        v = "PROMISING"
        reasons.append(f"{n_prom}/3 sets within {promising_ratio}x teacher and CER still improving ({cer_change:+.1%})")
    elif n_prom < 2 and cer_change is not None and not improving:  # flat is a measured trend, not an unknown one
        v = "NO-GO"
        reasons.append(f"only {n_prom}/3 sets within {promising_ratio}x teacher and CER flat")
    else:
        v = "INCONCLUSIVE"
        trend = ("CER trend unknown" if cer_change is None
                 else "CER not improving" if n_prom >= 2 else "CER still improving")
        reasons.append("outside the pre-registered tiers: " + (
            f"within {promising_ratio}x on {n_prom}/3 sets but {trend}" if n_prom >= 2
            else f"only {n_prom}/3 sets within {promising_ratio}x but {trend}"))
    if n_go >= 2 and not trunc_ok:
        reasons.append(f"truncation {trunc_rate:.2%} > {max_trunc:.2%} blocks GO")
    if n_go >= 2 and trunc_ok and not gap_ok and not overfit:
        reasons.append("probe/held-out KL gap widening (or unknown) blocks GO")

    trend = dict(window_steps=[r["step"] for r in window], cer_ratio_rel_change=cer_change,
                 improving=bool(improving), heldout_kl_rel_change=held_change,
                 probe_kl_rel_change=probe_change, gap_rel_change=gap_change,
                 gap_widening=None if gap_change is None else not gap_ok, overfit=bool(overfit))
    thresholds = dict(go_ratio=go_ratio, promising_ratio=promising_ratio, max_trunc=max_trunc,
                      tail_frac=tail_frac, min_points=min_points, improving_rel=improving_rel,
                      widening_rel=widening_rel, overfit_rel=overfit_rel)
    out = dict(verdict=v, reasons=reasons, sets=per_set, n_sets_go=int(n_go), n_sets_promising=int(n_prom),
               trunc_rate=trunc_rate, trunc_ok=bool(trunc_ok), trend=trend, thresholds=thresholds)
    if version == 2:
        trend.update(cer_window_steps=[r["step"] for r in cer_window], cer_scope=scope, deduplicated_steps=dropped,
                     n_pre_cooldown=len(pre))
        thresholds.update(min_epoch_gap=float(min_epoch_gap))
        out = dict(verdict=v, version=2, **{k: x for k, x in out.items() if k != "verdict"}, **extra)
    if family != "aed":  # the Cohere verdict's keys stay exactly as they were
        out.update(family=family, teacher=FAMILY_TEACHER[family],
                   teacher_prereg_status="registered" if prereg else "pending")
    if results.get("reference"):  # reported next to the tiers, never part of them
        out["reference"] = reference_bar(fin, results["reference"])
    return out


# ------------------------------------------------------------------------------------------- reference model


def load_reference(path, name: str | None = None) -> dict:
    """A reference model's corpus CER on the gate sets (eval.reference: {"name": ..., "path": ...}), for the bar the
    verdict reports next to the teacher gate (reference_bar). The file is JSON (the numbers here only show the format):
        {"name": "parakeet-tdt-0.6b-ja",
         "scope": "complete gate sets, corpus CER under kitsune.text.normalize_ja",
         "cer": {"eval_jsut": 0.0731, "eval_cv8": 0.0795, "eval_reazon": 0.0718}}
    cer holds fractions (0.0731 = 7.31 %) of the gate's own measure, corpus CER (corpus_cer: sum of char edits / sum of
    reference chars under normalize_ja) on the COMPLETE gate sets, so they compare with the verdict's student CER; a
    gate set left out is not compared. name (the config's wins when it gives one) and scope are labels. Anything else
    - no gate set, a set that is not a gate set, a CER outside [0, 1] (a percent typed as 7.31) - raises ValueError,
    so a bad file stops the trainer at start-up rather than at its verdict 4 h later."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    cer = raw.get("cer") if isinstance(raw, dict) else None
    if not isinstance(cer, dict) or not cer:
        raise ValueError(f"{p}: a reference file needs a non-empty \"cer\": {{set: corpus CER}} (load_reference)")
    unknown = sorted(set(cer) - set(GATE_SETS))
    if unknown:
        raise ValueError(f"{p}: {unknown} are not gate sets ({', '.join(GATE_SETS)}): only those are compared")
    out = {}
    for s in GATE_SETS:
        if s in cer:
            v = cer[s]
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1):
                raise ValueError(f"{p}: cer.{s} must be a fraction in [0, 1] (0.0731 = 7.31 %), got {v!r}")
            out[s] = float(v)
    label = name or raw.get("name")
    if not label:
        raise ValueError(f"{p}: the reference needs a name (in the file or eval.reference.name)")
    return dict(name=str(label), path=str(p), scope=raw.get("scope"), cer=out)


def reference_bar(final: dict | None, reference: dict) -> dict:
    """The student's final corpus CER next to a reference model's (load_reference) on each gate set both have, and
    pooled over those sets; ratio = student / reference (below 1: the student is better). Pooled: the student's sum of
    char edits / sum of reference chars (the final summary's ref_edits / ref_chars, the headline's val_cer when all
    three sets are there), and the reference's per-set CERs weighted by the same reference chars - what the reference
    scores on the pooled corpus, since its numbers are on the same complete sets and references. Reported only
    (gating false): no tier depends on it; it answers "is the student useful?", which the teacher-ratio gate does not
    (a student at a public model's level can still fail 1.5x of this teacher on CV8 and ReazonSpeech)."""
    fsets = (final or {}).get("sets") or {}
    rows = {}
    for s in GATE_SETS:
        if s in reference["cer"] and s in fsets:
            st, rf = float(fsets[s]["cer_ref_corpus"]), float(reference["cer"][s])
            rows[s] = dict(student=st, reference=rf, ratio=st / rf if rf > 0 else (0.0 if st == 0 else math.inf))
    pooled = None
    if rows and all(fsets[s].get("ref_chars") and "ref_edits" in fsets[s] for s in rows):
        chars = {s: float(fsets[s]["ref_chars"]) for s in rows}
        n = sum(chars.values())
        st = sum(float(fsets[s]["ref_edits"]) for s in rows) / n
        rf = sum(rows[s]["reference"] * chars[s] for s in rows) / n
        pooled = dict(student=st, reference=rf, ratio=st / rf if rf > 0 else (0.0 if st == 0 else math.inf),
                      sets=sorted(rows))
    return dict(name=reference["name"], scope=reference.get("scope"), path=reference.get("path"), gating=False,
                sets=rows, pooled=pooled, not_compared=[s for s in GATE_SETS if s not in rows])
