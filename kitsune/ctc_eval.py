"""Evaluation of a CTC-family student (the size study's Parakeet students) on a frame store, in the trainer's formats.

scripts/04_distill.py (family "ctc") evaluates with these what kitsune.evaluate evaluates for the AED students, and
hands the results to the same summary, history, headline and verdict code, so every record of a CTC run has the AED
run's shape. One forward pass per micro-batch gives both views, because a CTC student's greedy decode is the argmax
of the very log-probs the teacher-forced metrics score:

  teacher-forced   kitsune.ctc_kd.ctc_kd_losses against the stored Parakeet frame targets: the frame KL split into
                   dense frames (the stored top-8 U {blank} + rest) and blank-only frames (2 bins), the CTC loss on the
                   teacher's greedy CTC path, the frame argmax agreement with the teacher, and the student's and the
                   teacher's argmax-blank shares (the blank-collapse watch; the teacher's is 59-72 % on real data)
  greedy           kitsune.ctc_student.greedy_ctc_ids (argmax, collapse repeats, drop blanks) -> decode_ids, scored
                   against the reference AND against the teacher's stored CTC hypothesis ctc_hyp (imitation error); the
                   teacher baseline on the same ids is Parakeet CTC (ctc_hyp): the CTC students' own teacher

The teacher-forced summary (summarise_ctc_tf) uses the AED summary's keys where the quantities correspond, so
kitsune.evaluate.eval_record / headline / gate_kd_sums read it unchanged (headline's val_top1 re-pooled per frame by
frame_headline): per set `kl` = (KL dense + KL blank) summed
over every valid frame / N_u, the CTC target tokens (the objective's normalisation), `ce` = the CTC loss per target
token (the CTC family's second loss term, also under `ctc`), `top1` = the frame argmax agreement (also
`argmax_agree`), n_tok = N_u. So the training objective on an eval is w_kl * kl + w_ctc * ce (combined_loss_ctc), the
held-out KL of the history is the KL per target token, and the rest is under its own names (kl_dense, kl_blank,
kl_per_frame, argmax_blank, teacher_blank, frac_dense, frames_per_token, n_frames).

The model is put in eval mode for the pass and every module's training flag is restored afterwards (frozen BatchNorm
stays frozen; a training BatchNorm uses its running stats). Under CUDA the encoder runs in bf16 autocast, the CTC head
always in fp32 (kitsune.ctc_student.ctc_log_probs), as in training and as in the label pass.

One source for what both evaluators share: the teacher-forced summary IS kitsune.evaluate.summarise_ctc_tf (this
module's summarise_ctc_tf only renames the trainer's n_frames column), and the Parakeet CTC teacher's registered
numbers are kitsune.evaluate's (teacher_prereg: study/PREREG.json's baselines, the K6 numbers only while those are
pending). The pass itself stays two implementations - this one reads a frame store only and can skip the decode;
kitsune.evaluate.ctc_eval also scores a token store against separately loaded targets - and
tests/test_evaluate_study.py asserts that both give the same sums, summaries and greedy paths on the same batches.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from kitsune import evaluate as ev
from kitsune.ctc_kd import CTC_BLANK, ctc_kd_losses
from kitsune.trainset import dataset_for, eval_batches

GATE_SETS = ev.GATE_SETS
BASELINE_TOL = ev.BASELINE_TOL
TEACHER_SYSTEM = ev.FAMILY_TEACHER["ctc"]  # "parakeet-ctc", CONTRACT.md 5's system name of this teacher
# The Parakeet CTC teacher's corpus CER (fractions) on the complete gate sets as the label box stored them
# (parakeet_out/<set>/eval-*.jsonl ctc_hyp vs ref under kitsune.evaluate.corpus_cer): WP0's K6 on labels/full,
# 0.06706 / 0.07577 / 0.09706 (every eval row, which for the gate sets are exactly the manifest's). Only the stand-in
# while study/PREREG.json's baselines are pending: teacher_prereg.
PARAKEET_CTC_CER_K6 = {"eval_jsut": 0.0671, "eval_cv8": 0.0758, "eval_reazon": 0.0971}
# the TDT hypothesis ("hyp") on the same sets, K6: the off-the-shelf bar, reported only
PARAKEET_TDT_CER = {"eval_jsut": 0.0662, "eval_cv8": 0.0750, "eval_reazon": 0.1020}


def teacher_prereg() -> tuple[dict[str, float], str]:
    """(the Parakeet CTC teacher's gate CER per set, its source): kitsune.evaluate's PARAKEET_CTC_CER_PREREG, which
    reads study/PREREG.json's filled baselines ("registered"), else the K6 numbers ("pending: K6"). Read at call time,
    so a PREREG fill (or a test's monkeypatch of kitsune.evaluate) is seen without a re-import. ctc_teacher_baselines
    checks the recomputed CER against it at trainer start; the verdict's own teacher_prereg is kitsune.evaluate's
    (None while pending)."""
    reg = ev.family_teacher_prereg("ctc")
    return (reg, "registered") if reg else (dict(PARAKEET_CTC_CER_K6), "pending: K6")


# the per-utterance sums of one pass (ctc_kd_losses' per_utt terms)
SUMS = ("kl_dense", "kl_blank", "ctc", "n_dense", "n_blank_frames", "argmax_agree", "argmax_blank", "teacher_blank")
FRAME_KEYS = ("frame_mask", "dense_mask", "blank_lp", "topk_idx", "topk_lp", "ctc_targets", "ctc_target_lengths",
              "n_frames")


# ------------------------------------------------------------------------------------------------ helpers


@contextmanager
def eval_mode(model):
    """Eval mode for the block, then every module's own training flag back (keeps frozen BN frozen)."""
    flags = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for m, t in flags:
            m.training = t


def _amp(device, amp: bool | None) -> bool:
    return torch.device(device).type == "cuda" if amp is None else amp


def frame_batch(item: dict) -> dict:
    """The frame-target part of a FrameBatchDataset micro-batch: what kitsune.ctc_kd.ctc_kd_losses reads."""
    return {k: item[k] for k in FRAME_KEYS}


def _indices(store, ids: Iterable[str] | None) -> list[int] | None:
    if ids is None:
        return None
    pos = {u.id: i for i, u in enumerate(store.utts)}
    ids = list(ids)
    missing = [x for x in ids if x not in pos]
    if missing:
        raise KeyError(f"{len(missing)} ids not in the store, e.g. {missing[:3]}")
    return [pos[x] for x in ids]


def _prefetched(ds, batches: list[list[int]], threads: int = 4, ahead: int = 3):
    """Collated micro-batches; the decode of the next ones overlaps the model's work (soundfile and soxr drop the
    GIL). An undecodable utterance is dropped by the dataset and listed in item["dropped"]."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(ds.__getitem__, b) for b in batches[:ahead]]
        for j in range(len(batches)):
            item = futs[j].result()
            futs[j] = None
            if j + ahead < len(batches):
                futs.append(pool.submit(ds.__getitem__, batches[j + ahead]))
            yield item


def bad_audio_per_set(store, dropped: list[str]) -> dict[str, int]:
    if not dropped:
        return {}
    src = {u.id: u.source for u in store.utts}
    return dict(sorted(Counter(src.get(i, "?") for i in dropped).items()))


def store_text(store, ids: Iterable[str] | None = None) -> dict[str, dict]:
    """id -> {ref, hyp (the teacher's CTC hypothesis), cer (its per-utterance CER), truncated} of a frame store's rows
    (all, or `ids`), read from its index once: the caller keeps it (the trainer per run) rather than reading the index
    at every eval."""
    want = None if ids is None else set(ids)
    fr = store.frame()
    return {u.id: dict(ref=ref, hyp=hyp, cer=float(u.teacher_cer), truncated=bool(u.truncated))
            for u, ref, hyp in zip(store.utts, fr["ref"].tolist(), fr["hyp"].tolist()) if want is None or u.id in want}


def ctc_forward(model, featurizer, item: dict, device, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(log_probs (B,T,V) fp32, the student's valid frames per row (B,)) of one micro-batch: LogMel (fp32, outside
    autocast), the encoder under bf16 autocast when amp, the CTC head in fp32 (kitsune.ctc_student.ctc_log_probs)."""
    from kitsune.ctc_student import ctc_log_probs

    device = torch.device(device)
    wave, lengths = item["wave"].to(device, non_blocking=True), item["lengths"].to(device)
    with torch.autocast(device_type=device.type, enabled=False):
        feats, fmask = featurizer(wave, lengths)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return ctc_log_probs(model, feats, fmask)


# ------------------------------------------------------------------------------------------------ the pass


@torch.no_grad()
def ctc_eval_records(model, store, featurizer, device, batch_s: float = 400.0, *, ids: Iterable[str] | None = None,
                     amp: bool | None = None, decode: bool = True) -> tuple[pd.DataFrame, dict[str, list[int]], list]:
    """One pass over `ids` of a frame store (None: all of it), in eval_batches order: (raw, hyps, dropped). raw has one
    row per utterance: id, source, duration, n_tok (U, the CTC target tokens), n_frames, and sum_<term> of every
    ctc_kd_losses term (sum_kl = KL dense + KL blank); hyps maps id -> the greedy CTC ids (decode); dropped lists the
    ids whose audio could not be decoded."""
    from kitsune.ctc_student import greedy_ctc_ids

    device = torch.device(device)
    ds = dataset_for(store)
    batches = eval_batches(store.utts, batch_s, _indices(store, ids))
    recs, hyps, dropped = [], {}, []
    with eval_mode(model):
        for item in _prefetched(ds, batches):
            dropped += item["dropped"]
            n = len(item["ids"])
            if not n:
                continue
            lp, nf = ctc_forward(model, featurizer, item, device, _amp(device, amp))
            terms = ctc_kd_losses(lp, frame_batch(item), per_utt=True)
            v = {k: terms[k].detach().double().cpu().numpy() for k in SUMS}
            seqs = greedy_ctc_ids(lp, nf) if decode else None
            for r in range(n):
                rec = dict(id=item["ids"][r], source=item["sources"][r], duration=float(item["durations"][r]),
                           n_tok=int(item["n_tok"][r]), n_frames=int(item["n_frames"][r]),
                           sum_kl=float(v["kl_dense"][r] + v["kl_blank"][r]))
                rec.update({f"sum_{k}": float(v[k][r]) for k in SUMS})
                recs.append(rec)
                if seqs is not None:
                    hyps[item["ids"][r]] = seqs[r]
    return pd.DataFrame(recs), hyps, dropped


def summarise_ctc_tf(raw: pd.DataFrame, **extra) -> tuple[dict, pd.DataFrame]:
    """(summary, per_utt) of ctc_eval_records' raw rows: kitsune.evaluate.summarise_ctc_tf (its docstring defines every
    key; kitsune.evaluate.CTC_TF_COLUMNS the per_utt columns), after renaming the raw frame's n_frames to the
    sum_n_frames it reads - one summariser for the trainer's and 05's CTC evals. Per set and pooled ("all"): kl, ce
    (= ctc) per CTC target token, top1 (= argmax_agree) per valid frame, and the rest."""
    return ev.summarise_ctc_tf(raw.rename(columns={"n_frames": "sum_n_frames"}), **extra)


def greedy_frame(raw: pd.DataFrame, hyps: dict[str, list[int]], text: dict[str, dict], tokenizer) -> pd.DataFrame:
    """greedy_eval's per_utt columns (kitsune.evaluate) for the decoded rows of a pass: hyp = the student's greedy CTC
    text (kitsune.ctc_student.decode_ids), teacher_hyp = the teacher's stored ctc_hyp, CERs vs both, truncated False
    (a CTC decode never is), n_tok = the student's output tokens, teacher_cer / teacher_truncated the teacher's."""
    from kitsune.ctc_student import decode_ids
    from kitsune.text import cer as utt_cer

    recs = []
    for r in raw[["id", "source", "duration"]].itertuples(index=False):
        ids = [int(x) for x in hyps.get(r.id, [])]
        t = text.get(r.id)
        h = decode_ids(tokenizer, ids)
        ref, th = (t["ref"] or "", t["hyp"] or "") if t else ("", "")
        recs.append(dict(id=r.id, source=r.source, duration=float(r.duration), ref=ref, teacher_hyp=th, hyp=h,
                         cer_ref=utt_cer(h, ref), cer_teacher=utt_cer(h, th), truncated=False, n_tok=len(ids),
                         teacher_cer=t["cer"] if t else float("nan"), teacher_truncated=t["truncated"] if t else None,
                         hyp_ids=ids, has_teacher=t is not None))
    cols = ["id", "source", "duration", "ref", "teacher_hyp", "hyp", "cer_ref", "cer_teacher", "truncated", "n_tok",
            "teacher_cer", "teacher_truncated", "hyp_ids", "has_teacher"]
    return pd.DataFrame(recs, columns=cols)


def ctc_eval(model, store, featurizer, device, batch_s: float = 400.0, *, ids: Iterable[str] | None = None,
             tokenizer=None, text: dict[str, dict] | None = None, amp: bool | None = None,
             decode: bool = True) -> tuple[dict, pd.DataFrame, pd.DataFrame | None, dict]:
    """ctc_eval_records + its summaries: (tf_sum, tf_per_utt, greedy per_utt or None without decode, meta) - tf_sum
    as summarise_ctc_tf with n_bad_audio, bad_audio, bad_audio_per_set and wall_s (kitsune.evaluate.
    teacher_forced_eval's extras); meta = dict(wall_s, rtf, dropped, bad_audio_per_set) for the greedy summary
    (kitsune.evaluate.summarise_greedy(per_utt, **meta) gives greedy_eval's)."""
    t0 = time.time()
    raw, hyps, dropped = ctc_eval_records(model, store, featurizer, device, batch_s, ids=ids, amp=amp, decode=decode)
    wall = time.time() - t0
    bad = bad_audio_per_set(store, dropped)
    tf_sum, tf_df = summarise_ctc_tf(raw, n_bad_audio=len(dropped), bad_audio=dropped[:50], bad_audio_per_set=bad,
                                     wall_s=wall)
    gr_df = None
    if decode:
        if text is None:
            text = store_text(store, raw["id"].tolist() if len(raw) else [])
        gr_df = greedy_frame(raw, hyps, text, tokenizer)
    audio_s = float(raw["duration"].sum()) if len(raw) else 0.0
    meta = dict(n_bad_audio=len(dropped), bad_audio=dropped[:50], bad_audio_per_set=bad, wall_s=wall,
                rtf=wall / audio_s if audio_s else float("nan"))
    return tf_sum, tf_df, gr_df, meta


# ------------------------------------------------------------------------------------------------ objective


def gate_sums(tf: dict | None) -> dict | None:
    """A CTC teacher-forced summary's KL and CTC sums over its GATE sets (every evaluated set if none is one), as
    kitsune.evaluate.gate_kd_sums pools the AED ones: kl_sum, ctc_sum, n_tok (N_u), sets. None without a token."""
    sets = (tf or {}).get("sets") or {}
    gate = {s: d for s, d in sets.items() if s in GATE_SETS} or dict(sets)
    vt = {s: d for s, d in gate.items() if d.get("n_tok")}
    n = sum(int(d["n_tok"]) for d in vt.values())
    if not n:
        return None
    return dict(kl_sum=sum(float(d["kl"]) * int(d["n_tok"]) for d in vt.values()),
                ctc_sum=sum(float(d["ctc"]) * int(d["n_tok"]) for d in vt.values()), n_tok=n, sets=sorted(vt))


def frame_headline(head: dict, tf: dict | None) -> dict:
    """kitsune.evaluate.headline's record with val_top1 pooled per valid frame, the unit of a CTC set's top1 (the frame
    argmax agreement): headline pools the gate sets' top1 weighted by their target tokens, the AED unit, which would
    weight a CTC set by N_u instead of its frames. The pool is then every gate frame's agreement / the gate frames, as
    the per-set numbers are. A summary whose sets have no n_frames (an AED one) leaves the record as it was. In place;
    returns it."""
    sets = (tf or {}).get("sets") or {}
    gate = {s: d for s, d in sets.items() if s in GATE_SETS} or dict(sets)
    if "val_top1" not in head or not gate or not all("n_frames" in d for d in gate.values()):
        return head
    vf = {s: d for s, d in gate.items() if d.get("n_frames")}
    n = sum(int(d["n_frames"]) for d in vf.values())
    if n:
        head["val_top1"] = sum(float(d["top1"]) * int(d["n_frames"]) for d in vf.values()) / n
    return head


def combined_loss_ctc(tf: dict | None, w_kl: float, w_ctc: float) -> dict | None:
    """The CTC family's training objective on an eval (the combined loss's val / val_full points): (w_kl * sum KL +
    w_ctc * sum CTC) / N_u over the gate sets - kitsune.ctc_kd.ctc_kd_objective's expression over the eval's
    utterances instead of a step's. Returns value, kl, ctc (per token), the weights and gate_sums' record."""
    s = gate_sums(tf)
    if s is None:
        return None
    n = s["n_tok"]
    return dict(value=(float(w_kl) * s["kl_sum"] + float(w_ctc) * s["ctc_sum"]) / n, kl=s["kl_sum"] / n,
                ctc=s["ctc_sum"] / n, w_kl=float(w_kl), w_ctc=float(w_ctc), **s)


# ------------------------------------------------------------------------------------------------ teacher baselines


def load_parakeet_rows(parakeet_root, sets: Iterable[str], split: str = "eval") -> dict[str, list[dict]]:
    """set -> its parakeet_out/<set>/<split>-*.jsonl rows (id, hyp (TDT), ctc_hyp, ref, cer, ctc_cer, ...)."""
    out = {}
    for s in sets:
        rows = []
        for f in sorted((Path(parakeet_root) / s).glob(f"{split}-*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh if line.strip()]
        out[s] = rows
    return out


def ctc_teacher_baselines(parakeet_root, sets: Iterable[str] = GATE_SETS, check: bool = True) -> dict[str, dict]:
    """The CTC students' teacher baselines, recomputed from parakeet_out/<set>/eval-*.jsonl: per set the Parakeet CTC
    corpus CER (ctc_hyp vs ref; the teacher and ratio baseline) and mean CER, the TDT hypothesis's corpus CER (the
    off-the-shelf bar), rows, empty references, truncation, hours, and `prereg` / `prereg_source` (teacher_prereg: the
    PREREG.json number, or K6 while pending). With check=True a gate set whose recomputed CTC corpus CER is off that
    number by more than BASELINE_TOL raises ValueError (at trainer start-up, as kitsune.evaluate.teacher_baselines does
    for Cohere's): a changed eval set would silently move every ratio. A set without rows is FileNotFoundError."""
    from kitsune.evaluate import corpus_cer

    prereg, source = teacher_prereg()
    out = {}
    for s, rows in load_parakeet_rows(parakeet_root, sets).items():
        if not rows:
            raise FileNotFoundError(f"no parakeet_out/{s}/eval-*.jsonl under {parakeet_root}")
        refs = [r["ref"] for r in rows]
        c = corpus_cer([r["ctc_hyp"] for r in rows], refs)
        t = corpus_cer([r["hyp"] for r in rows], refs)
        out[s] = dict(system=TEACHER_SYSTEM, cer_corpus=c["cer"], cer_mean=float(np.mean([r["ctc_cer"] for r in rows])),
                      tdt_cer_corpus=t["cer"], n=len(rows), n_empty_ref=c["n_empty_ref"],
                      trunc_rate=float(np.mean([bool(r.get("truncated")) for r in rows])),
                      hours=float(sum(r["duration"] for r in rows)) / 3600, prereg=prereg.get(s),
                      prereg_source=source if s in prereg else None, tdt_card=PARAKEET_TDT_CER.get(s))
        if check and s in prereg and abs(c["cer"] - prereg[s]) > BASELINE_TOL:
            raise ValueError(f"{s}: Parakeet CTC corpus CER {100 * c['cer']:.3f} % differs from the pre-registered "
                             f"{100 * prereg[s]:.2f} % ({source}) by more than {100 * BASELINE_TOL:.2f} pp")
    return out


# ------------------------------------------------------------------------------------------------ verdict


def verdict_teacher(final: dict | None) -> dict[str, float]:
    """The pending fallback of kitsune.evaluate.verdict(..., family="ctc")'s teacher: for each gate set of the final
    numbers WITHOUT a same-ids Parakeet CTC CER (their teacher_cer_ref_corpus; the teacher text of a frame store is
    ctc_hyp, so the trainer's always have one), teacher_prereg's number - the registered one, or K6 while PREREG is
    pending, where the verdict alone would leave the set unjudged. Sets with a same-ids CER are left to the verdict
    ("same ids"). Pass it as results["teacher"]."""
    out = {}
    sets = (final or {}).get("sets") or {}
    prereg, _ = teacher_prereg()
    for s in GATE_SETS:
        tc = (sets.get(s) or {}).get("teacher_cer_ref_corpus")
        if s in sets and not (tc is not None and math.isfinite(float(tc))) and s in prereg:
            out[s] = prereg[s]
    return out


def relabel_verdict(verdict: dict) -> dict:
    """kitsune.evaluate.verdict(..., family="ctc")'s per-set records named for their teacher: teacher_system
    "parakeet-ctc" (the verdict's own teacher_prereg / baseline_drift are kitsune.evaluate's PARAKEET_CTC_CER_PREREG,
    None while pending; the top level says family, teacher and teacher_prereg_status). In place; returns it."""
    for d in (verdict.get("sets") or {}).values():
        d["teacher_system"] = TEACHER_SYSTEM
    return verdict


__all__ = ["BASELINE_TOL", "CTC_BLANK", "GATE_SETS", "PARAKEET_CTC_CER_K6", "PARAKEET_TDT_CER", "TEACHER_SYSTEM",
           "combined_loss_ctc", "ctc_eval", "ctc_eval_records", "ctc_forward", "ctc_teacher_baselines", "frame_batch",
           "frame_headline", "gate_sums", "greedy_frame", "relabel_verdict", "store_text", "summarise_ctc_tf",
           "teacher_prereg", "verdict_teacher"]
