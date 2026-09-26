"""A synthetic label root with BOTH teachers for the size study's selection (scripts/make_selection.py, study recipe).

    from fixtures_study import make_study_corpus
    st = make_study_corpus(tmp_path)          # fixtures.make_fake_corpus + parakeet_out + the laptop's kotoba file

On top of make_fake_corpus (Cohere teacher_out, second_out, audio; canonical source names so kitsune.extent plans
them) it writes
  parakeet_out/<source>/<stem>.{npz,jsonl} + meta.json   in kitsune/parakeet_targets.py's exact FORMAT, packed by its
                                                         own pack_shard / write_shard (the label box's writer)
  kotoba/galgame_eval.jsonl                              the laptop's second_out/galgame/eval-*.jsonl shape: {id, hyp2,
                                                         model2, agree, cer2} with kotoba-whisper hypotheses
and returns the ground truth: which rows it made to fail which study rule, so a test checks the selection against
what was WRITTEN. Every special row is picked among "clean" train rows (labelled, not truncated, agree exactly 0.0),
so it passes F0 under any threshold and fails exactly the rule it was made for:
  missing        train rows left out of parakeet_out            -> not_in_parakeet
  disagree       Parakeet's TDT hyp shares nothing with Cohere's -> f1a_disagree
  dup_ref        the Cohere jsonl `ref` set to an eval reference of >= 15 characters -> eval_dup
  dup_hyp        the Cohere jsonl `hyp` set to one (the Parakeet hyp follows, so F1a passes) -> eval_dup
  short_dup      `ref` set to an eval reference of 10 characters (under the 15-character bar) -> kept
  infeasible     the npz's stored n_frames one short of U + repeats of the stored CTC path -> ctc_infeasible
Other truth: ctc_need (U + adjacent repeats of every row's CTC target), parakeet_only (rows the Cohere pass skipped
but Parakeet labelled), neutral (galgame hold-out rows whose kotoba cer <= 0.5 with a non-empty reference) and
empty_ref (a galgame hold-out row whose reference was emptied; its kotoba hyp is empty too, cer 0, still not neutral).
The Parakeet CTC path of a row: its U tokens (about one per 3 frames, 20 % repeats of the previous token) on every
other frame, a token sometimes held for two frames (the collapse), blank elsewhere. Everything derives from `seed`.
"""
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import FakeCorpus, _KANA, make_fake_corpus  # noqa: E402

from kitsune import parakeet_targets as pt  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402
from kitsune.text import normalize_ja  # noqa: E402

SETTINGS = {"k_tdt": 4, "k_ctc": 8, "max_symbols": 10, "ctc_dense_thr": 0.95}
FRAME_S = 0.08
STUDY_SOURCES = {"reazon_small": (40, "train"), "emilia_yodas": (32, "train"),
                 "galgame": [(40, "train"), (16, "eval")], "eval_jsut": (12, "eval"), "eval_cv8": (8, "eval"),
                 "eval_reazon": (8, "eval")}
LONG_REF = "あいうえおかきくけこさしすせそたちつてと"  # 20 characters: over the 15-character dedup bar
SHORT_REF = "なにぬねのはひふへほ"  # 10 characters: under it


@dataclass
class FakeStudy:
    fc: FakeCorpus
    parakeet_out: Path
    kotoba: Path
    missing: set = field(default_factory=set)
    disagree: set = field(default_factory=set)
    dup_ref: set = field(default_factory=set)
    dup_hyp: set = field(default_factory=set)
    short_dup: set = field(default_factory=set)
    infeasible: set = field(default_factory=set)
    missing_eval: set = field(default_factory=set)
    parakeet_only: set = field(default_factory=set)
    neutral: set = field(default_factory=set)
    empty_ref: set = field(default_factory=set)
    ctc_need: dict = field(default_factory=dict)
    ctc_tokens: dict = field(default_factory=dict)

    @property
    def teacher_out(self) -> Path:
        return self.fc.teacher_out

    @property
    def second_out(self) -> Path:
        return self.fc.second_out

    @property
    def data(self) -> Path:
        return self.fc.data


def _rewrite_jsonl(path: Path, changes: dict[str, dict]):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r.update(changes.get(r["id"], {}))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _ctc_utt(rng: np.random.Generator, T: int) -> tuple[np.ndarray, list[int]]:
    """col0 per frame (the CTC argmax path) and its greedy target: U tokens on every other frame, 20 % of them a repeat
    of the previous token (a blank frame always separates them), some held for two frames when the next differs."""
    U = max(1, T // 3)
    toks: list[int] = []
    for _ in range(U):
        toks.append(toks[-1] if toks and rng.random() < 0.2 else int(rng.integers(0, pt.BLANK)))
    col0 = np.full(T, pt.BLANK, dtype=np.int64)
    for j, t in enumerate(toks):
        col0[2 * j] = t
        nxt = toks[j + 1] if j + 1 < U else None
        if nxt != t and 2 * j + 1 < T and rng.random() < 0.3:
            col0[2 * j + 1] = t  # held: collapses to one token
    return col0, toks


def _parakeet_utt(u, rng: np.random.Generator, col0: np.ndarray, n_frames: int) -> dict:
    T = len(col0)
    k_ctc, k_tdt = SETTINGS["k_ctc"], SETTINGS["k_tdt"]
    dense = np.flatnonzero(col0 != pt.BLANK)
    blank_lp = np.where(col0 != pt.BLANK, math.log(0.1), math.log(0.99)).astype(np.float16)
    topk_idx = np.empty((len(dense), k_ctc), dtype=np.int64)
    for r, f in enumerate(dense):
        others = [x for x in rng.choice(pt.BLANK, size=k_ctc + 2, replace=False) if x != col0[f]][: k_ctc - 2]
        topk_idx[r] = [col0[f], pt.BLANK, *others]
    lp_row = np.log([0.8, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001][:k_ctc])
    tdt_idx = np.empty((T, k_tdt), dtype=np.int64)
    for t in range(T):
        others = [x for x in rng.choice(pt.BLANK, size=k_tdt + 2, replace=False) if x != col0[t]][: k_tdt - 1]
        tdt_idx[t] = [col0[t], *others]
    return dict(id=u.id, duration=u.duration, n_frames=n_frames, truncated=False,
                step_frame=np.arange(T), step_dur=np.ones(T, dtype=np.int64), step_forced=np.zeros(T, dtype=bool),
                tdt_topk_idx=tdt_idx, tdt_topk_lp=np.tile(np.log([0.9, 0.05, 0.03, 0.02][:k_tdt]), (T, 1)),
                tdt_dur_lp=np.tile(np.log([0.1, 0.6, 0.2, 0.05, 0.05]), (T, 1)),
                ctc_blank_lp=blank_lp, ctc_dense_frame=dense, ctc_topk_idx=topk_idx,
                ctc_topk_lp=np.tile(lp_row, (len(dense), 1)))


def make_study_corpus(root, *, seed: int = 0, sources: dict | None = None, n_missing: int = 2, n_disagree: int = 3,
                      n_dup: int = 1, n_infeasible: int = 2, missing_eval: int = 0) -> FakeStudy:
    root = Path(root)
    fc = make_fake_corpus(root, sources or STUDY_SOURCES, rows_per_shard=8, seed=seed,
                          truncated={"reazon_small": 2, "galgame": 2}, null_agree={"emilia_yodas": 1},
                          no_second=("eval_jsut", "eval_cv8", "eval_reazon"), teacher_skipped={"reazon_small": 1},
                          dur_range=(0.6, 6.0))
    rng = np.random.default_rng([seed, 7])
    st = FakeStudy(fc, root / "parakeet_out", root / "kotoba" / "galgame_eval.jsonl")
    utts = list(fc.utts.values())
    clean = [u.id for u in utts if u.split == "train" and u.has_teacher and not u.truncated and u.agree == 0.0]
    pick = iter(rng.permutation(clean).tolist())

    def take(n: int) -> set:
        return {next(pick) for _ in range(n)}

    st.missing, st.disagree, st.infeasible = take(n_missing), take(n_disagree), take(n_infeasible)
    st.dup_ref, st.dup_hyp, st.short_dup = take(n_dup), take(n_dup), take(1)

    # the eval references the duplicates copy: one long (>= 15 chars), one short, both in eval_jsut
    jsut = [u for u in utts if u.source == "eval_jsut" and u.has_teacher]
    jsut[0].text, jsut[1].text = LONG_REF, SHORT_REF
    changes: dict[str, dict] = {jsut[0].id: {"ref": LONG_REF}, jsut[1].id: {"ref": SHORT_REF}}
    for i in st.dup_ref:
        fc.utts[i].text = "「" + LONG_REF + "」"  # the normalised text is equal, the raw text is not
        changes[i] = {"ref": fc.utts[i].text}
    for i in st.dup_hyp:
        fc.utts[i].hyp = LONG_REF + "。"
        changes[i] = {"hyp": fc.utts[i].hyp}
    for i in st.short_dup:
        fc.utts[i].text = SHORT_REF
        changes[i] = {"ref": SHORT_REF}
    gal_eval = [u for u in utts if u.source == "galgame" and u.split == "eval" and u.has_teacher]
    gal_eval[0].text = ""
    changes[gal_eval[0].id] = {"ref": ""}
    st.empty_ref = {gal_eval[0].id}
    for p in sorted(fc.teacher_out.glob("*/*.jsonl")):
        _rewrite_jsonl(p, changes)

    evals = [u.id for u in utts if u.split == "eval" and u.has_teacher and u.source != "galgame"]
    st.missing_eval = set(rng.choice(evals, size=missing_eval, replace=False).tolist()) if missing_eval else set()

    # parakeet_out: one shard per data shard (the teacher's stems), in data-shard row order
    shards: dict[tuple[str, str], list] = {}
    for u in utts:
        if u.has_audio:
            shards.setdefault((u.source, u.stem), []).append(u)
    for (source, stem), rows in sorted(shards.items()):
        packed, jrows = [], []
        for u in rows:
            if not u.has_teacher and not (source == "reazon_small" and u.split == "train"):
                continue
            if u.id in st.missing or u.id in st.missing_eval:
                continue
            if not u.has_teacher:
                st.parakeet_only.add(u.id)
            T = max(4, math.ceil(u.duration / FRAME_S))
            col0, toks = _ctc_utt(rng, T)
            need = len(toks) + sum(a == b for a, b in zip(toks, toks[1:]))
            st.ctc_need[u.id], st.ctc_tokens[u.id] = need, toks
            n_frames = need - 1 if u.id in st.infeasible else T
            packed.append(_parakeet_utt(u, rng, col0, n_frames))
            # a disagreeing Parakeet hyp shares no character with Cohere's kana: CER >= 1 against its own length
            phyp = "z" * max(8, len(normalize_ja(u.hyp))) if u.id in st.disagree else u.hyp
            jrows.append(dict(id=u.id, hyp=phyp, ctc_hyp=phyp, ref=u.text, cer=round(cer_fn(phyp, u.text), 4),
                              ctc_cer=round(cer_fn(phyp, u.text), 4), duration=round(u.duration, 3),
                              n_tok=len(toks), n_steps=T, n_frames=n_frames, n_forced=0, truncated=False))
        n_scanned = sum(1 for u in fc.utts.values() if u.source == source and u.stem == stem)
        arrays = pt.pack_shard(packed, settings=SETTINGS, n_scanned=n_scanned,
                               shard_ids_sha=ids_sha256([u.id for u in rows]))
        pt.write_shard(st.parakeet_out / source, stem, arrays, jrows)
    (st.parakeet_out / "meta.json").write_text(json.dumps(dict(format_version=pt.FORMAT_VERSION, **SETTINGS)),
                                               encoding="utf-8")

    # the laptop's kotoba file of the galgame hold-out: every other row right, the rest garbage
    st.kotoba.parent.mkdir(parents=True, exist_ok=True)
    with open(st.kotoba, "w", encoding="utf-8") as f:
        for j, u in enumerate(gal_eval):
            hyp2 = "" if u.id in st.empty_ref else u.text if j % 2 == 0 else "".join(rng.choice(_KANA, size=12))
            c = cer_fn(hyp2, u.text)
            if normalize_ja(u.text) and c <= 0.5:
                st.neutral.add(u.id)
            f.write(json.dumps(dict(id=u.id, hyp2=hyp2, model2="kotoba-tech/kotoba-whisper-v2.0", agree=0.0,
                                    cer2=round(c, 4)), ensure_ascii=False) + "\n")
    return st
