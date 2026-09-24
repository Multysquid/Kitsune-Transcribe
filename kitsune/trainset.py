"""Training/eval data for distillation: selection -> packed on-disk stores -> step plans -> micro-batches.

Why this shape:
- The index comes from the selection (scripts/make_selection.py), which is built from teacher_out/*.npz: only those
  rows have targets. data/manifest.jsonl also lists shards with no teacher output (galgame train-00096+).
- Rows are joined to their audio BY ID, never by shard stem + row number: an interrupted or re-run ingest (e.g. the
  rebuild on the training box) renumbers shards. A selected row whose audio cannot be found is dropped and logged.
- Everything a DataLoader worker reads is packed into a few flat files under the cache dir and np.memmap'ed lazily
  inside each worker. The Dataset itself is then a few small arrays, cheap to pickle under the `spawn` start method
  (Windows, and Linux by choice), all workers share one page-cache copy, and "read utterance i" is two slices
  instead of a parquet row-group read.
- One item is a whole MICRO-BATCH (a list of store indices), so padding/collation happens in the worker. The step
  planner decides which utterances form a micro-batch (bounded padded audio seconds, i.e. bounded activation memory)
  and which micro-batches form an optimizer step (~fixed real audio seconds, i.e. a steady gradient scale).

Cache layout (<cache_dir>/, written by build_stores, atomic per file, stores.json last):
  stores.json            fingerprint + build stats; a matching fingerprint makes build_stores a no-op
  index.parquet          one row per utterance in store order: the Utt fields + ref, hyp (teacher text)
  audio.bin              uint8, the original container bytes (FLAC/OGG/WAV/MP3) back to back
  audio_offsets.npy      (n+1,) int64   utterance i is audio.bin[off[i]:off[i+1]]
  targets_tokens.npy     (N,)   int16   greedy teacher tokens incl. the final EOS (absent only if truncated)
  targets_topk_idx.npy   (N,k)  int16   column 0 == tokens
  targets_topk_lp.npy    (N,k)  float16 teacher log-probs of those ids
  targets_offsets.npy    (n+1,) int64   utterance i owns target rows off[i]:off[i+1]

Teacher step t of an utterance (tokens[t]) is predicted by the decoder output at position len(prompt)-1+t when the
decoder input is prompt + tokens[:-1]; see AudioBatchDataset for the exact batch contract.
"""
import hashlib
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from kitsune.audio import decode_audio
from kitsune.store import fsync_path

PROMPT = [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13]  # teacher decoder prompt (ja, pnc, noitn, ...); see teacher_out/meta.json
EOS, PAD = 3, 2
EVAL_SETS = ("eval_jsut", "eval_cv8", "eval_reazon")
FORMAT_VERSION = 1  # bump when the cache layout changes: invalidates every existing cache


@dataclass(slots=True)
class Utt:
    """One utterance of a store. The first seven fields index the shared files; the rest are selection columns."""

    id: str
    source: str
    duration: float
    n_tok: int  # target positions T (teacher tokens incl. EOS), >= 1
    audio_off: int  # byte offset into audio.bin
    audio_len: int
    tok_off: int  # row offset into targets_*.npy
    row: int = -1  # row in index.parquet
    split: str = "train"
    agree: float = float("nan")  # CER(teacher hyp, second opinion); NaN if unknown
    teacher_cer: float = float("nan")
    truncated: bool = False
    in_probe: bool = False
    in_greedy_subset: bool = False


_UTT_COLS = [f.name for f in fields(Utt)]


@dataclass
class Stores:
    """A built cache (or a subset of one: same files, fewer utts). Indices everywhere are positions in `utts`."""

    cache_dir: Path
    utts: list[Utt]
    info: dict = field(default_factory=dict)
    _ds: "AudioBatchDataset | None" = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.utts)

    def _dataset(self) -> "AudioBatchDataset":
        if self._ds is None:
            self._ds = AudioBatchDataset(self)
        return self._ds

    def wave(self, i: int) -> np.ndarray:
        """Decoded 16 kHz mono float32 audio of utts[i]. With targets() this is the store protocol kitsune.evaluate
        uses. Maps the cache files into this process (on Windows a mapped cache cannot be rebuilt in place)."""
        return decode_audio(self._dataset().audio_bytes(i))

    def targets(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tokens (T,) int16, topk_idx (T,k) int16, topk_logprob (T,k) float16) of utts[i], as in the npz."""
        return self._dataset().targets(i)

    def indices(self, *, source: str | None = None, split: str | None = None, in_probe: bool | None = None,
                in_greedy_subset: bool | None = None) -> list[int]:
        return [i for i, u in enumerate(self.utts)
                if (source is None or u.source == source) and (split is None or u.split == split)
                and (in_probe is None or u.in_probe == in_probe)
                and (in_greedy_subset is None or u.in_greedy_subset == in_greedy_subset)]

    def subset(self, indices: Sequence[int]) -> "Stores":
        return Stores(self.cache_dir, [self.utts[i] for i in indices], self.info)

    def frame(self) -> pd.DataFrame:
        """index.parquet rows for `utts`, in `utts` order (includes the ref / teacher hyp text)."""
        df = pd.read_parquet(self.cache_dir / "index.parquet")
        return df.iloc[[u.row for u in self.utts]].reset_index(drop=True)

    @property
    def hours(self) -> float:
        return sum(u.duration for u in self.utts) / 3600


# ---------------------------------------------------------------------------------------------------------- build


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _teacher_meta(teacher_root: Path) -> dict:
    p = Path(teacher_root) / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return dict(prompt=[int(x) for x in meta.get("decoder_prompt_ids", PROMPT)], eos=int(meta.get("eos_token_id", EOS)),
                pad=int(meta.get("pad_token_id", PAD)), k=int(meta.get("k", 16)))


def read_selection(selection_path: Path, sources: Sequence[str], splits: Sequence[str]) -> pd.DataFrame:
    """Kept rows of the selection for these sources/splits, in selection order."""
    sel = pd.read_parquet(selection_path)
    sel = sel[sel["keep"] & sel["source"].isin(list(sources)) & sel["split"].isin(list(splits))].reset_index(drop=True)
    if sel["id"].duplicated().any():
        raise ValueError(f"{selection_path}: duplicate ids, e.g. {sel['id'][sel['id'].duplicated()].iloc[0]}")
    if (sel["n_tok"] < 1).any():
        raise ValueError(f"{selection_path}: rows with no target tokens, e.g. {sel['id'][sel['n_tok'] < 1].iloc[0]}")
    return sel


def _write_npy(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _cache_complete(cache_dir: Path) -> bool:
    names = ["index.parquet", "audio.bin", "audio_offsets.npy", "targets_tokens.npy", "targets_topk_idx.npy",
             "targets_topk_lp.npy", "targets_offsets.npy"]
    return all((cache_dir / n).exists() for n in names)


def build_stores(selection_path, data_root, teacher_root, cache_dir, sources: Sequence[str], splits: Sequence[str],
                 *, ids: Iterable[str] | None = None, log: Callable[[str], None] = print) -> Stores:
    """Pack the kept selection rows of `sources` x `splits` into <cache_dir> (layout in the module docstring).
    `ids` restricts the store further (e.g. a seeded calibration sample or a small smoke-run subset), so a laptop
    need not copy all of a source's audio.

    Idempotent: nothing is rebuilt if <cache_dir>/stores.json carries the same fingerprint (format version,
    selection file hash, sources, splits, prompt, k, npz sizes) and every shard that contributed audio still has its
    size. New shards (a download still appending to data/) only force a rebuild if the last build dropped rows for
    missing audio. Store order is the order in which the audio was found (sources in the given order, shards sorted
    by name, row order within).
    Selected rows without audio are dropped and reported in info["dropped"]; a selected row without teacher output
    is an error (the selection was made from a different teacher_out).
    """
    selection_path, data_root, teacher_root, cache_dir = map(Path, (selection_path, data_root, teacher_root, cache_dir))
    sources, splits = list(sources), list(splits)
    sel = read_selection(selection_path, sources, splits)
    if ids is not None:
        ids = set(ids)
        sel = sel[sel["id"].isin(ids)].reset_index(drop=True)
    if sel.empty:
        raise ValueError(f"{selection_path}: no kept rows for sources={sources} splits={splits}")
    meta = _teacher_meta(teacher_root)
    shard_files = [p for s in sources for sp in splits for p in sorted((data_root / "shards" / s).glob(f"{sp}-*.parquet"))]
    npz_files = [teacher_root / f"{f}.npz" for f in sorted(set(sel["teacher_file"]))]
    missing_npz = [p for p in npz_files if not p.exists()]
    if missing_npz:
        raise FileNotFoundError(f"teacher output missing for selected rows: {missing_npz[:3]}")

    fp_src = dict(version=FORMAT_VERSION, selection=_sha256(selection_path), sources=sources, splits=splits,
                  prompt=meta["prompt"], k=meta["k"],
                  ids=None if ids is None else hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                  npz=[(p.relative_to(teacher_root).as_posix(), p.stat().st_size) for p in npz_files])
    fingerprint = hashlib.sha256(json.dumps(fp_src, sort_keys=True).encode()).hexdigest()
    shard_size = {p.relative_to(data_root).as_posix(): p.stat().st_size for p in shard_files}
    info_path = cache_dir / "stores.json"
    if info_path.exists() and _cache_complete(cache_dir):
        old = json.loads(info_path.read_text(encoding="utf-8"))
        same_audio = all(shard_size.get(r) == sz for r, sz in old.get("shards", {}).items()) and (
            old["dropped"]["no_audio"]["n"] == 0 or sorted(shard_size) == old.get("all_shards"))
        if old.get("fingerprint") == fingerprint and same_audio:
            st = load_stores(cache_dir)
            log(f"stores: reusing {cache_dir} ({len(st)} utts, {st.hours:.2f} h, built {old.get('created')})")
            return st

    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path.unlink(missing_ok=True)  # invalidate first: a crash mid-build must not leave a cache that looks valid

    # 1. audio, joined by id, written in the order it is found
    want = dict(zip(sel["id"].tolist(), range(len(sel))))
    order, lens, used = [], [], {}
    audio_tmp = cache_dir / "audio.bin.tmp"
    with open(audio_tmp, "wb") as f:
        for path in shard_files:
            if not want:
                break
            pf = pq.ParquetFile(path)
            for g in range(pf.num_row_groups):
                ids = pf.read_row_group(g, columns=["id"]).column("id").to_pylist()
                hits = [(j, want.pop(x)) for j, x in enumerate(ids) if x in want]  # pop: first copy of an id wins
                if not hits:
                    continue
                audio = pf.read_row_group(g, columns=["audio"]).column("audio")
                used[path.relative_to(data_root).as_posix()] = shard_size[path.relative_to(data_root).as_posix()]
                for j, r in hits:
                    b = audio[j].as_py()
                    f.write(b)
                    order.append(r)
                    lens.append(len(b))
        f.flush()
        os.fsync(f.fileno())
    t_audio = time.time() - t0
    lost = sel.iloc[sorted(want.values())]
    if not order:
        raise ValueError(f"no audio found under {data_root / 'shards'} for any of the {len(sel)} selected rows")

    idx = sel.iloc[order].reset_index(drop=True)
    n = len(idx)
    audio_offsets = np.concatenate([[0], np.cumsum(np.array(lens, dtype=np.int64))])
    n_tok = idx["n_tok"].to_numpy(np.int64)
    tok_offsets = np.concatenate([[0], np.cumsum(n_tok)])
    N, k = int(tok_offsets[-1]), meta["k"]

    # 2. targets, joined by id, written in store order
    tokens = np.empty(N, np.int16)
    topk_idx = np.empty((N, k), np.int16)
    topk_lp = np.empty((N, k), np.float16)
    ref = np.empty(n, dtype=object)
    hyp = np.empty(n, dtype=object)
    filled = np.zeros(n, dtype=bool)
    row_of = dict(zip(idx["id"].tolist(), range(n)))
    for npz in npz_files:
        z = np.load(npz)
        if [int(x) for x in z["prompt"]] != meta["prompt"] or int(z["k"]) != k:
            raise ValueError(f"{npz}: prompt/k differ from {teacher_root / 'meta.json'}")
        z_ids, z_off = z["ids"].tolist(), z["tok_offsets"]
        z_tok, z_idx, z_lp = z["tokens"], z["topk_idx"], z["topk_logprob"]
        if not np.array_equal(z_idx[:, 0], z_tok):
            raise ValueError(f"{npz}: topk_idx[:, 0] != tokens")
        texts = {}
        with open(npz.with_suffix(".jsonl"), encoding="utf-8") as fj:
            for line in fj:
                if line.strip():
                    r = json.loads(line)
                    texts[r["id"]] = (r["ref"], r["hyp"])
        for j, x in enumerate(z_ids):
            r = row_of.get(x)
            if r is None:
                continue
            s, e = int(z_off[j]), int(z_off[j + 1])
            if e - s != n_tok[r]:
                raise ValueError(f"{x}: {e - s} tokens in {npz.name}, selection says {n_tok[r]}")
            o = int(tok_offsets[r])
            tokens[o:o + e - s] = z_tok[s:e]
            topk_idx[o:o + e - s] = z_idx[s:e]
            topk_lp[o:o + e - s] = z_lp[s:e]
            ref[r], hyp[r] = texts.get(x, (None, None))
            filled[r] = True
    if not filled.all():
        raise ValueError(f"{(~filled).sum()} selected rows not found in their teacher_file, e.g. "
                         f"{idx['id'][~filled].iloc[0]} - selection and teacher_out do not match")

    # 3. publish: data files first, index + stores.json last
    _write_npy(cache_dir / "audio_offsets.npy", audio_offsets)
    _write_npy(cache_dir / "targets_offsets.npy", tok_offsets)
    _write_npy(cache_dir / "targets_tokens.npy", tokens)
    _write_npy(cache_dir / "targets_topk_idx.npy", topk_idx)
    _write_npy(cache_dir / "targets_topk_lp.npy", topk_lp)
    audio_tmp.replace(cache_dir / "audio.bin")

    agree = idx["agree"].astype("float64").to_numpy() if "agree" in idx else np.full(n, np.nan)
    cols = dict(
        id=idx["id"].tolist(), source=idx["source"].tolist(), duration=idx["duration"].to_numpy(np.float32),
        n_tok=n_tok, audio_off=audio_offsets[:-1], audio_len=np.diff(audio_offsets), tok_off=tok_offsets[:-1],
        row=np.arange(n, dtype=np.int64), split=idx["split"].tolist(), agree=agree.astype(np.float32),
        teacher_cer=idx["teacher_cer"].to_numpy(np.float32), truncated=idx["truncated"].to_numpy(bool),
        in_probe=idx["in_probe"].to_numpy(bool), in_greedy_subset=idx["in_greedy_subset"].to_numpy(bool),
        ref=list(ref), hyp=list(hyp),
    )
    tmp = cache_dir / "index.parquet.tmp"
    pq.write_table(pa.table(cols), tmp)
    fsync_path(tmp)
    tmp.replace(cache_dir / "index.parquet")

    per_source = {s: dict(utts=int((idx["source"] == s).sum()), hours=float(idx["duration"][idx["source"] == s].sum() / 3600))
                  for s in sources}
    info = dict(
        fingerprint=fingerprint, format_version=FORMAT_VERSION, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        selection=str(selection_path), selection_sha256=fp_src["selection"], sources=sources, splits=splits,
        prompt=meta["prompt"], eos=meta["eos"], pad=meta["pad"], k=k,
        n_utts=n, hours=float(idx["duration"].sum() / 3600), n_targets=N, audio_bytes=int(audio_offsets[-1]),
        target_bytes=int(tokens.nbytes + topk_idx.nbytes + topk_lp.nbytes), per_source=per_source,
        dropped=dict(no_audio=dict(n=len(lost), hours=float(lost["duration"].sum() / 3600),
                                   by_source=lost["source"].value_counts().to_dict(), ids=lost["id"].tolist()[:50])),
        build_s=round(time.time() - t0, 2), audio_s=round(t_audio, 2),
        shards=used, all_shards=sorted(shard_size) if len(lost) else None,
    )
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    fsync_path(tmp)
    tmp.replace(info_path)
    log(f"stores: built {cache_dir} in {info['build_s']:.1f} s (audio pass {t_audio:.1f} s): {n} utts, "
        f"{info['hours']:.2f} h, audio {info['audio_bytes'] / 2**30:.2f} GiB, {N} target steps "
        f"({info['target_bytes'] / 2**20:.1f} MiB)"
        + (f"; DROPPED {len(lost)} rows without audio ({info['dropped']['no_audio']['hours']:.3f} h, "
           f"e.g. {lost['id'].iloc[0]})" if len(lost) else ""))
    return load_stores(cache_dir)


def load_stores(cache_dir) -> Stores:
    """Open an already built cache (no source data needed)."""
    cache_dir = Path(cache_dir)
    info = json.loads((cache_dir / "stores.json").read_text(encoding="utf-8"))
    df = pq.read_table(cache_dir / "index.parquet", columns=_UTT_COLS).to_pandas()
    utts = [Utt(*r) for r in df[_UTT_COLS].itertuples(index=False, name=None)]
    for u in utts:  # numpy scalars -> python, so Utt pickles small and compares cleanly
        u.duration, u.n_tok, u.audio_off, u.audio_len, u.tok_off, u.row = (
            float(u.duration), int(u.n_tok), int(u.audio_off), int(u.audio_len), int(u.tok_off), int(u.row))
        u.agree, u.teacher_cer = float(u.agree), float(u.teacher_cer)
        u.truncated, u.in_probe, u.in_greedy_subset = bool(u.truncated), bool(u.in_probe), bool(u.in_greedy_subset)
    return Stores(cache_dir, utts, info)


def eval_store(selection_path, data_root, teacher_root, cache_dir, eval_sets: Sequence[str] = EVAL_SETS,
               *, log: Callable[[str], None] = print) -> Stores:
    """All kept rows of the eval sets as one store: teacher-forced eval uses every row, greedy eval the rows with
    `in_greedy_subset` (or all of them for the final full decode). Use a cache_dir separate from the train store."""
    return build_stores(selection_path, data_root, teacher_root, cache_dir, eval_sets, ["eval"], log=log)


# -------------------------------------------------------------------------------------------------------- planner


def pack_micro_batches(order: Sequence[int], dur: np.ndarray, cap_s: float, dec_len: np.ndarray | None = None,
                       cap_tokens: int | None = None) -> list[list[int]]:
    """Greedily cut `order` (sorted by duration, longest first) into micro-batches with max_dur * n <= cap_s
    (padded audio seconds) and, if given, max_dec_len * n <= cap_tokens (padded decoder positions). An utterance
    longer than cap_s gets a micro-batch of its own."""
    out, cur, md, ml = [], [], 0.0, 0
    for i in order:
        d = float(dur[i])
        l = int(dec_len[i]) if dec_len is not None else 0
        if cur:
            m_d, m_l = max(md, d), max(ml, l)
            if m_d * (len(cur) + 1) > cap_s or (cap_tokens is not None and m_l * (len(cur) + 1) > cap_tokens):
                out.append(cur)
                cur = []
        if not cur:
            md, ml = d, l
        else:
            md, ml = max(md, d), max(ml, l)
        cur.append(int(i))
    if cur:
        out.append(cur)
    return out


def eval_batches(utts: Sequence[Utt], batch_s: float, indices: Sequence[int] | None = None) -> list[list[int]]:
    """Deterministic duration-sorted micro-batches (padded seconds <= batch_s) over `indices` (default: all).
    For make_loader, wrap each as a one-micro-batch step: make_loader(ds, [[b] for b in eval_batches(...)], ...)."""
    idx = np.arange(len(utts)) if indices is None else np.asarray(indices, dtype=np.int64)
    dur = np.array([u.duration for u in utts], dtype=np.float64)
    order = idx[np.argsort(-dur[idx], kind="stable")]
    return pack_micro_batches(order, dur, batch_s)


class StepPlanner:
    """Epoch plans: steps -> micro-batches -> utterance indices (positions in `utts`).

    Per epoch: repeat utterances by `weights` (None = natural mix, each once), shuffle, cut into pools of about
    `pool_micro` micro-batches' worth of audio, sort each pool by duration, greedily pack micro-batches by PADDED
    seconds (max_dur * n <= micro_audio_s; decoder positions max_dec_len * n <= max_micro_tokens if set), shuffle the
    micro-batches, group them into steps of ~step_audio_s REAL seconds (each step within +-half a micro-batch of it,
    except one remainder step per epoch), then shuffle the steps. Pools keep sorting local, so batches are
    duration-homogeneous (little padding) while the epoch order stays random. Deterministic per (seed, epoch).

    Utterances whose decoder input (prompt_len - 1 + n_tok) exceeds max_dec_len are excluded (self.n_excluded).
    Position for resume is (epoch, step_in_epoch): the next step to run. The trainer calls step_done(epoch, step)
    after each optimizer step and saves state_dict() with its checkpoint; make_loader(ds, planner, ...) then starts
    from that position. epoch_stats[e] holds plan_stats() of every epoch planned so far.
    """

    def __init__(self, utts: Sequence[Utt], step_audio_s: float = 1500.0, micro_audio_s: float = 400.0,
                 max_dec_len: int = 200, pool_micro: int = 50, seed: int = 0,
                 weights: dict[str, float] | Sequence[float] | None = None, *, prompt_len: int = len(PROMPT),
                 max_micro_tokens: int | None = None):
        self.step_audio_s, self.micro_audio_s = float(step_audio_s), float(micro_audio_s)
        self.max_dec_len, self.pool_micro, self.seed = int(max_dec_len), int(pool_micro), int(seed)
        self.max_micro_tokens = max_micro_tokens
        self.dur = np.array([u.duration for u in utts], dtype=np.float64)
        self.n_tok = np.array([u.n_tok for u in utts], dtype=np.int64)
        self.dec_len = prompt_len - 1 + self.n_tok
        if weights is None:
            w = np.ones(len(utts))
        elif isinstance(weights, dict):
            w = np.array([float(weights.get(u.source, 1.0)) for u in utts])
        else:
            w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(utts) or (w < 0).any():
            raise ValueError("weights must be non-negative, one per utterance (or a dict source -> weight)")
        eligible = self.dec_len <= self.max_dec_len
        self.n_excluded = int((~eligible).sum())
        self.w = np.where(eligible, w, 0.0)
        self.epoch, self.step_in_epoch = 0, 0
        self.stats: dict = {}  # plan_stats of the most recently planned epoch
        self.epoch_stats: dict[int, dict] = {}
        h = hashlib.sha256()
        for part in (self.dur.astype(np.float32).tobytes(), self.n_tok.tobytes(), self.w.tobytes(),
                     json.dumps([self.step_audio_s, self.micro_audio_s, self.max_dec_len, self.pool_micro, self.seed,
                                 prompt_len, max_micro_tokens]).encode()):
            h.update(part)
        self.fingerprint = h.hexdigest()[:16]

    def epoch_plan(self, epoch: int) -> list[list[list[int]]]:
        rng = np.random.default_rng([self.seed, int(epoch)])
        base = np.floor(self.w).astype(np.int64)
        reps = base + (rng.random(len(self.w)) < (self.w - base))
        order = np.repeat(np.arange(len(self.w)), reps)
        order = order[rng.permutation(len(order))]

        # pools of ~pool_micro micro-batches' worth of real audio, sorted longest-first inside
        d = self.dur[order]
        start = np.cumsum(d) - d
        pool_id = (start // (self.pool_micro * self.micro_audio_s)).astype(np.int64)
        micro: list[list[int]] = []
        for p in np.unique(pool_id):
            pool = order[pool_id == p]
            pool = pool[np.argsort(-self.dur[pool], kind="stable")]
            micro += pack_micro_batches(pool, self.dur, self.micro_audio_s, self.dec_len, self.max_micro_tokens)

        micro = [micro[i] for i in rng.permutation(len(micro))]
        steps, cur, cur_s = [], [], 0.0
        for mb in micro:
            s = float(self.dur[mb].sum())
            if cur and cur_s + s / 2 > self.step_audio_s:  # adding would overshoot more than stopping undershoots
                steps.append(cur)
                cur, cur_s = [], 0.0
            cur.append(mb)
            cur_s += s
        if cur:
            steps.append(cur)
        plan = [steps[i] for i in rng.permutation(len(steps))]
        self.stats = self.epoch_stats[int(epoch)] = dict(epoch=int(epoch), **self.plan_stats(plan))
        return plan

    def iter_steps(self) -> Iterator[tuple[int, int, list[list[int]]]]:
        """(epoch, step_idx, micro-batches) from the current position onward, across epochs, forever. The cursor is
        private: the position only moves through step_done(), so a prefetching consumer can run ahead safely."""
        epoch, start = self.epoch, self.step_in_epoch
        while True:
            plan = self.epoch_plan(epoch)
            if not plan:
                raise ValueError("empty epoch plan: no utterance fits max_dec_len or all weights are 0")
            for j in range(start, len(plan)):
                yield epoch, j, plan[j]
            epoch, start = epoch + 1, 0

    def plan_stats(self, plan: list[list[list[int]]]) -> dict:
        """Padding efficiency and size extremes of a plan (logged by the trainer; used to pick memory probes)."""
        mbs = [mb for step in plan for mb in step]
        if not mbs:
            return dict(steps=0, micro_batches=0)
        real = np.array([self.dur[mb].sum() for mb in mbs])
        padded = np.array([self.dur[mb].max() * len(mb) for mb in mbs])
        dec_real = np.array([self.dec_len[mb].sum() for mb in mbs])
        dec_pad = np.array([self.dec_len[mb].max() * len(mb) for mb in mbs])
        step_s = np.array([sum(self.dur[mb].sum() for mb in step) for step in plan])
        return dict(
            steps=len(plan), micro_batches=len(mbs), utts=int(sum(len(mb) for mb in mbs)),
            excluded_dec_len=self.n_excluded, real_h=float(real.sum() / 3600),
            pad_eff_audio=float(real.sum() / padded.sum()), pad_eff_dec=float(dec_real.sum() / dec_pad.sum()),
            micro_per_step=float(len(mbs) / len(plan)), step_real_s_mean=float(step_s.mean()),
            step_real_s_min=float(step_s.min()), step_real_s_max=float(step_s.max()),
            micro_padded_s_max=float(padded.max()), micro_utts_max=int(max(len(mb) for mb in mbs)),
            micro_dec_positions_max=int(dec_pad.max()),
            micro_targets_max=int(max(self.n_tok[mb].sum() for mb in mbs)),
            targets=int(sum(self.n_tok[mb].sum() for mb in mbs)),
        )

    def worst_micro_batches(self, plan: list[list[list[int]]] | None = None) -> dict[str, list[int]]:
        """Micro-batches of a plan (default epoch 0) that stress memory differently, for the smoke-phase probe:
        the one with the longest padded audio (encoder), and the ones with the most padded decoder positions and
        the most target positions (decoder activations and the (N, V) logits - these are the SHORT utterances)."""
        plan = self.epoch_plan(0) if plan is None else plan
        mbs = [mb for step in plan for mb in step]
        return dict(
            longest=max(mbs, key=lambda mb: (self.dur[mb].max(), self.dur[mb].max() * len(mb))),
            most_dec_positions=max(mbs, key=lambda mb: self.dec_len[mb].max() * len(mb)),
            most_targets=max(mbs, key=lambda mb: self.n_tok[mb].sum()),
        )

    def step_done(self, epoch: int, step_idx: int):
        """Record that step `step_idx` of epoch `epoch` has been applied: the position becomes the step after it."""
        n_steps = self.epoch_stats[epoch]["steps"] if epoch in self.epoch_stats else len(self.epoch_plan(epoch))
        self.epoch, self.step_in_epoch = (epoch, step_idx + 1) if step_idx + 1 < n_steps else (epoch + 1, 0)

    def state_dict(self) -> dict:
        return dict(epoch=self.epoch, step_in_epoch=self.step_in_epoch, seed=self.seed, n_utts=len(self.dur),
                    fingerprint=self.fingerprint)

    def load_state_dict(self, state: dict, strict: bool = True):
        """Restore the position. strict: refuse a state from a planner over different data or settings, since the
        same (seed, epoch) would then give a different order and the resumed epoch would repeat/skip utterances."""
        if strict and state.get("fingerprint") != self.fingerprint:
            raise ValueError(f"planner state is for different data/settings (fingerprint {state.get('fingerprint')} "
                             f"!= {self.fingerprint}, n_utts {state.get('n_utts')} vs {len(self.dur)})")
        self.epoch, self.step_in_epoch = int(state["epoch"]), int(state["step_in_epoch"])


# -------------------------------------------------------------------------------------------------------- dataset


class AudioBatchDataset(torch.utils.data.Dataset):
    """dataset[list of store indices] -> one collated micro-batch (dict):

      wave               (B, S)  float32  16 kHz mono, zero-padded
      lengths            (B,)    int64    valid samples per row
      decoder_input_ids  (B, L)  int64    prompt + tokens[:-1], PAD-padded; L = len(prompt) - 1 + max T
      dec_mask           (B, L)  int64    decoder_attention_mask (1 = real position)
      tgt_row, tgt_pos   (N,)    int64    target n is predicted by last_hidden_state[tgt_row[n], tgt_pos[n]];
                                          tgt_pos = len(prompt) - 1 + t for teacher step t. N = sum of T, row-major
      tgt_mask           (B, L)  bool     the same positions; h[tgt_mask] has the same order as h[tgt_row, tgt_pos]
      top_idx            (N, k)  int64    teacher top-k ids, column 0 = the greedy target token
      top_lp             (N, k)  float32  teacher log-probs
      n_tok (B,) int64, durations (B,) float32, agree (B,) float32 (NaN = unknown), index (B,) int64 store indices
      ids, sources       list[str]
      dropped            list[str]        ids whose audio failed to decode; they are absent from everything above
                                          (B can be 0 if all failed - skip such a micro-batch)

    Picklable without the data (spawn workers get only small arrays); the memmaps open lazily in each process.
    """

    def __init__(self, stores: Stores):
        u = stores.utts
        self.cache_dir = str(stores.cache_dir)
        self.prompt = np.array(stores.info.get("prompt", PROMPT), dtype=np.int64)
        self.pad = int(stores.info.get("pad", PAD))
        self.ids = [x.id for x in u]
        self.sources = [x.source for x in u]
        self.duration = np.array([x.duration for x in u], dtype=np.float32)
        self.agree = np.array([x.agree for x in u], dtype=np.float32)
        self.audio_off = np.array([x.audio_off for x in u], dtype=np.int64)
        self.audio_len = np.array([x.audio_len for x in u], dtype=np.int64)
        self.tok_off = np.array([x.tok_off for x in u], dtype=np.int64)
        self.n_tok = np.array([x.n_tok for x in u], dtype=np.int64)
        self._mm = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None  # never pickle memmaps: each process maps the files itself
        return state

    def _open(self) -> dict:
        if self._mm is None:
            d = Path(self.cache_dir)
            audio = (np.memmap(d / "audio.bin", dtype=np.uint8, mode="r") if (d / "audio.bin").stat().st_size
                     else np.zeros(0, np.uint8))
            self._mm = dict(audio=audio, tok=np.load(d / "targets_tokens.npy", mmap_mode="r"),
                            idx=np.load(d / "targets_topk_idx.npy", mmap_mode="r"),
                            lp=np.load(d / "targets_topk_lp.npy", mmap_mode="r"))
        return self._mm

    def audio_bytes(self, i: int) -> bytes:
        o = int(self.audio_off[i])
        return self._open()["audio"][o:o + int(self.audio_len[i])].tobytes()

    def targets(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tokens (T,) int16, topk_idx (T,k) int16, topk_lp (T,k) float16) of store index i."""
        mm = self._open()
        o, t = int(self.tok_off[i]), int(self.n_tok[i])
        return np.asarray(mm["tok"][o:o + t]), np.asarray(mm["idx"][o:o + t]), np.asarray(mm["lp"][o:o + t])

    def __getitem__(self, idx: Sequence[int]) -> dict:
        keep, waves, dropped = [], [], []
        for i in idx:
            i = int(i)
            try:  # the teacher decoded every row it labelled, but rebuilt shards on another box might not
                waves.append(decode_audio(self.audio_bytes(i)))
                keep.append(i)
            except Exception as e:
                print(f"trainset: undecodable audio, dropping {self.ids[i]}: {e}", file=sys.stderr)
                dropped.append(self.ids[i])
        B, P = len(keep), len(self.prompt)
        lengths = np.array([len(w) for w in waves], dtype=np.int64)
        wave = np.zeros((B, int(lengths.max()) if B else 0), dtype=np.float32)
        for b, w in enumerate(waves):
            wave[b, :len(w)] = w
        T = self.n_tok[keep]
        L = P - 1 + int(T.max()) if B else P
        dec = np.full((B, L), self.pad, dtype=np.int64)
        dec_mask = np.zeros((B, L), dtype=np.int64)
        tgt_mask = np.zeros((B, L), dtype=bool)
        rows, pos, top_idx, top_lp = [], [], [], []
        for b, i in enumerate(keep):
            tok, ti, tl = self.targets(i)
            t = len(tok)
            dec[b, :P] = self.prompt
            dec[b, P:P + t - 1] = tok[:-1]
            dec_mask[b, :P + t - 1] = 1
            tgt_mask[b, P - 1:P - 1 + t] = True
            rows.append(np.full(t, b, dtype=np.int64))
            pos.append(np.arange(P - 1, P - 1 + t, dtype=np.int64))
            top_idx.append(ti)
            top_lp.append(tl)
        k = self._open()["idx"].shape[1]
        return dict(
            wave=torch.from_numpy(wave), lengths=torch.from_numpy(lengths),
            decoder_input_ids=torch.from_numpy(dec), dec_mask=torch.from_numpy(dec_mask),
            tgt_row=_cat(rows, np.int64, (0,)), tgt_pos=_cat(pos, np.int64, (0,)), tgt_mask=torch.from_numpy(tgt_mask),
            top_idx=_cat(top_idx, np.int64, (0, k)), top_lp=_cat(top_lp, np.float32, (0, k)),
            n_tok=torch.from_numpy(T.astype(np.int64)), durations=torch.from_numpy(self.duration[keep]),
            agree=torch.from_numpy(self.agree[keep]), index=torch.tensor(keep, dtype=torch.int64),
            ids=[self.ids[i] for i in keep], sources=[self.sources[i] for i in keep], dropped=dropped,
        )


def _cat(parts: list[np.ndarray], dtype, empty_shape: tuple) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(parts).astype(dtype) if parts else np.zeros(empty_shape, dtype))


# --------------------------------------------------------------------------------------------------------- loader


def default_num_workers() -> int:
    """perf.num_workers = "auto": FLAC decode is ~4000x realtime per core, so a few workers keep up with any GPU."""
    if os.name == "nt":
        return 2
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _worker_init(_worker_id: int):
    torch.set_num_threads(1)  # workers only decode and pad; N workers x M intra-op threads would oversubscribe
    strategy = os.environ.get("KITSUNE_SHARING")
    if strategy:
        torch.multiprocessing.set_sharing_strategy(strategy)


class _StepSampler:
    """Flattens steps into micro-batches for the DataLoader and records each step's size, in order. The DataLoader
    pulls from it ahead of the consumer (prefetch), so a step's size is always known before its first item arrives."""

    def __init__(self, steps: Iterable[tuple[object, list[list[int]]]]):
        self.steps, self.sizes = steps, deque()

    def __iter__(self):
        for key, step in self.steps:
            self.sizes.append((key, len(step)))
            yield from step


def make_loader(dataset: AudioBatchDataset, plan: "list[list[list[int]]] | StepPlanner", num_workers: int = 2,
                prefetch: int = 4, *, start_step: int = 0, pin_memory: bool | None = None,
                mp_context: str = "spawn") -> Iterator[tuple[object, list[dict]]]:
    """Decode micro-batches in `num_workers` processes (`prefetch` micro-batches in flight per worker) and yield
    whole optimizer steps:
      plan = one epoch's list of steps    -> (step_idx, [micro-batch, ...]) for plan[start_step:]
      plan = a StepPlanner                -> ((epoch, step_idx), [micro-batch, ...]) from the planner's position,
                                             across epochs without end; the workers persist, so there is no
                                             worker-restart stall at epoch boundaries
    Workers use `spawn` on every OS, so Linux runs the same code path as the Windows smoke run. KITSUNE_SHARING
    (e.g. file_system), if set, becomes torch's sharing strategy in the main process and in the workers.
    Closing the iterator (or dropping it) shuts the workers down."""
    if isinstance(plan, StepPlanner):
        steps = (((e, j), step) for e, j, step in plan.iter_steps())
    else:
        steps = ((start_step + j, step) for j, step in enumerate(plan[start_step:]))
    sampler = _StepSampler(steps)
    strategy = os.environ.get("KITSUNE_SHARING")
    if strategy:
        torch.multiprocessing.set_sharing_strategy(strategy)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch if num_workers > 0 else None, worker_init_fn=_worker_init if num_workers > 0 else None,
        multiprocessing_context=mp_context if num_workers > 0 else None, persistent_workers=False,
    )

    def gen():
        it = iter(loader)
        try:
            while True:
                try:
                    first = next(it)
                except StopIteration:
                    return
                key, n = sampler.sizes.popleft()
                yield key, [first] + [next(it) for _ in range(n - 1)]
        finally:
            shutdown = getattr(it, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()

    return gen()
