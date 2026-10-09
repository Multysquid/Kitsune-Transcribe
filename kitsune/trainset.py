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

The CTC family (the size study's Parakeet students) trains on the Parakeet teacher's per-frame targets instead: a FRAME
store (build_frame_stores; layout and frame preflight in the "frame stores" section below) holds the same audio and,
per utterance, kitsune.ctc_targets.FrameTargets; FrameBatchDataset collates it, and StepPlanner(max_dec_len=None) plans
it without a decoder. dataset_for(stores) picks the dataset of a store. Either family's TRAIN loader may augment its
micro-batches (dataset_for(stores, augment=Augment(...)); the "augmentation" section below; scripts/04_distill.py
augment.*): rows cut inside a sentence (before one of its words, never between its last word and its mark),
utterances joined into longer rows, another row's speech mixed in. An AED row has no frame targets to cut, so its cuts
come from a cut table (kitsune.aed_cuts: the Parakeet teacher's alignment of the same rows, mapped onto the Cohere
tokens) joined to the dataset (cuts=CutIndex). A
dataset built without it - every eval, dev, probe, smoke and memory-probe one - returns the stored rows as they are.
"""
import copy
import hashlib
import json
import math
import os
import sys
import time
import warnings
import zipfile
from collections import deque
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from kitsune.acoustics import CODECS, DEFAULT_CODECS
from kitsune.audio import TARGET_SR, decode_audio
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

    def _dataset(self) -> "AudioBatchDataset | FrameBatchDataset":
        if self._ds is None:
            self._ds = dataset_for(self)
        return self._ds

    def wave(self, i: int) -> np.ndarray:
        """Decoded 16 kHz mono float32 audio of utts[i]. With targets() this is the store protocol kitsune.evaluate
        uses. Maps the cache files into this process (on Windows a mapped cache cannot be rebuilt in place)."""
        return decode_audio(self._dataset().audio_bytes(i))

    def targets(self, i: int):
        """(tokens (T,) int16, topk_idx (T,k) int16, topk_logprob (T,k) float16) of utts[i], as in the npz; for a
        frame store (build_frame_stores) its kitsune.ctc_targets.FrameTargets."""
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


def _npz_members(path: Path) -> list:
    """(name, CRC32, size) of every array in an .npz, read from the zip's central directory (no array data).
    np.savez stores uncompressed, so a teacher re-run with the same shapes keeps the file size; the CRCs do not."""
    with zipfile.ZipFile(path) as z:
        return sorted((i.filename, i.CRC, i.file_size) for i in z.infolist())


def _teacher_meta(teacher_root: Path) -> dict:
    p = Path(teacher_root) / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return dict(prompt=[int(x) for x in meta.get("decoder_prompt_ids", PROMPT)], eos=int(meta.get("eos_token_id", EOS)),
                pad=int(meta.get("pad_token_id", PAD)), k=int(meta.get("k", 16)))


def read_selection(selection_path: Path, sources: Sequence[str], splits: Sequence[str]) -> pd.DataFrame:
    """Kept rows of the selection for these sources/splits, in selection order. A split outside
    kitsune.fullrun.SPLITS (train, dev, eval) is a ValueError: a typo would silently select nothing."""
    from kitsune.fullrun import SPLITS

    if bad := [s for s in splits if s not in SPLITS]:
        raise ValueError(f"splits {bad} are not selection splits {list(SPLITS)}")
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


def _pack_audio(sel: pd.DataFrame, shard_files: list[Path], data_root: Path, shard_size: dict[str, int],
                audio_tmp: Path, shard_rows: list[int] | None = None
                ) -> tuple[list[int], list[int], dict[str, int], pd.DataFrame]:
    """The audio pass of a store build: the selected rows' container bytes, joined BY ID from the data shards (the
    first copy of an id wins), written back to back into `audio_tmp` in the order they are found. Returns (order: the
    selection row of each written utterance, lens: its byte count, used: the shards that contributed -> their size,
    lost: the selected rows without audio). No audio at all is an error. shard_rows (a list, filled in place): the
    rows each contributing shard wrote, in packing order (the frame preflight's decode tasks are cut by shard)."""
    want = dict(zip(sel["id"].tolist(), range(len(sel))))
    order, lens, used = [], [], {}
    with open(audio_tmp, "wb") as f:
        for path in shard_files:
            if not want:
                break
            first = len(order)
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
            if shard_rows is not None and len(order) > first:
                shard_rows.append(len(order) - first)
        f.flush()
        os.fsync(f.fileno())
    lost = sel.iloc[sorted(want.values())]
    if not order:
        raise ValueError(f"no audio found under {data_root / 'shards'} for any of the {len(sel)} selected rows")
    return order, lens, used, lost


def _shard_files(sel: pd.DataFrame, data_root: Path, sources: Sequence[str], splits: Sequence[str]) -> list[Path]:
    """The audio shards a store build reads: per source and split, data/shards/<source>/<split>-*.parquet (sorted),
    except the dev split, which has no shards of its own: a dev row's audio (and its labels, joined by teacher_file)
    lives in its train shard (kitsune.fullrun.shard_split), so the dev split reads exactly the existing shards its
    selected rows' teacher_file names (<source>/train-NNNNN) - not every train shard, whose id columns alone are
    hundreds of GB of row groups to scan on the full data. Deduplicated, in that order."""
    from kitsune.fullrun import DEV_SPLIT, shard_split

    out: list[Path] = []
    for s in sources:
        for sp in splits:
            if sp != DEV_SPLIT:
                out += sorted((data_root / "shards" / s).glob(f"{sp}-*.parquet"))
                continue
            stems = sorted(set(sel["teacher_file"][(sel["source"] == s) & (sel["split"] == sp)]))
            if bad := [f for f in stems if not Path(f).name.startswith(f"{shard_split(sp)}-")]:
                raise ValueError(f"{DEV_SPLIT} rows must name their {shard_split(sp)} shard in teacher_file, got "
                                 f"{bad[:3]}")
            out += [p for p in (data_root / "shards" / f"{f}.parquet" for f in stems) if p.is_file()]
    return list(dict.fromkeys(out))


def build_stores(selection_path, data_root, teacher_root, cache_dir, sources: Sequence[str], splits: Sequence[str],
                 *, ids: Iterable[str] | None = None, log: Callable[[str], None] = print) -> Stores:
    """Pack the kept selection rows of `sources` x `splits` into <cache_dir> (layout in the module docstring).
    `ids` restricts the store further (e.g. a seeded calibration sample or a small smoke-run subset), so a laptop
    need not copy all of a source's audio.

    Idempotent: nothing is rebuilt if <cache_dir>/stores.json carries the same fingerprint (format version,
    selection file hash, sources, splits, prompt, k, the npz members' CRCs, the .jsonl hashes) and every shard that
    contributed audio still has its size. New shards (a download still appending to data/) only force a rebuild if
    the last build dropped rows for missing audio. Store order is the order in which the audio was found (sources in the given order, shards sorted
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
    shard_files = _shard_files(sel, data_root, sources, splits)  # the dev split: its rows' train shards only
    npz_files = [teacher_root / f"{f}.npz" for f in sorted(set(sel["teacher_file"]))]
    missing_npz = [p for p in npz_files if not p.exists()]
    if missing_npz:
        raise FileNotFoundError(f"teacher output missing for selected rows: {missing_npz[:3]}")

    fp_src = dict(version=FORMAT_VERSION, selection=_sha256(selection_path), sources=sources, splits=splits,
                  prompt=meta["prompt"], k=meta["k"],
                  ids=None if ids is None else hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                  npz=[(p.relative_to(teacher_root).as_posix(), p.stat().st_size, _npz_members(p)) for p in npz_files],
                  # ref/hyp come from the .jsonl next to each npz; small enough to hash whole
                  jsonl=[_sha256(q) if q.exists() else None for q in (p.with_suffix(".jsonl") for p in npz_files)])
    fingerprint = hashlib.sha256(json.dumps(fp_src, sort_keys=True).encode()).hexdigest()
    shard_size = {p.relative_to(data_root).as_posix(): p.stat().st_size for p in shard_files}
    info_path = cache_dir / "stores.json"
    if info_path.exists() and _cache_complete(cache_dir):
        old = json.loads(info_path.read_text(encoding="utf-8"))
        same_audio = all(shard_size.get(r) == sz for r, sz in old.get("shards", {}).items()) and (
            old["dropped"]["no_audio"]["n"] == 0 or sorted(shard_size) == old.get("all_shards"))
        if old.get("fingerprint") == fingerprint and same_audio:
            st = load_stores(cache_dir)
            st.info["reused"] = True  # in memory only, as build_frame_stores (04_distill's data event: stores_reused)
            log(f"stores: reusing {cache_dir} ({len(st)} utts, {st.hours:.2f} h, built {old.get('created')})")
            return st

    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path.unlink(missing_ok=True)  # invalidate first: a crash mid-build must not leave a cache that looks valid

    # 1. audio, joined by id, written in the order it is found
    order, lens, used, lost = _pack_audio(sel, shard_files, data_root, shard_size, cache_dir / "audio.bin.tmp")
    audio_tmp = cache_dir / "audio.bin.tmp"
    t_audio = time.time() - t0

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


# ---------------------------------------------------------------------------------------------------- frame stores
#
# The CTC family (the Parakeet students of the size study, scripts/04_distill.py family "ctc") trains on per-frame
# targets of the Parakeet teacher (parakeet_out, kitsune/parakeet_targets.py FORMAT), not on Cohere's token targets.
# A frame store holds the same audio as a token store and, per utterance, what kitsune.ctc_targets.FrameTargets holds:
#   index.parquet          the Utt fields (n_tok = U, the CTC target tokens; tok_off = the row offset into ctc_ids;
#                          teacher_cer / truncated = the Parakeet CTC pass's ctc_cer / truncated) + ref, hyp (the
#                          teacher's CTC hypothesis ctc_hyp, what "CER vs teacher" compares with), tdt_hyp, n_frames,
#                          n_samples (the decoded audio's length, measured by the frame preflight)
#   audio.bin, audio_offsets.npy       as in a token store (every packed row; a dropped row's bytes stay unused)
#   frames_offsets.npy (n+1,) int64 + frames_blank_lp.npy (F,) float16   log p(blank) at every valid frame
#   dense_offsets.npy (n+1,) int64 + dense_frame.npy (D,) int16 + dense_topk_idx.npy (D,k) int16 + dense_topk_lp.npy
#                          (D,k) float16                     the frames with p(blank) < 0.95 and their top-k
#   ctc_offsets.npy (n+1,) int64 + ctc_ids.npy (U,) int16     the greedy CTC path (decision 22: the CTC target)
# The offsets are indexed by Utt.row. The frames must align 1:1 with the student's: its frame count is a function of
# the decoded audio's length (ctc_frames), and the stored n_frames is the teacher's. The FRAME PREFLIGHT (STUDY.md 3.2)
# checks every row at build time by decoding its audio - the duration field is not exact (K4: n_frames from the stored
# duration differs on 11.6 % of CV8 rows, from the decoded audio on none) - and applies decision 15: a mismatched row
# is dropped and counted; more than FRAME_MISMATCH_MAX_FRAC of the train rows, or any eval row, fails the build
# (FramePreflightFailed; its report is also written to <cache_dir>/frame_preflight.json). A row whose audio does not
# decode cannot be shown to align: it counts as a frame mismatch (dropped and counted in train, a hard fail in eval).
# The report is in stores.json (info["frame_preflight"]), so the trainer logs it for a reused cache too. The decode
# runs in processes, by shard (kitsune.ctc_preflight: threads serialise on the GIL).

FRAME_FORMAT_VERSION = 1  # bump when the frame store layout changes
FRAME_MISMATCH_MAX_FRAC = 0.001  # decision 15: above this share of the train rows the build fails
HOP = 160  # the Parakeet extractor's hop (10 ms at 16 kHz): valid mel frames = samples // HOP
CTC_SUBSAMPLING_CONVS = 3  # the encoder's three stride-2 convolutions (kernel 3, padding 1): 8x subsampling
FRAME_SAMPLES = HOP * 2 ** CTC_SUBSAMPLING_CONVS  # 1280 samples = one encoder frame (80 ms): ctc_frames(1280 c) == c
CTC_BLANK = 3072  # == kitsune.parakeet_targets.BLANK
CTC_VOCAB = 3073  # == kitsune.parakeet_targets.VOCAB
CTC_DENSE_THR = 0.95  # the label pass's dense-frame threshold (STUDY.md 2.1): below it p(blank) a frame keeps its top-k
FRAME_FILES = ("index.parquet", "audio.bin", "audio_offsets.npy", "frames_offsets.npy", "frames_blank_lp.npy",
               "dense_offsets.npy", "dense_frame.npy", "dense_topk_idx.npy", "dense_topk_lp.npy", "ctc_offsets.npy",
               "ctc_ids.npy")
_PARAKEET_KEYS = ("ids", "n_frames", "frame_offsets", "ctc_blank_lp", "dense_offsets", "ctc_dense_frame",
                  "ctc_topk_idx", "ctc_topk_lp", "k_ctc")


class FramePreflightFailed(ValueError):
    """The frame preflight's hard fail (decision 15); `report` is its record (also in frame_preflight.json)."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def ctc_frames(n_samples: int) -> int:
    """Encoder frames of a 16 kHz waveform of n_samples samples: the Parakeet extractor's valid mel frames
    (n // HOP, kitsune.features.feature_lengths with n_fft 512), then L -> (L - 1) // 2 + 1 per stride-2 convolution.
    Equal to kitsune.ctc_student.expected_n_frames (tests/test_ctc_trainer.py), computed here without importing it:
    the loader's worker processes check every row with it and should only pay for torch."""
    n = max(int(n_samples) // HOP, 0)
    for _ in range(CTC_SUBSAMPLING_CONVS):
        n = (n - 1) // 2 + 1
    return n


def is_frame_store(stores: "Stores") -> bool:
    return (stores.info or {}).get("kind") == "frames"


def dataset_for(stores: "Stores", augment: "Augment | None" = None, cuts=None, noise=None,
                rirs=None) -> "AudioBatchDataset | FrameBatchDataset":
    """The micro-batch dataset of a store: FrameBatchDataset for a frame store, else AudioBatchDataset. augment: the
    TRAIN loader's augmentation (Augment: a CTC one on a frame store; an AED one - max_tokens set - on a token store,
    with cuts, the cut table's kitsune.aed_cuts.CutIndex over the store's rows, when it cuts). Every other dataset -
    the evals, the dev slice, the probe, the smoke checks, the memory probe - is built without it and reads the stored
    rows as they are."""
    if is_frame_store(stores):
        if cuts is not None:
            raise ValueError("a cut index is an AED store's (kitsune.aed_cuts); a frame store cuts on its own targets")
        return FrameBatchDataset(stores, augment=augment, noise=noise, rirs=rirs)
    return AudioBatchDataset(stores, augment=augment, cuts=cuts, noise=noise, rirs=rirs)


def _frame_cache_complete(cache_dir: Path) -> bool:
    return all((cache_dir / n).exists() for n in FRAME_FILES)


def parakeet_meta(parakeet_root: Path) -> dict:
    """parakeet_out/meta.json (kitsune.parakeet_targets.build_meta), checked against what a frame store assumes: the
    format version, blank 3072 of a 3073-class vocabulary, the dense threshold 0.95 and a k_ctc (each npz must carry
    the same k_ctc; build_frame_stores checks). A missing file or another setting is an error, never a default: a
    root written with other settings (or a partial pull) would give the loss and every CER the wrong targets."""
    from kitsune import parakeet_targets as PT

    path = Path(parakeet_root) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: a parakeet_out root carries its settings there (a partial pull?)")
    meta = json.loads(path.read_text(encoding="utf-8"))
    bad = {k: meta.get(k) for k, want in (("format_version", PT.FORMAT_VERSION), ("blank", CTC_BLANK),
                                          ("vocab", CTC_VOCAB)) if meta.get(k) != want}
    thr, k = meta.get("ctc_dense_thr"), meta.get("k_ctc")
    if not isinstance(thr, (int, float)) or abs(float(thr) - CTC_DENSE_THR) > 1e-9:
        bad["ctc_dense_thr"] = thr
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        bad["k_ctc"] = k
    if bad:
        raise ValueError(f"{path}: {bad} - a frame store needs format_version {PT.FORMAT_VERSION}, blank {CTC_BLANK}, "
                         f"vocab {CTC_VOCAB}, ctc_dense_thr {CTC_DENSE_THR} and an integer k_ctc")
    return meta


def frame_preflight(ids: Sequence[str], sources: Sequence[str], splits: Sequence[str], durations, stored: Sequence[int],
                    decoded: Sequence, *, max_frac: float = FRAME_MISMATCH_MAX_FRAC) -> dict:
    """Decision 15 on the rows of one store build: `stored` = the teacher's n_frames, `decoded` = the decoded audio's
    length in samples, or an error text. A row whose ctc_frames(length) differs is a mismatch, and so is a row that
    does not decode (its frames cannot be shown to align; also counted as undecodable, with its error). Returns the
    report: counts per split (mismatch includes undecodable), the mismatched rows (dropped, with their stored and
    expected frames; expected None and the error for an undecodable one), the undecodable ones, and ok = no eval row
    mismatched and at most max_frac of the train rows. The dev split (the full runs' early-stop rows, cut from train
    shards) is held to the train rule on its own: at most max_frac of the dev rows (dev_mismatch_frac); any other
    split's mismatch fails (lenient_splits names the two)."""
    mism, bad = [], []
    by_split: dict[str, dict] = {}
    for i, (uid, src, sp, dur, t, ns) in enumerate(zip(ids, sources, splits, durations, stored, decoded)):
        b = by_split.setdefault(sp, dict(rows=0, mismatch=0, undecodable=0))
        b["rows"] += 1
        if not isinstance(ns, (int, np.integer)):
            b["undecodable"] += 1
            b["mismatch"] += 1
            bad.append(dict(id=uid, source=src, split=sp, error=str(ns)))
            mism.append(dict(i=i, id=uid, source=src, split=sp, stored=int(t), expected=None, n_samples=None,
                             duration=round(float(dur), 4), error=str(ns)))
            continue
        want = ctc_frames(int(ns))
        if want != int(t):
            b["mismatch"] += 1
            mism.append(dict(i=i, id=uid, source=src, split=sp, stored=int(t), expected=want, n_samples=int(ns),
                             duration=round(float(dur), 4)))
    lenient = ("train", "dev")  # each against max_frac on its own (kitsune.fullrun.DEV_SPLIT: train shards' rows)

    def frac_of(sp: str) -> float:
        b = by_split.get(sp) or {}
        return b["mismatch"] / b["rows"] if b.get("rows") else 0.0

    m_other = sum(b["mismatch"] for sp, b in by_split.items() if sp not in lenient)
    frac, dev_frac = frac_of("train"), frac_of("dev")
    ok = m_other == 0 and frac <= max_frac and dev_frac <= max_frac
    return dict(policy=f"decision 15: a row whose decoded audio gives another frame count than its stored n_frames, "
                       f"or whose audio does not decode, is dropped and counted; more than {100 * max_frac:g} % of the "
                       f"train rows (or of the dev rows), or any eval row, fails",
                max_frac=float(max_frac), ok=bool(ok), n_rows=len(ids), n_checked=len(ids) - len(bad),
                n_mismatch=len(mism), train_mismatch_frac=frac, dev_mismatch_frac=dev_frac,
                lenient_splits=list(lenient), by_split=by_split, n_undecodable=len(bad),
                undecodable=bad[:50], mismatches=[{k: v for k, v in m.items() if k != "i"} for m in mism],
                _rows=[m["i"] for m in mism])


def build_frame_stores(selection_path, data_root, parakeet_root, cache_dir, sources: Sequence[str],
                       splits: Sequence[str], *, ids: Iterable[str] | None = None, log: Callable[[str], None] = print,
                       workers: int | None = None, max_mismatch_frac: float = FRAME_MISMATCH_MAX_FRAC) -> Stores:
    """Pack the kept selection rows of `sources` x `splits` with their Parakeet frame targets into <cache_dir> (the
    layout above), running the frame preflight over every row. The selection's teacher_file (<source>/<stem>) names
    the parakeet_out shard too: the label box runs both passes over the same shards. teacher_out is never read (a CTC
    run need not have it for its train rows: kitsune.extent.pull_plan).

    Idempotent like build_stores: nothing is rebuilt if stores.json carries the same fingerprint (kind, format version,
    selection file hash, sources, splits, the npz members' CRCs, the .jsonl hashes, meta.json's settings, the preflight
    threshold) and the audio shards are unchanged. Selected rows without audio are dropped (info["dropped"]["no_audio"]);
    a selected row without Parakeet targets is an error (a selection made from another label root), and so are a
    missing or other-settings meta.json (parakeet_meta), a shard's missing .jsonl and a selected row without its jsonl
    line (its reference and teacher text: a partial pull must stop the build, not score CERs against empty strings); a
    frame mismatch is dropped (info["dropped"]["frame_mismatch"]) or fails the build (FramePreflightFailed) as
    frame_preflight decides (an undecodable row is a mismatch). workers: the preflight's decode processes
    (kitsune.ctc_preflight.decoded_lengths, by shard; default: every core up to 64)."""
    selection_path, data_root, parakeet_root, cache_dir = map(Path, (selection_path, data_root, parakeet_root,
                                                                     cache_dir))
    sources, splits = list(sources), list(splits)
    sel = read_selection(selection_path, sources, splits)
    if ids is not None:
        ids = set(ids)
        sel = sel[sel["id"].isin(ids)].reset_index(drop=True)
    if sel.empty:
        raise ValueError(f"{selection_path}: no kept rows for sources={sources} splits={splits}")
    pmeta = parakeet_meta(parakeet_root)
    shard_files = _shard_files(sel, data_root, sources, splits)  # the dev split: its rows' train shards only
    npz_files = [parakeet_root / f"{f}.npz" for f in sorted(set(sel["teacher_file"]))]
    # the pair of each shard: the npz holds the targets, its jsonl the reference and the teacher's text every CER reads
    missing = [q for p in npz_files for q in (p, p.with_suffix(".jsonl")) if not q.is_file()]
    if missing:
        raise FileNotFoundError(f"parakeet_out missing for selected rows ({len(missing)} files): {missing[:3]}")

    fp_src = dict(kind="frames", version=FRAME_FORMAT_VERSION, selection=_sha256(selection_path), sources=sources,
                  splits=splits, ids=None if ids is None else hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                  npz=[(p.relative_to(parakeet_root).as_posix(), p.stat().st_size, _npz_members(p)) for p in npz_files],
                  jsonl=[_sha256(p.with_suffix(".jsonl")) for p in npz_files],
                  settings={k: pmeta.get(k) for k in ("format_version", "k_ctc", "ctc_dense_thr", "blank", "vocab")},
                  # undecodable "mismatch": a cache built when an undecodable row was kept unchecked is rebuilt
                  preflight=dict(max_mismatch_frac=float(max_mismatch_frac), fail_on_eval=True, undecodable="mismatch"))
    fingerprint = hashlib.sha256(json.dumps(fp_src, sort_keys=True).encode()).hexdigest()
    shard_size = {p.relative_to(data_root).as_posix(): p.stat().st_size for p in shard_files}
    info_path = cache_dir / "stores.json"
    if info_path.exists() and _frame_cache_complete(cache_dir):
        old = json.loads(info_path.read_text(encoding="utf-8"))
        same_audio = all(shard_size.get(r) == sz for r, sz in old.get("shards", {}).items()) and (
            old["dropped"]["no_audio"]["n"] == 0 or sorted(shard_size) == old.get("all_shards"))
        if old.get("kind") == "frames" and old.get("fingerprint") == fingerprint and same_audio:
            st = load_stores(cache_dir)
            st.info["reused"] = True  # in memory only: this call did not build (nor preflight) it
            log(f"frame stores: reusing {cache_dir} ({len(st)} utts, {st.hours:.2f} h, built {old.get('created')})")
            return st

    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path.unlink(missing_ok=True)  # invalidate first: a crash mid-build must not leave a cache that looks valid
    (cache_dir / "frame_preflight.json").unlink(missing_ok=True)

    # 1. audio, joined by id, written in the order it is found (as build_stores)
    audio_tmp = cache_dir / "audio.bin.tmp"
    shard_rows: list[int] = []
    order, lens, used, lost = _pack_audio(sel, shard_files, data_root, shard_size, audio_tmp, shard_rows)
    t_audio = time.time() - t0
    idx = sel.iloc[order].reset_index(drop=True)
    n = len(idx)
    audio_offsets = np.concatenate([[0], np.cumsum(np.array(lens, dtype=np.int64))])

    # 2. frame targets, joined by id (only the CTC arrays of each npz are read)
    row_of = dict(zip(idx["id"].tolist(), range(n)))
    per_row: list = [None] * n
    text: list = [None] * n
    k = int(pmeta["k_ctc"])
    for npz in npz_files:
        with np.load(npz) as zf:
            z = {key: zf[key] for key in _PARAKEET_KEYS}
        if int(z["k_ctc"]) != k:
            raise ValueError(f"{npz}: k_ctc {int(z['k_ctc'])} differs from {parakeet_root / 'meta.json'}'s {k}")
        rows_json = {}
        jl = npz.with_suffix(".jsonl")
        with open(jl, encoding="utf-8") as fj:
            for line in fj:
                if line.strip():
                    r = json.loads(line)
                    rows_json[r["id"]] = r
        fo, do = z["frame_offsets"], z["dense_offsets"]
        for j, x in enumerate(str(u) for u in z["ids"]):
            r = row_of.get(x)
            if r is None:
                continue
            if x not in rows_json:  # the label pass writes one jsonl line per npz row: not one pass's output
                raise ValueError(f"{jl}: no line for {x}, which its npz holds - the reference and the teacher's text "
                                 f"of a selected row are missing")
            T = int(z["n_frames"][j])
            fs, ds_ = slice(int(fo[j]), int(fo[j + 1])), slice(int(do[j]), int(do[j + 1]))
            if fs.stop - fs.start != T:
                raise ValueError(f"{npz.name}: {x} has {fs.stop - fs.start} CTC frames for n_frames {T}")
            dense = z["ctc_dense_frame"][ds_].astype(np.int16)
            tidx = z["ctc_topk_idx"][ds_].astype(np.int16)
            col0 = np.full(T, CTC_BLANK, dtype=np.int64)
            if len(dense):
                col0[dense.astype(np.int64)] = tidx[:, 0]
            prev = np.concatenate([[-1], col0[:-1]])
            ctc = col0[(col0 != prev) & (col0 != CTC_BLANK)].astype(np.int16)  # == parakeet_targets.ctc_greedy
            per_row[r] = (T, z["ctc_blank_lp"][fs].astype(np.float16), dense, tidx,
                          z["ctc_topk_lp"][ds_].astype(np.float16), ctc)
            text[r] = rows_json[x]
    missing = [idx["id"][r] for r in range(n) if per_row[r] is None]
    if missing:
        raise ValueError(f"{len(missing)} selected rows not found in their parakeet_out shard, e.g. {missing[0]} - "
                         f"selection and parakeet_out do not match")

    # 3. the frame preflight: decode every row (processes, by shard), compare its frame count with the stored one
    from kitsune.ctc_preflight import decoded_lengths

    t1 = time.time()
    decoded, how = decoded_lengths(audio_tmp, audio_offsets, workers, shard_rows=shard_rows)
    nw = how["workers"]
    report = frame_preflight(idx["id"].tolist(), idx["source"].tolist(), idx["split"].tolist(),
                             idx["duration"].to_numpy(), [p[0] for p in per_row], decoded, max_frac=max_mismatch_frac)
    drop = set(report.pop("_rows"))
    wall = time.time() - t1
    report.update(workers=nw, decode=how["mode"], tasks=how["tasks"], wall_s=round(wall, 2),
                  rows_per_s=round(n / wall, 1) if wall > 0 else None, cache_dir=str(cache_dir))
    if not report["ok"]:
        tmp = cache_dir / "frame_preflight.json.tmp"
        tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
        tmp.replace(cache_dir / "frame_preflight.json")
        by = report["by_split"]
        raise FramePreflightFailed(
            f"frame preflight failed (decision 15) for {cache_dir}: {report['n_mismatch']} of {n} rows have another "
            f"frame count than their stored n_frames ("
            + ", ".join(f"{sp} {b['mismatch']}/{b['rows']}" for sp, b in sorted(by.items()))
            + f"; train share {100 * report['train_mismatch_frac']:.3f} %, dev share "
            f"{100 * report.get('dev_mismatch_frac', 0.0):.3f} %, limit {100 * max_mismatch_frac:g} % each, any "
            f"eval row fails), e.g. {report['mismatches'][:3]}", report)
    keep = [r for r in range(n) if r not in drop]
    dropped_rows = idx.iloc[sorted(drop)]
    kidx = idx.iloc[keep].reset_index(drop=True)
    kept = [per_row[r] for r in keep]
    ktext = [text[r] for r in keep]
    nk = len(kept)
    if not nk:
        raise ValueError(f"{selection_path}: no row left after the frame preflight")

    # 4. publish: data files first, index + stores.json last
    def offsets(sizes):
        return np.concatenate([[0], np.cumsum(np.asarray(sizes, dtype=np.int64))]).astype(np.int64)

    fr_off = offsets([p[0] for p in kept])
    de_off = offsets([len(p[2]) for p in kept])
    ctc_off = offsets([len(p[5]) for p in kept])
    _write_npy(cache_dir / "audio_offsets.npy", audio_offsets)
    _write_npy(cache_dir / "frames_offsets.npy", fr_off)
    _write_npy(cache_dir / "frames_blank_lp.npy", np.concatenate([p[1] for p in kept]).astype(np.float16))
    _write_npy(cache_dir / "dense_offsets.npy", de_off)
    _write_npy(cache_dir / "dense_frame.npy", np.concatenate([p[2] for p in kept]).astype(np.int16))
    _write_npy(cache_dir / "dense_topk_idx.npy",
               np.concatenate([p[3] for p in kept]).reshape(-1, k).astype(np.int16))
    _write_npy(cache_dir / "dense_topk_lp.npy",
               np.concatenate([p[4] for p in kept]).reshape(-1, k).astype(np.float16))
    _write_npy(cache_dir / "ctc_offsets.npy", ctc_off)
    _write_npy(cache_dir / "ctc_ids.npy", np.concatenate([p[5] for p in kept]).astype(np.int16))
    audio_tmp.replace(cache_dir / "audio.bin")

    agree = kidx["agree"].astype("float64").to_numpy() if "agree" in kidx else np.full(nk, np.nan)
    n_tok = np.array([len(p[5]) for p in kept], dtype=np.int64)
    a_off = audio_offsets[:-1][keep]  # a kept row's bytes in audio.bin (a dropped row's stay there, unused)
    a_len = np.diff(audio_offsets)[keep]
    cols = dict(
        id=kidx["id"].tolist(), source=kidx["source"].tolist(), duration=kidx["duration"].to_numpy(np.float32),
        n_tok=n_tok, audio_off=a_off.astype(np.int64), audio_len=a_len.astype(np.int64), tok_off=ctc_off[:-1],
        row=np.arange(nk, dtype=np.int64), split=kidx["split"].tolist(), agree=agree.astype(np.float32),
        teacher_cer=np.array([float(t.get("ctc_cer", np.nan)) for t in ktext], dtype=np.float32),
        truncated=np.array([bool(t.get("truncated", False)) for t in ktext], dtype=bool),
        in_probe=kidx["in_probe"].to_numpy(bool), in_greedy_subset=kidx["in_greedy_subset"].to_numpy(bool),
        ref=[t.get("ref") for t in ktext], hyp=[t.get("ctc_hyp") for t in ktext], tdt_hyp=[t.get("hyp") for t in ktext],
        n_frames=np.array([p[0] for p in kept], dtype=np.int64),
        n_samples=np.array([int(decoded[r]) for r in keep], dtype=np.int64),  # every kept row decoded
    )
    tmp = cache_dir / "index.parquet.tmp"
    pq.write_table(pa.table(cols), tmp)
    fsync_path(tmp)
    tmp.replace(cache_dir / "index.parquet")

    per_source = {s: dict(utts=int((kidx["source"] == s).sum()),
                          hours=float(kidx["duration"][kidx["source"] == s].sum() / 3600)) for s in sources}
    report["dropped"] = [m["id"] for m in report["mismatches"]]

    def dropped_rec(df: pd.DataFrame) -> dict:
        return dict(n=len(df), hours=float(df["duration"].sum() / 3600), by_source=df["source"].value_counts().to_dict(),
                    ids=df["id"].tolist()[:50])

    info = dict(
        kind="frames", fingerprint=fingerprint, format_version=FRAME_FORMAT_VERSION,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), selection=str(selection_path),
        selection_sha256=fp_src["selection"], sources=sources, splits=splits, parakeet_root=str(parakeet_root),
        parakeet_settings=fp_src["settings"], k=k, blank=CTC_BLANK, n_utts=nk, hours=float(kidx["duration"].sum() / 3600),
        n_frames=int(fr_off[-1]), n_dense=int(de_off[-1]), n_ctc_tokens=int(ctc_off[-1]),
        audio_bytes=int(audio_offsets[-1]),
        target_bytes=int(fr_off[-1] * 2 + de_off[-1] * (2 + 4 * k) + ctc_off[-1] * 2), per_source=per_source,
        dropped=dict(no_audio=dropped_rec(lost), frame_mismatch=dropped_rec(dropped_rows)),
        frame_preflight=report, build_s=round(time.time() - t0, 2), audio_s=round(t_audio, 2),
        shards=used, all_shards=sorted(shard_size) if len(lost) else None,
    )
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    fsync_path(tmp)
    tmp.replace(info_path)
    log(f"frame stores: built {cache_dir} in {info['build_s']:.1f} s (audio pass {t_audio:.1f} s, preflight "
        f"{report['wall_s']:.1f} s, {nw} {report['decode']}): {nk} utts, {info['hours']:.2f} h, {info['n_frames']} frames, "
        f"{info['n_ctc_tokens']} CTC target tokens; preflight {report['n_checked']} rows checked, "
        f"{report['n_mismatch']} dropped, {report['n_undecodable']} undecodable"
        + (f"; DROPPED {len(lost)} rows without audio" if len(lost) else ""))
    return load_stores(cache_dir)


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
    max_dec_len None is the frame planner of the CTC family (a frame store: no decoder, n_tok = the CTC target
    tokens): nothing is excluded and micro-batches are cut by padded audio alone - the CTC student's activations and
    its (frames x 3073) logits both scale with the padded frames, which scale with the padded audio.
    Position for resume is (epoch, step_in_epoch): the next step to run. The trainer calls step_done(epoch, step)
    after each optimizer step and saves state_dict() with its checkpoint; make_loader(ds, planner, ...) then starts
    from that position. epoch_stats[e] holds plan_stats() of every epoch planned so far.
    """

    def __init__(self, utts: Sequence[Utt], step_audio_s: float = 1500.0, micro_audio_s: float = 400.0,
                 max_dec_len: int | None = 200, pool_micro: int = 50, seed: int = 0,
                 weights: dict[str, float] | Sequence[float] | None = None, *, prompt_len: int = len(PROMPT),
                 max_micro_tokens: int | None = None):
        self.step_audio_s, self.micro_audio_s = float(step_audio_s), float(micro_audio_s)
        self.frames = max_dec_len is None
        self.max_dec_len = None if self.frames else int(max_dec_len)
        self.pool_micro, self.seed = int(pool_micro), int(seed)
        self.max_micro_tokens = max_micro_tokens
        self.dur = np.array([u.duration for u in utts], dtype=np.float64)
        self.n_tok = np.array([u.n_tok for u in utts], dtype=np.int64)
        self.dec_len = None if self.frames else prompt_len - 1 + self.n_tok
        if weights is None:
            w = np.ones(len(utts))
        elif isinstance(weights, dict):
            w = np.array([float(weights.get(u.source, 1.0)) for u in utts])
        else:
            w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(utts) or (w < 0).any():
            raise ValueError("weights must be non-negative, one per utterance (or a dict source -> weight)")
        eligible = np.ones(len(utts), dtype=bool) if self.frames else self.dec_len <= self.max_dec_len
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
        step_s = np.array([sum(self.dur[mb].sum() for mb in step) for step in plan])
        if self.frames:  # no decoder: the CTC target tokens are the targets, padded frames follow the padded audio
            return dict(
                steps=len(plan), micro_batches=len(mbs), utts=int(sum(len(mb) for mb in mbs)), excluded_dec_len=0,
                real_h=float(real.sum() / 3600), pad_eff_audio=float(real.sum() / padded.sum()),
                micro_per_step=float(len(mbs) / len(plan)), step_real_s_mean=float(step_s.mean()),
                step_real_s_min=float(step_s.min()), step_real_s_max=float(step_s.max()),
                micro_padded_s_max=float(padded.max()), micro_utts_max=int(max(len(mb) for mb in mbs)),
                micro_targets_max=int(max(self.n_tok[mb].sum() for mb in mbs)),
                targets=int(sum(self.n_tok[mb].sum() for mb in mbs)),
            )
        dec_real = np.array([self.dec_len[mb].sum() for mb in mbs])
        dec_pad = np.array([self.dec_len[mb].max() * len(mb) for mb in mbs])
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
        the most target positions (decoder activations and the (N, V) logits - these are the SHORT utterances).
        The frame planner (max_dec_len None): the longest and the one with the most padded audio, whose padded
        frames x 3073 CTC logits (fp32, with their log-softmax and gradient) are the CTC student's worst case."""
        plan = self.epoch_plan(0) if plan is None else plan
        mbs = [mb for step in plan for mb in step]
        longest = max(mbs, key=lambda mb: (self.dur[mb].max(), self.dur[mb].max() * len(mb)))
        if self.frames:
            return dict(longest=longest,
                        most_padded_frames=max(mbs, key=lambda mb: float(self.dur[mb].max()) * len(mb)))
        return dict(
            longest=longest,
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

    augment (an Augment with max_tokens set; the train loader's only, scripts/04_distill.py augment.* on an AED student)
    and cuts (a kitsune.aed_cuts.CutIndex over this dataset's rows; needed when augment.truncate_p > 0): each
    micro-batch is augmented after decoding (_augmented; the "AED rows" part of the "augmentation" section below) - its
    rows joined, cut at a cut-table frame and mixed - and collated as above, plus `aug`, a dict of counts (utts, rows,
    concat_groups, concat_utts, truncated, mixed, cut_padded and end_padded - always 0 -, concat_capped,
    cut_mismatch, end_trimmed). A joined row holds several utterances: its ids are theirs joined by "+", sources and index the
    first one's. None (the default) returns the stored rows exactly as before the augmentation existed, without `aug`.
    """

    def __init__(self, stores: Stores, augment: "Augment | None" = None, cuts=None, noise=None, rirs=None):
        u = stores.utts
        self.cache_dir = str(stores.cache_dir)
        self.prompt = np.array(stores.info.get("prompt", PROMPT), dtype=np.int64)
        self.pad = int(stores.info.get("pad", PAD))
        self.eos = int(stores.info.get("eos", EOS))
        self.ids = [x.id for x in u]
        self.sources = [x.source for x in u]
        self.duration = np.array([x.duration for x in u], dtype=np.float32)
        self.agree = np.array([x.agree for x in u], dtype=np.float32)
        self.audio_off = np.array([x.audio_off for x in u], dtype=np.int64)
        self.audio_len = np.array([x.audio_len for x in u], dtype=np.int64)
        self.tok_off = np.array([x.tok_off for x in u], dtype=np.int64)
        self.n_tok = np.array([x.n_tok for x in u], dtype=np.int64)
        self.augment, self.cuts = _token_augment(augment, cuts)
        self.noise = _noise_of(self.augment, noise)
        self.rirs = _rir_of(self.augment, rirs)
        self._mm = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None  # never pickle memmaps: each process maps the files itself
        return state

    def with_augment(self, augment: "Augment | None", cuts=None, noise=None, rirs=None) -> "AudioBatchDataset":
        """This dataset with another augmentation and cut index (None: none), as a shallow copy sharing the
        per-utterance arrays (FrameBatchDataset.with_augment's reason): the trainer keeps the plain dataset for the
        smoke checks and the memory probe."""
        out = copy.copy(self)
        out.augment, out.cuts = _token_augment(augment, cuts)
        out.noise = _noise_of(out.augment, noise)
        out.rirs = _rir_of(out.augment, rirs)
        out._mm = None
        return out

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
        if self.augment is not None:
            return self._augmented(idx, keep, waves, dropped)
        out = self._collate(waves, [self.targets(i) for i in keep])
        out.update(durations=torch.from_numpy(self.duration[keep]), agree=torch.from_numpy(self.agree[keep]),
                   index=torch.tensor(keep, dtype=torch.int64), ids=[self.ids[i] for i in keep],
                   sources=[self.sources[i] for i in keep], dropped=dropped)
        return out

    def _collate(self, waves: list[np.ndarray], targets: list[tuple]) -> dict:
        """The batch contract's tensors (all but the per-row metadata) of rows given as audio and (tokens, topk_idx,
        topk_lp), each row's tokens ending in its EOS."""
        B, P = len(waves), len(self.prompt)
        lengths = np.array([len(w) for w in waves], dtype=np.int64)
        wave = np.zeros((B, int(lengths.max()) if B else 0), dtype=np.float32)
        for b, w in enumerate(waves):
            wave[b, :len(w)] = w
        T = np.array([len(t[0]) for t in targets], dtype=np.int64)
        L = P - 1 + int(T.max()) if B else P
        dec = np.full((B, L), self.pad, dtype=np.int64)
        dec_mask = np.zeros((B, L), dtype=np.int64)
        tgt_mask = np.zeros((B, L), dtype=bool)
        rows, pos, top_idx, top_lp = [], [], [], []
        for b, (tok, ti, tl) in enumerate(targets):
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
            n_tok=torch.from_numpy(T),
        )

    def _augmented(self, idx: Sequence[int], keep: list[int], waves: list[np.ndarray], dropped: list[str]) -> dict:
        """__getitem__ under self.augment: the decoded rows (store indices `keep`, audio `waves`) joined, cut, end-trimmed
        and mixed in that order, each step drawing from augment_rng(seed, idx) - the micro-batch's own stream, as
        FrameBatchDataset._augmented - then collated as the plain path collates. A join is refused (the micro-batch
        stays as it is, concat_capped) when one of its rows would hold more than max_tokens target tokens: the planner
        never planned such a decoder. A cut lands on a frame of the rows' cut-table entries (aed_cut_candidates) and
        keeps the Cohere tokens that entry names, then EOS. Metadata of a row as FrameBatchDataset._augmented's."""
        a, eos = self.augment, self.eos
        rng = augment_rng(a.seed, idx)
        rows = [_TokRow(w, *self.targets(i), i, ctc_frames(len(w))) for w, i in zip(waves, keep)]
        groups = capped = 0
        if a.concat_p > 0 and len(rows) >= 2 and rng.random() < a.concat_p:
            k = concat_k(len(rows), max(r.n_frames for r in rows), a, rng)
            if k:
                order = rng.permutation(len(rows))  # which utterances share a row, and in which order: random
                parts = [[rows[j] for j in order[g:g + k]] for g in range(0, len(rows), k)]
                if all(sum(r.body(eos) for r in p[:-1]) + len(p[-1].tok) <= a.max_tokens for p in parts):
                    rows = [_join_tok_rows(p, eos, rng) for p in parts]
                    groups = len(rows)
                else:
                    capped = 1
        originals = [r.wave for r in rows]  # mix's interferers: the rows before any cut or mix
        n_cut = n_mixed = mismatch = 0
        if a.truncate_p > 0:
            for r in rows:
                if rng.random() < a.truncate_p:
                    frames, kept, pause, bad = aed_cut_candidates(r, self.cuts, a, eos)
                    mismatch += bad
                    if not len(frames):
                        continue
                    if a.truncate_pause_p > 0 and rng.random() < a.truncate_pause_p and pause.any():
                        p = np.flatnonzero(pause)
                        j = int(p[int(rng.integers(len(p)))])
                    else:
                        j = int(rng.integers(len(frames)))
                    r.cut_at(int(frames[j]), int(kept[j]), eos, rng)
                    n_cut += 1
        n_trim = 0
        if a.end_trim_p > 0:
            lo, hi = (int(round(s * TARGET_SR)) for s in END_TRIM_TAIL_S)
            for r in rows:
                if r.cut is not None or rng.random() >= a.end_trim_p:
                    continue
                n = voiced_end(r.wave) + int(rng.integers(lo, hi + 1))
                if n < len(r.wave):
                    r.wave, r.trimmed = r.wave[:n], True
                    n_trim += 1
        if a.mix_p > 0 and len(rows) >= 2:
            for p, r in enumerate(rows):
                if rng.random() >= a.mix_p:
                    continue
                q = int(rng.integers(len(rows) - 1))
                q += q >= p  # another row: a different utterance (every utterance is in exactly one row)
                got = mix_into(r.wave, originals[q], rng, a.mix_snr_db)
                if got is not None:
                    r.wave = got[0]
                    n_mixed += 1
        acoustic = acoustic_chain(rows, originals, self.noise, self.rirs, a, rng)
        B = len(rows)
        durations = np.array([len(r.wave) / TARGET_SR if r.cut is not None or r.trimmed
                              else self.duration[r.pieces].sum(dtype=np.float32) for r in rows], dtype=np.float32)
        agree = np.array([_nanmin(self.agree[r.pieces]) for r in rows], dtype=np.float32)
        out = self._collate([r.wave for r in rows], [(r.tok, r.ti, r.tl) for r in rows])
        out.update(durations=torch.from_numpy(durations), agree=torch.from_numpy(agree),
                   index=torch.tensor([r.pieces[0] for r in rows], dtype=torch.int64),
                   ids=["+".join(self.ids[i] for i in r.pieces) for r in rows],
                   sources=[self.sources[r.pieces[0]] for r in rows], dropped=dropped)
        out["aug"] = dict(utts=len(keep), rows=B, concat_groups=groups, concat_utts=len(keep) if groups else 0,
                          truncated=n_cut, mixed=n_mixed, cut_padded=0, end_padded=0, concat_capped=capped,
                          cut_mismatch=mismatch, end_trimmed=n_trim, **acoustic)
        return out


def _cat(parts: list[np.ndarray], dtype, empty_shape: tuple) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(parts).astype(dtype) if parts else np.zeros(empty_shape, dtype))


class FrameBatchDataset(torch.utils.data.Dataset):
    """dataset[list of store indices] -> one collated micro-batch of a FRAME store (build_frame_stores), the CTC
    family's twin of AudioBatchDataset:

      wave, lengths                    as AudioBatchDataset
      frame_mask, dense_mask, blank_lp, topk_idx, topk_lp, ctc_targets, ctc_target_lengths, n_frames
                                       kitsune.ctc_targets.collate_frame_targets of the rows' FrameTargets, padded to
                                       the batch's longest n_frames (the loss cuts the student's padding frames)
      n_tok (B,) int64                 the CTC target tokens U of each row (= ctc_target_lengths)
      durations, agree, index, ids, sources, dropped   as AudioBatchDataset

    A row whose audio does not decode, or whose decoded length does not give its stored n_frames (ctc_frames), is
    dropped and named in `dropped`. The build's frame preflight removed every mismatched row, so the second case means
    the audio changed since the build (a message on stderr says which). B can be 0: skip such a micro-batch.
    Picklable without the data; the memmaps open lazily in each process.

    augment (an Augment; the train loader's only, scripts/04_distill.py augment.*): each micro-batch is augmented after
    decoding (_augmented; the "augmentation" section below) - its rows joined, cut and mixed - and collated as above,
    plus `aug`, a dict of counts (utts, rows, concat_groups, concat_utts, truncated, mixed). A joined row holds several
    utterances: its ids are theirs joined by "+", sources and index the first one's. None (the default) returns the
    stored rows exactly as before the augmentation existed, without `aug`."""

    ARRAYS = ("frames_blank_lp", "dense_frame", "dense_topk_idx", "dense_topk_lp", "ctc_ids")

    def __init__(self, stores: Stores, augment: "Augment | None" = None, noise=None, rirs=None):
        self.augment = _frame_augment(augment)
        self.noise = _noise_of(self.augment, noise)
        self.rirs = _rir_of(self.augment, rirs)
        u = stores.utts
        d = Path(stores.cache_dir)
        self.cache_dir = str(d)
        self.ids = [x.id for x in u]
        self.sources = [x.source for x in u]
        self.duration = np.array([x.duration for x in u], dtype=np.float32)
        self.agree = np.array([x.agree for x in u], dtype=np.float32)
        self.audio_off = np.array([x.audio_off for x in u], dtype=np.int64)
        self.audio_len = np.array([x.audio_len for x in u], dtype=np.int64)
        rows = np.array([x.row for x in u], dtype=np.int64)
        offs = {name: np.load(d / f"{name}_offsets.npy") for name in ("frames", "dense", "ctc")}
        self.frame_off, self.n_frames = offs["frames"][rows], np.diff(offs["frames"])[rows]
        self.dense_off, self.n_dense = offs["dense"][rows], np.diff(offs["dense"])[rows]
        self.ctc_off, self.n_tok = offs["ctc"][rows], np.diff(offs["ctc"])[rows]
        self._mm = None

    def __len__(self) -> int:
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None  # never pickle memmaps: each process maps the files itself
        return state

    def with_augment(self, augment: "Augment | None", noise=None, rirs=None) -> "FrameBatchDataset":
        """This dataset with another augmentation (None: none), as a shallow copy: the per-utterance arrays (ids,
        offsets, durations; on the full data ~10M rows, most of a GiB) are shared, not built or held a second time, so
        the trainer keeps its plain dataset for the smoke checks and the memory probe next to the augmenting one its
        train loader pickles into the workers."""
        out = copy.copy(self)
        out.augment, out._mm = _frame_augment(augment), None
        out.noise = _noise_of(out.augment, noise)
        out.rirs = _rir_of(out.augment, rirs)
        return out

    def _open(self) -> dict:
        if self._mm is None:
            d = Path(self.cache_dir)
            audio = (np.memmap(d / "audio.bin", dtype=np.uint8, mode="r") if (d / "audio.bin").stat().st_size
                     else np.zeros(0, np.uint8))
            self._mm = dict(audio=audio, **{name: np.load(d / f"{name}.npy", mmap_mode="r") for name in self.ARRAYS})
        return self._mm

    def audio_bytes(self, i: int) -> bytes:
        o = int(self.audio_off[i])
        return self._open()["audio"][o:o + int(self.audio_len[i])].tobytes()

    def targets(self, i: int):
        """kitsune.ctc_targets.FrameTargets of store index i (copies of the stored arrays, stored dtypes)."""
        from kitsune.ctc_targets import FrameTargets

        mm = self._open()
        f0, T = int(self.frame_off[i]), int(self.n_frames[i])
        d0, D = int(self.dense_off[i]), int(self.n_dense[i])
        c0, U = int(self.ctc_off[i]), int(self.n_tok[i])
        return FrameTargets(T, np.array(mm["frames_blank_lp"][f0:f0 + T]),
                            np.array(mm["dense_frame"][d0:d0 + D], dtype=np.int32),
                            np.array(mm["dense_topk_idx"][d0:d0 + D]), np.array(mm["dense_topk_lp"][d0:d0 + D]),
                            np.array(mm["ctc_ids"][c0:c0 + U], dtype=np.int32))

    def __getitem__(self, idx: Sequence[int]) -> dict:
        from kitsune.ctc_targets import collate_frame_targets

        keep, waves, dropped = [], [], []
        for i in idx:
            i = int(i)
            try:
                w = decode_audio(self.audio_bytes(i))
            except Exception as e:
                print(f"trainset: undecodable audio, dropping {self.ids[i]}: {e}", file=sys.stderr)
                dropped.append(self.ids[i])
                continue
            if ctc_frames(len(w)) != int(self.n_frames[i]):
                print(f"trainset: {self.ids[i]} decodes to {len(w)} samples = {ctc_frames(len(w))} frames, its targets "
                      f"have {int(self.n_frames[i])}: dropping it (the store's frame preflight passed it)",
                      file=sys.stderr)
                dropped.append(self.ids[i])
                continue
            waves.append(w)
            keep.append(i)
        if self.augment is not None:
            return self._augmented(idx, keep, waves, dropped)
        B = len(keep)
        lengths = np.array([len(w) for w in waves], dtype=np.int64)
        wave = np.zeros((B, int(lengths.max()) if B else 0), dtype=np.float32)
        for b, w in enumerate(waves):
            wave[b, :len(w)] = w
        out = dict(wave=torch.from_numpy(wave), lengths=torch.from_numpy(lengths),
                   durations=torch.from_numpy(self.duration[keep]), agree=torch.from_numpy(self.agree[keep]),
                   index=torch.tensor(keep, dtype=torch.int64), ids=[self.ids[i] for i in keep],
                   sources=[self.sources[i] for i in keep], dropped=dropped)
        if B:
            out.update(collate_frame_targets([self.targets(i) for i in keep]))
        else:  # the empty batch's shapes (the trainer skips it)
            out.update(_empty_frame_targets())
        out["n_tok"] = out["ctc_target_lengths"].clone()
        return out

    def _augmented(self, idx: Sequence[int], keep: list[int], waves: list[np.ndarray], dropped: list[str]) -> dict:
        """__getitem__ under self.augment: the decoded rows (store indices `keep`, audio `waves`) joined, cut, padded
        and mixed in that order, each step drawing from augment_rng(seed, idx) - the micro-batch's own stream, a
        function of its index list alone - then collated as the plain path collates. The order matters: a joined row can
        be cut (inside any piece, never before truncate_min_s), a pad follows the cut (or the whole row's end), and mix
        takes its interferers from the rows as they were before the cut and before any mixing (clean speech, the length
        of the whole row). Metadata of a row: ids its pieces' joined by "+" (a cut row only those that start before the
        cut), sources and index the first piece's (the trainer's per-source sums count a joined row under its first
        piece's source), durations the pieces' stored durations summed (a cut or padded row: the seconds of audio it
        holds), agree the smallest known one (NaN if none is known)."""
        from kitsune.ctc_targets import collate_frame_targets

        a = self.augment
        rng = augment_rng(a.seed, idx)
        rows = [_AugRow(w, self.targets(i), [i], [0]) for w, i in zip(waves, keep)]
        groups = 0
        if a.concat_p > 0 and len(rows) >= 2 and rng.random() < a.concat_p:
            k = concat_k(len(rows), max(r.ft.n_frames for r in rows), a, rng)
            if k:
                order = rng.permutation(len(rows))  # which utterances share a row, and in which order: random
                rows = [_join_rows([rows[j] for j in order[g:g + k]], rng) for g in range(0, len(rows), k)]
                groups = len(rows)
        # mix's interferers and the pads' room tone: the rows before any cut (a view) or mix (a copy)
        originals = [r.wave for r in rows]
        # the micro-batch's padded frames after the join, before any cut: a pad never makes a row longer than this, so
        # the rows x longest rectangle the planner and the memory probe sized stays as it is
        width = max((r.ft.n_frames for r in rows), default=0)
        n_cut = n_mixed = n_cut_pad = n_end_pad = 0
        if a.truncate_p > 0:
            for r in rows:
                if rng.random() < a.truncate_p:
                    c = truncate_cut(r.ft, rng, a.truncate_min_frac, a.truncate_min_s, a.punct_ids,
                                     a.truncate_pause_p, a.truncate_min_row_s)
                    if c is not None:
                        r.cut_at(c, rng)
                        n_cut += 1
        if a.truncate_pad_p > 0 or a.end_pad_p > 0:
            for p, r in enumerate(rows):
                prob = a.truncate_pad_p if r.cut is not None else a.end_pad_p
                if prob <= 0 or rng.random() >= prob:
                    continue
                room = width - r.ft.n_frames
                if room < 1:  # the longest whole row: a pad would widen the micro-batch
                    continue
                r.pad_quiet(min(int(rng.integers(a.pad_frames[0], a.pad_frames[1] + 1)), room), originals[p], rng)
                if r.cut is not None:
                    n_cut_pad += 1
                else:
                    n_end_pad += 1
        n_trim = 0
        if a.end_trim_p > 0:
            for r in rows:
                if r.cut is None and not r.pad and rng.random() < a.end_trim_p and r.trim_end(a.punct_ids, rng):
                    n_trim += 1
        if a.mix_p > 0 and len(rows) >= 2:
            for p, r in enumerate(rows):
                if rng.random() >= a.mix_p:
                    continue
                q = int(rng.integers(len(rows) - 1))
                q += q >= p  # another row: a different utterance (every utterance is in exactly one row)
                got = mix_into(r.wave, originals[q], rng, a.mix_snr_db)
                if got is not None:
                    r.wave = got[0]
                    n_mixed += 1
        acoustic = acoustic_chain(rows, originals, self.noise, self.rirs, a, rng)
        B = len(rows)
        lengths = np.array([len(r.wave) for r in rows], dtype=np.int64)
        wave = np.zeros((B, int(lengths.max()) if B else 0), dtype=np.float32)
        for b, r in enumerate(rows):
            wave[b, :len(r.wave)] = r.wave
        durations = np.array([len(r.wave) / TARGET_SR if r.cut is not None or r.pad or r.trimmed
                              else self.duration[r.pieces].sum(dtype=np.float32) for r in rows], dtype=np.float32)
        agree = np.array([_nanmin(self.agree[r.pieces]) for r in rows], dtype=np.float32)
        out = dict(wave=torch.from_numpy(wave), lengths=torch.from_numpy(lengths),
                   durations=torch.from_numpy(durations), agree=torch.from_numpy(agree),
                   index=torch.tensor([r.pieces[0] for r in rows], dtype=torch.int64),
                   ids=["+".join(self.ids[i] for i in r.pieces) for r in rows],
                   sources=[self.sources[r.pieces[0]] for r in rows], dropped=dropped)
        out.update(collate_frame_targets([r.ft for r in rows]) if B else _empty_frame_targets())
        out["n_tok"] = out["ctc_target_lengths"].clone()
        out["aug"] = dict(utts=len(keep), rows=B, concat_groups=groups, concat_utts=len(keep) if groups else 0,
                          truncated=n_cut, mixed=n_mixed, cut_padded=n_cut_pad, end_padded=n_end_pad,
                          end_trimmed=n_trim, **acoustic)
        return out


def _empty_frame_targets() -> dict:
    """collate_frame_targets' keys for a micro-batch that lost every row (the trainer skips it)."""
    return dict(frame_mask=torch.zeros(0, 0, dtype=torch.bool), dense_mask=torch.zeros(0, 0, dtype=torch.bool),
                blank_lp=torch.zeros(0, 0), topk_idx=torch.zeros(0, 0, 0, dtype=torch.long),
                topk_lp=torch.zeros(0, 0, 0), ctc_targets=torch.zeros(0, dtype=torch.long),
                ctc_target_lengths=torch.zeros(0, dtype=torch.long), n_frames=torch.zeros(0, dtype=torch.long))


def _nanmin(x: np.ndarray) -> float:
    v = x[~np.isnan(x)]
    return float(v.min()) if len(v) else float("nan")


# ---------------------------------------------------------------------------------------------------- augmentation
#
# The train-time augmentation (scripts/04_distill.py augment.*, off by default; FrameBatchDataset(augment=Augment(...))
# for the CTC family, AudioBatchDataset(augment=..., cuts=...) for the AED students - "AED rows" below), in the loader's
# workers, after decoding. Why: trained on the stored rows alone, the student
# learns two artefacts of its data rather than of speech (the owner's review of P-0.1B inside the app):
#   - every row is ONE whole utterance that ends in a sentence mark, so "the input ends" becomes "emit 。": a decoded
#     chunk gets 。/？/！ at its end whether the speaker stopped or not (moving the chunk's end 3 s later moves the mark
#     with it, 30 of 30), and a real sentence end inside a long chunk gets none;
#   - no row ever holds two utterances or a second voice: joined utterances (ReazonSpeech 11.9 % CER per utterance,
#     33.7 % with 4 of them in one input; CV 10.3 -> 15.9 %) and conversational streams with crosstalk go badly.
# Three fixes, each keeping the teacher's per-frame targets right for the input the student gets:
#   concat    per micro-batch (concat_p): ALL its rows joined into groups of k, k drawn among the divisors of the row
#             count in [2, concat_max_n] with k x the longest row <= concat_max_s (none: no join). Every piece but the
#             last is padded (AUG_PAD_STD noise) or trimmed to exactly FRAME_SAMPLES x its stored n_frames, so each
#             piece starts on an encoder frame boundary and its targets hold unchanged at its frame offset; the targets
#             are concatenated and the greedy CTC path recomputed over the joined argmax. Real sentence marks then sit
#             INSIDE long rows, and multi-utterance inputs of up to concat_max_s become ordinary. The joined micro-batch
#             has rows / k rows of at most k x its longest row's frames: its padded frames (rows x longest, what the
#             encoder's activations and the (frames x 3073) logits scale with) never exceed the planned micro-batch's,
#             and the trainer clamps concat_max_s to the longest train utterance, so no joined row is longer than the
#             one the memory probe's "longest" micro-batch already ran. The audio rectangle can grow by the pads alone
#             (< 70 ms per join, read by the convolutional subsampling only)
#   truncate  per row (truncate_p): cut at a frame c >= max(truncate_min_frac x T, truncate_min_s) whose first removed
#             token (the first token starting at or after c in the teacher's argmax) is a WORD - never a sentence mark
#             or comma (punct_ids, the student tokenizer's) -, so the kept part ends inside a sentence with at least one
#             of its words gone; truncate_pause_p of the cuts land inside a pause (>= TRUNCATE_PAUSE_FRAMES teacher-blank
#             frames), as a voice-detector chunk ends. The targets at frame c; the audio ends at a sample drawn
#             uniformly where it is still c frames (end_samples: FRAME_SAMPLES x c - 1120 .. + 159), as a natural
#             utterance or an app's chunk ends anywhere: the first recipe cut at exactly FRAME_SAMPLES x c, and its
#             student learned "an input that ends on an 80 ms boundary has no mark" (the external review's own cuts:
#             2.8 % false marks on a boundary, 100 % mid-frame; and the natural ends that happen to fill their last
#             frame lost their real mark). Rows shorter than truncate_min_row_s are never cut (short utterances lost
#             more words under the first recipe).
#             The teacher made its targets from the WHOLE utterance, so the frames before c carry no final mark:
#             the input ending is no longer a reason to emit one. A cut between a sentence's last word and its mark
#             (the Parakeet teacher puts the mark after the trailing silence) would keep a COMPLETE sentence without its
#             mark and teach the opposite - dropping real marks -, so the word rule excludes it (truncate_frames)
#   pad       per cut row (truncate_pad_p) and per whole row (end_pad_p): n frames of quiet appended (n uniform in
#             pad_frames, 80 ms each; quiet_pad: PAD_ROOM_P of the pads the row's own room tone, the rest AUG_PAD_STD
#             noise) with pure-blank targets (blank_frame_targets). Why (the external review of the first recipe's
#             test model, 2026-10-03): the app ends a chunk at a window edge - mostly inside a word - and pads it with
#             0.2 s of quiet, and a cut that only ever ENDED the input taught nothing about "the speech stops, then
#             quiet": that model still put 。 after 99.7 % of cuts through a word. A cut row with a pad keeps no mark
#             (its sentence still runs); a whole row with a pad keeps its mark before the quiet (a complete sentence,
#             then silence), so silence alone is no cue either way. The audio is first squared to whole frames
#             (join_waves), so the pad starts on an encoder frame boundary. A pad never makes a row longer than the
#             micro-batch's longest row after the join (the planned rectangle): the longest whole row gets none, and a
#             pad is clipped to the room left
#   mix       per row (mix_p, micro-batches of 2+ rows): a random segment of another row's audio - as it was before any
#             cut or mix: clean, whole - added at a random offset, scaled to an SNR drawn from mix_snr_db (dB of this
#             row's power over the segment's, both over the span they share); the targets stay this row's (clean
#             teacher, noisy student), so the student transcribes the dominant voice through crosstalk and background
#             talk. The length never changes
# No step changes a row's frame count other than as its targets change (concat sums whole frames, truncate cuts at a
# frame boundary, a pad adds whole frames, mix keeps the length), so ctc_frames(samples) == n_frames holds for every row
# the loss reads.
# Deterministic: each micro-batch draws from augment_rng(seed, its index list) alone, so a resumed or branched run -
# which replays the same index lists - augments every micro-batch exactly as the original did, whichever worker decodes
# it (a micro-batch whose index list recurs in a later epoch is augmented the same way again; with the planner's
# reshuffled pools that needs the same utterances in the same order, which only a one-row micro-batch meets). Pure numpy
# in the worker - a join, a slice, one copy per mixed row -: measured on the laptop, 0.05-0.35 s more per 1600 s
# micro-batch with all three on (the most for 1 s rows, 1600 of them), next to the ~0.7-1.6 s its MP3/OGG decode takes
# on one core (default_num_workers' rates).
# Acoustic steps (DECISIONS H12; acoustic_chain), after the joins, cuts, pads, end trim and mix, for both families,
# each per row and from the same stream, the targets always the teacher's on the clean row - so the student learns that
# none of these changes what is said:
#   speech    background speech (speech_p): 1-4 voices (speech_talkers), each another row of the micro-batch (Japanese)
#             or the bank's speech (MUSAN: 17 languages), summed - two or more are babble - at speech_snr_db (10-25 dB)
#             of voiced power below the row, which stays the dominant voice (mix_into's single interferer at 5-20 dB
#             made two equally loud voices worse in the recipe test: these stay well below)
#   reverb    room echo (reverb_p): a room impulse response (aug/rirs-v1: OpenSLR 28's simulated and real rooms) over
#             the whole row, background speech included - one room -, its direct path at sample 0 (frames aligned)
#   noise     music without vocals, noise and songs with lyrics (noise_p; BACKGROUND_KINDS) under the whole row, dry
#   gain      volume (gain_p): gain_db (-20..+10 dB), clipped at full scale
#   codec     a real encoder and back in memory (codec_p; kitsune.acoustics: MP3 at 30-100 kbit/s, GSM 6.10 and
#             mu-law at 8 kHz), length and alignment kept (measured: lag 0 for each)
# background_min_row_s > 0 (DECISIONS H14's gentle recipe): a row whose audio at the chain's entry - as the student hears
# it, after the joins, cuts, pads and trim - is shorter than that many seconds is spared the speech and background
# steps (a short command under a song or babble is mostly the song): their gate is still drawn for it, so the rows
# before it draw what they drew, then it is skipped; reverb, gain and codec still apply. The chain then counts
# short_rows, speech_spared and noise_spared (0 included) on every micro-batch; at 0 (the default) it never does, and
# the stream and counts are the recipe's as before the key
SPEECH_KINDS = ("speech",)  # the bank's kinds the speech step draws
BACKGROUND_KINDS = ("music", "noise", "song")  # the bank's kinds the background step draws
SPEECH_MAX_TALKERS = 8  # the most voices speech_talkers may ask for

AUG_PAD_STD = 1e-5  # the noise between joined pieces (~-100 dBFS): never digital zeros, whose LogMel floor is a feature
# no recording has, which the student would then see at every join and nowhere else
MIX_SILENT_POWER = 1e-8  # a mean square below this (-80 dBFS) is silence: no level to scale an interferer to (or by)
MIX_MIN_COVER = 0.5  # the interferer's segment covers a uniform share in [MIX_MIN_COVER, 1] of the shorter of the rows
TRUNCATE_PAUSE_FRAMES = 4  # a pause cut lands in a run of >= this many teacher-blank frames (>= 320 ms of pause)
PAD_MAX_FRAMES = 25  # the longest pad augment.pad_frames may ask for (2 s)
# where a cut (or padded) row's audio may end and still be c encoder frames: ctc_frames(L) == c for every L in
# [FRAME_SAMPLES c - END_BELOW, FRAME_SAMPLES c + END_ABOVE] (the valid mel frames L // HOP run 8c - 7 .. 8c)
END_BELOW = FRAME_SAMPLES - HOP  # 1120 samples
END_ABOVE = HOP - 1  # 159 samples
PAD_ROOM_P = 0.5  # the share of pads cut from the row's own room tone (a voice-detector chunk's silence); the rest is
# AUG_PAD_STD noise (an app's zero pad, without the exact zeros whose LogMel floor no recording has)
PAD_ROOM_FRAMES = 8  # room tone: whole frames drawn among the row's this-many quietest ...
PAD_ROOM_MAX_POWER = 1e-4  # ... whose mean square is below this (-40 dBFS); a row without one gets noise
PAD_MIN_BLANK_LP = math.log(0.99)  # a pad frame's log p(blank): the row's most confident blank frame's, at least this


@dataclass(frozen=True)
class Augment:
    """What a train FrameBatchDataset (or AudioBatchDataset) does to each micro-batch (the section comment above).
    scripts/04_distill.py
    builds it from augment.* (from_config) with seed = augment.seed or the run's seed, concat_max_s clamped to the
    longest train utterance and punct_ids from the student's tokenizer. concat_p: per micro-batch; truncate_p, mix_p:
    per row; truncate_min_frac / truncate_min_s: the cut's lower bound (a share of the row's frames, seconds);
    truncate_pause_p: the share of cuts placed inside a pause (truncate_cut); punct_ids: the token ids a cut must never
    remove alone - the sentence marks and commas (truncate_frames; required when truncate_p > 0, since without them
    the rule cannot tell a sentence's mark from its words); concat_max_n / concat_max_s: the pieces and seconds of a
    joined row; mix_snr_db: (low, high) dB of the row's power over the interferer's, drawn uniformly; truncate_pad_p /
    end_pad_p: per cut row / per whole row, a quiet pad of (low, high) pad_frames 80 ms frames (inclusive) after it;
    truncate_min_row_s: rows shorter than this (seconds, the row's frames) are never cut.
    max_tokens: set for an AED (token-target) dataset - AudioBatchDataset's augmentation, the section comment's "AED
    rows" - and None for the CTC family's: the most target tokens (EOS included) a joined row may hold, the planner's
    max_dec_len less the prompt's length - 1. An AED augmentation takes no punct_ids (its cuts come from the cut table,
    whose rule already keeps every cut before a word) and no pads (truncate_pad_p and end_pad_p 0). end_trim_p: per
    row that was not cut (nor padded), trim its trailing silence to a short drawn tail, its sentence mark kept (end
    trim: an AED row's tokens as they are, a CTC row's mark frames moved up to its last word; a CTC one needs
    punct_ids). noise_p: per row, background audio from the bank (kitsune.noise_bank: its music, noise and songs)
    mixed under the whole row at an SNR drawn from noise_snr_db ((low, high) dB of the row's voiced power over the
    background's). The other acoustic steps (DECISIONS H12; the section comment's "acoustic steps"), each per row:
    speech_p - background speech, speech_talkers (low, high) voices summed, each another row of the micro-batch with
    probability speech_batch_p or the bank's speech, at speech_snr_db dB of voiced power below the row; reverb_p - a
    room impulse response from the RIR bank; gain_p - gain_db dB, clipped at full scale; codec_p - one of codecs
    (kitsune.acoustics) and back. background_min_row_s: rows shorter than this (seconds of their audio at the acoustic
    chain) get no background speech and no background (0: every row may; acoustic_chain)."""

    seed: int
    truncate_p: float = 0.0
    truncate_min_frac: float = 0.3
    truncate_min_s: float = 1.0
    truncate_pause_p: float = 0.5
    punct_ids: tuple = ()
    concat_p: float = 0.0
    concat_max_s: float = 28.0
    concat_max_n: int = 4
    mix_p: float = 0.0
    mix_snr_db: tuple = (5.0, 20.0)
    truncate_pad_p: float = 0.0
    end_pad_p: float = 0.0
    pad_frames: tuple = (1, 5)
    truncate_min_row_s: float = 0.0
    max_tokens: int | None = None
    end_trim_p: float = 0.0
    noise_p: float = 0.0
    noise_snr_db: tuple = (0.0, 20.0)
    speech_p: float = 0.0
    speech_snr_db: tuple = (10.0, 25.0)
    speech_talkers: tuple = (1, 4)
    speech_batch_p: float = 0.5
    reverb_p: float = 0.0
    gain_p: float = 0.0
    gain_db: tuple = (-20.0, 10.0)
    codec_p: float = 0.0
    codecs: tuple = DEFAULT_CODECS
    background_min_row_s: float = 0.0

    def __post_init__(self):
        snr = tuple(float(x) for x in self.mix_snr_db)
        object.__setattr__(self, "mix_snr_db", snr)
        nsnr = tuple(float(x) for x in self.noise_snr_db)
        object.__setattr__(self, "noise_snr_db", nsnr)
        ssnr = tuple(float(x) for x in self.speech_snr_db)
        object.__setattr__(self, "speech_snr_db", ssnr)
        gdb = tuple(float(x) for x in self.gain_db)
        object.__setattr__(self, "gain_db", gdb)
        st = tuple(self.speech_talkers)
        st_whole = len(st) == 2 and all(float(x) == int(x) for x in st)
        object.__setattr__(self, "speech_talkers", tuple(int(x) for x in st) if st_whole else st)
        object.__setattr__(self, "codecs", tuple(str(c) for c in self.codecs))
        object.__setattr__(self, "punct_ids", tuple(sorted(int(i) for i in self.punct_ids)))
        pf = tuple(self.pad_frames)
        whole = len(pf) == 2 and all(float(x) == int(x) for x in pf)
        object.__setattr__(self, "pad_frames", tuple(int(x) for x in pf) if whole else pf)
        probs = (self.truncate_p, self.truncate_pause_p, self.concat_p, self.mix_p, self.truncate_pad_p,
                 self.end_pad_p, self.end_trim_p, self.noise_p, self.speech_p, self.speech_batch_p, self.reverb_p,
                 self.gain_p, self.codec_p)
        aed = self.max_tokens is not None
        ok = (int(self.seed) >= 0 and all(0.0 <= float(p) <= 1.0 for p in probs)
              and 0.0 <= float(self.truncate_min_frac) < 1.0 and float(self.truncate_min_s) >= 0.0
              and float(self.concat_max_s) > 0.0 and int(self.concat_max_n) >= 2 and len(snr) == 2
              and 0.0 <= snr[0] <= snr[1] and all(0 <= i < CTC_BLANK for i in self.punct_ids)
              and (aed or float(self.truncate_p) == 0.0 or len(self.punct_ids) > 0)
              and whole and 1 <= self.pad_frames[0] <= self.pad_frames[1] <= PAD_MAX_FRAMES
              and float(self.truncate_min_row_s) >= 0.0
              and (not aed or (int(self.max_tokens) >= 2 and not self.punct_ids
                               and float(self.truncate_pad_p) == 0.0 and float(self.end_pad_p) == 0.0))
              and (aed or float(self.end_trim_p) == 0.0 or len(self.punct_ids) > 0)
              and len(nsnr) == 2 and nsnr[0] <= nsnr[1]
              and len(ssnr) == 2 and all(np.isfinite(ssnr)) and ssnr[0] <= ssnr[1]
              and len(gdb) == 2 and all(np.isfinite(gdb)) and gdb[0] <= gdb[1]
              and st_whole and 1 <= self.speech_talkers[0] <= self.speech_talkers[1] <= SPEECH_MAX_TALKERS
              and all(c in CODECS for c in self.codecs) and (float(self.codec_p) == 0.0 or len(self.codecs) > 0)
              and math.isfinite(float(self.background_min_row_s)) and float(self.background_min_row_s) >= 0.0)
        if not ok:
            raise ValueError(f"not an augmentation: {self} (probabilities in [0, 1], truncate_min_frac in [0, 1), "
                             "truncate_min_s >= 0, concat_max_s > 0, concat_max_n >= 2, 0 <= mix_snr_db low <= high, "
                             "punct_ids token ids below the blank and given whenever truncate_p > 0, pad_frames whole "
                             f"(low, high) with 1 <= low <= high <= {PAD_MAX_FRAMES}, truncate_min_row_s >= 0; an AED "
                             "one (max_tokens set): max_tokens >= 2, no punct_ids, truncate_pad_p and end_pad_p 0; "
                             "a CTC one's end_trim_p needs punct_ids; noise_snr_db, speech_snr_db and gain_db finite "
                             f"(low, high) with low <= high; speech_talkers whole with 1 <= low <= high <= "
                             f"{SPEECH_MAX_TALKERS}; codecs among {CODECS}, at least one when codec_p > 0; "
                             "background_min_row_s finite and >= 0)")

    @classmethod
    def from_config(cls, block: dict, seed: int, concat_max_s: float | None = None,
                    punct_ids: Sequence[int] = (), max_tokens: int | None = None) -> "Augment":
        """From scripts/04_distill.py's augment block: `enabled` and `seed` are the caller's (seed: the resolved one),
        concat_max_s (when given) replaces the block's - the trainer's clamp -, punct_ids the student tokenizer's (a CTC
        student's), max_tokens the planner's (an AED student's)."""
        kw = {f.name: block[f.name] for f in fields(cls)
              if f.name not in ("seed", "punct_ids", "max_tokens") and f.name in block}
        if concat_max_s is not None:
            kw["concat_max_s"] = float(concat_max_s)
        return cls(seed=int(seed), punct_ids=tuple(punct_ids), max_tokens=max_tokens, **kw)


def _augment_spec(augment) -> "Augment | None":
    if augment is None or isinstance(augment, Augment):
        return augment
    raise TypeError(f"augment must be an Augment or None, got {type(augment).__name__}")


def _noise_of(augment, noise):
    """A dataset's background bank (kitsune.noise_bank.NoiseBank): required when its augmentation's noise_p > 0, or its
    speech_p > 0 with talkers from the bank (speech_batch_p < 1); none without an augmentation."""
    if augment is None:
        if noise is not None:
            raise ValueError("a noise bank without an augmentation")
        return None
    if augment.noise_p > 0 and noise is None:
        raise ValueError("augment.noise_p > 0 needs a noise bank (kitsune.noise_bank: augment.noise_bank)")
    if augment.speech_p > 0 and augment.speech_batch_p < 1 and (noise is None or not noise.has(SPEECH_KINDS)):
        raise ValueError("augment.speech_p > 0 with speech_batch_p < 1 needs a noise bank with speech clips")
    return noise


def _rir_of(augment, rirs):
    """A dataset's room impulse responses (a NoiseBank of kind rir): required when its augmentation's reverb_p > 0."""
    if augment is None:
        if rirs is not None:
            raise ValueError("an RIR bank without an augmentation")
        return None
    if augment.reverb_p > 0 and rirs is None:
        raise ValueError("augment.reverb_p > 0 needs an RIR bank (augment.rir_bank)")
    return rirs


def _frame_augment(augment) -> "Augment | None":
    """A FrameBatchDataset's augmentation: a CTC one (max_tokens None)."""
    augment = _augment_spec(augment)
    if augment is not None and augment.max_tokens is not None:
        raise ValueError("an AED augmentation (max_tokens set) on a frame store: the CTC family's takes punct_ids and "
                         "no max_tokens")
    return augment


def _token_augment(augment, cuts) -> "tuple[Augment | None, object]":
    """An AudioBatchDataset's (augmentation, cut index): an AED augmentation (max_tokens set), with a cut index when it
    cuts; no cut index without an augmentation."""
    augment = _augment_spec(augment)
    if augment is None:
        if cuts is not None:
            raise ValueError("a cut index without an augmentation")
        return None, None
    if augment.max_tokens is None:
        raise ValueError("a CTC augmentation (no max_tokens) on a token store: an AED student's needs max_tokens - the "
                         "planner's decoder cap a joined row must keep")
    if augment.truncate_p > 0 and cuts is None:
        raise ValueError("augment.truncate_p > 0 on a token store needs the cut table's index (kitsune.aed_cuts): an "
                         "AED row has no frame targets to say what a cut keeps")
    return augment, cuts


def augment_rng(seed: int, idx: Sequence[int]) -> np.random.Generator:
    """The generator of one micro-batch's augmentation: seeded with (seed, a 64-bit BLAKE2b digest of its index list
    as int64), a pure function of the list - not of the worker, the process or anything drawn before it."""
    h = hashlib.blake2b(np.asarray([int(i) for i in idx], dtype=np.int64).tobytes(), digest_size=8).digest()
    return np.random.default_rng(np.random.SeedSequence([int(seed), int.from_bytes(h, "little")]))


def greedy_ids(col0: np.ndarray) -> np.ndarray:
    """kitsune.parakeet_targets.ctc_greedy without the Python loop (as build_frame_stores computes a store's ctc_ids):
    the argmax per frame with repeats collapsed and blanks dropped, int32."""
    col0 = np.asarray(col0, dtype=np.int64)
    if not len(col0):
        return np.zeros(0, np.int32)
    prev = np.concatenate([[-1], col0[:-1]])
    return col0[(col0 != prev) & (col0 != CTC_BLANK)].astype(np.int32)


def join_frame_targets(parts: Sequence) -> "FrameTargets":
    """One joined row's targets from its pieces' FrameTargets, in order: piece j's frames at offset sum(n_frames of the
    pieces before it) - blank_lp and the top-k concatenated, the dense frames shifted by the offset - and the greedy CTC
    path recomputed over the joined argmax (greedy_ids). That is the pieces' own paths one after another, except where a
    piece ends and the next starts on the same token: the joined frames then hold one run of it, which CTC reads as one
    token, so the target must too (the same token twice is two runs only with a blank between)."""
    from kitsune.ctc_targets import FrameTargets

    offs = np.cumsum([0] + [p.n_frames for p in parts[:-1]])
    ft = FrameTargets(int(sum(p.n_frames for p in parts)), np.concatenate([p.blank_lp for p in parts]),
                      np.concatenate([np.asarray(p.dense_frame, dtype=np.int32) + np.int32(o)
                                      for p, o in zip(parts, offs)]),
                      np.concatenate([p.topk_idx for p in parts]), np.concatenate([p.topk_lp for p in parts]),
                      np.zeros(0, np.int32))
    ft.ctc_ids = greedy_ids(ft.col0())
    return ft


def cut_frame_targets(ft: "FrameTargets", c: int) -> "FrameTargets":
    """The first c frames of ft (a cut row's targets): blank_lp[:c], the dense frames below c with their top-k, and
    the greedy CTC path of the argmax kept - a prefix of ft's path (a token whose run the cut splits is still emitted
    once, as CTC reads those frames)."""
    from kitsune.ctc_targets import FrameTargets

    c = int(c)
    nd = int(np.searchsorted(ft.dense_frame, c))  # the dense frames increase
    out = FrameTargets(c, ft.blank_lp[:c].copy(), ft.dense_frame[:nd].copy(), ft.topk_idx[:nd].copy(),
                       ft.topk_lp[:nd].copy(), np.zeros(0, np.int32))
    out.ctc_ids = greedy_ids(out.col0())
    return out


def next_token_start(col0: np.ndarray) -> np.ndarray:
    """For every frame t, the class of the first token that STARTS at or after t (a frame whose class is not blank and
    differs from the frame before it: where CTC reads a new token), CTC_BLANK when none does. A cut at frame t keeps the
    frames before t, so this is the first token the cut removes whole; a token whose run began before t is kept (the
    kept frames still hold the start of its run, and CTC emits it once)."""
    col0 = np.asarray(col0, dtype=np.int64)
    T = len(col0)
    prev = np.concatenate([[-1], col0[:-1]])
    st = np.flatnonzero((col0 != CTC_BLANK) & (col0 != prev))
    if not len(st):
        return np.full(T, CTC_BLANK, dtype=np.int64)
    pos = np.searchsorted(st, np.arange(T))
    return np.where(pos < len(st), col0[st[np.minimum(pos, len(st) - 1)]], CTC_BLANK)


def pause_frames(col0: np.ndarray, min_len: int = 0) -> np.ndarray:
    """bool (T,): the frames inside a run of at least min_len (default TRUNCATE_PAUSE_FRAMES) teacher-blank frames -
    a pause, where the app's voice detector would cut."""
    b = np.asarray(col0) == CTC_BLANK
    out = np.zeros(len(b), dtype=bool)
    if not len(b):
        return out
    min_len = int(min_len) or TRUNCATE_PAUSE_FRAMES
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= min_len:
            out[s:e] = True
    return out


def truncate_frames(ft: "FrameTargets", min_frac: float, min_s: float, punct_ids: Sequence[int],
                    min_row_s: float = 0.0) -> np.ndarray:
    """The frames truncate may cut a row at (increasing): c >= lo = max(ceil(min_frac x T), the frames of min_s, 1),
    and the first token the cut removes whole (next_token_start) is a WORD, never one of punct_ids (the vocabulary's
    sentence marks and commas, scripts/04_distill.py punct_token_ids) and never none. So the kept frames always end
    INSIDE a sentence, with at least one word of it gone, and their targets (the teacher's, made on the whole row) carry
    no final mark: the student learns that the input ending is no reason to emit one. A cut between a sentence's last
    word and its mark - where the Parakeet teacher puts the mark, after the trailing silence: ~22-28 % of the frames a
    plain "before the last token" bound allows (2026-10-02 measurement on the train shards) - would instead keep a
    complete sentence without its mark and teach the student to drop real marks; this rule excludes it, also before the
    mark of a piece inside a joined row (whose own sentence end, after its mark, stays a valid cut). A row shorter
    than min_row_s seconds (its frames) has none."""
    if ft.n_frames * FRAME_SAMPLES < float(min_row_s) * TARGET_SR - 1e-6:
        return np.zeros(0, np.int64)
    col0 = ft.col0()
    nxt = next_token_start(col0)
    ok = (nxt != CTC_BLANK) & ~np.isin(nxt, np.asarray(list(punct_ids), dtype=np.int64))
    lo = max(math.ceil(float(min_frac) * ft.n_frames - 1e-9),
             math.ceil(float(min_s) * TARGET_SR / FRAME_SAMPLES - 1e-9), 1)
    ok[:lo] = False
    return np.flatnonzero(ok)


def truncate_cut(ft: "FrameTargets", rng: np.random.Generator, min_frac: float, min_s: float,
                 punct_ids: Sequence[int], pause_p: float = 0.0, min_row_s: float = 0.0) -> int | None:
    """truncate's cut frame: with probability pause_p one drawn uniformly among the truncate_frames that lie inside a
    pause (pause_frames: the kept part then ends in silence, as a voice-detector chunk does, with the sentence still
    running), otherwise - or when there is no such frame - uniformly among all truncate_frames. None when there are
    none: a row without a word token after lo (no token, a short row, one whose last word starts before lo)."""
    cand = truncate_frames(ft, min_frac, min_s, punct_ids, min_row_s)
    if not len(cand):
        return None
    if pause_p > 0 and rng.random() < pause_p:
        p = cand[pause_frames(ft.col0())[cand]]
        if len(p):
            return int(p[int(rng.integers(len(p)))])
    return int(cand[int(rng.integers(len(cand)))])


def concat_k(n_rows: int, longest_frames: int, a: Augment, rng: np.random.Generator) -> int:
    """concat's group size for a micro-batch of n_rows rows whose longest has longest_frames frames: drawn uniformly
    among the k in [2, concat_max_n] that divide n_rows (every row joins a group of the same size, so the rows x
    longest rectangle is at most the original's) and keep k x the longest row (in whole frames, the length a piece
    takes in a joined row) within concat_max_s; 0 when none does - that micro-batch stays as it is."""
    longest_s = int(longest_frames) * FRAME_SAMPLES / TARGET_SR
    ks = [k for k in range(2, int(a.concat_max_n) + 1)
          if n_rows % k == 0 and k * longest_s <= float(a.concat_max_s) + 1e-9]
    return ks[int(rng.integers(len(ks)))] if ks else 0


def join_waves(waves: Sequence[np.ndarray], n_frames: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """A joined row's audio: every piece but the last padded with AUG_PAD_STD noise, or cut (by under HOP samples,
    which the extractor's valid mel frames, samples // HOP, never read), to exactly FRAME_SAMPLES x its n_frames, so
    piece j starts at sample FRAME_SAMPLES x its frame offset; the last piece as it is: after a whole number S of
    frames its own length gives its own frames (ctc_frames(FRAME_SAMPLES x S + n) == S + ctc_frames(n)), and padding it
    would only lengthen the row."""
    out = []
    for w, t in zip(waves[:-1], n_frames[:-1]):
        n = FRAME_SAMPLES * int(t)
        out.append(w[:n])
        if len(w) < n:
            out.append(np.float32(AUG_PAD_STD) * rng.standard_normal(n - len(w), dtype=np.float32))
    out.append(waves[-1])
    return np.concatenate(out).astype(np.float32, copy=False)


def mix_into(dst: np.ndarray, src: np.ndarray, rng: np.random.Generator,
             snr_db: Sequence[float]) -> tuple[np.ndarray, dict] | None:
    """dst with a segment of src added: n samples (a uniform share in [MIX_MIN_COVER, 1] of the shorter of the two)
    from a random start in src, at a random offset in dst, scaled so that 10 log10(power(dst over those n samples) /
    power(the scaled segment)) is an SNR drawn uniformly from snr_db. Returns (the new array - dst itself is never
    written: it may be another row's interferer -, {src_start, offset, n, snr_db, gain}), or None when there is
    nothing to mix (a row under 2 samples; either span silent, below MIX_SILENT_POWER: no level to scale to, and
    scaling a silent segment up would only turn its noise floor into a loud hiss)."""
    L = min(len(dst), len(src))
    if L < 2:
        return None
    n = int(rng.integers(math.ceil(MIX_MIN_COVER * L), L + 1))
    s0 = int(rng.integers(0, len(src) - n + 1))
    off = int(rng.integers(0, len(dst) - n + 1))
    snr = float(rng.uniform(float(snr_db[0]), float(snr_db[1])))
    seg = np.asarray(src[s0:s0 + n], dtype=np.float32)
    span = np.asarray(dst[off:off + n], dtype=np.float32)
    # the powers as float32 dot products (BLAS, no float64 copies of up to 30 s of audio per mixed row): ~1e-6 relative,
    # 1e-5 dB of SNR
    p_dst, p_seg = float(np.dot(span, span)) / n, float(np.dot(seg, seg)) / n
    if p_dst < MIX_SILENT_POWER or p_seg < MIX_SILENT_POWER:
        return None
    gain = math.sqrt(p_dst / (p_seg * 10.0 ** (snr / 10.0)))
    out = np.array(dst, dtype=np.float32, copy=True)
    out[off:off + n] += np.float32(gain) * seg
    return out, dict(src_start=s0, offset=off, n=n, snr_db=snr, gain=gain)


def end_samples(c: int, rng: np.random.Generator) -> int:
    """Where a row of c encoder frames ends when it is cut (or padded): a sample drawn uniformly in [FRAME_SAMPLES c -
    END_BELOW, FRAME_SAMPLES c + END_ABOVE], every one of which is still c frames (ctc_frames), so its last frame
    is as full or as empty as a natural utterance's or an app chunk's - never always exactly full, which the first
    recipe's student learned to read as "cut, so no mark"."""
    return FRAME_SAMPLES * int(c) + int(rng.integers(-END_BELOW, END_ABOVE + 1))


def blank_frame_targets(n: int, blank_lp: float, k: int) -> "FrameTargets":
    """n frames of pure blank, a quiet pad's targets: log p(blank) = blank_lp at every frame and no dense frame (the
    teacher's p(blank) there is above its dense threshold), so no token - and no mark - comes from them; top-k width k,
    the row's, for join_frame_targets."""
    from kitsune.ctc_targets import FrameTargets

    n, k = int(n), int(k)
    return FrameTargets(n, np.full(n, blank_lp, dtype=np.float16), np.zeros(0, np.int32), np.zeros((0, k), np.int16),
                        np.zeros((0, k), np.float16), np.zeros(0, np.int32))


def quiet_pad(src: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n x FRAME_SAMPLES samples of quiet for a pad: with probability PAD_ROOM_P the room tone of src (the row's audio
    before any cut) - n whole frames drawn, with replacement and in random order, among its PAD_ROOM_FRAMES quietest
    whose mean square is below PAD_ROOM_MAX_POWER, as a voice-detector chunk ends in its own silence -, otherwise, or
    when src has no such frame, AUG_PAD_STD noise, as an app's zero pad reads."""
    n = int(n)
    if rng.random() < PAD_ROOM_P:
        m = len(src) // FRAME_SAMPLES
        if m:
            fr = np.asarray(src[:m * FRAME_SAMPLES], dtype=np.float32).reshape(m, FRAME_SAMPLES)
            ms = np.einsum("ij,ij->i", fr, fr) / FRAME_SAMPLES
            order = np.argsort(ms, kind="stable")[:PAD_ROOM_FRAMES]
            quiet = order[ms[order] < PAD_ROOM_MAX_POWER]
            if len(quiet):
                return fr[quiet[rng.integers(len(quiet), size=n)]].reshape(-1).copy()
    return np.float32(AUG_PAD_STD) * rng.standard_normal(n * FRAME_SAMPLES, dtype=np.float32)


class _AugRow:
    """One row of an augmented micro-batch: its audio, targets, the store indices of its pieces and their first frames,
    the frame it was cut at (None: whole) and the frames of quiet appended after it (0: none)."""

    __slots__ = ("wave", "ft", "pieces", "offsets", "cut", "pad", "trimmed")

    def __init__(self, wave: np.ndarray, ft, pieces: list[int], offsets: list[int]):
        self.wave, self.ft, self.pieces, self.offsets, self.cut, self.pad = wave, ft, pieces, offsets, None, 0
        self.trimmed = False

    def pad_quiet(self, n: int, src: np.ndarray, rng: np.random.Generator):
        """Append n frames of quiet (quiet_pad, room tone from src) with pure-blank targets at the row's most confident
        blank (at least PAD_MIN_BLANK_LP): the audio first squared to exactly FRAME_SAMPLES x its frames as a joined
        piece is (join_waves), so the pad starts on an encoder frame boundary and the row's frames grow by exactly n;
        the padded row's end drawn as a cut's (end_samples).
        The targets before the pad are unchanged: a cut row still ends without a mark, a whole row keeps its own."""
        n, T = int(n), self.ft.n_frames
        b = max(float(np.max(self.ft.blank_lp)) if T else 0.0, PAD_MIN_BLANK_LP)
        wave = join_waves([self.wave, quiet_pad(src, n, rng)], [T, n], rng)
        self.wave = wave[:min(end_samples(T + n, rng), len(wave))]  # the end drawn, as a cut's (end_samples)
        self.ft = join_frame_targets([self.ft, blank_frame_targets(n, b, self.ft.k)])
        self.pad += n

    def cut_at(self, c: int, rng: np.random.Generator | None = None):
        """Keep the first c frames: the targets cut_frame_targets', only the pieces that start before c, and the audio
        up to end_samples(c, rng) - any sample where it is still c frames - or, without rng, exactly FRAME_SAMPLES x c
        (a row of T > c frames always has the audio: at least FRAME_SAMPLES x T - END_BELOW samples)."""
        n = FRAME_SAMPLES * int(c) if rng is None else min(end_samples(c, rng), len(self.wave))
        if ctc_frames(n) != int(c):  # never for a row of more than c frames; kept as the frame contract's guard
            n = FRAME_SAMPLES * int(c)
        self.wave = self.wave[:n]
        self.ft = cut_frame_targets(self.ft, c)
        n = sum(1 for o in self.offsets if o < c)
        self.pieces, self.offsets, self.cut = self.pieces[:n], self.offsets[:n], int(c)


    def trim_end(self, punct_ids: Sequence[int], rng: np.random.Generator) -> bool:
        """The CTC end trim: when the row ends in sentence marks (punct_ids) after a run of blank frames following its
        last word, the frames of that gap are dropped and the marks' frames moved up behind the word, the audio ending
        at a drawn sample of the new last frame (end_samples). The Parakeet teacher puts a mark after the trailing
        silence, so the audio cannot just be cut short - the mark would go with it. False (nothing done) for a row
        without a final mark or without such a gap."""
        tail = mark_tail(self.ft, punct_ids)
        if tail is None:
            return False
        e_w, s_m, e_m = tail
        ft = join_frame_targets([cut_frame_targets(self.ft, e_w + 1), slice_frame_targets(self.ft, s_m, e_m + 1)])
        n = min(end_samples(ft.n_frames, rng), len(self.wave))
        if ctc_frames(n) != ft.n_frames:  # never for a shortened row; the frame contract's guard
            n = FRAME_SAMPLES * ft.n_frames
        self.wave, self.ft, self.trimmed = self.wave[:n], ft, True
        return True


def slice_frame_targets(ft: "FrameTargets", a: int, b: int) -> "FrameTargets":
    """Frames [a, b) of ft as targets of their own: blank_lp[a:b], the dense frames in it shifted by -a with their top-k,
    the greedy path recomputed."""
    from kitsune.ctc_targets import FrameTargets

    a, b = int(a), int(b)
    lo, hi = (int(np.searchsorted(ft.dense_frame, x)) for x in (a, b))
    out = FrameTargets(b - a, ft.blank_lp[a:b].copy(), (ft.dense_frame[lo:hi] - a).astype(np.int32),
                       ft.topk_idx[lo:hi].copy(), ft.topk_lp[lo:hi].copy(), np.zeros(0, np.int32))
    out.ctc_ids = greedy_ids(out.col0())
    return out


def mark_tail(ft: "FrameTargets", punct_ids: Sequence[int]) -> tuple[int, int, int] | None:
    """(the last frame of the row's last word's run, the first frame of the marks after it, the last frame of the last
    mark's run) when the row's argmax ends in punct_ids tokens after a gap of at least one blank frame behind its last
    word; None otherwise (no word, no final mark, the mark right behind the word)."""
    from kitsune.aed_cuts import token_runs

    starts, ends, cls = token_runs(ft.col0())
    punct = np.isin(cls, np.asarray(list(punct_ids), dtype=np.int64))
    words = np.flatnonzero(~punct)
    if not len(words) or words[-1] == len(cls) - 1:
        return None
    w = int(words[-1])
    e_w, s_m, e_m = int(ends[w]), int(starts[w + 1]), int(ends[-1])
    return (e_w, s_m, e_m) if s_m > e_w + 1 else None


def acoustic_chain(rows, originals, noise, rirs, a: Augment, rng: np.random.Generator) -> dict:
    """The acoustic steps of either family's rows, in this order (the section comment's "acoustic steps"): background
    speech, room echo, background music / noise / songs, volume, codec - each from the micro-batch's own stream, its
    targets unchanged. Returns the aug counts {speech_mixed, reverbed, noised, gained, clipped, coded}; with
    a.background_min_row_s > 0 also short_rows (rows under it at the chain's entry: no step changes a row's length,
    so that is the length the student hears), speech_spared and noise_spared (rows the speech / background gate picked
    and the limit spared) - on every micro-batch, 0 included, since the train step sums every micro-batch's counts under
    the first one's keys (scripts/04_distill.py train_step). At 0 the keys, the draws and the rows are exactly the
    chain's before the key existed."""
    lim = int(round(float(a.background_min_row_s) * TARGET_SR))
    short = [len(r.wave) < lim for r in rows] if lim > 0 else None
    sp_s, sp_n = [], []
    out = dict(speech_mixed=mix_speech(rows, originals, noise, a, rng, short=short, spared=sp_s),
               reverbed=reverberate(rows, rirs, a, rng),
               noised=mix_background(rows, noise, a, rng, short=short, spared=sp_n))
    out.update(vary_gain(rows, a, rng))
    out["coded"] = apply_codecs(rows, a, rng)
    if short is not None:
        out.update(short_rows=int(sum(short)), speech_spared=len(sp_s), noise_spared=len(sp_n))
    return out


def _stretch(src: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n samples of src from a drawn start (src repeated when it is shorter than n)."""
    src = np.asarray(src, dtype=np.float32)
    if len(src) >= n:
        s = int(rng.integers(len(src) - n + 1))
        return src[s:s + n]
    return np.resize(np.roll(src, -int(rng.integers(len(src)))), n).astype(np.float32, copy=False)


def mix_speech(rows, originals, bank, a: Augment, rng: np.random.Generator, short=None, spared=None) -> int:
    """The background speech step (speech_p): per row, speech_talkers voices drawn (low, high), each another row of the
    micro-batch (its audio before any cut: originals) with probability speech_batch_p - Japanese - or else a stretch
    of the bank's speech (MUSAN: 17 languages), each at unit voiced power and summed (two or more: babble), then added
    at speech_snr_db dB of voiced power below the row - so the row stays the dominant voice, and its targets, the
    teacher's on the clean row, say that the voices behind it are not to be written. Returns the rows it changed.
    short (acoustic_chain; None: no row): per row, True spares it after its gate draw (the row's index joins spared,
    a list, when given), so the other rows' draws do not depend on whether it was short."""
    if a.speech_p <= 0:
        return 0
    from kitsune.noise_bank import background_segment, voiced_power

    bank_ok = bank is not None and bank.has(SPEECH_KINDS)
    n = 0
    for p, r in enumerate(rows):
        if rng.random() >= a.speech_p:
            continue
        if short is not None and short[p]:  # background_min_row_s: picked, spared
            if spared is not None:
                spared.append(p)
            continue
        m = len(r.wave)
        talkers = []
        for _ in range(int(rng.integers(a.speech_talkers[0], a.speech_talkers[1] + 1))):
            if len(originals) > 1 and (not bank_ok or rng.random() < a.speech_batch_p):
                q = int(rng.integers(len(originals) - 1))
                t = _stretch(originals[q + (q >= p)], m, rng)
            elif bank_ok:
                t = background_segment(bank, m, rng, SPEECH_KINDS)
            else:
                t = None
            pw = voiced_power(t) if t is not None else 0.0
            if pw > 1e-10:
                talkers.append(t * np.float32(pw ** -0.5))
        ps = voiced_power(r.wave)
        if not talkers or ps < 1e-8:
            continue
        voices = np.sum(talkers, axis=0, dtype=np.float32)
        pv = voiced_power(voices)
        if pv < 1e-10:
            continue
        snr = float(rng.uniform(*a.speech_snr_db))
        r.wave = np.asarray(r.wave, dtype=np.float32) + np.float32((ps / (pv * 10.0 ** (snr / 10.0))) ** 0.5) * voices
        n += 1
    return n


def reverberate(rows, rirs, a: Augment, rng: np.random.Generator) -> int:
    """The room echo step (reverb_p): per row, a room impulse response of the RIR bank drawn uniformly (every room once,
    not by length) and convolved with the whole row - its voices, background speech included, in one room; the row's
    length and voiced level kept (kitsune.acoustics.reverb). Returns the rows it changed."""
    if a.reverb_p <= 0 or rirs is None:
        return 0
    from kitsune.acoustics import reverb

    n = 0
    for r in rows:
        if rng.random() >= a.reverb_p:
            continue
        y = reverb(r.wave, rirs.draw_clip(rng))
        if y is not None:
            r.wave = y
            n += 1
    return n


def mix_background(rows, bank, a: Augment, rng: np.random.Generator, short=None, spared=None) -> int:
    """The background step of either family's rows (noise_p): per row, a stretch of the bank's music, noise and songs
    (BACKGROUND_KINDS; never its speech, the speech step's) as long as the row mixed under all of it at an SNR drawn
    from noise_snr_db (kitsune.noise_bank.add_background) - dry, as a stream mixes its game sound and music in -, its
    targets unchanged: the teacher's on the clean audio, so a song's lyrics are not to be written. Returns the rows it
    changed (a silent row or stretch is left as it is). short / spared: as mix_speech's."""
    if a.noise_p <= 0 or bank is None or not bank.has(BACKGROUND_KINDS):
        return 0
    from kitsune.noise_bank import add_background, background_segment

    n = 0
    for p, r in enumerate(rows):
        if rng.random() >= a.noise_p:
            continue
        if short is not None and short[p]:  # background_min_row_s: picked, spared
            if spared is not None:
                spared.append(p)
            continue
        seg = background_segment(bank, len(r.wave), rng, BACKGROUND_KINDS)  # a near-silent stretch is drawn again
        got = None if seg is None else add_background(r.wave, seg, rng, a.noise_snr_db)
        if got is not None:
            r.wave = got[0]
            n += 1
    return n


def vary_gain(rows, a: Augment, rng: np.random.Generator) -> dict:
    """The volume step (gain_p): per row, gain_db dB drawn uniformly, clipped at full scale (a quiet or an overdriven
    recording). Returns {gained, clipped}."""
    if a.gain_p <= 0:
        return dict(gained=0, clipped=0)
    from kitsune.acoustics import gain

    g = c = 0
    for r in rows:
        if rng.random() >= a.gain_p:
            continue
        r.wave, clipped = gain(r.wave, float(rng.uniform(*a.gain_db)))
        g += 1
        c += int(clipped)
    return dict(gained=g, clipped=c)


def apply_codecs(rows, a: Augment, rng: np.random.Generator) -> int:
    """The codec step (codec_p): per row, one of codecs drawn uniformly, the row encoded and decoded in memory
    (kitsune.acoustics.codec: libsndfile's MP3, GSM 6.10, mu-law, ...), as long and as aligned as before. Returns the
    rows it changed."""
    if a.codec_p <= 0 or not a.codecs:
        return 0
    from kitsune.acoustics import codec

    n = 0
    for r in rows:
        if rng.random() >= a.codec_p:
            continue
        r.wave = codec(r.wave, a.codecs[int(rng.integers(len(a.codecs)))], rng)
        n += 1
    return n


def _join_rows(parts: list[_AugRow], rng: np.random.Generator) -> _AugRow:
    """concat's join of single-utterance rows, in the given order."""
    t = [r.ft.n_frames for r in parts]
    return _AugRow(join_waves([r.wave for r in parts], t, rng), join_frame_targets([r.ft for r in parts]),
                   [r.pieces[0] for r in parts], np.cumsum([0] + t[:-1]).astype(int).tolist())


# AED rows (AudioBatchDataset(augment=Augment(max_tokens=...), cuts=...); scripts/04_distill.py augment.* on an AED
# student). The same three steps, on the Cohere teacher's token targets:
#   concat    as above (concat_k: the same divisor rule and concat_max_s), the pieces' audio joined by join_waves at
#             their ctc_frames, so each piece starts on an encoder frame boundary of the joined row; the targets are
#             the pieces' tokens one after another, each piece's EOS dropped but the last's - with their stored top-k
#             rows -, so a joined row reads "sentence. sentence. sentence.<EOS>" and the teacher's marks sit inside it.
#             The first token of a later piece keeps the top-k the teacher gave it at its own utterance's start, an
#             approximation of the teacher on the joined context. A join is refused for the whole micro-batch (no row
#             joined; aug concat_capped) when a joined row would hold more than max_tokens target tokens: the planner
#             excludes rows over max_dec_len and never planned a longer decoder. Kept within it, a joined micro-batch's
#             decoder rectangle - rows x longest - is at most the planned one's: rows / k rows of at most the k pieces'
#             tokens, sum(T) - (k - 1) <= k x max(T)
#   truncate  per row (truncate_p): a frame drawn as truncate_cut draws it - truncate_pause_p of the cuts inside a
#             pause, uniformly among the frames otherwise -, among the frames the cut table names for the row's pieces
#             (kitsune.aed_cuts: where the Parakeet teacher's CTC alignment has a word start whose boundary maps onto
#             the Cohere text), shifted to the piece's offset, at or after the same lower bound (truncate_min_frac of
#             the row's frames, truncate_min_s) and never on a row under truncate_min_row_s. The audio ends at a drawn
#             sample (end_samples), as a CTC row's; the targets keep the Cohere tokens that entry names (the pieces'
#             before it whole, EOS dropped) and end in EOS, without a sentence mark: the input ending inside a sentence
#             is no reason to emit one. The EOS position's target is one-hot (top-k row: EOS at log-prob 0, the
#             teacher's next ids at TOKEN_CUT_FLOOR_LP), so its KL is its CE: the teacher never saw the cut input. A
#             piece whose decoded audio gives other frames than the table recorded contributes no cuts (cut_mismatch)
#   end trim  per row that was not cut (end_trim_p): its trailing silence - the audio after its last voiced 10 ms frame
#             (voiced_end: within END_TRIM_DB of the row's loudest) - trimmed to a tail drawn uniformly in
#             END_TRIM_TAIL_S, its tokens kept, sentence mark and EOS included (a row whose silence is already shorter
#             stays as it is). Why (the P-0.3B review's issue B, 2026-10-05): a cut ends tightly before a word with no
#             mark, and the clean source, Galgame, almost always ends a complete row in a long pause (median 0.28 s;
#             ReazonSpeech and Emilia end tightly but are noisy or spontaneous), so "clean audio that ends tightly" was
#             "a cut": P-0.3B left the mark off 7 % of JSUT's complete sentences, whose audio ends within ~0.05 s of the
#             last word, and kept it on 100 % of Galgame's. Trimmed complete rows make a tight end mean nothing by
#             itself. Before mix, so the interferer does not hide the row's silence
#   mix       as above (the targets the clean row's)
# No pads: a quiet pad would add no token, and the recipe keeps them off for both families.
END_TRIM_DB = 35.0  # a 10 ms frame is voiced within this many dB of the row's loudest (the review's tail_sil.py rule)
END_TRIM_TAIL_S = (0.01, 0.08)  # the trimmed row keeps this much audio after its last voiced frame (JSUT: 0.00-0.05 s)
TOKEN_CUT_FLOOR_LP = -30.0  # the cut row's EOS position: the log-prob of its other top-k ids (e^-30 ~ 1e-13 each)


def voiced_end(wave: np.ndarray) -> int:
    """The sample just after a row's last voiced HOP-sample (10 ms) frame: voiced = mean power within END_TRIM_DB of
    the row's loudest frame. len(wave) for a row of under 5 frames or no frame at all; everything after it is the row's
    trailing silence (end trim)."""
    n = len(wave) // HOP
    if n < 5:
        return len(wave)
    fr = np.asarray(wave[:n * HOP], dtype=np.float32).reshape(n, HOP)
    db = 10.0 * np.log10(np.einsum("ij,ij->i", fr, fr) / HOP + 1e-12)
    voiced = np.flatnonzero(db > db.max() - END_TRIM_DB)
    return int((voiced[-1] + 1) * HOP) if len(voiced) else len(wave)


class _TokRow:
    """One row of an augmented AED micro-batch: its audio, tokens (EOS last) with their top-k rows, the store indices of
    its pieces, their first frames, their frames and the tokens of the pieces before each (EOS dropped), and the frame
    it was cut at (None: whole)."""

    __slots__ = ("wave", "tok", "ti", "tl", "pieces", "offsets", "frames", "before", "cut", "trimmed")

    def __init__(self, wave: np.ndarray, tok: np.ndarray, ti: np.ndarray, tl: np.ndarray, i: int, n_frames: int):
        self.wave, self.tok, self.ti, self.tl = wave, tok, ti, tl
        self.pieces, self.offsets, self.frames, self.before, self.cut = [int(i)], [0], [int(n_frames)], [0], None
        self.trimmed = False

    @property
    def n_frames(self) -> int:
        return int(sum(self.frames))

    def body(self, eos: int) -> int:
        """The tokens before the final EOS (all of them for a row the teacher's length cap cut, which has none)."""
        return len(self.tok) - int(len(self.tok) > 0 and int(self.tok[-1]) == int(eos))

    def cut_at(self, c: int, kept: int, eos: int, rng: np.random.Generator):
        """Keep the first c frames - the audio up to end_samples(c, rng), still c frames - and the first `kept` tokens,
        then EOS with a one-hot top-k row (the teacher's ids at the first removed token, EOS first, the rest at
        TOKEN_CUT_FLOOR_LP); only the pieces that start before c stay named."""
        n = min(end_samples(c, rng), len(self.wave))
        if ctc_frames(n) != int(c):  # never for a row of more than c frames; the frame contract's guard
            n = FRAME_SAMPLES * int(c)
        self.wave = self.wave[:n]
        k = self.ti.shape[1]
        nxt = [int(x) for x in self.ti[kept] if int(x) != int(eos)][:k - 1]
        e_idx = np.asarray([eos] + nxt, dtype=self.ti.dtype)
        e_lp = np.asarray([0.0] + [TOKEN_CUT_FLOOR_LP] * (k - 1), dtype=self.tl.dtype)
        self.tok = np.concatenate([self.tok[:kept], np.asarray([eos], dtype=self.tok.dtype)])
        self.ti = np.concatenate([self.ti[:kept], e_idx[None]])
        self.tl = np.concatenate([self.tl[:kept], e_lp[None]])
        m = sum(1 for o in self.offsets if o < c)
        self.pieces, self.offsets, self.frames, self.before = (self.pieces[:m], self.offsets[:m], self.frames[:m],
                                                               self.before[:m])
        self.cut = int(c)


def _join_tok_rows(parts: list[_TokRow], eos: int, rng: np.random.Generator) -> _TokRow:
    """concat's join of single-utterance AED rows, in the given order: the audio join_waves', the targets the pieces'
    without the EOS of every piece but the last."""
    body = [r.body(eos) for r in parts]
    t = [r.n_frames for r in parts]
    keep = [slice(0, n) for n in body[:-1]] + [slice(None)]
    row = _TokRow(join_waves([r.wave for r in parts], t, rng),
                  np.concatenate([r.tok[s] for r, s in zip(parts, keep)]),
                  np.concatenate([r.ti[s] for r, s in zip(parts, keep)]),
                  np.concatenate([r.tl[s] for r, s in zip(parts, keep)]), parts[0].pieces[0], t[0])
    row.pieces, row.frames = [r.pieces[0] for r in parts], t
    row.offsets = np.cumsum([0] + t[:-1]).astype(int).tolist()
    row.before = np.cumsum([0] + body[:-1]).astype(int).tolist()
    return row


def aed_cut_candidates(r: _TokRow, cuts, a: Augment, eos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """truncate's candidate frames of an AED row (kitsune.aed_cuts.row_candidates over its pieces' cut-table entries, at
    each piece's offset): (frames, the tokens each keeps, in a pause, the pieces the table holds whose frames differ
    from their decoded audio's - they add none). None on a row under truncate_min_row_s; never a frame below
    max(truncate_min_frac x its frames, truncate_min_s, 1 frame), nor one that keeps no token or every token."""
    from kitsune import aed_cuts

    z = np.zeros(0, np.int64)
    T = r.n_frames
    if T * FRAME_SAMPLES < float(a.truncate_min_row_s) * TARGET_SR - 1e-6:
        return z, z, np.zeros(0, dtype=bool), 0
    lo = max(math.ceil(float(a.truncate_min_frac) * T - 1e-9),
             math.ceil(float(a.truncate_min_s) * TARGET_SR / FRAME_SAMPLES - 1e-9), 1)
    pieces, bad = [], 0
    for i, off, nf, before in zip(r.pieces, r.offsets, r.frames, r.before):
        tf = cuts.n_frames(i)
        if tf != nf:
            bad += tf >= 0  # a row the table holds whose audio decodes to other frames
            continue
        pieces.append((cuts.entries(i), off, before))
    frames, kept, pause = aed_cuts.row_candidates(pieces, lo, TRUNCATE_PAUSE_FRAMES)
    ok = (kept >= 1) & (kept < r.body(eos))
    return frames[ok], kept[ok], pause[ok], bad


# --------------------------------------------------------------------------------------------------------- loader


def default_num_workers() -> int:
    """perf.num_workers = "auto". Most of the train audio is not FLAC: measured per core on the laptop, Emilia's 24 kHz
    MP3 (with a soxr_hq resample) decodes at ~1000-1700x realtime and galgame's OGG at ~1400-2200x (FLAC ~4000x),
    about half that on a slow core. 8 workers then give ~8-15k audio-s/s, well above one A100 (~2.5k at 30-40 % MFU);
    a loader cut to 1-2 workers (scripts/04_distill.py shm_cap) is not. Half of this trainer's share of the CPUs
    (kitsune.ctc_preflight.per_gpu_cpus: the affinity and the cgroup quota, divided by KITSUNE_N_GPUS), not of the
    host's os.cpu_count(), which on a vast container is the whole machine's (fix 2)."""
    if os.name == "nt":
        return 2
    from kitsune.ctc_preflight import per_gpu_cpus

    return max(1, min(8, per_gpu_cpus() // 2))


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
                mp_context: str = "spawn", timeout_s: float = 0.0) -> Iterator[tuple[object, list[dict]]]:
    """Decode micro-batches in `num_workers` processes (`prefetch` micro-batches in flight per worker) and yield
    whole optimizer steps:
      plan = one epoch's list of steps    -> (step_idx, [micro-batch, ...]) for plan[start_step:]
      plan = a StepPlanner                -> ((epoch, step_idx), [micro-batch, ...]) from the planner's position,
                                             across epochs without end; the workers persist, so there is no
                                             worker-restart stall at epoch boundaries
    Workers use `spawn` on every OS, so Linux runs the same code path as the Windows smoke run. KITSUNE_SHARING
    (e.g. file_system), if set, becomes torch's sharing strategy in the main process and in the workers.
    timeout_s > 0 (workers only; 0 = wait forever): a worker micro-batch that never arrives raises RuntimeError
    "DataLoader timed out" after that many seconds in total instead of blocking the trainer for good. A full /dev/shm
    (or, on Windows, a paging-file-backed shared mapping that fails under the commit limit) does not kill the worker:
    its queue feeder prints the error and drops the micro-batch, and the loader, which yields in order, would wait for
    it forever. It must cover the workers' start-up (the first wait). The wait runs in 5 s slices, so torch's 5 s
    dead-worker check (Windows' only one: no SIGCHLD handler) still reports a crashed worker within seconds.
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
    # torch looks for a dead worker only when a wait ends, and on Windows (no SIGCHLD handler) that is its only check:
    # one timeout_s-long wait would hide a crashed worker for timeout_s. Wait in 5 s slices (torch's
    # MP_STATUS_CHECK_INTERVAL) and give up after timeout_s in total.
    slice_s = min(5.0, float(timeout_s)) if num_workers > 0 and timeout_s > 0 else 0
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch if num_workers > 0 else None, worker_init_fn=_worker_init if num_workers > 0 else None,
        multiprocessing_context=mp_context if num_workers > 0 else None, persistent_workers=False,
        timeout=slice_s,  # in-process loading asserts timeout == 0
    )

    def fetch(it):
        t0 = time.monotonic()
        while True:
            try:
                return next(it)
            except RuntimeError as e:  # a timed-out wait leaves the iterator's state untouched: next() waits on
                if not (slice_s and str(e).startswith("DataLoader timed out")):
                    raise
                if time.monotonic() - t0 >= timeout_s:
                    raise RuntimeError(f"DataLoader timed out after {timeout_s:g} s (no micro-batch arrived)") from None

    def gen():
        it = iter(loader)
        try:
            while True:
                try:
                    first = fetch(it)
                except StopIteration:
                    return
                key, n = sampler.sizes.popleft()
                yield key, [first] + [fetch(it) for _ in range(n - 1)]
        finally:
            shutdown = getattr(it, "_shutdown_workers", None)
            if shutdown is not None:
                try:
                    shutdown()
                except RuntimeError as e:
                    # a worker that dies while the pool is being shut down (seen on an A100 box: SIGABRT in a worker's
                    # exit after the trainer's STOP, reported by torch's SIGCHLD handler inside the join) cannot have
                    # lost anything: every micro-batch the consumer took was whole, and torch's own finally still
                    # terminates the other workers and drops their pids. Raising here would mark a finished run (or
                    # an eval) failed; a death while batches are still wanted raises from fetch() above instead
                    if "DataLoader worker" not in str(e):
                        raise
                    warnings.warn(f"make_loader: a worker died while the loader shut down ({e.args[0].strip()}); "
                                  "nothing it yielded was lost", RuntimeWarning, stacklevel=2)

    return gen()
