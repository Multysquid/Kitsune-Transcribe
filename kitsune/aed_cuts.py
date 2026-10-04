"""Where an AED (Cohere-token) train row may be cut inside a sentence: cut points from the Parakeet CTC alignment.

Why. Recipe v2's truncate (kitsune.trainset's "augmentation" section) cuts a row inside a sentence so the student
learns that the input ending is no reason to emit a sentence mark. A CTC student's targets are per frame, so the cut
frame says which targets it keeps. An AED student's targets are the Cohere teacher's token sequence, with no timing:
nothing says which tokens the first c frames of audio hold. The Parakeet teacher labelled the same rows with per-frame
CTC targets (parakeet_out), and their argmax places each Parakeet token on the frames. So a cut is found as the P recipe
finds it - at a frame c whose first removed Parakeet token is a word (trainset.truncate_frames' rule) - and carried over
to Cohere's text by aligning the two teachers' texts character by character (punctuation and spaces dropped, since the
two put marks differently): the boundary must lie inside a run of matching characters, with at least CTX of them on
each side, and a Cohere token must start exactly there. The kept Cohere tokens are those before it. Rows where the two
teachers disagree near a boundary simply have no cut there.

Measured on 8,833 train rows of five shards (one per source, 2026-10-05): 96-100 % of the rows truncate may cut (>= 3 s,
a word after the lower bound) have at least one mapped cut, and 42-62 % of all rows have one inside a pause.

The table (tools/aed_cut_table.py writes it from the selection's kept train rows, the two label roots and the two
tokenizers; one parquet row per train row):
  id        string
  n_frames  int16        the row's Parakeet encoder frames (trainset.ctc_frames of its audio); the loader drops the cuts
                         of a row whose decoded audio gives other frames
  lo, b, hi, m  list<int16>  one entry per mapped word token u >= 1 of the row's CTC argmax (token u-1 starts at frame
                         lo - 1 and its run ends at frame b - 1; token u starts at frame hi): a cut at any frame c in
                         [lo, hi] keeps the Parakeet tokens before u and the Cohere tokens [0, m) (m >= 1); the frames
                         [b, hi - 1] are the blank gap before u, a pause (trainset.pause_frames) when hi - b >=
                         trainset.TRUNCATE_PAUSE_FRAMES. Entries are in frame order and do not overlap.
The frames [lo, hi] of all entries are exactly the frames truncate_frames allows on the row's CTC targets whose kept
Parakeet tokens map, before its lower bound (min_frac, min_s, min_row_s: the train-time Augment's, applied by
row_candidates).

The trainer joins the table to its train dataset by id once (build_cut_index: flat arrays in dataset order under the
store's cache dir, memory-mapped by the loader's workers); kitsune.trainset's AED augmentation reads a row's entries
with CutIndex.entries. Only numpy and pyarrow at import: the loader's workers import this module.
"""
import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np

CTX = 2  # matching characters a mapped boundary needs on each side
PK_BLANK = 3072  # == kitsune.parakeet_targets.BLANK (the CTC blank class)
TABLE_COLUMNS = ("id", "n_frames", "lo", "b", "hi", "m")
ENTRY_ARRAYS = ("lo", "b", "hi", "m")
INDEX_VERSION = 1


# ------------------------------------------------------------------------------------------------- the table rows


def token_runs(col0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, ends, classes) of the CTC argmax col0's tokens: a token starts at a frame that is not blank and differs
    from the frame before it (where CTC reads a new token, as trainset.next_token_start), and its run ends at the last
    frame of that class before a change. Each start pairs with the next end."""
    col0 = np.asarray(col0, dtype=np.int64)
    if not len(col0):
        z = np.zeros(0, np.int64)
        return z, z, z
    prev = np.concatenate([[-1], col0[:-1]])
    nxt = np.concatenate([col0[1:], [-1]])
    starts = np.flatnonzero((col0 != PK_BLANK) & (col0 != prev))
    ends = np.flatnonzero((col0 != PK_BLANK) & (col0 != nxt))
    return starts, ends, col0[starts]


def piece_text(piece: str) -> str:
    """A SentencePiece piece's text: the word-boundary marker dropped, a byte-fallback piece (<0x..>) one placeholder
    character that matches nothing, a special token (<unk>, <|...|>) nothing."""
    if piece.startswith("<0x") and piece.endswith(">"):
        return "�"
    if piece.startswith("<") and piece.endswith(">"):
        return ""
    return piece.replace("▁", "")


def text_spans(ids, texts: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """The text of a token sequence (texts: the piece text of every id) and each token's (start, end) characters."""
    out, spans, pos = [], [], 0
    for i in ids:
        i = int(i)
        t = texts[i] if 0 <= i < len(texts) else ""
        out.append(t)
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return "".join(out), spans


def kept_chars(text: str) -> np.ndarray:
    """Indices of the characters the alignment compares: not punctuation (Unicode P*), separators (Z*) or whitespace,
    so 。、「」?! and spaces - which the two teachers put differently - never decide a match."""
    return np.asarray([k for k, ch in enumerate(text)
                       if not (unicodedata.category(ch)[0] in "PZ" or ch.isspace())], dtype=np.int64)


def row_entries(col0, pk_texts: Sequence[str], pk_punct: Sequence[int], co_tokens, co_texts: Sequence[str],
                ctx: int = CTX) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The table entries (lo, b, hi, m: int16 arrays) of one row: col0 its Parakeet CTC argmax per frame, co_tokens its
    Cohere tokens without the final EOS. For every Parakeet token u >= 1 that is not one of pk_punct (the Parakeet
    vocabulary's sentence marks and commas: a cut never removes only a mark), the characters of its text start ->
    the same boundary in the punctuation-free texts -> a run of matching characters (rapidfuzz's Indel alignment) with
    >= ctx of them on each side -> the Cohere character there -> the Cohere token that starts exactly at it (m tokens
    before it). Tokens with no such mapping get no entry."""
    from rapidfuzz.distance import Indel

    starts, ends, cls = token_runs(col0)
    z = np.zeros(0, np.int16)
    if len(starts) < 2 or not len(co_tokens):
        return z, z, z, z
    p_text, p_spans = text_spans(cls, pk_texts)
    c_text, c_spans = text_spans(co_tokens, co_texts)
    pk, ck = kept_chars(p_text), kept_chars(c_text)
    pn, cn = "".join(p_text[k] for k in pk), "".join(c_text[k] for k in ck)
    blocks = [(o.src_start, o.src_end, o.dest_start) for o in Indel.opcodes(pn, cn) if o.tag == "equal"
              and o.src_end - o.src_start >= 2 * ctx]
    if not blocks:
        return z, z, z, z
    c_tok_at = {}
    for t, (s, e) in enumerate(c_spans):
        if e > s:
            c_tok_at.setdefault(s, t)
    punct = set(int(x) for x in pk_punct)
    out = []
    for u in range(1, len(starts)):
        if int(cls[u]) in punct:
            continue
        inorm = int(np.searchsorted(pk, p_spans[u][0]))  # punctuation-free characters before token u
        for a0, a1, b0 in blocks:
            if a0 + ctx <= inorm <= a1 - ctx:
                t = c_tok_at.get(int(ck[b0 + inorm - a0]))
                if t:  # m >= 1: at least one Cohere token kept
                    out.append((starts[u - 1] + 1, ends[u - 1] + 1, starts[u], t))
                break
    if not out:
        return z, z, z, z
    a = np.asarray(out, dtype=np.int64)
    return tuple(a[:, j].astype(np.int16) for j in range(4))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- the trainer's join


def build_cut_index(table_path, ids: Sequence[str], out_root, expect_sha256: str | None = None,
                    log=print) -> "CutIndex":
    """The table joined to a train dataset by id: under out_root/<key>/ (key: the table's sha256 and the ids' order),
    cuts_offsets.npy (n+1,) int64, cuts_{lo,b,hi,m}.npy int16 and cuts_n_frames.npy (n,) int16 (-1: the row is not in
    the table, so it is never cut) in the dataset's order, and index.json (counts) last; an existing complete index is
    reused. expect_sha256: the table's pinned sha256 (augment.cuts_sha256), checked before anything is read. Returns
    the CutIndex the dataset memory-maps."""
    import pyarrow.parquet as pq

    table_path, out_root = Path(table_path), Path(out_root)
    sha = sha256_file(table_path)
    if expect_sha256 and sha != expect_sha256:
        raise ValueError(f"{table_path}: sha256 {sha}, the config pins {expect_sha256}: a different cut table")
    ids_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    out = out_root / f"{sha[:16]}-{ids_sha[:16]}"
    info_path = out / "index.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if (info.get("table_sha256"), info.get("ids_sha256"), info.get("version")) == (sha, ids_sha, INDEX_VERSION):
            log(f"aed cuts: reusing {out} ({info['rows_with_cuts']} of {info['rows']} rows have cuts)")
            return CutIndex(str(out), info)
    t = pq.read_table(table_path, columns=list(TABLE_COLUMNS))
    lists = {k: t.column(k).combine_chunks() for k in ENTRY_ARRAYS}
    toff = np.asarray(lists["lo"].offsets, dtype=np.int64)
    for k in ENTRY_ARRAYS[1:]:
        if not np.array_equal(np.asarray(lists[k].offsets, dtype=np.int64), toff):
            raise ValueError(f"{table_path}: column {k} has other list lengths than lo")
    vals = {k: np.asarray(lists[k].values, dtype=np.int16) for k in ENTRY_ARRAYS}
    import pandas as pd

    tid = pd.Index(t.column("id").to_pylist())
    if not tid.is_unique:
        raise ValueError(f"{table_path}: duplicate ids")
    row = tid.get_indexer(list(ids))
    have = row >= 0
    tn = np.asarray(t.column("n_frames").to_numpy(), dtype=np.int16)
    n_frames = np.where(have, tn[np.maximum(row, 0)], -1).astype(np.int16)
    counts = np.where(have, np.diff(toff)[np.maximum(row, 0)], 0).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    take = np.repeat(np.where(have, toff[np.maximum(row, 0)], 0) - off[:-1], counts) + np.arange(int(off[-1]))
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(cuts_offsets=off, cuts_n_frames=n_frames, **{f"cuts_{k}": vals[k][take] for k in ENTRY_ARRAYS})
    for name, arr in arrays.items():
        tmp = out / f"{name}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out / f"{name}.npy")
    info = dict(version=INDEX_VERSION, table=str(table_path), table_sha256=sha, ids_sha256=ids_sha, rows=len(ids),
                rows_in_table=int(have.sum()), rows_with_cuts=int((counts > 0).sum()), entries=int(off[-1]))
    tmp = out / "index.json.tmp"
    tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    os.replace(tmp, info_path)
    log(f"aed cuts: built {out} from {table_path.name} ({info['rows_in_table']} of {len(ids)} rows in the table, "
        f"{info['rows_with_cuts']} with cuts, {info['entries']} entries)")
    return CutIndex(str(out), info)


class CutIndex:
    """A built index (build_cut_index), picklable without its arrays: they are memory-mapped lazily in each process,
    as the stores' are."""

    def __init__(self, path: str, info: dict):
        self.path, self.info, self._mm = str(path), dict(info), None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def _open(self) -> dict:
        if self._mm is None:
            d = Path(self.path)
            self._mm = {name: np.load(d / f"{name}.npy", mmap_mode="r")
                        for name in ("cuts_offsets", "cuts_n_frames", *(f"cuts_{k}" for k in ENTRY_ARRAYS))}
        return self._mm

    def n_frames(self, i: int) -> int:
        """The table's Parakeet frames of dataset row i, -1 if it is not in the table."""
        return int(self._open()["cuts_n_frames"][int(i)])

    def entries(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(lo, b, hi, m) of dataset row i, int64 copies (empty for a row without cuts)."""
        mm = self._open()
        s, e = int(mm["cuts_offsets"][int(i)]), int(mm["cuts_offsets"][int(i) + 1])
        return tuple(np.asarray(mm[f"cuts_{k}"][s:e], dtype=np.int64) for k in ENTRY_ARRAYS)


def row_candidates(pieces: Sequence[tuple[tuple, int, int]], lo_bound: int,
                   pause_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every frame a (possibly joined) row may be cut at, in order, with the Cohere tokens each keeps and whether it
    lies in a pause: pieces = [(entries (lo, b, hi, m) of the piece, its first frame in the row, the Cohere tokens of
    the pieces before it - EOS dropped)]; frames below lo_bound are left out (truncate's lower bound on the row's
    frames). Returns (frames, kept tokens, pause) as (int64, int64, bool) arrays."""
    fr, keep, pause = [], [], []
    for (lo, b, hi, m), off, before in pieces:
        if not len(lo):
            continue
        n = hi - lo + 1
        f = np.repeat(lo - np.cumsum(np.concatenate([[0], n[:-1]])), n) + np.arange(int(n.sum()))
        e = np.repeat(np.arange(len(lo)), n)
        fr.append(f + int(off))
        keep.append(m[e] + int(before))
        pause.append(((hi - b) >= int(pause_frames))[e] & (f >= b[e]) & (f < hi[e]))
    if not fr:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, dtype=bool)
    fr, keep, pause = np.concatenate(fr), np.concatenate(keep), np.concatenate(pause)
    ok = fr >= int(lo_bound)
    return fr[ok], keep[ok], pause[ok]
