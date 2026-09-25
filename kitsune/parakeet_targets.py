"""Parakeet teacher soft targets: packing, the stored format, loading and checking (numpy only, no torch).

scripts/02p_parakeet_pass.py writes, per data shard  data/shards/<source>/<split>-NNNNN.parquet,
  parakeet_out/<source>/<split>-NNNNN.npz     packed arrays (FORMAT below; np.savez_compressed)
  parakeet_out/<source>/<split>-NNNNN.jsonl   one line per utterance, same order as the npz
  parakeet_out/meta.json                      model pins, decoding, dtypes and settings (build_meta)

FORMAT (npz; n utterances, S joint steps, N tokens, F frames, D dense CTC frames):
  format_version   ()      int64    1
  ids              (n,)    <U       utterance ids, same order as the jsonl
  duration         (n,)    float32  seconds
  n_frames         (n,)    int32    valid encoder frames T_i (80 ms: 8x subsampling of 10 ms hops, 12.5 fps)
  truncated        (n,)    bool     hit hard_cap (never expected with the guard)
  n_scanned        ()      int64    rows of the data shard (incl. skipped >30 s / undecodable)
  shard_ids_sha256 ()      <U64     kitsune.store.ids_sha256 of the data shard's ids
  k_tdt, k_ctc, max_symbols ()  int64;  ctc_dense_thr ()  float64      settings, checked on resume
                                    (max_symbols 0 = guard off)
  step_offsets     (n+1,)  int64    utterance i owns joint steps [step_offsets[i], step_offsets[i+1])
  step_frame       (S,)    int16    encoder frame t of the evaluation (0 <= t < T_i)
  step_dur         (S,)    int8     APPLIED duration after the step (0..4; blank 0 -> 1; guard -> 1)
  step_forced      (S,)    bool     the max-symbols guard fired
  tdt_topk_idx     (S,k)   int16    top-k of log_softmax over the 3073 token logits (3072 = blank); col 0 = emitted
  tdt_topk_lp      (S,k)   float16  their log-probs (fp32 joint, then fp16)
  tdt_dur_lp       (S,5)   float16  log_softmax over the 5 duration logits, durations [0,1,2,3,4]
  tok_offsets      (n+1,)  int64
  tokens           (N,)    int16    the TDT hypothesis = tdt_topk_idx[:,0] where != 3072
  frame_offsets    (n+1,)  int64    utterance i owns CTC frames [frame_offsets[i], frame_offsets[i+1]) (T_i of them)
  ctc_blank_lp     (F,)    float16  CTC log p(blank) at every valid frame
  dense_offsets    (n+1,)  int64
  ctc_dense_frame  (D,)    int16    utterance-local frame index of the frames with p(blank) < ctc_dense_thr
  ctc_topk_idx     (D,k)   int16    top-k CTC classes over 3073 (blank may appear)
  ctc_topk_lp      (D,k)   float16

Tail mass outside a top-k = 1 - sum(exp(lp)), clamp at 0 (as for Cohere). No lse is stored: log_softmax(z/T) depends
only on the stored relative log-probs. The prefix length u_s (non-blank tokens before step s) is derived by the loader.

jsonl (same order): {id, hyp, ctc_hyp, ref, cer, ctc_cer, duration, n_tok, n_steps, n_frames, n_forced, truncated}

Input of `pack_shard`: one dict per kept utterance, in data-shard row order, whose keys are the per-utterance npz
names: id, duration, n_frames, truncated, step_frame, step_dur, step_forced, tdt_topk_idx, tdt_topk_lp, tdt_dur_lp,
ctc_blank_lp, ctc_dense_frame, ctc_topk_idx, ctc_topk_lp (the per-row result of ParakeetTeacher.run_batch plus id and
duration). Tokens are derived from column 0 of the TDT top-k, so they cannot disagree with it.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
BLANK = 3072
VOCAB = 3073
N_DUR = 5
SETTING_KEYS = ("k_tdt", "k_ctc", "max_symbols", "ctc_dense_thr")
LP_TOL = 1e-3  # fp16 rounding of a log-prob near log(0.95) is ~3e-5; the dense threshold check allows this much


def _setting(settings: dict, key: str):
    if key == "ctc_dense_thr":
        return float(settings[key])
    return int(settings[key] or 0)  # max_symbols None = guard off, stored as 0


def normalise_settings(settings: dict) -> dict:
    return {key: _setting(settings, key) for key in SETTING_KEYS}


def utt_tokens(utt: dict) -> np.ndarray:
    col0 = np.asarray(utt["tdt_topk_idx"])[:, 0] if len(utt["tdt_topk_idx"]) else np.zeros(0, np.int64)
    return col0[col0 != BLANK].astype(np.int16)


def ctc_col0(n_frames: int, dense_frame, topk_idx) -> np.ndarray:
    """The CTC argmax per frame: blank off the dense frames, the stored top-1 on them."""
    col0 = np.full(int(n_frames), BLANK, dtype=np.int64)
    if len(dense_frame):
        col0[np.asarray(dense_frame, dtype=np.int64)] = np.asarray(topk_idx)[:, 0]
    return col0


def ctc_greedy(idx_col0_per_frame) -> list[int]:
    """CTC greedy decode: collapse repeats, then drop blanks."""
    out, prev = [], None
    for c in (int(x) for x in idx_col0_per_frame):
        if c != prev and c != BLANK:
            out.append(c)
        prev = c
    return out


def _cat(parts: list, shape_tail: tuple, dtype) -> np.ndarray:
    if not parts:
        return np.zeros((0, *shape_tail), dtype=dtype)
    return np.concatenate([np.asarray(p).reshape(-1, *shape_tail) for p in parts]).astype(dtype)


def _offsets(lengths: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=np.int64))]).astype(np.int64)


def pack_shard(utts: list[dict], *, settings: dict, n_scanned: int, shard_ids_sha: str) -> dict[str, np.ndarray]:
    s = normalise_settings(settings)
    kt, kc = s["k_tdt"], s["k_ctc"]
    toks = [utt_tokens(u) for u in utts]
    return dict(
        format_version=np.int64(FORMAT_VERSION),
        ids=np.array([u["id"] for u in utts], dtype=str) if utts else np.zeros(0, dtype="<U1"),
        duration=np.array([u["duration"] for u in utts], dtype=np.float32),
        n_frames=np.array([u["n_frames"] for u in utts], dtype=np.int32),
        truncated=np.array([bool(u["truncated"]) for u in utts], dtype=bool),
        n_scanned=np.int64(n_scanned),
        shard_ids_sha256=np.array(shard_ids_sha),
        k_tdt=np.int64(kt),
        k_ctc=np.int64(kc),
        max_symbols=np.int64(s["max_symbols"]),
        ctc_dense_thr=np.float64(s["ctc_dense_thr"]),
        step_offsets=_offsets([len(u["step_frame"]) for u in utts]),
        step_frame=_cat([u["step_frame"] for u in utts], (), np.int16),
        step_dur=_cat([u["step_dur"] for u in utts], (), np.int8),
        step_forced=_cat([u["step_forced"] for u in utts], (), bool),
        tdt_topk_idx=_cat([u["tdt_topk_idx"] for u in utts], (kt,), np.int16),
        tdt_topk_lp=_cat([u["tdt_topk_lp"] for u in utts], (kt,), np.float16),
        tdt_dur_lp=_cat([u["tdt_dur_lp"] for u in utts], (N_DUR,), np.float16),
        tok_offsets=_offsets([len(t) for t in toks]),
        tokens=_cat(toks, (), np.int16),
        frame_offsets=_offsets([len(u["ctc_blank_lp"]) for u in utts]),
        ctc_blank_lp=_cat([u["ctc_blank_lp"] for u in utts], (), np.float16),
        dense_offsets=_offsets([len(u["ctc_dense_frame"]) for u in utts]),
        ctc_dense_frame=_cat([u["ctc_dense_frame"] for u in utts], (), np.int16),
        ctc_topk_idx=_cat([u["ctc_topk_idx"] for u in utts], (kc,), np.int16),
        ctc_topk_lp=_cat([u["ctc_topk_lp"] for u in utts], (kc,), np.float16),
    )


def write_shard(out_dir, stem, arrays, jsonl_rows):
    """jsonl first, then the npz (its presence marks the shard done); each tmp + fsync + rename."""
    from kitsune.labelpass import write_pair

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pair(out_dir / f"{stem}.jsonl", jsonl_rows, out_dir / f"{stem}.npz", arrays, compressed=True)


@dataclass
class ParakeetShard:
    z: dict

    @property
    def n(self) -> int:
        return len(self.z["ids"])

    @property
    def ids(self) -> list[str]:
        return [str(x) for x in self.z["ids"]]

    @property
    def settings(self) -> dict:
        return {key: _setting({key: self.z[key].item()}, key) for key in SETTING_KEYS}

    def _span(self, name: str, i: int) -> slice:
        o = self.z[name]
        return slice(int(o[i]), int(o[i + 1]))

    def utt(self, i: int) -> dict:
        st, tk = self._span("step_offsets", i), self._span("tok_offsets", i)
        idx = self.z["tdt_topk_idx"][st]
        emitted = (idx[:, 0] != BLANK).astype(np.int64) if len(idx) else np.zeros(0, np.int64)
        return dict(frames=self.z["step_frame"][st].astype(np.int64), dur=self.z["step_dur"][st].astype(np.int64),
                    forced=self.z["step_forced"][st], tdt_idx=idx.astype(np.int64),
                    tdt_lp=self.z["tdt_topk_lp"][st].astype(np.float32),
                    dur_lp=self.z["tdt_dur_lp"][st].astype(np.float32),
                    u=np.cumsum(emitted) - emitted, tokens=self.z["tokens"][tk].astype(np.int64))

    def ctc(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame top-k over all T_i frames; a blank-only frame is [3072, -1, ...] / [blank_lp, -inf, ...]."""
        fr, de = self._span("frame_offsets", i), self._span("dense_offsets", i)
        blank_lp = self.z["ctc_blank_lp"][fr].astype(np.float32)
        k = int(self.z["k_ctc"])
        T = len(blank_lp)
        idx = np.full((T, k), -1, dtype=np.int64)
        lp = np.full((T, k), -np.inf, dtype=np.float32)
        idx[:, 0], lp[:, 0] = BLANK, blank_lp
        dense = self.z["ctc_dense_frame"][de].astype(np.int64)
        idx[dense] = self.z["ctc_topk_idx"][de].astype(np.int64)
        lp[dense] = self.z["ctc_topk_lp"][de].astype(np.float32)
        return idx, lp


def load_shard(path) -> ParakeetShard:
    with np.load(path) as z:
        return ParakeetShard({key: z[key] for key in z.files})


def _ordered_subsequence(sub: list[str], full: list[str]) -> bool:
    it = iter(full)
    return all(any(x == y for y in it) for x in sub)


def _jsonl_ends(path: Path) -> tuple[str | None, str | None]:
    first = last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rid = json.loads(line)["id"]
                first = rid if first is None else first
                last = rid
    return first, last


def shard_done(npz, shard_ids, settings) -> bool:
    """Done only when the npz loads, covers exactly this data shard (n_scanned, ids sha, ids an ordered subsequence),
    was made with these settings, and its jsonl exists with the same first and last ids."""
    from kitsune.store import ids_sha256

    npz = Path(npz)
    if not npz.exists():
        return False
    try:
        # NpzFile is lazy per key: read only what the check needs, never the (compressed) label arrays
        with np.load(npz) as zf:
            z = {key: zf[key] for key in ("format_version", "n_scanned", "shard_ids_sha256", "ids", *SETTING_KEYS)}
        sh = ParakeetShard(z)
        ids = sh.ids
        if int(z["format_version"]) != FORMAT_VERSION or int(z["n_scanned"]) != len(shard_ids):
            return False
        if str(z["shard_ids_sha256"]) != ids_sha256(list(shard_ids)) or sh.settings != normalise_settings(settings):
            return False
        if not _ordered_subsequence(ids, list(shard_ids)):
            return False
        jsonl = npz.with_suffix(".jsonl")
        if not jsonl.exists():
            return False
        return _jsonl_ends(jsonl) == ((ids[0], ids[-1]) if ids else (None, None))
    except Exception:
        return False  # old / partial / corrupt file


def check_shard(path, meta, shard_ids=None) -> list[str]:
    """Every invariant of the format; returns the problems (empty = sound). `meta` is meta.json's dict."""
    try:
        sh = load_shard(path)
    except Exception as e:
        return [f"{path}: does not load: {e}"]
    z, n, p = sh.z, sh.n, []
    need = ("format_version ids duration n_frames truncated n_scanned shard_ids_sha256 k_tdt k_ctc max_symbols "
            "ctc_dense_thr step_offsets step_frame step_dur step_forced tdt_topk_idx tdt_topk_lp tdt_dur_lp tok_offsets "
            "tokens frame_offsets ctc_blank_lp dense_offsets ctc_dense_frame ctc_topk_idx ctc_topk_lp").split()
    missing = [key for key in need if key not in z]
    if missing:
        return [f"missing keys {missing}"]
    if int(z["format_version"]) != FORMAT_VERSION:
        p.append(f"format_version {int(z['format_version'])} != {FORMAT_VERSION}")
    want = {key: _setting(meta, key) for key in SETTING_KEYS if key in meta}
    got = sh.settings
    if any(got[key] != val for key, val in want.items()):
        p.append(f"settings {got} != meta {want}")
    if "format_version" in meta and int(meta["format_version"]) != int(z["format_version"]):
        p.append("format_version != meta")
    kt, kc = int(z["k_tdt"]), int(z["k_ctc"])
    for name, total in (("step_offsets", len(z["step_frame"])), ("tok_offsets", len(z["tokens"])),
                        ("frame_offsets", len(z["ctc_blank_lp"])), ("dense_offsets", len(z["ctc_dense_frame"]))):
        o = z[name]
        if len(o) != n + 1 or o[0] != 0 or o[-1] != total or np.any(np.diff(o) < 0):
            p.append(f"{name} not monotone from 0 to {total} over {n} utterances")
    for name, shape in (("tdt_topk_idx", (len(z["step_frame"]), kt)), ("tdt_topk_lp", (len(z["step_frame"]), kt)),
                        ("tdt_dur_lp", (len(z["step_frame"]), N_DUR)), ("step_dur", (len(z["step_frame"]),)),
                        ("step_forced", (len(z["step_frame"]),)), ("ctc_topk_idx", (len(z["ctc_dense_frame"]), kc)),
                        ("ctc_topk_lp", (len(z["ctc_dense_frame"]), kc))):
        if z[name].shape != shape:
            p.append(f"{name} shape {z[name].shape} != {shape}")
    for name in ("duration", "n_frames", "truncated"):
        if len(z[name]) != n:
            p.append(f"{name} has {len(z[name])} rows, not {n}")
    if p:
        return p  # the per-utterance checks below index with these arrays
    for name in ("tdt_topk_lp", "ctc_topk_lp"):
        lp = z[name].astype(np.float32)
        if lp.shape[1] > 1 and np.any(np.diff(lp, axis=1) > 0):
            p.append(f"{name} not sorted descending")
    if np.any((z["tdt_topk_idx"] < 0) | (z["tdt_topk_idx"] >= VOCAB)) or \
            np.any((z["ctc_topk_idx"] < 0) | (z["ctc_topk_idx"] >= VOCAB)):
        p.append("top-k index outside the vocabulary")
    if np.any((z["step_dur"] < 0) | (z["step_dur"] > N_DUR - 1)):
        p.append("step_dur outside 0..4")
    log_thr = math.log(float(z["ctc_dense_thr"]))
    for i in range(n):
        u = sh.utt(i)
        T = int(z["n_frames"][i])
        uid = sh.ids[i]
        fr = u["frames"]
        if len(fr):
            if np.any((fr < 0) | (fr >= T)):
                p.append(f"{uid}: step_frame outside [0, {T})")
            if not np.array_equal(fr, np.concatenate([[0], np.cumsum(u["dur"])[:-1]])):
                p.append(f"{uid}: step_frame != cumsum(step_dur) shifted")
            if not bool(z["truncated"][i]) and fr[-1] + u["dur"][-1] < T:
                p.append(f"{uid}: decoding stopped before the last frame")
        if np.any(u["dur"][u["tdt_idx"][:, 0] == BLANK] == 0) if len(fr) else False:
            p.append(f"{uid}: a blank step did not advance")
        if np.any(u["dur"][u["forced"]] != 1) if len(fr) else False:
            p.append(f"{uid}: a forced step did not advance by 1")
        col0 = u["tdt_idx"][:, 0] if len(fr) else np.zeros(0, np.int64)
        if not np.array_equal(u["tokens"], col0[col0 != BLANK]):
            p.append(f"{uid}: tokens != non-blank column 0")
        fs, ds = sh._span("frame_offsets", i), sh._span("dense_offsets", i)
        if fs.stop - fs.start != T:
            p.append(f"{uid}: {fs.stop - fs.start} CTC frames, n_frames {T}")
            continue
        dense = z["ctc_dense_frame"][ds].astype(np.int64)
        blank_lp = z["ctc_blank_lp"][fs].astype(np.float32)
        if np.any((dense < 0) | (dense >= T)) or np.any(np.diff(dense) <= 0):
            p.append(f"{uid}: dense frames not increasing inside [0, {T})")
            continue
        is_dense = np.zeros(T, dtype=bool)
        is_dense[dense] = True
        if np.any(blank_lp[is_dense] >= log_thr + LP_TOL) or np.any(blank_lp[~is_dense] < log_thr - LP_TOL):
            p.append(f"{uid}: dense frames do not match blank_lp < log({float(z['ctc_dense_thr'])})")
    if shard_ids is not None:
        from kitsune.store import ids_sha256

        if int(z["n_scanned"]) != len(shard_ids):
            p.append(f"n_scanned {int(z['n_scanned'])} != {len(shard_ids)} data shard rows")
        if str(z["shard_ids_sha256"]) != ids_sha256(list(shard_ids)):
            p.append("shard_ids_sha256 does not match the data shard")
        if not _ordered_subsequence(sh.ids, list(shard_ids)):
            p.append("ids are not an ordered subsequence of the data shard ids")
    return p


def build_meta(settings: dict) -> dict:
    """meta.json of a parakeet_out root: the pins, the decoding and the settings (compared on resume)."""
    from kitsune import parakeet as pk

    s = normalise_settings(settings)
    return dict(
        format_version=FORMAT_VERSION, model=pk.NEMO_REPO, nemo_revision=pk.NEMO_REVISION,
        files_sha256=dict(pk.PARAKEET_FILES), path_in_repo=pk.PARAKEET_PATH,
        decoding=f"greedy TDT, HF semantics + max_symbols {s['max_symbols']}",
        dtypes=dict(encoder="bfloat16", projector="float32", decoder="float32", joint="float32", ctc="float32"),
        vocab=pk.VOCAB, blank=pk.BLANK, durations=list(pk.DURATIONS), frame_s=pk.FRAME_S,
        features="ParakeetFeatureExtractor 80 mel, CPU, no dither",
        k_tdt=s["k_tdt"], k_ctc=s["k_ctc"], ctc_dense_thr=s["ctc_dense_thr"], max_symbols=s["max_symbols"],
        caveat="trained on ReazonSpeech: reazon targets are in-training-data predictions",
    )
