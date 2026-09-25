"""Synthetic on-disk corpus in the repo's exact formats, shared by the tests.

    from fixtures import make_fake_corpus, make_fake_selection   # pytest puts tests/ on sys.path (no __init__.py)
    fc = make_fake_corpus(tmp_path, sources={"src_a": (40, "train"), "eval_x": (12, "eval")})
    sel = make_fake_selection(fc)                                  # runs scripts/make_selection.py on it

make_fake_corpus writes under `root` (usually pytest's tmp_path):
  data/shards/<source>/<split>-NNNNN.parquet   kitsune.store.SCHEMA; audio = real 16 kHz mono FLAC (PCM_16) tones
  data/shards/<source>/progress.json           {"finished_inputs": [], "done": true}
  data/manifest.jsonl                          one kitsune.store.ShardInfo line per shard
  teacher_out/meta.json                        the real keys (decoder_prompt_ids, eos/pad, vocab_size, k, ...)
  teacher_out/<source>/<split>-NNNNN.npz       exactly the FORMAT in scripts/02_teacher_pass.py: ids, tok_offsets,
                                               tokens (int16, incl. EOS unless truncated), topk_idx (int16, col 0 ==
                                               tokens, log-probs sorted descending), topk_logprob (fp16), lse, cer,
                                               duration, n_scanned, k, save_encoder, prompt
  teacher_out/<source>/<split>-NNNNN.jsonl     {id, hyp, ref, cer, duration, n_tok, truncated}
  second_out/<source>/<split>-NNNNN.jsonl      exactly scripts/02b_second_opinion.py: {id, hyp2, model2, agree, cer2}
  second_out/meta.json
and returns a FakeCorpus with the paths and the ground truth per id (tokens, top-k, audio bytes, flags), so a test
checks joins and alignment against what was WRITTEN, not against the code under test.

`sources` maps a source name to (n_labelled_rows, split) or a list of those (a source with train and eval splits,
like galgame). Rows are cut into shards of `rows_per_shard`; teacher/second-opinion files mirror the shard stems.

Knobs for the real data's awkward cases (dicts are per source and apply to its first split; missing source = 0):
  truncated        rows whose teacher output has no final EOS; their agree is drawn in (1, 3] as in the real data
  high_agree_frac  fraction of the other rows whose second opinion disagrees: agree in (0.5, 3]
  null_agree       rows (non-truncated) with a null second opinion (hyp2 / agree / cer2 all None)
  no_second        sources with no second_out files at all (the real eval sets, or 02b not run yet)
  missing_audio    labelled rows (non-truncated; second opinion agrees, if the source has one) removed from the
                   parquet afterwards: a rebuild on another box lost them. Their npz n_scanned still counts them.
  teacher_skipped  parquet rows the teacher never labelled (undecodable at teacher time); n_scanned counts them
  extra_shard_rows one extra shard per source (split of its first entry) that is in the manifest but has no teacher
                   output at all, like galgame train-00096+
Other agree values: half exactly 0.0, the rest uniform in (0, 0.5], and one row per source at exactly 0.5 (the
inclusive boundary) when there are >= 10 rows. agree is drawn, not computed from hyp2. Everything derives from
`seed`: the same arguments give identical files.

Real data. The tests that check the code and this fixture against the REAL files (teacher_out/, data/, second_out/)
read them under REAL and call need_real(...) first. The data is gitignored, so a git worktree has none of it and those
tests skip (SPEC: skip cleanly if absent), which shows only as 's'. To verify a change made in a worktree, point them
at a checkout that has the data (they only read there) and turn a skip into a failure:
    KITSUNE_REAL_DATA_ROOT=<main checkout> KITSUNE_REQUIRE_REAL_DATA=1 python -m pytest -rs tests/...
The gated teacher processor in the local HF cache counts as real data too (teacher_processor()): the override fails
a test that would skip without it. KITSUNE_REAL_DATA_ROOT does not move it; HF_HOME does.
"""
import hashlib
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REAL = Path(os.environ.get("KITSUNE_REAL_DATA_ROOT") or ROOT)  # where the real-data tests read; code stays on ROOT

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402

from kitsune.store import SCHEMA, ShardInfo, append_manifest, save_progress  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402

PROMPT = [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13]
EOS, PAD = 3, 2
SR = 16000
_KANA = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")


@dataclass
class FakeUtt:
    """Ground truth for one generated utterance."""

    id: str
    source: str
    split: str
    stem: str  # shard stem, e.g. "train-00000" (teacher/second files use the same stem)
    duration: float  # float32 value as stored in the parquet and npz
    audio: bytes  # FLAC bytes as stored in the parquet
    text: str  # dataset transcript (parquet `text`, jsonl `ref`)
    has_teacher: bool
    has_audio: bool
    tokens: np.ndarray | None = None  # (T,) int16
    topk_idx: np.ndarray | None = None  # (T, k) int16
    topk_lp: np.ndarray | None = None  # (T, k) float16
    lse: np.ndarray | None = None  # (T,) float32
    truncated: bool = False
    hyp: str = ""
    cer: float = 0.0
    agree: float | None = None  # as written to second_out (rounded to 4), None if null / no second_out
    has_second: bool = False  # a second_out row exists


@dataclass
class FakeCorpus:
    root: Path
    data: Path
    teacher_out: Path
    second_out: Path
    prompt: list[int]
    eos: int
    pad: int
    k: int
    vocab_size: int
    utts: dict[str, FakeUtt] = field(default_factory=dict)  # generation order

    def ids(self, source: str | None = None, split: str | None = None, *, teacher: bool | None = True,
            audio: bool | None = None) -> list[str]:
        """Ids filtered by source/split and by whether they have teacher output / audio (None = either)."""
        return [u.id for u in self.utts.values()
                if (source is None or u.source == source) and (split is None or u.split == split)
                and (teacher is None or u.has_teacher == teacher) and (audio is None or u.has_audio == audio)]

    def sources(self, split: str) -> list[str]:
        return sorted({u.source for u in self.utts.values() if u.split == split and u.has_teacher})


def _flac(n: int, rng: np.random.Generator) -> bytes:
    t = np.arange(n) / SR
    wav = 0.3 * np.sin(2 * np.pi * rng.uniform(150, 1500) * t) + 0.01 * rng.standard_normal(n)
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _topk(tokens: np.ndarray, k: int, lo: int, hi: int, rng: np.random.Generator):
    """Top-k ids with col 0 = the greedy token and log-probs sorted descending, with a small tail outside the top-k."""
    T = len(tokens)
    idx = np.empty((T, k), dtype=np.int64)
    lp = np.empty((T, k), dtype=np.float64)
    for t in range(T):
        cand = rng.choice(hi - lo, size=k, replace=False) + lo
        cand = cand[cand != tokens[t]][: k - 1]
        idx[t, 0], idx[t, 1:] = tokens[t], cand
        p1 = rng.uniform(0.5, 0.999)
        rest = np.sort(rng.dirichlet(np.ones(k - 1)))[::-1] * (1 - p1) * rng.uniform(0.6, 1.0)
        lp[t] = np.log(np.concatenate([[p1], np.maximum(rest, 1e-30)]))
    return idx.astype(np.int16), lp.astype(np.float16), rng.uniform(15, 25, T).astype(np.float32)


def _pick(rng: np.random.Generator, pool: list[int], n: int) -> set[int]:
    if n > len(pool):
        raise ValueError(f"asked for {n} special rows but only {len(pool)} are eligible")
    return {pool[i] for i in rng.choice(len(pool), size=n, replace=False)} if n else set()


def make_fake_corpus(root, sources: dict | None = None, *, rows_per_shard: int = 32, row_group_size: int = 8,
                     seed: int = 0, dur_range: tuple[float, float] = (0.3, 6.0), tok_per_s: float = 4.0,
                     k: int = 16, vocab_size: int = 16384, prompt: list[int] = PROMPT, eos: int = EOS, pad: int = PAD,
                     token_range: tuple[int, int] | None = None, truncated: dict[str, int] | None = None,
                     high_agree_frac: float = 0.15, null_agree: dict[str, int] | None = None, no_second=(),
                     missing_audio: dict[str, int] | None = None, teacher_skipped: dict[str, int] | None = None,
                     extra_shard_rows: dict[str, int] | None = None) -> FakeCorpus:
    root = Path(root)
    sources = sources or {"src_a": (48, "train"), "eval_x": (16, "eval")}
    lo, hi = token_range or (256, vocab_size)
    if not (0 <= lo < hi <= vocab_size and hi - lo > k):
        raise ValueError(f"token_range {lo, hi} must lie inside the vocab and hold more than k ids")
    truncated, null_agree = truncated or {}, null_agree or {}
    missing_audio, teacher_skipped, extra_shard_rows = missing_audio or {}, teacher_skipped or {}, extra_shard_rows or {}
    fc = FakeCorpus(root, root / "data", root / "teacher_out", root / "second_out", list(prompt), eos, pad, k, vocab_size)
    fc.teacher_out.mkdir(parents=True, exist_ok=True)
    fc.second_out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def make_id(source: str, split: str, i: int) -> str:
        h = hashlib.sha1(f"{seed}/{source}/{split}/{i}".encode()).hexdigest()[:13]
        return f"{source}/{h[:3]}/{h}.flac"

    def make_row(source: str, split: str, i: int) -> FakeUtt:
        n = int(round(rng.uniform(*dur_range) * SR))
        ref = "".join(rng.choice(_KANA, size=max(1, int(n / SR * 3))))
        return FakeUtt(make_id(source, split, i), source, split, "", float(np.float32(n / SR)), _flac(n, rng), ref,
                       has_teacher=False, has_audio=True)

    for source, spec in sources.items():
        entries = spec if isinstance(spec, list) else [spec]
        for n_lab, split in entries:
            n_skip = teacher_skipped.get(source, 0) if split == entries[0][1] else 0
            total = n_lab + n_skip
            skipped = _pick(rng, list(range(total)), n_skip)
            rows = [make_row(source, split, i) for i in range(total)]
            lab = [i for i in range(total) if i not in skipped]
            tr = _pick(rng, lab, truncated.get(source, 0) if split == entries[0][1] else 0)
            nul = _pick(rng, [i for i in lab if i not in tr], null_agree.get(source, 0) if split == entries[0][1] else 0)
            write_second = source not in no_second
            ok_rows = []  # labelled, second opinion agrees: candidates for missing_audio
            for j, i in enumerate(lab):
                u = rows[i]
                u.has_teacher, u.truncated = True, i in tr
                body = rng.integers(lo, hi, size=max(1, int(round(tok_per_s * u.duration))))
                u.tokens = (body if u.truncated else np.concatenate([body, [eos]])).astype(np.int16)
                u.topk_idx, u.topk_lp, u.lse = _topk(u.tokens.astype(np.int64), k, lo, hi, rng)
                chars = list(u.text)
                for c in rng.choice(len(chars), size=int(rng.integers(0, 2)), replace=False):
                    chars[c] = str(rng.choice(_KANA))
                u.hyp = "".join(chars)
                u.cer = float(np.float32(cer_fn(u.hyp, u.text)))
                if write_second:
                    u.has_second = True
                    if i in nul:
                        u.agree = None
                    elif u.truncated:
                        u.agree = round(float(rng.uniform(1.0, 3.0)), 4)
                    elif rng.random() < high_agree_frac:
                        u.agree = round(float(rng.uniform(0.5001, 3.0)), 4)
                    else:
                        u.agree = 0.0 if rng.random() < 0.5 else round(float(rng.uniform(0.0, 0.5)), 4)
                if not u.truncated and (not write_second or (u.agree is not None and u.agree <= 0.5)):
                    ok_rows.append(i)
            gone = _pick(rng, ok_rows, missing_audio.get(source, 0) if split == entries[0][1] else 0)
            for i in gone:
                rows[i].has_audio = False
            edge = [i for i in ok_rows if i not in gone]
            if write_second and len(lab) >= 10 and edge:
                rows[edge[0]].agree = 0.5  # the inclusive boundary of the default gate

            shard_infos = []
            for s0 in range(0, total, rows_per_shard):
                stem = f"{split}-{s0 // rows_per_shard:05d}"
                chunk = rows[s0:s0 + rows_per_shard]
                for u in chunk:
                    u.stem = stem
                shard_infos.append(_write_shard(fc, source, split, stem, [u for u in chunk if u.has_audio], row_group_size))
                labelled = [u for u in chunk if u.has_teacher]
                _write_teacher(fc, source, stem, labelled, n_scanned=len(chunk))
                if write_second:
                    _write_second(fc.second_out / source / f"{stem}.jsonl", labelled)
            if source in extra_shard_rows and split == entries[0][1]:
                stem = f"{split}-{-(-total // rows_per_shard):05d}"
                extra = [make_row(source, split, total + i) for i in range(extra_shard_rows[source])]
                for u in extra:
                    u.stem = stem
                shard_infos.append(_write_shard(fc, source, split, stem, extra, row_group_size))
                rows += extra
            append_manifest(fc.data, shard_infos)
            for u in rows:
                fc.utts[u.id] = u
        save_progress(fc.data, source, {"finished_inputs": [], "done": True})

    (fc.teacher_out / "meta.json").write_text(json.dumps(dict(
        model="fake/teacher", model_revision="0" * 40, language="ja", punctuation=True, k=k, save_encoder=False,
        decoder_prompt_ids=list(prompt),
        decoder_prompt_tokens=[f"<{p}>" for p in prompt], eos_token_id=eos, pad_token_id=pad, vocab_size=vocab_size,
        encoder_hidden_size=1280, decoding="greedy", lm_head_dtype="float32", model_dtype="bfloat16",
    ), indent=2), encoding="utf-8")
    (fc.second_out / "meta.json").write_text(json.dumps(dict(
        join_model="whisper-large-v3 (precomputed)", gpu_model="kotoba-tech/kotoba-whisper-v2.0",
        tokenizer="openai/whisper-large-v3", agree="cer(teacher_hyp, hyp2)", cer2="cer(hyp2, dataset_text)"),
        indent=2), encoding="utf-8")
    return fc


def _write_shard(fc: FakeCorpus, source: str, split: str, stem: str, rows: list[FakeUtt], row_group_size: int) -> ShardInfo:
    d = fc.data / "shards" / source
    d.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(id=u.id, source=source, split=split, audio=u.audio, text=u.text,
                                       duration=u.duration, sr=SR) for u in rows], schema=SCHEMA)
    path = d / f"{stem}.parquet"
    pq.write_table(table, path, compression="none", row_group_size=row_group_size)
    return ShardInfo(path.relative_to(fc.data).as_posix(), source, split, len(rows), sum(u.duration for u in rows) / 3600)


def _write_teacher(fc: FakeCorpus, source: str, stem: str, rows: list[FakeUtt], n_scanned: int):
    """Byte-for-byte the packing of 02_teacher_pass.process_shard."""
    out = fc.teacher_out / source
    out.mkdir(parents=True, exist_ok=True)
    lens = np.array([len(u.tokens) for u in rows], dtype=np.int64)
    packed = dict(
        ids=np.array([u.id for u in rows]),
        tok_offsets=np.concatenate([[0], np.cumsum(lens)]),
        tokens=np.concatenate([u.tokens for u in rows]) if rows else np.zeros(0, np.int16),
        topk_idx=np.concatenate([u.topk_idx for u in rows]) if rows else np.zeros((0, fc.k), np.int16),
        topk_logprob=np.concatenate([u.topk_lp for u in rows]) if rows else np.zeros((0, fc.k), np.float16),
        lse=np.concatenate([u.lse for u in rows]) if rows else np.zeros(0, np.float32),
        cer=np.array([u.cer for u in rows], dtype=np.float32),
        duration=np.array([u.duration for u in rows], dtype=np.float32),
        n_scanned=np.int64(n_scanned),
        k=np.int64(fc.k),
        save_encoder=np.bool_(False),
        prompt=np.array(fc.prompt, dtype=np.int64),
    )
    with open(out / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for u in rows:
            f.write(json.dumps(dict(id=u.id, hyp=u.hyp, ref=u.text, cer=round(float(u.cer), 4),
                                    duration=round(float(u.duration), 3), n_tok=int(len(u.tokens)),
                                    truncated=bool(u.truncated)), ensure_ascii=False) + "\n")
    with open(out / f"{stem}.npz", "wb") as f:
        np.savez(f, **packed)


def _write_second(path: Path, rows: list[FakeUtt]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in rows:
            if u.agree is None:
                r = dict(id=u.id, hyp2=None, model2=None, agree=None, cer2=None)
            else:
                hyp2 = u.hyp if u.agree == 0.0 else u.hyp[::-1]
                r = dict(id=u.id, hyp2=hyp2, model2="whisper-large-v3", agree=u.agree,
                         cer2=round(cer_fn(hyp2, u.text), 4))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def no_real_data(reason: str) -> None:
    """Skip the calling test for missing real data, or FAIL it when KITSUNE_REQUIRE_REAL_DATA=1: a verification run
    that must check the real formats cannot then pass on a skip."""
    __tracebackhide__ = True  # report the skip/failure at the calling test's line, not here
    if os.environ.get("KITSUNE_REQUIRE_REAL_DATA") == "1":
        pytest.fail(f"{reason} (KITSUNE_REQUIRE_REAL_DATA=1, real data root {REAL})")
    pytest.skip(reason)


def need_real(*paths: Path) -> None:
    """no_real_data unless every one of these real-data paths exists."""
    __tracebackhide__ = True
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        no_real_data("real data not present: " + ", ".join(missing))


def teacher_processor():
    """The gated teacher's processor from the local HF cache (the tests run with HF_HUB_OFFLINE=1), or no_real_data."""
    __tracebackhide__ = True
    from kitsune.features import load_hf_processor

    try:
        return load_hf_processor()
    except Exception as e:  # noqa: BLE001 - not cached / offline
        no_real_data(f"teacher processor not in the local HF cache (HF_HOME): {type(e).__name__}: {e}")


def load_script(name: str):
    """Import scripts/<name>.py as a module (works for names starting with a digit, e.g. "02_teacher_pass")."""
    spec = importlib.util.spec_from_file_location(f"kitsune_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fake_selection(fc: FakeCorpus, out=None, *, sources: list[str] | None = None,
                        eval_sets: list[str] | None = None, agree_max: float = 0.5, seed: int = 1234,
                        greedy_n: int = 500, probe_n: int = 500, extra_args: tuple = ()) -> Path:
    """Run scripts/make_selection.py (its real CLI) on the fake corpus; default: every train source and every
    eval source of the corpus. Returns the selection parquet path."""
    out = Path(out) if out else fc.root / "selection" / "selection.parquet"
    sources = sources if sources is not None else fc.sources("train")
    eval_sets = eval_sets if eval_sets is not None else fc.sources("eval")
    argv = ["--sources", *sources, "--eval-sets", *eval_sets, "--agree-max", str(agree_max), "--out", str(out),
            "--teacher-out", str(fc.teacher_out), "--second-out", str(fc.second_out), "--data", str(fc.data),
            "--seed", str(seed), "--greedy-n", str(greedy_n), "--probe-n", str(probe_n), *extra_args]
    load_script("make_selection").main(argv)
    return out
