"""Label checks K1-K12 of the size study: what the label box stores is what the study will read (laptop, CPU).

The study design's section 3.4 and its checklist "What the label box must store NOW". The checks run against a LOCAL copy
of a label root (labels/full: teacher_out/, parakeet_out/, second_out/ and, once the box has sealed the root,
extent.json, COMPLETE.json and reports/) and write one JSON report. `pull` makes that copy from the data repo at one
pinned commit: every meta.json, every file of the eval sets, a few train stems per source (first, middle and last of the
stems both passes labelled), the root-level json files and reports/, plus the root's full listing with file sizes in
<out>/_pull.json. Coverage (K5, K6, K12) is judged on that listing, i.e. on the whole root; the row-level checks read
the pulled stems. Re-run both on the sealed root: K12 then decides, and the pending checks resolve.

Every check returns {"status": "pass" | "fail" | "pending" | "skipped", "reason": str, ...its numbers}:
  pending   the data it needs is not labelled yet, not pulled, or the root is not sealed;
  skipped   an input of this tool is missing (--model-dir, --data, --kotoba); the reason says which.

  K1  parakeet_out/meta.json has exactly the pins the study reads: format 1, k_ctc 8, ctc_dense_thr 0.95, k_tdt 8,
      max_symbols 10, the features string ("... 80 mel, CPU, no dither"), encoder bf16 / CTC fp32, the NeMo repo and
      revision, files_sha256 = kitsune.parakeet.PARAKEET_FILES, blank 3072, vocab 3073, frame 0.08 s.
  K2  every pulled parakeet npz passes kitsune.parakeet_targets.check_shard (with the data shard's ids when the --data
      root holds the same shard), stores the documented dtypes, and its jsonl has the same ids in the same order with
      the fields the study reads (JSONL_KEYS), n_frames and truncated equal to the npz's.
  K3  decode(ctc_greedy(ctc_col0(n_frames, ctc_dense_frame, ctc_topk_idx))) equals the jsonl ctc_hyp on every pulled
      row: the CTC target (decision 22) can be derived from the npz alone. Decoded with the pinned Parakeet
      processor, as scripts/02p_parakeet_pass.py decodes.
  K4  frame parity, about 50 rows per train source and eval set: the audio of the --data root (only shards whose ids
      digest equals the npz's shard_ids_sha256, i.e. the audio that was labelled) through (a) the
      ParakeetFeatureExtractor of --model-dir and (b) kitsune.features.LogMel on that extractor's filterbank with
      dither 0, each followed by the encoder's 8x subsampling length formula (subsampled_length), must give the
      stored n_frames on 100 % of rows. The closed form from the sample count alone is reported too.
  K5  coverage: every teacher_out stem has a parakeet_out stem and back; on the pulled pairs n_scanned is equal and
      the rows present in one root only are < 0.1 %; the parakeet shard_ids_sha256 equals the extent record's
      ids_sha256 of the stem (sealed root). Stems in one root only are in progress while the root is not sealed.
  K6  the eval sets are labelled whole by both passes (the same ids in the same order, the manifest's row counts) and
      the Parakeet corpus CER on JSUT is near the model card (TDT 6.4 / CTC 6.5 %; "near" = within CARD_TOL);
      reports/parakeet_baselines.json, once the box writes it (finalize F3), must equal the recomputation here.
  K7  CTC feasibility, U + repeats <= n_frames, of the target the study uses (decision 22, greedy CTC: feasible by
      construction, so a failure means an inconsistent npz) and, for reference, of the TDT tokens.
  K8  the rate of hyp != ctc_hyp and the corpus CER of ctc_hyp against hyp on the pulled train rows (decision 22).
  K9  parakeet_out bytes (npz + jsonl) per audio-hour on the pulled train stems (decision 5, HF storage).
  K10 the galgame eval-00000 ids of both label roots equal the laptop's kotoba file (second_out/galgame/eval-00000.jsonl),
      so its hypotheses give the neutral Galgame view (decision 11); its cer2 <= 0.5 row count is reported.
  K11 the Cohere eval baselines recomputed from teacher_out (kitsune.evaluate.teacher_baselines) equal
      TEACHER_CER_PREREG within BASELINE_TOL (0.05 pp).
  K12 the sealed extent covers the study mix in both roots (kitsune.extent.subset_stems of the mix): reazon_large >= 55
      inputs (60 for margin), emilia_nc >= 8, galgame >= 3, the emilia_yodas 300 h step and every eval set.
K7-K9 are statistics the design assumed ("about X" in its table); they pass up to ABOUT x X (2x) and fail above, since a
larger value breaks the assumption the named decision was taken on.

Usage:
  python tools/label_checks.py pull --out D:/kitsune-labels/full [--revision SHA] [--train-stems 3]
  python tools/label_checks.py run --labels D:/kitsune-labels/full --data D:/Shizu-ko-distill/data \\
      --model-dir D:/Shizu-ko-distill/cache/parakeet-tdt_ctc-0.6b-ja-hf \\
      --kotoba D:/Shizu-ko-distill/second_out/galgame/eval-00000.jsonl --out label_checks_report.json
`run` exits 1 if any check fails, else 0 (pending is not a failure). CPU only; `pull` is the only network step (reads).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from kitsune import extent as ext  # noqa: E402
from kitsune import parakeet_targets as pt  # noqa: E402
from kitsune.evaluate import BASELINE_TOL, TEACHER_CER_PREREG, corpus_cer, teacher_baselines  # noqa: E402
from kitsune.parakeet import (  # noqa: E402
    BLANK, FRAME_S, NEMO_REPO, NEMO_REVISION, PARAKEET_FILES, PARAKEET_PATH, SAMPLING_RATE, VOCAB,
)
from kitsune.store import ShardInfo, ids_sha256, read_ids  # noqa: E402
from kitsune.text import normalize_ja  # noqa: E402

DEFAULT_REPO = "Multy123/kitsune-data"
DEFAULT_ROOT = "labels/full"
PULL_FILE = "_pull.json"
ROOTS = ("teacher_out", "parakeet_out", "second_out")
TRAIN_SOURCES = ("reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame")
EVAL_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame")  # galgame = its eval-* hold-out stem
# the settings the study's loaders assume (the label box's configs/full.json "label.parakeet")
STUDY_SETTINGS = dict(k_tdt=8, k_ctc=8, max_symbols=10, ctc_dense_thr=0.95)
STUDY_MIX = {"reazon_large": 55, "emilia_yodas": "300h", "emilia_nc": 8, "galgame": 3}  # mix D minimum (3.1, K12)
MIX_MARGIN = {"reazon_large": 60}
EVAL_ROWS = {"eval_jsut": 5000, "eval_cv8": 4483, "eval_reazon": 5263, "galgame": 1000}  # the frozen manifest (4.1)
PARAKEET_CARD = {"eval_jsut": {"tdt": 0.064, "ctc": 0.065}}  # nvidia/parakeet-tdt_ctc-0.6b-ja model card, JSUT
CARD_TOL = 0.005  # "near the card": our normalize_ja scoring is not NeMo's, so 0.5 pp, not the 0.05 pp of K11
ONE_ROOT_MAX = 0.001  # K5: rows labelled by one pass only (02 and 02p skip the same >30 s rows; decode failures differ)
EXPECT = {"k7_infeasible": 0.002, "k8_cer": 0.01, "k9_mb_per_audio_h": 1.34}  # the design's figures (3.4)
ABOUT = 2.0
NEUTRAL_MAX = 0.5  # the neutral Galgame view keeps cer(kotoba, ref) <= 0.5 (decision 11: 810 rows)
KOTOBA_STEM = "eval-00000"
JSONL_KEYS = ("id", "hyp", "ctc_hyp", "ref", "cer", "ctc_cer", "n_frames", "duration", "truncated")
# the npz fields a CTC student reads, with their documented dtypes (kitsune/parakeet_targets.py FORMAT)
NPZ_DTYPES = {"format_version": np.int64, "n_frames": np.int32, "duration": np.float32, "truncated": np.bool_,
              "n_scanned": np.int64, "k_ctc": np.int64, "ctc_dense_thr": np.float64, "frame_offsets": np.int64,
              "ctc_blank_lp": np.float16, "dense_offsets": np.int64, "ctc_dense_frame": np.int16,
              "ctc_topk_idx": np.int16, "ctc_topk_lp": np.float16, "tokens": np.int16, "tok_offsets": np.int64}
NPZ_STRINGS = ("ids", "shard_ids_sha256")
FEATURES_PIN = "80 mel, CPU, no dither"
SUBSAMPLING = {"kernel": 3, "stride": 2, "factor": 8}  # config.json encoder_config of the pinned converted model
K4_ROWS, K4_UNITS = 50, 4  # rows per source, drawn from this many (shard, row group) units (bounded parquet reads)
PULL_TRAIN_STEMS, PULL_MAX_GB = 3, 5.0
MAX_LISTED = 20  # examples kept per problem list in the report


# ------------------------------------------------------------------------------------------------ frame counts


def feature_frames(n_samples: int, n_fft: int = 512, hop_length: int = 160) -> int:
    """ParakeetFeatureExtractor's valid frames (its attention_mask sum): (n + n_fft // 2 * 2 - n_fft) // hop."""
    return (int(n_samples) + n_fft // 2 * 2 - n_fft) // hop_length


def subsampled_length(n_feat: int, kernel: int = 3, stride: int = 2, factor: int = 8) -> int:
    """The encoder's valid length after its conv subsampling, as transformers computes it
    (ParakeetPreTrainedModel._get_subsampling_output_length, 5.13.1): log2(factor) times
    floor((L + add_pad) / stride + 1) with add_pad = 2 * ((kernel - 1) // 2) - kernel. The label pass's n_frames is
    the sum of the encoder's output attention mask, which is exactly this of the feature mask sum."""
    add_pad = (kernel - 1) // 2 * 2 - kernel
    length = float(n_feat)
    for _ in range(int(math.log2(factor))):
        length = math.floor((length + add_pad) / stride + 1.0)
    return int(length)


def expected_n_frames(n_samples: int, sub: dict | None = None) -> int:
    """n_frames of a 16 kHz waveform of n_samples samples: the closed form of the two steps above."""
    return subsampled_length(feature_frames(n_samples), **(sub or SUBSAMPLING))


def model_subsampling(model_dir: Path) -> dict:
    enc = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))["encoder_config"]
    return {"kernel": int(enc["subsampling_conv_kernel_size"]), "stride": int(enc["subsampling_conv_stride"]),
            "factor": int(enc["subsampling_factor"])}


def ctc_feasible_need(ids) -> int:
    """Frames a CTC alignment of `ids` needs: one per token plus one blank between equal neighbours."""
    ids = [int(x) for x in ids]
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


# ---------------------------------------------------------------------------------------------------- helpers


def read_json(path: Path) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def jsonl_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def result(status: str, reason: str = "", **numbers) -> dict:
    return {"status": status, "reason": reason, **numbers}


def worst(statuses) -> str:
    """fail > pending > skipped > pass over sub-results (an empty list is pending: nothing was checked)."""
    s = set(statuses)
    for level in ("fail", "pending", "skipped"):
        if level in s:
            return level
    return "pass" if s else "pending"


def is_eval_stem(src: str, stem: str) -> bool:
    return src.startswith("eval_") or stem.startswith("eval-")


def group_of(src: str, stem: str) -> str:
    """Report key of a stem: the source for train stems and eval sets, "<src>:eval" for a train source's hold-out."""
    return src if src.startswith("eval_") or not stem.startswith("eval-") else f"{src}:eval"


def eval_group(name: str) -> str:
    return name if name.startswith("eval_") else f"{name}:eval"


def local_listing(labels: Path) -> dict[str, int]:
    return {p.relative_to(labels).as_posix(): p.stat().st_size for p in Path(labels).rglob("*")
            if p.is_file() and p.name != PULL_FILE and not p.name.endswith(".tmp")}


def index_listing(listing) -> dict[tuple[str, str, str], set[str]]:
    """(root, source, "npz" | "jsonl") -> stems, from root-relative paths <root>/<source>/<stem>.<ext>."""
    idx: dict[tuple[str, str, str], set[str]] = {}
    for p in listing:
        parts = p.split("/")
        if len(parts) == 3 and parts[0] in ROOTS:
            stem, dot, suffix = parts[2].rpartition(".")
            if dot and suffix in ("npz", "jsonl"):
                idx.setdefault((parts[0], parts[1], suffix), set()).add(stem)
    return idx


def spread(items: list, n: int) -> list:
    """n items spread over the list, first and last included."""
    if len(items) <= n:
        return list(items)
    if n <= 1:
        return items[:n]
    return [items[i] for i in sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})]


def kitsune_sha() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a checkout / no git
        return None


# ------------------------------------------------------------------------------------------------------ context


@dataclass
class Ctx:
    """Everything the checks read. `listing` is the whole root's (root-relative path -> bytes); the files the checks
    open are the ones present under `labels`. `decode` (token id lists -> texts) and `fe` (a ParakeetFeatureExtractor)
    come from the pinned model dir; tests pass stand-ins."""

    labels: Path
    listing: dict
    pull: dict | None = None
    data: Path | None = None
    decode: Callable | None = None
    fe: object | None = None
    sub: dict = field(default_factory=lambda: dict(SUBSAMPLING))
    kotoba: Path | None = None
    train_sources: tuple = TRAIN_SOURCES
    eval_sets: tuple = EVAL_SETS
    mix: dict = field(default_factory=lambda: dict(STUDY_MIX))
    margin: dict = field(default_factory=lambda: dict(MIX_MARGIN))
    card: dict | None = field(default_factory=lambda: dict(PARAKEET_CARD))
    prereg: dict = field(default_factory=lambda: dict(TEACHER_CER_PREREG))
    eval_rows: dict = field(default_factory=lambda: dict(EVAL_ROWS))
    expect: dict = field(default_factory=lambda: dict(EXPECT))
    ctc_target: str = "greedy_ctc"  # decision 22; "tdt" = the TDT hypothesis tokens
    k4_rows: int = K4_ROWS
    k4_units: int = K4_UNITS
    seed: int = 0
    unavailable: dict = field(default_factory=dict)  # "decode" / "fe" -> why it is missing (the skip reason)

    def __post_init__(self):
        self.labels = Path(self.labels)
        self.idx = index_listing(self.listing)
        self._pk, self._rows, self._ctc, self._data_ids, self._t = {}, {}, {}, {}, {}

    @property
    def sealed(self) -> bool:
        return "COMPLETE.json" in self.listing or (self.labels / "COMPLETE.json").is_file()

    def listed(self, root: str, src: str, suffix: str = "npz") -> set[str]:
        return self.idx.get((root, src, suffix), set())

    def labelled(self, root: str, src: str) -> set[str]:
        """Stems with both files of a label root in the listing (second_out has jsonl only)."""
        if root == "second_out":
            return set(self.listed(root, src, "jsonl"))
        return self.listed(root, src, "npz") & self.listed(root, src, "jsonl")

    def listed_sources(self, root: str) -> list[str]:
        return sorted({src for (r, src, _) in self.idx if r == root})

    def path(self, root: str, src: str, stem: str, suffix: str) -> Path:
        return self.labels / root / src / f"{stem}.{suffix}"

    def has(self, root: str, src: str, stem: str) -> bool:
        suffixes = ("jsonl",) if root == "second_out" else ("npz", "jsonl")
        return all(self.path(root, src, stem, s).is_file() for s in suffixes)

    def pulled(self, root: str = "parakeet_out") -> list[tuple[str, str]]:
        """(source, stem) of every stem of `root` present locally with both of its files."""
        base = self.labels / root
        if not base.is_dir():
            return []
        return [(d.name, f.stem) for d in sorted(base.iterdir()) if d.is_dir() for f in sorted(d.glob("*.npz"))
                if self.has(root, d.name, f.stem)]

    def pk(self, src: str, stem: str) -> pt.ParakeetShard:
        key = (src, stem)
        if key not in self._pk:
            self._pk[key] = pt.load_shard(self.path("parakeet_out", src, stem, "npz"))
        return self._pk[key]

    def pk_rows(self, src: str, stem: str) -> list[dict]:
        key = (src, stem)
        if key not in self._rows:
            self._rows[key] = jsonl_rows(self.path("parakeet_out", src, stem, "jsonl"))
        return self._rows[key]

    def ctc_ids(self, src: str, stem: str) -> list[list[int]]:
        """The greedy CTC target of every row, derived from the npz only (decision 22)."""
        key = (src, stem)
        if key not in self._ctc:
            z = self.pk(src, stem).z
            nf, do = z["n_frames"], z["dense_offsets"]
            self._ctc[key] = [pt.ctc_greedy(pt.ctc_col0(int(nf[i]), z["ctc_dense_frame"][do[i]:do[i + 1]],
                                                        z["ctc_topk_idx"][do[i]:do[i + 1]]))
                              for i in range(len(nf))]
        return self._ctc[key]

    def teacher(self, src: str, stem: str) -> dict:
        """ids and n_scanned of a pulled teacher_out npz."""
        key = (src, stem)
        if key not in self._t:
            with np.load(self.path("teacher_out", src, stem, "npz")) as z:
                self._t[key] = {"ids": [str(x) for x in z["ids"]], "n_scanned": int(z["n_scanned"])}
        return self._t[key]

    def data_ids(self, src: str, stem: str) -> list[str] | None:
        """The --data root's ids of this shard (its id sidecar or the parquet), None if it has no such shard."""
        if self.data is None:
            return None
        key = (src, stem)
        if key not in self._data_ids:
            split = "eval" if stem.startswith("eval-") else "train"
            info = ShardInfo(f"shards/{src}/{stem}.parquet", src, split, 0, 0.0)
            try:
                self._data_ids[key] = [r["id"] for r in read_ids(self.data, info)]
            except FileNotFoundError:
                self._data_ids[key] = None
        return self._data_ids[key]

    def record(self) -> dict | None:
        return read_json(self.labels / ext.RECORD_FILE)


# -------------------------------------------------------------------------------------------------------- checks


def k1_meta(ctx: Ctx) -> dict:
    meta = read_json(ctx.labels / "parakeet_out" / "meta.json")
    if meta is None:
        why = "not pulled" if "parakeet_out/meta.json" in ctx.listing else "not written yet"
        return result("pending", f"parakeet_out/meta.json {why}")
    want = pt.build_meta(STUDY_SETTINGS)
    pins = {key: want[key] for key in ("format_version", "model", "nemo_revision", "files_sha256", "vocab", "blank",
                                       "frame_s", "k_tdt", "k_ctc", "ctc_dense_thr", "max_symbols", "features")}
    pins["files_sha256"], pins["nemo_revision"], pins["model"] = dict(PARAKEET_FILES), NEMO_REVISION, NEMO_REPO
    pins["vocab"], pins["blank"], pins["frame_s"] = VOCAB, BLANK, FRAME_S
    got = {key: meta.get(key) for key in pins}
    got["dtypes.encoder"], pins["dtypes.encoder"] = (meta.get("dtypes") or {}).get("encoder"), "bfloat16"
    got["dtypes.ctc"], pins["dtypes.ctc"] = (meta.get("dtypes") or {}).get("ctc"), "float32"
    diffs = {key: {"got": got[key], "want": val} for key, val in pins.items() if got[key] != val}
    if FEATURES_PIN not in str(meta.get("features", "")):
        diffs.setdefault("features", {"got": meta.get("features"), "want": f"... {FEATURES_PIN}"})
    if diffs:
        return result("fail", f"{len(diffs)} pin(s) differ from what the study reads", diffs=diffs, meta=meta)
    return result("pass", "every pin equal", pins=sorted(pins), meta=meta)


def k2_format(ctx: Ctx) -> dict:
    meta = read_json(ctx.labels / "parakeet_out" / "meta.json")
    pairs = ctx.pulled()
    if meta is None or not pairs:
        return result("pending", "no parakeet_out meta.json or shard pulled")
    problems, per = [], {}
    data_checked, data_other = [], []
    for src, stem in pairs:
        tag = f"{src}/{stem}"
        sh = ctx.pk(src, stem)
        z = sh.z
        g = per.setdefault(group_of(src, stem), {"shards": 0, "rows": 0})
        g["shards"] += 1
        g["rows"] += sh.n
        shard_ids = ctx.data_ids(src, stem)
        same = shard_ids is not None and "shard_ids_sha256" in z and ids_sha256(shard_ids) == str(z["shard_ids_sha256"])
        if same:
            data_checked.append(tag)
        elif shard_ids is not None:
            data_other.append(tag)
        problems += [f"{tag}: {p}" for p in pt.check_shard(ctx.path("parakeet_out", src, stem, "npz"), meta,
                                                            shard_ids if same else None)]
        for key, dt in NPZ_DTYPES.items():
            if key in z and z[key].dtype != np.dtype(dt):
                problems.append(f"{tag}: {key} dtype {z[key].dtype}, documented {np.dtype(dt)}")
        problems += [f"{tag}: {key} is not a string array" for key in NPZ_STRINGS if key in z and z[key].dtype.kind != "U"]
        if "ctc_topk_idx" in z and z["ctc_topk_idx"].ndim == 2 and z["ctc_topk_idx"].shape[1] != STUDY_SETTINGS["k_ctc"]:
            problems.append(f"{tag}: ctc_topk_idx has k {z['ctc_topk_idx'].shape[1]}, the study reads "
                            f"{STUDY_SETTINGS['k_ctc']}")
        rows = ctx.pk_rows(src, stem)
        if [r.get("id") for r in rows] != sh.ids:
            problems.append(f"{tag}: jsonl ids are not the npz ids in the same order ({len(rows)} vs {sh.n} rows)")
            continue
        missing = sorted({key for r in rows for key in JSONL_KEYS if key not in r})
        if missing:
            problems.append(f"{tag}: jsonl rows lack {missing}")
            continue
        nf, tr = z["n_frames"], z["truncated"]
        bad = [r["id"] for i, r in enumerate(rows)
               if int(r["n_frames"]) != int(nf[i]) or bool(r["truncated"]) != bool(tr[i])]
        if bad:
            problems.append(f"{tag}: {len(bad)} jsonl rows disagree with the npz on n_frames/truncated, e.g. {bad[:3]}")
    common = dict(shards=len(pairs), rows=sum(g["rows"] for g in per.values()), per_group=per,
                  data_ids_checked=len(data_checked), data_other_ids=data_other[:MAX_LISTED],
                  n_data_other_ids=len(data_other), n_problems=len(problems), problems=problems[:MAX_LISTED])
    if problems:
        return result("fail", f"{len(problems)} problem(s) in {len(pairs)} pulled shards", **common)
    return result("pass", f"{len(pairs)} shards sound", **common)


def k3_ctc_target(ctx: Ctx) -> dict:
    if ctx.decode is None:
        return result("skipped", ctx.unavailable.get("decode", "no Parakeet processor (--model-dir)"))
    pairs = ctx.pulled()
    if not pairs:
        return result("pending", "no parakeet_out shard pulled")
    per, examples, unaligned = {}, [], []
    for src, stem in pairs:
        rows, ids = ctx.pk_rows(src, stem), ctx.ctc_ids(src, stem)
        if [r.get("id") for r in rows] != ctx.pk(src, stem).ids:
            unaligned.append(f"{src}/{stem}")  # K2 reports it; the rows cannot be paired
            continue
        texts = ctx.decode(ids)
        g = per.setdefault(group_of(src, stem), {"rows": 0, "mismatch": 0})
        for r, t in zip(rows, texts):
            g["rows"] += 1
            if t != r.get("ctc_hyp"):
                g["mismatch"] += 1
                if len(examples) < MAX_LISTED:
                    examples.append({"id": r["id"], "stored": r.get("ctc_hyp"), "derived": t})
    n = sum(g["rows"] for g in per.values())
    bad = sum(g["mismatch"] for g in per.values())
    common = dict(rows=n, mismatch=bad, identical_frac=(n - bad) / n if n else None, per_group=per,
                  unaligned_shards=unaligned, examples=examples)
    if bad or unaligned:
        return result("fail", f"{bad} of {n} rows differ from the stored ctc_hyp" +
                      (f"; {len(unaligned)} shard(s) with unaligned jsonl" if unaligned else ""), **common)
    return result("pass", f"identical on all {n} rows", **common)


def _logmel(fe):
    from kitsune.features import LogMel

    return LogMel(fe.mel_filters, n_fft=fe.n_fft, hop_length=fe.hop_length, win_length=fe.win_length,
                  preemphasis=fe.preemphasis, dither=0.0, sampling_rate=fe.sampling_rate, n_mels=fe.feature_size)


def _k4_group(ctx: Ctx, stems: list[tuple[str, str]], logmel, rng) -> dict:
    """Frame parity on up to k4_rows rows of `stems`, drawn from up to k4_units (shard, row group) units."""
    import pyarrow.parquet as pq
    import torch

    from kitsune.audio import decode_audio
    from kitsune.features import pad_waves

    units = []  # (src, stem, row group, [(npz row, row in group)])
    for src, stem in stems:
        shard = ctx.data / "shards" / src / f"{stem}.parquet"
        pf = pq.ParquetFile(shard)
        pos = {rid: j for j, rid in enumerate(ctx.data_ids(src, stem))}
        starts = np.cumsum([0] + [pf.metadata.row_group(k).num_rows for k in range(pf.metadata.num_row_groups)])
        by_group: dict[int, list] = {}
        for i, rid in enumerate(ctx.pk(src, stem).ids):
            j = pos[rid]
            k = int(np.searchsorted(starts, j, side="right") - 1)
            by_group.setdefault(k, []).append((i, j - int(starts[k])))
        units += [(src, stem, k, rows) for k, rows in sorted(by_group.items())]
    chosen = [units[i] for i in sorted(rng.choice(len(units), size=min(ctx.k4_units, len(units)), replace=False))]
    flat = [(u, r) for u in chosen for r in u[3]]
    picks = sorted(rng.choice(len(flat), size=min(ctx.k4_rows, len(flat)), replace=False).tolist())
    want: dict[tuple, list] = {}
    for p in picks:
        u, r = flat[p]
        want.setdefault(u[:3], []).append(r)
    rows, mism, max_diff, max_dur = 0, {"hf": [], "logmel": [], "closed_form": []}, 0.0, 0.0
    for (src, stem, k), rs in want.items():
        pf = pq.ParquetFile(ctx.data / "shards" / src / f"{stem}.parquet")
        table = pf.read_row_group(k, columns=["id", "audio"])
        sh = ctx.pk(src, stem)
        for i, j in rs:
            rid = sh.ids[i]
            assert table.column("id")[j].as_py() == rid, (src, stem, k, j, rid)
            wave = decode_audio(table.column("audio")[j].as_py())
            stored = int(sh.z["n_frames"][i])
            out = ctx.fe([wave], sampling_rate=SAMPLING_RATE, return_tensors="pt")
            n_feat = int(out["attention_mask"].sum())
            got = {"hf": subsampled_length(n_feat, **ctx.sub), "closed_form": expected_n_frames(len(wave), ctx.sub)}
            if logmel is not None:
                w, lens = pad_waves([wave])
                feats, mask = logmel(w, lens)
                got["logmel"] = subsampled_length(int(mask.sum()), **ctx.sub)
                m = int(mask.sum())
                if m and m == n_feat:
                    d = (feats[0, :m] - out["input_features"][0, :m].to(torch.float32)).abs().max().item()
                    max_diff = max(max_diff, d)
            for name, val in got.items():
                if val != stored:
                    mism[name].append({"id": rid, "stored": stored, name: val, "samples": len(wave)})
            max_dur = max(max_dur, abs(len(wave) / SAMPLING_RATE - float(sh.z["duration"][i])))
            rows += 1
    return dict(rows=rows, stems=sorted({f"{u[0]}/{u[1]}" for u in chosen}), units=len(chosen),
                mismatch={name: len(v) for name, v in mism.items()},
                examples={name: v[:5] for name, v in mism.items() if v},
                logmel_vs_hf_max_abs=max_diff if logmel is not None else None, max_abs_duration_diff_s=max_dur)


def k4_frame_parity(ctx: Ctx) -> dict:
    if ctx.fe is None:
        return result("skipped", ctx.unavailable.get("fe", "no Parakeet feature extractor (--model-dir)"))
    if ctx.data is None:
        return result("skipped", "no --data root with the rebuilt audio")
    try:
        logmel, logmel_note = _logmel(ctx.fe), None
    except Exception as e:  # noqa: BLE001 - (b) is a second opinion; (a) still decides
        logmel, logmel_note = None, f"kitsune LogMel unavailable: {type(e).__name__}: {e}"
    groups = {**{s: s for s in ctx.train_sources}, **{eval_group(s): s for s in ctx.eval_sets}}
    per = {}
    for g, src in groups.items():
        listed = {st for st in ctx.labelled("parakeet_out", src) if group_of(src, st) == g}
        pulled = [(s, st) for s, st in ctx.pulled() if s == src and group_of(s, st) == g]
        usable, other, no_audio = [], [], []
        for s, st in pulled:
            ids = ctx.data_ids(s, st)
            if ids is None or not (ctx.data / "shards" / s / f"{st}.parquet").is_file():
                no_audio.append(st)
            elif ids_sha256(ids) != str(ctx.pk(s, st).z["shard_ids_sha256"]):
                other.append(st)
            else:
                usable.append((s, st))
        info = dict(labelled_stems=len(listed), pulled_stems=len(pulled), other_ids=other[:MAX_LISTED],
                    no_audio=no_audio[:MAX_LISTED])
        if not listed:
            per[g] = result("pending", "not labelled yet", **info)
        elif not pulled:
            per[g] = result("pending", "labelled but not pulled", **info)
        elif not usable:
            why = ("the --data root's shards hold other ids than the labelled ones" if other
                   else "the --data root has no audio for the pulled stems")
            per[g] = result("pending", f"{why}: rebuild the labelled stems (01 --extent-config)", **info)
        else:
            rng = np.random.default_rng([ctx.seed, zlib.crc32(g.encode())])
            r = _k4_group(ctx, usable, logmel, rng)
            bad = {k: v for k, v in r["mismatch"].items() if v}
            status = "fail" if bad else "pass" if r["rows"] else "pending"
            per[g] = result(status, f"n_frames differs on {bad}" if bad else f"{r['rows']} rows equal", **info, **r)
    status = worst(p["status"] for p in per.values())
    n = sum(p.get("rows", 0) for p in per.values())
    reason = {"pass": f"stored n_frames = extractor length on all {n} sampled rows",
              "fail": "stored n_frames differ from the extractor length",
              "pending": f"{n} rows checked; " + ", ".join(f"{g}: {p['reason']}" for g, p in per.items()
                                                            if p["status"] == "pending")}.get(status, "")
    return result(status, reason, rows=n, per_group=per, subsampling=ctx.sub, logmel_note=logmel_note)


def k5_coverage(ctx: Ctx) -> dict:
    t = {(s, st) for s in ctx.listed_sources("teacher_out") for st in ctx.labelled("teacher_out", s)}
    p = {(s, st) for s in ctx.listed_sources("parakeet_out") for st in ctx.labelled("parakeet_out", s)}
    if not t and not p:
        return result("pending", "nothing labelled yet")
    only_t, only_p = sorted(t - p), sorted(p - t)
    record = ctx.record()
    rec_stems = {(src, st["stem"]): st for src, s in ((record or {}).get("sources") or {}).items()
                 for inp in s.get("inputs", []) for st in inp.get("stems", [])}
    rows_union = rows_one = 0
    scanned_diff, sha_diff, one_root_examples = [], [], []
    pairs = [(s, st) for s, st in ctx.pulled() if ctx.has("teacher_out", s, st)]
    for s, st in pairs:
        tid, pid = ctx.teacher(s, st), ctx.pk(s, st)
        a, b = set(tid["ids"]), set(pid.ids)
        rows_union += len(a | b)
        rows_one += len(a ^ b)
        if a ^ b and len(one_root_examples) < MAX_LISTED:
            one_root_examples.append({"stem": f"{s}/{st}", "teacher_only": len(a - b), "parakeet_only": len(b - a)})
        if tid["n_scanned"] != int(pid.z["n_scanned"]):
            scanned_diff.append(f"{s}/{st}: teacher {tid['n_scanned']}, parakeet {int(pid.z['n_scanned'])}")
        rs = rec_stems.get((s, st))
        if rs is not None and (rs.get("ids_sha256") != str(pid.z["shard_ids_sha256"])
                               or rs.get("rows") != int(pid.z["n_scanned"])):
            sha_diff.append(f"{s}/{st}")
    stem_rows = None
    if ctx.sealed and record is not None:  # a whole stem in one root: its rows from the record
        stem_rows = sum(int((rec_stems.get(k) or {}).get("rows") or 0) for k in only_t + only_p)
        either = t | p
        total = sum(int(st.get("rows") or 0) for k, st in rec_stems.items() if k in either)
        frac = (rows_one / rows_union if rows_union else 0.0) + (stem_rows / total if total else 1.0)
    else:
        frac = rows_one / rows_union if rows_union else None
    numbers = dict(teacher_stems=len(t), parakeet_stems=len(p), both=len(t & p), teacher_only_stems=len(only_t),
                   parakeet_only_stems=len(only_p), teacher_only_examples=[f"{s}/{st}" for s, st in only_t[:MAX_LISTED]],
                   parakeet_only_examples=[f"{s}/{st}" for s, st in only_p[:MAX_LISTED]],
                   pulled_pairs=len(pairs), pulled_rows=rows_union, pulled_rows_one_root=rows_one,
                   one_root_frac=frac, one_root_examples=one_root_examples, n_scanned_diff=scanned_diff[:MAX_LISTED],
                   record_sha_diff=sha_diff[:MAX_LISTED], rows_of_one_root_stems=stem_rows,
                   record=record is not None)
    if scanned_diff or sha_diff:
        return result("fail", f"{len(scanned_diff)} stem(s) with another n_scanned, {len(sha_diff)} with another "
                              f"ids digest than the extent record", **numbers)
    if frac is not None and frac >= ONE_ROOT_MAX:
        return result("fail", f"{100 * frac:.3f} % of rows in one root only (limit {100 * ONE_ROOT_MAX:.1f} %)",
                      **numbers)
    if ctx.sealed and record is None:
        return result("fail", "sealed root without extent.json: the stems cannot be tied to the ingest", **numbers)
    if only_t or only_p:
        if ctx.sealed:
            return result("pass", f"{len(only_t) + len(only_p)} stem(s) in one root only, "
                                  f"{100 * frac:.3f} % of rows", **numbers)
        return result("pending", f"root not sealed: {len(only_t)} teacher-only and {len(only_p)} parakeet-only "
                                 f"stems still in progress; pulled pairs: {rows_one} of {rows_union} rows in one root",
                      **numbers)
    if not pairs:
        return result("pending", "stems cover each other, but no pair pulled to compare rows", **numbers)
    if not ctx.sealed:
        return result("pending", f"stems cover each other so far and {rows_one} of {rows_union} pulled rows are in "
                                 f"one root only; the root is not sealed", **numbers)
    return result("pass", f"stems cover each other; {rows_one} of {rows_union} pulled rows in one root", **numbers)


def _baselines(rows: list[dict]) -> dict:
    refs = [r.get("ref") or "" for r in rows]
    tdt = corpus_cer([r.get("hyp") or "" for r in rows], refs)
    ctc = corpus_cer([r.get("ctc_hyp") or "" for r in rows], refs)
    return {"n": len(rows), "tdt_cer_corpus": tdt["cer"], "ctc_cer_corpus": ctc["cer"], "n_empty_ref": tdt["n_empty_ref"]}


def k6_eval_sets(ctx: Ctx) -> dict:
    per = {}
    for name in ctx.eval_sets:
        g = eval_group(name)
        t = {st for st in ctx.labelled("teacher_out", name) if group_of(name, st) == g}
        p = {st for st in ctx.labelled("parakeet_out", name) if group_of(name, st) == g}
        info = dict(teacher_stems=sorted(t), parakeet_stems=sorted(p))
        if not t or not p or t != p:
            why = (f"stems differ between the roots (teacher {sorted(t)}, parakeet {sorted(p)})" if t and p
                   else f"not labelled yet by {'both passes' if not t and not p else 'Parakeet' if not p else 'Cohere'}")
            per[g] = result("fail" if ctx.sealed else "pending", why, **info)
            continue
        if not all(ctx.has(r, name, st) for r in ("teacher_out", "parakeet_out") for st in t):
            per[g] = result("pending", "labelled but not pulled", **info)
            continue
        problems, rows = [], []
        for st in sorted(t):
            tid, sh = ctx.teacher(name, st), ctx.pk(name, st)
            if tid["ids"] != sh.ids:
                problems.append(f"{st}: teacher and parakeet ids differ ({len(tid['ids'])} vs {sh.n} rows, "
                                f"{len(set(tid['ids']) ^ set(sh.ids))} in one root only)")
            if tid["n_scanned"] != int(sh.z["n_scanned"]):
                problems.append(f"{st}: n_scanned teacher {tid['n_scanned']}, parakeet {int(sh.z['n_scanned'])}")
            rows += ctx.pk_rows(name, st)
        n_t = sum(len(ctx.teacher(name, st)["ids"]) for st in t)
        want = ctx.eval_rows.get(name)
        if want is not None and (len(rows) != want or n_t != want):
            problems.append(f"{len(rows)} parakeet and {n_t} teacher rows, the manifest has {want}")
        base = _baselines(rows)
        card = (ctx.card or {}).get(name)
        if card:
            base["card"] = card
            off = {k: base[f"{k}_cer_corpus"] - v for k, v in card.items()}
            base["card_diff_pp"] = {k: round(100 * v, 3) for k, v in off.items()}
            if any(abs(v) > CARD_TOL for v in off.values()):
                problems.append(f"Parakeet corpus CER TDT {100 * base['tdt_cer_corpus']:.2f} % / CTC "
                                f"{100 * base['ctc_cer_corpus']:.2f} % is not within {100 * CARD_TOL:.1f} pp of the "
                                f"card's {100 * card['tdt']:.1f} / {100 * card['ctc']:.1f} %")
        per[g] = result("fail" if problems else "pass", "; ".join(problems) or f"{len(rows)} rows in both roots",
                        rows=len(rows), baselines=base, **info)
    report = read_json(ctx.labels / "reports" / "parakeet_baselines.json")
    report_diff = []
    if report is not None:
        for name, b in (report.get("parakeet") or {}).items():
            mine = (per.get(eval_group(name)) or {}).get("baselines")
            if mine is None:
                continue
            for k in ("tdt_cer_corpus", "ctc_cer_corpus"):
                if abs(float(b.get(k, float("nan"))) - mine[k]) > 1e-9:
                    report_diff.append(f"{name}.{k}: report {b.get(k)}, recomputed {mine[k]}")
    status = worst([p["status"] for p in per.values()] + (["fail"] if report_diff else []))
    note = ("reports/parakeet_baselines.json equals the recomputation" if report is not None and not report_diff
            else "reports/parakeet_baselines.json differs from the recomputation" if report_diff
            else "reports/parakeet_baselines.json not written yet (the box writes it at finalize, F3); recomputed here")
    reason = "; ".join(f"{g}: {p['reason']}" for g, p in per.items() if p["status"] != "pass") or "every eval set whole"
    return result(status, reason, per_set=per, baselines_report=note, baselines_report_diff=report_diff)


def _train_groups(ctx: Ctx) -> set[str]:
    return set(ctx.train_sources)


def k7_feasibility(ctx: Ctx) -> dict:
    pairs = ctx.pulled()
    if not pairs:
        return result("pending", "no parakeet_out shard pulled")
    per = {}
    for src, stem in pairs:
        sh = ctx.pk(src, stem)
        z, g = sh.z, per.setdefault(group_of(src, stem), {"rows": 0, "greedy_ctc": 0, "tdt": 0})
        to = z["tok_offsets"]
        for i, ids in enumerate(ctx.ctc_ids(src, stem)):
            nf = int(z["n_frames"][i])
            g["rows"] += 1
            g["greedy_ctc"] += ctc_feasible_need(ids) > nf
            g["tdt"] += ctc_feasible_need(z["tokens"][to[i]:to[i + 1]]) > nf
    train = [v for k, v in per.items() if k in _train_groups(ctx)]
    n = sum(v["rows"] for v in train)
    for v in per.values():
        v["greedy_ctc_rate"], v["tdt_rate"] = v["greedy_ctc"] / v["rows"], v["tdt"] / v["rows"]
    if not n:
        return result("pending", "no train shard pulled", per_group=per)
    rates = {k: sum(v[k] for v in train) / n for k in ("greedy_ctc", "tdt")}
    limit = ABOUT * ctx.expect["k7_infeasible"]
    rate = rates[ctx.ctc_target if ctx.ctc_target == "tdt" else "greedy_ctc"]
    numbers = dict(target=ctx.ctc_target, train_rows=n, train_rate=rates, limit=limit,
                   expected=ctx.expect["k7_infeasible"], per_group=per)
    if rate > limit:
        return result("fail", f"{100 * rate:.3f} % of train rows cannot align their {ctx.ctc_target} target "
                              f"(design: about {100 * ctx.expect['k7_infeasible']:.1f} %)", **numbers)
    return result("pass", f"{ctx.ctc_target} target infeasible on {100 * rate:.3f} % of train rows (TDT tokens: "
                          f"{100 * rates['tdt']:.3f} %)", **numbers)


def k8_hyp_vs_ctc(ctx: Ctx) -> dict:
    pairs = ctx.pulled()
    if not pairs:
        return result("pending", "no parakeet_out shard pulled")
    by: dict[str, list[dict]] = {}
    for src, stem in pairs:
        by.setdefault(group_of(src, stem), []).extend(ctx.pk_rows(src, stem))

    def stats(rows):
        hyp = [r.get("hyp") or "" for r in rows]
        ctc = [r.get("ctc_hyp") or "" for r in rows]
        c = corpus_cer(ctc, hyp)  # ctc_hyp scored against the TDT hypothesis
        return {"rows": len(rows), "raw_diff_rate": float(np.mean([h != x for h, x in zip(hyp, ctc)])),
                "norm_diff_rate": float(np.mean([normalize_ja(h) != normalize_ja(x) for h, x in zip(hyp, ctc)])),
                "cer_ctc_vs_tdt": c["cer"], "edits": c["edits"], "chars": c["ref_chars"],
                "n_empty_tdt": c["n_empty_ref"], **{k: v for k, v in _baselines(rows).items() if k != "n"}}

    per = {g: stats(rows) for g, rows in by.items()}
    train = [r for g, rows in by.items() if g in _train_groups(ctx) for r in rows]
    if not train:
        return result("pending", "no train shard pulled", per_group=per)
    pooled = stats(train)
    limit = ABOUT * ctx.expect["k8_cer"]
    numbers = dict(train=pooled, limit=limit, expected=ctx.expect["k8_cer"], per_group=per)
    msg = (f"ctc_hyp differs from hyp on {100 * pooled['raw_diff_rate']:.1f} % of train rows "
           f"({100 * pooled['norm_diff_rate']:.1f} % after normalisation); CER(ctc_hyp vs hyp) "
           f"{100 * pooled['cer_ctc_vs_tdt']:.2f} %; against the reference TDT {100 * pooled['tdt_cer_corpus']:.2f} %, "
           f"CTC {100 * pooled['ctc_cer_corpus']:.2f} %")
    if pooled["cer_ctc_vs_tdt"] > limit:
        return result("fail", f"{msg}; above {100 * limit:.1f} % (design: about {100 * ctx.expect['k8_cer']:.0f} %): "
                              f"a statistic for decision 22, not a label defect", **numbers)
    return result("pass", msg, **numbers)


def k9_bytes(ctx: Ctx) -> dict:
    pairs = ctx.pulled()
    if not pairs:
        return result("pending", "no parakeet_out shard pulled")
    per = {}
    for src, stem in pairs:
        g = per.setdefault(group_of(src, stem), {"stems": 0, "npz_bytes": 0, "jsonl_bytes": 0, "audio_h": 0.0,
                                                 "teacher_bytes": 0, "teacher_audio_h": 0.0})
        g["stems"] += 1
        g["npz_bytes"] += ctx.path("parakeet_out", src, stem, "npz").stat().st_size
        g["jsonl_bytes"] += ctx.path("parakeet_out", src, stem, "jsonl").stat().st_size
        h = float(np.sum(ctx.pk(src, stem).z["duration"], dtype=np.float64)) / 3600
        g["audio_h"] += h
        if ctx.has("teacher_out", src, stem):
            g["teacher_bytes"] += sum(ctx.path("teacher_out", src, stem, s).stat().st_size for s in ("npz", "jsonl"))
            g["teacher_audio_h"] += h
    for g in per.values():
        g["mb_per_audio_h"] = (g["npz_bytes"] + g["jsonl_bytes"]) / 1e6 / g["audio_h"] if g["audio_h"] else None
        g["teacher_mb_per_audio_h"] = g["teacher_bytes"] / 1e6 / g["teacher_audio_h"] if g["teacher_audio_h"] else None
    train = [v for k, v in per.items() if k in _train_groups(ctx)]
    listed = sum(int(v or 0) for p, v in ctx.listing.items() if p.startswith("parakeet_out/"))
    hours = sum(v["audio_h"] for v in train)
    if not hours:
        return result("pending", "no train shard pulled", per_group=per, listed_parakeet_bytes=listed)
    mb = sum(v["npz_bytes"] + v["jsonl_bytes"] for v in train) / 1e6 / hours
    limit = ABOUT * ctx.expect["k9_mb_per_audio_h"]
    numbers = dict(train_mb_per_audio_h=mb, train_audio_h=hours, gb_per_1000h=mb, limit=limit,
                   expected=ctx.expect["k9_mb_per_audio_h"], listed_parakeet_bytes=listed, per_group=per)
    msg = f"{mb:.3f} MB per audio-hour on {hours:.1f} h of pulled train stems ({mb:.2f} GB per 1,000 h)"
    if mb > limit:
        return result("fail", f"{msg}, above {limit:.2f} (design: about {ctx.expect['k9_mb_per_audio_h']})", **numbers)
    return result("pass", msg, **numbers)


def k10_galgame_ids(ctx: Ctx) -> dict:
    if ctx.kotoba is None or not Path(ctx.kotoba).is_file():
        return result("skipped", f"no laptop kotoba file (--kotoba {ctx.kotoba})")
    kot = jsonl_rows(Path(ctx.kotoba))
    kids = [r["id"] for r in kot]
    neutral = sum(1 for r in kot if r.get("cer2") is not None and float(r["cer2"]) <= NEUTRAL_MAX)
    numbers = dict(kotoba_rows=len(kids), kotoba_ids_sha256=ids_sha256(kids), neutral_view_rows=neutral)
    have = {r: ctx.has(r, "galgame", KOTOBA_STEM) for r in ("teacher_out", "parakeet_out")}
    if not all(KOTOBA_STEM in ctx.labelled(r, "galgame") for r in have):
        return result("pending", f"galgame/{KOTOBA_STEM} not labelled by both passes yet", **numbers)
    if not all(have.values()):
        return result("pending", f"galgame/{KOTOBA_STEM} labelled but not pulled", **numbers)
    ids = {"teacher_out": ctx.teacher("galgame", KOTOBA_STEM)["ids"], "parakeet_out": ctx.pk("galgame", KOTOBA_STEM).ids}
    if ctx.has("second_out", "galgame", KOTOBA_STEM):
        ids["second_out"] = [r["id"] for r in jsonl_rows(ctx.path("second_out", "galgame", KOTOBA_STEM, "jsonl"))]
    per = {r: {"rows": len(v), "ids_sha256": ids_sha256(v), "equal": v == kids,
               "missing_vs_kotoba": len(set(kids) - set(v)), "extra_vs_kotoba": len(set(v) - set(kids))}
           for r, v in ids.items()}
    bad = [r for r in ("teacher_out", "parakeet_out") if not per[r]["equal"]]
    if bad:
        return result("fail", f"{', '.join(bad)} galgame/{KOTOBA_STEM} ids differ from the laptop's kotoba file: the "
                              f"neutral view needs a kotoba decode of these rows", per_root=per, **numbers)
    return result("pass", f"the {len(kids)} ids are equal in order; the neutral view keeps {neutral} rows",
                  per_root=per, **numbers)


def k11_cohere_baselines(ctx: Ctx) -> dict:
    sets = list(ctx.prereg)
    missing = [s for s in sets if not ctx.labelled("teacher_out", s)]
    if missing:
        return result("pending", f"not labelled yet: {missing}")
    unpulled = [s for s in sets if not all(ctx.has("teacher_out", s, st) for st in ctx.labelled("teacher_out", s))]
    if unpulled:
        return result("pending", f"labelled but not (wholly) pulled: {unpulled}")
    base = teacher_baselines(ctx.labels / "teacher_out", sets, check=False)
    per = {s: {"cer_corpus_pct": 100 * b["cer_corpus"], "prereg_pct": 100 * ctx.prereg[s],
               "diff_pp": 100 * (b["cer_corpus"] - ctx.prereg[s]), "n": b["n"], "n_empty_ref": b["n_empty_ref"],
               "trunc_rate": b["trunc_rate"], "hours": b["hours"]} for s, b in base.items()}
    meta = read_json(ctx.labels / "teacher_out" / "meta.json") or {}
    numbers = dict(per_set=per, tol_pp=100 * BASELINE_TOL,
                   teacher_meta={k: meta.get(k) for k in ("model", "model_revision", "adopted_from")})
    off = [s for s, v in per.items() if abs(v["diff_pp"]) > 100 * BASELINE_TOL + 1e-9]
    text = " / ".join(f"{v['cer_corpus_pct']:.2f}" for v in per.values())
    if off:
        return result("fail", f"{off} drift more than {100 * BASELINE_TOL:.2f} pp from the pre-registered baselines "
                              f"({text} %): re-register or reuse the laptop's eval labels before the PREREG", **numbers)
    return result("pass", f"{text} % reproduce the pre-registered baselines", **numbers)


def _progress(ctx: Ctx) -> dict:
    out = {}
    for src in dict.fromkeys([*ctx.train_sources, *ctx.eval_sets]):
        t, p = ctx.labelled("teacher_out", src), ctx.labelled("parakeet_out", src)
        out[src] = {"teacher_stems": len(t), "parakeet_stems": len(p), "both": len(t & p),
                    "second_stems": len(ctx.labelled("second_out", src))}
    return out


def k12_extent(ctx: Ctx) -> dict:
    progress = _progress(ctx)
    if not ctx.sealed:
        return result("pending", "root not sealed (no COMPLETE.json): re-run on the sealed root", progress=progress,
                      required=ctx.mix, margin=ctx.margin)
    record = ctx.record()
    if record is None:
        return result("fail", f"sealed root without {ext.RECORD_FILE}", progress=progress)
    cfg = {"extent": {"name": record.get("name"), "root": record.get("root"), "inputs": dict(ctx.mix)},
           "sources": list(ctx.train_sources), "eval_sets": list(ctx.eval_sets)}
    need = ext.subset_stems(record, cfg)
    both = {src: ctx.labelled("teacher_out", src) & ctx.labelled("parakeet_out", src) for src in need}
    missing = {src: sorted(stems - both[src]) for src, stems in need.items() if stems - both[src]}
    empty = [src for src, stems in need.items() if not stems]
    prefix = {}
    for src, s in (record.get("sources") or {}).items():
        n = 0
        labelled = both.get(src) or (ctx.labelled("teacher_out", src) & ctx.labelled("parakeet_out", src))
        for inp in sorted(s.get("inputs", []), key=lambda i: i["ordinal"]):
            if not all(st["stem"] in labelled for st in inp.get("stems", [])):
                break
            n += 1
        prefix[src] = {"labelled_inputs": n, "inputs": len(s.get("inputs", []))}
    # subset_stems takes what the record has: a source ingested or labelled short of its cap gives fewer stems, not an
    # error, so the count of leading labelled inputs is required separately
    few = {src: {"labelled_inputs": prefix.get(src, {}).get("labelled_inputs", 0), "required": n}
           for src, n in ctx.mix.items() if isinstance(n, int) and not isinstance(n, bool)
           and prefix.get(src, {}).get("labelled_inputs", 0) < n}
    short = {src: {"labelled_inputs": prefix.get(src, {}).get("labelled_inputs", 0), "margin": m}
             for src, m in ctx.margin.items() if prefix.get(src, {}).get("labelled_inputs", 0) < m}
    numbers = dict(required=ctx.mix, margin=ctx.margin, required_stems={s: len(v) for s, v in need.items()},
                   missing={s: v[:MAX_LISTED] for s, v in missing.items()},
                   n_missing={s: len(v) for s, v in missing.items()}, empty=empty, too_few_inputs=few,
                   labelled_prefix=prefix, below_margin=short, progress=progress)
    if missing or empty or few:
        return result("fail", f"the study mix is not covered: {sum(len(v) for v in missing.values())} required "
                              f"stems unlabelled, {empty} without stems in the record, too few labelled inputs "
                              f"{few}", **numbers)
    return result("pass", "the study mix is covered" + (f"; below the margin: {short}" if short else ""), **numbers)


CHECKS = {"K1": k1_meta, "K2": k2_format, "K3": k3_ctc_target, "K4": k4_frame_parity, "K5": k5_coverage,
          "K6": k6_eval_sets, "K7": k7_feasibility, "K8": k8_hyp_vs_ctc, "K9": k9_bytes, "K10": k10_galgame_ids,
          "K11": k11_cohere_baselines, "K12": k12_extent}


def run_checks(ctx: Ctx, only=None, catch: bool = True) -> dict:
    """{K: result}. With catch, a check that raises is a failed check carrying the error (never a silent pending)."""
    out = {}
    for name, fn in CHECKS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            res = fn(ctx)
        except Exception as e:  # noqa: BLE001
            if not catch:
                raise
            res = result("fail", f"the check crashed: {type(e).__name__}: {e}")
        res["seconds"] = round(time.time() - t0, 2)
        out[name] = res
    return out


def build_report(ctx: Ctx, results: dict, **extra) -> dict:
    summary = {s: [k for k, v in results.items() if v["status"] == s] for s in ("fail", "pending", "skipped", "pass")}
    hub = {k: ctx.pull.get(k) for k in ("repo", "root", "revision", "pulled_utc", "bytes")} if ctx.pull else None
    return {"tool": "tools/label_checks.py", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kitsune_sha": kitsune_sha(), "labels": str(ctx.labels), "hub": hub, "sealed": ctx.sealed,
            "data": str(ctx.data) if ctx.data else None, "kotoba": str(ctx.kotoba) if ctx.kotoba else None,
            "unavailable": ctx.unavailable, **extra, "summary": summary, "checks": results}


# -------------------------------------------------------------------------------------------------------- pull


def select_pull(listing: dict, *, train_sources=TRAIN_SOURCES, eval_sets=EVAL_SETS,
                train_stems: int = PULL_TRAIN_STEMS) -> list[str]:
    """The root-relative files `run` needs: meta.json of each label root, the root-level json files and reports/,
    every file of the eval sets, and `train_stems` train stems per source (spread over the stems both passes have)."""
    idx = index_listing(listing)
    want = [p for p in listing if (p.count("/") == 1 and p.split("/")[0] in ROOTS and p.endswith("/meta.json"))
            or ("/" not in p and p.endswith(".json")) or p.startswith("reports/")]
    for name in eval_sets:
        want += [p for p in listing if len(parts := p.split("/")) == 3 and parts[0] in ROOTS and parts[1] == name
                 and is_eval_stem(name, parts[2])]
    for src in train_sources:
        common = sorted(st for st in (idx.get(("teacher_out", src, "npz"), set()) & idx.get(("teacher_out", src, "jsonl"), set())
                                      & idx.get(("parakeet_out", src, "npz"), set()) & idx.get(("parakeet_out", src, "jsonl"), set()))
                        if not is_eval_stem(src, st))
        for st in spread(common, train_stems):
            want += [p for p in (f"teacher_out/{src}/{st}.npz", f"teacher_out/{src}/{st}.jsonl",
                                 f"parakeet_out/{src}/{st}.npz", f"parakeet_out/{src}/{st}.jsonl",
                                 f"second_out/{src}/{st}.jsonl") if p in listing]
    return sorted(set(want))


def pull(out: Path, *, repo: str = DEFAULT_REPO, root: str = DEFAULT_ROOT, revision: str | None = None,
         train_stems: int = PULL_TRAIN_STEMS, max_gb: float = PULL_MAX_GB) -> dict:
    """Copy select_pull's files of <repo>/<root> at one commit into `out` (read-only on the Hub) and write
    <out>/_pull.json with the commit, the root's whole listing and the pulled files. Label files are write-once, so a
    file an earlier pull left in `out` is still valid."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    rev = revision or api.dataset_info(repo).sha  # one commit: the listing and every file come from it
    listing = {f.path[len(root) + 1:]: int(f.size) for f in api.list_repo_tree(
        repo, repo_type="dataset", path_in_repo=root, recursive=True, revision=rev) if isinstance(f, RepoFile)}
    want = select_pull(listing, train_stems=train_stems)
    total = sum(listing[p] for p in want)
    if total > max_gb * 1e9:
        raise SystemExit(f"the pull would take {total / 1e9:.2f} GB > {max_gb} GB; lower --train-stems")
    out = Path(out)
    staging = out.parent / f".{out.name}.pull-{rev[:12]}"
    t0 = time.time()
    snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=[f"{root}/{p}" for p in want],
                      local_dir=staging, max_workers=8)
    for p in want:
        dst = out / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / root / p, dst)
    shutil.rmtree(staging, ignore_errors=True)
    rec = {"repo": repo, "root": root, "revision": rev, "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seconds": round(time.time() - t0, 1), "bytes": total, "pulled": want, "listing": listing}
    (out / PULL_FILE).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ---------------------------------------------------------------------------------------------------------- CLI


def model_tools(model_dir: Path | None) -> tuple[Callable | None, object | None, dict, dict]:
    """(decode, feature extractor, subsampling, unavailable reasons) from the pinned converted Parakeet dir."""
    if model_dir is None or not Path(model_dir).is_dir():
        why = f"no Parakeet model dir ({model_dir}); pass --model-dir"
        return None, None, dict(SUBSAMPLING), {"decode": why, "fe": why}
    from kitsune.parakeet import verify_model_dir

    problems = verify_model_dir(Path(model_dir))
    if problems:
        why = f"{model_dir} is not the pinned Parakeet model: {problems}"
        return None, None, dict(SUBSAMPLING), {"decode": why, "fe": why}
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)

    def decode(seqs):  # exactly as scripts/02p_parakeet_pass.py decodes ctc_tokens
        return proc.batch_decode([[int(x) for x in s] for s in seqs], skip_special_tokens=True)

    return decode, proc.feature_extractor, model_subsampling(Path(model_dir)), {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull", help="copy the files `run` needs from the data repo (read-only)")
    p.add_argument("--out", required=True, help="local label root, e.g. D:/kitsune-labels/full")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--revision", default=None, help="commit (default: the repo's current head, recorded)")
    p.add_argument("--train-stems", type=int, default=PULL_TRAIN_STEMS)
    p.add_argument("--max-gb", type=float, default=PULL_MAX_GB)
    r = sub.add_parser("run", help="run K1-K12 on a local label root and write the report")
    r.add_argument("--labels", required=True, help="local label root (teacher_out/, parakeet_out/, second_out/ ...)")
    r.add_argument("--data", default=None, help="a data root with manifest shards (audio) for K4 and ids for K2")
    r.add_argument("--model-dir", default=str(ROOT / PARAKEET_PATH), help="the pinned converted Parakeet dir")
    r.add_argument("--kotoba", default=str(ROOT / "second_out" / "galgame" / f"{KOTOBA_STEM}.jsonl"),
                   help="the laptop's kotoba second opinions of galgame eval-00000 (K10)")
    r.add_argument("--out", required=True, help="the JSON report")
    r.add_argument("--checks", nargs="*", default=None, help="a subset, e.g. K1 K3")
    r.add_argument("--k4-rows", type=int, default=K4_ROWS)
    r.add_argument("--k4-units", type=int, default=K4_UNITS)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ctc-target", choices=["greedy_ctc", "tdt"], default="greedy_ctc",
                   help="the CTC target the study uses (decision 22), for K7")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "pull":
        rec = pull(Path(args.out), repo=args.repo, root=args.root, revision=args.revision,
                   train_stems=args.train_stems, max_gb=args.max_gb)
        print(f"pulled {len(rec['pulled'])} files, {rec['bytes'] / 1e6:.1f} MB, {args.repo}@{rec['revision']} "
              f"({len(rec['listing'])} files listed) into {args.out}", flush=True)
        return 0
    labels = Path(args.labels)
    pull_rec = read_json(labels / PULL_FILE)
    listing = pull_rec["listing"] if pull_rec else local_listing(labels)
    decode, fe, subs, unavailable = model_tools(Path(args.model_dir) if args.model_dir else None)
    ctx = Ctx(labels, listing, pull=pull_rec, data=Path(args.data) if args.data else None, decode=decode, fe=fe,
              sub=subs, kotoba=Path(args.kotoba) if args.kotoba else None, k4_rows=args.k4_rows,
              k4_units=args.k4_units, seed=args.seed, ctc_target=args.ctc_target, unavailable=unavailable)
    results = run_checks(ctx, only=set(args.checks) if args.checks else None)
    report = build_report(ctx, results, model_dir=args.model_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    for k, v in results.items():
        print(f"{k:4s} {v['status']:8s} {v['reason']}", flush=True)
    print(f"report: {args.out}", flush=True)
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
