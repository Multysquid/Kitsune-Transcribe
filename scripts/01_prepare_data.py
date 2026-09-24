"""Download the training/eval sources into the local shard store (data/).

Sources (all streamed file-by-file from HF; each raw download is deleted after conversion):
  reazon_small  japanese-asr/whisper_transcriptions.reazonspeech.small  ~100 h TV speech, ungated parquet mirror
  galgame       litagin/Galgame_Speech_ASR_16kHz                         visual-novel voices, --galgame-shards tars (~47 h each);
                the first GALGAME_EVAL_ROWS utterances are held out as an in-domain eval split
  cv            Common Voice ja  -- NOT on HF anymore. Download cv-corpus-*-ja.tar.gz from
                https://datacollective.mozillafoundation.org and extract to data/raw/common_voice/ ; this script ingests it.
  eval          japanese-asr/ja_asr.jsut_basic5000, ja_asr.common_voice_8_0, ja_asr.reazonspeech_test  (eval_jsut / eval_cv8 / eval_reazon)

Resumable: every flushed shard is added to the manifest immediately and shards/<source>/progress.json records the
input files that are complete, so an interrupted run continues where it stopped. A source is considered done only
once progress.json says so. --force wipes a source (shards + manifest lines) and re-ingests it.

Usage:
  python scripts/01_prepare_data.py --sources reazon_small galgame eval --galgame-shards 6
  python scripts/01_prepare_data.py --sources eval_cv8 --limit-rows 64      # smoke run -> writes to data_smoke/
"""
import argparse
import csv
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kitsune.audio import audio_info  # noqa: E402
from kitsune.store import (  # noqa: E402
    ShardWriter, iter_rows, load_progress, read_manifest, remove_source, save_progress,
)

MIN_DUR, MAX_DUR = 0.3, 30.0  # teacher fast path is <=30 s; shorter than 0.3 s is noise

HF_PARQUET_SOURCES = {
    # name: (repo, split)
    "reazon_small": ("japanese-asr/whisper_transcriptions.reazonspeech.small", "train"),
    "reazon_medium": ("japanese-asr/whisper_transcriptions.reazonspeech.medium", "train"),
    "eval_jsut": ("japanese-asr/ja_asr.jsut_basic5000", "eval"),
    "eval_cv8": ("japanese-asr/ja_asr.common_voice_8_0", "eval"),
    "eval_reazon": ("japanese-asr/ja_asr.reazonspeech_test", "eval"),
}
GALGAME_REPO = "litagin/Galgame_Speech_ASR_16kHz"
GALGAME_EVAL_ROWS = 1000  # first N kept utterances become the in-domain eval split
ALL_SOURCES = ["reazon_small", "reazon_medium", "galgame", "cv", "eval", "eval_jsut", "eval_cv8", "eval_reazon"]
# ReazonSpeech tiers are nested (small is a subset of medium), so a larger tier must skip rows already ingested
DEDUP_AGAINST = {"reazon_medium": "reazon_small"}


class Ingest:
    """Shared bookkeeping for one source: writers, stats, progress, download cleanup."""

    def __init__(self, root: Path, raw: Path, source: str, limit_rows: int | None):
        self.root, self.raw, self.source, self.limit = root, raw, source, limit_rows
        self.progress = load_progress(root, source)
        self.writers: dict[str, ShardWriter] = {}
        self.stats = dict(kept=0, empty_text=0, bad_audio=0, bad_duration=0, dup=0, seconds=0.0)
        self.counted = 0  # rows that count against --limit-rows

    def writer(self, split: str) -> ShardWriter:
        if split not in self.writers:
            self.writers[split] = ShardWriter(self.root, self.source, split)
        return self.writers[split]

    def add(self, split: str, id: str, audio: bytes, text: str, count: bool = True) -> bool:
        """Filter + write one utterance. Returns True if kept. `count=False` exempts it from --limit-rows."""
        text = text.strip()
        if not text:
            self.stats["empty_text"] += 1
            return False
        try:
            dur, sr = audio_info(audio)
        except Exception:
            self.stats["bad_audio"] += 1
            return False
        if not (MIN_DUR <= dur <= MAX_DUR):
            self.stats["bad_duration"] += 1
            return False
        self.writer(split).add(id, audio, text, dur, sr)
        self.stats["kept"] += 1
        self.stats["seconds"] += dur
        if count:
            self.counted += 1
        return True

    def limit_hit(self) -> bool:
        return bool(self.limit) and self.counted >= self.limit

    def is_finished(self, input_name: str) -> bool:
        return input_name in self.progress["finished_inputs"]

    def finish_input(self, input_name: str):
        # flush first so the rows of this input are on disk (and in the manifest) before it is marked complete
        for w in self.writers.values():
            w.flush()
        self.progress["finished_inputs"].append(input_name)
        save_progress(self.root, self.source, self.progress)

    def download(self, repo: str, filename: str) -> Path:
        return Path(hf_hub_download(repo, filename, repo_type="dataset", cache_dir=self.raw))

    @staticmethod
    def free_download(local: Path):
        """Delete a finished hf_hub_download: the snapshot entry AND its blob (the snapshot is a symlink or a copy
        depending on Windows privileges; unlinking only it leaves the multi-GB blob behind)."""
        real = local.resolve()
        local.unlink(missing_ok=True)
        if real != local:
            real.unlink(missing_ok=True)
        for parent in local.parents:
            if parent.name.startswith("datasets--"):
                for blob in (parent / "blobs").glob("*"):
                    if not blob.name.endswith(".incomplete"):
                        blob.unlink(missing_ok=True)
                break

    def done(self):
        for w in self.writers.values():
            w.close()
        self.progress["done"] = True
        save_progress(self.root, self.source, self.progress)
        s = self.stats
        print(f"  {self.source}: kept={s['kept']} ({s['seconds'] / 3600:.1f} h)  dropped: empty_text={s['empty_text']} "
              f"bad_audio={s['bad_audio']} bad_duration={s['bad_duration']} dup={s['dup']}")


def ingest_hf_parquet(ing: Ingest, repo: str, split: str):
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    columns = ["audio", "transcription"] + (["name"] if ing.source.startswith("reazon") else [])
    # dedup against a nested smaller tier AND against this source's own manifest shards (a crash mid input
    # file re-reads rows already flushed; only manifest-listed shards count - orphans are re-ingested).
    seen: set[str] = set()
    dedup_sources = {ing.source} | ({DEDUP_AGAINST[ing.source]} if ing.source in DEDUP_AGAINST else set())
    for sh in read_manifest(ing.root):
        if sh.source in dedup_sources:
            for row in iter_rows(ing.root / sh.path, columns=["id"]):
                seen.add(row["id"].split("/", 1)[1])
    if seen:
        print(f"  {ing.source}: deduplicating against {len(seen)} rows already in {sorted(dedup_sources)}")
    for f in tqdm(files, desc=ing.source, unit="file"):
        if ing.is_finished(f):
            continue
        local = ing.download(repo, f)
        for i, row in enumerate(iter_rows(local, columns=columns)):
            rid = row.get("name") or row["audio"].get("path") or f"{Path(f).stem}-{i}"
            if rid in seen:
                ing.stats["dup"] += 1
                continue
            ing.add(split, f"{ing.source}/{rid}", row["audio"]["bytes"], row["transcription"])
            if ing.limit_hit():
                break
        ing.finish_input(f)
        ing.free_download(local)
        if ing.limit_hit():
            break
    ing.done()


def ingest_galgame(ing: Ingest, n_shards: int):
    tars = sorted(f for f in HfApi().list_repo_files(GALGAME_REPO, repo_type="dataset") if f.endswith(".tar"))[:n_shards]
    # self-dedup: resuming an interrupted tar re-reads rows already stored; skip everything already ingested.
    # Gate on the manifest, not finished_inputs - a crash inside the FIRST tar also leaves flushed shards behind.
    seen: set[str] = set()
    eval_kept = 0
    for sh in read_manifest(ing.root):
        if sh.source == "galgame":
            if sh.split == "eval":
                eval_kept += sh.rows
            for row in iter_rows(ing.root / sh.path, columns=["id"]):
                seen.add(row["id"])
    if seen:
        print(f"  galgame: {len(seen)} rows already ingested (duplicates will be skipped), eval hold-out {eval_kept}")
    pending: dict[str, dict] = {}  # webdataset pairs <key>.ogg / <key>.txt can arrive in either order
    for f in tqdm(tars, desc="galgame", unit="tar"):
        if ing.is_finished(f):
            continue
        local = ing.download(GALGAME_REPO, f)
        with tarfile.open(local, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit(".", 1)
                d = pending.setdefault(key, {})
                d[ext] = tf.extractfile(m).read()
                if "ogg" in d and "txt" in d:
                    del pending[key]
                    txt = d["txt"].decode("utf-8", "replace")
                    if f"galgame/{key}" in seen:
                        ing.stats["dup"] += 1
                        continue
                    if eval_kept < GALGAME_EVAL_ROWS:  # hold-out first; it does not count against --limit-rows
                        eval_kept += ing.add("eval", f"galgame/{key}", d["ogg"], txt, count=False)
                    else:
                        ing.add("train", f"galgame/{key}", d["ogg"], txt)
                    if ing.limit_hit():
                        break
        ing.finish_input(f)
        ing.free_download(local)
        if ing.limit_hit():
            break
    if pending:
        print(f"  galgame: {len(pending)} unpaired members dropped")
    ing.done()


def ingest_common_voice(ing: Ingest):
    """Ingest a manually downloaded Common Voice ja corpus from data/raw/common_voice/<cv-corpus-*>/ja/."""
    cands = list((ing.raw / "common_voice").glob("**/ja/clips"))
    if not cands:
        print("  cv: nothing found under data/raw/common_voice/**/ja/clips - download cv-corpus-*-ja.tar.gz from "
              "https://datacollective.mozillafoundation.org, extract it there, and re-run with --sources cv")
        return
    ja = cands[0].parent
    for tsv, split in [("train.tsv", "train"), ("dev.tsv", "train"), ("test.tsv", "eval")]:
        p = ja / tsv
        if not p.exists() or ing.is_finished(tsv):
            continue
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        for r in tqdm(rows, desc=f"cv/{tsv}", unit="clip"):
            clip = ja / "clips" / r["path"]
            if not clip.exists():
                ing.stats["bad_audio"] += 1
                continue
            ing.add(split, f"cv/{Path(r['path']).stem}", clip.read_bytes(), r["sentence"])
            if ing.limit_hit():
                break
        ing.finish_input(tsv)
    ing.done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help=f"data root (default: {ROOT / 'data'}, or data_smoke with --limit-rows)")
    ap.add_argument("--sources", nargs="+", default=["reazon_small", "galgame", "eval"], choices=ALL_SOURCES)
    ap.add_argument("--galgame-shards", type=int, default=6, help="number of 0.88 GB tars (~47 h each) to take")
    ap.add_argument("--limit-rows", type=int, default=None, help="debug: stop each source after N kept rows (galgame hold-out exempt)")
    ap.add_argument("--force", action="store_true", help="wipe and re-ingest sources that are already present")
    args = ap.parse_args()

    # a row-limited run must never look like a finished dataset: keep it in its own root
    root = Path(args.data) if args.data else ROOT / ("data_smoke" if args.limit_rows else "data")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"data root: {root}")

    todo = []
    for s in args.sources:
        todo += ["eval_jsut", "eval_cv8", "eval_reazon"] if s == "eval" else [s]
    for s in todo:
        if args.force:
            remove_source(root, s)
            print(f"  {s}: wiped (--force). If teacher_out/{s} exists, delete it too - its outputs no longer match.")
        prog = load_progress(root, s)
        if prog["finished_inputs"]:
            print(f"  {s}: resuming, {len(prog['finished_inputs'])} input files already done")
        ing = Ingest(root, raw, s, args.limit_rows)
        if s in HF_PARQUET_SOURCES:
            ingest_hf_parquet(ing, *HF_PARQUET_SOURCES[s])
        elif s == "galgame":
            ingest_galgame(ing, args.galgame_shards)
        elif s == "cv":
            ingest_common_voice(ing)

    print("\n== manifest ==")
    by = {}
    for sh in read_manifest(root):
        k = (sh.source, sh.split)
        by.setdefault(k, [0, 0.0])
        by[k][0] += sh.rows
        by[k][1] += sh.hours
    for (src, split), (rows, hours) in sorted(by.items()):
        print(f"  {src:14s} {split:5s} rows={rows:7d}  {hours:7.1f} h")


if __name__ == "__main__":
    main()
