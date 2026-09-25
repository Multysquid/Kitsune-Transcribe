"""Download the training/eval sources into the local shard store (data/).

Sources (all streamed file-by-file from HF; each raw download is deleted after conversion):
  reazon_small  japanese-asr/whisper_transcriptions.reazonspeech.small  ~100 h TV speech, ungated parquet mirror
  galgame       litagin/Galgame_Speech_ASR_16kHz                         visual-novel voices, --galgame-shards tars (~47 h each);
                the first GALGAME_EVAL_ROWS kept keys of tar 0 are held out as an in-domain eval split. The keys are
                sorted content hashes, so this is a random utterance sample: the same games and voices are in train,
                i.e. a seen-speaker in-domain monitor, unlike eval_emilia, which is video-disjoint
  emilia_yodas  TTS-AGI/emilia-yodas JA/*.tar (ungated mirror of amphion/Emilia-Dataset Emilia-YODAS/JA, CC BY 4.0):
                YouTube CC-BY in-the-wild speech, pre-cut to 3-30 s, 24 kHz MP3; text = Emilia's WhisperX (Whisper
                medium) transcript. Tars are taken in order until --emilia-hours of kept audio. Replaces galgame for
                runs whose model must stay free of galgame's non-commercial / must-open-source terms.
                NEVER point this at the amphion Emilia/JA (non-YODAS) files: those are CC BY-NC.
  emilia_nc     laion/Emolia JA-B*_standard.tar.gz: Emilia's non-YODAS Japanese part (podcasts, talk shows; CC BY-NC 4.0,
                non-commercial models only), same format and filters as emilia_yodas, --emilia-nc-hours budget
  reazon_large  japanese-asr/whisper_transcriptions.reazonspeech.large (~5000 h, contains medium and small; rows
                already in reazon_small/reazon_medium are skipped)
  cv            Common Voice ja  -- NOT on HF anymore. Download cv-corpus-*-ja.tar.gz from
                https://datacollective.mozillafoundation.org and extract to data/raw/common_voice/ ; this script ingests it.
  eval_emilia   monitor-only Emilia hold-out: first 1000 Japanese clips of JA-B000029 from videos absent from
                emilia_yodas (ingest that first); scored against the teacher's hypothesis, not part of the gate
  eval          japanese-asr/ja_asr.jsut_basic5000, ja_asr.common_voice_8_0, ja_asr.reazonspeech_test  (eval_jsut / eval_cv8 / eval_reazon)

Resumable: every flushed shard is added to the manifest immediately and shards/<source>/progress.json records the
input files that are complete, so an interrupted run continues where it stopped. A source is considered done only
once progress.json says so. --force wipes a source (shards + manifest lines) and re-ingests it. One run per data root
at a time: a second one (another launcher started while the first still runs) exits at once instead of overwriting
the first one's shards (kitsune.store.lock_data_root; the OS drops the lock when its holder dies).

Every upstream repo is read at a pinned commit (REVISIONS): the training box rebuilds the audio shards from the
public repos instead of downloading them from home, and the teacher outputs are joined to that audio by utterance
id, so the box must see byte-identical upstream files. The pins are the commits the local data/ was built from
(data/raw/datasets--*/refs/main).

Stems are stable too: shards are numbered after the manifest's (an unlisted shard left by a kill is removed), and each
shard gets an id sidecar (shards/<source>/_ids/<stem>.parquet) naming the upstream input and ingest step it came from.
Reads of already-ingested ids (dedup, video exclusion, the Emilia budget) go through the sidecars, so they still work
after the label box has deleted labelled audio. For that box: --max-inputs takes a sorted prefix of a source's upstream
files, --whisper-dir keeps the ReazonSpeech mirrors' whisper transcripts while each file is local, --hold-file pauses
before the next download while the file exists, and --heartbeat is touched while the ingest makes progress.

--extent-config CFG ingests the extent of a run config (its extent block, sources and eval_sets) in the canonical order
of kitsune/extent.py, in this one process: the label box and the A100 rebuild run exactly that call, so they get the
same ids and stems (the order changes ids: reazon_small before reazon_large, Emilia as 300 h -> eval_emilia -> the
rest). It replaces --sources and the per-source flags, and <data>/extent_progress.json records the finished steps.

Usage:
  python scripts/01_prepare_data.py --sources reazon_small galgame eval --galgame-shards 6
  python scripts/01_prepare_data.py --sources eval_cv8 --limit-rows 64      # smoke run -> writes to data_smoke/
  python scripts/01_prepare_data.py --data /workspace/Kitsune-Transcribe/data --sources reazon_small eval   # vast box
  python scripts/01_prepare_data.py --sources reazon_large emilia_nc --max-inputs reazon_large=40,emilia_nc=6
  python scripts/01_prepare_data.py --data /workspace/Kitsune-Transcribe/data --extent-config configs/full.json
"""
import argparse
import csv
import json
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kitsune import extent  # noqa: E402
from kitsune.audio import audio_info  # noqa: E402
from kitsune.store import (  # noqa: E402
    ShardWriter, fsync_path, iter_rows, load_progress, lock_data_root, read_ids, read_manifest, remove_source,
    save_progress,
)

MIN_DUR, MAX_DUR = 0.3, 30.0  # teacher fast path is <=30 s; shorter than 0.3 s is noise

HF_PARQUET_SOURCES = {
    # name: (repo, split)
    "reazon_small": ("japanese-asr/whisper_transcriptions.reazonspeech.small", "train"),
    "reazon_medium": ("japanese-asr/whisper_transcriptions.reazonspeech.medium", "train"),
    "reazon_large": ("japanese-asr/whisper_transcriptions.reazonspeech.large", "train"),
    "eval_jsut": ("japanese-asr/ja_asr.jsut_basic5000", "eval"),
    "eval_cv8": ("japanese-asr/ja_asr.common_voice_8_0", "eval"),
    "eval_reazon": ("japanese-asr/ja_asr.reazonspeech_test", "eval"),
}
GALGAME_REPO = "litagin/Galgame_Speech_ASR_16kHz"
GALGAME_EVAL_ROWS = 1000  # first N kept utterances become the in-domain eval split
# repo -> commit SHA. A repo missing here would silently follow `main`, so every source repo must be listed.
REVISIONS = {
    GALGAME_REPO: "3fb86654222b3f0af0f7c332ae6a0ef9752a9451",
    "japanese-asr/whisper_transcriptions.reazonspeech.small": "c74b52fc164cf7b64936ca62aee4336eac626739",
    "japanese-asr/whisper_transcriptions.reazonspeech.medium": "c801154945d5cf756f727e06afbef472d60fed37",
    "japanese-asr/ja_asr.jsut_basic5000": "278db379fc96167ff2293d7abf9ab86976afcd78",
    "japanese-asr/ja_asr.common_voice_8_0": "bf8819e8d9a5feb51b0c718686bd20ea67a3c729",
    "japanese-asr/ja_asr.reazonspeech_test": "dd08bfb9dfc1cef4e4d0609fd78c3755d48b926f",
    "TTS-AGI/emilia-yodas": "613a372ba2cc5ecb6b27ea38a4a0926abb38263d",
    "japanese-asr/whisper_transcriptions.reazonspeech.large": "4ad8d64a13594f0ce1f0622627a18ef99b42b5e8",
    "laion/Emolia": "4d375b4bf276834e555022bb7c937f87091e895e",
}
EMILIA_REPO = "TTS-AGI/emilia-yodas"
EMILIA_EVAL_TAR, EMILIA_EVAL_ROWS = "JA/JA-B000029.tar", 1000  # last tar: far beyond any training budget
# Emilia's non-YODAS Japanese part (CC BY-NC 4.0 per amphion/Emilia-Dataset; the laion card's cc-by-4.0 tag does not
# override that) from LAION's ungated re-upload, which adds emotion annotations. Only for non-commercial models.
EMOLIA_REPO = "laion/Emolia"
# Emilia's language tag is per video, so a 'ja' tar still holds some English clips; require Japanese script instead
_JA_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
ALL_SOURCES = ["reazon_small", "reazon_medium", "reazon_large", "galgame", "emilia_yodas", "emilia_nc", "cv", "eval",
               "eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia"]
# ReazonSpeech tiers are nested (small is a subset of medium), so a larger tier must skip rows already ingested.
# Nothing dedups training against the gate eval sets (only eval_emilia is kept video-disjoint, in ingest_emilia):
# ~16 eval_reazon lines match a reazon_small text (re-aired broadcasts), worth <= ~0.3 pp of its CER at worst; expect
# more with the medium/large tiers.
DEDUP_AGAINST = {"reazon_medium": ("reazon_small",), "reazon_large": ("reazon_small", "reazon_medium")}
MIN_FREE_GB = 30.0  # stop a download (resumably) before it would fill the data disk
HOLD_POLL_S = 15  # --hold-file poll interval
HEARTBEAT_ROWS = 1000  # touch --heartbeat every this many kept rows (one input can take minutes)
# sources whose upstream files have a sorted listing that --max-inputs can cut (galgame: the same as --galgame-shards)
MAX_INPUT_SOURCES = (*HF_PARQUET_SOURCES, "galgame", "emilia_yodas", "emilia_nc")
REFUSED_EXIT = 3  # an --extent-config that is no valid extent: vast/bootstrap.sh's retry() does not repeat exit 3


class Ingest:
    """Shared bookkeeping for one source: writers, stats, progress, download cleanup.

    `step` (default: the source) and the current input go into every shard's id sidecar; `heartbeat` is touched while
    the ingest makes progress; while `hold_file` exists, the next download waits."""

    def __init__(self, root: Path, raw: Path, source: str, limit_rows: int | None, step: str | None = None,
                 heartbeat: Path | None = None, hold_file: Path | None = None):
        self.root, self.raw, self.source, self.limit = root, raw, source, limit_rows
        self.step = step or source
        self.heartbeat = Path(heartbeat) if heartbeat else None
        self.hold_file = Path(hold_file) if hold_file else None
        self.progress = load_progress(root, source)
        self.writers: dict[str, ShardWriter] = {}
        self.cur = {"input": None, "step": self.step}  # the sidecar meta of the rows being added
        self.stats = dict(kept=0, empty_text=0, bad_audio=0, bad_duration=0, dup=0, seconds=0.0)
        self.counted = 0  # rows that count against --limit-rows

    def writer(self, split: str) -> ShardWriter:
        if split not in self.writers:
            self.writers[split] = ShardWriter(self.root, self.source, split, meta=self.cur)
        return self.writers[split]

    def begin_input(self, name: str):
        """Name the upstream input whose rows follow, for the sidecars of every shard flushed from now on. Every input
        is flushed on its own (finish_input), so a shard's rows all come from the input named here; a budget-cut tar's
        rows are flushed by done(), still under its name."""
        self.cur = {"input": name, "step": self.step}
        for w in self.writers.values():
            w.meta = self.cur

    def beat(self):
        if self.heartbeat is not None:
            self.heartbeat.parent.mkdir(parents=True, exist_ok=True)
            self.heartbeat.touch()

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
        # count the float32 value the shard stores: a resumed run sums the stored durations (ingest_emilia*), so the
        # Emilia budget then cuts at the same clip in a fresh and a resumed run
        self.stats["seconds"] += float(np.float32(dur))
        if count:
            self.counted += 1
        if self.stats["kept"] % HEARTBEAT_ROWS == 0:
            self.beat()
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
        if self.hold_file is not None and self.hold_file.exists():
            # the label box holds the ingest while too much unlabelled audio is on disk; waiting is progress, not a hang
            print(f"  {self.source}: held by {self.hold_file}; waiting before {filename}")
            while self.hold_file.exists():
                self.beat()
                time.sleep(HOLD_POLL_S)
        here = next(d for d in (self.root, *self.root.parents) if d.exists())  # the data root may not exist yet
        free_gb = shutil.disk_usage(here).free / 1e9
        if free_gb < MIN_FREE_GB:  # everything flushed so far stays; a re-run continues with this input file
            raise SystemExit(f"  {self.source}: only {free_gb:.0f} GB free on the data disk (< {MIN_FREE_GB:.0f} GB); "
                             f"stopping before {filename}")
        self.beat()
        local = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo], cache_dir=self.raw))
        if local.is_file():  # for the extent record; saved now, as a kill before finish_input would lose it
            self.progress.setdefault("input_bytes", {})[filename] = local.stat().st_size
            save_progress(self.root, self.source, self.progress)
        return local

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


def capture_whisper(local: Path, part: Path):
    """Keep the mirror file's name + whisper_transcript columns (whisper-large-v3 token ids, nulls kept as null) while
    the file is on disk, for the second-opinion join: re-reading them later from the Hub costs the whole mirror again.
    Rewritten whole on a resume (the same file gives the same part)."""
    table = pq.read_table(local, columns=["name", "whisper_transcript"])
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    fsync_path(tmp)
    tmp.replace(part)


def ingest_hf_parquet(ing: Ingest, repo: str, split: str, max_inputs: int | None = None,
                      whisper_dir: Path | None = None):
    """`max_inputs`: only the first N of the sorted upstream files. `whisper_dir`: for the ReazonSpeech mirrors, keep
    each file's whisper transcripts in <whisper_dir>/<source>/<file name> before the file is marked finished."""
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=REVISIONS[repo])
                   if f.endswith(".parquet"))
    ing.progress["n_listed"] = len(files)
    files = files[:max_inputs]
    columns = ["audio", "transcription"] + (["name"] if ing.source.startswith("reazon") else [])
    capture = whisper_dir is not None and ing.source.startswith("reazon")  # only the mirrors carry the column
    # dedup against a nested smaller tier AND against this source's own manifest shards (a crash mid input
    # file re-reads rows already flushed; only manifest-listed shards count - orphans are re-ingested).
    seen: set[str] = set()
    dedup_sources = {ing.source} | set(DEDUP_AGAINST.get(ing.source, ()))
    for sh in read_manifest(ing.root):
        if sh.source in dedup_sources:
            for row in read_ids(ing.root, sh):
                seen.add(row["id"].split("/", 1)[1])
    if seen:
        print(f"  {ing.source}: deduplicating against {len(seen)} rows already in {sorted(dedup_sources)}")
    for f in tqdm(files, desc=ing.source, unit="file"):
        if ing.is_finished(f):
            continue
        ing.begin_input(f)
        local = ing.download(repo, f)
        for i, row in enumerate(iter_rows(local, columns=columns)):
            rid = row.get("name") or row["audio"].get("path") or f"{Path(f).stem}-{i}"
            if rid in seen:
                ing.stats["dup"] += 1
                continue
            ing.add(split, f"{ing.source}/{rid}", row["audio"]["bytes"], row["transcription"])
            if ing.limit_hit():
                break
        if capture:  # before finish_input: a finished input always has its part
            capture_whisper(local, Path(whisper_dir) / ing.source / Path(f).name)
        ing.finish_input(f)
        ing.free_download(local)
        if ing.limit_hit():
            break
    ing.done()


def ingest_galgame(ing: Ingest, n_shards: int):
    tars = sorted(f for f in HfApi().list_repo_files(GALGAME_REPO, repo_type="dataset", revision=REVISIONS[GALGAME_REPO])
                  if f.endswith(".tar"))
    ing.progress["n_listed"] = len(tars)
    tars = tars[:n_shards]
    # self-dedup: resuming an interrupted tar re-reads rows already stored; skip everything already ingested.
    # Gate on the manifest, not finished_inputs - a crash inside the FIRST tar also leaves flushed shards behind.
    seen: set[str] = set()
    eval_kept = 0
    for sh in read_manifest(ing.root):
        if sh.source == "galgame":
            if sh.split == "eval":
                eval_kept += sh.rows
            for row in read_ids(ing.root, sh):
                seen.add(row["id"])
    if seen:
        print(f"  galgame: {len(seen)} rows already ingested (duplicates will be skipped), eval hold-out {eval_kept}")
    pending: dict[str, dict] = {}  # webdataset pairs <key>.ogg / <key>.txt can arrive in either order
    for f in tqdm(tars, desc="galgame", unit="tar"):
        if ing.is_finished(f):
            continue
        ing.begin_input(f)  # the hold-out rows of tar 0 are flushed by its finish_input, so its eval shard names tar 0
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


def ingest_emilia(ing: Ingest, max_hours: float, max_inputs: int | None = None):
    """Emilia-YODAS JA webdataset tars (<key>.mp3 + <key>.json), taken in order until max_hours of kept audio.
    The budget is counted over everything already in the manifest, so a resumed or re-run ingest (e.g. on the
    training box) stops at the same utterance. A tar cut short by the budget is not marked finished, so raising
    --emilia-hours later continues inside it (the id dedup skips what is already stored). `max_inputs`: only the first
    N of the sorted train tars."""
    tars = sorted(f for f in HfApi().list_repo_files(EMILIA_REPO, repo_type="dataset", revision=REVISIONS[EMILIA_REPO])
                  if f.startswith("JA/") and f.endswith(".tar") and f != EMILIA_EVAL_TAR)  # never the hold-out's tar
    ing.progress["n_listed"] = len(tars)
    tars = tars[:max_inputs]
    # videos with clips in the eval_emilia hold-out never enter training (a video's clips can span tars)
    eval_videos = {_emilia_video(row["id"]) for sh in read_manifest(ing.root) if sh.source == "eval_emilia"
                   for row in read_ids(ing.root, sh)}
    ing.stats.setdefault("eval_video", 0)
    seen: set[str] = set()
    for sh in read_manifest(ing.root):
        if sh.source == ing.source:
            for row in read_ids(ing.root, sh, columns=["id", "duration"]):
                seen.add(row["id"])
                ing.stats["seconds"] += row["duration"]  # count stored audio against the budget
    budget_s = max_hours * 3600
    if seen:
        print(f"  {ing.source}: {len(seen)} rows / {ing.stats['seconds'] / 3600:.1f} h already ingested")
    ing.stats.setdefault("non_ja", 0)
    pending: dict[str, dict] = {}
    for f in tqdm(tars, desc=ing.source, unit="tar"):
        if ing.stats["seconds"] >= budget_s or ing.limit_hit():
            break
        if ing.is_finished(f):
            continue
        ing.begin_input(f)
        local = ing.download(EMILIA_REPO, f)
        cut_short = False
        with tarfile.open(local, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit("/", 1)[-1].rsplit(".", 1)
                d = pending.setdefault(key, {})
                d[ext] = tf.extractfile(m).read()
                if "mp3" not in d or "json" not in d:
                    continue
                del pending[key]
                rid = f"{ing.source}/{key}"
                if rid in seen:
                    ing.stats["dup"] += 1
                    continue
                if _emilia_video(key) in eval_videos:
                    ing.stats["eval_video"] += 1
                    continue
                meta = json.loads(d["json"].decode("utf-8", "replace"))
                text = meta.get("text") or ""
                if meta.get("language", "ja") != "ja" or not _JA_SCRIPT.search(text):
                    ing.stats["non_ja"] += 1
                    continue
                ing.add("train", rid, d["mp3"], text)
                if ing.stats["seconds"] >= budget_s or ing.limit_hit():
                    cut_short = True
                    break
        if not cut_short:
            ing.finish_input(f)
        ing.free_download(local)
    if pending:
        print(f"  {ing.source}: {len(pending)} unpaired members dropped")
    ing.done()
    print(f"  {ing.source}: dropped non-Japanese text={ing.stats['non_ja']}")


def ingest_emilia_nc(ing: Ingest, max_hours: float, max_inputs: int | None = None):
    """Emilia's non-YODAS JA part from laion/Emolia JA-B*_standard.tar.gz (<worker>/<key>.mp3 + .json), in order until
    max_hours of kept audio. Same filters and resume logic as ingest_emilia; the budget counts stored audio.
    `max_inputs`: only the first N of the sorted tars."""
    tars = sorted(f for f in HfApi().list_repo_files(EMOLIA_REPO, repo_type="dataset", revision=REVISIONS[EMOLIA_REPO])
                  if f.startswith("JA-") and f.endswith("_standard.tar.gz"))
    ing.progress["n_listed"] = len(tars)
    tars = tars[:max_inputs]
    seen: set[str] = set()
    for sh in read_manifest(ing.root):
        if sh.source == ing.source:
            for row in read_ids(ing.root, sh, columns=["id", "duration"]):
                seen.add(row["id"])
                ing.stats["seconds"] += row["duration"]
    budget_s = max_hours * 3600
    ing.stats.setdefault("non_ja", 0)
    for f in tqdm(tars, desc=ing.source, unit="tar"):
        if ing.stats["seconds"] >= budget_s or ing.limit_hit():
            break
        if ing.is_finished(f):
            continue
        ing.begin_input(f)
        local = ing.download(EMOLIA_REPO, f)
        pending: dict[str, dict] = {}
        cut_short = False
        with tarfile.open(local, "r:gz") as tf:
            for m in tf:
                if not m.isfile() or "." not in m.name.rsplit("/", 1)[-1]:
                    continue
                key, ext = m.name.rsplit("/", 1)[-1].rsplit(".", 1)
                if ext not in ("mp3", "json"):
                    continue
                d = pending.setdefault(key, {})
                d[ext] = tf.extractfile(m).read()
                if "mp3" not in d or "json" not in d:
                    continue
                del pending[key]
                rid = f"{ing.source}/{key}"
                if rid in seen:
                    ing.stats["dup"] += 1
                    continue
                meta = json.loads(d["json"].decode("utf-8", "replace"))
                text = meta.get("text") or ""
                if meta.get("language", "ja") != "ja" or not _JA_SCRIPT.search(text):
                    ing.stats["non_ja"] += 1
                    continue
                ing.add("train", rid, d["mp3"], text)
                if ing.stats["seconds"] >= budget_s or ing.limit_hit():
                    cut_short = True
                    break
        if pending:
            print(f"  {ing.source}/{f}: {len(pending)} unpaired members dropped")
        if not cut_short:
            ing.finish_input(f)
        ing.free_download(local)
    ing.done()
    print(f"  {ing.source}: dropped non-Japanese text={ing.stats['non_ja']}")


def ingest_emilia_eval(ing: Ingest, tar: str, n_rows: int, train_source: str = "emilia_yodas"):
    """Monitor-only in-domain hold-out: the first n_rows Japanese clips of a tar the training budget never reaches,
    minus any video that also has clips in the training set. Its text is Whisper-medium output, so the useful score
    is CER against the TEACHER's hypothesis; it is not part of the pre-registered gate. Run after `train_source`."""
    train_videos = set()
    for sh in read_manifest(ing.root):
        if sh.source == train_source:
            for row in read_ids(ing.root, sh):
                train_videos.add(_emilia_video(row["id"]))
    if not train_videos:
        raise SystemExit(f"{ing.source}: ingest {train_source} first (its video ids are excluded from the hold-out)")
    kept = sum(sh.rows for sh in read_manifest(ing.root) if sh.source == ing.source)
    if kept >= n_rows or ing.is_finished(tar):
        ing.done()
        return
    ing.stats.setdefault("train_video", 0)
    ing.stats.setdefault("non_ja", 0)
    ing.begin_input(tar)
    local = ing.download(EMILIA_REPO, tar)
    pending: dict[str, dict] = {}
    with tarfile.open(local, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit("/", 1)[-1].rsplit(".", 1)
            d = pending.setdefault(key, {})
            d[ext] = tf.extractfile(m).read()
            if "mp3" not in d or "json" not in d:
                continue
            del pending[key]
            if _emilia_video(key) in train_videos:
                ing.stats["train_video"] += 1
                continue
            meta = json.loads(d["json"].decode("utf-8", "replace"))
            text = meta.get("text") or ""
            if meta.get("language", "ja") != "ja" or not _JA_SCRIPT.search(text):
                ing.stats["non_ja"] += 1
                continue
            kept += ing.add("eval", f"{ing.source}/{key}", d["mp3"], text, count=False)
            if kept >= n_rows:
                break
    ing.finish_input(tar)
    ing.free_download(local)
    ing.done()
    print(f"  {ing.source}: skipped clips of training videos={ing.stats['train_video']} non-Japanese={ing.stats['non_ja']}")


def _emilia_video(id_or_key: str) -> str:
    """'emilia_yodas/JA_<video>_W000123' or 'JA_<video>_W000123' -> '<video>' (YODAS video ids may contain '_')."""
    key = id_or_key.rsplit("/", 1)[-1]
    return key[len("JA_"):key.rfind("_W")] if key.startswith("JA_") and "_W" in key else key


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
        ing.begin_input(tsv)
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


def source_repo(source: str) -> str | None:
    """The upstream repo a source is read from (None: cv, a manual download)."""
    if source in HF_PARQUET_SOURCES:
        return HF_PARQUET_SOURCES[source][0]
    return {"galgame": GALGAME_REPO, "emilia_yodas": EMILIA_REPO, "eval_emilia": EMILIA_REPO,
            "emilia_nc": EMOLIA_REPO}.get(source)


def ingest_source(ing: Ingest, cap: int | None, whisper_dir: Path | None = None, emilia_hours: float = 300.0,
                  emilia_nc_hours: float = float("inf")):
    """Ingest ing.source. `cap`: only its first N sorted upstream files (galgame: its tars); None: all of them."""
    s = ing.source
    if s in HF_PARQUET_SOURCES:
        ingest_hf_parquet(ing, *HF_PARQUET_SOURCES[s], max_inputs=cap, whisper_dir=whisper_dir)
    elif s == "galgame":
        ingest_galgame(ing, cap)
    elif s == "emilia_yodas":
        ingest_emilia(ing, emilia_hours, max_inputs=cap)
    elif s == "eval_emilia":
        ingest_emilia_eval(ing, EMILIA_EVAL_TAR, EMILIA_EVAL_ROWS)
    elif s == "emilia_nc":
        ingest_emilia_nc(ing, emilia_nc_hours, max_inputs=cap)
    elif s == "cv":
        ingest_common_voice(ing)


def load_extent_config(path: Path) -> dict:
    """The run config of --extent-config (a relative path not found here is taken from the checkout, as
    make_selection.py --config does). One that is no valid extent exits REFUSED_EXIT: a retry cannot fix it."""
    p = Path(path)
    try:
        cfg = json.loads((p if p.is_absolute() or p.exists() else ROOT / p).read_text(encoding="utf-8"))
        problems = extent.validate(cfg) if isinstance(cfg, dict) else ["not a JSON object"]
    except (OSError, ValueError) as e:
        problems = [f"{type(e).__name__}: {e}"]
    if problems:
        print(f"{path}: not an extent config: {'; '.join(problems)}", file=sys.stderr)
        sys.exit(REFUSED_EXIT)
    return cfg


def ingest_extent(root: Path, raw: Path, cfg: dict, config: Path, whisper_dir: Path | None = None,
                  heartbeat: Path | None = None, hold_file: Path | None = None):
    """The extent of the (valid) run config `cfg`, step by step in the canonical order (kitsune.extent.plan_steps), each
    step an ingest of its source under the step's key, cap and Emilia budget. After each step
    <root>/extent_progress.json lists it as completed, and a re-run skips it. Re-running a step is idempotent anyway
    (finished inputs are skipped, the 300 h step stops at once at its budget, eval_emilia returns early), so progress
    recorded for another plan (other caps or names) is dropped and every step of this one runs again."""
    plan = extent.plan_steps(cfg)
    steps = [[s.key, cap] for s, cap in plan]
    prev = extent.read_progress(root) or {}
    same = prev.get("canonical_version") == extent.CANONICAL_VERSION and prev.get("steps") == steps
    completed = list(prev.get("completed", [])) if same else []
    if prev.get("completed") and not same:
        print(f"  {extent.PROGRESS_FILE} records another plan ({prev.get('steps')}); every step runs again")
    repos = {s.source: source_repo(s.source) for s, _ in plan}
    progress = dict(canonical_version=extent.CANONICAL_VERSION, config=str(config), extent=extent.extent_block(cfg),
                    steps=steps, completed=completed, repos=repos,
                    revisions={r: REVISIONS[r] for r in dict.fromkeys(repos.values()) if r}, tools=extent.tools())
    extent.write_progress(root, progress)
    name = cfg["extent"]["name"]
    for step, cap in plan:
        if step.key in completed:
            print(f"== {name}: {step.key} already complete")
            continue
        print(f"== {name}: {step.key}" + (f" (the first {cap} inputs)" if cap is not None else ""))
        ing = Ingest(root, raw, step.source, None, step=step.key, heartbeat=heartbeat, hold_file=hold_file)
        ingest_source(ing, cap, whisper_dir=whisper_dir, emilia_hours=step.emilia_hours)
        completed.append(step.key)
        extent.write_progress(root, progress)
    print(f"== {name}: all {len(plan)} steps complete")


def parse_max_inputs(spec: str) -> dict[str, int]:
    """'reazon_large=40,emilia_nc=6' -> {source: N}. Comma-separated with no spaces, so that it passes the vast env."""
    out: dict[str, int] = {}
    for item in spec.split(","):
        src, sep, n = item.partition("=")
        if not sep or src not in MAX_INPUT_SOURCES or not n.isdigit() or int(n) < 1 or src in out:
            raise argparse.ArgumentTypeError(
                f"{item!r}: expected SRC=N with N >= 1, each SRC once, SRC one of {', '.join(MAX_INPUT_SOURCES)}")
        out[src] = int(n)
    return out


def main():
    global MIN_FREE_GB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help=f"data root (default: {ROOT / 'data'}, or data_smoke with --limit-rows)")
    ap.add_argument("--sources", nargs="+", default=None, choices=ALL_SOURCES,
                    help="sources to ingest, in this order (default: reazon_small galgame eval)")
    ap.add_argument("--galgame-shards", type=int, default=None, help="number of 0.88 GB tars (~47 h each) to take "
                    "(default 6; the same as --max-inputs galgame=N)")
    ap.add_argument("--emilia-hours", type=float, default=None,
                    help="kept hours of Emilia-YODAS JA to ingest (default 300; ~1 GB tar per 36-73 h)")
    ap.add_argument("--emilia-nc-hours", type=float, default=None,
                    help="kept hours of Emilia non-YODAS JA (CC BY-NC; default: all)")
    ap.add_argument("--max-inputs", type=parse_max_inputs, action="append", default=None, metavar="SRC=N[,SRC=N]",
                    help="take only the first N of a source's sorted upstream files (parquet sources, galgame, "
                         "emilia_yodas, emilia_nc); may be repeated, each SRC once")
    ap.add_argument("--whisper-dir", type=Path, default=None, help="keep the ReazonSpeech mirrors' name + "
                    "whisper_transcript columns in DIR/<source>/<file name> while each file is on disk")
    ap.add_argument("--hold-file", type=Path, default=None,
                    help="while this file exists, wait before the next download")
    ap.add_argument("--heartbeat", type=Path, default=None, help="touch this file at each download, every "
                    f"{HEARTBEAT_ROWS} kept rows and while held")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB, help="stop before a download would leave less free")
    ap.add_argument("--limit-rows", type=int, default=None, help="debug: stop each source after N kept rows (galgame hold-out exempt)")
    ap.add_argument("--force", action="store_true", help="wipe and re-ingest sources that are already present")
    ap.add_argument("--extent-config", type=Path, default=None, metavar="CFG",
                    help="ingest the extent of run config CFG (extent block, sources, eval_sets) in the canonical "
                         "order of kitsune/extent.py, as the label box and the A100 rebuild do; replaces --sources "
                         "and the per-source flags")
    args = ap.parse_args()
    cfg = None
    if args.extent_config is not None:
        # the config alone decides what is ingested: a flag next to it would silently give other ids or stems
        given = [flag for flag, v in (("--sources", args.sources), ("--galgame-shards", args.galgame_shards),
                                      ("--emilia-hours", args.emilia_hours),
                                      ("--emilia-nc-hours", args.emilia_nc_hours), ("--max-inputs", args.max_inputs),
                                      ("--limit-rows", args.limit_rows), ("--force", args.force or None))
                 if v is not None]
        if given:
            ap.error(f"--extent-config defines the sources and their caps; drop {', '.join(given)}")
        cfg = load_extent_config(args.extent_config)
    max_inputs: dict[str, int] = {}
    for caps in args.max_inputs or []:  # a repeated flag adds to the caps; it must never silently drop one
        if twice := sorted(caps.keys() & max_inputs.keys()):
            ap.error(f"--max-inputs caps {', '.join(twice)} more than once")
        max_inputs.update(caps)
    if args.galgame_shards is not None and "galgame" in max_inputs:
        ap.error("--galgame-shards and --max-inputs galgame=N both cap galgame; give one of them")
    galgame_shards = max_inputs.get("galgame", 6 if args.galgame_shards is None else args.galgame_shards)
    MIN_FREE_GB = args.min_free_gb

    # a row-limited run must never look like a finished dataset: keep it in its own root
    root = Path(args.data) if args.data else ROOT / ("data_smoke" if args.limit_rows else "data")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock = lock_data_root(root)  # held until main() returns: one ingest per data root (a second run clobbers shards)
    if lock is None:
        raise SystemExit(f"{root}: another 01_prepare_data is ingesting into this data root ({root / '.ingest.lock'}); "
                         "wait for it to finish")
    print(f"data root: {root}")

    todo, sources = [], args.sources or ["reazon_small", "galgame", "eval"]
    for s in [] if cfg is not None else sources:  # --extent-config: ingest_extent below
        todo += ["eval_jsut", "eval_cv8", "eval_reazon"] if s == "eval" else [s]
    for s in todo:
        if args.force:
            remove_source(root, s)
            print(f"  {s}: wiped (--force). If teacher_out/{s} exists, delete it too - its outputs no longer match.")
        prog = load_progress(root, s)
        if prog["finished_inputs"]:
            print(f"  {s}: resuming, {len(prog['finished_inputs'])} input files already done")
        ing = Ingest(root, raw, s, args.limit_rows, heartbeat=args.heartbeat, hold_file=args.hold_file)
        ingest_source(ing, galgame_shards if s == "galgame" else max_inputs.get(s), whisper_dir=args.whisper_dir,
                      emilia_hours=300.0 if args.emilia_hours is None else args.emilia_hours,
                      emilia_nc_hours=float("inf") if args.emilia_nc_hours is None else args.emilia_nc_hours)
    if cfg is not None:
        ingest_extent(root, raw, cfg, args.extent_config, whisper_dir=args.whisper_dir, heartbeat=args.heartbeat,
                      hold_file=args.hold_file)

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
