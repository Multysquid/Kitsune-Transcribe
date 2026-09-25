"""Second-opinion transcripts for every teacher-labelled shard, for agreement-based label filtering.

Why: the Galgame reference text is non-verbatim (teacher-vs-subtitle CER 0.31 mean / 20 % above 0.5), so
filtering pseudo-labels against the dataset text would throw away a third of the anime-voice data. Instead we
get an independent hypothesis from a second, architecturally different teacher and gate on TEACHER AGREEMENT:
two models rarely make the same mistake, so low Cohere-vs-Whisper CER marks a trustworthy label no matter what
the subtitle says.

Where the second opinion comes from:
  reazon_small / reazon_medium   free - the japanese-asr mirror parquets carry a `whisper_transcript` column
                                 (openai/whisper-large-v3 token ids). Only the name+transcript columns are
                                 streamed (no audio re-download) and cached under second_out/_cache/.
  emilia_yodas                   free - the dataset text IS an ASR transcript (Emilia-Pipe WhisperX, Whisper medium),
                                 so hyp2 = the stored reference and no audio is touched. Weaker than large-v3, so
                                 its agree values run higher; filter thresholds are per second-opinion model.
  everything else                kotoba-tech/kotoba-whisper-v2.0 (distil-whisper large-v3 tuned on Japanese,
                                 ~0.75B) is run over the shard audio on the GPU.

For each teacher shard  teacher_out/<source>/<stem>.jsonl  this writes  second_out/<source>/<stem>.jsonl:
  {id, hyp2, model2, agree, cer2}
    hyp2   second-opinion transcript (null if the source row has no precomputed transcript)
    agree  CER(teacher hyp vs hyp2), normalised - the label-quality gate for distillation
    cer2   CER(hyp2 vs dataset text) - for statistics
Shards are processed in manifest order, output is atomic per shard and existing outputs are skipped, so the
script is resumable and safe under scripts/run_teacher_pass.py, one source per run (see the --limit-shards note in
main), e.g. as scripts/start_second_opinion.cmd runs it:
  python scripts/run_teacher_pass.py --script scripts/02b_second_opinion.py --done-suffix .jsonl --batch 24
      --watch-dir second_out/galgame --sources galgame

Usage:
  python scripts/02b_second_opinion.py                     # everything with teacher output
  python scripts/02b_second_opinion.py --sources eval_jsut --limit-shards 2 --out second_out_smoke

The label box (vast/label.py) runs it every sync cycle, CPU only, with the flags below; without them the laptop path
above is unchanged.
  --judge parakeet --parakeet-out DIR   every source that is neither a join nor a self-text source (galgame) takes
                                        hyp2 from the Parakeet pass's jsonl `hyp` by id (02p); kotoba is never loaded
  --whisper-dir DIR                     join sources read the whisper transcripts that 01 --whisper-dir captured per
                                        upstream file (DIR/<source>/<file name>, nulls kept) instead of streaming the
                                        mirror again; the file is named by the shard's id sidecar. A part that is not
                                        there yet skips the shard; once <data>/extent_progress.json lists the step as
                                        complete, a missing part is an integrity error (exit 65)
  --require-npz                         a shard is eligible once the teacher npz (and with --parakeet-out the Parakeet
                                        npz) exists: both passes write the jsonl first, so a jsonl alone may belong to
                                        a pass that is still writing
  --strict-existing                     an existing output whose ids differ from the teacher's, a teacher row without
                                        a Parakeet row, or a meta.json with other settings exits 65 instead of being
                                        redone: the label root is write-once
With --parakeet-out, a join row whose whisper transcript is null or missing takes the Parakeet hypothesis (model2
FALLBACK_MODEL2); more than FALLBACK_MAX of a shard doing so exits 65 (a capture bug, not the mirror's nulls).
An existing output is kept only when its ids equal the teacher jsonl's in order. Stems checked that way are listed in
<out>/_cache/verified.txt (local, never uploaded), so the periodic runs stay cheap.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kitsune import extent  # noqa: E402
from kitsune.audio import TARGET_SR, decode_audio  # noqa: E402
from kitsune.generation import RepetitionStop  # noqa: E402
from kitsune.store import fsync_path, read_manifest, read_shard, sidecar_meta  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402

# every hub read is pinned, like the datasets in 01: a repo that moves between passes would mix two second opinions
MODEL2 = "kotoba-tech/kotoba-whisper-v2.0"
MODEL2_REVISION = "7eb575277d18909a4af8a24e3ae8cce2e99794ae"
WHISPER_TOK = "openai/whisper-large-v3"  # tokenizer that produced the precomputed transcripts
WHISPER_TOK_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
JOIN_SOURCES = {  # sources whose mirror parquets already carry whisper-large-v3 transcripts: (repo, commit), the
    # commits 01_prepare_data.py pins in REVISIONS (tests/test_infra.py checks that the two copies agree)
    "reazon_small": ("japanese-asr/whisper_transcriptions.reazonspeech.small", "c74b52fc164cf7b64936ca62aee4336eac626739"),
    "reazon_medium": ("japanese-asr/whisper_transcriptions.reazonspeech.medium", "c801154945d5cf756f727e06afbef472d60fed37"),
    "reazon_large": ("japanese-asr/whisper_transcriptions.reazonspeech.large", "4ad8d64a13594f0ce1f0622627a18ef99b42b5e8"),
}
SELF_TEXT_SOURCES = {  # sources whose dataset text is itself a second-model transcript
    "emilia_yodas": "whisper-medium (Emilia WhisperX)",
    "eval_emilia": "whisper-medium (Emilia WhisperX)",
    "emilia_nc": "whisper-medium (Emilia WhisperX)",
}
# --judge parakeet: the Parakeet pass (02p) of the laptop-published model, decoded with NeMo's max-symbols guard
# Tied to kitsune/parakeet.py NEMO_REVISION and configs/full.json label.parakeet.max_symbols (label.py passes that
# value to 02p): change all three together, or the recorded judge states the wrong decode setting.
PARAKEET_MODEL2 = "nvidia/parakeet-tdt_ctc-0.6b-ja@44edb27 (TDT greedy, max_symbols 10)"
FALLBACK_MODEL2 = "parakeet-fallback (no whisper transcript)"  # a join row with a null or missing transcript
FALLBACK_MAX = 0.20  # a larger fallback share in one shard is a capture bug, not the mirror's nulls
FALLBACK_FLOOR = 20  # ...but only above this many null rows: a tail shard of 1-4 rows with one mirror null is fine
INTEGRITY_EXIT = 65  # the label root is write-once: an inconsistent input or output stops the box, never a redo
VERIFIED_CACHE = "verified.txt"  # under <out>/_cache: "<source>/<stem>" outputs whose ids were checked
META_KEYS = ("join_model", "tokenizer", "tokenizer_revision", "join_revisions", "judges", "fallback", "agree", "cer2")


def integrity(msg: str):
    print(f"integrity: {msg}", file=sys.stderr, flush=True)
    sys.exit(INTEGRITY_EXIT)


def build_whisper_cache(source: str, repo: str, revision: str, cache_dir: Path) -> Path:
    """Stream name + whisper_transcript columns from the mirror and cache the decoded text locally.
    One atomically-written part per mirror file, so a network failure only costs the file in flight."""
    part_dir = cache_dir / f"whisper_{source}"
    part_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, HfFileSystem
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WHISPER_TOK, revision=WHISPER_TOK_REVISION)
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision) if f.endswith(".parquet"))
    fs = HfFileSystem()
    for f in tqdm(files, desc=f"whisper cache {source}", unit="file"):
        part = part_dir / Path(f).name
        if part.exists():
            continue
        with fs.open(f"datasets/{repo}@{revision}/{f}", "rb") as fh:
            t = pq.read_table(fh, columns=["name", "whisper_transcript"])
        lists = t.column("whisper_transcript").to_pylist()
        keep = [i for i, x in enumerate(lists) if x is not None]  # a null transcript never reaches batch_decode
        decoded = tok.batch_decode([lists[i] for i in keep], skip_special_tokens=True) if keep else []
        texts = [None] * len(lists)
        for i, x in zip(keep, decoded):
            texts[i] = x.strip()
        tmp = part.with_suffix(".parquet.tmp")
        pq.write_table(pa.table({"name": t.column("name").to_pylist(), "text": pa.array(texts, pa.string())}), tmp)
        fsync_path(tmp)
        tmp.replace(part)
    return part_dir


def load_whisper_map(source: str, cache_dir: Path) -> dict[str, str]:
    repo, revision = JOIN_SOURCES[source]
    part_dir = build_whisper_cache(source, repo, revision, cache_dir)
    out: dict[str, str] = {}
    for part in sorted(part_dir.glob("*.parquet")):
        t = pq.read_table(part)
        out.update(zip(t.column("name").to_pylist(), t.column("text").to_pylist()))
    return out


class WhisperParts:
    """The whisper transcripts 01 --whisper-dir captured while each mirror file was on disk:
    DIR/<source>/<upstream file name> with columns name and whisper_transcript (whisper-large-v3 token ids, null where
    the mirror has none). Decoded with the pinned tokenizer once per upstream file (consecutive stems share one);
    the null lists stay None, because batch_decode cannot take them."""

    def __init__(self, whisper_dir: Path):
        self.dir = Path(whisper_dir)
        self.tok = None
        self.cached: tuple[Path, dict[str, str | None]] | None = None

    def part(self, source: str, input_name: str) -> Path:
        return self.dir / source / Path(input_name).name

    def load(self, part: Path) -> dict[str, str | None]:
        if self.cached is None or self.cached[0] != part:
            if self.tok is None:
                from transformers import AutoTokenizer

                self.tok = AutoTokenizer.from_pretrained(WHISPER_TOK, revision=WHISPER_TOK_REVISION)
            t = pq.read_table(part, columns=["name", "whisper_transcript"])
            names, lists = t.column("name").to_pylist(), t.column("whisper_transcript").to_pylist()
            keep = [i for i, x in enumerate(lists) if x is not None]
            texts = self.tok.batch_decode([lists[i] for i in keep], skip_special_tokens=True) if keep else []
            m: dict[str, str | None] = dict.fromkeys(names)
            m.update((names[i], x.strip()) for i, x in zip(keep, texts))
            self.cached = (part, m)
        return self.cached[1]


def read_teacher_rows(teacher_jsonl: Path) -> list[dict]:
    with open(teacher_jsonl, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_parakeet_hyps(parakeet_jsonl: Path) -> dict[str, str]:
    """id -> the Parakeet TDT hypothesis (02p's jsonl, in the order of its npz); {} before the pass reached the stem."""
    if not parakeet_jsonl.is_file():
        return {}
    return {r["id"]: r["hyp"] for r in read_teacher_rows(parakeet_jsonl)}


def output_ids(path: Path) -> list[str] | None:
    """The ids of an existing output, or None when it cannot be read (a torn or foreign file)."""
    try:
        return [r["id"] for r in read_teacher_rows(path)]
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Verified:
    """<out>/_cache/verified.txt: the "<source>/<stem>" outputs whose ids were checked against the teacher jsonl, so
    a periodic run does not re-read every finished output. Local only (label_sync never uploads _cache)."""

    def __init__(self, out_root: Path):
        self.path = out_root / "_cache" / VERIFIED_CACHE
        self.keys = set(self.path.read_text(encoding="utf-8").split()) if self.path.is_file() else set()

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def add(self, key: str):
        if key in self.keys:
            return
        self.keys.add(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(key + "\n")


def write_meta_once(out_root: Path, meta: dict):
    """--strict-existing: meta.json is written once (atomically); a run with other settings exits 65 rather than mix
    two kinds of second opinion under one root."""
    path = out_root / "meta.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        diff = [k for k in META_KEYS if old.get(k) != meta.get(k)]
        if diff:
            integrity(f"{path} was written with other settings: "
                      + "; ".join(f"{k} {old.get(k)!r} != {meta.get(k)!r}" for k in diff))
        return
    tmp = path.with_name("meta.json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    fsync_path(tmp)
    tmp.replace(path)


def write_shard(out_jsonl: Path, rows: list[dict]):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fsync_path(tmp)
    tmp.replace(out_jsonl)


def second_row(id: str, teacher_hyp: str, ref: str, hyp2: str | None, model2: str | None) -> dict:
    if hyp2 is None:
        return dict(id=id, hyp2=None, model2=None, agree=None, cer2=None)
    return dict(id=id, hyp2=hyp2, model2=model2,
                agree=round(cer_fn(teacher_hyp, hyp2), 4), cer2=round(cer_fn(hyp2, ref), 4))


def process_join_shard(trows: list[dict], wmap: dict[str, str | None], source: str,
                       fallback: dict[str, str] | None = None) -> list[dict]:
    """`fallback` (id -> Parakeet hyp): a row whose transcript is null or missing takes the Parakeet hypothesis
    instead of no second opinion; one without a Parakeet row keeps none (main checks the coverage)."""
    out = []
    for r in trows:
        name = r["id"].split("/", 1)[1]  # "<source>/<name>" -> "<name>" as in the mirror
        hyp2 = wmap.get(name)
        if hyp2 is not None:
            out.append(second_row(r["id"], r["hyp"], r["ref"], hyp2, "whisper-large-v3"))
        elif fallback is not None and r["id"] in fallback:
            out.append(second_row(r["id"], r["hyp"], r["ref"], fallback[r["id"]], FALLBACK_MODEL2))
        else:
            out.append(second_row(r["id"], r["hyp"], r["ref"], None, None))
    return out


def process_parakeet_shard(trows: list[dict], phyps: dict[str, str]) -> list[dict]:
    """--judge parakeet: hyp2 is the Parakeet TDT hypothesis of the same id; a row without one gets none."""
    return [second_row(r["id"], r["hyp"], r["ref"], phyps.get(r["id"]), PARAKEET_MODEL2 if r["id"] in phyps else None)
            for r in trows]


class Kotoba:
    """Lazy-loaded kotoba-whisper for the sources without precomputed transcripts."""

    def __init__(self, batch: int):
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.torch = torch
        self.batch = batch
        self.processor = AutoProcessor.from_pretrained(MODEL2, revision=MODEL2_REVISION)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL2, revision=MODEL2_REVISION,
                                                                     dtype=torch.float16).to("cuda").eval()
        print(f"{MODEL2} loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B params, "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    def featurise(self, audios: list[np.ndarray]):
        return self.processor(audios, sampling_rate=TARGET_SR, return_tensors="pt", return_attention_mask=True)

    def transcribe(self, inputs) -> list[str]:
        torch = self.torch
        from transformers.generation import StoppingCriteriaList

        with torch.inference_mode():
            out = self.model.generate(
                input_features=inputs["input_features"].to("cuda", torch.float16),
                attention_mask=inputs["attention_mask"].to("cuda"),
                language="ja", task="transcribe", num_beams=1, max_new_tokens=196,
                # prompt is <|startoftranscript|><|ja|><|transcribe|><|notimestamps|>
                stopping_criteria=StoppingCriteriaList([RepetitionStop(prompt_len=4)]),
            )
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]


def process_gpu_shard(kotoba: Kotoba, data_shard: Path, trows: list[dict], pool: ThreadPoolExecutor) -> list[dict]:
    table = read_shard(data_shard, columns=["id", "audio"])
    idx = {id_: i for i, id_ in enumerate(table.column("id").to_pylist())}
    audio_col = table.column("audio")

    todo = []  # (trow, audio row) pairs that decode; the rest get a null second opinion
    out_by_id: dict[str, dict] = {}
    for r in trows:
        i = idx.get(r["id"])
        if i is None:
            out_by_id[r["id"]] = second_row(r["id"], r["hyp"], r["ref"], None, None)
        else:
            todo.append((r, i))

    def prepare(chunk):
        audios, rows = [], []
        for r, i in chunk:
            try:
                audios.append(decode_audio(audio_col[i].as_py()))
                rows.append(r)
            except Exception as e:
                tqdm.write(f"  bad audio, skipping {r['id']}: {e}")
                out_by_id[r["id"]] = second_row(r["id"], r["hyp"], r["ref"], None, None)
        return (kotoba.featurise(audios) if audios else None), rows

    chunks = [todo[i:i + kotoba.batch] for i in range(0, len(todo), kotoba.batch)]
    prefetch = 2
    futures = [pool.submit(prepare, c) for c in chunks[:prefetch]]
    for j, _ in enumerate(tqdm(chunks, desc=data_shard.stem, unit="batch", leave=False)):
        inputs, rows = futures[j].result()
        if j + prefetch < len(chunks):
            futures.append(pool.submit(prepare, chunks[j + prefetch]))
        futures[j] = None
        if inputs is None:
            continue
        for r, hyp2 in zip(rows, kotoba.transcribe(inputs)):
            out_by_id[r["id"]] = second_row(r["id"], r["hyp"], r["ref"], hyp2, MODEL2)
    return [out_by_id[r["id"]] for r in trows]


def step_completed(root: Path, step: str) -> bool:
    """True once 01 --extent-config recorded `step` as complete: from then on every whisper part of it must exist."""
    progress = extent.read_progress(root) or {}
    return step in progress.get("completed", [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--teacher-out", default=str(ROOT / "teacher_out"))
    ap.add_argument("--out", default=str(ROOT / "second_out"))
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=16, help="kotoba batch size (whisper pads everything to 30 s)")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--judge", choices=["kotoba", "parakeet"], default="kotoba",
                    help="second opinion of the sources without a transcript of their own (galgame)")
    ap.add_argument("--parakeet-out", default=None, help="02p's output root (the judge, and the join fallback)")
    ap.add_argument("--whisper-dir", default=None, help="the whisper parts 01 --whisper-dir captured (join sources)")
    ap.add_argument("--require-npz", action="store_true", help="eligible once the npz exist, not the jsonl")
    ap.add_argument("--strict-existing", action="store_true",
                    help="exit 65 on an output with other ids, missing Parakeet rows or another meta.json")
    args = ap.parse_args()
    if args.judge == "parakeet" and not args.parakeet_out:
        ap.error("--judge parakeet needs --parakeet-out")

    root, teacher_root, out_root = Path(args.data), Path(args.teacher_out), Path(args.out)
    parakeet_root = Path(args.parakeet_out) if args.parakeet_out else None
    shards = read_manifest(root)
    if args.split != "all":
        shards = [s for s in shards if s.split == args.split]
    if args.sources:
        shards = [s for s in shards if s.source in args.sources]

    # --require-npz: 02 and 02p write the jsonl and then the npz, so the npz marks a finished shard; a jsonl alone
    # may be one that is still being written (the 02/02b race)
    done_suffix = ".npz" if args.require_npz else ".jsonl"

    def ready(s) -> bool:
        stem = Path(s.path).stem
        if not (teacher_root / s.source / f"{stem}{done_suffix}").exists():
            return False
        return parakeet_root is None or (parakeet_root / s.source / f"{stem}{done_suffix}").exists()

    # --limit-shards slices the ELIGIBLE list (shards with teacher output), so the supervisor's blocking-mode
    # "--limit-shards <done>+1" reaches exactly the stuck shard even when teacher coverage has gaps - but only when the
    # outputs it counts under --watch-dir are a prefix of this list, i.e. --watch-dir second_out/<source> with
    # --sources <source>. Counting all of second_out also counts sources finished out of manifest order (emilia_yodas
    # after galgame), and the "one shard" blocking attempt then runs everything left, serialised.
    eligible = [s for s in shards if ready(s)]
    no_teacher = len(shards) - len(eligible)
    if args.limit_shards:
        eligible = eligible[: args.limit_shards]

    # done by ids: an existing output counts only when its ids are the teacher jsonl's, in order
    verified = Verified(out_root)

    def done(s) -> bool:
        if args.force:
            return False
        stem = Path(s.path).stem
        out, key = out_root / s.source / f"{stem}.jsonl", f"{s.source}/{stem}"
        if not out.exists():
            return False
        if key in verified:
            return True
        tjsonl = teacher_root / s.source / f"{stem}.jsonl"
        if not tjsonl.is_file():
            integrity(f"{tjsonl} is missing next to its npz")
        if output_ids(out) == [r["id"] for r in read_teacher_rows(tjsonl)]:
            verified.add(key)
            return True
        if args.strict_existing:
            integrity(f"{out} exists with other ids than {tjsonl}; the label root is write-once, nothing was redone")
        tqdm.write(f"{key}: the existing output has other ids than the teacher's; redoing it")
        return False

    todo = [s for s in eligible if not done(s)]
    print(f"output: {out_root}\n{len(todo)}/{len(eligible)} eligible shards to do "
          f"({no_teacher} without teacher output yet, {len(eligible) - len(todo)} already done)")
    if not todo:
        return

    out_root.mkdir(parents=True, exist_ok=True)
    if args.judge == "parakeet":
        meta = dict(join_model="whisper-large-v3 (precomputed)", tokenizer=WHISPER_TOK,
                    tokenizer_revision=WHISPER_TOK_REVISION, join_revisions=dict(JOIN_SOURCES.values()),
                    judges={"galgame": PARAKEET_MODEL2}, fallback=FALLBACK_MODEL2,
                    agree="cer(teacher_hyp, hyp2)", cer2="cer(hyp2, dataset_text)")
    else:
        meta = dict(join_model="whisper-large-v3 (precomputed)", gpu_model=MODEL2, gpu_model_revision=MODEL2_REVISION,
                    tokenizer=WHISPER_TOK, tokenizer_revision=WHISPER_TOK_REVISION,
                    join_revisions=dict(JOIN_SOURCES.values()),
                    agree="cer(teacher_hyp, hyp2)", cer2="cer(hyp2, dataset_text)")
        if parakeet_root is not None:
            meta["fallback"] = FALLBACK_MODEL2
    if args.strict_existing:
        write_meta_once(out_root, meta)
    else:
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    wmaps: dict[str, dict] = {}
    parts = WhisperParts(Path(args.whisper_dir)) if args.whisper_dir else None
    kotoba: Kotoba | None = None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for s in tqdm(todo, desc="shards", unit="shard"):
            stem = Path(s.path).stem
            key = f"{s.source}/{stem}"
            tjsonl = teacher_root / s.source / f"{stem}.jsonl"
            if not tjsonl.is_file():
                integrity(f"{tjsonl} is missing next to its npz")
            trows = read_teacher_rows(tjsonl)
            phyps = read_parakeet_hyps(parakeet_root / s.source / f"{stem}.jsonl") if parakeet_root else None
            n_fallback = 0
            if s.source in SELF_TEXT_SOURCES:
                rows = [second_row(r["id"], r["hyp"], r["ref"], r["ref"], SELF_TEXT_SOURCES[s.source]) for r in trows]
            elif s.source in JOIN_SOURCES:
                if parts is not None:
                    info = sidecar_meta(root, s) or {}
                    part = parts.part(s.source, info["input"]) if info.get("input") else None
                    if part is None or not part.is_file():
                        step = info.get("step") or s.source
                        what = f"{part}" if part is not None else f"the input of {s.path} (no id sidecar)"
                        if step_completed(root, step):
                            integrity(f"{key}: {what} is missing although step {step} is complete")
                        tqdm.write(f"{key}: {what} not captured yet; skipped until 01 has written it")
                        continue
                    wmap = parts.load(part)
                    # every row of this shard comes from the input its sidecar names: a name absent from that part
                    # is a capture or join bug on any count (a mirror null is present with a None value)
                    absent = [r["id"] for r in trows if r["id"].split("/", 1)[1] not in wmap]
                    if absent:
                        integrity(f"{key}: {len(absent)} rows are not in {part} (first {absent[0]}): "
                                  f"a capture or join bug")
                else:
                    if s.source not in wmaps:
                        wmaps[s.source] = load_whisper_map(s.source, out_root / "_cache")
                    wmap = wmaps[s.source]
                rows = process_join_shard(trows, wmap, s.source, fallback=phyps)
                n_fallback = sum(1 for r in rows if r["model2"] == FALLBACK_MODEL2)
                if phyps is not None:
                    # a fallback row, or one that wanted it and had no Parakeet row either
                    wanted = sum(1 for r in rows if r["model2"] != "whisper-large-v3")
                    if trows and wanted > max(FALLBACK_MAX * len(trows), FALLBACK_FLOOR):
                        integrity(f"{key}: {wanted}/{len(trows)} rows have no whisper transcript "
                                  f"(> {FALLBACK_MAX:.0%} and > {FALLBACK_FLOOR}): a capture bug, not the mirror's nulls")
            elif args.judge == "parakeet":
                rows = process_parakeet_shard(trows, phyps)
            else:
                if kotoba is None:
                    kotoba = Kotoba(args.batch)
                rows = process_gpu_shard(kotoba, root / s.path, trows, pool)
            if args.strict_existing and phyps is not None:
                # a judged or fallback row without a Parakeet row would become no_agree, which launch refuses
                missing = [r["id"] for r, t in zip(rows, trows) if r["hyp2"] is None
                           and s.source not in SELF_TEXT_SOURCES and t["id"] not in phyps]
                if missing:
                    integrity(f"{key}: {len(missing)} teacher rows have no Parakeet row (first {missing[0]})")
            write_shard(out_root / s.source / f"{stem}.jsonl", rows)
            verified.add(key)
            agrees = np.array([r["agree"] for r in rows if r["agree"] is not None], dtype=np.float32)
            n_null = sum(1 for r in rows if r["hyp2"] is None)
            fb = f", parakeet fallback {n_fallback} ({n_fallback / max(len(rows), 1):.1%})" if phyps is not None \
                and s.source in JOIN_SOURCES else ""
            tqdm.write(f"{s.source}/{stem}: {len(rows)} rows, no 2nd opinion {n_null}{fb}, "
                       f"agree mean {agrees.mean():.3f} med {np.median(agrees):.3f} >0.2 {(agrees > 0.2).mean():.1%}"
                       if len(agrees) else f"{s.source}/{stem}: {len(rows)} rows, all without 2nd opinion{fb}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
