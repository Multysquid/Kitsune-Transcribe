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
script is resumable and safe under scripts/run_teacher_pass.py:
  python scripts/run_teacher_pass.py --script scripts/02b_second_opinion.py --watch-dir second_out --done-suffix .jsonl

Usage:
  python scripts/02b_second_opinion.py                     # everything with teacher output
  python scripts/02b_second_opinion.py --sources eval_jsut --limit-shards 2 --out second_out_smoke
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

from kitsune.audio import TARGET_SR, decode_audio  # noqa: E402
from kitsune.generation import RepetitionStop  # noqa: E402
from kitsune.store import fsync_path, read_manifest, read_shard  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402

MODEL2 = "kotoba-tech/kotoba-whisper-v2.0"
WHISPER_TOK = "openai/whisper-large-v3"  # tokenizer that produced the precomputed transcripts
JOIN_SOURCES = {  # sources whose mirror parquets already carry whisper-large-v3 transcripts
    "reazon_small": "japanese-asr/whisper_transcriptions.reazonspeech.small",
    "reazon_medium": "japanese-asr/whisper_transcriptions.reazonspeech.medium",
}
SELF_TEXT_SOURCES = {  # sources whose dataset text is itself a second-model transcript
    "emilia_yodas": "whisper-medium (Emilia WhisperX)",
    "eval_emilia": "whisper-medium (Emilia WhisperX)",
}


def build_whisper_cache(source: str, repo: str, cache_dir: Path) -> Path:
    """Stream name + whisper_transcript columns from the mirror and cache the decoded text locally.
    One atomically-written part per mirror file, so a network failure only costs the file in flight."""
    part_dir = cache_dir / f"whisper_{source}"
    part_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, HfFileSystem
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WHISPER_TOK)
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    fs = HfFileSystem()
    for f in tqdm(files, desc=f"whisper cache {source}", unit="file"):
        part = part_dir / Path(f).name
        if part.exists():
            continue
        with fs.open(f"datasets/{repo}/{f}", "rb") as fh:
            t = pq.read_table(fh, columns=["name", "whisper_transcript"])
        texts = tok.batch_decode(t.column("whisper_transcript").to_pylist(), skip_special_tokens=True)
        tmp = part.with_suffix(".parquet.tmp")
        pq.write_table(pa.table({"name": t.column("name").to_pylist(), "text": [x.strip() for x in texts]}), tmp)
        fsync_path(tmp)
        tmp.replace(part)
    return part_dir


def load_whisper_map(source: str, cache_dir: Path) -> dict[str, str]:
    part_dir = build_whisper_cache(source, JOIN_SOURCES[source], cache_dir)
    out: dict[str, str] = {}
    for part in sorted(part_dir.glob("*.parquet")):
        t = pq.read_table(part)
        out.update(zip(t.column("name").to_pylist(), t.column("text").to_pylist()))
    return out


def read_teacher_rows(teacher_jsonl: Path) -> list[dict]:
    with open(teacher_jsonl, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def process_join_shard(trows: list[dict], wmap: dict[str, str], source: str) -> list[dict]:
    out = []
    for r in trows:
        name = r["id"].split("/", 1)[1]  # "<source>/<name>" -> "<name>" as in the mirror
        hyp2 = wmap.get(name)
        out.append(second_row(r["id"], r["hyp"], r["ref"], hyp2, "whisper-large-v3" if hyp2 is not None else None))
    return out


class Kotoba:
    """Lazy-loaded kotoba-whisper for the sources without precomputed transcripts."""

    def __init__(self, batch: int):
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.torch = torch
        self.batch = batch
        self.processor = AutoProcessor.from_pretrained(MODEL2)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL2, dtype=torch.float16).to("cuda").eval()
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
    args = ap.parse_args()

    root, teacher_root, out_root = Path(args.data), Path(args.teacher_out), Path(args.out)
    shards = read_manifest(root)
    if args.split != "all":
        shards = [s for s in shards if s.split == args.split]
    if args.sources:
        shards = [s for s in shards if s.source in args.sources]

    # --limit-shards slices the ELIGIBLE list (shards with teacher output), so the supervisor's blocking-mode
    # "--limit-shards <done>+1" always reaches exactly the stuck shard even when teacher coverage has gaps
    eligible = [s for s in shards if (teacher_root / s.source / f"{Path(s.path).stem}.jsonl").exists()]
    no_teacher = len(shards) - len(eligible)
    if args.limit_shards:
        eligible = eligible[: args.limit_shards]
    todo = [s for s in eligible
            if args.force or not (out_root / s.source / f"{Path(s.path).stem}.jsonl").exists()]
    print(f"output: {out_root}\n{len(todo)}/{len(eligible)} eligible shards to do "
          f"({no_teacher} without teacher output yet, {len(eligible) - len(todo)} already done)")
    if not todo:
        return

    out_root.mkdir(parents=True, exist_ok=True)
    meta = dict(join_model="whisper-large-v3 (precomputed)", gpu_model=MODEL2, tokenizer=WHISPER_TOK,
                agree="cer(teacher_hyp, hyp2)", cer2="cer(hyp2, dataset_text)")
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    wmaps: dict[str, dict] = {}
    kotoba: Kotoba | None = None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for s in tqdm(todo, desc="shards", unit="shard"):
            stem = Path(s.path).stem
            trows = read_teacher_rows(teacher_root / s.source / f"{stem}.jsonl")
            if s.source in SELF_TEXT_SOURCES:
                rows = [second_row(r["id"], r["hyp"], r["ref"], r["ref"], SELF_TEXT_SOURCES[s.source]) for r in trows]
            elif s.source in JOIN_SOURCES:
                if s.source not in wmaps:
                    wmaps[s.source] = load_whisper_map(s.source, out_root / "_cache")
                rows = process_join_shard(trows, wmaps[s.source], s.source)
            else:
                if kotoba is None:
                    kotoba = Kotoba(args.batch)
                rows = process_gpu_shard(kotoba, root / s.path, trows, pool)
            write_shard(out_root / s.source / f"{stem}.jsonl", rows)
            agrees = np.array([r["agree"] for r in rows if r["agree"] is not None], dtype=np.float32)
            n_null = sum(1 for r in rows if r["hyp2"] is None)
            tqdm.write(f"{s.source}/{stem}: {len(rows)} rows, no 2nd opinion {n_null}, "
                       f"agree mean {agrees.mean():.3f} med {np.median(agrees):.3f} >0.2 {(agrees > 0.2).mean():.1%}"
                       if len(agrees) else f"{s.source}/{stem}: {len(rows)} rows, all without 2nd opinion")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
