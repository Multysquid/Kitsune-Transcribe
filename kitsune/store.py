"""Local dataset store: uniform parquet shards + a manifest + per-source ingest progress.

Layout (relative to the data root):
    shards/<source>/<split>-<NNNNN>.parquet   columns: id, source, split, audio(bytes), text, duration, sr
    shards/<source>/progress.json             {"finished_inputs": [...], "done": bool}  (ingest resume state)
    manifest.jsonl                            one line per shard: path, source, split, rows, hours

Audio bytes are kept in their original container (FLAC/OGG/MP3); decoding happens at read time.
A shard's manifest line is appended the moment the shard is flushed, so an interrupted ingest keeps its
finished shards; `progress.json` records which input files are complete so a re-run skips them.
"""
import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("split", pa.string()),
        ("audio", pa.binary()),
        ("text", pa.string()),
        ("duration", pa.float32()),
        ("sr", pa.int32()),
    ]
)

ROWS_PER_SHARD = 2048  # ~150-400 MB per shard; small enough for cheap random access at train time


def fsync_path(path: Path):
    """Flush a file's data to disk. A hard process kill after rename-over can otherwise leave the target
    full of NUL bytes on NTFS (metadata committed, data blocks never written) - observed in practice."""
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


@dataclass
class ShardInfo:
    path: str  # relative to data root, forward slashes
    source: str
    split: str
    rows: int
    hours: float


def manifest_path(root: Path) -> Path:
    return Path(root) / "manifest.jsonl"


def append_manifest(root: Path, shards: list[ShardInfo]):
    p = manifest_path(root)
    lead = ""
    if p.exists() and p.stat().st_size > 0:
        with open(p, "rb") as g:  # a torn line from a killed append must not swallow the next good line
            g.seek(-1, os.SEEK_END)
            if g.read(1) != b"\n":
                lead = "\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(lead)
        for s in shards:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_manifest(root: Path, shards: list[ShardInfo]):
    """Rewrite the whole manifest (used when a source is removed)."""
    p = manifest_path(root)
    tmp = p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for s in shards:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def read_manifest(root: Path) -> list[ShardInfo]:
    p = manifest_path(root)
    if not p.exists():
        return []
    out, seen, bad = [], set(), 0
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s = ShardInfo(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                bad += 1  # a process killed mid-append leaves a garbage line; its shard is re-ingested on resume
                continue
            if s.path not in seen:  # append-only file; tolerate a duplicated line
                seen.add(s.path)
                out.append(s)
    if bad:
        print(f"  manifest: skipped {bad} corrupt line(s); shard files not in the manifest are orphans "
              f"(safe to delete - their rows were re-ingested)")
    return out


def source_dir(root: Path, source: str) -> Path:
    return Path(root) / "shards" / source


def load_progress(root: Path, source: str) -> dict:
    p = source_dir(root, source) / "progress.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"finished_inputs": [], "done": False}


def save_progress(root: Path, source: str, progress: dict):
    d = source_dir(root, source)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "progress.json.tmp"
    tmp.write_text(json.dumps(progress, indent=1), encoding="utf-8")
    fsync_path(tmp)
    tmp.replace(d / "progress.json")


def remove_source(root: Path, source: str):
    """Drop a source completely: its shard directory (incl. progress) and its manifest lines."""
    d = source_dir(root, source)
    if d.exists():
        shutil.rmtree(d)
    write_manifest(root, [s for s in read_manifest(root) if s.source != source])


class ShardWriter:
    """Accumulates rows for one (source, split); every flushed shard is immediately added to the manifest."""

    def __init__(self, root: Path, source: str, split: str, rows_per_shard: int = ROWS_PER_SHARD):
        self.root = Path(root)
        self.source, self.split = source, split
        self.rows_per_shard = rows_per_shard
        self.dir = source_dir(self.root, source)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf: list[dict] = []
        self.shard_idx = 0
        self.written: list[ShardInfo] = []
        # resume: continue numbering after existing shards so a re-run never clobbers finished ones
        existing = sorted(self.dir.glob(f"{split}-*.parquet"))
        if existing:
            self.shard_idx = int(existing[-1].stem.split("-")[-1]) + 1

    def add(self, id: str, audio: bytes, text: str, duration: float, sr: int):
        self.buf.append(
            dict(id=id, source=self.source, split=self.split, audio=audio, text=text, duration=duration, sr=sr)
        )
        if len(self.buf) >= self.rows_per_shard:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        path = self.dir / f"{self.split}-{self.shard_idx:05d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self.buf, schema=SCHEMA)
        pq.write_table(table, tmp, compression="none", row_group_size=256)  # audio is already compressed
        fsync_path(tmp)
        tmp.replace(path)
        hours = sum(r["duration"] for r in self.buf) / 3600
        info = ShardInfo(str(path.relative_to(self.root)).replace("\\", "/"), self.source, self.split, len(self.buf), hours)
        append_manifest(self.root, [info])
        self.written.append(info)
        self.buf.clear()
        self.shard_idx += 1

    def close(self) -> list[ShardInfo]:
        self.flush()
        return self.written


def read_shard(path: Path, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(path, columns=columns)


def iter_rows(path: Path, columns: list[str] | None = None, batch_rows: int = 256) -> Iterator[dict]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield from batch.to_pylist()
