"""Local dataset store: uniform parquet shards + a manifest + per-source ingest progress.

Layout (relative to the data root):
    shards/<source>/<split>-<NNNNN>.parquet   columns: id, source, split, audio(bytes), text, duration, sr
    shards/<source>/_ids/<split>-<NNNNN>.parquet   id sidecar: columns id, duration; schema metadata
                                              b"kitsune_ids" = {schema, input, step, rows, ids_sha256}
    shards/<source>/progress.json             {"finished_inputs": [...], "done": bool, "n_listed": int,
                                               "input_bytes": {input: bytes}}  (ingest resume state)
    manifest.jsonl                            one line per shard: path, source, split, rows, hours

Audio bytes are kept in their original container (FLAC/OGG/MP3); decoding happens at read time.
A shard's manifest line is appended the moment the shard is flushed, so an interrupted ingest keeps its
finished shards; `progress.json` records which input files are complete so a re-run skips them.

The id sidecar names the upstream input file and the ingest step a shard came from, and keeps its ids and durations
once the shard's audio is deleted (the label box prunes labelled audio): cross-source reads go through `read_ids`, which
prefers the sidecar. Shards the laptop wrote before sidecars existed have none, and `read_ids` reads the shard instead.
"""
import hashlib
import json
import os
import re
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
SIDECAR_DIR = "_ids"  # a subdirectory, so the non-recursive <split>-*.parquet globs of the readers never see it
SIDECAR_KEY = b"kitsune_ids"
SIDECAR_COLUMNS = ("id", "duration")
SIDECAR_SCHEMA_VERSION = 1


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


def lock_data_root(root: Path):
    """Exclusive, non-blocking lock on <root>/.ingest.lock, held for as long as the returned file object stays open;
    None if another ingest holds it. Two ingests into one root overwrite each other's shards (each numbers a new shard
    after the manifest's, deletes the other's flushed but not yet listed shard as an orphan and flushes through the
    same .tmp), rewrite progress.json and manifest.jsonl over each other and delete downloads the other still reads -
    per root, not per source: the manifest and the raw HF cache are shared. The OS drops the lock when the process
    dies (crash, kill, power loss), so nothing is left to clear by hand, as a pid file would be (and os.kill(pid, 0) on
    Windows sends Ctrl+C instead of probing)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / ".ingest.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)  # msvcrt locks bytes from the current position: byte 0, whatever the file holds
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def remove_source(root: Path, source: str):
    """Drop a source completely: its shard directory (incl. progress) and its manifest lines."""
    d = source_dir(root, source)
    if d.exists():
        shutil.rmtree(d)
    write_manifest(root, [s for s in read_manifest(root) if s.source != source])


def ids_sha256(ids: list[str]) -> str:
    """Digest of a shard's ids in row order: two shards with the same digest hold the same utterances in the same
    order, whichever machine ingested them."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def sidecar_path(root: Path, info: ShardInfo) -> Path:
    p = Path(info.path)
    return Path(root) / p.parent / SIDECAR_DIR / p.name


def sidecar_meta(root: Path, info: ShardInfo) -> dict | None:
    """The sidecar's {schema, input, step, rows, ids_sha256}, or None for a shard without one (laptop data)."""
    side = sidecar_path(root, info)
    if not side.is_file():
        return None
    raw = (pq.read_schema(side).metadata or {}).get(SIDECAR_KEY)
    return json.loads(raw) if raw else None


def read_ids(root: Path, info: ShardInfo, columns=("id",)) -> list[dict]:
    """Rows of `columns` for a manifest shard, from its sidecar when it has one and the columns are in it, else from
    the shard itself. The sidecar outlives the shard's audio, so reads of other sources' ids (dedup, video exclusion,
    the Emilia budget) keep working after the label box has pruned the audio."""
    columns = list(columns)
    side, shard = sidecar_path(root, info), Path(root) / info.path
    if set(columns) <= set(SIDECAR_COLUMNS) and side.is_file():
        return pq.read_table(side, columns=columns).to_pylist()
    if not shard.is_file():
        raise FileNotFoundError(f"{info.path}: neither the shard ({shard}) nor its id sidecar ({side}) is on disk")
    return pq.read_table(shard, columns=columns).to_pylist()


def shard_ids(root: Path, info: ShardInfo) -> list[str]:
    return [r["id"] for r in read_ids(root, info)]


def _write_sidecar(root: Path, info: ShardInfo, rows: list[dict], meta: dict | None):
    ids = [r["id"] for r in rows]
    side = sidecar_path(root, info)
    side.parent.mkdir(parents=True, exist_ok=True)
    md = dict(schema=SIDECAR_SCHEMA_VERSION, input=(meta or {}).get("input"), step=(meta or {}).get("step"),
              rows=len(ids), ids_sha256=ids_sha256(ids))
    table = pa.table({"id": pa.array(ids, pa.string()),
                      "duration": pa.array([r["duration"] for r in rows], pa.float32())})  # float32 as in the shard
    tmp = side.with_suffix(".parquet.tmp")
    pq.write_table(table.replace_schema_metadata({SIDECAR_KEY: json.dumps(md).encode()}), tmp)
    fsync_path(tmp)
    tmp.replace(side)


def _stem_number(name: str, split: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(split)}-(\d+)\.parquet", name)
    return int(m.group(1)) if m else None


class ShardWriter:
    """Accumulates rows for one (source, split); every flushed shard is immediately added to the manifest.

    `meta` ({input, step}) is written into each flushed shard's id sidecar; the ingest updates it at every input."""

    def __init__(self, root: Path, source: str, split: str, rows_per_shard: int = ROWS_PER_SHARD,
                 meta: dict | None = None):
        self.root = Path(root)
        self.source, self.split = source, split
        self.rows_per_shard = rows_per_shard
        self.meta = meta
        self.dir = source_dir(self.root, source)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf: list[dict] = []
        self.shard_idx = 0
        self.written: list[ShardInfo] = []
        # resume: continue numbering after the manifest's shards so a re-run never clobbers finished ones. The files on
        # disk are no guide: a kill between the rename and the manifest append leaves an unlisted shard, which would
        # shift every later stem against a clean run, and a pruned shard is listed but no longer on disk.
        listed = {s.path for s in read_manifest(self.root) if s.source == source and s.split == split}
        nums = [n for n in (_stem_number(Path(p).name, split) for p in listed) if n is not None]
        if nums:
            self.shard_idx = max(nums) + 1
        self._remove_orphan()

    def _remove_orphan(self):
        """Delete the unlisted shard (and sidecar) numbered shard_idx: flushed by a run that was killed before its
        manifest line, so its rows are ingested again and it would be overwritten anyway. A kill leaves at most that
        one stem, as the next writer removes it before its first flush. An unlisted stem above it means the manifest no
        longer describes the disk (deleted, emptied or restored): those shards' inputs may be marked finished and never
        read again, so stop and delete nothing."""
        orphan, above = [], []
        for d in (self.dir, self.dir / SIDECAR_DIR):
            for f in sorted(d.glob(f"{self.split}-*.parquet")):
                n = _stem_number(f.name, self.split)
                if n is not None and n >= self.shard_idx:  # every listed stem is below shard_idx
                    (orphan if n == self.shard_idx else above).append(f.relative_to(self.root).as_posix())
        if above:
            raise SystemExit(f"  {self.source}: {', '.join(above)} not in manifest.jsonl, above "
                             f"{self.split}-{self.shard_idx:05d} (the only stem a killed flush leaves unlisted): the "
                             f"manifest does not describe the disk. Nothing was deleted; restore manifest.jsonl, or "
                             f"re-ingest the source with --force")
        for rel in orphan:
            (self.root / rel).unlink()
            print(f"  {self.source}: removed orphan {rel} (flushed but not in the manifest; its rows are ingested "
                  f"again)")

    def add(self, id: str, audio: bytes, text: str, duration: float, sr: int):
        self.buf.append(
            dict(id=id, source=self.source, split=self.split, audio=audio, text=text, duration=duration, sr=sr)
        )
        if len(self.buf) >= self.rows_per_shard:
            self.flush()

    def flush(self):
        """Shard (tmp, fsync, rename), then its id sidecar (the same), then the manifest line: a listed shard always
        has its sidecar, and a kill before the manifest append leaves only orphans the next writer removes."""
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
        _write_sidecar(self.root, info, self.buf, self.meta)
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
