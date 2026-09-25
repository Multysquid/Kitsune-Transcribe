"""Write-once upload of the label root (labels/<name>/ in the data repo) with a lease, verified per directory.

Why it is not finish.py's upload_folder sync: the label box runs for many hours and its labels are unique until they
are on the Hub, so every 20 min it commits the files finished since the last sync, and nothing it commits may ever
replace a label file already there (a relaunched or second box, a bug, a mixed-up config). The rules (plan §2):
  1 prefix guard: every committed path starts with labels/ or label_runs/ (laptop roots, models/ and students/ are
    out of reach);
  2 write-once: the parent dirs of the candidates are listed first; absent -> added; present with the same size and
    hash -> recorded as verified; present with other content -> a conflict, except LEASE.json and the finalize
    artifacts (extent.json, selections/, reports/, provenance/) while COMPLETE.json is absent. Every non-conflicting
    file is still uploaded, then IntegrityError is raised: labels never pile up unsynced behind one conflict;
  3 sealed root: with COMPLETE.json on the Hub nothing is written (the lease included); a sync whose every finished
    file is already verified is a no-op success (the final --destroy), anything new or changed raises;
  4 no deletes, ever.
Commits are explicit create_commit batches of new paths only: an npz/jsonl pair is never split across commits (a
consumer treats npz present = finished), and a commit holds at most 200 files and 40 MB of text. After each sync the
touched dirs are listed again and every file is compared by size and hash (sha256 for LFS/Xet files, the git blob id
for small ones); what matched goes into the ledger ($KITSUNE_STATE/label_ledger.json), so the next sync skips it.

Ledger: {"remote": {path: [size, sha256 or null, blob_id or null]}, "hashes": {"<path> <size> <mtime_ns>":
[sha256 or null, blob_id or null]}}, written to a .tmp, fsynced, renamed.

Stdlib only at import; huggingface_hub is imported lazily, and the Hub helpers (hub_retry, sha256_file, git_blob_id,
hf_api) are finish.py's.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

LABEL_PREFIXES = ("labels/", "label_runs/")
PAIRED = ("teacher_out/", "parakeet_out/")  # <stem>.npz + <stem>.jsonl, npz present = finished
FINAL_ARTIFACTS = ("extent.json", "selections/", "reports/", "provenance/")
LEASE_FILE = "LEASE.json"
COMPLETE_FILE = "COMPLETE.json"
REPO_TYPE = "dataset"
SKIP_SUFFIXES = (".tmp", ".partial", ".lock")
TEXT_SUFFIXES = (".jsonl", ".json")
COMPLETE_SCHEMA = 1
LEASE_MAX_AGE_S = 2700  # a lease heartbeat younger than 45 min is live
LOCK_POLL_S = 5


class IntegrityError(RuntimeError):
    """A write the write-once contract refuses (a guard violation, a conflict, a sealed root)."""


class SyncError(RuntimeError):
    """An upload or its verification failed; what is on disk is intact and the next sync tries again."""


def _fin():
    """finish.py's module (its Hub helpers), imported on first use: finish imports this module lazily too."""
    mod = sys.modules.get("finish")
    if mod is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import finish as mod
    return mod


def _parent(path: str) -> str:
    return path.rsplit("/", 1)[0]


def guard(path: str):
    """Rule 1: only labels/ and label_runs/ are ever written."""
    if not path.startswith(LABEL_PREFIXES) or ".." in path.split("/"):
        raise IntegrityError(f"refusing to write {path!r}: outside {' / '.join(LABEL_PREFIXES)}")


def is_final_artifact(prefix: str, path: str) -> bool:
    rel = path[len(prefix) + 1:] if path.startswith(prefix + "/") else path
    return rel == FINAL_ARTIFACTS[0] or rel.startswith(FINAL_ARTIFACTS[1:])


# ----------------------------------------------------------------------------------------------------------- files

def _skipped(parts: list[str]) -> bool:
    for p in parts:
        stem = p.rsplit(".", 1)[0]
        if p.startswith(".") or "_cache" in p or p.endswith("_smoke") or stem.endswith("_smoke"):
            return True
    return parts[-1].endswith(SKIP_SUFFIXES)


def finished_files(local_root: Path, prefix: str) -> dict[str, Path]:
    """repo path -> local file for every finished file under local_root/prefix (local_root mirrors the repo root):
    teacher_out/parakeet_out pairs whose npz exists (and whose jsonl does: the writer puts the jsonl first, so an npz
    alone is broken and a jsonl alone is in flight), second_out/<s>/*.jsonl (never a _cache), every <dir>/meta.json and
    the finalize artifacts. Never *.tmp, *_smoke, COMPLETE.json (seal writes it) or LEASE.json (generated)."""
    prefix = prefix.strip("/")
    base = Path(local_root) / prefix
    out: dict[str, Path] = {}
    if not base.is_dir():
        return out
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(base).as_posix()
        parts = rel.split("/")
        if _skipped(parts) or rel in (COMPLETE_FILE, LEASE_FILE):
            continue
        path = f"{prefix}/{rel}"
        top = parts[0] + "/"
        if len(parts) == 2 and parts[1] == "meta.json":
            out[path] = f
        elif top in PAIRED:
            if f.suffix == ".npz" and len(parts) >= 3 and f.with_suffix(".jsonl").is_file():
                out[path] = f
                out[path[:-len(".npz")] + ".jsonl"] = f.with_suffix(".jsonl")
        elif top == "second_out/":
            if len(parts) >= 3 and f.suffix == ".jsonl":
                out[path] = f
        elif is_final_artifact(prefix, path):
            out[path] = f
    return dict(sorted(out.items()))


def _unit(path: str) -> str:
    """The commit unit of a path: an npz/jsonl pair shares one."""
    if any(f"/{d}" in path for d in PAIRED) and path.endswith((".npz", ".jsonl")):
        return path.rsplit(".", 1)[0]
    return path


def commit_batches(items: list[tuple[str, Path]], max_files: int = 200,
                   max_text_bytes: int = 40_000_000) -> list[list[tuple[str, Path]]]:
    """Split (path, local) items into commits of at most max_files files and max_text_bytes of .jsonl/.json; a pair
    is never split (a unit larger than a limit goes alone)."""
    units: dict[str, list[tuple[str, Path]]] = {}
    for path, local in items:
        units.setdefault(_unit(path), []).append((path, local))
    batches, cur, text = [], [], 0
    for unit in units.values():
        u_text = sum(Path(local).stat().st_size for path, local in unit if path.endswith(TEXT_SUFFIXES))
        if cur and (len(cur) + len(unit) > max_files or text + u_text > max_text_bytes):
            batches.append(cur)
            cur, text = [], 0
        cur = cur + unit
        text += u_text
    if cur:
        batches.append(cur)
    return batches


# ----------------------------------------------------------------------------------------------------------- lease

def lease_bytes(run_id: str, container_id: str, machine_id: str, sha: str, released: bool,
                now: float | None = None) -> bytes:
    lease = {"run_id": run_id, "container_id": str(container_id), "machine_id": str(machine_id or ""),
             "kitsune_sha": sha or "", "heartbeat": time.time() if now is None else now, "released": bool(released)}
    return (json.dumps(lease, indent=1, sort_keys=True) + "\n").encode("utf-8")


def lease_live(lease: dict | None, container_id: str, now: float, max_age_s: int = LEASE_MAX_AGE_S) -> bool:
    """True when the lease belongs to another container, is not released and its heartbeat is younger than
    max_age_s. A lease with an unreadable heartbeat counts as live (--steal-lease overrides)."""
    if not isinstance(lease, dict) or lease.get("released") is True:
        return False
    if str(lease.get("container_id")) == str(container_id):
        return False
    try:
        return now - float(lease["heartbeat"]) < max_age_s
    except (KeyError, TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------------------------------------- ledger

def load_ledger(path: Path) -> dict:
    try:
        led = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(led, dict) and isinstance(led.get("remote"), dict) and isinstance(led.get("hashes"), dict):
            return led
    except (OSError, ValueError):
        pass
    return {"remote": {}, "hashes": {}}


def save_ledger(path: Path, ledger: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(ledger, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def seed_ledger(ledger_path: Path, listing) -> int:
    """Record remote files (RepoFile-like items, e.g. step plan's recursive listing) in the ledger; the next sync
    still hashes each local file once before treating it as verified. Returns the number recorded."""
    ledger = load_ledger(ledger_path)
    n = 0
    for item in listing:
        if getattr(item, "size", None) is not None:
            ledger["remote"][item.path] = remote_entry(item)
            n += 1
    save_ledger(ledger_path, ledger)
    return n


def remote_entry(item) -> list:
    lfs = getattr(item, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    return [item.size, sha or None, getattr(item, "blob_id", None) or None]


def _matches(entry: list, ledger: dict, path: str, local: Path, check_hash: bool = True) -> bool:
    """entry ([size, sha256, blob_id] of the remote file) describes this local file; local hashes are cached in
    the ledger by (path, size, mtime_ns)."""
    size, sha, blob = entry
    st = local.stat()
    if size != st.st_size:
        return False
    if not check_hash or not (sha or blob):
        return True
    cache = ledger["hashes"].setdefault(f"{path} {st.st_size} {st.st_mtime_ns}", [None, None])
    if sha:
        if cache[0] is None:
            cache[0] = _fin().sha256_file(local)
        return cache[0] == sha
    if cache[1] is None:
        cache[1] = _fin().git_blob_id(local)
    return cache[1] == blob


def _local_sha256(ledger: dict, path: str, local: Path) -> str:
    st = local.stat()
    cache = ledger["hashes"].setdefault(f"{path} {st.st_size} {st.st_mtime_ns}", [None, None])
    if cache[0] is None:
        cache[0] = _fin().sha256_file(local)
    return cache[0]


def _verified(ledger: dict, path: str, local: Path) -> bool:
    entry = ledger["remote"].get(path)
    return entry is not None and _matches(entry, ledger, path, local)


# ------------------------------------------------------------------------------------------------------------- hub

def _not_found(e: Exception) -> bool:
    if any(c.__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError") for c in type(e).__mro__):
        return True
    return getattr(getattr(e, "response", None), "status_code", None) == 404


def list_dir(api, repo: str, path: str) -> dict:
    """repo path -> RepoFile for the files directly in `path` (folders skipped); a dir not on the Hub is empty.
    Retried by finish.hub_retry (the whole listing: its error comes while its pages are iterated)."""
    def listing():
        try:
            return {item.path: item for item in api.list_repo_tree(repo, path_in_repo=path, recursive=False,
                                                                  repo_type=REPO_TYPE)
                    if getattr(item, "size", None) is not None}
        except Exception as e:
            if _not_found(e):
                return {}
            raise
    return _fin().hub_retry(listing, f"listing {path}")


def _commit(api, repo: str, ops: list, message: str):
    _fin().hub_retry(lambda: api.create_commit(repo_id=repo, repo_type=REPO_TYPE, operations=ops,
                                               commit_message=message), message)


def _remote_json(api, repo: str, path: str) -> dict | None:
    """A small json file on the Hub, or None when it cannot be read (a lease that cannot be read is not live)."""
    try:
        f = api.hf_hub_download(repo_id=repo, filename=path, repo_type="dataset")
        return json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def sync(api, repo: str, local_root: Path, prefix: str, ledger_path: Path, *, lease: bytes | None,
         dry_run: bool = False, log=print) -> dict:
    """Upload every finished file under local_root/prefix not yet verified, write-once (module docstring); the lease
    rides in the last commit (alone when nothing else is new). Returns counts; raises IntegrityError (refused writes,
    after the rest went up) or SyncError (an upload or its verification failed)."""
    from huggingface_hub import CommitOperationAdd

    prefix = prefix.strip("/")
    guard(f"{prefix}/{LEASE_FILE}")
    local_root = Path(local_root)
    ledger = load_ledger(ledger_path)
    files = finished_files(local_root, prefix)
    for p in files:
        guard(p)
    listings = {prefix: list_dir(api, repo, prefix)}
    sealed = f"{prefix}/{COMPLETE_FILE}" in listings[prefix]
    if lease is not None and f"{prefix}/{LEASE_FILE}" in listings[prefix]:
        own = json.loads(lease.decode("utf-8")).get("container_id")
        remote_lease = _remote_json(api, repo, f"{prefix}/{LEASE_FILE}")
        if lease_live(remote_lease, own, time.time()):
            raise IntegrityError(f"{prefix}/{LEASE_FILE} is live for another box ({remote_lease}): this box "
                                 f"(container {own}) never writes that root")
    candidates = {p: f for p, f in files.items() if not _verified(ledger, p, f)}
    for d in sorted({_parent(p) for p in candidates}):
        if d not in listings:
            listings[d] = list_dir(api, repo, d)
    add, conflicts, same = [], [], 0
    for p, f in candidates.items():
        remote = listings[_parent(p)].get(p)
        if remote is None:
            add.append((p, f))
        elif _matches(remote_entry(remote), ledger, p, f):
            ledger["remote"][p] = remote_entry(remote)
            same += 1
        elif not sealed and is_final_artifact(prefix, p):
            add.append((p, f))
        else:
            conflicts.append(f"{p}: the Hub holds other content ({remote.size} B there, {f.stat().st_size} B here)")
    bad_units = {_unit(c.split(": ", 1)[0]) for c in conflicts}
    held = [(p, f) for p, f in add if _unit(p) in bad_units]  # never half a pair
    add = [(p, f) for p, f in add if _unit(p) not in bad_units]
    result = {"files": len(files), "verified_before": len(files) - len(candidates), "same": same, "new": len(add),
              "conflicts": conflicts[:20], "held": len(held), "sealed": sealed, "commits": 0, "uploaded": 0}
    if sealed:
        save_ledger(ledger_path, ledger)
        if add or held or conflicts:
            why = [p for p, _ in add + held][:20] + conflicts[:20]
            raise IntegrityError(f"{prefix} is sealed ({COMPLETE_FILE} on the Hub): refusing to write "
                                 f"{len(add) + len(held)} new and {len(conflicts)} changed file(s): {why}")
        log(f"sync: {prefix} is sealed and every finished file is verified; nothing to do")
        return result

    batches = commit_batches(add)
    lease_op = (CommitOperationAdd(path_in_repo=f"{prefix}/{LEASE_FILE}", path_or_fileobj=lease)
                if lease is not None else None)
    log(f"sync {prefix} -> {repo}: {len(files)} finished, {len(add)} new in {len(batches)} commit(s), {same} already "
        f"there, {len(conflicts)} conflict(s){', lease' if lease is not None else ''}")
    if dry_run:
        save_ledger(ledger_path, ledger)
        return result
    failures, done = [], []
    plan = batches or ([[]] if lease_op is not None else [])
    for i, batch in enumerate(plan):
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for p, f in batch]
        if i == len(plan) - 1 and lease_op is not None:
            ops.append(lease_op)
        try:
            _commit(api, repo, ops, f"label sync: {len(batch)} file(s)" + (f" ({i + 1}/{len(plan)})" if len(plan) > 1
                                                                           else ""))
            result["commits"] += 1
            done.extend(batch)
        except Exception as e:  # noqa: BLE001  (the next batch may still go up; raised below)
            failures.append(f"commit {i + 1}/{len(plan)} ({len(batch)} files): {type(e).__name__}: {e}")
            log(f"sync: {failures[-1]}")
    fresh: dict[str, dict] = {}
    for p, f in done:
        d = _parent(p)
        if d not in fresh:
            try:
                fresh[d] = list_dir(api, repo, d)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{d}: cannot list after the commit: {type(e).__name__}: {e}")
                fresh[d] = {}
        remote = fresh[d].get(p)
        if remote is not None and _matches(remote_entry(remote), ledger, p, f):
            ledger["remote"][p] = remote_entry(remote)
            result["uploaded"] += 1
        else:
            failures.append(f"{p}: not verified after its commit")
    save_ledger(ledger_path, ledger)
    log(f"sync: {result['uploaded']} uploaded and verified, {len(failures)} failure(s)")
    if conflicts:
        raise IntegrityError(f"write-once conflicts in {prefix} (the rest was uploaded): {conflicts[:20]}")
    if failures:
        raise SyncError(f"{len(failures)} sync failure(s): {failures[:20]}")
    return result


def verify(api, repo: str, expected: dict[str, Path], ledger_path: Path, check_hash: bool = True) -> list[str]:
    """Compare expected local files with the Hub, one non-recursive listing per parent dir. Returns problems; what
    matched is recorded in the ledger (local hashes cached by path, size and mtime)."""
    ledger = load_ledger(ledger_path)
    problems, listings = [], {}
    for p, f in sorted(expected.items()):
        d = _parent(p)
        if d not in listings:
            try:
                listings[d] = list_dir(api, repo, d)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{d}: cannot list the repo: {type(e).__name__}: {e}")
                listings[d] = None
        if listings[d] is None:
            continue
        remote = listings[d].get(p)
        size = f.stat().st_size
        if remote is None:
            problems.append(f"{p}: missing in {repo}")
        elif remote.size != size:
            problems.append(f"{p}: size {remote.size} in repo, {size} local")
        elif not _matches(remote_entry(remote), ledger, p, f, check_hash):
            problems.append(f"{p}: hash differs")
        else:
            ledger["remote"][p] = remote_entry(remote)
    save_ledger(ledger_path, ledger)
    return problems


def seal(api, repo: str, local_root: Path, prefix: str, info: dict, ledger_path: Path) -> None:
    """Write COMPLETE.json (and the released lease) in one last commit once every finished file is verified on the
    Hub, then verify it. info: name, run_ids, kitsune_sha, image, configs, and optionally lease (the released lease
    bytes) or run_id. Re-runnable: a COMPLETE.json already there equal to the local one is success; any other is an
    IntegrityError."""
    from huggingface_hub import CommitOperationAdd

    prefix = prefix.strip("/")
    local_root = Path(local_root)
    path = f"{prefix}/{COMPLETE_FILE}"
    guard(path)
    local = local_root / prefix / COMPLETE_FILE
    root = list_dir(api, repo, prefix)
    ledger = load_ledger(ledger_path)
    if path in root:
        if local.is_file() and _matches(remote_entry(root[path]), ledger, path, local):
            ledger["remote"][path] = remote_entry(root[path])
            save_ledger(ledger_path, ledger)
            return
        raise IntegrityError(f"{path} is already on the Hub and is not this run's")
    files = finished_files(local_root, prefix)
    problems = verify(api, repo, files, ledger_path) if files else [f"nothing to seal under {local_root / prefix}"]
    extent = local_root / prefix / FINAL_ARTIFACTS[0]
    if not extent.is_file():
        problems.append(f"{extent} is missing")
    if problems:
        raise SyncError(f"refusing to seal {prefix}: {len(problems)} problem(s): {problems[:20]}")
    ledger = load_ledger(ledger_path)
    lines = "\n".join(sorted(f"{p} {_local_sha256(ledger, p, f)}" for p, f in files.items()))
    run_ids = list(info.get("run_ids") or [])
    complete = {"schema": COMPLETE_SCHEMA, "name": info.get("name"), "run_ids": run_ids,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kitsune_sha": info.get("kitsune_sha"),
                "image": info.get("image"), "extent_sha256": _local_sha256(ledger, f"{prefix}/extent.json", extent),
                "configs": info.get("configs"), "files": len(files),
                "files_digest": hashlib.sha256(lines.encode("utf-8")).hexdigest()}
    tmp = local.with_name(local.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(complete, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(local)
    lease = info.get("lease") or lease_bytes(info.get("run_id") or (run_ids[-1] if run_ids else "unknown"),
                                             os.environ.get("CONTAINER_ID", "local"),
                                             os.environ.get("KITSUNE_MACHINE_ID", ""), info.get("kitsune_sha") or "",
                                             released=True)
    _commit(api, repo, [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local)),
                        CommitOperationAdd(path_in_repo=f"{prefix}/{LEASE_FILE}", path_or_fileobj=lease)],
            f"label seal: {prefix}")
    remote = list_dir(api, repo, prefix).get(path)
    if remote is None or not _matches(remote_entry(remote), ledger, path, local):
        save_ledger(ledger_path, ledger)
        raise SyncError(f"{path} not verified after its commit")
    ledger["remote"][path] = remote_entry(remote)
    save_ledger(ledger_path, ledger)


# ---------------------------------------------------------------------------------------------------------- misc

def run_id(state_dir: Path) -> str:
    """label.json's run_id, else <CONTAINER_ID or local>-unknown."""
    try:
        rid = json.loads((Path(state_dir) / "label.json").read_text(encoding="utf-8")).get("run_id")
        if isinstance(rid, str) and rid:
            return rid
    except (OSError, ValueError, AttributeError):
        pass
    return f"{os.environ.get('CONTAINER_ID') or 'local'}-unknown"


def acquire_sync_lock(path: Path, wait_s: float, poll_s: float = LOCK_POLL_S, sleep=time.sleep):
    """Exclusive flock on $STATE/sync.lock, polled up to wait_s: the returned file object holds it until closed or
    the process exits. None if it stayed busy. Without fcntl (Windows) there is nothing to guard: True."""
    try:
        import fcntl
    except ImportError:
        return True
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.monotonic() >= deadline:
                f.close()
                return None
            sleep(poll_s)
