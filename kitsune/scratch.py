"""Timed full states in the private scratch repo (the full runs' plan item 4a): the upload side.

A full-data run saves a full state every ckpt.upload_full_every_min minutes of loop clock (scripts/04_distill.py, reason
"timed") and sends it here, so a host that dies costs at most that much training instead of the run: a new host pulls
the newest state (kitsune.full_queue resume-pull) and resumes. The scratch repo is a PRIVATE model repo the owner
creates (Multy123/kitsune-scratch); nothing here creates a repo, so a typo cannot make a new one and a token scoped to
that repo is enough. Layout (kitsune.fullrun, contract 1.2 / 2.4), exactly one state per run:

  runs/<run_id>/checkpoints/full_step_<N>/{model.pt, optimizer.pt, l2sp.pt, trainer.pt, trainer.json[, aux_ctc.pt]}
  runs/<run_id>/timed_state.json          the pointer (format 1): step, epoch, planner fingerprint, train utterances,
                                          selection sha256, micro_audio_s, code sha, the saving host, and every
                                          committed file's size and sha256

  make_pointer(run_id, local_dir, *, step, ...)   the pointer of a local full_step_<N>/ dir: every state file in it
                                          (marker files never), hashed; fullrun.pointer_problems(pointer) == []
  upload_state(api, repo, run_id, local_dir, pointer, *, log, retries)   ONE create_commit that adds the state's files
                                          and the pointer and deletes every other runs/<run_id>/checkpoints/full_step_*
                                          folder (found with list_repo_tree: a run resumed on a new host does not know
                                          the old step), then super_squash_history, so the repo keeps one commit and
                                          the replaced states' storage is freed. A commit is atomic: the pointer and
                                          the files it names change together. Retried on the Hub's transient answers
                                          (RETRY_STATUS and 5xx, and errors without a status: a dropped connection)
                                          after RETRY_S; box 2's two trainers commit and squash the same repo, so a
                                          409/412 from a squash racing a commit is expected. A failed squash is
                                          recorded, not a failure: the next upload squashes again
  host_info(), code_sha(root)             the saving host ({hostname, machine_id, container_id}) and the code's git
                                          sha ($KITSUNE_SHA on a box), for the pointer and trainer.pt

Events (the trainer's RunLogger; the queue's smoke check 7 reads them): timed_state_upload_ok {name, step, attempt, gb,
upload_s, deleted, squash_ok}, timed_state_upload_error {name, step, attempt, status, error} per failed attempt,
timed_state_upload_failed {name, step, attempts, error} when every attempt failed or a failure cannot be retried.

The local side (scripts/04_distill.py ScratchUploader): the dir carries fullrun.SCRATCH_MARK while its upload is pending
(never the runs repo's UPLOAD_MARK, which vast/finish.py would treat as a state the runs repo must hold); the mark goes
when the upload ends, and a success also clears it from older states. Stdlib only at import (huggingface_hub inside
upload_state).
"""
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from kitsune import fullrun

RETRY_S = (15, 60, 180, 600)  # seconds before each retry of an upload (4 retries: ~14 min in all)
RETRY_STATUS = (408, 409, 412, 429)  # the Hub's transient answers, retried; so is every 5xx
_FULL_RE = re.compile(r"^full_step_(\d+)$")  # scripts/04_distill.py FULL_RE
_HASH_CHUNK = 8 << 20
_code_sha: dict[str, str | None] = {}  # checkout -> git HEAD (one subprocess per process)


class StateChanged(ValueError):
    """A state file's size differs from the pointer's: something rewrote the dir after make_pointer hashed it. Not
    retried (the pointer would name other bytes than the ones committed)."""


# ------------------------------------------------------------------------------------------------------ helpers


def host_info() -> dict:
    """The host a state is saved on: {hostname, machine_id ($KITSUNE_MACHINE_ID, the vast machine, set by launch),
    container_id ($CONTAINER_ID, the vast instance)}; the two ids are None off a box. A resume compares it with the
    state's (the `resume` event's new_host)."""
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = ""
    return dict(hostname=hostname, machine_id=os.environ.get(fullrun.ENV_MACHINE_ID) or None,
                container_id=os.environ.get("CONTAINER_ID") or None)


def code_sha(root=None) -> str | None:
    """The code's git sha: $KITSUNE_SHA (the sha launch rented the box at), else `git rev-parse HEAD` in root (the
    checkout; cached per process), else None."""
    env = os.environ.get(fullrun.ENV_SHA)
    if env:
        return env
    key = str(Path(root) if root is not None else fullrun.REPO)
    if key not in _code_sha:
        try:
            r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=key, capture_output=True, text=True, timeout=30)
            sha = r.stdout.strip()
            _code_sha[key] = sha if r.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None
        except (OSError, subprocess.SubprocessError):
            _code_sha[key] = None
    return _code_sha[key]


def file_sha256(path) -> str:
    """sha256 of a file, read in 8 MiB chunks (hashlib releases the GIL on them: the training loop keeps running)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def http_status(exc: BaseException) -> int | None:
    """The HTTP status of a huggingface_hub / httpx error (None: no response, e.g. a dropped connection)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def retryable(exc: BaseException) -> bool:
    """An attempt worth repeating: a transient Hub answer (RETRY_STATUS, 5xx) or an error without a status (network).
    Never a local error (a missing or changed state file, a bad argument) nor another status (401, 403, 404: a token
    without write access, a scratch repo that does not exist)."""
    status = http_status(exc)
    if status is not None:
        return status in RETRY_STATUS or status >= 500
    return not isinstance(exc, (ValueError, TypeError, FileNotFoundError, IsADirectoryError, PermissionError))


# ------------------------------------------------------------------------------------------------------ the pointer


def make_pointer(run_id: str, local_dir, *, step: int, epoch: float, planner_fingerprint: str, n_train_utts: int,
                 selection_sha256: str | None, micro_audio_s: float, kitsune_sha: str | None, host: dict) -> dict:
    """The pointer (contract 2.4, format 1) of the local full state `local_dir` = checkpoints/full_step_<step>/: files
    = every state file in it (fullrun.STATE_FILES_REQUIRED, and STATE_FILES_OPTIONAL when present) with its size and
    sha256, marker files never; exactly the files upload_state commits. ValueError when the dir is not full_step_<step>,
    lacks a required file, or the pointer would not pass fullrun.pointer_problems."""
    local_dir = Path(local_dir)
    name = f"full_step_{int(step)}"
    if local_dir.name != name:
        raise ValueError(f"{local_dir}: a timed state of step {step} is a {name}/ dir")
    files = {}
    for f in (*fullrun.STATE_FILES_REQUIRED, *fullrun.STATE_FILES_OPTIONAL):
        p = local_dir / f
        if p.is_file():
            files[f] = dict(size=p.stat().st_size, sha256=file_sha256(p))
    if missing := [f for f in fullrun.STATE_FILES_REQUIRED if f not in files]:
        raise ValueError(f"{local_dir}: not a complete full state (no {', '.join(missing)})")
    now = time.time()
    ptr = dict(format=fullrun.POINTER_FORMAT, run_id=str(run_id), name=name, step=int(step), epoch=float(epoch),
               wall=round(now, 3),
               time_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
               kitsune_sha=kitsune_sha, planner_fingerprint=str(planner_fingerprint), n_train_utts=int(n_train_utts),
               selection_sha256=selection_sha256, micro_audio_s=float(micro_audio_s), files=files,
               host=dict(hostname=str(host.get("hostname") or ""), machine_id=host.get("machine_id"),
                         container_id=host.get("container_id")))
    if problems := fullrun.pointer_problems(ptr):
        raise ValueError(f"{local_dir}: " + "; ".join(problems))
    return ptr


# ------------------------------------------------------------------------------------------------------ the upload


def previous_states(api, repo: str, run_id: str, keep: str) -> list[str]:
    """The repo paths of the run's timed state folders other than `keep` (runs/<run_id>/checkpoints/full_step_*, a
    non-recursive listing; a folder entry has no size). A run whose checkpoints/ folder does not exist yet (its first
    upload: a 404) has none."""
    base = f"runs/{run_id}/checkpoints"
    try:
        entries = list(api.list_repo_tree(repo, path_in_repo=base, repo_type="model"))
    except Exception as e:  # noqa: BLE001
        if http_status(e) == 404 or type(e).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError"):
            return []
        raise
    out = []
    for x in entries:
        path = str(getattr(x, "path", "")).rstrip("/")
        leaf = PurePosixPath(path).name
        if (PurePosixPath(path).parent.as_posix() == base and _FULL_RE.match(leaf) and leaf != keep
                and getattr(x, "size", None) is None):
            out.append(path)
    return sorted(out)


def _squash(api, repo: str, message: str, waits) -> tuple[bool, str | None]:
    """super_squash_history (the repo keeps one commit; replaced states' storage is freed), retried like an upload on
    the first two waits; (ok, the last error)."""
    err = None
    for attempt, wait in enumerate((0, *waits)):
        if wait:
            time.sleep(wait)
        try:
            api.super_squash_history(repo, repo_type="model", commit_message=message)
            return True, None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:500]
            if not retryable(e):
                break
    return False, err


def upload_state(api, repo: str, run_id: str, local_dir, pointer: dict, *, log=None, retries=RETRY_S) -> dict:
    """Upload the local full state `local_dir` as the run's one timed state: ONE create_commit that adds every file the
    pointer names under fullrun.scratch_state_dir(run_id, step), the pointer at fullrun.scratch_pointer(run_id), and
    deletes every other full_step_* folder of the run (CommitOperationDelete(is_folder=True)); then
    super_squash_history. Never create_repo, never a marker file. retries: the waits before each repeat of a failed
    attempt (retryable errors only). log: a logger with .event (the events in the module docstring), or None.
    Returns {ok, attempts, deleted (the replaced folder names), squash_ok (None before a commit went up), upload_s,
    gb[, error]}."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    local_dir = Path(local_dir)
    if problems := fullrun.pointer_problems(pointer):
        raise ValueError("; ".join(problems))
    if pointer["run_id"] != run_id:
        raise ValueError(f"pointer of run {pointer['run_id']!r} uploaded as {run_id!r}")
    name, step, files = pointer["name"], int(pointer["step"]), sorted(pointer["files"])
    dest = fullrun.scratch_state_dir(run_id, step)
    ptr_bytes = (json.dumps(pointer, indent=1, sort_keys=True) + "\n").encode()
    gb = round(sum(int(m["size"]) for m in pointer["files"].values()) / 1e9, 3)
    out = dict(ok=False, attempts=0, deleted=[], squash_ok=None, upload_s=None, gb=gb)

    def event(kind, **kw):
        if log is not None:
            log.event(kind, name=name, step=step, **kw)

    for attempt, wait in enumerate((0, *retries)):
        if wait:
            time.sleep(wait)
        out["attempts"] = attempt + 1
        t0 = time.time()
        try:
            changed = [f for f in files if (local_dir / f).stat().st_size != int(pointer["files"][f]["size"])]
            if changed:
                raise StateChanged(f"{local_dir}: {', '.join(changed)} changed size since the pointer was made")
            old = previous_states(api, repo, run_id, keep=name)
            ops = [CommitOperationAdd(path_in_repo=f"{dest}/{f}", path_or_fileobj=str(local_dir / f)) for f in files]
            ops.append(CommitOperationAdd(path_in_repo=fullrun.scratch_pointer(run_id), path_or_fileobj=ptr_bytes))
            ops += [CommitOperationDelete(path_in_repo=f"{p}/", is_folder=True) for p in old]
            api.create_commit(repo, operations=ops, commit_message=f"{run_id}: timed state {name}", repo_type="model")
        except Exception as e:  # noqa: BLE001  (every attempt is an event; the caller gets ok False, never a raise)
            out["error"] = f"{type(e).__name__}: {e}"[:2000]
            event("timed_state_upload_error", attempt=attempt, status=http_status(e), error=out["error"])
            if not retryable(e):
                break
            continue
        out.update(ok=True, upload_s=round(time.time() - t0, 1), deleted=[PurePosixPath(p).name for p in old])
        out.pop("error", None)
        out["squash_ok"], squash_err = _squash(api, repo, f"timed states (after {run_id} {name})", tuple(retries)[:2])
        event("timed_state_upload_ok", attempt=attempt, gb=gb, upload_s=out["upload_s"], deleted=out["deleted"],
              squash_ok=out["squash_ok"], **({"squash_error": squash_err} if squash_err else {}))
        return out
    event("timed_state_upload_failed", attempts=out["attempts"], error=out.get("error"))
    return out
