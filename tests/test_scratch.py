"""kitsune/scratch.py: the timed full states' pointer and their upload to the private scratch repo, against an in-memory
fake Hub (never the real one).

- make_pointer: every state file with its size and sha256, marker and stray files never, a pointer that passes
  kitsune.fullrun.pointer_problems; a dir that is not full_step_<step>/ or lacks a required file is refused
- upload_state: ONE create_commit that adds the state and the pointer and deletes the run's other full_step_* folders
  (another run's untouched), then a history squash; never create_repo; the first upload of a run (its folder 404s);
  retried on the Hub's transient answers only; a listing that never answers is a retried attempt (LIST_TIMEOUT_S); a
  failed squash is recorded, not a failure; a state rewritten after its pointer was made is not committed; the events

ScratchHub is also the fake HfApi of tests/test_full_trainer_state.py (the scratch and the runs repo at once: every repo
a dict path -> bytes)."""
import fnmatch
import hashlib
import json
import os
import sys
import threading
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from kitsune import fullrun, scratch  # noqa: E402

RUN = "full-p01-20260927T120000Z"
REPO = "owner/kitsune-scratch"


class HubError(Exception):
    """What huggingface_hub raises for an HTTP error: response.status_code is what the retry rule reads."""

    def __init__(self, status: int, msg: str = ""):
        super().__init__(f"{status} {msg}".strip())
        self.response = SimpleNamespace(status_code=status)


class ScratchHub:
    """The part of huggingface_hub.HfApi the trainer calls, in memory, for any number of repos (the scratch repo and
    the runs repo of a test run): create_commit with CommitOperationAdd / CommitOperationDelete(is_folder=True),
    super_squash_history, list_repo_tree (folders without a size, a 404 for a path with nothing under it),
    upload_folder / upload_file (the trainer's log syncs and checkpoint uploads, ignore_patterns as the hub matches
    them), hf_hub_download and create_repo (recorded: the scratch path must never call it).

    fail_commit / fail_squash: exceptions the next calls raise, in order. gate: a threading.Event every create_commit
    waits for (a slow upload). drop_syncs: log syncs of a run dir (upload_folder to runs/<id>, not its checkpoints)
    are recorded in `dropped` and not applied - a host that died before its next sync."""

    def __init__(self):
        self.repos: dict[str, dict[str, bytes]] = {}
        self.commits: list[dict] = []
        self.squashes: list[str] = []
        self.created: list[tuple] = []
        self.fail_commit: list[BaseException] = []
        self.fail_squash: list[BaseException] = []
        self.gate: threading.Event | None = None
        self.drop_syncs = False
        self.dropped: list[str] = []
        self.lock = threading.Lock()

    def files(self, repo: str, prefix: str = "") -> dict[str, bytes]:
        with self.lock:
            return {p: b for p, b in self.repos.get(repo, {}).items() if p.startswith(prefix)}

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False, **kw):
        self.created.append((repo_id, private))

    # --------------------------------------------------------------------------------------------- the scratch path

    def create_commit(self, repo_id, operations, *, commit_message=None, repo_type=None, **kw):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        if self.gate is not None:
            self.gate.wait(60)
        if self.fail_commit:
            raise self.fail_commit.pop(0)
        adds, deletes = {}, []
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                src = op.path_or_fileobj
                adds[op.path_in_repo] = src if isinstance(src, bytes) else Path(src).read_bytes()
            elif isinstance(op, CommitOperationDelete):
                deletes.append((op.path_in_repo, op.is_folder))
            else:
                raise TypeError(f"unexpected operation {op!r}")
        with self.lock:
            repo = self.repos.setdefault(repo_id, {})
            for path, folder in deletes:
                gone = [p for p in repo if (p.startswith(path.rstrip("/") + "/") if folder else p == path)]
                assert gone, f"delete of {path} matches nothing"
                for p in gone:
                    del repo[p]
            repo.update(adds)
            self.commits.append(dict(repo=repo_id, adds=sorted(adds), deletes=deletes, message=commit_message))

    def super_squash_history(self, repo_id, *, branch=None, commit_message=None, repo_type=None, **kw):
        if self.fail_squash:
            raise self.fail_squash.pop(0)
        self.squashes.append(repo_id)

    def list_repo_tree(self, repo_id, path_in_repo=None, *, recursive=False, repo_type=None, **kw):
        pre = (path_in_repo or "").rstrip("/")
        with self.lock:
            paths = sorted(p for p in self.repos.get(repo_id, {}) if not pre or p.startswith(pre + "/"))
            data = {p: self.repos[repo_id][p] for p in paths}
        if pre and not paths:
            raise HubError(404, f"{pre} not found")
        out, seen = [], set()
        for p in paths:
            rest = p[len(pre) + 1:] if pre else p
            if not recursive and "/" in rest:
                d = f"{pre}/{rest.split('/', 1)[0]}" if pre else rest.split("/", 1)[0]
                if d not in seen:
                    seen.add(d)
                    out.append(SimpleNamespace(path=d))  # a RepoFolder: no size
                continue
            out.append(SimpleNamespace(path=p, size=len(data[p]),
                                       lfs=SimpleNamespace(sha256=hashlib.sha256(data[p]).hexdigest())))
        return iter(out)

    # ------------------------------------------------------------------------------------- the runs repo's calls

    def upload_folder(self, *, repo_id, folder_path, path_in_repo="", ignore_patterns=None, commit_message=None,
                      repo_type=None, **kw):
        root, pre = Path(folder_path), (path_in_repo or "").rstrip("/")
        if self.drop_syncs and "/checkpoints" not in pre:
            self.dropped.append(pre)
            return
        files = {}
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or any(fnmatch.fnmatch(rel, pat) for pat in ignore_patterns or ()):
                continue
            files[f"{pre}/{rel}" if pre else rel] = p.read_bytes()
        with self.lock:
            self.repos.setdefault(repo_id, {}).update(files)
            self.commits.append(dict(repo=repo_id, adds=sorted(files), deletes=[], message=commit_message,
                                     folder=pre, ignore=list(ignore_patterns or ())))

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None, **kw):
        data = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
        with self.lock:
            self.repos.setdefault(repo_id, {})[path_in_repo] = data

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None, **kw):
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self.files(repo_id)[filename])
        return str(dst)


class Log:
    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append(dict(kind=kind, **kw))

    def kinds(self):
        return [e["kind"] for e in self.events]


def state_dir(root: Path, step: int, *, aux: bool = False, extra: dict | None = None) -> Path:
    """A local full_step_<step>/ with every required state file (distinct bytes per step), optionally aux_ctc.pt."""
    d = root / "checkpoints" / f"full_step_{step}"
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(fullrun.STATE_FILES_REQUIRED + (fullrun.STATE_FILES_OPTIONAL if aux else ())):
        (d / f).write_bytes(f"{f} of step {step} ".encode() * (i + 3))
    for f, b in (extra or {}).items():
        (d / f).write_bytes(b)
    return d


def meta(step: int, **kw) -> dict:
    return dict(dict(step=step, epoch=step / 10, planner_fingerprint="0123456789abcdef", n_train_utts=60,
                     selection_sha256="a" * 64, micro_audio_s=3.0, kitsune_sha="b" * 40,
                     host=dict(hostname="box", machine_id="151760", container_id="c1")), **kw)


# ------------------------------------------------------------------------------------------------------ helpers


def test_retry_rule_and_constants():
    assert scratch.RETRY_S == (15, 60, 180, 600) and scratch.RETRY_STATUS == (408, 409, 412, 429)
    for status in (408, 409, 412, 429, 500, 502, 503):
        assert scratch.retryable(HubError(status))
    for status in (400, 401, 403, 404, 422):
        assert not scratch.retryable(HubError(status))
    assert scratch.retryable(ConnectionError("reset by peer")) and scratch.retryable(TimeoutError())
    for local in (ValueError("x"), FileNotFoundError("x"), scratch.StateChanged("x"), TypeError("x")):
        assert not scratch.retryable(local)
    assert scratch.http_status(HubError(429)) == 429 and scratch.http_status(RuntimeError()) is None


def test_host_info_and_code_sha(monkeypatch):
    monkeypatch.setenv("KITSUNE_MACHINE_ID", "151760")
    monkeypatch.setenv("CONTAINER_ID", "c7")
    h = scratch.host_info()
    assert h["machine_id"] == "151760" and h["container_id"] == "c7" and isinstance(h["hostname"], str)
    monkeypatch.delenv("KITSUNE_MACHINE_ID")
    monkeypatch.delenv("CONTAINER_ID")
    assert scratch.host_info()["machine_id"] is None and scratch.host_info()["container_id"] is None
    monkeypatch.setenv("KITSUNE_SHA", "c" * 40)
    assert scratch.code_sha(ROOT) == "c" * 40  # the box's: the sha launch rented it at
    monkeypatch.delenv("KITSUNE_SHA")
    sha = scratch.code_sha(ROOT)
    assert sha is None or (len(sha) == 40 and int(sha, 16) >= 0)


# ------------------------------------------------------------------------------------------------------ the pointer


def test_make_pointer_lists_the_state_files_and_nothing_else(tmp_path):
    d = state_dir(tmp_path, 7, aux=True, extra={fullrun.SCRATCH_MARK: b"", ".upload_pending": b"",
                                                 "trainer.pt.tmp": b"a half-written rewrite"})
    ptr = scratch.make_pointer(RUN, d, **meta(7))
    assert fullrun.pointer_problems(ptr) == []
    assert sorted(ptr["files"]) == sorted(fullrun.STATE_FILES_REQUIRED + ("aux_ctc.pt",))
    for f, m in ptr["files"].items():
        assert m == dict(size=(d / f).stat().st_size, sha256=hashlib.sha256((d / f).read_bytes()).hexdigest())
    assert (ptr["format"], ptr["run_id"], ptr["name"], ptr["step"], ptr["epoch"]) == (1, RUN, "full_step_7", 7, 0.7)
    assert ptr["host"] == dict(hostname="box", machine_id="151760", container_id="c1")
    assert (ptr["planner_fingerprint"], ptr["n_train_utts"], ptr["micro_audio_s"]) == ("0123456789abcdef", 60, 3.0)
    assert ptr["kitsune_sha"] == "b" * 40 and ptr["selection_sha256"] == "a" * 64
    json.dumps(ptr)  # it goes up as JSON

    ptr = scratch.make_pointer(RUN, state_dir(tmp_path, 8), **meta(8, selection_sha256=None, kitsune_sha=None))
    assert "aux_ctc.pt" not in ptr["files"] and fullrun.pointer_problems(ptr) == []
    with pytest.raises(ValueError, match="full_step_9"):  # the dir must be the step's
        scratch.make_pointer(RUN, d, **meta(9))
    (d / "trainer.json").unlink()
    with pytest.raises(ValueError, match="trainer.json"):
        scratch.make_pointer(RUN, d, **meta(7))


# ------------------------------------------------------------------------------------------------------ the upload


def test_one_commit_adds_the_state_and_pointer_and_deletes_the_previous(tmp_path):
    hub, log = ScratchHub(), Log()
    other = "full-t06-20260927T120000Z"
    hub.repos[REPO] = {f"runs/{RUN}/checkpoints/full_step_2/model.pt": b"old",
                       f"runs/{RUN}/checkpoints/full_step_4/model.pt": b"old",
                       f"runs/{RUN}/checkpoints/full_step_4/trainer.json": b"{}",
                       f"runs/{RUN}/timed_state.json": b"{}",
                       f"runs/{other}/checkpoints/full_step_3/model.pt": b"box 2's other trainer"}
    d = state_dir(tmp_path, 7, extra={fullrun.SCRATCH_MARK: b"", ".upload_pending": b""})
    ptr = scratch.make_pointer(RUN, d, **meta(7))
    out = scratch.upload_state(hub, REPO, RUN, d, ptr, log=log, retries=())
    assert out["ok"] and out["attempts"] == 1 and out["deleted"] == ["full_step_2", "full_step_4"]
    assert out["squash_ok"] is True and out["gb"] == round(sum(m["size"] for m in ptr["files"].values()) / 1e9, 3)
    (c,) = hub.commits  # ONE commit: the pointer and the files it names change together
    dest = fullrun.scratch_state_dir(RUN, 7)
    assert c["adds"] == sorted([f"{dest}/{f}" for f in fullrun.STATE_FILES_REQUIRED] + [fullrun.scratch_pointer(RUN)])
    assert sorted(c["deletes"]) == [(f"runs/{RUN}/checkpoints/full_step_2/", True),
                                    (f"runs/{RUN}/checkpoints/full_step_4/", True)]
    files = hub.files(REPO)
    assert sorted(p for p in files if p.startswith(f"runs/{RUN}/")) == c["adds"]  # one state per run, no marker
    assert files[f"runs/{other}/checkpoints/full_step_3/model.pt"] == b"box 2's other trainer"
    assert json.loads(files[fullrun.scratch_pointer(RUN)]) == ptr
    for f, m in ptr["files"].items():
        assert hashlib.sha256(files[f"{dest}/{f}"]).hexdigest() == m["sha256"]
    assert hub.squashes == [REPO] and hub.created == []  # squashed; the repo is the owner's to create
    (ok,) = log.events
    assert ok == dict(kind="timed_state_upload_ok", name="full_step_7", step=7, attempt=0, gb=out["gb"],
                      upload_s=out["upload_s"], deleted=["full_step_2", "full_step_4"], squash_ok=True)

    # the next state replaces this one; the first upload of a run finds nothing under its folder (a 404)
    d8 = state_dir(tmp_path, 8)
    assert scratch.upload_state(hub, REPO, RUN, d8, scratch.make_pointer(RUN, d8, **meta(8)), retries=())["deleted"] \
        == ["full_step_7"]
    fresh = "full-p03-20260927T130000Z"
    d3 = state_dir(tmp_path / fresh, 3)
    assert scratch.upload_state(hub, REPO, fresh, d3, scratch.make_pointer(fresh, d3, **meta(3)), retries=())["ok"]
    runs = {PurePosixPath(p).parts[1]: PurePosixPath(p).parts[3] for p in hub.files(REPO) if "/checkpoints/" in p}
    assert runs == {RUN: "full_step_8", other: "full_step_3", fresh: "full_step_3"}


def test_transient_errors_are_retried_and_others_are_not(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(scratch.time, "sleep", sleeps.append)
    d = state_dir(tmp_path, 5)
    ptr = scratch.make_pointer(RUN, d, **meta(5))

    hub, log = ScratchHub(), Log()
    hub.fail_commit = [HubError(429, "Too Many Requests"), HubError(503), ConnectionError("reset")]
    out = scratch.upload_state(hub, REPO, RUN, d, ptr, log=log)
    assert out["ok"] and out["attempts"] == 4 and sleeps == [15, 60, 180]
    assert [(e["kind"], e["attempt"], e.get("status")) for e in log.events] == [
        ("timed_state_upload_error", 0, 429), ("timed_state_upload_error", 1, 503),
        ("timed_state_upload_error", 2, None), ("timed_state_upload_ok", 3, None)]

    hub, log = ScratchHub(), Log()
    hub.fail_commit = [HubError(403, "no write access")]
    out = scratch.upload_state(hub, REPO, RUN, d, ptr, log=log, retries=(1, 2))
    assert not out["ok"] and out["attempts"] == 1 and "403" in out["error"] and hub.commits == []
    assert log.kinds() == ["timed_state_upload_error", "timed_state_upload_failed"]
    assert log.events[-1]["attempts"] == 1 and hub.squashes == []

    hub, log = ScratchHub(), Log()
    hub.fail_commit = [HubError(500)] * 3
    out = scratch.upload_state(hub, REPO, RUN, d, ptr, log=log, retries=(0, 0))
    assert not out["ok"] and out["attempts"] == 3 and out["squash_ok"] is None
    assert log.kinds() == ["timed_state_upload_error"] * 3 + ["timed_state_upload_failed"]


def test_a_listing_the_hub_never_answers_is_a_retried_attempt(tmp_path, monkeypatch):
    """list_repo_tree gets no HTTP timeout from huggingface_hub: a listing that hangs ends its attempt after
    LIST_TIMEOUT_S (a TimeoutError, retried) instead of holding the scratch thread for the rest of the run."""
    monkeypatch.setattr(scratch, "LIST_TIMEOUT_S", 0.2)
    hub, log = ScratchHub(), Log()
    hang, calls, listing = threading.Event(), [], hub.list_repo_tree

    def list_repo_tree(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            hang.wait(30)  # the first listing never answers (until the test lets it go)
        return listing(*a, **kw)

    hub.list_repo_tree = list_repo_tree
    d = state_dir(tmp_path, 5)
    try:
        out = scratch.upload_state(hub, REPO, RUN, d, scratch.make_pointer(RUN, d, **meta(5)), log=log, retries=(0,))
    finally:
        hang.set()
    assert out["ok"] and out["attempts"] == 2 and len(calls) == 2 and len(hub.commits) == 1
    err = log.events[0]
    assert (err["kind"], err["attempt"], err["status"]) == ("timed_state_upload_error", 0, None)
    assert "TimeoutError" in err["error"] and "list_repo_tree" in err["error"]


def test_a_failed_squash_is_recorded_not_a_failure(tmp_path):
    hub, log = ScratchHub(), Log()
    hub.fail_squash = [HubError(412, "Precondition Failed"), HubError(403)]
    d = state_dir(tmp_path, 5)
    out = scratch.upload_state(hub, REPO, RUN, d, scratch.make_pointer(RUN, d, **meta(5)), log=log, retries=(0, 0))
    assert out["ok"] and out["squash_ok"] is False and len(hub.commits) == 1  # a 412 retried, the 403 not
    assert hub.fail_squash == [] and log.events[-1]["squash_ok"] is False and "403" in log.events[-1]["squash_error"]


def test_a_state_rewritten_after_its_pointer_is_not_committed(tmp_path):
    hub, log = ScratchHub(), Log()
    d = state_dir(tmp_path, 5)
    ptr = scratch.make_pointer(RUN, d, **meta(5))
    (d / "trainer.pt").write_bytes(b"the end save's rewrite, another size")
    out = scratch.upload_state(hub, REPO, RUN, d, ptr, log=log, retries=(0, 0))
    assert not out["ok"] and out["attempts"] == 1 and "trainer.pt" in out["error"] and hub.commits == []
    assert log.kinds() == ["timed_state_upload_error", "timed_state_upload_failed"]


def test_bad_pointers_are_refused(tmp_path):
    d = state_dir(tmp_path, 5)
    ptr = scratch.make_pointer(RUN, d, **meta(5))
    with pytest.raises(ValueError, match="marker"):
        scratch.upload_state(ScratchHub(), REPO, RUN, d, dict(ptr, files=dict(ptr["files"], **{
            fullrun.SCRATCH_MARK: dict(size=0, sha256=hashlib.sha256(b"").hexdigest())})), retries=())
    with pytest.raises(ValueError, match="uploaded as"):
        scratch.upload_state(ScratchHub(), REPO, "another-20260927T120000Z", d, ptr, retries=())
