"""A runs repo on disk that several processes commit into at once, with the Hub's commit rate limit (a 429), for the
test that runs study boxes A and B concurrently (tests/test_study_box.py).

    hub = DirHub.create(tmp_path / "hub", limit=6, window_s=0.5)   # at most 6 commits in any 0.5 s, else 429
    hub.commit({"runs/x/a.txt": b"..."}, writer="x", box="A")      # raises RateLimited over the limit

<dir>/files/<path in repo> holds the files, <dir>/commits.jsonl one line per commit attempt (wall, writer, box, paths,
accepted), <dir>/config.json the limit; a lock file serialises the commits of every process, as the Hub serialises a
repo's commits. FakeApi is the part of huggingface_hub.HfApi that kitsune.study_queue.HubUploader and vast/finish.py
call (upload_folder, upload_file, file_exists, list_repo_tree), over a DirHub; download / snapshot_download stand in
for huggingface_hub's. The fake trainer (tests/fake_study_trainer.py, FAKE_HUB) commits its log syncs through
DirHub.commit with the trainer's back-off, as kitsune/runlog.py does.
"""
import fnmatch
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


class RateLimited(RuntimeError):
    """What huggingface_hub raises for the Hub's 429: finish.retryable keys on response.status_code."""

    def __init__(self, msg: str):
        super().__init__(msg)
        self.response = SimpleNamespace(status_code=429)


class DirHub:
    def __init__(self, d):
        self.dir = Path(d)
        cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.limit, self.window_s = int(cfg["limit"]), float(cfg["window_s"])

    @classmethod
    def create(cls, d, limit: int, window_s: float) -> "DirHub":
        d = Path(d)
        (d / "files").mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(dict(limit=limit, window_s=window_s)), encoding="utf-8")
        return cls(d)

    # ------------------------------------------------------------------------------------------------ commits

    def _lock(self):
        p = self.dir / "commit.lock"
        t0 = time.time()
        while True:
            try:
                return os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (FileExistsError, PermissionError):  # Windows: a lock file being deleted is "access denied"
                if time.time() - t0 > 60:
                    raise TimeoutError(f"{p} held for 60 s")
                time.sleep(0.002)

    def _unlock(self, fd):
        os.close(fd)
        os.unlink(self.dir / "commit.lock")

    def commits(self) -> list[dict]:
        p = self.dir / "commits.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []

    def commit(self, files: dict[str, bytes], writer: str, box: str | None = None):
        fd = self._lock()
        try:
            now = time.time()
            recent = [c for c in self.commits() if c["accepted"] and now - c["wall"] < self.window_s]
            ok = len(recent) < self.limit
            if ok:
                for path, data in files.items():
                    f = self.dir / "files" / path
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(data)
            with open(self.dir / "commits.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(wall=now, writer=writer, box=box, paths=sorted(files), accepted=ok)) + "\n")
        finally:
            self._unlock(fd)
        if not ok:
            raise RateLimited(f"429: over {self.limit} commits in {self.window_s} s")

    # -------------------------------------------------------------------------------------------------- reads

    def path(self, path_in_repo: str) -> Path:
        return self.dir / "files" / path_in_repo

    def listing(self, prefix: str) -> list[str]:
        root = self.dir / "files"
        base = root / prefix if prefix else root
        return sorted(p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file()) if base.is_dir() else []


class FakeApi:
    """HfApi over a DirHub, one per box (box names the committer in the commit log)."""

    def __init__(self, hub: DirHub, box: str):
        self.hub, self.box = hub, box

    def upload_folder(self, repo_id, folder_path, path_in_repo="", repo_type=None, allow_patterns=None,
                      ignore_patterns=None, commit_message=None, **kw):
        root = Path(folder_path)
        files = {}
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or (allow_patterns and not any(fnmatch.fnmatch(rel, a) for a in allow_patterns)):
                continue
            if ignore_patterns and any(fnmatch.fnmatch(rel, a) for a in ignore_patterns):
                continue
            files[f"{path_in_repo.rstrip('/')}/{rel}" if path_in_repo else rel] = p.read_bytes()
        self.hub.commit(files, writer=f"queue-{self.box}", box=self.box)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id=None, repo_type=None, commit_message=None, **kw):
        self.hub.commit({path_in_repo: Path(path_or_fileobj).read_bytes()}, writer=f"queue-{self.box}", box=self.box)

    def file_exists(self, repo_id, filename, repo_type=None, **kw) -> bool:
        return self.hub.path(filename).is_file()

    def list_repo_tree(self, repo_id, path_in_repo="", recursive=False, repo_type=None, **kw):
        pre = path_in_repo.rstrip("/")
        out = []
        for rel in self.hub.listing(pre):
            rest = rel[len(pre) + 1:] if pre else rel
            if not recursive and "/" in rest:
                d = f"{pre}/{rest.split('/', 1)[0]}" if pre else rest.split("/", 1)[0]
                if not any(x.path == d for x in out):
                    out.append(SimpleNamespace(path=d, size=None))
                continue
            data = self.hub.path(rel).read_bytes()
            out.append(SimpleNamespace(path=rel, size=len(data),
                                       lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())))
        return out


def downloads(hub: DirHub):
    """(hf_hub_download, snapshot_download) stand-ins reading the DirHub."""

    def hf_hub_download(repo_id, filename, repo_type=None, local_dir=None, revision=None, **kw):
        src = hub.path(filename)
        if not src.is_file():  # huggingface_hub's EntryNotFoundError: a 404, which finish.hub_retry does not repeat
            e = FileNotFoundError(f"404: {filename}")
            e.response = SimpleNamespace(status_code=404)
            raise e
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return str(dst)

    def snapshot_download(repo_id, repo_type=None, local_dir=None, allow_patterns=None, **kw):
        for rel in hub.listing(""):
            if not allow_patterns or any(fnmatch.fnmatch(rel, a) for a in allow_patterns):
                hf_hub_download(repo_id, rel, local_dir=local_dir)
        return str(local_dir)

    return hf_hub_download, snapshot_download
