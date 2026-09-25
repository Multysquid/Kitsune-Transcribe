"""vast/label_sync.py and finish.py --job label: the label root goes up write-once, in pair-keeping batches, verified
per directory, with a lease, and a box is destroyed only when the Hub provably holds every finished label file.

CPU only, no network: the Hub is a FakeHub that lists one directory at a time (as label_sync asks it to) and applies
create_commit operations; vast REST calls are faked like tests/test_infra.py does.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vast"))
finish = importlib.import_module("finish")
label_sync = importlib.import_module("label_sync")
supervise = importlib.import_module("supervise")

REPO = "Multy123/kitsune-data"
PREFIX = "labels/full"


def entry(path: str, data: bytes):
    """What the Hub lists: LFS sha256 for binaries over 1000 B, a git blob id for small files."""
    lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data)) if len(data) > 1000 else None
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return SimpleNamespace(path=path, size=len(data), blob_id=blob, lfs=lfs)


class FakeHub:
    def __init__(self, files: dict[str, bytes] | None = None, fail_commits: int = 0):
        self.files = {p: entry(p, d) for p, d in (files or {}).items()}
        self.data = dict(files or {})
        self.commits, self.listed, self.fail_commits = [], [], fail_commits

    def list_repo_tree(self, repo, path_in_repo=None, recursive=False, repo_type=None, **kw):
        assert repo_type == "dataset" and not recursive, "label listings are per dir"
        self.listed.append(path_in_repo)
        items = [f for p, f in self.files.items() if p.rsplit("/", 1)[0] == path_in_repo]
        if not items and not any(p.startswith(path_in_repo + "/") for p in self.files):
            raise type("RemoteEntryNotFoundError", (Exception,), {})(path_in_repo)
        return [SimpleNamespace(path=path_in_repo + "/sub")] + items  # a folder has no size

    def hf_hub_download(self, repo_id=None, filename=None, repo_type=None, **kw):
        import tempfile

        if filename not in self.data:
            raise FileNotFoundError(filename)
        f = Path(tempfile.mkdtemp()) / Path(filename).name
        f.write_bytes(self.data[filename])
        return str(f)

    def create_commit(self, repo_id=None, repo_type=None, operations=(), commit_message="", **kw):
        if self.fail_commits:
            self.fail_commits -= 1
            raise RuntimeError("503 Service Unavailable")
        paths = [op.path_in_repo for op in operations]
        self.commits.append(paths)
        for op in operations:
            src = op.path_or_fileobj
            data = Path(src).read_bytes() if isinstance(src, str) else bytes(src)
            self.files[op.path_in_repo], self.data[op.path_in_repo] = entry(op.path_in_repo, data), data


def put(root: Path, rel: str, data: bytes = b"x") -> Path:
    p = root / PREFIX / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def make_labels(root: Path) -> None:
    put(root, "teacher_out/meta.json", b'{"k": 1}')
    for i in range(3):
        put(root, f"teacher_out/reazon_small/train-0000{i}.jsonl", b'{"id": "a"}\n' * 5)
        put(root, f"teacher_out/reazon_small/train-0000{i}.npz", bytes([i]) * 3000)
    put(root, "teacher_out/reazon_small/train-00003.jsonl", b"in flight")  # its npz is not written yet
    put(root, "teacher_out/reazon_small/train-00004.npz.tmp", b"half")
    put(root, "parakeet_out/meta.json", b"{}")
    put(root, "parakeet_out/galgame/train-00000.jsonl", b'{"id": "g"}\n')
    put(root, "parakeet_out/galgame/train-00000.npz", b"p" * 1500)
    put(root, "parakeet_out/galgame/train-00001.npz", b"no jsonl: broken")
    put(root, "second_out/meta.json", b"{}")
    put(root, "second_out/galgame/train-00000.jsonl", b'{"id": "g", "agree": true}\n')
    put(root, "second_out/_cache/galgame.parquet", b"cache")
    put(root, "second_out/galgame_smoke/train-00000.jsonl", b"smoke")
    put(root, "LEASE.json", b"{}")
    put(root, "COMPLETE.json", b"{}")
    put(root, "extent.json", b'{"schema": 1}')
    put(root, "reports/labels.json", b"{}")


def lease(released=False, now=1000.0):
    return label_sync.lease_bytes("run-1", "c1", "m1", "abc", released, now=now)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(finish, "HUB_RETRY_WAITS", (0, 0))


# ------------------------------------------------------------------------------------------------ files and batches

def test_finished_files_are_finished_pairs_metas_and_final_artifacts_only(tmp_path):
    """A consumer treats npz present = finished (the writer puts the jsonl first), so a jsonl alone is in flight and
    an npz alone is broken; tmp/cache/smoke files and the generated LEASE/COMPLETE are never uploaded as files."""
    make_labels(tmp_path)
    got = set(label_sync.finished_files(tmp_path, PREFIX))
    want = {f"{PREFIX}/{r}" for r in [
        "teacher_out/meta.json", "parakeet_out/meta.json", "second_out/meta.json", "extent.json", "reports/labels.json",
        "parakeet_out/galgame/train-00000.jsonl", "parakeet_out/galgame/train-00000.npz",
        "second_out/galgame/train-00000.jsonl"]}
    want |= {f"{PREFIX}/teacher_out/reazon_small/train-0000{i}.{e}" for i in range(3) for e in ("npz", "jsonl")}
    assert got == want


def test_commit_batches_keep_pairs_and_respect_the_file_and_text_budgets(tmp_path):
    items = []
    for i in range(7):
        items.append((f"{PREFIX}/teacher_out/s/train-{i:05d}.jsonl", put(tmp_path, f"t/{i}.jsonl", b"j" * 100)))
        items.append((f"{PREFIX}/teacher_out/s/train-{i:05d}.npz", put(tmp_path, f"t/{i}.npz", b"n" * 5000)))
    batches = label_sync.commit_batches(items, max_files=5)
    assert [len(b) for b in batches] == [4, 4, 4, 2]  # 5 would split a pair
    for b in batches:
        assert {p.rsplit(".", 1)[0] for p, _ in b} == {p.rsplit(".", 1)[0] for p, _ in b if p.endswith(".npz")}
    batches = label_sync.commit_batches(items, max_text_bytes=250)  # npz bytes do not count toward the text budget
    assert [len(b) for b in batches] == [4, 4, 4, 2]
    assert sum(label_sync.commit_batches(items), []) == items


# ------------------------------------------------------------------------------------------------------------ sync

def test_sync_uploads_finished_files_verifies_them_and_the_ledger_prevents_a_reupload(tmp_path):
    make_labels(tmp_path)
    hub, ledger = FakeHub(), tmp_path / "state" / "label_ledger.json"
    res = label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(), log=lambda m: None)
    files = label_sync.finished_files(tmp_path, PREFIX)
    assert res["uploaded"] == len(files) and res["commits"] == 1
    assert hub.commits[-1][-1] == f"{PREFIX}/LEASE.json", "the lease rides the last commit"
    assert set(json.loads(ledger.read_text())["remote"]) == set(files)
    assert all(p.startswith(PREFIX + "/") for c in hub.commits for p in c)
    hub.commits.clear()
    res = label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(), log=lambda m: None)
    assert res["new"] == 0 and hub.commits == [[f"{PREFIX}/LEASE.json"]], "only the lease heartbeat goes up"
    put(tmp_path, "teacher_out/reazon_small/train-00003.npz", b"late" * 400)  # the in-flight pair finished
    hub.commits.clear()
    label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=None, log=lambda m: None)
    assert hub.commits == [[f"{PREFIX}/teacher_out/reazon_small/train-00003.jsonl",
                            f"{PREFIX}/teacher_out/reazon_small/train-00003.npz"]]


def test_sync_skips_identical_remote_files_and_splits_commits(tmp_path, monkeypatch):
    make_labels(tmp_path)
    files = label_sync.finished_files(tmp_path, PREFIX)
    already = f"{PREFIX}/teacher_out/reazon_small/train-00000.npz"
    hub = FakeHub({already: files[already].read_bytes()})  # a relaunch: already on the Hub, not in this ledger
    monkeypatch.setattr(label_sync, "commit_batches",
                        lambda items: [items[i:i + 4] for i in range(0, len(items), 4)])
    res = label_sync.sync(hub, REPO, tmp_path, PREFIX, tmp_path / "l.json", lease=lease(), log=lambda m: None)
    assert res["same"] == 1 and already not in sum(hub.commits, [])
    assert len(hub.commits) == res["commits"] > 1
    assert hub.commits[-1][-1].endswith("LEASE.json") and not any(p.endswith("LEASE.json") for c in hub.commits[:-1]
                                                                   for p in c)


def test_the_prefix_guard_refuses_laptop_roots(tmp_path):
    put(tmp_path, "x.json")
    with pytest.raises(label_sync.IntegrityError, match="outside"):
        label_sync.sync(FakeHub(), REPO, tmp_path / "labels", "full", tmp_path / "l.json", lease=None)
    with pytest.raises(label_sync.IntegrityError):
        label_sync.guard("teacher_out/reazon_small/train-00000.npz")
    label_sync.guard("label_runs/run-1/kitsune.log")


def test_a_conflict_uploads_every_other_file_then_raises(tmp_path):
    """Write-once: a label file already on the Hub with other content is never replaced; labels must not pile up
    unsynced behind it, so the rest goes up (never half of the conflicting pair) and then IntegrityError ends the box."""
    make_labels(tmp_path)
    bad = f"{PREFIX}/teacher_out/reazon_small/train-00001.npz"
    hub = FakeHub({bad: b"other content" * 200})
    with pytest.raises(label_sync.IntegrityError, match="train-00001.npz"):
        label_sync.sync(hub, REPO, tmp_path, PREFIX, tmp_path / "l.json", lease=lease(), log=lambda m: None)
    sent = set(sum(hub.commits, []))
    assert bad not in sent and bad.replace(".npz", ".jsonl") not in sent
    assert f"{PREFIX}/teacher_out/reazon_small/train-00002.npz" in sent
    assert hub.data[bad] == b"other content" * 200


def test_final_artifacts_are_replaceable_until_the_root_is_sealed(tmp_path):
    make_labels(tmp_path)
    hub, ledger = FakeHub({f"{PREFIX}/extent.json": b'{"old": 1}', f"{PREFIX}/reports/labels.json": b"old"}), \
        tmp_path / "l.json"
    label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(), log=lambda m: None)
    assert hub.data[f"{PREFIX}/extent.json"] == b'{"schema": 1}'
    # sealed: a fully verified sync is a no-op success (the final --destroy), with no lease op
    hub.data[f"{PREFIX}/COMPLETE.json"] = b"{}"
    hub.files[f"{PREFIX}/COMPLETE.json"] = entry(f"{PREFIX}/COMPLETE.json", b"{}")
    hub.commits.clear()
    res = label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(True), log=lambda m: None)
    assert res["sealed"] and hub.commits == []
    put(tmp_path, "reports/labels.json", b"changed after the seal")
    with pytest.raises(label_sync.IntegrityError, match="sealed"):
        label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(True), log=lambda m: None)
    put(tmp_path, "reports/labels.json", b"{}")
    put(tmp_path, "second_out/galgame/train-00009.jsonl", b"new")
    with pytest.raises(label_sync.IntegrityError, match="sealed"):
        label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=None, log=lambda m: None)
    assert hub.commits == []


def test_a_failed_commit_raises_sync_error_and_the_next_sync_repairs_it(tmp_path):
    make_labels(tmp_path)
    hub, ledger = FakeHub(fail_commits=3), tmp_path / "l.json"
    with pytest.raises(label_sync.SyncError):
        label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(), log=lambda m: None)
    assert hub.commits == []
    res = label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(), log=lambda m: None)
    assert res["uploaded"] == len(label_sync.finished_files(tmp_path, PREFIX))


def test_verify_lists_per_dir_and_finds_missing_and_changed_files(tmp_path):
    make_labels(tmp_path)
    files = label_sync.finished_files(tmp_path, PREFIX)
    hub = FakeHub({p: f.read_bytes() for p, f in files.items()})
    assert label_sync.verify(hub, REPO, files, tmp_path / "l.json") == []
    assert f"{PREFIX}/teacher_out/reazon_small" in hub.listed
    npz = f"{PREFIX}/teacher_out/reazon_small/train-00002.npz"
    hub.files[npz] = entry(npz, bytes([9]) * 3000)  # same size, other bytes
    del hub.files[f"{PREFIX}/second_out/meta.json"]
    problems = label_sync.verify(hub, REPO, files, tmp_path / "l.json")
    assert any("train-00002.npz: hash differs" in p for p in problems)
    assert any("second_out/meta.json: missing" in p for p in problems)


def test_lease_live_rules():
    now = 10_000.0
    other = json.loads(label_sync.lease_bytes("r", "c2", "m", "s", False, now=now - 60))
    assert label_sync.lease_live(other, "c1", now)
    assert not label_sync.lease_live(other, "c2", now), "our own lease never blocks us"
    assert not label_sync.lease_live({**other, "released": True}, "c1", now)
    assert not label_sync.lease_live({**other, "heartbeat": now - 2700}, "c1", now), "45 min without a heartbeat"
    assert not label_sync.lease_live(None, "c1", now)
    assert label_sync.lease_live({**other, "heartbeat": "garbage"}, "c1", now)


def test_sync_never_overwrites_a_live_lease_of_another_box(tmp_path):
    """A refused box (or any box that lost the root) must not replace the running box's lease with its own."""
    import time as _time

    make_labels(tmp_path)
    (tmp_path / PREFIX / "COMPLETE.json").unlink(missing_ok=True)
    theirs = label_sync.lease_bytes("run-other", "c-other", "m", "s", False, now=_time.time() - 60)
    hub, ledger = FakeHub({f"{PREFIX}/LEASE.json": theirs}), tmp_path / "l.json"
    mine = label_sync.lease_bytes("run-1", "c-mine", "m", "s", True)
    with pytest.raises(label_sync.IntegrityError, match="live for another box"):
        label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=mine, log=lambda m: None)
    assert hub.commits == [] and hub.data[f"{PREFIX}/LEASE.json"] == theirs
    stale = label_sync.lease_bytes("run-other", "c-other", "m", "s", False, now=_time.time() - 3000)
    hub = FakeHub({f"{PREFIX}/LEASE.json": stale})
    label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=mine, log=lambda m: None)
    assert json.loads(hub.data[f"{PREFIX}/LEASE.json"])["container_id"] == "c-mine"


def test_seal_writes_complete_with_the_released_lease_and_is_rerunnable(tmp_path):
    make_labels(tmp_path)
    (tmp_path / PREFIX / "COMPLETE.json").unlink()
    hub, ledger = FakeHub(), tmp_path / "l.json"
    info = {"name": "full", "run_ids": ["run-1"], "kitsune_sha": "abc", "image": "img", "configs": ["configs/full.json"]}
    with pytest.raises(label_sync.SyncError, match="refusing to seal"):
        label_sync.seal(hub, REPO, tmp_path, PREFIX, info, ledger)  # nothing verified yet
    label_sync.sync(hub, REPO, tmp_path, PREFIX, ledger, lease=lease(), log=lambda m: None)
    label_sync.seal(hub, REPO, tmp_path, PREFIX, info, ledger)
    assert hub.commits[-1] == [f"{PREFIX}/COMPLETE.json", f"{PREFIX}/LEASE.json"]
    done = json.loads(hub.data[f"{PREFIX}/COMPLETE.json"])
    files = label_sync.finished_files(tmp_path, PREFIX)
    lines = "\n".join(sorted(f"{p} {hashlib.sha256(f.read_bytes()).hexdigest()}" for p, f in files.items()))
    assert done["files"] == len(files) and done["files_digest"] == hashlib.sha256(lines.encode()).hexdigest()
    assert done["extent_sha256"] == hashlib.sha256(b'{"schema": 1}').hexdigest()
    assert json.loads(hub.data[f"{PREFIX}/LEASE.json"])["released"] is True
    n = len(hub.commits)
    label_sync.seal(hub, REPO, tmp_path, PREFIX, info, ledger)  # a re-run after the commit landed
    assert len(hub.commits) == n


def test_run_id_comes_from_label_json(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTAINER_ID", "777")
    assert label_sync.run_id(tmp_path) == "777-unknown"
    (tmp_path / "label.json").write_text(json.dumps({"run_id": "20260925T000000Z-777", "phase": "loop"}))
    assert label_sync.run_id(tmp_path) == "20260925T000000Z-777"


# ------------------------------------------------------------------------------------------------ finish --job label

@pytest.fixture
def box(tmp_path, monkeypatch):
    """A label box: checkout with a config naming labels/full, labels on disk, state dir, faked vast REST."""
    cfg = tmp_path / "full.json"
    cfg.write_text(json.dumps({"extent": {"name": "full", "root": PREFIX, "inputs": {}}}))
    make_labels(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "label.json").write_text(json.dumps({"run_id": "run-7", "phase": "loop"}))
    for k, v in {"KITSUNE_JOB": "label", "KITSUNE_CONFIG": str(cfg), "KITSUNE_DIR": str(tmp_path),
                 "KITSUNE_DATA_REPO": REPO, "CONTAINER_ID": "c1", "KITSUNE_SHA": "abc"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(finish, "STATE_DIR", state)
    monkeypatch.setattr(finish, "INFRA_LOGS", [])
    actions, locks = [], []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    monkeypatch.setattr(finish, "vast_cli", lambda action: pytest.fail("CLI fallback must not be needed"))
    real_lock = label_sync.acquire_sync_lock
    monkeypatch.setattr(label_sync, "acquire_sync_lock", lambda p, w: locks.append((Path(p), w)) or real_lock(p, 0))
    hub = FakeHub()
    monkeypatch.setattr(finish, "hf_api", lambda: hub)
    return SimpleNamespace(root=tmp_path, state=state, hub=hub, actions=actions, locks=locks)


def test_finish_sync_only_uploads_labels_takes_the_lock_and_puts_infra_under_label_runs(box):
    assert finish.main(["--sync-only"]) == 0
    assert box.locks == [(box.state / "sync.lock", finish.SYNC_LOCK_WAIT_S)]
    first = box.hub.commits[0]
    assert first[-1] == f"{PREFIX}/LEASE.json" and json.loads(box.hub.data[first[-1]])["released"] is False
    assert json.loads(box.hub.data[first[-1]])["run_id"] == "run-7"
    infra = box.hub.commits[-1]
    assert infra and all(p.startswith("label_runs/run-7/") for p in infra)
    assert "label_runs/run-7/sync.lock" not in infra and box.actions == []
    box.hub.commits.clear()
    assert finish.main(["--sync-only", "--no-infra"]) == 0
    assert box.hub.commits == [[f"{PREFIX}/LEASE.json"]]


def test_finish_sync_only_returns_1_on_a_failed_upload_and_65_on_a_conflict(box):
    box.hub.fail_commits = 3
    assert finish.main(["--sync-only", "--no-infra"]) == 1
    bad = f"{PREFIX}/second_out/meta.json"
    box.hub.files[bad], box.hub.data[bad] = entry(bad, b"other"), b"other"
    assert finish.main(["--sync-only", "--no-infra"]) == finish.EXIT_INTEGRITY
    events = [json.loads(ln)["kind"] for ln in (box.state / "events.jsonl").read_text().splitlines()]
    assert "sync_failed" in events and "sync_integrity" in events


def test_finish_destroy_destroys_only_when_every_label_file_is_verified(box):
    assert finish.main(["--destroy", "--no-infra"]) == 0
    assert box.actions == ["destroy"]
    assert json.loads(box.hub.data[f"{PREFIX}/LEASE.json"])["released"] is True
    assert json.loads((box.state / "halt").read_text())["action"] == "destroy"


def test_finish_destroy_stops_when_verification_fails(box, monkeypatch):
    monkeypatch.setattr(label_sync, "verify", lambda *a, **k: ["x: missing"])
    assert finish.main(["--destroy", "--no-infra"]) == 2
    assert box.actions == ["stop"]


def test_finish_destroy_with_nothing_finished_stops_unless_allow_empty(box):
    import shutil
    shutil.rmtree(box.root / PREFIX)
    assert finish.main(["--destroy", "--no-infra"]) == 2 and box.actions == ["stop"]
    box.actions.clear()
    assert finish.main(["--destroy", "--no-infra", "--allow-empty"]) == 0 and box.actions == ["destroy"]


def test_finish_stop_syncs_with_a_released_lease_then_stops(box):
    assert finish.main(["--stop", "--reason", "integrity", "--no-infra"]) == 0
    assert box.actions == ["stop"] and json.loads(box.hub.data[f"{PREFIX}/LEASE.json"])["released"] is True


def test_finish_busy_sync_lock_skips_the_sync_with_rc_1(box, monkeypatch):
    monkeypatch.setattr(label_sync, "acquire_sync_lock", lambda p, w: None)
    assert finish.main(["--sync-only", "--no-infra"]) == 1 and box.hub.commits == []


def test_finish_train_default_is_unchanged(monkeypatch):
    """Without --job and KITSUNE_JOB the train path runs (the train finish tests in test_infra stay as they are)."""
    monkeypatch.delenv("KITSUNE_JOB", raising=False)
    monkeypatch.setattr(finish, "label_main", lambda args: pytest.fail("label path taken"))
    with pytest.raises(SystemExit):
        finish.main([])  # a mode is required: argparse exits before anything runs


def test_finish_verify_prefix_of_lists_each_dir_non_recursively(tmp_path):
    f = tmp_path / "a.json"
    f.write_bytes(b"{}")
    calls = []

    class Api:
        def list_repo_tree(self, repo, path_in_repo=None, recursive=True, repo_type=None):
            calls.append((path_in_repo, recursive))
            return [entry("labels/full/x/a.json", b"{}")]

    assert finish.verify(Api(), REPO, "dataset", {"labels/full/x/a.json": f},
                         prefix_of=lambda p: p.rsplit("/", 1)[0]) == []
    assert calls == [("labels/full/x", False)]


# -------------------------------------------------------------------------------------------------------- supervise

def test_load_state_required_phase_accepts_label_state_and_flags_others(tmp_path):
    p = tmp_path / "label.json"
    assert supervise.load_state(p, required="phase") == {"phase": None, "final": None}
    p.write_text(json.dumps({"version": 1, "phase": "loop", "run_id": "r"}))
    assert supervise.load_state(p, required="phase")["run_id"] == "r"
    p.write_text(json.dumps({"attempts": []}))
    state = supervise.load_state(p, required="phase")
    assert "corrupt" in state and state["phase"] is None and not p.exists()
    p.write_text(json.dumps({"attempts": [], "final": None}))
    assert supervise.load_state(p) == {"attempts": [], "final": None}, "the default is unchanged"
