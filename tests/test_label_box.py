"""vast/label.py, the label box controller: its steps against a FakeHfApi, and its loop (lanes, heartbeats, janitor,
sync cycles, rates, budget, finalize, endings) with a FakeRunner, a FakeClock and a recorded finish. No GPU, no Hub."""
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vast"))
label = importlib.import_module("label")

from kitsune import store  # noqa: E402

T0 = 1_000_000.0
GATES = ["eval_jsut", "eval_cv8", "eval_reazon"]


# ----------------------------------------------------------------------------------------------------------- fakes


class FakeClock:
    def __init__(self, t=T0):
        self.t = t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProc:
    def __init__(self, runner, name, rc):
        self.runner, self.name, self.rc, self.terminated = runner, name, rc, False

    def poll(self):
        if self.rc is None and self.runner.beats and self.name in self.runner.lane_names:
            hb = self.runner.state_dir / "hb" / self.name
            os.utime(hb, (self.runner.clock.t, self.runner.clock.t))
        return self.rc

    def terminate_group(self, grace=0):
        self.terminated = True
        if self.rc is None:
            self.rc = -15


class FakeRunner:
    """behave(name, argv) -> rc (None: keeps running). Lanes touch their heartbeat whenever they are polled."""
    lane_names = ("ingest", "cohere-0", "cohere-1", "parakeet")

    def __init__(self, clock, state_dir, behave=None, beats=True):
        self.clock, self.state_dir, self.beats = clock, state_dir, beats
        self.behave = behave or (lambda name, argv: None if name in self.lane_names else 0)
        self.started, self.procs = [], []

    def start(self, name, argv, env, log):
        self.started.append((name, [str(a) for a in argv], env))
        p = FakeProc(self, name, self.behave(name, argv))
        self.procs.append(p)
        return p

    def names(self):
        return [n for n, _, _ in self.started]


class FakeFinish:
    def __init__(self):
        self.calls = []

    def __call__(self, action, reason, extra):
        self.calls.append((action, reason, list(extra)))


@pytest.fixture
def box(tmp_path, monkeypatch):
    kdir, state = tmp_path / "repo", tmp_path / "ws" / "kitsune_state"
    (kdir / "configs").mkdir(parents=True)
    for c in ("full.json", "full_sub3k.json"):
        shutil.copy(ROOT / "configs" / c, kdir / "configs" / c)
    state.mkdir(parents=True)
    (state / "deadline").write_text(str(int(T0 + 30 * 3600)))
    write_manifest(kdir / "data", [("eval_jsut", "eval", 0, ["j1", "j2"])])
    env = {"KITSUNE_DIR": str(kdir), "KITSUNE_STATE": str(state), "KITSUNE_CONFIG": "configs/full.json",
           "KITSUNE_LABEL_CONFIGS": "configs/full.json,configs/full_sub3k.json", "KITSUNE_CORES": "16",
           "CONTAINER_ID": "c1", "KITSUNE_MACHINE_ID": "m1", "KITSUNE_DATA_REPO": "me/data",
           "KITSUNE_DATA_REVISION": "rev1", "PATH": os.environ.get("PATH", "")}
    monkeypatch.setattr(label.Controller, "disk_free", lambda self: 500e9)
    clock = FakeClock()

    def make(behave=None, beats=True, **env_over):
        runner = FakeRunner(clock, state, behave, beats)
        fin = FakeFinish()
        c = label.Controller(env=dict(env, **env_over), runner=runner, clock=clock, final_finish=fin, py="py")
        return c, runner, fin

    return SimpleNamespace(kdir=kdir, state=state, env=env, clock=clock, make=make)


def write_manifest(data: Path, shards):
    """shards: (source, split, number, ids); writes each shard as a tiny parquet with an id column."""
    infos = []
    for src, split, n, ids in shards:
        rel = f"shards/{src}/{split}-{n:05d}.parquet"
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"id": ids, "audio": [b"x" * 100] * len(ids)}), data / rel)
        infos.append(store.ShardInfo(path=rel, source=src, split=split, rows=len(ids), hours=1.0))
    store.write_manifest(data, [*store.read_manifest(data), *infos])
    return infos


def write_state(state: Path, **kw):
    st = {"version": 1, "run_id": "r1", "phase": "label", "steps_done": [], "lanes": {}, "sources": {}, "syncs": [],
          "rates": {}, "finalize_done": [], "final": None, "lanes_started": False, **kw}
    (state / "label.json").write_text(json.dumps(st))
    return st


def teacher_npz(path: Path, ids, n_scanned, k=8, jsonl=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    toks = np.arange(len(ids), dtype=np.int16)
    np.savez(path, ids=np.array(ids), n_scanned=np.int64(n_scanned), k=np.int64(k), tokens=toks,
             topk_idx=np.stack([toks] * k, axis=1), prompt=np.array([], dtype=np.int64))
    if jsonl:
        path.with_suffix(".jsonl").write_text("".join(json.dumps({"id": i, "hyp": "a", "ref": "a"}) + "\n"
                                                      for i in ids), encoding="utf-8")


def run_ctl(c):
    rc = c.run()
    return rc, json.loads((c.state_path).read_text())


# ------------------------------------------------------------------------------------------------------------ pins


def test_pins_match_their_sources():
    t = (ROOT / "scripts" / "02_teacher_pass.py").read_text(encoding="utf-8")
    assert f'MODEL_ID = "{label.COHERE_MODEL_ID}"' in t and f'MODEL_REVISION = "{label.COHERE_REVISION}"' in t
    assert f'WHISPER_TOK_REVISION = "{label.WHISPER_TOK_REVISION}"' in \
        (ROOT / "scripts" / "02b_second_opinion.py").read_text(encoding="utf-8")
    assert f'PARAKEET_PATH = "{label.PARAKEET_PATH}"' in (ROOT / "kitsune" / "parakeet.py").read_text(encoding="utf-8")
    cfg = json.loads((ROOT / "configs" / "full.json").read_text(encoding="utf-8"))
    assert label.knobs(cfg) == label.knobs({"label": label.KNOBS}), "configs/full.json label block == the defaults"


def test_controller_imports_no_torch():
    import subprocess

    code = "import sys; sys.path.insert(0, 'vast'); import label; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert out.stdout.strip() == "False", out.stderr


# ----------------------------------------------------------------------------------------------------------- steps


class HubError(Exception):
    def __init__(self, status):
        super().__init__(f"{status}")
        self.response = SimpleNamespace(status_code=status)


class GatedRepoError(Exception):
    pass


def item(path, size=10, sha=None):
    return SimpleNamespace(path=path, size=size, lfs=SimpleNamespace(sha256=sha) if sha else None, blob_id="b")


class FakeHfApi:
    def __init__(self, tmp: Path, **kw):
        self.tmp = tmp
        self.write_ok, self.gated, self.private, self.fail5xx = True, False, True, False
        self.labels = {}  # path -> bytes (under labels/full)
        self.seeds = ["teacher_out/meta.json"] + [f"teacher_out/{g}/eval-00000.{e}" for g in GATES
                                                  for e in ("npz", "jsonl")]
        self.models = {f"{label.PARAKEET_PATH}/model.safetensors": "aa"}
        self.remote_json = {"teacher_out/meta.json": {"model": "m", "k": 8}}
        self.commits, self.downloads, self.snapshots = [], [], []
        self.__dict__.update(kw)

    def whoami(self):
        if self.fail5xx:
            raise HubError(503)
        return {"name": "me"}

    def auth_check(self, repo, repo_type=None, write=False):
        if write and not self.write_ok:
            raise HubError(403)
        if repo == label.COHERE_MODEL_ID and self.gated:
            raise GatedRepoError("gated")

    def dataset_info(self, repo):
        return SimpleNamespace(private=self.private)

    def list_repo_tree(self, repo, path_in_repo, recursive=True, repo_type=None, revision=None):
        if path_in_repo == "labels/full":
            return [item(p, len(b)) for p, b in self.labels.items()]
        if path_in_repo == "teacher_out":
            return [item(p) for p in self.seeds]
        if path_in_repo == label.PARAKEET_PATH:
            return [item(p, sha=s) for p, s in self.models.items()]
        return []

    def list_repo_files(self, repo, repo_type=None):
        return list(self.labels) + self.seeds

    def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None, local_dir=None):
        self.downloads.append(filename)
        if local_dir is None:
            p = self.tmp / "cache" / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.remote_json.get(filename) or json.loads(self.labels[filename])))
            return str(p)
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.labels[filename])
        return str(p)

    def snapshot_download(self, **kw):
        self.snapshots.append(kw)

    def create_commit(self, repo, **kw):
        self.commits.append(kw)


@pytest.fixture
def stepctx(box, monkeypatch):
    monkeypatch.setitem(sys.modules, "kitsune.parakeet", SimpleNamespace(
        PARAKEET_FILES={"model.safetensors": "aa"}, PARAKEET_PATH=label.PARAKEET_PATH, verify_model_dir=lambda d: []))

    def make(api=None, run=None, **env):
        api = api or FakeHfApi(box.kdir.parent)
        env = dict(box.env, KITSUNE_RUN_ID="r1", **env)
        return label.StepCtx(env=env, api=api, run=run, now=lambda: T0), api

    return make


@pytest.mark.parametrize("setup, rc", [
    ({}, 0),
    ({"write_ok": False}, 3),
    ({"gated": True}, 3),
    ({"private": False}, 3),
    ({"labels": {"labels/full/COMPLETE.json": b"{}"}}, 3),
    ({"labels": {"labels/full/LEASE.json": json.dumps({"container_id": "other", "heartbeat": T0 - 60,
                                                       "released": False}).encode()}}, 3),
    ({"labels": {"labels/full/LEASE.json": json.dumps({"container_id": "other", "heartbeat": T0 - 99999,
                                                       "released": False}).encode()}}, 0),
    ({"seeds": ["teacher_out/meta.json", "teacher_out/eval_jsut/eval-00000.npz"]}, 3),
    ({"models": {f"{label.PARAKEET_PATH}/model.safetensors": "bb"}}, 3),
    ({"fail5xx": True}, 1),
])
def test_plan(stepctx, box, setup, rc):
    ctx, api = stepctx(FakeHfApi(box.kdir.parent, **setup))
    assert label.run_step("plan", [], ctx) == rc
    if rc == 0:
        (commit,) = api.commits
        (op,) = commit["operations"]
        assert op.path_in_repo == "labels/full/LEASE.json"
        lease = json.loads(op.path_or_fileobj)
        assert lease["container_id"] == "c1" and lease["released"] is False and lease["run_id"] == "r1"
    else:
        assert api.commits == []


def test_plan_takes_a_live_lease_only_when_stealing(stepctx, box):
    lease = json.dumps({"container_id": "other", "heartbeat": T0 - 60, "released": False}).encode()
    ctx, api = stepctx(FakeHfApi(box.kdir.parent, labels={"labels/full/LEASE.json": lease}), KITSUNE_STEAL_LEASE="1")
    assert label.run_step("plan", [], ctx) == 0 and len(api.commits) == 1


def test_plan_refuses_relaunch_with_other_parakeet_settings(stepctx, box):
    api = FakeHfApi(box.kdir.parent, labels={"labels/full/parakeet_out/meta.json": b"{}"})
    api.remote_json["labels/full/parakeet_out/meta.json"] = {"k_tdt": 4, "k_ctc": 8, "ctc_dense_thr": 0.95,
                                                             "max_symbols": 10}
    ctx, _ = stepctx(api)
    assert label.run_step("plan", [], ctx) == 3


def test_pull_downloads_only_missing_files_and_never_overwrites(stepctx, box):
    have = box.kdir / "labels/full/teacher_out/galgame/train-00000.jsonl"
    have.parent.mkdir(parents=True)
    have.write_bytes(b"local!")  # same size as the remote bytes, other content: kept as is
    api = FakeHfApi(box.kdir.parent, labels={"labels/full/teacher_out/galgame/train-00000.jsonl": b"remote",
                                             "labels/full/teacher_out/galgame/train-00001.jsonl": b"new",
                                             "labels/full/LEASE.json": b"{}"})
    ctx, _ = stepctx(api)
    label.write_json_atomic(box.state / "label_remote.json",
                            {p: [len(b), None, "b"] for p, b in api.labels.items()})
    assert label.run_step("pull", [], ctx) == 0
    assert api.downloads == ["labels/full/teacher_out/galgame/train-00001.jsonl"]
    assert have.read_bytes() == b"local!"
    assert (box.kdir / "labels/full/teacher_out/galgame/train-00001.jsonl").read_bytes() == b"new"
    pats = [p for s in api.snapshots for p in s["allow_patterns"]]
    assert "second_out/galgame/*" in pats and f"{label.PARAKEET_PATH}/*" in pats
    assert all(s["revision"] == "rev1" for s in api.snapshots)
    ledger = json.loads((box.state / "label_ledger.json").read_text())
    assert "labels/full/teacher_out/galgame/train-00001.jsonl" in ledger["remote"]
    assert "labels/full/LEASE.json" not in ledger["remote"]
    have.write_bytes(b"longer local")  # a local file with another size is an integrity problem
    assert label.run_step("pull", [], ctx) == 65


def test_roots_writes_meta_once_and_flags_a_mismatch(stepctx, box):
    ctx, _ = stepctx()
    seed = box.kdir / "seed/teacher_out/meta.json"
    seed.parent.mkdir(parents=True)
    seed.write_text(json.dumps({"model": label.COHERE_MODEL_ID, "k": 8, "save_encoder": False, "language": "ja",
                                "punctuation": True, "decoder_prompt_ids": [1, 2]}))
    assert label.run_step("roots", [], ctx) == 0
    meta = json.loads((box.kdir / "labels/full/teacher_out/meta.json").read_text())
    assert meta["model_revision"] == label.COHERE_REVISION and meta["adopted_from"] == "teacher_out/meta.json@rev1"
    assert meta["decoder_prompt_ids"] == [1, 2]
    assert label.run_step("roots", [], ctx) == 0
    seed.write_text(json.dumps({"model": label.COHERE_MODEL_ID, "k": 4}))
    assert label.run_step("roots", [], ctx) == 65


@pytest.mark.parametrize("box_hyp, rc", [("こんにちは世界", 0), ("こんばんは世界", 69)])
def test_golden_tolerance(stepctx, box, monkeypatch, box_hyp, rc):
    ids = [f"j{i}" for i in range(64)]
    seed = box.kdir / "seed/teacher_out/eval_jsut/eval-00000.jsonl"
    seed.parent.mkdir(parents=True)
    seed.write_text("".join(json.dumps({"id": i, "hyp": "こんにちは世界"}) + "\n" for i in ids), encoding="utf-8")
    calib = box.kdir.parent / "calib"
    golden = json.loads((ROOT / "kitsune" / "parakeet_golden.json").read_text(encoding="utf-8"))

    def run(argv, timeout, env=None):
        assert env["HF_HUB_OFFLINE"] == "1" and "--force" not in argv
        out = Path(argv[argv.index("--out") + 1]) / "eval_jsut" / "eval-00000.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        if "02_teacher_pass" in argv[1]:
            rows = [{"id": i, "hyp": box_hyp if n < 8 else "こんにちは世界"} for n, i in enumerate(ids)]
        else:
            rows = [{"id": i, "hyp": t, "ctc_hyp": c} for i, t, c in zip(golden["ids"], golden["tdt"], golden["ctc"])]
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return 0

    from kitsune import parakeet_targets

    monkeypatch.setattr(parakeet_targets, "check_shard", lambda path, meta, shard_ids=None: [])
    ctx, _ = stepctx(run=run, KITSUNE_CALIB=str(calib))
    assert label.run_step("golden", [], ctx) == rc
    res = json.loads((box.state / "golden.json").read_text())
    assert res["parakeet"]["tdt_cer"] == 0 and res["parakeet"]["n"] == 32
    assert (res["cohere"]["cer"] <= 0.01) == (rc == 0)


def test_golden_fails_on_a_bad_parakeet_shard(stepctx, box, monkeypatch):
    from kitsune import parakeet_targets

    monkeypatch.setattr(label, "golden_cer", lambda box_rows, ref, key="hyp": (0.0, 32))
    monkeypatch.setattr(parakeet_targets, "check_shard", lambda path, meta, shard_ids=None: ["offsets not monotone"])
    ctx, _ = stepctx(run=lambda argv, timeout, env=None: 0, KITSUNE_CALIB=str(box.kdir.parent / "calib"))
    assert label.run_step("golden", [], ctx) == 69


def consumer_tree(box, t_ids, p_ids, second=True):
    data = box.kdir / "data"
    write_manifest(data, [("reazon_small", "train", 0, ["a", "b", "c"])])
    (data / "shards/eval_jsut/eval-00000.parquet").unlink()  # only the reazon shard matters here
    store.write_manifest(data, [s for s in store.read_manifest(data) if s.source == "reazon_small"])
    teacher_npz(box.kdir / "labels/full/teacher_out/reazon_small/train-00000.npz", t_ids, 3)
    teacher_npz(box.kdir / "labels/full/parakeet_out/reazon_small/train-00000.npz", p_ids, 3)
    (box.kdir / "labels/full/teacher_out/meta.json").write_text(json.dumps({"k": 8}))
    if second:
        s = box.kdir / "labels/full/second_out/reazon_small/train-00000.jsonl"
        s.parent.mkdir(parents=True, exist_ok=True)
        s.write_text("".join(json.dumps({"id": i}) + "\n" for i in t_ids))


@pytest.mark.parametrize("p_ids, second, drift, want", [
    (["a", "b"], True, False, []),
    (["a"], True, False, ["Parakeet ids != teacher ids"]),
    (["a", "b"], False, False, ["second opinion missing"]),
    (["a", "b"], True, True, ["gate baselines"]),
])
def test_consumer_check_local(stepctx, box, monkeypatch, p_ids, second, drift, want):
    consumer_tree(box, ["a", "b"], p_ids, second)
    import kitsune.evaluate as ev

    def baselines(root, sets, check=True):
        if drift:
            raise ValueError("eval_jsut corpus CER drifted by 0.3 pp")
        return {}

    monkeypatch.setattr(ev, "teacher_baselines", baselines)
    ctx, _ = stepctx()
    problems = label.consumer_local_problems(ctx)
    assert len(problems) == len(want) and all(w in p for w, p in zip(want, problems)), problems


def test_consumer_check_step_is_65_on_problems(stepctx, box, monkeypatch):
    from kitsune import extent

    import launch

    consumer_tree(box, ["a", "b"], ["a"])
    monkeypatch.setattr(extent, "load_record", lambda p: {})
    monkeypatch.setattr(extent, "pull_plan", lambda cfg, record, files: {"problems": []})
    monkeypatch.setattr(launch, "extent_problems", lambda files, cfg, record: ["labels/full/COMPLETE.json missing"])
    monkeypatch.setattr(launch, "selection_problems", lambda path, name, cfg, have=None: [])
    monkeypatch.setattr(label, "consumer_local_problems", lambda ctx: [])
    ctx, _ = stepctx()
    # neither check can want the seal: F7 writes COMPLETE.json only after both passed (the first label box stopped
    # at F6 on exactly this problem)
    assert label.run_step("consumer-check", [], ctx) == 0
    assert label.run_step("consumer-check", ["--hub"], ctx) == 0
    monkeypatch.setattr(launch, "extent_problems", lambda files, cfg, record: ["labels/full/COMPLETE.json missing",
                                                                              "reazon_large/train-00001 missing"])
    assert label.run_step("consumer-check", ["--hub"], ctx) == 65
    assert json.loads((box.state / "consumer_check_hub.json").read_text())["n_problems"] == 2  # per config
    monkeypatch.setattr(launch, "extent_problems", lambda files, cfg, record: ["labels/full/COMPLETE.json missing"])
    monkeypatch.setattr(label, "consumer_local_problems", lambda ctx: ["reazon_small/train-00000: Parakeet ids"])
    assert label.run_step("consumer-check", [], ctx) == 65
    assert json.loads((box.kdir / "labels/full/reports/consumer_check.json").read_text())["n_problems"] == 1


def test_teacher_done_checks_ids_rows_and_jsonl(tmp_path):
    pytest.importorskip("kitsune.labelpass")
    npz = tmp_path / "t.npz"
    teacher_npz(npz, ["a", "c"], 3)
    assert label.teacher_done(npz, ["a", "b", "c"])
    assert not label.teacher_done(npz, ["c", "b", "a"])  # order
    assert not label.teacher_done(npz, ["a", "b", "c", "d"])  # n_scanned
    npz.with_suffix(".jsonl").unlink()
    assert not label.teacher_done(npz, ["a", "b", "c"])


# ------------------------------------------------------------------------------------------------------------ loop


def deadline_in(box, minutes):
    """A deadline that makes the budget end fire `minutes` from now (end_margin_min is 60)."""
    (box.state / "deadline").write_text(str(int(box.clock.t + (60 + minutes) * 60)))


def test_step_order_then_lanes_only_after_golden(box):
    deadline_in(box, 10)
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    names = [n for n in runner.names() if n not in ("sync", "second")]  # the first sync cycle starts at once
    assert names[:5] == ["step-plan", "step-pull", "step-models", "step-selftest", "step-roots"]
    assert names[5] == "ingest" and names[6] == "step-golden"
    assert names[7:10] == ["cohere-0", "cohere-1", "parakeet"]
    assert st["steps_done"] == ["plan", "pull", "models", "selftest", "roots", "golden"]
    assert st["final"]["class"] == "budget" and fin.calls[0][0] == "destroy"
    assert "--allow-empty" not in fin.calls[0][2] and fin.calls[0][2][:2] == ["--job", "label"]
    assert all(p.terminated for p in runner.procs if p.name in FakeRunner.lane_names)
    end = json.loads((box.state / "label_end.json").read_text())
    assert end["class"] == "budget" and end["machine_id"] == "m1" and end["run_id"] == st["run_id"]
    assert rc == 1


def test_golden_waits_for_the_eval_jsut_shard(box):
    store.write_manifest(box.kdir / "data", [])
    deadline_in(box, 10)
    c, runner, fin = box.make()
    run_ctl(c)
    assert "step-golden" not in runner.names() and "cohere-0" not in runner.names()


def test_argv_never_forces_and_cohere_lanes_adopt_only_the_gates(box):
    deadline_in(box, 5)
    c, runner, _ = box.make()
    run_ctl(c)
    for name, argv, env in runner.started:
        assert "--force" not in argv
        if name.startswith("cohere-"):
            i = argv.index("--adopt-only")
            assert argv[i + 1:i + 4] == GATES and "--strict-existing" in argv
            assert argv[argv.index("--shard-mod") + 1] == "2" and argv[argv.index("--shard-rem") + 1] == name[-1]
            assert argv[argv.index("--workers") + 1] == "4"  # max(2, round(0.3 * 13))
            assert env["HF_HUB_OFFLINE"] == "1" and env["OMP_NUM_THREADS"] == "2" and env["KITSUNE_RUN_ID"]
        if name == "parakeet":
            assert argv[argv.index("--workers") + 1] == "5" and argv[argv.index("--max-symbols") + 1] == "10"
            assert argv[argv.index("--follow") + 1] == str(box.state / "ingest.done")
        if name == "ingest":
            assert "HF_HUB_OFFLINE" not in env and argv[argv.index("--extent-config") + 1] == "configs/full.json"


def test_step_retries_transient_failures_then_host_failure(box):
    deadline_in(box, 10)
    tries = []

    def behave(name, argv):
        if name == "step-plan":
            tries.append(box.clock.t)
            return 1
        return 0 if name.startswith("step-") else None

    c, runner, fin = box.make(behave)
    rc, st = run_ctl(c)
    assert len(tries) == 3 and tries[1] - tries[0] >= 60 and tries[2] - tries[1] >= 120
    assert st["final"]["class"] == "host_failure" and fin.calls[0][0] == "destroy"
    assert "--allow-empty" in fin.calls[0][2]  # no lane had started


def test_refusal_stops_before_any_lane_with_allow_empty(box):
    c, runner, fin = box.make(lambda name, argv: 3 if name == "step-plan" else 0)
    rc, st = run_ctl(c)
    # no lease was taken: finish must not sync (it would overwrite another box's live lease)
    assert st["final"]["class"] == "refusal" and fin.calls == [("stop", fin.calls[0][1], ["--job", "label",
                                                                                           "--allow-empty", "--no-sync"])]
    assert runner.names() == ["step-plan"]


def test_rearm_with_labels_on_disk_runs_selftest_as_a_restart(box):
    """onstart.sh --rearm moves label.json aside: the restart disk floor must come from the tree on disk."""
    c, runner, fin = box.make(lambda name, argv: 3 if name == "step-selftest" else 0)
    (c.teacher / "meta.json").parent.mkdir(parents=True, exist_ok=True)
    (c.teacher / "meta.json").write_text("{}")
    run_ctl(c)
    (argv,) = [a for n, a, _ in runner.started if n == "step-selftest"]
    assert argv[-1] == "--restart"


def test_fresh_box_runs_selftest_without_restart(box):
    c, runner, fin = box.make(lambda name, argv: 3 if name == "step-selftest" else 0)
    run_ctl(c)
    (argv,) = [a for n, a, _ in runner.started if n == "step-selftest"]
    assert "--restart" not in argv


def test_restart_skips_recorded_steps_and_does_not_pull_again(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"], lanes_started=True)
    (box.state / "ingest.done").touch()
    deadline_in(box, 5)
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    names = runner.names()
    assert names[:5] == ["step-selftest", "ingest", "cohere-0", "cohere-1", "parakeet"]
    assert runner.started[0][1][-1] == "--restart"
    assert not {"step-plan", "step-pull", "step-models", "step-roots", "step-golden"} & set(names)
    assert not (box.state / "ingest.done").exists()  # deleted before the ingest lane restarted
    assert st["run_id"] == "r1" and st["lanes"]["ingest"]["starts"] == 1


def test_stale_heartbeat_kills_the_group_and_four_failures_end_the_box(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    c, runner, fin = box.make(beats=False)  # the lanes never touch their heartbeats: hung
    rc, st = run_ctl(c)
    killed = [p for p in runner.procs if p.name == "cohere-0"]  # hang_min 20 < ingest_hang_min 30: first to go
    assert len(killed) == 4 and all(p.terminated for p in killed)
    assert st["final"]["class"] == "host_failure" and "cohere-0 failed 4 times" in st["final"]["reason"]
    assert "without progress" in st["final"]["reason"] and fin.calls[0][0] == "destroy"
    assert box.clock.t - T0 >= 4 * 20 * 60 + 30 + 120 + 300  # 4 hang windows and 3 restart delays


def test_lane_failures_with_progress_do_not_count(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    deadline_in(box, 60)
    prog = box.state / "lanes" / "cohere-0.jsonl"

    def behave(name, argv):
        if name == "cohere-0":
            prog.parent.mkdir(parents=True, exist_ok=True)
            with open(prog, "a") as f:
                f.write(json.dumps({"stem": "x", "hours": 0.0, "wall_s": 1, "adopted": True}) + "\n")
            return 1
        return 0 if name.startswith("step-") else None

    c, runner, fin = box.make(behave)
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "budget"
    assert runner.names().count("cohere-0") > 4 and st["lanes"]["cohere-0"]["fails_no_progress"] == 0


def test_lane_exit_65_stops(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    c, runner, fin = box.make(lambda name, argv: 65 if name == "parakeet" else (0 if name.startswith("step-") else None))
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "integrity" and fin.calls[0][0] == "stop" and "--allow-empty" not in fin.calls[0][2]


def test_label_end_stop_turns_destroy_into_stop(box):
    deadline_in(box, 5)
    c, runner, fin = box.make(KITSUNE_LABEL_END="stop")
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "budget" and st["final"]["action"] == "stop" and fin.calls[0][0] == "stop"


def test_sync_cycles_every_20_min_with_the_sync_first_and_explicit_sources(box):
    """The sync (it carries the lease heartbeat) runs first in a cycle, and the first cycle starts at once."""
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    deadline_in(box, 65)
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    seconds = [(argv, env) for n, argv, env in runner.started if n == "second"]
    syncs = [argv for n, argv, _ in runner.started if n == "sync"]
    assert len(seconds) == 4 and len(syncs) == 4 and len(st["syncs"]) == 4
    cyc = [n for n in runner.names() if n in ("sync", "second")]
    assert cyc == ["sync", "second"] * 4
    t = [s["t"] for s in st["syncs"]]
    assert all(1190 <= b - a <= 1260 for a, b in zip(t, t[1:]))
    argv, env = seconds[0]
    srcs = argv[argv.index("--sources") + 1:]
    assert srcs == ["reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame", "eval_emilia"]
    assert not set(GATES) & set(srcs) and "--strict-existing" in argv and env["HF_HUB_OFFLINE"] == "1"
    assert argv[argv.index("--judge") + 1] == "parakeet"
    assert [("--no-infra" in a) for a in syncs] == [True, True, False, True]  # infra every 3rd cycle
    assert all(a[a.index("--job") + 1] == "label" and "--sync-only" in a for a in syncs)


def test_a_sync_integrity_exit_stops_the_box(box):
    """A write-once conflict or a sealed root (finish.py 65) stops the box at once, not after hours of labelling."""
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    c, runner, fin = box.make(lambda n, a: 65 if n == "sync" else (0 if n.startswith("step-") else None))
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "integrity" and "write-once" in st["final"]["reason"]
    assert fin.calls[0][0] == "stop"


def test_a_lane_refusal_is_not_a_host_failure(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    c, runner, fin = box.make(lambda n, a: 3 if n == "parakeet" else (0 if n.startswith("step-") else None))
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "refusal" and fin.calls[0][0] == "stop"


def test_02b_integrity_exit_stops(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    c, runner, fin = box.make(lambda n, a: 65 if n == "second" else (0 if n.startswith("step-") else None))
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "integrity" and fin.calls[0][0] == "stop"


def test_slow_host_floor_after_60_min(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    lanes = box.state / "lanes"
    lanes.mkdir()
    for i in range(2):  # 150x realtime each: 300x together < 400x, after 3600 s of work
        (lanes / f"cohere-{i}.jsonl").write_text("".join(
            json.dumps({"stem": f"s{j}", "hours": 25 / 6, "wall_s": 100, "adopted": False}) + "\n" for j in range(36))
            + json.dumps({"stem": "a", "hours": 1000, "wall_s": 1, "adopted": True}) + "\n")
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "slow_host" and fin.calls[0][0] == "destroy"
    assert st["rates"]["cohere_xrt"] == pytest.approx(300, rel=0.01)


def test_fast_host_passes_the_floor(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    deadline_in(box, 40)
    lanes = box.state / "lanes"
    lanes.mkdir()
    (lanes / "cohere-0.jsonl").write_text("".join(
        json.dumps({"stem": f"s{j}", "hours": 15, "wall_s": 100, "adopted": False}) + "\n" for j in range(40)))
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "budget"


def test_janitor_prunes_verified_train_shards_only_and_drives_the_hold(box, monkeypatch):
    data = box.kdir / "data"
    infos = write_manifest(data, [("galgame", "train", 0, ["a"]), ("galgame", "train", 1, ["b"]),
                                  ("galgame", "eval", 0, ["c"]), ("galgame", "train", 2, ["d"])])
    for sh in infos:
        stem = Path(sh.path).stem
        for root in ("teacher_out", "parakeet_out"):
            if stem != "train-00002" or root == "teacher_out":  # train-00002 has no Parakeet npz yet
                p = box.kdir / "labels/full" / root / "galgame" / f"{stem}.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"npz")
    c, _, _ = box.make()
    c.state = c.fresh_state()
    checked = []
    monkeypatch.setattr(c, "teacher_done", lambda npz, ids: checked.append(("t", npz.stem, ids)) or True)
    monkeypatch.setattr(c, "parakeet_done", lambda npz, ids: npz.stem != "train-00001")
    free = {"v": 500e9}
    monkeypatch.setattr(c, "disk_free", lambda: free["v"])
    c.janitor()
    assert not (data / infos[0].path).exists()  # both npz verify: pruned
    assert (data / infos[1].path).exists()  # the Parakeet npz does not verify
    assert (data / infos[2].path).exists()  # eval: never pruned
    assert (data / infos[3].path).exists()  # no Parakeet npz
    assert checked[0] == ("t", "train-00000", ["a"])
    pruned = [json.loads(line) for line in (data / "pruned.jsonl").read_text().splitlines()]
    assert [p["path"] for p in pruned] == [infos[0].path] and pruned[0]["bytes"] > 0
    hold = box.state / "ingest.hold"
    assert not hold.exists()
    c.k["backlog_gb"] = 0  # any backlog is over the limit
    c.janitor()
    assert hold.exists()
    c.k["backlog_gb"] = 80
    free["v"] = 50e9  # backlog fine, but free space between 45 and 60 GB: the hold stays
    c.janitor()
    assert hold.exists()
    free["v"] = 100e9
    c.janitor()
    assert not hold.exists()
    free["v"] = 40e9
    c.janitor()
    assert hold.exists()
    free["v"] = 10e9
    with pytest.raises(label.Ended):
        c.janitor()
    assert c.state["final"]["class"] == "host_failure"


def test_budget_end_at_deadline_minus_60(box):
    write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"])
    deadline_in(box, 30)
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "budget"
    assert box.clock.t >= float((box.state / "deadline").read_text()) - 3600


def lanes_done_state(box, **kw):
    lanes = {n: {"starts": 1, "fails_no_progress": 0, "last_progress_t": None, "rc": 0, "done": True}
             for n in ("ingest", "cohere-0", "cohere-1", "parakeet")}
    (box.state / "ingest.done").touch()
    return write_state(box.state, steps_done=["plan", "pull", "models", "selftest", "roots", "golden"], lanes=lanes,
                       lanes_started=True, **kw)


def test_finalize_is_refused_after_deadline_minus_90(box, monkeypatch):
    lanes_done_state(box)
    deadline_in(box, 20)  # deadline - 80 min: past the finalize margin
    monkeypatch.setattr(label.Controller, "unlabelled", lambda self: [])
    monkeypatch.setattr(label.Controller, "missing_second", lambda self: [])
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    assert st["final"]["class"] == "budget" and "too late to finalize" in st["final"]["reason"]
    assert st["finalize_done"] == [] and fin.calls[0][0] == "destroy"


def test_finalize_order_and_final_destroy(box, monkeypatch):
    lanes_done_state(box)
    order = []
    monkeypatch.setattr(label.Controller, "unlabelled", lambda self: [])
    monkeypatch.setattr(label.Controller, "missing_second", lambda self: [])
    for f in ("f_extent", "f_selections", "f_reports", "f_final_sync"):
        monkeypatch.setattr(label.Controller, f, lambda self, f=f: order.append(f))
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    steps = [(n, a[a.index("step") + 1:]) for n, a, _ in runner.started if n.startswith("step-")]
    assert "second" in runner.names()  # the final 02b ran before finalize
    assert order == ["f_extent", "f_selections", "f_reports", "f_final_sync"]
    assert steps[1:] == [("step-consumer-check", ["consumer-check"]),
                         ("step-consumer-check", ["consumer-check", "--hub"]), ("step-seal", ["seal"])]
    assert st["finalize_done"] == ["F1-extent", "F2-selections", "F3-reports", "F4-lease", "F4-consumer-check",
                                   "F5-final-sync",
                                   "F6-consumer-check-hub", "F7-seal"]
    assert st["final"]["class"] == "success" and fin.calls == [("destroy", "success: labels complete",
                                                                ["--job", "label"])]
    assert rc == 0


def test_finalize_selections_and_final_sync_argv(box, monkeypatch):
    lanes_done_state(box)
    c, runner, fin = box.make()
    c.state = json.loads((box.state / "label.json").read_text())
    c.f_selections()
    sel = [a for n, a, _ in runner.started if n == "make_selection"]
    assert sel == [["py", "scripts/make_selection.py", "--config", "configs/full.json"],
                   ["py", "scripts/make_selection.py", "--config", "configs/full_sub3k.json", "--from-selection",
                    "labels/full/selections/full.parquet"]]
    c.f_final_sync()
    assert [a[2:] for n, a, _ in runner.started if n in ("sync", "verify")] == [["--job", "label", "--sync-only"],
                                                                              ["--job", "label", "--verify-only"]]


def test_final_sync_unverifiable_stops(box):
    lanes_done_state(box)
    c, runner, fin = box.make(lambda n, a: 1 if n == "verify" else 0)
    c.state = json.loads((box.state / "label.json").read_text())
    with pytest.raises(label.Ended):
        c.f_final_sync()
    assert [n for n in runner.names()] == ["sync", "verify", "sync", "verify"]
    assert c.state["final"]["class"] == "sync_unverifiable" and fin.calls[0][0] == "stop"


def test_unlabelled_shards_after_all_lanes_finished_are_an_integrity_stop(box, monkeypatch):
    lanes_done_state(box)
    c, runner, fin = box.make()
    rc, st = run_ctl(c)  # the eval_jsut shard in the manifest has no npz
    assert st["final"]["class"] == "integrity" and "unlabelled" in st["final"]["reason"] and fin.calls[0][0] == "stop"


def test_unknown_exception_stops(box, monkeypatch):
    def boom(self):
        raise RuntimeError("bug")

    monkeypatch.setattr(label.Controller, "loop", boom)
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    assert rc == 1 and st["final"]["class"] == "unknown" and "bug" in st["final"]["reason"]
    assert fin.calls[0][0] == "stop"


def test_recorded_final_without_halt_is_replayed(box):
    write_state(box.state, final={"action": "destroy", "class": "success", "reason": "success: labels complete",
                                  "wall": T0, "allow_empty": False})
    c, runner, fin = box.make()
    assert c.run() == 0
    assert fin.calls == [("destroy", "success: labels complete", ["--job", "label"])] and runner.started == []
    (box.state / "halt").touch()
    c, runner, fin = box.make()
    assert c.run() == 0 and fin.calls == [] and runner.started == []


def test_corrupt_state_is_a_recorded_stop(box):
    (box.state / "label.json").write_bytes(b"\0\0\0")
    c, runner, fin = box.make()
    rc, st = run_ctl(c)
    assert st["final"]["action"] == "stop" and "unreadable" in st["final"]["reason"]
    assert fin.calls[0][0] == "stop" and runner.started == []
    assert list(box.state.glob("label.json.corrupt-*"))


def test_controller_heartbeat_is_fresh(box):
    deadline_in(box, 5)
    c, runner, fin = box.make()
    run_ctl(c)
    assert (box.state / "label_hb").stat().st_mtime >= box.clock.t - label.BEAT_S - label.LOOP_S


def test_cli_rejects_unknown_step(capsys):
    assert label.main(["step", "nope"]) == 2
