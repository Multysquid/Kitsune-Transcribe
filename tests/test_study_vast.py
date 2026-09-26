"""The size study's box on vast (vast/launch.py --job study, bootstrap.sh, supervise.py, finish.py, kitsune.extent's
sizing): the relaxed per-launch host filter and GPU count per box, the disk with both label stores and the box's
checkpoints, the refusals before renting (a pending PREREG, a selection hash that is not PREREG's, a student file or a
config missing, numbers already written), the box's students pulled and only those, the queue supervised (restarted
after a crash, stopped on a pre-registered halt, destroyed once done), the lean uploads and the Hub back-off. No
network, no vastai CLI, no GPU.
"""
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vast"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
launch = importlib.import_module("launch")
finish = importlib.import_module("finish")
supervise = importlib.import_module("supervise")
from kitsune import extent, prereg  # noqa: E402
from kitsune import study_queue as Q  # noqa: E402
from test_study_box import BOXES_CONTRACT, study_rules  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST_IMAGE = "ghcr.io/multysquid/kitsune-train@sha256:" + "ab" * 32
DATA = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
SIZING = dict(down_gb=58.0, shard_gb=45.0, sel_gb=33.0, stores=2, labels_gb=3.3, hours=1110.0, extra_gb=150.0,
              disk_gb=450, rebuild_timeout_min=54)


@pytest.fixture(autouse=True)
def box_rules(monkeypatch):
    """The rules with the box plans (CONTRACT.md 6; kitsune.prereg carries them once D1 is merged)."""
    orig = prereg.rules

    def rules(sidecar=None):
        r = orig(sidecar)
        r.setdefault("boxes", copy.deepcopy(BOXES_CONTRACT))
        return r

    monkeypatch.setattr(prereg, "rules", rules)


# ============================================================================================================ launch


class FakeVastai:
    def __init__(self, searches):
        self.searches, self.calls = list(searches), []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[1:3] == ["search", "offers"]:
            out = json.dumps(self.searches.pop(0))
        elif argv[1:3] == ["create", "instance"]:
            out = json.dumps({"success": True, "new_contract": 4242})
        else:
            raise AssertionError(f"unexpected vastai call {argv}")
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


OFFERS_4X = [{"id": 71, "machine_id": 7, "gpu_name": "A100 SXM4", "gpu_ram": 40960, "num_gpus": 4,
              "dph_total": 2.138, "reliability": 0.941, "inet_down_cost": 0.005, "inet_up_cost": 0.01}]


@pytest.fixture
def study_launch(monkeypatch):
    """launch.main for --job study with the vastai CLI faked and the Hub side passing: hf_preflight and
    extent_preflight (whose extra_gb, the box's checkpoints, is recorded), study_preflight (its own tests below)."""
    seen = {}
    monkeypatch.setattr(launch, "config_at", lambda sha, c: copy.deepcopy(DATA))
    monkeypatch.setattr(launch, "hf_preflight", lambda data, out, c: ("d" * 40, []))

    def extent_preflight(data, rev, cfg, extra_gb=0.0):
        seen["extra_gb"] = extra_gb
        return [], dict(SIZING, extra_gb=extra_gb)

    monkeypatch.setattr(launch, "extent_preflight", extent_preflight)
    monkeypatch.setattr(launch, "study_preflight", lambda *a: (seen.setdefault("preflight", []).append(a) or [],
                                                               ["study preflight ok"]))

    def go(searches, *args):
        fake = FakeVastai(searches)
        monkeypatch.setattr(launch.shutil, "which", lambda name: "/fake/vastai" if name == "vastai" else None)
        monkeypatch.setattr(launch.subprocess, "run", fake)
        rc = launch.main(["--job", "study", "--data-repo", "Multy123/kitsune-data", "--out-repo",
                          "Multy123/kitsune-runs", "--sha", SHA, "--image", DIGEST_IMAGE, "--skip-git-checks", *args])
        return rc, fake, seen
    return go


def env_of(create: list[str]) -> dict[str, str]:
    value = create[create.index("--env") + 1]
    pairs = value.split(" ")
    return dict(p.split("=", 1) for p in pairs[1::2])


def test_launch_study_box_a_rents_four_a100s_with_the_relaxed_filter(study_launch, capsys):
    rc, fake, seen = study_launch([OFFERS_4X], "--box", "A", "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    search = next(c for c in fake.calls if c[1:3] == ["search", "offers"])
    q = search[3]
    assert "num_gpus=4" in q and f"reliability>={launch.STUDY_RELIABILITY}" in q and "reliability>=0.98" not in q
    assert "gpu_name in [A100_SXM4,A100_PCIE]" in q and "gpu_ram<=48" in q and "cpu_cores_effective>=48" in q
    assert "disk_space>=450" in q and search[search.index("--storage") + 1] == "450"
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    env = env_of(create)
    assert env["KITSUNE_JOB"] == "study" and env["KITSUNE_BOX"] == "A" and env["KITSUNE_CONFIG"] == "study/data.json"
    assert env["KITSUNE_N_GPUS"] == "4"  # the queue refuses another count
    assert env["KITSUNE_OUT_REPO"] == "Multy123/kitsune-runs" and env["TZ"] == "UTC"
    assert env["KITSUNE_MAX_HOURS"] == f"{launch.STUDY_HOURS['A'][1] + 54 / 60:g}"
    assert env["KITSUNE_REBUILD_TIMEOUT_MIN"] == "54" and "HF_TOKEN" not in " ".join(create)
    assert create[create.index("--label") + 1].startswith("kitsune-study-A-")
    assert seen["extra_gb"] == Q.study_extra_gb("A", n_gpus=4) > 0
    assert "x 2 stores" in out and "study preflight ok" in out and "4x A100" in out
    assert f"x ~{launch.STUDY_HOURS['A'][0]:g} h" in out


@pytest.mark.parametrize("box_name", ["replicate", "shakedown"])
def test_launch_study_one_gpu_boxes(study_launch, capsys, box_name):
    one = [dict(OFFERS_4X[0], id=72, num_gpus=1, dph_total=0.6)]
    rc, fake, _ = study_launch([one], "--box", box_name, "--dry-run")
    assert rc == 0, capsys.readouterr().out
    q = next(c for c in fake.calls if c[1:3] == ["search", "offers"])[3]
    assert "num_gpus=1" in q and "cpu_cores_effective>=12" in q
    assert not any(c[1:3] == ["create", "instance"] for c in fake.calls)


def test_launch_study_needs_a_box_and_the_hf_preflight(study_launch, capsys):
    with pytest.raises(SystemExit):
        study_launch([OFFERS_4X])
    rc, fake, _ = study_launch([OFFERS_4X], "--box", "A", "--no-hf-check", "--disk-gb", "500", "--max-hours", "9",
                               "--yes")
    assert rc == 1 and "--job study needs the HF preflight" in capsys.readouterr().out
    assert not any(c[1:3] == ["create", "instance"] for c in fake.calls)


def test_launch_study_refuses_what_its_preflight_finds(study_launch, monkeypatch, capsys):
    monkeypatch.setattr(launch, "study_preflight", lambda *a: (["study/PREREG.json has 3 pending field(s)"], []))
    rc, fake, _ = study_launch([OFFERS_4X], "--box", "A", "--yes")
    assert rc == 1 and "refusing to create" in capsys.readouterr().out
    assert not any(c[1:3] == ["create", "instance"] for c in fake.calls)


# ------------------------------------------------------------------------------------------------ study_preflight


class PreflightHub:
    def __init__(self, files: dict, runs: set):
        self.files, self.runs = files, runs

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return sorted(self.files)

    def get_paths_info(self, repo, paths, repo_type=None, revision=None):
        return [SimpleNamespace(path=p, lfs=SimpleNamespace(sha256=self.files[p])) for p in paths if p in self.files]

    def file_exists(self, repo, path, repo_type=None):
        return path in self.runs


def data_files(rules: dict, boxes=("A", "B", "replicate", "shakedown"), sel_sha="ab" * 32) -> dict:
    files = {DATA["selection"]: sel_sha}
    for b in boxes:
        ctc = set(Q.box_ctc_students(b, rules))
        for s in Q.box_students(b, rules):
            for n in launch.STUDENT_FILES + ((launch.CTC_CARD,) if s in ctc else ()):
                files[f"{s}/{n}"] = None
        for d in Q.box_extra_dirs(b, rules):
            files[f"{d}/config.json"] = None
    return files


@pytest.fixture
def preflight(monkeypatch, tmp_path):
    rules = study_rules()

    def go(box_name, *, files=None, runs=(), prereg_json=None, missing_configs=(), numbers_rules=None):
        """numbers_rules: the rules_sha256 in the runs repo's numbers files (default: this PREREG's)."""
        pr = prereg_json if prereg_json is not None else dict(prereg.rules(), manifest=dict(
            prereg.rules()["manifest"], selection_sha256="ab" * 32))
        if prereg_json is None:  # a filled PREREG: nothing pending
            pr = json.loads(json.dumps(pr).replace('"pending"', '"x"'))
        hub = PreflightHub(files if files is not None else data_files(rules), set(runs))
        here = hashlib.sha256(prereg.rules_json(pr)).hexdigest()

        def download(repo, path, repo_type=None, revision=None, local_dir=None):
            assert repo == "Multy123/kitsune-runs" and path in runs, (repo, path)
            p = Path(local_dir) / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"box": "A", "rules_sha256": numbers_rules or here}), encoding="utf-8")
            return str(p)

        monkeypatch.setattr(launch, "_hub", lambda: (hub, download))

        def git(*a):
            if a[:2] == ("cat-file", "-e") and any(a[2].endswith(m) for m in missing_configs):
                raise subprocess.CalledProcessError(1, a)
            return ""

        monkeypatch.setattr(launch, "git", git)
        monkeypatch.setattr(launch, "git_show", lambda sha, path: json.dumps(pr).encode())
        return launch.study_preflight("Multy123/kitsune-data", "d" * 40, "Multy123/kitsune-runs", SHA, box_name,
                                      DATA)
    return go, rules


def test_study_preflight_passes_a_ready_box(preflight):
    go, rules = preflight
    problems, notes = go("A")
    assert problems == [] and any("= PREREG's" in n for n in notes)
    problems, notes = go("replicate", runs={"study/PREREG_numbers_A.json"})
    assert problems == [] and any("takes its numbers" in n for n in notes)


def test_study_preflight_refusals(preflight):
    go, rules = preflight
    raw = prereg.rules()  # pending fields: refused for a study box, a note for the shakedown
    assert any("pending field" in p for p in go("A", prereg_json=raw)[0])
    problems, notes = go("shakedown", prereg_json=raw)
    assert problems == [] and any("pending field" in n for n in notes)
    files = data_files(rules, sel_sha="cd" * 32)  # another selection than PREREG's
    assert any("PREREG's manifest.selection_sha256" in p for p in go("A", files=files)[0])
    files = data_files(rules)
    del files["students/study/t06/README.md"]
    assert any("students/study/t06 lacks ['README.md']" in p for p in go("A", files=files)[0])
    files = data_files(rules)
    del files["students/study/p03/MODEL_CARD.md"]
    assert any("p03 lacks ['MODEL_CARD.md']" in p for p in go("B", files=files)[0])
    assert not any("p03" in p for p in go("A", files=files)[0])  # box A does not pull the Parakeet students
    assert any("already holds study/PREREG_numbers_A.json" in p
               for p in go("A", runs={"study/PREREG_numbers_A.json"})[0])
    assert any("has no study/PREREG_numbers_A.json" in p for p in go("replicate")[0])
    # box A's numbers written under other rules than this commit's: the replicate box would refuse them after its boot
    assert any("written under the rules" in p
               for p in go("replicate", runs={"study/PREREG_numbers_A.json"}, numbers_rules="0" * 64)[0])
    assert any("calib-study-t06.json does not exist" in p
               for p in go("A", missing_configs=("calib-study-t06.json",))[0])


# ======================================================================================================= sizing


def test_sizing_counts_both_stores_the_parakeet_labels_and_the_extra():
    rec = {"schema": 1, "sources": {"reazon_small": {"hours": 100.0, "bytes": 3e9, "inputs": [
        {"ordinal": 0, "bytes": 3e9, "stems": [{"stem": "train-00000", "step": "reazon_small", "hours": 100.0,
                                                 "shard_bytes": 3e9}]}]}}}
    cfg = {"extent": {"name": "full", "root": "labels/full", "inputs": {}}, "sources": ["reazon_small"],
           "eval_sets": []}
    one = extent.sizing(rec, cfg)
    both = extent.sizing(rec, dict(cfg, pull_parakeet=True), extra_gb=100)
    assert one["stores"] == 1 and both["stores"] == 2 and both["extra_gb"] == 100
    assert both["labels_gb"] == pytest.approx(100 * (extent.LABEL_GB_PER_HOUR + extent.PARAKEET_GB_PER_HOUR))
    need = extent.DISK_MARGIN * (extent.DISK_BASE_GB + 3 + 2 * 3 + both["labels_gb"] + 100)
    assert both["disk_gb"] == int(-(-need // extent.DISK_STEP_GB) * extent.DISK_STEP_GB) > one["disk_gb"]


def test_study_extra_gb_follows_the_box():
    r = study_rules()
    a = Q.study_extra_gb("A", r)
    params = sum(r["runs"][x]["params_total"] for x in r["boxes"]["A"]["runs"])
    assert a > params * Q.STATE_BYTES_PER_PARAM * Q.STATES_PER_RUN / 1e9
    assert Q.study_extra_gb("shakedown", r) == Q.SHAKEDOWN_EXTRA_GB
    assert Q.study_extra_gb("replicate", r) < 20


# ==================================================================================================== bootstrap


def test_bootstrap_study_pulls_both_label_roots_and_only_the_boxs_students(tmp_path):
    """KITSUNE_JOB=study: the data block (no student) with pull_parakeet; the box's student dirs, each with its
    STUDENT_FILES (a Parakeet one with its CC-BY-4.0 card), and no other student."""
    import test_bootstrap_extent as tb

    rules = study_rules()
    boxmod = _make_box(tmp_path, tb, rules)
    cfg = {k: v for k, v in tb.make_cfg().items() if k != "student"}
    cfg["pull_parakeet"] = True
    boxmod.config(cfg)
    r = boxmod.helper("plan", KITSUNE_JOB="study", KITSUNE_BOX="A")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((boxmod.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    pulled = {f.split("/")[2] for f in plan["files"] if f.startswith("students/study/")}
    assert pulled == {Path(s).name for s in Q.box_students("A", rules)}
    assert any("parakeet_out" in f for f in plan["files"] + plan["explicit"])
    r = boxmod.helper("plan", KITSUNE_JOB="study", KITSUNE_BOX="B")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((boxmod.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    assert "students/study/p03/MODEL_CARD.md" in plan["files"]
    (boxmod.remote_dir / "students/study/p01/MODEL_CARD.md").unlink()
    r = boxmod.helper("plan", KITSUNE_JOB="study", KITSUNE_BOX="B")
    assert r.returncode == 3 and "students/study/p01/MODEL_CARD.md" in r.stderr


def _make_box(tmp_path, tb, rules):
    """tests/test_bootstrap_extent.py's box fixture with the study students on the remote and the box plans in the
    helper's kitsune.prereg (a sitecustomize on its PYTHONPATH, until kitsune.prereg carries them itself)."""
    import huggingface_hub.utils._paths as hf_paths

    stub = tmp_path / "stub" / "huggingface_hub"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(tb.STUB, encoding="utf-8")
    (stub / "utils.py").write_text(
        "import importlib.util\n"
        f"_s = importlib.util.spec_from_file_location('_hf_paths', {hf_paths.__file__!r})\n"
        "_m = importlib.util.module_from_spec(_s)\n_s.loader.exec_module(_m)\n"
        "filter_repo_objects = _m.filter_repo_objects\n", encoding="utf-8")
    (tmp_path / "stub" / "sitecustomize.py").write_text(
        "import copy\nimport kitsune.prereg as _p\n_r = _p.rules\n"
        f"_B = {json.dumps(BOXES_CONTRACT)!s}\n".replace("null", "None")
        + "def rules(sidecar=None):\n    r = _r(sidecar)\n    r.setdefault('boxes', copy.deepcopy(_B))\n    return r\n"
        "_p.rules = rules\n", encoding="utf-8")
    remote, kdir, state = tmp_path / "remote", tmp_path / "box", tmp_path / "state"
    files = list(tb.label_files())
    for b in ("A", "B"):
        ctc = set(Q.box_ctc_students(b, rules))
        for s in Q.box_students(b, rules):
            files += [f"{s}/{n}" for n in launch.STUDENT_FILES + ((launch.CTC_CARD,) if s in ctc else ())]
        files += [f"{d}/config.json" for d in Q.box_extra_dirs(b, rules)]
    for f in files:
        (remote / f).parent.mkdir(parents=True, exist_ok=True)
        (remote / f).write_bytes(b"x")
    (remote / tb.LR / "extent.json").write_text(json.dumps(tb.make_record()), encoding="utf-8")
    (kdir / "configs").mkdir(parents=True)
    state.mkdir()
    (tmp_path / "helper.py").write_text(tb.helper_source(), encoding="utf-8")
    env = dict(os.environ, KITSUNE_DIR=str(kdir), STATE=str(state), CONFIG="configs/c.json",
               KITSUNE_DATA_REPO="u/data", KITSUNE_OUT_REPO="u/runs", KITSUNE_DATA_REVISION="0123abc",
               FAKE_REMOTE=str(remote), FAKE_LOG=str(tmp_path / "hub.log"), CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(tmp_path / "stub"), str(ROOT)]))

    class Box:
        root, remote_dir, state_dir = kdir, remote, state

        @staticmethod
        def config(cfg: dict):
            (kdir / "configs" / "c.json").write_text(json.dumps(cfg), encoding="utf-8")

        @staticmethod
        def helper(cmd: str, **extra):
            return subprocess.run([sys.executable, str(tmp_path / "helper.py"), cmd], capture_output=True, text=True,
                                  env=dict(env, **extra), timeout=120)

    return Box


# =================================================================================================== supervise


def run_queue_supervisor(tmp_path, monkeypatch, rcs, summary_reason=None, state=None):
    """supervise_queue with the queue's exits scripted and finish.py recorded."""
    calls, finishes = [], []

    def queue(argv, env):
        calls.append(list(argv))
        return rcs.pop(0)

    monkeypatch.setattr(supervise, "run_trainer", queue)
    monkeypatch.setattr(supervise, "call_finish", lambda args, timeout=None: finishes.append(list(args)) or 0)
    state_path = tmp_path / "state" / "supervise.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state is not None:
        state_path.write_text(json.dumps(state), encoding="utf-8")
    if summary_reason:
        (tmp_path / "state" / "queue_summary.json").write_text(json.dumps({"reason": summary_reason}))
    rc = supervise.supervise_queue(["python", "-m", "kitsune.study_queue", "run", "--box", "A"], state_path)
    return rc, calls, finishes, json.loads(state_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("rc, n_fail, action", [(0, 0, "destroy"), (4, 1, "stop"), (3, 1, "stop"), (1, 1, "restart"),
                                               (None, 2, "restart"), (1, 3, "stop")])
def test_decide_queue(rc, n_fail, action):
    assert supervise.decide_queue(rc, n_fail)[0] == action


def test_supervise_queue_destroys_a_finished_box(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(tmp_path, monkeypatch, [0])
    assert rc == 0 and len(calls) == 1 and [f[0] for f in finishes] == ["--destroy"]


def test_supervise_queue_restarts_a_crashed_queue_then_stops(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(tmp_path, monkeypatch, [1, 1, 1])
    assert rc == 1 and len(calls) == 1 + supervise.MAX_QUEUE_RESTARTS
    assert [f[0] for f in finishes] == ["--sync-only"] * 3 + ["--stop"]
    rc, calls, finishes, state = run_queue_supervisor(tmp_path / "b", monkeypatch, [1, 0])
    assert rc == 0 and [f[0] for f in finishes] == ["--sync-only", "--destroy"]


def test_supervise_queue_stops_on_a_halt_with_its_reason(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(tmp_path, monkeypatch, [4], summary_reason="LR edge kept-t03")
    assert rc == 4 and finishes[-1][0] == "--stop" and "LR edge kept-t03" in state["final"]["reason"]


def test_supervise_queue_counts_an_interrupted_queue_and_never_reruns_a_final_box(tmp_path, monkeypatch):
    rc, calls, finishes, state = run_queue_supervisor(
        tmp_path, monkeypatch, [0], state={"attempts": [{"t0": 1.0, "queue": True}], "final": None})
    assert rc == 0 and state["attempts"][0]["interrupted"] is True and len(calls) == 1
    (tmp_path / "state" / "halt").write_text("{}")
    rc, calls, finishes, _ = run_queue_supervisor(tmp_path, monkeypatch, [], state=state)
    assert calls == [] and finishes == []


def test_supervise_main_runs_the_queue_for_a_study_box(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setenv("KITSUNE_JOB", "study")
    monkeypatch.setenv("KITSUNE_BOX", "B")
    monkeypatch.setattr(supervise, "supervise_queue", lambda cmd, state, dry: seen.update(cmd=cmd) or 0)
    assert supervise.main(["--state", str(tmp_path / "s.json")]) == 0
    assert seen["cmd"][1:] == ["-m", "kitsune.study_queue", "run", "--box", "B"]


# ====================================================================================================== finish


def lean_run(root: Path, M=1000, small=True, branch=False) -> Path:
    run = root / "runs" / "study-t01-20260926T000000Z"
    cfg = {"schedule": {"max_steps": M}, "ckpt": {"upload_full_at": ["frac:0.8"] if small else []},
           "branch": {"parent": "runs/x" if branch else None}}
    files = {"config.json": json.dumps({"config": cfg}).encode(), "summary.json": b"{}", "metrics/scalars.jsonl": b"1",
             "checkpoints/step_400/model.safetensors": b"w4", f"checkpoints/step_{M}/model.safetensors": b"wM",
             "checkpoints/full_step_400/model.pt": b"f4", "checkpoints/full_step_800/model.pt": b"f8",
             f"checkpoints/full_step_{M}/model.pt": b"fM", "checkpoints/full_step_700/model.pt": b"f7"}
    for rel, data in files.items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_bytes(data)
    return run


def test_lean_expected_files_are_the_weights_and_the_uploaded_full_states(tmp_path):
    run = lean_run(tmp_path)
    got = {p.split("/", 2)[2] for p in finish.expected_files(run, False, lean=True)}
    assert got == {"config.json", "summary.json", "metrics/scalars.jsonl", "checkpoints/step_400/model.safetensors",
                   "checkpoints/step_1000/model.safetensors", "checkpoints/full_step_800/model.pt"}
    big = lean_run(tmp_path / "big", small=False)
    assert not any("full_step" in p for p in finish.expected_files(big, False, lean=True))
    branch = lean_run(tmp_path / "br", branch=True)  # a branch ignores its fractions
    assert not any("full_step" in p for p in finish.expected_files(branch, False, lean=True))
    # the train job's rule is unchanged: the newest full state
    assert any("full_step_1000" in p for p in finish.expected_files(run))


def test_finish_study_job_is_lean_and_puts_its_infra_under_the_box(tmp_path, monkeypatch):
    run = lean_run(tmp_path)
    uploads, commits = [], []

    class Hub:
        def upload_folder(self, **kw):
            uploads.append(kw["allow_patterns"])

        def create_commit(self, **kw):
            commits.append([op.path_in_repo for op in kw["operations"]])

        def list_repo_tree(self, repo, path_in_repo=None, recursive=False, repo_type=None):
            exp = finish.expected_files(run, False, lean=True)
            return [SimpleNamespace(path=p, size=f.stat().st_size, lfs=None, blob_id=finish.git_blob_id(f))
                    for p, f in exp.items()]

    monkeypatch.setattr(finish, "hf_api", lambda: Hub())
    monkeypatch.setattr(finish, "STATE_DIR", tmp_path / "state")
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    monkeypatch.setenv("KITSUNE_JOB", "study")
    monkeypatch.setenv("KITSUNE_BOX", "A")
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "queue_summary.json").write_text("{}")
    # the queue's per-item logs (the store build, the anchor, the speed probes: no run dir of their own) and the state
    # a re-arm moved aside go up too; a big log as its end
    (tmp_path / "state" / "logs").mkdir()
    (tmp_path / "state" / "logs" / "speed-cohere.log").write_bytes(b"Traceback: CUDA error\n")
    (tmp_path / "state" / "logs" / "stores-aed.log").write_bytes(b"x" * 100 + b"stores: train 12 utts\n")
    (tmp_path / "state" / "rearm-20261001T000000Z").mkdir()
    (tmp_path / "state" / "rearm-20261001T000000Z" / "events.jsonl").write_bytes(b'{"kind": "prereg_numbers"}\n')
    monkeypatch.setattr(finish, "INFRA_MAX_FILE_BYTES", 30)
    sent = {}

    def create_commit(**kw):
        commits.append([op.path_in_repo for op in kw["operations"]])
        sent.update({op.path_in_repo: op.path_or_fileobj for op in kw["operations"]})

    Hub.create_commit = staticmethod(create_commit)
    rc = finish.main(["--destroy", "--repo", "r", "--runs-root", str(tmp_path / "runs")])
    assert rc == 0 and actions == ["destroy"]
    ck = [p for pats in uploads for p in pats if p.startswith("checkpoints/")]
    assert "checkpoints/full_step_800/model.pt" in ck and not any("full_step_1000" in p or "full_step_700" in p
                                                                   for p in ck)
    assert any(p.startswith("study/box-A/infra/") and p.endswith("queue_summary.json") for c in commits for p in c)
    infra = {p.split("/infra/", 1)[1].split("/", 1)[1]: v for p, v in sent.items() if "/infra/" in p}
    assert infra["logs/speed-cohere.log"] == b"Traceback: CUDA error\n"
    assert infra["logs/stores-aed.log"] == (b"x" * 100 + b"stores: train 12 utts\n")[-30:]
    assert infra["rearm-20261001T000000Z/events.jsonl"].startswith(b'{"kind": "prereg_numbers"}')
    # the train job's infra stays the top level only
    assert [n for n, _ in finish.infra_files()] == [n for n, _ in finish.infra_files(deep=True) if "/" not in n]


def test_hub_retry_backs_off_exponentially_and_only_on_what_a_retry_fixes(monkeypatch):
    waits = []
    monkeypatch.setattr(finish.time, "sleep", waits.append)
    assert finish.HUB_RETRY_WAITS[0] == 5 and max(finish.HUB_RETRY_WAITS) == 600
    assert all(b >= a for a, b in zip(finish.HUB_RETRY_WAITS, finish.HUB_RETRY_WAITS[1:]))

    def err(status):
        e = RuntimeError(f"HTTP {status}")
        e.response = SimpleNamespace(status_code=status)
        return e

    tries = []

    def flaky():
        tries.append(1)
        if len(tries) < 4:
            raise err(429)
        return "ok"

    assert finish.hub_retry(flaky, "x") == "ok" and waits == list(finish.HUB_RETRY_WAITS[:3])
    waits.clear()
    with pytest.raises(RuntimeError, match="401"):
        finish.hub_retry(lambda: (_ for _ in ()).throw(err(401)), "x")
    assert waits == []  # a refused token is not retried
    assert finish.retryable(err(503)) and finish.retryable(RuntimeError("connection reset"))
    assert not finish.retryable(err(404))
