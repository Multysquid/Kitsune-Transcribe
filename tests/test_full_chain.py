"""The chained box (contract addendum E: kitsune/fullrun.py chain boxes, kitsune/full_queue.py ChainController), CPU
only, on fakes.

- registry: the E.1.7 entry (built here on tests/fixtures_full.tiny_registry, never configs/full/boxes.json) validates
  and round-trips (box_spec / box_students on a loaded registry, a loaded registry loaded again); every E.1.4 rule is
  refused; the stage views of E.2.1 (stage 1 adds smoke-b's study_1000h.parquet / .json); box_env per stage; the
  derived spec of E.1.5; box_items and FullQueue refuse a chain; the CLI's check-students serves one stage
- controller, end to end on real part queues (tests/fake_study_trainer.py, DirHub runs and scratch repos, a stand-in
  alert watchdog, a fake stage-2 bootstrap script): full-smoke with its faults passes the gate, the mode file switches
  to stop 3600 first, smoke-b runs with its re-time probe not_needed, the stage-2 bootstrap gets its env, p01 trains,
  rc 0, every summary and verdict on the Hub, and the controller never beats during F5's freeze
- controller on scripted part queues (FakeParts): a failed gate (smoke-b still runs, no bootstrap, no p01), the gate's
  strictness (chain_gate), a lost verdict rebuilt or missing, a part's stop_at (real queues, FAKE_HANG), the stage-2
  bootstrap's attempts, refusals and bound (its children killed, no controller beat meanwhile), the fit rule at the
  handover and after the bootstrap, restarts at every step (no second gate, ended parts not run again, a live verified
  bootstrap killed, a recycled pid spared), p01 rc 4, smoke-b failing twice, the per-part item deadlines, the
  first_boot fallback, stage2_timeouts, the chain's plan and its refused resume-pull
"""
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "vast"))

import pytest  # noqa: E402

from fake_runs_repo import DirHub, FakeApi, downloads  # noqa: E402
from fixtures_chain import CHAIN, CHAIN_BOX, with_chain  # noqa: E402
from fixtures_full import tiny_registry  # noqa: E402
from kitsune import full_queue as F  # noqa: E402
from kitsune import fullrun, prereg  # noqa: E402
from kitsune import study_queue as Q  # noqa: E402
from test_full_queue import FAKE, PY, RESUME_ENV, FakeUploader, fake_evals, train_configs, weights_remote  # noqa: E402

H = 3600.0


def stage(reg: dict, k: int) -> dict:
    return reg["boxes"][CHAIN_BOX]["chain"][k]


def rewrite(root: Path, name: str, **changes):
    """configs/full/<name>.json with top-level keys changed (None deletes one)."""
    p = root / "configs" / "full" / f"{name}.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    for k, v in changes.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def recap(root: Path, names: tuple[str, ...], **inputs):
    """The extent caps of the named configs set to `inputs` (a data config and its item configs together, so their
    DATA_KEYS stay equal)."""
    for n in names:
        cfg = json.loads((root / "configs" / "full" / f"{n}.json").read_text(encoding="utf-8"))
        rewrite(root, n, extent=dict(cfg["extent"], inputs=dict(inputs)))


# ================================================================================================ the registry


def test_the_chain_entry_validates_round_trips_and_derives_its_spec(tmp_path):
    reg = with_chain(tiny_registry(tmp_path))
    assert fullrun.registry_problems(reg, root=tmp_path) == []
    loaded = fullrun.load_registry(reg, root=tmp_path)
    # the normalised raw form: the chain's own keys and the stages', rebuild and gate_by_hours filled; loading the
    # loaded registry again changes nothing (a stored derived spec would be refused: F2)
    assert loaded["boxes"][CHAIN_BOX] == {k: v for k, v in CHAIN.items()}
    assert fullrun.load_registry(fullrun.load_registry(reg, root=tmp_path)) == loaded
    assert fullrun.load_registry(dict(reg, boxes=dict(reg["boxes"], **{CHAIN_BOX: dict(
        CHAIN, chain=[{k: v for k, v in CHAIN["chain"][0].items() if k not in ("rebuild", "gate_by_hours")},
                      {"parts": ["p01"]}])})), root=tmp_path)["boxes"][CHAIN_BOX] == loaded["boxes"][CHAIN_BOX], \
        "rebuild defaults to parts[0]'s data config, gate_by_hours to the gate box's max_hours"
    spec = fullrun.box_spec(CHAIN_BOX, loaded)  # the derived spec (E.1.5), never stored
    assert {k: spec[k] for k in ("gpus", "data_config", "est_hours", "max_hours", "max_dph", "extra_gb", "gate",
                                  "watchdog", "deadline_reserve_min", "timed_states", "smoke", "faults", "items",
                                  "max_attempts")} == dict(
        gpus=1, data_config="configs/full/data-p01.json", est_hours=25.2, max_hours=35, max_dph=1.0, extra_gb=120,
        gate=True, watchdog={"orphan_s": 600, "action": "alert"}, deadline_reserve_min=45, timed_states=True,
        smoke=False, faults=[], items=[], max_attempts=4)
    assert [s["stage"] for s in spec["chain"]] == [1, 2] and spec["chain"][0]["data_configs"] == {
        "full-smoke": "configs/full/data-smoke.json", "smoke-b": "configs/full/data-smoke-b.json"}
    assert spec["chain"][1]["watchdog"] == {"orphan_s": 3600, "action": "stop"}
    assert fullrun.box_students(CHAIN_BOX, loaded, stage=1) == ["students/study/t06", "students/study/p03",
                                                                "students/study/p01", "students/study/p005"]
    assert fullrun.box_students(CHAIN_BOX, loaded, stage=2) == ["students/study/p01"]
    assert fullrun.box_ctc_students(CHAIN_BOX, loaded, stage=2) == ["students/study/p01"]
    assert fullrun.is_chain(CHAIN_BOX, loaded) and not fullrun.is_chain("p01", loaded)
    # the plain boxes read as before
    assert fullrun.box_spec("p01", loaded) == fullrun.box_spec("p01", fullrun.load_registry(tiny_registry(tmp_path)))
    assert fullrun.box_configs(CHAIN_BOX, loaded) == [
        "configs/full/data-smoke.json", "configs/full/smoke-p03.json", "configs/full/smoke-t06.json",
        "configs/full/smoke-p01.json", "configs/full/smoke-p005.json", fullrun.BOXES_FILE,
        "configs/full/data-smoke-b.json", "configs/full/smoke-b-t06.json", "configs/full/data-p01.json",
        "configs/full/full-p01.json"]


def test_the_stage_views_and_the_stage_env(tmp_path):
    loaded = fullrun.load_registry(with_chain(tiny_registry(tmp_path)), root=tmp_path)
    v1 = fullrun.stage_view(CHAIN_BOX, 1, loaded)
    assert (v1["parts"], v1["gate_box"], v1["rebuild"], v1["timed_states"]) == (
        ["full-smoke", "smoke-b"], "full-smoke", "configs/full/data-smoke.json", True)
    # E.2.1: smoke-b's data config is not stage 1's rebuild, so its frozen study selection comes along
    assert v1["extra_files"] == [fullrun.FROZEN_MANIFEST, fullrun.FULL_DIR + "/smoke.json",
                                 "labels/full/selections/study_1000h.parquet",
                                 "labels/full/selections/study_1000h.json"]
    assert v1["extra_dirs"] == ["models/parakeet-tdt_ctc-0.6b-ja-hf"]
    assert v1["watchdog"] == {"orphan_s": 600, "action": "alert"}
    v2 = fullrun.stage_view(CHAIN_BOX, 2, loaded)
    assert (v2["parts"], v2["gate_box"], v2["rebuild"], v2["students"], v2["extra_dirs"]) == (
        ["p01"], None, "configs/full/data-p01.json", ["students/study/p01"], [])
    assert v2["extra_files"] == [fullrun.FROZEN_MANIFEST, fullrun.FULL_DIR + "/full.json"]
    assert fullrun.box_extra_files(CHAIN_BOX, loaded, stage=2) == v2["extra_files"]
    assert fullrun.box_extra_files(CHAIN_BOX, loaded) == [*v1["extra_files"], fullrun.FULL_DIR + "/full.json"]
    assert fullrun.box_extra_dirs(CHAIN_BOX, loaded, stage=2) == [] and fullrun.box_extra_dirs(CHAIN_BOX, loaded) == [
        "models/parakeet-tdt_ctc-0.6b-ja-hf"]
    assert fullrun.box_env(CHAIN_BOX, loaded) == fullrun.box_env(CHAIN_BOX, loaded, stage=1) == {
        "KITSUNE_N_GPUS": "1", "KITSUNE_WATCHDOG_HB_FILE": "train_hb", "KITSUNE_WATCHDOG_ORPHAN_S": "600",
        "KITSUNE_WATCHDOG_ORPHAN_ACTION": "alert", "KITSUNE_CHAIN_STAGE": "1", "KITSUNE_WATCHDOG_HANDOVER_S": "34200"}
    assert fullrun.box_env(CHAIN_BOX, loaded, stage=2) == {
        "KITSUNE_N_GPUS": "1", "KITSUNE_WATCHDOG_HB_FILE": "train_hb", "KITSUNE_WATCHDOG_ORPHAN_S": "3600",
        "KITSUNE_WATCHDOG_ORPHAN_ACTION": "stop", "KITSUNE_CHAIN_STAGE": "2"}
    assert fullrun.box_env("p01", loaded, stage=2) == fullrun.box_env("p01", loaded)  # a plain box ignores stage
    with pytest.raises(fullrun.RegistryError, match="stages 1..2"):
        fullrun.stage_view(CHAIN_BOX, 3, loaded)
    assert fullrun.part_state_dir("smoke-b", tmp_path / "s") == tmp_path / "s" / "chain" / "smoke-b"
    # a chain runs through the chain controller: no items of its own, no FullQueue
    for fn in (fullrun.box_items, fullrun.train_items):
        with pytest.raises(fullrun.RegistryError, match="chain controller"):
            fn(CHAIN_BOX, loaded)
    with pytest.raises(Q.QueueError, match="chain controller"):
        F.FullQueue(CHAIN_BOX, F.FullSettings(root=tmp_path, state_dir=tmp_path / "state", gpus=["0"], out_repo=None,
                                              n_gpus=None), registry=loaded)
    with pytest.raises(Q.QueueError, match="not a chain box"):
        F.ChainController("p01", F.FullSettings(root=tmp_path, state_dir=tmp_path / "state", gpus=["0"]),
                          registry=loaded)


def _drop_sigstop(r, root):
    r["boxes"]["full-smoke"]["faults"] = [f for f in r["boxes"]["full-smoke"]["faults"] if f["id"] != "F1"]


RULES = {
    "a part twice": (lambda r, root: stage(r, 1)["parts"].append("smoke-b"), r"'smoke-b' is a part twice"),
    "a chain as a part": (lambda r, root: stage(r, 1)["parts"].append(CHAIN_BOX),
                          r"part 'p01-chain' is a chain box: a chain's parts are plain"),
    "a part that is no box": (lambda r, root: stage(r, 1)["parts"].append("full-smoke-2"),
                              r"part 'full-smoke-2' is not a registry box"),
    "gate_box not parts[0]": (lambda r, root: stage(r, 0).update(parts=["smoke-b", "full-smoke"]),
                              r"gate_box 'full-smoke' must be the stage's first part \('smoke-b'\)"),
    "a gate box that is not smoke": (lambda r, root: r["boxes"]["full-smoke"].update(smoke=False, faults=[]),
                                     r"gate_box 'full-smoke' must be a smoke box with >= 1 train item"),
    "a gate box without train items": (lambda r, root: stage(r, 0).update(parts=["smoke-b", "full-smoke"],
                                                                          gate_box="smoke-b"),
                                       r"gate_box 'smoke-b' must be a smoke box with >= 1 train item"),
    "faults outside the gate part": (lambda r, root: stage(r, 0).update(parts=["smoke-b", "full-smoke"],
                                                                        gate_box="smoke-b"),
                                     r"part 'full-smoke' has faults: only the gate part \('smoke-b'\) may"),
    "an alert watchdog outside the gate part": (
        lambda r, root: r["boxes"]["smoke-b"]["watchdog"].update(action="alert"),
        r"part 'smoke-b''s watchdog action is 'alert'"),
    "a part not within its stage's rebuild": (
        lambda r, root: (recap(root, ("data-smoke-b", "smoke-b-t06"), reazon_large=53, emilia_yodas="300h",
                               emilia_nc=8, galgame=2),
                         stage(r, 0).update(rebuild="configs/full/data-smoke-b.json")),
        r"part full-smoke's extent \(configs/full/data-smoke.json\) is not within stage 1's rebuild"),
    "stage 1 not within stage 2": (
        lambda r, root: recap(root, ("data-p01", "full-p01"), galgame=1),
        r"stage 1's rebuild is not within the last stage's"),
    "parakeet not pulled by the rebuild": (
        lambda r, root: (rewrite(root, "data-smoke-b", pull_parakeet=None), rewrite(root, "smoke-b-t06",
                                                                                     pull_parakeet=None),
                         stage(r, 0).update(rebuild="configs/full/data-smoke-b.json")),
        r"part full-smoke needs parakeet_out .* does not pull it"),
    "other data roots": (
        lambda r, root: (rewrite(root, "data-smoke-b", data_root="data2"), rewrite(root, "smoke-b-t06",
                                                                                    data_root="data2")),
        r"part smoke-b's configs/full/data-smoke-b.json differs from stage 1's rebuild .* in \['data_root'\]"),
    "the hours of rule 3": (lambda r, root: r["boxes"][CHAIN_BOX].update(max_hours=29),
                            r"leaves 18.5 h, below the last stage's est_hours 19.5"),
    "items on a chain": (lambda r, root: r["boxes"][CHAIN_BOX].update(items=[]), r"\['items'\] are derived"),
    "data_config on a chain": (lambda r, root: r["boxes"][CHAIN_BOX].update(data_config="configs/full/data-p01.json"),
                               r"\['data_config'\] are derived"),
    "watchdog on a chain": (lambda r, root: r["boxes"][CHAIN_BOX].update(watchdog={"orphan_s": 1, "action": "stop"}),
                            r"\['watchdog'\] are derived"),
    "unequal gpus": (lambda r, root: r["boxes"]["smoke-b"].update(gpus=2), r"its parts have different gpus"),
    "three stages": (lambda r, root: r["boxes"][CHAIN_BOX]["chain"].append({"parts": ["full"]}),
                     r"chain is not a list of exactly 2 stage objects"),
    "gate false": (lambda r, root: r["boxes"][CHAIN_BOX].update(gate=False), r"a chain's gate must be true"),
    "of_box naming a chain": (
        lambda r, root: next(it for it in r["boxes"]["full"]["items"] if it.get("of_box")).update(of_box=CHAIN_BOX),
        r"of_box 'p01-chain' is not a registry box with items \(it is a chain\)"),
    "only_if_new_machine naming a chain": (
        lambda r, root: next(it for it in r["boxes"]["smoke-b"]["items"] if it["kind"] == "speed").update(
            only_if_new_machine=CHAIN_BOX), r"only_if_new_machine 'p01-chain' is not another registry box"),
    "a chain name without chain": (lambda r, root: r["boxes"].update({CHAIN_BOX: copy.deepcopy(r["boxes"]["p01"])}),
                                   r"is a chain box \(CHAIN_NAMES\): its entry needs `chain`"),
    "a plain name with chain": (lambda r, root: r["boxes"].update(p01=dict(copy.deepcopy(CHAIN))),
                                r"boxes.p01: a plain box \(BOX_NAMES\) has no `chain`"),
    "a rebuild that is no part's": (lambda r, root: stage(r, 1).update(rebuild="configs/full/data-full.json"),
                                    r"rebuild 'configs/full/data-full.json' is not the data config of one"),
    "gate_by past stage 1": (lambda r, root: stage(r, 0).update(gate_by_hours=11), r"gate_by_hours 11"),
    "stage 1 as long as the chain": (lambda r, root: stage(r, 0).update(max_hours=35),
                                     r"max_hours 35 must be < the chain's max_hours 35"),
    "extra_gb below the last stage's": (lambda r, root: r["boxes"][CHAIN_BOX].update(extra_gb=10),
                                        r"extra_gb 10 is below its last stage's parts' \(25\)"),
    "gate_box on the last stage": (lambda r, root: stage(r, 1).update(gate_box="p01"),
                                   r"\['gate_box'\] belong to the gated first stage only"),
    "an unknown stage key": (lambda r, root: stage(r, 0).update(gpus=1), r"chain\[0\]: unknown key\(s\) \['gpus'\]"),
    "no parts": (lambda r, root: stage(r, 1).update(parts=[]), r"parts \[\] is not a non-empty list"),
    "a gate freeze not under the stop orphan_s": (
        lambda r, root: r["boxes"]["p01"].update(watchdog={"orphan_s": 900, "action": "stop"}),
        r"the gate part's freeze 'F5' \(900 s\) must be shorter than the last stage's stop orphan_s 900"),
}


@pytest.mark.parametrize("name", sorted(RULES))
def test_each_chain_rule_is_refused(tmp_path, name):
    reg = with_chain(tiny_registry(tmp_path))
    change, want = RULES[name]
    change(reg, tmp_path)
    problems = fullrun.registry_problems(reg, root=tmp_path)
    assert any(re.search(want, p) for p in problems), (name, problems)
    with pytest.raises(fullrun.RegistryError):
        fullrun.load_registry(reg, root=tmp_path)


def test_the_file_rules_run_only_with_check_files(tmp_path):
    reg = with_chain(tiny_registry(tmp_path))
    recap(tmp_path, ("data-p01", "full-p01"), galgame=1)
    assert fullrun.registry_problems(reg, root=tmp_path, check_files=False) == []
    assert fullrun.registry_problems(reg, root=tmp_path)  # E.1.4 rule 4 reads the configs (kitsune.extent.within)


def _meta(run: str) -> dict:
    spec = prereg.RUNS[run]
    return {"stage": "complete", **{k: spec[k] for k in ("family", "init_class", "seed", "params_total",
                                                         "params_non_embedding")},
            "closed_form_params": spec["params_total"], "calibration": {"ids_sha256": spec.get("calib_ids_sha256")}}


def test_the_cli_serves_one_stage(tmp_path, monkeypatch, capsys):
    reg = with_chain(tiny_registry(tmp_path))
    (tmp_path / fullrun.BOXES_FILE).write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.delenv(fullrun.ENV_REGISTRY, raising=False)
    monkeypatch.delenv(fullrun.ENV_CHAIN_STAGE, raising=False)

    def put(run):
        s = prereg.RUNS[run]["student"]
        (tmp_path / s).mkdir(parents=True, exist_ok=True)
        (tmp_path / s / "student_meta.json").write_text(json.dumps(_meta(run)), encoding="utf-8")

    def cli(*argv):
        rc = fullrun.main([*argv, "--root", str(tmp_path)])
        out = capsys.readouterr()
        return rc, out.out, out.err

    put("study-p01")  # box 1's student only: stage 2 passes, stage 1 (smoke A's four students) does not
    assert cli("check-students", "--box", CHAIN_BOX, "--stage", "2")[0] == 0
    rc, _, err = cli("check-students", "--box", CHAIN_BOX, "--stage", "1")
    assert rc == 2 and "students/study/t06/student_meta.json: cannot read it" in err
    for run in ("study-t06", "study-p03", "study-p005"):  # stage 1's students on disk
        put(run)
    rc, out, _ = cli("check-students", "--box", CHAIN_BOX, "--stage", "1")
    assert rc == 0 and "box p01-chain (stage 1): 4 student(s) are the registered builds" in out
    monkeypatch.setenv(fullrun.ENV_CHAIN_STAGE, "2")  # bootstrap's: the stage comes from the env
    rc, out, _ = cli("students", "--box", CHAIN_BOX)
    assert rc == 0 and out.split() == ["students/study/p01"]
    rc, out, _ = cli("extra-files", "--box", CHAIN_BOX, "--stage", "1")
    assert "labels/full/selections/study_1000h.parquet" in out.split()
    rc, out, _ = cli("show", "--box", CHAIN_BOX)
    doc = json.loads(out)
    assert rc == 0 and set(doc["stages"]) == {"1", "2"} and doc["spec"]["data_config"] == "configs/full/data-p01.json"
    assert doc["env"]["KITSUNE_CHAIN_STAGE"] == "2"
    assert cli("students", "--box", CHAIN_BOX, "--stage", "3")[0] == 2
    assert cli("students", "--box", "p01", "--stage", "2")[1].split() == ["students/study/p01"]  # ignored: plain box


# ====================================================================================================== harness

FAKE_BOOT = r'''"""a stand-in for vast/bootstrap.sh stage 2: records its env and train_hb's mtime, plays FAKE_BOOT's
rc per attempt ("hang": sleeps, with a child that sleeps too)"""
import json, os, subprocess, sys, time
from pathlib import Path
state = Path(os.environ["KITSUNE_STATE"])
log = Path(os.environ["FAKE_BOOT_LOG"])
n = (len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0) + 1
plan = json.loads(os.environ.get("FAKE_BOOT") or "[0]")
rc = plan[min(n - 1, len(plan) - 1)]
hb = state / "train_hb"
keys = ("KITSUNE_CHAIN_STAGE", "KITSUNE_CONFIG", "KITSUNE_PULL_TIMEOUT_MIN", "KITSUNE_REBUILD_TIMEOUT_MIN",
        "KITSUNE_PHASE_HB_MAX_S", "KITSUNE_GATE_BYTES", "KITSUNE_GATE_MAX_H", "KITSUNE_RESUME", "KITSUNE_RESUME_RESET",
        "KITSUNE_RESUME_SETS", "KITSUNE_STATE", "KITSUNE_DIR", "KITSUNE_BOX", "KITSUNE_JOB")
with open(log, "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(n=n, t0=time.time(), pid=os.getpid(), rc=rc, cwd=os.getcwd(),
                            env={k: os.environ.get(k) for k in keys},
                            hb=hb.stat().st_mtime if hb.exists() else None)) + "\n")
if rc == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (log.parent / f"boot-child-{n}").write_text(str(child.pid), encoding="utf-8")
    time.sleep(600)
time.sleep(float(os.environ.get("FAKE_BOOT_S") or 0.1))
sys.exit(int(rc))
'''


def verdict_doc(part: str, checks: dict | None = None, *, sha: str = "test-sha", box1_h=12.0) -> dict:
    """A smoke verdict whose built-in checks 1-11 pass unless `checks` says otherwise ({n: value}; "missing" drops
    the check)."""
    c = {str(n): {"pass": True, "evidence": {}} for n in range(1, 12)}
    c["3"]["evidence"] = {"box1_h": box1_h, "box2_h": 40.0}
    for n, v in (checks or {}).items():
        if v == "missing":
            c.pop(n, None)
        else:
            c[n] = {"pass": v, "evidence": {}}
    return dict(format=1, box=part, sha=sha, machine_id="m1", time_utc=F._now_utc(),
                overall="fail" if any(x["pass"] is False for x in c.values()) else "pass", checks=c)


class FakeParts:
    """ChainController.queue replaced: each part's queue plays a scripted rc (an exception is raised), writes its
    final, summary and (a smoke part) verdict into its state dir and puts them through the controller's uploader."""

    def __init__(self, rcs: dict | None = None, verdict=None, lose_verdict=False, rebuild_fails=False, on_run=None):
        self.rcs = {p: list(v) for p, v in (rcs or {}).items()}
        self.verdict = verdict or (lambda part: verdict_doc(part))
        self.lose_verdict, self.rebuild_fails = lose_verdict, rebuild_fails
        self.on_run = on_run  # on_run(queue): called as a part's run() starts (what the Hub holds meanwhile)
        self.calls, self.built = [], []

    def make(self, ctl, part, deadline, stop_at):
        return FakeQueue(self, ctl, part, ctl.part_settings(part, deadline, stop_at))

    def ran(self) -> list[str]:
        return [c["part"] for c in self.calls]


class FakeQueue:
    def __init__(self, parts: FakeParts, ctl, part: str, s):
        self.parts, self.ctl, self.part, self.s = parts, ctl, part, s
        self.dir = Path(s.state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        qf = self.dir / "queue.json"
        self.state = json.loads(qf.read_text(encoding="utf-8")) if qf.is_file() else dict(
            box=part, kind="full", started=time.time(), final=None)
        qf.write_text(json.dumps(self.state), encoding="utf-8")
        parts.built.append(part)

    def run(self) -> int:
        self.parts.calls.append(dict(part=self.part, deadline=self.s.deadline, stop_at=self.s.stop_at,
                                     box_state=self.s.box_state_dir, state=self.s.state_dir))
        if self.parts.on_run is not None:
            self.parts.on_run(self)
        seq = self.parts.rcs.setdefault(self.part, [0])
        rc = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(rc, BaseException):
            raise rc
        if rc in (0, 4):
            self.state["final"] = dict(rc=rc, status="complete" if rc == 0 else "halted", reason=None, wall=time.time())
            (self.dir / "queue.json").write_text(json.dumps(self.state), encoding="utf-8")
            self.write_summary(self.state["final"]["status"], None, rc)
            if self.ctl.registry["boxes"][self.part]["smoke"] and not self.parts.lose_verdict:
                self.write_verdict()
        return rc

    def _put(self, name: str, doc: dict, hub_path: str):
        (self.dir / name).write_text(json.dumps(doc), encoding="utf-8")
        if self.s.uploader is not None:
            self.s.uploader.put_file(self.dir / name, hub_path)

    def write_verdict(self):
        if self.parts.rebuild_fails:
            raise RuntimeError("the verdict cannot be built")
        self._put(fullrun.VERDICT_FILE, self.parts.verdict(self.part), fullrun.box_verdict_path(self.part))

    def write_summary(self, status, reason, rc):
        self._put(fullrun.SUMMARY_FILE, dict(format=1, kind="full", box=self.part, status=status, reason=reason, rc=rc,
                                             machine_id=self.s.machine_id, container_id=self.s.container_id,
                                             started=self.state["started"], items={}),
                  fullrun.box_summary_path(self.part))


@pytest.fixture
def ch(tmp_path, monkeypatch):
    """A chained box's checkout (tmp_path/box: tiny_registry's configs, fake-loadable, with the p01-chain entry), its
    state dir (first_boot now, a 35 h deadline, a passed download gate), the fake bootstrap."""
    for k in (*RESUME_ENV, fullrun.ENV_CHAIN_STAGE, fullrun.ENV_MAX_HOURS, fullrun.ENV_GATE_BYTES,
              fullrun.ENV_GATE_MAX_H, "KITSUNE_WATCHDOG_SYNC_LEAD_S"):
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "box"
    reg = with_chain(fake_evals(tiny_registry(root)))
    train_configs(root)
    (root / "runs").mkdir(parents=True)
    state, log, boot = tmp_path / "state", tmp_path / "fake-log", tmp_path / "fake_boot.py"
    boot.write_text(FAKE_BOOT, encoding="utf-8")
    boot_log = tmp_path / "boot.jsonl"

    def boot_state(first_boot: float | None = None, deadline: float | None = None, gate: dict | None = None):
        state.mkdir(parents=True, exist_ok=True)
        now = time.time()
        (state / "first_boot").write_text(f"{int(first_boot if first_boot is not None else now)}\n")
        (state / "deadline").write_text(f"{int(deadline if deadline is not None else now + 35 * H)}\n")
        (state / fullrun.GATE_FILE).write_text(json.dumps(gate or {"verdict": "pass", "rate_bytes_s": 1e8}))

    def make(registry=None, uploader=None, env=None, parts: FakeParts | None = None, boot_env=None, **settings):
        base = dict(root=root, state_dir=state, out_repo="u/kitsune-runs", gpus=["0"], python=PY,
                    train_cmd=[PY, FAKE, "train"], stores_cmd=[PY, FAKE, "stores"], eval_cmd=[PY, FAKE, "readout"],
                    speed_cmd=[PY, FAKE, "speed"], check_resume_cmd=[PY, FAKE, "check-resume"],
                    uploader=uploader if uploader is not None else FakeUploader(), n_gpus=None, poll_s=0.02,
                    kill_grace_s=0.5, summary_min_s=0.0, scratch_repo="u/kitsune-scratch", machine_id="m1",
                    sha="test-sha", container_id="c1", proc_root=tmp_path / "no-proc", cgroup=tmp_path / "no-cgroup",
                    env=dict(FAKE_LOG=str(log), FAKE_BOOT_LOG=str(boot_log), **(boot_env or {}), **(env or {})))
        base.update(settings)
        c = F.ChainController(CHAIN_BOX, F.FullSettings(**base), registry=registry if registry is not None else reg,
                              bootstrap_cmd=[PY, str(boot)], boot_poll_s=0.05, boot_kill_grace_s=0.5,
                              boot_reap_s=3.0)
        if parts is not None:
            c.queue = lambda part, deadline, stop_at: parts.make(c, part, deadline, stop_at)
        return c

    def events(kind=None, source="chain"):
        p = state / "events.jsonl"
        ev = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []
        return [e for e in ev if (kind is None or e["kind"] == kind) and (source is None or e.get("source") == source)]

    def boots() -> list[dict]:
        return [json.loads(x) for x in boot_log.read_text(encoding="utf-8").splitlines()] if boot_log.exists() else []

    def wipe():
        """A fresh box: no state, no bootstrap or fake-trainer records."""
        for p in (state, boot_log, log, *tmp_path.glob("boot-child-*")):
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()

    def chain_state() -> dict:
        return json.loads((state / "chain" / "chain.json").read_text(encoding="utf-8"))

    def summary() -> dict:
        return json.loads((state / fullrun.SUMMARY_FILE).read_text(encoding="utf-8"))

    return SimpleNamespace(root=root, reg=reg, state=state, tmp=tmp_path, log=log, make=make, boot_state=boot_state,
                           events=events, boots=boots, chain_state=chain_state, summary=summary, wipe=wipe)


def kinds(evs: list[dict]) -> list[str]:
    return [e["kind"] for e in evs]


def at(evs: list[dict], kind: str, **fields) -> int:
    """The index of the first event of this kind (and fields)."""
    return next(i for i, e in enumerate(evs) if e["kind"] == kind and all(e.get(k) == v for k, v in fields.items()))


# ================================================================================================ happy path


def test_the_chain_end_to_end_on_real_part_queues(ch, monkeypatch):
    """E.9.3 (1): full-smoke (its faults F2-F5, the gate's checks 1-11 on the fakes) -> the mode file stop 3600 -> the
    gate passes -> smoke-b (its bf16 re-time probe not_needed: the Hub's full-smoke summary has this machine) -> the
    stage-2 bootstrap with its env -> p01 trains -> rc 0. Every part's summary and every verdict is on the Hub, at the
    standalone boxes' paths, the chain's at full/box-p01-chain/; the controller never beat during F5's freeze."""
    import finish
    import huggingface_hub

    reg = copy.deepcopy(ch.reg)
    fs = reg["boxes"]["full-smoke"]
    fs["watchdog"] = {"orphan_s": 0, "action": "alert"}
    fs["faults"] = [f for f in fs["faults"] if f["action"] != "sigstop"]  # posix only; the queue's tests cover it
    for f in fs["faults"]:
        if f["action"] == "freeze_controller_hb":
            f["seconds"] = 60.0  # orphan_s 0 + fullrun.WATCHDOG_POLL_S; the stand-in's alert releases the hold at once
        if f["action"] == "wipe_run_dir":
            f["min_attempt"] = 1
    runs_hub = DirHub.create(ch.tmp / "runs-hub", limit=100000, window_s=1.0)
    scratch_hub = DirHub.create(ch.tmp / "scratch-hub", limit=100000, window_s=1.0)
    runs_hub.commit(weights_remote("study-p01", "study-p03", "study-t06"), writer="setup")
    dl, snap = downloads(runs_hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snap)
    monkeypatch.setattr(finish, "HUB_RETRY_WAITS", (0.05, 0.1))
    monkeypatch.setenv(fullrun.ENV_THREADS_PER_GPU, "8")
    for k in fullrun.ENV_THREAD_POOLS:  # onstart's six pools (fix 2): check 5 wants each in [1, t]
        monkeypatch.setenv(k, "8")
    up = Q.HubUploader("u/kitsune-runs")
    up._api = FakeApi(runs_hub, "chain")
    train_configs(ch.root, **{n: {"early_stop": {"min_delta_abs": 1e9}} for n in ("smoke-t06", "smoke-p03")})
    ch.boot_state()
    env = dict(FAKE_EPOCH_STEPS="500", FAKE_STEP_S="0.001", FAKE_FULL_EVERY="25",
               FAKE_EARLY=json.dumps({"smoke-t06": 0.7, "smoke-p03": 0.7}),
               FAKE_TIMED=json.dumps({"smoke-p01": 50}), FAKE_SCRATCH=str(scratch_hub.dir), FAKE_DEADLINE_S="1200",
               FAKE_STEP_VALUE=json.dumps({"smoke-t06": 1.5, "smoke-p03": 0.6, "smoke-p01": 0.4, "smoke-p005": 0.25}))
    ctl = ch.make(registry=reg, uploader=up, env=env, kill_grace_s=0.3,
                  runs_hub=F.Hub("u/kitsune-runs", api=FakeApi(runs_hub, "q")),
                  scratch_hub=F.Hub("u/kitsune-scratch", api=FakeApi(scratch_hub, "q")))
    hb, alerts, seen, stop = ch.state / fullrun.TRAIN_HB, ch.state / fullrun.ALERTS_FILE, [], threading.Event()

    def watchdog():  # the stage-1 alert watchdog in miniature (run_smoke's): an alert once train_hb is 0.3 s stale
        armed = True
        while not stop.is_set():
            try:
                m = hb.stat().st_mtime
            except OSError:
                m = None
            seen.append((time.time(), m))
            if m is not None and time.time() - m > 0.3 and armed:
                with open(alerts, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"wall": time.time(), "kind": "orphan_alert", "hb": "train_hb",
                                        "age_s": time.time() - m, "limit_s": 0.3}) + "\n")
                armed = False
            elif m is not None and time.time() - m < 0.1:
                armed = True
            time.sleep(0.01)

    th = threading.Thread(target=watchdog, daemon=True)
    th.start()
    try:
        rc = ctl.run()
    finally:
        stop.set()
        th.join(5)
    summ = ch.summary()
    assert rc == F.EXIT_OK, (summ["reason"], summ["gate"])
    assert (summ["kind"], summ["box"], summ["status"], summ["rc"], summ["stage"], summ["step"]) == (
        "chain", CHAIN_BOX, "complete", 0, 2, "done")
    gate = summ["gate"]
    assert gate["result"] == "pass" and gate["failed"] == [] and gate["problems"] == []
    assert gate["projected_box1_h"] == pytest.approx(0.4 * 110520 / 3600)
    # the order of the controller's steps
    evs = ch.events()
    order = [at(evs, "chain_part_end", part="full-smoke"), at(evs, "watchdog_mode", action="stop", orphan_s=3600),
             at(evs, "chain_gate", result="pass"), at(evs, "chain_part_start", part="smoke-b"),
             at(evs, "chain_part_end", part="smoke-b"), at(evs, "chain_handover"),
             at(evs, "chain_bootstrap_start", attempt=1), at(evs, "chain_bootstrap_end", rc=0, end="exit"),
             at(evs, "chain_part_start", part="p01"), at(evs, "chain_part_end", part="p01", rc=0),
             at(evs, "chain_end", rc=0)]
    assert order == sorted(order), kinds(evs)
    assert (ch.state / fullrun.WATCHDOG_MODE_FILE).read_bytes() == b"stop 3600\n"  # LF on any platform: bash reads it
    # every part in its own state dir, the box-wide files in the box's
    for part in ("full-smoke", "smoke-b", "p01"):
        pq = json.loads((fullrun.part_state_dir(part, ch.state) / "queue.json").read_text(encoding="utf-8"))
        assert pq["box"] == part and pq["final"]["rc"] in (0, 4), part
    sb = json.loads((fullrun.part_state_dir("smoke-b", ch.state) / "queue.json").read_text(encoding="utf-8"))
    assert sb["items"]["speed-study-p03"]["status"] == "not_needed"  # the Hub's full-smoke summary: machine m1
    p01 = json.loads((fullrun.part_state_dir("p01", ch.state) / "queue.json").read_text(encoding="utf-8"))
    assert all(p01["items"][n]["status"] == "done" for n in ("stores-ctc", "full-p01", "m4-full-p01"))
    # the Hub: four summaries and both verdicts, byte for byte the box's copies
    for part in ("full-smoke", "smoke-b", "p01"):
        local = fullrun.part_state_dir(part, ch.state) / fullrun.SUMMARY_FILE
        assert json.loads(runs_hub.path(fullrun.box_summary_path(part)).read_text()) == json.loads(local.read_text())
    for part in ("full-smoke", "smoke-b"):
        local = fullrun.part_state_dir(part, ch.state) / fullrun.VERDICT_FILE
        assert json.loads(runs_hub.path(fullrun.box_verdict_path(part)).read_text()) == json.loads(local.read_text())
    assert json.loads(runs_hub.path(fullrun.box_summary_path(CHAIN_BOX)).read_text()) == summ
    assert summ["verdicts"]["full-smoke"]["overall"] == "pass" and "smoke-b" in summ["verdicts"]
    assert summ["parts"]["p01"]["queue_started"] == p01["started"] and summ["parts"]["p01"]["verdict"] is None
    # the stage-2 bootstrap: its env (the stage, the last stage's rebuild, the controller's timeouts, no gate or
    # resume)
    (b,) = ch.boots()
    assert b["env"]["KITSUNE_CHAIN_STAGE"] == "2" and b["env"]["KITSUNE_CONFIG"] == "configs/full/data-p01.json"
    assert b["env"]["KITSUNE_PHASE_HB_MAX_S"] == "10800" and b["env"]["KITSUNE_BOX"] == CHAIN_BOX
    assert int(b["env"]["KITSUNE_PULL_TIMEOUT_MIN"]) >= 30 and int(b["env"]["KITSUNE_REBUILD_TIMEOUT_MIN"]) >= 30
    assert not any(b["env"][k] for k in ("KITSUNE_GATE_BYTES", "KITSUNE_GATE_MAX_H", "KITSUNE_RESUME"))
    assert Path(b["cwd"]).resolve() == ch.root.resolve()
    # stage 1's bootstrap records kept for the report
    assert (ch.state / "chain" / "stage1" / fullrun.GATE_FILE).is_file()
    # F5's freeze: train_hb never moved inside its window - neither the part's queue nor the controller beat it
    fz = json.loads((fullrun.part_state_dir("full-smoke", ch.state) / "queue.json").read_text())["faults"]["F5"]
    fsev = [json.loads(x) for x in (fullrun.part_state_dir("full-smoke", ch.state) / "events.jsonl").read_text(
        encoding="utf-8").splitlines()]
    until = min(fz["window_end"], next(e["wall"] for e in fsev if e["kind"] == "fault_outcome" and e["id"] == "F5"))
    window = [m for t, m in seen if fz["fired_at"]["wall"] + 0.05 < t < until - 0.05]
    assert window and len(set(window)) == 1
    # the part returned only once its freeze was released (hold_freeze), so set_mode's stop mode came after it
    assert fz["release"] in ("alert", "window") and fz["released"] <= until
    assert next(e["wall"] for e in ch.events("watchdog_mode")) >= fz["released"]
    # a restart of an ended chain does nothing
    assert ch.make(registry=reg, uploader=up).run() == 0 and len(ch.boots()) == 1


# =============================================================================================== the gate


def test_a_failed_gate_still_runs_smoke_b_and_ends_with_5(ch):
    """E.9.3 (2): a check-2 failure: the mode file is written anyway (before the gate), smoke-b still runs, no
    stage-2 bootstrap, no chain/p01/, no p01 summary on the Hub; the chain summary says gate_failed, rc 5."""
    ch.boot_state()
    up = FakeUploader()
    parts = FakeParts(verdict=lambda part: verdict_doc(part, {"2": False}))
    assert ch.make(parts=parts, uploader=up).run() == F.EXIT_CHAIN_DESTROY
    assert parts.ran() == ["full-smoke", "smoke-b"] and "p01" not in parts.built
    assert not (ch.state / "chain" / "p01").exists() and ch.boots() == []
    assert not [p for _, p, _ in up.put if p == fullrun.box_summary_path("p01")]
    assert (ch.state / fullrun.WATCHDOG_MODE_FILE).read_bytes() == b"stop 3600\n"
    s = ch.summary()
    assert (s["status"], s["rc"], s["stage"], s["gate"]["failed"], s["gate"]["result"]) == (
        "gate_failed", 5, 1, ["2"], "fail")
    assert s["reason"] == "chain gate failed: checks ['2'] problems []"
    evs = ch.events()
    assert at(evs, "watchdog_mode") < at(evs, "chain_gate") < at(evs, "chain_part_start", part="smoke-b")
    (g,) = ch.events("chain_gate")
    assert g["result"] == "fail" and g["failed"] == ["2"] and g["verdict_sha256"]
    st = ch.chain_state()
    assert st["final"]["rc"] == 5 and st["step"] == "done" and st["boot2"] is None
    assert json.loads(up.remote[fullrun.box_summary_path(CHAIN_BOX)]) == s


@pytest.mark.parametrize("case", ["rc4", "null", "missing", "other box", "sha", "no verdict"])
def test_the_gate_is_strict(case):
    """E.9.3 (3): rc 4 fails even with checks 1-11 true; a null or missing check fails; another box's verdict fails; a
    sha mismatch fails whenever KITSUNE_SHA is set; no verdict fails."""
    v, rc, sha = verdict_doc("full-smoke"), 0, "test-sha"
    if case == "rc4":
        rc = 4
    elif case == "null":
        v["checks"]["7"]["pass"] = None
    elif case == "missing":
        del v["checks"]["11"]
    elif case == "other box":
        v["box"] = "smoke-b"
    elif case == "sha":
        v["sha"] = "other"
    else:
        v = None
    g = F.chain_gate(v, "full-smoke", rc, sha)
    assert g["result"] == "fail", g
    want = {"rc4": ([], ["part full-smoke ended with rc 4"]), "null": (["7"], []), "missing": (["11"], []),
            "other box": ([], ["verdict of box 'smoke-b'"]), "sha": ([], ["verdict sha other"]),
            "no verdict": ([str(n) for n in range(1, 12)], ["verdict missing"])}[case]
    assert (g["failed"], g["problems"]) == want
    ok = F.chain_gate(verdict_doc("full-smoke", {"12": False}), "full-smoke", 0, "test-sha")
    assert ok["result"] == "pass" and set(ok["checks"]) == set(fullrun.GATE_CHECKS), "12-16 never enter the gate"
    assert F.chain_gate(dict(verdict_doc("full-smoke"), sha="x"), "full-smoke", 0, None)["result"] == "pass"


def test_a_gate_part_that_fails_exits_1_and_a_restart_goes_on(ch):
    """rc 1 of the gate part: the controller exits 1 with no gate record (the supervisor restarts it); the restart runs
    the part again (it resumes from its queue.json) and goes on."""
    ch.boot_state()
    parts = FakeParts(rcs={"full-smoke": [1, 0]})
    assert ch.make(parts=parts).run() == F.EXIT_FAIL
    st = ch.chain_state()
    assert st["gate"] is None and st["step"] == "s1_gate_part" and st["parts"]["full-smoke"]["rc"] == 1
    assert ch.summary()["rc"] == 1 and not (ch.state / fullrun.WATCHDOG_MODE_FILE).exists()
    assert ch.make(parts=parts).run() == F.EXIT_OK
    assert parts.ran() == ["full-smoke", "full-smoke", "smoke-b", "p01"]
    assert len(ch.events("chain_gate")) == 1


def test_a_lost_verdict_is_rebuilt_and_an_unbuildable_one_fails_the_gate(ch):
    """E.9.3 (4): the part saved its final but no verdict: it is built again from its queue.json, then the gate is
    evaluated; when it cannot be built, the gate fails with "verdict missing" (rc 5)."""
    ch.boot_state()
    parts = FakeParts(lose_verdict=True)
    assert ch.make(parts=parts).run() == F.EXIT_OK
    assert [e["part"] for e in ch.events("chain_verdict_rebuilt")] == ["full-smoke"]
    assert ch.chain_state()["gate"]["result"] == "pass"
    ch.wipe()
    ch.boot_state()
    parts = FakeParts(lose_verdict=True, rebuild_fails=True)
    assert ch.make(parts=parts).run() == F.EXIT_CHAIN_DESTROY
    g = ch.chain_state()["gate"]
    assert g["result"] == "fail" and g["problems"] == ["verdict missing"] and parts.ran() == ["full-smoke", "smoke-b"]
    assert ch.events("chain_verdict_rebuild_failed")


def test_an_old_verdict_is_rebuilt(ch):
    """A verdict older than the part's final (a restart between run()'s final and its verdict: a leftover of an
    earlier run) is not the gate's input: it is built again."""
    ch.boot_state()
    parts = FakeParts(rcs={"full-smoke": [1, 0]})
    ctl = ch.make(parts=parts)
    assert ctl.run() == F.EXIT_FAIL
    pdir = fullrun.part_state_dir("full-smoke", ch.state)
    old = dict(verdict_doc("full-smoke", {"5": False}), time_utc="2026-01-01T00:00:00+00:00")
    parts.lose_verdict = True
    (pdir / fullrun.VERDICT_FILE).write_text(json.dumps(old), encoding="utf-8")
    assert ch.make(parts=parts).run() == F.EXIT_OK
    assert ch.chain_state()["gate"]["result"] == "pass" and ch.events("chain_verdict_rebuilt")


# ================================================================================================= stop_at


def reduced(reg: dict, keep: dict[str, list[str]]) -> dict:
    """reg with the named boxes cut to the item names kept (their needs and faults cut with them)."""
    reg = copy.deepcopy(reg)
    for box, names in keep.items():
        b = reg["boxes"][box]
        b["items"] = [it for it in b["items"] if it["name"] in names]
        for it in b["items"]:
            it["needs"] = [n for n in it.get("needs", []) if n in names]
        b["faults"] = [f for f in b.get("faults", []) if f["item"] in names]
    return reg


def test_a_part_past_its_stop_at_halts_and_fails_the_gate(ch):
    """E.9.3 (5): a hung smoke trainer with gate_by 6 s away: the part halts at its stop_at (rc 4: the gate fails);
    smoke-b's running item is killed at the stage-1 sub-deadline (its rc 4 is report only); the chain ends with 5."""
    reg = reduced(ch.reg, {"full-smoke": ["stores-ctc", "smoke-p03"], "smoke-b": ["selftest"]})
    for it in reg["boxes"]["smoke-b"]["items"]:
        it.pop("max_hours", None)  # no no-start rule: it must start, and be killed at the part's stop time
        it.pop("verdict", None)
    now = time.time()
    ch.boot_state(first_boot=now - 9 * H + 6, deadline=now + 14)  # gate_by = now + 6, stage 1 = min(deadline, ...)
    up = FakeUploader(remote=weights_remote("study-p03", "study-t06"))
    ctl = ch.make(registry=reg, uploader=up, env=dict(FAKE_HANG=json.dumps({"smoke-p03": 5}),
                                                       FAKE_ITEM_S=json.dumps({"selftest": 120})))
    t0 = time.time()
    assert ctl.run() == F.EXIT_CHAIN_DESTROY
    assert time.time() - t0 < 60
    st = ch.chain_state()
    assert st["parts"]["full-smoke"]["rc"] == 4 and st["parts"]["smoke-b"]["rc"] == 4
    assert st["gate"]["result"] == "fail" and "part full-smoke ended with rc 4" in st["gate"]["problems"]
    for part, when in (("full-smoke", st["gate_by"]), ("smoke-b", st["stage1_deadline"])):
        pdir = fullrun.part_state_dir(part, ch.state)
        pq = json.loads((pdir / "queue.json").read_text(encoding="utf-8"))
        assert pq["final"]["status"] == "halted" and "stage deadline" in pq["final"]["reason"], part
        pev = [json.loads(x) for x in (pdir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        (sd,) = [e for e in pev if e["kind"] == "stage_deadline"]
        assert sd["stop_at"] == pytest.approx(when) and sd["wall"] >= when, part
        running = pq["items"][{"full-smoke": "smoke-p03", "smoke-b": "selftest"}[part]]
        # stopped at the stop time (interrupted), or - on a very slow host - never started before it
        assert running["status"] == ("interrupted" if running["attempts"] else "pending"), (part, running["status"])
    assert (fullrun.part_state_dir("full-smoke", ch.state) / fullrun.VERDICT_FILE).is_file()
    assert ch.summary()["status"] == "gate_failed"


# ======================================================================================== the stage-2 bootstrap


class HbSampler:
    """train_hb's mtime sampled every 10 ms on a thread (the controller must not beat while its bootstrap runs)."""

    def __init__(self, path: Path):
        self.path, self.seen, self._stop = path, [], threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.seen.append((time.time(), self.path.stat().st_mtime))
            except OSError:
                self.seen.append((time.time(), None))
            time.sleep(0.01)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join(5)

    def between(self, t0: float, t1: float) -> set:
        return {m for t, m in self.seen if t0 + 0.05 < t < t1 - 0.05}


@pytest.mark.parametrize("plan, attempts, why", [
    ([1, 1], 2, "stage 2 bootstrap failed (exit 1)"),
    ([3], 1, "stage 2 bootstrap failed (exit 3: a refusal, not retried)"),
    ([2], 1, "stage 2 bootstrap failed (exit 2: a refusal, not retried)"),
], ids=["rc1-twice", "rc3", "rc2"])
def test_a_failed_stage_2_bootstrap_ends_the_chain_with_5(ch, plan, attempts, why):
    """E.9.3 (6): rc 1 is retried once, rc 2 and 3 are not; then the chain failed (rc 5: destroy), p01 never ran."""
    ch.boot_state()
    parts = FakeParts()
    assert ch.make(parts=parts, boot_env=dict(FAKE_BOOT=json.dumps(plan))).run() == F.EXIT_CHAIN_DESTROY
    assert len(ch.boots()) == attempts and "p01" not in parts.ran()
    s = ch.summary()
    assert (s["status"], s["rc"], s["stage"], s["reason"], s["boot2"]["status"]) == ("failed", 5, 2, why, "failed")
    assert [e["rc"] for e in ch.events("chain_bootstrap_end")] == plan[:attempts]
    assert (ch.state / "logs" / "bootstrap-s2.log").is_file()
    # rc 1 then 0: the retry passes
    if plan == [1, 1]:
        ch.wipe()
        ch.boot_state()
        parts = FakeParts()
        assert ch.make(parts=parts, boot_env=dict(FAKE_BOOT="[1, 0]")).run() == F.EXIT_OK
        assert len(ch.boots()) == 2 and parts.ran()[-1] == "p01"


def test_a_hung_stage_2_bootstrap_is_killed_at_its_bound(ch, monkeypatch):
    """E.9.3 (6): a bootstrap that sleeps forever is killed at boot2_until (its session: its child too, on posix), the
    chain ends with 5, and the controller wrote no train_hb while it waited."""
    monkeypatch.setattr(F, "boot2_budget_s", lambda *a, **k: 3)  # the bound: 3 s after the handover
    monkeypatch.setattr(F, "BOOT2_MIN_LEFT_S", 1)
    ch.boot_state()
    with HbSampler(ch.state / fullrun.TRAIN_HB) as hb:
        rc = ch.make(parts=FakeParts(), boot_env=dict(FAKE_BOOT='["hang"]')).run()
    assert rc == F.EXIT_CHAIN_DESTROY
    s = ch.summary()
    assert s["status"] == "failed" and "stage 2 bootstrap timed out" in s["reason"]
    (start,), (end,) = ch.events("chain_bootstrap_start"), ch.events("chain_bootstrap_end")
    assert end["end"] == "timeout" and end["wall"] >= s["boot2"]["until"] - 0.1
    assert len(hb.between(start["wall"], end["wall"])) == 1, "the controller beat train_hb during the bootstrap"
    (b,) = ch.boots()
    child = int((ch.tmp / "boot-child-1").read_text())
    if os.name == "posix":
        for _ in range(50):
            if not Path(f"/proc/{child}").exists():
                break
            time.sleep(0.1)
        assert not Path(f"/proc/{child}").exists(), "the bootstrap's child outlived its session's kill"
    else:  # Windows: terminate() takes the bootstrap only (no process groups); the child is cleaned up here
        subprocess.run(["taskkill", "/F", "/PID", str(child)], capture_output=True)


def _gone(pid: int, wait_s: float = 10.0) -> bool:
    """posix: /proc/<pid> gone, or a zombie, within wait_s."""
    end = time.time() + wait_s
    while True:
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except OSError:
            return True
        if time.time() >= end:
            return False
        time.sleep(0.1)


def _reap(pid: int):
    if not _gone(pid, 0):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


POSIX_TIMEOUT = pytest.mark.skipif(
    os.name != "posix" or not shutil.which("timeout") or not shutil.which("bash"),
    reason="sessions, process groups and GNU timeout: posix only (the box's Linux)")
# a stand-in bootstrap's foreground phase as bootstrap.sh runs them: under GNU timeout (its own process group); the
# EXIT trap keeps bash from exec'ing the command in its place
PHASE = "trap 'echo exit trap' EXIT; timeout -k 5 600 sh -c 'echo $$ > {pidf}; exec sleep 600'"


def test_the_bootstraps_session_is_read_from_proc(ch):
    """ChainController.session: every live process whose /proc/<pid>/stat field 6 (session) is the bootstrap's pid,
    with its process group - a timeout-wrapped phase's own group included; zombies, other sessions and the controller
    itself left out. No /proc (Windows, the other tests' fake root): empty."""
    proc_root = ch.tmp / "proc"

    def stat(pid, state, pgrp, sid, start="7"):
        d = proc_root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        fields = [state, "1", str(pgrp), str(sid)] + ["0"] * 15 + [start] + ["0"] * 10
        (d / "stat").write_text(f"{pid} (a (b) c) " + " ".join(fields) + "\n", encoding="utf-8")

    stat(100, "S", 100, 100)  # the bootstrap: leads its session and its group
    stat(101, "S", 101, 100)  # timeout: a group of its own (setpgid), the same session
    stat(102, "R", 101, 100)  # 01 under timeout
    stat(103, "Z", 100, 100)  # a zombie
    stat(200, "S", 200, 200)  # another session
    stat(os.getpid(), "S", 100, 100)  # never the controller
    (proc_root / "self").mkdir()
    ctl = ch.make(proc_root=proc_root)
    assert ctl.session(100) == {100: 100, 101: 101, 102: 101}
    assert ctl.starttime(100) == "7" and ctl.session(999) == {} and ctl.pidns() is None
    assert ch.make(proc_root=ch.tmp / "no-such-proc").session(100) == {}


@POSIX_TIMEOUT
def test_a_timeout_wrapped_phase_dies_with_the_bootstraps_session(ch, monkeypatch):
    """bootstrap.sh runs its phases under GNU timeout, which moves itself and its command into a process group of their
    own (setpgid): a kill of the bootstrap's group alone left 01's rebuild running, reparented to init. A stand-in
    bootstrap whose foreground phase is `timeout 600 sleep`: at its bound the whole session goes, the sleep too."""
    monkeypatch.setattr(F, "boot2_budget_s", lambda *a, **k: 3)
    monkeypatch.setattr(F, "BOOT2_MIN_LEFT_S", 1)
    ch.boot_state()
    pidf = ch.tmp / "phase.pid"
    ctl = ch.make(parts=FakeParts(), proc_root=Path("/proc"))
    ctl.bootstrap_cmd = ["bash", "-c", PHASE.format(pidf=pidf)]
    assert ctl.run() == F.EXIT_CHAIN_DESTROY
    pid = int(pidf.read_text())
    try:
        assert _gone(pid), "the timeout-wrapped phase outlived the kill of the bootstrap"
        (a,) = ch.chain_state()["boot2"]["attempts"]
        assert a["end"] == "timeout" and a["pidns"] and a["pidns"].startswith("pid:")
    finally:
        _reap(pid)


@POSIX_TIMEOUT
@pytest.mark.parametrize("same_ns", [True, False], ids=["same-container", "another-container"])
def test_a_restart_kills_what_a_dead_bootstrap_left_in_its_session(ch, same_ns):
    """E.4.1 with the bootstrap itself gone (killed alone) and its timeout-wrapped phase still running in its session:
    a restarted controller kills that session while the pid namespace is the attempt's (a session id's pid is not
    reused while the session has a process); with another namespace (a container restart) nothing is killed."""
    ch.boot_state()
    pidf = ch.tmp / "phase.pid"
    boot = subprocess.Popen(["bash", "-c", PHASE.format(pidf=pidf)], start_new_session=True)
    orphan = None
    try:
        for _ in range(200):
            if pidf.is_file() and pidf.read_text().strip():
                break
            time.sleep(0.05)
        orphan = int(pidf.read_text())
        ctl = ch.make(parts=FakeParts(), proc_root=Path("/proc"))
        starttime = ctl.starttime(boot.pid)
        os.kill(boot.pid, signal.SIGKILL)  # the bootstrap alone dies; its phase runs on in its session
        boot.wait(10)
        assert not _gone(orphan, 0.3) and set(ctl.session(boot.pid)) >= {orphan}
        ctl.st.update(step="s2_boot", stage=2, gate=F.chain_gate(verdict_doc("full-smoke"), "full-smoke", 0,
                                                                 "test-sha"))
        for p in ("full-smoke", "smoke-b"):
            ctl.st["parts"][p].update(status="ended", rc=0)
        ctl.st["boot2"] = dict(status="running", until=time.time() + 3600, t0=time.time(), need_s=1, budget_s=1,
                               pull_min=30, rebuild_min=60, rest_bytes=0, pull_bytes=0, rate_bytes_s=1.0,
                               timeouts="test", attempts=[dict(t0=time.time() - 5, t1=None, pid=boot.pid,
                                                              starttime=starttime, rc=None, end=None,
                                                              pidns=ctl.pidns() if same_ns else "pid:[1]")])
        ctl.save()
        assert ch.make(parts=FakeParts(), proc_root=Path("/proc")).run() == F.EXIT_OK
        a = ch.chain_state()["boot2"]["attempts"]
        assert [x["end"] for x in a] == ["interrupted", "exit"] and a[0]["killed"] is same_ns
        assert _gone(orphan, 10 if same_ns else 1) is same_ns
    finally:
        if boot.poll() is None:
            boot.kill()
        if orphan:
            _reap(orphan)


def test_a_failed_attempt_with_no_time_left_names_its_exit(ch, monkeypatch):
    """E.4.1's words: an attempt that exited 1 and left no time before the bound for its retry ends the chain as
    "stage 2 bootstrap failed (exit 1)" (rc 5), not as a timeout that loses the exit code."""
    ch.boot_state()

    def attempt(self):
        b = self.st["boot2"]
        a = dict(t0=time.time(), t1=time.time(), pid=1, starttime=None, pidns=None, rc=1, end="exit")
        b["attempts"].append(a)
        b["until"] = time.time() - 1  # the attempt used the bound up
        self.save()
        return a

    monkeypatch.setattr(F.ChainController, "boot_attempt", attempt)
    parts = FakeParts()
    assert ch.make(parts=parts).run() == F.EXIT_CHAIN_DESTROY
    s = ch.summary()
    assert s["reason"].startswith("stage 2 bootstrap failed (exit 1) and its bound ") and s["status"] == "failed"
    assert len(s["boot2"]["attempts"]) == 1 and "p01" not in parts.ran()


def test_the_fit_rule(ch, monkeypatch):
    """E.9.3 (7): a huge projected box-1 time refuses at the handover (box 1 no longer fits, rc 5, no bootstrap); a
    bootstrap after which box 1 no longer fits ends the chain with 5 before p01 starts."""
    ch.boot_state()
    parts = FakeParts(verdict=lambda part: verdict_doc(part, box1_h=40.0))
    assert ch.make(parts=parts).run() == F.EXIT_CHAIN_DESTROY
    s = ch.summary()
    assert s["reason"].startswith("box 1 no longer fits: need ") and ch.boots() == [] and s["stage"] == 1
    need = s["boot2"]["need_s"]
    assert need == int(H * (40.0 + 0.3 + 0.5 + 0.35) + 60 * 45)
    (h,) = ch.events("chain_handover")
    assert h["need_s"] == need and h["left_s"] <= 35 * H
    # H2 passes; the bootstrap ends; box 1 no longer fits then (the need grew: a stand-in for a slow bootstrap)
    ch.wipe()
    ch.boot_state()
    calls = []
    real = F.ChainController.need_s

    def need_s(self):
        calls.append(1)
        return real(self) if len(calls) == 1 else int(40 * H)

    monkeypatch.setattr(F.ChainController, "need_s", need_s)
    parts = FakeParts()
    assert ch.make(parts=parts).run() == F.EXIT_CHAIN_DESTROY
    s = ch.summary()
    assert s["reason"].startswith("box 1 no longer fits") and len(ch.boots()) == 1 and "p01" not in parts.ran()
    assert s["boot2"]["status"] == "done" and s["stage"] == 2


def test_the_stage_2_bound_is_the_bootstraps_own_budget_or_box_1s_fit(ch):
    ch.boot_state()
    ctl = ch.make(parts=FakeParts())
    ctl.st["gate"] = dict(projected_box1_h=13.24)
    need = ctl.need_s()
    assert need == int(H * (13.24 + 0.3 + 0.5 + 0.35) + 60 * 45)
    assert F.phase_budget_s(3, 10) == 3 * (600 + 60) + 30 * 3 * 2 + 600
    assert F.boot2_budget_s(41, 400) == F.phase_budget_s(3, 10) + F.phase_budget_s(3, 30) + 2 * 10800 + \
        F.phase_budget_s(3, 400)
    assert F.boot2_budget_s(500, 45) == F.boot2_budget_s(500, 60) - F.phase_budget_s(3, 500) + F.phase_budget_s(3, 500)
    assert F.boot2_budget_s(10, 45) == F.boot2_budget_s(10, 60), "the rebuild's bound never below 60 min"


# ================================================================================================= restarts


def test_restarts_never_evaluate_the_gate_twice_nor_run_an_ended_part_again(ch, monkeypatch):
    """E.9.3 (8): a controller that dies after the gate (in smoke-b) and again during the handover goes on from
    chain.json: one gate, full-smoke run once, smoke-b resumed, the bootstrap and p01 once each."""
    ch.boot_state()
    parts = FakeParts(rcs={"smoke-b": [RuntimeError("controller killed"), 0]})
    real_report = F.ChainController.run_report_part

    def dies_in_smoke_b(self, part):  # an exception outside the part's own guard: the controller process dies
        if not getattr(dies_in_smoke_b, "done", False):
            dies_in_smoke_b.done = True
            raise KeyboardInterrupt("the container restarts")
        return real_report(self, part)

    monkeypatch.setattr(F.ChainController, "run_report_part", dies_in_smoke_b)
    with pytest.raises(KeyboardInterrupt):
        ch.make(parts=parts).run()
    st = ch.chain_state()
    assert st["step"] == "s1_rest" and st["gate"]["result"] == "pass" and st["parts"]["full-smoke"]["status"] == "ended"
    real_handover = F.ChainController.handover

    def dies_in_handover(self):
        if not getattr(dies_in_handover, "done", False):
            dies_in_handover.done = True
            raise KeyboardInterrupt("the container restarts")
        return real_handover(self)

    monkeypatch.setattr(F.ChainController, "handover", dies_in_handover)
    with pytest.raises(KeyboardInterrupt):
        ch.make(parts=parts).run()
    assert ch.make(parts=parts).run() == F.EXIT_OK
    assert len(ch.events("chain_gate")) == 1
    assert parts.ran().count("full-smoke") == 1 and parts.ran().count("p01") == 1
    assert ch.chain_state()["parts"]["smoke-b"]["status"] == "ended" and len(ch.boots()) == 1
    # an ended chain: a restart returns its rc
    assert ch.make(parts=parts).run() == F.EXIT_OK and parts.ran().count("p01") == 1


def fake_proc(proc_root: Path, pid: int, starttime: str, cmdline: str):
    d = proc_root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    fields = ["S"] + ["0"] * 18 + [starttime] + ["0"] * 10  # fields 3.. of /proc/<pid>/stat; field 22 = index 19
    (d / "stat").write_text(f"{pid} (bash) " + " ".join(fields) + "\n", encoding="utf-8")
    (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())


@pytest.mark.parametrize("verified", [True, False], ids=["live-verified", "recycled-pid"])
def test_a_restarted_controller_kills_only_a_verified_bootstrap(ch, verified):
    """E.9.3 (8): an attempt a dead controller left without an end: killed when /proc says it is that process
    (starttime and a vast/bootstrap.sh command line), spared when the pid was recycled; either way recorded
    interrupted (not counted) and a new attempt runs."""
    ch.boot_state()
    proc_root = ch.tmp / "proc"
    kw = dict(start_new_session=True) if os.name == "posix" else {}
    live = subprocess.Popen([PY, "-c", "import time; time.sleep(600)"], **kw)
    try:
        fake_proc(proc_root, live.pid, "4242" if verified else "1", "bash /workspace/K/vast/bootstrap.sh")
        parts = FakeParts()
        ctl = ch.make(parts=parts, proc_root=proc_root)
        # the dead controller's record: stage 1 done, the bootstrap running
        ctl.st.update(step="s2_boot", stage=2, gate=F.chain_gate(verdict_doc("full-smoke"), "full-smoke", 0,
                                                                 "test-sha"))
        for p in ("full-smoke", "smoke-b"):
            ctl.st["parts"][p].update(status="ended", rc=0)
        ctl.st["boot2"] = dict(status="running", until=time.time() + 3600, t0=time.time(), need_s=1, budget_s=1,
                               pull_min=30, rebuild_min=60, rest_bytes=0, pull_bytes=0, rate_bytes_s=1.0,
                               timeouts="test", attempts=[dict(t0=time.time() - 5, t1=None, pid=live.pid,
                                                              starttime="4242", rc=None, end=None)])
        ctl.save()
        assert ch.make(parts=parts, proc_root=proc_root).run() == F.EXIT_OK
        a = ch.chain_state()["boot2"]["attempts"]
        assert [x["end"] for x in a] == ["interrupted", "exit"] and a[0]["killed"] is verified
        for _ in range(50):
            if live.poll() is not None:
                break
            time.sleep(0.1)
        assert (live.poll() is not None) is verified
        assert parts.ran() == ["p01"] and len(ch.boots()) == 1
    finally:
        if live.poll() is None:
            live.kill()


# =============================================================================================== other ends


def test_p01_rc_4_halts_the_chain(ch):
    """E.9.3 (9): box 1's rc 4 (a training item failed for good) is the chain's: halted, stop."""
    ch.boot_state()
    assert ch.make(parts=FakeParts(rcs={"p01": [4]})).run() == F.EXIT_STOP
    s = ch.summary()
    assert (s["status"], s["rc"], s["parts"]["p01"]["status"]) == ("halted", 4, "ended")


def test_p01_rc_1_exits_1_and_resumes_it(ch):
    ch.boot_state()
    parts = FakeParts(rcs={"p01": [1, 0]})
    assert ch.make(parts=parts).run() == F.EXIT_FAIL
    assert ch.chain_state()["step"] == "s2_p01" and ch.summary()["rc"] == 1
    assert ch.make(parts=parts).run() == F.EXIT_OK
    assert parts.ran() == ["full-smoke", "smoke-b", "p01", "p01"] and len(ch.boots()) == 1


def test_the_hub_chain_summary_names_each_parts_queue_started_while_it_runs(ch, monkeypatch):
    """E.8's "started on this rental": a chain that dies in stage 2 leaves its Hub summary as it last put it, so that
    summary must name parts.p01.queue_started for the whole of box 1's run, not only once p01 has ended; with it and
    the Hub's p01 summary, launch --box p01 --resume says it continues the chain's stage 2."""
    import launch

    ch.boot_state()
    up, seen = FakeUploader(), {}

    def on_run(q):
        seen[q.part] = (json.loads(up.remote[fullrun.box_summary_path(CHAIN_BOX)]), q.state["started"])

    assert ch.make(parts=FakeParts(on_run=on_run), uploader=up).run() == F.EXIT_OK
    for part in ("full-smoke", "smoke-b", "p01"):
        hub, started = seen[part]
        assert (hub["parts"][part]["status"], hub["parts"][part]["queue_started"]) == ("running", started), part
    # the box dies during p01: the Hub keeps the chain summary seen then, and p01's own summary
    files = {fullrun.box_summary_path(CHAIN_BOX): json.dumps(seen["p01"][0]).encode(),
             fullrun.box_summary_path("p01"): up.remote[fullrun.box_summary_path("p01")]}

    class Hub:
        def file_exists(self, repo, path):
            return path in files

        def download(self, repo, path, local_dir):
            out = Path(local_dir) / Path(path).name
            out.write_bytes(files[path])
            return str(out)

    hub = Hub()
    monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
    problems, notes = launch.chain_resume_checks("u/kitsune-runs", "p01", CHAIN_BOX)
    assert problems == [] and len(notes) == 1 and "continues stage 2 of chain p01-chain (gate passed " in notes[0]


def test_an_ended_chain_writes_its_summary_again_on_a_restart(ch):
    """A controller that died between finish()'s chain.json and its summary: the restart returns the recorded rc and
    writes the summary to match it, so vast/supervise.py destroys on that exit 5 instead of restarting a chain whose
    summary still says running (and stopping it after its restarts)."""
    import supervise

    ch.boot_state()
    parts = FakeParts(verdict=lambda part: verdict_doc(part, {"2": False}))
    assert ch.make(parts=parts).run() == F.EXIT_CHAIN_DESTROY
    done = ch.summary()
    (ch.state / fullrun.SUMMARY_FILE).write_text(json.dumps(dict(done, status="running", rc=None, reason=None,
                                                                 ended=None)), encoding="utf-8")
    assert supervise.decide_queue(5, 0, summary=supervise.queue_summary(ch.state))[0] != "destroy"
    up = FakeUploader()
    assert ch.make(parts=parts, uploader=up).run() == F.EXIT_CHAIN_DESTROY
    assert ch.summary() == done, "the same record, its end time included"
    assert json.loads(up.remote[fullrun.box_summary_path(CHAIN_BOX)]) == done
    assert supervise.decide_queue(5, 0, summary=supervise.queue_summary(ch.state))[0] == "destroy"
    assert parts.ran() == ["full-smoke", "smoke-b"]


def test_smoke_b_failing_twice_is_recorded_and_the_chain_goes_on(ch):
    """E.9.3 (10): smoke-b rc 1, then an exception: recorded failed (chain_part_failed), the chain goes on to box 1."""
    ch.boot_state()
    parts = FakeParts(rcs={"smoke-b": [1, RuntimeError("smoke-b broke")]})
    assert ch.make(parts=parts).run() == F.EXIT_OK
    assert parts.ran() == ["full-smoke", "smoke-b", "smoke-b", "p01"]
    (f,) = ch.events("chain_part_failed")
    assert f["part"] == "smoke-b" and "smoke-b broke" in f["why"] and f["fails"] == 2
    assert ch.summary()["parts"]["smoke-b"]["status"] == "failed"


def test_every_part_gets_its_window_and_its_items_their_deadline(ch):
    """E.9.3 (11) and E.4.5: full-smoke until gate_by (its items' KITSUNE_DEADLINE gate_by - 20 min), smoke-b until the
    stage-1 sub-deadline (- 15 min), p01 on the box deadline (- 45 min); every part's box-wide files in $STATE."""
    now = int(time.time())
    ch.boot_state(first_boot=now, deadline=now + 35 * 3600)
    parts = FakeParts()
    assert ch.make(parts=parts).run() == F.EXIT_OK
    st = ch.chain_state()
    assert st["gate_by"] == now + 9 * 3600 and st["stage1_deadline"] == now + 10.5 * 3600
    by = {c["part"]: c for c in parts.calls}
    assert (by["full-smoke"]["deadline"], by["full-smoke"]["stop_at"]) == (st["gate_by"], st["gate_by"])
    assert (by["smoke-b"]["deadline"], by["smoke-b"]["stop_at"]) == (st["stage1_deadline"], st["stage1_deadline"])
    assert (by["p01"]["deadline"], by["p01"]["stop_at"]) == (None, None)
    assert all(c["box_state"] == ch.state and c["state"] == fullrun.part_state_dir(c["part"], ch.state)
               for c in parts.calls)
    ctl = ch.make()  # real part queues (not run): their item deadlines and box-wide files
    for part, item, want in (("full-smoke", "smoke-p03", st["gate_by"] - 20 * 60),
                             ("smoke-b", "selftest", st["stage1_deadline"] - 15 * 60),
                             ("p01", "full-p01", now + 35 * 3600 - 45 * 60)):
        q = ctl.queue(part, *ctl.part_window(part))
        assert q.item_deadline(item) == pytest.approx(want), part
        assert q.ctl_beat_path() == ch.state / fullrun.TRAIN_HB and q.box_deadline() is not None
        assert Path(q.s.state_dir) == fullrun.part_state_dir(part, ch.state)


def test_first_boot_falls_back_to_the_deadline_less_the_cap(ch, monkeypatch):
    """E.9.3 (13): no first_boot file: deadline - KITSUNE_MAX_HOURS (else the chain's max_hours), never the
    controller's own start; kept in chain.json across restarts."""
    now = int(time.time())
    ch.boot_state(deadline=now + 30 * 3600)
    (ch.state / "first_boot").unlink()
    monkeypatch.setenv(fullrun.ENV_MAX_HOURS, "32")
    assert ch.make(parts=FakeParts()).st["first_boot"] == now + 30 * 3600 - 32 * 3600
    monkeypatch.delenv(fullrun.ENV_MAX_HOURS)
    ctl = ch.make(parts=FakeParts())
    assert ctl.st["first_boot"] == now + 30 * 3600 - 35 * 3600
    assert ctl.st["gate_by"] == ctl.st["first_boot"] + 9 * 3600
    ctl.save()
    (ch.state / "first_boot").write_text(f"{now}\n")  # a later file never moves a recorded chain
    assert ch.make(parts=FakeParts()).st["first_boot"] == now + 30 * 3600 - 35 * 3600


def test_stage2_timeouts_from_the_stage_1_plan(ch, tmp_path):
    """E.4.2: the rest of the download (the last stage's upstream bytes less stage 1's), its labels + 2 GB, at the
    gate's rate, through netgate.timeouts; without the record: the gate's full download (never shorter)."""
    from kitsune import extent, netgate

    def inp(n, gb, step):  # one upstream input of `gb` GB and its one stem (the canonical step key = the source)
        return {"input": f"f{n}", "ordinal": n, "bytes": gb * 1e9, "stems": [
            {"stem": f"train-{n:05d}", "split": "train", "step": step, "rows": 10, "hours": 100.0,
             "ids_sha256": "0" * 64, "shard_bytes": gb * 1e9}]}

    record = {"schema": 1, "name": "full", "root": "labels/full", "canonical_version": extent.CANONICAL_VERSION,
              "names": [], "inputs": {}, "sources": {
                  "galgame": {"repo": "g", "rows": 50, "hours": 500.0, "bytes": 50e9,
                              "inputs": [inp(n, 10, "galgame") for n in range(5)]},
                  "reazon_large": {"repo": "r", "rows": 600, "hours": 6000.0, "bytes": 60e9,
                                   "inputs": [inp(n, 1, "reazon_large") for n in range(60)]}}}
    rec_path = tmp_path / "extent.json"
    rec_path.write_text(json.dumps(record), encoding="utf-8")
    s1 = ch.state / "chain" / "stage1"
    s1.mkdir(parents=True)
    (s1 / "bootstrap_plan.json").write_text(json.dumps({"record": str(rec_path)}), encoding="utf-8")
    ch.boot_state(gate={"verdict": "pass", "rate_bytes_s": 50e6})
    t = F.stage2_timeouts(ch.state, ch.root, "configs/full/data-smoke.json", "configs/full/data-p01.json")
    cfg1 = json.loads((ch.root / "configs/full/data-smoke.json").read_text())
    cfg2 = json.loads((ch.root / "configs/full/data-p01.json").read_text())
    z1, z2 = extent.sizing(record, cfg1), extent.sizing(record, cfg2)
    assert t["source"] == "record" and t["rest_bytes"] == int((z2["down_gb"] - z1["down_gb"]) * 1e9)
    assert t["rest_bytes"] == int(2 * 10e9 + 7 * 1e9), "galgame's last 2 tars, reazon_large's inputs 53-59"
    assert t["pull_bytes"] == int((z2["labels_gb"] + 2) * 1e9) and t["rate_bytes_s"] == 50e6
    assert (t["pull_min"], t["rebuild_min"]) == netgate.timeouts(50e6, t["rest_bytes"], t["pull_bytes"])
    (s1 / "bootstrap_plan.json").unlink()
    assert os.environ.get(fullrun.ENV_GATE_BYTES) is None  # the fixture cleared launch's env
    t = F.stage2_timeouts(ch.state, ch.root, "configs/full/data-smoke.json", "configs/full/data-p01.json")
    assert t["source"].startswith("fallback") and t["rest_bytes"] == int(netgate.GATE_REF_GB * 1e9)
    assert t["pull_bytes"] == int(F.PULL_FALLBACK_BYTES)


def test_plan_and_resume_pull_on_a_chain(ch, monkeypatch, capsys):
    (ch.root / fullrun.BOXES_FILE).write_text(json.dumps(ch.reg), encoding="utf-8")
    monkeypatch.setenv(fullrun.ENV_REGISTRY, str(ch.root / fullrun.BOXES_FILE))
    assert F.main(["plan", "--box", CHAIN_BOX]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [r["stage"] for r in rows if "parts" in r] == [1, 2]
    assert [r["item"] for r in rows if r.get("part") == "p01"] == ["stores-ctc", "full-p01", "m4-full-p01"]
    assert F.main(["resume-pull", "--box", CHAIN_BOX]) == F.EXIT_REFUSED
    out = capsys.readouterr().out
    assert "is not resumed as a chain" in out and "--box p01 --resume" in out

