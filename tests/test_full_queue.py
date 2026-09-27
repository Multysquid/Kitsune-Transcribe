"""The full-data runs' box queue (kitsune/full_queue.py; build contract sections 5 and 11), CPU only, on fakes:

- the order and argv: box p01's stores -> train -> readout with the binding readout argv (and runs/m4-<id>-r1 after a
  continuation), the train argv's hf sets, the per-item heartbeat and deadline env, the queue summary at
  full/box-<box>/queue_summary.json (put when a run dir appears), the item_done line in each readout/eval out dir
- box full on two fake GPUs: the CTC stores first, full-p03 during stores-aed, no eval-pool item while a training item
  waits, one process per GPU, a readout on its run's GPU, eval placeholders filled, runs-repo weights fetched and an
  unresolved of_box skipped
- failures: exit 3 on one item (failed alone, rc 4 at the end), a store build that fails for good (its dependants
  skipped, no QueueError), a retried train item resuming in its run dir, a readout that fails (recorded, not retried)
- the no-start rule (a droppable item skipped, a non-droppable one overridden) and KITSUNE_DEADLINE
- stalls (a hung trainer killed and resumed; an overrun kill for an item without a stall check), the controller
  heartbeat (beaten every poll, held during a freeze fault, across a summary put)
- smoke A end to end: the faults (kill, wipe_run_dir through the Hub and check-resume, deadline, freeze; sigstop on
  posix only), the verdict (checks 1-11, check 3's math); a smoke-b-style box's verdict from its registry specs only
- resume-pull (the newest of the scratch and runs states, checksum refusals, a Hub-complete run, a reset from the
  runs repo's pre_cooldown state, a set-only run past its cooldown, an unknown id) and the adoption of its plan (only
  without queue.json, done items not run again, a reset's readout into -r1, sets_once until the first state after the
  reset), only_if_new_machine, the resource peaks from a fake /proc, and the CLI's build-stores / check-resume / plan
Every process is tests/fake_study_trainer.py (fake GPUs = distinct CUDA_VISIBLE_DEVICES values); the Hub is a recording
fake uploader or tests/fake_runs_repo.py's DirHub.
"""
import copy
import json
import os
import shutil
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
from fixtures_full import STUDY_WEIGHTS, tiny_registry  # noqa: E402
from kitsune import full_queue as F  # noqa: E402
from kitsune import fullrun  # noqa: E402
from kitsune import study_queue as Q  # noqa: E402

FAKE = (ROOT / "tests" / "fake_study_trainer.py").as_posix()
PY = sys.executable
RESUME_ENV = (fullrun.ENV_RESUME, fullrun.ENV_RESUME_RESET, fullrun.ENV_RESUME_SETS, fullrun.ENV_THREADS_PER_GPU,
              fullrun.ENV_DEADLINE, fullrun.ENV_HEARTBEAT, fullrun.ENV_MACHINE_ID, fullrun.ENV_SCRATCH_REPO)


class FakeUploader:
    """The runs repo as the queue sees it: every run dir synced (with the lean files finish.py would verify), every
    file put (its JSON kept, for the summary puts), downloads served from `remote`."""

    def __init__(self, remote: dict | None = None, fail_sync: set | None = None):
        self.synced, self.put, self.remote = [], [], dict(remote or {})
        self.fail_sync = set(fail_sync or ())

    def sync_run(self, run_dir: Path) -> list[str]:
        import finish

        self.synced.append((time.time(), run_dir.name, sorted(finish.expected_files(run_dir, False, lean=True))))
        return ["missing"] if run_dir.name in self.fail_sync else []

    def put_file(self, local, path_in_repo) -> list[str]:
        data = Path(local).read_bytes()
        self.put.append((time.time(), path_in_repo, json.loads(data) if path_in_repo.endswith(".json") else None))
        self.remote[path_in_repo] = data
        return []

    def exists(self, path_in_repo) -> bool:
        return path_in_repo in self.remote

    def download(self, path_in_repo, local_dir) -> Path:
        p = Path(local_dir) / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.remote[path_in_repo])
        return p

    def download_dir(self, prefix, local_dir) -> Path:
        for k in [k for k in self.remote if k.startswith(prefix.rstrip("/") + "/")]:
            self.download(k, local_dir)
        return Path(local_dir) / prefix

    def list_dir(self, path_in_repo) -> list[str]:
        pre = path_in_repo.rstrip("/") + "/"
        return sorted({k[len(pre):].split("/", 1)[0] for k in self.remote if k.startswith(pre)})

    def summaries(self, box: str) -> list[dict]:
        return [x for _, p, x in self.put if p == fullrun.box_summary_path(box)]


def weights_remote(*names: str) -> dict:
    """The runs repo's copies of study weights (config.json + checkpoints/step_<N>/)."""
    out = {}
    for n in names:
        rid, step = STUDY_WEIGHTS[n]
        out[f"runs/{rid}/config.json"] = json.dumps({"config": {"run_name": n}}).encode()
        out[f"runs/{rid}/checkpoints/step_{step}/model.safetensors"] = b"w" * 16
    return out


def fake_evals(reg: dict) -> dict:
    """reg with every eval item's command replaced by the fake's eval mode (its placeholders kept after it)."""
    reg = copy.deepcopy(reg)
    for box in reg["boxes"].values():
        for it in box["items"]:
            if it["kind"] == "eval":
                it["argv"] = ["{python}", FAKE, "eval", *[a for a in it["argv"][1:] if "{" in a or a.startswith("--")]]
    return reg


def train_configs(root: Path, **extra):
    """tiny_registry's item configs made loadable by the fake trainer: the epochs clock, one epoch."""
    for p in (root / "configs" / "full").glob("*.json"):
        c = json.loads(p.read_text(encoding="utf-8"))
        if "run_name" in c:
            c["schedule"] = {"clock": "epochs", "epochs": 1}
            c.update(copy.deepcopy(extra.get(c["run_name"], {})))
            p.write_text(json.dumps(c, indent=1), encoding="utf-8")


@pytest.fixture
def fq(tmp_path, monkeypatch):
    """A box checkout in tmp_path/box: tiny_registry's configs (fake-loadable), a runs/ dir, state in
    tmp_path/state; the queue's processes are the fake trainer."""
    for k in RESUME_ENV:
        monkeypatch.delenv(k, raising=False)
    root = tmp_path / "box"
    reg = fake_evals(tiny_registry(root))
    train_configs(root)
    (root / "runs").mkdir(parents=True)
    log = tmp_path / "fake-log"
    state = tmp_path / "state"

    def make(box="p01", registry=None, uploader=None, env=None, gpus=None, **settings):
        r = registry if registry is not None else reg
        try:
            n = fullrun.box_spec(box, fullrun.load_registry(r, root=root, check_files=False))["gpus"]
        except fullrun.RegistryError:
            n = 1
        base = dict(root=root, state_dir=state, out_repo=None, gpus=list(gpus or [str(i) for i in range(n)]),
                    python=PY, train_cmd=[PY, FAKE, "train"], stores_cmd=[PY, FAKE, "stores"],
                    eval_cmd=[PY, FAKE, "readout"], speed_cmd=[PY, FAKE, "speed"],
                    check_resume_cmd=[PY, FAKE, "check-resume"],
                    uploader=uploader if uploader is not None else FakeUploader(), n_gpus=None, poll_s=0.02,
                    kill_grace_s=0.5, summary_min_s=0.0, scratch_repo=None, machine_id=None, sha="test-sha",
                    container_id="c1", proc_root=tmp_path / "no-proc", cgroup=tmp_path / "no-cgroup",
                    env=dict(FAKE_LOG=str(log), **(env or {})))
        base.update(settings)
        return F.FullQueue(box, F.FullSettings(**base), registry=r)

    def records(mode=None):
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in log.glob("*.json")] if log.is_dir() else []
        return sorted((x for x in recs if mode is None or x["mode"] == mode), key=lambda x: x["t0"])

    def st():
        return json.loads((state / "queue.json").read_text(encoding="utf-8"))

    def summary():
        return json.loads((state / fullrun.SUMMARY_FILE).read_text(encoding="utf-8"))

    def events(kind=None):
        p = state / "events.jsonl"
        ev = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []
        return [e for e in ev if kind is None or e["kind"] == kind]

    def reset():
        """A second queue run in the same checkout: no state, records, run dirs or fake launch counters."""
        for d in (state, log, root / "runs", root / ".fake_counts"):
            shutil.rmtree(d, ignore_errors=True)
        (root / "runs").mkdir()

    return SimpleNamespace(make=make, records=records, root=root, state=state, reg=reg, st=st, summary=summary,
                           events=events, tmp=tmp_path, reset=reset)


def by_item(recs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for x in recs:
        out.setdefault(x.get("item"), []).append(x)
    return out


def box_only(reg: dict, box: str, keep=None, **changes) -> dict:
    """reg reduced to one box (keep: its item names to keep), with box fields changed."""
    reg = copy.deepcopy(reg)
    b = reg["boxes"][box]
    if keep is not None:
        b["items"] = [it for it in b["items"] if it["name"] in keep]
        names = {it["name"] for it in b["items"]}
        for it in b["items"]:
            it["needs"] = [n for n in it.get("needs", []) if n in names]
        b["faults"] = [f for f in b.get("faults", []) if f["item"] in names]
    b.update(changes)
    reg["boxes"] = {box: b}
    return reg


def set_item(reg: dict, box: str, name: str, **fields) -> dict:
    for it in reg["boxes"][box]["items"]:
        if it["name"] == name:
            it.update(fields)
    return reg


# ================================================================================================ order and argv


def test_box_p01_runs_stores_train_readout_with_the_binding_argv(fq):
    up = FakeUploader()
    q = fq.make("p01", uploader=up, out_repo="u/kitsune-runs", scratch_repo="u/kitsune-scratch")
    assert q.run() == F.EXIT_OK
    recs = fq.records()
    assert [x["item"] for x in recs] == ["stores-ctc", "full-p01", "m4-full-p01"]
    stores, train, readout = recs
    assert stores["argv"] == ["stores", "--config", "configs/full/full-p01.json"]
    assert train["argv"] == ["train", "--config", "configs/full/full-p01.json", "--set", "run_name=full-p01",
                             "--set", "hf.output_repo=u/kitsune-runs", "--set", "hf.scratch_repo=u/kitsune-scratch"]
    st = fq.st()
    rd = st["items"]["full-p01"]["run_dir"]
    rid = Path(rd).name
    steps = st["items"]["full-p01"]["result"]["steps"]
    assert steps == 20 and rid.startswith("full-p01-")
    assert readout["argv"] == [
        "readout", "--config", f"{rd}/config.json", "--ckpt", f"{rd}/checkpoints/step_{steps}", "--out",
        f"runs/m4-{rid}", "--manifest", fullrun.FROZEN_MANIFEST, "--tables", f"runs/m4-{rid}/tables", "--system",
        "full-p01", "--cache-dir", (fq.root / "cache").as_posix(), "--max-temp", "0"]
    # every child: its heartbeat file (touched by the queue at its start); no deadline file -> no KITSUNE_DEADLINE
    for x in recs:
        assert x["heartbeat"] == str(fullrun.item_hb_path(x["item"], fq.state)) and x["deadline"] is None
        assert fullrun.item_hb_path(x["item"], fq.state).is_file()
    # the readout's out dir: study.json's numbers in the result, an item_done line, uploaded with the run dir
    res = st["items"]["m4-full-p01"]["result"]
    assert res["m4"] == pytest.approx(0.12) and res["jsut_cer_nostyle"] == pytest.approx(0.05)
    assert res["system"] == "full-p01" and res["out"] == f"runs/m4-{rid}"
    assert Q.read_events(fq.root / f"runs/m4-{rid}")[-1]["kind"] == "item_done"
    assert [n for _, n, _ in up.synced] == [rid, f"m4-{rid}"]
    assert st["items"]["full-p01"]["result"] == dict(run_id=rid, steps=20, status="complete", epochs=1.0,
                                                     stopped_early=False, end_reason="schedule", resume_resets=0)
    # the summary, in the runs repo: its shape; put when the run dir appeared (a running summary naming it)
    s = fq.summary()
    assert {"format", "kind", "box", "status", "reason", "rc", "sha", "machine_id", "container_id", "gpus",
            "registry_sha256", "deadline", "started", "ended", "host_mem_peak_gb", "items", "readouts_failed",
            "no_start", "not_needed", "faults", "resumed"} <= set(s)
    assert (s["format"], s["kind"], s["box"], s["status"], s["rc"], s["sha"]) == (1, "full", "p01", "complete", 0,
                                                                                 "test-sha")
    assert set(s["items"]["full-p01"]) >= {"kind", "status", "run_dir", "out", "of", "gpu", "verified", "attempts",
                                           "hb_max_gap_s", "stalls", "peak_rss_gb", "result"}
    puts = up.summaries("p01")
    assert puts[-1] == json.loads((fq.state / fullrun.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert any(p["status"] == "running" and p["items"]["full-p01"]["status"] == "running"
               and p["items"]["full-p01"]["run_dir"] == rd for p in puts)
    assert fq.events("run_dir_found")[0]["run_dir"] == rd
    # a restart of an ended box does nothing
    assert fq.make("p01", uploader=up, out_repo="u/kitsune-runs", scratch_repo="u/kitsune-scratch").run() == 0
    assert len(fq.records()) == 3


def test_a_timed_box_refuses_to_run_without_its_scratch_repo(fq):
    with pytest.raises(Q.QueueError, match="KITSUNE_SCRATCH_REPO"):
        fq.make("p01", out_repo="u/kitsune-runs")
    with pytest.raises(Q.QueueError, match="registered with 1 GPU"):
        fq.make("p01", gpus=["0", "1"])
    with pytest.raises(Q.QueueError, match="not in the registry"):
        fq.make("smoke-b", registry=box_only(fq.reg, "p01"))


def test_a_retried_train_item_resumes_in_its_run_dir_and_a_failed_one_skips_its_readout(fq):
    env = dict(FAKE_RC=json.dumps({"full-p01": [1, 0]}), FAKE_RC_AT="8")  # dies at step 8, after full_step_5
    assert fq.make("p01", env=env).run() == F.EXIT_OK
    tr = [x for x in fq.records("train")]
    assert [x["rc"] for x in tr] == [1, 0] and tr[1]["resumed"] and tr[1]["resumed_from"] == 5
    st = fq.st()
    assert tr[1]["argv"][:3] == ["train", "--resume", st["items"]["full-p01"]["run_dir"]]
    assert tr[0]["run_dir"] == tr[1]["run_dir"] and st["items"]["m4-full-p01"]["status"] == "done"
    # every attempt fails: failed after max_attempts, its readout skipped, the box stops (rc 4)
    fq.reset()
    reg = copy.deepcopy(fq.reg)
    reg["boxes"]["p01"]["max_attempts"] = 2
    assert fq.make("p01", registry=reg, env=dict(FAKE_RC=json.dumps({"full-p01": 1}))).run() == F.EXIT_STOP
    st = fq.st()
    assert st["items"]["full-p01"]["status"] == "failed" and len(st["items"]["full-p01"]["attempts"]) == 2
    assert st["items"]["m4-full-p01"]["status"] == "skipped" and "m4-full-p01" not in by_item(fq.records())
    assert fq.summary()["status"] == "halted" and "full-p01" in fq.summary()["reason"]


# ================================================================================================= box full


def test_box_full_shares_two_gpus_stores_first_and_the_pool_after_training(fq):
    up = FakeUploader(remote=weights_remote("study-t06"))
    (fq.state).mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 100 * 3600
    (fq.state / "deadline").write_text(f"{deadline}\n")
    env = dict(FAKE_ITEM_S=json.dumps({"stores-aed": 0.8, "speed-": 0.3}),
               FAKE_EPOCH_STEPS=json.dumps({"full-t06": 250}), FAKE_STEP_S="0.004")
    q = fq.make("full", uploader=up, env=env)
    assert q.run() == F.EXIT_OK
    recs = fq.records()
    it = {x["item"]: x for x in recs}
    # a speed item has the host to itself: no other item runs and no upload runs while it times
    for sp in ("speed-full-t06", "speed-study-t06"):
        s = it[sp]
        assert all(x["t1"] <= s["t0"] or x["t0"] >= s["t1"] for x in recs if x is not s), sp
        assert not [t for t, _, _ in up.synced if s["t0"] <= t <= s["t1"]], sp
    ctc, aed = it["stores-ctc"], it["stores-aed"]
    assert all(x["t0"] >= ctc["t1"] for x in recs if x is not ctc)  # the CTC store first, alone
    assert aed["t0"] < it["full-p03"]["t0"] < aed["t1"] and it["full-t06"]["t0"] >= aed["t1"]  # p03 during stores-aed
    assert it["full-p03"]["gpu"] != aed["gpu"]
    for gpu in ("0", "1"):  # one process per GPU at any time
        iv = sorted((x["t0"], x["t1"]) for x in recs if x["gpu"] == gpu)
        assert all(a[1] <= b[0] for a, b in zip(iv, iv[1:])), gpu
    last_train_start = max(it[n]["t0"] for n in ("full-t06", "full-p03", "full-p005"))
    for name in ("whisper-small", "quant-int8-w8a8-full-p03", "speed-study-t06", "speed-full-t06"):
        assert it[name]["t0"] >= last_train_start, name  # the eval pool only once no training item waits
    for n in ("full-p03", "full-p005"):  # a readout follows its run on that run's GPU, ahead of the next training
        assert it[f"m4-{n}"]["gpu"] == it[n]["gpu"] and it[f"m4-{n}"]["t0"] >= it[n]["t1"]
    assert it["m4-full-p03"]["t1"] <= it["full-p005"]["t0"]
    # the per-item deadline: the box deadline less deadline_reserve_min (60)
    assert all(int(x["deadline"]) == int(deadline - 3600) for x in recs)
    st = fq.st()
    # eval placeholders: {config} / {ckpt} of the same-box `of` (after its readout), {out} a stamped run dir
    p03 = st["items"]["full-p03"]
    qa = it["quant-int8-w8a8-full-p03"]["argv"]
    assert qa[qa.index("--config") + 1] == f"{p03['run_dir']}/config.json"
    assert qa[qa.index("--ckpt") + 1] == f"{p03['run_dir']}/checkpoints/step_{p03['result']['steps']}"
    out = st["items"]["quant-int8-w8a8-full-p03"]["out"]
    assert qa[qa.index("--out") + 1] == out and out.startswith("runs/quant-int8-w8a8-full-p03-")
    assert qa[qa.index("--manifest") + 1] == fullrun.FROZEN_MANIFEST
    assert Q.read_events(fq.root / out)[-1]["kind"] == "item_done"
    # the of_box item: box p01's summary is not in the runs repo -> skipped, not failed
    assert st["items"]["quant-int8-w8a8-full-p01"]["status"] == "skipped"
    assert st["items"]["quant-int8-w8a8-full-p01"]["why"] == "of_box unresolved"
    # speed: the study weights fetched into cache/hub, both items into one runs/speed-full-<stamp>/speed.json
    rid, step = STUDY_WEIGHTS["study-t06"]
    sp = it["speed-study-t06"]["argv"]
    assert sp[sp.index("--model") + 1] == (fq.root / f"cache/hub/runs/{rid}/checkpoints/step_{step}").as_posix()
    assert (fq.root / f"cache/hub/runs/{rid}/config.json").is_file()
    sd = st["speed_dir"]["run_dir"]
    got = json.loads((fq.root / sd / "speed.json").read_text(encoding="utf-8"))
    assert set(got["systems"]) == {"study-t06", "full-t06"} and sd.startswith("runs/speed-full-")
    assert [x for x in sp[sp.index("--per-set"):sp.index("--per-set") + 4]] == ["--per-set", "40", "--seed", "1234"]
    assert sp[sp.index("--out") + 1] == f"{sd}/speed.json" and "--require-idle" in sp
    assert any(n == Path(sd).name for _, n, _ in up.synced) and st["speed_dir"]["verified"] is True
    # the summary records every kind's result
    s = fq.summary()
    assert s["items"]["speed-full-t06"]["result"] == {"system": "full-t06", "out": sd}
    assert s["items"]["whisper-small"]["result"]["rc"] == 0 and s["deadline"] == pytest.approx(deadline)


def test_an_of_box_item_reads_the_other_boxs_summary_and_fetches_its_weights(fq):
    rid = "full-p01-20260927T120000Z"
    summ = {"machine_id": "m1", "items": {"full-p01": {"status": "done", "run_dir": f"runs/{rid}",
                                                       "result": {"run_id": rid, "steps": 110520}}}}
    remote = {fullrun.box_summary_path("p01"): json.dumps(summ).encode(),
              f"runs/{rid}/config.json": b"{}", f"runs/{rid}/checkpoints/step_110520/model.safetensors": b"w"}
    q = fq.make("full", uploader=FakeUploader(remote=remote))
    q.register()
    name = "quant-int8-w8a8-full-p01"
    assert q._prepare(name) is True
    assert q.item(name)["source"] == {"box": "p01", "run_id": rid, "steps": 110520}
    hub = (fq.root / "cache" / "hub").as_posix()
    argv = q.argv_for(name, q.item(name), None)
    assert argv[argv.index("--config") + 1] == f"{hub}/runs/{rid}/config.json"
    assert argv[argv.index("--ckpt") + 1] == f"{hub}/runs/{rid}/checkpoints/step_110520"
    assert (fq.root / "cache" / "hub" / "runs" / rid / "checkpoints" / "step_110520" / "model.safetensors").is_file()
    assert fq.events("weights_fetched")[0] == dict(fq.events("weights_fetched")[0], run_id=rid, step=110520)
    # resolved, but its weights are not in the runs repo: a failed readout, recorded
    shutil.rmtree(fq.root / "cache")
    shutil.rmtree(fq.state)
    q2 = fq.make("full", uploader=FakeUploader(remote={fullrun.box_summary_path("p01"): json.dumps(summ).encode()}))
    q2.register()
    assert q2._prepare(name) is False and q2.item(name)["status"] == "failed"
    assert "step_110520" in q2.state["readouts_failed"][name]
    # a template the queue cannot fill (its `of` has no run dir) fails that item, never the queue
    for n in ("full-p03", "m4-full-p03"):
        q2.item(n)["status"] = "done"
    assert q2._prepare("quant-int8-w8a8-full-p03") is False
    assert "{config}" in q2.state["readouts_failed"]["quant-int8-w8a8-full-p03"]


def test_exit_3_fails_that_item_only_and_the_box_ends_with_rc_4(fq):
    reg = box_only(fq.reg, "full", keep=["stores-ctc", "stores-aed", "full-t06", "full-p03", "full-p005",
                                         "m4-full-t06", "m4-full-p03", "m4-full-p005"])
    env = dict(FAKE_RC=json.dumps({"full-p005": 3}), FAKE_EPOCH_STEPS=json.dumps({"full-t06": 200}))
    assert fq.make("full", registry=reg, env=env).run() == F.EXIT_STOP
    st = fq.st()
    p005 = st["items"]["full-p005"]
    assert p005["status"] == "failed" and len(p005["attempts"]) == 1 and "throughput" in p005["why"]
    assert all(st["items"][n]["status"] == "done" for n in ("full-t06", "full-p03", "m4-full-t06", "m4-full-p03"))
    assert st["items"]["m4-full-p005"]["status"] == "skipped"
    assert [e["reason"] for e in fq.events("item_failed")] == ["throughput"]
    assert fq.summary()["status"] == "halted" and "full-p005" in fq.summary()["reason"]


def test_a_store_build_that_fails_for_good_skips_its_dependants(fq):
    reg = box_only(fq.reg, "full", keep=["stores-ctc", "stores-aed", "full-t06", "full-p03", "m4-full-t06",
                                         "m4-full-p03"], max_attempts=3)
    assert fq.make("full", registry=reg, env=dict(FAKE_FAIL=json.dumps({"stores-aed": 1}))).run() == F.EXIT_STOP
    st = fq.st()
    assert st["items"]["stores-aed"]["status"] == "failed" and len(st["items"]["stores-aed"]["attempts"]) == 3
    assert st["items"]["full-t06"]["status"] == "skipped" and st["items"]["m4-full-t06"]["status"] == "skipped"
    assert st["items"]["full-p03"]["status"] == "done" and st["items"]["m4-full-p03"]["status"] == "done"
    assert "full-t06" not in by_item(fq.records())
    # a store build that fails once is retried
    fq.reset()
    assert fq.make("full", registry=reg, env=dict(FAKE_FAIL=json.dumps({"stores-ctc": [1, 0]}))).run() == F.EXIT_OK
    assert [x["rc"] for x in by_item(fq.records())["stores-ctc"]] == [1, 0]


def test_a_failed_readout_is_recorded_and_never_retried(fq):
    assert fq.make("p01", env=dict(FAKE_FAIL=json.dumps({"m4-full-p01": 1}))).run() == F.EXIT_OK
    st = fq.st()
    assert st["items"]["m4-full-p01"]["status"] == "failed" and len(st["items"]["m4-full-p01"]["attempts"]) == 1
    s = fq.summary()
    assert s["readouts_failed"] == {"m4-full-p01": "exit 1"} and "m4-full-p01" in s["reason"]
    assert Q.read_events(fq.root / st["items"]["m4-full-p01"]["out"])[-1]["status"] == "failed"


# ============================================================================================ scheduling rules


def test_the_no_start_rule_skips_a_droppable_item_and_overrides_the_others(fq):
    reg = box_only(fq.reg, "full", keep=["stores-ctc", "stores-aed", "full-t06", "full-p03", "full-p005",
                                         "m4-full-p005"])
    fq.state.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 5 * 3600  # item deadline = +4 h: full-t06 (35 h) and full-p03 (22 h) do not fit
    (fq.state / "deadline").write_text(f"{deadline}\n")
    assert fq.make("full", registry=reg).run() == F.EXIT_OK
    st = fq.st()
    assert st["items"]["full-p005"]["status"] == "skipped" and st["items"]["m4-full-p005"]["status"] == "skipped"
    assert st["no_start"]["full-p005"]["skipped"] is True
    ns = fq.events("item_not_started")
    assert [e["item"] for e in ns] == ["full-p005"] and ns[0]["need_s"] == round(9.92 * 3600)
    assert 4 * 3600 - 60 < ns[0]["left_s"] <= 4 * 3600
    assert sorted(e["item"] for e in fq.events("no_start_overridden")) == ["full-p03", "full-t06"]
    assert st["items"]["full-t06"]["status"] == "done" and st["no_start"]["full-t06"]["overridden"] is True
    assert {x["item"]: int(x["deadline"]) for x in fq.records("train")} == {
        "full-t06": int(deadline - 3600), "full-p03": int(deadline - 3600)}


def test_a_held_item_is_tried_once_per_poll_while_its_gpu_takes_the_next_ready_item(fq):
    """check-resume exit 1 (the store is not built yet) holds the resumed item for a poll: on two GPUs it is tried
    once per poll, not once per free GPU, and the GPU starts the next ready training item meanwhile."""
    rid = "full-p03-20260927T120000Z"
    local_run(fq.root, rid, 10, config="full-p03", st={"total_steps": 20, "t_c": None, "pre_cooldown_done": False,
                                                       "early_stop": {"triggered": None}, "resume_resets": 0,
                                                       "end_reason": "schedule"})
    write_plan(fq, {"stores-ctc": {"status": "fresh"}, "full-p03": {"status": "resume", "run_dir": f"runs/{rid}"},
                    "full-p005": {"status": "fresh"}}, box="full")
    poll = 0.3
    q = fq.make("full", registry=box_only(fq.reg, "full", keep=["stores-ctc", "full-p03", "full-p005"]), poll_s=poll,
                env=dict(FAKE_CHECK_RESUME=json.dumps({"full-p03": 1})))
    assert q.run() == F.EXIT_STOP  # full-p03 held max_attempts times, then failed for good
    cr = fq.records("check-resume")
    assert len(cr) == 4 and len(fq.events("item_held")) == 4  # the 4th: held no more, failed
    gaps = [b["t0"] - a["t0"] for a, b in zip(cr, cr[1:])]
    assert all(g >= 0.8 * poll for g in gaps), gaps
    p005 = next(x for x in fq.records("train") if x["item"] == "full-p005")
    assert p005["t0"] < cr[1]["t0"]  # started on the GPU the held item left free, before its next try
    st = fq.st()
    assert st["items"]["full-p03"]["status"] == "failed" and "check-resume exit 1" in st["items"]["full-p03"]["why"]
    assert st["items"]["full-p005"]["status"] == "done"


# ============================================================================================ stalls and beats


def test_a_hung_trainer_is_killed_and_resumed_and_an_item_without_a_stall_check_overruns(fq):
    reg = set_item(copy.deepcopy(fq.reg), "p01", "full-p01", stall_min=0.02)  # 1.2 s without a beat
    reg["boxes"]["p01"]["items"].append({"name": "speed-slow", "kind": "speed", "system": "cohere",
                                          "speed_kind": "cohere", "stall_min": None, "max_hours": 0.0002})
    env = dict(FAKE_HANG=json.dumps({"full-p01": 12}), FAKE_ITEM_S=json.dumps({"speed-slow": 60}))
    assert fq.make("p01", registry=reg, env=env).run() == F.EXIT_OK
    st = fq.st()
    it = st["items"]["full-p01"]
    assert it["status"] == "done" and len(it["attempts"]) == 2 and it["stalls"] == 1
    assert it["attempts"][0]["stalled"]["limit_s"] == pytest.approx(1.2) and it["hb_max_gap_s"] >= 1.2
    tr = fq.records("train")
    assert len(tr) == 1 and tr[0]["resumed"] and tr[0]["resumed_from"] == 10  # the hung launch left no record
    assert tr[0]["argv"][1:3] == ["--resume", it["run_dir"]]
    stalled = fq.events("item_stalled")
    assert len(stalled) == 1 and stalled[0]["item"] == "full-p01" and stalled[0]["age_s"] >= 1.2
    # the overrun: killed after 2 x max_hours, a failed readout, the box complete
    assert st["items"]["speed-slow"]["status"] == "failed" and st["readouts_failed"] == {"speed-slow": "overrun"}
    assert fq.events("item_overrun")[0]["limit_s"] == pytest.approx(2 * 0.0002 * 3600)


def test_the_controller_heartbeat_is_beaten_every_poll_and_held_by_a_freeze(fq, monkeypatch):
    q = fq.make("p01")
    q.register()
    hb = fq.state / fullrun.TRAIN_HB
    q.monitor({})
    t1 = hb.stat().st_mtime
    time.sleep(0.05)
    q.monitor({})
    assert hb.stat().st_mtime > t1  # every poll (no 5 s rate limit on the controller's beat)
    q._freeze_until = time.time() + 30
    t2 = hb.stat().st_mtime
    time.sleep(0.05)
    q.monitor({})
    q.put_summary(force=True)  # a summary put inside the freeze: no beat
    q.drain_uploads()
    assert hb.stat().st_mtime == t2 and q.ctl_beat_path() is None
    q._freeze_until = 0.0
    q.monitor({})
    assert hb.stat().st_mtime > t2


def test_summary_puts_are_coalesced_but_never_lost(fq):
    up = FakeUploader()
    q = fq.make("p01", uploader=up, summary_min_s=0.4)
    q.register()
    q.put_summary()
    assert len(up.put) == 1
    q.put_summary()  # within summary_min_s: pending, not dropped
    assert len(up.put) == 1 and q._put_pending
    q.monitor({})
    assert len(up.put) == 1
    time.sleep(0.45)
    q.monitor({})  # the first poll past the mark puts it
    assert len(up.put) == 2 and not q._put_pending
    q.put_summary(force=True)  # item end, run-dir discovery, the end: never delayed
    assert len(up.put) == 3


# ================================================================================================ smoke A


def smoke_registry(reg: dict, *, sigstop: bool) -> dict:
    """box full-smoke of tiny_registry, sized for the fakes: the freeze fault 1 s on a watchdog of 0 s; without SIGSTOP
    (Windows) F1 is dropped and the wipe fires on the first attempt."""
    reg = box_only(reg, "full-smoke", watchdog={"orphan_s": 0, "action": "alert"})
    for f in reg["boxes"]["full-smoke"]["faults"]:
        if f["action"] == "freeze_controller_hb":
            f["seconds"] = 30.0  # longer than the rest of smoke-p005: its end falls inside, whatever the load
        if f["action"] == "wipe_run_dir" and not sigstop:
            f["min_attempt"] = 1
    if not sigstop:
        reg["boxes"]["full-smoke"]["faults"] = [f for f in reg["boxes"]["full-smoke"]["faults"]
                                                if f["action"] != "sigstop"]
    return reg


def run_smoke(fq, monkeypatch, *, sigstop: bool):
    """Smoke A on the fakes against DirHub runs and scratch repos (the queue's uploads through HubUploader and
    vast/finish.py), with a stand-in watchdog that appends an alert while train_hb is stale."""
    import finish
    import huggingface_hub

    runs_hub = DirHub.create(fq.tmp / "runs-hub", limit=100000, window_s=1.0)
    scratch_hub = DirHub.create(fq.tmp / "scratch-hub", limit=100000, window_s=1.0)
    runs_hub.commit({k: v for k, v in weights_remote("study-p01").items()}, writer="setup")
    dl, snap = downloads(runs_hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snap)
    monkeypatch.setattr(finish, "HUB_RETRY_WAITS", (0.05, 0.1))
    monkeypatch.setenv(fullrun.ENV_THREADS_PER_GPU, "8")
    up = Q.HubUploader("u/kitsune-runs")
    up._api = FakeApi(runs_hub, "full-smoke")
    train_configs(fq.root, **{n: {"early_stop": {"min_delta_abs": 1e9}} for n in ("smoke-t06", "smoke-p03")})
    fq.state.mkdir(parents=True, exist_ok=True)
    (fq.state / "deadline").write_text(f"{time.time() + 2 * 3600}\n")
    (fq.state / fullrun.GATE_FILE).write_text(json.dumps({"verdict": "pass"}))
    env = dict(FAKE_EPOCH_STEPS="500", FAKE_STEP_S="0.001", FAKE_FULL_EVERY="25",
               FAKE_EARLY=json.dumps({"smoke-t06": 0.7, "smoke-p03": 0.7}),
               FAKE_TIMED=json.dumps({"smoke-p01": 50}), FAKE_SCRATCH=str(scratch_hub.dir), FAKE_DEADLINE_S="1200",
               FAKE_STEP_VALUE=json.dumps({"smoke-t06": 1.5, "smoke-p03": 0.6, "smoke-p01": 0.4, "smoke-p005": 0.25}))
    q = fq.make("full-smoke", registry=smoke_registry(fq.reg, sigstop=sigstop), uploader=up, env=env,
                out_repo="u/kitsune-runs", scratch_repo="u/kitsune-scratch", machine_id="m1",
                runs_hub=F.Hub("u/kitsune-runs", api=FakeApi(runs_hub, "q")),
                scratch_hub=F.Hub("u/kitsune-scratch", api=FakeApi(scratch_hub, "q")),
                kill_grace_s=0.3)
    hb, alerts, seen, stop = fq.state / fullrun.TRAIN_HB, fq.state / fullrun.ALERTS_FILE, [], threading.Event()

    def watchdog():  # the alert-mode watchdog in miniature: an alert once train_hb is 0.3 s stale, re-armed after
        armed = True
        while not stop.is_set():
            try:
                m = hb.stat().st_mtime
            except OSError:
                m = None
            seen.append((time.time(), m))
            if m is not None and time.time() - m > 0.3 and armed:
                with open(alerts, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"wall": time.time(), "kind": "orphan_alert", "hb": str(hb),
                                        "age_s": time.time() - m, "limit_s": 0.3}) + "\n")
                armed = False
            elif m is not None and time.time() - m < 0.1:
                armed = True
            time.sleep(0.01)

    th = threading.Thread(target=watchdog, daemon=True)
    th.start()
    try:
        rc = q.run()
    finally:
        stop.set()
        th.join(5)
    return rc, q, runs_hub, scratch_hub, seen


def check_smoke(fq, rc, runs_hub, scratch_hub, seen):
    assert rc == F.EXIT_OK, fq.summary()["reason"]
    st = fq.st()
    v = json.loads((fq.state / fullrun.VERDICT_FILE).read_text(encoding="utf-8"))
    bad = {k: c for k, c in v["checks"].items() if c["pass"] is not True}
    assert v["overall"] == "pass" and not bad, bad
    assert sorted(v["checks"], key=int) == [str(i) for i in range(1, 12)]
    assert json.loads(runs_hub.path(fullrun.box_verdict_path("full-smoke")).read_text()) == v
    faults = {f["id"]: f for f in v["faults"]}
    assert faults["F2"]["outcome"] == "recovered" and faults["F3"]["outcome"] == "recovered"
    assert faults["F4"]["outcome"] == "recovered" and faults["F5"]["outcome"] == "recovered"
    # F2 kill: fired after a full state, not counted as a failure, resumed in the same run dir
    p03 = st["items"]["smoke-p03"]
    assert p03["attempts"][0]["fault"] == "F2" and p03["attempts"][1]["resume"] == p03["run_dir"]
    assert st["faults"]["F2"]["fired_at"]["step"] >= 130
    # F3 wipe: the run dir moved to runs/_wiped/, pulled from the Hub (scratch state), check-resume, then resumed
    p01 = st["items"]["smoke-p01"]
    assert (fq.root / "runs" / "_wiped" / Path(p01["run_dir"]).name).is_dir()
    hr = fq.events("hub_resume")[0]
    assert hr["item"] == "smoke-p01" and hr["source"] == "scratch" and hr["state"] == f"full_step_{hr['step']}"
    assert [e["rc"] for e in fq.events("check_resume")] == [0] and p01["check_resume"] == "ok"
    assert p01["attempts"][-1]["resume"] == p01["run_dir"] and p01["status"] == "done"
    assert len(fq.records("check-resume")) == 1
    # F4 deadline: KITSUNE_DEADLINE = its start + 600 s
    p005_rec = next(x for x in fq.records("train") if x["item"] == "smoke-p005")
    assert float(p005_rec["deadline"]) == pytest.approx(p005_rec["t0"] + 600, abs=5)
    # F5 freeze: train_hb unchanged from the fault until it ended (its window, or the run's end: end_faults), across
    # an item end and its summary put
    fired = st["faults"]["F5"]["fired_at"]["wall"]
    until = min(st["faults"]["F5"]["window_end"],
                next(e["wall"] for e in fq.events("fault_outcome") if e["id"] == "F5"))
    window = [m for t, m in seen if fired + 0.05 < t < until - 0.05]
    assert window and len(set(window)) == 1
    ends = [e for e in fq.events("item_end") if fired < e["wall"] < until]
    assert any(e["item"] == "smoke-p005" for e in ends), "smoke-p005 did not end inside the freeze"
    # smoke-nostart: skipped by the no-start rule; check 3's math on the fakes' step times
    assert st["items"]["smoke-nostart"]["status"] == "skipped" and "smoke-nostart" in st["no_start"]
    c3 = v["checks"]["3"]["evidence"]
    assert c3["sec_per_step"] == {"smoke-t06": 1.5, "smoke-p03": 0.6, "smoke-p01": 0.4, "smoke-p005": 0.25}
    assert c3["projected_h"]["smoke-t06"] == pytest.approx(1.5 * 73452 / 3600)
    assert c3["factor"]["smoke-p01"] == pytest.approx(13.56 / (0.4 * 110520 / 3600))
    assert c3["box2_h"] == pytest.approx(max(1.5 * 73452, 0.6 * 109608 + 0.25 * 110528) / 3600)
    # the scratch repo: one state per timed run, its pointer at the newest upload, the history squashed
    rid = Path(p01["run_dir"]).name
    assert len({p.split("/")[3] for p in scratch_hub.listing(f"runs/{rid}/checkpoints")}) == 1
    assert v["checks"]["7"]["evidence"]["scratch_commits"] <= 2
    return st, v


@pytest.mark.skipif(os.name != "posix", reason="SIGSTOP and process groups are posix only")
def test_smoke_a_with_every_fault_including_sigstop(fq, monkeypatch):
    reg = smoke_registry(fq.reg, sigstop=True)
    for it in reg["boxes"]["full-smoke"]["items"]:
        if it["name"].startswith("smoke-") and it["kind"] == "train":
            it["stall_min"] = 0.02
    rc, q, runs_hub, scratch_hub, seen = run_smoke(fq, monkeypatch, sigstop=True)
    st, v = check_smoke(fq, rc, runs_hub, scratch_hub, seen)
    assert st["faults"]["F1"]["outcome"] == "recovered" and st["items"]["smoke-p01"]["stalls"] == 1


def test_smoke_a_faults_and_verdict(fq, monkeypatch):
    """Smoke A end to end without SIGSTOP (Windows): F2 kill, F3 wipe_run_dir (first attempt), F4 deadline, F5 freeze;
    every built-in check passes."""
    rc, q, runs_hub, scratch_hub, seen = run_smoke(fq, monkeypatch, sigstop=False)
    check_smoke(fq, rc, runs_hub, scratch_hub, seen)


def test_a_fault_fires_only_after_a_full_state_and_is_missed_when_the_run_ends_first(fq):
    reg = box_only(fq.reg, "full-smoke", keep=["stores-ctc", "smoke-p03"])
    q = fq.make("full-smoke", registry=reg)
    q.register()
    it = q.item("smoke-p03")
    rd = fq.root / "runs" / "smoke-p03-20260927T000000Z"
    (rd / "metrics").mkdir(parents=True)
    with open(rd / "metrics" / "scalars.jsonl", "w", encoding="utf-8") as f:
        for s in range(1, 200):
            f.write(json.dumps({"step": s, "wall": time.time(), "tag": "sched/train_s", "value": float(s)}) + "\n")
    t0 = time.time()
    it.update(status="running", run_dir=Q.Queue._rel(q, rd), attempts=[dict(t0=t0, gpu="0", resume=None)])
    killed = []
    proc = SimpleNamespace(pid=999999, poll=lambda: None, kill=lambda: killed.append(1), terminate=lambda: None)
    q._killpg = lambda p, sig: False
    q._faults("smoke-p03", proc, time.time())
    assert q.state["faults"]["F2"]["fired_at"] is None and not killed  # step 199 >= 130, but no full state yet
    with open(rd / "events.jsonl", "a", encoding="utf-8") as f:  # an older attempt's checkpoint does not count
        f.write(json.dumps({"kind": "checkpoint", "ckpt": "full", "wall": t0 - 100}) + "\n")
    q._faults("smoke-p03", proc, time.time())
    assert not killed
    with open(rd / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "checkpoint", "ckpt": "full", "wall": t0 + 1}) + "\n")
    q._faults("smoke-p03", proc, time.time())
    assert killed and q.state["faults"]["F2"]["fired_at"]["step"] == 199 and it["attempts"][-1]["fault"] == "F2"
    # a run that ends before its fault fired: fault_missed
    shutil.rmtree(rd)
    q2 = fq.make("full-smoke", registry=box_only(fq.reg, "full-smoke", keep=["stores-ctc", "smoke-p03"]),
                 state_dir=fq.tmp / "state2", env=dict(FAKE_EPOCH_STEPS="100"))
    assert q2.run() == F.EXIT_OK
    assert q2.state["faults"]["F2"]["outcome"] == "missed"
    ev = [json.loads(x) for x in (fq.tmp / "state2" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["id"] for e in ev if e["kind"] == "fault_missed"] == ["F2"]


def test_a_smoke_b_box_reports_its_registry_specs_only(fq):
    q = fq.make("smoke-b", machine_id="m2")
    q.register()
    items = q.state["items"]

    def done(name, files: dict):
        out = f"runs/{name}-20260927T000000Z"
        items[name].update(status="done", out=out, attempts=[dict(t0=1.0, gpu="0", rc=0)])
        for rel, obj in files.items():
            p = fq.root / out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj), encoding="utf-8")

    done("selftest", {"selftest.json": {"ok": True}})
    done("cmp-int8-w8a8-study-p03", {"compare.json": {"same": False}})
    done("whisper-large-v3", {"whisper.json": {"sets": {"eval_jsut": {"cer_corpus": 0.07}}}})
    items["speed-study-p03"]["status"] = "not_needed"
    v = F.SmokeVerdict(q).build()
    assert sorted(v["checks"], key=int) == ["12", "14", "15", "16"]  # no built-ins, no spec-less numbers
    assert v["checks"]["12"]["pass"] is True and v["checks"]["14"]["pass"] is False
    assert v["checks"]["15"]["pass"] is True and v["checks"]["15"]["evidence"][0]["value"] == 0.07
    assert v["checks"]["16"]["pass"] is None  # a not_needed item: null, never false
    assert v["overall"] == "fail"
    # AND over the specs of one number: a second check-15 spec that fails fails the check
    q.items_spec["whisper-large-v3"]["verdict"].append({"check": "15", "json": "{out}/whisper.json",
                                                       "path": "sets.eval_jsut.cer_corpus", "max": 0.06})
    assert F.SmokeVerdict(q).build()["checks"]["15"]["pass"] is False
    items["cmp-int8-w8a8-study-p03"]["status"] = "failed"
    assert F.SmokeVerdict(q).build()["checks"]["14"]["pass"] is False


def test_check_8_takes_a_non_forced_run_on_the_epochs_clock_even_when_it_stopped_early_by_itself(fq):
    reg = box_only(fq.reg, "full-smoke", keep=["stores-ctc", "smoke-p03", "smoke-p01"])
    q = fq.make("full-smoke", registry=reg)
    q.register()

    def ran(name, min_delta_abs, clock="epochs"):
        rd = fq.root / "runs" / f"{name}-20260927T000000Z"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "config.json").write_text(json.dumps({"config": {"schedule": {"clock": clock},
                                                               "early_stop": {"min_delta_abs": min_delta_abs}}}))
        with open(rd / "events.jsonl", "w", encoding="utf-8") as f:
            for e in ({"kind": "early_stop", "action": "cooldown", "trigger": "patience"},
                      {"kind": "checkpoint", "ckpt": "full", "reason": "pre_cooldown", "name": "full_step_80"},
                      {"kind": "ckpt_upload_ok", "ckpt": "full", "name": "full_step_80"},
                      {"kind": "phase", "name": "cooldown"}):
                f.write(json.dumps(e) + "\n")
        q.item(name).update(status="done", run_dir=f"runs/{rd.name}", attempts=[dict(t0=1.0, gpu="0", rc=0)])

    ran("smoke-p03", 1e9)  # forced (check 10's items): never check 8's scheduled-cooldown item
    ran("smoke-p01", 0.0)  # not forced; its dev_ce stopped it early by itself, which check 8 does not hold against it
    ok, ev = F.SmokeVerdict(q).check8()
    assert ok and ev["scheduled_cooldown"] == ["smoke-p01"] and ev["forced"] == ["smoke-p03"]
    ran("smoke-p01", 0.0, clock="steps")  # the cooldown must come on the epochs clock
    ok, ev = F.SmokeVerdict(q).check8()
    assert not ok and ev["scheduled_cooldown"] == []


def test_only_if_new_machine_skips_a_re_time_on_the_same_machine(fq):
    summary = {"machine_id": "m1", "items": {}}
    up = FakeUploader(remote={fullrun.box_summary_path("full-smoke"): json.dumps(summary).encode()})
    q = fq.make("smoke-b", uploader=up, machine_id="m1")
    q.register()
    assert q.item("speed-study-p03")["status"] == "not_needed"
    assert q.state["not_needed"]["speed-study-p03"] == {"reason": "same_machine", "machine_id": "m1",
                                                       "box": "full-smoke"}
    assert fq.events("item_not_needed")[0]["machine_id"] == "m1"
    for machine, remote in (("m2", up.remote), ("m1", {})):  # another machine, or an unreadable summary: it runs
        shutil.rmtree(fq.state)
        q = fq.make("smoke-b", uploader=FakeUploader(remote=remote), machine_id=machine)
        q.register()
        assert q.item("speed-study-p03")["status"] == "pending"


def test_resource_peaks_from_a_fake_proc_and_the_cgroup(fq):
    proc, cg = fq.tmp / "proc", fq.tmp / "cg"
    for pid, pgrp, kb in ((1234, 1234, 1048576), (1240, 1234, 524288), (999, 999, 4 << 20)):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} (python x) S 1 {pgrp} {pgrp} 0 -1 4194560\n")
        (d / "status").write_text(f"Name:\tpython\nVmRSS:\t   {kb} kB\nThreads:\t4\n")
    (proc / "self").mkdir()
    cg.mkdir()
    (cg / "memory.current").write_text(f"{3 << 30}\n")
    q = fq.make("p01", proc_root=proc, cgroup=cg)
    q.register()
    q._resources({"0": {"name": "full-p01", "proc": SimpleNamespace(pid=1234)}})
    assert q.item("full-p01")["peak_rss_gb"] == pytest.approx(1.5)
    assert q.state["host_mem_peak_gb"] == pytest.approx(3.0)
    (cg / "memory.current").write_text(f"{1 << 30}\n")
    q._resources({})
    assert q.state["host_mem_peak_gb"] == pytest.approx(3.0)  # a peak


# ============================================================================================== resume-pull


RID = "full-p01-20260927T120000Z"


def hub_run(hub: DirHub, rid: str, *, fulls: dict[int, str] = (), step_export: int | None = None,
            complete: bool = False, st: dict | None = None):
    """A run in a DirHub runs repo: its logs, full states {step: reason} (with trainer.json), its export."""
    files = {f"runs/{rid}/config.json": json.dumps({"config": {"run_name": rid.rsplit("-", 1)[0],
                                                               "schedule": {"clock": "epochs", "epochs": 1}}}).encode(),
             f"runs/{rid}/events.jsonl": b'{"kind": "phase", "name": "setup"}\n',
             f"runs/{rid}/metrics/scalars.jsonl": b"",
             f"runs/{rid}/summary.json": json.dumps({"status": "complete" if complete else "running"}).encode()}
    for step, reason in dict(fulls).items():
        for f in ("model.pt", "optimizer.pt", "l2sp.pt", "trainer.pt"):
            files[f"runs/{rid}/checkpoints/full_step_{step}/{f}"] = f"{rid}:{step}:{f}".encode()
        files[f"runs/{rid}/checkpoints/full_step_{step}/trainer.json"] = json.dumps(
            {"format": 1, "step": step, "reason": reason, "st": st or {}}).encode()
    if step_export is not None:
        files[f"runs/{rid}/checkpoints/step_{step_export}/model.safetensors"] = b"export"
    hub.commit(files, writer="setup")


def scratch_state(hub: DirHub, rid: str, step: int, *, corrupt: bool = False, st: dict | None = None):
    import hashlib

    base = fullrun.scratch_state_dir(rid, step)
    files, meta = {}, {}
    for f in ("model.pt", "optimizer.pt", "l2sp.pt", "trainer.pt", "trainer.json"):
        data = json.dumps({"reason": "timed", "st": st or {}}).encode() if f == "trainer.json" else \
            f"scratch:{step}:{f}".encode()
        files[f"{base}/{f}"] = data
        meta[f] = {"size": len(data), "sha256": hashlib.sha256(data + (b"x" if corrupt and f == "l2sp.pt" else b""))
                   .hexdigest()}
    ptr = {"format": 1, "run_id": rid, "name": f"full_step_{step}", "step": step, "epoch": 0.5, "wall": 1.0,
           "time_utc": "2026-09-27T12:00:00+00:00", "kitsune_sha": "abc", "planner_fingerprint": "fp",
           "n_train_utts": 10, "selection_sha256": None, "micro_audio_s": 1600.0, "files": meta,
           "host": {"hostname": "h", "machine_id": "m1", "container_id": "c1"}}
    assert fullrun.pointer_problems(ptr) == []
    files[fullrun.scratch_pointer(rid)] = json.dumps(ptr).encode()
    hub.commit(files, writer="scratch")


def box_summary(hub: DirHub, box: str, items: dict):
    hub.commit({fullrun.box_summary_path(box): json.dumps({"format": 1, "kind": "full", "box": box, "sha": "abc",
                                                           "machine_id": "m1", "items": items}).encode()}, writer="q")


@pytest.fixture
def hubs(fq):
    runs = DirHub.create(fq.tmp / "runs-hub", limit=100000, window_s=1.0)
    scratch = DirHub.create(fq.tmp / "scratch-hub", limit=100000, window_s=1.0)
    new_root = fq.tmp / "newhost"
    (new_root / "runs").mkdir(parents=True)

    def pull(box="p01", reset=(), sets=None, root=new_root):
        return F.resume_pull(box, root, runs=F.Hub("u/runs", api=FakeApi(runs, "r")),
                             scratch=F.Hub("u/scratch", api=FakeApi(scratch, "s")),
                             registry=fullrun.load_registry(fq.reg, root=fq.root, check_files=False),
                             state_dir=fq.state, reset=list(reset), sets=dict(sets or {}), sha="abc")

    return SimpleNamespace(runs=runs, scratch=scratch, pull=pull, root=new_root)


def test_resume_pull_takes_the_newest_state_of_scratch_and_runs(fq, hubs):
    box_summary(hubs.runs, "p01", {"stores-ctc": {"status": "done"},
                                   "full-p01": {"status": "running", "run_dir": f"runs/{RID}"},
                                   "m4-full-p01": {"status": "pending"}})
    hub_run(hubs.runs, RID, fulls={100: "periodic"})
    scratch_state(hubs.scratch, RID, 150)
    plan = hubs.pull()
    e = plan["items"]["full-p01"]
    assert (e["status"], e["state"], e["step"], e["source"], e["run_dir"]) == (
        "resume", "full_step_150", 150, "scratch", f"runs/{RID}")
    assert plan["items"]["stores-ctc"]["status"] == "fresh" and plan["items"]["m4-full-p01"]["status"] == "fresh"
    local = hubs.root / "runs" / RID
    assert (local / "config.json").is_file() and (local / "events.jsonl").is_file()
    assert sorted(p.name for p in (local / "checkpoints").iterdir()) == ["full_step_150"]
    assert (local / "checkpoints" / "full_step_150" / "trainer.pt").read_bytes() == b"scratch:150:trainer.pt"
    assert json.loads((fq.state / fullrun.RESUME_PLAN).read_text(encoding="utf-8")) == plan
    assert plan["format"] == 1 and plan["box"] == "p01" and len(plan["summary_sha256"]) == 64
    # a newer runs-repo state wins; the scratch one is not downloaded
    shutil.rmtree(hubs.root / "runs" / RID)
    hub_run(hubs.runs, RID, fulls={100: "periodic", 200: "pre_cooldown"})
    e = hubs.pull()["items"]["full-p01"]
    assert (e["state"], e["source"]) == ("full_step_200", "runs")
    assert sorted(p.name for p in (hubs.root / "runs" / RID / "checkpoints").iterdir()) == ["full_step_200"]


def test_resume_pull_refuses_a_state_that_does_not_match_its_checksums(fq, hubs):
    box_summary(hubs.runs, "p01", {"full-p01": {"status": "running", "run_dir": f"runs/{RID}"}})
    hub_run(hubs.runs, RID, fulls={100: "periodic"})
    scratch_state(hubs.scratch, RID, 150, corrupt=True)
    with pytest.raises(F.ResumeRefused, match="sha256 differs"):
        hubs.pull()
    assert not (hubs.root / "runs" / RID / "checkpoints" / "full_step_150").exists()
    assert not (fq.state / fullrun.RESUME_PLAN).exists()


def test_resume_pull_keeps_a_hub_complete_run_done_and_pulls_its_export_for_its_readout(fq, hubs):
    result = {"run_id": RID, "steps": 300, "status": "complete"}
    box_summary(hubs.runs, "p01", {"full-p01": {"status": "done", "run_dir": f"runs/{RID}", "result": result},
                                   "m4-full-p01": {"status": "pending"}})
    hub_run(hubs.runs, RID, fulls={240: "pre_cooldown"}, step_export=300, complete=True)
    plan = hubs.pull()
    e = plan["items"]["full-p01"]
    assert (e["status"], e["result"], e["steps"], e["state"]) == ("done", result, 300, None)
    local = hubs.root / "runs" / RID / "checkpoints"
    assert sorted(p.name for p in local.iterdir()) == ["step_300"]  # the export, never a full state
    # its readout done too: no export pulled
    shutil.rmtree(hubs.root / "runs" / RID)
    box_summary(hubs.runs, "p01", {"full-p01": {"status": "done", "run_dir": f"runs/{RID}", "result": result},
                                   "m4-full-p01": {"status": "done", "out": f"runs/m4-{RID}", "result": {"m4": 0.1}}})
    plan = hubs.pull()
    assert plan["items"]["m4-full-p01"] == dict(plan["items"]["m4-full-p01"], status="done", out=f"runs/m4-{RID}",
                                                result={"m4": 0.1})
    assert not (hubs.root / "runs" / RID / "checkpoints").exists()
    # the box died between the trainer's end and the queue's summary put: the run's own Hub summary says done
    shutil.rmtree(hubs.root / "runs" / RID)
    box_summary(hubs.runs, "p01", {"full-p01": {"status": "running", "run_dir": f"runs/{RID}"}})
    hubs.runs.commit({f"runs/{RID}/summary.json": json.dumps({"status": "complete", "steps": 300, "epochs": 4.0,
                                                               "stopped_early": None, "end_reason": "schedule",
                                                               "resume_resets": 0}).encode()}, writer="t")
    e = hubs.pull()["items"]["full-p01"]
    assert e["status"] == "done" and e["result"] == dict(run_id=RID, steps=300, status="complete", epochs=4.0,
                                                         stopped_early=False, end_reason="schedule", resume_resets=0)
    assert sorted(p.name for p in (hubs.root / "runs" / RID / "checkpoints").iterdir()) == ["step_300"]


def test_resume_pull_resets_from_the_runs_repos_pre_cooldown_state(fq, hubs):
    result = {"run_id": RID, "steps": 300, "status": "complete"}
    box_summary(hubs.runs, "p01", {"full-p01": {"status": "done", "run_dir": f"runs/{RID}", "result": result},
                                   "m4-full-p01": {"status": "done", "out": f"runs/m4-{RID}"}})
    # the pulled events.jsonl has no checkpoint event: the choice comes from the states' trainer.json
    hub_run(hubs.runs, RID, fulls={200: "periodic", 240: "pre_cooldown", 300: "end"}, step_export=300,
            complete=True)
    scratch_state(hubs.scratch, RID, 260)
    plan = hubs.pull(reset=[RID], sets={RID: ["schedule.epochs=6"]})
    e = plan["items"]["full-p01"]
    assert (e["status"], e["state"], e["source"], e["reset"]) == ("resume", "full_step_240", "runs", True)
    assert e["sets"] == ["schedule.resume_reset=true", "schedule.epochs=6"]
    assert plan["items"]["m4-full-p01"]["status"] == "fresh"  # its readout runs again (into -r1)
    # no pre_cooldown state: refused
    hub2 = DirHub.create(fq.tmp / "runs-hub2", limit=1000, window_s=1.0)
    box_summary(hub2, "p01", {"full-p01": {"status": "running", "run_dir": f"runs/{RID}"}})
    hub_run(hub2, RID, fulls={200: "periodic"})
    with pytest.raises(F.ResumeRefused, match="no pre_cooldown"):
        F.resume_pull("p01", hubs.root, runs=F.Hub("u/runs", api=FakeApi(hub2, "r")), scratch=None,
                      registry=fullrun.load_registry(fq.reg, root=fq.root, check_files=False), state_dir=fq.state,
                      reset=[RID])


def test_resume_pull_refuses_a_set_only_run_past_its_cooldown_an_unknown_id_and_no_summary(fq, hubs):
    with pytest.raises(F.ResumeRefused, match="nothing to resume"):
        hubs.pull()
    box_summary(hubs.runs, "p01", {"full-p01": {"status": "running", "run_dir": f"runs/{RID}"}})
    hub_run(hubs.runs, RID, fulls={200: "pre_cooldown"}, st={"pre_cooldown_done": True})
    with pytest.raises(F.ResumeRefused, match="not a training run"):
        hubs.pull(reset=["full-p03-20260927T000000Z"])
    with pytest.raises(F.ResumeRefused, match="use --resume-reset"):
        hubs.pull(sets={RID: ["schedule.epochs=6"]})
    hub_run(hubs.runs, RID, fulls={200: "periodic"}, st={"pre_cooldown_done": False, "early_stop": {}})
    plan = hubs.pull(sets={RID: ["schedule.epochs=6"]})
    assert plan["items"]["full-p01"]["sets"] == ["schedule.resume_reset=true", "schedule.epochs=6"]


def test_resume_pull_cli_exit_codes(fq, hubs, monkeypatch):
    fq.state.mkdir(parents=True, exist_ok=True)
    tiny_registry(hubs.root, write_boxes=True)
    monkeypatch.setenv(fullrun.ENV_STATE, str(fq.state))
    monkeypatch.setenv(fullrun.ENV_OUT_REPO, "u/runs")
    monkeypatch.setenv(fullrun.ENV_SCRATCH_REPO, "u/scratch")
    apis = {"u/runs": FakeApi(hubs.runs, "r"), "u/scratch": FakeApi(hubs.scratch, "s")}
    real = F.Hub
    monkeypatch.setattr(F, "Hub", lambda repo, repo_type="model", api=None: real(repo, api=apis[repo]))
    argv = ["resume-pull", "--box", "p01", "--root", str(hubs.root)]
    assert F.main(argv) == F.EXIT_REFUSED  # no summary
    box_summary(hubs.runs, "p01", {"full-p01": {"status": "running", "run_dir": f"runs/{RID}"}})
    hub_run(hubs.runs, RID, fulls={100: "periodic"})
    monkeypatch.setenv(fullrun.ENV_RESUME_RESET, "full-p03-20260927T000000Z")
    assert F.main(argv) == F.EXIT_REFUSED  # an unknown id
    monkeypatch.setenv(fullrun.ENV_RESUME_RESET, "not a run id")
    assert F.main(argv) == F.EXIT_REFUSED
    monkeypatch.delenv(fullrun.ENV_RESUME_RESET)
    assert F.main(argv) == F.EXIT_OK
    assert json.loads((fq.state / fullrun.RESUME_PLAN).read_text())["items"]["full-p01"]["status"] == "resume"

    def boom(*a, **k):
        raise ConnectionError("the Hub is away")

    monkeypatch.setattr(F, "resume_pull", boom)
    assert F.main(argv) == F.EXIT_FAIL  # transient: bootstrap retries


# ================================================================================================= adoption


def local_run(root: Path, rid: str, step: int, st: dict | None = None, config: str = "full-p01"):
    """A pulled run dir as resume-pull leaves it: config.json, events.jsonl, one full state."""
    rd = root / "runs" / rid
    (rd / "metrics").mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps({"config": json.loads(
        (root / f"configs/full/{config}.json").read_text(encoding="utf-8"))}), encoding="utf-8")
    (rd / "events.jsonl").write_text(json.dumps({"kind": "phase", "name": "setup", "wall": 1.0}) + "\n")
    d = rd / "checkpoints" / f"full_step_{step}"
    d.mkdir(parents=True)
    for f in ("model.pt", "optimizer.pt", "l2sp.pt", "trainer.pt"):
        (d / f).write_bytes(b"x")
    (d / "trainer.json").write_text(json.dumps({"reason": "pre_cooldown", "st": st or {}}), encoding="utf-8")
    return rd


def write_plan(fq, items: dict, box: str = "p01"):
    fq.state.mkdir(parents=True, exist_ok=True)
    (fq.state / fullrun.RESUME_PLAN).write_text(json.dumps({"format": 1, "box": box, "created_utc": "x",
                                                            "summary_sha256": "0" * 64, "items": items}))


def test_adoption_resumes_the_pulled_run_after_check_resume_and_keeps_done_items(fq):
    local_run(fq.root, RID, 10, st={"total_steps": 20, "t_c": None, "pre_cooldown_done": False,
                                    "early_stop": {"triggered": None}, "resume_resets": 0,
                                    "end_reason": "schedule"})
    write_plan(fq, {"stores-ctc": {"status": "fresh"}, "full-p01": {"status": "resume", "run_dir": f"runs/{RID}"},
                    "m4-full-p01": {"status": "fresh"}})
    assert fq.make("p01").run() == F.EXIT_OK
    recs = fq.records()
    assert [x["mode"] for x in recs] == ["stores", "check-resume", "train", "readout"]
    assert recs[1]["argv"] == ["check-resume", "--run-dir", f"runs/{RID}"]
    assert recs[2]["argv"][:3] == ["train", "--resume", f"runs/{RID}"] and recs[2]["resumed_from"] == 10
    st = fq.st()
    assert st["resumed"]["items"] == {"stores-ctc": "pending", "full-p01": "interrupted", "m4-full-p01": "pending"}
    assert fq.events("queue_resume_adopted") and st["items"]["full-p01"]["check_resume"] == "ok"
    assert st["items"]["m4-full-p01"]["out"] == f"runs/m4-{RID}"
    # a done readout is never run again; a check-resume refusal fails the item without a retry
    shutil.rmtree(fq.state)
    shutil.rmtree(fq.tmp / "fake-log")
    write_plan(fq, {"stores-ctc": {"status": "fresh"}, "full-p01": {"status": "resume", "run_dir": f"runs/{RID}"},
                    "m4-full-p01": {"status": "done", "out": f"runs/m4-{RID}", "result": {"m4": 0.1}}})
    assert fq.make("p01", env=dict(FAKE_CHECK_RESUME=json.dumps({RID: 3}))).run() == F.EXIT_STOP
    st = fq.st()
    assert st["items"]["full-p01"]["status"] == "failed" and st["items"]["full-p01"]["check_resume"] == "refused"
    assert st["items"]["m4-full-p01"]["status"] == "done" and st["items"]["m4-full-p01"]["verified"] is True
    assert [x["mode"] for x in fq.records()] == ["stores", "check-resume"]


def test_adoption_happens_only_without_queue_json(fq):
    q = fq.make("p01")
    q.register()
    q.save()
    write_plan(fq, {"full-p01": {"status": "done", "run_dir": f"runs/{RID}", "result": {"steps": 20}}})
    q = fq.make("p01")
    q.register()
    assert q.state["resumed"] is None and q.item("full-p01")["status"] == "pending"
    (fq.state / "queue.json").unlink()
    q = fq.make("p01")
    q.register()
    assert q.state["resumed"] is not None and q.item("full-p01")["status"] == "done"
    # a queue.json of another kind or box refuses
    (fq.state / "queue.json").write_text(json.dumps({"box": "p01", "items": {}}))
    with pytest.raises(Q.QueueError, match="study queue's state"):
        fq.make("p01")


def test_a_reset_run_keeps_its_sets_until_the_state_after_the_reset_and_reads_out_into_r1(fq, monkeypatch):
    st0 = {"total_steps": 12, "t_c": 10, "pre_cooldown_done": True, "resume_resets": 0, "end_reason": "early_stop",
           "early_stop": {"triggered": {"trigger": "patience", "action": "cooldown", "at_step": 10}}}
    local_run(fq.root, RID, 10, st=st0)
    monkeypatch.setenv(fullrun.ENV_RESUME_RESET, RID)
    write_plan(fq, {"stores-ctc": {"status": "fresh"},
                    "full-p01": {"status": "resume", "run_dir": f"runs/{RID}", "reset": True,
                                 "sets": ["schedule.resume_reset=true"]},
                    "m4-full-p01": {"status": "fresh"}})
    # the first reset attempt dies right after its resume_reset event, before any full state
    env = dict(FAKE_RC=json.dumps({"full-p01": [1, 0]}), FAKE_RC_AT="11")
    assert fq.make("p01", env=env).run() == F.EXIT_OK
    tr = fq.records("train")
    assert [x["rc"] for x in tr] == [1, 0]
    assert all(x["argv"][-2:] == ["--set", "schedule.resume_reset=true"] for x in tr)  # on EVERY attempt until applied
    st = fq.st()
    it = st["items"]["full-p01"]
    assert it["sets_once"] == [] and it["resume_reset_applied"]["sets"] == ["schedule.resume_reset=true"]
    assert it["result"]["resume_resets"] == 1 and it["result"]["steps"] == 20
    assert st["items"]["m4-full-p01"]["out"] == f"runs/m4-{RID}-r1"
    ro = fq.records("readout")[0]["argv"]
    assert ro[ro.index("--out") + 1] == f"runs/m4-{RID}-r1"
    assert fq.events("resume_reset_applied")[0]["item"] == "full-p01"


# ===================================================================================================== CLI


class StubTrainer:
    def __init__(self, dev=True, resume=None):
        self.calls, self.dev, self.resume = [], dev, resume

    def load_config(self, path, sets):
        self.calls.append(("load_config", path, list(sets)))
        return {"family": "ctc", "path": path}

    def build_train_store(self, cfg, log):
        self.calls.append(("train",))
        return [1, 2, 3]

    def build_eval_store(self, cfg, log, frames=None):
        self.calls.append(("eval", frames))
        return [1, 2]

    def build_dev_store(self, cfg, log):
        self.calls.append(("dev",))
        return [1] if self.dev else None

    def resume_check(self, run_dir):
        self.calls.append(("resume_check", str(run_dir)))
        if isinstance(self.resume, Exception):
            raise self.resume
        return self.resume


def test_build_stores_builds_train_eval_and_dev_or_the_eval_store_alone(monkeypatch, capsys):
    D = StubTrainer()
    monkeypatch.setattr(Q, "_load_trainer", lambda: D)
    assert F.main(["build-stores", "--config", "configs/full/full-p03.json", "--set", "a.b=1"]) == 0
    assert D.calls == [("load_config", "configs/full/full-p03.json", ["a.b=1"]), ("train",), ("eval", None), ("dev",)]
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("stores_done ") and set(json.loads(out[-1][len("stores_done "):])) == {"peak_rss_gb",
                                                                                                       "wall_s"}
    D.calls.clear()
    assert F.main(["build-stores", "--config", "c.json", "--eval-only"]) == 0
    assert D.calls == [("load_config", "c.json", []), ("eval", None)]
    del StubTrainer.build_dev_store  # a trainer without the dev slice (before WP4a): train and eval only
    try:
        D.calls.clear()
        assert F.main(["build-stores", "--config", "c.json"]) == 0
        assert D.calls == [("load_config", "c.json", []), ("train",), ("eval", None)]
    finally:
        StubTrainer.build_dev_store = lambda self, cfg, log: self.calls.append(("dev",)) or [1]
    monkeypatch.setattr(Q, "_load_trainer", lambda: (_ for _ in ()).throw(RuntimeError("no torch")))
    assert F.main(["build-stores", "--config", "c.json"]) == F.EXIT_FAIL


@pytest.mark.parametrize("res, rc", [({"ok": True}, 0), ({"ok": False, "reason": "store not built"}, 1),
                                     ({"ok": False, "reason": "fingerprint differs"}, 3),
                                     (RuntimeError("torch"), 1)])
def test_check_resume_exit_codes(monkeypatch, res, rc):
    D = StubTrainer(resume=res)
    monkeypatch.setattr(Q, "_load_trainer", lambda: D)
    assert F.main(["check-resume", "--run-dir", "runs/x-20260927T000000Z"]) == rc
    assert D.calls == [("resume_check", str(Path("runs/x-20260927T000000Z")))]


def test_plan_lists_the_registry_items(fq, monkeypatch, capsys):
    tiny_registry(fq.tmp / "reg", write_boxes=True)
    monkeypatch.setenv(fullrun.ENV_REGISTRY, str(fq.tmp / "reg" / fullrun.BOXES_FILE))
    assert F.main(["plan", "--box", "p01"]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [r["item"] for r in rows] == ["stores-ctc", "full-p01", "m4-full-p01"]
    assert rows[2]["needs"] == ["full-p01"] and rows[1]["stall_min"] == 45
    monkeypatch.setenv(fullrun.ENV_REGISTRY, str(fq.tmp / "missing.json"))
    assert F.main(["plan", "--box", "p01"]) == F.EXIT_FAIL
    assert F.main(["run", "--box", "p01"]) == F.EXIT_FAIL  # a registry it cannot read: refused, no crash


def test_the_queue_never_imports_the_trainer_or_reads_the_study_box_plans(fq, monkeypatch):
    import subprocess

    code = ("import sys; import kitsune.full_queue as F; "
            "print(sorted(m for m in ('torch', 'numpy', 'huggingface_hub', 'pyarrow') if m in sys.modules))")
    out = subprocess.run([PY, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr

    def no_plan(*a, **k):
        raise AssertionError("a full box never reads prereg.rules()['boxes']")

    monkeypatch.setattr(Q, "box_plan", no_plan)
    q = fq.make("full")
    assert q.plan["shared_queue"] and q.plan["runs"] == ["full-t06", "full-p03", "full-p005"]
