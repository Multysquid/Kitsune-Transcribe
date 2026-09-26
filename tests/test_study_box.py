"""The size study's box path (STUDY.md 5.3-5.5, 6.1, 6.2; CONTRACT.md sections 2, 4, 6, 7):

- tools/make_study_configs.py: every config of every box from one base, the grids read from kitsune.prereg, each AED
  config valid for scripts/04_distill.py once the box fills its placeholders (and refused before), --check
- the trainer's calibrate block (scripts/04_distill.py): the box keys' validation, a tiny CPU calibration run that ends
  on its window or on the STOP file with its step-time table, and one that waits at its start barrier for GO
- kitsune/study_queue.py: the calibration statistics and windows, the /dev/shm fit; a CPU dry run of the whole Cohere
  box (stores once and alone, the calibration group released together on distinct GPUs and measured over the literal
  window, the probes with the edge extension, the numbers written and uploaded before the first study step, main runs
  then their branches on the same GPU, lean uploads), the halt paths (an LR edge after its extension, a calibration
  still loader-bound), a restart after a kill mid-wave, the smoke gate guard and a smoke failure without a second try,
  the GPU count and /dev/shm share, box B's plan (its wave calibrated together, the reference under the wave's load,
  the speed probes on trained weights; CTC items stubbed by the fake trainer until the CTC trainer is merged), the
  replicate's derived numbers, the shakedown; a relaunched box reusing its numbers from the Hub (and refusing ones
  written under other rules), the readout policy (a failed anchor or speed item: complete, recorded; the conditional
  replicate skipped with a note), and boxes A and B run concurrently against one rate-limited runs repo
  (tests/fake_runs_repo.py)
Every process the queue starts is tests/fake_study_trainer.py here (fake GPUs = distinct CUDA_VISIBLE_DEVICES values,
uploads recorded by a fake uploader); the real trainer runs in the calibrate-block test only. CPU only.
"""
import copy
import inspect
import json
import math
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

from fixtures import load_script  # noqa: E402
from kitsune import prereg  # noqa: E402
from kitsune import study_queue as Q  # noqa: E402

FAKE = ROOT / "tests" / "fake_study_trainer.py"
PY = sys.executable
BOXES_CONTRACT = {  # CONTRACT.md section 6, as kitsune.prereg.rules()["boxes"] carries it once D1 is merged
    "A": dict(runs=["study-t06", "study-t03", "study-t01", "study-t005"], probe_classes=["kept-t03", "scratch"],
              calibrate=["study-t06", "study-t03", "study-t01", "study-t005"], reference="study-t06",
              numbers_file="PREREG_numbers_A.json", numbers_from=None, extras=["anchor"]),
    "B": dict(runs=["study-p03", "study-p01", "study-p005", "study-bridge"],
              probe_classes=["lost", "kept-p03", "bridge"],
              calibrate=["study-p03", "study-p01", "study-p005", "study-bridge", "study-t06"], reference="study-t06",
              numbers_file="PREREG_numbers_B.json", numbers_from=None, extras=["speed"]),
    "replicate": dict(runs=["study-t01-s1235"], probe_classes=[], calibrate=[], reference=None,
                      numbers_file="PREREG_numbers_replicate.json", numbers_from="A", extras=[]),
}


def study_rules() -> dict:
    """kitsune.prereg.rules() with the box plans: its own once it has them (D1), else CONTRACT.md section 6's."""
    r = prereg.rules()
    r.setdefault("boxes", copy.deepcopy(BOXES_CONTRACT))
    return r


def make_configs():
    sys.path.insert(0, str(ROOT / "tools"))
    import make_study_configs as M

    return M


@pytest.fixture(scope="module")
def trainer():
    return load_script("04_distill")


# ================================================================================================= the configs


def test_the_generated_configs_are_the_committed_ones():
    """configs/study/ is what tools/make_study_configs.py writes from the rules today (a changed rule, e.g. a probe
    grid, shows up here until the configs are generated again and committed)."""
    M = make_configs()
    assert M.check() == []


def test_every_config_of_every_box_is_generated_with_the_rules_grids():
    M = make_configs()
    r = study_rules()
    names = set(M.all_configs(r))
    for run in r["runs"]:
        assert {run, f"{run}-half"} <= names
    for cls, p in r["lr_probes"]["classes"].items():
        assert {prereg.probe_run_name(cls, lr) for lr in p["grid"]} <= names
    for box in ("A", "B", "replicate", "shakedown"):
        assert {Path(c).stem for c in Q.box_configs(box, r)} <= names, box
    # the grids come from the rules: another grid, other probe configs
    r2 = copy.deepcopy(r)
    r2["lr_probes"]["classes"]["kept-t03"]["grid"] = [1e-4, 2e-4, 4e-4]
    assert "probe-kept-t03-4e-4" in M.all_configs(r2) and "probe-kept-t03-4e-4" not in names or \
        4e-4 in r["lr_probes"]["classes"]["kept-t03"]["grid"]


def fill(cfg_path: Path, m=9366, lr=2e-4) -> list[str]:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return [] if cfg["schedule"]["max_steps"] is not None else \
        [f"schedule.max_steps={m}", f"optim.lr={lr!r}", f"eval.mini.every_steps={round(0.025 * m)}"]


def test_every_aed_config_validates_once_filled_and_the_placeholders_are_refused(trainer):
    """Every generated Transcribe config loads with the trainer's load_config + validate (the runs and branches once
    the box has filled max_steps, LR and the mini cadence); an unfilled run config is refused. The Parakeet configs
    carry the CTC trainer's keys (family, loss.w_ctc): they are checked only once the trainer knows them (WP4b)."""
    ctc_known = "family" in trainer.DEFAULTS
    n = 0
    for p in sorted((ROOT / "configs" / "study").glob("*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        if cfg.get("family") == "ctc" and not ctc_known:
            continue
        loaded = trainer.load_config(str(p), fill(p))
        n += 1
        if p.stem.startswith("study-") and not p.stem.endswith("-half"):
            with pytest.raises(SystemExit, match="max_steps"):
                trainer.load_config(str(p), [])
            with pytest.raises(SystemExit, match="optim.lr"):
                trainer.load_config(str(p), ["schedule.max_steps=9366"])
            spec = prereg.rules()["runs"][p.stem]
            assert loaded["bn"]["mode"] == ("train" if spec["init_class"] == "scratch" else "frozen")
            assert loaded["loss"]["aux_ctc_weight"] == spec["aux_ctc"] and loaded["loss"]["l2sp_lambda"] == 0
            assert loaded["schedule"]["warmup_steps"] == spec["warmup_steps"]
            assert loaded["batch"]["micro_audio_s"] == spec["micro_audio_s"] and loaded["seed"] == spec["seed"]
            small = spec["params_total"] <= 110_000_000
            assert loaded["ckpt"]["full_at_fracs"] == ([0.4, 0.8] if small else [0.4])
            assert loaded["ckpt"]["upload_full_at"] == (["frac:0.8"] if small else [])
            assert loaded["eval"]["full_at_fracs"] == [0.2, 0.4, 0.6, 0.8] and loaded["eval"]["every_min"] is None
            assert loaded["early_stop"]["enabled"] is False and loaded["pull_parakeet"] is True
            assert loaded["hf"]["output_repo"] == "Multy123/kitsune-runs"
            assert loaded["selection_recipe"]["study"] == prereg.STUDY_SELECTION
    assert n >= 30


def test_branch_configs_differ_from_their_run_only_where_a_branch_may(trainer):
    """A T/2 branch's config equals its parent's but for BRANCH_FREE (the trainer refuses the branch otherwise)."""
    for p in sorted((ROOT / "configs" / "study").glob("study-*-half.json")):
        run = p.stem[: -len("-half")]
        a = json.loads((p.parent / f"{run}.json").read_text(encoding="utf-8"))
        b = json.loads(p.read_text(encoding="utf-8"))
        free = [k.rstrip(".") for k in trainer.BRANCH_FREE]
        assert {k for k in a if a[k] != b.get(k)} <= set(free) | {"branch"}
        assert b["branch"]["resume_frac"] == 0.4 and b["branch"]["end_frac"] == 0.5


def test_probe_and_calibration_configs(trainer):
    r = prereg.rules()
    for cls, p in r["lr_probes"]["classes"].items():
        cfg = json.loads((ROOT / "configs" / "study" / f"{prereg.probe_run_name(cls, p['grid'][0])}.json").read_text(
            encoding="utf-8"))
        assert cfg["lr_probe"]["enabled"] and cfg["schedule"]["max_steps"] == p["max_steps"]
        assert cfg["schedule"]["warmup_steps"] == p["warmup"]
        assert math.isclose(cfg["schedule"]["cooldown_frac"] * p["max_steps"], p["cooldown"])
        assert cfg["student"] == r["runs"][p["probed_on"]]["student"] and cfg["ckpt"]["upload_full_at"] == []
    cal = trainer.load_config(str(ROOT / "configs" / "study" / "calib-study-t06.json"), [])
    assert cal["calibrate"] == {"enabled": True, "window": list(prereg.CALIB_STEPS), "barrier": False}
    assert cal["batch"]["micro_audio_s"] == 600 and cal["smoke"]["enabled"] and cal["ckpt"]["full_after_smoke"] is False
    assert cal["lr_probe"]["enabled"]
    # the shakedown's smoke is the study run's own start: its warm-up, seed and smoke gate, at its class's lowest grid
    # LR; capped by the calibration's max_steps, ended by the box after the smoke. Every run of boxes A and B (both
    # families); the loss-trend check is on for the pruned runs (a flat start fails the shakedown) and off for the
    # scratch ones (a flat start is the queue's note)
    assert Q.shakedown_runs(r) == [*r["boxes"]["A"]["runs"], *r["boxes"]["B"]["runs"]]
    for run in Q.shakedown_runs(r):
        main = trainer.load_config(str(ROOT / "configs" / "study" / f"{run}.json"), fill(ROOT / "configs" / "study" /
                                                                                          f"{run}.json"))
        sm = trainer.load_config(str(ROOT / "configs" / "study" / f"shake-smoke-{run}.json"), [])
        assert main["smoke"]["require_loss_decrease"] is True
        assert sm["smoke"]["require_loss_decrease"] is (r["runs"][run]["init_class"] != "scratch")
        assert sm.get("family", "aed") == r["runs"][run]["family"]
        assert sm["schedule"]["warmup_steps"] == main["schedule"]["warmup_steps"] and sm["seed"] == main["seed"]
        assert sm["schedule"]["max_steps"] > sm["schedule"]["warmup_steps"] and sm["smoke"]["steps"] == 100
        grid = r["lr_probes"]["classes"][r["runs"][run]["lr_from"]]["grid"]
        assert sm["optim"]["lr"] == min(grid) and sm["calibrate"]["window"] == [50, 100]


# ========================================================================================= the trainer's block


def test_the_box_keys_are_validated(trainer):
    base = trainer.load_config(None, [])
    assert base["pull_parakeet"] is False and base["selection_recipe"]["study"] is None
    assert base["calibrate"] == {"enabled": False, "window": [50, 250], "barrier": False}
    for sets, match in ((["pull_parakeet=true"], "parakeet_root"),
                        (['selection_recipe.study={"f1a_max": 0.5}'], "selection_recipe.study"),
                        (["calibrate.window=[5, 5]"], "calibrate.window"),
                        (["calibrate.barrier=true"], "calibrate.barrier"),
                        (["calibrate.barrier=1"], "calibrate.barrier"),
                        (["calibrate.enabled=true"], "calibrate.enabled needs"),
                        (["calibrate.enabled=true", "lr_probe.enabled=true", "schedule.clock=steps",
                          "schedule.max_steps=40", "schedule.warmup_steps=1"], "ends before the window"),
                        (["optim.lr=null"], "optim.lr")):
        with pytest.raises(SystemExit, match=match):
            trainer.load_config(None, sets)
    ok = trainer.load_config(None, ["pull_parakeet=true", "parakeet_root=labels/full/parakeet_out",
                                    f"selection_recipe.study={json.dumps(prereg.STUDY_SELECTION)}"])
    assert ok["pull_parakeet"] is True


def test_calibration_run_ends_with_its_step_time_table(env, hub, trainer):
    """A tiny CPU calibration run: metrics only (no evals, no weights, no full state), a calibrate_result event and
    summary.json's calibrate over its window; one ended by the STOP file (the box ends a group that way) reports the
    steps it ran."""
    from test_study_trainer import events, one_run, write_config

    over = {"lr_probe": {"enabled": True}, "calibrate": {"enabled": True, "window": [3, 10]},
            "schedule": {"max_steps": 12, "warmup_steps": 2},
            "ckpt": {"full_local_every_min": None, "full_after_smoke": False}}
    assert trainer.main(["--config", write_config(env, "calib-a", over)]) == 0
    run = one_run(env["root"], "calib-a")
    res = next(e for e in events(run) if e["kind"] == "calibrate_result")
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "complete" and s["steps"] == 12 and s["calibrate"]["window"] == [3, 10]
    assert s["calibrate"]["steps_measured"] == 7 and s["calibrate"]["t_step_s"] > 0
    assert 0 <= s["calibrate"]["data_wait_frac"] < 1 and s["calibrate"]["micro_audio_s"] == 3
    assert res["t_step_s"] == s["calibrate"]["t_step_s"]
    assert not any(e["kind"] in ("eval_start", "lr_probe_result") for e in events(run))
    # no weights and no end state (a 12-step run reaches its cooldown, whose start saves the one full state; the box's
    # calibration runs end on STOP long before theirs)
    ck = run / "checkpoints"
    assert not list(ck.glob("step_*")) and not (ck / "full_step_12").exists()
    rows = Q.read_step_rows(run)
    assert Q.calibration_stats(rows, 3, 10)["t_step_s"] == s["calibrate"]["t_step_s"]

    # the STOP file ends it (the box stops a group once every run has its window)
    path = write_config(env, "calib-b", dict(over, schedule={"max_steps": 400, "warmup_steps": 2}))
    runs = env["root"] / "runs"

    def stopper():
        for _ in range(3000):
            for d in runs.glob("calib-b-2*"):
                if max(Q.read_step_rows(d), default=0) >= 15:
                    (d / "STOP").write_text("test\n")
                    return
            time.sleep(0.05)

    th = threading.Thread(target=stopper, daemon=True)
    th.start()
    assert trainer.main(["--config", path]) == 0
    th.join(5)
    s = json.loads((one_run(env["root"], "calib-b") / "summary.json").read_text(encoding="utf-8"))
    assert 15 <= s["steps"] < 400 and s["calibrate"]["steps_measured"] == 7

    # the start barrier (the box's calibration groups): READY once set up, no step before GO
    path = write_config(env, "calib-c", over)
    released = {}

    def releaser():
        for _ in range(6000):
            for d in runs.glob("calib-c-2*"):
                if (d / Q.READY_FILE).is_file():
                    time.sleep(0.5)  # the trainer waits meanwhile
                    released["wall"] = time.time()
                    (d / Q.GO_FILE).write_text("go\n")
                    return
            time.sleep(0.05)

    th = threading.Thread(target=releaser, daemon=True)
    th.start()
    assert trainer.main(["--config", path, "--set", "calibrate.barrier=true"]) == 0
    th.join(5)
    run = one_run(env["root"], "calib-c")
    bar = [e for e in events(run) if e["kind"] == "calibrate_barrier"]
    assert [e["state"] for e in bar] == ["ready", "released"] and bar[1]["by"] == "GO" and bar[1]["waited_s"] >= 0.4
    rows = Q.read_step_rows(run)
    assert min(rows) == 1 and rows[1]["wall"] >= released["wall"]
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["steps"] == 12 and s["calibrate"]["steps_measured"] == 7


def test_build_stores_builds_what_the_trainer_then_reuses(env, capsys):
    """The box's store step (study_queue build-stores) builds the train and eval stores of a config's data keys with
    the trainer's own code; a trainer on the same data finds them and builds nothing."""
    from test_study_trainer import write_config

    path = write_config(env, "stores-a", {"lr_probe": {"enabled": True}, "calibrate": {"enabled": True,
                                                                                       "window": [1, 2]},
                                          "schedule": {"max_steps": 3, "warmup_steps": 1}})
    cache = Path(env["base"]["cache_dir"])
    shutil.rmtree(cache, ignore_errors=True)
    assert Q.main(["build-stores", "--config", path]) == 0
    assert (cache / "train" / "stores.json").is_file() and (cache / "eval" / "stores.json").is_file()
    capsys.readouterr()
    assert Q.main(["build-stores", "--config", path]) == 0
    assert capsys.readouterr().out.count("stores: reusing") == 2


def test_the_cli_lists_a_boxs_students_and_plan(monkeypatch, capsys):
    monkeypatch.setattr(prereg, "rules", lambda sidecar=None, _r=prereg.rules: dict(
        _r(sidecar), boxes=_r(sidecar).get("boxes") or copy.deepcopy(BOXES_CONTRACT)))
    assert Q.main(["students", "--box", "A"]) == 0
    assert capsys.readouterr().out.split() == [prereg.RUNS[x]["student"] for x in BOXES_CONTRACT["A"]["runs"]]
    assert Q.main(["plan", "--box", "A"]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [r["phase"] for r in rows][:2] == ["stores", "calibrate"] and rows[-1] == {"phase": "extras",
                                                                                       "item": "anchor"}


from test_study_trainer import env, grad_enabled, hub  # noqa: E402,F401  (the tiny corpus + student fixtures)


# ================================================================================================ pure helpers


def rows_of(walls: list[float], waits: float = 0.1, dt: float = 1.0, start: int = 1) -> dict:
    return {start + i: dict(train_s=(start + i) * dt, data_wait_s=waits * dt, wall=w) for i, w in enumerate(walls)}


def test_calibration_stats_is_the_median_step_time_and_the_wait_share():
    rows = {s: dict(train_s=float(s * s) / 10, data_wait_s=0.01, wall=s) for s in range(1, 30)}
    st = Q.calibration_stats(rows, 5, 15)
    dts = sorted(((s * s) - (s - 1) * (s - 1)) / 10 for s in range(6, 16))
    assert st["steps_measured"] == 10 and st["window"] == [5, 15]
    assert st["t_step_s"] == pytest.approx((dts[4] + dts[5]) / 2)
    assert st["data_wait_frac"] == pytest.approx(0.1 / sum(dts))
    del rows[9]  # a skipped step: it and its successor drop out
    assert Q.calibration_stats(rows, 5, 15)["steps_measured"] == 8
    assert Q.calibration_stats({}, 5, 15)["t_step_s"] is None


def test_the_calibration_window_is_the_literal_one_and_a_skipped_step_is_replaced():
    a, n = 5, 10
    rows = rows_of([100.0 + i for i in range(30)])
    assert Q.window_end(rows, a, n) == 15 and Q.calibration_stats(rows, a, 15)["steps_measured"] == n
    del rows[9]  # a skipped step: it and its successor are not measured, two more steps take their place
    assert Q.window_end(rows, a, n) == 17 and Q.calibration_stats(rows, a, 17)["steps_measured"] == n
    assert Q.window_end(rows_of([1.0] * 12), a, n) is None and Q.window_end({}, a, n) is None


def test_shm_fit_cuts_as_the_trainers_shm_cap(trainer, monkeypatch, tmp_path):
    per = 600 * Q.SHM_BYTES_PER_AUDIO_S  # one micro-batch of 600 s
    assert Q.shm_fit(8, 4, 600, 32 * per) == (8, 4)
    assert Q.shm_fit(8, 4, 600, 16 * per) == (8, 2)  # the prefetch first, down to 2
    assert Q.shm_fit(8, 4, 600, 10 * per) == (5, 2)  # then the workers
    assert Q.shm_fit(8, 4, 600, 1.5 * per) == (1, 1)  # then the prefetch to 1; below that the trainer's own cut
    for k in (40, 16, 10, 3, 1.5):  # the trainer's shm_cap with the same budget (half its free space) agrees
        monkeypatch.setattr(trainer.shutil, "disk_usage",
                            lambda p, _f=2 * k * per: SimpleNamespace(free=_f, total=_f, used=0))
        n, p, _ = trainer.shm_cap(8, 4, 600.0, shm=str(tmp_path))
        assert (n, p) == Q.shm_fit(8, 4, 600, k * per), k


def test_step_tail_reads_only_what_was_appended(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    f = m / "scalars.jsonl"
    t = Q.StepTail(tmp_path)
    assert t.poll() == {}
    f.write_text(json.dumps({"step": 1, "wall": 1.0, "tag": "sched/train_s", "value": 1.0}) + "\n"
                 + '{"step": 2, "wall": 2.0, "tag": "sched/tr', encoding="utf-8")
    assert list(t.poll()) == [1]
    with open(f, "a", encoding="utf-8") as fh:
        fh.write('ain_s", "value": 2.0}\n')
    assert t.poll()[2]["train_s"] == 2.0 and Q.read_step_rows(tmp_path)[2]["wall"] == 2.0


# ================================================================================================= the queue


class FakeUploader:
    """Records every run dir synced (with the lean files finish.py would verify) and every file put; serves
    downloads from what was put (or from `remote`)."""

    def __init__(self, remote: dict | None = None, kill_on_sync: str | None = None):
        self.synced, self.put, self.remote, self.kill_on_sync = [], [], dict(remote or {}), kill_on_sync

    def sync_run(self, run_dir: Path) -> list[str]:
        import finish

        if self.kill_on_sync and run_dir.name.startswith(self.kill_on_sync):
            self.kill_on_sync = None
            raise KeyboardInterrupt("the queue dies mid-wave")
        self.synced.append((time.time(), run_dir.name, sorted(finish.expected_files(run_dir, False, lean=True))))
        return []

    def put_file(self, local, path_in_repo) -> list[str]:
        self.put.append((time.time(), path_in_repo))
        self.remote[path_in_repo] = Path(local).read_bytes()
        return []

    def exists(self, path_in_repo) -> bool:
        return path_in_repo in self.remote

    def download(self, path_in_repo, local_dir) -> Path:
        p = Path(local_dir) / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.remote[path_in_repo])
        return p

    def list_dir(self, path_in_repo) -> list[str]:
        pre = path_in_repo.rstrip("/") + "/"
        return sorted({k[len(pre):].split("/", 1)[0] for k in self.remote if k.startswith(pre)})

    def download_dir(self, prefix, local_dir) -> Path:
        for k in [k for k in self.remote if k.startswith(prefix.rstrip("/") + "/")]:
            self.download(k, local_dir)
        return Path(local_dir) / prefix


def write_numbers_stub(path, calibration, probes, lrs, max_steps, *, rules_path=None, host=None,
                       allow_pending=False):
    """What a per-box prereg.write_numbers does (CONTRACT.md 6; D1): max_steps as the rules derive them from the
    box's calibration, the LRs as its probes choose them, canonical JSON; returns the sha256."""
    import hashlib

    assert {k: int(v) for k, v in max_steps.items()} == prereg.max_steps(calibration)
    choices = prereg.choose_lr(probes)
    for run, lr in lrs.items():
        assert choices[prereg.RUNS[run]["lr_from"]]["lr"] == lr
    numbers = dict(calibration={r: {k: c[k] for k in prereg.CALIB_KEYS} for r, c in calibration.items()},
                   max_steps=dict(sorted(max_steps.items())), lr=dict(sorted(lrs.items())),
                   lr_probes={c: {prereg.lr_tag(lr): (v if math.isfinite(v) else None) for lr, v in res.items()}
                              for c, res in probes.items()}, rules_sha256="0" * 64, written_utc="now", host=host)
    data = (json.dumps(numbers, sort_keys=True, indent=1) + "\n").encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A box checkout in tmp_path: the generated configs, the committed PREREG, a runs/ dir; the queue's processes
    are the fake trainer, its GPUs "0".."3"."""
    shutil.copytree(ROOT / "configs" / "study", tmp_path / "configs" / "study")
    (tmp_path / "study").mkdir()
    shutil.copy(ROOT / "study" / "PREREG.json", tmp_path / "study" / "PREREG.json")
    if "box" not in inspect.signature(prereg.write_numbers).parameters:
        monkeypatch.setattr(prereg, "write_numbers", write_numbers_stub)
    log = tmp_path / "fake-log"

    def make(name="A", gpus=("0", "1", "2", "3"), uploader=None, env=None, **settings):
        # shm_bytes: a /dev/shm no loader outgrows, unless a test sets it (the host's own is not the test's)
        settings = dict(dict(shm_bytes=1 << 50, auto_workers=8, n_gpus=None), **settings)
        s = Q.Settings(root=tmp_path, state_dir=tmp_path / "state", out_repo=None, gpus=list(gpus), python=PY,
                       train_cmd=[PY, str(FAKE), "train"], stores_cmd=[PY, str(FAKE), "stores"],
                       anchor_cmd=[PY, str(FAKE), "anchor"], speed_cmd=[PY, str(FAKE), "speed"],
                       uploader=uploader if uploader is not None else FakeUploader(), rules=study_rules(),
                       rules_path=tmp_path / "study" / "PREREG.json", allow_pending=True, poll_s=0.02,
                       sync_offset_s=0.05, host="test-host", env=dict(FAKE_LOG=str(log), **(env or {})), **settings)
        return Q.Queue(name, s)

    def records():
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in log.glob("*.json")] if log.is_dir() else []
        return sorted(recs, key=lambda x: x["t0"])

    return make, records, tmp_path


def edge_then_inside(cls: str, r: dict) -> dict:
    """Probe objectives for which the grid's winner sits on its top edge, and the one extension point loses: the
    class is chosen after its extension, at the old top."""
    grid = sorted(r["lr_probes"]["classes"][cls]["grid"])
    obj = {prereg.probe_run_name(cls, lr): 3.0 - 0.1 * i for i, lr in enumerate(grid)}
    obj[prereg.probe_run_name(cls, grid[-1] * 2)] = 5.0
    return obj


def middle_wins(cls: str, r: dict) -> dict:
    grid = sorted(r["lr_probes"]["classes"][cls]["grid"])
    if len(grid) < 3:  # a 2-point grid has no inside point: the lower edge wins, /2 loses
        obj = {prereg.probe_run_name(cls, lr): 2.0 + i for i, lr in enumerate(grid)}
        obj[prereg.probe_run_name(cls, grid[0] / 2)] = 9.0
        return obj
    return {prereg.probe_run_name(cls, lr): (1.0 if i == len(grid) // 2 else 2.0) for i, lr in enumerate(grid)}


def box_a_env(r: dict) -> dict:
    return dict(FAKE_SETUP_S=json.dumps({"calib-study-t06": 0.4, "study-": 0.1}), FAKE_MAIN_S="0.3",
                FAKE_OBJ=json.dumps({**edge_then_inside("kept-t03", r), **middle_wins("scratch", r)}))


def test_box_a_dry_run(box):
    make, records, root = box
    r = study_rules()
    up = FakeUploader()
    q = make("A", uploader=up, env=box_a_env(r))
    assert q.run() == Q.EXIT_OK
    recs = records()
    by_item = {x["item"]: x for x in recs if x.get("item")}
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))

    # stores: once, alone, before anything else
    stores = [x for x in recs if x["mode"] == "stores"]
    assert len(stores) == 1 and all(stores[0]["t1"] <= x["t0"] for x in recs if x is not stores[0])

    # calibration: the four runs at once, one per GPU, released together (t06 set up last: the others waited for it),
    # each measured over the literal window (a, b] while every other run of the group trained
    cal = [by_item[f"calib-{run}"] for run in r["boxes"]["A"]["calibrate"]]
    assert sorted(x["gpu"] for x in cal) == ["0", "1", "2", "3"]
    assert max(x["t0"] for x in cal) < min(x["t1"] for x in cal)
    assert all("calibrate.barrier=true" in x["argv"] for x in cal)
    release = max(x["ready"] for x in cal)
    assert all(x["released"] >= release and x["first_step"] >= release for x in cal)
    assert by_item["calib-study-t06"]["ready"] == release  # the slow setup: the others waited at the barrier
    table = st["calibration"]
    a, b = prereg.CALIB_STEPS
    assert all(c["steps_measured"] == b - a and c["data_wait_frac"] < prereg.DATA_WAIT_MAX for c in table.values())
    assert all(c["window"] == [a, b] and c["released_wall"] <= min(x["first_step"] for x in cal) for c in table.values())
    rows = {run: Q.read_step_rows(root / c["run_dir"]) for run, c in table.items()}
    for run, c in table.items():  # every other run was stepping before this one's window and went on past it
        for other, orows in rows.items():
            assert orows[1]["wall"] <= rows[run][a]["wall"], (run, other)
            assert rows[run][b]["wall"] <= max(x["wall"] for x in orows.values()), (run, other)

    # probes: the grids, the kept-t03 extension (its grid winner sat on the top edge), both chosen
    kept = sorted(r["lr_probes"]["classes"]["kept-t03"]["grid"])
    ext = prereg.probe_run_name("kept-t03", kept[-1] * 2)
    assert ext in by_item and "--set" in by_item[ext]["argv"] and f"optim.lr={kept[-1] * 2!r}" in by_item[ext]["argv"]
    assert st["lr_choice"]["kept-t03"]["lr"] == pytest.approx(kept[-1])
    assert st["lr_choice"]["scratch"]["decision"] == "chosen"
    ext_start = by_item[ext]["t0"]
    assert all(by_item[prereg.probe_run_name("kept-t03", lr)]["t1"] <= ext_start for lr in kept)

    # the numbers: written, uploaded and logged before the first study step
    num = json.loads((root / "study" / "PREREG_numbers_A.json").read_text(encoding="utf-8"))
    per_box = {"box": "A"} if "box" in inspect.signature(prereg.max_steps).parameters else {}
    assert num["max_steps"] == prereg.max_steps({k: v for k, v in table.items()}, **per_box)
    put_t = next(t for t, p in up.put if p == "study/PREREG_numbers_A.json")
    mains = [by_item[run] for run in r["boxes"]["A"]["runs"]]
    assert put_t < min(x["t0"] for x in mains)
    ev = [json.loads(x) for x in (root / "state" / "events.jsonl").read_text().splitlines()]
    first_main = min(i for i, e in enumerate(ev) if e["kind"] == "item_start" and e["item"] in r["boxes"]["A"]["runs"])
    assert next(i for i, e in enumerate(ev) if e["kind"] == "prereg_numbers") < first_main

    # the wave: each main filled from the numbers, then its branch on the same GPU from the main's run dir
    for run in r["boxes"]["A"]["runs"]:
        m, h = by_item[run], by_item[f"{run}-half"]
        n = num["max_steps"][run]
        assert f"schedule.max_steps={n}" in m["argv"] and f"optim.lr={num['lr'][run]!r}" in m["argv"]
        assert f"eval.mini.every_steps={max(1, round(0.025 * n))}" in m["argv"]
        assert h["gpu"] == m["gpu"] and h["t0"] >= m["t1"] and h["parent"] == m["run_dir"]
        assert f"branch.parent=runs/{m['run_dir']}" in h["argv"]
    assert len({by_item[run]["gpu"] for run in r["boxes"]["A"]["runs"]}) == 4
    starts = sorted(by_item[run]["t0"] for run in r["boxes"]["A"]["runs"])
    assert all(b - a >= 0.04 for a, b in zip(starts, starts[1:]))  # the sync offsets

    # the anchor ran in a gap; every run dir went up (lean) and was verified
    assert by_item["anchor"]["mode"] == "anchor" and st["items"]["anchor"]["status"] == "done"
    synced = {name: files for _, name, files in up.synced}
    for run in r["boxes"]["A"]["runs"]:
        d = by_item[run]["run_dir"]
        files = synced[d]
        n = num["max_steps"][run]
        small = prereg.RUNS[run]["params_total"] <= 110_000_000
        want_full = {f"runs/{d}/checkpoints/full_step_{round(0.8 * n)}/model.pt"} if small else set()
        assert {f for f in files if "/full_step_" in f and f.endswith("model.pt")} == want_full
        assert f"runs/{d}/checkpoints/step_{round(0.4 * n)}/model.safetensors" in files
        assert f"runs/{d}/checkpoints/step_{n}/model.safetensors" in files
    assert all(it["verified"] for it in st["items"].values() if it["status"] == "done" and it["run_dir"]
               and it["kind"] != "stores")
    # the finished probes keep no local full state
    for name, it in st["items"].items():
        if it["kind"] == "probe":
            assert not list((root / it["run_dir"] / "checkpoints").glob("full_step_*"))
    summary = json.loads((root / "state" / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["numbers"]["sha256"]
    assert ("study/box-A/queue_summary.json" in up.remote)
    # a finished box does nothing again
    assert make("A", uploader=up).run() == Q.EXIT_OK and len(records()) == len(recs)


def test_an_lr_edge_after_its_extension_halts_the_box(box):
    make, records, root = box
    r = study_rules()
    grid = sorted(r["lr_probes"]["classes"]["kept-t03"]["grid"])
    obj = {prereg.probe_run_name("kept-t03", lr): 3.0 - 0.1 * i for i, lr in enumerate(grid)}
    obj[prereg.probe_run_name("kept-t03", grid[-1] * 2)] = 1.0  # the extension wins: an edge again
    env = dict(FAKE_OBJ=json.dumps({**obj, **middle_wins("scratch", r)}))
    q = make("A", env=env)
    assert q.run() == Q.EXIT_HALT
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert st["final"]["status"] == "halted" and "kept-t03" in st["final"]["reason"]
    assert st["numbers"] is None and not (root / "study" / "PREREG_numbers_A.json").exists()
    assert not any(x.get("item") in r["boxes"]["A"]["runs"] for x in records())
    assert make("A").run() == Q.EXIT_HALT  # a halted box stays halted


def test_a_loader_bound_calibration_runs_again_with_12_workers_then_halts(box):
    make, records, root = box
    r = study_rules()
    env = dict(FAKE_WAIT=json.dumps({"calib-study-t01": 0.2}), FAKE_OBJ=json.dumps(
        {**middle_wins("kept-t03", r), **middle_wins("scratch", r)}))
    assert make("A", env=env).run() == Q.EXIT_OK
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert st["num_workers"] == 12 and "calib-study-t01-w12" in st["items"]
    by_item = {x["item"]: x for x in records() if x.get("item")}
    assert "perf.num_workers=12" in by_item["study-t06"]["argv"] and "perf.num_workers=12" in by_item["probe-" + \
        "scratch-" + prereg.lr_tag(r["lr_probes"]["classes"]["scratch"]["grid"][0])]["argv"]


def test_still_loader_bound_halts(box):
    make, records, root = box
    env = dict(FAKE_WAIT=json.dumps({"calib-study-t01": 0.2, "calib-study-t01.w12": 0.2}))
    assert make("A", env=env).run() == Q.EXIT_HALT
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert "loader-bound" in st["final"]["reason"] and st["calibration"] is None


def test_a_restart_after_a_kill_mid_wave_resumes(box):
    """The queue dies mid-wave (here: while the first finished main's upload runs, the others still training): its
    running trainers are stopped, and a new queue process skips what is done, uploads what was not verified yet,
    resumes every interrupted main from its local full state, runs the branches from their parents, and ends
    complete. A main that crashes once resumes from its own full state within the same queue."""
    make, records, root = box
    r = study_rules()
    env = dict(box_a_env(r), FAKE_SETUP_S=json.dumps({"study-": 0.05}),
               FAKE_MAIN_S=json.dumps({"study-t005": 0.05, "study-": 2.0}), FAKE_CRASH=json.dumps({"study-t01": 0.5}))
    with pytest.raises(KeyboardInterrupt):
        make("A", uploader=FakeUploader(kill_on_sync="study-t005-2"), env=env).run()
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert st["items"]["study-t005"]["status"] == "done" and st["items"]["study-t005"]["verified"] is None
    interrupted = [run for run in ("study-t06", "study-t03", "study-t01") if st["items"][run]["status"] == "interrupted"]
    assert interrupted
    up = FakeUploader(remote={"study/PREREG_numbers_A.json": (root / "study" / "PREREG_numbers_A.json").read_bytes()})
    n_before = len(records())
    assert make("A", uploader=up, env=env).run() == Q.EXIT_OK
    later = records()[n_before:]
    by_item = {x["item"]: x for x in later if x.get("item")}
    assert "study-t005" not in by_item and "calib-study-t06" not in by_item  # done work is not redone
    for run in interrupted:
        assert by_item[run]["resumed"] is True and "--resume" in by_item[run]["argv"]
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert all(st["items"][f"{run}-half"]["status"] == "done" for run in r["boxes"]["A"]["runs"])
    assert any(n.startswith(st["items"]["study-t005"]["run_dir"].split("/")[-1]) for _, n, _ in up.synced)


def test_a_crashed_main_resumes_from_its_full_state(box):
    make, records, root = box
    r = study_rules()
    env = dict(box_a_env(r), FAKE_CRASH=json.dumps({"study-t01": 0.5}))
    assert make("A", env=env).run() == Q.EXIT_OK
    t01 = [x for x in records() if x.get("item") == "study-t01"]
    assert [x["rc"] for x in t01] == [1, 0] and t01[1]["resumed"] and t01[1]["resumed_from"] > 0
    assert t01[0]["run_dir"] == t01[1]["run_dir"]


def test_a_failed_calibration_group_runs_again_whole_and_a_diverged_probe_loses(box):
    """A calibration run that crashes stops its group, which runs again as a whole (the failed try is superseded,
    not a failure of the box); a probe whose trainer stops on non-finite gradients scores non-finite and loses."""
    make, records, root = box
    r = study_rules()
    grid = sorted(r["lr_probes"]["classes"]["scratch"]["grid"])
    obj = {**middle_wins("kept-t03", r), **middle_wins("scratch", r), prereg.probe_run_name("scratch", grid[-1]): "nan"}
    env = dict(FAKE_CALIB_CRASH=json.dumps({"calib-study-t03": 60}), FAKE_OBJ=json.dumps(obj))
    assert make("A", env=env).run() == Q.EXIT_OK
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert st["items"]["calib-study-t03"]["status"] == "superseded"
    assert st["items"]["calib-study-t03-try2"]["status"] == "done"
    assert st["calibration"]["study-t03"]["run_dir"].startswith("runs/calib-study-t03-try2-")
    first = [x for x in records() if x.get("item", "").startswith("calib-") and "-try" not in x["item"]]
    assert all(x["t1"] <= min(y["t0"] for y in records() if "-try2" in y.get("item", "")) for x in first)
    assert st["probes"]["scratch"][prereg.lr_tag(grid[-1])] is None
    assert st["lr_choice"]["scratch"]["decision"] == "chosen"


def box_a_on_the_hub(r: dict) -> dict:
    """The runs repo after box A: its queue summary (each run's run dir) and each run's exported weights."""
    remote, items = {}, {}
    for run in r["boxes"]["A"]["runs"]:
        d = f"runs/{run}-20261001T000000Z"
        items[run] = dict(kind="main", status="done", run_dir=d, verified=True)
        for step in (3748, 9370):
            remote[f"{d}/checkpoints/step_{step}/model.safetensors"] = f"{run}@{step}".encode()
        remote[f"{d}/checkpoints/full_step_3748/model.pt"] = b"state"
    remote["study/box-A/queue_summary.json"] = json.dumps(dict(box="A", status="complete", items=items)).encode()
    return remote


def test_box_b_plan_and_dry_run(box):
    """Box B: five calibrated runs on four GPUs - its wave's four together first, then the reference study-t06 under
    the load of three of them -, probes of three classes, the wave, the speed probes one model at a time at the end on
    trained weights: box B's own from its run dirs, box A's from the runs repo, the replicate's (not trained yet)
    skipped with a note. The CTC runs are the fake trainer's until the CTC trainer is merged (WP4b)."""
    make, records, root = box
    r = study_rules()
    items = Q.plan_items("B", r)
    groups = [x for x in items if x["phase"] == "calibrate"]
    cal, wave = r["boxes"]["B"]["calibrate"], r["boxes"]["B"]["runs"]
    assert len(cal) == 5 and groups[0]["runs"] == wave and groups[0]["load"] == []
    assert groups[1]["runs"] == ["study-t06"] and groups[1]["load"] == wave[:3]
    assert {x["cls"] for x in items if x["phase"] == "probe"} == {"lost", "kept-p03", "bridge"}
    assert [x["item"] for x in items if x["phase"] == "extras"] == ["speed"]
    # the box's own students only (the speed probes time trained weights, not the init dirs)
    assert set(Q.box_students("B", r)) == {r["runs"][x]["student"] for x in [*wave, "study-t06"]}
    assert Q.box_ctc_students("B", r) == [r["runs"][x]["student"] for x in ("study-p03", "study-p01", "study-p005")]
    obj = {**middle_wins("lost", r), **middle_wins("kept-p03", r), **middle_wins("bridge", r)}
    up = FakeUploader(remote=box_a_on_the_hub(r))
    # HF_TOKEN: the vast account env's, which the Cohere teacher's speed item reads the gated repo with (a test value)
    assert make("B", uploader=up, env=dict(FAKE_OBJ=json.dumps(obj), HF_TOKEN="hf_test_value")).run() == Q.EXIT_OK
    recs = records()
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert set(st["calibration"]) == set(r["boxes"]["B"]["calibrate"])
    first = [next(x for x in recs if x.get("item") == f"calib-{run}") for run in wave]
    assert max(x["t0"] for x in first) < min(x["t1"] for x in first)  # the wave's four at once
    load = [x for x in recs if x.get("item", "").startswith("calib-") and "-load" in x["item"]]
    # measured under the load of three others; box B's own name for it (box A calibrates study-t06 at the same time)
    last = next(x for x in recs if x.get("item") == "calib-study-t06-boxB")
    assert not any(x.get("item") == "calib-study-t06" for x in recs)
    assert st["calibration"]["study-t06"]["run_dir"].startswith("runs/calib-study-t06-boxB-")
    assert len(load) == 3 and all(x["t0"] < last["t1"] and x["t1"] >= last["t1"] - 1 for x in load)
    t06 = Q.read_step_rows(root / st["calibration"]["study-t06"]["run_dir"])
    a, b = prereg.CALIB_STEPS
    assert all(x["first_step"] <= t06[a]["wall"] and x["t1"] >= t06[b]["wall"] for x in load)
    speed = [x for x in recs if x["mode"] == "speed"]
    timed = [x for x in r["runs"] if x != prereg.REPLICATE]
    assert len(speed) == len(timed) + len(Q.TEACHERS) and all(a["t1"] <= b["t0"] for a, b in zip(speed, speed[1:]))
    got = json.loads((root / "runs" / "speed-B" / "speed.json").read_text(encoding="utf-8"))["systems"]
    assert set(got) == set(timed) | {"cohere", "parakeet-ctc", "parakeet-tdt"}
    assert got["study-p03"]["kind"] == "ctc" and got["study-t06"]["kind"] == "aed" and got["cohere"]["model"] is None
    assert got["cohere"]["hf_token"] is True  # the box's token reaches the teacher's speed item
    for run in wave:  # box B's own: the final weights in its run dir
        d = st["items"][run]["run_dir"]
        assert got[run]["model"] == f"{d}/checkpoints/step_{st['numbers']['max_steps'][run]}"
    for run in r["boxes"]["A"]["runs"]:  # box A's: its final weights from the runs repo
        m = Path(got[run]["model"])
        assert m.name == "step_9370" and (m / "model.safetensors").read_bytes() == f"{run}@9370".encode()
    # the replicate is conditional (CONTRACT.md 8): not trained, so skipped with a note - not a failed readout
    assert st["items"][f"speed-{prereg.REPLICATE}"]["status"] == "skipped"
    assert "conditional" in st["items"][f"speed-{prereg.REPLICATE}"]["source"]["why"]
    assert all("--require-idle" in x["argv"] for x in speed)
    last_train = max(x["t1"] for x in recs if x["mode"] != "speed")
    assert all(x["t0"] >= last_train for x in speed)
    num = json.loads((root / "study" / "PREREG_numbers_B.json").read_text(encoding="utf-8"))
    assert set(num["max_steps"]) >= set(r["boxes"]["B"]["runs"]) and "study-t06" in num["calibration"]
    summary = json.loads((root / "state" / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["reason"] is None and summary["readouts_failed"] == {}


def test_replicate_takes_box_as_numbers(box):
    make, records, root = box
    a_numbers = {"box": "A", "rules_sha256": prereg.rules_sha256(root / "study" / "PREREG.json"),
                 "max_steps": {"study-t01": 27400, "study-t06": 9370}, "lr": {"study-t01": 1e-3, "study-t06": 2e-4},
                 "calibration": {"study-t01": {"t_step_s": 0.4, "micro_audio_s": 800.0, "data_wait_frac": 0.01,
                                               "steps_measured": 200}}}
    # box A ran study-t01 without the loss-trend check (smoke_gate_guard): the replicate follows it
    gate = {"study-t01": {"loss_decreasing": False, "source": "runs/calib-study-t01-20261001T000000Z"}}
    up = FakeUploader(remote={"study/PREREG_numbers_A.json": json.dumps(a_numbers).encode(),
                              "study/box-A/queue_summary.json": json.dumps({"smoke_gate_off": gate}).encode()})
    assert make("replicate", gpus=("0",), uploader=up).run() == Q.EXIT_OK
    num = json.loads((root / "study" / "PREREG_numbers_replicate.json").read_text(encoding="utf-8"))
    assert num["max_steps"] == {"study-t01-s1235": 27400} and num["lr"] == {"study-t01-s1235": 1e-3}
    assert num["numbers_from"]["box"] == "A" and len(num["numbers_from"]["sha256"]) == 64
    by_item = {x["item"]: x for x in records() if x.get("item")}
    m = by_item["study-t01-s1235"]
    assert "schedule.max_steps=27400" in m["argv"] and "batch.micro_audio_s=800.0" in m["argv"]
    assert "smoke.require_loss_decrease=false" in m["argv"]
    assert "smoke.require_loss_decrease=false" in by_item["study-t01-s1235-half"]["argv"]
    assert by_item["study-t01-s1235-half"]["gpu"] == "0" and "study/PREREG_numbers_replicate.json" in up.remote
    assert not any(x["item"].startswith(("calib-", "probe-")) for x in by_item.values() if x.get("item"))


def earlier_box_a(root: Path, rules_sha: str | None = None) -> dict:
    """The runs repo after an earlier box A that wrote and uploaded its numbers (then died): its numbers file, under
    the committed rules unless rules_sha, and the queue summary it put up with them (the loader-bound retry's 12
    workers, study-t01 run without the loss-trend check)."""
    runs = ["study-t06", "study-t03", "study-t01", "study-t005"]
    numbers = {"box": "A", "rules_sha256": rules_sha or prereg.rules_sha256(root / "study" / "PREREG.json"),
               "max_steps": dict(zip(runs, [9370, 18520, 27400, 44010])),
               "lr": dict(zip(runs, [2e-4, 2e-4, 1e-3, 1e-3])),
               "calibration": {x: {"t_step_s": 0.4, "micro_audio_s": 777.0, "data_wait_frac": 0.01,
                                   "steps_measured": 200} for x in runs},
               "lr_probes": {}, "written_utc": "2026-10-01T00:00:00Z", "host": "earlier-host"}
    gate = {"study-t01": {"loss_decreasing": False, "source": "runs/calib-study-t01-20261001T000000Z"}}
    summary = {"box": "A", "status": "running", "num_workers": 12, "smoke_gate_off": gate}
    return {"study/PREREG_numbers_A.json": (json.dumps(numbers, sort_keys=True, indent=1) + "\n").encode(),
            "study/box-A/queue_summary.json": json.dumps(summary).encode()}


def test_a_relaunched_box_reuses_its_numbers_and_never_measures_again(box):
    """A relaunched box A (a fresh queue.json) finds its numbers file on the Hub, written under the committed rules: it
    takes that file as it is - no calibration, no probe, no second write or upload of it - logs its sha256 before the
    first study step, and runs its wave with the file's max_steps, LR and micro-batch, and the earlier box's loader
    retry and smoke gate decision from its queue summary."""
    make, records, root = box
    r = study_rules()
    up = FakeUploader(remote=earlier_box_a(root))
    before = up.remote["study/PREREG_numbers_A.json"]
    assert make("A", uploader=up).run() == Q.EXIT_OK
    items = [x["item"] for x in records() if x.get("item")]
    assert not any(i.startswith(("calib-", "probe-")) for i in items)
    assert set(r["boxes"]["A"]["runs"]) <= set(items)
    assert (root / "study" / "PREREG_numbers_A.json").read_bytes() == before == up.remote["study/PREREG_numbers_A.json"]
    assert not any(p == "study/PREREG_numbers_A.json" for _, p in up.put)  # never overwritten
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert st["numbers"]["reused"] and st["numbers"]["sha256"] == prereg.file_sha256(root / "study" /
                                                                                       "PREREG_numbers_A.json")
    assert st["calibration"] is None and st["lr_choice"] is None and st["num_workers"] == 12
    by_item = {x["item"]: x for x in records() if x.get("item")}
    t01 = by_item["study-t01"]["argv"]
    assert "schedule.max_steps=27400" in t01 and "optim.lr=0.001" in t01 and "batch.micro_audio_s=777.0" in t01
    assert "perf.num_workers=12" in t01 and "smoke.require_loss_decrease=false" in t01
    assert "smoke.require_loss_decrease=false" not in by_item["study-t06"]["argv"]
    ev = [json.loads(x) for x in (root / "state" / "events.jsonl").read_text().splitlines()]
    first_main = min(i for i, e in enumerate(ev) if e["kind"] == "item_start" and e["item"] in r["boxes"]["A"]["runs"])
    rec = next(i for i, e in enumerate(ev) if e["kind"] == "prereg_numbers")
    assert rec < first_main and ev[rec]["reused"] and ev[rec]["sha256"] == st["numbers"]["sha256"]


def test_a_relaunch_under_other_rules_refuses_and_a_foreign_file_halts(box):
    """Numbers on the Hub written under other rules than the committed PREREG (or not a numbers file at all) halt the
    relaunched box before it calibrates, probes or trains anything; the file stays as it was."""
    make, records, root = box
    for remote in (earlier_box_a(root, rules_sha="f" * 64), {"study/PREREG_numbers_A.json": b"{}"}):
        shutil.rmtree(root / "state", ignore_errors=True)
        up = FakeUploader(remote=remote)
        before = up.remote["study/PREREG_numbers_A.json"]
        n = len(records())
        assert make("A", uploader=up).run() == Q.EXIT_HALT
        st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
        assert "rules" in st["final"]["reason"] and st["numbers"] is None
        assert all(x["mode"] == "stores" for x in records()[n:])  # nothing measured, nothing trained
        assert up.remote["study/PREREG_numbers_A.json"] == before
        assert not (root / "study" / "PREREG_numbers_A.json").exists()


def test_a_failed_anchor_or_speed_readout_ends_the_box_complete_with_the_failure_recorded(box):
    make, records, root = box
    r = study_rules()
    up = FakeUploader(remote=earlier_box_a(root))
    assert make("A", uploader=up, env=dict(FAKE_FAIL=json.dumps({"anchor": 1}))).run() == Q.EXIT_OK
    summary = json.loads((root / "state" / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and "anchor" in summary["reason"]
    assert set(summary["readouts_failed"]) == {"anchor"}
    assert all(summary["items"][x]["status"] == "done" for x in r["boxes"]["A"]["runs"])
    # box B: the Cohere teacher's speed item without the box's HF_TOKEN, and a student's probe exiting 1
    shutil.rmtree(root / "state")
    obj = {**middle_wins("lost", r), **middle_wins("kept-p03", r), **middle_wins("bridge", r)}
    env = dict(FAKE_OBJ=json.dumps(obj), FAKE_FAIL=json.dumps({"speed-study-p01": 1}))
    os_token = os.environ.pop("HF_TOKEN", None)
    try:
        assert make("B", uploader=FakeUploader(remote=box_a_on_the_hub(r)), env=env).run() == Q.EXIT_OK
    finally:
        if os_token is not None:
            os.environ["HF_TOKEN"] = os_token
    summary = json.loads((root / "state" / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and set(summary["readouts_failed"]) == {"speed-cohere", "speed-study-p01"}
    assert "HF_TOKEN" in summary["readouts_failed"]["speed-cohere"]
    assert summary["items"][f"speed-{prereg.REPLICATE}"]["status"] == "skipped"


def test_shakedown_runs_every_check_on_one_gpu(box):
    make, records, root = box
    up = FakeUploader()
    assert make("shakedown", gpus=("0",), uploader=up).run() == Q.EXIT_OK
    recs = [x for x in records() if x["mode"] == "train"]
    items = [x["item"] for x in recs]
    plan = Q.shakedown_plan(study_rules())
    assert plan["runs"] == ["study-t06", "study-t03", "study-t01", "study-t005",
                            "study-p03", "study-p01", "study-p005", "study-bridge"]
    fam = ["shake-resume", "shake-parent", "shake-parent-half", "shake-eval"]
    assert plan["items"] == [f"shake-smoke-{run}" for run in plan["runs"]] + fam + [
        "shake-resume-ctc", "shake-parent-ctc", "shake-parent-ctc-half", "shake-eval-ctc"]
    resumes = ("shake-resume", "shake-resume-ctc")
    assert [i for i in items if i not in resumes] == [i for i in plan["items"] if i not in resumes]
    for name in resumes:  # one AED and one CTC crash and resume
        res = [x for x in recs if x["item"] == name]
        assert [x["rc"] for x in res] == [1, 0] and res[1]["resumed"] and res[1]["resumed_from"] == 40
    for parent in ("shake-parent", "shake-parent-ctc"):  # one AED and one CTC branch
        half = next(x for x in recs if x["item"] == f"{parent}-half")
        assert half["parent"] == next(x for x in recs if x["item"] == parent)["run_dir"]
    assert all(x["gpu"] == "0" for x in recs) and len(up.synced) == len(plan["items"])
    assert not (root / "study" / "PREREG_numbers_A.json").exists()
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    for run in plan["runs"]:  # each smoke ran its checks at its study warm-up, then the box stopped it
        s = json.loads((root / st["items"][f"shake-smoke-{run}"]["run_dir"] / "summary.json").read_text())
        assert 100 <= s["steps"] < 20000


def test_a_shakedown_smoke_whose_loss_does_not_fall_fails_once(box):
    """A smoke that fails the study run's gate (here: no falling loss) fails the shakedown, without a second try (it
    would start the same way); the other checks go on."""
    make, records, root = box
    flat = {"shake-smoke-study-p01": True, "shake-smoke-study-t01": True, "shake-smoke-study-bridge": True}
    assert make("shakedown", gpus=("0",), env=dict(FAKE_FLAT=json.dumps(flat))).run() == Q.EXIT_FAIL
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    it = st["items"]["shake-smoke-study-p01"]  # a pruned run (warm-up 1,000): a flat start is a failure
    assert it["status"] == "failed" and len(it["attempts"]) == 1
    assert all(x["status"] == "done" for n, x in st["items"].items() if n != "shake-smoke-study-p01")
    summary = json.loads((root / "state" / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and "shake-smoke-study-p01" in summary["reason"]
    # the scratch runs' flat starts are notes, not failures
    assert set(summary["shake_notes"]) == {"study-t01", "study-bridge"}
    assert summary["shake_notes"]["study-t01"]["loss_last"] >= summary["shake_notes"]["study-t01"]["loss_first"]


def test_the_smoke_gate_guard_and_a_smoke_failure_without_a_second_try(box):
    """A main whose calibration run (same student, seed, data and warm-up) showed no falling loss runs without the
    loss-trend check, its branch alike; a main whose smoke fails anyway fails at once (no identical second start), its
    branch is skipped, the box ends failed and the others finish."""
    make, records, root = box
    r = study_rules()
    env = dict(box_a_env(r), FAKE_FLAT=json.dumps({"calib-study-t005": True, "study-t005": True, "study-t01": True}))
    assert make("A", env=env).run() == Q.EXIT_FAIL
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert set(st["smoke_gate_off"]) == {"study-t005"}
    ev = st["smoke_gate_off"]["study-t005"]
    assert ev["loss_decreasing"] is False and ev["source"] == st["calibration"]["study-t005"]["run_dir"]
    by_item = {x["item"]: x for x in records() if x.get("item")}
    for name in ("study-t005", "study-t005-half"):
        assert "smoke.require_loss_decrease=false" in by_item[name]["argv"] and by_item[name]["rc"] == 0
    assert not any("smoke.require_loss_decrease" in a for a in by_item["study-t06"]["argv"])
    t01 = [x for x in records() if x.get("item") == "study-t01"]
    assert len(t01) == 1 and t01[0]["rc"] == 1 and st["items"]["study-t01"]["status"] == "failed"
    assert st["items"]["study-t01-half"]["status"] == "skipped"
    assert all(st["items"][x]["status"] == "done" for x in ("study-t06", "study-t06-half", "study-t03-half"))
    summary = json.loads((root / "state" / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["smoke_gate_off"] == st["smoke_gate_off"] and "study-t01" in summary["reason"]


def test_the_gpu_count_is_the_rented_one(box, monkeypatch):
    make, records, root = box
    with pytest.raises(Q.QueueError, match="one wave"):
        make("A", gpus=("0",))
    with pytest.raises(Q.QueueError, match="KITSUNE_N_GPUS"):
        make("A", n_gpus=8)
    monkeypatch.setattr(Q, "detect_gpus", lambda: [])
    with pytest.raises(Q.QueueError, match="no GPU found"):
        make("A", gpus=())
    assert make("shakedown", gpus=("0",), n_gpus=1).gpus == ["0"]
    monkeypatch.undo()
    monkeypatch.delenv("KITSUNE_GPUS", raising=False)

    def no_smi(*a, **k):
        raise OSError("nvidia-smi: not found")

    monkeypatch.setattr(Q.subprocess, "run", no_smi)
    assert Q.detect_gpus() == []
    monkeypatch.setenv("KITSUNE_GPUS", "2,3")
    assert Q.detect_gpus() == ["2", "3"]


def test_every_trainer_of_a_run_gets_the_same_share_of_dev_shm(box):
    """A /dev/shm the box's four trainers could overflow together: each gets half of it over the GPUs, its loader cut
    to fit (the prefetch, then the workers) - the same for the run's calibration, probes, main and branch."""
    make, records, root = box
    r = study_rules()
    q = make("A", env=box_a_env(r), shm_bytes=16_000_000_000)
    assert q.run() == Q.EXIT_OK
    by_item = {x["item"]: x for x in records() if x.get("item")}

    def perf(item):
        return sorted(a for a in by_item[item]["argv"] if a.startswith("perf."))

    assert perf("calib-study-t06") == [] and perf("study-t06") == []  # 8 x 4 x 600 s fits its 2 GB
    assert perf("calib-study-t03") == ["perf.num_workers=8", "perf.prefetch=2"]
    want = ["perf.num_workers=7", "perf.prefetch=2"]  # micro 1600: the prefetch to 2 is not enough
    scratch = [prereg.probe_run_name("scratch", lr) for lr in r["lr_probes"]["classes"]["scratch"]["grid"]]
    for item in ["calib-study-t01", "study-t01", "study-t01-half", *scratch]:
        assert perf(item) == want, item
    st = json.loads((root / "state" / "queue.json").read_text(encoding="utf-8"))
    assert st["shm"] == dict(total_bytes=16_000_000_000, gpus=4, per_trainer_bytes=2_000_000_000.0)


def test_box_plans_come_from_the_rules():
    r = study_rules()
    assert Q.box_plan("A", r)["reference"] == "study-t06"
    bad = copy.deepcopy(r)
    del bad["boxes"]
    with pytest.raises(Q.QueueError, match="no boxes.A"):
        Q.box_plan("A", bad)
    bad = copy.deepcopy(r)
    bad["boxes"]["A"]["calibrate"] = ["study-t03"]
    with pytest.raises(Q.QueueError, match="reference run"):
        Q.box_plan("A", bad)
    assert Q.box_students("A", r) == [r["runs"][x]["student"] for x in r["boxes"]["A"]["runs"]]
    assert Q.box_extra_dirs("A", r) == [] and Q.box_extra_dirs("B", r) == [Q.PARAKEET_DIR]
    assert Q.study_extra_gb("A", r) > Q.study_extra_gb("replicate", r) > 0


# ================================================================================ boxes A and B at the same time


def test_boxes_a_and_b_at_the_same_time_share_one_runs_repo_without_a_clash(tmp_path, monkeypatch):
    """CONTRACT.md 8: boxes A and B run at the same time against ONE runs repo. Here both queues run concurrently (two
    threads, each box its own checkout and state dir: two machines), their trainers are fake_study_trainer processes
    whose mains commit their log syncs into the same on-disk runs repo, and the queues' uploads go through the real
    HubUploader and vast/finish.py (sync, verify, put_file) over it. The repo answers more than LIMIT commits in any
    WINDOW_S with a 429. Proven: both boxes end complete and verified; no path of the repo is committed by both boxes
    (run ids - box B calibrates study-t06 as calib-study-t06-boxB -, queue summaries, numbers files are each box's
    own); the 8 mains' syncs and the queues' uploads were rate-limited (429) and backed off: every queue upload (the
    lean run dirs, the numbers files, the queue summaries) got through and verified, and a trainer sync whose retries
    all met the limit (kitsune/runlog.py drops it) is caught up by its next sync and by the queue's verified upload of
    the finished run dir; each box's mains start SYNC_OFFSET_S apart."""
    import finish
    import huggingface_hub
    from fake_runs_repo import DirHub, FakeApi, downloads

    LIMIT, WINDOW_S = 4, 0.25
    hub = DirHub.create(tmp_path / "hub", limit=LIMIT, window_s=WINDOW_S)
    monkeypatch.setattr(finish, "HUB_RETRY_WAITS", (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2))
    dl, snap = downloads(hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snap)
    r = study_rules()
    obj = {**middle_wins("kept-t03", r), **middle_wins("scratch", r), **middle_wins("lost", r),
           **middle_wins("kept-p03", r), **middle_wins("bridge", r)}
    queues, logs = {}, {}
    for name in ("A", "B"):
        root = tmp_path / f"box{name}"
        shutil.copytree(ROOT / "configs" / "study", root / "configs" / "study")
        (root / "study").mkdir(parents=True)
        shutil.copy(ROOT / "study" / "PREREG.json", root / "study" / "PREREG.json")
        up = Q.HubUploader("u/kitsune-runs")
        up._api = FakeApi(hub, name)
        logs[name] = tmp_path / f"fake-log-{name}"
        env = dict(FAKE_LOG=str(logs[name]), FAKE_OBJ=json.dumps(obj), FAKE_HUB=str(hub.dir), FAKE_BOX=name,
                   FAKE_SYNCS="3", FAKE_MAIN_S="1.5", FAKE_SYNC_WAITS=json.dumps([0.1, 0.3, 0.9, 2.0]),
                   HF_TOKEN="hf_test_value")
        s = Q.Settings(root=root, state_dir=root / "state", out_repo=None, gpus=["0", "1", "2", "3"], python=PY,
                       train_cmd=[PY, str(FAKE), "train"], stores_cmd=[PY, str(FAKE), "stores"],
                       anchor_cmd=[PY, str(FAKE), "anchor"], speed_cmd=[PY, str(FAKE), "speed"], uploader=up,
                       rules=r, rules_path=root / "study" / "PREREG.json", allow_pending=True, poll_s=0.02,
                       sync_offset_s=0.05, host=f"host-{name}", env=env, shm_bytes=1 << 50, auto_workers=8,
                       n_gpus=4)
        queues[name] = Q.Queue(name, s)
    rcs, errors = {}, []

    def run(name):
        try:
            rcs[name] = queues[name].run()
        except BaseException as e:  # noqa: BLE001  reported below
            errors.append((name, repr(e)))

    threads = [threading.Thread(target=run, args=(n,)) for n in queues]
    for t in threads:
        t.start()
    for t in threads:
        t.join(600)
    assert not errors and rcs == {"A": Q.EXIT_OK, "B": Q.EXIT_OK}, (errors, rcs)

    states = {n: json.loads((tmp_path / f"box{n}" / "state" / "queue.json").read_text(encoding="utf-8"))
              for n in queues}
    run_ids = {n: {Path(it["run_dir"]).name for it in st["items"].values() if it["run_dir"]}
               for n, st in states.items()}
    assert not run_ids["A"] & run_ids["B"]
    assert any(x.startswith("calib-study-t06-boxB-") for x in run_ids["B"])
    assert not any(x.startswith("calib-study-t06-box") for x in run_ids["A"])
    for n, st in states.items():  # every run dir verified on the Hub
        assert all(it["verified"] for it in st["items"].values()
                   if it["status"] == "done" and it["run_dir"] and it["kind"] not in ("stores", "speed")), n

    def owner(path: str) -> str | None:
        for n in queues:
            if path.startswith((f"study/box-{n}/", f"study/{r['boxes'][n]['numbers_file']}")) or \
                    path.split("/")[:2][-1] in run_ids[n]:
                return n
        return None

    commits = hub.commits()
    accepted = [c for c in commits if c["accepted"]]
    for c in accepted:  # each box commits only its own paths, and every path has one owner
        assert {owner(p) for p in c["paths"]} == {c["box"]}, c
    paths = {n: {p for c in accepted if c["box"] == n for p in c["paths"]} for n in queues}
    assert paths["A"] and paths["B"] and not paths["A"] & paths["B"]
    for n in queues:
        assert f"study/box-{n}/queue_summary.json" in paths[n]
        num = json.loads(hub.path(f"study/{r['boxes'][n]['numbers_file']}").read_text(encoding="utf-8"))
        assert num["box"] == n and num["rules_sha256"] == prereg.rules_sha256(ROOT / "study" / "PREREG.json")

    # the commit rate: never more than LIMIT in a window; the limit was hit (429s), and the back-offs got through
    walls = sorted(c["wall"] for c in accepted)
    assert all(walls[i + LIMIT] - walls[i] >= WINDOW_S for i in range(len(walls) - LIMIT))
    assert any(not c["accepted"] for c in commits)
    mains = {n: r["boxes"][n]["runs"] for n in queues}
    trainer_ev = [e for n in queues for m in mains[n]
                  for e in Q.read_events(tmp_path / f"box{n}" / states[n]["items"][m]["run_dir"])]
    assert any(e["kind"] == "sync_ok" and e["attempt"] > 0 for e in trainer_ev)  # a trainer sync through its back-off
    writers = {c["writer"] for c in accepted if not c["writer"].startswith("queue-")}
    main_dirs = {Path(states[n]["items"][m]["run_dir"]).name for n in queues for m in mains[n]}
    assert main_dirs <= writers and len(main_dirs) == 8  # the 8 mains synced into the one repo
    for n in queues:
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in logs[n].glob("*.json")]
        by_item = {x["item"]: x for x in recs if x.get("item")}
        for m in mains[n]:
            rd = tmp_path / f"box{n}" / states[n]["items"][m]["run_dir"]
            ev = Q.read_events(rd)
            assert sum(e["kind"] == "sync" for e in ev) == 3 and len(states[n]["items"][m]["attempts"]) == 1
            # whatever its in-loop syncs met, the run dir in the repo is the finished one (the queue's upload)
            remote = hub.path(f"{states[n]['items'][m]['run_dir']}/events.jsonl")
            assert remote.read_bytes() == (rd / "events.jsonl").read_bytes()
        starts = sorted(by_item[m]["t0"] for m in mains[n])
        assert all(b - a >= 0.04 for a, b in zip(starts, starts[1:]))
