"""tools/make_full_configs.py and configs/full/: the full-data runs' generated trainer and data configs, the hand-written
box registry configs/full/boxes.json (with its chained box p01-chain, contract addendum E: the entry, its derived spec
and stage views, and launch's look-only resolution of it), and the plan record they take their measured numbers from
(build contract 7).

CPU only; the trainer is imported (04_distill.load_config) but runs nothing. The real selections are read only with
KITSUNE_FULL_SELECTION_DIR pointing at a local labels/full/selections/full_study (the last test).
"""
import copy
import json
import math
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, load_script  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import make_full_configs as M  # noqa: E402
import make_study_configs as S  # noqa: E402

from kitsune import extent, fullrun, prereg  # noqa: E402

FULL = ROOT / "configs" / "full"
STUDENTS = ("t06", "p03", "p01", "p005")
GENERATED = ([f"full-{x}" for x in STUDENTS] + [f"smoke-{x}" for x in STUDENTS]
             + ["data-p01", "data-full", "data-smoke", "data-smoke-b"])
QUANT_FORMATS = ("fp16", "int8-w8a16", "int8-w8a8", "nvfp4-w4a16", "nvfp4-w4a4", "mxfp4-w4a4", "fp8-w8a8")
WHISPER_KEYS = ("whisper-large-v3", "whisper-large-v3-turbo", "kotoba-whisper-v2.0", "whisper-small")
STUDY_WEIGHTS = {"study-t06": ("study-t06-20260926T174027Z", 9370), "study-t03": ("study-t03-20260926T174113Z", 27510),
                 "study-t01": ("study-t01-20260926T174159Z", 26090),
                 "study-t005": ("study-t005-20260926T174244Z", 33380),
                 "study-p03": ("study-p03-20260926T172336Z", 25120), "study-p01": ("study-p01-20260926T172421Z", 34620),
                 "study-p005": ("study-p005-20260926T172507Z", 47690)}
PARAKEET = "models/parakeet-tdt_ctc-0.6b-ja-hf"
# what tools/full_plan.py measured on the built selections (the full-selection report of 2026-09-30)
FULL_SHA, SMOKE_SHA = ("e9a0695ac45b6b325e6c5c3434ed382e46c63782ece0f375a5f6bbeb310ea509",
                       "93dd422d873c966a1cbd8aacadcd879ad0bab13481394f554ac9e143a0eaf5b2")
MEASURED_T = {"t06": 71946, "p03": 105861, "p01": 107910, "p005": 107910}


def cfg(name: str) -> dict:
    return json.loads((FULL / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan():
    return M.load_plan()


@pytest.fixture(scope="module")
def reg():
    return fullrun.load_registry(FULL / "boxes.json")


@pytest.fixture(scope="module")
def trainer():
    return load_script("04_distill")


def items(reg, box) -> dict:
    return {it["name"]: it for it in fullrun.box_items(box, reg)}


def flat(d: dict, where: str = ""):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from flat(v, f"{where}{k}.")
        else:
            yield f"{where}{k}", v


def repo_copy(tmp_path) -> Path:
    """configs/full (with the plan record) and the study config smoke-b's store item reads, in a checkout of their
    own; returns its configs/full."""
    out = tmp_path / "configs" / "full"
    shutil.copytree(FULL, out)
    (tmp_path / "configs" / "study").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs" / "study" / "study-t06.json", tmp_path / "configs" / "study" / "study-t06.json")
    return out


# ================================================================================================ the generator


def test_the_committed_files_are_the_generators():
    """configs/full/ is what tools/make_full_configs.py writes, and boxes.json is a valid registry (--check, a frozen
    check of every package from WP2c on)."""
    assert M.check() == []
    assert sorted(M.all_configs()) == sorted(GENERATED)
    assert sorted(p.stem for p in FULL.glob("*.json")) == sorted([*GENERATED, "boxes"])


def test_check_reports_differences_stale_files_and_registry_problems(tmp_path, capsys):
    out = repo_copy(tmp_path)
    assert M.main(["--check", "--out-dir", str(out)]) == 0 and "up to date" in capsys.readouterr().out
    # a CRLF checkout compares equal (parsed JSON)
    p = out / "full-p03.json"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert M.check(out) == []
    c = json.loads(p.read_text(encoding="utf-8"))
    c["optim"]["lr"] = 3e-4
    p.write_text(json.dumps(c), encoding="utf-8")
    (out / "old.json").write_text("{}", encoding="utf-8")
    (out / "data-smoke.json").unlink()
    reg = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    reg["boxes"]["p01"]["items"][1]["family"] = "aed"
    (out / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    probs = M.check(out)
    assert "full-p03.json: differs from the generator's" in probs
    assert "data-smoke.json: missing" in probs
    assert "old.json: not made by the generator" in probs
    assert any(x.startswith("boxes.json: ") and "family 'aed', but configs/full/full-p01.json trains 'ctc'" in x
               for x in probs), probs
    assert any("data_config: configs/full/data-smoke.json does not exist" in x for x in probs), probs
    assert M.main(["--check", "--out-dir", str(out)]) == 1 and "problem(s)" in capsys.readouterr().out
    (out / "boxes.json").unlink()
    assert any("boxes.json: missing" in x for x in M.check(out))


def test_check_holds_the_plan_numbers_and_the_readout_reserve(tmp_path):
    """--check also refuses a valid registry whose plan-bound numbers differ from the plan record's (registry_drift:
    smoke A's plan_total_steps / plan_hours, F4's seconds, box 1's train hours), whose box full hours differ from the
    speed record's (box2_hours), or whose readout of a full box does not fit in its run's end reserve
    (readout_reserve_problems; smoke boxes are exempt)."""
    out = repo_copy(tmp_path)
    reg = json.loads((out / "boxes.json").read_text(encoding="utf-8"))

    def item(box, name):
        return next(it for it in reg["boxes"][box]["items"] if it["name"] == name)

    item("full-smoke", "smoke-t06")["plan_total_steps"] = 73452
    item("full-smoke", "smoke-p005")["plan_hours"] = 9.92
    next(f for f in reg["boxes"]["full-smoke"]["faults"] if f["id"] == "F4")["seconds"] = 140
    item("p01", "full-p01")["max_hours"] = 13.56
    item("full", "full-t06")["max_hours"] = 30.5  # not the speed record's
    reg["boxes"]["full"]["max_hours"] = 40
    item("full", "m4-full-p03")["max_hours"] = 0.4  # 24 + 10 min > full-p03's 30
    item("full-smoke", "m4-smoke-p03")["max_hours"] = 0.4  # smoke-p03's 2 min: exempt
    (out / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    probs = M.check(out)
    want = ["boxes.full-smoke.items.smoke-t06: {'plan_total_steps': 73452, 'plan_hours': 34.47}, the plan record "
            "gives {'plan_total_steps': 71946, 'plan_hours': 34.47}",
            "boxes.full-smoke.items.smoke-p005: {'plan_total_steps': 107910, 'plan_hours': 9.92}, the plan record "
            "gives {'plan_total_steps': 107910, 'plan_hours': 9.69}",
            "boxes.full-smoke.faults.F4: seconds 140, the plan record gives 150 (bound 151.05 s)",
            "boxes.p01.items.full-p01: {'max_hours': 13.56}, the plan record gives {'max_hours': 13.24}",
            "boxes.full.items.full-t06: {'max_hours': 30.5}, the speed record gives {'max_hours': 22.72}",
            "boxes.full: {'est_hours': 26.4, 'max_hours': 40}, the speed record gives {'est_hours': 26.4, "
            "'max_hours': 38}",
            "boxes.full.items.m4-full-p03: max_hours 0.4 (24 min) + 10 min of the trainer's end phase exceed "
            "configs/full/full-p03.json's schedule.end_reserve_min 30"]
    assert len(probs) == len(want) and all(any(p.startswith(f"boxes.json: {w}") for p in probs) for w in want), probs
    # the trainer's default end reserve on full-t06 (contract 7 as written): its 45 min readout no longer fits
    shutil.copyfile(FULL / "boxes.json", out / "boxes.json")
    c = cfg("full-t06")
    c["schedule"]["end_reserve_min"] = 30
    (out / "full-t06.json").write_text(json.dumps(c), encoding="utf-8")
    assert sorted(M.check(out)) == sorted([
        "full-t06.json: differs from the generator's",
        "boxes.json: boxes.full.items.m4-full-t06: max_hours 0.75 (45 min) + 10 min of the trainer's end phase exceed "
        "configs/full/full-t06.json's schedule.end_reserve_min 30: after a run the deadline cooldown shortened, the "
        "no-start rule would skip the readout (make_full_configs READOUT_RESERVE)"])


def test_write_all_keeps_the_hand_written_files(tmp_path):
    """write_all rewrites the generated files byte for byte and removes a stale one, never boxes.json or the plan."""
    out = repo_copy(tmp_path)
    boxes, plan_bytes = (out / "boxes.json").read_bytes(), (out / M.PLAN_FILE).read_bytes()
    (out / "old.json").write_text("{}", encoding="utf-8")
    for n in GENERATED[:3]:
        (out / f"{n}.json").unlink()
    written, removed = M.write_all(out)
    assert sorted(written) == sorted(GENERATED) and removed == ["old.json"]
    assert (out / "boxes.json").read_bytes() == boxes and (out / M.PLAN_FILE).read_bytes() == plan_bytes
    for n in GENERATED:
        assert (out / f"{n}.json").read_bytes() == (FULL / f"{n}.json").read_bytes().replace(b"\r\n", b"\n"), n
    assert M.check(out) == []


def test_every_config_loads_with_the_trainer(trainer, reg):
    """Each generated config passes 04_distill.load_config (DEFAULTS merge: no unknown key; validate incl.
    validate_full, validate_state and validate_box), and so does smoke-b's store config with the item's sets."""
    for n in GENERATED:
        c = trainer.load_config(str(FULL / f"{n}.json"), [])
        if not n.startswith("data-"):
            assert c["run_name"] == n and c["schedule"]["clock"] == "epochs"
    st = items(reg, "smoke-b")["stores-eval"]
    trainer.load_config(str(ROOT / st["config"]), st["sets"])
    with pytest.raises(SystemExit, match="max_steps"):  # the generated study config leaves them null: the sets fill
        trainer.load_config(str(ROOT / st["config"]), [])


# the keys a full config may differ in from its study run's config (make_study_configs.run_config): contract 7's
# table; smoke configs also SMOKE_ONLY. Data keys are compared whole (DATA_KEYS).
FULL_KEYS = {"run_name", "schedule.clock", "schedule.epochs", "schedule.max_steps", "schedule.warmup_steps",
             "schedule.deadline_cooldown", "schedule.end_reserve_min", "optim.lr", "batch.micro_audio_s",
             "batch.step_audio_s",
             "memory.max_oom_skips", "memory.probe_extended", "perf.peak_tflops", "eval.every_min", "eval.every_steps",
             "eval.full_every_epochs", "eval.full_at_fracs", "eval.mini.every_steps",
             *(f"eval.dev.{k}" for k in ("every_epochs", "every_steps", "per_source", "seed", "greedy", "at_start",
                                         "at_end")),
             "ckpt.weights_every_min", "ckpt.full_local_every_min", "ckpt.keep_local", "ckpt.full_at_fracs",
             "ckpt.weights_at_fracs", "ckpt.upload_full_at", "ckpt.upload_full_every_min", "log.sync_every_min",
             "log.full_scalars_every_steps", "log.scalars_parquet", "hf.output_repo", "hf.scratch_repo",
             *(f"early_stop.{k}" for k in ("enabled", "metric", "patience", "min_delta_rel", "min_delta_abs",
                                           "min_evals", "floor", "action", "smooth", "allow_test_sets"))}
SMOKE_ONLY = {"eval.final_full_greedy", "eval.greedy_subset", "schedule.deadline_check_steps",
              "schedule.deadline_window_steps", "memory.probe_shapes"}


@pytest.mark.parametrize("x", STUDENTS)
def test_a_config_is_its_study_run_but_for_the_contract_keys(x):
    base = S.run_config(f"study-{x}")
    for name, allowed in ((f"full-{x}", FULL_KEYS), (f"smoke-{x}", FULL_KEYS | SMOKE_ONLY)):
        c = cfg(name)
        a = dict(flat({k: v for k, v in base.items() if k not in fullrun.DATA_KEYS}))
        b = dict(flat({k: v for k, v in c.items() if k not in fullrun.DATA_KEYS}))
        diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
        assert diff <= allowed, (name, sorted(diff - allowed))
        # the study's settings stay: BN, L2-SP, aux-CTC, w_ctc, SpecAugment, the smoke block, the seed, verdict v2
        for k in ("bn", "specaug", "smoke", "seed", "student", "subset"):
            assert c[k] == base[k], (name, k)
        assert (c["loss"], c["eval"]["verdict_version"], c.get("family", "aed")) == (
            base["loss"], 2, "ctc" if x.startswith("p") else "aed")
        assert c["loss"]["l2sp_lambda"] == 0.0 and c["loss"]["aux_ctc_weight"] == 0.0
        assert c["loss"].get("w_ctc") == (0.8 if x.startswith("p") else None)  # the AED study config leaves it out


def test_the_section_7_table():
    """Contract 7's table, and the end reserve: the trainer's default 30 min but full-t06's 55 (READOUT_RESERVE, a
    deviation from contract 7 that test_a_readout_starts_after_a_shortened_run explains)."""
    want = {"t06": (3, 300, 2e-4, 450, 1730, False, 55), "p03": (3, 300, 2e-4, 600, 1350, True, 30),
            "p01": (4, 1000, 1e-3, 1600, 1500, True, 30), "p005": (4, 1000, 1e-3, 1600, 1500, True, 30)}
    for x, (epochs, warmup, lr, micro, step, greedy, reserve) in want.items():
        c = cfg(f"full-{x}")
        assert (c["schedule"]["epochs"], c["schedule"]["warmup_steps"], c["optim"]["lr"], c["batch"]["micro_audio_s"],
                c["batch"]["step_audio_s"], c["eval"]["dev"]["greedy"]) == (epochs, warmup, lr, micro, step, greedy), x
        assert warmup == prereg.rules()["runs"][f"study-{x}"]["warmup_steps"]  # the study's warm-up
        assert c["schedule"] | {"epochs": 0, "warmup_steps": 0} == {
            "warmup_steps": 0, "cooldown_frac": 0.2, "train_hours": 4.0, "clock": "epochs", "max_steps": None,
            "end_reserve_min": reserve, "epochs": 0, "deadline_cooldown": True}
        assert c["memory"]["max_oom_skips"] == 50 and c["memory"]["probe_extended"] is True
        assert "probe_shapes" not in c["memory"]  # the trainer's default null: the full runs probe their own data
        assert c["perf"]["peak_tflops"] == 209.5
        ev = c["eval"]
        assert (ev["every_min"], ev["every_steps"], ev["full_every_epochs"], ev["full_at_fracs"],
                ev["mini"]["every_steps"]) == (None, None, 1, None, 2000)
        assert ev["dev"] == {"every_epochs": 0.1, "every_steps": None, "per_source": 600, "seed": 1234,
                             "at_start": True, "at_end": True, "greedy": greedy}
        ck = c["ckpt"]
        assert {k: ck[k] for k in ("weights_every_min", "full_local_every_min", "keep_local", "full_at_fracs",
                                   "weights_at_fracs", "upload_full_at", "upload_full_every_min")} == {
            "weights_every_min": None, "full_local_every_min": 30, "keep_local": 2, "full_at_fracs": None,
            "weights_at_fracs": None, "upload_full_at": ["pre_cooldown"], "upload_full_every_min": 120}
        assert {k: c["log"][k] for k in ("sync_every_min", "full_scalars_every_steps", "scalars_parquet")} == {
            "sync_every_min": 60, "full_scalars_every_steps": 10, "scalars_parquet": "close"}
        assert c["hf"]["output_repo"] == "Multy123/kitsune-runs" and c["hf"]["scratch_repo"] is None
        assert c["early_stop"] == {"enabled": True, "metric": "dev_ce", "patience": 6, "min_delta_rel": 0.005,
                                   "min_delta_abs": 0.0, "min_evals": 5, "floor": None, "action": "cooldown",
                                   "smooth": 5, "allow_test_sets": False}


def test_the_smoke_variants(plan):
    for x in STUDENTS:
        f, s = cfg(f"full-{x}"), cfg(f"smoke-{x}")
        assert s["run_name"] == f"smoke-{x}" and s["schedule"]["epochs"] == f["schedule"]["epochs"]
        assert s["subset"] == f["subset"] and not any(s["subset"].values())  # no subset: the 100 h draw is the budget
        ev = s["eval"]
        assert (ev["full_every_epochs"], ev["final_full_greedy"], ev["greedy_subset"], ev["mini"]["every_steps"],
                ev["dev"]["per_source"]) == (None, False, 100, 100, 60)
        assert {k: v for k, v in ev["dev"].items() if k != "per_source"} == \
            {k: v for k, v in f["eval"]["dev"].items() if k != "per_source"}
        p01 = x == "p01"
        assert (s["ckpt"]["full_local_every_min"], s["ckpt"]["upload_full_every_min"]) == ((3, 3) if p01 else (5, 10))
        assert (s["log"]["sync_every_min"], s["log"]["full_scalars_every_steps"]) == (5 if p01 else 10, 1)
        assert (s["schedule"]["end_reserve_min"], s["schedule"]["deadline_check_steps"],
                s["schedule"]["deadline_window_steps"]) == (2, 10, 20)
        assert s["memory"]["probe_shapes"] == plan["full"]["students"][x]["worst_shapes"]
        assert [sh["name"] for sh in s["memory"]["probe_shapes"]] == ["most_rows", "longest", "most_targets"]
        es = dict(s["early_stop"])
        forced = {"t06": {"min_delta_abs": 1e9, "patience": 8}, "p03": {"min_delta_abs": 1e9}}.get(x, {})
        assert es == dict(f["early_stop"], **forced), x
        if forced:  # the queue's check 10 recognises a forced trigger by this
            from kitsune import full_queue
            assert es["min_delta_abs"] >= full_queue.FORCED_MIN_DELTA_ABS


def test_the_data_configs():
    def data(c):
        return {k: v for k, v in c.items() if k != "_comment"}

    assert data(cfg("data-p01")) == dict(fullrun.FULL_DATA, family="ctc")
    assert data(cfg("data-full")) == dict(fullrun.FULL_DATA, pull_parakeet=True)
    assert data(cfg("data-smoke")) == dict(fullrun.SMOKE_DATA, pull_parakeet=True)
    study = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
    assert data(cfg("data-smoke-b")) == {k: study[k] for k in fullrun.DATA_KEYS if k in study}
    assert "pull_parakeet" not in cfg("data-p01") and "pull_parakeet" not in cfg("full-p01")
    for n in ("data-p01", "data-full", "data-smoke", "data-smoke-b"):
        assert "tools/make_full_configs.py" in cfg(n)["_comment"]
    # the train configs carry their box's data keys (the registry checks it too); smoke-p01 pulls both roots, as
    # smoke A does
    for x in STUDENTS:
        for name, dc in ((f"full-{x}", "data-p01" if x == "p01" else "data-full"), (f"smoke-{x}", "data-smoke")):
            got, want = cfg(name), cfg(dc)
            assert {k: got.get(k) for k in fullrun.DATA_KEYS if k != "pull_parakeet"} == \
                {k: want.get(k) for k in fullrun.DATA_KEYS if k != "pull_parakeet"}, name
            assert bool(got.get("pull_parakeet")) == bool(want.get("pull_parakeet")), name
    # the chained box's stage 1 (contract addendum E.2.1): smoke A and smoke B rebuild one extent with one set of
    # label roots and both teachers
    a, b = cfg("data-smoke"), cfg("data-smoke-b")
    for k in ("extent", "data_root", "teacher_root", "second_root", "parakeet_root", "pull_parakeet"):
        assert a[k] == b[k], k
    assert extent.within(cfg("data-p01"), a) == [] and extent.within(a, b) == [] and extent.within(b, a) == []


def toy_record(cfg_: dict) -> dict:
    """An extent record of the full extent's names, two stems per train source (one of them the 300 h step's for
    emilia_yodas), galgame's eval hold-out and one eval stem per eval set."""
    def stem(name, step):
        return {"stem": name, "split": name.split("-")[0], "step": step, "rows": 1, "hours": 0.01,
                "ids_sha256": "0" * 64, "shard_bytes": 1000}

    stems = {s: [stem("train-00000", s), stem("train-00001", s)] for s in ("reazon_small", "reazon_large", "emilia_nc")}
    stems["emilia_yodas"] = [stem("train-00000", "emilia_yodas@300h"), stem("train-00001", "emilia_yodas")]
    stems["galgame"] = [stem("train-00000", "galgame"), stem("eval-00000", "galgame")]
    stems.update({e: [stem("eval-00000", e)] for e in ("eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia")})
    return {"schema": 1, "name": "full", "root": "labels/full", "canonical_version": extent.CANONICAL_VERSION,
            "names": extent.names(cfg_), "inputs": {},
            "sources": {s: {"repo": s, "rows": len(v), "hours": 0.01, "bytes": 1000,
                            "inputs": [{"input": "x", "ordinal": 0, "bytes": 1000, "stems": v}]}
                        for s, v in stems.items()}}


def test_box_1_pulls_parakeet_for_every_stem_and_teacher_for_the_eval_stems_only():
    """data-p01 (CTC, no pull_parakeet): kitsune.extent.pull_plan asks for parakeet_out of every stem and teacher_out
    of the eval sets' eval stems only (fix 9), and label_root_for reads the train stems' ids from parakeet_out; box 2's
    data-full pulls both roots for every stem."""
    p01, full = cfg("data-p01"), cfg("data-full")
    rec = toy_record(p01)
    assert extent.validate(p01) == [] and extent.record_problems(rec, p01) == []
    stems = {(n, st) for n, sts in extent.subset_stems(rec, p01).items() for st in sts}
    assert len(stems) == 14 and {st for _, st in stems} == {"train-00000", "train-00001", "eval-00000"}

    def npz(plan_, root):
        return {tuple(f.split("/")[-2:]) for f in plan_["required"] if f.startswith(f"labels/full/{root}/")
                and f.endswith(".npz")}

    got = extent.pull_plan(p01, rec, [])
    assert npz(got, "parakeet_out") == {(n, f"{st}.npz") for n, st in stems}
    evals = {(n, f"{st}.npz") for n, st in stems if st.startswith("eval-") and n in p01["eval_sets"]}
    assert npz(got, "teacher_out") == evals and len(evals) == 5
    assert extent.label_root_for(p01, "reazon_small", "train-00000") == "parakeet"
    assert extent.label_root_for(p01, "galgame", "train-00000") == "parakeet"
    assert extent.label_root_for(p01, "galgame", "eval-00000") == "teacher"
    both = extent.pull_plan(full, toy_record(full), [])
    assert npz(both, "teacher_out") == npz(both, "parakeet_out") == {(n, f"{st}.npz") for n, st in stems}
    assert extent.label_root_for(full, "reazon_small", "train-00000") == "teacher"


# ================================================================================================ the plan record


def test_the_plan_record(plan):
    assert M.plan_problems(plan) == []
    assert (plan["full"]["selection"], plan["smoke"]["selection"]) == (
        {"path": fullrun.FULL_SELECTION, "sha256": FULL_SHA}, {"path": fullrun.SMOKE_SELECTION, "sha256": SMOKE_SHA})
    for x in STUDENTS:
        s = plan["full"]["students"][x]
        assert s["total_steps"] == MEASURED_T[x] == sum(s["steps_per_epoch"])
        # the step values realise the study's audio per step on the full data (DECISIONS C10) ...
        assert abs(s["delta_pct"]) < 0.1 and abs(s["step_real_s"]["mean"] / s["study_step_real_s"] - 1) < 1e-3, x
        # ... and on the smoke's 100 h draw
        sm = plan["smoke"]["students"][x]
        assert abs(sm["step_real_s"]["mean"] / sm["study_step_real_s"] - 1) < 1e-3, x
    assert plan["smoke"]["students"]["p005"]["total_steps"] == 935
    assert math.isclose(plan["smoke"]["train_hours"], 100.0, abs_tol=0.01)


def test_import_plan(tmp_path, plan, capsys):
    """--import-plan records tools/full_plan.py's two JSON files (the local paths replaced by the repo's) and writes
    the configs; a record measured with other step values or on another file is refused and nothing is written."""
    out = repo_copy(tmp_path)
    before = (out / M.PLAN_FILE).read_bytes()
    src = {}
    for which, name in (("full", "full.parquet"), ("smoke", "smoke.parquet")):
        rec = copy.deepcopy(plan[which])
        rec["selection"]["path"] = f"D:\\somewhere\\full_study\\{name}"
        src[which] = tmp_path / f"plan_{which}.json"
        src[which].write_text(json.dumps(rec), encoding="utf-8")
    (out / M.PLAN_FILE).unlink()
    assert M.main(["--out-dir", str(out), "--import-plan", str(src["full"]), str(src["smoke"])]) == 0
    printed = capsys.readouterr().out
    assert "t06 plan_total_steps 71946 plan_hours 34.47" in printed and "F4 seconds 150" in printed
    assert "boxes.json: carries the plan record's numbers" in printed
    assert json.loads((out / M.PLAN_FILE).read_text(encoding="utf-8")) == json.loads(before)
    assert M.check(out) == []
    # a measurement the registry does not carry yet: recorded, the change printed, --check fails until boxes.json has it
    moved = json.loads(src["smoke"].read_text(encoding="utf-8"))
    moved["students"]["p005"]["total_steps"] = 1200  # F4's bound 0.5 x 1200 x 0.3231 s = 193.9 s
    src["smoke"].write_text(json.dumps(moved), encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-plan", str(src["full"]), str(src["smoke"])]) == 0
    printed = capsys.readouterr().out
    assert "F4 seconds 190" in printed and "boxes.json: 1 change(s) by hand" in printed
    assert "boxes.full-smoke.faults.F4: seconds 150, the plan record gives 190 (bound 193.86 s)" in printed
    assert M.check(out) == ["boxes.json: boxes.full-smoke.faults.F4: seconds 150, the plan record gives 190 (bound "
                            "193.86 s)"]
    bad = json.loads(src["full"].read_text(encoding="utf-8"))
    bad["students"]["p03"]["step_audio_s"] = 1200.0
    src["full"].write_text(json.dumps(bad), encoding="utf-8")
    (out / M.PLAN_FILE).unlink()
    assert M.main(["--out-dir", str(out), "--import-plan", str(src["full"]), str(src["smoke"])]) == 1
    assert "full.p03: measured with {'step_audio_s': 1200.0}" in capsys.readouterr().err
    assert not (out / M.PLAN_FILE).exists()
    with pytest.raises(M.PlanError, match="not a full.parquet"):
        M.import_plan(src["smoke"], src["full"], out)
    assert M.check(out)[0].startswith(str(out / M.PLAN_FILE))  # no record: --check says so, it does not crash


def test_the_registry_numbers_follow_the_plan(plan):
    """plan_hours = plan v3's hours x the measured T / the plan's T (2 decimals), and F4's seconds: contract 7's
    bound 0.5 x smoke-p005's steps on the smoke selection x plan v3's s/step of full-p005, rounded down to 10 s."""
    nums = M.registry_numbers(plan)
    assert nums["students"] == {"t06": {"plan_total_steps": 71946, "plan_hours": 34.47},
                                "p03": {"plan_total_steps": 105861, "plan_hours": 21.53},
                                "p01": {"plan_total_steps": 107910, "plan_hours": 13.24},
                                "p005": {"plan_total_steps": 107910, "plan_hours": 9.69}}  # 9.92 x 0.97631 = 9.685
    for x, (h, t) in M.PLAN_V3.items():
        assert nums["students"][x]["plan_hours"] == round(h * MEASURED_T[x] / t, 2)
    bound = 0.5 * 935 * 9.92 * 3600 / 110528
    assert math.isclose(M.deadline_fault_bound_s(plan), bound) and math.isclose(bound, 151.05, abs_tol=0.01)
    assert nums["deadline_fault_s"] == 150 <= bound


SMOKE_SPS = {"t06": 1.0429505235515535, "p03": 0.39240704476833344, "p01": 0.2715486469678581,
             "p005": 0.1792971519753337}  # smoke A's check 3 (instance 53693389, 14bfcad, machine 19048)


def smoke_verdict(sps=SMOKE_SPS, **kw) -> dict:
    """A smoke A verdict with check 3's evidence (as kitsune.full_queue.SmokeVerdict.check3 writes it)."""
    return dict({"format": 1, "box": "full-smoke", "sha": "14bfcadbc660394955d290ed5256d49398882d4a",
                 "machine_id": "19048", "time_utc": "2026-10-01T13:47:02+00:00", "overall": "fail",
                 "checks": {"3": {"pass": True, "evidence": {"sec_per_step": {f"smoke-{x}": v for x, v in sps.items()},
                                                             "box2_h": 20.84}}}}, **kw)


def test_box2_hours_follow_the_record(plan):
    """Box full's hours = box2_hours of the speed record: plan v3's run model (calc_v3: steps x s/step x (1 + o) +
    fixed + dev checks) at smoke A's measured s/step x r (box 1's / smoke A's P-0.1B s/step; 1 while the record has no
    box-1 part) with o (box 1's, else 0.08), est = the longer GPU lane, max = the pessimistic T-0.6B path + tail +
    reserve + one stall."""
    rec = M.load_speed()
    assert rec["box1"] is None and rec["smoke"]["sec_per_step"] == SMOKE_SPS  # provisional: smoke A only
    assert rec["smoke"]["source"] == "full/box-full-smoke/smoke_verdict.json"
    h = M.box2_hours(plan, rec)
    # t06: 71,946 steps x 1.04295 s x 1.08 + 435 s fixed + 30 dev checks x 9.6 s, rounded up to 0.01 h
    assert math.ceil((71946 * SMOKE_SPS["t06"] * 1.08 + 435 + 30 * 9.6) / 36) / 100 == 22.72
    assert h["run_h"] == {"t06": 22.72, "p03": 12.64, "p01": 9.12, "p005": 6.13}  # p01: box 1's cross-check
    assert h["items"] == {"full-t06": 22.72, "full-p03": 12.64, "full-p005": 6.13}
    assert (h["r"], h["o"], h["setup_h"], h["stores_h"]) == (1.0, 0.08, (1.6, 3.3), (1.37, 1.91))
    # lane A: setup 1.6 + both stores 1.37 + T 22.72 + its tail 0.3 + end 0.35; lane B: setup + the CTC store (2/3 of
    # 1.37) + P-0.3B + P-0.05B + the eval pool 1.2 + end
    assert h["lanes"] == {"A": 26.34, "B": 22.83} and h["est_hours"] == 26.4
    # max: ceil(3.3 + 1.91 + 22.72 x 1.26 + 1.2 + 60 min + 1.25) = ceil(37.29); T-0.6B's pessimistic end 33.84 h
    # leaves 38 - 1 - 55 min - 33.84 = 2.25 h before 4d would compress it
    assert (h["max_hours"], h["t06_slack_h"]) == (38, 2.25)
    # box 1 10 % slower than smoke A per step (r 1.1): every run scales, the setup and stores stay the plan's
    rec1 = copy.deepcopy(rec)
    rec1["box1"] = {"sec_per_step": 0.2987, "overhead": None, "stores_ctc_h": None, "bootstrap_h": None}
    h1 = M.box2_hours(plan, rec1)
    assert (h1["r"], h1["o"], h1["items"], h1["est_hours"], h1["max_hours"]) == (
        1.1, 0.08, {"full-t06": 24.97, "full-p03": 13.89, "full-p005": 6.71}, 28.6, 41)
    # box 1's overhead, CTC store and bootstrap feed o, the stores (x 1.5 for both, x 1.39 pessimistic) and the
    # pessimistic setup (its bootstrap + 0.2 h, when above 3.3)
    rec1["box1"].update(overhead=0.1, stores_ctc_h=1.0, bootstrap_h=3.6)
    h2 = M.box2_hours(plan, rec1)
    assert (h2["o"], h2["setup_h"], h2["stores_h"], h2["items"]["full-t06"], h2["est_hours"], h2["max_hours"]) == (
        0.1, (1.6, 3.8), (1.5, 2.09), 25.43, 29.2, 42)
    assert M.box2_hours(plan, rec, reserve_min=120)["max_hours"] == 39
    assert M.speed_problems(rec) == [] and M.speed_problems(rec1) == []
    bad = copy.deepcopy(rec1)
    del bad["smoke"]["sec_per_step"]["p01"]
    bad["smoke"]["sec_per_step"]["t06"] = 0
    bad["box1"].update(overhead=1.5, stores_ctc_h=-1)
    assert M.speed_problems(bad) == ["smoke.sec_per_step.t06: 0, not a number > 0",
                                     "smoke.sec_per_step.p01: None, not a number > 0",
                                     "box1.overhead: 1.5, not null or a number in [0, 1)",
                                     "box1.stores_ctc_h: -1, not null or a number > 0"]
    assert M.speed_problems({"smoke": None, "box1": []}) == [
        "smoke: no smoke A record (smoke verdict check 3's sec_per_step)", "box1: list, not an object or null"]


def test_import_speed(tmp_path, plan, capsys):
    """--import-speed records smoke A's check 3 (and, later, box 1's tools/box1_go.py --json) as the speed record,
    byte-stable, prints the hours boxes.json must carry, and --check holds boxes.json to them; a verdict without
    check 3's s/step is refused and nothing is written; without a record box full's hours are not held."""
    out = repo_copy(tmp_path)
    committed = (out / M.SPEED_FILE).read_bytes()
    (out / M.SPEED_FILE).unlink()
    assert M.check(out) == []  # no record: box full's hours are not held
    v = tmp_path / "smoke_verdict.json"
    v.write_text(json.dumps(smoke_verdict()), encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-speed", "--smoke-verdict", str(v)]) == 0
    printed = capsys.readouterr().out
    assert "smoke A only: provisional; r 1, o 0.08" in printed
    assert ("full-t06 max_hours 22.72, full-p03 max_hours 12.64, full-p005 max_hours 6.13; box full est_hours 26.4, "
            "max_hours 38") in printed and "boxes.json: carries the speed record's hours" in printed
    assert (out / M.SPEED_FILE).read_bytes().replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")
    assert M.check(out) == []
    # box 1's measurement: the smoke part is kept, the new hours printed, --check fails until boxes.json carries them
    g = tmp_path / "box1_go.json"
    g.write_text(json.dumps({"go": True, "box1": {"source": "full/box-p01/queue_summary.json", "sec_per_step": 0.2987,
                                                  "overhead": None, "stores_ctc_h": None, "bootstrap_h": None}}),
                 encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-speed", "--box1-go", str(g)]) == 0
    printed = capsys.readouterr().out
    assert "smoke A and box 1; r 1.1, o 0.08" in printed and "boxes.json: 4 change(s) by hand" in printed
    assert "boxes.full.items.full-t06: {'max_hours': 22.72}, the speed record gives {'max_hours': 24.97}" in printed
    rec = M.load_speed(out)
    assert rec["smoke"]["sec_per_step"] == SMOKE_SPS and rec["box1"]["sec_per_step"] == 0.2987
    assert sorted(M.check(out)) == sorted([
        "boxes.json: boxes.full.items.full-t06: {'max_hours': 22.72}, the speed record gives {'max_hours': 24.97}",
        "boxes.json: boxes.full.items.full-p03: {'max_hours': 12.64}, the speed record gives {'max_hours': 13.89}",
        "boxes.json: boxes.full.items.full-p005: {'max_hours': 6.13}, the speed record gives {'max_hours': 6.71}",
        "boxes.json: boxes.full: {'est_hours': 26.4, 'max_hours': 38}, the speed record gives {'est_hours': 28.6, "
        "'max_hours': 41}"])
    reg = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    reg["boxes"]["full"].update(est_hours=28.6, max_hours=41)
    for it in reg["boxes"]["full"]["items"]:
        if it["name"] in ("full-t06", "full-p03", "full-p005"):
            it["max_hours"] = {"full-t06": 24.97, "full-p03": 13.89, "full-p005": 6.71}[it["name"]]
    (out / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    assert M.check(out) == []
    # refusals: nothing is written
    before = (out / M.SPEED_FILE).read_bytes()
    v.write_text(json.dumps(smoke_verdict(checks={"1": {"pass": True}})), encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-speed", "--smoke-verdict", str(v)]) == 1
    assert "smoke check 3 has no sec_per_step for ['smoke-t06'" in capsys.readouterr().err
    g.write_text(json.dumps({"go": None, "box1": None}), encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-speed", "--box1-go", str(g)]) == 1
    assert "no box1 measurement" in capsys.readouterr().err
    assert (out / M.SPEED_FILE).read_bytes() == before
    with pytest.raises(SystemExit):
        M.main(["--out-dir", str(out), "--smoke-verdict", str(v)])
    assert "go with --import-speed" in capsys.readouterr().err
    (out / M.SPEED_FILE).write_text('{"smoke": {}}', encoding="utf-8")
    assert any("smoke.sec_per_step.t06: None" in x for x in M.check(out))


# ================================================================================================ the registry


BOX_TABLE = {  # contract 7: gpus, data config, est / max h, max_dph, extra_gb, reserve, watchdog, timed, gate, smoke
    "full-smoke": (1, "data-smoke", 5.2, 9, 1.00, 90, 20, (600, "alert"), True, True, True),
    "p01": (1, "data-p01", 19.5, 22, 1.00, 25, 45, (3600, "stop"), True, True, False),
    "full": (2, "data-full", 26.4, 38, 1.70, 110, 60, (3600, "stop"), True, True, False),  # box2_hours
    "smoke-b": (1, "data-smoke-b", 1.75, 3, 1.00, 40, 15, (3600, "stop"), False, False, True),
}


def test_the_registry_loads_with_its_boxes(reg, monkeypatch):
    monkeypatch.delenv(fullrun.ENV_REGISTRY, raising=False)
    assert fullrun.load_registry() == reg  # this checkout's configs/full/boxes.json, every file checked
    assert fullrun.registry_problems(json.loads((FULL / "boxes.json").read_text(encoding="utf-8")), root=ROOT) == []
    # every plain box, and the chained box of contract addendum E (test_the_chained_box)
    assert set(fullrun.BOX_NAMES) == set(BOX_TABLE) and set(reg["boxes"]) == set(fullrun.ALL_BOX_NAMES)
    for box, (gpus, dc, est, mx, dph, extra, reserve, wd, timed, gate, smoke) in BOX_TABLE.items():
        b = fullrun.box_spec(box, reg)
        assert (b["gpus"], b["data_config"], b["est_hours"], b["max_hours"], b["max_dph"], b["extra_gb"],
                b["deadline_reserve_min"], (b["watchdog"]["orphan_s"], b["watchdog"]["action"]), b["timed_states"],
                b["gate"], b["smoke"], b["max_attempts"]) == (
            gpus, f"configs/full/{dc}.json", est, mx, dph, extra, reserve, wd, timed, gate, smoke, 4), box
        assert fullrun.box_env(box, reg)[fullrun.ENV_N_GPUS] == str(gpus)
        assert bool(b["faults"]) == (box == "full-smoke")
    sidecar = {"full-smoke": "smoke.json", "p01": "full.json", "full": "full.json"}
    for box in BOX_TABLE:
        want = [fullrun.FROZEN_MANIFEST, f"{fullrun.FULL_DIR}/{sidecar[box]}"] if box in sidecar else []
        assert fullrun.box_extra_files(box, reg) == want, box
        assert fullrun.box_extra_dirs(box, reg) == ([PARAKEET] if box in ("full-smoke", "smoke-b") else []), box
    assert fullrun.box_students("p01", reg) == ["students/study/p01"]
    assert fullrun.box_ctc_students("full", reg) == ["students/study/p03", "students/study/p005"]
    assert fullrun.box_students("full-smoke", reg) == [f"students/study/{x}" for x in STUDENTS]
    assert fullrun.box_students("smoke-b", reg) == []
    # no argv, args or verdict json reads the state dir (a chained box's part runs in chain/<part>, addendum E.10)
    for box in BOX_TABLE:
        for it in fullrun.box_items(box, reg):
            templates = [*(it.get("argv") or []), *(it.get("args") or []), *(v.get("json", "") for v in it["verdict"])]
            assert not any("{state}" in t for t in templates), (box, it["name"])


def test_box_1(reg, plan):
    it = items(reg, "p01")
    assert list(it) == ["stores-ctc", "full-p01", "m4-full-p01"]
    assert it["stores-ctc"]["config"] == it["full-p01"]["config"] == "configs/full/full-p01.json"
    t = it["full-p01"]
    # the run's hours: plan v3's at the step count measured on full.parquet (the plan record; --check holds it too)
    assert (t["kind"], t["study_run"], t["family"], t["max_hours"], t["needs"], t["droppable"], t["stall_min"],
            t["plan_total_steps"]) == ("train", "study-p01", "ctc", M.registry_numbers(plan)["students"]["p01"][
                "plan_hours"], ["stores-ctc"], False, 45, None)
    assert t["max_hours"] == 13.24
    assert (it["m4-full-p01"]["of"], it["m4-full-p01"]["max_hours"], it["m4-full-p01"]["needs"]) == (
        "full-p01", 0.3, ["full-p01"])


@pytest.mark.parametrize("box", ["p01", "full"])
def test_a_readout_starts_after_a_shortened_run(tmp_path, reg, box):
    """A readout runs right after its run, and the run's KITSUNE_DEADLINE (full_queue.item_deadline) is the one the
    trainer's deadline cooldown (4d) plans against: 04_distill.fit_epochs_deadline ends a shortened run's end phase
    schedule.end_reserve_min before it, and the readout gets what the trainer's final saves and uploads leave of that.
    Here, on the real queue, for every readout of a full box: the trainer's end phase used all of END_PHASE_SLACK_MIN
    but a minute, and the no-start rule still starts the readout. With contract 7's default 30 min on full-t06, the
    rule tested against item_deadline left m4-full-t06 (0.75 h) about 28 min and skipped it, and every quantised
    readout and the speed re-time that need it with it: full-t06's end reserve is 55 min (make_full_configs
    READOUT_RESERVE). A no-start rule that gives a readout more room than item_deadline passes this too."""
    from kitsune import full_queue as F

    spec, its = fullrun.box_spec(box, reg), items(reg, box)
    readouts = [n for n, it in its.items() if it["kind"] == "readout"]
    assert readouts
    for n in readouts:
        r = its[n]
        reserve = cfg(Path(its[r["of"]]["config"]).stem)["schedule"]["end_reserve_min"]
        assert reserve >= r["max_hours"] * 60 + M.END_PHASE_SLACK_MIN, n
        assert box != "full" or n != "m4-full-t06" or r["max_hours"] * 60 + M.END_PHASE_SLACK_MIN > 30
        state = tmp_path / n
        state.mkdir()
        left = (reserve - M.END_PHASE_SLACK_MIN + 1) * 60  # at the readout's start, before the run's KITSUNE_DEADLINE
        (state / fullrun.DEADLINE_FILE).write_text(str(time.time() + spec["deadline_reserve_min"] * 60 + left))
        s = F.FullSettings(root=tmp_path, state_dir=state, gpus=[str(i) for i in range(spec["gpus"])], n_gpus=None,
                           python="python", out_repo=None, scratch_repo=None, uploader=None, machine_id=None,
                           proc_root=tmp_path / "no-proc", cgroup=tmp_path / "no-cgroup")
        q = F.FullQueue(box, s, registry=reg)
        q.register()
        rd = f"runs/{r['of']}-20261001T000000Z"
        q.item(r["of"]).update(status="done", run_dir=rd, result=dict(steps=100))
        (tmp_path / rd / "checkpoints" / "step_100").mkdir(parents=True, exist_ok=True)
        assert q._prepare(n) is True and q.item(n)["status"] == "pending", n
        assert n not in q.state["no_start"], n


def quant_argv(fmt: str, config="{config}", ckpt="{ckpt}") -> list[str]:
    return ["{python}", "-m", "kitsune.quant", "readout", "--config", config, "--ckpt", ckpt, "--fmt", fmt, "--out",
            "{out}", "--cache-dir", "{cache_dir}", "--manifest", "{manifest}", "--max-temp", "0"]


def whisper_argv(key: str, *extra: str) -> list[str]:
    return ["{python}", "tools/whisper_eval.py", "--model", key, "--store", "{cache_dir}/eval", "--manifest",
            "{manifest}", "--out", "{out}", "--tables", "{out}/tables", "--hf-cache", "{hf_cache}", "--device", "cuda",
            "--max-temp", "0", *extra]


def test_box_2(reg, plan):
    it = items(reg, "full")
    quant = {x: [f"quant-{f}-full-{x}" for f in QUANT_FORMATS] for x in ("p03", "p005", "p01", "t06")}
    assert list(it) == ["stores-ctc", "stores-aed", "full-t06", "full-p03", "full-p005", "m4-full-t06", "m4-full-p03",
                        "m4-full-p005", *quant["p03"], *quant["p005"], *WHISPER_KEYS, *quant["p01"], *quant["t06"],
                        "speed-full-t06", "speed-study-t06"]
    assert (it["stores-ctc"]["config"], it["stores-aed"]["config"], it["stores-aed"]["needs"]) == (
        "configs/full/full-p03.json", "configs/full/full-t06.json", ["stores-ctc"])
    # the train hours are box2_hours of the speed record (smoke A's measured s/step; provisional until the PR after
    # box 1 adds box 1's part, contract 7), and so are the box's est / max hours (--check holds them)
    hours = M.box2_hours(plan, M.load_speed(), reserve_min=fullrun.box_spec("full", reg)["deadline_reserve_min"])
    assert (fullrun.box_spec("full", reg)["est_hours"], fullrun.box_spec("full", reg)["max_hours"]) == (
        hours["est_hours"], hours["max_hours"]) == (26.4, 38)
    for x, h, st, drop in (("t06", 22.72, "stores-aed", False), ("p03", 12.64, "stores-ctc", False),
                           ("p005", 6.13, "stores-ctc", True)):
        t = it[f"full-{x}"]
        assert (t["config"], t["study_run"], t["family"], t["max_hours"], t["needs"], t["droppable"]) == (
            f"configs/full/full-{x}.json", f"study-{x}", "aed" if x == "t06" else "ctc", hours["items"][f"full-{x}"],
            [st], drop), x
        assert t["max_hours"] == h, x
        assert it[f"m4-full-{x}"]["max_hours"] == (0.75 if x == "t06" else 0.3)
    for x, names in quant.items():
        for f, n in zip(QUANT_FORMATS, names):
            q = it[n]
            assert q["kind"] == "eval" and q["argv"] == quant_argv(f) and q["of"] == f"full-{x}", n
            assert q["max_hours"] == (0.75 if x == "t06" else 0.3), n
            if x == "p01":  # box 1's run, from its Hub summary
                assert q["of_box"] == "p01" and q["needs"] == []
            else:  # after the readout of this box's run (the implicit need on the run itself added)
                assert q["of_box"] is None and set(q["needs"]) == {f"m4-full-{x}", f"full-{x}"}
    for k, h in zip(WHISPER_KEYS, (0.75, 0.5, 0.5, 0.3)):
        assert (it[k]["argv"], it[k]["max_hours"], it[k]["weights"], it[k]["of"]) == (whisper_argv(k), h, [], None)
    a, b = it["speed-full-t06"], it["speed-study-t06"]
    assert (a["system"], a["speed_kind"], a["of"], set(a["needs"]), a["max_hours"], a["droppable"]) == (
        "full-t06", "aed", "full-t06", {"m4-full-t06", "full-t06"}, 0.2, True)
    assert (b["system"], b["speed_kind"], b["weights"], b["max_hours"]) == (
        "study-t06", "aed", [{"name": "study-t06", "run_id": STUDY_WEIGHTS["study-t06"][0], "step": 9370}], 0.2)


def test_smoke_a(reg, plan):
    it = items(reg, "full-smoke")
    speed = [f"speed-{s}" for s in STUDY_WEIGHTS] + ["speed-cohere", "speed-parakeet-ctc", "speed-parakeet-tdt"]
    trains = [f"smoke-{x}" for x in STUDENTS]
    assert list(it) == ["stores-ctc", "stores-aed", *trains, "smoke-nostart", "m4-smoke-t06", "m4-smoke-p03", *speed]
    assert (it["stores-ctc"]["config"], it["stores-aed"]["config"]) == ("configs/full/smoke-p03.json",
                                                                        "configs/full/smoke-t06.json")
    nums = M.registry_numbers(plan)["students"]
    for x in STUDENTS:
        t = it[f"smoke-{x}"]
        assert (t["config"], t["study_run"], t["max_hours"], t["stall_min"], t["droppable"]) == (
            f"configs/full/smoke-{x}.json", f"study-{x}", 0.75, 10, False), x
        assert t["needs"] == ["stores-aed" if x == "t06" else "stores-ctc"]
        # smoke check 3's projection: the full run's measured T and its plan hours
        assert (t["plan_total_steps"], t["plan_hours"]) == (nums[x]["plan_total_steps"], nums[x]["plan_hours"])
    ns = it["smoke-nostart"]
    assert (ns["config"], ns["max_hours"], ns["droppable"], ns["plan_total_steps"]) == (
        "configs/full/smoke-p005.json", 999, True, None)
    assert (it["m4-smoke-t06"]["max_hours"], it["m4-smoke-p03"]["max_hours"]) == (0.75, 0.3)
    for s, (rid, step) in STUDY_WEIGHTS.items():
        sp = it[f"speed-{s}"]
        assert (sp["system"], sp["speed_kind"], sp["weights"], sp["stall_min"], sp["max_hours"]) == (
            s, "aed" if s.startswith("study-t") else "ctc", [{"name": s, "run_id": rid, "step": step}], None, 0.3)
    assert (it["speed-cohere"]["speed_kind"], it["speed-cohere"]["model"], it["speed-cohere"]["weights"]) == (
        "cohere", None, [])
    for k in ("parakeet-ctc", "parakeet-tdt"):
        assert (it[f"speed-{k}"]["speed_kind"], it[f"speed-{k}"]["model"], it[f"speed-{k}"]["stall_min"]) == (
            k, PARAKEET, None)
    faults = {f["id"]: {k: v for k, v in f.items() if not k.startswith("_")}
              for f in fullrun.box_spec("full-smoke", reg)["faults"]}
    base = {"at_step": None, "after_event": None, "min_attempt": 1, "seconds": None}
    assert faults == {
        "F1": dict(base, id="F1", action="sigstop", item="smoke-p01", at_step=150),
        "F2": dict(base, id="F2", action="kill", item="smoke-p03", at_step=130),
        "F3": dict(base, id="F3", action="wipe_run_dir", item="smoke-p01", after_event="timed_state_upload_ok",
                   min_attempt=2),
        "F4": dict(base, id="F4", action="deadline", item="smoke-p005", seconds=M.deadline_fault_s(plan)),
        "F5": dict(base, id="F5", action="freeze_controller_hb", item="smoke-p005", at_step=50, seconds=900)}
    assert faults["F4"]["seconds"] <= M.deadline_fault_bound_s(plan)
    assert faults["F5"]["seconds"] > fullrun.box_spec("full-smoke", reg)["watchdog"]["orphan_s"]


def test_smoke_b(reg):
    it = items(reg, "smoke-b")
    st = it["stores-eval"]
    assert (st["kind"], st["config"], st["eval_only"], st["sets"]) == (
        "stores", "configs/study/study-t06.json", True, ["optim.lr=0.0002", "schedule.max_steps=20000"])
    # every item may be dropped by the no-start rule (a report-only box; in the chain: the stage-1 sub-deadline)
    assert all(v["max_hours"] for v in it.values()) and all(v["droppable"] for v in it.values())
    assert not any(v["kind"] in ("train", "readout") for v in it.values())
    uses_store = [n for n, v in it.items() if v["kind"] == "speed" or "{cache_dir}" in " ".join(v.get("argv") or [])]
    assert uses_store and all("stores-eval" in it[n]["needs"] for n in uses_store)

    def specs(n):
        return [{k: v for k, v in s.items() if not k.startswith("_")} for s in it[n]["verdict"]]

    assert it["selftest"]["argv"] == ["{python}", "-m", "kitsune.quant", "selftest", "--device", "cuda:0", "--out",
                                      "{out}/selftest.json", "--ckpt", "{ckpt:study-p03}", "--ckpt",
                                      "{ckpt:study-t06}"]
    assert specs("selftest") == [{"check": "12", "json": "{out}/selftest.json", "path": "ok", "equals": True}]
    for x in STUDENTS:
        n = f"fp16-study-{x}"
        assert it[n]["argv"] == ["{python}", "scripts/05_evaluate.py", "--config", f"{{config:study-{x}}}", "--ckpt",
                                 f"{{ckpt:study-{x}}}", "--quant", "fp16", "--out", "{out}", "--tables",
                                 "{out}/tables", "--manifest", "{manifest}", "--cache-dir", "{cache_dir}",
                                 "--max-temp", "0"]
        assert specs(n) == [{"check": "13", "json": "{out}/study.json", "path": "quant.nonfinite.rows", "equals": 0}]
    # F2 / F4: the selftest carries the compile, zero-row, parity and padded sub-checks (0.5 h); fp8-w8a8's trio
    # (F1's bf16 weight scales, file against memory, padded batches through torchao) joined the other three, and every
    # trio's quant-* and mem-* readout must show no non-finite row (check 13: where pre-F1 fp8 made NaN)
    assert it["selftest"]["max_hours"] == 0.5
    nonfinite = [{"check": "13", "json": "{out}/study.json", "path": "quant.nonfinite.rows", "equals": 0}]
    for f in ("int8-w8a8", "nvfp4-w4a4", "mxfp4-w4a4", "fp8-w8a8"):
        q, m, c = f"quant-{f}-study-p03", f"mem-{f}-study-p03", f"cmp-{f}-study-p03"
        assert it[q]["argv"] == quant_argv(f, "{config:study-p03}", "{ckpt:study-p03}")
        assert "--quant" in it[m]["argv"] and it[m]["argv"][it[m]["argv"].index("--quant") + 1] == f
        assert specs(q) == specs(m) == nonfinite, f
        assert it[c]["argv"] == ["{python}", "-m", "kitsune.quant", "compare", f"{{out:{q}}}", f"{{out:{m}}}",
                                 "--exact", "--json-out", "{out}/compare.json"]
        assert set(it[c]["needs"]) == {q, m}  # the {out:<item>} placeholders' implicit needs
        assert specs(c) == [{"check": "14", "json": "{out}/compare.json", "path": "same", "equals": True}]
    emu = it["emu-nvfp4-w4a4-study-p03"]["argv"]
    assert emu[emu.index("--quant-impl") + 1] == "emulate" and emu[emu.index("--sets") + 1:][:2] == ["eval_jsut",
                                                                                                   "eval_cv8"]
    cmp = it["cmp-emu-nvfp4-w4a4"]
    assert "--tol-cer" in cmp["argv"] and cmp["argv"][cmp["argv"].index("--tol-cer") + 1] == "0.001"
    assert set(cmp["needs"]) == {"mem-nvfp4-w4a4-study-p03", "emu-nvfp4-w4a4-study-p03"}
    assert specs("cmp-emu-nvfp4-w4a4") == [{"check": "12", "json": "{out}/compare.json", "path": "same",
                                            "equals": True}]
    assert it["whisper-large-v3"]["argv"] == whisper_argv("whisper-large-v3", "--sets", "eval_jsut")
    assert specs("whisper-large-v3") == [{"check": "15", "json": "{out}/whisper.json",
                                          "path": "sets.eval_jsut.cer_corpus", "min": 0.051, "max": 0.091}]
    for k in WHISPER_KEYS[1:]:
        assert it[k]["argv"] == whisper_argv(k, "--limit-per-set", "50") and specs(k) == [{"check": "15"}]
    speed = {n: v for n, v in it.items() if v["kind"] == "speed"}
    assert all(specs(n) == [{"check": "16"}] for n in speed)  # check 16: every speed item done (not_needed: null)
    quant_sys = {f"study-{x}@{f}" for x in STUDENTS for f in QUANT_FORMATS if f != "mxfp4-w4a4"}
    compile_sys = {"study-p03+compile", "study-p03@int8-w8a8+compile", "study-p03@nvfp4-w4a4+compile"}
    retime = {"study-t06", "study-p03", "study-p01", "study-p005", "cohere", "parakeet-ctc", "parakeet-tdt"}
    assert {v["system"] for v in speed.values()} == quant_sys | set(WHISPER_KEYS) | compile_sys | retime
    assert len(speed) == 24 + 4 + 3 + 7
    for n, v in speed.items():
        assert n == "speed-" + v["system"].replace("@", "-").replace("+", "-")  # contract 1.1's item names
        sysname, fmt = v["system"], None
        if "@" in sysname:
            fmt = sysname.split("@")[1].replace("+compile", "")
        if v["speed_kind"] == "whisper":  # the only form fullrun and the queue accept for a Whisper key
            assert v["args"] == ["--model", sysname, "--hf-cache", "{hf_cache}"] and not v["weights"]
        elif sysname in retime:
            assert v["only_if_new_machine"] == "full-smoke" and v["args"] == [], n
        else:
            assert v["only_if_new_machine"] is None
            want = (["--quant", fmt] if fmt else []) + (["--compile"] if sysname.endswith("+compile")
                                                        else ["--profile-kernels"])
            assert v["args"] == want, n


def test_every_tool_and_flag_an_item_runs_exists(reg):
    """launch's full_preflight refuses a box whose eval argv target is missing at the sha, or whose speed args name a
    flag speed_probe lacks; here, for this commit: every -m module or script, and every --flag of an eval argv or
    speed args, is in its tool's source."""
    sys.path.insert(0, str(ROOT / "vast"))
    import launch

    for box in fullrun.BOX_NAMES:
        for it in fullrun.box_items(box, reg):
            if it["kind"] == "eval":
                target = launch.argv_target(it["argv"])
                src = ROOT / target
                assert src.is_file(), (box, it["name"], target)
                text = src.read_text(encoding="utf-8")
                if target == "kitsune/quant.py":
                    text += (ROOT / "scripts" / "05_evaluate.py").read_text(encoding="utf-8")
                flags = [a for a in it["argv"] if re.fullmatch(r"--[a-z][a-z0-9-]*", a)]
            elif it["kind"] == "speed":
                text = (ROOT / "tools" / "speed_probe.py").read_text(encoding="utf-8")
                flags = [a for a in it["args"] if a.startswith("--")]
                assert f'"{it["speed_kind"]}"' in text
            else:
                continue
            for f in flags:
                assert f'"{f}"' in text, (box, it["name"], f)
    from kitsune.quant import QUANT_FORMATS as QF
    from kitsune.whisper import WHISPER_MODELS

    assert tuple(QF) == QUANT_FORMATS and set(WHISPER_KEYS) <= set(WHISPER_MODELS)


def test_every_items_command_line_is_one_its_tool_accepts(tmp_path, reg, trainer):
    """The command line kitsune.full_queue builds for every item of every box (FullQueue.argv_for, its placeholders
    filled as on a box once the items it reads have run) is one its tool's own parser accepts: 05's, kitsune.quant's,
    whisper_eval's and speed_probe's (with their checks: a --quant system ends in @<fmt>, --compile in +compile, a
    Whisper key names its system); a trainer's config loads with the queue's --set values (run name, both repos)."""
    import speed_probe
    import whisper_eval

    from kitsune import full_queue as F
    from kitsune import quant

    ev05 = load_script("05_evaluate")
    stamp, steps = "20261001T000000Z", 1234
    for box in fullrun.BOX_NAMES:
        spec = fullrun.box_spec(box, reg)
        s = F.FullSettings(root=ROOT, state_dir=tmp_path / box, gpus=[str(i) for i in range(spec["gpus"])],
                           n_gpus=None, python="python", out_repo="Multy123/kitsune-runs",
                           scratch_repo="Multy123/kitsune-scratch", uploader=object(), machine_id=None,
                           proc_root=tmp_path / "no-proc", cgroup=tmp_path / "no-cgroup")
        q = F.FullQueue(box, s, registry=reg)
        q.register()
        for name in q.order:  # every item has run: the run dirs, results and out dirs the placeholders read
            it, sp = q.item(name), q.spec_of(name)
            if sp["kind"] == "train":
                it.update(run_dir=f"runs/{name}-{stamp}", result=dict(steps=steps))
            elif sp["kind"] == "speed":
                it["out"] = f"runs/speed-{box}-{stamp}"
            else:
                it["out"] = f"runs/{'m4-' + name if sp['kind'] == 'readout' else name}-{stamp}"
            if sp.get("of_box"):
                it["source"] = dict(box=sp["of_box"], run_id=f"{sp['of']}-{stamp}", steps=steps)
        seen = set()
        for name in q.order:
            sp = q.spec_of(name)
            argv = q.argv_for(name, q.item(name), None)
            assert not any(re.search(r"\{[a-z_]+(:[^{}]*)?\}", a) for a in argv), (box, name, argv)
            if sp["kind"] == "train":
                sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
                assert spec["timed_states"] and sets == [f"run_name={name}", "hf.output_repo=Multy123/kitsune-runs",
                                                         "hf.scratch_repo=Multy123/kitsune-scratch"], (box, name)
                c = trainer.load_config(str(ROOT / argv[argv.index("--config") + 1]), sets)
                assert c["run_name"] == name and c["hf"]["scratch_repo"] == "Multy123/kitsune-scratch"
            elif sp["kind"] == "stores":
                assert argv[1:4] == ["-m", "kitsune.full_queue", "build-stores"]
            elif sp["kind"] == "speed":
                a = speed_probe.parse_args(argv[[x.endswith("speed_probe.py") for x in argv].index(True) + 1:])
                assert (a.kind, a.system) == (sp["speed_kind"], sp["system"]), (box, name)
            elif "-m" in argv and argv[argv.index("-m") + 1] == "kitsune.quant":
                quant.parse_args(argv[argv.index("-m") + 2:])
            elif any(x.endswith("whisper_eval.py") for x in argv):
                a = whisper_eval.parse_args(argv[[x.endswith("whisper_eval.py") for x in argv].index(True) + 1:])
                assert a.system == name
            else:  # a readout or a 05 eval
                ev05.parse_args(argv[[x.endswith("05_evaluate.py") for x in argv].index(True) + 1:])
            seen.add(sp["kind"])
        assert seen == ({"stores", "train", "readout", "speed"} if box == "full-smoke" else
                        {"stores", "train", "readout"} if box == "p01" else
                        {"stores", "train", "readout", "eval", "speed"} if box == "full" else
                        {"stores", "eval", "speed"}), box


# ================================================================================================ the chained box

CHAIN = "p01-chain"
# contract addendum E.1.7's entry (DECISIONS D), its _comment aside
CHAIN_ENTRY = {"est_hours": 25.2, "max_hours": 35, "max_dph": 1.00, "extra_gb": 120, "gate": True,
               "chain": [{"parts": ["full-smoke", "smoke-b"], "gate_box": "full-smoke",
                          "rebuild": "configs/full/data-smoke.json", "gate_by_hours": 9, "max_hours": 10.5},
                         {"parts": ["p01"], "rebuild": "configs/full/data-p01.json"}]}
STUDY_SELECTION = "labels/full/selections/study_1000h.parquet"  # smoke-b's frozen study selection (data-smoke-b)
SIDECAR = {sel: sel[:-len(".parquet")] + ".json" for sel in (fullrun.FULL_SELECTION, fullrun.SMOKE_SELECTION,
                                                               STUDY_SELECTION)}
STAGE_1_ENV = {"KITSUNE_N_GPUS": "1", "KITSUNE_WATCHDOG_HB_FILE": "train_hb", "KITSUNE_WATCHDOG_ORPHAN_S": "600",
               "KITSUNE_WATCHDOG_ORPHAN_ACTION": "alert", "KITSUNE_CHAIN_STAGE": "1",
               "KITSUNE_WATCHDOG_HANDOVER_S": "34200"}  # 3600 x (gate_by_hours 9 + 0.5)


def test_the_chained_box(reg):
    """p01-chain as committed (contract addendum E.1.7): the addendum's entry, valid in the committed registry (the reg
    fixture's load_registry read every config: E.1.4's rules, each part's extent within its stage's rebuild config,
    stage 1's within stage 2's), kept in its normalised raw form (a loaded registry loads again unchanged), and what
    kitsune.fullrun derives from it: the spec launch rents by (E.1.5), both stage views (E.1.6, E.2.1), the readers and
    the stage env."""
    from fixtures_chain import CHAIN as TEST_ENTRY  # tests/test_full_chain.py's controller runs on this entry too

    raw = json.loads((FULL / "boxes.json").read_text(encoding="utf-8"))["boxes"][CHAIN]
    assert {k: v for k, v in raw.items() if k != "_comment"} == CHAIN_ENTRY and "addendum E" in raw["_comment"]
    assert {k: v for k, v in TEST_ENTRY.items() if k != "_comment"} == CHAIN_ENTRY
    assert reg["boxes"][CHAIN] == raw, "the entry writes every field it has, so validation fills nothing in"
    assert fullrun.load_registry(reg) == reg and fullrun.registry_problems(reg) == []
    assert [b for b in reg["boxes"] if fullrun.is_chain(b, reg)] == list(fullrun.CHAIN_NAMES) == [CHAIN]
    manifest, smoke_side, full_side = fullrun.FROZEN_MANIFEST, SIDECAR[fullrun.SMOKE_SELECTION], SIDECAR[
        fullrun.FULL_SELECTION]
    # E.1.5: the derived spec (computed, never stored)
    spec = fullrun.box_spec(CHAIN, reg)
    assert {k: v for k, v in spec.items() if k != "chain"} == dict(
        gpus=1, data_config="configs/full/data-p01.json", est_hours=25.2, max_hours=35, max_dph=1.0, extra_gb=120,
        gate=True, watchdog={"orphan_s": 600, "action": "alert"}, deadline_reserve_min=45, timed_states=True,
        extra_files=[manifest, smoke_side, STUDY_SELECTION, SIDECAR[STUDY_SELECTION], full_side],
        extra_dirs=[PARAKEET], smoke=False, faults=[], items=[], max_attempts=4)
    s1_configs = {"full-smoke": "configs/full/data-smoke.json", "smoke-b": "configs/full/data-smoke-b.json"}
    assert spec["chain"] == fullrun.chain_stages(CHAIN, reg) == [
        dict(stage=1, parts=["full-smoke", "smoke-b"], gate_box="full-smoke", gate_by_hours=9, max_hours=10.5,
             rebuild="configs/full/data-smoke.json", data_configs=s1_configs,
             watchdog={"orphan_s": 600, "action": "alert"}),
        dict(stage=2, parts=["p01"], gate_box=None, gate_by_hours=None, max_hours=None,
             rebuild="configs/full/data-p01.json", data_configs={"p01": "configs/full/data-p01.json"},
             watchdog={"orphan_s": 3600, "action": "stop"})]
    # E.1.6 / E.2.1: the stage views (bootstrap pulls and check-students checks one stage's)
    v1, v2 = fullrun.stage_view(CHAIN, 1, reg), fullrun.stage_view(CHAIN, 2, reg)
    assert v1 == dict(stage=1, parts=["full-smoke", "smoke-b"], gate_box="full-smoke",
                      rebuild="configs/full/data-smoke.json", data_configs=s1_configs,
                      students=[f"students/study/{x}" for x in STUDENTS],
                      ctc_students=[f"students/study/{x}" for x in STUDENTS if x.startswith("p")],
                      extra_files=[manifest, smoke_side, STUDY_SELECTION, SIDECAR[STUDY_SELECTION]],
                      extra_dirs=[PARAKEET], timed_states=True, watchdog={"orphan_s": 600, "action": "alert"})
    assert v2 == dict(stage=2, parts=["p01"], gate_box=None, rebuild="configs/full/data-p01.json",
                      data_configs={"p01": "configs/full/data-p01.json"}, students=["students/study/p01"],
                      ctc_students=["students/study/p01"], extra_files=[manifest, full_side], extra_dirs=[],
                      timed_states=True, watchdog={"orphan_s": 3600, "action": "stop"})
    # the files are the data configs' own: each part's selection sidecar, and smoke-b's frozen study selection, which
    # stage 1's rebuild (the smoke selection) does not bring
    assert (cfg("data-smoke")["selection"], cfg("data-p01")["selection"], cfg("data-smoke-b")["selection"]) == (
        fullrun.SMOKE_SELECTION, fullrun.FULL_SELECTION, STUDY_SELECTION)
    assert fullrun.box_extra_files("full-smoke", reg) + [STUDY_SELECTION, SIDECAR[STUDY_SELECTION]] == v1["extra_files"]
    assert fullrun.box_extra_files("p01", reg) == v2["extra_files"]
    # the readers: stage N's view, or (launch) the union of both
    for s, v in ((1, v1), (2, v2)):
        assert fullrun.box_students(CHAIN, reg, stage=s) == v["students"]
        assert fullrun.box_ctc_students(CHAIN, reg, stage=s) == v["ctc_students"]
        assert fullrun.box_extra_files(CHAIN, reg, stage=s) == v["extra_files"]
        assert fullrun.box_extra_dirs(CHAIN, reg, stage=s) == v["extra_dirs"]
    assert fullrun.box_students(CHAIN, reg) == v1["students"]  # box 1's student is smoke A's too
    assert (fullrun.box_extra_files(CHAIN, reg), fullrun.box_extra_dirs(CHAIN, reg)) == (spec["extra_files"],
                                                                                        spec["extra_dirs"])
    assert fullrun.box_configs(CHAIN, reg) == [
        "configs/full/data-smoke.json", "configs/full/smoke-p03.json", "configs/full/smoke-t06.json",
        "configs/full/smoke-p01.json", "configs/full/smoke-p005.json", fullrun.BOXES_FILE,
        "configs/full/data-smoke-b.json", "configs/study/study-t06.json", "configs/full/data-p01.json",
        "configs/full/full-p01.json"]
    # the stage env: launch's (stage 1: full-smoke's alert watchdog and the hand-over bound), the controller's stage-2
    # bootstrap (stop 3600)
    assert fullrun.box_env(CHAIN, reg) == fullrun.box_env(CHAIN, reg, stage=1) == STAGE_1_ENV
    assert fullrun.box_env(CHAIN, reg, stage=2) == {
        "KITSUNE_N_GPUS": "1", "KITSUNE_WATCHDOG_HB_FILE": "train_hb", "KITSUNE_WATCHDOG_ORPHAN_S": "3600",
        "KITSUNE_WATCHDOG_ORPHAN_ACTION": "stop", "KITSUNE_CHAIN_STAGE": "2"}
    for fn in (fullrun.box_items, fullrun.train_items):  # a chain has no items: its controller runs its parts' queues
        with pytest.raises(fullrun.RegistryError, match="chain controller"):
            fn(CHAIN, reg)


def test_the_chained_box_numbers(reg):
    """The chain's hours, disk and price against its parts (addendum E.1.4 rule 3, E.6, E.9.5): box 1 fits after
    stage 1's sub-deadline, launch's floor is 30 h, the 35 h cap covers the worst case the gate lets through, 25.2 h is
    the central wall, the gate part ends by its standalone cap, the extra disk holds box 1's and stage 1's leftovers."""
    from kitsune import full_queue as F

    spec, parts = fullrun.box_spec(CHAIN, reg), {b: fullrun.box_spec(b, reg) for b in ("full-smoke", "smoke-b", "p01")}
    s1 = spec["chain"][0]
    readout = sum(it["max_hours"] for it in parts["p01"]["items"] if it["kind"] == "readout")  # m4-full-p01 0.3
    assert spec["max_hours"] - s1["max_hours"] >= parts["p01"]["est_hours"]  # rule 3: 35 - 10.5 = 24.5 >= 19.5
    assert s1["max_hours"] + parts["p01"]["est_hours"] == 30  # launch refuses a --max-hours below this
    assert s1["gate_by_hours"] == parts["full-smoke"]["max_hours"] == 9 < s1["max_hours"]
    # E.6: stage 1 + the pessimistic stage-2 setup 6.1 h [X] + smoke check 3's box-1 limit + readout + end 0.35 h +
    # p01's reserve; E.9.5's central wall: smoke A 5.2 + smoke B's items 1.1 + stage-2 setup 4.7 + full-p01 13.56 +
    # readout + end (the plan's hours [X])
    assert math.isclose(s1["max_hours"] + 6.1 + F.BOX1_H_MAX + readout + 0.35
                        + parts["p01"]["deadline_reserve_min"] / 60, spec["max_hours"])
    assert math.isclose(5.2 + 1.1 + 4.7 + 13.56 + readout + 0.35, spec["est_hours"], abs_tol=0.05)
    assert spec["extra_gb"] - parts["p01"]["extra_gb"] == 95  # stage 1's leftovers on box 1's disk
    assert spec["max_dph"] == parts["p01"]["max_dph"] == parts["full-smoke"]["max_dph"] == 1.0
    # E.1.4 rule 2: only the gate part injects faults or merely alerts
    assert [b for b, p in parts.items() if p["faults"] or p["watchdog"]["action"] != "stop"] == ["full-smoke"]


def test_check_holds_the_chain(tmp_path, capsys):
    """make_full_configs --check validates the chain with the rest of the registry (fullrun.registry_problems)."""
    out = repo_copy(tmp_path)
    reg = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    reg["boxes"][CHAIN]["max_hours"] = 29
    reg["boxes"]["smoke-b"]["watchdog"]["action"] = "alert"
    (out / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    assert sorted(M.check(out)) == sorted([
        f"boxes.json: boxes.{CHAIN}: max_hours 29 - stage 1's 10.5 leaves 18.5 h, below the last stage's est_hours "
        f"19.5",
        f"boxes.json: boxes.{CHAIN}.chain[0]: part 'smoke-b''s watchdog action is 'alert': every part but the gate "
        f"part runs with the watchdog in stop mode"])
    assert M.main(["--check", "--out-dir", str(out)]) == 1 and "2 problem(s)" in capsys.readouterr().out


def test_the_cli_shows_the_chained_box(reg, monkeypatch, capsys):
    """python -m kitsune.fullrun show --box p01-chain (this checkout's registry): the derived spec, both stage views and
    the stage-1 env; students / extra-files of a stage as bootstrap reads them ($KITSUNE_CHAIN_STAGE)."""
    monkeypatch.delenv(fullrun.ENV_REGISTRY, raising=False)
    monkeypatch.delenv(fullrun.ENV_CHAIN_STAGE, raising=False)
    assert fullrun.main(["show", "--box", CHAIN]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["spec"] == fullrun.box_spec(CHAIN, reg) and doc["env"] == STAGE_1_ENV
    assert doc["stages"] == {str(s): fullrun.stage_view(CHAIN, s, reg) for s in (1, 2)}
    assert doc["configs"] == fullrun.box_configs(CHAIN, reg) and doc["students"] == doc["stages"]["1"]["students"]
    monkeypatch.setenv(fullrun.ENV_CHAIN_STAGE, "2")
    assert fullrun.main(["students", "--box", CHAIN]) == 0 and capsys.readouterr().out.split() == [
        "students/study/p01"]
    assert fullrun.main(["extra-files", "--box", CHAIN]) == 0 and capsys.readouterr().out.split() == [
        fullrun.FROZEN_MANIFEST, SIDECAR[fullrun.FULL_SELECTION]]


def test_launch_resolves_the_chained_box(monkeypatch, capsys):
    """vast/launch.py --job full --box p01-chain --dry-run on this checkout's registry and configs (served as the
    commit a box runs; the vastai CLI and the Hub faked, nothing rented): addendum E.1.8. The boot's KITSUNE_CONFIG is
    stage 1's rebuild; the disk and the gate are sized on box 1's extent with the chain's extra_gb, the boot's rebuild
    bytes, pull bytes and timeout on stage 1's without it; the stage-1 watchdog env; 35 h, 25.2 h, $1.00; every part
    preflighted as a box with its own data config (the scratch repo only for parts with timed states), each distinct
    data config's selection checked, then the chain's own files."""
    sys.path.insert(0, str(ROOT / "vast"))
    import launch

    sha, image = "0123456789abcdef0123456789abcdef01234567", "ghcr.io/multysquid/kitsune-train@sha256:" + "ab" * 32
    data, runs, scratch = "Multy123/kitsune-data", "Multy123/kitsune-runs", "Multy123/kitsune-scratch"

    def at(s, rel):
        assert s == sha, s
        return (ROOT / rel).read_bytes()

    monkeypatch.setattr(launch, "git_show", at)
    monkeypatch.setattr(launch, "config_at", lambda s, c: json.loads(at(s, c)))
    monkeypatch.setattr(launch, "git", lambda *a: sha if a[:1] == ("rev-parse",) else "")
    monkeypatch.setattr(launch, "worktree_file", lambda rel: None)
    # the sizings kitsune.extent.sizing gives on the sealed labels/full record with the built selections' kept hours
    # (2026-10-01): box 1's extent with the chain's extra_gb 120 (1.1 x 1,324 GB -> 1,500 GB; addendum E.9.5 estimated
    # 1,450 with ~486 GB of selected audio, the selection keeps 492), and stage 1's study extent
    size = {fullrun.FULL_SELECTION: dict(down_gb=571.22, shard_gb=546.74, sel_gb=492.0, stores=1, labels_gb=38.3,
                                         hours=13028.5, disk_gb=1500, rebuild_timeout_min=388),
            fullrun.SMOKE_SELECTION: dict(down_gb=58.31, shard_gb=54.92, sel_gb=9.23, stores=2, labels_gb=3.41,
                                          hours=1160.7, disk_gb=250, rebuild_timeout_min=67)}
    seen = {"hf": [], "extent": [], "preflight": [], "chain": []}

    def hf_preflight(data_repo, out_repo, c):
        seen["hf"].append(c["selection"])
        return "d" * 40, []

    def extent_preflight(data_repo, rev, c, extra_gb=0.0):
        seen["extent"].append((c["selection"], extra_gb))
        return [], dict(size[c["selection"]], extra_gb=extra_gb)

    def full_preflight(*a, **kw):
        seen["preflight"].append((a, kw))
        return [], [f"{a[5]} preflight ok"]

    def vastai(exe, args):
        assert args[:2] == ["search", "offers"], f"only the offer search runs look-only, not {args[:2]}"
        return json.dumps([{"id": 1, "machine_id": 54650, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 1,
                            "dph_total": 0.816, "reliability": 0.99, "verification": "verified",
                            "duration": 30 * 86400, "cpu_ram": 64439, "disk_space": 1568, "inet_down_cost": 0.00117,
                            "inet_up_cost": 0.001, "storage_cost": 0.1}])

    monkeypatch.setattr(launch, "hf_preflight", hf_preflight)
    monkeypatch.setattr(launch, "extent_preflight", extent_preflight)
    monkeypatch.setattr(launch, "full_preflight", full_preflight)
    monkeypatch.setattr(launch, "chain_preflight",
                        lambda *a, **kw: seen["chain"].append((a, kw)) or ([], ["chain preflight ok"]))
    monkeypatch.setattr(launch, "avoided_machines", lambda data_repo, rev: (set(), []))
    monkeypatch.setattr(launch, "gate_refusals", lambda out_repo: ({}, []))
    monkeypatch.setattr(launch, "vastai", vastai)
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/fake/vastai" if name == "vastai" else None)
    rc = launch.main(["--job", "full", "--box", CHAIN, "--data-repo", data, "--out-repo", runs, "--scratch-repo",
                      scratch, "--sha", sha, "--image", image, "--skip-git-checks", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "not creating anything (--dry-run)" in out, out
    create = out.split("create command:\n  vastai ", 1)[1].splitlines()[0]  # as printed look-only
    cargs = shlex.split(create)
    env_value = cargs[cargs.index("--env") + 1]
    env = dict(p.split("=", 1) for p in env_value.split(" ")[1::2])
    full, smoke = size[fullrun.FULL_SELECTION], size[fullrun.SMOKE_SELECTION]
    assert env == {
        "KITSUNE_JOB": "full", "KITSUNE_BOX": CHAIN, "KITSUNE_SHA": sha, "KITSUNE_CONFIG": "configs/full/data-smoke.json",
        "KITSUNE_DATA_REPO": data, "KITSUNE_OUT_REPO": runs, **STAGE_1_ENV, "KITSUNE_SCRATCH_REPO": scratch,
        "KITSUNE_GATE_BYTES": str(int(1e9 * max(full["down_gb"], 571.2))), "KITSUNE_GATE_MAX_H": "5",
        "KITSUNE_REBUILD_BYTES": str(int(1e9 * smoke["down_gb"])),
        "KITSUNE_PULL_BYTES": str(int(1e9 * (smoke["labels_gb"] + 2))), "KITSUNE_MAX_HOURS": "35", "TZ": "UTC",
        "KITSUNE_DATA_REVISION": "d" * 40, "KITSUNE_REBUILD_TIMEOUT_MIN": str(smoke["rebuild_timeout_min"]),
        "KITSUNE_DPH": "0.8160", "KITSUNE_MACHINE_ID": "54650"}
    assert cargs[cargs.index("--disk") + 1] == "1500" and "HF_TOKEN" not in create
    assert "disk_space>=1500" in out, "the offer search filters on box 1's disk (the machine's free disk)"
    assert cargs[cargs.index("--label") + 1] == f"kitsune-full-{CHAIN}-data-smoke-{sha[:7]}"
    # the sizing inputs: box 1's extent with the chain's extra_gb 120, stage 1's with none
    assert sorted(seen["extent"]) == sorted([(fullrun.FULL_SELECTION, 120.0), (fullrun.SMOKE_SELECTION, 0.0)])
    assert seen["hf"] == [fullrun.SMOKE_SELECTION, STUDY_SELECTION, fullrun.FULL_SELECTION]
    parts = {a[5]: (a, kw) for a, kw in seen["preflight"]}
    assert list(parts) == ["full-smoke", "smoke-b", "p01"]
    assert {p: (a[3], a[7]["selection"]) for p, (a, _) in parts.items()} == {
        "full-smoke": (scratch, fullrun.SMOKE_SELECTION), "smoke-b": (None, STUDY_SELECTION),
        "p01": (scratch, fullrun.FULL_SELECTION)}
    assert len(seen["chain"]) == 1 and seen["chain"][0][0][3:4] == (CHAIN,)
    assert "x ~25.2 h (box p01-chain; watchdog cap 35 h)" in out and "part smoke-b: smoke-b preflight ok" in out
    assert "chain p01-chain: the gate part must end by first boot + 9 h" in out and "+ 9.5 h" in out
    assert "stage 1 ends by + 10.5 h" in out and "chain stage 1 (configs/full/data-smoke.json)" in out


@pytest.mark.skipif(not os.environ.get("KITSUNE_FULL_SELECTION_DIR"),
                    reason="KITSUNE_FULL_SELECTION_DIR (a local labels/full/selections/full_study) not set")
def test_the_real_selections_fit_the_configs(plan):
    """launch's selection check of every data and train config against the built selection it names, the sidecars'
    own checks, and the plan record's sha256 = the selections'."""
    import hashlib

    sys.path.insert(0, str(ROOT / "vast"))
    import launch

    from kitsune import devslice

    base = Path(os.environ["KITSUNE_FULL_SELECTION_DIR"])
    for which in ("full", "smoke"):
        h = hashlib.sha256((base / f"{which}.parquet").read_bytes()).hexdigest()
        assert h == plan[which]["selection"]["sha256"]
        assert devslice.sidecar_problems(json.loads((base / f"{which}.json").read_text(encoding="utf-8")),
                                         selection_sha256=h, manifest_sha256=fullrun.FROZEN_MANIFEST_SHA256) == []
    for n in GENERATED:
        c = cfg(n)
        if c["selection"].startswith(fullrun.FULL_DIR):
            sel = base / c["selection"].rsplit("/", 1)[-1]
            assert launch.selection_problems(sel, sel.name, c) == [], n
