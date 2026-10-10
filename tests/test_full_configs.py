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
             + ["data-p01", "data-t", "data-p", "data-full", "data-smoke", "data-smoke-b"])
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
MEASURED_T = {"t06": 71946, "p03": 105861, "p01": 107910, "p005": 107910}  # at SMOKE_EPOCHS 3 / 3 / 4 / 4
# tools/full_plan.py on full.parquet at the boxes' epochs (the plan record's launch part, measured again 2026-10-10 for
# DECISIONS H15): T-0.6B 6 (8 under H7: 191,855 steps), P-0.3B 6, P-0.05B 10 and P-0.1B 4 - its cooldown re-run's
# (DECISIONS H13: the full part's 4-epoch measurement on the same file; H9's continuation had 8); every epoch's plan as
# long as the full part's at the same epoch (each run's first 4 epochs are its 4-epoch plan, epoch for epoch)
LAUNCH_EPOCHS = {"t06": 6, "p03": 6, "p01": 4, "p005": 10}  # P-0.3B 6: DECISIONS H5; T-0.6B 6: H15; P-0.05B 10: H9
LAUNCH_T = {"t06": 143891, "p03": 211720, "p01": 107910, "p005": 269777}  # P-0.1B 4: H13
# box_hours of the committed speed record (box 1 at c4604304: r 1.0361, o 0.0465), per HOURS_BOXES box, all at
# AUGMENT_STEP_FACTOR 1.05 (concat): full-t06 = T-0.6B with the AED recipe at 6 epochs (DECISIONS H15; 63.62 h at H7's
# 8); box p005's full-p005 =
# P-0.05B from step 0 at 10 epochs, the P test box (15.76 h, 15.03 without the factor; H9, H10); box p-cool's three = the
# cooldown re-runs of P-0.3B's, P-0.1B's and P-0.05B's runs from their pre_cooldown states, 42,344, 21,582 and 13,490
# steps (DECISIONS H14)
RUN_H = {"full-t": {"full-t06": 47.75}, "p005": {"full-p005": 15.76},
         "p-cool": {"full-p03": 5.39, "full-p01": 2.11, "full-p005": 1.01}}
BOX_H = {"full-t": (50.9, 68), "p005": (18.5, 27), "p-cool": (11.8, 19)}
# boxes full-p and p01 left HOURS_BOXES with DECISIONS H14 (their runs continue on box p-cool): their entries are
# origin/main's again, the records of the 6-epoch run (box full-p) and of H9's 8-epoch plan (box p01), their hours the
# speed record's of then, no longer held by --check
RECORD_RUN_H = {"full-p": {"full-p03": 26.53}, "p01": {"full-p01": 11.47}}
RECORD_BOX_H = {"full-p": (30.0, 42), "p01": (14.3, 22)}
P01_RID = "full-p01-20261001T184145Z"
P03_RID = "full-p03-20261003T230143Z"
P005_RID = "full-p005-20261008T160232Z"
# DECISIONS H12, H14: the cooldown re-runs' sets as the queue passes them to the trainer (fullrun.continue_sets of the
# registry's continues blocks, after schedule.resume_reset=true; one spelling each): the run's own epochs, 4d off,
# P-0.05B keeps its early-stop cooldown, then one per key of its recipe - recipe v4's ...
V4_SETS = ["augment.enabled=true", "augment.truncate_p=0.2", "augment.concat_p=0.5", "augment.mix_p=0.0",
           "augment.truncate_min_row_s=3.0", "augment.end_trim_p=0.3", "augment.noise_p=0.3",
           "augment.noise_bank=aug/musan-bg-v2", f"augment.noise_bank_sha256={M.NOISE_BANK_SHA256}",
           "augment.speech_p=0.15", "augment.reverb_p=0.2", "augment.rir_bank=aug/rirs-v1",
           f"augment.rir_bank_sha256={M.RIR_BANK_SHA256}", "augment.gain_p=0.3", "augment.codec_p=0.1"]
# ... and its ranges, written out: RECIPE_V4_FULL's the trainer's defaults, RECIPE_GENTLE's milder
FULL_RANGES = ["augment.noise_snr_db=[0.0,20.0]", "augment.speech_snr_db=[10.0,25.0]", "augment.gain_db=[-20.0,10.0]",
               'augment.codecs=["mp3","gsm","ulaw8k"]', "augment.background_min_row_s=0.0"]
GENTLE_RANGES = ["augment.noise_snr_db=[5.0,20.0]", "augment.speech_snr_db=[15.0,25.0]", "augment.gain_db=[-10.0,10.0]",
                 'augment.codecs=["mp3","ulaw8k"]', "augment.background_min_row_s=3.0"]
COOL_SETS = {"p03-cool": ["schedule.epochs=6", "schedule.deadline_cooldown=false", *V4_SETS, *GENTLE_RANGES],
             "p01-cool": ["schedule.epochs=4", "schedule.deadline_cooldown=false", *V4_SETS, *FULL_RANGES],
             "p005-cool": ["schedule.epochs=10", "schedule.deadline_cooldown=false",
                           "schedule.resume_reset_keep_cooldown=true", *V4_SETS, *GENTLE_RANGES]}


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
    smoke A's plan_total_steps / plan_hours, F4's seconds), whose hours of the HOURS_BOXES (full-t, p005, p-cool) differ
    from the speed record's (box_hours), or whose readout of a full box does not fit in its run's end reserve
    (readout_reserve_problems; smoke boxes are exempt). The records of boxes full-p and p01 (DECISIONS H14) are not
    held to the speed record."""
    out = repo_copy(tmp_path)
    reg = json.loads((out / "boxes.json").read_text(encoding="utf-8"))

    def item(box, name):
        return next(it for it in reg["boxes"][box]["items"] if it["name"] == name)

    item("full-smoke", "smoke-t06")["plan_total_steps"] = 73452
    item("full-smoke", "smoke-p005")["plan_hours"] = 9.92
    next(f for f in reg["boxes"]["full-smoke"]["faults"] if f["id"] == "F4")["seconds"] = 140
    item("p-cool", "full-p01")["max_hours"] = 13.24  # box 1's plan hours: the re-run's are the speed record's
    item("full-t", "full-t06")["max_hours"] = 30.5  # not the speed record's
    reg["boxes"]["p-cool"]["max_hours"] = 90
    item("p01", "full-p01")["max_hours"] = 13.24  # the records: not held
    reg["boxes"]["full-p"]["max_hours"] = 90
    item("full-p", "m4-full-p03")["max_hours"] = 0.4  # 24 + 10 min > full-p03's 30
    item("full-smoke", "m4-smoke-p03")["max_hours"] = 0.4  # smoke-p03's 2 min: exempt
    (out / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    probs = M.check(out)
    want = ["boxes.full-smoke.items.smoke-t06: {'plan_total_steps': 73452, 'plan_hours': 34.47}, the plan record "
            "gives {'plan_total_steps': 71946, 'plan_hours': 34.47}",
            "boxes.full-smoke.items.smoke-p005: {'plan_total_steps': 107910, 'plan_hours': 9.92}, the plan record "
            "gives {'plan_total_steps': 107910, 'plan_hours': 9.69}",
            "boxes.full-smoke.faults.F4: seconds 140, the plan record gives 150 (bound 151.05 s)",
            "boxes.p-cool.items.full-p01: {'max_hours': 13.24}, the speed record gives {'max_hours': 2.11}",
            "boxes.full-t.items.full-t06: {'max_hours': 30.5}, the speed record gives {'max_hours': 47.75}",
            "boxes.p-cool: {'est_hours': 11.8, 'max_hours': 90}, the speed record gives {'est_hours': 11.8, "
            "'max_hours': 19}",
            "boxes.full-p.items.m4-full-p03: max_hours 0.4 (24 min) + 10 min of the trainer's end phase exceed "
            "configs/full/full-p03.json's schedule.end_reserve_min 30"]
    assert len(probs) == len(want) and all(any(p.startswith(f"boxes.json: {w}") for p in probs) for w in want), probs
    # the trainer's default end reserve on full-t06 (contract 7 as written): its 45 min readout no longer fits
    shutil.copyfile(FULL / "boxes.json", out / "boxes.json")
    c = cfg("full-t06")
    c["schedule"]["end_reserve_min"] = 30
    (out / "full-t06.json").write_text(json.dumps(c), encoding="utf-8")
    assert sorted(M.check(out)) == sorted([
        "full-t06.json: differs from the generator's",
        "boxes.json: boxes.full-t.items.m4-full-t06: max_hours 0.75 (45 min) + 10 min of the trainer's end phase exceed "
        "configs/full/full-t06.json's schedule.end_reserve_min 30: after a run the deadline cooldown shortened, the "
        "no-start rule would skip the readout (make_full_configs READOUT_RESERVE)"])


def test_check_holds_the_continues_blocks(tmp_path, monkeypatch):
    """--check refuses a registry whose continues blocks differ from the generator's (continues_problems: box p-cool's
    train items carry continues_block of CONTINUATIONS, its sets in the same order and spelling), a continuation the
    registry lacks, a continues item CONTINUATIONS does not name and a CONTINUATIONS box the registry does not have."""
    out = repo_copy(tmp_path)
    assert M.continues_problems(fullrun.load_registry(FULL / "boxes.json")) == []
    raw = json.loads((out / "boxes.json").read_text(encoding="utf-8"))

    def write_and_check(change) -> list[str]:
        reg = copy.deepcopy(raw)
        change({it["name"]: it for it in reg["boxes"]["p-cool"]["items"]})
        (out / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
        return M.check(out)

    def moved(it, key, to_end=True):
        sets = it["continues"]["sets"]
        v = sets.pop(key)
        it["continues"]["sets"] = {**sets, key: v} if to_end else {key: v, **sets}

    w = "boxes.json: boxes.p-cool.items."
    assert write_and_check(lambda its: its["full-p005"]["continues"].update(from_step=67450)) == [
        f"{w}full-p005.continues differs from make_full_configs continues_block('p005-cool') in ['from_step']"]
    assert write_and_check(lambda its: its["full-p03"]["continues"].update(resets_before=1, box="p005")) != []
    assert write_and_check(lambda its: its["full-p01"]["continues"].update(resets_before=2)) == [
        f"{w}full-p01.continues differs from make_full_configs continues_block('p01-cool') in ['resets_before']"]
    # the sets in another order, or with another value (the codecs of the gentle recipe in another order too)
    assert write_and_check(lambda its: moved(its["full-p03"], "schedule.epochs")) == [
        f"{w}full-p03.continues differs from make_full_configs continues_block('p03-cool') in ['sets']"]
    assert write_and_check(lambda its: its["full-p005"]["continues"]["sets"].update(
        {"augment.codecs": ["ulaw8k", "mp3"]})) == [
        f"{w}full-p005.continues differs from make_full_configs continues_block('p005-cool') in ['sets']"]
    assert write_and_check(lambda its: its["full-p005"]["continues"]["sets"].pop(
        "schedule.resume_reset_keep_cooldown")) == [
        f"{w}full-p005.continues differs from make_full_configs continues_block('p005-cool') in ['sets']"]
    # comment keys are no difference
    assert write_and_check(lambda its: its["full-p01"]["continues"].update(_comment="A1's re-save")) == []
    # a run the registry trains anew: the continuation is missing
    assert write_and_check(lambda its: its["full-p01"].update(continues=None)) == [
        "boxes.json: boxes.p-cool: no train item full-p01 with a continues block (make_full_configs CONTINUATIONS "
        "'p01-cool')"]
    (out / "boxes.json").write_text(json.dumps(raw), encoding="utf-8")
    assert M.check(out) == []
    # CONTINUATIONS without P-0.05B's: the registry's item is no generator's; one on a box the registry lacks
    reg = fullrun.load_registry(FULL / "boxes.json")
    monkeypatch.delitem(M.CONTINUATIONS, "p005-cool")
    assert M.continues_problems(reg) == [
        "boxes.p-cool.items.full-p005.continues: not a make_full_configs CONTINUATIONS entry"]
    monkeypatch.setitem(M.CONTINUATIONS, "p005-cool", dict(M.CONTINUATIONS["p01-cool"], box="p-hot", student="p005"))
    assert M.continues_problems(reg) == [
        "boxes.p-cool.items.full-p005.continues: not a make_full_configs CONTINUATIONS entry",
        "boxes.p-hot: not in the registry (make_full_configs CONTINUATIONS)"]


def test_continuations_follow_the_plan(plan, monkeypatch):
    """--check holds CONTINUATIONS to the plan record's launch part (continuation_plan_problems): a re-run that plans
    anew goes from the step before its planned cooldown to the plan's T, a kept early-stop cooldown ends by the plan's
    T, and each runs at least one step at its run's own epochs."""
    assert M.continuation_plan_problems(plan) == []
    c = M.CONTINUATIONS
    for name, change, want in (
            ("p01-cool", dict(to_step=215815),
             "CONTINUATIONS.p01-cool: to_step 215815, the plan record's T at 4 epochs is 107910"),
            ("p03-cool", dict(from_step=169000),
             "CONTINUATIONS.p03-cool: from_step 169000, the plan's pre_cooldown step is 169376"),
            ("p005-cool", dict(to_step=270000), "CONTINUATIONS.p005-cool: to_step 270000 is past the plan's T 269777: a "
             "kept early-stop cooldown ends before the planned end"),
            ("p005-cool", dict(from_step=80939), "CONTINUATIONS.p005-cool: from_step 80939 .. to_step 80939 is no run"),
            ("p01-cool", dict(epochs=8), "CONTINUATIONS.p01-cool: epochs 8, the run's own are 4 (FULL_RUNS)")):
        monkeypatch.setitem(c, name, dict(c[name], **change))
        assert want in M.continuation_plan_problems(plan), (name, M.continuation_plan_problems(plan))
        monkeypatch.undo()
    # --check reports them (it reads this module's CONTINUATIONS)
    monkeypatch.setitem(c, "p01-cool", dict(c["p01-cool"], to_step=215815))
    assert "CONTINUATIONS.p01-cool: to_step 215815, the plan record's T at 4 epochs is 107910" in M.check()


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
        # full-p03 trains with the recipe, full-t06 with the AED recipe and two of recipe v5's fixes (DECISIONS H15),
        # full-p005 with recipe v4 (validate_augment: a CTC student; an AED one with its cut table; the rest of the
        # block the defaults)
        want = {"full-p03": M.RECIPE, "full-t06": M.RECIPE_AED_H15, "full-p005": M.RECIPE_V4}.get(n)
        assert trainer.augment_on(c) == (want is not None), n
        if want is not None:
            assert c["augment"] == dict(trainer.DEFAULTS["augment"], **want)
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
    # DECISIONS H1 step 2 / H7: full-p03 and full-t06 also carry their recipe's augment keys; no smoke config does
    # (smoke A ran them)
    recipe = {f"augment.{k}" for k in (M.FULL_RUNS[x]["augment"] or {})}
    recipe |= {"perf.num_workers"} if M.FULL_RUNS[x].get("workers") else set()  # P-0.05B's loader (DECISIONS H12)
    for name, allowed in ((f"full-{x}", FULL_KEYS | recipe), (f"smoke-{x}", FULL_KEYS | SMOKE_ONLY)):
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
    """Contract 7's table with the boxes' epochs now (DECISIONS G1, confirmed by the owner on 2026-10-02: T-0.6B 3,
    P-0.3B 3, P-0.05B 5, the 10-epoch request withdrawn; P-0.3B 6 since H5, T-0.6B 8 under H7 and 6 since H15,
    P-0.05B 10 since H9;
    full-p01 keeps box 1's 4, and its continuation's 8 epochs, patience 12 and recipe go in by --resume-set, H9), each
    with COMMON's early-stop patience 6, and the end
    reserve: the trainer's
    default 30 min but full-t06's 55 (READOUT_RESERVE, a deviation from contract 7 that
    test_a_readout_starts_after_a_shortened_run explains). No full config pulls both label roots."""
    want = {"t06": (6, 300, 2e-4, 450, 1730, False, 55), "p03": (6, 300, 2e-4, 600, 1350, True, 30),
            "p01": (4, 1000, 1e-3, 1600, 1500, True, 30), "p005": (10, 1000, 1e-3, 1600, 1500, True, 30)}
    assert {x: r["epochs"] for x, r in M.FULL_RUNS.items()} == {"t06": 6, "p03": 6, "p01": 4, "p005": 10}
    assert not any(r["pull_parakeet"] for r in M.FULL_RUNS.values())
    for x, (epochs, warmup, lr, micro, step, greedy, reserve) in want.items():
        c = cfg(f"full-{x}")
        assert "pull_parakeet" not in c, x
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
    # the augmentation recipe (DECISIONS H1 step 2): full-p03 carries RECIPE as its augment block, at the config's end
    # (only the keys it changes: 04_distill's DEFAULTS give the rest), and full-t06 RECIPE_AED_H15 - RECIPE_AED, recipe
    # v3 (the end trim, and the background since DECISIONS H11; recipe v4 since H12) plus its cut table (H7), with
    # augment.guard_per_piece and augment.end_trim_voiced on since DECISIONS H15; full-p005 RECIPE_V4 (H8, H12); no
    # other config has one (full-p01 is box 1's own, the smokes are smoke A's)
    assert {x: r["augment"] for x, r in M.FULL_RUNS.items()} == {"t06": M.RECIPE_AED_H15, "p03": M.RECIPE, "p01": None,
                                                                 "p005": M.RECIPE_V4}
    # P-0.3B trains recipe v2 (DECISIONS H6); P-0.05B recipe v4 (H8, H12); RECIPE_TEST stays the record of what the
    # first test ran; the cooldown re-runs (H14) recipe v4 with its ranges written out, at the defaults or gentler
    assert M.FULL_RUNS["p03"]["augment"] is M.RECIPE and M.CONTINUATIONS["p01-cool"]["augment"] is M.RECIPE_V4_FULL
    assert [c["augment"] for c in M.CONTINUATIONS.values()] == [M.RECIPE_GENTLE, M.RECIPE_V4_FULL, M.RECIPE_GENTLE]
    assert M.RECIPE_V3 == dict(M.RECIPE, end_trim_p=0.3, noise_p=0.3, noise_bank="aug/musan-bg-v2",
                               noise_bank_sha256=M.NOISE_BANK_SHA256)
    # DECISIONS H12: recipe v4 = v3 + background speech, room echo, volume and codecs (the targets the clean row's)
    assert M.RECIPE_V4 == dict(M.RECIPE_V3, speech_p=0.15, reverb_p=0.2, rir_bank="aug/rirs-v1",
                               rir_bank_sha256=M.RIR_BANK_SHA256, gain_p=0.3, codec_p=0.1)
    # P-0.05B's loader: 12 workers for the chain's CPU (its GPU takes ~7,900 audio-s/s); the smoke configs keep "auto"
    assert cfg("full-p005")["perf"]["num_workers"] == 12 and cfg("smoke-p005")["perf"]["num_workers"] == "auto"
    assert all(cfg(f"full-{x}")["perf"]["num_workers"] == "auto" for x in ("t06", "p03", "p01"))
    assert cfg("full-p03")["augment"] == M.RECIPE and list(cfg("full-p03"))[-1] == "augment"
    assert cfg("full-t06")["augment"] == M.RECIPE_AED_H15 and list(cfg("full-t06"))[-1] == "augment"
    # every run still to come trains with the background - music without vocals and noise (DECISIONS H8, H11)
    assert M.RECIPE_AED == dict(M.RECIPE_V4, cuts="labels/full/selections/full_study/aed_cuts.parquet",
                                cuts_sha256="cde6504ff3c9a1d539385eec745ca9301570b42fc1c182ffdc16ccbf4a9eb4bb")
    assert M.RECIPE_AED_H15 == dict(M.RECIPE_AED, guard_per_piece=True, end_trim_voiced=True)
    for a in (M.RECIPE_AED_H15, M.RECIPE_AED, M.RECIPE_V4, *(c["augment"] for c in M.CONTINUATIONS.values())):
        assert (a["noise_p"], a["noise_bank"], a["noise_bank_sha256"], a["end_trim_p"]) == (
            0.3, M.NOISE_BANK, M.NOISE_BANK_SHA256, 0.3)
        assert (a["speech_p"], a["reverb_p"], a["rir_bank"], a["rir_bank_sha256"], a["gain_p"], a["codec_p"]) == (
            0.15, 0.2, M.RIR_BANK, M.RIR_BANK_SHA256, 0.3, 0.1)
    assert [n for n in GENERATED if "augment" in cfg(n)] == ["full-t06", "full-p03", "full-p005"]


# the smoke configs smoke A ran (2026-10-01): SMOKE_EPOCHS keeps them byte for byte whatever the full runs' epochs are
SMOKE_SHA256 = {"t06": "9f38ffa8deb75d7cc66c5477bc7b01f5f6cd2f6ba2f13e37ef571eafdaff3d2f",
                "p03": "8fd502383f62f29f4c3dcb15228004785d20a068b0e61988acd2f17640226c96",
                "p01": "7087b294f2836eedba36788e48754fa2c738f8ec5bc779fc279bc81fe5e15f87",
                "p005": "7fa9b16688da386352d957f2a8b567f6f3ee646fc1592cb49cfec285c29bcaf2"}  # at 14bfcad, LF


def test_the_smoke_variants(plan):
    """smoke-<x> is full-<x> with SMOKE at SMOKE_EPOCHS (the epochs smoke A ran: 3 / 3 / 4 / 4), so full-p005's 5
    epochs moved none of them: smoke check 3's plan_total_steps (smoke-p005's 107,910, F4's 150 s) stay."""
    assert M.SMOKE_EPOCHS == {"t06": 3, "p03": 3, "p01": 4, "p005": 4}
    for x in STUDENTS:
        f, s = cfg(f"full-{x}"), cfg(f"smoke-{x}")
        assert s["run_name"] == f"smoke-{x}" and s["schedule"]["epochs"] == M.SMOKE_EPOCHS[x]
        assert s["pull_parakeet"] is True
        import hashlib
        raw = (FULL / f"smoke-{x}.json").read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == SMOKE_SHA256[x], x
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
    assert data(cfg("data-t")) == dict(fullrun.FULL_DATA, family="aed")
    assert data(cfg("data-p")) == data(cfg("data-p01"))  # box full-p pulls what box 1 pulled
    assert data(cfg("data-full")) == dict(fullrun.FULL_DATA, pull_parakeet=True)
    assert data(cfg("data-smoke")) == dict(fullrun.SMOKE_DATA, pull_parakeet=True)
    study = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
    assert data(cfg("data-smoke-b")) == {k: study[k] for k in fullrun.DATA_KEYS if k in study}
    for n in ("data-p01", "data-t", "data-p", "full-t06", "full-p03", "full-p01", "full-p005"):
        assert "pull_parakeet" not in cfg(n), n
    for n in ("data-p01", "data-t", "data-p", "data-full", "data-smoke", "data-smoke-b"):
        assert "tools/make_full_configs.py" in cfg(n)["_comment"]
    # the train configs carry their box's data keys (the registry checks it too); smoke-p01 pulls both roots, as
    # smoke A does
    for x in STUDENTS:
        box_dc = {"t06": "data-t", "p03": "data-p", "p01": "data-p01", "p005": "data-p"}[x]
        for name, dc in ((f"full-{x}", box_dc), (f"smoke-{x}", "data-smoke")):
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


def test_box_t_pulls_teacher_labels_only_and_each_box_one_store():
    """data-t (AED, no pull_parakeet): kitsune.extent.pull_plan asks for teacher_out of every stem and no parakeet_out
    (T-0.6B's store, its readouts and the AED eval read only teacher_root); data-p is data-p01. Each box's extent
    sizing counts one store (extent.stores: 1), not data-full's two."""
    t, pp = cfg("data-t"), cfg("data-p")
    rec = toy_record(t)
    assert extent.validate(t) == [] and extent.record_problems(rec, t) == []
    stems = {(n, st) for n, sts in extent.subset_stems(rec, t).items() for st in sts}

    def npz(plan_, root):
        return {tuple(f.split("/")[-2:]) for f in plan_["required"] if f.startswith(f"labels/full/{root}/")
                and f.endswith(".npz")}

    got = extent.pull_plan(t, rec, [])
    assert npz(got, "teacher_out") == {(n, f"{st}.npz") for n, st in stems} and npz(got, "parakeet_out") == set()
    assert extent.label_root_for(t, "reazon_small", "train-00000") == "teacher"
    assert npz(extent.pull_plan(pp, toy_record(pp), []), "parakeet_out") == npz(
        extent.pull_plan(cfg("data-p01"), toy_record(cfg("data-p01")), []), "parakeet_out")
    assert extent.stores(t) == extent.stores(pp) == extent.stores(cfg("data-p01")) == 1
    assert extent.stores(cfg("data-full")) == 2


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
    # the launch part: full.parquet (the same file) at the boxes' epochs, each epoch's plan as long as the 4-epoch one's
    assert plan["launch"]["selection"] == {"path": fullrun.FULL_SELECTION, "sha256": FULL_SHA}
    for x in STUDENTS:
        s, f = plan["launch"]["students"][x], plan["full"]["students"][x]
        assert (s["epochs"], s["total_steps"], len(s["steps_per_epoch"])) == (
            LAUNCH_EPOCHS[x], LAUNCH_T[x], LAUNCH_EPOCHS[x]) == (M.launch_epochs(x), M.launch_total_steps(x, plan),
                                                                  M.launch_epochs(x))
        assert s["steps_per_epoch"][:len(f["steps_per_epoch"])] == f["steps_per_epoch"], x
        assert s["worst_shapes"] == f["worst_shapes"], x  # epoch 0's: the smoke's probe shapes do not move
    # the cooldown re-runs (DECISIONS H13, H14): P-0.3B's and P-0.1B's each from its run's pre_cooldown state - the
    # step before its plan's cooldown at its own epochs - to the same T, so the whole re-run is the cooldown: P-0.1B
    # 86,328 of 107,910 (the launch part's 4-epoch plan is the full part's, step for step), P-0.3B 169,376 of 211,720;
    # P-0.05B's from its early stop's cooldown start, 67,449, long before its planned cooldown, to that cooldown's T
    c = M.CONTINUATIONS
    assert (c["p01-cool"]["from_step"], c["p01-cool"]["to_step"]) == (86328, 107910) == (
        plan["full"]["students"]["p01"]["cooldown_start_step"] - 1, plan["launch"]["students"]["p01"]["total_steps"])
    assert plan["launch"]["students"]["p01"] == plan["full"]["students"]["p01"]
    assert plan["launch"]["students"]["p01"]["cooldown_start_step"] == 86329 == math.ceil(0.8 * 107910) + 1
    assert (c["p03-cool"]["from_step"], c["p03-cool"]["to_step"]) == (169376, 211720) == (
        plan["launch"]["students"]["p03"]["cooldown_start_step"] - 1, plan["launch"]["students"]["p03"]["total_steps"])
    assert plan["launch"]["students"]["p03"]["cooldown_start_step"] == math.ceil(0.8 * 211720) + 1
    assert plan["launch"]["students"]["p005"]["cooldown_start_step"] == 215823 == math.ceil(0.8 * 269777) + 1
    assert c["p005-cool"]["from_step"] == 67449 < c["p005-cool"]["to_step"] == 80939 < 215823


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
    (out / M.PLAN_FILE).write_text(json.dumps({"launch": plan["launch"]}), encoding="utf-8")  # the launch part stays
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
    (out / M.PLAN_FILE).unlink()  # no launch part either: refused for both reasons
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


def test_import_launch_plan(tmp_path, plan, capsys):
    """--import-launch-plan records tools/full_plan.py's JSON of full.parquet at the boxes' epochs as the plan record's
    launch part (the full and smoke parts stay), and refuses a record at other epochs, of another full.parquet or with
    a student missing; nothing is written then."""
    out = repo_copy(tmp_path)
    before = (out / M.PLAN_FILE).read_bytes()
    rec = copy.deepcopy(plan["launch"])
    rec["selection"]["path"] = "D:\\somewhere\\full_study\\full.parquet"
    src = tmp_path / "plan_launch.json"
    src.write_text(json.dumps(rec), encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-launch-plan", str(src)]) == 0
    printed = capsys.readouterr().out
    assert "launch: t06 6 epochs T 143891, p03 6 epochs T 211720, p01 4 epochs T 107910, p005 10 epochs T 269777" \
        in printed and "boxes.json: carries the launch record's hours" in printed
    assert json.loads((out / M.PLAN_FILE).read_text(encoding="utf-8")) == json.loads(before) and M.check(out) == []
    for change, want in ((lambda r: r["students"]["t06"].update(epochs=10), "launch.t06: measured with {'epochs': 10}, "
                          "FULL_RUNS / CONTINUATIONS says {'epochs': 6}"),
                         (lambda r: r["students"]["p005"].update(epochs=4), "launch.p005: measured with {'epochs': 4}, "
                          "FULL_RUNS / CONTINUATIONS says {'epochs': 10}"),
                         (lambda r: r["students"]["p01"].update(epochs=8), "launch.p01: measured with {'epochs': 8}, "
                          "FULL_RUNS / CONTINUATIONS says {'epochs': 4}"),
                         (lambda r: r["selection"].update(sha256="0" * 64), "launch: measured on a full.parquet of "
                          "sha256"),
                         (lambda r: r["students"].pop("p005"), "launch: no student p005"),
                         (lambda r: r["students"]["p03"]["steps_per_epoch"].pop(), "launch.p03: steps_per_epoch")):
        bad = copy.deepcopy(rec)
        change(bad)
        src.write_text(json.dumps(bad), encoding="utf-8")
        assert M.main(["--out-dir", str(out), "--import-launch-plan", str(src)]) == 1
        assert want in capsys.readouterr().err, want
        assert (out / M.PLAN_FILE).read_bytes() == before.replace(b"\r\n", b"\n") or json.loads(
            (out / M.PLAN_FILE).read_text(encoding="utf-8")) == json.loads(before)
    with pytest.raises(SystemExit):
        M.main(["--out-dir", str(out), "--import-launch-plan", str(src), "--check"])


def test_box_hours_follow_the_record(plan):
    """The hours of the HOURS_BOXES - full-t, p005 and p-cool - = box_hours of the speed record: plan v3's run model
    (calc_v3: steps x s/step x r x (1 + o) + fixed + dev checks) at the launch part's step counts (a continuation: its
    own steps and their dev checks), smoke A's measured s/step x r (box 1's / smoke A's P-0.1B s/step) with box 1's o;
    one GPU per box: est = setup + one store + the runs + the tail + end, max = the pessimistic setup and store + the
    runs x 1.26 + the worst tail + reserve + one stall. The model reproduces box 1 (9.16 h vs 9.152 h measured) and so
    already holds its 0.305 / 0.2715 eval overhead."""
    rec = M.load_speed()
    b1 = rec["box1"]
    assert rec["smoke"]["sec_per_step"] == SMOKE_SPS and rec["smoke"]["source"] == "full/box-full-smoke/smoke_verdict.json"
    assert (b1["revision"], b1["run_id"], b1["steps"], b1["train_wall_h"], b1["stores_ctc_h"], b1["bootstrap_h"]) == (
        M.CONTINUATIONS["p01-cool"]["revision"], P01_RID, 107910, 9.152, 0.548, 3.09)
    r = b1["sec_per_step"] / SMOKE_SPS["p01"]
    assert round(r, 4) == 1.0361 and b1["overhead"] == 0.0465
    # box 1 by the same model, at its own 4 epochs: 9.16 h against 9.152 h measured
    assert M._run_h(107910, SMOKE_SPS["p01"], r, 0.0465, 837, 40, 8.5) == 9.16
    hs = {b: M.box_hours(plan, rec, b, reserve_min=60 if b == "full-t" else 45) for b in M.HOURS_BOXES}
    # box p-cool: the three cooldown re-runs in queue order (DECISIONS H14); boxes full-p and p01 left (their records);
    # box p005 P-0.05B's fresh run (H10, its record)
    assert M.HOURS_BOXES == {"full-t": ("t06",), "p005": ("p005",),
                             "p-cool": ("cont:p03-cool", "cont:p01-cool", "cont:p005-cool")}
    assert [x[len("cont:"):] for x in M.HOURS_BOXES["p-cool"]] == list(M.CONTINUATIONS)
    assert set(M.BOX_TAIL_H) == set(M.HOURS_BOXES) and M.BOX_TAIL_H["p-cool"] == M.PCOOL_TAIL_H == (0.4, 1.0)
    # t06 with the AED recipe, 6 epochs (DECISIONS H15): 143,891 steps x 1.04295 s x r x 1.0465 x 1.05 + 435 s fixed
    # + 60 dev checks x 9.6 s, rounded up to 0.01 h (H7's 8 epochs: 191,855 steps and 80 dev checks, 63.62 h)
    assert math.ceil((143891 * SMOKE_SPS["t06"] * 1.05 * r * 1.0465 + 435 + 60 * 9.6) / 36) / 100 == 47.75
    assert M._run_h(143891, SMOKE_SPS["t06"], r, 0.0465, 435, 60, 9.6) == 45.49  # without the recipe
    assert M._run_h(191855, SMOKE_SPS["t06"] * 1.05, r, 0.0465, 435, 80, 9.6) == 63.62  # H7's 8 epochs
    assert {b: h["items"] for b, h in hs.items()} == RUN_H
    assert {b: (h["est_hours"], h["max_hours"]) for b, h in hs.items()} == BOX_H
    # the tails: box p-cool's 0.4 / 1.0 h plus one pull per continuation (CONT_PULL_H 0.1 / 0.3), three
    assert {b: (h["r"], h["o"], h["setup_h"], h["store_h"], h["tail_h"]) for b, h in hs.items()} == {
        "full-t": (1.0361, 0.0465, (1.6, 3.3), (0.548, 0.76), (0.6, 1.35)),
        "p005": (1.0361, 0.0465, (1.6, 3.3), (0.548, 0.76), (0.2, 0.5)),
        "p-cool": (1.0361, 0.0465, (1.6, 3.3), (0.548, 0.76), (0.7, 1.9))}
    # est full-t: 1.6 + 0.548 + 47.75 + 0.6 + 0.35 = 50.848 -> 50.9; max: ceil(3.3 + 0.76 + 47.75 x 1.26 + 1.35 + 1 +
    # 1.25) = ceil(67.82) = 68, T-0.6B's pessimistic end leaving 68 - 1 - 55 min - 64.23 = 1.86 h (H7's 8 epochs: est
    # 66.8, max 88, the same 1.86 h);
    # est p005: 1.6 + 0.548 + 15.76 + the tail 0.2 + 0.35 = 18.458 -> 18.5; max: ceil(3.3 + 0.76 + 15.76 x 1.26 + 0.5 +
    # 0.75 + 1.25) = ceil(26.42) = 27, P-0.05B's pessimistic end leaving 27 - 0.75 - 30 min - 23.92 = 1.83 h;
    # est p-cool: 1.6 + 0.548 + (5.39 + 2.11 + 1.01) + the tail 0.7 + 0.35 = 11.708 -> 11.8; max: ceil(3.3 + 0.76 + 8.51
    # x 1.26 + 1.9 + 0.75 + 1.25) = ceil(18.68) = 19, the last re-run's pessimistic end leaving 19 - 0.75 - 30 min -
    # 14.78 = 2.97 h
    assert {b: h["slack_h"] for b, h in hs.items()} == {"full-t": 1.86, "p005": 1.83, "p-cool": 2.97}
    # P-0.3B's 6-epoch run as it was planned (DECISIONS H5; box full-p's record, 26.53 h): 211,720 steps x 0.3924 s x r
    # x 1.0465 x 1.05 + 348 s fixed + 60 dev checks x 9.2 s
    assert math.ceil((211720 * SMOKE_SPS["p03"] * 1.05 * r * 1.0465 + 348 + 60 * 9.2) / 36) / 100 == 26.53
    assert M._run_h(211720, SMOKE_SPS["p03"], r, 0.0465, 348, 60, 9.2) == 25.28  # without the recipe
    # the cooldown re-runs (DECISIONS H14), at AUGMENT_STEP_FACTOR 1.05 (concat's longer rows): P-0.3B's 211,720 -
    # 169,376 = 42,344 steps and 42,344 / (211,720 / 6) / 0.1 = 12 dev checks: 5.39 h; P-0.1B's, box 1's run from its
    # pre_cooldown state to the same T, 21,582 steps and 8 dev checks at box 1's pace: 2.11 h (2.02 h without the
    # factor, the first recipe test's); P-0.05B's kept early-stop cooldown, 80,939 - 67,449 = 13,490 steps and 13,490 /
    # (269,777 / 10) / 0.1 = 5.0 dev checks: 1.01 h (0.97 h without). (H9's 8-epoch continuation of P-0.1B, box p01's
    # record: 129,487 steps and 48 dev checks, 11.47 h, 10.94 h without the factor.)
    conts = hs["p-cool"]["continuations"]
    assert [(c["name"], c["student"], c["total_steps"], c["plan_total_steps"], c["steps"], c["dev_checks"],
             c["step_factor"], c["hours"]) for c in conts] == [
        ("p03-cool", "p03", 211720, 211720, 42344, 12.0, 1.05, 5.39),
        ("p01-cool", "p01", 107910, 107910, 21582, 8.0, 1.05, 2.11),
        ("p005-cool", "p005", 80939, 269777, 13490, 5.0, 1.05, 1.01)]
    assert hs["p-cool"]["continuation"] == conts[0] == M.continuation_run("p03-cool", plan, rec)
    assert M._run_h(42344, SMOKE_SPS["p03"] * 1.05, r, 0.0465, 348, 12, 9.2) == 5.39
    assert M._run_h(13490, SMOKE_SPS["p005"] * 1.05, r, 0.0465, 820, 13490 / (269777 / 10) / 0.1, 8.4) == 1.01
    assert M._run_h(13490, SMOKE_SPS["p005"], r, 0.0465, 820, 5, 8.4) == 0.97
    assert M._run_h(129487, SMOKE_SPS["p01"] * 1.05, r, 0.0465, 837, 48, 8.5) == RECORD_RUN_H["p01"]["full-p01"]
    assert M._run_h(129487, SMOKE_SPS["p01"], r, 0.0465, 837, 48, 8.5) == 10.94
    assert M.AUGMENT_STEP_FACTOR == 1.05 and M.augment_step_factor(M.RECIPE) == M.augment_step_factor(M.RECIPE_AED) == 1.05
    assert M.augment_step_factor(M.RECIPE_V4_FULL) == M.augment_step_factor(M.RECIPE_GENTLE) == 1.05
    assert M.augment_step_factor(None) == M.augment_step_factor(dict(M.RECIPE, concat_p=0.0)) == 1.0
    assert M.augment_step_factor(dict(M.RECIPE, enabled=False)) == 1.0
    assert M._run_h(21582, SMOKE_SPS["p01"], r, 0.0465, 837, 8, 8.5) == 2.02
    assert M._run_h(21582, SMOKE_SPS["p01"] * 1.05, r, 0.0465, 837, 8, 8.5) == 2.11
    assert hs["full-t"]["continuation"] is None and hs["full-t"]["continuations"] == []
    assert hs["p005"]["continuations"] == [] and hs["p005"]["tail_h"] == M.P005_TAIL_H  # no pull: a fresh run
    # one more epoch costs: T-0.6B 7.94 h, P-0.3B 4.41 h and P-0.05B 1.55 h with their recipes (7.56, 4.20 and 1.48
    # without) at this pace
    per_epoch = {x: round((hs["full-t"]["run_h"][x] - M.FIXED_S[x] / 3600) / M.launch_epochs(x), 2)
                 for x in ("t06", "p03", "p005")}  # the run less its fixed time, per epoch
    assert per_epoch == {"t06": 7.94, "p03": 4.41, "p005": 1.55}
    # without box 1's part: r 1, o 0.08, the store from calc_v3 (both stores / 1.5), provisional
    h0 = M.box_hours(plan, dict(rec, box1=None), "full-t")
    assert (h0["r"], h0["o"], h0["store_h"]) == (1.0, 0.08, (0.913, 1.27)) and h0["items"]["full-t06"] == 47.56
    assert M.box_hours(plan, rec, "full-t", reserve_min=120)["max_hours"] == 69
    # the boxes that left HOURS_BOXES have no box_hours (no tail): a record is never projected again
    for b in ("full-p", "p01"):
        with pytest.raises(KeyError):
            M.box_hours(plan, rec, b)
    assert not hasattr(M, "continuation_flags")  # the registry's continues blocks replaced launch's reset/set flags
    assert M.speed_problems(rec) == []
    bad = copy.deepcopy(rec)
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
    """--import-speed records smoke A's check 3 and box 1's tools/box1_go.py --json as the speed record, byte-stable,
    prints the hours of every HOURS_BOXES box boxes.json must carry (with each continuation's run and, for box p-cool,
    every continuation's run, state, readout and sets as the queue passes them), and --check holds boxes.json to them;
    a verdict without check 3's s/step is refused and nothing is written; without a record
    the boxes' hours are not held."""
    out = repo_copy(tmp_path)
    committed = (out / M.SPEED_FILE).read_bytes()
    box1 = json.loads(committed)["box1"]
    (out / M.SPEED_FILE).unlink()
    assert M.check(out) == []  # no record: the boxes' hours are not held
    v = tmp_path / "smoke_verdict.json"
    v.write_text(json.dumps(smoke_verdict()), encoding="utf-8")
    g = tmp_path / "box1_go.json"
    g.write_text(json.dumps({"go": False, "exit": 1, "lines": [], "box1": box1}), encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-speed", "--smoke-verdict", str(v), "--box1-go", str(g)]) == 0
    printed = capsys.readouterr().out
    assert "smoke A and box 1; r 1.0361, o 0.0465" in printed and "boxes.json: carries the speed record's hours" in printed
    assert "  box full-t: full-t06 max_hours 47.75; est_hours 50.9, max_hours 68" in printed
    assert "  box p005: full-p005 max_hours 15.76; est_hours 18.5, max_hours 27" in printed
    assert ("  box p-cool: full-p03 max_hours 5.39, full-p01 max_hours 2.11, full-p005 max_hours 1.01; est_hours 11.8, "
            "max_hours 19 (setup 1.6 / 3.3 h, store 0.548 / 0.76 h, tail 0.7 / 1.9 h; the pessimistic slack 2.97 h); "
            "continuation p03-cool 42344 steps to T 211720, 12 dev checks, s/step x 1.05 (the recipe's concat); "
            "continuation p01-cool 21582 steps to T 107910, 8 dev checks, s/step x 1.05 (the recipe's concat); "
            "continuation p005-cool 13490 steps to T 80939, 5 dev checks, s/step x 1.05 (the recipe's concat)\n") \
        in printed
    assert "box full-p:" not in printed and "box p01:" not in printed  # records, no projection
    for name, (src, rid, step, to, readout) in {
            "p03-cool": ("full-p", P03_RID, 169376, 211720, "r1"), "p01-cool": ("p01", P01_RID, 86328, 107910, "r2"),
            "p005-cool": ("p005", P005_RID, 67449, 80939, "r1")}.items():
        kept = " (its early-stop cooldown kept)" if name == "p005-cool" else ""
        assert (f"  continuation {name} on box p-cool: box {src}'s run {rid} from full_step_{step} to step {to}{kept}, "
                f"readout runs/m4-{rid}-{readout}, baseline revision {M.CONTINUATIONS[name]['revision'][:8]}; sets "
                + " ".join(COOL_SETS[name]) + "\n") in printed, name
    assert "  launch --box p-cool (the registry carries its continuations)\n" in printed
    assert "--resume-reset" not in printed and "--resume-set" not in printed
    assert "P-0.1B from step 0 at its continuation's epochs projects to 9.16 h (box 1's 4 epochs: 9.152 h measured)" \
        in printed
    assert (out / M.SPEED_FILE).read_bytes().replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")
    assert M.check(out) == []
    # a slower box 1: the smoke part is kept, the new hours printed, --check fails until boxes.json carries them
    g.write_text(json.dumps({"go": True, "box1": dict(box1, sec_per_step=0.2987)}), encoding="utf-8")
    assert M.main(["--out-dir", str(out), "--import-speed", "--box1-go", str(g)]) == 0
    printed = capsys.readouterr().out
    assert "smoke A and box 1; r 1.1, o 0.0465" in printed and "boxes.json: 8 change(s) by hand" in printed
    assert M.load_speed(out)["smoke"]["sec_per_step"] == SMOKE_SPS
    assert len(M.check(out)) == 8 and all(x.startswith("boxes.json: boxes.") for x in M.check(out))
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


BOX_TABLE = {  # contract 7: gpus, data config, est / max h, max_dph, extra_gb, reserve, watchdog, timed, gate, smoke,
    # min_ram_gb (the boxes' hours: box_hours; boxes p01 and full-p: the records, DECISIONS H14)
    "full-smoke": (1, "data-smoke", 5.2, 9, 1.00, 90, 20, (600, "alert"), True, True, True, None),
    "p01": (1, "data-p01", *RECORD_BOX_H["p01"], 1.10, 35, 45, (3600, "stop"), True, True, False, 96),
    "p005": (1, "data-p01", *BOX_H["p005"], 1.10, 35, 45, (3600, "stop"), True, True, False, 96),
    "full-t": (1, "data-t", *BOX_H["full-t"], 1.40, 70, 60, (3600, "stop"), True, True, False, 96),  # $1.40: H15
    "full-p": (1, "data-p", *RECORD_BOX_H["full-p"], 1.10, 60, 60, (3600, "stop"), True, True, False, 96),
    "smoke-b": (1, "data-smoke-b", 1.75, 3, 1.00, 40, 15, (3600, "stop"), False, False, True, None),
    "p-cool": (1, "data-p01", *BOX_H["p-cool"], 1.10, 70, 45, (3600, "stop"), True, True, False, 96),
}
assert BOX_H == {"full-t": (50.9, 68), "p005": (18.5, 27), "p-cool": (11.8, 19)}
assert RECORD_BOX_H == {"full-p": (30.0, 42), "p01": (14.3, 22)}


def test_the_registry_loads_with_its_boxes(reg, monkeypatch):
    monkeypatch.delenv(fullrun.ENV_REGISTRY, raising=False)
    assert fullrun.load_registry() == reg  # this checkout's configs/full/boxes.json, every file checked
    assert fullrun.registry_problems(json.loads((FULL / "boxes.json").read_text(encoding="utf-8")), root=ROOT) == []
    # every plain box, and the chained box of contract addendum E (test_the_chained_box); the retired 2x box full is a
    # box name (the tests' fixtures) but not in the registry, so launch --box full is refused
    assert set(fullrun.BOX_NAMES) == set(BOX_TABLE) | {"full"} and "full" not in reg["boxes"]
    assert set(reg["boxes"]) == set(fullrun.ALL_BOX_NAMES) - {"full"}
    for box, (gpus, dc, est, mx, dph, extra, reserve, wd, timed, gate, smoke, ram) in BOX_TABLE.items():
        b = fullrun.box_spec(box, reg)
        assert (b["gpus"], b["data_config"], b["est_hours"], b["max_hours"], b["max_dph"], b["extra_gb"],
                b["deadline_reserve_min"], (b["watchdog"]["orphan_s"], b["watchdog"]["action"]), b["timed_states"],
                b["gate"], b["smoke"], b["max_attempts"], b["min_ram_gb"]) == (
            gpus, f"configs/full/{dc}.json", est, mx, dph, extra, reserve, wd, timed, gate, smoke, 4, ram), box
        assert fullrun.box_env(box, reg)[fullrun.ENV_N_GPUS] == str(gpus)
        assert bool(b["faults"]) == (box == "full-smoke")
    sidecar = {"full-smoke": "smoke.json", "p01": "full.json", "full-t": "full.json", "full-p": "full.json",
               "p005": "full.json", "p-cool": "full.json"}
    for box in BOX_TABLE:
        want = [fullrun.FROZEN_MANIFEST, f"{fullrun.FULL_DIR}/{sidecar[box]}"] if box in sidecar else []
        want += [M.AED_CUTS] if box == "full-t" else []  # T-0.6B's cut table (DECISIONS H7)
        assert fullrun.box_extra_files(box, reg) == want, box
        assert fullrun.box_extra_dirs(box, reg) == ([PARAKEET] if box in ("full-smoke", "smoke-b") else
                                                    [M.NOISE_BANK, M.RIR_BANK] if box in ("p01", "p005", "full-t",
                                                                                          "p-cool")
                                                    else []), box
    for x in ("p01", "p005"):
        assert fullrun.box_students(x, reg) == fullrun.box_ctc_students(x, reg) == [f"students/study/{x}"]
    assert fullrun.box_students("full-t", reg) == ["students/study/t06"] and fullrun.box_ctc_students("full-t", reg) == []
    # box P pulls P-0.3B's student only now (P-0.05B postponed, DECISIONS H2)
    assert fullrun.box_students("full-p", reg) == fullrun.box_ctc_students("full-p", reg) == ["students/study/p03"]
    assert fullrun.box_students("full-smoke", reg) == [f"students/study/{x}" for x in STUDENTS]
    assert fullrun.box_students("smoke-b", reg) == []
    # no argv, args or verdict json reads the state dir (a chained box's part runs in chain/<part>, addendum E.10)
    for box in BOX_TABLE:
        for it in fullrun.box_items(box, reg):
            templates = [*(it.get("argv") or []), *(it.get("args") or []), *(v.get("json", "") for v in it["verdict"])]
            assert not any("{state}" in t for t in templates), (box, it["name"])


def run_items(it: dict, x: str, max_hours: float) -> None:
    """One P run's items of a box: the run (study_run study-<x>, its own config) at max_hours, its readout and the 7
    quantised readouts of its weights (a same-box `of` after the readout, so they read this box's final step; no
    of_box: never another box's weights)."""
    t = it[f"full-{x}"]
    assert (t["kind"], t["config"], t["study_run"], t["family"], t["max_hours"], t["needs"], t["droppable"],
            t["stall_min"], t["plan_total_steps"]) == ("train", f"configs/full/full-{x}.json", f"study-{x}", "ctc",
                                                       max_hours, ["stores-ctc"], False, 45, None)
    m4 = it[f"m4-full-{x}"]
    assert (m4["of"], m4["max_hours"], m4["needs"]) == (f"full-{x}", 0.3, [f"full-{x}"])
    for fmt in QUANT_FORMATS:
        q = it[f"quant-{fmt}-full-{x}"]
        assert (q["kind"], q["argv"], q["of"], q["of_box"], set(q["needs"]), q["max_hours"]) == (
            "eval", quant_argv(fmt), f"full-{x}", None, {f"m4-full-{x}", f"full-{x}"}, 0.3), q["name"]


def p_box_items(reg, box: str, x: str, max_hours: float) -> dict:
    """Box p01's or p005's items: the CTC store of the run's own config, then run_items (the run at max_hours)."""
    it = items(reg, box)
    quant = [f"quant-{f}-full-{x}" for f in QUANT_FORMATS]
    assert list(it) == ["stores-ctc", f"full-{x}", f"m4-full-{x}", *quant]
    assert it["stores-ctc"]["config"] == it[f"full-{x}"]["config"] == f"configs/full/full-{x}.json"
    run_items(it, x, max_hours)
    assert it[f"full-{x}"]["continues"] is None  # the run itself: the continuation is box p-cool's
    return it


def test_box_p005_is_the_p_test_box(reg, trainer):
    """Box p005 (DECISIONS H10, the owner: "lets only do p0.05 for now to reduce costs and first see if these changes
    helped"): P-0.05B from step 0 at 10 epochs (H9) with RECIPE_V4 (H8, H12) - its config's augment block, the
    banks pinned and pulled as the box's extra dirs - and its readouts. Its own box: box p01's Hub summary holds box
    1's run, which P-0.1B's continuation resumes, and a fresh queue there would overwrite it. Its run is done; its
    cooldown re-run is box p-cool's (DECISIONS H14), which needs its config byte for byte."""
    import hashlib
    p_box_items(reg, "p005", "p005", RUN_H["p005"]["full-p005"])
    c = trainer.load_config(str(FULL / "full-p005.json"), [])
    assert c["schedule"]["epochs"] == 10 and c["augment"] == dict(trainer.DEFAULTS["augment"], **M.RECIPE_V4)
    assert c["early_stop"]["patience"] == M.COMMON["early_stop"]["patience"] == 6
    assert fullrun.box_extra_dirs("p005", reg) == [M.NOISE_BANK, M.RIR_BANK] and M.HOURS_BOXES["p005"] == ("p005",)
    raw = (FULL / "full-p005.json").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == FULL_P005_SHA256


def test_box_p01_is_box_1s_record(reg):
    """Box p01 (DECISIONS H14): origin/main's entry again - box 1's items with H9's 8-epoch continuation hours (11.47 h,
    est 14.3 / max 22), the record of what it ran and had planned; P-0.1B's cooldown re-run is box p-cool's, which
    reads this box's Hub summary as the run's source and needs box 1's config byte for byte. Its hours are no longer
    held to the speed record (not an HOURS_BOXES box)."""
    import hashlib
    p_box_items(reg, "p01", "p01", RECORD_RUN_H["p01"]["full-p01"])
    # no registry item reads another box's run any more (the of_box code and its tests stay)
    assert not any(x.get("of_box") for b in fullrun.BOX_NAMES if b in reg["boxes"] for x in fullrun.box_items(b, reg))
    assert "p01" not in M.HOURS_BOXES and fullrun.continues_of("p01", reg) == {}
    raw_entry = json.loads((FULL / "boxes.json").read_text(encoding="utf-8"))["boxes"]["p01"]
    assert "continues on box p-cool" in raw_entry["_comment"] and "DECISIONS H14" in raw_entry["_comment"]
    assert "8-EPOCH CONTINUATION" in raw_entry["_comment"]  # H9's plan, the record
    assert M.RECIPE_TEST == {"enabled": True, "truncate_p": 0.3, "concat_p": 0.5, "mix_p": 0.05}  # H1, H4: as run
    assert M.RECIPE == {"enabled": True, "truncate_p": 0.2, "concat_p": 0.5, "mix_p": 0.0,
                        "truncate_min_row_s": 3.0}  # DECISIONS H6: v2, P-0.3B's
    # box 1's config keeps its 4 epochs, and so does the re-run (launch_epochs: the continuation's)
    assert M.FULL_RUNS["p01"]["epochs"] == cfg("full-p01")["schedule"]["epochs"] == 4 == M.launch_epochs("p01")
    raw = (FULL / "full-p01.json").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == FULL_P01_SHA256


def test_box_full_p_is_the_6_epoch_runs_record(reg):
    """Box full-p (DECISIONS H14): origin/main's entry again, the record of P-0.3B's 6-epoch run (26.53 h, est 30 / max
    42, no banks: recipe v2), its items as the box ran them; the run's cooldown re-run is box p-cool's, which needs the
    config the run trained with byte for byte."""
    import hashlib
    it = items(reg, "full-p")
    quant = [f"quant-{f}-full-p03" for f in QUANT_FORMATS]
    assert list(it) == ["stores-ctc", "full-p03", "m4-full-p03", *quant, *WHISPER_KEYS]
    assert (it["full-p03"]["max_hours"], it["full-p03"]["config"], it["full-p03"]["continues"]) == (
        RECORD_RUN_H["full-p"]["full-p03"], "configs/full/full-p03.json", None)
    assert fullrun.box_extra_dirs("full-p", reg) == [] and "full-p" not in M.HOURS_BOXES
    raw_entry = json.loads((FULL / "boxes.json").read_text(encoding="utf-8"))["boxes"]["full-p"]
    assert "continues on box p-cool" in raw_entry["_comment"] and "DECISIONS H14" in raw_entry["_comment"]
    assert M.FULL_RUNS["p03"]["epochs"] == cfg("full-p03")["schedule"]["epochs"] == 6 == M.launch_epochs("p03")
    assert M.FULL_RUNS["p03"]["augment"] is M.RECIPE  # its config keeps recipe v2: the re-run sets its recipe
    raw = (FULL / "full-p03.json").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == FULL_P03_SHA256


# the continuations' records (DECISIONS H14): source box, run id, from / to step, resets before, epochs, keep_cooldown,
# recipe, baseline revision
COOL = {"p03-cool": ("full-p", "p03", P03_RID, 169376, 211720, 0, 6, False, "RECIPE_GENTLE",
                     "3674d2d7ae7480594b079234b1549014fe96f783"),
        "p01-cool": ("p01", "p01", P01_RID, 86328, 107910, 1, 4, False, "RECIPE_V4_FULL",
                     "c4604304db76e068df7bbe39d00d006b74d6c134"),
        "p005-cool": ("p005", "p005", P005_RID, 67449, 80939, 0, 10, True, "RECIPE_GENTLE",
                      "d2caddfc308c10495253c0204c4edf8a6222dac0")}


def test_the_cooldown_recipes(trainer):
    """RECIPE_V4_FULL = recipe v4 with the trainer's defaults of the acoustic ranges and the short-row guard written
    out (so a continuation inherits none from the state it resumes); RECIPE_GENTLE = recipe v4 at milder ranges, no
    GSM, and no background under rows shorter than 3 s (DECISIONS H14). Both validate on a CTC student."""
    d = trainer.DEFAULTS["augment"]
    ranges = ("noise_snr_db", "speech_snr_db", "gain_db", "codecs", "background_min_row_s")
    assert M.RECIPE_V4_FULL == dict(M.RECIPE_V4, **{k: d[k] for k in ranges})
    assert {k: M.RECIPE_V4_FULL[k] for k in ranges} == {
        "noise_snr_db": [0.0, 20.0], "speech_snr_db": [10.0, 25.0], "gain_db": [-20.0, 10.0],
        "codecs": ["mp3", "gsm", "ulaw8k"], "background_min_row_s": 0.0}
    assert M.RECIPE_GENTLE == dict(M.RECIPE_V4, noise_snr_db=[5.0, 20.0], speech_snr_db=[15.0, 25.0],
                                   gain_db=[-10.0, 10.0], codecs=["mp3", "ulaw8k"], background_min_row_s=3.0)
    assert list(M.RECIPE_V4_FULL)[:len(M.RECIPE_V4)] == list(M.RECIPE_V4)  # recipe v4's keys first, in its order
    for recipe in (M.RECIPE_V4_FULL, M.RECIPE_GENTLE):
        c = trainer.load_config(str(FULL / "full-p01.json"), [f"augment.{k}={json.dumps(v)}" for k, v in recipe.items()])
        assert c["augment"] == dict(d, **recipe)


def test_box_p_cool_runs_the_cooldown_reruns(reg, plan, trainer, capsys):
    """Box p-cool (DECISIONS H14): one rental, data-p01, one CTC store, then per continuation in CONTINUATIONS' order
    (P-0.3B, P-0.1B, P-0.05B) the run with its continues block, its readout and its 7 quantised readouts - 28 items.
    Each train item's continues block is continues_block's (the source box with the same-name, same-config item, the
    run id, the steps, the resets before, the sets), its max_hours box_hours', and its sets - as the queue passes them
    (fullrun.continue_sets, after schedule.resume_reset=true) - are ones the trainer takes on a resume of the run's
    config, changing only augment.* and the schedule keys."""
    it = items(reg, "p-cool")
    assert list(it) == ["stores-ctc", *[n for x in ("p03", "p01", "p005") for n in (
        f"full-{x}", f"m4-full-{x}", *(f"quant-{f}-full-{x}" for f in QUANT_FORMATS))]]
    assert len(it) == 28 and it["stores-ctc"]["config"] == "configs/full/full-p03.json"
    assert it["stores-ctc"]["needs"] == [] and it["stores-ctc"]["sets"] == []
    assert fullrun.box_students("p-cool", reg) == fullrun.box_ctc_students("p-cool", reg) == [
        "students/study/p03", "students/study/p01", "students/study/p005"]
    assert list(fullrun.continues_of("p-cool", reg)) == ["full-p03", "full-p01", "full-p005"]
    assert [c["box"] for c in M.CONTINUATIONS.values()] == ["p-cool"] * 3
    for name, (src, x, rid, frm, to, before, epochs, keep, recipe, rev) in COOL.items():
        c = M.CONTINUATIONS[name]
        assert (c["source_box"], c["student"], c["run_id"], c["from_step"], c["to_step"], c["resets_before"],
                c["epochs"], c["keep_cooldown"], c["augment"], c["revision"]) == (
            src, x, rid, frm, to, before, epochs, keep, getattr(M, recipe), rev), name
        run_items(it, x, RUN_H["p-cool"][f"full-{x}"])
        block = it[f"full-{x}"]["continues"]
        assert block == M.continues_block(name) == fullrun.continues_of("p-cool", reg)[f"full-{x}"]
        assert block == dict(box=src, run_id=rid, from_step=frm, to_step=to, resets_before=before,
                             sets=M.continuation_set_dict(name))
        # the source box runs the same item with the same config (its Hub summary records the run)
        assert items(reg, src)[f"full-{x}"]["config"] == it[f"full-{x}"]["config"] == f"configs/full/full-{x}.json"
        # the sets: native JSON values in continuation_set_dict's order, the queue's spelling of them
        sets = M.continuation_set_dict(name)
        assert list(sets)[:2] == ["schedule.epochs", "schedule.deadline_cooldown"] and sets["schedule.epochs"] == epochs
        assert sets["schedule.deadline_cooldown"] is False
        assert sets.get("schedule.resume_reset_keep_cooldown") is (True if keep else None)
        assert {k[len("augment."):]: v for k, v in sets.items() if k.startswith("augment.")} == c["augment"]
        assert fullrun.continue_sets(block) == COOL_SETS[name], name
        assert {k for k in sets} <= set(fullrun.CONTINUE_SET_KEYS)
        # ... taken on a resume of the run's config: none RESUME_FIXED, only augment.* and the schedule keys change,
        # and the config validates with them (the augmentation on a CTC student)
        saved = trainer.load_config(str(FULL / f"full-{x}.json"), [])
        words = ["schedule.resume_reset=true", *COOL_SETS[name]]
        changed, same = trainer.resume_overrides(saved, [trainer.apply_set(copy.deepcopy(saved), w) for w in words])
        assert changed and all(k.startswith(("augment.", "schedule.")) for k in changed), (name, changed)
        assert {"schedule.resume_reset": True, "schedule.deadline_cooldown": False} == {
            k: changed[k] for k in ("schedule.resume_reset", "schedule.deadline_cooldown")}
        assert ("schedule.resume_reset_keep_cooldown" in changed) is keep
        assert same.get("schedule.epochs") == epochs  # the run's own epochs: the plan's T is the run's
        got = trainer.load_config(str(FULL / f"full-{x}.json"), words)
        assert got["augment"] == dict(trainer.DEFAULTS["augment"], **c["augment"]) and trainer.augment_on(got)
        assert got["schedule"]["deadline_cooldown"] is False and got["schedule"]["epochs"] == epochs
        assert got["schedule"]["resume_reset_keep_cooldown"] is keep
    # P-0.3B's changes: its config's recipe v2 stays, recipe v4's other keys and the gentle ranges come in
    saved = trainer.load_config(str(FULL / "full-p03.json"), [])
    words = ["schedule.resume_reset=true", *COOL_SETS["p03-cool"]]
    changed, same = trainer.resume_overrides(saved, [trainer.apply_set(copy.deepcopy(saved), w) for w in words])
    assert {k for k in same if k.startswith("augment.")} == {"augment.enabled", "augment.truncate_p",
                                                             "augment.concat_p", "augment.mix_p",
                                                             "augment.truncate_min_row_s"}  # recipe v2's, the run's own
    assert changed["augment.codecs"] == ["mp3", "ulaw8k"] and changed["augment.background_min_row_s"] == 3.0
    # P-0.1B's: box 1's config has no augment block, so every recipe key but mix_p (0.0, the default) and the ranges
    # at their defaults are repeats
    saved = trainer.load_config(str(FULL / "full-p01.json"), [])
    words = ["schedule.resume_reset=true", *COOL_SETS["p01-cool"]]
    changed, same = trainer.resume_overrides(saved, [trainer.apply_set(copy.deepcopy(saved), w) for w in words])
    assert {k for k in same if k.startswith("augment.")} == {"augment.mix_p", "augment.noise_snr_db",
                                                             "augment.speech_snr_db", "augment.gain_db",
                                                             "augment.codecs", "augment.background_min_row_s"}
    # the box: the banks, its readout dirs (m4-<run_id>-r<resets_before + 1>), no reset or set flags, always a resume
    assert fullrun.box_extra_dirs("p-cool", reg) == [M.NOISE_BANK, M.RIR_BANK]
    assert [f"runs/m4-{c['run_id']}-r{c['resets_before'] + 1}" for c in M.CONTINUATIONS.values()] == [
        f"runs/m4-{P03_RID}-r1", f"runs/m4-{P01_RID}-r2", f"runs/m4-{P005_RID}-r1"]
    assert fullrun.main(["has-continues", "--box", "p-cool"]) == 0 and capsys.readouterr().out.split() == ["1"]
    for b in ("p01", "full-p", "p005", "full-t"):
        assert fullrun.main(["has-continues", "--box", b]) == 0 and capsys.readouterr().out.split() == ["0"], b
    raw_entry = json.loads((FULL / "boxes.json").read_text(encoding="utf-8"))["boxes"]["p-cool"]
    assert "DECISIONS H14" in raw_entry["_comment"] and "est 11.8" in raw_entry["_comment"]
    assert "max 19" in raw_entry["_comment"]
    # the queue's store is P-0.3B's config's: the three configs share the full data (the registry holds each item's
    # data keys equal to data-p01's)
    for x in ("p03", "p01", "p005"):
        assert {k: cfg(f"full-{x}").get(k) for k in fullrun.DATA_KEYS} == {
            k: cfg("full-p03").get(k) for k in fullrun.DATA_KEYS}, x


# box 1's config as it ran (7715f3f): a continuation needs the very same file (check-resume, the planner)
FULL_P01_SHA256 = "918d8e020d1ff6c70789297e93963aefa40e3f6431f1c6d7cb9de8b1c5e5ee1d"
# P-0.3B's config as its 6-epoch run trained with it (box full-p at 30b13f3): its cooldown re-run needs the same file
FULL_P03_SHA256 = "23d0576681df372226a0a828bc90510b7deac00d9dbd806155b6ad477ba579cf"
# P-0.05B's config as box p005's 10-epoch run trained with it (b6d48d2, DECISIONS H12; origin/main's at the launch):
# its cooldown re-run on box p-cool needs the same file
FULL_P005_SHA256 = "d1c5b726a89e8661db7aa2c8120d85c74ae226e43a136fcc839a8fd3ba0fb68c"


def continues_resume_plan(state_dir: Path, box: str, reg) -> None:
    """A box with continues items never registers a fresh queue (DECISIONS H14: it always resumes): the empty
    resume plan resume-pull would write first, so the tests' queues of box p-cool register."""
    if fullrun.continues_of(box, reg):
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / fullrun.RESUME_PLAN).write_text(json.dumps({"format": 1, "box": box, "items": {}}),
                                                     encoding="utf-8")


@pytest.mark.parametrize("box", ["p01", "full-t", "full-p", "p005", "p-cool"])
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
        assert box != "full-t" or n != "m4-full-t06" or r["max_hours"] * 60 + M.END_PHASE_SLACK_MIN > 30
        state = tmp_path / n
        state.mkdir()
        continues_resume_plan(state, box, reg)
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


def test_boxes_t_and_p(reg, plan):
    """Box 2 as two 1x boxes (DECISIONS G2). full-t: the AED store (nothing to wait for), T-0.6B, its M4, its 7
    quantised readouts, the decision-22 re-time pair. full-p (DECISIONS H1 step 2): the CTC store, P-0.3B with the
    recipe (full-p03.json's augment block), its M4, its 7 quantised readouts and the 4 Whisper models; P-0.05B is
    postponed (H2): its config stays, none of its items is on the box. No speed item on full-p (the chart's speed
    numbers are smoke-B #2's host). Box full-t's hours are box_hours of the speed record (--check holds them); box
    full-p's are its 6-epoch run's record (DECISIONS H14)."""
    t, p = items(reg, "full-t"), items(reg, "full-p")
    quant = {x: [f"quant-{f}-full-{x}" for f in QUANT_FORMATS] for x in ("p03", "t06")}
    assert list(t) == ["stores-aed", "full-t06", "m4-full-t06", *quant["t06"], "speed-full-t06", "speed-study-t06"]
    assert list(p) == ["stores-ctc", "full-p03", "m4-full-p03", *quant["p03"], *WHISPER_KEYS]
    assert not any("p005" in n for n in p) and (FULL / "full-p005.json").is_file()  # postponed, its config kept
    assert (t["stores-aed"]["config"], t["stores-aed"]["needs"], p["stores-ctc"]["config"]) == (
        "configs/full/full-t06.json", [], "configs/full/full-p03.json")
    for it, x, st, drop in ((t, "t06", "stores-aed", False), (p, "p03", "stores-ctc", False)):
        r = it[f"full-{x}"]
        assert (r["config"], r["study_run"], r["family"], r["max_hours"], r["needs"], r["droppable"]) == (
            f"configs/full/full-{x}.json", f"study-{x}", "aed" if x == "t06" else "ctc",
            {**RUN_H["full-t"], **RECORD_RUN_H["full-p"]}[f"full-{x}"], [st],
            drop), x
        assert it[f"m4-full-{x}"]["max_hours"] == (0.75 if x == "t06" else 0.3)
    for x, names in quant.items():
        it = t if x == "t06" else p
        for f, n in zip(QUANT_FORMATS, names):
            q = it[n]
            assert (q["kind"], q["argv"], q["of"], q["of_box"], set(q["needs"])) == (
                "eval", quant_argv(f), f"full-{x}", None, {f"m4-full-{x}", f"full-{x}"}), n
            assert q["max_hours"] == (0.75 if x == "t06" else 0.3), n
    for k, h in zip(WHISPER_KEYS, (0.75, 0.5, 0.5, 0.3)):
        assert (p[k]["argv"], p[k]["max_hours"], p[k]["weights"], p[k]["of"]) == (whisper_argv(k), h, [], None)
    a, b = t["speed-full-t06"], t["speed-study-t06"]
    assert (a["system"], a["speed_kind"], a["of"], set(a["needs"]), a["max_hours"], a["droppable"]) == (
        "full-t06", "aed", "full-t06", {"m4-full-t06", "full-t06"}, 0.2, True)
    assert (b["system"], b["speed_kind"], b["weights"], b["max_hours"]) == (
        "study-t06", "aed", [{"name": "study-t06", "run_id": STUDY_WEIGHTS["study-t06"][0], "step": 9370}], 0.2)
    assert not any(v["kind"] == "speed" for v in p.values())
    for box in M.HOURS_BOXES:  # full-t, p005, p-cool
        h = M.box_hours(plan, M.load_speed(), box, reserve_min=fullrun.box_spec(box, reg)["deadline_reserve_min"])
        assert (fullrun.box_spec(box, reg)["est_hours"], fullrun.box_spec(box, reg)["max_hours"]) == (
            h["est_hours"], h["max_hours"]) == BOX_H[box]


# DECISIONS H15 changed box full-t and full-t06.json only: every other box's registry entry and every other generated
# config as the t-box branch's merge of PRs #51, #52 and #53 (b697737) left them - a box entry's sha256 over its
# canonical JSON (json.dumps sort_keys, ensure_ascii False), a config's over its bytes with LF line ends
H15_OTHER_BOXES_SHA256 = {
    "full-smoke": "49df1a701e8db2465d167d531caad6ca9a69cfebb4803277e9ca3306f82da36b",
    "p01": "8685854282a57963fa2460b23eb2fcf1cb5a4ae1e4e25ad796cb1cc6bd12fe15",
    "full-p": "07d1c82a3badad60212e006f304cb898f7bcac9835a064e779b00ed663c8a00b",
    "p005": "6447bed9ee641def5af71a80bcf91a20608699f5acfe57aed85e27f31d394c96",
    "p-cool": "adcf8381134446435ef7e18e44b00494dba08932c91aa6fd63a3e940d9edc612",
    "smoke-b": "73bf187a1159ec2de9bed1f7cbcfe7739e729fe196b36466d852e67ca2bd3d17",
    "p01-chain": "8e8a1a4e0780f32fdd60f45fb865d473929da412df8ccb3c8c17f90fc0175d10",
}
H15_OTHER_CONFIGS_SHA256 = {
    "full-p03": FULL_P03_SHA256, "full-p01": FULL_P01_SHA256, "full-p005": FULL_P005_SHA256,
    **{f"smoke-{x}": h for x, h in SMOKE_SHA256.items()},
    "data-p01": "f8ff78dd0724e437420167bf76cf45f032cc1c739594f355ef6a4f172c635ce8",
    "data-t": "47a0d5e8e02d66aa6af69c74f543eef5b8a0beaf12bc4f96d91d9090279dae46",
    "data-p": "5be8cdda70cb7212a6eb2b6b0c5c358b6825405c44900d7f01478bfaa713ed1e",
    "data-full": "e8cc5b0f3a395516730ed2b632807190c76b6487a9fa8b9a32f9926f415e505b",
    "data-smoke": "be08b2bd4693d0e1a456658016ce4afa274ad76b744e1f09f55502609eb14f19",
    "data-smoke-b": "de1abaa289f690cf6cf0d537f244a4d31bda850b5a5460d57c65d0c45278bdd2",
}


def test_box_t_is_decisions_h15(reg, plan, trainer):
    """DECISIONS H15 (the owner, 2026-10-10: "lets do the transcribe run anyway so we have a full checkpoint for
    transcribe too"): box full-t trains T-0.6B from step 0 at 6 epochs (8 under H7) with COMMON's early stop, on
    RECIPE_AED_H15 - the registered AED recipe (RECIPE_AED: recipe v4 plus the cut table, H12) with
    augment.guard_per_piece and augment.end_trim_voiced on and augment.cut_keep_word off (the trainer's default false:
    not in the block) -, its hours box_hours at the 6-epoch plan's 143,891 steps, its price cap the owner's $1.40/h.
    No other box's entry, no other generated config and no other student's plan changed."""
    import hashlib
    c = cfg("full-t06")
    a = c["augment"]
    assert (a["guard_per_piece"], a["end_trim_voiced"]) == (True, True) and "cut_keep_word" not in a
    assert a == M.RECIPE_AED_H15 == dict(M.RECIPE_AED, guard_per_piece=True, end_trim_voiced=True)
    assert list(a)[-2:] == ["guard_per_piece", "end_trim_voiced"] and list(c)[-1] == "augment"
    assert (a["cuts"], a["cuts_sha256"]) == (M.AED_CUTS, M.AED_CUTS_SHA256)  # the AED recipe's cut table, pinned
    assert M.FULL_RUNS["t06"]["augment"] is M.RECIPE_AED_H15
    assert "guard_per_piece" not in M.RECIPE_AED and "end_trim_voiced" not in M.RECIPE_AED  # H12's record stays
    assert c["schedule"]["epochs"] == 6 == M.FULL_RUNS["t06"]["epochs"] == M.launch_epochs("t06")
    sc = c["schedule"]
    assert (sc["clock"], sc["max_steps"], sc["end_reserve_min"], sc["deadline_cooldown"]) == ("epochs", None, 55, True)
    assert c["early_stop"] == M.COMMON["early_stop"] and (c["early_stop"]["enabled"], c["early_stop"]["metric"],
                                                          c["early_stop"]["patience"], c["early_stop"]["action"]) == (
        True, "dev_ce", 6, "cooldown")
    t = trainer.load_config(str(FULL / "full-t06.json"), [])
    assert not trainer.is_ctc(t) and trainer.augment_on(t)
    assert (t["augment"]["guard_per_piece"], t["augment"]["end_trim_voiced"], t["augment"]["cut_keep_word"]) == (
        True, True, False)
    assert t["augment"] == dict(trainer.DEFAULTS["augment"], **M.RECIPE_AED_H15)
    # the plan: 6 epochs on full.parquet, each epoch as long as H7's 8-epoch plan's (the seeded planner), the WSD
    # cooldown from ceil(0.8 T) + 1
    s = plan["launch"]["students"]["t06"]
    assert (s["epochs"], s["total_steps"], s["cooldown_start_step"]) == (6, 143891, math.ceil(0.8 * 143891) + 1) == (
        6, 143891, 115114)
    assert s["steps_per_epoch"] == [23982, 23982, 23982, 23982, 23982, 23981]
    # the registry: the speed record's hours (--check holds them), the owner's cap, the rest as H7 / H11 / H12 left it
    spec = fullrun.box_spec("full-t", reg)
    assert (spec["est_hours"], spec["max_hours"], spec["max_dph"]) == (*BOX_H["full-t"], 1.4) == (50.9, 68, 1.4)
    assert items(reg, "full-t")["full-t06"]["max_hours"] == RUN_H["full-t"]["full-t06"] == 47.75
    entry = json.loads((FULL / "boxes.json").read_text(encoding="utf-8"))["boxes"]["full-t"]
    for w in ("DECISIONS H15", "6 epochs", "143,891", "115,114", "47.75", "45.49", "est 50.9", "max 68", "1.86 h",
              "4 d", "RECIPE_AED_H15", "guard_per_piece", "end_trim_voiced", "cut_keep_word", "$1.40", "t-box",
              "--allow-unverified-quant"):
        assert w in entry["_comment"], w
    for w in ("191,855", "63.62", "60.61", "66.8", "87.82", "4.17"):  # H7's 8-epoch numbers are gone
        assert w not in entry["_comment"], w
    # nothing else moved
    raw = json.loads((FULL / "boxes.json").read_text(encoding="utf-8"))["boxes"]
    assert set(raw) == {"full-t", *H15_OTHER_BOXES_SHA256}
    for b, h in H15_OTHER_BOXES_SHA256.items():
        canonical = json.dumps(raw[b], sort_keys=True, ensure_ascii=False).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == h, b
    assert set(H15_OTHER_CONFIGS_SHA256) == set(GENERATED) - {"full-t06"}
    for n, h in H15_OTHER_CONFIGS_SHA256.items():
        assert hashlib.sha256((FULL / f"{n}.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest() == h, n
    assert {x: plan["launch"]["students"][x]["total_steps"] for x in ("p03", "p01", "p005")} == {
        "p03": 211720, "p01": 107910, "p005": 269777}


def test_the_retired_box_2s_pool_lives_on_exactly_one_new_box(reg):
    """The old box full's 28 quantised readouts, 4 Whisper models and 2 speed items: each on exactly one of full-t,
    full-p and p01 (7 / 7 + 4 / 7, the re-time pair on full-t); P-0.05B's 7, postponed with it (DECISIONS H2), came
    back with the P test box (H8, H10): on box p005."""
    old = [f"quant-{f}-full-{x}" for x in ("p03", "p01", "t06") for f in QUANT_FORMATS]
    old += [*WHISPER_KEYS, "speed-full-t06", "speed-study-t06"]
    where = {n: [b for b in ("full-t", "full-p", "p01") if n in items(reg, b)] for n in old}
    assert all(len(v) == 1 for v in where.values()), {n: v for n, v in where.items() if len(v) != 1}
    assert [b for b in ("full-t", "full-p", "p01", "p005", "p-cool") if any(f"quant-{f}-full-p005" in items(reg, b)
                                                                 for f in QUANT_FORMATS)] == ["p005", "p-cool"]
    count = {b: sum(1 for v in where.values() if v == [b]) for b in ("full-t", "full-p", "p01")}
    assert count == {"full-t": 9, "full-p": 11, "p01": 7}
    # box p-cool's 21 are the cooldown re-runs' own readouts (its own runs' weights, DECISIONS H14), next to the records
    assert [n for n in items(reg, "p-cool") if n.startswith("quant-")] == [
        f"quant-{f}-full-{x}" for x in ("p03", "p01", "p005") for f in QUANT_FORMATS]


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

    for box in [b for b in fullrun.BOX_NAMES if b in reg["boxes"]]:
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
    for box in [b for b in fullrun.BOX_NAMES if b in reg["boxes"]]:
        spec = fullrun.box_spec(box, reg)
        s = F.FullSettings(root=ROOT, state_dir=tmp_path / box, gpus=[str(i) for i in range(spec["gpus"])],
                           n_gpus=None, python="python", out_repo="Multy123/kitsune-runs",
                           scratch_repo="Multy123/kitsune-scratch", uploader=object(), machine_id=None,
                           proc_root=tmp_path / "no-proc", cgroup=tmp_path / "no-cgroup")
        continues_resume_plan(tmp_path / box, box, reg)
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
                        {"stores", "train", "readout", "eval"} if box in ("p01", "full-p", "p005", "p-cool") else
                        {"stores", "train", "readout", "eval", "speed"} if box == "full-t" else
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
        extra_dirs=[PARAKEET, M.NOISE_BANK, M.RIR_BANK], smoke=False, faults=[], items=[], max_attempts=4,
        min_ram_gb=96)
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
    # box p01 is P-0.1B's continuation box now (DECISIONS H9, H10): the banks of its recipe v4 are extra dirs
    assert v2 == dict(stage=2, parts=["p01"], gate_box=None, rebuild="configs/full/data-p01.json",
                      data_configs={"p01": "configs/full/data-p01.json"}, students=["students/study/p01"],
                      ctc_students=["students/study/p01"], extra_files=[manifest, full_side],
                      extra_dirs=[M.NOISE_BANK, M.RIR_BANK], timed_states=True,
                      watchdog={"orphan_s": 3600, "action": "stop"})
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
    """The chain's hours, disk and price against its parts (addendum E.1.4 rule 3, E.6, E.9.5), as it ran on
    2026-10-01 (history now: its part p01 is the record of H9's 8-epoch plan, 14.3 h, with box 1's numbers in E.6's): box
    p01 fits after stage 1's sub-deadline, the 35 h cap covered the worst case the gate let through, 25.2 h was the
    central wall, the gate part ends by its standalone cap, the extra disk holds box p01's and stage 1's leftovers."""
    from kitsune import full_queue as F

    spec, parts = fullrun.box_spec(CHAIN, reg), {b: fullrun.box_spec(b, reg) for b in ("full-smoke", "smoke-b", "p01")}
    s1 = spec["chain"][0]
    readout = sum(it["max_hours"] for it in parts["p01"]["items"] if it["kind"] == "readout")  # m4-full-p01 0.3
    assert readout == 0.3
    assert spec["max_hours"] - s1["max_hours"] >= parts["p01"]["est_hours"]  # rule 3: 35 - 10.5 = 24.5 >= 14.3
    assert s1["max_hours"] + parts["p01"]["est_hours"] == 24.8  # launch refuses a --max-hours below this
    assert s1["gate_by_hours"] == parts["full-smoke"]["max_hours"] == 9 < s1["max_hours"]
    # E.6: stage 1 + the pessimistic stage-2 setup 6.1 h [X] + smoke check 3's box-1 limit + readout + end 0.35 h +
    # p01's reserve; E.9.5's central wall: smoke A 5.2 + smoke B's items 1.1 + stage-2 setup 4.7 + full-p01 13.56 +
    # readout + end (the plan's hours [X])
    assert math.isclose(s1["max_hours"] + 6.1 + F.BOX1_H_MAX + readout + 0.35
                        + parts["p01"]["deadline_reserve_min"] / 60, spec["max_hours"])
    assert math.isclose(5.2 + 1.1 + 4.7 + 13.56 + readout + 0.35, spec["est_hours"], abs_tol=0.05)
    assert spec["extra_gb"] - parts["p01"]["extra_gb"] == 85  # stage 1's leftovers on box p01's disk (95 for box 1)
    assert spec["max_dph"] == parts["full-smoke"]["max_dph"] == 1.0 and parts["p01"]["max_dph"] == 1.1
    assert spec["min_ram_gb"] == parts["p01"]["min_ram_gb"] == 96  # the largest of its parts' (launch's RAM filter)
    # E.1.4 rule 2: only the gate part injects faults or merely alerts
    assert [b for b, p in parts.items() if p["faults"] or p["watchdog"]["action"] != "stop"] == ["full-smoke"]


def test_check_holds_the_chain(tmp_path, capsys):
    """make_full_configs --check validates the chain with the rest of the registry (fullrun.registry_problems)."""
    out = repo_copy(tmp_path)
    reg = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    reg["boxes"][CHAIN]["chain"][0]["max_hours"] = 31
    reg["boxes"]["smoke-b"]["watchdog"]["action"] = "alert"
    (out / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    assert sorted(M.check(out)) == sorted([
        f"boxes.json: boxes.{CHAIN}: max_hours 35 - stage 1's 31 leaves 4 h, below the last stage's est_hours 14.3",
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
        return json.dumps([{"id": 1, "machine_id": 70001, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 1,
                            "dph_total": 0.816, "reliability": 0.99, "verification": "verified",
                            "duration": 30 * 86400, "cpu_ram": 96000, "disk_space": 1568, "inet_down_cost": 0.00117,
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
        "KITSUNE_DPH": "0.8160", "KITSUNE_MACHINE_ID": "70001"}
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
    assert "RAM >= 96 GB (registry min_ram_gb)" in out and "cpu_ram>=90" in out  # box p01's, the chain's largest


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


def test_launch_rents_the_new_boxes_only_and_gates_their_quant_items(reg):
    """The retired 2x box full is not in the registry (launch --box full: "not in the registry", so the stale 2x box
    cannot be rented); boxes p01 and p005 carry quantised items, so smoke-B #2's quant go signal applies to them too,
    as to full-t and full-p; the boxes' hours come from the speed record (launch's warning while it is provisional)."""
    sys.path.insert(0, str(ROOT / "vast"))
    import launch

    with pytest.raises(fullrun.RegistryError, match="not in the registry"):
        fullrun.box_spec("full", reg)
    assert {b: len(launch.quant_items(fullrun.box_spec(b, reg))) for b in ("p01", "full-t", "full-p", "p005")} == {
        "p01": 7, "full-t": 7, "full-p": 7, "p005": 7}
    assert len(launch.quant_items(fullrun.box_spec("p-cool", reg))) == 21
    assert launch.quant_items(fullrun.box_spec("full-smoke", reg)) == []
    assert set(launch.SPEED_RECORD_BOXES) == set(M.HOURS_BOXES)
    assert [launch.min_rental_days(fullrun.box_spec(b, reg)["max_hours"]) for b in ("full-t", "full-p", "p01",
                                                                                    "p005", "p-cool")] == [
        4.0, 4.0, 4.0, 4.0, 4.0]
    # box T's 68 h cap at 6 epochs (DECISIONS H15) + RENTAL_MARGIN_DAYS, 3.33 d, is inside the 4 d floor (H7's 88 h cap
    # at 8 epochs needed 4.17 d)
    assert launch.min_rental_days(fullrun.box_spec("full-t", reg)["max_hours"]) == launch.MIN_RENTAL_DAYS
    assert round(fullrun.box_spec("full-t", reg)["max_hours"] / 24 + launch.RENTAL_MARGIN_DAYS, 2) == 3.33
    # the labels hold the data config's stem: no box's label prefix is another's
    labels = {b: f"kitsune-full-{b}-{Path(fullrun.box_spec(b, reg)['data_config']).stem}-" for b in BOX_TABLE}
    assert labels["full-t"] == "kitsune-full-full-t-data-t-" and labels["full-p"] == "kitsune-full-full-p-data-p-"
    assert (labels["p01"], labels["p005"]) == ("kitsune-full-p01-data-p01-", "kitsune-full-p005-data-p01-")
    assert labels["p-cool"] == "kitsune-full-p-cool-data-p01-"
    assert not any(a != b and labels[b].startswith(labels[a]) for a in labels for b in labels)
