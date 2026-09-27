"""tools/full_plan.py: the full students' step plans through kitsune.trainset.StepPlanner, on a synthetic selection
(and, with KITSUNE_STUDY_SELECTION pointing at the frozen study_1000h.parquet, parity with the study runs' plan events).
CPU only.
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT  # noqa: E402

from kitsune import trainset  # noqa: E402

spec = importlib.util.spec_from_file_location("full_plan", ROOT / "tools" / "full_plan.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def toy_selection(path: Path, n: int = 4000, seed: int = 0) -> Path:
    """Kept train rows with study-like durations (0.3-30 s, most short) and token counts, plus rows the planner must
    skip: dropped train rows, dev rows and eval rows."""
    rng = np.random.default_rng(seed)
    dur = np.clip(rng.lognormal(1.0, 0.7, n), 0.3, 30.0).astype(np.float32)
    n_tok = np.maximum(1, (dur * rng.uniform(3, 9, n))).astype(np.int32)
    n_tok[:5] = 250  # decoder inputs over max_dec_len 200: excluded by the aed planner
    split = np.array(["train"] * n, dtype=object)
    split[rng.choice(n, 200, replace=False)] = "dev"
    split[rng.choice(np.flatnonzero(split == "train"), 100, replace=False)] = "eval"
    keep = rng.random(n) > 0.1
    df = pd.DataFrame(dict(id=[f"s/{i:06d}" for i in range(n)], source=rng.choice(["a", "b"], n), split=split,
                           teacher_file="s/train-00000", duration=dur, n_tok=n_tok, truncated=False,
                           agree=np.float32(0), teacher_cer=np.float32(0), keep=keep,
                           reason=np.where(keep, "kept", "agree>0.5"), in_greedy_subset=False, in_probe=False))
    df.to_parquet(path)
    return path


def planner_of(sel: pd.DataFrame, family: str, micro: float, step: float) -> trainset.StepPlanner:
    rows = sel[sel["keep"] & (sel["split"] == "train")]
    utts = [trainset.Utt(i, s, float(d), int(t), 0, 0, 0)
            for i, s, d, t in zip(rows["id"], rows["source"], rows["duration"], rows["n_tok"])]
    return trainset.StepPlanner(utts, step_audio_s=step, micro_audio_s=micro,
                                max_dec_len=None if family == "ctc" else 200, pool_micro=50, seed=1234)


def test_equals_the_step_planner(tmp_path):
    """Per student the tool's plan is StepPlanner's over the kept split-train rows, in selection order, with the
    trainer's pool_micro and seed: steps per epoch, T, the step audio and the micro-batch stats."""
    path = toy_selection(tmp_path / "sel.parquet")
    sel = pd.read_parquet(path)
    out = tmp_path / "plan.json"
    assert fp.main(["--selection", str(path), "--student", "a=aed:60:200:2", "--student", "c=ctc:90:200:3",
                    "--json", str(out)]) == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    kept = sel[sel["keep"] & (sel["split"] == "train")]
    assert got["n_train_utts"] == len(kept) and got["train_hours"] == pytest.approx(kept["duration"].sum() / 3600)
    for name, family, micro, step, epochs in (("a", "aed", 60, 200, 2), ("c", "ctc", 90, 200, 3)):
        r, pl = got["students"][name], planner_of(sel, family, micro, step)
        plans = [pl.epoch_plan(e) for e in range(epochs)]
        s0 = pl.plan_stats(plans[0])
        assert r["steps_per_epoch"] == [len(p) for p in plans] and r["total_steps"] == sum(map(len, plans))
        assert r["step_real_s"] == {"mean": pytest.approx(s0["step_real_s_mean"]),
                                    "min": pytest.approx(s0["step_real_s_min"]),
                                    "max": pytest.approx(s0["step_real_s_max"])}
        assert r["micro_per_step"] == pytest.approx(s0["micro_per_step"])
        assert r["pad_eff"] == pytest.approx(s0["pad_eff_audio"])
        assert (r["micro_utts_max"], r["micro_targets_max"]) == (s0["micro_utts_max"], s0["micro_targets_max"])
        over = int((kept["n_tok"] + len(trainset.PROMPT) - 1 > 200).sum())
        assert r["excluded_dec_len"] == (over if family == "aed" else 0) == s0["excluded_dec_len"] and over > 0
        assert r["planner_fingerprint"] == pl.fingerprint
        assert r["steps_per_0p1_epoch"] == round(len(plans[0]) / 10, 1)
        assert r["cooldown_start_step"] == math.ceil(0.8 * r["total_steps"]) + 1
        assert r["study_step_real_s"] is None and r["delta_pct"] is None  # "a" / "c" are no study students
        assert ("pad_eff_dec" in r) == (family == "aed")
        # the probe shapes: epoch 0's micro-batches with the most rows, the longest one, the most targets
        mbs = [mb for st in plans[0] for mb in st]
        shapes = {s["name"]: s["durations"] for s in r["worst_shapes"]}
        assert list(shapes) == ["most_rows", "longest", "most_targets"]
        assert len(shapes["most_rows"]) == max(len(mb) for mb in mbs)
        assert shapes["longest"][0] == pytest.approx(max(pl.dur[mb].max() for mb in mbs), abs=1e-3)
        most = max(mbs, key=lambda mb: int(pl.n_tok[mb].sum()))
        assert shapes["most_targets"] == [round(float(x), 3) for x in sorted(pl.dur[most], reverse=True)]
        for d in shapes.values():
            assert d == sorted(d, reverse=True) and all(x > 0 for x in d)


def test_default_students_and_study_deltas(tmp_path, capsys):
    """Without --student: the four full students (p01 and p005 share one plan); a name starting with a study
    student's compares against that student's realised step audio."""
    path = toy_selection(tmp_path / "sel.parquet", n=3000)
    got = fp.run(path, fp.default_students() + [fp.parse_student("p03-1300=ctc:600:1300:1")], log=lambda *_: None)
    assert list(got["students"]) == ["t06", "p03", "p01", "p005", "p03-1300"]
    s = got["students"]
    assert (s["t06"]["family"], s["t06"]["micro_audio_s"], s["t06"]["step_audio_s"], s["t06"]["epochs"]) == \
        ("aed", 450, 1730, 3)
    assert (s["p03"]["micro_audio_s"], s["p03"]["step_audio_s"], s["p03"]["epochs"]) == (600, 1350, 3)
    assert {k: v for k, v in s["p01"].items()} == {k: v for k, v in s["p005"].items()}
    for name, study in (("t06", 1730.7220159744832), ("p03", 1176.470242226075), ("p01", 1538.4610859879444),
                        ("p03-1300", 1176.470242226075)):
        assert s[name]["study_step_real_s"] == study
        assert s[name]["delta_pct"] == pytest.approx(100 * (s[name]["step_real_s"]["mean"] - study) / study)
    assert fp.main(["--selection", str(path), "--student", "x=ctc:90:200:1"]) == 0
    assert "x ctc micro 90 step 200" in capsys.readouterr().out


def test_student_specs():
    assert fp.parse_student("p03=ctc:600:1350:3") == fp.Student("p03", "ctc", 600.0, 1350.0, 3)
    for bad in ("p03", "p03=rnnt:600:1350:3", "p03=ctc:600:1350", "p03=ctc:0:1350:3", "p03=ctc:600:1350:0",
                "=ctc:600:1350:3", "p03=ctc:a:1350:3"):
        with pytest.raises(Exception, match="NAME=FAMILY"):
            fp.parse_student(bad)
    with pytest.raises(SystemExit):
        fp.main(["--selection", "x.parquet", "--student", "a=ctc:1:1:1", "--student", "a=aed:1:1:1"])


@pytest.mark.skipif(not os.environ.get("KITSUNE_STUDY_SELECTION"),
                    reason="KITSUNE_STUDY_SELECTION (the frozen study_1000h.parquet) not set")
def test_the_study_plans_are_reproduced():
    """On the frozen study selection, at the study's batch settings, the tool gives the study runs' plan events
    exactly (epoch 0: steps and step_real_s_mean), and the full students' values reproduce the realised audio."""
    path = Path(os.environ["KITSUNE_STUDY_SELECTION"])
    students = [fp.parse_student(s) for s in ("t06-study=aed:600:1500:1", "p01-study=ctc:1600:1500:1",
                                              "p03-study=ctc:1200:1500:1")] + fp.default_students()
    got = fp.run(path, students, log=lambda *_: None)["students"]
    for name, steps, mean in (("t06-study", 2079, 1730.7220159744832), ("p01-study", 2340, 1538.4610859879444),
                              ("p03-study", 3060, 1176.470242226075)):
        assert got[name]["steps_per_epoch"] == [steps] and got[name]["step_real_s"]["mean"] == pytest.approx(mean)
    assert got["t06-study"]["excluded_dec_len"] == 62
    for name in ("t06", "p03", "p01", "p005"):
        assert abs(got[name]["delta_pct"]) < 0.1, (name, got[name]["delta_pct"])
