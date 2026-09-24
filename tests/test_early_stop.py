"""Early stopping in scripts/04_distill.py (config early_stop, the STOP file).

- the rule (early_stop_update): improvement = best - value > max(min_delta_abs, min_delta_rel * |best|), ties and
  non-finite values are no improvement, patience counting, min_evals, floor; the config validation
- the actions on a run: "stop" leaves the loop right after the triggering eval and the end phase still runs (final eval
  with the probe, verdict, summary.stopped_early, exit 0); "cooldown" starts the WSD cooldown at once and the run ends
  when it is over, on the steps, epochs and wall clocks; the STOP file stops any run
- the full state: a resume continues the patience count, one that had already triggered goes straight to the end
  phase, and a state written before early stopping existed still resumes
- disabled (the DEFAULTS) changes nothing: the same steps and the same losses as a run whose rule never fires
- verdict() on the short eval history an early stop leaves

CPU only (the laptop GPU runs other jobs), a tiny random student and a synthetic corpus in the real on-disk formats.
The metric is made "flat" deterministically with min_delta_abs = 1e9: only the first checked eval improves on it."""
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import load_script, make_fake_corpus, make_fake_selection  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
PEAK = 3e-3
FLAT = 1e9  # min_delta_abs: no eval after the first improves
EVENT_META = ("wall", "time", "elapsed_s", "step", "kind")


def events(run: Path, kind: str | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def merged(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merged(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def one_run(root: Path, name: str) -> Path:
    runs = list((root / "runs").glob(f"{name}-2*"))
    assert len(runs) == 1, runs
    return runs[0]


def summary(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def steps_of(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "metrics" / "steps.parquet")


def scalar(run: Path, tag: str) -> dict[int, float]:
    sc = pd.read_parquet(run / "metrics" / "scalars.parquet")
    sc = sc[sc["tag"] == tag]
    return dict(zip(sc["step"].astype(int), sc["value"]))


def rule(**kw) -> dict:
    return dict(dict(enabled=True, metric="heldout_kl", patience=3, min_delta_rel=0.0, min_delta_abs=0.0, min_evals=0,
                     floor=None, action="stop"), **kw)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + a tiny student saved exactly as 03 saves one, and a base config for it: the step
    clock, an eval after every step, no smoke phase, no periodic checkpoints."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("earlystop")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (36, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=7)
    sel = make_fake_selection(fc, greedy_n=4, probe_n=5)

    torch.manual_seed(0)
    cfg = CohereAsrConfig().to_dict()
    enc = dict(cfg["encoder_config"], num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=2,
               num_key_value_heads=2, subsampling_conv_channels=8)
    cfg.update(encoder_config=enc, num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=2,
               num_key_value_heads=2, head_dim=32)
    tcfg = CohereAsrConfig.from_dict(cfg)
    tcfg._attn_implementation = tcfg.encoder_config._attn_implementation = "sdpa"
    teacher = CohereAsrForConditionalGeneration(tcfg).eval()
    student = S.build_student(teacher, S.StudentSpec(enc_layers=[0, 1], ffn_dim=128, dec_layers=[0]), None)
    sdir = root / "student"
    S.save_student(student, sdir, proc, dict(format=1, stage="complete", spec=dict(enc_layers=[0, 1], ffn_dim=128,
                                                                                    dec_layers=[0], tie_head=True)))
    base = {
        "student": str(sdir), "data_root": str(fc.data), "teacher_root": str(fc.teacher_out),
        "second_root": str(fc.second_out), "selection": str(sel), "cache_dir": str(root / "cache"),
        "runs_root": str(root / "runs"), "sources": ["src_a", "src_b"], "eval_sets": EVAL,
        "device": "cpu", "autocast": "none",
        "optim": {"lr": PEAK},
        "schedule": {"warmup_steps": 2, "cooldown_frac": 0.5, "clock": "steps", "max_steps": 20},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_steps": 1, "greedy_subset": 2, "batch_s": 20, "check_baselines": False,
                 "final_full_greedy": False},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 1000, "keep_local": 5, "full_after_smoke": False},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
    }
    return dict(root=root, base=base, fc=fc)


def write_config(env, name: str, over: dict) -> str:
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------------------------------------- unit level


def test_improvement_rule_ties_patience_min_evals_and_floor():
    m = load_script("04_distill")

    def feed(values, **kw):
        es, ec = m.early_stop_state(), rule(**kw)
        return es, [m.early_stop_update(es, ec, v, step) for step, v in enumerate(values, start=1)]

    # (binary-exact values, so the ties are exact.) The first checked eval is the best whatever it is; the relative
    # delta dominates at best 1.0 (25 % = 0.25 > 0.125), and a tie with the threshold is no improvement ...
    es, out = feed([1.0, 0.8, 0.75, 0.625], min_delta_rel=0.25, min_delta_abs=0.125, patience=10)
    assert out == [None] * 4 and es["best"] == 0.625 and es["best_step"] == 4 and es["evals_since_best"] == 0
    es, _ = feed([1.0, 0.75], min_delta_rel=0.25, patience=10)
    assert es["best"] == 1.0 and es["best_step"] == 1 and es["evals_since_best"] == 1 and es["value"] == 0.75
    # ... the absolute one at best 0.5 (25 % = 0.125 < 0.25), a tie included; the best moves only on an improvement, so
    # small steps add up until one clears the threshold against the old best
    es, _ = feed([0.5, 0.375, 0.25, 0.125], min_delta_rel=0.25, min_delta_abs=0.25, patience=10)
    assert (es["best"], es["best_step"], es["evals_since_best"]) == (0.125, 4, 0)
    es, _ = feed([0.5, 0.375, 0.25], min_delta_rel=0.25, min_delta_abs=0.25, patience=10)
    assert (es["best"], es["best_step"], es["evals_since_best"]) == (0.5, 1, 2)
    es, _ = feed([1.0, 0.996, 0.992, 0.989], min_delta_rel=0.01, patience=10)
    assert (es["best"], es["best_step"], es["evals_since_best"]) == (0.989, 4, 0)
    es, _ = feed([1.0, 1.0], patience=10)  # equal is no improvement with zero deltas either
    assert es["evals_since_best"] == 1 and es["best_step"] == 1
    # missing and non-finite values never improve and never become the best
    es, out = feed([None, float("nan"), 2.0, float("inf"), 1.0], patience=10)
    assert es["best"] == 1.0 and es["best_step"] == 5 and es["evals"] == 5 and out == [None] * 5

    # patience: that many evals in a row without an improvement; an improvement starts the count again
    _, out = feed([1.0, 1.0, 1.0, 1.0], patience=3)
    assert out == [None, None, None, "patience"]
    es, out = feed([1.0, 1.0, 0.5, 0.5, 0.5], patience=2)
    assert out == [None, None, None, None, "patience"] and es["best_step"] == 3 and es["evals_since_best"] == 2
    # min_evals: no trigger before that many checked evals, then the count so far applies at once
    _, out = feed([1.0, 1.0, 1.0, 1.0, 1.0], patience=1, min_evals=4)
    assert out == [None, None, None, "patience", "patience"]
    # floor: value <= floor triggers without waiting for the patience, but not before min_evals either
    _, out = feed([0.4, 0.45, 0.6], floor=0.5, min_evals=2, patience=100)
    assert out == [None, "floor", None]
    _, out = feed([0.5], floor=0.5)
    assert out == ["floor"]
    _, out = feed([float("nan")], floor=0.5, patience=100)
    assert out == [None]


def test_config_keys_are_validated():
    m = load_script("04_distill")
    ok = m.load_config(None, ["early_stop.enabled=true", "early_stop.metric=train_loss", "early_stop.floor=0.01",
                              "early_stop.min_evals=0", "early_stop.action=stop"])
    assert ok["early_stop"]["floor"] == 0.01 and ok["early_stop"]["metric"] == "train_loss"
    for bad in (["early_stop.enabled=1"], ["early_stop.metric=loss"], ["early_stop.action=halt"],
                ["early_stop.patience=0"], ["early_stop.patience=2.5"], ["early_stop.min_evals=-1"],
                ["early_stop.min_evals=true"], ["early_stop.min_delta_rel=-0.1"], ["early_stop.min_delta_abs=x"],
                ["early_stop.floor=low"], ["early_stop.patiense=3"],
                ["early_stop.enabled=true", "early_stop.metric=probe_kl", "eval.probe=false"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
    m.load_config(None, ["early_stop.metric=probe_kl", "eval.probe=false"])  # disabled: no probe needed


def _unit_run(m, tmp_path, sets: list[str]):
    R = m.Run(cfg=m.load_config(None, sets), run_dir=tmp_path, device=torch.device("cpu"), amp=False)
    evs, sc = [], []
    R.log = SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw)),
                            scalars=lambda values, step: sc.append((step, dict(values))))
    return R, evs, sc


def test_cooldown_action_moves_the_schedule_on_every_clock(tmp_path, monkeypatch):
    """The early cooldown starts where it triggers and lasts cooldown_frac x the progress so far (whole steps on the
    step and epoch clocks, seconds on the wall clock); already in the cooldown, the schedule is left alone; a budget
    cut afterwards (fit_budget) still ends it."""
    m = load_script("04_distill")
    # steps: trigger after step 7 of 20 -> t_c 7, T 7 + ceil(0.3 * 7) = 10
    R, evs, sc = _unit_run(m, tmp_path, ["schedule.clock=steps", "schedule.max_steps=20", "schedule.cooldown_frac=0.3",
                                         "early_stop.enabled=true"])
    R.st["step"] = 7
    assert R.progress() == (7.0, 20.0) and R.cooldown_start(20.0) == pytest.approx(14.0)
    assert m.early_stop_trigger(R, "patience", "cooldown") is False  # the loop goes on
    assert R.st["early_stop"]["cooldown"] == dict(t_c=7.0, T=10.0, clock="steps", at_step=7)
    assert R.progress() == (7.0, 10.0) and R.cooldown_start(10.0) == 7.0
    assert evs[-1]["kind"] == "early_stop" and evs[-1]["cooldown"]["already"] is False
    assert evs[-1]["reason"] == "patience" and evs[-1]["action"] == "cooldown"
    assert sc[-1] == (7, {"early_stop/triggered": 1.0})
    lrs = [m.wsd_lr(1.0, s + 1, float(s), 10.0, 2, 0.3, t_c=7.0) for s in (6, 7, 8, 9)]
    assert lrs[0] == (1.0, 1) and lrs[1] == (1.0, 2) and lrs[2][0] == pytest.approx(1 - math.sqrt(1 / 3))
    assert lrs[3][0] == pytest.approx(1 - math.sqrt(2 / 3)) and lrs[3][1] == 2

    # already cooling down (t >= (1 - 0.3) * 20 = 14): no second cooldown, the scheduled end stays
    R, evs, _ = _unit_run(m, tmp_path, ["schedule.clock=steps", "schedule.max_steps=20", "schedule.cooldown_frac=0.3",
                                        "early_stop.enabled=true"])
    R.st["step"] = 15
    assert m.early_stop_trigger(R, "floor", "cooldown") is False
    assert R.st["early_stop"]["cooldown"] is None and R.progress() == (15.0, 20.0)
    assert evs[-1]["cooldown"] == dict(already=True, t_c=pytest.approx(14.0), T=20.0, clock="steps")
    assert R.st["early_stop"]["triggered"]["reason"] == "floor"

    # epochs: the planned total_steps is the budget; the plan ends early at a whole step
    R, _, _ = _unit_run(m, tmp_path, ["schedule.clock=epochs", "schedule.epochs=10", "schedule.cooldown_frac=0.2",
                                      "early_stop.enabled=true"])
    R.st.update(step=9, total_steps=40)
    m.early_stop_trigger(R, "patience", "cooldown")
    assert R.st["early_stop"]["cooldown"]["T"] == 11.0 and R.progress() == (9.0, 11.0)  # 9 + ceil(1.8)

    # wall: 1000 s trained of 4 h -> t_c 1000 s, T 1200 s; a later budget cut below that ends the run sooner
    monkeypatch.delenv("KITSUNE_DEADLINE", raising=False)
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    R, _, _ = _unit_run(m, tmp_path, ["early_stop.enabled=true"])
    R.st["train_s"] = 1000.0
    assert R.progress() == (1000.0, 4 * 3600.0)
    m.early_stop_trigger(R, "patience", "cooldown")
    assert R.st["early_stop"]["cooldown"] == dict(t_c=1000.0, T=pytest.approx(1200.0), clock="wall", at_step=0)
    assert R.progress() == (1000.0, pytest.approx(1200.0))
    R.budget_s = 1100.0
    assert R.progress() == (1000.0, 1100.0) and R.cooldown_start(1100.0) == 1000.0

    # "stop" and the STOP file end the loop; the STOP file works with early_stop disabled
    R, evs, _ = _unit_run(m, tmp_path / "run", [])
    (tmp_path / "run").mkdir()
    R.st["step"] = 3
    assert m.stop_requested(R) is False and not evs
    (tmp_path / "run" / "STOP").touch()
    assert m.stop_requested(R) is True and R.st["early_stop"]["stop"]
    assert {k: evs[-1][k] for k in ("kind", "reason", "action", "metric", "at_step")} == dict(
        kind="early_stop", reason="stop_file", action="stop", metric=None, at_step=3)


def test_verdict_on_a_short_history():
    """An early stop can leave very few eval records: verdict() must still return, with the trends unknown."""
    from kitsune import evaluate as ev

    fin = dict(sets={s: dict(cer_ref_corpus=0.1, teacher_cer_ref_corpus=0.1, n=10, n_truncated=0, trunc_rate=0.0)
                     for s in ev.GATE_SETS})
    rec = dict(step=0, elapsed_s=0.0, heldout_kl=1.0, probe_kl=0.9,
               greedy={s: dict(cer_ref_corpus=0.2, teacher_cer_ref_corpus=0.1, trunc_rate=0.0, n=10)
                       for s in ev.GATE_SETS})
    for n in (0, 1, 2):
        hist = [dict(rec, step=10 * i, heldout_kl=1.0 - 0.1 * i) for i in range(n)]
        v = ev.verdict(dict(final=fin, history=hist))
        assert v["verdict"] in ("GO", "PROMISING", "NO-GO", "INCONCLUSIVE")
        assert v["trend"]["window_steps"] == [r["step"] for r in hist]
        if n < 2:
            assert any("trend unknown" in r for r in v["reasons"]) and v["trend"]["gap_rel_change"] is None


# ------------------------------------------------------------------------------------------------ whole runs


def test_stop_action_runs_the_end_phase_and_a_crash_after_it_resumes_there(env, monkeypatch):
    """Held-out KL flat after the first in-loop eval, patience 3: the trigger is the eval after step 4 (the step-0 eval
    never counts). The loop ends right there, and the end phase still runs: final eval (with the probe) at step 4,
    verdict, summary.stopped_early, exit 0. A crash in that final eval leaves a full state that says "triggered": the
    resumed run trains no further step and goes straight to the end phase."""
    m = load_script("04_distill")
    path = write_config(env, "es-stop", {"early_stop": rule(patience=3, min_delta_abs=FLAT)})
    orig = m.run_eval

    def run_eval(R, step, final=False, **kw):
        if final:
            raise RuntimeError("simulated crash in the final eval")
        return orig(R, step, final, **kw)

    monkeypatch.setattr(m, "run_eval", run_eval)
    with pytest.raises(RuntimeError, match="final eval"):
        m.main(["--config", path])
    run = one_run(env["root"], "es-stop")
    s1 = summary(run)
    assert s1["status"] == "failed" and s1["steps"] == 4 and s1["stopped_early"]["reason"] == "patience"
    trainer = json.loads((run / "checkpoints" / "full_step_4" / "trainer.json").read_text(encoding="utf-8"))
    assert trainer["reason"] == "end" and trainer["st"]["early_stop"]["stop"] is True
    monkeypatch.setattr(m, "run_eval", orig)

    assert m.main(["--config", path, "--resume", str(run)]) == 0
    assert steps_of(run)["step"].tolist() == [1, 2, 3, 4]  # nothing trained after the trigger, before or after the crash
    (es,) = events(run, "early_stop")
    hist = summary(run)["history"]
    assert {k: v for k, v in es.items() if k not in EVENT_META} == dict(
        metric="heldout_kl", value=pytest.approx(hist[-1]["heldout_kl"], rel=1e-5),
        best=pytest.approx(hist[1]["heldout_kl"]), best_step=1, evals_since_best=3, reason="patience", action="stop",
        at_step=4, epoch=es["epoch"])
    assert 0 < es["epoch"] < 1
    ph = [e for e in events(run, "phase") if e["name"] in ("train", "end")]
    assert [(e["name"], e.get("skipped")) for e in ph] == [("train", None), ("end", None), ("train", "early_stop"),
                                                           ("end", None)]
    evals = [(e["at_step"], e["final"]) for e in events(run, "eval")]
    assert evals == [(0, False), (1, False), (2, False), (3, False), (4, False), (4, True)]
    assert next(e for e in events(run, "eval") if e["final"])["probe_kl"] is not None
    assert all((run / "evals" / "step_4" / f).is_file() for f in ("probe.parquet", "verdict.json"))

    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == 4 and s["resumes"] == 1
    assert s["stopped_early"] == {k: v for k, v in es.items() if k not in EVENT_META}
    assert s["verdict"]["verdict"] in ("GO", "PROMISING", "NO-GO", "INCONCLUSIVE") and events(run, "verdict")
    assert [r["step"] for r in hist] == [0, 1, 2, 3, 4]
    # the scalars: checked at the in-loop evals only, the step-0 eval never
    assert scalar(run, "early_stop/evals_since_best") == {1: 0, 2: 1, 3: 2, 4: 3}
    assert scalar(run, "early_stop/triggered") == {1: 0, 2: 0, 3: 0, 4: 1}
    best = scalar(run, "early_stop/best")
    assert sorted(best) == [1, 2, 3, 4] and all(v == pytest.approx(hist[1]["heldout_kl"]) for v in best.values())
    assert scalar(run, "early_stop/value")[4] == pytest.approx(hist[4]["heldout_kl"], rel=1e-5)


def test_train_loss_metric_and_floor(env):
    """metric train_loss = the mean loss/total of the steps since the previous eval (here 2 steps); a floor above every
    value triggers at the first eval allowed by min_evals, with reason "floor"."""
    m = load_script("04_distill")
    path = write_config(env, "es-floor", {"eval": {"every_steps": 2},
                                          "early_stop": rule(metric="train_loss", floor=1e9, min_evals=3, patience=100)})
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "es-floor")
    tot = dict(zip(steps_of(run)["step"], steps_of(run)["loss/total"]))
    assert sorted(tot) == [1, 2, 3, 4, 5, 6]
    val = scalar(run, "early_stop/value")
    assert sorted(val) == [2, 4, 6]
    for k in (2, 4, 6):
        assert val[k] == pytest.approx((tot[k - 1] + tot[k]) / 2, rel=1e-6)
    (es,) = events(run, "early_stop")
    assert es["reason"] == "floor" and es["at_step"] == 6 and es["metric"] == "train_loss"
    assert es["value"] == pytest.approx((tot[5] + tot[6]) / 2, rel=1e-6)
    assert summary(run)["stopped_early"]["reason"] == "floor"


def test_stop_file_stops_any_run(env, monkeypatch):
    """runs/<run_id>/STOP, created during step 3 of a run without early stopping and without in-loop evals: step 3
    finishes, no step 4 starts, and the end phase runs as after a full run."""
    m = load_script("04_distill")
    path = write_config(env, "es-file", {"eval": {"every_steps": 1000}})
    orig = m.train_step

    def train_step(R, step, lr, mbs, epoch):
        out = orig(R, step, lr, mbs, epoch)
        if step == 3:
            (R.run_dir / "STOP").touch()
        return out

    monkeypatch.setattr(m, "train_step", train_step)
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "es-file")
    assert steps_of(run)["step"].tolist() == [1, 2, 3]
    (es,) = events(run, "early_stop")
    assert es["reason"] == "stop_file" and es["action"] == "stop" and es["at_step"] == 3 and es["metric"] is None
    assert es["value"] is None and es["best"] is None
    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == 3 and s["stopped_early"]["reason"] == "stop_file"
    assert [r["step"] for r in s["history"]] == [0, 3] and s["verdict"]
    assert scalar(run, "early_stop/triggered") == {3: 1} and not scalar(run, "early_stop/value")


def test_cooldown_action_on_the_step_clock(env):
    """Trigger after step 4 of 20 (patience 3): the cooldown starts at once over ceil(0.5 * 4) = 2 steps, after a
    pre-cooldown full state, the LR follows 1 - sqrt from there, and the run ends after step 6 with the usual end
    phase (no in-loop eval at step 6: the final eval is at the same step)."""
    m = load_script("04_distill")
    path = write_config(env, "es-cool", {"early_stop": rule(patience=3, min_delta_abs=FLAT, action="cooldown")})
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "es-cool")
    st = steps_of(run)
    assert st["step"].tolist() == [1, 2, 3, 4, 5, 6]
    for step, lr, phase in zip(st["step"], st["opt/lr"], st["sched/phase"]):
        t = float(step - 1)
        want = (m.wsd_lr(PEAK, int(step), t, 20.0, 2, 0.5) if step <= 4
                else m.wsd_lr(PEAK, int(step), t, 6.0, 2, 0.5, t_c=4.0))
        assert lr == pytest.approx(want[0]) and phase == want[1], step
    assert st["opt/lr"].iloc[-1] == pytest.approx(PEAK * (1 - math.sqrt(0.5))) and st["sched/phase"].iloc[-1] == 2
    assert [(e["phase"], e["at_step"]) for e in events(run, "lr_phase")] == [("warmup", 1), ("stable", 2),
                                                                             ("cooldown", 5)]
    cool = [e for e in events(run, "phase") if e["name"] == "cooldown"]
    assert len(cool) == 1 and cool[0]["at_step"] == 4 and cool[0]["T"] == 6.0
    assert any(e["name"] == "full_step_4" and e["reason"] == "pre_cooldown" for e in events(run, "checkpoint"))
    (es,) = events(run, "early_stop")
    assert es["action"] == "cooldown" and es["at_step"] == 4 and es["reason"] == "patience"
    assert es["cooldown"] == dict(t_c=4.0, T=6.0, clock="steps", at_step=4, already=False)
    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == 6 and s["stopped_early"]["cooldown"]["T"] == 6.0
    assert [(e["at_step"], e["final"]) for e in events(run, "eval")] == [(i, False) for i in range(6)] + [(6, True)]
    assert sorted(scalar(run, "early_stop/value")) == [1, 2, 3, 4]  # not checked again after the trigger


def test_cooldown_action_on_the_epoch_clock(env):
    """Epoch clock, an eval after every epoch: flat after epoch 1, patience 2 -> the trigger is the end of epoch 3; the
    cooldown covers ceil(0.3 x the steps so far) more steps and the epoch plan ends early, mid-epoch if need be."""
    m = load_script("04_distill")
    path = write_config(env, "es-epochs", {
        "subset": {"train_audio_s": 8, "eval_audio_s": 4},
        "schedule": {"clock": "epochs", "epochs": 8, "warmup_steps": 300, "cooldown_frac": 0.3, "max_steps": None},
        "batch": {"step_audio_s": 4, "micro_audio_s": 3, "pool_micro": 4},
        "eval": {"every_steps": None, "every_epochs": 1, "probe_is_train": True},
        "early_stop": rule(metric="probe_kl", patience=2, min_delta_abs=FLAT, action="cooldown"),
    })
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "es-epochs")
    sched = events(run, "schedule")[0]
    per, total = sched["steps_per_epoch"], sched["total_steps"]
    s3 = sum(per[:3])
    end = s3 + math.ceil(0.3 * s3)
    assert s3 < (1 - 0.3) * total and end < total  # before the scheduled cooldown, and it cuts the plan short
    (es,) = events(run, "early_stop")
    assert es["at_step"] == s3 and es["epoch"] == 3.0 and es["metric"] == "probe_kl"
    assert es["cooldown"] == dict(t_c=float(s3), T=float(end), clock="epochs", at_step=s3, already=False)
    st = steps_of(run)
    assert st["step"].tolist() == list(range(1, end + 1))
    for step, lr, phase in zip(st["step"], st["opt/lr"], st["sched/phase"]):
        t = float(step - 1)
        want = (m.wsd_lr(PEAK, int(step), t, float(total), sched["warmup_steps"], 0.3) if step <= s3
                else m.wsd_lr(PEAK, int(step), t, float(end), sched["warmup_steps"], 0.3, t_c=float(s3)))
        assert lr == pytest.approx(want[0]) and phase == want[1], step
    s = summary(run)
    ends = np.cumsum(per).tolist()
    assert s["status"] == "complete" and s["steps"] == end and s["epochs"] < 8
    assert [r["step"] for r in s["history"]] == [0, *[e for e in ends if e < end], end]


def test_cooldown_action_on_the_wall_clock(env, monkeypatch):
    """Wall clock (a 1 h budget): the trigger after step 3 starts a cooldown over half the loop time so far; the run
    ends as soon as that time is up, far inside the budget, with the LR falling over the cooldown steps."""
    monkeypatch.delenv("KITSUNE_DEADLINE", raising=False)
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    m = load_script("04_distill")
    path = write_config(env, "es-wall", {
        "schedule": {"clock": "wall", "max_steps": None, "train_hours": 1.0, "cooldown_frac": 0.5},
        "early_stop": rule(patience=2, min_delta_abs=FLAT, action="cooldown"),
    })
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "es-wall")
    (es,) = events(run, "early_stop")
    cd = es["cooldown"]
    assert es["at_step"] == 3 and cd["clock"] == "wall" and not cd["already"]
    assert cd["T"] == pytest.approx(1.5 * cd["t_c"]) and 0 < cd["t_c"] < 600
    st = steps_of(run)
    after = st[st["step"] > 3]
    assert st["step"].tolist() == list(range(1, len(st) + 1)) and len(after) >= 1
    assert (after["sched/phase"] == 2).all() and (st[st["step"] <= 3]["sched/phase"] < 2).all()
    lr = after["opt/lr"].to_numpy()
    assert (lr <= PEAK).all() and (lr > 0).all() and (np.diff(lr) < 0).all()
    s = summary(run)
    assert s["status"] == "complete" and cd["T"] - 0.1 <= s["train_s"] < 600  # train_s is rounded to 0.1 s
    assert [(e["phase"], e["at_step"]) for e in events(run, "lr_phase")][-1] == ("cooldown", 4)


def test_resume_mid_patience_continues_the_count(env, monkeypatch):
    """Patience 5 with the metric flat after the first in-loop eval: an uninterrupted run triggers after step 6. A crash
    before step 5 and a resume from the step-3 full state (evals 3, 2 since the best) triggers at the same step."""
    m = load_script("04_distill")
    path = write_config(env, "es-resume", {"ckpt": {"full_every_steps": 3},
                                           "early_stop": rule(patience=5, min_delta_abs=FLAT)})
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "5")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", path])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    run = one_run(env["root"], "es-resume")
    trainer = json.loads((run / "checkpoints" / "full_step_3" / "trainer.json").read_text(encoding="utf-8"))
    saved = trainer["st"]["early_stop"]
    assert (saved["evals"], saved["evals_since_best"], saved["best_step"], saved["triggered"]) == (3, 2, 1, None)
    assert not events(run, "early_stop")

    assert m.main(["--config", path, "--resume", str(run / "checkpoints" / "full_step_3")]) == 0
    (es,) = events(run, "early_stop")
    assert es["at_step"] == 6 and es["evals_since_best"] == 5 and es["best_step"] == 1
    assert steps_of(run)["step"].tolist() == [1, 2, 3, 4, 5, 6]
    s = summary(run)
    assert s["resumes"] == 1 and s["steps"] == 6 and s["stopped_early"]["at_step"] == 6


def test_disabled_changes_nothing_and_old_states_resume(env, monkeypatch):
    """early_stop off (the DEFAULTS: the config does not mention it) vs on with a rule that never fires: the same steps,
    the same losses, LRs and eval results, bit for bit; the disabled run has no early-stop events or scalars and
    stopped_early null. A full state written before early stopping existed (no early_stop in its config or state)
    resumes with the defaults."""
    m = load_script("04_distill")
    over = {"schedule": {"max_steps": 8}, "eval": {"every_steps": 2}, "ckpt": {"full_every_steps": 4}}
    off = write_config(env, "es-off", over)
    on = write_config(env, "es-on", dict(over, early_stop=rule(metric="train_loss", patience=1000)))
    assert "early_stop" not in json.loads(Path(off).read_text(encoding="utf-8"))
    assert m.main(["--config", off]) == 0
    assert m.main(["--config", on]) == 0
    a, b = one_run(env["root"], "es-off"), one_run(env["root"], "es-on")
    sa, sb = steps_of(a), steps_of(b)
    assert sa["step"].tolist() == sb["step"].tolist() == list(range(1, 9))
    for col in ("loss/total", "loss/objective", "loss/kl", "loss/ce", "loss/l2sp", "opt/lr", "opt/grad_norm",
                "sched/phase", "tok/top1"):
        np.testing.assert_array_equal(sa[col].to_numpy(), sb[col].to_numpy(), err_msg=col)
    for step, lr, phase in zip(sa["step"], sa["opt/lr"], sa["sched/phase"]):
        want = m.wsd_lr(PEAK, int(step), float(step - 1), 8.0, 2, 0.5)
        assert lr == pytest.approx(want[0]) and phase == want[1], step
    ha, hb = summary(a)["history"], summary(b)["history"]
    assert [r["step"] for r in ha] == [r["step"] for r in hb] == [0, 2, 4, 6, 8]
    for ra, rb in zip(ha, hb):
        assert (ra["heldout_kl"], ra["probe_kl"]) == (rb["heldout_kl"], rb["probe_kl"])
    assert not events(a, "early_stop") and summary(a)["stopped_early"] is None
    assert not [t for t in pd.read_parquet(a / "metrics" / "scalars.parquet")["tag"].unique()
                if t.startswith("early_stop/")]
    # every in-loop eval is checked: on the step clock that includes the one at max_steps (the final eval follows it)
    assert sorted(scalar(b, "early_stop/value")) == [2, 4, 6, 8] and summary(b)["stopped_early"] is None

    # a state from before early stopping: strip it from the step-4 full state of the disabled run, then resume
    full = a / "checkpoints" / "full_step_4"
    state = torch.load(full / "trainer.pt", map_location="cpu", weights_only=True)
    del state["cfg"]["early_stop"], state["st"]["early_stop"]
    torch.save(state, full / "trainer.pt")
    assert m.main(["--config", off, "--resume", str(full)]) == 0
    s = summary(a)
    assert s["status"] == "complete" and s["steps"] == 8 and s["resumes"] == 1 and s["stopped_early"] is None
    assert s["config"]["early_stop"] == m.DEFAULTS["early_stop"]
    np.testing.assert_allclose(steps_of(a)["loss/total"].to_numpy(), sb["loss/total"].to_numpy(), rtol=1e-6)


def test_end_save_refreshes_the_trainer_state_of_an_existing_same_step_full(tmp_path, monkeypatch):
    """A STOP/early stop picked up right after a periodic full state at the same step must still leave a state whose
    trainer.json/pt says 'stop', so a resume (with the STOP file removed) goes straight to the end phase."""
    m = load_script("04_distill")
    d = tmp_path / "ckpt" / "full_step_3"
    d.mkdir(parents=True)
    for f in ("model.pt", "optimizer.pt", "l2sp.pt"):
        (d / f).write_bytes(b"x")
    (d / "trainer.json").write_text('{"reason": "periodic"}', encoding="utf-8")
    events = []
    st = {"fulls": [3], "early_stop": {"stop": True, "triggered": {"reason": "stop_file"}}}
    R = SimpleNamespace(ckpt_dir=tmp_path / "ckpt", st=st, clock=lambda: 12.0, run_dir=tmp_path / "run", cfg={},
                        planner=SimpleNamespace(state_dict=lambda: {}), log=SimpleNamespace(
                            state_dict=lambda: {}, event=lambda kind, **kw: events.append((kind, kw))),
                        uploader=None)
    monkeypatch.setattr(m, "rotate_full", lambda R: None)
    m.save_full(R, 3, "end")
    brief = json.loads((d / "trainer.json").read_text(encoding="utf-8"))
    assert brief["reason"] == "end" and (d / "model.pt").read_bytes() == b"x"  # weights untouched
    assert torch.load(d / "trainer.pt", weights_only=False)["st"]["early_stop"]["stop"] is True
    assert events and events[-1][1].get("trainer_only") is True
