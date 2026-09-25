"""The eval cadence of scripts/04_distill.py: mini evals, a full eval at every epoch end, the headline numbers, the gate.

- eval.mini: a small eval of its own every mini.every_steps optimizer steps - never at step 0, never at a step with a
  full eval in the loop, skipped when the loop is due to end at its step (the final eval follows there; on the wall
  clock judged from the last mini's run time) - on fixed seeded subsets (mini.val_per_set per gate set,
  mini.train_utts of the probe) that stay the same across the run and a resume; teacher-forced and greedy; logged
  apart (eval/mini/..., evals/step_<N>_mini/, `eval_mini` events, summary.json's mini_history) and never read by
  the early stop or the verdict; training itself is unchanged by it
- eval.full_every_epochs: an eval at every N-th epoch end on the steps and wall clocks (the planner's epochs), decoding
  the COMPLETE eval sets; every_min / every_steps are ignored then. When the loop ends at the step of one (max_steps
  at an epoch end, an eval that runs past T), the end phase reuses it as the final eval: one decode per step
- the headline numbers after every eval (kitsune.evaluate.headline): pooled corpus CER = sum edits / sum chars of the
  per-set numbers, gate sets only; summary/{full,mini}/<name> with <name>_pct copies; one console line per eval
- the `eval` event's per-set CERs are those of the headline's scope (greedy_scope): the complete sets at a complete
  eval, with the fixed subset's as *_subset next to them; the subset at a subset eval
- eval.gate false: summary.json's verdict is "N/A" with the numbers
- the combined loss (w_kl * KL + w_ce * CE): combined_loss/train at every step (= loss/objective), combined_loss/val at
  step 0 and at every mini eval (the mini val subset's gate sets), combined_loss/val_full at every complete-set eval;
  first in 2_loss_accuracy and overlaid in one Custom Scalars chart in every event file

CPU only (the laptop GPU runs other jobs), a tiny random student and a synthetic corpus in the real on-disk formats."""
import copy
import json
import math
import os
import re
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

from fixtures import load_script, make_fake_corpus, make_fake_selection, tb_layouts  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
PEAK = 3e-3
MAX_STEPS = 12
NEVER = dict(enabled=True, metric="heldout_kl", patience=1000, min_delta_rel=0.0, min_delta_abs=0.0, min_evals=0,
             floor=None, action="stop")  # checked after every in-loop eval, never fires


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
    """tag -> {step: value}; a step logged twice (replayed after a resume) keeps the later value."""
    sc = pd.read_parquet(run / "metrics" / "scalars.parquet")
    sc = sc[sc["tag"] == tag].sort_values("wall", kind="stable")
    return dict(zip(sc["step"].astype(int), sc["value"]))


def epoch_ends(run: Path) -> list[int]:
    """Steps that complete an epoch of the planner's plan: data/epoch_progress = e + (s + 1) / steps is whole there."""
    st = steps_of(run)
    return [int(s) for s, p in zip(st["step"], st["data/epoch_progress"]) if float(p).is_integer()]


def ids_of(d: Path, pattern: str) -> list[str]:
    return sorted(i for p in d.glob(pattern) for i in pd.read_parquet(p)["id"])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + a tiny student saved exactly as 03 saves one, and a base config for it: ~8 s of
    train audio (the whole of it the probe) in steps of ~3 s, so an epoch is a few steps; the complete eval sets are
    6 utterances each (greedy subset 2)."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("evalcadence")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (36, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=11)
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
        "subset": {"train_audio_s": 8},
        "schedule": {"warmup_steps": 2, "cooldown_frac": 0.3, "clock": "steps", "max_steps": MAX_STEPS},
        "batch": {"step_audio_s": 3, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_steps": 1, "full_every_epochs": 1, "greedy_subset": 2, "batch_s": 20, "check_baselines": False,
                 "probe_is_train": True, "probe_greedy_audio_s": 3,
                 "mini": {"every_steps": 2, "val_per_set": 2, "train_utts": 3, "greedy": True}},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 4, "keep_local": 5, "full_after_smoke": False},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
        "early_stop": NEVER,
    }
    return dict(root=root, base=base, fc=fc)


def write_config(env, name: str, over: dict) -> str:
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------------------------------------- unit level


def _greedy_set(edits, chars, t_edits, t_chars, n=3):
    return dict(ref_edits=edits, ref_chars=chars, cer_ref_corpus=edits / chars, cer_teacher_edits=t_edits,
                cer_teacher_chars=t_chars, cer_teacher_corpus=t_edits / t_chars, n=n)


def test_headline_pools_the_gate_sets():
    """val CER = sum edits / sum chars over the gate sets (a monitor-only set such as galgame is left out; the mean of
    the per-set CERs would be something else), val KL / top-1 token-weighted, train numbers from the train side, and
    what was not evaluated is left out."""
    from kitsune import evaluate as ev

    greedy = dict(sets=dict(eval_jsut=_greedy_set(10, 100, 4, 90), eval_cv8=_greedy_set(1, 50, 2, 40),
                            eval_reazon=_greedy_set(30, 200, 9, 180), galgame=_greedy_set(99, 100, 99, 100)))
    tf = dict(sets=dict(eval_jsut=dict(kl=1.0, top1=0.5, n_tok=10), eval_cv8=dict(kl=2.0, top1=0.25, n_tok=30),
                        eval_reazon=dict(kl=4.0, top1=1.0, n_tok=60), galgame=dict(kl=100.0, top1=0.0, n_tok=1000)),
              all=dict(kl=50.0, top1=0.1))
    probe = dict(all=dict(kl=0.25, top1=0.75), sets={})
    pg = dict(sets=dict(src_a=_greedy_set(3, 30, 1, 20), src_b=_greedy_set(1, 10, 0, 10)))
    h = ev.headline(tf=tf, greedy=greedy, probe=probe, probe_greedy=pg)
    assert h["val_cer"] == pytest.approx(41 / 350) and h["val_cer"] != pytest.approx(np.mean([0.1, 0.02, 0.15]))
    assert h["val_cer_vs_teacher"] == pytest.approx(15 / 310)
    assert h["val_loss"] == pytest.approx((1 * 10 + 2 * 30 + 4 * 60) / 100)
    assert h["val_top1"] == pytest.approx((0.5 * 10 + 0.25 * 30 + 1.0 * 60) / 100)
    assert h["train_cer"] == pytest.approx(4 / 40) and h["train_cer_vs_teacher"] == pytest.approx(1 / 30)
    assert (h["train_loss"], h["train_top1"]) == (0.25, 0.75)
    assert ev.headline_val_utts(greedy) == 9  # the utterances val_cer pools: the gate sets' (galgame left out)
    # no gate set evaluated (a config without them): every evaluated set pools
    only = ev.headline(greedy=dict(sets=dict(galgame=_greedy_set(5, 50, 1, 10))))
    assert only == dict(val_cer=0.1, val_cer_vs_teacher=0.1)
    assert ev.headline_val_utts(dict(sets=dict(galgame=_greedy_set(5, 50, 1, 10)))) == 3
    assert ev.headline_val_utts(None) == 0
    assert ev.headline() == {} and ev.headline(greedy=dict(sets={}), probe=dict(sets={})) == {}


def test_overfit_gap_scalars_pool_the_gate_sets(monkeypatch):
    """eval/kl_gap_heldout_minus_probe is the verdict's gap (heldout_kl - probe_kl, gate sets only), not the all-sets
    KL minus the probe's: the monitor-only sets (eval_emilia, galgame) come from training sources. The CER gap pools
    the gate sets of the fixed greedy subset likewise. run_eval itself, the evals replaced by fixed summaries."""
    from kitsune import evaluate as ev

    m = load_script("04_distill")
    gate, monitor = EVAL, ["eval_emilia", "galgame"]
    tf = dict(sets={**{s: dict(kl=0.5, ce=1.0, top1=0.9, n_tok=100) for s in gate},
                    **{s: dict(kl=2.0, ce=1.0, top1=0.5, n_tok=100) for s in monitor}},
              all=dict(kl=(0.5 * 300 + 2.0 * 200) / 500, ce=1.0, top1=0.74, n_tok=500), wall_s=1.0)
    probe = dict(sets={"src_a": dict(kl=0.3, ce=1.0, top1=0.95, n_tok=100)},
                 all=dict(kl=0.3, ce=1.0, top1=0.95, n_tok=100), wall_s=1.0)
    def gset(edits, chars, n=3):
        return dict(_greedy_set(edits, chars, edits, chars, n=n), teacher_cer_ref_corpus=0.1, ratio_vs_teacher=1.0,
                    trunc_rate=0.0)

    gr = dict(sets={**{s: gset(1, 10) for s in gate}, **{s: gset(5, 10) for s in monitor}}, all=gset(13, 50, n=15),
              n_utts=15, wall_s=1.0)
    pg = dict(sets={"src_a": _greedy_set(1, 20, 1, 20)}, all=_greedy_set(1, 20, 1, 20), n_utts=3, wall_s=1.0)
    no_rows = pd.DataFrame({"source": []})

    def tf_eval(model, store, feat, device, bs, ids=None, amp=None):
        return (probe, no_rows) if ids is not None else (tf, no_rows)

    def greedy_eval(model, store, ids, feat, device, bs, tokenizer=None, amp=None):
        return (pg, no_rows) if store == "train" else (gr, no_rows)

    monkeypatch.setattr(ev, "teacher_forced_eval", tf_eval)
    monkeypatch.setattr(ev, "greedy_eval", greedy_eval)
    monkeypatch.setattr(ev, "pick_samples", lambda *a, **k: [])
    scal = {}
    log = SimpleNamespace(event=lambda *a, **k: None, table=lambda *a, **k: None, eval_json=lambda *a, **k: None,
                          samples=lambda *a, **k: None, scalars=lambda row, step: scal.update(row))
    R = SimpleNamespace(cfg=m.load_config(None, []), log=log, model=torch.nn.Linear(1, 1), evalstore="eval",
                        train="train", feat_eval=None, device="cpu", amp=None, tokenizer=None, probe_ids=["p"],
                        probe_greedy_ids=["q"], greedy_ids=["g"], st=dict(history=[], epoch_progress=0.0),
                        clock=lambda: 0.0)
    m.run_eval(R, 0)
    rec = R.st["history"][-1]
    assert rec["heldout_kl"] == pytest.approx(0.5) and rec["heldout_kl_all_sets"] == pytest.approx(1.1)
    assert scal["eval/kl_gap_heldout_minus_probe"] == pytest.approx(rec["heldout_kl"] - rec["probe_kl"])
    assert scal["eval/kl_gap_heldout_minus_probe"] == pytest.approx(0.2)  # the all-sets gap would be 0.8
    assert scal["eval/kl_gap_heldout_minus_probe"] == pytest.approx(scal["summary/full/val_loss"]
                                                                    - scal["summary/full/train_loss"])
    assert scal["eval/cer_teacher_gap_heldout_minus_probe"] == pytest.approx(0.1 - 0.05)  # all sets: 0.26 - 0.05
    assert scal["eval/tf/all/kl"] == pytest.approx(1.1)  # the all-sets numbers stay logged


def test_eval_event_per_set_numbers_follow_the_headline_scope(monkeypatch):
    """Bug B-1: a complete eval's `eval` event gives each set's cer / ratio / trunc of the complete set (summary.json's
    greedy_full, what the verdict judges), no longer the fixed subset's, and keeps the subset's next to them as
    cer_subset / ratio_subset / trunc_subset (summary.json's greedy); greedy_scope names the scope, as headline_scope
    does. A subset eval's event is as before (the subset's numbers under the plain names, no *_subset), with its
    greedy_scope "subset". The history (the verdict's trend, the early stop) and the scalars keep the subset. run_eval
    itself on fixed per-utterance rows, the subset decoded better than the rest of each set."""
    from kitsune import evaluate as ev

    m = load_script("04_distill")
    ref, teach = "あいうえおかきくけこ", "あいうえおかきくけさ"  # the teacher: 1 error in 10 characters
    rows = []
    for k, x in enumerate(EVAL):
        for i in range(4):  # <set>_0 and _1: the fixed greedy subset, decoded as well as the teacher
            hyp = teach if i < 2 else "ん" * (k + 2) + ref[k + 2:]  # the rest: k + 2 errors, the last one truncated
            rows.append(dict(id=f"{x}_{i}", source=x, duration=1.0, n_tok=11, ref=ref, teacher_hyp=teach, hyp=hyp,
                             cer_ref=(1 if i < 2 else k + 2) / 10, cer_teacher=0.0 if i < 2 else (k + 3) / 10,
                             truncated=i == 3, has_teacher=True, teacher_cer=0.1, teacher_truncated=False))
    df = pd.DataFrame(rows)
    greedy_ids = [f"{x}_{i}" for x in EVAL for i in range(2)]
    tf = dict(sets={x: dict(kl=0.5, ce=1.0, top1=0.9, n_tok=100) for x in EVAL},
              all=dict(kl=0.5, ce=1.0, top1=0.9, n_tok=300), wall_s=1.0)

    def greedy_eval(model, store, ids, feat, device, bs, tokenizer=None, amp=None):
        g = (df if ids is None else df[df["id"].isin(ids)]).reset_index(drop=True)
        return ev.summarise_greedy(g, wall_s=1.0), g

    monkeypatch.setattr(ev, "teacher_forced_eval", lambda *a, **k: (tf, pd.DataFrame({"source": []})))
    monkeypatch.setattr(ev, "greedy_eval", greedy_eval)
    monkeypatch.setattr(ev, "pick_samples", lambda *a, **k: [])
    got, js, scal = [], {}, {}
    log = SimpleNamespace(event=lambda kind, **k: got.append(dict(k, kind=kind)), table=lambda *a, **k: None,
                          eval_json=lambda name, obj, step: js.update({step: obj}), samples=lambda *a, **k: None,
                          scalars=lambda row, step: scal.setdefault(step, {}).update(row))
    R = SimpleNamespace(cfg=m.load_config(None, []), log=log, model=torch.nn.Linear(1, 1), evalstore="eval",
                        train="train", feat_eval=None, device="cpu", amp=None, tokenizer=None, probe_ids=[],
                        probe_greedy_ids=[], greedy_ids=greedy_ids, st=dict(history=[], epoch_progress=0.0),
                        clock=lambda: 0.0)
    m.run_eval(R, 0)  # the step-0 eval: the fixed subset
    m.run_eval(R, 5, complete=True)  # an epoch-end eval: the complete sets
    sub_ev, full_ev = [e for e in got if e["kind"] == "eval"]
    assert (sub_ev["at_step"], full_ev["at_step"]) == (0, 5) and not sub_ev["complete"] and full_ev["complete"]
    assert sub_ev["greedy_scope"] == js[0]["headline_scope"]["val_greedy"] == "subset"
    assert full_ev["greedy_scope"] == js[5]["headline_scope"]["val_greedy"] == "complete"
    assert list(full_ev).index("greedy_scope") < list(full_ev).index("sets")  # ahead of them in the console line

    def brief(g: dict, suffix: str = "") -> dict:
        return {f"cer{suffix}": round(g["cer_ref_corpus"], 4), f"ratio{suffix}": round(g["ratio_vs_teacher"], 3),
                f"trunc{suffix}": round(g["trunc_rate"], 4)}

    for k, x in enumerate(EVAL):
        sub, full = js[5]["greedy"]["sets"][x], js[5]["greedy_full"]["sets"][x]
        assert (sub["n"], full["n"]) == (2, 4) and js[0]["greedy"]["sets"][x] == sub  # the same subset both times
        assert sub["cer_ref_corpus"] == pytest.approx(0.1) and sub["ratio_vs_teacher"] == pytest.approx(1.0)
        assert full["cer_ref_corpus"] == pytest.approx((2 + 2 * (k + 2)) / 40) and full["trunc_rate"] == 0.25
        assert full_ev["sets"][x] == dict(kl=0.5, top1=0.9, **brief(full), **brief(sub, "_subset"))
        assert (full_ev["sets"][x]["cer"], full_ev["sets"][x]["ratio"], full_ev["sets"][x]["trunc"]) == (
            round((2 + 2 * (k + 2)) / 40, 4), round((2 + 2 * (k + 2)) / 40 / 0.1, 3), 0.25)
        assert sub_ev["sets"][x] == dict(kl=0.5, top1=0.9, **brief(sub))  # unchanged: the subset, no *_subset
        # what stays on the subset: the history, and the scalars under their tags (the complete sets' own next to them)
        assert R.st["history"][-1]["greedy"][x]["cer_ref_corpus"] == sub["cer_ref_corpus"]
        assert scal[5][f"eval/greedy/{x}/cer_ref_corpus"] == sub["cer_ref_corpus"] == scal[0][
            f"eval/greedy/{x}/cer_ref_corpus"]
        assert scal[5][f"eval/greedy_full/{x}/cer_ref_corpus"] == full["cer_ref_corpus"]
    # the verdict judges the complete sets: what the complete eval's event now says
    v = ev.verdict(dict(final=js[5]["greedy_full"], history=R.st["history"]))
    assert all(round(v["sets"][x]["student"], 4) == full_ev["sets"][x]["cer"] for x in EVAL)


def test_config_keys_are_validated():
    m = load_script("04_distill")
    ok = m.load_config(None, ["eval.full_every_epochs=2", "eval.mini.every_steps=200", "eval.mini.val_per_set=0",
                              "eval.gate=false"])
    assert ok["eval"]["full_every_epochs"] == 2 and ok["eval"]["mini"]["every_steps"] == 200
    assert m.epoch_cadence(ok) == 2 and m.epoch_mode(ok)
    assert m.epoch_cadence(m.load_config(None, [])) is None and not m.epoch_mode(m.load_config(None, []))
    # the defaults keep today's behaviour: no minis, no complete-set evals in the loop, the gate on
    d = m.DEFAULTS["eval"]
    assert d["mini"]["every_steps"] is None and d["full_every_epochs"] is None and d["gate"] is True
    for bad in (["eval.full_every_epochs=0"], ["eval.full_every_epochs=1.5"], ["eval.every_epochs=1",
                                                                               "eval.full_every_epochs=1"],
                ["eval.mini.every_steps=0"], ["eval.mini.every_steps=true"], ["eval.mini.val_per_set=-1"],
                ["eval.mini.train_utts=2.5"], ["eval.mini.greedy=1"], ["eval.gate=no"], ["eval.mini.every=2"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)


def test_mini_due_and_the_console_line():
    m = load_script("04_distill")
    R = SimpleNamespace(cfg=m.load_config(None, ["eval.mini.every_steps=200"]), mini_val_ids=["a"], mini_train_ids=[])
    assert [s for s in range(0, 1001) if m.mini_due(R, s)] == [200, 400, 600, 800, 1000]  # never at step 0
    R.mini_val_ids = []
    assert not m.mini_due(R, 200)  # nothing to evaluate
    R.cfg["eval"]["mini"]["every_steps"] = None
    R.mini_val_ids = ["a"]
    assert not m.mini_due(R, 200)
    head = dict(val_cer=0.0831, val_cer_vs_teacher=0.0254, train_cer=0.1, train_cer_vs_teacher=0.0123,
                val_loss=0.51234, train_loss=0.2)
    assert m.headline_line("full", 1126, 1.0, head) == (
        "[full eval] step 1126 epoch 1.00 | val CER 8.3% (vs teacher 2.5%) | train CER vs teacher 1.2% "
        "(vs ref 10.0%) | val KL 0.512 train KL 0.200")
    # a full eval says what its val CER decoded: the step-0 subset and the complete sets later share the curve
    assert m.headline_line("full", 0, 0.0, head, dict(val_greedy="subset", val_cer_utts=1500)) == (
        "[full eval] step 0 epoch 0.00 | val CER 8.3% (vs teacher 2.5%) on 1500 utts (subset) | train CER vs teacher "
        "1.2% (vs ref 10.0%) | val KL 0.512 train KL 0.200")
    assert m.headline_line("mini", 200, 0.18, {}) == (
        "[mini eval] step 200 epoch 0.18 | val CER n/a (vs teacher n/a) | train CER vs teacher n/a (vs ref n/a) | "
        "val KL n/a train KL n/a")


def test_gate_off_reports_na_with_the_numbers():
    m = load_script("04_distill")
    computed = dict(verdict="GO", reasons=["3/3 sets within 1.2x"], sets={"eval_jsut": {"ratio": 1.0}}, n_sets_go=3,
                    trunc_rate=0.0, trend={"overfit": False}, thresholds={"go_ratio": 1.2})
    on = m.load_config(None, [])
    assert m.gate_verdict(on, computed) is computed
    off = m.gate_verdict(m.load_config(None, ["eval.gate=false"]), computed)
    assert off == dict(verdict="N/A", reason="gate disabled (sanity/overfit run)",
                       numbers=dict(sets={"eval_jsut": {"ratio": 1.0}}, n_sets_go=3, trunc_rate=0.0,
                                    trend={"overfit": False}, thresholds={"go_ratio": 1.2}))
    for name in ("overfit_1s", "overfit_10s", "overfit_1h"):  # the sanity runs have no gate; the real run has one
        assert m.load_config(str(ROOT / "configs" / f"{name}.json"), [])["eval"]["gate"] is False
    assert m.load_config(str(ROOT / "configs" / "viability.json"), [])["eval"]["gate"] is True


# ------------------------------------------------------------------------------------------------ whole runs


@pytest.fixture(scope="module")
def steps_runs(env):
    """The step clock, an eval at every epoch end (every_steps 1 set too: ignored), a mini eval every 2 steps, an
    early-stop rule that is checked but never fires. `mini` crashes before step 10 and resumes from its step-8 full
    state; `plain` is the same run without mini evals."""
    m = load_script("04_distill")
    mini = write_config(env, "cad-mini", {})
    plain = write_config(env, "cad-plain", {"eval": {"mini": {"every_steps": None}}})
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KITSUNE_CRASH_AT_STEP", "10")
        with pytest.raises(RuntimeError, match="simulated crash"):
            m.main(["--config", mini])
        mp.delenv("KITSUNE_CRASH_AT_STEP")
        run = one_run(env["root"], "cad-mini")
        first = [e for e in events(run, "subset") if e["split"].startswith("mini_")]
        rc = m.main(["--config", mini, "--resume", str(run / "checkpoints" / "full_step_8")])
    assert rc == 0 and m.main(["--config", plain]) == 0
    return dict(mini=run, plain=one_run(env["root"], "cad-plain"), first_subsets=first)


def test_full_eval_at_every_epoch_end_on_the_step_clock(steps_runs):
    """In the loop the evals come exactly at the planner's epoch ends (every_steps 1 is ignored), each one decoding the
    complete eval sets; the history the verdict and the early stop read is step 0, those, and the final eval."""
    run = steps_runs["mini"]
    ends = epoch_ends(run)
    assert len([e for e in ends if e < MAX_STEPS]) >= 2, ends  # several epochs end inside the run
    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == MAX_STEPS and s["resumes"] == 1
    assert [r["step"] for r in s["history"]] == sorted({0, *ends, MAX_STEPS})
    assert all(r["epoch"] == float(i) for i, r in enumerate(s["history"][:len(ends) + 1]))
    for step in ends:
        d = json.loads((run / "evals" / f"step_{step}" / "summary.json").read_text(encoding="utf-8"))
        assert d["complete"] and d["greedy_full"]["n_utts"] == 3 * 6 and d["greedy"]["n_utts"] == 3 * 2
        assert d["headline_scope"]["val_greedy"] == "complete" and d["headline_scope"]["val_greedy_utts"] == 18
        assert d["headline_scope"]["val_cer_utts"] == 18
        assert len(pd.read_parquet(run / "evals" / f"step_{step}" / "greedy_eval_jsut.parquet")) == 6
    s0 = json.loads((run / "evals" / "step_0" / "summary.json").read_text(encoding="utf-8"))
    assert not s0["complete"] and s0["greedy_full"] is None and s0["headline_scope"]["val_greedy"] == "subset"
    assert s0["headline_scope"]["val_cer_utts"] == 3 * 2  # the greedy subset: 2 per gate set
    # the early stop reads the in-loop full evals only (never the minis, never step 0 or the final eval)
    assert sorted(scalar(run, "early_stop/value")) == ends
    ev_scal = scalar(run, "eval/epoch")
    assert sorted(ev_scal) == sorted({0, *ends, MAX_STEPS}) and all(ev_scal[e] == float(i + 1) for i, e in
                                                                     enumerate(ends))
    # max_steps ends an epoch: the loop's complete eval at that step is the final eval, decoded once (the history
    # would hide a second decode: it replaces a same-step record)
    plain = steps_runs["plain"]
    assert MAX_STEPS in epoch_ends(plain), epoch_ends(plain)
    at = [e["at_step"] for e in events(plain, "eval")]
    assert at == [e["at_step"] for e in events(plain, "eval_start")] == sorted({0, *epoch_ends(plain)})
    assert not any(e["final"] for e in events(plain, "eval"))
    assert [e["at_step"] for e in events(plain, "eval_final_reused")] == [MAX_STEPS]
    # the crashed-and-resumed run: once per step after the resume too (its step-9 eval is replayed, legitimately)
    after = events(run)
    after = after[max(i for i, e in enumerate(after) if e["kind"] == "resumed"):]
    at = [e["at_step"] for e in after if e["kind"] == "eval"]
    assert len(at) == len(set(at)) and at[-1] == MAX_STEPS
    assert [e["at_step"] for e in after if e["kind"] == "eval_final_reused"] == [MAX_STEPS]


def test_mini_cadence_subsets_and_isolation(env, steps_runs):
    """Mini evals every 2 steps except at step 0 and at the full-eval steps (epoch ends, the last step); the same
    subsets at every one of them and in both launches, scored against the reference and teacher text the corpus
    holds; logged apart; the early stop, the verdict's history and the training itself are exactly those of the run
    without minis."""
    run, plain = steps_runs["mini"], steps_runs["plain"]
    ends = epoch_ends(run)
    s = summary(run)
    want = [k for k in range(2, MAX_STEPS + 1, 2) if k not in ends and k != MAX_STEPS]
    skipped = [k for k in range(2, MAX_STEPS + 1, 2) if k not in want]
    assert want and [k for k in skipped if k != MAX_STEPS], (ends, "the data must end an epoch at an even step")
    assert [r["step"] for r in s["mini_history"]] == want
    assert sorted({e["at_step"] for e in events(run, "eval_mini")}) == want
    assert sorted(int(p.name.split("_")[1]) for p in (run / "evals").glob("step_*_mini")) == want
    assert not set(want) & {r["step"] for r in s["history"]}

    # the subsets: sizes, content, and the same in the resumed launch
    sub = [e for e in events(run, "subset") if e["split"].startswith("mini_")]
    assert len(sub) == 4 and sub[:2] == steps_runs["first_subsets"]
    assert [{k: v for k, v in e.items() if k not in ("wall", "time", "elapsed_s", "step")} for e in sub[2:]] == \
        [{k: v for k, v in e.items() if k not in ("wall", "time", "elapsed_s", "step")} for e in sub[:2]]
    val, train = sub[0], sub[1]
    assert val["split"] == "mini_val" and val["n"] == 6 and val["per_source"] == {s_: 2 for s_ in EVAL}
    train_ids = next(e for e in events(run, "subset") if e["split"] == "train")["ids"]
    assert train["split"] == "mini_train" and train["n"] == 3 and set(train["ids"]) <= set(train_ids)
    for step in want:
        d = run / "evals" / f"step_{step}_mini"
        assert ids_of(d, "tf_*.parquet") == ids_of(d, "greedy_*.parquet") == sorted(val["ids"])
        assert ids_of(d, "probe.parquet") == ids_of(d, "probe_greedy.parquet") == sorted(train["ids"])
        js = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        assert js["mini"] and js["n_val"] == 6 and js["n_train"] == 3 and js["wall_s"] >= 0
        gr = pd.concat([pd.read_parquet(p) for p in [*d.glob("greedy_*.parquet"), d / "probe_greedy.parquet"]])
        truth = env["fc"].utts  # the text the fixture wrote: what the once-read teacher rows must hold
        assert gr["has_teacher"].all() and gr["ref"].tolist() == [truth[i].text for i in gr["id"]]
        assert gr["teacher_hyp"].tolist() == [truth[i].hyp for i in gr["id"]]
        assert gr["teacher_truncated"].tolist() == [truth[i].truncated for i in gr["id"]]
    wall = scalar(run, "eval/mini/wall_s")
    assert sorted(wall) == want and all(v >= 0 for v in wall.values())

    # isolation: same steps, losses, evals, early-stop values as without minis
    sa, sb = steps_of(run), steps_of(plain)
    assert sa["step"].tolist() == sb["step"].tolist() == list(range(1, MAX_STEPS + 1))
    for col in ("loss/total", "loss/kl", "loss/ce", "opt/lr", "opt/grad_norm", "data/epoch_progress"):
        np.testing.assert_allclose(sa[col].to_numpy(), sb[col].to_numpy(), rtol=1e-6, atol=1e-9, err_msg=col)
    pa_, pb = summary(plain), s
    assert [r["step"] for r in pa_["history"]] == [r["step"] for r in pb["history"]] and not pa_["mini_history"]
    for ra, rb in zip(pa_["history"], pb["history"]):
        assert ra["heldout_kl"] == pytest.approx(rb["heldout_kl"], rel=1e-5)
    assert scalar(plain, "early_stop/value").keys() == scalar(run, "early_stop/value").keys()
    assert not events(plain, "eval_mini") and not list((plain / "evals").glob("step_*_mini"))
    assert s["stopped_early"] is None and s["verdict"]["verdict"] in ("GO", "PROMISING", "NO-GO", "INCONCLUSIVE")


def test_headline_numbers_are_the_pooled_per_set_numbers(steps_runs):
    """After every eval, full and mini: summary/<kind>/val_cer = sum ref_edits / sum ref_chars of the per-set greedy
    numbers (the complete sets at an epoch end, the subset at step 0), the vs-teacher CER likewise, val_loss = the
    history's heldout_kl, train CER from the train-side decode; the _pct copies are 100x; one console line each."""
    run = steps_runs["mini"]
    s = summary(run)
    for rec in s["history"]:
        step = rec["step"]
        d = json.loads((run / "evals" / f"step_{step}" / "summary.json").read_text(encoding="utf-8"))
        g = (d["greedy_full"] or d["greedy"])["sets"]
        head = d["headline"]
        assert head == rec["headline"]
        assert head["val_cer"] == pytest.approx(sum(g[x]["ref_edits"] for x in EVAL) / sum(g[x]["ref_chars"]
                                                                                           for x in EVAL))
        assert head["val_cer_vs_teacher"] == pytest.approx(sum(g[x]["cer_teacher_edits"] for x in EVAL)
                                                           / sum(g[x]["cer_teacher_chars"] for x in EVAL))
        for x in EVAL:  # the new per-set numerator / denominator are the per-set vs-teacher CER's own
            assert g[x]["cer_teacher_corpus"] == pytest.approx(g[x]["cer_teacher_edits"] / g[x]["cer_teacher_chars"])
        assert head["val_loss"] == pytest.approx(rec["heldout_kl"]) and head["train_loss"] == rec["probe_kl"]
        pg = d["probe_greedy"]["all"]
        assert head["train_cer"] == pytest.approx(pg["ref_edits"] / pg["ref_chars"])
        assert head["train_cer_vs_teacher"] == pytest.approx(pg["cer_teacher_corpus"])
        for k in ("val_cer", "val_cer_vs_teacher", "train_cer", "train_cer_vs_teacher", "val_loss", "val_top1",
                  "train_loss", "train_top1"):
            assert scalar(run, f"summary/full/{k}")[step] == pytest.approx(head[k], rel=1e-6), (step, k)
        for k in ("val_cer", "val_cer_vs_teacher", "train_cer", "train_cer_vs_teacher"):
            assert scalar(run, f"summary/full/{k}_pct")[step] == pytest.approx(100 * head[k], rel=1e-6)
        # the scope of the headline val CER, next to it: the step-0 subset, then the complete sets
        n_val = sum(g[x]["n"] for x in EVAL)
        assert scalar(run, "summary/full/val_cer_utts")[step] == d["headline_scope"]["val_cer_utts"] == n_val
    for mrec in s["mini_history"]:
        step = mrec["step"]
        d = json.loads((run / "evals" / f"step_{step}_mini" / "summary.json").read_text(encoding="utf-8"))
        g, head = d["greedy"]["sets"], d["headline"]
        assert set(g) == set(EVAL) and {k: mrec[k] for k in head} == head
        assert head["val_cer"] == pytest.approx(sum(g[x]["ref_edits"] for x in EVAL) / sum(g[x]["ref_chars"]
                                                                                           for x in EVAL))
        tf = d["tf"]["sets"]
        assert head["val_loss"] == pytest.approx(sum(tf[x]["kl"] * tf[x]["n_tok"] for x in EVAL)
                                                 / sum(tf[x]["n_tok"] for x in EVAL))
        assert head["train_loss"] == pytest.approx(d["probe"]["all"]["kl"])
        assert head["train_cer_vs_teacher"] == pytest.approx(d["probe_greedy"]["all"]["cer_teacher_corpus"])
        assert scalar(run, "summary/mini/val_cer_pct")[step] == pytest.approx(100 * head["val_cer"], rel=1e-6)
        assert scalar(run, "summary/mini/train_cer_vs_teacher_pct")[step] == pytest.approx(
            100 * head["train_cer_vs_teacher"], rel=1e-6)
    assert summary(run)["headline"] == s["history"][-1]["headline"]
    out = (run / "logs" / "stdout.log").read_text(encoding="utf-8")
    assert "[full eval] step 0 epoch 0.00 | val CER " in out and " train KL " in out
    assert "on 6 utts (subset) | train CER" in out and "on 18 utts (complete) | train CER" in out
    for mrec in s["mini_history"]:
        assert f"[mini eval] step {mrec['step']} epoch " in out


def test_eval_events_match_the_summaries(steps_runs):
    """Every `eval` event against its step's summary.json (bug B-1): greedy_scope is headline_scope.val_greedy, and
    the per-set cer / ratio / trunc are that scope's (greedy_full at the complete evals - the epoch ends and the final
    eval, whose CERs are the verdict's - greedy at the step-0 subset eval), a complete eval's subset numbers next to
    them as *_subset. The console's [event] line names the scope ahead of the per-set numbers (it is cut at 300
    characters)."""
    def rnd(v, nd):
        return None if v is None or not math.isfinite(v) else round(v, nd)  # the event writes NaN as null

    def brief(g: dict, suffix: str = "") -> dict:
        return {f"cer{suffix}": rnd(g["cer_ref_corpus"], 4), f"ratio{suffix}": rnd(g["ratio_vs_teacher"], 3),
                f"trunc{suffix}": rnd(g["trunc_rate"], 4)}

    for run in (steps_runs["mini"], steps_runs["plain"]):
        last = {e["at_step"]: e for e in events(run, "eval")}  # a step replayed after the resume: its later event
        s = summary(run)
        assert sorted(last) == [r["step"] for r in s["history"]]
        for step, e in last.items():
            d = json.loads((run / "evals" / f"step_{step}" / "summary.json").read_text(encoding="utf-8"))
            assert e["greedy_scope"] == d["headline_scope"]["val_greedy"] == ("complete" if d["complete"] else "subset")
            assert e["complete"] == d["complete"] and (d["greedy_full"] is not None) == d["complete"]
            head = d["greedy_full"] if d["complete"] else d["greedy"]
            assert set(head["sets"]) == set(e["sets"]) == set(EVAL)
            for x, g in head["sets"].items():
                want = dict(brief(g), **(brief(d["greedy"]["sets"][x], "_subset") if d["complete"] else {}))
                assert {k: v for k, v in e["sets"][x].items() if k not in ("kl", "top1")} == want, (step, x)
        fin = last[MAX_STEPS]
        assert fin["complete"] and all(fin["sets"][x]["cer"] == round(s["verdict"]["sets"][x]["student"], 4)
                                       for x in EVAL)
        lines = [ln for ln in (run / "logs" / "stdout.log").read_text(encoding="utf-8").splitlines()
                 if ln.startswith("[event] eval {")]
        assert len(lines) >= len(last)
        assert all(re.search(r'"greedy_scope": "(complete|subset)"', ln.split('"sets"')[0]) for ln in lines)
        assert any('"greedy_scope": "subset"' in ln for ln in lines)
        assert any('"greedy_scope": "complete"' in ln for ln in lines)


def _pooled_objective(tf: dict, cfg: dict) -> float:
    """w_kl * KL + w_ce * CE over the gate sets of a summary.json's teacher-forced part, token-weighted."""
    n = sum(tf["sets"][x]["n_tok"] for x in EVAL)
    kl = sum(tf["sets"][x]["kl"] * tf["sets"][x]["n_tok"] for x in EVAL) / n
    ce = sum(tf["sets"][x]["ce"] * tf["sets"][x]["n_tok"] for x in EVAL) / n
    return cfg["loss"]["w_kl"] * kl + cfg["loss"]["w_ce"] * ce


def test_combined_loss_curves(steps_runs):
    """The overall loss chart's three series, each on one scope: combined_loss/train at every optimizer step (= the
    step's loss/objective, in steps.parquet too), combined_loss/val at step 0 and at every mini eval (the mini val
    subset's gate sets, pooled token-weighted; the step-0 point on the very same utterances, without a mini eval of its
    own), combined_loss/val_full at every complete-set eval (epoch ends, the final eval; never step 0). Each value is
    the eval's summary.json numbers recombined, and summary.json keeps the sums; a run without minis has no val curve.
    The run crashed and resumed: its replayed steps are read at their later values (scalar)."""
    run, plain = steps_runs["mini"], steps_runs["plain"]
    s = summary(run)
    cfg = s["config"]
    st = steps_of(run)
    assert (st["combined_loss/train"] == st["loss/objective"]).all()
    train = scalar(run, "combined_loss/train")
    assert sorted(train) == list(range(1, MAX_STEPS + 1))
    obj = scalar(run, "loss/objective")
    assert all(train[k] == obj[k] for k in train)

    minis = [r["step"] for r in s["mini_history"]]
    val = scalar(run, "combined_loss/val")
    assert sorted(val) == [0, *minis] and minis
    n_tok = set()
    for step in minis:
        d = json.loads((run / "evals" / f"step_{step}_mini" / "summary.json").read_text(encoding="utf-8"))
        c = d["combined_loss"]["val"]
        assert c["scope"] == "mini" and c["sets"] == sorted(EVAL) and (c["w_kl"], c["w_ce"]) == (1.0, 0.8)
        assert val[step] == pytest.approx(c["value"], rel=1e-9)
        assert c["value"] == pytest.approx(_pooled_objective(d["tf"], cfg), rel=1e-9)
        assert c["value"] == pytest.approx((c["kl_sum"] + 0.8 * c["ce_sum"]) / c["n_tok"])
        n_tok.add(c["n_tok"])
    s0 = json.loads((run / "evals" / "step_0" / "summary.json").read_text(encoding="utf-8"))
    c0 = s0["combined_loss"]["val"]
    assert c0["scope"] == "mini" and {c0["n_tok"]} == n_tok  # the mini val subset's tokens, at step 0 too
    assert val[0] == pytest.approx(c0["value"]) and "val_full" not in s0["combined_loss"]
    assert val[0] > min(val[k] for k in minis)  # the untrained student is worse on the same utterances
    assert not (run / "evals" / "step_0_mini").exists() and 0 not in minis

    ends = epoch_ends(run)
    full = scalar(run, "combined_loss/val_full")
    assert sorted(full) == sorted({*ends, MAX_STEPS}) and 0 not in full
    for step in full:
        d = json.loads((run / "evals" / f"step_{step}" / "summary.json").read_text(encoding="utf-8"))
        c = d["combined_loss"]["val_full"]
        assert d["complete"] and c["scope"] == "complete" and "val" not in d["combined_loss"]
        assert full[step] == pytest.approx(c["value"], rel=1e-9)
        assert c["value"] == pytest.approx(_pooled_objective(d["tf"], cfg), rel=1e-9)
        assert c["n_tok"] == sum(d["tf"]["sets"][x]["n_tok"] for x in EVAL)
    assert "combined_loss/val" not in set(pd.read_parquet(plain / "metrics" / "scalars.parquet")["tag"])
    assert sorted(scalar(plain, "combined_loss/val_full")) == sorted({*epoch_ends(plain), MAX_STEPS})
    assert "combined_loss" not in json.loads((plain / "evals" / "step_0" / "summary.json").read_text(encoding="utf-8"))
    assert [e["combined_loss"] for e in events(run, "eval_mini")][-1] == {"val": round(val[minis[-1]], 5)}


def test_tensorboard_places_the_new_tags(steps_runs):
    """summary/* first in 2_loss_accuracy (00_summary), eval/mini/* next to the full sections (<section>_mini), their
    cost in 1_operational/eval/mini; no tag of the run falls through to 3_misc unmatched."""
    run = steps_runs["mini"]
    assert not events(run, "tb_tag_unmapped")
    tag_map = json.loads((run / "metrics" / "tag_map.json").read_text(encoding="utf-8"))
    tags = set(pd.read_parquet(run / "metrics" / "scalars.parquet")["tag"])
    assert tags <= set(tag_map) and not any(e.get("unmapped") for e in tag_map.values())
    want = {
        "combined_loss/train": "2_loss_accuracy/00_combined/train",
        "combined_loss/val": "2_loss_accuracy/00_combined/val",
        "combined_loss/val_full": "2_loss_accuracy/00_combined/val_full",
        "summary/full/val_cer": "2_loss_accuracy/00_summary/full/val_cer",
        "summary/full/val_cer_pct": "2_loss_accuracy/00_summary/full/val_cer_pct",
        "summary/mini/train_cer_vs_teacher_pct": "2_loss_accuracy/00_summary/mini/train_cer_vs_teacher_pct",
        "summary/mini/val_loss": "2_loss_accuracy/00_summary/mini/val_loss",
        "eval/mini/tf/eval_jsut/kl": "2_loss_accuracy/val_loss_mini/eval_jsut/kl",
        "eval/mini/tf/all/top1": "2_loss_accuracy/val_accuracy_mini/all/top1",
        "eval/mini/greedy/eval_cv8/cer_ref_corpus": "2_loss_accuracy/val_accuracy_mini/eval_cv8/cer_ref_corpus",
        "eval/mini/probe/all/kl": "2_loss_accuracy/train_probe_loss_mini/all/kl",
        "eval/mini/probe_greedy/all/cer_teacher_corpus":
            "2_loss_accuracy/train_probe_accuracy_mini/all/cer_teacher_corpus",
        "eval/mini/greedy/all/ref_chars": "1_operational/eval/mini/greedy/all/ref_chars",
        "eval/mini/greedy/rtf": "1_operational/eval/mini/greedy/rtf",
        "eval/mini/wall_s": "1_operational/eval/mini/wall_s",
        "eval/mini/tf/eval_reazon/teacher_p1": "3_misc/eval/mini/tf/eval_reazon/teacher_p1",
        "eval/greedy_full/eval_jsut/cer_teacher_edits": "2_loss_accuracy/val_accuracy_full/eval_jsut/cer_teacher_edits",
        "eval/greedy_full/eval_jsut/cer_teacher_chars": "1_operational/eval/greedy_full/eval_jsut/cer_teacher_chars",
    }
    for tag, tb in want.items():
        assert tag in tags and tag_map[tag]["tb_tag"] == tb, tag
    buckets = {t: tag_map[t]["bucket"] for t in tags}
    assert all(buckets[t] == "2_loss_accuracy" for t in tags if t.startswith("summary/"))
    assert all(buckets[t] == "2_loss_accuracy" for t in tags if t.startswith("eval/mini/") and "/cer_" in t
               and not t.endswith("_chars"))
    # the combined-loss chart: in every event file (the crashed launch's and the resumed one's), its regexes drawing
    # exactly the three curves among the tags TensorBoard reads from the run
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from kitsune.runlog import TB_LAYOUT

    files = sorted(p for p in (run / "tb").iterdir() if "tfevents" in p.name)
    assert len(files) == 2
    (kind, regexes), = TB_LAYOUT["combined_loss"].values()
    assert all(tb_layouts(f) == [(0, {"combined_loss": {"combined_loss: train vs val": regexes}})] for f in files)
    acc = EventAccumulator(str(run / "tb"))
    acc.Reload()
    tb_scalars = acc.Tags()["scalars"]
    assert [[t for t in tb_scalars if re.match(rx, t)] for rx in regexes] == [
        [tag_map[t]["tb_tag"]] for t in ("combined_loss/train", "combined_loss/val", "combined_loss/val_full")]


def test_export_keeps_minis_apart(steps_runs, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_run", ROOT / "tools" / "export_run.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    run = steps_runs["mini"]
    t = exp.export(str(run), tmp_path / "export")
    want = [r["step"] for r in summary(run)["mini_history"]]
    assert sorted(set(t["eval_mini_tf"]["step"])) == sorted(set(t["eval_mini_probe_greedy"]["step"])) == want
    assert not set(t["eval_tf"]["step"]) & set(want)
    es = t["eval_summaries"]
    assert set(es.loc[es["mini"], "step"]) == set(want) and "headline/val_cer" in set(es["key"])
    # which eval sets headline/val_cer pools, and the final/complete flags, on every row of the step: step 0 decodes
    # the greedy subset, the epoch-end evals the complete sets; a mini eval has none of them
    assert es.loc[es["mini"], ["final", "complete", "scope"]].isna().all().all()
    full = es[~es["mini"]]
    for step, g in full.groupby("step"):
        s = json.loads((run / "evals" / f"step_{step}" / "summary.json").read_text(encoding="utf-8"))
        assert (g["scope"] == s["headline_scope"]["val_greedy"]).all(), step
        assert (g["complete"] == s["complete"]).all() and (g["final"] == s["final"]).all(), step
    assert set(full.loc[full["step"] == 0, "scope"]) == {"subset"} and not full.loc[full["step"] == 0, "complete"].any()
    ends = [e for e in epoch_ends(run) if e < MAX_STEPS]
    assert ends and set(full.loc[full["step"].isin(ends), "scope"]) == {"complete"}
    assert set(full.loc[full["final"].astype(bool), "step"]) <= {MAX_STEPS}
    assert {"final", "complete", "headline_scope/val_greedy"}.isdisjoint(es["key"])  # columns, not keys
    readme = (tmp_path / "export" / "README.md").read_text(encoding="utf-8")
    assert "| `scope` |" in readme and "| `final` |" in readme and "| `complete` |" in readme
    # the combined-loss chart as one table: each curve as TensorBoard draws it (the crashed launch's rows after the
    # restored step 8 left out), the sums of every eval's combined_loss in eval_summaries
    cl = t["combined_loss"]
    assert list(cl.columns) == ["series", "step", "value", "wall", "elapsed_s"]
    assert list(dict.fromkeys(cl["series"])) == ["train", "val", "val_full"]
    for series in ("train", "val", "val_full"):
        rows = cl[cl["series"] == series]
        assert rows["step"].tolist() == sorted(scalar(run, f"combined_loss/{series}")), series
        assert rows["value"].tolist() == pytest.approx([scalar(run, f"combined_loss/{series}")[k]
                                                        for k in rows["step"]])
    assert cl.loc[cl["series"] == "val", "step"].tolist() == [0, *want]
    assert {"combined_loss/val/kl_sum", "combined_loss/val_full/n_tok"} <= set(es["key"])
    assert (tmp_path / "export" / "combined_loss.csv").is_file() and "## `combined_loss.parquet`" in readme


def test_full_eval_every_other_epoch_on_the_wall_clock_and_gate_off(env, monkeypatch):
    """The wall clock (T = 1 h, the run cut at 10 steps): evals at the end of every 2nd epoch of the planner's plan
    (every_min 1e-6, which would fire after every step, is ignored), minis every 3 steps in between; eval.gate false
    makes the verdict "N/A" with the numbers."""
    monkeypatch.delenv("KITSUNE_DEADLINE", raising=False)
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    m = load_script("04_distill")
    path = write_config(env, "cad-wall", {
        "schedule": {"clock": "wall", "train_hours": 1.0, "max_steps": 10},
        "eval": {"every_steps": None, "every_min": 1e-6, "full_every_epochs": 2, "gate": False,
                 "mini": {"every_steps": 3}},
        "ckpt": {"full_every_steps": 1000},
    })
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "cad-wall")
    st = steps_of(run)
    prog = dict(zip(st["step"].astype(int), st["data/epoch_progress"]))
    ends = [s for s, p in prog.items() if float(p).is_integer() and int(p) % 2 == 0]
    assert ends and ends[0] < 10, prog
    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == 10 and (st["sched/phase"] < 2).all()
    assert [r["step"] for r in s["history"]] == sorted({0, *ends, 10})
    assert [r["epoch"] for r in s["history"] if r["step"] in ends] == [float(prog[e]) for e in ends]
    assert all(json.loads((run / "evals" / f"step_{e}" / "summary.json").read_text(encoding="utf-8"))["complete"]
               for e in ends)
    assert [r["step"] for r in s["mini_history"]] == [k for k in (3, 6, 9) if k not in ends]

    v = s["verdict"]
    assert v["verdict"] == "N/A" and v["reason"] == "gate disabled (sanity/overfit run)"
    assert set(v["numbers"]) >= {"sets", "n_sets_go", "trunc_rate", "trend", "thresholds"}
    assert "verdict" not in v["numbers"] and set(v["numbers"]["sets"]) == set(EVAL)
    assert json.loads((run / "evals" / "step_10" / "verdict.json").read_text(encoding="utf-8")) == v
    (ev,) = events(run, "verdict")
    assert ev["verdict"] == "N/A" and ev["numbers"] == v["numbers"]
    assert math.isfinite(s["headline"]["val_cer"])


def test_a_complete_eval_that_runs_past_T_is_the_final_eval(env, monkeypatch):
    """The wall clock: the first epoch-end eval (complete eval sets) takes longer than the budget left, so the loop
    ends at its step. The end phase reuses it as the final eval - one decode of the complete sets at that step, not
    two - and the verdict and the summary are that eval's numbers."""
    monkeypatch.delenv("KITSUNE_DEADLINE", raising=False)
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    m = load_script("04_distill")
    real = m.run_eval

    def slow(R, step, final=False, complete=None, **kw):  # the loop clock runs on during an eval: this one outlasts T
        out = real(R, step, final=final, complete=complete, **kw)  # kw: the step-0 eval's mini_val
        if complete and not final:
            R.st["train_s"] += float(R.cfg["schedule"]["train_hours"]) * 3600
        return out

    monkeypatch.setattr(m, "run_eval", slow)
    path = write_config(env, "cad-past-T", {
        "schedule": {"clock": "wall", "train_hours": 1.0, "max_steps": 10},  # max_steps: a cap only, never reached
        "eval": {"every_steps": None, "mini": {"every_steps": None}},
        "ckpt": {"full_every_steps": 1000},
    })
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "cad-past-T")
    ends = epoch_ends(run)
    s = summary(run)
    assert ends and s["status"] == "complete" and s["steps"] == ends[0] < 10
    assert [(e["at_step"], e["final"], e["complete"]) for e in events(run, "eval")] == [(0, False, False),
                                                                                      (ends[0], False, True)]
    assert [e["at_step"] for e in events(run, "eval_start")] == [0, ends[0]]
    assert [e["at_step"] for e in events(run, "eval_final_reused")] == [ends[0]]
    assert [r["step"] for r in s["history"]] == [0, ends[0]] and s["headline"] == s["history"][-1]["headline"]
    d = json.loads((run / "evals" / f"step_{ends[0]}" / "summary.json").read_text(encoding="utf-8"))
    for x in EVAL:
        assert s["verdict"]["sets"][x]["student"] == d["greedy_full"]["sets"][x]["cer_ref_corpus"]


def test_no_mini_that_would_run_past_T(env, monkeypatch):
    """The wall clock (T = 1 h): every mini eval takes 1000 s of loop clock. A mini whose run time, judged from the
    last one's, would carry the clock past T is skipped - it would end the loop at its step, and the final eval would
    then decode that same step again - so the minis stop at step 3 and the run trains on to max_steps."""
    monkeypatch.delenv("KITSUNE_DEADLINE", raising=False)
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    m = load_script("04_distill")
    real = m.run_mini_eval

    def slow(R, step):  # the loop clock runs on during a mini: 1000 s, and its recorded wall time says so
        out = real(R, step)
        R.st["train_s"] += 1000.0
        R.st["mini_history"][-1]["wall_s"] = 1000.0
        return out

    monkeypatch.setattr(m, "run_mini_eval", slow)
    path = write_config(env, "cad-mini-T", {
        "schedule": {"clock": "wall", "train_hours": 1.0, "max_steps": 6},  # max_steps: ends the run past the minis
        "eval": {"every_steps": None, "every_min": 1e6, "full_every_epochs": None, "mini": {"every_steps": 1}},
        "ckpt": {"full_every_steps": 1000},
    })
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "cad-mini-T")
    s = summary(run)
    # 1000, 2000, 3000 s after steps 1-3; at step 4, 3000 + 1000 >= 3600: skipped (without the look-ahead it ran,
    # the clock passed T and the final eval ran at step 4, the mini's step)
    assert [r["step"] for r in s["mini_history"]] == [1, 2, 3] and s["steps"] == 6
    assert [(e["at_step"], e["final"]) for e in events(run, "eval")] == [(0, False), (6, True)]
