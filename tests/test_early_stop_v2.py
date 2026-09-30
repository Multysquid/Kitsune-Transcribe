"""The full-data runs' trainer additions in scripts/04_distill.py (WP4a; CONTRACT.md 4.1, 4.2):

- the config keys and their validation (eval.dev, early_stop.smooth / allow_test_sets and the dev metrics,
  schedule.resume_reset / deadline_*, memory.probe_extended / probe_shapes, the epoch clock without max_steps,
  selection_recipe.full_study in validate_box) and the two new RESUME_FIXED keys
- the dev slice: the seeded dev pick (kitsune.fullrun.dev_pick over the selection's split "dev" rows), its store,
  runs/<run_id>/dev_ids.json, the `dev_store` event and the guard against other dev rows on a resume
- the dev evals: they change no training number; a dev_ce early stop on the steps and the epoch clocks (the dev
  cadence apart from the full evals), and on a CTC student
- early_stop.smooth (nothing counted before k values; min_evals and patience count smoothed checks)
- the COOLDOWN file, acting once or ignored inside a cooldown, through the module-level stop_requested (the study box's
  wrapper) and _stop_requested_trainer
- the resume reset flag: one-shot, epochs extension, the refusals, a repeat before the first save, the runs repo's
  COOLDOWN copy deleted (or, when that fails, ignored); a resume that switches to the epoch clock needs no flag
- 4c's extra memory-probe passes and synthetic shapes; 4d's deadline cooldown (schedule / start / compress / clear,
  no time left) and a whole run that ends by the deadline
- summary.json's end_reason and resume_resets

CPU only (the laptop GPU runs other jobs), tiny random students and a synthetic corpus in the real on-disk formats,
whose selection gets a dev slice from with_dev_split. A metric is made "flat" deterministically with min_delta_abs =
1e9: only the first checked value improves on it."""
import copy
import json
import math
import os
import sys
import time
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
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import load_script, make_fake_corpus, make_fake_selection  # noqa: E402
from kitsune import fullrun  # noqa: E402
from kitsune.store import ids_sha256 as store_ids_sha256  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
PEAK = 3e-3
FLAT = 1e9  # min_delta_abs: no check after the first improves
DEV_STEMS = {"src_a": ["train-00004"], "src_b": ["train-00003"]}


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


def trainer_json(run: Path, name: str) -> dict:
    return json.loads((run / "checkpoints" / name / "trainer.json").read_text(encoding="utf-8"))


def rule(**kw) -> dict:
    return dict(dict(enabled=True, metric="dev_ce", patience=3, min_delta_rel=0.0, min_delta_abs=0.0, min_evals=0,
                     floor=None, action="cooldown", smooth=1, allow_test_sets=True), **kw)


def with_dev_split(sel: Path, stems: dict[str, list[str]], out: Path | None = None) -> Path:
    """A copy of a selection whose kept rows of the named shards ({source: [stem, ...]}; teacher_file <source>/<stem>)
    are the dev slice, as make_selection.py's full mode marks it: split "dev", keep true, teacher_file their train
    shard, never in the probe or the greedy subset. The parquet's own metadata is kept."""
    table = pq.read_table(sel)
    df = table.to_pandas()
    files = {f"{s}/{st}" for s, v in stems.items() for st in v}
    dev = df["teacher_file"].isin(files) & df["keep"]
    assert dev.any(), files
    df.loc[dev, "split"] = fullrun.DEV_SPLIT
    df.loc[dev, ["in_probe", "in_greedy_subset"]] = False
    out = Path(out) if out else sel.with_name(sel.stem + "_dev.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(table.schema.metadata), out)
    return out


@pytest.fixture(autouse=True)
def grad_enabled():
    """Autograd on at the start of every test here, as in a trainer process (an earlier test file can leave it off)."""
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(prev)


@pytest.fixture(autouse=True)
def no_deadline(monkeypatch):
    monkeypatch.delenv("KITSUNE_DEADLINE", raising=False)
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    monkeypatch.delenv("KITSUNE_HEARTBEAT", raising=False)
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP", raising=False)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus in shards of 8 rows + selection with a dev slice (one train shard of each source) + a tiny
    AED student saved as 03 saves one, and a base config: the step clock, no smoke phase, no periodic checkpoints, a
    full eval only at step 0 and at the end unless a test asks for more."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("es_v2")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (40, "train"), "src_b": (32, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          rows_per_shard=8, dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=7)
    plain = make_fake_selection(fc, greedy_n=4, probe_n=5)
    sel = with_dev_split(plain, DEV_STEMS)

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
        "eval": {"every_steps": 1000, "greedy_subset": 2, "batch_s": 20, "check_baselines": False,
                 "final_full_greedy": False, "dev": {"per_source": 4}},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 1000, "keep_local": 5, "full_after_smoke": False},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2, "capture_env": False},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
    }
    return dict(root=root, base=base, fc=fc, sel=sel, plain=plain)


def write_config(env, name: str, over: dict) -> str:
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


def _unit_run(m, tmp_path, sets: list[str]):
    R = m.Run(cfg=m.load_config(None, sets), run_dir=tmp_path, device=torch.device("cpu"), amp=False)
    evs, sc = [], []
    R.log = SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw)),
                            scalars=lambda values, step: sc.append((step, dict(values))), elapsed=lambda: 0.0)
    return R, evs, sc


# ------------------------------------------------------------------------------------------------- config


def test_the_new_keys_default_to_the_trainer_before_them_and_are_validated():
    m = load_script("04_distill")
    d = m.load_config(None, [])
    assert d["eval"]["dev"] == dict(every_epochs=None, every_steps=None, per_source=600, seed=None, greedy=False,
                                    at_start=True, at_end=True)
    assert (d["early_stop"]["smooth"], d["early_stop"]["allow_test_sets"]) == (1, True)
    assert {k: d["schedule"][k] for k in ("resume_reset", "deadline_cooldown", "deadline_check_steps",
                                          "deadline_window_steps")} == dict(resume_reset=False, deadline_cooldown=False,
                                                                            deadline_check_steps=100,
                                                                            deadline_window_steps=1000)
    assert (d["memory"]["probe_extended"], d["memory"]["probe_shapes"]) == (False, None)
    assert d["selection_recipe"]["full_study"] is None and not m.dev_on(d) and not m.deadline_on(d)
    assert m.DEV_METRICS == ("dev_ce", "dev_kl", "dev_objective") and m.TEST_SET_METRICS == ("heldout_kl",)
    assert m.EARLY_STOP_METRICS == ("probe_kl", "heldout_kl", "train_loss", *m.DEV_METRICS)
    assert m.COOLDOWN_FILE == "COOLDOWN" and m.RESUME_FIXED[-2:] == ("eval.dev.per_source", "eval.dev.seed")

    ok = m.load_config(None, ["eval.dev.every_epochs=0.1", "eval.dev.seed=0", "eval.dev.greedy=true",
                              "early_stop.enabled=true", "early_stop.metric=dev_ce", "early_stop.smooth=5",
                              "early_stop.allow_test_sets=false", "schedule.clock=epochs", "schedule.epochs=3",
                              "schedule.deadline_cooldown=true", "memory.probe_extended=true",
                              'memory.probe_shapes=[{"name": "most_rows", "durations": [1.5, 2]}, '
                              '{"name": "long-1", "durations": [30]}]'])
    assert m.dev_on(ok) and m.dev_metric(ok) and m.deadline_on(ok)
    assert m.dev_on(m.load_config(None, ["eval.dev.every_steps=20"]))
    m.load_config(None, ["early_stop.metric=dev_kl"])  # a dev metric without eval.dev: fine while disabled
    m.load_config(None, ["early_stop.allow_test_sets=false", "early_stop.metric=probe_kl"])
    m.load_config(None, ["early_stop.allow_test_sets=false", "early_stop.metric=train_loss"])
    m.load_config(None, ["schedule.clock=steps", "schedule.max_steps=100", "schedule.warmup_steps=2",
                         "schedule.deadline_cooldown=true"])
    for bad in (["eval.dev.every_epochs=0"], ["eval.dev.every_epochs=-1"], ["eval.dev.every_epochs=true"],
                ["eval.dev.every_steps=0"], ["eval.dev.every_steps=2.5"],
                ["eval.dev.every_steps=1", "eval.dev.every_epochs=0.5"], ["eval.dev.per_source=0"],
                ["eval.dev.seed=-1"], ["eval.dev.seed=1.5"], ["eval.dev.greedy=x"], ["eval.dev.at_end=1"],
                ["eval.dev=null"], ["eval.dev.every=1"],
                ["early_stop.smooth=0"], ["early_stop.smooth=1.5"], ["early_stop.allow_test_sets=no"],
                ["early_stop.allow_test_sets=false"],  # the default metric heldout_kl reads the test sets
                ["early_stop.allow_test_sets=false", "early_stop.metric=heldout_kl", "early_stop.enabled=false"],
                ["early_stop.enabled=true", "early_stop.metric=dev_ce"],  # no eval.dev
                ["early_stop.metric=dev_loss"],
                ["eval.dev.every_steps=5", "lr_probe.enabled=true", "schedule.clock=steps", "schedule.max_steps=50",
                 "schedule.warmup_steps=2"],
                ["schedule.clock=epochs", "schedule.epochs=3", "schedule.max_steps=34620"],
                ["schedule.deadline_cooldown=true"],  # the wall clock
                ["schedule.deadline_cooldown=1"], ["schedule.deadline_check_steps=0"],
                ["schedule.deadline_window_steps=2.5"], ["schedule.resume_reset=yes"],
                ["memory.probe_extended=1"], ["memory.probe_shapes=[]"], ['memory.probe_shapes={"name": "a"}'],
                ['memory.probe_shapes=[{"name": "a b", "durations": [1]}]'],
                ['memory.probe_shapes=[{"name": "a", "durations": []}]'],
                ['memory.probe_shapes=[{"name": "a", "durations": [0]}]'],
                ['memory.probe_shapes=[{"name": "a", "durations": [1], "n": 2}]'],
                ['memory.probe_shapes=[{"name": "a", "durations": [1]}, {"name": "a", "durations": [2]}]']):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)


def test_full_study_recipe_is_the_box_paths_check():
    m = load_script("04_distill")
    block = json.dumps(fullrun.FULL_STUDY)
    ok = m.load_config(None, [f"selection_recipe.full_study={block}"])
    assert ok["selection_recipe"]["full_study"] == fullrun.FULL_STUDY
    m.load_config(None, [f"selection_recipe.full_study={json.dumps(fullrun.SMOKE_STUDY)}"])
    for bad in ([f"selection_recipe.full_study={json.dumps(dict(fullrun.FULL_STUDY, f1a_max=0.4))}"],
                [f"selection_recipe.full_study={json.dumps(dict(fullrun.FULL_STUDY, dev_rule=2))}"],
                ['selection_recipe.full_study={"f1a_max": 0.5}'], ["selection_recipe.full_study=1"],
                [f"selection_recipe.full_study={block}",
                 'selection_recipe.study={"agree_max": 0.5, "f1a_max": 0.5, "dedup_min_chars": 15, "probe_n": 300}']):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
    with pytest.raises(SystemExit, match="set one"):
        from kitsune import prereg

        m.load_config(None, [f"selection_recipe.full_study={block}",
                             f"selection_recipe.study={json.dumps(prereg.STUDY_SELECTION)}"])


def test_the_dev_pick_is_resume_fixed():
    m = load_script("04_distill")
    saved = m.load_config(None, ["eval.dev.every_steps=5"])
    for key, value in (("eval.dev.per_source", 500), ("eval.dev.seed", 7), ("eval.dev", dict(saved["eval"]["dev"],
                                                                                            every_steps=6))):
        with pytest.raises(SystemExit, match="changes the step plan"):
            m.resume_overrides(saved, [(key, value)])
    changed, same = m.resume_overrides(saved, [("eval.dev.every_steps", 9), ("eval.dev.per_source", 600),
                                               ("eval.dev.greedy", True), ("early_stop.smooth", 3)])
    assert changed == {"eval.dev.every_steps": 9, "eval.dev.greedy": True, "early_stop.smooth": 3}
    assert same == {"eval.dev.per_source": 600}


def test_end_reason_and_the_summary_fields(tmp_path):
    m = load_script("04_distill")
    R, _, _ = _unit_run(m, tmp_path, ["schedule.clock=steps", "schedule.max_steps=20", "schedule.warmup_steps=2"])
    R.st["step"] = 20
    s = m.summary_full(R)
    assert s == dict(end_reason="schedule", resume_resets=0)  # nothing else while there is nothing to say
    R.st["step"] = 7
    m.early_stop_trigger(R, "patience", "cooldown")
    assert m.end_reason(R) == "early_stop"
    R.st["early_stop"]["triggered"]["reason"] = "cooldown_file"
    assert m.end_reason(R) == "cooldown_file"
    assert R.st["early_stop"]["cooldown"]["T"] == 9.0  # 7 + ceil(0.2 x 7)
    R.st["deadline_cooldown"] = dict(t_c=7.0, T=8.0, action="compress")  # 4d cut the early cooldown shorter
    assert m.end_reason(R) == "deadline" and R.progress() == (7.0, 8.0)
    R.st["deadline_cooldown"]["T"] = 9.0  # a tie: 4d did not shorten it
    assert m.end_reason(R) == "cooldown_file"
    m.early_stop_trigger(R, "stop_file", "stop")
    assert m.end_reason(R) == "stop_file"
    R.st["dev_history"] = [dict(step=0, dev_ce=1.0)]
    R.st["schedule_resets"] = [dict(at_step=3)]
    R.st["resume_resets"] = 1
    s = m.summary_full(R)
    assert set(s) == {"end_reason", "resume_resets", "dev_history", "deadline_cooldown", "schedule_resets"}
    full = m.make_summary(R, "complete")
    assert full["end_reason"] == "stop_file" and full["resume_resets"] == 1 and full["stopped_early"]
    # the wall clock: a budget fitted to the deadline is the deadline's doing
    R, _, _ = _unit_run(m, tmp_path, [])
    assert m.end_reason(R) == "schedule"
    R.budget_s = 3600.0
    assert m.end_reason(R) == "deadline"
    # a run that failed before its epoch plan still gets its summary
    R, _, _ = _unit_run(m, tmp_path, ["schedule.clock=epochs", "schedule.epochs=2"])
    assert m.make_summary(R, "failed")["end_reason"] == "schedule"


# ----------------------------------------------------------------------------------------- early-stop smoothing


def test_smoothing_counts_nothing_before_k_values(tmp_path):
    """smooth k: every raw value joins the window; nothing is counted (evals, patience, min_evals) until there are k,
    then the rule reads their mean (a None in the window: no improvement). early_stop/value is the smoothed value,
    early_stop/raw the newest; the early_stop event gains raw and smooth."""
    m = load_script("04_distill")
    R, evs, sc = _unit_run(m, tmp_path, ["eval.dev.every_steps=1", "early_stop.enabled=true",
                                         "early_stop.metric=dev_ce", "early_stop.smooth=3", "early_stop.patience=2",
                                         "early_stop.min_evals=1", "early_stop.min_delta_rel=0",
                                         "early_stop.action=stop"])

    def check(step, v):
        R.st["step"] = step
        R.st["dev_history"].append(dict(step=step, dev_ce=v, dev_kl=None, dev_objective=None))
        return m.early_stop_check(R, step)

    assert check(1, 3.0) is False and check(2, 2.0) is False
    es = R.st["early_stop"]
    assert es["evals"] == 0 and es["best"] is None and es["recent"] == [3.0, 2.0]
    assert sc == [(1, {"early_stop/raw": 3.0}), (2, {"early_stop/raw": 2.0})]
    assert check(3, 1.0) is False  # mean 2.0: the first smoothed value is the best
    assert es["evals"] == 1 and es["best"] == pytest.approx(2.0) and es["recent"] == [3.0, 2.0, 1.0]
    assert sc[-1][1]["early_stop/value"] == pytest.approx(2.0) and sc[-1][1]["early_stop/raw"] == 1.0
    assert check(4, float("nan")) is False  # a missing value in the window: no improvement
    assert es["recent"] == [2.0, 1.0, None] and es["value"] is None and es["evals_since_best"] == 1
    assert check(5, 1.0) is True  # [1.0, None, 1.0]: still no improvement, patience 2 -> the stop
    (ev,) = [e for e in evs if e["kind"] == "early_stop"]
    assert ev["reason"] == "patience" and ev["smooth"] == 3 and ev["raw"] == 1.0 and ev["metric"] == "dev_ce"
    assert ev["value"] is None and ev["best"] == pytest.approx(2.0)
    # the dev value is the newest dev record's only at its own step
    R.st["step"] = 6
    assert m.early_stop_value(R, "dev_ce") is None and m.early_stop_value(R, "dev_kl") is None

    # decision 16's rule (smooth 5, patience 6, min_evals 5), flat after the first smoothed value: the first trigger
    # is at check 11 - the first smoothed value at check 5, then 6 without an improvement
    R, evs, _ = _unit_run(m, tmp_path, ["eval.dev.every_epochs=0.1", "early_stop.enabled=true",
                                        "early_stop.metric=dev_ce", "early_stop.smooth=5", "early_stop.patience=6",
                                        "early_stop.min_evals=5", f"early_stop.min_delta_abs={FLAT}"])
    for n in range(1, 21):
        if check(n, 1.0 / n):
            break
    (ev,) = [e for e in evs if e["kind"] == "early_stop"]
    assert ev["at_step"] == 11 and R.st["early_stop"]["evals"] == 7 and ev["action"] == "cooldown"
    # smooth 1 (the default): the event keeps exactly its fields of before
    R, evs, _ = _unit_run(m, tmp_path, ["eval.dev.every_steps=1", "early_stop.enabled=true",
                                        "early_stop.metric=dev_ce", "early_stop.patience=1", "early_stop.action=stop",
                                        "early_stop.min_evals=0", f"early_stop.min_delta_abs={FLAT}"])
    assert not check(1, 1.0) and check(2, 1.0)
    (ev,) = [e for e in evs if e["kind"] == "early_stop"]
    assert set(ev) == {"kind", "metric", "value", "best", "best_step", "evals_since_best", "reason", "action",
                       "at_step", "epoch"}


# ------------------------------------------------------------------------------------------------- the dev slice


def test_dev_ids_are_a_seeded_pick_of_the_dev_rows(env, tmp_path):
    m = load_script("04_distill")
    cfg = m.load_config(write_config(env, "unit-dev", {"eval": {"dev": {"every_steps": 1}}}), [])
    evs = []
    log = SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw)))
    ids, rec = m.dev_ids(cfg, log)
    sel = pd.read_parquet(env["sel"])
    dev = sel[(sel["split"] == "dev") & sel["keep"]]
    assert set(ids) <= set(dev["id"]) and len(ids) == 8 and len(dev) > 8  # 4 of each source's dev rows
    assert ids == fullrun.dev_pick(zip(dev["id"], dev["source"]), ["src_a", "src_b"], 4, 1234)
    assert ids == [i for i in dev["id"] if i in set(ids)]  # in selection order
    assert rec["ids"] == {s: [i for i in ids if i.startswith(s + "/")] for s in ("src_a", "src_b")}
    assert rec["ids_sha256"] == m._ids_sha256(ids) and rec["picked"] == {"src_a": 4, "src_b": 4}
    assert rec["pool"] == {s: int((dev["source"] == s).sum()) for s in ("src_a", "src_b")}
    (ev,) = evs
    assert ev["kind"] == "subset" and ev["split"] == "dev" and ev["ids_sha256"] == rec["ids_sha256"]
    assert "ids" not in ev  # the ids go to dev_ids.json
    # the selection-order digest is the one make_selection.py full mode's sidecar records for the same pick
    # (dev.scored_default.ids_sha256 = kitsune.store.ids_sha256 of fullrun.dev_pick's list)
    sidecar_way = store_ids_sha256(fullrun.dev_pick(zip(dev["id"], dev["source"]), ["src_a", "src_b"], 4, 1234))
    assert rec["ids_sha256_selection"] == sidecar_way == ev["ids_sha256_selection"]
    assert m.dev_ids(cfg, log)[0] == ids  # deterministic
    assert m.dev_store_name(rec) == f"dev_p4_s1234_{rec['ids_sha256'][:8]}"
    # the same rows in another order: the same pick and store (the sorted digest), and the selection-order digest
    # follows the file's order, as that file's sidecar would
    table = pq.read_table(env["sel"])
    rev = tmp_path / "reversed.parquet"
    pq.write_table(table.take(pa.array(range(table.num_rows - 1, -1, -1))), rev)
    ids_r, rec_r = m.dev_ids(dict(cfg, selection=str(rev)), log)
    assert ids_r == ids[::-1] and rec_r["ids_sha256"] == rec["ids_sha256"]
    assert m.dev_store_name(rec_r) == m.dev_store_name(rec)
    assert rec_r["ids_sha256_selection"] == store_ids_sha256(ids[::-1]) != rec["ids_sha256_selection"]
    cfg2 = copy.deepcopy(cfg)
    cfg2["eval"]["dev"].update(seed=99, per_source=50)  # a larger per_source takes every dev row
    assert sorted(m.dev_ids(cfg2, log)[0]) == sorted(dev["id"])
    assert m.build_dev_store(m.load_config(None, []), log) is None  # off: no store
    cfg3 = copy.deepcopy(cfg)
    cfg3["selection"] = str(env["plain"])  # a selection without a dev slice
    with pytest.raises(SystemExit, match="split 'dev'"):
        m.dev_ids(cfg3, log)


def test_setup_data_opens_the_dev_store_and_guards_its_rows(env, tmp_path):
    """setup_data under eval.dev: the dev store holds the picked dev rows and no train row, dev_ids.json and the
    `dev_store` / `data` events say so, and a state that scored other dev rows stops the resume."""
    m = load_script("04_distill")
    cfg = m.load_config(write_config(env, "unit-setup", {"eval": {"dev": {"every_steps": 1}}}), [])
    R = m.Run(cfg=cfg, run_dir=tmp_path, device=torch.device("cpu"), amp=False)
    evs = []
    R.log = SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw)))
    m.setup_data(R)
    ids, rec = m.dev_ids(cfg, SimpleNamespace(event=lambda *a, **k: None))
    assert sorted(u.id for u in R.devstore.utts) == sorted(ids) and {u.split for u in R.devstore.utts} == {"dev"}
    assert not {u.id for u in R.devstore.utts} & {u.id for u in R.train.utts}
    assert R.devstore.cache_dir.name == m.dev_store_name(rec)
    (ds,) = [e for e in evs if e["kind"] == "dev_store"]
    assert {k: ds[k] for k in ("name", "n", "n_ids", "per_source", "seed", "ids_sha256", "ids_sha256_selection",
                               "n_in_train")} == dict(
        name=m.dev_store_name(rec), n=8, n_ids=8, per_source=4, seed=1234, ids_sha256=rec["ids_sha256"],
        ids_sha256_selection=store_ids_sha256(ids), n_in_train=0)
    data = next(e for e in evs if e["kind"] == "data")
    assert data["dev_utts"] == 8 and data["stores_reused"] == {"train": False, "eval": False, "dev": False}
    assert set(data["dev_per_source"]) == {"src_a", "src_b"}
    got = json.loads((tmp_path / "dev_ids.json").read_text(encoding="utf-8"))
    assert got == dict(seed=1234, per_source=4, ids_sha256=rec["ids_sha256"],
                       ids_sha256_selection=rec["ids_sha256_selection"], ids=rec["ids"])
    assert R.st["dev"] == dict(ids_sha256=rec["ids_sha256"], n_ids=8, per_source=4, n=8,
                               store_ids_sha256=m._ids_sha256(ids))
    # the same rows again (a resume): fine, and every store is a reused cache now (the token stores say so too)
    evs.clear()
    m.setup_data(R)
    data = next(e for e in evs if e["kind"] == "data")
    assert data["stores_reused"] == {"train": True, "eval": True, "dev": True}
    # a state from before the store keys takes them from this store
    R.st["dev"] = dict(ids_sha256=rec["ids_sha256"], n_ids=8, per_source=4)
    m.setup_data(R)
    assert R.st["dev"]["n"] == 8 and R.st["dev"]["store_ids_sha256"] == m._ids_sha256(ids)
    # the same pick, but the store holds other rows (a shard missing on a new host dropped some): a SystemExit
    R.st["dev"] = dict(R.st["dev"], n=7, store_ids_sha256=m._ids_sha256(ids[1:]))
    with pytest.raises(SystemExit, match="holds 8 rows .* scored 7"):
        m.setup_data(R)
    # other picked rows: a SystemExit naming both
    R.st["dev"] = dict(R.st["dev"], ids_sha256="0" * 64)
    with pytest.raises(SystemExit, match="the dev rows changed"):
        m.setup_data(R)
    # eval.dev off: no dev store, no file, stores_reused without it
    R2 = m.Run(cfg=m.load_config(write_config(env, "unit-nodev", {}), []), run_dir=tmp_path / "b",
               device=torch.device("cpu"), amp=False)
    (tmp_path / "b").mkdir()
    evs.clear()
    R2.log = SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw)))
    m.setup_data(R2)
    data = next(e for e in evs if e["kind"] == "data")
    assert R2.devstore is None and "dev_utts" not in data and set(data["stores_reused"]) == {"train", "eval"}
    assert not (tmp_path / "b" / "dev_ids.json").exists() and not [e for e in evs if e["kind"] == "dev_store"]


def test_decode_preflight_decodes_the_dev_rows_after_the_others():
    m = load_script("04_distill")

    class Store:
        def __init__(self, sources, fail=()):
            self.utts = [SimpleNamespace(source=s) for s in sources]
            self.fail = fail
            self.read = []

        def wave(self, i):
            self.read.append(i)
            if self.utts[i].source in self.fail:
                raise RuntimeError("no")
            return np.zeros(8, np.float32)

    evs = []
    R = SimpleNamespace(cfg={"smoke": {"decode_per_set": 2}, "seed": 1}, train=Store(["a", "a", "b"]),
                        evalstore=Store(["e"] * 3), devstore=Store(["a", "b", "b"]),
                        log=SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw))))
    m.decode_preflight(R)
    assert set(evs[0]["sets"]) == {"train/a", "train/b", "eval/e", "dev/a", "dev/b"}
    R2 = SimpleNamespace(cfg=R.cfg, train=Store(["a", "a", "b"]), evalstore=Store(["e"] * 3),
                         log=SimpleNamespace(event=lambda kind, **kw: None))
    m.decode_preflight(R2)  # no dev store: the other draws as before it
    assert R2.train.read == R.train.read and R2.evalstore.read == R.evalstore.read
    R.devstore = Store(["a", "b"], fail=("b",))
    with pytest.raises(m.SmokeFailed, match="dev b"):
        m.decode_preflight(R)


# ------------------------------------------------------------------------------------------------ whole runs


def test_dev_evals_change_no_training_number(env):
    """eval.dev.every_steps 1 with the early stop off vs no dev eval: the same steps, losses, LRs, gradients and
    full-eval results, bit for bit. The dev run has a step-0 dev eval, one after every step but the last and the final
    one (at_end), each a dev_history record, an `eval_dev` event, evals/step_<N>_dev/ and eval/dev/* scalars, and never
    an `eval` event of its own."""
    m = load_script("04_distill")
    over = {"schedule": {"max_steps": 8}, "eval": {"every_steps": 4}}
    off = write_config(env, "dev-off", over)
    on = write_config(env, "dev-on", merged(over, {"eval": {"dev": {"every_steps": 1}}}))
    assert m.main(["--config", off]) == 0
    assert m.main(["--config", on]) == 0
    a, b = one_run(env["root"], "dev-off"), one_run(env["root"], "dev-on")
    sa, sb = steps_of(a), steps_of(b)
    assert sa["step"].tolist() == sb["step"].tolist() == list(range(1, 9))
    for col in ("loss/total", "loss/objective", "loss/kl", "loss/ce", "opt/lr", "opt/grad_norm", "sched/phase"):
        np.testing.assert_array_equal(sa[col].to_numpy(), sb[col].to_numpy(), err_msg=col)
    ha, hb = summary(a)["history"], summary(b)["history"]
    assert [r["step"] for r in ha] == [r["step"] for r in hb] == [0, 4, 8]
    assert [(r["heldout_kl"], r["probe_kl"]) for r in ha] == [(r["heldout_kl"], r["probe_kl"]) for r in hb]
    assert [(e["at_step"], e["final"]) for e in events(a, "eval")] == [(e["at_step"], e["final"])
                                                                        for e in events(b, "eval")]

    sb_ = summary(b)
    dh = sb_["dev_history"]
    assert [r["step"] for r in dh] == list(range(0, 9)) and [r["final"] for r in dh] == [False] * 8 + [True]
    assert "dev_history" not in summary(a) and summary(a)["end_reason"] == sb_["end_reason"] == "schedule"
    ed = events(b, "eval_dev")
    assert [e["at_step"] for e in ed] == list(range(0, 9)) and ed[-1]["final"] is True
    for r, e in zip(dh, ed):
        assert e["dev_ce"] == pytest.approx(r["dev_ce"]) and e["dev_kl"] == pytest.approx(r["dev_kl"])
        assert r["dev_objective"] == pytest.approx(1.0 * r["dev_kl"] + 0.8 * r["dev_ce"])
        pooled = sum(p["ce"] * p["n_tok"] for p in r["per_source"].values()) / r["n_tok"]
        assert set(r["per_source"]) == {"src_a", "src_b"} and r["dev_ce"] == pytest.approx(pooled)
        assert r["n_utts"] == 8 and "cer" not in r  # eval.dev.greedy false: no decode
    assert dh[0]["dev_ce"] > dh[-1]["dev_ce"]  # the student learns the dev rows' teacher too
    s3 = json.loads((b / "evals" / "step_3_dev" / "summary.json").read_text(encoding="utf-8"))
    assert s3["step"] == 3 and s3["dev"] is True and set(s3["tf"]["sets"]) == {"src_a", "src_b"}
    assert sorted(p.name for p in (b / "evals" / "step_3_dev").glob("*.parquet")) == ["tf_src_a.parquet",
                                                                                     "tf_src_b.parquet"]
    assert sorted(scalar(b, "eval/dev/ce")) == list(range(0, 9)) and scalar(b, "eval/dev/tf/src_a/kl")
    assert not scalar(a, "eval/dev/ce")
    (ds,) = events(b, "dev_store")
    assert ds["n_in_train"] == 0 and ds["n"] == 8
    ids = json.loads((b / "dev_ids.json").read_text(encoding="utf-8"))
    assert ids["ids_sha256"] == ds["ids_sha256"] and sum(len(v) for v in ids["ids"].values()) == 8
    assert ids["ids_sha256_selection"] == ds["ids_sha256_selection"]
    assert not any(e["kind"] == "early_stop" for e in events(b))
    from kitsune.runlog import load_tag_map

    tm = load_tag_map(b / "metrics" / "tag_map.json")
    assert not [t for t, e in tm.items() if t.startswith("eval/dev/") and e.get("unmapped")]


def test_dev_ce_early_stop_on_the_step_clock(env):
    """metric dev_ce, flat, smooth 2, patience 2, a dev eval after every step and a full eval every 3: the checks run
    after the dev evals only (the full eval at step 3 counts nothing), the first smoothed value at step 2, the trigger
    at step 4, the cooldown over ceil(0.5 x 4) = 2 steps after a pre_cooldown state at step 4, the end phase and exit
    0 with stopped_early naming dev_ce and end_reason early_stop."""
    m = load_script("04_distill")
    path = write_config(env, "dev-steps", {"eval": {"every_steps": 3, "dev": {"every_steps": 1}},
                                           "early_stop": rule(smooth=2, patience=2, min_delta_abs=FLAT)})
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "dev-steps")
    assert steps_of(run)["step"].tolist() == list(range(1, 7))
    (es,) = events(run, "early_stop")
    assert (es["at_step"], es["reason"], es["metric"], es["smooth"]) == (4, "patience", "dev_ce", 2)
    assert es["cooldown"] == dict(t_c=4.0, T=6.0, clock="steps", at_step=4, already=False)
    dh = {r["step"]: r for r in summary(run)["dev_history"]}
    assert es["raw"] == pytest.approx(dh[4]["dev_ce"])
    assert es["value"] == pytest.approx((dh[3]["dev_ce"] + dh[4]["dev_ce"]) / 2)
    assert scalar(run, "early_stop/evals_since_best") == {2: 0, 3: 1, 4: 2}
    assert sorted(scalar(run, "early_stop/raw")) == [1, 2, 3, 4]
    assert sorted(scalar(run, "early_stop/value")) == [2, 3, 4]
    assert [e["at_step"] for e in events(run, "eval")] == [0, 3, 6]
    assert sorted(dh) == [0, 1, 2, 3, 4, 5, 6]  # none at the last loop step 6: the final one covers it
    assert any(e["name"] == "full_step_4" and e["reason"] == "pre_cooldown" for e in events(run, "checkpoint"))
    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == 6 and s["end_reason"] == "early_stop"
    assert s["stopped_early"]["metric"] == "dev_ce" and s["stopped_early"]["at_step"] == 4


def test_dev_ce_early_stop_on_the_epoch_clock(env):
    """The epoch clock with a complete eval at every epoch end and a dev eval at every quarter epoch: the dev evals
    fall at the steps where the epoch progress crosses a multiple of 0.25 (whatever the full-eval cadence), a flat
    dev_ce with patience 2 triggers mid-epoch, the plan ends early (cooldown ceil(0.3 x t)), the complete evals stay
    at the epoch ends."""
    m = load_script("04_distill")
    path = write_config(env, "dev-epochs", {
        "subset": {"train_audio_s": 24, "eval_audio_s": 4},
        "schedule": {"clock": "epochs", "epochs": 4, "warmup_steps": 300, "cooldown_frac": 0.3, "max_steps": None},
        "batch": {"step_audio_s": 4, "micro_audio_s": 3, "pool_micro": 4},
        "eval": {"every_steps": None, "full_every_epochs": 1, "probe_is_train": True,
                 "dev": {"every_epochs": 0.25, "per_source": 3}},
        "early_stop": rule(patience=2, min_delta_abs=FLAT),
    })
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "dev-epochs")
    st = steps_of(run)
    ep = dict(zip(st["step"].astype(int), st["data/epoch_progress"]))
    (es,) = events(run, "early_stop")
    t = es["at_step"]
    end = t + math.ceil(0.3 * t)
    assert st["step"].tolist() == list(range(1, end + 1)) and es["cooldown"]["T"] == float(end)
    want, last = [], 0.0
    for s in range(1, end):  # never at the loop's last step
        if math.floor(ep[s] / 0.25 + 1e-9) > math.floor(last / 0.25 + 1e-9):
            want.append(s)
            last = ep[s]
    dh = summary(run)["dev_history"]
    assert [r["step"] for r in dh] == [0, *want, end] and dh[-1]["final"]
    assert t == want[2]  # the first dev check sets the best, then 2 without an improvement
    assert abs(ep[t] - round(ep[t])) > 1e-6  # a trigger inside an epoch, not at its end
    sched = events(run, "schedule")[0]
    ends = np.cumsum(sched["steps_per_epoch"]).tolist()
    assert end < sched["total_steps"]
    assert [e["at_step"] for e in events(run, "eval")] == [0, *[e for e in ends if e < end], end]
    s = summary(run)
    assert s["status"] == "complete" and s["end_reason"] == "early_stop" and s["epochs"] < 4


def test_the_cooldown_file(env, monkeypatch):
    """runs/<run_id>/COOLDOWN created during step 3 of a run without early stopping: the WSD cooldown starts after
    step 3 over ceil(0.5 x 3) = 2 steps (reason cooldown_file, metric None), exactly one early_stop event, the file
    stays, the state remembers it acted, summary.json says cooldown_file."""
    m = load_script("04_distill")
    path = write_config(env, "cool-file", {})
    orig = m.train_step

    def train_step(R, step, lr, mbs, epoch):
        out = orig(R, step, lr, mbs, epoch)
        if step == 3:
            (R.run_dir / "COOLDOWN").touch()
        return out

    monkeypatch.setattr(m, "train_step", train_step)
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "cool-file")
    assert steps_of(run)["step"].tolist() == [1, 2, 3, 4, 5]
    (es,) = events(run, "early_stop")
    assert (es["reason"], es["action"], es["metric"], es["at_step"]) == ("cooldown_file", "cooldown", None, 3)
    assert es["cooldown"] == dict(t_c=3.0, T=5.0, clock="steps", at_step=3, already=False)
    assert not events(run, "cooldown_file") and (run / "COOLDOWN").exists()
    assert any(e["reason"] == "pre_cooldown" and e["name"] == "full_step_3" for e in events(run, "checkpoint"))
    assert trainer_json(run, "full_step_5")["st"]["early_stop"]["cooldown_file"] == {"at_step": 3}
    s = summary(run)
    assert s["end_reason"] == "cooldown_file" and s["stopped_early"]["reason"] == "cooldown_file"


def test_the_cooldown_file_inside_a_cooldown_changes_nothing(tmp_path):
    """Inside the scheduled cooldown, or after a dev trigger's early cooldown, COOLDOWN is only recorded (one
    `cooldown_file` event, ignored) and never goes through early_stop_trigger, so stopped_early keeps the dev
    trigger. Checked through the module-level stop_requested (the study box's wrapper) and _stop_requested_trainer;
    without the file nothing is logged."""
    m = load_script("04_distill")
    sets = ["schedule.clock=steps", "schedule.max_steps=20", "schedule.cooldown_frac=0.3", "schedule.warmup_steps=2"]
    for name, fn in (("wrapper", m.stop_requested), ("trainer", m._stop_requested_trainer)):
        run = tmp_path / name
        run.mkdir()
        R, evs, _ = _unit_run(m, run, sets)
        R.st["step"] = 15  # >= t_c 14 of 20
        assert fn(R) is False and not evs
        (run / "COOLDOWN").touch()
        assert fn(R) is False and fn(R) is False
        assert evs == [dict(kind="cooldown_file", at_step=15, ignored="cooldown under way", epoch=0.0)]
        assert R.st["early_stop"]["triggered"] is None and R.progress() == (15.0, 20.0)

    run = tmp_path / "after-dev"
    run.mkdir()
    R, evs, _ = _unit_run(m, run, sets + ["eval.dev.every_steps=1", "early_stop.enabled=true",
                                          "early_stop.metric=dev_ce"])
    R.st["step"] = 7
    m.early_stop_trigger(R, "patience", "cooldown")  # the dev trigger: t_c 7, T 7 + ceil(2.1) = 10
    dev_trigger = m.make_summary(R, "complete")["stopped_early"]
    assert dev_trigger["metric"] == "dev_ce" and R.progress() == (7.0, 10.0)
    R.st["step"] = 8
    (run / "COOLDOWN").touch()
    assert m.stop_requested(R) is False
    assert [e["kind"] for e in evs] == ["early_stop", "cooldown_file"] and evs[-1]["ignored"]
    assert m.make_summary(R, "complete")["stopped_early"] == dev_trigger and R.progress() == (8.0, 10.0)

    # outside a cooldown, early stop disabled: it acts (metric None), once
    run = tmp_path / "acts"
    run.mkdir()
    R, evs, _ = _unit_run(m, run, sets)
    R.st["step"] = 4
    (run / "COOLDOWN").touch()
    assert m.stop_requested(R) is False and m.stop_requested(R) is False
    assert [e["kind"] for e in evs] == ["early_stop"] and evs[0]["metric"] is None
    assert R.st["early_stop"]["cooldown"] == dict(t_c=4.0, T=6.0, clock="steps", at_step=4)  # 4 + ceil(1.2)
    (run / "STOP").touch()  # the STOP file still ends the run
    assert m.stop_requested(R) is True and evs[-1]["reason"] == "stop_file"


def test_the_resume_reset(env, monkeypatch):
    """A dev-triggered run on the epoch clock ends early. The owner's continuation: --resume from its pre_cooldown
    state with --set schedule.resume_reset=true --set schedule.epochs=3. Refused: the epochs change without the flag,
    the flag on a fresh start, the flag with a STOP file present. A first continuation dies before its first save
    (after renaming COOLDOWN); the same argv again resets the same state the same way: one reset, total_steps planned
    again for 3 epochs, the early stop fresh, a new pre_cooldown state at the new t_c, the run to its new end, and
    every state after it saved with the flag false."""
    m = load_script("04_distill")
    over = {"subset": {"train_audio_s": 24, "eval_audio_s": 4},
            "schedule": {"clock": "epochs", "epochs": 2, "warmup_steps": 300, "cooldown_frac": 0.5, "max_steps": None},
            "batch": {"step_audio_s": 4, "micro_audio_s": 3, "pool_micro": 4},
            "eval": {"every_steps": None, "every_min": None, "probe_is_train": True,
                     "dev": {"every_steps": 1, "per_source": 3}},
            "early_stop": rule(patience=2, min_delta_abs=FLAT)}
    path = write_config(env, "reset", over)
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "reset")
    s1 = summary(run)
    T_old = events(run, "schedule")[0]["total_steps"]
    assert (s1["steps"], s1["end_reason"], s1["resume_resets"]) == (5, "early_stop", 0)  # trigger 3, T 3 + 2
    pc = run / "checkpoints" / "full_step_3"
    assert trainer_json(run, "full_step_3")["reason"] == "pre_cooldown"
    flag = ["--set", "schedule.resume_reset=true", "--set", "schedule.epochs=3", "--set", "early_stop.enabled=false"]

    with pytest.raises(SystemExit, match="schedule.resume_reset=true"):  # plan_epochs would ignore it silently
        m.main(["--config", path, "--resume", str(pc), "--set", "schedule.epochs=3"])
    with pytest.raises(SystemExit, match="one-shot flag"):
        m.main(["--config", path, "--set", "schedule.resume_reset=true"])
    assert len(list((env["root"] / "runs").glob("reset-2*"))) == 1  # neither made a run dir
    (run / "STOP").touch()
    with pytest.raises(SystemExit, match="STOP"):
        m.main(["--config", path, "--resume", str(pc), *flag])
    (run / "STOP").unlink()

    (run / "COOLDOWN").touch()  # left over from the first run: it must not fire again at once
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "6")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", path, "--resume", str(pc), *flag])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    assert not (run / "COOLDOWN").exists() and len(list(run.glob("COOLDOWN.consumed-*"))) == 1
    assert not list((run / "checkpoints").glob("full_step_[4-9]*"))  # nothing saved before the crash

    assert m.main(["--config", path, "--resume", str(pc), *flag]) == 0
    resets = events(run, "resume_reset")
    assert len(resets) == 2 and resets[0]["cooldown_file_consumed"].startswith("COOLDOWN.consumed-")
    r = resets[-1]
    T_new = events(run, "schedule")[-1]["total_steps"]
    assert T_new > T_old and r["total_steps_before"] == T_old and r["total_steps_after"] == T_new
    assert (r["at_step"], r["epochs"], r["resume_resets"]) == (3, 3, 1)
    assert r["early_stop_before"]["triggered"]["metric"] == "dev_ce" and r["pre_cooldown_full_before"] == "full_step_3"
    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == T_new and s["resume_resets"] == 1
    assert len(s["schedule_resets"]) == 1 and s["end_reason"] == "schedule" and s["stopped_early"] is None
    t_c = math.ceil(0.5 * T_new)
    pcs = [e["name"] for e in events(run, "checkpoint") if e["reason"] == "pre_cooldown"]
    assert pcs[-1] == f"full_step_{t_c}" and pcs.count("full_step_3") == 1
    end = trainer_json(run, f"full_step_{T_new}")
    assert end["cfg"]["schedule"]["resume_reset"] is False and end["cfg"]["schedule"]["epochs"] == 3
    assert end["st"]["resume_resets"] == 1 and end["st"]["early_stop"]["triggered"] is None
    assert trainer_json(run, f"full_step_{t_c}")["cfg"]["schedule"]["resume_reset"] is False
    assert steps_of(run)["step"].tolist() == list(range(1, T_new + 1))
    assert [p.name for p in (run / "checkpoints").glob("abandoned-*/step_5")]  # the first end, set aside


def test_resume_reset_refusals_on_their_own(tmp_path):
    m = load_script("04_distill")
    R, evs, _ = _unit_run(m, tmp_path, ["schedule.clock=steps", "schedule.max_steps=10", "schedule.warmup_steps=2",
                                        "schedule.resume_reset=true"])
    R.st.update(step=12, branch=dict(t_c=5.0, end_step=10))
    with pytest.raises(SystemExit, match="T/2 branch"):
        m.resume_reset_start(R)
    R.st["branch"] = None
    before = m.resume_reset_start(R)
    assert R.cfg["schedule"]["resume_reset"] is False and R.st["resume_resets"] == 1
    with pytest.raises(SystemExit, match="not after the state's 12"):  # max_steps 10 <= step 12
        m.resume_reset_finish(R, before)
    R.cfg["schedule"]["max_steps"] = 30
    m.resume_reset_finish(R, before)
    assert evs[-1]["kind"] == "resume_reset" and evs[-1]["T"] == 30.0 and R.st["schedule_resets"][-1]["at_step"] == 12


class _RunsRepo:
    """The runs repo as drop_hub_cooldown sees it: file_exists / delete_file over a set of paths (fail: what
    delete_file raises)."""

    def __init__(self, files=(), fail: Exception | None = None):
        self.files, self.fail, self.deleted = set(files), fail, []

    def file_exists(self, repo_id, filename, repo_type=None, **kw):
        return filename in self.files

    def delete_file(self, path_in_repo, repo_id=None, repo_type=None, commit_message=None, **kw):
        if self.fail is not None:
            raise self.fail
        self.files.discard(path_in_repo)
        self.deleted.append((repo_id, path_in_repo, repo_type))


def test_the_resume_reset_deletes_the_runs_repo_cooldown(tmp_path):
    """The log syncs upload the run dir without deleting, so after a reset consumed COOLDOWN the runs repo still holds
    runs/<id>/COOLDOWN, and a later resume of the continuation on a new host (resume-pull brings back runs/<id>/*)
    would put it back where the fresh early-stop state acts on it at once. The reset deletes that copy (after a log
    sync under way), and the owner's next COOLDOWN still acts. When the delete fails (or a sync is still running),
    the fresh state counts the file as acted on: the copy pulled back changes nothing, and the event warns. Without
    an output repo nothing is asked of the Hub."""
    m = load_script("04_distill")
    sets = ["schedule.clock=steps", "schedule.max_steps=40", "schedule.warmup_steps=2", "schedule.resume_reset=true"]

    def reset(name: str, repo: _RunsRepo | None, *, local: bool = True, synced: bool = True):
        run = tmp_path / name
        run.mkdir()
        R, evs, _ = _unit_run(m, run, sets)
        R.st["step"] = 5
        if local:
            R.st["early_stop"]["cooldown_file"] = {"at_step": 3}  # the COOLDOWN acted on before the reset
            (run / "COOLDOWN").touch()
        if repo is not None:
            R.uploader = SimpleNamespace(api=repo, repo="owner/runs")
            R.log.wait_sync = lambda timeout=None: synced
        return R, evs, m.resume_reset_start(R), run

    def new_host(R, run, repo):  # a later resume on a new host: the run dir gets back what the runs repo holds
        if f"runs/{run.name}/COOLDOWN" in repo.files:
            (run / "COOLDOWN").touch()
        R.st["step"] = 8
        return m.stop_requested(R)

    repo = _RunsRepo({"runs/ok/COOLDOWN", "runs/ok/events.jsonl"})
    R, evs, before, run = reset("ok", repo)
    assert before["cooldown_hub"] == "deleted" and repo.deleted == [("owner/runs", "runs/ok/COOLDOWN", "model")]
    assert repo.files == {"runs/ok/events.jsonl"} and "cooldown_warning" not in before
    assert not (run / "COOLDOWN").exists() and before["cooldown_file_consumed"].startswith("COOLDOWN.consumed-")
    assert new_host(R, run, repo) is False and not evs  # nothing came back: no cooldown
    (run / "COOLDOWN").touch()  # the owner's next request in the continuation acts
    assert m.stop_requested(R) is False and [e["reason"] for e in evs if e["kind"] == "early_stop"] == ["cooldown_file"]

    repo = _RunsRepo({"runs/nothing/events.jsonl"})  # the file never reached the Hub
    _, _, before, _ = reset("nothing", repo)
    assert before["cooldown_hub"] == "absent" and not repo.deleted

    for name, repo, synced, why in (
            ("fails", _RunsRepo({"runs/fails/COOLDOWN"}, fail=RuntimeError("503")), True, "failed: RuntimeError: 503"),
            ("syncing", _RunsRepo({"runs/syncing/COOLDOWN"}), False, "failed: a log sync")):
        R, evs, before, run = reset(name, repo, synced=synced)
        assert before["cooldown_hub"].startswith(why) and f"runs/{name}/COOLDOWN" in repo.files and not repo.deleted
        assert R.st["early_stop"]["cooldown_file"] == dict(at_step=5, ignored="consumed by resume_reset")
        assert new_host(R, run, repo) is False and not evs  # the stale copy came back and changes nothing
        m.resume_reset_finish(R, before)
        assert evs[-1]["kind"] == "resume_reset" and "ignores a COOLDOWN" in evs[-1]["cooldown_warning"]
        assert evs[-1]["cooldown_hub"] == before["cooldown_hub"]

    # a failed check on a run that never had a COOLDOWN: nothing to fall back on, a later COOLDOWN acts
    R, _, before, _ = reset("clean", _RunsRepo(), local=False, synced=False)
    assert before["cooldown_hub"].startswith("failed: ") and R.st["early_stop"]["cooldown_file"] is None
    assert "cooldown_warning" not in before
    # no output repo: no Hub call, no key
    R, _, before, _ = reset("offline", None)
    assert "cooldown_hub" not in before and R.st["early_stop"]["cooldown_file"] is None


def test_a_resume_that_switches_to_the_epoch_clock_needs_no_reset(env):
    """plan_epochs plans schedule.epochs when a resume switches a state without a plan (the steps clock) to the epoch
    clock, so build lets that --set schedule.epochs through without the reset flag; an epoch count that would change
    nothing (the steps clock never reads it) is still refused."""
    m = load_script("04_distill")
    path = write_config(env, "switch", {"schedule": {"max_steps": 6}, "ckpt": {"full_every_steps": 2}})
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "switch")
    st2 = str(run / "checkpoints" / "full_step_2")
    with pytest.raises(SystemExit, match="would change nothing"):
        m.main(["--config", path, "--resume", st2, "--set", "schedule.epochs=1"])
    assert m.main(["--config", path, "--resume", st2, "--set", "schedule.clock=epochs", "--set", "schedule.epochs=1",
                   "--set", "schedule.max_steps=null"]) == 0
    (sched,) = events(run, "schedule")
    s = summary(run)
    assert sched["epochs"] == 1 and sched["total_steps"] > 6 and s["steps"] == sched["total_steps"]
    assert s["resume_resets"] == 0 and s["status"] == "complete" and s["end_reason"] == "schedule"


# ----------------------------------------------------------------------------------------------------- CTC


@pytest.fixture(scope="module")
def ctc_env(tmp_path_factory):
    from fixtures_ctc import make_fake_parakeet_out, tiny_student_dir

    root = tmp_path_factory.mktemp("es_v2_ctc")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (40, "train"), "src_b": (32, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          rows_per_shard=8, dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL),
                          seed=21)
    sel = with_dev_split(make_fake_selection(fc, greedy_n=4, probe_n=5), DEV_STEMS)
    po = make_fake_parakeet_out(fc, seed=5)
    sdir, _ = tiny_student_dir(root / "student", seed=3, n_layers=2, ffn=48, name="tiny-p")
    base = {
        "family": "ctc", "parakeet_root": str(po.root),
        "student": str(sdir), "data_root": str(fc.data), "teacher_root": str(fc.teacher_out),
        "second_root": str(fc.second_out), "selection": str(sel), "cache_dir": str(root / "cache"),
        "runs_root": str(root / "runs"), "sources": ["src_a", "src_b"], "eval_sets": EVAL,
        "device": "cpu", "autocast": "none", "optim": {"lr": 3e-3}, "loss": {"l2sp_lambda": 0.0},
        "schedule": {"warmup_steps": 3, "cooldown_frac": 0.2, "clock": "steps", "max_steps": 20},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_min": None, "every_steps": None, "greedy_subset": 3, "batch_s": 20, "check_baselines": False},
        "ckpt": {"weights_every_min": None, "full_local_every_min": None, "keep_local": 2, "full_after_smoke": False,
                 "upload_full_at": []},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2, "capture_env": False},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
    }
    return dict(root=root, base=base, fc=fc, sel=sel)


def test_ctc_dev_store_and_early_stop(ctc_env):
    """Family ctc: the dev store is a frame store with its frame preflight (a `frame_preflight` event for store dev),
    the dev eval is one CTC pass per check whose greedy decode is free (eval.dev.greedy: cer in every record), dev_ce is
    the CTC loss per target token pooled over the sources (kitsune.ctc_eval.combined_loss_ctc's ctc), and a forced
    trigger starts the cooldown and ends the run with exit 0 (smoke check 10's shape)."""
    m = load_script("04_distill")
    cfg = merged(ctc_env["base"], {"run_name": "ctc-dev", "eval": {"dev": {"every_steps": 1, "per_source": 4,
                                                                            "greedy": True}},
                                   "early_stop": rule(patience=2, min_delta_abs=FLAT)})
    path = ctc_env["root"] / "ctc-dev.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert m.main(["--config", str(path)]) == 0
    run = one_run(ctc_env["root"], "ctc-dev")
    fp = {e["store"]: e for e in events(run, "frame_preflight")}
    assert set(fp) == {"train", "eval", "dev"} and fp["dev"]["ok"] and fp["dev"]["by_split"]["dev"]["rows"] == 8
    assert fp["dev"]["lenient_splits"] == ["train", "dev"] and fp["dev"]["dev_mismatch_frac"] == 0.0
    (ds,) = events(run, "dev_store")
    assert ds["n_in_train"] == 0 and ds["name"].startswith("ctc_dev_p4_s1234_")
    (es,) = events(run, "early_stop")
    assert (es["at_step"], es["metric"], es["action"]) == (3, "dev_ce", "cooldown")
    assert steps_of(run)["step"].tolist() == [1, 2, 3, 4]  # T = 3 + ceil(0.2 x 3)
    s = summary(run)
    assert s["status"] == "complete" and s["end_reason"] == "early_stop" and s["family"] == "ctc"
    for r in s["dev_history"]:
        pooled = sum(p["ce"] * p["n_tok"] for p in r["per_source"].values()) / r["n_tok"]
        assert r["dev_ce"] == pytest.approx(pooled) and 0 <= r["cer"] and 0 <= r["cer_vs_teacher"]
        assert r["dev_objective"] == pytest.approx(1.0 * r["dev_kl"] + 0.8 * r["dev_ce"])
    assert [r["step"] for r in s["dev_history"]] == [0, 1, 2, 3, 4]
    assert sorted(p.name for p in (run / "evals" / "step_2_dev").glob("*.parquet")) == [
        "greedy_src_a.parquet", "greedy_src_b.parquet", "tf_src_a.parquet", "tf_src_b.parquet"]
    assert scalar(run, "eval/dev/tf/src_a/ctc") and scalar(run, "eval/dev/greedy/all/cer_ref_corpus")


# ---------------------------------------------------------------------------------------------------- 4c


class _Rows:
    def __getitem__(self, idx):
        return list(idx)


def test_the_extended_memory_probe(monkeypatch):
    """memory.probe_extended adds the micro-batch with the most rows (both planners) and, for the frame planner, the
    most CTC targets and the largest CTC lattice; probe_shapes adds one synthetic micro-batch per shape (the nearest
    unused rows, longest first, within the micro-batch as the planner packs one), skipped when its targets exceed the
    micro-batch in use (after an OOM fallback, the smaller one)."""
    from kitsune import trainset

    m = load_script("04_distill")
    rng = np.random.default_rng(0)
    utts = [trainset.Utt(id=f"u{i}", source="s", duration=float(d), n_tok=int(n), audio_off=0, audio_len=0, tok_off=0)
            for i, (d, n) in enumerate(zip(rng.uniform(0.3, 4.9, 60), rng.integers(1, 40, 60)))]
    seen = {}
    monkeypatch.setattr(m, "fwd_bwd", lambda R, mb: (0.0, True))

    def probe(planner, **mem):
        cfg = m.load_config(None, [f"memory.{k}={json.dumps(v)}" for k, v in mem.items()]
                            + ["batch.micro_audio_s=10"])
        seen.clear()
        orig = m.fwd_bwd

        def fwd(R, mb):
            seen[len(seen)] = list(mb)
            return orig(R, mb)

        monkeypatch.setattr(m, "fwd_bwd", fwd)
        R = SimpleNamespace(device=torch.device("cpu"), ds=_Rows(), params=[], aux_ctc=None, cfg=cfg)
        rec = dict(peak_gb={})
        m.probe_passes(R, planner, rec)
        monkeypatch.setattr(m, "fwd_bwd", orig)
        return rec, list(seen.values())

    tok = trainset.StepPlanner(utts, step_audio_s=40.0, micro_audio_s=10.0, pool_micro=4, seed=0, max_dec_len=60)
    fr = trainset.StepPlanner(utts, step_audio_s=40.0, micro_audio_s=10.0, pool_micro=4, seed=0, max_dec_len=None)
    mbs = {p: [mb for step in p.epoch_plan(0) for mb in step] for p in (tok, fr)}
    for p, n_before in ((tok, 3), (fr, 2)):
        _, off = probe(p)
        assert len(off) == n_before  # off: the passes of before
        _, on = probe(p, probe_extended=True)
        most_rows = max(mbs[p], key=lambda mb: (len(mb), float(p.dur[mb].max())))
        assert on[n_before] == most_rows and len(on) == (4 if p is tok else 5)
    _, on = probe(fr, probe_extended=True)
    assert on[3] == max(mbs[fr], key=lambda mb: int(fr.n_tok[mb].sum()))
    assert on[4] == max(mbs[fr], key=lambda mb: len(mb) * float(fr.dur[mb].max()) * (2 * int(fr.n_tok[mb].max()) + 1))

    shapes = [{"name": "wide", "durations": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, {"name": "long", "durations": [4.9]}]
    rec, on = probe(fr, probe_shapes=shapes)
    idx = m.synth_micro_batch(fr, shapes[0]["durations"])
    assert len(idx) == len(set(idx)) == 6 and on[-2] == idx and len(on) == 4
    rest = sorted(range(len(utts)), key=lambda i: abs(utts[i].duration - 1.0))[:6]
    assert sorted(idx) == sorted(rest)  # the six rows nearest 1 s
    assert set(rec["shapes"]) == {"wide", "long"} and "skipped" not in rec["shapes"]["wide"]
    assert rec["shapes"]["wide"]["target_padded_s"] == 6.0 and rec["shapes"]["wide"]["padded_s"] <= 10.0
    small = trainset.StepPlanner(utts, step_audio_s=40.0, micro_audio_s=5.0, pool_micro=4, seed=0, max_dec_len=None)
    rec, on = probe(small, probe_shapes=shapes)  # after an OOM fallback to 5 s: wide (6 x ~1 s) no longer fits
    assert "OOM fallback" in rec["shapes"]["wide"]["skipped"] and "skipped" not in rec["shapes"]["long"]
    assert len(on) == 3
    # the token planner draws only the rows it can use (decoder input within max_dec_len)
    usable = {i for i in range(len(utts)) if tok.dec_len[i] <= tok.max_dec_len}
    assert set(m.synth_micro_batch(tok, [4.9] * 20)) <= usable

    # the nearest rows would overshoot the micro-batch (3 x 3.37 s > 10 s, targets 3 x 3.3 = 9.9 s): the probe stands
    # in the nearest rows that fit, as the planner packs a micro-batch; a shape whose targets exceed the configured
    # micro-batch is one training never meets (skipped); one row longer than it is probed (the planner gives such a row
    # a micro-batch of its own)
    near = [trainset.Utt(id=f"n{i}", source="s", duration=d, n_tok=5, audio_off=0, audio_len=0, tok_off=0)
            for i, d in enumerate([3.35, 3.36, 3.37, 1.0, 1.1, 1.2, 2.0, 2.1, 12.0])]
    p = trainset.StepPlanner(near, step_audio_s=40.0, micro_audio_s=10.0, pool_micro=4, seed=0, max_dec_len=None)
    assert float(p.dur[m.synth_micro_batch(p, [3.3] * 3)].max()) * 3 > 10.0  # nearest of all: over the micro
    shapes = [{"name": "tight", "durations": [3.3, 3.3, 3.3]}, {"name": "over", "durations": [3.0] * 4},
              {"name": "one", "durations": [11.0]}]
    rec, on = probe(p, probe_shapes=shapes)
    tight, over, one = (rec["shapes"][k] for k in ("tight", "over", "one"))
    assert tight["n"] == 3 and tight["padded_s"] <= 10.0 and tight["target_padded_s"] == 9.9 and "skipped" not in tight
    assert sorted(round(float(x), 2) for x in p.dur[on[-2]]) == [1.2, 2.0, 2.1]  # the nearest rows of <= 10 / 3 s
    assert over["n"] == 0 and over["skipped"] == "larger than micro_audio_s 10" and over["target_padded_s"] == 12.0
    assert one["n"] == 1 and one["padded_s"] == 12.0 and on[-1] == [8]
    assert len(on) == 2 + 2  # the frame planner's two passes, then tight and one


# ---------------------------------------------------------------------------------------------------- 4d


def _deadline_run(m, tmp_path, monkeypatch, *, left_s: float, step: int, sets=(), rate=1.0):
    R, evs, sc = _unit_run(m, tmp_path, ["schedule.clock=steps", "schedule.max_steps=1000", "schedule.warmup_steps=2",
                                         "schedule.cooldown_frac=0.2", "schedule.deadline_cooldown=true",
                                         "schedule.end_reserve_min=0", "smoke.enabled=false", *sets])
    R.st["step"] = step
    monkeypatch.setattr(m, "deadline_rate", lambda R: rate)
    monkeypatch.setenv("KITSUNE_DEADLINE", str(time.time() + left_s))
    return R, evs, sc


def test_the_deadline_cooldown_actions(tmp_path, monkeypatch):
    """fit_epochs_deadline at 1 s/step with no end reserve: T_new = t + the steps left. It fits: nothing. It does not,
    before t_c: "schedule" with t_c = T_new - ceil(0.2 T_new) (a full-share cooldown late), or "start" now when there is
    no room for that; a better projection clears a record whose cooldown has not begun; inside a cooldown "compress"
    keeps its t_c and is never relaxed; no step left ends the loop (T = t, `no_time_left`). progress / cooldown_start
    take the smaller T and the earlier t_c of the early stop's and 4d's records, and an early stop after a scheduled
    deadline cooldown still starts sooner."""
    m = load_script("04_distill")
    R, evs, sc = _deadline_run(m, tmp_path, monkeypatch, left_s=2000.5, step=100)
    assert m.fit_epochs_deadline(R) is None and R.st["deadline_cooldown"] is None
    assert [e["kind"] for e in evs] == ["deadline_check"] and evs[0]["fits"] is True  # the launch's first check
    assert sc[-1] == (100, {"sched/deadline_T": 2100.0}) and R.st["deadline_rate"] == 1.0
    evs.clear()

    monkeypatch.setenv("KITSUNE_DEADLINE", str(time.time() + 500.5))
    rec = m.fit_epochs_deadline(R)
    assert (rec["action"], rec["T"], rec["t_c"], rec["T_before"]) == ("schedule", 600.0, 480.0, 1000.0)
    assert R.progress() == (100.0, 600.0) and R.cooldown_start(600.0) == 480.0
    assert evs[-1]["kind"] == "deadline_cooldown" and R.st["deadline_cooldown"]["action"] == "schedule"
    assert m.fit_epochs_deadline(R) is None and len(evs) == 1  # the same projection again: no new record
    R.st["step"] = 110
    monkeypatch.setenv("KITSUNE_DEADLINE", str(time.time() + 2000.5))  # a better projection before t_c: cleared
    rec = m.fit_epochs_deadline(R)
    assert rec["action"] == "clear" and R.st["deadline_cooldown"] is None and R.progress() == (110.0, 1000.0)

    monkeypatch.setenv("KITSUNE_DEADLINE", str(time.time() + 20.5))
    R.st["step"] = 100
    rec = m.fit_epochs_deadline(R)  # T_new 120: 120 - 24 < 100, no room for a full share
    assert (rec["action"], rec["t_c"], rec["T"]) == ("start", 100.0, 120.0)

    R, evs, _ = _deadline_run(m, tmp_path, monkeypatch, left_s=100.5, step=850)  # inside the scheduled cooldown (800)
    rec = m.fit_epochs_deadline(R)
    assert (rec["action"], rec["t_c"], rec["T"]) == ("compress", 800.0, 950.0) and R.progress() == (850.0, 950.0)
    R.st["step"] = 860
    monkeypatch.setenv("KITSUNE_DEADLINE", str(time.time() + 500.5))
    assert m.fit_epochs_deadline(R) is None and R.progress()[1] == 950.0  # begun: never relaxed
    R.st["pre_cooldown_done"] = True
    monkeypatch.setenv("KITSUNE_DEADLINE", str(time.time() - 5))
    rec = m.fit_epochs_deadline(R)
    assert rec["action"] == "compress" and rec["T"] == 860.0 and R.progress() == (860.0, 860.0)
    assert evs[-1]["kind"] == "no_time_left" and evs[-1]["T"] == 860.0

    R, evs, _ = _deadline_run(m, tmp_path, monkeypatch, left_s=-5, step=10)  # no time at all, not in a cooldown
    rec = m.fit_epochs_deadline(R)
    assert (rec["action"], rec["t_c"], rec["T"]) == ("start", 10.0, 10.0) and evs[-1]["kind"] == "no_time_left"
    assert m.end_reason(R) == "deadline"

    # both records: the smaller T, the earlier t_c; an early stop after a scheduled deadline cooldown starts sooner
    R, evs, _ = _deadline_run(m, tmp_path, monkeypatch, left_s=500.5, step=100)
    m.fit_epochs_deadline(R)
    R.st["step"] = 300
    assert m.early_stop_trigger(R, "patience", "cooldown") is False
    assert R.st["early_stop"]["cooldown"] == dict(t_c=300.0, T=360.0, clock="steps", at_step=300)
    assert R.progress() == (300.0, 360.0) and R.cooldown_start(360.0) == 300.0
    assert m.end_reason(R) == "early_stop"

    # inactive: no deadline, the wall clock, or the smoke phase not done
    R, evs, _ = _deadline_run(m, tmp_path, monkeypatch, left_s=10, step=100)
    monkeypatch.delenv("KITSUNE_DEADLINE")
    assert m.fit_epochs_deadline(R) is None and not evs
    R, evs, _ = _deadline_run(m, tmp_path, monkeypatch, left_s=10, step=100, sets=["smoke.enabled=true"])
    assert m.fit_epochs_deadline(R) is None and not evs
    R.st["smoke_done"] = True
    assert m.fit_epochs_deadline(R)["action"] == "start"


def test_the_deadline_check_at_start_the_rate_and_the_evals_ahead(tmp_path, monkeypatch):
    m = load_script("04_distill")
    rate = m.deadline_rate  # _deadline_run replaces it
    R, evs, _ = _deadline_run(m, tmp_path, monkeypatch, left_s=2000.5, step=100)
    monkeypatch.setattr(m, "deadline_rate", lambda R: None)
    assert m.fit_epochs_deadline(R, at_start=True) is None
    assert evs[-1]["kind"] == "deadline_check" and evs[-1]["sec_per_step"] is None and evs[-1]["fits"] is None
    monkeypatch.setattr(m, "deadline_rate", lambda R: 1.0)
    m.fit_epochs_deadline(R, at_start=True)
    ev = evs[-1]
    assert ev["kind"] == "deadline_check" and ev["fits"] is True and ev["T"] == 1000.0 and ev["projected_end_utc"]

    # the rate: this launch's samples, over the newest window of >= deadline_window_steps (here 20) steps
    R, _, _ = _unit_run(m, tmp_path, ["schedule.clock=steps", "schedule.max_steps=100", "schedule.warmup_steps=2",
                                      "schedule.deadline_window_steps=20"])
    got = []
    for step, clock in ((0, 0.0), (10, 20.0), (20, 40.0), (30, 90.0), (40, 100.0)):
        R.st.update(step=step, train_s=clock)
        R.deadline_marks.append((step, clock))  # fit_epochs_deadline's sample of each check
        got.append(rate(R))
    assert got == [None, None, 2.0, 3.5, 3.0] and R.deadline_marks[0][0] == 20
    R2, _, _ = _unit_run(m, tmp_path, ["schedule.clock=steps", "schedule.max_steps=100", "schedule.warmup_steps=2"])
    R2.st["deadline_rate"] = 0.7  # an earlier launch's, until this one has its window
    assert rate(R2) == 0.7

    # the in-loop complete evals at epoch ends between two steps (the end phase's final eval not counted)
    R, _, _ = _unit_run(m, tmp_path, ["schedule.clock=epochs", "schedule.epochs=4", "eval.full_every_epochs=1"])
    R.st["total_steps"] = 100
    assert [m._evals_ahead(R, a, b) for a, b in ((10, 100), (10, 60), (25, 50), (0, 25), (0, 26))] == [3, 2, 0, 0, 1]
    R.cfg["eval"]["full_every_epochs"] = None
    assert m._evals_ahead(R, 0, 100) == 0


def test_a_run_that_ends_by_the_deadline(env, monkeypatch):
    """The steps clock, 20 steps planned, 4d on (checked every 2 steps) with a fixed 1 s/step and a deadline 12 steps
    away (the end reserve and the final-eval estimate 0): at loop start the cooldown is scheduled at t_c 6 = 12 -
    ceil(0.5 x 12), the pre_cooldown state is saved at step 6, the LR anneals to step 12, the run ends there with exit
    0, end_reason deadline."""
    m = load_script("04_distill")
    path = write_config(env, "deadline", {"schedule": {"deadline_cooldown": True, "deadline_check_steps": 2,
                                                       "end_reserve_min": 0}})
    holder = {}
    orig_train = m.train

    def train(R, state):
        holder["R"] = R
        return orig_train(R, state)

    monkeypatch.setattr(m, "train", train)
    monkeypatch.setattr(m, "deadline_rate", lambda R: 1.0)
    monkeypatch.setattr(m, "final_eval_estimate", lambda R: 0.0)
    # a deadline fixed at step 12 in the run's own time: 12 - step seconds away at 1 s/step
    monkeypatch.setattr(m, "deadline_unix", lambda: time.time() + 12.5 - holder["R"].st["step"])
    assert m.main(["--config", path]) == 0
    run = one_run(env["root"], "deadline")
    st = steps_of(run)
    assert st["step"].tolist() == list(range(1, 13))
    (dc,) = events(run, "deadline_check")
    assert dc["fits"] is False and dc["T"] == 20.0
    (d,) = events(run, "deadline_cooldown")
    assert (d["action"], d["t_c"], d["T"], d["at_step"]) == ("schedule", 6.0, 12.0, 0)
    assert any(e["name"] == "full_step_6" and e["reason"] == "pre_cooldown" for e in events(run, "checkpoint"))
    for step, lr, phase in zip(st["step"], st["opt/lr"], st["sched/phase"]):
        want = m.wsd_lr(PEAK, int(step), float(step - 1), 12.0, 2, 0.5, t_c=6.0)
        assert lr == pytest.approx(want[0]) and phase == want[1], step
    s = summary(run)
    assert s["status"] == "complete" and s["steps"] == 12 and s["end_reason"] == "deadline"
    assert s["deadline_cooldown"]["T"] == 12.0 and s["stopped_early"] is None
    assert sorted(scalar(run, "sched/deadline_T")) == [0, 2, 4, 6, 8, 10, 12]
