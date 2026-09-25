"""scripts/05_evaluate.py: a checkpoint evaluated exactly like the trainer's complete (final) eval.

A tiny trainer run on CPU with lr 0 ends with the trainer's own complete eval (evals/step_1/) of weights that are
exactly the bf16 values its checkpoint step_1/ holds (the student init is saved in bf16 and no update moves them).
05_evaluate on that checkpoint must give the same per-utterance tables, summary.json (timings aside) and verdict.json,
to the bit: evaluated in chunks, from a config with relative paths under --root or from the run's config.json, after a
crash in the middle of its pass (forced or not), after a stop while the pass's outputs were being written, and after a
host crash tore a file of its resume state. Also: sets already done are skipped, --force evaluates them again, an --out
of other eval settings is refused, no verdict without the probe the config has, the resume state is fsynced, and the
thermal guard runs before every batch, waits for the GPU to cool and stops when it can no longer read it.

CPU only (the laptop GPU runs other jobs), a tiny random student and a synthetic corpus in the real on-disk formats."""
import json
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

GATE = ["eval_jsut", "eval_cv8", "eval_reazon"]
SETS = [*GATE, "eval_emilia"]  # eval_emilia: a monitor-only hold-out, as in the viability run
BATCH_S = 6.0
TIMING = {"wall_s", "rtf"}  # measured, not computed: they differ between any two evals


def events(out: Path, kind: str | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def untimed(d):
    if isinstance(d, dict):
        return {k: untimed(v) for k, v in d.items() if k not in TIMING}
    if isinstance(d, list):
        return [untimed(v) for v in d]
    return d


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Corpus, selection (4 greedy-subset rows per eval set, 4 probe rows per train source), a tiny student saved as 03
    saves one, and the trainer's 1-step lr-0 run whose final eval decodes the complete eval sets."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("evalscript")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (30, "train"), "src_b": (20, "train"),
                                                    **{s: (6, "eval") for s in SETS}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(SETS), seed=7)
    sel = make_fake_selection(fc, greedy_n=4, probe_n=4)

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
    S.save_student(student, root / "student", proc, dict(format=1, stage="complete", spec=dict(
        enc_layers=[0, 1], ffn_dim=128, dec_layers=[0], tie_head=True)))

    rel = dict(student="student", data_root="corpus/data", teacher_root="corpus/teacher_out",
               second_root="corpus/second_out", selection=sel.relative_to(root).as_posix(), cache_dir="cache",
               runs_root="runs")
    common = {
        "run_name": "tiny-eval", "sources": ["src_a", "src_b"], "eval_sets": SETS, "device": "cpu", "autocast": "none",
        "optim": {"lr": 0.0},  # the weights stay the student init's bf16 values: the checkpoint holds them exactly
        "schedule": {"warmup_steps": 1, "cooldown_frac": 0.3, "clock": "steps", "max_steps": 1},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        # greedy_subset 3 of the 4 in_greedy_subset rows: the seeded draw of greedy_subset_ids; full_every_epochs:
        # epoch mode (summary.json's epoch)
        "eval": {"full_every_epochs": 1, "greedy_subset": 3, "batch_s": BATCH_S, "check_baselines": False,
                 "probe_greedy_audio_s": 3},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 1000, "keep_local": 2, "full_after_smoke": False},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2, "capture_env": False},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
    }
    (root / "trainer.json").write_text(json.dumps(dict(common, **{k: str(root / v) for k, v in rel.items()}),
                                                  indent=1), encoding="utf-8")
    (root / "relative.json").write_text(json.dumps(dict(common, **rel), indent=1), encoding="utf-8")
    m = load_script("04_distill")
    assert m.main(["--config", str(root / "trainer.json")]) == 0
    (run,) = list((root / "runs").glob("tiny-eval-*"))
    assert (run / "evals" / "step_1" / "verdict.json").exists()
    assert load(run / "evals" / "step_1" / "summary.json")["final"] is True
    return dict(root=root, run=run, ckpt=run / "checkpoints" / "step_1", relative=root / "relative.json")


def assert_the_trainers(env, out: Path):
    """--out holds the trainer's final eval of the same weights: the same tables, summary.json and verdict.json."""
    ref = env["run"] / "evals" / "step_1"
    names = sorted(p.name for p in ref.glob("*.parquet"))
    assert names == sorted(p.name for p in out.glob("*.parquet"))
    assert set(names) == ({f"tf_{s}.parquet" for s in SETS} | {f"greedy_{s}.parquet" for s in SETS}
                          | {"probe.parquet", "probe_greedy.parquet"})
    for name in names:
        a, b = pd.read_parquet(ref / name), pd.read_parquet(out / name)
        if "hyp_ids" in a:
            assert [list(x) for x in a.pop("hyp_ids")] == [list(x) for x in b.pop("hyp_ids")], name
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=name)
    s_ref, s_out = load(ref / "summary.json"), load(out / "summary.json")
    assert list(s_out) == list(s_ref)  # the trainer's keys, in its order
    assert s_out["train_s"] == pytest.approx(s_ref["train_s"], abs=0.06)  # the checkpoint keeps it to 0.1 s
    assert untimed(dict(s_out, train_s=None)) == untimed(dict(s_ref, train_s=None))
    assert s_out["combined_loss"]["val_full"]["sets"] == sorted(GATE)  # the monitor-only set never pools into it
    assert s_out["headline_scope"]["val_cer_utts"] == 3 * 6 and s_out["greedy_full"]["n_utts"] == 4 * 6
    assert load(out / "verdict.json") == load(ref / "verdict.json")


def test_the_trainers_complete_eval_to_the_bit(env, tmp_path, monkeypatch):
    """Relative config paths under --root, the pass cut into several chunks, the thermal guard in front of every
    batch: the trainer's tables, summary and verdict."""
    from kitsune import trainset

    m05 = load_script("05_evaluate")

    class CountingGuard:  # stands in for the GPU's: counts the checks, never pauses
        max_seen, checks = None, 0

        def check(self):
            self.checks += 1

        def record(self):
            return dict(checks=self.checks)

    guard = CountingGuard()
    monkeypatch.setattr(m05, "make_guard", lambda device, args, log: guard)
    real_load, caps = m05.load_trainer, []

    def load_trainer():  # records the VRAM caps asked for (a no-op on CPU)
        D = real_load()
        cap = D.cap_vram
        D.cap_vram = lambda R, max_frac=None: caps.append(max_frac) or cap(R, max_frac=max_frac)
        return D

    monkeypatch.setattr(m05, "load_trainer", load_trainer)
    out = tmp_path / "out"
    assert m05.main(["--root", str(env["root"]), "--config", str(env["relative"]), "--ckpt", str(env["ckpt"]),
                     "--out", str(out), "--probe", "--chunk-s", "5"]) == 0
    assert_the_trainers(env, out)
    assert caps == [None]  # the trainer's cap (Windows CUDA) without --vram-frac too, not only with it
    (pas,) = events(out, "pass")
    assert pas["sets"] == SETS and pas["chunks"] >= 3 and len(events(out, "chunk")) == pas["chunks"]
    assert not (out / ".work").exists()  # the finished pass's chunks are gone
    # before every batch of every pass: the pooled sets teacher-forced and greedy, the probe and its greedy part
    probe = trainset.load_stores(next((out / "cache").glob("probe_*")))
    pg = next(e for e in events(out, "subset") if e["split"] == "probe_greedy")["ids"]
    pos = {u.id: i for i, u in enumerate(probe.utts)}
    n_probe = len(trainset.eval_batches(probe.utts, BATCH_S)) + len(
        trainset.eval_batches(probe.utts, BATCH_S, [pos[i] for i in pg]))
    assert guard.checks == 2 * pas["batches"] + n_probe
    (inv,) = load(out / "evaluator.json")["invocations"]
    assert inv["status"] == "complete" and inv["step"] == 1 and inv["evaluated"] == SETS and inv["thermal"]


def test_a_crash_mid_pass_resumes_to_the_same_results(env, tmp_path, monkeypatch):
    """A GPU fault in the third chunk: the two chunks before it are kept, the same command evaluates only the rest,
    and the result is still the trainer's. A third run finds everything done and evaluates nothing. The run's own
    config.json works as --config."""
    from kitsune import evaluate as ev

    m05 = load_script("05_evaluate")
    out = tmp_path / "out"
    argv = ["--config", str(env["run"] / "config.json"), "--ckpt", str(env["ckpt"]), "--out", str(out), "--probe",
            "--chunk-s", "5"]
    real_greedy, real_tf, calls = ev.greedy_eval, ev.teacher_forced_records, []

    def faulty(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("CUDA error: an illegal instruction was encountered")
        return real_greedy(*a, **k)

    with monkeypatch.context() as mp:
        mp.setattr(ev, "greedy_eval", faulty)
        with pytest.raises(RuntimeError, match="illegal instruction"):
            m05.main(argv)
    assert not (out / "summary.json").exists() and not list(out.glob("*.parquet"))
    assert len(list((out / ".work").rglob("chunk_*.json"))) == 2
    tf_calls = []
    monkeypatch.setattr(ev, "teacher_forced_records", lambda *a, **k: tf_calls.append(1) or real_tf(*a, **k))
    assert m05.main(argv) == 0
    pas = events(out, "pass")[-1]
    assert pas["chunks_done"] == 2 and len(tf_calls) == pas["chunks"] - 2 + 1  # the rest of the pass, the probe
    assert_the_trainers(env, out)
    tf_calls.clear()
    assert m05.main(argv) == 0 and not tf_calls
    todo = events(out, "todo")[-1]
    assert todo["sets"] == [] and todo["skipped_done"] == SETS and todo["probe"] is False
    assert_the_trainers(env, out)
    assert [i["status"] for i in load(out / "evaluator.json")["invocations"]] == ["failed", "complete", "complete"]


def test_verdict_v2_is_the_trainers_too(env, tmp_path):
    """Under eval.verdict_version 2, with a reference model (eval.reference), the evaluator's verdict.json is still
    the trainer's: its record of the checkpoint carries the LR phase of the checkpoint's last step (from the
    checkpoint's trained block) and the complete sets' numbers, as the trainer's final record does, and it reads the
    same reference file. The same run under v1 (the fixture's) keeps the v1 verdict."""
    from kitsune import student as S

    m, m05 = load_script("04_distill"), load_script("05_evaluate")
    cfg = json.loads((env["root"] / "trainer.json").read_text(encoding="utf-8"))
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps(dict(name="ref", cer=dict(eval_jsut=0.3, eval_cv8=0.4))), encoding="utf-8")
    cfg.update(run_name="tiny-eval-v2", eval=dict(cfg["eval"], verdict_version=2, reference=dict(path=str(ref))))
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    assert m.main(["--config", str(path)]) == 0
    (run,) = list((env["root"] / "runs").glob("tiny-eval-v2-*"))
    ckpt = run / "checkpoints" / "step_1"
    assert S.load_meta(ckpt)["trained"]["lr_phase"] == "stable"  # warmup_steps 1: step 1 is at the peak LR
    got = load(run / "evals" / "step_1" / "verdict.json")
    assert got["version"] == 2 and "version" not in load(env["run"] / "evals" / "step_1" / "verdict.json")
    assert summary_history(run)[-1]["lr_phase"] == "stable" and "greedy_full" in summary_history(run)[-1]
    out = tmp_path / "out"
    assert m05.main(["--config", str(path), "--ckpt", str(ckpt), "--out", str(out), "--probe"]) == 0
    assert load(out / "verdict.json") == got and got["reference"]["not_compared"] == ["eval_reazon"]
    assert events(out, "reference")[-1]["cer"] == dict(eval_jsut=0.3, eval_cv8=0.4)


def summary_history(run: Path) -> list[dict]:
    return load(run / "summary.json")["history"]


def test_backfill_history_completes_an_older_runs_records(tmp_path):
    """A history written before the trainer recorded the LR phase and the complete sets' numbers gets them from the
    run's own files: the lr_phase events of events.jsonl, and each complete eval's evals/step_<N>/summary.json
    greedy_full (never a subset eval's). What a record has, it keeps; without the files nothing changes."""
    m05 = load_script("05_evaluate")
    run = tmp_path / "run"
    lines = [dict(kind="lr_phase", at_step=1, phase="warmup"), dict(kind="eval", at_step=4),
             dict(kind="lr_phase", at_step=3, phase="stable"), dict(kind="lr_phase", at_step=8, phase="cooldown")]
    (run / "evals").mkdir(parents=True)
    (run / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    full = dict(sets={"eval_jsut": dict(cer_ref_corpus=0.1, teacher_cer_ref_corpus=0.08, trunc_rate=0.0, n=9,
                                        ref_edits=3, ref_chars=30)})
    for step, gf in ((0, None), (5, full), (6, None), (9, full)):
        (run / "evals" / f"step_{step}").mkdir()
        (run / "evals" / f"step_{step}" / "summary.json").write_text(json.dumps(dict(greedy_full=gf)),
                                                                     encoding="utf-8")
    hist = [dict(step=0), dict(step=5), dict(step=6), dict(step=9, lr_phase="own", greedy_full={"x": 1})]
    got = m05.backfill_history(hist, str(run / "summary.json"))
    assert [r.get("lr_phase") for r in got] == [None, "stable", "stable", "own"]
    brief = dict(eval_jsut=dict(cer_ref_corpus=0.1, teacher_cer_ref_corpus=0.08, trunc_rate=0.0, n=9))
    assert [r.get("greedy_full") for r in got] == [None, brief, None, {"x": 1}]
    assert hist[1] == dict(step=5)  # the input records are not changed
    assert m05.backfill_history(hist, None) == hist
    assert m05.backfill_history(hist, str(tmp_path / "elsewhere" / "summary.json")) == hist


def fault_at(fn, n: int):
    """fn, raising a GPU fault on its n-th call."""
    calls = []

    def faulty(*a, **k):
        calls.append(1)
        if len(calls) == n:
            raise RuntimeError("CUDA error: an illegal instruction was encountered")
        return fn(*a, **k)

    return faulty


def tear(p: Path):
    """What an unflushed rename can leave after a host crash on NTFS: the file's size, all zeros."""
    p.write_bytes(b"\x00" * p.stat().st_size)


def test_a_forced_run_that_stops_continues_with_the_same_command(env, tmp_path, monkeypatch):
    """--force deletes the named sets' results once: after a GPU fault in its third chunk the same forced command keeps
    the two chunks done since and continues, to the trainer's results. The next forced command starts over."""
    from kitsune import evaluate as ev

    m05 = load_script("05_evaluate")
    out = tmp_path / "out"
    argv = ["--config", str(env["run"] / "config.json"), "--ckpt", str(env["ckpt"]), "--out", str(out), "--probe",
            "--chunk-s", "5"]
    assert m05.main(argv) == 0
    forced = argv + ["--force"]
    with monkeypatch.context() as mp:
        mp.setattr(ev, "greedy_eval", fault_at(ev.greedy_eval, 3))
        with pytest.raises(RuntimeError, match="illegal instruction"):
            m05.main(forced)
    assert not list(out.glob("*.parquet")) and (out / ".parts" / "force.json").exists()
    assert len(list((out / ".work").rglob("chunk_*.json"))) == 2
    assert m05.main(forced) == 0
    assert events(out, "force_continue") and events(out, "pass")[-1]["chunks_done"] == 2
    assert events(out, "force_done") and not (out / ".parts" / "force.json").exists()
    assert_the_trainers(env, out)
    assert m05.main(forced) == 0  # a new forced command: from scratch
    assert len(events(out, "force")) == 2 and events(out, "pass")[-1]["chunks_done"] == 0
    assert_the_trainers(env, out)


def test_a_stop_while_the_outputs_are_written_is_finished_from_the_chunks(env, tmp_path, monkeypatch):
    """A stop while the pass's per-set outputs are written (disk full after the first set and part of the second):
    the next run writes the rest from the pass's chunks instead of evaluating the other sets in a pass of their own
    (other batches), so the results are still the trainer's."""
    m05 = load_script("05_evaluate")
    out = tmp_path / "out"
    argv = ["--config", str(env["run"] / "config.json"), "--ckpt", str(env["ckpt"]), "--out", str(out), "--probe",
            "--chunk-s", "5"]
    real_table, greedy = m05._table, []

    def table(df, path):
        real_table(df, path)
        if path.name.startswith("greedy_"):
            greedy.append(path.name)
            if len(greedy) == 2:
                raise OSError(28, "No space left on device")

    with monkeypatch.context() as mp:
        mp.setattr(m05, "_table", table)
        with pytest.raises(OSError, match="No space"):
            m05.main(argv)
    assert [s for s in SETS if m05.set_done(out, s)] == SETS[:1] and (out / f"greedy_{SETS[1]}.parquet").exists()
    n_pass = len(events(out, "pass"))
    assert m05.main(argv) == 0
    (fin,) = events(out, "pass_finish")
    assert fin["sets"] == SETS and len(events(out, "pass")) == n_pass and not (out / ".work").exists()
    assert_the_trainers(env, out)


def test_a_torn_file_of_the_resume_state_is_evaluated_again(env, tmp_path, monkeypatch):
    """A host crash can leave a renamed file zero-filled: a chunk, a set or the probe whose record or table cannot be
    read counts as not done and is evaluated again, instead of failing every later run."""
    from kitsune import evaluate as ev

    m05 = load_script("05_evaluate")
    out = tmp_path / "out"
    argv = ["--config", str(env["run"] / "config.json"), "--ckpt", str(env["ckpt"]), "--out", str(out), "--probe",
            "--chunk-s", "5"]
    with monkeypatch.context() as mp:
        mp.setattr(ev, "greedy_eval", fault_at(ev.greedy_eval, 4))
        with pytest.raises(RuntimeError, match="illegal instruction"):
            m05.main(argv)
    (work,) = (out / ".work").iterdir()
    tear(work / "chunk_00001.json")
    tear(work / "chunk_00002.greedy.parquet")
    assert m05.main(argv) == 0
    assert events(out, "pass")[-1]["chunks_done"] == 1  # chunk 0 kept, chunks 1 and 2 evaluated again
    assert_the_trainers(env, out)
    tear(out / ".parts" / "eval_emilia.json")
    tear(out / "greedy_eval_cv8.parquet")
    tear(out / ".parts" / "probe.json")
    assert m05.main(argv) == 0
    todo = events(out, "todo")[-1]
    assert todo["sets"] == ["eval_cv8", "eval_emilia"] and todo["probe"] is True
    assert sorted(load(out / "summary.json")["greedy_full"]["sets"]) == sorted(SETS)
    assert (out / "verdict.json").exists()


def test_the_resume_state_is_fsynced_before_it_is_renamed(tmp_path, monkeypatch):
    """Every file 05 writes goes to disk before it takes its name; a torn one reads as missing."""
    m05 = load_script("05_evaluate")
    synced = []

    def fsync(p):
        p = Path(p)
        assert p.name.endswith(".tmp") and p.stat().st_size and not p.with_name(p.name[:-4]).exists()
        synced.append(p.name)

    monkeypatch.setattr(m05, "fsync_path", fsync)
    m05._write_json(tmp_path / "part.json", dict(kl=float("nan")))
    m05._table(pd.DataFrame(dict(id=["u1", "u2"], kl=[0.5, 0.25])), tmp_path / "t.parquet")
    m05._write_output_json(tmp_path / "summary.json", dict(kl=float("inf")))
    assert synced == ["part.json.tmp", "t.parquet.tmp", "summary.json.tmp"]
    assert np.isnan(m05._load_json(tmp_path / "part.json")["kl"]) and load(tmp_path / "summary.json") == {"kl": None}
    assert m05._rows(tmp_path / "t.parquet") == 2
    tear(tmp_path / "part.json")
    tear(tmp_path / "t.parquet")
    assert m05._load_json(tmp_path / "part.json") is None and m05._load_json(tmp_path / "none.json") is None
    assert m05._rows(tmp_path / "t.parquet") is None and m05._rows(tmp_path / "none.parquet") is None


def test_sets_done_are_skipped_forced_again_and_an_out_keeps_one_setting(env, tmp_path):
    """--sets evaluates those only (no verdict without the three gate sets); the next run adds the rest in a pass of
    their own and the summary covers all of them, but the verdict waits for the probe the config has (--probe);
    --force evaluates a set again; another batch size, other weights or a set the config does not have are refused."""
    m05 = load_script("05_evaluate")
    out = tmp_path / "out"
    base = ["--root", str(env["root"]), "--config", str(env["relative"]), "--ckpt", str(env["ckpt"]), "--out", str(out)]
    assert m05.main(base + ["--sets", "eval_jsut", "eval_emilia"]) == 0
    s = load(out / "summary.json")
    assert sorted(s["greedy_full"]["sets"]) == ["eval_emilia", "eval_jsut"] and s["probe"] is None
    assert s["combined_loss"]["val_full"]["sets"] == ["eval_jsut"] and "train_loss" not in s["headline"]
    assert not (out / "verdict.json").exists() and events(out, "verdict_skipped")[-1]["missing_gate_sets"] == GATE[1:]
    assert m05.main(base) == 0
    todo = events(out, "todo")[-1]
    assert todo["sets"] == ["eval_cv8", "eval_reazon"] and todo["skipped_done"] == ["eval_jsut", "eval_emilia"]
    s = load(out / "summary.json")
    assert sorted(s["greedy_full"]["sets"]) == sorted(SETS) and s["probe"] is None
    # the trainer's record of this eval holds the probe's KL, which the verdict's trends read: no verdict without it
    assert not (out / "verdict.json").exists() and events(out, "verdict_skipped")[-1]["probe_missing"] is True
    assert m05.main(base + ["--probe"]) == 0
    todo = events(out, "todo")[-1]
    assert todo["sets"] == [] and todo["probe"] is True and (out / "verdict.json").exists()
    # two passes batch their own sets together: the same rows and numbers within float noise of the trainer's pass
    ref = env["run"] / "evals" / "step_1"
    for name in [f"tf_{x}.parquet" for x in SETS]:
        a, b = pd.read_parquet(ref / name), pd.read_parquet(out / name)
        assert sorted(a["id"]) == sorted(b["id"])
        a, b = a.set_index("id").sort_index(), b.set_index("id").sort_index()
        assert np.allclose(a["kl"], b["kl"], rtol=1e-4, atol=1e-6) and (a["n_tok"] == b["n_tok"]).all()
    for name in [f"greedy_{x}.parquet" for x in SETS]:
        assert sorted(pd.read_parquet(ref / name)["id"]) == sorted(pd.read_parquet(out / name)["id"])
    assert m05.main(base + ["--sets", "eval_emilia", "--force"]) == 0
    assert events(out, "todo")[-1]["sets"] == ["eval_emilia"]
    with pytest.raises(SystemExit, match="batch_s"):
        m05.main(base + ["--batch-s", "4"])
    with pytest.raises(SystemExit, match="use a new --out"):  # the student init: step 0 (and maybe other bytes)
        m05.main(base[:-3] + [str(env["root"] / "student"), "--out", str(out)])
    with pytest.raises(SystemExit, match="not among"):
        m05.main(base + ["--sets", "galgame"])


def test_the_thermal_guard_waits_for_the_gpu_to_cool(monkeypatch):
    """At --max-temp or above the eval waits, polling, until the GPU is down to --resume-temp; a failed read neither
    stops nor unblocks it, but BLIND_READS of them in a row stop the eval, in a pause or between batches. nvidia-smi is
    read at most every NVSMI_EVERY_S. The guard sits in front of the featuriser, which the eval calls once per batch."""
    m05 = load_script("05_evaluate")
    temps = iter([70.0, 81.0, 79.0, None, 72.0, 70.0, 75.0])
    now, slept, evs = [0.0], [], []

    def read():
        t = next(temps)
        if t is None:
            raise RuntimeError("NVML_ERROR_UNKNOWN")
        return t

    def sleep(s):
        slept.append(s)
        now[0] += s

    log = SimpleNamespace(event=lambda kind, **kw: evs.append((kind, kw)))
    g = m05.ThermalGuard(read, 80, 70, 10, log, sleep=sleep, clock=lambda: now[0])
    g.check()
    assert slept == [] and evs == []
    g.check()  # 81: waits through 79, a failed read and 72 until 70
    assert slept == [10] * 4
    assert [k for k, _ in evs] == ["thermal_pause", "thermal_read_error", "thermal_resume"]
    assert evs[0][1]["temp_c"] == 81.0 and evs[-1][1] == dict(temp_c=70.0, waited_s=40.0, pauses=1)
    g.check()  # 75: below --max-temp, on it goes
    rec = g.record()
    assert rec["max_seen_c"] == 81.0 and rec["paused_s"] == 40.0 and rec["checks"] == 3 and rec["read_errors"] == 1
    assert [(p["temp_c"], p["resumed_c"]) for p in rec["pauses"]] == [(81.0, 70.0)]

    reads = []
    paced = m05.ThermalGuard(lambda: reads.append(1) or 50.0, 80, 70, 10, log, min_interval_s=m05.NVSMI_EVERY_S,
                             clock=lambda: now[0])
    for _ in range(3):
        paced.check()
    now[0] += m05.NVSMI_EVERY_S
    paced.check()
    assert len(reads) == 2 and paced.checks == 4

    blind = iter([85.0])

    def lost():
        t = next(blind, None)
        if t is None:
            raise OSError("nvidia-smi: GPU is lost")
        return t

    with pytest.raises(RuntimeError, match="no GPU temperature"):
        m05.ThermalGuard(lost, 80, 70, 1, log, sleep=sleep, clock=lambda: now[0]).check()

    # a reader lost between batches: BLIND_READS failed reads in a row stop the eval too (a good read resets the count)
    n = m05.BLIND_READS
    flaky_temps = iter([None] * (n - 1) + [60.0] + [None] * n)

    def flaky():
        t = next(flaky_temps)
        if t is None:
            raise RuntimeError("NVML_ERROR_GPU_IS_LOST")
        return t

    unread = m05.ThermalGuard(flaky, 80, 70, 10, log, sleep=sleep, clock=lambda: now[0])
    for _ in range(2 * n - 1):
        unread.check()
    with pytest.raises(RuntimeError, match=f"no GPU temperature for {n} reads in a row between batches"):
        unread.check()
    assert unread.read_errors == 2 * n - 1 and unread.pauses == []
    with pytest.raises(ValueError):
        m05.ThermalGuard(read, 70, 70, 10, log)

    order = []
    guarded = m05.Guarded(lambda w, n: order.append("features") or (w, n), SimpleNamespace(
        check=lambda: order.append("check")))
    assert guarded("wave", "lengths") == ("wave", "lengths") and order == ["check", "features"]

    args = SimpleNamespace(max_temp=80.0, resume_temp=70.0, poll_s=10.0)
    assert m05.make_guard(torch.device("cpu"), args, log) is None
    assert m05.make_guard(torch.device("cuda", 0), SimpleNamespace(max_temp=0), log) is None

    def unreadable(idx, bus):
        raise RuntimeError("no driver")

    monkeypatch.setattr(m05, "_pci_bus_id", lambda idx: None)
    monkeypatch.setattr(m05, "_nvml_reader", unreadable)
    monkeypatch.setattr(m05, "_smi_reader", unreadable)
    with pytest.raises(SystemExit, match="cannot read the GPU temperature"):  # no guard: no eval
        m05.make_guard(torch.device("cuda", 0), args, log)
    monkeypatch.setattr(m05, "_smi_reader", lambda idx, bus: lambda: 55.0)
    fallback = m05.make_guard(torch.device("cuda", 0), args, log)
    assert fallback.source == "nvidia-smi" and fallback.min_interval_s == m05.NVSMI_EVERY_S
    assert m05._bus_tuple("00000000:01:00.0") == m05._bus_tuple(b"0000:01:00.0") == m05._bus_tuple("01:00.0")
