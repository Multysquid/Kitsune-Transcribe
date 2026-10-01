"""The full runs' trainer state in scripts/04_distill.py (the full-run build contract 4.3, WP4b): timed full states to
the private scratch repo (4a), the trainer side of a resume on a new host (4b), the heartbeat call sites, the threads
event, 4e (scalars.parquet at close: tests/test_runlog.py) and fix 7 (the lean step rows).

- ScratchUploader: one upload at a time, SCRATCH_MARK gone when an upload ends (a failure too), older marks cleared on
  a success, UPLOAD_MARK never touched, create_repo never called; rotate_full keeps a dir while its timed upload runs
  and only the newest marked dir, so no marked dir leaks past rotation; the Uploader leaves SCRATCH_MARK behind
- save_full(scratch=True): the mark renamed in with a new dir, touched in an existing same-step one whose trainer.pt/
  .json are rewritten (reason and cadence; a pre_cooldown state at an existing step too, in a run with timed states,
  after waiting for a timed upload of that dir); trainer.pt/.json name the host
- a busy upload: one timed_state_skipped per missed due window, with the upload's running_s
- a crash that cut a timed upload short: the resume sends the state it resumes from at once; a resume without a
  scratch repo drops the marks instead
- build: a run asking for timed states with an output repo but no scratch repo is refused before its logger exists; no
  output repo: timed states skipped; the new-host guard refuses a state that records its host in a run dir without
  events.jsonl, before anything is written
- runs on the fake Hub: the scratch repo ends with one state per run and its pointer (its step the newest
  timed_state_upload_ok's), squashed after every upload, never created; a busy upload skips the next due states without
  a local save and is abandoned at the end; a new host resumes from the scratch state with the Hub's older logs and
  continues exactly as an uninterrupted run (rtol 1e-6); the `resume` event says new_host
- ResumeMismatch and resume_check (match, mismatch, store not built)
- $KITSUNE_HEARTBEAT: beaten by log_step, by every eval batch (Beating) and during the end phase's waits; `threads`
- fix 7: lean rows after the smoke phase, the full set every N-th step, mem/step_peak_reserved_gb every step

CPU only, a tiny random student and a synthetic corpus in the real on-disk formats; the Hub is tests/test_scratch.py's
in-memory ScratchHub (never the real one)."""
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import threading
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
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import load_script, make_fake_corpus, make_fake_selection  # noqa: E402
from kitsune import fullrun, scratch  # noqa: E402
from test_scratch import HubError, Log, ScratchHub, meta, state_dir  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
PEAK = 3e-3
RUNS, SCRATCH = "fake/kitsune-runs", "fake/kitsune-scratch"
TIMED = {"ckpt": {"upload_full_every_min": 1e-9}, "hf": {"output_repo": RUNS, "scratch_repo": SCRATCH}}  # every step


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


def steps_of(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "metrics" / "steps.parquet")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + a tiny student saved exactly as 03 saves one, and a base config: the step clock
    (10 steps, the cooldown from step 5: a pre_cooldown state), the step-0 and final evals only, no periodic checkpoints,
    no smoke phase, no periodic log sync (sync_every_min 1000: only the forced and the closing ones)."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("fullstate")
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
        "schedule": {"warmup_steps": 2, "cooldown_frac": 0.5, "clock": "steps", "max_steps": 10},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_steps": 1000, "greedy_subset": 2, "batch_s": 20, "check_baselines": False,
                 "final_full_greedy": False},
        "ckpt": {"weights_every_steps": 1000, "full_every_steps": 1000, "keep_local": 2, "full_after_smoke": False,
                 "upload_full_at": ["pre_cooldown"]},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 1000,
                "samples_per_eval": 2, "capture_env": False},
        "hf": {"output_repo": None},
        "smoke": {"enabled": False},
    }
    return dict(root=root, base=base, fc=fc)


def write_config(env, name: str, over: dict) -> str:
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


def trainer(monkeypatch, hub: ScratchHub, synchronous: bool = False, after=None):
    """04_distill with the fake Hub; synchronous: every timed upload ends before the loop goes on (deterministic), and
    after(scratch_uploader, local dir) runs once it has."""
    m = load_script("04_distill")
    monkeypatch.setattr(m, "hf_api", lambda: hub)
    if synchronous:
        submit = m.ScratchUploader.submit

        def sync_submit(self, local, name, meta=None):
            fut = submit(self, local, name, meta=meta)
            fut.result(timeout=60)
            if after is not None:
                after(self, Path(local))
            return fut

        monkeypatch.setattr(m.ScratchUploader, "submit", sync_submit)
    return m


def run_states(hub: ScratchHub, run_id: str) -> list[str]:
    return sorted({p.split("/")[3] for p in hub.files(SCRATCH, f"runs/{run_id}/checkpoints/")})


def pointer(hub: ScratchHub, run_id: str) -> dict:
    return json.loads(hub.files(SCRATCH)[fullrun.scratch_pointer(run_id)])


# ------------------------------------------------------------------------------------------------- unit level


def test_keys_constants_and_state_defaults():
    m = load_script("04_distill")
    cfg = m.load_config(None, [])
    assert cfg["ckpt"]["upload_full_every_min"] is None and cfg["hf"]["scratch_repo"] is None
    assert cfg["log"]["full_scalars_every_steps"] == 1 and cfg["log"]["scalars_parquet"] == "sync"
    assert m.TIMED_REASON == fullrun.TIMED_REASON and m.SCRATCH_MARK == fullrun.SCRATCH_MARK != m.UPLOAD_MARK
    assert m.CORE_STEP_TAGS == ("loss/", "combined_loss/", "opt/lr", "opt/grad_norm", "opt/clip_coef", "time/",
                                "perf/", "data/", "sched/", "mem/")
    ok = m.load_config(None, ["ckpt.upload_full_every_min=120", "hf.scratch_repo=Multy123/kitsune-scratch",
                              "hf.output_repo=Multy123/kitsune-runs", "log.full_scalars_every_steps=10",
                              "log.scalars_parquet=close"])
    assert ok["ckpt"]["upload_full_every_min"] == 120 and ok["hf"]["scratch_repo"] == "Multy123/kitsune-scratch"
    for bad in (["ckpt.upload_full_every_min=0"], ["ckpt.upload_full_every_min=-5"], ["ckpt.upload_full_every_min=x"],
                ["hf.scratch_repo=kitsune-scratch"], ["hf.scratch_repo=a/b/c"], ["hf.scratch_repo=5"],
                ["hf.scratch_repo=a/b", "hf.output_repo=a/b"], ["log.full_scalars_every_steps=0"],
                ["log.full_scalars_every_steps=2.5"], ["log.full_scalars_every_steps=true"],
                ["log.scalars_parquet=never"]):
        with pytest.raises(SystemExit):
            m.load_config(None, bad)
    R = m.Run(cfg=cfg, run_dir=Path("r"), device=torch.device("cpu"), amp=False)
    assert (R.st["last_timed_t"], R.st["last_timed_step"], R.st["timed"], R.scratch) == (0.0, 0, None, None)
    assert R.st["data_wait_s"] == 0.0
    assert m.st_timed_defaults() is not m.st_timed_defaults()


def test_scratch_uploader_marks_failures_and_no_leak_past_rotation(tmp_path):
    """A failed timed upload drops its SCRATCH_MARK too; a success clears the marks older dirs kept (a queued upload
    that was cancelled left one), never UPLOAD_MARK, never uploads a mark, never creates the repo; rotation then keeps
    no marked dir beyond the newest one - the leak a failure followed by a success would otherwise leave."""
    m = load_script("04_distill")
    hub, log = ScratchHub(), Log()
    run = tmp_path / "runs" / "full-p01-20260927T120000Z"
    d2 = state_dir(run, 2, extra={m.SCRATCH_MARK: b""})  # a cancelled upload's leftover
    d4 = state_dir(run, 4, extra={m.SCRATCH_MARK: b""})
    d6 = state_dir(run, 6, extra={m.SCRATCH_MARK: b"", m.UPLOAD_MARK: b""})  # also the pre_cooldown state
    sc = m.ScratchUploader(hub, SCRATCH, run.name, log, retries=())
    assert sc.retries == () and m.ScratchUploader(hub, SCRATCH, run.name, log).retries == scratch.RETRY_S
    hub.fail_commit = [HubError(403, "no write access")]
    assert sc.submit(d4, d4.name, meta=meta(4)).result(10) is False
    assert not (d4 / m.SCRATCH_MARK).exists() and (d2 / m.SCRATCH_MARK).exists() and sc.ok == []
    assert not sc.sync_due.is_set()
    assert sc.submit(d6, d6.name, meta=meta(6)).result(10) is True
    assert not any((d / m.SCRATCH_MARK).exists() for d in (d2, d4, d6))
    assert (d6 / m.UPLOAD_MARK).exists()  # still owed to the runs repo: finish.py and rotation go by it
    assert [(o["name"], o["step"]) for o in sc.ok] == [("full_step_6", 6)] and sc.sync_due.is_set()
    files = hub.files(SCRATCH)
    assert not any(p.endswith((m.SCRATCH_MARK, m.UPLOAD_MARK)) for p in files) and hub.created == []
    assert json.loads(files[fullrun.scratch_pointer(run.name)])["step"] == 6
    assert log.kinds() == ["timed_state_upload_error", "timed_state_upload_failed", "timed_state_upload_ok"]
    # a state it cannot hash (no meta: no pointer) is a logged failure, never an exception in the future
    d8 = state_dir(run, 8, extra={m.SCRATCH_MARK: b""})
    assert sc.submit(d8, d8.name).result(10) is False and not (d8 / m.SCRATCH_MARK).exists()
    assert log.kinds()[-1] == "timed_state_upload_failed" and log.events[-1]["attempts"] == 0
    sc.shutdown(5)

    evs = []
    R = SimpleNamespace(cfg={"ckpt": {"keep_local": 1}}, ckpt_dir=run / "checkpoints", st={},
                        uploader=SimpleNamespace(busy=set), scratch=sc,
                        log=SimpleNamespace(event=lambda kind, **kw: evs.append(kw["name"])))
    m.rotate_full(R)
    assert sorted(p.name for p in R.ckpt_dir.iterdir()) == ["full_step_6", "full_step_8"]  # 6: UPLOAD_MARK
    assert evs == ["full_step_2", "full_step_4"]


def test_rotation_keeps_scratch_busy_dirs_and_only_the_newest_marked(tmp_path):
    m = load_script("04_distill")
    ck = tmp_path / "checkpoints"
    for n in (2, 4, 6, 8, 10):
        (ck / f"full_step_{n}").mkdir(parents=True)
    for n in (2, 6):
        (ck / f"full_step_{n}" / m.SCRATCH_MARK).touch()
    evs = []
    R = SimpleNamespace(cfg={"ckpt": {"keep_local": 1}}, ckpt_dir=ck, st={}, uploader=SimpleNamespace(busy=set),
                        scratch=SimpleNamespace(busy=lambda: {ck / "full_step_4"}),
                        log=SimpleNamespace(event=lambda kind, **kw: evs.append(kw["name"])))
    m.rotate_full(R)
    # 4: its timed upload runs; 6: the newest marked dir; 2's mark is an older one's leftover; 10: keep_local
    assert sorted(p.name for p in ck.iterdir()) == ["full_step_10", "full_step_4", "full_step_6"]
    assert evs == ["full_step_2", "full_step_8"]


def test_timed_saves_carry_the_mark_and_a_same_step_one_reuses_the_dir(tmp_path, monkeypatch):
    """save_full(scratch=True): SCRATCH_MARK renamed in with a new dir (never UPLOAD_MARK), touched in the existing dir
    of the same step, whose weights are not saved again but whose trainer.pt/.json are rewritten (the state records
    its reason and the timed cadence); trainer.pt/.json name the host; a pre_cooldown submit of a run with timed states
    asks the loop for a log sync, and a pre_cooldown state at an existing step is rewritten in such a run only."""
    m = load_script("04_distill")
    monkeypatch.setenv("CONTAINER_ID", "c42")
    evs = []
    run = tmp_path / "runs" / "run"
    R = SimpleNamespace(cfg={"ckpt": {"keep_local": 5}}, run_dir=run, ckpt_dir=run / "checkpoints", st=dict(fulls=[]),
                        clock=lambda: 0.0, planner=SimpleNamespace(state_dict=dict),
                        log=SimpleNamespace(event=lambda kind, **kw: evs.append((kind, kw.get("reason"),
                                                                                 kw.get("trainer_only"))),
                                            state_dict=dict),
                        model=torch.nn.Linear(1, 1), opt=SimpleNamespace(state_dict=dict),
                        l2sp=SimpleNamespace(state_dict=dict),
                        uploader=SimpleNamespace(repo=None, busy=set, pending={}))
    R.ckpt_dir.mkdir(parents=True)

    def saved(d: Path) -> dict:
        return torch.load(d / "trainer.pt", weights_only=False)

    d = m.save_full(R, 4, "periodic")
    assert not (d / m.SCRATCH_MARK).exists()
    weights = (d / "model.pt").read_bytes()
    R.st.update(last_timed_step=4, last_timed_t=7.0)  # timed_state sets these before its save
    assert m.save_full(R, 4, m.TIMED_REASON, scratch=True) == d
    assert (d / m.SCRATCH_MARK).exists() and not (d / m.UPLOAD_MARK).exists()
    assert evs == [("checkpoint", "periodic", None), ("checkpoint", "timed", True)]  # trainer.pt/.json only
    assert (d / "model.pt").read_bytes() == weights
    assert (saved(d)["reason"], saved(d)["st"]["last_timed_step"], saved(d)["st"]["last_timed_t"]) == ("timed", 4, 7.0)
    assert json.loads((d / "trainer.json").read_text(encoding="utf-8"))["reason"] == "timed"
    R.st["dev_history"] = [dict(step=2, dev_ce=1.0)]  # WP4a's dev evals: in trainer.pt, never in the brief
    d6 = m.save_full(R, 6, m.TIMED_REASON, scratch=True)
    assert (d6 / m.SCRATCH_MARK).exists() and not (d6 / m.UPLOAD_MARK).exists()
    brief = json.loads((d6 / "trainer.json").read_text(encoding="utf-8"))
    assert brief["reason"] == "timed" and brief["host"]["container_id"] == "c42"
    assert saved(d6)["host"] == brief["host"]
    assert "dev_history" not in brief["st"] and saved(d6)["st"]["dev_history"] == [dict(step=2, dev_ce=1.0)]

    # a study run (no scratch uploader): a pre_cooldown state at an existing step is not rewritten, as before
    evs.clear()
    R.uploader = SimpleNamespace(repo="u/r", busy=set, pending={}, submit=lambda d, n: evs.append(("submit", n, None)))
    m.save_full(R, 6, "pre_cooldown", upload=True)
    assert evs == [("submit", "full_step_6", None)] and saved(d6)["reason"] == "timed"
    # a run with timed states: rewritten, so the runs repo's trainer.json says pre_cooldown (full_queue's reset pick)
    evs.clear()
    R.scratch = SimpleNamespace(sync_due=threading.Event(), busy=set, pending={})
    R.st.update(pre_cooldown_done=True, pre_cooldown_full="full_step_6")
    m.save_full(R, 6, "pre_cooldown", upload=True)
    assert evs == [("checkpoint", "pre_cooldown", True), ("submit", "full_step_6", None)]
    assert saved(d6)["reason"] == "pre_cooldown" and saved(d6)["st"]["pre_cooldown_done"] is True
    assert json.loads((d6 / "trainer.json").read_text(encoding="utf-8"))["reason"] == "pre_cooldown"
    assert R.scratch.sync_due.is_set() and (d6 / m.UPLOAD_MARK).exists()
    R.scratch.sync_due.clear()
    m.save_full(R, 8, "pre_cooldown", upload=True)
    assert R.scratch.sync_due.is_set() and ("submit", "full_step_8", None) in evs


def test_a_same_step_rewrite_waits_for_the_timed_upload_of_its_dir(tmp_path):
    """The cooldown starts at the step of a timed state whose upload is still running: the pre_cooldown rewrite of
    trainer.pt waits for it, so the scratch repo gets exactly the bytes its pointer hashed (the timed trainer.pt) and
    the runs repo the rewritten one."""
    m = load_script("04_distill")
    hub, log = ScratchHub(), Log()
    hub.gate = threading.Event()
    run = tmp_path / "runs" / "full-p01-20260927T120000Z"
    R = SimpleNamespace(cfg={"ckpt": {"keep_local": 5}}, run_dir=run, ckpt_dir=run / "checkpoints", st=dict(fulls=[]),
                        clock=lambda: 0.0, planner=SimpleNamespace(state_dict=dict), log=log,
                        model=torch.nn.Linear(1, 1), opt=SimpleNamespace(state_dict=dict),
                        l2sp=SimpleNamespace(state_dict=dict),
                        uploader=SimpleNamespace(repo=None, busy=set, pending={}))
    log.state_dict = dict
    R.ckpt_dir.mkdir(parents=True)
    R.scratch = m.ScratchUploader(hub, SCRATCH, run.name, log, retries=())
    d = m.save_full(R, 12, m.TIMED_REASON, scratch=True)
    timed_pt = (d / "trainer.pt").read_bytes()
    R.scratch.submit(d, d.name, meta=meta(12))
    deadline = time.monotonic() + 10
    while R.scratch.running is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert R.scratch.running is not None and R.scratch.running_s() >= 0
    threading.Timer(0.5, hub.gate.set).start()
    t0 = time.monotonic()
    try:
        R.st.update(pre_cooldown_done=True, pre_cooldown_full=d.name)
        m.save_full(R, 12, "pre_cooldown")
        assert time.monotonic() - t0 >= 0.4  # it waited for the upload
    finally:
        hub.gate.set()
    ptr = pointer(hub, run.name)
    got = hub.files(SCRATCH)[f"{fullrun.scratch_state_dir(run.name, 12)}/trainer.pt"]
    assert got == timed_pt and hashlib.sha256(got).hexdigest() == ptr["files"]["trainer.pt"]["sha256"]
    assert torch.load(d / "trainer.pt", weights_only=False)["reason"] == "pre_cooldown"
    assert R.scratch.running is None and not (d / m.SCRATCH_MARK).exists()
    R.scratch.shutdown(5)


def test_a_hung_upload_logs_one_skip_per_missed_window(tmp_path):
    """While a timed upload hangs, every due window it makes the loop miss logs one timed_state_skipped with the
    upload's growing running_s (not one event for the rest of the run); nothing is saved meanwhile."""
    m = load_script("04_distill")
    hub, log = ScratchHub(), Log()
    hub.gate = threading.Event()
    run = tmp_path / "runs" / "run"
    sc = m.ScratchUploader(hub, SCRATCH, "run", log, retries=())
    d = state_dir(run, 3, extra={m.SCRATCH_MARK: b""})
    sc.submit(d, d.name, meta=meta(3))
    deadline = time.monotonic() + 10
    while sc.running is None and time.monotonic() < deadline:
        time.sleep(0.01)
    R = SimpleNamespace(scratch=sc, cfg={"ckpt": {"upload_full_every_min": 2}}, log=log,
                        st=dict(m.st_timed_defaults(), last_timed_t=100.0, last_timed_step=3))
    log.wait_sync = lambda timeout=None: True
    try:
        # due from t = 220 (every 2 min after 100): windows 1 (220, 300), 2 (340, 400), 3 (460)
        for step, t in ((4, 150.0), (5, 220.0), (6, 300.0), (7, 340.0), (8, 400.0), (9, 460.0)):
            m.timed_state(R, t, step)
            time.sleep(0.02)
        skips = [e for e in log.events if e["kind"] == "timed_state_skipped"]
        assert [e["at_step"] for e in skips] == [5, 7, 9]
        assert all(e["reason"] == "busy" and e["uploading"] == ["full_step_3"] for e in skips)
        assert 0 <= skips[0]["running_s"] <= skips[1]["running_s"] <= skips[2]["running_s"]
        assert (R.st["last_timed_step"], R.st["timed"]) == (3, None)  # nothing saved
    finally:
        hub.gate.set()
    sc.shutdown(5)
    assert sc.running is None and sc.running_s() is None


def test_log_step_lean_rows_and_the_reserved_peak(tmp_path, monkeypatch):
    """Fix 7: with log.full_scalars_every_steps 3, the smoke phase's steps and every 3rd step log the full row (and
    read system_stats); the others only the CORE_STEP_TAGS keys. mem/step_peak_reserved_gb every step under CUDA."""
    from kitsune import runlog

    m = load_script("04_distill")
    rows, stats = {}, []
    monkeypatch.setattr(runlog, "system_stats", lambda: stats.append(1) or {"sys/proc/rss_gb": 1.5})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 3 * 2**30)

    def fresh(n: int):
        cfg = m.load_config(None, [f"log.full_scalars_every_steps={n}", "smoke.steps=2", "schedule.clock=steps",
                                   "schedule.max_steps=100", "schedule.warmup_steps=1"])
        R = m.Run(cfg=cfg, run_dir=tmp_path, device=torch.device("cpu"), amp=False)
        R.device = SimpleNamespace(type="cuda")
        R.log = SimpleNamespace(step_row=lambda row, step: rows.__setitem__(step, dict(row)),
                                train_utts=lambda u: None, scalars=lambda *a: None)
        R.planner, R.l2sp, R.src_index = SimpleNamespace(epoch_stats={0: {"steps": 100}}), None, {"src_a": 0}
        return R

    out = dict(n_tok=10, tot=np.arange(1.0, len(m.TERMS) + 1), l2sp=0.1, audio_real=5.0, audio_pad=6.0, dec_real=8.0,
               dec_pad=10.0, grad_norm=1.0, clip_coef=1.0, masked_frac=0.1, n_utts=2, dropped=0, n_micro=1,
               by_src=np.ones((1, 3)), n_src=np.array([10]), by_bucket=np.ones((2, 3)), n_bucket=np.array([4, 2]),
               utts=[])
    R = fresh(3)
    for step in range(1, 8):
        m.log_step(R, step, 1e-3, 1, out, 0.01, 0.1, 0, step - 1)
        if step == 2:
            R.st["smoke_done"] = True  # smoke_end, after the smoke phase's last step
    full, lean = (1, 2, 3, 6), (4, 5, 7)
    core = ("loss/objective", "combined_loss/train", "opt/lr", "opt/grad_norm", "opt/clip_coef", "time/step_s",
            "perf/audio_s_per_s", "data/epoch", "sched/phase", "mem/step_peak_gb", "mem/step_peak_reserved_gb")
    for step in (*full, *lean):
        assert all(k in rows[step] for k in core), step
        assert rows[step]["mem/step_peak_reserved_gb"] == 3.0
    for step in full:
        assert {"sys/proc/rss_gb", "tok/top1", "src/src_a/kl", "bucket/p1_gt_0.99/frac", "aug/masked_frac"} <= set(
            rows[step]), step
    for step in lean:
        assert all(k.startswith(m.CORE_STEP_TAGS) for k in rows[step]) and "aug/masked_frac" not in rows[step]
    assert len(stats) == len(full)
    rows.clear()
    R = fresh(1)  # the default: the full row every step, as before
    R.st["smoke_done"] = True
    for step in range(1, 4):
        m.log_step(R, step, 1e-3, 1, out, 0.01, 0.1, 0, step - 1)
    assert all("sys/proc/rss_gb" in rows[s] and "tok/top1" in rows[s] for s in (1, 2, 3))


def test_resume_check_reads_a_ctc_states_frame_store(tmp_path):
    """resume_check opens the store the state's family trains on: for family "ctc" the frame store (cache/ctc_train),
    never the token store of the same rows; the planner is the frame planner at the state's micro_audio_s."""
    import dataclasses

    from kitsune import trainset

    m = load_script("04_distill")
    cache = tmp_path / "cache"
    cfg = m._merge(m.DEFAULTS, {"family": "ctc", "parakeet_root": "parakeet_out", "cache_dir": str(cache),
                                "sources": ["src_a"], "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4}})
    utts = [trainset.Utt(id=f"src_a/u{i}", source="src_a", duration=0.5 + 0.1 * i, n_tok=3 + i, audio_off=0,
                         audio_len=1, tok_off=0, row=i) for i in range(12)]
    planner = trainset.StepPlanner(utts, step_audio_s=6, micro_audio_s=2.5, max_dec_len=None, pool_micro=4,
                                   seed=int(cfg["seed"]))  # the memory probe halved micro_audio_s
    full = tmp_path / "run" / "checkpoints" / "full_step_4"
    full.mkdir(parents=True)
    torch.save(dict(format=1, step=4, cfg=cfg, st={"memory": {"micro_audio_s": 2.5}}, planner=planner.state_dict()),
               full / "trainer.pt")

    def store(d: Path, files, kind: str | None):
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(b"")
        pd.DataFrame([dataclasses.asdict(u) for u in utts]).to_parquet(d / "index.parquet")
        (d / "stores.json").write_text(json.dumps({"kind": kind} if kind else {}), encoding="utf-8")

    store(cache / "train", ("audio.bin", "audio_offsets.npy", "targets_tokens.npy", "targets_topk_idx.npy",
                            "targets_topk_lp.npy", "targets_offsets.npy"), None)
    got = m.resume_check(tmp_path / "run")
    assert got["ok"] is False and got["reason"] == "store not built"  # the token store is not the CTC run's
    store(cache / "ctc_train", trainset.FRAME_FILES, "frames")
    got = m.resume_check(tmp_path / "run")
    assert got == dict(ok=True, reason=None, state_fingerprint=planner.fingerprint,
                       store_fingerprint=planner.fingerprint, n_utts_state=12, n_utts_store=12, micro_audio_s=2.5)


def test_close_failed_abandons_timed_uploads_without_waiting(tmp_path):
    """The failure path cancels the queued timed uploads and waits for none: one timed_state_upload_abandoned after
    the logs are closed; uploads_left_running counts the running one (the os._exit path)."""
    import kitsune.evaluate  # noqa: F401  make_summary's import, done here: only _close_failed's own waits are timed

    m = load_script("04_distill")
    hub, order = ScratchHub(), []
    hub.gate = threading.Event()
    log = SimpleNamespace(event=lambda kind, **kw: order.append((kind, kw.get("names"))),
                          close=lambda summary: order.append(("close", summary["status"])), elapsed=lambda: 1.0,
                          wait_sync=lambda timeout=None: True)
    R = m.Run(cfg=copy.deepcopy(m.DEFAULTS), run_dir=tmp_path / "run", device=torch.device("cpu"), amp=False, log=log)
    R.uploader = m.Uploader(None, None, "run", True, log)
    R.scratch = m.ScratchUploader(hub, SCRATCH, "run", log, retries=())
    run = tmp_path / "run"
    for n in (3, 5):
        R.scratch.submit(state_dir(run, n), f"full_step_{n}", meta=meta(n))
    try:
        t0 = time.monotonic()
        m._close_failed(R, "failed", RuntimeError("CUDA error"))
        assert time.monotonic() - t0 < 5
        assert order == [("close", "failed"), ("timed_state_upload_abandoned", ["full_step_3", "full_step_5"])]
        assert m.uploads_left_running(R)  # full_step_3 still runs
    finally:
        hub.gate.set()
    R.scratch.worker.join(10)
    assert not R.scratch.worker.is_alive() and run_states(hub, "run") == ["full_step_3"]  # 5 was cancelled
    assert not m.uploads_left_running(R)


def test_build_needs_a_scratch_repo_for_timed_states(env, monkeypatch):
    """With an output repo, ckpt.upload_full_every_min needs hf.scratch_repo, refused before the logger exists; without
    an output repo timed states are skipped (one event); with both a ScratchUploader is set up and the runs repo's
    Uploader leaves SCRATCH_MARK behind as it does UPLOAD_MARK."""
    hub = ScratchHub()
    m = trainer(monkeypatch, hub)
    cfg = write_config(env, "bld-refused", {"ckpt": {"upload_full_every_min": 120}, "hf": {"output_repo": RUNS}})
    with pytest.raises(SystemExit, match="hf.scratch_repo"):
        m.build(m.parse_args(["--config", cfg]))
    assert list(one_run(env["root"], "bld-refused").iterdir()) == []  # no logger, no files

    R, _ = m.build(m.parse_args(["--config", write_config(env, "bld-local", {"ckpt": {"upload_full_every_min": 120}})]))
    R.log.close()
    assert R.scratch is None and R.uploader.ignore == [m.UPLOAD_MARK]
    assert [e["reason"] for e in events(R.run_dir, "timed_state_skipped")] == ["no output repo"]

    R, _ = m.build(m.parse_args(["--config", write_config(env, "bld-timed", merged(TIMED, {
        "ckpt": {"upload_full_every_min": 120}}))]))
    try:
        assert isinstance(R.scratch, m.ScratchUploader) and R.scratch.repo == SCRATCH
        assert R.scratch.run_id == R.run_dir.name and R.uploader.ignore == [m.UPLOAD_MARK, m.SCRATCH_MARK]
        d = R.ckpt_dir / "full_step_3"
        d.mkdir(parents=True)
        for f, b in (("model.pt", b"w"), (m.UPLOAD_MARK, b""), (m.SCRATCH_MARK, b"")):
            (d / f).write_bytes(b)
        assert R.uploader.submit(d, d.name).result(10) is True
        (c,) = [c for c in hub.commits if c.get("folder", "").endswith("/checkpoints/full_step_3")]
        assert c["adds"] == [f"runs/{R.run_dir.name}/checkpoints/full_step_3/model.pt"]
        assert c["ignore"] == [m.UPLOAD_MARK, m.SCRATCH_MARK] and not events(R.run_dir, "timed_state_skipped")
    finally:
        R.uploader.shutdown(5)
        R.log.close()
    assert set(hub.created) == {(RUNS, True)}  # the trainer creates (exist_ok) the runs repo only


# ------------------------------------------------------------------------------------------------------ runs


def test_timed_states_go_to_the_scratch_repo_one_per_run(env, monkeypatch):
    """Every due timed state is saved, uploaded in one commit that replaces the run's previous one, and squashed; the
    scratch repo ends with the run's newest state and its pointer; the logs are synced after each upload; the
    pre_cooldown state goes to the runs repo without either mark; summary.json counts the uploads."""
    hub = ScratchHub()
    m = trainer(monkeypatch, hub, synchronous=True)
    assert m.main(["--config", write_config(env, "timed", TIMED)]) == 0
    run = one_run(env["root"], "timed")
    ok = events(run, "timed_state_upload_ok")
    timed = events(run, "timed_state")
    assert len(ok) == len(timed) >= 5 and [e["step"] for e in ok] == [e["step"] for e in timed]
    assert all(e["squash_ok"] for e in ok) and hub.squashes == [SCRATCH] * len(ok)
    assert [e["deleted"] for e in ok[1:]] == [[f"full_step_{e['step']}"] for e in ok[:-1]]
    assert run_states(hub, run.name) == [f"full_step_{ok[-1]['step']}"]
    ptr = pointer(hub, run.name)
    assert fullrun.pointer_problems(ptr) == [] and ptr["step"] == ok[-1]["step"] and ptr["run_id"] == run.name
    # the uploaded trainer.pt (the local one of the last step was rewritten by the end save, after its upload ended)
    trainer_pt = torch.load(io.BytesIO(hub.files(SCRATCH)[f"{fullrun.scratch_state_dir(run.name, ptr['step'])}/trainer.pt"]),
                            map_location="cpu", weights_only=True)
    assert ptr["planner_fingerprint"] == trainer_pt["planner"]["fingerprint"] and ptr["micro_audio_s"] == 3.0
    assert ptr["n_train_utts"] == trainer_pt["planner"]["n_utts"] and trainer_pt["reason"] == "timed"
    assert set(hub.created) == {(RUNS, True)}  # never the scratch repo
    assert not any(p.endswith((m.SCRATCH_MARK, m.UPLOAD_MARK)) for r in (RUNS, SCRATCH) for p in hub.files(r))
    assert not list(run.rglob(m.SCRATCH_MARK))  # every upload ended
    # the pre_cooldown state (step 5) went to the runs repo, marks left behind
    pre = [c for c in hub.commits if c.get("folder", "").endswith("/checkpoints/full_step_5")]
    assert len(pre) == 1 and pre[0]["ignore"] == [m.UPLOAD_MARK, m.SCRATCH_MARK]
    # log syncs in the loop after the uploads, though sync_every_min is 1000 (else only the end phase's and the close's)
    syncs = events(run, "sync_ok")
    assert len(syncs) >= 3 and any(e["step"] < 10 for e in syncs)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["timed_states"]["repo"] == SCRATCH
    assert summary["timed_states"]["count"] == len(ok) and summary["timed_states"]["last"]["step"] == ok[-1]["step"]
    assert [e["reason"] for e in events(run, "checkpoint") if e["ckpt"] == "full"].count("timed") == len(timed)


def test_a_busy_upload_skips_the_due_states_without_saving(env, monkeypatch):
    """One upload at a time: while the first timed upload hangs, the states that fall due are not saved (one
    timed_state_skipped per due window: every step here); the end phase abandons the upload without waiting for it and
    the run still ends 0."""
    hub = ScratchHub()
    hub.gate = threading.Event()
    m = trainer(monkeypatch, hub)
    try:
        assert m.main(["--config", write_config(env, "busy", TIMED)]) == 0
        run = one_run(env["root"], "busy")
        assert m._HARD_EXIT  # the scratch upload is still running: the __main__ entry would leave by os._exit
        first = events(run, "timed_state")
        assert len(first) == 1
        name = first[0]["name"]
        skipped = events(run, "timed_state_skipped")
        assert [e["at_step"] for e in skipped] == list(range(first[0]["step"] + 1, 11))
        assert all(e["reason"] == "busy" and e["uploading"] == [name] for e in skipped)
        # the hung upload's growing age (None only before the worker thread picked it up)
        ages = [e["running_s"] for e in skipped if e["running_s"] is not None]
        assert ages and ages == sorted(ages) and skipped[-1]["running_s"] is not None
        saved = [e["name"] for e in events(run, "checkpoint") if e["ckpt"] == "full" and e["reason"] == "timed"]
        assert saved == [name] and (run / "checkpoints" / name / m.SCRATCH_MARK).exists()
        assert [e["names"] for e in events(run, "timed_state_upload_abandoned")] == [[name]]
        assert events(run, "timed_state_upload_ok") == []
    finally:
        hub.gate.set()
    for th in threading.enumerate():
        if th.name == "ckpt-upload":
            th.join(10)
    assert run_states(hub, run.name) == [name] and not (run / "checkpoints" / name / m.SCRATCH_MARK).exists()


def test_a_timed_state_at_a_periodic_step_records_itself(env, monkeypatch):
    """A timed state due at the step of a periodic full state (full_every_steps 1: every step here) reuses its dir and
    rewrites only trainer.pt/.json, so each state the scratch repo gets says reason "timed" and has its own step as the
    timed cadence's (a resume from it is not due at once). The pre_cooldown state at such a step is rewritten too: the
    runs repo's trainer.json says pre_cooldown (full_queue's reset pick) and the checkpoint event names it."""
    hub = ScratchHub()
    uploaded = {}

    def keep(sc, local):  # the trainer.pt each timed upload committed, read back from the scratch repo
        step = int(local.name.split("_")[-1])
        data = hub.files(SCRATCH)[f"{fullrun.scratch_state_dir(local.parent.parent.name, step)}/trainer.pt"]
        uploaded[step] = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)

    m = trainer(monkeypatch, hub, synchronous=True, after=keep)
    assert m.main(["--config", write_config(env, "samestep", merged(TIMED, {"ckpt": {"full_every_steps": 1}}))]) == 0
    run = one_run(env["root"], "samestep")
    assert [e["step"] for e in events(run, "timed_state")] == sorted(uploaded) == list(range(1, 11))
    full = [e for e in events(run, "checkpoint") if e["ckpt"] == "full"]
    assert [(e["reason"], e.get("trainer_only", False)) for e in full if e["name"] == "full_step_3"] == [
        ("periodic", False), ("timed", True)]  # one save of the weights, then the trainer files only
    for step, tp in uploaded.items():
        assert (tp["reason"], tp["step"], tp["st"]["last_timed_step"], tp["st"]["last_full_step"]) == (
            "timed", step, step, step), step
        assert tp["st"]["last_timed_t"] <= tp["st"]["train_s"]
    pre = [e for e in full if e["reason"] == "pre_cooldown"]
    assert [(e["name"], e.get("trainer_only")) for e in pre] == [("full_step_5", True)]
    tj = json.loads(hub.files(RUNS)[f"runs/{run.name}/checkpoints/full_step_5/trainer.json"])
    assert (tj["reason"], tj["st"]["pre_cooldown_done"], tj["st"]["pre_cooldown_full"]) == (
        "pre_cooldown", True, "full_step_5")
    assert "full_step_5" in [e["name"] for e in events(run, "ckpt_upload_ok")]
    assert not list(run.rglob(m.SCRATCH_MARK))


def test_a_crash_during_a_timed_upload_sends_the_state_again_on_resume(env, monkeypatch, tmp_path):
    """A crash cuts the upload of the newest timed state short (its dir keeps SCRATCH_MARK; the scratch repo still
    holds the state before it). The resume on the same host sends the state it resumes from at once, before any window
    falls due, so the scratch repo is not two windows behind. A hand resume of the same crash without a scratch repo
    drops the mark instead, so rotation no longer keeps that dir."""
    hub = ScratchHub()
    snaps = {}

    def snap(sc, local):  # the scratch repo right after each upload
        with hub.lock:
            snaps[int(local.name.split("_")[-1])] = dict(hub.repos[SCRATCH])

    m = trainer(monkeypatch, hub, synchronous=True, after=snap)
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "8")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", write_config(env, "requeue", TIMED)])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    run = one_run(env["root"], "requeue")
    rid = run.name
    with hub.lock:  # the crash cut the upload of full_step_7 short: the Hub holds full_step_6, the dir keeps its mark
        hub.repos[SCRATCH] = dict(snaps[6])
    (run / "checkpoints" / "full_step_7" / m.SCRATCH_MARK).touch()
    assert pointer(hub, rid)["step"] == 6
    hand = tmp_path / "hand" / "runs" / rid
    shutil.copytree(run, hand)

    # the same host resumes, with a cadence that lets nothing fall due: only the state resumed from goes up
    n_ok = len(events(run, "timed_state_upload_ok"))
    assert m.main(["--resume", str(run), "--set", "ckpt.upload_full_every_min=1000"]) == 0
    again = events(run, "timed_state")[-1]
    assert (again["name"], again["step"], again["requeued"]) == ("full_step_7", 7, True)
    assert [e["step"] for e in events(run, "timed_state_upload_ok")[n_ok:]] == [7]
    assert pointer(hub, rid)["step"] == 7 and run_states(hub, rid) == ["full_step_7"]
    assert pointer(hub, rid)["host"] == events(run, "resume")[-1]["prev_host"]  # the host that saved it
    assert not list(run.rglob(m.SCRATCH_MARK))
    timed = json.loads((run / "summary.json").read_text(encoding="utf-8"))["timed_states"]
    assert timed["count"] == 7 and timed["last"]["step"] == 7

    # by hand, without the scratch repo: the mark is dropped, and the dir is rotated away like any other
    commits = len(hub.commits)
    assert m.main(["--resume", str(hand), "--set", "hf.scratch_repo=null", "--set", "ckpt.upload_full_every_min=null",
                   "--set", "ckpt.keep_local=1"]) == 0
    assert events(hand, "resume")[-1]["scratch_marks_dropped"] == ["full_step_7"]
    assert not list(hand.rglob(m.SCRATCH_MARK))
    assert sorted(p.name for p in (hand / "checkpoints").iterdir() if p.name.startswith("full_")) == ["full_step_10"]
    assert not [c for c in hub.commits[commits:] if c["repo"] == SCRATCH]


def test_a_new_host_resumes_from_the_scratch_state_and_the_hubs_older_logs(env, monkeypatch, tmp_path):
    """The host dies (a crash at step 8, after which nothing reaches the Hub any more) while its newest timed state is
    newer than the last log sync. A new host gets the run's logs from the runs repo and the state from the scratch
    repo (verified against the pointer): without the logs the resume is refused before anything is written;
    resume_check matches; the resumed run continues exactly as an uninterrupted one (rtol 1e-6), says new_host, and
    its own timed uploads replace the dead host's state."""
    hub = ScratchHub()

    def freeze(sc, local):  # the host's last log sync: none reaches the Hub after the upload of step >= 4
        if int(local.name.split("_")[-1]) >= 4:
            hub.drop_syncs = True

    m = trainer(monkeypatch, hub, synchronous=True, after=freeze)
    ref_cfg = write_config(env, "par-ref", TIMED)
    assert m.main(["--config", ref_cfg]) == 0
    hub.drop_syncs = False
    ref = one_run(env["root"], "par-ref")
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "8")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", write_config(env, "par", TIMED)])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    dead = one_run(env["root"], "par")
    rid = dead.name
    assert hub.dropped  # the crash's closing sync never arrived

    # the new host: the state from the scratch repo, verified against its pointer
    ptr = pointer(hub, rid)
    k = ptr["step"]
    assert k == events(dead, "timed_state_upload_ok")[-1]["step"] and 4 <= k <= 7
    new = tmp_path / "newhost" / "runs" / rid
    dest = new / "checkpoints" / ptr["name"]
    dest.mkdir(parents=True)
    for f, meta_ in ptr["files"].items():
        data = hub.files(SCRATCH)[f"{fullrun.scratch_state_dir(rid, k)}/{f}"]
        assert len(data) == meta_["size"] and hashlib.sha256(data).hexdigest() == meta_["sha256"]
        (dest / f).write_bytes(data)
    with pytest.raises(SystemExit, match="no events.jsonl"):  # the logs first
        m.main(["--resume", str(new)])
    assert sorted(p.name for p in new.iterdir()) == ["checkpoints"]  # the guard fired before the logger
    check = m.resume_check(new)
    assert check == dict(ok=True, reason=None, state_fingerprint=ptr["planner_fingerprint"],
                         store_fingerprint=ptr["planner_fingerprint"], n_utts_state=ptr["n_train_utts"],
                         n_utts_store=ptr["n_train_utts"], micro_audio_s=3.0)
    for path, data in hub.files(RUNS, f"runs/{rid}/").items():  # the logs, without checkpoints/
        rel = path[len(f"runs/{rid}/"):]
        if not rel.startswith("checkpoints/"):
            (new / rel).parent.mkdir(parents=True, exist_ok=True)
            (new / rel).write_bytes(data)
    assert (new / "events.jsonl").is_file() and (new / "metrics" / "steps.parquet").is_file()
    pulled = steps_of(new)
    assert pulled["step"].max() < k  # the state is newer than the last sync

    monkeypatch.setenv("CONTAINER_ID", "new-box")
    assert m.main(["--resume", str(new)]) == 0
    res = events(new, "resume")[-1]
    assert res["new_host"] is True and res["host"]["container_id"] == "new-box"
    assert res["prev_host"] == ptr["host"] and res["at_step"] == k
    a, b = steps_of(ref).set_index("step"), steps_of(new).set_index("step")
    after = list(range(k + 1, 11))
    assert b.index.tolist() == pulled["step"].tolist() + after  # the gap between the last sync and the state stays
    for col in ("loss/objective", "loss/kl", "loss/ce", "opt/lr", "opt/grad_norm", "tok/top1"):
        np.testing.assert_allclose(b.loc[after, col].to_numpy(), a.loc[after, col].to_numpy(), rtol=1e-6, atol=1e-9,
                                   err_msg=col)
    assert json.loads((new / "summary.json").read_text(encoding="utf-8"))["steps"] == 10
    newest = events(new, "timed_state_upload_ok")[-1]["step"]
    assert newest > k and run_states(hub, rid) == [f"full_step_{newest}"] and pointer(hub, rid)["step"] == newest
    assert pointer(hub, rid)["host"]["container_id"] == "new-box"


def test_resume_mismatch_and_resume_check(env, monkeypatch, tmp_path):
    """A state whose planner does not fit the train store: resume_check says so on the CPU (fingerprint, then
    n_utts), a store that is not on disk is "store not built", no state at all FileNotFoundError; the resume raises
    ResumeMismatch, whose message and summary.json error start with "ResumeMismatch"."""
    m = load_script("04_distill")
    assert m.main(["--config", write_config(env, "mism", {})]) == 0
    run = one_run(env["root"], "mism")
    assert m.resume_check(run)["ok"] is True

    def variant(name: str, edit) -> Path:
        d = tmp_path / name
        shutil.copytree(run, d)
        full = m.find_full_state(d)
        state = torch.load(full / "trainer.pt", map_location="cpu", weights_only=True)
        edit(state)
        torch.save(state, full / "trainer.pt")
        return d

    fp = variant("fp", lambda s: s["planner"].update(fingerprint="0" * 16))
    got = m.resume_check(fp)
    assert got["ok"] is False and got["reason"] == "fingerprint mismatch" and got["state_fingerprint"] == "0" * 16
    assert got["store_fingerprint"] == m.resume_check(run)["store_fingerprint"]
    got = m.resume_check(variant("nu", lambda s: s["planner"].update(n_utts=s["planner"]["n_utts"] + 1)))
    assert got["ok"] is False and got["reason"] == "n_utts mismatch"
    got = m.resume_check(variant("nostore", lambda s: s["cfg"].update(cache_dir=str(tmp_path / "empty-cache"))))
    assert got["ok"] is False and got["reason"] == "store not built" and got["store_fingerprint"] is None
    (tmp_path / "nostate").mkdir()
    with pytest.raises(FileNotFoundError):
        m.resume_check(tmp_path / "nostate")

    with pytest.raises(m.ResumeMismatch, match="^ResumeMismatch: "):
        m.main(["--resume", str(fp)])
    assert json.loads((fp / "summary.json").read_text(encoding="utf-8"))["error"].startswith("ResumeMismatch")


def test_heartbeats_threads_event_and_a_run_without_timed_states(env, monkeypatch, tmp_path):
    """$KITSUNE_HEARTBEAT is touched by log_step, by the evaluators' batches (the Beating featuriser) and during the
    end phase's waits; setup_processing logs the threads event. The run (an output repo, no scratch repo) behaves as
    before: no timed event, no timed_states in summary.json, the runs repo's uploads ignore UPLOAD_MARK only; and with
    log.full_scalars_every_steps 2 steps.parquet still has every step, the odd ones lean; summary.json's throughput
    has data_wait_frac."""
    from kitsune import heartbeat

    hub = ScratchHub()
    m = trainer(monkeypatch, hub)
    hb = tmp_path / "state" / "hb" / "train-item"
    monkeypatch.setenv("KITSUNE_HEARTBEAT", str(hb))
    monkeypatch.setenv("KITSUNE_THREADS_PER_GPU", "7")
    callers = []
    orig = heartbeat.beat

    def beat(path=None, *, force=False):
        callers.append(sys._getframe(1).f_code.co_name)
        orig(path, force=force)

    monkeypatch.setattr(heartbeat, "beat", beat)
    assert m.main(["--config", write_config(env, "hb", {"hf": {"output_repo": RUNS},
                                                         "log": {"full_scalars_every_steps": 2}})]) == 0
    run = one_run(env["root"], "hb")
    assert hb.is_file()
    assert {"log_step", "__call__", "beating"} <= set(callers)  # a step, an eval batch, the end phase's waits
    (th,) = events(run, "threads")
    assert th["torch_threads"] == torch.get_num_threads() and th["interop_threads"] == torch.get_num_interop_threads()
    assert list(th["env"]) == [*fullrun.ENV_THREAD_POOLS, "KITSUNE_CPU_QUOTA", "KITSUNE_THREADS_PER_GPU"]
    assert th["env"]["KITSUNE_THREADS_PER_GPU"] == "7"
    assert not [e for e in events(run) if e["kind"].startswith("timed_state")]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert "timed_states" not in summary and summary["status"] == "complete"
    assert 0.0 <= summary["throughput"]["data_wait_frac"] < 1.0  # the loader's share of the steps' time
    # what smoke check 5's steady-state wait reads (full_queue.steady_wait): the launch's phase train and every step's
    # time/data_wait_s and time/step_s, the lean steps included
    assert [e["at_step"] for e in events(run, "phase") if e.get("name") == "train"] == [0]
    rows = [json.loads(x) for x in (run / "metrics" / "scalars.jsonl").read_text(encoding="utf-8").splitlines()]
    for tag in ("time/data_wait_s", "time/step_s"):
        assert {r["step"] for r in rows if r["tag"] == tag} == set(range(1, 11)), tag
    ups =[c for c in hub.commits if "/checkpoints/" in c.get("folder", "")]
    assert ups and all(c["ignore"] == [m.UPLOAD_MARK] for c in ups)
    steps = steps_of(run)
    assert steps["step"].tolist() == list(range(1, 11))
    odd, even = steps[steps["step"] % 2 == 1], steps[steps["step"] % 2 == 0]
    assert odd["aug/masked_frac"].isna().all() and even["aug/masked_frac"].notna().all()
    assert odd["tok/top1"].isna().all() and odd["loss/objective"].notna().all() and odd["time/step_s"].notna().all()
