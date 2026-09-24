"""End to end: scripts/04_distill.py on CPU with a tiny random student and a synthetic corpus in the real on-disk formats.

One run goes through the smoke phase, the step-0 eval, training with evals, histograms and checkpoints, dies at a
simulated crash (KITSUNE_CRASH_AT_STEP), is resumed from its full state, reaches the cooldown (pre-cooldown full state),
does the final full eval + verdict + summary, and uploads to a fake HF API; tools/export_run.py then exports it. The
schedule runs on the step clock so every assertion is deterministic; the wall-clock WSD is covered by the schedule unit
tests. The synthetic teacher only ever emits 40 token ids, so a 2-layer d=64 student learns visibly in 20 steps."""
import json
import math
import os
import sys
from pathlib import Path

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
PEAK, WARMUP, COOLDOWN, MAX_STEPS = 3e-3, 3, 0.3, 20


class FakeHub:
    """Stands in for huggingface_hub.HfApi: records every upload, serves upload_file bytes back."""

    def __init__(self):
        self.files, self.folders, self.repos = {}, [], []

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False, **kw):
        self.repos.append((repo_id, private))

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None, **kw):
        self.files[path_in_repo] = bytes(path_or_fileobj)

    def upload_folder(self, *, repo_id, folder_path, path_in_repo=None, commit_message=None, repo_type=None, **kw):
        files = sorted(p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*") if p.is_file())
        self.folders.append((path_in_repo, files, kw.get("ignore_patterns")))

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.files[filename])
        return str(p)


def events(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + a tiny student saved exactly as 03 saves one, and a config for it."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    root = tmp_path_factory.mktemp("e2e")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (36, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=3)
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

    config = {
        "run_name": "tiny", "student": str(sdir), "data_root": str(fc.data), "teacher_root": str(fc.teacher_out),
        "second_root": str(fc.second_out), "selection": str(sel), "cache_dir": str(root / "cache"),
        "runs_root": str(root / "runs"), "sources": ["src_a", "src_b"], "eval_sets": EVAL,
        "device": "cpu", "autocast": "none",
        "optim": {"lr": PEAK},
        "schedule": {"warmup_steps": WARMUP, "cooldown_frac": COOLDOWN, "clock": "steps", "max_steps": MAX_STEPS},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_steps": 5, "greedy_subset": 3, "batch_s": 20, "check_baselines": False},
        "ckpt": {"weights_every_steps": 5, "full_every_steps": 5, "keep_local": 2},
        "log": {"layer_stats_every": 4, "hist_every": 8, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 4},
        "hf": {"output_repo": "fake-user/kitsune-runs"},
        "smoke": {"steps": 4, "min_audio_s_per_s": 0, "require_loss_decrease": False},
    }
    path = root / "config.json"
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    return dict(root=root, config=path, student=sdir, fc=fc)


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    h = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: h)
    return h


def test_schedule_wsd_and_cadence():
    m = load_script("04_distill")
    lr = [m.wsd_lr(1.0, s, t, 100.0, 10, 0.2) for s, t in [(1, 0), (5, 4), (10, 9), (50, 49), (80, 80), (91, 90),
                                                               (100, 99), (101, 100)]]
    assert lr[0] == (0.1, 0) and lr[1] == (0.5, 0) and lr[2] == (1.0, 1) and lr[3] == (1.0, 1)
    assert lr[4] == (1.0, 2)  # the cooldown starts at t = 0.8 T, at full LR
    assert lr[5][0] == pytest.approx(1 - math.sqrt(0.5)) and lr[5][1] == 2
    assert lr[6][0] == pytest.approx(1 - math.sqrt(19 / 20)) and lr[7][0] == 0.0
    assert m.wsd_lr(2.0, 1, 3600.0, 4 * 3600.0, 0, 0.2) == (2.0, 1)  # wall clock: seconds in, seconds out
    assert m.due(1200.0, 7, 0.0, 0, 20, None) and not m.due(1199.0, 7, 0.0, 0, 20, None)
    assert m.due(0.0, 10, 0.0, 5, 20, 5) and not m.due(9e9, 9, 0.0, 5, 20, 5)  # steps win when set


def test_uploader_never_waits_forever_on_a_stalled_upload(tmp_path):
    """A checkpoint upload that never returns (a Hub request the server accepted and never answered) must not hold the
    trainer: wait() and shutdown() return after their bound, the upload is reported as None, and its thread is a
    daemon, which the interpreter does not join at exit (a ThreadPoolExecutor's worker it does)."""
    import threading
    import time
    from types import SimpleNamespace

    m = load_script("04_distill")
    release, evs = threading.Event(), []

    class StalledHub(FakeHub):
        def upload_folder(self, **kw):
            release.wait(30)
            super().upload_folder(**kw)

    up = m.Uploader(StalledHub(), "u/r", "run", True, SimpleNamespace(event=lambda kind, **kw: evs.append(kind)),
                    retries=())
    d = tmp_path / "step_1"
    d.mkdir()
    (d / "w.bin").write_bytes(b"x")
    try:
        up.submit(d, "step_1")
        t0 = time.monotonic()
        assert up.wait(0.3) == {"step_1": None} and up.busy() == {d}
        up.shutdown(0.3)
        assert time.monotonic() - t0 < 5 and up.worker.daemon and up.worker.is_alive()
    finally:
        release.set()
    up.worker.join(5)
    assert not up.worker.is_alive() and up.wait(0) == {"step_1": True} and evs == ["ckpt_upload_ok"]


def test_a_crash_closes_the_logs_before_waiting_on_uploads(tmp_path, monkeypatch):
    """A crash with checkpoint uploads pending (the 8.6 GB pre_cooldown full state, weights): _close_failed waited for
    all of them before the partial summary and the forced sync, and the supervisor's resume waited too. The logs now
    close first; the uploads get FAILED_UPLOAD_WAIT_S, the queued ones are cancelled, the unfinished ones named."""
    import copy
    import threading
    import time
    from types import SimpleNamespace

    m = load_script("04_distill")
    monkeypatch.setattr(m, "FAILED_UPLOAD_WAIT_S", 0.3)
    release, order = threading.Event(), []

    class StalledHub(FakeHub):
        def upload_folder(self, **kw):
            release.wait(30)
            super().upload_folder(**kw)

    log = SimpleNamespace(event=lambda kind, **kw: order.append((kind, kw.get("names") or kw.get("name"))),
                          close=lambda summary: order.append(("close", summary["status"])), elapsed=lambda: 1.0)
    R = m.Run(cfg=copy.deepcopy(m.DEFAULTS), run_dir=tmp_path / "run", device=torch.device("cpu"), amp=False, log=log)
    R.uploader = m.Uploader(StalledHub(), "u/r", "run", True, log, retries=())
    for name in ("full_step_7", "step_7"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "w.bin").write_bytes(b"x")
        R.uploader.submit(tmp_path / name, name)
    t0 = time.monotonic()
    try:
        m._close_failed(R, "failed", RuntimeError("CUDA error: an illegal instruction was encountered"))
        assert time.monotonic() - t0 < 5 and R.uploader.worker.daemon
        assert order == [("close", "failed"), ("ckpt_upload_abandoned", ["full_step_7", "step_7"])]
    finally:
        release.set()
    R.uploader.worker.join(5)  # the running upload finishes; the queued one was cancelled
    assert not R.uploader.worker.is_alive() and order[2:] == [("ckpt_upload_ok", "full_step_7")]


def test_a_resume_sets_newer_states_aside_and_uploads_the_pre_cooldown_state_again(tmp_path, monkeypatch):
    """--resume from an older full state moves the abandoned attempt's newer ones aside (set_aside_newer). The
    pre_cooldown full state is uploaded only by the process that saved it (finish.py's syncs take the newest full
    state, the post-crash one none), so a crash that cut its upload short lost it: a resume from it, or from a later
    state, queues it again."""
    m = load_script("04_distill")
    hub = FakeHub()
    monkeypatch.setattr(m, "hf_api", lambda: hub)
    cfg = m._merge(m.DEFAULTS, {"student": str(tmp_path / "student"), "runs_root": str(tmp_path / "runs"),
                                "device": "cpu", "hf": {"output_repo": "u/r"}, "log": {"capture_env": False}})
    ck = tmp_path / "runs" / "run" / "checkpoints"
    for step in (6, 8):  # 6: the pre_cooldown state (its own trainer.pt names it), 8: the attempt backed out of
        st = m.Run(cfg=cfg, run_dir=ck.parent, device=torch.device("cpu"), amp=False).st
        st.update(step=step, pre_cooldown_done=True, pre_cooldown_full="full_step_6")
        (ck / f"full_step_{step}").mkdir(parents=True)
        torch.save(dict(format=1, step=step, cfg=cfg, st=st, logger=None), ck / f"full_step_{step}" / "trainer.pt")
    R, _ = m.build(m.parse_args(["--resume", str(ck / "full_step_6")]))
    try:
        assert R.uploader.wait(10) == {"full_step_6": True}
        assert [f[0] for f in hub.folders] == ["runs/run/checkpoints/full_step_6"]
        res = next(e for e in events(ck.parent) if e["kind"] == "resume")
        assert res["set_aside"] == ["full_step_8"] and res["upload_again"] == "full_step_6"
        assert m.find_full_state(ck.parent) == ck / "full_step_6"
    finally:
        R.uploader.shutdown(5)
        R.log.close()


def test_hf_roundtrip_retries_a_transient_hub_error(tmp_path):
    """The smoke round trip's create_repo and commit POSTs are sent once by the hub: one 503 must be retried (on the
    uploads' schedule) instead of stopping the paid run after its bootstrap; bad credentials fail at once."""
    from types import SimpleNamespace

    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    m = load_script("04_distill")

    class FlakyHub(FakeHub):
        def __init__(self, status):
            super().__init__()
            self.status, self.commits = status, 0

        def upload_file(self, **kw):
            self.commits += 1
            if self.commits == 1:
                resp = httpx.Response(self.status, request=httpx.Request("POST", "https://hf.invalid/commit/main"))
                raise HfHubHTTPError(f"{self.status} from the Hub", response=resp)
            super().upload_file(**kw)

    def roundtrip(hub):
        evs = []
        R = SimpleNamespace(uploader=SimpleNamespace(api=hub, retries=(0, 0)), run_dir=tmp_path / "run",
                            cfg={"hf": {"output_repo": "u/r", "private": True}},
                            log=SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw))))
        R.run_dir.mkdir(exist_ok=True)
        return m.hf_roundtrip(R), evs

    hub = FlakyHub(503)
    out, evs = roundtrip(hub)
    assert out["ok"] and out["attempt"] == 1 and hub.commits == 2
    assert [(e["kind"], e.get("attempt")) for e in evs] == [("smoke_hf_roundtrip_error", 0), ("smoke_hf_roundtrip", 1)]
    assert "503" in evs[0]["error"]
    hub = FlakyHub(401)
    with pytest.raises(HfHubHTTPError):
        roundtrip(hub)
    assert hub.commits == 1  # no retry on bad credentials


def test_trainer_hub_client_has_a_timeout():
    """huggingface_hub's shared client has no timeout by default: a commit POST the Hub accepted and never answered
    would keep the trainer, and a paid instance, up until the watchdog. hf_api() bounds it and keeps the hub's hook."""
    import huggingface_hub
    from huggingface_hub.utils import _http

    m = load_script("04_distill")
    try:
        m.hf_api()
        c = _http.get_session()
        assert (c.timeout.connect, c.timeout.read) == (60, 300)
        assert _http.hf_request_event_hook in c.event_hooks["request"] and c.follow_redirects
    finally:
        huggingface_hub.set_client_factory(_http.default_client_factory)


def test_configs_resolve():
    m = load_script("04_distill")
    via = m.load_config(str(ROOT / "configs" / "viability.json"), [])
    # the file spells out the defaults, turns early stopping on (off in DEFAULTS: a config that does not mention it
    # trains to its budget, as before early stopping existed) and sets the eval cadence: a full eval (complete eval
    # sets) at every epoch end, a mini eval every 200 steps, a greedy decode of ~600 s of the probe for the train CER
    # (all off in DEFAULTS: a config that does not mention them evaluates as before)
    assert not m.DEFAULTS["early_stop"]["enabled"] and via["early_stop"]["enabled"]
    ev_via = dict(m.DEFAULTS["eval"], full_every_epochs=1, probe_greedy_audio_s=600,
                  mini=dict(every_steps=200, val_per_set=32, train_utts=64, greedy=True))
    assert via == dict(m.DEFAULTS, early_stop=dict(m.DEFAULTS["early_stop"], enabled=True), eval=ev_via)
    assert via["eval"]["every_min"] == 20 and via["eval"]["gate"] is True  # every_min: the fallback cadence only
    assert m.epoch_cadence(via) == 1 and m.DEFAULTS["eval"]["mini"]["every_steps"] is None
    smoke = m.load_config(str(ROOT / "configs" / "smoke_laptop.json"), ["hf.output_repo=u/r", "seed=7"])
    assert smoke["student"] == "students/b4x2560-d2" and smoke["schedule"]["train_hours"] == 0.1
    assert smoke["batch"]["micro_audio_s"] == 60 and smoke["batch"]["step_audio_s"] == 120
    assert smoke["eval"]["greedy_subset"] == 20 and smoke["eval"]["every_min"] == 2
    assert smoke["perf"]["num_workers"] == 2 and smoke["hf"]["output_repo"] == "u/r" and smoke["seed"] == 7
    assert smoke["loss"] == m.DEFAULTS["loss"]  # untouched sections keep the viability values
    with pytest.raises(SystemExit):
        m.load_config(None, ["optim.lrr=1"])
    for key in ("sources", "eval_sets", "teacher_root", "second_root", "data_root", "selection", "student"):
        assert key in json.loads((ROOT / "configs" / "viability.json").read_text(encoding="utf-8"))  # bootstrap.sh


def test_throughput_floor_exits_3(env, hub):
    m = load_script("04_distill")
    rc = m.main(["--config", str(env["config"]), "--set", "run_name=tiny-slow", "--set", "smoke.min_audio_s_per_s=1e9",
                 "--set", "eval.every_steps=1000", "--set", "hf.output_repo=null"])
    assert rc == 3
    run = next((env["root"] / "runs").glob("tiny-slow-*"))
    kinds = [e["kind"] for e in events(run)]
    assert "throughput_too_low" in kinds and kinds[-1] == "logger_close"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "throughput_too_low" and summary["steps"] == 4


def test_padded_row_gate_fails_the_smoke(env, hub):
    """The padded-vs-alone gate is the mean KL over the shortest smoke.pad_utts utterances' positions, not one short
    utterance's argmax (a 3-5 token utterance flips on one bf16 near-tie); a violated bound stops the run."""
    m = load_script("04_distill")
    with pytest.raises(m.SmokeFailed, match="padded rows differ"):
        m.main(["--config", str(env["config"]), "--set", "run_name=tiny-padgate", "--set", "smoke.pad_max_mean_kl=-1",
                "--set", "hf.output_repo=null"])
    run = next((env["root"] / "runs").glob("tiny-padgate-*"))
    pad = next(e for e in events(run) if e["kind"] == "smoke_padded_row")
    assert pad["ok"] is False and pad["n_utts"] == 32 and pad["n_tok"] > 32 and pad["kl_mean"] < 1e-6
    assert pad["kl_bound"] == -1 and "bf16_noise_kl_mean" not in pad  # fp32: no noise floor to measure


def test_a_source_that_cannot_be_decoded_fails_the_smoke(env, hub, monkeypatch):
    """The dataset drops an undecodable row and goes on (one bad upstream file must not end a paid run), so a decode
    failure that hits a whole source - a codec or the resampler broken on a new box - must stop the smoke instead of
    thinning the data for the whole run: decode_preflight names the set before anything else in the smoke phase, and
    a failure only the loader hits is caught by the share of dropped rows over the smoke steps."""
    from kitsune import trainset

    m = load_script("04_distill")
    real_bytes, real_loader = trainset.AudioBatchDataset.audio_bytes, trainset.make_loader

    def audio_bytes(self, i):  # src_b's bytes are no audio: decode_audio raises on every row, as on a broken codec
        return b"not audio" if self.sources[i] == "src_b" else real_bytes(self, i)

    with monkeypatch.context() as mp:
        mp.setattr(trainset.AudioBatchDataset, "audio_bytes", audio_bytes)
        with pytest.raises(m.SmokeFailed, match=r"audio decode fails for whole sets: train src_b \(8/8 rows"):
            m.main(["--config", str(env["config"]), "--set", "run_name=tiny-nodecode", "--set", "hf.output_repo=null"])
    ev = events(next((env["root"] / "runs").glob("tiny-nodecode-*")))
    sets = next(e for e in ev if e["kind"] == "smoke_decode")["sets"]
    assert sets["train/src_b"]["failed"] == sets["train/src_b"]["n"] == 8 and sets["train/src_b"]["first_error"]
    assert sets["train/src_a"] == dict(n=8, failed=0, first_error=None)
    assert all(sets[f"eval/{s}"] == dict(n=6, failed=0, first_error=None) for s in EVAL)
    assert "memory_probe" not in {e["kind"] for e in ev}  # first thing in the smoke phase

    def lossy_loader(*a, **k):  # the workers lose a row of every micro-batch; the main process decodes fine
        inner = real_loader(*a, **k)
        try:
            for key, mbs in inner:
                for mb in mbs:
                    mb["dropped"] = [*mb["dropped"], "src_b/fake/undecodable.flac"]
                yield key, mbs
        finally:
            inner.close()

    monkeypatch.setattr(trainset, "make_loader", lossy_loader)
    with pytest.raises(m.SmokeFailed, match="had undecodable audio over the smoke steps"):
        m.main(["--config", str(env["config"]), "--set", "run_name=tiny-lossy", "--set", "hf.output_repo=null",
                "--set", "eval.every_steps=1000"])
    ev = events(next((env["root"] / "runs").glob("tiny-lossy-*")))
    smoke = next(e for e in ev if e["kind"] == "smoke_steps")
    n_drop = sum(e["n"] for e in ev if e["kind"] == "dropped_audio")
    assert smoke["steps"] == 4 and smoke["dropped"] == n_drop >= 4 and smoke["dropped_frac"] > 0.01
    assert not any(e["kind"] == "checkpoint" and e.get("reason") == "after_smoke" for e in ev)


def test_skipped_steps_and_dropped_audio_are_logged_with_ids(env, hub, monkeypatch):
    """A non-finite gradient (injected at step 2) and an undecodable file (reported at step 1) leave events that name
    the utterances, so they can be found afterwards."""
    m = load_script("04_distill")
    orig_step, orig_obj, seen = m.train_step, m.kd_objective, {"nan": False, "done": False}

    def train_step(R, step, lr, mbs, epoch):
        seen["nan"] = step == 2 and not seen["done"]
        seen["done"] |= seen["nan"]
        if step == 1:
            mbs[0]["dropped"] = ["src_a/fake/undecodable.flac"]
        return orig_step(R, step, lr, mbs, epoch)

    monkeypatch.setattr(m, "train_step", train_step)
    monkeypatch.setattr(m, "kd_objective", lambda *a: orig_obj(*a) * (float("nan") if seen["nan"] else 1.0))
    assert m.main(["--config", str(env["config"]), "--set", "run_name=tiny-skip", "--set", "smoke.enabled=false",
                   "--set", "schedule.max_steps=4", "--set", "eval.every_steps=1000", "--set", "hf.output_repo=null",
                   "--set", "ckpt.weights_every_steps=1000", "--set", "ckpt.full_every_steps=1000",
                   "--set", "eval.final_full_greedy=false"]) == 0
    run = next((env["root"] / "runs").glob("tiny-skip-*"))
    ev = events(run)
    drop = next(e for e in ev if e["kind"] == "dropped_audio")
    assert drop["at_step"] == 1 and drop["ids"] == ["src_a/fake/undecodable.flac"]
    skip = next(e for e in ev if e["kind"] == "nonfinite_grad_skipped")
    ids = [i for mb in skip["ids"] for i in mb]
    assert skip["at_step"] == 2 and ids and all(i.startswith(("src_a/", "src_b/")) for i in ids)
    assert len(skip["mb_kl_sum"]) == len(skip["mb_n_tok"]) == len(skip["ids"])
    assert all(map(math.isfinite, skip["mb_kl_sum"]))
    assert skip["nonfinite_ids"] == []  # the losses were finite; only the gradient was not
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["skipped"]["nonfinite"] == 1 and summary["steps"] == 4


def test_fit_budget_clips_T_to_the_instance_deadline(monkeypatch, tmp_path):
    """train_hours counts loop time; the watchdog counts from first boot. T is clipped so the final eval (scaled from
    the last eval) and the end reserve fit before the deadline; without a deadline T = train_hours."""
    import types
    import time

    m = load_script("04_distill")
    evs = []
    R = m.Run(cfg=m.load_config(None, ["schedule.end_reserve_min=10"]), run_dir=tmp_path, device=torch.device("cpu"),
              amp=False)
    R.log = types.SimpleNamespace(event=lambda kind, **kw: evs.append(dict(kind=kind, **kw)))
    R.evalstore = [None] * 1000
    R.st.update(train_s=100.0, eval_cost=dict(tf_s=30.0, probe_s=10.0, greedy_s=60.0, greedy_n=100))
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    monkeypatch.setenv("KITSUNE_DEADLINE", str(time.time() + 3600))
    m.fit_budget(R)
    # reserve = 10 min + 30 + 10 + 60 * 1000 / 100 = 1240 s; T = 100 s already trained + 3600 - 1240
    assert R.progress() == (100.0, pytest.approx(2460, abs=5)) and evs[-1]["clipped"] and evs[-1]["kind"] == "budget"
    assert m.wsd_lr(1.0, 500, 0.81 * 2460, R.progress()[1], 10, 0.2)[1] == 2  # the cooldown moved with T

    monkeypatch.delenv("KITSUNE_DEADLINE")
    monkeypatch.setenv("KITSUNE_STATE", str(tmp_path))  # vast/onstart.sh's file: plenty of time left
    (tmp_path / "deadline").write_text(f"{int(time.time()) + 10 * 3600}\n", encoding="utf-8")
    m.fit_budget(R)
    assert R.budget_s is None and R.progress()[1] == 4 * 3600 and not evs[-1]["clipped"]
    monkeypatch.delenv("KITSUNE_STATE")
    n = len(evs)
    m.fit_budget(R)  # off the box: no deadline, no event
    assert R.budget_s is None and len(evs) == n


def test_shm_cap_fits_workers_into_dev_shm(monkeypatch, tmp_path):
    import types

    m = load_script("04_distill")

    def usage(free):
        return lambda _p: types.SimpleNamespace(total=free, used=0, free=free)

    monkeypatch.setattr(m.shutil, "disk_usage", usage(2 * 2**30))
    assert m.shm_cap(8, 4, 400, shm=str(tmp_path)) == (8, 4, None)  # 32 micro-batches x 32 MB < half of 2 GiB
    monkeypatch.setattr(m.shutil, "disk_usage", usage(64 * 2**20))  # Docker's default /dev/shm
    n, p, info = m.shm_cap(8, 4, 400, shm=str(tmp_path))
    assert (n, p) == (1, 1) and info["workers"] == [8, 1] and info["prefetch"] == [4, 1]
    monkeypatch.setattr(m.shutil, "disk_usage", usage(16 * 2**20))
    assert m.shm_cap(8, 4, 400, shm=str(tmp_path))[:2] == (0, 1)  # decode in-process
    assert m.shm_cap(8, 4, 400, shm=str(tmp_path / "missing")) == (8, 4, None)  # no /dev/shm (Windows)


@pytest.fixture(scope="module")
def crash_resume(env):
    """The run that dies at a simulated crash at step 13 (KITSUNE_CRASH_AT_STEP) and is resumed from its full state at
    step 10 to the end, uploading to a FakeHub. Module-scoped, so every test that reads it gets the run whichever test
    runs first (or alone); what the crashed launch left behind is kept here, since the resume changes it."""
    import huggingface_hub

    hub = FakeHub()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(huggingface_hub, "HfApi", lambda *a, **k: hub)
        mp.setenv("KITSUNE_CRASH_AT_STEP", "13")
        with pytest.raises(RuntimeError, match="simulated crash"):
            load_script("04_distill").main(["--config", str(env["config"])])
        (run,) = (env["root"] / "runs").glob("tiny-2*")
        ck = run / "checkpoints"
        crashed = dict(events=events(run), summary=json.loads((run / "summary.json").read_text(encoding="utf-8")),
                       fulls=sorted(p.name for p in ck.glob("full_step_*")),
                       weights={p.name for p in ck.glob("step_*")},
                       utts=pd.concat([pd.read_parquet(p)
                                       for p in sorted((run / "metrics" / "train_utts").glob("part-*.parquet"))]))
        mp.delenv("KITSUNE_CRASH_AT_STEP")
        rc = load_script("04_distill").main(["--config", str(env["config"]), "--resume", str(ck / "full_step_10")])
    return dict(run=run, hub=hub, crashed=crashed, rc=rc)


def test_train_crash_resume_export(crash_resume, tmp_path):
    run, hub, crashed = crash_resume["run"], crash_resume["hub"], crash_resume["crashed"]
    ev1 = crashed["events"]
    assert [e for e in ev1 if e["kind"] == "exception"][0]["type"] == "RuntimeError"
    kinds1 = [e["kind"] for e in ev1]
    assert "sync_ok" in kinds1[kinds1.index("logger_close"):]  # the forced sync ran after the crash
    assert crashed["summary"]["status"] == "failed"
    ck = run / "checkpoints"
    # full states after the smoke steps (4) and every 5 steps; keep_local=2 leaves 5 and 10
    assert crashed["fulls"] == ["full_step_10", "full_step_5"]
    assert {"step_5", "step_10"} <= crashed["weights"]
    utts1 = crashed["utts"]
    assert crash_resume["rc"] == 0  # the resumed launch ran to the end

    # ---- the section 7 layout
    for rel in ["config.json", "env/git_sha.txt", "env/git_diff.patch", "env/pip_freeze.txt", "env/nvidia_smi.txt",
                "env/system.json", "env/sdpa_backends.json", "metrics/scalars.jsonl", "metrics/scalars.parquet",
                "metrics/steps.parquet", "events.jsonl", "logs/stdout.log", "summary.json",
                "evals/step_0/summary.json", "evals/step_0/probe.parquet", "samples/step_0.jsonl",
                f"evals/step_{MAX_STEPS}/summary.json", f"evals/step_{MAX_STEPS}/verdict.json"]:
        assert (run / rel).is_file(), rel
    for s in EVAL:
        for step in (0, 5, 10, 15, MAX_STEPS):
            assert (run / "evals" / f"step_{step}" / f"tf_{s}.parquet").is_file()
            assert (run / "evals" / f"step_{step}" / f"greedy_{s}.parquet").is_file()
    assert list((run / "tb").glob("events.out.tfevents.*"))
    assert list((run / "metrics" / "train_utts").glob("part-*.parquet"))
    assert list((run / "metrics" / "hist").glob("part-*.parquet"))
    assert not list(run.rglob("*.tmp"))

    # ---- one step axis across the crash; the schedule continues where it stopped
    steps = pd.read_parquet(run / "metrics" / "steps.parquet")
    assert steps["step"].tolist() == list(range(1, MAX_STEPS + 1))
    m_ref = load_script("04_distill")
    for step, lr, phase in zip(steps["step"], steps["opt/lr"], steps["sched/phase"]):
        want = m_ref.wsd_lr(PEAK, int(step), float(step - 1), float(MAX_STEPS), WARMUP, COOLDOWN)
        assert lr == pytest.approx(want[0]) and phase == want[1], step
    assert steps["sched/phase"].iloc[-1] == 2 and steps["opt/lr"].iloc[-1] > 0
    obj = steps["loss/objective"].to_numpy()
    assert np.isfinite(obj).all() and obj[-3:].mean() < 0.8 * obj[:3].mean(), obj  # it learns
    for col in ["loss/total", "loss/kl", "loss/ce", "loss/l2sp", "tok/top1", "tok/entropy_student",
                "tok/tail_teacher", "opt/grad_norm", "time/data_wait_s", "perf/audio_s_per_s", "perf/tokens_per_s",
                "perf/pad_eff_audio", "perf/mfu", "aug/masked_frac", "data/epoch_progress", "src/src_a/kl",
                "src/src_b/ce", "bucket/p1_gt_0.99/frac"]:
        assert col in steps.columns, col
    assert steps["loss/l2sp"].iloc[-1] > 0 and steps["data/epoch"].max() >= 1  # past an epoch boundary

    # the resumed steps replay exactly (planner position, SpecAugment seeds, weights and optimizer state restored)
    utts = pd.concat([pd.read_parquet(p) for p in sorted((run / "metrics" / "train_utts").glob("part-*.parquet"))])
    for step in (11, 12):
        a = utts1[utts1["step"] == step].sort_values("id")
        b = utts[(utts["step"] == step) & (utts["attempt"] == 1)].sort_values("id")  # the resumed launch's rows
        assert a["id"].tolist() == b["id"].tolist() and len(a) and set(a["attempt"]) == {0}
        assert len(utts[utts["step"] == step]) == 2 * len(a)  # both copies are kept, told apart by `attempt`
        np.testing.assert_allclose(a["kl"].to_numpy(), b["kl"].to_numpy(), rtol=1e-4, atol=1e-6)

    ev2 = events(run)
    kinds = [e["kind"] for e in ev2]
    for k in ("smoke_logmel_vs_hf", "smoke_longest_fwd_bwd", "smoke_padded_row", "smoke_sdpa", "smoke_flops",
              "smoke_hf_roundtrip", "memory_probe", "smoke_steps", "resume", "resumed", "verdict", "logger_close"):
        assert k in kinds, k
    assert next(e for e in ev2 if e["kind"] == "smoke_logmel_vs_hf")["max_abs_diff"] == 0.0  # bitwise on CPU
    pad = next(e for e in ev2 if e["kind"] == "smoke_padded_row")
    assert pad["ok"] and pad["argmax_agree"] == 1.0 and pad["kl_mean"] < 1e-6 and pad["n_utts"] == 32  # fp32: exact
    assert next(e for e in ev2 if e["kind"] == "resumed")["at_step"] == 10
    cool = next(e for e in ev2 if e["kind"] == "phase" and e.get("name") == "cooldown")
    assert cool["at_step"] == int((1 - COOLDOWN) * MAX_STEPS)
    # each LR phase switch once, the resume included (the phase is part of the full state)
    assert [(e["phase"], e["at_step"]) for e in ev2 if e["kind"] == "lr_phase"] == [
        ("warmup", 1), ("stable", WARMUP), ("cooldown", int((1 - COOLDOWN) * MAX_STEPS) + 1)]
    tags = set(pd.read_parquet(run / "metrics" / "scalars.parquet")["tag"])
    assert {"layers/grad_norm/encoder.layers.0", "layers/update_ratio/decoder.layers.0", "l2sp/dist/encoder.layers.1",
            "bn/max_abs_drift", "eval/tf/eval_jsut/kl", "eval/greedy/eval_cv8/cer_ref_corpus",
            "eval/probe/all/kl", "eval/greedy_full/eval_reazon/cer_ref_corpus"} <= tags
    # every tag the trainer logs has a TensorBoard bucket by a rule (none fell through to 3_misc unmatched)
    assert "tb_tag_unmapped" not in kinds
    tag_map = json.loads((run / "metrics" / "tag_map.json").read_text(encoding="utf-8"))
    assert tags <= set(tag_map) and not any(e.get("unmapped") for e in tag_map.values())
    assert tag_map["eval/greedy_full/eval_reazon/cer_ref_corpus"]["tb_tag"] == \
        "2_loss_accuracy/val_accuracy_full/eval_reazon/cer_ref_corpus"
    hist_tags = set(pd.concat([pd.read_parquet(p) for p in (run / "metrics" / "hist").glob("part-*.parquet")])["tag"])
    assert {"weight/encoder.layers.0", "grad/decoder.layers.0", "act/encoder.layers.1", "act/decoder.layers.0"} <= hist_tags

    # ---- final eval, verdict, summary
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["steps"] == MAX_STEPS and summary["resumes"] == 1
    assert summary["verdict"]["verdict"] in ("GO", "PROMISING", "NO-GO", "INCONCLUSIVE")
    assert set(summary["verdict"]["sets"]) == set(EVAL)
    assert [r["step"] for r in summary["history"]] == [0, 5, 10, 15, MAX_STEPS]
    final = json.loads((run / "evals" / f"step_{MAX_STEPS}" / "summary.json").read_text(encoding="utf-8"))
    assert final["final"] and final["greedy_full"]["n_utts"] == 3 * 6 and final["greedy"]["n_utts"] == 3 * 3
    gr = pd.read_parquet(run / "evals" / f"step_{MAX_STEPS}" / "greedy_eval_jsut.parquet")
    assert len(gr) == 6 and gr["in_greedy_subset"].sum() == 3

    # ---- checkpoints: weights load as a student; the full states resume; uploads went where finish.py looks
    from kitsune import student as S

    final_w = S.load_student(ck / f"step_{MAX_STEPS}", "cpu")
    assert S.load_meta(ck / f"step_{MAX_STEPS}")["trained"]["step"] == MAX_STEPS
    assert final_w.proj_out.weight is final_w.model.decoder.embed_tokens.weight
    fulls = sorted(p.name for p in ck.glob("full_step_*"))
    assert f"full_step_{MAX_STEPS}" in fulls and len(fulls) <= 3  # keep_local 2 (+ the state it resumed from)
    uploaded = {p for p, _, _ in hub.folders}
    prefix = f"runs/{run.name}"
    assert f"{prefix}/checkpoints/step_{MAX_STEPS}" in uploaded and f"{prefix}/checkpoints/full_step_{MAX_STEPS}" in uploaded
    assert f"{prefix}/checkpoints/full_step_{int((1 - COOLDOWN) * MAX_STEPS)}" in uploaded  # pre-cooldown
    syncs = [f for p, f, ign in hub.folders if p == prefix]
    assert syncs and all(ign and "checkpoints/*" in ign for p, _, ign in hub.folders if p == prefix)
    assert f"{prefix}/smoke/roundtrip.json" in hub.files

    # ---- export (tools/ is not under scripts/, so no load_script)
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_run", ROOT / "tools" / "export_run.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    tables = exp.export(str(run), tmp_path / "export")
    for name in ("steps", "scalars", "train_utts", "hist", "eval_tf", "eval_greedy", "eval_probe", "eval_summaries",
                 "samples", "events", "tb_scalars", "tb_histograms", "tb_text", "tag_map"):
        assert name in tables and len(tables[name]), name
        assert (tmp_path / "export" / f"{name}.parquet").is_file()
    assert (tmp_path / "export" / "README.md").is_file()
    assert set(tables["tb_scalars"]["tag"]) == set(tables["scalars"]["tag"])
    # the crashed launch's steps after the restored step 10 are marked discarded (the attempt read off events.jsonl);
    # what is left matches TensorBoard, one row per step and utterance, and no eval is taken for a discarded one
    sc, tb, tu = tables["scalars"], tables["tb_scalars"], tables["train_utts"]
    kl = sc[(sc["tag"] == "loss/kl") & ~sc["discarded"]]
    assert sorted(kl["step"]) == sorted(tb.loc[tb["tag"] == "loss/kl", "step"]) == list(range(1, MAX_STEPS + 1))
    gone = tu[tu["discarded"]]
    assert len(gone) and set(gone["attempt"]) == {0} and gone["step"].min() == 11
    assert not tu[~tu["discarded"]].duplicated(["step", "id"]).any()
    assert not tables["eval_tf"]["discarded"].any() and not tables["samples"]["discarded"].any()


def _tool(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regroup_tb_on_a_copy_of_the_run(crash_resume, tmp_path):
    """tools/regroup_tb.py on a copy of the crash + resume run (the crash_resume fixture, made here if no other test
    made it yet): TensorBoard shows the same points after the rebuild (the resume's purge replayed from events.jsonl),
    the original event files are kept under names TensorBoard skips, and a second run rebuilds the same file again
    instead of adding one."""
    import shutil

    from kitsune.runlog import TB_BUCKETS, load_tag_map, read_scalars_jsonl

    src = crash_resume["run"]
    run = tmp_path / "runs" / src.name
    shutil.copytree(src, run, ignore=shutil.ignore_patterns("checkpoints"))
    exp, rg = _tool("export_run"), _tool("regroup_tb")

    def view():
        t = exp.tb_tables(run / "tb", load_tag_map(run / "metrics" / "tag_map.json"))
        return {k: df.drop(columns=["wall_time", *(["sum", "sum_squares"] if k == "tb_histograms" else [])])
                .sort_values(["tag", "step"], kind="stable").reset_index(drop=True) for k, df in t.items()}

    before = view()
    originals = sorted(p.name for p in (run / "tb").iterdir())
    assert len(originals) == 2  # the crashed launch's event file and the resumed launch's
    assert rg.live_reason(run) is None
    res = rg.regroup(run)
    assert res["records"]["purge"] == 1 and res["unmapped"] == []
    after = view()
    for k in before:
        assert len(before[k]) and before[k].equals(after[k]), k
    kl = after["tb_scalars"][after["tb_scalars"]["tag"] == "loss/kl"]
    assert kl["step"].tolist() == list(range(1, MAX_STEPS + 1))  # steps 11-13 of the crashed launch purged
    logged = set(read_scalars_jsonl(run / "metrics" / "scalars.jsonl").column("tag").to_pylist())
    assert sum(res["counts"][b]["scalars"] for b in TB_BUCKETS) == len(logged)
    assert all(res["counts"][b]["scalars"] for b in TB_BUCKETS) and res["counts"]["3_misc"]["histograms"] > 0
    files = sorted(p.name for p in (run / "tb").iterdir())
    assert sorted(f for f in files if f.endswith(".bak")) == sorted(rg.backup_name(f) for f in originals)
    assert len([f for f in files if "tfevents" in f]) == 1

    res2 = rg.regroup(run)
    files2 = sorted(p.name for p in (run / "tb").iterdir())
    assert [f for f in files2 if f.endswith(".bak")] == [f for f in files if f.endswith(".bak")]
    assert len([f for f in files2 if "tfevents" in f]) == 1 and res2["counts"] == res["counts"]
    again = view()
    assert all(again[k].equals(after[k]) for k in after)
