"""kitsune.runlog.RunLogger and tools/export_run.py: every file of the run-dir layout is written, the parquet mirrors
agree with the jsonl they are rebuilt from, HF sync runs in the background without blocking (mocked, no network) and
its failures become events, resume keeps one step axis, and the export turns a tiny run into parquet/CSV + README.
CPU only."""
import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from kitsune import runlog  # noqa: E402
from kitsune.runlog import RunLogger, read_scalars_jsonl, system_stats  # noqa: E402


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    """Env capture probes CUDA (system info, SDPA kernels); keep it on the CPU even if CUDA was initialised by an
    earlier test module in the session."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


CFG = {"run_name": "tiny", "student": "students/none", "hf": {"output_repo": None, "private": True}, "seed": 1}


def load_export():
    spec = importlib.util.spec_from_file_location("export_run", ROOT / "tools" / "export_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeApi:
    """Stands in for huggingface_hub.HfApi: records calls, can be slow or fail."""

    def __init__(self, delay: float = 0.0, fail: int = 0):
        self.delay, self.fail = delay, fail
        self.calls, self.repos = [], []
        self.threads = set()

    def create_repo(self, repo_id, **kw):
        self.repos.append((repo_id, kw))

    def upload_folder(self, **kw):
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        if self.fail:
            self.fail -= 1
            raise ConnectionError("simulated HF outage")
        self.calls.append(dict(kw, files=sorted(p.relative_to(kw["folder_path"]).as_posix()
                                                for p in Path(kw["folder_path"]).rglob("*") if p.is_file())))


def events(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]


def fill(log: RunLogger, steps=range(1, 6)):
    """A miniature training run: every kind of record the trainer writes."""
    g = torch.Generator().manual_seed(0)
    for s in steps:
        row = {"loss/total": 2.0 / s, "loss/kl": 1.0 / s, "lr": 1e-4 * s, "grad_norm": 0.5}
        if s % 2 == 0:
            row["loss/kl/src_a"] = 0.9 / s  # a column that is only present on some steps
        if s == 3:
            row["grad_norm"] = float("nan")
            row["loss/ce"] = float("inf")
        log.step_row(row, s)
        log.scalars(system_stats(), s)
        log.train_utts([dict(step=s, epoch=0, id=f"src_a/{s}/{j}", source="src_a", duration=1.5 + j, n_tok=6,
                             kl=0.1 * j, ce=0.2, top1_acc=0.9, masked_frac=0.05, agree=0.0, extra_col=j)
                        for j in range(3)])
    log.hist("weights/enc0", torch.randn(10_000, generator=g), 5)
    log.hist("grads/enc0", torch.tensor([1.0, float("nan"), 2.0, float("inf"), 3.0]), 5)
    log.text("note", "hello **world**", 5)
    tf = pd.DataFrame(dict(id=["a", "b"], source=["eval_jsut"] * 2, n_tok=[3, 4], kl=[0.1, 0.2], ce=[0.3, 0.4],
                           top1=[1.0, 0.5], duration=[1.0, 2.0]))
    gr = pd.DataFrame(dict(id=["a"], source=["eval_jsut"], duration=[1.0], ref=["あい"], teacher_hyp=["あい"],
                           hyp=["あう"], cer_ref=[0.5], cer_teacher=[0.5], truncated=[False], n_tok=[3],
                           hyp_ids=[[5, 6, 3]]))
    log.table("tf_eval_jsut", tf, 5)
    log.table("greedy_eval_jsut", gr, 5)
    log.table("probe", tf.assign(source="src_a"), 5)
    log.eval_json("summary", {"sets": {"eval_jsut": {"kl": 0.15, "cer_ref_corpus": 0.5}}, "wall_s": 1.0}, 5)
    log.samples(5, gr.drop(columns=["hyp_ids"]).to_dict("records"))
    log.event("phase", name="train")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception(where="step 5")
    print("to stdout")
    print("to stderr", file=sys.stderr)


@pytest.fixture()
def run(tmp_path):
    log = RunLogger(tmp_path / "runs" / "tiny-run", CFG, sync_every_min=10)
    fill(log)
    log.close(summary={"verdict": "GO", "steps": 5})
    return tmp_path / "runs" / "tiny-run"


def test_layout_every_file(run):
    for rel in ["config.json", "env/git_sha.txt", "env/git_diff.patch", "env/pip_freeze.txt", "env/nvidia_smi.txt",
                "env/system.json", "env/sdpa_backends.json", "metrics/scalars.jsonl", "metrics/scalars.parquet",
                "metrics/steps.parquet", "metrics/text.jsonl", "evals/step_5/summary.json",
                "evals/step_5/tf_eval_jsut.parquet", "evals/step_5/greedy_eval_jsut.parquet", "evals/step_5/probe.parquet",
                "samples/step_5.jsonl", "events.jsonl", "logs/stdout.log", "summary.json"]:
        assert (run / rel).is_file(), rel
    assert list((run / "tb").glob("events.out.tfevents.*"))
    assert list((run / "metrics" / "train_utts").glob("part-*.parquet"))
    assert list((run / "metrics" / "hist").glob("part-*.parquet"))
    assert not list(run.rglob("*.tmp"))

    conf = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert conf["config"] == CFG and conf["run_id"] == "tiny-run" and "argv" in conf
    system = json.loads((run / "env" / "system.json").read_text(encoding="utf-8"))
    assert system["versions"]["torch"] and system["cpu_count"] and "libsndfile" in system
    assert len((run / "env" / "git_sha.txt").read_text().split()[0]) == 40
    assert "torch==" in (run / "env" / "pip_freeze.txt").read_text(encoding="utf-8")
    assert json.loads((run / "summary.json").read_text(encoding="utf-8")) == {"verdict": "GO", "steps": 5}

    log_txt = (run / "logs" / "stdout.log").read_text(encoding="utf-8")
    assert "to stdout" in log_txt and "to stderr" in log_txt and "[event] phase" in log_txt
    kinds = [e["kind"] for e in events(run)]
    assert kinds[0] == "logger_start" and kinds[-1] == "logger_close" and "phase" in kinds
    exc = next(e for e in events(run) if e["kind"] == "exception")
    assert exc["type"] == "RuntimeError" and "boom" in exc["traceback"] and exc["where"] == "step 5"
    assert all({"wall", "time", "elapsed_s", "step", "kind"} <= set(e) for e in events(run))


def test_scalars_parquet_matches_jsonl(run):
    lines = [json.loads(line) for line in (run / "metrics" / "scalars.jsonl").read_text(encoding="utf-8").splitlines()]
    pq_df = pd.read_parquet(run / "metrics" / "scalars.parquet")
    assert list(pq_df.columns) == ["tag", "step", "wall", "elapsed_s", "value"]
    assert len(pq_df) == len(lines)
    for row, line in zip(pq_df.itertuples(), lines):
        assert (row.tag, row.step, row.wall, row.elapsed_s) == (line["tag"], line["step"], line["wall"], line["elapsed_s"])
        want = float(line["nf"]) if "nf" in line else line["value"]
        assert (math.isnan(row.value) and math.isnan(want)) or row.value == want
    # non-finite values survive the JSON round trip
    by = pq_df.set_index(["tag", "step"])["value"]
    assert math.isnan(by[("grad_norm", 3)]) and by[("loss/ce", 3)] == math.inf and by[("loss/total", 4)] == 0.5
    assert pq_df.equals(read_scalars_jsonl(run / "metrics" / "scalars.jsonl").to_pandas())
    assert any(t.startswith("sys/proc/") for t in pq_df["tag"])  # system_stats degrades gracefully without NVML


def test_steps_parquet_is_wide(run):
    st = pd.read_parquet(run / "metrics" / "steps.parquet")
    assert st["step"].tolist() == [1, 2, 3, 4, 5]
    assert {"wall", "elapsed_s", "loss/total", "loss/kl/src_a", "loss/ce"} <= set(st.columns)
    assert st["loss/kl/src_a"].isna().tolist() == [True, False, True, False, True]  # absent on odd steps
    assert st["loss/total"].tolist() == pytest.approx([2.0 / s for s in range(1, 6)])
    assert (st["elapsed_s"].diff().dropna() >= 0).all()


def test_train_utts_and_hist_parts(run):
    tu = pd.concat([pd.read_parquet(p) for p in sorted((run / "metrics" / "train_utts").glob("part-*.parquet"))])
    assert len(tu) == 15 and tu["id"].is_unique
    schema = pq.read_schema(next((run / "metrics" / "train_utts").glob("part-*.parquet")))
    assert str(schema.field("duration").type) == "float" and str(schema.field("n_tok").type) == "int32"
    assert "extra_col" in tu.columns  # unknown keys are kept, not dropped

    h = pd.concat([pd.read_parquet(p) for p in (run / "metrics" / "hist").glob("part-*.parquet")]).set_index("tag")
    w = h.loc["weights/enc0"]
    x = torch.randn(10_000, generator=torch.Generator().manual_seed(0)).numpy()
    assert w["n"] == 10_000 and w["min"] == pytest.approx(x.min()) and w["max"] == pytest.approx(x.max())
    assert w["p50"] == pytest.approx(np.quantile(x, 0.5), abs=1e-4) and w["p99"] == pytest.approx(np.quantile(x, 0.99), abs=1e-4)
    assert w["mean"] == pytest.approx(x.mean(), abs=1e-6) and w["std"] == pytest.approx(x.std(), rel=1e-5)
    counts, edges = json.loads(w["counts"]), json.loads(w["edges"])
    assert len(counts) == 64 and len(edges) == 65 and sum(counts) == 10_000
    g = h.loc["grads/enc0"]
    assert g["n"] == 3 and g["n_nonfinite"] == 2 and g["max"] == 3.0


def test_sync_runs_in_background_and_excludes_checkpoints(tmp_path):
    api = FakeApi(delay=1.5)
    run = tmp_path / "bg-run"
    log = RunLogger(run, CFG, hf_repo="me/kitsune-runs", sync_every_min=0, api=api, capture=False, tee=False)
    (run / "checkpoints" / "step_1").mkdir(parents=True)
    (run / "checkpoints" / "step_1" / "model.safetensors").write_bytes(b"x" * 10)
    log.step_row({"loss": 1.0}, 1)
    t0 = time.monotonic()
    assert log.sync() is True
    assert time.monotonic() - t0 < 0.5  # the 1.5 s upload runs in the background
    log.step_row({"loss": 0.5}, 2)  # training continues meanwhile
    assert log.sync() is False  # the previous sync is still running: skipped, not queued
    log.wait_sync()
    assert api.threads == {"runlog-sync"} and len(api.calls) == 1
    call = api.calls[0]
    assert call["repo_id"] == "me/kitsune-runs" and call["path_in_repo"] == "runs/bg-run"
    assert "checkpoints/*" in call["ignore_patterns"]
    assert api.repos == [("me/kitsune-runs", dict(repo_type="model", private=True, exist_ok=True))]
    assert "metrics/steps.parquet" in call["files"] and "events.jsonl" in call["files"]
    log.close()
    assert len(api.calls) == 2  # the final forced sync waited for its upload
    assert pd.read_parquet(run / "metrics" / "steps.parquet")["step"].tolist() == [1, 2]
    assert [e["kind"] for e in events(run)].count("sync_ok") == 2


def test_sync_uploads_a_snapshot_the_trainer_cannot_change(tmp_path):
    """The Hub client sizes a file when it lists the folder and reads it later; the trainer keeps appending to its
    logs meanwhile. The logger uploads a copy with the append-only files cut at the lengths flushed under its lock."""
    run = tmp_path / "snap-run"
    seen = {}

    class GrowingApi(FakeApi):
        def upload_folder(self, **kw):
            folder = Path(kw["folder_path"])
            def read():
                return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

            before = read()
            log.step_row({"late": 1.0}, 99)  # the training thread goes on during the upload (and flushes)
            log.event("late_event")
            log.tb.flush()
            seen.update(folder=folder, before=before, after=read())
            super().upload_folder(**kw)

    log = RunLogger(run, CFG, hf_repo="me/kitsune-runs", sync_every_min=0, api=GrowingApi(), capture=False, tee=False)
    (run / "checkpoints" / "full_step_1").mkdir(parents=True)
    (run / "checkpoints" / "full_step_1" / "model.pt").write_bytes(b"x" * 10)
    log.step_row({"loss": 1.0}, 1)
    log.sync(force=True)
    assert seen["folder"] != run and run not in seen["folder"].parents
    assert seen["before"] == seen["after"]  # nothing the trainer wrote during the upload reached the uploaded copy
    staged = seen["before"]
    assert not any(k.startswith("checkpoints/") for k in staged)
    live = (run / "metrics" / "scalars.jsonl").read_bytes()
    assert live.startswith(staged["metrics/scalars.jsonl"]) and b'"late"' in live
    assert staged["metrics/scalars.jsonl"].endswith(b"\n") and b'"late"' not in staged["metrics/scalars.jsonl"]
    assert b"late_event" not in staged["events.jsonl"] and b"late_event" in (run / "events.jsonl").read_bytes()
    tb = [k for k in staged if k.startswith("tb/")]
    assert tb and all((run / k).read_bytes().startswith(staged[k]) for k in tb)
    assert {"metrics/steps.parquet", "metrics/scalars.parquet", "config.json"} <= set(staged)
    log.close()


def test_atomic_write_waits_for_a_reader(tmp_path):
    """On Windows a file another thread has open (the sync's snapshot copy) cannot be replaced for a moment; the
    trainer's atomic write must wait for it instead of crashing the run."""
    p = tmp_path / "summary.json"
    p.write_text("{}", encoding="utf-8")
    reader = open(p, "rb")
    threading.Timer(0.3, reader.close).start()
    runlog._atomic_json(p, {"new": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"new": 1}


def test_sync_errors_become_events(tmp_path):
    api = FakeApi(fail=10)  # every attempt fails
    run = tmp_path / "err-run"
    log = RunLogger(run, CFG, hf_repo="me/kitsune-runs", sync_every_min=0, api=api, capture=False, tee=False,
                    upload_retries=(0.0, 0.0))
    log.scalar("loss", 1.0, 1)
    log.sync(force=True)  # must not raise
    ev = events(run)
    errs = [e for e in ev if e["kind"] == "sync_error"]
    assert len(errs) == 3 and all("simulated HF outage" in e["error"] for e in errs)  # first try + 2 retries
    assert [e["attempt"] for e in errs] == [0, 1, 2] and ev[-1]["kind"] == "sync_failed"
    api.fail = 1  # next sync: one failure, then success on the retry
    log.sync(force=True)
    assert events(run)[-1]["kind"] == "sync_ok" and events(run)[-1]["attempt"] == 1
    log.close()
    assert (run / "metrics" / "scalars.parquet").exists()  # local mirrors are written even when uploads fail


def test_resume_keeps_one_step_axis(tmp_path):
    run = tmp_path / "res-run"
    log = RunLogger(run, CFG, sync_every_min=0, capture=False, tee=False)
    state = None
    for s in range(1, 6):
        log.step_row({"loss": 1.0 / s}, s)
        if s == 3:
            state = log.state_dict()  # the trainer's full-state save at step 3
    log.close()
    time.sleep(0.05)
    log2 = RunLogger(run, CFG, sync_every_min=0, capture=False, tee=False, resume=state)
    assert log2.elapsed() >= state["elapsed_s"]
    for s in range(4, 7):
        log2.step_row({"loss": 10.0 / s}, s)
    log2.close()
    st = pd.read_parquet(run / "metrics" / "steps.parquet")
    assert st["step"].tolist() == [1, 2, 3, 4, 5, 6]  # the lost steps 4-5 are replaced, not duplicated
    assert st["loss"].tolist()[3:] == pytest.approx([10 / 4, 10 / 5, 10 / 6])
    assert (st["elapsed_s"].diff().dropna() >= 0).all()
    starts = [e for e in events(run) if e["kind"] == "logger_start"]
    assert len(starts) == 2 and starts[1]["resume"]["step"] == 3 and starts[1]["restart"] is True
    assert len(list(run.glob("config*.json"))) == 2  # the original config.json is kept
    # the jsonl is append-only: both copies of steps 4-5 are there, the later wall wins
    sc = read_scalars_jsonl(run / "metrics" / "scalars.jsonl").to_pandas()
    assert (sc["step"] == 4).sum() == 2


def test_state_dict_persists_rows_before_a_crash(tmp_path):
    """A crash after a full-state save must not lose the wide step rows or buffered parts up to that step, even though
    no sync ran since (the process dies without close())."""
    run = tmp_path / "crash-run"
    log = RunLogger(run, CFG, sync_every_min=60, capture=False, tee=False)
    for s in range(1, 4):
        log.step_row({"loss": 1.0 / s}, s)
        log.train_utts([dict(step=s, epoch=0, id=f"u{s}", source="src_a", duration=1.0, n_tok=3, kl=0.1, ce=0.2,
                             top1_acc=1.0, masked_frac=0.0, agree=0.0)])
    log.hist("w", torch.ones(10), 3)
    state = log.state_dict()
    log.step_row({"loss": 0.2}, 4)  # logged after the save, then the process is killed: no close(), no sync
    log.tb.close()
    log._f_scalars.close()
    log._f_text.close()
    log._f_log.close()
    log2 = RunLogger(run, CFG, sync_every_min=60, capture=False, tee=False, resume=state)
    log2.step_row({"loss": 0.1}, 4)
    log2.close()
    assert pd.read_parquet(run / "metrics" / "steps.parquet")["step"].tolist() == [1, 2, 3, 4]
    tu = pd.concat([pd.read_parquet(p) for p in (run / "metrics" / "train_utts").glob("part-*.parquet")])
    assert sorted(tu["id"]) == ["u1", "u2", "u3"]
    assert len(list((run / "metrics" / "hist").glob("part-*.parquet"))) == 1


def test_torn_jsonl_tail_is_repaired(tmp_path):
    run = tmp_path / "torn"
    log = RunLogger(run, CFG, sync_every_min=0, capture=False, tee=False)
    log.scalar("a", 1.0, 1)
    log.close()
    with open(run / "metrics" / "scalars.jsonl", "a", encoding="utf-8") as f:
        f.write('{"step": 2, "wall": 1.0, "ta')  # a kill mid-append (or a copy uploaded mid-append)
    assert read_scalars_jsonl(run / "metrics" / "scalars.jsonl").num_rows == 1
    log = RunLogger(run, CFG, sync_every_min=0, capture=False, tee=False)
    log.scalar("a", 3.0, 3)
    log.close()
    sc = pd.read_parquet(run / "metrics" / "scalars.parquet")
    assert sc["step"].tolist() == [1, 3] and sc["value"].tolist() == [1.0, 3.0]


def test_env_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret")
    monkeypatch.setenv("CONTAINER_API_KEY", "vast_secret")
    monkeypatch.setenv("VAST_CONTAINERLABEL", "C.12345")
    monkeypatch.setenv("KITSUNE_DPH", "0.672")
    runlog.capture_env(tmp_path / "env")
    text = (tmp_path / "env" / "system.json").read_text(encoding="utf-8")
    assert "hf_supersecret" not in text and "vast_secret" not in text
    system = json.loads(text)
    assert system["env"]["HF_TOKEN"] == "<redacted>" and system["env"]["VAST_CONTAINERLABEL"] == "C.12345"
    assert system["instance_id"] == "C.12345" and system["offer_dph"] == 0.672


def test_export_run(run, tmp_path):
    exp = load_export()
    out = tmp_path / "export"
    exp.main([str(run), "--out", str(out)])
    for name in ["tb_scalars", "tb_histograms", "tb_text", "scalars", "steps", "train_utts", "hist", "text",
                 "eval_tf", "eval_greedy", "eval_probe", "eval_summaries", "samples", "events"]:
        assert (out / f"{name}.parquet").is_file(), name
        assert (out / f"{name}.csv").is_file(), name
    for f in ["config.json", "summary.json", "events.jsonl", "env/system.json", "README.md"]:
        assert (out / f).is_file(), f

    tb = pd.read_parquet(out / "tb_scalars.parquet")
    sc = pd.read_parquet(out / "scalars.parquet")
    assert set(tb["tag"]) == set(sc["tag"])  # TensorBoard mirrors every scalar
    t = tb[tb["tag"] == "loss/total"].sort_values("step")
    assert t["step"].tolist() == [1, 2, 3, 4, 5] and t["value"].tolist() == pytest.approx([2.0 / s for s in range(1, 6)])
    hi = pd.read_parquet(out / "tb_histograms.parquet")
    assert set(hi["tag"]) == {"weights/enc0", "grads/enc0"} and hi.set_index("tag").loc["weights/enc0", "num"] == 10_000
    tx = pd.read_parquet(out / "tb_text.parquet")
    assert {"note", "samples", "config"} <= set(tx["tag"]) and any(tx["tag"].str.startswith("events/"))
    assert "hello **world**" in tx.set_index("tag").loc["note", "text"]

    tf = pd.read_parquet(out / "eval_tf.parquet")
    assert tf["step"].tolist() == [5, 5] and tf["set"].tolist() == ["eval_jsut"] * 2
    gr = pd.read_parquet(out / "eval_greedy.parquet")
    assert list(gr["hyp_ids"].iloc[0]) == [5, 6, 3]
    assert json.loads(pd.read_csv(out / "eval_greedy.csv")["hyp_ids"].iloc[0]) == [5, 6, 3]
    es = pd.read_parquet(out / "eval_summaries.parquet").set_index("key")
    assert es.loc["sets/eval_jsut/cer_ref_corpus", "value"] == 0.5
    evs = pd.read_parquet(out / "events.parquet")
    assert "exception" in set(evs["kind"]) and "boom" in evs.set_index("kind").loc["exception", "fields_json"]
    assert len(pd.read_parquet(out / "train_utts.parquet")) == 15

    readme = (out / "README.md").read_text(encoding="utf-8")
    for name in ["tb_scalars", "scalars", "steps", "train_utts", "hist", "eval_tf", "eval_greedy", "events", "samples"]:
        assert f"`{name}.parquet`" in readme, name
        for col in pd.read_parquet(out / f"{name}.parquet").columns:
            assert f"`{col}`" in readme, (name, col)
    assert "`loss/total`" in readme  # the tag list


def test_export_keeps_infra_logs_and_restart_configs(run, tmp_path):
    """vast/finish.py puts the box's logs and state under runs/<id>/infra/; a resume writes config.<stamp>.json."""
    infra = run / "infra"
    infra.mkdir()
    (infra / "kitsune.log").write_text("[onstart] boot 1\n", encoding="utf-8")
    (infra / "events.jsonl").write_text(
        json.dumps({"wall": 1.0, "source": "finish", "kind": "verify", "files": 12, "problems": []}) + "\n"
        + json.dumps({"wall": 2.0, "source": "finish", "kind": "destroy", "reason": "run verified on the hub"}) + "\n",
        encoding="utf-8")
    (infra / "bootstrap_timings.jsonl").write_text('{"phase": "plan", "seconds": 1.5, "end": 1790000000}\n'
                                                   '{"phase": "pull_derived", "seconds": 60.2, "end": 1790000060}\n',
                                                   encoding="utf-8")
    (run / "config.20260924T010203Z.json").write_text('{"resume": {"step": 10}}', encoding="utf-8")
    out = tmp_path / "export"
    tables = load_export().export(str(run), out)
    assert (out / "infra" / "kitsune.log").is_file() and (out / "config.20260924T010203Z.json").is_file()
    assert tables["infra_events"]["kind"].tolist() == ["verify", "destroy"]
    assert set(tables["infra_events"]["emitter"]) == {"finish"}
    assert tables["infra_bootstrap_timings"]["seconds"].tolist() == [1.5, 60.2]
    readme = (out / "README.md").read_text(encoding="utf-8")
    for needle in ("`infra_events.parquet`", "`infra_bootstrap_timings.parquet`", "- `infra`:",
                   "- `config.20260924T010203Z.json`: config written by a restart", "`emitter`", "`seconds`"):
        assert needle in readme, needle


def test_export_hf_source_is_downloaded(run, tmp_path, monkeypatch):
    """hf://user/repo/runs/<id> -> snapshot_download of that prefix only (checkpoints excluded); mocked, no network."""
    import shutil

    import huggingface_hub

    seen = {}

    def fake_snapshot_download(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        dst = Path(kw["local_dir"]) / "runs" / "tiny-run"
        shutil.copytree(run, dst)
        return str(kw["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    exp = load_export()
    out = tmp_path / "export-hf"
    exp.main(["hf://me/kitsune-runs/runs/tiny-run", "--out", str(out)])
    assert seen["repo_id"] == "me/kitsune-runs" and seen["allow_patterns"] == ["runs/tiny-run/*"]
    assert seen["ignore_patterns"] == ["runs/tiny-run/checkpoints/*"]
    assert (out / "steps.parquet").is_file() and (out / "README.md").is_file()
