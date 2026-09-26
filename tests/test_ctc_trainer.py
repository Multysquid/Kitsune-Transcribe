"""The CTC family inside scripts/04_distill.py (family "ctc"; STUDY.md 2.1, 3.2, 4.2; decisions 15, 22-24):

- the frame store (kitsune.trainset.build_frame_stores): the stored Parakeet targets per id exactly as the label pass
  wrote them, the batch collation, the cache reuse, and the FRAME PREFLIGHT at build time (decision 15: a mismatched
  train row dropped and counted, more than 0.1 % of the train rows a hard fail, any eval row a hard fail)
- the frame planner (no decoder lengths) next to the token planner, whose plans and fingerprint are unchanged
- the loss: the trainer's micro-batch loss equals kitsune.ctc_kd's on the same batch built independently
- a tiny end-to-end CPU run (a tiny ParakeetForCTC from tests/fixtures_ctc.py, synthetic parakeet_out in the real
  format): smoke (frame preflight events, LogMel vs the Parakeet extractor, padded rows on frame log-probs, FLOPs),
  the step-0 / fraction / mini / final evals with the CTC metrics and the greedy CTC decode, the verdict against the
  Parakeet CTC teacher, summary.json (family, selection_sha256, started_utc), the TensorBoard tags; a crash resumed
  from a mid state ends with exactly the uninterrupted run's weights; the T/2 branch is the T/2 run; an LR probe; the
  export reloads with load_ctc_student and decodes as the trained weights do
CPU only, tiny models, synthetic data in the real on-disk formats."""
import copy
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
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
from fixtures_ctc import make_fake_parakeet_out, tiny_student_dir  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
PEAK, WARMUP, COOLDOWN, MAX_STEPS = 3e-3, 3, 0.2, 20
BLANK = 3072


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


def utts_of(run: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((run / "metrics" / "train_utts").glob("part-*.parquet"))])


def summary_of(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def n_samples(audio: bytes) -> int:
    import soundfile as sf

    return int(sf.info(io.BytesIO(audio)).frames)


@pytest.fixture(autouse=True)
def grad_enabled():
    """Autograd on at the start of every test here, as in a trainer process (see tests/test_study_trainer.py: an
    earlier test file can leave it off for the whole pytest process)."""
    import gc

    gc.collect()
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(prev)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic corpus + selection + parakeet_out (frames aligned to the audio) + a tiny ParakeetForCTC student saved
    as scripts/03c saves one, and a base config: family ctc on the steps clock (20 steps of ~6 audio-s in micro-batches
    of ~3 s), no smoke phase, no evals or checkpoints in the loop unless a test asks for them."""
    root = tmp_path_factory.mktemp("ctc")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (36, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL), seed=21)
    sel = make_fake_selection(fc, greedy_n=4, probe_n=5)
    po = make_fake_parakeet_out(fc, seed=5)
    sdir, _ = tiny_student_dir(root / "student", seed=3, n_layers=2, ffn=48, name="tiny-p")
    base = {
        "family": "ctc", "parakeet_root": str(po.root),
        "student": str(sdir), "data_root": str(fc.data), "teacher_root": str(fc.teacher_out),
        "second_root": str(fc.second_out), "selection": str(sel), "cache_dir": str(root / "cache"),
        "runs_root": str(root / "runs"), "sources": ["src_a", "src_b"], "eval_sets": EVAL,
        "device": "cpu", "autocast": "none",
        "optim": {"lr": PEAK},
        "loss": {"l2sp_lambda": 0.0},
        "schedule": {"warmup_steps": WARMUP, "cooldown_frac": COOLDOWN, "clock": "steps", "max_steps": MAX_STEPS},
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
    return dict(root=root, base=base, fc=fc, po=po, sel=sel, student=sdir)


def write_config(env, name: str, over: dict) -> str:
    path = env["root"] / f"{name}.json"
    path.write_text(json.dumps(merged(env["base"], dict(over, run_name=name)), indent=1), encoding="utf-8")
    return str(path)


def cfg_of(env, over: dict | None = None) -> dict:
    m = load_script("04_distill")
    return m.load_config(write_config(env, "unit", over or {}), [])


class Log:
    """Stands in for RunLogger where a test calls the setup functions directly."""

    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))


# ---------------------------------------------------------------------------------------------------- units


def test_ctc_frames_equals_the_extractor_count():
    """kitsune.trainset.ctc_frames (what the frame preflight and the loader's workers use, without importing
    transformers) is kitsune.ctc_student.expected_n_frames: the Parakeet extractor's valid mel frames, 8x subsampled
    as the encoder does."""
    from kitsune import ctc_student as CS
    from kitsune import trainset

    rng = np.random.default_rng(0)
    ns = [0, 1, 159, 160, 161, 319, 320, 1279, 1280, 1281, 1440, 16000, 16159, 16160] + rng.integers(
        0, 480000, 300).tolist()
    assert [trainset.ctc_frames(n) for n in ns] == [CS.expected_n_frames(n) for n in ns]


def test_token_planner_unchanged_and_the_frame_planner():
    """The token planner's plans and fingerprint are what they were before the frame mode (pinned on the code before
    it); the frame planner (max_dec_len None) excludes nothing, cuts micro-batches by padded audio alone, reports no
    decoder statistics and names the micro-batch with the most padded frames as a memory worst case."""
    from kitsune import trainset as T

    utts = [T.Utt(id=f"u{i}", source="s" + str(i % 2), duration=0.4 + 0.11 * (i * 7 % 13), n_tok=1 + i % 7,
                  audio_off=0, audio_len=0, tok_off=0) for i in range(40)]
    p = T.StepPlanner(utts, step_audio_s=6, micro_audio_s=3, max_dec_len=14, pool_micro=4, seed=3)
    assert p.fingerprint == "1ea5d296bcf643e4" and p.n_excluded == 10
    assert p.epoch_plan(1) == [[[37], [18, 31], [32, 17, 0]], [[24], [23, 21, 30], [4, 2, 28, 15]],
                               [[25, 8], [3, 1], [14, 10]], [[9]], [[36, 39], [35], [29, 38]], [[7, 16], [22], [11]]]
    assert p.worst_micro_batches() == {"longest": [24], "most_dec_positions": [30, 17, 2, 28],
                                       "most_targets": [30, 17, 2, 28]}

    long_u = [T.Utt(id=f"v{i}", source="s", duration=u.duration, n_tok=500 + i, audio_off=0, audio_len=0, tok_off=0)
              for i, u in enumerate(utts)]  # no decoder: any number of target tokens is fine
    f = T.StepPlanner(long_u, step_audio_s=6, micro_audio_s=3, max_dec_len=None, pool_micro=4, seed=3)
    assert f.frames and f.n_excluded == 0 and f.dec_len is None and f.fingerprint != p.fingerprint
    plan = f.epoch_plan(0)
    mbs = [mb for step in plan for mb in step]
    assert sorted(i for mb in mbs for i in mb) == list(range(40))
    assert all(max(f.dur[mb]) * len(mb) <= 3 or len(mb) == 1 for mb in mbs)
    assert "pad_eff_dec" not in f.stats and f.stats["excluded_dec_len"] == 0 and f.stats["targets"] == sum(
        500 + i for i in range(40))
    w = f.worst_micro_batches()
    assert set(w) == {"longest", "most_padded_frames"}
    assert float(f.dur[w["most_padded_frames"]].max()) * len(w["most_padded_frames"]) == max(
        float(f.dur[mb].max()) * len(mb) for mb in mbs)


def test_frame_preflight_rule():
    """Decision 15 on its own: a row whose decoded audio gives another frame count is a mismatch; up to 0.1 % of the
    train rows are dropped and counted, more fails, and so does a single eval row; a row that does not decode is a
    mismatch too (counted as undecodable as well): dropped in train within the 0.1 %, a hard fail in eval."""
    from kitsune import trainset as T

    n = 1000
    ids = [f"r{i}" for i in range(n)]
    samples = [3200 + 7 * i for i in range(n)]
    stored = [T.ctc_frames(s) for s in samples]

    def run(split, bad_rows, undecodable=()):
        dec = [("RuntimeError: no" if i in undecodable else s) for i, s in enumerate(samples)]
        st = [t + (1 if i in bad_rows else 0) for i, t in enumerate(stored)]
        return T.frame_preflight(ids, ["src"] * n, [split] * n, [0.2] * n, st, dec)

    ok = run("train", {5})
    assert ok["ok"] and ok["n_mismatch"] == 1 and ok["_rows"] == [5] and ok["n_undecodable"] == 0
    assert ok["mismatches"][0] == dict(id="r5", source="src", split="train", stored=stored[5] + 1, expected=stored[5],
                                       n_samples=samples[5], duration=0.2)
    assert ok["by_split"] == {"train": dict(rows=1000, mismatch=1, undecodable=0)} and ok["train_mismatch_frac"] == 0.001
    und = run("train", set(), undecodable={9})  # an undecodable row: dropped and counted as a mismatch
    assert und["ok"] and und["n_mismatch"] == 1 and und["_rows"] == [9] and und["n_undecodable"] == 1
    assert und["by_split"] == {"train": dict(rows=1000, mismatch=1, undecodable=1)} and und["n_checked"] == 999
    assert und["mismatches"][0] == dict(id="r9", source="src", split="train", stored=stored[9], expected=None,
                                        n_samples=None, duration=0.2, error="RuntimeError: no")
    assert not run("train", {5}, undecodable={9})["ok"]  # together 0.2 % > 0.1 %
    assert not run("train", {5, 6})["ok"]  # 0.2 % > 0.1 %
    assert not run("eval", {5})["ok"] and run("eval", set())["ok"]
    assert not run("eval", set(), undecodable={3})["ok"]  # any eval row: a hard fail
    assert run("train", set())["ok"] and run("train", set())["n_checked"] == 1000


def test_one_source_for_the_parakeet_ctc_baseline(env, monkeypatch):
    """The Parakeet CTC teacher's registered gate CER is kitsune.evaluate's (study/PREREG.json's filled baselines);
    kitsune.ctc_eval falls back to the K6 numbers only while those are pending. The trainer's start-up check
    (ctc_teacher_baselines) and the verdict's fallback for a set without a same-ids teacher CER read it at call time;
    the family verdict is kitsune.evaluate.verdict(..., family="ctc")."""
    from kitsune import ctc_eval as ce
    from kitsune import evaluate as ev

    m = load_script("04_distill")
    monkeypatch.setattr(ev, "PARAKEET_CTC_CER_PREREG", None)  # pending
    assert ce.teacher_prereg() == (ce.PARAKEET_CTC_CER_K6, "pending: K6")
    po = env["po"].root
    base = ce.ctc_teacher_baselines(po, EVAL, check=False)
    got = {x: base[x]["cer_corpus"] for x in EVAL}
    assert all(base[x]["prereg"] == ce.PARAKEET_CTC_CER_K6[x] and base[x]["prereg_source"] == "pending: K6"
               for x in EVAL)
    monkeypatch.setattr(ev, "PARAKEET_CTC_CER_PREREG", dict(got))  # registered: exactly the recomputed numbers
    assert ce.teacher_prereg() == (got, "registered")
    base = ce.ctc_teacher_baselines(po, EVAL, check=True)
    assert all(base[x]["prereg"] == got[x] and base[x]["prereg_source"] == "registered" for x in EVAL)
    monkeypatch.setattr(ev, "PARAKEET_CTC_CER_PREREG", dict(got, eval_cv8=got["eval_cv8"] + 0.001))
    with pytest.raises(ValueError, match=r"eval_cv8: Parakeet CTC corpus CER .* \(registered\)"):
        ce.ctc_teacher_baselines(po, EVAL, check=True)

    # the verdict: same-ids teacher CER where the final has one, the registered (or K6) number where not
    final = dict(sets={x: dict(cer_ref_corpus=0.1, teacher_cer_ref_corpus=0.08, n=6, n_truncated=0) for x in EVAL})
    final["sets"]["eval_cv8"]["teacher_cer_ref_corpus"] = float("nan")
    cfg = m.load_config(None, [])
    cfg["family"] = "ctc"
    monkeypatch.setattr(ev, "PARAKEET_CTC_CER_PREREG", None)
    v = m.family_verdict(cfg, final, [])
    assert v["family"] == "ctc" and v["teacher"] == "parakeet-ctc" and v["teacher_prereg_status"] == "pending"
    assert v["sets"]["eval_jsut"]["teacher"] == 0.08 and v["sets"]["eval_jsut"]["teacher_source"] == "same ids"
    assert v["sets"]["eval_cv8"]["teacher"] == ce.PARAKEET_CTC_CER_K6["eval_cv8"]
    assert v["sets"]["eval_cv8"]["teacher_source"] == "results.teacher"
    assert all(d["teacher_prereg"] is None and d["teacher_system"] == "parakeet-ctc" for d in v["sets"].values())
    reg = {x: 0.07 for x in EVAL}
    monkeypatch.setattr(ev, "PARAKEET_CTC_CER_PREREG", reg)
    v = m.family_verdict(cfg, final, [])
    assert v["teacher_prereg_status"] == "registered" and v["sets"]["eval_cv8"]["teacher"] == 0.07
    assert v["sets"]["eval_jsut"]["baseline_drift"] == pytest.approx(0.08 - 0.07)


def test_validate_the_family_keys(env):
    m = load_script("04_distill")
    assert m.DEFAULTS["family"] == "aed" and m.DEFAULTS["loss"]["w_ctc"] == 0.8 and m.DEFAULTS["parakeet_root"] is None
    assert m.load_config(None, [])["family"] == "aed"
    c = cfg_of(env)
    assert c["family"] == "ctc" and m.is_ctc(c) and not m.is_ctc({})
    for bad, match in ((["family=ctc"], "needs parakeet_root"), (["family=rnnt"], "family must be one of"),
                       (["loss.w_ctc=-1"], "loss.w_ctc"), (["loss.w_ctc=x"], "loss.w_ctc"),
                       (["family=ctc", "parakeet_root=po", "loss.aux_ctc_weight=0.3"], "aux-CTC head")):
        with pytest.raises(SystemExit, match=match):
            m.load_config(None, bad)
    saved = m.load_config(None, [])
    with pytest.raises(SystemExit, match="family"):
        m.resume_overrides(saved, [("family", "ctc")])  # RESUME_FIXED: the student family


# ---------------------------------------------------------------------------------------------------- the store


def test_frame_store_holds_the_label_pass_targets(env):
    """The trainer's frame stores (family ctc): per id the stored Parakeet targets exactly as the label pass wrote them
    (blank log-probs, dense frames, top-k, the greedy CTC path = the jsonl ctc_hyp tokens), n_tok = U, the teacher text
    = ctc_hyp; a micro-batch collates them as kitsune.ctc_targets.collate_frame_targets does, with n_frames = the
    decoded audio's frame count; the second build reuses the cache."""
    from kitsune import ctc_student as CS
    from kitsune import trainset
    from kitsune.audio import decode_audio
    from kitsune.ctc_targets import collate_frame_targets

    m = load_script("04_distill")
    cfg, log = cfg_of(env), Log()
    po = env["po"]
    stores = (("train", m.build_train_store(cfg, log)), ("eval", m.build_eval_store(cfg, log, frames=True)))
    for split, store in stores:
        assert trainset.is_frame_store(store) and store.cache_dir.name == f"ctc_{split}"
        want = {i for i, u in env["fc"].utts.items() if u.has_teacher and u.split == split
                and (u.source in cfg["sources"] if split == "train" else u.source in EVAL)}
        kept = set(pd.read_parquet(env["sel"]).query("keep")["id"])
        assert {u.id for u in store.utts} == want & kept
        ds = trainset.dataset_for(store)
        fr = store.frame()
        for i, u in enumerate(store.utts):
            t, g = store.targets(i), po.targets[u.id]
            assert t.n_frames == g.n_frames and u.n_tok == len(g.ctc_ids) == len(t.ctc_ids)
            for a in ("blank_lp", "dense_frame", "topk_idx", "topk_lp", "ctc_ids"):
                assert np.array_equal(getattr(t, a), getattr(g, a)), (u.id, a)
            assert fr["hyp"][i] == po.rows[u.id]["ctc_hyp"] and fr["tdt_hyp"][i] == po.rows[u.id]["hyp"]
            assert fr["ref"][i] == po.rows[u.id]["ref"] and fr["n_samples"][i] == po.n_samples[u.id]
        idx = list(range(min(4, len(store))))
        mb = ds[idx]
        want_b = collate_frame_targets([po.targets[store.utts[i].id] for i in idx])
        for k, v in want_b.items():
            assert torch.equal(mb[k], v), k
        assert torch.equal(mb["n_tok"], want_b["ctc_target_lengths"])
        for b, i in enumerate(idx):
            wave = decode_audio(ds.audio_bytes(i))
            assert int(mb["lengths"][b]) == len(wave) and int(mb["n_frames"][b]) == CS.expected_n_frames(len(wave))
        rep = store.info["frame_preflight"]
        assert rep["ok"] and rep["n_checked"] == len(store) and rep["n_mismatch"] == 0
    again = m.build_train_store(cfg, log)
    assert again.info.get("reused") and len(again) == len(m.build_train_store(cfg, log))
    assert m.make_planner(SimpleNamespace(cfg=cfg, train=again), 3.0).frames  # the frame planner for family ctc


def test_the_evaluator_and_the_box_get_the_token_eval_store(env, tmp_path):
    """What scripts/05_evaluate.py and the box's store step (kitsune/study_queue.py build-stores) call for a family-ctc
    config: build_eval_store(cfg, log) returns the TOKEN eval store (the evaluator scores every system on it and reads
    the frame targets from parakeet_out itself; the speed probe opens cache/eval) and builds the trainer's frame eval
    store of the same rows alongside, which the trainer then reuses; train_store_spec gives the plain name, under which
    05 packs its token probe stores, and the trainer's frame train store lives under ctc_<name>, so a token store built
    there (05's probe_is_train path) leaves the frame store as it was."""
    from kitsune import trainset

    m = load_script("04_distill")
    cache = tmp_path / "cache"
    cfg, log = cfg_of(env, {"cache_dir": str(cache)}), Log()
    ev = m.build_eval_store(cfg, log)
    assert not trainset.is_frame_store(ev) and ev.cache_dir == cache / "eval"
    item = trainset.AudioBatchDataset(ev)[[0, 1]]  # as 05's CTC eval opens it: the token targets are there
    assert not item["dropped"] and item["decoder_input_ids"].shape[0] == 2 and len(item["top_idx"])
    frames = m.build_eval_store(cfg, log, frames=True)
    assert trainset.is_frame_store(frames) and frames.info.get("reused") and frames.cache_dir == cache / "ctc_eval"
    assert [u.id for u in frames.utts] == [u.id for u in ev.utts]  # the same rows, in the same order
    assert m.greedy_subset_ids(cfg, frames) == m.greedy_subset_ids(cfg, ev)
    assert not trainset.is_frame_store(m.build_eval_store(cfg, log, frames=False))
    with pytest.raises(ValueError, match="family ctc"):
        m.build_eval_store(m.load_config(write_config(env, "aed-unit", {"family": "aed"}), []), log, frames=True)

    name, ids = m.train_store_spec(cfg, log)
    assert name == "train" and ids is None and m.store_dir(cfg, name) == cache / "ctc_train"
    train = m.build_train_store(cfg, log)
    assert trainset.is_frame_store(train) and train.cache_dir == cache / "ctc_train"
    tok = trainset.build_stores(cfg["selection"], cfg["data_root"], cfg["teacher_root"], cache / name, cfg["sources"],
                                ["train"], ids=ids, log=lambda s: None)  # as 05's probe_store with probe_is_train
    assert not trainset.is_frame_store(tok) and len(tok) == len(train)
    again = m.build_train_store(cfg, log)
    assert again.info.get("reused") and trainset.is_frame_store(again)

    # a duration subset: the frame store's ids are every row's draw, its name carries their hash under ctc_
    sub = cfg_of(env, {"cache_dir": str(cache), "subset": {"train_audio_s": 20, "eval_audio_s": 6}})
    sname, sids = m.train_store_spec(sub, log)
    assert sname.startswith("train_20s_") and m.build_train_store(sub, log).cache_dir == cache / f"ctc_{sname}"
    ename, eids = m.eval_store_spec(sub, log)
    assert m.build_eval_store(sub, log).cache_dir == cache / ename and (cache / f"ctc_{ename}" / "stores.json").is_file()


def test_frame_store_refuses_incomplete_label_files(env, tmp_path):
    """A partial or foreign parakeet_out stops the frame store build instead of a silent default: a missing meta.json,
    a meta.json with another blank / vocabulary / dense threshold / k_ctc than the shards', a shard without its .jsonl,
    and a selected row without its jsonl line (its reference and teacher text: every CER would read an empty string)."""
    import shutil

    from kitsune import trainset

    fc, sel = env["fc"], env["sel"]

    def build(po, name):
        return trainset.build_frame_stores(sel, fc.data, po, tmp_path / "cache" / name, EVAL, ["eval"],
                                           log=lambda s: None)

    def copy(name):
        dst = tmp_path / name
        shutil.copytree(env["po"].root, dst)
        return dst

    assert len(build(copy("ok"), "ok")) == 18
    po = copy("no_meta")
    (po / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        build(po, "no_meta")
    for key, value in (("blank", 3071), ("vocab", 4097), ("ctc_dense_thr", 0.9), ("k_ctc", None)):
        po = copy(f"meta_{key}")
        meta = json.loads((po / "meta.json").read_text(encoding="utf-8"))
        meta[key] = value
        (po / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            build(po, f"meta_{key}")
    po = copy("meta_k")
    meta = json.loads((po / "meta.json").read_text(encoding="utf-8"))
    meta["k_ctc"] = 4  # the shards hold k 8
    (po / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="k_ctc 8 differs"):
        build(po, "meta_k")
    po = copy("no_jsonl")
    (po / "eval_cv8" / "eval-00000.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="eval-00000.jsonl"):
        build(po, "no_jsonl")
    po = copy("short_jsonl")
    jl = po / "eval_cv8" / "eval-00000.jsonl"
    lines = [x for x in jl.read_text(encoding="utf-8").splitlines() if x.strip()]
    kept = set(pd.read_parquet(sel).query("keep")["id"])
    drop = next(i for i, x in enumerate(lines) if json.loads(x)["id"] in kept)
    gone = json.loads(lines[drop])["id"]
    jl.write_text("\n".join(lines[:drop] + lines[drop + 1:]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"no line for {gone}"):
        build(po, "short_jsonl")


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    """A corpus of 1,000 train rows (and 6 eval rows), so that one mismatched train row is exactly 0.1 %."""
    root = tmp_path_factory.mktemp("ctc_big")
    fc = make_fake_corpus(root / "corpus", sources={"src_big": (1000, "train"), "eval_jsut": (6, "eval")},
                          rows_per_shard=256, dur_range=(0.2, 0.45), token_range=(256, 296), no_second=("eval_jsut",),
                          seed=4, high_agree_frac=0.0)
    sel = make_fake_selection(fc, sources=["src_big"], eval_sets=["eval_jsut"], agree_max=3.0, greedy_n=4, probe_n=5)
    return dict(root=root, fc=fc, sel=sel)


def unique_length(fc, ids) -> list[str]:
    """The ids whose audio length (samples) no other labelled row shares: planted_parakeet_out plants by length."""
    counts: dict[int, int] = {}
    for u in fc.utts.values():
        if u.has_teacher:
            counts[n_samples(u.audio)] = counts.get(n_samples(u.audio), 0) + 1
    return [i for i in ids if counts[n_samples(fc.utts[i].audio)] == 1]


def planted_parakeet_out(fc, root, ids, monkeypatch, delta=1):
    """parakeet_out as the label pass writes it, but the rows `ids` stored with delta more frames than their audio
    gives (targets consistent with that count): a frame mismatch the preflight must find. The rows are found by their
    audio length (the fixture computes the frames from it), so each must be the only row of its length."""
    from kitsune import ctc_student as CS

    plant = {n_samples(fc.utts[i].audio) for i in ids}
    assert len(plant) == len(ids) and unique_length(fc, ids) == list(ids)
    real = CS.expected_n_frames
    with monkeypatch.context() as mp:
        mp.setattr(CS, "expected_n_frames", lambda n: real(n) + (delta if n in plant else 0))
        return make_fake_parakeet_out(fc, root, seed=7)


def test_frame_preflight_drops_and_fails_as_decision_15(big, monkeypatch):
    """One train row of 1,000 whose stored n_frames is not its audio's: dropped and counted (0.1 %, not above), the
    store holds the other 999; two: the build fails (0.2 %), writing frame_preflight.json and no stores.json; one eval
    row: the eval store's build fails."""
    from kitsune import trainset

    fc, sel = big["fc"], big["sel"]
    kept = pd.read_parquet(sel).query("keep")
    all_train = kept[kept["split"] == "train"]["id"].tolist()
    eval_ids = unique_length(fc, kept[kept["split"] == "eval"]["id"].tolist())
    assert len(all_train) == 1000 and eval_ids
    train_ids = unique_length(fc, all_train)
    assert len(train_ids) > 20
    one = planted_parakeet_out(fc, big["root"] / "po_one", [train_ids[17], eval_ids[0]], monkeypatch)
    st = trainset.build_frame_stores(sel, fc.data, one.root, big["root"] / "cache" / "one_train", ["src_big"],
                                     ["train"], log=lambda s: None)
    rep = st.info["frame_preflight"]
    assert rep["ok"] and rep["n_mismatch"] == 1 and rep["by_split"]["train"] == dict(rows=1000, mismatch=1,
                                                                                     undecodable=0)
    assert rep["dropped"] == [train_ids[17]] and rep["mismatches"][0]["stored"] == rep["mismatches"][0]["expected"] + 1
    assert st.info["dropped"]["frame_mismatch"]["n"] == 1 and st.info["dropped"]["frame_mismatch"]["ids"] == [
        train_ids[17]]
    assert len(st) == 999 and {u.id for u in st.utts} == set(all_train) - {train_ids[17]}
    ds = trainset.dataset_for(st)
    for i in range(0, 999, 97):  # every kept row still decodes to its stored frame count
        mb = ds[[i]]
        assert mb["ids"] == [st.utts[i].id] and not mb["dropped"]
    with pytest.raises(trainset.FramePreflightFailed, match=r"eval 1/6") as e:
        trainset.build_frame_stores(sel, fc.data, one.root, big["root"] / "cache" / "one_eval", ["eval_jsut"],
                                    ["eval"], log=lambda s: None)
    assert e.value.report["by_split"]["eval"]["mismatch"] == 1 and not e.value.report["ok"]
    assert not (big["root"] / "cache" / "one_eval" / "stores.json").exists()

    two = planted_parakeet_out(fc, big["root"] / "po_two", train_ids[3:5], monkeypatch)
    cache = big["root"] / "cache" / "two_train"
    with pytest.raises(trainset.FramePreflightFailed, match=r"train 2/1000; train share 0\.200 %"):
        trainset.build_frame_stores(sel, fc.data, two.root, cache, ["src_big"], ["train"], log=lambda s: None)
    saved = json.loads((cache / "frame_preflight.json").read_text(encoding="utf-8"))
    assert saved["n_mismatch"] == 2 and not saved["ok"] and not (cache / "stores.json").exists()


@pytest.mark.slow
def test_real_frame_preflight(tmp_path):
    """The frame preflight on real audio and real label-box targets: every row of CV8's first eval shard (MP3 at
    48 kHz, where n_frames from the stored duration is off on ~12 % of the rows, K4) decodes to its stored n_frames,
    and the store holds the npz's targets. Skips without the real files; from a worktree:
    KITSUNE_REAL_DATA_ROOT=<main checkout> KITSUNE_PARAKEET_OUT=<a labels/full/parakeet_out pull> pytest -m slow ...
    Measured on the laptop (labels/full rev 211b767, 15,627 rows of reazon_small, emilia_yodas, eval_cv8, eval_jsut
    and eval_reazon): 0 mismatches, 0 undecodable; the duration field alone would have mis-framed 546 of them."""
    from fixtures import REAL, need_real

    from kitsune import trainset
    from kitsune.ctc_targets import load_ctc_targets

    po = Path(os.environ.get("KITSUNE_PARAKEET_OUT") or REAL / "parakeet_out")
    npz = po / "eval_cv8" / "eval-00000.npz"
    need_real(npz, REAL / "data" / "shards" / "eval_cv8")
    rows = [json.loads(line) for line in npz.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines() if line]
    sel = tmp_path / "sel.parquet"
    pd.DataFrame(dict(id=[r["id"] for r in rows], source="eval_cv8", split="eval", teacher_file="eval_cv8/eval-00000",
                      duration=[float(r["duration"]) for r in rows], n_tok=1, truncated=False, agree=float("nan"),
                      teacher_cer=[float(r["ctc_cer"]) for r in rows], keep=True, in_probe=False,
                      in_greedy_subset=False)).to_parquet(sel, index=False)
    st = trainset.build_frame_stores(sel, REAL / "data", po, tmp_path / "cache", ["eval_cv8"], ["eval"],
                                     log=lambda s: None)
    rep = st.info["frame_preflight"]
    assert rep["ok"] and rep["n_mismatch"] == 0 and rep["n_undecodable"] == 0 and rep["n_checked"] == len(rows)
    want = load_ctc_targets(npz)
    for i in range(0, len(st), 97):
        t, g = st.targets(i), want[st.utts[i].id]
        assert t.n_frames == g.n_frames and np.array_equal(t.ctc_ids, g.ctc_ids)
        assert np.array_equal(t.topk_idx, g.topk_idx) and np.array_equal(t.blank_lp, g.blank_lp)


def test_an_eval_frame_mismatch_stops_the_run(env, monkeypatch, tmp_path):
    """The trainer builds its stores at setup: an eval row whose targets do not align with its audio stops the run
    there, with a `frame_preflight` event (ok false, the counts) - before any training."""
    from kitsune import trainset

    m = load_script("04_distill")
    fc = env["fc"]
    eid = unique_length(fc, [u.id for u in fc.utts.values() if u.split == "eval" and u.has_teacher
                             and u.source == "eval_cv8"])[0]
    po = planted_parakeet_out(fc, tmp_path / "po", [eid], monkeypatch)
    over = {"parakeet_root": str(po.root), "cache_dir": str(tmp_path / "cache")}
    with pytest.raises(trainset.FramePreflightFailed):
        m.main(["--config", write_config(env, "ctc-evalmismatch", over)])
    run = one_run(env["root"], "ctc-evalmismatch")
    pre = events(run, "frame_preflight")
    assert [(e["store"], e["ok"]) for e in pre] == [("train", True), ("eval", False)]
    assert pre[1]["by_split"]["eval"]["mismatch"] == 1 and pre[1]["mismatches"][0]["id"] == eid
    assert summary_of(run)["status"] == "failed" and [e["name"] for e in events(run, "phase")] == ["setup"]


# ---------------------------------------------------------------------------------------------------- the loss


def test_trainer_loss_equals_the_library(env):
    """The trainer's micro-batch loss (ctc_step_loss: its features, forward and frame batch) equals kitsune.ctc_kd's
    on the same utterances built independently: the audio decoded from the corpus, the Parakeet features of
    kitsune.ctc_student.ctc_features, a freshly loaded student, the targets as the label pass wrote them."""
    from kitsune import ctc_student as CS
    from kitsune.audio import decode_audio
    from kitsune.ctc_kd import ctc_kd_losses, ctc_kd_objective
    from kitsune.ctc_targets import collate_frame_targets

    m = load_script("04_distill")
    R = m.Run(cfg=cfg_of(env), run_dir=env["root"], device=torch.device("cpu"), amp=False, log=Log())
    m.setup_model(R, False, "frozen")
    m.setup_processing(R)
    m.setup_data(R)
    idx = list(range(5))
    mb = R.ds[idx]
    n_step = int(mb["n_tok"].sum()) + 7  # a step's N_u: the micro-batch is one of several
    with torch.no_grad():
        loss, losses, mfrac = m.ctc_step_loss(R, mb, R.feat_eval, n_step)
    assert mfrac is None

    ids = mb["ids"]
    waves = [decode_audio(env["fc"].utts[i].audio) for i in ids]
    feats, lens = CS.ctc_features(env["student"])(waves)
    mask = torch.arange(feats.shape[1])[None, :] < lens[:, None]
    model = CS.load_ctc_student(env["student"], "cpu")
    with torch.no_grad():
        lp, n = CS.ctc_log_probs(model, feats, mask)
    batch = collate_frame_targets([env["po"].targets[i] for i in ids])
    assert torch.equal(n, batch["n_frames"])
    want = ctc_kd_losses(lp, batch, per_utt=True)
    for k in ("kl_dense", "kl_blank", "ctc"):
        torch.testing.assert_close(losses[k], want[k], rtol=1e-5, atol=1e-5, msg=k)
    for k in ("n_dense", "n_frames", "n_tokens", "argmax_agree", "argmax_blank", "teacher_blank"):
        assert torch.equal(losses[k].cpu(), want[k]), k
    ref = ctc_kd_objective(want, n_step, 1.0, 0.8)
    assert float(loss) == pytest.approx(float(ref), rel=1e-5)
    assert float(ref) == pytest.approx(float((want["kl_dense"].sum() + want["kl_blank"].sum()
                                               + 0.8 * want["ctc"].sum()) / n_step), rel=1e-6)


def test_gradient_checkpointing_gives_the_same_gradients(env):
    """The memory probe's last fallback, per-layer gradient checkpointing, on the CTC student (frozen BN): the same
    loss and gradients as without it."""
    m = load_script("04_distill")
    grads = []
    for ckpt in (False, True):
        R = m.Run(cfg=cfg_of(env), run_dir=env["root"], device=torch.device("cpu"), amp=False, log=Log())
        m.setup_model(R, ckpt, "frozen")
        m.setup_processing(R)
        m.setup_data(R)
        assert R.model.encoder.gradient_checkpointing == ckpt
        loss, _, _ = m.ctc_step_loss(R, R.ds[[0, 1, 2]], R.feat_train, 20)
        loss.backward()
        grads.append((float(loss.detach()), [p.grad.clone() for p in R.params]))
    assert grads[0][0] == pytest.approx(grads[1][0], rel=1e-6)
    for a, b in zip(grads[0][1], grads[1][1]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------------------------------- end to end


# the reference run of the end-to-end tests: the smoke phase, the step-0 eval, a complete eval at 50 %, mini evals, a
# kept full state at 40 % (the T/2 branch's), weights at 50 %, histograms and layer stats, the final eval and verdict
REF_OVER = {"smoke": {"enabled": True, "steps": 3, "min_audio_s_per_s": 0, "require_loss_decrease": False,
                      "pad_utts": 4, "decode_per_set": 2},
            "eval": {"full_at_fracs": [0.5], "mini": {"every_steps": 5, "val_per_set": 2, "train_utts": 3},
                     "probe_greedy_audio_s": 4},
            "ckpt": {"full_at_fracs": [0.4], "weights_at_fracs": [0.5], "full_every_steps": 4, "keep_local": 5},
            "log": {"layer_stats_every": 4, "hist_every": 8}}


@pytest.fixture(scope="module")
def ref_run(env):
    m = load_script("04_distill")
    assert m.main(["--config", write_config(env, "ctc-ref", REF_OVER)]) == 0
    return one_run(env["root"], "ctc-ref")


def _eval_rows(env) -> dict:
    """The eval store's rows as written: id -> (source, ref, ctc_hyp, targets)."""
    kept = pd.read_parquet(env["sel"]).query("keep & split == 'eval'")
    po = env["po"]
    return {i: (s, po.rows[i]["ref"], po.rows[i]["ctc_hyp"], po.targets[i]) for i, s in zip(kept["id"], kept["source"])}


def test_ctc_reference_run(env, ref_run):
    """The run end to end: the frame preflight of both stores, the Parakeet CTC teacher baselines, the smoke checks on
    the CTC path, the evals at steps 0 / 10 / 20 (complete at 10 and 20) with minis at 5 and 15, the training rows and
    TensorBoard tags of the CTC loss, the verdict against the Parakeet CTC teacher on the same ids, and summary.json's
    family, selection_sha256 and started_utc."""
    from transformers import AutoProcessor

    from kitsune import ctc_eval as ce
    from kitsune import ctc_student as CS
    from kitsune import evaluate as ev

    m = load_script("04_distill")
    tokenizer = AutoProcessor.from_pretrained(str(env["student"])).tokenizer
    run = ref_run
    evs = events(run)
    kinds = [e["kind"] for e in evs]
    assert kinds[-1] == "logger_close" and "tb_tag_unmapped" not in kinds and "nonfinite_grad_skipped" not in kinds

    pre = {e["store"]: e for e in evs if e["kind"] == "frame_preflight"}
    assert set(pre) == {"train", "eval"} and all(e["ok"] and e["n_mismatch"] == 0 for e in pre.values())
    assert pre["eval"]["n_checked"] == 18 and pre["train"]["by_split"]["train"]["rows"] == pre["train"]["n_checked"]
    base = next(e for e in evs if e["kind"] == "teacher_baselines")
    rows = _eval_rows(env)
    assert base["teacher"] == "parakeet-ctc" and set(base["sets"]) == set(EVAL)
    for s in EVAL:
        mine = [r for r in rows.values() if r[0] == s]
        want = ev.corpus_cer([r[2] for r in mine], [r[1] for r in mine])["cer"]
        assert base["sets"][s]["cer_corpus"] == pytest.approx(want) and base["sets"][s]["n"] == 6
    model = next(e for e in evs if e["kind"] == "model")
    assert model["family"] == "ctc" and model["params"]["total"] == model["params"]["closed_form"]

    smoke = {e["kind"]: e for e in evs if e["kind"].startswith("smoke_")}
    assert smoke["smoke_logmel_vs_hf"]["mean_abs_diff"] == 0.0 and smoke["smoke_logmel_vs_hf"]["masks_equal"]
    assert "smoke_train_dither" not in smoke  # the Parakeet features have no dither
    assert smoke["smoke_longest_fwd_bwd"]["ok"] and smoke["smoke_longest_fwd_bwd"]["frames"] > 0
    pad = smoke["smoke_padded_row"]
    assert pad["ok"] and pad["unit"] == "frames" and pad["n_utts"] == 4 and pad["kl_mean"] < 1e-5
    assert pad["argmax_agree"] == 1.0 and smoke["smoke_flops"]["gflops_fwd_per_padded_audio_s"] > 0
    assert smoke["smoke_steps"]["steps"] == 3

    assert [(e["at_step"], e["complete"], e["final"]) for e in evs if e["kind"] == "eval"] == [
        (0, False, False), (10, True, False), (20, True, True)]
    assert [e["at_step"] for e in evs if e["kind"] == "eval_mini"] == [5, 15]

    st = steps_of(run)
    assert st["step"].tolist() == list(range(1, MAX_STEPS + 1))
    obj = st["loss/objective"].to_numpy()
    assert np.isfinite(obj).all() and obj[-3:].mean() < obj[:3].mean()
    np.testing.assert_allclose(st["loss/kl"], st["loss/kl_dense"] + st["loss/kl_blank"], rtol=1e-6)
    np.testing.assert_allclose(obj, st["loss/kl"] + 0.8 * st["loss/ctc"], rtol=1e-6)
    np.testing.assert_allclose(st["loss/total"], obj, rtol=1e-12)  # L2-SP 0
    np.testing.assert_allclose(st["combined_loss/train"], obj, rtol=1e-12)
    for tag in ("ctc/argmax_agree", "ctc/argmax_blank", "ctc/teacher_blank", "ctc/frac_dense"):
        assert ((st[tag] >= 0) & (st[tag] <= 1)).all(), tag
    assert "loss/ce" not in st and "perf/pad_eff_dec" not in st and (st["perf/mfu"] > 0).all()
    tags = set(pd.read_parquet(run / "metrics" / "scalars.parquet")["tag"])
    assert {"loss/kl_dense", "loss/kl_blank", "loss/ctc", "ctc/argmax_agree", "ctc/argmax_blank", "ctc/teacher_blank",
            "combined_loss/val", "combined_loss/val_full", "eval/tf/eval_jsut/argmax_blank",
            "src/src_a/ctc", "layers/grad_norm/ctc_head", "bn/max_abs_drift"} <= tags
    tag_map = json.loads((run / "metrics" / "tag_map.json").read_text(encoding="utf-8"))
    assert not any(e.get("unmapped") for e in tag_map.values())

    # the train records: n_tok = U, the teacher's CTC target tokens
    utts = utts_of(run)
    assert (utts["n_tok"] == utts["id"].map(lambda i: len(env["po"].targets[i].ctc_ids))).all()
    assert ((utts["top1_acc"] >= 0) & (utts["top1_acc"] <= 1)).all() and np.isfinite(utts["ce"]).all()

    # the final eval: frame metrics per set and the greedy CTC decode against the Parakeet CTC teacher
    s = json.loads((run / "evals" / "step_20" / "summary.json").read_text(encoding="utf-8"))
    for x in EVAL:
        d = s["tf"]["sets"][x]
        mine = [r for r in rows.values() if r[0] == x]
        frames = sum(r[3].n_frames for r in mine)
        assert d["n_tok"] == sum(len(r[3].ctc_ids) for r in mine) and d["n_frames"] == frames
        assert d["ce"] == d["ctc"] and d["top1"] == d["argmax_agree"]
        n_dense = sum(len(r[3].dense_frame) for r in mine)
        assert d["frac_dense"] == pytest.approx(n_dense / frames)  # kl_dense per dense, kl_blank per blank-only frame
        assert d["kl"] * d["n_tok"] == pytest.approx(d["kl_dense"] * n_dense + d["kl_blank"] * (frames - n_dense))
        assert d["teacher_blank"] == pytest.approx(sum(int((r[3].col0() == BLANK).sum()) for r in mine) / frames)
        g = s["greedy_full"]["sets"][x]
        assert g["n"] == 6 and g["trunc_rate"] == 0.0
        assert g["teacher_cer_ref_corpus"] == pytest.approx(
            ev.corpus_cer([r[2] for r in mine], [r[1] for r in mine])["cer"])
    comb = s["combined_loss"]["val_full"]
    gate = [s["tf"]["sets"][x] for x in EVAL]
    ntok = sum(d["n_tok"] for d in gate)
    # the headline's val_top1 pools the frame agreement per frame (its unit), not per target token
    nfr = sum(d["n_frames"] for d in gate)
    assert s["headline"]["val_top1"] == pytest.approx(sum(d["top1"] * d["n_frames"] for d in gate) / nfr)
    assert s["headline"]["val_loss"] == pytest.approx(sum(d["kl"] * d["n_tok"] for d in gate) / ntok)
    mini = json.loads((run / "evals" / "step_15_mini" / "summary.json").read_text(encoding="utf-8"))
    mg = [d for d in mini["tf"]["sets"].values()]
    assert mini["headline"]["val_top1"] == pytest.approx(
        sum(d["top1"] * d["n_frames"] for d in mg) / sum(d["n_frames"] for d in mg))
    assert comb["value"] == pytest.approx(sum(d["kl"] * d["n_tok"] + 0.8 * d["ctc"] * d["n_tok"] for d in gate) / ntok)
    assert comb["w_ctc"] == 0.8 and comb["sets"] == sorted(EVAL)
    # the train probe: teacher-forced on its rows, greedy CTC on the probe-greedy ones, against the teacher's text
    pg = pd.read_parquet(run / "evals" / "step_20" / "probe_greedy.parquet")
    pg_ids = next(e for e in evs if e["kind"] == "subset" and e.get("split") == "probe_greedy")["ids"]
    assert sorted(pg["id"]) == sorted(pg_ids) and (pg["teacher_hyp"] == pg["id"].map(
        lambda i: env["po"].rows[i]["ctc_hyp"])).all()
    assert s["probe"]["all"]["n_tok"] > 0 and s["probe_greedy"]["all"]["n"] == len(pg_ids)
    assert "train_cer_vs_teacher" in s["headline"]
    greedy = pd.read_parquet(run / "evals" / "step_20" / "greedy_eval_jsut.parquet")
    assert len(greedy) == 6 and (greedy["teacher_hyp"] == greedy["id"].map(lambda i: rows[i][2])).all()
    assert (greedy["hyp"] == greedy["hyp_ids"].map(lambda ids: CS.decode_ids(tokenizer, ids))).all()

    summ = summary_of(run)
    assert summ["family"] == "ctc" and summ["status"] == "complete"
    assert summ["selection_sha256"] == hashlib.sha256(Path(env["sel"]).read_bytes()).hexdigest()
    started = datetime.fromisoformat(summ["started_utc"])
    assert started.tzinfo is not None and started <= datetime.now(timezone.utc)
    ver = summ["verdict"]
    assert ver["verdict"] in ("GO", "PROMISING", "NO-GO", "INCONCLUSIVE") and set(ver["sets"]) == set(EVAL)
    # kitsune.evaluate.verdict(..., family="ctc"): its registered numbers are kitsune.evaluate's (PREREG.json)
    reg = ev.family_teacher_prereg("ctc")
    assert ver["family"] == "ctc" and ver["teacher"] == "parakeet-ctc"
    assert ver["teacher_prereg_status"] == ("registered" if reg else "pending")
    for x in EVAL:
        v = ver["sets"][x]
        assert v["teacher_system"] == "parakeet-ctc" and v["teacher_prereg"] == (reg or {}).get(x)
        assert v["teacher"] == pytest.approx(s["greedy_full"]["sets"][x]["teacher_cer_ref_corpus"])
        assert v["teacher_source"] == "same ids"
    assert [r["step"] for r in summ["history"]] == [0, 10, 20] and all("heldout_kl" in r for r in summ["history"])
    assert summ["history"][1]["greedy_full"] and summ["history"][0]["tf"]["eval_jsut"]["ce"] > 0
    assert [r["step"] for r in summ["mini_history"]] == [5, 15]

    ck = run / "checkpoints"
    assert (ck / "full_step_8").is_dir() and (ck / "step_10").is_dir() and (ck / "step_20").is_dir()
    assert m.RESUME_FIXED and "family" in m.RESUME_FIXED


def test_the_export_reloads_and_decodes_as_trained(env, ref_run):
    """step_20/ is a ParakeetForCTC dir: kitsune.ctc_student.load_ctc_student loads it; its weights are the trained
    fp32 weights in bf16 (BN statistics fp32); its greedy decode of every eval row equals that of the trained weights
    so rounded; it carries the Parakeet processor and tokenizer, the CC-BY-4.0 card and a `trained` block."""
    from kitsune import ctc_eval as ce
    from kitsune import ctc_student as CS
    from kitsune import trainset
    from kitsune.features import LogMel

    w = ref_run / "checkpoints" / "step_20"
    for f in ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "MODEL_CARD.md", "README.md",
              "student_meta.json", "config.json"):
        assert (w / f).is_file(), f
    card = (w / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "license: cc-by-4.0" in card and "NVIDIA" in card
    meta = json.loads((w / "student_meta.json").read_text(encoding="utf-8"))
    assert meta["trained"]["step"] == 20 and meta["trained"]["run_id"] == ref_run.name and meta["family"] == "ctc"
    exported = CS.load_ctc_student(w, "cpu")
    trained = torch.load(ref_run / "checkpoints" / "full_step_20" / "model.pt", weights_only=True)

    def as_saved(k, v):  # bf16 weights, fp32 BN statistics, integer buffers as they are
        bn = k.endswith(("running_mean", "running_var"))
        return v.to(torch.bfloat16).float() if v.is_floating_point() and not bn else v

    got = exported.state_dict()
    assert got.keys() == trained.keys()
    for k, v in trained.items():
        assert torch.equal(got[k].to(v.dtype), as_saved(k, v)), k
    rounded = CS.load_ctc_student(env["student"], "cpu")
    rounded.load_state_dict({k: as_saved(k, v) for k, v in trained.items()})
    m = load_script("04_distill")
    store = m.build_eval_store(cfg_of(env), Log(), frames=True)
    assert trainset.is_frame_store(store)
    feat = CS.ctc_features(w).logmel
    assert isinstance(feat, LogMel)
    _, hyps_a, _ = ce.ctc_eval_records(exported, store, feat, "cpu", 20.0, amp=False)
    _, hyps_b, _ = ce.ctc_eval_records(rounded, store, feat, "cpu", 20.0, amp=False)
    assert hyps_a == hyps_b and len(hyps_a) == 18


def test_a_crash_resumed_from_a_mid_state_is_the_uninterrupted_run(env, ref_run, monkeypatch):
    """A crash before step 10 resumed from full_step_8 ends with exactly the uninterrupted run's weights, the same
    utterances and SpecAugment masks per step and the same losses."""
    m = load_script("04_distill")
    monkeypatch.setenv("KITSUNE_CRASH_AT_STEP", "10")
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.main(["--config", write_config(env, "ctc-crash", REF_OVER)])
    monkeypatch.delenv("KITSUNE_CRASH_AT_STEP")
    crash = one_run(env["root"], "ctc-crash")
    assert m.main(["--resume", str(crash / "checkpoints" / "full_step_8")]) == 0
    assert next(e for e in events(crash) if e["kind"] == "resumed")["at_step"] == 8
    a = torch.load(ref_run / "checkpoints" / "full_step_20" / "model.pt", weights_only=True)
    b = torch.load(crash / "checkpoints" / "full_step_20" / "model.pt", weights_only=True)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ua, ub = utts_of(ref_run), utts_of(crash)
    for step in (9, 10, 11, 12):
        x = ua[ua["step"] == step].sort_values("id")
        y = ub[(ub["step"] == step) & (ub["attempt"] == 1)].sort_values("id")
        assert len(x) and x["id"].tolist() == y["id"].tolist() and x["masked_frac"].tolist() == y["masked_frac"].tolist()
    sa, sb = steps_of(ref_run).set_index("step"), steps_of(crash).drop_duplicates("step", keep="last").set_index("step")
    for tag in ("loss/objective", "loss/ctc", "ctc/argmax_agree"):
        np.testing.assert_array_equal(sa[tag].to_numpy(), sb[tag].to_numpy())
    # started_utc: the first start's, kept across the resume (the state the first attempt saved at step 8 has it)
    first = json.loads((crash / "checkpoints" / "full_step_8" / "trainer.json").read_text(encoding="utf-8"))
    assert first["st"]["started_utc"] and summary_of(crash)["started_utc"] == first["st"]["started_utc"]


def test_the_branch_is_the_t_half_run(env, ref_run):
    """The T/2 branch of the reference run (its full state at 40 % = step 8) against a run with max_steps 10: the same
    LR curve, data order and weights at its end; one complete final eval; the branch record in summary.json and its
    own started_utc."""
    m = load_script("04_distill")
    assert m.main(["--config", write_config(env, "ctc-half", merged(REF_OVER, {"schedule": {"max_steps": 10}}))]) == 0
    half = one_run(env["root"], "ctc-half")
    path = env["root"] / "ctc-branch.json"
    path.write_text(json.dumps(merged(env["base"], dict(REF_OVER, run_name="ctc-ref",
                                                        branch={"parent": str(ref_run)})), indent=1), encoding="utf-8")
    assert m.main(["--config", str(path)]) == 0
    branch = one_run(env["root"], "ctc-ref-half")
    s = summary_of(branch)
    assert s["branch"] == dict(parent_run_id=ref_run.name, resume_step=8, end_step=10, t_c=8.0)
    assert s["family"] == "ctc" and s["started_utc"] != summary_of(ref_run)["started_utc"]
    assert [(e["at_step"], e["complete"], e["final"]) for e in events(branch, "eval")] == [(10, True, True)]
    got = pd.concat([steps_of(ref_run).query("step <= 8"), steps_of(branch)])
    np.testing.assert_array_equal(got["opt/lr"].to_numpy(), steps_of(half)["opt/lr"].to_numpy())
    ub, uh = utts_of(branch), utts_of(half)
    for step in (9, 10):
        x, y = ub[ub["step"] == step].sort_values("id"), uh[uh["step"] == step].sort_values("id")
        assert len(x) and x["id"].tolist() == y["id"].tolist() and x["masked_frac"].tolist() == y["masked_frac"].tolist()
    wa = torch.load(branch / "checkpoints" / "full_step_10" / "model.pt", weights_only=True)
    wh = torch.load(half / "checkpoints" / "full_step_10" / "model.pt", weights_only=True)
    for k in wa:
        torch.testing.assert_close(wa[k], wh[k], rtol=1e-5, atol=1e-6, msg=k)


def test_lr_probe_ctc(env):
    """An LR probe of a CTC student: metrics only, and at the end the CTC objective w_kl x KL + w_ctc x CTC per target
    token, teacher-forced on the complete gate sets, pooled and per set - the same numbers kitsune.ctc_eval gives for
    the probe's final weights."""
    from kitsune import ctc_eval as ce
    from kitsune import ctc_student as CS

    m = load_script("04_distill")
    over = {"lr_probe": {"enabled": True}, "schedule": {"max_steps": 6, "warmup_steps": 2},
            "eval": {"every_steps": 2, "mini": {"every_steps": 3}}, "ckpt": {"weights_every_steps": 2}}
    assert m.main(["--config", write_config(env, "ctc-probe", over)]) == 0
    run = one_run(env["root"], "ctc-probe")
    kinds = [e["kind"] for e in events(run)]
    assert "eval_mini" not in kinds and "eval" not in kinds and kinds.count("eval_start") == 1
    assert not list((run / "checkpoints").glob("step_*"))
    res = next(e for e in events(run) if e["kind"] == "lr_probe_result")
    s = summary_of(run)
    assert s["lr_probe"]["family"] == "ctc" == res["family"] and s["lr_probe"]["w_ctc"] == 0.8
    assert res["objective"] == pytest.approx(s["lr_probe"]["kl"] + 0.8 * s["lr_probe"]["ctc"])
    model = CS.load_ctc_student(env["student"], "cpu")
    model.load_state_dict(torch.load(run / "checkpoints" / "full_step_6" / "model.pt", weights_only=True))
    store = m.build_eval_store(cfg_of(env), Log(), frames=True)
    tf, _, _, _ = ce.ctc_eval(model, store, CS.ctc_features(env["student"]).logmel, "cpu", 20.0, amp=False,
                              decode=False)
    assert res["objective"] == pytest.approx(ce.combined_loss_ctc(tf, 1.0, 0.8)["value"], rel=1e-6)
    for x in EVAL:
        d = tf["sets"][x]
        assert res["per_set"][x] == pytest.approx(d["kl"] + 0.8 * d["ctc"], rel=1e-6)


def test_the_padded_row_gate_fails_the_ctc_smoke(env):
    """The padded-vs-alone gate on the frame log-probs stops the run when violated (a bound of -1 here)."""
    m = load_script("04_distill")
    with pytest.raises(m.SmokeFailed, match="padded rows differ"):
        m.main(["--config", write_config(env, "ctc-padgate", merged(REF_OVER, {"smoke": {"pad_max_mean_kl": -1}}))])
    pad = events(one_run(env["root"], "ctc-padgate"), "smoke_padded_row")[0]
    assert pad["ok"] is False and pad["unit"] == "frames" and pad["n_tok"] > pad["n_utts"] and pad["kl_mean"] < 1e-5
