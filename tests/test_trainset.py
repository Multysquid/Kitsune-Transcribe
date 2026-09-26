"""Tests for kitsune/trainset.py and scripts/make_selection.py.

Everything runs on a synthetic corpus in the repo's exact on-disk formats (tests/fixtures.py) and is checked against
the fixture's ground truth, not against the code under test. Two tests read real files (one teacher shard, one data
shard) under fixtures.REAL and skip when they are absent (fail under KITSUNE_REQUIRE_REAL_DATA=1). CPU only.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import (  # noqa: E402
    REAL, ROOT, load_script, make_fake_corpus, make_fake_selection, need_real, no_real_data,
)

from kitsune.audio import decode_audio  # noqa: E402
from kitsune.store import SCHEMA  # noqa: E402
from kitsune.trainset import (  # noqa: E402
    AudioBatchDataset, StepPlanner, Utt, build_stores, eval_batches, eval_store, load_stores, make_loader,
)

TRAIN, EVAL = ["src_a", "src_b"], ["eval_x", "eval_y"]
REAL_NPZ = REAL / "teacher_out" / "reazon_small" / "train-00000.npz"
REAL_SHARD = REAL / "data" / "shards" / "reazon_small" / "train-00000.parquet"
ms = load_script("make_selection")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return make_fake_corpus(
        tmp_path_factory.mktemp("corpus"),
        {"src_a": (90, "train"), "src_b": (30, "train"), "eval_x": (24, "eval"), "eval_y": (12, "eval")},
        rows_per_shard=32, seed=7, truncated={"src_a": 4, "src_b": 1, "eval_x": 1}, null_agree={"src_a": 2},
        missing_audio={"src_a": 3, "eval_x": 1}, teacher_skipped={"src_a": 2}, extra_shard_rows={"src_a": 6},
        no_second=("eval_x", "eval_y"),
    )


@pytest.fixture(scope="module")
def selection(corpus):
    return make_fake_selection(corpus, greedy_n=10, probe_n=12)


@pytest.fixture(scope="module")
def train_store(corpus, selection, tmp_path_factory):
    return build_stores(selection, corpus.data, corpus.teacher_out, tmp_path_factory.mktemp("cache") / "train",
                        TRAIN, ["train"])


@pytest.fixture(scope="module")
def eval_st(corpus, selection, tmp_path_factory):
    return eval_store(selection, corpus.data, corpus.teacher_out, tmp_path_factory.mktemp("cache") / "eval", EVAL)


def first_keys(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return list(json.loads(f.readline()))


def expected_reason(u) -> str:
    if u.split == "eval":
        return "kept" if u.has_audio else "no_audio"
    if u.truncated:
        return "truncated"
    if u.agree is None:
        return "no_agree"
    if u.agree > 0.5:
        return "agree>0.5"
    return "kept" if u.has_audio else "no_audio"


# ------------------------------------------------------------------------------------------------ formats / selection


def test_fixture_matches_real_formats(corpus):
    """The fixture writes what 02/02b/01 write: same npz keys/dtypes/ranks, jsonl keys, parquet schema, meta keys."""
    need_real(REAL_NPZ)
    real, fake = np.load(REAL_NPZ), np.load(corpus.teacher_out / "src_a" / "train-00000.npz")
    assert set(real.files) == set(fake.files)
    for key in real.files:
        assert (real[key].dtype.kind, real[key].ndim) == (fake[key].dtype.kind, fake[key].ndim), key
        if real[key].dtype.kind != "U":
            assert real[key].dtype == fake[key].dtype, key
    assert first_keys(REAL_NPZ.with_suffix(".jsonl")) == first_keys(corpus.teacher_out / "src_a" / "train-00000.jsonl")
    real_second = REAL / "second_out" / "reazon_small" / "train-00000.jsonl"
    if real_second.exists():
        assert first_keys(real_second) == first_keys(corpus.second_out / "src_a" / "train-00000.jsonl")
    real_meta = json.loads((REAL / "teacher_out" / "meta.json").read_text(encoding="utf-8"))
    fake_meta = json.loads((corpus.teacher_out / "meta.json").read_text(encoding="utf-8"))
    assert set(real_meta) | {"model_revision"} == set(fake_meta)  # 02 back-fills it into an older meta.json on resume
    assert pq.read_schema(corpus.data / "shards" / "src_a" / "train-00000.parquet").remove_metadata().equals(SCHEMA)
    if REAL_SHARD.exists():
        assert pq.read_schema(REAL_SHARD).remove_metadata().equals(SCHEMA)


def test_selection_rows_and_reasons(corpus, selection):
    sel = pd.read_parquet(selection)
    # candidates are exactly the teacher-labelled rows: no teacher-skipped rows, nothing from the untaught shard
    want = {i for s in TRAIN for i in corpus.ids(s, "train")} | {i for s in EVAL for i in corpus.ids(s, "eval")}
    assert set(sel["id"]) == want and len(sel) == len(want)
    got = dict(zip(sel["id"], sel["reason"]))
    assert {i: expected_reason(corpus.utts[i]) for i in want} == got
    assert (sel["keep"] == (sel["reason"] == "kept")).all()
    assert set(sel["reason"]) == {"kept", "truncated", "no_agree", "agree>0.5", "no_audio"}  # every rule exercised
    for r in sel.itertuples():
        u = corpus.utts[r.id]
        assert (r.source, r.split, r.teacher_file) == (u.source, u.split, f"{u.source}/{u.stem}")
        assert r.duration == np.float32(u.duration) and r.n_tok == len(u.tokens) and r.truncated == u.truncated
        assert r.teacher_cer == np.float32(u.cer)
        assert (np.isnan(r.agree) and u.agree is None) or r.agree == np.float32(u.agree)
    boundary = sel[sel["agree"] == 0.5]
    assert len(boundary) and boundary["keep"].all()  # agree <= agree_max is inclusive
    ev = sel[sel["split"] == "eval"]
    assert ev["truncated"].any() and ev[ev["truncated"]]["keep"].all()  # eval sets are never label-filtered
    assert ev["agree"].isna().all()


def test_selection_subsets_are_seeded(corpus, selection):
    sel = pd.read_parquet(selection)
    for s in EVAL:
        rows = sel[sel["source"] == s]
        assert rows["in_greedy_subset"].sum() == min(10, rows["keep"].sum())
        assert rows[rows["in_greedy_subset"]]["keep"].all() and not rows["in_probe"].any()
    for s in TRAIN:
        rows = sel[sel["source"] == s]
        assert rows["in_probe"].sum() == min(12, rows["keep"].sum())
        assert rows[rows["in_probe"]]["keep"].all() and not rows["in_greedy_subset"].any()
    args = (corpus.teacher_out, corpus.second_out, corpus.data, TRAIN, EVAL, 0.5)
    again = ms.build_selection(*args, seed=1234, greedy_n=10, probe_n=12)
    pd.testing.assert_frame_equal(sel.reset_index(drop=True), again, check_dtype=False)
    other = ms.build_selection(*args, seed=1235, greedy_n=10, probe_n=12)
    assert (other["reason"] == sel["reason"]).all()
    assert not (other["in_probe"] == sel["in_probe"]).all() or not (other["in_greedy_subset"] == sel["in_greedy_subset"]).all()


def test_selection_cli_output(corpus, tmp_path, capsys):
    out = tmp_path / "sel.parquet"
    make_fake_selection(corpus, out, agree_max=0.3, greedy_n=5, probe_n=5)
    printed = capsys.readouterr().out
    assert "agree>0.3" in printed and "no_audio" in printed and "kept" in printed
    t = pq.read_table(out)
    assert t.schema.field("agree").type == pa.float32() and t.column("agree").null_count > 0
    assert t.column_names == ms.COLUMNS
    meta = json.loads(t.schema.metadata[b"kitsune_selection"])
    assert meta["args"]["agree_max"] == 0.3 and meta["kept"]


def test_selection_partial_second_opinion(corpus, tmp_path):
    """A source in partial_second_opinion trains on its judged shards only: the rows of a shard with no second-opinion
    file are not_judged (the run's decision) instead of no_agree; nothing else changes, and the CLI records the list
    (vast/launch.py compares it with the run config)."""
    src = TRAIN[0]
    second = tmp_path / "second_out"
    shutil.copytree(corpus.second_out, second)
    shard = sorted((second / src).glob("train-*.jsonl"))[0]
    shard.unlink()
    args = (corpus.teacher_out, second, corpus.data, TRAIN, EVAL, 0.5)
    plain = ms.build_selection(*args, greedy_n=10, probe_n=12)
    part = ms.build_selection(*args, greedy_n=10, probe_n=12, partial_second_opinion=[src])
    unjudged = (part["teacher_file"] == f"{src}/{shard.stem}").to_numpy()
    live = unjudged & ~part["truncated"].to_numpy()
    assert live.any()
    assert set(plain["reason"][live]) == {"no_agree"} and set(part["reason"][live]) == {"not_judged"}
    assert not part["keep"][unjudged].any()
    assert (part["reason"][~unjudged] == plain["reason"][~unjudged]).all()
    out = tmp_path / "sel.parquet"
    ms.main(["--sources", *TRAIN, "--eval-sets", *EVAL, "--partial-second-opinion", src, "--teacher-out",
             str(corpus.teacher_out), "--second-out", str(second), "--data", str(corpus.data), "--out", str(out)])
    meta = json.loads(pq.read_schema(out).metadata[b"kitsune_selection"])
    assert meta["args"]["partial_second_opinion"] == [src]
    assert "not_judged" in set(pd.read_parquet(out)["reason"])
    with pytest.raises(SystemExit):
        ms.main(["--sources", *TRAIN, "--partial-second-opinion", "nope", "--out", str(tmp_path / "bad.parquet")])


def test_selection_per_source_agree_threshold(corpus, tmp_path, capsys):
    """Each source's agree comes from a different second model, so the threshold can differ per source."""
    assert len(TRAIN) >= 2
    strict, loose = TRAIN[0], TRAIN[1]
    out = tmp_path / "sel.parquet"
    make_fake_selection(corpus, out, agree_max=0.5, greedy_n=5, probe_n=5,
                        extra_args=("--agree-max-source", f"{strict}=0.1"))
    sel = pd.read_parquet(out)
    s, l = sel[sel["source"] == strict], sel[sel["source"] == loose]
    assert set(s["reason"]) >= {"agree>0.1"} and "agree>0.5" not in set(s["reason"])
    assert not (s["keep"] & (s["agree"] > 0.1)).any()
    assert "agree>0.1" not in set(l["reason"]) and not (l["keep"] & (l["agree"] > 0.5)).any()
    assert (l["keep"] & (l["agree"] > 0.1)).any()  # the override did not leak into the other source
    with pytest.raises(SystemExit):
        make_fake_selection(corpus, tmp_path / "bad.parquet", extra_args=("--agree-max-source", "nope=0.2"))


# ------------------------------------------------------------------------------------------------------- stores


def test_build_stores_contents_and_idempotence(corpus, selection, tmp_path):
    logs = []
    cache, data = tmp_path / "cache", tmp_path / "data"
    shutil.copytree(corpus.data, data)  # this test adds a shard; keep the shared corpus untouched
    st = build_stores(selection, data, corpus.teacher_out, cache, TRAIN, ["train"], log=logs.append)
    sel = pd.read_parquet(selection)
    kept = sel[sel["keep"] & sel["source"].isin(TRAIN)]
    assert [u.id for u in st.utts] and {u.id for u in st.utts} == set(kept["id"])
    assert [u.source for u in st.utts] == sorted([u.source for u in st.utts])  # sources in the given order
    assert st.info["n_utts"] == len(st) and st.info["dropped"]["no_audio"]["n"] == 0
    assert st.info["n_targets"] == sum(u.n_tok for u in st.utts) == np.load(cache / "targets_tokens.npy").shape[0]
    assert abs(st.hours - kept["duration"].sum() / 3600) < 1e-6
    assert {u.id for u in st.utts if u.in_probe} == set(kept[kept["in_probe"]]["id"])
    frame = st.frame()
    assert frame["id"].tolist() == [u.id for u in st.utts]
    assert frame["hyp"].tolist() == [corpus.utts[u.id].hyp for u in st.utts]
    assert frame["ref"].tolist() == [corpus.utts[u.id].text for u in st.utts]

    stamp = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    # a new shard appearing (a download still appending) does not invalidate a cache that found all its audio
    shutil.copy(data / "shards" / "src_a" / "train-00000.parquet", data / "shards" / "src_a" / "train-00099.parquet")
    st2 = build_stores(selection, data, corpus.teacher_out, cache, TRAIN, ["train"], log=logs.append)
    assert "reusing" in logs[-1] and {p.name: p.stat().st_mtime_ns for p in cache.iterdir()} == stamp
    assert [u.id for u in st2.utts] == [u.id for u in st.utts] and st2.info["fingerprint"] == st.info["fingerprint"]

    # a selection made without the audio check keeps rows whose audio is gone: the build joins by id and drops them
    sel2 = make_fake_selection(corpus, tmp_path / "noaudio.parquet", greedy_n=10, probe_n=12,
                               extra_args=("--skip-audio-check",))
    st3 = build_stores(sel2, data, corpus.teacher_out, cache, TRAIN, ["train"], log=logs.append)
    lost = [i for s in TRAIN for i in corpus.ids(s, "train", audio=False) if expected_reason(corpus.utts[i]) == "no_audio"]
    assert lost and "DROPPED" in logs[-1]
    assert st3.info["fingerprint"] != st.info["fingerprint"]
    assert st3.info["dropped"]["no_audio"]["n"] == len(lost) and set(st3.info["dropped"]["no_audio"]["ids"]) == set(lost)
    assert {u.id for u in st3.utts} == {u.id for u in st.utts}
    assert load_stores(cache).info["fingerprint"] == st3.info["fingerprint"]
    few = [u.id for u in st.utts[::7]] + ["not/in/the/selection"]
    st4 = build_stores(selection, data, corpus.teacher_out, tmp_path / "few", TRAIN, ["train"], ids=few, log=logs.append)
    assert [u.id for u in st4.utts] == few[:-1] and st4.info["fingerprint"] != st.info["fingerprint"]
    build_stores(sel2, data, corpus.teacher_out, cache, TRAIN, ["train"], log=logs.append)
    assert "reusing" in logs[-1]
    # ... but after a build that dropped rows, a new shard might hold them: rebuild
    shutil.copy(data / "shards" / "src_b" / "train-00000.parquet", data / "shards" / "src_b" / "train-00099.parquet")
    build_stores(sel2, data, corpus.teacher_out, cache, TRAIN, ["train"], log=logs.append)
    assert "built" in logs[-1]
    # a shard that contributed audio changed size: rebuild
    shutil.copy(data / "shards" / "src_b" / "train-00000.parquet", data / "shards" / "src_a" / "train-00000.parquet")
    build_stores(selection, data, corpus.teacher_out, cache, TRAIN, ["train"], log=logs.append)
    assert "built" in logs[-1]


def test_build_stores_rebuilds_when_teacher_output_changes_in_place(corpus, selection, tmp_path):
    """A teacher re-run that keeps the tokens but changes the log-probs writes an npz of the same size (np.savez does
    not compress), and a changed ref/hyp lives only in the .jsonl: both must still invalidate the cache."""
    logs = []
    teacher, cache = tmp_path / "teacher_out", tmp_path / "cache"
    shutil.copytree(corpus.teacher_out, teacher)
    st = build_stores(selection, corpus.data, teacher, cache, TRAIN, ["train"], log=logs.append)
    lp0 = np.load(cache / "targets_topk_lp.npy")
    uid = st.utts[0].id
    npz = teacher / f"{pd.read_parquet(selection).set_index('id').loc[uid, 'teacher_file']}.npz"
    with np.load(npz) as f:
        z = dict(f)
    size0 = npz.stat().st_size

    np.savez(npz, **z)  # the same content written again: no rebuild
    build_stores(selection, corpus.data, teacher, cache, TRAIN, ["train"], log=logs.append)
    assert "reusing" in logs[-1]

    z["topk_logprob"] = (z["topk_logprob"] - 0.5).astype(z["topk_logprob"].dtype)
    np.savez(npz, **z)
    assert npz.stat().st_size == size0
    build_stores(selection, corpus.data, teacher, cache, TRAIN, ["train"], log=logs.append)
    assert "built" in logs[-1]
    assert not np.array_equal(np.load(cache / "targets_topk_lp.npy"), lp0)

    jl = npz.with_suffix(".jsonl")
    rows = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if r["id"] == uid:
            r["hyp"] += "改"
    jl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    st2 = build_stores(selection, corpus.data, teacher, cache, TRAIN, ["train"], log=logs.append)
    assert "built" in logs[-1]
    assert st2.frame().set_index("id").loc[uid, "hyp"] == corpus.utts[uid].hyp + "改"


def check_alignment(corpus, st, idx):
    ds = AudioBatchDataset(st)
    b = ds[idx]
    P, prompt = len(corpus.prompt), torch.tensor(corpus.prompt)
    assert b["ids"] == [st.utts[i].id for i in idx] and b["index"].tolist() == list(idx) and not b["dropped"]
    L = b["decoder_input_ids"].shape[1]
    assert L == P - 1 + max(len(corpus.utts[i].tokens) for i in b["ids"])
    n = 0
    for r, uid in enumerate(b["ids"]):
        u = corpus.utts[uid]
        T = len(u.tokens)
        dec = b["decoder_input_ids"][r]
        assert torch.equal(dec[:P], prompt)
        assert torch.equal(dec[P:P + T - 1], torch.from_numpy(u.tokens[:-1].astype(np.int64)))
        assert (dec[P + T - 1:] == corpus.pad).all()
        assert b["dec_mask"][r].tolist() == [1] * (P + T - 1) + [0] * (L - P - T + 1)
        assert b["n_tok"][r] == T and b["durations"][r] == np.float32(u.duration) and b["sources"][r] == u.source
        rows, pos = b["tgt_row"][n:n + T], b["tgt_pos"][n:n + T]
        assert (rows == r).all() and pos.tolist() == list(range(P - 1, P - 1 + T))
        assert torch.equal(b["top_idx"][n:n + T], torch.from_numpy(u.topk_idx.astype(np.int64)))
        assert torch.equal(b["top_lp"][n:n + T], torch.from_numpy(u.topk_lp.astype(np.float32)))
        assert torch.equal(b["top_idx"][n:n + T, 0], torch.from_numpy(u.tokens.astype(np.int64)))
        # teacher forcing: the target at step t is the decoder INPUT at the next position
        assert torch.equal(dec[pos[:-1] + 1], b["top_idx"][n:n + T - 1, 0])
        assert (u.tokens[-1] == corpus.eos) != u.truncated
        wav = decode_audio(u.audio)
        assert b["lengths"][r] == len(wav) == round(u.duration * 16000)
        assert torch.equal(b["wave"][r, :len(wav)], torch.from_numpy(wav)) and (b["wave"][r, len(wav):] == 0).all()
        n += T
    assert n == b["top_idx"].shape[0] == b["top_lp"].shape[0] == int(b["tgt_mask"].sum())
    h = torch.randn(len(idx), L, 3)
    assert torch.equal(h[b["tgt_mask"]], h[b["tgt_row"], b["tgt_pos"]])
    return b


def test_alignment_train(corpus, train_store):
    st = train_store
    n_tok = np.array([u.n_tok for u in st.utts])
    idx = [int(n_tok.argmin()), int(n_tok.argmax()), 0, len(st) - 1, len(st) // 2]
    idx = list(dict.fromkeys(idx))
    check_alignment(corpus, st, idx)
    check_alignment(corpus, st, [idx[1]])  # a lone utterance has no padding at all


def test_alignment_eval_with_truncated_row(corpus, eval_st):
    trunc = [i for i, u in enumerate(eval_st.utts) if u.truncated]
    assert trunc and set(u.source for u in eval_st.utts) == set(EVAL)
    b = check_alignment(corpus, eval_st, trunc + [0, len(eval_st) - 1])
    assert all(u.split == "eval" for u in eval_st.utts)
    assert sum(u.in_greedy_subset for u in eval_st.utts) == 20  # 10 per eval set
    assert b["agree"].isnan().all()


def test_store_protocol_for_evaluate(corpus, eval_st):
    """kitsune.evaluate duck-types a store as utts[i] / wave(i) / targets(i); subsets index their own utts."""
    sub = eval_st.subset(eval_st.indices(in_greedy_subset=True, source="eval_y"))
    assert len(sub) == 10 and all(u.source == "eval_y" and u.in_greedy_subset for u in sub.utts)
    for i in (0, len(sub) - 1):
        u = corpus.utts[sub.utts[i].id]
        assert np.array_equal(sub.wave(i), decode_audio(u.audio))
        tok, idx, lp = sub.targets(i)
        assert tok.dtype == np.int16 and idx.dtype == np.int16 and lp.dtype == np.float16
        assert np.array_equal(tok, u.tokens) and np.array_equal(idx, u.topk_idx) and np.array_equal(lp, u.topk_lp)
    assert sub.frame()["id"].tolist() == [u.id for u in sub.utts]


def test_dataset_pickles_small_and_opens_lazily(train_store):
    ds = AudioBatchDataset(train_store)
    first = ds[[0, 1]]
    assert ds._mm is not None
    blob = pickle.dumps(ds)
    assert len(blob) < 64 * 1024  # a few small arrays; no memmap, no audio
    ds2 = pickle.loads(blob)
    assert ds2._mm is None
    again = ds2[[0, 1]]
    assert all(torch.equal(first[k], again[k]) for k in first if isinstance(first[k], torch.Tensor))


def test_dataset_drops_undecodable_audio(train_store, tmp_path):
    cache = tmp_path / "cache"
    shutil.copytree(train_store.cache_dir, cache)
    st = load_stores(cache)
    u = st.utts[3]
    with open(cache / "audio.bin", "r+b") as f:  # garble one utterance's FLAC bytes
        f.seek(u.audio_off)
        f.write(b"\0" * u.audio_len)
    b = AudioBatchDataset(st)[[2, 3, 4]]
    assert b["dropped"] == [u.id] and b["ids"] == [st.utts[2].id, st.utts[4].id]
    assert b["wave"].shape[0] == 2 and set(b["tgt_row"].tolist()) == {0, 1}
    assert b["top_idx"].shape[0] == st.utts[2].n_tok + st.utts[4].n_tok


# ------------------------------------------------------------------------------------------------------ planner


def synthetic_utts(n=3000, seed=0, long_tok=0) -> list[Utt]:
    rng = np.random.default_rng(seed)
    dur = np.clip(rng.lognormal(np.log(5.5), 0.7, n), 0.3, 30.0)
    tok = np.maximum(1, np.round(dur * rng.uniform(2.5, 5.5, n))).astype(int) + 1
    tok[:long_tok] = 250  # decoder input 259 > max_dec_len
    return [Utt(f"s{i % 2}/{i}", f"s{i % 2}", float(d), int(t), 0, 0, 0) for i, (d, t) in enumerate(zip(dur, tok))]


def test_planner_limits():
    utts = synthetic_utts(long_tok=3)
    dur = np.array([u.duration for u in utts])
    dec = np.array([u.n_tok for u in utts]) + 9
    micro, target = 20.0, 300.0
    pl = StepPlanner(utts, step_audio_s=target, micro_audio_s=micro, max_dec_len=200, pool_micro=10, seed=3)
    plan = pl.epoch_plan(0)
    flat = [i for step in plan for mb in step for i in mb]
    assert pl.n_excluded == 3 and sorted(flat) == list(range(3, len(utts)))  # every eligible utt exactly once
    mbs = [mb for step in plan for mb in step]
    assert any(dur[mb].max() > micro for mb in mbs)
    for mb in mbs:
        if dur[mb].max() > micro:
            assert len(mb) == 1  # an over-long utterance travels alone
        else:
            assert dur[mb].max() * len(mb) <= micro
    half = max(dur[mb].sum() for mb in mbs) / 2
    step_s = np.array([dur[[i for mb in s for i in mb]].sum() for s in plan])
    off = np.abs(step_s - target) > half + 1e-9
    assert off.sum() <= 1 and (step_s[off] < target).all()  # only the epoch's remainder step may fall short
    st = pl.stats
    assert st["steps"] == len(plan) and st["pad_eff_audio"] > 0.85 and st["excluded_dec_len"] == 3
    assert abs(st["real_h"] - dur[3:].sum() / 3600) < 1e-9

    capped = StepPlanner(utts, step_audio_s=target, micro_audio_s=micro, pool_micro=10, seed=3, max_micro_tokens=400)
    for mb in (mb for step in capped.epoch_plan(0) for mb in step):
        assert len(mb) == 1 or dec[mb].max() * len(mb) <= 400
    worst = pl.worst_micro_batches(plan)
    assert dur[worst["longest"]].max() == dur[3:].max()  # the micro-batch holding the longest eligible utterance
    assert set(worst) == {"longest", "most_dec_positions", "most_targets"}


def test_planner_is_deterministic():
    utts = synthetic_utts(800)
    kw = dict(step_audio_s=200, micro_audio_s=40, pool_micro=5)
    a, b = StepPlanner(utts, seed=5, **kw), StepPlanner(utts, seed=5, **kw)
    assert a.epoch_plan(0) == b.epoch_plan(0) == a.epoch_plan(0)
    assert a.epoch_plan(1) == b.epoch_plan(1) != a.epoch_plan(0)
    assert StepPlanner(utts, seed=6, **kw).epoch_plan(0) != a.epoch_plan(0)


def test_planner_resume_across_epochs():
    utts = synthetic_utts(300)
    kw = dict(step_audio_s=300, micro_audio_s=60, pool_micro=4, seed=11)
    ref = StepPlanner(utts, **kw)
    n0 = len(ref.epoch_plan(0))
    it = ref.iter_steps()
    uninterrupted = [next(it) for _ in range(n0 + 5)]
    assert [(e, j) for e, j, _ in uninterrupted[n0 - 1:n0 + 1]] == [(0, n0 - 1), (1, 0)]

    run = StepPlanner(utts, **kw)
    it = run.iter_steps()
    for _ in range(n0 - 2):  # stop two steps before the epoch boundary
        e, j, _ = next(it)
        run.step_done(e, j)
    state = run.state_dict()
    assert (state["epoch"], state["step_in_epoch"]) == (0, n0 - 2)
    resumed = StepPlanner(utts, **kw)
    resumed.load_state_dict(state)
    it2 = resumed.iter_steps()
    assert [next(it2) for _ in range(7)] == uninterrupted[n0 - 2:n0 + 5]
    for e, j, _ in uninterrupted[n0 - 2:n0]:
        resumed.step_done(e, j)
    assert (resumed.epoch, resumed.step_in_epoch) == (1, 0)

    with pytest.raises(ValueError):
        StepPlanner(utts, **{**kw, "seed": 12}).load_state_dict(state)
    loose = StepPlanner(utts[:-1], **kw)
    loose.load_state_dict(state, strict=False)
    assert (loose.epoch, loose.step_in_epoch) == (0, n0 - 2)


def test_planner_weights():
    utts = synthetic_utts(400)
    pl = StepPlanner(utts, step_audio_s=300, micro_audio_s=60, seed=2, weights={"s0": 2.0, "s1": 0.5})
    plan = pl.epoch_plan(0)
    counts = np.bincount([i for s in plan for mb in s for i in mb], minlength=len(utts))
    s0 = np.array([u.source == "s0" for u in utts])
    assert (counts[s0] == 2).all()
    assert set(counts[~s0]) <= {0, 1} and 0.3 < counts[~s0].mean() < 0.7
    assert StepPlanner(utts, step_audio_s=300, micro_audio_s=60, seed=2, weights={"s0": 2.0, "s1": 0.5}).epoch_plan(0) == plan


def test_eval_batches():
    utts = synthetic_utts(500)
    dur = np.array([u.duration for u in utts])
    sub = list(range(0, 500, 3))
    batches = eval_batches(utts, 60.0, sub)
    assert sorted(i for b in batches for i in b) == sub
    assert all(dur[b].max() * len(b) <= 60.0 for b in batches)
    order = [i for b in batches for i in b]
    assert (np.diff(dur[order]) <= 0).all()
    assert eval_batches(utts, 60.0, sub) == batches


# ------------------------------------------------------------------------------------------------------- loader


def test_loader_list_plan_in_process(train_store):
    ds = AudioBatchDataset(train_store)
    plan = StepPlanner(train_store.utts, step_audio_s=20, micro_audio_s=8, pool_micro=3, seed=1).epoch_plan(0)
    # timeout_s is for workers only: in-process loading (shm_cap can choose 0 workers) asserts a zero timeout
    got = list(make_loader(ds, plan, num_workers=0, start_step=2, timeout_s=30))
    assert [k for k, _ in got] == list(range(2, len(plan)))
    for (k, mbs) in got:
        assert [m["ids"] for m in mbs] == [[train_store.utts[i].id for i in mb] for mb in plan[k]]


def test_loader_spawn_workers_across_epochs(train_store):
    """num_workers=2 under `spawn` (the Windows default; used on Linux too) gives exactly the in-process batches,
    in order, across an epoch boundary, starting from a resumed planner position."""
    ds = AudioBatchDataset(train_store)
    pl = StepPlanner(train_store.utts, step_audio_s=20, micro_audio_s=8, pool_micro=3, seed=4)
    n0 = len(pl.epoch_plan(0))
    pl.load_state_dict(dict(pl.state_dict(), epoch=0, step_in_epoch=n0 - 2))
    loader = make_loader(ds, pl, num_workers=2, prefetch=2)
    got = []
    for key, mbs in loader:
        got.append((key, mbs))
        pl.step_done(*key)
        if len(got) == 4:
            break
    loader.close()
    assert [k for k, _ in got] == [(0, n0 - 2), (0, n0 - 1), (1, 0), (1, 1)]
    assert (pl.epoch, pl.step_in_epoch) == (1, 2)
    for (e, j), mbs in got:
        ref = [ds[mb] for mb in StepPlanner(train_store.utts, step_audio_s=20, micro_audio_s=8, pool_micro=3,
                                            seed=4).epoch_plan(e)[j]]
        assert len(mbs) == len(ref)
        for m, r in zip(mbs, ref):
            assert m["ids"] == r["ids"]
            for k in r:
                if isinstance(r[k], torch.Tensor):
                    assert torch.equal(m[k], r[k]), k


class _NoShm:
    """Fails to pickle the way a full /dev/shm fails a worker's queue feeder (torch 2.14's message)."""

    def __reduce__(self):
        raise RuntimeError("unable to allocate shared memory(shm) for file </torch_1_2_3>: No space left on device "
                           "(28)")


class _FeederDropDataset(torch.utils.data.Dataset):
    """Micro-batch [1] never reaches the trainer: the worker's feeder thread prints the error and drops it, and the
    worker lives on (so the DataLoader sees no dead worker)."""

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return dict(ids=list(idx), x=_NoShm() if list(idx) == [1] else 0)


def test_loader_times_out_on_a_dropped_worker_batch():
    """A worker micro-batch that never arrives raises "DataLoader timed out" after timeout_s instead of blocking the
    trainer until the watchdog (the loader yields in order, so it would wait for that batch forever). The timeout
    covers the first wait's spawn and imports too: 8-11 s measured on an idle laptop, ~26 s beside a live training
    run and over 30 s late in the full suite there, so 90 s (production: perf.loader_timeout_s = 600)."""
    loader = make_loader(_FeederDropDataset(), [[[0]], [[1]], [[2]]], num_workers=1, prefetch=2, timeout_s=90)
    try:
        key, mbs = next(loader)
        assert key == 0 and mbs[0]["ids"] == [0]
        with pytest.raises(RuntimeError, match="timed out"):
            next(loader)
    finally:
        loader.close()


class _WorkerDiesDataset(torch.utils.data.Dataset):
    """The worker process dies on micro-batch [1] without a Python exception (a native decoder crash, a kill)."""

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        if list(idx) == [1]:
            os._exit(1)
        return dict(ids=list(idx))


def test_loader_reports_a_dead_worker_long_before_its_timeout():
    """torch checks for a dead worker only when a wait ends, and on Windows that poll is the only check. One
    timeout_s-long wait (production: 600 s) hid a crashed worker for all of it; the wait now runs in 5 s slices, so the
    worker's death is reported within seconds while timeout_s still bounds a dropped micro-batch. prefetch=1: the
    worker gets micro-batch [1] only after [0] has arrived; with more in flight its os._exit could land before [0] left
    its queue's feeder thread, and the death would surface one next() early (seen under full-suite load)."""
    import time

    loader = make_loader(_WorkerDiesDataset(), [[[0]], [[1]], [[2]]], num_workers=1, prefetch=1, timeout_s=90)
    try:
        key, mbs = next(loader)
        assert key == 0 and mbs[0]["ids"] == [0]
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            next(loader)
        assert time.monotonic() - t0 < 30
    finally:
        loader.close()


# ---------------------------------------------------------------------------------------------------- real data


def test_real_shard_alignment(tmp_path):
    """A tiny store from real rows: the join, offsets and decode work on the real formats (read-only on data/)."""
    need_real(REAL_NPZ, REAL_SHARD)
    z = np.load(REAL_NPZ)
    rows = ms.read_jsonl(REAL_NPZ.with_suffix(".jsonl"))[:12]
    off = z["tok_offsets"]
    sel = pd.DataFrame(dict(
        id=[r["id"] for r in rows], source="reazon_small", split="train", teacher_file="reazon_small/train-00000",
        duration=z["duration"][:12], n_tok=np.diff(off)[:12].astype(np.int32), truncated=[r["truncated"] for r in rows],
        agree=np.float32(0.0), teacher_cer=z["cer"][:12], keep=True, reason="kept", in_greedy_subset=False,
        in_probe=False))
    ms.write_selection(sel, tmp_path / "sel.parquet", {})
    st = build_stores(tmp_path / "sel.parquet", REAL / "data", REAL / "teacher_out", tmp_path / "cache",
                      ["reazon_small"], ["train"])
    assert [u.id for u in st.utts] == sel["id"].tolist()
    ds = AudioBatchDataset(st)
    b = ds[list(range(len(st)))]
    P = len(st.info["prompt"])
    assert st.info["prompt"] == [int(x) for x in z["prompt"]]
    n = 0
    for r in range(len(st)):
        s, e = int(off[r]), int(off[r + 1])
        assert torch.equal(b["top_idx"][n:n + e - s], torch.from_numpy(z["topk_idx"][s:e].astype(np.int64)))
        assert torch.equal(b["top_lp"][n:n + e - s], torch.from_numpy(z["topk_logprob"][s:e].astype(np.float32)))
        assert torch.equal(b["decoder_input_ids"][r, P:P + e - s - 1], torch.from_numpy(z["tokens"][s:e - 1].astype(np.int64)))
        assert abs(int(b["lengths"][r]) - float(z["duration"][r]) * 16000) <= 1
        n += e - s
    assert n == b["top_idx"].shape[0]


def test_real_data_guard(tmp_path, monkeypatch):
    """The real-data tests skip when the data is absent (a git worktree has none), but FAIL under
    KITSUNE_REQUIRE_REAL_DATA=1, so a verification run cannot pass on an 's'; KITSUNE_REAL_DATA_ROOT moves REAL."""
    there = tmp_path / "there"
    there.touch()
    monkeypatch.delenv("KITSUNE_REQUIRE_REAL_DATA", raising=False)
    need_real(there)  # present: the test runs on
    with pytest.raises(pytest.skip.Exception, match="gone"):
        need_real(there, tmp_path / "gone")
    monkeypatch.setenv("KITSUNE_REQUIRE_REAL_DATA", "1")
    need_real(there)
    with pytest.raises(pytest.fail.Exception, match="gone"):
        need_real(there, tmp_path / "gone")
    with pytest.raises(pytest.fail.Exception, match="no download cache"):
        no_real_data("no download cache")
    env = dict(os.environ, KITSUNE_REAL_DATA_ROOT=str(tmp_path))  # REAL is read at import: a fresh interpreter
    out = subprocess.run([sys.executable, "-c", "import fixtures; print(fixtures.REAL)"], cwd=Path(__file__).parent,
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert Path(out) == tmp_path


def test_selection_filters_only_named_monitor_eval_sets(corpus, tmp_path):
    """--filter-eval-sets gives a monitor-only hold-out the train label rules; other eval sets stay unfiltered."""
    ev = EVAL[0]
    out = tmp_path / "sel.parquet"
    make_fake_selection(corpus, out, greedy_n=5, probe_n=5, extra_args=("--filter-eval-sets", ev))
    sel = pd.read_parquet(out)
    rows = sel[(sel["source"] == ev) & (sel["split"] == "eval")]
    assert not rows[rows["truncated"]]["keep"].any()  # truncated hold-out rows now dropped
    assert set(rows["reason"]) - {"kept", "truncated", "no_agree", "no_audio"} <= {r for r in rows["reason"] if r.startswith("agree>")}
    for other in EVAL[1:]:
        o = sel[(sel["source"] == other) & (sel["split"] == "eval")]
        assert set(o["reason"]) <= {"kept", "no_audio"}
    with pytest.raises(SystemExit):  # the pre-registered gate sets can never be filtered
        make_fake_selection(corpus, tmp_path / "bad.parquet", extra_args=("--filter-eval-sets", "eval_jsut"))


def test_selection_from_the_run_config_and_the_launch_check(corpus, tmp_path):
    """--config takes the sources, the eval sets and the selection_recipe from the run config (none of those flags
    may be given too); vast/launch.py accepts what it builds and refuses a rebuild with a flag forgotten."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kitsune_launch", ROOT / "vast" / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    strict = TRAIN[0]
    cfg = {"sources": TRAIN, "eval_sets": EVAL, "selection": str(tmp_path / "from_config.parquet"),
           "selection_recipe": {"agree_max": 0.5, "agree_max_source": [f"{strict}=0.1"], "filter_eval_sets": []}}
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    paths = ["--teacher-out", str(corpus.teacher_out), "--second-out", str(corpus.second_out), "--data",
             str(corpus.data), "--greedy-n", "5", "--probe-n", "5"]
    ms.main(["--config", str(cfg_path), *paths])  # --out: the config's selection
    out = Path(cfg["selection"])
    sel = pd.read_parquet(out)
    assert set(sel.loc[sel["split"] == "train", "source"]) == set(TRAIN)
    assert set(sel.loc[sel["split"] == "eval", "source"]) == set(EVAL)
    s = sel[sel["source"] == strict]
    assert "agree>0.1" in set(s["reason"]) and not (s["keep"] & (s["agree"] > 0.1)).any()
    args = json.loads(pq.read_schema(out).metadata[b"kitsune_selection"])["args"]
    assert (args["sources"], args["eval_sets"], args["agree_max"], args["agree_max_source"],
            args["filter_eval_sets"]) == (TRAIN, EVAL, 0.5, [f"{strict}=0.1"], [])
    # the fixture has train rows without a second opinion: only the no_agree problem, nothing about the recipe
    problems = launch.selection_problems(out, "sel", cfg)
    assert len(problems) == 1 and "as no_agree" in problems[0], problems
    # the same corpus without the per-source threshold (the flag forgotten): refused
    forgot = make_fake_selection(corpus, tmp_path / "forgot.parquet", greedy_n=5, probe_n=5)
    problems = launch.selection_problems(forgot, "sel", cfg)
    assert any("was built with agree_max_source [], the run config says [('src_a', 0.1)]" in p for p in problems)
    # a hold-out filtered by the recipe that has no second opinion keeps nothing: refused as well
    cfg2 = dict(cfg, selection=str(tmp_path / "filtered.parquet"),
                selection_recipe=dict(cfg["selection_recipe"], filter_eval_sets=[EVAL[0]]))
    cfg_path.write_text(json.dumps(cfg2), encoding="utf-8")
    ms.main(["--config", str(cfg_path), *paths])
    problems = launch.selection_problems(Path(cfg2["selection"]), "sel", cfg2)
    assert any(f"keeps no eval rows of ['{EVAL[0]}']" in p for p in problems), problems
    for flag in (["--sources", *TRAIN], ["--agree-max", "0.3"], ["--filter-eval-sets"]):
        with pytest.raises(SystemExit):
            ms.main(["--config", str(cfg_path), *flag, *paths])
    with pytest.raises(SystemExit):  # neither --config nor --sources
        ms.main(paths)


class _ShutdownDiesIter:
    """A DataLoader iterator stand-in: yields the sampler's micro-batches, then its worker pool dies in the shutdown
    join as the A100 shakedown's did (torch's SIGCHLD handler raising inside _shutdown_workers)."""

    def __init__(self, sampler, message):
        self.src, self.message = iter(sampler), message

    def __next__(self):
        return next(self.src)

    def _shutdown_workers(self):
        raise RuntimeError(self.message)


def _fake_loader(message):
    class Loader:
        def __init__(self, dataset, *, sampler, **kw):
            self.sampler = sampler

        def __iter__(self):
            return _ShutdownDiesIter(self.sampler, message)

    return Loader


def test_loader_close_survives_a_worker_dying_in_shutdown(monkeypatch):
    """A worker that dies while the pool shuts down (after the consumer is done) is a warning, not an error: the
    trainer's final loader.close() after a STOP must not turn a finished run into a failed one."""
    import kitsune.trainset as T

    monkeypatch.setattr(T.torch.utils.data, "DataLoader",
                        _fake_loader("DataLoader worker (pid 11267) is killed by signal: Aborted. "))
    loader = make_loader(None, [[[0], [1]], [[2]]], num_workers=0)
    first = next(loader)
    assert first[0] == 0 and len(first[1]) == 2
    with pytest.warns(RuntimeWarning, match="died while the loader shut down"):
        loader.close()


def test_loader_close_still_raises_other_shutdown_errors(monkeypatch):
    import kitsune.trainset as T

    monkeypatch.setattr(T.torch.utils.data, "DataLoader", _fake_loader("something else broke"))
    loader = make_loader(None, [[[0]], [[1]]], num_workers=0)
    next(loader)
    with pytest.raises(RuntimeError, match="something else broke"):
        loader.close()
