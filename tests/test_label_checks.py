"""tools/label_checks.py: K1-K12 pass on a synthetic label root that obeys the label box's formats, and each check fails
(or waits) on a planted defect.

The root is built with the writers the box itself uses: kitsune.parakeet_targets.pack_shard/write_shard for
parakeet_out, tests/fixtures.make_fake_corpus for the audio shards (real 16 kHz FLAC), teacher_out and second_out. Each
Parakeet utterance gets exactly the frame count the extractor gives its audio, and a greedy CTC path and a TDT path
that both emit the same tokens, so a sound root is sound for every check; a test plants one defect in a copy.
CPU only; the Parakeet feature extractor is transformers' default one (the pinned model's settings), not the model dir.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from fixtures import ROOT, make_fake_corpus
from kitsune import parakeet_targets as pt
from kitsune.audio import decode_audio
from kitsune.evaluate import corpus_cer
from kitsune.store import ids_sha256, read_manifest
from kitsune.text import cer as cer_fn

_spec = importlib.util.spec_from_file_location("kitsune_tool_label_checks", ROOT / "tools" / "label_checks.py")
lc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lc  # its dataclass resolves the module's string annotations through sys.modules
_spec.loader.exec_module(lc)

SOURCES = {"reazon_small": (24, "train"), "eval_jsut": (10, "eval"), "galgame": [(8, "eval"), (24, "train")]}
TRAIN, EVALS = ("reazon_small", "galgame"), ("eval_jsut", "galgame")
KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
K = lc.STUDY_SETTINGS["k_ctc"]


def fake_decode(seqs):
    """Stands in for the Parakeet processor's batch_decode: one kana per token id."""
    return ["".join(KANA[int(i) % len(KANA)] for i in s) for s in seqs]


def _topk_rows(rng, first, first_lp):
    """k distinct classes per row, column 0 = `first`, log-probs descending from `first_lp`, blank only as column 0."""
    idx = np.empty((len(first), K), np.int64)
    lp = np.empty((len(first), K), np.float32)
    for r, (c0, l0) in enumerate(zip(first, first_lp)):
        rest = [int(c) for c in rng.choice(pt.BLANK, size=K + 1, replace=False) if int(c) != c0][: K - 1]
        idx[r] = [c0] + rest
        lp[r] = np.concatenate([[l0], l0 - np.cumsum(rng.uniform(0.3, 1.5, K - 1))])
    return idx, lp


def pk_utt(rng, uid, duration, T, seq, *, tdt_seq=None):
    """A decoder trace over T frames: the greedy CTC path emits `seq` (tokens on every other frame, so equal
    neighbours stay apart), the TDT path emits `tdt_seq` (default: the same tokens). Obeys every check_shard rule."""
    seq = [int(x) for x in seq[: (T + 1) // 2]]
    col0 = np.full(T, pt.BLANK, np.int64)
    slots = np.sort(rng.choice(np.arange(0, T, 2), size=len(seq), replace=False)) if seq else np.zeros(0, np.int64)
    col0[slots] = seq
    blank_lp = np.log(rng.uniform(0.97, 1.0, T))
    unsure = (col0 == pt.BLANK) & (rng.random(T) < 0.2)  # blank on top but under the dense threshold
    blank_lp[unsure] = np.log(rng.uniform(0.5, 0.9, int(unsure.sum())))
    blank_lp[col0 != pt.BLANK] = np.log(rng.uniform(0.01, 0.3, int((col0 != pt.BLANK).sum())))
    dense = np.nonzero(blank_lp < np.log(0.95))[0]
    first_lp = np.where(col0[dense] == pt.BLANK, blank_lp[dense], np.log(rng.uniform(0.6, 0.95, len(dense))))
    cidx, clp = _topk_rows(rng, col0[dense], first_lp)
    frames, durs, emitted = [], [], []
    q, t = list(seq if tdt_seq is None else tdt_seq), 0
    while t < T:
        remaining = T - t  # kept > len(q) until the queue is empty, so every queued token is emitted
        if q and (remaining <= len(q) or rng.random() < 0.5):
            tok, d = q.pop(0), 0 if remaining <= len(q) + 1 else int(rng.integers(0, 2))
        else:
            tok, d = pt.BLANK, int(rng.integers(1, 5))
            if q:
                d = min(d, remaining - len(q))
        frames.append(t)
        durs.append(d)
        emitted.append(tok)
        t += d
    tidx, tlp = _topk_rows(rng, np.array(emitted), np.log(rng.uniform(0.5, 0.99, len(emitted))))
    return dict(id=uid, duration=duration, n_frames=T, truncated=False, step_frame=np.array(frames),
                step_dur=np.array(durs), step_forced=np.zeros(len(frames), bool), tdt_topk_idx=tidx, tdt_topk_lp=tlp,
                tdt_dur_lp=np.log(rng.dirichlet(np.ones(5), size=len(frames))).astype(np.float32),
                ctc_blank_lp=blank_lp.astype(np.float32), ctc_dense_frame=dense, ctc_topk_idx=cidx, ctc_topk_lp=clp)


def write_pk(out_dir, stem, utts, refs, shard_ids, *, tweak_rows=None):
    """pack_shard + write_shard with the jsonl 02p writes (hyp / ctc_hyp decoded by fake_decode)."""
    arrays = pt.pack_shard(utts, settings=lc.STUDY_SETTINGS, n_scanned=len(shard_ids), shard_ids_sha=ids_sha256(shard_ids))
    rows = []
    for u in utts:
        toks = pt.utt_tokens(u).tolist()
        ctc = pt.ctc_greedy(pt.ctc_col0(u["n_frames"], u["ctc_dense_frame"], u["ctc_topk_idx"]))
        h, ch = fake_decode([toks])[0], fake_decode([ctc])[0]
        rows.append(dict(id=u["id"], hyp=h, ctc_hyp=ch, ref=refs[u["id"]], cer=round(cer_fn(h, refs[u["id"]]), 4),
                         ctc_cer=round(cer_fn(ch, refs[u["id"]]), 4), duration=round(float(u["duration"]), 3),
                         n_tok=len(toks), n_steps=len(u["step_frame"]), n_frames=int(u["n_frames"]), n_forced=0,
                         truncated=False))
    if tweak_rows:
        rows = tweak_rows(rows)
    pt.write_shard(out_dir, stem, arrays, rows)


def build_root(base: Path) -> SimpleNamespace:
    """data/ (audio), labels/{teacher_out, second_out, parakeet_out} and the laptop kotoba file of galgame eval-00000."""
    fc = make_fake_corpus(base, sources=SOURCES, rows_per_shard=12, dur_range=(0.3, 3.0), no_second=("eval_jsut",))
    labels = base / "labels"
    labels.mkdir()
    shutil.move(str(fc.teacher_out), labels / "teacher_out")
    shutil.move(str(fc.second_out), labels / "second_out")
    kotoba = base / "kotoba_eval-00000.jsonl"
    shutil.copy(labels / "second_out" / "galgame" / "eval-00000.jsonl", kotoba)
    (labels / "parakeet_out").mkdir()
    (labels / "parakeet_out" / "meta.json").write_text(json.dumps(pt.build_meta(lc.STUDY_SETTINGS)), encoding="utf-8")
    rng = np.random.default_rng(7)
    shards = {}
    for info in read_manifest(fc.data):
        t = pq.read_table(fc.data / info.path, columns=["id", "audio", "text", "duration"]).to_pylist()
        stem = Path(info.path).stem
        utts = [pk_utt(rng, r["id"], r["duration"], lc.expected_n_frames(len(decode_audio(r["audio"]))),
                       rng.integers(0, pt.BLANK, size=max(1, int(4 * r["duration"])))) for r in t]
        refs = {r["id"]: r["text"] for r in t}
        shard_ids = [r["id"] for r in t]
        write_pk(labels / "parakeet_out" / info.source, stem, utts, refs, shard_ids)
        shards[(info.source, stem)] = dict(utts=utts, refs=refs, shard_ids=shard_ids, rows=len(t))
    return SimpleNamespace(base=base, data=fc.data, labels=labels, kotoba=kotoba, shards=shards, fc=fc)


def eval_prereg(labels: Path) -> dict:
    from kitsune.evaluate import teacher_baselines

    return {"eval_jsut": teacher_baselines(labels / "teacher_out", ["eval_jsut"], check=False)["eval_jsut"]["cer_corpus"]}


def make_record(root: SimpleNamespace) -> dict:
    """The extent record the box would write for this corpus: galgame's eval stem and first train stem come from its
    first tar, the second train stem from the second."""
    def st(src, stem):
        s = root.shards[(src, stem)]
        return {"stem": stem, "split": "eval" if stem.startswith("eval-") else "train", "step": src, "rows": s["rows"],
                "hours": 0.01, "ids_sha256": ids_sha256(s["shard_ids"]), "shard_bytes": 1}

    inputs = {"reazon_small": [["train-00000", "train-00001"]], "eval_jsut": [["eval-00000"]],
              "galgame": [["eval-00000", "train-00000"], ["train-00001"]]}
    return {"schema": 1, "name": "t", "root": "labels/t", "canonical_version": 1, "names": [*TRAIN, "eval_jsut"],
            "inputs": {}, "sources": {src: {"inputs": [{"input": f"{src}-{i}", "ordinal": i, "bytes": 1,
                                                        "stems": [st(src, s) for s in stems]}
                                                       for i, stems in enumerate(ins)]}
                                      for src, ins in inputs.items()}}


def seal(root: SimpleNamespace):
    (root.labels / "extent.json").write_text(json.dumps(make_record(root)), encoding="utf-8")
    (root.labels / "COMPLETE.json").write_text(json.dumps({"run_id": "t"}), encoding="utf-8")


def make_ctx(root: SimpleNamespace, **kw) -> "lc.Ctx":
    from transformers import ParakeetFeatureExtractor

    # random synthetic targets compress worse than real ones: K9's figure is the test's own (test_k9 sets it)
    base = dict(data=root.data, decode=fake_decode, fe=ParakeetFeatureExtractor(), kotoba=root.kotoba,
                train_sources=TRAIN, eval_sets=EVALS, mix={"galgame": 2}, margin={"galgame": 2}, card=None,
                eval_rows={"eval_jsut": 10, "galgame": 8}, k4_rows=1000, k4_units=100,
                expect=dict(lc.EXPECT, k9_mb_per_audio_h=100.0))
    base.update(kw)
    if "prereg" not in kw:
        base["prereg"] = eval_prereg(root.labels)
    return lc.Ctx(root.labels, lc.local_listing(root.labels), **base)


@pytest.fixture(scope="module")
def sound(tmp_path_factory):
    root = build_root(tmp_path_factory.mktemp("sound"))
    seal(root)
    return root


@pytest.fixture
def root(sound, tmp_path):
    """A private copy of the sound, sealed root to plant a defect in (data and labels; the audio is shared)."""
    labels = tmp_path / "labels"
    shutil.copytree(sound.labels, labels)
    kotoba = tmp_path / "kotoba.jsonl"
    shutil.copy(sound.kotoba, kotoba)
    return SimpleNamespace(**{**vars(sound), "labels": labels, "kotoba": kotoba, "base": tmp_path})


def status(ctx, name, **kw):
    res = lc.CHECKS[name](ctx, **kw)
    return res["status"], res


def rewrite_npz(path: Path, **changes):
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def rewrite_jsonl(path: Path, fn):
    rows = lc.jsonl_rows(path)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fn(rows)), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the formulas


def test_subsampled_length_is_the_transformers_formula():
    """subsampled_length equals transformers' ParakeetPreTrainedModel._get_subsampling_output_length for every feature
    length up to 35 s, with the converted model's subsampling (kernel 3, stride 2, factor 8)."""
    import torch
    from transformers import ParakeetEncoderConfig
    from transformers.models.parakeet.modeling_parakeet import ParakeetPreTrainedModel

    cfg = ParakeetEncoderConfig()
    sub = dict(kernel=cfg.subsampling_conv_kernel_size, stride=cfg.subsampling_conv_stride, factor=cfg.subsampling_factor)
    assert sub == lc.SUBSAMPLING
    lengths = torch.arange(0, 3501)
    want = ParakeetPreTrainedModel._get_subsampling_output_length(SimpleNamespace(config=cfg), lengths).tolist()
    assert [lc.subsampled_length(int(n), **sub) for n in lengths] == want


def test_feature_frames_is_the_extractor_mask():
    """feature_frames(n) is the ParakeetFeatureExtractor attention-mask length, including around hop boundaries."""
    from transformers import ParakeetFeatureExtractor

    fe = ParakeetFeatureExtractor()
    rng = np.random.default_rng(0)
    for n in (400, 401, 559, 560, 561, 16000, 16159, 16160, 48_001):
        out = fe([rng.standard_normal(n).astype(np.float32)], sampling_rate=16000, return_tensors="pt")
        assert int(out["attention_mask"].sum()) == lc.feature_frames(n), n
    assert lc.expected_n_frames(16000) == 13 and lc.expected_n_frames(30 * 16000) == 375


def test_ctc_feasible_need_counts_a_blank_between_repeats():
    assert lc.ctc_feasible_need([]) == 0
    assert lc.ctc_feasible_need([5, 5, 5]) == 5
    assert lc.ctc_feasible_need([1, 2, 1]) == 3
    assert lc.ctc_feasible_need(np.array([7, 7], np.int16)) == 3


# ------------------------------------------------------------------------------------------------ the sound root


def test_every_check_passes_on_a_sound_sealed_root(sound):
    ctx = make_ctx(sound)
    res = lc.run_checks(ctx, catch=False)
    assert {k: v["status"] for k, v in res.items()} == {k: "pass" for k in lc.CHECKS}, \
        {k: v["reason"] for k, v in res.items() if v["status"] != "pass"}
    assert res["K3"]["rows"] == sum(s["rows"] for s in sound.shards.values())
    k4 = res["K4"]["per_group"]
    assert set(k4) == {"reazon_small", "galgame", "eval_jsut", "galgame:eval"}
    assert all(g["rows"] > 0 for g in k4.values())
    assert sum(g["rows"] for g in k4.values()) == res["K4"]["rows"] == sum(s["rows"] for s in sound.shards.values())
    assert all(g["logmel_vs_hf_max_abs"] < 1e-4 for g in k4.values())
    assert res["K10"]["neutral_view_rows"] == sum(1 for r in lc.jsonl_rows(sound.kotoba)
                                                  if r["cer2"] is not None and r["cer2"] <= 0.5)
    assert res["K12"]["labelled_prefix"]["galgame"] == {"labelled_inputs": 2, "inputs": 2}
    report = lc.build_report(ctx, res)
    assert report["summary"]["pass"] == list(lc.CHECKS) and report["sealed"]
    json.dumps(report, default=float)


def test_unsealed_root_leaves_coverage_and_extent_pending(root):
    (root.labels / "COMPLETE.json").unlink()
    ctx = make_ctx(root)
    assert status(ctx, "K12")[0] == "pending"
    assert status(ctx, "K5")[0] == "pending"  # every stem covered, but the root can still change
    for name in ("K1", "K2", "K3", "K4", "K6", "K7", "K8", "K9", "K10", "K11"):
        assert status(ctx, name)[0] == "pass", name


def test_missing_inputs_skip_or_wait_instead_of_failing(root):
    ctx = make_ctx(root, decode=None, fe=None, kotoba=root.base / "nope.jsonl",
                   unavailable={"decode": "no model dir", "fe": "no model dir"})
    st, res = status(ctx, "K3")
    assert st == "skipped" and res["reason"] == "no model dir"
    assert status(ctx, "K4")[0] == "skipped" and status(ctx, "K10")[0] == "skipped"
    assert status(make_ctx(root, data=None), "K4")[0] == "skipped"
    empty = lc.Ctx(root.base / "empty", {}, train_sources=TRAIN, eval_sets=EVALS)
    res = lc.run_checks(empty, catch=False)
    assert all(v["status"] in ("pending", "skipped") for v in res.values()), {k: v["status"] for k, v in res.items()}


# ---------------------------------------------------------------------------------------------- planted defects


def test_k1_fails_on_a_changed_pin_and_waits_for_a_missing_meta(root):
    meta_path = root.labels / "parakeet_out" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, val in (("k_ctc", 4), ("features", "ParakeetFeatureExtractor 80 mel, CUDA, dither"),
                     ("nemo_revision", "0" * 40), ("dtypes", {**meta["dtypes"], "ctc": "bfloat16"})):
        meta_path.write_text(json.dumps({**meta, key: val}), encoding="utf-8")
        st, res = status(make_ctx(root), "K1")
        assert st == "fail" and any(k.startswith(key) for k in res["diffs"]), key
    files = dict(meta["files_sha256"], **{"model.safetensors": "f" * 64})
    meta_path.write_text(json.dumps({**meta, "files_sha256": files}), encoding="utf-8")
    assert status(make_ctx(root), "K1")[0] == "fail"
    meta_path.unlink()
    assert status(make_ctx(root), "K1")[0] == "pending"


def test_k2_fails_on_a_broken_npz_a_wrong_dtype_and_an_unaligned_jsonl(root):
    npz = root.labels / "parakeet_out" / "reazon_small" / "train-00000.npz"
    original = npz.read_bytes()
    with np.load(npz) as z:
        lp, nf = z["ctc_topk_lp"].copy(), z["n_frames"]
    lp[0] = lp[0][::-1]
    rewrite_npz(npz, ctc_topk_lp=lp)
    st, res = status(make_ctx(root), "K2")
    assert st == "fail" and any("sorted" in p for p in res["problems"])
    npz.write_bytes(original)
    rewrite_npz(npz, n_frames=nf.astype(np.int64))
    st, res = status(make_ctx(root), "K2")
    assert st == "fail" and any("n_frames dtype" in p for p in res["problems"])
    npz.write_bytes(original)
    assert status(make_ctx(root), "K2")[0] == "pass"
    rewrite_jsonl(npz.with_suffix(".jsonl"), lambda rows: [rows[1], rows[0], *rows[2:]])
    st, res = status(make_ctx(root), "K2")
    assert st == "fail" and any("same order" in p for p in res["problems"])


def test_k2_checks_ids_against_the_data_shard_only_when_it_is_the_labelled_one(root):
    """The data root's shard with the same ids digest is checked row by row (ordered subsequence); a data shard with
    other ids (another ingest) is counted, not failed."""
    ctx = make_ctx(root)
    st, res = status(ctx, "K2")
    assert st == "pass" and res["data_ids_checked"] == len(root.shards) and res["n_data_other_ids"] == 0
    data = root.base / "data"
    shutil.copytree(root.data, data)
    shard = data / "shards" / "reazon_small" / "train-00001.parquet"
    t = pq.read_table(shard)
    pq.write_table(t.take(list(range(t.num_rows))[::-1]), shard)
    st, res = status(make_ctx(root, data=data), "K2")
    assert st == "pass" and res["n_data_other_ids"] == 1 and res["data_other_ids"] == ["reazon_small/train-00001"]
    # and K4 cannot use that shard's audio: its group falls back to the other shard
    st, res = status(make_ctx(root, data=data), "K4")
    assert st == "pass" and res["per_group"]["reazon_small"]["other_ids"] == ["train-00001"]


def test_k3_fails_when_a_stored_ctc_hyp_is_not_the_npz_path(root):
    jsonl = root.labels / "parakeet_out" / "galgame" / "train-00001.jsonl"
    rewrite_jsonl(jsonl, lambda rows: [dict(rows[0], ctc_hyp=rows[0]["ctc_hyp"] + "ん"), *rows[1:]])
    st, res = status(make_ctx(root), "K3")
    assert st == "fail" and res["mismatch"] == 1 and res["examples"][0]["id"] == lc.jsonl_rows(jsonl)[0]["id"]


def test_k4_fails_when_n_frames_is_not_the_extractor_length(root):
    """A shard whose every utterance has one frame too many is internally consistent (K2 passes) but not what the
    extractor gives its audio."""
    src, stem = "galgame", "train-00000"
    s = root.shards[(src, stem)]
    rng = np.random.default_rng(1)
    utts = [pk_utt(rng, u["id"], u["duration"], u["n_frames"] + 1, [1, 2, 3]) for u in s["utts"]]
    write_pk(root.labels / "parakeet_out" / src, stem, utts, s["refs"], s["shard_ids"])
    ctx = make_ctx(root)
    assert status(ctx, "K2")[0] == "pass"
    st, res = status(ctx, "K4")
    g = res["per_group"]["galgame"]
    assert st == "fail" and g["status"] == "fail" and g["mismatch"]["hf"] == g["mismatch"]["logmel"] == len(utts)
    assert res["per_group"]["reazon_small"]["status"] == "pass"


def test_k4_waits_for_unlabelled_sources_and_missing_audio(root):
    ctx = make_ctx(root, train_sources=(*TRAIN, "reazon_large"))
    st, res = status(ctx, "K4")
    assert st == "pending" and res["per_group"]["reazon_large"]["reason"] == "not labelled yet"
    st, res = status(make_ctx(root, data=root.base / "no_data"), "K4")
    assert st == "pending" and all("no audio" in g["reason"] for g in res["per_group"].values())


def test_k5_fails_on_rows_in_one_root_and_waits_for_stems_in_progress(root):
    src, stem = "reazon_small", "train-00000"
    s = root.shards[(src, stem)]
    write_pk(root.labels / "parakeet_out" / src, stem, s["utts"][1:], s["refs"], s["shard_ids"])  # one row lost
    st, res = status(make_ctx(root), "K5")
    assert st == "fail" and res["pulled_rows_one_root"] == 1
    write_pk(root.labels / "parakeet_out" / src, stem, s["utts"], s["refs"], s["shard_ids"])
    for suffix in ("npz", "jsonl"):
        (root.labels / "parakeet_out" / "galgame" / f"train-00001.{suffix}").unlink()
    st, res = status(make_ctx(root), "K5")  # sealed: a whole stem in one root is 12 of 66 rows
    assert st == "fail" and res["teacher_only_stems"] == 1 and res["rows_of_one_root_stems"] == 12
    (root.labels / "COMPLETE.json").unlink()
    st, res = status(make_ctx(root), "K5")
    assert st == "pending" and res["teacher_only_examples"] == ["galgame/train-00001"]


def test_k5_fails_when_a_stem_holds_other_ids_than_the_extent_record(root):
    record = json.loads((root.labels / "extent.json").read_text(encoding="utf-8"))
    record["sources"]["reazon_small"]["inputs"][0]["stems"][1]["ids_sha256"] = "0" * 64
    (root.labels / "extent.json").write_text(json.dumps(record), encoding="utf-8")
    st, res = status(make_ctx(root), "K5")
    assert st == "fail" and res["record_sha_diff"] == ["reazon_small/train-00001"]


def test_k6_fails_on_a_partial_eval_set_and_off_card_baselines(root):
    ctx = make_ctx(root)
    base = status(ctx, "K6")[1]["per_set"]["eval_jsut"]["baselines"]
    card = {"eval_jsut": {"tdt": base["tdt_cer_corpus"], "ctc": base["ctc_cer_corpus"]}}
    assert status(make_ctx(root, card=card), "K6")[0] == "pass"
    off = {"eval_jsut": {"tdt": base["tdt_cer_corpus"] + 0.01, "ctc": base["ctc_cer_corpus"]}}
    st, res = status(make_ctx(root, card=off), "K6")
    assert st == "fail" and "card" in res["per_set"]["eval_jsut"]["reason"]
    s = root.shards[("eval_jsut", "eval-00000")]
    write_pk(root.labels / "parakeet_out" / "eval_jsut", "eval-00000", s["utts"][:-1], s["refs"], s["shard_ids"])
    st, res = status(make_ctx(root), "K6")
    assert st == "fail" and "ids differ" in res["per_set"]["eval_jsut"]["reason"]


def test_k6_compares_the_boxs_baselines_report(root):
    ctx = make_ctx(root)
    per = status(ctx, "K6")[1]["per_set"]
    report = {"parakeet": {"eval_jsut": {k: per["eval_jsut"]["baselines"][k] for k in ("tdt_cer_corpus", "ctc_cer_corpus")}}}
    (root.labels / "reports").mkdir()
    (root.labels / "reports" / "parakeet_baselines.json").write_text(json.dumps(report), encoding="utf-8")
    st, res = status(make_ctx(root), "K6")
    assert st == "pass" and "equals" in res["baselines_report"]
    report["parakeet"]["eval_jsut"]["tdt_cer_corpus"] += 0.001
    (root.labels / "reports" / "parakeet_baselines.json").write_text(json.dumps(report), encoding="utf-8")
    assert status(make_ctx(root), "K6")[0] == "fail"


def test_k6_waits_for_an_eval_set_one_pass_has_not_labelled(root):
    (root.labels / "COMPLETE.json").unlink()
    for suffix in ("npz", "jsonl"):
        (root.labels / "parakeet_out" / "galgame" / f"eval-00000.{suffix}").unlink()
    st, res = status(make_ctx(root), "K6")
    assert st == "pending" and res["per_set"]["galgame:eval"]["reason"] == "not labelled yet by Parakeet"
    assert status(make_ctx(root), "K10")[0] == "pending"


def test_k7_judges_the_target_the_study_uses(root):
    """TDT tokens that cannot align (more tokens than frames) fail K7 only when the study's target is the TDT
    hypothesis; the greedy CTC target of the same rows is feasible by construction."""
    src, stem = "reazon_small", "train-00001"
    s = root.shards[(src, stem)]
    rng = np.random.default_rng(2)
    utts = [pk_utt(rng, u["id"], u["duration"], u["n_frames"], [5], tdt_seq=[9] * (u["n_frames"] + 2))
            for u in s["utts"]]
    write_pk(root.labels / "parakeet_out" / src, stem, utts, s["refs"], s["shard_ids"])
    assert status(make_ctx(root), "K2")[0] == "pass"
    st, res = status(make_ctx(root), "K7")
    assert st == "pass" and res["train_rate"]["greedy_ctc"] == 0 and res["train_rate"]["tdt"] > 0.1
    st, res = status(make_ctx(root, ctc_target="tdt"), "K7")
    assert st == "fail" and res["per_group"]["reazon_small"]["tdt"] == len(utts)


def test_k8_fails_when_ctc_hyp_and_hyp_disagree_beyond_the_design(root):
    ctx = make_ctx(root)
    st, res = status(ctx, "K8")
    assert st == "pass" and res["train"]["cer_ctc_vs_tdt"] == 0 and res["train"]["raw_diff_rate"] == 0
    for stem in ("train-00000", "train-00001"):
        rewrite_jsonl(root.labels / "parakeet_out" / "galgame" / f"{stem}.jsonl",
                      lambda rows: [dict(r, hyp=r["hyp"][::-1] + "ぬ") for r in rows])
    st, res = status(make_ctx(root), "K8")
    rows = [r for p in ("reazon_small/train-00000", "reazon_small/train-00001", "galgame/train-00000",
                        "galgame/train-00001") for r in lc.jsonl_rows(root.labels / "parakeet_out" / f"{p}.jsonl")]
    want = corpus_cer([r["ctc_hyp"] for r in rows], [r["hyp"] for r in rows])["cer"]
    assert st == "fail" and res["train"]["cer_ctc_vs_tdt"] == pytest.approx(want) and want > 0.02
    assert res["per_group"]["eval_jsut"]["cer_ctc_vs_tdt"] == 0  # the eval rows are reported, not pooled


def test_k9_fails_when_parakeet_out_grows_beyond_the_design(root):
    ctx = make_ctx(root)
    mb = status(ctx, "K9")[1]["train_mb_per_audio_h"]
    expect = dict(lc.EXPECT, k9_mb_per_audio_h=mb)
    assert status(make_ctx(root, expect=expect), "K9")[0] == "pass"
    rewrite_jsonl(root.labels / "parakeet_out" / "galgame" / "train-00000.jsonl",
                  lambda rows: [dict(r, pad="x" * 20000) for r in rows])
    st, res = status(make_ctx(root, expect=expect), "K9")
    assert st == "fail" and res["train_mb_per_audio_h"] > 2 * mb


def test_k10_fails_when_the_kotoba_ids_differ(root):
    rewrite_jsonl(root.kotoba, lambda rows: rows[::-1])
    st, res = status(make_ctx(root), "K10")
    assert st == "fail" and not res["per_root"]["parakeet_out"]["equal"]
    assert res["per_root"]["parakeet_out"]["missing_vs_kotoba"] == 0  # same rows, other order


def test_k11_fails_when_the_cohere_eval_labels_drift(root):
    prereg = eval_prereg(root.labels)  # registered before the drift
    jsonl = root.labels / "teacher_out" / "eval_jsut" / "eval-00000.jsonl"
    rewrite_jsonl(jsonl, lambda rows: [dict(rows[0], hyp=""), *rows[1:]])
    st, res = status(make_ctx(root, prereg=prereg), "K11")
    assert st == "fail" and abs(res["per_set"]["eval_jsut"]["diff_pp"]) > 0.05


def test_k11_waits_for_a_gate_set_not_labelled(root):
    shutil.rmtree(root.labels / "teacher_out" / "eval_jsut")
    assert status(make_ctx(root, prereg={"eval_jsut": 0.1}), "K11")[0] == "pending"


def test_k12_fails_when_the_sealed_extent_misses_the_mix(root):
    st, res = status(make_ctx(root, mix={"galgame": 3}, margin={}), "K12")  # the record has only 2 galgame tars
    assert st == "fail" and res["too_few_inputs"] == {"galgame": {"labelled_inputs": 2, "required": 3}}
    assert res["missing"] == {}  # subset_stems alone would have passed it: every recorded stem is labelled
    for suffix in ("npz", "jsonl"):
        (root.labels / "parakeet_out" / "galgame" / f"train-00001.{suffix}").unlink()
    st, res = status(make_ctx(root), "K12")
    assert st == "fail" and res["missing"] == {"galgame": ["train-00001"]}
    assert res["labelled_prefix"]["galgame"]["labelled_inputs"] == 1
    st, res = status(make_ctx(root, mix={"galgame": 1}, margin={"galgame": 2}), "K12")
    assert st == "pass" and res["below_margin"] == {"galgame": {"labelled_inputs": 1, "margin": 2}}
    (root.labels / "extent.json").unlink()
    assert status(make_ctx(root), "K12")[0] == "fail"


# ------------------------------------------------------------------------------------------------ pull and CLI


def test_select_pull_takes_meta_eval_sets_and_spread_train_stems():
    listing = {"LEASE.json": 1, "extent.json": 2, "reports/parakeet_baselines.json": 3, "selections/full.parquet": 9,
               "teacher_out/meta.json": 1, "parakeet_out/meta.json": 1, "second_out/meta.json": 1}
    for root, exts in (("teacher_out", ("npz", "jsonl")), ("parakeet_out", ("npz", "jsonl")), ("second_out", ("jsonl",))):
        for e in exts:
            listing[f"{root}/eval_jsut/eval-00000.{e}"] = 1
            listing[f"{root}/galgame/eval-00000.{e}"] = 1
            for i in range(7):
                listing[f"{root}/galgame/train-{i:05d}.{e}"] = 1
    for e in ("npz", "jsonl"):
        listing[f"teacher_out/galgame/train-00007.{e}"] = 1  # Parakeet has not reached it
    got = lc.select_pull(listing, train_sources=("galgame", "reazon_large"), eval_sets=("eval_jsut", "galgame"),
                         train_stems=3)
    train = sorted({p.split("/")[2].split(".")[0] for p in got if "/train-" in p})
    assert train == ["train-00000", "train-00003", "train-00006"]
    assert "selections/full.parquet" not in got and {"LEASE.json", "extent.json", "reports/parakeet_baselines.json",
                                                      "parakeet_out/meta.json"} <= set(got)
    assert {f"{r}/galgame/eval-00000.jsonl" for r in lc.ROOTS} <= set(got)
    assert "second_out/galgame/train-00003.jsonl" in got and "teacher_out/galgame/train-00007.npz" not in got


def test_pull_copies_one_commit_and_records_the_listing(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.hf_api import RepoFile

    files = {"labels/full/parakeet_out/meta.json": b"{}", "labels/full/teacher_out/meta.json": b"{}",
             "labels/full/parakeet_out/eval_jsut/eval-00000.npz": b"npz",
             "labels/full/parakeet_out/eval_jsut/eval-00000.jsonl": b"{}\n", "other/x.bin": b"no"}
    seen = {}

    class FakeApi:
        def dataset_info(self, repo):
            return SimpleNamespace(sha="c0ffee" * 6 + "abcd")

        def list_repo_tree(self, repo, repo_type, path_in_repo, recursive, revision):
            seen["tree_rev"] = revision
            return [RepoFile(path=p, size=len(b), oid="0", lfs=None, last_commit=None, security=None)
                    for p, b in files.items() if p.startswith(path_in_repo + "/")]

    def fake_snapshot(repo, repo_type, revision, allow_patterns, local_dir, max_workers):
        seen["snap_rev"], seen["patterns"] = revision, sorted(allow_patterns)
        for p in allow_patterns:
            dst = Path(local_dir) / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(files[p])

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    out = tmp_path / "labels" / "full"
    rec = lc.pull(out, repo="u/data")
    assert seen["tree_rev"] == seen["snap_rev"] == rec["revision"] == "c0ffee" * 6 + "abcd"
    assert (out / "parakeet_out" / "eval_jsut" / "eval-00000.npz").read_bytes() == b"npz"
    saved = json.loads((out / lc.PULL_FILE).read_text(encoding="utf-8"))
    assert set(saved["listing"]) == {p[len("labels/full/"):] for p in files if p.startswith("labels/full/")}
    assert sorted(saved["pulled"]) == sorted(saved["listing"])
    assert not any(p.name.startswith(".") for p in (tmp_path / "labels").iterdir())  # the staging dir is gone
    assert lc.local_listing(out) == {p: n for p, n in saved["listing"].items()}


def test_cli_run_writes_the_report_and_exits_by_failures(sound, tmp_path):
    """The CLI runs the study's defaults: on this synthetic root the format checks pass and K6 fails (its eval sets
    are not the manifest's); a model dir that is not there skips K3/K4 with the reason."""
    out = tmp_path / "r.json"
    args = ["run", "--labels", str(sound.labels), "--data", str(sound.data), "--model-dir", str(tmp_path / "none"),
            "--kotoba", str(sound.kotoba), "--out", str(out)]
    assert lc.main(args + ["--checks", "K1", "K2", "K3", "K10"]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert {k: v["status"] for k, v in rep["checks"].items()} == {"K1": "pass", "K2": "pass", "K3": "skipped",
                                                                  "K10": "pass"}
    assert "--model-dir" in rep["checks"]["K3"]["reason"] and rep["hub"] is None and rep["sealed"]
    assert lc.main(args + ["--checks", "K6"]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["fail"] == ["K6"]
