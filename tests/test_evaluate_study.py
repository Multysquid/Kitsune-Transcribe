"""The size study's evaluator (WP5b): the per-utterance tables, the no-style count, the CTC family's eval and verdict,
the Parakeet baselines, and scripts/05_evaluate.py's family switch, manifest checks and table modes.

  tables      kitsune.evaluate.utterance_table / utterance_scores sum to corpus_cer exactly, match kitsune.study_stats'
              own scorer, and load in tools/study_report.py; the no-style count forgives exactly the style regions
  CTC eval    a tiny Parakeet CTC student (tests/fixtures_ctc.py) whose stored "teacher" targets are its own: ctc_eval
              gives frame argmax agreement 1, a KL of the fp16 storage only, the teacher's text on every row; a row
              whose stored n_frames differs is left out of the frame metrics and counted; a frame store's batches
              (their own targets, as the CTC trainer's FrameBatchDataset collates them) give the same numbers, and
              where the CTC trainer's frame stores exist, ctc_eval on one is its kitsune.ctc_eval pass to the bit
  verdict     family "ctc" judges against Parakeet's CTC path, registered or pending; the Cohere verdict is unchanged
  05 (ctc)    the same student through scripts/05_evaluate.py on a study manifest, with the probe: tables, study.json,
              verdict v2 with family ctc; --teachers and --from-evals; the refusals: a manifest that does not hash to
              itself, one that does not hold the store's rows, one that differs from PREREG.json, an aed config on a
              CTC checkpoint - each before any model is loaded; decision 15: an eval row whose frames do not align
              with its stored targets refuses (again on the same command, before the rest is evaluated), and so do
              more than 0.1 % of the probe's rows
  real data   (skipped without it) the stored ctc_hyp of the label box's eval shards scores to its stored ctc_cer on
              every row and to the measured corpus CERs (Parakeet TDT / CTC JSUT 6.62 / 6.71, CV8 7.50 / 7.58, Reazon
              10.20 / 9.71 %; Cohere 8.30 / 4.07 / 6.28 %); the first A100 run's per-set CER from its eval parquets;
              ctc_eval of the real Parakeet CTC teacher on real eval rows against its stored frame targets and text
CPU only, synthetic data in the repo's formats; nothing needs the network.
"""
import json
import os
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import REAL, load_script, make_fake_corpus, make_fake_selection, need_real  # noqa: E402

from kitsune import evaluate as ev  # noqa: E402
from kitsune import study_stats as ss  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402
from kitsune.text import cer as utt_cer  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import study_report  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def pending_prereg(tmp_path_factory):
    """The toy corpora here are not the study's selection: 05_evaluate's default --prereg (the committed
    study/PREREG.json, filled from the uploaded selection) would refuse their manifests and teachers. Every 05 loaded
    by this module defaults to the rules with their sidecar fields pending instead, and the Parakeet CTC registration
    kitsune.evaluate read from the committed file at import is taken as pending; a test that needs a filled PREREG
    passes --prereg or sets PARAKEET_CTC_CER_PREREG itself."""
    from kitsune import prereg

    path = tmp_path_factory.mktemp("prereg") / "PREREG.json"
    path.write_bytes(prereg.rules_json(prereg.rules()))
    real = load_script

    def load_pending(name: str):
        mod = real(name)
        if hasattr(mod, "PREREG_JSON"):
            mod.PREREG_JSON = path
        return mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(globals(), "load_script", load_pending)
        mp.setattr(ev, "PARAKEET_CTC_CER_PREREG", None)
        yield path


_KANA =list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")
_KATA = list("アイウエオカキクケコサシスセソタチツテトナニヌネノ")
_KANJI = list("日本語今明晴雨漢字書読三百千万円")


def random_texts(rng, n: int) -> tuple[list[str], list[str]]:
    """Pairs of Japanese-like strings: a reference and a hypothesis that keeps, drops, swaps or restyles pieces of it,
    plus empty references and empty hypotheses."""
    pool = _KANA + _KATA + _KANJI + list("0123456789abc、。「」 ")
    refs, hyps = [], []
    for i in range(n):
        ref = "".join(rng.choice(pool, size=int(rng.integers(0, 25))))
        hyp = []
        for c in ref:
            r = rng.random()
            if r < 0.7:
                hyp.append(c)
            elif r < 0.8:
                hyp.append(rng.choice(pool))
            elif r < 0.9 and c in _KATA:
                hyp.append(chr(ord(c) - 0x60))  # the same kana in hiragana: a style region
            elif r < 0.95:
                hyp.append(c + rng.choice(pool))
        hyps.append("".join(hyp) if i % 11 else "")
        refs.append(ref if i % 13 else "")
    return refs, hyps


# ------------------------------------------------------------------------------------------------ tables


def test_utterance_scores_sum_to_corpus_cer_and_match_the_reports_scorer():
    """One jiwer alignment of every row with a reference: per-utterance edits and reference chars sum to corpus_cer's
    exactly (the gate's measure), the counts equal kitsune.study_stats.score_utterances', the no-style count is never
    above the raw one, and an empty reference scores its hypothesis as insertions with ref_len 0."""
    rng = np.random.default_rng(3)
    refs, hyps = random_texts(rng, 400)
    sc = ev.utterance_scores(refs, hyps)
    c = ev.corpus_cer(hyps, refs)
    has_ref = sc["ref_len"] > 0
    assert int(sc["edits"][has_ref].sum()) == c["edits"] and int(sc["ref_len"][has_ref].sum()) == c["ref_chars"]
    assert int((~has_ref).sum()) == c["n_empty_ref"] > 0
    theirs = ss.score_utterances(refs, hyps)
    for col in ("edits", "ref_len", "sub", "del", "ins", "hyp_len"):
        np.testing.assert_array_equal(sc[col].to_numpy(), theirs[col].to_numpy(), err_msg=col)
    assert (sc["edits_nostyle"] <= sc["edits"]).all() and (sc["edits_nostyle"] >= 0).all()
    assert (sc["edits_nostyle"] < sc["edits"]).any()  # the planted hiragana <-> katakana regions
    assert (sc["edits_style"][~has_ref] == 0).all()
    # the per-utterance CER is kitsune.text.cer's (what the label passes stored as cer / ctc_cer)
    for r, h, e, n in zip(refs, hyps, sc["edits"], sc["ref_len"]):
        if n:
            assert e / n == pytest.approx(utt_cer(h, r), abs=1e-12)


@pytest.mark.parametrize("ref, hyp, style", [
    ("三百円", "300円", 3), ("カタカナ", "かたかな", 4), ("漢字を書く", "かんじを書く", 3), ("かんじ", "漢字", 3),
    ("今日は晴れ", "明日は晴れ", 0), ("はれ", "あめ", 0), ("あい", "", 0), ("abc", "abd", 0), ("", "あ", 0)])
def test_the_no_style_count_forgives_the_style_regions_only(ref, hyp, style):
    """edits_nostyle = edits minus the edits of the regions whose class (region_kind, the first run's error analysis)
    is a style class: numerals, hiragana vs katakana, kanji vs kana. A different word stays an error."""
    t = ev.utterance_table(["u"], "eval_jsut", [ref], [hyp])
    assert t["edits_style"].iat[0] == style and t["edits_nostyle"].iat[0] == t["edits"].iat[0] - style


@pytest.mark.parametrize("ref, hyp, kind", [
    ("三百", "300", "numeral"), ("百", "千", "numeral"), ("カタカナ", "かたかな", "hira<->kata"), ("漢字", "かんじ", "kanji->kana"),
    ("かんじ", "漢字", "kana->kanji"), ("今", "明", "kanji->kanji"), ("今日", "明", "kanji->kanji(len)"),
    ("はれ", "あめ", "hira->hira"), ("アメ", "ハレ", "kata->kata"), ("", "あ", "insertion"), ("あ", "", "deletion"),
    ("abc", "abd", "latin"), ("ぱあてぃ", "パーティ", "kana-other")])
def test_region_kind(ref, hyp, kind):
    assert ev.region_kind(ref, hyp) == kind
    assert (kind in ev.STYLE_KINDS) == (kind in ("numeral", "hira<->kata", "kanji->kana", "kana->kanji",
                                                 "kana-other"))


def test_tables_load_in_the_study_report(tmp_path):
    """A system's tables in the layout 05 writes (tables/<system>/<set>.parquet) load in tools/study_report.py and
    align to the manifest; the report's corpus CER per stratum is utterance_table's, and a table that misses one
    manifest id is refused by the report."""
    rng = np.random.default_rng(5)
    man = {"sets": {}, "galgame_views": {}}
    tables = {}
    for s, n in (("eval_jsut", 30), ("eval_cv8", 20), ("eval_reazon", 20), ("galgame", 24)):
        ids = [f"{s}/{i:03d}" for i in range(n)]
        refs, hyps = random_texts(rng, n)
        tables[s] = ev.utterance_table(ids, s, refs, hyps)
        man["sets"][s] = {"ids": ids, "ids_sha256": ids_sha256(ids)}
    gids = man["sets"]["galgame"]["ids"]
    man["galgame_views"] = {"neutral": {"ids": gids[::2], "ids_sha256": ids_sha256(gids[::2])}, "all": gids,
                            "label_box": gids[1::3]}
    d = tmp_path / "tables" / "sys-a"
    d.mkdir(parents=True)
    for s, t in tables.items():
        t.to_parquet(d / f"{s}.parquet", index=False)
    manifest = ss.parse_manifest(man)
    loaded, ignored = study_report.load_tables(tmp_path / "tables", manifest.sets)
    corpus = ss.build_corpus(loaded, manifest)
    point = ss.point_sums(corpus)
    for s in ("eval_jsut", "eval_cv8", "eval_reazon"):
        assert float(ss.stratum_cer(point, s)[0]) == pytest.approx(ev.table_cer(tables[s])["cer"], abs=0)
        assert float(ss.stratum_cer(point, s, True)[0]) == pytest.approx(ev.table_cer(tables[s], nostyle=True)["cer"])
    neutral = ev.table_cer(tables["galgame"], gids[::2])["cer"]
    assert float(ss.stratum_cer(point, "galgame_neutral")[0]) == pytest.approx(neutral)
    tables["eval_cv8"].iloc[1:].to_parquet(d / "eval_cv8.parquet", index=False)
    with pytest.raises(ss.ManifestError, match="missing"):
        ss.build_corpus(study_report.load_tables(tmp_path / "tables", manifest.sets)[0], manifest)


def test_the_greedy_summary_equals_its_table():
    """The synthetic twin of the first run's reproduction: a greedy_eval per-utterance frame's corpus CER per set
    (summarise_greedy: cer_ref_corpus, ref_edits, ref_chars, the teacher's on the same rows) is its table's."""
    rng = np.random.default_rng(9)
    rows = []
    for s, n in (("eval_jsut", 40), ("eval_cv8", 30), ("galgame", 20)):
        refs, hyps = random_texts(rng, n)
        _, teach = random_texts(rng, n)
        for i in range(n):
            rows.append(dict(id=f"{s}/{i}", source=s, duration=1.0, ref=refs[i], teacher_hyp=teach[i], hyp=hyps[i],
                             cer_ref=utt_cer(hyps[i], refs[i]), cer_teacher=0.0, truncated=bool(i % 7 == 0), n_tok=3,
                             teacher_cer=utt_cer(teach[i], refs[i]), teacher_truncated=False, hyp_ids=[1],
                             has_teacher=True))
    g = pd.DataFrame(rows)
    summ = ev.summarise_greedy(g)
    for s, d in summ["sets"].items():
        part = g[g["source"] == s]
        t = ev.utterance_table(part["id"], s, part["ref"], part["hyp"], part["truncated"])
        c = ev.table_cer(t)
        assert (c["edits"], c["ref_chars"], c["n_empty_ref"]) == (d["ref_edits"], d["ref_chars"], d["n_empty_ref"])
        assert c["cer"] == d["cer_ref_corpus"] and int(t["truncated"].sum()) == d["n_truncated"]
        tt = ev.table_cer(ev.utterance_table(part["id"], s, part["ref"], part["teacher_hyp"]))
        assert tt["cer"] == d["teacher_cer_ref_corpus"]


# ------------------------------------------------------------------------------------------------ verdict


def _final(student: dict, teacher: dict | None) -> dict:
    sets = {}
    for s, c in student.items():
        sets[s] = dict(cer_ref_corpus=c, teacher_cer_ref_corpus=(teacher or {}).get(s, float("nan")), n=100,
                       n_truncated=0)
    return dict(sets=sets)


def test_verdict_family_ctc_judges_against_parakeets_ctc_path(monkeypatch):
    """family "ctc": the ratios are to Parakeet's CTC path on the same ids; the registered numbers are
    PARAKEET_CTC_CER_PREREG (pending: no drift, and a set without a same-ids teacher is not judged). The Cohere verdict
    is unchanged by the option, key for key."""
    stud = {"eval_jsut": 0.080, "eval_cv8": 0.090, "eval_reazon": 0.150}
    pk = {"eval_jsut": 0.0671, "eval_cv8": 0.0758, "eval_reazon": 0.0971}
    hist = [dict(step=s, heldout_kl=1.0, probe_kl=0.9, greedy={k: dict(cer_ref_corpus=v, teacher_cer_ref_corpus=pk[k],
                                                                           trunc_rate=0.0, n=100)
                                                               for k, v in stud.items()}) for s in (10, 20, 30)]
    monkeypatch.setattr(ev, "PARAKEET_CTC_CER_PREREG", None)
    v = ev.verdict(dict(final=_final(stud, pk), history=hist), family="ctc", version=2)
    assert v["family"] == "ctc" and v["teacher"] == "parakeet-ctc" and v["teacher_prereg_status"] == "pending"
    for s in stud:
        d = v["sets"][s]
        assert d["teacher"] == pk[s] and d["teacher_source"] == "same ids" and d["ratio"] == stud[s] / pk[s]
        assert d["teacher_prereg"] is None and d["baseline_drift"] is None
    assert v["n_sets_go"] == 2  # JSUT 1.19x, CV8 1.19x, Reazon 1.54x
    # pending and no teacher on these ids: the set is not judged (never judged against Cohere's numbers)
    nope = ev.verdict(dict(final=_final(stud, None), history=hist), family="ctc")
    assert nope["sets"] == {} and all("pending" in r for r in nope["reasons"][:3])
    reg = {"eval_jsut": 0.0670, "eval_cv8": 0.0760, "eval_reazon": 0.0970}
    monkeypatch.setattr(ev, "PARAKEET_CTC_CER_PREREG", reg)
    r = ev.verdict(dict(final=_final(stud, None), history=hist), family="ctc")
    assert r["teacher_prereg_status"] == "registered" and r["sets"]["eval_jsut"]["teacher"] == 0.0670
    assert r["sets"]["eval_jsut"]["teacher_source"] == "pre-registered"
    r2 = ev.verdict(dict(final=_final(stud, pk), history=hist), family="ctc")
    assert r2["sets"]["eval_cv8"]["baseline_drift"] == pytest.approx(pk["eval_cv8"] - reg["eval_cv8"])
    # the aed default: exactly the verdict as it was (no family keys; Cohere's registered numbers)
    a = ev.verdict(dict(final=_final(stud, None), history=hist))
    assert a == ev.verdict(dict(final=_final(stud, None), history=hist), family="aed")
    assert "family" not in a and a["sets"]["eval_jsut"]["teacher"] == ev.TEACHER_CER_PREREG["eval_jsut"]
    with pytest.raises(ValueError, match="family"):
        ev.verdict(dict(final=_final(stud, pk), history=hist), family="rnnt")


def test_prereg_teacher_cer_reads_the_filled_baselines(tmp_path, pending_prereg):
    """PARAKEET_CTC_CER_PREREG comes from study/PREREG.json's baselines once filled: the committed file (filled from
    the uploaded selection) gives its gate numbers, a pending one None; a filled block gives the gate sets' numbers; a
    partly filled or unreadable one gives None."""
    committed = json.loads((ROOT / "study" / "PREREG.json").read_text(encoding="utf-8"))
    if committed["baselines"]["status"] == "pending":
        assert ev.prereg_teacher_cer("parakeet-ctc") is None
    else:
        assert ev.prereg_teacher_cer("parakeet-ctc") == {s: committed["baselines"]["parakeet-ctc"][s]
                                                         for s in ev.GATE_SETS}
    pending = json.loads(pending_prereg.read_text(encoding="utf-8"))
    assert pending["baselines"]["status"] == "pending" and ev.prereg_teacher_cer("parakeet-ctc", pending) is None
    assert ev.family_teacher_prereg("ctc") is None  # this module takes the registration as pending
    assert ev.family_teacher_prereg("aed") == ev.TEACHER_CER_PREREG
    pk = {"eval_jsut": 0.0671, "eval_cv8": 0.0758, "eval_reazon": 0.0971, "m4": 0.1}
    filled = dict(committed, baselines=dict(status="filled", **{"parakeet-ctc": pk}))
    assert ev.prereg_teacher_cer("parakeet-ctc", filled) == {"eval_jsut": 0.0671, "eval_cv8": 0.0758,
                                                             "eval_reazon": 0.0971}
    p = tmp_path / "PREREG.json"
    p.write_text(json.dumps(filled), encoding="utf-8")
    assert ev.prereg_teacher_cer("parakeet-ctc", p)["eval_cv8"] == 0.0758
    part = dict(committed, baselines=dict(status="filled", **{"parakeet-ctc": {"eval_jsut": 0.0671,
                                                                               "eval_cv8": "pending"}}))
    assert ev.prereg_teacher_cer("parakeet-ctc", part) is None
    assert ev.prereg_teacher_cer("parakeet-ctc", tmp_path / "absent.json") is None


# ------------------------------------------------------------------------------------------------ CTC eval


def _tone(rng, lo: float = 0.4, hi: float = 2.5) -> np.ndarray:
    n = int(rng.uniform(lo, hi) * 16000)
    t = np.arange(n) / 16000
    return (0.3 * np.sin(2 * np.pi * rng.uniform(150, 1500) * t) + 0.01 * rng.standard_normal(n)).astype(np.float32)


def varied_ctc_model(seed: int = 0):
    """A tiny ParakeetForCTC whose CTC head reads the frame-to-frame variation of its encoder output: a random head over
    the whitened hidden states, their mean subtracted in the bias, the blank raised. About half the frames come out
    blank-certain and the rest as one of about ten tokens, so its own greedy path is a teacher with both kinds of
    frame. (The tiny random encoder alone gives nearly the same frame everywhere on the corpus' tones.)"""
    from fixtures_ctc import tiny_ctc_model
    from transformers import ParakeetFeatureExtractor

    from kitsune import ctc_student as CS
    from kitsune.features import LogMel

    m = tiny_ctc_model(seed=seed, n_layers=2, ffn=48)
    feat = CS.CtcFeatures(LogMel.from_feature_extractor(ParakeetFeatureExtractor()), "cpu")
    rng = np.random.default_rng(seed)
    feats, lens = feat([_tone(rng) for _ in range(6)])
    with torch.no_grad():
        o = m.encoder(input_features=feats, attention_mask=CS.lengths_to_mask(lens, feats.shape[1]).long(),
                      output_attention_mask=True)
        h = o.last_hidden_state[o.attention_mask.bool()]
        mu, sd = h.mean(0), h.std(0)
        w = torch.randn(CS.CTC_VOCAB, h.shape[1], generator=torch.Generator().manual_seed(seed)) / sd * 2.0
        b = -(w @ mu)
        b[CS.CTC_BLANK] += 10.0
        m.ctc_head.weight.copy_(w.unsqueeze(-1))
        m.ctc_head.bias.copy_(b)
    return m


def save_varied_student(d: Path, seed: int = 0) -> Path:
    """The student dir as a trained CTC checkpoint looks (HF weights, the processor files, student_meta.json), in fp32:
    the head's bias holds the subtracted mean to fp32 precision, which bf16 (save_ctc_student) would lose."""
    from fixtures_ctc import write_processor

    from kitsune import ctc_student as CS
    from kitsune.student import write_meta

    m = varied_ctc_model(seed)
    m.save_pretrained(d)
    write_processor(d, decoder_type="ctc")
    counts = CS.param_counts(m)
    write_meta(d, dict(name="tiny", family="ctc", init_class="pruned_kept", params_total=counts["total"],
                       params_non_embedding=counts["non_embedding"], closed_form_params=counts["closed_form"],
                       seed=seed, bn="teacher", teacher=f"{CS.TEACHER_REPO}@{CS.TEACHER_REVISION}"))
    return d


def load_patched(student_dir: Path):
    """The student as 05_evaluate loads it (perf.relpos_patch, the default: rel-pos once per batch)."""
    from kitsune import ctc_student as CS
    from kitsune.patches import patch_relpos_once_per_batch

    model = CS.load_ctc_student(student_dir, "cpu")
    patch_relpos_once_per_batch(model)
    return model


def exact_targets(student_dir: Path, store, batch_s: float) -> tuple[dict, dict]:
    """The student's own stored-format targets (kitsune.ctc_targets.targets_from_log_probs, as the label pass stores
    them) and greedy CTC text for every row of `store`, from log-probs computed on exactly the batches ctc_eval cuts
    over that store (kitsune.trainset.eval_batches of batch_s, the dataset's collate, the eval featuriser) by the model
    05_evaluate loads: the very log-probs the eval computes, so student and stored teacher agree on every frame to the
    bit."""
    from transformers import AutoProcessor

    from kitsune import ctc_student as CS
    from kitsune import trainset
    from kitsune.ctc_targets import targets_from_log_probs

    model = load_patched(student_dir)
    feat = CS.ctc_features(student_dir, "cpu").logmel
    tok = AutoProcessor.from_pretrained(str(student_dir)).tokenizer
    ds = trainset.AudioBatchDataset(store)
    targets, text = {}, {}
    with torch.no_grad():
        for b in trainset.eval_batches(store.utts, batch_s):
            item = ds[b]
            feats, fmask = ev._features(feat, item, torch.device("cpu"))
            lp, n = CS.ctc_log_probs(model, feats, fmask)
            for uid, ft, path in zip(item["ids"], targets_from_log_probs(lp, n), CS.greedy_ctc_ids(lp, n)):
                targets[uid], text[uid] = ft, CS.decode_ids(tok, path)
    return targets, text


def write_self_parakeet_out(fc, targets: dict, text: dict, root: Path) -> dict:
    """parakeet_out (kitsune.parakeet_targets' format, written by its own writer) for the rows of `targets`, in their
    teacher shards: the frame targets and ctc_hyp are the student's own, hyp (the TDT text) = ctc_hyp, cer and ctc_cer
    as the label pass computes them. Returns id -> the jsonl row."""
    import io

    import soundfile as sf
    from fixtures_ctc import SETTINGS, _tdt_path

    from kitsune import ctc_student as CS
    from kitsune import parakeet_targets as pt
    from kitsune.text import cer as cer_fn

    rng = np.random.default_rng(0)
    by: dict = {}
    for uid in targets:
        u = fc.utts[uid]
        by.setdefault((u.source, u.stem), []).append(u)
    rows_out = {}
    for (source, stem), chunk in sorted(by.items()):
        packed, rows = [], []
        for u in sorted(chunk, key=lambda x: list(fc.utts).index(x.id)):
            ft = targets[u.id]
            assert ft.n_frames == CS.expected_n_frames(sf.info(io.BytesIO(u.audio)).frames)  # K4 on the student
            packed.append(dict(id=u.id, duration=u.duration, n_frames=ft.n_frames, truncated=False,
                               ctc_blank_lp=ft.blank_lp.astype(np.float32), ctc_dense_frame=ft.dense_frame,
                               ctc_topk_idx=ft.topk_idx.astype(np.int64), ctc_topk_lp=ft.topk_lp.astype(np.float32),
                               **_tdt_path(ft, rng, 8)))
            h = text[u.id]
            rows.append(dict(id=u.id, hyp=h, ctc_hyp=h, ref=u.text, cer=round(cer_fn(h, u.text), 4),
                             ctc_cer=round(cer_fn(h, u.text), 4), duration=round(float(u.duration), 3),
                             n_tok=len(ft.ctc_ids), n_steps=ft.n_frames, n_frames=ft.n_frames, n_forced=0,
                             truncated=False))
            rows_out[u.id] = rows[-1]
        n_scanned = sum(1 for x in fc.utts.values() if x.source == source and x.stem == stem)
        arrays = pt.pack_shard(packed, settings=SETTINGS, n_scanned=n_scanned,
                               shard_ids_sha=ids_sha256([x.id for x in fc.utts.values()
                                                         if x.source == source and x.stem == stem]))
        pt.write_shard(root / source, stem, arrays, rows)
    (root / "meta.json").write_text(json.dumps(pt.build_meta(SETTINGS)), encoding="utf-8")
    return rows_out


EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "galgame"]
BATCH_S = 5.0


@pytest.fixture(scope="module")
def ctc_env(tmp_path_factory):
    """A corpus (a train source for the probe, the gate sets and a galgame hold-out), its selection, a tiny CTC student
    with varied frames, parakeet_out made from the student itself on the eval store's and the probe store's own batches
    (so it is its own teacher to the bit), the study manifest of the eval rows (with Galgame views) and a CTC study
    config with relative paths."""
    from kitsune import trainset

    root = tmp_path_factory.mktemp("ctc_eval")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (16, "train"), "eval_jsut": (6, "eval"),
                                                    "eval_cv8": (5, "eval"), "eval_reazon": (5, "eval"),
                                                    "galgame": (8, "eval")},
                          dur_range=(0.4, 2.5), token_range=(256, 296), no_second=tuple(EVAL_SETS), seed=11)
    sel = make_fake_selection(fc, greedy_n=3, probe_n=4)
    student = save_varied_student(root / "student")
    s = pd.read_parquet(sel)
    eval_store = trainset.eval_store(sel, fc.data, fc.teacher_out, root / "cache_t" / "eval", EVAL_SETS)
    train = s[(s["split"] == "train") & s["keep"]]
    # the probe store as 05's probe_store builds it (the same rows in the same order: the same batches)
    probe_store = trainset.build_stores(sel, fc.data, fc.teacher_out, root / "cache_t" / "probe", ["src_a"], ["train"],
                                        ids=train["id"][train["in_probe"]].tolist())
    rest = trainset.build_stores(sel, fc.data, fc.teacher_out, root / "cache_t" / "train", ["src_a"], ["train"])
    targets, text = {}, {}
    for st in (rest, eval_store, probe_store):  # later stores' exact batches win for their rows
        t, x = exact_targets(student, st, BATCH_S)
        targets.update(t)
        text.update(x)
    pk_rows = write_self_parakeet_out(fc, targets, text, root / "corpus" / "parakeet_out")
    manifest = {"schema": 1, "sets": {}}
    for e in EVAL_SETS:
        ids = s["id"][(s["source"] == e) & (s["split"] == "eval") & s["keep"]].tolist()
        manifest["sets"][e] = {"n": len(ids), "ids_sha256": ids_sha256(ids), "ids": ids}
    gal = manifest["sets"]["galgame"]["ids"]
    manifest["galgame_views"] = {v: {"ids": x, "ids_sha256": ids_sha256(x), "n": len(x)}
                                 for v, x in (("neutral", gal[::2]), ("all", gal), ("label_box", gal[1:]))}
    (sel.parent / "study_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {"run_name": "study-p01", "family": "ctc", "student": "student", "data_root": "corpus/data",
           "teacher_root": "corpus/teacher_out", "second_root": "corpus/second_out",
           "parakeet_root": "corpus/parakeet_out", "selection": sel.relative_to(root).as_posix(), "cache_dir": "cache",
           "runs_root": "runs", "sources": ["src_a"], "eval_sets": EVAL_SETS, "device": "cpu", "autocast": "none",
           "loss": {"w_ctc": 0.8},
           "eval": {"batch_s": BATCH_S, "greedy_subset": 3, "check_baselines": True, "probe_greedy_audio_s": 3,
                    "verdict_version": 2}}
    (root / "ctc.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return dict(root=root, fc=fc, sel=sel, student=student, pk_rows=pk_rows, manifest=manifest, targets=targets,
                text=text, eval_store=eval_store, manifest_path=sel.parent / "study_manifest.json",
                config=root / "ctc.json")


def test_ctc_eval_on_its_own_targets(ctc_env):
    """ctc_eval of a student whose stored targets are its own: frame argmax agreement 1 and the teacher's text on
    every row (cer_teacher 0, the teacher's CER = the student's), a KL of the targets' fp16 storage only, the CTC
    target tokens = the greedy path's, blank-certain and dense frames both present; a row whose stored n_frames
    differs is left out of the frame metrics and listed; summarise_ctc_tf's measures follow from the sums."""
    from transformers import AutoProcessor

    from kitsune import ctc_student as CS
    from kitsune.ctc_targets import FrameTargets

    env = ctc_env
    store = env["eval_store"]
    ids = [u.id for u in store.utts]
    targets, text = dict(env["targets"]), env["text"]
    model = load_patched(env["student"])
    feat = CS.ctc_features(env["student"], "cpu").logmel
    tok = AutoProcessor.from_pretrained(str(env["student"])).tokenizer
    refs = {u.id: r for u, r in zip(store.utts, store.frame()["ref"])}
    trows = ev.ctc_teacher_rows({i: dict(ref="x", ctc_hyp=text[i], ctc_cer=utt_cer(text[i], refs[i])) for i in ids},
                                refs=refs)
    bad = ids[3]
    t = targets[bad]
    targets[bad] = FrameTargets(t.n_frames + 1, np.concatenate([t.blank_lp, [0.0]]).astype(np.float16), t.dense_frame,
                                t.topk_idx, t.topk_lp, t.ctc_ids)
    res = ev.ctc_eval(model, store, None, feat, "cpu", BATCH_S, tokenizer=tok, teacher_rows=trows, targets=targets)
    g = res["greedy"]
    assert list(g.columns) == list(ev.GREEDY_COLUMNS) and sorted(g["id"]) == sorted(ids)
    assert (g["hyp"] == g["teacher_hyp"]).all() and (g["cer_teacher"] == 0).all() and not g["truncated"].any()
    assert (g["cer_ref"] == g["teacher_cer"]).all() and (g["ref"] == g["id"].map(refs)).all()
    assert (g["n_tok"] > 0).all() and g["hyp"].str.len().gt(0).all()
    assert res["frame_mismatch"] == [dict(id=bad, student=t.n_frames, stored=t.n_frames + 1)]
    tf = res["tf"]
    assert bad not in set(tf["id"]) and len(tf) == len(ids) - 1 and not res["no_targets"]
    assert tf["n_tok"].tolist() == [len(targets[i].ctc_ids) for i in tf["id"]]
    summ, per = ev.summarise_ctc_tf(tf)
    a = summ["all"]
    assert a["top1"] == 1.0 and abs(a["kl_per_frame"]) < 1e-3 and a["n_tok"] == int(tf["n_tok"].sum())
    assert a["kl"] == pytest.approx((tf["sum_kl_dense"].sum() + tf["sum_kl_blank"].sum()) / tf["n_tok"].sum())
    assert a["ce"] == a["ctc"] and a["argmax_blank"] == a["teacher_blank"]
    assert 0.2 < a["teacher_blank"] < 0.8 and 0.2 < a["frac_dense"] < 0.8  # both kinds of frame are there
    assert set(summ["sets"]) == set(EVAL_SETS) and len(per) == len(tf)
    # the combined loss is the CTC objective per target token
    c = ev.combined_loss(summ, 1.0, 0.8)
    gate = [d for s, d in summ["sets"].items() if s in ev.GATE_SETS]
    n = sum(d["n_tok"] for d in gate)
    assert c["value"] == pytest.approx(sum((d["kl"] + 0.8 * d["ce"]) * d["n_tok"] for d in gate) / n)


def _student(env):
    from transformers import AutoProcessor

    from kitsune import ctc_student as CS

    return (load_patched(env["student"]), CS.ctc_features(env["student"], "cpu").logmel,
            AutoProcessor.from_pretrained(str(env["student"])).tokenizer)


def test_ctc_eval_reads_a_frame_stores_batches(ctc_env, monkeypatch):
    """A frame store's batches carry their rows' targets (the CTC trainer's FrameBatchDataset: collate_frame_targets of
    the rows, padded to the batch's longest n_frames; here its twin over the token store): ctc_eval takes them from the
    batch (batch_dataset) and gives the targets-dict path's per-utterance sums and text to the bit."""
    from kitsune import trainset
    from kitsune.ctc_targets import collate_frame_targets

    env = ctc_env
    store, targets = env["eval_store"], env["targets"]
    model, feat, tok = _student(env)

    class FrameLike:  # FrameBatchDataset's items over a token store
        def __init__(self, st):
            self.ds = trainset.AudioBatchDataset(st)

        def __getitem__(self, idx):
            item = self.ds[idx]
            if item["ids"]:
                item.update(collate_frame_targets([targets[i] for i in item["ids"]]))
            return item

    a = ev.ctc_eval(model, store, None, feat, "cpu", BATCH_S, tokenizer=tok, targets=targets)
    monkeypatch.setattr(ev, "batch_dataset", FrameLike)
    b = ev.ctc_eval(model, store, None, feat, "cpu", BATCH_S, tokenizer=tok)  # no targets: the batches have them
    assert len(a["tf"]) == len(store) and not b["frame_mismatch"] and not b["no_targets"]
    pd.testing.assert_frame_equal(a["tf"], b["tf"], check_exact=True)
    assert a["greedy"]["hyp"].tolist() == b["greedy"]["hyp"].tolist()
    sa, pa_ = ev.summarise_ctc_tf(a["tf"])
    sb, pb = ev.summarise_ctc_tf(b["tf"])
    assert sa == sb and tuple(pb.columns) == ev.CTC_TF_COLUMNS
    assert {"argmax_agree", "frames_per_token", "kl_per_frame", "teacher_blank"} <= set(sb["all"])


def test_the_ctc_eval_is_the_ctc_trainers_on_its_frame_store(ctc_env, tmp_path):
    """Where the CTC trainer's frame stores exist (kitsune.trainset.build_frame_stores, kitsune.ctc_eval; skipped
    before they do): ctc_eval over a frame store of the eval sets gives, row for row, the sums of the trainer's own
    pass (ctc_eval_records), and summarise_ctc_tf its summary and per-utterance table to the bit, key for key; the
    greedy CTC paths are the same."""
    from kitsune import trainset

    if not hasattr(trainset, "build_frame_stores"):
        pytest.skip("no frame stores in kitsune.trainset (the CTC trainer's, wave 2)")
    CE = pytest.importorskip("kitsune.ctc_eval")
    env = ctc_env
    fc = env["fc"]
    fstore = trainset.build_frame_stores(env["sel"], fc.data, env["root"] / "corpus" / "parakeet_out", tmp_path / "fs",
                                         EVAL_SETS, ["eval"], log=print)
    model, feat, tok = _student(env)
    res = ev.ctc_eval(model, fstore, None, feat, "cpu", BATCH_S, tokenizer=tok)
    raw, hyps, dropped = CE.ctc_eval_records(model, fstore, feat, "cpu", BATCH_S)
    assert not res["dropped"] and not dropped and len(res["tf"]) == len(raw) == len(fstore)
    tf = res["tf"]
    assert tf["id"].tolist() == raw["id"].tolist() and tf["n_tok"].tolist() == raw["n_tok"].tolist()
    for k in ("kl_dense", "kl_blank", "ctc", "n_dense", "n_blank_frames", "argmax_agree", "argmax_blank",
              "teacher_blank"):
        assert tf[f"sum_{k}"].tolist() == raw[f"sum_{k}"].tolist(), k
    mine, mine_per = ev.summarise_ctc_tf(tf)
    theirs, theirs_per = CE.summarise_ctc_tf(raw)
    assert mine["sets"] == theirs["sets"] and mine["all"] == theirs["all"]
    pd.testing.assert_frame_equal(mine_per, theirs_per, check_exact=True, check_dtype=False)
    g = res["greedy"]
    assert [list(x) for x in g["hyp_ids"]] == [list(hyps[i]) for i in g["id"]]
    assert (g["hyp"] == g["teacher_hyp"]).all()  # the frame store's text is the stored ctc_hyp: its own here


def test_parakeet_baselines_and_the_stored_ctc_cer(ctc_env, tmp_path):
    """parakeet_baselines: the corpus CER of the stored ctc_hyp and hyp per set (on every row, or the manifest's), and
    a drift of more than 0.05 pp from the registered numbers refuses; the stored per-utterance ctc_cer is this
    scorer's CER of the stored ctc_hyp on every row."""
    pk = ctc_env["root"] / "corpus" / "parakeet_out"
    rows = ev.load_teacher_rows(pk, ev.GATE_SETS, split="eval")
    base = ev.parakeet_baselines(pk, ev.GATE_SETS, prereg={})
    for s in ev.GATE_SETS:
        r = [x for x in rows.values() if x["source"] == s]
        assert base[s]["cer_corpus"] == ev.corpus_cer([x["ctc_hyp"] for x in r], [x["ref"] for x in r])["cer"]
        assert base[s]["n"] == len(r) and base[s]["prereg"] is None
    t = ev.utterance_table(list(rows), [r["source"] for r in rows.values()], [r["ref"] for r in rows.values()],
                           [r["ctc_hyp"] for r in rows.values()])
    has = t["ref_len"] > 0
    np.testing.assert_allclose(np.round(t["edits"][has] / t["ref_len"][has], 4),
                               [r["ctc_cer"] for r, h in zip(rows.values(), has) if h], atol=1e-12)
    reg = {s: base[s]["cer_corpus"] for s in ev.GATE_SETS}
    assert ev.parakeet_baselines(pk, prereg=reg)["eval_cv8"]["prereg"] == reg["eval_cv8"]
    with pytest.raises(ValueError, match="eval_jsut: Parakeet CTC corpus CER"):
        ev.parakeet_baselines(pk, prereg=dict(reg, eval_jsut=reg["eval_jsut"] + 0.001))
    assert ev.parakeet_baselines(pk, prereg=dict(reg, eval_jsut=reg["eval_jsut"] + 0.001), check=False)


# ------------------------------------------------------------------------------------------------ 05 (ctc)


def events(out: Path, kind: str) -> list[dict]:
    return [r for r in (json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
                        if x.strip()) if r["kind"] == kind]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ctc_run(ctc_env, tmp_path_factory):
    """05_evaluate on the CTC student with the probe, against the manifest next to the selection, tables to a dir of
    their own: the family switch end to end."""
    m05 = load_script("05_evaluate")
    out = tmp_path_factory.mktemp("ctc_out") / "out"
    tables = out.parent / "tables"
    assert m05.main(["--root", str(ctc_env["root"]), "--config", str(ctc_env["config"]), "--ckpt",
                     str(ctc_env["student"]), "--out", str(out), "--probe", "--history", "none", "--tables",
                     str(tables), "--chunk-s", "4"]) == 0
    return dict(out=out, tables=tables, m05=m05)


def test_05_ctc_family_end_to_end(ctc_env, ctc_run):
    """The CTC student through 05: Parakeet's CTC path is its teacher (the student here, so the ratio is 1 on every
    set), the frame metrics and the probe come from the stored targets, summary.json has the trainer's keys, the
    verdict v2 is the family's, and a second run evaluates nothing."""
    out, m05 = ctc_run["out"], ctc_run["m05"]
    s = load(out / "summary.json")
    assert set(s["greedy_full"]["sets"]) == set(EVAL_SETS) and s["tf"]["n_frame_mismatch"] == 0
    for e, d in s["greedy_full"]["sets"].items():
        assert d["ratio_vs_teacher"] == 1.0 and d["cer_teacher_corpus"] == 0 and d["n_truncated"] == 0
    assert s["tf"]["all"]["top1"] == 1.0 and s["tf"]["all"]["kl_per_frame"] < 1e-3
    assert s["probe"]["all"]["top1"] == 1.0 and s["probe_greedy"]["all"]["cer_teacher_corpus"] == 0
    assert s["combined_loss"]["val_full"]["family"] == "ctc" and s["combined_loss"]["val_full"]["w_ctc"] == 0.8
    assert s["headline"]["val_top1"] == 1.0
    v = load(out / "verdict.json")
    assert v["version"] == 2 and v["family"] == "ctc" and v["teacher"] == "parakeet-ctc"
    assert v["teacher_prereg_status"] == "pending" and all(d["ratio"] == 1.0 for d in v["sets"].values())
    for e in EVAL_SETS:
        g = pd.read_parquet(out / f"greedy_{e}.parquet")
        pk = ctc_env["pk_rows"]
        assert (g["teacher_hyp"] == g["id"].map(lambda i: pk[i]["ctc_hyp"])).all()
        assert (g["teacher_cer"] == g["id"].map(lambda i: pk[i]["ctc_cer"])).all()
    (base,) = events(out, "teacher_baselines")
    assert base["family"] == "ctc" and base["prereg"] == "pending" and set(base["sets"]) == set(EVAL_SETS)
    (inv,) = load(out / "evaluator.json")["invocations"]
    assert inv["status"] == "complete" and inv["family"] == "ctc" and inv["system"] == "study-p01"
    assert m05.main(["--root", str(ctc_env["root"]), "--config", str(ctc_env["config"]), "--ckpt",
                     str(ctc_env["student"]), "--out", str(out), "--probe", "--history", "none", "--tables",
                     str(ctc_run["tables"])]) == 0
    assert events(out, "todo")[-1]["sets"] == [] and events(out, "todo")[-1]["probe"] is False


def test_05_tables_and_study_json(ctc_env, ctc_run):
    """The student's tables (one per set, manifest order, CONTRACT.md 5 columns) load in tools/study_report.py against
    the manifest; study.json holds every stratum with the Galgame views separate, the teacher's CER on the same rows
    (ratio 1 here) and M4."""
    out, tables = ctc_run["out"], ctc_run["tables"]
    man = ss.parse_manifest(ctc_env["manifest"])
    t = {e: pd.read_parquet(tables / "study-p01" / f"{e}.parquet") for e in EVAL_SETS}
    for e, df in t.items():
        assert tuple(df.columns) == ev.TABLE_COLUMNS and df["id"].tolist() == man.sets[e]
        g = pd.read_parquet(out / f"greedy_{e}.parquet").set_index("id").loc[man.sets[e]]
        assert df["hyp"].tolist() == g["hyp"].tolist()
        assert ev.table_cer(df)["cer"] == load(out / "summary.json")["greedy_full"]["sets"][e]["cer_ref_corpus"]
    loaded, _ = study_report.load_tables(tables, man.sets)
    corpus = ss.build_corpus(loaded, man)
    assert set(corpus.strata) == {"eval_jsut", "eval_cv8", "eval_reazon", "galgame_neutral", "galgame_all",
                                  "galgame_label_box"}
    st = load(out / "study.json")
    assert st["system"] == "study-p01" and st["family"] == "ctc" and st["teacher"] == "parakeet-ctc"
    assert set(st["strata"]) == set(corpus.strata) and st["refused"] == {}
    assert all(r["ratio_vs_teacher"] in (1.0, None) for r in st["strata"].values())
    point = ss.point_sums(corpus)
    for name in corpus.strata:
        assert st["strata"][name]["cer"] == pytest.approx(float(ss.stratum_cer(point, name)[0]))
    assert st["metrics"]["m4"] == pytest.approx(np.mean([st["strata"][k]["cer"] for k in ss.M4_SETS]))
    assert st["metrics"]["m4_ratio"] == 1.0
    assert st["manifest"]["sha256"] and st["manifest"]["prereg_check"]["status"] == "not_checked"


def test_05_teachers_and_from_evals(ctc_env, ctc_run, tmp_path):
    """--teachers: cohere, parakeet-ctc and parakeet-tdt tables from the stored hypotheses on the manifest rows, the
    stored per-utterance CERs this scorer's; --from-evals: the tables of an eval dir, equal to those 05 wrote after
    its eval. Both load in the report next to the student's."""
    m05 = ctc_run["m05"]
    tables = tmp_path / "tables"
    out = tmp_path / "teachers"
    off = ["--set", "eval.check_baselines=false"]  # Cohere's D32a numbers are the real sets', not this corpus'
    assert m05.main(["--root", str(ctc_env["root"]), "--config", str(ctc_env["config"]), "--teachers", "--out",
                     str(out), "--tables", str(tables), *off]) == 0
    rec = load(out / "teachers.json")
    assert set(rec["teachers"]) == {"cohere", "parakeet-ctc", "parakeet-tdt"}
    assert all(v["stored_cer_mismatches"] == 0 for v in rec["teachers"].values())
    man = ss.parse_manifest(ctc_env["manifest"])
    teach = ev.load_teacher_rows(ctc_env["fc"].teacher_out, EVAL_SETS, split="eval")
    for e in EVAL_SETS:
        c = pd.read_parquet(tables / "cohere" / f"{e}.parquet")
        assert c["id"].tolist() == man.sets[e] and c["hyp"].tolist() == [teach[i]["hyp"] for i in man.sets[e]]
        p = pd.read_parquet(tables / "parakeet-ctc" / f"{e}.parquet")
        assert p["hyp"].tolist() == [ctc_env["pk_rows"][i]["ctc_hyp"] for i in man.sets[e]]
    ev_out = tmp_path / "from"
    assert m05.main(["--from-evals", str(ctc_run["out"]), "--system", "study-p01-copy", "--manifest",
                     str(ctc_env["manifest_path"]), "--out", str(ev_out), "--tables", str(tables)]) == 0
    for e in EVAL_SETS:
        pd.testing.assert_frame_equal(pd.read_parquet(tables / "study-p01-copy" / f"{e}.parquet"),
                                      pd.read_parquet(ctc_run["tables"] / "study-p01" / f"{e}.parquet"))
    corpus = ss.build_corpus(study_report.load_tables(tables, man.sets)[0], man)
    assert set(corpus.systems) == {"cohere", "parakeet-ctc", "parakeet-tdt", "study-p01-copy"}
    # the student's teacher is itself here: its tables equal parakeet-ctc's hypotheses
    j = corpus.index("study-p01-copy")
    np.testing.assert_array_equal(corpus.strata["eval_jsut"].edits[:, j],
                                  corpus.strata["eval_jsut"].edits[:, corpus.index("parakeet-ctc")])
    # a reference that differs between the roots refuses the teachers' tables
    bad = tmp_path / "pk_bad"
    import shutil

    shutil.copytree(ctc_env["root"] / "corpus" / "parakeet_out", bad)
    f = next((bad / "eval_cv8").glob("*.jsonl"))
    lines = f.read_text(encoding="utf-8").splitlines()
    r0 = json.loads(lines[0])
    r0["ref"] = r0["ref"] + "違"
    f.write_text("\n".join([json.dumps(r0, ensure_ascii=False)] + lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="another reference"):
        m05.main(["--root", str(ctc_env["root"]), "--config", str(ctc_env["config"]), "--teachers", "--out",
                  str(tmp_path / "t2"), "--set", f"parakeet_root={bad.as_posix()}", *off])


def _refused(m05, env, tmp_path, monkeypatch, *extra, match: str):
    def no_model(*a, **k):
        raise AssertionError("a model was loaded before the refusal")

    monkeypatch.setattr(m05, "setup_ctc", no_model)
    with pytest.raises(SystemExit, match=match):
        m05.main(["--root", str(env["root"]), "--config", str(env["config"]), "--ckpt", str(env["student"]), "--out",
                  str(tmp_path / "o"), "--history", "none", *extra])


def test_05_refuses_a_manifest_that_does_not_match(ctc_env, tmp_path, monkeypatch):
    """Refused before any model is loaded: a manifest whose ids do not hash to its recorded sha256; one that lacks a row
    of the eval store (hashes consistent); one that differs from a filled PREREG.json manifest block; an aed config on
    a CTC checkpoint."""
    m05 = load_script("05_evaluate")
    man = json.loads(json.dumps(ctc_env["manifest"]))
    man["sets"]["eval_cv8"]["ids_sha256"] = "0" * 64
    p = tmp_path / "m1.json"
    p.write_text(json.dumps(man), encoding="utf-8")
    _refused(m05, ctc_env, tmp_path, monkeypatch, "--manifest", str(p), match="hash to")
    man = json.loads(json.dumps(ctc_env["manifest"]))
    ids = man["sets"]["eval_reazon"]["ids"][1:]
    man["sets"]["eval_reazon"].update(ids=ids, ids_sha256=ids_sha256(ids), n=len(ids))
    p = tmp_path / "m2.json"
    p.write_text(json.dumps(man), encoding="utf-8")
    _refused(m05, ctc_env, tmp_path, monkeypatch, "--manifest", str(p), match="not in the manifest")
    prereg = json.loads((ROOT / "study" / "PREREG.json").read_text(encoding="utf-8"))
    prereg["manifest"] = dict(prereg["manifest"], status="filled",
                              sets={s: {"ids_sha256": b["ids_sha256"], "n": b["n"]}
                                    for s, b in ctc_env["manifest"]["sets"].items()})
    prereg["manifest"]["sets"]["eval_jsut"]["ids_sha256"] = "a" * 64
    pp = tmp_path / "PREREG.json"
    pp.write_text(json.dumps(prereg), encoding="utf-8")
    _refused(m05, ctc_env, tmp_path, monkeypatch, "--prereg", str(pp), match="differs from PREREG")
    cfg = load(ctc_env["config"])
    cfg["family"] = "aed"
    q = tmp_path / "aed.json"
    q.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(SystemExit, match="is a ctc model"):
        m05.main(["--root", str(ctc_env["root"]), "--config", str(q), "--ckpt", str(ctc_env["student"]), "--out",
                  str(tmp_path / "o2")])


def test_05_refuses_frames_that_do_not_align(ctc_env, tmp_path, monkeypatch):
    """Decision 15 on a token store: an eval row whose student frame count is not its stored n_frames refuses the eval
    at the chunk that holds it (the finished chunks kept); the same command refuses again before it evaluates any
    other chunk. A probe row does too (1 of 4 is above 0.1 % of the probe's train rows), after the sets are done. (The
    stored targets are right here; the eval reads others: the CTC trainer's frame preflight on the real files passes.)"""
    from kitsune.ctc_targets import FrameTargets

    m05 = load_script("05_evaluate")
    real = m05.parakeet_data
    env = ctc_env
    bad_eval = env["manifest"]["sets"]["eval_cv8"]["ids"][1]

    def off_by_one(t):
        return FrameTargets(t.n_frames + 1, np.concatenate([t.blank_lp, [0.0]]).astype(np.float16), t.dense_frame,
                            t.topk_idx, t.topk_lp, t.ctc_ids)

    def corrupt(which):
        def parakeet_data(cfg, store, log, what="eval"):
            rows, targets, files = real(cfg, store, log, what)
            if targets is not None and what == which:
                bad = bad_eval if which == "eval" else sorted(targets)[0]
                targets = dict(targets, **{bad: off_by_one(targets[bad])})
            return rows, targets, files
        return parakeet_data

    argv = ["--root", str(env["root"]), "--config", str(env["config"]), "--ckpt", str(env["student"]), "--history",
            "none", "--chunk-s", "3"]
    out = tmp_path / "o"
    monkeypatch.setattr(m05, "parakeet_data", corrupt("eval"))
    with pytest.raises(SystemExit, match=r"REFUSED \(decision 15\): 1 eval row"):
        m05.main([*argv, "--out", str(out)])
    n_chunks = len(events(out, "chunk"))
    assert events(out, "frame_mismatch_refused")[0]["mismatch"][0]["id"] == bad_eval
    with pytest.raises(SystemExit, match="decision 15"):
        m05.main([*argv, "--out", str(out)])
    assert len(events(out, "chunk")) == n_chunks and len(events(out, "frame_mismatch_refused")) == 2
    assert not (out / "summary.json").exists() and not (out / "study.json").exists()
    monkeypatch.setattr(m05, "parakeet_data", corrupt("probe"))
    out2 = tmp_path / "p"
    with pytest.raises(SystemExit, match=r"REFUSED \(decision 15\): 1 probe row\(s\) of 4 .*more than 0.1 %"):
        m05.main([*argv, "--out", str(out2), "--probe"])
    assert all((out2 / f"greedy_{e}.parquet").exists() for e in EVAL_SETS)  # the sets were done first


def test_later_keys_are_set_aside_only_while_the_trainer_lacks_them():
    """A study config's family / loss.w_ctc / pull_parakeet / selection_recipe.study pass the trainer's merge before
    04_distill.DEFAULTS has them, and come back with their values; a key DEFAULTS knows is merged as usual."""
    m05 = load_script("05_evaluate")

    class FakeD:
        DEFAULTS = {"loss": {"w_kl": 1.0}, "selection_recipe": {"agree_max": 0.5}, "family": "aed"}

    raw = {"family": "ctc", "loss": {"w_kl": 1.0, "w_ctc": 0.5}, "selection_recipe": {"study": {"x": 1}},
           "pull_parakeet": True}
    later = m05.set_aside_later_keys(FakeD, raw)
    assert later == {"loss.w_ctc": 0.5, "pull_parakeet": True, "selection_recipe.study": {"x": 1}}
    assert raw == {"family": "ctc", "loss": {"w_kl": 1.0}, "selection_recipe": {}}


# ------------------------------------------------------------------------------------------------ real data

LABELS = Path(os.environ.get("KITSUNE_LABELS_ROOT") or "D:/kitsune-labels/full")
FIRST_RUN = REAL / "cache" / "hf_runs" / "runs" / "viability-b20x2560-20260925T071746Z"


def test_the_label_boxs_stored_hyps_score_to_their_stored_cers_and_the_measured_baselines():
    """The label box's eval shards (a read-only sample pull of labels/full): every row's stored ctc_cer and cer are this
    scorer's CER of its stored ctc_hyp and hyp, and the corpus CERs are the measured ones: Parakeet TDT / CTC JSUT
    6.62 / 6.71, CV8 7.50 / 7.58, Reazon 10.20 / 9.71 %; Cohere (teacher_out) 8.30 / 4.07 / 6.28 %."""
    pk, co = LABELS / "parakeet_out", LABELS / "teacher_out"
    need_real(*(pk / s for s in ev.GATE_SETS), *(co / s for s in ev.GATE_SETS))
    rows = ev.load_teacher_rows(pk, ev.GATE_SETS, split="eval")
    ids = list(rows)
    for hk, ck in (("ctc_hyp", "ctc_cer"), ("hyp", "cer")):
        t = ev.utterance_table(ids, [rows[i]["source"] for i in ids], [rows[i]["ref"] for i in ids],
                               [rows[i][hk] for i in ids])
        got = [round(int(e) / int(n), 4) for e, n in zip(t["edits"], t["ref_len"]) if n]
        want = [rows[i][ck] for i, n in zip(ids, t["ref_len"]) if n]
        assert len(got) > 10_000 and got == want, hk  # every row: the stored CER is this scorer's
    base = ev.parakeet_baselines(pk, ev.GATE_SETS, prereg={})
    for s, (tdt, ctc) in {"eval_jsut": (0.0662, 0.0671), "eval_cv8": (0.0750, 0.0758),
                          "eval_reazon": (0.1020, 0.0971)}.items():
        assert base[s]["tdt_cer_corpus"] == pytest.approx(tdt, abs=5e-5)
        assert base[s]["cer_corpus"] == pytest.approx(ctc, abs=5e-5)
    ev.teacher_baselines(co, ev.GATE_SETS, check=True)  # Cohere's D32a numbers, 0.05 pp


def test_the_real_parakeet_ctc_teacher_through_ctc_eval(tmp_path):
    """kitsune.evaluate.ctc_eval with the real Parakeet CTC teacher (the unpruned 24x4096 anchor, fp32 on CPU) as the
    student, on real eval rows of the three gate sets (the first run's selection and data shards, the label box's
    stored targets and text): every row's frames align, the greedy text is the stored ctc_hyp on (nearly) every row
    (the label box ran the encoder in bf16 on a GPU), the frame argmax agrees on > 99 % of the frames, the KL is small
    and the argmax-blank share is the teacher's."""
    import json as _json

    from transformers import AutoProcessor

    from kitsune import ctc_student as CS
    from kitsune import trainset
    from kitsune.ctc_targets import load_ctc_targets
    from kitsune.patches import patch_relpos_once_per_batch

    sel_path, data = REAL / "selection" / "viability.parquet", REAL / "data"
    pk_dir = REAL / "cache" / "parakeet-tdt_ctc-0.6b-ja-hf"
    need_real(sel_path, data / "shards" / "eval_jsut", pk_dir / "ctc_head.safetensors",
              *(LABELS / "parakeet_out" / s for s in ev.GATE_SETS), *(LABELS / "teacher_out" / s for s in ev.GATE_SETS))
    sel = pd.read_parquet(sel_path)
    e = sel[(sel["split"] == "eval") & sel["keep"] & sel["source"].isin(ev.GATE_SETS)]
    ids = [i for s in ev.GATE_SETS for i in e["id"][e["source"] == s].tolist()[:8]]
    store = trainset.build_stores(sel_path, data, LABELS / "teacher_out", tmp_path / "store", list(ev.GATE_SETS),
                                  ["eval"], ids=ids, log=print)
    targets, rows = {}, {}
    for f in sorted(set(e["teacher_file"][e["id"].isin(ids)])):
        targets.update({k: v for k, v in load_ctc_targets(LABELS / "parakeet_out" / f"{f}.npz").items() if k in ids})
        for line in (LABELS / "parakeet_out" / f"{f}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip() and (r := _json.loads(line))["id"] in set(ids):
                rows[r["id"]] = r
    refs = {u.id: r for u, r in zip(store.utts, store.frame()["ref"].tolist())}
    torch.set_num_threads(4)
    model = CS.load_parakeet_ctc(pk_dir, "cpu")
    patch_relpos_once_per_batch(model)
    res = ev.ctc_eval(model, store, None, CS.ctc_features(pk_dir, "cpu").logmel, "cpu", 60.0,
                      tokenizer=AutoProcessor.from_pretrained(str(pk_dir), local_files_only=True).tokenizer,
                      teacher_rows=ev.ctc_teacher_rows(rows, refs=refs), targets=targets)
    g = res["greedy"]
    assert len(g) == len(ids) and not res["frame_mismatch"] and not res["no_targets"] and not res["dropped"]
    assert (g["hyp"] == g["teacher_hyp"]).mean() >= 0.9
    a = ev.summarise_ctc_tf(res["tf"])[0]["all"]
    assert a["top1"] > 0.99 and a["kl_per_frame"] < 0.05 and abs(a["argmax_blank"] - a["teacher_blank"]) < 0.01
    assert 0.5 < a["teacher_blank"] < 0.8


def test_the_first_a100_runs_per_set_cer_from_its_eval_parquets():
    """The first A100 run's final eval (its downloaded run dir, read only): the tables made from its
    greedy_<set>.parquet give its summary.json's per-set corpus CER to the edit (cer_ref_corpus, ref_edits, ref_chars)
    and the teacher's on the same rows, and load in the study report's corpus (manifest = its own ids)."""
    step = FIRST_RUN / "evals" / "step_9774"
    need_real(step / "summary.json", *(step / f"greedy_{s}.parquet" for s in ev.GATE_SETS))
    summ = load(step / "summary.json")["greedy_full"]["sets"]
    tables, teach, man = {}, {}, {"sets": {}}
    for s in summ:
        g = pd.read_parquet(step / f"greedy_{s}.parquet")
        tables[s] = ev.utterance_table(g["id"], s, g["ref"], g["hyp"], g["truncated"])
        teach[s] = ev.utterance_table(g["id"], s, g["ref"], g["teacher_hyp"], g["teacher_truncated"])
        c, d = ev.table_cer(tables[s]), summ[s]
        assert (c["edits"], c["ref_chars"]) == (d["ref_edits"], d["ref_chars"]) and c["cer"] == d["cer_ref_corpus"]
        assert ev.table_cer(teach[s])["cer"] == d["teacher_cer_ref_corpus"]
        assert int(tables[s]["truncated"].sum()) == d["n_truncated"]
        man["sets"][s] = {"ids": g["id"].tolist(), "ids_sha256": ids_sha256(g["id"].tolist())}
    assert round(100 * summ["eval_jsut"]["cer_ref_corpus"], 2) == 12.65
    corpus = ss.build_corpus({"anchor-b20": pd.concat(tables.values(), ignore_index=True),
                              "cohere": pd.concat(teach.values(), ignore_index=True)}, man)
    pooled = ss.metric_values(ss.point_sums(corpus), "gate_pooled")
    assert float(pooled[corpus.index("anchor-b20")]) == pytest.approx(0.1215, abs=5e-5)  # the run's val_cer 12.15 %
