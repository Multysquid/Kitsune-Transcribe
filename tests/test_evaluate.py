"""kitsune.evaluate: CER conventions, the pre-registered verdict, the teacher baselines, and both evals on a tiny random
CohereAsr model over a real (synthetic) trainset store. CPU only; the teacher tokenizer is used from the HF cache if
present (HF_HUB_OFFLINE), otherwise a stand-in."""
import json
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
import pytest  # noqa: E402
import torch  # noqa: E402

from kitsune import evaluate as ev  # noqa: E402
from kitsune.audio import decode_audio  # noqa: E402
from kitsune.text import cer as utt_cer  # noqa: E402

TEACHER_OUT = ROOT / "teacher_out"
EVAL = ["eval_jsut", "eval_cv8"]  # fake corpus sources named like the real sets


# ------------------------------------------------------------------------------------------------------------ CER


def test_corpus_cer_vs_per_utt_mean():
    refs = ["あいうえお", "かき", "さしすせそたちつてと"]
    hyps = ["あいうえお", "かく", "さしすせそたちつてと"]
    per = [utt_cer(h, r) for h, r in zip(hyps, refs)]
    assert per == [0.0, 0.5, 0.0]
    c = ev.corpus_cer(hyps, refs)
    assert c["edits"] == 1 and c["ref_chars"] == 17 and c["n"] == 3
    assert c["cer"] == pytest.approx(1 / 17)  # vs the per-utterance mean 1/6: short utterances dominate the mean
    assert np.mean(per) == pytest.approx(1 / 6)
    # insertions count against the reference length; deletion of a whole utterance counts its chars
    c = ev.corpus_cer(["あいうえおか", ""], ["あいうえお", "かき"])
    assert (c["edits"], c["ref_chars"]) == (3, 7)


def test_corpus_cer_normalisation_and_empty_refs():
    # NFKC + lowercase + punctuation/space stripping, exactly kitsune.text.normalize_ja
    c = ev.corpus_cer(["ABC、です。", "えー"], ["ａｂｃ です！", "…。"])
    assert c["cer"] == 0.0 and c["ref_chars"] == 5
    assert c["n"] == 1 and c["n_empty_ref"] == 1  # the punctuation-only reference is skipped, and counted
    assert np.isnan(ev.corpus_cer(["x"], ["、"])["cer"])


# -------------------------------------------------------------------------------------------------------- verdict


def final(ratios, trunc=(0, 0, 0), n=1000):
    """Final full-set greedy summary with the teacher at its pre-registered CER on the same ids."""
    sets = {}
    for s, r, k in zip(ev.GATE_SETS, ratios, trunc):
        t = ev.TEACHER_CER_PREREG[s]
        sets[s] = dict(cer_ref_corpus=r * t, teacher_cer_ref_corpus=t, n=n, n_truncated=k, trunc_rate=k / n)
    return dict(sets=sets)


def history(cer_ratio, heldout, probe, steps=None):
    steps = steps or [100 * i for i in range(len(cer_ratio))]
    out = []
    for st, c, h, p in zip(steps, cer_ratio, heldout, probe):
        g = {s: dict(cer_ref_corpus=c * t, teacher_cer_ref_corpus=t, trunc_rate=0.0, n=500)
             for s, t in ev.TEACHER_CER_PREREG.items()}
        out.append(dict(step=st, elapsed_s=st * 1.0, heldout_kl=h, probe_kl=p, greedy=g))
    return out


N = 12
FALLING = list(np.linspace(2.0, 1.1, N))  # CER ratio still improving at the end
FLAT = [1.3] * N
KL_DOWN = list(np.linspace(0.8, 0.30, N))
PROBE_DOWN = list(np.linspace(0.7, 0.25, N))  # gap 0.1 -> 0.05: not widening


def test_verdict_go():
    v = ev.verdict(dict(final=final((1.1, 1.15, 1.6), trunc=(1, 2, 0)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "GO", v["reasons"]
    assert v["n_sets_go"] == 2 and v["trunc_ok"] and v["trend"]["gap_widening"] is False
    assert v["sets"]["eval_jsut"]["threshold_go"] == pytest.approx(1.2 * 0.0830)
    assert v["sets"]["eval_jsut"]["teacher_source"] == "same ids"
    assert len(v["trend"]["window_steps"]) == 3  # max(3, ceil(0.2 * 12))


def test_verdict_truncation_blocks_go():
    v = ev.verdict(dict(final=final((1.1, 1.15, 1.6), trunc=(10, 5, 5)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "PROMISING" and not v["trunc_ok"]  # 20/3000 = 0.67 % > 0.5 %, still improving
    assert any("truncation" in r for r in v["reasons"])


def test_verdict_widening_gap_blocks_go():
    probe = list(np.linspace(0.7, 0.10, N))  # probe falls much faster than held-out: gap widens (no overfit yet)
    v = ev.verdict(dict(final=final((1.1, 1.1, 1.1)), history=history(FLAT, KL_DOWN, probe)))
    assert v["trend"]["gap_widening"] is True and not v["trend"]["overfit"]
    assert v["verdict"] == "INCONCLUSIVE"  # within 1.5x but CER flat: not a pre-registered tier


def test_verdict_promising():
    v = ev.verdict(dict(final=final((1.3, 1.4, 1.45)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "PROMISING" and v["trend"]["improving"] and v["n_sets_go"] == 0


def test_verdict_nogo_flat():
    v = ev.verdict(dict(final=final((1.8, 2.0, 1.4)), history=history([1.9] * N, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "NO-GO" and v["n_sets_promising"] == 1 and not v["trend"]["improving"]


def test_verdict_overfit_is_nogo_even_when_cer_passes():
    heldout = list(np.linspace(0.30, 0.45, N))  # held-out KL rising ...
    probe = list(np.linspace(0.30, 0.10, N))  # ... while the probe keeps falling
    v = ev.verdict(dict(final=final((1.0, 1.0, 1.0)), history=history(FALLING, heldout, probe)))
    assert v["verdict"] == "NO-GO" and v["trend"]["overfit"]
    assert any("over-fitting" in r for r in v["reasons"])


def test_verdict_inconclusive_and_teacher_override():
    v = ev.verdict(dict(final=final((1.8, 1.9, 2.0)), history=history(FALLING, KL_DOWN, PROBE_DOWN)))
    assert v["verdict"] == "INCONCLUSIVE"  # beyond 1.5x but improving: pre-registration is silent
    # results["teacher"] overrides the same-ids teacher CER; the drift vs the pre-registered number is reported
    fin = final((1.1, 1.1, 1.1))
    v = ev.verdict(dict(final=fin, history=history(FALLING, KL_DOWN, PROBE_DOWN),
                        teacher={"eval_jsut": 0.0831}))
    assert v["sets"]["eval_jsut"]["teacher_source"] == "results.teacher"
    assert v["sets"]["eval_jsut"]["baseline_drift"] == pytest.approx(0.0001)
    # no history at all: trends unknown, so GO cannot be certified
    v = ev.verdict(dict(final=final((1.0, 1.0, 1.0)), history=[]))
    assert v["verdict"] == "INCONCLUSIVE" and v["trend"]["gap_widening"] is None


def test_verdict_trends_leave_out_the_untrained_step_0_eval():
    """Per-epoch evals leave short histories: step 0, an epoch end or two, the final eval. The untrained student's
    step-0 record (CER ratio ~30, held-out KL ~5) must not enter the trends, or it makes every run 'improving' and hides
    over-fitting; the verdict is the one the trained records give."""
    steps = [0, 1126, 2252]
    # over-fitting on the trained records: held-out KL +10 % while the probe falls, CER within 1.2x
    for probe0 in (4.5, 5.75):  # the step-0 probe KL below or above the step-0 held-out KL
        v = ev.verdict(dict(final=final((1.1, 1.1, 1.15)),
                            history=history([30, 1.12, 1.10], [4.87, 0.60, 0.66], [probe0, 0.62, 0.45], steps)))
        assert v["verdict"] == "NO-GO" and v["trend"]["overfit"], v["reasons"]
        assert v["trend"]["window_steps"] == [1126, 2252]
    # a flat tail with 1/3 sets within 1.5x is the pre-registered NO-GO, not 'still improving'
    v = ev.verdict(dict(final=final((1.8, 2.0, 1.4)),
                        history=history([30, 1.9, 1.9], [4.87, 0.50, 0.45], [4.5, 0.45, 0.40], steps)))
    assert v["verdict"] == "NO-GO" and not v["trend"]["improving"], v["reasons"]
    # a GO on the trained records stays GO whatever the step-0 numbers were
    for probe0 in (4.5, 5.75):
        v = ev.verdict(dict(final=final((1.1, 1.1, 1.15)),
                            history=history([30, 1.25, 1.10], [4.87, 0.70, 0.60], [probe0, 0.62, 0.53], steps)))
        assert v["verdict"] == "GO", v["reasons"]
    # one trained record: the trend is unknown, not 'improving' (from step 0) and not 'flat'
    for ratios in ((1.3, 1.4, 1.45), (1.8, 2.0, 1.4)):
        v = ev.verdict(dict(final=final(ratios), history=history([30, 1.3], [4.87, 0.6], [4.5, 0.55], [0, 1126])))
        assert v["verdict"] == "INCONCLUSIVE" and v["trend"]["cer_ratio_rel_change"] is None, v["reasons"]
        assert v["trend"]["window_steps"] == [1126] and any("CER trend unknown" in r for r in v["reasons"])
        assert not any("still improving" in r or "CER flat" in r for r in v["reasons"])


def test_eval_record_and_flatten():
    tf = dict(sets={"eval_jsut": dict(kl=0.2, ce=0.3, top1=0.9, n_tok=10)}, all=dict(kl=0.2, ce=0.3, top1=0.9),
              wall_s=1.5, bad_audio=["x"])
    probe = dict(sets={}, all=dict(kl=0.1))
    rec = ev.eval_record(300, 1200.0, tf=tf, probe=probe)
    assert rec["heldout_kl"] == 0.2 and rec["probe_kl"] == 0.1 and rec["tf"]["eval_jsut"]["top1"] == 0.9
    flat = ev.flatten(tf, "eval_tf")
    assert flat["eval_tf/eval_jsut/kl"] == 0.2 and flat["eval_tf/all/top1"] == 0.9 and flat["eval_tf/wall_s"] == 1.5
    assert not any("bad_audio" in k for k in flat)


# ------------------------------------------------------------------------------------------------ teacher baselines


@pytest.mark.skipif(not all((TEACHER_OUT / s).is_dir() for s in ev.GATE_SETS), reason="real teacher_out not present")
def test_teacher_baselines_match_preregistered():
    b = ev.teacher_baselines(TEACHER_OUT)  # raises if any set drifts by > 0.05 pp
    for s, want in {"eval_jsut": 0.0830, "eval_cv8": 0.0407, "eval_reazon": 0.0628}.items():
        assert abs(b[s]["cer_corpus"] - want) <= 0.0005, (s, b[s])
        assert b[s]["n_empty_ref"] == 0
    assert b["eval_jsut"]["n"] == 5000 and b["eval_cv8"]["n"] == 4483 and b["eval_reazon"]["n"] == 5263
    assert b["eval_reazon"]["cer_mean"] > b["eval_reazon"]["cer_corpus"]  # the mean over-weights short utterances


def test_teacher_baselines_refuse_drift(tmp_path):
    d = tmp_path / "eval_jsut"
    d.mkdir()
    rows = [dict(id=f"eval_jsut/{i}", hyp="あいうえおかきくけこ", ref="あいうえおかきくけこ", cer=0.0, duration=1.0,
                 n_tok=5, truncated=False) for i in range(9)]
    rows.append(dict(id="eval_jsut/9", hyp="", ref="あいうえおかきくけこ", cer=1.0, duration=1.0, n_tok=1, truncated=False))
    (d / "eval-00000.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered"):
        ev.teacher_baselines(tmp_path, ["eval_jsut"])
    b = ev.teacher_baselines(tmp_path, ["eval_jsut"], check=False)
    assert b["eval_jsut"]["cer_corpus"] == pytest.approx(0.1) and b["eval_jsut"]["cer_mean"] == pytest.approx(0.1)


# --------------------------------------------------------------------------------------- tiny model integration


def tiny_model(seed=0):
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    enc = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
               subsampling_conv_channels=8)
    cfg = CohereAsrConfig(encoder_config=enc, vocab_size=16384, hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=64, max_position_embeddings=128,
                          decoder_start_token_id=13764)
    torch.manual_seed(seed)
    m = CohereAsrForConditionalGeneration._from_config(cfg, attn_implementation="sdpa")
    with torch.no_grad():
        m.proj_out.weight.mul_(40.0)  # peaky next-token distributions: no near-ties for argmax to flip on
    return m.eval()


class StandInTokenizer:
    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["".join(chr(0x4E00 + t % 2000) for t in s if not (skip_special_tokens and t < 16)) for s in seqs]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return ev.teacher_tokenizer()
    except Exception:
        return StandInTokenizer()


@pytest.fixture(scope="module")
def featurizer():
    from kitsune.features import LogMel

    return LogMel()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from fixtures import make_fake_corpus

    return make_fake_corpus(tmp_path_factory.mktemp("corpus"),
                            {"src_a": (4, "train"), "eval_jsut": (7, "eval"), "eval_cv8": (5, "eval")},
                            rows_per_shard=8, seed=3, dur_range=(0.4, 2.2), no_second=("eval_jsut", "eval_cv8"))


@pytest.fixture(scope="module")
def selection(corpus):
    from fixtures import make_fake_selection

    return make_fake_selection(corpus, greedy_n=4, probe_n=2)


def self_targets(model, featurizer, wave: np.ndarray, T: int, k: int = 16):
    """The model's own greedy continuation of PROMPT for T steps and its top-k log-probs, by an independent path:
    one utterance, full forward per step (no batching, no padding, no KV cache)."""
    feats, mask = featurizer(torch.from_numpy(wave)[None], torch.tensor([len(wave)]))
    dec = list(ev.PROMPT)
    toks, idx, lps = [], [], []
    with torch.no_grad():
        for _ in range(T):
            logits = model(input_features=feats, attention_mask=mask, decoder_input_ids=torch.tensor([dec]),
                           use_cache=False).logits[0, -1].float()
            v, i = torch.topk(logits.log_softmax(-1), k)
            toks.append(int(i[0]))
            idx.append(i.numpy())
            lps.append(v.numpy())
            dec.append(int(i[0]))
    return np.array(toks, np.int16), np.stack(idx).astype(np.int16), np.stack(lps).astype(np.float16)


@pytest.fixture(scope="module")
def self_store(corpus, selection, featurizer, tmp_path_factory):
    """Eval store whose teacher targets are replaced by the tiny model's own outputs (student == teacher)."""
    from kitsune.trainset import eval_store, load_stores

    cache = tmp_path_factory.mktemp("cache") / "eval"
    st = eval_store(selection, corpus.data, corpus.teacher_out, cache, EVAL, log=lambda s: None)
    model = tiny_model(0)
    tok = np.load(cache / "targets_tokens.npy")
    idx = np.load(cache / "targets_topk_idx.npy")
    lp = np.load(cache / "targets_topk_lp.npy")
    for u in st.utts:  # audio from the corpus ground truth, so no memmap is open while the files are rewritten
        t, i, l = self_targets(model, featurizer, decode_audio(corpus.utts[u.id].audio), u.n_tok)
        tok[u.tok_off:u.tok_off + u.n_tok], idx[u.tok_off:u.tok_off + u.n_tok], lp[u.tok_off:u.tok_off + u.n_tok] = t, i, l
    for name, arr in (("tokens", tok), ("topk_idx", idx), ("topk_lp", lp)):
        np.save(cache / f"targets_{name}.npy", arr)
    return load_stores(cache)


def test_teacher_forced_self_consistency(self_store, featurizer):
    """Student == teacher: batched, padded, teacher-forced eval through the training collate must give top-1 = 1 and
    KL ~ 0 at every position - an off-by-one in the target positions would not."""
    from kitsune.patches import freeze_batchnorm

    model = tiny_model(0)
    model.train()
    freeze_batchnorm(model)
    flags = {n: m.training for n, m in model.named_modules()}
    summary, per_utt = ev.teacher_forced_eval(model, self_store, featurizer, "cpu", batch_s=3.0)
    assert {n: m.training for n, m in model.named_modules()} == flags  # train mode and frozen BN restored

    assert len(per_utt) == len(self_store.utts) and set(per_utt["source"]) == set(EVAL)
    assert list(per_utt.columns[:7]) == ["id", "source", "n_tok", "kl", "ce", "top1", "duration"]
    assert (per_utt["top1"] == 1.0).all()
    assert per_utt["kl"].abs().max() < 2e-3  # fp16 storage of the log-probs is the only difference
    ntok = sum(u.n_tok for u in self_store.utts)
    assert summary["all"]["n_tok"] == ntok and summary["n_utts"] == len(self_store.utts)
    assert summary["all"]["top1"] == 1.0 and summary["all"]["kl"] < 2e-3
    # CE of the teacher's own token == the teacher's NLL of it (stored top-1 log-prob)
    nll = -np.concatenate([self_store.targets(i)[2][:, 0].astype(np.float64) for i in range(len(self_store.utts))])
    assert summary["all"]["ce"] == pytest.approx(nll.mean(), abs=2e-3)
    for s in EVAL:
        d = summary["sets"][s]
        assert d["n_utts"] == sum(u.source == s for u in self_store.utts)
        assert d["frac_p1_gt_0.99"] + d["frac_p1_lt_0.9"] <= 1.0
    assert "student_tail" in per_utt.columns and "teacher_p1" in per_utt.columns

    # a different model disagrees: KL clearly positive
    other, _ = ev.teacher_forced_eval(tiny_model(1), self_store, featurizer, "cpu", batch_s=3.0)
    assert other["all"]["kl"] > 0.1 and other["all"]["top1"] < 0.5

    # a subset by id (the probe path)
    ids = [u.id for u in self_store.utts][:3]
    sub, sub_utt = ev.teacher_forced_eval(model, self_store, featurizer, "cpu", batch_s=3.0, ids=ids)
    assert sorted(sub_utt["id"]) == sorted(ids)


def manual_greedy(model, featurizer, wave, max_new, eos=3):
    feats, mask = featurizer(torch.from_numpy(wave)[None], torch.tensor([len(wave)]))
    dec = list(ev.PROMPT)
    with torch.no_grad():
        for _ in range(max_new):
            t = int(model(input_features=feats, attention_mask=mask, decoder_input_ids=torch.tensor([dec]),
                          use_cache=False).logits[0, -1].argmax())
            dec.append(t)
            if t == eos:
                break
    return dec[len(ev.PROMPT):]


def test_greedy_eval(corpus, self_store, featurizer, tokenizer):
    model = tiny_model(0)
    ids = [u.id for u in self_store.utts if u.in_greedy_subset]
    assert ids
    summary, per_utt = ev.greedy_eval(model, self_store, ids, featurizer, "cpu", batch_s=3.0, tokenizer=tokenizer)
    assert list(per_utt.columns[:10]) == ["id", "source", "duration", "ref", "teacher_hyp", "hyp", "cer_ref",
                                          "cer_teacher", "truncated", "n_tok"]
    assert sorted(per_utt["id"]) == sorted(ids)
    max_new = min(int(16 + 10 * per_utt["duration"].max()), 128 - 10 - 1)  # per batch: from its longest utterance
    for r in per_utt.itertuples():
        u = corpus.utts[r.id]
        assert r.ref == u.text and r.teacher_hyp == u.hyp  # from the store index == teacher_out jsonl
        assert r.cer_ref == pytest.approx(utt_cer(r.hyp, r.ref)) and r.cer_teacher == pytest.approx(utt_cer(r.hyp, r.teacher_hyp))
        assert r.n_tok == len(r.hyp_ids) and r.truncated == (3 not in r.hyp_ids)
        assert r.n_tok <= max_new
        # batched generate with KV cache == the cacheless single-utterance greedy loop (RepetitionStop only cuts)
        ref_seq = manual_greedy(model, featurizer, decode_audio(u.audio), max_new)
        assert list(r.hyp_ids) == ref_seq[: r.n_tok]
    # teacher baseline on the same ids, from the fake teacher_out
    for s in EVAL:
        g = per_utt[per_utt["source"] == s]
        if not len(g):
            continue
        want = ev.corpus_cer([corpus.utts[i].hyp for i in g["id"]], [corpus.utts[i].text for i in g["id"]])["cer"]
        assert summary["sets"][s]["teacher_cer_ref_corpus"] == pytest.approx(want)
        assert summary["sets"][s]["cer_ref_corpus"] == pytest.approx(ev.corpus_cer(g["hyp"], g["ref"])["cer"])
    assert summary["all"]["n"] == len(ids) and summary["n_bad_audio"] == 0

    # teacher_rows (teacher_out jsonl) override the store text: with the student's own output as the teacher hyp,
    # the imitation CER is 0
    rows = {r.id: dict(ref=r.ref, hyp=r.hyp, cer=0.0, truncated=False) for r in per_utt.itertuples()}
    s2, p2 = ev.greedy_eval(model, self_store, ids, featurizer, "cpu", batch_s=3.0, tokenizer=tokenizer,
                            teacher_rows=rows)
    assert (p2["cer_teacher"] == 0.0).all()
    c = s2["all"]["cer_teacher_corpus"]
    assert c == 0.0 or np.isnan(c)  # NaN only if every hypothesis normalises to "" (nothing to divide by)


def test_greedy_eval_stops_at_eos(self_store, featurizer, tokenizer):
    model = tiny_model(0)
    with torch.no_grad():
        model.proj_out.bias[3] = 1e4  # EOS always wins
    summary, per_utt = ev.greedy_eval(model, self_store, None, featurizer, "cpu", batch_s=3.0, tokenizer=tokenizer)
    assert len(per_utt) == len(self_store.utts)
    assert (per_utt["n_tok"] == 1).all() and not per_utt["truncated"].any()
    assert (per_utt["hyp"] == "").all() and (per_utt["cer_ref"] == 1.0).all()
    assert summary["all"]["trunc_rate"] == 0.0 and summary["all"]["cer_ref_corpus"] == 1.0
    assert summary["all"]["n_empty_hyp"] == len(per_utt)
    samples = ev.pick_samples(per_utt, n=4)
    assert len(samples) == 4 and set(samples[0]) >= {"id", "ref", "teacher_hyp", "hyp", "cer_ref", "cer_teacher"}
