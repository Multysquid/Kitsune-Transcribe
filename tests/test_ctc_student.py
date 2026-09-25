"""Parakeet-family students (kitsune/ctc_student.py, scripts/03c_build_ctc_student.py) on CPU.

Everything here runs on TINY random models from tests/fixtures_ctc.py (the real vocab, blank, features and 8x
subsampling; a 32-wide encoder) plus the real dims on the meta device for the counts. The tests that load the real
0.6B Parakeet are in tests/test_ctc_real.py (marked slow)."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixtures_ctc import (BLANK, V, tiny_ctc_config, tiny_ctc_model, tiny_parakeet_dir, tiny_student_dir,  # noqa: E402
                          tiny_tokenizer, write_processor)

from kitsune import ctc_student as CS  # noqa: E402
from kitsune import parakeet as pk  # noqa: E402
from kitsune import student as S  # noqa: E402


def rand_importance(layers, width, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {(l, n): torch.rand(width, generator=g) for l in layers for n in S.FFN_NAMES}


def waves_of(lengths, seed=0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(0.1 * np.sin(np.arange(n) / 16000 * 2 * np.pi * rng.uniform(100, 2000)) +
             0.02 * rng.standard_normal(n)).astype(np.float32) for n in lengths]


def hf_features():
    from transformers import ParakeetFeatureExtractor

    from kitsune.features import LogMel

    fe = ParakeetFeatureExtractor()
    return fe, CS.CtcFeatures(LogMel.from_feature_extractor(fe))


# ------------------------------------------------------------------------------------------------ constants + counts


def test_constants_match_the_label_pass():
    assert (CS.CTC_VOCAB, CS.CTC_BLANK, CS.FRAME_S, CS.SUBSAMPLING) == (pk.VOCAB, pk.BLANK, pk.FRAME_S, 8)
    assert (CS.TEACHER_REPO, CS.TEACHER_REVISION) == (pk.NEMO_REPO, pk.NEMO_REVISION)


def test_closed_form():
    for L in (1, 4, 8, 16, 24):
        for F in (1, 768, 2560, 4096):
            assert CS.closed_form_ctc_params(L, F) == 5_911_553 + L * (8_422_400 + 4_098 * F)
    assert CS.closed_form_ctc_params(24, 4096) == 610_898_945


@pytest.mark.parametrize("n, ffn, layers, total, non_emb", [
    (16, 2560, [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23], 308_524_033, 305_374_208),
    (8, 768, [0, 3, 7, 10, 13, 16, 20, 23], 98_468_865, 95_319_040),
    (4, 768, [0, 8, 15, 23], 52_190_209, 49_040_384),
    (24, 4096, list(range(24)), 610_898_945, 607_749_120),
])
def test_study_shapes_on_the_meta_device(n, ffn, layers, total, non_emb):
    """STUDY.md 1.1: the P students' layers and counts at the real dims (the default ParakeetEncoderConfig is the
    converted teacher's encoder)."""
    from transformers import ParakeetForCTC
    from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig

    assert CS.resolve_layers([str(n)]) == CS.resolve_layers(n) == layers and layers[-1] == 23
    cfg = CS.ctc_config(ParakeetEncoderConfig(), n, ffn)
    with torch.device("meta"):
        m = ParakeetForCTC(cfg)
    c = CS.param_counts(m)
    assert c["total"] == c["closed_form"] == CS.closed_form_ctc_params(n, ffn) == total
    assert c["non_embedding"] == non_emb and c["ctc_head"] == 3_149_825 and c["subsampling"] == 2_761_728


def test_resolve_layers():
    assert CS.resolve_layers("all") == CS.resolve_layers(["all"]) == list(range(24))
    assert CS.resolve_layers(["0", "11", "23"]) == [0, 11, 23]
    assert CS.resolve_layers(2, 4) == [0, 3]


def test_ctc_config_zeroes_dropout_and_uses_sdpa():
    cfg = tiny_ctc_config()
    e = cfg.encoder_config
    assert all(getattr(e, k) == 0.0 for k in CS.DROPOUT_KEYS)
    assert cfg._attn_implementation == e._attn_implementation == "sdpa"
    assert (cfg.vocab_size, cfg.pad_token_id, cfg.ctc_loss_reduction) == (V, BLANK, "sum")
    assert e.num_key_value_heads == e.num_attention_heads == 2 and e.scale_input
    m = tiny_ctc_model()
    assert CS.param_counts(m)["total"] == CS.config_closed_form(m.config)


# ------------------------------------------------------------------------------------------------ load + build


def test_load_parakeet_ctc_reads_the_converted_dir(tmp_path):
    d, ref = tiny_parakeet_dir(tmp_path / "teacher")
    m = CS.load_parakeet_ctc(d)
    assert not m.training and m.config._attn_implementation == "sdpa"
    sd, want = m.state_dict(), ref.state_dict()
    assert set(sd) == set(want) and all(torch.equal(sd[k], want[k]) for k in sd)
    assert m.ctc_head.weight.shape == (V, 32, 1)
    assert all(k.startswith(("encoder.", "ctc_head.")) for k in sd)  # the TDT decoder/joint are not read


def test_unpruned_build_reproduces_the_teacher(tmp_path):
    d, _ = tiny_parakeet_dir(tmp_path / "teacher", seed=1)
    teacher = CS.load_parakeet_ctc(d)
    student = CS.build_ctc_student(teacher, list(range(4)), 64, None)
    assert all(torch.equal(a, b) for a, b in zip(student.state_dict().values(), teacher.state_dict().values()))
    _, feats = hf_features()
    f, fl = feats(waves_of([16000, 7000, 31000]))
    with torch.no_grad():
        a, na = CS.ctc_log_probs(teacher, f, CS.lengths_to_mask(fl, f.shape[1]))
        b, nb = CS.ctc_log_probs(student, f, CS.lengths_to_mask(fl, f.shape[1]))
    assert torch.equal(na, nb) and float((a - b).abs().max()) == 0.0


def expected_tensor(teacher_sd: dict, key: str, layers: list[int], keep: dict) -> torch.Tensor:
    """What student tensor `key` must equal, spelled out."""
    parts = key.split(".")
    if parts[:2] != ["encoder", "layers"]:
        return teacher_sd[key]
    tl = layers[int(parts[2])]
    rest = ".".join(parts[3:])
    src = teacher_sd[f"encoder.layers.{tl}.{rest}"]
    if parts[3].startswith("feed_forward") and (tl, parts[3]) in keep:
        idx = keep[(tl, parts[3])]
        if parts[4] == "linear1":
            return src[idx]
        if parts[5] == "weight":
            return src[:, idx]
    return src


def test_pruned_build_tensors_and_count(tmp_path):
    d, _ = tiny_parakeet_dir(tmp_path / "teacher", seed=2)
    teacher = CS.load_parakeet_ctc(d)
    layers = [0, 2, 3]
    imp = rand_importance(range(4), 64, seed=3)
    st = CS.build_ctc_student(teacher, layers, 40, imp)
    keep = S.select_ffn_neurons(imp, layers, 40, 64)
    tsd = teacher.state_dict()
    for k, v in st.state_dict().items():
        assert torch.equal(v, expected_tensor(tsd, k, layers, keep)), k
    for i, tl in enumerate(layers):  # BN: the teacher's running stats, verbatim
        assert torch.equal(st.encoder.layers[i].conv.norm.running_var, teacher.encoder.layers[tl].conv.norm.running_var)
        assert st.encoder.layers[i].self_attn.layer_idx == i
    c = CS.param_counts(st)
    assert c["total"] == c["closed_form"] == CS.closed_form_ctc_params(3, 40, d=32, channels=8)
    assert st.config.encoder_config.num_hidden_layers == 3 and st.config.encoder_config.intermediate_size == 40
    assert all(p.dtype == torch.float32 for p in st.parameters()) and not st.training
    kept = CS.kept_neurons(imp, layers, 40, 64)
    assert sorted(kept) == [f"{l}.{n}" for l in layers for n in S.FFN_NAMES]


def test_build_rejects_bad_requests(tmp_path):
    d, _ = tiny_parakeet_dir(tmp_path / "teacher")
    teacher = CS.load_parakeet_ctc(d)
    imp = rand_importance(range(4), 64)
    with pytest.raises(ValueError, match="last teacher layer"):
        CS.build_ctc_student(teacher, [0, 1, 2], 64, None)
    with pytest.raises(ValueError, match="ascending"):
        CS.build_ctc_student(teacher, [3, 0], 64, None)
    with pytest.raises(ValueError, match="ffn"):
        CS.build_ctc_student(teacher, [0, 3], 65, imp)
    with pytest.raises(ValueError, match="importance"):
        CS.build_ctc_student(teacher, [0, 3], 32, None)


def test_pruning_dead_neurons_keeps_the_function(tmp_path):
    """Neurons that never fire (zero linear1 row and bias) get zero importance, so pruning exactly them leaves the
    output unchanged: importance, selection and slicing agree with each other."""
    d, _ = tiny_parakeet_dir(tmp_path / "teacher", seed=4)
    teacher = CS.load_parakeet_ctc(d)
    g = torch.Generator().manual_seed(0)
    dead = {(l, n): torch.randperm(64, generator=g)[:24] for l in range(4) for n in S.FFN_NAMES}
    with torch.no_grad():
        for (l, n), idx in dead.items():
            lin1 = getattr(teacher.encoder.layers[l], n).linear1
            lin1.weight[idx] = 0.0
            lin1.bias[idx] = -30.0  # silu(-30) ~ -3e-12: dead, and |act| ~ 0 in the importance
    _, feats = hf_features()
    f, fl = feats(waves_of([12000, 20000, 9000], seed=5))
    batches = [(f, CS.lengths_to_mask(fl, f.shape[1]))]
    imp = CS.ffn_importance_ctc(teacher, batches, range(4))
    assert not teacher.training  # the view restored the encoder's own flag, not its own default (train)
    for key, idx in dead.items():
        assert float(imp[key][idx].max()) < 1e-9 < float(imp[key].max())
    st = CS.build_ctc_student(teacher, list(range(4)), 40, imp)
    mask = CS.lengths_to_mask(fl, f.shape[1])
    with torch.no_grad():
        a, _ = CS.ctc_log_probs(teacher, f, mask)
        b, _ = CS.ctc_log_probs(st, f, mask)
    assert float((a - b).abs().max()) < 1e-4


def test_ffn_importance_matches_an_unpadded_reference(tmp_path):
    d, _ = tiny_parakeet_dir(tmp_path / "teacher", seed=6)
    teacher = CS.load_parakeet_ctc(d)
    _, feats = hf_features()
    waves = waves_of([16000, 5000, 27000], seed=6)
    f, fl = feats(waves)
    batched = CS.ffn_importance_ctc(teacher, [(f, CS.lengths_to_mask(fl, f.shape[1]))], [1, 3])
    sums, frames = {}, 0
    for w in waves:  # one utterance at a time: plain mean over its valid frames
        fi, li = feats([w])
        acts = {}
        hooks = [getattr(teacher.encoder.layers[l], n).linear1.register_forward_hook(
            lambda m, i, o, key=(l, n): acts.__setitem__(key, torch.nn.functional.silu(o).abs()[0]))
            for l in (1, 3) for n in S.FFN_NAMES]
        with torch.no_grad():
            out = teacher.encoder(input_features=fi, attention_mask=CS.lengths_to_mask(li, fi.shape[1]).long())
        for h in hooks:
            h.remove()
        n_valid = int(out.attention_mask.sum())  # the extractor's last (partial) mel frame can add one encoder frame
        assert n_valid == CS.expected_n_frames(len(w))
        for key, a in acts.items():
            sums[key] = sums.get(key, 0) + a[:n_valid].double().sum(0)
        frames += n_valid
    for key in sums:
        assert torch.allclose(batched[key].double(), sums[key] / frames, rtol=1e-4, atol=1e-6), key


# ------------------------------------------------------------------------------------------------ save / load


def test_save_load_roundtrip(tmp_path):
    from safetensors import safe_open
    from transformers import AutoProcessor

    out, m = tiny_student_dir(tmp_path / "student", seed=7, name="P-tiny", enc_layers=[0, 1, 3])
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "model.safetensors", "generation_config.json", "processor_config.json", "tokenizer.json",
            "tokenizer_config.json", "MODEL_CARD.md", "README.md", "student_meta.json"} <= names
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["dtype"] == "bfloat16" and cfg["vocab_size"] == V and cfg["pad_token_id"] == BLANK
    assert cfg["architectures"] == ["ParakeetForCTC"]
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        dt = {k: f.get_slice(k).get_dtype() for k in f.keys()}
    assert all(v == "F32" for k, v in dt.items() if k.endswith(("running_mean", "running_var")))
    assert all(v == "BF16" for k, v in dt.items() if k.endswith((".weight", ".bias", "bias_u", "bias_v")))
    assert dt["encoder.layers.0.conv.norm.num_batches_tracked"] == "I64"
    card = (out / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert card == (out / "README.md").read_text(encoding="utf-8")
    for s in ("license: cc-by-4.0", "base_model: nvidia/parakeet-tdt_ctc-0.6b-ja", "by **NVIDIA**",
              "creativecommons.org/licenses/by/4.0", "Changes made", "P-tiny"):
        assert s in card, s
    meta = CS.load_meta(out)
    assert meta["family"] == "ctc" and meta["params_total"] == meta["closed_form_params"]
    back = CS.load_ctc_student(out, "cpu")
    assert not back.training and back.config._attn_implementation == "sdpa" == back.encoder.config._attn_implementation
    sd = m.state_dict()
    for k, v in back.state_dict().items():
        if k.endswith(("running_mean", "running_var", "num_batches_tracked")):
            assert torch.equal(v, sd[k]), k
        else:
            assert torch.equal(v, sd[k].to(torch.bfloat16).float()), k
    proc = AutoProcessor.from_pretrained(str(out))
    assert proc.decoder_type == "ctc"
    # HF's own CTC API on the saved files gives the same text as our explicit greedy + decode_ids
    fe, feats = hf_features()
    waves = waves_of([16000, 9000], seed=8)
    inp = proc(waves, sampling_rate=16000)
    f, fl = feats(waves)
    assert torch.equal(inp["input_features"], f)
    with torch.no_grad():
        ids = back.generate(input_features=inp["input_features"], attention_mask=inp["attention_mask"])
        lp, n = CS.ctc_log_probs(back, f, CS.lengths_to_mask(fl, f.shape[1]))
    ours = CS.decode_batch(proc.tokenizer, CS.greedy_ctc_ids(lp, n))
    assert proc.batch_decode(ids, skip_special_tokens=True) == ours


def test_ctc_features_is_the_parakeet_extractor(tmp_path):
    from transformers import ParakeetFeatureExtractor

    from kitsune.features import LogMel

    lm = LogMel.from_feature_extractor(ParakeetFeatureExtractor())
    assert lm.dither == 0.0 and lm.mel_filters.shape == (80, 257)  # the extractor has no `dither` attribute
    d = write_processor(tmp_path / "proc")
    feats = CS.ctc_features(d)
    fe = ParakeetFeatureExtractor.from_pretrained(str(d))
    waves = waves_of([16000, 4321, 30 * 16000, 160 * 7 + 3], seed=9)
    f, fl = feats(waves)
    ref = fe(waves, sampling_rate=16000, return_tensors="pt")
    assert f.shape == ref["input_features"].shape and torch.equal(fl, ref["attention_mask"].sum(1))
    assert torch.equal(f, ref["input_features"])  # bitwise on CPU


def test_expected_n_frames_equals_the_encoders_own_length():
    """Over many audio lengths (edge cases around multiples of 160 and 1280 samples, up to 30 s): expected_n_frames
    == the encoder's output mask length, batched and alone."""
    m = tiny_ctc_model(seed=3, n_layers=1)
    _, feats = hf_features()
    lengths = sorted({*[1280 * k + r for k in (1, 2, 3, 17) for r in (-161, -160, -1, 0, 1, 159, 160)],
                      400, 16000, 16001, 99_999, 212_345, 30 * 16000})
    for i in range(0, len(lengths), 9):
        chunk = lengths[i:i + 9]
        f, fl = feats(waves_of(chunk, seed=i))
        with torch.no_grad():
            out = m.encoder(input_features=f, attention_mask=CS.lengths_to_mask(fl, f.shape[1]).long())
        got = out.attention_mask.sum(-1).tolist()
        assert got == [CS.expected_n_frames(n) for n in chunk], chunk
        assert got == [-(-(n // 160) // 8) for n in chunk]  # ceil(mel frames / 8)
    assert CS.expected_n_frames(0) == 0 and CS.expected_n_frames(159) == 0 and CS.expected_n_frames(160) == 1


def test_ctc_log_probs_is_normalised_and_padding_invariant():
    m = tiny_ctc_model(seed=4)
    _, feats = hf_features()
    waves = waves_of([23000, 8000, 15000], seed=10)
    f, fl = feats(waves)
    with torch.no_grad():
        lp, n = CS.ctc_log_probs(m, f, CS.lengths_to_mask(fl, f.shape[1]))
        ref = m(input_features=f, attention_mask=CS.lengths_to_mask(fl, f.shape[1]).long()).logits.log_softmax(-1)
    assert lp.dtype == torch.float32 and n.tolist() == [CS.expected_n_frames(len(w)) for w in waves]
    assert torch.allclose(lp.exp().sum(-1), torch.ones(()), atol=1e-5)
    assert torch.allclose(lp, ref, atol=1e-5)
    for b, w in enumerate(waves):
        fi, li = feats([w])
        with torch.no_grad():
            alone, k = CS.ctc_log_probs(m, fi, CS.lengths_to_mask(li, fi.shape[1]))
        assert int(k[0]) == int(n[b]) and torch.allclose(alone[0], lp[b, :int(n[b])], atol=1e-4)


def test_train_mode_with_frozen_bn_equals_eval_and_the_relpos_patch_applies():
    """Every dropout and layerdrop is 0 and BN is frozen by kitsune.patches.train_mode, so a training forward equals
    the eval forward; the trainer's rel-pos-once-per-batch patch finds ParakeetForCTC's encoder and keeps the
    log-probs and the gradients."""
    from kitsune.patches import assert_bn_frozen, patch_relpos_once_per_batch, train_mode

    m = tiny_ctc_model(seed=12)
    _, feats = hf_features()
    f, fl = feats(waves_of([20000, 11000, 16000], seed=12))
    mask = CS.lengths_to_mask(fl, f.shape[1])
    with torch.no_grad():
        ref, _ = CS.ctc_log_probs(m, f, mask)
    train_mode(m)
    assert m.training and assert_bn_frozen(m) == 4
    lp, n = CS.ctc_log_probs(m, f, mask)
    assert torch.allclose(lp, ref, atol=1e-5)  # not bitwise: sdpa picks another kernel when autograd records
    loss = lp[CS.lengths_to_mask(n, lp.shape[1])][:, :100].sum()
    loss.backward()
    g_ref = {k: p.grad.clone() for k, p in m.named_parameters()}
    m.zero_grad()
    unpatch = patch_relpos_once_per_batch(m)
    lp2, _ = CS.ctc_log_probs(m, f, mask)
    assert torch.allclose(lp2, ref, atol=1e-6)
    lp2[CS.lengths_to_mask(n, lp.shape[1])][:, :100].sum().backward()
    for k, p in m.named_parameters():
        assert torch.allclose(p.grad, g_ref[k], rtol=1e-4, atol=1e-6), k
    unpatch()


# ------------------------------------------------------------------------------------------------ decoding


def test_greedy_ctc_ids_equals_ctc_collapse():
    g = torch.Generator().manual_seed(0)
    lp = torch.log_softmax(torch.randn(5, 40, 9, generator=g) * 3, -1)
    lp[0, :, 8] += 5  # mostly blank
    lp[1, 10:14] = torch.log_softmax(torch.tensor([0, 5.0, 0, 0, 0, 0, 0, 0, 0]), -1)  # a run of one token
    lengths = torch.tensor([40, 31, 1, 0, 17])
    got = CS.greedy_ctc_ids(lp, lengths, blank=8)
    for b, n in enumerate(lengths.tolist()):
        assert got[b] == pk.ctc_collapse(lp[b, :n].argmax(-1).tolist(), 8)
    assert got[3] == []
    x = torch.full((1, 6, 4), -9.0)
    for t, c in enumerate([1, 1, 3, 1, 2, 2]):  # blank = 3 separates the repeated 1
        x[0, t, c] = 0.0
    assert CS.greedy_ctc_ids(x, torch.tensor([6]), blank=3) == [[1, 1, 2]]


def test_decode_ids_never_groups_repeats(tmp_path):
    tok = tiny_tokenizer()
    ids = [10, 10, 11]  # already collapsed: the repeat is real (a blank separated it)
    assert tok.decode(ids, skip_special_tokens=True) != CS.decode_ids(tok, ids)  # the tokenizer's default groups
    assert CS.decode_ids(tok, ids) == tok.decode(ids, skip_special_tokens=True, group_tokens=False)
    from transformers import AutoProcessor

    for dt in ("ctc", "tdt"):
        proc = AutoProcessor.from_pretrained(str(write_processor(tmp_path / dt, decoder_type=dt)))
        assert CS.decode_ids(proc, ids) == CS.decode_ids(tok, ids)
    assert CS.decode_ids(tok, [BLANK, 10, 0]) == CS.decode_ids(tok, [10])  # special tokens are skipped


# ------------------------------------------------------------------------------------------------ 03c


@pytest.fixture
def build_env(tmp_path, monkeypatch):
    """A fake corpus (train source + the three gate sets), its selection, a tiny teacher dir with patched pins."""
    from fixtures import load_script, make_fake_corpus, make_fake_selection

    fc = make_fake_corpus(tmp_path / "corpus", sources={"src_a": (40, "train"), "eval_jsut": (8, "eval"),
                                                        "eval_reazon": (8, "eval"), "eval_cv8": (8, "eval")})
    sel = make_fake_selection(fc, greedy_n=4, probe_n=4)
    d, _ = tiny_parakeet_dir(tmp_path / "teacher", seed=11)
    monkeypatch.setattr(pk, "PARAKEET_FILES", {p.name: pk.file_sha256(p) for p in d.iterdir() if p.is_file()})
    mod = load_script("03c_build_ctc_student")
    base = ["--model-dir", str(d), "--selection", str(sel), "--data", str(fc.data), "--sources", "src_a",
            "--calib-utts", "12", "--calib-per-group", "4", "--calib-batch-s", "30", "--gate-utts", "3",
            "--threads", "2"]
    return mod, base, tmp_path


def test_build_script_end_to_end_tiny(build_env):
    mod, base, tmp = build_env
    imp = tmp / "importance.pt"
    anchor = tmp / "anchor"
    assert mod.main(base + ["--enc-layers", "all", "--ffn", "64", "--out", str(anchor)]) == 0
    m = CS.load_meta(anchor)
    assert m["stage"] == "complete" and m["init_class"] == "pruned_kept"
    built = m["step0"]["built"]
    assert built["max_abs_diff_log_probs"] == 0.0 and built["frame_kl"] == 0.0 and built["cer_vs_teacher"] == 0.0
    assert m["importance"]["skipped"] and m["enc_layers"] == [0, 1, 2, 3] and m["ffn"] == 64
    assert len(m["step0"]["gate"]["ids"]) == 9 and m["step0"]["saved"]["n"] == 9

    out = tmp / "p_small"
    assert mod.main(base + ["--enc-layers", "2", "--ffn", "40", "--name", "P-x", "--out", str(out),
                            "--importance", str(imp)]) == 0
    m = CS.load_meta(out)
    for key in ("family", "init_class", "params_total", "params_non_embedding", "closed_form_params", "seed", "bn",
                "teacher", "enc_layers", "ffn"):  # CONTRACT.md section 1
        assert m.get(key) is not None, key
    assert m["family"] == "ctc" and m["bn"] == "teacher" and m["seed"] == 1234 and m["name"] == "P-x"
    assert m["teacher"] == f"{pk.NEMO_REPO}@{pk.NEMO_REVISION}"
    assert m["enc_layers"] == [0, 3] and m["ffn"] == 40 and m["init_class"] in ("pruned_kept", "pruned_lost")
    assert m["params_total"] == m["closed_form_params"] == CS.closed_form_ctc_params(2, 40, d=32, channels=8)
    assert not m["importance"]["reused"] and m["calibration"]["n"] == 12 and imp.exists()
    cache = torch.load(imp, weights_only=True)
    assert len(cache["ids"]) == 12 and cache["key"]["ids_sha256"] == m["calibration"]["ids_sha256"]
    assert cache["layers"] == [0, 1, 2, 3]  # all teacher layers, whatever this student keeps
    assert sorted(m["kept"]["ffn"]) == [f"{l}.{n}" for l in (0, 3) for n in S.FFN_NAMES]
    assert all(len(v) == 40 for v in m["kept"]["ffn"].values())
    s = m["step0"]["saved"]
    assert 0 <= s["cer_vs_teacher"] and s["frames"] > 0 and "kd_objective" in s and "argmax_agree" in s
    st = CS.load_ctc_student(out, "cpu")
    assert st.config.encoder_config.num_hidden_layers == 2 and st.config.encoder_config.intermediate_size == 40

    assert mod.main(base + ["--enc-layers", "2", "--ffn", "40", "--out", str(out)]) == 0  # complete: nothing to do
    out2 = tmp / "p_smaller"
    assert mod.main(base + ["--enc-layers", "0", "3", "--ffn", "24", "--out", str(out2), "--importance", str(imp),
                            "--gate-utts", "0", "--init-class", "pruned_lost"]) == 0
    m2 = CS.load_meta(out2)
    assert m2["importance"]["reused"] and m2["init_class"] == "pruned_lost" and m2["step0"] == {"skipped": True}
    assert m2["calibration"]["ids_sha256"] == m["calibration"]["ids_sha256"]
    with pytest.raises(SystemExit):
        mod.main(base + ["--enc-layers", "2", "--ffn", "40", "--out", str(tmp / "x"), "--expect-calib-sha", "0" * 64])


def test_build_script_golden_runs_on_the_built_and_the_saved_student(build_env, monkeypatch):
    """--golden checks the fp32 build (the reproduction claim) and the bf16 copy (the storage rounding) separately.
    The golden rows themselves are real JSUT audio, so the reader is stubbed here (tests/test_ctc_real.py runs it)."""
    mod, base, tmp = build_env
    seen = []

    def fake_golden(model, feats, tokenizer, data_root):
        seen.append((model, next(model.parameters()).dtype))
        return dict(n=32, match=32 - len(seen) + 1, mismatches=[])

    monkeypatch.setattr(mod, "golden_check", fake_golden)
    assert mod.main(base + ["--enc-layers", "all", "--ffn", "64", "--golden", "--gate-utts", "1",
                            "--out", str(tmp / "anchor")]) == 0
    g = CS.load_meta(tmp / "anchor")["golden"]
    assert g["built"]["match"] == 32 and g["saved"]["match"] == 31
    assert len(seen) == 2 and seen[0][0] is not seen[1][0]


def test_build_script_refuses_an_unpinned_dir_and_needs_an_init_class(build_env, monkeypatch):
    mod, base, tmp = build_env
    monkeypatch.setattr(pk, "PARAKEET_FILES", {"model.safetensors": "0" * 64})
    with pytest.raises(SystemExit, match="not the pinned"):
        mod.main(base + ["--out", str(tmp / "a")])
    with pytest.raises(SystemExit):
        mod.parse_args(["--out", str(tmp / "a"), "--gate-utts", "0"])


def test_build_script_help_is_light():
    """--help works without importing torch/transformers."""
    import subprocess

    script = ROOT / "scripts" / "03c_build_ctc_student.py"
    code = ("import runpy, sys\n"
            "sys.argv = ['03c_build_ctc_student.py', '--help']\n"
            "try:\n"
            f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    assert e.code in (0, None), e.code\n"
            "print('HEAVY' if {'torch', 'transformers'} & set(sys.modules) else 'LIGHT')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, encoding="utf-8",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode == 0, r.stderr
    assert "--enc-layers" in r.stdout and "--golden" in r.stdout
    assert r.stdout.strip().endswith("LIGHT")
