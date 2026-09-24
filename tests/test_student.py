"""Student construction (kitsune/student.py): layer choice, FFN pruning, strict remap, BN recalibration, save/load.

Everything runs on CPU on a TINY random CohereAsr model built from the teacher's own config (shrunk). The one test
that loads the real 2B teacher is marked `slow`."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the teacher config/weights come from the local HF cache or not at all

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig, CohereAsrConfig, CohereAsrForConditionalGeneration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitsune import student as S  # noqa: E402

PROMPT = [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13]
N_MEL = 128


def tiny_config(n_enc: int = 4, n_dec: int = 3, d: int = 64, ffn: int = 128) -> CohereAsrConfig:
    """The real teacher config (NeMo extras and all) at toy sizes; CohereAsrConfig defaults (== the teacher's HF
    fields) when the teacher is not in the local cache. Full 16384 vocab, so the real prompt ids are valid."""
    try:
        base = AutoConfig.from_pretrained(S.TEACHER_ID, local_files_only=True)
    except Exception:
        base = CohereAsrConfig()
    cfg = base.to_dict()
    enc = dict(cfg["encoder_config"], num_hidden_layers=n_enc, hidden_size=d, intermediate_size=ffn,
               num_attention_heads=2, num_key_value_heads=2, subsampling_conv_channels=8)
    cfg.update(encoder_config=enc, num_hidden_layers=n_dec, hidden_size=d, intermediate_size=2 * d,
               num_attention_heads=2, num_key_value_heads=2, head_dim=d // 2)
    cfg = CohereAsrConfig.from_dict(cfg)
    cfg._attn_implementation = "sdpa"
    cfg.encoder_config._attn_implementation = "sdpa"
    return cfg


def tiny_teacher(seed: int = 0, **kw) -> CohereAsrForConditionalGeneration:
    torch.manual_seed(seed)
    m = CohereAsrForConditionalGeneration(tiny_config(**kw)).eval()
    assert m.proj_out.weight is not m.model.decoder.embed_tokens.weight  # untied, like the real teacher
    with torch.no_grad():
        for bn in (x for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d)):
            bn.running_mean.normal_(0.0, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
            bn.num_batches_tracked.fill_(276000)
    return m


def feat_batches(lengths: list[list[int]], seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Random normalised-log-mel-like batches in LogMel's layout: (B,T,128) float32, zero past each length, bool mask."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for lens in lengths:
        T = max(lens)
        mask = torch.arange(T)[None, :] < torch.tensor(lens)[:, None]
        out.append((torch.randn(len(lens), T, N_MEL, generator=g) * mask[..., None], mask))
    return out


def rand_importance(layers, width, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {(l, n): torch.rand(width, generator=g) for l in layers for n in S.FFN_NAMES}


def expected_tensor(teacher_sd: dict, key: str, spec: S.StudentSpec, keep: dict) -> torch.Tensor:
    """What the student tensor `key` must equal, spelled out case by case."""
    parts = key.split(".")
    if key.startswith("model.encoder.layers."):
        tl = spec.enc_layers[int(parts[3])]
        rest = ".".join(parts[4:])
        src = teacher_sd[f"model.encoder.layers.{tl}.{rest}"]
        if parts[4] in S.FFN_NAMES and (tl, parts[4]) in keep:
            idx = keep[(tl, parts[4])]
            if rest.endswith("linear1.weight"):
                return src[idx, :]
            if rest.endswith("linear1.bias"):
                return src[idx]
            if rest.endswith("linear2.weight"):
                return src[:, idx]
        return src
    if key.startswith("model.decoder.layers."):
        return teacher_sd[f"model.decoder.layers.{spec.dec_layers[int(parts[3])]}.{'.'.join(parts[4:])}"]
    if key == "proj_out.weight" and spec.tie_head:
        return teacher_sd["model.decoder.embed_tokens.weight"]
    return teacher_sd[key]


def teacher_forced_logits(model, feats, mask, n_dec: int = 5) -> torch.Tensor:
    B = feats.shape[0]
    dec = torch.tensor([PROMPT[:n_dec]] * B)
    with torch.no_grad():
        h = model.model(input_features=feats, attention_mask=mask.long(), decoder_input_ids=dec, use_cache=False)
        return model.proj_out(h.last_hidden_state)


# ------------------------------------------------------------------------------------------------ layer choice


def test_evenly_spaced():
    assert S.evenly_spaced(20, 48) == [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47]
    assert S.evenly_spaced(4, 8) == [0, 2, 5, 7]
    assert S.evenly_spaced(4, 48) == [0, 16, 31, 47]  # the laptop smoke student
    assert S.evenly_spaced(1, 48) == [0]
    assert S.evenly_spaced(48, 48) == list(range(48))
    for n in range(1, 49):  # strictly increasing, both ends kept
        v = S.evenly_spaced(n, 48)
        assert len(set(v)) == n and v == sorted(v) and v[0] == 0 and (n == 1 or v[-1] == 47)
    with pytest.raises(ValueError):
        S.evenly_spaced(49, 48)
    with pytest.raises(ValueError):
        S.evenly_spaced(0, 48)


def test_default_spec_and_real_param_count_on_meta():
    spec = S.default_spec()
    assert spec.enc_layers == S.evenly_spaced(20, 48) and spec.ffn_dim == 2560
    assert spec.dec_layers == [0, 2, 5, 7] and spec.tie_head
    cfg = S.student_config(CohereAsrConfig(), spec)  # CohereAsrConfig() defaults are the teacher's dims
    with torch.device("meta"):
        m = CohereAsrForConditionalGeneration(cfg)
    rep = S.param_report(m)
    assert rep["total"] == rep["closed_form"] == S.EXPECTED_DEFAULT_PARAMS == 616_963_328
    assert rep["encoder_layer"]["total"] == 13_149_440 + 5_122 * 2560
    assert rep["decoder_layer"]["total"] == 16_796_672 and rep["proj_out"] == 16384
    with torch.device("meta"):  # and the teacher reproduces report_model.md exactly
        t = CohereAsrForConditionalGeneration(CohereAsrConfig())
    assert S.param_report(t)["total"] == S.closed_form_params(t.config) == 2_065_647_872


# ------------------------------------------------------------------------------------------------ build


def test_build_student_tiny():
    teacher = tiny_teacher()
    spec = S.StudentSpec(enc_layers=S.evenly_spaced(3, 4), ffn_dim=48, dec_layers=[0, 2])
    assert spec.enc_layers == [0, 2, 3]
    imp = rand_importance(spec.enc_layers, 128)
    student = S.build_student(teacher, spec, imp)

    cfg = student.config
    assert cfg.encoder_config.num_hidden_layers == 3 and cfg.encoder_config.intermediate_size == 48
    assert cfg.num_hidden_layers == 2 and cfg.tie_word_embeddings and cfg.decoder_start_token_id == 13764
    assert cfg._attn_implementation == "sdpa" and student.model.encoder.config._attn_implementation == "sdpa"
    assert not hasattr(cfg, "transf_decoder") and not hasattr(cfg, "encoder")  # stale NeMo shape keys dropped
    assert not student.training
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in student.parameters())

    # every tensor (params AND buffers, incl. BN stats) is exactly the teacher's tensor or its sorted-index slice
    keep = S.select_ffn_neurons(imp, spec.enc_layers, 48, 128)
    for (l, n), idx in keep.items():
        assert torch.equal(idx, torch.topk(imp[(l, n)], 48).indices.sort().values)
    tsd, ssd = teacher.state_dict(), student.state_dict()
    for k, v in ssd.items():
        exp = expected_tensor(tsd, k, spec, keep)
        assert v.shape == exp.shape and torch.equal(v, exp.to(v.dtype)), k
    ff = student.model.encoder.layers[1].feed_forward2  # student layer 1 <- teacher layer 2
    idx = keep[(2, "feed_forward2")]
    assert torch.equal(ff.linear1.weight, teacher.model.encoder.layers[2].feed_forward2.linear1.weight[idx])
    assert torch.equal(ff.linear2.weight, teacher.model.encoder.layers[2].feed_forward2.linear2.weight[:, idx])
    assert torch.equal(ff.linear2.bias, teacher.model.encoder.layers[2].feed_forward2.linear2.bias)
    bn_s, bn_t = student.model.encoder.layers[2].conv.norm, teacher.model.encoder.layers[3].conv.norm
    assert torch.equal(bn_s.running_var, bn_t.running_var) and int(bn_s.num_batches_tracked) == 276000
    assert bn_s.running_var.data_ptr() != bn_t.running_var.data_ptr()  # copied, not shared

    # layer_idx renumbered 0..n-1 (the decoder's index is its KV-cache slot)
    for j, layer in enumerate(student.model.decoder.layers):
        assert layer.self_attn.layer_idx == j and layer.encoder_attn.layer_idx == j
    for j, layer in enumerate(student.model.encoder.layers):
        assert layer.self_attn.layer_idx == j

    # tied head: one storage; the bias is the teacher's
    assert student.proj_out.weight is student.model.decoder.embed_tokens.weight
    assert student.proj_out.weight.data_ptr() == student.model.decoder.embed_tokens.weight.data_ptr()
    assert torch.equal(student.proj_out.bias, teacher.proj_out.bias)

    # generate() through a 2-layer cache: a stale layer_idx 2 would IndexError here
    feats, mask = feat_batches([[160, 97]])[0]
    out = student.generate(input_features=feats, attention_mask=mask.long(), decoder_input_ids=torch.tensor([PROMPT] * 2),
                           max_new_tokens=3, min_new_tokens=3, do_sample=False, num_beams=1)
    assert out.shape == (2, len(PROMPT) + 3)

    rep = S.param_report(student)
    assert rep["total"] == rep["closed_form"] == sum(p.numel() for p in student.parameters())
    assert rep["tied_head"] and rep["proj_out"] == 16384
    assert rep["encoder_layers"] == 3 and rep["decoder_layers"] == 2


def test_build_is_exact_when_nothing_is_dropped():
    """Keep every layer and neuron (FFN rows permuted by nothing) and tie a head that the teacher already has equal
    to its embedding: the student must compute the teacher's function."""
    teacher = tiny_teacher(seed=1)
    with torch.no_grad():
        teacher.proj_out.weight.copy_(teacher.model.decoder.embed_tokens.weight)  # as in the real checkpoint
    spec = S.StudentSpec(enc_layers=[0, 1, 2, 3], ffn_dim=128, dec_layers=[0, 1, 2])
    student = S.build_student(teacher, spec, importance=None)  # no pruning -> importance not needed
    feats, mask = feat_batches([[150, 90, 40]], seed=3)[0]
    torch.testing.assert_close(teacher_forced_logits(student, feats, mask), teacher_forced_logits(teacher, feats, mask),
                               rtol=0, atol=0)


def test_pruning_dead_neurons_preserves_function():
    """Neurons whose linear2 column is zero contribute nothing. Rank them last and prune exactly them: the output must
    not change. This catches any mismatch between the linear1 rows and linear2 columns that are kept."""
    teacher = tiny_teacher(seed=2)
    g = torch.Generator().manual_seed(5)
    imp = {}
    with torch.no_grad():
        for l, layer in enumerate(teacher.model.encoder.layers):
            for n in S.FFN_NAMES:
                dead = torch.randperm(128, generator=g)[:80]
                getattr(layer, n).linear2.weight[:, dead] = 0.0
                imp[(l, n)] = torch.rand(128, generator=g) + 1.0
                imp[(l, n)][dead] = 0.0
    spec = S.StudentSpec(enc_layers=[0, 1, 2, 3], ffn_dim=48, dec_layers=[0, 1, 2], tie_head=False)
    student = S.build_student(teacher, spec, imp)
    assert student.model.encoder.layers[0].feed_forward1.linear1.weight.shape == (48, 64)
    feats, mask = feat_batches([[150, 90]], seed=4)[0]
    torch.testing.assert_close(teacher_forced_logits(student, feats, mask), teacher_forced_logits(teacher, feats, mask),
                               rtol=1e-5, atol=1e-6)


def test_build_from_bf16_teacher_gives_fp32_student(tmp_path):
    """from_pretrained(dtype=bf16) leaves dtype=bf16 on the encoder sub-config too; the student must still be fp32."""
    tiny_teacher(seed=7).save_pretrained(tmp_path)
    teacher = CohereAsrForConditionalGeneration.from_pretrained(tmp_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    assert teacher.config.encoder_config.dtype == torch.bfloat16
    spec = S.StudentSpec([0, 3], 64, [0, 2])
    student = S.build_student(teacher, spec, rand_importance([0, 3], 128))
    assert {p.dtype for p in student.parameters()} == {torch.float32}
    assert {b.dtype for n, b in student.named_buffers() if ".running_" in n} == {torch.float32}
    assert torch.equal(student.model.encoder.subsampling.linear.weight,
                       teacher.model.encoder.subsampling.linear.weight.float())


def test_build_rejects_bad_requests():
    teacher = tiny_teacher()
    with pytest.raises(ValueError, match="importance"):
        S.build_student(teacher, S.StudentSpec([0, 3], 48, [0]), importance=None)
    with pytest.raises(ValueError):
        S.build_student(teacher, S.StudentSpec([0, 4], 48, [0]), importance=None)  # teacher has layers 0..3
    with pytest.raises(ValueError):
        S.build_student(teacher, S.StudentSpec([0, 3], 256, [0]), importance=None)  # wider than the teacher
    with pytest.raises(ValueError):
        S.StudentSpec([3, 0], 48, [0])  # not ascending


# ------------------------------------------------------------------------------------------------ calibration


def test_ffn_importance_matches_unpadded_reference():
    teacher = tiny_teacher(seed=3)
    batches = feat_batches([[160, 120, 97], [200, 64]], seed=1)  # padded batches
    layers = [0, 2]
    imp = S.ffn_importance(teacher, batches, layers, device="cpu")
    assert set(imp) == {(l, n) for l in layers for n in S.FFN_NAMES}
    assert all(v.shape == (128,) and v.dtype == torch.float32 and v.device.type == "cpu" for v in imp.values())

    # reference: each utterance alone and unpadded; act(linear1(LN(x))) recomputed by hand from the FFN inputs
    sums = {k: torch.zeros(128, dtype=torch.float64) for k in imp}
    frames = 0
    enc = teacher.model.encoder
    captured = {}
    hooks = [getattr(enc.layers[l], f"norm_{n}").register_forward_hook(
        lambda _m, _i, out, key=(l, n): captured.__setitem__(key, out)) for l in layers for n in S.FFN_NAMES]
    with torch.no_grad():
        for feats, mask in batches:
            for i in range(feats.shape[0]):
                L = int(mask[i].sum())
                enc(input_features=feats[i:i + 1, :L], attention_mask=torch.ones(1, L, dtype=torch.long))
                for (l, n), x in captured.items():
                    ff = getattr(enc.layers[l], n)
                    sums[(l, n)] += torch.nn.functional.silu(ff.linear1(x[0])).abs().sum(0).double()
                frames += x.shape[1]
    for h in hooks:
        h.remove()
    for k in imp:
        torch.testing.assert_close(imp[k], (sums[k] / frames).float(), rtol=1e-4, atol=1e-7)
    assert frames == sum(int(enc._get_subsampling_output_length(m.sum(1)).sum()) for _, m in batches)


def test_recalibrate_batchnorm_cumulative_average_batch_size_one():
    torch.manual_seed(0)
    teacher = tiny_teacher(seed=4)
    student = S.build_student(teacher, S.StudentSpec([0, 1, 3], 64, [0, 2]), rand_importance([0, 1, 3], 128))
    batches = feat_batches([[160, 120, 97], [200, 64]], seed=2)
    max_utts = 4

    # reference: the same utterances one at a time (BN in train mode = batch statistics), BN inputs captured
    ref = S.build_student(teacher, S.StudentSpec([0, 1, 3], 64, [0, 2]), rand_importance([0, 1, 3], 128))
    bns = [m for m in ref.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    stats = {id(bn): [] for bn in bns}
    hooks = [bn.register_forward_pre_hook(lambda m, inp: stats[id(m)].append(
        (inp[0][0].mean(-1), inp[0][0].var(-1, unbiased=True)))) for bn in bns]
    for bn in bns:
        bn.train()
    utts = [(f[i:i + 1, :int(m[i].sum())]) for f, m in batches for i in range(f.shape[0])][:max_utts]
    with torch.no_grad():
        for x in utts:
            ref.model.encoder(input_features=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long))
    for h in hooks:
        h.remove()

    from kitsune.patches import freeze_batchnorm

    freeze_batchnorm(student)  # a frozen BN ignores .train(); recalibration must still take effect
    drift = S.recalibrate_batchnorm(student, batches, device="cpu", max_utts=max_utts)
    sbns = [m for m in student.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(sbns) == 3
    for bn, rbn in zip(sbns, bns):
        means = torch.stack([m for m, _ in stats[id(rbn)]])
        vars_ = torch.stack([v for _, v in stats[id(rbn)]])
        assert means.shape[0] == max_utts
        torch.testing.assert_close(bn.running_mean, means.mean(0), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(bn.running_var, vars_.mean(0), rtol=1e-4, atol=1e-6)
        assert int(bn.num_batches_tracked) == max_utts
        assert not bn.training and bn.momentum == 0.1  # frozen again, momentum restored
    assert not student.training
    assert drift["n_utts"] == max_utts and drift["n_mel_frames"] == 160 + 120 + 97 + 200
    assert len(drift["layers"]) == 3 and drift["mean_shift_std"] > 0 and drift["abs_log_var_ratio"] > 0

    with pytest.raises(ValueError):
        S.recalibrate_batchnorm(student, [], device="cpu")


# ------------------------------------------------------------------------------------------------ save / load


def test_save_load_roundtrip(tmp_path):
    teacher = tiny_teacher(seed=5)
    spec = S.StudentSpec([0, 2, 3], 48, [0, 2])
    imp = rand_importance(spec.enc_layers, 128)
    student = S.build_student(teacher, spec, imp)
    with torch.no_grad():  # values bf16 cannot hold: they must survive as fp32
        for bn in (m for m in student.modules() if isinstance(m, torch.nn.BatchNorm1d)):
            bn.running_var.copy_(1.0 + 1e-4 * torch.arange(bn.running_var.numel()))
    keep = S.select_ffn_neurons(imp, spec.enc_layers, 48, 128)
    meta = dict(spec=spec, kept={f"{l}.{n}": v for (l, n), v in keep.items()}, note="roundtrip")
    out = tmp_path / "student"
    S.save_student(student, out, processor=None, meta=meta)

    assert {"config.json", "generation_config.json", "model.safetensors", "student_meta.json"} <= set(os.listdir(out))
    # the model card (licence terms, the Apache-2.0 modified-from notice) travels with the weights
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card == S.MODEL_CARD.read_text(encoding="utf-8")
    assert "license: other" in card and f"base_model: {S.TEACHER_ID}" in card
    assert "b1eacc2686a3d08ceaae5f24a88b1d519620bc09" in card and "non-commercial" in card
    with safe_open(out / "model.safetensors", "pt") as f:
        keys = set(f.keys())
        dtypes = {k: f.get_slice(k).get_dtype() for k in keys}
    assert "model.encoder.layers.0.conv.norm.running_var" in keys  # HF names, not the NeMo ones
    assert "proj_out.weight" not in keys and "model.decoder.embed_tokens.weight" in keys  # tied: stored once
    for k, dt in dtypes.items():
        want = "I64" if k.endswith("num_batches_tracked") else "F32" if ".running_" in k else "BF16"
        assert dt == want, (k, dt)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["tie_word_embeddings"] and cfg["decoder_start_token_id"] == 13764 and cfg["dtype"] == "bfloat16"
    assert cfg["encoder_config"]["num_hidden_layers"] == 3 and "transf_decoder" not in cfg
    gen = json.loads((out / "generation_config.json").read_text(encoding="utf-8"))
    assert (gen["decoder_start_token_id"], gen["eos_token_id"], gen["pad_token_id"]) == (13764, 3, 2)
    m = S.load_meta(out)
    assert m["spec"]["enc_layers"] == [0, 2, 3] and m["kept"]["2.feed_forward1"] == keep[(2, "feed_forward1")].tolist()
    assert student.config.dtype == torch.float32  # the in-memory config is untouched

    loaded = S.load_student(out, "cpu")
    assert not loaded.training and loaded.model.encoder.config._attn_implementation == "sdpa"
    assert loaded.proj_out.weight is loaded.model.decoder.embed_tokens.weight
    rounded = {k: (v if ".running_" in k or not v.is_floating_point() else v.to(torch.bfloat16).float())
               for k, v in student.state_dict().items()}
    for k, v in loaded.state_dict().items():
        assert v.dtype == rounded[k].dtype and torch.equal(v, rounded[k]), k
    for j, layer in enumerate(loaded.model.decoder.layers):
        assert layer.self_attn.layer_idx == j
    feats, mask = feat_batches([[120, 80]], seed=6)[0]
    out_ids = loaded.generate(input_features=feats, attention_mask=mask.long(), decoder_input_ids=torch.tensor([PROMPT] * 2),
                              max_new_tokens=3, min_new_tokens=3, do_sample=False)
    assert out_ids.shape == (2, len(PROMPT) + 3)


def test_save_with_processor(tmp_path):
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:  # not in the local cache
        pytest.skip(f"teacher processor not cached: {e}")
    student = S.build_student(tiny_teacher(seed=6), S.StudentSpec([0, 3], 128, [1]), importance=None)
    S.save_student(student, tmp_path, proc, {"x": 1})
    names = set(os.listdir(tmp_path))
    # transformers 5 keeps the feature-extractor settings inside processor_config.json
    assert {"processor_config.json", "tokenizer.json", "tokenizer_config.json", "student_meta.json"} <= names, names
    again = AutoProcessor.from_pretrained(str(tmp_path))
    assert again.tokenizer.convert_tokens_to_ids(again.tokenizer.convert_ids_to_tokens(PROMPT)) == PROMPT
    fe, fe0 = again.feature_extractor, proc.feature_extractor
    assert (fe.feature_size, fe.n_fft, fe.hop_length, fe.win_length, fe.dither, fe.preemphasis) == \
        (fe0.feature_size, fe0.n_fft, fe0.hop_length, fe0.win_length, fe0.dither, fe0.preemphasis)
    assert torch.equal(torch.as_tensor(fe.mel_filters), torch.as_tensor(fe0.mel_filters))


# ------------------------------------------------------------------------------------------------ 03_build_student


def test_build_script_help_is_light():
    """--help must work without importing torch/transformers (or the rest of the training stack)."""
    script = ROOT / "scripts" / "03_build_student.py"
    code = ("import runpy, sys\n"
            "sys.argv = ['03_build_student.py', '--help']\n"
            "try:\n"
            f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    assert e.code in (0, None), e.code\n"
            "print('HEAVY' if {'torch', 'transformers'} & set(sys.modules) else 'LIGHT')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "--enc-layers" in r.stdout and "--step0-eval-utts" in r.stdout
    assert r.stdout.strip().endswith("LIGHT")


def test_build_script_end_to_end_tiny(tmp_path):
    """03 on a synthetic corpus in the real on-disk formats with a tiny saved teacher, on CPU: build, the no-op re-run,
    a forced rebuild that reuses importance.pt, and the resume of a run that died in the step-0 eval."""
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not cached: {e}")
    from fixtures import load_script, make_fake_corpus, make_fake_selection

    fc = make_fake_corpus(tmp_path / "corpus", sources={"src_a": (48, "train"), "eval_x": (12, "eval")})
    sel = make_fake_selection(fc, greedy_n=6, probe_n=4)
    teacher_dir = tmp_path / "teacher"
    tiny_teacher(seed=8).save_pretrained(teacher_dir)
    proc.save_pretrained(teacher_dir)
    out = tmp_path / "student"
    argv = ["--teacher", str(teacher_dir), "--enc-layers", "3", "--ffn", "48", "--dec-layers", "0", "2",
            "--calib-utts", "12", "--bn-utts", "8", "--sources", "src_a", "--device", "cpu",
            "--step0-eval-utts", "4", "--eval-sets", "eval_x", "--selection", str(sel), "--data", str(fc.data),
            "--teacher-root", str(fc.teacher_out), "--eval-cache", str(tmp_path / "cache_eval"), "--out", str(out),
            "--calib-per-group", "4", "--calib-batch-s", "20", "--eval-batch-s", "20"]
    mod = load_script("03_build_student")
    assert mod.main(argv) == 0

    meta = S.load_meta(out)
    assert meta["stage"] == "complete"
    assert meta["spec"] == {"enc_layers": [0, 2, 3], "ffn_dim": 48, "dec_layers": [0, 2], "tie_head": True}
    assert meta["calibration"]["n_importance"] == 12 and meta["bn"]["n_utts"] == 8
    assert len(meta["kept"]["ffn"]["2.feed_forward1"]) == 48 and meta["kept"]["ffn"]["2.feed_forward1"] == sorted(
        meta["kept"]["ffn"]["2.feed_forward1"])
    assert 0 < meta["importance"]["kept_mass_min"] <= meta["importance"]["kept_mass_mean"] < 1
    assert meta["params"]["total"] == meta["params"]["closed_form"]
    assert "teacher_revision" in meta and meta["teacher_revision"] is None  # a local teacher dir has no hub commit
    assert (out / "README.md").exists()  # the model card
    assert meta["step0"]["n_per_set"] == {"eval_x": 4}
    assert meta["step0"]["greedy"]["sets"]["eval_x"]["n"] == 4
    assert "kl" in meta["step0"]["teacher_forced"]["sets"]["eval_x"]
    assert {"started", "saved", "finished"} <= set(meta["timestamps"]) and "sha" in meta["git"]
    assert (out / "importance.pt").exists() and (out / "step0" / "greedy.parquet").exists()
    student = S.load_student(out, "cpu")
    assert student.config.encoder_config.num_hidden_layers == 3 and student.proj_out.weight is \
        student.model.decoder.embed_tokens.weight
    assert all(int(m.num_batches_tracked) == 8 for m in student.modules() if isinstance(m, torch.nn.BatchNorm1d))

    assert mod.main(argv) == 0  # complete -> nothing to do
    assert S.load_meta(out)["timestamps"]["finished"] == meta["timestamps"]["finished"]

    mtime = (out / "importance.pt").stat().st_mtime_ns
    assert mod.main(argv + ["--force", "--step0-eval-utts", "0"]) == 0
    m2 = S.load_meta(out)
    assert m2["importance"]["reused"] and (out / "importance.pt").stat().st_mtime_ns == mtime
    assert m2["kept"] == meta["kept"] and m2["step0"] == {"skipped": True}

    m2["stage"] = "saved"  # as if the process died in the step-0 eval
    m2.pop("step0")
    S.write_meta(out, m2)
    assert mod.main(argv) == 0
    m3 = S.load_meta(out)
    assert m3["stage"] == "complete" and "resumed" in m3["timestamps"] and m3["step0"]["n_per_set"] == {"eval_x": 4}


def test_importance_state_roundtrip(tmp_path):
    imp = rand_importance([0, 16, 47], 5120)
    torch.save(S.importance_to_state(imp), tmp_path / "importance.pt")
    back = S.importance_from_state(torch.load(tmp_path / "importance.pt", weights_only=True))
    assert set(back) == set(imp) and all(torch.equal(back[k], imp[k]) for k in imp)


# ------------------------------------------------------------------------------------------------ the real teacher


def _teacher_weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(S.TEACHER_ID, "model.safetensors"), str)
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _teacher_weights_cached(), reason="teacher weights not in the local HF cache")
def test_build_default_student_from_real_teacher():
    """The real B20x2560/dec4 student from the cached bf16 teacher on CPU (random importance: shape, not quality)."""
    t0 = time.time()
    teacher = CohereAsrForConditionalGeneration.from_pretrained(S.TEACHER_ID, dtype=torch.bfloat16,
                                                                attn_implementation="sdpa")
    t_load = time.time() - t0
    spec = S.default_spec()
    imp = rand_importance(spec.enc_layers, 5120, seed=7)
    student = S.build_student(teacher, spec, imp)  # strict=True inside: raises on any missing/unexpected key
    t_build = time.time() - t0 - t_load
    rep = S.param_report(student)
    assert abs(rep["total"] - 616_963_328) <= 0.001 * 616_963_328
    assert rep["total"] == rep["closed_form"] == S.EXPECTED_DEFAULT_PARAMS
    assert rep["tied_head"] and all(p.dtype == torch.float32 for p in student.parameters())

    keep = S.select_ffn_neurons(imp, spec.enc_layers, 2560, 5120)
    t_enc, s_enc = teacher.model.encoder.layers, student.model.encoder.layers
    for i in (0, 9, 19):
        tl = spec.enc_layers[i]
        idx = keep[(tl, "feed_forward1")]
        assert torch.equal(s_enc[i].feed_forward1.linear1.weight, t_enc[tl].feed_forward1.linear1.weight[idx].float())
        assert torch.equal(s_enc[i].feed_forward1.linear2.weight, t_enc[tl].feed_forward1.linear2.weight[:, idx].float())
        assert torch.equal(s_enc[i].self_attn.relative_k_proj.weight, t_enc[tl].self_attn.relative_k_proj.weight.float())
        assert torch.equal(s_enc[i].conv.norm.running_var, t_enc[tl].conv.norm.running_var.float())
    for j, tj in enumerate(spec.dec_layers):
        assert torch.equal(student.model.decoder.layers[j].mlp.fc1.weight, teacher.model.decoder.layers[tj].mlp.fc1.weight.float())
        assert student.model.decoder.layers[j].encoder_attn.layer_idx == j
    assert torch.equal(student.proj_out.weight, teacher.proj_out.weight.float())  # bitwise == embed in the checkpoint
    print(f"\nreal teacher: load {t_load:.1f} s, build {t_build:.1f} s, student {rep['total']:,} params")
