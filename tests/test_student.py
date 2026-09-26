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
        # HF zero-inits every Linear bias; the real teacher's are not (FFN |b| ~0.03, max ~0.12). Non-zero here so the
        # FFN bias slices (and proj_out.bias) are checked by value, not as 0 == 0.
        for lin in (x for x in m.modules() if isinstance(x, torch.nn.Linear) and x.bias is not None):
            lin.bias.normal_(0.0, 0.1)
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


@pytest.mark.parametrize("n_enc,dec,total,non_emb", [(20, (0, 2, 5, 7), 616_963_328, 599_137_536),
                                                     (8, (0, 2, 5, 7), 301_822_208, 283_996_416)])
def test_study_pruned_counts_on_meta(n_enc, dec, total, non_emb):
    """T-0.6B and T-0.3B (STUDY.md 1.1; T-0.3B = B8x2560 + decoder {0,2,5,7} by the owner's decision of 2026-09-26,
    about 301.82M): total and non-embedding (embed + pos_emb out, the head bias in)."""
    spec = S.StudentSpec(S.evenly_spaced(n_enc, 48), 2560, list(dec))
    assert S.PRUNED_EXPECTED_PARAMS[(n_enc, 2560, dec)] == total
    assert set(S.PRUNED_EXPECTED_PARAMS) == {(20, 2560, (0, 2, 5, 7)), (8, 2560, (0, 2, 5, 7))}  # no B10 + {0,7}
    if n_enc == 8:
        assert spec.enc_layers == [0, 7, 13, 20, 27, 34, 40, 47]
        assert round(total / 1e6, 2) == 301.82  # the figure STUDY.md quoted for this option of decision 17
        # the report_model.md closed form: 5,383,424 + L_enc (13,149,440 + 5,122 F) + L_dec 16,796,672 + 2,364,416
        # + 1024 V + V (tied)
        assert total == 5_383_424 + 8 * (13_149_440 + 5_122 * 2560) + 4 * 16_796_672 + 2_364_416 + 1025 * 16384
    with torch.device("meta"):
        m = CohereAsrForConditionalGeneration(S.student_config(CohereAsrConfig(), spec))
    assert S.count_fields(m) == dict(params_total=total, params_non_embedding=non_emb, closed_form_params=total)
    with torch.device("meta"):  # untied (the raw config): the head weight is vocabulary-sized, so it is not counted
        u = CohereAsrForConditionalGeneration(S.student_config(CohereAsrConfig(), S.StudentSpec(
            spec.enc_layers, 2560, list(dec), tie_head=False)))
    assert S.non_embedding_params(u) == non_emb


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
    assert torch.equal(ff.linear1.bias, teacher.model.encoder.layers[2].feed_forward2.linear1.bias[idx])
    assert torch.equal(ff.linear2.weight, teacher.model.encoder.layers[2].feed_forward2.linear2.weight[:, idx])
    assert torch.equal(ff.linear2.bias, teacher.model.encoder.layers[2].feed_forward2.linear2.bias)
    assert ff.linear1.bias.abs().sum() > 0  # a zero bias would make the slice checks above vacuous
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


def test_build_with_a_given_ffn_selection():
    """build_student(keep=...) (03 --ffn-from): the same student as from the importance that chose that selection;
    keep_from_meta reads what a build recorded; check_keep rejects a selection that does not fit the spec."""
    teacher = tiny_teacher(seed=11)
    spec = S.StudentSpec([0, 2, 3], 48, [0, 2])
    imp = rand_importance(spec.enc_layers, 128, seed=3)
    a = S.build_student(teacher, spec, imp)
    keep = S.select_ffn_neurons(imp, spec.enc_layers, 48, 128)
    meta = json.loads(S.dump_json(dict(kept=dict(ffn={f"{l}.{n}": v for (l, n), v in keep.items()}))))
    back = S.keep_from_meta(meta)
    assert set(back) == set(keep) and all(torch.equal(back[k], keep[k]) for k in keep)
    b = S.build_student(teacher, spec, importance=None, keep=back)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    # a superset (all four layers) is restricted to the spec's layers
    wide = back | {(1, n): torch.arange(48) for n in S.FFN_NAMES}
    assert set(S.check_keep(wide, spec, 128)) == set(keep)
    assert S.check_keep(back, S.StudentSpec([0, 2, 3], 128, [0]), 128) == {}  # nothing pruned
    bad = []
    k0 = (0, "feed_forward1")
    for v in (back[k0][:-1],  # too few
              back[k0].flip(0),  # not ascending
              torch.cat([back[k0][:-1], torch.tensor([128])])):  # outside the teacher's width
        bad.append(back | {k0: v})
    bad.append({k: v for k, v in back.items() if k != (3, "feed_forward2")})  # a layer missing
    for sel in bad:
        with pytest.raises(ValueError):
            S.build_student(teacher, spec, importance=None, keep=sel)
    with pytest.raises(ValueError, match="kept.ffn"):
        S.keep_from_meta({"kept": {"ffn": {}}})


def test_build_keeps_the_teachers_bn_stats():
    """Without recalibration (03 --bn keep) every kept layer's BN running stats are the teacher layer's, bitwise."""
    teacher = tiny_teacher(seed=12)
    spec = S.StudentSpec([1, 3], 64, [0])
    student = S.build_student(teacher, spec, rand_importance([1, 3], 128))
    for j, tl in enumerate(spec.enc_layers):
        s_bn, t_bn = student.model.encoder.layers[j].conv.norm, teacher.model.encoder.layers[tl].conv.norm
        for b in ("running_mean", "running_var", "num_batches_tracked"):
            assert torch.equal(getattr(s_bn, b), getattr(t_bn, b)), (j, b)


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
    # the licence texts it names (the Galgame card's own license_link points to a missing LICENSE.md)
    assert "https://www.gnu.org/licenses/gpl-3.0.txt" in card and "3fb86654222b3f0af0f7c332ae6a0ef9752a9451" in card
    assert all(f"https://creativecommons.org/licenses/by/{v}/" in card for v in ("4.0", "3.0"))
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


def test_build_script_calibrates_on_the_viability_sources_by_default():
    """The documented plain command rebuilds students/b20x2560-d4, which was calibrated on every viability source."""
    from fixtures import load_script

    viability = json.loads((ROOT / "configs" / "viability.json").read_text(encoding="utf-8"))
    args = load_script("03_build_student").parse_args([])
    assert args.sources == viability["sources"] and args.out == ROOT / viability["student"]


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
    # the BN recalibration stops its feature generator early (max_utts); closing it must not leave grad mode off for
    # the rest of the process (any later test that calls backward would fail)
    assert torch.is_grad_enabled()

    meta = S.load_meta(out)
    assert meta["stage"] == "complete"
    assert meta["spec"] == {"enc_layers": [0, 2, 3], "ffn_dim": 48, "dec_layers": [0, 2], "tie_head": True}
    assert meta["calibration"]["n_importance"] == 12 and meta["bn"] == "recal" and meta["bn_recal"]["n_utts"] == 8
    assert meta["format"] == 2 and meta["family"] == "aed" and meta["init_class"] == "pruned_kept"
    assert meta["params_total"] == meta["closed_form_params"] == meta["params"]["total"]
    assert meta["params_non_embedding"] == meta["params_total"] - 16384 * 64 - 1024 * 64  # tied: embed + pos_emb out
    assert (meta["enc_layers"], meta["ffn"], meta["dec_layers"], meta["seed"]) == ([0, 2, 3], 48, [0, 2], 1234)
    assert meta["teacher"] == str(teacher_dir)  # a local dir has no commit: no "@<commit>"
    assert len(meta["kept"]["ffn"]["2.feed_forward1"]) == 48 and meta["kept"]["ffn"]["2.feed_forward1"] == sorted(
        meta["kept"]["ffn"]["2.feed_forward1"])
    assert 0 < meta["importance"]["kept_mass_min"] <= meta["importance"]["kept_mass_mean"] < 1
    assert meta["params"]["total"] == meta["params"]["closed_form"]
    assert "teacher_revision" in meta and meta["teacher_revision"] is None  # a local teacher dir has no hub commit
    card = (out / "README.md").read_text(encoding="utf-8")  # the model card, its notice describing this student
    assert "It was pruned (encoder: 3 of 48 layers, FFN 5120 -> 48; decoder: 2 of 8 layers), its output head tied to " \
           "the token embedding and its BatchNorm statistics recalibrated, then trained" in card
    assert meta["step0"]["n_per_set"] == {"eval_x": 4} and meta["step0"]["seed"] == 1234
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


def test_build_script_study_flags_end_to_end_tiny(tmp_path):
    """The size study's pruned builds on a tiny saved teacher: --bn keep (teacher BN stats, no calibration audio for
    BN), --importance-layers all (one cache over every teacher layer) reused by another size through
    --importance-cache, and --ffn-from (another student's FFN selection, cross-checked against its importance.pt;
    no calibration audio at all). Plus the step-0 function check and the build identity guard."""
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not cached: {e}")
    from fixtures import load_script, make_fake_corpus, make_fake_selection
    from safetensors.torch import load_file

    fc = make_fake_corpus(tmp_path / "corpus", sources={"src_a": (40, "train"), "eval_x": (12, "eval")})
    sel = make_fake_selection(fc, greedy_n=6, probe_n=4)
    teacher_dir = tmp_path / "teacher"
    teacher = tiny_teacher(seed=9)
    teacher.save_pretrained(teacher_dir)
    proc.save_pretrained(teacher_dir)
    common = ["--teacher", str(teacher_dir), "--calib-utts", "10", "--bn-utts", "6", "--sources", "src_a",
              "--device", "cpu", "--eval-sets", "eval_x", "--selection", str(sel), "--data", str(fc.data),
              "--teacher-root", str(fc.teacher_out), "--eval-cache", str(tmp_path / "cache_eval"),
              "--calib-per-group", "4", "--calib-batch-s", "20", "--eval-batch-s", "20", "--teacher-dtype", "fp32"]
    mod = load_script("03_build_student")

    a = tmp_path / "a"  # B3x48 / dec {0,2}, BN kept, importance over all 4 teacher layers
    argv_a = common + ["--enc-layers", "3", "--ffn", "48", "--dec-layers", "0", "2", "--bn", "keep",
                       "--importance-layers", "all", "--step0-eval-utts", "3", "--out", str(a)]
    assert mod.main(argv_a) == 0
    ma = S.load_meta(a)
    assert ma["bn"] == "teacher" and "bn_recal" not in ma and ma["calibration"]["n_bn"] == 0
    assert ma["calibration"]["n_importance"] == 10 and ma["importance"]["n_layers"] == 4
    assert ma["importance"]["teacher_dtype"] == "torch.float32" and not ma["importance"]["reused"]
    assert (ma["family"], ma["init_class"], ma["enc_layers"], ma["ffn"]) == ("aed", "pruned_kept", [0, 2, 3], 48)
    assert ma["params_total"] == ma["closed_form_params"] and ma["expected_params"] is None  # not a study shape
    fcheck = ma["function_check"]
    assert fcheck["on_gate"] is False and fcheck["threshold"] == 0.9 and isinstance(fcheck["over_threshold"], bool)
    assert fcheck["function"] == "kept" and fcheck["by"] == mod.AED_FUNCTION_RULE  # the owner's decision, not the rule
    assert fcheck["cer_vs_teacher"] == ma["step0"]["greedy"]["all"]["cer_teacher_corpus"]
    assert ma["build"]["seed"] == 1234 and ma["build"]["calibration"] == dict(
        sources=["src_a"], calib_utts=10, bn_utts=6, per_group=4, ids_sha256=None)
    card = (a / "README.md").read_text(encoding="utf-8")
    assert "It was pruned (encoder: 3 of 48 layers, FFN 5120 -> 48; decoder: 2 of 8 layers), its output head tied to " \
           "the token embedding, then trained by distillation from the teacher's outputs. Its BatchNorm statistics " \
           "are the teacher's." in card and "recalibrated" not in card
    cache = torch.load(a / "importance.pt", weights_only=True)
    assert cache["layers"] == [0, 1, 2, 3] and len(cache["importance"]) == 8
    tsd = teacher.state_dict()
    sd = load_file(a / "model.safetensors")
    for j, tl in enumerate([0, 2, 3]):  # the teacher's BN stats, fp32, bitwise
        for b in ("running_mean", "running_var", "num_batches_tracked"):
            assert torch.equal(sd[f"model.encoder.layers.{j}.conv.norm.{b}"],
                               tsd[f"model.encoder.layers.{tl}.conv.norm.{b}"])

    # another size reuses the all-layer cache: nothing recomputed, nothing written
    b = tmp_path / "b"
    mtime = (a / "importance.pt").stat().st_mtime_ns
    assert mod.main(common + ["--enc-layers", "2", "--ffn", "32", "--dec-layers", "1", "--bn", "keep",
                              "--importance-cache", str(a / "importance.pt"), "--step0-eval-utts", "0",
                              "--out", str(b)]) == 0
    mb = S.load_meta(b)
    assert mb["importance"]["reused"] and mb["importance"]["path"] == str(a / "importance.pt")
    assert (a / "importance.pt").stat().st_mtime_ns == mtime and not (b / "importance.pt").exists()
    imp = S.importance_from_state(cache["importance"])
    want = S.select_ffn_neurons(imp, [0, 3], 32, 128)
    assert mb["kept"]["ffn"] == {f"{l}.{n}": v.tolist() for (l, n), v in sorted(want.items())}
    # a named cache that does not match (other calibration ids; the same ids through a bf16 teacher, which ranks the
    # neurons at the cut differently) is an input: refused, never overwritten
    with pytest.raises(SystemExit, match="does not match"):
        mod.main([x if x != "10" else "9" for x in common] + [
            "--enc-layers", "2", "--ffn", "32", "--dec-layers", "1", "--bn", "keep", "--importance-cache",
            str(a / "importance.pt"), "--step0-eval-utts", "0", "--out", str(tmp_path / "b2")])
    with pytest.raises(SystemExit, match="does not match"):
        mod.main([x if x != "fp32" else "bf16" for x in common] + [
            "--enc-layers", "2", "--ffn", "32", "--dec-layers", "1", "--bn", "keep", "--importance-cache",
            str(a / "importance.pt"), "--step0-eval-utts", "0", "--out", str(tmp_path / "b2")])
    assert (a / "importance.pt").stat().st_mtime_ns == mtime
    assert cache["key"]["teacher_dtype"] == "torch.float32"

    # the sampled ids are written in sample order; --calib-ids reads exactly them back (another seed would draw
    # others), so the cache key matches again
    ids_a = (a / "calibration_ids.txt").read_text(encoding="utf-8").split()
    assert len(ids_a) == 10 and ma["calibration"]["ids_file"] == "calibration_ids.txt"
    assert mod.ids_sha(ids_a) == ma["calibration"]["importance_ids_sha256"] == cache["key"]["ids_sha256"]
    b3 = tmp_path / "b3"
    assert mod.main(common + [
        "--seed", "99", "--calib-ids", str(a / "calibration_ids.txt"), "--enc-layers", "2", "--ffn", "32",
        "--dec-layers", "1", "--bn", "keep", "--importance-cache", str(a / "importance.pt"), "--step0-eval-utts", "0",
        "--out", str(b3)]) == 0
    mb3 = S.load_meta(b3)
    assert mb3["importance"]["reused"] and mb3["calibration"]["ids_from"] == str(a / "calibration_ids.txt")
    assert mb3["build"]["seed"] == 99 and mb3["build"]["calibration"]["ids_sha256"] == mod.ids_sha(ids_a)
    assert (b3 / "calibration_ids.txt").read_text(encoding="utf-8").split() == ids_a
    (tmp_path / "bad_ids.txt").write_text("\n".join(ids_a[:9] + ["no-such-id"]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not kept train rows"):
        mod.main(common + ["--calib-ids", str(tmp_path / "bad_ids.txt"), "--enc-layers", "2", "--ffn", "32",
                           "--dec-layers", "1", "--bn", "keep", "--step0-eval-utts", "0",
                           "--out", str(tmp_path / "b4")])

    # --ffn-from: a's selection, verbatim; no calibration audio; the same weights as a
    c = tmp_path / "c"
    argv_c = common + ["--enc-layers", "3", "--ffn", "48", "--dec-layers", "0", "2", "--bn", "keep",
                       "--ffn-from", str(a), "--step0-eval-utts", "0", "--out", str(c)]
    assert mod.main(argv_c) == 0
    mc = S.load_meta(c)
    assert mc["kept"] == ma["kept"] and mc["importance"]["reproduced_by_importance"]
    assert mc["importance"]["source_teacher_dtype"] == "torch.float32"  # the selection is a's, measured in fp32
    assert mc["importance"]["ffn_from"] == str(a) and mc["calibration"]["n_importance"] == 0
    assert mc["calibration"]["importance_ids_sha256"] == ma["calibration"]["importance_ids_sha256"]
    assert mc["build"]["ffn_from"] == str(a) and "sample" in mc["durations_s"]
    sc = load_file(c / "model.safetensors")
    assert set(sc) == set(sd) and all(torch.equal(sc[k], sd[k]) for k in sd)
    with pytest.raises(ValueError):  # a's selection has no 5120 -> 32 FFNs for these layers
        mod.main(common + ["--enc-layers", "3", "--ffn", "32", "--dec-layers", "0", "--bn", "keep",
                           "--ffn-from", str(a), "--step0-eval-utts", "0", "--out", str(tmp_path / "d")])
    tampered = dict(ma, kept=dict(ma["kept"], ffn=dict(ma["kept"]["ffn"])))
    ks = tampered["kept"]["ffn"]["2.feed_forward1"]
    ks = sorted(set(range(128)) - set(ks))[:1] + ks[1:]
    tampered["kept"]["ffn"]["2.feed_forward1"] = sorted(ks)
    (tmp_path / "e").mkdir()
    S.write_meta(tmp_path / "e", tampered)
    (tmp_path / "e" / "importance.pt").write_bytes((a / "importance.pt").read_bytes())
    with pytest.raises(SystemExit, match="does not reproduce"):
        mod.main(common + ["--enc-layers", "3", "--ffn", "48", "--dec-layers", "0", "2", "--bn", "keep",
                           "--ffn-from", str(tmp_path / "e"), "--step0-eval-utts", "0", "--out", str(tmp_path / "f")])

    # the build identity: the same dir with recalibrated BN is another student; so is one with another seed (it draws
    # the calibration sample and the step-0 ids), even when only the step-0 eval is left to resume
    with pytest.raises(SystemExit, match="different student"):
        mod.main([x if x != "keep" else "recal" for x in argv_a])
    assert mod.main(argv_a) == 0  # the same build: already complete
    S.write_meta(a, dict(S.load_meta(a), stage="saved"))
    with pytest.raises(SystemExit, match="different student"):
        mod.main(argv_a + ["--seed", "5"])


def test_build_script_refuses_a_teacher_commit_the_targets_did_not_come_from(tmp_path):
    """The student's init weights and tokenizer must come from the teacher commit teacher_out/meta.json records; an
    unknown side (a local teacher dir, a meta.json from before model_revision was recorded, none at all) passes."""
    from fixtures import load_script

    mod = load_script("03_build_student")
    a, b = "a" * 40, "b" * 40
    mod.check_teacher_matches_targets(tmp_path, a)  # no meta.json
    (tmp_path / "meta.json").write_text(json.dumps({"model": S.TEACHER_ID}), encoding="utf-8")
    mod.check_teacher_matches_targets(tmp_path, a)  # legacy meta.json
    (tmp_path / "meta.json").write_text(json.dumps({"model": S.TEACHER_ID, "model_revision": a}), encoding="utf-8")
    mod.check_teacher_matches_targets(tmp_path, a)
    mod.check_teacher_matches_targets(tmp_path, None)  # local teacher dir
    with pytest.raises(SystemExit, match=f"--teacher-revision {a}"):
        mod.check_teacher_matches_targets(tmp_path, b)


def test_importance_cache_is_keyed_on_the_teacher_commit(tmp_path, monkeypatch):
    """importance.pt computed from one teacher commit is not reused for another, nor is one written before the key
    held the commit: other weights rank other FFN neurons."""
    from types import SimpleNamespace

    from fixtures import load_script

    mod = load_script("03_build_student")
    computed = []

    def fake_importance(teacher, feats, layers, device):
        computed.append(teacher.config._commit_hash)
        return {(l, n): torch.rand(8) for l in layers for n in S.FFN_NAMES}

    monkeypatch.setattr(S, "ffn_importance", fake_importance)
    utts = [dict(id=f"u{i}", duration=1.0, source="s", wave=None) for i in range(3)]
    args = SimpleNamespace(teacher=S.TEACHER_ID, calib_batch_s=10.0)
    spec = SimpleNamespace(enc_layers=[0, 2])

    def reused(commit):
        teacher = SimpleNamespace(config=SimpleNamespace(_commit_hash=commit))
        _, info = mod.load_or_compute_importance(teacher, None, utts, args, spec, "cpu", tmp_path, log=lambda *_: None)
        return info["reused"]

    a, b = "a" * 40, "b" * 40
    assert [reused(a), reused(a), reused(b)] == [False, True, False] and computed == [a, b]
    c = torch.load(tmp_path / "importance.pt", weights_only=True)  # as written before the key held the commit
    c["key"].pop("teacher_revision")
    torch.save(c, tmp_path / "importance.pt")
    assert reused(b) is False and computed == [a, b, b]


def test_importance_cache_is_keyed_on_the_teacher_dtype(tmp_path, monkeypatch):
    """The same utterances through a bf16 and an fp32 teacher rank the neurons at the cut differently, so a cache is
    only reused by a build in its own dtype. A cache from before the key held the dtype counts as the dtype its info
    recorded, or bf16 (the only one before --teacher-dtype)."""
    from types import SimpleNamespace

    from fixtures import load_script

    mod = load_script("03_build_student")
    computed = []

    def fake_importance(teacher, feats, layers, device):
        computed.append(mod.teacher_dtype(teacher))
        return {(l, n): torch.rand(8) for l in layers for n in S.FFN_NAMES}

    monkeypatch.setattr(S, "ffn_importance", fake_importance)
    utts = [dict(id=f"u{i}", duration=1.0, source="s", wave=None) for i in range(3)]
    args = SimpleNamespace(teacher=S.TEACHER_ID, calib_batch_s=10.0)
    spec = SimpleNamespace(enc_layers=[0, 2])
    path = tmp_path / "importance.pt"

    def reused(dtype):
        teacher = torch.nn.Linear(2, 2).to(dtype)
        teacher.config = SimpleNamespace(_commit_hash="a" * 40)
        _, info = mod.load_or_compute_importance(teacher, None, utts, args, spec, "cpu", tmp_path, log=lambda *_: None)
        return info["reused"]

    assert [reused(torch.float32), reused(torch.float32), reused(torch.bfloat16)] == [False, True, False]
    c = torch.load(path, weights_only=True)
    assert c["key"]["teacher_dtype"] == c["info"]["teacher_dtype"] == "torch.bfloat16"
    c["key"].pop("teacher_dtype")  # written before the key held the dtype; info says bf16
    torch.save(c, path)
    assert reused(torch.bfloat16) is True
    c["key"].pop("teacher_dtype", None)
    c["info"]["teacher_dtype"] = "torch.float32"  # this branch's first caches: the dtype only in info
    torch.save(c, path)
    assert reused(torch.float32) is True and reused(torch.bfloat16) is False
    c = torch.load(path, weights_only=True)
    c["key"].pop("teacher_dtype")
    c["info"].pop("teacher_dtype")  # the first run's: no dtype anywhere, computed in bf16
    torch.save(c, path)
    assert mod.importance_key(c)["teacher_dtype"] == "torch.bfloat16" and reused(torch.bfloat16) is True
    assert computed == ["torch.float32", "torch.bfloat16", "torch.bfloat16"]


def test_feature_batches_leave_grad_mode_alone():
    """A consumer under @torch.no_grad() that stops early leaves feature_batches suspended; closing it afterwards must
    not switch grad mode off for the rest of the process (it did while the generator yielded inside its no_grad)."""
    import gc

    import numpy as np
    from fixtures import load_script

    mod = load_script("03_build_student")
    utts = [dict(id=f"u{i}", duration=1.0, wave=np.zeros(16000, np.float32)) for i in range(3)]

    @torch.no_grad()
    def first(it):
        for item in it:
            return item

    g = mod.feature_batches(utts, lambda w, n: (w, n), "cpu", 1.0, "test")  # one utterance per batch
    wave, _ = first(g)
    assert torch.is_grad_enabled() and wave.shape == (1, 16000)
    del g
    gc.collect()
    assert torch.is_grad_enabled()


def test_build_identity_gate_and_legacy_meta():
    """A pruned build's identity holds the seed and the calibration inputs; the first run's format-1 meta still reads
    as the default build; function_check's on_gate needs the gate's 20 per set AND its seed."""
    from fixtures import load_script

    mod = load_script("03_build_student")
    default = mod.build_identity(mod.parse_args([]))
    assert default["seed"] == 1234 and default["calibration"] == dict(
        sources=["reazon_small", "emilia_yodas", "galgame"], calib_utts=1000, bn_utts=1000, per_group=16,
        ids_sha256=None)
    first_run = {"format": 1, "args": {  # students/b20x2560-d4/student_meta.json
        "enc_layers": 20, "ffn": 2560, "dec_layers": [0, 2, 5, 7], "no_tie_head": False, "calib_utts": 1000,
        "bn_utts": 1000, "sources": ["reazon_small", "emilia_yodas", "galgame"], "seed": 1234, "calib_per_group": 16}}
    assert mod.legacy_identity(first_run) == default
    assert mod.build_identity(mod.parse_args(["--seed", "7"])) != default

    step0 = dict(n_per_set={s: 20 for s in mod.EVAL_SETS}, seed=1234, greedy={"all": {"cer_teacher_corpus": 0.95}},
                 teacher_forced={"all": {"kl": 4.0}})
    fc = mod.function_check(step0)
    # the owner's decision of 2026-09-26: a pruned Transcribe student is function kept even over the 90 % line (T-0.6B
    # measures 132 %); the 90 % rule classifies the Parakeet students only (03c)
    assert fc == dict(cer_vs_teacher=0.95, kl=4.0, threshold=0.9, over_threshold=True, function="kept",
                      by=mod.AED_FUNCTION_RULE, on_gate=True)
    assert "Parakeet students only" in mod.AED_FUNCTION_RULE and "kept-t03" in mod.AED_FUNCTION_RULE
    assert mod.function_check(dict(step0, seed=7))["on_gate"] is False  # another 60 utterances
    assert mod.function_check(dict(step0, n_per_set={"eval_jsut": 20}))["on_gate"] is False
    no_seed = {k: v for k, v in step0.items() if k != "seed"}  # recorded before step-0 held its seed
    assert mod.function_check(no_seed)["on_gate"] is False
    under = mod.function_check(dict(step0, greedy={"all": {"cer_teacher_corpus": 0.6}}))
    assert (under["over_threshold"], under["function"]) == (False, "kept")


def test_model_card_describes_the_student():
    """README.md's Apache-2.0 modification notice says what was done to the teacher: the first run's student keeps
    the card as written; a study student gets its own shape and BN mode; a scratch student says its weights are
    random. The card's wording is what model_card rewrites (a changed card would only warn)."""
    from dataclasses import asdict

    card = S.MODEL_CARD.read_bytes().decode("utf-8")
    assert S._CARD_CHANGES.search(card)
    assert S.model_card({}) == card and S.model_card({"format": 1, "bn": {"n_utts": 1000}}) == card

    def split(text):  # (the card outside the notice's what-was-done part, that part with whitespace normalised)
        head, rest = text.split("LICENSE-2.0). ", 1)
        part, tail = rest.split("\n## Training data", 1)
        return head + tail, " ".join(part.split())

    def notice(meta):
        outside, part = split(S.model_card(meta))
        assert outside == split(card)[0]  # everything else is the card's, byte for byte
        return part

    pruned = dict(init_class="pruned_kept", enc_layers=S.evenly_spaced(20, 48), ffn=2560, dec_layers=[0, 2, 5, 7],
                  build={"tie_head": True})
    assert notice(dict(pruned, bn="recal")) == split(card)[1]  # the first run's student: the card's own words
    t03 = notice(dict(pruned, bn="teacher", enc_layers=S.evenly_spaced(8, 48)))
    assert t03.startswith("It was pruned (encoder: 8 of 48 layers, FFN 5120 -> 2560; decoder: 4 of 8 layers), its "
                          "output head tied to the token embedding, then trained by distillation from the teacher's "
                          "outputs. Its BatchNorm statistics are the teacher's.") and "recalibrated" not in t03
    untied = notice(dict(pruned, bn="recal", ffn=5120, build={"tie_head": False}))
    assert "FFN 5120;" in untied and "tied" not in untied and "BatchNorm statistics recalibrated" in untied
    scratch = notice(dict(init_class="scratch", bn="fresh", scratch=dict(name="t01", **asdict(S.SCRATCH_SHAPES["t01"])),
                          build={"init": "scratch"}))
    assert scratch.startswith("It has that model's architecture at another size (encoder: 12 layers, width 512, FFN "
                              "2048; decoder: 4 layers, width 512, FFN 2048;") and "randomly initialised" in scratch
    assert "pruned" not in scratch and "FFN neurons kept" not in scratch


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
        for n in S.FFN_NAMES:  # the real FFN biases are non-zero, so the bias slices are checked by value here
            idx = keep[(tl, n)]
            s_ff, t_ff = getattr(s_enc[i], n), getattr(t_enc[tl], n)
            assert torch.equal(s_ff.linear1.weight, t_ff.linear1.weight[idx].float())
            assert torch.equal(s_ff.linear1.bias, t_ff.linear1.bias[idx].float())
            assert torch.equal(s_ff.linear2.weight, t_ff.linear2.weight[:, idx].float())
            assert torch.equal(s_ff.linear2.bias, t_ff.linear2.bias.float())
        assert torch.equal(s_enc[i].self_attn.relative_k_proj.weight, t_enc[tl].self_attn.relative_k_proj.weight.float())
        assert torch.equal(s_enc[i].conv.norm.running_var, t_enc[tl].conv.norm.running_var.float())
    for j, tj in enumerate(spec.dec_layers):
        assert torch.equal(student.model.decoder.layers[j].mlp.fc1.weight, teacher.model.decoder.layers[tj].mlp.fc1.weight.float())
        assert student.model.decoder.layers[j].encoder_attn.layer_idx == j
    assert torch.equal(student.proj_out.weight, teacher.proj_out.weight.float())  # bitwise == embed in the checkpoint
    print(f"\nreal teacher: load {t_load:.1f} s, build {t_build:.1f} s, student {rep['total']:,} params")
