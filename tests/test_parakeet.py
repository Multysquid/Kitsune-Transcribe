"""kitsune/parakeet.py (guarded greedy TDT, CTC targets, pins) and tools/publish_parakeet.py (head, model json)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from kitsune import parakeet as pk  # noqa: E402

V, BLANK_T = 12, 11          # vocab 11 + blank
DUR = (0, 1, 2, 3, 4)


def tiny_tdt(seed=0):
    from transformers import ParakeetForTDT
    from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig, ParakeetTDTConfig
    torch.manual_seed(seed)
    enc = ParakeetEncoderConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                intermediate_size=64, num_mel_bins=16, subsampling_factor=8,
                                subsampling_conv_channels=8, conv_kernel_size=3)
    cfg = ParakeetTDTConfig(encoder_config=enc.to_dict(), vocab_size=V, blank_token_id=BLANK_T, pad_token_id=BLANK_T,
                            decoder_hidden_size=24, num_decoder_layers=2, durations=list(DUR),
                            max_symbols_per_step=10, hidden_act="relu")
    m = ParakeetForTDT(cfg).eval()
    with torch.no_grad():                           # rows emit and advance: favour durations 1-2, mild blank
        m.joint.head.bias[V:] += torch.tensor([0.0, 2.0, 2.0, 0.5, 0.0])
        m.joint.head.weight.mul_(4.0)
    m.generation_config.decoder_start_token_id = BLANK_T
    m.generation_config.pad_token_id = BLANK_T
    m.generation_config.suppress_tokens = list(range(V, V + len(DUR)))
    return m


def run_ours(m, feats, am, max_symbols=None, k=4):
    with torch.no_grad():
        enc = m.encoder(input_features=feats, attention_mask=am, output_attention_mask=True)
        h = enc.last_hidden_state
        f = m.encoder_projector(h)
        valid = enc.attention_mask.sum(-1)
        from transformers.activations import ACT2FN
        return pk.greedy_tdt(f, valid, pk.lstm_decoder(m.decoder), m.joint.head, ACT2FN[m.config.hidden_act],
                             blank=BLANK_T, durations=DUR, k_tdt=k, max_symbols=max_symbols,
                             hard_cap=11 * h.shape[1] + 1), valid


def hf_rows(out, valid):
    """Per row: (tokens incl. blank, durations) over the steps taken while the row was active."""
    seqs, durs = out.sequences[:, 1:], out.durations[:, 1:]
    rows = []
    for b in range(seqs.shape[0]):
        fr, toks, ds = 0, [], []
        for s in range(seqs.shape[1]):
            if fr >= int(valid[b]):
                break
            toks.append(int(seqs[b, s]))
            ds.append(int(durs[b, s]))
            fr += int(durs[b, s])
        rows.append((toks, ds))
    return rows


def test_greedy_tdt_without_guard_equals_generate():
    m = tiny_tdt()
    g = torch.Generator().manual_seed(1)
    n_tok = 0
    for batch in range(10):
        B = 1 + batch % 4
        L = 160 + 40 * batch
        feats = torch.randn(B, L, 16, generator=g)
        lens = torch.randint(L // 2, L + 1, (B,), generator=g)
        lens[0] = L
        am = (torch.arange(L)[None] < lens[:, None]).long()
        with torch.no_grad():
            out = m.generate(input_features=feats, attention_mask=am)
        ours, valid = run_ours(m, feats, am)
        for b, (toks, ds) in enumerate(hf_rows(out, valid)):
            col0 = ours["tdt_topk_idx"][b][:, 0].tolist()
            assert col0 == toks, (batch, b)
            assert ours["step_dur"][b].tolist() == ds, (batch, b)
            assert ours["tokens"][b].tolist() == [t for t in toks if t != BLANK_T]
            assert not ours["step_forced"][b].any() and not ours["truncated"][b]
            fr = np.concatenate([[0], np.cumsum(ours["step_dur"][b])[:-1]])
            assert ours["step_frame"][b].tolist() == fr.tolist()
            n_tok += len(ours["tokens"][b])
    assert n_tok > 20                                  # the joint bias makes rows actually emit


def test_guard_bounds_symbols_per_frame():
    m = tiny_tdt()
    with torch.no_grad():                              # always a non-blank token with duration 0
        m.joint.head.weight.zero_()
        m.joint.head.bias.fill_(-10.0)
        m.joint.head.bias[3] = 10.0
        m.joint.head.bias[V] = 10.0
    feats = torch.randn(2, 64, 16)
    am = torch.ones(2, 64, dtype=torch.long)
    ours, valid = run_ours(m, feats, am, max_symbols=10)
    for b in range(2):
        fr, forced, dur = ours["step_frame"][b], ours["step_forced"][b], ours["step_dur"][b]
        assert not ours["truncated"][b]
        assert np.bincount(fr).max() == 10             # 10 symbols per frame, then forced on
        assert forced.sum() == int(valid[b]) and (dur[forced] == 1).all() and (dur[~forced] == 0).all()
        assert len(ours["tokens"][b]) == 10 * int(valid[b])
    unguarded, _ = run_ours(m, feats, am, max_symbols=None)
    assert all(unguarded["truncated"])                 # without the guard only hard_cap stops it


def test_topk_column_zero_is_the_emitted_token_and_sorted():
    m = tiny_tdt(3)
    feats = torch.randn(2, 200, 16)
    ours, _ = run_ours(m, feats, torch.ones(2, 200, dtype=torch.long), max_symbols=10, k=5)
    for b in range(2):
        lp = ours["tdt_topk_lp"][b]
        assert lp.shape[1] == 5 and (np.diff(lp, axis=1) <= 1e-6).all()
        assert np.allclose(np.exp(ours["tdt_dur_lp"][b]).sum(1), 1, atol=1e-5)


def test_ctc_head_linear_equals_conv1d():
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(32, 13, 1)
    lin = pk.ctc_linear(conv.weight.detach(), conv.bias.detach())
    h = torch.randn(3, 20, 32)
    with torch.no_grad():
        a = torch.log_softmax(conv(h.transpose(1, 2)).transpose(1, 2), -1)
        b = torch.log_softmax(lin(h), -1)
    assert torch.allclose(a, b, atol=1e-5)


def test_ctc_targets_dense_compaction():
    Vc = 6
    lp = torch.full((2, 5, Vc), -20.0)
    lp[..., Vc - 1] = 0.0                              # blank-certain frames
    lp[0, 1] = torch.log_softmax(torch.tensor([3.0, 0, 0, 0, 0, 0.5]), -1)
    lp[0, 2] = torch.log_softmax(torch.tensor([3.0, 0, 0, 0, 0, 0.5]), -1)
    lp[0, 4] = torch.log_softmax(torch.tensor([0, 4.0, 0, 0, 0, 0]), -1)
    lp[1, 3] = torch.log_softmax(torch.tensor([0, 0, 5.0, 0, 0, 0]), -1)   # beyond valid length 3
    out = pk.ctc_targets(lp, torch.tensor([5, 3]), k_ctc=3, dense_thr=0.95)
    assert out["ctc_dense_frame"][0].tolist() == [1, 2, 4]
    assert out["ctc_dense_frame"][1].tolist() == []
    assert out["ctc_topk_idx"][0].shape == (3, 3) and out["ctc_topk_idx"][1].shape == (0, 3)
    assert out["ctc_topk_idx"][0][:, 0].tolist() == [0, 0, 1]
    assert len(out["ctc_blank_lp"][0]) == 5 and len(out["ctc_blank_lp"][1]) == 3
    assert out["ctc_tokens"][0].tolist() == [0, 1] and out["ctc_tokens"][1].tolist() == []
    assert pk.ctc_collapse([5, 1, 1, 5, 1, 2, 2], blank=5) == [1, 1, 2]


def test_verify_model_dir_flags_a_mismatch(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_bytes(b"{}")
    (tmp_path / "b.bin").write_bytes(b"xyz")
    monkeypatch.setattr(pk, "PARAKEET_FILES", {"a.json": pk.file_sha256(tmp_path / "a.json"),
                                               "b.bin": "0" * 64, "c.txt": "1" * 64})
    probs = pk.verify_model_dir(tmp_path)
    assert len(probs) == 2 and any("b.bin" in p for p in probs) and any("c.txt: missing" in p for p in probs)
    monkeypatch.setattr(pk, "PARAKEET_FILES", {"a.json": pk.file_sha256(tmp_path / "a.json")})
    assert pk.verify_model_dir(tmp_path) == []


def test_pins_cover_the_eight_files():
    assert sorted(pk.PARAKEET_FILES) == sorted(["config.json", "ctc_head.safetensors", "generation_config.json",
                                                "kitsune_model.json", "model.safetensors", "processor_config.json",
                                                "tokenizer.json", "tokenizer_config.json"])
    assert all(len(v) == 64 for v in pk.PARAKEET_FILES.values())


def test_publish_head_extraction_and_deterministic_files(tmp_path):
    import publish_parakeet as pp
    torch.manual_seed(0)
    state = {pp.HEAD_KEYS[0]: torch.randn(pk.VOCAB, 1024, 1, dtype=torch.float16),
             pp.HEAD_KEYS[1]: torch.randn(pk.VOCAB, dtype=torch.float16), "other": torch.zeros(1)}
    head = pp.extract_head(state)
    assert head["weight"].shape == (pk.VOCAB, 1024) and head["bias"].shape == (pk.VOCAB,)
    assert head["weight"].dtype == torch.float32
    pp.write_head(head, tmp_path / "h1.safetensors")
    pp.write_head(pp.extract_head(state), tmp_path / "h2.safetensors")
    assert (tmp_path / "h1.safetensors").read_bytes() == (tmp_path / "h2.safetensors").read_bytes()
    with pytest.raises(SystemExit):
        pp.extract_head({pp.HEAD_KEYS[0]: torch.zeros(10, 1024, 1), pp.HEAD_KEYS[1]: torch.zeros(10)})
    for name in pp.HF_FILES:
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / pk.CTC_HEAD_FILE).write_bytes((tmp_path / "h1.safetensors").read_bytes())
    pp.write_json(tmp_path / "m1.json", pp.model_json(tmp_path, "abc"))
    pp.write_json(tmp_path / "m2.json", pp.model_json(tmp_path, "abc"))
    assert (tmp_path / "m1.json").read_bytes() == (tmp_path / "m2.json").read_bytes()
    mj = json.loads((tmp_path / "m1.json").read_text(encoding="utf-8"))
    assert mj["source_revision"] == pk.NEMO_REVISION and len(mj["files"]) == 7
    assert "timestamp" not in json.dumps(mj)


def test_publish_refuses_a_wrong_config():
    import publish_parakeet as pp
    cfg = {"model_type": "parakeet_tdt", "vocab_size": 3073, "blank_token_id": 3072, "pad_token_id": 3072,
           "durations": [0, 1, 2, 3, 4], "max_symbols_per_step": 10,
           "encoder_config": {"hidden_size": 1024, "num_hidden_layers": 24, "subsampling_factor": 8}}
    gen = {"decoder_start_token_id": 3072}
    assert pp.config_problems(cfg, gen) == []
    bad = dict(cfg, pad_token_id=0)
    assert any("pad_token_id" in p for p in pp.config_problems(bad, gen))
    assert pp.config_problems(cfg, {"decoder_start_token_id": 0})
    assert pp.nemo_snapshot().name == pk.NEMO_REVISION


def test_golden_file_shape():
    p = ROOT / "kitsune" / "parakeet_golden.json"
    g = json.loads(p.read_text(encoding="utf-8"))
    assert len(g["ids"]) == len(g["tdt"]) == len(g["ctc"]) == 32
    assert g["files_sha256"] == pk.PARAKEET_FILES
    assert g["settings"]["max_symbols"] == 10
