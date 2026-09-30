"""Tiny offline Whisper models and a small eval store for the Whisper tests (kitsune/whisper.py, tools/whisper_eval.py,
tools/speed_probe.py --kind whisper). Nothing is downloaded: the model, its feature extractor and a word-level
tokenizer are built in memory and saved as a snapshot dir the loaders read with local_files_only.

    from fixtures_whisper import tiny_whisper_dir, whisper_env
    d = tiny_whisper_dir(tmp_path / "tiny")            # config.json, generation_config.json, model.safetensors, ...
    env = whisper_env(tmp_path / "env")                 # eval store + study manifest (Galgame views) + pending PREREG

The tiny model keeps the traps of the real ones (kitsune/whisper.py's docstring):
  pad_token_id == eos_token_id (2), as in every published Whisper generation config, so a row the repetition stop cut
  is padded with EOS; the prompt is decoder_start (1), <|ja|> (5), <|transcribe|> (6), <|notimestamps|> (V - 1), and
  the first timestamp id is V (out of the vocabulary), so no seek loop can start; max_target_positions 64, so the
  decode cap is 59 and generate refuses more than 60 new tokens; 1500 source positions, since the encoder takes exactly
  3,000 feature frames (30 s) whatever the clip's length.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fixtures import make_fake_corpus, make_fake_selection  # noqa: E402

V = 64
SOT, EOS, PAD, JA, TRANSCRIBE, TRANSLATE = 1, 2, 2, 5, 6, 7
NO_TS = V - 1
PROMPT = [SOT, JA, TRANSCRIBE, NO_TS]
MAX_TARGET = 64
SPECIAL = ["<unk>", "<|startoftranscript|>", "<|endoftext|>", "<|en|>", "<|de|>", "<|ja|>", "<|transcribe|>",
           "<|translate|>"]
_KANA = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")
EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "galgame"]


def tiny_tokenizer():
    """A fast word-level tokenizer over the special tokens, kana and fillers (V ids): decode concatenates the tokens
    (a Fuse decoder, no spaces) and skip_special_tokens drops the specials and <|notimestamps|>."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(SPECIAL)}
    for c in _KANA:
        if len(vocab) >= V - 1:
            break
        vocab[c] = len(vocab)
    while len(vocab) < V - 1:
        vocab[f"<x{len(vocab)}>"] = len(vocab)
    vocab["<|notimestamps|>"] = NO_TS
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>", bos_token="<|startoftranscript|>",
                                   additional_special_tokens=[*SPECIAL[3:], "<|notimestamps|>"])


def tiny_whisper(seed: int = 0, *, n_mels: int = 128):
    """(model, feature extractor, tokenizer): a random 1+1-layer WhisperForConditionalGeneration (d 16) with a
    Whisper-shaped generation config (lang_to_id, task_to_id, no_timestamps_token_id, is_multilingual)."""
    import torch
    from transformers import GenerationConfig, WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

    cfg = WhisperConfig(vocab_size=V, num_mel_bins=n_mels, encoder_layers=1, decoder_layers=1, d_model=16,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=1500, max_target_positions=MAX_TARGET, decoder_start_token_id=SOT,
                        eos_token_id=EOS, pad_token_id=PAD, bos_token_id=SOT, begin_suppress_tokens=None,
                        suppress_tokens=None)
    torch.manual_seed(seed)
    model = WhisperForConditionalGeneration(cfg).eval()
    # as the published configs: the deprecated forced_decoder_ids (language detection, transcribe), which language= /
    # task= override; and the long-form fallback thresholds of OpenAI's reference decoding, which load_whisper turns
    # off (with them a row whose log-prob is low would be skipped)
    model.generation_config = GenerationConfig(
        decoder_start_token_id=SOT, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=SOT,
        lang_to_id={"<|en|>": 3, "<|de|>": 4, "<|ja|>": JA}, task_to_id={"transcribe": TRANSCRIBE,
                                                                        "translate": TRANSLATE},
        no_timestamps_token_id=NO_TS, is_multilingual=True, max_length=MAX_TARGET, return_timestamps=False,
        begin_suppress_tokens=None, suppress_tokens=None, forced_decoder_ids=[[1, None], [2, TRANSCRIBE], [3, NO_TS]],
        no_speech_threshold=0.6, logprob_threshold=-1.0, compression_ratio_threshold=2.4)
    return model, WhisperFeatureExtractor(feature_size=n_mels), tiny_tokenizer()


def tiny_whisper_dir(d, seed: int = 0, *, n_mels: int = 128) -> Path:
    """tiny_whisper saved as a snapshot dir (save_pretrained of the three), as kitsune.whisper.fetch leaves one."""
    d = Path(d)
    model, fe, tok = tiny_whisper(seed, n_mels=n_mels)
    model.save_pretrained(d)
    fe.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def pending_prereg(path) -> Path:
    """PREREG.json with the rules only (the manifest block pending): the toy manifests below are not the study's, so
    05's manifest check against the committed (filled) PREREG.json would refuse them."""
    from kitsune import prereg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prereg.rules_json(prereg.rules()))
    return path


def whisper_env(root) -> dict:
    """A corpus (a train source for the selection, the three gate sets and a Galgame hold-out; clips of 0.4-2 s), its
    selection, the eval TOKEN store of the four sets, the study manifest of their kept rows with Galgame views, and a
    pending PREREG.json."""
    import pandas as pd

    from kitsune import trainset
    from kitsune.store import ids_sha256

    root = Path(root)
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (4, "train"), "eval_jsut": (7, "eval"),
                                                    "eval_cv8": (5, "eval"), "eval_reazon": (5, "eval"),
                                                    "galgame": (6, "eval")},
                          dur_range=(0.4, 2.0), token_range=(256, 296), no_second=tuple(EVAL_SETS), seed=17)
    sel = make_fake_selection(fc, greedy_n=3, probe_n=2)
    store_dir = root / "cache" / "eval"
    store = trainset.eval_store(sel, fc.data, fc.teacher_out, store_dir, EVAL_SETS)
    s = pd.read_parquet(sel)
    manifest = {"schema": 1, "sets": {}}
    for e in EVAL_SETS:
        ids = s["id"][(s["source"] == e) & (s["split"] == "eval") & s["keep"]].tolist()
        manifest["sets"][e] = {"n": len(ids), "ids_sha256": ids_sha256(ids), "ids": ids}
    gal = manifest["sets"]["galgame"]["ids"]
    manifest["galgame_views"] = {v: {"ids": x, "ids_sha256": ids_sha256(x), "n": len(x)}
                                 for v, x in (("neutral", gal[::2]), ("all", gal), ("label_box", gal[1:]))}
    mpath = root / "study_manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return dict(root=root, fc=fc, sel=sel, store=store, store_dir=store_dir, manifest=manifest, manifest_path=mpath,
                prereg=pending_prereg(root / "PREREG.json"))
