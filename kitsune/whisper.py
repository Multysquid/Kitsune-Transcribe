"""Whisper yardsticks for the full-data runs (plan v3 decision 24; the build contract, section 9): four pinned Whisper
models scored on the study's frozen manifest with the students' own decode rules, so their M4 / M4-all and their speed
sit next to the students' and the teachers' in tools/full_report.py.

Models (WHISPER_MODELS; the system name is the key, the tables' and the speed record's name):
  whisper-large-v3        openai/whisper-large-v3 @ 06f233fe (scripts/02b_second_opinion.py's WHISPER_TOK pin)
  whisper-large-v3-turbo  openai/whisper-large-v3-turbo @ 41f01f3f (4 decoder layers)
  kotoba-whisper-v2.0     kotoba-tech/kotoba-whisper-v2.0 @ 7eb57527 (02b's MODEL2 pin: the model that chose the
                          Galgame-neutral rows, so its M4 is biased low by construction; the report prints M4-all)
  whisper-small           openai/whisper-small @ 973afd24 (pinned from the Hub when this module was written; 80 mel
                          bins: the loader takes the feature extractor of each model's own snapshot, never 128)
Every Hub read is pinned, like the datasets and the teachers: a repo that moves between smoke B and box 2 would score a
different model under the same name. fetch downloads only ALLOW_PATTERNS (the single fp16/bf16 model.safetensors and
the processor files): the large-v3 repo also holds fp32 shards, flax and .bin copies, several GB nobody reads.

The decode is the AED students' (kitsune.evaluate.greedy_generate), transposed to Whisper's generate:
  prompt     SOT, <|ja|>, <|transcribe|>, <|notimestamps|> (PROMPT_LEN 4): language forced (no language-ID errors on
             short clips), no timestamps, no initial prompt (it makes Whisper hallucinate); greedy (num_beams 1, no
             sampling), temperature None and no fallback thresholds (a tuple of temperatures plus a threshold would turn
             the long-form fallback on; a no-speech threshold would skip rows)
  max_new    min(int(16 + 10 x the batch's longest seconds), max_target_positions - PROMPT_LEN - 1) = 443 at 448: the
             students' rule; generate refuses max_new + PROMPT_LEN > max_target_positions
  stop       kitsune.generation.RepetitionStop(prompt_len=4) (Whisper's generate hands the criteria the full decoder
             ids, prompt included), recorded per row (RecordingRepetitionStop): every Whisper generation config sets
             pad_token_id = eos_token_id (50257), so greedy_generate's "first EOS or pad" rule would call a row the
             stop cut EOS-ended; split_generated tells the three ends apart (eos / repetition / length)
  one pass   force_unique_generate_call=True: in no-timestamp mode transformers 5.13.1 still parses two consecutive
             ids >= no_timestamps_token_id + 1 as a timestamp pair and re-decodes from a new seek, which loops for
             minutes on a model that emits them; with it, generate is one call and returns the prompt followed by the
             generated ids (asserted on every batch: API drift refuses). Timestamp ids are stripped and counted
  precision  the snapshot's weights in bf16 (fp32 on CPU), SDPA attention, and the LM head (proj_out) in fp32
             (kitsune.evaluate._fp32_head, the AED students' rule: a bf16 head moves argmax on near-ties; the build's
             decision C15, --no-fp32-head in the tools turns it off)
  features   WhisperFeatureExtractor on the device: every clip padded to 30 s (the encoder refuses any other length and
             ignores its attention mask, so a batch's make-up changes only bf16 numerics)
  batches    whisper_batches: the eval's own duration-sorted batches (kitsune.trainset.eval_batches, batch_s padded
             seconds), each cut into consecutive pieces of at most max_rows rows. Every clip costs a 30 s encoder
             window, so 400 s of short clips (a 273-row JSUT batch) would not fit: the row cap (64 for large-v3, 128 for
             the others) keeps the batches the students' while bounding the memory

Audio and references come from the eval TOKEN store (<cache>/eval, kitsune.trainset.load_stores): the same 16 kHz decode
every student sees and the same reference (kitsune.study_stats.build_corpus refuses a system whose reference lengths
differ). tools/whisper_eval.py runs the CER pass and writes greedy_<set>.parquet + whisper.json (WHISPER_META);
scripts/05_evaluate.py --from-evals tables it as family "whisper" (no teacher); tools/speed_probe.py --kind whisper
times the very functions below (features, greedy_whisper, decode_texts).
"""
import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from kitsune.generation import RepetitionStop

PROMPT_LEN = 4  # SOT, <|ja|>, <|transcribe|>, <|notimestamps|> (asserted on every batch)
LANGUAGE, TASK = "ja", "transcribe"
WHISPER_META = "whisper.json"  # tools/whisper_eval.py's record; scripts/05_evaluate.py evals_family keys on it
FAMILY = "whisper"  # study.json's family of a Whisper system (it has no teacher: kitsune.evaluate.FAMILY_TEACHER)
SAMPLE_RATE = 16000
# the snapshot files a Whisper eval reads: one weights file (fp16 / bf16), the configs and the tokenizer's files; no
# fp32 shards, flax or .bin copies
ALLOW_PATTERNS = ("config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
                  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json",
                  "added_tokens.json", "special_tokens_map.json")
REQUIRED_FILES = ("config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
                  "tokenizer_config.json")  # a snapshot without one of these is refused, never loaded half
DEFAULT_MAX_ROWS = 64  # the row cap of a model dir outside WHISPER_MODELS: large-v3's, the smallest (safest) one
LOAD_BEAT_MAX_S = 3600  # heartbeat.beating's bound around a download + load (the contract's section 9)
REPETITION = dict(window=24, max_period=12)  # kitsune.generation.RepetitionStop's defaults, recorded in whisper.json
FALLBACK_THRESHOLDS = ("logprob_threshold", "compression_ratio_threshold", "no_speech_threshold")  # load_whisper: off


@dataclass(frozen=True)
class WhisperSpec:
    system: str  # the system name in the tables, whisper.json and the speed records (= the WHISPER_MODELS key)
    repo: str
    revision: str  # a 40-hex commit: every read is pinned
    max_rows: int  # the row cap of a CER / speed batch (whisper_batches, speed_probe's batched pass)
    licence: str  # the model card's licence metadata
    display: str


WHISPER_MODELS: dict[str, WhisperSpec] = {s.system: s for s in (
    WhisperSpec("whisper-large-v3", "openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", 64,
                "Apache-2.0", "Whisper large-v3"),
    WhisperSpec("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
                128, "MIT", "Whisper large-v3 turbo"),
    WhisperSpec("kotoba-whisper-v2.0", "kotoba-tech/kotoba-whisper-v2.0", "7eb575277d18909a4af8a24e3ae8cce2e99794ae",
                128, "Apache-2.0", "Kotoba-Whisper v2.0"),
    # the owner's yes (2026-09-27); revision = HfApi().model_info("openai/whisper-small").sha read that day
    WhisperSpec("whisper-small", "openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", 128,
                "Apache-2.0", "Whisper small"),
)}


def max_new_tokens(longest_s: float, max_target_positions: int, prompt_len: int = PROMPT_LEN) -> int:
    """The batch's decode cap: kitsune.evaluate.greedy_generate's rule min(int(16 + 10 s), positions - prompt - 1) for
    the longest row's seconds (443 at Whisper's 448 positions; generate itself allows one more)."""
    return min(int(16 + 10 * float(longest_s)), int(max_target_positions) - int(prompt_len) - 1)


# ------------------------------------------------------------------------------------------------ the snapshots


def fetch(spec: WhisperSpec, cache_dir=None) -> Path:
    """The pinned snapshot's dir (huggingface_hub.snapshot_download of spec.repo @ spec.revision, ALLOW_PATTERNS only,
    into cache_dir: the box's <root>/cache/hf; None = the HF default cache). A cached snapshot is used as it is (also
    with HF_HUB_OFFLINE=1). Raises when the snapshot lacks a REQUIRED_FILES entry."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=spec.repo, revision=spec.revision, allow_patterns=list(ALLOW_PATTERNS),
                                  cache_dir=str(cache_dir) if cache_dir else None))
    if missing := [f for f in REQUIRED_FILES if not (path / f).is_file()]:
        raise RuntimeError(f"{spec.repo}@{spec.revision}: the snapshot at {path} lacks {missing}")
    return path


def resolve(model: str, cache_dir=None) -> tuple[Path, WhisperSpec | None]:
    """--model of the tools: a WHISPER_MODELS key -> (its fetched snapshot, its spec); an existing model dir -> (the
    dir, None). Anything else raises ValueError."""
    if model in WHISPER_MODELS:
        spec = WHISPER_MODELS[model]
        return fetch(spec, cache_dir), spec
    p = Path(model)
    if p.is_dir() and (p / "config.json").is_file():
        return p, None
    raise ValueError(f"--model {model!r} is neither a Whisper key ({', '.join(WHISPER_MODELS)}) nor a model dir")


# ------------------------------------------------------------------------------------------------ the model


@dataclass
class WhisperModel:
    model: object  # a transformers WhisperForConditionalGeneration, eval mode, on `device` in `dtype`
    feature_extractor: object
    tokenizer: object
    prompt_ids: list[int]
    eos: int
    timestamp_begin: int  # no_timestamps_token_id + 1: ids from here on are timestamps
    n_mels: int
    params_total: int
    weights_file_bytes: int
    spec: WhisperSpec | None
    path: Path
    device: torch.device
    dtype: torch.dtype


def weights_file_bytes(path) -> int:
    """The weights file's bytes on disk: model.safetensors, else every *.safetensors of the dir."""
    p = Path(path)
    if (p / "model.safetensors").is_file():
        return int((p / "model.safetensors").stat().st_size)
    return int(sum(f.stat().st_size for f in p.glob("*.safetensors")))


def load_whisper(path, device, dtype: torch.dtype, *, spec: WhisperSpec | None = None) -> WhisperModel:
    """A snapshot dir as a WhisperModel: the weights in `dtype` with SDPA attention (local files only: fetch first),
    the feature extractor and tokenizer of the same dir (the model's own n_mels: small has 80), and the ids the decode
    needs from its generation config - the prompt (decoder_start, <|ja|>, transcribe, no_timestamps), EOS and the first
    timestamp id. Refuses a model whose mel bins differ from its feature extractor's."""
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

    path, device = Path(path), torch.device(device)
    model = WhisperForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="sdpa",
                                                            local_files_only=True).eval().to(device)
    fe = WhisperFeatureExtractor.from_pretrained(path, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if int(fe.feature_size) != int(model.config.num_mel_bins):
        raise ValueError(f"{path}: the feature extractor gives {fe.feature_size} mel bins, the model takes "
                         f"{model.config.num_mel_bins}")
    gc = model.generation_config
    lang = getattr(gc, "lang_to_id", None) or {}
    task = getattr(gc, "task_to_id", None) or {}
    nts = getattr(gc, "no_timestamps_token_id", None)
    if f"<|{LANGUAGE}|>" not in lang or TASK not in task or nts is None or gc.decoder_start_token_id is None:
        raise ValueError(f"{path}: its generation config lacks the Japanese prompt ids (lang_to_id <|{LANGUAGE}|>, "
                         f"task_to_id {TASK}, no_timestamps_token_id, decoder_start_token_id)")
    eos = gc.eos_token_id
    eos = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
    # the long-form fallback off on the model's own config: transformers fills every None of a generate call's config
    # from this one, so a threshold a snapshot sets would come back (a no-speech threshold skips rows)
    for k in FALLBACK_THRESHOLDS:
        setattr(gc, k, None)
    return WhisperModel(model=model, feature_extractor=fe, tokenizer=tok,
                        prompt_ids=[int(gc.decoder_start_token_id), int(lang[f"<|{LANGUAGE}|>"]), int(task[TASK]),
                                    int(nts)],
                        eos=eos, timestamp_begin=int(nts) + 1, n_mels=int(fe.feature_size),
                        params_total=int(sum(p.numel() for p in model.parameters())),
                        weights_file_bytes=weights_file_bytes(path), spec=spec, path=path, device=device, dtype=dtype)


def features(wm: WhisperModel, waves: Sequence[np.ndarray]) -> torch.Tensor:
    """(B, n_mels, 3000) log-mel features of 16 kHz waveforms, computed on the model's device, in its dtype."""
    f = wm.feature_extractor(list(waves), sampling_rate=SAMPLE_RATE, return_tensors="pt", device=str(wm.device),
                             return_attention_mask=False)["input_features"]
    return f.to(wm.device, wm.dtype)


# ------------------------------------------------------------------------------------------------ the decode


class RecordingRepetitionStop(RepetitionStop):
    """kitsune.generation.RepetitionStop that also records, per batch row, the generated length at which it first
    fired (fired(): {row: length}). The record stays on the device during the decode (no host sync per step: the speed
    probe times this criterion). A row that ended with EOS gets padded (pad == EOS in every Whisper config), and its
    padding soon looks periodic too: split_generated keeps the EOS when it comes first."""

    def __init__(self, prompt_len: int = PROMPT_LEN, window: int = REPETITION["window"],
                 max_period: int = REPETITION["max_period"]):
        super().__init__(prompt_len, window, max_period)
        self._fired: torch.Tensor | None = None

    def __call__(self, input_ids, scores, **kwargs):
        done = super().__call__(input_ids, scores, **kwargs)
        n = input_ids.shape[1] - self.prompt_len
        if self._fired is None:
            self._fired = torch.full((input_ids.shape[0],), -1, dtype=torch.long, device=input_ids.device)
        self._fired = torch.where(done & (self._fired < 0), torch.full_like(self._fired, n), self._fired)
        return done

    def fired(self) -> dict[int, int]:
        if self._fired is None:
            return {}
        return {r: int(v) for r, v in enumerate(self._fired.tolist()) if v >= 0}


def split_generated(gen: Sequence[int], *, eos: int, timestamp_begin: int, fired_at: int | None,
                    max_new: int) -> tuple[list[int], list[int], str, int]:
    """One row's generated ids (after the prompt) -> (hyp_ids, text_ids, stop, n_timestamp_tokens).

    stop: "eos" when EOS comes before the repetition stop fired (or it never fired), "repetition" when the stop fired
    first (the row is padded with pad == EOS after it: not an EOS end), "length" when neither (the row ran to max_new).
    hyp_ids: greedy_generate's convention - the ids up to and including EOS when it ended, the ids up to the stop
    otherwise (its n_tok is their count). text_ids: the same without EOS and without timestamp ids (>= timestamp_begin),
    whose count is n_timestamp_tokens."""
    gen = [int(t) for t in gen]
    e = next((i for i, t in enumerate(gen) if t == eos), None)
    if e is not None and (fired_at is None or e < fired_at):
        body, hyp, stop = gen[:e], gen[:e + 1], "eos"
    elif fired_at is not None:
        body = hyp = gen[:fired_at]
        stop = "repetition"
    else:
        body = hyp = gen[:max_new]
        stop = "length"
    text = [t for t in body if t < timestamp_begin]
    return list(hyp), text, stop, len(body) - len(text)


class _DropLengthWarning(logging.Filter):
    """Drops transformers' "Both `max_new_tokens` and `max_length` seem to have been set" warning: every Whisper
    generation config sets max_length 448 and every batch passes its own max_new_tokens (which wins, as intended), so
    it would print once per batch - thousands of lines in a box log that should show the real problems."""

    def filter(self, record):
        return not str(record.getMessage()).startswith("Both `max_new_tokens`")


_LENGTH_WARNING = _DropLengthWarning()


@contextmanager
def _quiet_length_warning():
    lg = logging.getLogger("transformers.generation.utils")
    lg.addFilter(_LENGTH_WARNING)
    try:
        yield
    finally:
        lg.removeFilter(_LENGTH_WARNING)


def generation_config(wm: WhisperModel, max_new: int):
    """One call's generation config: the model's (the published one, its fallback thresholds off since load_whisper)
    with the greedy settings and this batch's max_new_tokens. They go in the config, not as generate() kwargs:
    transformers 5.13 deprecates passing both. temperature None (a single pass: no fallback) is Whisper's own
    argument."""
    gc = copy.deepcopy(wm.model.generation_config)
    gc.max_new_tokens, gc.do_sample, gc.num_beams = int(max_new), False, 1
    for k in FALLBACK_THRESHOLDS:
        setattr(gc, k, None)
    return gc


@torch.no_grad()
def greedy_whisper(wm: WhisperModel, feats: torch.Tensor, longest_s: float, *, fp32_head: bool = True) -> list[dict]:
    """One batch's greedy decode (module docstring): per row dict(hyp_ids, text_ids, stop, truncated = stop != "eos",
    n_timestamp_tokens, max_new). generate is called once (force_unique_generate_call); its output must start with the
    prompt ids, else RuntimeError (the transformers API drifted). fp32_head: the LM head in fp32 for the call (False
    when the caller already holds kitsune.evaluate._fp32_head, as the speed probe's context does)."""
    from contextlib import nullcontext

    from transformers.generation import StoppingCriteriaList

    from kitsune.evaluate import _fp32_head

    mx = max_new_tokens(longest_s, wm.model.config.max_target_positions)
    stop = RecordingRepetitionStop(PROMPT_LEN)
    with _fp32_head(wm.model) if fp32_head else nullcontext(), _quiet_length_warning():
        seq = wm.model.generate(input_features=feats, generation_config=generation_config(wm, mx), language=LANGUAGE,
                                task=TASK, return_timestamps=False, temperature=None,
                                stopping_criteria=StoppingCriteriaList([stop]), force_unique_generate_call=True)
    seq = (seq.sequences if hasattr(seq, "sequences") else seq).cpu()
    want = torch.tensor(wm.prompt_ids, dtype=seq.dtype)
    if seq.ndim != 2 or seq.shape[0] != feats.shape[0] or seq.shape[1] < PROMPT_LEN or not bool(
            (seq[:, :PROMPT_LEN] == want).all()):
        raise RuntimeError(f"Whisper generate returned {tuple(seq.shape)} ids not starting with the prompt "
                           f"{wm.prompt_ids} (first row: {seq[0, :PROMPT_LEN].tolist() if seq.numel() else []}): the "
                           "transformers API changed; the decode refuses to guess")
    fired = stop.fired()
    rows = []
    for r, g in enumerate(seq[:, PROMPT_LEN:].tolist()):
        hyp_ids, text_ids, kind, n_ts = split_generated(g, eos=wm.eos, timestamp_begin=wm.timestamp_begin,
                                                        fired_at=fired.get(r), max_new=mx)
        rows.append(dict(hyp_ids=hyp_ids, text_ids=text_ids, stop=kind, truncated=kind != "eos",
                         n_timestamp_tokens=n_ts, max_new=mx))
    return rows


def decode_texts(wm: WhisperModel, rows: list[dict]) -> list[str]:
    """The rows' texts: their text_ids detokenised without special tokens, stripped (the students' texts go through
    batch_decode the same way; kitsune.text.normalize_ja drops the rest before scoring)."""
    return [t.strip() for t in wm.tokenizer.batch_decode([r["text_ids"] for r in rows], skip_special_tokens=True)]


def decode_record(wm: WhisperModel) -> dict:
    """whisper.json's decode block: every setting the decode above runs with."""
    return dict(language=LANGUAGE, task=TASK, num_beams=1, do_sample=False, temperature=None, return_timestamps=False,
                force_unique_generate_call=True, fallback_thresholds=None,
                repetition_stop=dict(prompt_len=PROMPT_LEN, **REPETITION),
                max_new_rule=f"min(int(16 + 10 * longest_s), max_target_positions - {PROMPT_LEN} - 1)",
                max_target_positions=int(wm.model.config.max_target_positions), prompt_ids=list(wm.prompt_ids),
                eos=wm.eos, timestamp_begin=wm.timestamp_begin)


def model_record(wm: WhisperModel, fp32_head: bool) -> dict:
    """whisper.json's model block."""
    s = wm.spec
    names = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}  # the tools' --dtype names
    return dict(repo=s.repo if s else None, revision=s.revision if s else None, path=str(wm.path),
                dtype=names.get(wm.dtype, str(wm.dtype)), attn="sdpa", fp32_head=bool(fp32_head),
                params_total=wm.params_total, weights_file_bytes=wm.weights_file_bytes, n_mels=wm.n_mels,
                licence=s.licence if s else None, display=s.display if s else None)


# ------------------------------------------------------------------------------------------------ batching


def whisper_batches(utts, batch_s: float, max_rows: int | None, indices: Sequence[int] | None = None
                    ) -> list[list[int]]:
    """The eval's own batches (kitsune.trainset.eval_batches: duration-sorted, at most batch_s padded seconds) over
    `indices`, each cut into consecutive pieces of at most max_rows rows (None or 0: uncut). A 273-row batch at cap 64
    becomes 64, 64, 64, 64, 17 in the same order."""
    from kitsune.trainset import eval_batches

    return split_rows(eval_batches(utts, batch_s, indices), max_rows)


def split_rows(batches: list[list[int]], max_rows: int | None) -> list[list[int]]:
    """Each batch cut into consecutive pieces of at most max_rows rows (None or 0: as they are)."""
    if not max_rows:
        return [list(b) for b in batches]
    m = int(max_rows)
    return [list(b[i:i + m]) for b in batches for i in range(0, len(b), m)]
