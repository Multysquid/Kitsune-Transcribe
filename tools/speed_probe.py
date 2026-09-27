"""The size study's speed probe (study/STUDY.md 4.2 "Speed" and "VRAM", 4.7 Pareto; decision 28): one model at a time,
a batched RTF and a batch-1 latency (p50 / p95) on a fixed id list, and the peak VRAM, on the study box at the end, on
an idle host (the queue calls it once per system).

Kinds (--kind), each decoded exactly as the study scores it:
  aed           a Transcribe student dir (kitsune.student.load_student): the greedy AED decode with the teacher pass's
                settings (kitsune.evaluate.greedy_generate: num_beams 1, RepetitionStop, max_new 16 + 10 s), the LM head
                in fp32 (kitsune.evaluate._fp32_head), the student's own LogMel; its decode length: --decode-len below
  cohere        the Cohere teacher (CohereLabs/cohere-transcribe-03-2026 at the revision its labels came from, or
                --model), the same decode
  ctc           a Parakeet CTC student dir (kitsune.ctc_student.load_ctc_student): the encoder, the CTC head in fp32,
                greedy CTC (argmax, collapse repeats, drop blanks), detokenised as ctc_hyp was (decode_ids)
  parakeet-ctc  the Parakeet teacher's CTC path: the converted dir's unpruned encoder + ctc_head (load_parakeet_ctc),
                the same decode
  parakeet-tdt  the Parakeet teacher's TDT path: the converted dir's ParakeetForTDT (kitsune.parakeet.ParakeetTeacher's
                modules: encoder bf16 on CUDA, projector, prediction net and joint fp32), greedy TDT with NeMo's
                max-symbols guard (kitsune.parakeet.greedy_tdt, k_tdt 1, max_symbols 10), the processor's batch_decode
                as the label pass wrote `hyp`
Both Parakeet paths and the CTC students take Parakeet's own features (ctc_features: 80 mel, no dither) on the device.
Every encoder (all five kinds are FastConformers) runs with kitsune.patches.patch_relpos_once_per_batch, as the trainer
and the evaluator run it (perf.relpos_patch; --no-relpos-patch times it without, and the record says which).

What the clock covers: from the decoded 16 kHz waveforms in host memory (the FLAC decode and resampling happen before,
off the clock) to the text - host->device copy, log-mel features on the device, the model, the decode loop and
detokenisation - with torch.cuda.synchronize at both ends.
  batched   the id list in the eval's own batches (kitsune.trainset.pack_micro_batches: duration-sorted, at most
            --batch-s padded seconds): rtf = wall seconds / audio seconds
  batch-1   each of the first --latency-n ids alone: p50_s / p95_s / mean_s seconds per utterance (numpy's linear
            percentile), and rtf_1_p50, the median of their per-utterance RTFs
Warm-up, never timed, and no allocator growth inside a clock: before the batched pass its first --warmup batches (the
longest: the plan is duration-sorted, so the caching allocator grows to the pass's size), then only the peak counters
are reset; after it the allocator's cache is emptied (batch-1 measures its own VRAM), the batch-1 warm-up runs - the
longest of the latency ids, then the first --warmup-1 ids (--warmup-1 0: none) - and the peak counters are reset again.
VRAM (CUDA): each pass's peak allocated and reserved bytes are kept, and vram_gb = the larger reserved peak / 1e9, what
the model took from the card at that batch size. On CPU the VRAM fields are null.

AED decode length (--decode-len; aed and cohere kinds): an AED decode's time grows with the tokens it emits, so a model
that does not stop where a trained one would (an untrained init dir: a scratch decoder runs to max_new, a pruned one
derails at step 0) times its weights, not its shape. "teacher" pins every batch to exactly the teacher's token count of
its longest row (the store's n_tok: the stored greedy tokens with EOS; batch-1: the utterance's own; greedy_generate's
pin_new: EOS held off until then, no RepetitionStop), which is what a student that has learnt to stop decodes. "greedy":
the free decode, as the evaluator runs it. "auto" (default): teacher for a student dir without the trainer's "trained"
record in student_meta.json (a study student's init dir, which the study box times as its shape), greedy otherwise (a
trained checkpoint, the Cohere teacher). The record says which (decode_len), whether the weights were trained, and the
tokens decoded per utterance (tokens_per_utt; CTC and TDT: the output tokens).

Numerics (--dtype): auto = bf16 on CUDA, fp32 on CPU. bf16: the weights in bf16 and bf16 autocast (the heads computed
in fp32 as above; the TDT path's encoder in bf16, the rest fp32, as the label box ran it). fp32: fp32 weights, no
autocast. The device, the dtype, the GPU name, the weights' bytes and the parameter count are recorded; a CER against
the store's references on the batched decode (cer_ref_corpus) shows the model decoded sensibly at that dtype.

The id list: --ids FILE (one id per line, or a JSON list), or --per-set N ids of every eval set in the store (a seeded
draw, --seed; sorted by id within a set). The list and its sha256 are written into --out; an --out holding systems
timed on another list refuses (exit 2): every system of one Pareto view is timed on the same audio. Audio comes from a
built store (--store: a kitsune.trainset cache dir, e.g. the eval store <cache_dir>/eval the trainer and 05_evaluate
build).

Idle host: before loading, nvidia-smi lists the GPU's compute processes (and utilisation); --require-idle refuses
(exit 3) while another process is there. What was seen is recorded either way.

Output (--out, JSON, merged, written atomically): {"schema": 1, "ids": [...], "ids_sha256": ..., "systems": {name:
record}}. A record: kind, model, device, gpu, dtype, batch_s, n_utts, audio_s, batches, rtf, wall_s, the batch-1
fields (n_latency, p50_s, p95_s, mean_s, rtf_1_p50), vram_peak_allocated_bytes / _reserved_bytes for each pass,
vram_gb, params_total, weights_bytes, relpos_patch, decode_len, trained, tokens_per_utt, cer_ref_corpus, idle,
versions, time_utc. tools/study_report.py --speed reads it (the Pareto view: rtf, vram_gb, p50_s, p95_s;
kitsune.study_stats.speed_entry).

Quantised variants (the full runs; kitsune.quant, contract section 8; aed and ctc kinds): --quant <fmt> times the
student quantised after its dtype cast (kitsune.quant.apply: torchao's kernels on CUDA; --quant-impl emulate only when
asked, recorded emulated: true and never a speed; mxfp4 has no kernel here and needs it; fp16 keeps the model fp16
under fp16 autocast, dtype "fp16"), then the rel-pos patch. The bf16 export is what is timed: a variant dir as --model
is refused (its weights are the same bits). The system must be <base>@<fmt> (+compile with --compile: the encoder, and
the aed decoder, compiled in place, counters off, --warmup >= 2). --profile-kernels adds an untimed batched and
batch-1 decode under kitsune.quant.kernel_census (kernels: the matmul ops and GEMM kernel classes, the int8 fallbacks);
--threads sets torch's threads before loading; --hyps-out writes the list's hypotheses (id, ref, hyp, hyp_1, duration).
Every record gains quant, quant_impl, quant_scope, mx_rounding, emulated, compile, threads, autocast_dtype,
quant_counters, kernels, weights_bytes_resident and hyp_diff_1 (the latency ids whose batch-1 hypothesis differs from
the batched one: NVFP4 W4A4's whole-call activation scale makes a row depend on its batch-mates); weights_bytes is the
deployable packed bytes for a format. The idle check counts only the processes on the probe's own GPU (by PCI bus
id), so --require-idle works while the box's other GPU trains. The probe beats $KITSUNE_HEARTBEAT once per timed
repeat, and under kitsune.heartbeat.beating (max 1800 s) through the model load and the warm-ups (torch.compile's).

Usage (on the box, at its end, one call per system; kitsune/study_queue.py phase_speed passes these):
  python tools/speed_probe.py --kind aed --model students/study/t03 --system study-t03 \
      --store cache/eval --per-set 40 --out runs/speed-B/speed.json --require-idle
  python tools/speed_probe.py --kind cohere --system cohere --store cache/eval --per-set 40 \
      --out runs/speed-B/speed.json --require-idle
  python tools/speed_probe.py --kind parakeet-tdt --model models/parakeet-tdt_ctc-0.6b-ja-hf --system parakeet-tdt \
      --store cache/eval --per-set 40 --out runs/speed-B/speed.json --require-idle
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kitsune import trainset  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402

SCHEMA = 1
KINDS = ("aed", "cohere", "ctc", "parakeet-ctc", "parakeet-tdt")
AED_KINDS = ("aed", "cohere")  # the autoregressive decodes: their time depends on the tokens they emit
DECODE_LENS = ("auto", "greedy", "teacher")
TEACHER_ID = "CohereLabs/cohere-transcribe-03-2026"
TEACHER_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"  # == scripts/03_build_student.TEACHER_REVISION (tested)
MAX_SYMBOLS = 10  # the label pass's TDT guard (kitsune.parakeet; vast/label.py)
EXIT_REFUSED, EXIT_BUSY = 2, 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ------------------------------------------------------------------------------------------------ the id list


def pick_ids(store, per_set: int, seed: int) -> list[str]:
    """per_set ids of every source (eval set) of the store: a seeded draw without replacement from its ids sorted by
    id (the draw depends on (seed, set name) only), sorted by id; the sets in the store's order."""
    by_set: dict[str, list[str]] = {}
    for u in store.utts:
        by_set.setdefault(u.source, []).append(u.id)
    out = []
    for s, ids in by_set.items():
        ids = sorted(ids)
        rng = np.random.default_rng([int(seed), zlib.crc32(s.encode())])
        pick = rng.choice(len(ids), size=min(int(per_set), len(ids)), replace=False)
        out += sorted(ids[i] for i in pick)
    return out


def read_ids(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return [str(x) for x in json.loads(text)]
    return [x.strip() for x in text.splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------ the host


def _bus_tuple(s) -> tuple[int, ...]:
    """'00000000:01:00.0' / '0000:01:00.0' / '01:00.0' -> (domain, bus, device, function) (as 05_evaluate's)."""
    head, _, fn = str(s).strip().partition(".")
    parts = [int(x, 16) for x in head.split(":")]
    return tuple(([0] * (3 - len(parts)) + parts) + [int(fn or "0", 16)])


def _pci_bus_id(device: torch.device) -> str | None:
    """The PCI bus id of the probe's CUDA device (nvidia-smi numbers GPUs its own way), None when unknown."""
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        return f"{int(p.pci_domain_id):08X}:{int(p.pci_bus_id):02X}:{int(p.pci_device_id):02X}.0"
    except Exception:  # noqa: BLE001
        return None


def gpu_state(device: torch.device) -> dict | None:
    """What nvidia-smi shows before the probe: the compute processes on the probe's GPU (other than this one; matched
    by PCI bus id when both are known, so a 2-GPU box's training on the other GPU is not "busy") and the GPUs'
    utilisation; None on CPU or without nvidia-smi."""
    if device.type != "cuda" or not shutil.which("nvidia-smi"):
        return None
    exe = shutil.which("nvidia-smi")
    bus = _pci_bus_id(device)
    out = dict(processes=[], utilization=None, gpu_bus_id=bus, other_gpus=[])
    try:
        r = subprocess.run([exe, "--query-compute-apps=pid,process_name,used_memory,gpu_bus_id",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30, check=True)
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) != os.getpid():
                proc = dict(pid=int(parts[0]), name=parts[1], used_mib=parts[2])
                if len(parts) >= 4 and parts[3]:
                    proc["gpu_bus_id"] = parts[3]
                    try:
                        if bus and _bus_tuple(parts[3]) != _bus_tuple(bus):
                            out["other_gpus"].append(proc)
                            continue
                    except ValueError:  # an unparsable bus id: counted, as without one
                        pass
                out["processes"].append(proc)
        r = subprocess.run([exe, "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30, check=True)
        out["utilization"] = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    except Exception as e:  # noqa: BLE001 - a probe without the readout still runs; the record says why
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


# ------------------------------------------------------------------------------------------------ the models


def _amp(device: torch.device, dtype: str):
    """A runner's autocast: torch.bfloat16 for bf16 on CUDA, else False (a dtype or False: kitsune.evaluate.amp_dtype;
    a fp16 variant sets torch.float16 after load_runner quantised it)."""
    return torch.bfloat16 if dtype == "bf16" and device.type == "cuda" else False


def _weights_bytes(model) -> int:
    return int(sum(t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()]))


class AedRunner:
    """A Cohere ASR model (a Transcribe student or the teacher): greedy_generate on the eval's LogMel features; with
    pin (decode_len "teacher") every batch decodes exactly its longest row's teacher token count."""

    tokens = 0  # tokens decoded since the last reset (every runner counts them)

    def __init__(self, model, processor, device, dtype: str, prompt, eos: int, pad: int, pin: bool = False):
        from kitsune import evaluate as ev
        from kitsune.features import LogMel

        self.ev, self.device, self.amp, self.pin = ev, device, _amp(device, dtype), bool(pin)
        self.model = model.to(device, dtype=torch.bfloat16 if dtype == "bf16" else torch.float32).eval()
        self.feat = LogMel.from_feature_extractor(processor.feature_extractor).to(device)
        self.tokenizer = processor.tokenizer
        self.prompt, self.eos, self.pad = [int(x) for x in prompt], int(eos), int(pad)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.weights_bytes = _weights_bytes(self.model)

    def context(self):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(torch.inference_mode())
        stack.enter_context(self.ev._eval_mode(self.model))
        stack.enter_context(self.ev._fp32_head(self.model))
        return stack

    def decode(self, waves: list[np.ndarray], durations: list[float], n_tok: list[int]) -> list[str]:
        from kitsune.features import pad_waves

        wave, lengths = pad_waves(waves)
        wave, lengths = wave.to(self.device, non_blocking=True), lengths.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            feats, fmask = self.feat(wave, lengths)
        # max_new from the store's durations, as greedy_eval takes them
        rows = self.ev.greedy_generate(self.model, feats, fmask, max(durations), prompt_ids=self.prompt, eos=self.eos,
                                       pad=self.pad, amp=self.amp, pin_new=max(n_tok) if self.pin else None)
        self.tokens += sum(len(ids) for ids, _ in rows)
        return self.tokenizer.batch_decode([ids for ids, _ in rows], skip_special_tokens=True)


class CtcRunner:
    """A ParakeetForCTC (a CTC student or the teacher's CTC path): encoder, fp32 CTC head, greedy CTC."""

    tokens = 0

    def __init__(self, model, feat_dir, tokenizer, device, dtype: str):
        from kitsune import ctc_student as CS

        self.CS, self.device, self.amp = CS, device, _amp(device, dtype)
        self.model = model.to(device, dtype=torch.bfloat16 if dtype == "bf16" else torch.float32).eval()
        self.feat = CS.ctc_features(feat_dir, device)
        self.tokenizer = tokenizer
        self.params = sum(p.numel() for p in self.model.parameters())
        self.weights_bytes = _weights_bytes(self.model)

    def context(self):
        return torch.inference_mode()

    def decode(self, waves: list[np.ndarray], durations: list[float], n_tok: list[int]) -> list[str]:
        from kitsune.evaluate import amp_dtype

        feats, lens = self.feat(waves)
        mask = self.CS.lengths_to_mask(lens, feats.shape[1])
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype(self.amp), enabled=bool(self.amp)):
            lp, n = self.CS.ctc_log_probs(self.model, feats, mask)
        paths = self.CS.greedy_ctc_ids(lp, n)
        self.tokens += sum(len(x) for x in paths)
        return [self.CS.decode_ids(self.tokenizer, x) for x in paths]


class TdtRunner:
    """The Parakeet teacher's TDT path: ParakeetTeacher's modules (encoder in bf16 on CUDA), greedy TDT with the
    max-symbols guard, its processor's batch_decode."""

    tokens = 0

    def __init__(self, model_dir, device, dtype: str):
        from kitsune import ctc_student as CS
        from kitsune import parakeet as P

        self.CS, self.P, self.device = CS, P, device
        enc_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.teacher = P.ParakeetTeacher(Path(model_dir), device=str(device), encoder_dtype=enc_dtype)
        self.feat = CS.ctc_features(model_dir, device)
        self.enc_dtype = enc_dtype
        m = self.teacher.model
        self.params = sum(p.numel() for p in m.parameters())
        self.weights_bytes = _weights_bytes(m)  # the TDT model; ParakeetTeacher's CTC head is not used here

    def context(self):
        return torch.inference_mode()

    def decode(self, waves: list[np.ndarray], durations: list[float], n_tok: list[int]) -> list[str]:
        P, t = self.P, self.teacher
        feats, lens = self.feat(waves)
        mask = self.CS.lengths_to_mask(lens, feats.shape[1]).long()
        enc = t.model.encoder(input_features=feats.to(self.enc_dtype), attention_mask=mask, output_attention_mask=True)
        h = enc.last_hidden_state.float()
        T = h.shape[1]
        valid = enc.attention_mask.sum(-1).long()
        f = t.model.encoder_projector(h)
        tdt = P.greedy_tdt(f, valid, t.decoder, t.model.joint.head, t.act, blank=P.BLANK, durations=P.DURATIONS,
                           k_tdt=1, max_symbols=MAX_SYMBOLS, hard_cap=(MAX_SYMBOLS + 1) * T + 1)
        seqs = [r.tolist() for r in tdt["tokens"]]
        self.tokens += sum(len(x) for x in seqs)
        return t.processor.batch_decode(seqs, skip_special_tokens=True)


def trained_weights(kind: str, model: str | None) -> bool | None:
    """Whether a student dir holds trained weights: the trainer's "trained" record in its student_meta.json (a
    checkpoint the trainer exported has it, a student init dir from 03 / 03c does not); None for the teachers."""
    if kind not in ("aed", "ctc") or not model:
        return None
    try:
        meta = json.loads((Path(model) / "student_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(meta.get("trained"))


def decode_len_of(kind: str, choice: str, trained: bool | None) -> str | None:
    """The AED decode length a probe runs (module docstring): None for the non-autoregressive kinds; "auto" is
    "teacher" for an untrained student dir, "greedy" otherwise."""
    if kind not in AED_KINDS:
        return None
    if choice != "auto":
        return choice
    return "teacher" if kind == "aed" and trained is False else "greedy"


def load_runner(kind: str, model: str | None, device: torch.device, dtype: str, store, revision: str,
                relpos_patch: bool = True, decode_len: str | None = None, quant: dict | None = None,
                compile: bool = False):
    """The runner of one kind and its model's description; relpos_patch: kitsune.patches.patch_relpos_once_per_batch
    on its encoder, as the trainer and the evaluator run every one of these encoders (perf.relpos_patch); decode_len
    "teacher": an AED runner pinned to the teacher's token counts. quant {fmt, impl, scope, mx_rounding}: the model
    quantised after the runner's dtype cast and before the patch (fp16: the runner keeps fp32 weights, kitsune.quant
    casts them, autocast fp16), runner.quant = the recipe, weights_bytes = the deployable bytes. compile: the encoder
    (and an AED's decoder) compiled in place last, the quant counters off (their increments are Python side effects)."""
    fp16 = bool(quant) and quant["fmt"] == "fp16"
    runner, desc = _runner(kind, model, device, "fp32" if fp16 else dtype, store.info or {}, revision,
                           pin=decode_len == "teacher")
    runner.quant = None
    runner.weights_bytes_resident = runner.weights_bytes
    if quant:
        from kitsune import quant as Q

        runner.quant = Q.apply(runner.model, quant["fmt"], impl=quant.get("impl") or "auto",
                               scope=quant.get("scope") or "linear+pw", mx_rounding=quant.get("mx_rounding") or "rceil")
        if fp16:
            runner.amp = torch.float16
        wb = Q.weight_bytes(runner.model)
        runner.weights_bytes, runner.weights_bytes_resident = wb["deployable"], wb["resident"]
    if relpos_patch:
        from kitsune.patches import patch_relpos_once_per_batch

        patch_relpos_once_per_batch(runner.teacher.model if isinstance(runner, TdtRunner) else runner.model)
    if compile:
        m = runner.model
        if quant:
            from kitsune import quant as Q

            Q.set_counting(m, False)
        if kind in ("ctc", "parakeet-ctc"):
            m.encoder.compile(dynamic=True)
        else:
            m.model.encoder.compile(dynamic=True)
            m.model.decoder.compile(dynamic=True)
    return runner, desc


def _runner(kind: str, model: str | None, device: torch.device, dtype: str, info: dict, revision: str,
            pin: bool = False):
    if kind in ("aed", "cohere"):
        from transformers import AutoProcessor

        from kitsune import student as S

        if kind == "cohere" and model is None:
            from transformers import CohereAsrForConditionalGeneration

            m = CohereAsrForConditionalGeneration.from_pretrained(TEACHER_ID, revision=revision, dtype=torch.float32,
                                                                  attn_implementation="sdpa")
            proc = AutoProcessor.from_pretrained(TEACHER_ID, revision=revision)
            desc = f"{TEACHER_ID}@{revision}"
        else:
            m = S.load_student(model, "cpu", dtype=torch.float32)
            proc = AutoProcessor.from_pretrained(str(model))
            desc = str(model)
        return AedRunner(m, proc, device, dtype, info.get("prompt", trainset.PROMPT), info.get("eos", trainset.EOS),
                         info.get("pad", trainset.PAD), pin=pin), desc
    if kind in ("ctc", "parakeet-ctc"):
        from transformers import AutoProcessor

        from kitsune import ctc_student as CS

        m = CS.load_ctc_student(model, "cpu") if kind == "ctc" else CS.load_parakeet_ctc(model, "cpu")
        tok = AutoProcessor.from_pretrained(str(model), local_files_only=True).tokenizer
        return CtcRunner(m, model, tok, device, dtype), str(model)
    if kind == "parakeet-tdt":
        return TdtRunner(model, device, dtype), str(model)
    raise ValueError(f"--kind must be one of {KINDS}, got {kind!r}")


# ------------------------------------------------------------------------------------------------ the probe


def _peak(device: torch.device) -> dict:
    if device.type != "cuda":
        return dict(allocated=None, reserved=None)
    return dict(allocated=int(torch.cuda.max_memory_allocated(device)),
                reserved=int(torch.cuda.max_memory_reserved(device)))


def reset_peak(device: torch.device):
    """The peak counters only: what the allocator holds stays (no cudaMalloc inside the clock that follows)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def free_cache(device: torch.device):
    """The allocator's cached blocks back to the card (between the passes, never right before a clock)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def probe(runner, waves: list[np.ndarray], durations: list[float], refs: list[str], device: torch.device, *,
          batch_s: float, warmup: int, warmup_1: int, latency_n: int | None, n_tok: list[int] | None = None,
          profile_kernels: bool = False) -> dict:
    """The timed passes over `waves` (module docstring): the batched pass after its warm-up, then the batch-1 pass
    after its own. durations: the store's seconds of each wave (the batching and the RTF's audio seconds); n_tok: the
    teacher's tokens of each (an AED runner pinned to them reads them). Returns the record's numbers, with the
    batch-1 hypotheses kept (hyp_diff_1; "_hyps": (batched, batch-1) for --hyps-out) and, with profile_kernels, an
    untimed batched and batch-1 decode under kitsune.quant.kernel_census (kernels). Beats the item's heartbeat once per
    timed repeat, and under beating (max 1800 s) through the warm-ups."""
    from kitsune import heartbeat
    from kitsune.evaluate import corpus_cer

    dur = np.asarray(durations, dtype=np.float64)
    tok = [1] * len(waves) if n_tok is None else [int(x) for x in n_tok]
    order = np.argsort(-dur, kind="stable")
    plan = trainset.pack_micro_batches(order, dur, float(batch_s))
    n1 = len(waves) if latency_n is None else min(int(latency_n), len(waves))

    def run(idx):
        return runner.decode([waves[i] for i in idx], [float(dur[i]) for i in idx], [tok[i] for i in idx])

    with runner.context():
        with heartbeat.beating(max_s=1800):  # a torch.compile warm-up can take minutes
            for b in plan[:warmup]:  # the longest batches: the allocator grows to the pass's size here
                run(b)
        reset_peak(device)
        runner.tokens = 0
        hyps: list[str | None] = [None] * len(waves)
        sync(device)
        t0 = time.perf_counter()
        for b in plan:
            for i, h in zip(b, run(b)):
                hyps[i] = h
            heartbeat.beat()  # rate-limited: a file touch every 5 s at most
        sync(device)
        wall = time.perf_counter() - t0
        peak_b = _peak(device)
        tokens_b = int(runner.tokens)
        free_cache(device)  # the batched pass's blocks go: batch-1's own reserved VRAM is measured
        warm_1 = ([max(range(n1), key=lambda i: dur[i])] if n1 and warmup_1 else []) + list(
            range(min(int(warmup_1), len(waves))))
        with heartbeat.beating(max_s=1800):
            for i in warm_1:  # the longest latency id first: the allocator grows to batch-1's largest need
                run([i])
        reset_peak(device)
        lat, hyps_1 = [], []
        for i in range(n1):
            sync(device)
            t = time.perf_counter()
            hyps_1 += run([i])
            sync(device)
            lat.append(time.perf_counter() - t)
            heartbeat.beat()
        peak_1 = _peak(device)
        kernels = None
        if profile_kernels:  # untimed, after both clocks
            from kitsune.quant import kernel_census, quant_layers

            m = getattr(runner, "model", None)
            qm = m if m is not None and quant_layers(m) else None
            kernels = dict(batched=kernel_census(lambda: run(plan[0]), device, model=qm),
                           batch1=kernel_census(lambda: run([0]), device, model=qm) if waves else None)
    lat_a = np.asarray(lat, np.float64)
    rtf1 = lat_a / dur[:n1] if n1 else np.zeros(0)
    audio = float(dur.sum())
    reserved = [p["reserved"] for p in (peak_b, peak_1) if p["reserved"] is not None]
    return dict(n_utts=len(waves), audio_s=round(audio, 3), batches=len(plan), batch_s=float(batch_s),
                wall_s=round(wall, 4), rtf=wall / audio if audio else None, warmup_batches=min(warmup, len(plan)),
                warmup_1=len(warm_1), n_latency=n1, tokens_per_utt=tokens_b / len(waves) if waves else None,
                p50_s=float(np.percentile(lat_a, 50)) if n1 else None,
                p95_s=float(np.percentile(lat_a, 95)) if n1 else None,
                mean_s=float(lat_a.mean()) if n1 else None,
                rtf_1_p50=float(np.percentile(rtf1, 50)) if n1 else None,
                latencies_s=[round(float(x), 5) for x in lat_a],
                vram_peak_allocated_bytes=peak_b["allocated"], vram_peak_reserved_bytes=peak_b["reserved"],
                vram_peak_allocated_bytes_1=peak_1["allocated"], vram_peak_reserved_bytes_1=peak_1["reserved"],
                vram_gb=max(reserved) / 1e9 if reserved else None,
                cer_ref_corpus=corpus_cer([h or "" for h in hyps], refs)["cer"],
                hyp_diff_1=sum(a != b for a, b in zip(hyps_1, hyps)), kernels=kernels, _hyps=(hyps, hyps_1))


def write_hyps(path: Path, ids: list[str], refs: list[str], hyps: list, hyps_1: list, durations: list[float]):
    """--hyps-out: the list's batched and batch-1 hypotheses (hyp_1 null past --latency-n), zstd parquet."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    h1 = list(hyps_1) + [None] * (len(ids) - len(hyps_1))
    df = pd.DataFrame(dict(id=ids, ref=refs, hyp=[h or "" for h in hyps], hyp_1=h1, duration=durations))
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def merge_out(path: Path, ids: list[str], system: str, record: dict) -> dict:
    """The output file with this system's record (replacing an older one of the same name). Refuses (SystemExit 2) a
    file whose systems were timed on another id list."""
    sha = ids_sha256(ids)
    doc = dict(schema=SCHEMA, ids=ids, ids_sha256=sha, systems={})
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("ids_sha256") != sha:
            print(f"REFUSED: {path} holds systems timed on another id list ({str(old.get('ids_sha256'))[:12]}, this "
                  f"one {sha[:12]}): use its list (--ids) or another --out", file=sys.stderr)
            raise SystemExit(EXIT_REFUSED)
        doc = dict(old, schema=SCHEMA)
    doc.setdefault("systems", {})[system] = record
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return doc


def _versions() -> dict:
    import transformers

    from kitsune.quant import torchao_version

    return dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, host=platform.node(), torchao=torchao_version())


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--model", default=None,
                    help="the model dir (a student, or the converted Parakeet dir); cohere: default the HF teacher")
    ap.add_argument("--system", default=None,
                    help="the name in the output (default for the teachers: cohere, parakeet-ctc, parakeet-tdt; "
                         "students need it: their run name)")
    ap.add_argument("--store", required=True, help="a built kitsune.trainset store dir holding the ids' audio")
    ap.add_argument("--ids", default=None, help="the fixed id list: one id per line, or a JSON list")
    ap.add_argument("--per-set", type=int, default=40,
                    help="without --ids (and no list in --out yet): this many seeded ids per eval set (default "
                         "%(default)s)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, help="the speed JSON, merged per system")
    ap.add_argument("--batch-s", type=float, default=400.0,
                    help="padded audio seconds per batch of the batched pass (default %(default)s: the eval's)")
    ap.add_argument("--latency-n", type=int, default=None, help="batch-1 on the first N ids only (default: all)")
    ap.add_argument("--warmup", type=int, default=2, help="warm-up batches, not timed (default %(default)s)")
    ap.add_argument("--warmup-1", type=int, default=3, help="warm-up single utterances (default %(default)s)")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cuda, cuda:N or cpu")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp32"))
    ap.add_argument("--revision", default=TEACHER_REVISION, help="the Cohere teacher's revision (kind cohere)")
    ap.add_argument("--require-idle", action="store_true",
                    help="refuse (exit 3) while another compute process is on the GPU")
    ap.add_argument("--no-relpos-patch", action="store_true",
                    help="time the encoder without the rel-pos-once-per-batch patch (the trainer and evaluator use it)")
    ap.add_argument("--decode-len", default="auto", choices=DECODE_LENS,
                    help="aed / cohere: teacher = every batch decodes its longest row's teacher token count, greedy = "
                         "the free decode, auto = teacher for an untrained student dir, else greedy (default "
                         "%(default)s)")
    from kitsune.quant import COMPILE_SUFFIX, IMPLS, MX_ROUNDINGS, QUANT_FORMATS, SCOPES

    ap.add_argument("--quant", default="none", choices=("none", *QUANT_FORMATS),
                    help="aed / ctc: time the student quantised in this format (the system must end in @<fmt>)")
    ap.add_argument("--quant-impl", default="auto", choices=IMPLS,
                    help="auto: torchao on CUDA (emulate on CPU); emulate: simulated, recorded emulated (mxfp4 on "
                         "CUDA needs it)")
    ap.add_argument("--quant-scope", default="linear+pw", choices=SCOPES)
    ap.add_argument("--mx-rounding", default="rceil", choices=MX_ROUNDINGS)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder (and the aed decoder) in place; the system ends in +compile")
    ap.add_argument("--profile-kernels", action="store_true",
                    help="an untimed batched and batch-1 decode under the kernel census (record: kernels)")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads before loading")
    ap.add_argument("--hyps-out", default=None, help="a parquet of the list's hypotheses (id, ref, hyp, hyp_1, "
                                                     "duration)")
    args = ap.parse_args(argv)
    if args.quant != "none":
        from kitsune.quant import split_system

        if args.kind not in ("aed", "ctc"):
            ap.error(f"--quant times a student: kinds aed and ctc, not {args.kind}")
        base = (args.system or "")[:-len(COMPILE_SUFFIX)] if args.compile else (args.system or "")
        if split_system(base)[1] != args.quant or base.endswith(COMPILE_SUFFIX):
            suffix = COMPILE_SUFFIX if args.compile else ""
            ap.error(f"--quant {args.quant} needs --system <base>@{args.quant}{suffix}: a variant's record must never "
                     "replace the base system's")
        if args.quant == "fp16" and args.dtype != "auto":
            ap.error("--quant fp16 sets the dtype (fp16 weights under fp16 autocast): no --dtype")
    if args.compile and args.kind == "parakeet-tdt":
        ap.error("--compile compiles a ParakeetForCTC or Cohere ASR encoder: not the TDT path")
    if args.compile and not (args.system or "").endswith(COMPILE_SUFFIX):
        ap.error(f"--compile needs a --system ending in {COMPILE_SUFFIX}")
    if args.compile and args.warmup < 2:
        ap.error("--compile needs --warmup >= 2 (the compile happens in the warm-up)")
    if args.threads is not None and args.threads < 1:
        ap.error("--threads must be >= 1")
    if args.kind in ("aed", "ctc", "parakeet-ctc", "parakeet-tdt") and not args.model:
        ap.error(f"--kind {args.kind} needs --model")
    if args.kind in ("aed", "ctc") and not args.system:
        ap.error(f"--kind {args.kind} needs --system (the student's run name)")
    if args.per_set < 1 or args.batch_s <= 0 or args.warmup < 0 or args.warmup_1 < 0:
        ap.error("--per-set >= 1, --batch-s > 0, --warmup and --warmup-1 >= 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    from kitsune import heartbeat
    from kitsune import quant as Q

    if args.threads:
        torch.set_num_threads(int(args.threads))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dtype = ("bf16" if device.type == "cuda" else "fp32") if args.dtype == "auto" else args.dtype
    system = args.system or args.kind
    quant = None
    if args.quant != "none":
        if args.model and Q.is_quantized_dir(args.model):
            print(f"REFUSED: {args.model} is a quantised variant dir: time the bf16 export with --quant "
                  f"{Q.read_recipe(args.model)['format']} (the variant's weights are the same bits)", file=sys.stderr)
            return EXIT_REFUSED
        if device.type == "cuda" and args.quant_impl == "auto" and args.quant == "mxfp4-w4a4":
            print("REFUSED: mxfp4-w4a4 has no kernel on this GPU (decision 20): --quant-impl emulate times its "
                  "simulation, recorded emulated", file=sys.stderr)
            return EXIT_REFUSED
        try:
            impl = Q.resolve_impl(args.quant, args.quant_impl, device)
        except Q.QuantError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return EXIT_REFUSED
        quant = dict(fmt=args.quant, impl=args.quant_impl, resolved=impl, scope=args.quant_scope,
                     mx_rounding=args.mx_rounding)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu = gpu_state(device)
    idle = None if gpu is None else not gpu["processes"]
    if args.require_idle and idle is False:
        print(f"BUSY: other compute processes on the GPU: {gpu['processes']}", file=sys.stderr)
        return EXIT_BUSY
    store = trainset.load_stores(args.store)
    old = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    if args.ids:
        ids = read_ids(Path(args.ids))
    elif old.get("ids"):
        ids = list(old["ids"])  # the list the other systems were timed on
    else:
        ids = pick_ids(store, args.per_set, args.seed)
    if old.get("ids_sha256") and old["ids_sha256"] != ids_sha256(ids):
        print(f"REFUSED: {out} holds systems timed on another id list: use its list or another --out", file=sys.stderr)
        return EXIT_REFUSED
    pos = {u.id: i for i, u in enumerate(store.utts)}
    if missing := [i for i in ids if i not in pos]:
        print(f"REFUSED: {len(missing)} ids are not in {args.store}, e.g. {missing[:3]}", file=sys.stderr)
        return EXIT_REFUSED
    trained = trained_weights(args.kind, args.model)
    decode_len = decode_len_of(args.kind, args.decode_len, trained)
    if decode_len == "teacher" and (store.info or {}).get("kind") == "frames":
        print(f"REFUSED: {args.store} is a frame store, whose token counts are the CTC targets': an AED decode pinned "
              "to the teacher's length needs a token store (the eval store, cache/eval)", file=sys.stderr)
        return EXIT_REFUSED
    waves = [store.wave(pos[i]) for i in ids]  # decoded before any clock starts
    durations = [float(store.utts[pos[i]].duration) for i in ids]
    n_tok = [int(store.utts[pos[i]].n_tok) for i in ids]  # the teacher's tokens with EOS (a token store's)
    refs = store.frame().set_index("id").loc[ids, "ref"].tolist()
    t0 = time.time()
    with heartbeat.beating(max_s=1800):
        runner, desc = load_runner(args.kind, args.model, device, dtype, store, args.revision,
                                   relpos_patch=not args.no_relpos_patch, decode_len=decode_len, quant=quant,
                                   compile=args.compile)
    load_s = time.time() - t0
    amp = getattr(runner, "amp", None)
    rec = dict(kind=args.kind, model=desc, device=str(device),
               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               dtype="fp16" if quant and quant["fmt"] == "fp16" else dtype,
               params_total=int(runner.params), weights_bytes=int(runner.weights_bytes), load_s=round(load_s, 2),
               relpos_patch=not args.no_relpos_patch, decode_len=decode_len, trained=trained,
               quant=quant["fmt"] if quant else None, quant_impl=quant["resolved"] if quant else None,
               quant_scope=quant["scope"] if quant else None, mx_rounding=quant["mx_rounding"] if quant else None,
               emulated=bool(quant) and quant["resolved"] == "emulate", compile=bool(args.compile),
               threads=torch.get_num_threads(), autocast_dtype=str(amp).replace("torch.", "") if amp else None,
               weights_bytes_resident=int(getattr(runner, "weights_bytes_resident", runner.weights_bytes)))
    res = probe(runner, waves, durations, refs, device, batch_s=args.batch_s, warmup=args.warmup,
                warmup_1=args.warmup_1, latency_n=args.latency_n, n_tok=n_tok, profile_kernels=args.profile_kernels)
    hyps, hyps_1 = res.pop("_hyps")
    rec.update(res)
    rec["quant_counters"] = {k: v for k, v in Q.counters(runner.model).items() if k != "rows"} if quant else None
    if args.hyps_out:
        write_hyps(Path(args.hyps_out), ids, refs, hyps, hyps_1, durations)
    rec.update(idle=idle, gpu_state=gpu, versions=_versions(), store=str(args.store), time_utc=_now())
    merge_out(out, ids, system, rec)
    print(f"{system}: RTF {rec['rtf']:.5f} batched ({rec['n_utts']} utts, {rec['audio_s']:.0f} s), batch-1 p50 "
          f"{rec['p50_s']:.3f} s p95 {rec['p95_s']:.3f} s, VRAM "
          + ("n/a" if rec["vram_gb"] is None else f"{rec['vram_gb']:.2f} GB") + f", CER {rec['cer_ref_corpus']:.4f}"
          f" -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
