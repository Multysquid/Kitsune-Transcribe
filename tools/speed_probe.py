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
  whisper       a Whisper yardstick (kitsune.whisper: a WHISPER_MODELS key, fetched at its pinned revision into
                --hf-cache, or a model dir): the weights in bf16 (fp32 on CPU), its own log-mel features on the device
                (every clip padded to 30 s), greedy_whisper with the fp32 head (the build's decision C15, as the AED
                students; --no-fp32-head times the head in the weights' dtype), decode_texts - the very functions
                tools/whisper_eval.py scores with. Its batches are the batched pass's below, row-capped at the model's
                max_rows (--batch-rows overrides it): a 30 s encoder window per clip does not fit 400 s of short clips
Both Parakeet paths and the CTC students take Parakeet's own features (ctc_features: 80 mel, no dither) on the device.
Every FastConformer encoder (the five kinds above whisper) runs with kitsune.patches.patch_relpos_once_per_batch, as the
trainer and the evaluator run it (perf.relpos_patch; --no-relpos-patch times it without, and the record says which);
Whisper has no such encoder (relpos_patch null).

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
record}, "failed": {name: why}}. A record: kind, model, device, gpu, dtype, batch_s, n_utts, audio_s, batches, rtf,
wall_s, the batch-1
fields (n_latency, p50_s, p95_s, mean_s, rtf_1_p50), vram_peak_allocated_bytes / _reserved_bytes for each pass,
vram_gb, params_total, weights_bytes, relpos_patch, decode_len, trained, tokens_per_utt, cer_ref_corpus, max_rows (the
batched pass's row cap, null without one), n_truncated / n_timestamp_tokens (whisper's batched pass: rows the stop or
the length cap cut, timestamp ids stripped; null for the other kinds), fp32_head (whisper: whether its LM head ran in
fp32; null for the other kinds, whose heads are fixed), idle, versions (python, torch, transformers, cuda, host,
torchao, machine_id = $KITSUNE_MACHINE_ID, cpu), time_utc. tools/study_report.py --speed reads it (the Pareto view:
rtf, vram_gb, p50_s, p95_s; kitsune.study_stats.speed_entry); the full runs' report groups records by machine_id + gpu.

Quantised variants (the full runs; kitsune.quant, contract section 8; aed and ctc kinds): --quant <fmt> times the
student quantised after its dtype cast (kitsune.quant.apply: torchao's kernels on CUDA; --quant-impl emulate only when
asked, recorded emulated: true and never a speed; mxfp4 has no kernel here and needs it; fp16 keeps the model fp16
under fp16 autocast, dtype "fp16"), then the rel-pos patch. The bf16 export is what is timed: a variant dir as --model
is refused (its weights are the same bits). The system must be <base>@<fmt> (+compile with --compile: the encoder, and
the aed decoder, compiled in place, counters off, --warmup >= 2). --profile-kernels adds an untimed batched and
batch-1 decode under kitsune.quant.kernel_census (kernels: the matmul ops and GEMM kernel classes, the int8 fallbacks:
the int8 calls whose torch._int_mm did not return); --threads sets torch's threads before loading; --hyps-out writes
the list's hypotheses (id, ref, hyp, hyp_1, duration). With --compile the layers' counters are off, so quant_counters
and the census's int8 counts are null (not counted, rather than zeros that would read as a measurement).
Every record gains quant_recipe ({version, sha256}: kitsune.quant's RECIPE_VERSION and recipe_sha256 of the timed
format; null without --quant), reference ({system, dtype, cer_ref_corpus, tokens_per_utt, batches, wall_s}: the bf16
decode below; null without --quant / --compile) and sanity ({ok, cer, cer_bf16, delta, max_delta, hyps_differ,
reference, reason}; null likewise), quant, quant_impl, quant_scope, mx_rounding, emulated, compile, threads,
autocast_dtype, quant_counters, kernels, weights_bytes_resident and hyp_diff_1 (the latency ids whose batch-1
hypothesis differs from the batched one: NVFP4 W4A4's whole-call activation scale makes a row depend on its
batch-mates); weights_bytes is the deployable packed bytes for a format. The idle check counts only the processes
on the probe's own GPU (by PCI bus id), so --require-idle works while the box's other GPU trains. The probe beats
$KITSUNE_HEARTBEAT once per timed repeat, and under kitsune.heartbeat.beating (max 1800 s) through the model load
and the warm-ups (torch.compile's).

Sanity (F4, DECISIONS F; smoke B check 16): a timed number of a model that decodes garbage is not a speed. Box
53693389's fp8-w8a8 records decoded NaN-filled garbage (cer_ref_corpus 0.939-0.945 against 0.19-0.27 for every other
format) and passed check 16 as "done". So every --quant or --compile probe, after both clocks and the census (nothing
it does can touch the timings or the VRAM peaks), frees the timed model and loads the same weights once more in the
probe's dtype, eager, unquantised, with the rel-pos patch and decode length as timed (reference_pass), and decodes the
same ids in the very batches of the batched pass. The record's sanity: ok when both corpus CERs are finite and the
timed model's is at most SANITY_MAX_DELTA (5 pp) above the reference's. The bound: on the study's 200 speed ids every
format that worked on box 53693389's smoke B stayed within 0.8 pp of its student's bf16 CER (int8 W8A16 / W8A8, NVFP4
W4A16 and fp16: -0.16 to +0.79 pp, bf16 itself at 0.187-0.270); on the full manifest NVFP4 W4A4 cost P-0.3B +0.35 pp
M4 and MXFP4 +0.70 pp; the broken fp8 was +67 to +71 pp. 5 pp is about 6x the largest genuine cost and 13x below that
failure (about 20-25 % relative): it catches gross breakage (NaN, garbage, a wrong scale) and is not a measurement -
the readouts measure a format's cost with CIs. A record that fails it is still written, with its numbers, and the
probe exits EXIT_INSANE (4): the queue fails a speed item on any non-zero exit with no retry, so check 16 ("item done")
fails and the speed dir's events.jsonl says "exit 4". A load, probe or reference that raises is recorded too, in the
file's top-level failed {system: {error, stage, quant, compile, time_utc, versions}} (a later successful probe of the
system removes it), and the exit stays 1: tools/full_report.py lists failed and left-out probes instead of leaving a
row silently empty.

Usage (on the box, at its end, one call per system; kitsune/study_queue.py phase_speed passes these):
  python tools/speed_probe.py --kind aed --model students/study/t03 --system study-t03 \
      --store cache/eval --per-set 40 --out runs/speed-B/speed.json --require-idle
  python tools/speed_probe.py --kind cohere --system cohere --store cache/eval --per-set 40 \
      --out runs/speed-B/speed.json --require-idle
  python tools/speed_probe.py --kind parakeet-tdt --model models/parakeet-tdt_ctc-0.6b-ja-hf --system parakeet-tdt \
      --store cache/eval --per-set 40 --out runs/speed-B/speed.json --require-idle
  python tools/speed_probe.py --kind whisper --model whisper-large-v3 --hf-cache cache/hf --store cache/eval \
      --per-set 40 --out runs/speed-smoke-b-<stamp>/speed.json --require-idle      # system = the key
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
KINDS = ("aed", "cohere", "ctc", "parakeet-ctc", "parakeet-tdt", "whisper")
AED_KINDS = ("aed", "cohere")  # the autoregressive decodes: their time depends on the tokens they emit
DECODE_LENS = ("auto", "greedy", "teacher")
TEACHER_ID = "CohereLabs/cohere-transcribe-03-2026"
TEACHER_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"  # == scripts/03_build_student.TEACHER_REVISION (tested)
MAX_SYMBOLS = 10  # the label pass's TDT guard (kitsune.parakeet; vast/label.py)
EXIT_REFUSED, EXIT_BUSY = 2, 3
EXIT_INSANE = 4  # a --quant / --compile record whose CER is not sane against its bf16 reference (written, then 4)
# The CER sanity bound (module docstring, "Sanity"; F4): the timed model's corpus CER on the id list may be at most this
# far above the same weights' bf16 eager decode of the same ids in the same batches. Working formats cost <= 0.8 pp on
# the study's 200 ids (box 53693389's smoke B) and <= 0.70 pp M4 on the full manifest; the broken fp8 cost 67-71 pp.
# A gross-breakage alarm, never a measurement; changing it is an owner call
SANITY_MAX_DELTA = 0.05


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


class WhisperRunner:
    """A Whisper model (kitsune.whisper): features on the device, greedy_whisper, decode_texts - whisper_eval's decode.
    max_rows: the row cap of the batched pass (probe). It counts the rows the stop cut (n_truncated) and the timestamp
    ids it stripped, and beats the item's heartbeat once per decode (kitsune.heartbeat.beat: rate-limited, a no-op
    without KITSUNE_HEARTBEAT)."""

    tokens = 0
    n_truncated = 0
    n_timestamp_tokens = 0

    def __init__(self, model_dir, spec, device, dtype: str, max_rows: int | None = None, fp32_head: bool = True):
        from kitsune import heartbeat
        from kitsune import whisper as W

        self.W, self.heartbeat, self.device = W, heartbeat, device
        self.wm = W.load_whisper(model_dir, device, torch.bfloat16 if dtype == "bf16" else torch.float32, spec=spec)
        self.model = self.wm.model
        self.params = self.wm.params_total
        self.weights_bytes = _weights_bytes(self.model)
        self.max_rows = int(max_rows or (spec.max_rows if spec else W.DEFAULT_MAX_ROWS))
        self.fp32_head = bool(fp32_head)  # the record's fp32_head: whisper_eval's model.fp32_head, the same switch

    def context(self):
        from contextlib import ExitStack

        from kitsune.evaluate import _fp32_head

        stack = ExitStack()
        stack.enter_context(torch.inference_mode())
        if self.fp32_head:
            stack.enter_context(_fp32_head(self.model))
        return stack

    def decode(self, waves: list[np.ndarray], durations: list[float], n_tok: list[int]) -> list[str]:
        self.heartbeat.beat()
        feats = self.W.features(self.wm, waves)
        rows = self.W.greedy_whisper(self.wm, feats, max(durations), fp32_head=False)  # context() holds the fp32 head
        self.tokens += sum(len(r["hyp_ids"]) for r in rows)
        self.n_truncated += sum(bool(r["truncated"]) for r in rows)
        self.n_timestamp_tokens += sum(int(r["n_timestamp_tokens"]) for r in rows)
        return self.W.decode_texts(self.wm, rows)


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
    "teacher" for an untrained student dir, "greedy" otherwise; Whisper always decodes freely ("greedy": it has no
    stored token counts of its own, parse_args refuses "teacher")."""
    if kind == "whisper":
        return "greedy"
    if kind not in AED_KINDS:
        return None
    if choice != "auto":
        return choice
    return "teacher" if kind == "aed" and trained is False else "greedy"


def load_runner(kind: str, model: str | None, device: torch.device, dtype: str, store, revision: str,
                relpos_patch: bool = True, decode_len: str | None = None, quant: dict | None = None,
                compile: bool = False, *, hf_cache: str | None = None, max_rows: int | None = None,
                fp32_head: bool = True):
    """The runner of one kind and its model's description; relpos_patch: kitsune.patches.patch_relpos_once_per_batch
    on its encoder, as the trainer and the evaluator run every one of these encoders (perf.relpos_patch; never for
    whisper, which has no FastConformer: the patch refuses a model without Parakeet's rel-pos encoding); decode_len
    "teacher": an AED runner pinned to the teacher's token counts. quant {fmt, impl, scope, mx_rounding}: the model
    quantised after the runner's dtype cast and before the patch (fp16: the runner keeps fp32 weights, kitsune.quant
    casts them, autocast fp16), runner.quant = the recipe, weights_bytes = the deployable bytes. The runner's bf16 cast
    comes first (the study's timing convention: BatchNorm statistics bf16 too), and kitsune.quant.apply leaves those
    statistics as it finds them. compile: the encoder (and an AED's decoder) compiled in place last, the quant counters
    off (their increments are Python side effects; quant_counters then records null). hf_cache / max_rows /
    fp32_head: whisper's snapshot cache, row cap and LM-head precision."""
    fp16 = bool(quant) and quant["fmt"] == "fp16"
    runner, desc = _runner(kind, model, device, "fp32" if fp16 else dtype, store.info or {}, revision,
                           pin=decode_len == "teacher", hf_cache=hf_cache, max_rows=max_rows, fp32_head=fp32_head)
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
    if relpos_patch and kind != "whisper":
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
            pin: bool = False, hf_cache: str | None = None, max_rows: int | None = None, fp32_head: bool = True):
    if kind == "whisper":
        from kitsune import heartbeat
        from kitsune import whisper as W

        with heartbeat.beating(max_s=W.LOAD_BEAT_MAX_S):  # a pinned snapshot's download + load
            path, spec = W.resolve(model, hf_cache)
            runner = WhisperRunner(path, spec, device, dtype, max_rows=max_rows, fp32_head=fp32_head)
        return runner, f"{spec.repo}@{spec.revision}" if spec else str(path)
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


def batch_plan(dur: np.ndarray, batch_s: float, max_rows: int | None) -> list[list[int]]:
    """The batched pass's batches (indices into the id list): the eval's own (pack_micro_batches over the ids sorted by
    duration, longest first, at most batch_s padded seconds), cut at max_rows rows when given. probe and
    reference_pass both use it, so the bf16 reference decodes the very batches the timed model did."""
    dur = np.asarray(dur, dtype=np.float64)
    order = np.argsort(-dur, kind="stable")
    return trainset.pack_micro_batches(order, dur, float(batch_s), dec_len=np.ones(len(dur)) if max_rows else None,
                                       cap_tokens=int(max_rows) if max_rows else None)


def probe(runner, waves: list[np.ndarray], durations: list[float], refs: list[str], device: torch.device, *,
          batch_s: float, warmup: int, warmup_1: int, latency_n: int | None, n_tok: list[int] | None = None,
          profile_kernels: bool = False, batch_rows: int | None = None) -> dict:
    """The timed passes over `waves` (module docstring): the batched pass after its warm-up, then the batch-1 pass
    after its own. durations: the store's seconds of each wave (the batching and the RTF's audio seconds); n_tok: the
    teacher's tokens of each (an AED runner pinned to them reads them); the batched pass's row cap: the runner's
    max_rows (whisper), else batch_rows, else none - the same real-seconds batches as every kind, cut at that many
    rows (pack_micro_batches' cap_tokens over one "token" per row). Returns the record's numbers, with the batch-1
    hypotheses kept (hyp_diff_1; "_hyps": (batched, batch-1) for --hyps-out) and, with profile_kernels, an untimed
    batched and batch-1 decode under kitsune.quant.kernel_census (kernels). Beats the item's heartbeat once per timed
    repeat, and under beating (max 1800 s) through the warm-ups."""
    from kitsune import heartbeat
    from kitsune.evaluate import corpus_cer

    dur = np.asarray(durations, dtype=np.float64)
    tok = [1] * len(waves) if n_tok is None else [int(x) for x in n_tok]
    max_rows = getattr(runner, "max_rows", None) or batch_rows
    plan = batch_plan(dur, batch_s, max_rows)
    counted = [k for k in ("n_truncated", "n_timestamp_tokens") if hasattr(runner, k)]  # whisper's per-pass counts
    n1 = len(waves) if latency_n is None else min(int(latency_n), len(waves))

    def run(idx):
        return runner.decode([waves[i] for i in idx], [float(dur[i]) for i in idx], [tok[i] for i in idx])

    with runner.context():
        with heartbeat.beating(max_s=1800):  # a torch.compile warm-up can take minutes
            for b in plan[:warmup]:  # the longest batches: the allocator grows to the pass's size here
                run(b)
        reset_peak(device)
        runner.tokens = 0
        for k in counted:
            setattr(runner, k, 0)
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
        counts_b = {k: int(getattr(runner, k)) for k in counted}
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
                hyp_diff_1=sum(a != b for a, b in zip(hyps_1, hyps)), kernels=kernels,
                max_rows=int(max_rows) if max_rows else None, n_truncated=counts_b.get("n_truncated"),
                n_timestamp_tokens=counts_b.get("n_timestamp_tokens"), _hyps=(hyps, hyps_1))


def reference_pass(kind: str, model: str | None, device: torch.device, dtype: str, store, revision: str,
                   waves: list[np.ndarray], durations: list[float], refs: list[str], n_tok: list[int] | None, *,
                   relpos_patch: bool, decode_len: str | None, batch_s: float, batch_rows: int | None,
                   hf_cache: str | None = None, fp32_head: bool = True) -> dict:
    """The sanity's bf16 reference (module docstring, "Sanity"): the same weights loaded once more in the probe's
    dtype (bf16 on CUDA), eager and unquantised (load_runner with quant None, compile False), with the rel-pos patch
    and decode length the timed probe ran, decoding the id list in the batched pass's batches (batch_plan), untimed.
    Called after both clocks, with the timed model freed. Returns {dtype, cer_ref_corpus, tokens_per_utt, batches,
    wall_s, _hyps}."""
    from kitsune import heartbeat
    from kitsune.evaluate import corpus_cer

    t0 = time.time()
    with heartbeat.beating(max_s=1800):
        runner, _ = load_runner(kind, model, device, dtype, store, revision, relpos_patch=relpos_patch,
                                decode_len=decode_len, quant=None, compile=False, hf_cache=hf_cache,
                                max_rows=batch_rows, fp32_head=fp32_head)
        dur = np.asarray(durations, dtype=np.float64)
        tok = [1] * len(waves) if n_tok is None else [int(x) for x in n_tok]
        plan = batch_plan(dur, batch_s, getattr(runner, "max_rows", None) or batch_rows)
        hyps: list[str | None] = [None] * len(waves)
        runner.tokens = 0
        with runner.context():
            for b in plan:
                out = runner.decode([waves[i] for i in b], [float(dur[i]) for i in b], [tok[i] for i in b])
                for i, h in zip(b, out):
                    hyps[i] = h
                heartbeat.beat()
        sync(device)
    tokens = int(runner.tokens)
    del runner
    return dict(dtype=dtype, cer_ref_corpus=corpus_cer([h or "" for h in hyps], refs)["cer"],
                tokens_per_utt=tokens / len(waves) if waves else None, batches=len(plan),
                wall_s=round(time.time() - t0, 2), _hyps=hyps)


def cer_sanity(cer, cer_bf16, *, hyps_differ: int | None = None, reference: str | None = None) -> dict:
    """The record's sanity (module docstring): ok when both corpus CERs are finite numbers and cer - cer_bf16 <=
    SANITY_MAX_DELTA (a quantised model better than its bf16 weights by any margin passes); reason says why not."""
    import math

    finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in (cer, cer_bf16))
    delta = float(cer) - float(cer_bf16) if finite else None
    ok = finite and delta <= SANITY_MAX_DELTA
    reason = None if ok else (f"a corpus CER is not a finite number (cer {cer}, bf16 reference {cer_bf16})"
                              if not finite else
                              f"cer_ref_corpus {cer:.4f} is {100 * delta:.2f} pp above the bf16 reference's "
                              f"{cer_bf16:.4f} (bound {100 * SANITY_MAX_DELTA:g} pp)")
    return dict(ok=bool(ok), cer=cer, cer_bf16=cer_bf16, delta=delta, max_delta=SANITY_MAX_DELTA,
                hyps_differ=hyps_differ, reference=reference, reason=reason)


def quant_counters(model) -> dict | None:
    """The record's quant_counters: the quantised layers' {calls, padded, fallback_risk}; None when they were not
    counted (--compile turns the counters off, and zeros would then read as a measurement)."""
    from kitsune import quant as Q

    if any(not m.kq.count for m in Q.quant_layers(model).values()):
        return None
    return {k: v for k, v in Q.counters(model).items() if k != "rows"}


def write_hyps(path: Path, ids: list[str], refs: list[str], hyps: list, hyps_1: list, durations: list[float]):
    """--hyps-out: the list's batched and batch-1 hypotheses (hyp_1 null past --latency-n), zstd parquet."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    h1 = list(hyps_1) + [None] * (len(ids) - len(hyps_1))
    df = pd.DataFrame(dict(id=ids, ref=refs, hyp=[h or "" for h in hyps], hyp_1=h1, duration=durations))
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _merge(path: Path, ids: list[str], edit) -> dict:
    """The output file read (or a new one), edit(doc) applied, written atomically. Refuses (SystemExit 2) a file
    whose systems were timed on another id list."""
    sha = ids_sha256(ids)
    doc = dict(schema=SCHEMA, ids=ids, ids_sha256=sha, systems={})
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("ids_sha256") != sha:
            print(f"REFUSED: {path} holds systems timed on another id list ({str(old.get('ids_sha256'))[:12]}, this "
                  f"one {sha[:12]}): use its list (--ids) or another --out", file=sys.stderr)
            raise SystemExit(EXIT_REFUSED)
        doc = dict(old, schema=SCHEMA)
    edit(doc)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return doc


def merge_out(path: Path, ids: list[str], system: str, record: dict) -> dict:
    """The output file with this system's record (replacing an older one of the same name), and the system's entry
    in the top-level failed block gone (a probe that ran supersedes an earlier failure). Refuses (SystemExit 2) a
    file whose systems were timed on another id list."""
    def edit(doc):
        doc.setdefault("systems", {})[system] = record
        failed = doc.get("failed")
        if isinstance(failed, dict):
            failed.pop(system, None)

    return _merge(path, ids, edit)


def merge_failure(path: Path, ids: list[str], system: str, info: dict) -> dict:
    """A probe that raised (module docstring, "Sanity"): failed[system] = info ({error, stage: load | probe |
    reference, quant, compile, time_utc, versions}); the system's last good record, if any, stays as it was."""
    def edit(doc):
        failed = doc.get("failed")
        doc["failed"] = failed = failed if isinstance(failed, dict) else {}
        failed[system] = info

    return _merge(path, ids, edit)


def _cpu_model() -> str | None:
    """The CPU's model name (/proc/cpuinfo on Linux; platform.processor() elsewhere, which on Linux is only the
    architecture), None when unknown."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip() or None
    except OSError:
        pass
    return platform.processor() or None


def _versions() -> dict:
    """The software (python, torch, transformers, cuda, torchao: kitsune.quant's torchao_version) and the host (host,
    machine_id, cpu). host is the container's hostname, which differs per rental: machine_id ($KITSUNE_MACHINE_ID,
    the vast machine launch rented) and the CPU model group records of one host."""
    import transformers

    from kitsune.quant import torchao_version

    return dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, host=platform.node(), torchao=torchao_version(),
                machine_id=os.environ.get("KITSUNE_MACHINE_ID") or None, cpu=_cpu_model())


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
    ap.add_argument("--hf-cache", default=None,
                    help="whisper: the HF cache a key's pinned snapshot is fetched into (the box's <root>/cache/hf)")
    ap.add_argument("--batch-rows", type=int, default=None,
                    help="the batched pass's row cap (default: whisper, the model's max_rows; the others, none)")
    ap.add_argument("--no-fp32-head", action="store_true",
                    help="whisper: the LM head in the weights' dtype (default: fp32, as whisper_eval scores it)")
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
    if args.kind in ("aed", "ctc", "parakeet-ctc", "parakeet-tdt", "whisper") and not args.model:
        ap.error(f"--kind {args.kind} needs --model" + (" (a Whisper key or a model dir)" if args.kind == "whisper"
                                                        else ""))
    if args.kind == "whisper":
        from kitsune.whisper import WHISPER_MODELS

        if args.decode_len == "teacher":
            ap.error("--kind whisper decodes freely: --decode-len teacher pins an AED decode to Cohere's token counts")
        if not args.system:
            if args.model not in WHISPER_MODELS:
                ap.error("--kind whisper with a model dir needs --system")
            args.system = args.model  # a key names its system
    if args.batch_rows is not None and args.batch_rows < 1:
        ap.error("--batch-rows must be >= 1")
    if args.no_fp32_head and args.kind != "whisper":
        ap.error("--no-fp32-head is --kind whisper's: the other kinds time the head the evaluator scores them with")
    if args.kind in ("aed", "ctc") and not args.system:
        ap.error(f"--kind {args.kind} needs --system (the student's run name)")
    if args.per_set < 1 or args.batch_s <= 0 or args.warmup < 0 or args.warmup_1 < 0:
        ap.error("--per-set >= 1, --batch-s > 0, --warmup and --warmup-1 >= 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
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
    stages: list[str] = []
    try:
        rec = _timed_record(args, device, dtype, store, quant, decode_len, trained, waves, durations, n_tok, refs, ids,
                            stages)
    except Exception as e:  # noqa: BLE001 - recorded in the file's failed block, then re-raised: the exit stays 1
        try:
            merge_failure(out, ids, system, dict(error=f"{type(e).__name__}: {e}"[:1000],
                                                 stage=stages[-1] if stages else "load",
                                                 quant=quant["fmt"] if quant else None, compile=bool(args.compile),
                                                 time_utc=_now(), versions=_versions()))
        except BaseException as w:  # noqa: BLE001 - the record of a failure must never hide the failure itself
            print(f"speed_probe: could not record the failure in {out}: {type(w).__name__}: {w}", file=sys.stderr)
        raise
    rec.update(idle=idle, gpu_state=gpu, versions=_versions(), store=str(args.store), time_utc=_now())
    merge_out(out, ids, system, rec)
    print(f"{system}: RTF {rec['rtf']:.5f} batched ({rec['n_utts']} utts, {rec['audio_s']:.0f} s), batch-1 p50 "
          f"{rec['p50_s']:.3f} s p95 {rec['p95_s']:.3f} s, VRAM "
          + ("n/a" if rec["vram_gb"] is None else f"{rec['vram_gb']:.2f} GB") + f", CER {rec['cer_ref_corpus']:.4f}"
          f" -> {out}", flush=True)
    s = rec.get("sanity")
    if s is not None and not s["ok"]:
        print(f"INSANE: {system}: {s['reason']} ({s['hyps_differ']} of {rec['n_utts']} hypotheses differ from the bf16 "
              f"reference {s['reference']}): the record is written, but its timings are not a speed (exit "
              f"{EXIT_INSANE})", file=sys.stderr, flush=True)
        return EXIT_INSANE
    return 0


def _timed_record(args, device, dtype, store, quant, decode_len, trained, waves, durations, n_tok, refs, ids,
                  stages: list) -> dict:
    """main's load, probe and (for --quant / --compile) the bf16 reference with the sanity, as one record; the stage
    it is in is appended to stages (main's failure record names it)."""
    import gc

    from kitsune import heartbeat
    from kitsune.quant import split_system

    stages.append("load")
    t0 = time.time()
    with heartbeat.beating(max_s=1800):
        runner, desc = load_runner(args.kind, args.model, device, dtype, store, args.revision,
                                   relpos_patch=not args.no_relpos_patch, decode_len=decode_len, quant=quant,
                                   compile=args.compile, hf_cache=args.hf_cache, max_rows=args.batch_rows,
                                   fp32_head=not args.no_fp32_head)
    load_s = time.time() - t0
    amp = getattr(runner, "amp", None)
    qrec = getattr(runner, "quant", None) or {}
    rec = dict(kind=args.kind, model=desc, device=str(device),
               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               dtype="fp16" if quant and quant["fmt"] == "fp16" else dtype,
               params_total=int(runner.params), weights_bytes=int(runner.weights_bytes), load_s=round(load_s, 2),
               relpos_patch=None if args.kind == "whisper" else not args.no_relpos_patch, decode_len=decode_len,
               trained=trained, fp32_head=getattr(runner, "fp32_head", None),
               quant=quant["fmt"] if quant else None, quant_impl=quant["resolved"] if quant else None,
               quant_scope=quant["scope"] if quant else None, mx_rounding=quant["mx_rounding"] if quant else None,
               quant_recipe=dict(version=qrec.get("recipe_version"), sha256=qrec.get("recipe_sha256")) if quant
               else None,
               emulated=bool(quant) and quant["resolved"] == "emulate", compile=bool(args.compile),
               threads=torch.get_num_threads(), autocast_dtype=str(amp).replace("torch.", "") if amp else None,
               weights_bytes_resident=int(getattr(runner, "weights_bytes_resident", runner.weights_bytes)))
    stages.append("probe")
    res = probe(runner, waves, durations, refs, device, batch_s=args.batch_s, warmup=args.warmup,
                warmup_1=args.warmup_1, latency_n=args.latency_n, n_tok=n_tok, profile_kernels=args.profile_kernels,
                batch_rows=args.batch_rows)
    hyps, hyps_1 = res.pop("_hyps")
    rec.update(res)
    rec["quant_counters"] = quant_counters(runner.model) if quant else None
    if args.hyps_out:
        write_hyps(Path(args.hyps_out), ids, refs, hyps, hyps_1, durations)
    rec.update(reference=None, sanity=None)
    if quant or args.compile:  # the sanity (module docstring): after both clocks, with the timed model freed
        stages.append("reference")
        del runner, res
        gc.collect()
        if args.compile:
            torch.compiler.reset()
        free_cache(device)
        base = split_system(args.system or args.kind)[0]
        ref = reference_pass(args.kind, args.model, device, dtype, store, args.revision, waves, durations, refs, n_tok,
                             relpos_patch=not args.no_relpos_patch, decode_len=decode_len, batch_s=args.batch_s,
                             batch_rows=args.batch_rows, hf_cache=args.hf_cache, fp32_head=not args.no_fp32_head)
        ref_hyps = ref.pop("_hyps")
        rec["reference"] = dict(ref, system=base)
        rec["sanity"] = cer_sanity(rec["cer_ref_corpus"], ref["cer_ref_corpus"], reference=base,
                                   hyps_differ=sum((a or "") != (b or "") for a, b in zip(hyps, ref_hyps)))
    return rec


if __name__ == "__main__":
    sys.exit(main())
