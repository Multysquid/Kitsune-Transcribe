"""Small ASR bake-off: the same random clips through several models, scored with our CER (kitsune.text).

A one-off decision aid (is Cohere Transcribe still the right teacher?), separate from the training pipeline. It reads
our eval / hold-out shards and writes everything under runs/bakeoff/ (git-ignored):

  sample.jsonl            the drawn clips: id, set, shard, row, duration, ref
  hyps/<model>.jsonl      one line per clip: id, hyp
  hyps/<model>.meta.json  repo, revision, dtype, batch, wall time, audio seconds, RTF, peak VRAM
  report.md / report.json per-set corpus CER, mean/median per-clip CER, RTF

The laptop GPU crashes under long loads, so each model runs in its own process (one `run` call), the whole sample is
~1.5 h of audio, and every batch waits while the GPU is above --max-temp. A `run` whose hyps file is complete is
skipped, so the steps can simply be re-issued after a crash.

Usage:
  python tools/asr_bakeoff.py sample --per-set 200 --seed 0
  python tools/asr_bakeoff.py run --model cohere        # whisper | cohere | qwen | parakeet
  python tools/asr_bakeoff.py score
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from kitsune.audio import TARGET_SR, decode_audio  # noqa: E402
from kitsune.text import cer, normalize_ja  # noqa: E402

OUT = ROOT / "runs" / "bakeoff"
SHARDS = ROOT / "data" / "shards"
# set name -> shard glob; galgame and emilia are the monitor-only hold-outs, never trained on
SETS = {
    "jsut": "eval_jsut/eval-*.parquet",
    "cv8": "eval_cv8/eval-*.parquet",
    "reazon": "eval_reazon/eval-*.parquet",
    "galgame": "galgame/eval-*.parquet",
    "emilia": "eval_emilia/eval-*.parquet",
}
MAX_DUR = 30.0  # Cohere and Whisper take at most one 30 s window without chunking

MODELS = {
    "whisper": dict(repo="openai/whisper-large-v3", revision=None, batch=8),
    "cohere": dict(repo="CohereLabs/cohere-transcribe-03-2026", revision=None, batch_s=400.0),
    "qwen": dict(repo="Qwen/Qwen3-ASR-1.7B-hf", revision="bcd2b5b7f32b480ab5790554cfa8347f246a14f3", batch=8),
    # converted from nvidia/parakeet-tdt_ctc-0.6b-ja@44edb27 with transformers' convert_nemo_to_hf.py (TDT head)
    "parakeet": dict(repo=str(ROOT / "cache" / "parakeet-tdt_ctc-0.6b-ja-hf"), revision=None, batch_s=600.0),
}


# ---------------------------------------------------------------- sample

def cmd_sample(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, pattern in SETS.items():
        cand = []
        for shard in sorted(SHARDS.glob(pattern)):
            t = pq.read_table(shard, columns=["id", "text", "duration"])
            for r, (i, text, d) in enumerate(zip(*(t.column(c).to_pylist() for c in ("id", "text", "duration")))):
                if 0.3 <= d <= MAX_DUR and normalize_ja(text):
                    cand.append(dict(id=i, set=name, shard=str(shard.relative_to(ROOT)), row=r, duration=d, ref=text))
        pick = rng.choice(len(cand), size=min(args.per_set, len(cand)), replace=False)
        rows += [cand[int(k)] for k in sorted(pick)]
        print(f"{name:8s} {len(pick)} of {len(cand)} clips, {sum(cand[int(k)]['duration'] for k in pick) / 60:.1f} min")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} h -> {OUT / 'sample.jsonl'}")


def load_sample() -> list[dict]:
    with open(OUT / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_audio(rows: list[dict]) -> dict[str, np.ndarray]:
    by_shard: dict[str, list[dict]] = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    waves = {}
    for shard, rs in by_shard.items():
        col = pq.read_table(ROOT / shard, columns=["audio"]).column("audio")
        for r in rs:
            waves[r["id"]] = decode_audio(col[r["row"]].as_py())
    return waves


# ---------------------------------------------------------------- GPU guard

def gpu_temp() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def wait_cool(max_temp: int, resume_temp: int) -> float:
    """Block while the GPU is above max_temp, until it is back at resume_temp. Returns seconds waited."""
    t = gpu_temp()
    if t is None or t < max_temp:
        return 0.0
    t0 = time.time()
    print(f"  GPU at {t} C, pausing until {resume_temp} C", flush=True)
    while t is not None and t > resume_temp:
        time.sleep(5)
        t = gpu_temp()
    return time.time() - t0


# ---------------------------------------------------------------- models

def batches_by_count(ids, n):
    return [ids[i:i + n] for i in range(0, len(ids), n)]


def batches_by_seconds(ids, durs, max_s):
    out, cur, longest = [], [], 0.0
    for i in ids:
        longest_new = max(longest, durs[i])
        if cur and longest_new * (len(cur) + 1) > max_s:
            out.append(cur)
            cur, longest_new = [], durs[i]
        cur.append(i)
        longest = longest_new
    if cur:
        out.append(cur)
    return out


class Whisper:
    def __init__(self, cfg, torch):
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        self.torch, self.cfg = torch, cfg
        self.proc = AutoProcessor.from_pretrained(cfg["repo"], revision=cfg["revision"])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=torch.float16).to("cuda").eval()
        self.dtype = "float16"

    def batches(self, ids, durs):
        return batches_by_count(ids, self.cfg["batch"])

    def __call__(self, waves):
        inp = self.proc(waves, sampling_rate=TARGET_SR, return_tensors="pt")
        feats = inp.input_features.to("cuda", self.torch.float16)
        out = self.model.generate(feats, language="ja", task="transcribe", num_beams=1, do_sample=False,
                                  max_new_tokens=224)
        return [t.strip() for t in self.proc.batch_decode(out, skip_special_tokens=True)]


class Cohere:
    """Same settings as scripts/02_teacher_pass.py: bf16, sdpa, prompt ja + punctuation, greedy, repetition stop."""

    def __init__(self, cfg, torch):
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        from transformers.generation import StoppingCriteriaList

        from kitsune.generation import RepetitionStop
        self.torch, self.cfg, self.SCL, self.RS = torch, cfg, StoppingCriteriaList, RepetitionStop
        self.proc = AutoProcessor.from_pretrained(cfg["repo"])
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            cfg["repo"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.dtype = "bfloat16"

    def batches(self, ids, durs):
        return batches_by_seconds(ids, durs, self.cfg["batch_s"])

    def __call__(self, waves):
        inp = self.proc(waves, language="ja", punctuation=True, sampling_rate=TARGET_SR, return_tensors="pt")
        prompt = inp["decoder_input_ids"].to("cuda")
        max_dur = max(len(w) for w in waves) / TARGET_SR
        max_new = min(int(16 + 10 * max_dur), self.model.config.max_position_embeddings - prompt.shape[1] - 1)
        out = self.model.generate(
            input_features=inp["input_features"].to("cuda", self.torch.bfloat16),
            attention_mask=inp["attention_mask"].to("cuda"), decoder_input_ids=prompt,
            max_new_tokens=max_new, do_sample=False, num_beams=1,
            stopping_criteria=self.SCL([self.RS(prompt.shape[1])]))
        gen = out[:, prompt.shape[1]:]
        return [t.strip() for t in self.proc.batch_decode(gen, skip_special_tokens=True)]


class Qwen:
    def __init__(self, cfg, torch):
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        self.torch, self.cfg = torch, cfg
        self.proc = AutoProcessor.from_pretrained(cfg["repo"], revision=cfg["revision"])
        self.proc.tokenizer.padding_side = "left"  # decoder-only LM: batched generation needs left padding
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=torch.bfloat16).to("cuda").eval()
        self.dtype = "bfloat16"

    def batches(self, ids, durs):
        return batches_by_count(ids, self.cfg["batch"])

    def __call__(self, waves):
        inp = self.proc.apply_transcription_request(audio=list(waves), language="ja")
        inp = {k: (v.to("cuda", self.torch.bfloat16) if v.is_floating_point() else v.to("cuda"))
               for k, v in inp.items()}
        out = self.model.generate(**inp, max_new_tokens=256, do_sample=False, num_beams=1)
        gen = out[:, inp["input_ids"].shape[1]:]
        return [self.proc.decode(g, return_format="transcription_only").strip() for g in gen]


class Parakeet:
    def __init__(self, cfg, torch):
        from transformers import AutoProcessor, ParakeetForTDT
        self.torch, self.cfg = torch, cfg
        self.proc = AutoProcessor.from_pretrained(cfg["repo"])
        self.model = ParakeetForTDT.from_pretrained(cfg["repo"], dtype=torch.bfloat16).to("cuda").eval()
        self.dtype = "bfloat16"

    def batches(self, ids, durs):
        return batches_by_seconds(ids, durs, self.cfg["batch_s"])

    def __call__(self, waves):
        inp = self.proc(list(waves), sampling_rate=TARGET_SR, return_tensors="pt")
        inp = {k: (v.to("cuda", self.torch.bfloat16) if v.is_floating_point() else v.to("cuda"))
               for k, v in inp.items()}
        out = self.model.generate(**inp)
        seqs = out.sequences if hasattr(out, "sequences") else out
        return [t.strip() for t in self.proc.batch_decode(seqs, skip_special_tokens=True)]


RUNNERS = {"whisper": Whisper, "cohere": Cohere, "qwen": Qwen, "parakeet": Parakeet}


def cmd_run(args):
    import torch

    rows = load_sample()
    hyp_path = OUT / "hyps" / f"{args.model}.jsonl"
    if hyp_path.exists() and sum(1 for _ in open(hyp_path, encoding="utf-8")) == len(rows) and not args.force:
        print(f"{args.model}: already complete ({hyp_path})")
        return
    cfg = MODELS[args.model]
    print(f"loading audio for {len(rows)} clips ...", flush=True)
    waves = load_audio(rows)
    durs = {r["id"]: r["duration"] for r in rows}
    ids = sorted(durs, key=durs.get, reverse=True)  # longest first: an OOM shows up at once

    print(f"loading {args.model} ({cfg['repo']}) ...", flush=True)
    runner = RUNNERS[args.model](cfg, torch)
    with torch.inference_mode():  # warm-up so kernel selection is not timed
        runner([waves[ids[-1]]])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    hyps, busy, paused = {}, 0.0, 0.0
    batches = runner.batches(ids, durs)
    for j, b in enumerate(batches):
        paused += wait_cool(args.max_temp, args.resume_temp)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = runner([waves[i] for i in b])
        torch.cuda.synchronize()
        busy += time.perf_counter() - t0
        hyps.update(zip(b, out))
        if j % 10 == 0 or j == len(batches) - 1:
            print(f"  batch {j + 1}/{len(batches)}  {busy:.0f} s busy  GPU {gpu_temp()} C", flush=True)
        if args.pause:
            time.sleep(args.pause)

    audio_s = sum(durs.values())
    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hyp_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(id=r["id"], hyp=hyps[r["id"]]), ensure_ascii=False) + "\n")
    meta = dict(model=args.model, repo=cfg["repo"], revision=cfg["revision"], dtype=runner.dtype,
                batching={k: v for k, v in cfg.items() if k.startswith("batch")}, clips=len(rows),
                audio_s=round(audio_s, 1), busy_s=round(busy, 1), paused_s=round(paused, 1),
                rtf=round(busy / audio_s, 5), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    import transformers
    meta["transformers"] = transformers.__version__
    (OUT / "hyps" / f"{args.model}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- score

def corpus_cer(pairs, cap: float | None = None):
    """Character-weighted CER over (hyp, ref) pairs: sum of edits / sum of reference characters. With `cap`, each
    clip's CER is clipped first, so one hallucination loop (Whisper: 150x "あ") cannot outweigh a whole set."""
    num = den = 0.0
    for hyp, ref in pairs:
        n = len(normalize_ja(ref))
        if n:
            c = cer(hyp, ref)
            num += (min(c, cap) if cap is not None else c) * n
            den += n
    return num / den if den else float("nan")


def cmd_score(args):
    rows = load_sample()
    sets = list(SETS)
    report = {}
    for meta_path in sorted((OUT / "hyps").glob("*.meta.json")):
        name = meta_path.name.removesuffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with open(OUT / "hyps" / f"{name}.jsonl", encoding="utf-8") as f:
            hyps = {d["id"]: d["hyp"] for d in map(json.loads, f)}
        per_set = {}
        for s in sets + ["all"]:
            rs = [r for r in rows if s == "all" or r["set"] == s]
            clip = [cer(hyps[r["id"]], r["ref"]) for r in rs]
            pairs = [(hyps[r["id"]], r["ref"]) for r in rs]
            per_set[s] = dict(corpus=corpus_cer(pairs), capped=corpus_cer(pairs, cap=1.0),
                              mean=statistics.mean(clip), median=statistics.median(clip),
                              runaway=sum(1 for c in clip if c > 1.0),
                              empty=sum(1 for r in rs if not normalize_ja(hyps[r["id"]])), n=len(rs))
        report[name] = dict(meta=meta, cer=per_set)
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["| model | " + " | ".join(sets) + " | all | RTF | peak VRAM |",
             "|---" * (len(sets) + 4) + "|"]
    for name, r in report.items():
        cells = [f"{100 * r['cer'][s]['corpus']:.1f}" for s in sets + ["all"]]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['meta']['rtf']:.4f} | {r['meta']['peak_vram_gb']} GB |")
    lines.append("\nCorpus CER % (edits / reference characters) with kitsune.text normalisation.")
    for key, title in (("capped", "Corpus CER % with each clip capped at 100 % (hallucination loops count once):"),
                       ("median", "Median per-clip CER %:")):
        lines += ["", title, "", "| model | " + " | ".join(sets) + " | all |", "|---" * (len(sets) + 2) + "|"]
        for name, r in report.items():
            lines.append(f"| {name} | " + " | ".join(f"{100 * r['cer'][s][key]:.1f}" for s in sets + ["all"]) + " |")
    lines += ["", "Runaway clips (CER > 100 %, i.e. more inserted characters than the reference has):", "",
              "| model | " + " | ".join(sets) + " | all |", "|---" * (len(sets) + 2) + "|"]
    for name, r in report.items():
        lines.append(f"| {name} | " + " | ".join(str(r["cer"][s]["runaway"]) for s in sets + ["all"]) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--per-set", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--max-temp", type=int, default=80, help="pause batches while the GPU is at or above this (C)")
    p.add_argument("--resume-temp", type=int, default=70)
    p.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between batches")
    p.add_argument("--force", action="store_true")
    sub.add_parser("score")
    args = ap.parse_args()
    {"sample": cmd_sample, "run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
