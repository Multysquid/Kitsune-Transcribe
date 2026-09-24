"""Run the Cohere Transcribe teacher over every shard in data/ and save what distillation needs.

For each input shard  data/shards/<source>/<split>-NNNNN.parquet  this writes
  teacher_out/<source>/<split>-NNNNN.npz     packed arrays (see FORMAT below)
  teacher_out/<source>/<split>-NNNNN.jsonl   one line per utterance: id, hyp, ref, cer, duration, n_tok, truncated
  teacher_out/meta.json                      model id, decoder prompt, k, generation settings

FORMAT (npz):
  ids           (n,)        utterance ids, same order as the jsonl
  tok_offsets   (n+1,)      int64  token span of utterance i is tokens[tok_offsets[i]:tok_offsets[i+1]]
  tokens        (N,)        int16  greedy teacher tokens incl. the final EOS (prompt excluded)
  topk_idx      (N, k)      int16  top-k vocab ids at each step; column 0 is always the greedy token
  topk_logprob  (N, k)      float16 exact teacher log-probabilities of those ids (fp32 LM head, then log_softmax)
  lse           (N,)        float32 logsumexp of the fp32 logits (raw logit = logprob + lse if ever needed)
  cer           (n,)        float32 CER of the teacher hypothesis vs. the dataset transcript (normalised)
  duration      (n,)        float32 seconds
  n_scanned, k, save_encoder, prompt    scalars/1-d: settings that produced this shard (checked on resume)
  enc_offsets   (n+1,)      int64   only with --save-encoder: frame span into enc_states
  enc_states    (F, 1280)   float16 only with --save-encoder: final encoder layer output (unpadded frames)

The remaining probability mass outside the top-k is 1 - sum(exp(topk_logprob)); clamp it at 0, since fp16 rounding of
the log-probs can push the stored top-k mass ~1e-4 above 1.

Utterances longer than 30 s are skipped (above that the feature extractor may chunk the audio and the logits would
no longer be one sequence). Output is atomic per shard; shards whose npz exists with matching settings are skipped,
so the script can be interrupted and re-run. A --limit-rows run writes to teacher_out_smoke/ so it can never be
mistaken for a finished shard.

Usage:
  python scripts/02_teacher_pass.py                       # every shard in the manifest
  python scripts/02_teacher_pass.py --split eval          # teacher baseline CER on the eval sets only
  python scripts/02_teacher_pass.py --limit-shards 1 --limit-rows 64   # quick smoke run -> teacher_out_smoke/
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # measured: 7.1 -> 4.7 GiB reserved

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoProcessor, CohereAsrForConditionalGeneration  # noqa: E402
from transformers.generation import StoppingCriteriaList  # noqa: E402

from kitsune.audio import TARGET_SR, decode_audio  # noqa: E402
from kitsune.generation import RepetitionStop  # noqa: E402
from kitsune.store import read_manifest, read_shard  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MAX_DUR = 30.0  # feature-extractor fast path: max_audio_clip_s (35) - overlap_chunk_second (5)
LANG = "ja"
PUNCT = True


class FP32Head(torch.nn.Linear):
    """LM head evaluated in fp32 so the stored distributions are not quantised by bf16 (0.125-0.25 logit steps)."""

    def forward(self, h):
        return F.linear(h.float(), self.weight, self.bias)


def fp32_head(model):
    old = model.proj_out
    head = FP32Head(old.in_features, old.out_features, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():  # fp32 copy: proj_out is its own bf16 parameter (untied, though bitwise equal to embed_tokens)
        head.weight.copy_(old.weight)
        if old.bias is not None:
            head.bias.copy_(old.bias)
    head.requires_grad_(False)
    model.proj_out = head


def make_batches(durations: np.ndarray, max_batch_seconds: float, max_batch: int) -> list[np.ndarray]:
    """Group indices into batches of similar length; cost is padded length x batch size."""
    order = np.argsort(-durations)  # longest first: the biggest batch runs first, so OOM shows up immediately
    batches, cur, cur_max = [], [], 0.0
    for i in order:
        d = float(durations[i])
        new_max = max(cur_max, d)
        if cur and (new_max * (len(cur) + 1) > max_batch_seconds or len(cur) >= max_batch):
            batches.append(np.array(cur))
            cur, cur_max = [], 0.0
            new_max = d
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(np.array(cur))
    return batches


@torch.inference_mode()
def run_batch(model, inputs, max_dur: float, k: int, save_encoder: bool):
    """`inputs` is the processor output for one batch (featurised on a CPU worker thread)."""
    feats = inputs["input_features"].to(model.device, model.dtype)
    amask = inputs["attention_mask"].to(model.device)
    prompt = inputs["decoder_input_ids"].to(model.device)
    B = feats.shape[0]

    # run the encoder once, hand its output to generate (no recompute) and optionally keep it
    enc = model.model.encoder(input_features=feats, attention_mask=amask)
    # observed ~3.6 tokens per audio second (max ~6.5); 16 + 10*s is a 1.5x margin on the max
    max_new = min(int(16 + 10 * max_dur), model.config.max_position_embeddings - prompt.shape[1] - 1)
    out = model.generate(
        encoder_outputs=enc,
        attention_mask=amask,
        decoder_input_ids=prompt,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_logits=True,
        stopping_criteria=StoppingCriteriaList([RepetitionStop(prompt.shape[1])]),
    )
    T = len(out.logits)  # generated steps
    gen = out.sequences[:, -T:]  # (B, T)
    # reduce step by step: stacking (B, T, V) fp32 would cost >1 GiB at full batch
    lse_l, lp_l, idx_l = [], [], []
    for step_logits in out.logits:  # each (B, V), fp32 from the FP32Head
        sl = step_logits.float()
        lse_l.append(torch.logsumexp(sl, dim=-1))
        v, i = torch.topk(sl, k, dim=-1)
        lp_l.append(v - lse_l[-1][:, None])
        idx_l.append(i)
    lse = torch.stack(lse_l, dim=1)  # (B, T)
    top_lp, top_idx = torch.stack(lp_l, dim=1), torch.stack(idx_l, dim=1)  # (B, T, k)

    eos, pad = model.generation_config.eos_token_id, model.generation_config.pad_token_id
    # a row ends at its EOS, or at the first pad if it was cut by RepetitionStop (generate pads finished rows)
    stop = (gen == eos) | (gen == pad)
    has_stop = stop.any(dim=1)
    first_stop = torch.where(has_stop, stop.int().argmax(dim=1), torch.full((B,), T - 1, dtype=torch.long, device=gen.device))
    ended_with_eos = gen[torch.arange(B, device=gen.device), first_stop] == eos
    lengths = (first_stop + ended_with_eos.long()).tolist()  # keep the EOS token so the student learns to stop
    truncated = (~ended_with_eos).tolist()  # cut by max_new_tokens or by RepetitionStop

    result = dict(
        gen=gen.cpu().numpy(),
        lengths=lengths,
        truncated=truncated,
        top_idx=top_idx.to(torch.int16).cpu().numpy(),
        top_lp=top_lp.to(torch.float16).cpu().numpy(),
        lse=lse.cpu().numpy(),
        prompt=prompt[0].tolist(),
    )
    if save_encoder:
        h = enc.last_hidden_state.to(torch.float16).cpu().numpy()  # (B, F, 1280)
        n_frames = enc.attention_mask.sum(dim=1).tolist() if enc.attention_mask is not None else [h.shape[1]] * B
        result["enc"] = [h[i, : n_frames[i]] for i in range(B)]
    return result


def process_shard(model, processor, shard_path: Path, out_dir: Path, args, pool: ThreadPoolExecutor):
    t0 = time.time()
    table = read_shard(shard_path, columns=["id", "audio", "text", "duration"])
    ids = table.column("id").to_pylist()
    texts = table.column("text").to_pylist()
    durs = np.array(table.column("duration").to_pylist(), dtype=np.float32)
    audio_col = table.column("audio")
    n = len(ids)
    if args.limit_rows:
        n = min(n, args.limit_rows)

    keep = np.array([i for i in range(n) if durs[i] <= MAX_DUR], dtype=np.int64)
    skipped_long = n - len(keep)
    batches = [keep[b] for b in make_batches(durs[keep], args.max_batch_seconds, args.max_batch)]

    # decode + featurise batch j+1.. on CPU threads while the GPU is busy with batch j
    def prepare(idx):
        audios, ok = [], []
        for i in idx:
            try:  # ingest only checked the container header; a truncated payload still fails full decode
                audios.append(decode_audio(audio_col[int(i)].as_py()))
                ok.append(int(i))
            except Exception as e:
                tqdm.write(f"  bad audio, skipping {ids[int(i)]}: {e}")
        ok = np.array(ok, dtype=np.int64)
        if not audios:
            return None, ok
        return processor(audios, language=LANG, punctuation=PUNCT, sampling_rate=TARGET_SR, return_tensors="pt"), ok

    prefetch = max(1, args.prefetch)
    futures = [pool.submit(prepare, b) for b in batches[:prefetch]]
    per = {}  # row index -> dict
    audio_seconds = 0.0
    bad_decode = 0
    prompt = None
    for j, full_b in enumerate(tqdm(batches, desc=shard_path.stem, unit="batch", leave=False)):
        inputs, b = futures[j].result()
        if j + prefetch < len(batches):
            futures.append(pool.submit(prepare, batches[j + prefetch]))
        futures[j] = None
        bad_decode += len(full_b) - len(b)
        if inputs is None:
            continue
        r = run_batch(model, inputs, float(durs[b].max()), args.k, args.save_encoder)
        for bi, i in enumerate(b):
            L = r["lengths"][bi]
            per[int(i)] = dict(
                tokens=r["gen"][bi, :L].astype(np.int16),
                top_idx=r["top_idx"][bi, :L],
                top_lp=r["top_lp"][bi, :L],
                lse=r["lse"][bi, :L],
                truncated=r["truncated"][bi],
                enc=r["enc"][bi] if args.save_encoder else None,
            )
        audio_seconds += float(durs[b].sum())
        prompt = r["prompt"]

    # pack in original row order
    rows = sorted(per)
    hyps = processor.batch_decode([per[i]["tokens"].astype(np.int64) for i in rows], skip_special_tokens=True)
    cers = np.array([cer_fn(h, texts[i]) for h, i in zip(hyps, rows)], dtype=np.float32)
    tok_lens = np.array([len(per[i]["tokens"]) for i in rows], dtype=np.int64)
    packed = dict(
        ids=np.array([ids[i] for i in rows]),
        tok_offsets=np.concatenate([[0], np.cumsum(tok_lens)]),
        tokens=np.concatenate([per[i]["tokens"] for i in rows]) if rows else np.zeros(0, np.int16),
        topk_idx=np.concatenate([per[i]["top_idx"] for i in rows]) if rows else np.zeros((0, args.k), np.int16),
        topk_logprob=np.concatenate([per[i]["top_lp"] for i in rows]) if rows else np.zeros((0, args.k), np.float16),
        lse=np.concatenate([per[i]["lse"] for i in rows]) if rows else np.zeros(0, np.float32),
        cer=cers,
        duration=durs[rows],
        n_scanned=np.int64(n),
        k=np.int64(args.k),
        save_encoder=np.bool_(args.save_encoder),
        prompt=np.array(prompt if prompt is not None else [], dtype=np.int64),
    )
    if args.save_encoder:
        enc_lens = np.array([len(per[i]["enc"]) for i in rows], dtype=np.int64)
        packed["enc_offsets"] = np.concatenate([[0], np.cumsum(enc_lens)])
        packed["enc_states"] = np.concatenate([per[i]["enc"] for i in rows]) if rows else np.zeros((0, 1280), np.float16)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{shard_path.stem}.npz"
    jsonl = out_dir / f"{shard_path.stem}.jsonl"
    with open(jsonl.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as f:
        for h, i, c in zip(hyps, rows, cers):
            f.write(json.dumps(dict(id=ids[i], hyp=h, ref=texts[i], cer=round(float(c), 4), duration=round(float(durs[i]), 3),
                                    n_tok=int(len(per[i]["tokens"])), truncated=bool(per[i]["truncated"])), ensure_ascii=False) + "\n")
    jsonl.with_suffix(".jsonl.tmp").replace(jsonl)
    with open(npz.with_suffix(".npz.tmp"), "wb") as f:
        np.savez(f, **packed)
    npz.with_suffix(".npz.tmp").replace(npz)  # npz last: its presence marks the shard as done

    return dict(rows=len(rows), skipped_long=skipped_long, bad_decode=bad_decode, audio_s=audio_seconds,
                wall_s=time.time() - t0, mean_cer=float(cers.mean()) if len(cers) else float("nan"),
                truncated=int(sum(per[i]["truncated"] for i in rows)), prompt=prompt)


def shard_is_done(npz: Path, expected_rows: int, args) -> bool:
    """Done only if the output exists AND was produced over the whole shard with the same settings."""
    if not npz.exists():
        return False
    try:
        z = np.load(npz)
        return (int(z["n_scanned"]) == expected_rows and int(z["k"]) == args.k
                and bool(z["save_encoder"]) == args.save_encoder)
    except Exception:
        return False  # old/partial/corrupt file -> recompute


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help=f"output root (default: {ROOT / 'teacher_out'}, or teacher_out_smoke with --limit-rows)")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--sources", nargs="*", default=None, help="restrict to these sources (default: all in manifest)")
    ap.add_argument("--k", type=int, default=16, help="top-k log-probs to keep per step")
    ap.add_argument("--max-batch-seconds", type=float, default=240.0, help="padded audio seconds per batch (VRAM knob)")
    ap.add_argument("--max-batch", type=int, default=96, help="decode cost per step is batch-independent, so large is good")
    ap.add_argument("--prefetch", type=int, default=2, help="batches decoded+featurised ahead on CPU (min 1)")
    ap.add_argument("--save-encoder", action="store_true",
                    help="also store final encoder states (fp16, ~32 KB per audio second: ~45 GB for 400 h)")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--limit-rows", type=int, default=None, help="debug: only the first N rows of each shard")
    ap.add_argument("--force", action="store_true", help="recompute shards that already have output; overwrite meta.json")
    args = ap.parse_args()

    root = Path(args.data)
    out_root = Path(args.out) if args.out else ROOT / ("teacher_out_smoke" if args.limit_rows else "teacher_out")
    shards = read_manifest(root)
    if args.split != "all":
        shards = [s for s in shards if s.split == args.split]
    if args.sources:
        shards = [s for s in shards if s.source in args.sources]
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        sys.exit("no shards selected - run scripts/01_prepare_data.py first")

    # settings must be uniform across teacher_out, otherwise the student loader gets heterogeneous shards
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "meta.json"
    settings = dict(model=MODEL_ID, language=LANG, punctuation=PUNCT, k=args.k, save_encoder=args.save_encoder)
    if meta_path.exists() and not args.force:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        diff = {key: (old.get(key), val) for key, val in settings.items() if old.get(key) != val}
        if diff:
            sys.exit(f"{meta_path} was written with different settings {diff}; use --force (recomputes everything) or a different --out")

    todo = [s for s in shards if args.force or not shard_is_done(out_root / s.source / f"{Path(s.path).stem}.npz", s.rows, args)]
    print(f"output: {out_root}\n{len(todo)}/{len(shards)} shards to do, {sum(s.hours for s in todo):.1f} h audio "
          f"({len(shards) - len(todo)} already done)")
    if not todo:
        return

    free, total = torch.cuda.mem_get_info()
    print(f"GPU free before load: {free / 2**30:.2f} of {total / 2**30:.1f} GiB")
    if free < 6 * 2**30:
        print("WARNING: <6 GiB free - other apps are holding VRAM; the pass may spill to shared memory and slow down 5x")
    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    fp32_head(model)
    print(f"teacher loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    totals = dict(rows=0, skipped_long=0, bad_decode=0, audio_s=0.0, wall_s=0.0, truncated=0)
    remaining_h = sum(s.hours for s in todo)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for s in tqdm(todo, desc="shards", unit="shard"):
            shard_path = shard_path_of(root, s)
            r = process_shard(model, processor, shard_path, out_root / s.source, args, pool)
            for key in totals:
                totals[key] += r[key]
            remaining_h -= s.hours
            rtf = totals["wall_s"] / max(totals["audio_s"], 1e-6)
            eta_h = remaining_h * rtf
            tqdm.write(f"{s.source}/{shard_path.stem}: {r['rows']} utts, {r['audio_s'] / 3600:.2f} h in {r['wall_s'] / 60:.1f} min "
                       f"(RTF {r['wall_s'] / max(r['audio_s'], 1e-6):.4f}), CER vs ref {r['mean_cer']:.3f}, truncated {r['truncated']}, "
                       f">30s {r['skipped_long']}, bad audio {r['bad_decode']} | running RTF {rtf:.4f}, ETA {eta_h:.2f} h | VRAM alloc "
                       f"{torch.cuda.max_memory_allocated() / 2**30:.2f} reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} "
                       f"free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
            if (args.force or not meta_path.exists()) and r["prompt"]:
                meta_path.write_text(json.dumps(dict(
                    **settings, model_revision=getattr(model.config, "_commit_hash", None),  # the commit `main` resolved to
                    decoder_prompt_ids=r["prompt"],
                    decoder_prompt_tokens=processor.tokenizer.convert_ids_to_tokens(r["prompt"]),
                    eos_token_id=model.generation_config.eos_token_id, pad_token_id=model.generation_config.pad_token_id,
                    vocab_size=model.config.vocab_size, encoder_hidden_size=model.config.encoder_config.hidden_size,
                    decoding="greedy", lm_head_dtype="float32", model_dtype="bfloat16",
                ), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ndone: {totals['rows']} utterances, {totals['audio_s'] / 3600:.2f} h audio in {totals['wall_s'] / 3600:.2f} h wall "
          f"(RTF {totals['wall_s'] / max(totals['audio_s'], 1e-6):.4f}); truncated={totals['truncated']}, "
          f"skipped >30s={totals['skipped_long']}, bad audio={totals['bad_decode']}")


def shard_path_of(root: Path, s) -> Path:
    return root / s.path


if __name__ == "__main__":
    main()
