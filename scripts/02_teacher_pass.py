"""Run the Cohere Transcribe teacher over every shard in data/ and save what distillation needs.

For each input shard  data/shards/<source>/<split>-NNNNN.parquet  this writes
  teacher_out/<source>/<split>-NNNNN.npz     packed arrays (see FORMAT below)
  teacher_out/<source>/<split>-NNNNN.jsonl   one line per utterance: id, hyp, ref, cer, duration, n_tok, truncated
  teacher_out/meta.json                      model id and pinned commit, decoder prompt, k, generation settings

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
no longer be one sequence). Output is atomic per shard (jsonl, then npz, each fsynced); a shard is done when its npz
holds this shard's ids (an ordered subsequence: skipped rows are left out) over all its rows with matching settings,
so the script can be interrupted and re-run. A --limit-rows run writes to teacher_out_smoke/ so it can never be
mistaken for a finished shard.

The shards run on kitsune/labelpass.py: one stream of batches across shard boundaries, a background writer, an OOM
halves the batch. The label box adds --follow/--heartbeat/--progress (a lane behind a still-ingesting 01),
--shard-mod/--shard-rem (two lanes on one GPU), --adopt-from/--adopt-only (reuse the laptop's finished shards
byte-identically when the ids match; the gate sets are never recomputed) and --strict-existing (an output that does
not match the shard exits 65 instead of being recomputed over).

Usage:
  python scripts/02_teacher_pass.py                       # every shard in the manifest
  python scripts/02_teacher_pass.py --split eval          # teacher baseline CER on the eval sets only
  python scripts/02_teacher_pass.py --limit-shards 1 --limit-rows 64   # quick smoke run -> teacher_out_smoke/
  python scripts/02_teacher_pass.py --out labels/full/teacher_out --follow ingest.done --shard-mod 2 --shard-rem 0       --adopt-from seed/teacher_out --adopt-only eval_jsut eval_cv8 eval_reazon --strict-existing   # a label-box lane
"""
import argparse
import json
import os
import shutil
import sys
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
from kitsune.labelpass import (ShardJob, append_progress, ordered_subsequence, partition_ok, run, touch,  # noqa: E402
                               write_json_atomic, write_pair)
from kitsune.store import fsync_path, read_manifest, read_shard, shard_ids  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
# pinned like the datasets in 01: a teacher repo that moves between passes would mix two teachers' labels in teacher_out
MODEL_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"
MAX_DUR = 30.0  # feature-extractor fast path: max_audio_clip_s (35) - overlap_chunk_second (5)
LANG = "ja"
PUNCT = True
INTEGRITY_EXIT = 65  # an existing output that does not match: never recomputed over (the label box stops)


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


def load_table(root: Path, job: ShardJob, limit_rows: int | None) -> dict:
    """One shard's columns, plus the rows the pass keeps (<= MAX_DUR s)."""
    table = read_shard(root / job.info.path, columns=["id", "audio", "text", "duration"])
    ids = table.column("id").to_pylist()
    durs = np.array(table.column("duration").to_pylist(), dtype=np.float32)
    n = len(ids)
    if limit_rows:
        n = min(n, limit_rows)
    keep = np.array([i for i in range(n) if durs[i] <= MAX_DUR], dtype=np.int64)
    return dict(ids=ids, texts=table.column("text").to_pylist(), durs=durs, audio_col=table.column("audio"), n=n,
                keep=keep, skipped_long=n - len(keep), prompt=None)


def plan_batches(t: dict, args) -> list[np.ndarray]:
    keep = t["keep"]
    return [keep[b] for b in make_batches(t["durs"][keep], args.max_batch_seconds, args.max_batch)]


def prepare(processor, t: dict, idx):
    """Decode + featurise one batch on a CPU worker thread while the GPU is busy with an earlier one."""
    audios, ok = [], []
    for i in idx:
        try:  # ingest only checked the container header; a truncated payload still fails full decode
            audios.append(decode_audio(t["audio_col"][int(i)].as_py()))
            ok.append(int(i))
        except Exception as e:
            tqdm.write(f"  bad audio, skipping {t['ids'][int(i)]}: {e}")
    ok = np.array(ok, dtype=np.int64)
    if not audios:
        return None, ok
    return processor(audios, language=LANG, punctuation=PUNCT, sampling_rate=TARGET_SR, return_tensors="pt"), ok


def gpu(model, inputs, ok, t: dict, args) -> dict:
    r = run_batch(model, inputs, float(t["durs"][ok].max()), args.k, args.save_encoder)
    per = {}
    for bi, i in enumerate(ok):
        L = r["lengths"][bi]
        per[int(i)] = dict(
            tokens=r["gen"][bi, :L].astype(np.int16),
            top_idx=r["top_idx"][bi, :L],
            top_lp=r["top_lp"][bi, :L],
            lse=r["lse"][bi, :L],
            truncated=r["truncated"][bi],
            enc=r["enc"][bi] if args.save_encoder else None,
        )
    t["prompt"] = r["prompt"]
    return per


def pack_teacher_shard(ids, texts, durs, n, per, rows, k, save_encoder, prompt) -> dict:
    """The npz arrays of one shard (FORMAT above), rows in original row order. `per[i]["hyp"]` is the decoded
    hypothesis of row i; its CER against texts[i] is stored. Byte-for-byte the packing of tests/fixtures.py."""
    cers = np.array([cer_fn(per[i]["hyp"], texts[i]) for i in rows], dtype=np.float32)
    tok_lens = np.array([len(per[i]["tokens"]) for i in rows], dtype=np.int64)
    packed = dict(
        ids=np.array([ids[i] for i in rows]),
        tok_offsets=np.concatenate([[0], np.cumsum(tok_lens)]),
        tokens=np.concatenate([per[i]["tokens"] for i in rows]) if rows else np.zeros(0, np.int16),
        topk_idx=np.concatenate([per[i]["top_idx"] for i in rows]) if rows else np.zeros((0, k), np.int16),
        topk_logprob=np.concatenate([per[i]["top_lp"] for i in rows]) if rows else np.zeros((0, k), np.float16),
        lse=np.concatenate([per[i]["lse"] for i in rows]) if rows else np.zeros(0, np.float32),
        cer=cers,
        duration=durs[rows],
        n_scanned=np.int64(n),
        k=np.int64(k),
        save_encoder=np.bool_(save_encoder),
        prompt=np.array(prompt if prompt is not None else [], dtype=np.int64),
    )
    if save_encoder:
        enc_lens = np.array([len(per[i]["enc"]) for i in rows], dtype=np.int64)
        packed["enc_offsets"] = np.concatenate([[0], np.cumsum(enc_lens)])
        packed["enc_states"] = np.concatenate([per[i]["enc"] for i in rows]) if rows else np.zeros((0, 1280), np.float16)
    return packed


def jsonl_rows(packed: dict, ids, texts, durs, per, rows) -> list[dict]:
    return [dict(id=ids[i], hyp=per[i]["hyp"], ref=texts[i], cer=round(float(c), 4), duration=round(float(durs[i]), 3),
                 n_tok=int(len(per[i]["tokens"])), truncated=bool(per[i]["truncated"]))
            for i, c in zip(rows, packed["cer"])]


def finish_shard(processor, job: ShardJob, t: dict, per: dict, stats: dict, args) -> dict:
    """CER + pack + write one shard (on the runner's writer thread)."""
    rows = sorted(per)
    hyps = processor.batch_decode([per[i]["tokens"].astype(np.int64) for i in rows], skip_special_tokens=True)
    for h, i in zip(hyps, rows):
        per[i]["hyp"] = h
    packed = pack_teacher_shard(t["ids"], t["texts"], t["durs"], t["n"], per, rows, args.k, args.save_encoder,
                                t["prompt"])
    write_pair(job.out_dir / f"{job.stem}.jsonl", jsonl_rows(packed, t["ids"], t["texts"], t["durs"], per, rows),
               job.out_dir / f"{job.stem}.npz", packed, compressed=False)  # npz last: its presence marks the shard done
    cers = packed["cer"]
    return dict(rows=len(rows), skipped_long=t["skipped_long"], bad_decode=stats["bad_decode"],
                audio_s=float(t["durs"][rows].sum()) if rows else 0.0, mean_cer=float(cers.mean()) if len(cers) else None,
                truncated=int(sum(per[i]["truncated"] for i in rows)), prompt=t["prompt"])


def _jsonl_ids(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln)["id"] for ln in f if ln.strip()]


def _jsonl_ends(path: Path) -> tuple:
    """(first id, last id) of a jsonl, (None, None) when empty."""
    first = last = None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                last = ln
                if first is None:
                    first = ln
    return (json.loads(first)["id"] if first else None, json.loads(last)["id"] if last else None)


def shard_is_done(npz: Path, shard_ids: list[str], args) -> bool:
    """Done only if the npz was produced over exactly these shard rows with the same settings: n_scanned is the
    shard's row count, its ids are the shard's ids with at most some rows skipped (> 30 s, bad audio), and its jsonl
    is there with the same first and last id. Equal row counts alone would accept another ingest's rows."""
    if not npz.exists():
        return False
    try:
        with np.load(npz) as z:
            if (int(z["n_scanned"]) != len(shard_ids) or int(z["k"]) != args.k
                    or bool(z["save_encoder"]) != args.save_encoder):
                return False
            ids = [str(x) for x in z["ids"].tolist()]
        if not ordered_subsequence(ids, shard_ids):
            return False
        jsonl = npz.with_suffix(".jsonl")
        if not jsonl.is_file():
            return False
        return _jsonl_ends(jsonl) == ((ids[0], ids[-1]) if ids else (None, None))
    except Exception:
        return False  # old/partial/corrupt file -> recompute (or exit 65 under --strict-existing)


def _copy_atomic(src: Path, dst: Path):
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    fsync_path(tmp)
    os.replace(tmp, dst)


def try_adopt(src_dir: Path, job: ShardJob, shard_ids: list[str], prompt, args) -> bool:
    """Copy a finished shard from another teacher_out byte-identically, if it is provably this shard: done over these
    ids with these settings, made with this decoder prompt, and its jsonl holds exactly the npz's ids."""
    if prompt is None:
        return False
    npz = src_dir / job.info.source / f"{job.stem}.npz"
    jsonl = npz.with_suffix(".jsonl")
    if not shard_is_done(npz, shard_ids, args):
        return False
    try:
        with np.load(npz) as z:
            ids = [str(x) for x in z["ids"].tolist()]
            # a shard none of whose rows reached the GPU (all >30 s or bad audio) was written with prompt=[]
            if ids and z["prompt"].tolist() != list(prompt):
                return False
        if _jsonl_ids(jsonl) != ids:
            return False
    except Exception:
        return False
    job.out_dir.mkdir(parents=True, exist_ok=True)
    _copy_atomic(jsonl, job.out_dir / jsonl.name)  # jsonl then npz, as write_pair
    _copy_atomic(npz, job.out_dir / npz.name)
    return True


def check_meta(meta_path: Path, settings: dict, strict: bool = False):
    """Exit if an existing meta.json was written with other settings (65 under --strict-existing). A meta.json from
    before the teacher was pinned has no model_revision; that output all came from b1eacc2 (the only snapshot ever
    cached, and still `main`), so it counts as the pin and the key is back-filled (meta.json is otherwise only written
    when missing or with --force)."""
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    backfill = "model_revision" not in old
    old.setdefault("model_revision", MODEL_REVISION)
    diff = {key: (old.get(key), val) for key, val in settings.items() if old.get(key) != val}
    if diff:
        msg = f"{meta_path} was written with different settings {diff}; use --force (recomputes everything) or a different --out"
        if strict:
            print(msg, file=sys.stderr)
            sys.exit(INTEGRITY_EXIT)
        sys.exit(msg)
    if backfill:
        write_json_atomic(meta_path, old)


def meta_prompt(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("decoder_prompt_ids")
    except (OSError, ValueError):
        return None


def integrity_exit(msg: str):
    print(msg, file=sys.stderr, flush=True)
    sys.exit(INTEGRITY_EXIT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help=f"output root (default: {ROOT / 'teacher_out'}, or teacher_out_smoke with --limit-rows)")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--sources", nargs="*", default=None, help="restrict to these sources (default: all in manifest)")
    ap.add_argument("--k", type=int, default=16, help="top-k log-probs to keep per step")
    ap.add_argument("--max-batch-seconds", type=float, default=240.0, help="padded audio seconds per batch (VRAM knob)")
    ap.add_argument("--max-batch", type=int, default=96, help="decode cost per step is batch-independent, so large is good")
    ap.add_argument("--prefetch", type=int, default=2, help="batches decoded+featurised ahead on CPU, across shards (min 1)")
    ap.add_argument("--workers", type=int, default=4, help="CPU threads that decode + featurise batches")
    ap.add_argument("--torch-threads", type=int, default=None, help="torch.set_num_threads for this process")
    ap.add_argument("--save-encoder", action="store_true",
                    help="also store final encoder states (fp16, ~32 KB per audio second: ~45 GB for 400 h)")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--limit-rows", type=int, default=None, help="debug: only the first N rows of each shard")
    ap.add_argument("--force", action="store_true", help="recompute shards that already have output; overwrite meta.json")
    ap.add_argument("--follow", default=None, metavar="MARKER",
                    help="keep polling the manifest for new shards; exit once MARKER exists and nothing is left")
    ap.add_argument("--heartbeat", default=None, help="file touched after every batch and every follow poll")
    ap.add_argument("--progress", default=None, help="append one JSON line per finished shard to this file")
    ap.add_argument("--poll-s", type=float, default=30.0, help="follow-mode poll interval")
    ap.add_argument("--shard-mod", type=int, default=1, help="run only shards with crc32(path) %% MOD == REM")
    ap.add_argument("--shard-rem", type=int, default=0)
    ap.add_argument("--vram-fraction", type=float, default=None, help="torch.cuda.set_per_process_memory_fraction")
    ap.add_argument("--strict-existing", action="store_true",
                    help=f"an existing output that is not done over the shard's ids, or a meta.json mismatch, exits {INTEGRITY_EXIT}")
    ap.add_argument("--adopt-from", default=None, metavar="DIR",
                    help="copy byte-identically a finished shard of DIR (another teacher_out) whose ids and prompt match")
    ap.add_argument("--adopt-only", nargs="*", default=[], metavar="SRC",
                    help=f"sources that are only adopted, never computed: a shard that cannot be adopted exits {INTEGRITY_EXIT}")
    args = ap.parse_args()
    if not 0 <= args.shard_rem < args.shard_mod:
        ap.error("--shard-rem must be in [0, --shard-mod)")
    if args.adopt_only and not args.adopt_from:
        ap.error("--adopt-only needs --adopt-from")

    root = Path(args.data)
    out_root = Path(args.out) if args.out else ROOT / ("teacher_out_smoke" if args.limit_rows else "teacher_out")
    follow = Path(args.follow) if args.follow else None
    heartbeat = Path(args.heartbeat) if args.heartbeat else None
    progress = Path(args.progress) if args.progress else None
    adopt_from = Path(args.adopt_from) if args.adopt_from else None

    # settings must be uniform across teacher_out, otherwise the student loader gets heterogeneous shards
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "meta.json"
    settings = dict(model=MODEL_ID, model_revision=MODEL_REVISION, language=LANG, punctuation=PUNCT, k=args.k,
                    save_encoder=args.save_encoder)
    if meta_path.exists() and not args.force:
        check_meta(meta_path, settings, strict=args.strict_existing)
    adopt_prompt = None
    if adopt_from is not None:
        src_meta = adopt_from / "meta.json"
        if src_meta.exists():
            try:
                check_meta_readonly(src_meta, settings)
                adopt_prompt = meta_prompt(meta_path) if meta_path.exists() else meta_prompt(src_meta)
            except ValueError as e:
                print(f"not adopting from {adopt_from}: {e}")
        else:
            print(f"not adopting from {adopt_from}: no meta.json")

    def select():
        shards = read_manifest(root)
        if args.split != "all":
            shards = [s for s in shards if s.split == args.split]
        if args.sources:
            shards = [s for s in shards if s.source in args.sources]
        shards = [s for s in shards if partition_ok(s.path, args.shard_mod, args.shard_rem)]
        if args.limit_shards:
            shards = shards[: args.limit_shards]
        return shards

    done: set = set()  # (source, stem) found done or adopted in this process: never checked again

    def todo() -> list[ShardJob]:
        jobs = []
        for s in select():
            stem = Path(s.path).stem
            if (s.source, stem) in done:
                continue
            job = ShardJob(s, stem, out_root / s.source)
            if not args.force:
                npz = job.out_dir / f"{stem}.npz"
                ids = shard_ids(root, s) if (npz.exists() or adopt_from is not None) else None
                if npz.exists():
                    if shard_is_done(npz, ids, args):
                        done.add((s.source, stem))
                        continue
                    if args.strict_existing:
                        integrity_exit(f"{s.source}/{stem}: {npz} exists but is not done over this shard's ids "
                                       f"(other rows or settings); refusing to recompute over it")
                if adopt_from is not None and try_adopt(adopt_from, job, ids, adopt_prompt, args):
                    done.add((s.source, stem))
                    print(f"{s.source}/{stem}: adopted from {adopt_from}")
                    with np.load(job.out_dir / f"{stem}.npz") as z:
                        n_adopted = len(z["ids"])  # rows labelled, as a computed shard reports, not rows scanned
                    append_progress(progress, dict(stem=stem, source=s.source, split=s.split, hours=float(s.hours),
                                                   wall_s=0.0, adopted=True, rows=n_adopted))
                    touch(heartbeat)
                    continue
            if s.source in args.adopt_only:
                integrity_exit(f"{s.source}/{stem}: --adopt-only, but {adopt_from} has no matching finished shard "
                               f"(ids, settings and prompt); it is never computed")
            jobs.append(job)
        return jobs

    first = todo()
    n_sel = len(select())
    if not n_sel and follow is None:
        sys.exit("no shards selected - run scripts/01_prepare_data.py first")
    print(f"output: {out_root}\n{len(first)}/{n_sel} shards to do, {sum(j.info.hours for j in first):.1f} h audio "
          f"({n_sel - len(first)} already done)")
    if not first and follow is None:
        return

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if args.vram_fraction:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free before load: {free / 2**30:.2f} of {total / 2**30:.1f} GiB")
    if total < 12 * 2**30 and free < 6 * 2**30:  # with several lanes on a big card "free" says nothing
        print("WARNING: <6 GiB free - other apps are holding VRAM; the pass may spill to shared memory and slow down 5x")
    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = CohereAsrForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                              dtype=torch.bfloat16).to("cuda").eval()
    fp32_head(model)
    print(f"teacher loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    totals = dict(rows=0, skipped_long=0, bad_decode=0, audio_s=0.0, wall_s=0.0, truncated=0)
    remaining = [sum(j.info.hours for j in first)]

    def finish(job, t, per, stats):
        r = finish_shard(processor, job, t, per, stats, args)
        done.add((job.info.source, job.stem))  # written by this process: todo() never re-checks it (follow polls)
        r["wall_s"] = stats["wall_s"]
        for key in totals:
            totals[key] += r[key]
        remaining[0] = max(0.0, remaining[0] - job.info.hours)
        rtf = totals["wall_s"] / max(totals["audio_s"], 1e-6)
        cer = f"{r['mean_cer']:.3f}" if r["mean_cer"] is not None else "n/a"
        tqdm.write(f"{job.info.source}/{job.stem}: {r['rows']} utts, {r['audio_s'] / 3600:.2f} h in {r['wall_s'] / 60:.1f} min "
                   f"(RTF {r['wall_s'] / max(r['audio_s'], 1e-6):.4f}), CER vs ref {cer}, truncated {r['truncated']}, "
                   f">30s {r['skipped_long']}, bad audio {r['bad_decode']}, OOM splits {stats['oom_splits']} | "
                   f"running RTF {rtf:.4f}, ETA {remaining[0] * rtf:.2f} h | VRAM alloc "
                   f"{torch.cuda.max_memory_allocated() / 2**30:.2f} reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} "
                   f"free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
        if (args.force or not meta_path.exists()) and r["prompt"]:
            write_json_atomic(meta_path, dict(
                **settings,
                decoder_prompt_ids=r["prompt"],
                decoder_prompt_tokens=processor.tokenizer.convert_ids_to_tokens(r["prompt"]),
                eos_token_id=model.generation_config.eos_token_id, pad_token_id=model.generation_config.pad_token_id,
                vocab_size=model.config.vocab_size, encoder_hidden_size=model.config.encoder_config.hidden_size,
                decoding="greedy", lm_head_dtype="float32", model_dtype="bfloat16",
            ))
        return dict(rows=r["rows"])

    pending = [first]
    run(lambda: pending.pop() if pending else todo(),
        load_table=lambda job: load_table(root, job, args.limit_rows),
        plan_batches=lambda job, t: plan_batches(t, args),
        prepare=lambda t, idx: prepare(processor, t, idx),
        gpu=lambda inputs, ok, t: gpu(model, inputs, ok, t, args),
        finish=finish, workers=args.workers, prefetch=args.prefetch, follow_marker=follow, heartbeat=heartbeat,
        progress=progress, poll_s=args.poll_s, on_oom=torch.cuda.empty_cache, log=tqdm.write)

    print(f"\ndone: {totals['rows']} utterances, {totals['audio_s'] / 3600:.2f} h audio in {totals['wall_s'] / 3600:.2f} h wall "
          f"(RTF {totals['wall_s'] / max(totals['audio_s'], 1e-6):.4f}); truncated={totals['truncated']}, "
          f"skipped >30s={totals['skipped_long']}, bad audio={totals['bad_decode']}")


def check_meta_readonly(meta_path: Path, settings: dict):
    """ValueError when another teacher_out's meta.json was written with other settings (never writes it)."""
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    old.setdefault("model_revision", MODEL_REVISION)
    diff = {key: (old.get(key), val) for key, val in settings.items() if old.get(key) != val}
    if diff:
        raise ValueError(f"{meta_path} has different settings {diff}")


if __name__ == "__main__":
    main()
