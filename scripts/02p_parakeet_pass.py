"""Run the Parakeet TDT-CTC teacher over every shard in data/ and save its soft targets beside the Cohere ones.

For each input shard  data/shards/<source>/<split>-NNNNN.parquet  this writes
  parakeet_out/<source>/<split>-NNNNN.npz     TDT greedy-path + CTC soft targets (np.savez_compressed)
  parakeet_out/<source>/<split>-NNNNN.jsonl   {id, hyp, ctc_hyp, ref, cer, ctc_cer, duration, n_tok, n_steps, n_frames,
                                              n_forced, truncated}
  parakeet_out/meta.json                      model pins, decoding, dtypes and settings (written atomically before
                                              the first shard; compared on resume, a mismatch exits 65)
The npz FORMAT is the docstring of kitsune/parakeet_targets.py (one definition).

The model is the converted HF dir the laptop published (tools/publish_parakeet.py), checked against the sha256 pins in
kitsune/parakeet.py before anything loads (a problem exits 3); it is always read with local_files_only=True. Decoding
is the guarded greedy TDT loop of kitsune.parakeet (HF semantics + NeMo's max-symbols guard); the encoder runs in bf16,
the rest in fp32.

Utterances longer than 30 s are skipped and counted, exactly as in 02, so both passes keep the same rows. A shard is done
when its npz covers the data shard's ids (n_scanned, ids sha256, ordered subsequence) with the same settings and its
jsonl exists; --strict-existing turns an existing npz that fails that check into exit 65 instead of a recompute.
A --limit-rows run writes to parakeet_out_smoke/ so it can never be mistaken for a finished shard.

Usage:
  python scripts/02p_parakeet_pass.py --model-dir models/parakeet-tdt_ctc-0.6b-ja-hf
  python scripts/02p_parakeet_pass.py --model-dir DIR --limit-shards 1 --limit-rows 64     # smoke -> parakeet_out_smoke/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kitsune import labelpass  # noqa: E402
from kitsune import parakeet_targets as pt  # noqa: E402
from kitsune.audio import decode_audio  # noqa: E402
from kitsune.parakeet import ParakeetTeacher, verify_model_dir  # noqa: E402
from kitsune.store import ids_sha256, manifest_path, read_manifest, read_shard, shard_ids  # noqa: E402
from kitsune.text import cer as cer_fn  # noqa: E402

MAX_DUR = 30.0  # the same cut as 02, so the two passes keep the same rows
EXIT_INTEGRITY = 65
EXIT_MODEL = 3


def make_batches(durations: np.ndarray, max_batch_seconds: float, max_batch: int) -> list[np.ndarray]:
    """Longest first (the biggest batch runs first, so OOM shows up immediately); cost is padded length x batch."""
    order = np.argsort(-durations, kind="stable")
    batches, cur, cur_max = [], [], 0.0
    for i in order:
        d = float(durations[i])
        new_max = max(cur_max, d)
        if cur and (new_max * (len(cur) + 1) > max_batch_seconds or len(cur) >= max_batch):
            batches.append(np.array(cur))
            cur, new_max = [], d
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(np.array(cur))
    return batches


def fail(code: int, msg: str):
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


def ensure_meta(out_root: Path, meta: dict):
    """Write meta.json atomically if missing; otherwise it must equal what this run would write."""
    path = out_root / "meta.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        diff = {key: (old.get(key), val) for key, val in meta.items() if old.get(key) != val}
        if diff:
            fail(EXIT_INTEGRITY, f"{path} was written with different settings {diff}; use a different --out")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    labelpass.write_json_atomic(path, meta)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="output root (default: parakeet_out, or parakeet_out_smoke with --limit-rows)")
    ap.add_argument("--model-dir", required=True, help="the converted HF dir (checked against kitsune.parakeet pins)")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--device", default="cuda", help="cpu is for tests and the golden generation")
    ap.add_argument("--k-tdt", type=int, default=8)
    ap.add_argument("--k-ctc", type=int, default=8)
    ap.add_argument("--ctc-dense-thr", type=float, default=0.95)
    ap.add_argument("--max-symbols", type=int, default=10)
    ap.add_argument("--max-batch-seconds", type=float, default=2400.0)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=None)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--follow", default=None, metavar="MARKER", help="keep polling the manifest until MARKER exists")
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--shard-mod", type=int, default=1)
    ap.add_argument("--shard-rem", type=int, default=0)
    ap.add_argument("--vram-fraction", type=float, default=None)
    ap.add_argument("--strict-existing", action="store_true",
                    help="an existing npz that fails the done check exits 65 instead of being recomputed")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--limit-rows", type=int, default=None, help="debug: only the first N rows of each shard")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.data)
    out_root = Path(args.out) if args.out else ROOT / ("parakeet_out_smoke" if args.limit_rows else "parakeet_out")
    model_dir = Path(args.model_dir)
    problems = verify_model_dir(model_dir)
    if problems:
        fail(EXIT_MODEL, f"{model_dir} is not the pinned Parakeet model:\n  " + "\n  ".join(problems))

    settings = dict(k_tdt=args.k_tdt, k_ctc=args.k_ctc, max_symbols=args.max_symbols, ctc_dense_thr=args.ctc_dense_thr)
    ensure_meta(out_root, pt.build_meta(settings))

    submitted: set[str] = set()

    def select():
        shards = read_manifest(root) if manifest_path(root).exists() else []
        if args.split != "all":
            shards = [s for s in shards if s.split == args.split]
        if args.sources:
            shards = [s for s in shards if s.source in args.sources]
        shards = [s for s in shards if labelpass.partition_ok(s.path, args.shard_mod, args.shard_rem)]
        return shards[: args.limit_shards] if args.limit_shards else shards

    def expected_ids(info) -> list[str]:
        ids = shard_ids(root, info)
        return ids[: args.limit_rows] if args.limit_rows else ids

    def todo():
        jobs = []
        for s in select():
            stem = Path(s.path).stem
            key = f"{s.source}/{stem}"
            if key in submitted:
                continue
            out_dir = out_root / s.source
            npz = out_dir / f"{stem}.npz"
            if npz.exists() and not pt.shard_done(npz, expected_ids(s), settings):
                if args.strict_existing:
                    fail(EXIT_INTEGRITY, f"{npz} exists but does not cover {key} with these settings (--strict-existing)")
                print(f"{key}: existing output does not match, recomputing", flush=True)
            elif npz.exists():
                submitted.add(key)
                continue
            submitted.add(key)
            jobs.append(labelpass.ShardJob(info=s, stem=stem, out_dir=out_dir))
        return jobs

    first = todo()
    print(f"output: {out_root}\n{len(first)} shards to do, {sum(j.info.hours for j in first):.1f} h audio", flush=True)
    if not first and not args.follow:
        return
    pending = list(first)

    def todo_once():
        if pending:
            jobs = list(pending)
            pending.clear()
            return jobs
        return todo()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        if args.vram_fraction:
            torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    teacher = ParakeetTeacher(model_dir, device=args.device)

    def load_table(job):
        t0 = time.time()
        table = read_shard(root / job.info.path, columns=["id", "audio", "text", "duration"])
        ids = table.column("id").to_pylist()
        n = min(len(ids), args.limit_rows) if args.limit_rows else len(ids)
        durs = np.array(table.column("duration").to_pylist(), dtype=np.float32)
        keep = np.array([i for i in range(n) if durs[i] <= MAX_DUR], dtype=np.int64)
        return SimpleNamespace(ids=ids, texts=table.column("text").to_pylist(), durs=durs, audio=table.column("audio"),
                               n=n, keep=keep, skipped_long=n - len(keep), t0=t0,
                               sha=ids_sha256(expected_ids(job.info)))

    def plan_batches(job, table):
        return [table.keep[b] for b in make_batches(table.durs[table.keep], args.max_batch_seconds, args.max_batch)]

    def prepare(table, idx):
        waves, ok = [], []
        for i in idx:
            try:  # ingest only checked the container header; a truncated payload still fails full decode
                waves.append(decode_audio(table.audio[int(i)].as_py()))
                ok.append(int(i))
            except Exception as e:
                print(f"  bad audio, skipping {table.ids[int(i)]}: {e}", flush=True)
        ok = np.array(ok, dtype=np.int64)
        return (teacher.features(waves) if waves else None), ok

    def gpu(inputs, ok_idx, table):
        if inputs is None or not len(ok_idx):
            return {}
        rows = teacher.run_batch(inputs["input_features"], inputs["attention_mask"], k_tdt=args.k_tdt, k_ctc=args.k_ctc,
                                 ctc_dense_thr=args.ctc_dense_thr, max_symbols=args.max_symbols)
        return {int(i): r for i, r in zip(ok_idx, rows)}

    def finish(job, table, per_row, stats):
        order = sorted(per_row)
        utts = [dict(per_row[i], id=table.ids[i], duration=float(table.durs[i])) for i in order]
        arrays = pt.pack_shard(utts, settings=settings, n_scanned=table.n, shard_ids_sha=table.sha)
        rows, cers, ctc_cers = [], [], []
        for i, u in zip(order, utts):
            ref, h, ch, t = table.texts[i], u["hyp"], u["ctc_hyp"], pt.utt_tokens(u)
            c, cc = cer_fn(h, ref), cer_fn(ch, ref)
            cers.append(c)
            ctc_cers.append(cc)
            rows.append(dict(id=u["id"], hyp=h, ctc_hyp=ch, ref=ref, cer=round(float(c), 4), ctc_cer=round(float(cc), 4),
                             duration=round(float(u["duration"]), 3), n_tok=int(len(t)), n_steps=int(len(u["step_frame"])),
                             n_frames=int(u["n_frames"]), n_forced=int(np.sum(u["step_forced"])),
                             truncated=bool(u["truncated"])))
        pt.write_shard(job.out_dir, job.stem, arrays, rows)
        audio_s = float(table.durs[order].sum()) if order else 0.0
        wall = float(stats.get("wall_s", time.time() - table.t0))
        vram = (f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if args.device.startswith("cuda")
                and torch.cuda.is_available() else "n/a")
        mean = (lambda xs: float(np.mean(xs)) if xs else float("nan"))
        print(f"{job.info.source}/{job.stem}: {len(order)} utts, {audio_s / 3600:.2f} h in {wall / 60:.1f} min "
              f"(RTF {wall / max(audio_s, 1e-6):.4f}), CER TDT {mean(cers):.3f} CTC {mean(ctc_cers):.3f}, "
              f"forced steps {int(arrays['step_forced'].sum())}, truncated {int(arrays['truncated'].sum())}, "
              f">30s {table.skipped_long}, bad audio {stats.get('bad_decode', 0)} | VRAM {vram}", flush=True)
        return dict(audio_s=round(audio_s, 3), skipped_long=table.skipped_long, bad_decode=stats.get("bad_decode", 0),
                    forced=int(arrays["step_forced"].sum()))

    def on_oom():
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    totals = labelpass.run(
        todo_once, load_table=load_table, plan_batches=plan_batches, prepare=prepare, gpu=gpu, finish=finish,
        workers=args.workers, prefetch=args.prefetch, follow_marker=Path(args.follow) if args.follow else None,
        heartbeat=Path(args.heartbeat) if args.heartbeat else None,
        progress=Path(args.progress) if args.progress else None, on_oom=on_oom)
    print(f"\ndone: {totals}", flush=True)


if __name__ == "__main__":
    main()
