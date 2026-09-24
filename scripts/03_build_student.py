"""Build the Kitsune student from the teacher: layer selection, FFN pruning by activation importance, BN recalibration.

The shape and the reasons for it are in kitsune/student.py. This script runs the pipeline end to end:
  1. seeded calibration sample of the selection's kept TRAIN rows (all --sources, equal share each), audio read by id
  2. load the teacher in bf16 on --device
  3. FFN importance over --calib-utts utterances, log-mels computed on the device (kitsune.features.LogMel)
  4. build the student on CPU (fresh model, strict remapped state_dict)
  5. free the teacher
  6. student -> device in fp32, recalibrate BatchNorm on --bn-utts utterances (batch size 1)
  7. save: bf16 weights with fp32 BN stats, processor, generation config, student_meta.json (stage "saved")
  8. step-0 eval on --step0-eval-utts utterances per eval set, teacher-forced + greedy (kitsune.evaluate)
  9. student_meta.json rewritten with stage "complete"

It is resumable, because the laptop GPU throws intermittent CUDA illegal-instruction faults:
  <out>/importance.pt   written after step 3; reused if it covers the same teacher, layers and calibration ids
  student_meta.json     stage "saved": a re-run skips 1-7 and only redoes the step-0 eval; stage "complete": nothing
                        to do (--force rebuilds; a matching importance.pt is still reused - delete it to recompute)
The step-0 eval runs after the save (the spec lists it before), so a fault there cannot lose the built student. If
the eval sets are not on this machine, finish with --step0-eval-utts 0.

The calibration sample is clustered to keep I/O small: random (shard, row group) pairs, then up to --calib-per-group
kept rows from each. A row group of reazon_small is ~256 utterances (~25 MB), so 1000 utterances read ~1.6 GB instead
of the whole 7 GB source.

Usage:
  python scripts/03_build_student.py                                  # B20x2560 / dec 0 2 5 7 -> students/b20x2560-d4
  python scripts/03_build_student.py --enc-layers 4 --dec-layers 0 7  # laptop smoke student   -> students/b4x2560-d2
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# torch / transformers / the kitsune training modules are imported inside the functions, so --help is instant and
# does not depend on the rest of the stack.
TEACHER_ID = "CohereLabs/cohere-transcribe-03-2026"
EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon"]
MAX_SECONDS = 30.0  # LogMel / HF fast path; the teacher pass skipped longer utterances anyway
IMPORTANCE_FORMAT = 1


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_info() -> dict:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()

    try:
        return dict(sha=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                    dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except Exception as e:  # no git on the box is not a reason to fail a build
        return dict(error=repr(e))


def host_info(device) -> dict:
    import torch
    import transformers

    info = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__, device=str(device))
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return info


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ calibration audio


def sample_calibration(selection: Path, data_root: Path, sources: list[str], n: int, seed: int, per_group: int,
                       log=print) -> list[dict]:
    """Up to `n` kept train utterances as {id, source, duration, wave (16 kHz float32)}, split evenly over the sources
    and joined to the audio BY ID (shard numbering may differ between machines), in a seeded shuffled order."""
    import numpy as np
    import pyarrow.parquet as pq

    from kitsune.audio import TARGET_SR, decode_audio
    from kitsune.trainset import read_selection

    sel = read_selection(selection, sources, ["train"])
    sel = sel[sel["duration"] <= MAX_SECONDS]
    rng = np.random.default_rng(seed)
    active = [s for s in sources if (sel["source"] == s).any()]
    if not active:
        raise ValueError(f"{selection}: no kept train rows for sources {sources}")
    quotas = {s: n // len(active) + (1 if i < n % len(active) else 0) for i, s in enumerate(active)}
    out = []
    for source in active:
        kept = set(sel.loc[sel["source"] == source, "id"])
        files = {p: pq.ParquetFile(p) for p in sorted((data_root / "shards" / source).glob("train-*.parquet"))}
        groups = [(p, g) for p, pf in files.items() for g in range(pf.num_row_groups)]
        got, bad, t0 = [], 0, time.time()
        for gi in rng.permutation(len(groups)):
            if len(got) >= quotas[source]:
                break
            p, g = groups[gi]
            ids = files[p].read_row_group(g, columns=["id"]).column("id").to_pylist()
            hits = [j for j, x in enumerate(ids) if x in kept]
            if not hits:
                continue
            take = rng.choice(hits, size=min(per_group, len(hits), quotas[source] - len(got)), replace=False)
            audio = files[p].read_row_group(g, columns=["audio"]).column("audio")
            for j in sorted(int(x) for x in take):
                kept.discard(ids[j])  # first copy of an id wins
                try:
                    wave = decode_audio(audio[j].as_py())
                except Exception as e:
                    bad += 1
                    log(f"  calibration: bad audio, skipping {ids[j]}: {e}")
                    continue
                if 0 < len(wave) <= MAX_SECONDS * TARGET_SR:
                    got.append(dict(id=ids[j], source=source, duration=len(wave) / TARGET_SR, wave=wave))
        log(f"  calibration sample {source}: {len(got)}/{quotas[source]} utts, "
            f"{sum(u['duration'] for u in got) / 3600:.2f} h, {bad} bad audio, {time.time() - t0:.1f} s")
        if len(got) < quotas[source]:
            log(f"  WARNING: only {len(got)} kept rows with audio found for {source} (wanted {quotas[source]})")
        out += got
    order = rng.permutation(len(out))  # interleave sources so every prefix is a mixed sample
    return [out[i] for i in order]


def pack(durations: list[float], batch_s: float) -> list[list[int]]:
    """Longest first (an OOM shows up on the first batch), padded seconds max_dur * n <= batch_s."""
    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    batches, cur = [], []
    for i in order:
        if cur and durations[cur[0]] * (len(cur) + 1) > batch_s:
            batches.append(cur)
            cur = []
        cur.append(i)
    return batches + ([cur] if cur else [])


def feature_batches(utts: list[dict], featurizer, device, batch_s: float, desc: str):
    """Yield (feats (B,T,128) fp32, mask (B,T) bool) on the device, featurised batch by batch."""
    import torch
    from tqdm import tqdm

    from kitsune.features import pad_waves

    batches = pack([u["duration"] for u in utts], batch_s)
    for b in tqdm(batches, desc=desc, unit="batch", leave=False):
        wave, lengths = pad_waves([utts[i]["wave"] for i in b])
        with torch.no_grad():
            yield featurizer(wave.to(device), lengths.to(device))


# ------------------------------------------------------------------------------------------------ stages


def load_or_compute_importance(teacher, featurizer, utts: list[dict], args, spec, device, out: Path, log=print):
    """Stage 3, cached in <out>/importance.pt (atomic). Returns (importance dict, info dict)."""
    import torch

    from kitsune import student as S

    path = out / "importance.pt"
    key = dict(teacher=args.teacher, ids_sha256=ids_sha([u["id"] for u in utts]), n_utts=len(utts))
    if path.exists():
        try:
            c = torch.load(path, map_location="cpu", weights_only=True)
            ok = (c.get("format") == IMPORTANCE_FORMAT and c.get("key") == key
                  and set(spec.enc_layers) <= set(c.get("layers", [])))
        except Exception as e:  # a torn file from a killed run
            ok, c = False, None
            log(f"  importance cache unreadable ({e}); recomputing")
        if ok:
            log(f"  importance: reusing {path} ({c['info']['n_utts']} utts, computed {c['info']['created']})")
            return S.importance_from_state(c["importance"]), dict(c["info"], reused=True)
        if c is not None:
            log(f"  importance cache {path} does not match this run (teacher/sample/layers); recomputing")

    t0 = time.time()
    imp = S.ffn_importance(teacher, feature_batches(utts, featurizer, device, args.calib_batch_s, "importance"),
                           spec.enc_layers, device)
    info = dict(n_utts=len(utts), audio_s=float(sum(u["duration"] for u in utts)), wall_s=round(time.time() - t0, 1),
                created=now(), sources=sorted({u["source"] for u in utts}), ids_sha256=key["ids_sha256"])
    tmp = path.with_suffix(".pt.tmp")
    torch.save(dict(format=IMPORTANCE_FORMAT, key=key, layers=list(spec.enc_layers), info=info,
                    importance=S.importance_to_state(imp)), tmp)
    tmp.replace(path)
    log(f"  importance: {len(utts)} utts ({info['audio_s'] / 3600:.2f} h) in {info['wall_s']:.0f} s -> {path}")
    return imp, dict(info, reused=False)


def step0_ids(store, eval_sets: list[str], n: int, seed: int) -> dict[str, list[str]]:
    """n ids per eval set, drawn (seeded) from the set's greedy subset, or from the whole set if that is smaller."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = {}
    for s in eval_sets:
        cand = store.indices(source=s, in_greedy_subset=True)
        if len(cand) < n:
            cand = store.indices(source=s)
        pick = sorted(rng.choice(cand, size=n, replace=False).tolist()) if len(cand) > n else cand
        out[s] = [store.utts[i].id for i in pick]
    return out


def step0_eval(student, processor, featurizer, args, device, out: Path, log=print) -> dict:
    """Stage 8: the untrained student on N utterances per eval set, teacher-forced and greedy (the trainer's own
    eval code, so the numbers line up with its step-0 eval). Per-utterance tables go to <out>/step0/."""
    from kitsune import evaluate, trainset

    t0 = time.time()
    store = trainset.eval_store(args.selection, args.data, args.teacher_root, args.eval_cache, args.eval_sets, log=log)
    per_set = step0_ids(store, args.eval_sets, args.step0_eval_utts, args.seed)
    ids = [x for s in args.eval_sets for x in per_set[s]]
    tf_sum, tf_df = evaluate.teacher_forced_eval(student, store, featurizer, device, args.eval_batch_s, ids=ids)
    gr_sum, gr_df = evaluate.greedy_eval(student, store, ids, featurizer, device, args.eval_batch_s,
                                         tokenizer=processor.tokenizer)
    d = out / "step0"
    d.mkdir(parents=True, exist_ok=True)
    tf_df.to_parquet(d / "teacher_forced.parquet", index=False)
    gr_df.to_parquet(d / "greedy.parquet", index=False)
    for s, g in sorted(gr_sum.get("sets", {}).items()):
        t = tf_sum.get("sets", {}).get(s, {})
        log(f"  step-0 {s}: KL {t.get('kl', float('nan')):.3f} top1 {t.get('top1', float('nan')):.3f} | CER "
            f"{g['cer_ref_corpus']:.4f} (teacher {g['teacher_cer_ref_corpus']:.4f}, x{g['ratio_vs_teacher']:.2f}), "
            f"vs teacher {g['cer_teacher_corpus']:.4f}, truncated {g['trunc_rate']:.3f}")
    return dict(n_per_set={s: len(v) for s, v in per_set.items()}, ids_sha256=ids_sha(ids),
                teacher_forced=tf_sum, greedy=gr_sum, samples=evaluate.pick_samples(gr_df, 8, seed=args.seed),
                wall_s=round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc-layers", type=int, default=20, help="number of teacher encoder layers kept, evenly spaced")
    ap.add_argument("--ffn", type=int, default=2560, help="encoder FFN width (5120 = no pruning)")
    ap.add_argument("--dec-layers", type=int, nargs="+", default=[0, 2, 5, 7], help="teacher decoder layers kept")
    ap.add_argument("--no-tie-head", action="store_true", help="keep proj_out.weight as its own parameter")
    ap.add_argument("--calib-utts", type=int, default=1000, help="utterances for FFN importance")
    ap.add_argument("--bn-utts", type=int, default=1000, help="utterances for the BatchNorm recalibration")
    ap.add_argument("--sources", nargs="+", default=["reazon_small"], help="train sources to calibrate on")
    ap.add_argument("--out", default=None, help="student dir (default: students/b<enc>x<ffn>-d<n_dec>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--step0-eval-utts", type=int, default=200, help="per eval set; 0 skips the step-0 eval")
    ap.add_argument("--eval-sets", nargs="+", default=EVAL_SETS)
    ap.add_argument("--teacher", default=TEACHER_ID, help="HF repo id or local dir of the teacher")
    ap.add_argument("--selection", default=str(ROOT / "selection" / "viability.parquet"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--teacher-root", default=str(ROOT / "teacher_out"))
    ap.add_argument("--eval-cache", default=str(ROOT / "cache" / "eval"), help="kitsune.trainset eval store cache")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--calib-per-group", type=int, default=16, help="max utterances taken from one parquet row group")
    ap.add_argument("--calib-batch-s", type=float, default=240.0, help="padded audio seconds per importance batch")
    ap.add_argument("--eval-batch-s", type=float, default=240.0, help="padded audio seconds per step-0 eval batch")
    ap.add_argument("--force", action="store_true", help="rebuild even if student_meta.json says complete/saved")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(ROOT / "students" / f"b{args.enc_layers}x{args.ffn}-d{len(args.dec_layers)}")
    for k in ("selection", "data", "teacher_root", "eval_cache", "out"):
        setattr(args, k, Path(getattr(args, k)))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformers import AutoConfig, AutoProcessor, CohereAsrForConditionalGeneration

    from kitsune import student as S
    from kitsune.features import LogMel

    out = args.out
    n_teacher_layers = AutoConfig.from_pretrained(args.teacher).encoder_config.num_hidden_layers  # 48 for the real one
    spec = S.StudentSpec(enc_layers=S.evenly_spaced(args.enc_layers, n_teacher_layers), ffn_dim=args.ffn,
                         dec_layers=sorted(args.dec_layers), tie_head=not args.no_tie_head)
    old = S.load_meta(out)
    same_spec = old.get("spec") == json.loads(S.dump_json(spec))
    if old.get("stage") == "complete" and not args.force:
        print(f"{out}: already built ({old.get('timestamps', {}).get('finished')}); --force to rebuild")
        return 0
    if old.get("stage") and not same_spec and not args.force:
        sys.exit(f"{out} holds a different student ({old.get('spec')}); use another --out or --force")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU free: {free / 2**30:.2f} of {total / 2**30:.1f} GiB")
    print(f"student: enc {spec.enc_layers} ffn {spec.ffn_dim} dec {spec.dec_layers} tied={spec.tie_head} -> {out}")

    processor = AutoProcessor.from_pretrained(args.teacher)
    featurizer = LogMel.from_feature_extractor(processor.feature_extractor).to(device)
    t_start = time.time()

    if old.get("stage") == "saved" and not args.force:
        print(f"resuming: {out} is saved; only the step-0 eval is left")
        meta = old
        meta.setdefault("timestamps", {})["resumed"] = now()
        student = S.load_student(out, device)
    else:
        meta = dict(format=1, stage="building", teacher=args.teacher, spec=spec,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    git=git_info(), host=host_info(device), timestamps=dict(started=now()), durations_s={})
        dur = meta["durations_s"]

        t = time.time()
        n_sample = max(args.calib_utts, args.bn_utts)
        print(f"[1/8] calibration sample: {n_sample} utts from {args.sources}")
        utts = sample_calibration(args.selection, args.data, args.sources, n_sample, args.seed, args.calib_per_group)
        imp_utts, bn_utts = utts[:args.calib_utts], utts[:args.bn_utts]
        dur["sample"] = round(time.time() - t, 1)

        t = time.time()
        print(f"[2/8] teacher {args.teacher} (bf16) -> {device}")
        teacher = CohereAsrForConditionalGeneration.from_pretrained(args.teacher, dtype=torch.bfloat16,
                                                                    attn_implementation="sdpa").to(device).eval()
        # provenance: the teacher is read from `main`, so record which commit that was (None for a local directory)
        meta["teacher_revision"] = getattr(teacher.config, "_commit_hash", None)
        dur["teacher_load"] = round(time.time() - t, 1)

        t = time.time()
        print(f"[3/8] FFN importance over {len(imp_utts)} utts")
        teacher_ffn = teacher.config.encoder_config.intermediate_size
        if spec.ffn_dim < teacher_ffn:
            importance, imp_info = load_or_compute_importance(teacher, featurizer, imp_utts, args, spec, device, out)
        else:
            importance, imp_info = None, dict(skipped="ffn_dim == teacher width: nothing to prune")
        dur["importance"] = round(time.time() - t, 1)

        t = time.time()
        print("[4/8] build the student on CPU")
        student = S.build_student(teacher, spec, importance)
        keep = S.select_ffn_neurons(importance, spec.enc_layers, spec.ffn_dim, teacher_ffn)
        report = S.param_report(student)
        print(f"  {report['total']:,} params (closed form {report['closed_form']:,}), encoder share "
              f"{report['encoder_share']:.1%}")
        if spec == S.default_spec() and n_teacher_layers == S.TEACHER_ENC_LAYERS:  # the viability student (D12)
            assert abs(report["total"] - S.EXPECTED_DEFAULT_PARAMS) <= 1e-3 * S.EXPECTED_DEFAULT_PARAMS, report["total"]
        dur["build"] = round(time.time() - t, 1)

        print("[5/8] free the teacher")
        del teacher
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        t = time.time()
        print(f"[6/8] BatchNorm recalibration on {len(bn_utts)} utts (fp32, batch size 1)")
        student.to(device=device, dtype=torch.float32)
        bn = S.recalibrate_batchnorm(student, feature_batches(bn_utts, featurizer, device, args.calib_batch_s, "bn"),
                                     device, max_utts=args.bn_utts)
        print(f"  BN drift vs teacher stats: mean shift {bn['mean_shift_std']:.3f} std, "
              f"|log var ratio| {bn['abs_log_var_ratio']:.3f} ({bn['n_utts']} utts)")
        dur["bn"] = round(time.time() - t, 1)

        t = time.time()
        print(f"[7/8] save -> {out}")
        meta.update(
            stage="saved",
            kept=dict(enc_layers=spec.enc_layers, dec_layers=spec.dec_layers,
                      ffn={f"{l}.{n}": idx for (l, n), idx in sorted(keep.items())}),
            importance=dict(imp_info, **(S.importance_summary(importance, keep) if keep else {})),
            bn=bn, params=report,
            calibration=dict(sources=args.sources, seed=args.seed, per_group=args.calib_per_group,
                             n_importance=len(imp_utts), n_bn=bn["n_utts"],
                             importance_ids_sha256=ids_sha([u["id"] for u in imp_utts]),
                             hours_importance=sum(u["duration"] for u in imp_utts) / 3600,
                             per_source={s: sum(u["source"] == s for u in imp_utts) for s in args.sources}),
        )
        meta["timestamps"]["saved"] = now()
        del utts, imp_utts, bn_utts
        S.save_student(student, out, processor, meta)
        dur["save"] = round(time.time() - t, 1)

    if args.step0_eval_utts > 0:
        print(f"[8/8] step-0 eval: {args.step0_eval_utts} utts per set of {args.eval_sets}")
        try:
            meta["step0"] = step0_eval(student, processor, featurizer, args, device, out)
        except Exception as e:
            meta["step0"] = dict(error=repr(e), traceback=traceback.format_exc())
            S.write_meta(out, meta)  # still stage "saved": a re-run retries only the eval
            print(f"step-0 eval failed ({e!r}); the student is saved. Re-run to retry the eval, or finish with "
                  f"--step0-eval-utts 0")
            raise
    else:
        meta["step0"] = dict(skipped=True)
    meta["stage"] = "complete"
    meta["timestamps"]["finished"] = now()
    meta.setdefault("durations_s", {})["total_this_run"] = round(time.time() - t_start, 1)
    S.write_meta(out, meta)
    print(f"done: {out} ({meta['params']['total']:,} params) in {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
