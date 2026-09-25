"""Build the Kitsune student from the teacher: layer selection, FFN pruning by activation importance, BN recalibration.
Or (--scratch) a randomly initialised student of the teacher's architecture at another width.

The shape and the reasons for it are in kitsune/student.py. This script runs the pipeline end to end:
  1. seeded calibration sample of the selection's kept TRAIN rows (all --sources, equal share each), audio read by id
     (skipped when nothing needs audio: --ffn-from with --bn keep)
  2. load the teacher in --teacher-dtype (bf16) on --device, at --teacher-revision (the commit 02_teacher_pass read)
  3. FFN importance over --calib-utts utterances, log-mels computed on the device (kitsune.features.LogMel), in the
     kept layers (--importance-layers spec) or in all 48 (all); or the FFN selection of another student (--ffn-from)
  4. build the student on CPU (fresh model, strict remapped state_dict); assert its count
  5. free the teacher
  6. student -> device in fp32; --bn recal: recalibrate BatchNorm on --bn-utts utterances (batch size 1);
     --bn keep: keep the teacher's running stats (the size study, decision 16)
  7. save: bf16 weights with fp32 BN stats, processor, generation config, student_meta.json (stage "saved")
  8. step-0 eval on --step0-eval-utts utterances per eval set, teacher-forced + greedy (kitsune.evaluate)
  9. student_meta.json rewritten with stage "complete"

--scratch <shape> (kitsune.student.SCRATCH_SHAPES: t01, t005, bridge) replaces 1-6: the teacher's config and processor
are read (not its weights) and kitsune.student.build_scratch_student builds a seeded random init (--seed; fresh BN).

It is resumable, because the laptop GPU throws intermittent CUDA illegal-instruction faults:
  importance cache      <out>/importance.pt, or --importance-cache: written after step 3; reused if it covers the same
                        teacher commit, calibration ids and the layers asked for. One --importance-layers all pass
                        serves every size: point the other builds at it with --importance-cache (it is only ever
                        written when it does not match)
  student_meta.json     stage "saved": a re-run skips 1-7 and only redoes the step-0 eval; stage "complete": nothing
                        to do (--force rebuilds; a matching importance cache is still reused - delete it to recompute)
The step-0 eval runs after the save (the spec lists it before), so a fault there cannot lose the built student. If
the eval sets are not on this machine, finish with --step0-eval-utts 0.

The calibration sample is clustered to keep I/O small: random (shard, row group) pairs, then up to --calib-per-group
kept rows from each. A row group of reazon_small is ~256 utterances (~25 MB), so 1000 utterances read ~1.6 GB instead
of the whole 7 GB source.

student_meta.json (format 2) carries the size study's fields next to the build record: family "aed", init_class
("pruned_kept" | "scratch"), params_total, params_non_embedding, closed_form_params (asserted equal to params_total
before the save), seed, bn ("teacher" | "recal" | "fresh"; the recalibration's drift stats are under bn_recal),
teacher "<repo>@<commit>", and enc_layers / ffn / dec_layers for a pruned student. A step-0 eval of a pruned student
adds function_check: STUDY.md 2.2 calls a pruned student "function lost" when its step-0 CER against the teacher is
at least 90 % on the 60-utterance CPU gate (--device cpu --step0-eval-utts 20: 20 per gate set).

Usage:
  python scripts/03_build_student.py                                  # B20x2560 / dec 0 2 5 7 -> students/b20x2560-d4
                                                                      # (calibrated on reazon_small emilia_yodas galgame)
  python scripts/03_build_student.py --enc-layers 4 --dec-layers 0 7  # laptop smoke student   -> students/b4x2560-d2
The size study's Transcribe students (on the laptop CPU; the first run's data dirs, 1,000 calibration ids):
  python scripts/03_build_student.py --device cpu --teacher-dtype fp32 --step0-eval-utts 20 --bn keep \\
      --ffn-from students/b20x2560-d4 --out students/study/t06                      # T-0.6B: the first run's FFNs
  python scripts/03_build_student.py --device cpu --teacher-dtype fp32 --step0-eval-utts 20 --bn keep \\
      --enc-layers 10 --dec-layers 0 7 --importance-layers all --out students/study/t03                 # T-0.3B
  python scripts/03_build_student.py --scratch bridge --device cpu --step0-eval-utts 0 --out students/study/bridge
  python scripts/03_build_student.py --scratch t01 --device cpu --step0-eval-utts 0 --out students/study/t01
  python scripts/03_build_student.py --scratch t01 --seed 1235 --device cpu --step0-eval-utts 0 \\
      --out students/study/t01-s1235                                                  # the replicate
  python scripts/03_build_student.py --scratch t005 --device cpu --step0-eval-utts 0 --out students/study/t005
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
# the teacher commit read, the one the targets came from: must equal 02_teacher_pass.MODEL_REVISION (and the commit
# MODEL_CARD.md names); without a pin every hub read silently follows `main`
TEACHER_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"
EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon"]
SOURCES = ["reazon_small", "emilia_yodas", "galgame"]  # configs/viability.json "sources": calibrate on what it trains on
MAX_SECONDS = 30.0  # LogMel / HF fast path; the teacher pass skipped longer utterances anyway
IMPORTANCE_FORMAT = 1
META_FORMAT = 2  # 2: the size study's fields; "bn" is the mode string (format 1 had the recalibration stats there)
FUNCTION_LOST_CER = 0.9  # STUDY.md 2.2: step-0 CER vs the teacher >= this on the 60-utterance gate = function lost
BN_MODES = {"keep": "teacher", "recal": "recal"}  # --bn value -> student_meta.json "bn"


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


def check_teacher_matches_targets(teacher_root: Path, commit: str | None) -> None:
    """Exit if <teacher_root>/meta.json names another teacher commit than the one just loaded: the student would start
    from other weights (and ship another tokenizer) than the ones its distillation targets came from. Skipped when
    either side is unknown (a local teacher dir; a teacher_out/meta.json from before 02 recorded model_revision)."""
    path = Path(teacher_root) / "meta.json"
    if commit is None or not path.exists():
        return
    targets = json.loads(path.read_text(encoding="utf-8")).get("model_revision")
    if targets is not None and targets != commit:
        sys.exit(f"teacher commit {commit} is not the one {path} was computed from ({targets}); pass "
                 f"--teacher-revision {targets}, or recompute the targets with 02_teacher_pass.py")


def load_or_compute_importance(teacher, featurizer, utts: list[dict], args, spec, device, out: Path, log=print):
    """Stage 3, cached in --importance-cache (default <out>/importance.pt; atomic). Returns (importance, info).

    The layers measured are spec's kept layers, or all the teacher's with --importance-layers all: FFN importance is
    measured inside the full teacher, so it does not depend on which layers a student keeps, and one all-layer pass
    serves every size. A cache is reused when it has the same key (teacher, commit, calibration ids) and covers the
    layers asked for; it is written only when it is recomputed. A cache named with --importance-cache that exists but
    does not match stops the build: it is shared, and recomputing would overwrite it (maybe with fewer layers)."""
    import torch

    from kitsune import student as S

    named = getattr(args, "importance_cache", None)
    path = Path(named or out / "importance.pt")
    if getattr(args, "importance_layers", "spec") == "all":
        layers = list(range(teacher.config.encoder_config.num_hidden_layers))
    else:
        layers = list(spec.enc_layers)
    # the resolved commit (None for a local dir): importance from other teacher weights is recomputed, not reused
    key = dict(teacher=args.teacher, teacher_revision=getattr(teacher.config, "_commit_hash", None),
               ids_sha256=ids_sha([u["id"] for u in utts]), n_utts=len(utts))
    if path.exists():
        try:
            c = torch.load(path, map_location="cpu", weights_only=True)
            ok = (c.get("format") == IMPORTANCE_FORMAT and c.get("key") == key
                  and set(layers) <= set(c.get("layers", [])))
        except Exception as e:  # a torn file from a killed run
            ok, c = False, None
            log(f"  importance cache unreadable ({e}); recomputing")
        if ok:
            log(f"  importance: reusing {path} ({c['info']['n_utts']} utts, {len(c['layers'])} layers, computed "
                f"{c['info']['created']})")
            return S.importance_from_state(c["importance"]), dict(c["info"], reused=True, path=str(path))
        if named:
            sys.exit(f"--importance-cache {path} does not match this build (teacher commit, calibration ids or layers:"
                     f" it has {c and c.get('key')}, {len(c.get('layers', [])) if c else 0} layers; this build needs "
                     f"{key}, layers {layers}); it is left as it is")
        if c is not None:
            log(f"  importance cache {path} does not match this run (teacher/sample/layers); recomputing")

    t0 = time.time()
    imp = S.ffn_importance(teacher, feature_batches(utts, featurizer, device, args.calib_batch_s, "importance"),
                           layers, device)
    info = dict(n_utts=len(utts), audio_s=float(sum(u["duration"] for u in utts)), wall_s=round(time.time() - t0, 1),
                created=now(), sources=sorted({u["source"] for u in utts}), ids_sha256=key["ids_sha256"],
                n_layers=len(layers), device=str(device))
    if hasattr(teacher, "parameters"):
        info["teacher_dtype"] = str(next(teacher.parameters()).dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(dict(format=IMPORTANCE_FORMAT, key=key, layers=layers, info=info,
                    importance=S.importance_to_state(imp)), tmp)
    tmp.replace(path)
    log(f"  importance: {len(utts)} utts ({info['audio_s'] / 3600:.2f} h), {len(layers)} layers in "
        f"{info['wall_s']:.0f} s -> {path}")
    return imp, dict(info, reused=False, path=str(path))


def ffn_from_student(src: Path, spec, teacher_ffn: int, log=print) -> tuple[dict, dict | None, dict]:
    """--ffn-from: the FFN selection another student recorded (its student_meta.json kept.ffn), read-only.

    Returns (keep, importance or None, info). When the source dir has an importance.pt that covers spec's layers, it
    is loaded (for the kept-mass summary) and must reproduce the recorded selection under today's ranking rule; a
    selection the ranking does not reproduce is an error, not a silent substitute."""
    import torch

    from kitsune import student as S

    meta = S.load_meta(src)
    if not meta:
        sys.exit(f"--ffn-from {src}: no student_meta.json")
    keep = S.check_keep(S.keep_from_meta(meta), spec, teacher_ffn)
    info = dict(ffn_from=str(src), source_spec=meta.get("spec"),
                source_meta_sha256=hashlib.sha256((Path(src) / "student_meta.json").read_bytes()).hexdigest(),
                source_calibration_ids_sha256=(meta.get("calibration") or {}).get("importance_ids_sha256"))
    importance = None
    ipath = Path(src) / "importance.pt"
    if ipath.exists():
        c = torch.load(ipath, map_location="cpu", weights_only=True)
        if set(spec.enc_layers) <= set(c.get("layers", [])):
            importance = S.importance_from_state(c["importance"])
            again = S.select_ffn_neurons(importance, spec.enc_layers, spec.ffn_dim, teacher_ffn)
            if set(again) != set(keep) or not all(torch.equal(again[k], keep[k]) for k in keep):
                sys.exit(f"--ffn-from {src}: its importance.pt does not reproduce its recorded FFN selection")
            info.update(source_importance=str(ipath), source_importance_key=c.get("key"),
                        source_importance_info=c.get("info"), reproduced_by_importance=True,
                        source_importance_sha256=hashlib.sha256(ipath.read_bytes()).hexdigest())
    log(f"  FFN selection from {src} ({len(keep)} FFNs"
        f"{', reproduced by its importance.pt' if importance is not None else ''})")
    return keep, importance, info


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


def function_check(step0: dict) -> dict:
    """STUDY.md 2.2's rule on a step-0 eval: function "lost" when the pooled greedy CER against the teacher's own
    transcripts is >= 90 %. `on_gate` says whether the eval was the rule's 60-utterance gate (20 per gate set); the
    study builds it with --device cpu --step0-eval-utts 20."""
    cer = float(step0["greedy"]["all"]["cer_teacher_corpus"])
    return dict(cer_vs_teacher=cer, kl=float(step0["teacher_forced"]["all"]["kl"]), threshold=FUNCTION_LOST_CER,
                function="lost" if cer >= FUNCTION_LOST_CER else "kept",
                on_gate=step0.get("n_per_set") == {s: 20 for s in EVAL_SETS})


def build_identity(args) -> dict:
    """What makes two builds the same student (a "saved" dir is only resumed by an identical build)."""
    if args.scratch:
        return dict(init="scratch", shape=args.scratch, seed=args.seed)
    return dict(init="pruned", enc_layers=args.enc_layers, ffn=args.ffn, dec_layers=sorted(args.dec_layers),
                tie_head=not args.no_tie_head, bn=args.bn, ffn_from=str(args.ffn_from) if args.ffn_from else None)


def legacy_identity(old: dict) -> dict | None:
    """build_identity of a format-1 student_meta.json (pruned, recalibrated, no --ffn-from)."""
    a = old.get("args") or {}
    if "enc_layers" not in a:
        return None
    return dict(init="pruned", enc_layers=a["enc_layers"], ffn=a.get("ffn"), dec_layers=sorted(a.get("dec_layers", [])),
                tie_head=not a.get("no_tie_head", False), bn="recal", ffn_from=None)


# ------------------------------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc-layers", type=int, default=20, help="number of teacher encoder layers kept, evenly spaced")
    ap.add_argument("--ffn", type=int, default=2560, help="encoder FFN width (5120 = no pruning)")
    ap.add_argument("--dec-layers", type=int, nargs="+", default=[0, 2, 5, 7], help="teacher decoder layers kept")
    ap.add_argument("--no-tie-head", action="store_true", help="keep proj_out.weight as its own parameter")
    ap.add_argument("--calib-utts", type=int, default=1000, help="utterances for FFN importance")
    ap.add_argument("--bn-utts", type=int, default=1000, help="utterances for the BatchNorm recalibration")
    ap.add_argument("--sources", nargs="+", default=SOURCES,
                    help="train sources to calibrate on, equal share each (default: the viability run's)")
    ap.add_argument("--out", default=None, help="student dir (default: students/b<enc>x<ffn>-d<n_dec>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--step0-eval-utts", type=int, default=200, help="per eval set; 0 skips the step-0 eval")
    ap.add_argument("--eval-sets", nargs="+", default=EVAL_SETS)
    ap.add_argument("--teacher", default=TEACHER_ID, help="HF repo id or local dir of the teacher")
    ap.add_argument("--teacher-revision", default=TEACHER_REVISION,
                    help="commit of --teacher to read (default: the pin 02_teacher_pass read); ignored for a local dir")
    ap.add_argument("--selection", default=str(ROOT / "selection" / "viability.parquet"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--teacher-root", default=str(ROOT / "teacher_out"))
    ap.add_argument("--eval-cache", default=str(ROOT / "cache" / "eval"), help="kitsune.trainset eval store cache")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--calib-per-group", type=int, default=16, help="max utterances taken from one parquet row group")
    ap.add_argument("--calib-batch-s", type=float, default=240.0, help="padded audio seconds per importance batch")
    ap.add_argument("--eval-batch-s", type=float, default=240.0, help="padded audio seconds per step-0 eval batch")
    ap.add_argument("--force", action="store_true", help="rebuild even if student_meta.json says complete/saved")
    ap.add_argument("--teacher-dtype", choices=["bf16", "fp32"], default="bf16",
                    help="dtype the teacher is loaded and run in (importance pass); fp32 is the fast one on a CPU "
                         "without bf16 units. The student is fp32 and bitwise the same either way (bf16 checkpoint)")
    ap.add_argument("--bn", choices=sorted(BN_MODES), default=None,
                    help="BatchNorm of a pruned student: recal = recalibrate on --bn-utts utterances (default, the "
                         "first run), keep = the teacher's running stats (the size study, decision 16)")
    ap.add_argument("--importance-layers", choices=["spec", "all"], default="spec",
                    help="measure FFN importance in the kept layers only, or in all teacher layers (one pass that "
                         "every size can reuse through --importance-cache)")
    ap.add_argument("--importance-cache", default=None,
                    help="FFN importance cache to reuse or write (default: <out>/importance.pt)")
    ap.add_argument("--ffn-from", default=None,
                    help="reuse the FFN selection of this built student (its student_meta.json kept.ffn; read-only) "
                         "instead of measuring importance: T-0.6B takes the first run's students/b20x2560-d4")
    ap.add_argument("--scratch", default=None, metavar="SHAPE",
                    help="build a randomly initialised student of this kitsune.student.SCRATCH_SHAPES shape (t01, "
                         "t005, bridge) instead of pruning; --seed seeds the init")
    args = ap.parse_args(argv)
    if args.scratch:
        bad = [f for f, v in (("--bn", args.bn), ("--ffn-from", args.ffn_from),
                              ("--importance-cache", args.importance_cache)) if v]
        if args.importance_layers != "spec":
            bad.append("--importance-layers")
        if bad:
            ap.error(f"--scratch builds a random init (fresh BN, no FFN selection): {', '.join(bad)} do not apply")
    else:
        args.bn = args.bn or "recal"
        if args.ffn_from and (args.importance_layers != "spec" or args.importance_cache):
            ap.error("--ffn-from takes the FFN selection as it is: --importance-layers/--importance-cache do not apply")
    if args.out is None:
        name = (f"scratch-{args.scratch}-s{args.seed}" if args.scratch
                else f"b{args.enc_layers}x{args.ffn}-d{len(args.dec_layers)}")
        args.out = str(ROOT / "students" / name)
    for k in ("selection", "data", "teacher_root", "eval_cache", "out", "importance_cache", "ffn_from"):
        if getattr(args, k) is not None:
            setattr(args, k, Path(getattr(args, k)))
    return args


def teacher_tag(repo: str, commit: str | None) -> str:
    """student_meta.json "teacher": <repo>@<commit> (just the repo/dir when the commit is unknown, e.g. a local dir)."""
    return f"{repo}@{commit}" if commit else str(repo)


def assert_count(report: dict, expected: int | None, what: str) -> None:
    """The count assert every study builder runs before it saves: total == closed form (== the pre-registered count
    of a named study shape)."""
    if report["total"] != report["closed_form"]:
        raise AssertionError(f"{what}: {report['total']:,} params != closed form {report['closed_form']:,}")
    if expected is not None and report["total"] != expected:
        raise AssertionError(f"{what}: {report['total']:,} params != the study's {expected:,}")


def build_pruned(args, spec, teacher_cfg, processor, featurizer, device, meta: dict, log=print):
    """Stages 1-6 of a pruned student. Returns the student (fp32 on `device`); fills `meta`."""
    import torch
    from transformers import CohereAsrForConditionalGeneration

    from kitsune import student as S

    dur = meta["durations_s"]
    teacher_ffn = teacher_cfg.encoder_config.intermediate_size
    prune = spec.ffn_dim < teacher_ffn
    need_importance = prune and not args.ffn_from
    need_audio = need_importance or args.bn == "recal"

    t = time.time()
    n_sample = max(args.calib_utts if need_importance else 0, args.bn_utts if args.bn == "recal" else 0)
    utts = []
    if need_audio:
        log(f"[1/8] calibration sample: {n_sample} utts from {args.sources}")
        utts = sample_calibration(args.selection, args.data, args.sources, n_sample, args.seed, args.calib_per_group,
                                  log=log)
    else:
        log("[1/8] calibration sample: not needed (FFN selection given, BN kept)")
    imp_utts = utts[:args.calib_utts] if need_importance else []
    bn_utts = utts[:args.bn_utts] if args.bn == "recal" else []
    dur["sample"] = round(time.time() - t, 1)

    t = time.time()
    dtype = torch.float32 if args.teacher_dtype == "fp32" else torch.bfloat16
    log(f"[2/8] teacher {args.teacher} ({args.teacher_dtype}) -> {device}")
    teacher = CohereAsrForConditionalGeneration.from_pretrained(args.teacher, revision=args.teacher_revision,
                                                                dtype=dtype, attn_implementation="sdpa")
    teacher = teacher.to(device).eval()
    # provenance: the commit --teacher-revision resolved to (None for a local directory); it must be the one the
    # teacher_out targets came from
    meta["teacher_revision"] = getattr(teacher.config, "_commit_hash", None)
    meta["teacher"] = teacher_tag(args.teacher, meta["teacher_revision"])
    check_teacher_matches_targets(args.teacher_root, meta["teacher_revision"])
    dur["teacher_load"] = round(time.time() - t, 1)

    t = time.time()
    keep_given = None
    if not prune:
        log("[3/8] FFN importance: nothing to prune")
        importance, imp_info = None, dict(skipped="ffn_dim == teacher width: nothing to prune")
    elif args.ffn_from:
        log(f"[3/8] FFN selection from {args.ffn_from}")
        keep_given, importance, imp_info = ffn_from_student(args.ffn_from, spec, teacher_ffn, log=log)
    else:
        log(f"[3/8] FFN importance over {len(imp_utts)} utts ({args.importance_layers} layers)")
        importance, imp_info = load_or_compute_importance(teacher, featurizer, imp_utts, args, spec, device,
                                                          args.out, log=log)
    dur["importance"] = round(time.time() - t, 1)

    t = time.time()
    log("[4/8] build the student on CPU")
    student = S.build_student(teacher, spec, importance, keep=keep_given)
    keep = (S.check_keep(keep_given, spec, teacher_ffn) if keep_given is not None
            else S.select_ffn_neurons(importance, spec.enc_layers, spec.ffn_dim, teacher_ffn))
    report = S.param_report(student)
    log(f"  {report['total']:,} params (closed form {report['closed_form']:,}), encoder share "
        f"{report['encoder_share']:.1%}")
    expected = None
    if teacher_cfg.encoder_config.num_hidden_layers == S.TEACHER_ENC_LAYERS and spec.tie_head:
        expected = S.PRUNED_EXPECTED_PARAMS.get((len(spec.enc_layers), spec.ffn_dim, tuple(spec.dec_layers)))
    assert_count(report, expected, "pruned student")
    dur["build"] = round(time.time() - t, 1)

    log("[5/8] free the teacher")
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t = time.time()
    student.to(device=device, dtype=torch.float32)
    bn_recal = None
    if args.bn == "recal":
        log(f"[6/8] BatchNorm recalibration on {len(bn_utts)} utts (fp32, batch size 1)")
        bn_recal = S.recalibrate_batchnorm(student, feature_batches(bn_utts, featurizer, device, args.calib_batch_s,
                                                                    "bn"), device, max_utts=args.bn_utts)
        log(f"  BN drift vs teacher stats: mean shift {bn_recal['mean_shift_std']:.3f} std, "
            f"|log var ratio| {bn_recal['abs_log_var_ratio']:.3f} ({bn_recal['n_utts']} utts)")
    else:
        log("[6/8] BatchNorm: keeping the teacher's running stats (--bn keep)")
    dur["bn"] = round(time.time() - t, 1)

    calibration = dict(sources=args.sources, seed=args.seed, per_group=args.calib_per_group,
                       n_importance=len(imp_utts), n_bn=bn_recal["n_utts"] if bn_recal else 0)
    if imp_utts:
        calibration.update(importance_ids_sha256=ids_sha([u["id"] for u in imp_utts]),
                           hours_importance=sum(u["duration"] for u in imp_utts) / 3600,
                           per_source={s: sum(u["source"] == s for u in imp_utts) for s in args.sources})
    elif args.ffn_from:
        calibration["importance_ids_sha256"] = imp_info.get("source_calibration_ids_sha256")
    if bn_utts:
        calibration["bn_ids_sha256"] = ids_sha([u["id"] for u in bn_utts[:calibration["n_bn"]]])
    meta.update(
        family="aed", init_class="pruned_kept", bn=BN_MODES[args.bn], seed=args.seed,
        enc_layers=spec.enc_layers, ffn=spec.ffn_dim, dec_layers=spec.dec_layers,
        **S.count_fields(student), expected_params=expected,
        kept=dict(enc_layers=spec.enc_layers, dec_layers=spec.dec_layers,
                  ffn={f"{l}.{n}": idx for (l, n), idx in sorted(keep.items())}),
        importance=dict(imp_info, **(S.importance_summary(importance, keep) if keep and importance else {})),
        params=report, calibration=calibration,
    )
    if bn_recal is not None:
        meta["bn_recal"] = bn_recal
    del utts, imp_utts, bn_utts
    return student


def build_scratch(args, teacher_cfg, device, meta: dict, log=print):
    """Stages 1-6 of a from-scratch student (kitsune.student.build_scratch_student). Returns the student on `device`."""
    from dataclasses import asdict

    import torch

    from kitsune import student as S

    shape = S.SCRATCH_SHAPES[args.scratch]
    t = time.time()
    meta["teacher_revision"] = getattr(teacher_cfg, "_commit_hash", None)
    meta["teacher"] = teacher_tag(args.teacher, meta["teacher_revision"])
    check_teacher_matches_targets(args.teacher_root, meta["teacher_revision"])  # its tokenizer and processor ship
    log(f"[1/3] build {args.scratch} from scratch on CPU (seed {args.seed}): {shape}")
    student = S.build_scratch_student(teacher_cfg, shape, args.seed, name=args.scratch)
    report = S.param_report(student)
    expected = S.SCRATCH_EXPECTED_PARAMS.get(args.scratch)
    log(f"  {report['total']:,} params (closed form {report['closed_form']:,}"
        f"{f', the study {expected:,}' if expected else ''}), encoder share {report['encoder_share']:.1%}")
    assert_count(report, expected, f"scratch student {args.scratch}")
    meta.update(
        family="aed", init_class="scratch", bn="fresh", seed=args.seed, **S.count_fields(student),
        expected_params=expected, scratch=dict(name=args.scratch, **asdict(shape)),
        init=dict(weights="HF default (N(0, initializer_range), zero biases, LayerNorm 1/0)",
                  subsampling_conv="PyTorch default reset_parameters (kaiming-uniform)",
                  pos_emb="sinusoid / sqrt(D), NeMo FixedPositionalEncoding (frozen by the trainer)",
                  batchnorm="fresh: running mean 0, var 1, trains", tie_head=True, scale_input=False, dropout=0.0),
        params=report,
    )
    meta["durations_s"]["build"] = round(time.time() - t, 1)
    return student.to(device=device, dtype=torch.float32)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformers import AutoConfig, AutoProcessor

    from kitsune import student as S
    from kitsune.features import LogMel

    out = args.out
    if args.scratch and args.scratch not in S.SCRATCH_SHAPES:
        sys.exit(f"--scratch {args.scratch}: not one of {sorted(S.SCRATCH_SHAPES)}")
    teacher_cfg = AutoConfig.from_pretrained(args.teacher, revision=args.teacher_revision)
    n_teacher_layers = teacher_cfg.encoder_config.num_hidden_layers  # 48 for the real one
    spec = None if args.scratch else S.StudentSpec(
        enc_layers=S.evenly_spaced(args.enc_layers, n_teacher_layers), ffn_dim=args.ffn,
        dec_layers=sorted(args.dec_layers), tie_head=not args.no_tie_head)
    identity = build_identity(args)
    old = S.load_meta(out)
    same = old.get("build", legacy_identity(old)) == identity
    # checked before "complete": a finished dir of another build (other shape, seed or BN mode) is not "already built"
    if old.get("stage") and not same and not args.force:
        sys.exit(f"{out} holds a different student ({old.get('build') or old.get('spec')}); use another --out or "
                 f"--force")
    if old.get("stage") == "complete" and not args.force:
        print(f"{out}: already built ({old.get('timestamps', {}).get('finished')}); --force to rebuild")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU free: {free / 2**30:.2f} of {total / 2**30:.1f} GiB")
    if spec is not None:
        print(f"student: enc {spec.enc_layers} ffn {spec.ffn_dim} dec {spec.dec_layers} tied={spec.tie_head} "
              f"bn={args.bn} -> {out}")
    else:
        print(f"student: {args.scratch} from scratch, seed {args.seed} -> {out}")

    processor = AutoProcessor.from_pretrained(args.teacher, revision=args.teacher_revision)
    featurizer = LogMel.from_feature_extractor(processor.feature_extractor).to(device)
    t_start = time.time()

    if old.get("stage") == "saved" and not args.force:
        print(f"resuming: {out} is saved; only the step-0 eval is left")
        meta = old
        meta.setdefault("timestamps", {})["resumed"] = now()
        student = S.load_student(out, device)
    else:
        meta = dict(format=META_FORMAT, stage="building", teacher=args.teacher, build=identity,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    git=git_info(), host=host_info(device), timestamps=dict(started=now()), durations_s={})
        if spec is not None:
            meta["spec"] = spec
            student = build_pruned(args, spec, teacher_cfg, processor, featurizer, device, meta)
        else:
            student = build_scratch(args, teacher_cfg, device, meta)
        t = time.time()
        print(f"[save] -> {out}")
        meta["stage"] = "saved"
        meta["timestamps"]["saved"] = now()
        S.save_student(student, out, processor, meta)
        meta["durations_s"]["save"] = round(time.time() - t, 1)

    if args.step0_eval_utts > 0:
        print(f"[step-0] eval: {args.step0_eval_utts} utts per set of {args.eval_sets}")
        try:
            meta["step0"] = step0_eval(student, processor, featurizer, args, device, out)
        except Exception as e:
            meta["step0"] = dict(error=repr(e), traceback=traceback.format_exc())
            S.write_meta(out, meta)  # still stage "saved": a re-run retries only the eval
            print(f"step-0 eval failed ({e!r}); the student is saved. Re-run to retry the eval, or finish with "
                  f"--step0-eval-utts 0")
            raise
        if meta.get("init_class", "").startswith("pruned"):
            fc = meta["function_check"] = function_check(meta["step0"])
            print(f"  function check: step-0 CER vs teacher {fc['cer_vs_teacher']:.3f} -> function {fc['function']} "
                  f"(rule: lost at >= {FUNCTION_LOST_CER:.0%}{'' if fc['on_gate'] else '; NOT the 60-utterance gate'})")
            if fc["function"] == "lost":
                print(f"  WARNING: by STUDY.md 2.2 this pruned student is 'function lost', but init_class says "
                      f"{meta['init_class']}")
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
