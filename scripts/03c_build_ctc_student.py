"""Build a Parakeet-family student of the size study (P-0.3B, P-0.1B, P-0.05B) from the converted Parakeet dir, on CPU.

The shape and the reasons for each step are in kitsune/ctc_student.py. The pipeline:
  1. check the model dir against kitsune.parakeet's sha256 pins; load the anchor (ParakeetForCTC 24x4096, fp32)
  2. FFN importance over ALL teacher layers on the first run's calibration utterances (--calib-ids: exactly the ids
     T-0.3B calibrates on, read by scripts/03_build_student.py's read_calibration_ids; without it, today's draw of
     03's sample_calibration: the selection's kept train rows of the viability sources, seed 1234), featurised on
     Parakeet's path; the ids go to <out>/calibration_ids.txt. Cached in --importance and reused by every size: the
     cache key is the teacher files, the ids and the feature path, never the student shape.
  3. build: kept layers (evenly spaced, the last always), FFN width, the teacher's BN stats; params == closed form
  4. step-0 gate of the built (fp32) student; --golden: the 32 golden JSUT rows through it; for the unpruned shape
     the reproduction check below (a failure exits non-zero before anything is saved)
  5. save: bf16 weights with fp32 BN stats, Parakeet processor + tokenizer, MODEL_CARD.md (CC-BY-4.0),
     student_meta.json (stage "saved")
  6. step-0 gate of the SAVED student (what training starts from) -> init_class; --golden: the golden rows again
  7. student_meta.json stage "complete"

The step-0 gate is the study's 60-utterance CPU gate (STUDY 2.2): 20 utterances of <= 15 s from the first shard of each
of eval_jsut, eval_reazon, eval_cv8 (seeded row-group/row order, seed 2), through the anchor and the student. The init
class is fixed in advance: "pruned_lost" when the mean per-utterance CER of the student's greedy CTC output against the
teacher's is >= 90 %, else "pruned_kept" (this rule classifies the Parakeet students only: the pruned Transcribe
students are "function kept" by the owner's decision of 2026-09-26, scripts/03_build_student.py). Also reported: corpus
CERs, CER against the reference, the full-vocab frame KL, the CTC-KD objective on the teacher's own targets
(kitsune.ctc_kd), argmax agreement and blank shares.

`--enc-layers all --ffn 4096` must reproduce the teacher: the built student's gate log-probs equal the anchor's (max
abs diff 0, frame KL 0) and --golden gives the 32 golden CTC transcripts of kitsune/parakeet_golden.json. The script
enforces it (anchor_problems: also every tensor equal to the anchor's): if any of these fails, student_meta.json gets
stage "failed" with the problems, no weights are written and the exit code is non-zero. Only the golden transcripts
catch a change in the forward code itself (the anchor runs the same modules), so run the check with --golden. The saved
copy is bf16, like the encoder that made the label box's targets; that rounding is reported separately (measured on
the anchor: frame KL 0.0008, 59/60 gate hypotheses and 31/32 golden transcripts identical - the other one drops a
Japanese comma, so its normalised CER is 0).

Calibration ids. The first run's exact 1,000 ids (importance_ids_sha256 5e31cd68...) cannot be drawn again from the
laptop's selection: selection/viability.parquet was rewritten on 2026-09-25 and emilia_yodas / galgame gained shards
after that build. They were recovered from the HF dataset's history (the selection at revision 704d1eeace, the draw
replayed over the shards that existed at the first build; owner-approved 2026-09-26) and every pruned student of the
study calibrates on them: --calib-ids <file> (one id per line, in order; each a kept train row of --selection with
audio under --data). --expect-calib-sha pins the ids either way, and they are stored in the importance cache and in
<out>/calibration_ids.txt.

Usage (laptop, CPU; IDS = the first run's calibration_ids.txt, e.g. D:/kitsune-students/study/t03/calibration_ids.txt):
  python scripts/03c_build_ctc_student.py --enc-layers 16 --ffn 2560 --name P-0.3B --out D:/kitsune-students/study/p03 \
      --importance D:/kitsune-students/study/parakeet_importance.pt --calib-ids IDS --expect-calib-sha 5e31cd68...
  python scripts/03c_build_ctc_student.py --enc-layers 8 --ffn 768 --name P-0.1B --out .../p01 --importance ... \
      --calib-ids IDS
  python scripts/03c_build_ctc_student.py --enc-layers 4 --ffn 768 --name P-0.05B --out .../p005 --importance ... \
      --calib-ids IDS
  python scripts/03c_build_ctc_student.py --enc-layers all --ffn 4096 --golden --out D:/kitsune-tmp/anchor   # the check
"""
import argparse
import gc
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# torch / transformers are imported inside the functions, so --help is instant
DEFAULT_MODEL_DIR = ROOT / "cache" / "parakeet-tdt_ctc-0.6b-ja-hf"
GATE_SETS = ["eval_jsut", "eval_reazon", "eval_cv8"]
GATE_UTTS, GATE_SEED, GATE_MAX_S = 20, 2, 15.0
LOST_CER = 0.90  # STUDY 2.2: step-0 CER vs the teacher >= 90 % on the gate = "function lost"
GOLDEN_SHARD = Path("shards") / "eval_jsut" / "eval-00000.parquet"
IMPORTANCE_FORMAT = 1
FEATURES = "ParakeetFeatureExtractor path: kitsune LogMel, 80 mel, preemphasis 0.97, dither 0"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_info() -> dict:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()

    try:
        return dict(sha=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                    dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except Exception as e:  # no git is not a reason to fail a build
        return dict(error=repr(e))


def host_info() -> dict:
    import torch
    import transformers

    return dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, transformers=transformers.__version__, threads=torch.get_num_threads())


def build_script():
    """scripts/03_build_student.py as a module: its sample_calibration IS the first run's calibration draw."""
    spec = importlib.util.spec_from_file_location("kitsune_script_03_build_student",
                                                  ROOT / "scripts" / "03_build_student.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ importance


def importance_key(args, ids_sha: str, n: int) -> dict:
    from kitsune import parakeet as pk

    return dict(teacher=f"{pk.NEMO_REPO}@{pk.NEMO_REVISION}", model_sha256=pk.PARAKEET_FILES.get("model.safetensors"),
                ctc_head_sha256=pk.PARAKEET_FILES.get("ctc_head.safetensors"), ids_sha256=ids_sha, n_utts=n,
                features=FEATURES)


def load_or_compute_importance(anchor, feats, args, log=print):
    """(importance over all anchor layers, info, the calibration record, the calibration ids in order). Cached in
    args.importance (atomic write); main writes the ids to <out>/calibration_ids.txt with the saved student, so a
    rebuild that fails before its save leaves the previous student's id list in place."""
    import torch

    from kitsune import ctc_student as CS
    from kitsune import student as S

    m03 = build_script()
    path = Path(args.importance)
    n_layers = anchor.config.encoder_config.num_hidden_layers
    t0 = time.time()
    if args.calib_ids:
        ids = m03.read_ids_file(args.calib_ids)
        if len(ids) < args.calib_utts:
            sys.exit(f"--calib-ids {args.calib_ids}: {len(ids)} ids, this build needs {args.calib_utts}")
        log(f"[2/7] calibration: the first {args.calib_utts} of {len(ids)} ids in {args.calib_ids}")
        utts = m03.read_calibration_ids(args.selection, args.data, args.sources, ids[:args.calib_utts], log=log)
    else:
        log(f"[2/7] calibration sample: {args.calib_utts} utts of {args.sources} from {args.selection}")
        utts = m03.sample_calibration(args.selection, args.data, args.sources, args.calib_utts, args.seed,
                                      args.calib_per_group, log=log)
    ids = [u["id"] for u in utts]
    key = importance_key(args, m03.ids_sha(ids), len(utts))
    sample_s = round(time.time() - t0, 1)
    if args.expect_calib_sha and key["ids_sha256"] != args.expect_calib_sha:
        sys.exit(f"calibration ids sha256 {key['ids_sha256']} != --expect-calib-sha {args.expect_calib_sha}: not the "
                 f"first run's calibration sample")
    calib = dict(sources=args.sources, seed=args.seed, per_group=args.calib_per_group, n=len(utts),
                 ids_sha256=key["ids_sha256"], hours=sum(u["duration"] for u in utts) / 3600,
                 per_source={s: sum(u["source"] == s for u in utts) for s in args.sources}, sample_s=sample_s,
                 ids_from=str(args.calib_ids) if args.calib_ids else "seeded sample", ids_file="calibration_ids.txt")
    if path.exists():
        try:
            c = torch.load(path, map_location="cpu", weights_only=True)
            ok = c.get("format") == IMPORTANCE_FORMAT and c.get("key") == key and \
                set(range(n_layers)) <= set(c.get("layers", []))
        except Exception as e:  # a torn file from a killed run
            ok, c = False, None
            log(f"  importance cache unreadable ({e}); recomputing")
        if ok:
            log(f"  importance: reusing {path} (computed {c['info']['created']})")
            return S.importance_from_state(c["importance"]), dict(c["info"], reused=True, path=str(path)), calib, ids
        if c is not None:
            log(f"  importance cache {path} does not match (teacher files / calibration ids / features); recomputing")
    log(f"[3/7] FFN importance over all {n_layers} layers, {len(utts)} utts ({calib['hours']:.2f} h)")
    t0 = time.time()
    batches = m03.feature_batches(utts, feats.logmel, feats.device, args.calib_batch_s, "importance")
    imp = CS.ffn_importance_ctc(anchor, batches, list(range(n_layers)))
    info = dict(n_utts=len(utts), audio_s=float(sum(u["duration"] for u in utts)), wall_s=round(time.time() - t0, 1),
                created=now(), ids_sha256=key["ids_sha256"], layers=list(range(n_layers)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    # the ids themselves travel with the cache: the draw depends on the laptop's selection and shard list
    torch.save(dict(format=IMPORTANCE_FORMAT, key=key, layers=list(range(n_layers)), info=info, ids=ids,
                    importance=S.importance_to_state(imp)), tmp)
    tmp.replace(path)
    log(f"  importance: {info['wall_s']:.0f} s -> {path}")
    return imp, dict(info, reused=False, path=str(path)), calib, ids


# ------------------------------------------------------------------------------------------------ step-0 gate


def gate_sample(data_root: Path, sets: list[str], n: int, seed: int, max_s: float, log=print) -> list[dict]:
    """n utterances <= max_s per set from the set's first eval shard, in a seeded row-group/row order (the study's
    60-utterance gate; per set a fresh generator with `seed`)."""
    import numpy as np
    import pyarrow.parquet as pq

    from kitsune.audio import decode_audio

    out = []
    for s in sets:
        rng = np.random.default_rng(seed)
        files = sorted((data_root / "shards" / s).glob("eval*.parquet"))
        if not files:
            raise FileNotFoundError(f"no eval shard for {s} under {data_root / 'shards' / s}")
        pf = pq.ParquetFile(files[0])
        got = []
        for g in rng.permutation(pf.num_row_groups):
            t = pf.read_row_group(int(g), columns=["id", "audio", "text", "duration"])
            for j in rng.permutation(t.num_rows):
                if t.column("duration")[int(j)].as_py() > max_s:
                    continue
                try:
                    w = decode_audio(t.column("audio")[int(j)].as_py())
                except Exception as e:
                    log(f"  gate: bad audio, skipping {t.column('id')[int(j)].as_py()}: {e}")
                    continue
                got.append(dict(id=t.column("id")[int(j)].as_py(), set=s, text=t.column("text")[int(j)].as_py(),
                                wave=w))
                if len(got) >= n:
                    break
            if len(got) >= n:
                break
        out += got
    return out


def run_ctc(model, feats, waves: list, batch: int = 8) -> list[tuple]:
    """[(log_probs (T_i, V) fp32 CPU, n_frames)] per utterance, batches of `batch` in order."""
    import torch

    from kitsune import ctc_student as CS

    out = []
    with torch.inference_mode():
        for i in range(0, len(waves), batch):
            f, fl = feats(waves[i:i + batch])
            lp, n = CS.ctc_log_probs(model, f, CS.lengths_to_mask(fl, f.shape[1]))
            lp, n = lp.cpu(), n.cpu()
            out += [(lp[b, :int(n[b])], int(n[b])) for b in range(lp.shape[0])]
    return out


def gate_metrics(student_out: list, teacher_out: list, utts: list[dict], tokenizer) -> dict:
    """Step-0 numbers of a student against the teacher's CTC output on the gate utterances."""
    import numpy as np
    import torch

    from kitsune import ctc_kd, ctc_targets
    from kitsune import ctc_student as CS
    from kitsune.evaluate import corpus_cer
    from kitsune.text import cer, normalize_ja

    def hyps(outs):
        return [CS.decode_ids(tokenizer, CS.greedy_ctc_ids(lp[None], torch.tensor([n]))[0]) for lp, n in outs]

    t_h, s_h = hyps(teacher_out), hyps(student_out)
    refs = [u["text"] for u in utts]
    kl_sum, frames, max_diff = 0.0, 0, 0.0
    agree = s_blank = t_blank = 0
    losses = []
    for (slp, sn), (tlp, tn) in zip(student_out, teacher_out):
        if sn != tn:
            raise AssertionError(f"student has {sn} frames, the teacher {tn}")
        kl_sum += float((tlp.exp() * (tlp - slp)).sum())
        frames += tn
        max_diff = max(max_diff, float((slp - tlp).abs().max())) if tn else max_diff
        sa, ta = slp.argmax(-1), tlp.argmax(-1)
        agree += int((sa == ta).sum())
        s_blank += int((sa == CS.CTC_BLANK).sum())
        t_blank += int((ta == CS.CTC_BLANK).sum())
        tgt = ctc_targets.targets_from_log_probs(tlp[None], torch.tensor([tn]))
        losses.append(ctc_kd.ctc_kd_losses(slp[None], ctc_targets.collate_frame_targets(tgt)))
    tot = {k: float(sum(float(x[k]) for x in losses)) for k in ("kl_dense", "kl_blank", "ctc", "n_tokens", "n_dense",
                                                                "n_blank_frames")}
    per_set = {}
    for s in sorted({u["set"] for u in utts}):
        idx = [i for i, u in enumerate(utts) if u["set"] == s]
        per_set[s] = dict(n=len(idx), cer_vs_teacher=float(np.mean([cer(s_h[i], t_h[i]) for i in idx])),
                          cer_vs_ref=float(np.mean([cer(s_h[i], refs[i]) for i in idx])))
    n_tok = max(tot["n_tokens"], 1.0)
    return dict(
        n=len(utts), frames=frames,
        cer_vs_teacher=float(np.mean([cer(h, t) for h, t in zip(s_h, t_h)])),
        cer_vs_teacher_corpus=corpus_cer(s_h, t_h)["cer"],
        cer_vs_ref=float(np.mean([cer(h, r) for h, r in zip(s_h, refs)])),
        cer_vs_ref_corpus=corpus_cer(s_h, refs)["cer"],
        teacher_cer_vs_ref=float(np.mean([cer(h, r) for h, r in zip(t_h, refs)])),
        teacher_cer_vs_ref_corpus=corpus_cer(t_h, refs)["cer"],
        frame_kl=kl_sum / max(frames, 1), max_abs_diff_log_probs=max_diff,
        kd_objective=(tot["kl_dense"] + tot["kl_blank"] + ctc_kd.W_CTC * tot["ctc"]) / n_tok,
        kl_per_token=(tot["kl_dense"] + tot["kl_blank"]) / n_tok, ctc_per_token=tot["ctc"] / n_tok,
        kl_dense_per_frame=tot["kl_dense"] / max(tot["n_dense"], 1.0),
        kl_blank_per_frame=tot["kl_blank"] / max(tot["n_blank_frames"], 1.0),
        argmax_agree=agree / max(frames, 1), student_argmax_blank=s_blank / max(frames, 1),
        teacher_argmax_blank=t_blank / max(frames, 1), empty_hyps=int(sum(not normalize_ja(h) for h in s_h)),
        identical_hyps=int(sum(a == b for a, b in zip(s_h, t_h))), per_set=per_set,
        samples=[dict(id=utts[i]["id"], teacher=t_h[i], student=s_h[i])
                 for i in range(0, len(utts), max(1, len(utts) // 6))],
    )


def golden_check(model, feats, tokenizer, data_root: Path) -> dict:
    """The 32 golden JSUT rows (tools/publish_parakeet.py --golden: first 32 rows of eval_jsut/eval-00000, batches
    of 8) through `model`; greedy CTC text vs kitsune/parakeet_golden.json."""
    import pyarrow.parquet as pq
    import torch

    from kitsune import ctc_student as CS
    from kitsune.audio import decode_audio

    golden = json.loads((ROOT / "kitsune" / "parakeet_golden.json").read_text(encoding="utf-8"))
    n = len(golden["ids"])
    t = pq.read_table(data_root / GOLDEN_SHARD, columns=["id", "audio"]).slice(0, n)
    ids = t.column("id").to_pylist()
    if ids != golden["ids"]:
        raise AssertionError("the local eval_jsut shard does not start with the golden ids")
    waves = [decode_audio(a) for a in t.column("audio").to_pylist()]
    outs = run_ctc(model, feats, waves, batch=8)
    got = [CS.decode_ids(tokenizer, CS.greedy_ctc_ids(lp[None], torch.tensor([k]))[0]) for lp, k in outs]
    bad = [dict(id=i, golden=g, got=h) for i, g, h in zip(ids, golden["ctc"], got) if g != h]
    return dict(n=n, match=n - len(bad), mismatches=bad)


def anchor_problems(student, anchor, meta: dict) -> list[str]:
    """Why the fp32 build of the unpruned shape is not the teacher ([] = it is). Checked: every tensor equals the
    anchor's; with the gate, the gate log-probs are identical (max abs diff 0, frame KL 0); with --golden, all golden
    CTC transcripts match. The tensors alone cannot catch a forward-code change (both run the same modules); the golden
    transcripts can, which is why the reproduction check is run with --golden."""
    import torch

    problems = []
    s_sd, a_sd = student.state_dict(), anchor.state_dict()
    if s_sd.keys() != a_sd.keys():
        problems.append(f"state_dict keys differ: {sorted(s_sd.keys() ^ a_sd.keys())[:8]}")
    else:
        diff = [k for k in s_sd if s_sd[k].dtype != a_sd[k].dtype or not torch.equal(s_sd[k], a_sd[k])]
        if diff:
            problems.append(f"{len(diff)} tensors differ from the anchor's, e.g. {diff[:4]}")
    built = meta.get("step0", {}).get("built")
    if built is not None and (built["max_abs_diff_log_probs"] != 0.0 or built["frame_kl"] != 0.0):
        problems.append(f"gate log-probs differ: max |dlogp| {built['max_abs_diff_log_probs']:.3g}, frame KL "
                        f"{built['frame_kl']:.3g} (both must be 0)")
    golden = meta.get("golden", {}).get("built")
    if golden is not None and golden["match"] != golden["n"]:
        problems.append(f"golden: {golden['match']}/{golden['n']} CTC transcripts equal, e.g. "
                        f"{golden['mismatches'][:2]}")
    return problems


# ------------------------------------------------------------------------------------------------ main


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="the converted Parakeet HF dir (pinned files)")
    ap.add_argument("--enc-layers", nargs="+", default=["all"],
                    help="'all', N (evenly spaced over the 24 teacher layers) or explicit teacher layer indices")
    ap.add_argument("--ffn", type=int, default=4096, help="encoder FFN width (4096 = no pruning)")
    ap.add_argument("--out", required=True, help="student dir")
    ap.add_argument("--name", default=None, help="study name for the card and meta, e.g. P-0.3B")
    ap.add_argument("--importance", default=None, help="importance cache (default <out>/importance.pt); share it "
                                                       "between sizes")
    ap.add_argument("--calib-utts", type=int, default=1000, help="calibration utterances for FFN importance")
    ap.add_argument("--sources", nargs="+", default=None, help="calibration sources (default: 03_build_student's)")
    ap.add_argument("--selection", default=str(ROOT / "selection" / "viability.parquet"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=1234, help="calibration draw seed (the first run's)")
    ap.add_argument("--calib-per-group", type=int, default=16)
    ap.add_argument("--calib-batch-s", type=float, default=240.0, help="padded audio seconds per importance batch")
    ap.add_argument("--expect-calib-sha", default=None,
                    help="exit unless the calibration ids hash to this (the first run's importance_ids_sha256)")
    ap.add_argument("--calib-ids", default=None,
                    help="calibrate on exactly these ids (one per line, in order; the first --calib-utts of them; e.g. "
                         "the first run's calibration_ids.txt) instead of the seeded sample, which a changed selection "
                         "or shard list no longer reproduces")
    ap.add_argument("--gate-sets", nargs="+", default=GATE_SETS)
    ap.add_argument("--gate-utts", type=int, default=GATE_UTTS, help="per gate set; 0 skips the step-0 gate")
    ap.add_argument("--gate-seed", type=int, default=GATE_SEED)
    ap.add_argument("--gate-max-s", type=float, default=GATE_MAX_S)
    ap.add_argument("--init-class", choices=["pruned_kept", "pruned_lost"], default=None,
                    help="only with --gate-utts 0: the init class the gate would otherwise measure")
    ap.add_argument("--golden", action="store_true",
                    help="also run the 32 golden JSUT rows through the built and the saved student")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--force", action="store_true", help="rebuild even if student_meta.json says complete")
    args = ap.parse_args(argv)
    for k in ("model_dir", "out", "selection", "data"):
        setattr(args, k, Path(getattr(args, k)))
    args.calib_ids = Path(args.calib_ids) if args.calib_ids else None
    args.importance = Path(args.importance) if args.importance else args.out / "importance.pt"
    if args.gate_utts <= 0 and args.init_class is None:
        ap.error("--gate-utts 0 needs --init-class (the gate is what measures it)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from kitsune import ctc_student as CS
    from kitsune import parakeet as pk
    from kitsune import student as S

    torch.set_num_threads(args.threads)
    if args.sources is None:
        args.sources = list(build_script().SOURCES)
    out = args.out
    old = CS.load_meta(out)
    if old.get("stage") == "complete" and not args.force:
        print(f"{out}: already built ({old.get('timestamps', {}).get('finished')}); --force to rebuild")
        return 0
    t_start = time.time()
    dur = {}

    print(f"[1/7] model dir {args.model_dir}: checking the pins")
    problems = pk.verify_model_dir(args.model_dir)
    if problems:
        sys.exit(f"{args.model_dir} is not the pinned Parakeet model:\n  " + "\n  ".join(problems))
    t = time.time()
    anchor = CS.load_parakeet_ctc(args.model_dir)
    n_teacher = anchor.config.encoder_config.num_hidden_layers
    t_ffn = anchor.config.encoder_config.intermediate_size
    layers = CS.resolve_layers(args.enc_layers, n_teacher)
    name = args.name or f"b{len(layers)}x{args.ffn}"
    print(f"  anchor {CS.param_counts(anchor)['total']:,} params; student {name}: layers {layers}, FFN {args.ffn}")
    dur["load"] = round(time.time() - t, 1)
    feats = CS.ctc_features(args.model_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)

    meta = dict(format=1, stage="building", name=name, family="ctc", init_class=None, bn="teacher",
                teacher=f"{pk.NEMO_REPO}@{pk.NEMO_REVISION}", seed=args.seed, enc_layers=layers, ffn=args.ffn,
                model_files_sha256=dict(pk.PARAKEET_FILES), features=FEATURES,
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                git=git_info(), host=host_info(), timestamps=dict(started=now()), durations_s=dur)

    t = time.time()
    if args.ffn < t_ffn:
        importance, imp_info, calib, calib_ids = load_or_compute_importance(anchor, feats, args)
    else:
        importance, imp_info, calib = None, dict(skipped="ffn == the teacher's width: nothing to prune"), None
        calib_ids = None
    dur["importance"] = round(time.time() - t, 1)

    t = time.time()
    print("[4/7] build the student")
    student = CS.build_ctc_student(anchor, layers, args.ffn, importance)
    counts = CS.param_counts(student)
    closed = counts["closed_form"]  # from the config's fields; at the real dims 5,911,553 + L (8,422,400 + 4,098 F)
    print(f"  {counts['total']:,} params (closed form {closed:,}), non-embedding {counts['non_embedding']:,}")
    assert counts["total"] == closed, counts
    if anchor.config.encoder_config.hidden_size == 1024:
        assert closed == CS.closed_form_ctc_params(len(layers), args.ffn), (closed, len(layers), args.ffn)
    keep = S.select_ffn_neurons(importance, layers, args.ffn, t_ffn)
    meta.update(params_total=counts["total"], params_non_embedding=counts["non_embedding"], closed_form_params=closed,
                params=counts,
                kept=dict(enc_layers=layers, ffn={f"{l}.{n}": idx for (l, n), idx in sorted(keep.items())}),
                importance=dict(imp_info, **(S.importance_summary(importance, keep) if keep else {})),
                calibration=calib)
    dur["build"] = round(time.time() - t, 1)

    gate = teacher_out = None
    if args.gate_utts > 0:
        t = time.time()
        print(f"[5/7] step-0 gate: {args.gate_utts} utts of <= {args.gate_max_s:.0f} s from each of {args.gate_sets}")
        gate = gate_sample(args.data, args.gate_sets, args.gate_utts, args.gate_seed, args.gate_max_s)
        waves = [u["wave"] for u in gate]
        teacher_out = run_ctc(anchor, feats, waves)
        built = gate_metrics(run_ctc(student, feats, waves), teacher_out, gate, tokenizer)
        print(f"  built (fp32): CER vs teacher {built['cer_vs_teacher']:.4f} (corpus "
              f"{built['cer_vs_teacher_corpus']:.4f}), frame KL {built['frame_kl']:.4f}, max |dlogp| "
              f"{built['max_abs_diff_log_probs']:.3g}")
        meta["step0"] = dict(rule=f"init_class = pruned_lost if the saved student's mean per-utterance CER vs the "
                                  f"teacher's greedy CTC >= {LOST_CER} else pruned_kept",
                             gate=dict(sets=args.gate_sets, n_per_set=args.gate_utts, seed=args.gate_seed,
                                       max_s=args.gate_max_s, ids=[u["id"] for u in gate]),
                             built=built)
        dur["gate_built"] = round(time.time() - t, 1)
    if args.golden:
        t = time.time()
        meta["golden"] = dict(built=golden_check(student, feats, tokenizer, args.data))
        print(f"  golden, built (fp32): {meta['golden']['built']['match']}/{meta['golden']['built']['n']} CTC "
              f"transcripts equal")
        dur["golden_built"] = round(time.time() - t, 1)
    if layers == list(range(n_teacher)) and args.ffn == t_ffn:
        # the unpruned build IS the reproduction check: a failed one must not end as a "complete" student dir
        problems = anchor_problems(student, anchor, meta)
        meta["anchor_check"] = dict(ok=not problems, problems=problems)
        if problems:
            meta.update(stage="failed", error=dict(error="the unpruned build does not reproduce the teacher",
                                                   problems=problems))
            meta["timestamps"]["failed"] = now()
            out.mkdir(parents=True, exist_ok=True)
            S.write_meta(out, meta)
            sys.exit("the unpruned build does not reproduce the teacher (nothing saved):\n  " + "\n  ".join(problems))
        print("  anchor check: the unpruned build reproduces the teacher")

    t = time.time()
    print(f"[6/7] save -> {out}")
    del anchor
    gc.collect()
    meta["stage"] = "saved"
    meta["timestamps"]["saved"] = now()
    CS.save_ctc_student(student, out, args.model_dir, meta)
    if calib_ids:  # as 03 writes it: the ids this student's FFNs were ranked on, next to the weights ranked on them
        build_script().write_calibration_ids(out, [{"id": x} for x in calib_ids])
    del student
    gc.collect()
    saved = CS.load_ctc_student(out, "cpu")
    dur["save"] = round(time.time() - t, 1)

    try:
        if gate is not None:
            t = time.time()
            res = gate_metrics(run_ctc(saved, feats, [u["wave"] for u in gate]), teacher_out, gate, tokenizer)
            meta["step0"]["saved"] = res
            meta["init_class"] = "pruned_lost" if res["cer_vs_teacher"] >= LOST_CER else "pruned_kept"
            print(f"  saved (bf16 weights): CER vs teacher {res['cer_vs_teacher']:.4f} (corpus "
                  f"{res['cer_vs_teacher_corpus']:.4f}), vs ref {res['cer_vs_ref']:.4f} (teacher "
                  f"{res['teacher_cer_vs_ref']:.4f}), frame KL {res['frame_kl']:.4f}, KD objective "
                  f"{res['kd_objective']:.4f}, argmax agree {res['argmax_agree']:.3f}, empty {res['empty_hyps']} "
                  f"-> {meta['init_class']}")
            dur["gate_saved"] = round(time.time() - t, 1)
        else:
            meta["init_class"] = args.init_class
            meta["step0"] = dict(skipped=True)
        if args.golden:  # bf16 storage may move a close argmax (measured: one comma in 32 on the anchor)
            t = time.time()
            meta["golden"]["saved"] = golden_check(saved, feats, tokenizer, args.data)
            print(f"  golden, saved (bf16 weights): {meta['golden']['saved']['match']}/{meta['golden']['saved']['n']} "
                  f"CTC transcripts equal")
            dur["golden_saved"] = round(time.time() - t, 1)
    except Exception as e:
        meta["error"] = dict(error=repr(e), traceback=traceback.format_exc())
        S.write_meta(out, meta)  # stage "saved": the weights are there; a re-run with --force redoes everything
        raise
    meta["stage"] = "complete"
    meta["timestamps"]["finished"] = now()
    dur["total"] = round(time.time() - t_start, 1)
    S.write_meta(out, meta)
    print(f"[7/7] done: {out} ({meta['params_total']:,} params, {meta['init_class']}) in {dur['total']:.0f} s")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.exit(main())
