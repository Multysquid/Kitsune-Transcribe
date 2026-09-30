"""Score a Whisper model on the study's frozen eval manifest as the students are scored (the full-data runs' Whisper
yardsticks, plan v3 decision 24; the build contract, section 9): the greedy decode of kitsune/whisper.py on the eval
store's audio, scored against its references, then - with --tables - the study tables and study.json of
scripts/05_evaluate.py --from-evals, in process, so one queue item is one command.

Flow:
  1. The store (--store: a built token store, <cache_dir>/eval; a frame store is refused) and, with --manifest, the
     frozen manifest (05's load_manifest: its sets hash to their sha256, and to PREREG.json's once filled): every set to
     score must hold exactly its manifest ids in the store. Any refusal exits 2 before a model is loaded.
  2. --out's identity (05's check_identity: the system, the model's repo / revision / dir, dtype, fp32 head, device,
     max_rows, batch_s, the store's fingerprint, the manifest's sha256, --limit-per-set and --seed, transformers): a run
     with other settings refuses (exit 2): use a new --out.
  3. The model: a WHISPER_MODELS key is fetched at its pinned revision into --hf-cache (only its allowed files), a
     model dir is used as it is; loaded under a heartbeat (kitsune.heartbeat.beating, bounded: LOAD_BEAT_MAX_S).
  4. Per set, resumable at set granularity (a set whose greedy_<set>.parquet and .parts/<set>.json agree is skipped):
     whisper_batches (the eval's batches, row-capped at --max-rows), the audio through kitsune.evaluate._prefetched,
     features, greedy_whisper, decode_texts; the thermal guard (05's, --max-temp) and a heartbeat beat before every
     batch. --force decodes the named sets again, as 05's does: its first run deletes their outputs and records what it
     forces (.parts/force.json), so the same forced command after a stop (a stall kill, a new host) finds the record,
     deletes nothing and decodes only the sets not done since; the record goes once every set it names is done (delete
     it to start a stopped forced run over).
  5. whisper.json after every set (status running, then complete; failed with the error when it stops), then with
     --tables (never with --limit-per-set, below) 05 --from-evals on --out, over every manifest set done in --out (not
     only the sets this command named: a partial re-run, e.g. --force --sets eval_cv8, must not drop the other sets'
     tables or M4 from study.json; the identity check makes every set in --out the same model and settings).

Outputs in --out:
  greedy_<set>.parquet  one row per decoded manifest row, in batch order: id, source, duration, ref (the store's), hyp,
                        truncated (stop != eos), stop (eos / repetition / length), n_tok (the generated ids incl. EOS,
                        as the students' greedy parquets count them), n_timestamp_tokens, max_new, batch. No
                        teacher_hyp: a Whisper model has no teacher (05 then writes no teacher table)
  whisper.json          schema 1: system, family "whisper", model{repo, revision, path, dtype, attn, fp32_head,
                        params_total, weights_file_bytes, n_mels, licence, display}, decode{...}, batching{batch_s,
                        max_rows, pooled_max_rows_seen, n_batches, n_split}, store{path, fingerprint}, manifest{path,
                        sha256} | null, limit_per_set, seed, sets{<set>{n, dropped, cer_corpus, n_truncated,
                        n_repetition, n_length, n_empty_hyp, n_timestamp_rows, audio_s, wall_s, rtf, n_batches,
                        n_split, pooled_max_rows}}, versions{python, torch, transformers, cuda, host, machine_id, cpu,
                        gpu}, status, time_utc
  events.jsonl          one JSON line per event (so the queue and finish see a run dir); .parts/ the resume state
  with --tables         05's study.json (family whisper, teacher null, the whisper.json model + decode blocks) and
                        <tables>/<system>/<set>.parquet

--limit-per-set N (smoke B's quick rows): N ids per set, speed_probe.pick_ids's seeded draw (--seed). Such a run
scores part of each set, so it cannot be tabled (05 --from-evals refuses a set that is not its manifest ids): a
--tables given with it is skipped, with a note on stderr and a tables_skipped event, and no study.json or tables are
written. It is not refused, because smoke B's turbo, kotoba and small items are the box's Whisper argv template
(contract 7, which carries --tables {out}/tables) plus --limit-per-set 50: a refusal would fail those items at once.
--manifest is optional with it (given, the sets are still checked against it). --fetch-only: only step 3's download,
on CPU (a prefetch; nothing else is read or written).

Exit codes: 0 done; 2 refused (a store whose rows are not the manifest's, a frame store, the manifest, identity, tables
refused by 05, bad arguments incl. a --model that is neither a key nor a model dir); 1 any other error, including a
--store that is not built yet (retryable: the queue's stores item may still be building it). Never 3 (the queue's
throughput code).

Usage (box 2's registry eval item; smoke B's large-v3 item adds --sets eval_jsut):
  python tools/whisper_eval.py --model whisper-large-v3 --store <root>/cache/eval \
      --manifest labels/full/selections/study_manifest.json --out runs/whisper-large-v3-<stamp> \
      --tables runs/whisper-large-v3-<stamp>/tables --hf-cache <root>/cache/hf --device cuda --max-temp 0
smoke B's quick rows (turbo, kotoba, small): the same plus --limit-per-set 50 (its --tables is skipped; --manifest
  is optional)
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for _p in (str(ROOT), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from kitsune import heartbeat  # noqa: E402
from kitsune import trainset  # noqa: E402
from kitsune import whisper as W  # noqa: E402

EXIT_REFUSED = 2
PARTS = ".parts"
FORCE = "force.json"  # in .parts: the sets an unfinished --force run decodes again (05's start_force record)
SCHEMA = 1
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
COLUMNS = ("id", "source", "duration", "ref", "hyp", "truncated", "stop", "n_tok", "n_timestamp_tokens", "max_new",
           "batch")


class Refused(Exception):
    """A refusal: exit 2, nothing decoded after it."""


_M05 = None


def m05():
    """scripts/05_evaluate.py as a module (its name starts with a digit): the manifest, identity, thermal guard and
    durable-write helpers, and --from-evals for the tables."""
    global _M05
    if _M05 is None:
        spec = importlib.util.spec_from_file_location("kitsune_evaluate_for_whisper",
                                                      ROOT / "scripts" / "05_evaluate.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _M05 = mod
    return _M05


def _refusing(fn, *a, **k):
    """fn(*a, **k) with its SystemExit (05's helpers refuse that way) turned into a Refused."""
    try:
        return fn(*a, **k)
    except SystemExit as e:
        raise Refused(str(e.code)) from e


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help=f"a Whisper key ({', '.join(W.WHISPER_MODELS)}: fetched at its pinned revision) or a "
                         "model dir")
    ap.add_argument("--system", default=None, help="the system name (default: the key; a model dir needs it)")
    ap.add_argument("--store", default=None,
                    help="the eval TOKEN store (<cache_dir>/eval), a built kitsune.trainset dir")
    ap.add_argument("--manifest", default=None,
                    help="the study's frozen eval manifest (study_manifest.json); required except with "
                         "--limit-per-set or --fetch-only")
    ap.add_argument("--prereg", default=str(ROOT / "study" / "PREREG.json"),
                    help="PREREG.json whose manifest block the manifest is compared with once filled (default "
                         "%(default)s)")
    ap.add_argument("--out", default=None, help="the output dir (resumable per set)")
    ap.add_argument("--sets", nargs="+", default=None,
                    help="the sets to score (default: every manifest set, or every set of the store without one)")
    ap.add_argument("--limit-per-set", type=int, default=None,
                    help="only this many seeded ids per set (smoke rows; speed_probe.pick_ids; a --tables given with "
                         "it is skipped with a note)")
    ap.add_argument("--seed", type=int, default=1234, help="the --limit-per-set draw's seed (default %(default)s)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="the row cap of a batch (default: the model's, kitsune.whisper.WHISPER_MODELS; "
                         f"{W.DEFAULT_MAX_ROWS} for a model dir)")
    ap.add_argument("--batch-s", type=float, default=400.0,
                    help="padded audio seconds of the eval batches before the row cap (default %(default)s, the "
                         "students' eval.batch_s)")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cpu, cuda or cuda:N")
    ap.add_argument("--dtype", default="auto", choices=("auto", *DTYPES),
                    help="auto = bf16 on CUDA, fp32 on CPU")
    ap.add_argument("--no-fp32-head", action="store_true",
                    help="the LM head in the model's dtype (default: fp32, the AED students' rule)")
    ap.add_argument("--hf-cache", default=None, help="the HF cache the pinned snapshot goes to (box: <root>/cache/hf)")
    ap.add_argument("--fetch-only", action="store_true", help="download the pinned snapshot and stop (CPU)")
    ap.add_argument("--tables", default=None,
                    help="also 05 --from-evals: <tables>/<system>/<set>.parquet and study.json in --out")
    ap.add_argument("--force", action="store_true", help="decode the named sets again")
    ap.add_argument("--max-temp", type=float, default=80.0,
                    help="pause before a batch at this GPU temperature (C); 0: no thermal guard (default %(default)s)")
    ap.add_argument("--resume-temp", type=float, default=70.0, help="resume at this temperature (default %(default)s)")
    ap.add_argument("--poll-s", type=float, default=10.0,
                    help="seconds between reads while paused (default %(default)s)")
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]
    args.tables_skipped = None
    if args.system is None:
        if args.model in W.WHISPER_MODELS:
            args.system = args.model
        elif not args.fetch_only:
            ap.error("--model is a model dir: name its system with --system")
    if args.fetch_only:
        if args.model not in W.WHISPER_MODELS:
            ap.error("--fetch-only downloads a Whisper key's pinned snapshot: a model dir needs no fetch")
        return args
    for flag, v in (("--store", args.store), ("--out", args.out)):
        if not v:
            ap.error(f"{flag} is required")
    # before --out's identity is written: a typo there would spoil the out dir for the corrected command
    if args.model not in W.WHISPER_MODELS and not (Path(args.model) / "config.json").is_file():
        ap.error(f"--model {args.model!r} is neither a Whisper key ({', '.join(W.WHISPER_MODELS)}) nor a model dir")
    if args.limit_per_set is not None and args.tables:
        # a limited run scores part of each set, which 05 --from-evals cannot table (a set must be its manifest ids).
        # Skipped, not refused: smoke B's quick items are the box's Whisper template (it carries --tables) plus
        # --limit-per-set, and an argument error would fail them at once. run() logs the note.
        args.tables_skipped, args.tables = args.tables, None
    if args.limit_per_set is None and not args.manifest:
        ap.error("--manifest is required (the frozen study manifest the tables are checked against)")
    if args.limit_per_set is not None and args.limit_per_set < 1:
        ap.error("--limit-per-set must be >= 1")
    if args.max_rows is not None and args.max_rows < 1:
        ap.error("--max-rows must be >= 1")
    if args.batch_s <= 0:
        ap.error("--batch-s must be > 0")
    if args.max_temp and not args.resume_temp < args.max_temp:
        ap.error("--resume-temp must be below --max-temp")
    return args


# ------------------------------------------------------------------------------------------------ the plan


def plan_sets(args, store, M) -> tuple[list[str], dict[str, list[str]]]:
    """The sets to score and, per set, the ids to decode (manifest order when there is a manifest, else the store's).
    Refuses a set that is not in the manifest or not in the store, or whose store rows are not its manifest ids."""
    man = M["manifest"] if M else None
    in_store: dict[str, list[str]] = {}
    for u in store.utts:
        in_store.setdefault(u.source, []).append(u.id)
    sets = list(dict.fromkeys(args.sets or (man.sets if man else in_store)))
    bad = []
    for s in sets:
        if man is not None and s not in man.sets:
            bad.append(f"{s}: not a set of the manifest ({sorted(man.sets)})")
        elif s not in in_store:
            bad.append(f"{s}: no rows in the store {args.store}")
        elif man is not None and (p := m05()._id_problems(s, in_store[s], man.sets[s])):
            bad.append(p)
    if bad:
        lost = ((store.info.get("dropped") or {}).get("no_audio") or {}).get("ids") or []
        raise Refused("the eval store does not hold the sets as the manifest registers them: " + "; ".join(bad)
                      + (f" (rows without audio: {lost[:5]})" if lost else ""))
    ids = {s: list(man.sets[s]) if man is not None else in_store[s] for s in sets}
    if args.limit_per_set is not None:
        from speed_probe import pick_ids

        picked = set(pick_ids(store, args.limit_per_set, args.seed))
        ids = {s: [i for i in v if i in picked] for s, v in ids.items()}
    return sets, ids


def set_done(out: Path, s: str) -> bool:
    """A set is done when its parquet and its parts record agree (a torn write is decoded again, never trusted)."""
    part = m05()._load_json(out / PARTS / f"{s}.json")
    return (isinstance(part, dict) and part.get("n") is not None
            and m05()._rows(out / f"greedy_{s}.parquet") == part["n"])


def start_force(out: Path, sets: list[str], log):
    """--force (05's start_force, per set): the first run deletes the named sets' outputs and records what it forces
    (.parts/force.json). The same forced command after a stop finds that record and deletes nothing, so it decodes only
    the sets not done since - a stall kill or a new host must not throw away hours of large-v3 (end_force drops it)."""
    M = m05()
    old = M._load_json(out / PARTS / FORCE)
    if isinstance(old, dict) and old.get("sets") == list(sets):
        log.event("force_continue", since=old.get("time_utc"), sets=list(sets))
        return
    for s in sets:
        for p in (out / f"greedy_{s}.parquet", out / PARTS / f"{s}.json"):
            p.unlink(missing_ok=True)
    M._write_json(out / PARTS / FORCE, dict(sets=list(sets), time_utc=M._now()))
    log.event("force", sets=list(sets))


def end_force(out: Path, log):
    """The --force record goes once every set it names is done, by the forced command or any other."""
    p = out / PARTS / FORCE
    if not p.exists():
        return
    rec = m05()._load_json(p)
    if isinstance(rec, dict) and not all(set_done(out, s) for s in rec.get("sets") or ()):
        return  # an unreadable record is of no use: it goes
    p.unlink(missing_ok=True)
    log.event("force_done", sets=(rec or {}).get("sets") if isinstance(rec, dict) else None)


def tabled_sets(out: Path, manifest_sets) -> list[str] | None:
    """The sets --tables hands 05 --from-evals: every manifest set done in --out, whatever this command named (None =
    all of them: 05's default). A partial re-run (--force --sets eval_cv8) would otherwise table eval_cv8 alone, and
    05's publish_tables would delete the other sets' tables and study.json lose M4."""
    done = [s for s in manifest_sets if set_done(out, s)]
    return None if len(done) == len(manifest_sets) else done


def set_summary(df: pd.DataFrame, dropped: list[str], wall_s: float, stats: dict) -> dict:
    """whisper.json's per-set block from a set's rows."""
    from kitsune.evaluate import corpus_cer
    from kitsune.text import normalize_ja

    audio = float(df["duration"].sum()) if len(df) else 0.0
    c = corpus_cer(df["hyp"].tolist(), df["ref"].tolist())
    return dict(n=int(len(df)), dropped=list(dropped), cer_corpus=c["cer"], edits=c["edits"], ref_chars=c["ref_chars"],
                n_empty_ref=c["n_empty_ref"], n_truncated=int(df["truncated"].sum()),
                n_repetition=int((df["stop"] == "repetition").sum()), n_length=int((df["stop"] == "length").sum()),
                n_empty_hyp=int((df["hyp"].map(lambda h: normalize_ja(h or "")) == "").sum()),
                n_timestamp_rows=int((df["n_timestamp_tokens"] > 0).sum()), audio_s=round(audio, 3),
                wall_s=round(wall_s, 3), rtf=wall_s / audio if audio else None, **stats)


def decode_set(args, wm, store, s: str, ids: list[str], max_rows: int, fp32_head: bool, guard, log) -> tuple[
        pd.DataFrame, list[str], float, dict]:
    """One set's greedy decode in its row-capped eval batches: (rows in batch order, undecodable ids, wall s, batching
    stats). A heartbeat beat and the thermal guard before every batch."""
    from kitsune.evaluate import _prefetched

    pos = {u.id: i for i, u in enumerate(store.utts)}
    idx = [pos[i] for i in ids]
    pooled = trainset.eval_batches(store.utts, args.batch_s, idx)
    batches = W.split_rows(pooled, max_rows)
    ref = dict(zip((u.id for u in store.utts), store.frame()["ref"].tolist()))
    recs, dropped = [], []
    t0 = time.time()
    for b, item in enumerate(_prefetched(trainset.AudioBatchDataset(store), batches)):
        heartbeat.beat()
        if guard is not None:
            guard.check()
        dropped += item["dropped"]
        n = len(item["ids"])
        if not n:
            continue
        lengths = item["lengths"].tolist()
        waves = [item["wave"][r, :lengths[r]].numpy() for r in range(n)]
        durations = [float(x) for x in item["durations"].tolist()]
        try:
            feats = W.features(wm, waves)
            rows = W.greedy_whisper(wm, feats, max(durations), fp32_head=fp32_head)
        except torch.OutOfMemoryError as e:
            raise RuntimeError(f"out of memory on a batch of {n} rows ({max(durations):.1f} s): score with a smaller "
                               "--max-rows into a new --out") from e
        texts = W.decode_texts(wm, rows)
        for r, (row, text) in enumerate(zip(rows, texts)):
            uid = item["ids"][r]
            recs.append(dict(id=uid, source=item["sources"][r], duration=durations[r], ref=ref[uid], hyp=text,
                             truncated=bool(row["truncated"]), stop=row["stop"], n_tok=len(row["hyp_ids"]),
                             n_timestamp_tokens=int(row["n_timestamp_tokens"]), max_new=int(row["max_new"]), batch=b))
    wall = time.time() - t0
    df = pd.DataFrame(recs, columns=list(COLUMNS))
    stats = dict(n_batches=len(batches), n_split=len(batches) - len(pooled),
                 pooled_max_rows=max((len(p) for p in pooled), default=0))
    log.event("whisper_set", set=s, n=len(df), dropped=dropped, wall_s=round(wall, 2), **stats)
    return df, dropped, wall, stats


# ------------------------------------------------------------------------------------------------ the record


def versions(device: torch.device) -> dict:
    """The software, host, machine and GPU a record comes from (speed_probe's _versions, plus the GPU)."""
    from speed_probe import _versions

    v = _versions()
    v.update(gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    return v


def whisper_record(args, out: Path, meta: dict, status: str, **extra) -> dict:
    """whisper.json from `meta` (the run's model / decode / store / manifest blocks) and every set done in --out."""
    M = m05()
    sets, batching = {}, dict(batch_s=float(args.batch_s), max_rows=meta["max_rows"], pooled_max_rows_seen=0,
                              n_batches=0, n_split=0)
    for s in meta["sets_order"]:
        part = M._load_json(out / PARTS / f"{s}.json")
        if isinstance(part, dict) and set_done(out, s):
            sets[s] = part["summary"]
            batching["pooled_max_rows_seen"] = max(batching["pooled_max_rows_seen"], part["summary"]["pooled_max_rows"])
            batching["n_batches"] += part["summary"]["n_batches"]
            batching["n_split"] += part["summary"]["n_split"]
    return dict(schema=SCHEMA, system=args.system, family=W.FAMILY, model=meta["model"], decode=meta["decode"],
                batching=batching, store=meta["store"], manifest=meta["manifest"], limit_per_set=args.limit_per_set,
                seed=args.seed, sets=sets, versions=meta["versions"], status=status, time_utc=M._now(), **extra)


def run(args) -> int:
    M = m05()
    if args.fetch_only:
        with heartbeat.beating(max_s=W.LOAD_BEAT_MAX_S):
            path, spec = W.resolve(args.model, args.hf_cache)
        print(f"{spec.repo}@{spec.revision} -> {path}", flush=True)
        return 0
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / PARTS).mkdir(exist_ok=True)
    log = M.EvalLog(out / "events.jsonl")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dname = ("bf16" if device.type == "cuda" else "fp32") if args.dtype == "auto" else args.dtype
    fp32_head = not args.no_fp32_head
    spec = W.WHISPER_MODELS.get(args.model)
    max_rows = int(args.max_rows or (spec.max_rows if spec else W.DEFAULT_MAX_ROWS))
    log.event("whisper_start", argv=args.argv, model=args.model, system=args.system, out=str(out),
              device=str(device), dtype=dname, max_rows=max_rows, batch_s=args.batch_s,
              limit_per_set=args.limit_per_set)
    if args.tables_skipped:
        print(f"NOTE: --tables {args.tables_skipped} skipped: --limit-per-set {args.limit_per_set} scores part of each "
              "set, which 05's tables cannot hold (no study.json or tables are written)", file=sys.stderr, flush=True)
        log.event("tables_skipped", tables=str(args.tables_skipped), limit_per_set=args.limit_per_set)
    try:
        store = trainset.load_stores(args.store)
    except FileNotFoundError as e:  # exit 1, not a refusal: the box's stores item may not have built it yet
        raise SystemExit(f"the eval store {args.store} is not built ({e.filename or e}): build it first (exit 1, "
                         "retryable)") from e
    if (store.info or {}).get("kind") == "frames":
        raise Refused(f"{args.store} is a frame store (the CTC trainer's): Whisper reads the eval TOKEN store "
                      "(<cache_dir>/eval), whose references every system is scored against")
    Mf = _refusing(M.load_manifest, Path(args.manifest).resolve(), args.prereg, log) if args.manifest else None
    sets, ids = plan_sets(args, store, Mf)
    import transformers

    identity = dict(system=args.system, repo=spec.repo if spec else None, revision=spec.revision if spec else None,
                    model_dir=None if spec else str(Path(args.model).resolve()), dtype=dname, fp32_head=fp32_head,
                    device=device.type, max_rows=max_rows, batch_s=float(args.batch_s),
                    store=(store.info or {}).get("fingerprint"), manifest=Mf["sha256"] if Mf else None,
                    limit_per_set=args.limit_per_set, seed=int(args.seed), transformers=transformers.__version__)
    _refusing(M.check_identity, out, identity)
    if args.force:
        start_force(out, sets, log)
    todo = [s for s in sets if not set_done(out, s)]
    log.event("todo", sets=todo, skipped_done=[s for s in sets if s not in todo])
    old = M._load_json(out / W.WHISPER_META)
    meta = dict(max_rows=max_rows, sets_order=list(dict.fromkeys([*sets, *((old or {}).get("sets") or {})])),
                store=dict(path=str(Path(args.store).resolve()), fingerprint=(store.info or {}).get("fingerprint")),
                manifest=dict(path=Mf["path"], sha256=Mf["sha256"]) if Mf else None, versions=versions(device),
                model=(old or {}).get("model"), decode=(old or {}).get("decode"))
    wm = None
    try:
        if todo:
            with heartbeat.beating(max_s=W.LOAD_BEAT_MAX_S):
                path, spec = W.resolve(args.model, args.hf_cache)
                wm = W.load_whisper(path, device, DTYPES[dname], spec=spec)
            meta.update(model=W.model_record(wm, fp32_head), decode=W.decode_record(wm))
            log.event("whisper_model", **meta["model"], prompt_ids=wm.prompt_ids)
            guard = M.make_guard(device, args, log)
            for s in todo:
                df, dropped, wall, stats = decode_set(args, wm, store, s, ids[s], max_rows, fp32_head, guard, log)
                M._table(df, out / f"greedy_{s}.parquet")
                M._write_json(out / PARTS / f"{s}.json", dict(n=int(len(df)), summary=set_summary(df, dropped, wall,
                                                                                                    stats)))
                M._write_output_json(out / W.WHISPER_META, whisper_record(args, out, meta, "running"))
        end_force(out, log)
        rec = whisper_record(args, out, meta, "complete")
        M._write_output_json(out / W.WHISPER_META, rec)
        log.event("whisper_done", sets={s: v["cer_corpus"] for s, v in rec["sets"].items()})
    except BaseException as e:
        try:
            M._write_output_json(out / W.WHISPER_META, whisper_record(args, out, meta, "failed",
                                                                      error=f"{type(e).__name__}: {e}"[:2000]))
            log.event("whisper_failed", error=f"{type(e).__name__}: {e}"[:2000])
        except Exception as e2:  # noqa: BLE001 - never masks the run's own error
            print(f"could not record the failure: {e2!r}", file=sys.stderr)
        raise
    if args.tables:
        argv = ["--from-evals", str(out), "--system", args.system, "--manifest", str(Path(args.manifest).resolve()),
                "--prereg", str(args.prereg), "--out", str(out), "--tables", str(Path(args.tables).resolve())]
        if (done := tabled_sets(out, list(Mf["manifest"].sets))) is not None:
            argv += ["--sets", *done]
        rc = _refusing(M.main, argv)
        if rc:
            raise Refused(f"05 --from-evals exited {rc}")
    return 0


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0) if isinstance(e.code, int) else EXIT_REFUSED
    try:
        return run(args)
    except Refused as e:
        print(f"REFUSED: {str(e).removeprefix('REFUSED: ')}", file=sys.stderr)
        return EXIT_REFUSED
    except SystemExit as e:  # anything else that exits (never the queue's 3: that is a trainer's throughput halt)
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
        return 1 if code == 3 else code


if __name__ == "__main__":
    sys.exit(main())
