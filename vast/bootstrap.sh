#!/bin/bash
# Put the training data on the vast box (run by vast/onstart.sh from the cloned repo, cwd-independent).
#
# Why this split: home upload is slow, so only the small DERIVED data is parked in the private HF dataset
# $KITSUNE_DATA_REPO (teacher_out for the train sources and eval sets, second_out, the selection parquet, the student
# init; ~2.3 GB). The ~23 GB of audio (reazon_small ~7, Emilia-YODAS 300 h ~8, Galgame's 6 tars ~5, eval sets ~3;
# sized in vast/launch.py's DISK_GB comment) is REBUILT here from the original public HF datasets by
# scripts/01_prepare_data.py, which reads every upstream repo at a pinned commit, so the utterance ids match the
# teacher outputs exactly. If the data repo also holds data/shards/<source>/*.parquet for a source, those parked
# shards are pulled (and listed in data/manifest.jsonl, which 01 reads) instead of rebuilding that source. The trainer
# joins everything by utterance id and drops (and logs) rows without audio; this script already fails if coverage is
# below KITSUNE_MIN_COVERAGE, because a broken join would waste the whole paid run.
#
# The repo layout mirrors the laptop's repo root: <teacher_root>/..., <second_root>/..., <selection>, <student>/...,
# optionally <data_root>/shards/... (paths from the run config). Idempotent: snapshot_download and 01 both skip what is
# already on disk. Phase timings go to $KITSUNE_STATE/bootstrap_timings.jsonl.
#
# The plan phase also checks that HF_TOKEN can read and write $KITSUNE_OUT_REPO: launch.py checks the repos with the
# laptop's own login, and the trainer's first upload comes only after the pull, the audio rebuild and the model load.
#
# Env: KITSUNE_DATA_REPO and KITSUNE_OUT_REPO (required), KITSUNE_DATA_REVISION (default main), KITSUNE_CONFIG (default
# configs/viability.json), KITSUNE_PREP_ARGS (extra args for 01; leave it empty for the viability data: its teacher
# outputs cover exactly the first 6 Galgame tars and 300 h of Emilia-YODAS, which are 01's defaults, and a larger
# --galgame-shards/--emilia-hours only downloads audio without teacher output), KITSUNE_MIN_COVERAGE (default 0.99),
# HF_TOKEN (vast account env; never printed), KITSUNE_REBUILD_TIMEOUT_MIN (extent mode only: the per-attempt timeout
# of the audio rebuild in minutes, default 60; vast/launch.py sizes it from the extent record's upstream bytes).
#
# Extent mode (a run config with an "extent" block, labels from the label box under <extent root>/, e.g. labels/full):
# the plan requires <root>/COMPLETE.json and <root>/extent.json in the listing (else it refuses, exit 3), downloads the
# record into $KITSUNE_STATE and asks kitsune.extent.pull_plan for exactly the extent's label files: directory globs for
# an uncapped source, explicit <stem>.npz/.jsonl files for a capped one (a prefix of a big source), parakeet_out only
# for a CTC student or with pull_parakeet (the size study pulls both label roots).
#
# Study box (KITSUNE_JOB=study, KITSUNE_BOX=A|B|replicate|shakedown; KITSUNE_CONFIG is study/data.json, the data block,
# which names no student): the students pulled are the box's own (kitsune.study_queue.box_students), each with
# STUDENT_FILES and, for a Parakeet-derived one, its CC-BY-4.0 MODEL_CARD.md; plus box_extra_dirs (the Parakeet
# teacher for the speed probes).
# The rebuild is `01 --extent-config $KITSUNE_CONFIG`, the canonical ingest sequence the label box ran, so the ids and
# stems are the labelled ones (KITSUNE_PREP_ARGS is ignored), and coverage is exact: every pulled stem's rebuilt id
# sidecar hashes to the record's ids_sha256, its teacher ids are a subset of its ids, and every split joins at 1.0.
set -euo pipefail

KITSUNE_DIR="${KITSUNE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STATE="${KITSUNE_STATE:-/workspace/kitsune_state}"
CONFIG="${KITSUNE_CONFIG:-configs/viability.json}"
PY="${KITSUNE_PY:-/venv/main/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
export KITSUNE_DIR CONFIG STATE
export KITSUNE_DATA_REVISION="${KITSUNE_DATA_REVISION:-main}"
export KITSUNE_MIN_COVERAGE="${KITSUNE_MIN_COVERAGE:-0.99}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-30}"  # 01's progress bars go to a log file, not a terminal

log() { printf '%s [bootstrap] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

if [ -z "${KITSUNE_DATA_REPO:-}" ]; then
    log "KITSUNE_DATA_REPO is not set"
    exit 2
fi
if [ -z "${KITSUNE_OUT_REPO:-}" ]; then
    log "KITSUNE_OUT_REPO is not set (vast/launch.py passes it)"
    exit 2
fi
if [ -z "${HF_TOKEN:-}" ]; then
    log "HF_TOKEN is not set: add it under vast Account -> Settings -> Environment Variables (see vast/README.md)"
    exit 2
fi
cd "$KITSUNE_DIR"
mkdir -p "$STATE"
TIMINGS="$STATE/bootstrap_timings.jsonl"
HELPER="$(mktemp --suffix=.py)"
trap 'rm -f "$HELPER"' EXIT

phase() {  # phase <name> <command...>: run it and append its wall time
    local name=$1 t0 t1
    shift
    t0=$(date +%s.%N)
    log "phase $name ..."
    "$@"
    t1=$(date +%s.%N)
    printf '{"phase": "%s", "seconds": %s, "end": %s}\n' "$name" \
        "$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')" "${t1%.*}" >> "$TIMINGS"
    log "phase $name done in $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }') s"
}

# retry <tries> <command...>: re-run a resumable network step after a failure. One Hub 5xx/429 or dropped connection
# (01's listing calls are sent once, with no HTTP timeout) would otherwise stop the box after the paid boot; a stalled
# call is cut by the timeout the caller puts in <command>. The exit code of the last attempt is returned. Exit 3 is a
# refusal a retry cannot fix (the plan helper's NO_RETRY: a token or data repo the Hub refused, data it lacks) and
# returns at once; neither 01_prepare_data.py (1, argparse 2) nor timeout (124-127, 137) exits 3.
retry() {
    local n=$1 i rc=0
    shift
    for (( i = 1; i <= n; i++ )); do
        rc=0
        "$@" || rc=$?
        [ "$rc" -eq 0 ] && return 0
        log "attempt $i/$n of $* failed (exit $rc)"
        if [ "$rc" -eq 3 ]; then log "exit 3 is a refusal a retry cannot fix; not retrying"; return 3; fi
        if [ "$i" -lt "$n" ]; then sleep $(( i * 60 )); fi
    done
    return "$rc"
}

cat > "$HELPER" <<'PYEOF'
"""bootstrap helper: plan / pull / coverage. Reads the run config for sources and paths."""
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

root = Path(os.environ["KITSUNE_DIR"])
state = Path(os.environ["STATE"])
cfg = json.loads((root / os.environ["CONFIG"]).read_text(encoding="utf-8"))
repo = os.environ["KITSUNE_DATA_REPO"]
rev = os.environ["KITSUNE_DATA_REVISION"]
sources = list(cfg.get("sources", []))
evals = list(cfg.get("eval_sets", []))
# a source can be a train source AND an eval set (galgame keeps its hold-out in its own eval split)
names = list(dict.fromkeys(sources + evals))
teacher_root = cfg.get("teacher_root", "teacher_out")
second_root = cfg.get("second_root", "second_out")
data_root = cfg.get("data_root", "data")
selection = cfg["selection"]
# the student dir files the trainer loads (it has no processor fallback; it reads student_meta.json, and README.md is
# the model card with the modification notice): the same list as vast/launch.py STUDENT_FILES
STUDENT_FILES = ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "student_meta.json", "README.md")
CTC_CARD = "MODEL_CARD.md"  # a Parakeet-derived student's CC-BY-4.0 attribution (kitsune.ctc_student)
if os.environ.get("KITSUNE_JOB") == "study":
    # a size-study box (kitsune/study_queue.py): the data block has no student; the box pulls the student dirs of its
    # own runs only (box_students), with the Parakeet students' attribution card, and any other dir it needs
    sys.path.insert(0, str(root))
    from kitsune import study_queue as _q
    _box = os.environ["KITSUNE_BOX"]
    students = _q.box_students(_box)
    ctc_students = set(_q.box_ctc_students(_box))
    extra_dirs = _q.box_extra_dirs(_box)
else:
    students, ctc_students, extra_dirs = [cfg["student"].rstrip("/")], set(), []
student = students[0] if students else ""


def student_files(s: str) -> list:
    return [f"{s}/{n}" for n in STUDENT_FILES + ((CTC_CARD,) if s in ctc_students else ())]
plan_path = state / "bootstrap_plan.json"
NO_RETRY = 3  # the exit code bootstrap's retry() does not repeat: the Hub would refuse again


def refused(e: BaseException) -> bool:
    """A refusal is the token's or the config's fault (404/401 = RepoNotFound: the Hub hides a private repo the token
    cannot see; 404 also for a missing revision); a 5xx, 429 or dropped connection is the Hub's, and the plan phase
    runs under retry()."""
    return getattr(getattr(e, "response", None), "status_code", None) in (401, 403, 404)


def refuse(msg: str):
    print(f"{msg} (not retried)", file=sys.stderr)
    sys.exit(NO_RETRY)


def plan():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as e:
        if refused(e):
            refuse(f"HF_TOKEN is not a valid token ({type(e).__name__}: {e}): put a working fine-grained token in the "
                   f"vast account env (vast/README.md step 2.3)")
        raise
    print(f"hf user: {user}")
    # the box token's first use of the output repo would otherwise be the trainer's hf_roundtrip, after the pull, the
    # audio rebuild and the model load; auth_check is a GET (no commit), and the trainer reads its uploads back
    out_repo = os.environ["KITSUNE_OUT_REPO"]
    for write in (False, True):
        try:
            api.auth_check(out_repo, repo_type="model", write=write)
        except Exception as e:
            if refused(e):
                refuse(f"HF_TOKEN cannot {'write' if write else 'read'} {out_repo} ({type(e).__name__}: {e}): give "
                       f"the fine-grained token read and write access to it (vast/README.md step 2.3)")
            sys.exit(f"Hub error checking {out_repo} ({type(e).__name__}: {e}); bootstrap retries the plan")
    try:
        files = api.list_repo_files(repo, repo_type="dataset", revision=rev)
    except Exception as e:
        if refused(e):
            refuse(f"HF_TOKEN cannot list {repo}@{rev} ({type(e).__name__}: {e}): check KITSUNE_DATA_REPO, "
                   f"KITSUNE_DATA_REVISION and the token's read access to it (vast/README.md step 2.3)")
        raise
    if cfg.get("extent"):
        return plan_extent(files)

    def has(pat: str) -> bool:
        return any(fnmatch.fnmatch(f, pat) for f in files)

    patterns = [f"{teacher_root}/meta.json", f"{second_root}/meta.json", selection, *(f"{s}/*" for s in students)]
    required = [f"{selection}", *(f for s in students for f in student_files(s))]
    for s in names:
        patterns += [f"{teacher_root}/{s}/*", f"{second_root}/{s}/*"]
        required.append(f"{teacher_root}/{s}/*.npz")
    required += [f"{second_root}/{s}/*.jsonl" for s in sources]
    missing = [p for p in required if not has(p)]
    # every teacher shard needs its second opinion (the same rule as vast/launch.py data_problems), except for the
    # sources the run config's selection_recipe.partial_second_opinion trains on their judged shards only
    partial = set((cfg.get("selection_recipe") or {}).get("partial_second_opinion", []))
    for s in sources:
        teacher = {f.rsplit("/", 1)[1][:-4] for f in files if f.startswith(f"{teacher_root}/{s}/") and f.endswith(".npz")}
        second = {f.rsplit("/", 1)[1][:-6] for f in files if f.startswith(f"{second_root}/{s}/") and f.endswith(".jsonl")}
        if s in partial:
            if not teacher & second:
                missing.append(f"{second_root}/{s}: none of its {len(teacher)} shards has a second opinion")
        elif teacher - second:
            missing.append(f"{second_root}/{s}: {len(teacher - second)} of {len(teacher)} shards without a second opinion")
    if missing:
        refuse(f"data repo {repo}@{rev} lacks: {missing}")
    parked = [s for s in names if has(f"{data_root}/shards/{s}/*.parquet")]
    patterns += [f"{data_root}/shards/{s}/*.parquet" for s in parked]
    rebuild = [s for s in names if s not in parked]
    # the files the pull must leave on disk, by snapshot_download's own matcher (pull() checks them)
    want = list(filter_repo_objects(files, allow_patterns=patterns))
    out = dict(repo=repo, revision=rev, patterns=patterns, parked=parked, rebuild=rebuild, data_root=data_root,
               repo_files=len(files), files=want, wall=time.time())
    plan_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"plan: pull {len(patterns)} patterns; parked audio {parked or 'none'}; rebuild audio {rebuild or 'none'}")


def plan_extent(files: list):
    """The config's extent, labelled by the label box under <root>/: pull exactly its label files (kitsune.extent
    pull_plan: directory globs for uncapped sources, explicit files for capped ones, never parakeet_out) and rebuild
    all of its audio with `01 --extent-config`. A root the label box has not sealed (no COMPLETE.json) is refused: its
    files may still change, and its extent.json is written only at the seal."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import filter_repo_objects

    sys.path.insert(0, str(root))  # this helper runs from a mktemp path
    from kitsune.extent import RECORD_FILE, load_record, names as extent_names, pull_plan

    eroot = cfg["extent"].get("root", "")
    lacks = [f"{eroot}/{n}" for n in ("COMPLETE.json", RECORD_FILE) if f"{eroot}/{n}" not in set(files)]
    if lacks:
        refuse(f"data repo {repo}@{rev} lacks {lacks}: the label box has not sealed the extent root {eroot!r}")
    try:
        path = hf_hub_download(repo, f"{eroot}/{RECORD_FILE}", repo_type="dataset", revision=rev, local_dir=state)
    except Exception as e:
        if refused(e):
            refuse(f"HF_TOKEN cannot read {repo}@{rev}/{eroot}/{RECORD_FILE} ({type(e).__name__}: {e})")
        raise
    p = pull_plan(cfg, load_record(path), files)
    problems = list(p["problems"])
    have = set(files)
    problems += [f"no {f}" for s in students for f in student_files(s) if f not in have]
    problems += [f"no {d}/*" for d in extra_dirs if not any(f.startswith(f"{d}/") for f in files)]
    if problems:
        refuse(f"data repo {repo}@{rev} cannot serve extent {cfg['extent'].get('name')!r}: {problems}")
    # a study box's students (pull_plan pulls the config's one student, and the data block has none) and other dirs
    p["dir_patterns"] += [f"{d}/*" for d in [*students, *extra_dirs] if f"{d}/*" not in p["dir_patterns"]]
    # the files the pull must leave on disk: the globs by snapshot_download's own matcher, plus the explicit files
    want = list(dict.fromkeys([*filter_repo_objects(files, allow_patterns=p["dir_patterns"]), *p["explicit"]]))
    rebuild = extent_names(cfg)
    out = dict(repo=repo, revision=rev, patterns=p["dir_patterns"], parked=[], rebuild=rebuild, data_root=data_root,
               repo_files=len(files), files=want, wall=time.time(), extent=True, explicit=p["explicit"],
               record=str(Path(path).resolve()))
    plan_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"plan: extent {cfg['extent'].get('name')!r}: pull {len(p['dir_patterns'])} patterns and "
          f"{len(p['explicit'])} explicit files; rebuild audio {rebuild} with 01 --extent-config")


def register_parked(parked: list):
    """List the pulled shards of the parked sources in <data_root>/manifest.jsonl, as a local ingest would have (the
    data repo holds no manifest). 01 reads other sources' rows from the manifest - eval_emilia leaves out emilia_yodas's
    videos ("ingest emilia_yodas first" otherwise), the reazon tiers dedup against the smaller ones - so an unlisted
    parked source would stop the rebuild. Paths already listed are skipped, so a re-run adds nothing."""
    if not parked:
        return
    import pyarrow.parquet as pq

    sys.path.insert(0, str(root))  # this helper runs from a mktemp path
    from kitsune.store import ShardInfo, append_manifest, read_manifest

    droot = root / data_root
    listed = {s.path for s in read_manifest(droot)}
    new = []
    for s in parked:
        for f in sorted((droot / "shards" / s).glob("*.parquet")):
            rel = f.relative_to(droot).as_posix()
            if rel not in listed:
                dur = pq.read_table(f, columns=["duration"]).column("duration").to_pylist()
                new.append(ShardInfo(rel, s, f.name.rsplit("-", 1)[0], len(dur), sum(dur) / 3600))
    if new:
        append_manifest(droot, new)
    print(f"manifest: listed {len(new)} parked shard(s) of {parked}")


def pull():
    from huggingface_hub import snapshot_download

    p = json.loads(plan_path.read_text(encoding="utf-8"))
    t0 = time.time()
    snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=root, allow_patterns=p["patterns"],
                      max_workers=16)
    if p.get("explicit"):
        # a capped source's files one by one (fnmatch over ~2k patterns x ~27k repo files would be too slow); a file
        # already on disk is complete (downloads land as .incomplete files renamed when done), so a retry skips it
        from concurrent.futures import ThreadPoolExecutor

        from huggingface_hub import hf_hub_download

        todo = [f for f in p["explicit"] if not (root / f).is_file()]
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda f: hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=root), todo))
        print(f"pulled {len(todo)} of {len(p['explicit'])} explicit files ({len(p['explicit']) - len(todo)} on disk)")
    # when its one repo_info request fails (a Hub 5xx/429, a dropped connection) snapshot_download returns local_dir
    # (the checkout, never empty) with only a warning and downloads nothing, and 01's paid audio rebuild would run
    # before the coverage check caught it: fail here instead, so the shell's retry pulls again
    missing = [f for f in p["files"] if not (root / f).is_file()]
    if missing:
        sys.exit(f"pull incomplete: {len(missing)} of {len(p['files'])} planned files missing, e.g. {missing[:3]}")
    register_parked(p["parked"])
    dirs = [root / d for d in (teacher_root, second_root, *students, f"{data_root}/shards") if (root / d).is_dir()]
    size = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
    print(f"pulled in {time.time() - t0:.0f} s; derived data + parked shards on disk: {size / 1e9:.2f} GB")


def extent_stems(report: dict) -> list:
    """Extent mode: every pulled stem (kitsune.extent subset_stems) must have been rebuilt with the labelled ids. The
    rebuilt id sidecar's ids must hash to the record's ids_sha256 (same ids, same order, same stem), and the teacher
    ids of the stem must be a subset of them. Returns the failures; report["extent_stems"] gets the counts."""
    import numpy as np

    sys.path.insert(0, str(root))  # this helper runs from a mktemp path
    from kitsune.extent import load_record, subset_stems
    from kitsune.store import ids_sha256, read_manifest, shard_ids

    p = json.loads(plan_path.read_text(encoding="utf-8"))
    record = load_record(Path(p["record"]))
    droot = root / data_root
    shards = {(s.source, Path(s.path).stem): s for s in read_manifest(droot)}
    bad, checked = [], 0
    for s, stems in sorted(subset_stems(record, cfg).items()):
        want = {st["stem"]: st["ids_sha256"] for inp in record["sources"].get(s, {}).get("inputs", [])
                for st in inp["stems"]}
        for stem in sorted(stems):
            checked += 1
            info = shards.get((s, stem))
            if info is None:
                bad.append(f"{s}/{stem}: not rebuilt (no manifest line)")
                continue
            try:
                ids = shard_ids(droot, info)
            except FileNotFoundError as e:
                bad.append(f"{s}/{stem}: {e}")
                continue
            if ids_sha256(ids) != want[stem]:
                bad.append(f"{s}/{stem}: rebuilt ids_sha256 {ids_sha256(ids)[:12]} != the record's {want[stem][:12]}")
                continue
            with np.load(root / teacher_root / s / f"{stem}.npz", allow_pickle=False) as z:
                extra = {str(i) for i in z["ids"]} - set(ids)
            if extra:
                bad.append(f"{s}/{stem}: {len(extra)} teacher ids not in the rebuilt stem, e.g. {sorted(extra)[:2]}")
    report["extent_stems"] = dict(checked=checked, failed=len(bad), failures=bad[:50])
    for b in bad[:20]:
        print(f"  extent: {b}")
    print(f"  extent: {checked - len(bad)} of {checked} pulled stems rebuilt with the labelled ids")
    return bad


def coverage():
    import numpy as np
    import pyarrow.parquet as pq

    floor = float(os.environ.get("KITSUNE_MIN_COVERAGE", "0.99"))
    report, bad = {}, []
    if cfg.get("extent"):  # the rebuild replayed the label box's ingest: short of an exact join, it is another extent
        floor = 1.0
        stems_bad = extent_stems(report)
        bad += [f"{len(stems_bad)} extent stem(s) (listed above)"] if stems_bad else []
    for s in names:
        # per split (<split>-NNNNN in both trees), as the trainer joins them: pooled over a source's splits,
        # galgame's 1,000-row hold-out is 0.5 % of its ids and could vanish (or trade rows with train) above the floor
        tids, aids = {}, {}
        for npz in sorted((root / teacher_root / s).glob("*.npz")):
            with np.load(npz, allow_pickle=False) as z:
                tids.setdefault(npz.stem.rsplit("-", 1)[0], set()).update(str(i) for i in z["ids"])
        for shard in sorted((root / data_root / "shards" / s).glob("*.parquet")):
            aids.setdefault(shard.stem.rsplit("-", 1)[0], set()).update(
                pq.read_table(shard, columns=["id"]).column("id").to_pylist())
        for sp in sorted(tids) or [None]:  # no teacher ids at all: coverage 0
            name = f"{s}/{sp}" if sp else s
            t, a = tids.get(sp, set()), aids.get(sp, set())
            hit = len(t & a)
            cov = hit / len(t) if t else 0.0
            report[name] = dict(teacher_ids=len(t), audio_ids=len(a), joined=hit, coverage=round(cov, 5))
            print(f"  {name:20s} teacher {len(t):7d}  audio {len(a):7d}  joined {hit:7d}  coverage {cov:.4f}")
            if cov < floor:
                bad.append(name)
    (state / "bootstrap_coverage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if bad:
        sys.exit(f"coverage below {floor} for {bad}: the audio does not match the teacher outputs")


{"plan": plan, "pull": pull, "coverage": coverage}[sys.argv[1]]()
PYEOF

log "repo $KITSUNE_DIR at $(git -C "$KITSUNE_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'), config $CONFIG, data $KITSUNE_DATA_REPO@$KITSUNE_DATA_REVISION"
log "disk free before: $(df -h --output=avail "$KITSUNE_DIR" | tail -1 | tr -d ' ')"
# the plan only reads the Hub and rewrites bootstrap_plan.json; neither its GETs nor snapshot_download's repo_info and
# tree listing have an HTTP timeout, so each attempt gets one. A killed pull resumes: downloads land as .incomplete
# files renamed when done, and snapshot_download skips what is already on disk (~2.3 GB in all)
phase plan retry 3 timeout -k 30 10m "$PY" "$HELPER" plan
phase pull_derived retry 3 timeout -k 30 30m "$PY" "$HELPER" pull

DATA_ROOT="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["data_root"])' "$STATE/bootstrap_plan.json")"
mapfile -t REBUILD < <("$PY" -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["rebuild"]))' "$STATE/bootstrap_plan.json" | sed '/^$/d')
EXTENT="$("$PY" -c 'import json,sys; print("1" if json.load(open(sys.argv[1])).get("extent") else "")' "$STATE/bootstrap_plan.json")"
if [ "${#REBUILD[@]}" -gt 0 ] && [ -z "$EXTENT" ]; then
    read -r -a PREP_ARGS <<< "${KITSUNE_PREP_ARGS:-}"
    # 01 resumes from its progress.json and manifest and its writes are kill-safe, so a retry, or a timeout that proves
    # too short for a healthy rebuild, only redoes the input in progress; 3 x 60 min caps a stall well before the
    # 5.5 h watchdog
    phase rebuild_audio retry 3 timeout -k 60 60m "$PY" scripts/01_prepare_data.py --data "$KITSUNE_DIR/$DATA_ROOT" \
        --sources "${REBUILD[@]}" "${PREP_ARGS[@]}"
elif [ -n "$EXTENT" ]; then
    # the canonical ingest sequence the label box ran, capped by the config's extent (a retry skips 01's finished
    # inputs and steps); vast/launch.py sizes the per-attempt timeout from the record's upstream bytes
    [ -z "${KITSUNE_PREP_ARGS:-}" ] || log "KITSUNE_PREP_ARGS ignored: the config's extent defines the rebuild"
    phase rebuild_audio retry 3 timeout -k 60 "${KITSUNE_REBUILD_TIMEOUT_MIN:-60}m" "$PY" \
        scripts/01_prepare_data.py --data "$KITSUNE_DIR/$DATA_ROOT" --extent-config "$CONFIG"
else
    log "every source has parked shards; nothing to rebuild"
fi
phase coverage "$PY" "$HELPER" coverage
log "disk free after: $(df -h --output=avail "$KITSUNE_DIR" | tail -1 | tr -d ' ')"
log "bootstrap complete"
