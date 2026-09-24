#!/bin/bash
# Put the training data on the vast box (run by vast/onstart.sh from the cloned repo, cwd-independent).
#
# Why this split: home upload is slow, so only the small DERIVED data is parked in the private HF dataset
# $KITSUNE_DATA_REPO (teacher_out for the train sources and eval sets, second_out, the selection parquet, the student
# init; ~1.5 GB). The ~10 GB of audio is REBUILT here from the original public HF datasets by
# scripts/01_prepare_data.py, which reads every upstream repo at a pinned commit, so the utterance ids match the
# teacher outputs exactly. If the data repo also holds data/shards/<source>/*.parquet for a source, those parked
# shards are pulled instead of rebuilding that source. The trainer joins everything by utterance id and drops (and
# logs) rows without audio; this script already fails if coverage is below KITSUNE_MIN_COVERAGE, because a broken
# join would waste the whole paid run.
#
# The repo layout mirrors the laptop's repo root: <teacher_root>/..., <second_root>/..., <selection>, <student>/...,
# optionally <data_root>/shards/... (paths from the run config). Idempotent: snapshot_download and 01 both skip what is
# already on disk. Phase timings go to $KITSUNE_STATE/bootstrap_timings.jsonl.
#
# Env: KITSUNE_DATA_REPO (required), KITSUNE_DATA_REVISION (default main), KITSUNE_CONFIG (default
# configs/viability.json), KITSUNE_PREP_ARGS (extra args for 01; leave it empty for the viability data: its teacher
# outputs cover exactly the first 6 Galgame tars and 300 h of Emilia-YODAS, which are 01's defaults, and a larger
# --galgame-shards/--emilia-hours only downloads audio without teacher output), KITSUNE_MIN_COVERAGE (default 0.99),
# HF_TOKEN (vast account env; never printed).
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
# call is cut by the timeout the caller puts in <command>. The exit code of the last attempt is returned.
retry() {
    local n=$1 i rc=0
    shift
    for (( i = 1; i <= n; i++ )); do
        rc=0
        "$@" || rc=$?
        [ "$rc" -eq 0 ] && return 0
        log "attempt $i/$n of $* failed (exit $rc)"
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
student = cfg["student"].rstrip("/")
plan_path = state / "bootstrap_plan.json"


def plan():
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"hf user: {api.whoami()['name']}")
    files = api.list_repo_files(repo, repo_type="dataset", revision=rev)

    def has(pat: str) -> bool:
        return any(fnmatch.fnmatch(f, pat) for f in files)

    patterns = [f"{teacher_root}/meta.json", f"{second_root}/meta.json", selection, f"{student}/*"]
    required = [f"{selection}", f"{student}/config.json"]
    for s in names:
        patterns += [f"{teacher_root}/{s}/*", f"{second_root}/{s}/*"]
        required.append(f"{teacher_root}/{s}/*.npz")
    required += [f"{second_root}/{s}/*.jsonl" for s in sources]
    missing = [p for p in required if not has(p)]
    for s in sources:  # every teacher shard needs its second opinion (the same rule as vast/launch.py data_problems)
        teacher = {f.rsplit("/", 1)[1][:-4] for f in files if f.startswith(f"{teacher_root}/{s}/") and f.endswith(".npz")}
        second = {f.rsplit("/", 1)[1][:-6] for f in files if f.startswith(f"{second_root}/{s}/") and f.endswith(".jsonl")}
        if teacher - second:
            missing.append(f"{second_root}/{s}: {len(teacher - second)} of {len(teacher)} shards without a second opinion")
    if missing:
        sys.exit(f"data repo {repo}@{rev} lacks: {missing}")
    parked = [s for s in names if has(f"{data_root}/shards/{s}/*.parquet")]
    patterns += [f"{data_root}/shards/{s}/*.parquet" for s in parked]
    rebuild = [s for s in names if s not in parked]
    out = dict(repo=repo, revision=rev, patterns=patterns, parked=parked, rebuild=rebuild, data_root=data_root,
               repo_files=len(files), wall=time.time())
    plan_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"plan: pull {len(patterns)} patterns; parked audio {parked or 'none'}; rebuild audio {rebuild or 'none'}")


def pull():
    from huggingface_hub import snapshot_download

    p = json.loads(plan_path.read_text(encoding="utf-8"))
    t0 = time.time()
    snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=root, allow_patterns=p["patterns"],
                      max_workers=16)
    dirs = [root / d for d in (teacher_root, second_root, student, f"{data_root}/shards") if (root / d).is_dir()]
    size = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
    print(f"pulled in {time.time() - t0:.0f} s; derived data + parked shards on disk: {size / 1e9:.2f} GB")


def coverage():
    import numpy as np
    import pyarrow.parquet as pq

    floor = float(os.environ.get("KITSUNE_MIN_COVERAGE", "0.99"))
    report, bad = {}, []
    for s in names:
        tids = set()
        for npz in sorted((root / teacher_root / s).glob("*.npz")):
            with np.load(npz, allow_pickle=False) as z:
                tids.update(str(i) for i in z["ids"])
        aids = set()
        for shard in sorted((root / data_root / "shards" / s).glob("*.parquet")):
            aids.update(pq.read_table(shard, columns=["id"]).column("id").to_pylist())
        hit = len(tids & aids)
        cov = hit / len(tids) if tids else 0.0
        report[s] = dict(teacher_ids=len(tids), audio_ids=len(aids), joined=hit, coverage=round(cov, 5))
        print(f"  {s:14s} teacher {len(tids):7d}  audio {len(aids):7d}  joined {hit:7d}  coverage {cov:.4f}")
        if cov < floor:
            bad.append(s)
    (state / "bootstrap_coverage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if bad:
        sys.exit(f"coverage below {floor} for {bad}: the audio does not match the teacher outputs")


{"plan": plan, "pull": pull, "coverage": coverage}[sys.argv[1]]()
PYEOF

log "repo $KITSUNE_DIR at $(git -C "$KITSUNE_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'), config $CONFIG, data $KITSUNE_DATA_REPO@$KITSUNE_DATA_REVISION"
log "disk free before: $(df -h --output=avail "$KITSUNE_DIR" | tail -1 | tr -d ' ')"
phase plan "$PY" "$HELPER" plan
phase pull_derived "$PY" "$HELPER" pull

DATA_ROOT="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["data_root"])' "$STATE/bootstrap_plan.json")"
mapfile -t REBUILD < <("$PY" -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["rebuild"]))' "$STATE/bootstrap_plan.json" | sed '/^$/d')
if [ "${#REBUILD[@]}" -gt 0 ]; then
    read -r -a PREP_ARGS <<< "${KITSUNE_PREP_ARGS:-}"
    # 01 resumes from its progress.json and manifest and its writes are kill-safe, so a retry, or a timeout that proves
    # too short for a healthy rebuild, only redoes the input in progress; 3 x 60 min caps a stall well before the
    # 5.5 h watchdog
    phase rebuild_audio retry 3 timeout -k 60 60m "$PY" scripts/01_prepare_data.py --data "$KITSUNE_DIR/$DATA_ROOT" \
        --sources "${REBUILD[@]}" "${PREP_ARGS[@]}"
else
    log "every source has parked shards; nothing to rebuild"
fi
phase coverage "$PY" "$HELPER" coverage
log "disk free after: $(df -h --output=avail "$KITSUNE_DIR" | tail -1 | tr -d ' ')"
log "bootstrap complete"
