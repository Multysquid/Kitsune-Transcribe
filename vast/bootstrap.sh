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
# for a CTC student or with pull_parakeet (the size study pulls both label roots); a CTC run without pull_parakeet (box
# p01) pulls teacher_out only for its eval sets' eval stems, so its extent and coverage checks read the other stems' ids
# from parakeet_out (kitsune.extent.label_root_for, fix 9).
#
# Study box (KITSUNE_JOB=study, KITSUNE_BOX=A|B|replicate|shakedown; KITSUNE_CONFIG is study/data.json, the data block,
# which names no student): the students pulled are the box's own (kitsune.study_queue.box_students), each with
# STUDENT_FILES and, for a Parakeet-derived one, its CC-BY-4.0 MODEL_CARD.md; plus box_extra_dirs (the Parakeet
# teacher for the speed probes). After the pull, `python -m kitsune.study_queue check-students` refuses (exit 2) a
# pulled student that is not the registered build (kitsune.prereg.student_problems).
# The rebuild is `01 --extent-config $KITSUNE_CONFIG`, the canonical ingest sequence the label box ran, so the ids and
# stems are the labelled ones (KITSUNE_PREP_ARGS is ignored), and coverage is exact: every pulled stem's rebuilt id
# sidecar hashes to the record's ids_sha256, its teacher ids are a subset of its ids, and every split joins at 1.0.
#
# Full-data box (KITSUNE_JOB=full, KITSUNE_BOX; KITSUNE_CONFIG is the box's data config configs/full/data-*.json, which
# names no student): the students, extra files (the frozen eval manifest, the selection sidecar) and extra dirs come
# from the box registry (kitsune/fullrun.py, configs/full/boxes.json of this checkout). Phases: download_gate
# (kitsune.netgate, with KITSUNE_GATE_BYTES: exit 3 is a slow host, not retried, and it sets the label pull's and the
# rebuild's per-attempt timeouts KITSUNE_PULL_TIMEOUT_MIN / KITSUNE_REBUILD_TIMEOUT_MIN), plan (HF_TOKEN must also read
# and write KITSUNE_SCRATCH_REPO; a box with timed states refuses without it), pull_derived (all but the labels),
# resume_pull (KITSUNE_RESUME=1: python -m kitsune.full_queue resume-pull; exit 3 not retried), check_students (python
# -m kitsune.fullrun check-students), pull_labels in the background while 01 rebuilds the audio (decision 13), the
# rebuild, pull_labels_wait, coverage. Each phase keeps $STATE/train_hb fresh (vast/watchdog.sh reads it) while it runs,
# for at most its own worst case (the label pull and the rebuild: their three attempts' timeouts) or else
# KITSUNE_PHASE_HB_MAX_S (12 h), and never after bootstrap has exited. vast/README.md "Full-data runs" has the box's
# whole life.
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

stop_label_pull() {  # job full, at exit: the background label pull and everything it started. Killing its subshell
    # alone leaves its children running, holding onstart's supervise.lock (fd 7): the pull's timeout (SIGTERM reaches
    # its python through timeout) and the phase's train_hb toucher (which also ends by itself once bootstrap is gone).
    # The children are listed first: once the subshell is dead they belong to init. ps -ef has PID in column 2 and
    # PPID in column 3, with procps and Git Bash alike
    [ -n "${LABELS_PID:-}" ] || return 0
    local kids
    kids=$(ps -ef 2>/dev/null | awk -v p="$LABELS_PID" '$3 == p { print $2 }') || true
    kill "$LABELS_PID" 2>/dev/null || true
    [ -z "$kids" ] || kill $kids 2>/dev/null || true
    LABELS_PID=""
}
trap 'rm -f "$HELPER"; stop_label_pull' EXIT

beat_train_hb() {  # beat_train_hb <max_s>: touch $STATE/train_hb (the watchdog's heartbeat on a full box) every 60 s
    # for at most max_s, and only while this bootstrap runs ($$ is its pid in every subshell): a toucher left behind
    # by a killed bootstrap would keep an orphaned box looking alive
    local end=$(( $(date +%s) + $1 ))
    while [ "$(date +%s)" -lt "$end" ] && kill -0 "$$" 2>/dev/null; do
        touch "$STATE/train_hb" 2>/dev/null || true
        sleep 60
    done
}

phase_budget_s() {  # phase_budget_s <tries> <minutes>: the longest `retry <tries> timeout -k <=60 <minutes>m ...` runs
    # (every attempt with its kill grace, retry's pauses of 60, 120, ... s), plus 10 min: a long phase's toucher bound
    echo $(( $1 * ($2 * 60 + 60) + 30 * $1 * ($1 - 1) + 600 ))
}

phase() {  # phase <name> <command...>: run it and append its wall time
    local name=$1 t0 t1 hb max_s rc=0
    shift
    t0=$(date +%s.%N)
    log "phase $name ..."
    if [ "${KITSUNE_JOB:-}" = full ]; then
        # the box controller's heartbeat while the phase runs. Every phase is bounded by its own timeout, and so is
        # its toucher: by PHASE_HB_MAX_S, the budget of a phase that may run long (the label pull, the rebuild), else
        # by KITSUNE_PHASE_HB_MAX_S (12 h, far above the other phases' timeouts). A hung bootstrap still goes stale
        max_s=${PHASE_HB_MAX_S:-${KITSUNE_PHASE_HB_MAX_S:-43200}}
        log "phase $name keeps train_hb fresh for at most $max_s s"
        beat_train_hb "$max_s" &
        hb=$!
        "$@" || rc=$?
        kill "$hb" 2>/dev/null || true
        wait "$hb" 2>/dev/null || true
        [ "$rc" -eq 0 ] || return "$rc"
    else
        "$@"
    fi
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
"""bootstrap helper: plan / pull / coverage. Reads the run config for sources and paths; a full-data box (KITSUNE_JOB=full)
also its box registry, and pulls its labels with pull_labels, apart from the rest."""
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
extra_files, timed_states = [], False  # a full-data box's extra data-repo files; whether it needs the scratch repo
if os.environ.get("KITSUNE_JOB") == "study":
    # a size-study box (kitsune/study_queue.py): the data block has no student; the box pulls the student dirs of its
    # own runs only (box_students), with the Parakeet students' attribution card, and any other dir it needs
    sys.path.insert(0, str(root))
    from kitsune import study_queue as _q
    _box = os.environ["KITSUNE_BOX"]
    students = _q.box_students(_box)
    ctc_students = set(_q.box_ctc_students(_box))
    extra_dirs = _q.box_extra_dirs(_box)
elif os.environ.get("KITSUNE_JOB") == "full":
    # a full-data box (kitsune/full_queue.py): the data block has no student either; the box registry (kitsune/fullrun.py,
    # configs/full/boxes.json of this checkout, which launch.py checked at the same commit) names the train items'
    # students, the extra files (the frozen manifest, the selection sidecar) and dirs. A registry this checkout cannot
    # load is a refusal (exit 3, as refuse() below): a retry would read the same files
    sys.path.insert(0, str(root))
    from kitsune import fullrun as _f
    _box = os.environ["KITSUNE_BOX"]
    try:
        _reg = _f.load_registry(root=root)
        students, ctc_students = _f.box_students(_box, _reg), set(_f.box_ctc_students(_box, _reg))
        extra_dirs, extra_files = _f.box_extra_dirs(_box, _reg), _f.box_extra_files(_box, _reg)
        timed_states = _f.box_spec(_box, _reg)["timed_states"]
    except _f.RegistryError as e:
        print(f"box {_box}: {e} (not retried)", file=sys.stderr)
        sys.exit(3)
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
    # audio rebuild and the model load; auth_check is a GET (no commit), and the trainer reads its uploads back. A
    # full-data box checks its scratch repo the same way (the trainers' timed states go there; the owner creates it
    # and scopes the token, vast/README.md "Full-data runs")
    out_repo = os.environ["KITSUNE_OUT_REPO"]
    scratch = os.environ.get("KITSUNE_SCRATCH_REPO") or None
    if timed_states and not scratch:
        refuse(f"box {os.environ.get('KITSUNE_BOX')} keeps timed states, but KITSUNE_SCRATCH_REPO is not set: relaunch "
               f"with vast/launch.py --scratch-repo <the private scratch model repo>")
    checks = [(out_repo, "vast/README.md step 2.3")] + ([(scratch, "vast/README.md, full-data runs")] if scratch else [])
    for repo_id, doc in checks:
        for write in (False, True):
            try:
                api.auth_check(repo_id, repo_type="model", write=write)
            except Exception as e:
                if refused(e):
                    refuse(f"HF_TOKEN cannot {'write' if write else 'read'} {repo_id} ({type(e).__name__}: {e}): give "
                           f"the fine-grained token read and write access to it ({doc})")
                sys.exit(f"Hub error checking {repo_id} ({type(e).__name__}: {e}); bootstrap retries the plan")
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
    pull_plan: directory globs for uncapped sources, explicit files for capped ones; parakeet_out only for a CTC
    student or with pull_parakeet) and rebuild all of its audio with `01 --extent-config`. A root the label box has not
    sealed (no COMPLETE.json) is refused: its files may still change, and its extent.json is written only at the seal.
    A full-data box also pulls the registry's extra files, and its labels go to the plan's "labels" part, which
    pull_labels fetches while 01 rebuilds the audio (pull fetches the rest)."""
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
    problems += [f"no {f}" for f in extra_files if f not in have]
    if problems:
        refuse(f"data repo {repo}@{rev} cannot serve extent {cfg['extent'].get('name')!r}: {problems}")
    # a study box's students (pull_plan pulls the config's one student, and the data block has none) and other dirs
    p["dir_patterns"] += [f"{d}/*" for d in [*students, *extra_dirs] if f"{d}/*" not in p["dir_patterns"]]
    explicit = [*p["explicit"], *(f for f in extra_files if f not in p["explicit"])]
    labels = None
    if os.environ.get("KITSUNE_JOB") == "full":  # the labels come down during the rebuild (pull_labels)
        labels = dict(patterns=[x for x in p["dir_patterns"] if is_label(x)],
                      explicit=[x for x in explicit if is_label(x)])
        labels["files"] = list(dict.fromkeys([*filter_repo_objects(files, allow_patterns=labels["patterns"]),
                                              *labels["explicit"]]))
        p["dir_patterns"] = [x for x in p["dir_patterns"] if not is_label(x)]
        explicit = [x for x in explicit if not is_label(x)]
    # the files the pull must leave on disk: the globs by snapshot_download's own matcher, plus the explicit files
    want = list(dict.fromkeys([*filter_repo_objects(files, allow_patterns=p["dir_patterns"]), *explicit]))
    rebuild = extent_names(cfg)
    out = dict(repo=repo, revision=rev, patterns=p["dir_patterns"], parked=[], rebuild=rebuild, data_root=data_root,
               repo_files=len(files), files=want, wall=time.time(), extent=True, explicit=explicit,
               record=str(Path(path).resolve()))
    if labels is not None:
        out["labels"] = labels
    plan_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"plan: extent {cfg['extent'].get('name')!r}: pull {len(p['dir_patterns'])} patterns and "
          f"{len(explicit)} explicit files; rebuild audio {rebuild} with 01 --extent-config")
    if labels is not None:
        print(f"plan: pull_labels {len(labels['patterns'])} patterns and {len(labels['explicit'])} explicit files "
              f"({len(labels['files'])} files) during the rebuild")


def is_label(path: str) -> bool:
    """A label file or glob: under a label root's per-name dirs (<root>/<name>/...). The roots' meta.json files, the
    selection, the students and the extra files are not (pull fetches them before the rebuild)."""
    roots = [teacher_root, second_root] + ([cfg["parakeet_root"]] if cfg.get("parakeet_root") else [])
    return any(path.startswith(f"{r}/") and "/" in path[len(r) + 1:] for r in roots)


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


def fetch(patterns: list, explicit: list, want: list):
    """snapshot_download of the patterns and hf_hub_download of the explicit files; then every file of `want` must be
    on disk."""
    from huggingface_hub import snapshot_download

    if patterns:
        snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=root, allow_patterns=patterns,
                          max_workers=16)
    if explicit:
        # a capped source's files one by one (fnmatch over ~2k patterns x ~27k repo files would be too slow); a file
        # already on disk is complete (downloads land as .incomplete files renamed when done), so a retry skips it
        from concurrent.futures import ThreadPoolExecutor

        from huggingface_hub import hf_hub_download

        todo = [f for f in explicit if not (root / f).is_file()]
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda f: hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=root), todo))
        print(f"pulled {len(todo)} of {len(explicit)} explicit files ({len(explicit) - len(todo)} on disk)")
    # when its one repo_info request fails (a Hub 5xx/429, a dropped connection) snapshot_download returns local_dir
    # (the checkout, never empty) with only a warning and downloads nothing, and 01's paid audio rebuild would run
    # before the coverage check caught it: fail here instead, so the shell's retry pulls again
    missing = [f for f in want if not (root / f).is_file()]
    if missing:
        sys.exit(f"pull incomplete: {len(missing)} of {len(want)} planned files missing, e.g. {missing[:3]}")


def pull():
    p = json.loads(plan_path.read_text(encoding="utf-8"))
    t0 = time.time()
    fetch(p["patterns"], p.get("explicit") or [], p["files"])
    register_parked(p["parked"])
    dirs = [root / d for d in (teacher_root, second_root, *students, f"{data_root}/shards") if (root / d).is_dir()]
    size = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
    print(f"pulled in {time.time() - t0:.0f} s; derived data + parked shards on disk: {size / 1e9:.2f} GB")


def pull_labels():
    """A full-data box's labels (the plan's "labels" part), pulled in the background while 01 rebuilds the audio;
    bootstrap waits for it before the coverage check (pull_labels_wait)."""
    p = json.loads(plan_path.read_text(encoding="utf-8"))
    labels = p.get("labels")
    if labels is None:
        sys.exit(f"{plan_path} has no labels part (pull_labels is for KITSUNE_JOB=full with an extent config)")
    t0 = time.time()
    fetch(labels["patterns"], labels["explicit"], labels["files"])
    print(f"pulled the labels ({len(labels['files'])} files) in {time.time() - t0:.0f} s")


def label_dir(s: str, stem: str) -> tuple:
    """(name, root) of the label root that holds stem's ids on this box (kitsune.extent label_root_for, fix 9): a CTC
    run without pull_parakeet pulled parakeet_out for every stem and teacher_out only for its eval sets' eval stems."""
    if str(root) not in sys.path:  # this helper runs from a mktemp path (and this is called once per stem)
        sys.path.insert(0, str(root))
    from kitsune.extent import label_root_for

    which = label_root_for(cfg, s, stem)
    return which, (cfg.get("parakeet_root") if which == "parakeet" else teacher_root)


def extent_stems(report: dict) -> list:
    """Extent mode: every pulled stem (kitsune.extent subset_stems) must have been rebuilt with the labelled ids. The
    rebuilt id sidecar's ids must hash to the record's ids_sha256 (same ids, same order, same stem), and the label ids
    of the stem (teacher_out's, or parakeet_out's where label_dir says so) must be a subset of them. Returns the
    failures; report["extent_stems"] gets the counts."""
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
            which, lroot = label_dir(s, stem)
            try:
                with np.load(root / lroot / s / f"{stem}.npz", allow_pickle=False) as z:
                    extra = {str(i) for i in z["ids"]} - set(ids)
            except FileNotFoundError as e:
                bad.append(f"{s}/{stem}: no {which} labels ({e})")
                continue
            if extra:
                bad.append(f"{s}/{stem}: {len(extra)} {which} ids not in the rebuilt stem, e.g. {sorted(extra)[:2]}")
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
    # extent mode: each stem's ids come from the one root label_dir names (fix 9: parakeet_out on a CTC box without
    # pull_parakeet, whose train stems have no teacher_out); an AED or both-roots box, and the legacy pull, read
    # teacher_out only
    roots = [teacher_root] + ([cfg["parakeet_root"]] if cfg.get("extent") and cfg.get("parakeet_root") else [])
    for s in names:
        # per split (<split>-NNNNN in both trees), as the trainer joins them: pooled over a source's splits,
        # galgame's 1,000-row hold-out is 0.5 % of its ids and could vanish (or trade rows with train) above the floor
        tids, aids, from_parakeet, picked = {}, {}, set(), {}
        for r in roots:
            for npz in sorted((root / r / s).glob("*.npz")):
                if len(roots) == 1 or label_dir(s, npz.stem)[1] == r:
                    picked.setdefault(npz.stem, (r, npz))
        for stem, (r, npz) in sorted(picked.items()):
            with np.load(npz, allow_pickle=False) as z:
                tids.setdefault(stem.rsplit("-", 1)[0], set()).update(str(i) for i in z["ids"])
            if r != teacher_root:
                from_parakeet.add(stem.rsplit("-", 1)[0])
        for shard in sorted((root / data_root / "shards" / s).glob("*.parquet")):
            aids.setdefault(shard.stem.rsplit("-", 1)[0], set()).update(
                pq.read_table(shard, columns=["id"]).column("id").to_pylist())
        for sp in sorted(tids) or [None]:  # no teacher ids at all: coverage 0
            name = f"{s}/{sp}" if sp else s
            t, a = tids.get(sp, set()), aids.get(sp, set())
            hit = len(t & a)
            cov = hit / len(t) if t else 0.0
            report[name] = dict(teacher_ids=len(t), audio_ids=len(a), joined=hit, coverage=round(cov, 5))
            if sp in from_parakeet:  # the label ids of a CTC-only box's train split are Parakeet's (fix 9)
                report[name]["root"] = "parakeet"
            who = "parakeet" if sp in from_parakeet else "teacher"
            print(f"  {name:20s} {who} {len(t):7d}  audio {len(a):7d}  joined {hit:7d}  coverage {cov:.4f}")
            if cov < floor:
                bad.append(name)
    (state / "bootstrap_coverage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if bad:
        sys.exit(f"coverage below {floor} for {bad}: the audio does not match the teacher outputs")


{"plan": plan, "pull": pull, "pull_labels": pull_labels, "coverage": coverage}[sys.argv[1]]()
PYEOF

log "repo $KITSUNE_DIR at $(git -C "$KITSUNE_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'), config $CONFIG, data $KITSUNE_DATA_REPO@$KITSUNE_DATA_REVISION"
log "disk free before: $(df -h --output=avail "$KITSUNE_DIR" | tail -1 | tr -d ' ')"
if [ "${KITSUNE_JOB:-}" = "full" ]; then
    # the box controller's heartbeat (vast/watchdog.sh reads $STATE/train_hb on a full box): fresh from the first
    # minute, then every phase keeps it so (phase)
    touch "$STATE/train_hb"
fi
if [ "${KITSUNE_JOB:-}" = "full" ] && [ -n "${KITSUNE_GATE_BYTES:-}" ]; then
    # fix 1, the download gate: time three pinned upstream files the way 01 downloads them before anything is pulled
    # or rebuilt. Exit 3 is a host too slow for the reference 571 GB within KITSUNE_GATE_MAX_H (not retried): the
    # bootstrap fails, and onstart's finish.py --abort destroys the box, which has no run dir yet, after its infra
    # upload put download_gate.json on the Hub (launch.py then avoids the machine). A pass sets the per-attempt
    # timeouts of the label pull and the rebuild from the measured rate (the rebuild's never below launch's sizing)
    phase download_gate retry 2 timeout -k 30 30m "$PY" -m kitsune.netgate --out "$STATE/download_gate.json" \
        --dir "$STATE/netgate"
    read -r KITSUNE_PULL_TIMEOUT_MIN KITSUNE_REBUILD_TIMEOUT_MIN < <("$PY" -m kitsune.netgate --timeouts \
        "$STATE/download_gate.json")
    export KITSUNE_PULL_TIMEOUT_MIN KITSUNE_REBUILD_TIMEOUT_MIN
    log "download gate passed: per attempt ${KITSUNE_PULL_TIMEOUT_MIN} min for the labels," \
        "${KITSUNE_REBUILD_TIMEOUT_MIN} min for the rebuild"
fi
# the plan only reads the Hub and rewrites bootstrap_plan.json; neither its GETs nor snapshot_download's repo_info and
# tree listing have an HTTP timeout, so each attempt gets one. A killed pull resumes: downloads land as .incomplete
# files renamed when done, and snapshot_download skips what is already on disk (~2.3 GB in all)
phase plan retry 3 timeout -k 30 10m "$PY" "$HELPER" plan
phase pull_derived retry 3 timeout -k 30 30m "$PY" "$HELPER" pull
if [ "${KITSUNE_JOB:-}" = "study" ]; then
    # every pulled student is the registered build (kitsune.prereg.student_problems through study_queue.student_checks:
    # stage, family, init class, seed, the exact parameter counts, a pruned student's calibration ids). launch.py
    # checked the data repo's metas before renting; this checks the files the box will train, before the audio rebuild
    # it would otherwise pay for. Exit 2 (a refusal) stops the bootstrap
    phase check_students "$PY" -m kitsune.study_queue check-students --box "$KITSUNE_BOX" --root "$KITSUNE_DIR"
fi
if [ "${KITSUNE_JOB:-}" = "full" ]; then
    if [ "${KITSUNE_RESUME:-}" = 1 ]; then
        # a relaunch on a new host (launch.py --resume): the box's Hub queue summary says what is done and which run
        # dirs resume; resume-pull pulls them with their newest full state (scratch or runs repo) and writes
        # $STATE/resume_plan.json, which the queue adopts. Before the paid rebuild, so a refusal (exit 3: no summary,
        # an unknown or finished run id, a state that does not match its pointer) costs minutes; not retried
        phase resume_pull retry 3 timeout -k 30 60m "$PY" -m kitsune.full_queue resume-pull --box "$KITSUNE_BOX" \
            --root "$KITSUNE_DIR"
    fi
    # the box registry's students are the registered builds (kitsune.prereg.student_problems), as for a study box;
    # exit 2 refuses
    phase check_students "$PY" -m kitsune.fullrun check-students --box "$KITSUNE_BOX" --root "$KITSUNE_DIR"
    # decision 13: the labels (tens of GB at the full extent) come down while 01 rebuilds the audio, per attempt
    # within the gate's KITSUNE_PULL_TIMEOUT_MIN; pull_labels_wait below fails the bootstrap on a failed pull, and a
    # bootstrap that fails first stops the pull (stop_label_pull). Its toucher, and pull_labels_wait's, last as long
    # as its three attempts may
    LABELS_HB_MAX_S=$(phase_budget_s 3 "${KITSUNE_PULL_TIMEOUT_MIN:-30}")
    PHASE_HB_MAX_S=$LABELS_HB_MAX_S
    phase pull_labels retry 3 timeout -k 30 "${KITSUNE_PULL_TIMEOUT_MIN:-30}m" "$PY" "$HELPER" pull_labels &
    LABELS_PID=$!
    log "pull_labels runs in the background (pid $LABELS_PID)"
    # the rebuild's toucher lasts its three attempts of KITSUNE_REBUILD_TIMEOUT_MIN (at least the 60 min of the
    # non-extent line): at the gate's floor rate the full extent's ~481 min per attempt make ~24 h, past the 12 h
    # default, so the watchdog stops a box only once the rebuild's own timeouts have run out, never a slow healthy one
    REBUILD_HB_MIN=${KITSUNE_REBUILD_TIMEOUT_MIN:-60}
    PHASE_HB_MAX_S=$(phase_budget_s 3 "$(( REBUILD_HB_MIN > 60 ? REBUILD_HB_MIN : 60 ))")
fi

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
if [ "${KITSUNE_JOB:-}" = "full" ] && [ -n "${LABELS_PID:-}" ]; then
    PHASE_HB_MAX_S=$LABELS_HB_MAX_S  # the wait lasts at most what is left of the pull's own budget
    phase pull_labels_wait wait "$LABELS_PID"  # the label pull's exit: a failed pull fails here, before coverage
    LABELS_PID=""
fi
if [ "${KITSUNE_JOB:-}" = "full" ]; then
    PHASE_HB_MAX_S=""  # coverage: the default bound
fi
phase coverage "$PY" "$HELPER" coverage
log "disk free after: $(df -h --output=avail "$KITSUNE_DIR" | tail -1 | tr -d ' ')"
log "bootstrap complete"
