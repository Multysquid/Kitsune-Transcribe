#!/bin/bash
# vast.ai on-start script for the Kitsune-Transcribe viability run (SSH launch mode; runs as root at EVERY container
# start). vast/launch.py passes the small vast/onstart_stub.sh with `vastai create instance --onstart` (the API may cap
# that field near 4 KB); the stub clones the repo at $KITSUNE_SHA and execs this file from the clone. It still clones
# by itself if run on a box without the stub (clone_repo is a no-op when the checkout is already at $KITSUNE_SHA).
#
# Steps: sync the env to /etc/environment (SSH/tmux sessions do not inherit the container env), raise the nofile limit,
# check /dev/shm and the pids budget, start TensorBoard and the vast portal, clone the repo at $KITSUNE_SHA, start
# vast/watchdog.sh (hard cost cap), then detached: vast/bootstrap.sh (data) and vast/supervise.py (training +
# stop/destroy), all logging to /workspace/kitsune.log.
#
# Restarts: the watchdog deadline is fixed at first boot; a `halt` marker (written by finish.py or by a failure here)
# means the run is over, so a restarted container only brings up the env and the portal for inspection. An interrupted
# run is handled by supervise.py's own history: a restart that finds $KITSUNE_STATE/supervise.json skips bootstrap (the
# data passed its coverage check before the supervisor first ran) and hands straight over to it. Any failure before
# the supervisor takes over stops the instance (not destroy), unless KITSUNE_NO_SELF_STOP=1. Bootstrap + supervisor
# hold $KITSUNE_STATE/supervise.lock, so running this script again by hand during a run starts nothing new.
#
# Re-arm a halted box for a fresh run: `bash vast/onstart.sh --rearm` moves the halt marker, the deadline and the
# supervisor/finish history to $KITSUNE_STATE/rearm-<stamp>/, then boots as if for the first time (new 5.5 h cap). It
# refuses while a watchdog or supervisor of this container still runs (stop/start the instance first).
#
# Instance env (set by launch.py): KITSUNE_SHA KITSUNE_CONFIG KITSUNE_DATA_REPO KITSUNE_OUT_REPO TZ, optional
# KITSUNE_DATA_REVISION KITSUNE_MAX_HOURS. HF_TOKEN comes from the vast account env and is never printed.
set -euo pipefail
set -o errtrace

export KITSUNE_REPO_URL="${KITSUNE_REPO_URL:-https://github.com/Multysquid/Kitsune-Transcribe}"
export KITSUNE_DIR="${KITSUNE_DIR:-/workspace/Kitsune-Transcribe}"
export KITSUNE_STATE="${KITSUNE_STATE:-/workspace/kitsune_state}"
export KITSUNE_LOG="${KITSUNE_LOG:-/workspace/kitsune.log}"
export KITSUNE_CONFIG="${KITSUNE_CONFIG:-configs/viability.json}"
export KITSUNE_MAX_HOURS="${KITSUNE_MAX_HOURS:-5.5}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTHONUNBUFFERED=1
export PATH="/venv/main/bin:$PATH"
PY=/venv/main/bin/python
[ -x "$PY" ] || PY="$(command -v python3)"
export KITSUNE_PY="$PY"

mkdir -p "$KITSUNE_STATE" "$(dirname "$KITSUNE_LOG")"
exec >>"$KITSUNE_LOG" 2>&1

log() { printf '%s [onstart] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

stop_instance() {  # $1 = reason
    if [ "${KITSUNE_NO_SELF_STOP:-0}" = 1 ]; then
        log "KITSUNE_NO_SELF_STOP=1: leaving the instance running ($1)"
        return 0
    fi
    if [ -f "$KITSUNE_DIR/vast/finish.py" ] \
        && "$PY" "$KITSUNE_DIR/vast/finish.py" --stop --no-sync --reason "$1"; then
        return 0
    fi
    if [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
        # the key goes through curl's stdin config, never argv; a 2xx reply can still refuse ({"success": false}, as
        # finish.py vast_rest reads it)
        local r=""
        r=$(printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" \
            | curl -fsS --retry 3 --max-time 30 --config - -X PUT -H 'Content-Type: application/json' \
                -d '{"state": "stopped"}' "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/") \
            && ! [[ $r =~ \"success\"[[:space:]]*:[[:space:]]*false ]] && { log "stop requested via REST"; return 0; }
        log "vast REST stop failed: ${r:-no reply}"
    fi
    log "could not stop the instance; the watchdog (if running) will stop it at the deadline"
    return 1
}

fail() {  # ERR trap: record why, stop the box, keep the disk for inspection
    local rc=$? line=$1
    trap - ERR
    log "FAILED with exit $rc at onstart.sh line $line"
    printf '{"action": "stop", "reason": "onstart failed at line %s (exit %s)", "wall": %s}\n' \
        "$line" "$rc" "$(date +%s)" > "$KITSUNE_STATE/halt"
    stop_instance "onstart failed at line $line (exit $rc)" || true
    exit "$rc"
}
trap 'fail $LINENO' ERR

sync_env() {
    # rewrite only our block so repeated boots do not grow the file; values are written, never echoed. On the first
    # boot the portal's 10-prep-env.sh (start_portal) then truncates the file and rewrites it from the env it inherited
    # from us (our block markers go, the values stay); later boots keep its marker line, so it leaves the file alone
    local tmp k v
    tmp=$(mktemp)
    if [ -f /etc/environment ]; then
        sed '/^# >>> kitsune env >>>$/,/^# <<< kitsune env <<<$/d' /etc/environment > "$tmp"
    fi
    {
        echo '# >>> kitsune env >>>'
        while IFS= read -r -d '' kv; do
            k=${kv%%=*}
            v=${kv#*=}
            case "$k" in PWD|OLDPWD|SHLVL|_|HOME|HOSTNAME|TERM|SHELL|USER|LOGNAME|BASH_*) continue ;; esac
            [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
            [[ "$v" == *$'\n'* || "$v" == *'"'* ]] && continue
            printf '%s="%s"\n' "$k" "$v"
        done < <(env -0)
        echo '# <<< kitsune env <<<'
    } >> "$tmp"
    install -m 600 "$tmp" /etc/environment
    rm -f "$tmp"
}

port_busy() {
    "$PY" -c 'import socket, sys; s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$1"
}

start_tensorboard() {
    # our own TensorBoard on localhost, so `ssh -L 6006:localhost:<port>` works whatever the portal does with 6006.
    # A viewer that fails must never fail the boot (the ERR trap under errtrace stops the box): every command that can
    # fail is guarded, and the wait for the port only decides which line is logged. The first A100 box logged a start
    # while tensorboard.main had died at import (setuptools without pkg_resources): say whether it answers, is still
    # starting (alive, no answer yet) or died (its log's last line). scalars=30000: every step of a 4 h run (~17k) in
    # the charts, not the default random 1000 (~1.5 KB of RAM per point, ~2 GB for the ~70 per-step tags)
    local port pid last="" waited=0 wait_s=20
    mkdir -p "$KITSUNE_DIR/runs" 2>/dev/null || true
    for port in 6006 6007 6008; do
        if ! port_busy "$port"; then
            nohup setsid "$PY" -m tensorboard.main --logdir "$KITSUNE_DIR/runs" --host 127.0.0.1 --port "$port" \
                --samples_per_plugin scalars=30000 > /workspace/tensorboard.log 2>&1 < /dev/null &
            pid=$!  # the viewer itself: no job control here, so setsid runs it in place
            echo "$port" > "$KITSUNE_STATE/tensorboard_port" 2>/dev/null || true
            while [ "$waited" -lt "$wait_s" ] && kill -0 "$pid" 2>/dev/null; do
                if port_busy "$port"; then
                    log "TensorBoard up on 127.0.0.1:$port (laptop: ssh -p <port> root@<ip> -L 6006:localhost:$port)"
                    return 0
                fi
                sleep 1 || true
                waited=$(( waited + 1 ))
            done
            if kill -0 "$pid" 2>/dev/null; then
                log "TensorBoard not answering on 127.0.0.1:$port after $wait_s s (still starting?);" \
                    "see /workspace/tensorboard.log"
                return 0
            fi
            last=$(tail -n 1 /workspace/tensorboard.log 2>/dev/null || true)
            log "TensorBoard did not start: ${last:-/workspace/tensorboard.log is empty (no answer on port $port)}"
            return 0
        fi
    done
    log "ports 6006-6008 busy; TensorBoard not started"
}

start_portal() {
    local ep=/opt/instance-tools/bin/entrypoint.sh
    if [ -x "$ep" ] && ! pgrep -f supervisord >/dev/null 2>&1; then
        export TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-$KITSUNE_DIR/runs}"
        nohup setsid "$ep" > /workspace/portal.log 2>&1 < /dev/null &
        log "vast portal entrypoint started"
    fi
}

clone_repo() {
    if ! [[ "$KITSUNE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
        log "KITSUNE_SHA must be a full 40-hex commit SHA, got '$KITSUNE_SHA'"
        return 1
    fi
    if [ -d "$KITSUNE_DIR/.git" ] && [ "$(git -C "$KITSUNE_DIR" rev-parse HEAD 2>/dev/null || true)" = "$KITSUNE_SHA" ]; then
        log "repo already at $KITSUNE_SHA"
        return 0
    fi
    mkdir -p "$KITSUNE_DIR"
    git -C "$KITSUNE_DIR" init -q
    git -C "$KITSUNE_DIR" remote remove origin 2>/dev/null || true
    git -C "$KITSUNE_DIR" remote add origin "$KITSUNE_REPO_URL"
    local i
    for i in 1 2 3; do
        timeout 300 git -C "$KITSUNE_DIR" fetch -q --depth 1 origin "$KITSUNE_SHA" && break
        log "git fetch attempt $i failed"
        sleep $(( i * 15 ))
    done
    git -C "$KITSUNE_DIR" checkout -q --force --detach "$KITSUNE_SHA"
    log "cloned $KITSUNE_REPO_URL at $KITSUNE_SHA"
}

rearm() {  # --rearm: archive the previous run's lifecycle state (see the header)
    local f d
    if command -v flock >/dev/null; then
        for f in watchdog supervise; do
            if ! flock -n "$KITSUNE_STATE/$f.lock" true; then
                log "--rearm: a $f of this container is still running; stop and start the instance, then re-arm"
                exit 1
            fi
        done
    fi
    d="$KITSUNE_STATE/rearm-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$d"
    for f in halt deadline first_boot supervise.json events.jsonl bootstrap_timings.jsonl bootstrap_coverage.json; do
        if [ -e "$KITSUNE_STATE/$f" ]; then
            mv "$KITSUNE_STATE/$f" "$d/"
        fi
    done
    log "--rearm: previous state moved to $d; starting a fresh run with a new deadline"
}

# ------------------------------------------------------------------------------------------------------------ main
exec 8>"$KITSUNE_STATE/onstart.lock"
if command -v flock >/dev/null && ! flock -n 8; then
    log "another onstart is running; exiting"
    exit 0
fi
# background children inherit fd 8; unlock explicitly so a manual re-run after this one is not locked out
trap 'flock -u 8 2>/dev/null || true' EXIT
if [ "${1:-}" = "--rearm" ]; then
    rearm
fi

boots=$(( $(cat "$KITSUNE_STATE/boots" 2>/dev/null || echo 0) + 1 ))
echo "$boots" > "$KITSUNE_STATE/boots"
now=$(date +%s)
if [ ! -s "$KITSUNE_STATE/deadline" ]; then
    echo "$now" > "$KITSUNE_STATE/first_boot"
    echo $(( now + $(awk -v h="$KITSUNE_MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }') )) > "$KITSUNE_STATE/deadline"
fi
log "boot $boots, instance ${CONTAINER_ID:-?}, sha ${KITSUNE_SHA:-unset}, config $KITSUNE_CONFIG, " \
    "HF_TOKEN $([ -n "${HF_TOKEN:-}" ] && echo set || echo MISSING)"

ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || log "could not raise the nofile soft limit"
log "nofile soft limit $(ulimit -Sn)"
shm_kb=$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 { print $2 }' || true)
[[ "$shm_kb" =~ ^[0-9]+$ ]] || shm_kb=0
if [ "$shm_kb" -lt $(( 2 * 1024 * 1024 )) ]; then
    # informational only: both torch sharing strategies live in /dev/shm; 04_distill.py fits its DataLoader
    # workers x prefetch into the free /dev/shm itself (shm_cap)
    export KITSUNE_SHARING=file_system
    log "/dev/shm is $(( shm_kb / 1024 )) MB (< 2 GB): KITSUNE_SHARING=file_system"
else
    log "/dev/shm is $(( shm_kb / 1024 )) MB"
fi
# the base image's 12-cpu-thread-limits.sh caps the CPU thread pools on a host whose pids budget is below 16 per
# visible CPU (else pools sized to nproc hit EAGAIN), but only in the portal's shell, not in ours (bootstrap's hf_xet
# downloads, the trainer and its DataLoader workers): same trigger here, before sync_env so SSH sessions get it too.
# Every read is guarded: with errtrace a failing $(cat ...) would fire the ERR trap and stop the box
pids_max=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || echo unknown)
ncpu=$(nproc 2>/dev/null || echo 0)
log "pids.max $pids_max, nproc $ncpu, cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
if [[ "$pids_max" =~ ^[0-9]+$ && "$ncpu" =~ ^[0-9]+$ ]] && (( pids_max < ncpu * 16 )); then
    for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS RAYON_NUM_THREADS \
        TOKIO_WORKER_THREADS; do
        export "$v=${!v:-16}"  # a value already set wins
    done
    log "low pids budget: CPU thread pools capped to 16"
fi
sync_env

if [ -f "$KITSUNE_STATE/halt" ]; then
    log "halt marker present: $(cat "$KITSUNE_STATE/halt")"
    log "not starting a run; to start a fresh one: bash $KITSUNE_DIR/vast/onstart.sh --rearm"
    start_tensorboard
    start_portal
    exit 0
fi

for var in KITSUNE_SHA KITSUNE_DATA_REPO; do
    if [ -z "${!var:-}" ]; then
        log "$var is not set (vast/launch.py passes it)"
        false  # -> ERR trap: halt marker + stop
    fi
done
start_tensorboard
start_portal
clone_repo
nohup setsid bash "$KITSUNE_DIR/vast/watchdog.sh" >> /workspace/watchdog.log 2>&1 < /dev/null &
log "watchdog started (deadline $(date -u -d "@$(cat "$KITSUNE_STATE/deadline")" +%H:%M:%SZ))"

# bootstrap + supervisor detached from vast's on-start hook; the ERR trap (errtrace) still covers bootstrap failures
(
    trap '' HUP
    exec 7>"$KITSUNE_STATE/supervise.lock"
    if command -v flock >/dev/null && ! flock -n 7; then
        log "a bootstrap or supervisor of this container is already running; not starting another"
        exit 0
    fi
    if [ -s "$KITSUNE_STATE/supervise.json" ]; then
        # a restart of this run: the supervisor starts only after a bootstrap that passed its coverage check and
        # records its history before the first attempt (--rearm moves it aside), so the data is on disk; bootstrap's
        # Hub calls would turn a Hub outage at restart time into halt + stop instead of the resume / the recorded finish
        log "supervisor history present: bootstrap already done for this run; handing over to the supervisor"
    else
        log "bootstrap start"
        bash "$KITSUNE_DIR/vast/bootstrap.sh"
        log "bootstrap done; starting the supervisor"
    fi
    exec 7>&-  # handed over: supervise.py takes the same lock itself for its lifetime
    exec "$PY" "$KITSUNE_DIR/vast/supervise.py"
) < /dev/null &
disown || true
log "onstart done; follow with: tail -f $KITSUNE_LOG"
