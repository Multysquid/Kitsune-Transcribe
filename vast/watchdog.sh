#!/bin/bash
# Hard cost cap for the vast instance: stop it KITSUNE_MAX_HOURS (default 5.5) after the FIRST boot, whatever the run
# is doing. Started detached by vast/onstart.sh.
#
# Why a separate process: the supervisor can hang (a stuck NCCL/CUDA call, a hung upload) and a forgotten instance bills
# until someone notices. The deadline is written once to $KITSUNE_STATE/deadline at first boot, so a container restart
# cannot extend it. SYNC_LEAD seconds before the deadline it runs a best-effort log upload (finish.py --sync-only), then
# it stops the instance (finish.py --stop, which writes the halt marker so a restart does not start a new run), and
# retries the stop every few minutes until the container dies. Stop, not destroy: the disk survives for inspection.
#
# Label box (KITSUNE_WATCHDOG_ORPHAN_S > 0; launch.py --job label sets 900): the controller vast/label.py touches
# $KITSUNE_STATE/label_hb at least every 30 s. If it goes stale by ORPHAN_S (label.py crashed, was OOM-killed or hangs),
# nothing would end the box before the cap, so the watchdog runs finish.py --sync-only (the labels on disk get to the
# Hub) and then stops the instance, retrying the stop like at the deadline. The rule arms only once this boot's
# controller has touched the file (a heartbeat older than the watchdog's own start is a previous boot's); if it never
# does, it fires at 2 x ORPHAN_S. A halt marker (finish.py has taken over) disables it. ORPHAN_S=0 (the train job)
# leaves the watchdog exactly as it was.
#
# Study box (KITSUNE_JOB=study): the same cap, per box (vast/launch.py STUDY_HOURS plus the rebuild timeout); its syncs
# are lean (finish.py reads KITSUNE_JOB: logs, weights and the uploaded full states of every run dir, never the resume
# states), and the queue's trainers stop with the instance.
#
# Full-data box (KITSUNE_JOB=full; vast/launch.py sets the three knobs from the box registry): the same rule on the box
# controllers' heartbeat, KITSUNE_WATCHDOG_HB_FILE=train_hb under $KITSUNE_STATE (bootstrap's phases, the queue's poll,
# the supervisor's bounded finish calls touch it; the trainers beat their own item files, which the queue's stall check
# reads), with KITSUNE_WATCHDOG_ORPHAN_S from the registry. KITSUNE_WATCHDOG_ORPHAN_ACTION=stop (boxes p01 and full) is
# the label rule with the reason "watchdog: box controller heartbeat stale"; alert (the smoke box, whose fault test
# freezes the heartbeat on purpose) only appends {"wall", "kind": "orphan_alert", "hb", "age_s", "limit_s"} to
# $KITSUNE_STATE/watchdog_alerts.jsonl, never syncs or stops, and re-arms once the file is fresh again. The defaults
# (label_hb, stop) keep the label box exactly as it was.
#
# Chain box (contract addendum E.5; p01-chain): the env is stage 1's (alert 600). Every poll also reads the mode file
# $KITSUNE_STATE/watchdog_mode ("<stop|alert> <orphan_s>"), which the chain controller writes once its gate part has
# ended (stop 3600); a valid one overrides the env, an empty or malformed one is ignored. With
# KITSUNE_WATCHDOG_HANDOVER_S > 0 (stage 1) and no mode file by first boot + that, a hung stage-1 bootstrap or
# controller can no longer hold the box: sync, then stop, whatever the heartbeat says. Halt retry (KITSUNE_JOB=full,
# not KITSUNE_NO_SELF_STOP=1): a halt marker written during this container's life and older than HALT_RETRY_S means the
# stop or destroy that wrote it did not take: the stop is requested again every RETRY_S.
#
# Usage: vast/watchdog.sh            (loop until the deadline)
#        vast/watchdog.sh --dry-run  (print the deadline and the planned actions, then exit)
set -euo pipefail

KITSUNE_DIR="${KITSUNE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STATE="${KITSUNE_STATE:-/workspace/kitsune_state}"
MAX_HOURS="${KITSUNE_MAX_HOURS:-5.5}"
SYNC_LEAD_S="${KITSUNE_WATCHDOG_SYNC_LEAD_S:-600}"
POLL_S="${KITSUNE_WATCHDOG_POLL_S:-60}"
RETRY_S=300
ORPHAN_S="${KITSUNE_WATCHDOG_ORPHAN_S:-0}"
HB_NAME="${KITSUNE_WATCHDOG_HB_FILE:-label_hb}"
HB="$STATE/$HB_NAME"
ACTION="${KITSUNE_WATCHDOG_ORPHAN_ACTION:-stop}"
MODE_FILE="$STATE/watchdog_mode"                 # "<stop|alert> <orphan_s>", written by the chain controller
HANDOVER_S="${KITSUNE_WATCHDOG_HANDOVER_S:-0}"   # chain stage 1: the gate part must hand over by first boot + this
HALT_RETRY_S="${KITSUNE_WATCHDOG_HALT_RETRY_S:-1200}"   # a halt marker this old with the instance still up: stop again
WD_START=$(date +%s)
PY="${KITSUNE_PY:-/venv/main/bin/python}"
[ -x "$PY" ] || PY="$(command -v "$PY" || command -v python3 || command -v python)"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

log() { printf '%s [watchdog] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
utc() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "@$1"; }

max_s=$(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }')
if ! [[ "$ORPHAN_S" =~ ^[0-9]+$ ]]; then
    log "KITSUNE_WATCHDOG_ORPHAN_S must be a whole number of seconds, got '$ORPHAN_S': orphan rule off"
    ORPHAN_S=0
fi
if [ "$ACTION" != stop ] && [ "$ACTION" != alert ]; then
    log "KITSUNE_WATCHDOG_ORPHAN_ACTION must be stop or alert, got '$ACTION': stop"
    ACTION=stop
fi
[[ "$HALT_RETRY_S" =~ ^[0-9]+$ ]] || HALT_RETRY_S=1200
if ! [[ "$HANDOVER_S" =~ ^[0-9]+$ ]]; then
    log "KITSUNE_WATCHDOG_HANDOVER_S must be a whole number of seconds, got '$HANDOVER_S': stage-1 rule off"
    HANDOVER_S=0
fi
if [ "$HB_NAME" = label_hb ]; then
    WHO="label controller"
    REASON="watchdog: label controller dead"
else
    WHO="box controller"
    REASON="watchdog: box controller heartbeat stale"
fi
mkdir -p "$STATE"
if [ -s "$STATE/deadline" ]; then
    deadline=$(cat "$STATE/deadline")
else
    deadline=$(( $(date +%s) + max_s ))
    [ "$DRY" = 1 ] || echo "$deadline" > "$STATE/deadline"
fi
sync_at=$(( deadline - SYNC_LEAD_S ))
log "cap ${MAX_HOURS} h: log sync at $(utc "$sync_at"), stop at $(utc "$deadline") ($(( (deadline - $(date +%s)) / 60 )) min from now)"
FIRST_BOOT=$(cat "$STATE/first_boot" 2>/dev/null) || FIRST_BOOT=""
if ! [[ "$FIRST_BOOT" =~ ^[0-9]+$ ]]; then
    FIRST_BOOT=$(( deadline - max_s ))
    [ "$HANDOVER_S" = 0 ] || log "no first_boot: using deadline - cap"
fi
synced=0; orphaned=0; alerted=0; bad_mode=0; moded=0; s1_over=0; halt_retry=0
read_mode() {  # a valid mode file overrides the env's action and limit (the chain controller writes it)
    local a="" s=""
    [ -s "$MODE_FILE" ] && { read -r a s < "$MODE_FILE" || true; }
    s=${s%$'\r'}  # a mode file written with CRLF
    if [[ "$a" =~ ^(stop|alert)$ && "$s" =~ ^[0-9]+$ ]]; then
        moded=1
        if [ "$a" != "$ACTION" ] || [ "$s" != "$ORPHAN_S" ]; then
            log "mode file: $a $s (was $ACTION $ORPHAN_S)"; ACTION=$a; ORPHAN_S=$s; alerted=0
        fi
    elif [ -n "$a$s" ] && [ "$bad_mode" = 0 ]; then bad_mode=1; log "ignoring malformed $MODE_FILE"; fi
}

if [ "$DRY" = 1 ]; then
    log "dry run: would run '$PY $KITSUNE_DIR/vast/finish.py --sync-only' at $(utc "$sync_at")"
    log "dry run: would run '$PY $KITSUNE_DIR/vast/finish.py --stop --no-sync' at $(utc "$deadline"), then retry every ${RETRY_S} s"
    if [ "$HANDOVER_S" -gt 0 ]; then
        log "dry run: mode file $MODE_FILE (\"<stop|alert> <orphan_s>\", the chain controller's) overrides the" \
            "heartbeat rule's action and limit on every poll"
        log "dry run: chain stage 1: with no mode file by first boot + ${HANDOVER_S} s" \
            "($(utc $(( FIRST_BOOT + HANDOVER_S )))), would run '$PY $KITSUNE_DIR/vast/finish.py --sync-only', then" \
            "stop the instance, whatever the heartbeat says"
    fi
    if [ "${KITSUNE_JOB:-}" = full ] && [ "${KITSUNE_NO_SELF_STOP:-0}" != 1 ]; then
        log "dry run: halt retry: a halt marker written after this start and older than ${HALT_RETRY_S} s: would run" \
            "'$PY $KITSUNE_DIR/vast/finish.py --stop --no-sync' every ${RETRY_S} s"
    fi
    if [ "$ORPHAN_S" -gt 0 ] && [ "$ACTION" = alert ]; then
        log "dry run: orphan rule: if $HB is stale by ${ORPHAN_S} s (never touched since this start:" \
            "$(( 2 * ORPHAN_S )) s), would append an orphan_alert to $STATE/watchdog_alerts.jsonl (action alert:" \
            "no sync, no stop)"
    elif [ "$ORPHAN_S" -gt 0 ]; then
        log "dry run: orphan rule: if $HB is stale by ${ORPHAN_S} s (never touched since this start:" \
            "$(( 2 * ORPHAN_S )) s), would run '$PY $KITSUNE_DIR/vast/finish.py --sync-only', then stop the instance"
    fi
    exit 0
fi

# one watchdog per container (onstart runs again after a restart)
if command -v flock >/dev/null; then
    exec 9>"$STATE/watchdog.lock"
    flock -n 9 || { log "another watchdog holds $STATE/watchdog.lock; exiting"; exit 0; }
fi

stop_now() {  # $1 = reason (default: the cap)
    if "$PY" "$KITSUNE_DIR/vast/finish.py" --stop --no-sync --reason "${1:-watchdog: ${MAX_HOURS} h cap reached}"; then
        return 0
    fi
    # last resort without python: same REST call, key passed through curl's stdin config so it never shows in argv; a
    # 2xx reply can still refuse ({"success": false}, as finish.py vast_rest reads it)
    if [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
        local r=""
        r=$(printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" \
            | curl -fsS --max-time 30 --config - -X PUT -H 'Content-Type: application/json' \
                -d '{"state": "stopped"}' "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/") \
            && ! [[ $r =~ \"success\"[[:space:]]*:[[:space:]]*false ]] && return 0
        log "vast REST stop failed: ${r:-no reply}"
    fi
    return 1
}

while :; do
    now=$(date +%s)
    read_mode
    # chain stage 1 (HANDOVER_S > 0): no mode file by first boot + HANDOVER_S -> sync, then the stop loop below
    if [ "$HANDOVER_S" -gt 0 ] && [ "$moded" = 0 ] && [ "$s1_over" = 0 ] && [ "$orphaned" = 0 ] \
        && [ ! -e "$STATE/halt" ] && [ "$now" -gt $(( FIRST_BOOT + HANDOVER_S )) ]; then
        s1_over=1
        log "chain stage 1 did not hand over by first boot + $HANDOVER_S s"
        timeout 1800 "$PY" "$KITSUNE_DIR/vast/finish.py" --sync-only || log "sync failed or timed out"
        orphaned=1
        REASON="watchdog: chain stage 1 over its sub-deadline"
    fi
    # halt retry: finish wrote the marker during this container's life, and the instance is still up long after
    if [ "$halt_retry" = 0 ] && [ "${KITSUNE_JOB:-}" = full ] && [ "${KITSUNE_NO_SELF_STOP:-0}" != 1 ] \
        && [ -e "$STATE/halt" ]; then
        hm=$(stat -c %Y "$STATE/halt" 2>/dev/null || echo 0)
        if [ "$hm" -ge "$WD_START" ] && [ $(( now - hm )) -gt "$HALT_RETRY_S" ]; then
            halt_retry=1
            REASON="watchdog: halt marker $(( now - hm )) s old, instance still up"
        fi
    fi
    if [ "$halt_retry" = 1 ]; then
        stop_now "$REASON" && log "stop requested" || log "stop request failed"
        sleep "$RETRY_S"
        continue
    fi
    if [ "$orphaned" = 0 ] && [ "$ORPHAN_S" -gt 0 ] && [ ! -e "$STATE/halt" ]; then
        hb=$(stat -c %Y "$HB" 2>/dev/null || echo 0)
        stale=""
        if [ "$hb" -ge "$WD_START" ] && [ $(( now - hb )) -gt "$ORPHAN_S" ]; then
            age=$(( now - hb ))
            limit=$ORPHAN_S
            stale="$WHO heartbeat stale for $age s (limit ${ORPHAN_S} s)"
        elif [ "$hb" -lt "$WD_START" ] && [ $(( now - WD_START )) -gt $(( 2 * ORPHAN_S )) ]; then
            age=$(( now - WD_START ))
            limit=$(( 2 * ORPHAN_S ))
            stale="$WHO never touched $HB in $age s since the watchdog started (limit $limit s)"
        fi
        if [ -n "$stale" ] && [ "$ACTION" = alert ]; then
            if [ "$alerted" = 0 ]; then  # one record per stale spell; nothing is synced or stopped
                alerted=1
                log "ALERT: $stale (action alert: recorded in $STATE/watchdog_alerts.jsonl, no sync, no stop)"
                printf '{"wall": %s, "kind": "orphan_alert", "hb": "%s", "age_s": %s, "limit_s": %s}\n' \
                    "$now" "$HB_NAME" "$age" "$limit" >> "$STATE/watchdog_alerts.jsonl" || log "cannot write the alert"
            fi
        elif [ -n "$stale" ]; then
            orphaned=1
            log "$stale: the controller is dead"
            log "best-effort ${WHO%% *} sync, then stop"
            timeout 1800 "$PY" "$KITSUNE_DIR/vast/finish.py" --sync-only || log "sync failed or timed out"
        elif [ "$alerted" = 1 ]; then
            alerted=0
            log "$WHO heartbeat fresh again: the alert re-armed"
        fi
    fi
    if [ "$orphaned" = 1 ]; then
        stop_now "$REASON" && log "stop requested" || log "stop request failed"
        sleep "$RETRY_S"
        continue
    fi
    if [ "$now" -ge "$deadline" ]; then
        log "deadline reached; stopping the instance"
        stop_now && log "stop requested" || log "stop request failed"
        sleep "$RETRY_S"
        continue
    fi
    if [ "$synced" = 0 ] && [ "$now" -ge "$sync_at" ]; then
        synced=1
        log "deadline in $(( (deadline - now) / 60 )) min: best-effort log sync"
        timeout $(( deadline - now )) "$PY" "$KITSUNE_DIR/vast/finish.py" --sync-only || log "sync failed or timed out"
        continue
    fi
    sleep "$POLL_S"
done
