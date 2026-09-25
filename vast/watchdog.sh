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
# Label box only (KITSUNE_WATCHDOG_ORPHAN_S > 0; launch.py --job label sets 900): the controller vast/label.py touches
# $KITSUNE_STATE/label_hb at least every 30 s. If it goes stale by ORPHAN_S (label.py crashed, was OOM-killed or hangs),
# nothing would end the box before the cap, so the watchdog runs finish.py --sync-only (the labels on disk get to the
# Hub) and then stops the instance, retrying the stop like at the deadline. The rule arms only once this boot's
# controller has touched the file (a heartbeat older than the watchdog's own start is a previous boot's); if it never
# does, it fires at 2 x ORPHAN_S. A halt marker (finish.py has taken over) disables it. ORPHAN_S=0 (the train job)
# leaves the watchdog exactly as it was.
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
mkdir -p "$STATE"
if [ -s "$STATE/deadline" ]; then
    deadline=$(cat "$STATE/deadline")
else
    deadline=$(( $(date +%s) + max_s ))
    [ "$DRY" = 1 ] || echo "$deadline" > "$STATE/deadline"
fi
sync_at=$(( deadline - SYNC_LEAD_S ))
log "cap ${MAX_HOURS} h: log sync at $(utc "$sync_at"), stop at $(utc "$deadline") ($(( (deadline - $(date +%s)) / 60 )) min from now)"

if [ "$DRY" = 1 ]; then
    log "dry run: would run '$PY $KITSUNE_DIR/vast/finish.py --sync-only' at $(utc "$sync_at")"
    log "dry run: would run '$PY $KITSUNE_DIR/vast/finish.py --stop --no-sync' at $(utc "$deadline"), then retry every ${RETRY_S} s"
    if [ "$ORPHAN_S" -gt 0 ]; then
        log "dry run: orphan rule: if $STATE/label_hb is stale by ${ORPHAN_S} s (never touched since this start:" \
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

synced=0
orphaned=0
while :; do
    now=$(date +%s)
    if [ "$orphaned" = 0 ] && [ "$ORPHAN_S" -gt 0 ] && [ ! -e "$STATE/halt" ]; then
        hb=$(stat -c %Y "$STATE/label_hb" 2>/dev/null || echo 0)
        if [ "$hb" -ge "$WD_START" ] && [ $(( now - hb )) -gt "$ORPHAN_S" ]; then
            orphaned=1
            log "label controller heartbeat stale for $(( now - hb )) s (limit ${ORPHAN_S} s): the controller is dead"
        elif [ "$hb" -lt "$WD_START" ] && [ $(( now - WD_START )) -gt $(( 2 * ORPHAN_S )) ]; then
            orphaned=1
            log "label controller never touched $STATE/label_hb in $(( now - WD_START )) s since the watchdog" \
                "started (limit $(( 2 * ORPHAN_S )) s): the controller is dead"
        fi
        if [ "$orphaned" = 1 ]; then
            log "best-effort label sync, then stop"
            timeout 1800 "$PY" "$KITSUNE_DIR/vast/finish.py" --sync-only || log "sync failed or timed out"
        fi
    fi
    if [ "$orphaned" = 1 ]; then
        stop_now "watchdog: label controller dead" && log "stop requested" || log "stop request failed"
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
