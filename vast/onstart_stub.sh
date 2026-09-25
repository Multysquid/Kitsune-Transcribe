#!/bin/bash
# vast.ai on-start stub: what vast/launch.py passes with `vastai create instance --onstart`. It must stay small (the
# API's onstart field may be capped near 4 KB; launch.py refuses above 4000 bytes) and ASCII-only. It clones the repo
# at $KITSUNE_SHA and hands over to vast/onstart.sh, which does everything else (watchdog, bootstrap, supervisor).
# Until then no watchdog runs, so if the clone fails the instance is stopped here through the vast REST API (the
# per-instance key vast injects), unless KITSUNE_NO_SELF_STOP=1. Runs at every container start. Every command that may
# fail sits in an && / || list, so errexit never skips the stop.
set -euo pipefail
D="${KITSUNE_DIR:-/workspace/Kitsune-Transcribe}"
URL="${KITSUNE_REPO_URL:-https://github.com/Multysquid/Kitsune-Transcribe}"
L="${KITSUNE_LOG:-/workspace/kitsune.log}"
mkdir -p "$(dirname "$L")"
exec >>"$L" 2>&1

log() { printf '%s [stub] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

stop_box() {  # $1 = reason; the key goes through curl's stdin config, never argv
    log "$1"
    if [ "${KITSUNE_NO_SELF_STOP:-0}" = 1 ]; then
        log "KITSUNE_NO_SELF_STOP=1: leaving the instance running"
    elif [ -n "${CONTAINER_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
        printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" \
            | curl -fsS --retry 3 --max-time 30 --config - -X PUT -H 'Content-Type: application/json' \
                -d '{"state": "stopped"}' "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" >/dev/null \
            && log "stop requested via REST" || log "stop request FAILED: destroy the instance by hand"
    else
        log "no per-instance vast key in the env: destroy the instance by hand"
    fi
    exit 1
}

if [ "$(git -C "$D" rev-parse HEAD 2>/dev/null)" != "${KITSUNE_SHA:-}" ]; then
    [[ "${KITSUNE_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || stop_box "KITSUNE_SHA must be a full 40-hex commit SHA, got '${KITSUNE_SHA:-}'"
    mkdir -p "$D" || stop_box "cannot create $D"
    git -C "$D" init -q || stop_box "git init failed in $D"
    git -C "$D" remote remove origin 2>/dev/null || true
    git -C "$D" remote add origin "$URL" || stop_box "git remote add failed in $D"
    ok=0
    for i in 1 2 3; do
        timeout 300 git -C "$D" fetch -q --depth 1 origin "$KITSUNE_SHA" && ok=1 && break
        log "git fetch attempt $i failed"
        sleep $(( i * 15 ))
    done
    [ "$ok" = 1 ] && git -C "$D" checkout -q --force --detach "$KITSUNE_SHA" \
        || stop_box "could not clone $URL at $KITSUNE_SHA"
    log "cloned $URL at $KITSUNE_SHA"
fi
[ -f "$D/vast/onstart.sh" ] || stop_box "$D/vast/onstart.sh missing at $KITSUNE_SHA"
exec bash "$D/vast/onstart.sh" "$@"
