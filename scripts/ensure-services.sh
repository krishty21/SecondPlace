#!/bin/bash
# CipherMind Sentinel — ensure all services are running (idempotent).
# Safe to call repeatedly (e.g., from cron or QA loops).
# Services are daemonized via scripts/daemonize.py (double-fork) so they
# survive bash-tool session reaping.

set -u
PROJECT=/home/z/my-project
DAEMONIZE="python3 $PROJECT/scripts/daemonize.py"

port_up() {
    curl -s -o /dev/null -m 3 "http://localhost:$1$2" && return 0 || return 1
}

echo "[ensure-services] checking at $(date '+%H:%M:%S') ..."

# ---- soc-engine (port 3010) -------------------------------------------------
if port_up 3010 /api/health; then
    echo "[ensure-services] soc-engine: UP"
else
    echo "[ensure-services] soc-engine: DOWN — starting"
    pkill -f "soc-engine/src/index.ts" 2>/dev/null || true
    sleep 1
    DAEMONIZE_CWD="$PROJECT/mini-services/soc-engine" \
        $DAEMONIZE "$PROJECT/ml/soc-engine.log" bun --hot src/index.ts
    # wait for boot (12k-event scoring takes ~15s)
    for i in $(seq 1 40); do
        sleep 2
        if port_up 3010 /api/health; then echo "[ensure-services] soc-engine: BOOTED (${i}x2s)"; break; fi
    done
fi

# ---- Next.js dev server (port 3000) ----------------------------------------
if port_up 3000 /; then
    echo "[ensure-services] next-dev: UP"
else
    echo "[ensure-services] next-dev: DOWN — starting"
    pkill -f "next dev" 2>/dev/null || true
    pkill -f "next-server" 2>/dev/null || true
    sleep 1
    DAEMONIZE_CWD="$PROJECT" \
        $DAEMONIZE "$PROJECT/dev.log" bun run dev
    for i in $(seq 1 45); do
        sleep 2
        if port_up 3000 /; then echo "[ensure-services] next-dev: UP (${i}x2s)"; break; fi
    done
fi

# ---- summary ----------------------------------------------------------------
S=down; N=down
port_up 3010 /api/health && S=up
port_up 3000 / && N=up
echo "[ensure-services] final state: soc-engine=$S next-dev=$N"
