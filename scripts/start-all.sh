#!/usr/bin/env bash
# ============================================================
# CipherMind Sentinel — start the full stack locally (Linux/macOS)
#   1. soc-engine  : ML inference engine  -> http://localhost:3010
#   2. frontend    : Next.js SOC UI       -> http://localhost:3000
#
# Requirements: Bun (https://bun.sh). First run installs dependencies.
# Stop everything with:  scripts/stop-all.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

command -v bun >/dev/null 2>&1 || { echo "ERROR: bun is not installed — see https://bun.sh"; exit 1; }

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[env] created .env from .env.example (NEXT_PUBLIC_ENGINE_URL=http://localhost:3010)"
fi

echo "== CipherMind Sentinel — starting =="

# ---------------------------------------------------------- soc-engine :3010
if curl -s -m 3 http://localhost:3010/api/health > /dev/null 2>&1; then
  echo "[engine]   already running on :3010"
else
  if [ ! -d mini-services/soc-engine/node_modules ]; then
    echo "[engine]   installing dependencies (first run) ..."
    (cd mini-services/soc-engine && bun install)
  fi
  echo "[engine]   starting soc-engine on :3010 ..."
  (cd mini-services/soc-engine && nohup bun run dev > engine.log 2>&1 &)
fi

# ---------------------------------------------------------- frontend  :3000
if curl -s -m 3 -o /dev/null http://localhost:3000/ 2>/dev/null; then
  echo "[frontend] already running on :3000"
else
  if [ ! -d node_modules ]; then
    echo "[frontend] installing dependencies (first run — this can take a few minutes) ..."
    bun install
  fi
  echo "[frontend] starting Next.js dev server on :3000 ..."
  nohup bun run dev > frontend.log 2>&1 &
fi

# ---------------------------------------------------------- wait for engine
echo "[wait]     engine boot takes ~20-60s (it scores 12,000 boot events through the real models)"
ENGINE_UP=0
for _ in $(seq 1 45); do
  sleep 2
  if curl -s -m 3 http://localhost:3010/api/health > /dev/null 2>&1; then ENGINE_UP=1; break; fi
done
[ "$ENGINE_UP" = "1" ] && echo "[engine]   UP" || echo "[engine]   STILL BOOTING — check mini-services/soc-engine/engine.log"

echo ""
echo "============================================================"
echo "  Frontend (SOC UI) : http://localhost:3000"
echo "  Engine health     : http://localhost:3010/api/health"
echo "  Stop the stack    : scripts/stop-all.sh"
echo "============================================================"
