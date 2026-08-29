#!/usr/bin/env bash
# ============================================================
# CipherMind Sentinel — stop the local stack (Linux/macOS)
# Stops the soc-engine (:3010) and the Next.js dev server (:3000).
# ============================================================
set -u
echo "stopping soc-engine ..."
pkill -f "bun --hot src/index.ts" 2>/dev/null || true
pkill -f "bun src/index.ts" 2>/dev/null || true
echo "stopping Next.js dev server ..."
pkill -f "next dev" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
sleep 1
echo "done."
