#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/nohup/process.pid"
OUT_LOG="$ROOT_DIR/nohup/nohup.out"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "RUNNING: PID $PID"
else
  echo "NOT RUNNING: stale PID file with PID $PID"
fi

if [[ -f "$OUT_LOG" ]]; then
  echo "--- Last 30 lines of nohup log ---"
  tail -n 30 "$OUT_LOG"
else
  echo "No nohup log found at $OUT_LOG"
fi
