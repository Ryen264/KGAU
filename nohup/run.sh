#!/usr/bin/env bash
set -euo pipefail

# Usage: nohup/run [config_path]
# Example: nohup/run config/config_wn18rr.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-config/config_wn18rr.yaml}"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
RUN_DIR="$ROOT_DIR/nohup"
OUT_LOG="$RUN_DIR/nohup.out"
PID_FILE="$RUN_DIR/process.pid"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Error: $VENV_PYTHON not found. Create venv first: python3 -m venv .venv"
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Process already running with PID $OLD_PID"
    exit 1
  fi
fi

mkdir -p "$RUN_DIR"
cd "$ROOT_DIR"
nohup "$VENV_PYTHON" main.py "$CONFIG_PATH" > "$OUT_LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "Started PID $NEW_PID"
echo "Config: $CONFIG_PATH"
echo "Nohup log: $OUT_LOG"
