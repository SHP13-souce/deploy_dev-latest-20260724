#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHM_NAME="${DEBUG_WEB_SHM:-/hnu_vision_debug}"
SAFE_SHM="${SHM_NAME#/}"
SAFE_SHM="${SAFE_SHM//\//_}"
HOST="${DEBUG_WEB_HOST:-127.0.0.1}"
PORT="${DEBUG_WEB_PORT:-5000}"
PYTHON="${DEBUG_WEB_PYTHON:-python3}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <vision-command> [args...]" >&2
  exit 2
fi

cleanup() {
  trap - EXIT INT TERM
  [[ -n "${VISION_PID:-}" ]] && kill "$VISION_PID" 2>/dev/null || true
  [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
  [[ -n "${VISION_PID:-}" ]] && wait "$VISION_PID" 2>/dev/null || true
  [[ -n "${WEB_PID:-}" ]] && wait "$WEB_PID" 2>/dev/null || true
  rm -f "/dev/shm/$SAFE_SHM" "/tmp/$SAFE_SHM.json" "/tmp/$SAFE_SHM.json.tmp."*
  rm -f "/dev/shm/${SAFE_SHM}_annotated" "/dev/shm/${SAFE_SHM}_gray" "/dev/shm/${SAFE_SHM}_binary"
  rm -f "/tmp/${SAFE_SHM}_annotated.json" "/tmp/${SAFE_SHM}_gray.json" "/tmp/${SAFE_SHM}_binary.json"
  rm -f "/tmp/$SAFE_SHM.tuning_request.yaml"
  rm -f "/tmp/$SAFE_SHM.tuning_state.json"
}
trap cleanup EXIT INT TERM

"$PYTHON" "$ROOT/debug_web/app.py" --shm-name "$SHM_NAME" --host "$HOST" --port "$PORT" &
WEB_PID=$!
"$@" &
VISION_PID=$!
echo "Debug Web: http://$HOST:$PORT (web=$WEB_PID vision=$VISION_PID)"
wait "$VISION_PID"
