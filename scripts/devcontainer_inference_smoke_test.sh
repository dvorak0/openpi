#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -x /.venv/bin/python ]]; then
  echo "Expected /.venv/bin/python from .devcontainer/Dockerfile" >&2
  exit 1
fi

ENV_NAME="${OPENPI_SMOKE_ENV:-ALOHA_SIM}"
PORT="${OPENPI_SMOKE_PORT:-8000}"
NUM_STEPS="${OPENPI_SMOKE_NUM_STEPS:-1}"
LOG_DIR="${OPENPI_SMOKE_LOG_DIR:-/tmp/openpi-inference-smoke}"
CLIENT_TIMEOUT="${OPENPI_SMOKE_CLIENT_TIMEOUT:-1800}"
SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"

mkdir -p "$LOG_DIR"
rm -f "$SERVER_LOG" "$CLIENT_LOG"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export PYTHONPATH="${PYTHONPATH:-/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src}"

/.venv/bin/python scripts/serve_policy.py --env "$ENV_NAME" --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server exited before becoming ready. Log:" >&2
    cat "$SERVER_LOG" >&2
    exit 1
  fi
  if grep -q "Starting server" "$SERVER_LOG" || grep -q "Uvicorn running" "$SERVER_LOG" || grep -q "server listening" "$SERVER_LOG"; then
    break
  fi
  if /.venv/bin/python - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(0.2)
s.connect(("127.0.0.1", int("$PORT")))
s.close()
PY
  then
    break
  fi
  sleep 2
done

if ! timeout "$CLIENT_TIMEOUT" /.venv/bin/python examples/simple_client/main.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --env "$ENV_NAME" \
  --num-steps "$NUM_STEPS" >"$CLIENT_LOG" 2>&1; then
  echo "Inference client failed or timed out. Server log:" >&2
  cat "$SERVER_LOG" >&2
  echo "Client log:" >&2
  cat "$CLIENT_LOG" >&2
  exit 1
fi

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Policy server exited during inference. Server log:" >&2
  cat "$SERVER_LOG" >&2
  exit 1
fi

cat "$CLIENT_LOG"
