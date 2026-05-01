#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${OPENPI_DEV_IMAGE:-openpi-dev:local}"
GPU_DEVICE_FLAG="${OPENPI_DOCKER_GPU_FLAG:---device=nvidia.com/gpu=all}"
OPENPI_DATA_HOME_HOST="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

mkdir -p "$OPENPI_DATA_HOME_HOST"

if [[ $# -eq 0 ]]; then
  set -- bash
fi

TTY_ARGS=(-i)
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
  "$GPU_DEVICE_FLAG" \
  --network host \
  -v "$PWD:/workspace/openpi" \
  -v "$OPENPI_DATA_HOME_HOST:/openpi_assets" \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  -e PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
  -e OPENPI_SMOKE_ENV="${OPENPI_SMOKE_ENV:-}" \
  -e OPENPI_SMOKE_PORT="${OPENPI_SMOKE_PORT:-}" \
  -e OPENPI_SMOKE_NUM_STEPS="${OPENPI_SMOKE_NUM_STEPS:-}" \
  -e OPENPI_SMOKE_LOG_DIR="${OPENPI_SMOKE_LOG_DIR:-}" \
  -e OPENPI_SMOKE_CLIENT_TIMEOUT="${OPENPI_SMOKE_CLIENT_TIMEOUT:-}" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-}" \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" \
  "$IMAGE" "$@"
