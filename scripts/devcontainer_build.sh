#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${OPENPI_DEV_IMAGE:-openpi-dev:local}"

DOCKER_BUILDKIT=1 docker build \
  -f .devcontainer/Dockerfile \
  -t "$IMAGE" \
  .
