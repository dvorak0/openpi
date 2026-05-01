#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -x /.venv/bin/python ]]; then
  echo "Expected /.venv/bin/python from .devcontainer/Dockerfile" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src}"

/.venv/bin/python - <<'PY'
import importlib.util
import sys

required = ["openpi", "openpi_client", "jax", "torch", "transformers"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required modules: {missing}")

import jax
import torch

print(f"python={sys.version.split()[0]}")
print(f"jax={jax.__version__} devices={jax.devices()}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

/.venv/bin/python -m pytest \
  src/openpi/transforms_test.py \
  packages/openpi-client/src/openpi_client/ \
  "$@"
