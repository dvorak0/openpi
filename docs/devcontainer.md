# Development Container Bring-up

This repo includes a lightweight devcontainer-style image for editing on one
machine and testing on a Docker/NVIDIA host with the source tree bind-mounted.
It is intended for workflows such as testing on `nixos-1` under
`/home/dvorak/workspace/openpi` while keeping normal code edits in the git tree.

## Build

From the repository root:

```bash
scripts/devcontainer_build.sh
```

Equivalent raw Docker command:

```bash
docker build -f .devcontainer/Dockerfile -t openpi-dev:local .
```

## Run with mounted source

```bash
scripts/devcontainer_run.sh
```

Or run a command directly inside the mounted-source container:

```bash
scripts/devcontainer_run.sh scripts/devcontainer_smoke_test.sh
```

Equivalent raw Docker command:

```bash
mkdir -p "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
docker run --rm -it --device nvidia.com/gpu=all --network host \
  -v "$PWD:/workspace/openpi" \
  -v "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}:/openpi_assets" \
  -e OPENPI_DATA_HOME=/openpi_assets \
  openpi-dev:local bash
```

The `--device nvidia.com/gpu=all` flag matches the CDI-style NVIDIA setup on
`nixos-1`. On hosts configured with the classic NVIDIA Docker runtime, replace it
with `--gpus all`.

The smoke test fails fast if the prebuilt virtualenv is missing, verifies that
core modules import, prints JAX/Torch device visibility, and runs a small pytest
subset that does not require downloading model checkpoints.

## Inference smoke tests

For a first model bring-up, prefer the direct in-process smoke test. It avoids
websocket keepalive timeouts during the first JAX compile, which can be much
slower than subsequent inference calls:

```bash
scripts/devcontainer_run.sh python scripts/direct_inference_smoke_test.py
```

After direct inference works, this command starts the policy server and queries
it with the simple client:

```bash
scripts/devcontainer_run.sh scripts/devcontainer_inference_smoke_test.sh
```

Defaults:

- direct smoke: `pi0_aloha_sim`, `ALOHA_SIM`, 3 inference calls
- websocket smoke: `OPENPI_SMOKE_ENV=ALOHA_SIM`
- websocket smoke: `OPENPI_SMOKE_PORT=8000`
- websocket smoke: `OPENPI_SMOKE_NUM_STEPS=1`

The first run downloads the selected checkpoint into `OPENPI_DATA_HOME`, so it
can take a while and requires enough disk/GPU memory for the selected model.

## VS Code / Dev Containers

The `.devcontainer/devcontainer.json` uses the same Dockerfile, mounts the repo
at `/workspace/openpi`, mounts `OPENPI_DATA_HOME` to `/openpi_assets`, and runs
`scripts/devcontainer_smoke_test.sh` after creation.
