# Development container for iterating on openpi with the source tree mounted.
#
# Build from the repository root:
#   docker build -f .devcontainer/Dockerfile -t openpi-dev:local .
#
# Run with the repository mounted:
#   docker run --rm -it --device nvidia.com/gpu=all --network host \
#     -v "$PWD:/workspace/openpi" \
#     -v "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}:/openpi_assets" \
#     -e OPENPI_DATA_HOME=/openpi_assets \
#     openpi-dev:local bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV OPENPI_DATA_HOME=/openpi_assets
ENV IS_DOCKER=true
ENV PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src

WORKDIR /workspace/openpi

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    clang \
    curl \
    git \
    git-lfs \
    linux-headers-generic \
    openssh-client \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv --python 3.11.9 "$UV_PROJECT_ENVIRONMENT"

# Pre-install dependencies using only files that affect dependency resolution.
# The source tree is mounted at runtime, so ordinary code edits do not require
# rebuilding this image.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project

# Keep the patched transformer files aligned with the dependency environment,
# matching scripts/docker/serve_policy.Dockerfile.
RUN --mount=type=bind,source=src/openpi/models_pytorch/transformers_replace,target=/tmp/transformers_replace,readonly \
    /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {}

CMD ["bash"]
