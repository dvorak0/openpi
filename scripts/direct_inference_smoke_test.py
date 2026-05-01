#!/usr/bin/env python3
"""Run a minimal in-process OpenPI inference smoke test.

This avoids websocket keepalive timeouts during the first JAX compile, which can
be much slower than subsequent inference calls.
"""

import dataclasses
import os
import time
from typing import Literal

import numpy as np
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as train_config
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_aloha_sim"
    checkpoint: str = "gs://openpi-assets/checkpoints/pi0_aloha_sim"
    env: Literal["ALOHA_SIM"] = "ALOHA_SIM"
    num_steps: int = 3
    default_prompt: str | None = None


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _observation_fn(env: str):
    if env == "ALOHA_SIM":
        return _random_observation_aloha
    raise ValueError(f"Unsupported env for direct inference smoke test: {env}")


def main(args: Args) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    start = time.time()
    config = train_config.get_config(args.config_name)
    checkpoint_dir = download.maybe_download(args.checkpoint)
    print(f"checkpoint_dir={checkpoint_dir}", flush=True)

    policy = policy_config.create_trained_policy(config, checkpoint_dir, default_prompt=args.default_prompt)
    print(f"policy_loaded_sec={time.time() - start:.2f}", flush=True)

    obs_fn = _observation_fn(args.env)
    for i in range(args.num_steps):
        infer_start = time.time()
        result = policy.infer(obs_fn())
        actions = result["actions"]
        print(
            f"infer_{i}_sec={time.time() - infer_start:.2f} "
            f"actions_shape={actions.shape} dtype={actions.dtype} "
            f"min={actions.min():.4f} max={actions.max():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
