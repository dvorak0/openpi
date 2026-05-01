#!/usr/bin/env python3
"""Run a minimal OpenPI PyTorch checkpoint inference smoke test."""

import dataclasses
import time

import numpy as np
from openpi.policies import policy_config
from openpi.training import config as train_config
import torch
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_aloha_sim"
    checkpoint: str = "/workspace/openpi_data/openpi-assets/checkpoints/pi0_aloha_sim_pytorch"
    env: str = "ALOHA_SIM"
    num_steps: int = 5
    device: str = "cuda"
    compile_mode: str | None = None


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
    raise ValueError(f"Unsupported env for PyTorch smoke test: {env}")


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args: Args) -> None:
    config = train_config.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, pytorch_compile_mode=args.compile_mode),
    )
    print(f"compile_mode={config.model.pytorch_compile_mode}", flush=True)

    start = time.time()
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    print(f"policy_loaded_sec={time.time() - start:.2f}", flush=True)

    obs_fn = _observation_fn(args.env)
    for i in range(args.num_steps):
        _sync(args.device)
        infer_start = time.time()
        result = policy.infer(obs_fn())
        _sync(args.device)
        actions = result["actions"]
        memory = ""
        if args.device.startswith("cuda") and torch.cuda.is_available():
            memory = (
                f" mem_alloc={torch.cuda.memory_allocated() / 1024**3:.2f}GiB"
                f" mem_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GiB"
            )
        print(
            f"infer_{i}_sec={time.time() - infer_start:.3f} "
            f"actions_shape={actions.shape} dtype={actions.dtype} "
            f"min={actions.min():.4f} max={actions.max():.4f}{memory}",
            flush=True,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
