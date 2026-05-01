#!/usr/bin/env python3
"""Export a fixed-shape PI0 PyTorch denoise_step subgraph to ONNX.

This is an experimental TensorRT bring-up script for pi0_aloha_sim. It exports
only one denoising step with a precomputed prefix KV cache, not the full policy.
"""

import dataclasses
import os
import time

import jax
import numpy as np
from openpi.models import model as model_lib
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import config as config_lib
import torch
from transformers.cache_utils import DynamicCache
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_aloha_sim"
    checkpoint: str = "/workspace/openpi_data/openpi-assets/checkpoints/pi0_aloha_sim_pytorch"
    output: str = "/workspace/openpi_artifacts/pi0_aloha_sim_denoise_step.onnx"
    device: str = "cuda"
    opset: int = 17


def _raw_aloha_obs() -> dict:
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


class DenoiseStepWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, prefix_pad_masks, x_t, timestep, *past_flat):
        legacy_cache = tuple((past_flat[i], past_flat[i + 1]) for i in range(0, len(past_flat), 2))
        past_key_values = DynamicCache.from_legacy_cache(legacy_cache)
        return self.model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)


def main(args: Args) -> None:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    model = policy._model.eval()  # noqa: SLF001

    inputs = policy._input_transform(jax.tree.map(lambda x: x, _raw_aloha_obs()))  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(args.device)[None, ...], inputs)
    observation = model_lib.Observation.from_dict(inputs)

    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = model._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int32), dim=1) - 1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        x_t = model.sample_noise((1, config.model.action_horizon, config.model.action_dim), args.device)
        timestep = torch.tensor([1.0], device=args.device, dtype=torch.float32)
        past_flat = [tensor for pair in past_key_values.to_legacy_cache() for tensor in pair]
        wrapper = DenoiseStepWrapper(model).eval()
        ref = wrapper(state, prefix_pad_masks, x_t, timestep, *past_flat)
        print(f"ref_shape={tuple(ref.shape)} ref_dtype={ref.dtype}", flush=True)
        input_names = ["state", "prefix_pad_masks", "x_t", "timestep"] + [f"past_{i}" for i in range(len(past_flat))]
        start = time.time()
        torch.onnx.export(
            wrapper,
            (state, prefix_pad_masks, x_t, timestep, *past_flat),
            args.output,
            input_names=input_names,
            output_names=["v_t"],
            opset_version=args.opset,
            do_constant_folding=True,
            external_data=True,
        )
        print(f"export_done_sec={time.time() - start:.2f} output={args.output}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
