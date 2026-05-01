#!/usr/bin/env python3
"""Benchmark PI0 sampling with TensorRT for the repeated denoise_step subgraph.

This keeps preprocessing, prefix embedding, and prefix KV cache generation in
PyTorch, then runs the 10 denoising iterations through a fixed-shape TensorRT
engine exported by scripts/export_denoise_step_onnx.py.
"""

import dataclasses
import time

import jax
import numpy as np
from openpi.models import model as model_lib
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import config as config_lib
import tensorrt as trt
import torch
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_aloha_sim"
    checkpoint: str = "/workspace/openpi_data/openpi-assets/checkpoints/pi0_aloha_sim_pytorch"
    engine: str = "/workspace/openpi_artifacts/pi0_aloha_sim_denoise_step_bf16.engine"
    device: str = "cuda"
    num_steps: int = 10
    warmup: int = 3
    iterations: int = 20


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


def _torch_dtype(dtype: trt.DataType):
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.bfloat16:
        return torch.bfloat16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(dtype)


class TrtDenoiseStep:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.output_name = "v_t"
        self._static_buffers = {}
        self._past_names = [f"past_{i}" for i in range(36)]
        self._last_inputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
                self._static_buffers[name] = torch.empty(shape, device="cuda", dtype=dtype)
                self.context.set_tensor_address(name, self._static_buffers[name].data_ptr())

    def __call__(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        legacy_cache = past_key_values.to_legacy_cache()
        past_flat = [tensor for pair in legacy_cache for tensor in pair]
        inputs = {
            "state": state.float().contiguous(),
            "prefix_pad_masks": prefix_pad_masks.contiguous(),
            "x_t": x_t.float().contiguous(),
            "timestep": timestep.float().contiguous(),
        }
        inputs.update({name: tensor.contiguous() for name, tensor in zip(self._past_names, past_flat, strict=True)})
        self._last_inputs = inputs
        for name, tensor in self._last_inputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT denoise_step execution failed")
        return self._static_buffers[self.output_name]


def _make_observation(policy, device: str):
    inputs = policy._input_transform(jax.tree.map(lambda x: x, _raw_aloha_obs()))  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    return model_lib.Observation.from_dict(inputs)


def _prepare_prefix(model, observation):
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
    return state, prefix_pad_masks, past_key_values


def _sample_with_trt(model, trt_denoise, state, prefix_pad_masks, past_key_values, noise, num_steps: int):
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
    x_t = noise
    time_value = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
    while time_value >= -dt / 2:
        timestep = time_value.expand(noise.shape[0])
        v_t = trt_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep)
        x_t = x_t + dt * v_t
        time_value += dt
    return x_t


def main(args: Args) -> None:
    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    model = policy._model.eval()  # noqa: SLF001
    trt_denoise = TrtDenoiseStep(args.engine)
    observation = _make_observation(policy, args.device)

    with torch.no_grad():
        state, prefix_pad_masks, past_key_values = _prepare_prefix(model, observation)
        noise = model.sample_noise((1, config.model.action_horizon, config.model.action_dim), args.device)
        timestep = torch.tensor([1.0], device=args.device, dtype=torch.float32)
        torch_ref = model.denoise_step(state, prefix_pad_masks, past_key_values, noise, timestep)
        trt_ref = trt_denoise(state, prefix_pad_masks, past_key_values, noise, timestep)
        torch.cuda.synchronize()
        max_abs_diff = (torch_ref - trt_ref).abs().max().item()
        mean_abs_diff = (torch_ref - trt_ref).abs().mean().item()
        print(f"single_step_diff max_abs={max_abs_diff:.6f} mean_abs={mean_abs_diff:.6f}", flush=True)

        for _ in range(args.warmup):
            _sample_with_trt(model, trt_denoise, state, prefix_pad_masks, past_key_values, noise, args.num_steps)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            _sample_with_trt(model, trt_denoise, state, prefix_pad_masks, past_key_values, noise, args.num_steps)
        end.record()
        torch.cuda.synchronize()
        trt_loop_ms = start.elapsed_time(end) / args.iterations
        print(f"trt_denoise_loop_ms={trt_loop_ms:.4f} num_steps={args.num_steps} iterations={args.iterations}", flush=True)

        # Measure PyTorch denoise loop with the same already-computed prefix cache.
        for _ in range(args.warmup):
            x_t = noise
            dt = torch.tensor(-1.0 / args.num_steps, dtype=torch.float32, device=args.device)
            time_value = torch.tensor(1.0, dtype=torch.float32, device=args.device)
            while time_value >= -dt / 2:
                x_t = x_t + dt * model.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, time_value.expand(noise.shape[0])
                )
                time_value += dt
        torch.cuda.synchronize()
        start.record()
        for _ in range(args.iterations):
            x_t = noise
            dt = torch.tensor(-1.0 / args.num_steps, dtype=torch.float32, device=args.device)
            time_value = torch.tensor(1.0, dtype=torch.float32, device=args.device)
            while time_value >= -dt / 2:
                x_t = x_t + dt * model.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, time_value.expand(noise.shape[0])
                )
                time_value += dt
        end.record()
        torch.cuda.synchronize()
        torch_loop_ms = start.elapsed_time(end) / args.iterations
        print(f"torch_denoise_loop_ms={torch_loop_ms:.4f} num_steps={args.num_steps} iterations={args.iterations}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
