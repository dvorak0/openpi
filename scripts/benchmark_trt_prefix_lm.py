#!/usr/bin/env python3
"""Benchmark TensorRT engines for the PI0 prefix-LM KV-cache fill.

The benchmark compares the PyTorch prefix LM path against one or more TensorRT
engines exported from precomputed prefix embeddings to KV cache tensors.
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
import tensorrt as trt
import torch
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_aloha_sim"
    checkpoint: str = "/workspace/openpi_data/openpi-assets/checkpoints/pi0_aloha_sim_pytorch"
    engines: tuple[str, ...] = (
        "/workspace/openpi_artifacts/pi0_aloha_sim_prefix_lm_bf16.engine",
        "/workspace/openpi_artifacts/pi0_aloha_sim_prefix_lm_fp16.engine",
    )
    iterations: int = 100
    warmup: int = 5


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


class Engine:
    def __init__(self, path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        start = time.time()
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.load_ms = (time.time() - start) * 1000
        self.context = self.engine.create_execution_context()
        self.outputs = {}
        self.last_inputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
                self.outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)
                self.context.set_tensor_address(name, self.outputs[name].data_ptr())

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.last_inputs = {name: tensor.contiguous() for name, tensor in inputs.items()}
        for name, tensor in self.last_inputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT prefix LM execution failed")
        return self.outputs


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


def _prepare_prefix_inputs(model, policy):
    inputs = policy._input_transform(jax.tree.map(lambda x: x, _raw_aloha_obs()))  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to("cuda")[None, ...], inputs)
    observation = model_lib.Observation.from_dict(inputs)
    images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    attention_mask = model._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
    position_ids = torch.cumsum(prefix_pad_masks.to(torch.int32), dim=1) - 1
    return prefix_embs, attention_mask, position_ids


def main(args: Args) -> None:
    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device="cuda")
    model = policy._model.eval()  # noqa: SLF001

    with torch.no_grad():
        prefix_embs, attention_mask, position_ids = _prepare_prefix_inputs(model, policy)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        torch_times = []
        reference = None
        for i in range(args.warmup + args.iterations):
            torch.cuda.synchronize()
            start = time.time()
            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            torch.cuda.synchronize()
            if i >= args.warmup:
                torch_times.append((time.time() - start) * 1000)
            reference = [tensor.detach().clone() for pair in past_key_values.to_legacy_cache() for tensor in pair]
        print(
            f"torch_prefix_lm_ms avg={sum(torch_times) / len(torch_times):.4f} "
            f"min={min(torch_times):.4f} max={max(torch_times):.4f}",
            flush=True,
        )

        for engine_path in args.engines:
            if not os.path.exists(engine_path):
                print(f"missing_engine {engine_path}", flush=True)
                continue
            engine = Engine(engine_path)
            print(f"engine {engine_path} load_ms={engine.load_ms:.2f}", flush=True)
            engine_inputs = {
                "prefix_embs": prefix_embs,
                "attention_mask": attention_mask,
                "position_ids": position_ids.to(torch.int64),
            }
            outputs = None
            for _ in range(args.warmup):
                outputs = engine.run(engine_inputs)
            torch.cuda.synchronize()
            diffs = []
            for i, expected in enumerate(reference):
                actual = outputs[f"past_{i}"]
                abs_diff = (expected.float() - actual.float()).abs()
                diffs.append((abs_diff.max().item(), abs_diff.mean().item()))
            print(
                f"diff max_abs_max={max(item[0] for item in diffs):.6f} "
                f"mean_abs_avg={sum(item[1] for item in diffs) / len(diffs):.6f}",
                flush=True,
            )

            trt_times = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                start = time.time()
                engine.run(engine_inputs)
                torch.cuda.synchronize()
                trt_times.append((time.time() - start) * 1000)
            print(
                f"trt_prefix_lm_ms avg={sum(trt_times) / len(trt_times):.4f} "
                f"min={min(trt_times):.4f} max={max(trt_times):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
