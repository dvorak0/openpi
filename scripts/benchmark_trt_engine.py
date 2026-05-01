#!/usr/bin/env python3
"""Benchmark a fixed-shape TensorRT engine with random inputs."""

import dataclasses

import tensorrt as trt
import torch
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    engine: str = "/workspace/openpi_artifacts/pi0_aloha_sim_denoise_step_fp16.engine"
    warmup: int = 20
    iterations: int = 500


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


def main(args: Args) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = _torch_dtype(engine.get_tensor_dtype(name))
        if mode == trt.TensorIOMode.INPUT:
            if dtype == torch.bool:
                tensor = torch.ones(shape, device="cuda", dtype=dtype)
            elif dtype in (torch.int32, torch.int64):
                tensor = torch.ones(shape, device="cuda", dtype=dtype)
            else:
                tensor = torch.randn(shape, device="cuda", dtype=dtype)
        else:
            tensor = torch.empty(shape, device="cuda", dtype=dtype)
        tensors[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())
        print(f"{mode.name} {name} shape={shape} dtype={dtype}", flush=True)

    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(args.warmup):
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execution failed during warmup")
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iterations):
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execution failed")
    end.record()
    torch.cuda.synchronize()
    print(f"trt_avg_ms={start.elapsed_time(end) / args.iterations:.4f} iterations={args.iterations}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
