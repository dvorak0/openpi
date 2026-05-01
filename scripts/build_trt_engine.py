#!/usr/bin/env python3
"""Build a TensorRT engine from a fixed-shape ONNX model."""

import dataclasses
import os
import time
from typing import Literal

import tensorrt as trt
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    onnx: str = "/workspace/openpi_artifacts/pi0_aloha_sim_denoise_step.onnx"
    engine: str = "/workspace/openpi_artifacts/pi0_aloha_sim_denoise_step_bf16.engine"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    workspace_gib: int = 8


def main(args: Args) -> None:
    engine_dir = os.path.dirname(args.engine)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # TensorRT resolves ONNX external data paths relative to the current working directory.
    # Exported large models often place external data next to the ONNX file, so parse from that directory.
    onnx_path = os.path.abspath(args.onnx)
    onnx_dir = os.path.dirname(onnx_path)
    old_cwd = os.getcwd()
    os.chdir(onnx_dir)
    try:
        with open(os.path.basename(onnx_path), "rb") as f:
            parsed = parser.parse(f.read())
    finally:
        os.chdir(old_cwd)
    print(f"parse_ok={parsed} errors={parser.num_errors}", flush=True)
    for i in range(parser.num_errors):
        print(parser.get_error(i), flush=True)
    if not parsed:
        raise RuntimeError("TensorRT failed to parse ONNX")

    print(f"inputs={network.num_inputs} outputs={network.num_outputs} layers={network.num_layers}", flush=True)
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"IN {i} {tensor.name} shape={tensor.shape} dtype={tensor.dtype}", flush=True)
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"OUT {i} {tensor.name} shape={tensor.shape} dtype={tensor.dtype}", flush=True)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib << 30)
    if args.precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Requested fp16 but platform_has_fast_fp16 is false")
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.precision != "fp32":
        raise ValueError(args.precision)

    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_sec = time.time() - start
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(args.engine, "wb") as f:
        f.write(serialized)
    print(
        f"build_sec={build_sec:.2f} engine={args.engine} size_mib={os.path.getsize(args.engine) / 1024**2:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
