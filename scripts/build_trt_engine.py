#!/usr/bin/env python3
"""Build a TensorRT engine from a fixed-shape ONNX model."""

import dataclasses
import os
import time

import tensorrt as trt
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    onnx: str = "/workspace/openpi_artifacts/pi0_aloha_sim_denoise_step.onnx"
    engine: str = "/workspace/openpi_artifacts/pi0_aloha_sim_denoise_step_fp16.engine"
    fp16: bool = True
    workspace_gib: int = 8


def main(args: Args) -> None:
    os.makedirs(os.path.dirname(args.engine), exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx, "rb") as f:
        parsed = parser.parse(f.read())
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
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Requested fp16 but platform_has_fast_fp16 is false")
        config.set_flag(trt.BuilderFlag.FP16)

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
