#!/usr/bin/env python3
"""Verify a SONIC release-schema deploy bundle after unzipping."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def dims(value):
    return [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]


def check_model(path: Path, expected_input: list[int], expected_output: list[int], output_name: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    inputs = [(item.name, dims(item)) for item in model.graph.input]
    outputs = [(item.name, dims(item)) for item in model.graph.output]
    print(f"{path}:")
    print(f"  inputs:  {inputs}")
    print(f"  outputs: {outputs}")
    if inputs != [("obs_dict", expected_input)]:
        raise SystemExit(f"bad input schema for {path}: expected obs_dict {expected_input}, got {inputs}")
    if outputs != [(output_name, expected_output)]:
        raise SystemExit(f"bad output schema for {path}: expected {output_name} {expected_output}, got {outputs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundle",
        help=(
            "Bundle directory, e.g. "
            "gear_sonic_deploy/policy/local/finetuned_release_schema/optimal_B_003500"
        ),
    )
    args = parser.parse_args()
    bundle = Path(args.bundle)
    check_model(bundle / "model_encoder.onnx", [1, 1762], [1, 64], "encoded_tokens")
    check_model(bundle / "model_decoder.onnx", [1, 994], [1, 29], "action")
    config = (bundle / "observation_config.yaml").read_text()
    required = [
        'name: "motion_root_z_position_10frame_step5"',
        'name: "motion_root_z_position"',
        "enabled: true",
    ]
    if not all(item in config for item in required):
        raise SystemExit("observation_config.yaml does not look like release 1762-D config with root-z slots enabled")
    print("OK: bundle is release-schema compatible.")


if __name__ == "__main__":
    main()
