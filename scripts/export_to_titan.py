#!/usr/bin/env python3
"""
export_to_titan.py - Convert a PyTorch MLP to TitanInfer .titan format

Usage:
    python export_to_titan.py model.pt output.titan

Supports: nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax

The .titan binary format (little-endian):
  Header: 4-byte magic "TITN" | uint32 version | uint32 layer_count
  Per layer:
    uint32 layer_type (1=Dense, 2=ReLU, 3=Sigmoid, 4=Tanh, 5=Softmax)
    Dense only: uint32 in_features | uint32 out_features | uint8 has_bias
                float32[out*in] weights | float32[out] bias (if has_bias)
"""

import argparse
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

MAGIC = b"TITN"
FORMAT_VERSION = 1

# Layer type enum (must match C++ LayerType in format.hpp)
LAYER_TYPES = {
    nn.Linear: 1,
    nn.ReLU: 2,
    nn.Sigmoid: 3,
    nn.Tanh: 4,
    nn.Softmax: 5,
}


def export_to_titan(model: nn.Module, filepath: str) -> None:
    """Export a PyTorch Sequential model to .titan binary format."""
    if not isinstance(model, nn.Sequential):
        raise ValueError("Model must be nn.Sequential")

    layers = list(model.children())

    with open(filepath, "wb") as f:
        # Write header
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", len(layers)))

        for layer in layers:
            layer_type = None
            for cls, type_id in LAYER_TYPES.items():
                if isinstance(layer, cls):
                    layer_type = type_id
                    break

            if layer_type is None:
                raise ValueError(
                    f"Unsupported layer type: {type(layer).__name__}"
                )

            f.write(struct.pack("<I", layer_type))

            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                has_bias = 1 if layer.bias is not None else 0

                f.write(struct.pack("<I", in_features))
                f.write(struct.pack("<I", out_features))
                f.write(struct.pack("<B", has_bias))

                # PyTorch Linear weight shape: (out_features, in_features)
                # This matches TitanInfer DenseLayer layout exactly
                weight_data = (
                    layer.weight.detach().cpu().numpy().astype(np.float32)
                )
                f.write(weight_data.tobytes())

                if has_bias:
                    bias_data = (
                        layer.bias.detach().cpu().numpy().astype(np.float32)
                    )
                    f.write(bias_data.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch MLP to TitanInfer .titan format"
    )
    parser.add_argument("model_path", help="Path to saved PyTorch model (.pt)")
    parser.add_argument("output_path", help="Output .titan file path")
    args = parser.parse_args()

    model = torch.load(
        args.model_path, map_location="cpu", weights_only=False
    )

    if not isinstance(model, nn.Sequential):
        print(
            "Error: Expected nn.Sequential model. "
            "If you saved only state_dict, re-save with "
            "torch.save(model, path)",
            file=sys.stderr,
        )
        sys.exit(1)

    export_to_titan(model, args.output_path)

    # Print summary
    num_layers = len(list(model.children()))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Exported to {args.output_path}")
    print(f"  Layers: {num_layers}")
    print(f"  Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
