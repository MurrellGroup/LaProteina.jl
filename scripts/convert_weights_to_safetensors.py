#!/usr/bin/env python3
"""Convert PyTorch .ckpt checkpoint files to .safetensors format.

Handles all LaProteina model variants:
  Score networks: LD1-LD7
  Autoencoders:   AE1-AE3

Usage:
    python convert_weights_to_safetensors.py input.ckpt output.safetensors
    python convert_weights_to_safetensors.py input.ckpt  # auto-names output

The script handles PyTorch Lightning state_dict wrapping (removes 'model.'
prefix if present) and converts all tensors to CPU float32.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_checkpoint(input_path: str, output_path: str | None = None) -> str:
    """Convert a PyTorch checkpoint to SafeTensors format.

    Parameters
    ----------
    input_path : str
        Path to .ckpt or .pt file.
    output_path : str, optional
        Path for output .safetensors file. If None, replaces extension.

    Returns
    -------
    str
        Path to the saved .safetensors file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".safetensors")
    else:
        output_path = Path(output_path)

    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Extract state_dict (handle Lightning wrappers)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # Assume the dict itself is the state_dict
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # Clean up keys: remove 'model.' prefix from Lightning wrapping
    cleaned = {}
    for key, tensor in state_dict.items():
        # Remove common prefixes from Lightning
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]

        # Convert to float32 CPU tensor
        if tensor.is_floating_point():
            cleaned[clean_key] = tensor.float().contiguous()
        else:
            # SafeTensors supports integer types too
            cleaned[clean_key] = tensor.contiguous()

    print(f"Converted {len(cleaned)} tensors")

    # Print summary of tensor shapes
    total_params = sum(t.numel() for t in cleaned.values())
    print(f"Total parameters: {total_params:,}")

    # Save
    save_file(cleaned, str(output_path))
    print(f"Saved to: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints to SafeTensors format"
    )
    parser.add_argument("input", help="Input .ckpt or .pt file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output .safetensors file (default: replace extension)")
    parser.add_argument("--list-keys", action="store_true",
                        help="Print all tensor keys and shapes")
    args = parser.parse_args()

    output_path = convert_checkpoint(args.input, args.output)

    if args.list_keys:
        from safetensors import safe_open
        print("\nTensor keys and shapes:")
        with safe_open(output_path, framework="pt") as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)
                print(f"  {key}: {list(tensor.shape)} ({tensor.dtype})")


if __name__ == "__main__":
    main()
