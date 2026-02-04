#!/usr/bin/env python3
"""
Extract encoder weights from AE1_ucond_512.ckpt to encoder.npz format
for use in Julia.

Usage:
    LA_PROTEINA_PATH=/path/to/la-proteina python extract_encoder_weights.py

Environment variables:
    LA_PROTEINA_PATH: Path to la-proteina repository (required)
    CKPT_PATH: Path to checkpoint (default: LA_PROTEINA_PATH/checkpoints_laproteina/AE1_ucond_512.ckpt)
    OUTPUT_PATH: Output path for weights (default: ../weights/encoder.npz)
"""

import os
import sys
from pathlib import Path

# Get la-proteina path from environment
la_proteina_path = os.environ.get('LA_PROTEINA_PATH')
if la_proteina_path:
    sys.path.insert(0, la_proteina_path)
    import torch_scatter_compat  # Must be before proteinfoundation imports

import numpy as np
import torch

# Paths with defaults
script_dir = Path(__file__).parent
default_ckpt = Path(la_proteina_path) / "checkpoints_laproteina/AE1_ucond_512.ckpt" if la_proteina_path else None
ckpt_path = os.environ.get('CKPT_PATH', str(default_ckpt) if default_ckpt else None)
output_path = os.environ.get('OUTPUT_PATH', str(script_dir.parent / "weights" / "encoder.npz"))

if not ckpt_path or not Path(ckpt_path).exists():
    print("Error: Checkpoint not found. Set LA_PROTEINA_PATH or CKPT_PATH environment variable.")
    sys.exit(1)

print(f"Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Get encoder weights from state dict
state_dict = ckpt['state_dict']

# Extract encoder weights
encoder_weights = {}
for name, param in state_dict.items():
    if name.startswith('encoder.'):
        # Remove 'encoder.' prefix and convert dots to dots (keep format)
        key = name.replace('encoder.', '')
        encoder_weights[key] = param.detach().cpu().numpy()
        print(f"  {key}: {param.shape}")

print(f"\nFound {len(encoder_weights)} encoder weight tensors")

# Save as npz
np.savez(output_path, **encoder_weights)
print(f"Saved encoder weights to {output_path}")

# Print some summary statistics
print("\nKey patterns found:")
key_prefixes = set()
for key in encoder_weights.keys():
    parts = key.split('.')
    if len(parts) >= 2:
        key_prefixes.add('.'.join(parts[:2]))
    else:
        key_prefixes.add(parts[0])

for prefix in sorted(key_prefixes):
    count = sum(1 for k in encoder_weights if k.startswith(prefix))
    print(f"  {prefix}: {count} tensors")
