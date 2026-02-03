#!/usr/bin/env python3
"""
Extract encoder weights from AE1_ucond_512.ckpt to encoder.npz format
for use in Julia.
"""

import sys
sys.path.insert(0, '/home/claudey/JuProteina/la-proteina')
import torch_scatter_compat  # Must be before proteinfoundation imports

import numpy as np
import torch
from pathlib import Path

# Checkpoint path
ckpt_path = "/home/claudey/JuProteina/la-proteina/checkpoints_laproteina/AE1_ucond_512.ckpt"
output_path = "/home/claudey/JuProteina/JuProteina/weights/encoder.npz"

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
