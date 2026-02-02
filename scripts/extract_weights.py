#!/usr/bin/env python3
"""Extract PyTorch weights and save as NPZ for Julia loading."""

import torch
import numpy as np
import os
import sys

def extract_ld1_weights(checkpoint_path, output_path):
    """Extract score network weights from LD1 checkpoint."""
    print(f"Loading {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    # Extract nn.* weights (score network)
    weights = {}
    for k, v in sd.items():
        if k.startswith('nn.'):
            # Remove 'nn.' prefix
            key = k[3:]
            # Convert to numpy
            weights[key] = v.cpu().numpy()

    print(f"Extracted {len(weights)} weight tensors")
    np.savez(output_path, **weights)
    print(f"Saved to {output_path}")
    return weights

def extract_ae1_weights(checkpoint_path, output_path):
    """Extract autoencoder decoder weights from AE1 checkpoint."""
    print(f"Loading {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    # Extract decoder.* weights
    weights = {}
    for k, v in sd.items():
        if k.startswith('decoder.'):
            # Remove 'decoder.' prefix
            key = k[8:]
            weights[key] = v.cpu().numpy()

    print(f"Extracted {len(weights)} weight tensors")
    np.savez(output_path, **weights)
    print(f"Saved to {output_path}")
    return weights

if __name__ == "__main__":
    base_dir = "/home/claudey/JuProteina/la-proteina/checkpoints_laproteina"
    out_dir = "/home/claudey/JuProteina/JuProteina/weights"
    os.makedirs(out_dir, exist_ok=True)

    # Extract score network weights
    ld1_path = os.path.join(base_dir, "LD1_ucond_notri_512.ckpt")
    if os.path.exists(ld1_path):
        extract_ld1_weights(ld1_path, os.path.join(out_dir, "score_network.npz"))

    # Extract decoder weights
    ae1_path = os.path.join(base_dir, "AE1_ucond_512.ckpt")
    if os.path.exists(ae1_path):
        extract_ae1_weights(ae1_path, os.path.join(out_dir, "decoder.npz"))

    print("\nDone!")
