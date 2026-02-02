#!/usr/bin/env python3
"""
Test full model parity - run a forward pass through the score network
with trained weights and save outputs for comparison with Julia.
"""

import sys
sys.path.insert(0, '/home/claudey/BFlaproteina/la-proteina')
import torch_scatter_compat  # Must be before proteinfoundation imports

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Load model
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

# Load checkpoint
ckpt_path = "/home/claudey/BFlaproteina/la-proteina/checkpoints_laproteina/LD1_ucond_notri_512.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Load config
config = OmegaConf.load("/home/claudey/BFlaproteina/la-proteina/configs/nn/local_latents_score_nn_160M.yaml")

# Add cropped_flag_seq to match checkpoint (checkpoint was trained with this feature)
config.feats_seq = list(config.feats_seq) + ["cropped_flag_seq"]

# Create model
model = LocalLatentsTransformer(**OmegaConf.to_container(config), latent_dim=8)

# Load weights (from state_dict with nn. prefix)
state_dict = ckpt['state_dict']
nn_state_dict = {k.replace('nn.', ''): v for k, v in state_dict.items() if k.startswith('nn.')}
model.load_state_dict(nn_state_dict)
model.eval()

print("Loaded model")

# Create deterministic test input
torch.manual_seed(42)
np.random.seed(42)

B, L = 1, 10
t_val = 0.5

# Create input batch (same format as actual inference)
x_t = {
    'bb_ca': torch.randn(B, L, 3) * 0.5,
    'local_latents': torch.randn(B, L, 8) * 0.5
}
t = {
    'bb_ca': torch.full((B,), t_val),
    'local_latents': torch.full((B,), t_val)
}
x_sc = {
    'bb_ca': torch.zeros(B, L, 3),  # No self-conditioning
    'local_latents': torch.zeros(B, L, 8)
}
residue_mask = torch.ones(B, L)

batch = {
    'x_t': x_t,
    't': t,
    'x_sc': x_sc,
    'residue_mask': residue_mask,
    'mask': residue_mask.bool()
}

print(f"Input shapes:")
print(f"  x_t bb_ca: {x_t['bb_ca'].shape}")
print(f"  x_t local_latents: {x_t['local_latents'].shape}")
print(f"  t: {t_val}")

# Forward pass
with torch.no_grad():
    output = model(batch)

# Extract outputs
bb_ca_out = output['bb_ca']['v']  # [B, L, 3]
local_latents_out = output['local_latents']['v']  # [B, L, 8]

print(f"\nOutput shapes:")
print(f"  bb_ca: {bb_ca_out.shape}")
print(f"  local_latents: {local_latents_out.shape}")

print(f"\nOutput values (first 5):")
print(f"  bb_ca[0, 0, :]: {bb_ca_out[0, 0, :]}")
print(f"  local_latents[0, 0, :5]: {local_latents_out[0, 0, :5]}")

# Save for Julia comparison
output_dir = Path('/home/claudey/JuProteina/JuProteina/test')
np.save(output_dir / 'full_model_x_t_bb_ca.npy', x_t['bb_ca'].numpy())
np.save(output_dir / 'full_model_x_t_local_latents.npy', x_t['local_latents'].numpy())
np.save(output_dir / 'full_model_x_sc_bb_ca.npy', x_sc['bb_ca'].numpy())
np.save(output_dir / 'full_model_x_sc_local_latents.npy', x_sc['local_latents'].numpy())
np.save(output_dir / 'full_model_t_val.npy', np.array([t_val]))
np.save(output_dir / 'full_model_mask.npy', residue_mask.numpy())
np.save(output_dir / 'full_model_out_bb_ca.npy', bb_ca_out.numpy())
np.save(output_dir / 'full_model_out_local_latents.npy', local_latents_out.numpy())

print(f"\nSaved test data to {output_dir}")
