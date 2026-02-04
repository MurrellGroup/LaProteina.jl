#!/usr/bin/env python3
"""
Generate reference data for encoder parity testing.
Creates inputs and outputs from the pretrained encoder to compare with Julia.
"""

import sys
la_proteina = os.environ.get('LA_PROTEINA_PATH', '')
if la_proteina: sys.path.insert(0, la_proteina)
import torch_scatter_compat  # Must be before proteinfoundation imports

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Load the encoder from checkpoint
ckpt_path = os.path.join(la_proteina, "checkpoints_laproteina", "AE1_ucond_512.ckpt"
config_path = os.path.join(la_proteina, "configs", "nn_ae/nn_130m.yaml"

print(f"Loading config from {config_path}")
config = OmegaConf.load(config_path)

print(f"Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Import encoder class
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer

# The checkpoint uses features WITHOUT chain_idx (468 seq dims, 316 pair dims)
# So we need to modify the config to match
encoder_cfg = OmegaConf.to_container(config.encoder, resolve=True)
# Remove chain_idx features to match pretrained weights
encoder_cfg['feats_seq'] = ["chain_break_per_res", "x1_aatype", "x1_a37coors_nm", "x1_a37coors_nm_rel", "x1_bb_angles", "x1_sidechain_angles"]
encoder_cfg['feats_pair_repr'] = ["rel_seq_sep", "x1_bb_pair_dists_nm", "x1_bb_pair_orientation"]
encoder_cfg['latent_z_dim'] = config.latent_z_dim

# Create encoder with config dict (expects kwargs with 'encoder' key, plus latent_z_dim at top level)
kwargs = {
    'encoder': encoder_cfg,
    'latent_z_dim': config.latent_z_dim
}
encoder = EncoderTransformer(**kwargs)

# Load weights
state_dict = ckpt['state_dict']
encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
encoder.load_state_dict(encoder_state)
encoder.eval()

print(f"Encoder loaded with {sum(p.numel() for p in encoder.parameters())} parameters")

# Create test input (batch of 1, length 10)
torch.manual_seed(42)
np.random.seed(42)

B, L = 1, 10
device = 'cpu'

# Create test inputs matching what the encoder expects
# The encoder uses strict_feats=False, so missing features get defaults

# Atom37 coordinates in nm (37 atoms per residue, 3 coords each)
coords_a37 = torch.randn(B, L, 37, 3) * 0.1  # Small random coordinates in nm
# Make sure CA (index 1) is centered around 0
coords_a37[:, :, 1, :] = torch.randn(B, L, 3) * 0.3

# Coordinate mask (which atoms are present)
coord_mask = torch.ones(B, L, 37)
# Typical backbone atoms present: N(0), CA(1), C(2), O(3)
# Set some sidechain atoms to 0 for realism
coord_mask[:, :, 5:] = torch.randint(0, 2, (B, L, 32)).float()

# Residue types (0-19 for amino acids)
aatype = torch.randint(0, 20, (B, L))

# Mask (which residues are valid)
mask = torch.ones(B, L)

# Chain breaks (0 = no break, 1 = break)
chain_breaks = torch.zeros(B, L)

# Chain index (all same chain for simplicity)
chain_idx = torch.zeros(B, L, dtype=torch.long)

# Build batch dictionary with expected format
# The encoder expects mask_dict for extracting the sequence mask
# Also needs 'coords' for batch size extraction in zero-feature factories
seq_mask = mask.bool()  # [B, L] - which residues are valid

# mask_dict['coords'] needs to be 4D: [B, L, A, C] where encoder takes [..., 0, 0] to get [B, L]
# We can expand coord_mask to 4D by adding a dummy last dimension
# Must be boolean for the mask operations
coord_mask_4d = coord_mask.bool().unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, L, 37, 3]

batch = {
    'coords_nm': coords_a37,  # [B, L, 37, 3] in nanometers
    'coords': coords_a37 * 10.0,  # [B, L, 37, 3] in Angstroms (for batch size extraction)
    'coord_mask': coord_mask,  # [B, L, 37]
    'residue_type': aatype,    # [B, L]
    'chain_idx': chain_idx,    # [B, L]
    'mask': mask,              # [B, L] - residue-level mask
    'mask_dict': {
        'coords': coord_mask_4d,  # [B, L, 37, 3] - encoder extracts [..., 0, 0] for seq mask
        'residue_type': seq_mask,  # [B, L] - residue-level mask
    },
}

print("\nInput shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")

# Run encoder (deterministic mode by setting latent sample to mean)
print("\nRunning encoder...")
with torch.no_grad():
    output = encoder(batch)

print("\nOutput keys and shapes:")
for k, v in output.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: {type(v)}")

# Save inputs and outputs
output_dir = Path(__file__).parent / "data"
output_dir.mkdir(exist_ok=True)

# Save inputs
np.save(output_dir / "encoder_coords_nm.npy", coords_a37.numpy())
np.save(output_dir / "encoder_coord_mask.npy", coord_mask.numpy())
np.save(output_dir / "encoder_aatype.npy", aatype.numpy())
np.save(output_dir / "encoder_mask.npy", mask.numpy())

# Save outputs
np.save(output_dir / "encoder_mean.npy", output['mean'].numpy())
np.save(output_dir / "encoder_log_scale.npy", output['log_scale'].numpy())

print(f"\nSaved test data to {output_dir}")

# Print sample values for debugging
print("\nSample output values:")
print(f"  mean[0, 0, :4]: {output['mean'][0, 0, :4]}")
print(f"  log_scale[0, 0, :4]: {output['log_scale'][0, 0, :4]}")
