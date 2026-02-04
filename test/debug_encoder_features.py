#!/usr/bin/env python3
"""
Debug encoder features - save intermediate outputs for comparison with Julia.
"""

import sys
la_proteina = os.environ.get('LA_PROTEINA_PATH', '')
if la_proteina: sys.path.insert(0, la_proteina)
import torch_scatter_compat

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

ckpt_path = os.path.join(la_proteina, "checkpoints_laproteina", "AE1_ucond_512.ckpt"
config_path = os.path.join(la_proteina, "configs", "nn_ae/nn_130m.yaml"

print(f"Loading config and checkpoint...")
config = OmegaConf.load(config_path)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer

# Use features matching the pretrained weights
encoder_cfg = OmegaConf.to_container(config.encoder, resolve=True)
encoder_cfg['feats_seq'] = ["chain_break_per_res", "x1_aatype", "x1_a37coors_nm", "x1_a37coors_nm_rel", "x1_bb_angles", "x1_sidechain_angles"]
encoder_cfg['feats_pair_repr'] = ["rel_seq_sep", "x1_bb_pair_dists_nm", "x1_bb_pair_orientation"]
encoder_cfg['latent_z_dim'] = config.latent_z_dim

kwargs = {'encoder': encoder_cfg, 'latent_z_dim': config.latent_z_dim}
encoder = EncoderTransformer(**kwargs)

state_dict = ckpt['state_dict']
encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
encoder.load_state_dict(encoder_state)
encoder.eval()

# Create simple test input
torch.manual_seed(42)
np.random.seed(42)

B, L = 1, 10

coords_a37 = torch.randn(B, L, 37, 3) * 0.1
coords_a37[:, :, 1, :] = torch.randn(B, L, 3) * 0.3  # CA
coord_mask = torch.ones(B, L, 37)
coord_mask[:, :, 5:] = torch.randint(0, 2, (B, L, 32)).float()
aatype = torch.randint(0, 20, (B, L))
mask = torch.ones(B, L)
chain_idx = torch.zeros(B, L, dtype=torch.long)

seq_mask = mask.bool()
coord_mask_4d = coord_mask.bool().unsqueeze(-1).expand(-1, -1, -1, 3)

batch = {
    'coords_nm': coords_a37,
    'coords': coords_a37 * 10.0,
    'coord_mask': coord_mask,
    'residue_type': aatype,
    'chain_idx': chain_idx,
    'mask': mask,
    'mask_dict': {
        'coords': coord_mask_4d,
        'residue_type': seq_mask,
    },
}

output_dir = Path(__file__).parent / "data"

# Save inputs for Julia
np.save(output_dir / "encoder_debug_coords_nm.npy", coords_a37.numpy())
np.save(output_dir / "encoder_debug_coord_mask.npy", coord_mask.numpy())
np.save(output_dir / "encoder_debug_aatype.npy", aatype.numpy())
np.save(output_dir / "encoder_debug_mask.npy", mask.numpy())

print("\n=== Raw Feature Extraction ===")

# Extract RAW features (before projection) using individual feature creators
with torch.no_grad():
    # Get raw features from each feature creator
    raw_feats = []
    for fc in encoder.init_repr_factory.feat_creators:
        feat = fc(batch)  # [B, L, dim]
        raw_feats.append(feat)
        print(f"  {fc.__class__.__name__}: {feat.shape}, first values: {feat[0, 0, :min(4, feat.shape[-1])].tolist()}")

    # Concatenate raw features
    raw_seq_features = torch.cat(raw_feats, dim=-1)  # [B, L, 468]
    print(f"\nRaw concatenated shape: {raw_seq_features.shape}")

np.save(output_dir / "encoder_debug_raw_seq_features.npy", raw_seq_features.numpy())

# Also get projected features
with torch.no_grad():
    seq_features = encoder.init_repr_factory(batch)  # [B, L, 768]
    pair_features = encoder.pair_rep_factory(batch)  # [B, L, L, 256]

    # Raw pair features
    raw_pair_feats = []
    for fc in encoder.pair_rep_factory.feat_creators:
        feat = fc(batch)  # [B, L, L, dim]
        raw_pair_feats.append(feat)
        print(f"  Pair {fc.__class__.__name__}: {feat.shape}, first values: {feat[0, 0, 0, :min(4, feat.shape[-1])].tolist()}")

    raw_pair_features = torch.cat(raw_pair_feats, dim=-1)  # [B, L, L, 316]
    print(f"\nRaw pair concatenated shape: {raw_pair_features.shape}")

print(f"\nProjected seq_features shape: {seq_features.shape}")
np.save(output_dir / "encoder_debug_seq_features.npy", seq_features.numpy())
np.save(output_dir / "encoder_debug_pair_features.npy", pair_features.numpy())
np.save(output_dir / "encoder_debug_raw_pair_features.npy", raw_pair_features.numpy())

# Run full encoder
print("\n=== Full Encoder Output ===")
with torch.no_grad():
    output = encoder(batch)

np.save(output_dir / "encoder_debug_mean.npy", output['mean'].numpy())
np.save(output_dir / "encoder_debug_log_scale.npy", output['log_scale'].numpy())

print(f"mean[0, 0, :4]: {output['mean'][0, 0, :4].tolist()}")
print(f"log_scale[0, 0, :4]: {output['log_scale'][0, 0, :4].tolist()}")

print(f"\nSaved debug data to {output_dir}")
