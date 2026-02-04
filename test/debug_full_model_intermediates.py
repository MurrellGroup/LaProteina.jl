#!/usr/bin/env python3
"""
Save intermediate values from Python model for debugging parity.
"""

import sys
la_proteina = os.environ.get('LA_PROTEINA_PATH', '')
if la_proteina: sys.path.insert(0, la_proteina)
import torch_scatter_compat

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

# Load checkpoint
ckpt_path = os.path.join(la_proteina, "checkpoints_laproteina", "LD1_ucond_notri_512.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Load config
config = OmegaConf.load(os.path.join(la_proteina, "configs", "nn/local_latents_score_nn_160M.yaml")

# Add cropped_flag_seq to match checkpoint
config.feats_seq = list(config.feats_seq) + ["cropped_flag_seq"]

# Create model
model = LocalLatentsTransformer(**OmegaConf.to_container(config), latent_dim=8)

# Load weights
state_dict = ckpt['state_dict']
nn_state_dict = {k.replace('nn.', ''): v for k, v in state_dict.items() if k.startswith('nn.')}
model.load_state_dict(nn_state_dict)
model.eval()

# Create deterministic test input
torch.manual_seed(42)
np.random.seed(42)

B, L = 1, 10
t_val = 0.5

x_t = {
    'bb_ca': torch.randn(B, L, 3) * 0.5,
    'local_latents': torch.randn(B, L, 8) * 0.5
}
t = {
    'bb_ca': torch.full((B,), t_val),
    'local_latents': torch.full((B,), t_val)
}
x_sc = {
    'bb_ca': torch.zeros(B, L, 3),
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

output_dir = Path(__file__).parent

print("=== RAW INPUT FEATURES ===")
print(f"x_t bb_ca [0,0,:]: {x_t['bb_ca'][0, 0, :]}")
print(f"x_t local_latents [0,0,:]: {x_t['local_latents'][0, 0, :]}")

# Access internal components
init_repr_factory = model.init_repr_factory
cond_factory = model.cond_factory
pair_repr_builder = model.pair_repr_builder

# Get raw features before projection
with torch.no_grad():
    # Sequence features raw - compute using feat_creators
    raw_feats_list = [fcreator(batch) for fcreator in init_repr_factory.feat_creators]
    raw_seq_feats = torch.cat(raw_feats_list, dim=-1)  # [B, L, D_features]
    print(f"\n=== RAW SEQUENCE FEATURES ===")
    print(f"Shape: {raw_seq_feats.shape}")
    print(f"raw_seq_feats[0, 0, :5]: {raw_seq_feats[0, 0, :5]}")
    print(f"raw_seq_feats[0, 0, :3] (should be x_t bb_ca): {raw_seq_feats[0, 0, :3]}")
    print(f"raw_seq_feats[0, 0, 3:11] (should be x_t local_latents): {raw_seq_feats[0, 0, 3:11]}")

    # Projected sequence features
    seq_repr = init_repr_factory(batch)  # [B, L, token_dim]
    print(f"\n=== PROJECTED SEQUENCE FEATURES ===")
    print(f"Shape: {seq_repr.shape}")
    print(f"seq_repr[0, 0, :5]: {seq_repr[0, 0, :5]}")

    # Time embeddings - skip detailed time embedding test
    print(f"\n=== TIME EMBEDDINGS ===")
    print(f"(skipping direct time embedding check)")

    # Conditioning features
    cond_repr = cond_factory(batch)  # [B, L, dim_cond]
    print(f"\n=== CONDITIONING FEATURES ===")
    print(f"Shape: {cond_repr.shape}")
    print(f"cond_repr[0, 0, :5]: {cond_repr[0, 0, :5]}")

    # Pair representation
    pair_repr = pair_repr_builder(batch)  # [B, L, L, pair_dim]
    print(f"\n=== PAIR REPRESENTATION ===")
    print(f"Shape: {pair_repr.shape}")
    print(f"pair_repr[0, 0, 0, :5]: {pair_repr[0, 0, 0, :5]}")
    print(f"pair_repr[0, 0, 1, :5]: {pair_repr[0, 0, 1, :5]}")

    # Conditioning transitions
    c = cond_repr
    c = model.transition_c_1(c, batch["mask"])
    print(f"\n=== AFTER TRANSITION_C_1 ===")
    print(f"Shape: {c.shape}")
    print(f"c[0, 0, :5]: {c[0, 0, :5]}")

    c = model.transition_c_2(c, batch["mask"])
    print(f"\n=== AFTER TRANSITION_C_2 ===")
    print(f"Shape: {c.shape}")
    print(f"c[0, 0, :5]: {c[0, 0, :5]}")

    # Save intermediates
    np.save(output_dir / 'debug_raw_seq_feats.npy', raw_seq_feats.numpy())
    np.save(output_dir / 'debug_seq_repr.npy', seq_repr.numpy())
    np.save(output_dir / 'debug_cond_repr.npy', cond_repr.numpy())
    np.save(output_dir / 'debug_pair_repr.npy', pair_repr.numpy())
    np.save(output_dir / 'debug_cond_after_trans.npy', c.numpy())

    print(f"\nSaved debug data to {output_dir}")
