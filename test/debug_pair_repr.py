#!/usr/bin/env python3
"""
Debug pair representation building in Python.
Saves all intermediate values for comparison with Julia.
"""

import sys
sys.path.insert(0, '/home/claudey/BFlaproteina/la-proteina')

# Import torch_scatter compat before proteinfoundation imports
import torch_scatter_compat

import numpy as np
import torch
from pathlib import Path

# Load pretrained checkpoint
ckpt_path = "/home/claudey/BFlaproteina/la-proteina/checkpoints_laproteina/LD1_ucond_notri_512.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Extract the score network (nn)
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from omegaconf import OmegaConf

# Create model with correct config
config = OmegaConf.load("/home/claudey/BFlaproteina/la-proteina/configs/nn/local_latents_score_nn_160M.yaml")

# Pass full config as kwargs
model = LocalLatentsTransformer(**OmegaConf.to_container(config))

# The checkpoint has nested structure - need to find correct key
print("Checkpoint keys:", list(ckpt.keys())[:10])
if 'state_dict' in ckpt:
    # PyTorch Lightning format
    state_dict = ckpt['state_dict']
    # Filter for nn. prefix
    nn_state_dict = {k.replace('nn.', ''): v for k, v in state_dict.items() if k.startswith('nn.')}
    model.load_state_dict(nn_state_dict)
else:
    model.load_state_dict(ckpt['nn'])
model.eval()

print("Loaded model")

# Create test input
torch.manual_seed(42)
B, L = 1, 5
t_val = 0.5

# Create input batch
x_t = {
    'bb_ca': torch.randn(B, L, 3) * 0.5,  # CA coords
    'local_latents': torch.randn(B, L, 8) * 0.5  # latent vectors
}
t = {
    'bb_ca': torch.full((B,), t_val),
    'local_latents': torch.full((B,), t_val)
}
x_sc = {
    'bb_ca': torch.zeros(B, L, 3),  # zero self-conditioning
    'local_latents': torch.zeros(B, L, 8)
}
residue_mask = torch.ones(B, L)

batch = {
    'x_t': x_t,
    't': t,
    'x_sc': x_sc,
    'residue_mask': residue_mask
}

print(f"Input x_t bb_ca shape: {x_t['bb_ca'].shape}")
print(f"Input x_t local_latents shape: {x_t['local_latents'].shape}")

# ============================================================================
# Trace through pair representation building
# ============================================================================

print("\n=== Tracing Pair Representation ===")

# Get pair representation builder
pair_repr_builder = model.pair_repr_builder

# 1. Extract features using the init_repr feature factory
init_repr_factory = pair_repr_builder.init_repr
cond_factory = pair_repr_builder.cond

print(f"init_repr_factory: {init_repr_factory}")
print(f"cond_factory: {cond_factory}")

# Check what features are in the feature factory
print(f"\nFeature configs:")
for i, feat in enumerate(init_repr_factory.feats):
    print(f"  {i}: {feat.__class__.__name__} -> {feat.out_dim}")

# Run feature extraction
raw_features = []
for feat in init_repr_factory.feats:
    feat_out = feat(batch)
    raw_features.append(feat_out)
    print(f"  {feat.__class__.__name__}: {feat_out.shape}")

# Concatenate raw features
concat_features = torch.cat(raw_features, dim=-1)
print(f"\nConcatenated features shape: {concat_features.shape}")

# Get the projection output
# The init_repr factory applies: concat -> proj -> layer_norm
pair_proj = init_repr_factory.proj(concat_features)
print(f"After projection shape: {pair_proj.shape}")

# After layer norm in init_repr
if init_repr_factory.layer_norm is not None:
    pair_proj_ln = init_repr_factory.layer_norm(pair_proj)
    print(f"After layer_norm shape: {pair_proj_ln.shape}")
else:
    pair_proj_ln = pair_proj
    print("No layer norm in init_repr")

# Full init_repr output
init_repr_out = init_repr_factory(batch)
print(f"init_repr full output shape: {init_repr_out.shape}")

# Check if pair_proj_ln == init_repr_out
diff = torch.abs(pair_proj_ln - init_repr_out).max().item()
print(f"Diff between manual and full init_repr: {diff}")

# 2. Get conditioning features
print("\n=== Conditioning Features ===")
for i, feat in enumerate(cond_factory.feats):
    feat_out = feat(batch)
    print(f"  {i}: {feat.__class__.__name__}: {feat_out.shape}")

cond_out = cond_factory(batch)
print(f"Conditioning output shape: {cond_out.shape}")

# 3. Apply AdaLN
print("\n=== AdaptiveLayerNormIdentical ===")
adaln = pair_repr_builder.adaln

# Compute pair mask
residue_mask = batch['residue_mask']
pair_mask = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)  # [B, L, L]

final_pair_repr = adaln(init_repr_out, cond_out, pair_mask)
print(f"Final pair repr shape: {final_pair_repr.shape}")

# Full pair_repr_builder output
full_output = pair_repr_builder(batch)
print(f"Full pair_repr_builder output shape: {full_output.shape}")

# Check parity
diff = torch.abs(final_pair_repr - full_output).max().item()
print(f"Diff between manual and full pair_repr_builder: {diff}")

# ============================================================================
# Save all intermediate values
# ============================================================================

output_dir = Path('/home/claudey/JuProteina/JuProteina/test')

# Save inputs
np.save(output_dir / 'pair_debug_x_t_bb_ca.npy', x_t['bb_ca'].numpy())
np.save(output_dir / 'pair_debug_x_t_local_latents.npy', x_t['local_latents'].numpy())
np.save(output_dir / 'pair_debug_x_sc_bb_ca.npy', x_sc['bb_ca'].numpy())
np.save(output_dir / 'pair_debug_x_sc_local_latents.npy', x_sc['local_latents'].numpy())
np.save(output_dir / 'pair_debug_t_val.npy', np.array([t_val]))
np.save(output_dir / 'pair_debug_mask.npy', residue_mask.numpy())

# Save raw feature outputs (before concatenation)
for i, feat_out in enumerate(raw_features):
    np.save(output_dir / f'pair_debug_raw_feat_{i}.npy', feat_out.detach().numpy())

# Save intermediates
np.save(output_dir / 'pair_debug_concat_features.npy', concat_features.detach().numpy())
np.save(output_dir / 'pair_debug_pair_proj.npy', pair_proj.detach().numpy())
np.save(output_dir / 'pair_debug_pair_proj_ln.npy', pair_proj_ln.detach().numpy())
np.save(output_dir / 'pair_debug_init_repr_out.npy', init_repr_out.detach().numpy())
np.save(output_dir / 'pair_debug_cond_out.npy', cond_out.detach().numpy())
np.save(output_dir / 'pair_debug_final_pair_repr.npy', final_pair_repr.detach().numpy())

# Save weights
proj_weight = init_repr_factory.proj.weight.data.numpy()
np.save(output_dir / 'pair_debug_proj_weight.npy', proj_weight)
if init_repr_factory.proj.bias is not None:
    np.save(output_dir / 'pair_debug_proj_bias.npy', init_repr_factory.proj.bias.data.numpy())

if init_repr_factory.layer_norm is not None:
    ln_weight = init_repr_factory.layer_norm.weight.data.numpy()
    ln_bias = init_repr_factory.layer_norm.bias.data.numpy()
    np.save(output_dir / 'pair_debug_ln_weight.npy', ln_weight)
    np.save(output_dir / 'pair_debug_ln_bias.npy', ln_bias)

# Save conditioning weights
cond_proj_weight = cond_factory.proj.weight.data.numpy()
np.save(output_dir / 'pair_debug_cond_proj_weight.npy', cond_proj_weight)
if cond_factory.layer_norm is not None:
    np.save(output_dir / 'pair_debug_cond_ln_weight.npy', cond_factory.layer_norm.weight.data.numpy())
    np.save(output_dir / 'pair_debug_cond_ln_bias.npy', cond_factory.layer_norm.bias.data.numpy())

# Save AdaLN weights
np.save(output_dir / 'pair_debug_adaln_norm_cond_weight.npy', adaln.norm_cond.weight.data.numpy())
np.save(output_dir / 'pair_debug_adaln_norm_cond_bias.npy', adaln.norm_cond.bias.data.numpy())
np.save(output_dir / 'pair_debug_adaln_to_gamma_weight.npy', adaln.to_gamma[0].weight.data.numpy())
np.save(output_dir / 'pair_debug_adaln_to_gamma_bias.npy', adaln.to_gamma[0].bias.data.numpy())
np.save(output_dir / 'pair_debug_adaln_to_beta_weight.npy', adaln.to_beta.weight.data.numpy())

print("\nSaved all debug data to test/")

# ============================================================================
# Print feature config details
# ============================================================================

print("\n=== Feature Config Details ===")
print(f"Number of features: {len(init_repr_factory.feats)}")
total_dim = 0
for i, feat in enumerate(init_repr_factory.feats):
    dim = feat.out_dim
    total_dim += dim
    print(f"  {i}: {feat.__class__.__name__}")
    print(f"      out_dim: {dim}")
    if hasattr(feat, 'data_mode'):
        print(f"      data_mode: {feat.data_mode}")
    if hasattr(feat, 'min_d') and hasattr(feat, 'max_d'):
        print(f"      min_d: {feat.min_d}, max_d: {feat.max_d}")
    if hasattr(feat, 'max_sep'):
        print(f"      max_sep: {feat.max_sep}")

print(f"\nTotal input dimension: {total_dim}")
print(f"Projection output dim: {init_repr_factory.proj.out_features}")

# Check what types of features are in the factory
print("\n=== Feature Class Names ===")
for feat in init_repr_factory.feats:
    print(f"  {feat.__class__.__name__}")
