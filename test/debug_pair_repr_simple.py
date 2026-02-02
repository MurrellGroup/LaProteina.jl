#!/usr/bin/env python3
"""
Simple debugging of pair representation building.
Computes features and applies weights step by step without loading full model.
"""

import numpy as np
import os
from pathlib import Path

np.random.seed(42)

# Test parameters
B, L = 1, 5
t_val = 0.5

# Create test inputs - matching Julia conventions after conversion
# Python format: [B, L, D] and [B, L, L, D]
x_t_bb_ca = np.random.randn(B, L, 3).astype(np.float32) * 0.5
x_t_local_latents = np.random.randn(B, L, 8).astype(np.float32) * 0.5
x_sc_bb_ca = np.zeros((B, L, 3), dtype=np.float32)  # zero self-conditioning
x_sc_local_latents = np.zeros((B, L, 8), dtype=np.float32)
mask = np.ones((B, L), dtype=np.float32)

# Time arrays
t_bb_ca = np.full((B,), t_val, dtype=np.float32)
t_local_latents = np.full((B,), t_val, dtype=np.float32)

print(f"Input shapes:")
print(f"  x_t_bb_ca: {x_t_bb_ca.shape}")
print(f"  x_t_local_latents: {x_t_local_latents.shape}")
print(f"  t_bb_ca: {t_bb_ca.shape}")
print(f"  mask: {mask.shape}")

# ============================================================================
# Step 1: Compute raw pair features
# ============================================================================

print("\n=== Computing Raw Pair Features ===")

# Feature 1: Relative sequence separation (127 dims, max_sep=63)
# Python format: [B, L_i, L_j, D]
max_sep = 63  # (127 - 1) / 2
positions = np.arange(L)
rel_sep = positions[:, None] - positions[None, :]  # [L, L] where rel_sep[i,j] = i - j
rel_sep_clamped = np.clip(rel_sep, -max_sep, max_sep)
rel_sep_shifted = rel_sep_clamped + max_sep  # [0, 2*max_sep]

# One-hot encode
n_classes = 2 * max_sep + 1  # 127
rel_seq_sep_feat = np.zeros((B, L, L, n_classes), dtype=np.float32)
for i in range(L):
    for j in range(L):
        idx = rel_sep_shifted[i, j]
        rel_seq_sep_feat[:, i, j, idx] = 1.0

print(f"rel_seq_sep_feat shape: {rel_seq_sep_feat.shape}, non-zeros per position: {rel_seq_sep_feat.sum()}")

# Feature 2: xt_bb_ca_pair_dists (30 bins, 0.1-3.0nm)
def compute_pairwise_distances(coords):
    """Compute pairwise Euclidean distances. coords: [B, L, 3] -> [B, L, L]"""
    # Using broadcasting: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    sq_norms = np.sum(coords ** 2, axis=-1, keepdims=True)  # [B, L, 1]
    dot_prod = np.matmul(coords, coords.transpose(0, 2, 1))  # [B, L, L]
    sq_dists = sq_norms + sq_norms.transpose(0, 2, 1) - 2 * dot_prod
    sq_dists = np.maximum(sq_dists, 0)  # Handle numerical issues
    return np.sqrt(sq_dists)

def bin_distances(dists, min_dist, max_dist, n_bins):
    """Bin distances into one-hot vectors. dists: [B, L, L] -> [B, L, L, n_bins]"""
    bin_edges = np.linspace(min_dist, max_dist, n_bins - 1)
    bin_indices = np.zeros_like(dists, dtype=np.int32)
    for i, edge in enumerate(bin_edges):
        bin_indices += (dists > edge).astype(np.int32)

    # One-hot encode
    result = np.zeros((*dists.shape, n_bins), dtype=np.float32)
    for b in range(dists.shape[0]):
        for i in range(dists.shape[1]):
            for j in range(dists.shape[2]):
                result[b, i, j, bin_indices[b, i, j]] = 1.0
    return result

# Compute xt distances
xt_dists = compute_pairwise_distances(x_t_bb_ca)  # [B, L, L]
xt_pair_dist_feat = bin_distances(xt_dists, 0.1, 3.0, 30)  # [B, L, L, 30]
print(f"xt_pair_dist_feat shape: {xt_pair_dist_feat.shape}")

# Feature 3: x_sc_bb_ca_pair_dists (30 bins, 0.1-3.0nm)
xsc_dists = compute_pairwise_distances(x_sc_bb_ca)  # [B, L, L]
xsc_pair_dist_feat = bin_distances(xsc_dists, 0.1, 3.0, 30)  # [B, L, L, 30]
print(f"xsc_pair_dist_feat shape: {xsc_pair_dist_feat.shape}")

# Feature 4: optional_ca_pair_dist (30 bins, 0.1-3.0nm) - zeros when not conditioning
optional_pair_dist_feat = np.zeros((B, L, L, 30), dtype=np.float32)
print(f"optional_pair_dist_feat shape: {optional_pair_dist_feat.shape}")

# Concatenate features: 127 + 30 + 30 + 30 = 217
concat_features = np.concatenate([
    rel_seq_sep_feat,
    xt_pair_dist_feat,
    xsc_pair_dist_feat,
    optional_pair_dist_feat
], axis=-1)
print(f"\nConcatenated features shape: {concat_features.shape}")
print(f"  Expected: (1, 5, 5, 217)")

# ============================================================================
# Step 2: Apply projection and LayerNorm
# ============================================================================

print("\n=== Applying Projection ===")

# Load weights from NPZ
weights_path = Path("/home/claudey/JuProteina/JuProteina/weights/score_network.npz")
weights = np.load(weights_path)

proj_weight = weights['pair_repr_builder.init_repr_factory.linear_out.weight']  # [256, 217]
ln_weight = weights['pair_repr_builder.init_repr_factory.ln_out.weight']  # [256]
ln_bias = weights['pair_repr_builder.init_repr_factory.ln_out.bias']  # [256]

print(f"Projection weight shape: {proj_weight.shape}")
print(f"LayerNorm weight shape: {ln_weight.shape}")

# Project: [B, L_i, L_j, 217] @ [217, 256] -> [B, L_i, L_j, 256]
# PyTorch Linear: y = x @ W.T, but our weight is already [out, in]
# So we need: [B, L_i, L_j, 217] @ [217, 256] = [B, L_i, L_j, 256]
pair_proj = np.einsum('bijd,od->bijo', concat_features, proj_weight)
print(f"After projection shape: {pair_proj.shape}")

# LayerNorm
def layer_norm(x, weight, bias, eps=1e-5):
    """Apply LayerNorm along last dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias

pair_proj_ln = layer_norm(pair_proj, ln_weight, ln_bias)
print(f"After LayerNorm shape: {pair_proj_ln.shape}")

# ============================================================================
# Step 3: Compute conditioning features
# ============================================================================

print("\n=== Computing Conditioning Features ===")

def sinusoidal_time_embedding(t, dim, max_positions=2000):
    """Create sinusoidal time embedding matching la-proteina.

    From proteinfoundation/nn/feature_factory.py:
        t = t * max_positions
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([sin(emb), cos(emb)], dim=1)

    t: [B] -> [B, dim]
    """
    t_scaled = t * max_positions
    half_dim = dim // 2
    emb_scale = np.log(max_positions) / (half_dim - 1)
    freqs = np.exp(np.arange(half_dim, dtype=np.float32) * -emb_scale)  # [half_dim]
    args = t_scaled[:, None] * freqs[None, :]  # [B, half_dim]
    embedding = np.concatenate([np.sin(args), np.cos(args)], axis=-1)  # [B, dim]
    return embedding.astype(np.float32)

# Time embeddings (256 dim each)
t_emb_bb_ca = sinusoidal_time_embedding(t_bb_ca, 256)  # [B, 256]
t_emb_local_latents = sinusoidal_time_embedding(t_local_latents, 256)  # [B, 256]

print(f"t_emb_bb_ca shape: {t_emb_bb_ca.shape}")
print(f"t_emb_local_latents shape: {t_emb_local_latents.shape}")

# Concatenate and broadcast to pair: [B, 512] -> [B, L, L, 512]
t_emb_concat = np.concatenate([t_emb_bb_ca, t_emb_local_latents], axis=-1)  # [B, 512]
t_emb_pair = t_emb_concat[:, None, None, :]  # [B, 1, 1, 512]
t_emb_pair = np.broadcast_to(t_emb_pair, (B, L, L, 512))  # [B, L, L, 512]
print(f"t_emb_pair shape: {t_emb_pair.shape}")

# Project conditioning
cond_proj_weight = weights['pair_repr_builder.cond_factory.linear_out.weight']  # [256, 512]
cond_ln_weight = weights['pair_repr_builder.cond_factory.ln_out.weight']  # [256]
cond_ln_bias = weights['pair_repr_builder.cond_factory.ln_out.bias']  # [256]

print(f"Cond projection weight shape: {cond_proj_weight.shape}")

cond_proj = np.einsum('bijd,od->bijo', t_emb_pair, cond_proj_weight)
cond_proj_ln = layer_norm(cond_proj, cond_ln_weight, cond_ln_bias)
print(f"Conditioning after projection and LN shape: {cond_proj_ln.shape}")

# ============================================================================
# Step 4: Apply AdaptiveLayerNormIdentical
# ============================================================================

print("\n=== Applying AdaptiveLayerNormIdentical ===")

adaln_norm_cond_weight = weights['pair_repr_builder.adaln.norm_cond.weight']  # [256]
adaln_norm_cond_bias = weights['pair_repr_builder.adaln.norm_cond.bias']  # [256]
adaln_to_gamma_weight = weights['pair_repr_builder.adaln.to_gamma.0.weight']  # [256, 256]
adaln_to_gamma_bias = weights['pair_repr_builder.adaln.to_gamma.0.bias']  # [256]
adaln_to_beta_weight = weights['pair_repr_builder.adaln.to_beta.weight']  # [256, 256]

print(f"AdaLN to_gamma weight shape: {adaln_to_gamma_weight.shape}")
print(f"AdaLN to_beta weight shape: {adaln_to_beta_weight.shape}")

# IMPORTANT: Python's AdaptiveLayerNormIdentical expects cond as [B, D], not [B, L, L, D]
# Since all positions have the same conditioning, we reduce by taking first position
cond_batch = cond_proj_ln[:, 0, 0, :]  # [B, D=256]
print(f"cond_batch (reduced from pair) shape: {cond_batch.shape}")

# Step 1: LayerNorm the conditioning (batch-level, not pair-level)
cond_normed = layer_norm(cond_batch, adaln_norm_cond_weight, adaln_norm_cond_bias)  # [B, D]

# Step 2: Compute gamma (with sigmoid gate) - [B, D]
gamma_linear = cond_normed @ adaln_to_gamma_weight.T + adaln_to_gamma_bias
gamma = 1.0 / (1.0 + np.exp(-gamma_linear))  # sigmoid [B, D]

# Step 3: Compute beta (no bias) - [B, D]
beta = cond_normed @ adaln_to_beta_weight.T  # [B, D]

print(f"gamma shape: {gamma.shape}, range: [{gamma.min():.4f}, {gamma.max():.4f}]")
print(f"beta shape: {beta.shape}, range: [{beta.min():.4f}, {beta.max():.4f}]")

# Step 4: Apply LayerNorm (no affine) to x, then gamma * x + beta
# Python's AdaptiveLayerNormIdentical has: self.norm = LayerNorm(dim, elementwise_affine=False)
# It applies: normed = self.norm(x), then output = normed * gamma_brc + beta_brc

def layer_norm_no_affine(x, eps=1e-5):
    """LayerNorm without learnable affine parameters, normalizing along last dim."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

# Apply no-affine LayerNorm to pair_proj_ln
normed_x = layer_norm_no_affine(pair_proj_ln)  # [B, L, L, D]
print(f"normed_x shape: {normed_x.shape}")

# Broadcast gamma and beta from [B, D] to [B, 1, 1, D] for pair mode
gamma_brc = gamma[:, None, None, :]  # [B, 1, 1, D]
beta_brc = beta[:, None, None, :]    # [B, 1, 1, D]

# Apply: output = normed_x * gamma_brc + beta_brc
final_pair_repr = normed_x * gamma_brc + beta_brc

# Apply pair mask
pair_mask = mask[:, :, None] * mask[:, None, :]  # [B, L, L]
pair_mask = pair_mask[:, :, :, None]  # [B, L, L, 1]
final_pair_repr = final_pair_repr * pair_mask

print(f"Final pair repr shape: {final_pair_repr.shape}")
print(f"Final pair repr range: [{final_pair_repr.min():.4f}, {final_pair_repr.max():.4f}]")

# ============================================================================
# Save all intermediate values
# ============================================================================

output_dir = Path('/home/claudey/JuProteina/JuProteina/test')

# Save inputs (Python format [B, L, D])
np.save(output_dir / 'pair_debug_x_t_bb_ca.npy', x_t_bb_ca)
np.save(output_dir / 'pair_debug_x_t_local_latents.npy', x_t_local_latents)
np.save(output_dir / 'pair_debug_x_sc_bb_ca.npy', x_sc_bb_ca)
np.save(output_dir / 'pair_debug_x_sc_local_latents.npy', x_sc_local_latents)
np.save(output_dir / 'pair_debug_t_bb_ca.npy', t_bb_ca)
np.save(output_dir / 'pair_debug_t_local_latents.npy', t_local_latents)
np.save(output_dir / 'pair_debug_mask.npy', mask)

# Save raw features (Python format [B, L, L, D])
np.save(output_dir / 'pair_debug_rel_seq_sep.npy', rel_seq_sep_feat)
np.save(output_dir / 'pair_debug_xt_pair_dist.npy', xt_pair_dist_feat)
np.save(output_dir / 'pair_debug_xsc_pair_dist.npy', xsc_pair_dist_feat)
np.save(output_dir / 'pair_debug_optional_pair_dist.npy', optional_pair_dist_feat)
np.save(output_dir / 'pair_debug_concat_features.npy', concat_features)

# Save after projection and LN
np.save(output_dir / 'pair_debug_pair_proj.npy', pair_proj)
np.save(output_dir / 'pair_debug_pair_proj_ln.npy', pair_proj_ln)

# Save conditioning
np.save(output_dir / 'pair_debug_t_emb_bb_ca.npy', t_emb_bb_ca)
np.save(output_dir / 'pair_debug_t_emb_local_latents.npy', t_emb_local_latents)
np.save(output_dir / 'pair_debug_t_emb_pair.npy', t_emb_pair)
np.save(output_dir / 'pair_debug_cond_proj.npy', cond_proj)
np.save(output_dir / 'pair_debug_cond_proj_ln.npy', cond_proj_ln)

# Save AdaLN intermediates
np.save(output_dir / 'pair_debug_cond_normed.npy', cond_normed)
np.save(output_dir / 'pair_debug_gamma.npy', gamma)
np.save(output_dir / 'pair_debug_beta.npy', beta)

# Save final output
np.save(output_dir / 'pair_debug_final_pair_repr.npy', final_pair_repr)

print("\nSaved all debug data to test/")

# Print some sample values for verification
print("\n=== Sample Values ===")
print(f"concat_features[0, 0, 1, :5]: {concat_features[0, 0, 1, :5]}")
print(f"pair_proj[0, 0, 1, :5]: {pair_proj[0, 0, 1, :5]}")
print(f"pair_proj_ln[0, 0, 1, :5]: {pair_proj_ln[0, 0, 1, :5]}")
print(f"cond_proj_ln[0, 0, 1, :5]: {cond_proj_ln[0, 0, 1, :5]}")
print(f"gamma[0, :5]: {gamma[0, :5]}")
print(f"beta[0, :5]: {beta[0, :5]}")
print(f"final_pair_repr[0, 0, 1, :5]: {final_pair_repr[0, 0, 1, :5]}")
