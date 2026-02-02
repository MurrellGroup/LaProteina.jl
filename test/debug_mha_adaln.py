#!/usr/bin/env python3
"""
Debug MHA ADALN step.
"""

import sys
sys.path.insert(0, '/home/claudey/BFlaproteina/la-proteina')
import torch_scatter_compat

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

# Load checkpoint
ckpt_path = "/home/claudey/BFlaproteina/la-proteina/checkpoints_laproteina/LD1_ucond_notri_512.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Load config
config = OmegaConf.load("/home/claudey/BFlaproteina/la-proteina/configs/nn/local_latents_score_nn_160M.yaml")
config.feats_seq = list(config.feats_seq) + ["cropped_flag_seq"]

# Create model
model = LocalLatentsTransformer(**OmegaConf.to_container(config), latent_dim=8)
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

with torch.no_grad():
    mask = batch["mask"]

    # Initial representations
    seqs = model.init_repr_factory(batch)
    cond = model.cond_factory(batch)
    pair = model.pair_repr_builder(batch)

    # Mask seqs
    seqs = seqs * mask[..., None]

    # Conditioning transitions
    cond = model.transition_c_1(cond, mask)
    cond = model.transition_c_2(cond, mask)

    print("=== INPUTS TO LAYER 0 ===")
    print(f"seqs: {seqs.shape} sample: {seqs[0, 0, :5]}")
    print(f"cond: {cond.shape} sample: {cond[0, 0, :5]}")

    # Get MHBA components
    layer0 = model.transformer_layers[0]
    mhba = layer0.mhba
    adaln = mhba.adaln
    mha = mhba.mha  # PairBiasAttnWithLN
    scale_output = mhba.scale_output

    print("\n=== MHA ADALN STEP ===")
    x_in = seqs * mask[..., None]
    print(f"x_in to MHA (after mask): {x_in[0, 0, :5]}")

    # Apply ADALN
    x_adaln = adaln(x_in, cond, mask)
    print(f"x after adaln: {x_adaln[0, 0, :5]}")
    print(f"x after adaln stats: min={x_adaln.min():.4f} max={x_adaln.max():.4f}")

    # ADALN internals
    print("\n  ADALN internals:")
    normed = adaln.norm(x_in)
    print(f"  normed: {normed[0, 0, :5]}")
    print(f"  normed stats: min={normed.min():.4f} max={normed.max():.4f}")

    normed_cond = adaln.norm_cond(cond)
    print(f"  normed_cond: {normed_cond[0, 0, :5]}")

    gamma = adaln.to_gamma(normed_cond)
    print(f"  gamma: {gamma[0, 0, :5]}")
    print(f"  gamma stats: min={gamma.min():.4f} max={gamma.max():.4f}")

    beta = adaln.to_beta(normed_cond)
    print(f"  beta: {beta[0, 0, :5]}")
    print(f"  beta stats: min={beta.min():.4f} max={beta.max():.4f}")

    out_manual = normed * gamma + beta
    print(f"  normed * gamma + beta: {out_manual[0, 0, :5]}")

    # Save intermediate for Julia comparison
    output_dir = Path('/home/claudey/JuProteina/JuProteina/test')
    np.save(output_dir / 'debug_normed.npy', normed.numpy())
    np.save(output_dir / 'debug_normed_cond.npy', normed_cond.numpy())
    np.save(output_dir / 'debug_gamma.npy', gamma.numpy())
    np.save(output_dir / 'debug_beta.npy', beta.numpy())
    np.save(output_dir / 'debug_mha_adaln_out.npy', x_adaln.numpy())
    print(f"\nSaved debug data")
    print("Exiting early for comparison")
    sys.exit(0)

    # Apply attention
    print("\n=== ATTENTION STEP ===")
    # The mha expects (x, pair_feats, mask) based on PairBiasAttnWithLN
    x_attn = mha(node_feats=x_adaln, pair_feats=pair, mask=mask)
    print(f"x after attention: {x_attn[0, 0, :5]}")
    print(f"x after attention stats: min={x_attn.min():.4f} max={x_attn.max():.4f}")

    # Scale output
    print("\n=== SCALE OUTPUT ===")
    x_scaled = scale_output(x_attn, cond, mask)
    print(f"x after scale: {x_scaled[0, 0, :5]}")
    print(f"x after scale stats: min={x_scaled.min():.4f} max={x_scaled.max():.4f}")

    # Full MHA
    print("\n=== FULL MHBA ===")
    x_mha = mhba(seqs, pair, cond, mask)
    print(f"x_mha: {x_mha[0, 0, :5]}")
    print(f"x_mha stats: min={x_mha.min():.4f} max={x_mha.max():.4f}")

    # Save intermediate for Julia comparison
    output_dir = Path('/home/claudey/JuProteina/JuProteina/test')
    np.save(output_dir / 'debug_mha_adaln_out.npy', x_adaln.numpy())
    np.save(output_dir / 'debug_mha_attn_out.npy', x_attn.numpy())
    np.save(output_dir / 'debug_mha_scale_out.npy', x_scaled.numpy())
    np.save(output_dir / 'debug_normed.npy', normed.numpy())
    np.save(output_dir / 'debug_normed_cond.npy', normed_cond.numpy())
    np.save(output_dir / 'debug_gamma.npy', gamma.numpy())
    np.save(output_dir / 'debug_beta.npy', beta.numpy())
    print(f"\nSaved debug data")
