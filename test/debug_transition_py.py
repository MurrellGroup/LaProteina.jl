#!/usr/bin/env python3
"""
Debug transition layer in Python.
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
    print(f"pair: {pair.shape} sample: {pair[0, 0, 0, :5]}")

    # Get layer 0 components
    layer0 = model.transformer_layers[0]
    mhba = layer0.mhba
    transition = layer0.transition

    # Run MHBA
    x_mha = mhba(seqs, pair, cond, mask)
    print(f"\n=== AFTER MHA ===")
    print(f"x_mha: {x_mha.shape} sample: {x_mha[0, 0, :5]}")

    # Add residual
    x_after_mha = seqs + x_mha
    x_after_mha = x_after_mha * mask[..., None]
    print(f"x_after_mha: {x_after_mha.shape} sample: {x_after_mha[0, 0, :5]}")

    # Run transition
    print("\n=== TRANSITION DEBUG ===")
    trans_adaln = transition.adaln
    trans_inner = transition.transition
    trans_scale = transition.scale_output

    x_trans_adaln = trans_adaln(x_after_mha, cond, mask)
    print(f"After transition ADALN: {x_trans_adaln[0, 0, :5]}")
    print(f"After transition ADALN stats: min={x_trans_adaln.min():.4f} max={x_trans_adaln.max():.4f}")

    # Inner transition (SwiGLU)
    x_trans_inner = trans_inner(x_trans_adaln, mask)
    print(f"After transition inner (SwiGLU): {x_trans_inner[0, 0, :5]}")
    print(f"After transition inner stats: min={x_trans_inner.min():.4f} max={x_trans_inner.max():.4f}")

    # Step through inner transition manually
    print("\n  === INNER TRANSITION STEP BY STEP ===")
    if trans_inner.use_layer_norm:
        x_ln = trans_inner.ln(x_trans_adaln)
        print(f"  After LayerNorm: {x_ln[0, 0, :5]}")
    else:
        x_ln = x_trans_adaln
        print(f"  No LayerNorm")

    # swish_linear is Sequential(Linear, SwiGLU)
    x_proj = trans_inner.swish_linear[0](x_ln)  # Linear
    print(f"  After linear_in: {x_proj.shape} sample={x_proj[0, 0, :5]}")
    print(f"  After linear_in stats: min={x_proj.min():.4f} max={x_proj.max():.4f}")

    x_swiglu = trans_inner.swish_linear[1](x_proj)  # SwiGLU
    print(f"  After SwiGLU: {x_swiglu.shape} sample={x_swiglu[0, 0, :5]}")
    print(f"  After SwiGLU stats: min={x_swiglu.min():.4f} max={x_swiglu.max():.4f}")

    x_out = trans_inner.linear_out(x_swiglu)  # linear_out
    print(f"  After linear_out: {x_out.shape} sample={x_out[0, 0, :5]}")
    print(f"  After linear_out stats: min={x_out.min():.4f} max={x_out.max():.4f}")

    x_trans_scale = trans_scale(x_trans_inner, cond, mask)
    print(f"\nAfter transition scale: {x_trans_scale[0, 0, :5]}")
    print(f"After transition scale stats: min={x_trans_scale.min():.4f} max={x_trans_scale.max():.4f}")

    # Full transition
    x_tr = transition(x_after_mha, cond, mask)
    print(f"\nFull transition output: {x_tr[0, 0, :5]}")
    print(f"Full transition output stats: min={x_tr.min():.4f} max={x_tr.max():.4f}")

    # Final layer 0 output
    x_final = x_after_mha + x_tr
    x_final = x_final * mask[..., None]
    print(f"\n=== LAYER 0 FINAL OUTPUT ===")
    print(f"x_final: {x_final.shape} sample: {x_final[0, 0, :5]}")
    print(f"x_final stats: min={x_final.min():.4f} max={x_final.max():.4f}")

    # Compare with full layer
    x_layer0 = layer0(seqs, pair, cond, mask)
    print(f"\nx from full layer0: {x_layer0[0, 0, :5]}")
    print(f"x from full layer0 stats: min={x_layer0.min():.4f} max={x_layer0.max():.4f}")
