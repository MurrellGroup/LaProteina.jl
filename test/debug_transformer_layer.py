#!/usr/bin/env python3
"""
Debug transformer layer values.
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

output_dir = Path('/home/claudey/JuProteina/JuProteina/test')

with torch.no_grad():
    # Manually step through the model
    init_repr_factory = model.init_repr_factory
    cond_factory = model.cond_factory
    pair_repr_builder = model.pair_repr_builder
    transition_c_1 = model.transition_c_1
    transition_c_2 = model.transition_c_2
    transformer_layers = model.transformer_layers

    mask = batch["mask"]

    # Initial representations
    seqs = init_repr_factory(batch)
    cond = cond_factory(batch)
    pair = pair_repr_builder(batch)

    print(f"=== INITIAL REPRESENTATIONS ===")
    print(f"seqs shape: {seqs.shape}")
    print(f"seqs[0,0,:5]: {seqs[0, 0, :5]}")
    print(f"cond shape: {cond.shape}")
    print(f"cond[0,0,:5]: {cond[0, 0, :5]}")
    print(f"pair shape: {pair.shape}")
    print(f"pair[0,0,0,:5]: {pair[0, 0, 0, :5]}")

    # Conditioning transitions
    cond = transition_c_1(cond, mask)
    print(f"\n=== AFTER TRANS_C_1 ===")
    print(f"cond[0,0,:5]: {cond[0, 0, :5]}")

    cond = transition_c_2(cond, mask)
    print(f"\n=== AFTER TRANS_C_2 ===")
    print(f"cond[0,0,:5]: {cond[0, 0, :5]}")

    # First transformer layer
    layer0 = transformer_layers[0]
    print(f"\n=== TRANSFORMER LAYER 0 ===")
    print(f"Layer type: {type(layer0)}")

    seqs_after_layer0 = layer0(seqs, pair, cond, mask)
    print(f"seqs after layer0 shape: {seqs_after_layer0.shape}")
    print(f"seqs_after_layer0[0,0,:5]: {seqs_after_layer0[0, 0, :5]}")
    print(f"seqs_after_layer0[0,0,-5:]: {seqs_after_layer0[0, 0, -5:]}")

    # Save for comparison
    np.save(output_dir / 'debug_seqs_after_layer0.npy', seqs_after_layer0.numpy())

    # Run all layers and save final seqs
    seqs = seqs_after_layer0
    for i, layer in enumerate(transformer_layers[1:], start=1):
        seqs = layer(seqs, pair, cond, mask)

    print(f"\n=== AFTER ALL TRANSFORMER LAYERS ===")
    print(f"seqs shape: {seqs.shape}")
    print(f"seqs[0,0,:5]: {seqs[0, 0, :5]}")
    print(f"seqs[0,0,-5:]: {seqs[0, 0, -5:]}")

    np.save(output_dir / 'debug_seqs_after_all_layers.npy', seqs.numpy())

    # Output projections
    ca_out = model.ca_linear(seqs)
    ll_out = model.local_latents_linear(seqs)

    print(f"\n=== OUTPUT PROJECTIONS ===")
    print(f"ca_out shape: {ca_out.shape}")
    print(f"ca_out[0,0,:]: {ca_out[0, 0, :]}")
    print(f"ll_out shape: {ll_out.shape}")
    print(f"ll_out[0,0,:5]: {ll_out[0, 0, :5]}")

    np.save(output_dir / 'debug_ca_out.npy', ca_out.numpy())
    np.save(output_dir / 'debug_ll_out.npy', ll_out.numpy())

    print(f"\nSaved debug data to {output_dir}")
