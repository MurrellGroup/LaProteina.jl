#!/usr/bin/env python3
"""
Generate decoder reference data for Julia parity tests.

Loads the decoder from AE1_ucond_512.ckpt and runs a forward pass
with deterministic test inputs, saving outputs for comparison with Julia.
"""

import sys
sys.path.insert(0, '/home/claudey/JuProteina/la-proteina')
import torch_scatter_compat  # Must be before proteinfoundation imports

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Load decoder model directly
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer

# Checkpoint path
ckpt_path = "/home/claudey/JuProteina/la-proteina/checkpoints_laproteina/AE1_ucond_512.ckpt"

# Load checkpoint - it contains the hyperparameters used for training
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Get the config from checkpoint's hyperparameters
hparams = ckpt.get('hyper_parameters', {})
cfg_ae = hparams.get('cfg_ae', None)

if cfg_ae is None:
    # Fallback to config file
    config_path = "/home/claudey/JuProteina/la-proteina/configs/training_ae.yaml"
    config = OmegaConf.load(config_path)
    nn_ae_config = OmegaConf.load("/home/claudey/JuProteina/la-proteina/configs/nn_ae/nn_130m.yaml")
    config.nn_ae = nn_ae_config
    cfg_ae = config

# Create decoder with checkpoint's config
print("Creating decoder with checkpoint config...")
nn_ae = cfg_ae.nn_ae if hasattr(cfg_ae, 'nn_ae') else cfg_ae['nn_ae']
decoder = DecoderTransformer(**OmegaConf.to_container(nn_ae))

# Load only decoder weights from checkpoint state_dict
state_dict = ckpt['state_dict']
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
decoder.load_state_dict(decoder_state_dict)
decoder.eval()

print("Loaded decoder model")
print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

# Create deterministic test input
torch.manual_seed(42)
np.random.seed(42)

B, L = 1, 10
latent_dim = 8

# Create decoder inputs
# z_latent: [B, L, latent_dim]
z_latent = torch.randn(B, L, latent_dim) * 0.5
# ca_coors: [B, L, 3] - CA coordinates in nanometers
ca_coors = torch.randn(B, L, 3) * 0.5
# mask: [B, L] boolean
mask = torch.ones(B, L, dtype=torch.bool)

print(f"\nInput shapes:")
print(f"  z_latent: {z_latent.shape}")
print(f"  ca_coors: {ca_coors.shape}")
print(f"  mask: {mask.shape}")

# Create decoder input dict
decoder_input = {
    'z_latent': z_latent,
    'ca_coors_nm': ca_coors,
    'residue_mask': mask,
    'mask': mask,
}

# Forward pass through decoder
with torch.no_grad():
    output = decoder(decoder_input)

# Extract outputs
coors_nm = output['coors_nm']  # [B, L, 37, 3]
seq_logits = output['seq_logits']  # [B, L, 20]
aatype_max = output['aatype_max']  # [B, L]
atom_mask = output['atom_mask']  # [B, L, 37]
residue_mask = output['residue_mask']  # [B, L]

print(f"\nOutput shapes:")
print(f"  coors_nm: {coors_nm.shape}")
print(f"  seq_logits: {seq_logits.shape}")
print(f"  aatype_max: {aatype_max.shape}")
print(f"  atom_mask: {atom_mask.shape}")

print(f"\nOutput values (first residue):")
print(f"  coors_nm[0, 0, 0, :]: {coors_nm[0, 0, 0, :]}")  # First atom coords
print(f"  coors_nm[0, 0, 1, :]: {coors_nm[0, 0, 1, :]}")  # CA coords (should match input)
print(f"  seq_logits[0, 0, :5]: {seq_logits[0, 0, :5]}")
print(f"  aatype_max[0, :5]: {aatype_max[0, :5]}")

# Verify CA coordinates match input (abs_coors=False means relative, CA should be same as input)
print(f"\nCA coordinate check:")
print(f"  Input CA[0, 0, :]: {ca_coors[0, 0, :]}")
print(f"  Output CA[0, 0, :]: {coors_nm[0, 0, 1, :]}")

# Save for Julia comparison
output_dir = Path('/home/claudey/JuProteina/JuProteina/test')

# Save inputs
np.save(output_dir / 'decoder_z_latent.npy', z_latent.numpy())
np.save(output_dir / 'decoder_ca_coors.npy', ca_coors.numpy())
np.save(output_dir / 'decoder_mask.npy', mask.numpy().astype(np.float32))

# Save outputs
np.save(output_dir / 'decoder_out_coors.npy', coors_nm.numpy())
np.save(output_dir / 'decoder_out_logits.npy', seq_logits.numpy())
np.save(output_dir / 'decoder_out_aatype.npy', aatype_max.numpy())
np.save(output_dir / 'decoder_out_atom_mask.npy', atom_mask.numpy())

print(f"\nSaved decoder test data to {output_dir}")

# Also save decoder weights for Julia
print("\nExtracting decoder weights...")
decoder_weights = {}
for name, param in decoder.named_parameters():
    # Convert to numpy
    decoder_weights[name.replace('.', '_')] = param.detach().cpu().numpy()

# Also get buffers (like layer norm running stats if any)
for name, buffer in decoder.named_buffers():
    decoder_weights[name.replace('.', '_')] = buffer.detach().cpu().numpy()

print(f"  Found {len(decoder_weights)} decoder weight tensors")
np.savez(output_dir.parent / 'weights' / 'decoder_new.npz', **decoder_weights)
print(f"  Saved to {output_dir.parent / 'weights' / 'decoder_new.npz'}")
