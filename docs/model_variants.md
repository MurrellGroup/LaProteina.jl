# LaProteina Model Variants (LD1-LD7 + AE1-AE3)

La-proteina has 7 latent diffusion (score network) variants and 3 autoencoder variants.
All share the same base architecture but differ in feature composition, training data,
and sampling parameters.

## Autoencoder Variants

All three autoencoders share identical architecture: 12-layer transformer, 768 token dim,
256 pair dim, 12 heads, 128 conditioning dim, 8-dim latent space, QK layer norm enabled.

| Autoencoder | Checkpoint | Max Length | Purpose |
|---|---|---|---|
| AE1 | `AE1_ucond_512.safetensors` | 512 | Unconditional, standard-length proteins |
| AE2 | `AE2_ucond_800.safetensors` | 896 | Unconditional, long proteins (300-800 res) |
| AE3 | `AE3_motif.safetensors` | 256 | Motif-conditioned scaffolding |

Constructor (same for all three):
```julia
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=128, latent_dim=8,
    qk_ln=true, update_pair_repr=false,
)
load_decoder_weights_st!(decoder, checkpoint_path)  # prefix="decoder." by default
```

SafeTensors files contain both encoder and decoder weights. Use `prefix="decoder."` for
the decoder and `prefix="encoder."` for the encoder.

## Score Network Variants

All score networks: 14 transformer layers, 768 token dim, 256 pair dim, 12 heads,
256 conditioning dim, 8-dim latent, v-parameterization, QK layer norm.

### LD1: Unconditional, No Triangular Update (up to 512 residues)

The base model. No pair update triangular multiplication.

- **Checkpoint**: `LD1_ucond_notri_512.safetensors`
- **Decoder**: AE1
- **Architecture**: `cropped_flag=true` (46D seq features = 45 base + 1 crop flag)
- **Pair features**: 217D base

```julia
sn = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v, qk_ln=true,
    cropped_flag=true,
)
```

### LD2: Unconditional, Triangular Update (up to 512 residues)

Same as LD1 but with triangular multiplicative pair updates every 2 layers.

- **Checkpoint**: `LD2_ucond_tri_512.safetensors`
- **Decoder**: AE1
- **Architecture**: `update_pair_repr=true`, `use_tri_mult=true`, `update_pair_every_n=2`
- **Features**: Same as LD1

```julia
sn = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v, qk_ln=true,
    update_pair_repr=true, use_tri_mult=true, update_pair_every_n=2,
)
```

### LD3: Unconditional, Long Proteins (300-800 residues)

Identical architecture to LD1 but trained on longer proteins with different noise scales.

- **Checkpoint**: `LD3_ucond_notri_800.safetensors`
- **Decoder**: AE2
- **Architecture**: Same as LD1
- **Sampling difference**: CA noise 0.15 (vs 0.1), latent noise 0.05 (vs 0.1)

```julia
sn = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v, qk_ln=true,
)
```

### LD4: Indexed Motif Scaffolding, All-Atom (up to 256 residues)

Motif-conditioned model that takes all-atom motif coordinates as input features.
Indexed means motif residues have known positions in the output sequence.

- **Checkpoint**: `LD4_motif_idx_aa.safetensors`
- **Decoder**: AE3
- **Architecture**: `motif_mode=:aa` (549D seq features, 301D pair features)
- **Seq features**: 45 base + 504 BulkAllAtomXmotif (abs coords 148 + rel coords 148 + seq 20 + sidechain angles 88 + torsion angles 63 + mask 37)
- **Pair features**: 217 base + 84 motif pair dists (30 CA + 30 CB + 24 tip)

```julia
sn = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v, qk_ln=true,
    motif_mode=:aa,
)
```

### LD5: Indexed Motif Scaffolding, Tip-Atom (up to 256 residues)

Like LD4 but uses only sidechain tip atoms instead of all atoms.

- **Checkpoint**: `LD5_motif_idx_tip.safetensors`
- **Decoder**: AE3
- **Architecture**: `motif_mode=:tip` (250D seq features, 217D pair features)
- **Seq features**: 45 base + 205 motif (abs coords 148 + seq 20 + mask 37)
- **Pair features**: 217D base only (no motif pair dists)

```julia
sn = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v, qk_ln=true,
    motif_mode=:tip,
)
```

### LD6: Unindexed Motif Scaffolding, All-Atom

Like LD4 but with unindexed motif conditioning (motif position is not fixed in the
output sequence). Uses a different network architecture class.

- **Checkpoint**: `LD6_motif_uidx_aa.safetensors`
- **Decoder**: AE3
- **Status**: Architecture class `LocalLatentsTransformerMotifUidx` not yet implemented in Julia. Currently runs unconditionally.

### LD7: Unindexed Motif Scaffolding, Tip-Atom

Like LD5 but with unindexed motif conditioning.

- **Checkpoint**: `LD7_motif_uidx_tip.safetensors`
- **Decoder**: AE3
- **Status**: Same as LD6 (unindexed architecture not yet in Julia).

## Sampling Parameters

All models share the same base sampling configuration with small per-model overrides.

### Base Configuration (all models)

```
Steps:              400
Self-conditioning:  true

BB_CA (Alpha Carbon):
  Time schedule:    log, p=2.0
  SDE gt mode:      1/t, param=1.0
  Noise scale:      0.1       (exception: LD3 uses 0.15)
  Score scale:      1.0
  t_lim_ode:        0.98      (switch to ODE above this)
  t_lim_ode_below:  0.02      (switch to ODE below this) [NOT YET IN JULIA]
  zero_com:         true
  center_every_step: true     (exception: LD4-7 use false) [NOT YET IN JULIA]

Local Latents:
  Time schedule:    power, p=2.0
  SDE gt mode:      tan, param=1.0
  Noise scale:      0.1       (exception: LD3 uses 0.05)
  Score scale:      1.0
  t_lim_ode:        0.98
  t_lim_ode_below:  0.02      [NOT YET IN JULIA]
  zero_com:         false
  center_every_step: false
```

### Per-Model Overrides

| Parameter | LD1 | LD2 | LD3 | LD4 | LD5 | LD6 | LD7 |
|---|---|---|---|---|---|---|---|
| CA noise scale | 0.1 | 0.1 | **0.15** | 0.1 | 0.1 | 0.1 | 0.1 |
| Latent noise scale | 0.1 | 0.1 | **0.05** | 0.1 | 0.1 | 0.1 | 0.1 |
| CA center_every_step | true | true | true | **false** | **false** | **false** | **false** |

### Piecewise SDE Behavior

All models use a three-interval piecewise SDE during sampling:

1. **t in [0, 0.02)**: Pure ODE (deterministic, no noise)
2. **t in [0.02, 0.98]**: SDE with stochastic noise injection
3. **t in (0.98, 1.0]**: Pure ODE (deterministic approach to data distribution)

The SDE region uses:
```
dx = [v + g(t) * score] dt + sqrt(2 * g(t) * noise_scale) dW
```
where g(t) is the noise coefficient: `1/t` for CA, `(pi/2)*tan((1-t)*pi/2)` for latents.

### Known Julia Gaps

Two Python sampling parameters are not yet in the Julia Flowfusion.jl implementation:

1. **`t_lim_ode_below`**: Python uses ODE for t < 0.02. Julia only implements the upper
   bound (`t_lim_ode`). Impact: minor — the low-t ODE region is small.

2. **`center_every_step`**: Python separates zero-COM noise sampling from per-step
   re-centering. In Julia's Flowfusion.jl, `zero_com=true` does both. For motif models
   (LD4-7), we need zero-COM noise but NOT per-step re-centering (because motif positions
   are fixed constraints). Impact: only affects motif-conditioned generation, not
   unconditional sampling.

## Weight Loading

All SafeTensors weights use strict validation by default. Loading errors on:
- Wrong prefix (e.g., using `"nn."` for decoder weights that use `"decoder."`)
- Missing weights (model expects parameters not in the file)
- Extra weights (file has parameters the model didn't load)

```julia
# Score networks: prefix "nn." (default)
load_score_network_weights_st!(sn, "LD1_ucond_notri_512.safetensors")

# Decoders: prefix "decoder." (default)
load_decoder_weights_st!(dec, "AE1_ucond_512.safetensors")

# Encoders: prefix "encoder." (default)
load_encoder_weights_st!(enc, "AE1_ucond_512.safetensors")

# Override prefix if needed
load_score_network_weights_st!(sn, path; prefix="custom.")

# Disable strict validation (NOT recommended)
load_decoder_weights_st!(dec, path; strict=false)
```
