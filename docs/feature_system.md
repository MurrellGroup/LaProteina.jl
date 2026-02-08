# Feature Extraction System

Complete guide to LaProteina's feature extraction pipeline.

## Architecture

The feature system has three layers:

1. **Feature**: Abstract type representing a single feature (e.g., CA coordinates, time embedding). Each Feature knows its dimension and how to extract raw values from a batch.
2. **FeatureFactory**: Groups multiple Features, concatenates their outputs, and projects through a Dense layer to the target dimension.
3. **Pre-configured factories**: Functions like `score_network_seq_features(768)` that return FeatureFactories with the correct features for each model component.

```
batch Dict → Feature.extract() → [raw_dim, L, B] → concat → Dense(raw_dim → out_dim) → [out_dim, L, B]
```

## Feature Types

### Sequence Features (Input: batch Dict, Output: [dim, L, B])

| Feature | Dim | Key | Description |
|---------|-----|-----|-------------|
| `XtBBCAFeature` | 3 | `:x_t[:bb_ca]` | Noisy CA coordinates at time t |
| `XtLocalLatentsFeature` | 8 | `:x_t[:local_latents]` | Noisy latent vectors at time t |
| `XscBBCAFeature` | 3 | `:x_sc[:bb_ca]` | Self-conditioning CA prediction |
| `XscLocalLatentsFeature` | 8 | `:x_sc[:local_latents]` | Self-conditioning latent prediction |
| `OptionalCACoorsFeature` | 3 | `:optional[:ca_coors]` | Optional conditioning CA coords (zeros if absent) |
| `OptionalResTypeFeature` | 20 | `:optional[:res_type]` | Optional conditioning residue type (one-hot, zeros if absent) |
| `CroppedFlagFeature` | 1 | `:cropped_flag` | Whether positions are cropped (all zeros for unconditional) |
| `TimeFeature` | configurable | `:t[:bb_ca]` or `:t[:local_latents]` | Sinusoidal time embedding |
| `PositionFeature` | configurable | (computed) | Sinusoidal position embedding |
| `CACoordFeature` | 3 | `:ca_coors` | CA coordinates (decoder/encoder input) |
| `LatentFeature` | 8 | `:z_latent` | Latent vectors (decoder input) |
| `ChainBreakFeature` | 1 | `:chain_break` | Chain break indicator |
| `ResidueTypeFeature` | 20 | `:aatype` | One-hot residue type |
| `Atom37CoordFeature` | 148 | `:coords`, `:atom_mask` | Atom37 coords (37*3) + mask (37), absolute or relative |
| `BackboneTorsionFeature` | 63 | `:coords` | Backbone torsion angles (phi, psi, omega), binned into 21 bins each |
| `SidechainAngleFeature` | 88 | `:coords` | Sidechain chi angles (4 * 21 bins + 4 mask values) |
| `ChainIdxSeqFeature` | 1 | `:chain_idx` | Chain index (multi-chain support) |

### Pair Features (Input: batch Dict, Output: [dim, L, L, B])

| Feature | Dim | Key | Description |
|---------|-----|-----|-------------|
| `RelSeqSepFeature` | 127 | (computed) | Relative sequence separation, one-hot encoded, clamped to [-63, 63] |
| `XtBBCAPairDistFeature` | 30 | `:x_t[:bb_ca]` | Binned pairwise CA distances at time t (0.1-3.0 nm, 30 bins) |
| `XscBBCAPairDistFeature` | 30 | `:x_sc[:bb_ca]` | Binned pairwise CA distances from SC prediction |
| `OptionalCAPairDistFeature` | 30 | `:optional[:ca_coors]` | Optional conditioning pairwise distances |
| `CAPairDistFeature` | 30 | `:ca_coors` | CA pairwise distances (decoder/encoder) |
| `BackbonePairDistFeature` | 84 | `:coords` | Backbone atom pair distances (N, CA, C, O), 4 * 21 bins |
| `ResidueOrientationFeature` | 105 | `:coords` | Pairwise backbone orientations, 5 * 21 bins |
| `ChainIdxPairFeature` | 1 | `:chain_idx` | Same-chain indicator |
| `TimePairFeature` | configurable | `:t[:bb_ca]` or `:t[:local_latents]` | Time embedding broadcast to pair dimensions |
| `DistanceBinFeature` | configurable | (computed) | Generic binned pairwise distances (legacy) |

### Time Embedding Features

| Feature | Dim | Description |
|---------|-----|-------------|
| `TimeFeature(dim, modality)` | dim (e.g. 256) | Sinusoidal embedding of schedule-transformed time, per modality |
| `TimePairFeature(dim, modality)` | dim | Same as TimeFeature but broadcast to [dim, L, L, B] |

Time embeddings use sinusoidal encoding with geometric frequency scaling, matching PyTorch's implementation.

## Pre-Configured Feature Factories

### ScoreNetwork Features

| Factory | Input Dims | Output Dim | Features |
|---------|-----------|------------|----------|
| `score_network_seq_features(768)` | 46 → 768 | token_dim | XtBBCA(3) + XtLatents(8) + XscBBCA(3) + XscLatents(8) + OptionalCA(3) + OptionalResType(20) + CroppedFlag(1) |
| `score_network_cond_features(256)` | 512 → 256 | dim_cond | TimeFeature_CA(256) + TimeFeature_LL(256) |
| `score_network_pair_features(256)` | 217 → 256 | pair_dim | RelSeqSep(127) + XtPairDist(30) + XscPairDist(30) + OptionalPairDist(30) |
| `score_network_pair_cond_features(256)` | 512 → 256 | pair_dim | TimePairFeature_CA(256) + TimePairFeature_LL(256) |

### Encoder Features

| Factory | Input Dims | Output Dim | Features |
|---------|-----------|------------|----------|
| `encoder_seq_features(768)` | 468 → 768 | token_dim | ChainBreak(1) + ResidueType(20) + Atom37Coord(148) + Atom37CoordRel(148) + BackboneTorsion(63) + SidechainAngle(88) |
| `encoder_cond_features(128)` | 0 → 128 | dim_cond | (zeros — no conditioning features) |
| `encoder_pair_features(256)` | 316 → 256 | pair_dim | RelSeqSep(127) + BackbonePairDist(84) + ResidueOrientation(105) |

### Decoder Features

| Factory | Input Dims | Output Dim | Features |
|---------|-----------|------------|----------|
| `decoder_seq_features(768)` | 11 → 768 | token_dim | CACoord(3) + Latent(8) |
| `decoder_cond_features(128)` | 0 → 128 | dim_cond | (zeros — no conditioning features) |
| `decoder_pair_features(256)` | 157 → 256 | pair_dim | RelSeqSep(127) + CAPairDist(30) |

## Self-Conditioning Features

The following channels in the raw feature tensors are SC-dependent (updated during self-conditioning passes):

### Sequence SC Channels

| Feature | Dims | Offset in seq_raw |
|---------|------|-------------------|
| XscBBCAFeature | 3 | Channels 12-14 |
| XscLocalLatentsFeature | 8 | Channels 15-22 |

Total: 11 SC-dependent sequence dimensions out of 46.

### Pair SC Channels

| Feature | Dims | Offset in pair_raw |
|---------|------|-------------------|
| XscBBCAPairDistFeature | 30 | Channels 158-187 |

Total: 30 SC-dependent pair dimensions out of 217.

### In-Place Update

`compute_sc_feature_offsets(model)` returns the exact channel ranges for each SC feature. `update_sc_raw_features!()` overwrites only these channels on GPU, avoiding the cost of re-extracting all features.

## ScoreNetworkRawFeatures

Container for unprojected features (extracted outside the gradient context):

```julia
struct ScoreNetworkRawFeatures{A3, A4, A2}
    seq_raw::A3        # [raw_seq_dim, L, B] = [46, L, B]
    cond_raw::A3       # [raw_cond_dim, L, B] = [512, L, B]
    pair_raw::A4       # [raw_pair_dim, L, L, B] = [217, L, L, B]
    pair_cond_raw::A4  # [raw_pair_cond_dim, L, L, B] = [512, L, L, B]
    mask::A2           # [L, B]
end
```

The training loop extracts raw features once on CPU (`extract_raw_features()`), transfers to GPU, then calls `forward_from_raw_features()` inside the gradient context. Only the projection Dense layers and transformer are differentiated.

## Key Files

| File | Contents |
|------|----------|
| `src/features/feature_factory.jl` | All Feature types, FeatureFactory struct, pre-configured factory functions |
| `src/features/time_embedding.jl` | `get_time_embedding()`, `get_index_embedding()`, sinusoidal encoding |
| `src/features/pair_features.jl` | `bin_pairwise_distances()`, `relative_sequence_separation()`, `bin_values()` |
| `src/features/geometry.jl` | `backbone_torsion_angles()`, `sidechain_torsion_angles()`, `signed_dihedral_angle()`, `bond_angle()` |
| `src/nn/score_network.jl` | `ScoreNetworkRawFeatures`, `extract_raw_features()`, `compute_sc_feature_offsets()`, `update_sc_raw_features!()` |
