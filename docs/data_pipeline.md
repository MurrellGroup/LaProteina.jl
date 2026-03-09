# Data Pipeline

End-to-end data flow from raw protein structures to GPU training batches.

## Overview

```
AlphaFold DB mmCIF files
    → load_pdb() → Dict(:coords, :aatype, :atom_mask, ...)
    → precompute_single_protein() → PrecomputedProteinNT(ca_coords, z_mean, z_log_scale, mask)
    → save to sharded JLD2 files
    → load_precomputed_shard() → Vector{PrecomputedProteinNT}
    → protein_to_X1_simple() → (ContinuousState, ContinuousState)
    → branching_bridge() → bridged batch at time t
    → extract_raw_features() → ScoreNetworkRawFeatures
    → pad_batch_to(TILE_SIZE) → padded features
    → dev() → GPU CuArrays
    → forward_from_raw_features() → model output
```

## Stage 1: Raw Data

### Source

AlphaFold Protein Structure Database mmCIF files:
```
~/shared_data/afdb_laproteina/raw/
├── AF-A0A000-F1-model_v4.cif
├── AF-A0A001-F1-model_v4.cif
└── ... (~289k files)
```

### PDB/mmCIF Loading (`src/data/pdb_loading.jl`)

```julia
data = load_pdb("path/to/file.cif"; chain_id="A")
```

Auto-detects format from extension (`.cif`/`.mmcif` → MMCIFFormat, `.pdb` → PDBFormat). Uses BioStructures.jl for parsing.

**Output Dict:**
| Key | Shape | Description |
|-----|-------|-------------|
| `:coords` | [3, 37, L] | Atom37 coordinates in nanometers (Angstrom / 10) |
| `:aatype` | [L] | Amino acid indices (1-20, 21=unknown) |
| `:atom_mask` | [37, L] | Boolean mask of present atoms |
| `:residue_mask` | [L] | All true |
| `:sequence` | String | Amino acid sequence |

**CA extraction:**
```julia
ca_coords = extract_ca_coords(data)  # [3, L] — CA is atom index 2 in atom37
```

**Coordinate convention:** All coordinates in nanometers (nm), not Angstroms. The PDB loader divides by 10 during loading.

### Batching

```julia
batched = batch_pdb_data([data1, data2, ...]; pad_length=256)
```

Pads all tensors to uniform length, adding zero-masks for padding positions.

## Stage 2: Encoder Precomputation

### Single Protein Processing (`src/training/precompute_encoder.jl`)

```julia
protein = precompute_single_protein(encoder_cpu, encoder_gpu, data)
```

1. Loads protein data Dict
2. Extracts CA coordinates: `ca_coords = data[:coords][:, 2, :]`  → [3, L]
3. Centers: `ca_coords .-= mean(ca_coords, dims=2)`
4. Prepares encoder input batch (all-atom coords, residue types, masks)
5. Runs frozen encoder on GPU to get `z_mean` and `z_log_scale`
6. Returns `PrecomputedProteinNT`

### Precomputed Format

```julia
const PrecomputedProteinNT = NamedTuple{
    (:ca_coords, :z_mean, :z_log_scale, :mask),
    Tuple{Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}}
}
```

- `ca_coords`: [3, L] — Zero-centered CA positions (nm)
- `z_mean`: [8, L] — Encoder posterior mean
- `z_log_scale`: [8, L] — Encoder posterior log standard deviation
- `mask`: [L] — All 1.0 (valid residues only after length filtering)

### Sharding (`scripts/precompute_all_training_data.jl`)

Proteins are distributed across 10 shards for efficient loading:

```julia
precompute_dataset_sharded(encoder_cpu, encoder_gpu, data_list;
    output_dir="precomputed_shards", n_shards=10,
    min_length=30, max_length=256)
```

Output: `train_shard_01.jld2` through `train_shard_10.jld2`

Each shard stores a `Vector{PrecomputedProteinNT}` under the key `"proteins"`.

### Loading

```julia
proteins = load_precomputed_shard("train_shard_01.jld2")
# Returns Vector{PrecomputedProteinNT}
```

All shards are loaded into memory at training start.

## Stage 3: Training Batch Construction

### BranchingState Conversion (`src/branching/branching_states.jl`)

Each precomputed protein is converted to a BranchingState for the CoalescentFlow:

```julia
function protein_to_branching_state(protein; sample_z=true)
    ca = ContinuousState(reshape(protein.ca_coords, 3, 1, L))  # [3, 1, L]
    if sample_z
        z = protein.z_mean .+ randn(Float32, size(protein.z_mean)) .* exp.(protein.z_log_scale)
    else
        z = protein.z_mean
    end
    ll = ContinuousState(reshape(z, 8, 1, L))  # [8, 1, L]
    idx = DiscreteState(0, collect(1:L))         # Index tracking
    mask = Bool.(protein.mask)

    return BranchingState((
        MaskedState(ca, mask, mask),
        MaskedState(ll, mask, mask),
        MaskedState(idx, mask, mask)
    ), ones(Int, L); flowmask=mask, branchmask=mask)
end
```

The X0 sampler creates single-element noise states:
```julia
function X0_sampler_laproteina(latent_dim=8)
    return root -> (
        ContinuousState(randn(Float32, 3, 1, 1)),
        ContinuousState(randn(Float32, latent_dim, 1, 1)),
        DiscreteState(0, [1])
    )
end
```

### Bridging

`branching_bridge()` from BranchingFlows.jl:
1. Takes target proteins (X1) as BranchingStates
2. Samples coalescent forest structures connecting noise (X0) to targets (X1)
3. Samples random time t and computes bridged state Xt
4. Returns Xt, X1 targets, split targets, deletion targets, masks

```julia
bat = branching_bridge(P, X0_sampler, X1s, t_dist;
    coalescence_factor=1.0, use_branching_time_prob=0.5,
    merger=BranchingFlows.canonical_anchor_merge,
    length_mins=Poisson(100), deletion_pad=1.1)
```

### Feature Extraction

Raw features are extracted outside the gradient context:

```julia
raw = extract_raw_features(model, batch)
# → ScoreNetworkRawFeatures(seq_raw, cond_raw, pair_raw, pair_cond_raw, mask)
```

This computes all feature values (time embeddings, distance bins, sequence separations, etc.) but does NOT apply the projection Dense layers.

### Tile-Size Padding

All sequences are padded to the next multiple of TILE_SIZE=64:

```julia
function pad_batch_to(batch, tile_size)
    L = current_length(batch)
    padded_L = cld(L, tile_size) * tile_size
    # Zero-pad all tensors, extend mask with zeros
end
```

This ensures optimal GPU kernel utilization (flash attention tile dimensions must align).

## Stage 4: GPU Transfer and Forward Pass

### Transfer to GPU

```julia
bd = dev(bd_cpu)  # dev = Flux.gpu
```

All tensors in the batch NamedTuple are transferred to GPU via `Flux.gpu()`.

### Self-Conditioning Update (In-Place)

After the initial forward pass (no gradients), SC features are updated directly on GPU:

```julia
update_sc_raw_features!(raw_features, sc_offsets, x_sc_bb_ca, x_sc_local_latents)
```

This overwrites specific channels in `raw_features.seq_raw` and `raw_features.pair_raw` without re-extracting all features.

### Training Forward Pass (Inside Gradient)

```julia
loss, grads = Flux.withgradient(model) do m
    out = forward_from_raw_features(m, raw_features)
    # ... compute losses ...
end
```

Only the projection Dense layers and transformer parameters are differentiated. The raw feature extraction is not part of the computational graph.

## Key Files

| File | Purpose |
|------|---------|
| `src/data/pdb_loading.jl` | `load_pdb()`, `extract_ca_coords()`, `batch_pdb_data()`, `save_pdb()` |
| `src/training/precompute_encoder.jl` | `PrecomputedProteinNT`, `precompute_single_protein()`, sharding, loading |
| `src/branching/branching_states.jl` | `protein_to_branching_state()`, `X0_sampler_laproteina()` |
| `src/nn/score_network.jl` | `ScoreNetworkRawFeatures`, `extract_raw_features()`, `forward_from_raw_features()` |
| `src/features/feature_factory.jl` | Feature extraction logic |
| `scripts/precompute_all_training_data.jl` | Precomputation script |
