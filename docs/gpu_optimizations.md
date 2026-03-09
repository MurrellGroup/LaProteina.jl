# GPU Optimizations

Guide to the GPU acceleration stack in LaProteina.

## Architecture

LaProteina's GPU code routes through [Onion.jl](https://github.com/MurrellGroup/Onion.jl) dispatch hooks, which are overridden at runtime by [OnionTile.jl](https://github.com/MurrellGroup/OnionTile.jl) with cuTile kernels:

- `layernorm_first_forward(x, w, b; eps)` -> OnionTile `fused_layernorm_first` (cuTile)
- `flash_attention_forward(Q, K, V; scale)` -> OnionTile `flash_attention` (cuTile)
- `flash_attention_bias_forward(Q, K, V, bias; scale)` -> OnionTile `flash_attention_bias` (cuTile)
- `within_gradient(x)` -> ONIONop (AD-context detection)

LaProteina does not implement GPU kernels directly. OnionTile provides the cuTile flash attention, fused LayerNorm, and their backward passes (ChainRulesCore rrules). LaProteina-specific GPU code in `src/gpu/layers.jl` handles:

- Pre-normalized pair features (shared pair norm across transformer layers)
- TransformerBlock in-place residuals
- SwiGLU view-based forward
- `forward_from_raw_features_gpu` (batch-level pair conditioning)

## Operating Modes

Three modes controlled by environment variables (set BEFORE loading LaProteina):

| Mode | Env Variable | What's Enabled |
|------|-------------|----------------|
| **Default (cuTile)** | (none) | cuTile flash attention + fused LayerNorm + CuArray dispatch overrides + TF32 |
| **NocuTile** | `LAPROTEINA_NOCUTILE=1` | NNlib attention + CuArray layer overrides + TF32 (no cuTile kernels) |
| **No overrides** | `LAPROTEINA_NO_OVERRIDES=1` | Standard Flux/NNlib GPU path + TF32 + utility functions only |

```julia
# Set before `using LaProteina`
ENV["LAPROTEINA_NOCUTILE"] = "1"   # Disable cuTile kernels
# or
ENV["LAPROTEINA_NO_OVERRIDES"] = "1"  # Disable all GPU overrides
```

Mode selection is in `src/gpu/gpu.jl`:
```julia
const _NO_OVERRIDES = get(ENV, "LAPROTEINA_NO_OVERRIDES", "") != ""
const _FORCE_NOCUTILE = get(ENV, "LAPROTEINA_NOCUTILE", "") != ""
```

Training requires the default cuTile mode — the training script checks and errors if cuTile is disabled.

## Pre-Normalized Pair Features (`src/gpu/layers.jl`)

When `update_pair_repr=false` (the default for most model variants), all 14 transformer layers share the same pair representation. Instead of normalizing it 14 times, we normalize once before the transformer stack:

```julia
# In the efficient forward pass:
if !model.update_pair_repr
    pair_eps = first_pba.pair_norm.eps
    pair_normed = pytorch_normalise(pair_rep; dims=1, eps=pair_eps)
end

for i in 1:model.n_layers
    seqs = _transformer_block_prenormed(
        model.transformer_layers[i], seqs, pair_rep, pair_normed, cond, mask)
end
```

Each layer then applies only the per-head affine transform (a cheap `(n_heads, pair_dim)` multiply) instead of the full LayerNorm.

## In-Place Self-Conditioning (`src/nn/score_network.jl`)

The naive self-conditioning path re-extracts all features on CPU after each SC pass. The optimized path updates only the SC-dependent channels on GPU:

### Step 1: Compute Offsets (Once)

```julia
sc_offsets = compute_sc_feature_offsets(model)
# Returns: (seq=[(12, 14, :bb_ca), (15, 22, :local_latents)],
#           pair=[(158, 187, 0.1, 3.0, 30)])
```

This identifies exactly which channels in the raw feature tensors correspond to self-conditioning inputs.

### Step 2: Update In-Place on GPU (Each SC Pass)

```julia
function update_sc_raw_features!(raw_features, sc_offsets, x_sc_bb_ca, x_sc_local_latents)
    # Sequence features: overwrite CA and latent channels
    for (start, stop, ftype) in sc_offsets.seq
        if ftype == :bb_ca
            raw_features.seq_raw[start:stop, :, :] .= x_sc_bb_ca
        elseif ftype == :local_latents
            raw_features.seq_raw[start:stop, :, :] .= x_sc_local_latents
        end
    end
    # Pair features: recompute binned pairwise distances from SC CA coords
    for (start, stop, min_dist, max_dist, n_bins) in sc_offsets.pair
        sc_pair_dists = bin_pairwise_distances(x_sc_bb_ca, min_dist, max_dist, n_bins)
        raw_features.pair_raw[start:stop, :, :, :] .= sc_pair_dists
    end
end
```

### Performance Impact

| Method | Per-batch (BS=8) | Per-sample |
|--------|-----------------|------------|
| Old: CPU round-trip | 3139 ms | 392 ms |
| New: In-place GPU | 1495 ms | 187 ms |
| **Speedup** | **2.1x** | **2.1x** |

## TF32 Math Mode

```julia
function enable_tf32!()
    CUDA.math_mode!(CUDA.FAST_MATH)
end
```

Enables TF32 tensor cores on Ampere+ GPUs for ~2x speedup on matmuls. The cuTile kernels also use TF32 internally.

Additionally, the cuDNN softmax algorithm is overridden to always use ACCURATE mode (not FAST) to prevent NaN with large attention scores.

## Gradient Checkpointing (`src/gpu/checkpointing.jl`)

Standard `Zygote.checkpointed` doesn't work correctly with in-place operations because the recomputation forward pass runs outside Zygote's AD tracing, causing `within_gradient()` to return `false` and triggering the inference (in-place) code path.

`safe_checkpointed` solves this by copying inputs before the forward pass:

```julia
function CRC.rrule(::typeof(safe_checkpointed), f, args...; kwargs...)
    saved_args = map(copy, args)  # Copy inputs before forward
    y = f(args...; kwargs...)
    function pullback(dy)
        _, back = Zygote.pullback(f, saved_args...; kwargs...)
        return (NoTangent(), NoTangent(), back(dy)...)
    end
    return y, pullback
end
```

## `within_gradient` Context Detection

Imported from ONIONop, `within_gradient(x)` distinguishes training from inference to select optimal code paths:

- **Training** (inside `Zygote.gradient`): Returns `true` -> uses allocating, AD-compatible code paths
- **Inference**: Returns `false` -> uses in-place operations, eager `CUDA.unsafe_free!`

## Tile-Size Padding

Batch sequences must be padded to multiples of 64 for optimal cuTile kernel utilization:

```julia
const TILE_SIZE = 64

function pad_batch_to(batch, tile_size)
    L = size(batch.mask, 1)
    padded_L = cld(L, tile_size) * tile_size
    # Pad all tensors to padded_L with zeros
    ...
end
```

This is handled automatically in the `BatchDataset` used by `Flux.DataLoader`.

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/gpu/gpu.jl` | Mode selection, Onion hook imports |
| `src/gpu/layers.jl` | CuArray PairBiasAttention, pre-normed pairs, transformer overrides |
| `src/gpu/checkpointing.jl` | `safe_checkpointed` |
| `src/gpu/utils_nocutile.jl` | Utilities for NocuTile mode |
| `src/gpu/stubs.jl` | Stubs when GPU unavailable |
| `src/nn/score_network.jl` | `ScoreNetworkRawFeatures`, `compute_sc_feature_offsets`, `update_sc_raw_features!` |
