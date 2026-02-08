# GPU Optimizations

Comprehensive guide to the GPU acceleration stack in LaProteina.

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

## cuTile Flash Attention

Custom CUDA kernels for multi-head attention, adapted from OnionTile.jl. Layout: `(D, SeqLen, Heads, Batch)` with D_k=D_v=64, h=12.

### Forward Kernels (`src/gpu/kernels/fmha.jl`)

Four kernel variants:

| Kernel | Bias | LSE | Use Case |
|--------|------|-----|----------|
| `_fmha_kernel` | No | No | Inference without pair bias |
| `_fmha_bias_kernel` | Yes | No | Inference with pair bias |
| `_fmha_lse_kernel` | No | Yes | Training without pair bias |
| `_fmha_bias_lse_kernel` | Yes | Yes | Training with pair bias |

All kernels use:
- Online softmax accumulation with log2 scaling
- TF32 tensor cores for Q*K^T and P*V matmuls (via `_to_tf32()`)
- Out-of-bounds masking via `-Inf` penalty on the final tile
- Configurable tile sizes (default: TILE_M=64, TILE_N=64)

Training variants save the logsumexp (LSE) in `(1, SeqLen, H, B)` format for the backward pass.

Pair bias enters as an additive term in log2 space: `qk = qk * qk_scale_log2 + bias * INV_LOG_2`. Bias batch dimension supports broadcasting (bias_batch < actual batch).

### Backward Kernels (`src/gpu/kernels/fmha_bwd.jl`)

Three-step backward pass:
1. `_fmha_bwd_preprocess_kernel`: Compute Delta = rowsum(dO * O)
2. `_fmha_bwd_dkdv_kernel`: Compute dK, dV (and dBias if applicable)
3. `_fmha_bwd_dq_kernel`: Compute dQ

Public API:
```julia
flash_attention_backward(dO, Out, Lse, Q, K, V; scale)        # Without bias
flash_attention_bias_backward(dO, Out, Lse, Q, K, V, bias; scale)  # With bias → also returns dBias
```

### Tile Size Selection (`src/gpu/kernels/utils.jl`)

```julia
function _select_fmha_tiles(D_k, seq_len, heads, batch)
    if D_k <= 32 && batch * heads >= 64
        return (32, 64)  # More blocks for better SM utilization
    else
        return (64, 64)  # Default for D_k=64
    end
end
```

## Fused LayerNorm (`src/gpu/kernels/layernorm.jl`)

Single-kernel LayerNorm with affine transform, replacing the standard multi-kernel broadcast implementation.

### Variants

| Function | In-place | Saves Stats | Use Case |
|----------|----------|-------------|----------|
| `fused_layernorm_first` | No | No | Inference |
| `fused_layernorm_first!` | Yes | No | Inference (in-place) |
| `fused_layernorm_first_train` | No | Yes (mu, inv_std) | Training forward |

Backward: `fused_layernorm_backward(dy, x, w, mu_2d, inv_std_2d)` → returns `(dx, dw, db)`.

### Dispatch Thresholds (`src/gpu/dispatch.jl`)

The fused kernel is only competitive for specific tensor shapes:
```julia
function layernorm_forward(x, w, b, eps)
    C = size(x, 1)
    N = div(length(x), C)
    if ispow2(C) && C >= 512 && N <= 2048
        return fused_layernorm(x, w, b, eps)
    end
    # Fallback: broadcast implementation
    return pytorch_normalise(x; dims=1, eps=eps) .* w .+ b
end
```

### Tile Size Selection

| C (channel dim) | Tile N |
|-----------------|--------|
| C <= 256 | 64 |
| C <= 1024 | 16 |
| C > 1024 | 8 |

## ChainRulesCore Integration (`src/gpu/rrules.jl`)

Custom AD backward rules connect Zygote to the cuTile kernels:

```julia
# LayerNorm: fused forward saves mu/inv_std, fused backward uses them
function CRC.rrule(::typeof(layernorm_forward), x, w, b; eps)
    y, mu_2d, inv_std_2d = fused_layernorm_first_train(x, w, b, eps)
    function pullback(dy)
        dx, dw, db = fused_layernorm_backward(dy, x, w, mu_2d, inv_std_2d)
        CUDA.unsafe_free!(mu_2d)  # Eagerly free training intermediates
        CUDA.unsafe_free!(inv_std_2d)
        return NoTangent(), dx, dw, db
    end
    return y, pullback
end

# Flash attention: forward saves Out+Lse, backward uses them
function CRC.rrule(::typeof(flash_attention_bias_forward), Q, K, V, bias; scale)
    Out, Lse = flash_attention_bias_train(Q, K, V, bias; scale=scale)
    function pullback(dy)
        dQ, dK, dV, dBias = flash_attention_bias_backward(dy, Out, Lse, Q, K, V, bias; scale)
        CUDA.unsafe_free!(Lse)
        return NoTangent(), dQ, dK, dV, dBias
    end
    return Out, pullback
end
```

Non-pow2 channel dimensions fall back to standard Zygote differentiation via `CRC.rrule_via_ad`.

## Pre-Normalized Pair Features (`src/gpu/layers.jl`)

When `update_pair_repr=false` (the default), all 14 transformer layers share the same pair representation. Instead of normalizing it 14 times, we normalize once before the transformer stack:

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

## TF32 Math Mode (`src/gpu/utils.jl`)

```julia
function enable_tf32!()
    CUDA.math_mode!(CUDA.FAST_MATH)
end
```

Enables TF32 tensor cores on Ampere+ GPUs for ~2x speedup on matmuls. The cuTile kernels also use TF32 internally via `_to_tf32(tile, Float32)`.

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

## Buffer Pooling (`src/gpu/utils.jl`)

Pre-allocated GPU buffers for permutation operations, keyed by (slot, shape):

```julia
const _perm_buf_pool = Dict{Tuple{Int, Tuple}, CuArray}()

function _get_perm_buf(slot::Int, shape::Tuple)
    key = (slot, shape)
    if !haskey(_perm_buf_pool, key)
        _perm_buf_pool[key] = CUDA.zeros(Float32, shape...)
    end
    return _perm_buf_pool[key]
end
```

Used in `PairBiasAttention` inference path to avoid repeated allocation of Q/K/V permutation buffers (slots 20-23) and pair bias buffers (slot 30+).

## `within_gradient` Context Detection (`src/gpu/utils.jl`)

Distinguishes training from inference to select optimal code paths:

```julia
within_gradient(x) = false

function CRC.rrule(::typeof(within_gradient), x)
    return true, _ -> (NoTangent(), NoTangent())
end
```

- **Training** (inside `Zygote.gradient`): Returns `true` → uses allocating, AD-compatible code paths
- **Inference**: Returns `false` → uses in-place operations, buffer pooling, eager `CUDA.unsafe_free!`

## Tile-Size Padding

Batch sequences must be padded to multiples of 64 for optimal kernel utilization:

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
| `src/gpu/gpu.jl` | Mode selection, conditional includes |
| `src/gpu/dispatch.jl` | Dispatch thresholds (when to use fused kernels) |
| `src/gpu/rrules.jl` | ChainRulesCore backward rules |
| `src/gpu/layers.jl` | CuArray PairBiasAttention, pre-normed pairs, transformer overrides |
| `src/gpu/layers_nocutile.jl` | NNlib fallback overrides |
| `src/gpu/checkpointing.jl` | `safe_checkpointed` |
| `src/gpu/utils.jl` | TF32, softmax fix, buffer pooling, `within_gradient` |
| `src/gpu/kernels/fmha.jl` | Flash attention forward kernels |
| `src/gpu/kernels/fmha_bwd.jl` | Flash attention backward kernels |
| `src/gpu/kernels/layernorm.jl` | Fused LayerNorm kernels |
| `src/nn/score_network.jl` | `ScoreNetworkRawFeatures`, `compute_sc_feature_offsets`, `update_sc_raw_features!` |
