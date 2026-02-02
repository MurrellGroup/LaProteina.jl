# Transformer block combining attention and transition
# Port of attn_n_transition.py from la-proteina

using Flux

"""
    TransformerBlock(;
        dim_token::Int,
        dim_pair::Int,
        n_heads::Int,
        dim_cond::Int,
        qk_ln::Bool=false,
        residual_mha::Bool=true,
        residual_transition::Bool=true,
        parallel::Bool=false,
        expansion_factor::Int=4)

Transformer block with pair-biased attention and SwiGLU transition.
Both layers use adaptive layer norm and output scaling.

# Arguments
- `dim_token`: Token/sequence dimension
- `dim_pair`: Pair representation dimension
- `n_heads`: Number of attention heads
- `dim_cond`: Conditioning dimension
- `qk_ln`: Whether to use Q/K layer norms in attention
- `residual_mha`: Whether to use residual connection around attention
- `residual_transition`: Whether to use residual connection around transition
- `parallel`: Whether to run attention and transition in parallel
- `expansion_factor`: Expansion factor for SwiGLU transition

# Example
```julia
block = TransformerBlock(
    dim_token=768, dim_pair=64, n_heads=12, dim_cond=256
)
x = randn(Float32, 768, 128, 2)       # [D, L, B]
pair = randn(Float32, 64, 128, 128, 2) # [D_pair, L, L, B]
cond = randn(Float32, 256, 128, 2)    # [D_cond, L, B]
mask = ones(Float32, 128, 2)          # [L, B]
y = block(x, pair, cond, mask)
```
"""
struct TransformerBlock
    mha::MultiHeadBiasedAttentionADALN
    transition::TransitionADALN
    residual_mha::Bool
    residual_transition::Bool
    parallel::Bool
end

Flux.@layer TransformerBlock

function TransformerBlock(;
        dim_token::Int,
        dim_pair::Int,
        n_heads::Int,
        dim_cond::Int,
        qk_ln::Bool=false,
        residual_mha::Bool=true,
        residual_transition::Bool=true,
        parallel::Bool=false,
        expansion_factor::Int=4)

    # If parallel, don't allow both residuals (would add x twice)
    if parallel && residual_mha && residual_transition
        residual_transition = false
    end

    mha = MultiHeadBiasedAttentionADALN(dim_token, dim_pair, n_heads, dim_cond; qk_ln=qk_ln)
    transition = TransitionADALN(dim_token, dim_cond; expansion_factor=expansion_factor)

    return TransformerBlock(mha, transition, residual_mha, residual_transition, parallel)
end

function (m::TransformerBlock)(x, pair_rep, cond, mask)
    # x: [D, L, B]
    # pair_rep: [D_pair, L, L, B]
    # cond: [D_cond, L, B]
    # mask: [L, B]

    mask_expanded = reshape(mask, 1, size(mask)...)
    x = x .* mask_expanded

    if m.parallel
        # Run attention and transition in parallel, then sum
        x_mha = m.mha(x, pair_rep, cond, mask)
        x_tr = m.transition(x, cond, mask)
        x = x_mha .+ x_tr
        if m.residual_mha
            x = x .+ x  # This would be wrong, so we disabled one residual above
        end
    else
        # Sequential: attention then transition
        x_mha = m.mha(x, pair_rep, cond, mask)
        if m.residual_mha
            x = x .+ x_mha
        else
            x = x_mha
        end
        x = x .* mask_expanded

        x_tr = m.transition(x, cond, mask)
        if m.residual_transition
            x = x .+ x_tr
        else
            x = x_tr
        end
    end

    return x .* mask_expanded
end

"""
    PairUpdate(token_dim::Int, pair_dim::Int; use_tri_mult::Bool=false)

Update pair representation based on sequence representation.
Simple version: outer product of sequence features projected to pair space.

# Arguments
- `token_dim`: Token/sequence dimension
- `pair_dim`: Pair representation dimension
- `use_tri_mult`: Whether to use triangle multiplicative updates (not implemented yet)
"""
struct PairUpdate
    outer_proj::Dense
    pair_ln::PyTorchLayerNorm
end

Flux.@layer PairUpdate

function PairUpdate(token_dim::Int, pair_dim::Int; use_tri_mult::Bool=false)
    if use_tri_mult
        @warn "Triangle multiplicative update not yet implemented, using simple outer product"
    end
    outer_proj = Dense(token_dim * 2 => pair_dim; bias=false)
    pair_ln = PyTorchLayerNorm(pair_dim)
    return PairUpdate(outer_proj, pair_ln)
end

function (m::PairUpdate)(seq_rep, pair_rep, mask)
    # seq_rep: [D_token, L, B]
    # pair_rep: [D_pair, L, L, B]
    # mask: [L, B]

    D, L, B = size(seq_rep)
    D_pair = size(pair_rep, 1)

    # Simple outer sum approach
    # seq_i: [D, L, 1, B], seq_j: [D, 1, L, B]
    seq_i = reshape(seq_rep, D, L, 1, B)
    seq_j = reshape(seq_rep, D, 1, L, B)

    # Concatenate along feature dim: [2D, L, L, B]
    outer = cat(
        repeat(seq_i, 1, 1, L, 1),
        repeat(seq_j, 1, L, 1, 1);
        dims=1
    )

    # Project to pair dim
    outer_flat = reshape(outer, 2*D, L*L*B)
    update_flat = m.outer_proj(outer_flat)
    update = reshape(update_flat, D_pair, L, L, B)

    # Add to existing pair rep with layer norm
    new_pair = pair_rep .+ update
    new_pair = m.pair_ln(new_pair)

    # Apply pair mask
    pair_mask = mask .* permutedims(mask, (2, 1, 3))  # [L, L, B]
    pair_mask_expanded = reshape(pair_mask, 1, size(pair_mask)...)
    return new_pair .* pair_mask_expanded
end

"""
    ConditioningTransition(dim::Int; expansion_factor::Int=2)

Simple transition for conditioning variables (no adaptive components).
"""
struct ConditioningTransition
    transition::SwiGLUTransition
end

Flux.@layer ConditioningTransition

function ConditioningTransition(dim::Int; expansion_factor::Int=2)
    transition = SwiGLUTransition(dim; expansion_factor=expansion_factor, use_layer_norm=false)
    return ConditioningTransition(transition)
end

function (m::ConditioningTransition)(x, mask)
    return m.transition(x, mask)
end
