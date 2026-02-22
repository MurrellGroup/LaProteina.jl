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
    PairUpdate(token_dim::Int, pair_dim::Int; use_tri_mult::Bool=false, tri_mult_c::Int=196)

Update pair representation based on sequence representation.
Port of PairReprUpdate from la-proteina.

Base path: LayerNorm(seq) → Linear(token_dim → 2*pair_dim, split) → outer sum → add to pair.
With `use_tri_mult=true`: additionally applies TriangleMultiplication (outgoing + incoming) and PairTransition.

# Arguments
- `token_dim`: Token/sequence dimension
- `pair_dim`: Pair representation dimension
- `use_tri_mult`: Whether to use triangle multiplicative updates
- `tri_mult_c`: Hidden dimension for triangle multiplication (clamped to min(pair_dim, 196))
"""
struct PairUpdate
    seq_ln::PyTorchLayerNorm             # LayerNorm on sequence input
    outer_proj::Dense                     # Projects seq to 2*pair_dim (split for outer sum)
    tri_out::Union{TriangleMultiplication, Nothing}  # Outgoing triangle mult
    tri_in::Union{TriangleMultiplication, Nothing}   # Incoming triangle mult
    pair_transition::PairTransition       # Pair feedforward (always present)
end

Flux.@layer PairUpdate

function PairUpdate(token_dim::Int, pair_dim::Int; use_tri_mult::Bool=false, tri_mult_c::Int=196)
    seq_ln = PyTorchLayerNorm(token_dim)
    outer_proj = Dense(token_dim => 2 * pair_dim; bias=false)

    if use_tri_mult
        c_hidden = min(pair_dim, tri_mult_c)
        tri_out = TriangleMultiplication(pair_dim; c_hidden=c_hidden, mode=:outgoing)
        tri_in = TriangleMultiplication(pair_dim; c_hidden=c_hidden, mode=:incoming)
    else
        tri_out = nothing
        tri_in = nothing
    end

    # PairTransition is always created (matches Python PairReprUpdate)
    pair_transition = PairTransition(pair_dim; n=2)

    return PairUpdate(seq_ln, outer_proj, tri_out, tri_in, pair_transition)
end

function (m::PairUpdate)(seq_rep, pair_rep, mask)
    # seq_rep: [D_token, L, B]
    # pair_rep: [D_pair, L, L, B]
    # mask: [L, B]

    D, L, B = size(seq_rep)
    D_pair = size(pair_rep, 1)

    # Compute pair mask
    pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)  # [L, L, B]
    pair_mask_exp = reshape(pair_mask, 1, L, L, B)

    # Mask sequence input (matching Python: x = x * mask[..., None])
    mask_seq_exp = reshape(mask, 1, L, B)  # [1, L, B]
    seq_masked = seq_rep .* mask_seq_exp

    # LayerNorm on masked sequence, then project to 2*pair_dim and split
    seq_normed = m.seq_ln(seq_masked)  # [D_token, L, B]
    seq_flat = reshape(seq_normed, D, L * B)
    x_proj = m.outer_proj(seq_flat)  # [2*D_pair, L*B]
    x_proj = reshape(x_proj, 2 * D_pair, L, B)

    # Split into two halves for outer sum
    # Python: x_proj_1[:, None, :, :] + x_proj_2[:, :, None, :]
    # x_proj_1 broadcasts over i (provides j component)
    # x_proj_2 broadcasts over j (provides i component)
    x_proj_1 = x_proj[1:D_pair, :, :]         # [D_pair, L, B]
    x_proj_2 = x_proj[D_pair+1:end, :, :]     # [D_pair, L, B]

    x_j = reshape(x_proj_1, D_pair, 1, L, B)  # broadcasts over i
    x_i = reshape(x_proj_2, D_pair, L, 1, B)  # broadcasts over j
    pair_rep = pair_rep .+ x_i .+ x_j

    # Apply pair mask
    pair_rep = pair_rep .* pair_mask_exp

    # Triangle multiplicative updates (residual)
    if !isnothing(m.tri_out)
        pair_rep = pair_rep .+ m.tri_out(pair_rep, pair_mask)
        pair_rep = pair_rep .* pair_mask_exp
    end

    if !isnothing(m.tri_in)
        pair_rep = pair_rep .+ m.tri_in(pair_rep, pair_mask)
        pair_rep = pair_rep .* pair_mask_exp
    end

    # Pair transition (residual, always present)
    pair_rep = pair_rep .+ m.pair_transition(pair_rep, pair_mask)
    pair_rep = pair_rep .* pair_mask_exp

    return pair_rep
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
