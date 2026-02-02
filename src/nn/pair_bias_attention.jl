# Pair-biased attention with gating
# Port of pair_bias_attn.py from la-proteina

using Flux
using NNlib: softmax, batched_mul, sigmoid

"""
    PairBiasAttention(node_dim::Int, n_heads::Int;
        dim_head::Int=node_dim÷n_heads,
        pair_dim::Union{Int,Nothing}=nothing,
        qk_ln::Bool=false,
        bias::Bool=true)

AF3-style pair-biased attention with sigmoid gating.

# Architecture
1. LayerNorm on node features
2. Q, K, V projections
3. Optional Q/K layer norms
4. Attention with optional pair bias
5. Sigmoid gating: output = sigmoid(g) * attention_output
6. Output projection

# Arguments
- `node_dim`: Input/output dimension for node features
- `n_heads`: Number of attention heads
- `dim_head`: Dimension per head (default: node_dim ÷ n_heads)
- `pair_dim`: Dimension of pair features (nothing if no pair bias)
- `qk_ln`: Whether to apply layer norm to Q and K
- `bias`: Whether to use bias in linear projections

# Example
```julia
attn = PairBiasAttention(768, 12; pair_dim=64)
x = randn(Float32, 768, 128, 2)      # [D, L, B]
pair = randn(Float32, 64, 128, 128, 2)  # [D_pair, L, L, B]
mask = ones(Float32, 128, 2)         # [L, B]
y = attn(x, pair, mask)
```
"""
struct PairBiasAttention
    node_norm::PyTorchLayerNorm
    to_qkv::Dense
    to_g::Dense
    to_out::Dense
    q_norm::Union{PyTorchLayerNorm, typeof(identity)}
    k_norm::Union{PyTorchLayerNorm, typeof(identity)}
    pair_norm::Union{PyTorchLayerNorm, Nothing}
    to_bias::Union{Dense, Nothing}
    n_heads::Int
    dim_head::Int
    scale::Float32
end

Flux.@layer PairBiasAttention

function PairBiasAttention(node_dim::Int, n_heads::Int;
        dim_head::Int=node_dim ÷ n_heads,
        pair_dim::Union{Int,Nothing}=nothing,
        qk_ln::Bool=false,
        bias::Bool=true)

    inner_dim = dim_head * n_heads

    node_norm = PyTorchLayerNorm(node_dim)
    to_qkv = Dense(node_dim => inner_dim * 3; bias=bias)
    to_g = Dense(node_dim => inner_dim; bias=bias)
    to_out = Dense(inner_dim => node_dim; bias=bias)

    q_norm = qk_ln ? PyTorchLayerNorm(inner_dim) : identity
    k_norm = qk_ln ? PyTorchLayerNorm(inner_dim) : identity

    if !isnothing(pair_dim)
        pair_norm = PyTorchLayerNorm(pair_dim)
        to_bias = Dense(pair_dim => n_heads; bias=false)
    else
        pair_norm = nothing
        to_bias = nothing
    end

    scale = Float32(dim_head ^ -0.5)

    return PairBiasAttention(
        node_norm, to_qkv, to_g, to_out,
        q_norm, k_norm, pair_norm, to_bias,
        n_heads, dim_head, scale
    )
end

function (m::PairBiasAttention)(node_feats, pair_feats, mask)
    # node_feats: [D, L, B]
    # pair_feats: [D_pair, L, L, B] or nothing
    # mask: [L, B]

    D, L, B = size(node_feats)
    h = m.n_heads
    d = m.dim_head

    # Normalize inputs
    node_feats = m.node_norm(node_feats)

    # QKV projection
    qkv = m.to_qkv(node_feats)  # [3*inner_dim, L, B]
    inner_dim = h * d
    q = qkv[1:inner_dim, :, :]
    k = qkv[inner_dim+1:2*inner_dim, :, :]
    v = qkv[2*inner_dim+1:end, :, :]

    # Q/K layer norms
    q = m.q_norm(q)
    k = m.k_norm(k)

    # Gate projection (before attention)
    g = m.to_g(node_feats)  # [inner_dim, L, B]

    # Reshape for multi-head attention: [inner_dim, L, B] -> [d, L, h, B]
    q = reshape(q, d, h, L, B)
    k = reshape(k, d, h, L, B)
    v = reshape(v, d, h, L, B)
    g = reshape(g, d, h, L, B)

    # Permute to [d, L, h, B] for attention
    q = permutedims(q, (1, 3, 2, 4))  # [d, L, h, B]
    k = permutedims(k, (1, 3, 2, 4))  # [d, L, h, B]
    v = permutedims(v, (1, 3, 2, 4))  # [d, L, h, B]
    g = permutedims(g, (1, 3, 2, 4))  # [d, L, h, B]

    # Compute pair bias
    if !isnothing(pair_feats) && !isnothing(m.pair_norm)
        pair_feats = m.pair_norm(pair_feats)  # [D_pair, L, L, B]
        bias = m.to_bias(pair_feats)  # [h, L, L, B]
    else
        bias = nothing
    end

    # Compute attention
    attn_out = _attention(q, k, v, bias, mask, m.scale)  # [d, L, h, B]

    # Apply sigmoid gating
    attn_out = sigmoid.(g) .* attn_out

    # Reshape back: [d, L, h, B] -> [d, h, L, B] -> [inner_dim, L, B]
    attn_out = permutedims(attn_out, (1, 3, 2, 4))  # [d, h, L, B]
    attn_out = reshape(attn_out, inner_dim, L, B)

    # Output projection
    out = m.to_out(attn_out)

    return out
end

"""
Internal attention computation.
"""
function _attention(q, k, v, pair_bias, mask, scale)
    # q, k, v: [d, L, h, B]
    # pair_bias: [h, L_q, L_k, B] or nothing
    # mask: [L, B]

    d, L, h, B = size(q)

    # Compute attention scores: q^T k / sqrt(d)
    # Reshape for batched matmul: [d, L, h*B]
    q_flat = reshape(q, d, L, h * B)
    k_flat = reshape(k, d, L, h * B)

    # scores = q^T @ k: [L_q, L_k, h*B]
    # Using batched matmul with transposed q
    scores = batched_mul(permutedims(q_flat, (2, 1, 3)), k_flat)  # [L, L, h*B]
    scores = scores .* scale

    # Reshape back to [L, L, h, B]
    scores = reshape(scores, L, L, h, B)

    # Add pair bias if present: bias is [h, L_q, L_k, B]
    if !isnothing(pair_bias)
        # Permute bias to [L_q, L_k, h, B]
        bias_perm = permutedims(pair_bias, (2, 3, 1, 4))
        scores = scores .+ bias_perm
    end

    # Apply mask
    if !isnothing(mask)
        # Create attention mask: [L_k, B] -> [1, L_k, 1, B]
        # We mask keys, so expand for query and head dimensions
        attn_mask = reshape(mask, 1, L, 1, B)
        # Set masked positions to large negative value
        scores = scores .+ (1.0f0 .- attn_mask) .* (-1.0f10)
    end

    # Softmax over key dimension (dim 2)
    attn_weights = softmax(scores; dims=2)  # [L_q, L_k, h, B]

    # Reshape for batched matmul
    attn_flat = reshape(attn_weights, L, L, h * B)  # [L_q, L_k, h*B]
    v_flat = reshape(v, d, L, h * B)  # [d, L_k, h*B]

    # output = attn @ v: [L_q, d, h*B]
    out = batched_mul(attn_flat, permutedims(v_flat, (2, 1, 3)))  # [L_q, d, h*B]
    out = permutedims(out, (2, 1, 3))  # [d, L_q, h*B]

    # Reshape back
    out = reshape(out, d, L, h, B)  # [d, L, h, B]

    return out
end

"""
    MultiHeadBiasedAttentionADALN(dim_token::Int, dim_pair::Int, n_heads::Int, dim_cond::Int;
        qk_ln::Bool=false)

Pair-biased attention with adaptive layer norm and output scaling.
Wraps PairBiasAttention with ProteINAAdaLN input normalization and AdaptiveOutputScale.

# Arguments
- `dim_token`: Token dimension
- `dim_pair`: Pair representation dimension
- `n_heads`: Number of attention heads
- `dim_cond`: Conditioning dimension
- `qk_ln`: Whether to use Q/K layer norms
"""
struct MultiHeadBiasedAttentionADALN
    adaln::ProteINAAdaLN
    mha::PairBiasAttention
    scale_output::AdaptiveOutputScale
end

Flux.@layer MultiHeadBiasedAttentionADALN

function MultiHeadBiasedAttentionADALN(dim_token::Int, dim_pair::Int, n_heads::Int, dim_cond::Int;
        qk_ln::Bool=false)
    adaln = ProteINAAdaLN(dim_token, dim_cond)
    mha = PairBiasAttention(dim_token, n_heads; pair_dim=dim_pair, qk_ln=qk_ln)
    scale_output = AdaptiveOutputScale(dim_token, dim_cond)
    return MultiHeadBiasedAttentionADALN(adaln, mha, scale_output)
end

function (m::MultiHeadBiasedAttentionADALN)(x, pair_rep, cond, mask)
    # x: [D, L, B]
    # pair_rep: [D_pair, L, L, B]
    # cond: [D_cond, L, B]
    # mask: [L, B]

    # AdaLN input normalization
    x_normed = m.adaln(x, cond, mask)

    # Attention (pass mask for key masking)
    attn_out = m.mha(x_normed, pair_rep, mask)

    # Adaptive output scaling
    out = m.scale_output(attn_out, cond, mask)

    # Apply mask
    mask_expanded = reshape(mask, 1, size(mask)...)
    return out .* mask_expanded
end
