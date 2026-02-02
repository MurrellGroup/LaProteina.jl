# Adaptive Layer Normalization modules
# Port of adaptive_ln_scale.py from la-proteina

using Flux
using NNlib: sigmoid

"""
    ProteINAAdaLN(dim::Int, dim_cond::Int)

Adaptive Layer Normalization with sigmoid gating on scale.
Different from standard AdaLN: uses sigmoid(gamma) instead of (1 + gamma).

Formula: output = LayerNorm(x) * sigmoid(gamma(cond)) + beta(cond)

# Arguments
- `dim`: Dimension of the input representation
- `dim_cond`: Dimension of the conditioning variables

# Example
```julia
adaln = ProteINAAdaLN(768, 256)
x = randn(Float32, 768, 128, 2)      # [D, L, B]
cond = randn(Float32, 256, 128, 2)   # [D_cond, L, B]
mask = ones(Float32, 128, 2)         # [L, B]
y = adaln(x, cond, mask)
```
"""
struct ProteINAAdaLN
    norm::PyTorchLayerNorm
    norm_cond::PyTorchLayerNorm
    to_gamma::Flux.Chain
    to_beta::Dense
end

Flux.@layer ProteINAAdaLN

function ProteINAAdaLN(dim::Int, dim_cond::Int)
    norm = PyTorchLayerNorm(dim; affine=false)
    norm_cond = PyTorchLayerNorm(dim_cond)
    to_gamma = Flux.Chain(Dense(dim_cond => dim), x -> sigmoid.(x))
    to_beta = Dense(dim_cond => dim; bias=false)
    return ProteINAAdaLN(norm, norm_cond, to_gamma, to_beta)
end

function (m::ProteINAAdaLN)(x, cond, mask)
    # x: [D, L, B], cond: [D_cond, L, B], mask: [L, B]
    normed = m.norm(x)
    normed_cond = m.norm_cond(cond)

    gamma = m.to_gamma(normed_cond)  # [D, L, B]
    beta = m.to_beta(normed_cond)    # [D, L, B]

    out = normed .* gamma .+ beta

    # Apply mask
    mask_expanded = reshape(mask, 1, size(mask)...)  # [1, L, B]
    return out .* mask_expanded
end

"""
    AdaptiveOutputScale(dim::Int, dim_cond::Int; bias_init::Float32=-2.0f0)

Adaptive output scaling with zero-initialized weights and sigmoid activation.
Used to scale outputs before residual connections.

Formula: output = x * sigmoid(linear(cond))

The linear layer is initialized with zero weights and bias = bias_init (default -2.0).
This means initial scale ≈ sigmoid(-2) ≈ 0.12, allowing gradual learning.

# Arguments
- `dim`: Dimension of the input
- `dim_cond`: Dimension of the conditioning variables
- `bias_init`: Initial value for the bias (default -2.0)

# Example
```julia
scale = AdaptiveOutputScale(768, 256)
x = randn(Float32, 768, 128, 2)
cond = randn(Float32, 256, 128, 2)
mask = ones(Float32, 128, 2)
y = scale(x, cond, mask)
```
"""
struct AdaptiveOutputScale
    linear::Dense
end

Flux.@layer AdaptiveOutputScale

function AdaptiveOutputScale(dim::Int, dim_cond::Int; bias_init::Float32=-2.0f0)
    linear = Dense(dim_cond => dim)
    # Zero-initialize weights
    fill!(linear.weight, 0)
    # Initialize bias to bias_init
    fill!(linear.bias, bias_init)
    return AdaptiveOutputScale(linear)
end

function (m::AdaptiveOutputScale)(x, cond, mask)
    # x: [D, L, B], cond: [D_cond, L, B], mask: [L, B]
    gamma = sigmoid.(m.linear(cond))  # [D, L, B]
    out = x .* gamma

    # Apply mask
    mask_expanded = reshape(mask, 1, size(mask)...)  # [1, L, B]
    return out .* mask_expanded
end

"""
    AdaptiveLayerNormIdentical(dim::Int, dim_cond::Int; mode::Symbol=:single, use_ln_cond::Bool=false)

Adaptive Layer Norm where conditioning is identical across sequence positions.
More efficient when cond doesn't vary along the sequence dimension.

# Arguments
- `dim`: Dimension of the input
- `dim_cond`: Dimension of the conditioning (batch-level)
- `mode`: :single for sequence [D, L, B], :pair for pair [D, L, L, B]
- `use_ln_cond`: Whether to layer norm the conditioning

# Example
```julia
adaln = AdaptiveLayerNormIdentical(768, 256; mode=:single)
x = randn(Float32, 768, 128, 2)  # [D, L, B]
cond = randn(Float32, 256, 2)    # [D_cond, B] - no L dimension!
mask = ones(Float32, 128, 2)     # [L, B]
y = adaln(x, cond, mask)
```
"""
struct AdaptiveLayerNormIdentical
    norm::PyTorchLayerNorm
    norm_cond::Union{PyTorchLayerNorm, Nothing}
    to_gamma::Flux.Chain
    to_beta::Dense
    mode::Symbol
end

Flux.@layer AdaptiveLayerNormIdentical

function AdaptiveLayerNormIdentical(dim::Int, dim_cond::Int; mode::Symbol=:single, use_ln_cond::Bool=false)
    @assert mode in (:single, :pair) "Mode must be :single or :pair"
    norm = PyTorchLayerNorm(dim; affine=false)
    norm_cond = use_ln_cond ? PyTorchLayerNorm(dim_cond) : nothing
    to_gamma = Flux.Chain(Dense(dim_cond => dim), x -> sigmoid.(x))
    to_beta = Dense(dim_cond => dim; bias=false)
    return AdaptiveLayerNormIdentical(norm, norm_cond, to_gamma, to_beta, mode)
end

function (m::AdaptiveLayerNormIdentical)(x, cond, mask)
    # x: [D, L, B] (single) or [D, L, L, B] (pair)
    # cond: [D_cond, B] (batch-level) or [D_cond, L, L, B] (per-pair, will be reduced)
    # mask: [L, B] (single) or [L, L, B] (pair)

    normed = m.norm(x)

    # Handle per-pair conditioning: reduce to batch-level by taking position (1,1)
    if ndims(cond) == 4
        # cond is [D_cond, L, L, B] - take first position (all should be same due to broadcasting)
        cond_batch = cond[:, 1, 1, :]  # [D_cond, B]
    else
        cond_batch = cond  # Already [D_cond, B]
    end

    normed_cond = isnothing(m.norm_cond) ? cond_batch : m.norm_cond(cond_batch)

    gamma = m.to_gamma(normed_cond)  # [D, B]
    beta = m.to_beta(normed_cond)    # [D, B]

    # Broadcast to match x dimensions
    if m.mode == :single
        # x: [D, L, B], gamma/beta: [D, B] -> [D, 1, B]
        gamma_brc = reshape(gamma, size(gamma, 1), 1, size(gamma, 2))
        beta_brc = reshape(beta, size(beta, 1), 1, size(beta, 2))
    else  # :pair
        # x: [D, L, L, B], gamma/beta: [D, B] -> [D, 1, 1, B]
        gamma_brc = reshape(gamma, size(gamma, 1), 1, 1, size(gamma, 2))
        beta_brc = reshape(beta, size(beta, 1), 1, 1, size(beta, 2))
    end

    out = normed .* gamma_brc .+ beta_brc

    # Apply mask
    mask_expanded = reshape(mask, 1, size(mask)...)
    return out .* mask_expanded
end
