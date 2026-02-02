# Transition layers with SwiGLU activation
# Port of seq_transition_af3.py from la-proteina

using Flux
using Functors
using NNlib: swish

"""
    SwiGLU()

SwiGLU activation function.
Takes input of size 2*dim and returns output of size dim.

Formula: SwiGLU(x, g) = swish(x) * g

where x and g are the first and second halves of the input.
"""
struct SwiGLU end

# Empty struct - mark as leaf so Functors doesn't try to traverse it
Functors.@leaf SwiGLU

function (::SwiGLU)(x)
    # x: [2*dim, ...]
    # Python: x, gates = x.chunk(2, dim=-1); return silu(gates) * x
    # x1 = first half = "x" in Python, x2 = second half = "gates" in Python
    dim = size(x, 1) ÷ 2
    x1 = x[1:dim, axes(x)[2:end]...]
    x2 = x[dim+1:end, axes(x)[2:end]...]
    return swish.(x2) .* x1  # silu(gates) * x
end

"""
    SwiGLUTransition(dim::Int; expansion_factor::Int=4, use_layer_norm::Bool=false)

Transition layer using SwiGLU activation.

# Architecture
1. Optional LayerNorm
2. Linear: dim -> 2 * dim_inner (where dim_inner = dim * expansion_factor)
3. SwiGLU activation: 2 * dim_inner -> dim_inner
4. Linear: dim_inner -> dim

# Arguments
- `dim`: Input/output dimension
- `expansion_factor`: Expansion factor for inner dimension (default 4)
- `use_layer_norm`: Whether to apply LayerNorm at the start

# Example
```julia
trans = SwiGLUTransition(768; expansion_factor=4)
x = randn(Float32, 768, 128, 2)  # [D, L, B]
mask = ones(Float32, 128, 2)     # [L, B]
y = trans(x, mask)
```
"""
struct SwiGLUTransition
    ln::Union{PyTorchLayerNorm, Nothing}
    linear_in::Dense
    swiglu::SwiGLU
    linear_out::Dense
end

Flux.@layer SwiGLUTransition

function SwiGLUTransition(dim::Int; expansion_factor::Int=4, use_layer_norm::Bool=false)
    dim_inner = dim * expansion_factor
    ln = use_layer_norm ? PyTorchLayerNorm(dim) : nothing
    linear_in = Dense(dim => dim_inner * 2; bias=false)
    swiglu = SwiGLU()
    linear_out = Dense(dim_inner => dim; bias=false)
    return SwiGLUTransition(ln, linear_in, swiglu, linear_out)
end

function (m::SwiGLUTransition)(x, mask)
    # x: [D, L, B], mask: [L, B]

    if !isnothing(m.ln)
        x = m.ln(x)
    end

    x = m.linear_in(x)
    x = m.swiglu(x)
    x = m.linear_out(x)

    # Apply mask
    mask_expanded = reshape(mask, 1, size(mask)...)
    return x .* mask_expanded
end

"""
    TransitionADALN(dim::Int, dim_cond::Int; expansion_factor::Int=4)

Transition layer with adaptive layer norm input and adaptive output scaling.

# Architecture
1. AdaptiveLayerNorm(x, cond)
2. SwiGLU Transition
3. AdaptiveOutputScale

# Arguments
- `dim`: Input/output dimension
- `dim_cond`: Conditioning dimension
- `expansion_factor`: Expansion factor for SwiGLU (default 4)

# Example
```julia
trans = TransitionADALN(768, 256)
x = randn(Float32, 768, 128, 2)    # [D, L, B]
cond = randn(Float32, 256, 128, 2) # [D_cond, L, B]
mask = ones(Float32, 128, 2)       # [L, B]
y = trans(x, cond, mask)
```
"""
struct TransitionADALN
    adaln::ProteINAAdaLN
    transition::SwiGLUTransition
    scale_output::AdaptiveOutputScale
end

Flux.@layer TransitionADALN

function TransitionADALN(dim::Int, dim_cond::Int; expansion_factor::Int=4)
    adaln = ProteINAAdaLN(dim, dim_cond)
    transition = SwiGLUTransition(dim; expansion_factor=expansion_factor, use_layer_norm=false)
    scale_output = AdaptiveOutputScale(dim, dim_cond)
    return TransitionADALN(adaln, transition, scale_output)
end

function (m::TransitionADALN)(x, cond, mask)
    # x: [D, L, B]
    # cond: [D_cond, L, B]
    # mask: [L, B]

    x = m.adaln(x, cond, mask)
    x = m.transition(x, mask)
    x = m.scale_output(x, cond, mask)

    # Apply mask
    mask_expanded = reshape(mask, 1, size(mask)...)
    return x .* mask_expanded
end
