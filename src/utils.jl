# Tensor conversion utilities between Python (PyTorch) and Julia conventions
# Python (PyTorch): [Batch, Length, Dim] (row-major)
# Julia (Onion.jl): [Dim, Length, Batch] (column-major)

"""
    python_to_julia(arr::AbstractArray{T,3}) where T

Convert tensor from Python [B,L,D] to Julia [D,L,B] convention.
"""
function python_to_julia(arr::AbstractArray{T,3}) where T
    return permutedims(arr, (3, 2, 1))
end

"""
    julia_to_python(arr::AbstractArray{T,3}) where T

Convert tensor from Julia [D,L,B] to Python [B,L,D] convention.
"""
function julia_to_python(arr::AbstractArray{T,3}) where T
    return permutedims(arr, (3, 2, 1))
end

"""
    python_to_julia_pair(arr::AbstractArray{T,4}) where T

Convert pair tensor from Python [B,L_i,L_j,D] to Julia [D,L_i,L_j,B] convention.
Preserves the ordering of the L_i and L_j dimensions.
"""
function python_to_julia_pair(arr::AbstractArray{T,4}) where T
    # Python dims: 1=B, 2=L_i, 3=L_j, 4=D
    # Julia dims:  1=D, 2=L_i, 3=L_j, 4=B
    return permutedims(arr, (4, 2, 3, 1))
end

"""
    julia_to_python_pair(arr::AbstractArray{T,4}) where T

Convert pair tensor from Julia [D,L_i,L_j,B] to Python [B,L_i,L_j,D] convention.
"""
function julia_to_python_pair(arr::AbstractArray{T,4}) where T
    # Julia dims:  1=D, 2=L_i, 3=L_j, 4=B
    # Python dims: 1=B, 2=L_i, 3=L_j, 4=D
    return permutedims(arr, (4, 2, 3, 1))
end

"""
    python_to_julia_mask(arr::AbstractArray{T,2}) where T

Convert mask from Python [B,L] to Julia [L,B] convention.
"""
function python_to_julia_mask(arr::AbstractArray{T,2}) where T
    return permutedims(arr, (2, 1))
end

"""
    julia_to_python_mask(arr::AbstractArray{T,2}) where T

Convert mask from Julia [L,B] to Python [B,L] convention.
"""
function julia_to_python_mask(arr::AbstractArray{T,2}) where T
    return permutedims(arr, (2, 1))
end

"""
    python_to_julia_coords(arr::AbstractArray{T,4}) where T

Convert atom coordinates from Python [B,L,37,3] to Julia [3,37,L,B] convention.
"""
function python_to_julia_coords(arr::AbstractArray{T,4}) where T
    return permutedims(arr, (4, 3, 2, 1))
end

"""
    julia_to_python_coords(arr::AbstractArray{T,4}) where T

Convert atom coordinates from Julia [3,37,L,B] to Python [B,L,37,3] convention.
"""
function julia_to_python_coords(arr::AbstractArray{T,4}) where T
    return permutedims(arr, (4, 3, 2, 1))
end

# Center of mass utilities for zero-COM constraint

"""
    center_of_mass(x::AbstractArray{T}, mask=nothing; dims=2) where T

Compute center of mass along specified dimension.

# Arguments
- `x`: Coordinates array
- `mask`: Optional mask (true = included in COM calculation)
- `dims`: Dimension(s) to average over

# Returns
- Center of mass with same dimensionality as input
"""
function center_of_mass(x::AbstractArray{T}, mask=nothing; dims=2) where T
    if isnothing(mask)
        return mean(x; dims=dims)
    else
        # Expand mask to match x dimensions
        expanded_mask = expand_mask(mask, ndims(x))
        masked_x = x .* expanded_mask
        n = sum(expanded_mask; dims=dims)
        n = max.(n, one(T))  # Avoid division by zero
        return sum(masked_x; dims=dims) ./ n
    end
end

"""
    zero_center_of_mass(x::AbstractArray{T}, mask=nothing; dims=2) where T

Subtract center of mass to make x zero-centered.
"""
function zero_center_of_mass(x::AbstractArray{T}, mask=nothing; dims=2) where T
    com = center_of_mass(x, mask; dims=dims)
    result = x .- com
    if !isnothing(mask)
        expanded_mask = expand_mask(mask, ndims(x))
        result = result .* expanded_mask
    end
    return result
end

"""
    expand_mask(mask::AbstractArray, target_ndims::Int)

Expand a mask to match target number of dimensions by adding leading singleton dims.
"""
function expand_mask(mask::AbstractArray, target_ndims::Int)
    current_ndims = ndims(mask)
    if current_ndims >= target_ndims
        return mask
    end
    new_shape = (ntuple(_ -> 1, target_ndims - current_ndims)..., size(mask)...)
    return reshape(mask, new_shape)
end

"""
    masked_mean(x::AbstractArray{T}, mask::AbstractArray; dims=:) where T

Compute mean of x over masked positions.
"""
function masked_mean(x::AbstractArray{T}, mask::AbstractArray; dims=:) where T
    expanded_mask = expand_mask(mask, ndims(x))
    if dims == (:)
        n = sum(expanded_mask)
        return sum(x .* expanded_mask) / max(n, one(T))
    else
        n = sum(expanded_mask; dims=dims)
        n = max.(n, one(T))
        return sum(x .* expanded_mask; dims=dims) ./ n
    end
end

"""
    masked_mse(pred::AbstractArray{T}, target::AbstractArray{T}, mask::AbstractArray) where T

Compute masked mean squared error.
"""
function masked_mse(pred::AbstractArray{T}, target::AbstractArray{T}, mask::AbstractArray) where T
    diff_sq = (pred .- target) .^ 2
    return masked_mean(diff_sq, mask)
end

# Weight initialization helpers

"""
    zero_init!(layer)

Initialize a Dense layer with zero weights (useful for AdaLN zero initialization).
"""
function zero_init!(layer)
    if hasproperty(layer, :weight)
        fill!(layer.weight, 0)
    end
    if hasproperty(layer, :bias) && !isnothing(layer.bias)
        fill!(layer.bias, 0)
    end
    return layer
end

"""
    constant_init!(layer, weight_val, bias_val)

Initialize a Dense layer with constant values.
"""
function constant_init!(layer, weight_val, bias_val)
    if hasproperty(layer, :weight)
        fill!(layer.weight, weight_val)
    end
    if hasproperty(layer, :bias) && !isnothing(layer.bias)
        fill!(layer.bias, bias_val)
    end
    return layer
end

# ============================================================================
# PyTorch-compatible LayerNorm
# ============================================================================
# Flux uses sqrt(var + eps^2) while PyTorch uses sqrt(var + eps)
# This causes massive divergence when variance is tiny (< eps)

"""
    pytorch_normalise(x; dims, eps=1f-5)

PyTorch-compatible normalization using sqrt(var + eps) instead of Flux's sqrt(var + eps^2).
"""
function pytorch_normalise(x::AbstractArray{T}; dims, eps::Real=1f-5) where T
    μ = mean(x, dims=dims)
    σ² = var(x, dims=dims, mean=μ, corrected=false)
    ε = T(eps)
    return @. (x - μ) / sqrt(σ² + ε)
end

"""
    PyTorchLayerNorm{F,S,B,N}

LayerNorm that matches PyTorch's behavior: normalizes using sqrt(var + eps).

Flux's LayerNorm uses sqrt(var + eps^2) which causes divergence when variance << eps.
"""
struct PyTorchLayerNorm{F,S,B,N}
    λ::F
    scale::S
    bias::B
    ϵ::Float32
    size::NTuple{N,Int}
    affine::Bool
end

Flux.@layer PyTorchLayerNorm

function PyTorchLayerNorm(size::Tuple{Vararg{Int}}, λ=identity; affine::Bool=true, eps::Real=1f-5)
    scale = affine ? ones(Float32, size...) : nothing
    bias = affine ? zeros(Float32, size...) : nothing
    return PyTorchLayerNorm(λ, scale, bias, Float32(eps), size, affine)
end

PyTorchLayerNorm(size::Integer...; kw...) = PyTorchLayerNorm(Int.(size); kw...)
PyTorchLayerNorm(size_act...; kw...) = PyTorchLayerNorm(Int.(size_act[1:end-1]), size_act[end]; kw...)

function (ln::PyTorchLayerNorm)(x::AbstractArray)
    y = pytorch_normalise(x; dims=1:length(ln.size), eps=ln.ϵ)
    if ln.affine
        y = y .* ln.scale .+ ln.bias
    end
    return ln.λ(y)
end

function Base.show(io::IO, ln::PyTorchLayerNorm)
    print(io, "PyTorchLayerNorm(", join(ln.size, ", "))
    ln.λ === identity || print(io, ", ", ln.λ)
    ln.affine || print(io, ", affine=false")
    print(io, ")")
end
