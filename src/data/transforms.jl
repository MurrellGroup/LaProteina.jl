# Data augmentation transforms for protein structures

using LinearAlgebra
using Random

"""
    random_rotation_matrix(T::Type=Float32)

Generate a random 3x3 rotation matrix (uniformly distributed on SO(3)).
"""
function random_rotation_matrix(T::Type=Float32)
    # Use QR decomposition of random matrix for uniform distribution on SO(3)
    M = randn(T, 3, 3)
    Q, R = qr(M)
    Q = Matrix(Q) * Diagonal(sign.(diag(R)))

    # Ensure proper rotation (det = +1)
    if det(Q) < 0
        Q[:, 1] *= -1
    end

    return T.(Q)
end

"""
    random_rotation(coords::AbstractArray{T,3}; mask=nothing) where T

Apply random rotation to coordinates.

# Arguments
- `coords`: Coordinates of shape [3, ..., L, B] or [3, 37, L, B]
- `mask`: Optional mask [L, B]

# Returns
Rotated coordinates with same shape
"""
function random_rotation(coords::AbstractArray{T,3}; mask=nothing) where T
    # coords: [3, L, B] for CA or similar
    B = size(coords, 3)
    result = similar(coords)

    for b in 1:B
        R = random_rotation_matrix(T)
        # Apply rotation: R @ coords[:, :, b]
        result[:, :, b] = R * coords[:, :, b]
    end

    if !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        result = result .* mask_exp
    end

    return result
end

function random_rotation(coords::AbstractArray{T,4}; mask=nothing) where T
    # coords: [3, 37, L, B] for all-atom
    B = size(coords, 4)
    result = similar(coords)

    for b in 1:B
        R = random_rotation_matrix(T)
        for l in axes(coords, 3), a in axes(coords, 2)
            result[:, a, l, b] = R * coords[:, a, l, b]
        end
    end

    if !isnothing(mask)
        mask_exp = reshape(mask, 1, 1, size(mask)...)
        result = result .* mask_exp
    end

    return result
end

"""
    center_coords(coords::AbstractArray{T}; mask=nothing, ca_only::Bool=true) where T

Center coordinates at the center of mass.

# Arguments
- `coords`: Coordinates [3, L, B] or [3, 37, L, B]
- `mask`: Optional mask [L, B]
- `ca_only`: If true and coords is all-atom, use CA for COM

# Returns
Centered coordinates
"""
function center_coords(coords::AbstractArray{T,3}; mask=nothing) where T
    # coords: [3, L, B]
    if isnothing(mask)
        com = mean(coords; dims=2)
    else
        mask_exp = reshape(mask, 1, size(mask)...)
        masked_coords = coords .* mask_exp
        n = sum(mask_exp; dims=2)
        n = max.(n, one(T))
        com = sum(masked_coords; dims=2) ./ n
    end

    result = coords .- com

    if !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        result = result .* mask_exp
    end

    return result
end

function center_coords(coords::AbstractArray{T,4}; mask=nothing, ca_only::Bool=true) where T
    # coords: [3, 37, L, B]
    if ca_only
        # Use CA for COM
        ca_coords = coords[:, CA_INDEX, :, :]  # [3, L, B]
        com = center_coords(ca_coords; mask=mask)
        com_4d = reshape(mean(ca_coords; dims=2) .- mean(com; dims=2), 3, 1, 1, size(coords, 4))
        result = coords .- reshape(com_4d, 3, 1, 1, :)
    else
        # Use all atoms (more complex)
        if isnothing(mask)
            com = mean(coords; dims=(2, 3))
        else
            mask_exp = reshape(mask, 1, 1, size(mask)...)
            masked_coords = coords .* mask_exp
            n = sum(mask_exp; dims=(2, 3))
            n = max.(n, one(T))
            com = sum(masked_coords; dims=(2, 3)) ./ n
        end
        result = coords .- com
    end

    if !isnothing(mask)
        mask_exp = reshape(mask, 1, 1, size(mask)...)
        result = result .* mask_exp
    end

    return result
end

"""
    add_noise(coords::AbstractArray{T}, sigma::Real; mask=nothing, zero_com::Bool=true) where T

Add Gaussian noise to coordinates.

# Arguments
- `coords`: Coordinates of any shape
- `sigma`: Noise standard deviation
- `mask`: Optional mask
- `zero_com`: If true, re-center after adding noise

# Returns
Noisy coordinates
"""
function add_noise(coords::AbstractArray{T}, sigma::Real; mask=nothing, zero_com::Bool=true) where T
    noise = randn(T, size(coords)) .* T(sigma)

    if !isnothing(mask)
        # Expand mask to match coords dims
        while ndims(mask) < ndims(coords)
            mask = reshape(mask, 1, size(mask)...)
        end
        noise = noise .* mask
    end

    result = coords .+ noise

    if zero_com && ndims(coords) == 3  # Only for [3, L, B] format
        result = center_coords(result; mask=nothing)
        if !isnothing(mask)
            result = result .* mask
        end
    end

    return result
end

"""
    random_crop(data::Dict, max_length::Int)

Randomly crop protein to maximum length.

# Arguments
- `data`: Dict with :coords, :aatype, etc.
- `max_length`: Maximum length after cropping

# Returns
Cropped data dict
"""
function random_crop(data::Dict, max_length::Int)
    L = size(data[:coords], ndims(data[:coords]) - 1)  # L is second-to-last dim

    if L <= max_length
        return data
    end

    # Random start position
    start_idx = rand(1:(L - max_length + 1))
    end_idx = start_idx + max_length - 1

    # Crop all arrays
    result = Dict{Symbol, Any}()

    for (key, val) in data
        if val isa AbstractArray
            ndim = ndims(val)
            # Assume L is the second-to-last dimension
            if ndim >= 2
                idx = ntuple(i -> i == ndim - 1 ? (start_idx:end_idx) : Colon(), ndim)
                result[key] = val[idx...]
            else
                result[key] = val
            end
        else
            result[key] = val
        end
    end

    result[:cropped] = true
    return result
end

"""
    apply_transforms(data::Dict;
        rotate::Bool=true,
        center::Bool=true,
        max_length::Union{Int,Nothing}=nothing,
        noise_sigma::Float32=0.0f0)

Apply a sequence of transforms to protein data.
"""
function apply_transforms(data::Dict;
        rotate::Bool=true,
        center::Bool=true,
        max_length::Union{Int,Nothing}=nothing,
        noise_sigma::Float32=0.0f0)

    result = copy(data)

    # Crop if needed
    if !isnothing(max_length)
        result = random_crop(result, max_length)
    end

    # Get mask
    mask = get(result, :residue_mask, get(result, :mask, nothing))

    # Center
    if center && haskey(result, :coords)
        result[:coords] = center_coords(result[:coords]; mask=mask)
    end

    # Rotate
    if rotate && haskey(result, :coords)
        result[:coords] = random_rotation(result[:coords]; mask=mask)
    end

    # Add noise
    if noise_sigma > 0 && haskey(result, :coords)
        result[:coords] = add_noise(result[:coords], noise_sigma; mask=mask)
    end

    return result
end
