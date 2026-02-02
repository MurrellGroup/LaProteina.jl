# Pair features for pair representation
# Port of distance binning and relative sequence separation from feature_factory.py

using NNlib: batched_mul

"""
    pairwise_distances(x::AbstractArray{T,3}) where T

Compute pairwise Euclidean distances between positions.

# Arguments
- `x`: Coordinates of shape [3, L, B] (Julia convention)

# Returns
- Distance matrix of shape [L, L, B]
"""
function pairwise_distances(x::AbstractArray{T,3}) where T
    # x: [3, L, B]
    d, L, B = size(x)

    # Compute squared norms: sum(x^2, dim=1) -> [1, L, B]
    sq_norms = sum(x .^ 2; dims=1)  # [1, L, B]

    # Reshape for broadcasting
    # x_i: [3, L, 1, B], x_j: [3, 1, L, B]
    x_i = reshape(x, d, L, 1, B)
    x_j = reshape(x, d, 1, L, B)

    # Squared distances: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    sq_i = reshape(sq_norms, 1, L, 1, B)  # [1, L, 1, B]
    sq_j = reshape(sq_norms, 1, 1, L, B)  # [1, 1, L, B]
    dot_ij = sum(x_i .* x_j; dims=1)      # [1, L, L, B]

    sq_dists = sq_i .+ sq_j .- 2 .* dot_ij  # [1, L, L, B]
    sq_dists = dropdims(sq_dists; dims=1)   # [L, L, B]

    # Clamp to avoid numerical issues with sqrt
    sq_dists = max.(sq_dists, T(0))
    return sqrt.(sq_dists)
end

"""
    bin_values(tensor::AbstractArray{T}, bin_limits::AbstractVector) where T

Bin values and convert to one-hot encoding. GPU-compatible implementation.

# Arguments
- `tensor`: Input tensor of any shape
- `bin_limits`: Vector of bin boundaries [l1, l2, ..., l_{d-1}]
                Creates d bins: (-∞, l1], (l1, l2], ..., (l_{d-1}, ∞)

# Returns
- One-hot tensor with an additional leading dimension of size d = length(bin_limits) + 1
"""
function bin_values(tensor::AbstractArray{T}, bin_limits::AbstractVector) where T
    n_bins = length(bin_limits) + 1

    # Adapt bin_limits to same device as tensor
    limits_dev = similar(tensor, length(bin_limits))
    copyto!(limits_dev, T.(bin_limits))

    # Compute bin indices using cumulative sum of comparisons
    # bin_index = 1 + sum(tensor > limit_i for all i)
    # This gives bin 1 for smallest values, bin n_bins for largest

    # Reshape limits for broadcasting: [n_limits, 1, 1, ...]
    ndim_tensor = ndims(tensor)
    limits_shape = (length(bin_limits), ntuple(_ -> 1, ndim_tensor)...)
    limits_reshaped = reshape(limits_dev, limits_shape...)

    # Reshape tensor for broadcasting: [1, size(tensor)...]
    tensor_reshaped = reshape(tensor, 1, size(tensor)...)

    # Compare tensor to each limit: [n_limits, size(tensor)...]
    # comparisons[i, ...] = 1 if tensor > limits[i], else 0
    comparisons = T.(tensor_reshaped .> limits_reshaped)

    # Sum along first dim to get bin indices (0-based, then add 1)
    bin_indices = dropdims(sum(comparisons; dims=1); dims=1) .+ one(T)  # [size(tensor)...]

    # Convert to one-hot using NNlib.onehotbatch-style approach
    # Create result tensor
    result_shape = (n_bins, size(tensor)...)
    result = similar(tensor, result_shape)
    fill!(result, zero(T))

    # For each bin, create a mask and set values
    # Use broadcasting: result[b, ...] = (bin_indices == b)
    bin_range_cpu = T.(1:n_bins)
    bin_range_dev = similar(tensor, n_bins)
    copyto!(bin_range_dev, bin_range_cpu)
    bin_range = reshape(bin_range_dev, n_bins, ntuple(_ -> 1, ndim_tensor)...)
    bin_indices_exp = reshape(bin_indices, 1, size(bin_indices)...)

    result .= T.(bin_range .== bin_indices_exp)

    return result
end

"""
    bin_pairwise_distances(x::AbstractArray{T,3}, min_dist::Real, max_dist::Real, n_bins::Int) where T

Bin pairwise distances into one-hot vectors.

# Arguments
- `x`: Coordinates of shape [3, L, B]
- `min_dist`: Minimum distance (right limit of first bin)
- `max_dist`: Maximum distance (left limit of last bin)
- `n_bins`: Number of bins

# Returns
- One-hot binned distances of shape [n_bins, L, L, B]
"""
function bin_pairwise_distances(x::AbstractArray{T,3}, min_dist::Real, max_dist::Real, n_bins::Int) where T
    # Compute pairwise distances
    dists = pairwise_distances(x)  # [L, L, B]

    # Create bin limits
    bin_limits = T.(range(min_dist, max_dist, length=n_bins-1))

    # Bin and one-hot encode
    return bin_values(dists, bin_limits)  # [n_bins, L, L, B]
end

"""
    relative_sequence_separation(seq_len::Int, batch_size::Int; max_sep::Int=32, T=Float32)

Compute relative sequence separation features as one-hot vectors.

# Arguments
- `seq_len`: Sequence length L
- `batch_size`: Batch size B
- `max_sep`: Maximum separation to encode (values beyond are binned together)

# Returns
- One-hot encoded relative separations of shape [2*max_sep+1, L, L, B]
"""
function relative_sequence_separation(seq_len::Int, batch_size::Int; max_sep::Int=32, T=Float32)
    # Create position indices
    positions = 1:seq_len

    # Compute relative separations: i - j (matching Python convention)
    # Python: rel_sep[b, i, j] = positions[b, i] - positions[b, j]
    rel_sep = positions .- positions'  # [L, L] where rel_sep[i, j] = i - j

    # Clamp to [-max_sep, max_sep]
    clamped = clamp.(rel_sep, -max_sep, max_sep)

    # Shift to 1-indexed: [-max_sep, max_sep] -> [1, 2*max_sep+1]
    shifted = clamped .+ (max_sep + 1)

    # Create one-hot encoding
    n_classes = 2 * max_sep + 1
    onehot = zeros(T, n_classes, seq_len, seq_len, batch_size)

    for i in 1:seq_len, j in 1:seq_len
        class_idx = shifted[i, j]
        onehot[class_idx, i, j, :] .= one(T)
    end

    return onehot
end

"""
    relative_sequence_separation(positions::AbstractMatrix{<:Integer}; max_sep::Int=32, T=Float32)

Compute relative sequence separation from actual position indices.

# Arguments
- `positions`: Position indices of shape [L, B]
- `max_sep`: Maximum separation to encode

# Returns
- One-hot encoded relative separations of shape [2*max_sep+1, L, L, B]
"""
function relative_sequence_separation(positions::AbstractMatrix{<:Integer}; max_sep::Int=32, T=Float32)
    L, B = size(positions)
    n_classes = 2 * max_sep + 1

    onehot = zeros(T, n_classes, L, L, B)

    for b in 1:B
        for i in 1:L, j in 1:L
            # Python: rel_sep[b, i, j] = positions[b, i] - positions[b, j]
            rel_sep = positions[i, b] - positions[j, b]
            clamped = clamp(rel_sep, -max_sep, max_sep)
            class_idx = clamped + max_sep + 1
            onehot[class_idx, i, j, b] = one(T)
        end
    end

    return onehot
end

"""
    outer_product_mean(x::AbstractArray{T,3}; dims=1) where T

Compute outer product mean for pair features.

# Arguments
- `x`: Sequence features of shape [D, L, B]

# Returns
- Pair features of shape [D, D, L, L, B] reduced to [D², L, L, B] if flattened
"""
function outer_product(x::AbstractArray{T,3}) where T
    D, L, B = size(x)
    # x_i: [D, L, 1, B], x_j: [D, 1, L, B]
    x_i = reshape(x, D, L, 1, B)
    x_j = reshape(x, D, 1, L, B)
    # Outer product: [D, D, L, L, B] - but we'll flatten D dimensions
    # For now, just return the element-wise product sum approach
    return x_i .* x_j  # Broadcasting gives [D, L, L, B] if we sum over appropriate dims
end

"""
    pair_features_from_coords(coords::AbstractArray{T,3};
        dist_min::Real=0.0, dist_max::Real=20.0, n_dist_bins::Int=64,
        max_sep::Int=32) where T

Create pair features from coordinates.

# Arguments
- `coords`: CA coordinates of shape [3, L, B]
- `dist_min`, `dist_max`, `n_dist_bins`: Distance binning parameters
- `max_sep`: Maximum relative sequence separation

# Returns
Named tuple with:
- `dist_features`: [n_dist_bins, L, L, B]
- `sep_features`: [2*max_sep+1, L, L, B]
"""
function pair_features_from_coords(coords::AbstractArray{T,3};
        dist_min::Real=0.0, dist_max::Real=20.0, n_dist_bins::Int=64,
        max_sep::Int=32) where T
    L, B = size(coords, 2), size(coords, 3)

    dist_features = bin_pairwise_distances(coords, dist_min, dist_max, n_dist_bins)
    sep_features = relative_sequence_separation(L, B; max_sep=max_sep, T=T)

    return (dist_features=dist_features, sep_features=sep_features)
end
