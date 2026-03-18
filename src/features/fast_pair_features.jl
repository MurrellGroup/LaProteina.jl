# Vectorized pair feature extraction — replaces scalar loops with broadcasting
# These are drop-in replacements for the slow pair feature callables.
#
# The bottleneck in the original code is:
#   1. ResidueOrientationFeature: triple-nested loop computing 5 angles per L² pairs
#   2. BackbonePairDistFeature: triple-nested loop for distance binning
#   3. RelSeqSepFeature: double-nested loop for one-hot encoding
#
# This file provides vectorized alternatives using broadcasting.

# ============================================================================
# Vectorized one-hot binning utility
# ============================================================================

"""
    _vectorized_bin_onehot(values, edges, n_bins, mask)

Bin continuous values into one-hot representation.
Replaces the scalar loop: for each value, count how many edges it exceeds.

Returns: [n_bins, size(values)...] one-hot encoded bins
"""
function _vectorized_bin_onehot(values::AbstractArray{T, 3}, edges::Vector{T},
                                 n_bins::Int, mask::AbstractArray{T, 3}) where T
    L1, L2, B = size(values)
    result = zeros(T, n_bins, L1, L2, B)

    # Compute bin indices vectorized: count edges exceeded
    bin_indices = ones(Int, L1, L2, B)
    @inbounds for edge in edges
        bin_indices .+= (values .>= edge)
    end
    clamp!(bin_indices, 1, n_bins)

    # Write one-hot (this loop is cheap — just indexed writes, no computation)
    @inbounds for b in 1:B, j in 1:L2, i in 1:L1
        if mask[i, j, b] > T(0.5)
            result[bin_indices[i, j, b], i, j, b] = one(T)
        end
    end
    return result
end

# ============================================================================
# Fast RelSeqSepFeature
# ============================================================================

function fast_relative_sequence_separation(seq_len::Int, batch_size::Int;
                                            max_sep::Int=32, T=Float32)
    n_classes = 2 * max_sep + 1
    rel_sep = (1:seq_len) .- (1:seq_len)'  # [L, L]
    clamped = clamp.(rel_sep, -max_sep, max_sep)
    shifted = clamped .+ (max_sep + 1)

    onehot = zeros(T, n_classes, seq_len, seq_len, batch_size)
    @inbounds for j in 1:seq_len, i in 1:seq_len
        idx = shifted[i, j]
        for b in 1:batch_size
            onehot[idx, i, j, b] = one(T)
        end
    end
    return onehot
end

function fast_relative_sequence_separation(positions::AbstractMatrix{<:Integer};
                                            max_sep::Int=32, T=Float32)
    L, B = size(positions)
    n_classes = 2 * max_sep + 1
    onehot = zeros(T, n_classes, L, L, B)

    for b in 1:B
        pos_b = @view positions[:, b]
        @inbounds for j in 1:L, i in 1:L
            rel_sep = pos_b[i] - pos_b[j]
            clamped = clamp(rel_sep, -max_sep, max_sep)
            idx = clamped + max_sep + 1
            onehot[idx, i, j, b] = one(T)
        end
    end
    return onehot
end

# ============================================================================
# Fast BackbonePairDistFeature
# ============================================================================

function fast_backbone_pair_dist(batch::Dict, L::Int, B::Int; n_bins::Int=21)
    dim = 4 * n_bins

    if !haskey(batch, :coords_nm) && !haskey(batch, :coords)
        return zeros(Float32, dim, L, L, B)
    end

    coords = get(batch, :coords_nm, get(batch, :coords, nothing))
    coord_mask = get(batch, :coord_mask, ones(Float32, 37, L, B))

    N_IDX, CA_IDX, C_IDX, CB_IDX = 1, 2, 3, 4

    N  = coords[:, N_IDX, :, :]
    CA = coords[:, CA_IDX, :, :]
    C  = coords[:, C_IDX, :, :]
    CB = coords[:, CB_IDX, :, :]

    ca_mask = coord_mask[CA_IDX, :, :]
    cb_mask = coord_mask[CB_IDX, :, :]

    # Pairwise distances (vectorized — same as original)
    CA_i = reshape(CA, 3, L, 1, B)
    N_j  = reshape(N,  3, 1, L, B)
    CA_j = reshape(CA, 3, 1, L, B)
    C_j  = reshape(C,  3, 1, L, B)
    CB_j = reshape(CB, 3, 1, L, B)

    dist_CA_N  = dropdims(sqrt.(sum((CA_i .- N_j ).^2, dims=1)), dims=1)
    dist_CA_CA = dropdims(sqrt.(sum((CA_i .- CA_j).^2, dims=1)), dims=1)
    dist_CA_C  = dropdims(sqrt.(sum((CA_i .- C_j ).^2, dims=1)), dims=1)
    dist_CA_CB = dropdims(sqrt.(sum((CA_i .- CB_j).^2, dims=1)), dims=1)

    pair_mask    = reshape(ca_mask, L, 1, B) .* reshape(ca_mask, 1, L, B)
    cb_pair_mask = reshape(cb_mask, 1, L, B)

    dist_CA_N  .*= pair_mask
    dist_CA_CA .*= pair_mask
    dist_CA_C  .*= pair_mask
    dist_CA_CB .*= pair_mask .* cb_pair_mask

    n_edges = n_bins - 1
    edges = collect(range(0.1f0, 2.0f0, length=n_edges))

    result = zeros(Float32, dim, L, L, B)

    # Vectorized binning for each distance type
    # NOTE: The original code uses pair_mask (not cb_pair_mask) for the one-hot write.
    # This means for distance 4 (CA→CB), even when CB is missing the distance is zero
    # but the one-hot bin 1 still gets pair_mask=1.0. We match this for parity.
    for (d_idx, dist) in enumerate([dist_CA_N, dist_CA_CA, dist_CA_C, dist_CA_CB])
        offset = (d_idx - 1) * n_bins
        binned = _vectorized_bin_onehot(dist, edges, n_bins, pair_mask)
        result[offset+1:offset+n_bins, :, :, :] .= binned
    end

    return result
end

# ============================================================================
# Fast ResidueOrientationFeature — vectorized dihedrals and bond angles
# ============================================================================

# selectdim(a, 1, k:k) extracts component k as a view keeping all dims.
# For [3, L, 1, B] → [1, L, 1, B]; for [3, L, L, B] → [1, L, L, B].
@inline _c(a, k) = selectdim(a, 1, k:k)

"""
    _vectorized_dihedral(a, b, c, d)

Vectorized dihedral angle for 4-point sequences. Inputs are [3, ...] arrays
that broadcast together. Returns angles with the first (size-3) dim removed.

Handles degenerate cases (colinear atoms) by returning 0, matching the
scalar `_scalar_dihedral` behavior.
"""
function _vectorized_dihedral(a::AbstractArray{T}, b::AbstractArray{T},
                               c::AbstractArray{T}, d::AbstractArray{T}) where T
    b0 = b .- a
    b1 = c .- b
    b2 = d .- c

    # Cross products (component-wise for first dim = 3)
    n1_x = _c(b0,2) .* _c(b1,3) .- _c(b0,3) .* _c(b1,2)
    n1_y = _c(b0,3) .* _c(b1,1) .- _c(b0,1) .* _c(b1,3)
    n1_z = _c(b0,1) .* _c(b1,2) .- _c(b0,2) .* _c(b1,1)

    n2_x = _c(b1,2) .* _c(b2,3) .- _c(b1,3) .* _c(b2,2)
    n2_y = _c(b1,3) .* _c(b2,1) .- _c(b1,1) .* _c(b2,3)
    n2_z = _c(b1,1) .* _c(b2,2) .- _c(b1,2) .* _c(b2,1)

    # Norms (without epsilon — check degenerate separately)
    n1_sq = n1_x.^2 .+ n1_y.^2 .+ n1_z.^2
    n2_sq = n2_x.^2 .+ n2_y.^2 .+ n2_z.^2
    n1_norm = sqrt.(n1_sq)
    n2_norm = sqrt.(n2_sq)

    # Degenerate mask: either normal is near-zero (matches scalar threshold 1e-8)
    degen = (n1_norm .< T(1e-8)) .| (n2_norm .< T(1e-8))

    # Safe normalize (add epsilon only to avoid NaN in division)
    safe_n1 = n1_norm .+ T(1e-16)
    safe_n2 = n2_norm .+ T(1e-16)
    n1_x = n1_x ./ safe_n1; n1_y = n1_y ./ safe_n1; n1_z = n1_z ./ safe_n1
    n2_x = n2_x ./ safe_n2; n2_y = n2_y ./ safe_n2; n2_z = n2_z ./ safe_n2

    # cos(angle) = n1 · n2
    cos_angle = n1_x .* n2_x .+ n1_y .* n2_y .+ n1_z .* n2_z

    # n1 × n2 for sin
    cx = n1_y .* n2_z .- n1_z .* n2_y
    cy = n1_z .* n2_x .- n1_x .* n2_z
    cz = n1_x .* n2_y .- n1_y .* n2_x
    sin_mag = sqrt.(cx.^2 .+ cy.^2 .+ cz.^2)

    # Sign from dot(n1×n2, b1)
    sign_val = sign.(cx .* _c(b1,1) .+ cy .* _c(b1,2) .+ cz .* _c(b1,3))

    angles = atan.(sign_val .* sin_mag, cos_angle)

    # Zero out degenerate cases (matching scalar code)
    angles = ifelse.(degen, zero(T), angles)

    return dropdims(angles, dims=1)
end

"""
    _vectorized_bond_angle(a, b, c)

Vectorized bond angle at vertex a between vectors a→b and a→c.
Handles degenerate cases by returning 0.
"""
function _vectorized_bond_angle(a::AbstractArray{T}, b::AbstractArray{T},
                                 c::AbstractArray{T}) where T
    v0 = b .- a
    v1 = c .- a

    n0 = sqrt.(sum(v0.^2, dims=1))
    n1 = sqrt.(sum(v1.^2, dims=1))

    # Degenerate mask
    degen = (n0 .< T(1e-8)) .| (n1 .< T(1e-8))

    # Safe normalize
    v0n = v0 ./ (n0 .+ T(1e-16))
    v1n = v1 ./ (n1 .+ T(1e-16))

    cos_angle = sum(v0n .* v1n, dims=1)

    # Cross product for sin
    cx = _c(v0n,2) .* _c(v1n,3) .- _c(v0n,3) .* _c(v1n,2)
    cy = _c(v0n,3) .* _c(v1n,1) .- _c(v0n,1) .* _c(v1n,3)
    cz = _c(v0n,1) .* _c(v1n,2) .- _c(v0n,2) .* _c(v1n,1)
    sin_angle = sqrt.(cx.^2 .+ cy.^2 .+ cz.^2)

    angles = atan.(sin_angle, cos_angle)
    angles = ifelse.(degen, zero(T), angles)

    return dropdims(angles, dims=1)
end

"""
    fast_residue_orientation(batch, L, B; n_bins=21)

Vectorized residue orientation features. Computes all 5 angles for all L² pairs
simultaneously using broadcasting, then bins into one-hot.

Drop-in replacement for ResidueOrientationFeature()(batch, L, B).
"""
function fast_residue_orientation(batch::Dict, L::Int, B::Int; n_bins::Int=21)
    dim = 5 * n_bins

    if !haskey(batch, :coords) && !haskey(batch, :coords_nm)
        return zeros(Float32, dim, L, L, B)
    end

    coords = get(batch, :coords, get(batch, :coords_nm, nothing))
    coord_mask = get(batch, :coord_mask, ones(Float32, 37, L, B))

    N_IDX, CA_IDX, CB_IDX = 1, 2, 4

    N  = coords[:, N_IDX, :, :]   # [3, L, B]
    CA = coords[:, CA_IDX, :, :]
    CB = coords[:, CB_IDX, :, :]

    ca_mask = coord_mask[CA_IDX, :, :]
    cb_mask = coord_mask[CB_IDX, :, :]
    pair_mask = reshape(ca_mask .* cb_mask, L, 1, B) .* reshape(ca_mask .* cb_mask, 1, L, B)

    # Expand for pairwise broadcasting: _i → [3, L, 1, B], _j → [3, 1, L, B]
    N_i  = reshape(N,  3, L, 1, B);  N_j  = reshape(N,  3, 1, L, B)
    CA_i = reshape(CA, 3, L, 1, B);  CA_j = reshape(CA, 3, 1, L, B)
    CB_i = reshape(CB, 3, L, 1, B);  CB_j = reshape(CB, 3, 1, L, B)

    # All 5 angles vectorized over [L, L, B]
    theta_12 = _vectorized_dihedral(N_i, CA_i, CB_i, CB_j)
    theta_21 = _vectorized_dihedral(N_j, CA_j, CB_j, CB_i)
    phi_12   = _vectorized_bond_angle(CA_i, CB_i, CB_j)
    phi_21   = _vectorized_bond_angle(CA_j, CB_j, CB_i)
    omega    = _vectorized_dihedral(CA_i, CB_i, CB_j, CA_j)

    # Bin all 5 angles
    n_edges = n_bins - 1
    edges = collect(range(Float32(-π), Float32(π), length=n_edges))

    result = zeros(Float32, dim, L, L, B)
    for (a_idx, angles) in enumerate([theta_12, theta_21, phi_12, phi_21, omega])
        offset = (a_idx - 1) * n_bins
        binned = _vectorized_bin_onehot(angles, edges, n_bins, pair_mask)
        result[offset+1:offset+n_bins, :, :, :] .= binned
    end

    return result
end

# ============================================================================
# Fast feature extraction: drop-in replacement for extract_encoder_features
# ============================================================================

"""
    extract_encoder_features_fast(encoder, batch)

Drop-in replacement for `extract_encoder_features` that uses vectorized pair
feature computation. Returns identical EncoderRawFeatures.

The pair features (RelSeqSep, BackbonePairDist, ResidueOrientation) are
computed with vectorized broadcasting instead of scalar loops.
"""
function extract_encoder_features_fast(encoder, batch::Dict)
    mask = batch[:mask]
    L, B = size(mask)

    # Sequence features — unchanged (already fast, no L² operations)
    seq_features = [f(batch, L, B) for f in encoder.init_repr_factory.features]
    seq_raw = cat(seq_features..., dims=1)

    # Conditioning features — unchanged
    cond_features = [f(batch, L, B) for f in encoder.cond_factory.features]
    if isempty(cond_features)
        cond_raw = zeros(Float32, encoder.cond_factory.out_dim, L, B)
    else
        cond_raw = cat(cond_features..., dims=1)
    end

    # Pair features — use fast vectorized versions
    pair_parts = []
    for f in encoder.pair_rep_factory.features
        if f isa RelSeqSepFeature
            if haskey(batch, :pdb_idx)
                push!(pair_parts, fast_relative_sequence_separation(batch[:pdb_idx]; max_sep=f.max_sep))
            else
                push!(pair_parts, fast_relative_sequence_separation(L, B; max_sep=f.max_sep))
            end
        elseif f isa BackbonePairDistFeature
            push!(pair_parts, fast_backbone_pair_dist(batch, L, B; n_bins=f.n_bins))
        elseif f isa ResidueOrientationFeature
            push!(pair_parts, fast_residue_orientation(batch, L, B; n_bins=f.n_bins))
        else
            # Fallback to original for unknown feature types
            push!(pair_parts, f(batch, L, B))
        end
    end
    pair_raw = cat(pair_parts..., dims=1)

    return EncoderRawFeatures(seq_raw, cond_raw, pair_raw, Float32.(mask))
end
