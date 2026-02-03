# Geometry utilities for angle calculations
# Port of proteinfoundation/utils/angle_utils.py

using LinearAlgebra

"""
    normalize_last_dim(v::AbstractArray)

Normalize vectors along the last dimension.
Returns v / ||v|| with clamping to avoid division by zero.
"""
function normalize_last_dim(v::AbstractArray{T}) where T
    # Compute norm along last dimension
    norm_v = sqrt.(sum(v .^ 2, dims=ndims(v)))
    # Clamp to avoid division by zero
    norm_v = max.(norm_v, T(1e-8))
    return v ./ norm_v
end

"""
    signed_dihedral_angle(a, b, c, d)

Computes the signed dihedral angle for 4 points a, b, c, d.
The angle is measured between planes (a,b,c) and (b,c,d).

Supports broadcasting - all inputs should have shape [*, 3]
where * can be any number of batch dimensions.

Returns angle in range [-π, π] with shape [*].
"""
function signed_dihedral_angle(a::AbstractArray{T}, b::AbstractArray{T},
                               c::AbstractArray{T}, d::AbstractArray{T}) where T
    # Bond vectors
    b0 = b .- a  # [*, 3]
    b1 = c .- b  # [*, 3]
    b2 = d .- c  # [*, 3]

    # Compute normal vectors to the planes
    n1 = _cross_3d(b0, b1)  # [*, 3]
    n2 = _cross_3d(b1, b2)  # [*, 3]

    # Normalize
    n1 = normalize_last_dim(n1)
    n2 = normalize_last_dim(n2)

    # Compute angle components
    cos_angle = _dot_last_dim(n1, n2)  # [*]
    n1_cross_n2 = _cross_3d(n1, n2)    # [*, 3]
    sin_angle_magnitude = sqrt.(sum(n1_cross_n2 .^ 2, dims=ndims(n1_cross_n2)))  # [*]
    sin_angle_magnitude = dropdims(sin_angle_magnitude, dims=ndims(sin_angle_magnitude))

    # Determine sign from cross product with b1
    sign_val = sign.(_dot_last_dim(n1_cross_n2, b1))  # [*]

    return atan.(sign_val .* sin_angle_magnitude, cos_angle)
end

"""
    bond_angle(a, b, c)

Computes the bond angle at point b for 3 points a, b, c.
The angle is between vectors (b→a) and (b→c).

Supports broadcasting - all inputs should have shape [*, 3].

Returns angle in range [0, π] with shape [*].
"""
function bond_angle(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}) where T
    # Vectors from b to a and c
    b0 = a .- b  # [*, 3]
    b1 = c .- b  # [*, 3]

    # Normalize
    b0 = normalize_last_dim(b0)
    b1 = normalize_last_dim(b1)

    # Compute angle using dot product and cross product
    cos_angle = _dot_last_dim(b0, b1)  # [*]
    cross = _cross_3d(b0, b1)          # [*, 3]
    sin_angle = sqrt.(sum(cross .^ 2, dims=ndims(cross)))  # [*]
    sin_angle = dropdims(sin_angle, dims=ndims(sin_angle))

    return atan.(sin_angle, cos_angle)
end

"""
    _cross_3d(a, b)

Compute cross product for 3D vectors along last dimension.
Input shapes: [*, 3], Output shape: [*, 3]
"""
function _cross_3d(a::AbstractArray{T}, b::AbstractArray{T}) where T
    # Get all dimensions except last
    dims = ntuple(i -> Colon(), ndims(a) - 1)

    # Extract components
    a1 = a[dims..., 1]
    a2 = a[dims..., 2]
    a3 = a[dims..., 3]
    b1 = b[dims..., 1]
    b2 = b[dims..., 2]
    b3 = b[dims..., 3]

    # Compute cross product components
    c1 = a2 .* b3 .- a3 .* b2
    c2 = a3 .* b1 .- a1 .* b3
    c3 = a1 .* b2 .- a2 .* b1

    # Stack along last dimension
    return cat(c1, c2, c3; dims=ndims(a))
end

"""
    _dot_last_dim(a, b)

Compute dot product along last dimension.
Input shapes: [*, 3], Output shape: [*]
"""
function _dot_last_dim(a::AbstractArray, b::AbstractArray)
    result = sum(a .* b, dims=ndims(a))
    return dropdims(result, dims=ndims(a))
end

# ============================================================================
# Backbone torsion angles (phi, psi, omega)
# ============================================================================

"""
    backbone_torsion_angles(coords_a37, mask; pdb_idx=nothing)

Compute backbone torsion angles (phi, psi, omega) from atom37 coordinates.

# Arguments
- `coords_a37`: Atom37 coordinates [3, 37, L, B]
- `mask`: Residue mask [L, B]
- `pdb_idx`: Optional residue indices [L, B]. If not provided, assumes contiguous indices.

# Returns
- `angles`: [3, L, B] tensor with phi, psi, omega angles for each residue
  - phi: C(i-1) - N(i) - CA(i) - C(i)
  - psi: N(i) - CA(i) - C(i) - N(i+1)
  - omega: CA(i-1) - C(i-1) - N(i) - CA(i)

Angles are in radians, range [-π, π]. Invalid angles (at chain breaks or termini) are 0.
"""
function backbone_torsion_angles(coords_a37::AbstractArray{T}, mask::AbstractArray;
                                  pdb_idx=nothing) where T
    # coords_a37: [3, 37, L, B]
    # Atom indices: N=1, CA=2, C=3 (Julia 1-indexed; Python uses 0, 1, 2)
    N_IDX = 1
    CA_IDX = 2
    C_IDX = 3

    L, B = size(mask)

    # Extract backbone atoms: [3, L, B]
    N = coords_a37[:, N_IDX, :, :]
    CA = coords_a37[:, CA_IDX, :, :]
    C = coords_a37[:, C_IDX, :, :]

    # Determine valid pairs (good_pair in Python)
    # Default: contiguous indices
    if isnothing(pdb_idx)
        # All pairs are good (contiguous residues)
        good_pair = trues(L-1, B)
    else
        # Chain break if index jump > 1
        idx_diff = pdb_idx[2:end, :] .- pdb_idx[1:end-1, :]
        good_pair = idx_diff .== 1
    end

    # Initialize angles - shape [3, L, B] with order [psi, omega, phi] to match Python
    angles = zeros(T, 3, L, B)

    # Compute angles using Python's convention:
    # All angles are computed for the BOND i→i+1 and stored at position i
    # psi[i] = dihedral(N[i], CA[i], C[i], N[i+1])
    # omega[i] = dihedral(CA[i], C[i], N[i+1], CA[i+1])
    # phi[i] = dihedral(C[i], N[i+1], CA[i+1], C[i+1])
    # Last position (L) gets zeros
    for b in 1:B
        for i in 1:(L-1)
            if !good_pair[i, b]
                continue
            end

            # psi at position i (angle 1)
            angles[1, i, b] = _scalar_dihedral(
                N[:, i, b], CA[:, i, b], C[:, i, b], N[:, i+1, b])

            # omega at position i (angle 2)
            angles[2, i, b] = _scalar_dihedral(
                CA[:, i, b], C[:, i, b], N[:, i+1, b], CA[:, i+1, b])

            # phi at position i (angle 3) - this is phi of residue i+1
            angles[3, i, b] = _scalar_dihedral(
                C[:, i, b], N[:, i+1, b], CA[:, i+1, b], C[:, i+1, b])
        end
    end
    # angles[:, L, :] stays zero (no i+1 residue)

    return angles  # [3, L, B] with order [psi, omega, phi]
end

"""
Helper for scalar dihedral angle computation.
"""
function _scalar_dihedral(a::AbstractVector{T}, b::AbstractVector{T},
                          c::AbstractVector{T}, d::AbstractVector{T}) where T
    b0 = b .- a
    b1 = c .- b
    b2 = d .- c

    n1 = cross(b0, b1)
    n2 = cross(b1, b2)

    n1_norm = norm(n1)
    n2_norm = norm(n2)

    if n1_norm < T(1e-8) || n2_norm < T(1e-8)
        return T(0)
    end

    n1 = n1 ./ n1_norm
    n2 = n2 ./ n2_norm

    cos_angle = dot(n1, n2)
    n1_cross_n2 = cross(n1, n2)
    sin_magnitude = norm(n1_cross_n2)
    sign_val = sign(dot(n1_cross_n2, b1))

    return atan(sign_val * sin_magnitude, cos_angle)
end

# ============================================================================
# Sidechain torsion angles (chi angles)
# ============================================================================

# Chi angle definitions: which atoms define each chi angle for each residue type
# Based on OpenFold/AF2 conventions
# Format: chi_atoms[restype][chi_idx] = [atom1, atom2, atom3, atom4]
# Atom indices are 1-based atom37 indices

# Chi angle definitions from OpenFold residue_constants.py
# Julia atom37 indices (1-indexed):
# N=1, CA=2, C=3, CB=4, O=5, CG=6, CG1=7, CG2=8, OG=9, OG1=10, SG=11,
# CD=12, CD1=13, CD2=14, ND1=15, ND2=16, OD1=17, OD2=18, SD=19, CE=20, CE1=21,
# CE2=22, CE3=23, NE=24, NE1=25, NE2=26, OE1=27, OE2=28, CH2=29, NH1=30, NH2=31,
# OH=32, CZ=33, CZ2=34, CZ3=35, NZ=36, OXT=37
const CHI_ATOM_INDICES = Dict{Int, Vector{NTuple{4, Int}}}(
    # Each entry is (restype_idx, [(a1,a2,a3,a4) for chi1, chi2, chi3, chi4])
    # Restype indices are 1-based matching Julia RESTYPES: A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V
    # ALA (1) - no chi angles
    1 => NTuple{4,Int}[],
    # ARG (2) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD, chi3: CB-CG-CD-NE, chi4: CG-CD-NE-CZ
    2 => [(1, 2, 4, 6), (2, 4, 6, 12), (4, 6, 12, 24), (6, 12, 24, 33)],
    # ASN (3) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-OD1
    3 => [(1, 2, 4, 6), (2, 4, 6, 17)],
    # ASP (4) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-OD1
    4 => [(1, 2, 4, 6), (2, 4, 6, 17)],
    # CYS (5) - chi1: N-CA-CB-SG
    5 => [(1, 2, 4, 11)],
    # GLN (6) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD, chi3: CB-CG-CD-OE1
    6 => [(1, 2, 4, 6), (2, 4, 6, 12), (4, 6, 12, 27)],
    # GLU (7) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD, chi3: CB-CG-CD-OE1
    7 => [(1, 2, 4, 6), (2, 4, 6, 12), (4, 6, 12, 27)],
    # GLY (8) - no chi angles
    8 => NTuple{4,Int}[],
    # HIS (9) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-ND1
    9 => [(1, 2, 4, 6), (2, 4, 6, 15)],
    # ILE (10) - chi1: N-CA-CB-CG1, chi2: CA-CB-CG1-CD1
    10 => [(1, 2, 4, 7), (2, 4, 7, 13)],
    # LEU (11) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD1
    11 => [(1, 2, 4, 6), (2, 4, 6, 13)],
    # LYS (12) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD, chi3: CB-CG-CD-CE, chi4: CG-CD-CE-NZ
    12 => [(1, 2, 4, 6), (2, 4, 6, 12), (4, 6, 12, 20), (6, 12, 20, 36)],
    # MET (13) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-SD, chi3: CB-CG-SD-CE
    13 => [(1, 2, 4, 6), (2, 4, 6, 19), (4, 6, 19, 20)],
    # PHE (14) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD1
    14 => [(1, 2, 4, 6), (2, 4, 6, 13)],
    # PRO (15) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD
    15 => [(1, 2, 4, 6), (2, 4, 6, 12)],
    # SER (16) - chi1: N-CA-CB-OG
    16 => [(1, 2, 4, 9)],
    # THR (17) - chi1: N-CA-CB-OG1
    17 => [(1, 2, 4, 10)],
    # TRP (18) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD1
    18 => [(1, 2, 4, 6), (2, 4, 6, 13)],
    # TYR (19) - chi1: N-CA-CB-CG, chi2: CA-CB-CG-CD1
    19 => [(1, 2, 4, 6), (2, 4, 6, 13)],
    # VAL (20) - chi1: N-CA-CB-CG1
    20 => [(1, 2, 4, 7)],
)

const MAX_CHI_ANGLES = 4  # Maximum number of chi angles (OpenFold ignores chi5 for ARG)

"""
    sidechain_torsion_angles(coords_a37, aatype, mask, coord_mask)

Compute sidechain chi angles from atom37 coordinates.

# Arguments
- `coords_a37`: Atom37 coordinates [3, 37, L, B]
- `aatype`: Amino acid types [L, B], 1-indexed (1-20)
- `mask`: Residue mask [L, B]
- `coord_mask`: Atom presence mask [37, L, B] (optional)

# Returns
- `chi_angles`: [MAX_CHI_ANGLES, L, B] tensor with chi angles
- `chi_mask`: [MAX_CHI_ANGLES, L, B] boolean mask indicating valid angles

Angles are in radians, range [-π, π].
"""
function sidechain_torsion_angles(coords_a37::AbstractArray{T}, aatype::AbstractArray{<:Integer},
                                   mask::AbstractArray; coord_mask=nothing) where T
    L, B = size(mask)

    chi_angles = zeros(T, MAX_CHI_ANGLES, L, B)
    chi_mask = falses(MAX_CHI_ANGLES, L, B)

    for b in 1:B
        for i in 1:L
            if mask[i, b] < 0.5
                continue
            end

            aa = clamp(Int(aatype[i, b]), 1, 20)
            chi_defs = get(CHI_ATOM_INDICES, aa, NTuple{4,Int}[])

            for (chi_idx, (a1, a2, a3, a4)) in enumerate(chi_defs)
                if chi_idx > MAX_CHI_ANGLES
                    break
                end

                # Check if all atoms are present using coord_mask
                if !isnothing(coord_mask)
                    atoms_present = coord_mask[a1, i, b] > 0.5 &&
                                    coord_mask[a2, i, b] > 0.5 &&
                                    coord_mask[a3, i, b] > 0.5 &&
                                    coord_mask[a4, i, b] > 0.5
                else
                    # Fallback: check coordinate norms
                    p1 = coords_a37[:, a1, i, b]
                    p2 = coords_a37[:, a2, i, b]
                    p3 = coords_a37[:, a3, i, b]
                    p4 = coords_a37[:, a4, i, b]
                    atoms_present = norm(p1) > T(1e-6) && norm(p2) > T(1e-6) &&
                                    norm(p3) > T(1e-6) && norm(p4) > T(1e-6)
                end

                if atoms_present
                    # Get atom coordinates
                    p1 = coords_a37[:, a1, i, b]
                    p2 = coords_a37[:, a2, i, b]
                    p3 = coords_a37[:, a3, i, b]
                    p4 = coords_a37[:, a4, i, b]

                    chi_angles[chi_idx, i, b] = _scalar_dihedral(p1, p2, p3, p4)
                    chi_mask[chi_idx, i, b] = true
                end
            end
        end
    end

    return chi_angles, chi_mask
end

# ============================================================================
# Binning utilities
# ============================================================================

"""
    bin_angles(angles, n_bins; min_val=-π, max_val=π)

Bin angles into one-hot vectors.

# Arguments
- `angles`: Angle tensor of any shape [*]
- `n_bins`: Number of bins (produces n_bins+1 one-hot dimensions)
- `min_val`: Left edge of first bin
- `max_val`: Right edge of last bin

# Returns
- One-hot tensor of shape [n_bins+1, *]
"""
function bin_angles(angles::AbstractArray{T}, n_bins::Int;
                    min_val::Real=T(-π), max_val::Real=T(π)) where T
    # Create bin edges
    edges = range(T(min_val), T(max_val), length=n_bins)

    # Compute bin indices
    shape = size(angles)
    angles_flat = vec(angles)
    n = length(angles_flat)

    # One-hot encoding
    onehot = zeros(T, n_bins + 1, n)

    for i in 1:n
        a = angles_flat[i]
        # Find bin index
        bin_idx = 1
        for (j, edge) in enumerate(edges)
            if a >= edge
                bin_idx = j + 1
            end
        end
        bin_idx = clamp(bin_idx, 1, n_bins + 1)
        onehot[bin_idx, i] = one(T)
    end

    # Reshape back to original shape with bin dimension first
    return reshape(onehot, n_bins + 1, shape...)
end
