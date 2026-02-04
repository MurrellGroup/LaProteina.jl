# State construction utilities for Branching Flows
# Converts LaProteina data to BranchingFlows state format

using BranchingFlows: BranchingState
using ForwardBackward: ContinuousState, DiscreteState, MaskedState
using Flowfusion: element

"""
    protein_to_branching_state(protein::NamedTuple; sample_z::Bool=true)

Convert a precomputed protein (NamedTuple with ca_coords, z_mean, z_log_scale, mask)
to an unbatched BranchingState suitable for `branching_bridge`.

The state includes:
1. CA coordinates as ContinuousState
2. Local latents (sampled from VAE) as ContinuousState
3. Index tracking state as DiscreteState (for self-conditioning with splits)

# Arguments
- `protein`: NamedTuple with fields :ca_coords, :z_mean, :z_log_scale, :mask
- `sample_z`: If true, sample z from VAE distribution. If false, use z_mean directly.

# Returns
BranchingState (unbatched - groupings/masks are vectors [L])
"""
function protein_to_branching_state(protein::NamedTuple; sample_z::Bool=true)
    ca_coords = protein.ca_coords       # [3, L]
    z_mean = protein.z_mean             # [latent_dim, L]
    z_log_scale = protein.z_log_scale   # [latent_dim, L]
    mask = protein.mask                  # [L]

    L = length(mask)
    latent_dim = size(z_mean, 1)

    # Sample z from VAE distribution (reparameterization trick)
    if sample_z
        z_latent = z_mean .+ randn(Float32, size(z_mean)) .* exp.(z_log_scale)
    else
        z_latent = copy(z_mean)
    end

    # Create mask as boolean (BranchingFlows expects Bool vectors for unbatched)
    valid_mask = Vector{Bool}(mask .> 0.5)

    # CA state: [3, L] -> MaskedState(ContinuousState([3, 1, L]))
    # Flowfusion format: [D, 1, L] for unbatched (the 1 is a "batch" dim of size 1)
    ca_state = MaskedState(
        ContinuousState(reshape(ca_coords, 3, 1, L)),
        valid_mask,
        valid_mask
    )

    # Latent state: [latent_dim, L] -> MaskedState(ContinuousState([D, 1, L]))
    latent_state = MaskedState(
        ContinuousState(reshape(z_latent, latent_dim, 1, L)),
        valid_mask,
        valid_mask
    )

    # Index tracking state: [L] integers 1:L
    # This tracks which original position each element came from after splits/deletions
    # K=0 means no one-hot categories, just raw integer indices
    index_state = MaskedState(
        DiscreteState(0, collect(1:L)),
        valid_mask,
        valid_mask
    )

    # Group IDs: all same group for single-chain proteins
    # Shape: [L] vector for unbatched
    groupings = ones(Int, L)

    # Create BranchingState with all masks as vectors
    return BranchingState(
        (ca_state, latent_state, index_state),
        groupings;
        flowmask = valid_mask,     # All valid positions flow
        branchmask = valid_mask    # All valid positions can branch
    )
end

"""
    X0_sampler_laproteina(latent_dim::Int)

Create an X0 sampler function for branching_bridge.
Returns a single-element noise state tuple.

The sampler is called by forest_bridge for each root node in the forest.
It should return a tuple of states (not MaskedStates) - branching_bridge
handles the masking.

# Arguments
- `latent_dim`: Dimension of local latents (default 8)

# Returns
Function that takes a root FlowNode and returns single-element state tuple
"""
function X0_sampler_laproteina(latent_dim::Int=8)
    function sampler(root)
        # Single-element states (branching_bridge/forest_bridge will handle batching)
        # Format: [D, 1, 1] for single element with "batch" dim of 1
        ca_state = ContinuousState(randn(Float32, 3, 1, 1))
        latent_state = ContinuousState(randn(Float32, latent_dim, 1, 1))
        # Index state: just tracks the index, K=0 means raw integers
        index_state = DiscreteState(0, [1])

        return (ca_state, latent_state, index_state)
    end
    return sampler
end

"""
    proteins_to_X1_states(proteins::Vector, indices::Vector{Int})

Convert a batch of proteins to X1 states for branching_bridge.
Returns a vector of unbatched BranchingStates.

# Arguments
- `proteins`: Vector of precomputed proteins (NamedTuples)
- `indices`: Indices of proteins to use

# Returns
Vector of BranchingStates (unbatched)
"""
function proteins_to_X1_states(proteins::Vector, indices::Vector{Int})
    return [protein_to_branching_state(proteins[i]) for i in indices]
end

"""
    extract_state_tensors(branching_state::BranchingState)

Extract raw tensors from a BranchingState for model input.

# Returns
NamedTuple with:
- ca: [3, L, B] CA coordinates
- latents: [latent_dim, L, B] local latents
- indices: [L, B] index tracking
- mask: [L, B] padding mask
"""
function extract_state_tensors(branching_state::BranchingState)
    ca_state = branching_state.state[1]
    latent_state = branching_state.state[2]
    index_state = branching_state.state[3]

    # Extract tensors from MaskedState -> ContinuousState/DiscreteState
    ca = tensor(ca_state.S)  # From ForwardBackward
    latents = tensor(latent_state.S)
    indices = index_state.S.state  # DiscreteState stores raw values

    # Get mask from BranchingState
    mask = Float32.(branching_state.padmask)

    return (ca=ca, latents=latents, indices=indices, mask=mask)
end

"""
    expand_by_indices(arr::AbstractArray{T,3}, indices) where T

Expand array by indexing into the second dimension using indices.
Used for self-conditioning expansion when splits occur.

# Arguments
- `arr`: [D, L_old, B] array of predictions
- `indices`: [L_new, B] or [L_new] indices mapping new -> old positions

# Returns
[D, L_new, B] expanded array
"""
function expand_by_indices(arr::AbstractArray{T,3}, indices::AbstractMatrix{<:Integer}) where T
    D, L_old, B = size(arr)
    L_new = size(indices, 1)
    expanded = similar(arr, D, L_new, B)
    for b in 1:B
        for i in 1:L_new
            src_idx = indices[i, b]
            expanded[:, i, b] = arr[:, src_idx, b]
        end
    end
    return expanded
end

function expand_by_indices(arr::AbstractArray{T,3}, indices::AbstractVector{<:Integer}) where T
    D, L_old, B = size(arr)
    L_new = length(indices)
    expanded = similar(arr, D, L_new, B)
    for b in 1:B
        for i in 1:L_new
            src_idx = indices[i]
            expanded[:, i, b] = arr[:, src_idx, b]
        end
    end
    return expanded
end

# Differentiable version using NNlib or similar
function expand_by_indices_diff(arr::AbstractArray{T,3}, indices::AbstractMatrix{<:Integer}) where T
    # For now, use the non-diff version and rely on Zygote's scalar indexing
    # In production, this could be optimized with gather operations
    D, L_old, B = size(arr)
    L_new = size(indices, 1)

    # Use batched index select
    expanded = similar(arr, D, L_new, B)
    for b in 1:B
        expanded[:, :, b] = arr[:, indices[:, b], b]
    end
    return expanded
end
