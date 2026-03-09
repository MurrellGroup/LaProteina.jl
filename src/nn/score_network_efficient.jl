# Efficient GPU-native ScoreNetwork forward pass
# Minimizes CPU-GPU transfer and avoids wasteful memory allocation

using Flux
using NNlib
using CUDA

# Access Zygote through Flux
const Zygote = Flux.Zygote

# ============================================================================
# Efficient feature computation (GPU-friendly, wrapped in @ignore)
# ============================================================================

"""
    compute_time_embedding_gpu(t::AbstractVector{T}, dim::Int) where T

GPU-friendly time embedding computation.
Returns [dim, B] (not expanded to L).
"""
function compute_time_embedding_gpu(t::AbstractVector{T}, dim::Int; max_positions::Int=2000) where T
    B = length(t)
    half_dim = dim ÷ 2

    # Compute on same device as t
    t_scaled = t .* T(max_positions)

    # Create frequency scale on device
    log_max = T(log(max_positions))
    k = T.(0:(half_dim-1))
    if !(t isa Array)
        k = Flux.gpu(k)
    end
    emb_scale = exp.(-log_max .* k ./ T(max(half_dim - 1, 1)))  # [half_dim]

    # Outer product: [half_dim] × [B] → [half_dim, B]
    emb = emb_scale .* t_scaled'

    # Sin and cos
    result = vcat(sin.(emb), cos.(emb))  # [dim, B]

    return result
end

"""
    compute_pairwise_distances_gpu(coords::AbstractArray{T,3}) where T

Compute pairwise Euclidean distances.
coords: [3, L, B] → distances: [L, L, B]
"""
function compute_pairwise_distances_gpu(coords::AbstractArray{T,3}) where T
    # coords: [3, L, B]
    L = size(coords, 2)
    B = size(coords, 3)

    # Reshape for broadcasting: [3, L, 1, B] - [3, 1, L, B]
    c1 = reshape(coords, 3, L, 1, B)
    c2 = reshape(coords, 3, 1, L, B)

    # Squared differences and sum
    diff_sq = sum((c1 .- c2).^2, dims=1)  # [1, L, L, B]

    # Sqrt and remove leading dim
    return dropdims(sqrt.(diff_sq), dims=1)  # [L, L, B]
end

"""
    compute_distance_bin_indices_gpu(dists, min_dist, max_dist, n_bins)

Compute bin indices for distances (GPU-friendly).
Matches original bin_values logic: creates n_bins-1 limits from min_dist to max_dist,
bin index = 1 + count(dist > limit for each limit).
Returns float indices in 1:n_bins.
"""
function compute_distance_bin_indices_gpu(dists::AbstractArray{T}, min_dist::Real, max_dist::Real, n_bins::Int) where T
    # Create bin limits matching original: range(min_dist, max_dist, length=n_bins-1)
    n_limits = n_bins - 1
    limits = T.(range(T(min_dist), T(max_dist), length=n_limits))

    # Move limits to same device as dists
    if !(dists isa Array)
        limits = Flux.gpu(limits)
    end

    # Reshape for broadcasting: limits [n_limits, 1, 1, 1], dists [1, L, L, B]
    limits_reshaped = reshape(limits, n_limits, ntuple(_ -> 1, ndims(dists))...)
    dists_reshaped = reshape(dists, 1, size(dists)...)

    # Compare: comparisons[i, ...] = 1 if dist > limits[i]
    comparisons = T.(dists_reshaped .> limits_reshaped)  # [n_limits, L, L, B]

    # Sum along first dim to get bin index - 1, then add 1
    indices = dropdims(sum(comparisons; dims=1); dims=1) .+ one(T)  # [L, L, B]

    return indices
end

"""
    compute_seq_sep_indices_gpu(L::Int, B::Int, max_sep::Int, device_array)

Compute sequence separation indices (GPU-friendly).
Returns [L, L] indices in 1:(2*max_sep+1).
"""
function compute_seq_sep_indices_gpu(L::Int, max_sep::Int, ::Type{T}, device_array) where T
    # Create position indices on device
    positions = T.(1:L)
    if !(device_array isa Array)
        positions = Flux.gpu(positions)
    end

    # Compute i - j for all pairs
    rel_sep = positions .- positions'  # [L, L]

    # Clamp and shift to 1-indexed
    clamped = clamp.(rel_sep, T(-max_sep), T(max_sep))
    indices = clamped .+ T(max_sep + 1)  # [L, L] in range [1, 2*max_sep+1]

    return indices
end

"""
    indices_to_onehot_gpu(indices, n_classes)

Convert indices to one-hot encoding on GPU.
indices: [...] with values in 1:n_classes
Returns: [n_classes, ...] one-hot tensor
"""
function indices_to_onehot_gpu(indices::AbstractArray{T}, n_classes::Int) where T
    # Create class indices [n_classes]
    classes = T.(1:n_classes)
    if !(indices isa Array)
        classes = Flux.gpu(classes)
    end

    # Reshape for broadcasting: classes [n_classes, 1, 1, ...], indices [1, size(indices)...]
    classes_shape = (n_classes, ntuple(_ -> 1, ndims(indices))...)
    classes_reshaped = reshape(classes, classes_shape)
    indices_reshaped = reshape(indices, 1, size(indices)...)

    # One-hot via broadcasting comparison
    return T.(classes_reshaped .== indices_reshaped)
end

# ============================================================================
# Efficient ScoreNetwork batch structure
# ============================================================================

"""
    EfficientScoreNetworkBatch

Minimal batch structure for efficient GPU computation.
All tensors should already be on GPU.
"""
struct EfficientScoreNetworkBatch{A3<:AbstractArray, A2<:AbstractArray, A1<:AbstractVector}
    x_t_ca::A3           # [3, L, B] noisy CA coords
    x_t_ll::A3           # [latent_dim, L, B] noisy local latents
    t_ca::A1             # [B] time for CA
    t_ll::A1             # [B] time for local latents
    mask::A2             # [L, B]
    # Optional self-conditioning (can be nothing)
    x_sc_ca::Union{A3, Nothing}
    x_sc_ll::Union{A3, Nothing}
end

function EfficientScoreNetworkBatch(x_t_ca::A3, x_t_ll::A3, t_ca::A1, t_ll::A1, mask::A2;
                                     x_sc_ca::Union{A3, Nothing}=nothing,
                                     x_sc_ll::Union{A3, Nothing}=nothing) where {A3<:AbstractArray, A2<:AbstractArray, A1<:AbstractVector}
    EfficientScoreNetworkBatch{A3, A2, A1}(x_t_ca, x_t_ll, t_ca, t_ll, mask, x_sc_ca, x_sc_ll)
end

# ============================================================================
# Efficient forward pass
# ============================================================================

"""
    forward_efficient(model::ScoreNetwork, batch::EfficientScoreNetworkBatch)

Efficient GPU-native forward pass.
Features computed with @ignore, projections get gradients.
"""
function forward_efficient(model::ScoreNetwork, batch::EfficientScoreNetworkBatch)
    x_t_ca = batch.x_t_ca      # [3, L, B]
    x_t_ll = batch.x_t_ll      # [latent_dim, L, B]
    t_ca = batch.t_ca          # [B]
    t_ll = batch.t_ll          # [B]
    mask = batch.mask          # [L, B]

    L, B = size(mask)
    latent_dim = size(x_t_ll, 1)
    mask_seq = reshape(mask, 1, L, B)  # [1, L, B]
    mask_pair = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)  # [L, L, B]
    mask_pair_exp = reshape(mask_pair, 1, L, L, B)  # [1, L, L, B]

    # ========================================
    # Feature computation (no gradients needed)
    # ========================================
    T = eltype(x_t_ca)
    is_gpu = !(x_t_ca isa Array)

    seq_raw, cond_raw, pair_raw, pair_cond_raw = Zygote.@ignore begin
        # Helper to create zeros on correct device
        make_zeros(dims...) = is_gpu ? CUDA.zeros(T, dims...) : zeros(T, dims...)

        # --- Sequence features [46, L, B] ---
        # x_t CA and latents
        seq_parts = Any[x_t_ca, x_t_ll]

        # Self-conditioning (zeros if not provided)
        if isnothing(batch.x_sc_ca)
            push!(seq_parts, make_zeros(3, L, B))
            push!(seq_parts, make_zeros(latent_dim, L, B))
        else
            push!(seq_parts, batch.x_sc_ca)
            push!(seq_parts, batch.x_sc_ll)
        end

        # Optional CA coords (zeros for flow matching)
        push!(seq_parts, make_zeros(3, L, B))

        # Optional residue type (zeros for flow matching - no sequence info)
        push!(seq_parts, make_zeros(20, L, B))

        # Cropped flag (zeros)
        push!(seq_parts, make_zeros(1, L, B))

        seq_raw = cat(seq_parts..., dims=1)  # [46, L, B]

        # --- Conditioning features [512, B] (NOT [512, L, B]) ---
        t_emb_ca = compute_time_embedding_gpu(t_ca, 256)    # [256, B]
        t_emb_ll = compute_time_embedding_gpu(t_ll, 256)    # [256, B]
        cond_raw = cat(t_emb_ca, t_emb_ll, dims=1)          # [512, B]

        # --- Pair features [217, L, L, B] ---
        # Sequence separation: [127, L, L, B]
        seq_sep_indices = compute_seq_sep_indices_gpu(L, 63, T, x_t_ca)  # [L, L]
        seq_sep_onehot = indices_to_onehot_gpu(seq_sep_indices, 127)     # [127, L, L]
        # Expand to batch: [127, L, L, 1] → [127, L, L, B]
        seq_sep = repeat(seq_sep_onehot, 1, 1, 1, B)

        # Distance bins for x_t CA: [30, L, L, B]
        dists_xt = compute_pairwise_distances_gpu(x_t_ca)
        dist_indices_xt = compute_distance_bin_indices_gpu(dists_xt, 0.1f0, 3.0f0, 30)
        dist_onehot_xt = indices_to_onehot_gpu(dist_indices_xt, 30)

        # Distance bins for x_sc CA (zeros if no self-conditioning): [30, L, L, B]
        if isnothing(batch.x_sc_ca)
            dist_onehot_sc = make_zeros(30, L, L, B)
        else
            dists_sc = compute_pairwise_distances_gpu(batch.x_sc_ca)
            dist_indices_sc = compute_distance_bin_indices_gpu(dists_sc, 0.1f0, 3.0f0, 30)
            dist_onehot_sc = indices_to_onehot_gpu(dist_indices_sc, 30)
        end

        # Optional CA distances (zeros for flow matching): [30, L, L, B]
        dist_optional = make_zeros(30, L, L, B)

        pair_raw = cat(seq_sep, dist_onehot_xt, dist_onehot_sc, dist_optional, dims=1)  # [217, L, L, B]

        # --- Pair conditioning [512, B] (NOT [512, L, L, B]) ---
        # Same time embeddings as cond, but will be broadcast in AdaLN
        pair_cond_raw = cond_raw  # [512, B]

        (seq_raw, cond_raw, pair_raw, pair_cond_raw)
    end

    # ========================================
    # Projections and forward pass (with gradients)
    # ========================================

    # Project sequence features: [46, L, B] → [768, L, B]
    seqs = model.init_repr_factory.projection(seq_raw)
    if model.init_repr_factory.use_ln && !isnothing(model.init_repr_factory.ln)
        seqs = model.init_repr_factory.ln(seqs)
    end
    seqs = seqs .* mask_seq

    # Project conditioning: [512, B] → broadcast to [512, L, B] → project to [256, L, B]
    # Must broadcast BEFORE projection to match original FeatureFactory behavior
    cond_raw_broadcast = repeat(reshape(cond_raw, size(cond_raw, 1), 1, B), 1, L, 1)  # [512, L, B]
    cond = model.cond_factory.projection(cond_raw_broadcast)  # [256, L, B]
    if model.cond_factory.use_ln && !isnothing(model.cond_factory.ln)
        cond = model.cond_factory.ln(cond)
    end
    cond = cond .* mask_seq

    # Apply conditioning transitions
    cond = model.transition_c_1(cond, mask)
    cond = model.transition_c_2(cond, mask)

    # Project pair features: [217, L, L, B] → [256, L, L, B]
    pair_rep = model.pair_rep_builder.init_repr_factory.projection(pair_raw)
    if model.pair_rep_builder.init_repr_factory.use_ln && !isnothing(model.pair_rep_builder.init_repr_factory.ln)
        pair_rep = model.pair_rep_builder.init_repr_factory.ln(pair_rep)
    end
    pair_rep = pair_rep .* mask_pair_exp

    # Pair conditioning with AdaLN
    if !isnothing(model.pair_rep_builder.adaln) && !isnothing(model.pair_rep_builder.cond_factory)
        # Broadcast [512, B] → [512, L, L, B] BEFORE projection (for exact parity)
        pair_cond_broadcast = repeat(reshape(pair_cond_raw, size(pair_cond_raw, 1), 1, 1, B), 1, L, L, 1)
        pair_cond = model.pair_rep_builder.cond_factory.projection(pair_cond_broadcast)
        if model.pair_rep_builder.cond_factory.use_ln && !isnothing(model.pair_rep_builder.cond_factory.ln)
            pair_cond = model.pair_rep_builder.cond_factory.ln(pair_cond)
        end
        pair_cond = pair_cond .* mask_pair_exp
        pair_rep = model.pair_rep_builder.adaln(pair_rep, pair_cond, mask_pair)
    end

    # Transformer layers
    for i in 1:model.n_layers
        seqs = model.transformer_layers[i](seqs, pair_rep, cond, mask)

        if model.update_pair_repr && i < model.n_layers
            if !isnothing(model.pair_update_layers[i])
                pair_rep = model.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Output projections
    local_latents_out = model.local_latents_proj(seqs) .* mask_seq
    ca_out = model.ca_proj(seqs) .* mask_seq

    out_key = model.output_param
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out)
    )
end

# ============================================================================
# Helper to create batch from Dict format
# ============================================================================

"""
    to_efficient_batch(batch::Dict)

Convert Dict-format batch to EfficientScoreNetworkBatch.
"""
function to_efficient_batch(batch::Dict)
    x_t = batch[:x_t]
    t = batch[:t]
    mask = batch[:mask]

    x_t_ca = x_t[:bb_ca]
    x_t_ll = x_t[:local_latents]
    t_ca = isa(t, Dict) ? t[:bb_ca] : t
    t_ll = isa(t, Dict) ? t[:local_latents] : t

    x_sc_ca = nothing
    x_sc_ll = nothing
    if haskey(batch, :x_sc)
        x_sc = batch[:x_sc]
        x_sc_ca = get(x_sc, :bb_ca, nothing)
        x_sc_ll = get(x_sc, :local_latents, nothing)
    end

    return EfficientScoreNetworkBatch(x_t_ca, x_t_ll, t_ca, t_ll, mask; x_sc_ca=x_sc_ca, x_sc_ll=x_sc_ll)
end
