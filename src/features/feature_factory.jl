# Feature factory for extracting and combining features
# Port of feature_factory.py

using Flux
using Functors
using Adapt

"""
Abstract type for feature extractors.
"""
abstract type Feature end

"""
    get_dim(f::Feature) -> Int

Get the output dimension of a feature.
"""
get_dim(f::Feature) = f.dim

# ============================================================================
# Sequence Features
# ============================================================================

"""
    ZeroFeature(dim::Int, mode::Symbol=:seq)

Returns zero tensor of specified dimension. Useful as placeholder.
"""
struct ZeroFeature <: Feature
    dim::Int
    mode::Symbol  # :seq or :pair
end

ZeroFeature(dim::Int) = ZeroFeature(dim, :seq)

function (f::ZeroFeature)(batch::Dict, L::Int, B::Int)
    T = Float32
    if f.mode == :seq
        return zeros(T, f.dim, L, B)
    else  # :pair
        return zeros(T, f.dim, L, L, B)
    end
end

"""
    TimeFeature(dim::Int, data_mode::Symbol)

Sinusoidal time embedding feature.
Extracts time from batch[:t][data_mode] and creates embedding.
"""
struct TimeFeature <: Feature
    dim::Int
    data_mode::Symbol  # :bb_ca or :local_latents
end

function (f::TimeFeature)(batch::Dict, L::Int, B::Int)
    # Get time for this data mode
    if haskey(batch, :t)
        t_dict = batch[:t]
        if isa(t_dict, Dict) && haskey(t_dict, f.data_mode)
            t = t_dict[f.data_mode]
        elseif isa(t_dict, AbstractVector)
            t = t_dict  # Same time for all modes
        else
            t = zeros(Float32, B)
        end
    else
        t = zeros(Float32, B)
    end
    t_emb = get_time_embedding(t, f.dim)  # [dim, B]
    # Broadcast to sequence length
    return broadcast_time_embedding(t_emb, L)  # [dim, L, B]
end

"""
    TimePairFeature(dim::Int, data_mode::Symbol)

Time embedding for pair representation.
"""
struct TimePairFeature <: Feature
    dim::Int
    data_mode::Symbol
end

function (f::TimePairFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :t)
        t_dict = batch[:t]
        if isa(t_dict, Dict) && haskey(t_dict, f.data_mode)
            t = t_dict[f.data_mode]
        elseif isa(t_dict, AbstractVector)
            t = t_dict
        else
            t = zeros(Float32, B)
        end
    else
        t = zeros(Float32, B)
    end
    t_emb = get_time_embedding(t, f.dim)  # [dim, B]
    # Broadcast to pair: [dim, B] -> [dim, L, L, B]
    t_expanded = reshape(t_emb, f.dim, 1, 1, B)
    return repeat(t_expanded, 1, L, L, 1)
end

"""
    PositionFeature(dim::Int; max_len::Int=2056)

Sinusoidal position embedding feature.
"""
struct PositionFeature <: Feature
    dim::Int
    max_len::Int
end

PositionFeature(dim::Int) = PositionFeature(dim, 2056)

function (f::PositionFeature)(batch::Dict, L::Int, B::Int)
    # Use pdb_idx if available, otherwise 1:L
    if haskey(batch, :pdb_idx)
        indices = batch[:pdb_idx]  # [L, B]
    else
        indices = repeat(1:L, 1, B)  # [L, B]
    end
    return get_index_embedding(indices, f.dim; max_len=f.max_len)  # [dim, L, B]
end

"""
    XtBBCAFeature()

x_t backbone CA coordinates from flow matching.
Extracts batch[:x_t][:bb_ca].
"""
struct XtBBCAFeature <: Feature
    dim::Int
end

XtBBCAFeature() = XtBBCAFeature(3)

function (f::XtBBCAFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :x_t) && haskey(batch[:x_t], :bb_ca)
        return batch[:x_t][:bb_ca]  # [3, L, B]
    else
        return zeros(Float32, 3, L, B)
    end
end

"""
    XtLocalLatentsFeature(latent_dim::Int)

x_t local latents from flow matching.
"""
struct XtLocalLatentsFeature <: Feature
    dim::Int
end

function (f::XtLocalLatentsFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :x_t) && haskey(batch[:x_t], :local_latents)
        return batch[:x_t][:local_latents]  # [latent_dim, L, B]
    else
        return zeros(Float32, f.dim, L, B)
    end
end

"""
    XscBBCAFeature()

Self-conditioning backbone CA coordinates.
Falls back to zeros if not present.
"""
struct XscBBCAFeature <: Feature
    dim::Int
    mode_key::Symbol
end

XscBBCAFeature() = XscBBCAFeature(3, :x_sc)

function (f::XscBBCAFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, f.mode_key) && haskey(batch[f.mode_key], :bb_ca)
        return batch[f.mode_key][:bb_ca]  # [3, L, B]
    else
        return zeros(Float32, 3, L, B)
    end
end

"""
    XscLocalLatentsFeature(latent_dim::Int)

Self-conditioning local latents.
"""
struct XscLocalLatentsFeature <: Feature
    dim::Int
    mode_key::Symbol
end

XscLocalLatentsFeature(dim::Int) = XscLocalLatentsFeature(dim, :x_sc)

function (f::XscLocalLatentsFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, f.mode_key) && haskey(batch[f.mode_key], :local_latents)
        return batch[f.mode_key][:local_latents]  # [dim, L, B]
    else
        return zeros(Float32, f.dim, L, B)
    end
end

"""
    OptionalCACoorsFeature()

Optional CA coordinates in nm (used for conditional generation).
Returns zeros if use_ca_coors_nm_feature is not set in batch.
"""
struct OptionalCACoorsFeature <: Feature
    dim::Int
end

OptionalCACoorsFeature() = OptionalCACoorsFeature(3)

function (f::OptionalCACoorsFeature)(batch::Dict, L::Int, B::Int)
    if get(batch, :use_ca_coors_nm_feature, false)
        if haskey(batch, :ca_coors_nm)
            return batch[:ca_coors_nm]
        elseif haskey(batch, :ca_coors)
            return batch[:ca_coors]
        end
    end
    return zeros(Float32, 3, L, B)
end

"""
    OptionalResTypeFeature()

Optional residue type one-hot encoding.
Returns zeros if use_residue_type_feature is not set.
"""
struct OptionalResTypeFeature <: Feature
    dim::Int
end

OptionalResTypeFeature() = OptionalResTypeFeature(20)  # 20 AA types (matches Python)

function (f::OptionalResTypeFeature)(batch::Dict, L::Int, B::Int)
    if get(batch, :use_residue_type_feature, false) && haskey(batch, :residue_type)
        rtype = batch[:residue_type]  # [L, B]
        # One-hot encode (20 classes)
        onehot = zeros(Float32, f.dim, L, B)
        for b in 1:B, l in 1:L
            idx = clamp(rtype[l, b], 1, f.dim)
            onehot[idx, l, b] = 1.0f0
        end
        return onehot
    end
    return zeros(Float32, f.dim, L, B)
end

"""
    LatentFeature(dim::Int)

Pass through latent z directly.
"""
struct LatentFeature <: Feature
    dim::Int
end

function (f::LatentFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :z_latent)
        return batch[:z_latent]  # Should be [dim, L, B]
    else
        return zeros(Float32, f.dim, L, B)
    end
end

"""
    CroppedFlagFeature()

Feature of shape [1, L, B] indicating if protein is cropped.
Returns 1s if cropped flag is set, 0s otherwise.
"""
struct CroppedFlagFeature <: Feature
    dim::Int
end

CroppedFlagFeature() = CroppedFlagFeature(1)

function (f::CroppedFlagFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :cropped)
        cropped = batch[:cropped]  # [B] boolean or float
        # Broadcast to [1, L, B]
        ones_tensor = ones(Float32, f.dim, L, B)
        cropped_expanded = reshape(Float32.(cropped), 1, 1, B)
        return ones_tensor .* cropped_expanded
    else
        return zeros(Float32, f.dim, L, B)
    end
end

"""
    CACoordFeature()

CA coordinate feature.
"""
struct CACoordFeature <: Feature
    dim::Int  # Should be 3
end

CACoordFeature() = CACoordFeature(3)

function (f::CACoordFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :ca_coors)
        return batch[:ca_coors]  # [3, L, B]
    elseif haskey(batch, :ca_coors_nm)
        return batch[:ca_coors_nm]  # [3, L, B]
    elseif haskey(batch, :x_t) && haskey(batch[:x_t], :bb_ca)
        return batch[:x_t][:bb_ca]  # [3, L, B]
    elseif haskey(batch, :coords)
        # Extract CA from atom37: coords is [3, 37, L, B], CA is index 2
        return batch[:coords][:, CA_INDEX, :, :]  # [3, L, B]
    else
        return zeros(Float32, 3, L, B)
    end
end

# ============================================================================
# Pair Features
# ============================================================================

"""
    DistanceBinFeature(dim::Int; min_dist::Real=0.0, max_dist::Real=20.0)

Binned pairwise CA distance feature for pair representation.
"""
struct DistanceBinFeature <: Feature
    dim::Int
    min_dist::Float32
    max_dist::Float32
end

DistanceBinFeature(dim::Int) = DistanceBinFeature(dim, 0.0f0, 20.0f0)
DistanceBinFeature(dim::Int, min_dist, max_dist) = DistanceBinFeature(dim, Float32(min_dist), Float32(max_dist))

function (f::DistanceBinFeature)(batch::Dict, L::Int, B::Int)
    # Get CA coordinates
    ca = nothing
    if haskey(batch, :ca_coors)
        ca = batch[:ca_coors]
    elseif haskey(batch, :ca_coors_nm)
        ca = batch[:ca_coors_nm]
    elseif haskey(batch, :x_t) && haskey(batch[:x_t], :bb_ca)
        ca = batch[:x_t][:bb_ca]
    elseif haskey(batch, :coords)
        ca = batch[:coords][:, CA_INDEX, :, :]
    end

    if isnothing(ca)
        return zeros(Float32, f.dim, L, L, B)
    end

    return bin_pairwise_distances(ca, f.min_dist, f.max_dist, f.dim)  # [dim, L, L, B]
end

"""
    XtBBCAPairDistFeature(dim, min_dist, max_dist)

Pairwise distances from x_t backbone CA.
"""
struct XtBBCAPairDistFeature <: Feature
    dim::Int
    min_dist::Float32
    max_dist::Float32
end

function (f::XtBBCAPairDistFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :x_t) && haskey(batch[:x_t], :bb_ca)
        ca = batch[:x_t][:bb_ca]
        return bin_pairwise_distances(ca, f.min_dist, f.max_dist, f.dim)
    else
        return zeros(Float32, f.dim, L, L, B)
    end
end

"""
    XscBBCAPairDistFeature(dim, min_dist, max_dist)

Pairwise distances from self-conditioning CA.
"""
struct XscBBCAPairDistFeature <: Feature
    dim::Int
    min_dist::Float32
    max_dist::Float32
    mode_key::Symbol
end

XscBBCAPairDistFeature(dim, min_dist, max_dist) = XscBBCAPairDistFeature(dim, Float32(min_dist), Float32(max_dist), :x_sc)

function (f::XscBBCAPairDistFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, f.mode_key) && haskey(batch[f.mode_key], :bb_ca)
        ca = batch[f.mode_key][:bb_ca]
        return bin_pairwise_distances(ca, f.min_dist, f.max_dist, f.dim)
    else
        return zeros(Float32, f.dim, L, L, B)
    end
end

"""
    OptionalCAPairDistFeature(dim)

Optional CA pairwise distance feature (for conditional generation).
"""
struct OptionalCAPairDistFeature <: Feature
    dim::Int
    min_dist::Float32
    max_dist::Float32
end

OptionalCAPairDistFeature() = OptionalCAPairDistFeature(30, 0.1f0, 3.0f0)

function (f::OptionalCAPairDistFeature)(batch::Dict, L::Int, B::Int)
    if get(batch, :use_ca_coors_nm_feature, false)
        ca = nothing
        if haskey(batch, :ca_coors_nm)
            ca = batch[:ca_coors_nm]
        elseif haskey(batch, :ca_coors)
            ca = batch[:ca_coors]
        end
        if !isnothing(ca)
            return bin_pairwise_distances(ca, f.min_dist, f.max_dist, f.dim)
        end
    end
    return zeros(Float32, f.dim, L, L, B)
end

"""
    CAPairDistFeature(dim, min_dist, max_dist)

CA pairwise distance feature - always uses CA coordinates.
Matches Python ca_coors_nm_pair_dists (30 bins, 0.1-3.0nm).
"""
struct CAPairDistFeature <: Feature
    dim::Int
    min_dist::Float32
    max_dist::Float32
end

CAPairDistFeature() = CAPairDistFeature(30, 0.1f0, 3.0f0)

function (f::CAPairDistFeature)(batch::Dict, L::Int, B::Int)
    ca = nothing
    if haskey(batch, :ca_coors_nm)
        ca = batch[:ca_coors_nm]
    elseif haskey(batch, :ca_coors)
        ca = batch[:ca_coors]
    end
    if !isnothing(ca)
        return bin_pairwise_distances(ca, f.min_dist, f.max_dist, f.dim)
    end
    return zeros(Float32, f.dim, L, L, B)
end

"""
    RelSeqSepFeature(; max_sep::Int=32)

Relative sequence separation feature for pair representation.
Uses seq_sep_dim from config (should be odd).
"""
struct RelSeqSepFeature <: Feature
    max_sep::Int
end

function RelSeqSepFeature(; seq_sep_dim::Int=127)
    # seq_sep_dim is 2*max_sep+1, so max_sep = (seq_sep_dim-1)/2
    max_sep = (seq_sep_dim - 1) ÷ 2
    return RelSeqSepFeature(max_sep)
end

get_dim(f::RelSeqSepFeature) = 2 * f.max_sep + 1

function (f::RelSeqSepFeature)(batch::Dict, L::Int, B::Int)
    if haskey(batch, :pdb_idx)
        return relative_sequence_separation(batch[:pdb_idx]; max_sep=f.max_sep)
    else
        return relative_sequence_separation(L, B; max_sep=f.max_sep)
    end
end

# ============================================================================
# FeatureFactory
# ============================================================================

"""
    FeatureFactory

Combines multiple features and projects to output dimension.
Supports ret_zero mode for empty features (like Python's FeatureFactory).
"""
struct FeatureFactory
    features::Vector{<:Feature}
    projection::Union{Dense, Nothing}  # Nothing when ret_zero=true
    use_ln::Bool
    ln::Union{PyTorchLayerNorm, Nothing}
    mode::Symbol  # :seq or :pair
    ret_zero::Bool  # If true, just output zeros (no features, no projection)
    out_dim::Int    # Output dimension (needed for ret_zero mode)
end

# Only traverse projection and ln for GPU transfer - features is just config
Functors.@functor FeatureFactory (projection, ln)

"""
    FeatureFactory(features::Vector{<:Feature}, out_dim::Int; mode::Symbol=:seq, use_ln::Bool=false)

Create a FeatureFactory that combines features and projects to output dimension.

# Arguments
- `features`: Vector of Feature objects to combine
- `out_dim`: Output dimension
- `mode`: :seq for sequence features, :pair for pair features
- `use_ln`: Whether to apply LayerNorm to output
"""
function FeatureFactory(features::Vector{<:Feature}, out_dim::Int; mode::Symbol=:seq, use_ln::Bool=false)
    total_dim = sum(get_dim(f) for f in features)
    projection = Dense(total_dim => out_dim; bias=false)
    ln = use_ln ? PyTorchLayerNorm(out_dim) : nothing
    return FeatureFactory(features, projection, use_ln, ln, mode, false, out_dim)
end

"""
    FeatureFactory(out_dim::Int; mode::Symbol=:seq, ret_zero::Bool=true)

Create a zero-returning FeatureFactory (no features, just outputs zeros).
Matches Python's behavior when feats=[] or feats=None.
"""
function FeatureFactory(out_dim::Int; mode::Symbol=:seq)
    return FeatureFactory(Feature[], nothing, false, nothing, mode, true, out_dim)
end

function (ff::FeatureFactory)(batch::Dict)
    # Determine L and B from batch
    L, B = _get_dims(batch)

    # Determine target device from batch data
    target_array = _get_reference_array(batch)

    # If ret_zero mode, just return zeros on the same device
    if ff.ret_zero
        if ff.mode == :seq
            result = similar(target_array, Float32, ff.out_dim, L, B)
            fill!(result, zero(Float32))
            return result
        else  # :pair
            result = similar(target_array, Float32, ff.out_dim, L, L, B)
            fill!(result, zero(Float32))
            return result
        end
    end

    # Extract features
    feats = [f(batch, L, B) for f in ff.features]

    # Check if target is on GPU (CuArray)
    is_gpu = !(target_array isa Array)

    # Ensure all features are on the same device as target
    feats_same_device = if is_gpu
        # Move all features to GPU
        map(f -> f isa Array ? Flux.gpu(f) : f, feats)
    else
        # Move all features to CPU
        map(f -> f isa Array ? f : Array(f), feats)
    end

    combined = cat(feats_same_device...; dims=1)  # Concatenate along feature dimension

    # Project
    out = ff.projection(combined)

    # Optional layer norm
    if ff.use_ln && !isnothing(ff.ln)
        out = ff.ln(out)
    end

    # Apply mask if available
    if haskey(batch, :mask)
        mask = batch[:mask]  # [L, B]
        if ff.mode == :seq
            out = out .* reshape(mask, 1, size(mask)...)
        else  # :pair
            pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)  # [L, L, B]
            out = out .* reshape(pair_mask, 1, L, L, B)
        end
    end

    return out
end

"""
Helper to determine L and B from batch.
"""
function _get_dims(batch::Dict)
    # Try various keys to determine dimensions
    if haskey(batch, :z_latent)
        v = batch[:z_latent]
        return size(v, 2), size(v, 3)  # [D, L, B]
    elseif haskey(batch, :ca_coors)
        v = batch[:ca_coors]
        return size(v, 2), size(v, 3)  # [3, L, B]
    elseif haskey(batch, :ca_coors_nm)
        v = batch[:ca_coors_nm]
        return size(v, 2), size(v, 3)
    elseif haskey(batch, :x_t)
        if haskey(batch[:x_t], :bb_ca)
            v = batch[:x_t][:bb_ca]
            return size(v, 2), size(v, 3)
        elseif haskey(batch[:x_t], :local_latents)
            v = batch[:x_t][:local_latents]
            return size(v, 2), size(v, 3)
        end
    elseif haskey(batch, :coords)
        v = batch[:coords]  # [3, 37, L, B]
        return size(v, 3), size(v, 4)
    elseif haskey(batch, :mask)
        v = batch[:mask]  # [L, B]
        return size(v, 1), size(v, 2)
    end
    error("Cannot determine dimensions from batch")
end

"""
Helper to get a reference array from batch to determine device.
"""
function _get_reference_array(batch::Dict)
    # Try various keys to get a reference array
    if haskey(batch, :z_latent)
        return batch[:z_latent]
    elseif haskey(batch, :ca_coors)
        return batch[:ca_coors]
    elseif haskey(batch, :ca_coors_nm)
        return batch[:ca_coors_nm]
    elseif haskey(batch, :x_t)
        if haskey(batch[:x_t], :bb_ca)
            return batch[:x_t][:bb_ca]
        elseif haskey(batch[:x_t], :local_latents)
            return batch[:x_t][:local_latents]
        end
    elseif haskey(batch, :coords)
        return batch[:coords]
    elseif haskey(batch, :mask)
        return batch[:mask]
    end
    # Default to CPU array
    return zeros(Float32, 1)
end

# ============================================================================
# Convenience constructors matching Python config
# ============================================================================

"""
    score_network_seq_features(token_dim::Int; latent_dim::Int=8, t_emb_dim::Int=256)

Create FeatureFactory for score network sequence representation.
Matches Python checkpoint config: feats_seq: ["xt_bb_ca", "xt_local_latents", "x_sc_bb_ca", "x_sc_local_latents",
                                               "optional_ca_coors_nm_seq_feat", "optional_res_type_seq_feat",
                                               "cropped_flag_seq"]
Total: 3 + 8 + 3 + 8 + 3 + 20 + 1 = 46 dims
"""
function score_network_seq_features(token_dim::Int; latent_dim::Int=8)
    features = Feature[
        XtBBCAFeature(),                  # 3
        XtLocalLatentsFeature(latent_dim), # latent_dim (8)
        XscBBCAFeature(),                 # 3
        XscLocalLatentsFeature(latent_dim), # latent_dim (8)
        OptionalCACoorsFeature(),          # 3
        OptionalResTypeFeature(),          # 20
        CroppedFlagFeature(),              # 1
    ]
    return FeatureFactory(features, token_dim; mode=:seq)
end

"""
    score_network_cond_features(cond_dim::Int; t_emb_dim::Int=256)

Create FeatureFactory for score network conditioning.
Matches Python config: feats_cond_seq: ["time_emb_bb_ca", "time_emb_local_latents"]
"""
function score_network_cond_features(cond_dim::Int; t_emb_dim::Int=256)
    features = Feature[
        TimeFeature(t_emb_dim, :bb_ca),       # t_emb_dim
        TimeFeature(t_emb_dim, :local_latents), # t_emb_dim
    ]
    return FeatureFactory(features, cond_dim; mode=:seq)
end

"""
    score_network_pair_features(pair_dim::Int; xt_pair_dist_dim::Int=30,
                                x_sc_pair_dist_dim::Int=30, seq_sep_dim::Int=127)

Create FeatureFactory for score network pair representation.
Matches Python config: feats_pair_repr: ["rel_seq_sep", "xt_bb_ca_pair_dists",
                                          "x_sc_bb_ca_pair_dists", "optional_ca_pair_dist"]
"""
function score_network_pair_features(pair_dim::Int;
        xt_pair_dist_dim::Int=30, xt_pair_dist_min::Real=0.1, xt_pair_dist_max::Real=3.0,
        x_sc_pair_dist_dim::Int=30, x_sc_pair_dist_min::Real=0.1, x_sc_pair_dist_max::Real=3.0,
        seq_sep_dim::Int=127)
    features = Feature[
        RelSeqSepFeature(; seq_sep_dim=seq_sep_dim),
        XtBBCAPairDistFeature(xt_pair_dist_dim, Float32(xt_pair_dist_min), Float32(xt_pair_dist_max)),
        XscBBCAPairDistFeature(x_sc_pair_dist_dim, Float32(x_sc_pair_dist_min), Float32(x_sc_pair_dist_max)),
        OptionalCAPairDistFeature(),
    ]
    return FeatureFactory(features, pair_dim; mode=:pair, use_ln=true)
end

"""
    score_network_pair_cond_features(cond_dim::Int; t_emb_dim::Int=256)

Create FeatureFactory for score network pair conditioning.
Matches Python config: feats_pair_cond: ["time_emb_bb_ca", "time_emb_local_latents"]
"""
function score_network_pair_cond_features(cond_dim::Int; t_emb_dim::Int=256)
    features = Feature[
        TimePairFeature(t_emb_dim, :bb_ca),
        TimePairFeature(t_emb_dim, :local_latents),
    ]
    return FeatureFactory(features, cond_dim; mode=:pair, use_ln=true)
end

# Legacy compatibility functions

"""
    encoder_seq_features(token_dim::Int; latent_dim::Int=8)

Create FeatureFactory for encoder sequence representation.
"""
function encoder_seq_features(token_dim::Int; latent_dim::Int=8)
    features = Feature[
        CACoordFeature(),
        PositionFeature(64),
    ]
    return FeatureFactory(features, token_dim; mode=:seq)
end

"""
    encoder_cond_features(cond_dim::Int)

Create FeatureFactory for encoder conditioning.
"""
function encoder_cond_features(cond_dim::Int)
    features = Feature[
        ZeroFeature(64),  # Placeholder - can add more conditioning
    ]
    return FeatureFactory(features, cond_dim; mode=:seq)
end

"""
    encoder_pair_features(pair_dim::Int)

Create FeatureFactory for encoder pair representation.
"""
function encoder_pair_features(pair_dim::Int)
    features = Feature[
        DistanceBinFeature(64),
        RelSeqSepFeature(32),
    ]
    return FeatureFactory(features, pair_dim; mode=:pair)
end

"""
    decoder_seq_features(token_dim::Int; latent_dim::Int=8)

Create FeatureFactory for decoder sequence representation.
Matches Python: ["ca_coors_nm", "z_latent_seq"] = 3 + 8 = 11 input dims.
"""
function decoder_seq_features(token_dim::Int; latent_dim::Int=8)
    features = Feature[
        CACoordFeature(),      # 3 dims (ca_coors_nm)
        LatentFeature(latent_dim),  # 8 dims (z_latent_seq)
    ]
    # Total input: 11 dims -> projects to token_dim
    return FeatureFactory(features, token_dim; mode=:seq)
end

"""
    decoder_pair_features(pair_dim::Int; seq_sep_dim::Int=127)

Create FeatureFactory for decoder pair representation.
Matches Python: ["rel_seq_sep", "ca_coors_nm_pair_dists"] = 127 + 30 = 157 input dims.
"""
function decoder_pair_features(pair_dim::Int; seq_sep_dim::Int=127)
    features = Feature[
        RelSeqSepFeature(; seq_sep_dim=seq_sep_dim),  # 127 dims
        CAPairDistFeature(),  # 30 dims (0.1-3.0nm, 30 bins)
    ]
    # Total input: 157 dims -> projects to pair_dim
    return FeatureFactory(features, pair_dim; mode=:pair)
end

"""
    decoder_cond_features(cond_dim::Int)

Create FeatureFactory for decoder conditioning (empty features, outputs zeros).
Matches Python: feats_cond_seq is empty -> ret_zero=True.
"""
function decoder_cond_features(cond_dim::Int)
    # No features - just return zeros (ret_zero mode like Python)
    return FeatureFactory(cond_dim; mode=:seq)
end
