# Score Network for Flow Matching
# Port of local_latents_transformer.py from la-proteina

using Flux
using Functors

"""
    PairReprBuilder

Builds initial pair representation with optional AdaLN conditioning.
Port of pair_rep_initial.py.
"""
struct PairReprBuilder
    init_repr_factory::FeatureFactory
    cond_factory::Union{FeatureFactory, Nothing}
    adaln::Union{AdaptiveLayerNormIdentical, Nothing}
end

Flux.@layer PairReprBuilder

function PairReprBuilder(pair_dim::Int, cond_dim::Int;
        use_conditioning::Bool=true, motif_mode::Symbol=:none,
        xt_pair_dist_dim::Int=30, xt_pair_dist_min::Real=0.1, xt_pair_dist_max::Real=3.0,
        x_sc_pair_dist_dim::Int=30, x_sc_pair_dist_min::Real=0.1, x_sc_pair_dist_max::Real=3.0,
        seq_sep_dim::Int=127, t_emb_dim::Int=256)

    # Pair representation features
    pair_kwargs = (xt_pair_dist_dim=xt_pair_dist_dim, xt_pair_dist_min=xt_pair_dist_min, xt_pair_dist_max=xt_pair_dist_max,
        x_sc_pair_dist_dim=x_sc_pair_dist_dim, x_sc_pair_dist_min=x_sc_pair_dist_min, x_sc_pair_dist_max=x_sc_pair_dist_max,
        seq_sep_dim=seq_sep_dim)
    if motif_mode == :aa
        # LD4: base 217D + motif pair dists 84D = 301D
        init_repr_factory = score_network_pair_features_motif_aa(pair_dim; pair_kwargs...)
    else
        # LD1/LD2/LD3/LD5/LD6/LD7: base 217D (LD5 does NOT use motif pair dists)
        init_repr_factory = score_network_pair_features(pair_dim; pair_kwargs...)
    end

    # Optional conditioning
    if use_conditioning
        cond_factory = score_network_pair_cond_features(cond_dim; t_emb_dim=t_emb_dim)
        adaln = AdaptiveLayerNormIdentical(pair_dim, cond_dim; mode=:pair, use_ln_cond=true)
    else
        cond_factory = nothing
        adaln = nothing
    end

    return PairReprBuilder(init_repr_factory, cond_factory, adaln)
end

function (m::PairReprBuilder)(batch::Dict)
    mask = batch[:mask]  # [L, B]
    L, B = size(mask)

    # Compute pair mask
    pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)  # [L, L, B]

    # Get initial pair representation
    repr = m.init_repr_factory(batch)  # [pair_dim, L, L, B]

    # Apply conditioning if present
    if !isnothing(m.cond_factory) && !isnothing(m.adaln)
        cond = m.cond_factory(batch)  # [cond_dim, L, L, B]
        repr = m.adaln(repr, cond, pair_mask)
    end

    return repr
end

"""
    ScoreNetwork(;
        n_layers::Int=14,
        token_dim::Int=768,
        pair_dim::Int=256,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        qk_ln::Bool=true,
        update_pair_repr::Bool=false,
        update_pair_every_n::Int=3,
        use_tri_mult::Bool=false,
        motif_mode::Symbol=:none,
        cropped_flag::Bool=false,
        output_param::Symbol=:v,
        t_emb_dim::Int=256,
        seq_sep_dim::Int=127,
        xt_pair_dist_dim::Int=30,
        x_sc_pair_dist_dim::Int=30)

Score/velocity network for flow matching on protein structure.
Takes noisy x_t (CA coordinates and latents), time t, and predicts velocity or x1.

# Motif mode
- `:none` — unconditional (LD1/LD2/LD3), seq=45D (or 46D with cropped_flag)
- `:aa` — all-atom indexed motif (LD4), seq=549D, pair=301D
- `:tip` — tip-atom indexed motif (LD5), seq=250D, pair=217D
- `:uidx` — unindexed motif (LD6/LD7), seq=45D base + separate 205D motif projection

Set `cropped_flag=true` for LD1 compatibility (46 seq features instead of 45).

Matches Python config: local_latents_score_nn_160M.yaml
"""
struct ScoreNetwork
    # Feature factories
    init_repr_factory::FeatureFactory
    cond_factory::FeatureFactory
    pair_rep_builder::PairReprBuilder

    # Unindexed motif projection (LD6/LD7): Dense(205 => token_dim, bias=false)
    # Motif tokens are extracted as compact set and appended to sequence.
    motif_uidx_proj::Union{Nothing, Dense}

    # Conditioning transitions
    transition_c_1::ConditioningTransition
    transition_c_2::ConditioningTransition

    # Transformer layers
    transformer_layers::Vector{TransformerBlock}

    # Pair update layers (optional)
    pair_update_layers::Vector{Union{PairUpdate, Nothing}}

    # Output projections
    local_latents_proj::Flux.Chain
    ca_proj::Flux.Chain

    # Config
    n_layers::Int
    update_pair_repr::Bool
    output_param::Symbol
end

Flux.@layer ScoreNetwork

# Custom loadmodel! with strict validation (matching SafeTensors loader pattern).
# strict=true (default): errors if checkpoint and model fields don't match.
# strict=false: skips missing fields (for backward compat with old JLD2 checkpoints).
function Flux.loadmodel!(dst::ScoreNetwork, src::NamedTuple;
                         strict::Bool=true, filter=(_ -> true), cache=Base.IdSet())
    src_keys = keys(src)
    dst_keys = [f for f in fieldnames(ScoreNetwork)
                if getfield(dst, f) !== nothing && !(f in (:n_layers, :update_pair_repr, :output_param))]
    if strict
        missing_in_src = setdiff(dst_keys, src_keys)
        missing_in_dst = setdiff([k for k in src_keys if !(k in (:n_layers, :update_pair_repr, :output_param))], dst_keys)
        mismatches = String[]
        !isempty(missing_in_src) && push!(mismatches, "model has fields not in checkpoint: $(missing_in_src)")
        !isempty(missing_in_dst) && push!(mismatches, "checkpoint has fields not in model: $(missing_in_dst)")
        !isempty(mismatches) && error("ScoreNetwork loadmodel! mismatch: $(join(mismatches, "; "))")
    end
    for fname in fieldnames(ScoreNetwork)
        fname in src_keys || continue
        dst_field = getfield(dst, fname)
        src_field = src[fname]
        dst_field === nothing && continue
        src_field === nothing && continue
        if Functors.isleaf(dst_field)
            push!(cache, dst_field)
            Flux.loadleaf!(dst_field, src_field)
        else
            Flux.loadmodel!(dst_field, src_field; filter, cache)
        end
    end
    return dst
end

function ScoreNetwork(;
        n_layers::Int=14,
        token_dim::Int=768,
        pair_dim::Int=256,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        qk_ln::Bool=true,
        update_pair_repr::Bool=false,
        update_pair_every_n::Int=3,
        use_tri_mult::Bool=false,
        motif_mode::Symbol=:none,
        cropped_flag::Bool=false,
        output_param::Symbol=:v,
        t_emb_dim::Int=256,
        seq_sep_dim::Int=127,
        xt_pair_dist_dim::Int=30,
        xt_pair_dist_min::Real=0.1,
        xt_pair_dist_max::Real=3.0,
        x_sc_pair_dist_dim::Int=30,
        x_sc_pair_dist_min::Real=0.1,
        x_sc_pair_dist_max::Real=3.0)

    # Sequence feature factory (matching Python feats_seq)
    if motif_mode == :aa
        init_repr_factory = score_network_seq_features_motif_aa(token_dim; latent_dim=latent_dim)
    elseif motif_mode == :tip
        init_repr_factory = score_network_seq_features_motif_tip(token_dim; latent_dim=latent_dim)
    else
        # :none and :uidx both use base 45D features (uidx motif features go through separate projection)
        init_repr_factory = score_network_seq_features(token_dim; latent_dim=latent_dim, cropped_flag=cropped_flag)
    end

    # Unindexed motif projection (LD6/LD7): separate Dense for compact motif tokens
    # 205D = atom37 abs coords (148) + residue type one-hot (20) + atom37 mask (37)
    if motif_mode == :uidx
        motif_uidx_proj = Dense(205 => token_dim; bias=false)
    else
        motif_uidx_proj = nothing
    end

    # Conditioning feature factory (matching Python feats_cond_seq)
    cond_factory = score_network_cond_features(dim_cond; t_emb_dim=t_emb_dim)

    # Pair representation builder (matching Python pair_repr_builder)
    pair_rep_builder = PairReprBuilder(pair_dim, dim_cond;
        use_conditioning=true, motif_mode=motif_mode,
        xt_pair_dist_dim=xt_pair_dist_dim, xt_pair_dist_min=xt_pair_dist_min, xt_pair_dist_max=xt_pair_dist_max,
        x_sc_pair_dist_dim=x_sc_pair_dist_dim, x_sc_pair_dist_min=x_sc_pair_dist_min, x_sc_pair_dist_max=x_sc_pair_dist_max,
        seq_sep_dim=seq_sep_dim, t_emb_dim=t_emb_dim)

    # Conditioning transitions
    transition_c_1 = ConditioningTransition(dim_cond; expansion_factor=2)
    transition_c_2 = ConditioningTransition(dim_cond; expansion_factor=2)

    # Transformer layers
    transformer_layers = [
        TransformerBlock(
            dim_token=token_dim,
            dim_pair=pair_dim,
            n_heads=n_heads,
            dim_cond=dim_cond,
            qk_ln=qk_ln,
            residual_mha=true,
            residual_transition=true,
            parallel=false
        )
        for _ in 1:n_layers
    ]

    # Pair update layers
    pair_update_layers = Vector{Union{PairUpdate, Nothing}}(nothing, n_layers - 1)
    if update_pair_repr
        for i in 1:(n_layers-1)
            if (i - 1) % update_pair_every_n == 0  # Match Python: i % n == 0 for 0-indexed
                pair_update_layers[i] = PairUpdate(token_dim, pair_dim; use_tri_mult=use_tri_mult)
            end
        end
    end

    # Output projections
    local_latents_proj = Flux.Chain(
        PyTorchLayerNorm(token_dim),
        Dense(token_dim => latent_dim; bias=false)
    )
    ca_proj = Flux.Chain(
        PyTorchLayerNorm(token_dim),
        Dense(token_dim => 3; bias=false)
    )

    return ScoreNetwork(
        init_repr_factory, cond_factory, pair_rep_builder,
        motif_uidx_proj,
        transition_c_1, transition_c_2,
        transformer_layers, pair_update_layers,
        local_latents_proj, ca_proj,
        n_layers, update_pair_repr, output_param
    )
end

function (m::ScoreNetwork)(batch::Dict)
    # Expected batch contents:
    # - x_t: Dict with :bb_ca => [3, L, B] and :local_latents => [latent_dim, L, B]
    # - t: Dict with :bb_ca => [B] and :local_latents => [B]
    # - mask: [L, B]
    # Optional:
    # - x_sc: Dict with :bb_ca and :local_latents for self-conditioning

    mask = get(batch, :mask, nothing)

    # Get dimensions from x_t
    x_t = batch[:x_t]
    bb_ca = x_t[:bb_ca]  # [3, L, B]
    L, B = size(bb_ca, 2), size(bb_ca, 3)

    if isnothing(mask)
        mask = ones(Float32, L, B)
    end

    # Ensure mask is in batch for feature factories
    batch_with_mask = copy(batch)
    batch_with_mask[:mask] = mask

    # Get conditioning variables
    cond = m.cond_factory(batch_with_mask)  # [dim_cond, L, B]
    cond = m.transition_c_1(cond, mask)
    cond = m.transition_c_2(cond, mask)

    # Get initial sequence representation
    seqs = m.init_repr_factory(batch_with_mask)  # [token_dim, L, B]
    mask_exp = reshape(mask, 1, size(mask)...)
    seqs = seqs .* mask_exp

    # Get pair representation with conditioning
    pair_rep = m.pair_rep_builder(batch_with_mask)  # [pair_dim, L, L, B]

    # Unindexed motif: extract compact motif tokens and append to sequence
    n_orig = L
    if !isnothing(m.motif_uidx_proj) && haskey(batch, :motif_mask)
        motif_tokens, motif_mask_compact = _extract_uidx_motif_tokens(
            batch, m.motif_uidx_proj)
        n_motif = size(motif_tokens, 2)  # max motif residues across batch
        if n_motif > 0
            # Concatenate motif tokens to sequence
            seqs = cat(seqs, motif_tokens; dims=2)  # [token_dim, L+n_motif, B]
            # Extend mask
            mask = cat(mask, motif_mask_compact; dims=1)  # [L+n_motif, B]
            mask_exp = reshape(mask, 1, size(mask)...)
            # Extend conditioning with zeros (motif tokens get no time conditioning)
            cond = cat(cond, zeros(Float32, size(cond, 1), n_motif, B); dims=2)
            # Extend pair representation with zeros
            D_pair = size(pair_rep, 1)
            L_ext = L + n_motif
            pair_ext = zeros(Float32, D_pair, L_ext, L_ext, B)
            pair_ext[:, 1:L, 1:L, :] .= pair_rep
            pair_rep = pair_ext
            L = L_ext
        end
    end

    # Run transformer layers
    for i in 1:m.n_layers
        seqs = m.transformer_layers[i](seqs, pair_rep, cond, mask)

        # Optional pair update
        if m.update_pair_repr && i < m.n_layers
            if !isnothing(m.pair_update_layers[i])
                pair_rep = m.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Trim back to original sequence length (discard motif token outputs)
    if n_orig < L
        seqs = seqs[:, 1:n_orig, :]
        mask = batch_with_mask[:mask]
        mask_exp = reshape(mask, 1, size(mask)...)
    end

    # Project to outputs
    local_latents_out = m.local_latents_proj(seqs)  # [latent_dim, L, B]
    local_latents_out = local_latents_out .* mask_exp

    ca_out = m.ca_proj(seqs)  # [3, L, B]
    ca_out = ca_out .* mask_exp

    # Return in format matching Python
    out_key = m.output_param  # :v or :x1
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out)
    )
end

"""
    score_network_forward(model::ScoreNetwork, x_t, t; mask=nothing, x_sc=nothing)

Convenience function to run score network.
"""
function score_network_forward(model::ScoreNetwork, x_t, t; mask=nothing, x_sc=nothing)
    batch = Dict{Symbol, Any}(
        :x_t => x_t,
        :t => t
    )
    if !isnothing(mask)
        batch[:mask] = mask
    end
    if !isnothing(x_sc)
        batch[:x_sc] = x_sc
    end
    return model(batch)
end

# ============================================================================
# Unindexed motif token extraction (LD6/LD7)
# ============================================================================

"""
    _extract_uidx_motif_tokens(batch, proj) -> (motif_tokens, motif_mask_compact)

Extract compact unindexed motif tokens from the batch.
Filters to only residues with at least one motif atom, pads across batch.

Returns:
- `motif_tokens`: [token_dim, max_n_motif, B] projected motif features
- `motif_mask_compact`: [max_n_motif, B] binary mask
"""
function _extract_uidx_motif_tokens(batch::Dict, proj::Dense)
    motif_mask = batch[:motif_mask]   # [37, L, B] — atom-level mask
    x_motif = batch[:x_motif]         # [3, 37, L, B] — atom coordinates
    seq_motif = batch[:seq_motif]     # [20, L, B] — one-hot residue types

    B = size(motif_mask, 3)
    L = size(motif_mask, 2)

    # Per-residue motif flag: at least one atom present
    res_has_motif = dropdims(sum(motif_mask; dims=1); dims=1) .> 0  # [L, B] Bool

    # Find max number of motif residues across batch
    n_motifs = [count(res_has_motif[:, b]) for b in 1:B]
    max_n_motif = maximum(n_motifs; init=0)

    if max_n_motif == 0
        token_dim = size(proj.weight, 1)
        return zeros(Float32, token_dim, 0, B), zeros(Float32, 0, B)
    end

    # Build compact feature tensor [205, max_n_motif, B]
    feat = zeros(Float32, 205, max_n_motif, B)
    compact_mask = zeros(Float32, max_n_motif, B)

    for b in 1:B
        idx = 0
        for l in 1:L
            if res_has_motif[l, b]
                idx += 1
                # atom37 abs coords: [3, 37] → flatten to [111] then mask [37] = 148D
                coords = x_motif[:, :, l, b]     # [3, 37]
                amask = motif_mask[:, l, b]       # [37]
                feat[1:111, idx, b] .= vec(coords)        # [3*37] flattened coordinates
                feat[112:148, idx, b] .= amask             # [37] atom mask
                # residue type one-hot: 20D
                feat[149:168, idx, b] .= seq_motif[:, l, b]
                # atom37 mask again as float: 37D
                feat[169:205, idx, b] .= amask
                compact_mask[idx, b] = 1.0f0
            end
        end
    end

    # Project to token_dim
    motif_tokens = proj(feat)  # [token_dim, max_n_motif, B]
    motif_tokens = motif_tokens .* reshape(compact_mask, 1, max_n_motif, B)

    return motif_tokens, compact_mask
end

# ============================================================================
# Separated feature extraction and forward pass for training
# ============================================================================

"""
    ScoreNetworkFeatures

Container for pre-extracted features from ScoreNetwork.
Used to separate non-differentiable feature extraction from differentiable forward pass.
"""
struct ScoreNetworkFeatures{T}
    seq_features::Array{T, 3}      # [token_dim, L, B] - sequence features (projected)
    cond_features::Array{T, 3}     # [dim_cond, L, B] - conditioning features (projected)
    pair_features::Array{T, 4}     # [pair_dim, L, L, B] - pair features (projected)
    pair_cond::Array{T, 4}         # [dim_cond, L, L, B] - pair conditioning (projected)
    mask::Array{T, 2}              # [L, B]
end

"""
    ScoreNetworkRawFeatures

Container for raw (unprojected) features from ScoreNetwork.
Used for GPU training where projection happens on GPU inside the gradient context.
Works with both CPU Arrays and GPU CuArrays.
"""
struct ScoreNetworkRawFeatures{A3, A4, A2}
    seq_raw::A3        # [raw_seq_dim, L, B]
    cond_raw::A3       # [raw_cond_dim, L, B]
    pair_raw::A4       # [raw_pair_dim, L, L, B]
    pair_cond_raw::A4  # [raw_pair_cond_dim, L, L, B]
    mask::A2           # [L, B]
end

"""
    extract_raw_features(model::ScoreNetwork, batch::Dict)

Extract raw features (before projection) from batch. This includes all feature
computation which may involve non-differentiable operations (scalar indexing, etc.).

Call this OUTSIDE the gradient context, then pass to `forward_from_raw_features`.

# Returns
ScoreNetworkRawFeatures containing unprojected features for GPU projection.
"""
function extract_raw_features(model::ScoreNetwork, batch::Dict)
    mask = get(batch, :mask, nothing)
    x_t = batch[:x_t]
    bb_ca = x_t[:bb_ca]
    L, B = size(bb_ca, 2), size(bb_ca, 3)

    if isnothing(mask)
        mask = ones(Float32, L, B)
    end

    batch_with_mask = copy(batch)
    batch_with_mask[:mask] = mask

    # Extract raw features (before projection)
    seq_raw = cat([f(batch_with_mask, L, B) for f in model.init_repr_factory.features]...; dims=1)
    cond_raw = cat([f(batch_with_mask, L, B) for f in model.cond_factory.features]...; dims=1)
    pair_raw = cat([f(batch_with_mask, L, B) for f in model.pair_rep_builder.init_repr_factory.features]...; dims=1)

    # Pair conditioning raw features
    if !isnothing(model.pair_rep_builder.cond_factory)
        pair_cond_raw = cat([f(batch_with_mask, L, B) for f in model.pair_rep_builder.cond_factory.features]...; dims=1)
    else
        pair_cond_raw = zeros(Float32, 0, L, L, B)
    end

    return ScoreNetworkRawFeatures(seq_raw, cond_raw, pair_raw, pair_cond_raw, Float32.(mask))
end

"""
    compute_sc_feature_offsets(model::ScoreNetwork)

Compute the byte offsets of self-conditioning features within raw feature tensors.
Returns (seq_offsets, pair_offsets) where each is a vector of (start, stop, feature_type)
tuples for the SC-related features.

Only needs to be called once per model — cache the result.
"""
function compute_sc_feature_offsets(model::ScoreNetwork)
    # Sequence features: find XscBBCAFeature and XscLocalLatentsFeature
    seq_offsets = Tuple{Int,Int,Symbol}[]
    offset = 0
    for f in model.init_repr_factory.features
        d = get_dim(f)
        if f isa XscBBCAFeature
            push!(seq_offsets, (offset + 1, offset + d, :bb_ca))
        elseif f isa XscLocalLatentsFeature
            push!(seq_offsets, (offset + 1, offset + d, :local_latents))
        end
        offset += d
    end

    # Pair features: find XscBBCAPairDistFeature
    pair_offsets = Tuple{Int,Int,Float32,Float32,Int}[]
    offset = 0
    for f in model.pair_rep_builder.init_repr_factory.features
        d = get_dim(f)
        if f isa XscBBCAPairDistFeature
            push!(pair_offsets, (offset + 1, offset + d, f.min_dist, f.max_dist, d))
        end
        offset += d
    end

    return (seq=seq_offsets, pair=pair_offsets)
end

"""
    update_sc_raw_features!(raw_features, sc_offsets, x_sc_bb_ca, x_sc_local_latents)

In-place update of self-conditioning channels in raw features.
Operates directly on GPU tensors — no CPU round-trip needed.

`sc_offsets` should come from `compute_sc_feature_offsets(model)`.
`x_sc_bb_ca` is [3, L, B] and `x_sc_local_latents` is [D, L, B], both on GPU.
"""
function update_sc_raw_features!(raw_features::ScoreNetworkRawFeatures,
                                  sc_offsets,
                                  x_sc_bb_ca, x_sc_local_latents)
    # Update sequence features
    for (start, stop, ftype) in sc_offsets.seq
        if ftype == :bb_ca
            raw_features.seq_raw[start:stop, :, :] .= x_sc_bb_ca
        elseif ftype == :local_latents
            raw_features.seq_raw[start:stop, :, :] .= x_sc_local_latents
        end
    end

    # Update pair features (binned pairwise distances from SC CA coords)
    for (start, stop, min_dist, max_dist, n_bins) in sc_offsets.pair
        sc_pair_dists = bin_pairwise_distances(x_sc_bb_ca, min_dist, max_dist, n_bins)
        raw_features.pair_raw[start:stop, :, :, :] .= sc_pair_dists
    end

    return raw_features
end

"""
    forward_from_raw_features(model::ScoreNetwork, raw_features::ScoreNetworkRawFeatures)

Run full forward pass from raw (unprojected) features.
This projects features and runs the transformer - fully differentiable.
Call this INSIDE the gradient context.

# Returns
Dict with :bb_ca and :local_latents outputs (same format as full model).
"""
function forward_from_raw_features(model::ScoreNetwork, raw_features::ScoreNetworkRawFeatures)
    mask = raw_features.mask
    L, B = size(mask)
    mask_exp = reshape(mask, 1, L, B)
    pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)  # [L, L, B]
    pair_mask_exp = reshape(pair_mask, 1, L, L, B)  # [1, L, L, B]

    # Project features, apply optional LayerNorm, and masks (matching FeatureFactory behavior)
    # Cond features
    cond = model.cond_factory.projection(raw_features.cond_raw)
    if model.cond_factory.use_ln && !isnothing(model.cond_factory.ln)
        cond = model.cond_factory.ln(cond)
    end
    cond = cond .* mask_exp

    # Seq features
    seqs = model.init_repr_factory.projection(raw_features.seq_raw)
    if model.init_repr_factory.use_ln && !isnothing(model.init_repr_factory.ln)
        seqs = model.init_repr_factory.ln(seqs)
    end
    seqs = seqs .* mask_exp

    # Pair features
    pair_rep = model.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw)
    if model.pair_rep_builder.init_repr_factory.use_ln && !isnothing(model.pair_rep_builder.init_repr_factory.ln)
        pair_rep = model.pair_rep_builder.init_repr_factory.ln(pair_rep)
    end
    pair_rep = pair_rep .* pair_mask_exp

    # Apply conditioning transitions (no residual, matches full model forward)
    cond = model.transition_c_1(cond, mask)
    cond = model.transition_c_2(cond, mask)

    # Apply pair AdaLN conditioning if present
    if !isnothing(model.pair_rep_builder.adaln) && !isnothing(model.pair_rep_builder.cond_factory)
        pair_cond = model.pair_rep_builder.cond_factory.projection(raw_features.pair_cond_raw)
        if model.pair_rep_builder.cond_factory.use_ln && !isnothing(model.pair_rep_builder.cond_factory.ln)
            pair_cond = model.pair_rep_builder.cond_factory.ln(pair_cond)
        end
        pair_cond = pair_cond .* pair_mask_exp
        pair_rep = model.pair_rep_builder.adaln(pair_rep, pair_cond, pair_mask)
    end

    # Run transformer layers
    for i in 1:model.n_layers
        seqs = model.transformer_layers[i](seqs, pair_rep, cond, mask)

        # Optional pair update
        if model.update_pair_repr && i < model.n_layers
            if !isnothing(model.pair_update_layers[i])
                pair_rep = model.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Project to outputs
    local_latents_out = model.local_latents_proj(seqs) .* mask_exp
    ca_out = model.ca_proj(seqs) .* mask_exp

    # Return in same format as full model
    out_key = model.output_param
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out)
    )
end

"""
    extract_features(model::ScoreNetwork, batch::Dict)

Extract and project features from batch. This includes all feature computation
which may involve non-differentiable operations (scalar indexing, etc.).

Call this OUTSIDE the gradient context, then pass features to `forward_from_features`.

# Returns
ScoreNetworkFeatures containing projected seq, cond, and pair representations.
"""
function extract_features(model::ScoreNetwork, batch::Dict)
    mask = get(batch, :mask, nothing)
    x_t = batch[:x_t]
    bb_ca = x_t[:bb_ca]
    L, B = size(bb_ca, 2), size(bb_ca, 3)

    if isnothing(mask)
        mask = ones(Float32, L, B)
    end

    batch_with_mask = copy(batch)
    batch_with_mask[:mask] = mask

    # Extract and project sequence features
    seq_features = model.init_repr_factory(batch_with_mask)  # [token_dim, L, B]

    # Extract and project conditioning features
    cond_features = model.cond_factory(batch_with_mask)  # [dim_cond, L, B]

    # Extract and project pair features
    pair_features = model.pair_rep_builder.init_repr_factory(batch_with_mask)  # [pair_dim, L, L, B]

    # Extract and project pair conditioning (if present)
    if !isnothing(model.pair_rep_builder.cond_factory)
        pair_cond = model.pair_rep_builder.cond_factory(batch_with_mask)  # [dim_cond, L, L, B]
    else
        pair_cond = zeros(Float32, 0, L, L, B)
    end

    return ScoreNetworkFeatures(seq_features, cond_features, pair_features, pair_cond, Float32.(mask))
end

"""
    forward_from_features(model::ScoreNetwork, features::ScoreNetworkFeatures)

Run the trainable forward pass from pre-extracted features.
This is fully differentiable and should be called INSIDE the gradient context.

# Returns
Dict with :bb_ca and :local_latents outputs (same format as full model).
"""
function forward_from_features(model::ScoreNetwork, features::ScoreNetworkFeatures)
    mask = features.mask
    L, B = size(mask)
    mask_exp = reshape(mask, 1, L, B)

    # Apply conditioning transitions (no residual, matches full model forward)
    cond = features.cond_features
    cond = model.transition_c_1(cond, mask)
    cond = model.transition_c_2(cond, mask)

    # Apply mask to sequence features
    seqs = features.seq_features .* mask_exp

    # Apply pair AdaLN conditioning if present
    pair_rep = features.pair_features
    if !isnothing(model.pair_rep_builder.adaln) && size(features.pair_cond, 1) > 0
        pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)
        pair_rep = model.pair_rep_builder.adaln(pair_rep, features.pair_cond, pair_mask)
    end

    # Run transformer layers
    for i in 1:model.n_layers
        seqs = model.transformer_layers[i](seqs, pair_rep, cond, mask)

        # Optional pair update
        if model.update_pair_repr && i < model.n_layers
            if !isnothing(model.pair_update_layers[i])
                pair_rep = model.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Project to outputs
    local_latents_out = model.local_latents_proj(seqs) .* mask_exp
    ca_out = model.ca_proj(seqs) .* mask_exp

    # Return in same format as full model
    out_key = model.output_param
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out)
    )
end

# ============================================================================
# Utility functions for flow matching
# ============================================================================

"""
    v_to_x1(x_t, v, t)

Convert velocity prediction to x1 prediction.
x_1 = x_t + (1-t) * v
"""
function v_to_x1(x_t::AbstractArray{T}, v::AbstractArray{T}, t) where T
    # t can be scalar or array
    if isa(t, Number)
        return x_t .+ (one(T) - T(t)) .* v
    else
        # t is [B], need to broadcast properly
        t_exp = reshape(T.(t), 1, 1, :)  # [1, 1, B]
        return x_t .+ (one(T) .- t_exp) .* v
    end
end

"""
    x1_to_v(x_t, x1, t)

Convert x1 prediction to velocity.
v = (x_1 - x_t) / (1-t)
"""
function x1_to_v(x_t::AbstractArray{T}, x1::AbstractArray{T}, t; eps::T=T(1e-5)) where T
    if isa(t, Number)
        return (x1 .- x_t) ./ max(one(T) - T(t), eps)
    else
        t_exp = reshape(T.(t), 1, 1, :)
        return (x1 .- x_t) ./ max.(one(T) .- t_exp, eps)
    end
end

"""
    self_condition_input(batch::Dict, prev_output::Union{Dict, Nothing})

Add self-conditioning from previous output to batch.
If prev_output is nothing, uses zeros.
"""
function self_condition_input(batch::Dict, prev_output::Union{Dict, Nothing})
    x_t = batch[:x_t]

    if isnothing(prev_output)
        # Use zeros for self-conditioning
        sc_bb_ca = zeros(eltype(x_t[:bb_ca]), size(x_t[:bb_ca]))
        sc_local_latents = zeros(eltype(x_t[:local_latents]), size(x_t[:local_latents]))
    else
        # Extract x1 predictions (or v and convert)
        if haskey(prev_output[:bb_ca], :x1)
            sc_bb_ca = prev_output[:bb_ca][:x1]
            sc_local_latents = prev_output[:local_latents][:x1]
        else
            # Convert v to x1
            t = batch[:t]
            t_ca = isa(t, Dict) ? t[:bb_ca] : t
            t_ll = isa(t, Dict) ? t[:local_latents] : t
            v_ca = prev_output[:bb_ca][:v]
            v_ll = prev_output[:local_latents][:v]
            sc_bb_ca = v_to_x1(x_t[:bb_ca], v_ca, t_ca)
            sc_local_latents = v_to_x1(x_t[:local_latents], v_ll, t_ll)
        end
    end

    # Add self-conditioning to batch
    new_batch = copy(batch)
    new_batch[:x_sc] = Dict(
        :bb_ca => sc_bb_ca,
        :local_latents => sc_local_latents
    )
    return new_batch
end
