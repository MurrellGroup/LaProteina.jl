# Score Network for Flow Matching
# Port of local_latents_transformer.py from la-proteina

using Flux

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
        use_conditioning::Bool=true,
        xt_pair_dist_dim::Int=30, xt_pair_dist_min::Real=0.1, xt_pair_dist_max::Real=3.0,
        x_sc_pair_dist_dim::Int=30, x_sc_pair_dist_min::Real=0.1, x_sc_pair_dist_max::Real=3.0,
        seq_sep_dim::Int=127, t_emb_dim::Int=256)

    # Pair representation features
    init_repr_factory = score_network_pair_features(pair_dim;
        xt_pair_dist_dim=xt_pair_dist_dim, xt_pair_dist_min=xt_pair_dist_min, xt_pair_dist_max=xt_pair_dist_max,
        x_sc_pair_dist_dim=x_sc_pair_dist_dim, x_sc_pair_dist_min=x_sc_pair_dist_min, x_sc_pair_dist_max=x_sc_pair_dist_max,
        seq_sep_dim=seq_sep_dim)

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
        output_param::Symbol=:v,
        t_emb_dim::Int=256,
        seq_sep_dim::Int=127,
        xt_pair_dist_dim::Int=30,
        x_sc_pair_dist_dim::Int=30)

Score/velocity network for flow matching on protein structure.
Takes noisy x_t (CA coordinates and latents), time t, and predicts velocity or x1.

Matches Python config: local_latents_score_nn_160M.yaml
"""
struct ScoreNetwork
    # Feature factories
    init_repr_factory::FeatureFactory
    cond_factory::FeatureFactory
    pair_rep_builder::PairReprBuilder

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
    init_repr_factory = score_network_seq_features(token_dim; latent_dim=latent_dim)

    # Conditioning feature factory (matching Python feats_cond_seq)
    cond_factory = score_network_cond_features(dim_cond; t_emb_dim=t_emb_dim)

    # Pair representation builder (matching Python pair_repr_builder)
    pair_rep_builder = PairReprBuilder(pair_dim, dim_cond;
        use_conditioning=true,
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
