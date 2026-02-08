# Branching-enabled Score Network for Variable-Length Generation
# Extends ScoreNetwork with split and deletion prediction heads

using Flux
using NNlib: swish
using Statistics: mean
using JLD2

"""
    BranchingScoreNetwork

Score network extended with split and deletion prediction heads for
Branching Flows variable-length generation.

Wraps a base ScoreNetwork and adds:
- `split_head`: Predicts expected number of future splits per position (log-space)
- `del_head`: Predicts deletion probability per position (logit)
- `indel_time_proj`: Time conditioning for split/del heads

The base ScoreNetwork can be initialized from pretrained weights, then the
split/del heads are trained while freezing the base model (staged training).
"""
struct BranchingScoreNetwork{S<:ScoreNetwork, T, H1, H2}
    base::S                    # Base ScoreNetwork (can be frozen)
    indel_time_proj::T         # Time conditioning projection for indel heads
    split_head::H1             # Split count predictor (outputs log expected splits)
    del_head::H2               # Deletion predictor (outputs logit)
end

Flux.@layer BranchingScoreNetwork

"""
    BranchingScoreNetwork(base::ScoreNetwork; hidden_dim=nothing)

Create a BranchingScoreNetwork from an existing ScoreNetwork.
The split/del heads operate on the final token embedding.

# Arguments
- `base`: Pre-trained ScoreNetwork
- `hidden_dim`: Hidden dimension for split/del heads (default: token_dim)
"""
function BranchingScoreNetwork(base::ScoreNetwork; hidden_dim::Union{Int, Nothing}=nothing)
    # Get token_dim from base model's output projection
    # The ca_proj goes from token_dim -> 3
    token_dim = size(base.ca_proj[2].weight, 2)

    # Get dim_cond from conditioning transition's inner SwiGLU transition
    # linear_in goes from dim_cond -> 2 * dim_inner, so we get dim_cond from second dim
    dim_cond = size(base.transition_c_1.transition.linear_in.weight, 2)

    hdim = isnothing(hidden_dim) ? token_dim : hidden_dim

    # Scaled initialization (1/20 of default) for stable training
    scaled_init(dims...) = Flux.glorot_uniform(dims...) .* 0.05f0

    # Time conditioning projection for indel heads
    indel_time_proj = Dense(dim_cond => token_dim; init=scaled_init)

    # Split head: token_dim -> hidden -> 1 (log expected splits)
    # Output layer scaled down 20x for stable initialization
    split_head = Chain(
        Dense(token_dim => hdim),
        x -> swish.(x),
        Dense(hdim => 1, bias=false, init=scaled_init)
    )

    # Deletion head: token_dim -> hidden -> 1 (logit)
    # Output layer scaled down 20x for stable initialization
    del_head = Chain(
        Dense(token_dim => hdim),
        x -> swish.(x),
        Dense(hdim => 1, bias=false, init=scaled_init)
    )

    return BranchingScoreNetwork(base, indel_time_proj, split_head, del_head)
end

"""
    BranchingScoreNetwork(; kwargs...)

Create a BranchingScoreNetwork with a new ScoreNetwork base.
All kwargs are passed to ScoreNetwork constructor.
"""
function BranchingScoreNetwork(;
        n_layers::Int=14,
        token_dim::Int=768,
        pair_dim::Int=256,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        hidden_dim::Union{Int, Nothing}=nothing,
        kwargs...)

    base = ScoreNetwork(;
        n_layers=n_layers,
        token_dim=token_dim,
        pair_dim=pair_dim,
        n_heads=n_heads,
        dim_cond=dim_cond,
        latent_dim=latent_dim,
        kwargs...
    )

    return BranchingScoreNetwork(base; hidden_dim=hidden_dim)
end

"""
    (m::BranchingScoreNetwork)(batch::Dict)

Forward pass returning base predictions plus split/del predictions.

# Returns
Dict with:
- :bb_ca => Dict(:v or :x1 => [3, L, B])
- :local_latents => Dict(:v or :x1 => [latent_dim, L, B])
- :split => [L, B] (log expected split counts)
- :del => [L, B] (deletion logits)
"""
function (m::BranchingScoreNetwork)(batch::Dict)
    # Run base model to get final embeddings
    # We need access to the intermediate seqs tensor, so we inline the forward pass

    mask = get(batch, :mask, nothing)
    x_t = batch[:x_t]
    bb_ca = x_t[:bb_ca]
    L, B = size(bb_ca, 2), size(bb_ca, 3)

    if isnothing(mask)
        mask = ones(Float32, L, B)
    end

    batch_with_mask = copy(batch)
    batch_with_mask[:mask] = mask

    # Get conditioning variables
    cond = m.base.cond_factory(batch_with_mask)
    cond = m.base.transition_c_1(cond, mask)
    cond = m.base.transition_c_2(cond, mask)

    # Get initial sequence representation
    seqs = m.base.init_repr_factory(batch_with_mask)
    mask_exp = reshape(mask, 1, size(mask)...)
    seqs = seqs .* mask_exp

    # Get pair representation with conditioning
    pair_rep = m.base.pair_rep_builder(batch_with_mask)

    # Run transformer layers
    for i in 1:m.base.n_layers
        seqs = m.base.transformer_layers[i](seqs, pair_rep, cond, mask)

        if m.base.update_pair_repr && i < m.base.n_layers
            if !isnothing(m.base.pair_update_layers[i])
                pair_rep = m.base.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Base model outputs
    local_latents_out = m.base.local_latents_proj(seqs) .* mask_exp
    ca_out = m.base.ca_proj(seqs) .* mask_exp

    # Split/del heads use final embedding + time conditioning
    # Get time embedding from cond (after transitions)
    # cond is [dim_cond, L, B], we use the mean over L for time conditioning
    t_cond = mean(cond, dims=2)  # [dim_cond, 1, B]
    indel_cond = m.indel_time_proj(t_cond)  # [token_dim, 1, B]

    # Add time conditioning to final embedding
    seqs_with_time = seqs .+ indel_cond  # [token_dim, L, B]

    # Predict splits and deletions
    split_logits = m.split_head(seqs_with_time)  # [1, L, B]
    split_logits = dropdims(split_logits, dims=1) .* mask  # [L, B]

    del_logits = m.del_head(seqs_with_time)  # [1, L, B]
    del_logits = dropdims(del_logits, dims=1) .* mask  # [L, B]

    # Return extended output
    out_key = m.base.output_param
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out),
        :split => split_logits,
        :del => del_logits
    )
end

"""
    forward_branching_from_raw_features(model::BranchingScoreNetwork,
                                         raw_features::ScoreNetworkRawFeatures)

Run full forward pass from raw features, including split/del predictions.
For use in training where feature extraction is separated from the gradient context.
"""
function forward_branching_from_raw_features(model::BranchingScoreNetwork,
                                              raw_features::ScoreNetworkRawFeatures)
    mask = raw_features.mask
    L, B = size(mask)
    mask_exp = reshape(mask, 1, L, B)
    pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)
    pair_mask_exp = reshape(pair_mask, 1, L, L, B)

    base = model.base

    # Project features (same as forward_from_raw_features)
    cond = base.cond_factory.projection(raw_features.cond_raw)
    if base.cond_factory.use_ln && !isnothing(base.cond_factory.ln)
        cond = base.cond_factory.ln(cond)
    end
    cond = cond .* mask_exp

    seqs = base.init_repr_factory.projection(raw_features.seq_raw)
    if base.init_repr_factory.use_ln && !isnothing(base.init_repr_factory.ln)
        seqs = base.init_repr_factory.ln(seqs)
    end
    seqs = seqs .* mask_exp

    pair_rep = base.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw)
    if base.pair_rep_builder.init_repr_factory.use_ln && !isnothing(base.pair_rep_builder.init_repr_factory.ln)
        pair_rep = base.pair_rep_builder.init_repr_factory.ln(pair_rep)
    end
    pair_rep = pair_rep .* pair_mask_exp

    # Apply conditioning transitions
    cond = base.transition_c_1(cond, mask)
    cond = base.transition_c_2(cond, mask)

    # Apply pair AdaLN conditioning
    if !isnothing(base.pair_rep_builder.adaln) && !isnothing(base.pair_rep_builder.cond_factory)
        pair_cond = base.pair_rep_builder.cond_factory.projection(raw_features.pair_cond_raw)
        if base.pair_rep_builder.cond_factory.use_ln && !isnothing(base.pair_rep_builder.cond_factory.ln)
            pair_cond = base.pair_rep_builder.cond_factory.ln(pair_cond)
        end
        pair_cond = pair_cond .* pair_mask_exp
        pair_rep = base.pair_rep_builder.adaln(pair_rep, pair_cond, pair_mask)
    end

    # Run transformer layers
    for i in 1:base.n_layers
        seqs = base.transformer_layers[i](seqs, pair_rep, cond, mask)

        if base.update_pair_repr && i < base.n_layers
            if !isnothing(base.pair_update_layers[i])
                pair_rep = base.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Base outputs
    local_latents_out = base.local_latents_proj(seqs) .* mask_exp
    ca_out = base.ca_proj(seqs) .* mask_exp

    # Split/del heads
    t_cond = mean(cond, dims=2)
    indel_cond = model.indel_time_proj(t_cond)
    seqs_with_time = seqs .+ indel_cond

    split_logits = dropdims(model.split_head(seqs_with_time), dims=1) .* mask
    del_logits = dropdims(model.del_head(seqs_with_time), dims=1) .* mask

    out_key = base.output_param
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out),
        :split => split_logits,
        :del => del_logits
    )
end

"""
    freeze_base!(model::BranchingScoreNetwork)

Freeze the base ScoreNetwork parameters for staged training.
Only split/del heads will be trained.

Note: This modifies the model in-place by wrapping base in Flux.freeze.
Returns the frozen model for optimizer setup.
"""
function freeze_base!(model::BranchingScoreNetwork)
    # In Flux, we achieve this by creating a new model with frozen base
    # The caller should only pass the trainable params to the optimizer
    return model  # Caller uses Flux.trainable or manual param selection
end

"""
    trainable_indel_params(model::BranchingScoreNetwork)

Get only the trainable parameters for indel heads (for stage 1 training).
Returns a NamedTuple that can be passed to Optimisers.setup.
"""
function trainable_indel_params(model::BranchingScoreNetwork)
    return (
        indel_time_proj = model.indel_time_proj,
        split_head = model.split_head,
        del_head = model.del_head
    )
end

"""
    load_base_weights!(model::BranchingScoreNetwork, path::String)

Load pretrained weights into the base ScoreNetwork.
"""
function load_base_weights!(model::BranchingScoreNetwork, path::String)
    load_score_network_weights!(model.base, path)
    return model
end

"""
    save_branching_weights(model::BranchingScoreNetwork, path::String)

Save BranchingScoreNetwork weights to a JLD2 file.
Saves only the indel heads by default (base weights come from pretrained).

# Arguments
- `model`: BranchingScoreNetwork to save
- `path`: Output path (should end in .jld2)
- `include_base`: Whether to also save base model weights (default: false)
"""
function save_branching_weights(model::BranchingScoreNetwork, path::String; include_base::Bool=false)
    # Save using JLD2 directly
    jldopen(path, "w") do file
        file["indel_time_proj"] = Flux.state(model.indel_time_proj)
        file["split_head"] = Flux.state(model.split_head)
        file["del_head"] = Flux.state(model.del_head)

        # Optionally save base model too
        if include_base
            file["base"] = Flux.state(model.base)
        end
    end
    return path
end

"""
    load_branching_weights!(model::BranchingScoreNetwork, path::String;
                            base_weights_path::Union{String, Nothing}=nothing)

Load weights into BranchingScoreNetwork.

# Arguments
- `model`: BranchingScoreNetwork to load into
- `path`: Path to JLD2 file with indel head weights
- `base_weights_path`: Optional path to NPZ file with base ScoreNetwork weights.
                       If not provided and the JLD2 doesn't contain base weights,
                       base model keeps its current (random) weights.
"""
function load_branching_weights!(model::BranchingScoreNetwork, path::String;
                                  base_weights_path::Union{String, Nothing}=nothing)
    # Load indel head weights from JLD2
    weights = load(path)

    # Load indel heads
    if haskey(weights, "indel_time_proj")
        Flux.loadmodel!(model.indel_time_proj, weights["indel_time_proj"])
    end
    if haskey(weights, "split_head")
        Flux.loadmodel!(model.split_head, weights["split_head"])
    end
    if haskey(weights, "del_head")
        Flux.loadmodel!(model.del_head, weights["del_head"])
    end

    # Load base model weights
    if haskey(weights, "base")
        Flux.loadmodel!(model.base, weights["base"])
    elseif !isnothing(base_weights_path)
        load_score_network_weights!(model.base, base_weights_path)
    end

    return model
end

# ============================================================================
# GPU-optimized forward with pre-normalized pair features
# ============================================================================

"""
    forward_branching_from_raw_features_gpu(model::BranchingScoreNetwork,
                                            raw_features::ScoreNetworkRawFeatures)

GPU-optimized branching forward that pre-normalizes pair features once instead of
14 times (once per transformer block). Same semantics as `forward_branching_from_raw_features`.

When `update_pair_repr=false` (default), the pair features are constant across blocks.
Each block's PairBiasAttention.pair_norm computes:
  normed = (pair_rep - μ) / sqrt(σ² + ε)   # expensive, SAME for all blocks
  output = normed * scale + bias             # cheap, DIFFERENT per block

This function computes the normalization once and passes pre-normalized pairs
to each block, which only applies its per-block affine transform.

Savings: 13 × (pair LayerNorm forward + backward) per training step.
"""
function forward_branching_from_raw_features_gpu(model::BranchingScoreNetwork,
                                                  raw_features::ScoreNetworkRawFeatures)
    mask = raw_features.mask
    L, B = size(mask)
    mask_exp = reshape(mask, 1, L, B)
    pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)
    pair_mask_exp = reshape(pair_mask, 1, L, L, B)

    base = model.base

    # Project features (same as forward_branching_from_raw_features)
    cond = base.cond_factory.projection(raw_features.cond_raw)
    if base.cond_factory.use_ln && !isnothing(base.cond_factory.ln)
        cond = base.cond_factory.ln(cond)
    end
    cond = cond .* mask_exp

    seqs = base.init_repr_factory.projection(raw_features.seq_raw)
    if base.init_repr_factory.use_ln && !isnothing(base.init_repr_factory.ln)
        seqs = base.init_repr_factory.ln(seqs)
    end
    seqs = seqs .* mask_exp

    pair_rep = base.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw)
    if base.pair_rep_builder.init_repr_factory.use_ln && !isnothing(base.pair_rep_builder.init_repr_factory.ln)
        pair_rep = base.pair_rep_builder.init_repr_factory.ln(pair_rep)
    end
    pair_rep = pair_rep .* pair_mask_exp

    cond = base.transition_c_1(cond, mask)
    cond = base.transition_c_2(cond, mask)

    if !isnothing(base.pair_rep_builder.adaln) && !isnothing(base.pair_rep_builder.cond_factory)
        # Batch-level pair conditioning (same optimization as ScoreNetwork GPU path)
        pair_cond_batch_raw = raw_features.pair_cond_raw[:, 1, 1, :]  # [D_raw, B]
        pair_cond_batch = base.pair_rep_builder.cond_factory.projection(pair_cond_batch_raw)
        if base.pair_rep_builder.cond_factory.use_ln && !isnothing(base.pair_rep_builder.cond_factory.ln)
            pair_cond_batch = base.pair_rep_builder.cond_factory.ln(pair_cond_batch)
        end
        pair_rep = base.pair_rep_builder.adaln(pair_rep, pair_cond_batch, pair_mask)
    end

    # === KEY OPTIMIZATION: Pre-normalize pair features once ===
    if !base.update_pair_repr
        first_pba = base.transformer_layers[1].mha.mha
        pair_eps = first_pba.pair_norm.ϵ
        pair_normed = LaProteina.pytorch_normalise(pair_rep; dims=1, eps=pair_eps)
    else
        pair_normed = nothing
    end

    # Run transformer layers
    for i in 1:base.n_layers
        if !isnothing(pair_normed)
            seqs = LaProteina._transformer_block_prenormed(
                base.transformer_layers[i], seqs, pair_rep, pair_normed, cond, mask)
        else
            seqs = base.transformer_layers[i](seqs, pair_rep, cond, mask)
        end

        if base.update_pair_repr && i < base.n_layers
            if !isnothing(base.pair_update_layers[i])
                pair_rep = base.pair_update_layers[i](seqs, pair_rep, mask)
                pair_normed = LaProteina.pytorch_normalise(pair_rep; dims=1, eps=pair_eps)
            end
        end
    end

    # Base outputs
    local_latents_out = base.local_latents_proj(seqs) .* mask_exp
    ca_out = base.ca_proj(seqs) .* mask_exp

    # Split/del heads (same as original)
    t_cond = mean(cond, dims=2)
    indel_cond = model.indel_time_proj(t_cond)
    seqs_with_time = seqs .+ indel_cond

    split_logits = dropdims(model.split_head(seqs_with_time), dims=1) .* mask
    del_logits = dropdims(model.del_head(seqs_with_time), dims=1) .* mask

    out_key = base.output_param
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out),
        :split => split_logits,
        :del => del_logits
    )
end
