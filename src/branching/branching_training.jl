# Training utilities for Branching Flows
# Handles branching_bridge, loss computation, and staged training

using Flux
using Statistics
using Optimisers
using BranchingFlows: branching_bridge, CoalescentFlow
using Distributions: Uniform, Poisson

"""
    softclamp(loss; threshold=2.0, hardcap=3.0)

Soft clamp for loss values to prevent gradient explosions.
Above threshold, transitions to log growth: threshold + log(loss - threshold + 1)
Then applies a hard cap to ensure output never exceeds hardcap.
"""
function softclamp(loss; threshold=3.5f0, hardcap=5.0f0)
    if loss > threshold
        clamped = threshold + log((loss - threshold) + 1)
        return min(clamped, hardcap)
    else
        return loss
    end
end

"""
    branching_training_batch(proteins::Vector, indices::Vector{Int}, P::CoalescentFlow;
                              latent_dim::Int=8,
                              mean_length::Int=100,
                              deletion_pad::Int=10,
                              use_branching_time_prob::Float64=0.5)

Create a training batch using branching_bridge.

# Arguments
- `proteins`: Vector of precomputed proteins (NamedTuples)
- `indices`: Indices of proteins to use in this batch
- `P`: CoalescentFlow process
- `latent_dim`: Dimension of local latents
- `mean_length`: Mean starting length for X0 (Poisson)
- `deletion_pad`: Padding for deletions
- `use_branching_time_prob`: Probability of using branching time sampling

# Returns
NamedTuple with all tensors needed for loss computation (on CPU)
"""
function branching_training_batch(proteins::Vector, indices::Vector{Int}, P::CoalescentFlow;
                                   latent_dim::Int=8,
                                   mean_length::Int=100,
                                   deletion_pad::Int=10,
                                   use_branching_time_prob::Float64=0.5)

    # Convert proteins to X1 BranchingStates
    X1s = proteins_to_X1_states(proteins, indices)

    # Create X0 sampler
    X0_sampler = X0_sampler_laproteina(latent_dim)

    # Sample time distribution
    t_dist = Uniform(0f0, 1f0)

    # Run branching_bridge to get training targets
    bat = branching_bridge(P, X0_sampler, X1s, t_dist,
        coalescence_factor = 1.0,
        use_branching_time_prob = use_branching_time_prob,
        merger = BranchingFlows.canonical_anchor_merge,
        length_mins = Poisson(mean_length),
        deletion_pad = deletion_pad,
    )

    # Extract tensors from batched BranchingState
    Xt = bat.Xt  # BranchingState at time t
    t = bat.t    # [B] time values

    # Extract Xt state tensors
    xt_tensors = extract_state_tensors(Xt)

    return (
        # Input state at time t
        xt_ca = xt_tensors.ca,           # [3, L, B]
        xt_ll = xt_tensors.latents,       # [latent_dim, L, B]
        xt_indices = xt_tensors.indices,  # [L, B] index tracking
        mask = xt_tensors.mask,           # [L, B]

        # Time
        t = t,  # [B]

        # Targets for base flow matching
        x1_ca_target = bat.X1_locs_target,    # [3, L, B] (or similar field name)
        x1_ll_target = bat.X1_latents_target, # [latent_dim, L, B]

        # Targets for branching
        splits_target = bat.splits_target,    # [L, B] expected split counts
        del_target = bat.del,                 # [L, B] deletion indicators

        # Masks for loss computation
        branchmask = Xt.branchmask,           # [L, B] which positions can branch
        padmask = Xt.padmask,                 # [L, B] valid positions
    )
end

"""
    branching_flow_loss(model::BranchingScoreNetwork, batch;
                        split_weight::Float32=1.0f0,
                        del_weight::Float32=1.0f0)

Compute full branching flow matching loss.

# Arguments
- `model`: BranchingScoreNetwork
- `batch`: NamedTuple from branching_training_batch
- `split_weight`: Weight for split loss
- `del_weight`: Weight for deletion loss

# Returns
Total loss (scalar)
"""
function branching_flow_loss(model::BranchingScoreNetwork, batch;
                              split_weight::Float32=1.0f0,
                              del_weight::Float32=1.0f0)

    # Build model input
    input = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => batch.xt_ca, :local_latents => batch.xt_ll),
        :t => Dict(:bb_ca => batch.t, :local_latents => batch.t),
        :mask => batch.mask
    )

    # Forward pass
    output = model(input)

    # Extract predictions
    out_key = model.base.output_param

    if out_key == :v
        v_ca = output[:bb_ca][:v]
        v_ll = output[:local_latents][:v]
        # Convert to x1 predictions
        t_exp = reshape(batch.t, 1, 1, :)
        x1_ca_pred = batch.xt_ca .+ (1f0 .- t_exp) .* v_ca
        x1_ll_pred = batch.xt_ll .+ (1f0 .- t_exp) .* v_ll
    else
        x1_ca_pred = output[:bb_ca][:x1]
        x1_ll_pred = output[:local_latents][:x1]
    end

    split_pred = output[:split]  # [L, B]
    del_pred = output[:del]      # [L, B]

    # Combined mask for branching losses
    combined_mask = batch.padmask .* batch.branchmask  # [L, B]

    # === Base flow matching losses ===
    # MSE on x1 predictions, weighted by 1/(1-t)^2
    t_scale = 1f0 ./ max.(1f0 .- batch.t, 1f-5).^2  # [B]
    t_scale_exp = reshape(t_scale, 1, 1, :)  # [1, 1, B]

    # CA loss
    ca_diff = (x1_ca_pred .- batch.x1_ca_target).^2
    ca_loss = sum(ca_diff .* reshape(batch.mask, 1, :, size(batch.mask, 2)) .* t_scale_exp) /
              max(sum(batch.mask), 1f0)

    # Latent loss
    ll_diff = (x1_ll_pred .- batch.x1_ll_target).^2
    ll_loss = sum(ll_diff .* reshape(batch.mask, 1, :, size(batch.mask, 2)) .* t_scale_exp) /
              max(sum(batch.mask), 1f0)

    # === Branching losses ===
    t_scale_2d = reshape(t_scale, 1, :)  # [1, B]

    # Split loss: Bregman Poisson loss
    # L(pred, target) = exp(pred) - target * pred (for log-space predictions)
    # Or equivalently: mu - target * log(mu) where mu = exp(pred)
    split_loss = bregman_poisson_loss(split_pred, batch.splits_target,
                                       combined_mask, t_scale_2d)

    # Deletion loss: Binary cross-entropy
    del_loss = logistic_bce_loss(del_pred, batch.del_target,
                                  combined_mask, t_scale_2d)

    # Total loss
    total_loss = ca_loss + ll_loss + split_weight * split_loss + del_weight * del_loss

    return total_loss
end

"""
    bregman_poisson_loss(pred, target, mask, scale)

Bregman Poisson loss for split count prediction.
pred is in log-space (log expected counts), target is the actual count.

L(pred, target) = exp(pred) - target * pred
"""
function bregman_poisson_loss(pred::AbstractArray{T}, target, mask, scale) where T
    mu = exp.(pred)  # Expected count
    # Bregman divergence for Poisson
    loss_per_pos = mu .- target .* pred

    # Apply mask and scale
    loss_per_pos = loss_per_pos .* mask .* scale

    return sum(loss_per_pos) / max(sum(mask), one(T))
end

"""
    logistic_bce_loss(logits, targets, mask, scale)

Binary cross-entropy loss for deletion prediction.
logits are raw (not sigmoid-ed), targets are 0/1.
"""
function logistic_bce_loss(logits::AbstractArray{T}, targets, mask, scale) where T
    # BCE = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
    #     = -[y * log(σ(x)) + (1-y) * log(σ(-x))]
    #     = -[y * (-softplus(-x)) + (1-y) * (-softplus(x))]
    #     = y * softplus(-x) + (1-y) * softplus(x)
    #     = (1-y) * x + softplus(-x)  # More stable form

    loss_per_pos = (one(T) .- targets) .* logits .+ NNlib.softplus.(-logits)

    # Apply mask and scale
    loss_per_pos = loss_per_pos .* mask .* scale

    return sum(loss_per_pos) / max(sum(mask), one(T))
end

"""
    indel_only_loss(model::BranchingScoreNetwork, batch;
                    split_weight::Float32=1.0f0,
                    del_weight::Float32=1.0f0)

Compute only split/deletion losses (for stage 1 training with frozen base).
"""
function indel_only_loss(model::BranchingScoreNetwork, batch;
                          split_weight::Float32=1.0f0,
                          del_weight::Float32=1.0f0)

    # Build model input
    input = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => batch.xt_ca, :local_latents => batch.xt_ll),
        :t => Dict(:bb_ca => batch.t, :local_latents => batch.t),
        :mask => batch.mask
    )

    # Forward pass (with Flux.Zygote.ignore for base model if needed)
    output = model(input)

    split_pred = output[:split]
    del_pred = output[:del]

    # Combined mask
    combined_mask = batch.padmask .* batch.branchmask

    # Time scaling
    t_scale = 1f0 ./ max.(1f0 .- batch.t, 1f-5).^2
    t_scale_2d = reshape(t_scale, 1, :)

    # Only branching losses
    split_loss = bregman_poisson_loss(split_pred, batch.splits_target,
                                       combined_mask, t_scale_2d)
    del_loss = logistic_bce_loss(del_pred, batch.del_target,
                                  combined_mask, t_scale_2d)

    return split_weight * split_loss + del_weight * del_loss
end

"""
    staged_training_step!(model::BranchingScoreNetwork, batch, opt_state;
                          stage::Int=1,
                          split_weight::Float32=1.0f0,
                          del_weight::Float32=1.0f0)

Perform one training step with staged training support.

Stage 1: Only train indel heads (base frozen)
Stage 2: Train everything

# Returns
Loss value
"""
function staged_training_step!(model::BranchingScoreNetwork, batch, opt_state;
                                stage::Int=1,
                                split_weight::Float32=1.0f0,
                                del_weight::Float32=1.0f0)

    if stage == 1
        # Stage 1: Only train indel heads
        loss, grads = Flux.withgradient(model) do m
            indel_only_loss(m, batch; split_weight=split_weight, del_weight=del_weight)
        end

        # Only update indel parameters
        # This requires careful handling - we mask the gradients for base params
        # For now, assume opt_state only contains indel params
        Flux.update!(opt_state, model, grads[1])

    else
        # Stage 2: Train everything
        loss, grads = Flux.withgradient(model) do m
            branching_flow_loss(m, batch; split_weight=split_weight, del_weight=del_weight)
        end
        Flux.update!(opt_state, model, grads[1])
    end

    return loss
end

"""
    freeze_base_in_state!(opt_state, model::BranchingScoreNetwork)

Freeze the base ScoreNetwork parameters in the optimizer state for stage 1 training.
Only the indel heads (split_head, del_head, indel_time_proj) will be trainable.

Use `thaw_base_in_state!` after stage 1 to unfreeze for full fine-tuning.
"""
function freeze_base_in_state!(opt_state, model::BranchingScoreNetwork)
    Optimisers.freeze!(opt_state.base)
    return opt_state
end

"""
    thaw_base_in_state!(opt_state, model::BranchingScoreNetwork)

Unfreeze the base ScoreNetwork parameters in the optimizer state for stage 2.
"""
function thaw_base_in_state!(opt_state, model::BranchingScoreNetwork)
    Optimisers.thaw!(opt_state.base)
    return opt_state
end

"""
    setup_optimizer(model::BranchingScoreNetwork, lr::Float64; freeze_base::Bool=false)

Create optimizer state for training.

# Arguments
- `model`: BranchingScoreNetwork
- `lr`: Learning rate
- `freeze_base`: If true, freeze the base model params (stage 1 training)
"""
function setup_optimizer(model::BranchingScoreNetwork, lr::Float64; freeze_base::Bool=false)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), model)
    if freeze_base
        freeze_base_in_state!(opt_state, model)
    end
    return opt_state
end
