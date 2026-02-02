# Training utilities for VAE and Flow Matching

using Flux
using Flowfusion
using ForwardBackward

"""
    train_vae!(model::Autoencoder, train_loader, opt_state;
        n_epochs::Int=1,
        kl_weight::Float32=1.0f0,
        coord_weight::Float32=1.0f0,
        seq_weight::Float32=1.0f0,
        log_every::Int=10)

Train VAE autoencoder.

# Arguments
- `model`: Autoencoder model
- `train_loader`: Iterator over batches (each batch is a Dict)
- `opt_state`: Flux optimizer state
- `n_epochs`: Number of training epochs
- `kl_weight`, `coord_weight`, `seq_weight`: Loss weights
- `log_every`: Log every N iterations

# Returns
Vector of loss values
"""
function train_vae!(model::Autoencoder, train_loader, opt_state;
        n_epochs::Int=1,
        kl_weight::Float32=1.0f0,
        coord_weight::Float32=1.0f0,
        seq_weight::Float32=1.0f0,
        log_every::Int=10)

    losses = Float32[]

    for epoch in 1:n_epochs
        for (i, batch) in enumerate(train_loader)
            # Compute loss and gradients
            l, grads = Flux.withgradient(model) do m
                loss_dict = vae_loss(m, batch;
                    kl_weight=kl_weight,
                    coord_weight=coord_weight,
                    seq_weight=seq_weight)
                loss_dict.total
            end

            # Update parameters
            Flux.update!(opt_state, model, grads[1])

            push!(losses, l)

            if i % log_every == 0
                @info "Epoch $epoch, Iter $i: Loss = $l"
            end
        end
    end

    return losses
end

"""
    train_flow!(model::ScoreNetwork, train_loader, opt_state, process;
        n_epochs::Int=1,
        t_sampler::Function=sample_t_uniform,
        self_conditioning::Bool=false,
        sc_prob::Float32=0.5f0,
        log_every::Int=10)

Train flow matching score network.

# Arguments
- `model`: ScoreNetwork model
- `train_loader`: Iterator over batches
- `opt_state`: Flux optimizer state
- `process`: Tuple of (RDNFlow for CA, RDNFlow for latents)
- `n_epochs`: Number of training epochs
- `t_sampler`: Function to sample times, signature (n::Int) -> Vector{Float32}
- `self_conditioning`: Whether to use self-conditioning
- `sc_prob`: Probability of using self-conditioning (vs zeros)
- `log_every`: Log every N iterations

# Returns
Vector of loss values
"""
function train_flow!(model::ScoreNetwork, train_loader, opt_state, process;
        n_epochs::Int=1,
        t_sampler::Function=sample_t_uniform,
        self_conditioning::Bool=false,
        sc_prob::Float32=0.5f0,
        log_every::Int=10)

    P_ca, P_latents = process
    losses = Float32[]

    for epoch in 1:n_epochs
        for (i, batch) in enumerate(train_loader)
            # Get target data (X1)
            ca_coords = get_ca_from_batch(batch)      # [3, L, B]
            local_latents = batch[:z_latent]          # [latent_dim, L, B]
            mask = get(batch, :mask, nothing)
            B = size(ca_coords, 3)
            L = size(ca_coords, 2)

            # Sample times
            t = t_sampler(B)

            # Sample noise (X0)
            x0_ca = sample_rdn_noise(P_ca, L, B; mask=mask)
            x0_latents = sample_rdn_noise(P_latents, L, B; mask=mask)

            # Bridge to get X_t
            X0_ca = ContinuousState(x0_ca)
            X1_ca = ContinuousState(ca_coords)
            Xt_ca = bridge(P_ca, X0_ca, X1_ca, t)

            X0_ll = ContinuousState(x0_latents)
            X1_ll = ContinuousState(local_latents)
            Xt_ll = bridge(P_latents, X0_ll, X1_ll, t)

            x_t = Dict(
                :bb_ca => tensor(Xt_ca),
                :local_latents => tensor(Xt_ll)
            )

            # Self-conditioning (optional)
            if self_conditioning && rand() < sc_prob
                # Run model once without gradients to get self-cond
                with_no_grad() do
                    input_sc = Dict(:x_t => x_t, :t => t, :mask => mask)
                    prev_out = model(input_sc)
                end
                input = Dict(
                    :x_t => x_t,
                    :t => t,
                    :mask => mask,
                    :self_cond => prev_out
                )
            else
                input = Dict(:x_t => x_t, :t => t, :mask => mask)
            end

            # Compute loss and gradients
            l, grads = Flux.withgradient(model) do m
                flow_loss(m, input, (ca_coords, local_latents), t, (P_ca, P_latents), mask)
            end

            # Update parameters
            Flux.update!(opt_state, model, grads[1])

            push!(losses, l)

            if i % log_every == 0
                @info "Epoch $epoch, Iter $i: Flow Loss = $l"
            end
        end
    end

    return losses
end

"""
    flow_loss(model::ScoreNetwork, input::Dict, X1_targets::Tuple, t, process, mask)

Compute flow matching loss.

Uses scalefloss from Flowfusion for 1/(1-t)^2 weighting.
"""
function flow_loss(model::ScoreNetwork, input::Dict, X1_targets::Tuple, t, process, mask)
    ca_target, ll_target = X1_targets
    P_ca, P_ll = process

    # Forward pass
    output = model(input)

    # Get predictions
    out_key = model.output_param  # :v or :x1

    if out_key == :v
        # Convert velocity to x1 prediction
        x_t = input[:x_t]
        ca_pred = v_to_x1(x_t[:bb_ca], output[:bb_ca][:v], t)
        ll_pred = v_to_x1(x_t[:local_latents], output[:local_latents][:v], t)
    else
        ca_pred = output[:bb_ca][:x1]
        ll_pred = output[:local_latents][:x1]
    end

    # Loss scaling
    scale = scalefloss((P_ca, P_ll), t)

    # MSE losses
    T = eltype(ca_pred)
    if isnothing(mask)
        mask = ones(T, size(ca_target, 2), size(ca_target, 3))
    end

    # CA loss
    ca_loss = compute_scaled_mse(ca_pred, ca_target, mask, scale[1])

    # Latent loss
    ll_loss = compute_scaled_mse(ll_pred, ll_target, mask, scale[2])

    return ca_loss + ll_loss
end

"""
    compute_scaled_mse(pred, target, mask, scale)

Compute scaled masked MSE loss.
"""
function compute_scaled_mse(pred::AbstractArray{T}, target::AbstractArray{T}, mask, scale) where T
    diff_sq = (pred .- target) .^ 2  # [D, L, B]

    # Sum over feature dim
    err_per_pos = sum(diff_sq; dims=1)  # [1, L, B]
    err_per_pos = dropdims(err_per_pos; dims=1)  # [L, B]

    # Apply mask
    err_per_pos = err_per_pos .* mask

    # Scale by 1/(1-t)^2
    scale_exp = reshape(scale, 1, length(scale))  # [1, B]
    scaled_err = err_per_pos .* scale_exp

    # Mean over valid positions
    n_valid = max(sum(mask), one(T))
    return sum(scaled_err) / n_valid
end

"""
Helper to get CA coords from batch.
"""
function get_ca_from_batch(batch::Dict)
    if haskey(batch, :ca_coors)
        return batch[:ca_coors]
    elseif haskey(batch, :coords)
        return batch[:coords][:, CA_INDEX, :, :]
    else
        error("Cannot find CA coordinates in batch")
    end
end

# Stub for no-gradient execution
function with_no_grad(f)
    # In a full implementation, this would use Flux.Zygote.ignore
    # For now, just run the function (gradients will be computed but discarded)
    return f()
end
