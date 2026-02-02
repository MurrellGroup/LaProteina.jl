# Inference and sampling utilities
# Port of product_space_flow_matcher.py full_simulation

"""
    get_schedule(mode::Symbol, nsteps::Int; p::Real=1.0)

Get time schedule for sampling steps.
Returns array of length nsteps+1 from 0 to 1.

# Arguments
- `mode`: :uniform, :power, or :log
- `nsteps`: Number of sampling steps
- `p`: Parameter for power/log schedules
"""
function get_schedule(mode::Symbol, nsteps::Int; p::Real=1.0)
    if mode == :uniform
        return Float32.(range(0, 1, length=nsteps+1))
    elseif mode == :power
        t = Float32.(range(0, 1, length=nsteps+1))
        return t .^ Float32(p)
    elseif mode == :log
        t = 1.0f0 .- Float32.(exp10.(range(-Float32(p), 0, length=nsteps+1)))
        t = reverse(t)
        t = t .- minimum(t)
        t = t ./ maximum(t)
        return t
    else
        error("Unknown schedule mode: $mode")
    end
end

"""
    get_gt(t::AbstractVector, mode::Symbol, param::Real; clamp_val::Real=1e5)

Compute noise injection schedule gt for SDE sampling.
Returns tensor of same length as t.
"""
function get_gt(t::AbstractVector{T}, mode::Symbol, param::Real; clamp_val::Real=1e5, eps::Real=1e-2) where T
    t = clamp.(t, T(0), T(1) - T(1e-5))

    if mode == :const
        gt = fill(T(param), length(t))
    elseif mode == Symbol("1-t/t")
        num = one(T) .- t
        den = t
        gt = num ./ (den .+ T(eps))
    elseif mode == :tan
        num = sin.((one(T) .- t) .* T(π / 2))
        den = cos.((one(T) .- t) .* T(π / 2))
        gt = T(π / 2) .* num ./ (den .+ T(eps))
    elseif mode == Symbol("1/t")
        gt = one(T) ./ (t .+ T(eps))
    else
        error("Unknown gt mode: $mode")
    end

    # Apply power transformation if param != 1
    if param != 1.0 && mode != :const
        log_gt = log.(gt)
        mean_log_gt = mean(log_gt)
        log_gt_centered = log_gt .- mean_log_gt
        normalized = sigmoid.(log_gt_centered)
        normalized = normalized .^ T(param)
        log_gt_centered_rec = _logit.(clamp.(normalized, T(1e-6), T(1-1e-6)))
        log_gt_rec = log_gt_centered_rec .+ mean_log_gt
        gt = exp.(log_gt_rec)
    end

    return clamp.(gt, T(0), T(clamp_val))
end

# Note: Using NNlib.sigmoid for sigmoid function
# logit is the inverse of sigmoid
_logit(x) = log(x / (1 - x))

"""
    sample_rdn_noise(dim::Int, L::Int, B::Int; zero_com::Bool=false, mask=nothing)

Sample from reference distribution (Gaussian, optionally zero-centered).
Returns array [dim, L, B].
"""
function sample_rdn_noise(dim::Int, L::Int, B::Int; zero_com::Bool=false, mask=nothing)
    noise = randn(Float32, dim, L, B)

    # Apply mask
    if !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        noise = noise .* mask_exp
    end

    # Zero center of mass if requested
    if zero_com
        noise = zero_center_of_mass(noise, mask; dims=2)
    end

    return noise
end

"""
    rdn_interpolate(x0, x1, t)

Linear interpolation: x_t = (1-t)*x_0 + t*x_1
"""
function rdn_interpolate(x0::AbstractArray{T}, x1::AbstractArray{T}, t) where T
    if isa(t, Number)
        return (one(T) - T(t)) .* x0 .+ T(t) .* x1
    else
        t_exp = reshape(T.(t), 1, 1, :)
        return (one(T) .- t_exp) .* x0 .+ t_exp .* x1
    end
end

"""
    rdn_ode_step(x_t, v, dt; mask=nothing, center::Bool=false)

ODE integration step: x_{t+dt} = x_t + v * dt
"""
function rdn_ode_step(x_t::AbstractArray{T}, v::AbstractArray{T}, dt::Real; mask=nothing, center::Bool=false) where T
    x_next = x_t .+ T(dt) .* v

    if !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        x_next = x_next .* mask_exp
    end

    if center
        x_next = zero_center_of_mass(x_next, mask; dims=2)
    end

    return x_next
end

"""
    vf_to_score(x_t, v, t)

Convert velocity field to score.
s(x_t, t) = (t * v(x_t, t) - x_t) / (1 - t)
"""
function vf_to_score(x_t::AbstractArray{T}, v::AbstractArray{T}, t) where T
    if isa(t, Number)
        return (T(t) .* v .- x_t) ./ (one(T) - T(t))
    else
        t_exp = reshape(T.(t), 1, 1, :)
        return (t_exp .* v .- x_t) ./ (one(T) .- t_exp)
    end
end

"""
    score_to_vf(x_t, score, t)

Convert score to velocity field.
v(x_t, t) = (x_t + (1-t) * score) / t
"""
function score_to_vf(x_t::AbstractArray{T}, score::AbstractArray{T}, t) where T
    if isa(t, Number)
        return (x_t .+ (one(T) - T(t)) .* score) ./ T(t)
    else
        t_exp = reshape(T.(t), 1, 1, :)
        return (x_t .+ (one(T) .- t_exp) .* score) ./ t_exp
    end
end

"""
    rdn_sde_step(x_t, v, t, dt, gt; mask=nothing, center::Bool=false,
                 sc_scale_noise::Real=1.0, sc_scale_score::Real=1.0)

SDE integration step with optional noise injection.
d x_t = [v(x_t, t) + g(t) * s(x_t, t)] dt + sqrt(2 * g(t)) dw_t
"""
function rdn_sde_step(x_t::AbstractArray{T}, v::AbstractArray{T}, t, dt::Real, gt::Real;
        mask=nothing, center::Bool=false, sc_scale_noise::Real=1.0, sc_scale_score::Real=1.0) where T

    t_scalar = isa(t, Number) ? t : t[1]

    # If gt is very small, use ODE
    if gt < 1e-5
        return rdn_ode_step(x_t, v, dt; mask=mask, center=center)
    end

    # Compute score from velocity
    score = vf_to_score(x_t, v, t)

    # Scale score
    scaled_score = score .* T(sc_scale_score)

    # Noise
    eps = randn(T, size(x_t))
    std_eps = sqrt(T(2 * gt * sc_scale_noise * dt))

    # SDE step
    x_next = x_t .+ (v .+ T(gt) .* scaled_score) .* T(dt) .+ std_eps .* eps

    if !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        x_next = x_next .* mask_exp
    end

    if center
        x_next = zero_center_of_mass(x_next, mask; dims=2)
    end

    return x_next
end

# ============================================================================
# Main Sampling Functions
# ============================================================================

"""
    full_simulation(score_net::ScoreNetwork, L::Int, B::Int;
        nsteps::Int=100,
        latent_dim::Int=8,
        self_cond::Bool=true,
        schedule_mode::Symbol=:power,
        schedule_p::Real=2.0,
        gt_mode::Symbol=:const,
        gt_param::Real=0.0,
        gt_clamp::Real=1e5,
        sampling_mode::Symbol=:vf,
        center_ca::Bool=true,
        mask=nothing,
        device=identity)

Generate samples using flow matching ODE/SDE simulation.
Matches Python ProductSpaceFlowMatcher.full_simulation.

# Arguments
- `score_net`: Trained ScoreNetwork
- `L`: Sequence length
- `B`: Batch size (number of samples)
- `nsteps`: Number of integration steps
- `latent_dim`: Dimension of local latents
- `self_cond`: Whether to use self-conditioning
- `schedule_mode`: Time schedule (:uniform, :power, :log)
- `schedule_p`: Schedule parameter
- `gt_mode`: Noise injection mode (:const, :tan, etc.)
- `gt_param`: Noise injection parameter
- `sampling_mode`: :vf (ODE), :sc (SDE with noise scaling)
- `center_ca`: Whether to zero-center CA coordinates each step
- `mask`: Optional mask [L, B]

# Returns
Dict with:
- :bb_ca => [3, L, B] generated CA coordinates
- :local_latents => [latent_dim, L, B] generated latents
"""
function full_simulation(score_net::ScoreNetwork, L::Int, B::Int;
        nsteps::Int=100,
        latent_dim::Int=8,
        self_cond::Bool=true,
        schedule_mode::Symbol=:power,
        schedule_p::Real=2.0,
        gt_mode::Symbol=:const,
        gt_param::Real=0.0,
        gt_clamp::Real=1e5,
        sampling_mode::Symbol=:vf,
        center_ca::Bool=true,
        mask=nothing,
        device=identity)

    # Initialize mask
    if isnothing(mask)
        mask = ones(Float32, L, B)
    end

    # Get time schedule
    ts = get_schedule(schedule_mode, nsteps; p=schedule_p)

    # Get noise injection schedule (if using SDE)
    gt = sampling_mode == :vf ? zeros(Float32, nsteps) : get_gt(ts[1:end-1], gt_mode, gt_param; clamp_val=gt_clamp)

    # Sample initial noise
    x_ca = sample_rdn_noise(3, L, B; zero_com=center_ca, mask=mask)
    x_latents = sample_rdn_noise(latent_dim, L, B; zero_com=false, mask=mask)

    # Self-conditioning state
    x1_pred_ca = nothing
    x1_pred_ll = nothing

    # Main simulation loop
    for step in 1:nsteps
        t_curr = ts[step]
        t_next = ts[step + 1]
        dt = t_next - t_curr
        gt_step = gt[step]

        # Build batch for score network
        t_vec_ca = fill(Float32(t_curr), B)
        t_vec_ll = fill(Float32(t_curr), B)

        batch = Dict{Symbol, Any}(
            :x_t => Dict(:bb_ca => x_ca, :local_latents => x_latents),
            :t => Dict(:bb_ca => t_vec_ca, :local_latents => t_vec_ll),
            :mask => mask
        )

        # Add self-conditioning
        if self_cond && step > 1 && !isnothing(x1_pred_ca)
            batch[:x_sc] = Dict(:bb_ca => x1_pred_ca, :local_latents => x1_pred_ll)
        end

        # Get network prediction
        output = score_net(batch)

        # Extract velocity or x1
        if score_net.output_param == :v
            v_ca = output[:bb_ca][:v]
            v_ll = output[:local_latents][:v]
            x1_pred_ca = v_to_x1(x_ca, v_ca, t_curr)
            x1_pred_ll = v_to_x1(x_latents, v_ll, t_curr)
        else
            x1_pred_ca = output[:bb_ca][:x1]
            x1_pred_ll = output[:local_latents][:x1]
            v_ca = x1_to_v(x_ca, x1_pred_ca, t_curr)
            v_ll = x1_to_v(x_latents, x1_pred_ll, t_curr)
        end

        # Integration step
        if sampling_mode == :vf
            x_ca = rdn_ode_step(x_ca, v_ca, dt; mask=mask, center=center_ca)
            x_latents = rdn_ode_step(x_latents, v_ll, dt; mask=mask, center=false)
        else
            x_ca = rdn_sde_step(x_ca, v_ca, t_curr, dt, gt_step; mask=mask, center=center_ca)
            x_latents = rdn_sde_step(x_latents, v_ll, t_curr, dt, gt_step; mask=mask, center=false)
        end
    end

    return Dict(
        :bb_ca => x_ca,
        :local_latents => x_latents,
        :mask => mask
    )
end

"""
    sample(score_net::ScoreNetwork, decoder::DecoderTransformer,
           L::Int, B::Int;
           nsteps::Int=100,
           latent_dim::Int=8,
           self_cond::Bool=true,
           schedule_mode::Symbol=:power,
           schedule_p::Real=2.0,
           sampling_mode::Symbol=:vf,
           mask=nothing)

Generate protein structures using flow matching and VAE decoder.

# Arguments
- `score_net`: Trained ScoreNetwork
- `decoder`: Trained DecoderTransformer
- `L`: Sequence length
- `B`: Batch size (number of samples)
- `nsteps`: Number of integration steps
- Other args passed to full_simulation

# Returns
Dict with:
- :ca_coords => [3, L, B] generated CA coordinates
- :latents => [latent_dim, L, B] generated latents
- :seq_logits => [20, L, B] sequence logits from decoder
- :all_atom_coords => [3, 37, L, B] all-atom coordinates from decoder
- :aatype => [L, B] predicted amino acid types
- :atom_mask => [37, L, B] atom mask
"""
function sample(score_net::ScoreNetwork, decoder::DecoderTransformer,
                L::Int, B::Int;
                nsteps::Int=100,
                latent_dim::Int=8,
                self_cond::Bool=true,
                schedule_mode::Symbol=:power,
                schedule_p::Real=2.0,
                sampling_mode::Symbol=:vf,
                mask=nothing)

    # Generate flow matching samples
    flow_samples = full_simulation(score_net, L, B;
        nsteps=nsteps,
        latent_dim=latent_dim,
        self_cond=self_cond,
        schedule_mode=schedule_mode,
        schedule_p=schedule_p,
        sampling_mode=sampling_mode,
        center_ca=true,
        mask=mask)

    ca_coords = flow_samples[:bb_ca]
    latents = flow_samples[:local_latents]
    mask = flow_samples[:mask]

    # Decode to get all-atom structure
    dec_input = Dict(
        :z_latent => latents,
        :ca_coors => ca_coords,
        :mask => mask
    )
    dec_out = decoder(dec_input)

    return Dict(
        :ca_coords => ca_coords,
        :latents => latents,
        :seq_logits => dec_out[:seq_logits],
        :all_atom_coords => dec_out[:coors],
        :aatype => dec_out[:aatype_max],
        :atom_mask => dec_out[:atom_mask],
        :mask => mask
    )
end

"""
    samples_to_pdb(samples::Dict, output_dir::String;
                   prefix::String="sample",
                   save_all_atom::Bool=true)

Save generated samples as PDB files.

# Arguments
- `samples`: Dict from sample() function
- `output_dir`: Directory to save PDB files
- `prefix`: Filename prefix
- `save_all_atom`: If true, save all-atom coords; if false, save CA only
"""
function samples_to_pdb(samples::Dict, output_dir::String;
                        prefix::String="sample",
                        save_all_atom::Bool=true)

    mkpath(output_dir)

    B = size(samples[:ca_coords], 3)

    for b in 1:B
        filename = joinpath(output_dir, "$(prefix)_$(b).pdb")

        if save_all_atom && haskey(samples, :all_atom_coords)
            coords = samples[:all_atom_coords][:, :, :, b]  # [3, 37, L]
            aatype = samples[:aatype][:, b]                  # [L]
            atom_mask = samples[:atom_mask][:, :, b]         # [37, L]
            save_pdb(filename, coords, aatype; atom_mask=atom_mask)
        else
            # CA-only PDB (simplified)
            ca_coords = samples[:ca_coords][:, :, b]  # [3, L]
            L = size(ca_coords, 2)

            # Create dummy all-atom (just CA)
            coords = zeros(Float32, 3, 37, L)
            coords[:, CA_INDEX, :] = ca_coords
            aatype = haskey(samples, :aatype) ? samples[:aatype][:, b] : fill(1, L)
            atom_mask = zeros(Bool, 37, L)
            atom_mask[CA_INDEX, :] .= true

            save_pdb(filename, coords, aatype; atom_mask=atom_mask)
        end

        @info "Saved $(filename)"
    end
end

# ============================================================================
# Flow Matching Loss Functions
# ============================================================================

"""
    fm_loss(x0, x1, t, x1_pred; mask=nothing)

Compute flow matching loss with 1/(1-t)^2 weighting.
"""
function fm_loss(x0::AbstractArray{T}, x1::AbstractArray{T}, t, x1_pred::AbstractArray{T}; mask=nothing) where T
    # x_t = (1-t)*x0 + t*x1
    # Target is x1

    if !isnothing(mask)
        mask_exp = reshape(mask, 1, size(mask)...)
        x1 = x1 .* mask_exp
        x1_pred = x1_pred .* mask_exp
    end

    err = x1 .- x1_pred
    err_sq = err .^ 2

    # Sum over spatial dims, mean over sequence
    if !isnothing(mask)
        nres = sum(mask; dims=1)  # [1, B]
        loss_per_sample = dropdims(sum(err_sq; dims=(1, 2)); dims=(1, 2)) ./ max.(dropdims(nres; dims=1), one(T))
    else
        loss_per_sample = dropdims(mean(err_sq; dims=(1, 2)); dims=(1, 2))
    end

    # Apply 1/(1-t)^2 weighting
    if isa(t, Number)
        weight = one(T) / ((one(T) - T(t))^2 + T(1e-5))
        return loss_per_sample .* weight
    else
        t_exp = T.(t)
        weight = one(T) ./ ((one(T) .- t_exp).^2 .+ T(1e-5))
        return loss_per_sample .* weight
    end
end
