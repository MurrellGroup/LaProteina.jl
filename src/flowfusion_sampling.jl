# Flowfusion API integration for protein generation
# Uses gen() from Flowfusion.jl for sampling

import Flowfusion
import Flowfusion: RDNFlow, gen
using ForwardBackward: ContinuousState, tensor
using Flux: cpu, gpu

"""
    ScoreNetworkWrapper

Wrapper that adapts ScoreNetwork to Flowfusion's gen() API.
gen() expects model(t, Xₜ) -> X̂₁ where Xₜ is a tuple of ContinuousStates.
"""
struct ScoreNetworkWrapper
    score_net::ScoreNetwork
    L::Int                     # Sequence length
    B::Int                     # Batch size
    self_cond::Bool            # Whether to use self-conditioning
    x_sc::Union{Nothing, Tuple{Array{Float32,3}, Array{Float32,3}}}  # Self-conditioning state
end

function ScoreNetworkWrapper(score_net::ScoreNetwork, L::Int, B::Int; self_cond::Bool=true)
    ScoreNetworkWrapper(score_net, L, B, self_cond, nothing)
end

"""
    (wrapper::ScoreNetworkWrapper)(t, Xₜ)

Model function for gen(). Takes time t (scalar) and Xₜ (tuple of ContinuousStates),
returns tuple of X̂₁ predictions as ContinuousStates.
"""
function (wrapper::ScoreNetworkWrapper)(t, Xₜ)
    # Unpack states
    X_ca, X_ll = Xₜ
    x_ca = tensor(X_ca)      # [3, L, B]
    x_ll = tensor(X_ll)      # [8, L, B]

    L = wrapper.L
    B = wrapper.B
    mask = ones(Float32, L, B)

    # Build batch dict for ScoreNetwork
    t_vec = fill(Float32(t), B)
    batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => x_ca, :local_latents => x_ll),
        :t => Dict(:bb_ca => t_vec, :local_latents => t_vec),
        :mask => mask
    )

    # Add self-conditioning if available
    if wrapper.self_cond && !isnothing(wrapper.x_sc)
        batch[:x_sc] = Dict(:bb_ca => wrapper.x_sc[1], :local_latents => wrapper.x_sc[2])
    end

    # Call ScoreNetwork
    output = wrapper.score_net(batch)

    # Extract X̂₁ predictions
    if wrapper.score_net.output_param == :v
        v_ca = output[:bb_ca][:v]
        v_ll = output[:local_latents][:v]
        x1_ca = v_to_x1(x_ca, v_ca, t)
        x1_ll = v_to_x1(x_ll, v_ll, t)
    else
        x1_ca = output[:bb_ca][:x1]
        x1_ll = output[:local_latents][:x1]
    end

    # Update self-conditioning state
    if wrapper.self_cond
        wrapper.x_sc = (x1_ca, x1_ll)
    end

    # Return as tuple of ContinuousStates
    return (ContinuousState(x1_ca), ContinuousState(x1_ll))
end

"""
    MutableScoreNetworkWrapper

Mutable wrapper to allow self-conditioning state updates during gen().
Supports GPU acceleration via device function.
"""
mutable struct MutableScoreNetworkWrapper{D}
    score_net::ScoreNetwork
    L::Int
    B::Int
    self_cond::Bool
    dev::D  # Device function: gpu or identity for CPU
    x_sc::Union{Nothing, Tuple{AbstractArray{Float32,3}, AbstractArray{Float32,3}}}
end

function MutableScoreNetworkWrapper(score_net::ScoreNetwork, L::Int, B::Int;
                                     self_cond::Bool=true, dev=identity)
    MutableScoreNetworkWrapper(score_net, L, B, self_cond, dev, nothing)
end

function (wrapper::MutableScoreNetworkWrapper)(t, Xₜ)
    # Unpack states - these come from Flowfusion on CPU
    X_ca, X_ll = Xₜ
    x_ca = tensor(X_ca)  # [3, L, B] on CPU
    x_ll = tensor(X_ll)  # [8, L, B] on CPU

    L = wrapper.L
    B = wrapper.B
    dev = wrapper.dev

    # Create mask and time vector
    mask = ones(Float32, L, B)
    t_vec = fill(Float32(t), B)

    # Move data to device for model forward pass
    x_ca_dev = dev(x_ca)
    x_ll_dev = dev(x_ll)
    mask_dev = dev(mask)
    t_vec_dev = dev(t_vec)

    # Build batch dict on device
    batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => x_ca_dev, :local_latents => x_ll_dev),
        :t => Dict(:bb_ca => t_vec_dev, :local_latents => t_vec_dev),
        :mask => mask_dev
    )

    # Add self-conditioning on device
    if wrapper.self_cond && !isnothing(wrapper.x_sc)
        batch[:x_sc] = Dict(:bb_ca => wrapper.x_sc[1], :local_latents => wrapper.x_sc[2])
    end

    # Call ScoreNetwork on device
    output = wrapper.score_net(batch)

    # Extract X̂₁ on device
    if wrapper.score_net.output_param == :v
        v_ca = output[:bb_ca][:v]
        v_ll = output[:local_latents][:v]
        x1_ca_dev = v_to_x1(x_ca_dev, v_ca, t)
        x1_ll_dev = v_to_x1(x_ll_dev, v_ll, t)
    else
        x1_ca_dev = output[:bb_ca][:x1]
        x1_ll_dev = output[:local_latents][:x1]
    end

    # Update self-conditioning state (keep on device for next iteration)
    if wrapper.self_cond
        wrapper.x_sc = (x1_ca_dev, x1_ll_dev)
    end

    # Move back to CPU for Flowfusion's step function
    x1_ca_cpu = cpu(x1_ca_dev)
    x1_ll_cpu = cpu(x1_ll_dev)

    return (ContinuousState(x1_ca_cpu), ContinuousState(x1_ll_cpu))
end

"""
    generate_with_flowfusion(score_net::ScoreNetwork, L::Int, B::Int; kwargs...)

Generate samples using Flowfusion's gen() API with la-proteina default settings.

# Arguments
- `score_net`: Trained ScoreNetwork (can be on GPU)
- `L`: Sequence length
- `B`: Batch size (number of samples)

# Keyword Arguments
- `nsteps::Int=400`: Number of integration steps
- `latent_dim::Int=8`: Dimension of local latents
- `self_cond::Bool=true`: Whether to use self-conditioning
- `dev=identity`: Device function (use `gpu` for GPU acceleration)

## BB_CA settings (backbone CA coordinates):
- `ca_schedule_mode::Symbol=:log`: Time schedule for CA
- `ca_schedule_p::Real=2.0`: Schedule parameter
- `ca_gt_mode::Symbol=Symbol("1/t")`: Noise schedule mode
- `ca_gt_param::Real=1.0`: Base noise parameter
- `ca_sc_scale_noise::Real=0.1`: Noise scaling (0 = ODE)
- `ca_sc_scale_score::Real=1.0`: Score scaling
- `ca_t_lim_ode::Real=0.98`: Switch to ODE above this t

## Local latents settings:
- `ll_schedule_mode::Symbol=:power`: Time schedule for latents
- `ll_schedule_p::Real=2.0`: Schedule parameter
- `ll_gt_mode::Symbol=:tan`: Noise schedule mode
- `ll_gt_param::Real=1.0`: Base noise parameter
- `ll_sc_scale_noise::Real=0.1`: Noise scaling (0 = ODE)
- `ll_sc_scale_score::Real=1.0`: Score scaling
- `ll_t_lim_ode::Real=0.98`: Switch to ODE above this t

# Returns
Dict with:
- :bb_ca => [3, L, B] generated CA coordinates (on CPU)
- :local_latents => [latent_dim, L, B] generated latents (on CPU)
- :mask => [L, B]

# Example
```julia
# CPU sampling (default)
samples = generate_with_flowfusion(score_net, 100, 3)

# GPU sampling
score_net_gpu = score_net |> gpu
samples = generate_with_flowfusion(score_net_gpu, 100, 3; dev=gpu)

# ODE-only sampling (no noise)
samples = generate_with_flowfusion(score_net, 100, 3;
    ca_sc_scale_noise=0.0,
    ll_sc_scale_noise=0.0
)
```
"""
function generate_with_flowfusion(score_net::ScoreNetwork, L::Int, B::Int;
        nsteps::Int=400,
        latent_dim::Int=8,
        self_cond::Bool=true,
        dev=identity,
        # BB_CA settings (la-proteina defaults)
        ca_schedule_mode::Symbol=:log,
        ca_schedule_p::Real=2.0,
        ca_gt_mode::Symbol=Symbol("1/t"),
        ca_gt_param::Real=1.0,
        ca_sc_scale_noise::Real=0.1,
        ca_sc_scale_score::Real=1.0,
        ca_t_lim_ode::Real=0.98,
        # Local latents settings (la-proteina defaults)
        ll_schedule_mode::Symbol=:power,
        ll_schedule_p::Real=2.0,
        ll_gt_mode::Symbol=:tan,
        ll_gt_param::Real=1.0,
        ll_sc_scale_noise::Real=0.1,
        ll_sc_scale_score::Real=1.0,
        ll_t_lim_ode::Real=0.98)

    # Create RDNFlow processes with per-modality SDE parameters
    P_ca = RDNFlow(3;
        zero_com=true,
        sde_gt_mode=ca_gt_mode,
        sde_gt_param=ca_gt_param,
        sc_scale_noise=ca_sc_scale_noise,
        sc_scale_score=ca_sc_scale_score,
        t_lim_ode=ca_t_lim_ode
    )
    P_ll = RDNFlow(latent_dim;
        zero_com=false,
        sde_gt_mode=ll_gt_mode,
        sde_gt_param=ll_gt_param,
        sc_scale_noise=ll_sc_scale_noise,
        sc_scale_score=ll_sc_scale_score,
        t_lim_ode=ll_t_lim_ode
    )
    P = (P_ca, P_ll)

    # Sample initial noise using Flowfusion's RDN noise sampler (on CPU)
    x0_ca = Flowfusion.sample_rdn_noise(P_ca, L, B)      # [3, L, B]
    x0_ll = Flowfusion.sample_rdn_noise(P_ll, L, B)      # [latent_dim, L, B]

    # Wrap as ContinuousStates (on CPU - Flowfusion operates on CPU)
    X0 = (ContinuousState(x0_ca), ContinuousState(x0_ll))

    # Create time steps - use CA schedule (both modalities use same time steps in gen)
    # Note: la-proteina uses different schedules per modality, but gen() uses shared steps
    # We use CA schedule as primary since coordinates are more sensitive
    steps = Float32.(get_schedule(ca_schedule_mode, nsteps; p=ca_schedule_p))

    # Create model wrapper with device function
    model = MutableScoreNetworkWrapper(score_net, L, B; self_cond=self_cond, dev=dev)

    # Run generation with Flowfusion's gen()
    # gen() operates on CPU, model wrapper handles GPU transfer internally
    X_final = gen(P, X0, model, steps)

    # Extract final states (already on CPU from gen())
    x_ca = tensor(X_final[1])
    x_ll = tensor(X_final[2])

    mask = ones(Float32, L, B)

    return Dict(
        :bb_ca => x_ca,
        :local_latents => x_ll,
        :mask => mask
    )
end

"""
    sample_with_flowfusion(score_net::ScoreNetwork, decoder::DecoderTransformer,
        L::Int, B::Int; kwargs...)

Generate protein structures using Flowfusion's gen() API and decode to all-atom.

See generate_with_flowfusion for all available keyword arguments.

# Returns
Dict with all-atom structure information (all on CPU).
"""
function sample_with_flowfusion(score_net::ScoreNetwork, decoder::DecoderTransformer,
        L::Int, B::Int; dev=identity, kwargs...)

    # Generate with Flowfusion - pass through all kwargs
    flow_samples = generate_with_flowfusion(score_net, L, B; dev=dev, kwargs...)

    ca_coords = flow_samples[:bb_ca]
    latents = flow_samples[:local_latents]
    mask = flow_samples[:mask]

    # Decode to all-atom (move to device if needed)
    dec_input = Dict(
        :z_latent => dev(latents),
        :ca_coors => dev(ca_coords),
        :mask => dev(mask)
    )
    dec_out = decoder(dec_input)

    # Move outputs to CPU
    return Dict(
        :ca_coords => ca_coords,
        :latents => latents,
        :seq_logits => cpu(dec_out[:seq_logits]),
        :all_atom_coords => cpu(dec_out[:coors]),
        :aatype => cpu(dec_out[:aatype_max]),
        :atom_mask => cpu(dec_out[:atom_mask]),
        :mask => mask
    )
end
