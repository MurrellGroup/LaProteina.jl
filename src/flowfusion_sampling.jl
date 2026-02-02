# Flowfusion API integration for protein generation
# Uses gen() from Flowfusion.jl for sampling

using Flowfusion
using ForwardBackward: ContinuousState, tensor

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
"""
mutable struct MutableScoreNetworkWrapper
    score_net::ScoreNetwork
    L::Int
    B::Int
    self_cond::Bool
    x_sc::Union{Nothing, Tuple{Array{Float32,3}, Array{Float32,3}}}
end

function MutableScoreNetworkWrapper(score_net::ScoreNetwork, L::Int, B::Int; self_cond::Bool=true)
    MutableScoreNetworkWrapper(score_net, L, B, self_cond, nothing)
end

function (wrapper::MutableScoreNetworkWrapper)(t, Xₜ)
    # Unpack states
    X_ca, X_ll = Xₜ
    x_ca = tensor(X_ca)
    x_ll = tensor(X_ll)

    L = wrapper.L
    B = wrapper.B
    mask = ones(Float32, L, B)

    # Build batch dict
    t_vec = fill(Float32(t), B)
    batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => x_ca, :local_latents => x_ll),
        :t => Dict(:bb_ca => t_vec, :local_latents => t_vec),
        :mask => mask
    )

    # Add self-conditioning
    if wrapper.self_cond && !isnothing(wrapper.x_sc)
        batch[:x_sc] = Dict(:bb_ca => wrapper.x_sc[1], :local_latents => wrapper.x_sc[2])
    end

    # Call ScoreNetwork
    output = wrapper.score_net(batch)

    # Extract X̂₁
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

    return (ContinuousState(x1_ca), ContinuousState(x1_ll))
end

"""
    generate_with_flowfusion(score_net::ScoreNetwork, L::Int, B::Int;
        nsteps::Int=100,
        latent_dim::Int=8,
        self_cond::Bool=true,
        schedule_mode::Symbol=:power,
        schedule_p::Real=2.0)

Generate samples using Flowfusion's gen() API.

# Arguments
- `score_net`: Trained ScoreNetwork
- `L`: Sequence length
- `B`: Batch size (number of samples)
- `nsteps`: Number of integration steps
- `latent_dim`: Dimension of local latents
- `self_cond`: Whether to use self-conditioning
- `schedule_mode`: Time schedule (:uniform, :power, :log)
- `schedule_p`: Schedule parameter

# Returns
Dict with:
- :bb_ca => [3, L, B] generated CA coordinates
- :local_latents => [latent_dim, L, B] generated latents
- :mask => [L, B]
"""
function generate_with_flowfusion(score_net::ScoreNetwork, L::Int, B::Int;
        nsteps::Int=100,
        latent_dim::Int=8,
        self_cond::Bool=true,
        schedule_mode::Symbol=:power,
        schedule_p::Real=2.0)

    # Create RDNFlow processes
    P_ca = RDNFlow(3; zero_com=true)      # CA coords with zero COM
    P_ll = RDNFlow(latent_dim; zero_com=false)  # Latents without zero COM
    P = (P_ca, P_ll)

    # Sample initial noise
    x0_ca = sample_rdn_noise(P_ca, L, B)      # [3, L, B]
    x0_ll = sample_rdn_noise(P_ll, L, B)      # [latent_dim, L, B]

    # Wrap as ContinuousStates
    X0 = (ContinuousState(x0_ca), ContinuousState(x0_ll))

    # Create time steps
    steps = Float32.(get_schedule(schedule_mode, nsteps; p=schedule_p))

    # Create model wrapper
    model = MutableScoreNetworkWrapper(score_net, L, B; self_cond=self_cond)

    # Run generation with Flowfusion's gen()
    X_final = gen(P, X0, model, steps)

    # Extract final states
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

See generate_with_flowfusion for arguments.

# Returns
Dict with all-atom structure information.
"""
function sample_with_flowfusion(score_net::ScoreNetwork, decoder::DecoderTransformer,
        L::Int, B::Int;
        nsteps::Int=100,
        latent_dim::Int=8,
        self_cond::Bool=true,
        schedule_mode::Symbol=:power,
        schedule_p::Real=2.0)

    # Generate with Flowfusion
    flow_samples = generate_with_flowfusion(score_net, L, B;
        nsteps=nsteps,
        latent_dim=latent_dim,
        self_cond=self_cond,
        schedule_mode=schedule_mode,
        schedule_p=schedule_p)

    ca_coords = flow_samples[:bb_ca]
    latents = flow_samples[:local_latents]
    mask = flow_samples[:mask]

    # Decode to all-atom
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
