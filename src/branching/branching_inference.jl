# Inference utilities for Branching Flows generation
# Includes self-conditioning wrapper that handles variable-length states

using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, DiscreteState, tensor
using Flowfusion: MaskedState, RDNFlow, element, schedule_transform, sample_rdn_noise
import Flowfusion
import Flowfusion: gen
using Flux: cpu, gpu
using Distributions: Beta, Poisson

# NullProcess for tracking indices through splits/deletions (BranchChain pattern)
struct NullProcess <: Flowfusion.Process end
Flowfusion.endpoint_conditioned_sample(Xa, Xc, p::NullProcess, t_a, t_b, t_c) = Xa
Flowfusion.step(P::NullProcess, Xₜ::Flowfusion.MaskedState, X1targets, s₁, s₂) = Xₜ

"""
    BranchingScoreNetworkWrapper

Mutable wrapper for BranchingScoreNetwork that handles self-conditioning
with variable-length states during generation.

When splits/deletions occur, the wrapper uses the index state to expand/contract
the self-conditioning predictions to match the new state size.

The wrapper applies per-modality schedule transforms to convert raw uniform progress
to the actual interpolation times the model was trained on, using the processes tuple.

The wrapper returns predictions in the format expected by CoalescentFlow.step:
    (X1_targets, split_logits, del_logits)
"""
mutable struct BranchingScoreNetworkWrapper{M, D, P}
    model::M                    # BranchingScoreNetwork
    dev::D                      # Device function (gpu or identity)
    latent_dim::Int
    self_cond::Bool
    processes::P                # (P_ca, P_ll) for schedule_transform
    # Self-conditioning state (stored from previous step)
    sc_ca::Union{Nothing, AbstractArray}
    sc_ll::Union{Nothing, AbstractArray}
    # Index tracking for self-conditioning expansion
    prev_L::Int                 # Previous sequence length
end

function BranchingScoreNetworkWrapper(model, latent_dim::Int;
                                       self_cond::Bool=true, dev=identity,
                                       processes=nothing)
    BranchingScoreNetworkWrapper(model, dev, latent_dim, self_cond, processes,
                                  nothing, nothing, 0)
end

"""
    (w::BranchingScoreNetworkWrapper)(t, Xt::BranchingState)

Step function for Flowfusion.gen with BranchingFlows.
Returns (X1_targets, split_logits, del_logits) tuple as expected by CoalescentFlow.step.

# Arguments
- `t`: Current time (scalar, raw uniform progress)
- `Xt`: BranchingState at time t

# Returns
Tuple: (X1_targets, split_logits, del_logits)
- X1_targets: Tuple of (ContinuousState, ContinuousState) for CA and latent predictions
- split_logits: [L] vector of split logits
- del_logits: [L] vector of deletion logits
"""
function (w::BranchingScoreNetworkWrapper)(t, Xt::BranchingState)
    # Extract state components
    ca_state = Xt.state[1]      # MaskedState wrapping ContinuousState
    latent_state = Xt.state[2]  # MaskedState wrapping ContinuousState

    # Get tensors from MaskedState - already in [D, L, B] format from CoalescentFlow.step
    x_ca = tensor(ca_state.S)        # [3, L, B]
    x_ll = tensor(latent_state.S)    # [latent_dim, L, B]

    # Handle different tensor formats
    if ndims(x_ca) == 3
        L = size(x_ca, 2)
        B = size(x_ca, 3)
        x_ca_model = x_ca  # Already [D, L, B]
        x_ll_model = x_ll
    else
        # Shouldn't happen but handle gracefully
        error("Unexpected tensor format: $(size(x_ca))")
    end

    mask = Float32.(Xt.padmask)  # [L, B] or [L]
    if ndims(mask) == 1
        mask = reshape(mask, L, 1)
    end

    # Handle self-conditioning with length changes using index tracking
    sc_ca, sc_ll = nothing, nothing
    if w.self_cond && !isnothing(w.sc_ca)
        if L == w.prev_L
            # Same length - use previous predictions directly
            sc_ca = w.sc_ca
            sc_ll = w.sc_ll
        else
            # Length changed - use index state to expand/contract self-conditioning
            # The 3rd state (NullProcess) tracks indices through splits/deletions
            if length(Xt.state) >= 3
                idx_state = Xt.state[3]
                frominds = vec(idx_state.S.state)  # [L, B] -> [L*B], get indices
                # Expand self-conditioning using indices
                sc_ca = w.sc_ca[:, frominds, :]
                sc_ll = w.sc_ll[:, frominds, :]
            end
        end
    end

    # Reset index state AFTER using frominds (BranchChain pattern)
    if length(Xt.state) >= 3
        Xt.state[3].S.state .= reshape(collect(1:L), L, B)
    end

    # Apply per-modality schedule transforms to convert raw uniform progress
    # to the actual interpolation times the model was trained on.
    t_raw = isa(t, Number) ? Float32(t) : Float32(t[1])
    if !isnothing(w.processes)
        P_ca, P_ll = w.processes
        t_ca = schedule_transform(P_ca, t_raw)
        t_ll = schedule_transform(P_ll, t_raw)
    else
        t_ca = t_raw
        t_ll = t_raw
    end

    t_vec_ca = fill(t_ca, B)
    t_vec_ll = fill(t_ll, B)

    batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => w.dev(x_ca_model), :local_latents => w.dev(x_ll_model)),
        :t => Dict(:bb_ca => w.dev(t_vec_ca), :local_latents => w.dev(t_vec_ll)),
        :mask => w.dev(mask)
    )

    # Add self-conditioning if available
    if !isnothing(sc_ca)
        batch[:x_sc] = Dict(:bb_ca => w.dev(sc_ca), :local_latents => w.dev(sc_ll))
    end

    # Run model
    output = w.model(batch)

    # Extract X1 predictions — use per-modality transformed times for v_to_x1
    out_key = w.model.base.output_param
    if out_key == :v
        v_ca = output[:bb_ca][:v]
        v_ll = output[:local_latents][:v]
        x1_ca = v_to_x1(w.dev(x_ca_model), v_ca, t_ca)
        x1_ll = v_to_x1(w.dev(x_ll_model), v_ll, t_ll)
    else
        x1_ca = output[:bb_ca][:x1]
        x1_ll = output[:local_latents][:x1]
    end

    split_logits = output[:split]  # [L, 1]
    del_logits = output[:del]      # [L, 1]

    # Store predictions for self-conditioning (on CPU)
    w.sc_ca = cpu(x1_ca)
    w.sc_ll = cpu(x1_ll)
    w.prev_L = L

    # X1 targets should be in [D, L, B] format for CoalescentFlow.step
    x1_ca_ff = cpu(x1_ca)  # [3, L, B]
    x1_ll_ff = cpu(x1_ll)  # [latent_dim, L, B]

    # Return in format expected by CoalescentFlow.step:
    # (X1_targets, split_logits, del_logits)
    # Include DiscreteState for index process with [L, B] dimensions
    X1_targets = (ContinuousState(x1_ca_ff), ContinuousState(x1_ll_ff), DiscreteState(0, reshape(collect(1:L), L, B)))

    # Split/del logits - convert to vector [L]
    split_vec = cpu(vec(split_logits))  # [L] (flatten from [L, B])
    del_vec = cpu(vec(del_logits))      # [L]

    return (X1_targets, split_vec, del_vec)
end

"""
    create_branching_processes(; latent_dim=8, kwargs...)

Create CoalescentFlow-wrapped processes for branching generation.
Processes carry their schedules and SDE parameters internally, matching
la-proteina defaults.

# Keyword Arguments
- `latent_dim`: Dimension of local latents
- CA and latent schedule/SDE parameters (passed to RDNFlow)
- Branch time distribution (default: Beta(1, 2))

# Returns
CoalescentFlow wrapping (P_ca, P_ll, P_idx)
"""
function create_branching_processes(;
        latent_dim::Int=8,
        # CA schedule + SDE
        ca_schedule::Symbol=:log,
        ca_schedule_param::Real=2.0,
        ca_gt_mode::Symbol=Symbol("1/t"),
        ca_gt_param::Real=1.0,
        ca_sc_scale_noise::Real=0.1,
        ca_sc_scale_score::Real=1.0,
        ca_t_lim_ode::Real=0.98,
        # Latent schedule + SDE
        ll_schedule::Symbol=:power,
        ll_schedule_param::Real=2.0,
        ll_gt_mode::Symbol=:tan,
        ll_gt_param::Real=1.0,
        ll_sc_scale_noise::Real=0.1,
        ll_sc_scale_score::Real=1.0,
        ll_t_lim_ode::Real=0.98,
        # Branching parameters
        branch_time_alpha::Real=1.0,
        branch_time_beta::Real=2.0)

    # Create base processes with full la-proteina defaults
    P_ca = RDNFlow(3;
        zero_com=false,
        schedule=ca_schedule,
        schedule_param=Float32(ca_schedule_param),
        sde_gt_mode=ca_gt_mode,
        sde_gt_param=Float32(ca_gt_param),
        sc_scale_noise=Float32(ca_sc_scale_noise),
        sc_scale_score=Float32(ca_sc_scale_score),
        t_lim_ode=Float32(ca_t_lim_ode)
    )

    P_ll = RDNFlow(latent_dim;
        zero_com=false,
        schedule=ll_schedule,
        schedule_param=Float32(ll_schedule_param),
        sde_gt_mode=ll_gt_mode,
        sde_gt_param=Float32(ll_gt_param),
        sc_scale_noise=Float32(ll_sc_scale_noise),
        sc_scale_score=Float32(ll_sc_scale_score),
        t_lim_ode=Float32(ll_t_lim_ode)
    )

    # Create branch time distribution
    branch_time_dist = Beta(branch_time_alpha, branch_time_beta)

    # NullProcess for index tracking through splits/deletions
    P_idx = NullProcess()

    # Wrap in CoalescentFlow with index tracking as 3rd process
    P = CoalescentFlow((P_ca, P_ll, P_idx), branch_time_dist)

    return P
end

"""
    create_initial_state(P_ca::RDNFlow, P_ll::RDNFlow, L::Int, latent_dim::Int; T=Float32)

Create initial BranchingState for generation (at t=0).
Uses sample_rdn_noise to properly handle zero-COM for CA coordinates.

# Arguments
- `P_ca`: RDNFlow process for CA (used for proper noise sampling)
- `P_ll`: RDNFlow process for latents (used for proper noise sampling)
- `L`: Initial sequence length
- `latent_dim`: Dimension of local latents
- `T`: Element type

# Returns
BranchingState at t=0 (noise)
"""
function create_initial_state(P_ca::RDNFlow, P_ll::RDNFlow, L::Int, latent_dim::Int; T=Float32)
    # Sample noise using Flowfusion's noise sampler
    ca_noise = sample_rdn_noise(P_ca, L, 1; T=T)      # [3, L, 1]
    ll_noise = sample_rdn_noise(P_ll, L, 1; T=T)      # [latent_dim, L, 1]

    # Create masks as matrices [L, B] for batch size 1
    flowmask = ones(Bool, L, 1)
    padmask = ones(Bool, L, 1)
    branchmask = ones(Bool, L, 1)

    # MaskedState wraps ContinuousState with masks
    ca_state = MaskedState(ContinuousState(ca_noise), flowmask, padmask)
    ll_state = MaskedState(ContinuousState(ll_noise), flowmask, padmask)

    # Index state for tracking positions through splits/deletions (NullProcess)
    idx_state = MaskedState(DiscreteState(0, reshape(collect(1:L), L, 1)), flowmask, padmask)

    # Group IDs (all same group for single chain)
    groupings = ones(Int, L, 1)

    return BranchingState(
        (ca_state, ll_state, idx_state),
        groupings;
        flowmask = flowmask,
        branchmask = branchmask,
        padmask = padmask
    )
end

# Backward-compatible version without processes (uses plain randn)
function create_initial_state(L::Int, latent_dim::Int; T=Float32)
    ca_noise = randn(T, 3, L, 1)
    ll_noise = randn(T, latent_dim, L, 1)

    flowmask = ones(Bool, L, 1)
    padmask = ones(Bool, L, 1)
    branchmask = ones(Bool, L, 1)

    ca_state = MaskedState(ContinuousState(ca_noise), flowmask, padmask)
    ll_state = MaskedState(ContinuousState(ll_noise), flowmask, padmask)
    idx_state = MaskedState(DiscreteState(0, reshape(collect(1:L), L, 1)), flowmask, padmask)
    groupings = ones(Int, L, 1)

    return BranchingState(
        (ca_state, ll_state, idx_state),
        groupings;
        flowmask = flowmask,
        branchmask = branchmask,
        padmask = padmask
    )
end

"""
    generate_with_branching(model::BranchingScoreNetwork, initial_length::Int;
                            nsteps=400, latent_dim=8, dev=identity, kwargs...)

Generate a protein structure with variable length using Branching Flows.

Uses uniform time steps — the process-internal schedule_transform handles
non-uniform time mapping. The model wrapper applies per-modality schedule
transforms for correct conditioning.

# Arguments
- `model`: BranchingScoreNetwork
- `initial_length`: Starting sequence length (will change during generation)
- `nsteps`: Number of integration steps
- `latent_dim`: Dimension of local latents
- `self_cond`: Whether to use self-conditioning
- `dev`: Device function (gpu or identity)

# Returns
NamedTuple with:
- `ca_coords`: [3, L_final] CA coordinates
- `latents`: [latent_dim, L_final] local latent vectors
- `final_length`: Final sequence length
- `trajectory_lengths`: Vector of lengths at each step (for analysis)
"""
function generate_with_branching(model::BranchingScoreNetwork, initial_length::Int;
                                  nsteps::Int=400,
                                  latent_dim::Int=8,
                                  self_cond::Bool=true,
                                  dev=identity,
                                  verbose::Bool=false,
                                  kwargs...)

    # Create processes (carry schedules + SDE params internally)
    P = create_branching_processes(; latent_dim=latent_dim, kwargs...)

    # Extract the two real processes for the wrapper (skip NullProcess)
    P_ca = P.P[1]
    P_ll = P.P[2]

    # Create wrapper with process-aware time transforms
    wrapper = BranchingScoreNetworkWrapper(dev(model), latent_dim;
                                            self_cond=self_cond, dev=dev,
                                            processes=(P_ca, P_ll))

    # Create initial state with proper noise (zero-COM for CA)
    X0 = create_initial_state(P_ca, P_ll, initial_length, latent_dim)

    # Uniform time steps — process-internal schedule_transform handles non-uniformity
    ts = Float32.(range(0, 1, length=nsteps+1))

    # Track trajectory
    trajectory_lengths = Int[initial_length]

    # Run generation
    Xt = X0
    for i in 1:nsteps
        t1, t2 = ts[i], ts[i+1]

        # Get model predictions (wrapper applies schedule_transform internally)
        hat = wrapper(t1, Xt)

        # Step forward (RDNFlow.step applies schedule_transform internally)
        Xt = Flowfusion.step(P, Xt, hat, t1, t2)

        # Track length
        L_current = size(Xt.groupings, 1)
        push!(trajectory_lengths, L_current)

        if verbose && i % 100 == 0
            println("Step $i/$nsteps: t=$(round(t2, digits=3)), L=$L_current")
        end
    end

    # Extract final state
    ca_state = Xt.state[1]
    ll_state = Xt.state[2]

    ca_tensor = tensor(ca_state.S)  # [3, L, B]
    ll_tensor = tensor(ll_state.S)  # [latent_dim, L, B]

    # Convert to [D, L] format (drop batch dimension)
    ca_coords = dropdims(ca_tensor, dims=3)  # [3, L]
    latents = dropdims(ll_tensor, dims=3)    # [latent_dim, L]

    final_length = size(ca_coords, 2)

    return (
        ca_coords = ca_coords,
        latents = latents,
        final_length = final_length,
        trajectory_lengths = trajectory_lengths
    )
end

"""
    reset_self_conditioning!(wrapper::BranchingScoreNetworkWrapper)

Reset self-conditioning state (e.g., at the start of a new generation).
"""
function reset_self_conditioning!(wrapper::BranchingScoreNetworkWrapper)
    wrapper.sc_ca = nothing
    wrapper.sc_ll = nothing
    wrapper.prev_L = 0
    return wrapper
end
