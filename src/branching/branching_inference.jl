# Inference utilities for Branching Flows generation
# Includes self-conditioning wrapper that handles variable-length states

using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, tensor
using Flowfusion: MaskedState, RDNFlow, element
import Flowfusion: gen
using Flux: cpu, gpu
using Distributions: Beta, Poisson

"""
    BranchingScoreNetworkWrapper

Mutable wrapper for BranchingScoreNetwork that handles self-conditioning
with variable-length states during generation.

When splits/deletions occur, the wrapper uses the index state to expand/contract
the self-conditioning predictions to match the new state size.

The wrapper returns predictions in the format expected by CoalescentFlow.step:
    (X1_targets, split_logits, del_logits)
"""
mutable struct BranchingScoreNetworkWrapper{M, D}
    model::M                    # BranchingScoreNetwork
    dev::D                      # Device function (gpu or identity)
    latent_dim::Int
    self_cond::Bool
    # Self-conditioning state (stored from previous step)
    sc_ca::Union{Nothing, AbstractArray}
    sc_ll::Union{Nothing, AbstractArray}
    # Index tracking for self-conditioning expansion
    prev_L::Int                 # Previous sequence length
end

function BranchingScoreNetworkWrapper(model, latent_dim::Int;
                                       self_cond::Bool=true, dev=identity)
    BranchingScoreNetworkWrapper(model, dev, latent_dim, self_cond, nothing, nothing, 0)
end

"""
    (w::BranchingScoreNetworkWrapper)(t, Xt::BranchingState)

Step function for Flowfusion.gen with BranchingFlows.
Returns (X1_targets, split_logits, del_logits) tuple as expected by CoalescentFlow.step.

# Arguments
- `t`: Current time (scalar)
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

    # Handle self-conditioning with length changes
    sc_ca, sc_ll = nothing, nothing
    if w.self_cond && !isnothing(w.sc_ca) && L == w.prev_L
        # Same length - use previous predictions directly
        sc_ca = w.sc_ca
        sc_ll = w.sc_ll
    elseif w.self_cond && !isnothing(w.sc_ca) && L != w.prev_L
        # Length changed - need to expand/contract self-conditioning
        # For now, reset self-conditioning when length changes
        # TODO: Implement proper index-based expansion using Xt.ids
        sc_ca = nothing
        sc_ll = nothing
    end

    # Build batch for model
    t_scalar = isa(t, Number) ? Float32(t) : Float32(t[1])
    t_vec = fill(t_scalar, B)

    batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => w.dev(x_ca_model), :local_latents => w.dev(x_ll_model)),
        :t => Dict(:bb_ca => w.dev(t_vec), :local_latents => w.dev(t_vec)),
        :mask => w.dev(mask)
    )

    # Add self-conditioning if available
    if !isnothing(sc_ca)
        batch[:x_sc] = Dict(:bb_ca => w.dev(sc_ca), :local_latents => w.dev(sc_ll))
    end

    # Run model
    output = w.model(batch)

    # Extract X1 predictions
    out_key = w.model.base.output_param
    if out_key == :v
        v_ca = output[:bb_ca][:v]
        v_ll = output[:local_latents][:v]
        x1_ca = v_to_x1(w.dev(x_ca_model), v_ca, t_scalar)
        x1_ll = v_to_x1(w.dev(x_ll_model), v_ll, t_scalar)
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
    X1_targets = (ContinuousState(x1_ca_ff), ContinuousState(x1_ll_ff))

    # Split/del logits - convert to vector [L]
    split_vec = cpu(vec(split_logits))  # [L] (flatten from [L, B])
    del_vec = cpu(vec(del_logits))      # [L]

    return (X1_targets, split_vec, del_vec)
end

"""
    create_branching_processes(; latent_dim=8, kwargs...)

Create CoalescentFlow-wrapped processes for branching generation.

# Keyword Arguments
- `latent_dim`: Dimension of local latents
- Branch time distribution (default: Beta(2, 2))
- CA and latent schedule parameters (passed to RDNFlow)

# Returns
CoalescentFlow wrapping (P_ca, P_ll)
"""
function create_branching_processes(;
        latent_dim::Int=8,
        # CA schedule
        ca_schedule::Symbol=:log,
        ca_schedule_param::Real=2.0,
        ca_gt_mode::Symbol=Symbol("1/t"),
        ca_gt_param::Real=1.0,
        # Latent schedule
        ll_schedule::Symbol=:power,
        ll_schedule_param::Real=2.0,
        ll_gt_mode::Symbol=:tan,
        ll_gt_param::Real=1.0,
        # Branching parameters
        branch_time_alpha::Real=2.0,
        branch_time_beta::Real=2.0)

    # Create base processes
    P_ca = RDNFlow(3;
        zero_com=true,
        schedule=ca_schedule,
        schedule_param=Float32(ca_schedule_param),
        sde_gt_mode=ca_gt_mode,
        sde_gt_param=Float32(ca_gt_param)
    )

    P_ll = RDNFlow(latent_dim;
        zero_com=false,
        schedule=ll_schedule,
        schedule_param=Float32(ll_schedule_param),
        sde_gt_mode=ll_gt_mode,
        sde_gt_param=Float32(ll_gt_param)
    )

    # Create branch time distribution
    branch_time_dist = Beta(branch_time_alpha, branch_time_beta)

    # Wrap in CoalescentFlow
    P = CoalescentFlow((P_ca, P_ll), branch_time_dist)

    return P
end

"""
    create_initial_state(L::Int, latent_dim::Int; T=Float32)

Create initial BranchingState for generation (at t=0).
Starts from noise.

# Arguments
- `L`: Initial sequence length
- `latent_dim`: Dimension of local latents
- `T`: Element type

# Returns
BranchingState at t=0 (noise)
"""
function create_initial_state(L::Int, latent_dim::Int; T=Float32)
    # Sample noise in Flowfusion format: [D, L, B] where B=1
    # CoalescentFlow.step expects [D, L, B] format
    ca_noise = randn(T, 3, L, 1)
    ll_noise = randn(T, latent_dim, L, 1)

    # Create masks as matrices [L, B] for batch size 1
    flowmask = ones(Bool, L, 1)
    padmask = ones(Bool, L, 1)
    branchmask = ones(Bool, L, 1)

    # MaskedState wraps ContinuousState with masks
    # For CoalescentFlow.step, masks should be [L, 1] matrices
    ca_state = MaskedState(ContinuousState(ca_noise), flowmask, padmask)
    ll_state = MaskedState(ContinuousState(ll_noise), flowmask, padmask)

    # Group IDs (all same group for single chain)
    groupings = ones(Int, L, 1)

    return BranchingState(
        (ca_state, ll_state),
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
                                  schedule::Symbol=:log,
                                  schedule_param::Real=2.0,
                                  verbose::Bool=false,
                                  kwargs...)

    # Create processes
    P = create_branching_processes(; latent_dim=latent_dim, kwargs...)

    # Create wrapper
    wrapper = BranchingScoreNetworkWrapper(dev(model), latent_dim;
                                            self_cond=self_cond, dev=dev)

    # Create initial state
    X0 = create_initial_state(initial_length, latent_dim)

    # Time schedule (t goes from 0 to 1)
    if schedule == :log
        # Log schedule: more steps near t=1
        ts = Float32.(1.0 .- 10.0 .^ (-schedule_param .* range(0, 1, length=nsteps+1)))
    else
        # Linear schedule
        ts = Float32.(range(0, 1, length=nsteps+1))
    end

    # Track trajectory
    trajectory_lengths = Int[initial_length]

    # Run generation
    Xt = X0
    for i in 1:nsteps
        t1, t2 = ts[i], ts[i+1]

        # Get model predictions
        hat = wrapper(t1, Xt)

        # Step forward
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
