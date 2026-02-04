# Inference utilities for Branching Flows generation
# Includes self-conditioning wrapper that handles variable-length states

using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, tensor
using Flux: cpu, gpu

"""
    BranchingScoreNetworkWrapper

Mutable wrapper for BranchingScoreNetwork that handles self-conditioning
with variable-length states during generation.

When splits/deletions occur, the wrapper uses the index state to expand/contract
the self-conditioning predictions to match the new state size.
"""
mutable struct BranchingScoreNetworkWrapper{M, D}
    model::M                    # BranchingScoreNetwork
    dev::D                      # Device function (gpu or identity)
    latent_dim::Int
    self_cond::Bool
    # Self-conditioning state (stored at previous step's indices)
    sc_ca::Union{Nothing, AbstractArray}
    sc_ll::Union{Nothing, AbstractArray}
end

function BranchingScoreNetworkWrapper(model, latent_dim::Int;
                                       self_cond::Bool=true, dev=identity)
    BranchingScoreNetworkWrapper(model, dev, latent_dim, self_cond, nothing, nothing)
end

"""
    (w::BranchingScoreNetworkWrapper)(t, Xt::BranchingState)

Step function for Flowfusion.gen with BranchingFlows.
Handles self-conditioning expansion when splits/deletions occur.

# Arguments
- `t`: Current time (scalar or [B])
- `Xt`: BranchingState at time t

# Returns
Tuple of (X1_ca, X1_ll, split_logits, del_logits) as expected by CoalescentFlow.step
"""
function (w::BranchingScoreNetworkWrapper)(t, Xt::BranchingState)
    # Extract state components
    ca_state = Xt.state[1]      # MaskedState wrapping ContinuousState
    latent_state = Xt.state[2]  # MaskedState wrapping ContinuousState
    index_state = Xt.state[3]   # MaskedState wrapping DiscreteState (indices)

    # Get tensors
    x_ca = tensor(ca_state.S)        # [3, 1, L] or [3, L, B]
    x_ll = tensor(latent_state.S)    # [latent_dim, 1, L] or [latent_dim, L, B]
    current_indices = index_state.S.state  # [L] or [L, B] - which original position each came from

    # Handle dimension ordering: Flowfusion uses [D, 1, L] for unbatched
    # but our model expects [D, L, B]
    if ndims(x_ca) == 3 && size(x_ca, 2) == 1
        # Unbatched format [D, 1, L] -> [D, L, 1]
        x_ca = permutedims(x_ca, (1, 3, 2))
        x_ll = permutedims(x_ll, (1, 3, 2))
    end

    L, B = size(x_ca, 2), size(x_ca, 3)
    mask = Float32.(Xt.padmask)

    # Expand self-conditioning to match current length using index state
    if w.self_cond && !isnothing(w.sc_ca)
        # current_indices[i] tells us: position i came from original position current_indices[i]
        # Use this to index into the stored predictions
        if ndims(current_indices) == 1
            sc_ca = expand_by_indices(w.sc_ca, current_indices)
            sc_ll = expand_by_indices(w.sc_ll, current_indices)
        else
            sc_ca = expand_by_indices(w.sc_ca, current_indices)
            sc_ll = expand_by_indices(w.sc_ll, current_indices)
        end
    else
        sc_ca, sc_ll = nothing, nothing
    end

    # Build batch for model
    t_scalar = isa(t, Number) ? t : t[1]
    t_vec = fill(Float32(t_scalar), B)

    batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => w.dev(x_ca), :local_latents => w.dev(x_ll)),
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
        x1_ca = v_to_x1(w.dev(x_ca), v_ca, t_scalar)
        x1_ll = v_to_x1(w.dev(x_ll), v_ll, t_scalar)
    else
        x1_ca = output[:bb_ca][:x1]
        x1_ll = output[:local_latents][:x1]
    end

    split_logits = output[:split]  # [L, B]
    del_logits = output[:del]      # [L, B]

    # Store predictions for next step (on CPU for Flowfusion)
    # These will be re-indexed when splits/deletions happen
    w.sc_ca = cpu(x1_ca)
    w.sc_ll = cpu(x1_ll)

    # Return in format expected by CoalescentFlow.step
    # Convert back to Flowfusion format if needed
    x1_ca_cpu = cpu(x1_ca)
    x1_ll_cpu = cpu(x1_ll)

    if size(x1_ca_cpu, 3) == 1
        # Convert [D, L, 1] back to [D, 1, L] for unbatched Flowfusion
        x1_ca_cpu = permutedims(x1_ca_cpu, (1, 3, 2))
        x1_ll_cpu = permutedims(x1_ll_cpu, (1, 3, 2))
    end

    return (ContinuousState(x1_ca_cpu), ContinuousState(x1_ll_cpu),
            cpu(split_logits), cpu(del_logits))
end

"""
    create_branching_processes(; latent_dim=8, kwargs...)

Create CoalescentFlow-wrapped processes for branching generation.

# Keyword Arguments
- `latent_dim`: Dimension of local latents
- CA and latent schedule parameters (passed to RDNFlow)
- Branching parameters (passed to CoalescentFlow)

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
        # Branching parameters (CoalescentFlow)
        branch_time_dist=nothing,  # Will use default if nothing
        split_transform=exp,       # Maps logits -> split intensity
        deletion_hazard=nothing)   # Will use default if nothing

    # Import process types
    using Flowfusion: RDNFlow

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

    # Wrap in CoalescentFlow
    # Note: The actual CoalescentFlow constructor signature depends on BranchingFlows version
    # This is a placeholder - adjust based on actual API
    P = CoalescentFlow(
        (P_ca, P_ll);
        # branch_time_dist, split_transform, deletion_hazard would go here
    )

    return P
end

"""
    generate_with_branching(model::BranchingScoreNetwork, target_length::Int;
                            nsteps=400, latent_dim=8, dev=identity, kwargs...)

Generate a protein structure with variable length using Branching Flows.

# Arguments
- `model`: BranchingScoreNetwork
- `target_length`: Target sequence length (actual length may vary)
- `nsteps`: Number of integration steps
- `latent_dim`: Dimension of local latents
- `dev`: Device function (gpu or identity)

# Returns
Dict with generated structure information
"""
function generate_with_branching(model::BranchingScoreNetwork, target_length::Int;
                                  nsteps::Int=400,
                                  latent_dim::Int=8,
                                  self_cond::Bool=true,
                                  dev=identity,
                                  kwargs...)

    # This is a placeholder implementation
    # Full implementation requires:
    # 1. Creating CoalescentFlow processes
    # 2. Using branching_bridge to initialize X0
    # 3. Running gen() with the BranchingScoreNetworkWrapper

    error("generate_with_branching not yet fully implemented - requires BranchingFlows integration")

    # Sketch of what the full implementation would look like:
    #=
    # Create processes
    P = create_branching_processes(; latent_dim=latent_dim, kwargs...)

    # Create X0 sampler
    X0_sampler = X0_sampler_laproteina(latent_dim)

    # Create dummy X1 template (for unconditional generation)
    # Or use a real protein as template for conditional generation
    template = create_dummy_template(target_length, latent_dim)
    X1s = [template]

    # Sample time
    t = Uniform(0f0, 1f0)

    # Get initial state via branching_bridge
    bridge_result = branching_bridge(P, X0_sampler, X1s, t,
        length_mins = Poisson(target_length),
        ...
    )
    X0 = bridge_result.Xt

    # Create wrapper
    wrapper = BranchingScoreNetworkWrapper(dev(model), latent_dim;
                                            self_cond=self_cond, dev=dev)

    # Time steps
    steps = Float32.(get_schedule(:log, nsteps; p=2.0))

    # Run generation
    X_final = gen(P, X0, wrapper, steps)

    # Extract results
    ...
    =#
end

"""
    reset_self_conditioning!(wrapper::BranchingScoreNetworkWrapper)

Reset self-conditioning state (e.g., at the start of a new generation).
"""
function reset_self_conditioning!(wrapper::BranchingScoreNetworkWrapper)
    wrapper.sc_ca = nothing
    wrapper.sc_ll = nothing
    return wrapper
end
