#!/usr/bin/env julia
# Test that branching training works with raw features pattern
# This validates the fix for the Zygote mutation error

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

println("=" ^ 60)
println("Testing Branching Training with Raw Features")
println("=" ^ 60)

using LaProteina
using LaProteina: ScoreNetworkRawFeatures, extract_raw_features
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow, branching_bridge
using ForwardBackward: ContinuousState, tensor
using Flowfusion: RDNFlow, MaskedState
using Distributions: Uniform, Beta
using Flux
using Optimisers
using Statistics

# Include branching module
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))

latent_dim = 8

# Create fake protein data
function make_fake_protein(L::Int)
    return (
        ca_coords = randn(Float32, 3, L),
        z_mean = randn(Float32, latent_dim, L),
        z_log_scale = randn(Float32, latent_dim, L) .- 1f0,
        mask = ones(Float32, L)
    )
end

proteins = [make_fake_protein(rand(20:30)) for _ in 1:5]
println("Created $(length(proteins)) fake proteins")

# Create model (small for testing)
println("\n=== Creating Model ===")
base = ScoreNetwork(
    n_layers=2, token_dim=64, pair_dim=32, n_heads=4,
    dim_cond=32, latent_dim=latent_dim, output_param=:v
)
model = BranchingScoreNetwork(base)
println("Model created")

# Create CoalescentFlow
P_ca = RDNFlow(3; zero_com=true, schedule=:log, schedule_param=2.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, schedule=:power, schedule_param=2.0f0)
branch_time_dist = Beta(2.0, 2.0)
P = CoalescentFlow((P_ca, P_ll), branch_time_dist)
println("CoalescentFlow created")

# X0 sampler
function X0_sampler_simple(root)
    ca_state = ContinuousState(randn(Float32, 3, 1, 1))
    latent_state = ContinuousState(randn(Float32, latent_dim, 1, 1))
    return (ca_state, latent_state)
end

# Convert protein to BranchingState (without index tracking)
function protein_to_X1_simple(protein)
    ca_coords = protein.ca_coords
    z_mean = protein.z_mean
    z_log_scale = protein.z_log_scale
    mask = protein.mask
    L = length(mask)

    z_latent = z_mean .+ randn(Float32, size(z_mean)) .* exp.(z_log_scale)
    valid_mask = Vector{Bool}(mask .> 0.5)

    ca_state = MaskedState(
        ContinuousState(reshape(ca_coords, 3, 1, L)),
        valid_mask, valid_mask
    )
    latent_state = MaskedState(
        ContinuousState(reshape(z_latent, size(z_latent, 1), 1, L)),
        valid_mask, valid_mask
    )

    groupings = ones(Int, L)
    return BranchingState(
        (ca_state, latent_state),
        groupings;
        flowmask = valid_mask,
        branchmask = valid_mask
    )
end

# Setup optimizer
opt_state = setup_optimizer(model, 1e-3; freeze_base=true)
println("Optimizer created")

println("\n=== Testing Training Loop with Raw Features ===")

for batch_idx in 1:3
    println("\n--- Batch $batch_idx ---")

    # Sample batch
    batch_size = 2
    indices = rand(1:length(proteins), batch_size)
    X1s = [protein_to_X1_simple(proteins[i]) for i in indices]

    # Run branching_bridge
    t_dist = Uniform(0f0, 1f0)
    batch = branching_bridge(
        P, X0_sampler_simple, X1s, t_dist;
        coalescence_factor = 0.5,
        use_branching_time_prob = 0.3,
        length_mins = nothing,
        deletion_pad = 0
    )

    # Extract tensors (CPU)
    L_batch, B = size(batch.Xt.groupings)
    t_vec_cpu = Float32.(batch.t)

    ca_tensor = tensor(batch.Xt.state[1])
    ll_tensor = tensor(batch.Xt.state[2])
    if ndims(ca_tensor) == 4
        xt_ca_cpu = dropdims(ca_tensor, dims=2)
        xt_ll_cpu = dropdims(ll_tensor, dims=2)
    else
        xt_ca_cpu = ca_tensor
        xt_ll_cpu = ll_tensor
    end

    mask_cpu = Float32.(batch.Xt.padmask)
    branchmask_cpu = Float32.(batch.Xt.branchmask)
    combined_mask_cpu = mask_cpu .* branchmask_cpu

    # Targets
    x1_ca_target_cpu = tensor(batch.X1anchor[1])
    x1_ll_target_cpu = tensor(batch.X1anchor[2])
    if ndims(x1_ca_target_cpu) == 4
        x1_ca_target_cpu = dropdims(x1_ca_target_cpu, dims=2)
        x1_ll_target_cpu = dropdims(x1_ll_target_cpu, dims=2)
    end

    split_target_cpu = Float32.(batch.splits_target)
    del_target_cpu = Float32.(batch.del)

    println("  Batch shapes: L=$L_batch, B=$B")

    # Build CPU batch for feature extraction
    cpu_batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
        :t => Dict(:bb_ca => t_vec_cpu, :local_latents => t_vec_cpu),
        :mask => mask_cpu
    )

    # KEY: Extract raw features OUTSIDE gradient context
    println("  Extracting raw features...")
    raw_features = extract_raw_features(model.base, cpu_batch)
    println("  Raw features extracted: seq=$(size(raw_features.seq_raw)), cond=$(size(raw_features.cond_raw))")

    # Compute loss and gradients
    println("  Computing gradients...")
    loss, grads = Flux.withgradient(model) do m
        out = forward_branching_from_raw_features(m, raw_features)

        # Get x1 predictions from velocity
        v_ca = out[:bb_ca][:v]
        v_ll = out[:local_latents][:v]
        t_exp = reshape(t_vec_cpu, 1, 1, :)
        x1_ca = xt_ca_cpu .+ (1f0 .- t_exp) .* v_ca
        x1_ll = xt_ll_cpu .+ (1f0 .- t_exp) .* v_ll

        # Time scaling
        t_scale = 1f0 ./ max.(1f0 .- t_vec_cpu, 1f-5).^2

        # CA loss
        ca_diff = (x1_ca .- x1_ca_target_cpu).^2
        ca_loss = sum(ca_diff .* reshape(mask_cpu, 1, size(mask_cpu)...) .* reshape(t_scale, 1, 1, :)) / sum(mask_cpu)

        # Latent loss
        ll_diff = (x1_ll .- x1_ll_target_cpu).^2
        ll_loss = sum(ll_diff .* reshape(mask_cpu, 1, size(mask_cpu)...) .* reshape(t_scale, 1, 1, :)) / sum(mask_cpu)

        # Split loss (Bregman Poisson)
        mu = exp.(out[:split])
        split_l = sum((mu .- split_target_cpu .* out[:split]) .* combined_mask_cpu .* reshape(t_scale, 1, :)) / max(sum(combined_mask_cpu), 1f0)

        # Del loss (BCE)
        del_l = sum(((1f0 .- del_target_cpu) .* out[:del] .+ log1p.(exp.(-out[:del]))) .* combined_mask_cpu .* reshape(t_scale, 1, :)) / max(sum(combined_mask_cpu), 1f0)

        # Total
        ca_loss * 0.1f0 + ll_loss * 0.1f0 + split_l + del_l
    end

    println("  Loss: $loss")
    println("  Gradients computed: $(grads[1] !== nothing)")

    # Update
    Optimisers.update!(opt_state, model, grads[1])
    println("  Parameters updated successfully")
end

println("\n" * "=" ^ 60)
println("Raw Features Training Test PASSED!")
println("=" ^ 60)
