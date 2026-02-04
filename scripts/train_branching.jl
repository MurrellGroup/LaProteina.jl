#!/usr/bin/env julia
# Train Branching Flows model on precomputed VAE encoder outputs
#
# Usage:
#   julia scripts/train_branching.jl
#
# Environment variables:
#   SHARD_DIR - Directory containing precomputed shards
#   WEIGHTS_DIR - Directory containing pretrained weights
#   BATCH_SIZE - Batch size (default: 4)
#   N_BATCHES - Number of training batches (default: 1000)
#   STAGE - Training stage: 1 (freeze base) or 2 (full) (default: 1)
#   LR - Learning rate (default: 1e-4 for stage 1, 1e-5 for stage 2)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: ScoreNetworkRawFeatures, extract_raw_features, cpu
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow, branching_bridge
using ForwardBackward: ContinuousState, DiscreteState, tensor
using Flowfusion: RDNFlow, MaskedState
using Distributions: Uniform, Beta
using Flux
using CUDA
using Optimisers
using Statistics
using Random
using Printf
using JLD2

# Include branching module
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))

Random.seed!(42)

println("=" ^ 70)
println("Branching Flows Training")
println("=" ^ 70)

# ============================================================================
# Configuration
# ============================================================================
shard_dir = get(ENV, "SHARD_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards"))
weights_dir = get(ENV, "WEIGHTS_DIR", joinpath(@__DIR__, "..", "weights"))
batch_size = parse(Int, get(ENV, "BATCH_SIZE", "4"))
n_batches = parse(Int, get(ENV, "N_BATCHES", "1000"))
stage = parse(Int, get(ENV, "STAGE", "1"))
default_lr = stage == 1 ? 1e-4 : 1e-5
lr = parse(Float64, get(ENV, "LR", string(default_lr)))
latent_dim = 8

println("\n=== Configuration ===")
println("Shard dir: $shard_dir")
println("Weights dir: $weights_dir")
println("Batch size: $batch_size")
println("N batches: $n_batches")
println("Stage: $stage ($(stage == 1 ? "freeze base" : "full fine-tune"))")
println("Learning rate: $lr")

# ============================================================================
# GPU Check
# ============================================================================
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: $(CUDA.device())")
    println("Memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB available")
    dev = gpu
else
    println("No CUDA - using CPU")
    dev = identity
end

# ============================================================================
# Load Precomputed Data
# ============================================================================
println("\n=== Loading Data ===")
shard_files = filter(f -> startswith(f, "train_shard_") && endswith(f, ".jld2"), readdir(shard_dir))
sort!(shard_files)
println("Found $(length(shard_files)) shard files")

# Load first shard for now (can extend to multi-shard later)
shard_path = joinpath(shard_dir, shard_files[1])
println("Loading: $shard_path")
t_load = @elapsed begin
    proteins = load_precomputed_shard(shard_path)
end
println("Loaded $(length(proteins)) proteins in $(round(t_load, digits=2))s")

# ============================================================================
# Create Model
# ============================================================================
println("\n=== Creating Model ===")

# Create base ScoreNetwork
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)

# Load pretrained weights
weights_path = joinpath(weights_dir, "score_network.npz")
if isfile(weights_path)
    println("Loading pretrained weights from: $weights_path")
    load_score_network_weights!(base, weights_path)
else
    println("WARNING: No pretrained weights found at $weights_path")
    println("Training from scratch")
end

# Wrap in BranchingScoreNetwork
model = BranchingScoreNetwork(base)
model = dev(model)
println("BranchingScoreNetwork created and moved to device")

# Setup optimizer with stage-appropriate freezing
opt_state = setup_optimizer(model, lr; freeze_base=(stage == 1))
println("Optimizer created (base frozen: $(stage == 1))")

# ============================================================================
# Create Branching Flow Process
# ============================================================================
println("\n=== Creating CoalescentFlow ===")

# Base processes (matching LaProteina inference settings)
P_ca = RDNFlow(3; zero_com=true, schedule=:log, schedule_param=2.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, schedule=:power, schedule_param=2.0f0)

# Branch time distribution - Beta(2,2) peaks in middle
branch_time_dist = Beta(2.0, 2.0)
P = CoalescentFlow((P_ca, P_ll), branch_time_dist)
println("CoalescentFlow created")

# X0 sampler for branching (returns (ca, latent) tuple)
function X0_sampler_train(root)
    ca = ContinuousState(randn(Float32, 3, 1, 1))
    ll = ContinuousState(randn(Float32, latent_dim, 1, 1))
    return (ca, ll)
end

# ============================================================================
# Training Loop
# ============================================================================
println("\n=== Training ===")

losses = Float32[]
split_losses = Float32[]
del_losses = Float32[]
batch_times = Float64[]

# Helper to convert protein to BranchingState (without index tracking for 2-process flow)
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

println("Starting training loop...")
println("Batch size: $batch_size, N batches: $n_batches")

t_train_start = time()

for batch_idx in 1:n_batches
    t_batch = time()

    # Sample batch of proteins
    indices = rand(1:length(proteins), batch_size)
    X1s = [protein_to_X1_simple(proteins[i]) for i in indices]

    # Run branching_bridge
    t_dist = Uniform(0f0, 1f0)
    batch = branching_bridge(
        P, X0_sampler_train, X1s, t_dist;
        coalescence_factor = 0.5,
        use_branching_time_prob = 0.3,  # Sometimes sample at split times
        length_mins = nothing,
        deletion_pad = 0
    )

    # Extract tensors (all on CPU initially)
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

    # Targets (CPU)
    x1_ca_target_cpu = tensor(batch.X1anchor[1])
    x1_ll_target_cpu = tensor(batch.X1anchor[2])
    if ndims(x1_ca_target_cpu) == 4
        x1_ca_target_cpu = dropdims(x1_ca_target_cpu, dims=2)
        x1_ll_target_cpu = dropdims(x1_ll_target_cpu, dims=2)
    end

    split_target_cpu = Float32.(batch.splits_target)
    del_target_cpu = Float32.(batch.del)

    # Build CPU batch for feature extraction (OUTSIDE gradient context)
    # This avoids Zygote mutation errors from bin_values operations
    cpu_batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
        :t => Dict(:bb_ca => t_vec_cpu, :local_latents => t_vec_cpu),
        :mask => mask_cpu
    )

    # Extract raw features on CPU
    raw_features = extract_raw_features(model.base, cpu_batch)

    # Move everything to GPU
    xt_ca = dev(xt_ca_cpu)
    xt_ll = dev(xt_ll_cpu)
    mask = dev(mask_cpu)
    combined_mask = dev(combined_mask_cpu)
    x1_ca_target = dev(x1_ca_target_cpu)
    x1_ll_target = dev(x1_ll_target_cpu)
    split_target = dev(split_target_cpu)
    del_target = dev(del_target_cpu)
    t_vec = dev(t_vec_cpu)

    # Move raw features to GPU
    raw_features_gpu = ScoreNetworkRawFeatures(
        dev(raw_features.seq_raw),
        dev(raw_features.cond_raw),
        dev(raw_features.pair_raw),
        dev(raw_features.pair_cond_raw),
        dev(raw_features.mask)
    )

    # Compute loss and gradients using forward_branching_from_raw_features
    loss, grads = Flux.withgradient(model) do m
        out = forward_branching_from_raw_features(m, raw_features_gpu)

        # Get x1 predictions from velocity
        v_ca = out[:bb_ca][:v]
        v_ll = out[:local_latents][:v]
        t_exp = reshape(t_vec, 1, 1, :)
        x1_ca = xt_ca .+ (1f0 .- t_exp) .* v_ca
        x1_ll = xt_ll .+ (1f0 .- t_exp) .* v_ll

        # Time scaling
        t_scale = 1f0 ./ max.(1f0 .- t_vec, 1f-5).^2

        # CA loss
        ca_diff = (x1_ca .- x1_ca_target).^2
        ca_loss = sum(ca_diff .* reshape(mask, 1, size(mask)...) .* reshape(t_scale, 1, 1, :)) / sum(mask)

        # Latent loss
        ll_diff = (x1_ll .- x1_ll_target).^2
        ll_loss = sum(ll_diff .* reshape(mask, 1, size(mask)...) .* reshape(t_scale, 1, 1, :)) / sum(mask)

        # Split loss (Bregman Poisson)
        mu = exp.(out[:split])
        split_l = sum((mu .- split_target .* out[:split]) .* combined_mask .* reshape(t_scale, 1, :)) / max(sum(combined_mask), 1f0)

        # Del loss (BCE)
        del_l = sum(((1f0 .- del_target) .* out[:del] .+ log1p.(exp.(-out[:del]))) .* combined_mask .* reshape(t_scale, 1, :)) / max(sum(combined_mask), 1f0)

        # Total (weight split/del more in stage 1)
        if stage == 1
            ca_loss * 0.1f0 + ll_loss * 0.1f0 + split_l + del_l
        else
            ca_loss + ll_loss + split_l + del_l
        end
    end

    # Update
    Optimisers.update!(opt_state, model, grads[1])

    push!(losses, Float32(cpu(loss)))
    push!(batch_times, time() - t_batch)

    # Logging
    if batch_idx % 50 == 0 || batch_idx == 1
        avg_loss = mean(losses[max(1, end-49):end])
        avg_time = mean(batch_times[max(1, end-49):end]) * 1000
        throughput = batch_size / (avg_time / 1000)
        @printf("Batch %4d: loss=%.4f, time=%.0fms, throughput=%.1f samples/sec\n",
                batch_idx, avg_loss, avg_time, throughput)

        if CUDA.functional()
            GC.gc()
            CUDA.reclaim()
        end
    end
end

t_train = time() - t_train_start

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 70)
println("Training Complete")
println("=" ^ 70)

println("\nSummary:")
@printf("  Total time: %.1f minutes\n", t_train / 60)
@printf("  Avg batch time: %.1f ms\n", mean(batch_times) * 1000)
@printf("  Throughput: %.1f samples/sec\n", n_batches * batch_size / t_train)

first_50 = losses[1:min(50, length(losses))]
last_50 = losses[max(1, end-49):end]
@printf("  First 50 loss: %.4f\n", mean(first_50))
@printf("  Last 50 loss: %.4f\n", mean(last_50))
@printf("  Change: %.2f%%\n", 100 * (mean(last_50) - mean(first_50)) / mean(first_50))

println("\nNext steps:")
if stage == 1
    println("  - Run with STAGE=2 to fine-tune full model")
end
println("  - Save model weights")
println("  - Test generation with branching")

println("\nDone!")
