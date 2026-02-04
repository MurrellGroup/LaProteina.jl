#!/usr/bin/env julia
# Train Branching Flows model with parallel data loading
#
# Usage:
#   julia scripts/train_branching_parallel.jl
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

# CRITICAL: Override unsafe_free! for proper GPU memory management with DataLoader
Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

Random.seed!(42)

println("=" ^ 70)
println("Branching Flows Training (Parallel Data Loading)")
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
# Parallel Data Loading Setup
# ============================================================================
println("\n=== Setting up Parallel Data Loading ===")

# Helper to convert protein to BranchingState
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

"""
Prepare a full training batch on CPU (called by DataLoader workers).
Returns a NamedTuple with all tensors needed for training.
"""
function prepare_training_batch(indices, proteins_ref, P_ref, X0_sampler, base_model)
    # Convert proteins to BranchingStates
    X1s = [protein_to_X1_simple(proteins_ref[i]) for i in indices]

    # Run branching_bridge
    t_dist = Uniform(0f0, 1f0)
    batch = branching_bridge(
        P_ref, X0_sampler, X1s, t_dist;
        coalescence_factor = 0.5,
        use_branching_time_prob = 0.3,
        length_mins = nothing,
        deletion_pad = 0
    )

    # Extract tensors
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

    # Build CPU batch for feature extraction
    cpu_batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
        :t => Dict(:bb_ca => t_vec_cpu, :local_latents => t_vec_cpu),
        :mask => mask_cpu
    )

    # Extract raw features on CPU (the slow part we want to parallelize)
    raw_features = extract_raw_features(base_model, cpu_batch)

    return (
        raw_features = raw_features,
        xt_ca = xt_ca_cpu,
        xt_ll = xt_ll_cpu,
        mask = mask_cpu,
        combined_mask = combined_mask_cpu,
        x1_ca_target = x1_ca_target_cpu,
        x1_ll_target = x1_ll_target_cpu,
        split_target = split_target_cpu,
        del_target = del_target_cpu,
        t_vec = t_vec_cpu
    )
end

# BatchDataset for parallel loading (follows BranchChain pattern)
# Each getindex call returns a fully prepared batch
struct BatchDataset{P, S, M}
    batch_indices::Vector{Vector{Int}}  # Pre-sampled batch indices
    proteins::P
    process::S
    base_model::M
end

Base.length(x::BatchDataset) = length(x.batch_indices)

# getindex returns the prepared batch - this is called by DataLoader workers in parallel
function Base.getindex(x::BatchDataset, i::Int)
    prepare_training_batch(x.batch_indices[i], x.proteins, x.process, X0_sampler_train, x.base_model)
end

# Pre-sample batch indices
println("Pre-sampling batch indices...")
batch_indices = [rand(1:length(proteins), batch_size) for _ in 1:n_batches]
println("Created $(length(batch_indices)) batch index sets")

# Create dataset and dataloader
# batchsize=-1: each getindex returns one complete batch, no further batching
# parallel=true: prepare next batch in background while current batch trains
dataset = BatchDataset(batch_indices, proteins, P, cpu(model.base))
dataloader = Flux.DataLoader(dataset; batchsize=-1, parallel=true)
println("DataLoader created with parallel=true, batchsize=-1")

# ============================================================================
# Training Loop
# ============================================================================
println("\n=== Training ===")

losses = Float32[]
batch_times = Float64[]

println("Starting training loop...")
println("Batch size: $batch_size, N batches: $n_batches")

t_train_start = time()

for (batch_idx, bd_cpu) in enumerate(dataloader)
    t_batch = time()

    # Transfer entire batch to GPU at once
    bd = dev(bd_cpu)

    # Compute loss and gradients
    loss, grads = Flux.withgradient(model) do m
        out = forward_branching_from_raw_features(m, bd.raw_features)

        # Get x1 predictions from velocity
        v_ca = out[:bb_ca][:v]
        v_ll = out[:local_latents][:v]
        t_exp = reshape(bd.t_vec, 1, 1, :)
        x1_ca = bd.xt_ca .+ (1f0 .- t_exp) .* v_ca
        x1_ll = bd.xt_ll .+ (1f0 .- t_exp) .* v_ll

        # Time scaling
        t_scale = 1f0 ./ max.(1f0 .- bd.t_vec, 1f-5).^2

        # CA loss
        ca_diff = (x1_ca .- bd.x1_ca_target).^2
        ca_loss = sum(ca_diff .* reshape(bd.mask, 1, size(bd.mask)...) .* reshape(t_scale, 1, 1, :)) / sum(bd.mask)

        # Latent loss
        ll_diff = (x1_ll .- bd.x1_ll_target).^2
        ll_loss = sum(ll_diff .* reshape(bd.mask, 1, size(bd.mask)...) .* reshape(t_scale, 1, 1, :)) / sum(bd.mask)

        # Split loss (Bregman Poisson)
        mu = exp.(out[:split])
        split_l = sum((mu .- bd.split_target .* out[:split]) .* bd.combined_mask .* reshape(t_scale, 1, :)) / max(sum(bd.combined_mask), 1f0)

        # Del loss (BCE)
        del_l = sum(((1f0 .- bd.del_target) .* out[:del] .+ log1p.(exp.(-out[:del]))) .* bd.combined_mask .* reshape(t_scale, 1, :)) / max(sum(bd.combined_mask), 1f0)

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

    # Stop after n_batches
    batch_idx >= n_batches && break
end

t_train = time() - t_train_start

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 70)
println("Training Complete")
println("=" ^ 70)

# Exclude first 10 batches (warmup/compilation) from timing statistics
warmup_batches = 10
post_warmup_times = batch_times[min(warmup_batches+1, end):end]

println("\nSummary:")
@printf("  Total time: %.1f minutes\n", t_train / 60)
@printf("  Avg batch time (all): %.1f ms\n", mean(batch_times) * 1000)
if length(post_warmup_times) > 0
    @printf("  Avg batch time (post-warmup): %.1f ms\n", mean(post_warmup_times) * 1000)
    @printf("  Throughput (post-warmup): %.1f samples/sec\n", batch_size / mean(post_warmup_times))
end

first_50 = losses[1:min(50, length(losses))]
last_50 = losses[max(1, end-49):end]
@printf("  First 50 loss: %.4f\n", mean(first_50))
@printf("  Last 50 loss: %.4f\n", mean(last_50))
@printf("  Change: %.2f%%\n", 100 * (mean(last_50) - mean(first_50)) / mean(first_50))

# ============================================================================
# Save Weights
# ============================================================================
println("\n=== Saving Weights ===")
output_dir = get(ENV, "OUTPUT_DIR", weights_dir)
mkpath(output_dir)

# Save indel head weights (base weights are from pretrained)
model_cpu = cpu(model)
save_path = joinpath(output_dir, "branching_indel_stage$(stage).jld2")
save_branching_weights(model_cpu, save_path; include_base=false)
println("Saved indel head weights to: $save_path")

# For stage 2, also save full model
if stage == 2
    full_save_path = joinpath(output_dir, "branching_full.jld2")
    save_branching_weights(model_cpu, full_save_path; include_base=true)
    println("Saved full model weights to: $full_save_path")
end

println("\nNext steps:")
if stage == 1
    println("  - Run with STAGE=2 to fine-tune full model")
end
println("  - Test generation with branching")

println("\nDone!")
