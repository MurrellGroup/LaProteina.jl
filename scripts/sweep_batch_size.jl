#!/usr/bin/env julia
# Batch Size Sweep — find per-sample efficiency sweet spot
#
# Single continuous training run that changes batch size periodically.
# No model reloading — just a normal training loop where we swap the
# DataLoader every N batches and record per-sample timing.
#
# Usage:
#   julia -t 8 scripts/sweep_batch_size.jl
#
# Environment variables:
#   N_BATCHES_PER_SIZE - batches per batch size (default: 50)
#   SHARD_DIR, WEIGHTS_DIR - data/weight paths (same as train script)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: ScoreNetworkRawFeatures, extract_raw_features, cpu
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow, branching_bridge, Deletion
using ForwardBackward: ContinuousState, DiscreteState, tensor
using Flowfusion: RDNFlow, MaskedState, floss, scalefloss, schedule_transform
using Distributions: Uniform, Beta, Poisson
using Flux
using Flux: Zygote
using CUDA
using Optimisers
using CannotWaitForTheseOptimisers: Muon
using LearningSchedules: burnin_learning_schedule, next_rate
using Statistics
using Random
using Printf
using JLD2
using Dates

# Include branching module
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

# Override unsafe_free! for proper GPU memory management with DataLoader
Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

Random.seed!(42)

# ============================================================================
# Configuration (matching train_branching_full.jl exactly)
# ============================================================================
shard_dir = get(ENV, "SHARD_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards"))
weights_dir = get(ENV, "WEIGHTS_DIR", joinpath(@__DIR__, "..", "weights"))
n_batches_per_size = parse(Int, get(ENV, "N_BATCHES_PER_SIZE", "50"))
latent_dim = 8
X0_mean_length = 100
deletion_pad = 1.1
ca_loss_scale = 2.0f0
ll_loss_scale = 0.1f0

const TILE_SIZE = 64
batch_sizes = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32]

println("=" ^ 70)
println("Batch Size Sweep — Per-Sample Efficiency")
println("=" ^ 70)
println("Batch sizes: $batch_sizes")
println("Batches per size: $n_batches_per_size")
println("cuTile available: $(LaProteina._HAS_CUTILE)")
println("No overrides: $(LaProteina._NO_OVERRIDES)")
println("Tile padding: $TILE_SIZE")

# ============================================================================
# GPU Check
# ============================================================================
println("\n=== GPU Status ===")
@assert CUDA.functional() "CUDA not functional!"
println("Device: $(CUDA.device())")
println("Memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB available")
LaProteina.enable_tf32!()
println("TF32 math mode: $(CUDA.math_mode())")
dev = gpu

# ============================================================================
# Load Data
# ============================================================================
println("\n=== Loading Data ===")
shard_files = filter(f -> startswith(f, "train_shard_") && endswith(f, ".jld2"), readdir(shard_dir))
sort!(shard_files)
println("Found $(length(shard_files)) shard files")

all_proteins = []
for (shard_idx, shard_file) in enumerate(shard_files)
    shard_path = joinpath(shard_dir, shard_file)
    proteins = load_precomputed_shard(shard_path)
    append!(all_proteins, proteins)
    println("  Shard $shard_idx: $(length(proteins)) proteins (total: $(length(all_proteins)))")
end
println("Total proteins: $(length(all_proteins))")

# ============================================================================
# Create Model (once)
# ============================================================================
println("\n=== Creating Model ===")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)

base_weights_path = joinpath(weights_dir, "score_network.npz")
indel_weights_path = joinpath(weights_dir, "branching_indel_stage1.jld2")
load_branching_weights!(model, indel_weights_path; base_weights_path=base_weights_path)

base_model_cpu = deepcopy(model.base)
model = dev(model)
println("Model loaded and on GPU")

# ============================================================================
# Optimizer (once — same as train_branching_full.jl)
# ============================================================================
sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.99995f0)
opt_state = Flux.setup(Muon(eta=sched.lr), model)

# ============================================================================
# Branching Flow Process
# ============================================================================
P_ca = RDNFlow(3;
    zero_com=false,
    schedule=:log, schedule_param=2.0f0,
    sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)
P_ll = RDNFlow(latent_dim;
    zero_com=false,
    schedule=:power, schedule_param=2.0f0,
    sde_gt_mode=:tan, sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)
branch_time_dist = Beta(1.0, 2.0)
P = CoalescentFlow((P_ca, P_ll), branch_time_dist)

function X0_sampler_train(root)
    ca = ContinuousState(randn(Float32, 3, 1, 1))
    ll = ContinuousState(randn(Float32, latent_dim, 1, 1))
    return (ca, ll)
end

# ============================================================================
# Batch Preparation (identical to train_branching_full.jl)
# ============================================================================
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

"""Pad all batch tensors so seq_len is a multiple of `pad_to`. No-op when already aligned."""
function pad_batch_to(bd, pad_to::Int)
    pad_to <= 0 && return bd
    L = size(bd.mask, 1)
    rem = L % pad_to
    rem == 0 && return bd
    pad_len = pad_to - rem
    L_new = L + pad_len
    B = size(bd.mask, 2)

    pad3(x) = cat(x, zeros(Float32, size(x,1), pad_len, size(x,3)); dims=2)
    pad2(x) = cat(x, zeros(Float32, pad_len, B); dims=1)
    function pad4_pair(x)
        x2 = cat(x, zeros(Float32, size(x,1), pad_len, size(x,3), B); dims=2)
        cat(x2, zeros(Float32, size(x2,1), L_new, pad_len, B); dims=3)
    end

    rf = bd.raw_features
    new_rf = LaProteina.ScoreNetworkRawFeatures(
        pad3(rf.seq_raw), pad3(rf.cond_raw),
        pad4_pair(rf.pair_raw), pad4_pair(rf.pair_cond_raw),
        pad2(rf.mask)
    )

    new_cpu_batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => pad3(bd.cpu_batch[:x_t][:bb_ca]),
                      :local_latents => pad3(bd.cpu_batch[:x_t][:local_latents])),
        :t => bd.cpu_batch[:t],
        :mask => pad2(bd.cpu_batch[:mask])
    )

    return (
        raw_features = new_rf, cpu_batch = new_cpu_batch,
        xt_ca = pad3(bd.xt_ca), xt_ll = pad3(bd.xt_ll),
        mask = pad2(bd.mask), combined_mask = pad2(bd.combined_mask),
        x1_ca_target = pad3(bd.x1_ca_target), x1_ll_target = pad3(bd.x1_ll_target),
        split_target = pad2(bd.split_target), del_target = pad2(bd.del_target),
        t_vec = bd.t_vec, t_ca = bd.t_ca, t_ll = bd.t_ll
    )
end

function prepare_training_batch(indices, proteins_ref, P_ref, X0_sampler, base_model; pad_to::Int=0)
    X1s = [protein_to_X1_simple(proteins_ref[i]) for i in indices]

    t_dist = Uniform(0f0, 1f0)
    batch = branching_bridge(
        P_ref, X0_sampler, X1s, t_dist;
        coalescence_factor = 1.0,
        use_branching_time_prob = 0.5,
        length_mins = Poisson(X0_mean_length),
        deletion_pad = deletion_pad
    )

    L_batch, B = size(batch.Xt.groupings)
    t_vec_cpu = Float32.(batch.t)

    P_ca_ref = P_ref.P[1]
    P_ll_ref = P_ref.P[2]
    t_ca_cpu = Float32.(schedule_transform.(Ref(P_ca_ref), t_vec_cpu))
    t_ll_cpu = Float32.(schedule_transform.(Ref(P_ll_ref), t_vec_cpu))

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

    x1_ca_target_cpu = tensor(batch.X1anchor[1])
    x1_ll_target_cpu = tensor(batch.X1anchor[2])
    if ndims(x1_ca_target_cpu) == 4
        x1_ca_target_cpu = dropdims(x1_ca_target_cpu, dims=2)
        x1_ll_target_cpu = dropdims(x1_ll_target_cpu, dims=2)
    end

    split_target_cpu = Float32.(batch.splits_target)
    del_target_cpu = Float32.(batch.del)

    cpu_batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
        :t => Dict(:bb_ca => t_ca_cpu, :local_latents => t_ll_cpu),
        :mask => mask_cpu
    )

    raw_features = extract_raw_features(base_model, cpu_batch)

    bd = (
        raw_features = raw_features,
        cpu_batch = cpu_batch,
        xt_ca = xt_ca_cpu,
        xt_ll = xt_ll_cpu,
        mask = mask_cpu,
        combined_mask = combined_mask_cpu,
        x1_ca_target = x1_ca_target_cpu,
        x1_ll_target = x1_ll_target_cpu,
        split_target = split_target_cpu,
        del_target = del_target_cpu,
        t_vec = t_vec_cpu,
        t_ca = t_ca_cpu,
        t_ll = t_ll_cpu
    )

    return pad_batch_to(bd, pad_to)
end

# BatchDataset struct
struct BatchDataset{P, S, M}
    batch_indices::Vector{Vector{Int}}
    proteins::P
    process::S
    base_model::M
    pad_to::Int
end

Base.length(x::BatchDataset) = length(x.batch_indices)
function Base.getindex(x::BatchDataset, i::Int)
    prepare_training_batch(x.batch_indices[i], x.proteins, x.process, X0_sampler_train, x.base_model; pad_to=x.pad_to)
end

# ============================================================================
# Sweep: single continuous training loop, switching batch size periodically
# ============================================================================
println("\n=== Starting Batch Size Sweep ===\n")

results = NamedTuple[]

global_batch_idx = 0

for bs in batch_sizes
    println("-" ^ 50)
    println("Batch size: $bs")
    println("-" ^ 50)

    GC.gc()
    CUDA.reclaim()
    println("  GPU memory available: $(round(CUDA.available_memory() / 1e9, digits=2)) GB")

    # New DataLoader for this batch size
    batch_indices_bs = [rand(1:length(all_proteins), bs) for _ in 1:n_batches_per_size]
    dataset_bs = BatchDataset(batch_indices_bs, all_proteins, P, base_model_cpu, TILE_SIZE)
    dataloader_bs = Flux.DataLoader(dataset_bs; batchsize=-1, parallel=true)

    batch_times_bs = Float64[]
    gpu_times_bs = Float64[]  # GPU-only time (transfer + fwd + bwd + optim)
    seq_lens_bs = Int[]
    oom = false

    for (batch_idx, bd_cpu) in enumerate(dataloader_bs)
        global global_batch_idx += 1
        t_batch = time()

        try
            t_gpu_start = time()
            bd = dev(bd_cpu)
            raw_features_for_training = dev(bd.raw_features)
            L_batch = size(bd.mask, 1)
            push!(seq_lens_bs, L_batch)

            # Self-conditioning (1 pass)
            out_sc = forward_branching_from_raw_features_gpu(model, raw_features_for_training)
            v_ca_sc = out_sc[:bb_ca][:v]
            v_ll_sc = out_sc[:local_latents][:v]
            t_ca_exp_sc = reshape(bd.t_ca, 1, 1, :)
            t_ll_exp_sc = reshape(bd.t_ll, 1, 1, :)
            x1_ca_sc = bd.xt_ca .+ (1f0 .- t_ca_exp_sc) .* v_ca_sc
            x1_ll_sc = bd.xt_ll .+ (1f0 .- t_ll_exp_sc) .* v_ll_sc
            x_sc = Dict(:bb_ca => cpu(x1_ca_sc), :local_latents => cpu(x1_ll_sc))

            batch_with_sc = Dict{Symbol, Any}(
                :x_t => Dict(:bb_ca => bd_cpu.xt_ca, :local_latents => bd_cpu.xt_ll),
                :t => Dict(:bb_ca => bd_cpu.t_ca, :local_latents => bd_cpu.t_ll),
                :mask => bd_cpu.mask,
                :x_sc => x_sc
            )
            raw_features_for_training = dev(extract_raw_features(base_model_cpu, batch_with_sc))

            # Forward + backward + optimizer step
            loss, grads = Flux.withgradient(model) do m
                out = forward_branching_from_raw_features_gpu(m, raw_features_for_training)

                v_ca = out[:bb_ca][:v]
                v_ll = out[:local_latents][:v]
                t_ca_exp = reshape(bd.t_ca, 1, 1, :)
                t_ll_exp = reshape(bd.t_ll, 1, 1, :)
                x1_ca = bd.xt_ca .+ (1f0 .- t_ca_exp) .* v_ca
                x1_ll = bd.xt_ll .+ (1f0 .- t_ll_exp) .* v_ll

                t_scale_ca = 1f0 ./ max.(1f0 .- bd.t_ca, 0.1f0).^2
                t_scale_ll = 1f0 ./ max.(1f0 .- bd.t_ll, 0.1f0).^2

                ca_diff = (x1_ca .- bd.x1_ca_target).^2
                ca_loss = sum(ca_diff .* reshape(bd.mask, 1, size(bd.mask)...) .* reshape(t_scale_ca, 1, 1, :)) / sum(bd.mask)

                ll_diff = (x1_ll .- bd.x1_ll_target).^2
                ll_loss = sum(ll_diff .* reshape(bd.mask, 1, size(bd.mask)...) .* reshape(t_scale_ll, 1, 1, :)) / sum(bd.mask)

                indel_scale = scalefloss(P, bd.t_vec, 1, 0.2f0)
                split_l = floss(P, out[:split], bd.split_target, bd.combined_mask, indel_scale)
                del_l = floss(P.deletion_policy, out[:del], bd.del_target, bd.combined_mask, indel_scale)

                ca_scaled = ca_loss * ca_loss_scale
                ll_scaled = ll_loss * ll_loss_scale
                ca_clamped = isfinite(ca_scaled) ? softclamp(ca_scaled) : 0.0f0
                ll_clamped = isfinite(ll_scaled) ? softclamp(ll_scaled) : 0.0f0
                split_clamped = isfinite(split_l) ? softclamp(split_l) : 0.0f0
                del_clamped = isfinite(del_l) ? softclamp(del_l) : 0.0f0

                total_loss = ca_clamped + ll_clamped + split_clamped + del_clamped
                min(total_loss, 20.0f0)
            end

            Flux.update!(opt_state, model, grads[1])
            CUDA.synchronize()
            gpu_time = (time() - t_gpu_start) * 1000
            push!(gpu_times_bs, gpu_time)

            # LR schedule: adjust every 10 global batches (matching original)
            if global_batch_idx % 10 == 0
                Flux.adjust!(opt_state, next_rate(sched))
            end

            batch_time = (time() - t_batch) * 1000
            push!(batch_times_bs, batch_time)

            loss_val = Float32(cpu(loss))
            if batch_idx <= 3 || batch_idx % 10 == 0
                @printf("  Batch %3d (global %d): loss=%.3f, lr=%.2e, time=%.0f ms (gpu=%.0f ms, %.1f ms/sample), L=%d\n",
                        batch_idx, global_batch_idx, loss_val, sched.lr, batch_time, gpu_time, batch_time / bs, L_batch)
            end

            # Bail early if model has gone NaN
            if !isfinite(loss_val)
                println("  WARNING: NaN/Inf loss at batch $(batch_idx), model may have diverged")
            end

        catch e
            if isa(e, CUDA.OutOfMemoryError) || (isa(e, ErrorException) && occursin("out of memory", string(e)))
                println("  OOM at batch $(batch_idx)! Stopping this batch size.")
                oom = true
                GC.gc()
                CUDA.reclaim()
                break
            else
                rethrow(e)
            end
        end
    end

    if !isempty(batch_times_bs)
        # Skip first batch (warmup for new batch size shape)
        times = length(batch_times_bs) > 1 ? batch_times_bs[2:end] : batch_times_bs
        gpu_t = length(gpu_times_bs) > 1 ? gpu_times_bs[2:end] : gpu_times_bs
        med = median(times)
        avg = mean(times)
        gpu_med = median(gpu_t)
        gpu_avg = mean(gpu_t)
        avg_L = isempty(seq_lens_bs) ? 0 : round(Int, mean(seq_lens_bs))
        n_done = length(batch_times_bs)

        push!(results, (
            batch_size = bs,
            median_ms = med,
            mean_ms = avg,
            gpu_median_ms = gpu_med,
            gpu_mean_ms = gpu_avg,
            per_sample_median = med / bs,
            per_sample_mean = avg / bs,
            avg_seq_len = avg_L,
            n_completed = n_done
        ))

        @printf("  => BS=%d: total median=%.0f ms (gpu=%.0f ms), per-sample=%.1f ms, avg_L=%d (%d/%d batches)\n",
                bs, med, gpu_med, med / bs, avg_L, n_done, n_batches_per_size)
    else
        println("  => BS=$bs: No batches completed (OOM on first batch)")
    end

    if oom
        println("\nStopping sweep — OOM at batch size $bs")
        break
    end
end

# ============================================================================
# Summary Table
# ============================================================================
println("\n" * "=" ^ 70)
println("BATCH SIZE SWEEP RESULTS")
println("=" ^ 70)
println("GPU path: $(LaProteina._NO_OVERRIDES ? "original (no overrides)" : (LaProteina._HAS_CUTILE ? "cuTile" : "nocutile"))")
println("TF32: enabled")
println("Batches per size: $n_batches_per_size (first excluded from stats)")
println("Total training batches: $global_batch_idx")
println("Final LR: $(sched.lr)")
println()

@printf("%-6s  %-10s  %-10s  %-10s  %-12s  %-12s  %-6s  %-9s\n",
        "BS", "Total(ms)", "GPU(ms)", "Prep(ms)", "/samp(ms)", "GPU/samp", "AvgL", "Done")
@printf("%-6s  %-10s  %-10s  %-10s  %-12s  %-12s  %-6s  %-9s\n",
        "------", "----------", "----------", "----------", "------------", "------------", "------", "---------")

best_per_sample = Inf
best_bs = 0
for r in results
    prep_med = r.median_ms - r.gpu_median_ms
    @printf("%-6d  %-10.0f  %-10.0f  %-10.0f  %-12.1f  %-12.1f  %-6d  %d/%d\n",
            r.batch_size, r.median_ms, r.gpu_median_ms, prep_med,
            r.per_sample_median, r.gpu_median_ms / r.batch_size,
            r.avg_seq_len, r.n_completed, n_batches_per_size)
    if r.per_sample_median < best_per_sample
        best_per_sample = r.per_sample_median
        best_bs = r.batch_size
    end
end

println()
@printf("Best per-sample efficiency: BS=%d at %.1f ms/sample (median total)\n", best_bs, best_per_sample)
println("\nNote: Prep(ms) = Total - GPU = time waiting for DataLoader (batch prep on CPU)")
println("If Prep >> GPU, the data loader is the bottleneck.")
println("\nDone!")
