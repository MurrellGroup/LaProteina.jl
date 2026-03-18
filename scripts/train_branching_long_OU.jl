#!/usr/bin/env julia
# Train Branching Flows model for long proteins (300-896 residues)
# Fine-tunes from LD3 pretrained weights using AE2-precomputed shards
#
# Key differences from train_branching_full_OU.jl:
#   - Uses LD3 base weights (trained on proteins up to 800 residues)
#   - Uses AE2 precomputed shards (256-896 residue proteins)
#   - batch_size=2 (pair tensor at L=800 is ~655 MB, too large for bigger batches)
#   - Uses AE2 decoder for sampling (consistent with LD3 latent space)
#
# Usage:
#   julia -t 8 --project=/home/claudey/JuProteina/run scripts/train_branching_long_OU.jl
#
# Environment variables:
#   SHARD_DIR - Directory containing AE2 precomputed shards
#   BATCH_SIZE - Batch size (default: 2, max safe for L~800)
#   N_BATCHES - Number of training batches (default: 40000)
#   WARMDOWN_BATCHES - Linear warmdown batches at end (default: 5000)
#   SAMPLE_EVERY - Save checkpoint every N batches (default: 1000)
#   RESUME_FROM - Path to checkpoint JLD2 to resume from (loads model + optimizer state)

using LaProteina
using OnionTile
using LaProteina: ScoreNetworkRawFeatures, extract_raw_features, cpu
using LaProteina: compute_sc_feature_offsets, update_sc_raw_features!
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow, branching_bridge, Deletion
using ForwardBackward: ContinuousState, DiscreteState, tensor, OUBridgeExpVar
using Flowfusion: MaskedState, floss, scalefloss
using Distributions: Uniform, Beta, Poisson
using Flux
using Flux: Zygote
using CUDA
using Optimisers
using CannotWaitForTheseOptimisers: Muon
using LearningSchedules: burnin_learning_schedule, linear_decay_schedule, next_rate
using Statistics
using Random
using Printf
using JLD2
using Dates

Random.seed!(42)

println("=" ^ 70)
println("Branching Flows Training - Long Proteins (LD3 + AE2)")
println("=" ^ 70)

# ============================================================================
# Configuration
# ============================================================================
shard_dir = get(ENV, "SHARD_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards_ae2_long"))
checkpoints_dir = "/home/claudey/JuProteina/la-proteina/checkpoints_laproteina"
indel_weights_dir = "/home/claudey/JuProteina/ArchivedJuProteina/weights"
output_dir = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "..", "outputs", "branching_long_OU_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))
batch_size = parse(Int, get(ENV, "BATCH_SIZE", "2"))
n_batches = parse(Int, get(ENV, "N_BATCHES", "40000"))
warmdown_batches = parse(Int, get(ENV, "WARMDOWN_BATCHES", "5000"))
named_checkpoint_every = parse(Int, get(ENV, "NAMED_CKPT_EVERY", "10000"))
rolling_checkpoint_every = parse(Int, get(ENV, "ROLLING_CKPT_EVERY", "500"))
resume_from = get(ENV, "RESUME_FROM", "")
start_batch = parse(Int, get(ENV, "START_BATCH", "1"))
warmdown_only = haskey(ENV, "WARMDOWN_ONLY")  # If set, override schedule to pure warmdown
latent_dim = 8

# LR schedule — start lower for fine-tuning from LD3
if warmdown_only
    # Pure warmdown: start from peak LR, linearly decay to ~0 over n_batches
    # LR adjusted every 10 batches, so n_batches ÷ 10 LR steps
    warmdown_start_lr = parse(Float32, get(ENV, "WARMDOWN_START_LR", "0.000089"))
    sched = linear_decay_schedule(warmdown_start_lr, 0.000000001f0, n_batches ÷ 10)
    warmdown_batches = n_batches  # entire run is warmdown
    println("WARMDOWN ONLY mode: LR $(warmdown_start_lr) → 0 over $n_batches batches")
else
    sched = burnin_learning_schedule(0.000005f0, 0.000100f0, 1.05f0, 0.99995f0)
end

# Branching parameters
X0_mean_length = 0  # Poisson(0) → start from L=1
deletion_pad = 1.1

# Loss calibration (may need tuning for longer sequences)
ca_calibration = 1.0f0
ll_calibration = 3.0f0

mkpath(output_dir)
mkpath(joinpath(output_dir, "checkpoints"))

log_file = joinpath(output_dir, "training_log.txt")

println("\n=== Configuration ===")
println("Shard dir: $shard_dir")
println("Output dir: $output_dir")
println("Batch size: $batch_size")
println("N batches: $n_batches")
println("Warmdown: $warmdown_batches$(warmdown_only ? " (WARMDOWN ONLY)" : "")")
println("LR schedule: $(warmdown_only ? "linear_decay($(sched.lr), 1e-9, $(n_batches ÷ 10))" : "burnin(5e-6, 1e-4, 1.05, 0.99995)")")
println("Start batch: $start_batch")
if !isempty(resume_from)
    println("Resume from: $resume_from")
end
println("Log file: $log_file")

log_mode = start_batch > 1 ? "a" : "w"
open(log_file, log_mode) do io
    println(io, "# Branching Flows Long Protein Training (LD3 + AE2)")
    println(io, "# Started: $(now()), start_batch=$start_batch")
    println(io, "# Config: batch_size=$batch_size, n_batches=$n_batches, warmdown=$warmdown_batches")
    println(io, "# Base: LD3_ucond_notri_800, Encoder: AE2_ucond_800")
    println(io, "# LR: burnin(5e-6, 1e-4, 1.05, 0.99995)")
    println(io, "#")
    if start_batch == 1
        println(io, "# Columns: batch,lr,total_loss,ca_scaled,ll_scaled,split,del,t_min,t_max,time_ms,seq_len")
    end
end

# ============================================================================
# GPU Check
# ============================================================================
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: $(CUDA.device())")
    println("Memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB available")
    LaProteina.enable_tf32!()
    dev = gpu
else
    error("GPU required for long protein training")
end

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
    println("Loading shard $shard_idx: $shard_file")
    proteins = load_precomputed_shard(shard_path)
    append!(all_proteins, proteins)
    println("  Loaded $(length(proteins)) proteins (total: $(length(all_proteins)))")
end
println("Total proteins loaded: $(length(all_proteins))")

# Report length distribution
all_lengths = [length(p.mask) for p in all_proteins]
println("Length distribution: min=$(minimum(all_lengths)), max=$(maximum(all_lengths)), mean=$(round(mean(all_lengths), digits=0)), median=$(round(median(all_lengths), digits=0))")

# ============================================================================
# Create Model — LD3 architecture (same as LD1, no triangle updates)
# ============================================================================
println("\n=== Creating Model ===")

base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false, cropped_flag=false
)
model = BranchingScoreNetwork(base)

if !isempty(resume_from)
    # Resume from checkpoint (saved by save_branching_weights with include_base=true)
    println("Resuming from checkpoint: $resume_from")
    ckpt = load(resume_from)
    if haskey(ckpt, "base")
        Flux.loadmodel!(model.base, ckpt["base"])
        println("  base loaded")
    end
    Flux.loadmodel!(model.indel_time_proj, ckpt["indel_time_proj"])
    Flux.loadmodel!(model.split_head, ckpt["split_head"])
    Flux.loadmodel!(model.del_head, ckpt["del_head"])
    println("  Full model loaded from checkpoint")
else
    # Load LD3 pretrained base weights
    ld3_path = joinpath(checkpoints_dir, "LD3_ucond_notri_800.safetensors")
    println("Loading LD3 base weights from: $ld3_path")
    load_score_network_weights_st!(model.base, "LD3_ucond_notri_800.safetensors"; local_path=ld3_path)
    println("  LD3 base weights loaded")

    # Branching heads: random init (scaled down 20x by BranchingScoreNetwork constructor)
    # Cannot reuse LD1-trained indel heads — LD3 has a different token embedding space
    println("  Branching heads: random init (will train from scratch on LD3 embeddings)")
end

# CPU copy for feature extraction
base_model_cpu = deepcopy(model.base)
model = dev(model)
println("Model moved to GPU")

sc_offsets = compute_sc_feature_offsets(base_model_cpu)
println("SC feature offsets: seq=$(sc_offsets.seq), pair=$(sc_offsets.pair)")

# Advance LR schedule if resuming (skip for warmdown_only — already set up)
warmdown_start = n_batches - warmdown_batches
if start_batch > 1 && !warmdown_only
    for i in 1:((start_batch - 1) ÷ 10)
        global sched
        if i * 10 == warmdown_start
            sched = linear_decay_schedule(sched.lr, 0.000000001f0, warmdown_batches ÷ 10)
        end
        next_rate(sched)
    end
    println("LR schedule advanced to batch $start_batch: lr=$(sched.lr)")
end

# Optimizer
opt_state = Flux.setup(Muon(eta=sched.lr), model)
println("Optimizer: Muon (lr=$(sched.lr))")

# Try to load optimizer state if resuming
if !isempty(resume_from)
    opt_dir = dirname(resume_from)
    opt_file = replace(basename(resume_from), "checkpoint_" => "opt_state_")
    opt_path = joinpath(opt_dir, opt_file)
    if isfile(opt_path)
        println("Loading optimizer state from: $opt_path")
        opt_saved = load(opt_path, "opt_state")
        Flux.loadmodel!(opt_state, opt_saved)
        println("  Optimizer state restored")
    else
        println("  No optimizer state found, starting fresh (momentum buffers reset)")
    end
end

# ============================================================================
# Create Branching Flow Process
# ============================================================================
println("\n=== Creating CoalescentFlow ===")
P_ca = OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)
P_ll = OUBridgeExpVar(100f0, 50f0, 0.000000001f0, dec = -0.1f0)
branch_time_dist = Beta(1.0, 2.0)
P = CoalescentFlow((P_ca, P_ll), branch_time_dist)
println("CoalescentFlow created")

function X0_sampler_train(root)
    ca = ContinuousState(randn(Float32, 3, 1, 1))
    ll = ContinuousState(randn(Float32, latent_dim, 1, 1))
    return (ca, ll)
end

# ============================================================================
# Batch Preparation
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

function prepare_training_batch(indices, proteins_ref, P_ref, X0_sampler, base_model)
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
    t_ca_cpu = t_vec_cpu
    t_ll_cpu = t_vec_cpu

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

    return bd
end

# ============================================================================
# DataLoader
# ============================================================================
println("\n=== Setting up DataLoader ===")

batch_indices = Vector{Vector{Int}}(undef, n_batches)
for i in 1:n_batches
    batch_indices[i] = rand(1:length(all_proteins), batch_size)
end
println("Pre-sampled $n_batches batch index sets (batch_size=$batch_size)")

struct BatchDataset{P, S, M}
    batch_indices::Vector{Vector{Int}}
    proteins::P
    process::S
    base_model::M
end

Base.length(x::BatchDataset) = length(x.batch_indices)
function Base.getindex(x::BatchDataset, i::Int)
    prepare_training_batch(x.batch_indices[i], x.proteins, x.process, X0_sampler_train, x.base_model)
end

# Slice batch_indices to skip already-completed batches (only if resuming within same run)
if start_batch <= n_batches
    active_indices = batch_indices[start_batch:end]
else
    # start_batch > n_batches means we're continuing numbering from a previous run
    # (e.g. warmdown_only starting at batch 30001 with n_batches=5000)
    active_indices = batch_indices
end
dataset = BatchDataset(active_indices, all_proteins, P, base_model_cpu)
dataloader = Flux.DataLoader(dataset; batchsize=-1, parallel=false)
println("DataLoader ready ($(length(dataset)) batches, starting at $start_batch)")

# ============================================================================
# Training Loop
# ============================================================================
warmdown_start = n_batches - warmdown_batches

println("\n=== Training ===")
println("Batch size: $batch_size, N batches: $n_batches")
println("Starting training loop...")

losses = Float32[]
ca_losses = Float32[]
ll_losses = Float32[]
split_losses = Float32[]
del_losses = Float32[]
batch_times = Float64[]

t_train_start = time()

last_saved_batch = 0

for (loop_idx, bd_cpu) in enumerate(dataloader)
    batch_idx = loop_idx + start_batch - 1
    t_batch = time()

    ca_loss_ref = Ref(0f0)
    ll_loss_ref = Ref(0f0)
    split_loss_ref = Ref(0f0)
    del_loss_ref = Ref(0f0)
    t_min_ref = Ref(0f0)
    t_max_ref = Ref(0f0)

    # Wrap GPU operations in try-catch for CUDA error recovery
    local loss
    cuda_error = false
    try
        bd = dev(bd_cpu)

        # Self-conditioning
        raw_features_for_training = bd.raw_features
        n_sc_passes = rand(Poisson(1))
        if n_sc_passes > 0
            for _ in 1:n_sc_passes
                out_sc = forward_branching_from_raw_features_gpu(model, raw_features_for_training)
                v_ca_sc = out_sc[:bb_ca][:v]
                v_ll_sc = out_sc[:local_latents][:v]
                t_ca_exp_sc = reshape(bd.t_ca, 1, 1, :)
                t_ll_exp_sc = reshape(bd.t_ll, 1, 1, :)
                x1_ca_sc = bd.xt_ca .+ (1f0 .- t_ca_exp_sc) .* v_ca_sc
                x1_ll_sc = bd.xt_ll .+ (1f0 .- t_ll_exp_sc) .* v_ll_sc
                update_sc_raw_features!(raw_features_for_training, sc_offsets, x1_ca_sc, x1_ll_sc)
            end
        end

        # Loss + gradients
        loss, grads = Flux.withgradient(model) do m
            out = forward_branching_from_raw_features_gpu(m, raw_features_for_training)

            v_ca = out[:bb_ca][:v]
            v_ll = out[:local_latents][:v]
            t_ca_exp = reshape(bd.t_ca, 1, 1, :)
            t_ll_exp = reshape(bd.t_ll, 1, 1, :)
            x1_ca = bd.xt_ca .+ (1f0 .- t_ca_exp) .* v_ca
            x1_ll = bd.xt_ll .+ (1f0 .- t_ll_exp) .* v_ll

            ca_target = MaskedState(ContinuousState(bd.x1_ca_target), bd.combined_mask, bd.combined_mask)
            ll_target = MaskedState(ContinuousState(bd.x1_ll_target), bd.combined_mask, bd.combined_mask)

            ca_scale = scalefloss(P.P[1], bd.t_vec, 1, 0.2f0)
            ll_scale = scalefloss(P.P[2], bd.t_vec, 1, 0.2f0)
            indel_scale = scalefloss(P.P[1], bd.t_vec, 1, 0.2f0)

            ca_loss = floss(P.P[1], x1_ca, ca_target, ca_scale) * ca_calibration
            ll_loss = floss(P.P[2], x1_ll, ll_target, ll_scale) * ll_calibration
            split_l = floss(P, out[:split], bd.split_target, bd.combined_mask, indel_scale)
            del_l = floss(P.deletion_policy, out[:del], bd.del_target, bd.combined_mask, indel_scale)

            Zygote.ignore() do
                ca_loss_ref[] = Float32(cpu(ca_loss))
                ll_loss_ref[] = Float32(cpu(ll_loss))
                split_loss_ref[] = Float32(cpu(split_l))
                del_loss_ref[] = Float32(cpu(del_l))
                t_min_ref[] = Float32(minimum(cpu(bd.t_vec)))
                t_max_ref[] = Float32(maximum(cpu(bd.t_vec)))
            end

            ca_clamped = isfinite(ca_loss) ? LaProteina.softclamp(ca_loss) : 0.0f0
            ll_clamped = isfinite(ll_loss) ? LaProteina.softclamp(ll_loss) : 0.0f0
            split_clamped = isfinite(split_l) ? LaProteina.softclamp(split_l) : 0.0f0
            del_clamped = isfinite(del_l) ? LaProteina.softclamp(del_l) : 0.0f0

            total_loss = ca_clamped + ll_clamped + split_clamped + del_clamped
            min(total_loss, 20.0f0)
        end

        if !isfinite(Float64(cpu(loss)))
            L_batch = size(bd.mask, 1)
            @printf("  FATAL: NaN/Inf loss at batch %d (L=%d). Stopping.\n", batch_idx, L_batch)
            open(log_file, "a") do io
                @printf(io, "# STOPPED: NaN/Inf loss at batch %d, L=%d\n", batch_idx, L_batch)
            end
            break
        end

        Flux.update!(opt_state, model, grads[1])

        # LR schedule (adjust every 10 batches)
        if batch_idx % 10 == 0
            if !warmdown_only && batch_idx >= warmdown_start && batch_idx < warmdown_start + 10
                global sched
                sched = linear_decay_schedule(sched.lr, 0.000000001f0, warmdown_batches ÷ 10)
            end
            Flux.adjust!(opt_state, next_rate(sched))
        end

        batch_time = (time() - t_batch) * 1000
        push!(losses, Float32(cpu(loss)))
        push!(ca_losses, ca_loss_ref[])
        push!(ll_losses, ll_loss_ref[])
        push!(split_losses, split_loss_ref[])
        push!(del_losses, del_loss_ref[])
        push!(batch_times, batch_time)

        L_log = size(bd.mask, 1)
        open(log_file, "a") do io
            @printf(io, "%d,%.2e,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.0f,%d\n",
                    batch_idx, sched.lr,
                    losses[end], ca_losses[end], ll_losses[end], split_losses[end], del_losses[end],
                    t_min_ref[], t_max_ref[], batch_time, L_log)
        end

        if batch_idx % 100 == 0 || batch_idx == 1
            avg_loss = mean(losses[max(1, end-99):end])
            avg_ca = mean(ca_losses[max(1, end-99):end])
            avg_ll = mean(ll_losses[max(1, end-99):end])
            avg_split = mean(split_losses[max(1, end-99):end])
            avg_del = mean(del_losses[max(1, end-99):end])
            avg_time = mean(batch_times[max(1, end-99):end])

            wd_status = batch_idx >= warmdown_start ? " (warmdown)" : ""

            @printf("Batch %6d: loss=%.3f (ca=%.3f, ll=%.3f, split=%.3f, del=%.3f), lr=%.2e%s, L=%d, time=%.0fms\n",
                    batch_idx, avg_loss, avg_ca, avg_ll, avg_split, avg_del, sched.lr, wd_status, L_log, avg_time)
            flush(stdout)

            GC.gc()
            CUDA.reclaim()
        end

        # Checkpoint: rolling current/previous every rolling_checkpoint_every,
        # named checkpoints every named_checkpoint_every
        if batch_idx == 1 || batch_idx % rolling_checkpoint_every == 0
            ckpt_dir = joinpath(output_dir, "checkpoints")
            model_cpu = cpu(model)
            opt_st_cpu = cpu(Flux.state(opt_state))

            # Rolling: current → previous, then write new current
            cur_ckpt = joinpath(ckpt_dir, "checkpoint_current.jld2")
            cur_opt = joinpath(ckpt_dir, "opt_state_current.jld2")
            prev_ckpt = joinpath(ckpt_dir, "checkpoint_previous.jld2")
            prev_opt = joinpath(ckpt_dir, "opt_state_previous.jld2")
            if isfile(cur_ckpt)
                mv(cur_ckpt, prev_ckpt; force=true)
                isfile(cur_opt) && mv(cur_opt, prev_opt; force=true)
            end
            save_branching_weights(model_cpu, cur_ckpt; include_base=true)
            jldsave(cur_opt; opt_state=opt_st_cpu, batch_idx=batch_idx)
            println("  Saved rolling checkpoint at batch $batch_idx")

            # Named: permanent checkpoint at milestone batches
            if batch_idx % named_checkpoint_every == 0
                named_ckpt = joinpath(ckpt_dir, @sprintf("checkpoint_batch%06d.jld2", batch_idx))
                named_opt = joinpath(ckpt_dir, @sprintf("opt_state_batch%06d.jld2", batch_idx))
                cp(cur_ckpt, named_ckpt; force=true)
                cp(cur_opt, named_opt; force=true)
                println("  Saved named checkpoint: $named_ckpt")
            end

            global last_saved_batch = batch_idx
        end
    catch e
        if isa(e, CUDA.CuError)
            @printf("  CUDA error at batch %d: %s\n", batch_idx, e)
            open(log_file, "a") do io
                @printf(io, "# CUDA ERROR at batch %d: %s\n", batch_idx, string(e))
            end
            cuda_error = true
        else
            rethrow(e)
        end
    end

    if cuda_error
        @printf("  Last saved checkpoint was at batch %d\n", last_saved_batch)
        @printf("  Resume with: RESUME_FROM=<checkpoint_path> START_BATCH=%d\n", last_saved_batch + 1)
        flush(stdout)
        exit(1)
    end

    loop_idx >= length(active_indices) && break
end

t_train = time() - t_train_start

# ============================================================================
# Summary + Save
# ============================================================================
println("\n" * "=" ^ 70)
println("Training Complete")
println("=" ^ 70)

warmup_skip = 10
post_warmup_times = batch_times[min(warmup_skip+1, end):end]
@printf("  Total time: %.1f hours\n", t_train / 3600)
@printf("  Avg batch time (post-warmup): %.1f ms\n", mean(post_warmup_times))
@printf("  Throughput: %.1f samples/sec\n", batch_size / (mean(post_warmup_times) / 1000))

first_1k = losses[1:min(1000, length(losses))]
last_1k = losses[max(1, end-999):end]
@printf("  First 1k loss: %.4f\n", mean(first_1k))
@printf("  Last 1k loss: %.4f\n", mean(last_1k))

model_cpu = cpu(model)
final_path = joinpath(output_dir, "branching_long_final.jld2")
save_branching_weights(model_cpu, final_path; include_base=true)
println("Saved final model: $final_path")
println("Log file: $log_file")
println("Done!")
