#!/usr/bin/env julia
# Train Branching Flows model with motif conditioning (LD4 base)
# Fine-tunes branching heads on top of LD4 (motif_mode=:aa) base model
#
# Usage:
#   LAPROTEINA_NOCUTILE=1 julia -t 8 --project=/home/claudey/JuProteina/run scripts/train_branching_motif.jl
#
# Environment variables:
#   SHARD_DIR - Directory containing AE3 precomputed shards (with atom37 data)
#   CHECKPOINTS_DIR - Directory containing pretrained safetensors weights
#   INDEL_WEIGHTS - Path to branching indel head weights (stage1 JLD2)
#   OUTPUT_DIR - Directory for outputs (logs, checkpoints)
#   BATCH_SIZE - Batch size (default: 6)
#   N_BATCHES - Number of training batches (default: 40000)
#   WARMDOWN_BATCHES - Linear warmdown batches at end (default: 4000)
#   MOTIF_PCT - Max fraction of protein to use as motif (default: 0.3)
#   UNCOND_PROB - Probability of unconditional training (no motif) (default: 0.15)

using LaProteina
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

# Check for OnionTile (optional — nocutile path works without it)
try
    @eval using OnionTile
    println("OnionTile loaded — cuTile path active")
catch
    println("OnionTile not found — using nocutile path")
end

Random.seed!(42)

println("=" ^ 70)
println("Branching Flows Training - Motif Conditioning (LD4 base)")
println("=" ^ 70)

# ============================================================================
# Configuration
# ============================================================================
shard_dir = get(ENV, "SHARD_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards_ae3"))
checkpoints_dir = get(ENV, "CHECKPOINTS_DIR", "/home/claudey/JuProteina/la-proteina/checkpoints_laproteina")
indel_weights_path = get(ENV, "INDEL_WEIGHTS", "/home/claudey/JuProteina/ArchivedJuProteina/weights/branching_indel_stage1.jld2")
output_dir = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "..", "outputs", "branching_motif_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))
batch_size = parse(Int, get(ENV, "BATCH_SIZE", "6"))
n_batches = parse(Int, get(ENV, "N_BATCHES", "40000"))
warmdown_batches = parse(Int, get(ENV, "WARMDOWN_BATCHES", "4000"))
sample_every = parse(Int, get(ENV, "SAMPLE_EVERY", "5000"))
latent_dim = 8

# Motif parameters
motif_max_pct = parse(Float32, get(ENV, "MOTIF_PCT", "0.3"))
uncond_prob = parse(Float32, get(ENV, "UNCOND_PROB", "0.15"))

# Learning rate schedule
sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.99995f0)

# Branching parameters
X0_mean_length = 0  # Poisson(0) → start from length=1
deletion_pad = 1.1

# Loss calibration
ca_calibration = 1.0f0
ll_calibration = 3.0f0

mkpath(output_dir)
mkpath(joinpath(output_dir, "checkpoints"))
log_file = joinpath(output_dir, "training_log.txt")

println("\n=== Configuration ===")
println("Shard dir: $shard_dir")
println("Checkpoints dir: $checkpoints_dir")
println("Indel weights: $indel_weights_path")
println("Output dir: $output_dir")
println("Batch size: $batch_size")
println("N batches: $n_batches")
println("Motif max pct: $motif_max_pct")
println("Uncond prob: $uncond_prob")

open(log_file, "w") do io
    println(io, "# Branching Flows Motif Training Log (LD4 base, OUBridgeExpVar)")
    println(io, "# Started: $(now())")
    println(io, "# Config: batch_size=$batch_size, n_batches=$n_batches, warmdown=$warmdown_batches")
    println(io, "# Motif: max_pct=$motif_max_pct, uncond_prob=$uncond_prob")
    println(io, "#")
    println(io, "# Columns: batch,lr,total_loss,ca_scaled,ll_scaled,split,del,t_min,t_max,time_ms,seq_len,n_motif")
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
    println("TF32 math mode: $(CUDA.math_mode())")
    dev = gpu
else
    println("No CUDA - using CPU")
    dev = identity
end

# ============================================================================
# Load All Shards (AE3 format with atom37 data)
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

# Verify AE3 format (has atom37 fields)
test_p = all_proteins[1]
if !haskey(test_p, :atom37_coords)
    error("Shards missing atom37 data! Use AE3 shards from precompute_ae3_shards.jl")
end
println("Verified: shards contain atom37 data for motif conditioning")

# ============================================================================
# Create Model (LD4 base with motif_mode=:aa)
# ============================================================================
println("\n=== Creating Model ===")

base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false,
    motif_mode=:aa  # LD4: 549D seq features, 301D pair features
)

model = BranchingScoreNetwork(base)

# Load LD4 pretrained base weights
ld4_path = joinpath(checkpoints_dir, "LD4_motif_idx_aa.safetensors")
println("Loading LD4 base weights from: $ld4_path")
load_score_network_weights_st!(model.base, "LD4_motif_idx_aa.safetensors"; local_path=ld4_path)

# Load branching indel head weights from existing checkpoint
println("Loading indel head weights from: $indel_weights_path")
indel_weights = load(indel_weights_path)
if haskey(indel_weights, "indel_time_proj")
    Flux.loadmodel!(model.indel_time_proj, indel_weights["indel_time_proj"])
    println("  Loaded indel_time_proj")
end
if haskey(indel_weights, "split_head")
    Flux.loadmodel!(model.split_head, indel_weights["split_head"])
    println("  Loaded split_head")
end
if haskey(indel_weights, "del_head")
    Flux.loadmodel!(model.del_head, indel_weights["del_head"])
    println("  Loaded del_head")
end

# Save CPU copy for feature extraction
base_model_cpu = deepcopy(model.base)

model = dev(model)
println("Model loaded and moved to device")

sc_offsets = compute_sc_feature_offsets(base_model_cpu)
println("SC feature offsets: seq=$(sc_offsets.seq), pair=$(sc_offsets.pair)")

opt_state = Flux.setup(Muon(eta=sched.lr), model)
println("Optimizer: Muon with burnin_learning_schedule")

# ============================================================================
# Motif Selection
# ============================================================================

"""
    select_random_motif(L::Int; max_pct=0.3f0) -> Vector{Int}

Select random contiguous motif segments totaling up to max_pct of protein length.
Returns vector of 1-indexed motif position indices (empty for unconditional).
"""
function select_random_motif(L::Int; max_pct::Float32=motif_max_pct)
    max_motif = max(1, floor(Int, L * max_pct))
    motif_positions = Int[]

    # Choose 1-3 contiguous segments
    n_segments = rand(1:3)
    remaining = max_motif

    for _ in 1:n_segments
        remaining <= 0 && break

        seg_len = rand(1:remaining)
        start_pos = rand(1:max(1, L - seg_len + 1))
        end_pos = min(start_pos + seg_len - 1, L)

        for p in start_pos:end_pos
            if p ∉ motif_positions
                push!(motif_positions, p)
                remaining -= 1
            end
        end
    end

    sort!(motif_positions)
    return motif_positions
end

# ============================================================================
# Create CoalescentFlow Process
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
# Batch Preparation with Motif Conditioning
# ============================================================================

function protein_to_X1_motif(protein, motif_positions::Vector{Int})
    ca_coords = protein.ca_coords
    z_mean = protein.z_mean
    z_log_scale = protein.z_log_scale
    mask = protein.mask
    L = length(mask)

    z_latent = z_mean .+ randn(Float32, size(z_mean)) .* exp.(z_log_scale)
    valid_mask = Vector{Bool}(mask .> 0.5)

    # branchmask: true for positions that can branch/delete, false for motif positions
    bmask = copy(valid_mask)
    for p in motif_positions
        if 1 <= p <= L
            bmask[p] = false
        end
    end

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
        branchmask = bmask
    )
end

function prepare_training_batch(indices, proteins_ref, P_ref, X0_sampler, base_model)
    B = length(indices)

    # Step 1: Select motif positions for each protein
    motif_positions_per_protein = Vector{Vector{Int}}(undef, B)
    for (b, idx) in enumerate(indices)
        protein = proteins_ref[idx]
        L = length(protein.mask)

        if rand() < uncond_prob
            # Unconditional (no motif) — for classifier-free guidance
            motif_positions_per_protein[b] = Int[]
        else
            motif_positions_per_protein[b] = select_random_motif(L)
        end
    end

    # Step 2: Convert to X1 states with motif-aware branchmask
    X1s = [protein_to_X1_motif(proteins_ref[indices[b]], motif_positions_per_protein[b]) for b in 1:B]

    # Step 3: Run branching_bridge
    t_dist = Uniform(0f0, 1f0)
    batch = branching_bridge(
        P_ref, X0_sampler, X1s, t_dist;
        coalescence_factor = 1.0,
        use_branching_time_prob = 0.5,
        length_mins = Poisson(X0_mean_length),
        deletion_pad = deletion_pad
    )

    L_batch, _ = size(batch.Xt.groupings)
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

    x1_ca_target_cpu = tensor(batch.X1anchor[1])
    x1_ll_target_cpu = tensor(batch.X1anchor[2])
    if ndims(x1_ca_target_cpu) == 4
        x1_ca_target_cpu = dropdims(x1_ca_target_cpu, dims=2)
        x1_ll_target_cpu = dropdims(x1_ll_target_cpu, dims=2)
    end

    split_target_cpu = Float32.(batch.splits_target)
    del_target_cpu = Float32.(batch.del)

    # Step 4: Map motif positions from original protein space to batched state space
    # batch.Xt.ids [L_batch, B] maps each position back to original protein index (1:L)
    # Elements with id=0 are noise inserts / padding
    ids = batch.Xt.ids  # [L_batch, B]

    # Build motif feature arrays in batched space
    x_motif_cpu = zeros(Float32, 3, 37, L_batch, B)     # atom37 coords at motif positions
    motif_mask_cpu = zeros(Float32, 37, L_batch, B)      # atom37 mask at motif positions
    seq_motif_cpu = zeros(Float32, L_batch, B)            # residue types at motif
    seq_motif_mask_cpu = zeros(Float32, L_batch, B)       # per-residue motif mask
    motif_loss_mask_cpu = zeros(Float32, L_batch, B)      # 1 at motif positions (for loss exclusion)
    n_motif_total = 0

    for b in 1:B
        protein = proteins_ref[indices[b]]
        motif_pos = motif_positions_per_protein[b]

        if isempty(motif_pos)
            continue
        end

        # Create Set for O(1) lookup
        motif_set = Set(motif_pos)

        for i in 1:L_batch
            orig_id = ids[i, b]
            if orig_id > 0 && orig_id in motif_set && orig_id <= length(protein.mask)
                # This batched position maps to a motif position in the original protein
                seq_motif_mask_cpu[i, b] = 1.0f0
                motif_loss_mask_cpu[i, b] = 1.0f0
                n_motif_total += 1

                # Copy atom37 data from precomputed protein
                x_motif_cpu[:, :, i, b] = protein.atom37_coords[:, :, orig_id]
                motif_mask_cpu[:, i, b] = Float32.(protein.atom37_mask[:, orig_id])
                seq_motif_cpu[i, b] = Float32(protein.aatype[orig_id])

                # Override xt at motif positions with clean X1 values (no noise)
                xt_ca_cpu[:, i, b] = x1_ca_target_cpu[:, i, b]
                xt_ll_cpu[:, i, b] = x1_ll_target_cpu[:, i, b]
            end
        end
    end

    # Combined mask: padmask * branchmask * (1 - motif_loss_mask)
    # This zeros out CA/LL/split/del loss at both padding and motif positions
    combined_mask_cpu = mask_cpu .* branchmask_cpu .* (1f0 .- motif_loss_mask_cpu)

    # Step 5: Build batch dict for feature extraction
    cpu_batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
        :t => Dict(:bb_ca => t_ca_cpu, :local_latents => t_ll_cpu),
        :mask => mask_cpu,
        # Motif features for BulkAllAtomXmotifFeature (549D seq) and MotifPairDistFeature (301D pair)
        :x_motif => x_motif_cpu,           # [3, 37, L, B]
        :motif_mask => motif_mask_cpu,     # [37, L, B]
        :seq_motif => seq_motif_cpu,       # [L, B]
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
        t_ll = t_ll_cpu,
        n_motif = n_motif_total
    )

    return bd
end

# ============================================================================
# DataLoader Setup
# ============================================================================
println("\n=== Setting up Data Loading ===")

batch_indices = Vector{Vector{Int}}(undef, n_batches)
for i in 1:n_batches
    batch_indices[i] = rand(1:length(all_proteins), batch_size)
end
println("Created $n_batches batch index sets")

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

dataset = BatchDataset(batch_indices, all_proteins, P, base_model_cpu)
dataloader = Flux.DataLoader(dataset; batchsize=-1, parallel=false)
println("DataLoader created")

# ============================================================================
# Training Loop
# ============================================================================
warmdown_start = n_batches - warmdown_batches

println("\n=== Training ===")

losses = Float32[]
ca_losses = Float32[]
ll_losses = Float32[]
split_losses = Float32[]
del_losses = Float32[]
batch_times = Float64[]

println("Batch size: $batch_size, N batches: $n_batches")
println("Starting training loop...")

t_train_start = time()

for (batch_idx, bd_cpu) in enumerate(dataloader)
    t_batch = time()

    bd = dev(bd_cpu)

    ca_loss_ref = Ref(0f0)
    ll_loss_ref = Ref(0f0)
    split_loss_ref = Ref(0f0)
    del_loss_ref = Ref(0f0)
    t_min_ref = Ref(0f0)
    t_max_ref = Ref(0f0)

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

    # Forward + loss + gradients
    loss, grads = Flux.withgradient(model) do m
        out = forward_branching_from_raw_features_gpu(m, raw_features_for_training)

        # NaN diagnostic (first 5 batches)
        Zygote.ignore() do
            if batch_idx <= 5
                for (k, v_dict) in out
                    if v_dict isa Dict
                        for (k2, val) in v_dict
                            if val isa CuArray
                                n_nan = count(isnan, Array(val))
                                n_nan > 0 && @printf("  OUTPUT NaN: out[%s][%s] has %d NaN\n", string(k), string(k2), n_nan)
                            end
                        end
                    elseif v_dict isa CuArray
                        n_nan = count(isnan, Array(v_dict))
                        n_nan > 0 && @printf("  OUTPUT NaN: out[%s] has %d NaN\n", string(k), n_nan)
                    end
                end
            end
        end

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

    if batch_idx % 10 == 0
        if batch_idx == warmdown_start
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
    n_motif_log = haskey(bd, :n_motif) ? bd.n_motif : 0
    open(log_file, "a") do io
        @printf(io, "%d,%.2e,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.0f,%d,%d\n",
                batch_idx, sched.lr,
                losses[end], ca_losses[end], ll_losses[end], split_losses[end], del_losses[end],
                t_min_ref[], t_max_ref[], batch_time, L_log, n_motif_log)
    end

    if batch_idx % 100 == 0 || batch_idx == 1
        avg_loss = mean(losses[max(1, end-99):end])
        avg_ca = mean(ca_losses[max(1, end-99):end])
        avg_ll = mean(ll_losses[max(1, end-99):end])
        avg_split = mean(split_losses[max(1, end-99):end])
        avg_del = mean(del_losses[max(1, end-99):end])
        avg_time = mean(batch_times[max(1, end-99):end])

        warmup_status = batch_idx >= warmdown_start ? " (warmdown)" : ""

        @printf("Batch %6d: loss=%.3f (ca=%.3f, ll=%.3f, split=%.3f, del=%.3f), lr=%.2e%s, time=%.0fms\n",
                batch_idx, avg_loss, avg_ca, avg_ll, avg_split, avg_del, sched.lr, warmup_status, avg_time)

        if CUDA.functional()
            GC.gc()
            CUDA.reclaim()
        end
    end

    if batch_idx == 1 || batch_idx % sample_every == 0
        checkpoint_path = joinpath(output_dir, "checkpoints", @sprintf("checkpoint_batch%06d.jld2", batch_idx))
        model_cpu = cpu(model)
        save_branching_weights(model_cpu, checkpoint_path; include_base=true)
        println("  Saved checkpoint: $checkpoint_path")
    end

    batch_idx >= n_batches && break
end

t_train = time() - t_train_start

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 70)
println("Training Complete")
println("=" ^ 70)

warmup_skip = 10
post_warmup_times = batch_times[min(warmup_skip+1, end):end]

println("\nSummary:")
@printf("  Total time: %.1f hours\n", t_train / 3600)
@printf("  Avg batch time (post-warmup): %.1f ms\n", mean(post_warmup_times))
@printf("  Throughput: %.1f samples/sec\n", batch_size / (mean(post_warmup_times) / 1000))

first_1k = losses[1:min(1000, length(losses))]
last_1k = losses[max(1, end-999):end]
@printf("  First 1k loss: %.4f\n", mean(first_1k))
@printf("  Last 1k loss: %.4f\n", mean(last_1k))
@printf("  Change: %.2f%%\n", 100 * (mean(last_1k) - mean(first_1k)) / mean(first_1k))

# Save final
println("\n=== Saving Final Weights ===")
model_cpu = cpu(model)
final_path = joinpath(output_dir, "branching_motif_final.jld2")
save_branching_weights(model_cpu, final_path; include_base=true)
println("Saved final model to: $final_path")

println("\nLog file: $log_file")
println("Output directory: $output_dir")
println("\nDone!")
