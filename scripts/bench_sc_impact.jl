#!/usr/bin/env julia
# Quick A/B benchmark: self-conditioning ON vs OFF
# Runs 100 batches each at BS=8, compares median batch times.

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

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Flux.MLDataDevices.Internal.unsafe_free!(x) = (Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x); return nothing)

Random.seed!(42)

# --- Log file ---
log_path = joinpath(@__DIR__, "..", "bench_sc_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
const LOG_IO = open(log_path, "w")
function logprintln(args...)
    println(args...)
    println(LOG_IO, args...)
    flush(LOG_IO)
end

logprintln("=" ^ 60)
logprintln("Self-Conditioning A/B Benchmark")
logprintln("=" ^ 60)
logprintln("Log file: $log_path")

# --- Config ---
shard_dir = get(ENV, "SHARD_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards"))
weights_dir = get(ENV, "WEIGHTS_DIR", joinpath(@__DIR__, "..", "weights"))
latent_dim = 8
X0_mean_length = 100
deletion_pad = 1.1
ca_loss_scale = 2.0f0
ll_loss_scale = 0.1f0
const TILE_SIZE = 64
const BS = 8
const N_BATCHES = 100  # per phase
const N_WARMUP = 5     # skip first N for timing

# --- GPU ---
@assert CUDA.functional()
logprintln("Device: $(CUDA.device())")
logprintln("Memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB")
LaProteina.enable_tf32!()
logprintln("cuTile: $(LaProteina._HAS_CUTILE)")
dev = gpu

# --- Data ---
logprintln("\nLoading data...")
shard_files = filter(f -> startswith(f, "train_shard_") && endswith(f, ".jld2"), readdir(shard_dir))
sort!(shard_files)
all_proteins = []
for sf in shard_files
    append!(all_proteins, load_precomputed_shard(joinpath(shard_dir, sf)))
end
logprintln("Loaded $(length(all_proteins)) proteins")

# --- Model ---
logprintln("Loading model...")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)
load_branching_weights!(model, joinpath(weights_dir, "branching_indel_stage1.jld2");
    base_weights_path=joinpath(weights_dir, "score_network.npz"))
base_model_cpu = deepcopy(model.base)
model = dev(model)

sched = burnin_learning_schedule(0.00001f0, 0.000250f0, 1.05f0, 0.99995f0)
opt_state = Flux.setup(Muon(eta=sched.lr), model)

# --- Flow process ---
P_ca = RDNFlow(3; zero_com=false, schedule=:log, schedule_param=2.0f0,
    sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)
P_ll = RDNFlow(latent_dim; zero_com=false, schedule=:power, schedule_param=2.0f0,
    sde_gt_mode=:tan, sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)
P = CoalescentFlow((P_ca, P_ll), Beta(1.0, 2.0))

function X0_sampler_train(root)
    ca = ContinuousState(randn(Float32, 3, 1, 1))
    ll = ContinuousState(randn(Float32, latent_dim, 1, 1))
    return (ca, ll)
end

# --- Batch prep (same as sweep/training) ---
function protein_to_X1_simple(protein)
    ca_coords = protein.ca_coords
    z_mean = protein.z_mean
    z_log_scale = protein.z_log_scale
    mask = protein.mask
    L = length(mask)
    z_latent = z_mean .+ randn(Float32, size(z_mean)) .* exp.(z_log_scale)
    valid_mask = Vector{Bool}(mask .> 0.5)
    ca_state = MaskedState(ContinuousState(reshape(ca_coords, 3, 1, L)), valid_mask, valid_mask)
    latent_state = MaskedState(ContinuousState(reshape(z_latent, size(z_latent, 1), 1, L)), valid_mask, valid_mask)
    return BranchingState((ca_state, latent_state), ones(Int, L); flowmask=valid_mask, branchmask=valid_mask)
end

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
        pad4_pair(rf.pair_raw), pad4_pair(rf.pair_cond_raw), pad2(rf.mask))
    new_cpu_batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => pad3(bd.cpu_batch[:x_t][:bb_ca]),
                      :local_latents => pad3(bd.cpu_batch[:x_t][:local_latents])),
        :t => bd.cpu_batch[:t], :mask => pad2(bd.cpu_batch[:mask]))
    return (raw_features=new_rf, cpu_batch=new_cpu_batch,
        xt_ca=pad3(bd.xt_ca), xt_ll=pad3(bd.xt_ll),
        mask=pad2(bd.mask), combined_mask=pad2(bd.combined_mask),
        x1_ca_target=pad3(bd.x1_ca_target), x1_ll_target=pad3(bd.x1_ll_target),
        split_target=pad2(bd.split_target), del_target=pad2(bd.del_target),
        t_vec=bd.t_vec, t_ca=bd.t_ca, t_ll=bd.t_ll)
end

function prepare_training_batch(indices, proteins_ref, P_ref, X0_sampler, base_model; pad_to::Int=0)
    X1s = [protein_to_X1_simple(proteins_ref[i]) for i in indices]
    batch = branching_bridge(P_ref, X0_sampler, X1s, Uniform(0f0, 1f0);
        coalescence_factor=1.0, use_branching_time_prob=0.5,
        length_mins=Poisson(X0_mean_length), deletion_pad=deletion_pad)
    L_batch, B = size(batch.Xt.groupings)
    t_vec_cpu = Float32.(batch.t)
    t_ca_cpu = Float32.(schedule_transform.(Ref(P_ref.P[1]), t_vec_cpu))
    t_ll_cpu = Float32.(schedule_transform.(Ref(P_ref.P[2]), t_vec_cpu))
    ca_tensor = tensor(batch.Xt.state[1])
    ll_tensor = tensor(batch.Xt.state[2])
    xt_ca_cpu = ndims(ca_tensor) == 4 ? dropdims(ca_tensor, dims=2) : ca_tensor
    xt_ll_cpu = ndims(ll_tensor) == 4 ? dropdims(ll_tensor, dims=2) : ll_tensor
    mask_cpu = Float32.(batch.Xt.padmask)
    combined_mask_cpu = mask_cpu .* Float32.(batch.Xt.branchmask)
    x1_ca_target_cpu = tensor(batch.X1anchor[1])
    x1_ll_target_cpu = tensor(batch.X1anchor[2])
    x1_ca_target_cpu = ndims(x1_ca_target_cpu) == 4 ? dropdims(x1_ca_target_cpu, dims=2) : x1_ca_target_cpu
    x1_ll_target_cpu = ndims(x1_ll_target_cpu) == 4 ? dropdims(x1_ll_target_cpu, dims=2) : x1_ll_target_cpu
    cpu_batch = Dict{Symbol,Any}(
        :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
        :t => Dict(:bb_ca => t_ca_cpu, :local_latents => t_ll_cpu), :mask => mask_cpu)
    raw_features = extract_raw_features(base_model, cpu_batch)
    bd = (raw_features=raw_features, cpu_batch=cpu_batch,
        xt_ca=xt_ca_cpu, xt_ll=xt_ll_cpu, mask=mask_cpu, combined_mask=combined_mask_cpu,
        x1_ca_target=x1_ca_target_cpu, x1_ll_target=x1_ll_target_cpu,
        split_target=Float32.(batch.splits_target), del_target=Float32.(batch.del),
        t_vec=t_vec_cpu, t_ca=t_ca_cpu, t_ll=t_ll_cpu)
    return pad_batch_to(bd, pad_to)
end

struct BatchDataset{P, S, M}
    batch_indices::Vector{Vector{Int}}; proteins::P; process::S; base_model::M; pad_to::Int
end
Base.length(x::BatchDataset) = length(x.batch_indices)
Base.getindex(x::BatchDataset, i::Int) = prepare_training_batch(
    x.batch_indices[i], x.proteins, x.process, X0_sampler_train, x.base_model; pad_to=x.pad_to)

# --- Run one phase ---
function run_phase(phase_name::String, enable_sc::Bool, batch_indices_all, global_idx_start)
    logprintln("\n--- Phase: $phase_name (SC=$(enable_sc)) ---")
    dataset = BatchDataset(batch_indices_all, all_proteins, P, base_model_cpu, TILE_SIZE)
    dataloader = Flux.DataLoader(dataset; batchsize=-1, parallel=true)

    batch_times = Float64[]
    global_idx = global_idx_start

    for (batch_idx, bd_cpu) in enumerate(dataloader)
        global_idx += 1
        t_batch = time()

        bd = dev(bd_cpu)
        raw_features_for_training = dev(bd.raw_features)

        if enable_sc
            # Self-conditioning: GPU fwd -> CPU round-trip -> GPU
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
                :mask => bd_cpu.mask, :x_sc => x_sc)
            raw_features_for_training = dev(extract_raw_features(base_model_cpu, batch_with_sc))
        end

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
            ca_clamped = isfinite(ca_loss * ca_loss_scale) ? softclamp(ca_loss * ca_loss_scale) : 0.0f0
            ll_clamped = isfinite(ll_loss * ll_loss_scale) ? softclamp(ll_loss * ll_loss_scale) : 0.0f0
            split_clamped = isfinite(split_l) ? softclamp(split_l) : 0.0f0
            del_clamped = isfinite(del_l) ? softclamp(del_l) : 0.0f0
            min(ca_clamped + ll_clamped + split_clamped + del_clamped, 20.0f0)
        end

        Flux.update!(opt_state, model, grads[1])
        CUDA.synchronize()

        if global_idx % 10 == 0
            Flux.adjust!(opt_state, next_rate(sched))
        end

        batch_time = (time() - t_batch) * 1000
        push!(batch_times, batch_time)

        if batch_idx <= 3 || batch_idx % 20 == 0
            logprintln("  Batch $batch_idx: loss=$(round(Float32(cpu(loss)), digits=3)), time=$(round(Int, batch_time)) ms")
        end
    end

    # Stats (skip warmup)
    timed = batch_times[min(N_WARMUP+1, end):end]
    med = median(timed)
    avg = mean(timed)
    logprintln("  => $phase_name: median=$(round(Int, med)) ms, mean=$(round(Int, avg)) ms, per-sample=$(round(med/BS, digits=1)) ms ($(length(timed)) batches after warmup)")
    return (name=phase_name, median_ms=med, mean_ms=avg, per_sample=med/BS, times=timed, global_idx=global_idx)
end

# --- Pre-sample batch indices (same seed for both phases) ---
logprintln("\nPre-sampling batch indices...")
Random.seed!(123)
indices_a = [rand(1:length(all_proteins), BS) for _ in 1:N_BATCHES]
indices_b = [rand(1:length(all_proteins), BS) for _ in 1:N_BATCHES]

# --- Phase A: SC ON ---
result_a = run_phase("SC_ON", true, indices_a, 0)

# --- Phase B: SC OFF ---
result_b = run_phase("SC_OFF", false, indices_b, result_a.global_idx)

# --- Summary ---
logprintln("\n" * "=" ^ 60)
logprintln("RESULTS (BS=$BS, $N_BATCHES batches each, $N_WARMUP warmup skipped)")
logprintln("=" ^ 60)
logprintln("  SC ON:  median=$(round(Int, result_a.median_ms)) ms, per-sample=$(round(result_a.per_sample, digits=1)) ms")
logprintln("  SC OFF: median=$(round(Int, result_b.median_ms)) ms, per-sample=$(round(result_b.per_sample, digits=1)) ms")

speedup = result_a.median_ms / result_b.median_ms
sc_overhead_pct = (result_a.median_ms - result_b.median_ms) / result_a.median_ms * 100
logprintln("  Speedup without SC: $(round(speedup, digits=2))x")
logprintln("  SC overhead: $(round(sc_overhead_pct, digits=1))% of total batch time")
logprintln("\nDone!")
close(LOG_IO)
