#!/usr/bin/env julia
# Parity test: verify that update_sc_raw_features! produces identical results
# to full extract_raw_features recomputation.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: ScoreNetworkRawFeatures, extract_raw_features, cpu
using LaProteina: compute_sc_feature_offsets, update_sc_raw_features!
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow, branching_bridge
using ForwardBackward: ContinuousState, tensor
using Flowfusion: RDNFlow, MaskedState, schedule_transform
using Distributions: Uniform, Beta, Poisson
using Flux
using CUDA
using Statistics
using Random
using Printf

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)

latent_dim = 8
X0_mean_length = 100
deletion_pad = 1.1

println("=== SC Feature Parity Test ===\n")

# --- Load data ---
shard_dir = get(ENV, "SHARD_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards"))
weights_dir = get(ENV, "WEIGHTS_DIR", joinpath(@__DIR__, "..", "weights"))

println("Loading one shard...")
shard_files = filter(f -> startswith(f, "train_shard_") && endswith(f, ".jld2"), readdir(shard_dir))
sort!(shard_files)
proteins = load_precomputed_shard(joinpath(shard_dir, shard_files[1]))
println("  Loaded $(length(proteins)) proteins")

# --- Create model ---
println("Creating model...")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)
load_branching_weights!(model, joinpath(weights_dir, "branching_indel_stage1.jld2");
    base_weights_path=joinpath(weights_dir, "score_network.npz"))
base_model_cpu = deepcopy(model.base)
model_gpu = gpu(model)

# --- Compute SC offsets (once) ---
println("Computing SC feature offsets...")
sc_offsets = compute_sc_feature_offsets(base_model_cpu)
println("  Seq offsets: $(sc_offsets.seq)")
println("  Pair offsets: $(sc_offsets.pair)")

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

# --- Pad function ---
const TILE_SIZE = 64

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

function prepare_batch(indices)
    X1s = [protein_to_X1_simple(proteins[i]) for i in indices]
    batch = branching_bridge(P, X0_sampler_train, X1s, Uniform(0f0, 1f0);
        coalescence_factor=1.0, use_branching_time_prob=0.5,
        length_mins=Poisson(X0_mean_length), deletion_pad=deletion_pad)
    L_batch, B = size(batch.Xt.groupings)
    t_vec_cpu = Float32.(batch.t)
    t_ca_cpu = Float32.(schedule_transform.(Ref(P.P[1]), t_vec_cpu))
    t_ll_cpu = Float32.(schedule_transform.(Ref(P.P[2]), t_vec_cpu))
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
    raw_features = extract_raw_features(base_model_cpu, cpu_batch)
    bd = (raw_features=raw_features, cpu_batch=cpu_batch,
        xt_ca=xt_ca_cpu, xt_ll=xt_ll_cpu, mask=mask_cpu, combined_mask=combined_mask_cpu,
        x1_ca_target=x1_ca_target_cpu, x1_ll_target=x1_ll_target_cpu,
        split_target=Float32.(batch.splits_target), del_target=Float32.(batch.del),
        t_vec=t_vec_cpu, t_ca=t_ca_cpu, t_ll=t_ll_cpu)
    return pad_batch_to(bd, TILE_SIZE)
end

# --- Run parity tests ---
println("\n=== Running Parity Tests ===\n")

BS = 8
n_tests = 10
all_pass = true

for test_idx in 1:n_tests
    Random.seed!(test_idx * 7)  # different seed each test
    indices = rand(1:length(proteins), BS)
    bd_cpu = prepare_batch(indices)

    # Move to GPU
    bd = gpu(bd_cpu)

    # --- Method 1: Full recomputation (reference) ---
    raw_features_gpu = gpu(bd.raw_features)

    # SC forward pass
    out_sc = forward_branching_from_raw_features_gpu(model_gpu, raw_features_gpu)
    v_ca_sc = out_sc[:bb_ca][:v]
    v_ll_sc = out_sc[:local_latents][:v]
    t_ca_exp_sc = reshape(bd.t_ca, 1, 1, :)
    t_ll_exp_sc = reshape(bd.t_ll, 1, 1, :)
    x1_ca_sc = bd.xt_ca .+ (1f0 .- t_ca_exp_sc) .* v_ca_sc
    x1_ll_sc = bd.xt_ll .+ (1f0 .- t_ll_exp_sc) .* v_ll_sc

    # Full recomputation on CPU (the old way)
    x_sc_cpu = Dict(:bb_ca => cpu(x1_ca_sc), :local_latents => cpu(x1_ll_sc))
    batch_with_sc = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => bd_cpu.xt_ca, :local_latents => bd_cpu.xt_ll),
        :t => Dict(:bb_ca => bd_cpu.t_ca, :local_latents => bd_cpu.t_ll),
        :mask => bd_cpu.mask, :x_sc => x_sc_cpu)
    ref_features = gpu(extract_raw_features(base_model_cpu, batch_with_sc))

    # --- Method 2: In-place GPU update (the new way) ---
    # Start from the same pre-SC features, make a copy so we don't mutate the reference
    inplace_features = ScoreNetworkRawFeatures(
        copy(raw_features_gpu.seq_raw),
        copy(raw_features_gpu.cond_raw),
        copy(raw_features_gpu.pair_raw),
        copy(raw_features_gpu.pair_cond_raw),
        copy(raw_features_gpu.mask))

    update_sc_raw_features!(inplace_features, sc_offsets, x1_ca_sc, x1_ll_sc)

    # --- Compare SC channels only (the ones we're modifying) ---
    # Non-SC channels can legitimately differ in padded positions because the
    # pre-SC features were computed at original L then zero-padded, while the
    # reference recomputes everything at padded L. The mask zeros these out
    # in the model, so the difference is harmless.

    seq_ref = Array(ref_features.seq_raw)
    seq_new = Array(inplace_features.seq_raw)
    pair_ref = Array(ref_features.pair_raw)
    pair_new = Array(inplace_features.pair_raw)

    # SC seq channels: compare directly (both operate at padded L)
    sc_seq_diffs = Float64[]
    for (start, stop, ftype) in sc_offsets.seq
        d = maximum(abs.(seq_ref[start:stop, :, :] .- seq_new[start:stop, :, :]))
        push!(sc_seq_diffs, d)
    end
    sc_seq_max = maximum(sc_seq_diffs)

    # SC pair channels: compare directly
    sc_pair_diffs = Float64[]
    for (start, stop, _, _, _) in sc_offsets.pair
        d = maximum(abs.(pair_ref[start:stop, :, :, :] .- pair_new[start:stop, :, :, :]))
        push!(sc_pair_diffs, d)
    end
    sc_pair_max = maximum(sc_pair_diffs)

    # Non-SC channels: compare only within real (unmasked) positions
    mask_cpu = Array(bd.mask)  # [L, B]
    real_mask_seq = reshape(mask_cpu, 1, size(mask_cpu)...)  # [1, L, B]
    real_mask_pair = reshape(mask_cpu, size(mask_cpu, 1), 1, size(mask_cpu, 2)) .*
                     reshape(mask_cpu, 1, size(mask_cpu)...)  # [L, L, B]
    real_mask_pair = reshape(real_mask_pair, 1, size(real_mask_pair)...)  # [1, L, L, B]

    # Build masks for non-SC seq channels
    all_seq_dims = 1:size(seq_ref, 1)
    sc_seq_dims = Set{Int}()
    for (start, stop, _) in sc_offsets.seq
        for d in start:stop; push!(sc_seq_dims, d); end
    end
    nonsc_seq_dims = [d for d in all_seq_dims if d ∉ sc_seq_dims]

    nonsc_seq_diff = 0.0
    if !isempty(nonsc_seq_dims)
        ref_nonsc = seq_ref[nonsc_seq_dims, :, :]
        new_nonsc = seq_new[nonsc_seq_dims, :, :]
        diff_nonsc = abs.(ref_nonsc .- new_nonsc) .* real_mask_seq
        nonsc_seq_diff = maximum(diff_nonsc)
    end

    # Non-SC pair channels
    all_pair_dims = 1:size(pair_ref, 1)
    sc_pair_dims = Set{Int}()
    for (start, stop, _, _, _) in sc_offsets.pair
        for d in start:stop; push!(sc_pair_dims, d); end
    end
    nonsc_pair_dims = [d for d in all_pair_dims if d ∉ sc_pair_dims]

    nonsc_pair_diff = 0.0
    if !isempty(nonsc_pair_dims)
        ref_nonsc_p = pair_ref[nonsc_pair_dims, :, :, :]
        new_nonsc_p = pair_new[nonsc_pair_dims, :, :, :]
        diff_nonsc_p = abs.(ref_nonsc_p .- new_nonsc_p) .* real_mask_pair
        nonsc_pair_diff = maximum(diff_nonsc_p)
    end

    L = size(bd.mask, 1)
    pass = sc_seq_max < 1e-5 && sc_pair_max < 1e-5 && nonsc_seq_diff < 1e-5 && nonsc_pair_diff < 1e-5
    status = pass ? "PASS" : "FAIL"
    if !pass
        global all_pass = false
    end

    @printf("Test %2d (L=%3d): %s | sc_seq=%.2e, sc_pair=%.2e, nonsc_seq(masked)=%.2e, nonsc_pair(masked)=%.2e\n",
            test_idx, L, status, sc_seq_max, sc_pair_max, nonsc_seq_diff, nonsc_pair_diff)

    if !pass
        # Diagnose which SC channels differ
        for (start, stop, ftype) in sc_offsets.seq
            d = maximum(abs.(seq_ref[start:stop, :, :] .- seq_new[start:stop, :, :]))
            if d > 1e-5
                println("    SC seq $ftype (channels $start:$stop): max_diff=$d")
            end
        end
        for (start, stop, _, _, _) in sc_offsets.pair
            d = maximum(abs.(pair_ref[start:stop, :, :, :] .- pair_new[start:stop, :, :, :]))
            if d > 1e-5
                println("    SC pair (channels $start:$stop): max_diff=$d")
            end
        end
        if nonsc_seq_diff > 1e-5
            println("    Non-SC seq (masked): max_diff=$nonsc_seq_diff")
        end
        if nonsc_pair_diff > 1e-5
            println("    Non-SC pair (masked): max_diff=$nonsc_pair_diff")
        end
    end
end

println("\n" * "=" ^ 60)
if all_pass
    println("ALL $n_tests TESTS PASSED — in-place update matches full recomputation")
else
    println("SOME TESTS FAILED — in-place update does NOT match full recomputation")
end
println("=" ^ 60)
