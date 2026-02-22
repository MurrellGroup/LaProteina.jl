#!/usr/bin/env julia
# Integration Test Script for LaProteina.jl
#
# Tests 4 capabilities sequentially (one compute process at a time):
#   1. Standard flow training (50 batches)
#   2. Standard sampling (2 samples)
#   3. Branching training (50 batches)
#   4. Branching sampling (2 samples)
#
# Each test section is a condensed snippet from existing scripts:
#   - Tests 1/2: standard ScoreNetwork (base extracted from branching checkpoint)
#   - Tests 3/4: from train_branching_full_OU.jl / sample_branching_full_OU.jl
#
# Usage:
#   julia --project=. -t 4 scripts/integration_test.jl

# --- Same imports as train_branching_full_OU.jl ---
using LaProteina
using OnionTile  # Activates cuTile CuArray overrides for Onion dispatch hooks
using LaProteina: ScoreNetworkRawFeatures, extract_raw_features, cpu
using LaProteina: compute_sc_feature_offsets, update_sc_raw_features!
using LaProteina: DecoderTransformer, load_decoder_weights_st!, samples_to_pdb
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow, branching_bridge
using ForwardBackward: ContinuousState, DiscreteState, tensor, OUBridgeExpVar
using Flowfusion: MaskedState, floss, scalefloss
import Flowfusion
using Distributions: Uniform, Beta, Poisson
using Flux
using Flux: Zygote
using CUDA
using Optimisers
using Statistics
using Random
using Printf
using JLD2

Random.seed!(42)

# ============================================================================
# Configuration
# ============================================================================
const CHECKPOINT_PATH = "/home/claudey/safe_models/branching_OU_100k_20260216.jld2"
const DECODER_FILE = "AE1_ucond_512.safetensors"
const SHARD_PATH = expanduser("~/shared_data/afdb_laproteina/precomputed_shards/train_shard_01.jld2")
const LATENT_DIM = 8
const BATCH_SIZE = 2
const N_TRAIN_BATCHES = 50
const N_SAMPLES = 2
const N_SAMPLE_STEPS = 50

println("=" ^ 70)
println("LaProteina Integration Test")
println("=" ^ 70)

# --- GPU check (from train_branching_full_OU.jl lines 117-132) ---
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: $(CUDA.device())")
    println("Memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB available")
    LaProteina.enable_tf32!()
    dev = gpu
else
    println("No CUDA - using CPU")
    dev = identity
end

# Output directory
output_dir = mktempdir(prefix="laproteina_inttest_")
println("Output dir: $output_dir")
println()

# Results tracking
results = Dict{String, Symbol}()

"""Check that all elements of an array are finite (no NaN/Inf)."""
function check_finite(arr, name::String)
    a = arr isa CUDA.CuArray ? Array(arr) : arr
    n_nan = count(isnan, a)
    n_inf = count(isinf, a)
    if n_nan > 0 || n_inf > 0
        println("    WARNING: $name has $n_nan NaN, $n_inf Inf (shape=$(size(a)))")
        return false
    end
    return true
end

# ============================================================================
# Load shared resources
# ============================================================================
println("=== Loading shared resources ===")

println("  Loading checkpoint: $CHECKPOINT_PATH")
checkpoint_weights = load(CHECKPOINT_PATH)
println("  Checkpoint keys: $(keys(checkpoint_weights))")

# --- Load shard (from train_branching_full_OU.jl lines 137-153) ---
println("  Loading shard: $SHARD_PATH")
all_proteins = load_precomputed_shard(SHARD_PATH)
println("  Loaded $(length(all_proteins)) proteins")

# --- Load decoder (from sample_branching_full_OU.jl lines 72-78) ---
println("  Loading decoder: $DECODER_FILE (from HuggingFace)")
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=LATENT_DIM, qk_ln=true, update_pair_repr=false
)
load_decoder_weights_st!(decoder, DECODER_FILE)
println("  Decoder loaded (CPU)")
println()


# ############################################################################
# TEST 1: Standard Flow Training (50 batches)
# ############################################################################
#
# Adapted from train_branching_full_OU.jl: uses the same extract_raw_features +
# forward_from_raw_features_gpu + SC pattern, but on the base ScoreNetwork only
# (no split/del heads).
#
println("=" ^ 70)
println("TEST 1: Standard Flow Training ($N_TRAIN_BATCHES batches)")
println("=" ^ 70)

score_net = nothing

try
    # --- Create model (from train_branching_full_OU.jl lines 170-175) ---
    base_model = ScoreNetwork(
        n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
        dim_cond=256, latent_dim=LATENT_DIM, output_param=:v,
        qk_ln=true, update_pair_repr=false, cropped_flag=true
    )

    # Load base weights from branching checkpoint
    @assert haskey(checkpoint_weights, "base") "Checkpoint missing 'base' key"
    Flux.loadmodel!(base_model, checkpoint_weights["base"])
    println("  Base weights loaded")

    # CPU copy for feature extraction (from train_branching_full_OU.jl line 192)
    base_model_cpu = deepcopy(base_model)

    base_model = dev(base_model)

    # SC offsets (from train_branching_full_OU.jl lines 198-199)
    sc_offsets = compute_sc_feature_offsets(base_model_cpu)
    println("  SC offsets: seq=$(length(sc_offsets.seq)), pair=$(length(sc_offsets.pair))")

    opt_state = Flux.setup(Optimisers.Adam(1f-4), base_model)

    # --- Training loop (adapted from train_branching_full_OU.jl lines 437-628) ---
    # Uses same batch prep + SC pattern, but with base ScoreNetwork and simple MSE loss
    losses = Float32[]
    all_finite = true

    for batch_idx in 1:N_TRAIN_BATCHES
        # Sample proteins and build padded batch
        indices = rand(1:length(all_proteins), BATCH_SIZE)
        selected = [all_proteins[i] for i in indices]
        max_L = maximum(length(p.mask) for p in selected)

        x1_ca = zeros(Float32, 3, max_L, BATCH_SIZE)
        x1_ll = zeros(Float32, LATENT_DIM, max_L, BATCH_SIZE)
        mask_cpu = zeros(Float32, max_L, BATCH_SIZE)
        for (b, p) in enumerate(selected)
            L = length(p.mask)
            x1_ca[:, 1:L, b] = p.ca_coords
            x1_ll[:, 1:L, b] = p.z_mean .+ randn(Float32, size(p.z_mean)) .* exp.(p.z_log_scale)
            mask_cpu[1:L, b] = p.mask
        end

        # Sample time and interpolate: x_t = (1-t)*x0 + t*x1
        t_vec_cpu = rand(Float32, BATCH_SIZE)
        t_exp = reshape(t_vec_cpu, 1, 1, :)
        x0_ca = randn(Float32, 3, max_L, BATCH_SIZE)
        x0_ll = randn(Float32, LATENT_DIM, max_L, BATCH_SIZE)
        xt_ca_cpu = (1f0 .- t_exp) .* x0_ca .+ t_exp .* x1_ca
        xt_ll_cpu = (1f0 .- t_exp) .* x0_ll .+ t_exp .* x1_ll

        # --- Extract features on CPU (from train_branching_full_OU.jl line 364) ---
        cpu_batch = Dict{Symbol, Any}(
            :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
            :t => Dict(:bb_ca => t_vec_cpu, :local_latents => t_vec_cpu),
            :mask => mask_cpu
        )
        raw_features = extract_raw_features(base_model_cpu, cpu_batch)

        # --- Move to GPU (from train_branching_full_OU.jl line 441) ---
        bd = dev((
            raw_features = raw_features,
            xt_ca = xt_ca_cpu,
            xt_ll = xt_ll_cpu,
            mask = mask_cpu,
            x1_ca_target = x1_ca,
            x1_ll_target = x1_ll,
            t_vec = t_vec_cpu,
        ))

        # --- Self-conditioning (from train_branching_full_OU.jl lines 457-471) ---
        raw_features_for_training = bd.raw_features
        n_sc_passes = rand(Poisson(1))
        if n_sc_passes > 0
            for _ in 1:n_sc_passes
                out_sc = forward_from_raw_features_gpu(base_model, raw_features_for_training)
                v_ca_sc = out_sc[:bb_ca][:v]
                v_ll_sc = out_sc[:local_latents][:v]
                t_exp_sc = reshape(bd.t_vec, 1, 1, :)
                x1_ca_sc = bd.xt_ca .+ (1f0 .- t_exp_sc) .* v_ca_sc
                x1_ll_sc = bd.xt_ll .+ (1f0 .- t_exp_sc) .* v_ll_sc
                update_sc_raw_features!(raw_features_for_training, sc_offsets, x1_ca_sc, x1_ll_sc)
            end
        end

        # --- Loss + gradient (from train_branching_full_OU.jl lines 474-545) ---
        loss, grads = Flux.withgradient(base_model) do m
            out = forward_from_raw_features_gpu(m, raw_features_for_training)

            v_ca = out[:bb_ca][:v]
            v_ll = out[:local_latents][:v]
            t_exp_d = reshape(bd.t_vec, 1, 1, :)
            x1_ca_pred = bd.xt_ca .+ (1f0 .- t_exp_d) .* v_ca
            x1_ll_pred = bd.xt_ll .+ (1f0 .- t_exp_d) .* v_ll

            # Simple MSE loss (no floss since we're using linear interpolation, not OUBridge)
            mask_exp = reshape(bd.mask, 1, size(bd.mask)...)
            n_valid = max(sum(bd.mask), 1f0)
            ca_loss = sum((x1_ca_pred .- bd.x1_ca_target).^2 .* mask_exp) / n_valid
            ll_loss = sum((x1_ll_pred .- bd.x1_ll_target).^2 .* mask_exp) / n_valid

            ca_loss + ll_loss
        end

        loss_val = Float32(cpu(loss))
        if !isfinite(loss_val)
            println("  NaN/Inf loss at batch $batch_idx")
            all_finite = false
            break
        end

        Flux.update!(opt_state, base_model, grads[1])
        push!(losses, loss_val)

        if batch_idx % 10 == 0 || batch_idx == 1
            @printf("  Batch %3d: loss = %.4f\n", batch_idx, loss_val)
        end
    end

    if !all_finite
        results["1_standard_training"] = :fail
        println("  FAIL: Non-finite loss detected")
    elseif length(losses) < N_TRAIN_BATCHES
        results["1_standard_training"] = :fail
        println("  FAIL: Only completed $(length(losses))/$N_TRAIN_BATCHES batches")
    else
        @printf("  Loss: first10=%.4f, last10=%.4f\n", mean(losses[1:10]), mean(losses[end-9:end]))
        results["1_standard_training"] = :pass
        println("  PASS")
    end

    global score_net = base_model

catch e
    results["1_standard_training"] = :fail
    println("  FAIL: $(typeof(e)): $e")
    Base.showerror(stdout, e, catch_backtrace())
    println()
end

println()


# ############################################################################
# TEST 2: Standard Sampling (2 samples)
# ############################################################################
#
# Uses generate_with_flowfusion() from src/flowfusion_sampling.jl + decoder
# from sample_branching_full_OU.jl decode pattern (lines 164-181).
#
println("=" ^ 70)
println("TEST 2: Standard Sampling ($N_SAMPLES samples, $N_SAMPLE_STEPS steps)")
println("=" ^ 70)

try
    if isnothing(score_net)
        error("Skipping: Test 1 did not produce a model")
    end

    sample_dir = joinpath(output_dir, "standard_samples")
    mkpath(sample_dir)

    all_ok = true
    for s in 1:N_SAMPLES
        L = 50
        println("  Generating sample $s (L=$L, $N_SAMPLE_STEPS steps)...")

        # --- generate_with_flowfusion from src/flowfusion_sampling.jl ---
        flow_out = generate_with_flowfusion(score_net, L, 1;
            nsteps=N_SAMPLE_STEPS, latent_dim=LATENT_DIM,
            self_cond=true, dev=dev)

        ca_out = flow_out[:bb_ca]          # [3, L, 1]
        ll_out = flow_out[:local_latents]  # [latent_dim, L, 1]
        mask_out = flow_out[:mask]         # [L, 1]

        # Check shapes and values
        @assert size(ca_out) == (3, L, 1) "CA shape: $(size(ca_out))"
        @assert size(ll_out) == (LATENT_DIM, L, 1) "LL shape: $(size(ll_out))"
        ok = check_finite(ca_out, "CA") && check_finite(ll_out, "LL")

        # --- Decode + save PDB (from sample_branching_full_OU.jl lines 164-181) ---
        dec_input = Dict(:z_latent => ll_out, :ca_coors => ca_out, :mask => mask_out)
        dec_out = decoder(dec_input)
        check_finite(dec_out[:coors], "decoded coors")

        samples = Dict(
            :ca_coords => ca_out, :latents => ll_out,
            :all_atom_coords => dec_out[:coors], :aatype => dec_out[:aatype_max],
            :atom_mask => dec_out[:atom_mask], :mask => mask_out
        )
        samples_to_pdb(samples, sample_dir; prefix="standard_$s", save_all_atom=true)

        aatype = dec_out[:aatype_max][:, 1]
        seq = join([index_to_aa(aa) for aa in aatype])
        println("    L=$L, seq=$(seq[1:min(30, length(seq))])...")

        if !ok; all_ok = false; end
    end

    results["2_standard_sampling"] = all_ok ? :pass : :fail
    println(all_ok ? "  PASS" : "  FAIL: Non-finite values in samples")

catch e
    results["2_standard_sampling"] = :fail
    println("  FAIL: $(typeof(e)): $e")
    Base.showerror(stdout, e, catch_backtrace())
    println()
end

# Free standard model memory
score_net = nothing
GC.gc()
CUDA.functional() && CUDA.reclaim()
println()


# ############################################################################
# TEST 3: Branching Training (50 batches)
# ############################################################################
#
# Direct snippet from train_branching_full_OU.jl with N reduced to 50 batches.
# Model creation (lines 170-202), batch prep (lines 287-383), training loop
# (lines 437-628).
#
println("=" ^ 70)
println("TEST 3: Branching Training ($N_TRAIN_BATCHES batches)")
println("=" ^ 70)

branching_model = nothing

try
    # --- Create model (from train_branching_full_OU.jl lines 170-199) ---
    base = ScoreNetwork(
        n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
        dim_cond=256, latent_dim=LATENT_DIM, output_param=:v,
        qk_ln=true, update_pair_repr=false, cropped_flag=true
    )
    model = BranchingScoreNetwork(base)

    # --- Load weights (from sample_branching_full_OU.jl lines 51-68) ---
    if haskey(checkpoint_weights, "base")
        Flux.loadmodel!(model.base, checkpoint_weights["base"])
        println("  Loaded fine-tuned base weights")
    end
    if haskey(checkpoint_weights, "indel_time_proj")
        Flux.loadmodel!(model.indel_time_proj, checkpoint_weights["indel_time_proj"])
    end
    if haskey(checkpoint_weights, "split_head")
        Flux.loadmodel!(model.split_head, checkpoint_weights["split_head"])
    end
    if haskey(checkpoint_weights, "del_head")
        Flux.loadmodel!(model.del_head, checkpoint_weights["del_head"])
    end

    # CPU copy for feature extraction (from train_branching_full_OU.jl line 192)
    base_model_cpu = deepcopy(model.base)
    model = dev(model)
    println("  Model loaded and moved to device")

    # SC offsets (from train_branching_full_OU.jl lines 198-199)
    sc_offsets = compute_sc_feature_offsets(base_model_cpu)
    println("  SC offsets: seq=$(sc_offsets.seq), pair=$(sc_offsets.pair)")

    opt_state = Flux.setup(Optimisers.Adam(1f-4), model)

    # --- Branching processes (from train_branching_full_OU.jl lines 223-226) ---
    P_ca = OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)
    P_ll = OUBridgeExpVar(100f0, 50f0, 0.000000001f0, dec = -0.1f0)
    branch_time_dist = Beta(1.0, 2.0)
    P = CoalescentFlow((P_ca, P_ll), branch_time_dist)

    X0_mean_length = 100
    deletion_pad = 1.1
    ca_calibration = 1.0f0
    ll_calibration = 3.0f0

    # --- X0 sampler (from train_branching_full_OU.jl lines 230-234) ---
    function X0_sampler_train(root)
        ca = ContinuousState(randn(Float32, 3, 1, 1))
        ll = ContinuousState(randn(Float32, LATENT_DIM, 1, 1))
        return (ca, ll)
    end

    # --- protein_to_X1_simple (from train_branching_full_OU.jl lines 287-313) ---
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

    # --- prepare_training_batch (from train_branching_full_OU.jl lines 315-383) ---
    function prepare_batch(indices)
        X1s = [protein_to_X1_simple(all_proteins[i]) for i in indices]

        t_dist = Uniform(0f0, 1f0)
        batch = branching_bridge(
            P, X0_sampler_train, X1s, t_dist;
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
        xt_ca_cpu = ndims(ca_tensor) == 4 ? dropdims(ca_tensor, dims=2) : ca_tensor
        xt_ll_cpu = ndims(ll_tensor) == 4 ? dropdims(ll_tensor, dims=2) : ll_tensor

        mask_cpu = Float32.(batch.Xt.padmask)
        branchmask_cpu = Float32.(batch.Xt.branchmask)
        combined_mask_cpu = mask_cpu .* branchmask_cpu

        x1_ca_target_cpu = tensor(batch.X1anchor[1])
        x1_ll_target_cpu = tensor(batch.X1anchor[2])
        x1_ca_target_cpu = ndims(x1_ca_target_cpu) == 4 ? dropdims(x1_ca_target_cpu, dims=2) : x1_ca_target_cpu
        x1_ll_target_cpu = ndims(x1_ll_target_cpu) == 4 ? dropdims(x1_ll_target_cpu, dims=2) : x1_ll_target_cpu

        split_target_cpu = Float32.(batch.splits_target)
        del_target_cpu = Float32.(batch.del)

        cpu_batch_dict = Dict{Symbol, Any}(
            :x_t => Dict(:bb_ca => xt_ca_cpu, :local_latents => xt_ll_cpu),
            :t => Dict(:bb_ca => t_ca_cpu, :local_latents => t_ll_cpu),
            :mask => mask_cpu
        )
        raw_features = extract_raw_features(base_model_cpu, cpu_batch_dict)

        return (
            raw_features = raw_features,
            xt_ca = xt_ca_cpu, xt_ll = xt_ll_cpu,
            mask = mask_cpu, combined_mask = combined_mask_cpu,
            x1_ca_target = x1_ca_target_cpu, x1_ll_target = x1_ll_target_cpu,
            split_target = split_target_cpu, del_target = del_target_cpu,
            t_vec = t_vec_cpu, t_ca = t_ca_cpu, t_ll = t_ll_cpu
        )
    end

    # --- Training loop (from train_branching_full_OU.jl lines 437-628) ---
    losses = Float32[]
    all_finite = true

    for batch_idx in 1:N_TRAIN_BATCHES
        bd_cpu = prepare_batch(rand(1:length(all_proteins), BATCH_SIZE))
        bd = dev(bd_cpu)

        # --- Self-conditioning (from train_branching_full_OU.jl lines 457-471) ---
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

        # --- Loss + gradients (from train_branching_full_OU.jl lines 474-545) ---
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

            ca_clamped = isfinite(ca_loss) ? LaProteina.softclamp(ca_loss) : 0.0f0
            ll_clamped = isfinite(ll_loss) ? LaProteina.softclamp(ll_loss) : 0.0f0
            split_clamped = isfinite(split_l) ? LaProteina.softclamp(split_l) : 0.0f0
            del_clamped = isfinite(del_l) ? LaProteina.softclamp(del_l) : 0.0f0

            total_loss = ca_clamped + ll_clamped + split_clamped + del_clamped
            min(total_loss, 20.0f0)
        end

        loss_val = Float32(cpu(loss))
        if !isfinite(loss_val)
            L_batch = size(bd.mask, 1)
            @printf("  FATAL: NaN/Inf loss at batch %d (L=%d)\n", batch_idx, L_batch)
            all_finite = false
            break
        end

        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss_val)

        if batch_idx % 10 == 0 || batch_idx == 1
            L_batch = size(bd.mask, 1)
            @printf("  Batch %3d: loss = %.4f (L=%d)\n", batch_idx, loss_val, L_batch)
        end

        if batch_idx % 25 == 0 && CUDA.functional()
            GC.gc(); CUDA.reclaim()
        end
    end

    if !all_finite
        results["3_branching_training"] = :fail
        println("  FAIL: Non-finite loss detected")
    elseif length(losses) < N_TRAIN_BATCHES
        results["3_branching_training"] = :fail
        println("  FAIL: Only completed $(length(losses))/$N_TRAIN_BATCHES batches")
    else
        @printf("  Loss: first10=%.4f, last10=%.4f\n", mean(losses[1:10]), mean(losses[end-9:end]))
        results["3_branching_training"] = :pass
        println("  PASS")
    end

    global branching_model = model

catch e
    results["3_branching_training"] = :fail
    println("  FAIL: $(typeof(e)): $e")
    Base.showerror(stdout, e, catch_backtrace())
    println()
end

println()


# ############################################################################
# TEST 4: Branching Sampling (2 samples)
# ############################################################################
#
# Direct snippet from sample_branching_full_OU.jl lines 96-191, with:
#   - 50 cosine steps instead of 500
#   - 2 samples instead of 10
#
println("=" ^ 70)
println("TEST 4: Branching Sampling ($N_SAMPLES samples, $N_SAMPLE_STEPS cosine steps)")
println("=" ^ 70)

try
    if isnothing(branching_model)
        error("Skipping: Test 3 did not produce a model")
    end

    sample_dir = joinpath(output_dir, "branching_samples")
    mkpath(sample_dir)

    # --- Processes (from sample_branching_full_OU.jl lines 98-103) ---
    P_ca_s = OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)
    P_ll_s = OUBridgeExpVar(100f0, 50f0, 0.000000001f0, dec = -0.1f0)
    P_idx_s = NullProcess()
    branch_time_dist_s = Beta(1.0, 2.0)
    P_s = CoalescentFlow((P_ca_s, P_ll_s, P_idx_s), branch_time_dist_s)

    # --- Cosine time schedule (from sample_branching_full_OU.jl lines 87-91) ---
    step_func(t) = Float32(1 - (cos(t * pi) + 1) / 2)
    steps = step_func.(0f0:Float32(1/N_SAMPLE_STEPS):1f0)
    nsteps = length(steps) - 1
    println("  Steps: $nsteps (cosine schedule)")
    println("  Processes: OUBridgeExpVar (CA: θ=100,v0=150; LL: θ=100,v0=50)")

    # --- Wrapper (from sample_branching_full_OU.jl lines 106-107) ---
    wrapper = BranchingScoreNetworkWrapper(branching_model, LATENT_DIM;
        self_cond=true, dev=dev, processes=nothing)

    n_ok = 0
    n_skipped = 0
    n_bad = 0

    # --- Sampling loop (from sample_branching_full_OU.jl lines 109-191) ---
    for sample_idx in 1:N_SAMPLES
        println("  --- Sample $sample_idx ---")

        initial_length = max(1, rand(Poisson(100)))

        reset_self_conditioning!(wrapper)
        X0 = create_initial_state(initial_length, LATENT_DIM)

        Xt = X0
        sample_failed = false
        for i in 1:nsteps
            t1, t2 = steps[i], steps[i+1]

            L_current = size(Xt.groupings, 1)
            if L_current == 0
                println("    WARNING: Protein reached L=0 at step $i, skipping")
                sample_failed = true
                break
            elseif L_current > 600
                println("    WARNING: Protein grew to L=$L_current at step $i (>600), skipping")
                sample_failed = true
                break
            end

            hat = wrapper(t1, Xt)
            Xt = Flowfusion.step(P_s, Xt, hat, t1, t2)

            if i % max(1, nsteps ÷ 5) == 0
                L_now = size(Xt.groupings, 1)
                println("    Step $i/$nsteps: t=$(round(t2, digits=3)), L=$L_now")
            end
        end

        if sample_failed
            n_skipped += 1
            continue
        end

        # --- Extract + decode (from sample_branching_full_OU.jl lines 145-181) ---
        ca_tensor = tensor(Xt.state[1].S)
        ll_tensor = tensor(Xt.state[2].S)
        ca_coords = dropdims(ca_tensor, dims=3)
        latents = dropdims(ll_tensor, dims=3)
        final_L = size(ca_coords, 2)

        println("    Final length: $final_L (started at $initial_length)")

        if final_L == 0
            println("    Skipping empty protein")
            all_ok = false
            continue
        end

        ok = check_finite(ca_coords, "CA") && check_finite(latents, "LL")

        # Decode and save PDB
        ca_3d = reshape(Array(ca_coords), 3, final_L, 1)
        ll_3d = reshape(Array(latents), LATENT_DIM, final_L, 1)
        mask = ones(Float32, final_L, 1)

        dec_input = Dict(:z_latent => ll_3d, :ca_coors => ca_3d, :mask => mask)
        dec_out = decoder(dec_input)

        samples = Dict(
            :ca_coords => ca_3d, :latents => ll_3d,
            :all_atom_coords => dec_out[:coors], :aatype => dec_out[:aatype_max],
            :atom_mask => dec_out[:atom_mask], :mask => mask
        )
        samples_to_pdb(samples, sample_dir; prefix="branching_$sample_idx", save_all_atom=true)

        aatype = dec_out[:aatype_max][:, 1]
        seq = join([index_to_aa(aa) for aa in aatype])
        println("    Sequence: $(seq[1:min(40, length(seq))])...")

        if ok
            n_ok += 1
        else
            n_bad += 1
        end

        GC.gc()
        CUDA.functional() && CUDA.reclaim()
    end

    println("  Results: $n_ok ok, $n_skipped skipped (length), $n_bad bad (NaN/Inf)")
    if n_ok >= 1
        results["4_branching_sampling"] = :pass
        println("  PASS (at least 1 sample completed successfully)")
    else
        results["4_branching_sampling"] = :fail
        println("  FAIL: No samples completed successfully")
    end

catch e
    results["4_branching_sampling"] = :fail
    println("  FAIL: $(typeof(e)): $e")
    Base.showerror(stdout, e, catch_backtrace())
    println()
end


# ############################################################################
# SUMMARY
# ############################################################################
println()
println("=" ^ 70)
println("INTEGRATION TEST SUMMARY")
println("=" ^ 70)

test_names = [
    "1_standard_training"  => "Standard Flow Training ($N_TRAIN_BATCHES batches)",
    "2_standard_sampling"  => "Standard Sampling ($N_SAMPLES samples)",
    "3_branching_training" => "Branching Training ($N_TRAIN_BATCHES batches)",
    "4_branching_sampling" => "Branching Sampling ($N_SAMPLES samples)"
]

n_pass = 0
n_fail = 0
n_skip = 0

for (key, desc) in test_names
    status = get(results, key, :skip)
    if status == :pass
        global n_pass += 1
        println("  PASS  $desc")
    elseif status == :fail
        global n_fail += 1
        println("  FAIL  $desc")
    else
        global n_skip += 1
        println("  SKIP  $desc")
    end
end

println()
println("Results: $n_pass passed, $n_fail failed, $n_skip skipped out of $(length(test_names))")
println("Output dir: $output_dir")

if n_fail == 0 && n_skip == 0
    println("\nAll tests PASSED!")
else
    println("\nSome tests FAILED or were SKIPPED.")
    exit(1)
end
