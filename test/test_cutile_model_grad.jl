#!/usr/bin/env julia
# Minimal test: full model forward+backward with cuTile, check for NaN.
# Also runs a few optimizer steps to detect divergence.
# Compare: julia -t 1 test/test_cutile_model_grad.jl
# vs:      LAPROTEINA_NOCUTILE=1 julia -t 1 test/test_cutile_model_grad.jl
# IMPORTANT: clear cache between: rm -rf ~/.julia/compiled/v1.12/LaProteina/

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CUDA
using Flux
using Zygote
using Statistics
using Printf
using Random

using LaProteina

println("cuTile: ", LaProteina._HAS_CUTILE, " | nocutile: ", LaProteina._FORCE_NOCUTILE)
LaProteina.enable_tf32!()
CUDA.allowscalar(false)

# ============================================================================
# Load model
# ============================================================================

model_path = joinpath(@__DIR__, "..", "models", "branching_score_network_bf.bson")
if !isfile(model_path)
    model_path = joinpath(@__DIR__, "..", "models", "branching_score_network.bson")
end
println("Loading model from: ", model_path)
model = LaProteina.load_branching_model(model_path) |> gpu
println("Model loaded")

# ============================================================================
# Create a synthetic batch with fixed seed
# ============================================================================

function make_synthetic_batch(model, L, B; seed=42)
    Random.seed!(seed)
    # Match what prepare_training_batch produces
    base = model.base

    # ScoreNetworkRawFeatures has: seq_raw, cond_raw, pair_raw, pair_cond_raw, mask
    # Get dims from model's projection layers
    seq_dim = size(base.init_repr_factory.projection.weight, 2)
    cond_dim = size(base.cond_factory.projection.weight, 2)
    pair_dim = size(base.pair_rep_builder.init_repr_factory.projection.weight, 2)

    # Pair conditioning: typically same as cond_dim
    pair_cond_dim = cond_dim
    if !isnothing(base.pair_rep_builder.cond_factory)
        pair_cond_dim = size(base.pair_rep_builder.cond_factory.projection.weight, 2)
    end

    seq_raw = CUDA.randn(Float32, seq_dim, L, B) .* 0.1f0
    cond_raw = CUDA.randn(Float32, cond_dim, L, B) .* 0.1f0
    pair_raw = CUDA.randn(Float32, pair_dim, L, L, B) .* 0.01f0
    pair_cond_raw = CUDA.randn(Float32, pair_cond_dim, L, L, B) .* 0.01f0
    mask = CUDA.ones(Float32, L, B)

    features = LaProteina.ScoreNetworkRawFeatures(seq_raw, cond_raw, pair_raw, pair_cond_raw, mask)

    # Targets
    ca_tgt = CUDA.randn(Float32, 3, L, B) .* 0.5f0
    ll_tgt = CUDA.randn(Float32, 8, L, B) .* 0.5f0

    return features, ca_tgt, ll_tgt, mask
end

# ============================================================================
# Loss function
# ============================================================================

function compute_loss(model, features, ca_tgt, ll_tgt, mask)
    out = LaProteina.forward_branching_from_raw_features_gpu(model, features)

    ca_pred = out[:bb_ca][model.base.output_param]
    ll_pred = out[:local_latents][model.base.output_param]

    mask_exp = reshape(mask, 1, size(mask)...)

    ca_diff = (ca_pred .- ca_tgt) .* mask_exp
    ll_diff = (ll_pred .- ll_tgt) .* mask_exp

    n_valid = max(sum(mask), 1f0)

    ca_loss = sum(ca_diff .^ 2) / n_valid * 2f0
    ll_loss = sum(ll_diff .^ 2) / n_valid * 0.1f0

    return ca_loss + ll_loss
end

# ============================================================================
# Gradient analysis
# ============================================================================

function analyze_grads(grads; show_all=false)
    n_nan = 0
    n_inf = 0
    n_params = 0
    max_grad = 0f0
    max_grad_name = ""

    function walk(tree, prefix)
        if tree === nothing
            return
        elseif tree isa NamedTuple
            for name in keys(tree)
                walk(getfield(tree, name), "$(prefix).$(name)")
            end
        elseif tree isa Tuple
            for (i, v) in enumerate(tree)
                walk(v, "$(prefix)[$(i)]")
            end
        elseif tree isa AbstractArray
            arr = Array(tree)
            local_nan = count(isnan, arr)
            local_inf = count(isinf, arr)
            local_max = length(arr) > 0 ? maximum(x -> isnan(x) || isinf(x) ? 0f0 : abs(x), arr) : 0f0
            n_nan += local_nan
            n_inf += local_inf
            n_params += 1
            if local_max > max_grad
                max_grad = local_max
                max_grad_name = prefix
            end
            if local_nan > 0 || local_inf > 0 || show_all
                @printf("    %s: max=%.3e, nNaN=%d, nInf=%d, shape=%s\n",
                        prefix, local_max, local_nan, local_inf, string(size(tree)))
            end
        end
    end

    walk(grads, "model")
    return n_nan, n_inf, n_params, max_grad, max_grad_name
end

# ============================================================================
# Test 1: Single forward+backward
# ============================================================================

println("\n=== Test 1: Single forward+backward (L=100, B=4) ===")
features, ca_tgt, ll_tgt, mask = make_synthetic_batch(model, 100, 4)

loss_val, grads = Flux.withgradient(model) do m
    compute_loss(m, features, ca_tgt, ll_tgt, mask)
end

@printf("Loss: %.6f (NaN: %s)\n", Float64(loss_val), isnan(loss_val))
n_nan, n_inf, n_params, max_grad, max_name = analyze_grads(grads[1])
@printf("Grads: %d params, %d NaN, %d Inf, max=%.4e at %s\n",
        n_params, n_nan, n_inf, max_grad, max_name)

# ============================================================================
# Test 2: Multiple optimizer steps (detect divergence)
# ============================================================================

println("\n=== Test 2: 10 optimizer steps (L=100, B=4, LR=1e-5) ===")

# Reload model fresh
model2 = LaProteina.load_branching_model(model_path) |> gpu
opt = Flux.setup(Muon(eta=Float64(1e-5)), model2)

for step in 1:10
    features, ca_tgt, ll_tgt, mask = make_synthetic_batch(model2, 100, 4; seed=step)

    loss_val, grads = Flux.withgradient(model2) do m
        compute_loss(m, features, ca_tgt, ll_tgt, mask)
    end

    loss_f = Float64(loss_val)
    n_nan, n_inf, _, max_grad, _ = analyze_grads(grads[1])

    @printf("Step %2d: loss=%.4f, grad_max=%.3e, nNaN=%d, nInf=%d",
            step, loss_f, max_grad, n_nan, n_inf)

    if isnan(loss_f)
        println(" [LOSS NaN — STOPPING]")
        break
    elseif n_nan > 0
        println(" [GRAD NaN — STOPPING]")
        # Show which params have NaN
        analyze_grads(grads[1]; show_all=false)
        break
    else
        println()
    end

    Flux.update!(opt, model2, grads[1])

    # Check weights
    param_nan = 0
    for p in Flux.params(model2)
        param_nan += count(isnan, Array(p))
    end
    if param_nan > 0
        @printf("  WEIGHTS have %d NaN after step %d — STOPPING\n", param_nan, step)
        break
    end
end

println("\n=== Done ===")
