#!/usr/bin/env julia
# Verify gradient correctness: GPU-optimized path vs standard path
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
using CUDA
using Flux: Zygote
using Printf
using Statistics

LaProteina.enable_tf32!()
println("GPU: ", CUDA.name(CUDA.device()))
println("cuTile: ", LaProteina._HAS_CUTILE)

L = 64; B = 2  # Smaller for gradient test

base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

model = BranchingScoreNetwork(base) |> gpu

seq_raw_dim = size(base.init_repr_factory.projection.weight, 2)
cond_raw_dim = size(base.cond_factory.projection.weight, 2)
pair_raw_dim = size(base.pair_rep_builder.init_repr_factory.projection.weight, 2)
pair_cond_raw_dim = size(base.pair_rep_builder.cond_factory.projection.weight, 2)

# Create raw features
seq_raw = CUDA.randn(Float32, seq_raw_dim, L, B)
cond_raw = CUDA.randn(Float32, cond_raw_dim, L, B)
pair_raw = CUDA.randn(Float32, pair_raw_dim, L, L, B)
pair_cond_raw = CUDA.randn(Float32, pair_cond_raw_dim, L, L, B)
mask = CUDA.ones(Float32, L, B)

raw_features = ScoreNetworkRawFeatures(seq_raw, cond_raw, pair_raw, pair_cond_raw, mask)

# ============================================
# Test: Forward values match
# ============================================
println("\n--- Forward comparison ---")
out_std = forward_branching_from_raw_features(model, raw_features)
out_gpu = forward_branching_from_raw_features_gpu(model, raw_features)

for key in [:bb_ca, :local_latents]
    vs = Array(out_std[key][:v])
    vg = Array(out_gpu[key][:v])
    local d = maximum(abs.(vs .- vg))
    r = d / (maximum(abs.(vs)) + 1e-8)
    @printf("  %-20s max diff: %.6f  rel: %.6f\n", key, d, r)
end

# ============================================
# Test: Gradient of loss w.r.t. seq_raw
# ============================================
println("\n--- Gradient comparison (seq_raw) ---")

function loss_std(sr)
    rf = ScoreNetworkRawFeatures(sr, cond_raw, pair_raw, pair_cond_raw, mask)
    out = forward_branching_from_raw_features(model, rf)
    return sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end

function loss_gpu(sr)
    rf = ScoreNetworkRawFeatures(sr, cond_raw, pair_raw, pair_cond_raw, mask)
    out = forward_branching_from_raw_features_gpu(model, rf)
    return sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end

# Warmup
println("  Computing standard gradient...")
grad_std = Zygote.gradient(loss_std, seq_raw)[1]
println("  Computing GPU gradient...")
grad_gpu = Zygote.gradient(loss_gpu, seq_raw)[1]

gs = Array(grad_std)
gg = Array(grad_gpu)
d_grad = maximum(abs.(gs .- gg))
r_grad = d_grad / (maximum(abs.(gs)) + 1e-8)
@printf("  seq_raw gradient: max diff = %.6f  max|grad| = %.2f  rel = %.6f\n",
    d_grad, maximum(abs.(gs)), r_grad)

# ============================================
# Test: Gradient of loss w.r.t. pair_raw
# ============================================
println("\n--- Gradient comparison (pair_raw) ---")

function loss_std_pair(pr)
    rf = ScoreNetworkRawFeatures(seq_raw, cond_raw, pr, pair_cond_raw, mask)
    out = forward_branching_from_raw_features(model, rf)
    return sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end

function loss_gpu_pair(pr)
    rf = ScoreNetworkRawFeatures(seq_raw, cond_raw, pr, pair_cond_raw, mask)
    out = forward_branching_from_raw_features_gpu(model, rf)
    return sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end

println("  Computing standard gradient...")
grad_std_p = Zygote.gradient(loss_std_pair, pair_raw)[1]
println("  Computing GPU gradient...")
grad_gpu_p = Zygote.gradient(loss_gpu_pair, pair_raw)[1]

gsp = Array(grad_std_p)
ggp = Array(grad_gpu_p)
d_gradp = maximum(abs.(gsp .- ggp))
r_gradp = d_gradp / (maximum(abs.(gsp)) + 1e-8)
@printf("  pair_raw gradient: max diff = %.6f  max|grad| = %.2f  rel = %.6f\n",
    d_gradp, maximum(abs.(gsp)), r_gradp)

# Summary
println("\n--- Summary ---")
@printf("  Forward:  max rel diff = %.6f\n", maximum([
    maximum(abs.(Array(out_std[:bb_ca][:v]) .- Array(out_gpu[:bb_ca][:v]))) / (maximum(abs.(Array(out_std[:bb_ca][:v]))) + 1e-8),
    maximum(abs.(Array(out_std[:local_latents][:v]) .- Array(out_gpu[:local_latents][:v]))) / (maximum(abs.(Array(out_std[:local_latents][:v]))) + 1e-8)
]))
@printf("  Gradient seq_raw:  rel = %.6f\n", r_grad)
@printf("  Gradient pair_raw: rel = %.6f\n", r_gradp)
println("\nDone!")
