#!/usr/bin/env julia
# Benchmark: cuTile (flash attention + fused LayerNorm) vs nocutile path
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

L = 128; B = 4

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

raw_features = ScoreNetworkRawFeatures(
    CUDA.randn(Float32, seq_raw_dim, L, B),
    CUDA.randn(Float32, cond_raw_dim, L, B),
    CUDA.randn(Float32, pair_raw_dim, L, L, B),
    CUDA.randn(Float32, pair_cond_raw_dim, L, L, B),
    CUDA.ones(Float32, L, B)
)

# ============================================
# Benchmark: Forward only (GPU path)
# ============================================
println("\n--- Forward (GPU-optimized, cuTile) ---")
# Warmup
for _ in 1:3
    out = forward_branching_from_raw_features_gpu(model, raw_features)
    CUDA.synchronize()
end

N = 50
t_fwd_gpu = Float64[]
for _ in 1:N
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        out = forward_branching_from_raw_features_gpu(model, raw_features)
    end
    push!(t_fwd_gpu, t)
end
@printf("  GPU forward:  mean=%.1fms  median=%.1fms  min=%.1fms\n",
    mean(t_fwd_gpu)*1000, median(t_fwd_gpu)*1000, minimum(t_fwd_gpu)*1000)

# ============================================
# Benchmark: Forward only (standard path)
# ============================================
println("\n--- Forward (standard, no flash/fused) ---")
for _ in 1:3
    out = forward_branching_from_raw_features(model, raw_features)
    CUDA.synchronize()
end

t_fwd_std = Float64[]
for _ in 1:N
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        out = forward_branching_from_raw_features(model, raw_features)
    end
    push!(t_fwd_std, t)
end
@printf("  Std forward:  mean=%.1fms  median=%.1fms  min=%.1fms\n",
    mean(t_fwd_std)*1000, median(t_fwd_std)*1000, minimum(t_fwd_std)*1000)

# ============================================
# Benchmark: Forward + Backward (GPU path)
# ============================================
println("\n--- Forward + Backward (GPU-optimized, cuTile) ---")
seq_raw = raw_features.seq_raw
cond_raw = raw_features.cond_raw
pair_raw = raw_features.pair_raw
pair_cond_raw = raw_features.pair_cond_raw
mask = raw_features.mask

function loss_gpu(sr)
    rf = ScoreNetworkRawFeatures(sr, cond_raw, pair_raw, pair_cond_raw, mask)
    out = forward_branching_from_raw_features_gpu(model, rf)
    return sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end

# Warmup
for _ in 1:2
    Zygote.gradient(loss_gpu, seq_raw)
    CUDA.synchronize()
end

t_train_gpu = Float64[]
for _ in 1:30
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        Zygote.gradient(loss_gpu, seq_raw)
    end
    push!(t_train_gpu, t)
end
@printf("  GPU fwd+bwd:  mean=%.1fms  median=%.1fms  min=%.1fms\n",
    mean(t_train_gpu)*1000, median(t_train_gpu)*1000, minimum(t_train_gpu)*1000)

# ============================================
# Benchmark: Forward + Backward (standard path)
# ============================================
println("\n--- Forward + Backward (standard, no flash/fused) ---")

function loss_std(sr)
    rf = ScoreNetworkRawFeatures(sr, cond_raw, pair_raw, pair_cond_raw, mask)
    out = forward_branching_from_raw_features(model, rf)
    return sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end

for _ in 1:2
    Zygote.gradient(loss_std, seq_raw)
    CUDA.synchronize()
end

t_train_std = Float64[]
for _ in 1:30
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        Zygote.gradient(loss_std, seq_raw)
    end
    push!(t_train_std, t)
end
@printf("  Std fwd+bwd:  mean=%.1fms  median=%.1fms  min=%.1fms\n",
    mean(t_train_std)*1000, median(t_train_std)*1000, minimum(t_train_std)*1000)

# ============================================
# Summary
# ============================================
println("\n" * "="^60)
println("SUMMARY (L=$L, B=$B, 14 blocks)")
println("="^60)
@printf("  Forward:   GPU=%.1fms  Std=%.1fms  Speedup=%.2fx\n",
    median(t_fwd_gpu)*1000, median(t_fwd_std)*1000,
    median(t_fwd_std)/median(t_fwd_gpu))
@printf("  Fwd+Bwd:   GPU=%.1fms  Std=%.1fms  Speedup=%.2fx\n",
    median(t_train_gpu)*1000, median(t_train_std)*1000,
    median(t_train_std)/median(t_train_gpu))

# Estimate training time
steps_20k = 20000
est_gpu = steps_20k * median(t_train_gpu) / 3600
est_std = steps_20k * median(t_train_std) / 3600
@printf("  Est 20k steps: GPU=%.1fh  Std=%.1fh\n", est_gpu, est_std)
println("="^60)
