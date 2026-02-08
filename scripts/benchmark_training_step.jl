#!/usr/bin/env julia
# Benchmark realistic training step (gradient w.r.t. ALL model parameters)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
using CUDA
using Flux
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

opt_state = Flux.setup(Flux.Adam(2.5f-4), model)

# GPU-optimized training step
println("\n--- GPU-optimized training step ---")

function train_step_gpu!(model, raw_features, opt_state)
    local loss_val
    gs = Zygote.gradient(model) do m
        out = forward_branching_from_raw_features_gpu(m, raw_features)
        loss_val = sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
        return loss_val
    end
    Flux.update!(opt_state, model, gs[1])
    return loss_val
end

for _ in 1:3
    train_step_gpu!(model, raw_features, opt_state)
    CUDA.synchronize()
end

N = 30
t_gpu = Float64[]
for _ in 1:N
    CUDA.synchronize()
    t = CUDA.@elapsed train_step_gpu!(model, raw_features, opt_state)
    push!(t_gpu, t)
end
@printf("  GPU step:  mean=%.1fms  median=%.1fms  min=%.1fms\n",
    mean(t_gpu)*1000, median(t_gpu)*1000, minimum(t_gpu)*1000)

# Standard training step
println("\n--- Standard training step ---")

opt_state2 = Flux.setup(Flux.Adam(2.5f-4), model)

function train_step_std!(model, raw_features, opt_state)
    local loss_val
    gs = Zygote.gradient(model) do m
        out = forward_branching_from_raw_features(m, raw_features)
        loss_val = sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
        return loss_val
    end
    Flux.update!(opt_state, model, gs[1])
    return loss_val
end

for _ in 1:3
    train_step_std!(model, raw_features, opt_state2)
    CUDA.synchronize()
end

t_std = Float64[]
for _ in 1:N
    CUDA.synchronize()
    t = CUDA.@elapsed train_step_std!(model, raw_features, opt_state2)
    push!(t_std, t)
end
@printf("  Std step:  mean=%.1fms  median=%.1fms  min=%.1fms\n",
    mean(t_std)*1000, median(t_std)*1000, minimum(t_std)*1000)

# Summary
println("\n" * "="^60)
println("REALISTIC TRAINING STEP (L=$L, B=$B, 14 blocks)")
println("="^60)
@printf("  GPU step:  %.1fms (median)\n", median(t_gpu)*1000)
@printf("  Std step:  %.1fms (median)\n", median(t_std)*1000)
@printf("  Speedup:   %.2fx\n", median(t_std)/median(t_gpu))
@printf("  Est 20k steps: GPU=%.1fh  Std=%.1fh\n",
    20000 * median(t_gpu) / 3600, 20000 * median(t_std) / 3600)
@printf("  Est 100k steps: GPU=%.1fh  Std=%.1fh\n",
    100000 * median(t_gpu) / 3600, 100000 * median(t_std) / 3600)
println("="^60)
