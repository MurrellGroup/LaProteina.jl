#!/usr/bin/env julia
# Test that forward_branching_from_raw_features_gpu works correctly
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4

# Create base ScoreNetwork + BranchingScoreNetwork
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

println("Testing forward_branching_from_raw_features_gpu...")
out_gpu = forward_branching_from_raw_features_gpu(model, raw_features)
println("  bb_ca output size: ", size(out_gpu[:bb_ca][:v]))
println("  local_latents output size: ", size(out_gpu[:local_latents][:v]))
println("  split output size: ", size(out_gpu[:split]))
println("  del output size: ", size(out_gpu[:del]))

println("\nTesting backward...")
loss_val, grads = Flux.withgradient(model) do m
    out = forward_branching_from_raw_features_gpu(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end
println("  Loss value: ", loss_val)
println("  Has gradients: ", !isnothing(grads[1]))

# Compare with standard path
println("\nComparing standard vs GPU-optimized...")
out_std = forward_branching_from_raw_features(model, raw_features)

for key in [:bb_ca, :local_latents]
    d = maximum(abs.(Array(out_std[key][:v]) .- Array(out_gpu[key][:v])))
    @printf("  %s max diff: %.6f %s\n", key, d, d < 0.01 ? "[PASS]" : "[FAIL]")
end
for key in [:split, :del]
    d = maximum(abs.(Array(out_std[key]) .- Array(out_gpu[key])))
    @printf("  %s max diff: %.6f %s\n", key, d, d < 0.01 ? "[PASS]" : "[FAIL]")
end

# Timing
function bench(name, f, n_warmup=2, n_iter=5)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-45s %.1f ms\n", name, t / n_iter * 1000)
end

println("\n--- Branching model timing (L=$L, B=$B) ---")
bench("Standard forward", () -> forward_branching_from_raw_features(model, raw_features))
bench("GPU-optimized forward", () -> forward_branching_from_raw_features_gpu(model, raw_features))

bench("Standard backward", () -> begin
    Flux.withgradient(model) do m
        out = forward_branching_from_raw_features(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
    end
end)

bench("GPU-optimized backward", () -> begin
    Flux.withgradient(model) do m
        out = forward_branching_from_raw_features_gpu(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
    end
end)

println("\nDone!")
