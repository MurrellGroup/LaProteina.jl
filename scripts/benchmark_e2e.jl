#!/usr/bin/env julia
# End-to-end benchmark: standard vs GPU-optimized ScoreNetwork forward.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()
println("TF32 math mode: ", CUDA.math_mode())
println("GPU: ", CUDA.name(CUDA.device()))
println("cuTile available: ", LaProteina._HAS_CUTILE)

L = 128; B = 4
token_dim = 768; pair_dim = 256; n_heads = 12; dim_cond = 256

println("\n" * "="^70)
println("End-to-End ScoreNetwork Benchmark (L=$L, B=$B)")
println("="^70)

# Create model
model = ScoreNetwork(
    n_layers=14, token_dim=token_dim, pair_dim=pair_dim,
    n_heads=n_heads, dim_cond=dim_cond, qk_ln=true,
    update_pair_repr=false
) |> gpu

# Create synthetic raw features (simulating what comes from data loading)
seq_raw_dim = size(model.init_repr_factory.projection.weight, 2)
cond_raw_dim = size(model.cond_factory.projection.weight, 2)
pair_raw_dim = size(model.pair_rep_builder.init_repr_factory.projection.weight, 2)
pair_cond_raw_dim = size(model.pair_rep_builder.cond_factory.projection.weight, 2)

raw_features = ScoreNetworkRawFeatures(
    CUDA.randn(Float32, seq_raw_dim, L, B),
    CUDA.randn(Float32, cond_raw_dim, L, B),
    CUDA.randn(Float32, pair_raw_dim, L, L, B),
    CUDA.randn(Float32, pair_cond_raw_dim, L, L, B),
    CUDA.ones(Float32, L, B)
)

function bench_fwd(name, f, n_warmup=3, n_iter=5)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-50s %.1f ms\n", name, t / n_iter * 1000)
end

function bench_bwd(name, f_loss, x, n_warmup=2, n_iter=3)
    for _ in 1:n_warmup
        Zygote.gradient(f_loss, x)
    end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter
        Zygote.gradient(f_loss, x)
    end
    @printf("  %-50s %.1f ms\n", name, t / n_iter * 1000)
end

println("\n--- Forward pass (14 blocks) ---")
bench_fwd("forward_from_raw_features (standard)", () -> begin
    forward_from_raw_features(model, raw_features)
    nothing
end)
bench_fwd("forward_from_raw_features_gpu (prenorm)", () -> begin
    forward_from_raw_features_gpu(model, raw_features)
    nothing
end)

println("\n--- Backward pass (14 blocks, gradient w.r.t. model) ---")
bench_bwd("standard backward", m -> begin
    out = forward_from_raw_features(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
end, model)

bench_bwd("GPU-optimized backward", m -> begin
    out = forward_from_raw_features_gpu(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
end, model)

println("\n--- GPU Memory ---")
CUDA.memory_status()
@printf("  Available: %.2f GB\n", CUDA.available_memory() / 1e9)

println("\n" * "="^70)
println("End-to-end benchmark complete.")
println("="^70)
