#!/usr/bin/env julia
# Profile GPU memory allocations during forward+backward
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4

model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

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

# Warmup
for _ in 1:2
    Flux.withgradient(model) do m
        out = forward_from_raw_features_gpu(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
    end
end
CUDA.synchronize()

# Key tensor sizes
println("="^70)
println("Key Tensor Sizes (L=$L, B=$B)")
println("="^70)

token_size = 768 * L * B * 4
pair_size = 256 * L * L * B * 4
attn_size = L * L * 12 * B * 4
@printf("  Token tensor [768,%d,%d]: %.2f MB\n", L, B, token_size / 1e6)
@printf("  Pair tensor [256,%d,%d,%d]: %.2f MB\n", L, L, B, pair_size / 1e6)
@printf("  Pair normed [256,%d,%d,%d]: %.2f MB\n", L, L, B, pair_size / 1e6)
@printf("  Attention scores [%d,%d,12,%d]: %.2f MB\n", L, L, B, attn_size / 1e6)
@printf("  QKV tensor [2304,%d,%d]: %.2f MB\n", L, B, 2304 * L * B * 4 / 1e6)
@printf("  SwiGLU expanded [6144,%d,%d]: %.2f MB\n", L, B, 6144 * L * B * 4 / 1e6)
@printf("  Pair bias [12,%d,%d,%d]: %.2f MB\n", L, L, B, attn_size / 1e6)

# Model parameter count
total_params = sum(length, Flux.params(model))
@printf("\n  Model parameters: %.2fM (%.2f MB)\n", total_params / 1e6, total_params * 4 / 1e6)

# Overall memory
println("\n--- GPU Memory ---")
CUDA.memory_status()
@printf("  Available: %.2f GB\n", CUDA.available_memory() / 1e9)

println("\n" * "="^70)
println("Done.")
println("="^70)
