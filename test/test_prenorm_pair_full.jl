#!/usr/bin/env julia
# Test pre-normalized pair path with full-size dimensions.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

function maxabsdiff(a, b)
    maximum(abs.(Array(a) .- Array(b)))
end

println("="^70)
println("Pre-normalized Pair Path - Full Size Test")
println("="^70)

# Full training dimensions
L = 128; B = 4; token_dim = 768; pair_dim = 256; n_heads = 12; dim_cond = 256

block = TransformerBlock(
    dim_token=token_dim, dim_pair=pair_dim, n_heads=n_heads,
    dim_cond=dim_cond, qk_ln=true
) |> gpu

x = CUDA.randn(Float32, token_dim, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

pba = block.mha.mha
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

function bench(name, f, n_warmup=3, n_iter=10)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-50s %.3f ms\n", name, t / n_iter * 1000)
end

println("\n--- Single block ---")
bench("Standard forward", () -> block(x, pair, cond, mask))
bench("Prenorm forward", () -> LaProteina._transformer_block_prenormed(block, x, pair, pair_normed, cond, mask))

bench("Standard backward", () -> begin
    Zygote.gradient(x -> sum(block(x, pair, cond, mask)), x)
    nothing
end)
bench("Prenorm backward", () -> begin
    Zygote.gradient(x -> sum(LaProteina._transformer_block_prenormed(
        block, x, pair, pair_normed, cond, mask)), x)
    nothing
end)

# Multi-block comparison
blocks = [TransformerBlock(
    dim_token=token_dim, dim_pair=pair_dim, n_heads=n_heads,
    dim_cond=dim_cond, qk_ln=true
) |> gpu for _ in 1:3]  # 3 blocks

function standard_forward(x, pair, cond, mask, blocks)
    for b in blocks; x = b(x, pair, cond, mask); end
    return x
end

function prenorm_forward(x, pair, pair_normed, cond, mask, blocks)
    for b in blocks
        x = LaProteina._transformer_block_prenormed(b, x, pair, pair_normed, cond, mask)
    end
    return x
end

println("\n--- 3 blocks (L=$L, B=$B, pair_dim=$pair_dim) ---")
bench("Standard forward (3 blocks)", () -> standard_forward(x, pair, cond, mask, blocks))
bench("Prenorm forward (3 blocks)", () -> prenorm_forward(x, pair, pair_normed, cond, mask, blocks))

bench("Standard backward (3 blocks)", () -> begin
    Zygote.gradient(x -> sum(standard_forward(x, pair, cond, mask, blocks)), x)
    nothing
end)

bench("Prenorm backward (3 blocks)", () -> begin
    Zygote.gradient(x -> begin
        pn = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)
        sum(prenorm_forward(x, pair, pn, cond, mask, blocks))
    end, x)
    nothing
end)

println("\n--- forward_from_raw_features_gpu comparison ---")
println("  (Requires full ScoreNetwork - testing with individual blocks only)")
println("  Pair tensor: $(pair_dim) × $(L) × $(L) × $(B) = $(pair_dim * L * L * B) floats")
println("  Pair tensor size: $(pair_dim * L * L * B * 4 / 1e6) MB")

# Estimate savings
println("\n--- Estimated savings for 14-block model ---")
println("  Per-block pair_norm: ~6ms forward, ~10ms backward")
println("  Pre-normalize once: ~6ms forward, ~10ms backward")
println("  Affine-only per block: ~3ms forward, ~2ms backward (estimate)")
println("  Standard 14 blocks: 14 × (6+10) = 224ms for pair_norm")
println("  Prenorm 14 blocks: 1 × (6+10) + 14 × (3+2) = 86ms for pair processing")
println("  Estimated savings: ~138ms per training step")

println("\n" * "="^70)
println("Test complete.")
println("="^70)
