#!/usr/bin/env julia
# Test that pre-normalized pair path produces same results as standard path.
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
println("Pre-normalized Pair Path Correctness Test")
println("="^70)

# Create a small model for testing
L = 32; B = 2; token_dim = 64; pair_dim = 32; n_heads = 4; dim_cond = 32

block = TransformerBlock(
    dim_token=token_dim, dim_pair=pair_dim, n_heads=n_heads,
    dim_cond=dim_cond, qk_ln=true
) |> gpu

x = CUDA.randn(Float32, token_dim, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

# Get pair_norm eps from the block's attention
pba = block.mha.mha
pair_eps = pba.pair_norm.ϵ

# Pre-normalize pair features
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

println("\n1. Forward correctness")

# Standard path
y_standard = block(x, pair, cond, mask)

# Pre-normalized path
y_prenorm = LaProteina._transformer_block_prenormed(block, x, pair, pair_normed, cond, mask)

fwd_diff = maxabsdiff(y_standard, y_prenorm)
@printf("  Forward max diff: %.8f %s\n", fwd_diff, fwd_diff < 1e-4 ? "[PASS]" : "[FAIL]")

println("\n2. Backward correctness (gradient w.r.t. x)")

# Standard path gradient
g_standard = Zygote.gradient(x -> sum(block(x, pair, cond, mask)), x)

# Pre-normalized path gradient
g_prenorm = Zygote.gradient(x -> sum(LaProteina._transformer_block_prenormed(
    block, x, pair, pair_normed, cond, mask)), x)

bwd_diff = maxabsdiff(g_standard[1], g_prenorm[1])
@printf("  Backward max diff: %.8f %s\n", bwd_diff, bwd_diff < 1e-3 ? "[PASS]" : "[FAIL]")

println("\n3. Backward correctness (gradient w.r.t. pair)")

# Standard path gradient w.r.t. pair
g_pair_standard = Zygote.gradient(p -> sum(block(x, p, cond, mask)), pair)

# Pre-normalized path: need gradient through pair_normed
g_pair_prenorm = Zygote.gradient(p -> begin
    pn = LaProteina.pytorch_normalise(p; dims=1, eps=pair_eps)
    sum(LaProteina._transformer_block_prenormed(block, x, p, pn, cond, mask))
end, pair)

bwd_pair_diff = maxabsdiff(g_pair_standard[1], g_pair_prenorm[1])
@printf("  Backward (pair) max diff: %.8f %s\n", bwd_pair_diff, bwd_pair_diff < 1e-3 ? "[PASS]" : "[FAIL]")

println("\n4. Multi-block forward (simulating 14 blocks)")
blocks = [TransformerBlock(
    dim_token=token_dim, dim_pair=pair_dim, n_heads=n_heads,
    dim_cond=dim_cond, qk_ln=true
) |> gpu for _ in 1:3]  # Use 3 blocks for speed

# Standard path: each block normalizes pair separately
function standard_forward(x, pair, cond, mask, blocks)
    for b in blocks
        x = b(x, pair, cond, mask)
    end
    return x
end

# Pre-normalized path: normalize once, each block uses affine only
function prenorm_forward(x, pair, pair_normed, cond, mask, blocks)
    for b in blocks
        x = LaProteina._transformer_block_prenormed(b, x, pair, pair_normed, cond, mask)
    end
    return x
end

y_std_multi = standard_forward(x, pair, cond, mask, blocks)
y_pre_multi = prenorm_forward(x, pair, pair_normed, cond, mask, blocks)

multi_fwd_diff = maxabsdiff(y_std_multi, y_pre_multi)
@printf("  Multi-block forward max diff: %.8f %s\n", multi_fwd_diff, multi_fwd_diff < 1e-3 ? "[PASS]" : "[FAIL]")

# Multi-block backward
g_std_multi = Zygote.gradient(x -> sum(standard_forward(x, pair, cond, mask, blocks)), x)
g_pre_multi = Zygote.gradient(x -> begin
    pn = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)
    sum(prenorm_forward(x, pair, pn, cond, mask, blocks))
end, x)

multi_bwd_diff = maxabsdiff(g_std_multi[1], g_pre_multi[1])
@printf("  Multi-block backward max diff: %.8f %s\n", multi_bwd_diff, multi_bwd_diff < 1e-2 ? "[PASS]" : "[FAIL]")

println("\n5. Timing comparison (3 blocks)")
function bench(name, f, n_warmup=5, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-40s %.3f ms\n", name, t / n_iter * 1000)
end

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

println("\n" * "="^70)
println("Test complete.")
println("="^70)
