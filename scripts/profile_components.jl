#!/usr/bin/env julia
# Profile individual components to find remaining bottlenecks
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
using CUDA
using Flux: Zygote
using Printf
using Statistics

LaProteina.enable_tf32!()
println("GPU: ", CUDA.name(CUDA.device()))

L = 128; B = 4

base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

block = base.transformer_layers[1]
pba = block.mha.mha

node_feats = CUDA.randn(Float32, 768, L, B)
pair_feats = CUDA.randn(Float32, 256, L, L, B)
cond = CUDA.randn(Float32, 256, L, B)
mask = CUDA.ones(Float32, L, B)
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair_feats; dims=1, eps=pair_eps)

function bench(f, n_warmup=5, n_iter=50)
    for _ in 1:n_warmup; f(); CUDA.synchronize(); end
    times = Float64[]
    for _ in 1:n_iter
        CUDA.synchronize()
        t = CUDA.@elapsed f()
        push!(times, t)
    end
    return median(times) * 1000
end

println("\n--- Forward components ---")

# PBA prenormed forward
t = bench(() -> LaProteina._pair_bias_attn_prenormed(pba, node_feats, pair_normed, mask))
@printf("  PBA prenormed fwd:     %.2fms\n", t)

# AdaLN
t = bench(() -> block.mha.adaln(node_feats, cond, mask))
@printf("  AdaLN fwd:             %.2fms\n", t)

# SwiGLU transition
t = bench(() -> block.transition(node_feats, cond, mask))
@printf("  Transition fwd:        %.2fms\n", t)

# Full prenormed block
t = bench(() -> LaProteina._transformer_block_prenormed(block, node_feats, pair_feats, pair_normed, cond, mask))
@printf("  Block prenormed fwd:   %.2fms\n", t)

# Pre-normalization (pytorch_normalise on pair)
t = bench(() -> LaProteina.pytorch_normalise(pair_feats; dims=1, eps=pair_eps))
@printf("  pair_normalise fwd:    %.2fms\n", t)

# _apply_pair_affine
t = bench(() -> LaProteina._apply_pair_affine(pair_normed, pba.pair_norm, pba.to_bias))
@printf("  _apply_pair_affine:    %.2fms\n", t)

# Flash attention isolated
d, h = pba.dim_head, pba.n_heads
Q = CUDA.randn(Float32, d, L, h, B)
K = CUDA.randn(Float32, d, L, h, B)
V = CUDA.randn(Float32, d, L, h, B)
bias = CUDA.randn(Float32, L, L, h, B)
t = bench(() -> LaProteina.flash_attention_bias(Q, K, V, bias; scale=pba.scale))
@printf("  flash_attn_bias fwd:   %.2fms\n", t)

println("\n--- Backward components ---")

# PBA prenormed backward
t = bench(() -> Zygote.gradient(x -> sum(LaProteina._pair_bias_attn_prenormed(pba, x, pair_normed, mask)), node_feats))
@printf("  PBA prenormed fwd+bwd: %.2fms\n", t)

# Full prenormed block backward
t = bench(() -> Zygote.gradient(x -> sum(LaProteina._transformer_block_prenormed(block, x, pair_feats, pair_normed, cond, mask)), node_feats))
@printf("  Block prenormed f+b:   %.2fms\n", t)

# Flash attention backward
t = bench(() -> Zygote.gradient(Q) do q
    LaProteina.flash_attention_bias_forward(q, K, V, bias; scale=pba.scale) |> sum
end)
@printf("  flash_attn_bias f+b:   %.2fms\n", t)

# LayerNorm backward (768)
ln768 = base.transformer_layers[1].mha.mha.node_norm
x768 = CUDA.randn(Float32, 768, L, B)
t = bench(() -> Zygote.gradient(x -> sum(ln768(x)), x768))
@printf("  LayerNorm(768) f+b:    %.2fms\n", t)

# AdaLN backward
t = bench(() -> Zygote.gradient(x -> sum(block.mha.adaln(x, cond, mask)), node_feats))
@printf("  AdaLN f+b:             %.2fms\n", t)

println("\n--- Summary ---")
@printf("  Block fwd:  %.2fms  ×14 = %.1fms\n", 
    bench(() -> LaProteina._transformer_block_prenormed(block, node_feats, pair_feats, pair_normed, cond, mask)),
    bench(() -> LaProteina._transformer_block_prenormed(block, node_feats, pair_feats, pair_normed, cond, mask)) * 14)
t_block_fb = bench(() -> Zygote.gradient(x -> sum(LaProteina._transformer_block_prenormed(block, x, pair_feats, pair_normed, cond, mask)), node_feats))
@printf("  Block f+b:  %.2fms  ×14 = %.1fms\n", t_block_fb, t_block_fb * 14)

println("\nDone!")
