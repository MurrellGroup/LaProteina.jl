#!/usr/bin/env julia
# Test fused affine with explicit gradient (not Flux.params)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 32; B = 2; pair_dim = 256; n_heads = 12

block = TransformerBlock(
    dim_token=768, dim_pair=pair_dim, n_heads=n_heads,
    dim_cond=256, qk_ln=true
) |> gpu

x = CUDA.randn(Float32, 768, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
cond = CUDA.randn(Float32, 256, L, B)
mask = CUDA.ones(Float32, L, B)

pba = block.mha.mha
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

# Test 1: Full block gradient w.r.t. block params (the actual use case)
println("Test 1: Full block gradient (standard vs prenorm)")
g_std = Zygote.gradient(b -> sum(b(x, pair, cond, mask)), block)
g_pre = Zygote.gradient(b -> begin
    sum(LaProteina._transformer_block_prenormed(b, x, pair, pair_normed, cond, mask))
end, block)

# Compare pair_norm.scale gradient
scale_g_std = g_std[1].mha.mha.pair_norm.scale
scale_g_pre = g_pre[1].mha.mha.pair_norm.scale
if !isnothing(scale_g_std) && !isnothing(scale_g_pre)
    d = maximum(abs.(Array(scale_g_std) .- Array(scale_g_pre)))
    r = d / (maximum(abs.(Array(scale_g_std))) + 1e-8)
    @printf("pair_norm.scale grad maxdiff: %.6f (rel: %.6f) %s\n", d, r, r < 0.01 ? "[PASS]" : "[FAIL]")
else
    println("pair_norm.scale gradient: std=$(typeof(scale_g_std)), pre=$(typeof(scale_g_pre))")
end

# Compare pair_norm.bias gradient
bias_g_std = g_std[1].mha.mha.pair_norm.bias
bias_g_pre = g_pre[1].mha.mha.pair_norm.bias
if !isnothing(bias_g_std) && !isnothing(bias_g_pre)
    d = maximum(abs.(Array(bias_g_std) .- Array(bias_g_pre)))
    r = d / (maximum(abs.(Array(bias_g_std))) + 1e-8)
    @printf("pair_norm.bias grad maxdiff: %.6f (rel: %.6f) %s\n", d, r, r < 0.01 ? "[PASS]" : "[FAIL]")
else
    println("pair_norm.bias gradient: std=$(typeof(bias_g_std)), pre=$(typeof(bias_g_pre))")
end

# Compare to_bias.weight gradient
W_g_std = g_std[1].mha.mha.to_bias.weight
W_g_pre = g_pre[1].mha.mha.to_bias.weight
if !isnothing(W_g_std) && !isnothing(W_g_pre)
    d = maximum(abs.(Array(W_g_std) .- Array(W_g_pre)))
    r = d / (maximum(abs.(Array(W_g_std))) + 1e-8)
    @printf("to_bias.weight grad maxdiff: %.6f (rel: %.6f) %s\n", d, r, r < 0.01 ? "[PASS]" : "[FAIL]")
else
    println("to_bias.weight gradient: std=$(typeof(W_g_std)), pre=$(typeof(W_g_pre))")
end

# Compare output projection weight gradient (should be identical)
out_g_std = g_std[1].mha.mha.to_out.weight
out_g_pre = g_pre[1].mha.mha.to_out.weight
if !isnothing(out_g_std) && !isnothing(out_g_pre)
    d = maximum(abs.(Array(out_g_std) .- Array(out_g_pre)))
    r = d / (maximum(abs.(Array(out_g_std))) + 1e-8)
    @printf("to_out.weight grad maxdiff: %.6f (rel: %.6f) %s\n", d, r, r < 0.01 ? "[PASS]" : "[FAIL]")
end

# Compare transition weight gradient
tr_g_std = g_std[1].transition.transition.adaln.ln.scale
tr_g_pre = g_pre[1].transition.transition.adaln.ln.scale
if !isnothing(tr_g_std) && !isnothing(tr_g_pre)
    d = maximum(abs.(Array(tr_g_std) .- Array(tr_g_pre)))
    r = d / (maximum(abs.(Array(tr_g_std))) + 1e-8)
    @printf("transition adaln grad maxdiff: %.6f (rel: %.6f) %s\n", d, r, r < 0.01 ? "[PASS]" : "[FAIL]")
end

println("\nDone!")
