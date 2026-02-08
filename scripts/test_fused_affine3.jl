#!/usr/bin/env julia
# Check pair_norm.bias gradient magnitude
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf
using Statistics

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

g_std = Zygote.gradient(b -> sum(b(x, pair, cond, mask)), block)
g_pre = Zygote.gradient(b -> begin
    sum(LaProteina._transformer_block_prenormed(b, x, pair, pair_normed, cond, mask))
end, block)

bias_g_std = Array(g_std[1].mha.mha.pair_norm.bias)
bias_g_pre = Array(g_pre[1].mha.mha.pair_norm.bias)

@printf("pair_norm.bias grad:\n")
@printf("  Standard: mean=%.8f, std=%.8f, max=%.8f\n", mean(bias_g_std), std(bias_g_std), maximum(abs.(bias_g_std)))
@printf("  Prenorm:  mean=%.8f, std=%.8f, max=%.8f\n", mean(bias_g_pre), std(bias_g_pre), maximum(abs.(bias_g_pre)))
@printf("  Maxdiff:  %.8f\n", maximum(abs.(bias_g_std .- bias_g_pre)))

# The gradient magnitudes
scale_g_std = Array(g_std[1].mha.mha.pair_norm.scale)
scale_g_pre = Array(g_pre[1].mha.mha.pair_norm.scale)
@printf("\npair_norm.scale grad:\n")
@printf("  Standard: mean=%.6f, max=%.6f\n", mean(scale_g_std), maximum(abs.(scale_g_std)))
@printf("  Prenorm:  mean=%.6f, max=%.6f\n", mean(scale_g_pre), maximum(abs.(scale_g_pre)))
@printf("  Maxdiff:  %.6f\n", maximum(abs.(scale_g_std .- scale_g_pre)))

# Now test with larger dimensions to be sure
println("\n--- Full size test (L=64, B=2) ---")
L2 = 64; B2 = 2
x2 = CUDA.randn(Float32, 768, L2, B2)
pair2 = CUDA.randn(Float32, pair_dim, L2, L2, B2)
cond2 = CUDA.randn(Float32, 256, L2, B2)
mask2 = CUDA.ones(Float32, L2, B2)
pair_normed2 = LaProteina.pytorch_normalise(pair2; dims=1, eps=pair_eps)

g_std2 = Zygote.gradient(b -> sum(b(x2, pair2, cond2, mask2)), block)
g_pre2 = Zygote.gradient(b -> begin
    sum(LaProteina._transformer_block_prenormed(b, x2, pair2, pair_normed2, cond2, mask2))
end, block)

for (name, getter) in [
    ("pair_norm.scale", g -> g.mha.mha.pair_norm.scale),
    ("pair_norm.bias", g -> g.mha.mha.pair_norm.bias),
    ("to_bias.weight", g -> g.mha.mha.to_bias.weight),
    ("to_out.weight", g -> g.mha.mha.to_out.weight),
    ("to_qkv.weight", g -> g.mha.mha.to_qkv.weight),
]
    v_std = getter(g_std2[1])
    v_pre = getter(g_pre2[1])
    if !isnothing(v_std) && !isnothing(v_pre)
        d = maximum(abs.(Array(v_std) .- Array(v_pre)))
        m = maximum(abs.(Array(v_std)))
        @printf("  %-25s maxdiff=%.6f  max_std=%.6f  rel=%.6f %s\n",
            name, d, m, d/(m+1e-8), d/(m+1e-8) < 0.01 ? "[PASS]" : "[FAIL]")
    end
end

println("\nDone!")
