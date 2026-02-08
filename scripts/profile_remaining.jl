#!/usr/bin/env julia
# Profile remaining bottlenecks in the GPU-optimized path
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4
token_dim = 768; pair_dim = 256; n_heads = 12; dim_cond = 256

model = ScoreNetwork(
    n_layers=14, token_dim=token_dim, pair_dim=pair_dim,
    n_heads=n_heads, dim_cond=dim_cond, qk_ln=true,
    update_pair_repr=false
) |> gpu

x = CUDA.randn(Float32, token_dim, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

function bench(name, f, n_warmup=3, n_iter=10)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
end

block = model.transformer_layers[1]
pba = block.mha.mha
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

println("="^70)
println("Remaining Bottleneck Analysis")
println("="^70)

println("\n--- Prenorm block breakdown (backward) ---")

# Total block backward
bench("Prenorm block total", () -> begin
    Zygote.gradient(b -> begin
        sum(LaProteina._transformer_block_prenormed(b, x, pair, pair_normed, cond, mask))
    end, block)
end)

# MHA AdaLN
bench("  adaln(x, cond, mask)", () -> begin
    Zygote.gradient(x -> sum(block.mha.adaln(x, cond, mask)), x)
end)

# PairBiasAttn prenorm
bench("  _pair_bias_attn_prenormed", () -> begin
    Zygote.gradient(m -> sum(LaProteina._pair_bias_attn_prenormed(m, x, pair_normed, mask)), pba)
end)

# scale_output
bench("  scale_output", () -> begin
    Zygote.gradient(x -> sum(block.mha.scale_output(x, cond, mask)), x)
end)

# Transition
bench("  transition(x, cond, mask)", () -> begin
    Zygote.gradient(x -> sum(block.transition(x, cond, mask)), x)
end)

# Deeper into PairBiasAttn
println("\n--- PairBiasAttn prenorm breakdown (backward) ---")

# node_norm
bench("  node_norm", () -> begin
    Zygote.gradient(x -> sum(pba.node_norm(x)), x)
end)

# to_qkv
bench("  to_qkv", () -> begin
    xn = pba.node_norm(x)
    Zygote.gradient(m -> sum(m(xn)), pba.to_qkv)
end)

# q_norm + k_norm
qkv = pba.to_qkv(pba.node_norm(x))
inner = n_heads * 64
q_raw = qkv[1:inner, :, :]
k_raw = qkv[inner+1:2*inner, :, :]
bench("  q_norm", () -> begin
    Zygote.gradient(q -> sum(pba.q_norm(q)), q_raw)
end)
bench("  k_norm", () -> begin
    Zygote.gradient(k -> sum(pba.k_norm(k)), k_raw)
end)

# to_g
bench("  to_g", () -> begin
    xn = pba.node_norm(x)
    Zygote.gradient(m -> sum(m(xn)), pba.to_g)
end)

# permutedims (4x for q,k,v,g)
q_r = reshape(q_raw, 64, n_heads, L, B)
bench("  4x permutedims (1324)", () -> begin
    Zygote.gradient(q -> begin
        q1 = permutedims(q, (1,3,2,4))
        q2 = permutedims(q, (1,3,2,4))
        q3 = permutedims(q, (1,3,2,4))
        q4 = permutedims(q, (1,3,2,4))
        sum(q1) + sum(q2) + sum(q3) + sum(q4)
    end, q_r)
end)

# _apply_pair_affine
bench("  _apply_pair_affine", () -> begin
    Zygote.gradient(pn -> sum(LaProteina._apply_pair_affine(pn, pba.pair_norm, pba.to_bias)), pair_normed)
end)

# _attention
q_4d = CUDA.randn(Float32, 64, L, n_heads, B)
k_4d = CUDA.randn(Float32, 64, L, n_heads, B)
v_4d = CUDA.randn(Float32, 64, L, n_heads, B)
bias = CUDA.randn(Float32, n_heads, L, L, B)
bench("  _attention", () -> begin
    Zygote.gradient(q -> sum(LaProteina._attention(q, k_4d, v_4d, bias, mask, 0.125f0)), q_4d)
end)

# sigmoid gate + output permute + to_out
bench("  to_out (768→768)", () -> begin
    out = CUDA.randn(Float32, inner, L, B)
    Zygote.gradient(m -> sum(m(out)), pba.to_out)
end)

# Deeper into Transition
println("\n--- Transition breakdown (backward) ---")
bench("  adaln", () -> begin
    Zygote.gradient(x -> sum(block.transition.adaln(x, cond, mask)), x)
end)
bench("  SwiGLUTransition", () -> begin
    Zygote.gradient(x -> sum(block.transition.transition(x, mask)), x)
end)
bench("  scale_output", () -> begin
    Zygote.gradient(x -> sum(block.transition.scale_output(x, cond, mask)), x)
end)

println("\n--- Dense matmuls (backward) ---")
bench("Dense 256→768 backward", () -> begin
    c = CUDA.randn(Float32, 256, L, B)
    Zygote.gradient(m -> sum(m(c)), pba.to_g)
end)
bench("Dense 768→768 backward", () -> begin
    out = CUDA.randn(Float32, 768, L, B)
    Zygote.gradient(m -> sum(m(out)), pba.to_out)
end)
bench("Dense 768→2304 backward", () -> begin
    xn = pba.node_norm(x)
    Zygote.gradient(m -> sum(m(xn)), pba.to_qkv)
end)
bench("Dense 768→6144 backward", () -> begin
    Zygote.gradient(m -> sum(m(x)), block.transition.transition.linear_in)
end)

println("\n" * "="^70)
println("Done.")
println("="^70)
