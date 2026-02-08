#!/usr/bin/env julia
# Profile TransformerBlock forward breakdown.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina, CUDA, Flux, Printf

LaProteina.enable_tf32!()

L = 128; B = 4; h = 12; d = 64; pair_dim = 256; token_dim = 768; dim_cond = 256

function bench(name, f, n_warmup=5, n_iter=50)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-50s %.3f ms\n", name, t / n_iter * 1000)
end

x = CUDA.randn(Float32, token_dim, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

block = TransformerBlock(dim_token=token_dim, dim_pair=pair_dim, n_heads=h, dim_cond=dim_cond, qk_ln=true) |> gpu

println("="^70)
println("TransformerBlock Forward Breakdown")
println("="^70)

println("\n--- Full block ---")
bench("TransformerBlock forward", () -> block(x, pair, cond, mask))

println("\n--- MHA sub-components ---")
mha = block.mha

# AdaLN
bench("mha.adaln", () -> mha.adaln(x, cond, mask))
x_normed = mha.adaln(x, cond, mask)

# PairBiasAttention (core)
bench("mha.mha (PairBiasAttention)", () -> mha.mha(x_normed, pair, mask))

# AdaptiveOutputScale
attn_out = mha.mha(x_normed, pair, mask)
bench("mha.scale_output", () -> mha.scale_output(attn_out, cond, mask))

println("\n--- PairBiasAttention sub-components ---")
pba = mha.mha

bench("  node_norm", () -> pba.node_norm(x_normed))
normed = pba.node_norm(x_normed)

bench("  to_qkv", () -> pba.to_qkv(normed))
bench("  to_g", () -> pba.to_g(normed))

qkv = pba.to_qkv(normed)
inner_dim = h * d
q = qkv[1:inner_dim, :, :]
k = qkv[inner_dim+1:2*inner_dim, :, :]

bench("  q_norm", () -> pba.q_norm(q))
bench("  k_norm", () -> pba.k_norm(k))

bench("  pair_norm", () -> pba.pair_norm(pair))
pair_normed = pba.pair_norm(pair)
bench("  to_bias", () -> pba.to_bias(pair_normed))

println("\n--- Transition sub-components ---")
trans = block.transition

bench("transition.adaln", () -> trans.adaln(x, cond, mask))
bench("transition.transition", () -> trans.transition(x, mask))
bench("transition.scale_output", () -> trans.scale_output(x, cond, mask))

println("\n--- Memory usage ---")
println("  pair tensor: $(prod(size(pair)) * 4 / 1e6) MB")
println("  node tensor: $(prod(size(x)) * 4 / 1e6) MB")

println("\n" * "="^70)
