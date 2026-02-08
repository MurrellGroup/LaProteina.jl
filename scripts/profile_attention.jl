#!/usr/bin/env julia
# Profile individual operations within _attention to identify bottlenecks.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina, CUDA, Flux, Flux.Zygote, Printf, NNlib

LaProteina.enable_tf32!()

L = 128; B = 4; d = 64; h = 12; pair_dim = 256; token_dim = 768

function bench_op(name, f, n_warmup=5, n_iter=50)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for _ in 1:n_iter; f(); end
    end
    @printf("  %-45s %.3f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("Profile: Inner _attention breakdown (L=$L, B=$B, h=$h, d=$d)")
println("="^70)

# Setup
q = CUDA.randn(Float32, d, L, h, B)
k = CUDA.randn(Float32, d, L, h, B)
v = CUDA.randn(Float32, d, L, h, B)
q_flat = reshape(q, d, L, h*B)
k_flat = reshape(k, d, L, h*B)
v_flat = reshape(v, d, L, h*B)

println("\n--- Forward path ops ---")
bench_op("permutedims q_flat (2,1,3)", () -> permutedims(q_flat, (2,1,3)))
bench_op("batched_transpose q_flat", () -> NNlib.batched_transpose(q_flat))

bench_op("batched_mul(q^T, k) permutedims", () -> batched_mul(permutedims(q_flat, (2,1,3)), k_flat))
bench_op("batched_mul(q^T, k) batched_transpose", () -> batched_mul(NNlib.batched_transpose(q_flat), k_flat))

scores = batched_mul(NNlib.batched_transpose(q_flat), k_flat)
bench_op("scores .* scale", () -> scores .* 0.125f0)

scores4 = CUDA.randn(Float32, L, L, h, B)
bench_op("permutedims bias (2,3,1,4)", () -> permutedims(CUDA.randn(Float32, h, L, L, B), (2,3,1,4)))

scores3 = reshape(scores4, L, L, h*B)
bench_op("softmax (L×L, dim=1)", () -> softmax(scores3; dims=1))

attn_flat = CUDA.randn(Float32, L, L, h*B)
bench_op("batched_mul(attn, v^T) permutedims", () -> batched_mul(attn_flat, permutedims(v_flat, (2,1,3))))
bench_op("batched_mul(attn, v^T) batched_transpose", () -> batched_mul(attn_flat, NNlib.batched_transpose(v_flat)))

out3 = CUDA.randn(Float32, L, d, h*B)
bench_op("permutedims output (2,1,3)", () -> permutedims(out3, (2,1,3)))
bench_op("batched_transpose output", () -> NNlib.batched_transpose(out3))

println("\n--- Full forward comparison ---")
attn_layer = PairBiasAttention(token_dim, h; pair_dim=pair_dim, qk_ln=true) |> gpu
x = CUDA.randn(Float32, token_dim, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
mask = CUDA.ones(Float32, L, B)

bench_op("Full PairBiasAttention forward", () -> attn_layer(x, pair, mask), 3, 20)

println("\n--- QKV projection vs attention ---")
bench_op("node_norm", () -> attn_layer.node_norm(x))
normed = attn_layer.node_norm(x)
bench_op("to_qkv (Dense 768->2304)", () -> attn_layer.to_qkv(normed))
bench_op("to_g (Dense 768->768)", () -> attn_layer.to_g(normed))

qkv = attn_layer.to_qkv(normed)
inner_dim = h * d
q2 = qkv[1:inner_dim, :, :]
k2 = qkv[inner_dim+1:2*inner_dim, :, :]

bench_op("q_norm", () -> attn_layer.q_norm(q2))
bench_op("k_norm", () -> attn_layer.k_norm(k2))

println("\n--- Pair bias computation ---")
bench_op("pair_norm (256, L×L×B)", () -> attn_layer.pair_norm(pair))
pair_normed = attn_layer.pair_norm(pair)
bench_op("to_bias (Dense 256->12)", () -> attn_layer.to_bias(pair_normed))

println("\n--- 4D permutedims (reshape+permute path) ---")
q4d = CUDA.randn(Float32, d, h, L, B)
bench_op("permutedims (d,h,L,B) -> (d,L,h,B)", () -> permutedims(q4d, (1,3,2,4)))

println("\n--- Full backward ---")
x2 = CUDA.randn(Float32, token_dim, L, B)
bench_op("Full backward (gradient)", () -> begin
    Zygote.gradient(x -> sum(attn_layer(x, pair, mask)), x2)
    nothing
end, 2, 5)

println("\n--- NNlib dot_product_attention ---")
# Compare with NNlib's implementation
# NNlib expects (features, seq_len, batch) and handles head split internally
q_nnlib = CUDA.randn(Float32, token_dim, L, B)
k_nnlib = CUDA.randn(Float32, token_dim, L, B)
v_nnlib = CUDA.randn(Float32, token_dim, L, B)
bench_op("NNlib dot_product_attention (nheads=$h)", () -> begin
    NNlib.dot_product_attention(q_nnlib, k_nnlib, v_nnlib; nheads=h)
    nothing
end, 3, 20)

# With bias
bias_nnlib = CUDA.randn(Float32, L, L, h, B)
bench_op("NNlib dot_product_attention + bias", () -> begin
    NNlib.dot_product_attention(q_nnlib, k_nnlib, v_nnlib, bias_nnlib; nheads=h)
    nothing
end, 3, 20)

println("\n" * "="^70)
println("Profile complete.")
println("="^70)
