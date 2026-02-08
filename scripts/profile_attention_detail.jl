#!/usr/bin/env julia
# Profile _attention sub-operations
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using NNlib: batched_mul, batched_transpose, softmax
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4; h = 12; d = 64

q = CUDA.randn(Float32, d, L, h, B)
k = CUDA.randn(Float32, d, L, h, B)
v = CUDA.randn(Float32, d, L, h, B)
pair_bias = CUDA.randn(Float32, h, L, L, B)
mask = CUDA.ones(Float32, L, B)

function bench(name, f, n_warmup=5, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("_attention Sub-Operation Profiling")
println("="^70)

println("\n--- Forward ---")

q_flat = reshape(q, d, L, h*B)
k_flat = reshape(k, d, L, h*B)
bench("batched_mul (Q^T @ K)", () -> batched_mul(batched_transpose(q_flat), k_flat))

scores = reshape(batched_mul(batched_transpose(q_flat), k_flat), L, L, h, B)
bench("permutedims bias (h,Lq,Lk,B)->(Lq,Lk,h,B)", () -> permutedims(pair_bias, (2,3,1,4)))

bias_perm = permutedims(pair_bias, (2,3,1,4))
bench("scale + bias broadcast", () -> scores .* 0.125f0 .+ bias_perm)
bench("softmax", () -> softmax(scores; dims=2))

attn = softmax(scores; dims=2)
attn_flat = reshape(attn, L, L, h*B)
v_flat = reshape(v, d, L, h*B)
bench("batched_mul (attn @ V^T)", () -> batched_mul(attn_flat, batched_transpose(v_flat)))

out = batched_mul(attn_flat, batched_transpose(v_flat))
bench("permutedims output (Lq,d,hB)->(d,Lq,hB)", () -> permutedims(out, (2,1,3)))

bench("Full _attention", () -> LaProteina._attention(q, k, v, pair_bias, mask, 0.125f0))

println("\n--- Backward ---")
bench("Full _attention backward", () -> begin
    Zygote.gradient(q -> sum(LaProteina._attention(q, k, v, pair_bias, mask, 0.125f0)), q)
end)

bench("batched_mul backward", () -> begin
    Zygote.gradient(q -> begin
        qf = reshape(q, d, L, h*B)
        sum(batched_mul(batched_transpose(qf), k_flat))
    end, q)
end)

bench("permutedims bias backward", () -> begin
    Zygote.gradient(b -> sum(permutedims(b, (2,3,1,4))), pair_bias)
end)

bench("softmax backward", () -> begin
    Zygote.gradient(s -> sum(softmax(s; dims=2)), scores)
end)

# Test: what if bias was already in [Lq, Lk, h, B] format?
bias_already_permuted = permutedims(pair_bias, (2,3,1,4))
function attention_no_bias_perm(q, k, v, bias_perm, mask, scale)
    d, L, h, B = size(q)
    q_flat = reshape(q, d, L, h * B)
    k_flat = reshape(k, d, L, h * B)
    scores = batched_mul(batched_transpose(q_flat), k_flat)
    scores = reshape(scores, L, L, h, B)
    scores = scores .* scale .+ bias_perm
    attn_mask = reshape(mask, 1, L, 1, B)
    scores = scores .+ (1.0f0 .- attn_mask) .* Float32(-1.0f10)
    attn_weights = softmax(scores; dims=2)
    attn_flat = reshape(attn_weights, L, L, h * B)
    v_flat = reshape(v, d, L, h * B)
    out = batched_mul(attn_flat, batched_transpose(v_flat))
    out = permutedims(out, (2, 1, 3))
    return reshape(out, d, L, h, B)
end

bench("_attention (bias pre-permuted)", () -> attention_no_bias_perm(q, k, v, bias_already_permuted, mask, 0.125f0))
bench("_attention backward (bias pre-permuted)", () -> begin
    Zygote.gradient(q -> sum(attention_no_bias_perm(q, k, v, bias_already_permuted, mask, 0.125f0)), q)
end)

println("\nDone!")
