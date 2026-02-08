#!/usr/bin/env julia
# Test flash attention + fused LayerNorm correctness
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using NNlib: softmax, batched_mul, batched_transpose
using Statistics
using Printf

LaProteina.enable_tf32!()
println("GPU: ", CUDA.name(CUDA.device()))
println("cuTile: ", LaProteina._HAS_CUTILE)

d = 64; L = 128; h = 12; B = 4

Q = CUDA.randn(Float32, d, L, h, B)
K = CUDA.randn(Float32, d, L, h, B)
V = CUDA.randn(Float32, d, L, h, B)
bias = CUDA.randn(Float32, h, L, L, B)

scale = 1f0 / sqrt(Float32(d))

# Flash attention with bias
bias_fmha = permutedims(bias, (3, 2, 1, 4))  # [L_k, L_q, h, B]
out_flash = LaProteina.flash_attention_bias(Q, K, V, bias_fmha; scale=scale)

# NNlib reference
q_flat = reshape(Q, d, L, h * B)
k_flat = reshape(K, d, L, h * B)
scores = batched_mul(batched_transpose(q_flat), k_flat)
scores = reshape(scores, L, L, h, B)
bias_perm = permutedims(bias, (2, 3, 1, 4))
scores = scores .* scale .+ bias_perm
attn_weights = softmax(scores; dims=2)
v_flat = reshape(V, d, L, h * B)
attn_flat = reshape(attn_weights, L, L, h * B)
out = batched_mul(attn_flat, batched_transpose(v_flat))
out = permutedims(out, (2, 1, 3))
out_ref = reshape(out, d, L, h, B)

d_max = maximum(abs.(Array(out_flash) .- Array(out_ref)))
d_mean = mean(abs.(Array(out_flash) .- Array(out_ref)))
@printf("Flash attention with bias:\n")
@printf("  Max abs diff:  %.6f\n", d_max)
@printf("  Mean abs diff: %.8f\n", d_mean)
@printf("  Max |out|:     %.2f\n", maximum(abs.(Array(out_ref))))
@printf("  Rel error:     %.6f\n", d_max / maximum(abs.(Array(out_ref))))

# Without bias
out_flash_nb = LaProteina.flash_attention(Q, K, V; scale=scale)
scores_nb = batched_mul(batched_transpose(q_flat), k_flat)
scores_nb = reshape(scores_nb, L, L, h, B) .* scale
attn_nb = softmax(scores_nb; dims=2)
attn_nb_flat = reshape(attn_nb, L, L, h * B)
out_nb = batched_mul(attn_nb_flat, batched_transpose(v_flat))
out_nb = permutedims(out_nb, (2, 1, 3))
out_nb_ref = reshape(out_nb, d, L, h, B)
d_max_nb = maximum(abs.(Array(out_flash_nb) .- Array(out_nb_ref)))
@printf("\nFlash attention (no bias):\n")
@printf("  Max abs diff:  %.6f\n", d_max_nb)

# Fused LayerNorm tests
println("\n--- Fused LayerNorm ---")
x256 = CUDA.randn(Float32, 256, 128, 4)
ln = PyTorchLayerNorm(256) |> gpu
y_fused = ln(x256)
y_ref = LaProteina.pytorch_normalise(x256; dims=1, eps=ln.ϵ) .* ln.scale .+ ln.bias
d_ln = maximum(abs.(Array(y_fused) .- Array(y_ref)))
@printf("  LayerNorm (256): max diff = %.8f\n", d_ln)

x768 = CUDA.randn(Float32, 768, 128, 4)
ln768 = PyTorchLayerNorm(768) |> gpu
y_fused768 = ln768(x768)
y_ref768 = LaProteina.pytorch_normalise(x768; dims=1, eps=ln768.ϵ) .* ln768.scale .+ ln768.bias
d_ln768 = maximum(abs.(Array(y_fused768) .- Array(y_ref768)))
@printf("  LayerNorm (768): max diff = %.8f\n", d_ln768)

# Test non-affine
x256na = CUDA.randn(Float32, 256, 128, 4)
ln_na = PyTorchLayerNorm(256; affine=false) |> gpu
y_na = ln_na(x256na)
y_na_ref = LaProteina.pytorch_normalise(x256na; dims=1, eps=ln_na.ϵ)
d_na = maximum(abs.(Array(y_na) .- Array(y_na_ref)))
@printf("  LayerNorm (256 non-affine): max diff = %.8f\n", d_na)

println("\nDone!")
