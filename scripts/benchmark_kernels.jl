#!/usr/bin/env julia
# Benchmark: cuTile kernels vs NNlib equivalents (isolated)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using NNlib: softmax, batched_mul, batched_transpose
using Printf
using Statistics

LaProteina.enable_tf32!()
println("GPU: ", CUDA.name(CUDA.device()))
println("cuTile: ", LaProteina._HAS_CUTILE)

d = 64; L = 128; h = 12; B = 4

# ============================================
# 1. Flash Attention vs NNlib attention
# ============================================
println("\n--- Attention: Flash vs NNlib (d=$d, L=$L, h=$h, B=$B) ---")

Q = CUDA.randn(Float32, d, L, h, B)
K = CUDA.randn(Float32, d, L, h, B)
V = CUDA.randn(Float32, d, L, h, B)
bias = CUDA.randn(Float32, L, L, h, B)  # [L_k, L_q, h, B] (FMHA layout)
scale = 1f0 / sqrt(Float32(d))

# Flash attention
for _ in 1:5
    LaProteina.flash_attention_bias(Q, K, V, bias; scale=scale)
    CUDA.synchronize()
end
t_flash = Float64[]
for _ in 1:100
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        LaProteina.flash_attention_bias(Q, K, V, bias; scale=scale)
    end
    push!(t_flash, t)
end
@printf("  Flash attn:  mean=%.2fms  median=%.2fms  min=%.2fms\n",
    mean(t_flash)*1000, median(t_flash)*1000, minimum(t_flash)*1000)

# NNlib attention (batched_mul path)
q_flat = reshape(Q, d, L, h * B)
k_flat = reshape(K, d, L, h * B)
v_flat = reshape(V, d, L, h * B)
bias_perm = permutedims(bias, (2, 1, 3, 4))  # [L_q, L_k, h, B]
bias_flat = reshape(bias_perm, L, L, h * B)

for _ in 1:5
    scores = batched_mul(batched_transpose(q_flat), k_flat)
    scores = scores .* scale .+ bias_flat
    attn = softmax(scores; dims=2)
    out = batched_mul(attn, batched_transpose(v_flat))
    CUDA.synchronize()
end
t_nnlib = Float64[]
for _ in 1:100
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        scores = batched_mul(batched_transpose(q_flat), k_flat)
        scores = scores .* scale .+ bias_flat
        attn = softmax(scores; dims=2)
        out = batched_mul(attn, batched_transpose(v_flat))
    end
    push!(t_nnlib, t)
end
@printf("  NNlib attn:  mean=%.2fms  median=%.2fms  min=%.2fms\n",
    mean(t_nnlib)*1000, median(t_nnlib)*1000, minimum(t_nnlib)*1000)
@printf("  Speedup: %.2fx\n", median(t_nnlib) / median(t_flash))

# ============================================
# 2. Fused LayerNorm vs standard
# ============================================
println("\n--- LayerNorm: Fused vs Standard ---")

for (C, label) in [(256, "pair_dim=256"), (768, "token_dim=768")]
    x = CUDA.randn(Float32, C, L, B)
    ln = PyTorchLayerNorm(C) |> gpu

    # The CuArray override dispatches to fused kernel if cuTile
    for _ in 1:5
        ln(x)
        CUDA.synchronize()
    end
    t_fused = Float64[]
    for _ in 1:200
        CUDA.synchronize()
        t = CUDA.@elapsed ln(x)
        push!(t_fused, t)
    end

    # Manual standard path
    for _ in 1:5
        y = LaProteina.pytorch_normalise(x; dims=1, eps=ln.ϵ) .* ln.scale .+ ln.bias
        CUDA.synchronize()
    end
    t_std = Float64[]
    for _ in 1:200
        CUDA.synchronize()
        t = CUDA.@elapsed begin
            LaProteina.pytorch_normalise(x; dims=1, eps=ln.ϵ) .* ln.scale .+ ln.bias
        end
        push!(t_std, t)
    end

    @printf("  %s: fused=%.3fms  std=%.3fms  speedup=%.1fx\n",
        label, median(t_fused)*1000, median(t_std)*1000, median(t_std)/median(t_fused))
end

# 4D pair tensor
x4d = CUDA.randn(Float32, 256, L, L, B)
ln4d = PyTorchLayerNorm(256) |> gpu
for _ in 1:5; ln4d(x4d); CUDA.synchronize(); end
t_4d_fused = Float64[]
for _ in 1:50
    CUDA.synchronize()
    t = CUDA.@elapsed ln4d(x4d)
    push!(t_4d_fused, t)
end
for _ in 1:5
    LaProteina.pytorch_normalise(x4d; dims=1, eps=ln4d.ϵ) .* ln4d.scale .+ ln4d.bias
    CUDA.synchronize()
end
t_4d_std = Float64[]
for _ in 1:50
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        LaProteina.pytorch_normalise(x4d; dims=1, eps=ln4d.ϵ) .* ln4d.scale .+ ln4d.bias
    end
    push!(t_4d_std, t)
end
@printf("  pair[256,L,L,B]: fused=%.2fms  std=%.2fms  speedup=%.1fx\n",
    median(t_4d_fused)*1000, median(t_4d_std)*1000, median(t_4d_std)/median(t_4d_fused))

# ============================================
# 3. Single TransformerBlock comparison
# ============================================
println("\n--- Single TransformerBlock ---")

model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

block = model.transformer_layers[1]
node_feats = CUDA.randn(Float32, 768, L, B)
pair_feats = CUDA.randn(Float32, 256, L, L, B)
cond = CUDA.randn(Float32, 256, L, B)
mask = CUDA.ones(Float32, L, B)

# Standard block call (uses CuArray override with cuTile flash attn)
for _ in 1:5
    block(node_feats, pair_feats, cond, mask)
    CUDA.synchronize()
end
t_block = Float64[]
for _ in 1:50
    CUDA.synchronize()
    t = CUDA.@elapsed block(node_feats, pair_feats, cond, mask)
    push!(t_block, t)
end
@printf("  Block (cuTile):  mean=%.2fms  median=%.2fms  min=%.2fms\n",
    mean(t_block)*1000, median(t_block)*1000, minimum(t_block)*1000)

# Pre-normalized block
pba = block.mha.mha
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair_feats; dims=1, eps=pair_eps)

for _ in 1:5
    LaProteina._transformer_block_prenormed(block, node_feats, pair_feats, pair_normed, cond, mask)
    CUDA.synchronize()
end
t_prenormed = Float64[]
for _ in 1:50
    CUDA.synchronize()
    t = CUDA.@elapsed LaProteina._transformer_block_prenormed(block, node_feats, pair_feats, pair_normed, cond, mask)
    push!(t_prenormed, t)
end
@printf("  Block (prenormed): mean=%.2fms  median=%.2fms  min=%.2fms\n",
    mean(t_prenormed)*1000, median(t_prenormed)*1000, minimum(t_prenormed)*1000)

@printf("  × 14 blocks: cuTile=%.1fms  prenormed=%.1fms\n",
    median(t_block)*1000*14, median(t_prenormed)*1000*14)

println("\nDone!")
