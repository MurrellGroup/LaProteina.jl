#!/usr/bin/env julia
# Debug: exactly where does PBA standard vs prenormed diverge?
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Printf
using NNlib: sigmoid
using Statistics

LaProteina.enable_tf32!()
println("cuTile: ", LaProteina._HAS_CUTILE)

L = 128; B = 4

model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

mask = CUDA.ones(Float32, L, B)
pair_feats = CUDA.randn(Float32, 256, L, L, B)
node_feats = CUDA.randn(Float32, 768, L, B)

m = model.transformer_layers[1].mha.mha  # PairBiasAttention

D = 768; h = m.n_heads; d = m.dim_head; inner_dim = h * d
pair_eps = m.pair_norm.ϵ

# Step 1: Check that both paths get same q/k/v
node_normed = m.node_norm(node_feats)
qkv = m.to_qkv(node_normed)
q = qkv[1:inner_dim, :, :]
k = qkv[inner_dim+1:2*inner_dim, :, :]
v = qkv[2*inner_dim+1:end, :, :]
q = m.q_norm(q)
k = m.k_norm(k)
g = m.to_g(node_normed)

# Standard path: pair_norm + to_bias
pair_normed_std = m.pair_norm(pair_feats)
bias_std = m.to_bias(pair_normed_std)  # [h, L, L, B]

# Pre-normalized path
pair_normed_manual = LaProteina.pytorch_normalise(pair_feats; dims=1, eps=pair_eps)
bias_pre = LaProteina._apply_pair_affine(pair_normed_manual, m.pair_norm, m.to_bias)

@printf("Bias diff: %.6f\n", maximum(abs.(Array(bias_std) .- Array(bias_pre))))

# Step 2: Both use same q/k/v, only bias differs slightly
q4 = permutedims(reshape(q, d, h, L, B), (1, 3, 2, 4))  # [d, L, h, B]
k4 = permutedims(reshape(k, d, h, L, B), (1, 3, 2, 4))
v4 = permutedims(reshape(v, d, h, L, B), (1, 3, 2, 4))
g4 = permutedims(reshape(g, d, h, L, B), (1, 3, 2, 4))

# Run flash attention with standard bias
bias_std_fmha = permutedims(bias_std, (3, 2, 1, 4))  # [L_k, L_q, h, B]
mask_bias = reshape((1.0f0 .- mask) .* (-1.0f10), L, 1, 1, B)
bias_std_fmha = bias_std_fmha .+ mask_bias
out_std = LaProteina.flash_attention_bias(q4, k4, v4, bias_std_fmha; scale=m.scale)

# Run flash attention with prenormed bias
bias_pre_fmha = permutedims(bias_pre, (3, 2, 1, 4))
bias_pre_fmha = bias_pre_fmha .+ mask_bias
out_pre = LaProteina.flash_attention_bias(q4, k4, v4, bias_pre_fmha; scale=m.scale)

@printf("Flash attn output diff: %.6f\n", maximum(abs.(Array(out_std) .- Array(out_pre))))

# Apply gating
gated_std = sigmoid.(g4) .* out_std
gated_pre = sigmoid.(g4) .* out_pre
@printf("After gating diff: %.6f\n", maximum(abs.(Array(gated_std) .- Array(gated_pre))))

# Permute back and project
gated_std_flat = reshape(permutedims(gated_std, (1, 3, 2, 4)), inner_dim, L, B)
gated_pre_flat = reshape(permutedims(gated_pre, (1, 3, 2, 4)), inner_dim, L, B)
out_final_std = m.to_out(gated_std_flat)
out_final_pre = m.to_out(gated_pre_flat)
@printf("Final output diff: %.6f\n", maximum(abs.(Array(out_final_std) .- Array(out_final_pre))))

# Now compare against actual PBA call
println("\n--- Now compare against actual PBA(x, pair, mask) call ---")
out_pba = m(node_feats, pair_feats, mask)
out_prenormed = LaProteina._pair_bias_attn_prenormed(m, node_feats, pair_normed_manual, mask)

@printf("PBA standard vs manual reproduction: %.6f\n",
    maximum(abs.(Array(out_pba) .- Array(out_final_std))))
@printf("Prenormed vs manual reproduction: %.6f\n",
    maximum(abs.(Array(out_prenormed) .- Array(out_final_pre))))
@printf("PBA standard vs prenormed: %.6f\n",
    maximum(abs.(Array(out_pba) .- Array(out_prenormed))))

println("\nDone!")
