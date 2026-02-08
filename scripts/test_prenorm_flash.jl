#!/usr/bin/env julia
# Test pre-normalized pair path + flash attention vs standard path
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
using CUDA
using Flux
using Printf
using Statistics

LaProteina.enable_tf32!()
println("GPU: ", CUDA.name(CUDA.device()))
println("cuTile: ", LaProteina._HAS_CUTILE)

L = 128; B = 4

base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

model = BranchingScoreNetwork(base) |> gpu

seq_raw_dim = size(base.init_repr_factory.projection.weight, 2)
cond_raw_dim = size(base.cond_factory.projection.weight, 2)
pair_raw_dim = size(base.pair_rep_builder.init_repr_factory.projection.weight, 2)
pair_cond_raw_dim = size(base.pair_rep_builder.cond_factory.projection.weight, 2)

raw_features = ScoreNetworkRawFeatures(
    CUDA.randn(Float32, seq_raw_dim, L, B),
    CUDA.randn(Float32, cond_raw_dim, L, B),
    CUDA.randn(Float32, pair_raw_dim, L, L, B),
    CUDA.randn(Float32, pair_cond_raw_dim, L, L, B),
    CUDA.ones(Float32, L, B)
)

# ============================================
# Test 1: Single block comparison
# ============================================
println("\n--- Test 1: Single PBA block ---")

# Set up common inputs
mask = raw_features.mask
pair_feats = CUDA.randn(Float32, 256, L, L, B)
node_feats = CUDA.randn(Float32, 768, L, B)

pba = base.transformer_layers[1].mha.mha

# Standard path: pair_norm + to_bias, then flash attention
out_std = pba(node_feats, pair_feats, mask)

# Pre-normalized path
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair_feats; dims=1, eps=pair_eps)
out_pre = LaProteina._pair_bias_attn_prenormed(pba, node_feats, pair_normed, mask)

d = maximum(abs.(Array(out_std) .- Array(out_pre)))
@printf("  Single block PBA: max diff = %.6f\n", d)

# ============================================
# Test 2: Check _apply_pair_affine vs standard
# ============================================
println("\n--- Test 2: _apply_pair_affine correctness ---")

pair_normed_std = pba.pair_norm(pair_feats)  # cuTile fused LN: normalize + affine
bias_std = pba.to_bias(pair_normed_std)  # [h, L, L, B]

pair_normed_manual = LaProteina.pytorch_normalise(pair_feats; dims=1, eps=pair_eps)
bias_pre = LaProteina._apply_pair_affine(pair_normed_manual, pba.pair_norm, pba.to_bias)

d_bias = maximum(abs.(Array(bias_std) .- Array(bias_pre)))
@printf("  Pair bias: max diff = %.6f\n", d_bias)
@printf("  Pair bias: max |val| = %.2f\n", maximum(abs.(Array(bias_std))))

# ============================================
# Test 3: Full forward comparison
# ============================================
println("\n--- Test 3: Full forward ---")
out_std = forward_branching_from_raw_features(model, raw_features)
out_gpu = forward_branching_from_raw_features_gpu(model, raw_features)

for key in [:bb_ca, :local_latents]
    vs = Array(out_std[key][:v])
    vg = Array(out_gpu[key][:v])
    d = maximum(abs.(vs .- vg))
    r = d / (maximum(abs.(vs)) + 1e-8)
    @printf("  %-20s max diff: %.6f  max|val|: %.2f  rel: %.6f\n", key, d, maximum(abs.(vs)), r)
end
for key in [:split, :del]
    vs = Array(out_std[key])
    vg = Array(out_gpu[key])
    d = maximum(abs.(vs .- vg))
    r = d / (maximum(abs.(vs)) + 1e-8)
    @printf("  %-20s max diff: %.6f  max|val|: %.2f  rel: %.6f\n", key, d, maximum(abs.(vs)), r)
end

println("\nDone!")
