#!/usr/bin/env julia
# Full model gradient comparison: standard vs GPU-optimized.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf
using Statistics

LaProteina.enable_tf32!()

L = 64; B = 2
token_dim = 768; pair_dim = 256; n_heads = 12; dim_cond = 256

model = ScoreNetwork(
    n_layers=14, token_dim=token_dim, pair_dim=pair_dim,
    n_heads=n_heads, dim_cond=dim_cond, qk_ln=true,
    update_pair_repr=false
) |> gpu

seq_raw_dim = size(model.init_repr_factory.projection.weight, 2)
cond_raw_dim = size(model.cond_factory.projection.weight, 2)
pair_raw_dim = size(model.pair_rep_builder.init_repr_factory.projection.weight, 2)
pair_cond_raw_dim = size(model.pair_rep_builder.cond_factory.projection.weight, 2)

raw_features = ScoreNetworkRawFeatures(
    CUDA.randn(Float32, seq_raw_dim, L, B),
    CUDA.randn(Float32, cond_raw_dim, L, B),
    CUDA.randn(Float32, pair_raw_dim, L, L, B),
    CUDA.randn(Float32, pair_cond_raw_dim, L, L, B),
    CUDA.ones(Float32, L, B)
)

function loss_std(m)
    out = forward_from_raw_features(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
end

function loss_gpu(m)
    out = forward_from_raw_features_gpu(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
end

println("Computing standard gradient...")
g_std = Zygote.gradient(loss_std, model)
println("Computing GPU-optimized gradient...")
g_gpu = Zygote.gradient(loss_gpu, model)

# Compare gradients for key parameters across all layers
println("\n" * "="^70)
println("Full Model Gradient Comparison (14 layers, L=$L, B=$B)")
println("="^70)

for i in 1:14
    block_std = g_std[1].transformer_layers[i]
    block_gpu = g_gpu[1].transformer_layers[i]

    params_to_check = [
        ("layer $i pair_norm.scale", block_std.mha.mha.pair_norm.scale, block_gpu.mha.mha.pair_norm.scale),
        ("layer $i pair_norm.bias", block_std.mha.mha.pair_norm.bias, block_gpu.mha.mha.pair_norm.bias),
        ("layer $i to_bias.weight", block_std.mha.mha.to_bias.weight, block_gpu.mha.mha.to_bias.weight),
        ("layer $i to_qkv.weight", block_std.mha.mha.to_qkv.weight, block_gpu.mha.mha.to_qkv.weight),
        ("layer $i to_out.weight", block_std.mha.mha.to_out.weight, block_gpu.mha.mha.to_out.weight),
    ]

    for (name, v_std, v_gpu) in params_to_check
        if isnothing(v_std) || isnothing(v_gpu)
            @printf("  %-35s SKIP (nil)\n", name)
            continue
        end
        a = Array(v_std)
        b = Array(v_gpu)
        d = maximum(abs.(a .- b))
        m = maximum(abs.(a))
        rel = d / (m + 1e-8)
        pass = rel < 0.02  # 2% relative tolerance for TF32
        @printf("  %-35s maxdiff=%.4f  max=%.4f  rel=%.4f %s\n",
            name, d, m, rel, pass ? "✓" : "✗")
    end
end

# Check output projections
for (name, getter) in [
    ("ca_proj.ln.scale", g -> g.ca_proj.ln.scale),
    ("ca_proj.dense.weight", g -> g.ca_proj.dense.weight),
    ("local_latents_proj.ln.scale", g -> g.local_latents_proj.ln.scale),
    ("local_latents_proj.dense.weight", g -> g.local_latents_proj.dense.weight),
]
    v_std = getter(g_std[1])
    v_gpu = getter(g_gpu[1])
    if !isnothing(v_std) && !isnothing(v_gpu)
        a = Array(v_std)
        b = Array(v_gpu)
        d = maximum(abs.(a .- b))
        m = maximum(abs.(a))
        rel = d / (m + 1e-8)
        @printf("  %-35s maxdiff=%.4f  max=%.4f  rel=%.4f %s\n",
            name, d, m, rel, rel < 0.02 ? "✓" : "✗")
    end
end

println("\n" * "="^70)
println("Full model gradient comparison complete.")
println("="^70)
