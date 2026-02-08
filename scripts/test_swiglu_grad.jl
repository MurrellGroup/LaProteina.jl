#!/usr/bin/env julia
# Quick full-model gradient correctness check
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 64; B = 2

model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
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

println("Computing gradients...")
g_std = Zygote.gradient(m -> begin
    out = forward_from_raw_features(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
end, model)

g_gpu = Zygote.gradient(m -> begin
    out = forward_from_raw_features_gpu(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
end, model)

# Check a few key parameters from each layer
max_rel_err = 0.0
for i in 1:14
    bs = g_std[1].transformer_layers[i]
    bg = g_gpu[1].transformer_layers[i]
    for (name, gs, gg) in [
        ("L$i to_qkv.W", bs.mha.mha.to_qkv.weight, bg.mha.mha.to_qkv.weight),
        ("L$i to_out.W", bs.mha.mha.to_out.weight, bg.mha.mha.to_out.weight),
        ("L$i pair_norm.s", bs.mha.mha.pair_norm.scale, bg.mha.mha.pair_norm.scale),
    ]
        if isnothing(gs) || isnothing(gg); continue; end
        a, b = Array(gs), Array(gg)
        d = maximum(abs.(a .- b))
        m_val = maximum(abs.(a))
        rel = d / (m_val + 1e-8)
        global max_rel_err = max(max_rel_err, rel)
        status = rel < 0.02 ? "✓" : "✗"
        if rel >= 0.005
            @printf("  %-25s rel=%.4f %s\n", name, rel, status)
        end
    end
end

@printf("\nMax relative error across all checked params: %.6f %s\n",
    max_rel_err, max_rel_err < 0.02 ? "[PASS]" : "[FAIL]")

println("Done!")
