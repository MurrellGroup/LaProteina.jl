#!/usr/bin/env julia
# Debug: is the PBA inference vs training path the issue?
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4

model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

mask = CUDA.ones(Float32, L, B)
pair_feats = CUDA.randn(Float32, 256, L, L, B)
node_feats = CUDA.randn(Float32, 768, L, B)

m = model.transformer_layers[1].mha.mha

# Call PBA in inference mode (within_gradient=false)
out_infer = m(node_feats, pair_feats, mask)

# Force training mode by wrapping in gradient
out_train = nothing
Zygote.gradient(node_feats) do x
    global out_train = m(x, pair_feats, mask)
    sum(out_train)
end

@printf("PBA inference vs training: max diff = %.6f\n",
    maximum(abs.(Array(out_infer) .- Array(out_train))))

# Also test: call inference twice
out_infer2 = m(node_feats, pair_feats, mask)
@printf("PBA inference call 1 vs 2: max diff = %.6f\n",
    maximum(abs.(Array(out_infer) .- Array(out_infer2))))

# Manual prenormed path (known correct)
pair_eps = m.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair_feats; dims=1, eps=pair_eps)
out_prenormed = LaProteina._pair_bias_attn_prenormed(m, node_feats, pair_normed, mask)

@printf("Prenormed vs inference: max diff = %.6f\n",
    maximum(abs.(Array(out_prenormed) .- Array(out_infer))))
@printf("Prenormed vs training: max diff = %.6f\n",
    maximum(abs.(Array(out_prenormed) .- Array(out_train))))

println("\nDone!")
