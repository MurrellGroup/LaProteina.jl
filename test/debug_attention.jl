#!/usr/bin/env julia
"""
Debug attention layer step by step.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using NPZ
using Flux
using LinearAlgebra
using Statistics

# Paths
test_dir = @__DIR__
weights_path = joinpath(@__DIR__, "..", "weights", "score_network.npz")

# Load inputs
x_t_bb_ca_py = npzread(joinpath(test_dir, "full_model_x_t_bb_ca.npy"))
x_t_local_latents_py = npzread(joinpath(test_dir, "full_model_x_t_local_latents.npy"))
x_sc_bb_ca_py = npzread(joinpath(test_dir, "full_model_x_sc_bb_ca.npy"))
x_sc_local_latents_py = npzread(joinpath(test_dir, "full_model_x_sc_local_latents.npy"))
t_val = npzread(joinpath(test_dir, "full_model_t_val.npy"))[1]
mask_py = npzread(joinpath(test_dir, "full_model_mask.npy"))

# Convert to Julia format [D, L, B]
x_t_bb_ca = python_to_julia(x_t_bb_ca_py)
x_t_local_latents = python_to_julia(x_t_local_latents_py)
x_sc_bb_ca = python_to_julia(x_sc_bb_ca_py)
x_sc_local_latents = python_to_julia(x_sc_local_latents_py)
mask = python_to_julia_mask(mask_py)

B = size(x_t_bb_ca, 3)
L = size(x_t_bb_ca, 2)

# Create model
model = ScoreNetwork(
    n_layers=14,
    token_dim=768,
    n_heads=12,
    latent_dim=8,
    dim_cond=256,
    t_emb_dim=256,
    pair_dim=256
)

# Load weights
load_score_network_weights!(model, weights_path)

# Build batch
batch = Dict{Symbol, Any}(
    :x_t => Dict{Symbol, Any}(
        :bb_ca => x_t_bb_ca,
        :local_latents => x_t_local_latents
    ),
    :x_sc => Dict{Symbol, Any}(
        :bb_ca => x_sc_bb_ca,
        :local_latents => x_sc_local_latents
    ),
    :t => Dict{Symbol, Any}(
        :bb_ca => fill(Float32(t_val), B),
        :local_latents => fill(Float32(t_val), B)
    ),
    :mask => mask
)

# Get initial representations
seqs = model.init_repr_factory(batch)
cond = model.cond_factory(batch)
pair = model.pair_rep_builder(batch)

mask_exp = reshape(mask, 1, L, B)
seqs = seqs .* mask_exp

# Conditioning transitions
cond = model.transition_c_1(cond, mask)
cond = model.transition_c_2(cond, mask)

println("=== INPUTS TO LAYER 0 ===")
println("seqs: ", size(seqs), " sample: ", seqs[1:5, 1, 1])
println("cond: ", size(cond), " sample: ", cond[1:5, 1, 1])
println("pair: ", size(pair), " sample: ", pair[1:5, 1, 1, 1])

# Get layer 0 components
layer0 = model.transformer_layers[1]
mha = layer0.mha  # MultiHeadBiasedAttentionADALN
adaln = mha.adaln
attention = mha.mha  # PairBiasAttention
scale_output = mha.scale_output

println("\n=== ADALN STEP ===")
# x input to adaln
x = seqs .* mask_exp
println("x before adaln: ", size(x), " sample: ", x[1:5, 1, 1])

# adaln does: x_normed * gamma(cond) + beta(cond)
x_adaln = adaln(x, cond, mask)
println("x after adaln: ", size(x_adaln), " sample: ", x_adaln[1:5, 1, 1])
println("x after adaln stats: min=", minimum(x_adaln), " max=", maximum(x_adaln), " mean=", mean(x_adaln))

println("\n=== ATTENTION STEP ===")
# attention: x -> qkv
println("Attention input x_adaln: ", size(x_adaln))
println("Attention pair: ", size(pair))

# Try calling attention directly
x_attn = attention(x_adaln, pair, mask)
println("x after attention: ", size(x_attn), " sample: ", x_attn[1:5, 1, 1])
println("x after attention stats: min=", minimum(x_attn), " max=", maximum(x_attn), " mean=", mean(x_attn))

println("\n=== SCALE OUTPUT STEP ===")
x_scaled = scale_output(x_attn, cond, mask)
println("x after scale_output: ", size(x_scaled), " sample: ", x_scaled[1:5, 1, 1])
println("x after scale_output stats: min=", minimum(x_scaled), " max=", maximum(x_scaled), " mean=", mean(x_scaled))

println("\n=== FULL MHA STEP ===")
x_mha = mha(seqs, pair, cond, mask)
println("x after full mha: ", size(x_mha), " sample: ", x_mha[1:5, 1, 1])
println("x after full mha stats: min=", minimum(x_mha), " max=", maximum(x_mha), " mean=", mean(x_mha))

# Check weight shapes
println("\n=== WEIGHT SHAPES ===")
println("attention.to_qkv.weight: ", size(attention.to_qkv.weight))
println("attention.to_g.weight: ", size(attention.to_g.weight))
println("attention.to_out.weight: ", size(attention.to_out.weight))
println("attention.to_bias.weight: ", size(attention.to_bias.weight))

# Check weight values
println("\n=== WEIGHT SAMPLE VALUES ===")
println("attention.to_qkv.weight[1:3, 1:3]: ", attention.to_qkv.weight[1:3, 1:3])
println("attention.to_out.weight[1:3, 1:3]: ", attention.to_out.weight[1:3, 1:3])

println("\n=== TRANSITION STEP ===")
transition = layer0.transition  # TransitionADALN
# After MHA, the x is: seqs + x_mha (if residual)
x_after_mha = seqs .+ x_mha
x_after_mha = x_after_mha .* mask_exp
println("x after mha+residual: ", size(x_after_mha), " sample: ", x_after_mha[1:5, 1, 1])

# Debug transition components
trans_adaln = transition.adaln
trans_swiglu = transition.transition
trans_scale = transition.scale_output

println("\n  Transition ADALN input x_after_mha: ", x_after_mha[1:5, 1, 1])
x_trans_adaln = trans_adaln(x_after_mha, cond, mask)
println("  Transition ADALN output: ", x_trans_adaln[1:5, 1, 1])
println("  Transition ADALN stats: min=", minimum(x_trans_adaln), " max=", maximum(x_trans_adaln))

println("\n  DEBUG SwiGLU STEP BY STEP:")
println("  trans_swiglu.linear_in weight shape: ", size(trans_swiglu.linear_in.weight))
println("  trans_swiglu.linear_out weight shape: ", size(trans_swiglu.linear_out.weight))

# Step through SwiGLU manually
if !isnothing(trans_swiglu.ln)
    x_ln = trans_swiglu.ln(x_trans_adaln)
    println("  After LayerNorm: ", x_ln[1:5, 1, 1])
else
    x_ln = x_trans_adaln
    println("  No LayerNorm, x_ln = x_trans_adaln")
end

x_proj_in = trans_swiglu.linear_in(x_ln)
println("  After linear_in: shape=", size(x_proj_in), " sample=", x_proj_in[1:5, 1, 1])
println("  After linear_in stats: min=", minimum(x_proj_in), " max=", maximum(x_proj_in))

x_swiglu_act = trans_swiglu.swiglu(x_proj_in)
println("  After SwiGLU act: shape=", size(x_swiglu_act), " sample=", x_swiglu_act[1:5, 1, 1])
println("  After SwiGLU act stats: min=", minimum(x_swiglu_act), " max=", maximum(x_swiglu_act))

x_proj_out = trans_swiglu.linear_out(x_swiglu_act)
println("  After linear_out: shape=", size(x_proj_out), " sample=", x_proj_out[1:5, 1, 1])
println("  After linear_out stats: min=", minimum(x_proj_out), " max=", maximum(x_proj_out))

x_trans_swiglu = trans_swiglu(x_trans_adaln, mask)
println("  Transition SwiGLU output: ", x_trans_swiglu[1:5, 1, 1])
println("  Transition SwiGLU stats: min=", minimum(x_trans_swiglu), " max=", maximum(x_trans_swiglu))

x_trans_scale = trans_scale(x_trans_swiglu, cond, mask)
println("  Transition Scale output: ", x_trans_scale[1:5, 1, 1])
println("  Transition Scale stats: min=", minimum(x_trans_scale), " max=", maximum(x_trans_scale))

# Run transition
x_tr = transition(x_after_mha, cond, mask)
println("x after transition (full): ", size(x_tr), " sample: ", x_tr[1:5, 1, 1])
println("x after transition stats: min=", minimum(x_tr), " max=", maximum(x_tr), " mean=", mean(x_tr))

# Add residual for transition
x_final = x_after_mha .+ x_tr
x_final = x_final .* mask_exp
println("x final (after transition+residual): ", size(x_final), " sample: ", x_final[1:5, 1, 1])
println("x final stats: min=", minimum(x_final), " max=", maximum(x_final), " mean=", mean(x_final))

println("\n=== COMPARE WITH FULL LAYER ===")
x_layer0 = layer0(seqs, pair, cond, mask)
println("x from full layer0: ", size(x_layer0), " sample: ", x_layer0[1:5, 1, 1])
println("x from full layer0 stats: min=", minimum(x_layer0), " max=", maximum(x_layer0), " mean=", mean(x_layer0))
