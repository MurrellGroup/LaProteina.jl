#!/usr/bin/env julia
"""
Debug MHA ADALN in Julia.
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

# Load Python reference
py_normed = npzread(joinpath(test_dir, "debug_normed.npy"))
py_normed_cond = npzread(joinpath(test_dir, "debug_normed_cond.npy"))
py_gamma = npzread(joinpath(test_dir, "debug_gamma.npy"))
py_beta = npzread(joinpath(test_dir, "debug_beta.npy"))
py_adaln_out = npzread(joinpath(test_dir, "debug_mha_adaln_out.npy"))

println("Python references loaded")
println("py_normed[0,0,:5]: ", py_normed[1, 1, 1:5])
println("py_gamma[0,0,:5]: ", py_gamma[1, 1, 1:5])
println("py_beta[0,0,:5]: ", py_beta[1, 1, 1:5])
println("py_adaln_out[0,0,:5]: ", py_adaln_out[1, 1, 1:5])

# Convert to Julia format
x_t_bb_ca = python_to_julia(x_t_bb_ca_py)
x_t_local_latents = python_to_julia(x_t_local_latents_py)
x_sc_bb_ca = python_to_julia(x_sc_bb_ca_py)
x_sc_local_latents = python_to_julia(x_sc_local_latents_py)
mask = python_to_julia_mask(mask_py)

B = size(x_t_bb_ca, 3)
L = size(x_t_bb_ca, 2)

# Create and load model
model = ScoreNetwork(
    n_layers=14,
    token_dim=768,
    n_heads=12,
    latent_dim=8,
    dim_cond=256,
    t_emb_dim=256,
    pair_dim=256
)
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

mask_exp = reshape(mask, 1, L, B)
seqs = seqs .* mask_exp

# Conditioning transitions
cond = model.transition_c_1(cond, mask)
cond = model.transition_c_2(cond, mask)

println("\n=== INPUTS ===")
println("seqs[:5, 1, 1]: ", seqs[1:5, 1, 1])
println("cond[:5, 1, 1]: ", cond[1:5, 1, 1])

# Get MHA ADALN
mha = model.transformer_layers[1].mha
adaln = mha.adaln

println("\n=== ADALN INTERNALS ===")

# Apply input mask
x_in = seqs .* mask_exp
println("x_in[:5, 1, 1]: ", x_in[1:5, 1, 1])

# Step 1: LayerNorm on x
normed = adaln.norm(x_in)
println("\nJulia normed[:5, 1, 1]: ", normed[1:5, 1, 1])
println("Julia normed stats: min=", minimum(normed), " max=", maximum(normed))
py_normed_jl = python_to_julia(py_normed)
println("Python normed[:5, 1, 1]: ", py_normed_jl[1:5, 1, 1])
diff_normed = maximum(abs.(normed .- py_normed_jl))
println("Max diff normed: ", diff_normed)

# Step 2: LayerNorm on cond
normed_cond = adaln.norm_cond(cond)
println("\nJulia normed_cond[:5, 1, 1]: ", normed_cond[1:5, 1, 1])
py_normed_cond_jl = python_to_julia(py_normed_cond)
println("Python normed_cond[:5, 1, 1]: ", py_normed_cond_jl[1:5, 1, 1])
diff_normed_cond = maximum(abs.(normed_cond .- py_normed_cond_jl))
println("Max diff normed_cond: ", diff_normed_cond)

# Step 3: to_gamma (Dense + sigmoid)
gamma = adaln.to_gamma(normed_cond)
println("\nJulia gamma[:5, 1, 1]: ", gamma[1:5, 1, 1])
println("Julia gamma stats: min=", minimum(gamma), " max=", maximum(gamma))
py_gamma_jl = python_to_julia(py_gamma)
println("Python gamma[:5, 1, 1]: ", py_gamma_jl[1:5, 1, 1])
diff_gamma = maximum(abs.(gamma .- py_gamma_jl))
println("Max diff gamma: ", diff_gamma)

# Step 4: to_beta (Dense)
beta = adaln.to_beta(normed_cond)
println("\nJulia beta[:5, 1, 1]: ", beta[1:5, 1, 1])
println("Julia beta stats: min=", minimum(beta), " max=", maximum(beta))
py_beta_jl = python_to_julia(py_beta)
println("Python beta[:5, 1, 1]: ", py_beta_jl[1:5, 1, 1])
diff_beta = maximum(abs.(beta .- py_beta_jl))
println("Max diff beta: ", diff_beta)

# Step 5: Final computation
out_manual = normed .* gamma .+ beta
println("\nJulia out (normed * gamma + beta)[:5, 1, 1]: ", out_manual[1:5, 1, 1])
println("Julia out stats: min=", minimum(out_manual), " max=", maximum(out_manual))
py_adaln_out_jl = python_to_julia(py_adaln_out)
println("Python out[:5, 1, 1]: ", py_adaln_out_jl[1:5, 1, 1])
diff_out = maximum(abs.(out_manual .- py_adaln_out_jl))
println("Max diff out: ", diff_out)

# Full ADALN call
out_full = adaln(x_in, cond, mask)
println("\nJulia full adaln[:5, 1, 1]: ", out_full[1:5, 1, 1])
