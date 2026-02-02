#!/usr/bin/env julia
"""
Debug transformer layer values.
"""

using Pkg
Pkg.activate("/home/claudey/JuProteina/JuProteina")

using JuProteina
using NPZ
using Flux
using LinearAlgebra
using Statistics

# Paths
test_dir = "/home/claudey/JuProteina/JuProteina/test"
weights_path = "/home/claudey/JuProteina/JuProteina/weights/score_network.npz"

# Load Python reference data
println("Loading Python reference data...")
py_seqs_after_layer0 = npzread(joinpath(test_dir, "debug_seqs_after_layer0.npy"))  # [B, L, D]
py_seqs_after_all = npzread(joinpath(test_dir, "debug_seqs_after_all_layers.npy"))  # [B, L, D]
py_ca_out = npzread(joinpath(test_dir, "debug_ca_out.npy"))  # [B, L, 3]
py_ll_out = npzread(joinpath(test_dir, "debug_ll_out.npy"))  # [B, L, 8]

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
println("\nCreating ScoreNetwork...")
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
println("Loading weights...")
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

println("\n=== INITIAL REPRESENTATIONS ===")
seqs = model.init_repr_factory(batch)  # [D, L, B]
cond = model.cond_factory(batch)  # [D, L, B]
pair = model.pair_rep_builder(batch)  # [D, L, L, B]

println("seqs shape: $(size(seqs))")
println("seqs[1:5, 1, 1]: $(seqs[1:5, 1, 1])")
println("cond shape: $(size(cond))")
println("cond[1:5, 1, 1]: $(cond[1:5, 1, 1])")
println("pair shape: $(size(pair))")
println("pair[1:5, 1, 1, 1]: $(pair[1:5, 1, 1, 1])")

# Apply mask to seqs (as Python does)
mask_exp = reshape(mask, 1, L, B)
seqs = seqs .* mask_exp

# Conditioning transitions
cond = model.transition_c_1(cond, mask)
println("\n=== AFTER TRANS_C_1 ===")
println("cond[1:5, 1, 1]: $(cond[1:5, 1, 1])")

cond = model.transition_c_2(cond, mask)
println("\n=== AFTER TRANS_C_2 ===")
println("cond[1:5, 1, 1]: $(cond[1:5, 1, 1])")

# First transformer layer
println("\n=== TRANSFORMER LAYER 0 ===")
layer0 = model.transformer_layers[1]
seqs_after_layer0 = layer0(seqs, pair, cond, mask)  # [D, L, B]
seqs_after_layer0_py = julia_to_python(seqs_after_layer0)

println("seqs after layer0 shape: $(size(seqs_after_layer0))")
println("Julia  seqs_after_layer0[1:5, 1, 1]: $(seqs_after_layer0[1:5, 1, 1])")
println("Python seqs_after_layer0[0,0,:5]: $(py_seqs_after_layer0[1, 1, 1:5])")
diff_layer0 = abs.(seqs_after_layer0_py .- py_seqs_after_layer0)
println("Max diff: $(maximum(diff_layer0))")

# All layers
seqs = seqs_after_layer0
for i in 2:length(model.transformer_layers)
    seqs = model.transformer_layers[i](seqs, pair, cond, mask)
end

println("\n=== AFTER ALL TRANSFORMER LAYERS ===")
seqs_py = julia_to_python(seqs)
println("seqs shape: $(size(seqs))")
println("Julia  seqs[1:5, 1, 1]: $(seqs[1:5, 1, 1])")
println("Python seqs[0,0,:5]: $(py_seqs_after_all[1, 1, 1:5])")
diff_all = abs.(seqs_py .- py_seqs_after_all)
println("Max diff: $(maximum(diff_all))")

# Output projections
println("\n=== OUTPUT PROJECTIONS ===")
ca_out = model.ca_proj(seqs)  # [3, L, B]
ca_out = ca_out .* mask_exp

ll_out = model.local_latents_proj(seqs)  # [8, L, B]
ll_out = ll_out .* mask_exp

ca_out_py = julia_to_python(ca_out)
ll_out_py = julia_to_python(ll_out)

println("Julia  ca_out[:, 1, 1]: $(ca_out[:, 1, 1])")
println("Python ca_out[0,0,:]: $(py_ca_out[1, 1, :])")
diff_ca = abs.(ca_out_py .- py_ca_out)
println("Max diff ca_out: $(maximum(diff_ca))")

println("\nJulia  ll_out[1:5, 1, 1]: $(ll_out[1:5, 1, 1])")
println("Python ll_out[0,0,:5]: $(py_ll_out[1, 1, 1:5])")
diff_ll = abs.(ll_out_py .- py_ll_out)
println("Max diff ll_out: $(maximum(diff_ll))")

# Summary
println("\n" * "="^60)
println("TRANSFORMER LAYER PARITY SUMMARY")
println("="^60)

tol = 0.01
function check(name, max_diff, tol)
    status = max_diff < tol ? "✓" : "✗"
    println("$status $name: max_diff=$max_diff (tol=$tol)")
    return max_diff < tol
end

passed = true
passed &= check("After layer 0", maximum(diff_layer0), tol)
passed &= check("After all layers", maximum(diff_all), tol)
passed &= check("CA output", maximum(diff_ca), tol)
passed &= check("LL output", maximum(diff_ll), tol)

if passed
    println("\n✓ ALL TRANSFORMER LAYERS MATCH!")
else
    println("\n✗ TRANSFORMER LAYER MISMATCH DETECTED")
end
