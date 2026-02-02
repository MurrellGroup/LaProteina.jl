#!/usr/bin/env julia
"""
Test full model parity - run a forward pass through the score network
with trained weights and compare outputs with Python reference.
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
x_t_bb_ca_py = npzread(joinpath(test_dir, "full_model_x_t_bb_ca.npy"))  # [B, L, 3]
x_t_local_latents_py = npzread(joinpath(test_dir, "full_model_x_t_local_latents.npy"))  # [B, L, 8]
x_sc_bb_ca_py = npzread(joinpath(test_dir, "full_model_x_sc_bb_ca.npy"))  # [B, L, 3]
x_sc_local_latents_py = npzread(joinpath(test_dir, "full_model_x_sc_local_latents.npy"))  # [B, L, 8]
t_val = npzread(joinpath(test_dir, "full_model_t_val.npy"))[1]
mask_py = npzread(joinpath(test_dir, "full_model_mask.npy"))  # [B, L]
out_bb_ca_py = npzread(joinpath(test_dir, "full_model_out_bb_ca.npy"))  # [B, L, 3]
out_local_latents_py = npzread(joinpath(test_dir, "full_model_out_local_latents.npy"))  # [B, L, 8]

println("Python output bb_ca[0,0,:]: ", out_bb_ca_py[1, 1, :])
println("Python output local_latents[0,0,:5]: ", out_local_latents_py[1, 1, 1:5])

# Convert to Julia format [D, L, B]
x_t_bb_ca = python_to_julia(x_t_bb_ca_py)
x_t_local_latents = python_to_julia(x_t_local_latents_py)
x_sc_bb_ca = python_to_julia(x_sc_bb_ca_py)
x_sc_local_latents = python_to_julia(x_sc_local_latents_py)
mask = python_to_julia_mask(mask_py)  # [L, B]

B = size(x_t_bb_ca, 3)
L = size(x_t_bb_ca, 2)

println("\nInput shapes (Julia format):")
println("  x_t_bb_ca: ", size(x_t_bb_ca))
println("  x_t_local_latents: ", size(x_t_local_latents))
println("  mask: ", size(mask))
println("  t: ", t_val)

# Create the model
println("\nCreating ScoreNetwork...")
model = ScoreNetwork(
    n_layers=14,
    token_dim=768,
    n_heads=12,
    latent_dim=8,
    dim_cond=256,
    t_emb_dim=256,
    pair_dim=256,
    seq_sep_dim=127,
    xt_pair_dist_dim=30,
    xt_pair_dist_min=0.1f0,
    xt_pair_dist_max=3.0f0,
    x_sc_pair_dist_dim=30,
    x_sc_pair_dist_min=0.1f0,
    x_sc_pair_dist_max=3.0f0
)

# Load weights
println("Loading weights...")
load_score_network_weights!(model, weights_path)

# Create batch dict - model expects nested dicts for x_t, t, x_sc
batch = Dict{Symbol, Any}(
    :x_t => Dict{Symbol, Any}(
        :bb_ca => x_t_bb_ca,
        :local_latents => x_t_local_latents
    ),
    :t => Dict{Symbol, Any}(
        :bb_ca => fill(Float32(t_val), B),
        :local_latents => fill(Float32(t_val), B)
    ),
    :x_sc => Dict{Symbol, Any}(
        :bb_ca => x_sc_bb_ca,
        :local_latents => x_sc_local_latents
    ),
    :mask => mask
)

# Forward pass
println("\nRunning forward pass...")
output = model(batch)

v_bb_ca = output[:bb_ca][:v]  # [3, L, B]
v_local_latents = output[:local_latents][:v]  # [8, L, B]

println("\nJulia output shapes:")
println("  v_bb_ca: ", size(v_bb_ca))
println("  v_local_latents: ", size(v_local_latents))

# Convert to Python format for comparison
v_bb_ca_jl = julia_to_python(v_bb_ca)  # [B, L, 3]
v_local_latents_jl = julia_to_python(v_local_latents)  # [B, L, 8]

println("\nJulia output bb_ca[0,0,:]: ", v_bb_ca_jl[1, 1, :])
println("Julia output local_latents[0,0,:5]: ", v_local_latents_jl[1, 1, 1:5])

# Compare outputs
diff_bb_ca = abs.(v_bb_ca_jl .- out_bb_ca_py)
diff_local_latents = abs.(v_local_latents_jl .- out_local_latents_py)

max_diff_bb_ca = maximum(diff_bb_ca)
max_diff_local_latents = maximum(diff_local_latents)
mean_diff_bb_ca = mean(diff_bb_ca)
mean_diff_local_latents = mean(diff_local_latents)

println("\n" * "="^60)
println("PARITY RESULTS")
println("="^60)
println("bb_ca output:")
println("  Max diff: ", max_diff_bb_ca)
println("  Mean diff: ", mean_diff_bb_ca)
println("  Python: ", out_bb_ca_py[1, 1, :])
println("  Julia:  ", v_bb_ca_jl[1, 1, :])

println("\nlocal_latents output:")
println("  Max diff: ", max_diff_local_latents)
println("  Mean diff: ", mean_diff_local_latents)
println("  Python: ", out_local_latents_py[1, 1, 1:5])
println("  Julia:  ", v_local_latents_jl[1, 1, 1:5])

# Check if parity is achieved
tol = 1e-3
if max_diff_bb_ca < tol && max_diff_local_latents < tol
    println("\n✓ PARITY ACHIEVED! (tol=$tol)")
else
    println("\n✗ PARITY FAILED (tol=$tol)")
    println("  bb_ca max diff: $max_diff_bb_ca")
    println("  local_latents max diff: $max_diff_local_latents")
end
