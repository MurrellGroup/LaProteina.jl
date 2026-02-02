#!/usr/bin/env julia
"""
Compare Julia intermediate values with Python reference.
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
println("Loading Python intermediate data...")
py_seq_repr = npzread(joinpath(test_dir, "debug_seq_repr.npy"))  # [B, L, D]
py_cond_repr = npzread(joinpath(test_dir, "debug_cond_repr.npy"))  # [B, L, D]
py_pair_repr = npzread(joinpath(test_dir, "debug_pair_repr.npy"))  # [B, L, L, D]
py_cond_after_trans = npzread(joinpath(test_dir, "debug_cond_after_trans.npy"))  # [B, L, D]

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

# ============================================================
# Compare sequence features
# ============================================================
println("\n" * "="^60)
println("SEQUENCE FEATURES (init_repr_factory)")
println("="^60)

jl_seq_repr = model.init_repr_factory(batch)  # [D, L, B]
jl_seq_repr_py_fmt = julia_to_python(jl_seq_repr)  # [B, L, D]

println("Python seq_repr[0,0,:5]: $(py_seq_repr[1, 1, 1:5])")
println("Julia  seq_repr[0,0,:5]: $(jl_seq_repr_py_fmt[1, 1, 1:5])")
diff_seq = abs.(jl_seq_repr_py_fmt .- py_seq_repr)
println("Max diff: $(maximum(diff_seq))")
println("Mean diff: $(mean(diff_seq))")

# ============================================================
# Compare conditioning features
# ============================================================
println("\n" * "="^60)
println("CONDITIONING FEATURES (cond_factory)")
println("="^60)

jl_cond_repr = model.cond_factory(batch)  # [D, L, B]
jl_cond_repr_py_fmt = julia_to_python(jl_cond_repr)  # [B, L, D]

println("Python cond_repr[0,0,:5]: $(py_cond_repr[1, 1, 1:5])")
println("Julia  cond_repr[0,0,:5]: $(jl_cond_repr_py_fmt[1, 1, 1:5])")
diff_cond = abs.(jl_cond_repr_py_fmt .- py_cond_repr)
println("Max diff: $(maximum(diff_cond))")
println("Mean diff: $(mean(diff_cond))")

# ============================================================
# Compare pair representation
# ============================================================
println("\n" * "="^60)
println("PAIR REPRESENTATION (pair_rep_builder)")
println("="^60)

jl_pair_repr = model.pair_rep_builder(batch)  # [D, L, L, B]
jl_pair_repr_py_fmt = julia_to_python_pair(jl_pair_repr)  # [B, L, L, D]

println("Python pair_repr[0,0,0,:5]: $(py_pair_repr[1, 1, 1, 1:5])")
println("Julia  pair_repr[0,0,0,:5]: $(jl_pair_repr_py_fmt[1, 1, 1, 1:5])")
println("Python pair_repr[0,0,1,:5]: $(py_pair_repr[1, 1, 2, 1:5])")
println("Julia  pair_repr[0,0,1,:5]: $(jl_pair_repr_py_fmt[1, 1, 2, 1:5])")
diff_pair = abs.(jl_pair_repr_py_fmt .- py_pair_repr)
println("Max diff: $(maximum(diff_pair))")
println("Mean diff: $(mean(diff_pair))")

# ============================================================
# Compare conditioning after transitions
# ============================================================
println("\n" * "="^60)
println("CONDITIONING AFTER TRANSITIONS")
println("="^60)

# First apply transition_c_1
jl_c = jl_cond_repr
jl_c = model.transition_c_1(jl_c, mask)  # [D, L, B]  - mask is [L, B]
println("Julia after trans_c_1 [0,0,:5]: $(jl_c[1:5, 1, 1])")

# Then apply transition_c_2
jl_c = model.transition_c_2(jl_c, mask)  # [D, L, B]
jl_c_py_fmt = julia_to_python(jl_c)  # [B, L, D]

println("Python cond after trans[0,0,:5]: $(py_cond_after_trans[1, 1, 1:5])")
println("Julia  cond after trans[0,0,:5]: $(jl_c_py_fmt[1, 1, 1:5])")
diff_c = abs.(jl_c_py_fmt .- py_cond_after_trans)
println("Max diff: $(maximum(diff_c))")
println("Mean diff: $(mean(diff_c))")

# ============================================================
# Summary
# ============================================================
println("\n" * "="^60)
println("PARITY SUMMARY")
println("="^60)

tol = 1e-4
passed = true
function check_parity(name, max_diff, tol)
    status = max_diff < tol ? "✓" : "✗"
    println("$status $name: max_diff=$max_diff (tol=$tol)")
    return max_diff < tol
end

passed &= check_parity("Sequence features", maximum(diff_seq), tol)
passed &= check_parity("Conditioning features", maximum(diff_cond), tol)
passed &= check_parity("Pair representation", maximum(diff_pair), tol)
passed &= check_parity("Cond after transitions", maximum(diff_c), tol)

if passed
    println("\n✓ ALL INTERMEDIATE VALUES MATCH!")
else
    println("\n✗ SOME INTERMEDIATE VALUES DON'T MATCH")
end
