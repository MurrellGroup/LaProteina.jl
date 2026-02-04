#!/usr/bin/env julia
"""
Debug full model parity step-by-step.
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

# Load Python reference data
println("Loading Python reference data...")
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

println("B=$B, L=$L")

# Load weights
weights = npzread(weights_path)
println("\nAvailable weight keys (first 20):")
for (i, k) in enumerate(sort(collect(keys(weights))))
    i > 20 && break
    println("  $k: $(size(weights[k]))")
end

# Check init_repr_factory weights
println("\n=== CHECKING INIT_REPR_FACTORY ===")
w = weights["init_repr_factory.linear_out.weight"]
println("init_repr_factory weight shape: $(size(w))")  # Should be [768, 46]

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

# Check model projection weight shape before loading
println("Model init_repr_factory projection weight shape: $(size(model.init_repr_factory.projection.weight))")

# Load weights
println("\nLoading weights...")
load_score_network_weights!(model, weights_path)

# Check that weights were loaded correctly
model_w = model.init_repr_factory.projection.weight
println("Model weight after loading (first row, first 5 cols): $(model_w[1, 1:5])")
println("NPZ weight (first row, first 5 cols): $(w[1, 1:5])")

# Manual feature computation test
println("\n=== MANUAL FEATURE COMPUTATION ===")

# Build the sequence batch for features - using nested structure
seq_batch = Dict{Symbol, Any}(
    :x_t => Dict{Symbol, Any}(
        :bb_ca => x_t_bb_ca,
        :local_latents => x_t_local_latents
    ),
    :x_sc => Dict{Symbol, Any}(
        :bb_ca => x_sc_bb_ca,
        :local_latents => x_sc_local_latents
    ),
    :mask => mask
)

# Compute sequence features
seq_feats = model.init_repr_factory(seq_batch)
println("Sequence features shape: $(size(seq_feats))")
println("Sequence features [1:5, 1, 1]: $(seq_feats[1:5, 1, 1])")

# x_t_bb_ca should be the first 3 features (already checked)
println("\nFeature breakdown:")
println("  x_t_bb_ca (from input):     $(x_t_bb_ca[:, 1, 1])")
println("  seq_feats[1:3, 1, 1]:       $(seq_feats[1:3, 1, 1])")
println("  x_t_local_latents (input):  $(x_t_local_latents[:, 1, 1])")
println("  seq_feats[4:11, 1, 1]:      $(seq_feats[4:11, 1, 1])")

# Check time embeddings
println("\n=== TIME EMBEDDING CHECK ===")
t_bb_ca = fill(Float32(t_val), B)
t_local_latents = fill(Float32(t_val), B)

cond_batch = Dict{Symbol, Any}(
    :t => Dict{Symbol, Any}(
        :bb_ca => t_bb_ca,
        :local_latents => t_local_latents
    ),
    :mask => mask
)
cond_feats = model.cond_factory(cond_batch)
println("Conditioning features shape: $(size(cond_feats))")
println("Conditioning features [1:5, 1, 1]: $(cond_feats[1:5, 1, 1])")

# Get raw time embedding for comparison
time_emb = get_time_embedding(t_bb_ca, 256)
println("Raw time embedding (bb_ca) [1:5, 1]: $(time_emb[1:5, 1])")
