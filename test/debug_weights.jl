#!/usr/bin/env julia
"""
Debug weight loading - compare weights between Julia and Python.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using NPZ
using Flux
using LinearAlgebra

# Load weights file
weights_path = joinpath(@__DIR__, "..", "weights", "score_network.npz")
weights = npzread(weights_path)

# Check transition.transition.swish_linear.0.weight
key = "transformer_layers.0.transition.transition.swish_linear.0.weight"
w_npz = weights[key]
println("=== TRANSITION SWISH_LINEAR WEIGHT ===")
println("NPZ weight shape: ", size(w_npz))
println("NPZ weight [1:3, 1:3]: ", w_npz[1:3, 1:3])
println("NPZ weight min/max: ", minimum(w_npz), " / ", maximum(w_npz))

# Create model and load weights
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

# Check loaded weight
w_jl = model.transformer_layers[1].transition.transition.linear_in.weight
println("\nJulia weight shape: ", size(w_jl))
println("Julia weight [1:3, 1:3]: ", w_jl[1:3, 1:3])
println("Julia weight min/max: ", minimum(w_jl), " / ", maximum(w_jl))

# Check linear_out weight
key_out = "transformer_layers.0.transition.transition.linear_out.weight"
w_out_npz = weights[key_out]
w_out_jl = model.transformer_layers[1].transition.transition.linear_out.weight
println("\n=== TRANSITION LINEAR_OUT WEIGHT ===")
println("NPZ weight shape: ", size(w_out_npz))
println("Julia weight shape: ", size(w_out_jl))
println("NPZ weight [1:3, 1:3]: ", w_out_npz[1:3, 1:3])
println("Julia weight [1:3, 1:3]: ", w_out_jl[1:3, 1:3])

# Test the linear_in operation
println("\n=== TEST LINEAR_IN OPERATION ===")
# Create a simple input
x_test = randn(Float32, 768)  # [D]
println("Input x_test [1:5]: ", x_test[1:5])

# Python-style matmul: output = x @ W.T  (where x is [D], W is [out, in])
# This is equivalent to: W * x
y_test = w_jl * x_test
println("Output y_test [1:5] (Julia): ", y_test[1:5])
println("Output y_test shape: ", size(y_test))

# Compare with numpy-style: output = W @ x (if W is [out, in] and x is [in])
# NPZ weight is [out, in] in row-major (which is [in, out] in column-major)
# So we might need to transpose
y_test_npz = w_npz * x_test
println("Output y_test_npz [1:5] (NPZ direct): ", y_test_npz[1:5])

# Check if we need to transpose
println("\n=== CHECKING IF TRANSPOSE NEEDED ===")
println("w_npz shape: ", size(w_npz), " (row-major: [out=6144, in=768])")
println("w_jl shape: ", size(w_jl), " (Julia Dense expects [out, in])")

# If NPZ loads [out, in] in row-major, in Julia it's read as [out, in] column-major
# which is actually the transposed matrix [in, out] when viewed as data
# This is confusing, let me check by element
println("\nComparing specific elements:")
println("w_npz[1, 1] = ", w_npz[1, 1])
println("w_jl[1, 1] = ", w_jl[1, 1])
println("w_npz[1, 2] = ", w_npz[1, 2])
println("w_jl[1, 2] = ", w_jl[1, 2])
println("w_npz[2, 1] = ", w_npz[2, 1])
println("w_jl[2, 1] = ", w_jl[2, 1])

# Check transition ADALN weights
println("\n=== TRANSITION ADALN WEIGHTS ===")

# norm_cond
key = "transformer_layers.0.transition.adaln.norm_cond.weight"
if haskey(weights, key)
    w_npz = weights[key]
    w_jl = model.transformer_layers[1].transition.adaln.norm_cond.diag.scale
    println("norm_cond.weight NPZ shape: ", size(w_npz))
    println("norm_cond.weight Julia shape: ", size(w_jl))
    println("norm_cond.weight NPZ [1:5]: ", w_npz[1:5])
    println("norm_cond.weight Julia [1:5]: ", w_jl[1:5])
    println("Match: ", w_npz[1:5] ≈ w_jl[1:5])
end

# to_gamma weight
key = "transformer_layers.0.transition.adaln.to_gamma.0.weight"
if haskey(weights, key)
    w_npz = weights[key]
    w_jl = model.transformer_layers[1].transition.adaln.to_gamma.layers[1].weight
    println("\nto_gamma.weight NPZ shape: ", size(w_npz))
    println("to_gamma.weight Julia shape: ", size(w_jl))
    println("to_gamma.weight NPZ [1:3, 1:3]: ", w_npz[1:3, 1:3])
    println("to_gamma.weight Julia [1:3, 1:3]: ", w_jl[1:3, 1:3])
    println("Match: ", all(abs.(w_npz .- w_jl) .< 1e-5))
end

# to_beta weight
key = "transformer_layers.0.transition.adaln.to_beta.weight"
if haskey(weights, key)
    w_npz = weights[key]
    w_jl = model.transformer_layers[1].transition.adaln.to_beta.weight
    println("\nto_beta.weight NPZ shape: ", size(w_npz))
    println("to_beta.weight Julia shape: ", size(w_jl))
    println("to_beta.weight NPZ [1:3, 1:3]: ", w_npz[1:3, 1:3])
    println("to_beta.weight Julia [1:3, 1:3]: ", w_jl[1:3, 1:3])
    println("Match: ", all(abs.(w_npz .- w_jl) .< 1e-5))
end
