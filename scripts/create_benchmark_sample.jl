#!/usr/bin/env julia
# Create a fixed benchmark sample from random data for GPU timing tests.
# Saves synthetic ScoreNetwork inputs as JLD2 for reproducible benchmarking.
#
# Usage: julia scripts/create_benchmark_sample.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using JLD2
using Random

Random.seed!(42)

println("Creating benchmark sample...")

# Fixed dimensions matching training config
L = 128    # sequence length
B = 4      # batch size
token_dim = 768
pair_dim = 256
dim_cond = 256
latent_dim = 8
n_heads = 12

# Generate random input tensors (CPU)
seq_feats = randn(Float32, token_dim, L, B)
cond_feats = randn(Float32, dim_cond, L, B)
pair_feats = randn(Float32, pair_dim, L, L, B)
mask = ones(Float32, L, B)

# Input coordinates and targets
xt_ca = randn(Float32, 3, L, B) .* 5f0
xt_ll = randn(Float32, latent_dim, L, B)
x1_ca = randn(Float32, 3, L, B) .* 5f0
x1_ll = randn(Float32, latent_dim, L, B)

# Times
t_ca = rand(Float32, B) .* 0.99f0 .+ 0.01f0
t_ll = rand(Float32, B) .* 0.99f0 .+ 0.01f0

# Save benchmark sample
output_path = joinpath(@__DIR__, "..", "test", "benchmark_sample.jld2")
mkpath(dirname(output_path))

jldsave(output_path;
    seq_feats = seq_feats,
    cond_feats = cond_feats,
    pair_feats = pair_feats,
    mask = mask,
    xt_ca = xt_ca,
    xt_ll = xt_ll,
    x1_ca = x1_ca,
    x1_ll = x1_ll,
    t_ca = t_ca,
    t_ll = t_ll,
    L = L,
    B = B
)

println("Benchmark sample saved to: $output_path")
println("  Shapes: seq=$(size(seq_feats)), cond=$(size(cond_feats)), pair=$(size(pair_feats))")
println("  L=$L, B=$B")
