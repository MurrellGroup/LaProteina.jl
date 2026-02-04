#!/usr/bin/env julia
# Test that BranchingScoreNetwork can be constructed and has correct structure

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

println("=" ^ 70)
println("Testing Branching Score Network Structure")
println("=" ^ 70)

# Load LaProteina first
println("\n=== Loading LaProteina ===")
using LaProteina
println("LaProteina loaded")

# Now include the branching module files directly (since it's not yet integrated)
println("\n=== Loading Branching Module ===")
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
println("BranchingScoreNetwork loaded")

# Test 1: Create a small ScoreNetwork
println("\n=== Test 1: Create BranchingScoreNetwork ===")
base = ScoreNetwork(
    n_layers=2, token_dim=64, pair_dim=32, n_heads=4,
    dim_cond=32, latent_dim=8, output_param=:v
)
println("Base ScoreNetwork created")
println("  n_layers: $(base.n_layers)")
println("  output_param: $(base.output_param)")

# Create BranchingScoreNetwork
branching_net = BranchingScoreNetwork(base)
println("BranchingScoreNetwork created")

# Check structure
println("\n=== Test 2: Check Structure ===")
println("  indel_time_proj: $(typeof(branching_net.indel_time_proj))")
println("  split_head: $(typeof(branching_net.split_head))")
println("  del_head: $(typeof(branching_net.del_head))")

# Test forward pass shape
println("\n=== Test 3: Forward Pass Shape ===")
L, B = 10, 2
latent_dim = 8

batch = Dict{Symbol, Any}(
    :x_t => Dict(
        :bb_ca => randn(Float32, 3, L, B),
        :local_latents => randn(Float32, latent_dim, L, B)
    ),
    :t => Dict(
        :bb_ca => rand(Float32, B),
        :local_latents => rand(Float32, B)
    ),
    :mask => ones(Float32, L, B)
)

output = branching_net(batch)

println("Output keys: $(keys(output))")
println("  :bb_ca[:v] shape: $(size(output[:bb_ca][:v]))")
println("  :local_latents[:v] shape: $(size(output[:local_latents][:v]))")
println("  :split shape: $(size(output[:split]))")
println("  :del shape: $(size(output[:del]))")

# Verify shapes
@assert size(output[:bb_ca][:v]) == (3, L, B) "CA output wrong shape"
@assert size(output[:local_latents][:v]) == (latent_dim, L, B) "Latent output wrong shape"
@assert size(output[:split]) == (L, B) "Split output wrong shape"
@assert size(output[:del]) == (L, B) "Del output wrong shape"

println("\nAll shape checks passed!")

# Test trainable params extraction
println("\n=== Test 4: Trainable Params ===")
indel_params = trainable_indel_params(branching_net)
println("Number of indel parameter arrays: $(length(indel_params))")

# Count parameters
n_indel = sum(length, indel_params)
n_total = sum(length, Flux.params(branching_net))
println("Indel params: $n_indel")
println("Total params: $n_total")
println("Base params: $(n_total - n_indel)")

println("\n" * "=" ^ 70)
println("All tests passed!")
println("=" ^ 70)
