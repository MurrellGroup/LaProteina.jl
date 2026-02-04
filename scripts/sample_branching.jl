#!/usr/bin/env julia
# Sample from trained Branching Flows model
#
# Usage:
#   julia scripts/sample_branching.jl
#
# Environment variables:
#   WEIGHTS_DIR - Directory containing trained weights
#   N_SAMPLES - Number of samples to generate (default: 1)
#   X0_MEAN_LENGTH - Mean of Poisson distribution for initial length (default: 100)
#   N_STEPS - Number of integration steps (default: 100)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, tensor
using Flowfusion: RDNFlow, MaskedState
import Flowfusion
using Distributions: Poisson
using Flux
using CUDA
using Random
using Printf
using JLD2

# Include branching module
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)

println("=" ^ 60)
println("Branching Flows Sampling")
println("=" ^ 60)

# Configuration
weights_dir = get(ENV, "WEIGHTS_DIR", joinpath(@__DIR__, "..", "weights"))
n_samples = parse(Int, get(ENV, "N_SAMPLES", "1"))
X0_mean_length = parse(Int, get(ENV, "X0_MEAN_LENGTH", "100"))
n_steps = parse(Int, get(ENV, "N_STEPS", "100"))
latent_dim = 8

println("\n=== Configuration ===")
println("Weights dir: $weights_dir")
println("N samples: $n_samples")
println("X0 mean length: $X0_mean_length")
println("N steps: $n_steps")

# GPU Check
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: $(CUDA.device())")
    dev = gpu
else
    println("No CUDA - using CPU")
    dev = identity
end

# Create model
println("\n=== Creating Model ===")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)

# Load weights
base_weights_path = joinpath(weights_dir, "score_network.npz")
indel_weights_path = joinpath(weights_dir, "branching_indel_stage1.jld2")

println("Loading base weights from: $base_weights_path")
if !isfile(base_weights_path)
    error("Base weights not found at $base_weights_path")
end

println("Loading indel head weights from: $indel_weights_path")
if !isfile(indel_weights_path)
    error("Indel weights not found at $indel_weights_path")
end

load_branching_weights!(model, indel_weights_path; base_weights_path=base_weights_path)
model = dev(model)
println("Model loaded and moved to device")

# Sample
println("\n=== Generating Samples ===")
for i in 1:n_samples
    # Sample initial length from Poisson(X0_mean_length)
    initial_length = max(1, rand(Poisson(X0_mean_length)))
    println("\nSample $i: initial_length = $initial_length")

    t_start = time()
    try
        result = generate_with_branching(
            model,
            initial_length;
            nsteps=n_steps,
            latent_dim=latent_dim,
            self_cond=true,
            dev=dev,
            schedule=:linear,
            verbose=true
        )

        t_elapsed = time() - t_start
        println("  Generation completed in $(round(t_elapsed, digits=1))s")
        println("  Final length: $(result.final_length)")
        println("  CA coords shape: $(size(result.ca_coords))")
        println("  Trajectory: $(result.trajectory_lengths[1]) -> $(result.trajectory_lengths[end])")

        # Save result
        output_path = joinpath(weights_dir, "sample_$(i).jld2")
        jldopen(output_path, "w") do file
            file["ca_coords"] = Array(result.ca_coords)
            file["latents"] = Array(result.latents)
            file["final_length"] = result.final_length
            file["initial_length"] = initial_length
            file["trajectory_lengths"] = result.trajectory_lengths
        end
        println("  Saved to: $output_path")

    catch e
        println("  ERROR: $e")
        println("  Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

println("\n" * "=" ^ 60)
println("Sampling Complete")
println("=" ^ 60)
