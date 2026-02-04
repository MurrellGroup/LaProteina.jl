#!/usr/bin/env julia
# Sample from trained Branching Flows model
#
# Usage:
#   julia scripts/sample_branching.jl
#
# Environment variables:
#   WEIGHTS_DIR - Directory containing trained weights
#   OUTPUT_DIR - Directory to save PDB files (default: weights_dir)
#   N_SAMPLES - Number of samples to generate (default: 1)
#   X0_MEAN_LENGTH - Mean of Poisson distribution for initial length (default: 100)
#   N_STEPS - Number of integration steps (default: 100)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: DecoderTransformer, load_decoder_weights!, samples_to_pdb
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
output_dir = get(ENV, "OUTPUT_DIR", weights_dir)
n_samples = parse(Int, get(ENV, "N_SAMPLES", "1"))
X0_mean_length = parse(Int, get(ENV, "X0_MEAN_LENGTH", "100"))
n_steps = parse(Int, get(ENV, "N_STEPS", "100"))
latent_dim = 8

println("\n=== Configuration ===")
println("Weights dir: $weights_dir")
println("Output dir: $output_dir")
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

# Create score network
println("\n=== Creating Score Network ===")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)

# Load score network weights
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
println("Score network loaded and moved to device")

# Create and load decoder
println("\n=== Creating Decoder ===")
decoder = DecoderTransformer(
    n_layers=12,
    token_dim=768,
    pair_dim=256,
    n_heads=12,
    dim_cond=128,
    latent_dim=latent_dim,
    qk_ln=true,
    update_pair_repr=false
)

decoder_weights_path = joinpath(weights_dir, "decoder.npz")
println("Loading decoder weights from: $decoder_weights_path")
if !isfile(decoder_weights_path)
    error("Decoder weights not found at $decoder_weights_path")
end

load_decoder_weights!(decoder, decoder_weights_path)
decoder = dev(decoder)
println("Decoder loaded and moved to device")

# Sample
println("\n=== Generating Samples ===")
mkpath(output_dir)

for i in 1:n_samples
    # Sample initial length from Poisson(X0_mean_length)
    initial_length = max(1, rand(Poisson(X0_mean_length)))
    println("\nSample $i: initial_length = $initial_length")

    t_start = time()
    try
        # Generate CA coords and latents using branching flows
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

        t_gen = time() - t_start
        println("  Flow generation completed in $(round(t_gen, digits=1))s")
        println("  Final length: $(result.final_length)")
        println("  Trajectory: $(result.trajectory_lengths[1]) -> $(result.trajectory_lengths[end])")

        # Decode to get all-atom structure (on CPU to avoid scalar indexing issues)
        println("  Running decoder...")
        L = result.final_length
        ca_coords = reshape(Array(result.ca_coords), 3, L, 1)  # [3, L, 1]
        latents = reshape(Array(result.latents), latent_dim, L, 1)  # [latent_dim, L, 1]
        mask = ones(Float32, L, 1)

        dec_input = Dict(
            :z_latent => latents,
            :ca_coors => ca_coords,
            :mask => mask
        )
        decoder_cpu = cpu(decoder)
        dec_out = decoder_cpu(dec_input)

        # Prepare samples dict for PDB export (already on CPU)
        samples = Dict(
            :ca_coords => ca_coords,
            :latents => latents,
            :seq_logits => dec_out[:seq_logits],
            :all_atom_coords => dec_out[:coors],
            :aatype => dec_out[:aatype_max],
            :atom_mask => dec_out[:atom_mask],
            :mask => mask
        )

        # Save PDB
        pdb_path = joinpath(output_dir, "sample_$(i).pdb")
        samples_to_pdb(samples, output_dir; prefix="sample_$(i)", save_all_atom=true)

        t_total = time() - t_start
        println("  Total time: $(round(t_total, digits=1))s")
        println("  Saved PDB to: $pdb_path")

        # Also save JLD2 with full data
        jld2_path = joinpath(output_dir, "sample_$(i).jld2")
        jldopen(jld2_path, "w") do file
            file["ca_coords"] = ca_coords
            file["latents"] = latents
            file["all_atom_coords"] = dec_out[:coors]
            file["aatype"] = dec_out[:aatype_max]
            file["final_length"] = result.final_length
            file["initial_length"] = initial_length
            file["trajectory_lengths"] = result.trajectory_lengths
        end
        println("  Saved JLD2 to: $jld2_path")

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
