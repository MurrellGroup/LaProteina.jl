#!/usr/bin/env julia
# Test branching flows inference/generation

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

println("=" ^ 60)
println("Testing Branching Flows Inference")
println("=" ^ 60)

using LaProteina
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, tensor
using Flowfusion: RDNFlow, MaskedState
import Flowfusion
using Distributions: Beta
using Flux
using Random

# Include branching module
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)
latent_dim = 8

# Create small model for testing
println("\n=== Creating Model ===")
base = ScoreNetwork(
    n_layers=2, token_dim=64, pair_dim=32, n_heads=4,
    dim_cond=32, latent_dim=latent_dim, output_param=:v
)
model = BranchingScoreNetwork(base)
println("Model created")

# Test 1: Create processes
println("\n=== Test 1: Create Branching Processes ===")
P = create_branching_processes(latent_dim=latent_dim)
println("CoalescentFlow created")
println("  Base processes: $(typeof(P.P))")

# Test 2: Create initial state
println("\n=== Test 2: Create Initial State ===")
initial_L = 20
X0 = create_initial_state(initial_L, latent_dim)
println("Initial state created")
println("  Length: $(size(X0.groupings, 1))")
println("  CA shape: $(size(tensor(X0.state[1].S)))")
println("  Latent shape: $(size(tensor(X0.state[2].S)))")

# Test 3: Create wrapper and run single step
println("\n=== Test 3: Single Forward Step ===")
wrapper = BranchingScoreNetworkWrapper(model, latent_dim; self_cond=true, dev=identity)
println("Wrapper created")

t = 0.1f0
hat = wrapper(t, X0)
println("Model forward pass completed")
println("  X1_targets: $(typeof(hat[1]))")
println("  split_logits shape: $(size(hat[2]))")
println("  del_logits shape: $(size(hat[3]))")

# Test 4: Run Flowfusion step
println("\n=== Test 4: Flowfusion Step ===")
t1, t2 = 0.1f0, 0.15f0
Xt = Flowfusion.step(P, X0, hat, t1, t2)
println("Step completed")
L_after = size(Xt.groupings, 1)
println("  Length before: $initial_L")
println("  Length after: $L_after")
if L_after != initial_L
    println("  Length change: $(L_after - initial_L) (splits/deletions occurred)")
else
    println("  No length change")
end

# Test 5: Run a few more steps
println("\n=== Test 5: Multiple Steps ===")
let
    current_Xt = X0
    ts = Float32[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    reset_self_conditioning!(wrapper)

    for i in 1:(length(ts)-1)
        t1, t2 = ts[i], ts[i+1]
        hat = wrapper(t1, current_Xt)
        current_Xt = Flowfusion.step(P, current_Xt, hat, t1, t2)
        L = size(current_Xt.groupings, 1)
        println("  Step $i: t=$t1 -> $t2, L=$L")
    end
end

# Test 6: Full generation (short run)
println("\n=== Test 6: Full Generation (10 steps) ===")
result = generate_with_branching(model, 15; nsteps=10, latent_dim=latent_dim,
                                  self_cond=false, verbose=true, schedule=:linear)
println("\nGeneration completed")
println("  Initial length: 15")
println("  Final length: $(result.final_length)")
println("  CA coords shape: $(size(result.ca_coords))")
println("  Latents shape: $(size(result.latents))")
println("  Trajectory: $(result.trajectory_lengths)")

println("\n" * "=" ^ 60)
println("Branching Flows Inference Tests PASSED!")
println("=" ^ 60)
