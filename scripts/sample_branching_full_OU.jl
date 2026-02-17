#!/usr/bin/env julia
# Sample from the fully fine-tuned branching model (OUBridgeExpVar processes)
# with cosine time steps. Loads final weights from the most recent training run.
#
# Usage: julia -t 4 scripts/sample_branching_full_OU.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: DecoderTransformer, load_decoder_weights!, samples_to_pdb
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, DiscreteState, tensor, OUBridgeExpVar
using Flowfusion: MaskedState
import Flowfusion
using Flux: cpu, gpu
using CUDA
using Distributions: Beta, Poisson
using Random
using Statistics
using JLD2

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)

println("=" ^ 70)
println("Branching Full Model Sampling - OUBridgeExpVar - Cosine Steps")
println("=" ^ 70)

dev = CUDA.functional() ? gpu : identity
println("Device: $(CUDA.functional() ? "GPU" : "CPU")")

# Load model with full fine-tuned weights
weights_dir = joinpath(@__DIR__, "..", "weights")
safe_models_dir = get(ENV, "SAFE_MODELS_DIR", "/home/claudey/safe_models")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)

# Load the fully fine-tuned weights (includes base)
# Prefer safe_models dir (validated weights), fall back to weights dir
safe_weights_path = joinpath(safe_models_dir, "branching_OU_100k_20260216.jld2")
full_weights_path = isfile(safe_weights_path) ? safe_weights_path : joinpath(weights_dir, "branching_full.jld2")
println("Loading weights from: $full_weights_path")
weights = load(full_weights_path)
if haskey(weights, "base")
    Flux.loadmodel!(model.base, weights["base"])
    println("  Loaded fine-tuned base weights")
else
    # Fallback: load pretrained base + indel heads separately
    load_score_network_weights!(model.base, joinpath(weights_dir, "score_network.npz"))
    println("  Loaded pretrained base weights (no fine-tuned base in checkpoint)")
end
if haskey(weights, "indel_time_proj")
    Flux.loadmodel!(model.indel_time_proj, weights["indel_time_proj"])
end
if haskey(weights, "split_head")
    Flux.loadmodel!(model.split_head, weights["split_head"])
end
if haskey(weights, "del_head")
    Flux.loadmodel!(model.del_head, weights["del_head"])
end
model = dev(model)
println("Model loaded")

# Load decoder
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, joinpath(weights_dir, "decoder.npz"))
println("Decoder loaded (CPU)")

output_dir = joinpath(@__DIR__, "samples_branching_full_OU")
mkpath(output_dir)

latent_dim = 8
n_samples = 10

# Cosine time schedule — denser steps near t=0 and t=1
step_func(t) = Float32(1 - (cos(t * pi) + 1) / 2)
step_number = 500
steps = step_func.(0f0:Float32(1/step_number):1f0)
nsteps = length(steps) - 1

println("Steps: $nsteps (cosine schedule)")
println("First 5 steps: $(round.(steps[1:5], digits=4))")
println("Last 5 steps: $(round.(steps[end-4:end], digits=4))")
println()

# OUBridgeExpVar processes (same as training)
P_ca = OUBridgeExpVar(100f0, 150f0, 0.000000001f0, dec = -3f0)
P_ll = OUBridgeExpVar(100f0, 50f0, 0.000000001f0, dec = -0.1f0)
P_idx = NullProcess()
branch_time_dist = Beta(1.0, 2.0)
P = CoalescentFlow((P_ca, P_ll, P_idx), branch_time_dist)
println("Processes: OUBridgeExpVar (CA: θ=100, v0=150, v1=1e-9, dec=-3; LL: θ=100, v0=50, v1=1e-9, dec=-0.1)")

# Create wrapper once (reuse across samples, reset self-conditioning between)
wrapper = BranchingScoreNetworkWrapper(model, latent_dim;
    self_cond=true, dev=dev, processes=nothing)

for sample_idx in 1:n_samples
    println("--- Sample $sample_idx ---")

    initial_length = max(1, rand(Poisson(100)))

    # Reset self-conditioning for new sample
    reset_self_conditioning!(wrapper)

    # Create initial state (randn noise, no process-specific sampling)
    X0 = create_initial_state(initial_length, latent_dim)

    # Run generation with cosine steps
    Xt = X0
    for i in 1:nsteps
        t1, t2 = steps[i], steps[i+1]

        # Guard against empty or excessively large proteins
        L_current = size(Xt.groupings, 1)
        if L_current == 0
            println("  WARNING: Protein reached L=0 at step $i, skipping")
            break
        elseif L_current > 400
            println("  WARNING: Protein grew to L=$L_current at step $i (>400), skipping to avoid OOM")
            break
        end

        hat = wrapper(t1, Xt)
        Xt = Flowfusion.step(P, Xt, hat, t1, t2)

        L_current = size(Xt.groupings, 1)
        if i % 100 == 0
            println("  Step $i/$nsteps: t=$(round(t2, digits=3)), L=$L_current")
        end
    end

    # Extract final state
    ca_tensor = tensor(Xt.state[1].S)
    ll_tensor = tensor(Xt.state[2].S)
    ca_coords = dropdims(ca_tensor, dims=3)
    latents = dropdims(ll_tensor, dims=3)
    final_L = size(ca_coords, 2)

    println("  Final length: $final_L (started at $initial_length)")

    if final_L == 0
        println("  Skipping empty protein")
        println()
        continue
    end

    # CA-CA distances
    dists = [sqrt(sum((ca_coords[:, i+1] .- ca_coords[:, i]).^2)) for i in 1:(final_L-1)]
    mean_d = final_L > 1 ? mean(dists) : 0.0
    println("  Mean CA-CA: $(round(mean_d, digits=3)) nm")

    # Decode and save
    ca_3d = reshape(ca_coords, 3, final_L, 1)
    ll_3d = reshape(latents, latent_dim, final_L, 1)
    mask = ones(Float32, final_L, 1)

    dec_input = Dict(:z_latent => ll_3d, :ca_coors => ca_3d, :mask => mask)
    dec_out = decoder(dec_input)

    samples = Dict(
        :ca_coords => ca_3d,
        :latents => ll_3d,
        :all_atom_coords => dec_out[:coors],
        :aatype => dec_out[:aatype_max],
        :atom_mask => dec_out[:atom_mask],
        :mask => mask
    )

    samples_to_pdb(samples, output_dir; prefix="OU_cosine_$sample_idx", save_all_atom=true)

    aatype = dec_out[:aatype_max][:, 1]
    seq = join([index_to_aa(aa) for aa in aatype])
    println("  Sequence: $(seq[1:min(40, length(seq))])...")
    println()

    # Clean up GPU memory between samples
    GC.gc()
    CUDA.reclaim()
end

println("=" ^ 70)
println("Samples saved to: $output_dir")
println("=" ^ 70)
