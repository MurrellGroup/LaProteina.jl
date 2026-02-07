#!/usr/bin/env julia
# Sanity check: run branching sampler with split/del logits forced to -1000
# With all fixes (process-aware time transforms, SDE params, proper noise),
# this should produce identical quality to standard la-proteina inference.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: DecoderTransformer, load_decoder_weights!, samples_to_pdb
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, DiscreteState, tensor
using Flowfusion: MaskedState, RDNFlow
import Flowfusion
using Flux: cpu, gpu
using CUDA
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)

println("=" ^ 70)
println("Branching Sampler - Null Logits Test (fixed)")
println("Force split/del logits to -1000 => no splits/deletions")
println("Should match standard la-proteina quality (~0.38 nm CA-CA)")
println("=" ^ 70)

# GPU
dev = CUDA.functional() ? gpu : identity
println("Device: $(CUDA.functional() ? "GPU" : "CPU")")

# Load model
weights_dir = joinpath(@__DIR__, "..", "weights")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)

base_weights_path = joinpath(weights_dir, "score_network.npz")
indel_weights_path = joinpath(weights_dir, "branching_indel_stage1.jld2")
load_branching_weights!(model, indel_weights_path; base_weights_path=base_weights_path)
model = dev(model)
println("Model loaded")

# Load decoder
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, joinpath(weights_dir, "decoder.npz"))
println("Decoder loaded (CPU)")

# Output
output_dir = joinpath(@__DIR__, "samples_branching_null_logits")
mkpath(output_dir)

# Generate 3 samples
L = 110
latent_dim = 8
nsteps = 400

for sample_idx in 1:3
    println("\n--- Sample $sample_idx ---")

    # Create processes with full la-proteina defaults (SDE params included)
    P = create_branching_processes(; latent_dim=latent_dim)
    P_ca = P.P[1]
    P_ll = P.P[2]

    # Create wrapper with process-aware time transforms
    wrapper = BranchingScoreNetworkWrapper(model, latent_dim;
        self_cond=true, dev=dev, processes=(P_ca, P_ll))

    # Create initial state with proper noise (zero-COM for CA)
    X0 = create_initial_state(P_ca, P_ll, L, latent_dim)

    # Uniform time steps — internal schedule handles non-uniformity
    ts = Float32.(range(0, 1, length=nsteps+1))

    Xt = X0
    for i in 1:nsteps
        t1, t2 = ts[i], ts[i+1]

        # Get model predictions
        X1_targets, split_vec, del_vec = wrapper(t1, Xt)

        # FORCE split/del logits to -1000 => no insertions/deletions
        forced_split = fill(-1000f0, length(split_vec))
        forced_del = fill(-1000f0, length(del_vec))

        # Step forward with forced null logits
        hat = (X1_targets, forced_split, forced_del)
        Xt = Flowfusion.step(P, Xt, hat, t1, t2)

        L_current = size(Xt.groupings, 1)
        if i % 100 == 0
            println("  Step $i/$nsteps: t=$(round(t2, digits=3)), L=$L_current")
        end
    end

    # Extract final state
    ca_tensor = tensor(Xt.state[1].S)
    ll_tensor = tensor(Xt.state[2].S)
    ca_coords = dropdims(ca_tensor, dims=3)  # [3, L]
    latents = dropdims(ll_tensor, dims=3)     # [latent_dim, L]
    final_L = size(ca_coords, 2)

    println("  Final length: $final_L (started at $L)")

    # CA-CA distances
    dists = [sqrt(sum((ca_coords[:, i+1] .- ca_coords[:, i]).^2)) for i in 1:(final_L-1)]
    mean_d = mean(dists)
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

    samples_to_pdb(samples, output_dir; prefix="null_logits_$sample_idx", save_all_atom=true)

    aatype = dec_out[:aatype_max][:, 1]
    seq = join([index_to_aa(aa) for aa in aatype])
    println("  Sequence: $(seq[1:min(40, length(seq))])...")
end

println("\n" * "=" ^ 70)
println("Samples saved to: $output_dir")
println("=" ^ 70)
