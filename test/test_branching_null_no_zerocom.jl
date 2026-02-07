#!/usr/bin/env julia
# Test: null branching inference with zero_com=false for CA
# Compare quality against the standard zero_com=true version

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
using Distributions: Beta
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)

println("=" ^ 70)
println("Branching Sampler - Null Logits, zero_com=FALSE")
println("Testing if zero_com can be disabled entirely")
println("=" ^ 70)

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

output_dir = joinpath(@__DIR__, "samples_branching_null_no_zerocom")
mkpath(output_dir)

L = 110
latent_dim = 8
nsteps = 400

for sample_idx in 1:3
    println("\n--- Sample $sample_idx ---")

    # Create processes with zero_com=FALSE for CA
    P_ca = RDNFlow(3;
        zero_com=false,  # <-- DISABLED
        schedule=:log, schedule_param=2.0f0,
        sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0,
        sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)
    P_ll = RDNFlow(latent_dim;
        zero_com=false,
        schedule=:power, schedule_param=2.0f0,
        sde_gt_mode=:tan, sde_gt_param=1.0f0,
        sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)

    branch_time_dist = Beta(2.0, 2.0)
    P_idx = NullProcess()
    P = CoalescentFlow((P_ca, P_ll, P_idx), branch_time_dist)

    # Wrapper with process-aware time transforms
    wrapper = BranchingScoreNetworkWrapper(model, latent_dim;
        self_cond=true, dev=dev, processes=(P_ca, P_ll))

    # Initial noise: plain randn (no zero-COM projection)
    ca_noise = randn(Float32, 3, L, 1)
    ll_noise = randn(Float32, latent_dim, L, 1)

    flowmask = ones(Bool, L, 1)
    padmask = ones(Bool, L, 1)
    branchmask = ones(Bool, L, 1)

    ca_state = MaskedState(ContinuousState(ca_noise), flowmask, padmask)
    ll_state = MaskedState(ContinuousState(ll_noise), flowmask, padmask)
    idx_state = MaskedState(DiscreteState(0, reshape(collect(1:L), L, 1)), flowmask, padmask)

    groupings = ones(Int, L, 1)
    X0 = BranchingState(
        (ca_state, ll_state, idx_state),
        groupings;
        flowmask=flowmask, branchmask=branchmask, padmask=padmask
    )

    ts = Float32.(range(0, 1, length=nsteps+1))

    Xt = X0
    for i in 1:nsteps
        t1, t2 = ts[i], ts[i+1]
        X1_targets, split_vec, del_vec = wrapper(t1, Xt)
        forced_split = fill(-1000f0, length(split_vec))
        forced_del = fill(-1000f0, length(del_vec))
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
    ca_coords = dropdims(ca_tensor, dims=3)
    latents = dropdims(ll_tensor, dims=3)
    final_L = size(ca_coords, 2)

    println("  Final length: $final_L (started at $L)")

    # CA-CA distances
    dists = [sqrt(sum((ca_coords[:, i+1] .- ca_coords[:, i]).^2)) for i in 1:(final_L-1)]
    mean_d = mean(dists)
    println("  Mean CA-CA: $(round(mean_d, digits=3)) nm")

    # Check COM drift
    com = mean(ca_coords, dims=2)
    println("  COM: $(round.(vec(com), digits=4))")

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

    samples_to_pdb(samples, output_dir; prefix="no_zerocom_$sample_idx", save_all_atom=true)

    aatype = dec_out[:aatype_max][:, 1]
    seq = join([index_to_aa(aa) for aa in aatype])
    println("  Sequence: $(seq[1:min(40, length(seq))])...")
end

println("\n" * "=" ^ 70)
println("Samples saved to: $output_dir")
println("=" ^ 70)
