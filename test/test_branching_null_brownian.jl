#!/usr/bin/env julia
# Test: null branching inference using BrownianMotion + FProcess instead of RDNFlow
#
# NOTE: CA-CA distances look OK (~0.39 nm) but macro-level fold geometry is poor.
# Need schedule tweaks to get closer to RDNFlow quality without additional finetuning,
# otherwise the starting point may be too far from what the current model expects.
#
# Replaces the RDNFlow SDE sampler with Flowfusion's BrownianMotion process wrapped
# in FProcess to apply the same schedule transforms. Split/del logits are forced to
# -1000 (null branching) so this tests purely the continuous state dynamics.
#
# Design rationale:
#
# The la-proteina RDNFlow SDE in τ-space is:
#   dx = [v + g(τ)·score_scale·score] dτ + √(2·g(τ)·noise_scale) dW
# where g(τ)=1/(τ+ε) for CA and g(τ)=tan-based for LL, with noise_scale=0.1.
#
# BrownianMotion bridge gives:
#   mean = exact same interpolation as the ODE (via FProcess time transform)
#   variance = v · (F(s₂)-F(s₁)) · (1-F(s₂)) / (1-F(s₁))
#
# By matching F_ca(s) = log_schedule(s) and F_ll(s) = s², the mean trajectory
# is identical to the RDNFlow ODE. The BM variance `v` controls path wandering,
# analogous to sc_scale_noise in the SDE:
#   - Noise peaks early (where 1-F(s) is large) and vanishes as s→1
#   - For CA with log schedule: F(0.5)≈0.91, so most noise is at very early s
#   - For LL with power schedule: F(0.5)=0.25, noise peaks at s≈0.7
#
# Choice of v=0.1:
#   Peak bridge σ per dim ≈ √(v·F·(1-F)) ≈ 0.16 at the peak
#   This is modest relative to data scale (~1-2 nm for CA, ~1 for latents),
#   comparable to the RDNFlow SDE with sc_scale_noise=0.1.
#   Can be tuned: lower → closer to ODE, higher → more exploration.
#
# Usage: julia -t 4 test/test_branching_null_brownian.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: DecoderTransformer, load_decoder_weights!, samples_to_pdb
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, DiscreteState, BrownianMotion, tensor
using Flowfusion: MaskedState, FProcess, schedule_transform
import Flowfusion
using Flux: cpu, gpu
using CUDA
using Distributions: Beta
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

# ============================================================================
# Make schedule_transform work with FProcess so the BranchingScoreNetworkWrapper
# can apply per-modality time transforms for model conditioning.
# ============================================================================
Flowfusion.schedule_transform(P::FProcess, t::T) where T<:Real = T(P.F(t))
Flowfusion.schedule_transform(P::FProcess, t::AbstractArray) = P.F.(t)

# ============================================================================
# Schedule transform functions matching la-proteina's RDNFlow schedules
# ============================================================================

# CA: log schedule with param=2.0
#   τ(s) = (1 - 10^(-p·s)) / (1 - 10^(-p))  where p=2.0
# Reaches τ=0.5 at s≈0.089, τ=0.9 at s≈0.48 — fast early interpolation
const LOG_DENOM = 1.0 - 10.0^(-2.0)
F_ca(s) = (1.0 - 10.0^(-2.0 * s)) / LOG_DENOM

# LL: power schedule with param=2.0
#   τ(s) = s^p  where p=2.0
# Reaches τ=0.5 at s≈0.71, τ=0.9 at s≈0.95 — slow early, fast late
F_ll(s) = s^2

# ============================================================================
# BrownianMotion variance selection
# ============================================================================
# v=0.1 gives peak per-dimension σ ≈ 0.16 at the bridge variance maximum.
# For CA (log schedule): max variance at very early s (F≈0.5), σ_per_dim≈0.16
# For LL (power schedule): max variance at s≈0.58 (F≈0.33), σ_per_dim≈0.15
#
# The RDNFlow SDE injects total noise variance (integrated over path):
#   CA: ∫ 2·0.1/(τ+0.01) dτ from 0 to 0.98 ≈ 0.92 (but score corrects most)
#   LL: similar magnitude
# With BM bridge, the score correction is implicit in endpoint conditioning.
# v=0.1 is a conservative starting point — increase for more diversity.
const BM_VARIANCE = 0.1f0

Random.seed!(42)

println("=" ^ 70)
println("Branching Sampler - Null Logits, BrownianMotion + FProcess")
println("Replacing RDNFlow SDE with BrownianMotion(v=$(BM_VARIANCE))")
println("Schedule transforms: CA=log(p=2), LL=power(p=2)")
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

output_dir = joinpath(@__DIR__, "samples_branching_null_brownian")
mkpath(output_dir)

L = 110
latent_dim = 8
nsteps = 400

for sample_idx in 1:3
    println("\n--- Sample $sample_idx ---")

    # ====================================================================
    # Create BrownianMotion processes wrapped in FProcess
    # The FProcess applies the schedule transform, making the BM bridge
    # interpolation follow the same trajectory as the RDNFlow ODE.
    # ====================================================================
    P_ca = FProcess(BrownianMotion(BM_VARIANCE), F_ca)
    P_ll = FProcess(BrownianMotion(BM_VARIANCE), F_ll)

    # NullProcess for index tracking (unchanged from RDNFlow version)
    P_idx = NullProcess()

    # CoalescentFlow wrapping all three processes
    branch_time_dist = Beta(1.0, 2.0)
    P = CoalescentFlow((P_ca, P_ll, P_idx), branch_time_dist)

    # Wrapper with process-aware time transforms for model conditioning.
    # The schedule_transform dispatch for FProcess (defined above) ensures
    # the model sees τ_ca = F_ca(s) and τ_ll = F_ll(s), matching training.
    wrapper = BranchingScoreNetworkWrapper(model, latent_dim;
        self_cond=true, dev=dev, processes=(P_ca, P_ll))

    # Initial noise: plain randn (no zero-COM needed with BrownianMotion)
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

    # Uniform time steps — FProcess handles the non-uniform scheduling
    ts = Float32.(range(0, 1, length=nsteps+1))

    Xt = X0
    for i in 1:nsteps
        t1, t2 = ts[i], ts[i+1]

        # Get model predictions (wrapper applies schedule_transform via FProcess)
        X1_targets, split_vec, del_vec = wrapper(t1, Xt)

        # FORCE split/del logits to -1000 => no insertions/deletions
        forced_split = fill(-1000f0, length(split_vec))
        forced_del = fill(-1000f0, length(del_vec))

        # Step forward using BrownianMotion bridge (via CoalescentFlow.step)
        # This bridges from Xt to predicted X1 between F(t1) and F(t2),
        # adding BrownianMotion noise proportional to:
        #   σ² = v · (F(t2)-F(t1)) · (1-F(t2)) / (1-F(t1))
        hat = (X1_targets, forced_split, forced_del)
        Xt = Flowfusion.step(P, Xt, hat, t1, t2)

        L_current = size(Xt.groupings, 1)
        if i % 100 == 0
            # Show both raw s and transformed τ for each modality
            τ_ca = F_ca(Float64(t2))
            τ_ll = F_ll(Float64(t2))
            println("  Step $i/$nsteps: s=$(round(t2, digits=3)), τ_ca=$(round(τ_ca, digits=3)), τ_ll=$(round(τ_ll, digits=3)), L=$L_current")
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

    samples_to_pdb(samples, output_dir; prefix="brownian_$sample_idx", save_all_atom=true)

    aatype = dec_out[:aatype_max][:, 1]
    seq = join([index_to_aa(aa) for aa in aatype])
    println("  Sequence: $(seq[1:min(40, length(seq))])...")
end

println("\n" * "=" ^ 70)
println("Samples saved to: $output_dir")
println("=" ^ 70)
