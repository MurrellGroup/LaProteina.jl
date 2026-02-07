#!/usr/bin/env julia
"""
Test GPU sampling using process-internal schedules.
RDNFlow carries schedule=:log/:power internally.
The model wrapper applies schedule_transform per modality to get the correct
conditioning times. Time steps passed to gen() are uniform.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: cpu
using Random
using CUDA
using Flux: gpu
import Flowfusion
using Flowfusion: RDNFlow, gen, schedule_transform, sample_rdn_noise
using ForwardBackward: ContinuousState, tensor

# Check GPU availability
println("=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: ", CUDA.device())
    println("Memory: ", CUDA.available_memory() / 1e9, " GB available")
else
    println("CUDA not functional, running on CPU")
end

# Paths
weights_dir = joinpath(@__DIR__, "..", "weights")
score_net_path = joinpath(weights_dir, "score_network.npz")
decoder_path = joinpath(weights_dir, "decoder.npz")
output_dir = joinpath(@__DIR__, "samples_internal_schedule")
mkpath(output_dir)

println("\n=== Loading Models ===")
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=256, t_emb_dim=256, pair_dim=256
)
load_score_network_weights!(score_net, score_net_path)

decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, decoder_path)
println("Models loaded on CPU")

dev = CUDA.functional() ? gpu : identity
score_net_dev = score_net |> dev
println("Score network moved to GPU, decoder on CPU")

# Parameters
L = 110
B = 3
nsteps = 400
latent_dim = 8

# Create RDNFlow with INTERNAL schedules + SDE params (la-proteina defaults)
P_ca = RDNFlow(3;
    zero_com=true,
    schedule=:log, schedule_param=2.0f0,
    sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0
)
P_ll = RDNFlow(latent_dim;
    zero_com=false,
    schedule=:power, schedule_param=2.0f0,
    sde_gt_mode=:tan, sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0
)
P = (P_ca, P_ll)

println("\n=== Process Config ===")
println("P_ca: schedule=:log, schedule_param=2.0, sc_scale_noise=0.1, t_lim_ode=0.98")
println("P_ll: schedule=:power, schedule_param=2.0, sc_scale_noise=0.1, t_lim_ode=0.98")

# Verify schedule transforms match expected values
println("\nSchedule transform spot check:")
println("  CA  schedule_transform(0.5) = $(schedule_transform(P_ca, 0.5f0))")
println("  LL  schedule_transform(0.5) = $(schedule_transform(P_ll, 0.5f0))")

# UNIFORM time steps — internal schedule_transform handles non-uniformity
steps = Float32.(range(0, 1, length=nsteps+1))
println("Time steps: uniform [0, 1], $(length(steps)) points")

# Sample initial noise using Flowfusion's noise sampler (handles zero-COM)
Random.seed!(42)
CUDA.functional() && CUDA.seed!(42)

x0_ca = sample_rdn_noise(P_ca, L, B)
x0_ll = sample_rdn_noise(P_ll, L, B)
X0 = (ContinuousState(x0_ca), ContinuousState(x0_ll))

# Model wrapper with process-aware time transforms
model = MutableScoreNetworkWrapper(score_net_dev, L, B;
    self_cond=true, dev=dev, processes=(P_ca, P_ll))

println("\n=== Running gen() with internal schedules + process-aware wrapper ===")
@time X_final = gen(P, X0, model, steps)

# Extract results
ca_coords = tensor(X_final[1])
ll_coords = tensor(X_final[2])

println("CA coords shape: ", size(ca_coords))
for b in 1:B
    dists = [sqrt(sum((ca_coords[:, i+1, b] .- ca_coords[:, i, b]).^2)) for i in 1:(L-1)]
    mean_d = sum(dists) / length(dists)
    std_d = sqrt(sum((dists .- mean_d).^2) / length(dists))
    println("  Sample $b: Mean CA-CA = $(round(mean_d, digits=3)) nm, Std = $(round(std_d, digits=3))")
end

# Decode and save
println("\nDecoding and saving samples...")
mask = ones(Float32, L, B)
dec_input = Dict(:z_latent => ll_coords, :ca_coors => ca_coords, :mask => mask)
@time dec_out = decoder(dec_input)

samples = Dict(
    :ca_coords => ca_coords,
    :latents => ll_coords,
    :all_atom_coords => cpu(dec_out[:coors]),
    :aatype => cpu(dec_out[:aatype_max]),
    :atom_mask => cpu(dec_out[:atom_mask]),
    :mask => mask
)

samples_to_pdb(samples, output_dir; prefix="internal_sched", save_all_atom=true)

for b in 1:B
    aatype = cpu(dec_out[:aatype_max])[:, b]
    aa_seq = join([index_to_aa(aa) for aa in aatype])
    println("  Sample $b: $(aa_seq[1:min(40, length(aa_seq))])...")
end

println("\n=== Samples saved to: $output_dir ===")
println("Files: internal_sched_1.pdb, internal_sched_2.pdb, internal_sched_3.pdb")

if CUDA.functional()
    GC.gc()
    CUDA.reclaim()
end

println("\n=== Done! ===")
