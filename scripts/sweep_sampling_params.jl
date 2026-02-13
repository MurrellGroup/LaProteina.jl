#!/usr/bin/env julia
# Sweep sampling parameters for branching generation
# Loads model once, runs multiple configs, saves with descriptive names

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
using Distributions: Beta, Poisson
using Random
using Statistics
using JLD2

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

dev = CUDA.functional() ? gpu : identity
println("Device: $(CUDA.functional() ? "GPU" : "CPU")")

# Load model once
weights_dir = joinpath(@__DIR__, "..", "weights")
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)
weights = load(joinpath(weights_dir, "branching_full.jld2"))
if haskey(weights, "base")
    Flux.loadmodel!(model.base, weights["base"])
end
Flux.loadmodel!(model.indel_time_proj, weights["indel_time_proj"])
Flux.loadmodel!(model.split_head, weights["split_head"])
Flux.loadmodel!(model.del_head, weights["del_head"])
model = dev(model)
println("Model loaded")

# Load decoder
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, joinpath(weights_dir, "decoder.npz"))
println("Decoder loaded (CPU)")

output_root = joinpath(@__DIR__, "..", "outputs", "param_sweep")
mkpath(output_root)

latent_dim = 8
n_samples = 10

# Cosine time schedule (same as full sampling script)
step_func(t) = Float32(1 - (cos(t * pi) + 1) / 2)
step_number = 500
steps = step_func.(0f0:Float32(1/step_number):1f0)
nsteps = length(steps) - 1

# Parameter configs to sweep
configs = [
    # (name, ca_sc_noise, ll_sc_noise, ca_sc_score, ll_sc_score)
    ("base_0.1_0.1",       0.1,  0.1,  1.0, 1.0),   # Current default
    ("long_0.15_0.05",     0.15, 0.05, 1.0, 1.0),   # Python long-seq config
    ("high_0.5_0.5",       0.5,  0.5,  1.0, 1.0),   # Higher noise
    ("vhigh_1.0_1.0",      1.0,  1.0,  1.0, 1.0),   # Very high noise
    ("ode_0.0_0.0",        0.0,  0.0,  1.0, 1.0),   # Pure ODE (no SDE noise)
    ("asym_0.5_0.1",       0.5,  0.1,  1.0, 1.0),   # More CA noise, normal LL
    ("low_0.05_0.05",      0.05, 0.05, 1.0, 1.0),   # Lower noise
]

# Results table
results = []

for (config_name, ca_noise, ll_noise, ca_score, ll_score) in configs
    println("\n" * "=" ^ 70)
    println("Config: $config_name (ca_noise=$ca_noise, ll_noise=$ll_noise)")
    println("=" ^ 70)

    out_dir = joinpath(output_root, config_name)
    mkpath(out_dir)

    # Use fixed seed per config for reproducibility
    Random.seed!(42)

    lengths_init = Int[]
    lengths_final = Int[]
    ca_dists = Float64[]

    for i in 1:n_samples
        initial_length = max(1, rand(Poisson(100)))

        P = create_branching_processes(;
            latent_dim=latent_dim,
            ca_sc_scale_noise=ca_noise,
            ll_sc_scale_noise=ll_noise,
            ca_sc_scale_score=ca_score,
            ll_sc_scale_score=ll_score
        )
        P_ca = P.P[1]
        P_ll = P.P[2]

        wrapper = BranchingScoreNetworkWrapper(dev(model), latent_dim;
            self_cond=true, dev=dev, processes=(P_ca, P_ll))

        X0 = create_initial_state(P_ca, P_ll, initial_length, latent_dim)

        Xt = X0
        for j in 1:nsteps
            t1, t2 = steps[j], steps[j+1]
            hat = wrapper(t1, Xt)
            Xt = Flowfusion.step(P, Xt, hat, t1, t2)
        end

        ca_tensor = tensor(Xt.state[1].S)
        ll_tensor = tensor(Xt.state[2].S)
        ca_coords = dropdims(ca_tensor, dims=3)
        latents = dropdims(ll_tensor, dims=3)
        final_L = size(ca_coords, 2)

        # CA-CA distances
        dists = [sqrt(sum((ca_coords[:, k+1] .- ca_coords[:, k]).^2)) for k in 1:(final_L-1)]
        mean_d = final_L > 1 ? mean(dists) : NaN

        push!(lengths_init, initial_length)
        push!(lengths_final, final_L)
        push!(ca_dists, mean_d)

        println("  Sample $i: L=$initial_length -> $final_L, CA-CA=$(round(mean_d, digits=3)) nm")

        # Decode and save PDB
        ca_3d = reshape(ca_coords, 3, final_L, 1)
        ll_3d = reshape(latents, latent_dim, final_L, 1)
        mask = ones(Float32, final_L, 1)
        dec_input = Dict(:z_latent => ll_3d, :ca_coors => ca_3d, :mask => mask)
        dec_out = decoder(dec_input)
        samples = Dict(
            :ca_coords => ca_3d, :latents => ll_3d,
            :all_atom_coords => dec_out[:coors],
            :aatype => dec_out[:aatype_max],
            :atom_mask => dec_out[:atom_mask], :mask => mask
        )
        samples_to_pdb(samples, out_dir; prefix="sample_$i", save_all_atom=true)
    end

    push!(results, (
        name=config_name,
        ca_noise=ca_noise, ll_noise=ll_noise,
        mean_init=mean(lengths_init), mean_final=mean(lengths_final),
        min_final=minimum(lengths_final), max_final=maximum(lengths_final),
        std_final=round(std(lengths_final), digits=1),
        mean_caca=round(mean(ca_dists), digits=3)
    ))
end

# Print summary table
println("\n\n" * "=" ^ 100)
println("SUMMARY")
println("=" ^ 100)
println(rpad("Config", 22), rpad("CA noise", 10), rpad("LL noise", 10),
        rpad("Mean init", 10), rpad("Mean final", 12), rpad("Min", 6), rpad("Max", 6),
        rpad("Std", 8), rpad("CA-CA (nm)", 10))
println("-" ^ 100)
for r in results
    println(rpad(r.name, 22), rpad(r.ca_noise, 10), rpad(r.ll_noise, 10),
            rpad(round(r.mean_init, digits=1), 10),
            rpad(round(r.mean_final, digits=1), 12),
            rpad(r.min_final, 6), rpad(r.max_final, 6),
            rpad(r.std_final, 8),
            rpad(r.mean_caca, 10))
end
println("=" ^ 100)
println("\nSamples saved to: $output_root")
