#!/usr/bin/env julia
"""
Test GPU-accelerated sampling with Flowfusion gen() API.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Random
using CUDA

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
output_dir = joinpath(@__DIR__, "samples_gpu_110"
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

# Move score network to GPU if available (decoder stays on CPU - only runs once)
dev = CUDA.functional() ? gpu : identity
score_net_dev = score_net |> dev
println("Score network moved to GPU, decoder on CPU")

# Test parameters
L = 110
B = 3
nsteps = 400

# Helper to print CA-CA stats
function print_ca_stats(ca_coords, label)
    B = size(ca_coords, 3)
    L = size(ca_coords, 2)
    println("$label CA coords shape: ", size(ca_coords))
    for b in 1:B
        dists = [sqrt(sum((ca_coords[:, i+1, b] .- ca_coords[:, i, b]).^2)) for i in 1:(L-1)]
        mean_d = sum(dists) / length(dists)
        std_d = sqrt(sum((dists .- mean_d).^2) / length(dists))
        println("  Sample $b: Mean CA-CA = $(round(mean_d, digits=3)) nm, Std = $(round(std_d, digits=3))")
    end
end

# Helper to decode and save samples
function decode_and_save(flow_samples, decoder, output_dir, prefix; dev=identity)
    ca_coords = flow_samples[:bb_ca]
    latents = flow_samples[:local_latents]
    L, B = size(ca_coords, 2), size(ca_coords, 3)
    mask = ones(Float32, L, B)

    # Decode on device
    dec_input = Dict(:z_latent => dev(latents), :ca_coors => dev(ca_coords), :mask => dev(mask))
    dec_out = decoder(dec_input)

    # Build samples dict with CPU data
    samples = Dict(
        :ca_coords => ca_coords,
        :latents => latents,
        :all_atom_coords => cpu(dec_out[:coors]),
        :aatype => cpu(dec_out[:aatype_max]),
        :atom_mask => cpu(dec_out[:atom_mask]),
        :mask => mask
    )

    # Save to PDB
    samples_to_pdb(samples, output_dir; prefix=prefix, save_all_atom=true)

    # Print sequences
    for b in 1:B
        aatype = cpu(dec_out[:aatype_max])[:, b]
        aa_seq = join([index_to_aa(aa) for aa in aatype])
        println("  $prefix sample $b: $(aa_seq[1:min(40, length(aa_seq))])...")
    end

    return samples
end

println("\n=== Test: GPU SDE Sampling with la-proteina defaults ===")
println("CA: gt_mode=1/t, schedule=log, sc_scale_noise=0.1")
println("LL: gt_mode=tan, schedule=power, sc_scale_noise=0.1")
println("nsteps=$nsteps, t_lim_ode=0.98")

Random.seed!(42)
CUDA.functional() && CUDA.seed!(42)

@time samples = generate_with_flowfusion(score_net_dev, L, B;
    nsteps=nsteps,
    self_cond=true,
    dev=dev
)

print_ca_stats(samples[:bb_ca], "GPU SDE")

println("\nDecoding and saving samples...")
@time decode_and_save(samples, decoder, output_dir, "gpu_sde")

println("\n=== Samples saved to: $output_dir ===")
println("Files: gpu_sde_1.pdb, gpu_sde_2.pdb, gpu_sde_3.pdb")

# Memory cleanup
if CUDA.functional()
    GC.gc()
    CUDA.reclaim()
    println("\nGPU memory after cleanup: ", CUDA.available_memory() / 1e9, " GB available")
end

println("\n=== Done! ===")
