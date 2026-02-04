#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Random
using CUDA

println("=== Loading Models ===")
weights_dir = joinpath(@__DIR__, "..", "weights")
score_net = ScoreNetwork(n_layers=14, token_dim=768, n_heads=12, latent_dim=8, dim_cond=256, t_emb_dim=256, pair_dim=256)
load_score_network_weights!(score_net, joinpath(weights_dir, "score_network.npz"))

decoder = DecoderTransformer(n_layers=12, token_dim=768, pair_dim=256, n_heads=12, dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false)
load_decoder_weights!(decoder, joinpath(weights_dir, "decoder.npz"))

score_net_gpu = score_net |> gpu
println("Models loaded")

output_dir = joinpath(@__DIR__, "samples_varying_length"
mkpath(output_dir)

for L in [250, 300, 350]
    println("\n=== Generating L=$L ===")
    Random.seed!(42)
    CUDA.seed!(42)

    @time samples = generate_with_flowfusion(score_net_gpu, L, 1; nsteps=400, self_cond=true, dev=gpu)

    ca = samples[:bb_ca]
    dists = [sqrt(sum((ca[:, i+1, 1] .- ca[:, i, 1]).^2)) for i in 1:(L-1)]
    println("  Mean CA-CA = $(round(sum(dists)/length(dists), digits=3)) nm")

    # Decode on CPU
    dec_input = Dict(:z_latent => samples[:local_latents], :ca_coors => ca, :mask => ones(Float32, L, 1))
    dec_out = decoder(dec_input)

    samples_dict = Dict(
        :ca_coords => ca,
        :latents => samples[:local_latents],
        :all_atom_coords => dec_out[:coors],
        :aatype => dec_out[:aatype_max],
        :atom_mask => dec_out[:atom_mask],
        :mask => ones(Float32, L, 1)
    )
    samples_to_pdb(samples_dict, output_dir; prefix="L$(L)", save_all_atom=true)
end

println("\nDone! Samples saved to: $output_dir")
