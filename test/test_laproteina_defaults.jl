#!/usr/bin/env julia
"""
Test sampling with exact la-proteina Python defaults.
Saves all samples to PDB for inspection.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Random

Random.seed!(42)

weights_dir = joinpath(@__DIR__, "..", "weights")
score_net_path = joinpath(weights_dir, "score_network.npz")
decoder_path = joinpath(weights_dir, "decoder.npz")
output_dir = joinpath(@__DIR__, "samples_laproteina_defaults"

println("=== Loading Models ===")
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
println("Done!")

L = 100
B = 3

# Helper to decode and save samples
function decode_and_save(flow_samples, decoder, output_dir, prefix)
    ca_coords = flow_samples[:bb_ca]
    latents = flow_samples[:local_latents]
    L, B = size(ca_coords, 2), size(ca_coords, 3)
    mask = ones(Float32, L, B)

    # Decode
    dec_input = Dict(:z_latent => latents, :ca_coors => ca_coords, :mask => mask)
    dec_out = decoder(dec_input)

    # Build samples dict
    samples = Dict(
        :ca_coords => ca_coords,
        :latents => latents,
        :all_atom_coords => dec_out[:coors],
        :aatype => dec_out[:aatype_max],
        :atom_mask => dec_out[:atom_mask],
        :mask => mask
    )

    # Save to PDB
    samples_to_pdb(samples, output_dir; prefix=prefix, save_all_atom=true)

    # Print sequences
    for b in 1:B
        aatype = dec_out[:aatype_max][:, b]
        aa_seq = join([index_to_aa(aa) for aa in aatype])
        println("  $prefix sample $b: $(aa_seq[1:min(40, length(aa_seq))])...")
    end

    return samples
end

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

println("\n=== Test 1: SDE Sampling with la-proteina defaults ===")
println("CA: gt_mode=1/t, schedule=log, sc_scale_noise=0.1")
println("LL: gt_mode=tan, schedule=power, sc_scale_noise=0.1")
println("nsteps=400, t_lim_ode=0.98")

Random.seed!(100)
@time samples_sde = generate_with_flowfusion(score_net, L, B;
    nsteps=400,
    self_cond=true,
    # Using defaults which match la-proteina
)

print_ca_stats(samples_sde[:bb_ca], "SDE")

println("\nDecoding and saving SDE samples...")
@time decode_and_save(samples_sde, decoder, output_dir, "sde")

println("\n=== Test 2: ODE Sampling (no noise) ===")
Random.seed!(100)
@time samples_ode = generate_with_flowfusion(score_net, L, B;
    nsteps=400,
    self_cond=true,
    ca_sc_scale_noise=0.0,
    ll_sc_scale_noise=0.0
)

print_ca_stats(samples_ode[:bb_ca], "ODE")

println("\nDecoding and saving ODE samples...")
@time decode_and_save(samples_ode, decoder, output_dir, "ode")

println("\n=== Comparison ===")
diff = sum(abs.(samples_sde[:bb_ca] .- samples_ode[:bb_ca])) / length(samples_sde[:bb_ca])
println("Mean absolute difference SDE vs ODE: $(round(diff, digits=4))")

println("\n=== Samples saved to: $output_dir ===")
println("Files: sde_1.pdb, sde_2.pdb, sde_3.pdb, ode_1.pdb, ode_2.pdb, ode_3.pdb")
