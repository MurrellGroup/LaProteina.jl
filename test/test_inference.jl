#!/usr/bin/env julia
"""
Test inference: generate protein samples and save to PDB.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Paths
weights_dir = joinpath(@__DIR__, "..", "weights")
score_net_path = joinpath(weights_dir, "score_network.npz")
decoder_path = joinpath(weights_dir, "decoder.npz")
output_dir = joinpath(@__DIR__, "samples"

println("=== Creating ScoreNetwork ===")
score_net = ScoreNetwork(
    n_layers=14,
    token_dim=768,
    n_heads=12,
    latent_dim=8,
    dim_cond=256,
    t_emb_dim=256,
    pair_dim=256
)
println("Loading ScoreNetwork weights...")
load_score_network_weights!(score_net, score_net_path)
println("Done!")

println("\n=== Creating Decoder ===")
decoder = DecoderTransformer(
    n_layers=12,
    token_dim=768,
    pair_dim=256,
    n_heads=12,
    dim_cond=128,
    latent_dim=8,
    qk_ln=true,
    update_pair_repr=false
)
println("Loading Decoder weights...")
load_decoder_weights!(decoder, decoder_path)
println("Done!")

# Generate a sample
L = 50  # 50 residue protein
B = 1   # 1 sample
nsteps = 50  # Fewer steps for faster testing

println("\n=== Running Flow Matching Simulation ===")
println("Generating $B sample(s) of length $L with $nsteps steps...")

@time flow_samples = full_simulation(score_net, L, B;
    nsteps=nsteps,
    latent_dim=8,
    self_cond=true,
    schedule_mode=:power,
    schedule_p=2.0,
    sampling_mode=:vf,
    center_ca=true
)

ca_coords = flow_samples[:bb_ca]
latents = flow_samples[:local_latents]

println("CA coords shape: ", size(ca_coords))
println("Latents shape: ", size(latents))

# Compute CA-CA distances to verify protein structure
println("\n=== Checking CA-CA distances ===")
ca_1d = ca_coords[:, :, 1]  # [3, L]
dists = Float32[]
for i in 1:(L-1)
    d = sqrt(sum((ca_1d[:, i+1] .- ca_1d[:, i]).^2))
    push!(dists, d)
end
println("CA-CA distances (first 10): ", round.(dists[1:min(10, length(dists))], digits=3))
println("Mean CA-CA distance: ", round(sum(dists) / length(dists), digits=3), " nm")
println("(Expected: ~0.38 nm for ideal protein)")

# Decode to get all-atom structure
println("\n=== Decoding to All-Atom Structure ===")
mask = ones(Float32, L, B)
dec_input = Dict(
    :z_latent => latents,
    :ca_coors => ca_coords,
    :mask => mask
)
@time dec_out = decoder(dec_input)

println("Decoded sequence logits shape: ", size(dec_out[:seq_logits]))
println("Decoded coords shape: ", size(dec_out[:coors]))
println("Predicted aatype (first 10): ", dec_out[:aatype_max][1:min(10, L), 1])

# Get amino acid sequence
aatype = dec_out[:aatype_max][:, 1]
aa_seq = join([index_to_aa(aa) for aa in aatype])
println("Predicted sequence: ", aa_seq)

# Build full samples dict
samples = Dict(
    :ca_coords => ca_coords,
    :latents => latents,
    :seq_logits => dec_out[:seq_logits],
    :all_atom_coords => dec_out[:coors],
    :aatype => dec_out[:aatype_max],
    :atom_mask => dec_out[:atom_mask],
    :mask => mask
)

# Save to PDB
println("\n=== Saving to PDB ===")
samples_to_pdb(samples, output_dir; prefix="test_sample", save_all_atom=true)

println("\n=== Done! ===")
println("Samples saved to: ", output_dir)
