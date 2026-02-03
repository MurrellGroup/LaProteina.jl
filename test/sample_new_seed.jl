using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Random
using CUDA
using Statistics

# Different seed
Random.seed!(12345)
if CUDA.functional()
    CUDA.seed!(12345)
end

weights_dir = joinpath(@__DIR__, "..", "weights")
output_dir = joinpath(@__DIR__, "samples_gpu_110")

println("Loading models...")
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=256, t_emb_dim=256, pair_dim=256
)
load_score_network_weights!(score_net, joinpath(weights_dir, "score_network.npz"))

decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, joinpath(weights_dir, "decoder.npz"))

score_net_gpu = score_net |> gpu

println("Generating L=110, B=3 with seed=12345...")
@time samples = generate_with_flowfusion(score_net_gpu, 110, 3; nsteps=400, self_cond=true, dev=gpu)

ca_coords = samples[:bb_ca]
for b in 1:3
    dists = [sqrt(sum((ca_coords[:, i+1, b] .- ca_coords[:, i, b]).^2)) for i in 1:109]
    println("Sample $b: Mean CA-CA = $(round(mean(dists), digits=3)) nm")
end

# Decode
latents = samples[:local_latents]
mask = ones(Float32, 110, 3)
dec_input = Dict(:z_latent => latents, :ca_coors => ca_coords, :mask => mask)
dec_out = decoder(dec_input)

samples_dict = Dict(
    :ca_coords => ca_coords,
    :latents => latents,
    :all_atom_coords => dec_out[:coors],
    :aatype => dec_out[:aatype_max],
    :atom_mask => dec_out[:atom_mask],
    :mask => mask
)

samples_to_pdb(samples_dict, output_dir; prefix="gpu_sde_new", save_all_atom=true)

for b in 1:3
    aatype = dec_out[:aatype_max][:, b]
    aa_seq = join([index_to_aa(aa) for aa in aatype])
    println("Sample $b: $(aa_seq[1:40])...")
end

println("\nSaved to $output_dir/gpu_sde_new_*.pdb")
