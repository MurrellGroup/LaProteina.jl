# Run VAE on AFDB samples and save original + reconstructed PDBs

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Statistics

println("=" ^ 60)
println("VAE Comparison: Original vs Reconstructed")
println("=" ^ 60)

# Directory with input CIF files
input_dir = joinpath(@__DIR__, "afdb_vae_comparison")
output_dir = input_dir  # Save outputs in same directory

# Load models
println("\nLoading models...")
encoder = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_encoder_weights!(encoder, joinpath(@__DIR__, "..", "weights", "encoder.npz"))

decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, joinpath(@__DIR__, "..", "weights", "decoder.npz"))

# Get CIF files
cif_files = filter(f -> endswith(f, ".cif"), readdir(input_dir))
println("Found $(length(cif_files)) CIF files\n")

for filename in cif_files
    filepath = joinpath(input_dir, filename)
    protein_id = replace(filename, ".cif" => "")

    println("Processing $protein_id...")

    # Load structure
    data = load_pdb(filepath; chain_id="A")
    L = length(data[:aatype])
    println("  Length: $L residues")

    # Save original as PDB
    orig_pdb_path = joinpath(output_dir, "$(protein_id)_original.pdb")
    save_pdb(orig_pdb_path, data[:coords], data[:aatype];
             atom_mask=data[:atom_mask], chain_id="A")
    println("  Saved original: $(basename(orig_pdb_path))")

    # Run VAE
    batched = batch_pdb_data([data])

    encoder_batch = Dict{Symbol, Any}(
        :coords => batched[:coords],
        :coord_mask => batched[:atom_mask],
        :residue_type => batched[:aatype],
        :mask => Float32.(batched[:mask]),
    )
    enc_result = encoder(encoder_batch)

    ca_coords = batched[:coords][:, CA_INDEX, :, :]
    decoder_batch = Dict{Symbol, Any}(
        :z_latent => enc_result[:z_latent],
        :ca_coors => ca_coords,
        :mask => Float32.(batched[:mask]),
    )
    dec_result = decoder(decoder_batch)

    # Extract reconstructed structure
    recon_coords = dec_result[:coors][:, :, :, 1]  # [3, 37, L]
    recon_aatype = dec_result[:aatype_max][:, 1]   # [L]
    recon_atom_mask = dec_result[:atom_mask][:, :, 1] .> 0.5  # [37, L]

    # Save reconstructed as PDB
    recon_pdb_path = joinpath(output_dir, "$(protein_id)_vae_recon.pdb")
    save_pdb(recon_pdb_path, recon_coords, recon_aatype;
             atom_mask=recon_atom_mask, chain_id="A")
    println("  Saved reconstructed: $(basename(recon_pdb_path))")

    # Compute RMSD
    orig_coords = batched[:coords]
    atom_mask = batched[:atom_mask]
    non_ca_mask = copy(atom_mask)
    non_ca_mask[CA_INDEX, :, :] .= 0
    diff_sq = sum((orig_coords .- dec_result[:coors]).^2, dims=1)
    diff_sq = dropdims(diff_sq, dims=1)
    masked_diff_sq = diff_sq .* non_ca_mask
    full_rmsd = sqrt(sum(masked_diff_sq) / max(sum(non_ca_mask), 1)) * 10

    # Sequence accuracy
    orig_aatype = batched[:aatype][:, 1]
    seq_accuracy = mean(recon_aatype .== orig_aatype) * 100

    println("  RMSD (excl CA): $(round(full_rmsd, digits=3)) Å")
    println("  Sequence accuracy: $(round(seq_accuracy, digits=1))%")
    println()
end

println("=" ^ 60)
println("Output files saved to:")
println(output_dir)
println("=" ^ 60)

# List output files
println("\nGenerated PDB files:")
for f in sort(filter(f -> endswith(f, ".pdb"), readdir(output_dir)))
    println("  $f")
end
