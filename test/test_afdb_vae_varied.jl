# Test VAE on AFDB samples with varied lengths
# Samples structures of different sizes to test robustness

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Statistics
using Random

Random.seed!(42)

println("=" ^ 60)
println("AFDB VAE Test - Varied Lengths")
println("=" ^ 60)

# AFDB data directory
afdb_dir = expanduser("~/shared_data/afdb_laproteina/raw")

# Get all CIF files
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)
println("Total CIF files available: $(length(cif_files))")

# Sample randomly
n_samples = 10
sample_indices = randperm(length(cif_files))[1:n_samples]
sample_files = cif_files[sample_indices]

# Create encoder and decoder
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

println("\n" * "=" ^ 60)
println("Testing on $(n_samples) random AFDB structures")
println("=" ^ 60)

results = []

for (i, filename) in enumerate(sample_files)
    filepath = joinpath(afdb_dir, filename)
    protein_id = replace(filename, ".cif" => "")

    print("[$i/$n_samples] $protein_id: ")

    try
        # Load structure
        data = load_pdb(filepath; chain_id="A")
        L = length(data[:aatype])

        # Skip very long sequences for memory
        if L > 400
            println("$L res - skipped (too long)")
            continue
        end

        # Batch and run VAE
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

        # Compute metrics
        orig_coords = batched[:coords]
        recon_coords = dec_result[:coors]
        atom_mask = batched[:atom_mask]

        # Full atom RMSD (excluding CA)
        non_ca_mask = copy(atom_mask)
        non_ca_mask[CA_INDEX, :, :] .= 0
        diff_sq = sum((orig_coords .- recon_coords).^2, dims=1)
        diff_sq = dropdims(diff_sq, dims=1)
        masked_diff_sq = diff_sq .* non_ca_mask
        full_rmsd = sqrt(sum(masked_diff_sq) / max(sum(non_ca_mask), 1)) * 10  # Convert to Å

        # Sequence accuracy
        orig_aatype = batched[:aatype][:, 1]
        recon_aatype = dec_result[:aatype_max][:, 1]
        seq_accuracy = mean(recon_aatype .== orig_aatype) * 100

        println("$L res, RMSD=$(round(full_rmsd, digits=2))Å, seq=$(round(seq_accuracy, digits=1))%")

        push!(results, (
            id=protein_id,
            length=L,
            full_rmsd=full_rmsd,
            seq_accuracy=seq_accuracy
        ))

    catch e
        println("Error: $(typeof(e))")
        continue
    end
end

# Summary
println("\n" * "=" ^ 60)
println("Summary ($(length(results)) structures)")
println("=" ^ 60)

if length(results) > 0
    lengths = [r.length for r in results]
    rmsds = [r.full_rmsd for r in results]
    accs = [r.seq_accuracy for r in results]

    println("Length range: $(minimum(lengths)) - $(maximum(lengths)) residues")
    println("RMSD: mean=$(round(mean(rmsds), digits=3))Å, min=$(round(minimum(rmsds), digits=3))Å, max=$(round(maximum(rmsds), digits=3))Å")
    println("Sequence accuracy: mean=$(round(mean(accs), digits=1))%, min=$(round(minimum(accs), digits=1))%, max=$(round(maximum(accs), digits=1))%")
end
