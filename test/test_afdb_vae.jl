# Test VAE on AFDB samples
# Loads multiple structures from AFDB and runs encoder-decoder round trip

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Statistics

println("=" ^ 60)
println("AFDB VAE End-to-End Test")
println("=" ^ 60)

# AFDB data directory
afdb_dir = expanduser("~/shared_data/afdb_laproteina/raw")

# Get a sample of CIF files
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)
println("Total CIF files available: $(length(cif_files))")

# Sample a few files for testing
n_samples = 5
sample_files = cif_files[1:n_samples]

# Create encoder
println("\nCreating encoder...")
encoder = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
encoder_weights = joinpath(@__DIR__, "..", "weights", "encoder.npz")
load_encoder_weights!(encoder, encoder_weights)

# Create decoder
println("Creating decoder...")
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
decoder_weights = joinpath(@__DIR__, "..", "weights", "decoder.npz")
load_decoder_weights!(decoder, decoder_weights)

println("\n" * "=" ^ 60)
println("Testing on $(n_samples) AFDB structures")
println("=" ^ 60)

results = []

for (i, filename) in enumerate(sample_files)
    filepath = joinpath(afdb_dir, filename)
    protein_id = replace(filename, ".cif" => "")

    println("\n[$i/$n_samples] Loading $protein_id...")

    try
        # Load structure
        data = load_pdb(filepath; chain_id="A")
        L = length(data[:aatype])
        println("  Length: $L residues")
        println("  Sequence: $(data[:sequence][1:min(30, end)])...")

        # Skip very long sequences for memory
        if L > 512
            println("  Skipping (too long for test)")
            continue
        end

        # Batch the data
        batched = batch_pdb_data([data])

        # Prepare encoder input
        encoder_batch = Dict{Symbol, Any}(
            :coords => batched[:coords],
            :coord_mask => batched[:atom_mask],
            :residue_type => batched[:aatype],
            :mask => Float32.(batched[:mask]),
        )

        # Run encoder
        enc_result = encoder(encoder_batch)

        # Prepare decoder input
        ca_coords = batched[:coords][:, CA_INDEX, :, :]  # [3, L, B]
        decoder_batch = Dict{Symbol, Any}(
            :z_latent => enc_result[:z_latent],
            :ca_coors => ca_coords,
            :mask => Float32.(batched[:mask]),
        )

        # Run decoder
        dec_result = decoder(decoder_batch)

        # Compute reconstruction metrics
        orig_coords = batched[:coords]
        recon_coords = dec_result[:coors]
        atom_mask = batched[:atom_mask]

        # Backbone RMSD (excluding CA)
        N_IDX, C_IDX, O_IDX = 1, 3, 5
        backbone_rmsds = Float64[]

        for (name, idx) in [("N", N_IDX), ("C", C_IDX), ("O", O_IDX)]
            orig_atom = orig_coords[:, idx, :, :]
            recon_atom = recon_coords[:, idx, :, :]
            mask = atom_mask[idx, :, :]

            diff_sq = sum((orig_atom .- recon_atom).^2, dims=1)
            diff_sq = dropdims(diff_sq, dims=1)
            masked_diff_sq = diff_sq .* mask
            rmsd = sqrt(sum(masked_diff_sq) / max(sum(mask), 1))
            push!(backbone_rmsds, rmsd)
        end

        # Full atom RMSD (excluding CA)
        non_ca_mask = copy(atom_mask)
        non_ca_mask[CA_INDEX, :, :] .= 0
        diff_sq = sum((orig_coords .- recon_coords).^2, dims=1)
        diff_sq = dropdims(diff_sq, dims=1)
        masked_diff_sq = diff_sq .* non_ca_mask
        full_rmsd = sqrt(sum(masked_diff_sq) / max(sum(non_ca_mask), 1))

        # Sequence accuracy
        orig_aatype = batched[:aatype][:, 1]
        recon_aatype = dec_result[:aatype_max][:, 1]
        seq_accuracy = mean(recon_aatype .== orig_aatype)

        println("  Backbone RMSD: N=$(round(backbone_rmsds[1]*10, digits=3))Å, C=$(round(backbone_rmsds[2]*10, digits=3))Å, O=$(round(backbone_rmsds[3]*10, digits=3))Å")
        println("  Full atom RMSD (excl CA): $(round(full_rmsd*10, digits=3))Å")
        println("  Sequence accuracy: $(round(seq_accuracy*100, digits=1))%")

        push!(results, (
            id=protein_id,
            length=L,
            n_rmsd=backbone_rmsds[1],
            c_rmsd=backbone_rmsds[2],
            o_rmsd=backbone_rmsds[3],
            full_rmsd=full_rmsd,
            seq_accuracy=seq_accuracy
        ))

    catch e
        println("  Error: $e")
        continue
    end
end

# Summary statistics
println("\n" * "=" ^ 60)
println("Summary Statistics ($(length(results)) structures)")
println("=" ^ 60)

if length(results) > 0
    avg_n_rmsd = mean([r.n_rmsd for r in results]) * 10
    avg_c_rmsd = mean([r.c_rmsd for r in results]) * 10
    avg_o_rmsd = mean([r.o_rmsd for r in results]) * 10
    avg_full_rmsd = mean([r.full_rmsd for r in results]) * 10
    avg_seq_acc = mean([r.seq_accuracy for r in results]) * 100

    println("Average backbone RMSD:")
    println("  N: $(round(avg_n_rmsd, digits=3)) Å")
    println("  C: $(round(avg_c_rmsd, digits=3)) Å")
    println("  O: $(round(avg_o_rmsd, digits=3)) Å")
    println("Average full atom RMSD (excl CA): $(round(avg_full_rmsd, digits=3)) Å")
    println("Average sequence accuracy: $(round(avg_seq_acc, digits=1))%")
end

println("\n" * "=" ^ 60)
println("AFDB VAE Test Complete!")
println("=" ^ 60)
