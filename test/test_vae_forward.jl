# VAE End-to-End Test
# Tests encoder -> decoder round trip with real PDB data

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Test
using Statistics

println("=" ^ 60)
println("VAE End-to-End Test")
println("=" ^ 60)

# Load a real PDB file
pdb_path = joinpath(@__DIR__, "samples_gpu", "gpu_sde_1.pdb")
println("\nLoading PDB: $pdb_path")

data = load_pdb(pdb_path; chain_id="A")
println("  Length: $(length(data[:aatype]))")
println("  Sequence: $(data[:sequence][1:min(20, end)])...")

# Batch the data (single sample)
batched = batch_pdb_data([data])
println("\nBatched data shapes:")
println("  coords: $(size(batched[:coords]))")  # [3, 37, L, B]
println("  aatype: $(size(batched[:aatype]))")  # [L, B]
println("  mask: $(size(batched[:mask]))")      # [L, B]

# ============================================================================
# Test 1: Encoder Forward Pass
# ============================================================================
println("\n" * "=" ^ 40)
println("Test 1: Encoder Forward Pass")
println("=" ^ 40)

# Create encoder
encoder = EncoderTransformer(
    n_layers=12,
    token_dim=768,
    pair_dim=256,
    n_heads=12,
    dim_cond=128,
    latent_dim=8,
    qk_ln=true,
    update_pair_repr=false
)

# Load pretrained weights
encoder_weights = joinpath(@__DIR__, "..", "weights", "encoder.npz")
if isfile(encoder_weights)
    println("Loading encoder weights...")
    load_encoder_weights!(encoder, encoder_weights)
    println("  Loaded!")
else
    println("Warning: No encoder weights found, using random initialization")
end

# Prepare encoder batch
# The encoder expects:
#   :coords or :coords_nm -> [3, 37, L, B]
#   :coord_mask -> [37, L, B]
#   :residue_type -> [L, B] (1-indexed)
#   :mask -> [L, B]
encoder_batch = Dict{Symbol, Any}(
    :coords => batched[:coords],  # Already in nm
    :coord_mask => batched[:atom_mask],
    :residue_type => batched[:aatype],
    :mask => Float32.(batched[:mask]),
)

println("\nRunning encoder...")
enc_result = encoder(encoder_batch)
println("Encoder output shapes:")
println("  mean: $(size(enc_result[:mean]))")
println("  log_scale: $(size(enc_result[:log_scale]))")
println("  z_latent: $(size(enc_result[:z_latent]))")

# ============================================================================
# Test 2: Decoder Forward Pass
# ============================================================================
println("\n" * "=" ^ 40)
println("Test 2: Decoder Forward Pass")
println("=" ^ 40)

# Create decoder
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

# Load pretrained weights
decoder_weights = joinpath(@__DIR__, "..", "weights", "decoder.npz")
if isfile(decoder_weights)
    println("Loading decoder weights...")
    load_decoder_weights!(decoder, decoder_weights)
    println("  Loaded!")
else
    println("Warning: No decoder weights found, using random initialization")
end

# Prepare decoder batch
# The decoder expects:
#   :z_latent -> [latent_dim, L, B]
#   :ca_coors -> [3, L, B] in nm
#   :mask -> [L, B]
ca_coords = extract_ca_coords(data)  # [3, L]
ca_coords_batched = reshape(ca_coords, 3, :, 1)  # [3, L, B=1]

decoder_batch = Dict{Symbol, Any}(
    :z_latent => enc_result[:z_latent],  # Use encoder output
    :ca_coors => ca_coords_batched,
    :mask => Float32.(batched[:mask]),
)

println("\nRunning decoder...")
dec_result = decoder(decoder_batch)
println("Decoder output shapes:")
println("  coors: $(size(dec_result[:coors]))")
println("  seq_logits: $(size(dec_result[:seq_logits]))")
println("  atom_mask: $(size(dec_result[:atom_mask]))")
println("  aatype_max: $(size(dec_result[:aatype_max]))")

# ============================================================================
# Test 3: Reconstruction Quality
# ============================================================================
println("\n" * "=" ^ 40)
println("Test 3: Reconstruction Analysis")
println("=" ^ 40)

# Compare original and reconstructed coordinates
orig_coords = batched[:coords]  # [3, 37, L, B]
recon_coords = dec_result[:coors]  # [3, 37, L, B]
atom_mask = batched[:atom_mask]  # [37, L, B]

# NOTE: CA coordinates (index 2) are INPUT to decoder, not predicted
# So CA RMSD will be 0 by design. Compare other backbone atoms instead.

# Backbone atom indices (1-indexed): N=1, CA=2, C=3, O=5
N_IDX, C_IDX, O_IDX = 1, 3, 5

# Compute backbone RMSD (excluding CA which is input)
println("\nBackbone reconstruction (excluding CA, which is input):")
for (name, idx) in [("N", N_IDX), ("C", C_IDX), ("O", O_IDX)]
    orig_atom = orig_coords[:, idx, :, :]
    recon_atom = recon_coords[:, idx, :, :]
    mask = atom_mask[idx, :, :]

    # Compute masked RMSD
    diff_sq = sum((orig_atom .- recon_atom).^2, dims=1)  # [1, L, B]
    diff_sq = dropdims(diff_sq, dims=1)  # [L, B]
    masked_diff_sq = diff_sq .* mask
    rmsd = sqrt(sum(masked_diff_sq) / max(sum(mask), 1))
    println("  $name RMSD: $(round(rmsd, digits=4)) nm")
end

# Full atom37 RMSD (excluding CA)
non_ca_mask = copy(atom_mask)
non_ca_mask[CA_INDEX, :, :] .= 0  # Exclude CA
diff_sq = sum((orig_coords .- recon_coords).^2, dims=1)  # [1, 37, L, B]
diff_sq = dropdims(diff_sq, dims=1)  # [37, L, B]
masked_diff_sq = diff_sq .* non_ca_mask
full_rmsd = sqrt(sum(masked_diff_sq) / max(sum(non_ca_mask), 1))
println("\nFull atom37 RMSD (excluding CA input): $(round(full_rmsd, digits=4)) nm")

# Compare amino acid predictions
orig_aatype = batched[:aatype]  # [L, B]
recon_aatype = dec_result[:aatype_max][:, 1]  # [L] - decoder already computes argmax

accuracy = mean(recon_aatype .== orig_aatype[:, 1])
println("\nAmino acid prediction:")
println("  accuracy: $(round(accuracy * 100, digits=1))%")

# Sample sequence comparison
orig_seq = data[:sequence]
recon_seq = join([index_to_aa(aa) for aa in recon_aatype])
println("  Original:      $(orig_seq[1:min(30, end)])...")
println("  Reconstructed: $(recon_seq[1:min(30, end)])...")

# ============================================================================
# Run Tests
# ============================================================================
println("\n" * "=" ^ 40)
println("Test Results")
println("=" ^ 40)

@testset "VAE Forward Pass Tests" begin
    @testset "Encoder" begin
        @test size(enc_result[:mean]) == (8, 100, 1)
        @test size(enc_result[:log_scale]) == (8, 100, 1)
        @test size(enc_result[:z_latent]) == (8, 100, 1)
        @test all(isfinite.(enc_result[:mean]))
        @test all(isfinite.(enc_result[:log_scale]))
    end

    @testset "Decoder" begin
        @test size(dec_result[:coors]) == (3, 37, 100, 1)
        @test size(dec_result[:seq_logits]) == (20, 100, 1)
        @test all(isfinite.(dec_result[:coors]))
        @test all(isfinite.(dec_result[:seq_logits]))
    end

    @testset "Reconstruction" begin
        # With pretrained weights, we expect reasonable reconstruction
        # Note: These tests may fail with random initialization
        # RMSD bounds are loose since VAE reconstruction is inherently lossy
        @test full_rmsd < 2.0  # Full atom RMSD < 2nm
        @test accuracy > 0.5   # At least 50% sequence accuracy
    end
end

println("\n" * "=" ^ 60)
println("VAE End-to-End Test Complete!")
println("=" ^ 60)
