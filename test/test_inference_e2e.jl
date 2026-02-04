#!/usr/bin/env julia
# End-to-end inference test for JuProteina
# Generates samples using flow matching + VAE decoder and saves as PDB

using Pkg
Pkg.activate(".")

using LaProteina
using Random
using Statistics

function test_full_inference_pipeline()
    println("=" ^ 60)
    println("JuProteina End-to-End Inference Test")
    println("=" ^ 60)

    Random.seed!(42)

    # Parameters
    L = 50           # Sequence length
    B = 2            # Number of samples
    nsteps = 50      # Inference steps (lower for quick test)
    latent_dim = 8

    println("\nParameters:")
    println("  Sequence length: $L")
    println("  Batch size: $B")
    println("  Inference steps: $nsteps")
    println("  Latent dimension: $latent_dim")

    # Create score network with smaller config for testing
    println("\nCreating ScoreNetwork...")
    score_net = ScoreNetwork(
        n_layers=2,           # Fewer layers for quick test
        token_dim=128,        # Smaller dimension
        pair_dim=32,
        n_heads=4,
        dim_cond=64,
        latent_dim=latent_dim,
        qk_ln=true,
        update_pair_repr=false,
        output_param=:v
    )
    println("  ✓ ScoreNetwork created")

    # Create decoder with smaller config for testing
    println("\nCreating DecoderTransformer...")
    decoder = DecoderTransformer(
        n_layers=2,           # Fewer layers for quick test
        token_dim=128,        # Smaller dimension
        pair_dim=32,
        n_heads=4,
        dim_cond=64,
        latent_dim=latent_dim,
        qk_ln=false,
        update_pair_repr=false
    )
    println("  ✓ DecoderTransformer created")

    # Test full_simulation only (without decoder)
    println("\n--- Testing full_simulation ---")

    mask = ones(Float32, L, B)

    flow_samples = full_simulation(
        score_net, L, B;
        nsteps=nsteps,
        latent_dim=latent_dim,
        self_cond=true,
        schedule_mode=:power,
        schedule_p=2.0,
        sampling_mode=:vf,  # ODE mode
        center_ca=true,
        mask=mask
    )

    ca_coords = flow_samples[:bb_ca]
    latents = flow_samples[:local_latents]

    println("  Output shapes:")
    println("    CA coordinates: $(size(ca_coords))")
    println("    Latents: $(size(latents))")

    # Check CA is zero-centered
    com = dropdims(mean(ca_coords; dims=2); dims=2)
    max_com = maximum(abs.(com))
    println("    Max center of mass deviation: $(max_com)")
    @assert max_com < 0.1 "Center of mass not zero!"
    println("  ✓ full_simulation works correctly")

    # Test decoder
    println("\n--- Testing Decoder ---")

    dec_input = Dict(
        :z_latent => latents,
        :ca_coors => ca_coords,
        :mask => mask
    )
    dec_output = decoder(dec_input)

    println("  Output shapes:")
    println("    All-atom coordinates: $(size(dec_output[:coors]))")
    println("    Sequence logits: $(size(dec_output[:seq_logits]))")
    println("    Amino acid types: $(size(dec_output[:aatype_max]))")
    println("    Atom mask: $(size(dec_output[:atom_mask]))")
    println("  ✓ Decoder works correctly")

    # Test full sample() function
    println("\n--- Testing Full Sample Pipeline ---")

    samples = sample(
        score_net, decoder, L, B;
        nsteps=nsteps,
        latent_dim=latent_dim,
        self_cond=true,
        schedule_mode=:power,
        schedule_p=2.0,
        sampling_mode=:vf,
        mask=mask
    )

    println("  Output keys: $(keys(samples))")
    println("  CA coords shape: $(size(samples[:ca_coords]))")
    println("  All-atom coords shape: $(size(samples[:all_atom_coords]))")
    println("  ✓ sample() works correctly")

    # Save to PDB
    println("\n--- Saving to PDB ---")
    output_dir = "/tmp/juproteina_test_output"
    samples_to_pdb(samples, output_dir; prefix="test_sample", save_all_atom=true)

    # Verify files exist
    for b in 1:B
        pdb_file = joinpath(output_dir, "test_sample_$(b).pdb")
        if isfile(pdb_file)
            file_size = filesize(pdb_file)
            println("  ✓ Created $(pdb_file) ($(file_size) bytes)")
        else
            error("Failed to create $(pdb_file)")
        end
    end

    # Show sample of generated PDB
    println("\n--- Sample PDB Content ---")
    pdb_file = joinpath(output_dir, "test_sample_1.pdb")
    lines = readlines(pdb_file)
    for line in lines[1:min(10, length(lines))]
        println("  $line")
    end
    if length(lines) > 10
        println("  ... ($(length(lines) - 10) more lines)")
    end

    println("\n" * "=" ^ 60)
    println("END-TO-END INFERENCE TEST: PASSED")
    println("=" ^ 60)

    return true
end

# Run the test
test_full_inference_pipeline()
