#!/usr/bin/env julia
# Run inference with trained La-Proteina weights
# Generates protein structures using flow matching + VAE decoder

using Pkg
Pkg.activate(dirname(@__DIR__))

using JuProteina
using Random
using Statistics

function main()
    println("=" ^ 70)
    println("La-Proteina Inference in Julia (JuProteina)")
    println("=" ^ 70)

    # Paths
    weights_dir = joinpath(dirname(@__DIR__), "weights")
    output_dir = joinpath(dirname(@__DIR__), "generated_structures")
    mkpath(output_dir)

    score_net_weights = joinpath(weights_dir, "score_network.npz")
    decoder_weights = joinpath(weights_dir, "decoder.npz")

    if !isfile(score_net_weights) || !isfile(decoder_weights)
        error("Weights not found. Run scripts/extract_weights.py first.")
    end

    # Parameters matching Python config
    L = 100          # Sequence length
    B = 4            # Number of samples
    nsteps = 100     # Inference steps
    latent_dim = 8

    # Full model parameters (matching LD1_ucond_notri_512)
    n_layers = 14
    token_dim = 768
    pair_dim = 256
    n_heads = 12
    dim_cond = 256

    println("\nParameters:")
    println("  Sequence length: $L")
    println("  Batch size: $B")
    println("  Inference steps: $nsteps")
    println("  Model: 14 layers, 768 dim, 12 heads")

    Random.seed!(42)

    # Create score network
    println("\nCreating ScoreNetwork (160M params)...")
    score_net = ScoreNetwork(
        n_layers=n_layers,
        token_dim=token_dim,
        pair_dim=pair_dim,
        n_heads=n_heads,
        dim_cond=dim_cond,
        latent_dim=latent_dim,
        qk_ln=true,
        update_pair_repr=false,  # No triangle updates for LD1
        output_param=:v
    )
    println("  Loading weights...")
    load_score_network_weights!(score_net, score_net_weights)
    println("  ✓ ScoreNetwork ready")

    # Create decoder - use Python defaults
    println("\nCreating DecoderTransformer...")
    decoder = DecoderTransformer(
        n_layers=12,
        token_dim=token_dim,
        # pair_dim=256 (default from Python config)
        n_heads=n_heads,
        # dim_cond=128 (default from Python config)
        latent_dim=latent_dim
        # qk_ln=true (default from Python config)
        # update_pair_repr=false (default from Python config)
        # abs_coors=false (default from Python config)
    )
    println("  Loading weights...")
    load_decoder_weights!(decoder, decoder_weights)
    println("  ✓ DecoderTransformer ready")

    # Run flow matching simulation
    # Using Python inference config: inference_ucond_notri.yaml
    # bb_ca: schedule log p=2.0, gt 1/t p=1.0, sampling_mode sc
    # local_latents: schedule power p=2.0, gt tan p=1.0
    println("\n" * "-" ^ 70)
    println("Running flow matching simulation...")
    println("-" ^ 70)

    mask = ones(Float32, L, B)

    # NOTE: Python uses different settings per modality, but Julia full_simulation
    # uses unified settings. Using bb_ca settings (log schedule).
    flow_samples = full_simulation(
        score_net, L, B;
        nsteps=nsteps,        # Python uses 400
        latent_dim=latent_dim,
        self_cond=true,
        schedule_mode=:log,   # Python bb_ca uses log
        schedule_p=2.0,
        gt_mode=Symbol("1/t"),  # Python bb_ca uses "1/t"
        gt_param=1.0,
        sampling_mode=:sc,    # Python uses score-based SDE
        center_ca=true,
        mask=mask
    )

    ca_coords = flow_samples[:bb_ca]
    latents = flow_samples[:local_latents]

    println("  Generated CA coordinates: $(size(ca_coords))")
    println("  Generated latents: $(size(latents))")

    # Check CA is zero-centered
    com = dropdims(mean(ca_coords; dims=2); dims=2)
    max_com = maximum(abs.(com))
    println("  Max COM deviation: $(max_com)")

    # Decode to all-atom
    println("\nDecoding to all-atom structures...")
    dec_input = Dict(
        :z_latent => latents,
        :ca_coors => ca_coords,
        :mask => mask
    )
    dec_output = decoder(dec_input)

    println("  All-atom coordinates: $(size(dec_output[:coors]))")
    println("  Sequence logits: $(size(dec_output[:seq_logits]))")

    # Prepare samples dict
    samples = Dict(
        :ca_coords => ca_coords,
        :latents => latents,
        :seq_logits => dec_output[:seq_logits],
        :all_atom_coords => dec_output[:coors],
        :aatype => dec_output[:aatype_max],
        :atom_mask => dec_output[:atom_mask],
        :mask => mask
    )

    # Save to PDB
    println("\nSaving structures to PDB...")
    samples_to_pdb(samples, output_dir; prefix="laproteina_jl", save_all_atom=true)

    # Print sample info
    println("\n" * "-" ^ 70)
    println("Generated Structures")
    println("-" ^ 70)

    for b in 1:B
        pdb_file = joinpath(output_dir, "laproteina_jl_$(b).pdb")
        file_size = filesize(pdb_file)

        # Get sequence
        aatype_b = samples[:aatype][:, b]
        seq = join([index_to_aa(aa) for aa in aatype_b])

        println("\nSample $b:")
        println("  File: $pdb_file ($file_size bytes)")
        println("  Sequence (first 50): $(seq[1:min(50, length(seq))])...")

        # Compute CA-CA distances to check structure quality
        ca_b = ca_coords[:, :, b]  # [3, L]
        distances = Float32[]
        for i in 1:(L-1)
            d = sqrt(sum((ca_b[:, i] .- ca_b[:, i+1]).^2))
            push!(distances, d)
        end
        mean_dist = mean(distances) * 10  # Convert to Angstroms
        std_dist = std(distances) * 10

        println("  Mean CA-CA distance: $(round(mean_dist, digits=2)) ± $(round(std_dist, digits=2)) Å")
    end

    println("\n" * "=" ^ 70)
    println("INFERENCE COMPLETE")
    println("=" ^ 70)

    return samples
end

# Run
samples = main()
