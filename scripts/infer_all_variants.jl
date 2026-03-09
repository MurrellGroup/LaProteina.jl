#!/usr/bin/env julia
#
# Inference for all 10 LaProteina model variants (LD1-LD7 + AE1-AE3)
#
# Generates 2 samples per model at L=100, decodes to all-atom, saves PDB files,
# and computes basic geometry metrics (CA-CA bond lengths, clash scores).
#
# Usage:
#   julia --project=/home/claudey/JuProteina/run scripts/infer_all_variants.jl
#
# Output goes to: /home/claudey/JuProteina/inference_outputs/

using LaProteina
using LaProteina: MutableScoreNetworkWrapper, generate_with_flowfusion, sample_with_flowfusion
using LaProteina: samples_to_pdb, get_schedule, v_to_x1
using LaProteina: ScoreNetworkRawFeatures, extract_raw_features
using OnionTile  # GPU kernels
using Flux
using CUDA
using Random
using Printf
using Statistics
using LinearAlgebra

Random.seed!(42)

# ============================================================================
# Configuration
# ============================================================================

const OUTPUT_DIR = "/home/claudey/JuProteina/inference_outputs"
const N_SAMPLES = 2
const N_STEPS = 400
const LATENT_DIM = 8

# Model variant definitions
# See docs/model_variants.md for full documentation of each variant.
#
# Sampling parameters per model (from Python inference configs):
#   All models: 400 steps, self_cond=true, log schedule (CA), power schedule (latents)
#   LD1/LD2:    ca_noise=0.1, ll_noise=0.1
#   LD3:        ca_noise=0.15, ll_noise=0.05 (asymmetric for long proteins)
#   LD4-LD7:    ca_noise=0.1, ll_noise=0.1
#
# Known gaps vs Python:
#   - t_lim_ode_below=0.02 not implemented (minor: ODE at early t)
#   - center_every_step: Python uses False for motif models (LD4-7),
#     but Julia's zero_com=true always re-centers. Only matters for
#     actual motif-conditioned generation (not unconditional).
const MODEL_CONFIGS = [
    # --- Unconditional models ---
    (
        name = "LD1",
        desc = "Unconditional, no triangle update (up to 512 res)",
        sn_kwargs = Dict{Symbol,Any}(:cropped_flag => true),
        sn_file = "LD1_ucond_notri_512.safetensors",
        dec_file = "AE1_ucond_512.safetensors",
        seq_length = 100,   # Python default: [100, 200, 300, 400, 500]
        ca_noise = 0.1f0,
        ll_noise = 0.1f0,
        strict = true,
    ),
    (
        name = "LD2",
        desc = "Unconditional, triangle update (up to 512 res)",
        sn_kwargs = Dict{Symbol,Any}(:update_pair_repr => true, :update_pair_every_n => 2, :use_tri_mult => true),
        sn_file = "LD2_ucond_tri_512.safetensors",
        dec_file = "AE1_ucond_512.safetensors",
        seq_length = 100,   # Python default: [100, 200, 300, 400, 500]
        ca_noise = 0.1f0,
        ll_noise = 0.1f0,
        strict = true,
    ),
    (
        name = "LD3",
        desc = "Unconditional, long proteins (300-800 res)",
        sn_kwargs = Dict{Symbol,Any}(),
        sn_file = "LD3_ucond_notri_800.safetensors",
        dec_file = "AE2_ucond_800.safetensors",
        seq_length = 400,   # Python default: [300, 400, 500, 600, 700, 800]
        ca_noise = 0.15f0,  # Higher CA noise for long proteins
        ll_noise = 0.05f0,  # Lower latent noise for long proteins
        strict = true,
    ),
    # --- Motif scaffolding models (indexed) ---
    # LD4/LD5 run unconditionally here since motif data pipeline is not yet complete.
    # For proper scaffolding, use prepare_motif_batch() to provide motif conditioning.
    (
        name = "LD4",
        desc = "Indexed motif scaffolding, all-atom (up to 256 res)",
        sn_kwargs = Dict{Symbol,Any}(:motif_mode => :aa),
        sn_file = "LD4_motif_idx_aa.safetensors",
        dec_file = "AE3_motif.safetensors",
        seq_length = 100,   # Running unconditionally; with motif: up to 256 res
        ca_noise = 0.1f0,
        ll_noise = 0.1f0,
        strict = true,
    ),
    (
        name = "LD5",
        desc = "Indexed motif scaffolding, tip-atom (up to 256 res)",
        sn_kwargs = Dict{Symbol,Any}(:motif_mode => :tip),
        sn_file = "LD5_motif_idx_tip.safetensors",
        dec_file = "AE3_motif.safetensors",
        seq_length = 100,   # Running unconditionally; with motif: up to 256 res
        ca_noise = 0.1f0,
        ll_noise = 0.1f0,
        strict = true,
    ),
    # --- Motif scaffolding models (unindexed) ---
    (
        name = "LD6",
        desc = "Unindexed motif scaffolding, all-atom (up to 256 res)",
        sn_kwargs = Dict{Symbol,Any}(:motif_mode => :uidx),
        sn_file = "LD6_motif_uidx_aa.safetensors",
        dec_file = "AE3_motif.safetensors",
        seq_length = 100,   # Running unconditionally; with motif: up to 256 res
        ca_noise = 0.1f0,
        ll_noise = 0.1f0,
        strict = true,
    ),
    (
        name = "LD7",
        desc = "Unindexed motif scaffolding, tip-atom (up to 256 res)",
        sn_kwargs = Dict{Symbol,Any}(:motif_mode => :uidx),
        sn_file = "LD7_motif_uidx_tip.safetensors",
        dec_file = "AE3_motif.safetensors",
        seq_length = 100,   # Running unconditionally; with motif: up to 256 res
        ca_noise = 0.1f0,
        ll_noise = 0.1f0,
        strict = true,
    ),
]

# ============================================================================
# Geometry metrics
# ============================================================================

"""Compute CA-CA bond lengths for consecutive residues (in nm)."""
function ca_bond_lengths(ca_coords::AbstractMatrix{Float32})
    # ca_coords: [3, L]
    L = size(ca_coords, 2)
    if L < 2
        return Float32[]
    end
    diffs = ca_coords[:, 2:end] .- ca_coords[:, 1:end-1]
    return sqrt.(sum(diffs .^ 2; dims=1))[1, :]  # [L-1]
end

"""Compute all-atom clash score: fraction of non-bonded atom pairs with distance < 0.2nm."""
function clash_score(all_atom_coords::Array{Float32,3}, atom_mask::AbstractMatrix)
    # all_atom_coords: [3, 37, L], atom_mask: [37, L]
    L = size(all_atom_coords, 3)
    if L < 2
        return 0.0
    end

    # Collect all valid atom positions
    positions = Float32[]
    for l in 1:L, a in 1:37
        if atom_mask[a, l] > 0.5
            append!(positions, all_atom_coords[:, a, l])
        end
    end
    n_atoms = length(positions) ÷ 3
    if n_atoms < 2
        return 0.0
    end

    coords = reshape(positions, 3, n_atoms)

    # Count clashes (distance < 0.2nm, skip bonded neighbors)
    n_clashes = 0
    n_pairs = 0
    # Sample pairs to keep computation tractable
    max_pairs = min(n_atoms * (n_atoms - 1) ÷ 2, 50000)
    step = max(1, n_atoms * (n_atoms - 1) ÷ 2 ÷ max_pairs)

    pair_idx = 0
    for i in 1:n_atoms
        for j in (i+2):n_atoms  # skip immediate neighbors
            pair_idx += 1
            if pair_idx % step != 0
                continue
            end
            d = sqrt(sum((coords[:, i] .- coords[:, j]) .^ 2))
            n_pairs += 1
            if d < 0.2f0  # 0.2 nm = 2 Angstroms
                n_clashes += 1
            end
        end
    end

    return n_pairs > 0 ? n_clashes / n_pairs : 0.0
end

"""Compute radius of gyration (nm)."""
function radius_of_gyration(ca_coords::AbstractMatrix{Float32})
    # ca_coords: [3, L]
    L = size(ca_coords, 2)
    if L < 2
        return 0.0f0
    end
    center = mean(ca_coords; dims=2)  # [3, 1]
    diffs = ca_coords .- center
    return sqrt(mean(sum(diffs .^ 2; dims=1)))
end

# ============================================================================
# Main inference loop
# ============================================================================

function run_inference()
    mkpath(OUTPUT_DIR)

    # Open summary file
    summary_path = joinpath(OUTPUT_DIR, "summary.txt")
    summary_io = open(summary_path, "w")

    header = @sprintf("%-6s  %-8s  %8s  %8s  %8s  %8s  %8s  %s\n",
        "Model", "Sample", "Length", "CA_mean", "CA_std", "Rg(nm)", "Clashes", "Status")
    print(header)
    print(summary_io, header)
    println("-" ^ 80)
    println(summary_io, "-" ^ 80)

    # Cache loaded decoders to avoid reloading (AE1, AE2, AE3)
    decoder_cache = Dict{String, DecoderTransformer}()

    for config in MODEL_CONFIGS
        model_dir = joinpath(OUTPUT_DIR, config.name)
        mkpath(model_dir)

        println("\n=== $(config.name): $(config.desc) ===")
        println("  Score network: $(config.sn_file)")
        println("  Decoder: $(config.dec_file)")
        println("  Sequence length: $(config.seq_length)")

        try
            # Build and load score network
            sn = ScoreNetwork(;
                n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
                dim_cond=256, latent_dim=LATENT_DIM, output_param=:v,
                qk_ln=true,
                config.sn_kwargs...
            )
            load_score_network_weights_st!(sn, config.sn_file; strict=config.strict)
            sn_gpu = Flux.gpu(sn)
            println("  Score network loaded and moved to GPU")

            # Load decoder (cached)
            decoder = get!(decoder_cache, config.dec_file) do
                dec = DecoderTransformer(
                    n_layers=12, token_dim=768, pair_dim=256,
                    n_heads=12, dim_cond=128, latent_dim=LATENT_DIM,
                    qk_ln=true, update_pair_repr=false,
                )
                load_decoder_weights_st!(dec, config.dec_file)
                println("  Decoder loaded (new)")
                dec
            end
            println("  Decoder ready")

            # Generate samples
            println("  Generating $(N_SAMPLES) samples at L=$(config.seq_length)...")
            samples = sample_with_flowfusion(sn_gpu, decoder, config.seq_length, N_SAMPLES;
                nsteps=N_STEPS,
                latent_dim=LATENT_DIM,
                self_cond=true,
                dev=Flux.gpu,
                ca_sc_scale_noise=config.ca_noise,
                ll_sc_scale_noise=config.ll_noise,
            )
            println("  Generation complete")

            # Save PDB files
            samples_to_pdb(samples, model_dir; prefix=config.name)

            # Compute and report metrics per sample
            for b in 1:N_SAMPLES
                ca = samples[:ca_coords][:, :, b]  # [3, L]
                bonds = ca_bond_lengths(ca)
                rg = radius_of_gyration(ca)

                aa_coords = samples[:all_atom_coords][:, :, :, b]  # [3, 37, L]
                amask = samples[:atom_mask][:, :, b]  # [37, L]
                clashes = clash_score(aa_coords, amask)

                bond_mean = mean(bonds)
                bond_std = std(bonds)
                # Ideal CA-CA distance is ~0.38nm
                status = (0.30 < bond_mean < 0.45 && bond_std < 0.10) ? "OK" : "WARN"
                if clashes > 0.05
                    status = "WARN"
                end

                line = @sprintf("%-6s  %-8s  %8d  %8.4f  %8.4f  %8.4f  %8.4f  %s\n",
                    config.name, "s$(b)", config.seq_length, bond_mean, bond_std, rg, clashes, status)
                print(line)
                print(summary_io, line)
            end

            # Free GPU memory
            sn_gpu = nothing
            GC.gc()
            CUDA.reclaim()

        catch e
            msg = @sprintf("%-6s  %-8s  %8s  %8s  %8s  %8s  %8s  %s\n",
                config.name, "-", "-", "-", "-", "-", "-", "ERROR: $(sprint(showerror, e))")
            print(msg)
            print(summary_io, msg)
            # Print stack trace to stderr for debugging
            for (exc, bt) in Base.catch_stack()
                showerror(stderr, exc, bt)
                println(stderr)
            end
            GC.gc()
            CUDA.reclaim()
        end
    end

    println("\n" * "=" ^ 80)
    println("All PDB files saved to: $(OUTPUT_DIR)")
    println("Summary saved to: $(summary_path)")

    close(summary_io)
end

run_inference()
