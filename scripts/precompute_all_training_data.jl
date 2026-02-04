#!/usr/bin/env julia
# Precompute VAE encoder outputs for all training data
# Processes files incrementally (not loading all into memory)
# Saves to 10 sharded JLD2 files with randomized order
#
# Environment variables:
#   START_SHARD - Resume from this shard (1-indexed), default 1
#   AFDB_INPUT_DIR - Directory containing mmCIF files
#   AFDB_OUTPUT_DIR - Directory for output shards
#   LOGFILE - Path for progress log (default: precompute_progress.log in current dir)
#
# Example:
#   START_SHARD=5 julia scripts/precompute_all_training_data.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Random
using Dates
using JLD2

# Configuration from environment variables with sensible defaults
const LOGFILE = get(ENV, "LOGFILE", joinpath(pwd(), "precompute_progress.log"))

function log_msg(msg)
    open(LOGFILE, "a") do f
        println(f, "[$(Dates.now())] $msg")
    end
    println(msg)
end

# IMPORTANT: Same seed for reproducible shuffling (enables resume)
Random.seed!(42)

# Resume from this shard (1-indexed), default 1
start_shard = parse(Int, get(ENV, "START_SHARD", "1"))

log_msg("=" ^ 70)
log_msg("Precompute All Training Data (Incremental)")
log_msg("Starting from shard: $start_shard")
log_msg("=" ^ 70)

# Check GPU
log_msg("=== GPU Status ===")
if !CUDA.functional()
    error("This script requires CUDA")
end
log_msg("CUDA is functional!")
log_msg("Device: $(CUDA.device())")
log_msg("Memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB available")

# ============================================================================
# Configuration
# ============================================================================
afdb_dir = get(ENV, "AFDB_INPUT_DIR", expanduser("~/shared_data/afdb_laproteina/raw"))
output_dir = get(ENV, "AFDB_OUTPUT_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards"))
n_shards = 10
min_length = 30
max_length = 256

log_msg("=== Configuration ===")
log_msg("Input dir: $afdb_dir")
log_msg("Output dir: $output_dir")
log_msg("N shards: $n_shards")
log_msg("Length filter: $min_length - $max_length")

# ============================================================================
# Get and shuffle file list (SAME SHUFFLE AS ORIGINAL RUN due to seed)
# ============================================================================
log_msg("=== Preparing File List ===")
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)
log_msg("Found $(length(cif_files)) CIF files")

# Shuffle for random order - MUST use same seed as original
Random.shuffle!(cif_files)
log_msg("Shuffled file order (seed=42)")

# Split into shards
files_per_shard = ceil(Int, length(cif_files) / n_shards)
file_shards = [cif_files[i:min(i + files_per_shard - 1, length(cif_files))]
               for i in 1:files_per_shard:length(cif_files)]
log_msg("Split into $(length(file_shards)) shards, ~$files_per_shard files each")

if start_shard > 1
    log_msg("Skipping shards 1-$(start_shard-1) (already completed)")
end

# ============================================================================
# Load encoder
# ============================================================================
log_msg("=== Loading Encoder ===")
latent_dim = 8

encoder_cpu = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=latent_dim, qk_ln=true, update_pair_repr=false
)
load_encoder_weights!(encoder_cpu, joinpath(@__DIR__, "..", "weights", "encoder.npz"))
log_msg("Encoder loaded on CPU")

encoder_gpu = deepcopy(encoder_cpu) |> gpu
log_msg("Encoder copied to GPU")

GC.gc()
CUDA.reclaim()
log_msg("GPU memory after encoder: $(round(CUDA.available_memory() / 1e9, digits=2)) GB available")

# ============================================================================
# Process each shard incrementally
# ============================================================================
log_msg("=== Processing Shards ===")
mkpath(output_dir)

t_start = time()
total_processed = 0
total_failed = 0
total_skipped = 0

for (shard_idx, shard_files) in enumerate(file_shards)
    # Skip already completed shards
    if shard_idx < start_shard
        continue
    end

    shard_start = time()
    log_msg("--- Shard $shard_idx / $(length(file_shards)) ($(length(shard_files)) files) ---")

    proteins = PrecomputedProtein[]
    failed = 0
    skipped = 0

    for (i, f) in enumerate(shard_files)
        filepath = joinpath(afdb_dir, f)

        # Load single file
        local data
        try
            data = load_pdb(filepath; chain_id="A")
        catch e
            failed += 1
            continue
        end

        # Check length
        L = length(data[:aatype])
        if !(min_length <= L <= max_length)
            skipped += 1
            continue
        end

        # Encode single protein
        protein = precompute_single_protein(encoder_cpu, encoder_gpu, data)
        if !isnothing(protein)
            push!(proteins, protein)
        else
            failed += 1
        end

        # Progress every 1000 files
        if i % 1000 == 0
            log_msg("  Shard $shard_idx: $i / $(length(shard_files)), encoded $(length(proteins)) proteins")
        end
    end

    # Save shard
    output_file = joinpath(output_dir, "train_shard_$(lpad(shard_idx, 2, '0')).jld2")
    jldsave(output_file; proteins=proteins)

    shard_time = time() - shard_start
    size_mb = round(filesize(output_file) / 1e6, digits=1)
    log_msg("  Saved $(length(proteins)) proteins to $output_file ($size_mb MB)")
    log_msg("  Shard time: $(round(shard_time / 60, digits=1)) min (failed: $failed, skipped: $skipped)")

    log_msg("  Updating totals...")
    global total_processed += length(proteins)
    global total_failed += failed
    global total_skipped += skipped

    # Clean up
    log_msg("  Running GC.gc()...")
    GC.gc()
    log_msg("  Running CUDA.reclaim()...")
    CUDA.reclaim()
    log_msg("  Shard $shard_idx complete, moving to next...")
end

t_elapsed = time() - t_start

# ============================================================================
# Summary
# ============================================================================
log_msg("=" ^ 70)
log_msg("Precomputation Complete!")
log_msg("=" ^ 70)
log_msg("Total time: $(round(t_elapsed / 60, digits=1)) minutes")
log_msg("Total proteins (this run): $total_processed")
log_msg("Total failed: $total_failed")
log_msg("Total skipped (length): $total_skipped")
if total_processed > 0
    log_msg("Time per protein: $(round(t_elapsed / total_processed * 1000, digits=1)) ms")
end

log_msg("Output files:")
for shard_idx in 1:length(file_shards)
    f = joinpath(output_dir, "train_shard_$(lpad(shard_idx, 2, '0')).jld2")
    if isfile(f)
        size_mb = round(filesize(f) / 1e6, digits=1)
        log_msg("  $f ($size_mb MB)")
    end
end

log_msg("Finished!")
