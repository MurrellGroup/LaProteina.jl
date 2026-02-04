#!/usr/bin/env julia
"""
Convert precomputed shards from PrecomputedProtein struct to NamedTuples.

This removes the dependency on package-specific types for serialization,
making the data portable across package renames and versions.

Usage:
    julia scripts/convert_shards_to_namedtuples.jl

Environment variables:
    SHARD_DIR - Directory containing shards (default: ~/shared_data/afdb_laproteina/precomputed_shards)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2
using Dates

# Configuration
shard_dir = get(ENV, "SHARD_DIR", expanduser("~/shared_data/afdb_laproteina/precomputed_shards"))

println("=" ^ 70)
println("Converting Shards to NamedTuples")
println("=" ^ 70)
println("Shard directory: $shard_dir")
println()

# Find all shard files
shard_files = filter(f -> startswith(f, "train_shard_") && endswith(f, ".jld2"), readdir(shard_dir))
sort!(shard_files)

println("Found $(length(shard_files)) shard files")

for shard_file in shard_files
    input_path = joinpath(shard_dir, shard_file)

    # Create backup name and output name
    backup_path = joinpath(shard_dir, replace(shard_file, ".jld2" => "_old_struct.jld2"))
    output_path = input_path  # Overwrite original

    println("\n--- Processing $shard_file ---")

    # Check if already converted (backup exists)
    if isfile(backup_path)
        println("  Backup exists, skipping (already converted)")
        continue
    end

    # Load the shard (JLD2 will reconstruct the type)
    println("  Loading...")
    t_load = @elapsed begin
        data = jldopen(input_path, "r") do f
            f["proteins"]
        end
    end
    println("  Loaded $(length(data)) proteins in $(round(t_load, digits=2))s")

    # Convert to NamedTuples
    println("  Converting to NamedTuples...")
    proteins = Vector{NamedTuple{(:ca_coords, :z_mean, :z_log_scale, :mask),
                                  Tuple{Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}}}}(undef, length(data))

    for (i, p) in enumerate(data)
        proteins[i] = (
            ca_coords = p.ca_coords,
            z_mean = p.z_mean,
            z_log_scale = p.z_log_scale,
            mask = p.mask
        )
    end

    # Rename original to backup
    println("  Backing up original to $(basename(backup_path))...")
    mv(input_path, backup_path)

    # Save as NamedTuples
    println("  Saving NamedTuple version...")
    t_save = @elapsed jldsave(output_path; proteins=proteins)

    size_mb = round(filesize(output_path) / 1e6, digits=1)
    println("  Saved $size_mb MB in $(round(t_save, digits=2))s")
end

println("\n" * "=" ^ 70)
println("Conversion complete!")
println("=" ^ 70)
println("\nBackups saved with '_old_struct.jld2' suffix")
println("You can delete backups after verifying the new files work correctly")
