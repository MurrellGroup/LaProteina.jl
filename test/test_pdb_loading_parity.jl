# PDB loading parity test against Python reference
# Tests that Julia PDB loading produces the same outputs as Python

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using NPZ
using Test
using Statistics

# Load reference data
println("Loading Python reference data...")
data_dir = joinpath(@__DIR__, "data")

coords_py = Float32.(npzread(joinpath(data_dir, "pdb_coords_nm.npy")))  # [L, 37, 3]
aatype_py = npzread(joinpath(data_dir, "pdb_aatype.npy"))  # [L] (0-indexed)
atom_mask_py = Float32.(npzread(joinpath(data_dir, "pdb_atom_mask.npy")))  # [L, 37]
ca_coords_py = Float32.(npzread(joinpath(data_dir, "pdb_ca_coords_nm.npy")))  # [L, 3]

println("Python reference shapes:")
println("  coords: $(size(coords_py))")
println("  aatype: $(size(aatype_py))")
println("  atom_mask: $(size(atom_mask_py))")
println("  ca_coords: $(size(ca_coords_py))")

# Load PDB using Julia
println("\nLoading PDB with Julia...")
pdb_path = joinpath(@__DIR__, "samples_gpu", "gpu_sde_1.pdb")
data_jl = load_pdb(pdb_path; chain_id="A")

println("Julia loaded data:")
println("  coords: $(size(data_jl[:coords]))")  # [3, 37, L]
println("  aatype: $(size(data_jl[:aatype]))")  # [L]
println("  atom_mask: $(size(data_jl[:atom_mask]))")  # [37, L]
println("  sequence: $(data_jl[:sequence])")

# Convert Julia format to Python format for comparison
# Julia: [3, 37, L] -> Python: [L, 37, 3]
coords_jl = permutedims(data_jl[:coords], (3, 2, 1))

# Julia: [37, L] -> Python: [L, 37]
atom_mask_jl = permutedims(data_jl[:atom_mask], (2, 1))

# Julia aatype is 1-indexed, Python is 0-indexed
aatype_jl = data_jl[:aatype] .- 1

# Extract CA coordinates
# Julia: data[:coords][:, CA_INDEX, :] -> [3, L]
# Convert to Python: [L, 3]
ca_jl = extract_ca_coords(data_jl)
ca_coords_jl = permutedims(ca_jl, (2, 1))

println("\nComparing with Python reference...")

# Compare aatype
aatype_match = all(aatype_jl .== aatype_py)
println("aatype match: $aatype_match")
if !aatype_match
    println("  Mismatches at indices: $(findall(aatype_jl .!= aatype_py))")
end

# Compare atom mask
mask_match = all(Float32.(atom_mask_jl) .== atom_mask_py)
println("atom_mask match: $mask_match")
if !mask_match
    diff_count = sum(Float32.(atom_mask_jl) .!= atom_mask_py)
    println("  Number of mismatches: $diff_count")
end

# Compare CA coordinates
ca_diff = abs.(ca_coords_jl .- ca_coords_py)
println("\nCA coordinate differences:")
println("  max: $(maximum(ca_diff))")
println("  mean: $(mean(ca_diff))")

# Compare all coordinates (for non-masked atoms)
coords_diff = abs.(coords_jl .- coords_py)
mask_3d = repeat(reshape(atom_mask_py, size(atom_mask_py)..., 1), 1, 1, 3)
masked_coords_diff = coords_diff .* mask_3d

println("\nAll atom coordinate differences (masked):")
println("  max: $(maximum(masked_coords_diff))")
println("  mean: $(mean(masked_coords_diff[mask_3d .> 0]))")

# Print sample values
println("\nSample CA coordinates (first 3 residues):")
println("  Python:")
for i in 1:3
    println("    res $i: $(ca_coords_py[i, :])")
end
println("  Julia:")
for i in 1:3
    println("    res $i: $(ca_coords_jl[i, :])")
end

# Run tests
tolerance = 1e-5  # Very tight tolerance - should be exact match

@testset "PDB Loading Parity Tests" begin
    @test aatype_match
    @test mask_match
    @test maximum(ca_diff) < tolerance
    @test maximum(masked_coords_diff) < tolerance
end

println("\nPDB loading parity test complete!")
