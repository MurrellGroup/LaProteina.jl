# Debug pair features - compare raw pair features with Python

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using NPZ
using Statistics

# Load Python debug data
println("Loading Python debug data...")
data_dir = joinpath(@__DIR__, "data")

coords_nm_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coords_nm.npy")))
coord_mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coord_mask.npy")))
aatype_py = npzread(joinpath(data_dir, "encoder_debug_aatype.npy"))
mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_mask.npy")))
raw_pair_features_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_raw_pair_features.npy")))  # [B, L, L, 316]

println("Python raw pair features shape: $(size(raw_pair_features_py))")

# Convert Python -> Julia format
coords_jl = permutedims(coords_nm_py, (4, 3, 2, 1))  # [3, 37, L, B]
coord_mask_jl = permutedims(coord_mask_py, (3, 2, 1))  # [37, L, B]
aatype_2d = dropdims(aatype_py, dims=1)
aatype_jl = reshape(aatype_2d .+ 1, :, 1)  # [L, B], 1-indexed
mask_2d = dropdims(mask_py, dims=1)
mask_jl = reshape(mask_2d, :, 1)

L, B = size(mask_jl)

# Prepare batch
batch = Dict{Symbol, Any}(
    :coords => coords_jl,
    :coord_mask => coord_mask_jl,
    :residue_type => aatype_jl,
    :mask => mask_jl,
)

# Python: [B, L, L, 316] -> [316, L, L, B]
raw_pair_py_jl = permutedims(raw_pair_features_py, (4, 2, 3, 1))

# Julia individual pair features
feat_types = [
    (RelSeqSepFeature(63), 127),
    (BackbonePairDistFeature(), 84),
    (ResidueOrientationFeature(), 105),
]

# Python feature slices (cumulative offsets)
py_offsets = [0, 127, 211]  # cumulative: 127, 84, 105 = 316

println("\n=== Comparing Individual Pair Features ===")
for (i, (ftype, expected_dim)) in enumerate(feat_types)
    # Extract Julia feature
    jl_feat = ftype(batch, L, B)  # [dim, L, L, B]

    # Get Python slice
    dim = get_dim(ftype)
    py_start = py_offsets[i] + 1
    py_end = py_offsets[i] + dim
    py_feat = raw_pair_py_jl[py_start:py_end, :, :, :]  # [dim, L, L, B]

    # Compare
    diff = abs.(jl_feat .- py_feat)

    println("\n$(typeof(ftype)) (dim=$(dim)):")
    println("  Python[0,0,:4]: $(py_feat[1:min(4, dim), 1, 1, 1])")
    println("  Julia[:4,1,1,1]: $(jl_feat[1:min(4, dim), 1, 1, 1])")
    println("  Max diff: $(maximum(diff))")
    println("  Mean diff: $(mean(diff))")

    # Show positions with largest differences
    if maximum(diff) > 1e-4
        println("  Positions with largest diffs:")
        for idx in findall(diff .> (maximum(diff) * 0.9))
            println("    $idx: py=$(py_feat[idx]), jl=$(jl_feat[idx]), diff=$(diff[idx])")
        end
    end
end

# Full raw pair comparison
println("\n=== Full Raw Pair Features ===")
raw_feats_jl = [ftype(batch, L, B) for (ftype, _) in feat_types]
raw_concat_jl = cat(raw_feats_jl...; dims=1)  # [316, L, L, B]

raw_diff = abs.(raw_concat_jl .- raw_pair_py_jl)
println("Raw pair feature diff - max: $(maximum(raw_diff)), mean: $(mean(raw_diff))")
