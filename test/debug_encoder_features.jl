# Debug encoder features - compare raw features with Python

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using NPZ
using Statistics

# Load Python debug data
println("Loading Python debug data...")
data_dir = joinpath(@__DIR__, "data")

coords_nm_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coords_nm.npy")))
coord_mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coord_mask.npy")))
aatype_py = npzread(joinpath(data_dir, "encoder_debug_aatype.npy"))
mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_mask.npy")))
raw_seq_features_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_raw_seq_features.npy")))  # [B, L, 468]
mean_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_mean.npy")))

println("Python raw features shape: $(size(raw_seq_features_py))")

# Convert Python -> Julia format
coords_jl = permutedims(coords_nm_py, (4, 3, 2, 1))  # [3, 37, L, B]
coord_mask_jl = permutedims(coord_mask_py, (3, 2, 1))  # [37, L, B]
aatype_2d = dropdims(aatype_py, dims=1)
aatype_jl = reshape(aatype_2d .+ 1, :, 1)  # [L, B], 1-indexed
mask_2d = dropdims(mask_py, dims=1)
mask_jl = reshape(mask_2d, :, 1)

# Prepare batch
batch = Dict{Symbol, Any}(
    :coords => coords_jl,
    :coord_mask => coord_mask_jl,
    :residue_type => aatype_jl,
    :mask => mask_jl,
)

L, B = size(mask_jl)

println("\n=== Comparing Individual Features ===")

# Extract each feature and compare
feat_names = ["ChainBreak", "ResidueType", "Atom37Abs", "Atom37Rel", "BBTorsion", "SCAngles"]
feat_types = [
    ChainBreakFeature(),
    ResidueTypeFeature(),
    Atom37CoordFeature(relative=false),
    Atom37CoordFeature(relative=true),
    BackboneTorsionFeature(),
    SidechainAngleFeature()
]

# Python feature slices (cumulative offsets)
py_offsets = [0, 1, 21, 169, 317, 380]  # cumulative: 1, 20, 148, 148, 63, 88 = 468

for (i, (name, ftype)) in enumerate(zip(feat_names, feat_types))
    # Extract Julia feature
    jl_feat = ftype(batch, L, B)  # [dim, L, B]

    # Get Python slice
    dim = get_dim(ftype)
    py_start = py_offsets[i] + 1
    py_end = py_offsets[i] + dim
    # Python is [B, L, dim], need to get [dim] for first sample, first position
    py_feat = raw_seq_features_py[1, 1, py_start:py_end]

    # Julia is [dim, L, B], get first position
    jl_feat_sample = jl_feat[:, 1, 1]

    # Compare
    diff = abs.(jl_feat_sample .- py_feat)

    println("\n$name (dim=$dim):")
    println("  Python[0:4]: $(py_feat[1:min(4, dim)])")
    println("  Julia[1:4]:  $(jl_feat_sample[1:min(4, dim)])")
    println("  Max diff: $(maximum(diff))")
    println("  Mean diff: $(mean(diff))")
end

# Now check full raw feature concatenation
println("\n=== Full Raw Features ===")
raw_feats_jl = [ftype(batch, L, B) for ftype in feat_types]
raw_concat_jl = cat(raw_feats_jl...; dims=1)  # [468, L, B]

# Python: [B, L, 468] -> [468, L, B]
raw_seq_py_jl = permutedims(raw_seq_features_py, (3, 2, 1))

raw_diff = abs.(raw_concat_jl .- raw_seq_py_jl)
println("Raw feature diff - max: $(maximum(raw_diff)), mean: $(mean(raw_diff))")

# Create encoder and load weights
println("\n=== Loading Encoder ===")
encoder = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
weights_path = joinpath(@__DIR__, "..", "weights", "encoder.npz")
load_encoder_weights!(encoder, weights_path)

# Run full encoder
result = encoder(batch)
mean_jl = result[:mean]

# Python: [B, L, 8] -> [8, L, B]
mean_py_jl = permutedims(mean_py, (3, 2, 1))

mean_diff = abs.(mean_jl .- mean_py_jl)
println("\n=== Final Output ===")
println("Mean diff - max: $(maximum(mean_diff)), mean: $(mean(mean_diff))")
println("Python mean[:4]: $(mean_py[1,1,1:4])")
println("Julia mean[:4]:  $(mean_jl[1:4, 1, 1])")
