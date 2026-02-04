# Debug encoder projected features - compare with Python

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using NPZ
using Statistics
using Flux

# Load Python debug data
println("Loading Python debug data...")
data_dir = joinpath(@__DIR__, "data")

coords_nm_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coords_nm.npy")))
coord_mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coord_mask.npy")))
aatype_py = npzread(joinpath(data_dir, "encoder_debug_aatype.npy"))
mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_mask.npy")))
seq_features_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_seq_features.npy")))  # [B, L, 768]
pair_features_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_pair_features.npy")))  # [B, L, L, 256]
raw_seq_features_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_raw_seq_features.npy")))  # [B, L, 468]

println("Python projected seq features shape: $(size(seq_features_py))")
println("Python pair features shape: $(size(pair_features_py))")

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

# Create encoder
encoder = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)

# Load weights
weights_path = joinpath(@__DIR__, "..", "weights", "encoder.npz")
load_encoder_weights!(encoder, weights_path)

# Get raw features from Julia
println("\n=== Raw Seq Features ===")
raw_feats_jl = [feat(batch, L, B) for feat in encoder.init_repr_factory.features]
raw_concat_jl = cat(raw_feats_jl...; dims=1)  # [468, L, B]

# Python: [B, L, 468] -> [468, L, B]
raw_seq_py_jl = permutedims(raw_seq_features_py, (3, 2, 1))

raw_diff = abs.(raw_concat_jl .- raw_seq_py_jl)
println("Raw feature diff - max: $(maximum(raw_diff)), mean: $(mean(raw_diff))")

# Project to seq features
println("\n=== Projected Seq Features ===")
projected_jl = encoder.init_repr_factory.projection(raw_concat_jl)  # [768, L, B]

# Python: [B, L, 768] -> [768, L, B]
seq_py_jl = permutedims(seq_features_py, (3, 2, 1))

proj_diff = abs.(projected_jl .- seq_py_jl)
println("Projected seq feature diff - max: $(maximum(proj_diff)), mean: $(mean(proj_diff))")
println("Sample Python[:4]: $(seq_py_jl[1:4, 1, 1])")
println("Sample Julia[:4]:  $(projected_jl[1:4, 1, 1])")

# Now check pair features
println("\n=== Pair Features ===")
pair_feats_jl = [feat(batch, L, B) for feat in encoder.pair_rep_factory.features]
raw_pair_concat_jl = cat(pair_feats_jl...; dims=1)  # [316, L, L, B]
projected_pair_jl = encoder.pair_rep_factory.projection(raw_pair_concat_jl)  # [256, L, L, B]

# Python: [B, L, L, 256] -> [256, L, L, B]
pair_py_jl = permutedims(pair_features_py, (4, 2, 3, 1))

pair_diff = abs.(projected_pair_jl .- pair_py_jl)
println("Projected pair feature diff - max: $(maximum(pair_diff)), mean: $(mean(pair_diff))")
println("Sample Python[0,0,:4]: $(pair_py_jl[1:4, 1, 1, 1])")
println("Sample Julia[:4,1,1,1]:  $(projected_pair_jl[1:4, 1, 1, 1])")

# Check raw pair features
println("\n=== Raw Pair Features Debug ===")
println("Julia raw pair features shape: $(size(raw_pair_concat_jl))")
for (i, feat) in enumerate(encoder.pair_rep_factory.features)
    jl_feat = pair_feats_jl[i]
    println("  $(typeof(feat)): shape=$(size(jl_feat))")
end
