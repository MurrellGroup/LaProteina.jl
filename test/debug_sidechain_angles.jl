# Debug sidechain angles specifically

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using NPZ
using Statistics

# Load Python debug data
data_dir = joinpath(@__DIR__, "data")
raw_seq_features_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_raw_seq_features.npy")))  # [B, L, 468]
coords_nm_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coords_nm.npy")))
coord_mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_coord_mask.npy")))
aatype_py = npzread(joinpath(data_dir, "encoder_debug_aatype.npy"))
mask_py = Float32.(npzread(joinpath(data_dir, "encoder_debug_mask.npy")))

# Convert Python -> Julia format
coords_jl = permutedims(coords_nm_py, (4, 3, 2, 1))  # [3, 37, L, B]
coord_mask_jl = permutedims(coord_mask_py, (3, 2, 1))  # [37, L, B]
aatype_2d = dropdims(aatype_py, dims=1)
aatype_jl = reshape(aatype_2d .+ 1, :, 1)  # [L, B], 1-indexed
mask_2d = dropdims(mask_py, dims=1)
mask_jl = reshape(mask_2d, :, 1)

L, B = size(mask_jl)
println("L=$L, B=$B")

# Python feature slices
py_sc_start = 380 + 1  # After chainbreak(1) + restype(20) + atom37abs(148) + atom37rel(148) + bb(63) = 380
py_sc_end = py_sc_start + 87  # 88 dims

# Get Python SCAngles
py_sc = raw_seq_features_py[1, :, py_sc_start:py_sc_end]  # [L, 88]
println("\nPython SCAngles shape: $(size(py_sc))")

# Get Julia SCAngles
batch = Dict{Symbol, Any}(
    :coords => coords_jl,
    :coord_mask => coord_mask_jl,
    :residue_type => aatype_jl,
    :mask => mask_jl,
)
jl_sc = SidechainAngleFeature()(batch, L, B)  # [88, L, B]
jl_sc_2d = jl_sc[:, :, 1]'  # [L, 88]
println("Julia SCAngles shape: $(size(jl_sc_2d))")

# RESTYPES in Julia
RESTYPES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

println("\n=== Per-position comparison ===")
for pos in 1:L
    aa_idx = aatype_jl[pos, 1]
    aa_char = aa_idx <= 20 ? RESTYPES[aa_idx] : 'X'
    diff = abs.(py_sc[pos, :] .- jl_sc_2d[pos, :])
    max_diff = maximum(diff)

    # Only show if there's a difference
    if max_diff > 1e-6
        println("\nPosition $pos, AA=$aa_char (idx=$aa_idx), max_diff=$max_diff")

        # Show where differences are
        for i in 1:88
            if diff[i] > 1e-6
                println("  dim $i: Python=$(py_sc[pos, i]), Julia=$(jl_sc_2d[pos, i])")
            end
        end
    end
end

# Chi angle structure: 4 angles * 21 bins + 4 mask = 88
# chi1: dims 1-21
# chi2: dims 22-42
# chi3: dims 43-63
# chi4: dims 64-84
# masks: dims 85-88 (chi1_mask, chi2_mask, chi3_mask, chi4_mask)

println("\n=== Chi mask comparison ===")
for pos in 1:L
    aa_idx = aatype_jl[pos, 1]
    aa_char = aa_idx <= 20 ? RESTYPES[aa_idx] : 'X'

    # Python masks (dims 85-88)
    py_masks = py_sc[pos, 85:88]
    jl_masks = jl_sc_2d[pos, 85:88]

    if any(abs.(py_masks .- jl_masks) .> 1e-6)
        println("Position $pos, AA=$aa_char: Py masks=$(py_masks), Jl masks=$(jl_masks)")
    end
end

println("\n=== Summary by chi angle ===")
for chi in 1:4
    start_idx = (chi-1)*21 + 1
    end_idx = chi*21
    chi_diff = abs.(py_sc[:, start_idx:end_idx] .- jl_sc_2d[:, start_idx:end_idx])
    println("Chi$chi (dims $start_idx-$end_idx): max=$(maximum(chi_diff)), mean=$(mean(chi_diff))")
end

mask_diff = abs.(py_sc[:, 85:88] .- jl_sc_2d[:, 85:88])
println("Masks (dims 85-88): max=$(maximum(mask_diff)), mean=$(mean(mask_diff))")
