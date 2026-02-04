#!/usr/bin/env julia
"""
Debug pair representation building in Julia.
Compares intermediate values with Python reference data.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NPZ
using LinearAlgebra
using Statistics
using Flux

# Import LaProteina
include(joinpath(@__DIR__, "..", "src", "LaProteina.jl"))
using .LaProteina

# ============================================================================
# Helper functions
# ============================================================================

"""Convert from Python [B, L, D] to Julia [D, L, B]."""
function py_to_jl(arr::AbstractArray{T, 3}) where T
    return permutedims(arr, (3, 2, 1))
end

"""Convert from Python [B, L_i, L_j, D] to Julia [D, L_i, L_j, B]."""
function py_to_jl_pair(arr::AbstractArray{T, 4}) where T
    return permutedims(arr, (4, 2, 3, 1))
end

"""Check parity between Julia and Python arrays."""
function check_parity(name::String, jl_arr, py_arr; rtol=1e-4, atol=1e-5)
    if size(jl_arr) != size(py_arr)
        println("  ✗ $name: Shape mismatch - Julia $(size(jl_arr)) vs Python $(size(py_arr))")
        return false
    end

    max_diff = maximum(abs.(jl_arr .- py_arr))
    rel_diff = max_diff / (maximum(abs.(py_arr)) + 1e-10)

    if isapprox(jl_arr, py_arr; rtol=rtol, atol=atol)
        println("  ✓ $name: PASS (max diff: $max_diff)")
        return true
    else
        println("  ✗ $name: FAIL (max diff: $max_diff, rel diff: $rel_diff)")
        # Show sample values
        println("    Julia[:5]: $(jl_arr[1:min(5, length(jl_arr))])")
        println("    Python[:5]: $(py_arr[1:min(5, length(py_arr))])")
        return false
    end
end

# ============================================================================
# Load Python reference data
# ============================================================================

println("=" ^ 60)
println("Pair Representation Parity Debug")
println("=" ^ 60)

test_dir = @__DIR__

# Load inputs
x_t_bb_ca_py = Float32.(npzread(joinpath(test_dir, "pair_debug_x_t_bb_ca.npy")))  # [1, 5, 3]
x_t_local_latents_py = Float32.(npzread(joinpath(test_dir, "pair_debug_x_t_local_latents.npy")))
x_sc_bb_ca_py = Float32.(npzread(joinpath(test_dir, "pair_debug_x_sc_bb_ca.npy")))
x_sc_local_latents_py = Float32.(npzread(joinpath(test_dir, "pair_debug_x_sc_local_latents.npy")))
t_bb_ca_py = Float32.(npzread(joinpath(test_dir, "pair_debug_t_bb_ca.npy")))
t_local_latents_py = Float32.(npzread(joinpath(test_dir, "pair_debug_t_local_latents.npy")))
mask_py = Float32.(npzread(joinpath(test_dir, "pair_debug_mask.npy")))

# Convert inputs to Julia format
x_t_bb_ca = py_to_jl(x_t_bb_ca_py)  # [3, 5, 1]
x_t_local_latents = py_to_jl(x_t_local_latents_py)
x_sc_bb_ca = py_to_jl(x_sc_bb_ca_py)
x_sc_local_latents = py_to_jl(x_sc_local_latents_py)
mask = permutedims(mask_py, (2, 1))  # [5, 1]

B = 1
L = 5

println("\nInput shapes (Julia format):")
println("  x_t_bb_ca: $(size(x_t_bb_ca))")
println("  x_t_local_latents: $(size(x_t_local_latents))")
println("  mask: $(size(mask))")

# Create batch dict for feature factories
batch = Dict(
    :x_t => Dict(:bb_ca => x_t_bb_ca, :local_latents => x_t_local_latents),
    :x_sc => Dict(:bb_ca => x_sc_bb_ca, :local_latents => x_sc_local_latents),
    :t => Dict(:bb_ca => t_bb_ca_py, :local_latents => t_local_latents_py),
    :mask => mask
)

# ============================================================================
# Step 1: Compare raw pair features
# ============================================================================

println("\n=== Step 1: Raw Pair Features ===")

# Load Python reference
rel_seq_sep_py = Float32.(npzread(joinpath(test_dir, "pair_debug_rel_seq_sep.npy")))
xt_pair_dist_py = Float32.(npzread(joinpath(test_dir, "pair_debug_xt_pair_dist.npy")))
xsc_pair_dist_py = Float32.(npzread(joinpath(test_dir, "pair_debug_xsc_pair_dist.npy")))
optional_pair_dist_py = Float32.(npzread(joinpath(test_dir, "pair_debug_optional_pair_dist.npy")))
concat_features_py = Float32.(npzread(joinpath(test_dir, "pair_debug_concat_features.npy")))

# Compute Julia features
rel_seq_sep_feat = RelSeqSepFeature(; seq_sep_dim=127)
rel_seq_sep_jl = rel_seq_sep_feat(batch, L, B)  # [127, 5, 5, 1]

println("rel_seq_sep Julia shape: $(size(rel_seq_sep_jl))")
println("rel_seq_sep Python shape: $(size(rel_seq_sep_py))")

# Convert Python to Julia format and compare
rel_seq_sep_py_jl = py_to_jl_pair(rel_seq_sep_py)
check_parity("rel_seq_sep", rel_seq_sep_jl, rel_seq_sep_py_jl)

# xt pair distances
xt_pair_dist_feat = XtBBCAPairDistFeature(30, 0.1f0, 3.0f0)
xt_pair_dist_jl = xt_pair_dist_feat(batch, L, B)
xt_pair_dist_py_jl = py_to_jl_pair(xt_pair_dist_py)
check_parity("xt_pair_dist", xt_pair_dist_jl, xt_pair_dist_py_jl)

# x_sc pair distances
xsc_pair_dist_feat = XscBBCAPairDistFeature(30, 0.1f0, 3.0f0)
xsc_pair_dist_jl = xsc_pair_dist_feat(batch, L, B)
xsc_pair_dist_py_jl = py_to_jl_pair(xsc_pair_dist_py)
check_parity("xsc_pair_dist", xsc_pair_dist_jl, xsc_pair_dist_py_jl)

# optional pair distances (should be zeros)
optional_pair_dist_feat = OptionalCAPairDistFeature()
optional_pair_dist_jl = optional_pair_dist_feat(batch, L, B)
optional_pair_dist_py_jl = py_to_jl_pair(optional_pair_dist_py)
check_parity("optional_pair_dist", optional_pair_dist_jl, optional_pair_dist_py_jl)

# Concatenated features
concat_jl = cat(rel_seq_sep_jl, xt_pair_dist_jl, xsc_pair_dist_jl, optional_pair_dist_jl; dims=1)
concat_py_jl = py_to_jl_pair(concat_features_py)
check_parity("concat_features", concat_jl, concat_py_jl)

println("concat_jl shape: $(size(concat_jl))")

# ============================================================================
# Step 2: Compare projection and LayerNorm
# ============================================================================

println("\n=== Step 2: Projection and LayerNorm ===")

# Load Python reference
pair_proj_py = Float32.(npzread(joinpath(test_dir, "pair_debug_pair_proj.npy")))
pair_proj_ln_py = Float32.(npzread(joinpath(test_dir, "pair_debug_pair_proj_ln.npy")))

# Load weights
weights_path = joinpath(@__DIR__, "..", "weights", "score_network.npz"))
weights = npzread(weights_path)

proj_weight = Float32.(weights["pair_repr_builder.init_repr_factory.linear_out.weight"])  # [256, 217]
ln_weight = Float32.(weights["pair_repr_builder.init_repr_factory.ln_out.weight"])  # [256]
ln_bias = Float32.(weights["pair_repr_builder.init_repr_factory.ln_out.bias"])  # [256]

println("Projection weight shape: $(size(proj_weight))")

# Create Dense layer
proj = Dense(217 => 256; bias=false)
proj.weight .= proj_weight

# Apply projection: Julia format is [D_in, L, L, B] -> [D_out, L, L, B]
pair_proj_jl = proj(concat_jl)

pair_proj_py_jl = py_to_jl_pair(pair_proj_py)
check_parity("pair_proj", pair_proj_jl, pair_proj_py_jl)

# Apply LayerNorm
ln = Flux.LayerNorm(256)
ln.diag.scale .= ln_weight
ln.diag.bias .= ln_bias

pair_proj_ln_jl = ln(pair_proj_jl)

pair_proj_ln_py_jl = py_to_jl_pair(pair_proj_ln_py)
check_parity("pair_proj_ln", pair_proj_ln_jl, pair_proj_ln_py_jl)

# ============================================================================
# Step 3: Compare conditioning features
# ============================================================================

println("\n=== Step 3: Conditioning Features ===")

# Load Python reference
t_emb_bb_ca_py = Float32.(npzread(joinpath(test_dir, "pair_debug_t_emb_bb_ca.npy")))  # [1, 256]
t_emb_local_latents_py = Float32.(npzread(joinpath(test_dir, "pair_debug_t_emb_local_latents.npy")))
t_emb_pair_py = Float32.(npzread(joinpath(test_dir, "pair_debug_t_emb_pair.npy")))  # [1, 5, 5, 512]
cond_proj_py = Float32.(npzread(joinpath(test_dir, "pair_debug_cond_proj.npy")))
cond_proj_ln_py = Float32.(npzread(joinpath(test_dir, "pair_debug_cond_proj_ln.npy")))

# Compute time embeddings in Julia
t_emb_bb_ca_jl = get_time_embedding(t_bb_ca_py, 256)  # [256, 1]
t_emb_local_latents_jl = get_time_embedding(t_local_latents_py, 256)

println("Time embedding Julia shape: $(size(t_emb_bb_ca_jl))")
println("Time embedding Python shape: $(size(t_emb_bb_ca_py))")

# Python format: [B, D], Julia format: [D, B]
t_emb_bb_ca_py_jl = permutedims(t_emb_bb_ca_py, (2, 1))
check_parity("t_emb_bb_ca", t_emb_bb_ca_jl, t_emb_bb_ca_py_jl)

t_emb_local_latents_py_jl = permutedims(t_emb_local_latents_py, (2, 1))
check_parity("t_emb_local_latents", t_emb_local_latents_jl, t_emb_local_latents_py_jl)

# Use the TimePairFeature to get pair-broadcasted time embeddings
time_pair_feat_bb_ca = TimePairFeature(256, :bb_ca)
time_pair_feat_ll = TimePairFeature(256, :local_latents)

t_pair_bb_ca_jl = time_pair_feat_bb_ca(batch, L, B)  # [256, L, L, B]
t_pair_ll_jl = time_pair_feat_ll(batch, L, B)

# Concatenate
t_emb_pair_jl = cat(t_pair_bb_ca_jl, t_pair_ll_jl; dims=1)  # [512, L, L, B]

t_emb_pair_py_jl = py_to_jl_pair(t_emb_pair_py)
check_parity("t_emb_pair", t_emb_pair_jl, t_emb_pair_py_jl)

# Apply conditioning projection and LayerNorm
cond_proj_weight = Float32.(weights["pair_repr_builder.cond_factory.linear_out.weight"])  # [256, 512]
cond_ln_weight = Float32.(weights["pair_repr_builder.cond_factory.ln_out.weight"])
cond_ln_bias = Float32.(weights["pair_repr_builder.cond_factory.ln_out.bias"])

cond_proj_layer = Dense(512 => 256; bias=false)
cond_proj_layer.weight .= cond_proj_weight

cond_proj_jl = cond_proj_layer(t_emb_pair_jl)
cond_proj_py_jl = py_to_jl_pair(cond_proj_py)
check_parity("cond_proj", cond_proj_jl, cond_proj_py_jl)

cond_ln = Flux.LayerNorm(256)
cond_ln.diag.scale .= cond_ln_weight
cond_ln.diag.bias .= cond_ln_bias

cond_proj_ln_jl = cond_ln(cond_proj_jl)
cond_proj_ln_py_jl = py_to_jl_pair(cond_proj_ln_py)
check_parity("cond_proj_ln", cond_proj_ln_jl, cond_proj_ln_py_jl)

# ============================================================================
# Step 4: Compare AdaptiveLayerNormIdentical
# ============================================================================

println("\n=== Step 4: AdaptiveLayerNormIdentical ===")

# Load Python reference
cond_normed_py = Float32.(npzread(joinpath(test_dir, "pair_debug_cond_normed.npy")))
gamma_py = Float32.(npzread(joinpath(test_dir, "pair_debug_gamma.npy")))
beta_py = Float32.(npzread(joinpath(test_dir, "pair_debug_beta.npy")))
final_pair_repr_py = Float32.(npzread(joinpath(test_dir, "pair_debug_final_pair_repr.npy")))

# Load AdaLN weights
adaln_norm_cond_weight = Float32.(weights["pair_repr_builder.adaln.norm_cond.weight"])
adaln_norm_cond_bias = Float32.(weights["pair_repr_builder.adaln.norm_cond.bias"])
adaln_to_gamma_weight = Float32.(weights["pair_repr_builder.adaln.to_gamma.0.weight"])
adaln_to_gamma_bias = Float32.(weights["pair_repr_builder.adaln.to_gamma.0.bias"])
adaln_to_beta_weight = Float32.(weights["pair_repr_builder.adaln.to_beta.weight"])

# Step 1: Reduce conditioning from [D, L, L, B] to [D, B] and apply LayerNorm
# (All positions have same conditioning due to broadcast)
cond_batch = cond_proj_ln_jl[:, 1, 1, :]  # [D, B]

adaln_norm_cond = Flux.LayerNorm(256)
adaln_norm_cond.diag.scale .= adaln_norm_cond_weight
adaln_norm_cond.diag.bias .= adaln_norm_cond_bias

cond_normed_jl = adaln_norm_cond(cond_batch)  # [D, B]

# Python cond_normed is [B, D], convert to [D, B]
cond_normed_py_jl = permutedims(cond_normed_py, (2, 1))
check_parity("cond_normed", cond_normed_jl, cond_normed_py_jl)

# Step 2: Compute gamma (Linear + Sigmoid) - [D, B]
to_gamma = Dense(256 => 256)
to_gamma.weight .= adaln_to_gamma_weight
to_gamma.bias .= adaln_to_gamma_bias

gamma_linear = to_gamma(cond_normed_jl)
gamma_jl = NNlib.sigmoid.(gamma_linear)

# Python gamma is [B, D], convert to [D, B]
gamma_py_jl = permutedims(gamma_py, (2, 1))
check_parity("gamma", gamma_jl, gamma_py_jl)

# Step 3: Compute beta (Linear, no bias) - [D, B]
to_beta = Dense(256 => 256; bias=false)
to_beta.weight .= adaln_to_beta_weight

beta_jl = to_beta(cond_normed_jl)

beta_py_jl = permutedims(beta_py, (2, 1))
check_parity("beta", beta_jl, beta_py_jl)

# Step 4: Apply LayerNorm (no affine) to pair_proj_ln, then gamma * normed + beta
# Create no-affine LayerNorm
adaln_norm_x = Flux.LayerNorm(256; affine=false)
normed_x_jl = adaln_norm_x(pair_proj_ln_jl)  # [D, L, L, B]

# Broadcast gamma and beta from [D, B] to [D, 1, 1, B]
gamma_brc = reshape(gamma_jl, 256, 1, 1, B)
beta_brc = reshape(beta_jl, 256, 1, 1, B)

final_pair_repr_jl = normed_x_jl .* gamma_brc .+ beta_brc

# Apply pair mask
pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)  # [L, L, B]
pair_mask_exp = reshape(pair_mask, 1, L, L, B)
final_pair_repr_jl = final_pair_repr_jl .* pair_mask_exp

final_pair_repr_py_jl = py_to_jl_pair(final_pair_repr_py)
check_parity("final_pair_repr", final_pair_repr_jl, final_pair_repr_py_jl)

# ============================================================================
# Now test using the full FeatureFactory and PairReprBuilder
# ============================================================================

println("\n=== Full PairReprBuilder Test ===")

# Create PairReprBuilder with loaded weights
pair_rep_builder = PairReprBuilder(256, 256;
    use_conditioning=true,
    xt_pair_dist_dim=30, xt_pair_dist_min=0.1, xt_pair_dist_max=3.0,
    x_sc_pair_dist_dim=30, x_sc_pair_dist_min=0.1, x_sc_pair_dist_max=3.0,
    seq_sep_dim=127, t_emb_dim=256)

# Load weights into the PairReprBuilder
# init_repr_factory projection
pair_rep_builder.init_repr_factory.projection.weight .= proj_weight

# init_repr_factory LayerNorm (if it has one and use_ln is true)
if pair_rep_builder.init_repr_factory.use_ln && !isnothing(pair_rep_builder.init_repr_factory.ln)
    pair_rep_builder.init_repr_factory.ln.diag.scale .= ln_weight
    pair_rep_builder.init_repr_factory.ln.diag.bias .= ln_bias
end

# cond_factory projection and LN
pair_rep_builder.cond_factory.projection.weight .= cond_proj_weight
if pair_rep_builder.cond_factory.use_ln && !isnothing(pair_rep_builder.cond_factory.ln)
    pair_rep_builder.cond_factory.ln.diag.scale .= cond_ln_weight
    pair_rep_builder.cond_factory.ln.diag.bias .= cond_ln_bias
end

# AdaLN weights
pair_rep_builder.adaln.norm_cond.diag.scale .= adaln_norm_cond_weight
pair_rep_builder.adaln.norm_cond.diag.bias .= adaln_norm_cond_bias
pair_rep_builder.adaln.to_gamma[1].weight .= adaln_to_gamma_weight
pair_rep_builder.adaln.to_gamma[1].bias .= adaln_to_gamma_bias
pair_rep_builder.adaln.to_beta.weight .= adaln_to_beta_weight

# Run full PairReprBuilder
full_pair_repr_jl = pair_rep_builder(batch)

check_parity("full_pair_repr_builder", full_pair_repr_jl, final_pair_repr_py_jl)

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 60)
println("Debug Complete")
println("=" ^ 60)
