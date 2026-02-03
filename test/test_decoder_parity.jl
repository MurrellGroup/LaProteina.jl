#!/usr/bin/env julia
"""
Test decoder parity - run a forward pass through the decoder
with trained weights and compare outputs with Python reference.

Run generate_decoder_reference.py first to create the reference data.
"""

using Pkg
Pkg.activate("/home/claudey/JuProteina/JuProteina")

using JuProteina
using NPZ
using Flux
using LinearAlgebra
using Statistics

# Paths
test_dir = "/home/claudey/JuProteina/JuProteina/test"
weights_path = "/home/claudey/JuProteina/JuProteina/weights/decoder.npz"

# Check if reference data exists
if !isfile(joinpath(test_dir, "decoder_z_latent.npy"))
    error("Reference data not found. Run generate_decoder_reference.py first.")
end

# Load Python reference data
println("Loading Python reference data...")
z_latent_py = npzread(joinpath(test_dir, "decoder_z_latent.npy"))  # [B, L, latent_dim]
ca_coors_py = npzread(joinpath(test_dir, "decoder_ca_coors.npy"))  # [B, L, 3]
mask_py = npzread(joinpath(test_dir, "decoder_mask.npy"))  # [B, L]

out_coors_py = npzread(joinpath(test_dir, "decoder_out_coors.npy"))  # [B, L, 37, 3]
out_logits_py = npzread(joinpath(test_dir, "decoder_out_logits.npy"))  # [B, L, 20]
out_aatype_py = npzread(joinpath(test_dir, "decoder_out_aatype.npy"))  # [B, L]
out_atom_mask_py = npzread(joinpath(test_dir, "decoder_out_atom_mask.npy"))  # [B, L, 37]

println("Python output shapes:")
println("  coors: ", size(out_coors_py))
println("  logits: ", size(out_logits_py))
println("  aatype: ", size(out_aatype_py))
println("  atom_mask: ", size(out_atom_mask_py))

println("\nPython output values (first residue):")
println("  coors[0,0,0,:]: ", out_coors_py[1, 1, 1, :])  # First atom
println("  coors[0,0,1,:]: ", out_coors_py[1, 1, 2, :])  # CA atom
println("  logits[0,0,:5]: ", out_logits_py[1, 1, 1:5])

# Convert to Julia format
# Python: [B, L, D] -> Julia: [D, L, B]
z_latent = python_to_julia(z_latent_py)  # [latent_dim, L, B]
ca_coors = python_to_julia(ca_coors_py)  # [3, L, B]
mask = python_to_julia_mask(mask_py)  # [L, B]

B = size(z_latent, 3)
L = size(z_latent, 2)
latent_dim = size(z_latent, 1)

println("\nInput shapes (Julia format):")
println("  z_latent: ", size(z_latent))
println("  ca_coors: ", size(ca_coors))
println("  mask: ", size(mask))

# Create the decoder with same config as Python
# From nn_130m.yaml decoder section:
# - nlayers: 12
# - token_dim: 768
# - pair_repr_dim: 256
# - nheads: 12
# - dim_cond: 128
# - abs_coors: False
# - use_qkln: True
# - update_pair_repr: False
# - seq_sep_dim: 127
# - latent_z_dim: 8
println("\nCreating DecoderTransformer...")
decoder = DecoderTransformer(
    n_layers=12,
    token_dim=768,
    pair_dim=256,
    n_heads=12,
    dim_cond=128,
    latent_dim=latent_dim,
    qk_ln=true,
    update_pair_repr=false,
    abs_coors=false
)

# Load weights
println("Loading weights...")
load_decoder_weights!(decoder, weights_path)

# Create batch dict
batch = Dict{Symbol, Any}(
    :z_latent => z_latent,
    :ca_coors => ca_coors,
    :mask => mask
)

# Forward pass
println("\nRunning forward pass...")
output = decoder(batch)

coors_jl = output[:coors]  # [3, 37, L, B]
logits_jl = output[:seq_logits]  # [20, L, B]
aatype_jl = output[:aatype_max]  # [L, B]
atom_mask_jl = output[:atom_mask]  # [37, L, B]

println("\nJulia output shapes:")
println("  coors: ", size(coors_jl))
println("  logits: ", size(logits_jl))
println("  aatype: ", size(aatype_jl))
println("  atom_mask: ", size(atom_mask_jl))

# Convert to Python format for comparison
# Julia: [3, 37, L, B] -> Python: [B, L, 37, 3]
coors_jl_py = permutedims(coors_jl, (4, 3, 2, 1))  # [B, L, 37, 3]
# Julia: [20, L, B] -> Python: [B, L, 20]
logits_jl_py = permutedims(logits_jl, (3, 2, 1))  # [B, L, 20]
# Julia: [L, B] -> Python: [B, L]
aatype_jl_py = permutedims(aatype_jl, (2, 1))  # [B, L]
# Julia: [37, L, B] -> Python: [B, L, 37]
atom_mask_jl_py = permutedims(atom_mask_jl, (3, 2, 1))  # [B, L, 37]

println("\nJulia output values (first residue):")
println("  coors[0,0,0,:]: ", coors_jl_py[1, 1, 1, :])
println("  coors[0,0,1,:]: ", coors_jl_py[1, 1, 2, :])
println("  logits[0,0,:5]: ", logits_jl_py[1, 1, 1:5])

# Compare outputs
diff_coors = abs.(coors_jl_py .- out_coors_py)
diff_logits = abs.(logits_jl_py .- out_logits_py)

# For atom coordinates, only compare where mask is valid
# Actually for now let's compare all
max_diff_coors = maximum(diff_coors)
max_diff_logits = maximum(diff_logits)
mean_diff_coors = mean(diff_coors)
mean_diff_logits = mean(diff_logits)

# Also check aatype matches (Julia is 1-indexed, Python is 0-indexed)
aatype_match = all(Int.(aatype_jl_py) .- 1 .== Int.(out_aatype_py))

println("\n" * "="^60)
println("PARITY RESULTS")
println("="^60)

println("Coordinate output:")
println("  Max diff: ", max_diff_coors)
println("  Mean diff: ", mean_diff_coors)
println("  Python[0,0,0,:]: ", out_coors_py[1, 1, 1, :])
println("  Julia[0,0,0,:]:  ", coors_jl_py[1, 1, 1, :])

println("\nCA coordinate check:")
println("  Input CA[0,0,:]: ", ca_coors_py[1, 1, :])
println("  Python CA[0,0,:]: ", out_coors_py[1, 1, 2, :])
println("  Julia CA[0,0,:]:  ", coors_jl_py[1, 1, 2, :])

println("\nLogits output:")
println("  Max diff: ", max_diff_logits)
println("  Mean diff: ", mean_diff_logits)
println("  Python[0,0,:5]: ", out_logits_py[1, 1, 1:5])
println("  Julia[0,0,:5]:  ", logits_jl_py[1, 1, 1:5])

println("\nSequence prediction (Julia 1-indexed, Python 0-indexed):")
println("  aatype match: ", aatype_match)
println("  Python (0-idx): ", Int.(out_aatype_py[1, 1:5]))
println("  Julia (1-idx):  ", Int.(aatype_jl_py[1, 1:5]))

# Check if parity is achieved
tol = 1e-3
parity_achieved = max_diff_coors < tol && max_diff_logits < tol

if parity_achieved
    println("\n✓ PARITY ACHIEVED! (tol=$tol)")
else
    println("\n✗ PARITY FAILED (tol=$tol)")
    println("  coors max diff: $max_diff_coors")
    println("  logits max diff: $max_diff_logits")
end

# Return test result for CI
exit(parity_achieved ? 0 : 1)
