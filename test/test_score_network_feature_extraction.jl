# Test that extract_features + forward_from_features gives identical output to full model

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Statistics
using Random

Random.seed!(42)

println("=" ^ 60)
println("ScoreNetwork Feature Extraction Parity Test")
println("=" ^ 60)

# Create ScoreNetwork
println("\n=== Creating ScoreNetwork ===")
latent_dim = 8
model = ScoreNetwork(
    n_layers=14,
    token_dim=768,
    pair_dim=256,
    n_heads=12,
    dim_cond=256,
    latent_dim=latent_dim,
    output_param=:v,
    qk_ln=true,
    update_pair_repr=false
)

# Load pretrained weights
weights_path = joinpath(@__DIR__, "..", "weights", "score_network.npz")
if isfile(weights_path)
    println("Loading pretrained weights...")
    load_score_network_weights!(model, weights_path)
end

# Create test batch
println("\n=== Creating Test Batch ===")
L = 20
B = 2

x_t = Dict(
    :bb_ca => randn(Float32, 3, L, B) * 0.1f0,
    :local_latents => randn(Float32, latent_dim, L, B) * 0.1f0
)
t = Dict(
    :bb_ca => rand(Float32, B),
    :local_latents => rand(Float32, B)
)
mask = ones(Float32, L, B)
# Make a few positions masked out for testing
mask[end-1:end, :] .= 0f0

batch = Dict{Symbol, Any}(
    :x_t => x_t,
    :t => t,
    :mask => mask
)

println("Input shapes:")
println("  bb_ca: ", size(x_t[:bb_ca]))
println("  local_latents: ", size(x_t[:local_latents]))
println("  t: ", size(t[:bb_ca]))
println("  mask: ", size(mask), " ($(Int(sum(mask))) active residues)")

# Run full model
println("\n=== Running Full Model ===")
full_output = model(batch)
full_ca = full_output[:bb_ca][:v]
full_ll = full_output[:local_latents][:v]

println("Full model output shapes:")
println("  bb_ca: ", size(full_ca))
println("  local_latents: ", size(full_ll))

# Run with separated feature extraction
println("\n=== Running Separated Feature Extraction ===")
features = extract_features(model, batch)

println("Features shapes:")
println("  seq_features: ", size(features.seq_features))
println("  cond_features: ", size(features.cond_features))
println("  pair_features: ", size(features.pair_features))
println("  pair_cond: ", size(features.pair_cond))
println("  mask: ", size(features.mask))

sep_output = forward_from_features(model, features)
sep_ca = sep_output[:bb_ca][:v]
sep_ll = sep_output[:local_latents][:v]

println("\nSeparated output shapes:")
println("  bb_ca: ", size(sep_ca))
println("  local_latents: ", size(sep_ll))

# Compare outputs
println("\n=== Comparing Outputs ===")
diff_ca = abs.(full_ca .- sep_ca)
diff_ll = abs.(full_ll .- sep_ll)

max_diff_ca = maximum(diff_ca)
max_diff_ll = maximum(diff_ll)
mean_diff_ca = mean(diff_ca)
mean_diff_ll = mean(diff_ll)

println("CA output difference:")
println("  max: $(max_diff_ca)")
println("  mean: $(mean_diff_ca)")

println("\nLocal latents output difference:")
println("  max: $(max_diff_ll)")
println("  mean: $(mean_diff_ll)")

# Test passes if differences are essentially zero (floating point tolerance)
tolerance = 1e-5
passed = max_diff_ca < tolerance && max_diff_ll < tolerance

if passed
    println("\n" * "=" ^ 60)
    println("PASSED: Outputs match within tolerance $(tolerance)")
    println("=" ^ 60)
else
    println("\n" * "=" ^ 60)
    println("FAILED: Outputs differ more than tolerance $(tolerance)")
    println("=" ^ 60)
    exit(1)
end

# Test with multiple random batches
println("\n=== Testing Multiple Random Batches ===")
n_test_batches = 5
all_passed = true

for i in 1:n_test_batches
    L_test = rand(10:30)
    B_test = rand(1:4)

    x_t_test = Dict(
        :bb_ca => randn(Float32, 3, L_test, B_test) * 0.1f0,
        :local_latents => randn(Float32, latent_dim, L_test, B_test) * 0.1f0
    )
    t_test = Dict(
        :bb_ca => rand(Float32, B_test),
        :local_latents => rand(Float32, B_test)
    )
    mask_test = Float32.(rand(L_test, B_test) .> 0.1)  # Random mask

    batch_test = Dict{Symbol, Any}(
        :x_t => x_t_test,
        :t => t_test,
        :mask => mask_test
    )

    # Full model
    full_out = model(batch_test)

    # Separated
    feats = extract_features(model, batch_test)
    sep_out = forward_from_features(model, feats)

    max_diff = max(
        maximum(abs.(full_out[:bb_ca][:v] .- sep_out[:bb_ca][:v])),
        maximum(abs.(full_out[:local_latents][:v] .- sep_out[:local_latents][:v]))
    )

    status = max_diff < tolerance ? "OK" : "FAIL"
    println("  Batch $i (L=$L_test, B=$B_test): max_diff=$max_diff [$status]")

    if max_diff >= tolerance
        all_passed = false
    end
end

if all_passed
    println("\n" * "=" ^ 60)
    println("ALL TESTS PASSED!")
    println("extract_features + forward_from_features = model(batch)")
    println("=" ^ 60)
else
    println("\n" * "=" ^ 60)
    println("SOME TESTS FAILED")
    println("=" ^ 60)
    exit(1)
end

# ============================================================
# Test raw features API (for GPU training)
# ============================================================
println("\n" * "=" ^ 60)
println("Testing Raw Features API")
println("=" ^ 60)

println("\n=== Running Raw Features Extraction ===")
raw_features = extract_raw_features(model, batch)

println("Raw features shapes:")
println("  seq_raw: ", size(raw_features.seq_raw))
println("  cond_raw: ", size(raw_features.cond_raw))
println("  pair_raw: ", size(raw_features.pair_raw))
println("  pair_cond_raw: ", size(raw_features.pair_cond_raw))
println("  mask: ", size(raw_features.mask))

raw_output = forward_from_raw_features(model, raw_features)
raw_ca = raw_output[:bb_ca][:v]
raw_ll = raw_output[:local_latents][:v]

println("\nRaw features output shapes:")
println("  bb_ca: ", size(raw_ca))
println("  local_latents: ", size(raw_ll))

# Compare with full model
println("\n=== Comparing Raw Features Output to Full Model ===")
diff_ca_raw = abs.(full_ca .- raw_ca)
diff_ll_raw = abs.(full_ll .- raw_ll)

max_diff_ca_raw = maximum(diff_ca_raw)
max_diff_ll_raw = maximum(diff_ll_raw)

println("CA output difference:")
println("  max: $(max_diff_ca_raw)")

println("\nLocal latents output difference:")
println("  max: $(max_diff_ll_raw)")

raw_passed = max_diff_ca_raw < tolerance && max_diff_ll_raw < tolerance

if raw_passed
    println("\n" * "=" ^ 60)
    println("RAW FEATURES API TEST PASSED!")
    println("extract_raw_features + forward_from_raw_features = model(batch)")
    println("=" ^ 60)
else
    println("\n" * "=" ^ 60)
    println("RAW FEATURES API TEST FAILED!")
    println("=" ^ 60)
    exit(1)
end

# Test with multiple random batches
println("\n=== Testing Raw Features with Multiple Random Batches ===")
raw_all_passed = true

for i in 1:n_test_batches
    L_test = rand(10:30)
    B_test = rand(1:4)

    x_t_test = Dict(
        :bb_ca => randn(Float32, 3, L_test, B_test) * 0.1f0,
        :local_latents => randn(Float32, latent_dim, L_test, B_test) * 0.1f0
    )
    t_test = Dict(
        :bb_ca => rand(Float32, B_test),
        :local_latents => rand(Float32, B_test)
    )
    mask_test = Float32.(rand(L_test, B_test) .> 0.1)

    batch_test = Dict{Symbol, Any}(
        :x_t => x_t_test,
        :t => t_test,
        :mask => mask_test
    )

    # Full model
    full_out = model(batch_test)

    # Raw features
    raw_feats = extract_raw_features(model, batch_test)
    raw_out = forward_from_raw_features(model, raw_feats)

    max_diff = max(
        maximum(abs.(full_out[:bb_ca][:v] .- raw_out[:bb_ca][:v])),
        maximum(abs.(full_out[:local_latents][:v] .- raw_out[:local_latents][:v]))
    )

    status = max_diff < tolerance ? "OK" : "FAIL"
    println("  Batch $i (L=$L_test, B=$B_test): max_diff=$max_diff [$status]")

    if max_diff >= tolerance
        raw_all_passed = false
    end
end

if raw_all_passed
    println("\n" * "=" ^ 60)
    println("ALL RAW FEATURES TESTS PASSED!")
    println("=" ^ 60)
else
    println("\n" * "=" ^ 60)
    println("SOME RAW FEATURES TESTS FAILED!")
    println("=" ^ 60)
    exit(1)
end
