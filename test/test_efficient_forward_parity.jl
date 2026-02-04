# Test parity between efficient forward and original forward pass

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Flux
using CUDA
using Statistics
using Random

Random.seed!(42)

println("=" ^ 60)
println("Efficient Forward Parity Test")
println("=" ^ 60)

# Check GPU
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: ", CUDA.device())
    use_gpu = true
else
    println("CUDA not functional, running on CPU")
    use_gpu = false
end

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
weights_path = joinpath(@__DIR__, "..", "weights", "score_network.npz"))
if isfile(weights_path)
    println("Loading pretrained weights...")
    load_score_network_weights!(model, weights_path)
end

# Create test batch
println("\n=== Creating Test Batch (CPU) ===")
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
mask[end-1:end, :] .= 0f0  # Mask out last 2 positions

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

# ============================================================
# Test on CPU first
# ============================================================
println("\n=== CPU Parity Test ===")

# Original forward
println("Running original forward...")
orig_output = model(batch)
orig_ca = orig_output[:bb_ca][:v]
orig_ll = orig_output[:local_latents][:v]

println("Original output shapes:")
println("  bb_ca: ", size(orig_ca))
println("  local_latents: ", size(orig_ll))

# Efficient forward
println("Running efficient forward...")
eff_batch = to_efficient_batch(batch)
eff_output = forward_efficient(model, eff_batch)
eff_ca = eff_output[:bb_ca][:v]
eff_ll = eff_output[:local_latents][:v]

println("Efficient output shapes:")
println("  bb_ca: ", size(eff_ca))
println("  local_latents: ", size(eff_ll))

# Compare
diff_ca = maximum(abs.(orig_ca .- eff_ca))
diff_ll = maximum(abs.(orig_ll .- eff_ll))

println("\nCPU Parity:")
println("  CA max diff: ", diff_ca)
println("  LL max diff: ", diff_ll)

tolerance = 1e-4
cpu_passed = diff_ca < tolerance && diff_ll < tolerance
if cpu_passed
    println("  CPU PASSED!")
else
    println("  CPU FAILED!")
end

# ============================================================
# Test on GPU
# ============================================================
if use_gpu
    println("\n=== GPU Parity Test ===")

    # Move model to GPU
    model_gpu = model |> gpu

    # Move batch to GPU
    batch_gpu = Dict{Symbol, Any}(
        :x_t => Dict(
            :bb_ca => gpu(x_t[:bb_ca]),
            :local_latents => gpu(x_t[:local_latents])
        ),
        :t => Dict(
            :bb_ca => gpu(t[:bb_ca]),
            :local_latents => gpu(t[:local_latents])
        ),
        :mask => gpu(mask)
    )

    # Original forward on GPU
    println("Running original forward on GPU...")
    orig_output_gpu = model_gpu(batch_gpu)
    orig_ca_gpu = orig_output_gpu[:bb_ca][:v]
    orig_ll_gpu = orig_output_gpu[:local_latents][:v]

    # Efficient forward on GPU
    println("Running efficient forward on GPU...")
    eff_batch_gpu = to_efficient_batch(batch_gpu)
    eff_output_gpu = forward_efficient(model_gpu, eff_batch_gpu)
    eff_ca_gpu = eff_output_gpu[:bb_ca][:v]
    eff_ll_gpu = eff_output_gpu[:local_latents][:v]

    # Compare (bring back to CPU)
    diff_ca_gpu = maximum(abs.(cpu(orig_ca_gpu) .- cpu(eff_ca_gpu)))
    diff_ll_gpu = maximum(abs.(cpu(orig_ll_gpu) .- cpu(eff_ll_gpu)))

    println("\nGPU Parity:")
    println("  CA max diff: ", diff_ca_gpu)
    println("  LL max diff: ", diff_ll_gpu)

    gpu_passed = diff_ca_gpu < tolerance && diff_ll_gpu < tolerance
    if gpu_passed
        println("  GPU PASSED!")
    else
        println("  GPU FAILED!")
    end

    # Also verify CPU and GPU give same results for original model
    diff_cpu_gpu_ca = maximum(abs.(orig_ca .- cpu(orig_ca_gpu)))
    diff_cpu_gpu_ll = maximum(abs.(orig_ll .- cpu(orig_ll_gpu)))
    println("\nCPU vs GPU (original model):")
    println("  CA max diff: ", diff_cpu_gpu_ca)
    println("  LL max diff: ", diff_cpu_gpu_ll)

    # Clean up GPU memory
    CUDA.reclaim()
else
    gpu_passed = true  # Skip if no GPU
end

# ============================================================
# Test with multiple random batches
# ============================================================
println("\n=== Testing Multiple Random Batches ===")
n_tests = 5
all_passed = true

for i in 1:n_tests
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

    # Original
    orig_out = model(batch_test)

    # Efficient
    eff_out = forward_efficient(model, to_efficient_batch(batch_test))

    max_diff = max(
        maximum(abs.(orig_out[:bb_ca][:v] .- eff_out[:bb_ca][:v])),
        maximum(abs.(orig_out[:local_latents][:v] .- eff_out[:local_latents][:v]))
    )

    status = max_diff < tolerance ? "OK" : "FAIL"
    println("  Batch $i (L=$L_test, B=$B_test): max_diff=$max_diff [$status]")

    if max_diff >= tolerance
        all_passed = false
    end
end

# ============================================================
# Summary
# ============================================================
println("\n" * "=" ^ 60)
if cpu_passed && gpu_passed && all_passed
    println("ALL PARITY TESTS PASSED!")
else
    println("SOME TESTS FAILED")
    exit(1)
end
println("=" ^ 60)
