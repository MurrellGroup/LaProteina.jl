# GPU Inference Test with Efficient Forward

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Flux
using CUDA
using Statistics
using Random

Random.seed!(42)

println("=" ^ 60)
println("GPU Inference Test with Efficient Forward")
println("=" ^ 60)

# Create model
println("\n=== Creating ScoreNetwork ===")
latent_dim = 8
model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)

weights_path = joinpath(@__DIR__, "..", "weights", "score_network.npz")
if isfile(weights_path)
    println("Loading pretrained weights...")
    load_score_network_weights!(model, weights_path)
end

model_gpu = model |> gpu

# Create test batch on GPU
println("\n=== Creating GPU Batch ===")
L = 50
B = 4
T = Float32

x_t = Dict(
    :bb_ca => CUDA.randn(T, 3, L, B) * 0.1f0,
    :local_latents => CUDA.randn(T, latent_dim, L, B) * 0.1f0
)
t = Dict(:bb_ca => CUDA.rand(T, B), :local_latents => CUDA.rand(T, B))
mask = CUDA.ones(T, L, B)
mask[end-5:end, :] .= 0f0

batch_gpu = Dict{Symbol, Any}(:x_t => x_t, :t => t, :mask => mask)

println("Input shapes: L=$L, B=$B")

# Timing comparison
println("\n=== Timing Comparison ===")

# Warmup
_ = model_gpu(batch_gpu)
eff_batch = to_efficient_batch(batch_gpu)
_ = forward_efficient(model_gpu, eff_batch)
CUDA.synchronize()

# Time original
println("Timing original forward (10 iters)...")
CUDA.@elapsed begin
    for _ in 1:10
        _ = model_gpu(batch_gpu)
    end
    CUDA.synchronize()
end
t_orig = @elapsed begin
    for _ in 1:10
        _ = model_gpu(batch_gpu)
    end
    CUDA.synchronize()
end
println("  Original: $(round(t_orig/10 * 1000, digits=2)) ms/iter")

# Time efficient
println("Timing efficient forward (10 iters)...")
t_eff = @elapsed begin
    for _ in 1:10
        eff_batch = to_efficient_batch(batch_gpu)
        _ = forward_efficient(model_gpu, eff_batch)
    end
    CUDA.synchronize()
end
println("  Efficient: $(round(t_eff/10 * 1000, digits=2)) ms/iter")

println("\n=== GPU Memory Usage ===")
CUDA.reclaim()
mem_before = CUDA.used_memory()
_ = model_gpu(batch_gpu)
CUDA.synchronize()
mem_orig = CUDA.used_memory() - mem_before

CUDA.reclaim()
mem_before = CUDA.used_memory()
eff_batch = to_efficient_batch(batch_gpu)
_ = forward_efficient(model_gpu, eff_batch)
CUDA.synchronize()
mem_eff = CUDA.used_memory() - mem_before

println("  Original peak: $(round(mem_orig / 1e6, digits=2)) MB")
println("  Efficient peak: $(round(mem_eff / 1e6, digits=2)) MB")

println("\n" * "=" ^ 60)
println("GPU Inference Test COMPLETED")
println("=" ^ 60)
