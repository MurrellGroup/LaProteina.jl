#!/usr/bin/env julia
using Pkg
Pkg.activate("/home/claudey/JuProteina/JuProteina")
using JuProteina
using Random
using CUDA

println("=== Comparing GPU vs CPU for L=100 ===")
weights_dir = "/home/claudey/JuProteina/JuProteina/weights"
score_net = ScoreNetwork(n_layers=14, token_dim=768, n_heads=12, latent_dim=8, dim_cond=256, t_emb_dim=256, pair_dim=256)
load_score_network_weights!(score_net, joinpath(weights_dir, "score_network.npz"))

# CPU run
println("\nCPU run (L=100):")
Random.seed!(42)
@time cpu_samples = generate_with_flowfusion(score_net, 100, 1; nsteps=400, self_cond=true, dev=identity)
ca_cpu = cpu_samples[:bb_ca]
dists_cpu = [sqrt(sum((ca_cpu[:, i+1, 1] .- ca_cpu[:, i, 1]).^2)) for i in 1:99]
println("  Mean CA-CA: ", round(sum(dists_cpu)/length(dists_cpu), digits=4))
println("  Std CA-CA: ", round(sqrt(sum((dists_cpu .- sum(dists_cpu)/length(dists_cpu)).^2)/length(dists_cpu)), digits=4))
println("  Min CA-CA: ", round(minimum(dists_cpu), digits=4))
println("  Max CA-CA: ", round(maximum(dists_cpu), digits=4))

# GPU run
score_net_gpu = score_net |> gpu
println("\nGPU run (L=100):")
Random.seed!(42)
CUDA.seed!(42)
@time gpu_samples = generate_with_flowfusion(score_net_gpu, 100, 1; nsteps=400, self_cond=true, dev=gpu)
ca_gpu = gpu_samples[:bb_ca]
dists_gpu = [sqrt(sum((ca_gpu[:, i+1, 1] .- ca_gpu[:, i, 1]).^2)) for i in 1:99]
println("  Mean CA-CA: ", round(sum(dists_gpu)/length(dists_gpu), digits=4))
println("  Std CA-CA: ", round(sqrt(sum((dists_gpu .- sum(dists_gpu)/length(dists_gpu)).^2)/length(dists_gpu)), digits=4))
println("  Min CA-CA: ", round(minimum(dists_gpu), digits=4))
println("  Max CA-CA: ", round(maximum(dists_gpu), digits=4))

println("\nDifference CPU vs GPU (L=100):")
diff = maximum(abs.(ca_cpu .- ca_gpu))
println("  Max coord diff: ", diff)

# Now test longer sequence on CPU to see if model handles it
println("\n\n=== Testing L=250 on CPU ===")
Random.seed!(42)
@time cpu_samples_250 = generate_with_flowfusion(score_net, 250, 1; nsteps=400, self_cond=true, dev=identity)
ca_cpu_250 = cpu_samples_250[:bb_ca]
dists_cpu_250 = [sqrt(sum((ca_cpu_250[:, i+1, 1] .- ca_cpu_250[:, i, 1]).^2)) for i in 1:249]
println("  Mean CA-CA: ", round(sum(dists_cpu_250)/length(dists_cpu_250), digits=4))
println("  Std CA-CA: ", round(sqrt(sum((dists_cpu_250 .- sum(dists_cpu_250)/length(dists_cpu_250)).^2)/length(dists_cpu_250)), digits=4))
println("  Min CA-CA: ", round(minimum(dists_cpu_250), digits=4))
println("  Max CA-CA: ", round(maximum(dists_cpu_250), digits=4))
