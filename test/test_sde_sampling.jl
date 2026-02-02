#!/usr/bin/env julia
"""
Test SDE sampling with Flowfusion gen() API.
Compares ODE vs SDE sampling results.
"""

using Pkg
Pkg.activate("/home/claudey/JuProteina/JuProteina")

using JuProteina
using Random

# Set random seed
Random.seed!(789)

# Paths
weights_dir = "/home/claudey/JuProteina/JuProteina/weights"
score_net_path = joinpath(weights_dir, "score_network.npz")

println("=== Loading ScoreNetwork ===")
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=256, t_emb_dim=256, pair_dim=256
)
load_score_network_weights!(score_net, score_net_path)
println("Done!")

L = 80
B = 2
nsteps = 50

println("\n=== Test 1: ODE Sampling (deterministic) ===")
Random.seed!(100)
@time ode_samples = generate_with_flowfusion(score_net, L, B;
    nsteps=nsteps,
    self_cond=true,
    schedule_mode=:power,
    schedule_p=2.0,
    sde_gt_mode=:const,
    sde_gt_param=0.0  # No noise = ODE
)

ca_ode = ode_samples[:bb_ca]
println("ODE CA coords shape: ", size(ca_ode))

# Check CA-CA distances
for b in 1:B
    dists = [sqrt(sum((ca_ode[:, i+1, b] .- ca_ode[:, i, b]).^2)) for i in 1:(L-1)]
    println("  Sample $b: Mean CA-CA = $(round(sum(dists)/length(dists), digits=3)) nm")
end

println("\n=== Test 2: SDE Sampling with constant noise ===")
Random.seed!(100)
@time sde_const_samples = generate_with_flowfusion(score_net, L, B;
    nsteps=nsteps,
    self_cond=true,
    schedule_mode=:power,
    schedule_p=2.0,
    sde_gt_mode=:const,
    sde_gt_param=0.1  # Constant noise
)

ca_sde_const = sde_const_samples[:bb_ca]
println("SDE (const) CA coords shape: ", size(ca_sde_const))

for b in 1:B
    dists = [sqrt(sum((ca_sde_const[:, i+1, b] .- ca_sde_const[:, i, b]).^2)) for i in 1:(L-1)]
    println("  Sample $b: Mean CA-CA = $(round(sum(dists)/length(dists), digits=3)) nm")
end

println("\n=== Test 3: SDE Sampling with tangent noise schedule ===")
Random.seed!(100)
@time sde_tan_samples = generate_with_flowfusion(score_net, L, B;
    nsteps=nsteps,
    self_cond=true,
    schedule_mode=:power,
    schedule_p=2.0,
    sde_gt_mode=:tan,
    sde_gt_param=0.3  # Tangent schedule
)

ca_sde_tan = sde_tan_samples[:bb_ca]
println("SDE (tan) CA coords shape: ", size(ca_sde_tan))

for b in 1:B
    dists = [sqrt(sum((ca_sde_tan[:, i+1, b] .- ca_sde_tan[:, i, b]).^2)) for i in 1:(L-1)]
    println("  Sample $b: Mean CA-CA = $(round(sum(dists)/length(dists), digits=3)) nm")
end

# Compare differences
println("\n=== Comparing Results ===")
ode_vs_sde_const = sum(abs.(ca_ode .- ca_sde_const)) / length(ca_ode)
ode_vs_sde_tan = sum(abs.(ca_ode .- ca_sde_tan)) / length(ca_ode)
sde_const_vs_tan = sum(abs.(ca_sde_const .- ca_sde_tan)) / length(ca_ode)

println("Mean absolute difference:")
println("  ODE vs SDE(const): $(round(ode_vs_sde_const, digits=4))")
println("  ODE vs SDE(tan): $(round(ode_vs_sde_tan, digits=4))")
println("  SDE(const) vs SDE(tan): $(round(sde_const_vs_tan, digits=4))")

if ode_vs_sde_const > 0.01 && ode_vs_sde_tan > 0.01
    println("\nSDE sampling is producing different results than ODE - noise injection working!")
else
    println("\nWARNING: SDE and ODE results are too similar - check noise injection")
end

println("\n=== Done! ===")
