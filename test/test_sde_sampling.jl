#!/usr/bin/env julia
"""
Test SDE sampling with Flowfusion gen() API.
Compares ODE vs SDE sampling results with per-modality settings.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Random

# Set random seed
Random.seed!(789)

# Paths
weights_dir = joinpath(@__DIR__, "..", "weights")
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

# Helper to print CA-CA stats
function print_ca_stats(ca_coords, label)
    B = size(ca_coords, 3)
    L = size(ca_coords, 2)
    println("$label CA coords shape: ", size(ca_coords))
    for b in 1:B
        dists = [sqrt(sum((ca_coords[:, i+1, b] .- ca_coords[:, i, b]).^2)) for i in 1:(L-1)]
        mean_d = sum(dists) / length(dists)
        println("  Sample $b: Mean CA-CA = $(round(mean_d, digits=3)) nm")
    end
end

println("\n=== Test 1: ODE Sampling (deterministic) ===")
Random.seed!(100)
@time ode_samples = generate_with_flowfusion(score_net, L, B;
    nsteps=nsteps,
    self_cond=true,
    ca_schedule_mode=:power,
    ca_schedule_p=2.0,
    ca_sc_scale_noise=0.0,  # No noise = ODE
    ll_schedule_mode=:power,
    ll_schedule_p=2.0,
    ll_sc_scale_noise=0.0   # No noise = ODE
)

ca_ode = ode_samples[:bb_ca]
print_ca_stats(ca_ode, "ODE")

println("\n=== Test 2: SDE Sampling with constant noise ===")
Random.seed!(100)
@time sde_const_samples = generate_with_flowfusion(score_net, L, B;
    nsteps=nsteps,
    self_cond=true,
    ca_schedule_mode=:power,
    ca_schedule_p=2.0,
    ca_gt_mode=:const,
    ca_gt_param=1.0,
    ca_sc_scale_noise=0.1,  # Constant noise
    ll_schedule_mode=:power,
    ll_schedule_p=2.0,
    ll_gt_mode=:const,
    ll_gt_param=1.0,
    ll_sc_scale_noise=0.1   # Constant noise
)

ca_sde_const = sde_const_samples[:bb_ca]
print_ca_stats(ca_sde_const, "SDE const")

println("\n=== Test 3: SDE Sampling with la-proteina defaults ===")
println("CA: gt_mode=1/t, LL: gt_mode=tan")
Random.seed!(100)
@time sde_laproteina_samples = generate_with_flowfusion(score_net, L, B;
    nsteps=nsteps,
    self_cond=true,
    # la-proteina CA defaults
    ca_schedule_mode=:log,
    ca_schedule_p=2.0,
    ca_gt_mode=Symbol("1/t"),
    ca_gt_param=1.0,
    ca_sc_scale_noise=0.1,
    ca_t_lim_ode=0.98,
    # la-proteina LL defaults
    ll_schedule_mode=:power,
    ll_schedule_p=2.0,
    ll_gt_mode=:tan,
    ll_gt_param=1.0,
    ll_sc_scale_noise=0.1,
    ll_t_lim_ode=0.98
)

ca_sde_laproteina = sde_laproteina_samples[:bb_ca]
print_ca_stats(ca_sde_laproteina, "SDE la-proteina")

# Compare differences
println("\n=== Comparing Results ===")
ode_vs_sde_const = sum(abs.(ca_ode .- ca_sde_const)) / length(ca_ode)
ode_vs_sde_laproteina = sum(abs.(ca_ode .- ca_sde_laproteina)) / length(ca_ode)
sde_const_vs_laproteina = sum(abs.(ca_sde_const .- ca_sde_laproteina)) / length(ca_ode)

println("Mean absolute difference:")
println("  ODE vs SDE const: $(round(ode_vs_sde_const, digits=4))")
println("  ODE vs SDE la-proteina: $(round(ode_vs_sde_laproteina, digits=4))")
println("  SDE const vs la-proteina: $(round(sde_const_vs_laproteina, digits=4))")

if ode_vs_sde_const > 0.01 && ode_vs_sde_laproteina > 0.01
    println("\nSDE sampling is producing different results than ODE - noise injection working!")
else
    println("\nWARNING: SDE and ODE results are too similar - check noise injection")
end

println("\n=== Done! ===")
