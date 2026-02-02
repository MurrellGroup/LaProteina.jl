#!/usr/bin/env julia
"""
Test sampling with exact la-proteina Python defaults.
"""

using Pkg
Pkg.activate("/home/claudey/JuProteina/JuProteina")

using JuProteina
using Random

Random.seed!(42)

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

println("\n=== Test 1: La-proteina defaults (SDE with per-modality settings) ===")
println("CA: gt_mode=1/t, schedule=log, sc_scale_noise=0.1")
println("LL: gt_mode=tan, schedule=power, sc_scale_noise=0.1")
println("Both: t_lim_ode=0.98, nsteps=400")

Random.seed!(100)
@time samples_sde = generate_with_flowfusion(score_net, L, B;
    nsteps=400,  # Full 400 steps like Python
    self_cond=true,
    # CA defaults
    ca_schedule_mode=:log,
    ca_schedule_p=2.0,
    ca_gt_mode=Symbol("1/t"),
    ca_gt_param=1.0,
    ca_sc_scale_noise=0.1,
    ca_sc_scale_score=1.0,
    ca_t_lim_ode=0.98,
    # LL defaults
    ll_schedule_mode=:power,
    ll_schedule_p=2.0,
    ll_gt_mode=:tan,
    ll_gt_param=1.0,
    ll_sc_scale_noise=0.1,
    ll_sc_scale_score=1.0,
    ll_t_lim_ode=0.98
)

ca_sde = samples_sde[:bb_ca]
println("SDE CA coords shape: ", size(ca_sde))
for b in 1:B
    dists = [sqrt(sum((ca_sde[:, i+1, b] .- ca_sde[:, i, b]).^2)) for i in 1:(L-1)]
    println("  Sample $b: Mean CA-CA = $(round(sum(dists)/length(dists), digits=3)) nm")
end

println("\n=== Test 2: Pure ODE (no noise) ===")
Random.seed!(100)
@time samples_ode = generate_with_flowfusion(score_net, L, B;
    nsteps=400,
    self_cond=true,
    ca_sc_scale_noise=0.0,  # No noise
    ll_sc_scale_noise=0.0   # No noise
)

ca_ode = samples_ode[:bb_ca]
println("ODE CA coords shape: ", size(ca_ode))
for b in 1:B
    dists = [sqrt(sum((ca_ode[:, i+1, b] .- ca_ode[:, i, b]).^2)) for i in 1:(L-1)]
    println("  Sample $b: Mean CA-CA = $(round(sum(dists)/length(dists), digits=3)) nm")
end

println("\n=== Comparison ===")
diff = sum(abs.(ca_sde .- ca_ode)) / length(ca_sde)
println("Mean absolute difference SDE vs ODE: $(round(diff, digits=4))")
if diff > 0.01
    println("SDE is producing different samples than ODE - noise injection working!")
else
    println("WARNING: SDE and ODE too similar")
end

println("\n=== Done! ===")
