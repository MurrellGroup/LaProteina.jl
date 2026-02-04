using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Flux
using CUDA
using Statistics
using Random
using Printf
using JLD2

import Flowfusion
import Flowfusion: RDNFlow

Random.seed!(42)

println("=" ^ 60)
println("Debug Loss - Single Sample Testing")
println("=" ^ 60)

println("\nLoading precomputed data...")
proteins = load_precomputed_shard(joinpath(@__DIR__, "smol_demo_training_data.jld2"))
println("Loaded $(length(proteins)) proteins")

println("\nLoading ScoreNetwork...")
latent_dim = 8
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
load_score_network_weights!(score_net, joinpath(@__DIR__, "..", "weights", "score_network.npz"))
score_net_gpu = score_net |> gpu
println("ScoreNetwork on GPU")

P_ca = RDNFlow(3; zero_com=true, sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

println("\n" * "-" ^ 40)
println("Testing batch_size=1 with various t values")
println("-" ^ 40)

# Test with fixed t values
for t_val in [0.1f0, 0.5f0, 0.9f0, 0.95f0, 0.99f0]
    indices = [1]  # Single sample
    batch = batch_from_precomputed(proteins, indices, P)

    # Override t with fixed value
    t_fixed = CUDA.fill(t_val, 1)

    loss = efficient_flow_loss_gpu(score_net_gpu, batch.xt_ca, batch.xt_ll,
                                    batch.x1_ca, batch.x1_ll, t_fixed, batch.mask)

    # Compute weight manually
    weight = 1f0 / ((1f0 - t_val)^2 + 1f-5)

    @printf("t=%.2f: loss=%.4f, time_weight=%.1f\n", t_val, cpu(loss), weight)
end

println("\n" * "-" ^ 40)
println("Testing batch_size=1 with random t (20 samples)")
println("-" ^ 40)

losses = Float32[]
t_vals = Float32[]
for i in 1:20
    indices = [rand(1:length(proteins))]
    batch = batch_from_precomputed(proteins, indices, P)

    loss = efficient_flow_loss_gpu(score_net_gpu, batch.xt_ca, batch.xt_ll,
                                    batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    t_val = cpu(batch.t)[1]

    push!(losses, Float32(cpu(loss)))
    push!(t_vals, t_val)

    @printf("Sample %2d: t=%.3f, loss=%.4f\n", i, t_val, losses[end])
end

println("\nSummary:")
@printf("Loss range: %.4f - %.4f\n", minimum(losses), maximum(losses))
@printf("t range: %.3f - %.3f\n", minimum(t_vals), maximum(t_vals))

# Find correlation
high_loss_idx = findall(losses .> 10)
if !isempty(high_loss_idx)
    println("\nHigh loss samples (loss > 10):")
    for idx in high_loss_idx
        @printf("  Sample %d: t=%.3f, loss=%.4f\n", idx, t_vals[idx], losses[idx])
    end
end

println("\n" * "-" ^ 40)
println("Testing batch_size=8 vs 8x batch_size=1")
println("-" ^ 40)

# Batch of 8
indices = collect(1:8)
batch = batch_from_precomputed(proteins, indices, P)
loss_batch8 = cpu(efficient_flow_loss_gpu(score_net_gpu, batch.xt_ca, batch.xt_ll,
                                           batch.x1_ca, batch.x1_ll, batch.t, batch.mask))
println("Batch size 8: loss = $loss_batch8")
println("  Sequence length in batch: $(size(batch.xt_ca, 2))")
println("  t values: ", round.(cpu(batch.t), digits=3))

# 8 individual samples
individual_losses = Float32[]
for i in 1:8
    indices_single = [i]
    batch_single = batch_from_precomputed(proteins, indices_single, P)
    loss_single = cpu(efficient_flow_loss_gpu(score_net_gpu, batch_single.xt_ca, batch_single.xt_ll,
                                               batch_single.x1_ca, batch_single.x1_ll, batch_single.t, batch_single.mask))
    push!(individual_losses, Float32(loss_single))
    println("  Sample $i (L=$(size(batch_single.xt_ca, 2))): loss = $(round(loss_single, digits=4))")
end

println("\nMean of 8 individual losses: $(mean(individual_losses))")
println("Batched loss: $loss_batch8")
