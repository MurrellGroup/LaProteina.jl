using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Flux
using CUDA
using Statistics
using Random
using Optimisers
using Printf
using JLD2

import Flowfusion
import Flowfusion: RDNFlow

Random.seed!(42)

println("Loading precomputed data...")
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
score_net = nothing  # Free CPU copy
GC.gc()
println("ScoreNetwork on GPU")

# RDNFlow with schedule parameters matching inference:
# - CA uses :log schedule (reaches high t quickly)
# - Latents use :power schedule (stays at low t longer)
P_ca = RDNFlow(3; zero_com=true, schedule=:log, schedule_param=2.0f0,
               sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, schedule=:power, schedule_param=2.0f0,
               sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

opt_state = Optimisers.setup(Adam(1e-5), score_net_gpu)

batch_size = 4
n_total_batches = 1100  # 50 first + 1000 middle + 50 last

println("\nTraining $n_total_batches batches, batch_size=$batch_size")

# Warmup (not counted)
println("Warmup...")
for _ in 1:3
    indices = rand(1:length(proteins), batch_size)
    batch = batch_from_precomputed(proteins, indices, P)
    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll,
                                 batch.t_ca, batch.t_ll, batch.t_model, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])
end
CUDA.synchronize()
GC.gc(); CUDA.reclaim()
println("Warmup done")

# Training
losses = Float32[]

for i in 1:n_total_batches
    indices = rand(1:length(proteins), batch_size)
    batch = batch_from_precomputed(proteins, indices, P)

    loss, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll,
                                 batch.t_ca, batch.t_ll, batch.t_model, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])

    push!(losses, Float32(cpu(loss)))

    if i % 100 == 0
        CUDA.synchronize()
        @printf("Batch %d: loss=%.4f\n", i, losses[end])
        GC.gc()
    end
end

CUDA.synchronize()

# Compute statistics
first_50 = losses[1:50]
last_50 = losses[end-49:end]

println("\n" * "="^60)
println("TRAINING LOSS COMPARISON")
println("="^60)
@printf("First 50 batches: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n",
        mean(first_50), std(first_50), minimum(first_50), maximum(first_50))
@printf("Last 50 batches:  mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n",
        mean(last_50), std(last_50), minimum(last_50), maximum(last_50))
@printf("\nChange: %.4f (%.2f%%)\n", mean(last_50) - mean(first_50),
        100 * (mean(last_50) - mean(first_50)) / mean(first_50))

# Also show some intermediate stats
println("\nLoss progression (every 100 batches):")
for i in 100:100:n_total_batches
    window = losses[max(1, i-49):i]
    @printf("  Batch %4d: avg loss = %.4f\n", i, mean(window))
end

println("\nDone!")
