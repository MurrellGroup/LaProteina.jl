using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
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

batch_size = 4  # Smaller batch
n_batches = 50  # Quick test

println("\nTraining $n_batches batches, batch_size=$batch_size")

# Warmup
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
times = Float64[]

for i in 1:n_batches
    t0 = time()
    indices = rand(1:length(proteins), batch_size)
    batch = batch_from_precomputed(proteins, indices, P)

    loss, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll,
                                 batch.t_ca, batch.t_ll, batch.t_model, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])

    push!(losses, Float32(cpu(loss)))
    push!(times, time() - t0)

    if i % 10 == 0
        CUDA.synchronize()
        @printf("Batch %d: loss=%.4f, time=%.0fms\n", i, losses[end], times[end]*1000)
        GC.gc()
    end
end

CUDA.synchronize()
println("\n=== Summary ===")
@printf("Avg loss: %.4f\n", mean(losses))
@printf("Avg time: %.1f ms/batch\n", mean(times)*1000)
@printf("Throughput: %.1f samples/sec\n", batch_size / mean(times))
