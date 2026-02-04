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

println("=" ^ 60)
println("GPU Utilization Test - Sustained Training")
println("=" ^ 60)

println("\nLoading precomputed data...")
proteins = load_precomputed_shard(joinpath(@__DIR__, "smol_demo_training_data.jld2"))
println("Loaded $(length(proteins)) proteins")

sampler = LengthBucketedSampler(proteins; bucket_size=32)
println("Created bucketed sampler")

println("\nLoading ScoreNetwork...")
latent_dim = 8
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
load_score_network_weights!(score_net, joinpath(@__DIR__, "..", "weights", "score_network.npz"))
score_net_gpu = score_net |> gpu
score_net = nothing
GC.gc()
println("ScoreNetwork on GPU")

P_ca = RDNFlow(3; zero_com=true, sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

opt_state = Optimisers.setup(Adam(1e-5), score_net_gpu)

batch_size = 8
n_batches = 100

println("\n" * "-" ^ 40)
println("Running $n_batches batches (batch_size=$batch_size)")
println("Monitor with: watch -n0.5 nvidia-smi")
println("-" ^ 40)

# Warmup
println("Warming up...")
for _ in 1:10
    indices = sample_batch(sampler, batch_size)
    batch = batch_from_precomputed(proteins, indices, P)
    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])
end
CUDA.synchronize()
GC.gc(); CUDA.reclaim()

# Sustained training
println("Starting sustained training...")
losses = Float32[]
times = Float64[]

total_start = time()
for i in 1:n_batches
    t0 = time()

    indices = sample_batch(sampler, batch_size)
    batch = batch_from_precomputed(proteins, indices, P)

    loss, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])

    # Don't synchronize every batch - let GPU pipeline
    if i % 10 == 0
        CUDA.synchronize()
        push!(losses, Float32(cpu(loss)))
        push!(times, (time() - total_start) / i)
        @printf("Batch %d: loss=%.4f, avg_time=%.0fms, throughput=%.1f samples/sec\n",
                i, losses[end], times[end]*1000, batch_size / times[end])
    end
end

CUDA.synchronize()
total_time = time() - total_start

println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
@printf("Total time: %.1f s\n", total_time)
@printf("Total samples: %d\n", batch_size * n_batches)
@printf("Throughput: %.1f samples/sec\n", batch_size * n_batches / total_time)
@printf("Avg batch time: %.0f ms\n", total_time / n_batches * 1000)
@printf("Final loss: %.4f\n", losses[end])
