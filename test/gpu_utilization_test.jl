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

println("=" ^ 60)
println("GPU Utilization Test - Precomputed Training")
println("=" ^ 60)

println("\nLoading precomputed data...")
proteins = load_precomputed_shard(joinpath(@__DIR__, "smol_demo_training_data.jld2"))
println("Loaded $(length(proteins)) proteins")

# Check data sizes
sample = proteins[1]
println("Sample shape: CA=$(size(sample.ca_coords)), z=$(size(sample.z_mean))")

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

# Test different batch sizes
for batch_size in [4, 8, 16]
    println("\n" * "-" ^ 40)
    println("Testing batch_size = $batch_size")
    println("-" ^ 40)

    n_batches = 30

    # Warmup
    for _ in 1:5
        indices = rand(1:length(proteins), batch_size)
        batch = batch_from_precomputed(proteins, indices, P)
        _, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])
    end
    CUDA.synchronize()
    GC.gc(); CUDA.reclaim()

    # Timed run
    losses = Float32[]
    times = Float64[]

    total_start = time()
    for i in 1:n_batches
        t0 = time()
        indices = rand(1:length(proteins), batch_size)
        batch = batch_from_precomputed(proteins, indices, P)

        loss, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])

        CUDA.synchronize()
        push!(losses, Float32(cpu(loss)))
        push!(times, time() - t0)
    end
    total_time = time() - total_start

    @printf("  Avg time: %.1f ms/batch\n", mean(times)*1000)
    @printf("  Throughput: %.1f samples/sec\n", batch_size * n_batches / total_time)
    @printf("  Loss: %.4f\n", mean(losses))

    GC.gc(); CUDA.reclaim()
end

println("\n" * "=" ^ 60)
println("Test complete")
println("=" ^ 60)
