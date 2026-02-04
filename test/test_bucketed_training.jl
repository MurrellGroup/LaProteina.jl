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
println("Bucketed Training Test")
println("=" ^ 60)

println("\nLoading precomputed data...")
proteins = load_precomputed_shard(joinpath(@__DIR__, "smol_demo_training_data.jld2"))
println("Loaded $(length(proteins)) proteins")

# Check length distribution
lengths = [length(p.mask) for p in proteins]
println("Lengths: min=$(minimum(lengths)), max=$(maximum(lengths)), mean=$(round(mean(lengths), digits=1))")

println("\nCreating length-bucketed sampler (bucket_size=32)...")
sampler = LengthBucketedSampler(proteins; bucket_size=32)
println("Created $(length(sampler.buckets)) buckets:")
for (i, (bucket, bound)) in enumerate(zip(sampler.buckets, sampler.bucket_bounds))
    if !isempty(bucket)
        println("  Bucket $i (≤$bound): $(length(bucket)) samples")
    end
end

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
n_batches = 50

println("\n" * "-" ^ 40)
println("Random sampling (baseline)")
println("-" ^ 40)

# Warmup
for _ in 1:5
    indices = rand(1:length(proteins), batch_size)
    batch = batch_from_precomputed(proteins, indices, P)
    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])
end
CUDA.synchronize(); GC.gc(); CUDA.reclaim()

# Timed random sampling
random_times = Float64[]
random_lengths = Int[]
total_start = time()
for i in 1:n_batches
    t0 = time()
    indices = rand(1:length(proteins), batch_size)
    batch = batch_from_precomputed(proteins, indices, P)
    min_len = size(batch.xt_ca, 2)

    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])

    CUDA.synchronize()
    push!(random_times, time() - t0)
    push!(random_lengths, min_len)
end
random_total = time() - total_start

@printf("  Avg time: %.1f ms/batch\n", mean(random_times)*1000)
@printf("  Throughput: %.1f samples/sec\n", batch_size * n_batches / random_total)
@printf("  Avg batch length: %.1f residues\n", mean(random_lengths))

GC.gc(); CUDA.reclaim()

println("\n" * "-" ^ 40)
println("Bucketed sampling")
println("-" ^ 40)

# Warmup
for _ in 1:5
    indices = sample_batch(sampler, batch_size)
    batch = batch_from_precomputed(proteins, indices, P)
    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])
end
CUDA.synchronize(); GC.gc(); CUDA.reclaim()

# Timed bucketed sampling
bucketed_times = Float64[]
bucketed_lengths = Int[]
total_start = time()
for i in 1:n_batches
    t0 = time()
    indices = sample_batch(sampler, batch_size)
    batch = batch_from_precomputed(proteins, indices, P)
    min_len = size(batch.xt_ca, 2)

    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])

    CUDA.synchronize()
    push!(bucketed_times, time() - t0)
    push!(bucketed_lengths, min_len)
end
bucketed_total = time() - total_start

@printf("  Avg time: %.1f ms/batch\n", mean(bucketed_times)*1000)
@printf("  Throughput: %.1f samples/sec\n", batch_size * n_batches / bucketed_total)
@printf("  Avg batch length: %.1f residues\n", mean(bucketed_lengths))

println("\n" * "=" ^ 60)
println("Comparison")
println("=" ^ 60)
speedup = (batch_size * n_batches / bucketed_total) / (batch_size * n_batches / random_total)
@printf("Bucketed is %.1fx faster\n", speedup)
@printf("Avg length: %.1f → %.1f residues (+%.0f%%)\n",
        mean(random_lengths), mean(bucketed_lengths),
        100 * (mean(bucketed_lengths) / mean(random_lengths) - 1))
