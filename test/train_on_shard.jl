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

println("=" ^ 70)
println("Training on Precomputed Shard (with z sampling)")
println("=" ^ 70)

# Check GPU
println("\n=== GPU Status ===")
if !CUDA.functional()
    error("This test requires CUDA")
end
println("CUDA is functional!")
println("Device: ", CUDA.device())
println("Memory: ", round(CUDA.available_memory() / 1e9, digits=2), " GB available")

# ============================================================================
# Load shard with timing
# ============================================================================
shard_path = expanduser("~/shared_data/afdb_laproteina/precomputed_shards/train_shard_01.jld2")
println("\n=== Loading Shard ===")
println("Path: $shard_path")
println("Size: $(round(filesize(shard_path) / 1e6, digits=1)) MB")

t_load = @elapsed begin
    proteins = load_precomputed_shard(shard_path)
end
println("Loaded $(length(proteins)) proteins in $(round(t_load, digits=2)) seconds")
println("Load speed: $(round(filesize(shard_path) / t_load / 1e6, digits=1)) MB/s")

# Check struct has new fields
sample = proteins[1]
println("\nSample fields: ", fieldnames(typeof(sample)))
println("  ca_coords: $(size(sample.ca_coords))")
println("  z_mean: $(size(sample.z_mean))")
println("  z_log_scale: $(size(sample.z_log_scale))")
println("  mask: $(size(sample.mask))")

# ============================================================================
# Load ScoreNetwork
# ============================================================================
println("\n=== Loading ScoreNetwork ===")
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

# RDNFlow with schedule parameters matching inference
P_ca = RDNFlow(3; zero_com=true, schedule=:log, schedule_param=2.0f0,
               sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, schedule=:power, schedule_param=2.0f0,
               sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

opt_state = Optimisers.setup(Adam(1e-5), score_net_gpu)

# ============================================================================
# Training
# ============================================================================
batch_size = 4
n_batches = 1100  # 50 first + 1000 middle + 50 last

println("\n=== Training ===")
println("Batch size: $batch_size")
println("N batches: $n_batches")
println("N proteins: $(length(proteins))")

# Warmup
println("\nWarming up...")
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

# Training loop
losses = Float32[]
batch_times = Float64[]

println("\nTraining...")
t_train_start = time()

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
    push!(batch_times, time() - t0)

    if i % 100 == 0
        CUDA.synchronize()
        @printf("Batch %d: loss=%.4f, time=%.0fms\n", i, losses[end], batch_times[end]*1000)
        GC.gc()
    end
end

t_train = time() - t_train_start
CUDA.synchronize()

# ============================================================================
# Summary
# ============================================================================
first_50 = losses[1:50]
last_50 = losses[end-49:end]

println("\n" * "=" ^ 70)
println("RESULTS")
println("=" ^ 70)

println("\nShard loading:")
@printf("  Time: %.2f seconds\n", t_load)
@printf("  Speed: %.1f MB/s\n", filesize(shard_path) / t_load / 1e6)

println("\nTraining:")
@printf("  Total time: %.1f seconds\n", t_train)
@printf("  Avg batch time: %.1f ms\n", mean(batch_times) * 1000)
@printf("  Throughput: %.1f samples/sec\n", n_batches * batch_size / t_train)

println("\nLoss comparison:")
@printf("  First 50 batches: mean=%.4f, std=%.4f\n", mean(first_50), std(first_50))
@printf("  Last 50 batches:  mean=%.4f, std=%.4f\n", mean(last_50), std(last_50))
@printf("  Change: %.4f (%.2f%%)\n", mean(last_50) - mean(first_50),
        100 * (mean(last_50) - mean(first_50)) / mean(first_50))

println("\nLoss progression (every 100 batches):")
for i in 100:100:n_batches
    window = losses[max(1, i-49):i]
    @printf("  Batch %4d: avg loss = %.4f\n", i, mean(window))
end

println("\n" * "=" ^ 70)
println("Done!")
