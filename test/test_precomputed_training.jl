# Test precomputed training with sustained GPU utilization
# Creates smol_demo_training_data.jld2 and runs training loop

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

println("=" ^ 70)
println("Precomputed Training Demo")
println("=" ^ 70)

# Check GPU
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: ", CUDA.device())
    println("Memory: ", round(CUDA.available_memory() / 1e9, digits=2), " GB available")
else
    error("This test requires CUDA")
end

# ============================================================================
# Load training data (1000 random samples)
# ============================================================================
println("\n=== Loading Training Data ===")
afdb_dir = expanduser("~/shared_data/afdb_laproteina/raw")
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)

# Shuffle and take 1000
Random.shuffle!(cif_files)

n_samples = 1000
train_data = Dict{Symbol, Any}[]
failed = 0

println("Loading $n_samples samples...")
for (i, f) in enumerate(cif_files)
    if length(train_data) >= n_samples
        break
    end

    filepath = joinpath(afdb_dir, f)
    try
        data = load_pdb(filepath; chain_id="A")
        L = length(data[:aatype])
        if 30 <= L <= 256  # Reasonable length range
            push!(train_data, data)
        end
    catch
        failed += 1
        continue
    end

    if i % 500 == 0
        println("  Scanned $i files, loaded $(length(train_data)) samples")
    end
end

println("Loaded $(length(train_data)) samples (failed: $failed)")
lengths = [length(d[:aatype]) for d in train_data]
println("Length distribution: min=$(minimum(lengths)), max=$(maximum(lengths)), mean=$(round(mean(lengths), digits=1))")

# ============================================================================
# Load encoder
# ============================================================================
println("\n=== Loading Encoder ===")
latent_dim = 8

encoder_cpu = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=latent_dim, qk_ln=true, update_pair_repr=false
)
load_encoder_weights!(encoder_cpu, joinpath(@__DIR__, "..", "weights", "encoder.npz"))
encoder_gpu = deepcopy(encoder_cpu) |> gpu
println("Encoder loaded")

# ============================================================================
# Precompute encoder outputs
# ============================================================================
output_file = joinpath(@__DIR__, "smol_demo_training_data.jld2")

if isfile(output_file)
    println("\n=== Loading Existing Precomputed Data ===")
    println("File exists: $output_file")
    proteins = load_precomputed_shard(output_file)
    println("Loaded $(length(proteins)) precomputed proteins")
else
    println("\n=== Precomputing Encoder Outputs ===")
    println("Output: $output_file")

    t_precompute = @elapsed begin
        precompute_dataset_single(encoder_cpu, encoder_gpu, train_data;
                                   output_file=output_file, shuffle=true, verbose=true)
    end

    println("Precomputation time: $(round(t_precompute, digits=1)) s")
    println("Time per sample: $(round(t_precompute / length(train_data) * 1000, digits=1)) ms")

    proteins = load_precomputed_shard(output_file)
end

# Check file size
file_size = filesize(output_file) / 1e6
println("File size: $(round(file_size, digits=2)) MB")
println("Size per sample: $(round(file_size / length(proteins) * 1000, digits=1)) KB")

# ============================================================================
# Load score network
# ============================================================================
println("\n=== Loading ScoreNetwork ===")
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=latent_dim, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
load_score_network_weights!(score_net, joinpath(@__DIR__, "..", "weights", "score_network.npz"))
score_net_gpu = score_net |> gpu
println("ScoreNetwork loaded and moved to GPU")

# RDNFlow processes
P_ca = RDNFlow(3; zero_com=true, sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

# Optimizer
opt_state = Optimisers.setup(Adam(1e-5), score_net_gpu)

# ============================================================================
# Training loop with GPU utilization tracking
# ============================================================================
println("\n=== Training Loop ===")

batch_size = 8
n_epochs = 3
n_samples = length(proteins)
batches_per_epoch = n_samples ÷ batch_size

println("Batch size: $batch_size")
println("Samples: $n_samples")
println("Batches per epoch: $batches_per_epoch")
println("Total epochs: $n_epochs")

# Warmup
println("\nWarming up...")
for _ in 1:5
    indices = rand(1:length(proteins), batch_size)
    batch = batch_from_precomputed(proteins, indices, P)
    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])
end
CUDA.synchronize()
GC.gc()
CUDA.reclaim()
println("Warmup complete")

# Training
println("\nStarting training...")
all_losses = Float32[]
batch_times = Float64[]

total_start = time()

for epoch in 1:n_epochs
    epoch_start = time()
    epoch_losses = Float32[]

    # Shuffle indices for this epoch
    perm = randperm(n_samples)

    for batch_idx in 1:batches_per_epoch
        batch_start = time()

        # Get batch indices
        start_i = (batch_idx - 1) * batch_size + 1
        end_i = min(batch_idx * batch_size, n_samples)
        indices = perm[start_i:end_i]

        # Create batch and train
        batch = batch_from_precomputed(proteins, indices, P)

        loss, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])

        batch_time = time() - batch_start
        push!(batch_times, batch_time)
        push!(epoch_losses, Float32(cpu(loss)))
        push!(all_losses, Float32(cpu(loss)))
    end

    epoch_time = time() - epoch_start
    @printf("Epoch %d: loss=%.4f, time=%.1fs, %.1f ms/batch, %.1f samples/sec\n",
            epoch, mean(epoch_losses), epoch_time,
            epoch_time / batches_per_epoch * 1000,
            n_samples / epoch_time)
end

total_time = time() - total_start

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 70)
println("Training Summary")
println("=" ^ 70)

@printf("Total time:        %.1f s\n", total_time)
@printf("Total batches:     %d\n", length(all_losses))
@printf("Avg batch time:    %.1f ms\n", mean(batch_times) * 1000)
@printf("Throughput:        %.1f samples/sec\n", length(all_losses) * batch_size / total_time)

println("\nBatch time distribution:")
@printf("  min:    %.1f ms\n", minimum(batch_times) * 1000)
@printf("  median: %.1f ms\n", median(batch_times) * 1000)
@printf("  max:    %.1f ms\n", maximum(batch_times) * 1000)
@printf("  std:    %.1f ms\n", std(batch_times) * 1000)

println("\nLoss progression:")
n_check = min(50, length(all_losses) ÷ 2)
@printf("  First %d batches: %.4f\n", n_check, mean(all_losses[1:n_check]))
@printf("  Last %d batches:  %.4f\n", n_check, mean(all_losses[end-n_check+1:end]))

println("\nGPU Memory:")
println("  Available: ", round(CUDA.available_memory() / 1e9, digits=2), " GB")

println("\n" * "=" ^ 70)
println("Done!")
println("=" ^ 70)
