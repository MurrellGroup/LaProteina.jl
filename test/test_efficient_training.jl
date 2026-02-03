# Benchmark efficient GPU-native flow matching training
# Compares: old (encoder on CPU) vs new (encoder on GPU, frozen)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Flux
using CUDA
using Statistics
using Random
using Optimisers
using Printf

import Flowfusion
import Flowfusion: RDNFlow, sample_rdn_noise

Random.seed!(42)

println("=" ^ 60)
println("Efficient GPU Training Benchmark")
println("=" ^ 60)

# Check GPU
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: ", CUDA.device())
    println("Memory: ", round(CUDA.available_memory() / 1e9, digits=2), " GB available")
else
    error("This test requires CUDA")
end

# Load AFDB samples
println("\n=== Loading Training Data ===")
afdb_dir = expanduser("~/shared_data/afdb_laproteina/raw")
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)

n_train = 50
train_data = Dict{Symbol, Any}[]
for i in 1:min(n_train * 3, length(cif_files))
    filepath = joinpath(afdb_dir, cif_files[i])
    try
        data = load_pdb(filepath; chain_id="A")
        L = length(data[:aatype])
        if 50 <= L <= 150
            push!(train_data, data)
            if length(train_data) >= n_train
                break
            end
        end
    catch
        continue
    end
end
println("Loaded $(length(train_data)) training samples")
println("Length range: $(minimum(length(d[:aatype]) for d in train_data)) - $(maximum(length(d[:aatype]) for d in train_data))")

# Load models
println("\n=== Loading Models ===")
latent_dim = 8

# Encoder - need both CPU (for features) and GPU (for transformer)
encoder_cpu = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=latent_dim, qk_ln=true, update_pair_repr=false
)
load_encoder_weights!(encoder_cpu, joinpath(@__DIR__, "..", "weights", "encoder.npz"))
encoder_gpu = deepcopy(encoder_cpu) |> gpu
println("Encoder loaded (CPU for features, GPU for transformer)")

# Score network - for training
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
# Benchmark: Efficient GPU Training (encoder transformer on GPU)
# ============================================================================
println("\n=== Benchmarking Efficient GPU Training ===")
batch_size = 4
n_batches = 20

# Extended warmup - run 10 iterations to ensure everything is compiled
println("Warming up (10 iterations)...")
for _ in 1:10
    idx = rand(1:length(train_data)-batch_size)
    batch_data = train_data[idx:idx+batch_size-1]
    batch = flow_matching_batch_gpu(encoder_cpu, encoder_gpu, batch_data, P)
    _, grads = Flux.withgradient(score_net_gpu) do m
        efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    Optimisers.update!(opt_state, score_net_gpu, grads[1])
end
CUDA.synchronize()
GC.gc()
CUDA.reclaim()
println("Warmup complete.")

# Benchmark
println("Running $n_batches training steps...")
losses_efficient = Float32[]
t_efficient = @elapsed begin
    for i in 1:n_batches
        # Random batch
        idx = rand(1:length(train_data)-batch_size)
        local batch_data = train_data[idx:idx+batch_size-1]

        # Prepare batch (features on CPU, encoder transformer on GPU)
        local batch = flow_matching_batch_gpu(encoder_cpu, encoder_gpu, batch_data, P)

        # Training step
        local loss, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])

        push!(losses_efficient, Float32(cpu(loss)))
    end
    CUDA.synchronize()
end

@printf("Efficient GPU Training:\n")
@printf("  Total time: %.2f s\n", t_efficient)
@printf("  Per batch:  %.1f ms\n", t_efficient / n_batches * 1000)
@printf("  Mean loss:  %.4f\n", mean(losses_efficient))

# ============================================================================
# Pre-compute encoder outputs
# ============================================================================
println("\n=== Pre-computing Encoder Outputs ===")
t_precompute = @elapsed begin
    precomputed = precompute_encoder_outputs(encoder_cpu, encoder_gpu, train_data; verbose=true)
end
@printf("Pre-computation time: %.1f s (%.1f ms/sample)\n", t_precompute, t_precompute / length(train_data) * 1000)

# ============================================================================
# Benchmark with pre-computed features
# ============================================================================
println("\n=== Benchmarking with Pre-computed Features ===")

# Warmup
indices = rand(1:length(precomputed), batch_size)
batch = flow_matching_batch_from_precomputed(precomputed, indices, P)
_, _ = Flux.withgradient(score_net_gpu) do m
    efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
end
CUDA.synchronize()

# Benchmark
println("Running $n_batches training steps with pre-computed features...")
losses_precomputed = Float32[]
t_precomputed = @elapsed begin
    for i in 1:n_batches
        local indices = rand(1:length(precomputed), batch_size)
        local batch = flow_matching_batch_from_precomputed(precomputed, indices, P)

        local loss, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])

        push!(losses_precomputed, Float32(cpu(loss)))
    end
    CUDA.synchronize()
end

@printf("Pre-computed Training:\n")
@printf("  Total time: %.2f s\n", t_precomputed)
@printf("  Per batch:  %.1f ms\n", t_precomputed / n_batches * 1000)
@printf("  Mean loss:  %.4f\n", mean(losses_precomputed))
@printf("  Speedup vs on-the-fly: %.2fx\n", t_efficient / t_precomputed)

# Breakdown for pre-computed path
println("\n--- Pre-computed path breakdown ---")
indices = rand(1:length(precomputed), batch_size)
t_batch_prep = @elapsed begin
    for _ in 1:n_batches
        batch = flow_matching_batch_from_precomputed(precomputed, indices, P)
    end
    CUDA.synchronize()
end
@printf("Batch prep from precomputed: %.1f ms/batch\n", t_batch_prep / n_batches * 1000)

batch = flow_matching_batch_from_precomputed(precomputed, indices, P)
t_train_step = @elapsed begin
    for _ in 1:n_batches
        _, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])
    end
    CUDA.synchronize()
end
@printf("Score forward+backward+update: %.1f ms/batch\n", t_train_step / n_batches * 1000)
@printf("Total estimated: %.1f ms/batch\n", (t_batch_prep + t_train_step) / n_batches * 1000)

# ============================================================================
# Detailed timing breakdown
# ============================================================================
println("\n=== Detailed Timing Breakdown ===")

batch_data = train_data[1:batch_size]

# Warmup all components
batch = flow_matching_batch_gpu(encoder_cpu, encoder_gpu, batch_data, P)
_ = efficient_flow_loss_gpu(score_net_gpu, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
_, _ = Flux.withgradient(score_net_gpu) do m
    efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
end
CUDA.synchronize()
GC.gc()
CUDA.reclaim()

# Time feature extraction on CPU
t_features = @elapsed begin
    for _ in 1:n_batches
        encoder_batch, _, _ = prepare_encoder_batch_cpu(batch_data)
        features = extract_encoder_features(encoder_cpu, encoder_batch)
    end
end
@printf("1. Feature extraction (CPU):    %.1f ms/batch\n", t_features / n_batches * 1000)

# Time encoder transformer on GPU
encoder_batch, ca_cpu, mask_cpu = prepare_encoder_batch_cpu(batch_data)
features_cpu = extract_encoder_features(encoder_cpu, encoder_batch)
features_gpu = EncoderRawFeatures(gpu(features_cpu.seq_raw), gpu(features_cpu.cond_raw),
                                   gpu(features_cpu.pair_raw), gpu(features_cpu.mask))
CUDA.synchronize()

t_enc_transformer = @elapsed begin
    for _ in 1:n_batches
        _ = encode_from_features_gpu(encoder_gpu, features_gpu)
    end
    CUDA.synchronize()
end
@printf("2. Encoder transformer (GPU):   %.1f ms/batch\n", t_enc_transformer / n_batches * 1000)

# Time CPU->GPU transfer
t_transfer = @elapsed begin
    for _ in 1:n_batches
        _ = gpu(features_cpu.seq_raw)
        _ = gpu(features_cpu.cond_raw)
        _ = gpu(features_cpu.pair_raw)
        _ = gpu(features_cpu.mask)
        _ = gpu(ca_cpu)
        _ = gpu(mask_cpu)
    end
    CUDA.synchronize()
end
@printf("3. CPU->GPU transfer:           %.1f ms/batch\n", t_transfer / n_batches * 1000)

# Time noise sampling + interpolation
batch = flow_matching_batch_gpu(encoder_cpu, encoder_gpu, batch_data, P)  # Get x1 on GPU
CUDA.synchronize()
t_interp = @elapsed begin
    for _ in 1:n_batches
        L, B = size(batch.mask)
        x0_ca = gpu(Float32.(Flowfusion.sample_rdn_noise(P[1], L, B)))
        x0_ll = gpu(Float32.(Flowfusion.sample_rdn_noise(P[2], L, B)))
        t_vec = gpu(rand(Float32, B))
        t_bc = reshape(t_vec, 1, 1, B)
        xt_ca = (1f0 .- t_bc) .* x0_ca .+ t_bc .* batch.x1_ca
        xt_ll = (1f0 .- t_bc) .* x0_ll .+ t_bc .* batch.x1_ll
        xt_ca = xt_ca .- mean(xt_ca, dims=2)
    end
    CUDA.synchronize()
end
@printf("4. Noise + interpolation:       %.1f ms/batch\n", t_interp / n_batches * 1000)

# Time score network forward only
t_forward = @elapsed begin
    for _ in 1:n_batches
        _ = efficient_flow_loss_gpu(score_net_gpu, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
    end
    CUDA.synchronize()
end
@printf("5. Score forward only:          %.1f ms/batch\n", t_forward / n_batches * 1000)

# Time forward + backward
t_grad = @elapsed begin
    for _ in 1:n_batches
        _, _ = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
    end
    CUDA.synchronize()
end
@printf("6. Score forward + backward:    %.1f ms/batch\n", t_grad / n_batches * 1000)

# Time optimizer update
grads = Flux.gradient(score_net_gpu) do m
    efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
end
CUDA.synchronize()
t_update = @elapsed begin
    for _ in 1:n_batches
        Optimisers.update!(opt_state, score_net_gpu, grads[1])
    end
    CUDA.synchronize()
end
@printf("7. Optimizer update:            %.1f ms/batch\n", t_update / n_batches * 1000)

# Time full flow_matching_batch_gpu
t_full_batch = @elapsed begin
    for _ in 1:n_batches
        batch = flow_matching_batch_gpu(encoder_cpu, encoder_gpu, batch_data, P)
    end
    CUDA.synchronize()
end
@printf("8. Full batch prep function:    %.1f ms/batch\n", t_full_batch / n_batches * 1000)

# Time full training step (batch + grad + update) - SAME batch each time
t_full_step = @elapsed begin
    for _ in 1:n_batches
        batch = flow_matching_batch_gpu(encoder_cpu, encoder_gpu, batch_data, P)
        _, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])
    end
    CUDA.synchronize()
end
@printf("9. Full step (same batch):      %.1f ms/batch\n", t_full_step / n_batches * 1000)

# Time full training step with RANDOM batches (like original loop)
t_full_step_random = @elapsed begin
    for _ in 1:n_batches
        idx = rand(1:length(train_data)-batch_size)
        random_batch_data = train_data[idx:idx+batch_size-1]
        batch = flow_matching_batch_gpu(encoder_cpu, encoder_gpu, random_batch_data, P)
        _, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll, batch.x1_ca, batch.x1_ll, batch.t, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])
    end
    CUDA.synchronize()
end
@printf("10. Full step (random batches): %.1f ms/batch\n", t_full_step_random / n_batches * 1000)

# Check if lengths vary
println("\nBatch length distribution:")
for i in 1:5
    idx = rand(1:length(train_data)-batch_size)
    sample_batch = train_data[idx:idx+batch_size-1]
    min_len = minimum(length(d[:aatype]) for d in sample_batch)
    println("  Random batch $i: min_len = $min_len")
end

# Sum of components
total_est = (t_features + t_enc_transformer + t_transfer + t_interp + t_grad + t_update) / n_batches * 1000
@printf("\nEstimated total (components):   %.1f ms/batch\n", total_est)
@printf("Full batch prep measured:       %.1f ms/batch\n", t_full_batch / n_batches * 1000)
@printf("Full step measured:             %.1f ms/batch\n", t_full_step / n_batches * 1000)
@printf("Original loop measured:         %.1f ms/batch\n", t_efficient / n_batches * 1000)

# ============================================================================
# Memory usage
# ============================================================================
println("\n=== GPU Memory ===")
CUDA.reclaim()
GC.gc()
println("After cleanup: ", round(CUDA.available_memory() / 1e9, digits=2), " GB available")

println("\n" * "=" ^ 60)
println("Benchmark Complete!")
println("=" ^ 60)
