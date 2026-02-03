# Test encoder GPU training loop
# Features computed on CPU (outside gradient), transformer runs on GPU

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Flux
using Functors
using CUDA
using Statistics
using Random
using Optimisers

Random.seed!(42)

println("=" ^ 60)
println("Encoder GPU Training Test")
println("=" ^ 60)

# Check GPU
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: ", CUDA.device())
    println("Memory: ", round(CUDA.available_memory() / 1e9, digits=2), " GB available")
    use_gpu = true
else
    println("CUDA not functional, running on CPU")
    use_gpu = false
end

# Load some AFDB samples for training
println("\n=== Loading Training Data ===")
afdb_dir = expanduser("~/shared_data/afdb_laproteina/raw")
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)

# Load a few samples
n_train = 10
train_data = Dict{Symbol, Any}[]
for i in 1:min(n_train * 2, length(cif_files))
    filepath = joinpath(afdb_dir, cif_files[i])
    try
        data = load_pdb(filepath; chain_id="A")
        L = length(data[:aatype])
        if 50 <= L <= 150  # Filter by length
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
println("Lengths: ", [length(d[:aatype]) for d in train_data])

# Create encoder on CPU first
println("\n=== Creating Encoder ===")
encoder_cpu = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)

# Load pretrained weights
weights_path = joinpath(@__DIR__, "..", "weights", "encoder.npz")
if isfile(weights_path)
    println("Loading pretrained weights...")
    load_encoder_weights!(encoder_cpu, weights_path)
end

# Create a trainable model struct that holds GPU components
# Note: cond_factory has no features/projection in Python encoder - it returns zeros
# So we don't include cond_proj (it's Nothing)
struct GPUEncoderModel
    seq_proj::Any
    pair_proj::Any
    transition_c_1::Any
    transition_c_2::Any
    transformer_layers::Vector
    latent_proj::Any
    dim_cond::Int
end

Functors.@functor GPUEncoderModel (seq_proj, pair_proj, transition_c_1, transition_c_2, transformer_layers, latent_proj,)

# Create GPU model
println("Setting up GPU training...")

dim_cond = 128  # From encoder config

if use_gpu
    gpu_model = GPUEncoderModel(
        encoder_cpu.init_repr_factory.projection |> gpu,
        encoder_cpu.pair_rep_factory.projection |> gpu,
        encoder_cpu.transition_c_1 |> gpu,
        encoder_cpu.transition_c_2 |> gpu,
        [layer |> gpu for layer in encoder_cpu.transformer_layers],
        encoder_cpu.latent_proj |> gpu,
        dim_cond,
    )
else
    gpu_model = GPUEncoderModel(
        encoder_cpu.init_repr_factory.projection,
        encoder_cpu.pair_rep_factory.projection,
        encoder_cpu.transition_c_1,
        encoder_cpu.transition_c_2,
        encoder_cpu.transformer_layers,
        encoder_cpu.latent_proj,
        dim_cond,
    )
end

n_params = sum(length, Flux.trainables(gpu_model))
println("Total trainable parameters: ", n_params)

# Optimizer (using Optimisers.jl)
opt_state = Optimisers.setup(Adam(1e-4), gpu_model)

# VAE loss function (KL divergence)
function vae_kl_loss(mean_out, log_scale_out)
    # KL(q(z|x) || p(z)) where p(z) = N(0,1)
    kl_div = -0.5f0 * mean(1.0f0 .+ 2.0f0 .* log_scale_out .- mean_out.^2 .- exp.(2.0f0 .* log_scale_out))
    return kl_div
end

# Forward pass function (GPU part only - takes precomputed raw features)
# cond_zeros is precomputed zeros tensor passed in (not created inside gradient)
function forward_gpu(model::GPUEncoderModel, seq_raw, pair_raw, mask, cond_zeros)
    # Project features
    seq_repr = model.seq_proj(seq_raw)
    pair_repr = model.pair_proj(pair_raw)

    # Conditioning: use passed-in zeros (Python encoder has no cond features)
    cond = cond_zeros

    # Conditioning transitions (applied to zeros) - needs mask
    cond = cond .+ model.transition_c_1(cond, mask)
    cond = cond .+ model.transition_c_2(cond, mask)

    # Transformer layers
    for layer in model.transformer_layers
        seq_repr = layer(seq_repr, pair_repr, cond, mask)
    end

    # Output projection
    latent_params = model.latent_proj(seq_repr)  # [2*latent_dim, L, B]
    latent_dim = size(latent_params, 1) ÷ 2
    mean_out = latent_params[1:latent_dim, :, :]
    log_scale_out = latent_params[latent_dim+1:end, :, :]

    # Clamp log_scale for stability
    log_scale_out = clamp.(log_scale_out, -10.0f0, 2.0f0)

    return mean_out, log_scale_out
end

# Function to compute raw features on CPU (no gradients needed)
function compute_raw_features(batch, encoder_cpu)
    L, B = size(batch[:mask])

    # Extract raw features (before projection) on CPU
    seq_raw = cat([f(batch, L, B) for f in encoder_cpu.init_repr_factory.features]...; dims=1)
    pair_raw = cat([f(batch, L, B) for f in encoder_cpu.pair_rep_factory.features]...; dims=1)
    mask = Float32.(batch[:mask])

    # Note: cond_factory has no features in Python encoder (returns zeros)
    # So we skip computing cond_raw and create zeros in forward_gpu

    return seq_raw, pair_raw, mask
end

# Training loop
println("\n=== Training Loop ===")
n_epochs = 3
batch_size = 2

for epoch in 1:n_epochs
    epoch_losses = Float32[]

    # Shuffle data
    shuffled_data = shuffle(train_data)

    for batch_start in 1:batch_size:length(shuffled_data)
        batch_end = min(batch_start + batch_size - 1, length(shuffled_data))
        batch_data = shuffled_data[batch_start:batch_end]

        if length(batch_data) < 2
            continue
        end

        # Find common length
        min_len = minimum(length(d[:aatype]) for d in batch_data)

        # Truncate and batch
        truncated_data = Dict{Symbol, Any}[]
        for d in batch_data
            truncated = Dict{Symbol, Any}(
                :coords => d[:coords][:, :, 1:min_len],
                :atom_mask => d[:atom_mask][:, 1:min_len],
                :aatype => d[:aatype][1:min_len],
                :residue_mask => d[:residue_mask][1:min_len],
                :sequence => d[:sequence][1:min_len]
            )
            push!(truncated_data, truncated)
        end

        batched = batch_pdb_data(truncated_data)

        # Prepare encoder batch
        encoder_batch = Dict{Symbol, Any}(
            :coords => batched[:coords],
            :coord_mask => batched[:atom_mask],
            :residue_type => batched[:aatype],
            :mask => Float32.(batched[:mask]),
        )

        try
            # Compute raw features on CPU (no gradient)
            seq_raw, pair_raw, mask = compute_raw_features(encoder_batch, encoder_cpu)
            L, B = size(mask)

            # Create cond zeros outside gradient context
            cond_zeros = zeros(Float32, dim_cond, L, B)

            # Move to GPU if needed
            if use_gpu
                seq_raw = gpu(seq_raw)
                pair_raw = gpu(pair_raw)
                mask = gpu(mask)
                cond_zeros = gpu(cond_zeros)
            end

            # Compute loss and gradients (explicit gradient API)
            loss, grads = Flux.withgradient(gpu_model) do m
                mean_out, log_scale_out = forward_gpu(m, seq_raw, pair_raw, mask, cond_zeros)
                vae_kl_loss(mean_out, log_scale_out)
            end

            # Update parameters
            Optimisers.update!(opt_state, gpu_model, grads[1])

            push!(epoch_losses, Float32(cpu(loss)))

            if use_gpu && length(epoch_losses) % 2 == 0
                CUDA.reclaim()
            end
        catch e
            println("  Error in batch: ", typeof(e))
            if e isa ErrorException
                println("    ", e.msg[1:min(200, length(e.msg))])
            end
            rethrow(e)
        end
    end

    if !isempty(epoch_losses)
        avg_loss = mean(epoch_losses)
        println("Epoch $epoch: avg KL loss = $(round(avg_loss, digits=4)), $(length(epoch_losses)) batches")
    else
        println("Epoch $epoch: no successful batches")
    end
end

# Validation
println("\n=== Final Validation ===")
test_data = train_data[1]
batched = batch_pdb_data([test_data])
encoder_batch = Dict{Symbol, Any}(
    :coords => batched[:coords],
    :coord_mask => batched[:atom_mask],
    :residue_type => batched[:aatype],
    :mask => Float32.(batched[:mask]),
)

# Compute raw features
seq_raw, pair_raw, mask = compute_raw_features(encoder_batch, encoder_cpu)
L, B = size(mask)
cond_zeros = zeros(Float32, dim_cond, L, B)

if use_gpu
    seq_raw = gpu(seq_raw)
    pair_raw = gpu(pair_raw)
    mask = gpu(mask)
    cond_zeros = gpu(cond_zeros)
end

mean_out, log_scale_out = forward_gpu(gpu_model, seq_raw, pair_raw, mask, cond_zeros)

mean_out_cpu = cpu(mean_out)
log_scale_out_cpu = cpu(log_scale_out)

println("Output mean shape: ", size(mean_out_cpu))
println("Mean range: [$(round(minimum(mean_out_cpu), digits=3)), $(round(maximum(mean_out_cpu), digits=3))]")
println("Log_scale range: [$(round(minimum(log_scale_out_cpu), digits=3)), $(round(maximum(log_scale_out_cpu), digits=3))]")

println("\n" * "=" ^ 60)
println("GPU Training Test Complete!")
println("=" ^ 60)
