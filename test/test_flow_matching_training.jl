# Flow matching training using pretrained frozen VAE
# Features computed on CPU, model forward on GPU
# Loss matches Python la-proteina: MSE / nres * (1 / ((1-t)^2 + eps))
# Uses clean extract_raw_features + forward_from_raw_features API

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
println("Flow Matching Training (Frozen Encoder)")
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

# Load AFDB samples
println("\n=== Loading Training Data ===")
afdb_dir = expanduser("~/shared_data/afdb_laproteina/raw")
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)

n_train = 100  # More samples for proper training
train_data = Dict{Symbol, Any}[]
for i in 1:min(n_train * 3, length(cif_files))
    filepath = joinpath(afdb_dir, cif_files[i])
    try
        data = load_pdb(filepath; chain_id="A")
        L = length(data[:aatype])
        if 50 <= L <= 200  # Wider length range
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

# Load frozen VAE encoder (stays on CPU)
println("\n=== Loading Frozen VAE Encoder ===")
encoder = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_encoder_weights!(encoder, joinpath(@__DIR__, "..", "weights", "encoder.npz"))
println("Encoder loaded (CPU, frozen)")

# Create ScoreNetwork - keep CPU copy for feature extraction, GPU copy for training
println("\n=== Creating ScoreNetwork ===")
latent_dim = 8
score_net_cpu = ScoreNetwork(
    n_layers=14,
    token_dim=768,
    pair_dim=256,
    n_heads=12,
    dim_cond=256,
    latent_dim=latent_dim,
    output_param=:v,
    qk_ln=true,
    update_pair_repr=false
)
load_score_network_weights!(score_net_cpu, joinpath(@__DIR__, "..", "weights", "score_network.npz"))
println("ScoreNetwork loaded (CPU copy for feature extraction)")

# Create GPU model for training
if use_gpu
    score_net = score_net_cpu |> gpu
    println("ScoreNetwork moved to GPU for training")
else
    score_net = score_net_cpu
    println("ScoreNetwork on CPU")
end

n_params = sum(length, Flux.trainables(score_net))
println("Trainable parameters: ", n_params)

# RDNFlow processes
println("\n=== Setting up Flow Processes ===")
P_ca = RDNFlow(3; zero_com=true, sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

# Optimizer
opt_state = Optimisers.setup(Adam(1e-5), score_net)

# Get X1 from frozen encoder (called OUTSIDE gradient)
function get_x1_from_encoder(encoder, data_batch)
    batched = batch_pdb_data(data_batch)
    encoder_batch = Dict{Symbol, Any}(
        :coords => batched[:coords],
        :coord_mask => batched[:atom_mask],
        :residue_type => batched[:aatype],
        :mask => Float32.(batched[:mask]),
    )
    enc_result = encoder(encoder_batch)
    ca_coords = batched[:coords][:, CA_INDEX, :, :]
    ca_coords_centered = ca_coords .- mean(ca_coords, dims=2)
    z_latent = enc_result[:mean]
    return ca_coords_centered, z_latent, batched[:mask]
end

# Prepare batch: X1 from encoder, X0 from noise, Xt from interpolation
function prepare_training_batch(data_list, encoder, P)
    min_len = minimum(length(d[:aatype]) for d in data_list)
    truncated_data = Dict{Symbol, Any}[]
    for d in data_list
        push!(truncated_data, Dict{Symbol, Any}(
            :coords => d[:coords][:, :, 1:min_len],
            :atom_mask => d[:atom_mask][:, 1:min_len],
            :aatype => d[:aatype][1:min_len],
            :residue_mask => d[:residue_mask][1:min_len],
            :sequence => d[:sequence][1:min_len]
        ))
    end
    x1_ca, x1_ll, mask = get_x1_from_encoder(encoder, truncated_data)
    L, B = size(mask)

    # Sample noise X0
    x0_ca = sample_rdn_noise(P[1], L, B)
    x0_ll = sample_rdn_noise(P[2], L, B)

    # Random times (different t for each sample in batch)
    t_vec = rand(Float32, B)
    t_bc = reshape(t_vec, 1, 1, B)

    # Linear interpolation: x_t = (1-t) * x0 + t * x1
    x1_ca_f32 = Float32.(x1_ca)
    x1_ll_f32 = Float32.(x1_ll)
    xt_ca = (1f0 .- t_bc) .* x0_ca .+ t_bc .* x1_ca_f32
    xt_ll = (1f0 .- t_bc) .* x0_ll .+ t_bc .* x1_ll_f32

    # Zero COM for CA
    if P[1].zero_com
        xt_ca = xt_ca .- mean(xt_ca, dims=2)
    end

    return (
        xt_ca=xt_ca, xt_ll=xt_ll,
        x1_ca=x1_ca_f32, x1_ll=x1_ll_f32,
        t=t_vec, mask=Float32.(mask)
    )
end

# Flow matching loss matching Python la-proteina:
# loss = sum((x_1 - x_1_pred)^2) / nres * (1 / ((1-t)^2 + eps))
function compute_loss(model, raw_features, xt_ca, xt_ll, x1_ca, x1_ll, t_vec, mask)
    # Forward pass: project raw features and run transformer (differentiable)
    output = forward_from_raw_features(model, raw_features)
    v_ca = output[:bb_ca][:v]
    v_ll = output[:local_latents][:v]

    L, B = size(mask)
    t_bc = reshape(t_vec, 1, 1, B)

    # v-prediction: x1_pred = xt + (1-t) * v
    x1_pred_ca = xt_ca .+ (1f0 .- t_bc) .* v_ca
    x1_pred_ll = xt_ll .+ (1f0 .- t_bc) .* v_ll

    # Squared error
    err_ca = (x1_pred_ca .- x1_ca).^2
    err_ll = (x1_pred_ll .- x1_ll).^2

    # Mask and sum over features/positions, then normalize by nres
    mask_3d = reshape(mask, 1, L, B)
    nres = sum(mask, dims=1)  # [1, B]
    nres = reshape(nres, 1, 1, B)  # [1, 1, B] for broadcasting

    # Sum over feature and position dims, divide by nres
    loss_ca_per_sample = sum(err_ca .* mask_3d, dims=(1,2)) ./ (nres .* 3f0)  # [1, 1, B]
    loss_ll_per_sample = sum(err_ll .* mask_3d, dims=(1,2)) ./ (nres .* 8f0)  # [1, 1, B]

    # Time weighting: 1 / ((1-t)^2 + eps) - matches Python
    eps = 1f-5
    t_weight = 1f0 ./ ((1f0 .- t_bc).^2 .+ eps)  # [1, 1, B]

    # Weighted loss
    loss_ca = mean(loss_ca_per_sample .* t_weight)
    loss_ll = mean(loss_ll_per_sample .* t_weight)

    return loss_ca + loss_ll
end

# Helper to convert ScoreNetworkRawFeatures to GPU
function raw_features_to_gpu(features::ScoreNetworkRawFeatures)
    return ScoreNetworkRawFeatures(
        gpu(features.seq_raw),
        gpu(features.cond_raw),
        gpu(features.pair_raw),
        gpu(features.pair_cond_raw),
        gpu(features.mask)
    )
end

# Training loop with proper logging
println("\n=== Training ===")
n_epochs = 10
batch_size = 4
all_losses = Float32[]

println("Training for $n_epochs epochs, batch_size=$batch_size")
println("Will compare first 100 vs last 100 batch losses\n")

epoch_start_time = time()

for epoch in 1:n_epochs
    shuffled_data = shuffle(train_data)

    for batch_start in 1:batch_size:length(shuffled_data)
        batch_end = min(batch_start + batch_size - 1, length(shuffled_data))
        batch_data = shuffled_data[batch_start:batch_end]

        if length(batch_data) < 2
            continue
        end

        try
            # Prepare batch (encoder runs here on CPU, OUTSIDE gradient)
            batch = prepare_training_batch(batch_data, encoder, P)

            # Create score network batch format
            score_batch = Dict{Symbol, Any}(
                :x_t => Dict(:bb_ca => batch.xt_ca, :local_latents => batch.xt_ll),
                :t => Dict(:bb_ca => batch.t, :local_latents => batch.t),
                :mask => batch.mask
            )

            # Extract raw features on CPU (outside gradient, uses CPU model)
            raw_features = extract_raw_features(score_net_cpu, score_batch)

            # Move features and batch data to GPU
            if use_gpu
                raw_features = raw_features_to_gpu(raw_features)
                xt_ca = gpu(batch.xt_ca)
                xt_ll = gpu(batch.xt_ll)
                x1_ca = gpu(batch.x1_ca)
                x1_ll = gpu(batch.x1_ll)
                t_vec = gpu(batch.t)
                mask = gpu(batch.mask)
            else
                xt_ca, xt_ll = batch.xt_ca, batch.xt_ll
                x1_ca, x1_ll = batch.x1_ca, batch.x1_ll
                t_vec, mask = batch.t, batch.mask
            end

            # Compute loss and gradients (score_net is differentiated)
            loss_val, grads = Flux.withgradient(score_net) do m
                compute_loss(m, raw_features, xt_ca, xt_ll, x1_ca, x1_ll, t_vec, mask)
            end

            # Update
            Optimisers.update!(opt_state, score_net, grads[1])

            push!(all_losses, Float32(cpu(loss_val)))

            if use_gpu && length(all_losses) % 10 == 0
                CUDA.reclaim()
            end
        catch e
            println("  Error in batch: ", typeof(e))
            rethrow(e)
        end
    end

    # Epoch summary
    epoch_losses = all_losses[max(1, end-length(train_data)÷batch_size+1):end]
    elapsed = time() - epoch_start_time
    @printf("Epoch %2d: loss=%.4f (last %d batches), elapsed=%.1fs\n",
            epoch, mean(epoch_losses), length(epoch_losses), elapsed)
end

# Final analysis: compare first 100 vs last 100 batches
println("\n=== Loss Analysis ===")
n_compare = min(100, length(all_losses) ÷ 2)
first_losses = all_losses[1:n_compare]
last_losses = all_losses[end-n_compare+1:end]

@printf("First %d batches: mean=%.4f, std=%.4f\n", n_compare, mean(first_losses), std(first_losses))
@printf("Last  %d batches: mean=%.4f, std=%.4f\n", n_compare, mean(last_losses), std(last_losses))
@printf("Change: %.4f (%.1f%%)\n", mean(last_losses) - mean(first_losses),
        100 * (mean(last_losses) - mean(first_losses)) / mean(first_losses))

println("\n" * "=" ^ 60)
println("Training Complete!")
println("Total batches: ", length(all_losses))
println("=" ^ 60)
