# Flow matching training using pretrained frozen VAE
# Features computed on CPU, model forward on GPU

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuProteina
using Flux
using Functors
using CUDA
using Statistics
using Random
using Optimisers

import Flowfusion
import Flowfusion: RDNFlow, sample_rdn_noise

Random.seed!(42)

println("=" ^ 60)
println("Flow Matching Training Test")
println("=" ^ 60)

# Check GPU
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: ", CUDA.device())
    println("Memory: ", round(CUDA.available_memory() / 1e9, digits=2), " GB available")
    use_gpu = true
    dev = gpu
else
    println("CUDA not functional, running on CPU")
    use_gpu = false
    dev = identity
end

# Load AFDB samples
println("\n=== Loading Training Data ===")
afdb_dir = expanduser("~/shared_data/afdb_laproteina/raw")
all_files = readdir(afdb_dir)
cif_files = filter(f -> endswith(f, ".cif"), all_files)

n_train = 20
train_data = Dict{Symbol, Any}[]
for i in 1:min(n_train * 3, length(cif_files))
    filepath = joinpath(afdb_dir, cif_files[i])
    try
        data = load_pdb(filepath; chain_id="A")
        L = length(data[:aatype])
        if 80 <= L <= 120
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

# Load frozen VAE encoder (stays on CPU)
println("\n=== Loading Frozen VAE Encoder ===")
encoder = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_encoder_weights!(encoder, joinpath(@__DIR__, "..", "weights", "encoder.npz"))
println("Encoder loaded (CPU)")

# Create ScoreNetwork (stays on CPU, we'll extract trainable parts)
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

# Create trainable GPU model (only projections and transformers)
struct GPUScoreNetworkModel
    cond_proj::Any                # Conditioning projection
    seq_proj::Any                 # Sequence feature projection
    pair_proj::Any                # Pair feature projection
    transition_c_1::Any           # Conditioning transition 1
    transition_c_2::Any           # Conditioning transition 2
    transformer_layers::Vector    # Transformer layers
    local_latents_proj::Any       # Output projection for latents
    ca_proj::Any                  # Output projection for CA
    dim_cond::Int
end

Functors.@functor GPUScoreNetworkModel (cond_proj, seq_proj, pair_proj, transition_c_1, transition_c_2, transformer_layers, local_latents_proj, ca_proj,)

println("Setting up GPU model...")
if use_gpu
    gpu_model = GPUScoreNetworkModel(
        score_net_cpu.cond_factory.projection |> gpu,
        score_net_cpu.init_repr_factory.projection |> gpu,
        score_net_cpu.pair_rep_builder.init_repr_factory.projection |> gpu,
        score_net_cpu.transition_c_1 |> gpu,
        score_net_cpu.transition_c_2 |> gpu,
        [layer |> gpu for layer in score_net_cpu.transformer_layers],
        score_net_cpu.local_latents_proj |> gpu,
        score_net_cpu.ca_proj |> gpu,
        256  # dim_cond
    )
else
    gpu_model = GPUScoreNetworkModel(
        score_net_cpu.cond_factory.projection,
        score_net_cpu.init_repr_factory.projection,
        score_net_cpu.pair_rep_builder.init_repr_factory.projection,
        score_net_cpu.transition_c_1,
        score_net_cpu.transition_c_2,
        score_net_cpu.transformer_layers,
        score_net_cpu.local_latents_proj,
        score_net_cpu.ca_proj,
        256
    )
end

n_params = sum(length, Flux.trainables(gpu_model))
println("Trainable parameters: ", n_params)

# RDNFlow processes
println("\n=== Setting up Flow Processes ===")
P_ca = RDNFlow(3; zero_com=true, sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(latent_dim; zero_com=false, sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

# Optimizer
opt_state = Optimisers.setup(Adam(1e-5), gpu_model)

# Compute raw features on CPU (outside gradient)
function compute_raw_features(score_net_cpu, xt_ca, xt_ll, t_vec, mask)
    L, B = size(mask)

    # Build batch for feature extraction
    batch = Dict{Symbol, Any}(
        :x_t => Dict(:bb_ca => xt_ca, :local_latents => xt_ll),
        :t => Dict(:bb_ca => t_vec, :local_latents => t_vec),
        :mask => mask
    )

    # Extract raw features (before projection)
    cond_raw = cat([f(batch, L, B) for f in score_net_cpu.cond_factory.features]...; dims=1)
    seq_raw = cat([f(batch, L, B) for f in score_net_cpu.init_repr_factory.features]...; dims=1)
    pair_raw = cat([f(batch, L, B) for f in score_net_cpu.pair_rep_builder.init_repr_factory.features]...; dims=1)

    return cond_raw, seq_raw, pair_raw
end

# GPU forward pass (differentiable)
function forward_gpu(model::GPUScoreNetworkModel, cond_raw, seq_raw, pair_raw, mask)
    L, B = size(mask)

    # Project features
    cond = model.cond_proj(cond_raw)
    cond = cond .+ model.transition_c_1(cond, mask)
    cond = cond .+ model.transition_c_2(cond, mask)

    seqs = model.seq_proj(seq_raw)
    mask_exp = reshape(mask, 1, L, B)
    seqs = seqs .* mask_exp

    pair_rep = model.pair_proj(pair_raw)

    # Transformer layers
    for layer in model.transformer_layers
        seqs = layer(seqs, pair_rep, cond, mask)
    end

    # Output projections
    local_latents_out = model.local_latents_proj(seqs) .* mask_exp
    ca_out = model.ca_proj(seqs) .* mask_exp

    return ca_out, local_latents_out
end

# Get X1 from encoder
function get_x1_from_encoder(encoder, data_batch)
    batched = batch_pdb_data(data_batch)

    encoder_batch = Dict{Symbol, Any}(
        :coords => batched[:coords],
        :coord_mask => batched[:atom_mask],
        :residue_type => batched[:aatype],
        :mask => Float32.(batched[:mask]),
    )

    enc_result = encoder(encoder_batch)

    # CA coordinates (centered)
    ca_coords = batched[:coords][:, CA_INDEX, :, :]
    ca_coords_centered = ca_coords .- mean(ca_coords, dims=2)

    # Local latents from encoder mean
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

    # Random times
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

# Flow matching loss (v-prediction)
function compute_loss(model, cond_raw, seq_raw, pair_raw, xt_ca, xt_ll, x1_ca, x1_ll, t_vec, mask)
    # Forward pass
    v_ca, v_ll = forward_gpu(model, cond_raw, seq_raw, pair_raw, mask)

    B = length(t_vec)
    t_bc = reshape(t_vec, 1, 1, B)

    # v-prediction: x1 = xt + (1-t) * v
    x1_pred_ca = xt_ca .+ (1f0 .- t_bc) .* v_ca
    x1_pred_ll = xt_ll .+ (1f0 .- t_bc) .* v_ll

    # Time-weighted MSE: 1/(1-t+eps)^2
    scale = 1f0 ./ (1f0 .- t_bc .+ 0.05f0).^2
    mask_3d = reshape(mask, 1, size(mask)...)

    loss_ca = mean(scale .* (x1_pred_ca .- x1_ca).^2 .* mask_3d)
    loss_ll = mean(scale .* (x1_pred_ll .- x1_ll).^2 .* mask_3d)

    return loss_ca + loss_ll
end

# Training loop
println("\n=== Training Loop ===")
n_epochs = 3
batch_size = 4

for epoch in 1:n_epochs
    epoch_losses = Float32[]
    shuffled_data = shuffle(train_data)

    for batch_start in 1:batch_size:length(shuffled_data)
        batch_end = min(batch_start + batch_size - 1, length(shuffled_data))
        batch_data = shuffled_data[batch_start:batch_end]

        if length(batch_data) < 2
            continue
        end

        try
            # Prepare batch (encoder runs here, CPU)
            batch = prepare_training_batch(batch_data, encoder, P)

            # Compute features on CPU (outside gradient)
            cond_raw, seq_raw, pair_raw = compute_raw_features(
                score_net_cpu, batch.xt_ca, batch.xt_ll, batch.t, batch.mask
            )

            # Move everything to GPU
            if use_gpu
                cond_raw = gpu(cond_raw)
                seq_raw = gpu(seq_raw)
                pair_raw = gpu(pair_raw)
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

            # Compute loss and gradients
            loss_val, grads = Flux.withgradient(gpu_model) do m
                compute_loss(m, cond_raw, seq_raw, pair_raw, xt_ca, xt_ll, x1_ca, x1_ll, t_vec, mask)
            end

            # Update
            Optimisers.update!(opt_state, gpu_model, grads[1])

            push!(epoch_losses, Float32(cpu(loss_val)))

            if use_gpu
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
        println("Epoch $epoch: loss=$(round(mean(epoch_losses), digits=4)), $(length(epoch_losses)) batches")
    else
        println("Epoch $epoch: no successful batches")
    end
end

println("\n" * "=" ^ 60)
println("Flow Matching Training Test Complete!")
println("=" ^ 60)
