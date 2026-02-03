# Efficient frozen encoder for flow matching training
# Features computed on CPU, transformer runs on GPU

using Flux
using CUDA

# Access Zygote through Flux
const ZygoteEncoder = Flux.Zygote

"""
    EncoderRawFeatures

Raw features for encoder, computed on CPU.
"""
struct EncoderRawFeatures{A3<:AbstractArray, A4<:AbstractArray, A2<:AbstractArray}
    seq_raw::A3      # [seq_feat_dim, L, B]
    cond_raw::A3     # [cond_dim, L, B]
    pair_raw::A4     # [pair_feat_dim, L, L, B]
    mask::A2         # [L, B]
end

"""
    extract_encoder_features(encoder::EncoderTransformer, batch::Dict)

Extract raw features from encoder on CPU. These are the concatenated features
BEFORE projection.

# Arguments
- `encoder`: EncoderTransformer (CPU)
- `batch`: Dict with :coords, :coord_mask, :residue_type, :mask (CPU)

# Returns
EncoderRawFeatures with seq_raw, cond_raw, pair_raw, mask
"""
function extract_encoder_features(encoder::EncoderTransformer, batch::Dict)
    mask = batch[:mask]
    L, B = size(mask)

    # Extract raw features using FeatureFactory
    # These are the concatenated features before projection
    seq_features = [f(batch, L, B) for f in encoder.init_repr_factory.features]
    seq_raw = cat(seq_features..., dims=1)

    cond_features = [f(batch, L, B) for f in encoder.cond_factory.features]
    if isempty(cond_features)
        # No cond features - create zeros
        cond_raw = zeros(Float32, encoder.cond_factory.out_dim, L, B)
    else
        cond_raw = cat(cond_features..., dims=1)
    end

    pair_features = [f(batch, L, B) for f in encoder.pair_rep_factory.features]
    pair_raw = cat(pair_features..., dims=1)

    return EncoderRawFeatures(seq_raw, cond_raw, pair_raw, Float32.(mask))
end

"""
    encode_from_features_gpu(encoder::EncoderTransformer, features::EncoderRawFeatures)

Run encoder from pre-computed features. Features should already be on GPU.
Encoder model should be on GPU.

# Arguments
- `encoder`: EncoderTransformer on GPU
- `features`: EncoderRawFeatures on GPU

# Returns
Dict with :mean, :z_latent (deterministic)
"""
function encode_from_features_gpu(encoder::EncoderTransformer, features::EncoderRawFeatures)
    mask = features.mask
    L, B = size(mask)
    mask_exp = reshape(mask, 1, L, B)
    mask_pair = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)
    mask_pair_exp = reshape(mask_pair, 1, L, L, B)

    # Project sequence features
    seqs = encoder.init_repr_factory.projection(features.seq_raw)
    if encoder.init_repr_factory.use_ln && !isnothing(encoder.init_repr_factory.ln)
        seqs = encoder.init_repr_factory.ln(seqs)
    end
    seqs = seqs .* mask_exp

    # Project conditioning (or use zeros if empty features)
    if length(encoder.cond_factory.features) == 0
        # No features - just zeros projected
        cond = zeros(eltype(seqs), encoder.cond_factory.out_dim, L, B)
        if !(seqs isa Array)
            cond = Flux.gpu(cond)
        end
    else
        cond = encoder.cond_factory.projection(features.cond_raw)
        if encoder.cond_factory.use_ln && !isnothing(encoder.cond_factory.ln)
            cond = encoder.cond_factory.ln(cond)
        end
    end
    cond = cond .* mask_exp

    # Apply conditioning transitions
    cond = encoder.transition_c_1(cond, mask)
    cond = encoder.transition_c_2(cond, mask)

    # Project pair features
    pair_rep = encoder.pair_rep_factory.projection(features.pair_raw)
    if encoder.pair_rep_factory.use_ln && !isnothing(encoder.pair_rep_factory.ln)
        pair_rep = encoder.pair_rep_factory.ln(pair_rep)
    end
    pair_rep = pair_rep .* mask_pair_exp

    # Run transformer layers
    for i in 1:encoder.n_layers
        seqs = encoder.transformer_layers[i](seqs, pair_rep, cond, mask)

        if encoder.update_pair_repr && i < encoder.n_layers
            if !isnothing(encoder.pair_update_layers[i])
                pair_rep = encoder.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Project to latent space
    latent_out = encoder.latent_proj(seqs)
    latent_out = latent_out .* mask_exp

    # Split into mean and log_scale
    latent_dim = size(latent_out, 1) ÷ 2
    mean_out = latent_out[1:latent_dim, :, :]

    # Return mean (deterministic for training stability)
    return Dict(:mean => mean_out, :z_latent => mean_out)
end

"""
    encode_frozen_efficient(encoder_cpu::EncoderTransformer,
                            encoder_gpu::EncoderTransformer,
                            batch::Dict)

Efficient frozen encoder: features on CPU, transformer on GPU.

# Arguments
- `encoder_cpu`: EncoderTransformer on CPU (for feature extraction)
- `encoder_gpu`: EncoderTransformer on GPU (for transformer)
- `batch`: Dict with :coords, :coord_mask, :residue_type, :mask (on CPU)

# Returns
Dict with :mean, :z_latent on GPU
"""
function encode_frozen_efficient(encoder_cpu::EncoderTransformer,
                                  encoder_gpu::EncoderTransformer,
                                  batch::Dict)
    # Extract features on CPU
    features_cpu = extract_encoder_features(encoder_cpu, batch)

    # Move features to GPU
    features_gpu = EncoderRawFeatures(
        gpu(features_cpu.seq_raw),
        gpu(features_cpu.cond_raw),
        gpu(features_cpu.pair_raw),
        gpu(features_cpu.mask)
    )

    # Run encoder transformer on GPU (in @ignore since frozen)
    result = ZygoteEncoder.@ignore begin
        encode_from_features_gpu(encoder_gpu, features_gpu)
    end

    return result
end

"""
    prepare_encoder_batch_cpu(data_list)

Prepare encoder input batch from PDB data list on CPU.

# Arguments
- `data_list`: List of PDB data dicts from load_pdb()

# Returns
- encoder_batch: Dict with :coords, :coord_mask, :residue_type, :mask on CPU
- ca_coords_centered: [3, L, B] centered CA coordinates on CPU
- mask: [L, B] residue mask on CPU
"""
function prepare_encoder_batch_cpu(data_list)
    # Truncate to minimum length
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

    # Batch data (CPU)
    batched = batch_pdb_data(truncated_data)

    # Extract CA coordinates and center
    ca_coords = batched[:coords][:, CA_INDEX, :, :]  # [3, L, B]
    ca_coords_centered = ca_coords .- mean(ca_coords, dims=2)

    # Build encoder batch (CPU)
    encoder_batch = Dict{Symbol, Any}(
        :coords => Float32.(batched[:coords]),
        :coord_mask => Float32.(batched[:atom_mask]),
        :residue_type => batched[:aatype],
        :mask => Float32.(batched[:mask]),
    )

    return encoder_batch, Float32.(ca_coords_centered), Float32.(batched[:mask])
end

"""
    flow_matching_batch_gpu(encoder_cpu, encoder_gpu, data_list, P)

Prepare complete flow matching training batch.
Features computed on CPU, encoder transformer on GPU, interpolation on GPU.

# Arguments
- `encoder_cpu`: EncoderTransformer on CPU (for feature extraction)
- `encoder_gpu`: EncoderTransformer on GPU (for transformer)
- `data_list`: List of PDB data dicts
- `P`: Tuple of RDNFlow processes (P_ca, P_ll)

# Returns
Named tuple with all tensors on GPU:
- xt_ca, xt_ll: interpolated states
- x1_ca, x1_ll: target states from encoder
- t: time vector
- mask: residue mask
"""
function flow_matching_batch_gpu(encoder_cpu, encoder_gpu, data_list, P)
    # Prepare encoder batch on CPU
    encoder_batch, ca_coords_centered, mask = prepare_encoder_batch_cpu(data_list)

    L, B = size(mask)

    # Run frozen encoder: features on CPU, transformer on GPU
    enc_result = encode_frozen_efficient(encoder_cpu, encoder_gpu, encoder_batch)
    x1_ll = enc_result[:z_latent]  # [latent_dim, L, B] on GPU
    x1_ca = gpu(ca_coords_centered)  # [3, L, B] move to GPU

    # Sample noise on CPU then move to GPU (RDNFlow samples on CPU)
    x0_ca = gpu(Float32.(Flowfusion.sample_rdn_noise(P[1], L, B)))
    x0_ll = gpu(Float32.(Flowfusion.sample_rdn_noise(P[2], L, B)))

    # Mask on GPU
    mask_gpu = gpu(mask)

    # Random times on GPU
    t_vec = gpu(rand(Float32, B))
    t_bc = reshape(t_vec, 1, 1, B)

    # Linear interpolation on GPU: x_t = (1-t) * x0 + t * x1
    xt_ca = (1f0 .- t_bc) .* x0_ca .+ t_bc .* x1_ca
    xt_ll = (1f0 .- t_bc) .* x0_ll .+ t_bc .* x1_ll

    # Zero COM for CA on GPU
    if P[1].zero_com
        xt_ca = xt_ca .- mean(xt_ca, dims=2)
    end

    return (
        xt_ca=xt_ca, xt_ll=xt_ll,
        x1_ca=x1_ca, x1_ll=x1_ll,
        t=t_vec, mask=mask_gpu
    )
end

"""
    PrecomputedSample

Pre-computed encoder outputs for a single protein.
All tensors are on CPU for storage efficiency.
"""
struct PrecomputedSample
    ca_coords::Array{Float32, 2}    # [3, L] centered CA coordinates
    z_latent::Array{Float32, 2}     # [latent_dim, L] encoder mean output
    mask::Vector{Float32}           # [L] residue mask
end

"""
    precompute_encoder_outputs(encoder_cpu, encoder_gpu, data_list; verbose=true)

Pre-compute encoder outputs for a list of proteins.
Returns a vector of PrecomputedSample structs.

# Arguments
- `encoder_cpu`: EncoderTransformer on CPU (for feature extraction)
- `encoder_gpu`: EncoderTransformer on GPU (for transformer)
- `data_list`: List of PDB data dicts from load_pdb()
- `verbose`: Print progress

# Returns
Vector{PrecomputedSample} - one per input protein
"""
function precompute_encoder_outputs(encoder_cpu::EncoderTransformer,
                                     encoder_gpu::EncoderTransformer,
                                     data_list::Vector{Dict{Symbol, Any}};
                                     verbose::Bool=true)
    results = PrecomputedSample[]

    for (i, data) in enumerate(data_list)
        # Process single protein
        L = length(data[:aatype])

        # Build encoder batch for single sample
        encoder_batch = Dict{Symbol, Any}(
            :coords => Float32.(reshape(data[:coords], 3, 37, L, 1)),
            :coord_mask => Float32.(reshape(data[:atom_mask], 37, L, 1)),
            :residue_type => reshape(data[:aatype], L, 1),
            :mask => Float32.(reshape(data[:residue_mask], L, 1)),
        )

        # Extract CA coords and center
        ca_coords = encoder_batch[:coords][:, CA_INDEX, :, 1]  # [3, L]
        ca_coords_centered = ca_coords .- mean(ca_coords, dims=2)

        # Run encoder (features on CPU, transformer on GPU)
        features_cpu = extract_encoder_features(encoder_cpu, encoder_batch)
        features_gpu = EncoderRawFeatures(
            gpu(features_cpu.seq_raw),
            gpu(features_cpu.cond_raw),
            gpu(features_cpu.pair_raw),
            gpu(features_cpu.mask)
        )

        enc_result = encode_from_features_gpu(encoder_gpu, features_gpu)
        z_latent = cpu(enc_result[:z_latent][:, :, 1])  # [latent_dim, L]

        # Store result
        push!(results, PrecomputedSample(
            ca_coords_centered[:, :],
            z_latent,
            Float32.(data[:residue_mask])
        ))

        if verbose && i % 10 == 0
            println("  Precomputed $i / $(length(data_list))")
        end
    end

    if verbose
        println("  Precomputed $(length(results)) samples")
    end

    return results
end

"""
    flow_matching_batch_from_precomputed(precomputed::Vector{PrecomputedSample},
                                          indices::Vector{Int}, P)

Create flow matching batch from pre-computed encoder outputs.
Much faster than running encoder each batch.

# Arguments
- `precomputed`: Vector of PrecomputedSample
- `indices`: Which samples to include in batch
- `P`: Tuple of RDNFlow processes

# Returns
Named tuple with tensors on GPU
"""
function flow_matching_batch_from_precomputed(precomputed::Vector{PrecomputedSample},
                                               indices::Vector{Int}, P)
    # Get samples and find min length
    samples = [precomputed[i] for i in indices]
    min_len = minimum(size(s.ca_coords, 2) for s in samples)
    B = length(samples)

    # Stack and truncate to min_len
    x1_ca = zeros(Float32, 3, min_len, B)
    x1_ll = zeros(Float32, size(samples[1].z_latent, 1), min_len, B)
    mask = zeros(Float32, min_len, B)

    for (b, s) in enumerate(samples)
        x1_ca[:, :, b] = s.ca_coords[:, 1:min_len]
        x1_ll[:, :, b] = s.z_latent[:, 1:min_len]
        mask[:, b] = s.mask[1:min_len]
    end

    # Move to GPU
    x1_ca_gpu = gpu(x1_ca)
    x1_ll_gpu = gpu(x1_ll)
    mask_gpu = gpu(mask)

    L = min_len

    # Sample noise on CPU then move to GPU
    x0_ca = gpu(Float32.(Flowfusion.sample_rdn_noise(P[1], L, B)))
    x0_ll = gpu(Float32.(Flowfusion.sample_rdn_noise(P[2], L, B)))

    # Random times on GPU
    t_vec = gpu(rand(Float32, B))
    t_bc = reshape(t_vec, 1, 1, B)

    # Linear interpolation on GPU
    xt_ca = (1f0 .- t_bc) .* x0_ca .+ t_bc .* x1_ca_gpu
    xt_ll = (1f0 .- t_bc) .* x0_ll .+ t_bc .* x1_ll_gpu

    # Zero COM for CA
    if P[1].zero_com
        xt_ca = xt_ca .- mean(xt_ca, dims=2)
    end

    return (
        xt_ca=xt_ca, xt_ll=xt_ll,
        x1_ca=x1_ca_gpu, x1_ll=x1_ll_gpu,
        t=t_vec, mask=mask_gpu
    )
end

"""
    efficient_flow_loss_gpu(model, xt_ca, xt_ll, x1_ca, x1_ll, t_ca, t_ll, t_model, mask)

Compute flow matching loss entirely on GPU using efficient forward.

# Arguments
- xt_ca, xt_ll: interpolated states (at different positions due to schedules)
- x1_ca, x1_ll: target clean states
- t_ca, t_ll: actual interpolation times for each component (for loss computation)
- t_model: single time value passed to the model (the "progress" u)
- mask: residue mask
All tensors should be on GPU.
"""
function efficient_flow_loss_gpu(model, xt_ca, xt_ll, x1_ca, x1_ll, t_ca, t_ll, t_model, mask)
    L, B = size(mask)
    latent_dim = size(xt_ll, 1)

    # Create efficient batch - model sees single time value for both components
    eff_batch = EfficientScoreNetworkBatch(xt_ca, xt_ll, t_model, t_model, mask)

    # Forward pass with efficient method
    output = forward_efficient(model, eff_batch)
    v_ca = output[:bb_ca][:v]
    v_ll = output[:local_latents][:v]

    # v-prediction: x1_pred = xt + (1-t) * v
    # Use the ACTUAL interpolation times for loss computation
    t_ca_bc = reshape(t_ca, 1, 1, B)
    t_ll_bc = reshape(t_ll, 1, 1, B)
    x1_pred_ca = xt_ca .+ (1f0 .- t_ca_bc) .* v_ca
    x1_pred_ll = xt_ll .+ (1f0 .- t_ll_bc) .* v_ll

    # Squared error
    err_ca = (x1_pred_ca .- x1_ca).^2
    err_ll = (x1_pred_ll .- x1_ll).^2

    # Mask and normalize
    mask_3d = reshape(mask, 1, L, B)
    nres = reshape(sum(mask, dims=1), 1, 1, B)

    loss_ca_per_sample = sum(err_ca .* mask_3d, dims=(1,2)) ./ (nres .* 3f0)
    loss_ll_per_sample = sum(err_ll .* mask_3d, dims=(1,2)) ./ (nres .* Float32(latent_dim))

    # Time weighting: 1 / ((1-t)^2 + eps)
    # Using eps=0.1 for stability (caps max weight at 10)
    # Use respective t for each component's weighting
    eps = 0.1f0
    t_weight_ca = 1f0 ./ ((1f0 .- t_ca_bc).^2 .+ eps)
    t_weight_ll = 1f0 ./ ((1f0 .- t_ll_bc).^2 .+ eps)

    loss_ca = mean(loss_ca_per_sample .* t_weight_ca)
    loss_ll = mean(loss_ll_per_sample .* t_weight_ll)

    return loss_ca + loss_ll
end
