# Precompute VAE encoder outputs for training dataset
# Saves minimal representation: CA coords + encoder mean/log_scale per position
# Uses NamedTuples for portable serialization (no package-specific types)

using JLD2
using Random
using Statistics

# Type alias for precomputed protein data (NamedTuple for portable serialization)
const PrecomputedProteinNT = NamedTuple{
    (:ca_coords, :z_mean, :z_log_scale, :mask),
    Tuple{Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}}
}

"""
    precompute_single_protein(encoder_cpu, encoder_gpu, data::Dict)

Precompute encoder output for a single protein.
Returns a NamedTuple with fields:
- ca_coords: [3, L] centered CA coordinates in nm
- z_mean: [latent_dim, L] encoder mean output
- z_log_scale: [latent_dim, L] encoder log_scale output
- mask: [L] residue mask (1.0 for valid, 0.0 for padding)

At training time, z is sampled as: z = z_mean + randn() * exp(z_log_scale)

Returns nothing if encoding fails.
"""
function precompute_single_protein(encoder_cpu::EncoderTransformer,
                                    encoder_gpu::EncoderTransformer,
                                    data::Dict)
    try
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
        z_mean = cpu(enc_result[:mean][:, :, 1])  # [latent_dim, L]
        z_log_scale = cpu(enc_result[:log_scale][:, :, 1])  # [latent_dim, L]

        # Return as NamedTuple (portable serialization)
        return (
            ca_coords = Float32.(ca_coords_centered),
            z_mean = Float32.(z_mean),
            z_log_scale = Float32.(z_log_scale),
            mask = Float32.(data[:residue_mask])
        )
    catch e
        @warn "Failed to encode protein: $e"
        return nothing
    end
end

"""
    precompute_dataset_sharded(encoder_cpu, encoder_gpu, data_list;
                                output_dir, n_shards=10, prefix="train_shard",
                                shuffle=true, verbose=true)

Precompute encoder outputs for entire dataset and save as sharded JLD2 files.

# Arguments
- `encoder_cpu`: EncoderTransformer on CPU (for feature extraction)
- `encoder_gpu`: EncoderTransformer on GPU (for transformer)
- `data_list`: Vector of PDB data dicts
- `output_dir`: Directory to save JLD2 files
- `n_shards`: Number of output files (default 10)
- `prefix`: Filename prefix (default "train_shard")
- `shuffle`: Randomize order before sharding (default true)
- `verbose`: Print progress (default true)

# Returns
Vector of output file paths
"""
function precompute_dataset_sharded(encoder_cpu::EncoderTransformer,
                                     encoder_gpu::EncoderTransformer,
                                     data_list::Vector;
                                     output_dir::String,
                                     n_shards::Int=10,
                                     prefix::String="train_shard",
                                     shuffle::Bool=true,
                                     verbose::Bool=true)
    mkpath(output_dir)

    # Shuffle if requested
    indices = collect(1:length(data_list))
    if shuffle
        Random.shuffle!(indices)
    end

    # Compute shard assignments
    n_total = length(data_list)
    samples_per_shard = ceil(Int, n_total / n_shards)

    output_files = String[]

    for shard_idx in 1:n_shards
        start_idx = (shard_idx - 1) * samples_per_shard + 1
        end_idx = min(shard_idx * samples_per_shard, n_total)

        if start_idx > n_total
            break
        end

        shard_indices = indices[start_idx:end_idx]
        shard_proteins = PrecomputedProteinNT[]

        if verbose
            println("Processing shard $shard_idx / $n_shards ($(length(shard_indices)) samples)...")
        end

        for (i, data_idx) in enumerate(shard_indices)
            protein = precompute_single_protein(encoder_cpu, encoder_gpu, data_list[data_idx])
            if !isnothing(protein)
                push!(shard_proteins, protein)
            end

            if verbose && i % 100 == 0
                println("  Processed $i / $(length(shard_indices))")
            end
        end

        # Save shard
        output_file = joinpath(output_dir, "$(prefix)_$(lpad(shard_idx, 2, '0')).jld2")
        jldsave(output_file; proteins=shard_proteins)
        push!(output_files, output_file)

        if verbose
            println("  Saved $(length(shard_proteins)) proteins to $output_file")
        end

        # Clean up GPU memory
        GC.gc()
        CUDA.reclaim()
    end

    return output_files
end

"""
    precompute_dataset_single(encoder_cpu, encoder_gpu, data_list;
                               output_file, shuffle=true, verbose=true)

Precompute encoder outputs and save to a single JLD2 file.
Useful for small datasets or demos.
"""
function precompute_dataset_single(encoder_cpu::EncoderTransformer,
                                    encoder_gpu::EncoderTransformer,
                                    data_list::Vector;
                                    output_file::String,
                                    shuffle::Bool=true,
                                    verbose::Bool=true)
    # Shuffle if requested
    indices = collect(1:length(data_list))
    if shuffle
        Random.shuffle!(indices)
    end

    proteins = PrecomputedProteinNT[]

    for (i, data_idx) in enumerate(indices)
        protein = precompute_single_protein(encoder_cpu, encoder_gpu, data_list[data_idx])
        if !isnothing(protein)
            push!(proteins, protein)
        end

        if verbose && i % 100 == 0
            println("  Processed $i / $(length(data_list))")
        end
    end

    # Save
    mkpath(dirname(output_file))
    jldsave(output_file; proteins=proteins)

    if verbose
        println("Saved $(length(proteins)) proteins to $output_file")
    end

    return output_file
end

"""
    load_precomputed_shard(filepath::String)

Load a precomputed shard from JLD2 file.
Returns Vector of NamedTuples with fields: ca_coords, z_mean, z_log_scale, mask
"""
function load_precomputed_shard(filepath::String)
    return load(filepath, "proteins")
end

"""
    LengthBucketedSampler

Sampler that groups proteins by similar lengths to minimize padding/truncation waste.
"""
struct LengthBucketedSampler
    buckets::Vector{Vector{Int}}     # indices grouped by length bucket
    bucket_bounds::Vector{Int}       # upper bounds for each bucket
    current_bucket::Ref{Int}         # current bucket index
    current_pos::Ref{Int}            # position within current bucket
    shuffled_buckets::Vector{Vector{Int}}  # shuffled copies for iteration
end

function LengthBucketedSampler(proteins::Vector; bucket_size::Int=32)
    lengths = [length(p.mask) for p in proteins]
    min_len, max_len = extrema(lengths)

    # Create bucket bounds
    bucket_bounds = collect(min_len:bucket_size:max_len+bucket_size)

    # Assign indices to buckets
    buckets = [Int[] for _ in 1:length(bucket_bounds)-1]
    for (i, len) in enumerate(lengths)
        bucket_idx = findfirst(b -> len < b, bucket_bounds) - 1
        bucket_idx = clamp(bucket_idx, 1, length(buckets))
        push!(buckets[bucket_idx], i)
    end

    # Remove empty buckets
    non_empty = findall(!isempty, buckets)
    buckets = buckets[non_empty]
    bucket_bounds = bucket_bounds[non_empty .+ 1]

    # Initialize shuffled copies
    shuffled_buckets = [shuffle(copy(b)) for b in buckets]

    return LengthBucketedSampler(buckets, bucket_bounds, Ref(1), Ref(1), shuffled_buckets)
end

"""
    sample_batch(sampler::LengthBucketedSampler, batch_size::Int)

Sample a batch of indices from proteins with similar lengths.
Returns vector of indices.
"""
function sample_batch(sampler::LengthBucketedSampler, batch_size::Int)
    # Pick a random non-empty bucket weighted by size
    weights = Float64[length(b) for b in sampler.buckets]
    weights ./= sum(weights)
    bucket_idx = rand(Distributions.Categorical(weights))

    bucket = sampler.shuffled_buckets[bucket_idx]

    # Sample from this bucket (with replacement if needed)
    if length(bucket) >= batch_size
        return bucket[rand(1:length(bucket), batch_size)]
    else
        return rand(bucket, batch_size)
    end
end

"""
    reset_epoch!(sampler::LengthBucketedSampler)

Reshuffle all buckets for a new epoch.
"""
function reset_epoch!(sampler::LengthBucketedSampler)
    for i in eachindex(sampler.shuffled_buckets)
        sampler.shuffled_buckets[i] = shuffle(copy(sampler.buckets[i]))
    end
    sampler.current_bucket[] = 1
    sampler.current_pos[] = 1
end

"""
    batch_from_precomputed(proteins::Vector, indices::Vector{Int}, P)

Create flow matching batch from precomputed proteins (NamedTuples).
Uses the schedule transforms defined in the RDNFlow processes P[1] (CA) and P[2] (latents).
The schedule is now baked into Flowfusion's RDNFlow, ensuring consistency with inference.

Samples z_latent from the stored mean and log_scale using reparameterization trick,
matching Python training behavior.

Returns named tuple with all tensors on GPU.
"""
function batch_from_precomputed(proteins::Vector,
                                 indices::Vector{Int}, P)
    samples = [proteins[i] for i in indices]
    min_len = minimum(length(s.mask) for s in samples)
    B = length(samples)
    latent_dim = size(samples[1].z_mean, 1)

    # Stack and truncate
    x1_ca = zeros(Float32, 3, min_len, B)
    z_mean = zeros(Float32, latent_dim, min_len, B)
    z_log_scale = zeros(Float32, latent_dim, min_len, B)
    mask = zeros(Float32, min_len, B)

    for (b, s) in enumerate(samples)
        x1_ca[:, :, b] = s.ca_coords[:, 1:min_len]
        z_mean[:, :, b] = s.z_mean[:, 1:min_len]
        z_log_scale[:, :, b] = s.z_log_scale[:, 1:min_len]
        mask[:, b] = s.mask[1:min_len]
    end

    # Sample z_latent using reparameterization trick (matches Python training)
    x1_ll = z_mean .+ randn(Float32, size(z_mean)) .* exp.(z_log_scale)

    # Move to GPU
    x1_ca_gpu = gpu(x1_ca)
    x1_ll_gpu = gpu(x1_ll)
    mask_gpu = gpu(mask)

    L = min_len

    # Sample noise
    x0_ca = gpu(Float32.(Flowfusion.sample_rdn_noise(P[1], L, B)))
    x0_ll = gpu(Float32.(Flowfusion.sample_rdn_noise(P[2], L, B)))

    # Sample uniform "progress" u, then use Flowfusion's schedule_transform
    # to get the actual interpolation times for each component
    u_vec = rand(Float32, B)
    t_ca_vec = Flowfusion.schedule_transform(P[1], u_vec)  # Uses schedule from P[1]
    t_ll_vec = Flowfusion.schedule_transform(P[2], u_vec)  # Uses schedule from P[2]

    t_ca_gpu = gpu(t_ca_vec)
    t_ll_gpu = gpu(t_ll_vec)
    t_ca_bc = reshape(t_ca_gpu, 1, 1, B)
    t_ll_bc = reshape(t_ll_gpu, 1, 1, B)

    # Interpolation with different t for each component
    xt_ca = (1f0 .- t_ca_bc) .* x0_ca .+ t_ca_bc .* x1_ca_gpu
    xt_ll = (1f0 .- t_ll_bc) .* x0_ll .+ t_ll_bc .* x1_ll_gpu

    # Zero COM for CA
    if P[1].zero_com
        xt_ca = xt_ca .- mean(xt_ca, dims=2)
    end

    # t_model is the "progress" u that gets passed to the model
    t_model_gpu = gpu(u_vec)

    return (
        xt_ca=xt_ca, xt_ll=xt_ll,
        x1_ca=x1_ca_gpu, x1_ll=x1_ll_gpu,
        t_ca=t_ca_gpu, t_ll=t_ll_gpu,
        t_model=t_model_gpu,
        mask=mask_gpu
    )
end
