# Full VAE Autoencoder
# Combines Encoder and Decoder with loss functions

using Flux
using NNlib: logsoftmax, softmax

"""
    Autoencoder(;
        n_layers::Int=12,
        token_dim::Int=768,
        pair_dim::Int=64,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        qk_ln::Bool=false,
        update_pair_repr::Bool=true,
        update_pair_every_n::Int=3,
        abs_coors::Bool=true)

VAE Autoencoder combining encoder and decoder.

# Arguments
See EncoderTransformer and DecoderTransformer for argument descriptions.
"""
struct Autoencoder
    encoder::EncoderTransformer
    decoder::DecoderTransformer
    latent_dim::Int
end

Flux.@layer Autoencoder

function Autoencoder(;
        n_layers::Int=12,
        token_dim::Int=768,
        pair_dim::Int=64,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        qk_ln::Bool=false,
        update_pair_repr::Bool=true,
        update_pair_every_n::Int=3,
        abs_coors::Bool=true)

    encoder = EncoderTransformer(;
        n_layers=n_layers,
        token_dim=token_dim,
        pair_dim=pair_dim,
        n_heads=n_heads,
        dim_cond=dim_cond,
        latent_dim=latent_dim,
        qk_ln=qk_ln,
        update_pair_repr=update_pair_repr,
        update_pair_every_n=update_pair_every_n
    )

    decoder = DecoderTransformer(;
        n_layers=n_layers,
        token_dim=token_dim,
        pair_dim=pair_dim,
        n_heads=n_heads,
        dim_cond=dim_cond,
        latent_dim=latent_dim,
        qk_ln=qk_ln,
        update_pair_repr=update_pair_repr,
        update_pair_every_n=update_pair_every_n,
        abs_coors=abs_coors
    )

    return Autoencoder(encoder, decoder, latent_dim)
end

function (m::Autoencoder)(batch::Dict)
    # Encode
    enc_out = m.encoder(batch)

    # Get CA coordinates for decoder
    if haskey(batch, :ca_coors)
        ca_coors = batch[:ca_coors]
    elseif haskey(batch, :coords)
        ca_coors = batch[:coords][:, CA_INDEX, :, :]
    else
        error("Cannot determine CA coordinates from batch")
    end

    # Decode
    dec_input = Dict(
        :z_latent => enc_out[:z_latent],
        :ca_coors => ca_coors,
        :mask => get(batch, :mask, nothing)
    )
    dec_out = m.decoder(dec_input)

    # Combine outputs
    return merge(enc_out, dec_out)
end

"""
    vae_loss(model::Autoencoder, batch::Dict;
        kl_weight::Float32=1.0f0,
        coord_weight::Float32=1.0f0,
        seq_weight::Float32=1.0f0)

Compute VAE loss: KL divergence + reconstruction loss (coordinates + sequence).

# Arguments
- `model`: Autoencoder model
- `batch`: Input batch with :coords, :aatype, :mask
- `kl_weight`: Weight for KL divergence loss
- `coord_weight`: Weight for coordinate reconstruction loss
- `seq_weight`: Weight for sequence reconstruction loss

# Returns
Named tuple with :total, :kl, :coord, :seq losses
"""
function vae_loss(model::Autoencoder, batch::Dict;
        kl_weight::Float32=1.0f0,
        coord_weight::Float32=1.0f0,
        seq_weight::Float32=1.0f0)

    # Forward pass
    output = model(batch)

    # Get targets
    coords_target = batch[:coords]  # [3, 37, L, B]
    aatype_target = batch[:aatype]  # [L, B] - integer indices 1-20
    mask = get(batch, :mask, ones(Float32, size(aatype_target)))  # [L, B]

    # KL divergence loss
    mean = output[:mean]      # [latent_dim, L, B]
    log_scale = output[:log_scale]  # [latent_dim, L, B]
    kl_loss = kl_divergence(mean, log_scale, mask)

    # Coordinate reconstruction loss
    coords_pred = output[:coors]  # [3, 37, L, B]
    coord_loss = coordinate_loss(coords_pred, coords_target, mask, batch)

    # Sequence reconstruction loss
    seq_logits = output[:seq_logits]  # [20, L, B]
    seq_loss = sequence_loss(seq_logits, aatype_target, mask)

    # Total loss
    total_loss = kl_weight * kl_loss + coord_weight * coord_loss + seq_weight * seq_loss

    return (
        total=total_loss,
        kl=kl_loss,
        coord=coord_loss,
        seq=seq_loss
    )
end

"""
    kl_divergence(mean, log_scale, mask)

KL divergence between N(mean, exp(2*log_scale)) and N(0, I).

KL = 0.5 * sum(exp(2*log_scale) + mean^2 - 1 - 2*log_scale)
"""
function kl_divergence(mean::AbstractArray{T}, log_scale::AbstractArray{T}, mask) where T
    # mean, log_scale: [latent_dim, L, B]
    # mask: [L, B]

    variance = exp.(2 .* log_scale)
    kl_per_element = T(0.5) .* (variance .+ mean.^2 .- one(T) .- 2 .* log_scale)

    # Sum over latent dim
    kl_per_position = sum(kl_per_element; dims=1)  # [1, L, B]
    kl_per_position = dropdims(kl_per_position; dims=1)  # [L, B]

    # Masked mean
    n_valid = max(sum(mask), one(T))
    return sum(kl_per_position .* mask) / n_valid
end

"""
    coordinate_loss(pred, target, mask, batch)

MSE loss on atom coordinates, accounting for atom mask.
"""
function coordinate_loss(pred::AbstractArray{T}, target::AbstractArray{T}, mask, batch) where T
    # pred, target: [3, 37, L, B]
    # mask: [L, B]

    # Get atom mask from target sequence
    if haskey(batch, :atom_mask)
        atom_mask = batch[:atom_mask]  # [37, L, B]
    elseif haskey(batch, :aatype)
        aatype = batch[:aatype]
        atom_mask_lb37 = get_atom_mask_from_aatype(Int.(aatype))  # [L, B, 37]
        atom_mask = permutedims(atom_mask_lb37, (3, 1, 2))  # [37, L, B]
    else
        # Use all atoms
        atom_mask = ones(T, 37, size(mask)...)
    end

    # Expand masks
    residue_mask_exp = reshape(mask, 1, 1, size(mask)...)  # [1, 1, L, B]
    atom_mask_exp = reshape(atom_mask, 1, size(atom_mask)...)  # [1, 37, L, B]
    full_mask = residue_mask_exp .* atom_mask_exp  # [1, 37, L, B]

    # MSE
    sq_diff = (pred .- target) .^ 2  # [3, 37, L, B]
    masked_sq_diff = sq_diff .* full_mask

    # Mean over valid atoms
    n_valid = max(sum(full_mask), one(T))
    return sum(masked_sq_diff) / (3 * n_valid)  # Normalize by 3 for xyz
end

"""
    sequence_loss(logits, target, mask)

Cross-entropy loss for sequence prediction.
"""
function sequence_loss(logits::AbstractArray{T}, target::AbstractArray{<:Integer}, mask) where T
    # logits: [20, L, B]
    # target: [L, B] - indices 1-20
    # mask: [L, B]

    # Log softmax
    log_probs = logsoftmax(logits; dims=1)  # [20, L, B]

    # Gather target probabilities
    L, B = size(target)
    ce_per_position = zeros(T, L, B)
    for b in 1:B, l in 1:L
        idx = clamp(target[l, b], 1, 20)
        ce_per_position[l, b] = -log_probs[idx, l, b]
    end

    # Masked mean
    n_valid = max(sum(mask), one(T))
    return sum(ce_per_position .* mask) / n_valid
end

"""
    encode_decode(model::Autoencoder, batch::Dict; deterministic::Bool=false)

Run full encode-decode pipeline.
"""
function encode_decode(model::Autoencoder, batch::Dict; deterministic::Bool=false)
    enc_out = encode(model.encoder, batch; deterministic=deterministic)

    # Get CA coordinates
    if haskey(batch, :ca_coors)
        ca_coors = batch[:ca_coors]
    elseif haskey(batch, :coords)
        ca_coors = batch[:coords][:, CA_INDEX, :, :]
    else
        error("Cannot determine CA coordinates")
    end

    dec_out = decode(model.decoder, enc_out[:z_latent], ca_coors;
        mask=get(batch, :mask, nothing))

    return merge(enc_out, dec_out)
end
