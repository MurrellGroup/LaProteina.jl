# VAE Encoder
# Port of encoder.py from la-proteina

using Flux

"""
    EncoderTransformer(;
        n_layers::Int=12,
        token_dim::Int=768,
        pair_dim::Int=64,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        qk_ln::Bool=false,
        update_pair_repr::Bool=true,
        update_pair_every_n::Int=3,
        use_tri_mult::Bool=false)

VAE Encoder transformer with pair-biased attention.

# Architecture
1. Feature factories for initial seq/pair representations and conditioning
2. Conditioning transitions (2x SwiGLU)
3. N transformer layers with optional pair updates
4. Output projection to mean and log_scale

# Arguments
- `n_layers`: Number of transformer layers (default 12)
- `token_dim`: Token/sequence dimension (default 768)
- `pair_dim`: Pair representation dimension (default 64)
- `n_heads`: Number of attention heads (default 12)
- `dim_cond`: Conditioning dimension (default 256)
- `latent_dim`: Output latent dimension (default 8)
- `qk_ln`: Whether to use Q/K layer norms (default false)
- `update_pair_repr`: Whether to update pair representation (default true)
- `update_pair_every_n`: Update pair every N layers (default 3)
- `use_tri_mult`: Use triangle multiplicative in pair updates (default false)
"""
struct EncoderTransformer
    # Feature factories
    init_repr_factory::FeatureFactory
    cond_factory::FeatureFactory
    pair_rep_factory::FeatureFactory

    # Conditioning transitions
    transition_c_1::ConditioningTransition
    transition_c_2::ConditioningTransition

    # Transformer layers
    transformer_layers::Vector{TransformerBlock}

    # Pair update layers (optional)
    pair_update_layers::Vector{Union{PairUpdate, Nothing}}

    # Output projection
    latent_proj::Flux.Chain

    # Config
    n_layers::Int
    update_pair_repr::Bool
end

Flux.@layer EncoderTransformer

function EncoderTransformer(;
        n_layers::Int=12,
        token_dim::Int=768,
        pair_dim::Int=64,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        qk_ln::Bool=false,
        update_pair_repr::Bool=true,
        update_pair_every_n::Int=3,
        use_tri_mult::Bool=false)

    # Feature factories
    init_repr_factory = encoder_seq_features(token_dim; latent_dim=latent_dim)
    cond_factory = encoder_cond_features(dim_cond)
    pair_rep_factory = encoder_pair_features(pair_dim)

    # Conditioning transitions
    transition_c_1 = ConditioningTransition(dim_cond; expansion_factor=2)
    transition_c_2 = ConditioningTransition(dim_cond; expansion_factor=2)

    # Transformer layers
    transformer_layers = [
        TransformerBlock(
            dim_token=token_dim,
            dim_pair=pair_dim,
            n_heads=n_heads,
            dim_cond=dim_cond,
            qk_ln=qk_ln,
            residual_mha=true,
            residual_transition=true,
            parallel=false
        )
        for _ in 1:n_layers
    ]

    # Pair update layers
    pair_update_layers = Vector{Union{PairUpdate, Nothing}}(nothing, n_layers - 1)
    if update_pair_repr
        for i in 1:(n_layers-1)
            if i % update_pair_every_n == 0
                pair_update_layers[i] = PairUpdate(token_dim, pair_dim; use_tri_mult=use_tri_mult)
            end
        end
    end

    # Output projection: LayerNorm -> Linear to 2*latent_dim (mean and log_scale)
    latent_proj = Flux.Chain(
        PyTorchLayerNorm(token_dim),
        Dense(token_dim => 2 * latent_dim; bias=false)
    )

    return EncoderTransformer(
        init_repr_factory, cond_factory, pair_rep_factory,
        transition_c_1, transition_c_2,
        transformer_layers, pair_update_layers, latent_proj,
        n_layers, update_pair_repr
    )
end

function (m::EncoderTransformer)(batch::Dict)
    # Extract mask from batch
    mask = get(batch, :mask, nothing)
    if isnothing(mask)
        # Try to infer mask from coords
        if haskey(batch, :coords)
            L, B = size(batch[:coords], 3), size(batch[:coords], 4)
            mask = ones(Float32, L, B)
        else
            error("Cannot determine mask from batch")
        end
    end

    # Get conditioning variables
    cond = m.cond_factory(batch)  # [dim_cond, L, B]
    cond = m.transition_c_1(cond, mask)
    cond = m.transition_c_2(cond, mask)

    # Get initial sequence representation
    seqs = m.init_repr_factory(batch)  # [token_dim, L, B]
    mask_exp = reshape(mask, 1, size(mask)...)
    seqs = seqs .* mask_exp

    # Get pair representation
    pair_rep = m.pair_rep_factory(batch)  # [pair_dim, L, L, B]

    # Run transformer layers
    for i in 1:m.n_layers
        seqs = m.transformer_layers[i](seqs, pair_rep, cond, mask)

        # Optional pair update
        if m.update_pair_repr && i < m.n_layers
            if !isnothing(m.pair_update_layers[i])
                pair_rep = m.pair_update_layers[i](seqs, pair_rep, mask)
            end
        end
    end

    # Project to latent space
    latent_out = m.latent_proj(seqs)  # [2*latent_dim, L, B]
    latent_out = latent_out .* mask_exp

    # Split into mean and log_scale
    latent_dim = size(latent_out, 1) ÷ 2
    mean = latent_out[1:latent_dim, :, :]
    log_scale = latent_out[latent_dim+1:end, :, :]

    # Reparameterization trick
    z_latent = mean .+ randn(eltype(mean), size(log_scale)) .* exp.(log_scale)
    z_latent = z_latent .* mask_exp

    return Dict(
        :mean => mean,
        :log_scale => log_scale,
        :z_latent => z_latent,
        :z_latent_pre_ln => z_latent  # Could add a separate LN if needed
    )
end

"""
    encode(encoder::EncoderTransformer, batch::Dict; deterministic::Bool=false)

Convenience function to encode a batch.
If deterministic=true, returns mean instead of sampled z.
"""
function encode(encoder::EncoderTransformer, batch::Dict; deterministic::Bool=false)
    result = encoder(batch)
    if deterministic
        result[:z_latent] = result[:mean]
    end
    return result
end
