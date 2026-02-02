# VAE Decoder
# Port of decoder.py from la-proteina

using Flux

"""
    DecoderTransformer(;
        n_layers::Int=12,
        token_dim::Int=768,
        pair_dim::Int=64,
        n_heads::Int=12,
        dim_cond::Int=256,
        latent_dim::Int=8,
        qk_ln::Bool=false,
        update_pair_repr::Bool=true,
        update_pair_every_n::Int=3,
        use_tri_mult::Bool=false,
        abs_coors::Bool=true)

VAE Decoder transformer with pair-biased attention.
Takes z_latent and CA coordinates, outputs sequence logits and all-atom coordinates.

# Architecture
1. Feature factories for initial seq/pair representations and conditioning
2. Conditioning transitions (2x SwiGLU)
3. N transformer layers with optional pair updates
4. Output projections for sequence logits and atom coordinates

# Arguments
- `n_layers`: Number of transformer layers (default 12)
- `token_dim`: Token/sequence dimension (default 768)
- `pair_dim`: Pair representation dimension (default 64)
- `n_heads`: Number of attention heads (default 12)
- `dim_cond`: Conditioning dimension (default 256)
- `latent_dim`: Input latent dimension (default 8)
- `qk_ln`: Whether to use Q/K layer norms (default false)
- `update_pair_repr`: Whether to update pair representation (default true)
- `update_pair_every_n`: Update pair every N layers (default 3)
- `use_tri_mult`: Use triangle multiplicative in pair updates (default false)
- `abs_coors`: Use absolute coordinates (CA at input position) vs relative
"""
struct DecoderTransformer
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

    # Output projections
    logit_proj::Flux.Chain
    struct_proj::Flux.Chain

    # Config
    n_layers::Int
    update_pair_repr::Bool
    abs_coors::Bool
end

Flux.@layer DecoderTransformer

function DecoderTransformer(;
        n_layers::Int=12,
        token_dim::Int=768,
        pair_dim::Int=256,      # Python default: 256
        n_heads::Int=12,
        dim_cond::Int=128,      # Python default: 128
        latent_dim::Int=8,
        qk_ln::Bool=true,       # Python default: True
        update_pair_repr::Bool=false,  # Python default: False
        update_pair_every_n::Int=3,
        use_tri_mult::Bool=false,
        abs_coors::Bool=false)  # Python default: False

    # Feature factories for decoder - match Python exactly
    init_repr_factory = decoder_seq_features(token_dim; latent_dim=latent_dim)
    cond_factory = decoder_cond_features(dim_cond)  # Decoder-specific (empty features -> zeros)
    pair_rep_factory = decoder_pair_features(pair_dim)  # Decoder-specific (rel_seq_sep + ca_pair_dist)

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

    # Output projections
    # Sequence logits: 20 amino acid classes
    logit_proj = Flux.Chain(
        PyTorchLayerNorm(token_dim),
        Dense(token_dim => 20; bias=false)
    )

    # Structure output: 37 atoms * 3 coordinates = 111
    struct_proj = Flux.Chain(
        PyTorchLayerNorm(token_dim),
        Dense(token_dim => 37 * 3; bias=false)
    )

    return DecoderTransformer(
        init_repr_factory, cond_factory, pair_rep_factory,
        transition_c_1, transition_c_2,
        transformer_layers, pair_update_layers,
        logit_proj, struct_proj,
        n_layers, update_pair_repr, abs_coors
    )
end

function (m::DecoderTransformer)(input::Dict)
    # Required inputs:
    # - z_latent: [latent_dim, L, B]
    # - ca_coors: [3, L, B] (CA coordinates)
    # - mask or residue_mask: [L, B]

    # Extract inputs
    z_latent = input[:z_latent]
    ca_coors = get(input, :ca_coors, get(input, :ca_coors_nm, nothing))
    mask = get(input, :mask, get(input, :residue_mask, nothing))

    if isnothing(ca_coors)
        error("Decoder requires ca_coors or ca_coors_nm in input")
    end
    if isnothing(mask)
        L, B = size(z_latent, 2), size(z_latent, 3)
        mask = ones(Float32, L, B)
    end

    # Prepare batch dict for feature factories
    batch = Dict(
        :z_latent => z_latent,
        :ca_coors => ca_coors,
        :mask => mask
    )

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

    # Project to outputs
    logits = m.logit_proj(seqs)  # [20, L, B]
    logits = logits .* mask_exp

    struct_flat = m.struct_proj(seqs)  # [111, L, B]
    struct_flat = struct_flat .* mask_exp

    # Reshape structure output: [111, L, B] -> [3, 37, L, B]
    L, B = size(struct_flat, 2), size(struct_flat, 3)
    coors_a37 = reshape(struct_flat, 3, 37, L, B)

    # Handle CA coordinates
    if m.abs_coors
        # Set CA (index 2) to input CA coordinates
        coors_a37[:, CA_INDEX, :, :] .= ca_coors
    else
        # Add CA coordinates to all atoms (relative prediction)
        coors_a37[:, CA_INDEX, :, :] .= 0  # Zero out CA
        coors_a37 = coors_a37 .+ reshape(ca_coors, 3, 1, L, B)
    end

    # Get predicted sequence
    _, aatype_idx = findmax(logits; dims=1)  # Returns tuple (max_values, indices)
    aatype_max = dropdims(getindex.(aatype_idx, 1); dims=1)  # Extract first component, [L, B]

    # Get atom mask from predicted sequence
    atom_mask = get_atom_mask_from_aatype(aatype_max)  # [L, B, 37]
    atom_mask = permutedims(atom_mask, (3, 1, 2))  # [37, L, B]
    # Apply residue mask and convert to Bool
    mask_expanded = reshape(mask, 1, size(mask)...) .> 0.5f0
    atom_mask = atom_mask .& mask_expanded

    return Dict(
        :coors => coors_a37,           # [3, 37, L, B]
        :seq_logits => logits,         # [20, L, B]
        :residue_mask => mask,         # [L, B]
        :aatype_max => aatype_max,     # [L, B]
        :atom_mask => atom_mask        # [37, L, B]
    )
end

"""
    get_atom_mask_from_aatype(aatype::AbstractArray{<:Integer})

Get atom37 mask from amino acid types.
"""
function get_atom_mask_from_aatype(aatype::AbstractArray{<:Integer})
    # aatype: [L, B] with values 1-20
    # RESTYPE_ATOM37_MASK: [21, 37]
    L, B = size(aatype)
    mask = zeros(Bool, L, B, 37)
    for b in 1:B, l in 1:L
        aa = clamp(aatype[l, b], 1, 21)
        mask[l, b, :] .= RESTYPE_ATOM37_MASK[aa, :]
    end
    return mask
end

"""
    decode(decoder::DecoderTransformer, z_latent, ca_coors; mask=nothing)

Convenience function to decode latents with CA coordinates.
"""
function decode(decoder::DecoderTransformer, z_latent, ca_coors; mask=nothing)
    L, B = size(z_latent, 2), size(z_latent, 3)
    if isnothing(mask)
        mask = ones(Float32, L, B)
    end
    input = Dict(
        :z_latent => z_latent,
        :ca_coors => ca_coors,
        :mask => mask
    )
    return decoder(input)
end
