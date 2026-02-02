# Time and positional embeddings
# Port of get_time_embedding and get_index_embedding from feature_factory.py

"""
    get_time_embedding(t::AbstractVector{T}, edim::Int; max_positions::Int=2000) where T

Create sinusoidal time embeddings for a vector of times.

Code from Frameflow, which got it from
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

# Arguments
- `t`: Vector of times (float) of shape [B]
- `edim`: Dimension of the embeddings
- `max_positions`: Maximum position scaling factor

# Returns
- Embedding tensor of shape [edim, B] (Julia convention)
"""
function get_time_embedding(t::AbstractVector{T}, edim::Int; max_positions::Int=2000) where T
    t_scaled = t .* T(max_positions)
    half_dim = edim ÷ 2

    # exp(-log(max_positions) * k / (half_dim - 1)) for k = 0, 1, ..., half_dim-1
    log_max = T(log(max_positions))
    k = T.(0:(half_dim-1))
    emb_scale = exp.(-log_max .* k ./ T(half_dim - 1))  # [half_dim]

    # Outer product: t_scaled[b] * emb_scale[k]
    emb = t_scaled' .* emb_scale  # [half_dim, B]

    # Concatenate sin and cos
    emb_sin = sin.(emb)  # [half_dim, B]
    emb_cos = cos.(emb)  # [half_dim, B]
    result = vcat(emb_sin, emb_cos)  # [edim, B]

    # Handle odd edim by padding with zeros
    if edim % 2 == 1
        result = vcat(result, zeros(T, 1, length(t)))
    end

    return result
end

# Convenience method for single time value
function get_time_embedding(t::T, edim::Int; max_positions::Int=2000) where T<:Real
    return get_time_embedding([t], edim; max_positions=max_positions)[:, 1]
end

"""
    get_index_embedding(indices::AbstractVector{<:Integer}, edim::Int; max_len::Int=2056)

Create sinusoidal positional embeddings from indices.

# Arguments
- `indices`: Vector of integer indices of shape [L]
- `edim`: Dimension of the embeddings to create
- `max_len`: Maximum length for positional encoding

# Returns
- Positional embedding of shape [edim, L]
"""
function get_index_embedding(indices::AbstractVector{<:Integer}, edim::Int; max_len::Int=2056)
    T = Float32
    half_dim = edim ÷ 2
    k = T.(0:(half_dim-1))  # [edim/2]

    # pi / (max_len^(2k/edim))
    scale = T(π) ./ (T(max_len) .^ (2 .* k ./ T(edim)))  # [edim/2]

    # indices[:, newaxis] * scale[newaxis, :]
    angles = T.(indices)' .* scale  # [half_dim, L]

    emb_sin = sin.(angles)
    emb_cos = cos.(angles)
    result = vcat(emb_sin, emb_cos)  # [edim, L]

    return result
end

"""
    get_index_embedding(indices::AbstractMatrix{<:Integer}, edim::Int; max_len::Int=2056)

Create sinusoidal positional embeddings from batched indices.

# Arguments
- `indices`: Matrix of indices of shape [L, B]
- `edim`: Dimension of the embeddings to create
- `max_len`: Maximum length for positional encoding

# Returns
- Positional embedding of shape [edim, L, B]
"""
function get_index_embedding(indices::AbstractMatrix{<:Integer}, edim::Int; max_len::Int=2056)
    T = Float32
    L, B = size(indices)
    half_dim = edim ÷ 2
    k = T.(0:(half_dim-1))  # [edim/2]

    # Compute scale factors
    scale = T(π) ./ (T(max_len) .^ (2 .* k ./ T(edim)))  # [edim/2]

    # Reshape for broadcasting: indices [L, B] -> [1, L, B], scale [edim/2] -> [edim/2, 1, 1]
    angles = reshape(scale, :, 1, 1) .* reshape(T.(indices), 1, L, B)  # [edim/2, L, B]

    emb_sin = sin.(angles)
    emb_cos = cos.(angles)
    result = vcat(emb_sin, emb_cos)  # [edim, L, B]

    return result
end

"""
    broadcast_time_embedding(t_emb::AbstractMatrix{T}, L::Int) where T

Broadcast time embedding [edim, B] to sequence length [edim, L, B].
"""
function broadcast_time_embedding(t_emb::AbstractMatrix{T}, L::Int) where T
    edim, B = size(t_emb)
    return reshape(t_emb, edim, 1, B) .* ones(T, 1, L, 1)
end

# Time sampling utilities for flow matching

"""
    sample_t_uniform(n::Int; T=Float32)

Sample times uniformly from [0, 1].
"""
function sample_t_uniform(n::Int; T=Float32)
    return rand(T, n)
end

"""
    sample_t_beta(n::Int, alpha::Real, beta::Real; T=Float32)

Sample times from Beta(alpha, beta) distribution.
"""
function sample_t_beta(n::Int, alpha::Real, beta::Real; T=Float32)
    d = Beta(T(alpha), T(beta))
    return T.(rand(d, n))
end

"""
    sample_t_mix_unif_beta(n::Int, p_unif::Real, alpha::Real, beta::Real; T=Float32)

Sample times from mixture of Uniform and Beta distributions.

# Arguments
- `n`: Number of samples
- `p_unif`: Probability of sampling from uniform (vs Beta)
- `alpha`, `beta`: Parameters for Beta distribution

# Returns
- Vector of sampled times
"""
function sample_t_mix_unif_beta(n::Int, p_unif::Real, alpha::Real, beta::Real; T=Float32)
    t = zeros(T, n)
    for i in 1:n
        if rand() < p_unif
            t[i] = rand(T)
        else
            t[i] = T(rand(Beta(T(alpha), T(beta))))
        end
    end
    return t
end

"""
    gt_schedule(t::T, mode::Symbol, param::Real) where T

Compute noise injection schedule g(t) for SDE sampling.

# Arguments
- `t`: Current time
- `mode`: Schedule mode (:const, :tan, :linear)
- `param`: Schedule parameter

# Returns
- Noise injection value g(t)
"""
function gt_schedule(t::T, mode::Symbol, param::Real) where T
    if mode == :const
        return T(param)
    elseif mode == :tan
        # g(t) = param * tan(π/2 * t) - peaks near t=1
        return T(param) * tan(T(π/2) * t)
    elseif mode == :linear
        return T(param) * t
    else
        error("Unknown gt_schedule mode: $mode")
    end
end

"""
    inference_time_steps(n_steps::Int; start::Real=0, stop::Real=1, T=Float32)

Generate evenly spaced time steps for inference.

# Returns
- Vector of times from start to stop (inclusive)
"""
function inference_time_steps(n_steps::Int; start::Real=0, stop::Real=1, T=Float32)
    return T.(range(start, stop, length=n_steps+1))
end
