# Flash Multi-Head Attention kernels (cuTile) for LaProteina
# Adapted from OnionTile.jl. Layout: (D, SeqLen, Heads, Batch)
# LaProteina dims: D_k=D_v=64, h=12, B=4, L=100-200

# ============================================================================
# Flash attention kernel (no bias)
# ============================================================================

function _fmha_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Out::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)

    m_i = ct.full((1, TILE_M[]), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M[]), Float32)
    acc = ct.zeros((D_V[], TILE_M[]), Float32)

    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q = reshape(q, (D_K[], TILE_M[]))

    k_seqlen = K.sizes[2]
    Tc = cld(k_seqlen, TILE_N[])

    j = Int32(1)
    while j <= Tc
        k = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1), latency=2)
        k = reshape(k, (D_K[], TILE_N[]))
        k = transpose(k)

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k, T), _to_tf32(q, T), qk)
        else
            qk = ct.muladd(k, q, qk)
        end

        if j == Tc
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        m_ij = max.(m_i, maximum(qk, dims=1) * qk_scale_log2)
        qk = qk * qk_scale_log2 .- m_ij
        p = exp2.(qk)
        l_ij = sum(p, dims=1)
        alpha = exp2.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        v = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1), latency=4)
        v = reshape(v, (D_V[], TILE_N[]))
        p = ct.astype(p, T)
        if T === Float32
            acc = ct.muladd(_to_tf32(v, T), _to_tf32(p, T), acc)
        else
            acc = ct.muladd(v, p, acc)
        end
        m_i = m_ij

        j += Int32(1)
    end

    acc = acc ./ l_i
    acc = reshape(acc, (D_V[], TILE_M[], 1, 1))
    acc = ct.astype(acc, T)
    ct.store(Out, (1, bid_x, head_idx, batch_idx), acc)
    return
end

# ============================================================================
# Flash attention kernel with additive bias
# ============================================================================

function _fmha_bias_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Bias::ct.TileArray{T,4},
    Out::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
    BIAS_BATCH::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    bias_batch_idx = BIAS_BATCH[] == Int32(1) ? Int32(1) : batch_idx

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)

    m_i = ct.full((1, TILE_M[]), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M[]), Float32)
    acc = ct.zeros((D_V[], TILE_M[]), Float32)

    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q = reshape(q, (D_K[], TILE_M[]))

    k_seqlen = K.sizes[2]
    Tc = cld(k_seqlen, TILE_N[])

    j = Int32(1)
    while j <= Tc
        k = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1), latency=2)
        k = reshape(k, (D_K[], TILE_N[]))
        k = transpose(k)

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k, T), _to_tf32(q, T), qk)
        else
            qk = ct.muladd(k, q, qk)
        end

        qk = qk * qk_scale_log2

        # Bias layout: (SeqK, SeqQ, H, bias_batch)
        bias_tile = ct.load(Bias, (j, bid_x, head_idx, bias_batch_idx), (TILE_N[], TILE_M[], 1, 1))
        bias_tile = reshape(bias_tile, (TILE_N[], TILE_M[]))
        qk = qk .+ bias_tile * Float32(_INV_LOG_2)

        if j == Tc
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        m_ij = max.(m_i, maximum(qk, dims=1))
        qk = qk .- m_ij
        p = exp2.(qk)
        l_ij = sum(p, dims=1)
        alpha = exp2.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        v = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1), latency=4)
        v = reshape(v, (D_V[], TILE_N[]))
        p = ct.astype(p, T)
        if T === Float32
            acc = ct.muladd(_to_tf32(v, T), _to_tf32(p, T), acc)
        else
            acc = ct.muladd(v, p, acc)
        end
        m_i = m_ij

        j += Int32(1)
    end

    acc = acc ./ l_i
    acc = reshape(acc, (D_V[], TILE_M[], 1, 1))
    acc = ct.astype(acc, T)
    ct.store(Out, (1, bid_x, head_idx, batch_idx), acc)
    return
end

# ============================================================================
# Training variant (no bias) — saves logsumexp
# ============================================================================

function _fmha_lse_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Out::ct.TileArray{T,4},
    Lse::ct.TileArray{Float32,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)

    m_i = ct.full((1, TILE_M[]), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M[]), Float32)
    acc = ct.zeros((D_V[], TILE_M[]), Float32)

    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q = reshape(q, (D_K[], TILE_M[]))

    k_seqlen = K.sizes[2]
    Tc = cld(k_seqlen, TILE_N[])

    j = Int32(1)
    while j <= Tc
        k = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1), latency=2)
        k = reshape(k, (D_K[], TILE_N[]))
        k = transpose(k)

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k, T), _to_tf32(q, T), qk)
        else
            qk = ct.muladd(k, q, qk)
        end

        if j == Tc
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        m_ij = max.(m_i, maximum(qk, dims=1) * qk_scale_log2)
        qk = qk * qk_scale_log2 .- m_ij
        p = exp2.(qk)
        l_ij = sum(p, dims=1)
        alpha = exp2.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        v = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1), latency=4)
        v = reshape(v, (D_V[], TILE_N[]))
        p = ct.astype(p, T)
        if T === Float32
            acc = ct.muladd(_to_tf32(v, T), _to_tf32(p, T), acc)
        else
            acc = ct.muladd(v, p, acc)
        end
        m_i = m_ij

        j += Int32(1)
    end

    lse = m_i .+ log2.(l_i)
    lse = reshape(lse, (1, TILE_M[], 1, 1))
    ct.store(Lse, (1, bid_x, head_idx, batch_idx), lse)

    acc = acc ./ l_i
    acc = reshape(acc, (D_V[], TILE_M[], 1, 1))
    acc = ct.astype(acc, T)
    ct.store(Out, (1, bid_x, head_idx, batch_idx), acc)
    return
end

# ============================================================================
# Training variant with bias — saves logsumexp
# ============================================================================

function _fmha_bias_lse_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Bias::ct.TileArray{T,4},
    Out::ct.TileArray{T,4},
    Lse::ct.TileArray{Float32,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
    BIAS_BATCH::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    bias_batch_idx = BIAS_BATCH[] == Int32(1) ? Int32(1) : batch_idx

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)

    m_i = ct.full((1, TILE_M[]), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M[]), Float32)
    acc = ct.zeros((D_V[], TILE_M[]), Float32)

    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q = reshape(q, (D_K[], TILE_M[]))

    k_seqlen = K.sizes[2]
    Tc = cld(k_seqlen, TILE_N[])

    j = Int32(1)
    while j <= Tc
        k = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1), latency=2)
        k = reshape(k, (D_K[], TILE_N[]))
        k = transpose(k)

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k, T), _to_tf32(q, T), qk)
        else
            qk = ct.muladd(k, q, qk)
        end

        qk = qk * qk_scale_log2

        bias_tile = ct.load(Bias, (j, bid_x, head_idx, bias_batch_idx), (TILE_N[], TILE_M[], 1, 1))
        bias_tile = reshape(bias_tile, (TILE_N[], TILE_M[]))
        qk = qk .+ bias_tile * Float32(_INV_LOG_2)

        if j == Tc
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        m_ij = max.(m_i, maximum(qk, dims=1))
        qk = qk .- m_ij
        p = exp2.(qk)
        l_ij = sum(p, dims=1)
        alpha = exp2.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        v = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1), latency=4)
        v = reshape(v, (D_V[], TILE_N[]))
        p = ct.astype(p, T)
        if T === Float32
            acc = ct.muladd(_to_tf32(v, T), _to_tf32(p, T), acc)
        else
            acc = ct.muladd(v, p, acc)
        end
        m_i = m_ij

        j += Int32(1)
    end

    lse = m_i .+ log2.(l_i)
    lse = reshape(lse, (1, TILE_M[], 1, 1))
    ct.store(Lse, (1, bid_x, head_idx, batch_idx), lse)

    acc = acc ./ l_i
    acc = reshape(acc, (D_V[], TILE_M[], 1, 1))
    acc = ct.astype(acc, T)
    ct.store(Out, (1, bid_x, head_idx, batch_idx), acc)
    return
end

# ============================================================================
# Public API
# ============================================================================

"""
    flash_attention(Q, K, V; scale=nothing) -> Out

Fused multi-head attention using cuTile with TF32 tensor cores.
Q, K, V: (D, SeqLen, Heads, Batch). Returns: (D, SeqLen, Heads, Batch).
"""
function flash_attention(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
    output::Union{Nothing,CuArray{T,4}}=nothing,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))
    Out = output === nothing ? similar(V, T, D_v, seq_q, heads, batch) : output

    grid_x = cld(seq_q, tile_m)
    grid_y = heads * batch
    grid = (grid_x, grid_y)

    ct.launch(_fmha_kernel, grid, Q, K, V, Out,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n))

    return Out
end

"""
    flash_attention_bias(Q, K, V, bias; scale=nothing) -> Out

Fused multi-head attention with additive bias.
Q, K, V: (D, SeqLen, Heads, Batch).
bias: (SeqK, SeqQ, Heads, Batch) — additive attention bias.
"""
function flash_attention_bias(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4},
    bias::CuArray{<:Real,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
    output::Union{Nothing,CuArray{T,4}}=nothing,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)
    bias_batch = size(bias, 4)

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))
    Out = output === nothing ? similar(V, T, D_v, seq_q, heads, batch) : output
    bias_t = eltype(bias) === T ? bias : T.(bias)

    grid_x = cld(seq_q, tile_m)
    grid_y = heads * batch
    grid = (grid_x, grid_y)

    ct.launch(_fmha_bias_kernel, grid, Q, K, V, bias_t, Out,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n),
        ct.Constant(bias_batch))

    return Out
end

"""
    flash_attention_train(Q, K, V; scale=nothing) -> (Out, Lse)

Training variant: returns output and logsumexp (in log2 space) for backward pass.
"""
function flash_attention_train(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))
    Out = similar(V, T, D_v, seq_q, heads, batch)
    Lse = CUDA.zeros(Float32, 1, seq_q, heads, batch)

    grid_x = cld(seq_q, tile_m)
    grid_y = heads * batch
    grid = (grid_x, grid_y)

    ct.launch(_fmha_lse_kernel, grid, Q, K, V, Out, Lse,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n))

    return Out, Lse
end

"""
    flash_attention_bias_train(Q, K, V, bias; scale=nothing) -> (Out, Lse)

Training variant with additive bias.
"""
function flash_attention_bias_train(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4},
    bias::CuArray{<:Real,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)
    bias_batch = size(bias, 4)

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))
    Out = similar(V, T, D_v, seq_q, heads, batch)
    Lse = CUDA.zeros(Float32, 1, seq_q, heads, batch)
    bias_t = eltype(bias) === T ? bias : T.(bias)

    grid_x = cld(seq_q, tile_m)
    grid_y = heads * batch
    grid = (grid_x, grid_y)

    ct.launch(_fmha_bias_lse_kernel, grid, Q, K, V, bias_t, Out, Lse,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n),
        ct.Constant(bias_batch))

    return Out, Lse
end
