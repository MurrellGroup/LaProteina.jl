# Flash Multi-Head Attention backward kernels (cuTile) for LaProteina
# Adapted from OnionTile.jl. FlashAttention-2 backward with recomputation in log2 space.
# Two-pass: dK/dV kernel + dQ kernel — no atomic operations.

# ============================================================================
# Kernel A: Preprocess — compute D_i = rowsum(dO_i ⊙ O_i) per query position
# ============================================================================

function _fmha_bwd_preprocess_kernel(
    DO::ct.TileArray{T,4},
    O::ct.TileArray{T,4},
    Delta::ct.TileArray{Float32,4},
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    do_tile = ct.load(DO, (1, bid_x, head_idx, batch_idx), (D_V[], TILE_M[], 1, 1))
    do_tile = reshape(do_tile, (D_V[], TILE_M[]))

    o_tile = ct.load(O, (1, bid_x, head_idx, batch_idx), (D_V[], TILE_M[], 1, 1))
    o_tile = reshape(o_tile, (D_V[], TILE_M[]))

    d_i = sum(do_tile .* o_tile, dims=1)
    d_i = reshape(d_i, (1, TILE_M[], 1, 1))
    ct.store(Delta, (1, bid_x, head_idx, batch_idx), d_i)
    return
end

# ============================================================================
# Kernel B: dK/dV — one thread block per KV tile, iterates over Q tiles
# ============================================================================

function _fmha_bwd_dkdv_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    DO::ct.TileArray{T,4},
    Lse::ct.TileArray{Float32,4},
    Delta::ct.TileArray{Float32,4},
    DK::ct.TileArray{T,4},
    DV::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
    Q_TILES::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)
    qk_scale_f32 = Float32(qk_scale)
    k_seqlen = K.sizes[2]

    k_tile = ct.load(K, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1))
    k_tile = reshape(k_tile, (D_K[], TILE_N[]))
    k_t = transpose(k_tile)

    v_tile = ct.load(V, (1, bid_x, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1))
    v_tile = reshape(v_tile, (D_V[], TILE_N[]))
    v_t = transpose(v_tile)

    dk_acc = ct.zeros((D_K[], TILE_N[]), Float32)
    dv_acc = ct.zeros((D_V[], TILE_N[]), Float32)

    i = Int32(1)
    while i <= Q_TILES[]
        q_tile = ct.load(Q, (1, i, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
        q_tile = reshape(q_tile, (D_K[], TILE_M[]))

        do_tile = ct.load(DO, (1, i, head_idx, batch_idx), (D_V[], TILE_M[], 1, 1))
        do_tile = reshape(do_tile, (D_V[], TILE_M[]))

        lse_tile = ct.load(Lse, (1, i, head_idx, batch_idx), (1, TILE_M[], 1, 1))
        lse_tile = reshape(lse_tile, (1, TILE_M[]))

        delta_tile = ct.load(Delta, (1, i, head_idx, batch_idx), (1, TILE_M[], 1, 1))
        delta_tile = reshape(delta_tile, (1, TILE_M[]))

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k_t, T), _to_tf32(q_tile, T), qk)
        else
            qk = ct.muladd(k_t, q_tile, qk)
        end

        if bid_x == cld(k_seqlen, TILE_N[])
            k_idx = (bid_x - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        p = exp2.(qk * qk_scale_log2 .- lse_tile)

        p_t = transpose(p)
        p_t_cast = ct.astype(p_t, T)
        if T === Float32
            dv_acc = ct.muladd(_to_tf32(do_tile, T), _to_tf32(p_t_cast, T), dv_acc)
        else
            dv_acc = ct.muladd(do_tile, p_t_cast, dv_acc)
        end

        dp = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            dp = ct.muladd(_to_tf32(v_t, T), _to_tf32(do_tile, T), dp)
        else
            dp = ct.muladd(v_t, do_tile, dp)
        end

        ds = p .* (dp .- delta_tile)

        ds_t = transpose(ds)
        ds_t_cast = ct.astype(ds_t, T)
        if T === Float32
            dk_acc = ct.muladd(_to_tf32(q_tile, T), _to_tf32(ds_t_cast, T), dk_acc)
        else
            dk_acc = ct.muladd(q_tile, ds_t_cast, dk_acc)
        end

        i += Int32(1)
    end

    dk_acc = dk_acc * qk_scale_f32
    dk_acc = reshape(dk_acc, (D_K[], TILE_N[], 1, 1))
    dk_acc = ct.astype(dk_acc, T)
    ct.store(DK, (1, bid_x, head_idx, batch_idx), dk_acc)

    dv_acc = reshape(dv_acc, (D_V[], TILE_N[], 1, 1))
    dv_acc = ct.astype(dv_acc, T)
    ct.store(DV, (1, bid_x, head_idx, batch_idx), dv_acc)
    return
end

# ============================================================================
# Kernel B (bias variant): dK/dV/dBias
# ============================================================================

function _fmha_bwd_dkdv_bias_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Bias::ct.TileArray{T,4},
    DO::ct.TileArray{T,4},
    Lse::ct.TileArray{Float32,4},
    Delta::ct.TileArray{Float32,4},
    DK::ct.TileArray{T,4},
    DV::ct.TileArray{T,4},
    DBias::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
    Q_TILES::_ConstInt,
    BIAS_BATCH::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])
    bias_batch_idx = BIAS_BATCH[] == Int32(1) ? Int32(1) : batch_idx

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)
    qk_scale_f32 = Float32(qk_scale)
    k_seqlen = K.sizes[2]

    k_tile = ct.load(K, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1))
    k_tile = reshape(k_tile, (D_K[], TILE_N[]))
    k_t = transpose(k_tile)

    v_tile = ct.load(V, (1, bid_x, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1))
    v_tile = reshape(v_tile, (D_V[], TILE_N[]))
    v_t = transpose(v_tile)

    dk_acc = ct.zeros((D_K[], TILE_N[]), Float32)
    dv_acc = ct.zeros((D_V[], TILE_N[]), Float32)

    i = Int32(1)
    while i <= Q_TILES[]
        q_tile = ct.load(Q, (1, i, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
        q_tile = reshape(q_tile, (D_K[], TILE_M[]))

        do_tile = ct.load(DO, (1, i, head_idx, batch_idx), (D_V[], TILE_M[], 1, 1))
        do_tile = reshape(do_tile, (D_V[], TILE_M[]))

        lse_tile = ct.load(Lse, (1, i, head_idx, batch_idx), (1, TILE_M[], 1, 1))
        lse_tile = reshape(lse_tile, (1, TILE_M[]))

        delta_tile = ct.load(Delta, (1, i, head_idx, batch_idx), (1, TILE_M[], 1, 1))
        delta_tile = reshape(delta_tile, (1, TILE_M[]))

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k_t, T), _to_tf32(q_tile, T), qk)
        else
            qk = ct.muladd(k_t, q_tile, qk)
        end

        qk = qk * qk_scale_log2
        bias_tile = ct.load(Bias, (bid_x, i, head_idx, bias_batch_idx), (TILE_N[], TILE_M[], 1, 1))
        bias_tile = reshape(bias_tile, (TILE_N[], TILE_M[]))
        qk = qk .+ bias_tile * Float32(_INV_LOG_2)

        if bid_x == cld(k_seqlen, TILE_N[])
            k_idx = (bid_x - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        p = exp2.(qk .- lse_tile)

        p_t = transpose(p)
        p_t_cast = ct.astype(p_t, T)
        if T === Float32
            dv_acc = ct.muladd(_to_tf32(do_tile, T), _to_tf32(p_t_cast, T), dv_acc)
        else
            dv_acc = ct.muladd(do_tile, p_t_cast, dv_acc)
        end

        dp = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            dp = ct.muladd(_to_tf32(v_t, T), _to_tf32(do_tile, T), dp)
        else
            dp = ct.muladd(v_t, do_tile, dp)
        end

        ds = p .* (dp .- delta_tile)

        ds_out = reshape(ds, (TILE_N[], TILE_M[], 1, 1))
        ds_out = ct.astype(ds_out, T)
        ct.store(DBias, (bid_x, i, head_idx, bias_batch_idx), ds_out)

        ds_t = transpose(ds)
        ds_t_cast = ct.astype(ds_t, T)
        if T === Float32
            dk_acc = ct.muladd(_to_tf32(q_tile, T), _to_tf32(ds_t_cast, T), dk_acc)
        else
            dk_acc = ct.muladd(q_tile, ds_t_cast, dk_acc)
        end

        i += Int32(1)
    end

    dk_acc = dk_acc * qk_scale_f32
    dk_acc = reshape(dk_acc, (D_K[], TILE_N[], 1, 1))
    dk_acc = ct.astype(dk_acc, T)
    ct.store(DK, (1, bid_x, head_idx, batch_idx), dk_acc)

    dv_acc = reshape(dv_acc, (D_V[], TILE_N[], 1, 1))
    dv_acc = ct.astype(dv_acc, T)
    ct.store(DV, (1, bid_x, head_idx, batch_idx), dv_acc)
    return
end

# ============================================================================
# Kernel C: dQ — one thread block per Q tile, iterates over KV tiles
# ============================================================================

function _fmha_bwd_dq_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    DO::ct.TileArray{T,4},
    Lse::ct.TileArray{Float32,4},
    Delta::ct.TileArray{Float32,4},
    DQ::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
    KV_TILES::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)
    qk_scale_f32 = Float32(qk_scale)
    k_seqlen = K.sizes[2]

    q_tile = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q_tile = reshape(q_tile, (D_K[], TILE_M[]))

    do_tile = ct.load(DO, (1, bid_x, head_idx, batch_idx), (D_V[], TILE_M[], 1, 1))
    do_tile = reshape(do_tile, (D_V[], TILE_M[]))

    lse_tile = ct.load(Lse, (1, bid_x, head_idx, batch_idx), (1, TILE_M[], 1, 1))
    lse_tile = reshape(lse_tile, (1, TILE_M[]))

    delta_tile = ct.load(Delta, (1, bid_x, head_idx, batch_idx), (1, TILE_M[], 1, 1))
    delta_tile = reshape(delta_tile, (1, TILE_M[]))

    dq_acc = ct.zeros((D_K[], TILE_M[]), Float32)

    j = Int32(1)
    while j <= KV_TILES[]
        k_tile = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1))
        k_tile = reshape(k_tile, (D_K[], TILE_N[]))
        k_t = transpose(k_tile)

        v_tile = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1))
        v_tile = reshape(v_tile, (D_V[], TILE_N[]))
        v_t = transpose(v_tile)

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k_t, T), _to_tf32(q_tile, T), qk)
        else
            qk = ct.muladd(k_t, q_tile, qk)
        end

        if j == KV_TILES[]
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        p = exp2.(qk * qk_scale_log2 .- lse_tile)

        dp = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            dp = ct.muladd(_to_tf32(v_t, T), _to_tf32(do_tile, T), dp)
        else
            dp = ct.muladd(v_t, do_tile, dp)
        end

        ds = p .* (dp .- delta_tile)

        ds_cast = ct.astype(ds, T)
        if T === Float32
            dq_acc = ct.muladd(_to_tf32(k_tile, T), _to_tf32(ds_cast, T), dq_acc)
        else
            dq_acc = ct.muladd(k_tile, ds_cast, dq_acc)
        end

        j += Int32(1)
    end

    dq_acc = dq_acc * qk_scale_f32
    dq_acc = reshape(dq_acc, (D_K[], TILE_M[], 1, 1))
    dq_acc = ct.astype(dq_acc, T)
    ct.store(DQ, (1, bid_x, head_idx, batch_idx), dq_acc)
    return
end

# ============================================================================
# Kernel C (bias variant): dQ
# ============================================================================

function _fmha_bwd_dq_bias_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Bias::ct.TileArray{T,4},
    DO::ct.TileArray{T,4},
    Lse::ct.TileArray{Float32,4},
    Delta::ct.TileArray{Float32,4},
    DQ::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
    KV_TILES::_ConstInt,
    BIAS_BATCH::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])
    bias_batch_idx = BIAS_BATCH[] == Int32(1) ? Int32(1) : batch_idx

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)
    qk_scale_f32 = Float32(qk_scale)
    k_seqlen = K.sizes[2]

    q_tile = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q_tile = reshape(q_tile, (D_K[], TILE_M[]))

    do_tile = ct.load(DO, (1, bid_x, head_idx, batch_idx), (D_V[], TILE_M[], 1, 1))
    do_tile = reshape(do_tile, (D_V[], TILE_M[]))

    lse_tile = ct.load(Lse, (1, bid_x, head_idx, batch_idx), (1, TILE_M[], 1, 1))
    lse_tile = reshape(lse_tile, (1, TILE_M[]))

    delta_tile = ct.load(Delta, (1, bid_x, head_idx, batch_idx), (1, TILE_M[], 1, 1))
    delta_tile = reshape(delta_tile, (1, TILE_M[]))

    dq_acc = ct.zeros((D_K[], TILE_M[]), Float32)

    j = Int32(1)
    while j <= KV_TILES[]
        k_tile = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1))
        k_tile = reshape(k_tile, (D_K[], TILE_N[]))
        k_t = transpose(k_tile)

        v_tile = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1))
        v_tile = reshape(v_tile, (D_V[], TILE_N[]))
        v_t = transpose(v_tile)

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k_t, T), _to_tf32(q_tile, T), qk)
        else
            qk = ct.muladd(k_t, q_tile, qk)
        end

        qk = qk * qk_scale_log2
        bias_tile = ct.load(Bias, (j, bid_x, head_idx, bias_batch_idx), (TILE_N[], TILE_M[], 1, 1))
        bias_tile = reshape(bias_tile, (TILE_N[], TILE_M[]))
        qk = qk .+ bias_tile * Float32(_INV_LOG_2)

        if j == KV_TILES[]
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        p = exp2.(qk .- lse_tile)

        dp = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            dp = ct.muladd(_to_tf32(v_t, T), _to_tf32(do_tile, T), dp)
        else
            dp = ct.muladd(v_t, do_tile, dp)
        end

        ds = p .* (dp .- delta_tile)

        ds_cast = ct.astype(ds, T)
        if T === Float32
            dq_acc = ct.muladd(_to_tf32(k_tile, T), _to_tf32(ds_cast, T), dq_acc)
        else
            dq_acc = ct.muladd(k_tile, ds_cast, dq_acc)
        end

        j += Int32(1)
    end

    dq_acc = dq_acc * qk_scale_f32
    dq_acc = reshape(dq_acc, (D_K[], TILE_M[], 1, 1))
    dq_acc = ct.astype(dq_acc, T)
    ct.store(DQ, (1, bid_x, head_idx, batch_idx), dq_acc)
    return
end

# ============================================================================
# Public API: backward passes
# ============================================================================

"""
    flash_attention_backward(dO, Out, Lse, Q, K, V; scale) -> (dQ, dK, dV)

Backward pass for flash attention. All inputs in (D, SeqLen, H, B) layout.
"""
function flash_attention_backward(
    dO::CuArray{T,4}, Out::CuArray{T,4}, Lse::CuArray{Float32,4},
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)
    seq_kv = size(K, 2)

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))

    q_tiles = cld(seq_q, tile_m)
    kv_tiles = cld(seq_kv, tile_n)
    grid_y = heads * batch

    Delta = similar(Q, Float32, 1, seq_q, heads, batch)
    ct.launch(_fmha_bwd_preprocess_kernel, (q_tiles, grid_y),
        dO, Out, Delta,
        ct.Constant(D_v), ct.Constant(heads), ct.Constant(tile_m))

    dK = similar(K)
    dV = similar(V)
    ct.launch(_fmha_bwd_dkdv_kernel, (kv_tiles, grid_y),
        Q, K, V, dO, Lse, Delta, dK, dV,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n), ct.Constant(q_tiles))

    dQ = similar(Q)
    ct.launch(_fmha_bwd_dq_kernel, (q_tiles, grid_y),
        Q, K, V, dO, Lse, Delta, dQ,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n), ct.Constant(kv_tiles))

    CUDA.unsafe_free!(Delta)
    return dQ, dK, dV
end

"""
    flash_attention_bias_backward(dO, Out, Lse, Q, K, V, bias; scale) -> (dQ, dK, dV, dBias)

Backward pass for flash attention with additive bias.
"""
function flash_attention_bias_backward(
    dO::CuArray{T,4}, Out::CuArray{T,4}, Lse::CuArray{Float32,4},
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4},
    bias::CuArray{<:Real,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)
    seq_kv = size(K, 2)
    bias_batch = size(bias, 4)

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))
    bias_t = eltype(bias) === T ? bias : T.(bias)

    q_tiles = cld(seq_q, tile_m)
    kv_tiles = cld(seq_kv, tile_n)
    grid_y = heads * batch

    Delta = similar(Q, Float32, 1, seq_q, heads, batch)
    ct.launch(_fmha_bwd_preprocess_kernel, (q_tiles, grid_y),
        dO, Out, Delta,
        ct.Constant(D_v), ct.Constant(heads), ct.Constant(tile_m))

    dK = similar(K)
    dV = similar(V)
    dBias = similar(bias_t, T, seq_kv, seq_q, heads, bias_batch)
    ct.launch(_fmha_bwd_dkdv_bias_kernel, (kv_tiles, grid_y),
        Q, K, V, bias_t, dO, Lse, Delta, dK, dV, dBias,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n), ct.Constant(q_tiles),
        ct.Constant(bias_batch))

    dQ = similar(Q)
    ct.launch(_fmha_bwd_dq_bias_kernel, (q_tiles, grid_y),
        Q, K, V, bias_t, dO, Lse, Delta, dQ,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n), ct.Constant(kv_tiles),
        ct.Constant(bias_batch))

    CUDA.unsafe_free!(Delta)
    return dQ, dK, dV, dBias
end
