# Fused LayerNorm kernels (cuTile) for LaProteina
# Adapted from OnionTile.jl — same math: (x - μ) / sqrt(σ² + ε) * w + b
# Three variants: forward inference, forward training (saves stats), backward

# ============================================================================
# Forward kernel (inference) — affine: y = (x - μ) * inv_std * w + b
# ============================================================================

function _layernorm_kernel(
    X::ct.TileArray{T,2},
    Out::ct.TileArray{T,2},
    W::ct.TileArray{T,2},
    B_arr::ct.TileArray{T,2},
    eps::AbstractFloat,
    C_DIM::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid = ct.bid(1)

    x = ct.load(X, (1, bid), (C_DIM[], TILE_N[]))
    w = ct.load(W, (1, 1), (C_DIM[], 1))
    b = ct.load(B_arr, (1, 1), (C_DIM[], 1))

    inv_c = Float32(1.0 / C_DIM[])
    mu = sum(x, dims=1) * inv_c
    diff = x .- mu
    sigma2 = sum(diff .* diff, dims=1) * inv_c
    inv_std = 1f0 ./ sqrt.(sigma2 .+ Float32(eps))

    out = @. diff * inv_std * w + b
    ct.store(Out, (1, bid), out)
    return
end

# ============================================================================
# Forward kernel (inference) — non-affine: y = (x - μ) * inv_std
# ============================================================================

function _layernorm_noaffine_kernel(
    X::ct.TileArray{T,2},
    Out::ct.TileArray{T,2},
    eps::AbstractFloat,
    C_DIM::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid = ct.bid(1)

    x = ct.load(X, (1, bid), (C_DIM[], TILE_N[]))

    inv_c = Float32(1.0 / C_DIM[])
    mu = sum(x, dims=1) * inv_c
    diff = x .- mu
    sigma2 = sum(diff .* diff, dims=1) * inv_c
    inv_std = 1f0 ./ sqrt.(sigma2 .+ Float32(eps))

    out = diff .* inv_std
    ct.store(Out, (1, bid), out)
    return
end

# ============================================================================
# Forward kernel (training) — saves mu and inv_std for backward pass
# ============================================================================

function _layernorm_train_kernel(
    X::ct.TileArray{T,2},
    Out::ct.TileArray{T,2},
    W::ct.TileArray{T,2},
    B_arr::ct.TileArray{T,2},
    Mu_out::ct.TileArray{Float32,2},
    InvStd_out::ct.TileArray{Float32,2},
    eps::AbstractFloat,
    C_DIM::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid = ct.bid(1)

    x = ct.load(X, (1, bid), (C_DIM[], TILE_N[]))
    w = ct.load(W, (1, 1), (C_DIM[], 1))
    b = ct.load(B_arr, (1, 1), (C_DIM[], 1))

    inv_c = Float32(1.0 / C_DIM[])
    mu = sum(x, dims=1) * inv_c
    diff = x .- mu
    sigma2 = sum(diff .* diff, dims=1) * inv_c
    inv_std = 1f0 ./ sqrt.(sigma2 .+ Float32(eps))

    out = @. diff * inv_std * w + b
    ct.store(Out, (1, bid), out)
    ct.store(Mu_out, (1, bid), mu)
    ct.store(InvStd_out, (1, bid), inv_std)
    return
end

# ============================================================================
# Non-affine training kernel — saves mu and inv_std
# ============================================================================

function _layernorm_noaffine_train_kernel(
    X::ct.TileArray{T,2},
    Out::ct.TileArray{T,2},
    Mu_out::ct.TileArray{Float32,2},
    InvStd_out::ct.TileArray{Float32,2},
    eps::AbstractFloat,
    C_DIM::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid = ct.bid(1)

    x = ct.load(X, (1, bid), (C_DIM[], TILE_N[]))

    inv_c = Float32(1.0 / C_DIM[])
    mu = sum(x, dims=1) * inv_c
    diff = x .- mu
    sigma2 = sum(diff .* diff, dims=1) * inv_c
    inv_std = 1f0 ./ sqrt.(sigma2 .+ Float32(eps))

    out = diff .* inv_std
    ct.store(Out, (1, bid), out)
    ct.store(Mu_out, (1, bid), mu)
    ct.store(InvStd_out, (1, bid), inv_std)
    return
end

# ============================================================================
# Backward kernel — affine
# ============================================================================

function _layernorm_bwd_kernel(
    DY::ct.TileArray{T,2},
    X::ct.TileArray{T,2},
    W::ct.TileArray{T,2},
    Mu::ct.TileArray{Float32,2},
    InvStd::ct.TileArray{Float32,2},
    DX::ct.TileArray{T,2},
    DW_partial::ct.TileArray{Float32,2},
    DB_partial::ct.TileArray{Float32,2},
    C_DIM::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid = ct.bid(1)

    dy = ct.load(DY, (1, bid), (C_DIM[], TILE_N[]))
    x = ct.load(X, (1, bid), (C_DIM[], TILE_N[]))
    w = ct.load(W, (1, 1), (C_DIM[], 1))
    mu = ct.load(Mu, (1, bid), (1, TILE_N[]))
    inv_std = ct.load(InvStd, (1, bid), (1, TILE_N[]))

    inv_c = Float32(1.0 / C_DIM[])

    xhat = (x .- mu) .* inv_std
    dy_w = dy .* w

    c1 = sum(dy_w .* xhat, dims=1) * inv_c
    c2 = sum(dy_w, dims=1) * inv_c
    dx = inv_std .* (dy_w .- xhat .* c1 .- c2)

    ct.store(DX, (1, bid), dx)

    dw_tile = sum(dy .* xhat, dims=2)
    db_tile = sum(dy, dims=2)
    ct.store(DW_partial, (1, bid), dw_tile)
    ct.store(DB_partial, (1, bid), db_tile)
    return
end

# ============================================================================
# Backward kernel — non-affine (no dw, db)
# ============================================================================

function _layernorm_noaffine_bwd_kernel(
    DY::ct.TileArray{T,2},
    X::ct.TileArray{T,2},
    Mu::ct.TileArray{Float32,2},
    InvStd::ct.TileArray{Float32,2},
    DX::ct.TileArray{T,2},
    C_DIM::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid = ct.bid(1)

    dy = ct.load(DY, (1, bid), (C_DIM[], TILE_N[]))
    x = ct.load(X, (1, bid), (C_DIM[], TILE_N[]))
    mu = ct.load(Mu, (1, bid), (1, TILE_N[]))
    inv_std = ct.load(InvStd, (1, bid), (1, TILE_N[]))

    inv_c = Float32(1.0 / C_DIM[])

    xhat = (x .- mu) .* inv_std

    c1 = sum(dy .* xhat, dims=1) * inv_c
    c2 = sum(dy, dims=1) * inv_c
    dx = inv_std .* (dy .- xhat .* c1 .- c2)

    ct.store(DX, (1, bid), dx)
    return
end

# ============================================================================
# Public API: affine LayerNorm
# ============================================================================

"""
    fused_layernorm(x, w, b, eps) -> y

Fused LayerNorm: 1 kernel launch, 0 intermediate allocations.
Replaces ~9 kernel launches of the broadcast implementation.
"""
function fused_layernorm(x::CuArray{T}, w_vec::CuArray{T}, b_vec::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    orig_size = size(x)
    N = div(length(x), C)

    x2 = reshape(x, C, N)
    out2 = similar(x2)
    w2 = reshape(w_vec, C, 1)
    b2 = reshape(b_vec, C, 1)

    tile_n = C <= 256 ? 64 : (C <= 1024 ? 16 : 8)
    grid = (cld(N, tile_n),)

    ct.launch(_layernorm_kernel, grid, x2, out2, w2, b2,
        eps, ct.Constant(C), ct.Constant(tile_n))

    return reshape(out2, orig_size)
end

"""
    fused_layernorm_train(x, w, b, eps) -> (y, mu_2d, inv_std_2d)

Training variant: returns output plus saved statistics for backward.
"""
function fused_layernorm_train(x::CuArray{T}, w_vec::CuArray{T}, b_vec::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    orig_size = size(x)
    N = div(length(x), C)

    x2 = reshape(x, C, N)
    out2 = similar(x2)
    mu_2d = similar(x2, Float32, 1, N)
    inv_std_2d = similar(x2, Float32, 1, N)
    w2 = reshape(w_vec, C, 1)
    b2 = reshape(b_vec, C, 1)

    tile_n = C <= 256 ? 64 : (C <= 1024 ? 16 : 8)
    grid = (cld(N, tile_n),)

    ct.launch(_layernorm_train_kernel, grid, x2, out2, w2, b2, mu_2d, inv_std_2d,
        eps, ct.Constant(C), ct.Constant(tile_n))

    return reshape(out2, orig_size), mu_2d, inv_std_2d
end

"""
    fused_layernorm_backward(dy, x, w, mu, inv_std) -> (dx, dw, db)

Backward pass for fused LayerNorm.
"""
function fused_layernorm_backward(dy::CuArray{T}, x::CuArray{T}, w_vec::CuArray{T},
                                   mu_2d::CuArray{Float32}, inv_std_2d::CuArray{Float32}) where T
    C = size(x, 1)
    orig_size = size(x)
    N = div(length(x), C)

    dy2 = reshape(dy, C, N)
    x2 = reshape(x, C, N)
    w2 = reshape(w_vec, C, 1)
    dx2 = similar(x2)

    tile_n = C <= 128 ? 32 : (C <= 512 ? 16 : 8)
    n_tiles = cld(N, tile_n)
    dw_partial = similar(x2, Float32, C, n_tiles)
    db_partial = similar(x2, Float32, C, n_tiles)
    grid = (n_tiles,)

    ct.launch(_layernorm_bwd_kernel, grid, dy2, x2, w2, mu_2d, inv_std_2d,
        dx2, dw_partial, db_partial,
        ct.Constant(C), ct.Constant(tile_n))

    dw = vec(sum(dw_partial; dims=2))
    db = vec(sum(db_partial; dims=2))
    CUDA.unsafe_free!(dw_partial)
    CUDA.unsafe_free!(db_partial)

    return reshape(dx2, orig_size), T.(dw), T.(db)
end

# ============================================================================
# Public API: non-affine LayerNorm
# ============================================================================

"""
    fused_layernorm_noaffine(x, eps) -> y

Non-affine fused LayerNorm (no w, b). Used by ProteINAAdaLN.
"""
function fused_layernorm_noaffine(x::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    orig_size = size(x)
    N = div(length(x), C)

    x2 = reshape(x, C, N)
    out2 = similar(x2)

    tile_n = C <= 256 ? 64 : (C <= 1024 ? 16 : 8)
    grid = (cld(N, tile_n),)

    ct.launch(_layernorm_noaffine_kernel, grid, x2, out2,
        eps, ct.Constant(C), ct.Constant(tile_n))

    return reshape(out2, orig_size)
end

"""
    fused_layernorm_noaffine_train(x, eps) -> (y, mu_2d, inv_std_2d)

Training variant of non-affine LayerNorm.
"""
function fused_layernorm_noaffine_train(x::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    orig_size = size(x)
    N = div(length(x), C)

    x2 = reshape(x, C, N)
    out2 = similar(x2)
    mu_2d = similar(x2, Float32, 1, N)
    inv_std_2d = similar(x2, Float32, 1, N)

    tile_n = C <= 256 ? 64 : (C <= 1024 ? 16 : 8)
    grid = (cld(N, tile_n),)

    ct.launch(_layernorm_noaffine_train_kernel, grid, x2, out2, mu_2d, inv_std_2d,
        eps, ct.Constant(C), ct.Constant(tile_n))

    return reshape(out2, orig_size), mu_2d, inv_std_2d
end

"""
    fused_layernorm_noaffine_backward(dy, x, mu, inv_std) -> dx

Backward for non-affine LayerNorm.
"""
function fused_layernorm_noaffine_backward(dy::CuArray{T}, x::CuArray{T},
                                            mu_2d::CuArray{Float32}, inv_std_2d::CuArray{Float32}) where T
    C = size(x, 1)
    orig_size = size(x)
    N = div(length(x), C)

    dy2 = reshape(dy, C, N)
    x2 = reshape(x, C, N)
    dx2 = similar(x2)

    tile_n = C <= 128 ? 32 : (C <= 512 ? 16 : 8)
    grid = (cld(N, tile_n),)

    ct.launch(_layernorm_noaffine_bwd_kernel, grid, dy2, x2, mu_2d, inv_std_2d,
        dx2, ct.Constant(C), ct.Constant(tile_n))

    return reshape(dx2, orig_size)
end
