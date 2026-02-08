# GPU utility functions: TF32, within_gradient, buffer pool
# Phase 1 "quick wins" for LaProteina GPU optimization

# ============================================================================
# Constants (shared with kernel code)
# ============================================================================

const _INV_LOG_2 = Float32(1 / log(2))
const _ConstInt = ct.Constant{Int}

# ============================================================================
# TF32 Math Mode
# ============================================================================

"""
    enable_tf32!()

Enable TF32 math mode for cuBLAS matmuls (~2x Tensor Core speedup on Ampere+).
cuTile kernels use TF32 natively via ct.TFloat32 conversion.
"""
function enable_tf32!()
    CUDA.math_mode!(CUDA.FAST_MATH)
    # Fix: NNlib's cuDNN softmax uses CUDNN_SOFTMAX_FAST when math_mode==FAST_MATH,
    # which skips numerical stability (max subtraction). This causes NaN with large
    # attention scores (>~80). Override to always use ACCURATE algorithm.
    _fix_softmax_algo!()
end

"""
    _fix_softmax_algo!()

Override NNlib's softmaxalgo() to always return CUDNN_SOFTMAX_ACCURATE.
NNlib defaults to CUDNN_SOFTMAX_FAST when CUDA.math_mode()==FAST_MATH,
which is numerically unstable for large attention scores.
"""
function _fix_softmax_algo!()
    ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)
    if ext !== nothing
        # Override the softmaxalgo function in the extension module
        @eval ext softmaxalgo() = cuDNN.CUDNN_SOFTMAX_ACCURATE
        @info "Fixed cuDNN softmax: always using ACCURATE algorithm"
    else
        @warn "NNlibCUDACUDNNExt not loaded, cannot fix softmax algo"
    end
end

# Helper: convert tile to TFloat32 for tensor core acceleration on Float32 data
@inline _to_tf32(tile, ::Type{Float32}) = convert(ct.Tile{ct.TFloat32}, tile)
@inline _to_tf32(tile, ::Type{T}) where T = tile  # no-op for Float16/BFloat16

# ============================================================================
# within_gradient: detect AD tracing context
# ============================================================================

"""
    within_gradient(x) -> Bool

Returns `false` during normal execution and `true` during Zygote AD tracing.
Used to guard in-place operations that are NOT AD-compatible.

During training (within gradient): use allocating out-of-place operations.
During inference (not within gradient): use in-place buffer-pooled operations.
"""
within_gradient(x) = false

function CRC.rrule(::typeof(within_gradient), x)
    return true, _ -> (NoTangent(), NoTangent())
end

# ============================================================================
# Pre-allocated buffer pool for permutedims! (avoids allocation per call)
# ============================================================================

"""
    _perm_buf_pool - Global cache for pre-allocated CuArray buffers.
    Keyed by (slot_id, shape) to allow multiple buffers of the same shape.
"""
const _perm_buf_pool = Dict{Tuple{Int, Tuple}, CuArray}()

function _get_perm_buf(slot::Int, shape::Tuple)
    key = (slot, shape)
    if !haskey(_perm_buf_pool, key)
        _perm_buf_pool[key] = CUDA.zeros(Float32, shape...)
    end
    return _perm_buf_pool[key]
end

# ============================================================================
# Flash attention tile selection
# ============================================================================

"""
    _select_fmha_tiles(D_k, seq_len, heads, batch) -> (tile_m, tile_n)

Auto-select flash attention tile sizes based on problem dimensions.
LaProteina: D_k=64, h=12, B=4, L=100-200 -> default (64, 64).
"""
function _select_fmha_tiles(D_k::Int, seq_len::Int, heads::Int, batch::Int)
    if D_k <= 32 && batch * heads >= 64
        return (32, 64)
    else
        return (64, 64)
    end
end
