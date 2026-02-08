# GPU utility functions (no cuTile dependency)
# TF32 math mode, within_gradient, buffer pool

# ============================================================================
# TF32 Math Mode
# ============================================================================

"""
    enable_tf32!()

Enable TF32 math mode for cuBLAS matmuls (~2x Tensor Core speedup on Ampere+).
"""
function enable_tf32!()
    CUDA.math_mode!(CUDA.FAST_MATH)
    # Fix: NNlib's cuDNN softmax uses CUDNN_SOFTMAX_FAST when math_mode==FAST_MATH,
    # which skips numerical stability (max subtraction). This causes NaN with large
    # attention scores (>~80). Override to always use ACCURATE algorithm.
    _fix_softmax_algo!()
end

function _fix_softmax_algo!()
    ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)
    if ext !== nothing
        @eval ext softmaxalgo() = cuDNN.CUDNN_SOFTMAX_ACCURATE
        @info "Fixed cuDNN softmax: always using ACCURATE algorithm"
    else
        @warn "NNlibCUDACUDNNExt not loaded, cannot fix softmax algo"
    end
end

# ============================================================================
# within_gradient: detect AD tracing context
# ============================================================================

"""
    within_gradient(x) -> Bool

Returns `false` during normal execution and `true` during Zygote AD tracing.
Used to guard in-place operations that are NOT AD-compatible.
"""
within_gradient(x) = false

function CRC.rrule(::typeof(within_gradient), x)
    return true, _ -> (NoTangent(), NoTangent())
end

# ============================================================================
# Pre-allocated buffer pool
# ============================================================================

const _perm_buf_pool = Dict{Tuple{Int, Tuple}, CuArray}()

function _get_perm_buf(slot::Int, shape::Tuple)
    key = (slot, shape)
    if !haskey(_perm_buf_pool, key)
        _perm_buf_pool[key] = CUDA.zeros(Float32, shape...)
    end
    return _perm_buf_pool[key]
end
