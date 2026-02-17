# GPU utility functions
# TF32 math mode and softmax fix.
# within_gradient is imported from ONIONop in gpu.jl.

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
