# CuArray dispatch hooks for LaProteina GPU optimization.
# These functions route CuArray inputs to fused cuTile kernels.
# The rrules in rrules.jl provide AD backward rules for these dispatch functions.

# ============================================================================
# LayerNorm dispatch (affine)
# ============================================================================

"""
    layernorm_forward(x::CuArray, w, b, eps) -> y

Dispatch hook for affine LayerNorm. Routes to fused cuTile kernel when
dimension is power-of-2, falls back to broadcast implementation otherwise.
"""
function layernorm_forward(x::CuArray{T}, w::CuArray{T}, b::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    N = div(length(x), C)
    # Fused kernel competitive only for C>=512, small N
    # (C=256 fused is 30x slower than broadcast on GB10; C=768 is ~same)
    if ispow2(C) && C >= 512 && N <= 2048
        return fused_layernorm(x, w, b, eps)
    end
    # Fallback: standard broadcast implementation (still on GPU)
    return pytorch_normalise(x; dims=1, eps=eps) .* w .+ b
end

# ============================================================================
# LayerNorm dispatch (non-affine)
# ============================================================================

"""
    layernorm_noaffine_forward(x::CuArray, eps) -> y

Dispatch hook for non-affine LayerNorm.
"""
function layernorm_noaffine_forward(x::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    N = div(length(x), C)
    if ispow2(C) && C >= 512 && N <= 2048
        return fused_layernorm_noaffine(x, eps)
    end
    return pytorch_normalise(x; dims=1, eps=eps)
end

# ============================================================================
# Flash attention dispatch (no bias)
# ============================================================================

"""
    flash_attention_forward(Q::CuArray, K, V; scale, output) -> Out

Dispatch hook for flash attention without bias.
"""
function flash_attention_forward(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4};
    scale=nothing, output=nothing,
) where T
    return flash_attention(Q, K, V; scale=scale, output=output)
end

# ============================================================================
# Flash attention dispatch (with bias)
# ============================================================================

"""
    flash_attention_bias_forward(Q::CuArray, K, V, bias; scale, output) -> Out

Dispatch hook for flash attention with additive bias.
Q, K, V: (D, SeqLen, H, B). bias: (SeqK, SeqQ, H, B).
"""
function flash_attention_bias_forward(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4},
    bias::CuArray{<:Real,4};
    scale=nothing, output=nothing,
) where T
    return flash_attention_bias(Q, K, V, bias; scale=scale, output=output)
end
