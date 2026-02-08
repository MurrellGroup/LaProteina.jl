# ChainRulesCore rrules for LaProteina GPU dispatch hooks.
# Each rrule calls the cuTile train/backward kernels directly.
# Pullbacks eagerly free consumed tensors with CUDA.unsafe_free!.

# ============================================================================
# layernorm_forward rrule (affine)
# ============================================================================

function CRC.rrule(::typeof(layernorm_forward),
                   x::CuArray{T}, w::CuArray{T}, b::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    N = div(length(x), C)
    if ispow2(C) && C >= 512 && N <= 2048
        y, mu_2d, inv_std_2d = fused_layernorm_train(x, w, b, eps)
        function layernorm_pullback_fused(dy_raw)
            dy = unthunk(dy_raw)
            dx, dw, db = fused_layernorm_backward(dy, x, w, mu_2d, inv_std_2d)
            CUDA.unsafe_free!(mu_2d)
            CUDA.unsafe_free!(inv_std_2d)
            return NoTangent(), dx, dw, db, NoTangent()
        end
        return y, layernorm_pullback_fused
    end
    # Large-N or non-pow2 fallback: use custom LN rrule (avoids fused kernel overhead)
    xhat = pytorch_normalise(x; dims=1, eps=eps)
    y = xhat .* w .+ b
    function layernorm_pullback_broadcast(dy_raw)
        dy = unthunk(dy_raw)
        inv_c = Float32(1.0 / C)
        # Standard LayerNorm backward
        dy_w = dy .* w
        c1 = sum(dy_w .* xhat; dims=1) .* inv_c
        c2 = sum(dy_w; dims=1) .* inv_c
        # inv_std = 1/sqrt(var+eps), but we can recover it from xhat
        # Actually, compute from scratch for correctness
        mu = mean(x; dims=1)
        diff = x .- mu
        var_x = mean(diff .* diff; dims=1)
        inv_std = 1f0 ./ sqrt.(var_x .+ eps)
        dx = inv_std .* (dy_w .- xhat .* c1 .- c2)
        batch_dims = ntuple(i -> i + 1, ndims(dy) - 1)
        dw = vec(sum(dy .* xhat; dims=batch_dims))
        db = vec(sum(dy; dims=batch_dims))
        return NoTangent(), dx, T.(dw), T.(db), NoTangent()
    end
    return y, layernorm_pullback_broadcast
end

# ============================================================================
# layernorm_noaffine_forward rrule
# ============================================================================

function CRC.rrule(::typeof(layernorm_noaffine_forward),
                   x::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    N = div(length(x), C)
    if ispow2(C) && C >= 512 && N <= 2048
        y, mu_2d, inv_std_2d = fused_layernorm_noaffine_train(x, eps)
        function noaffine_pullback_fused(dy_raw)
            dy = unthunk(dy_raw)
            dx = fused_layernorm_noaffine_backward(dy, x, mu_2d, inv_std_2d)
            CUDA.unsafe_free!(mu_2d)
            CUDA.unsafe_free!(inv_std_2d)
            return NoTangent(), dx, NoTangent()
        end
        return y, noaffine_pullback_fused
    end
    # Large-N or non-pow2 fallback
    xhat = pytorch_normalise(x; dims=1, eps=eps)
    function noaffine_pullback_broadcast(dy_raw)
        dy = unthunk(dy_raw)
        inv_c = Float32(1.0 / C)
        c1 = sum(dy .* xhat; dims=1) .* inv_c
        c2 = sum(dy; dims=1) .* inv_c
        mu = mean(x; dims=1)
        diff = x .- mu
        var_x = mean(diff .* diff; dims=1)
        inv_std = 1f0 ./ sqrt.(var_x .+ eps)
        dx = inv_std .* (dy .- xhat .* c1 .- c2)
        return NoTangent(), dx, NoTangent()
    end
    return xhat, noaffine_pullback_broadcast
end

# ============================================================================
# flash_attention_forward rrule (no bias)
# ============================================================================

function CRC.rrule(::typeof(flash_attention_forward),
                   Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4};
                   scale=nothing, output=nothing) where T
    qk_scale = something(scale, 1f0 / sqrt(Float32(size(Q, 1))))
    Out, Lse = flash_attention_train(Q, K, V; scale=qk_scale)
    function fmha_pullback(dy_raw)
        dy = unthunk(dy_raw)
        dQ, dK, dV = flash_attention_backward(dy, Out, Lse, Q, K, V; scale=qk_scale)
        CUDA.unsafe_free!(Lse)
        return NoTangent(), dQ, dK, dV
    end
    return Out, fmha_pullback
end

# ============================================================================
# flash_attention_bias_forward rrule (with bias)
# ============================================================================

function CRC.rrule(::typeof(flash_attention_bias_forward),
                   Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4},
                   bias::CuArray{<:Real,4};
                   scale=nothing, output=nothing) where T
    qk_scale = something(scale, 1f0 / sqrt(Float32(size(Q, 1))))
    Out, Lse = flash_attention_bias_train(Q, K, V, bias; scale=qk_scale)
    function fmha_bias_pullback(dy_raw)
        dy = unthunk(dy_raw)
        dQ, dK, dV, dBias = flash_attention_bias_backward(
            dy, Out, Lse, Q, K, V, bias; scale=qk_scale)
        CUDA.unsafe_free!(Lse)
        return NoTangent(), dQ, dK, dV, dBias
    end
    return Out, fmha_bias_pullback
end
