#!/usr/bin/env julia
# Test flash attention gradient correctness against reference materialized attention.
# Run with: julia -t 1 test/test_flash_attention_grad.jl
# Must have cuTile available (default, no env overrides).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CUDA
using Flux
using Zygote
using NNlib: softmax, batched_mul, batched_transpose
using Statistics
using Printf

# Load LaProteina (with OnionTile if available)
using LaProteina
using Onion

println("OnionTile available: ", LaProteina._HAS_ONIONTILE)
println("No overrides: ", LaProteina._NO_OVERRIDES)

if !LaProteina._HAS_ONIONTILE
    error("This test requires OnionTile (cuTile kernels). Run without LAPROTEINA_NOCUTILE.")
end

LaProteina.enable_tf32!()

# ============================================================================
# Reference materialized attention (same math as nocutile _attention)
# ============================================================================

"""Reference attention: Q,K,V in [d, L, h, B], bias in [h, L_q, L_k, B]"""
function ref_attention(q, k, v, bias, scale)
    d, L, h, B = size(q)

    # scores = Q^T K: [L_q, L_k, h*B]
    q_flat = reshape(q, d, L, h * B)
    k_flat = reshape(k, d, L, h * B)
    scores = batched_mul(batched_transpose(q_flat), k_flat)  # [L, L, h*B]
    scores = reshape(scores, L, L, h, B)

    # Add bias: permute from [h, L_q, L_k, B] to [L_q, L_k, h, B]
    if !isnothing(bias)
        bias_perm = permutedims(bias, (2, 3, 1, 4))
        scores = scores .* scale .+ bias_perm
    else
        scores = scores .* scale
    end

    # Softmax over key dim
    attn_weights = softmax(scores; dims=2)

    # Output = attn @ V^T
    attn_flat = reshape(attn_weights, L, L, h * B)
    v_flat = reshape(v, d, L, h * B)
    out = batched_mul(attn_flat, batched_transpose(v_flat))  # [L_q, d, h*B]
    out = permutedims(out, (2, 1, 3))
    return reshape(out, d, L, h, B)
end

# ============================================================================
# Test 1: Flash attention forward correctness
# ============================================================================

function test_forward(; d=64, L=32, h=12, B=4)
    println("\n=== Test 1: Forward correctness (d=$d, L=$L, h=$h, B=$B) ===")

    Q = CUDA.randn(Float32, d, L, h, B)
    K = CUDA.randn(Float32, d, L, h, B)
    V = CUDA.randn(Float32, d, L, h, B)
    bias_raw = CUDA.randn(Float32, h, L, L, B) .* 0.1f0  # small bias

    scale = 1f0 / sqrt(Float32(d))

    # Reference
    out_ref = ref_attention(Q, K, V, bias_raw, scale)

    # Flash attention (cuTile)
    # Need to transform inputs to FMHA layout
    # FMHA expects bias as [SeqK, SeqQ, H, B]
    bias4 = permutedims(bias_raw, (3, 2, 1, 4))  # [L_k, L_q, h, B]
    out_flash = LaProteina.flash_attention_bias(Q, K, V, bias4; scale=scale)

    # Compare
    diff = Array(out_flash .- out_ref)
    max_diff = maximum(abs.(diff))
    mean_diff = mean(abs.(diff))
    rel_diff = max_diff / max(maximum(abs.(Array(out_ref))), 1f-8)

    @printf("  Max abs diff:  %.6e\n", max_diff)
    @printf("  Mean abs diff: %.6e\n", mean_diff)
    @printf("  Max rel diff:  %.6e\n", rel_diff)

    if max_diff < 0.01
        println("  PASS ✓")
    else
        println("  FAIL ✗ — forward outputs differ significantly")
    end
    return max_diff < 0.01
end

# ============================================================================
# Test 2: Flash attention backward via Zygote
# ============================================================================

function test_backward(; d=64, L=32, h=12, B=4)
    println("\n=== Test 2: Backward correctness (d=$d, L=$L, h=$h, B=$B) ===")

    Q = CUDA.randn(Float32, d, L, h, B)
    K = CUDA.randn(Float32, d, L, h, B)
    V = CUDA.randn(Float32, d, L, h, B)
    bias_raw = CUDA.randn(Float32, h, L, L, B) .* 0.1f0

    scale = 1f0 / sqrt(Float32(d))

    # Random upstream gradient (for the output)
    dO = CUDA.randn(Float32, d, L, h, B)

    # ------- Reference gradients (materialized attention) -------
    function ref_loss(q, k, v, bias_raw)
        out = ref_attention(q, k, v, bias_raw, scale)
        return sum(out .* dO)
    end

    grads_ref = Zygote.gradient(ref_loss, Q, K, V, bias_raw)
    dQ_ref, dK_ref, dV_ref, dBias_ref = grads_ref

    # ------- Flash attention gradients -------
    function flash_loss(q, k, v, bias_raw)
        bias4 = permutedims(bias_raw, (3, 2, 1, 4))  # [L_k, L_q, h, B]
        out = LaProteina.flash_attention_bias_forward(q, k, v, bias4; scale=scale)
        return sum(out .* dO)
    end

    grads_flash = Zygote.gradient(flash_loss, Q, K, V, bias_raw)
    dQ_flash, dK_flash, dV_flash, dBias_flash = grads_flash

    # ------- Compare -------
    for (name, g_ref, g_flash) in [
        ("dQ", dQ_ref, dQ_flash),
        ("dK", dK_ref, dK_flash),
        ("dV", dV_ref, dV_flash),
        ("dBias", dBias_ref, dBias_flash),
    ]
        if isnothing(g_ref) && isnothing(g_flash)
            println("  $name: both nothing ✓")
            continue
        end
        if isnothing(g_ref) || isnothing(g_flash)
            println("  $name: FAIL — one is nothing, other is not")
            continue
        end
        diff = Array(g_flash .- g_ref)
        max_diff = maximum(abs.(diff))
        ref_max = maximum(abs.(Array(g_ref)))
        rel_diff = max_diff / max(ref_max, 1f-8)
        has_nan = any(isnan.(diff))

        @printf("  %6s: max_abs=%.4e, max_rel=%.4e, ref_max=%.4e",
                name, max_diff, rel_diff, ref_max)
        if has_nan
            print(" NaN!")
        end
        if rel_diff < 0.05
            println(" ✓")
        else
            println(" ✗ FAIL")
        end
    end
end

# ============================================================================
# Test 3: Flash attention backward — direct kernel test (bypass Zygote)
# ============================================================================

function test_backward_direct(; d=64, L=32, h=12, B=4)
    println("\n=== Test 3: Direct backward kernel test (d=$d, L=$L, h=$h, B=$B) ===")

    Q = CUDA.randn(Float32, d, L, h, B)
    K = CUDA.randn(Float32, d, L, h, B)
    V = CUDA.randn(Float32, d, L, h, B)
    bias4 = CUDA.randn(Float32, L, L, h, B) .* 0.1f0  # Already in FMHA layout [SeqK, SeqQ, H, B]

    scale = 1f0 / sqrt(Float32(d))

    # Forward (training variant — saves LSE)
    Out, Lse = LaProteina.flash_attention_bias_train(Q, K, V, bias4; scale=scale)

    # Random upstream gradient
    dO = CUDA.randn(Float32, d, L, h, B)

    # Backward (cuTile kernel)
    dQ, dK, dV, dBias = LaProteina.flash_attention_bias_backward(
        dO, Out, Lse, Q, K, V, bias4; scale=scale)

    # Reference: compute using materialized attention
    # First compute reference output to verify
    q_flat = reshape(Q, d, L, h * B)
    k_flat = reshape(K, d, L, h * B)
    scores = batched_mul(batched_transpose(q_flat), k_flat)  # [L, L, h*B]
    scores = reshape(scores, L, L, h, B)

    # bias4 is [L_k, L_q, h, B], scores are [L_q, L_k, h, B]
    # Need to permute bias to match scores layout
    bias_perm = permutedims(bias4, (2, 1, 3, 4))  # [L_q, L_k, h, B]
    scores_biased = scores .* scale .+ bias_perm

    attn_weights = softmax(scores_biased; dims=2)

    # Reference gradients via Zygote
    function ref_fwd(q, k, v, b4)
        q_f = reshape(q, d, L, h * B)
        k_f = reshape(k, d, L, h * B)
        s = batched_mul(batched_transpose(q_f), k_f)
        s = reshape(s, L, L, h, B)
        bp = permutedims(b4, (2, 1, 3, 4))
        s = s .* scale .+ bp
        w = softmax(s; dims=2)
        w_f = reshape(w, L, L, h * B)
        v_f = reshape(v, d, L, h * B)
        o = batched_mul(w_f, batched_transpose(v_f))
        o = permutedims(o, (2, 1, 3))
        return reshape(o, d, L, h, B)
    end

    function ref_loss(q, k, v, b4)
        return sum(ref_fwd(q, k, v, b4) .* dO)
    end

    grads_ref = Zygote.gradient(ref_loss, Q, K, V, bias4)
    dQ_ref, dK_ref, dV_ref, dBias_ref = grads_ref

    println("  Comparing direct kernel backward vs Zygote reference:")
    all_pass = true
    for (name, g_kernel, g_ref) in [
        ("dQ", dQ, dQ_ref),
        ("dK", dK, dK_ref),
        ("dV", dV, dV_ref),
        ("dBias", dBias, dBias_ref),
    ]
        diff = Array(g_kernel .- g_ref)
        max_diff = maximum(abs.(diff))
        ref_max = maximum(abs.(Array(g_ref)))
        rel_diff = max_diff / max(ref_max, 1f-8)
        has_nan = any(isnan.(diff))
        has_nan_kernel = any(isnan.(Array(g_kernel)))

        @printf("  %6s: max_abs=%.4e, max_rel=%.4e, ref_max=%.4e",
                name, max_diff, rel_diff, ref_max)
        if has_nan_kernel
            print(" [kernel NaN!]")
        end
        if has_nan
            print(" [diff NaN!]")
        end
        if rel_diff < 0.05 && !has_nan
            println(" ✓")
        else
            println(" ✗ FAIL")
            all_pass = false
        end
    end
    return all_pass
end

# ============================================================================
# Test 4: LayerNorm gradient correctness (cuTile rrule vs reference)
# ============================================================================

function test_layernorm_grad(; C=768, L=128, B=4)
    println("\n=== Test 4: LayerNorm grad (C=$C, L=$L, B=$B) ===")

    x = CUDA.randn(Float32, C, L, B)
    w = CUDA.randn(Float32, C)
    b = CUDA.randn(Float32, C)
    eps = 1f-5
    dy = CUDA.randn(Float32, C, L, B)

    # Onion dispatch path (layernorm_first_forward → OnionTile fused kernel)
    function dispatch_loss(x, w, b)
        y = Onion.layernorm_first_forward(x, w, b; eps=eps)
        return sum(y .* dy)
    end

    grads_cutile = Zygote.gradient(dispatch_loss, x, w, b)

    # Reference: standard LayerNorm
    function ref_loss(x, w, b)
        mu = mean(x; dims=1)
        centered = x .- mu
        var_x = mean(centered .^ 2; dims=1)
        inv_std = 1f0 ./ sqrt.(var_x .+ eps)
        normed = centered .* inv_std
        y = normed .* w .+ b
        return sum(y .* dy)
    end

    grads_ref = Zygote.gradient(ref_loss, x, w, b)

    for (name, g_ct, g_ref) in [
        ("dx", grads_cutile[1], grads_ref[1]),
        ("dw", grads_cutile[2], grads_ref[2]),
        ("db", grads_cutile[3], grads_ref[3]),
    ]
        diff = Array(g_ct .- g_ref)
        max_diff = maximum(abs.(diff))
        ref_max = maximum(abs.(Array(g_ref)))
        rel_diff = max_diff / max(ref_max, 1f-8)
        @printf("  %4s: max_abs=%.4e, max_rel=%.4e, ref_max=%.4e", name, max_diff, rel_diff, ref_max)
        if rel_diff < 0.01
            println(" ✓")
        else
            println(" ✗")
        end
    end
end

# ============================================================================
# Test 5: Gradient magnitude analysis (what goes into the optimizer)
# ============================================================================

function test_gradient_magnitude(; d=64, L=64, h=12, B=4)
    println("\n=== Test 5: Gradient magnitude analysis (L=$L) ===")

    Q = CUDA.randn(Float32, d, L, h, B) .* 0.1f0
    K = CUDA.randn(Float32, d, L, h, B) .* 0.1f0
    V = CUDA.randn(Float32, d, L, h, B) .* 0.1f0
    bias_raw = CUDA.randn(Float32, h, L, L, B) .* 0.01f0

    scale = 1f0 / sqrt(Float32(d))

    dO = CUDA.randn(Float32, d, L, h, B) .* 0.01f0

    # Reference
    function ref_loss(q, k, v, br)
        out = ref_attention(q, k, v, br, scale)
        return sum(out .* dO)
    end
    grads_ref = Zygote.gradient(ref_loss, Q, K, V, bias_raw)

    # Flash
    function flash_loss(q, k, v, br)
        b4 = permutedims(br, (3, 2, 1, 4))
        out = LaProteina.flash_attention_bias_forward(q, k, v, b4; scale=scale)
        return sum(out .* dO)
    end
    grads_flash = Zygote.gradient(flash_loss, Q, K, V, bias_raw)

    println("  Gradient norms:")
    for (name, g_ref, g_flash) in [
        ("dQ", grads_ref[1], grads_flash[1]),
        ("dK", grads_ref[2], grads_flash[2]),
        ("dV", grads_ref[3], grads_flash[3]),
        ("dBias", grads_ref[4], grads_flash[4]),
    ]
        norm_ref = sqrt(sum(Array(g_ref).^2))
        norm_flash = sqrt(sum(Array(g_flash).^2))
        ratio = norm_flash / max(norm_ref, 1f-8)
        @printf("  %6s: ref_norm=%.4e, flash_norm=%.4e, ratio=%.4f\n",
                name, norm_ref, norm_flash, ratio)
    end
end

# ============================================================================
# Run all tests
# ============================================================================

println("\n" * "="^60)
println("Flash Attention Gradient Correctness Tests")
println("="^60)

CUDA.allowscalar(false)

test_forward()
test_backward()
test_backward_direct()
test_layernorm_grad()
test_layernorm_grad(C=256, L=64, B=4)  # pair dim
test_gradient_magnitude()

# Also test with larger L to catch tile boundary issues
test_forward(L=100)
test_backward_direct(L=100)
test_backward_direct(L=65)  # Non-power-of-2

println("\n" * "="^60)
println("All tests complete")
println("="^60)
