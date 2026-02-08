#!/usr/bin/env julia
# Gradient correctness test for GPU optimization overrides.
# Verifies that CuArray layer overrides produce the same gradients as CPU.
#
# This is critical: if gradients are wrong, training will diverge.
#
# Usage: julia test/test_gpu_gradients.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Statistics
using Printf

LaProteina.enable_tf32!()

println("="^70)
println("Gradient Correctness Tests")
println("="^70)

function maxabsdiff(a, b)
    return maximum(abs.(Array(a) .- Array(b)))
end

function test_gradient(name, f_cpu, f_gpu, args_cpu, args_gpu; tol=1e-2)
    # CPU gradient
    g_cpu = Zygote.gradient(f_cpu, args_cpu...)
    # GPU gradient
    g_gpu = Zygote.gradient(f_gpu, args_gpu...)

    # Compare
    all_ok = true
    for (i, (gc, gg)) in enumerate(zip(g_cpu, g_gpu))
        if gc === nothing && gg === nothing
            continue
        end
        if gc === nothing || gg === nothing
            println("  $name arg $i: one gradient is nothing")
            all_ok = false
            continue
        end
        diff = maxabsdiff(gc, gg)
        ok = diff < tol
        status = ok ? "PASS" : "FAIL"
        @printf("  %s arg %d: max diff = %.6f [%s]\n", name, i, diff, status)
        if !ok
            all_ok = false
        end
    end
    return all_ok
end

# ============================================================================
# Test 1: PyTorchLayerNorm (affine)
# ============================================================================
println("\n1. PyTorchLayerNorm (affine)")

ln_cpu = PyTorchLayerNorm(64)
ln_gpu = deepcopy(ln_cpu) |> gpu

x_cpu = randn(Float32, 64, 16, 2)
x_gpu = CUDA.CuArray(x_cpu)

test_gradient("LN_affine",
    x -> sum(ln_cpu(x)),
    x -> sum(ln_gpu(x)),
    (x_cpu,), (x_gpu,);
    tol=1e-4)

# ============================================================================
# Test 2: PyTorchLayerNorm (non-affine)
# ============================================================================
println("\n2. PyTorchLayerNorm (non-affine)")

ln_na_cpu = PyTorchLayerNorm(64; affine=false)
ln_na_gpu = deepcopy(ln_na_cpu) |> gpu

test_gradient("LN_noaffine",
    x -> sum(ln_na_cpu(x)),
    x -> sum(ln_na_gpu(x)),
    (x_cpu,), (x_gpu,);
    tol=1e-4)

# ============================================================================
# Test 3: PairBiasAttention
# ============================================================================
println("\n3. PairBiasAttention")

attn_cpu = PairBiasAttention(64, 4; pair_dim=32, qk_ln=true)
attn_gpu = deepcopy(attn_cpu) |> gpu

x_attn_cpu = randn(Float32, 64, 16, 2)
pair_cpu = randn(Float32, 32, 16, 16, 2)
mask_cpu = ones(Float32, 16, 2)

x_attn_gpu = CUDA.CuArray(x_attn_cpu)
pair_gpu = CUDA.CuArray(pair_cpu)
mask_gpu = CUDA.CuArray(mask_cpu)

test_gradient("PairBiasAttn",
    x -> sum(attn_cpu(x, pair_cpu, mask_cpu)),
    x -> sum(attn_gpu(x, pair_gpu, mask_gpu)),
    (x_attn_cpu,), (x_attn_gpu,);
    tol=0.05)  # TF32 tolerance

# ============================================================================
# Test 4: SwiGLU
# ============================================================================
println("\n4. SwiGLU")

swiglu = SwiGLU()
x_sg_cpu = randn(Float32, 128, 16, 2)
x_sg_gpu = CUDA.CuArray(x_sg_cpu)

test_gradient("SwiGLU",
    x -> sum(swiglu(x)),
    x -> sum(swiglu(x)),
    (x_sg_cpu,), (x_sg_gpu,);
    tol=1e-4)

# ============================================================================
# Test 5: SwiGLUTransition
# ============================================================================
println("\n5. SwiGLUTransition")

tr_cpu = SwiGLUTransition(64; expansion_factor=4)
tr_gpu = deepcopy(tr_cpu) |> gpu

x_tr_cpu = randn(Float32, 64, 16, 2)
x_tr_gpu = CUDA.CuArray(x_tr_cpu)
mask_tr_cpu = ones(Float32, 16, 2)
mask_tr_gpu = CUDA.CuArray(mask_tr_cpu)

test_gradient("SwiGLUTransition",
    x -> sum(tr_cpu(x, mask_tr_cpu)),
    x -> sum(tr_gpu(x, mask_tr_gpu)),
    (x_tr_cpu,), (x_tr_gpu,);
    tol=0.01)

# ============================================================================
# Test 6: ProteINAAdaLN
# ============================================================================
println("\n6. ProteINAAdaLN")

adaln_cpu = ProteINAAdaLN(64, 32)
adaln_gpu = deepcopy(adaln_cpu) |> gpu

x_adaln_cpu = randn(Float32, 64, 16, 2)
cond_cpu = randn(Float32, 32, 16, 2)
x_adaln_gpu = CUDA.CuArray(x_adaln_cpu)
cond_gpu = CUDA.CuArray(cond_cpu)

test_gradient("ProteINAAdaLN",
    x -> sum(adaln_cpu(x, cond_cpu, mask_tr_cpu)),
    x -> sum(adaln_gpu(x, cond_gpu, mask_tr_gpu)),
    (x_adaln_cpu,), (x_adaln_gpu,);
    tol=0.01)

# ============================================================================
# Test 7: TransformerBlock
# ============================================================================
println("\n7. TransformerBlock")

block_cpu = TransformerBlock(dim_token=64, dim_pair=32, n_heads=4, dim_cond=32, qk_ln=true)
block_gpu = deepcopy(block_cpu) |> gpu

x_blk_cpu = randn(Float32, 64, 16, 2)
pair_blk_cpu = randn(Float32, 32, 16, 16, 2)
cond_blk_cpu = randn(Float32, 32, 16, 2)
mask_blk_cpu = ones(Float32, 16, 2)

x_blk_gpu = CUDA.CuArray(x_blk_cpu)
pair_blk_gpu = CUDA.CuArray(pair_blk_cpu)
cond_blk_gpu = CUDA.CuArray(cond_blk_cpu)
mask_blk_gpu = CUDA.CuArray(mask_blk_cpu)

test_gradient("TransformerBlock",
    x -> sum(block_cpu(x, pair_blk_cpu, cond_blk_cpu, mask_blk_cpu)),
    x -> sum(block_gpu(x, pair_blk_gpu, cond_blk_gpu, mask_blk_gpu)),
    (x_blk_cpu,), (x_blk_gpu,);
    tol=0.1)  # Larger tolerance for full block with TF32

# ============================================================================
# Test 8: within_gradient behavior
# ============================================================================
println("\n8. within_gradient behavior")

# Should be false outside gradient
println("  Outside gradient: ", LaProteina.within_gradient(1.0), " (expected: false)")

# Should be true inside gradient
grad_val = Zygote.gradient(x -> begin
    wg = LaProteina.within_gradient(x)
    wg ? x * 2.0f0 : x * 1.0f0
end, 1.0f0)
# If within_gradient works, grad should be 2.0 (not 1.0)
println("  Inside gradient: grad = ", grad_val[1], " (expected: 2.0)")

println("\n" * "="^70)
println("Gradient tests complete.")
println("="^70)
