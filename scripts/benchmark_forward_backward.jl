#!/usr/bin/env julia
# Benchmark forward and backward passes for LaProteina ScoreNetwork layers.
# Times each layer individually and the full transformer block.
#
# Usage: julia scripts/benchmark_forward_backward.jl
#
# Outputs timings in ms with GPU memory allocated.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Statistics
using Printf

# Enable TF32 for tensor core acceleration
LaProteina.enable_tf32!()
println("TF32 math mode: ", CUDA.math_mode())
println("GPU: ", CUDA.name(CUDA.device()))
println("Memory: ", @sprintf("%.2f GB", CUDA.available_memory() / 1e9))
println("cuTile available: ", LaProteina._HAS_CUTILE)

# ============================================================================
# Helper function for GPU timing
# ============================================================================

function bench(f, n_warmup=5, n_iter=50)
    # Warmup
    for _ in 1:n_warmup
        f()
    end
    CUDA.synchronize()

    # Time
    t = CUDA.@elapsed begin
        for _ in 1:n_iter
            f()
        end
    end
    return t / n_iter * 1000  # ms per iteration
end

function bench_backward(f_loss, x_args, n_warmup=3, n_iter=20)
    for _ in 1:n_warmup
        Zygote.gradient(() -> f_loss(x_args...), Flux.params(x_args...))
    end
    CUDA.synchronize()

    t = CUDA.@elapsed begin
        for _ in 1:n_iter
            Zygote.gradient(() -> f_loss(x_args...), Flux.params(x_args...))
        end
    end
    return t / n_iter * 1000
end

# ============================================================================
# Setup test data
# ============================================================================

L = 128
B = 4
token_dim = 768
pair_dim = 256
dim_cond = 256
n_heads = 12
dim_head = 64

println("\n" * "="^70)
println("Benchmark: L=$L, B=$B, token_dim=$token_dim, pair_dim=$pair_dim")
println("="^70)

x = CUDA.randn(Float32, token_dim, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
mask = CUDA.ones(Float32, L, B)

# ============================================================================
# 1. PyTorchLayerNorm Benchmark
# ============================================================================
println("\n--- PyTorchLayerNorm ---")

ln_affine = PyTorchLayerNorm(token_dim) |> gpu
ln_noaffine = PyTorchLayerNorm(token_dim; affine=false) |> gpu
ln_pair = PyTorchLayerNorm(pair_dim) |> gpu

t_ln_fwd = bench(() -> ln_affine(x))
@printf("  Forward  (affine, %d):     %.3f ms\n", token_dim, t_ln_fwd)

t_ln_na_fwd = bench(() -> ln_noaffine(x))
@printf("  Forward  (no-affine, %d):  %.3f ms\n", token_dim, t_ln_na_fwd)

t_ln_pair_fwd = bench(() -> ln_pair(pair))
@printf("  Forward  (pair, %d):       %.3f ms\n", pair_dim, t_ln_pair_fwd)

# Backward
x_ln = CUDA.randn(Float32, token_dim, L, B)
t_ln_bwd = bench(() -> begin
    Zygote.gradient(x -> sum(ln_affine(x)), x_ln)
    nothing
end)
@printf("  Backward (affine, %d):     %.3f ms\n", token_dim, t_ln_bwd)

# ============================================================================
# 2. Dense (matmul) Benchmark
# ============================================================================
println("\n--- Dense (cuBLAS matmul) ---")

dense_768 = Dense(token_dim => token_dim) |> gpu
dense_qkv = Dense(token_dim => token_dim * 3) |> gpu
dense_expand = Dense(token_dim => token_dim * 8) |> gpu

t_dense = bench(() -> dense_768(x))
@printf("  %d -> %d:    %.3f ms\n", token_dim, token_dim, t_dense)

t_dense_qkv = bench(() -> dense_qkv(x))
@printf("  %d -> %d:   %.3f ms\n", token_dim, token_dim * 3, t_dense_qkv)

t_dense_exp = bench(() -> dense_expand(x))
@printf("  %d -> %d:   %.3f ms\n", token_dim, token_dim * 8, t_dense_exp)

# ============================================================================
# 3. PairBiasAttention Benchmark
# ============================================================================
println("\n--- PairBiasAttention ---")

attn = PairBiasAttention(token_dim, n_heads; pair_dim=pair_dim, qk_ln=true) |> gpu

t_attn_fwd = bench(() -> attn(x, pair, mask), 3, 20)
@printf("  Forward:   %.3f ms\n", t_attn_fwd)

x_attn = CUDA.randn(Float32, token_dim, L, B)
t_attn_bwd = bench(() -> begin
    Zygote.gradient(x -> sum(attn(x, pair, mask)), x_attn)
    nothing
end, 3, 10)
@printf("  Backward:  %.3f ms\n", t_attn_bwd)

# ============================================================================
# 4. SwiGLU Transition Benchmark
# ============================================================================
println("\n--- SwiGLUTransition ---")

transition = SwiGLUTransition(token_dim; expansion_factor=4) |> gpu

t_tr_fwd = bench(() -> transition(x, mask))
@printf("  Forward:   %.3f ms\n", t_tr_fwd)

x_tr = CUDA.randn(Float32, token_dim, L, B)
t_tr_bwd = bench(() -> begin
    Zygote.gradient(x -> sum(transition(x, mask)), x_tr)
    nothing
end, 3, 20)
@printf("  Backward:  %.3f ms\n", t_tr_bwd)

# ============================================================================
# 5. ProteINAAdaLN Benchmark
# ============================================================================
println("\n--- ProteINAAdaLN ---")

adaln = ProteINAAdaLN(token_dim, dim_cond) |> gpu

t_adaln_fwd = bench(() -> adaln(x, cond, mask))
@printf("  Forward:   %.3f ms\n", t_adaln_fwd)

# ============================================================================
# 6. Full TransformerBlock Benchmark
# ============================================================================
println("\n--- TransformerBlock (full) ---")

block = TransformerBlock(
    dim_token=token_dim, dim_pair=pair_dim, n_heads=n_heads,
    dim_cond=dim_cond, qk_ln=true
) |> gpu

t_block_fwd = bench(() -> block(x, pair, cond, mask), 3, 20)
@printf("  Forward:   %.3f ms\n", t_block_fwd)

x_block = CUDA.randn(Float32, token_dim, L, B)
t_block_bwd = bench(() -> begin
    Zygote.gradient(x -> sum(block(x, pair, cond, mask)), x_block)
    nothing
end, 2, 5)
@printf("  Backward:  %.3f ms\n", t_block_bwd)

# ============================================================================
# 7. Full 14-layer stack estimate
# ============================================================================
println("\n--- Estimated full ScoreNetwork (14 blocks) ---")
@printf("  Forward:   %.1f ms (14 × %.3f ms)\n", t_block_fwd * 14, t_block_fwd)
@printf("  Backward:  %.1f ms (14 × %.3f ms)\n", t_block_bwd * 14, t_block_bwd)
@printf("  Total:     %.1f ms\n", (t_block_fwd + t_block_bwd) * 14)

# ============================================================================
# Memory summary
# ============================================================================
println("\n--- GPU Memory ---")
CUDA.memory_status()
@printf("  Available: %.2f GB\n", CUDA.available_memory() / 1e9)

println("\n" * "="^70)
println("Benchmark complete.")
println("="^70)
