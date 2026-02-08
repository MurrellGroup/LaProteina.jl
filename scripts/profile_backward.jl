#!/usr/bin/env julia
# Profile backward pass to understand gradient computation costs.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina, CUDA, Flux, Flux.Zygote, Printf

LaProteina.enable_tf32!()

L = 128; B = 4; h = 12; d = 64; pair_dim = 256; token_dim = 768; dim_cond = 256

function bench_bwd(name, f_loss, x; n_warmup=3, n_iter=10)
    for _ in 1:n_warmup
        Zygote.gradient(f_loss, x)
    end
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for _ in 1:n_iter
            Zygote.gradient(f_loss, x)
        end
    end
    @printf("  %-45s %.3f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("Backward Pass Profile (L=$L, B=$B)")
println("="^70)

# Setup
x = CUDA.randn(Float32, token_dim, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

println("\n--- Individual layer backward ---")

# LayerNorm backward
ln = PyTorchLayerNorm(token_dim) |> gpu
bench_bwd("PyTorchLayerNorm (768, token)", x -> sum(ln(x)), x)

ln_pair = PyTorchLayerNorm(pair_dim) |> gpu
bench_bwd("PyTorchLayerNorm (256, pair)", x -> sum(ln_pair(x)), pair)

# Dense backward
dense = Dense(token_dim => token_dim * 3) |> gpu
bench_bwd("Dense 768->2304", x -> sum(dense(x)), x)

dense_pair = Dense(pair_dim => h) |> gpu
bench_bwd("Dense 256->12 (pair)", x -> sum(dense_pair(x)), pair)

# AdaLN backward
adaln = ProteINAAdaLN(token_dim, dim_cond) |> gpu
bench_bwd("ProteINAAdaLN", x -> sum(adaln(x, cond, mask)), x)

# AdaptiveOutputScale backward
aos = AdaptiveOutputScale(token_dim, dim_cond) |> gpu
bench_bwd("AdaptiveOutputScale", x -> sum(aos(x, cond, mask)), x)

# SwiGLUTransition backward
transition = SwiGLUTransition(token_dim; expansion_factor=4) |> gpu
bench_bwd("SwiGLUTransition", x -> sum(transition(x, mask)), x)

# PairBiasAttention backward
attn = PairBiasAttention(token_dim, h; pair_dim=pair_dim, qk_ln=true) |> gpu
bench_bwd("PairBiasAttention (full)", x -> sum(attn(x, pair, mask)), x)

# PairBiasAttention without pair bias
attn_nopair = PairBiasAttention(token_dim, h; pair_dim=nothing, qk_ln=true) |> gpu
bench_bwd("PairBiasAttention (no pair bias)", x -> sum(attn_nopair(x, nothing, mask)), x)

# Full MHA with AdaLN
mha = MultiHeadBiasedAttentionADALN(token_dim, pair_dim, h, dim_cond; qk_ln=true) |> gpu
bench_bwd("MultiHeadBiasedAttentionADALN", x -> sum(mha(x, pair, cond, mask)), x)

# TransitionADALN
tr_adaln = TransitionADALN(token_dim, dim_cond) |> gpu
bench_bwd("TransitionADALN", x -> sum(tr_adaln(x, cond, mask)), x)

# Full TransformerBlock
block = TransformerBlock(dim_token=token_dim, dim_pair=pair_dim, n_heads=h, dim_cond=dim_cond, qk_ln=true) |> gpu
bench_bwd("TransformerBlock", x -> sum(block(x, pair, cond, mask)), x)

println("\n--- Attention internals backward ---")
# Just the attention core
q = CUDA.randn(Float32, d, L, h, B)
k = CUDA.randn(Float32, d, L, h, B)
v = CUDA.randn(Float32, d, L, h, B)
bias4 = CUDA.randn(Float32, h, L, L, B)
scale = Float32(d^-0.5)

bench_bwd("_attention (with bias)", x -> sum(LaProteina._attention(x, k, v, bias4, mask, scale)), q)
bench_bwd("_attention (no bias)", x -> sum(LaProteina._attention(x, k, v, nothing, mask, scale)), q)

println("\n--- batched_mul backward ---")
a3 = CUDA.randn(Float32, L, d, h*B)
b3 = CUDA.randn(Float32, d, L, h*B)
bench_bwd("batched_mul (L×d × d×L)", x -> sum(NNlib.batched_mul(x, b3)), a3)

scores3 = CUDA.randn(Float32, L, L, h*B)
bench_bwd("softmax backward", x -> sum(NNlib.softmax(x; dims=1)), scores3)

println("\n" * "="^70)
println("Backward profile complete.")
println("="^70)
