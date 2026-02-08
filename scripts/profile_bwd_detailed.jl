#!/usr/bin/env julia
# Detailed backward profiling: identify remaining bottlenecks in GPU-optimized path.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()
println("GPU: ", CUDA.name(CUDA.device()))

L = 128; B = 4
token_dim = 768; pair_dim = 256; n_heads = 12; dim_cond = 256

model = ScoreNetwork(
    n_layers=14, token_dim=token_dim, pair_dim=pair_dim,
    n_heads=n_heads, dim_cond=dim_cond, qk_ln=true,
    update_pair_repr=false
) |> gpu

seq_raw_dim = size(model.init_repr_factory.projection.weight, 2)
cond_raw_dim = size(model.cond_factory.projection.weight, 2)
pair_raw_dim = size(model.pair_rep_builder.init_repr_factory.projection.weight, 2)
pair_cond_raw_dim = size(model.pair_rep_builder.cond_factory.projection.weight, 2)

raw_features = ScoreNetworkRawFeatures(
    CUDA.randn(Float32, seq_raw_dim, L, B),
    CUDA.randn(Float32, cond_raw_dim, L, B),
    CUDA.randn(Float32, pair_raw_dim, L, L, B),
    CUDA.randn(Float32, pair_cond_raw_dim, L, L, B),
    CUDA.ones(Float32, L, B)
)

function bench(name, f, n_warmup=2, n_iter=3)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.1f ms\n", name, t / n_iter * 1000)
end

# Profile individual block operations
x = CUDA.randn(Float32, token_dim, L, B)
pair = CUDA.randn(Float32, pair_dim, L, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

block = model.transformer_layers[1] |> gpu

# Get pre-normalized pair
pba = block.mha.mha
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

println("\n" * "="^70)
println("Detailed Backward Profiling")
println("="^70)

println("\n--- Single block backward ---")

bench("Standard block backward", () -> begin
    Zygote.gradient(m -> begin
        out = m(x, pair, cond, mask)
        sum(out)
    end, block)
end)

bench("Prenorm block backward", () -> begin
    Zygote.gradient(m -> begin
        out = LaProteina._transformer_block_prenormed(m, x, pair, pair_normed, cond, mask)
        sum(out)
    end, block)
end)

# Profile sub-operations backward
println("\n--- Sub-operation backward profiling ---")

# PairBiasAttention backward (standard vs prenorm)
bench("PairBiasAttn backward (standard)", () -> begin
    Zygote.gradient(m -> begin
        out = m(x, pair, mask)
        sum(out)
    end, pba)
end)

bench("PairBiasAttn backward (prenorm)", () -> begin
    Zygote.gradient(m -> begin
        out = LaProteina._pair_bias_attn_prenormed(m, x, pair_normed, mask)
        sum(out)
    end, pba)
end)

# pair_norm backward only
bench("pair_norm backward (256-dim, L=128, B=4)", () -> begin
    Zygote.gradient(p -> sum(pba.pair_norm(p)), pair)
end)

# Affine-only backward
bench("_apply_pair_affine backward", () -> begin
    Zygote.gradient(pn -> begin
        out = LaProteina._apply_pair_affine(pn, pba.pair_norm, pba.to_bias)
        sum(out)
    end, pair_normed)
end)

# _attention backward
q = CUDA.randn(Float32, 64, L, n_heads, B)
k = CUDA.randn(Float32, 64, L, n_heads, B)
v = CUDA.randn(Float32, 64, L, n_heads, B)
pair_bias = CUDA.randn(Float32, n_heads, L, L, B)

bench("_attention backward (with pair bias)", () -> begin
    Zygote.gradient(q -> begin
        out = LaProteina._attention(q, k, v, pair_bias, mask, 0.125f0)
        sum(out)
    end, q)
end)

bench("_attention backward (no pair bias)", () -> begin
    Zygote.gradient(q -> begin
        out = LaProteina._attention(q, k, v, nothing, mask, 0.125f0)
        sum(out)
    end, q)
end)

# Transition backward
bench("SwiGLUTransition backward", () -> begin
    Zygote.gradient(x -> sum(block.transition(x, cond, mask)), x)
end)

# AdaLN backward
bench("AdaLN backward (token)", () -> begin
    Zygote.gradient(x -> sum(block.mha.adaln(x, cond, mask)), x)
end)

# Dense layer backward (to_qkv: 768 → 2304)
bench("Dense to_qkv backward (768→2304)", () -> begin
    x_n = pba.node_norm(x)
    Zygote.gradient(m -> sum(m(x_n)), pba.to_qkv)
end)

# Permutedims backward
x_4d = CUDA.randn(Float32, 64, 12, L, B)
bench("permutedims backward (64,12,L,B)", () -> begin
    Zygote.gradient(x -> sum(permutedims(x, (1, 3, 2, 4))), x_4d)
end)

# Softmax backward
scores = CUDA.randn(Float32, L, L, n_heads, B)
bench("softmax backward (L,L,h,B)", () -> begin
    Zygote.gradient(s -> sum(NNlib.softmax(s; dims=2)), scores)
end)

# batched_mul backward
a = CUDA.randn(Float32, 64, L, n_heads*B)
b = CUDA.randn(Float32, 64, L, n_heads*B)
bench("batched_mul backward (64,L,h*B)", () -> begin
    Zygote.gradient(a -> sum(NNlib.batched_mul(NNlib.batched_transpose(a), b)), a)
end)

println("\n--- Full model backward breakdown ---")
bench("Full model backward (standard)", () -> begin
    Zygote.gradient(m -> begin
        out = forward_from_raw_features(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
    end, model)
end)

bench("Full model backward (gpu-optimized)", () -> begin
    Zygote.gradient(m -> begin
        out = forward_from_raw_features_gpu(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
    end, model)
end)

println("\n" * "="^70)
println("Profiling complete.")
println("="^70)
