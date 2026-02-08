#!/usr/bin/env julia
# Profile per-block costs and overhead
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4

model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

x = CUDA.randn(Float32, 768, L, B)
pair = CUDA.randn(Float32, 256, L, L, B)
cond = CUDA.randn(Float32, 256, L, B)
mask = CUDA.ones(Float32, L, B)

pba = model.transformer_layers[1].mha.mha
pair_eps = pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

function bench(name, f, n_warmup=3, n_iter=10)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-50s %.2f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("Per-Block Cost Analysis")
println("="^70)

# N blocks forward+backward (prenorm path)
for n in [1, 2, 3, 5, 7, 14]
    bench("$n blocks prenorm bwd", () -> begin
        blocks = model.transformer_layers[1:n]
        Zygote.gradient(x -> begin
            y = x
            for i in 1:n
                y = LaProteina._transformer_block_prenormed(blocks[i], y, pair, pair_normed, cond, mask)
            end
            sum(y)
        end, x)
    end)
end

# Compute marginal cost
println("\n--- Marginal cost analysis ---")
times = Float64[]
for n in [1, 2, 3, 5, 7, 14]
    f = () -> begin
        blocks = model.transformer_layers[1:n]
        Zygote.gradient(x -> begin
            y = x
            for i in 1:n
                y = LaProteina._transformer_block_prenormed(blocks[i], y, pair, pair_normed, cond, mask)
            end
            sum(y)
        end, x)
    end
    for _ in 1:3; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:10; f(); end
    push!(times, t / 10 * 1000)
end

ns = [1, 2, 3, 5, 7, 14]
@printf("\n  %-8s %10s %10s %10s\n", "Blocks", "Total ms", "Marginal ms", "Per-block ms")
for (i, n) in enumerate(ns)
    if i > 1
        marginal = (times[i] - times[i-1]) / (ns[i] - ns[i-1])
        @printf("  %-8d %10.1f %10.1f %10.1f\n", n, times[i], marginal, times[i] / n)
    else
        @printf("  %-8d %10.1f %10s %10.1f\n", n, times[i], "-", times[i] / n)
    end
end

println("\n" * "="^70)
println("Done.")
println("="^70)
