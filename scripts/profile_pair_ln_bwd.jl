#!/usr/bin/env julia
# Focused profiling of pair LayerNorm backward.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina, CUDA, Flux, Flux.Zygote, Printf

LaProteina.enable_tf32!()

pair_dim = 256; L = 128; B = 4

function bench_bwd(name, f_loss, x; n_warmup=5, n_iter=20)
    for _ in 1:n_warmup
        Zygote.gradient(f_loss, x)
    end
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for _ in 1:n_iter
            Zygote.gradient(f_loss, x)
        end
    end
    @printf("  %-50s %.3f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("Pair LayerNorm Backward (multiple runs)")
println("="^70)

x = CUDA.randn(Float32, pair_dim, L, L, B)
ln = PyTorchLayerNorm(pair_dim) |> gpu
ln_na = PyTorchLayerNorm(pair_dim; affine=false) |> gpu

for run in 1:5
    println("\nRun $run:")
    bench_bwd("LN backward (affine, $pair_dim, pair)", x -> sum(ln(x)), x)
    bench_bwd("LN backward (non-affine, $pair_dim, pair)", x -> sum(ln_na(x)), x)
end

# Also test: what does _pytorch_layernorm_fwd cost in isolation?
println("\n--- _pytorch_layernorm_fwd isolation ---")
scale = ln.scale
bias = ln.bias
for run in 1:3
    bench_bwd("_pytorch_layernorm_fwd backward (affine)",
        x -> sum(LaProteina._pytorch_layernorm_fwd(x, scale, bias, 1f-5, 1:1)[1]), x)
    bench_bwd("_pytorch_layernorm_fwd backward (non-affine)",
        x -> sum(LaProteina._pytorch_layernorm_fwd(x, nothing, nothing, 1f-5, 1:1)[1]), x)
end

println("\n" * "="^70)
