#!/usr/bin/env julia
# Profile AdaLN in detail
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4; dim = 768; dim_cond = 256

adaln = ProteINAAdaLN(dim, dim_cond) |> gpu
x = CUDA.randn(Float32, dim, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

function bench(name, f, n_warmup=5, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("ProteINAAdaLN Profiling")
println("="^70)

println("\n--- Forward ---")
bench("Full AdaLN forward", () -> adaln(x, cond, mask))
bench("  norm(x) [768, noaffine]", () -> adaln.norm(x))
bench("  norm_cond(cond) [256, affine]", () -> adaln.norm_cond(cond))
nc = adaln.norm_cond(cond)
bench("  to_gamma(normed_cond)", () -> adaln.to_gamma(nc))
bench("  to_beta(normed_cond)", () -> adaln.to_beta(nc))
gamma = adaln.to_gamma(nc)
beta = adaln.to_beta(nc)
normed = adaln.norm(x)
bench("  normed .* gamma .+ beta", () -> normed .* gamma .+ beta)

println("\n--- Backward ---")
bench("Full AdaLN backward (w.r.t. x)", () -> begin
    Zygote.gradient(x -> sum(adaln(x, cond, mask)), x)
end)
bench("Full AdaLN backward (w.r.t. model)", () -> begin
    Zygote.gradient(m -> sum(m(x, cond, mask)), adaln)
end)
bench("  norm backward (768, noaffine)", () -> begin
    Zygote.gradient(x -> sum(adaln.norm(x)), x)
end)
bench("  norm_cond backward (256, affine)", () -> begin
    Zygote.gradient(c -> sum(adaln.norm_cond(c)), cond)
end)

println("\nDone!")
